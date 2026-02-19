import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# ==========================================
# 🔑 STEP 1: SETUP API KEY
# ==========================================
MY_API_KEY = os.getenv("GOOGLE_API_KEY")

if not MY_API_KEY:
    st.error("🚫 API Key not found! Please create a .env file with GOOGLE_API_KEY inside it.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = MY_API_KEY

st.set_page_config(page_title="Manish's RAG Bot", page_icon="📄", layout="wide")

# ==========================================
# 🧠 AVAILABLE GEMINI MODELS
# ==========================================
candidate_models = [
    "models/gemini-1.5-flash",
    "models/gemini-flash-lite-latest",
    "models/gemini-pro"
]

# Initialize History early
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Upload a PDF and ask me anything about it."}
    ]

# ==========================================
# ⚙️ SIDEBAR : SETTINGS
# ==========================================
with st.sidebar:
    st.markdown("## 🤖 AI Resume Assistant")
    st.caption("Chat with Manish's AI profile")
    st.markdown("---")

    # 📄 Upload Document
    st.markdown("### 📄 Document")
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type="pdf",
        label_visibility="collapsed"
    )

    if uploaded_file:
        st.success("Custom document loaded")
    else:
        st.info("Using default resume")

    st.markdown("---")

    # 🗑 Clear Chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Powered by Gemini + LangChain")


# ==========================================
# 📚 KNOWLEDGE BASE LOGIC
# ==========================================
def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Default Data
default_data = """
The Virtual Cluster ECU project is being developed by Manish. 
It uses Python and Pytest for validation. 
The system simulates vehicle communication protocols like CAN and LIN.
Manish is a Test Lead at Wipro working for Ford Motors.
He has 5+ years of experience in Automotive Testing.
He is skilled in UDS, CAN, and Diagnostics.
"""

if uploaded_file is not None:
    raw_text = get_pdf_text(uploaded_file)
else:
    raw_text = default_data


# 🔄 Cache the Vector Store
@st.cache_resource
def create_vectorstore(text_content):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text_content])

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=MY_API_KEY
    )
    return Chroma.from_documents(docs, embeddings)


try:
    vectorstore = create_vectorstore(raw_text)
    retriever = vectorstore.as_retriever()
except Exception as e:
    st.error(f"❌ Error processing document: {e}")
    st.stop()


# ==========================================
# 🧠 MODEL SETUP (FIXED HERE)
# ==========================================
def get_rag_chain(model_name):  # <--- FIXED: Added argument here
    llm = ChatGoogleGenerativeAI(
        model=model_name,  # <--- FIXED: Using the argument
        temperature=0,
        google_api_key=MY_API_KEY
    )

    prompt = ChatPromptTemplate.from_template("""
    You are an AI Assistant. Answer based ONLY on the context provided below.
    If you don't know, just say "I don't know based on this document".

    Context:
    {context}

    Question: {question}
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
    )


# ==========================================
# 💬 CHAT INTERFACE
# ==========================================
st.title("💬 Chat with Manish's Personal Assistant")

if uploaded_file:
    st.caption(f"📂 Analyzing: **{uploaded_file.name}**")
else:
    st.caption("📂 Analyzing: **Default Resume Data**")

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question..."):

    # User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Assistant Message
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        success = False
        used_model = ""

        # 🔄 The Auto-Switch Loop
        for model in candidate_models:
            try:
                # Pass the model name to the function
                rag_chain = get_rag_chain(model)

                response = rag_chain.invoke(prompt)

                full_response = response.content
                message_placeholder.markdown(full_response)
                used_model = model
                success = True
                break
            except Exception as e:
                # UNCOMMENT THIS IF YOU WANT TO SEE WHY IT FAILED
                # st.error(f"Debug Error on {model}: {e}")
                continue

        if success:
            st.caption(f"⚡ Answered by: `{used_model}`")
        else:
            message_placeholder.error("❌ Service busy. Please try again later.")
            full_response = "Error: Service Unavailable"

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# ==========================================
# 🕵️ DATABASE INSPECTOR (Outside the loop)
# ==========================================
st.markdown("---")
with st.expander("🕵️ Database Inspector (See what the AI reads)"):
    # Check if vectorstore exists
    if 'vectorstore' in globals() or 'vectorstore' in locals():
        data = vectorstore.get()
        num_chunks = len(data['ids'])
        st.write(f"**Total Document Chunks:** {num_chunks}")

        if num_chunks > 0:
            st.write("preview of first chunk:")
            st.code(data['documents'][0])