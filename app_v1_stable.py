import streamlit as st
import os
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# LLM and Local Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings  # 🆓 Free Local Embeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ==========================================
# 🔑 STEP 1: SETUP & API KEY
# ==========================================
load_dotenv()
MY_API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="Manish's RAG Bot", page_icon="📄", layout="wide")

if not MY_API_KEY:
    st.error("🚫 API Key not found! Please create a .env file with GOOGLE_API_KEY inside it.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = MY_API_KEY

# ==========================================
# 🧠 AVAILABLE GEMINI MODELS & HISTORY
# ==========================================
# Kept your 3-model failover list
candidate_models = [
    "models/gemini-1.5-flash",
    "models/gemini-flash-lite-latest",
    "models/gemini-pro"
]

# State Initialization (Merged)
if "processing" not in st.session_state:
    st.session_state.processing = False

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Upload a PDF and ask me anything about it."}
    ]

# ==========================================
# 📚 DEFAULT DATA & FUNCTIONS
# ==========================================
default_data = """
The Virtual Cluster ECU project is being developed by Manish.
It uses Python and Pytest for validation.
The system simulates vehicle communication protocols like CAN and LIN.
Manish is a Test Lead at Wipro working for Ford Motors.
He has 5+ years of experience in Automotive Testing.
He is skilled in UDS, CAN, and Diagnostics.
"""


def get_file_content(uploaded_file):
    """Your robust file extraction logic."""
    uploaded_file.seek(0)
    file_type = uploaded_file.name.split('.')[-1].lower()

    if file_type == 'pdf':
        text = ""
        try:
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception as e:
            return f"Error reading PDF: {e}"

    elif file_type in ['txt', 'py']:
        try:
            return uploaded_file.read().decode("utf-8")
        except Exception as e:
            return f"Error reading text file: {e}"
    return None


@st.cache_resource
def create_vectorstore(text_content):
    """Colleague's Local Embedding Upgrade - Saves API Quota!"""
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text_content])

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    return Chroma.from_documents(docs, embeddings)


def get_rag_chain(model_name, active_retriever):
    """Colleague's Explicit Scope: passing active_retriever safely."""
    llm = ChatGoogleGenerativeAI(
        model=model_name,
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
            {"context": active_retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
    )


# ==========================================
# ⚙️ SIDEBAR : SETTINGS & LOGIC
# ==========================================
with st.sidebar:
    st.markdown("## 🤖 AI Resume Assistant")
    st.caption("Chat with Manish's AI profile")
    st.markdown("---")

    st.markdown("### 📄Upload Document to Analyse")
    uploaded_file = st.file_uploader(
        "Upload PDF, TXT, or PY",
        type=['pdf', 'txt', 'py'],
        label_visibility="collapsed"
    )

    # Your defensive logic
    raw_text = default_data
    if uploaded_file is not None:
        extracted_text = get_file_content(uploaded_file)
        if extracted_text and not extracted_text.startswith("Error"):
            raw_text = extracted_text
            st.success("✅ File loaded successfully!")
        else:
            st.warning("⚠️ File is empty or unreadable. Using default data.")
            raw_text = default_data
    else:
        st.info("Using default resume")

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.processing = False  # Reset lock
        st.rerun()

    st.markdown("---")
    st.caption("Powered by Gemini + LangChain")
    st.write("Developed by @iammku")

# ==========================================
# 🧠 INITIALIZE VECTOR STORE
# ==========================================
# Your safe try/except block
try:
    vectorstore = create_vectorstore(raw_text)
    retriever = vectorstore.as_retriever()
except Exception as e:
    st.error(f"❌ Error processing document: {e}")
    st.stop()

# ==========================================
# 💬 CHAT INTERFACE
# ==========================================
st.title("💬 Chat with Manish's Personal Assistant")

if uploaded_file:
    st.caption(f"📂 Analyzing: **{uploaded_file.name}**")
else:
    st.caption("📂 Analyzing: **Default Resume Data**")

# Display History with Custom Avatars
for message in st.session_state.messages:
    avatar_icon = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])

# User Chat Input
if prompt := st.chat_input("Ask a question about Manish's experience..."):

    # 🛡️ The Concurrency Lock
    if st.session_state.processing:
        st.stop()
    st.session_state.processing = True

    # 1. Display User Message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Display Assistant Message with Failover Logic
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = ""
        success = False
        used_model = ""

        with st.spinner("🤖 Consulting multiple AI models for the best answer..."):
            for model_name in candidate_models:
                try:
                    rag_chain = get_rag_chain(model_name, retriever)
                    response = rag_chain.invoke(prompt)

                    full_response = response.content
                    message_placeholder.markdown(full_response)
                    used_model = model_name
                    success = True
                    break
                except Exception as e:
                    time.sleep(1)  # Breathe before hitting the backup API
                    continue

        if success:
            st.caption(f"⚡ Answered by: `{used_model}`")
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            message_placeholder.error("❌ All models are currently busy. Please try again in a moment.")
            st.session_state.messages.append({"role": "assistant", "content": "Error: All models failed."})

    # 🔓 Unlock the chat
    st.session_state.processing = False

# ==========================================
# 🕵️ DATABASE INSPECTOR
# ==========================================
st.markdown("---")
# Your safe inspector code
with st.expander("🕵️ Database Inspector (See what the AI reads)"):
    if 'vectorstore' in globals() or 'vectorstore' in locals():
        try:
            data = vectorstore.get()
            num_chunks = len(data['ids'])
            st.write(f"**Total Document Chunks:** {num_chunks}")

            if num_chunks > 0:
                st.write("preview of first chunk:")
                st.code(data['documents'][0])
        except Exception as e:
            st.warning("Inspector unavailable for this configuration.")