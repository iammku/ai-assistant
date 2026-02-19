## AI Resume Assistant (RAG Chatbot)

An intelligent Retrieval-Augmented Generation (RAG) assistant built to transform a static resume into an interactive conversational experience. This project demonstrates how to bridge the gap between Large Language Models (LLMs) and private data.

## Overview
Traditional keyword-based resume screening is limited. This tool uses **Google Gemini AI** and **Streamlit** to allow users to "chat" with my professional background, asking specific questions about my experience in SDET, Virtual Cluster ECUs, and Python automation.

## Tech Stack
* **Language:** Python 3.x
* **AI Model:** Google Gemini 1.5 Flash (via Google Generative AI)
* **Framework:** Streamlit (Frontend/UI)
* **Retrieval:** RAG-based context injection
* **Deployment:** Streamlit Community Cloud

## ✨ Key Features
* **Contextual Q&A:** Answers questions based specifically on the uploaded resume data.
* **Real-time Interaction:** Fast, streaming responses using Gemini's optimized API.
* **Professional UI:** Clean, minimalist interface designed for hiring managers.

## Setup & Installation
1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/iammku/ai-assistant.git](https://github.com/iammku/ai-assistant.git)
    cd ai-assistant
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up Environment Variables:**
    Create a `.env` file and add your Google API Key:
    ```text
    GOOGLE_API_KEY=your_actual_key_here
    ```
4.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

## Future Enhancements
* Add Support for PDF/DOCX multi-file uploads.
* Integrate Vector Databases (FAISS or ChromaDB) for persistent memory.
* Implement automated testing scripts for the UI components.

---
*Developed by Manish Kumar*
*krmanish1998@gmail.com*