import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import google.generativeai as genai
from dotenv import load_dotenv

# -------------------- ENV SETUP --------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Add it in Streamlit Secrets.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# -------------------- PDF FUNCTIONS --------------------
def get_pdf_text(pdf_docs, max_pages=10):
    """
    Read text from PDFs (limit pages for speed)
    """
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for i, page in enumerate(reader.pages):
            if i >= max_pages:
                break
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80
    )
    return splitter.split_text(text)


# -------------------- VECTOR STORE --------------------
@st.cache_resource(show_spinner=False)
def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"  # ⚡ FAST
    )
    return FAISS.from_texts(chunks, embedding=embeddings)


# -------------------- CHAT UI --------------------
def showman():
    st.header("📄 Chat with PDF")

    user_question = st.text_input(
        "Ask a question from the PDF",
        key="user_question"
    )

    if user_question:
        # 🔥 Direct Gemini (NO LangChain, NO v1beta)
        model = genai.GenerativeModel("gemini-1.0-pro")

        # 🔎 Retrieve relevant chunks
        docs = st.session_state["docsearch"].similarity_search(
            user_question,
            k=2
        )

        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
You are a helpful assistant.
Answer the question ONLY using the context below.
If the answer is not present, say "Not found in the document".

Context:
{context}

Question:
{user_question}
"""

        with st.spinner("Thinking..."):
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 512
                }
            )

        st.write("### Answer")
        st.write(response.text)


# -------------------- MAIN APP --------------------
def show():
    with st.sidebar:
        st.title("Menu")

        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True
        )

        if pdf_docs and st.button("Submit & Process"):
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs, max_pages=10)
                chunks = get_text_chunks(raw_text)
                st.session_state["docsearch"] = build_vector_store(chunks)

            st.success("PDFs processed successfully!")

    if "docsearch" in st.session_state:
        showman()
