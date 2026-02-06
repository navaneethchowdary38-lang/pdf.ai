import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

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
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
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
def build_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(text_chunks, embedding=embeddings)


# -------------------- CHAT UI --------------------
def showman():
    st.header("📄 Chat with PDF")

    user_question = st.text_input(
        "Ask a question from the PDF",
        key="user_question"
    )

    if user_question:
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-pro",
            temperature=0.2,
            max_output_tokens=512,
            timeout=30
        )

        retriever = st.session_state["docsearch"].as_retriever(
            search_kwargs={"k": 2}
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )

        with st.spinner("Thinking..."):
            response = qa.invoke(user_question)

        st.write("### Answer")
        st.write(response["result"])


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
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                st.session_state["docsearch"] = build_vector_store(chunks)

            st.success("PDFs processed successfully!")

    if "docsearch" in st.session_state:
        showman()
