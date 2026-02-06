import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain.chains import RetrievalQA

from langchain_google_genai import GoogleGenerativeAIEmbeddings
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
        chunk_size=500,
chunk_overlap=100

    )
    return splitter.split_text(text)


def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

        vectorstore = FAISS.from_texts(
            text_chunks,
            embedding=embeddings
        )
        return vectorstore
    except Exception as e:
        st.error(f"Embedding failed: {e}")
        st.stop()



def showman():
    st.header("📄 Chat with PDF")

    user_question = st.text_input(
        "Ask a question from the PDF",
        key="user_question"
    )

    if user_question:
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-1.5-pro-latest",
            temperature=0.7
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state["docsearch"].as_retriever()
        )

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
                st.session_state["docsearch"] = get_vector_store(chunks)
            st.success("PDFs processed successfully!")

    if "docsearch" in st.session_state:
        showman()



