import streamlit as st

st.set_page_config(
    page_title="Gemini_Student",
    page_icon="✏️"
)

st.sidebar.title("Welcome to Gemini_Student")

choice = st.sidebar.radio(
    "Select Module",
    (
        "Chatbot",
        "Image_QA_Gemini",
        "QA_Gemini",
        "MCQ_Gen",
        "chat_with_pdf",
    ),
)

if choice == "Chatbot":
    import Chatbot
    Chatbot.show()

elif choice == "Image_QA_Gemini":
    import Image_QA_Gemini
    Image_QA_Gemini.show()

elif choice == "QA_Gemini":
    import QA_Gemini
    QA_Gemini.show()

elif choice == "MCQ_Gen":
    import MCQ_Gen
    MCQ_Gen.show()

elif choice == "chat_with_pdf":
    import chat_with_pdf
    chat_with_pdf.show()
