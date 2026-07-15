import os
import streamlit as st

from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="GenAI PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 GenAI PDF Chatbot")
st.write("Upload any PDF and chat with your document")


# -----------------------------
# Load Groq API Key
# -----------------------------
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("GROQ_API_KEY not found in Streamlit Secrets")
    st.stop()


# -----------------------------
# Upload PDF
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)


if uploaded_file:

    # Read PDF
    pdf_reader = PdfReader(uploaded_file)

    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()


    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)


    # -----------------------------
    # Create Embeddings
    # -----------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    # Create Vector Database
    vectorstore = FAISS.from_texts(
        chunks,
        embeddings
    )


    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )


    # -----------------------------
    # Groq LLM
    # -----------------------------
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0
    )


    # Retrieval QA Chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )


    # -----------------------------
    # Ask Question
    # -----------------------------
    question = st.text_input(
        "Ask a question from your PDF"
    )


    if question:

        with st.spinner("Thinking..."):

            answer = qa.invoke(
                {
                    "query": question
                }
            )

        st.subheader("Answer")
        st.write(answer["result"])
