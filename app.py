import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA


st.set_page_config(
    page_title="GenAI PDF Chatbot",
    page_icon="📄"
)

st.title("📄 GenAI PDF Chatbot")
st.write("Upload any PDF and chat with your document")


uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)


if uploaded_file:

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)


    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    vectorstore = FAISS.from_texts(
        chunks,
        embeddings
    )


    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )


    llm = OllamaLLM(
        model="llama3"
    )


    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )


    question = st.text_input(
        "Ask a question from your PDF"
    )


    if question:

        answer = qa.invoke(
            {"query": question}
        )

        st.subheader("Answer")
        st.write(answer["result"])