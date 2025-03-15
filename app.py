import streamlit as st
import os
import openai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return text

# Function to create FAISS vector store with chunking
def create_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

# Function to initialize chatbot
def initialize_chatbot(vector_store):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 relevant chunks
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return qa_chain

# Streamlit UI
st.title("PDF-Based Chatbot with Summarization")
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    vector_store = create_vector_store(text)
    chatbot = initialize_chatbot(vector_store)
    st.success("PDF uploaded and processed successfully!")

    # User input for chat
    user_input = st.text_input("Ask a question about your document:")
    if st.button("Get Answer") and user_input:
        response = chatbot.run(user_input)
        st.write("**Answer:**", response)
    
    # Summarization option
    if st.button("Summarize Document"):
        summary = chatbot.run("Summarize this document in a few sentences.")
        st.write("**Summary:**", summary)

    # Clear chat history option
    if st.button("Clear Chat History"):
        st.experimental_rerun()
