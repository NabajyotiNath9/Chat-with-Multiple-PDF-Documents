import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores.docarray import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.agents import AgentExecutor
import tempfile
from pathlib import Path

# Set up your Google Gemini API key
os.environ["GEMINI_API_KEY"] = "your_api_key_here"

# Initialize the Google Generative AI model
llm = GoogleGenerativeAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    model="gemini-2.0-flash-exp",
    temperature=0.7
)

# Initialize the Google Generative AI chat model
chat_model = ChatGoogleGenerativeAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    model="gemini-2.0-flash-exp",
    temperature=0.7
)

# Initialize the OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Initialize the PDF loader
loader = PyPDFLoader()

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# Initialize the vector store
vector_store = None

# Function to load and split PDF documents
def load_and_split_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name
    documents = loader.load_and_split(file_path=tmp_file_path)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

# Function to create a vector store from text chunks
def create_vector_store(text_chunks):
    return DocArrayInMemorySearch.from_documents(text_chunks, embeddings)

# Function to handle user input and get a response
def handle_user_input(user_question):
    if vector_store is None:
        st.error("Please upload and process a PDF document first.")
        return

    docs = vector_store.similarity_search(user_question)
    chain = load_qa_chain(chat_model, chain_type="stuff")
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response['output_text']

# Streamlit app layout
st.title("Chat with PDF Documents")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    text_chunks = load_and_split_pdf(uploaded_file)
    vector_store = create_vector_store(text_chunks)
    st.success("PDF loaded and processed successfully.")

# User input
user_question = st.text_input("Ask a question about the document:")

if user_question:
    response = handle_user_input(user_question)
    st.write(response)
