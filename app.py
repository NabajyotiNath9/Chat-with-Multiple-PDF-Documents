import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.docarray import DocArrayInMemorySearch
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
import io

# Fetch Gemini API key from Streamlit secrets
gemini_api_key = st.secrets["google"]["gemini_api_key"]  # Ensure this matches your secrets.toml entry

# Initialize Google Gemini 2.0 model (flash)
llm = ChatGoogleGenerativeAI(
    api_key=gemini_api_key,
    model="gemini-2.0-flash-exp",
    temperature=0.7
)

# Initialize PDF loader and text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# Function to load and split PDF file
def load_and_split_pdf(file):
    # Load the PDF as a BytesIO object directly from the uploaded file
    file_stream = io.BytesIO(file.getvalue())
    loader = PyPDFLoader(file_stream)
    
    # Load and split the document
    documents = loader.load_and_split()
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

# Function to create vector store
def create_vector_store(text_chunks, embeddings):
    return DocArrayInMemorySearch.from_documents(text_chunks, embeddings)

# Function to handle user input and fetch response
def handle_user_input(user_question, vector_store):
    if vector_store is None:
        st.error("Please upload and process a PDF document first.")
        return
    
    docs = vector_store.similarity_search(user_question)
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response['output_text']

# Main function for Streamlit app logic
def main():
    st.title("Chat with PDF Documents")

    # Initialize variables
    vector_store = None

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        # Process and split the PDF
        text_chunks = load_and_split_pdf(uploaded_file)
        
        # Create vector store with embeddings
        vector_store = create_vector_store(text_chunks, llm)
        st.success("PDF loaded and processed successfully.")
    
    # User input for querying the document
    user_question = st.text_input("Ask a question about the document:")

    if user_question:
        response = handle_user_input(user_question, vector_store)
        st.write(response)

# Run the main function when the script is executed
if __name__ == "__main__":
    main()
