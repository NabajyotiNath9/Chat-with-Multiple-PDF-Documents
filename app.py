import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.docarray import DocArrayInMemorySearch
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
import tempfile

# Set your API key for Gemini 2.0
os.environ["GEMINI_API_KEY"] = "your_gemini_api_key_here"  # Replace with your API key

# Initialize Google Gemini 2.0 model (flash)
llm = ChatGoogleGenerativeAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    model="gemini-2.0-flash-exp",
    temperature=0.7
)

# Initialize PDF loader and text splitter
loader = PyPDFLoader()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# Function to load and split PDF file
def load_and_split_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name
    
    documents = loader.load_and_split(file_path=tmp_file_path)
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
