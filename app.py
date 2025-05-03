import os
import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Function to load and split PDF
def load_and_split_pdf(uploaded_file):
    # Create a temporary file path
    temp_file_path = "temp_pdf.pdf"
    
    # Write the uploaded file into the temporary file
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.getbuffer())

    # Now load the PDF using the file path with PyMuPDFLoader
    loader = PyMuPDFLoader(temp_file_path)
    documents = loader.load()

    # Split the documents into text chunks
    text_chunks = split_text_into_chunks(documents)
    
    # Clean up the temporary file after processing
    os.remove(temp_file_path)
    
    return text_chunks

# Function to split text into chunks
def split_text_into_chunks(documents, chunk_size=500):
    text_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    
    for doc in documents:
        # Split the document into chunks
        chunks = splitter.split_text(doc.page_content)  # Assuming `page_content` holds the text
        text_chunks.extend(chunks)
    
    return text_chunks

# Main function for Streamlit app
def main():
    st.title("Chat with PDF Documents")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file is not None:
        # Load and split the PDF
        text_chunks = load_and_split_pdf(uploaded_file)
        
        # Display a preview of the extracted text
        st.write("Extracted Text Chunks:")
        st.write(text_chunks[:5])  # Display first 5 chunks as preview

if __name__ == "__main__":
    main()
