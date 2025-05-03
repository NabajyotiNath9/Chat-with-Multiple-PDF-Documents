import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import os

# --- Configure Gemini API key from Streamlit secrets ---
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# --- PDF Text Extraction ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

# --- Split text into chunks ---
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks

# --- Create Chroma vector store (in memory) ---
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="model/embedding-001")
    vector_store = Chroma.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# --- Set up Gemini QA chain ---
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, just say "answer is not available in the context".
    
    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# --- Handle user questions ---
def user_input(user_question):
    try:
        if "vector_store" not in st.session_state:
            st.error("Please upload and process PDFs before asking questions.")
            return
        vector_store = st.session_state.vector_store
        docs = vector_store.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply:", response["output_text"])
    except Exception as e:
        st.error(f"An error occurred while processing your question: {e}")

# --- Main App ---
def main():
    st.set_page_config(page_title="Chat with Multiple PDF")
    st.header("📄 Chat with Multiple PDF Files (Powered by Gemini)")

    with st.sidebar:
        st.title("Upload PDFs")
        pdf_docs = st.file_uploader("Upload your PDF files and click Submit & Process", accept_multiple_files=True)
        if st.button("Submit & Process") and pdf_docs:
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.vector_store = vector_store
                    st.success("PDFs processed! You can now ask questions.")
                except Exception as e:
                    st.error(f"Error processing PDFs: {e}")

    # Input box for user question
    user_question = st.text_input("Ask a question from your uploaded PDFs:")
    if user_question:
        user_input(user_question)
    else:
        st.info("Upload PDFs and ask a question to begin.")

# --- Run App ---
if __name__ == "__main__":
    main()
