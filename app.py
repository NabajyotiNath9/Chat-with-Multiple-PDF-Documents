import os
import re
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# --- Configure API key ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# --- Clean text to avoid embedding issues ---
def clean_text(text: str) -> str:
    return re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text).strip()

# --- Extract text from uploaded PDFs ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += clean_text(content)
    return text

# --- Split long text into manageable chunks ---
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# --- Generate vector embeddings and store in memory ---
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp")
    return DocArrayInMemorySearch.from_texts(text_chunks, embedding=embeddings)

# --- Create a QA chain using Gemini 2.0 Flash ---
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
    model = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.3)
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# --- Run when user asks a question ---
def user_input(user_question):
    if "vector_store" not in st.session_state:
        st.error("Please upload and process PDFs before asking questions.")
        return
    vector_store = st.session_state.vector_store
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    try:
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("**Answer:**")
        st.success(response["output_text"])
    except Exception as e:
        st.error(f"An error occurred during response generation: {e}")

# --- Streamlit App Layout ---
def main():
    st.set_page_config(page_title="Chat with PDFs - Gemini 2.0 Flash")
    st.title("📄 Chat with Your PDFs (Powered by Gemini 2.0 Flash)")

    with st.sidebar:
        st.header("Upload PDF Files")
        pdf_docs = st.file_uploader("Upload multiple PDF files", accept_multiple_files=True)
        if st.button("Submit & Process") and pdf_docs:
            with st.spinner("Extracting and embedding text..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.vector_store = vector_store
                    st.success("PDFs processed successfully!")
                except Exception as e:
                    st.error(f"Processing error: {e}")

    user_question = st.text_input("Ask a question based on your PDFs:")
    if user_question:
        user_input(user_question)
    else:
        st.info("Upload and process PDFs, then ask a question.")

# --- Run the app ---
if __name__ == "__main__":
    main()
