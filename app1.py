# app.py
import os
import streamlit as st
import fitz  # PyMuPDF
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# Function to extract text from PDF using PyMuPDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to create FAISS index from extracted text
def create_faiss_index_from_text(text, index_path, embedding_model='sentence-transformers/all-MiniLM-L6-v2'):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vector_store = FAISS.from_texts([text], embeddings)
    vector_store.save_local(index_path)
    return vector_store

# Load FAISS index
def load_faiss_index(index_path, embedding_model):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store

# Create the RAG system
def create_rag_system(vector_store, model_name="llama2"):
    llm = Ollama(model=model_name)

    prompt_template = """
    You are an expert assistant with access to the following context extracted from documents. Your job is to answer the user's question as accurately as possible, using the context below.

    Context:
    {context}

    Given this information, please provide a comprehensive and relevant answer to the following question:
    Question: {question}

    If the context does not contain enough information, clearly state that the information is not available in the context provided.
    If possible, provide a step-by-step explanation and highlight key details.
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain

# Streamlit App
st.set_page_config(page_title="Document RAG Chatbot", page_icon="📄🤖", layout="wide")
st.title("📄 Document RAG Chatbot")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF or Text File", type=["pdf", "txt"])

if uploaded_file is not None:
    with st.spinner('Processing document...'):
        # Create a temporary directory to store FAISS index
        with tempfile.TemporaryDirectory() as temp_dir:
            # Handle PDF
            if uploaded_file.name.endswith(".pdf"):
                text = extract_text_from_pdf(uploaded_file)
            # Handle Text file
            elif uploaded_file.name.endswith(".txt"):
                text = uploaded_file.read().decode("utf-8")
            else:
                st.error("Unsupported file format!")
                st.stop()

            # Create FAISS index
            index_path = os.path.join(temp_dir, "faiss_index")
            vector_store = create_faiss_index_from_text(text, index_path)

            # Create RAG system
            rag_system = create_rag_system(vector_store)

            st.success("Document processed. You can now ask questions!")

            # Chat Interface
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            user_question = st.text_input("Ask a question based on the uploaded document:")

            if st.button("Get Answer"):
                if user_question.strip() == "":
                    st.warning("Please ask a question.")
                else:
                    answer = rag_system.run(user_question)

                    st.session_state.chat_history.append(("You", user_question))
                    st.session_state.chat_history.append(("Bot", answer))

            # Display chat history
            for role, message in st.session_state.chat_history:
                if role == "You":
                    st.markdown(f"**You:** {message}")
                else:
                    st.markdown(f"**Bot:** {message}")

else:
    st.info("Please upload a document to begin.")
