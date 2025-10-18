import streamlit as st
import os
import tempfile
import time

# Set page config first
st.set_page_config(
    page_title="ChatPDF Pro",
    page_icon="🤖",
    layout="wide"
)

# Import compatible packages
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "processed_pdfs" not in st.session_state:
    st.session_state.processed_pdfs = []
if "show_sources" not in st.session_state:
    st.session_state.show_sources = False

# Use Streamlit secrets for API key
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except:
    st.error("GROQ_API_KEY not found in secrets.")
    st.stop()

st.title("ChatPDF Pro - Your Document Assistant")
st.info("Upload a PDF to start chatting with your document!")

# Simple file upload and processing
uploaded_file = st.file_uploader("Upload PDF file", type="pdf")

if uploaded_file and st.button("Process Document"):
    with st.spinner("Processing PDF..."):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Initialize embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Load PDF
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            
            if len(docs) == 0:
                st.error("No content could be extracted from the PDF.")
            else:
                # Simple text splitting (we'll add proper splitting later)
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200
                )
                final_docs = text_splitter.split_documents(docs)
                
                # Create vector store
                vectors = FAISS.from_documents(final_docs, embeddings)
                st.session_state.vectors = vectors
                
                # Initialize LLM
                llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant")
                
                st.session_state.processed_pdfs.append(uploaded_file.name)
                st.success(f"Document processed successfully - {len(docs)} pages, {len(final_docs)} chunks")
                
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
        finally:
            # Clean up temporary file
            if 'tmp_file_path' in locals():
                os.unlink(tmp_file_path)

# Chat interface
if st.session_state.processed_pdfs:
    st.info(f"📄 **Currently analyzing:** {st.session_state.processed_pdfs[-1]}")
    
    user_input = st.chat_input("Ask a question about the PDF...")
    if user_input:
        st.write(f"User: {user_input}")
        st.info("Full chat functionality will be enabled in the next update!")

# Controls
col1, col2 = st.columns([1, 1])
with col1:
    st.toggle("Show sources", value=st.session_state.show_sources, key="sources_toggle")
with col2:
    if st.button("🗑️ Clear History"):
        st.session_state.chat_history = []
        st.rerun()
