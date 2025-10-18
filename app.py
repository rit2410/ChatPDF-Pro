import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import time
import json
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

# Set page config first
st.set_page_config(
    page_title="ChatPDF Pro",
    page_icon="🤖",
    layout="wide"
)

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

groq_api_key = st.secrets["GROQ_API_KEY"]


# Initialize session state with default values
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "processed_pdfs" not in st.session_state:
    st.session_state.processed_pdfs = []
if "show_sources" not in st.session_state:
    st.session_state.show_sources = False  # Default to untoggled

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .bot-message {
        background-color: #f5f5f5;
        margin-right: 20%;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for PDF upload
with st.sidebar:
    st.markdown("### Document Upload")
    
    uploaded_file = st.file_uploader("Upload PDF file", type="pdf")
    
    if uploaded_file is not None:
        
        if st.button("Process Document", use_container_width=True):
            with st.spinner("Reading document and creating smart embeddings..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                        embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-mpnet-base-v2",
                            model_kwargs={'device': 'cpu'},
                            encode_kwargs={'normalize_embeddings': True}
)
                    # Load PDF
                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()
                    
                    # Check if documents were loaded successfully
                    if len(docs) == 0:
                        st.error("No content could be extracted from the PDF. The file might be scanned, encrypted, or corrupted.")
                    else:
                        # Split documents
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
                        
                        # Create enhanced prompt template
                        prompt_template = ChatPromptTemplate.from_template("""
                        You are ChatPDF Pro, a helpful AI assistant specialized in document analysis. 
                        Answer the user's questions using the provided context from their uploaded PDF combined with your general knowledge.

                        IMPORTANT GUIDELINES:
                        - First, try to answer using the provided context from the PDF
                        - If the context doesn't fully answer the question, supplement with your general knowledge
                        - Provide comprehensive and helpful answers that address the user's query
                        - Maintain a professional and helpful tone
                        - When using information from the PDF, reference specific sections or pages when possible
                        - Don't start every answer with "Based on the provided context" - be natural in your responses
                        - If the PDF context is insufficient, provide the best answer you can using your knowledge

                        Context: {context}

                        Conversation History: {chat_history}

                        Question: {question}

                        Answer: 
                        """)
                        
                        # Create conversation chain with memory
                        memory = ConversationBufferWindowMemory(
                            k=5,
                            memory_key="chat_history",
                            return_messages=True,
                            output_key="answer"
                        )
                        
                        st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                            llm=llm,
                            retriever=vectors.as_retriever(search_kwargs={"k": 3}),
                            memory=memory,
                            combine_docs_chain_kwargs={"prompt": prompt_template},
                            return_source_documents=True
                        )
                        
                        st.session_state.processed_pdfs.append(uploaded_file.name)
                        st.session_state.user_questions_count = 0  # Reset counter for new document
                        
                        st.success(f"Document processed successfully - {len(docs)} pages, {len(final_docs)} chunks")
                    
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                
                finally:
                    # Clean up temporary file
                    if 'tmp_file_path' in locals():
                        os.unlink(tmp_file_path)

# Main chat area
st.header("ChatPDF Pro - Your Document Assistant")

# Display processed PDFs
if st.session_state.processed_pdfs:
    st.info(f"📄 **Currently analyzing:** {st.session_state.processed_pdfs[-1]}")

# Chat container with bottom padding for fixed input
chat_container = st.container()
with chat_container:
    st.markdown("<div>", unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>Assistant:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
            
            # Show sources if enabled
            if st.session_state.show_sources and "sources" in message and message["sources"]:
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.write(f"**Source {i+1}:**")
                        st.write(source.page_content[:500] + "..." if len(source.page_content) > 500 else source.page_content)
                        st.write("---")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Chat input and controls
with st.container():
    # Chat input
    user_input = None
    if st.session_state.conversation is not None:
        user_input = st.chat_input("Ask a question about the PDF...")
        
        if user_input:
            # Add user message to chat history immediately
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Rerun immediately to show the user's question
            st.rerun()

    # Check if we need to generate a response for the latest user message
    if (st.session_state.chat_history and 
        st.session_state.chat_history[-1]["role"] == "user" and 
        st.session_state.conversation is not None and
        (not st.session_state.chat_history or st.session_state.chat_history[-1]["role"] != "assistant")):
        
        # Get the latest user message that needs a response
        latest_user_message = st.session_state.chat_history[-1]["content"]
        
        # Show loading state and generate response
        with st.spinner("Thinking..."):
            try:
                start_time = time.time()
                response = st.session_state.conversation.invoke({"question": latest_user_message})
                response_time = time.time() - start_time
                
                # Add bot response to chat history
                bot_response = {
                    "role": "assistant", 
                    "content": response["answer"],
                    "sources": response.get("source_documents", [])
                }
                st.session_state.chat_history.append(bot_response)
                
                # Rerun to show the bot's response
                st.rerun()
                
            except Exception as e:
                st.error(f"Error getting response: {str(e)}")
    elif not st.session_state.conversation:
        st.info("Please upload and process a PDF file to start chatting!")

    # Controls row - below the chat input (ALWAYS SHOW CONTROLS)
    col1, col2 = st.columns([1, 1])

    with col1:
        # Left side: Show sources toggle
        st.session_state.show_sources = st.toggle(
            "Show sources", 
            value=st.session_state.show_sources,
            key="sources_toggle"
        )

    with col2:
        # Right side: Clear chat history button
        if st.button("🗑️ Clear History", key="clear_chat", use_container_width=True):
            st.session_state.chat_history = []
            if st.session_state.conversation:
                st.session_state.conversation.memory.clear()
            st.rerun()