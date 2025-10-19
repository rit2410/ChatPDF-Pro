import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import time
import json
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

groq_api_key = st.secrets["GROQ_API_KEY"]

# Streamlit page config
st.set_page_config(page_title="InsightIQ – Your Intelligent Document Analyst", page_icon="🤖", layout="wide")

# --- Unified Modern UI Styling for InsightIQ (White Buttons Version) ---
st.markdown("""
<style>
    /* ---- Layout ---- */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1150px;
        margin: auto;
    }

    /* ---- Page Title ---- */
    h1, .stHeader {
        text-align: center;
        font-weight: 800;
        color: #1f77b4;
        margin-bottom: 1.5rem;
    }

    /* ---- Global Font ---- */
    html, body, div, label, input, textarea, button {
        font-family: "Source Sans Pro", sans-serif;
        font-size: 1.05rem;
        color: #222;
    }

    /* ---- Upload Section ---- */
    .upload-section {
        border: 2px dashed #1f77b4;
        border-radius: 12px;
        padding: 2.5rem;
        text-align: center;
        background: #f8f9fa;
        margin: 2rem 0;
        transition: 0.3s;
    }

    .upload-section:hover {
        background: #e3f2fd;
        border-color: #1565c0;
    }

    /* ---- Unified Buttons (Browse, Clear, Process, etc.) ---- */
    .stButton > button,
    .stFileUploader label div[data-testid="stFileUploaderDropzone"] {
        background-color: white !important;
        color: #111 !important;
        border: 2px solid #d3d3d3 !important;
        border-radius: 10px !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.5rem !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        transition: all 0.2s ease-in-out;
    }

    .stButton > button:hover,
    .stFileUploader label div[data-testid="stFileUploaderDropzone"]:hover {
        background-color: #f5f5f5 !important;
        border-color: #bbb !important;
        transform: scale(1.03);
        cursor: pointer;
    }

    /* ---- Chat Input ---- */
    div[data-testid="stChatInput"] textarea {
        font-size: 1.05rem;
        line-height: 1.6;
    }
            /* ---- Tabs Color & Spacing Fix ---- */
.stTabs [data-baseweb="tab-list"] {
    margin-top: 1rem;
    gap: 4rem;
}

/* Tabs: typography */
.stTabs [data-baseweb="tab"] p {
  font-size: 1.1rem !important;      /* bigger labels */
  font-weight: 800 !important;
  margin: 0;
}

/* Tabs: default & hover color */
.stTabs [data-baseweb="tab"] p { color: #222 !important; }
.stTabs [data-baseweb="tab"]:hover p { color: #1f77b4 !important; }

/* Tabs: active text color */
.stTabs button[data-baseweb="tab"][aria-selected="true"] p {
  color: #1f77b4 !important;
}

/* Tabs: underline (the moving highlight bar) */
.stTabs [data-baseweb="tab-highlight"] {
  background-color: #1f77b4 !important;  /* was red */
  height: 4px !important;                /* thicker */
  border-radius: 2px;
}

/* Optional: add a little breathing room above tabs */
.stTabs [data-baseweb="tab-list"] {
  margin-top: 1rem;
  gap: 4rem;
}


</style>
""", unsafe_allow_html=True)


# --- Mode Selection Sidebar ---
st.sidebar.title("⚙️ App Mode")
mode = st.sidebar.radio(
    "Select Application Mode:",
    ["Research Mode", "Analyst Mode"],
    captions=["For research papers", "For business / data reports"]
)



# Persist mode in session state
st.session_state.mode = mode

# Main Title
st.header(f"InsightIQ – {mode}")

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


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

from functools import lru_cache

@lru_cache(maxsize=50)
def cached_detect_chat_intent(query: str) -> str:
    """Lightweight detector for basic conversational understanding."""
    query_lower = query.lower().strip()
    casual_triggers = ["hi", "hello", "hey", "thank", "thanks", "bye", "goodbye", "see you", "ok", "okay"]
    if any(word in query_lower for word in casual_triggers):
        return "casual"
    return "document"




if mode == "Research Mode":
    tab1, tab2 = st.tabs(["📄 Upload & Analyze", "💬 Chat"])

    with tab1:
        st.info("Upload research papers or academic documents to analyze, summarize, and explore key insights.")
        
        uploaded_file = st.file_uploader("Upload PDF file", type="pdf")

        if uploaded_file is not None:
            if st.button("Process Document", use_container_width=True):
                with st.spinner("Reading document and creating smart embeddings..."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name

                        embeddings = load_embeddings()

                        # Load PDF
                        loader = PyPDFLoader(tmp_file_path)
                        docs = loader.load()

                        if len(docs) == 0:
                            st.error("No content could be extracted from the PDF. The file might be scanned, encrypted, or corrupted.")
                        else:
                            # Split documents
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1000, 
                                chunk_overlap=200
                            )
                            final_docs = text_splitter.split_documents(docs)
                            st.session_state.final_docs = final_docs 
                            vectors = FAISS.from_documents(final_docs, embeddings)
                            st.session_state.vectors = vectors
                            llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant")
                            prompt_template = ChatPromptTemplate.from_template("""
                            You are InsightIQ, an intelligent document analysis assistant. You specialize in understanding research papers, reports, and analytical documents.
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
                            
                            st.success(f"✅ Document processed successfully – {len(docs)} pages split into {len(final_docs)} sections for analysis.")

                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
                    
                    finally:
                        # Clean up temporary file
                        if 'tmp_file_path' in locals():
                            os.unlink(tmp_file_path)

        # --- PHASE 2: Research Tools ---
        if "final_docs" in st.session_state and st.session_state.vectors:
            st.markdown("### 🔍 Research Tools")
            # Create three equal-width columns
            col1, col2, col3 = st.columns(3)
            with col1:
                # Summarization Tool
                with st.expander("📄 Generate Summary"):
                    if st.button("Generate Summary"):
                        with st.spinner("Generating paper summary..."):
                            context_docs = st.session_state.final_docs[:3]
                            llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant")

                            summary_prompt = f"""
                            Provide a structured summary of this research paper.
                            Include:
                            1. **Short Summary** – 1 paragraph
                            2. **Medium Summary** – 3–5 bullet points
                            3. **Detailed Summary** – section-wise overview
                            Context:
                            {context_docs}
                            """
                            summary = llm.invoke(summary_prompt).content

                            st.markdown("#### Summary")
                            st.write(summary)

                            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                                tmp_file.write(summary.encode("utf-8"))
                                tmp_path = tmp_file.name

                            st.download_button("📥 Download Summary", data=open(tmp_path, "rb"), file_name="summary.txt")
            with col2:
                # Key-Idea Extraction Tool
                with st.expander("💡 Extract Key Ideas"):
                    if st.button("Extract Key Ideas"):
                        with st.spinner("Extracting key research ideas..."):
                            context_docs = st.session_state.final_docs[:3]
                            llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant")

                            ideas_prompt = f"""
                            From this research paper, extract the following clearly:
                            - **Problem Statement**
                            - **Methodology / Approach**
                            - **Key Results**
                            - **Limitations and Future Work**
                            Context:
                            {context_docs}
                            """
                            ideas = llm.invoke(ideas_prompt).content

                            st.markdown("#### Key Ideas")
                            st.write(ideas)

                            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                                tmp_file.write(ideas.encode("utf-8"))
                                tmp_path = tmp_file.name

                            st.download_button("📥 Download Key Ideas", data=open(tmp_path, "rb"), file_name="key_ideas.txt")
            
            with col3:
                # Paper Comparison Tool
                with st.expander("📚 Compare Another Research Paper"):
                    compare_file = st.file_uploader("Upload a second paper for comparison", type="pdf", key="compare_upload")

                    if compare_file is not None:
                        if st.button("Compare Papers"):
                            with st.spinner("Analyzing and comparing both papers..."):
                                try:
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_compare:
                                        tmp_compare.write(compare_file.getvalue())
                                        tmp_compare_path = tmp_compare.name

                                    loader2 = PyPDFLoader(tmp_compare_path)
                                    docs2 = loader2.load()
                                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                                    final_docs2 = text_splitter.split_documents(docs2)

                                    embeddings = load_embeddings()
                                    vectors2 = FAISS.from_documents(final_docs2, embeddings)

                                    llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant")
                                    compare_prompt = f"""
                                    Compare these two research papers based on:
                                    1. **Research Objective / Problem**
                                    2. **Methodology**
                                    3. **Results / Findings**
                                    4. **Limitations / Future Work**
                                    5. **Novelty and Overlap**

                                    --- PAPER 1 CONTEXT ---
                                    {st.session_state.final_docs[:3]}

                                    --- PAPER 2 CONTEXT ---
                                    {final_docs2[:3]}
                                    """

                                    comparison = llm.invoke(compare_prompt).content

                                    st.markdown("#### 📊 Paper Comparison")
                                    st.write(comparison)

                                    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                                        tmp_file.write(comparison.encode("utf-8"))
                                        tmp_path = tmp_file.name

                                    st.download_button("📥 Download Comparison", data=open(tmp_path, "rb"), file_name="paper_comparison.txt")
                                except Exception as e:
                                    st.error(f"Error comparing papers: {e}")
                                finally:
                                    if 'tmp_compare_path' in locals():
                                        os.unlink(tmp_compare_path)

    with tab2:
        if st.session_state.processed_pdfs:
            st.info(f"📄 **Currently analyzing:** {st.session_state.processed_pdfs[-1]}")

        # Chat container
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

                    # Optional: show sources for assistant messages
                    if (message["role"] == "assistant" 
                        and st.session_state.show_sources 
                        and "sources" in message 
                        and message["sources"]):
                        with st.expander("View Sources"):
                            for i, source in enumerate(message["sources"]):
                                st.write(f"**Source {i+1}:**")
                                st.write(source.page_content[:500] + "..." if len(source.page_content) > 500 else source.page_content)
                                st.write("---")
        with st.container():
            user_input = None
            if st.session_state.conversation is not None:
                user_input = st.chat_input("Ask a question or request insights...")

                if user_input:
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    st.rerun()

            if (st.session_state.chat_history and 
                st.session_state.chat_history[-1]["role"] == "user" and 
                st.session_state.conversation is not None and
                (not st.session_state.chat_history or st.session_state.chat_history[-1]["role"] != "assistant")):

                latest_user_message = st.session_state.chat_history[-1]["content"]
                query_lower = latest_user_message.lower()
                doc_uploaded = bool(st.session_state.vectors)

                with st.spinner("Thinking..."):
                    try:
                        start_time = time.time()
                        intent = cached_detect_chat_intent(latest_user_message)

                        # Smart context reset for document-related questions
                        if doc_uploaded and any(word in query_lower for word in ["document", "pdf", "paper", "report", "file", "upload", "content", "about", "topic", "summary"]):
                            intent = "document"
                        elif intent == "casual" and doc_uploaded and any(word in query_lower for word in ["what", "about", "contain", "include", "analyze", "summarize", "discuss", "explain"]):
                            intent = "document"

                        # Handle responses
                        if intent == "casual":
                            if any(word in query_lower for word in ["hi", "hello", "hey"]):
                                if doc_uploaded:
                                    reply_text = "Hey there! 👋 I see your document is ready — what would you like to explore first?"
                                else:
                                    reply_text = "Hey there! 👋 You can upload a PDF to get started!"
                            elif any(word in query_lower for word in ["thank", "thanks"]):
                                reply_text = "You're very welcome! 😊"
                            elif any(word in query_lower for word in ["bye", "goodbye", "see you"]):
                                reply_text = "Goodbye! 👋 Hope your analysis goes great!"
                            else:
                                if doc_uploaded:
                                    reply_text = "All set! Would you like me to summarize, analyze, or find insights from your document?"
                                else:
                                    reply_text = "Hi there! Upload a document so I can help analyze or summarize it."
                            response = {"answer": reply_text, "source_documents": []}

                        else:
                            # Use the same conversation chain in both modes
                            if doc_uploaded:
                                response = st.session_state.conversation.invoke({"question": latest_user_message})
                            else:
                                general_llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant")
                                reply_text = general_llm.invoke(latest_user_message).content
                                response = {"answer": reply_text, "source_documents": []}

                        # Store and show bot response
                        response_time = time.time() - start_time
                        bot_response = {
                            "role": "assistant",
                            "content": response["answer"],
                            "sources": response.get("source_documents", [])
                        }
                        st.session_state.chat_history.append(bot_response)
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error getting response: {str(e)}")

            elif not st.session_state.conversation:
                st.info("Please upload and process a PDF or report to start chatting!")

            # Controls row
            col1, col2 = st.columns([1, 1])
            with col1:
                st.session_state.show_sources = st.toggle(
                    "Show sources", 
                    value=st.session_state.show_sources,
                    key="sources_toggle"
                )
            with col2:
                if st.button("🗑️ Clear History", key="clear_chat", use_container_width=True):
                    st.session_state.chat_history = []
                    if st.session_state.conversation:
                        st.session_state.conversation.memory.clear()
                    st.rerun()
                
if mode == "Analyst Mode":
    tab1, tab2 = st.tabs(["📄 Upload & Analyze", "💬 Chat"])
    with tab1:
        st.info("Upload financial, business, or analytical reports to extract key insights, trends, and executive summaries.")

        uploaded_file = st.file_uploader("Upload PDF file", type="pdf")

        if uploaded_file is not None:
            if st.button("Process Report", use_container_width=True):
                with st.spinner("Processing report and creating smart embeddings..."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name

                        embeddings = load_embeddings()
                        loader = PyPDFLoader(tmp_file_path)
                        docs = loader.load()

                        if len(docs) == 0:
                            st.error("No content could be extracted from the report. The file might be scanned or empty.")
                        else:
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                            final_docs = text_splitter.split_documents(docs)
                            st.session_state.final_docs = final_docs

                            vectors = FAISS.from_documents(final_docs, embeddings)
                            st.session_state.vectors = vectors

                            llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant")

                            prompt_template = ChatPromptTemplate.from_template("""
                            You are InsightIQ, a professional business analyst AI. 
                            You specialize in reading business, financial, and analytical reports.

                            Using the provided report context, generate clear, actionable insights.
                            Your answers should sound professional and executive-ready.

                            CONTEXT HANDLING RULES:
                            - Prioritize the uploaded report context.
                            - If something is missing, reason logically using your general knowledge.
                            - Always respond with structured, easy-to-read insights.
                            - Use bullets or subheadings for clarity when possible.
                            - Avoid generic filler phrases.

                            Context: {context}

                            Conversation History: {chat_history}

                            Question: {question}

                            Answer:
                            """)

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
                            st.success(f"✅ Report processed successfully – {len(docs)} pages split into {len(final_docs)} sections for analysis.")

                    except Exception as e:
                        st.error(f"Error processing report: {str(e)}")

                    finally:
                        if 'tmp_file_path' in locals():
                            os.unlink(tmp_file_path)

        # Display processed reports
        if st.session_state.processed_pdfs:
            st.info(f"📊 **Currently analyzing:** {st.session_state.processed_pdfs[-1]}")

        # --- Business Analysis Tools ---
        if "final_docs" in st.session_state and st.session_state.vectors:
            st.markdown("### 📈 Business Analysis Tools")

            # Executive Summary
            if st.button("Generate Executive Summary"):
                with st.spinner("Generating executive summary..."):
                    summary_prompt = f"""
                    Create a clear, professional **Executive Summary** from this business report.
                    Include:
                    1. **Overall Summary** – concise overview
                    2. **Key Performance Highlights**
                    3. **Important Risks or Concerns**
                    4. **Recommendations (if applicable)**

                    Context:
                    {st.session_state.final_docs[:3]}
                    """
                    llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant")
                    summary = llm.invoke(summary_prompt).content

                    st.markdown("#### 🧾 Executive Summary")
                    st.write(summary)

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                        tmp_file.write(summary.encode("utf-8"))
                        tmp_path = tmp_file.name
                    st.download_button("📥 Download Summary", data=open(tmp_path, "rb"), file_name="executive_summary.txt")

            # Key Insights
            if st.button("Extract Key Insights"):
                with st.spinner("Extracting key insights..."):
                    insights_prompt = f"""
                    From this report, extract **Key Insights** under these headings:
                    - **Business Performance Highlights**
                    - **Operational Trends**
                    - **Challenges / Risks**
                    - **Strategic Recommendations**

                    Context:
                    {st.session_state.final_docs[:3]}
                    """
                    llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant")
                    insights = llm.invoke(insights_prompt).content

                    st.markdown("#### 💡 Key Insights")
                    st.write(insights)

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                        tmp_file.write(insights.encode("utf-8"))
                        tmp_path = tmp_file.name
                    st.download_button("📥 Download Insights", data=open(tmp_path, "rb"), file_name="key_insights.txt")

            # Trend Analysis
            if st.button("Analyze Trends"):
                with st.spinner("Analyzing trends..."):
                    trend_prompt = f"""
                    Analyze this business report and summarize:
                    - **Emerging Trends**
                    - **Positive and Negative Patterns**
                    - **Future Outlook**

                    Context:
                    {st.session_state.final_docs[:3]}
                    """
                    llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant")
                    trends = llm.invoke(trend_prompt).content

                    st.markdown("#### 📊 Trend Analysis")
                    st.write(trends)

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                        tmp_file.write(trends.encode("utf-8"))
                        tmp_path = tmp_file.name
                    st.download_button("📥 Download Trend Report", data=open(tmp_path, "rb"), file_name="trend_analysis.txt")

    with tab2:
        if st.session_state.processed_pdfs:
            st.info(f"📄 **Currently analyzing:** {st.session_state.processed_pdfs[-1]}")

               # Chat container
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

                    # Optional: show sources for assistant messages
                    if (message["role"] == "assistant" 
                        and st.session_state.show_sources 
                        and "sources" in message 
                        and message["sources"]):
                        with st.expander("View Sources"):
                            for i, source in enumerate(message["sources"]):
                                st.write(f"**Source {i+1}:**")
                                st.write(source.page_content[:500] + "..." if len(source.page_content) > 500 else source.page_content)
                                st.write("---")
        with st.container():
            user_input = None
            if st.session_state.conversation is not None:
                user_input = st.chat_input("Ask a question or request insights...")

                if user_input:
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    st.rerun()

            if (st.session_state.chat_history and 
                st.session_state.chat_history[-1]["role"] == "user" and 
                st.session_state.conversation is not None and
                (not st.session_state.chat_history or st.session_state.chat_history[-1]["role"] != "assistant")):

                latest_user_message = st.session_state.chat_history[-1]["content"]
                query_lower = latest_user_message.lower()
                doc_uploaded = bool(st.session_state.vectors)

                with st.spinner("Thinking..."):
                    try:
                        start_time = time.time()
                        intent = cached_detect_chat_intent(latest_user_message)

                        # Smart context reset for document-related questions
                        if doc_uploaded and any(word in query_lower for word in ["document", "pdf", "paper", "report", "file", "upload", "content", "about", "topic", "summary"]):
                            intent = "document"
                        elif intent == "casual" and doc_uploaded and any(word in query_lower for word in ["what", "about", "contain", "include", "analyze", "summarize", "discuss", "explain"]):
                            intent = "document"

                        # Handle responses
                        if intent == "casual":
                            if any(word in query_lower for word in ["hi", "hello", "hey"]):
                                if doc_uploaded:
                                    reply_text = "Hey there! 👋 I see your document is ready — what would you like to explore first?"
                                else:
                                    reply_text = "Hey there! 👋 You can upload a PDF to get started!"
                            elif any(word in query_lower for word in ["thank", "thanks"]):
                                reply_text = "You're very welcome! 😊"
                            elif any(word in query_lower for word in ["bye", "goodbye", "see you"]):
                                reply_text = "Goodbye! 👋 Hope your analysis goes great!"
                            else:
                                if doc_uploaded:
                                    reply_text = "All set! Would you like me to summarize, analyze, or find insights from your document?"
                                else:
                                    reply_text = "Hi there! Upload a document so I can help analyze or summarize it."
                            response = {"answer": reply_text, "source_documents": []}

                        else:
                            # Use the same conversation chain in both modes
                            if doc_uploaded:
                                response = st.session_state.conversation.invoke({"question": latest_user_message})
                            else:
                                general_llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant")
                                reply_text = general_llm.invoke(latest_user_message).content
                                response = {"answer": reply_text, "source_documents": []}

                        # Store and show bot response
                        response_time = time.time() - start_time
                        bot_response = {
                            "role": "assistant",
                            "content": response["answer"],
                            "sources": response.get("source_documents", [])
                        }
                        st.session_state.chat_history.append(bot_response)
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error getting response: {str(e)}")

            elif not st.session_state.conversation:
                st.info("Please upload and process a PDF or report to start chatting!")

            # Controls row
            col1, col2 = st.columns([1, 1])
            with col1:
                st.session_state.show_sources = st.toggle(
                    "Show sources", 
                    value=st.session_state.show_sources,
                    key="sources_toggle"
                )
            with col2:
                if st.button("🗑️ Clear History", key="clear_chat", use_container_width=True):
                    st.session_state.chat_history = []
                    if st.session_state.conversation:
                        st.session_state.conversation.memory.clear()
                    st.rerun()

        







