import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    :root {
        --primary-color: #0066cc;
        --secondary-color: #00a8e8;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0066cc 0%, #00a8e8 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
    }
    
    .stat-card {
        background: #f0f4ff;
        border-left: 4px solid #0066cc;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    
    .response-box {
        background: #f9f9f9;
        border-left: 4px solid #00a8e8;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .context-box {
        background: #fffbf0;
        border: 1px solid #ffd699;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    
    .success-badge {
        background: #d4edda;
        color: #155724;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "llm" not in st.session_state:
    st.session_state.llm = ChatMistralAI(model="ministral-3b-2512")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "db_stats" not in st.session_state:
    st.session_state.db_stats = {"chunks": 0, "collection": "dna_book"}

# Constants
DB_PATH = "Chroma-DB"
COLLECTION_NAME = "dna_book"

def load_vector_store():
    """Load existing vector store from disk"""
    try:
        embedding_model = MistralAIEmbeddings(model="mistral-embed")
        vector_store = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME,
        )
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )
        return vector_store, retriever
    except Exception as e:
        return None, None

def process_pdf(pdf_file, progress_placeholder):
    """Process uploaded PDF and create vector store"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name
        
        # Create DB directory
        Path(DB_PATH).mkdir(exist_ok=True)
        
        # Load PDF
        progress_placeholder.info("📖 Loading PDF...")
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        
        # Split documents
        progress_placeholder.info("✂️ Splitting documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100,
        )
        chunks = splitter.split_documents(docs)
        
        # Create embeddings and vector store
        progress_placeholder.info("🧠 Creating embeddings...")
        embedding_model = MistralAIEmbeddings(model="mistral-embed")
        
        progress_placeholder.info("💾 Storing in vector database...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=DB_PATH,
            collection_name=COLLECTION_NAME,
        )
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Update session state
        st.session_state.vector_store = vector_store
        st.session_state.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )
        st.session_state.db_stats = {
            "chunks": len(chunks),
            "collection": COLLECTION_NAME,
            "pages": len(docs),
        }
        
        return True, len(chunks), len(docs)
    except Exception as e:
        return False, str(e), None

def query_rag(query):
    """Process query through RAG system"""
    if st.session_state.retriever is None:
        return None, None, "No vector database loaded. Please upload a document first."
    
    try:
        # Retrieve context
        docs = st.session_state.retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a helpful AI assistant specialized in document analysis.
                Use ONLY the provided context to answer the question.
                If the answer is not present in the context, say: "I could not find the answer in the document."
                Be concise and accurate in your responses.""",
            ),
            (
                "human",
                """context:

                {context}

                question:
                {question}
                """,
            ),
        ])
        
        # Get response
        final_prompt = prompt.invoke({
            "context": context,
            "question": query,
        })
        response = st.session_state.llm.invoke(final_prompt)
        
        return response.content, docs, None
    except Exception as e:
        return None, None, str(e)

# Header
st.markdown("""
    <div class="main-header">
        <h1>📚 RAG Document Assistant</h1>
        <p>Intelligent document retrieval and question answering powered by Mistral AI</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    add_vertical_space(1)
    
    st.subheader("📤 Document Management")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type="pdf",
        help="Upload a PDF file to create or update the vector database"
    )
    
    if uploaded_file is not None:
        if st.button("🚀 Process Document", use_container_width=True):
            progress_placeholder = st.empty()
            success, result1, result2 = process_pdf(uploaded_file, progress_placeholder)
            
            if success:
                progress_placeholder.empty()
                st.success(f"✅ Database created successfully!")
                st.info(f"📊 **Processed:**\n- **Pages:** {result2}\n- **Chunks:** {result1}")
            else:
                progress_placeholder.empty()
                st.error(f"❌ Error processing PDF: {result1}")
    
    add_vertical_space(2)
    
    # Database status
    st.subheader("📊 Database Status")
    
    if Path(DB_PATH).exists():
        vector_store, retriever = load_vector_store()
        if vector_store is not None:
            st.session_state.vector_store = vector_store
            st.session_state.retriever = retriever
            
            try:
                count = vector_store._collection.count()
                st.markdown(f"""
                    <div class="stat-card">
                        <strong>Vector Database Active</strong><br>
                        Chunks: <strong>{count}</strong><br>
                        Collection: <strong>{COLLECTION_NAME}</strong>
                    </div>
                """, unsafe_allow_html=True)
            except:
                st.warning("⚠️ Could not connect to database")
        else:
            st.warning("⚠️ No database found")
    else:
        st.info("ℹ️ No database yet. Upload a PDF to get started!")
    
    add_vertical_space(2)
    
    # Clear database
    if st.button("🗑️ Clear Database", help="Delete all data from vector store", use_container_width=True):
        if Path(DB_PATH).exists():
            import shutil
            shutil.rmtree(DB_PATH)
            st.session_state.vector_store = None
            st.session_state.retriever = None
            st.session_state.chat_history = []
            st.success("✅ Database cleared!")
            st.rerun()
    
    add_vertical_space(3)
    
    # Settings
    st.subheader("⚙️ Model Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model", "ministral-3b")
    with col2:
        st.metric("Context Chunks", "4")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🤖 Chat with Your Documents")
    add_vertical_space(1)
    
    # Query input
    query = st.text_input(
        "Ask a question about your document:",
        placeholder="What is the main topic of this document?",
        help="Enter your question and press Enter or click Search"
    )
    
    col_search, col_clear = st.columns([3, 1])
    
    with col_search:
        search_button = st.button("🔍 Search", use_container_width=True, type="primary")
    
    with col_clear:
        if st.button("🔄 Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    add_vertical_space(2)
    
    # Process query
    if search_button and query:
        if st.session_state.retriever is None:
            st.error("❌ No vector database loaded. Please upload a document first.")
        else:
            with st.spinner("🤔 Thinking..."):
                answer, context_docs, error = query_rag(query)
            
            if error:
                st.error(f"❌ Error: {error}")
            else:
                # Add to history
                st.session_state.chat_history.append({
                    "query": query,
                    "answer": answer,
                    "docs_used": len(context_docs) if context_docs else 0
                })
                
                # Display response
                st.markdown(f"""
                    <div class="response-box">
                        <strong>💬 Answer:</strong><br><br>
                        {answer}
                    </div>
                """, unsafe_allow_html=True)
                
                # Display retrieved context
                with st.expander(f"📖 View Retrieved Context ({len(context_docs)} chunks)", expanded=False):
                    for i, doc in enumerate(context_docs, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.markdown(f"""
                            <div class="context-box">
                                {doc.page_content[:500]}...
                            </div>
                        """, unsafe_allow_html=True)

with col2:
    st.subheader("📜 Chat History")
    add_vertical_space(1)
    
    if st.session_state.chat_history:
        for i, item in enumerate(reversed(st.session_state.chat_history), 1):
            with st.container(border=True):
                st.caption(f"Query {len(st.session_state.chat_history) - i + 1}")
                st.markdown(f"**Q:** {item['query'][:100]}...")
                st.markdown(f"*Chunks used: {item['docs_used']}*")
    else:
        st.info("💭 No queries yet. Ask a question to get started!")

add_vertical_space(3)

# Footer
st.divider()
col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    st.caption("🤖 Powered by Mistral AI")

with col_footer2:
    st.caption("📚 Vector DB: Chroma")

with col_footer3:
    st.caption("⚡ Built with Streamlit")