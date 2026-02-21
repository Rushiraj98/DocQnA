import streamlit as st
import plotly.graph_objects as go
from utils.document_processor import DocumentProcessor
from utils.rag import RAGSystem
from utils.graph_rag import GraphRAGSystem
from utils.gcp_services import GCPServices
from utils.embeddings import EmbeddingManager
import config

st.set_page_config(page_title="DocQnA", page_icon="📚", layout="wide")

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'graph_rag_system' not in st.session_state:
    st.session_state.graph_rag_system = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'gcp_services' not in st.session_state:
    st.session_state.gcp_services = GCPServices()

st.title("📚 DocQnA - Advanced Document Q&A System")
st.markdown("GraphRAG + RAG | Multiple Embeddings | GCP Integration")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    rag_mode = st.selectbox(
        "RAG Mode",
        ["Standard RAG", "GraphRAG", "Hybrid"]
    )
    
    embedding_model = st.selectbox(
        "Embedding Model",
        ["openai", "sentence-transformers", "vertex"]
    )
    
    vector_store = st.selectbox(
        "Vector Store",
        ["ChromaDB", "FAISS"]
    )
    
    use_gcp = st.checkbox("Enable GCP Services", value=True)
    
    st.divider()
    st.header("📊 Stats")
    if st.session_state.chunks:
        st.metric("Document Chunks", len(st.session_state.chunks))

# Main content
tab1, tab2, tab3 = st.tabs(["📄 Upload & Process", "💬 Ask Questions", "🔬 Embeddings Comparison"])

with tab1:
    st.header("Upload Document")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    
    if uploaded_file:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    # Process document
                    processor = DocumentProcessor()
                    text = processor.extract_text_from_pdf(uploaded_file)
                    st.session_state.chunks = processor.chunk_text(text)
                    
                    # Upload to GCS if enabled
                    if use_gcp:
                        try:
                            uploaded_file.seek(0)
                            gcs_path = st.session_state.gcp_services.upload_to_gcs(
                                uploaded_file.read(),
                                uploaded_file.name
                            )
                            st.success(f"✅ Uploaded to GCS: {gcs_path}")
                        except Exception as e:
                            st.warning(f"GCS upload failed: {str(e)}")
                    
                    # Initialize RAG systems
                    if rag_mode in ["Standard RAG", "Hybrid"]:
                        st.session_state.rag_system = RAGSystem(embedding_model)
                        st.session_state.rag_system.index_documents(
                            st.session_state.chunks,
                            use_faiss=(vector_store == "FAISS")
                        )
                    
                    if rag_mode in ["GraphRAG", "Hybrid"]:
                        st.session_state.graph_rag_system = GraphRAGSystem(embedding_model)
                        st.session_state.graph_rag_system.build_knowledge_graph(
                            st.session_state.chunks
                        )
                    
                    st.success(f"✅ Processed {len(st.session_state.chunks)} chunks")
        
        if st.session_state.chunks:
            st.subheader("Document Preview")
            st.text_area("First chunk", st.session_state.chunks[0], height=200)

with tab2:
    st.header("Ask Questions")
    
    if not st.session_state.chunks:
        st.warning("⚠️ Please upload and process a document first")
    else:
        query = st.text_input("Enter your question:")
        
        if query:
            with st.spinner("Generating answer..."):
                answers = {}
                
                if rag_mode == "Standard RAG" and st.session_state.rag_system:
                    answers["Standard RAG"] = st.session_state.rag_system.answer_question(query)
                
                elif rag_mode == "GraphRAG" and st.session_state.graph_rag_system:
                    answers["GraphRAG"] = st.session_state.graph_rag_system.answer_question(query)
                
                elif rag_mode == "Hybrid":
                    if st.session_state.rag_system:
                        answers["Standard RAG"] = st.session_state.rag_system.answer_question(query)
                    if st.session_state.graph_rag_system:
                        answers["GraphRAG"] = st.session_state.graph_rag_system.answer_question(query)
                
                # Display answers
                for method, answer in answers.items():
                    st.subheader(f"📝 {method}")
                    st.write(answer)
                    st.divider()
                
                # Log to GCP
                if use_gcp and answers:
                    try:
                        st.session_state.gcp_services.log_query(
                            query,
                            str(answers),
                            rag_mode
                        )
                    except Exception as e:
                        st.warning(f"Logging failed: {str(e)}")

with tab3:
    st.header("Embeddings Comparison")
    
    if st.session_state.chunks:
        test_text = st.text_area("Test text", st.session_state.chunks[0][:500])
        
        if st.button("Compare Embeddings"):
            with st.spinner("Generating embeddings..."):
                embedding_manager = EmbeddingManager()
                results = {}
                
                for model_name in ["openai", "sentence-transformers", "vertex"]:
                    try:
                        embeddings = embedding_manager.get_embeddings([test_text], model_name)
                        results[model_name] = {
                            "dimension": len(embeddings[0]),
                            "sample": embeddings[0][:10]
                        }
                    except Exception as e:
                        results[model_name] = {"error": str(e)}
                
                # Display results
                for model, data in results.items():
                    st.subheader(f"🔹 {model}")
                    if "error" in data:
                        st.error(f"Error: {data['error']}")
                    else:
                        st.write(f"Dimension: {data['dimension']}")
                        st.write(f"Sample (first 10): {data['sample']}")
    else:
        st.warning("⚠️ Please upload a document first")

# Footer
st.divider()
st.markdown("Built with Streamlit | GraphRAG + RAG | GCP Services")
