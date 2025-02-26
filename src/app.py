import streamlit as st
from data.storage import VectorStore
from data.embeddings import EmbeddingModel
from data.llm import LocalLLM
from data.chains import CDPQueryChain
from data.config import EMBEDDING_MODEL, LLM_MODEL, VECTOR_STORE_PATH
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="CDP How-To Assistant",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load models and vector store (cached to avoid reloading)"""
    try:
        # Load vector store
        vector_store = VectorStore.load(str(VECTOR_STORE_PATH))
        if not vector_store:
            st.error("Failed to load vector store. Please run data ingestion first.")
            st.stop()
            
        # Initialize embedding model
        embedding_model = EmbeddingModel(model_name=EMBEDDING_MODEL)
        
        # Initialize LLM
        llm = LocalLLM(model_name=LLM_MODEL)
        
        # Create query chain
        query_chain = CDPQueryChain(
            vector_store=vector_store,
            embedding_model=embedding_model,
            llm=llm
        )
        
        return query_chain
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        st.stop()

# Header
st.title("CDP How-To Assistant 🤖")
st.markdown("""
Ask me any question about how to use Customer Data Platforms:
- Segment
- mParticle
- Lytics 
- Zeotap

I'll try to answer based on their official documentation.
""")

# Sidebar
st.sidebar.title("About")
st.sidebar.info(
    "This app uses AI to answer questions about CDP platforms "
    "based on their official documentation."
)
st.sidebar.markdown("### Supported CDPs")
st.sidebar.markdown("""
- [Segment](https://segment.com/docs/)
- [mParticle](https://docs.mparticle.com/)
- [Lytics](https://docs.lytics.com/)
- [Zeotap](https://docs.zeotap.com/)
""")

# Input
query = st.text_input("Your question:", placeholder="How do I track events in Segment?")

# Process query
if query:
    with st.spinner("Searching for an answer..."):
        # Load models (cached)
        query_chain = load_models()
        
        # Process the query
        response, relevant_docs = query_chain.process_query(query)
        
        # Display response
        st.markdown("### Answer")
        st.markdown(response)
        
        # Display sources
        if relevant_docs:
            st.markdown("### Sources")
            for i, doc in enumerate(relevant_docs):
                with st.expander(f"Source {i+1}: {doc['platform']} - {doc['source']}"):
                    st.markdown(doc['content'])