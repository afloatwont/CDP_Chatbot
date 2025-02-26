from typing import Dict, Any, List, Tuple
import numpy as np
from .storage import VectorStore
from .embeddings import EmbeddingModel
from .llm import LocalLLM

class CDPQueryChain:
    """Chain that processes CDP queries using retrieved context"""
    
    def __init__(
        self, 
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        llm: LocalLLM,
        top_k: int = 5
    ):
        """
        Initialize the query chain
        
        Args:
            vector_store: Vector store containing indexed documents
            embedding_model: Model for embedding queries
            llm: Language model for generating responses
            top_k: Number of relevant documents to retrieve
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.llm = llm
        self.top_k = top_k
        
    def _retrieve_relevant_docs(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query"""
        # Generate embedding for the query
        query_embedding = self.embedding_model.generate_embeddings([query])[0]
        
        # Search for similar documents
        docs = self.vector_store.search(query_embedding, k=self.top_k)
        
        return docs
        
    def process_query(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process a user query
        
        Args:
            query: User question about CDPs
            
        Returns:
            Tuple of (response text, relevant documents)
        """
        # Extract CDP name from query if present
        cdp_names = ["segment", "mparticle", "lytics", "zeotap"]
        specified_cdp = None
        
        for cdp in cdp_names:
            if cdp.lower() in query.lower():
                specified_cdp = cdp
                break
                
        # Retrieve relevant documents
        relevant_docs = self._retrieve_relevant_docs(query)
        
        # Filter by specified CDP if any
        if specified_cdp:
            filtered_docs = [
                doc for doc in relevant_docs 
                if doc["platform"].lower() == specified_cdp.lower()
            ]
            # If we found relevant docs for the specified CDP, use those
            if filtered_docs:
                relevant_docs = filtered_docs
        
        # Generate response using the LLM with context
        response = self.llm.generate_response(query, context=relevant_docs)
        
        return response, relevant_docs