from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    """Generate embeddings for text chunks using SentenceTransformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model = SentenceTransformer(model_name)
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Numpy array of embeddings
        """
        return self.model.encode(texts)
    
    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Embed a list of document chunks
        
        Args:
            documents: List of document dictionaries containing at least 'content' key
            
        Returns:
            List of documents with 'embedding' key added
        """
        # Extract text content from documents
        texts = [doc["content"] for doc in documents]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Add embeddings to documents
        embedded_docs = []
        for i, doc in enumerate(documents):
            doc_with_embedding = doc.copy()
            doc_with_embedding["embedding"] = embeddings[i]
            embedded_docs.append(doc_with_embedding)
            
        return embedded_docs