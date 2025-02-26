from typing import List, Dict, Any, Optional
import faiss
import numpy as np
import pickle 
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """Store and retrieve document embeddings using FAISS"""
    
    def __init__(self, dim: int = 384):  # Default dimension for all-MiniLM-L6-v2
        """
        Initialize the vector store
        
        Args:
            dim: Dimension of the embeddings
        """
        self.index = faiss.IndexFlatL2(dim)  # L2 distance
        self.documents = []
        
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents with embeddings to the store
        
        Args:
            documents: List of documents with 'embedding' key
        """
        if not documents:
            return
            
        embeddings = np.array([doc["embedding"] for doc in documents], dtype=np.float32)
        
        # Store original documents (without embeddings to save space)
        for doc in documents:
            doc_copy = doc.copy()
            if "embedding" in doc_copy:
                del doc_copy["embedding"]  # Don't store the embedding twice
            self.documents.append(doc_copy)
            
        # Add embeddings to the index
        self.index.add(embeddings)
        logger.info(f"Added {len(documents)} documents to vector store")
        
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of most similar documents with distance scores
        """
        if self.index.ntotal == 0:
            return []
            
        # Ensure the embedding is in the right shape
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Search the index
        distances, indices = self.index.search(query_embedding.astype(np.float32), k=min(k, self.index.ntotal))
        
        # Get the corresponding documents
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # FAISS might return -1 for not enough results
                doc = self.documents[idx].copy()
                doc["distance"] = float(distances[0][i])
                results.append(doc)
                
        return results
    
    def save(self, directory: str, name: str = "vector_store") -> None:
        """
        Save the vector store to disk
        
        Args:
            directory: Directory to save to
            name: Base filename
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save the index
        index_path = os.path.join(directory, f"{name}_index.faiss")
        faiss.write_index(self.index, index_path)
        
        # Save the documents
        docs_path = os.path.join(directory, f"{name}_docs.pkl")
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)
            
        logger.info(f"Saved vector store to {directory}")
    
    @classmethod
    def load(cls, directory: str, name: str = "vector_store") -> Optional["VectorStore"]:
        """
        Load a vector store from disk
        
        Args:
            directory: Directory to load from
            name: Base filename
            
        Returns:
            Loaded VectorStore or None if loading fails
        """
        try:
            # Load the index
            index_path = os.path.join(directory, f"{name}_index.faiss")
            index = faiss.read_index(index_path)
            
            # Load the documents
            docs_path = os.path.join(directory, f"{name}_docs.pkl")
            with open(docs_path, "rb") as f:
                documents = pickle.load(f)
                
            # Create and populate the vector store
            vector_store = cls(dim=index.d)
            vector_store.index = index
            vector_store.documents = documents
            
            logger.info(f"Loaded vector store from {directory} with {len(documents)} documents")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return None