from typing import List, Dict, Any
import re
import nltk
from nltk.tokenize import sent_tokenize
import logging

logger = logging.getLogger(__name__)

# Download necessary NLTK data (fixed version)
def ensure_nltk_resources():
    """Ensure required NLTK resources are downloaded"""
    try:
        # Check if punkt is available
        nltk.data.find('tokenizers/punkt')
        logger.info("NLTK punkt tokenizer is already downloaded")
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer")
        nltk.download('punkt')

# Make sure we download the resources
ensure_nltk_resources()

class TextProcessor:
    """Process raw documentation text into chunks suitable for embedding"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def clean_text(self, text: str) -> str:
        """Clean up the text by removing extra whitespace and special characters"""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that aren't useful
        text = re.sub(r'[^\w\s.,?!:;()\[\]{}-]', '', text)
        
        return text.strip()
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        text = self.clean_text(text)
        
        # Use simple chunking if text is very short
        if len(text) <= self.chunk_size:
            return [text]
            
        try:
            sentences = sent_tokenize(text)
        except Exception as e:
            logger.error(f"Error in sentence tokenization: {e}")
            # Fallback to simple chunking by characters if sentence tokenization fails
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size-self.chunk_overlap)]
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Current chunk is full, save it
                chunks.append(" ".join(current_chunk))
                
                # Keep overlap sentences for the next chunk
                overlap_size = 0
                overlap_chunk = []
                
                while current_chunk and overlap_size < self.chunk_overlap:
                    sentence = current_chunk.pop(-1)
                    overlap_size += len(sentence)
                    overlap_chunk.insert(0, sentence)
                
                current_chunk = overlap_chunk
                current_size = overlap_size
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a list of documents into chunks"""
        processed_docs = []
        
        for doc in documents:
            chunks = self.chunk_text(doc["content"])
            
            for i, chunk in enumerate(chunks):
                processed_docs.append({
                    "content": chunk,
                    "source": doc["source"],
                    "platform": doc["platform"],
                    "chunk_id": i
                })
        
        return processed_docs