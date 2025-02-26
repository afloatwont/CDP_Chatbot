import logging
from typing import List, Dict, Any
import os
import argparse

from .loaders import load_all_documentation
from .processors import TextProcessor
from .embeddings import EmbeddingModel
from .storage import VectorStore
from .config import (
    MAX_PAGES_PER_CDP, 
    CHUNK_SIZE, 
    CHUNK_OVERLAP, 
    EMBEDDING_MODEL, 
    VECTOR_STORE_PATH
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_docs(max_pages: int = MAX_PAGES_PER_CDP) -> None:
    """
    Ingest documentation from all CDPs and create a vector store
    
    Args:
        max_pages: Maximum pages to scrape per CDP
    """
    logger.info("Starting documentation ingestion")
    
    # Step 1: Load raw documentation from websites
    logger.info("Loading documentation from CDP websites")
    raw_docs = load_all_documentation(max_pages_per_cdp=max_pages)
    logger.info(f"Loaded {len(raw_docs)} raw documents")
    
    # Step 2: Process and chunk the documents
    logger.info(f"Processing documents with chunk size {CHUNK_SIZE}")
    processor = TextProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    processed_docs = processor.process_documents(raw_docs)
    logger.info(f"Created {len(processed_docs)} document chunks")
    
    # Step 3: Generate embeddings for each chunk
    logger.info(f"Generating embeddings using {EMBEDDING_MODEL}")
    embedding_model = EmbeddingModel(model_name=EMBEDDING_MODEL)
    embedded_docs = embedding_model.embed_documents(processed_docs)
    
    # Step 4: Store the embeddings in a vector store
    logger.info("Creating vector store")
    vector_store = VectorStore(dim=embedded_docs[0]["embedding"].shape[0])
    vector_store.add_documents(embedded_docs)
    
    # Step 5: Save the vector store to disk
    logger.info(f"Saving vector store to {VECTOR_STORE_PATH}")
    vector_store.save(directory=str(VECTOR_STORE_PATH))
    
    logger.info("Ingestion complete!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest CDP documentation")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES_PER_CDP,
                      help=f"Maximum pages to scrape per CDP (default: {MAX_PAGES_PER_CDP})")
    args = parser.parse_args()
    
    ingest_docs(max_pages=args.max_pages)