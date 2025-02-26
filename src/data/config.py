import os
from pathlib import Path

# Project structure
BASE_DIR = Path(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Create directories if they don't exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Vector store configuration
VECTOR_STORE_PATH = PROCESSED_DIR / "vector_store"

# Embedding model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Small and efficient model

# LLM configuration
LLM_MODEL = "EleutherAI/pythia-160m"  # Smaller model with good performance

# Chunking configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Scraping configuration
MAX_PAGES_PER_CDP = 50