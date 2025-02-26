from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any   
from .data.storage import VectorStore
from .data.embeddings import EmbeddingModel
from .data.llm import LocalLLM
from .data.chains import CDPQueryChain
from .data.config import EMBEDDING_MODEL, LLM_MODEL, VECTOR_STORE_PATH
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Data models for request and response
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

# Load models (as a global variable for simplicity)
try:
    vector_store = VectorStore.load(str(VECTOR_STORE_PATH))
    if not vector_store:
        raise ValueError("Failed to load vector store. Please run data ingestion first.")
    embedding_model = EmbeddingModel(model_name=EMBEDDING_MODEL)
    llm = LocalLLM(model_name=LLM_MODEL)
    query_chain = CDPQueryChain(
        vector_store=vector_store,
        embedding_model=embedding_model,
        llm=llm
    )
except Exception as e: 
    logger.error(f"Failed to load models: {str(e)}")
    raise  # Re-raise the exception to prevent the app from starting

# API endpoint for processing queries
@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        query = request.query
        response, relevant_docs = query_chain.process_query(query)
        return QueryResponse(answer=response, sources=relevant_docs)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))