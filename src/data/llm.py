from typing import Dict, Any, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalLLM:
    """Interface for local large language models"""
    
    def __init__(
        self, 
        model_name: str = "EleutherAI/pythia-160m", 
        max_length: int = 2048,
        temperature: float = 0.7
    ):
        """
        Initialize the local LLM
        
        Args:
            model_name: HuggingFace model to use
            max_length: Maximum token length for generation
            temperature: Sampling temperature (higher = more creative)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        
        logger.info(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Use 8-bit quantization if available to reduce memory usage
        if torch.cuda.is_available():
            self.device = "cuda"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True
            )
        else:
            self.device = "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
        logger.info(f"Model loaded on {self.device}")
        
    def generate_response(
        self, 
        prompt: str, 
        context: Optional[List[Dict[str, Any]]] = None,
        max_new_tokens: int = 512
    ) -> str:
        """
        Generate a response based on a prompt and optional context
        
        Args:
            prompt: User query
            context: List of relevant documents
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Generated response
        """
        # Build full prompt with context
        if context:
            context_str = "\n\n".join([
                f"Document from {doc['platform']} ({doc['source']}):\n{doc['content']}"
                for doc in context
            ])
            
            full_prompt = f"""You are an AI assistant specialized in Customer Data Platforms (CDPs).
Based on the following documentation, answer the question as helpfully as possible.
If you don't know the answer, say "I don't have enough information about that."

CONTEXT:
{context_str}

QUESTION:
{prompt}

ANSWER:
"""
        else:
            full_prompt = f"""You are an AI assistant specialized in Customer Data Platforms (CDPs).
Answer the following question as helpfully as possible.
If you don't know the answer, say "I don't have enough information about that."

QUESTION:
{prompt}

ANSWER:
"""
        
        # Tokenize and generate
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # Decode and return only the newly generated text
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_response = full_response[len(full_prompt):]
        
        return generated_response