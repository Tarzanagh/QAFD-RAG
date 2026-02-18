"""
OpenAI Embeddings model for QAFD-RAG
"""

from typing import List, Optional
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)


class OpenAIEmbeddingModel:
    """OpenAI embedding model - standalone version"""
    
    def __init__(self, global_config, embedding_model_name: Optional[str] = None, api_key: Optional[str] = None):
        self.global_config = global_config
        self.embedding_model_name = embedding_model_name or global_config.embedding_model_name
        
        # ✅ FIX: Try explicit api_key parameter first, then environment variable
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it with: export OPENAI_API_KEY='your-key-here'"
            )
        
        # Determine dimensions based on model
        if "text-embedding-3-large" in self.embedding_model_name:
            self.embedding_dim = 3072  # Default for 3-large
        elif "text-embedding-3-small" in self.embedding_model_name:
            self.embedding_dim = 1536
        elif "text-embedding-ada-002" in self.embedding_model_name:
            self.embedding_dim = 1536
        else:
            self.embedding_dim = 1536  # Default
        
        # Allow dimension override for 3-large (can reduce to 256-3072)
        if hasattr(global_config, 'embedding_dimensions'):
            self.embedding_dim = global_config.embedding_dimensions
        
        self.batch_size = getattr(global_config, 'embedding_batch_size', 100)
        self.max_seq_len = getattr(global_config, 'embedding_max_seq_len', 8191)
        
        logger.info(f"Initializing OpenAI: {self.embedding_model_name}")
        logger.info(f"✅ OpenAI model ready: {self.embedding_dim}-dim")
    
    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode texts to embeddings using OpenAI API"""
        from openai import OpenAI
        
        if isinstance(texts, str):
            texts = [texts]
        
        client = OpenAI(api_key=self.api_key)
        
        # Build request params
        request_params = {
            "model": self.embedding_model_name,
            "input": texts,
            "encoding_format": "float",
        }
        
        # Add dimensions param only for text-embedding-3 models
        if "text-embedding-3" in self.embedding_model_name:
            request_params["dimensions"] = self.embedding_dim
        
        # Encode with OpenAI
        response = client.embeddings.create(**request_params)
        
        # Extract embeddings
        embeddings = np.array([item.embedding for item in response.data])
        
        return embeddings