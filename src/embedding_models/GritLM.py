"""
GritLM embedding model for QAFD-RAG
"""

from typing import List, Optional
import torch
import numpy as np
from copy import deepcopy
from gritlm import GritLM
import logging

logger = logging.getLogger(__name__)


class GritLMEmbeddingModel:
    """GritLM embedding model - standalone version"""
    
    def __init__(self, global_config, embedding_model_name: Optional[str] = None):
        self.global_config = global_config
        self.embedding_model_name = embedding_model_name or global_config.embedding_model_name
        
        # Initialize GritLM model
        logger.info(f"Initializing GritLM: {self.embedding_model_name}")
        
        self.embedding_model = GritLM(
            model_name_or_path=self.embedding_model_name,
            torch_dtype=getattr(global_config, 'embedding_model_dtype', "auto"),
            device_map="auto"
        )
        
        # Determine actual dimension by doing a test encode
        logger.info("Testing GritLM to determine actual embedding dimension...")
        test_embedding = self.embedding_model.encode(
            sentences=["test"],
            instruction="<|embed|>\n",
            batch_size=1
        )
        
        if isinstance(test_embedding, torch.Tensor):
            self.embedding_dim = test_embedding.shape[-1]
        else:
            self.embedding_dim = test_embedding.shape[-1]
        
        self.batch_size = getattr(global_config, 'embedding_batch_size', 16)
        self.normalize = getattr(global_config, 'embedding_return_as_normalized', True)
        self.device = self.embedding_model.device
        
        logger.info(f"✅ GritLM model loaded: {self.embedding_dim}-dim (actual measured dimension)")
    
    def _get_formatted_instruction(self, instruction: str) -> str:
        """Format instruction for GritLM"""
        return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"
    
    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode texts to embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        batch_size = kwargs.get('batch_size', self.batch_size)
        instruction = kwargs.get('instruction', '')
        
        # Format instruction if provided
        if instruction:
            formatted_instruction = self._get_formatted_instruction(instruction)
        else:
            formatted_instruction = "<|embed|>\n"
        
        # Encode
        results = self.embedding_model.encode(
            sentences=texts,
            instruction=formatted_instruction,
            batch_size=batch_size
        )
        
        # Convert to numpy
        if isinstance(results, torch.Tensor):
            results = results.cpu().numpy()
        
        # Normalize if requested
        if self.normalize:
            results = (results.T / np.linalg.norm(results, axis=1)).T
        
        return results
