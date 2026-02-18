"""
NVIDIA NV-Embed-v2 embedding model for QAFD-RAG
"""

from copy import deepcopy
from typing import List, Optional
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel
import logging

logger = logging.getLogger(__name__)


class NVEmbedV2EmbeddingModel:
    """NVIDIA NV-Embed-v2 embedding model - standalone version"""
    
    def __init__(self, global_config, embedding_model_name: Optional[str] = None):
        self.global_config = global_config
        self.embedding_model_name = embedding_model_name or global_config.embedding_model_name
        
        # Initialize embedding model
        logger.info(f"Initializing NVIDIA NV-Embed-v2: {self.embedding_model_name}")
        
        self.embedding_model = AutoModel.from_pretrained(
            self.embedding_model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=getattr(global_config, 'embedding_model_dtype', "auto")
        )
        
        self.embedding_dim = self.embedding_model.config.hidden_size
        self.batch_size = getattr(global_config, 'embedding_batch_size', 16)
        self.max_seq_len = getattr(global_config, 'embedding_max_seq_len', 32768)
        self.normalize = getattr(global_config, 'embedding_return_as_normalized', True)
        
        logger.info(f"✅ NVIDIA model loaded: {self.embedding_dim}-dim")
    
    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode texts to embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        batch_size = kwargs.get('batch_size', self.batch_size)
        max_length = kwargs.get('max_length', self.max_seq_len)
        instruction = kwargs.get('instruction', '')
        
        # Add instruction if provided
        if instruction:
            instruction_text = f"Instruct: {instruction}\nQuery: "
        else:
            instruction_text = ""
        
        # Batch processing
        if len(texts) <= batch_size:
            prompts = [instruction_text + t for t in texts] if instruction_text else texts
            results = self.embedding_model.encode(
                prompts=prompts,
                max_length=max_length
            )
        else:
            results = []
            pbar = tqdm(total=len(texts), desc="Batch Encoding")
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                prompts = [instruction_text + t for t in batch_texts] if instruction_text else batch_texts
                batch_results = self.embedding_model.encode(
                    prompts=prompts,
                    max_length=max_length
                )
                results.append(batch_results)
                pbar.update(len(batch_texts))
            pbar.close()
            results = torch.cat(results, dim=0)
        
        # Convert to numpy
        if isinstance(results, torch.Tensor):
            results = results.cpu().numpy()
        
        # Normalize if requested
        if self.normalize:
            results = (results.T / np.linalg.norm(results, axis=1)).T
        
        return results
