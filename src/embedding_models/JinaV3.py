"""
Jina v3 Embeddings model for QAFD-RAG
"""

from typing import List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class JinaV3EmbeddingModel:
    """Jina Embeddings v3 model - standalone version"""
    
    def __init__(self, global_config, embedding_model_name: Optional[str] = None):
        self.global_config = global_config
        self.embedding_model_name = embedding_model_name or global_config.embedding_model_name
        
        # Initialize Jina v3 model
        logger.info(f"Initializing Jina v3: {self.embedding_model_name}")
        
        from sentence_transformers import SentenceTransformer
        
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            trust_remote_code=True
        )
        
        self.embedding_dim = 1024  # Jina v3 fixed dimension
        self.batch_size = getattr(global_config, 'embedding_batch_size', 32)
        self.max_seq_len = getattr(global_config, 'embedding_max_seq_len', 8192)
        self.normalize = getattr(global_config, 'embedding_return_as_normalized', True)
        
        logger.info(f"✅ Jina v3 model loaded: {self.embedding_dim}-dim")
    
    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode texts to embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        batch_size = kwargs.get('batch_size', self.batch_size)
        
        # Encode with Jina v3
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=self.normalize
        )
        
        return embeddings
