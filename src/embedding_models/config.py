"""
Minimal configuration for embedding models.
"""

class MinimalConfig:
    """Minimal configuration for embedding models"""
    
    def __init__(
        self,
        embedding_model_name="jinaai/jina-embeddings-v3",
        embedding_batch_size=16,
        embedding_model_dtype="auto",
        embedding_return_as_normalized=True,
        embedding_max_seq_len=8192,
        embedding_dimensions=None,  # For OpenAI 3-large
    ):
        self.embedding_model_name = embedding_model_name
        self.embedding_batch_size = embedding_batch_size
        self.embedding_model_dtype = embedding_model_dtype
        self.embedding_return_as_normalized = embedding_return_as_normalized
        self.embedding_max_seq_len = embedding_max_seq_len
        self.embedding_dimensions = embedding_dimensions
