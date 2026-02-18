"""Standalone embedding models for QAFD-RAG"""

from .JinaV3 import JinaV3EmbeddingModel
from .NVEmbedV2 import NVEmbedV2EmbeddingModel
from .OpenAI import OpenAIEmbeddingModel

try:
    from .GritLM import GritLMEmbeddingModel
except ImportError:
    GritLMEmbeddingModel = None

__all__ = ['JinaV3EmbeddingModel', 'NVEmbedV2EmbeddingModel', 'OpenAIEmbeddingModel', 'GritLMEmbeddingModel']
