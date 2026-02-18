"""Indexing module - knowledge graph building components.

This module provides modular components for building knowledge graphs
from various sources:

Components:
    - Chunkers: Text splitting strategies (TokenChunker)
    - Extractors: Entity/relationship extraction (LLMEntityExtractor)
    - Builders: Schema-based graph construction (DatabaseSchemaBuilder, ExcelSchemaBuilder)

Usage:
    # Direct imports
    from src.indexing import TokenChunker, DatabaseSchemaBuilder

    chunker = TokenChunker(max_token_size=1024)
    chunks = chunker.chunk("Long text here...")

    # Or use registry pattern
    from src.indexing import get_chunker, get_indexer

    chunker = get_chunker("token", max_token_size=512)
    builder = get_indexer("schema", graph_storage=..., entities_vdb=..., relationships_vdb=...)

Adding Custom Components:
    from src.indexing import BaseChunker, register_chunker

    class MyChunker(BaseChunker):
        def chunk(self, content, **kwargs):
            # Implementation
            pass

    register_chunker("my_chunker", MyChunker)
"""

from .base import (
    BaseChunker,
    BaseExtractor,
    BaseIndexer,
    ChunkResult,
    ExtractionResult,
    IndexingResult,
)
from .chunkers import TokenChunker, chunking_by_token_size
from .extractors import (
    extract_entities,
    _handle_entity_relation_summary,
    _handle_single_entity_extraction,
    _handle_single_relationship_extraction,
    _merge_nodes_then_upsert,
    _merge_edges_then_upsert,
)
from .schema_builder import DatabaseSchemaBuilder, detect_schema_type
from .excel_builder import ExcelSchemaBuilder


# =============================================================================
# Registry Pattern
# =============================================================================

CHUNKER_REGISTRY = {
    "token": TokenChunker,
}

INDEXER_REGISTRY = {
    "schema": DatabaseSchemaBuilder,
    "database": DatabaseSchemaBuilder,  # Alias
    "excel": ExcelSchemaBuilder,
}


def get_chunker(name: str, **kwargs) -> BaseChunker:
    """Factory function to get a chunker by name.

    Args:
        name: Chunker name (e.g., "token")
        **kwargs: Arguments to pass to the chunker constructor

    Returns:
        Instantiated chunker

    Raises:
        ValueError: If chunker name is not found

    Example:
        chunker = get_chunker("token", max_token_size=512)
    """
    if name not in CHUNKER_REGISTRY:
        available = ", ".join(CHUNKER_REGISTRY.keys())
        raise ValueError(f"Unknown chunker: '{name}'. Available: {available}")
    return CHUNKER_REGISTRY[name](**kwargs)


def get_indexer(name: str, **kwargs) -> BaseIndexer:
    """Factory function to get an indexer by name.

    Args:
        name: Indexer name (e.g., "schema", "excel")
        **kwargs: Arguments to pass to the indexer constructor

    Returns:
        Instantiated indexer

    Raises:
        ValueError: If indexer name is not found

    Example:
        indexer = get_indexer("schema", graph_storage=gs, entities_vdb=ev, relationships_vdb=rv)
    """
    if name not in INDEXER_REGISTRY:
        available = ", ".join(INDEXER_REGISTRY.keys())
        raise ValueError(f"Unknown indexer: '{name}'. Available: {available}")
    return INDEXER_REGISTRY[name](**kwargs)


def register_chunker(name: str, cls: type):
    """Register a custom chunker class.

    Args:
        name: Name to register the chunker under
        cls: Chunker class (should inherit from BaseChunker)
    """
    CHUNKER_REGISTRY[name] = cls


def register_indexer(name: str, cls: type):
    """Register a custom indexer class.

    Args:
        name: Name to register the indexer under
        cls: Indexer class (should inherit from BaseIndexer)
    """
    INDEXER_REGISTRY[name] = cls


def list_chunkers() -> list:
    """List all available chunker names."""
    return list(CHUNKER_REGISTRY.keys())


def list_indexers() -> list:
    """List all available indexer names."""
    return list(INDEXER_REGISTRY.keys())


__all__ = [
    # Base classes
    "BaseChunker",
    "BaseExtractor",
    "BaseIndexer",
    "ChunkResult",
    "ExtractionResult",
    "IndexingResult",
    # Chunkers
    "TokenChunker",
    "chunking_by_token_size",
    # Extractors
    "extract_entities",
    "_handle_entity_relation_summary",
    "_handle_single_entity_extraction",
    "_handle_single_relationship_extraction",
    "_merge_nodes_then_upsert",
    "_merge_edges_then_upsert",
    # Builders
    "DatabaseSchemaBuilder",
    "ExcelSchemaBuilder",
    "detect_schema_type",
    # Registry functions
    "get_chunker",
    "get_indexer",
    "register_chunker",
    "register_indexer",
    "list_chunkers",
    "list_indexers",
    "CHUNKER_REGISTRY",
    "INDEXER_REGISTRY",
]
