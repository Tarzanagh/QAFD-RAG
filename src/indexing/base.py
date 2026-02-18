"""Abstract base classes for indexing components.

This module defines the interfaces for:
- Chunkers: Text splitting strategies
- Extractors: Entity/relationship extraction
- Indexers: Knowledge graph builders
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union

from ..base import BaseGraphStorage, BaseVectorStorage, TextChunkSchema


@dataclass
class ChunkResult:
    """Result from a chunking operation.

    Attributes:
        chunks: List of chunk dictionaries with content and metadata
        total_tokens: Total number of tokens in the original content
    """
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0


@dataclass
class ExtractionResult:
    """Result from entity/relationship extraction.

    Attributes:
        entities: List of extracted entity dictionaries
        relationships: List of extracted relationship dictionaries
        stats: Statistics about the extraction process
    """
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)


@dataclass
class IndexingResult:
    """Result from an indexing/building operation.

    Attributes:
        nodes_added: Number of nodes added to the graph
        edges_added: Number of edges added to the graph
        entities_indexed: Number of entities indexed in vector DB
        relationships_indexed: Number of relationships indexed in vector DB
        metadata: Additional metadata about the operation
    """
    nodes_added: int = 0
    edges_added: int = 0
    entities_indexed: int = 0
    relationships_indexed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseChunker(ABC):
    """Abstract base class for text chunking strategies.

    Chunkers are responsible for splitting text content into smaller,
    overlapping or non-overlapping chunks suitable for processing.
    """

    @abstractmethod
    def chunk(self, content: str, **kwargs) -> ChunkResult:
        """Split content into chunks.

        Args:
            content: Text content to split
            **kwargs: Chunker-specific parameters

        Returns:
            ChunkResult containing the chunks and metadata
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class BaseExtractor(ABC):
    """Abstract base class for entity/relationship extraction.

    Extractors are responsible for identifying and extracting entities
    and relationships from text chunks, typically using LLMs or
    rule-based approaches.
    """

    @abstractmethod
    async def extract(
        self,
        chunks: Dict[str, TextChunkSchema],
        knowledge_graph: BaseGraphStorage,
        **kwargs
    ) -> ExtractionResult:
        """Extract entities and relationships from chunks.

        Args:
            chunks: Dictionary of chunk_id -> chunk data
            knowledge_graph: Graph storage to add entities/relationships to
            **kwargs: Extractor-specific parameters

        Returns:
            ExtractionResult with entities, relationships, and stats
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class BaseIndexer(ABC):
    """Abstract base class for knowledge graph builders.

    Indexers are responsible for building knowledge graphs from
    various sources (schemas, files, etc.) and storing them in
    the appropriate storage backends.
    """

    def __init__(
        self,
        graph_storage: BaseGraphStorage,
        entities_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
    ):
        """Initialize the indexer with storage backends.

        Args:
            graph_storage: Graph storage for nodes and edges
            entities_vdb: Vector database for entity embeddings
            relationships_vdb: Vector database for relationship embeddings
        """
        self.graph_storage = graph_storage
        self.entities_vdb = entities_vdb
        self.relationships_vdb = relationships_vdb

    @abstractmethod
    async def build(self, source: Any, **kwargs) -> IndexingResult:
        """Build knowledge graph from source.

        Args:
            source: Source data (file path, schema dict, etc.)
            **kwargs: Builder-specific parameters

        Returns:
            IndexingResult with statistics about the build
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
