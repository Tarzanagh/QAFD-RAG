"""Abstract base class for graph retrievers.

This module defines the interface that all retriever algorithms must implement.
Retrievers are responsible for traversing knowledge graphs to find relevant
nodes and paths based on various algorithms (flow diffusion, PageRank, BFS, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any


@dataclass
class RetrieverResult:
    """Result from a retriever operation.

    Attributes:
        nodes: Dictionary mapping node IDs to their scores/flow values
        path: Optional list of node IDs representing a path
        score: Overall score for the retrieval result
        metadata: Optional additional metadata from the retrieval
    """
    nodes: Dict[str, float] = field(default_factory=dict)
    path: Optional[List[str]] = None
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseRetriever(ABC):
    """Abstract base class for graph traversal/retrieval algorithms.

    All retriever implementations should inherit from this class and
    implement the required abstract methods.

    Attributes:
        graph: The graph object to traverse (typically NetworkX graph)
    """

    def __init__(self, graph, **kwargs):
        """Initialize the retriever with a graph.

        Args:
            graph: The graph to traverse
            **kwargs: Additional algorithm-specific parameters
        """
        self.graph = graph

    @abstractmethod
    def retrieve(
        self,
        source_node: str,
        target_node: Optional[str] = None,
        **kwargs
    ) -> RetrieverResult:
        """Retrieve relevant nodes from the graph.

        Args:
            source_node: Starting node for retrieval
            target_node: Optional target node for path finding
            **kwargs: Algorithm-specific parameters

        Returns:
            RetrieverResult containing nodes, optional path, and score
        """
        pass

    def find_path(
        self,
        source: str,
        target: str
    ) -> Tuple[Optional[List[str]], float]:
        """Find path between source and target nodes.

        Optional — not all retrievers support pathfinding (e.g. flow diffusion
        returns ranked clusters, not paths).

        Args:
            source: Source node ID
            target: Target node ID

        Returns:
            Tuple of (path as list of node IDs or None, score)
        """
        return None, 0.0

    def get_node_scores(self) -> Dict[str, float]:
        """Get the scores/values for all processed nodes.

        Returns:
            Dictionary mapping node IDs to their scores
        """
        return {}
