"""Retrievers module - modular graph retrieval algorithms.

This module provides pluggable retrieval algorithms for knowledge graph traversal.
Similar to how neural network frameworks organize optimizers (Adam, SGD, etc.),
this module organizes graph traversal algorithms.

Available Retrievers:
    - FlowDiffusionRetriever: Query-Aware Weighted Flow Diffusion algorithm

Usage:
    # Direct import
    from src.retrievers import FlowDiffusionRetriever

    retriever = FlowDiffusionRetriever(
        graph=my_graph,
        source_node="node_a",
        target_node="node_b",
        confidence=0.5
    )
    result = retriever.retrieve()

    # Or use the registry
    from src.retrievers import get_retriever

    retriever = get_retriever(
        "flow_diffusion",
        graph=my_graph,
        source_node="node_a",
        target_node="node_b"
    )

Adding Custom Retrievers:
    from src.retrievers import BaseRetriever, register_retriever

    class MyCustomRetriever(BaseRetriever):
        def retrieve(self, source_node, target_node=None, **kwargs):
            # Implementation
            pass

        def find_path(self, source, target):
            # Implementation
            pass

    register_retriever("my_custom", MyCustomRetriever)
"""

from .base import BaseRetriever, RetrieverResult
from .flow_diffusion import FlowDiffusionRetriever, QueryAwareWeightedFlowDiffusion

# Registry for retriever classes
RETRIEVER_REGISTRY = {
    "flow_diffusion": FlowDiffusionRetriever,
    "qafd": FlowDiffusionRetriever,  # Alias
}


def get_retriever(name: str, **kwargs) -> BaseRetriever:
    """Factory function to get a retriever by name.

    Args:
        name: Retriever name (e.g., "flow_diffusion", "qafd")
        **kwargs: Arguments to pass to the retriever constructor

    Returns:
        Instantiated retriever

    Raises:
        ValueError: If retriever name is not found

    Example:
        retriever = get_retriever(
            "flow_diffusion",
            graph=my_graph,
            source_node="node_a",
            target_node="node_b",
            confidence=0.5
        )
    """
    if name not in RETRIEVER_REGISTRY:
        available = ", ".join(RETRIEVER_REGISTRY.keys())
        raise ValueError(f"Unknown retriever: '{name}'. Available retrievers: {available}")

    return RETRIEVER_REGISTRY[name](**kwargs)


def register_retriever(name: str, cls: type):
    """Register a custom retriever class.

    Args:
        name: Name to register the retriever under
        cls: Retriever class (should inherit from BaseRetriever)

    Example:
        register_retriever("my_retriever", MyRetrieverClass)
    """
    if not issubclass(cls, BaseRetriever):
        raise TypeError(f"Retriever class must inherit from BaseRetriever, got {cls}")
    RETRIEVER_REGISTRY[name] = cls


def list_retrievers() -> list:
    """List all available retriever names.

    Returns:
        List of registered retriever names
    """
    return list(RETRIEVER_REGISTRY.keys())


__all__ = [
    # Base classes
    "BaseRetriever",
    "RetrieverResult",
    # Retriever implementations
    "FlowDiffusionRetriever",
    "QueryAwareWeightedFlowDiffusion",  # Backward compatibility alias
    # Registry functions
    "get_retriever",
    "register_retriever",
    "list_retrievers",
    "RETRIEVER_REGISTRY",
]
