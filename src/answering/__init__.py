"""
Answering module for QAFD-RAG.

This module handles the query-time operations: from user question to final answer.
It coordinates keyword extraction, entity retrieval, graph traversal, context assembly,
and response generation.

Submodules:
-----------
- handler: Main query entry point (kg_query)
- context: Context building based on query modes
- clusters: Flow diffusion cluster operations
- text_units: Text chunk retrieval

Note: Evaluation functions have been moved to src/evaluation.py
"""

from .handler import kg_query
from .context import build_query_context
from .clusters import (
    get_embeddings_for_flow_diffusion,
    convert_subgraph_to_json,
    find_flow_diffusion_clusters_and_summarize,
    summarize_clusters_batch_with_llm,
    summarize_cluster_with_llm,
)
from .text_units import find_most_related_text_unit_from_entities

__all__ = [
    # Handler
    "kg_query",
    # Context
    "build_query_context",
    # Clusters
    "get_embeddings_for_flow_diffusion",
    "convert_subgraph_to_json",
    "find_flow_diffusion_clusters_and_summarize",
    "summarize_clusters_batch_with_llm",
    "summarize_cluster_with_llm",
    # Text units
    "find_most_related_text_unit_from_entities",
]
