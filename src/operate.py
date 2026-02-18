"""
QAFD-RAG Query Operations Facade.

This module provides backward compatibility for the refactored codebase.
The actual implementations have been moved to:

- answering/handler.py: Main query entry point (kg_query)
- answering/context.py: Context building based on query modes
- answering/clusters.py: Flow diffusion cluster operations
- answering/text_units.py: Text chunk retrieval
- evaluation.py: Answer evaluation and comparison (standalone module)

For new code, prefer importing directly from the respective modules.
"""

# Re-export from indexing (backward compatibility)
from .indexing import (
    chunking_by_token_size,
    extract_entities,
    _handle_entity_relation_summary,
    _handle_single_entity_extraction,
    _handle_single_relationship_extraction,
    _merge_nodes_then_upsert,
    _merge_edges_then_upsert,
)

# Re-export from retrievers (backward compatibility)
from .retrievers import FlowDiffusionRetriever, QueryAwareWeightedFlowDiffusion

# Re-export from answering module
from .answering import (
    # Main query function
    kg_query,
    # Context building
    build_query_context as _build_query_context,
    # Cluster operations
    get_embeddings_for_flow_diffusion as _get_embeddings_for_flow_diffusion,
    convert_subgraph_to_json as _convert_subgraph_to_json,
    find_flow_diffusion_clusters_and_summarize as _find_flow_diffusion_clusters_and_summarize,
    summarize_clusters_batch_with_llm as _summarize_clusters_batch_with_llm,
    summarize_cluster_with_llm as _summarize_cluster_with_llm,
    # Text units
    find_most_related_text_unit_from_entities as _find_most_related_text_unit_from_entities,
)

# Re-export from evaluation module (standalone)
from .evaluation import (
    evaluate_multiple_answers,
    compare_two_answers,
    calculate_evaluation_metrics,
)

# Note: The following functions have been moved to src/answering/:
# - kg_query -> answering/handler.py
# - _build_query_context -> answering/context.py
# - _get_node_data_with_flow_diffusion -> answering/context.py
# - _get_embeddings_for_flow_diffusion -> answering/clusters.py
# - _convert_subgraph_to_json -> answering/clusters.py
# - _find_flow_diffusion_clusters_and_summarize -> answering/clusters.py
# - _summarize_clusters_batch_with_llm -> answering/clusters.py
# - _summarize_cluster_with_llm -> answering/clusters.py
# - _find_most_related_text_unit_from_entities -> answering/text_units.py

# Note: Evaluation functions have been moved to src/evaluation.py:
# - evaluate_multiple_answers
# - compare_two_answers
# - calculate_evaluation_metrics

# The following functions have been moved to src/indexing/extractors.py:
# - _handle_entity_relation_summary
# - _handle_single_entity_extraction
# - _handle_single_relationship_extraction
# - _merge_nodes_then_upsert
# - _merge_edges_then_upsert
# - extract_entities
# They are imported above for backward compatibility.

__all__ = [
    # Indexing functions
    "chunking_by_token_size",
    "extract_entities",
    "_handle_entity_relation_summary",
    "_handle_single_entity_extraction",
    "_handle_single_relationship_extraction",
    "_merge_nodes_then_upsert",
    "_merge_edges_then_upsert",
    # Retriever classes
    "FlowDiffusionRetriever",
    "QueryAwareWeightedFlowDiffusion",
    # Query functions
    "kg_query",
    "_build_query_context",
    "_get_embeddings_for_flow_diffusion",
    "_convert_subgraph_to_json",
    "_find_flow_diffusion_clusters_and_summarize",
    "_summarize_clusters_batch_with_llm",
    "_summarize_cluster_with_llm",
    "_find_most_related_text_unit_from_entities",
    # Evaluation functions
    "evaluate_multiple_answers",
    "compare_two_answers",
    "calculate_evaluation_metrics",
]
