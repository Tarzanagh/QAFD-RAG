"""
Context building for RAG query processing.

This module provides functions to build query context from the knowledge graph
based on different query modes (local, global, hybrid).
"""

import asyncio
from ..base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from ..utils import (
    logger,
    list_of_list_to_csv,
    csv_string_to_list,
)
from .clusters import find_flow_diffusion_clusters_and_summarize
from .text_units import find_most_related_text_unit_from_entities


async def build_query_context(
    query: list,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    """
    Build query context based on extracted keywords and query mode.

    Parameters:
    -----------
    query : list
        List containing [ll_keywords, hl_keywords]
    knowledge_graph_inst : BaseGraphStorage
        Knowledge graph storage instance
    entities_vdb : BaseVectorStorage
        Entity vector database
    relationships_vdb : BaseVectorStorage
        Relationships vector database
    text_chunks_db : BaseKVStorage[TextChunkSchema]
        Text chunks database
    query_param : QueryParam
        Query parameters including mode (local/global/hybrid)
    global_config : dict
        Global configuration

    Returns:
    --------
    str
        Formatted context string for LLM response generation
    """
    ll_keywords, hl_keywords = query[0], query[1]

    # Initialize context variables
    entities_context, relations_context, text_units_context = "", "", ""

    if query_param.mode == "local":
        # Local mode: use ll_keywords only
        if ll_keywords == "":
            logger.warning("Low level keywords is empty for local mode")
            return "", "", ""

        (
            entities_context,
            relations_context,
            text_units_context,
        ) = await _get_node_data_with_flow_diffusion(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
            global_config,
        )

    elif query_param.mode == "global":
        # Global mode: use hl_keywords only
        if hl_keywords == "":
            logger.warning("High level keywords is empty for global mode")
            return "", "", ""

        (
            entities_context,
            relations_context,
            text_units_context,
        ) = await _get_node_data_with_flow_diffusion(
            hl_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
            global_config,
        )

    elif query_param.mode == "hybrid":
        # Hybrid mode: use combined keywords (both ll_keywords and hl_keywords)
        if ll_keywords == "" and hl_keywords == "":
            logger.warning("Both Low Level and High Level keywords are empty for hybrid mode")
            return "", "", ""

        # Get local information using ll_keywords
        local_entities_context, local_relations_context, local_text_units_context = "", "", ""
        if ll_keywords:
            (
                local_entities_context,
                local_relations_context,
                local_text_units_context,
            ) = await _get_node_data_with_flow_diffusion(
                ll_keywords,
                knowledge_graph_inst,
                entities_vdb,
                text_chunks_db,
                query_param,
                global_config,
            )

        # Get global information using hl_keywords
        global_entities_context, global_relations_context, global_text_units_context = "", "", ""
        if hl_keywords:
            (
                global_entities_context,
                global_relations_context,
                global_text_units_context,
            ) = await _get_node_data_with_flow_diffusion(
                hl_keywords,
                knowledge_graph_inst,
                entities_vdb,
                text_chunks_db,
                query_param,
                global_config,
            )

    # Return context based on mode
    if query_param.mode == "local":
        if query_param.return_raw_entities:
            return entities_context
        elif query_param.return_raw_clusters:
            return relations_context

        return f"""
-----local-information-----
-----low-level entity information-----
```csv
{entities_context}
```
-----low-level relationship information-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""
    elif query_param.mode == "global":
        if query_param.return_raw_entities:
            return entities_context
        elif query_param.return_raw_clusters:
            return relations_context

        return f"""
-----global-information-----
-----high-level entity information-----
```csv
{entities_context}
```
-----high-level relationship information-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""
    elif query_param.mode == "hybrid":
        if query_param.return_raw_entities:
            # Merge local + global entities CSV into one CSV and reindex id
            merged_rows = []
            if local_entities_context:
                merged_rows += csv_string_to_list(local_entities_context)[1:]
            if global_entities_context:
                merged_rows += csv_string_to_list(global_entities_context)[1:]
            for idx, row in enumerate(merged_rows):
                if row:
                    row[0] = str(idx)
            merged_entities_csv = list_of_list_to_csv(
                [["id", "entity", "entity_type", "description", "rank"]] + merged_rows
            )
            return merged_entities_csv
        elif query_param.return_raw_clusters:
            return local_relations_context + global_relations_context

        return f"""
-----hybrid-information-----
-----local information (from low-level keywords)-----
-----local entity information-----
```csv
{local_entities_context}
```
-----local relationship information-----
```csv
{local_relations_context}
```
-----local sources-----
```csv
{local_text_units_context}
```
-----global information (from high-level keywords)-----
-----global entity information-----
```csv
{global_entities_context}
```
-----global relationship information-----
```csv
{global_relations_context}
```
-----global sources-----
```csv
{global_text_units_context}
```
"""
    else:
        return ""


async def _get_node_data_with_flow_diffusion(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    """
    Get node data using flow diffusion for finding relationships.

    Parameters:
    -----------
    query : str
        Query string (can be either ll_keywords or hl_keywords)
    knowledge_graph_inst : BaseGraphStorage
        Knowledge graph storage instance
    entities_vdb : BaseVectorStorage
        Entity vector database
    text_chunks_db : BaseKVStorage[TextChunkSchema]
        Text chunks database
    query_param : QueryParam
        Query parameters
    global_config : dict
        Global configuration

    Returns:
    --------
    tuple
        (entities_context, relations_context, text_units_context)
    """
    results = await entities_vdb.query(query, top_k=query_param.max_source_nodes)
    if not len(results):
        return "", "", ""

    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]

    use_text_units = await find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )

    # Use flow diffusion instead of the original relationship finding method
    use_relations = await find_flow_diffusion_clusters_and_summarize(
        node_datas, query, query_param, knowledge_graph_inst, global_config
    )

    logger.info(
        f"Flow diffusion query uses {len(node_datas)} entities, {len(use_relations)} cluster summaries, {len(use_text_units)} text units"
    )

    entites_section_list = [["id", "entity", "entity_type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append([
            i,
            n["entity_name"],
            n.get("entity_type", "UNKNOWN"),
            n.get("description", "UNKNOWN"),
            n["rank"],
        ])
    entities_context = list_of_list_to_csv(entites_section_list)

    # Relations context: return JSON clusters when requested; otherwise CSV
    if query_param.return_raw_clusters:
        relations_context = use_relations
    else:
        relations_section_list = [["id", "cluster_summary"]]
        for i, summary in enumerate(use_relations):
            relations_section_list.append([i, summary])
        relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    return entities_context, relations_context, text_units_context
