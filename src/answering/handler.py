"""
Main query handler for the QAFD-RAG system.

This module provides the main entry point for knowledge graph queries,
handling keyword extraction, context building, and response generation.
"""

import re
import json
from ..base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from ..prompts import PROMPTS
from ..utils import (
    logger,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
)
from .context import build_query_context


async def kg_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
    hashing_kv: BaseKVStorage = None,
) -> str:
    """
    Main query function for the QAFD-RAG system.

    This function orchestrates the entire query pipeline:
    1. Check cache for existing response
    2. Extract keywords from query using LLM
    3. Build context from knowledge graph
    4. Generate response using LLM

    Parameters:
    -----------
    query : str
        The user's query string
    knowledge_graph_inst : BaseGraphStorage
        Knowledge graph storage instance
    entities_vdb : BaseVectorStorage
        Entity vector database
    relationships_vdb : BaseVectorStorage
        Relationships vector database
    text_chunks_db : BaseKVStorage[TextChunkSchema]
        Text chunks database
    query_param : QueryParam
        Query parameters including mode, top_k, etc.
    global_config : dict
        Global configuration including LLM functions
    hashing_kv : BaseKVStorage, optional
        Cache storage for responses

    Returns:
    --------
    str
        The generated response or context (based on query_param settings)
    """
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query)
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode
    )
    if cached_response is not None:
        return cached_response

    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["keywords_extraction_examples"]):
        examples = "\n".join(
            PROMPTS["keywords_extraction_examples"][: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS["keywords_extraction_examples"])
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    if query_param.mode not in ["local", "global", "hybrid"]:
        logger.error(f"Unknown mode {query_param.mode} in kg_query")
        return PROMPTS["fail_response"]

    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query, examples=examples, language=language)
    result = await use_model_func(kw_prompt, keyword_extraction=True)
    logger.debug("kw_prompt result: %s", result)
    try:
        match = re.search(r"\{.*\}", result, re.DOTALL)
        if match:
            result = match.group(0)
            keywords_data = json.loads(result)

            hl_keywords = keywords_data.get("high_level_keywords", [])
            ll_keywords = keywords_data.get("low_level_keywords", [])
        else:
            logger.error("No JSON-like structure found in the result.")
            return PROMPTS["fail_response"]

    except json.JSONDecodeError as e:
        logger.warning("JSON parsing error: %s - %s", e, result)
        return PROMPTS["fail_response"]

    if hl_keywords == [] and ll_keywords == []:
        logger.warning("low_level_keywords and high_level_keywords is empty")
        return PROMPTS["fail_response"]

    # Validate keywords based on mode
    if query_param.mode == "local" and ll_keywords == []:
        logger.warning("low_level_keywords is empty for local mode")
        return PROMPTS["fail_response"]
    elif query_param.mode == "global" and hl_keywords == []:
        logger.warning("high_level_keywords is empty for global mode")
        return PROMPTS["fail_response"]
    elif query_param.mode == "hybrid" and ll_keywords == [] and hl_keywords == []:
        logger.warning("Both low_level_keywords and high_level_keywords are empty for hybrid mode")
        return PROMPTS["fail_response"]

    # Convert lists to strings
    if ll_keywords:
        ll_keywords = ", ".join(ll_keywords)
    else:
        ll_keywords = ""

    if hl_keywords:
        hl_keywords = ", ".join(hl_keywords)
    else:
        hl_keywords = ""

    keywords = [ll_keywords, hl_keywords]
    context = await build_query_context(
        keywords,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        global_config,
    )

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    if query_param.only_need_prompt:
        return sys_prompt
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
        ),
    )
    return response
