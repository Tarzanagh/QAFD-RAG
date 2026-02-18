"""
Caching utilities for QAFD-RAG.

Provides embedding-based caching with similarity matching.
"""

import json
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from .logging import logger


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Parameters:
    -----------
    v1, v2 : np.ndarray
        Input vectors

    Returns:
    --------
    float
        Cosine similarity score
    """
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot_product / (norm1 * norm2)


def quantize_embedding(embedding: np.ndarray, bits: int = 8) -> tuple:
    """
    Quantize an embedding to reduce storage size.

    Parameters:
    -----------
    embedding : np.ndarray
        Float embedding vector
    bits : int
        Bit depth for quantization (default: 8)

    Returns:
    --------
    tuple
        (quantized_array, min_val, max_val)
    """
    min_val = embedding.min()
    max_val = embedding.max()

    scale = (2**bits - 1) / (max_val - min_val)
    quantized = np.round((embedding - min_val) * scale).astype(np.uint8)

    return quantized, min_val, max_val


def dequantize_embedding(
    quantized: np.ndarray, min_val: float, max_val: float, bits: int = 8
) -> np.ndarray:
    """
    Dequantize an embedding back to float values.

    Parameters:
    -----------
    quantized : np.ndarray
        Quantized embedding
    min_val : float
        Original minimum value
    max_val : float
        Original maximum value
    bits : int
        Bit depth used in quantization

    Returns:
    --------
    np.ndarray
        Reconstructed float embedding
    """
    scale = (max_val - min_val) / (2**bits - 1)
    return (quantized * scale + min_val).astype(np.float32)


async def get_best_cached_response(
    hashing_kv,
    current_embedding,
    similarity_threshold: float = 0.95,
    mode: str = "default",
    use_llm_check: bool = False,
    llm_func=None,
    original_prompt: Optional[str] = None,
) -> Union[str, None]:
    """
    Find the best matching cached response based on embedding similarity.

    Parameters:
    -----------
    hashing_kv : BaseKVStorage
        Key-value storage for cache
    current_embedding : np.ndarray
        Embedding of the current query
    similarity_threshold : float
        Minimum similarity score to accept
    mode : str
        Cache mode/namespace
    use_llm_check : bool
        Whether to use LLM for additional similarity verification
    llm_func : callable, optional
        LLM function for similarity checking
    original_prompt : str, optional
        Original prompt for LLM comparison

    Returns:
    --------
    str or None
        Cached response if found, None otherwise
    """
    # Import here to avoid circular import
    from ..prompts import PROMPTS

    mode_cache = await hashing_kv.get_by_id(mode)
    if not mode_cache:
        return None

    best_similarity = -1
    best_response = None
    best_prompt = None
    best_cache_id = None

    for cache_id, cache_data in mode_cache.items():
        if cache_data["embedding"] is None:
            continue

        cached_quantized = np.frombuffer(
            bytes.fromhex(cache_data["embedding"]), dtype=np.uint8
        ).reshape(cache_data["embedding_shape"])
        cached_embedding = dequantize_embedding(
            cached_quantized,
            cache_data["embedding_min"],
            cache_data["embedding_max"],
        )

        similarity = cosine_similarity(current_embedding, cached_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_response = cache_data["return"]
            best_prompt = cache_data["original_prompt"]
            best_cache_id = cache_id

    if best_similarity > similarity_threshold:
        if use_llm_check and llm_func and original_prompt and best_prompt:
            compare_prompt = PROMPTS["similarity_check"].format(
                original_prompt=original_prompt, cached_prompt=best_prompt
            )

            try:
                llm_result = await llm_func(compare_prompt)
                llm_result = llm_result.strip()
                llm_similarity = float(llm_result)

                best_similarity = llm_similarity
                if best_similarity < similarity_threshold:
                    log_data = {
                        "event": "llm_check_cache_rejected",
                        "original_question": original_prompt[:100] + "..."
                        if len(original_prompt) > 100
                        else original_prompt,
                        "cached_question": best_prompt[:100] + "..."
                        if len(best_prompt) > 100
                        else best_prompt,
                        "similarity_score": round(best_similarity, 4),
                        "threshold": similarity_threshold,
                    }
                    logger.info(json.dumps(log_data, ensure_ascii=False))
                    return None
            except Exception as e:
                logger.warning(f"LLM similarity check failed: {e}")
                return None

        prompt_display = (
            best_prompt[:50] + "..." if len(best_prompt) > 50 else best_prompt
        )
        log_data = {
            "event": "cache_hit",
            "mode": mode,
            "similarity": round(best_similarity, 4),
            "cache_id": best_cache_id,
            "original_prompt": prompt_display,
        }
        logger.info(json.dumps(log_data, ensure_ascii=False))
        return best_response
    return None


async def handle_cache(hashing_kv, args_hash: str, prompt: str, mode: str = "default"):
    """
    Handle cache lookup with optional embedding-based matching.

    Parameters:
    -----------
    hashing_kv : BaseKVStorage
        Key-value storage for cache
    args_hash : str
        Hash of the arguments for exact matching
    prompt : str
        The prompt text for embedding-based matching
    mode : str
        Cache mode/namespace

    Returns:
    --------
    tuple
        (cached_response, quantized, min_val, max_val)
    """
    if hashing_kv is None:
        return None, None, None, None

    if mode == "naive":
        mode_cache = await hashing_kv.get_by_id(mode) or {}
        if args_hash in mode_cache:
            return mode_cache[args_hash]["return"], None, None, None
        return None, None, None, None

    embedding_cache_config = hashing_kv.global_config.get(
        "embedding_cache_config",
        {"enabled": False, "similarity_threshold": 0.95, "use_llm_check": False},
    )
    is_embedding_cache_enabled = embedding_cache_config["enabled"]
    use_llm_check = embedding_cache_config.get("use_llm_check", False)

    quantized = min_val = max_val = None
    if is_embedding_cache_enabled:
        embedding_model_func = hashing_kv.global_config["embedding_func"]["func"]
        llm_model_func = hashing_kv.global_config.get("llm_model_func")

        current_embedding = await embedding_model_func([prompt])
        quantized, min_val, max_val = quantize_embedding(current_embedding[0])
        best_cached_response = await get_best_cached_response(
            hashing_kv,
            current_embedding[0],
            similarity_threshold=embedding_cache_config["similarity_threshold"],
            mode=mode,
            use_llm_check=use_llm_check,
            llm_func=llm_model_func if use_llm_check else None,
            original_prompt=prompt if use_llm_check else None,
        )
        if best_cached_response is not None:
            return best_cached_response, None, None, None
    else:
        mode_cache = await hashing_kv.get_by_id(mode) or {}
        if args_hash in mode_cache:
            return mode_cache[args_hash]["return"], None, None, None

    return None, quantized, min_val, max_val


@dataclass
class CacheData:
    """Data structure for cache entries."""

    args_hash: str
    content: str
    prompt: str
    quantized: Optional[np.ndarray] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mode: str = "default"


async def save_to_cache(hashing_kv, cache_data: CacheData):
    """
    Save data to the cache.

    Parameters:
    -----------
    hashing_kv : BaseKVStorage
        Key-value storage for cache
    cache_data : CacheData
        Data to cache
    """
    if hashing_kv is None or hasattr(cache_data.content, "__aiter__"):
        return

    mode_cache = await hashing_kv.get_by_id(cache_data.mode) or {}

    mode_cache[cache_data.args_hash] = {
        "return": cache_data.content,
        "embedding": cache_data.quantized.tobytes().hex()
        if cache_data.quantized is not None
        else None,
        "embedding_shape": cache_data.quantized.shape
        if cache_data.quantized is not None
        else None,
        "embedding_min": cache_data.min_val,
        "embedding_max": cache_data.max_val,
        "original_prompt": cache_data.prompt,
    }

    await hashing_kv.upsert({cache_data.mode: mode_cache})


__all__ = [
    "cosine_similarity",
    "quantize_embedding",
    "dequantize_embedding",
    "get_best_cached_response",
    "handle_cache",
    "CacheData",
    "save_to_cache",
]
