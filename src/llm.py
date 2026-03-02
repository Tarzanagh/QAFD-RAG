import base64
import copy
import json
import os
import re
import struct
from functools import lru_cache
from typing import List, Dict, Callable, Any, Union, Optional
import aioboto3
import aiohttp
import numpy as np
import ollama
import torch
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    Timeout,
    AsyncAzureOpenAI,
)
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from transformers import AutoTokenizer, AutoModelForCausalLM

from .utils import (
    wrap_embedding_func_with_attrs,
    locate_json_string_body_from_string,
    safe_unicode_decode,
    logger,
)

import sys

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator

import warnings
import logging

# Suppress flash_attn and torch_dtype warnings from Jina v3
warnings.filterwarnings("ignore", message="flash_attn is not installed")
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated")
logging.getLogger("transformers_modules").setLevel(logging.ERROR)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================================
# GLOBAL API KEY STORAGE (to avoid async context issues)
# ============================================================================
_GLOBAL_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


# ============================================================================
# LLM COMPLETION FUNCTIONS
# ============================================================================

@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=4, max=100),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def openai_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    base_url="https://api.openai.com/v1",
    api_key="",
    **kwargs,
) -> str:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    openai_async_client = (
        AsyncOpenAI() if base_url is None else AsyncOpenAI(base_url=base_url)
    )
    
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    logger.debug("===== Query Input to LLM =====")
    logger.debug(f"Query: {prompt}")
    logger.debug(f"System prompt: {system_prompt}")
    
    if "response_format" in kwargs:
        # Use JSON mode for OpenAI models, strip for local models
        if any(prefix in model.lower() for prefix in ["gpt-4", "gpt-5"]):
            kwargs["response_format"] = {"type": "json_object"}
        else:
            kwargs.pop("response_format", None)

    if "gpt-5" in model.lower():
        # GPT-5 uses reasoning tokens that count against max_completion_tokens.
        # Need ~2000+ total (reasoning + output) for reliable responses.
        if "max_tokens" in kwargs:
            max_tokens_value = kwargs.pop("max_tokens")
            kwargs["max_completion_tokens"] = max(max_tokens_value, 2000)
            logger.debug(f"Converted max_tokens to max_completion_tokens for GPT-5")
        elif "max_completion_tokens" not in kwargs:
            kwargs["max_completion_tokens"] = 4000
            logger.debug(f"Set default max_completion_tokens=4000 for GPT-5")
    
    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    if hasattr(response, "__aiter__"):
        async def inner():
            async for chunk in response:
                content = chunk.choices[0].delta.content
                if content is None:
                    continue
                if r"\u" in content:
                    content = safe_unicode_decode(content.encode("utf-8"))
                yield content
        return inner()
    else:
        content = response.choices[0].message.content
        if r"\u" in content:
            content = safe_unicode_decode(content.encode("utf-8"))
        return content


class GPTKeywordExtractionFormat(BaseModel):
    high_level_keywords: List[str]
    low_level_keywords: List[str]


async def gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    keyword_extraction = keyword_extraction or kwargs.pop("keyword_extraction", False)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    keyword_extraction = keyword_extraction or kwargs.pop("keyword_extraction", False)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat
    return await openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_5_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    keyword_extraction = keyword_extraction or kwargs.pop("keyword_extraction", False)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat
    return await openai_complete_if_cache(
        "gpt-5",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_5_mini_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    keyword_extraction = keyword_extraction or kwargs.pop("keyword_extraction", False)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat

    if "max_tokens" in kwargs:
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")

    return await openai_complete_if_cache(
        "gpt-5-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_5_nano_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    """GPT-5 Nano - smallest, fastest, most affordable GPT-5 model"""
    keyword_extraction = keyword_extraction or kwargs.pop("keyword_extraction", False)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat

    # GPT-5 models use max_completion_tokens instead of max_tokens
    if "max_tokens" in kwargs:
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
    
    return await openai_complete_if_cache(
        "gpt-5-nano",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


# ============================================================================
# LOCAL / CUSTOM MODEL FUNCTIONS
# ============================================================================
# Configure via environment variables:
#   LOCAL_LLM_BASE_URL  - Base URL for your vLLM / OpenAI-compatible server
#   LOCAL_LLM_API_KEY   - API key (use "dummy" for local servers)
#   LOCAL_LLM_MODEL     - Model name served by the endpoint


def _get_local_llm_config():
    """Get local LLM configuration from environment variables."""
    base_url = os.environ.get("LOCAL_LLM_BASE_URL")
    if not base_url:
        raise ValueError(
            "LOCAL_LLM_BASE_URL not set. "
            "Export it, e.g.: export LOCAL_LLM_BASE_URL='http://localhost:8000/v1'"
        )
    return {
        "base_url": base_url,
        "api_key": os.environ.get("LOCAL_LLM_API_KEY", "dummy"),
        "model": os.environ.get("LOCAL_LLM_MODEL", "llm_base_model"),
    }


async def gpt_oss_120b_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """Complete using a local OpenAI-compatible server (configured via env vars)."""
    kwargs.pop("keyword_extraction", None)
    kwargs.pop("response_format", None)
    kwargs.pop("hashing_kv", None)

    cfg = _get_local_llm_config()
    return await openai_complete_if_cache(
        model=cfg["model"],
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        **kwargs,
    )


# ============================================================================
# EMBEDDING FUNCTIONS - LOCAL MODELS
# ============================================================================

@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
async def local_sentence_embedding(
    texts: list[str],
    model: str = "jinaai/jina-embeddings-v3",
    base_url=None,
    api_key="",
) -> np.ndarray:
    """Jina v3 local embedding (1024-dim, 8192 tokens) - standalone"""
    try:
        import sys
        from pathlib import Path
        
        embedding_models_path = Path(__file__).parent / "embedding_models"
        if str(embedding_models_path) not in sys.path:
            sys.path.insert(0, str(embedding_models_path))
        
        from JinaV3 import JinaV3EmbeddingModel
        from config import MinimalConfig
        
        if not hasattr(local_sentence_embedding, '_model'):
            logger.info("Initializing Jina v3 (local)...")
            
            config = MinimalConfig(
                embedding_model_name=model,
                embedding_batch_size=32,
                embedding_model_dtype="auto",
                embedding_return_as_normalized=True,
                embedding_max_seq_len=8192
            )
            
            local_sentence_embedding._model = JinaV3EmbeddingModel(
                global_config=config,
                embedding_model_name=model
            )
            
            logger.info("✅ Jina v3 ready (1024-dim, standalone)")
        
        embeddings = local_sentence_embedding._model.batch_encode(texts)
        return np.array(embeddings) if not isinstance(embeddings, np.ndarray) else embeddings
        
    except Exception as e:
        logger.error(f"Jina v3 failed ({e})")
        raise


@wrap_embedding_func_with_attrs(embedding_dim=4096, max_token_size=32768)
async def nvidia_nv_embed_v2_embedding(
    texts: list[str],
    model: str = "nvidia/NV-Embed-v2",
    **kwargs
) -> np.ndarray:
    """NVIDIA NV-Embed-v2 local embedding (4096-dim, 32K tokens)"""
    try:
        # Import from local embedding_models directory
        import sys
        from pathlib import Path
        
        embedding_models_path = Path(__file__).parent / "embedding_models"
        if str(embedding_models_path) not in sys.path:
            sys.path.insert(0, str(embedding_models_path))
        
        from NVEmbedV2 import NVEmbedV2EmbeddingModel
        from config import MinimalConfig
        
        if not hasattr(nvidia_nv_embed_v2_embedding, '_model'):
            logger.info("Initializing NVIDIA NV-Embed-v2 (local)...")
            
            config = MinimalConfig(
                embedding_model_name=model,
                embedding_batch_size=16,
                embedding_model_dtype="auto",
                embedding_return_as_normalized=True,
                embedding_max_seq_len=32768
            )
            
            nvidia_nv_embed_v2_embedding._model = NVEmbedV2EmbeddingModel(
                global_config=config,
                embedding_model_name=model
            )
            
            logger.info("✅ NVIDIA NV-Embed-v2 ready (4096-dim, standalone)")
        
        embeddings = nvidia_nv_embed_v2_embedding._model.batch_encode(texts)
        return np.array(embeddings) if not isinstance(embeddings, np.ndarray) else embeddings
        
    except Exception as e:
        logger.warning(f"NVIDIA failed ({e}), using Jina v3")
        return await local_sentence_embedding(texts)



@wrap_embedding_func_with_attrs(embedding_dim=4096, max_token_size=8192)
async def gritlm_embedding(
    texts: list[str],
    model: str = "GritLM/GritLM-7B",
    **kwargs
) -> np.ndarray:
    """GritLM local embedding (7168-dim, 8192 tokens) - standalone"""
    try:
        import sys
        from pathlib import Path
        
        embedding_models_path = Path(__file__).parent / "embedding_models"
        if str(embedding_models_path) not in sys.path:
            sys.path.insert(0, str(embedding_models_path))
        
        from GritLM import GritLMEmbeddingModel
        from config import MinimalConfig
        
        if not hasattr(gritlm_embedding, '_model'):
            logger.info("Initializing GritLM (local)...")
            
            config = MinimalConfig(
                embedding_model_name=model,
                embedding_batch_size=16,
                embedding_model_dtype="auto",
                embedding_return_as_normalized=True,
                embedding_max_seq_len=8192
            )
            
            gritlm_embedding._model = GritLMEmbeddingModel(
                global_config=config,
                embedding_model_name=model
            )
            
            logger.info("✅ GritLM ready (7168-dim, standalone)")
        
        embeddings = gritlm_embedding._model.batch_encode(texts, **kwargs)
        return np.array(embeddings) if not isinstance(embeddings, np.ndarray) else embeddings
        
    except Exception as e:
        logger.warning(f"GritLM failed ({e}), using Jina v3")
        return await local_sentence_embedding(texts)


# ============================================================================
# EMBEDDING FUNCTIONS - CLOUD/API MODELS
# ============================================================================

@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=4, max=100),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def openai_small_embedding(
    texts: list[str],
    model: str = "text-embedding-3-small",
    dimensions: int = 1536,
    base_url="https://api.openai.com/v1",
    api_key="",
) -> np.ndarray:
    """OpenAI text-embedding-3-small - standalone (1536-dim)"""
    
    # ✅ FIX: Use global API key captured at module import time
    global _GLOBAL_OPENAI_API_KEY
    if not api_key or api_key == "":
        api_key = _GLOBAL_OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
    
    try:
        import sys
        from pathlib import Path
        
        embedding_models_path = Path(__file__).parent / "embedding_models"
        if str(embedding_models_path) not in sys.path:
            sys.path.insert(0, str(embedding_models_path))
        
        from OpenAI import OpenAIEmbeddingModel
        from config import MinimalConfig
        
        if not hasattr(openai_small_embedding, '_model') or openai_small_embedding._last_dims != dimensions:
            logger.info(f"Initializing OpenAI Small (standalone): {model} with {dimensions}-dim")
            
            config = MinimalConfig(
                embedding_model_name=model,
                embedding_batch_size=100,
                embedding_dimensions=dimensions,
                # openai_api_key=api_key  # Pass the API key here
            )
            
            openai_small_embedding._model = OpenAIEmbeddingModel(
                global_config=config,
                embedding_model_name=model,
                api_key=api_key
            )
            openai_small_embedding._last_dims = dimensions
            
            logger.info(f"✅ OpenAI Small ready ({dimensions}-dim, standalone)")
        
        embeddings = openai_small_embedding._model.batch_encode(texts)
        return np.array(embeddings) if not isinstance(embeddings, np.ndarray) else embeddings
        
    except Exception as e:
        logger.warning(f"OpenAI Small standalone failed ({e}), using direct API call")
        
        # API key already checked at the start
        if base_url is None:
            openai_async_client = AsyncOpenAI(api_key=api_key)
        else:
            openai_async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        
        response = await openai_async_client.embeddings.create(
            model=model, 
            input=texts, 
            encoding_format="float",
            dimensions=dimensions
        )
        return np.array([dp.embedding for dp in response.data])


@wrap_embedding_func_with_attrs(embedding_dim=3072, max_token_size=8192)
@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=4, max=100),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def openai_large_embedding(
    texts: list[str],
    model: str = "text-embedding-3-large",
    dimensions: int = 3072,
    base_url="https://api.openai.com/v1",
    api_key="",
) -> np.ndarray:
    """OpenAI text-embedding-3-large - standalone (3072-dim)"""
    
    # ✅ FIX: Use global API key captured at module import time
    global _GLOBAL_OPENAI_API_KEY
    if not api_key or api_key == "":
        api_key = _GLOBAL_OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
    
    try:
        import sys
        from pathlib import Path
        
        embedding_models_path = Path(__file__).parent / "embedding_models"
        if str(embedding_models_path) not in sys.path:
            sys.path.insert(0, str(embedding_models_path))
        
        from OpenAI import OpenAIEmbeddingModel
        from config import MinimalConfig
        
        if not hasattr(openai_large_embedding, '_model') or openai_large_embedding._last_dims != dimensions:
            logger.info(f"Initializing OpenAI Large (standalone): {model} with {dimensions}-dim")
            
            # ✅ FIX: Pass API key to config
            config = MinimalConfig(
                embedding_model_name=model,
                embedding_batch_size=100,
                embedding_dimensions=dimensions,
                # openai_api_key=api_key  # Pass the API key here
            )
            
            openai_large_embedding._model = OpenAIEmbeddingModel(
                global_config=config,
                embedding_model_name=model,
                api_key=api_key
            )
            openai_large_embedding._last_dims = dimensions
            
            logger.info(f"✅ OpenAI Large ready ({dimensions}-dim, standalone)")
        
        embeddings = openai_large_embedding._model.batch_encode(texts)
        return np.array(embeddings) if not isinstance(embeddings, np.ndarray) else embeddings
        
    except Exception as e:
        logger.warning(f"OpenAI Large standalone failed ({e}), using direct API call")
        
        # API key already checked at the start
        if base_url is None:
            openai_async_client = AsyncOpenAI(api_key=api_key)
        else:
            openai_async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        
        response = await openai_async_client.embeddings.create(
            model=model, 
            input=texts, 
            encoding_format="float",
            dimensions=dimensions
        )
        return np.array([dp.embedding for dp in response.data])


# Backward compatibility alias (defaults to large)
async def openai_cloud_embedding(*args, **kwargs):
    """Alias for backward compatibility - routes to openai_large_embedding"""
    return await openai_large_embedding(*args, **kwargs)


@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
async def openai_embedding(
    texts: list[str],
    model: str = "jinaai/jina-embeddings-v3",
    base_url=None,
    api_key="",
) -> np.ndarray:
    """
    Smart embedding function - ALWAYS 1024 dimensions.
    
    Uses LOCAL by default (Jina v3, 1024-dim, free).
    Set USE_OPENAI_EMBEDDINGS=1 to use OpenAI (1024-dim, costs money).
    """
    use_openai = os.environ.get("USE_OPENAI_EMBEDDINGS", "0") == "1"
    
    if use_openai:
        logger.info("Using OpenAI cloud embeddings (1024-dim)")
        return await openai_large_embedding(texts, model="text-embedding-3-large", dimensions=1024, base_url=base_url, api_key=api_key)
    else:
        return await local_sentence_embedding(texts, model=model, base_url=base_url, api_key=api_key)


# ============================================================================
# EMBEDDING MODEL REGISTRY
# ============================================================================

EMBEDDING_CONFIGS = {
    # ===== LOCAL EMBEDDINGS (FREE) =====
    "jina-v3": {
        "provider": "local",
        "model": "jinaai/jina-embeddings-v3",
        "dimensions": 1024,
        "max_tokens": 8192,
        "description": "Jina Embeddings v3 (1024-dim, 8K context, free, local)",
        "embedding_func": local_sentence_embedding,
        "cost": "free",
    },
    "nvidia-nv-embed-v2": {
        "provider": "local",
        "model": "nvidia/NV-Embed-v2",
        "dimensions": 4096,
        "max_tokens": 32768,
        "description": "NVIDIA NV-Embed-v2 (4096-dim, 32K context, local, high quality)",
        "embedding_func": nvidia_nv_embed_v2_embedding,
        "cost": "free",
    },
    "gritlm": {
        "provider": "local",
        "model": "GritLM/GritLM-7B",
        "dimensions": 4096,
        "max_tokens": 8192,
        "description": "GritLM (4096-dim, 8K context, local, unified embedding+generation)",
        "embedding_func": gritlm_embedding,
        "cost": "free",
    },
    
    # ===== CLOUD EMBEDDINGS (PAID) =====
    "openai-large": {
        "provider": "openai",
        "model": "text-embedding-3-large",
        "dimensions": 3072,
        "max_tokens": 8192,
        "description": "OpenAI text-embedding-3-large (3072-dim, cloud, costs money)",
        "embedding_func": openai_large_embedding,
        "cost": "$0.13/1M tokens",
    },
    "openai-small": {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "dimensions": 1536,
        "max_tokens": 8192,
        "description": "OpenAI text-embedding-3-small (1536-dim, cloud, costs money)",
        "embedding_func": openai_small_embedding,
        "cost": "$0.02/1M tokens",
    },
}


def get_embedding_func_for_model(embedding_key: str):
    """Get embedding function and config for specified embedding model"""
    if embedding_key not in EMBEDDING_CONFIGS:
        logger.warning(f"Unknown embedding model '{embedding_key}', defaulting to 'jina-v3'")
        embedding_key = "jina-v3"
    
    config = EMBEDDING_CONFIGS[embedding_key]
    logger.info(f"[Embedding] {embedding_key} → {config['description']}")
    
    return config["embedding_func"], config["dimensions"], config


def list_available_embeddings():
    """Print all available embedding models"""
    print("\n" + "=" * 80)
    print("📊 Available Embedding Models")
    print("=" * 80)
    
    print("\n🆓 FREE LOCAL MODELS:")
    for key, config in EMBEDDING_CONFIGS.items():
        if config["cost"] == "free":
            print(f"\n  {key:25s} → {config['description']}")
            print(f"  {'':25s}   Dimensions: {config['dimensions']}, Max tokens: {config['max_tokens']}")
    
    print("\n💰 PAID CLOUD MODELS:")
    for key, config in EMBEDDING_CONFIGS.items():
        if config["cost"] != "free":
            print(f"\n  {key:25s} → {config['description']}")
            print(f"  {'':25s}   Dimensions: {config['dimensions']}, Cost: {config['cost']}")
    
    print("\n" + "=" * 80 + "\n")


# ============================================================================
# MODEL TYPE DETECTION HELPERS
# ============================================================================

LOCAL_MODEL_PREFIXES = [
    "gpt-oss", "qwen", "llama", "mistral", "internlm", "ollama", "local"
]
CLOUD_MODEL_PREFIXES = [
    "gpt-4", "gpt-5", "nvidia", "azure", "bedrock", "zhipu", "openai"
]

def is_local_model(model_name: str) -> bool:
    """Return True if model_name corresponds to a locally hosted model."""
    if not model_name:
        return False
    model_name = model_name.lower()
    local = any(model_name.startswith(prefix) for prefix in LOCAL_MODEL_PREFIXES)
    if not local and not any(model_name.startswith(prefix) for prefix in CLOUD_MODEL_PREFIXES):
        logger.warning(f"[Model Detection] Unknown model prefix for '{model_name}'. Defaulting to cloud mode.")
    return local

def is_cloud_model(model_name: str) -> bool:
    """Return True if model_name corresponds to a cloud-hosted model."""
    if not model_name:
        return False
    model_name = model_name.lower()
    return any(model_name.startswith(prefix) for prefix in CLOUD_MODEL_PREFIXES)


# =====================================================================
# STARTUP SUMMARY (auto-prints when QAFD.llm is imported)
# =====================================================================

# Embedding mode (for reference, no startup print)
_embedding_mode_env = os.environ.get("EMBEDDING_MODEL_KEY", "auto")


if __name__ == "__main__":
    import asyncio

    async def main():
        # Test listing
        list_available_embeddings()
        
        # Test embedding function
        result = await gpt_4o_mini_complete("How are you?")
        print(result)

    asyncio.run(main())