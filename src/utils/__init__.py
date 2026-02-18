"""
QAFD-RAG Utilities Package.

This package provides various utility functions for the QAFD-RAG system.

Modules:
--------
- hashing: MD5-based hashing functions
- tokenization: Tiktoken-based encoding/decoding
- parsing: JSON extraction and string manipulation
- data: CSV conversion and context combination
- logging: Centralized logging configuration
- embedding: Embedding function wrappers
- async_utils: Async function decorators
- file_io: JSON and XML file operations
- caching: Embedding-based caching utilities
"""

# Hashing utilities
from .hashing import (
    compute_args_hash,
    compute_mdhash_id,
)

# Tokenization utilities
from .tokenization import (
    encode_string_by_tiktoken,
    decode_tokens_by_tiktoken,
    truncate_list_by_token_size,
)

# Parsing utilities
from .parsing import (
    locate_json_string_body_from_string,
    convert_response_to_json,
    split_string_by_multi_markers,
    clean_str,
    is_float_regex,
    safe_unicode_decode,
    pack_user_ass_to_openai_messages,
)

# Data utilities
from .data import (
    list_of_list_to_csv,
    csv_string_to_list,
    process_combine_contexts,
)

# Logging utilities
from .logging import (
    logger,
    set_logger,
)

# Embedding utilities
from .embedding import (
    UnlimitedSemaphore,
    EmbeddingFunc,
    wrap_embedding_func_with_attrs,
)

# Async utilities
from .async_utils import (
    limit_async_func_call,
)

# File I/O utilities
from .file_io import (
    load_json,
    write_json,
    save_data_to_file,
    xml_to_json,
)

# Caching utilities
from .caching import (
    cosine_similarity,
    quantize_embedding,
    dequantize_embedding,
    get_best_cached_response,
    handle_cache,
    CacheData,
    save_to_cache,
)


__all__ = [
    # Hashing
    "compute_args_hash",
    "compute_mdhash_id",
    # Tokenization
    "encode_string_by_tiktoken",
    "decode_tokens_by_tiktoken",
    "truncate_list_by_token_size",
    # Parsing
    "locate_json_string_body_from_string",
    "convert_response_to_json",
    "split_string_by_multi_markers",
    "clean_str",
    "is_float_regex",
    "safe_unicode_decode",
    "pack_user_ass_to_openai_messages",
    # Data
    "list_of_list_to_csv",
    "csv_string_to_list",
    "process_combine_contexts",
    # Logging
    "logger",
    "set_logger",
    # Embedding
    "UnlimitedSemaphore",
    "EmbeddingFunc",
    "wrap_embedding_func_with_attrs",
    # Async
    "limit_async_func_call",
    # File I/O
    "load_json",
    "write_json",
    "save_data_to_file",
    "xml_to_json",
    # Caching
    "cosine_similarity",
    "quantize_embedding",
    "dequantize_embedding",
    "get_best_cached_response",
    "handle_cache",
    "CacheData",
    "save_to_cache",
]
