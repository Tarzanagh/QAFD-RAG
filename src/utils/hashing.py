"""
Hashing utilities for QAFD-RAG.

Provides MD5-based hashing functions for content identification and caching.
"""

from hashlib import md5


def compute_args_hash(*args) -> str:
    """
    Compute MD5 hash from arbitrary arguments.

    Used for cache key generation from function arguments.

    Parameters:
    -----------
    *args : Any
        Arguments to hash

    Returns:
    --------
    str
        Hexadecimal MD5 hash string
    """
    return md5(str(args).encode()).hexdigest()


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute MD5 hash ID for content with optional prefix.

    Used for generating unique identifiers for text chunks and entities.

    Parameters:
    -----------
    content : str
        Content to hash
    prefix : str, optional
        Prefix to prepend to the hash

    Returns:
    --------
    str
        Prefixed hexadecimal MD5 hash string
    """
    return prefix + md5(content.encode()).hexdigest()


__all__ = [
    "compute_args_hash",
    "compute_mdhash_id",
]
