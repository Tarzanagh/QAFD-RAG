"""
Tokenization utilities for QAFD-RAG.

Provides tiktoken-based encoding/decoding and token-aware list truncation.
"""

from typing import Callable, List, TypeVar

import tiktoken

# Global encoder instance (lazy initialized)
_ENCODER = None


def _get_encoder(model_name: str = "gpt-4o-mini"):
    """Get or initialize the tiktoken encoder."""
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.encoding_for_model(model_name)
    return _ENCODER


def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o-mini") -> List[int]:
    """
    Encode a string into tokens using tiktoken.

    Parameters:
    -----------
    content : str
        Text content to encode
    model_name : str, optional
        Model name for tokenizer selection (default: gpt-4o-mini)

    Returns:
    --------
    List[int]
        List of token IDs
    """
    encoder = _get_encoder(model_name)
    return encoder.encode(content)


def decode_tokens_by_tiktoken(tokens: List[int], model_name: str = "gpt-4o-mini") -> str:
    """
    Decode tokens back to a string using tiktoken.

    Parameters:
    -----------
    tokens : List[int]
        List of token IDs
    model_name : str, optional
        Model name for tokenizer selection (default: gpt-4o-mini)

    Returns:
    --------
    str
        Decoded text content
    """
    encoder = _get_encoder(model_name)
    return encoder.decode(tokens)


T = TypeVar('T')


def truncate_list_by_token_size(
    list_data: List[T],
    key: Callable[[T], str],
    max_token_size: int
) -> List[T]:
    """
    Truncate a list based on cumulative token count.

    Iterates through the list and includes items until the total token
    count exceeds max_token_size.

    Parameters:
    -----------
    list_data : List[T]
        List of items to truncate
    key : Callable[[T], str]
        Function to extract text content from each item
    max_token_size : int
        Maximum total tokens allowed

    Returns:
    --------
    List[T]
        Truncated list that fits within token limit
    """
    if max_token_size <= 0:
        return []

    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


__all__ = [
    "encode_string_by_tiktoken",
    "decode_tokens_by_tiktoken",
    "truncate_list_by_token_size",
]
