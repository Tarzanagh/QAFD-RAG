"""
Embedding utilities for QAFD-RAG.

Provides embedding function wrappers and utilities.
"""

import asyncio
from dataclasses import dataclass

import numpy as np


class UnlimitedSemaphore:
    """A no-op semaphore that doesn't limit concurrency."""

    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc, tb):
        pass


@dataclass
class EmbeddingFunc:
    """
    Wrapper for embedding functions with rate limiting.

    Attributes:
    -----------
    embedding_dim : int
        Dimension of the embedding vectors
    max_token_size : int
        Maximum token size for input text
    func : callable
        The async embedding function to wrap
    concurrent_limit : int
        Maximum concurrent calls (0 for unlimited)
    """

    embedding_dim: int
    max_token_size: int
    func: callable
    concurrent_limit: int = 16

    def __post_init__(self):
        if self.concurrent_limit != 0:
            self._semaphore = asyncio.Semaphore(self.concurrent_limit)
        else:
            self._semaphore = UnlimitedSemaphore()

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        async with self._semaphore:
            return await self.func(*args, **kwargs)


def wrap_embedding_func_with_attrs(**kwargs):
    """
    Decorator to wrap an embedding function with EmbeddingFunc attributes.

    Parameters:
    -----------
    **kwargs
        Arguments passed to EmbeddingFunc (embedding_dim, max_token_size, etc.)

    Returns:
    --------
    EmbeddingFunc
        Wrapped embedding function
    """

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro


__all__ = [
    "UnlimitedSemaphore",
    "EmbeddingFunc",
    "wrap_embedding_func_with_attrs",
]
