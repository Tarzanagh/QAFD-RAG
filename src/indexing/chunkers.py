"""Text chunking strategies.

This module provides various text chunking strategies for splitting
documents into smaller pieces suitable for processing and embedding.
"""

from typing import List, Dict, Any

from ..utils import encode_string_by_tiktoken, decode_tokens_by_tiktoken
from .base import BaseChunker, ChunkResult


class TokenChunker(BaseChunker):
    """Token-based text chunker using tiktoken.

    This chunker splits text into overlapping chunks based on token count,
    which is useful for processing with models that have token limits.

    Attributes:
        max_token_size: Maximum number of tokens per chunk
        overlap_token_size: Number of overlapping tokens between chunks
        tiktoken_model: Model name for tiktoken encoding
    """

    def __init__(
        self,
        max_token_size: int = 1024,
        overlap_token_size: int = 128,
        tiktoken_model: str = "gpt-4o"
    ):
        """Initialize the TokenChunker.

        Args:
            max_token_size: Maximum tokens per chunk (default: 1024)
            overlap_token_size: Overlapping tokens between chunks (default: 128)
            tiktoken_model: Model name for tokenization (default: "gpt-4o")
        """
        self.max_token_size = max_token_size
        self.overlap_token_size = overlap_token_size
        self.tiktoken_model = tiktoken_model

    def chunk(self, content: str, **kwargs) -> ChunkResult:
        """Split content into overlapping token-based chunks.

        Args:
            content: Text content to split
            **kwargs: Override parameters:
                - max_token_size: Override max tokens per chunk
                - overlap_token_size: Override overlap size
                - tiktoken_model: Override tokenizer model

        Returns:
            ChunkResult containing chunks with metadata
        """
        max_size = kwargs.get('max_token_size', self.max_token_size)
        overlap = kwargs.get('overlap_token_size', self.overlap_token_size)
        model = kwargs.get('tiktoken_model', self.tiktoken_model)

        tokens = encode_string_by_tiktoken(content, model_name=model)
        chunks = []

        step_size = max_size - overlap
        if step_size <= 0:
            step_size = max_size  # Fallback if overlap >= max_size

        for index, start in enumerate(range(0, len(tokens), step_size)):
            chunk_tokens = tokens[start:start + max_size]
            chunk_content = decode_tokens_by_tiktoken(chunk_tokens, model_name=model)
            chunks.append({
                "tokens": len(chunk_tokens),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            })

        return ChunkResult(chunks=chunks, total_tokens=len(tokens))

    def __repr__(self) -> str:
        return (
            f"TokenChunker(max_token_size={self.max_token_size}, "
            f"overlap_token_size={self.overlap_token_size}, "
            f"tiktoken_model='{self.tiktoken_model}')"
        )


# =============================================================================
# Backward Compatibility Functions
# =============================================================================

def chunking_by_token_size(
    content: str,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
    tiktoken_model: str = "gpt-4o"
) -> List[Dict[str, Any]]:
    """Split content into token-based chunks.

    This is a legacy function for backward compatibility. New code should
    use the TokenChunker class directly.

    Args:
        content: Text content to split
        overlap_token_size: Number of overlapping tokens (default: 128)
        max_token_size: Maximum tokens per chunk (default: 1024)
        tiktoken_model: Model name for tokenization (default: "gpt-4o")

    Returns:
        List of chunk dictionaries with keys:
        - tokens: Number of tokens in this chunk
        - content: The chunk text content
        - chunk_order_index: Sequential index of this chunk

    Example:
        >>> chunks = chunking_by_token_size("Long text here...", max_token_size=512)
        >>> for chunk in chunks:
        ...     print(f"Chunk {chunk['chunk_order_index']}: {chunk['tokens']} tokens")
    """
    chunker = TokenChunker(
        max_token_size=max_token_size,
        overlap_token_size=overlap_token_size,
        tiktoken_model=tiktoken_model
    )
    result = chunker.chunk(content)
    return result.chunks
