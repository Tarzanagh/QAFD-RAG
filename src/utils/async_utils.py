"""
Async utilities for QAFD-RAG.

Provides async function decorators and helpers.
"""

import asyncio
from functools import wraps


def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    """
    Decorator to limit concurrent async function calls.

    Parameters:
    -----------
    max_size : int
        Maximum number of concurrent calls
    waitting_time : float
        Time to wait (in seconds) between checks when at limit

    Returns:
    --------
    callable
        Decorated function with concurrency limiting
    """

    def final_decro(func):
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return final_decro


__all__ = [
    "limit_async_func_call",
]
