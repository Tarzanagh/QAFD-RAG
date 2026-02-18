"""
Logging utilities for QAFD-RAG.

Provides centralized logging configuration.
"""

import logging

logger = logging.getLogger("QAFD_RAG")


def set_logger(log_file: str):
    """
    Configure the QAFD_RAG logger.

    Parameters:
    -----------
    log_file : str
        Path to log file (currently unused - logs go to console only)
    """
    logger.setLevel(logging.INFO)
    # File logging disabled - logs go to console only


__all__ = [
    "logger",
    "set_logger",
]
