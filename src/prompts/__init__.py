"""
Prompt registry for QAFD-RAG.

Usage:
    # Default (QA mode) — backward compatible:
    from src.prompts import PROMPTS, GRAPH_FIELD_SEP

    # Task-specific:
    from src.prompts import get_prompts
    prompts = get_prompts("text2sql")   # SQL-oriented prompts
    prompts = get_prompts("qa")         # Standard QA prompts (default)
"""

from .common import GRAPH_FIELD_SEP, COMMON_PROMPTS
from .prompts_qa import QA_PROMPTS as _QA_PROMPTS
from .prompts_text2sql import TEXT2SQL_PROMPTS as _TEXT2SQL_PROMPTS

# Default PROMPTS for backward compatibility (QA mode)
PROMPTS = {**COMMON_PROMPTS, **_QA_PROMPTS}


def get_prompts(task: str = "qa") -> dict:
    """
    Get merged prompt dictionary for a specific task.

    Args:
        task: "qa" for standard document QA, "text2sql" for SQL generation

    Returns:
        Merged dict of common + task-specific prompts
    """
    if task == "text2sql":
        return {**COMMON_PROMPTS, **_TEXT2SQL_PROMPTS}
    return {**COMMON_PROMPTS, **_QA_PROMPTS}
