"""
Parsing utilities for QAFD-RAG.

Provides string parsing, JSON extraction, and text cleaning functions.
"""

import html
import json
import logging
import re
from typing import Any, List, Optional, Union

logger = logging.getLogger("QAFD_RAG")


def locate_json_string_body_from_string(content: str) -> Optional[str]:
    """
    Extract JSON object from a string containing mixed content.

    Useful for parsing LLM responses that contain JSON within other text.

    Parameters:
    -----------
    content : str
        String potentially containing JSON

    Returns:
    --------
    Optional[str]
        Extracted JSON string, or None if not found
    """
    try:
        maybe_json_str = re.search(r"{.*}", content, re.DOTALL)
        if maybe_json_str is not None:
            maybe_json_str = maybe_json_str.group(0)
            maybe_json_str = maybe_json_str.replace("\\n", "")
            maybe_json_str = maybe_json_str.replace("\n", "")
            maybe_json_str = maybe_json_str.replace("'", '"')
            return maybe_json_str
    except Exception:
        pass
    return None


def convert_response_to_json(response: str) -> dict:
    """
    Convert an LLM response string to a JSON dictionary.

    Parameters:
    -----------
    response : str
        LLM response string containing JSON

    Returns:
    --------
    dict
        Parsed JSON dictionary

    Raises:
    -------
    AssertionError
        If no JSON structure is found
    json.JSONDecodeError
        If JSON parsing fails
    """
    json_str = locate_json_string_body_from_string(response)
    assert json_str is not None, f"Unable to parse JSON from response: {response}"
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {json_str}")
        raise e from None


def split_string_by_multi_markers(content: str, markers: List[str]) -> List[str]:
    """
    Split a string by multiple marker strings.

    Parameters:
    -----------
    content : str
        String to split
    markers : List[str]
        List of marker strings to split by

    Returns:
    --------
    List[str]
        List of non-empty stripped substrings
    """
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def clean_str(input: Any) -> str:
    """
    Clean a string by unescaping HTML and removing control characters.

    Parameters:
    -----------
    input : Any
        Input value (returns unchanged if not a string)

    Returns:
    --------
    str
        Cleaned string
    """
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


def is_float_regex(value: str) -> bool:
    """
    Check if a string represents a valid float number.

    Parameters:
    -----------
    value : str
        String to check

    Returns:
    --------
    bool
        True if string is a valid float representation
    """
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def safe_unicode_decode(content: bytes) -> str:
    """
    Safely decode bytes to string, handling unicode escape sequences.

    Parameters:
    -----------
    content : bytes
        Bytes content to decode

    Returns:
    --------
    str
        Decoded string with unicode escapes resolved
    """
    unicode_escape_pattern = re.compile(r"\\u([0-9a-fA-F]{4})")

    def replace_unicode_escape(match):
        return chr(int(match.group(1), 16))

    decoded_content = unicode_escape_pattern.sub(
        replace_unicode_escape, content.decode("utf-8")
    )
    return decoded_content


def pack_user_ass_to_openai_messages(*args: str) -> List[dict]:
    """
    Pack alternating user/assistant messages into OpenAI message format.

    Parameters:
    -----------
    *args : str
        Alternating user and assistant message contents

    Returns:
    --------
    List[dict]
        List of message dictionaries with role and content
    """
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


__all__ = [
    "locate_json_string_body_from_string",
    "convert_response_to_json",
    "split_string_by_multi_markers",
    "clean_str",
    "is_float_regex",
    "safe_unicode_decode",
    "pack_user_ass_to_openai_messages",
]
