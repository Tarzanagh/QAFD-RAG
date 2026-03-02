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


def extract_all_blocks(content: str, code_format: str = None) -> List[str]:
    """
    Extract code/text blocks from an LLM response.

    Tries three strategies in order:
      1. <Answer>...</Answer> XML tags
      2. <sql>...</sql> XML tags
      3. Markdown code fences (```sql ... ```)

    Ported from CoFD-M/methods/CoFD/utils.py.
    """
    blocks = []

    possible_patterns = [
        ("<Answer>", "</Answer>"),
        ("<Answer>", "<Answer>"),
        ("<answer>", "</answer>"),
    ]

    if code_format == "json":
        for start_tag, end_tag in possible_patterns:
            if start_tag in content and end_tag in content:
                extracted = content.split(start_tag)[1].split(end_tag)[0].strip()
                blocks.append(extracted)
        return blocks

    if code_format == "text":
        if "<Answer>" in content and "</Answer>" in content:
            extracted = content.split("<Answer>")[1].split("</Answer>")[0].strip()
            blocks.append(extracted)
        return blocks

    for start_tag, end_tag in possible_patterns:
        if start_tag in content and end_tag in content:
            try:
                extracted = content.split(start_tag)[1].split(end_tag)[0].strip()
                blocks.append(extracted)
                return blocks
            except Exception:
                continue

    if code_format == "sql" and not blocks:
        xml_matches = re.findall(
            r'<sql>\s*(.*?)\s*</sql>', content, re.DOTALL | re.IGNORECASE
        )
        if xml_matches:
            blocks.extend([m.strip() for m in xml_matches if m.strip()])
            if blocks:
                return blocks

    start = 0
    while True:
        search_pattern = f"```{code_format}" if code_format else "```"
        block_start = content.find(search_pattern, start)
        if block_start == -1:
            break
        block_end = content.find("```", block_start + len(search_pattern))
        if block_end == -1:
            break
        block = content[block_start + len(search_pattern):block_end].strip()
        blocks.append(block)
        start = block_end + len("```")

    return blocks


def extract_schema_from_context(schema_context: str) -> dict:
    """
    Extract tables and columns from QAFD-RAG text2sql schema_context output.

    Parses the CSV entity blocks to find entities with entity_type 'column'
    or 'complete_table', deduplicates across local/global sections.

    Returns:
        dict with 'tables' (list of str) and 'columns' (list of 'table.column' str)
    """
    import csv
    from io import StringIO

    tables = set()
    columns = set()

    # Extract all CSV blocks from the context
    csv_blocks = extract_all_blocks(schema_context, "csv")

    for block in csv_blocks:
        # Skip non-entity blocks (relationship summaries, sources, etc.)
        if "entity_type" not in block:
            continue

        reader = csv.DictReader(StringIO(block))
        for row in reader:
            entity_type = row.get("entity_type", "").strip()
            entity = row.get("entity", "").strip().strip('"')

            if entity_type == "complete_table":
                tables.add(entity)
            elif entity_type == "column":
                columns.add(entity)

    return {
        "tables": sorted(tables),
        "columns": sorted(columns),
    }


def extract_schema_from_clusters(clusters: list) -> dict:
    """
    Extract tables and columns from raw QAFD-RAG cluster data.

    Each cluster has an 'entities' (or 'nodes') list with entity_type and entity fields.

    Returns:
        dict with 'tables' (list of str) and 'columns' (list of 'table.column' str)
    """
    tables = set()
    columns = set()

    for cluster in clusters:
        entities = cluster.get("entities", cluster.get("nodes", []))
        for entity in entities:
            entity_type = entity.get("entity_type", entity.get("type", "")).strip()
            entity_name = entity.get("entity", "").strip().strip('"')

            if entity_type == "complete_table":
                tables.add(entity_name)
            elif entity_type == "column" and "." in entity_name:
                columns.add(entity_name)

    return {
        "tables": sorted(tables),
        "columns": sorted(columns),
    }


def extract_schema_from_create_table(create_table_text: str) -> dict:
    """
    Extract tables and columns from CREATE TABLE statements.

    Parses:
        CREATE TABLE `table_name` (
          `col1` TYPE ...,
          `col2` TYPE ...
        );

    Returns:
        dict with 'tables' (list of str) and 'columns' (list of 'table.column' str)
    """
    tables = set()
    columns = set()

    current_table = None
    for line in create_table_text.splitlines():
        stripped = line.strip()

        # Match CREATE TABLE `name` or CREATE TABLE name
        table_match = re.match(r'CREATE\s+TABLE\s+`?(\w+)`?\s*\(', stripped, re.IGNORECASE)
        if table_match:
            current_table = table_match.group(1)
            tables.add(current_table)
            continue

        # Match column definition: `col_name` TYPE ...
        if current_table and stripped.startswith('`'):
            col_match = re.match(r'`(\w+)`\s+\w+', stripped)
            if col_match:
                col_name = col_match.group(1)
                columns.add(f"{current_table}.{col_name}")

        # End of CREATE TABLE
        if stripped.startswith(');'):
            current_table = None

    return {
        "tables": sorted(tables),
        "columns": sorted(columns),
    }


__all__ = [
    "locate_json_string_body_from_string",
    "convert_response_to_json",
    "split_string_by_multi_markers",
    "clean_str",
    "is_float_regex",
    "safe_unicode_decode",
    "pack_user_ass_to_openai_messages",
    "extract_all_blocks",
    "extract_schema_from_context",
    "extract_schema_from_clusters",
    "extract_schema_from_create_table",
]
