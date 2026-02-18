"""
Data handling utilities for QAFD-RAG.

Provides CSV conversion and context combination functions.
"""

import csv
import io
from typing import List


def list_of_list_to_csv(data: List[List[str]]) -> str:
    """
    Convert a list of lists to a CSV string.

    Parameters:
    -----------
    data : List[List[str]]
        2D list of string values

    Returns:
    --------
    str
        CSV formatted string
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(data)
    return output.getvalue()


def csv_string_to_list(csv_string: str) -> List[List[str]]:
    """
    Convert a CSV string to a list of lists.

    Parameters:
    -----------
    csv_string : str
        CSV formatted string

    Returns:
    --------
    List[List[str]]
        2D list of string values
    """
    output = io.StringIO(csv_string)
    reader = csv.reader(output)
    return [row for row in reader]


def process_combine_contexts(hl: str, ll: str) -> str:
    """
    Combine high-level and low-level context CSV strings.

    Merges two CSV context strings, deduplicates entries, and re-indexes.

    Parameters:
    -----------
    hl : str
        High-level context CSV string
    ll : str
        Low-level context CSV string

    Returns:
    --------
    str
        Combined and deduplicated context string
    """
    header = None
    list_hl = csv_string_to_list(hl.strip())
    list_ll = csv_string_to_list(ll.strip())

    if list_hl:
        header = list_hl[0]
        list_hl = list_hl[1:]
    if list_ll:
        header = list_ll[0]
        list_ll = list_ll[1:]
    if header is None:
        return ""

    if list_hl:
        list_hl = [",".join(item[1:]) for item in list_hl if item]
    if list_ll:
        list_ll = [",".join(item[1:]) for item in list_ll if item]

    combined_sources = []
    seen = set()

    for item in list_hl + list_ll:
        if item and item not in seen:
            combined_sources.append(item)
            seen.add(item)

    combined_sources_result = [",\t".join(header)]

    for i, item in enumerate(combined_sources, start=1):
        combined_sources_result.append(f"{i},\t{item}")

    combined_sources_result = "\n".join(combined_sources_result)

    return combined_sources_result


__all__ = [
    "list_of_list_to_csv",
    "csv_string_to_list",
    "process_combine_contexts",
]
