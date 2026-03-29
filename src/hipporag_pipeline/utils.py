"""
Utility functions and data classes for the HippoRAG-style KG pipeline.

Adapted from HippoRAG's misc_utils.py and llm_utils.py.
"""

import json
import re
import logging
from dataclasses import dataclass
from hashlib import md5
from typing import Dict, Any, List, Tuple, Literal, Union, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NerRawOutput:
    chunk_id: str
    response: str
    unique_entities: List[str]
    metadata: Dict[str, Any]


@dataclass
class TripleRawOutput:
    chunk_id: str
    response: str
    triples: List[List[str]]
    metadata: Dict[str, Any]


@dataclass
class QuerySolution:
    question: str
    docs: List[str]
    doc_scores: np.ndarray = None
    answer: str = None
    gold_answers: List[str] = None
    gold_docs: Optional[List[str]] = None

    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "gold_answers": self.gold_answers,
            "docs": self.docs[:5],
            "doc_scores": (
                [round(v, 4) for v in self.doc_scores.tolist()[:5]]
                if self.doc_scores is not None
                else None
            ),
            "gold_docs": self.gold_docs,
        }


Triple = Union[List[str], Tuple[str, str, str]]

# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Compute the MD5 hash of *content* and optionally prepend *prefix*."""
    return prefix + md5(content.encode()).hexdigest()

# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

def text_processing(text):
    """Lower-case, strip non-alphanumeric characters (except spaces)."""
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    return re.sub('[^A-Za-z0-9 ]', ' ', text.lower()).strip()

# ---------------------------------------------------------------------------
# OpenIE helpers
# ---------------------------------------------------------------------------

def extract_entity_nodes(chunk_triples: List[List[Triple]]) -> Tuple[List[str], List[List[str]]]:
    """Extract unique entity nodes from chunk triples.

    Returns:
        graph_nodes: globally unique list of entity strings.
        chunk_triple_entities: per-chunk list of entity strings.
    """
    chunk_triple_entities = []
    for triples in chunk_triples:
        triple_entities = set()
        for t in triples:
            if len(t) == 3:
                triple_entities.update([t[0], t[2]])
            else:
                logger.warning(f"Invalid triple during graph construction: {t}")
        chunk_triple_entities.append(list(triple_entities))
    graph_nodes = list(np.unique([ent for ents in chunk_triple_entities for ent in ents]))
    return graph_nodes, chunk_triple_entities


def flatten_facts(chunk_triples: List[List[Triple]]) -> List[Tuple]:
    """Flatten per-chunk triples into a unique list of tuples."""
    graph_triples = []
    for triples in chunk_triples:
        graph_triples.extend([tuple(t) for t in triples])
    return list(set(graph_triples))


def reformat_openie_results(corpus_openie_results):
    """Convert saved openie JSON list into (ner_dict, triple_dict)."""
    ner_output_dict = {
        chunk_item['idx']: NerRawOutput(
            chunk_id=chunk_item['idx'],
            response=None,
            metadata={},
            unique_entities=list(np.unique(chunk_item['extracted_entities']))
        )
        for chunk_item in corpus_openie_results
    }
    triple_output_dict = {
        chunk_item['idx']: TripleRawOutput(
            chunk_id=chunk_item['idx'],
            response=None,
            metadata={},
            triples=filter_invalid_triples(triples=chunk_item['extracted_triples'])
        )
        for chunk_item in corpus_openie_results
    }
    return ner_output_dict, triple_output_dict

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def min_max_normalize(x: np.ndarray) -> np.ndarray:
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    if range_val == 0:
        return np.ones_like(x)
    return (x - min_val) / range_val

# ---------------------------------------------------------------------------
# JSON repair helpers (from HippoRAG llm_utils)
# ---------------------------------------------------------------------------

def fix_broken_generated_json(json_str: str) -> str:
    """Attempt to fix truncated JSON by closing open brackets/braces."""
    def find_unclosed(s):
        unclosed = []
        inside_string = False
        escape_next = False
        for char in s:
            if inside_string:
                if escape_next:
                    escape_next = False
                elif char == '\\':
                    escape_next = True
                elif char == '"':
                    inside_string = False
            else:
                if char == '"':
                    inside_string = True
                elif char in '{[':
                    unclosed.append(char)
                elif char in '}]':
                    if unclosed and (
                        (char == '}' and unclosed[-1] == '{') or
                        (char == ']' and unclosed[-1] == '[')
                    ):
                        unclosed.pop()
        return unclosed

    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        pass

    last_comma_index = json_str.rfind(',')
    if last_comma_index != -1:
        json_str = json_str[:last_comma_index]

    unclosed = find_unclosed(json_str)
    closing_map = {'{': '}', '[': ']'}
    for open_char in reversed(unclosed):
        json_str += closing_map[open_char]
    return json_str


def filter_invalid_triples(triples: List[List[str]]) -> List[List[str]]:
    """Keep only unique triples with exactly 3 elements."""
    unique_triples = set()
    valid_triples = []
    for triple in triples:
        if len(triple) != 3:
            continue
        valid_triple = [str(item) for item in triple]
        key = tuple(valid_triple)
        if key not in unique_triples:
            unique_triples.add(key)
            valid_triples.append(valid_triple)
    return valid_triples


def string_to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError(f"Cannot convert {v!r} to bool")
