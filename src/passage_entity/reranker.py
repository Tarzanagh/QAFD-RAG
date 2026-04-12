"""
Fact reranker following the original DSPy filter logic.

Uses QAFD-RAG's async LLM functions (wrapped synchronously) to call the
same prompt structure that the original DSPyFilter uses.
"""

import ast
import asyncio
import difflib
import json
import logging
import re
from copy import deepcopy
from typing import Callable, Dict, List, Tuple, Any

from .prompts import make_reranker_messages

logger = logging.getLogger(__name__)


def _run_sync(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


class FactReranker:
    """Rerank candidate fact triples using an LLM (DSPy-style filtering).

    Parameters
    ----------
    llm_func : callable
        Async LLM function from ``src/llm.py``.
    dspy_file_path : str or None
        Path to a DSPy-saved JSON file with custom demos/system prompt.
        If ``None``, uses built-in demos from ``prompts.py``.
    """

    def __init__(self, llm_func: Callable, dspy_file_path: str = None):
        self.llm_func = llm_func
        self.dspy_file_path = dspy_file_path

        if dspy_file_path is not None:
            self._custom_template = self._load_dspy_template(dspy_file_path)
        else:
            self._custom_template = None

    # ------------------------------------------------------------------
    @staticmethod
    def _load_dspy_template(path: str) -> List[Dict[str, str]]:
        """Load a DSPy-saved JSON and convert to chat messages."""
        data = json.load(open(path))
        system_prompt = data["prog"]["system"]
        demos = data["prog"]["demos"]

        one_in = (
            "[[ ## question ## ]]\n{question}\n\n"
            "[[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\n"
            "Respond with the corresponding output fields, starting with the field "
            "`[[ ## fact_after_filter ## ]]` (must be formatted as a valid Python Fact), "
            "and then ending with the marker for `[[ ## completed ## ]]`."
        )
        one_out = (
            "[[ ## fact_after_filter ## ]]\n{fact_after_filter}\n\n"
            "[[ ## completed ## ]]"
        )

        msgs = [{"role": "system", "content": system_prompt}]
        for demo in demos:
            msgs.append({
                "role": "user",
                "content": one_in.format(
                    question=demo["question"],
                    fact_before_filter=demo["fact_before_filter"],
                ),
            })
            if "fact_after_filter" in demo:
                msgs.append({
                    "role": "assistant",
                    "content": one_out.format(
                        fact_after_filter=demo["fact_after_filter"],
                    ),
                })
        return msgs

    # ------------------------------------------------------------------
    def _build_messages(
        self, question: str, fact_before_filter_json: str
    ) -> List[Dict[str, str]]:
        if self._custom_template is not None:
            msgs = deepcopy(self._custom_template)
            one_in = (
                "[[ ## question ## ]]\n{question}\n\n"
                "[[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\n"
                "Respond with the corresponding output fields, starting with the field "
                "`[[ ## fact_after_filter ## ]]` (must be formatted as a valid Python Fact), "
                "and then ending with the marker for `[[ ## completed ## ]]`."
            )
            msgs.append({
                "role": "user",
                "content": one_in.format(
                    question=question,
                    fact_before_filter=fact_before_filter_json,
                ),
            })
            return msgs
        else:
            return make_reranker_messages(question, fact_before_filter_json)

    # ------------------------------------------------------------------
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        # Pass messages directly to OpenAI — the old decompose/recompose
        # loop scrambled demo order (assistant before user in each pair).
        return _run_sync(
            self.llm_func(
                prompt=messages[-1]["content"],
                system_prompt=messages[0]["content"] if messages[0]["role"] == "system" else None,
                history_messages=messages[1:-1],
                max_tokens=512,
            )
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_filter(response: str) -> List[List[str]]:
        """Extract fact_after_filter from the DSPy-style response."""
        sections = [(None, [])]
        header_re = re.compile(r'\[\[ ## (\w+) ## \]\]')
        for line in response.splitlines():
            m = header_re.match(line.strip())
            if m:
                sections.append((m.group(1), []))
            else:
                sections[-1][1].append(line)
        sections = [(k, "\n".join(v).strip()) for k, v in sections]

        parsed: List[List[str]] = []
        for k, value in sections:
            if k == "fact_after_filter":
                try:
                    try:
                        pv = json.loads(value)
                    except json.JSONDecodeError:
                        try:
                            pv = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            pv = value
                    if isinstance(pv, dict) and "fact" in pv:
                        parsed = pv["fact"]
                except Exception as e:
                    logger.warning(f"Error parsing reranker output: {e}")
        return parsed

    # ------------------------------------------------------------------
    def rerank(
        self,
        query: str,
        candidate_items: List[Tuple],
        candidate_indices: List[int],
        len_after_rerank: int = None,
    ) -> Tuple[List[int], List[Tuple], dict]:
        """Rerank candidate facts by LLM-based filtering.

        Returns
        -------
        (sorted_indices, sorted_facts, metadata_dict)
        """
        fact_json = json.dumps({"fact": [list(c) for c in candidate_items]})
        try:
            msgs = self._build_messages(query, fact_json)
            response = self._call_llm(msgs)
            generated_facts = self._parse_filter(response)
        except Exception as e:
            logger.warning(f"Reranker exception: {e}")
            generated_facts = []

        result_indices = []
        for gf in generated_facts:
            matches = difflib.get_close_matches(
                str(gf), [str(i) for i in candidate_items], n=1, cutoff=0.0
            )
            if matches:
                try:
                    result_indices.append(candidate_items.index(eval(matches[0])))
                except Exception as e:
                    logger.warning(f"Index matching error: {e}")

        sorted_indices = [candidate_indices[i] for i in result_indices]
        sorted_items = [candidate_items[i] for i in result_indices]

        if len_after_rerank is not None:
            sorted_indices = sorted_indices[:len_after_rerank]
            sorted_items = sorted_items[:len_after_rerank]

        return sorted_indices, sorted_items, {"confidence": None}

    def __call__(self, *args, **kwargs):
        return self.rerank(*args, **kwargs)
