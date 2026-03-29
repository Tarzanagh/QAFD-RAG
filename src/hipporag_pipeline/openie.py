"""
OpenIE extraction (NER + triple extraction) using QAFD-RAG's LLM functions.

Follows HippoRAG's openie_openai.py logic but calls the async LLM wrappers
from ``QAFD-RAG/src/llm.py`` synchronously via ``asyncio.run``.
"""

import asyncio
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, TypedDict, Callable

from tqdm import tqdm

from .prompts import make_ner_messages, make_triple_messages
from .utils import (
    NerRawOutput,
    TripleRawOutput,
    fix_broken_generated_json,
    filter_invalid_triples,
)

logger = logging.getLogger(__name__)


class ChunkInfo(TypedDict):
    num_tokens: int
    content: str


def _run_sync(coro):
    """Run async coroutine from sync context."""
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


def _extract_ner_from_response(response_text: str) -> List[str]:
    pattern = r'\{[^{}]*"named_entities"\s*:\s*\[[^\]]*\][^{}]*\}'
    match = re.search(pattern, response_text, re.DOTALL)
    if match is None:
        return []
    try:
        return eval(match.group())["named_entities"]
    except Exception:
        return []


def _extract_triples_from_response(response_text: str) -> List[List[str]]:
    pattern = r'\{[^{}]*"triples"\s*:\s*\[[^\]]*\][^{}]*\}'
    match = re.search(pattern, response_text, re.DOTALL)
    if match is None:
        return []
    try:
        return eval(match.group())["triples"]
    except Exception:
        return []


class OpenIE:
    """Synchronous OpenIE using QAFD-RAG's async LLM function.

    Parameters
    ----------
    llm_func : callable
        An async function with the signature::

            async def llm_func(prompt, system_prompt=None,
                               history_messages=[], **kwargs) -> str

        Typically one of the ``gpt_*_complete`` helpers from ``src/llm.py``.
    """

    def __init__(self, llm_func: Callable):
        self.llm_func = llm_func

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a single LLM call."""
        system_prompt = None
        history = []
        user_prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "assistant":
                history.append(msg)
            elif msg["role"] == "user":
                # All user messages except the last go into history
                if user_prompt:
                    history.append({"role": "user", "content": user_prompt})
                user_prompt = msg["content"]

        return _run_sync(
            self.llm_func(
                prompt=user_prompt,
                system_prompt=system_prompt,
                history_messages=history,
                max_tokens=2048,
            )
        )

    # ------------------------------------------------------------------
    def ner(self, chunk_key: str, passage: str) -> NerRawOutput:
        messages = make_ner_messages(passage)
        raw_response = ""
        metadata: Dict[str, Any] = {}
        try:
            raw_response = self._call_llm(messages)
            real_response = fix_broken_generated_json(raw_response)
            extracted = _extract_ner_from_response(real_response)
            unique_entities = list(dict.fromkeys(extracted))
        except Exception as e:
            logger.warning(f"NER error for chunk {chunk_key}: {e}")
            metadata["error"] = str(e)
            return NerRawOutput(
                chunk_id=chunk_key,
                response=raw_response,
                unique_entities=[],
                metadata=metadata,
            )

        return NerRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            unique_entities=unique_entities,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    def triple_extraction(
        self, chunk_key: str, passage: str, named_entities: List[str]
    ) -> TripleRawOutput:
        messages = make_triple_messages(passage, named_entities)
        raw_response = ""
        metadata: Dict[str, Any] = {}
        try:
            raw_response = self._call_llm(messages)
            real_response = fix_broken_generated_json(raw_response)
            extracted = _extract_triples_from_response(real_response)
            triplets = filter_invalid_triples(triples=extracted)
        except Exception as e:
            logger.warning(f"Triple extraction error for chunk {chunk_key}: {e}")
            metadata["error"] = str(e)
            return TripleRawOutput(
                chunk_id=chunk_key,
                response=raw_response,
                metadata=metadata,
                triples=[],
            )

        return TripleRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            metadata=metadata,
            triples=triplets,
        )

    # ------------------------------------------------------------------
    def openie(self, chunk_key: str, passage: str) -> Dict[str, Any]:
        ner_output = self.ner(chunk_key=chunk_key, passage=passage)
        triple_output = self.triple_extraction(
            chunk_key=chunk_key,
            passage=passage,
            named_entities=ner_output.unique_entities,
        )
        return {"ner": ner_output, "triplets": triple_output}

    # ------------------------------------------------------------------
    def batch_openie(
        self, chunks: Dict[str, dict]
    ) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """Run NER + triple extraction over all chunks using multithreading.

        Parameters
        ----------
        chunks : dict
            Mapping ``chunk_hash_id -> {"content": text, ...}``.

        Returns
        -------
        (ner_dict, triple_dict)
        """
        chunk_passages = {k: v["content"] for k, v in chunks.items()}

        # ---- NER pass ----
        ner_results: List[NerRawOutput] = []

        with ThreadPoolExecutor() as executor:
            ner_futures = {
                executor.submit(self.ner, ckey, passage): ckey
                for ckey, passage in chunk_passages.items()
            }
            for future in tqdm(
                as_completed(ner_futures), total=len(ner_futures), desc="NER"
            ):
                ner_results.append(future.result())

        # ---- Triple extraction pass ----
        triple_results: List[TripleRawOutput] = []

        with ThreadPoolExecutor() as executor:
            re_futures = {
                executor.submit(
                    self.triple_extraction,
                    nr.chunk_id,
                    chunk_passages[nr.chunk_id],
                    nr.unique_entities,
                ): nr.chunk_id
                for nr in ner_results
            }
            for future in tqdm(
                as_completed(re_futures),
                total=len(re_futures),
                desc="Triple extraction",
            ):
                triple_results.append(future.result())

        ner_dict = {r.chunk_id: r for r in ner_results}
        triple_dict = {r.chunk_id: r for r in triple_results}
        return ner_dict, triple_dict
