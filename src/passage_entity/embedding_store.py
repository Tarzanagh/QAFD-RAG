"""
Parquet-backed embedding store, adapted from the original EmbeddingStore.

Uses QAFD-RAG's async embedding functions (wrapped synchronously) so we
can share models / GPU memory with the rest of the QAFD-RAG system.
"""

import asyncio
import logging
import os
from copy import deepcopy
from typing import List, Dict, Optional, Callable, Any

import numpy as np
import pandas as pd

from .utils import compute_mdhash_id

logger = logging.getLogger(__name__)


class EmbeddingModelWrapper:
    """Thin sync wrapper around a QAFD-RAG *async* embedding function.

    The wrapped function must have the signature::

        async def embed(texts: list[str], **kwargs) -> np.ndarray

    Parameters
    ----------
    embed_func : callable
        An async embedding function from ``QAFD-RAG/src/llm.py``.
    batch_size : int
        Max texts per call.
    """

    def __init__(self, embed_func: Callable, batch_size: int = 16):
        self._embed_func = embed_func
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    def batch_encode(self, texts, instruction: str = None, norm: bool = True) -> np.ndarray:
        """Synchronously encode *texts* into embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        all_embeddings = []
        # Use larger batch for API-based embeddings (OpenAI supports up to 2048)
        effective_batch = max(self.batch_size, 512)
        for start in range(0, len(texts), effective_batch):
            batch = texts[start : start + effective_batch]
            if instruction:
                batch = [f"{instruction} {t}" for t in batch]
            embs = self._run_async(self._embed_func(batch))
            if not isinstance(embs, np.ndarray):
                embs = np.array(embs)
            if norm:
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                embs = embs / norms
            all_embeddings.append(embs)
        return np.vstack(all_embeddings)

    # ------------------------------------------------------------------
    @staticmethod
    def _run_async(coro):
        """Run an async coroutine synchronously."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # We are inside an already-running event loop (e.g. Jupyter).
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return asyncio.run(coro)


class EmbeddingStore:
    """Parquet-backed vector store.

    Mirrors the original EmbeddingStore but uses ``EmbeddingModelWrapper``
    (which calls QAFD-RAG's async embedding functions under the hood).
    """

    def __init__(
        self,
        embedding_model: EmbeddingModelWrapper,
        db_filename: str,
        batch_size: int,
        namespace: str,
    ):
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = namespace

        if not os.path.exists(db_filename):
            logger.info(f"Creating directory: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)

        self.filename = os.path.join(db_filename, f"vdb_{self.namespace}.parquet")
        self._load_data()

    # ------------------------------------------------------------------
    # Data persistence
    # ------------------------------------------------------------------

    def _load_data(self):
        if os.path.exists(self.filename):
            df = pd.read_parquet(self.filename)
            self.hash_ids = df["hash_id"].values.tolist()
            self.texts = df["content"].values.tolist()
            self.embeddings = df["embedding"].values.tolist()
            self._rebuild_indices()
            assert len(self.hash_ids) == len(self.texts) == len(self.embeddings)
            logger.info(f"Loaded {len(self.hash_ids)} records from {self.filename}")
        else:
            self.hash_ids, self.texts, self.embeddings = [], [], []
            self.hash_id_to_idx: Dict[str, int] = {}
            self.hash_id_to_row: Dict[str, dict] = {}
            self.hash_id_to_text: Dict[str, str] = {}
            self.text_to_hash_id: Dict[str, str] = {}

    def _rebuild_indices(self):
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_row = {
            h: {"hash_id": h, "content": t} for h, t in zip(self.hash_ids, self.texts)
        }
        self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}

    def _save_data(self):
        data = pd.DataFrame({
            "hash_id": self.hash_ids,
            "content": self.texts,
            "embedding": self.embeddings,
        })
        data.to_parquet(self.filename, index=False)
        self._rebuild_indices()
        logger.info(f"Saved {len(self.hash_ids)} records to {self.filename}")

    def _upsert(self, hash_ids, texts, embeddings):
        self.embeddings.extend(embeddings)
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)
        self._save_data()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_missing_string_hash_ids(self, texts: List[str]) -> Dict[str, dict]:
        nodes_dict = {}
        for text in texts:
            hid = compute_mdhash_id(text, prefix=self.namespace + "-")
            nodes_dict[hid] = {"content": text}

        if not nodes_dict:
            return {}

        existing = set(self.hash_id_to_row.keys())
        missing = {h: {"hash_id": h, "content": v["content"]}
                   for h, v in nodes_dict.items() if h not in existing}
        return missing

    def insert_strings(self, texts: List[str]):
        nodes_dict = {}
        for text in texts:
            if not text or not text.strip():
                continue
            hid = compute_mdhash_id(text, prefix=self.namespace + "-")
            nodes_dict[hid] = {"content": text}

        all_ids = list(nodes_dict.keys())
        if not all_ids:
            return

        existing = set(self.hash_id_to_row.keys())
        missing_ids = [h for h in all_ids if h not in existing]

        logger.info(
            f"Inserting {len(missing_ids)} new records, "
            f"{len(all_ids) - len(missing_ids)} already exist."
        )
        if not missing_ids:
            return

        texts_to_encode = [nodes_dict[h]["content"] for h in missing_ids]
        missing_embeddings = self.embedding_model.batch_encode(texts_to_encode)
        # Convert ndarray rows to list of lists for parquet storage
        if isinstance(missing_embeddings, np.ndarray):
            missing_embeddings = missing_embeddings.tolist()
        self._upsert(missing_ids, texts_to_encode, missing_embeddings)

    def delete(self, hash_ids):
        indices = sorted(
            [self.hash_id_to_idx[h] for h in hash_ids], reverse=True
        )
        for idx in indices:
            self.hash_ids.pop(idx)
            self.texts.pop(idx)
            self.embeddings.pop(idx)
        self._save_data()

    # Lookups
    def get_row(self, hash_id: str) -> dict:
        return self.hash_id_to_row[hash_id]

    def get_hash_id(self, text: str) -> str:
        return self.text_to_hash_id[text]

    def get_rows(self, hash_ids: List[str], dtype=np.float32) -> Dict[str, dict]:
        if not hash_ids:
            return {}
        return {hid: self.hash_id_to_row[hid] for hid in hash_ids}

    def get_all_ids(self) -> List[str]:
        return deepcopy(self.hash_ids)

    def get_all_id_to_rows(self) -> Dict[str, dict]:
        return deepcopy(self.hash_id_to_row)

    def get_all_texts(self) -> set:
        return set(row["content"] for row in self.hash_id_to_row.values())

    def get_embedding(self, hash_id: str, dtype=np.float32) -> np.ndarray:
        return np.array(self.embeddings[self.hash_id_to_idx[hash_id]], dtype=dtype)

    def get_embeddings(self, hash_ids: List[str], dtype=np.float32) -> np.ndarray:
        if not hash_ids:
            return np.array([])
        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        all_embs = np.array(self.embeddings, dtype=dtype)
        return all_embs[indices]
