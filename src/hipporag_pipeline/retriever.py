"""
Full retrieval pipeline following HippoRAG's retrieve + graph_search_with_fact_entities.

Steps:
    1. Encode query for fact matching and passage matching.
    2. Score facts, rerank with LLM.
    3. Compute entity seed weights and passage weights.
    4. Run MultiSeedFlowDiffusionRetriever (QAFD) on the KG.
    5. Return ranked passages.
"""

import json
import logging
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import igraph as ig
import numpy as np
from tqdm import tqdm

from .config import HippoRAGConfig
from .embedding_store import EmbeddingStore, EmbeddingModelWrapper
from .graph_adapter import run_igraph_qafd
from .prompts import get_query_instruction
from .reranker import FactReranker
from .utils import (
    QuerySolution,
    NerRawOutput,
    TripleRawOutput,
    compute_mdhash_id,
    text_processing,
    extract_entity_nodes,
    flatten_facts,
    min_max_normalize,
    reformat_openie_results,
)

logger = logging.getLogger(__name__)


class HippoRAGRetriever:
    """End-to-end retriever:  query -> ranked passages.

    Uses the pre-built KG (igraph), embedding stores, and QAFD flow diffusion.
    """

    def __init__(
        self,
        config: HippoRAGConfig,
        embedding_model: EmbeddingModelWrapper,
        reranker: FactReranker,
        graph: ig.Graph,
        chunk_embedding_store: EmbeddingStore,
        entity_embedding_store: EmbeddingStore,
        fact_embedding_store: EmbeddingStore,
        openie_results_path: str,
    ):
        self.config = config
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.graph = graph

        self.chunk_store = chunk_embedding_store
        self.entity_store = entity_embedding_store
        self.fact_store = fact_embedding_store
        self.openie_results_path = openie_results_path

        # Filled by prepare()
        self._ready = False
        self.entity_node_keys: List[str] = []
        self.passage_node_keys: List[str] = []
        self.fact_node_keys: List[str] = []
        self.entity_embeddings: np.ndarray = np.array([])
        self.passage_embeddings: np.ndarray = np.array([])
        self.fact_embeddings: np.ndarray = np.array([])
        self.node_name_to_vertex_idx: Dict[str, int] = {}
        self.entity_node_idxs: List[int] = []
        self.passage_node_idxs: List[int] = []
        self.ent_node_to_chunk_ids: Dict[str, set] = {}

        # Cached query embeddings
        self._query_emb_fact: Dict[str, np.ndarray] = {}
        self._query_emb_pass: Dict[str, np.ndarray] = {}

        # Timing accumulators
        self.rerank_time = 0.0
        self.qafd_time = 0.0
        self.total_time = 0.0

    # ------------------------------------------------------------------
    # Preparation (mirrors HippoRAG prepare_retrieval_objects)
    # ------------------------------------------------------------------

    def prepare(self):
        """Load embeddings, build lookup structures. Call once before retrieve()."""
        logger.info("Preparing retrieval objects ...")

        self.entity_node_keys = list(self.entity_store.get_all_ids())
        self.passage_node_keys = list(self.chunk_store.get_all_ids())
        self.fact_node_keys = list(self.fact_store.get_all_ids())

        # Node index mapping
        try:
            name_to_idx = {v["name"]: idx for idx, v in enumerate(self.graph.vs)}
            self.node_name_to_vertex_idx = name_to_idx
            self.entity_node_idxs = [name_to_idx[k] for k in self.entity_node_keys]
            self.passage_node_idxs = [name_to_idx[k] for k in self.passage_node_keys]
        except Exception as e:
            logger.error(f"Graph index mapping failed: {e}")
            self.node_name_to_vertex_idx = {}
            self.entity_node_idxs = []
            self.passage_node_idxs = []

        # Embeddings
        self.entity_embeddings = np.array(
            self.entity_store.get_embeddings(self.entity_node_keys)
        ) if self.entity_node_keys else np.array([])

        self.passage_embeddings = np.array(
            self.chunk_store.get_embeddings(self.passage_node_keys)
        ) if self.passage_node_keys else np.array([])

        self.fact_embeddings = np.array(
            self.fact_store.get_embeddings(self.fact_node_keys)
        ) if self.fact_node_keys else np.array([])

        # Build ent_node_to_chunk_ids from openie results
        self.ent_node_to_chunk_ids = {}
        if os.path.isfile(self.openie_results_path):
            all_info = json.load(open(self.openie_results_path)).get("docs", [])
            ner_dict, triple_dict = reformat_openie_results(all_info)
            for cid in self.passage_node_keys:
                if cid not in triple_dict:
                    continue
                triples = [text_processing(t) for t in triple_dict[cid].triples]
                for triple in triples:
                    if len(triple) == 3:
                        for ent in [triple[0], triple[2]]:
                            nk = compute_mdhash_id(ent, prefix="entity-")
                            self.ent_node_to_chunk_ids.setdefault(nk, set()).add(cid)

        self._ready = True
        logger.info(
            f"Ready. entities={len(self.entity_node_keys)}, "
            f"passages={len(self.passage_node_keys)}, "
            f"facts={len(self.fact_node_keys)}"
        )

    # ------------------------------------------------------------------
    # Query embedding
    # ------------------------------------------------------------------

    def _encode_queries(self, queries: List[str]):
        to_encode = [q for q in queries if q not in self._query_emb_fact]
        if not to_encode:
            return

        fact_embs = self.embedding_model.batch_encode(
            to_encode, instruction=get_query_instruction("query_to_fact"), norm=True
        )
        pass_embs = self.embedding_model.batch_encode(
            to_encode, instruction=get_query_instruction("query_to_passage"), norm=True
        )
        for q, fe, pe in zip(to_encode, fact_embs, pass_embs):
            self._query_emb_fact[q] = fe
            self._query_emb_pass[q] = pe

    # ------------------------------------------------------------------
    # Fact scoring
    # ------------------------------------------------------------------

    def _get_fact_scores(self, query: str) -> np.ndarray:
        qe = self._query_emb_fact.get(query)
        if qe is None:
            qe = self.embedding_model.batch_encode(
                query, instruction=get_query_instruction("query_to_fact"), norm=True
            )
        if len(self.fact_embeddings) == 0:
            return np.array([])
        scores = np.dot(self.fact_embeddings, qe.T)
        scores = np.squeeze(scores) if scores.ndim == 2 else scores
        return min_max_normalize(scores)

    # ------------------------------------------------------------------
    # Dense passage retrieval (fallback)
    # ------------------------------------------------------------------

    def _dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        qe = self._query_emb_pass.get(query)
        if qe is None:
            qe = self.embedding_model.batch_encode(
                query, instruction=get_query_instruction("query_to_passage"), norm=True
            )
        scores = np.dot(self.passage_embeddings, qe.T)
        scores = np.squeeze(scores) if scores.ndim == 2 else scores
        scores = min_max_normalize(scores)
        sorted_ids = np.argsort(scores)[::-1]
        return sorted_ids, scores[sorted_ids]

    # ------------------------------------------------------------------
    # Rerank facts
    # ------------------------------------------------------------------

    def _rerank_facts(
        self, query: str, fact_scores: np.ndarray
    ) -> Tuple[List[int], List[tuple], dict]:
        link_top_k = self.config.linking_top_k
        if len(fact_scores) == 0 or len(self.fact_node_keys) == 0:
            return [], [], {}

        if len(fact_scores) <= link_top_k:
            cand_indices = np.argsort(fact_scores)[::-1].tolist()
        else:
            cand_indices = np.argsort(fact_scores)[-link_top_k:][::-1].tolist()

        real_ids = [self.fact_node_keys[i] for i in cand_indices]
        rows = self.fact_store.get_rows(real_ids)
        cand_facts = [eval(rows[rid]["content"]) for rid in real_ids]

        top_indices, top_facts, meta = self.reranker(
            query, cand_facts, cand_indices, len_after_rerank=link_top_k
        )
        return top_indices, top_facts, meta

    # ------------------------------------------------------------------
    # Graph search (core of HippoRAG retrieval)
    # ------------------------------------------------------------------

    def _graph_search(
        self,
        query: str,
        fact_scores: np.ndarray,
        top_k_facts: List[tuple],
        top_k_fact_indices: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute seed weights -> run QAFD -> return sorted passage ids + scores."""
        link_top_k = self.config.linking_top_k
        n_nodes = self.graph.vcount()

        # --- entity seed weights ---
        linking_score_map: Dict[str, float] = {}
        phrase_scores: Dict[str, list] = {}
        phrase_weights = np.zeros(n_nodes)
        passage_weights = np.zeros(n_nodes)
        number_of_occurs = np.zeros(n_nodes)
        phrases_and_ids = set()

        for rank, f in enumerate(top_k_facts):
            subj = f[0].lower()
            obj = f[2].lower()
            fs = (
                fact_scores[top_k_fact_indices[rank]]
                if fact_scores.ndim > 0
                else float(fact_scores)
            )

            for phrase in [subj, obj]:
                pk = compute_mdhash_id(phrase, prefix="entity-")
                pid = self.node_name_to_vertex_idx.get(pk)
                if pid is not None:
                    wfs = fs
                    num_chunks = len(self.ent_node_to_chunk_ids.get(pk, set()))
                    if num_chunks > 0:
                        wfs /= num_chunks
                    phrase_weights[pid] += wfs
                    number_of_occurs[pid] += 1
                phrases_and_ids.add((phrase, pid))

        # Normalise
        nonzero = number_of_occurs > 0
        phrase_weights[nonzero] /= number_of_occurs[nonzero]

        for phrase, pid in phrases_and_ids:
            if pid is not None:
                phrase_scores.setdefault(phrase, []).append(phrase_weights[pid])

        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        # Keep only top-k entity seeds
        if link_top_k and linking_score_map:
            linking_score_map = dict(
                sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[
                    :link_top_k
                ]
            )
            top_phrases = {
                compute_mdhash_id(p, prefix="entity-")
                for p in linking_score_map
            }
            for nk in self.node_name_to_vertex_idx:
                if nk not in top_phrases:
                    pid = self.node_name_to_vertex_idx.get(nk)
                    if pid is not None:
                        phrase_weights[pid] = 0.0

        # --- passage seed weights ---
        dpr_ids, dpr_scores = self._dense_passage_retrieval(query)
        norm_dpr = min_max_normalize(dpr_scores)
        pw = self.config.passage_node_weight

        for i, did in enumerate(dpr_ids.tolist()):
            pk = self.passage_node_keys[did]
            pid = self.node_name_to_vertex_idx.get(pk)
            if pid is not None:
                passage_weights[pid] = norm_dpr[i] * pw

        node_weights = phrase_weights + passage_weights

        if np.sum(node_weights) == 0:
            logger.warning("All node weights are zero after seed selection, falling back to DPR")
            return dpr_ids, dpr_scores

        # --- Build node embeddings dict (cached) ---
        if not hasattr(self, '_node_emb_dict') or self._node_emb_dict is None:
            self._node_emb_dict = {}
            for i, nk in enumerate(self.entity_node_keys):
                if i < len(self.entity_embeddings):
                    self._node_emb_dict[nk] = self.entity_embeddings[i]
            for i, nk in enumerate(self.passage_node_keys):
                if i < len(self.passage_embeddings):
                    self._node_emb_dict[nk] = self.passage_embeddings[i]

        query_emb = self._query_emb_fact.get(query)

        # --- Run QAFD ---
        qafd_start = time.time()
        sorted_ids, sorted_scores = run_igraph_qafd(
            graph=self.graph,
            node_name_to_idx=self.node_name_to_vertex_idx,
            passage_node_idxs=self.passage_node_idxs,
            source_weights=node_weights,
            node_embeddings=self._node_emb_dict,
            query_embedding=query_emb,
            alpha=self.config.qafd_alpha,
            epsilon=self.config.qafd_epsilon,
            max_iterations=self.config.qafd_max_iterations,
            step_size=self.config.qafd_step_size,
            weight_scheme=self.config.qafd_weight_scheme,
            use_node_degree=self.config.qafd_use_node_degree,
            random_seed=self.config.qafd_random_seed,
            sim_mode=self.config.sim_mode,
            qa_sink_gamma=self.config.qa_sink_gamma,
            qa_warm_delta=self.config.qa_warm_delta,
            qa_warm_walk=self.config.qa_warm_walk,
            qa_warm_steps=self.config.qa_warm_steps,
            qa_accum_gamma=self.config.qa_accum_gamma,
            qa_post_lambda=self.config.qa_post_lambda,
            batch_push=self.config.batch_push,
        )
        qafd_elapsed = time.time() - qafd_start
        self.qafd_time += qafd_elapsed
        logger.info(f"QAFD completed in {qafd_elapsed:.2f}s")

        return sorted_ids, sorted_scores

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(
        self,
        queries: List[str],
        num_to_retrieve: int = None,
        gold_docs: List[List[str]] = None,
    ) -> List[QuerySolution]:
        """Retrieve documents for a batch of queries.

        Returns a list of ``QuerySolution`` objects.
        """
        if not self._ready:
            self.prepare()

        if num_to_retrieve is None:
            num_to_retrieve = self.config.retrieval_top_k

        self._encode_queries(queries)

        results = []
        t0 = time.time()

        for q in tqdm(queries, desc="Retrieving"):
            rerank_t0 = time.time()
            fact_scores = self._get_fact_scores(q)
            top_indices, top_facts, _ = self._rerank_facts(q, fact_scores)
            self.rerank_time += time.time() - rerank_t0

            if len(top_facts) == 0:
                logger.info("No facts after reranking -> fallback to DPR")
                sorted_ids, sorted_scores = self._dense_passage_retrieval(q)
            else:
                sorted_ids, sorted_scores = self._graph_search(
                    q, fact_scores, top_facts, top_indices
                )

            top_docs = [
                self.chunk_store.get_row(self.passage_node_keys[idx])["content"]
                for idx in sorted_ids[:num_to_retrieve]
            ]
            results.append(
                QuerySolution(
                    question=q,
                    docs=top_docs,
                    doc_scores=sorted_scores[:num_to_retrieve],
                )
            )

        self.total_time += time.time() - t0
        logger.info(
            f"Retrieval done. total={self.total_time:.1f}s, "
            f"rerank={self.rerank_time:.1f}s, qafd={self.qafd_time:.1f}s"
        )
        return results
