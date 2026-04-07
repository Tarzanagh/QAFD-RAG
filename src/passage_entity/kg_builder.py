"""
Knowledge graph builder following the original ``index()`` method.

Steps:
    1. Insert docs into chunk embedding store.
    2. Run OpenIE (NER + triple extraction).
    3. Build igraph with entity nodes, passage nodes, fact edges,
       passage-to-entity edges, and synonymy edges.
    4. Save to ``graph.pickle``.
"""

import json
import logging
import os
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import igraph as ig
import numpy as np
import torch
from tqdm import tqdm

from .config import PassageEntityConfig
from .embedding_store import EmbeddingStore, EmbeddingModelWrapper
from .openie import OpenIE
from .utils import (
    NerRawOutput,
    TripleRawOutput,
    compute_mdhash_id,
    text_processing,
    extract_entity_nodes,
    flatten_facts,
    reformat_openie_results,
    filter_invalid_triples,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KNN helper (simplified from the original embed_utils.py)
# ---------------------------------------------------------------------------

def retrieve_knn(
    query_ids: List[str],
    key_ids: List[str],
    query_vecs: np.ndarray,
    key_vecs: np.ndarray,
    k: int = 2047,
    query_batch_size: int = 1000,
    key_batch_size: int = 10000,
) -> Dict:
    """Batched top-k cosine nearest-neighbour search using PyTorch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(key_vecs) == 0:
        return {}

    q = torch.tensor(np.array(query_vecs), dtype=torch.float32)
    q = torch.nn.functional.normalize(q, dim=1)
    keys = torch.tensor(np.array(key_vecs), dtype=torch.float32)
    keys = torch.nn.functional.normalize(keys, dim=1)

    results = {}

    def _batches(vecs, bs):
        for i in range(0, len(vecs), bs):
            yield vecs[i : i + bs], i

    for qb, qstart in tqdm(
        _batches(q, query_batch_size),
        total=(len(q) + query_batch_size - 1) // query_batch_size,
        desc="KNN",
    ):
        qb = qb.to(device)
        batch_scores, batch_indices = [], []
        offset = 0
        for kb, _ in _batches(keys, key_batch_size):
            kb = kb.to(device)
            actual_kb_size = kb.size(0)
            sim = torch.mm(qb, kb.T)
            topk_s, topk_i = torch.topk(sim, min(k, actual_kb_size), dim=1, largest=True, sorted=True)
            topk_i += offset
            batch_scores.append(topk_s)
            batch_indices.append(topk_i)
            del sim
            kb = kb.cpu()
            torch.cuda.empty_cache()
            offset += actual_kb_size

        batch_scores = torch.cat(batch_scores, dim=1)
        batch_indices = torch.cat(batch_indices, dim=1)
        final_s, final_i = torch.topk(
            batch_scores, min(k, batch_scores.size(1)), dim=1, largest=True, sorted=True
        )
        final_i = final_i.cpu()
        final_s = final_s.cpu()
        for i in range(final_i.size(0)):
            qi = qstart + i
            topk_rel = batch_indices[i][final_i[i]].cpu()
            topk_keys = [key_ids[idx] for idx in topk_rel.numpy()]
            results[query_ids[qi]] = (topk_keys, final_s[i].cpu().numpy().tolist())
        qb = qb.cpu()
        torch.cuda.empty_cache()

    return results


# ===========================================================================
# KGBuilder
# ===========================================================================

class KGBuilder:
    """Build a passage-entity knowledge graph and persist it to disk."""

    def __init__(
        self,
        config: PassageEntityConfig,
        embedding_model: EmbeddingModelWrapper,
        openie: OpenIE,
    ):
        self.config = config
        self.embedding_model = embedding_model
        self.openie = openie

        wd = config.working_dir
        os.makedirs(wd, exist_ok=True)

        self.chunk_embedding_store = EmbeddingStore(
            embedding_model, os.path.join(wd, "chunk_embeddings"),
            config.embedding_batch_size, "chunk",
        )
        self.entity_embedding_store = EmbeddingStore(
            embedding_model, os.path.join(wd, "entity_embeddings"),
            config.embedding_batch_size, "entity",
        )
        self.fact_embedding_store = EmbeddingStore(
            embedding_model, os.path.join(wd, "fact_embeddings"),
            config.embedding_batch_size, "fact",
        )

        self._graph_pickle_path = os.path.join(wd, "graph.pickle")
        self.openie_results_path = os.path.join(
            config.save_dir,
            f"openie_results_ner_{config.llm_model.replace('/', '_')}.json",
        )

        self.graph: ig.Graph = self._load_or_create_graph()
        self.node_to_node_stats: Dict[Tuple[str, str], float] = {}
        self.ent_node_to_chunk_ids: Dict[str, set] = {}

    # ------------------------------------------------------------------
    # Graph init
    # ------------------------------------------------------------------

    def _load_or_create_graph(self) -> ig.Graph:
        if (
            not self.config.force_index_from_scratch
            and os.path.exists(self._graph_pickle_path)
        ):
            g = ig.Graph.Read_Pickle(self._graph_pickle_path)
            logger.info(
                f"Loaded graph from {self._graph_pickle_path}: "
                f"{g.vcount()} nodes, {g.ecount()} edges"
            )
            return g
        return ig.Graph(directed=self.config.is_directed_graph)

    # ------------------------------------------------------------------
    # index()
    # ------------------------------------------------------------------

    def index(self, docs: List[str]):
        """Index documents: embed chunks, run OpenIE, build KG, save."""
        logger.info("=== Indexing documents ===")

        # 1) Insert chunks into embedding store
        self.chunk_embedding_store.insert_strings(docs)
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()

        # 2) Run OpenIE (or load cached)
        all_openie_info, chunk_keys_to_process = self._load_existing_openie(
            chunk_to_rows.keys()
        )
        new_openie_rows = {k: chunk_to_rows[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            logger.info(f"Running OpenIE on {len(chunk_keys_to_process)} new chunks")
            ner_dict, triple_dict = self.openie.batch_openie(new_openie_rows)
            self._merge_openie_results(
                all_openie_info, new_openie_rows, ner_dict, triple_dict
            )

        if self.config.save_openie:
            self._save_openie_results(all_openie_info)

        ner_results, triple_results = reformat_openie_results(all_openie_info)

        # Sanity check — fill missing entries
        for cid in chunk_to_rows:
            if cid not in ner_results:
                ner_results[cid] = NerRawOutput(cid, None, [], {})
            if cid not in triple_results:
                triple_results[cid] = TripleRawOutput(cid, None, [], {})

        chunk_ids = list(chunk_to_rows.keys())
        chunk_triples = [
            [text_processing(t) for t in triple_results[cid].triples]
            for cid in chunk_ids
        ]
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        facts = flatten_facts(chunk_triples)

        # 3) Encode entities + facts
        logger.info("Encoding entities")
        self.entity_embedding_store.insert_strings(entity_nodes)
        logger.info("Encoding facts")
        self.fact_embedding_store.insert_strings([str(f) for f in facts])

        # 4) Build graph edges
        logger.info("Building graph edges")
        self.node_to_node_stats = {}
        self.ent_node_to_chunk_ids = {}

        self._add_fact_edges(chunk_ids, chunk_triples)
        num_new = self._add_passage_edges(chunk_ids, chunk_triple_entities)

        if num_new > 0:
            logger.info(f"{num_new} new chunks → adding synonymy edges")
            self._add_synonymy_edges()
            self._augment_graph()
            self._save_graph()

        logger.info("=== Indexing complete ===")

    # ------------------------------------------------------------------
    # Edge builders
    # ------------------------------------------------------------------

    def _add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[list]):
        current_nodes = set(self.graph.vs["name"]) if "name" in self.graph.vs.attribute_names() else set()

        for chunk_key, triples in tqdm(
            zip(chunk_ids, chunk_triples), desc="Fact edges", total=len(chunk_ids)
        ):
            entities_in_chunk: set = set()
            if chunk_key not in current_nodes:
                for triple in triples:
                    triple = tuple(triple)
                    nk1 = compute_mdhash_id(triple[0], prefix="entity-")
                    nk2 = compute_mdhash_id(triple[2], prefix="entity-")
                    self.node_to_node_stats[(nk1, nk2)] = (
                        self.node_to_node_stats.get((nk1, nk2), 0.0) + 1
                    )
                    self.node_to_node_stats[(nk2, nk1)] = (
                        self.node_to_node_stats.get((nk2, nk1), 0.0) + 1
                    )
                    entities_in_chunk.update([nk1, nk2])

                for node in entities_in_chunk:
                    self.ent_node_to_chunk_ids[node] = (
                        self.ent_node_to_chunk_ids.get(node, set()) | {chunk_key}
                    )

    def _add_passage_edges(
        self, chunk_ids: List[str], chunk_triple_entities: List[List[str]]
    ) -> int:
        current_nodes = set(self.graph.vs["name"]) if "name" in self.graph.vs.attribute_names() else set()
        num_new = 0
        for idx, chunk_key in tqdm(
            enumerate(chunk_ids), desc="Passage edges", total=len(chunk_ids)
        ):
            if chunk_key not in current_nodes:
                for ent in chunk_triple_entities[idx]:
                    nk = compute_mdhash_id(ent, prefix="entity-")
                    self.node_to_node_stats[(chunk_key, nk)] = 1.0
                num_new += 1
        return num_new

    def _add_synonymy_edges(self):
        logger.info("Expanding graph with synonymy edges")
        entity_id_to_row = self.entity_embedding_store.get_all_id_to_rows()
        entity_node_keys = list(entity_id_to_row.keys())
        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)

        knn = retrieve_knn(
            query_ids=entity_node_keys,
            key_ids=entity_node_keys,
            query_vecs=entity_embs,
            key_vecs=entity_embs,
            k=self.config.synonymy_edge_topk,
            query_batch_size=self.config.synonymy_edge_query_batch_size,
            key_batch_size=self.config.synonymy_edge_key_batch_size,
        )

        for nk in tqdm(knn, desc="Synonymy edges"):
            entity = entity_id_to_row[nk]["content"]
            if len(re.sub('[^A-Za-z0-9]', '', entity)) <= 2:
                continue
            nns_keys, nns_scores = knn[nk]
            num_nns = 0
            for nn, score in zip(nns_keys, nns_scores):
                if score < self.config.synonymy_edge_sim_threshold or num_nns > 100:
                    break
                nn_phrase = entity_id_to_row.get(nn, {}).get("content", "")
                if nn != nk and nn_phrase:
                    self.node_to_node_stats[(nk, nn)] = score
                    num_nns += 1

    # ------------------------------------------------------------------
    # Graph augmentation
    # ------------------------------------------------------------------

    def _augment_graph(self):
        self._add_new_nodes()
        self._add_new_edges()
        info = self._get_graph_info()
        logger.info(f"Graph info: {info}")

    def _add_new_nodes(self):
        existing = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}

        entity_rows = self.entity_embedding_store.get_all_id_to_rows()
        passage_rows = self.chunk_embedding_store.get_all_id_to_rows()
        all_rows = {**entity_rows, **passage_rows}

        new_nodes: Dict[str, list] = {}
        for nid, node in all_rows.items():
            node["name"] = nid
            if nid not in existing:
                for k, v in node.items():
                    new_nodes.setdefault(k, []).append(v)

        if new_nodes:
            self.graph.add_vertices(
                n=len(next(iter(new_nodes.values()))), attributes=new_nodes
            )

    def _add_new_edges(self):
        edge_src, edge_tgt, weights = [], [], []
        for (s, t), w in self.node_to_node_stats.items():
            if s == t:
                continue
            edge_src.append(s)
            edge_tgt.append(t)
            weights.append(w)

        current_ids = set(self.graph.vs["name"])
        valid_edges, valid_w = [], []
        for s, t, w in zip(edge_src, edge_tgt, weights):
            if s in current_ids and t in current_ids:
                valid_edges.append((s, t))
                valid_w.append(w)
            else:
                logger.warning(f"Skipping invalid edge {s} -> {t}")

        self.graph.add_edges(valid_edges, attributes={"weight": valid_w})

    def _save_graph(self):
        logger.info(
            f"Writing graph: {self.graph.vcount()} nodes, {self.graph.ecount()} edges"
        )
        self.graph.write_pickle(self._graph_pickle_path)

    def _get_graph_info(self) -> Dict:
        ent_keys = set(self.entity_embedding_store.get_all_ids())
        pass_keys = set(self.chunk_embedding_store.get_all_ids())
        return {
            "num_entity_nodes": len(ent_keys),
            "num_passage_nodes": len(pass_keys),
            "num_total_nodes": len(ent_keys) + len(pass_keys),
            "num_facts": len(self.fact_embedding_store.get_all_ids()),
            "num_edges": len(self.node_to_node_stats),
        }

    # ------------------------------------------------------------------
    # OpenIE persistence
    # ------------------------------------------------------------------

    def _load_existing_openie(self, chunk_keys) -> Tuple[list, set]:
        chunk_keys_to_save: set = set()
        if (
            not self.config.force_openie_from_scratch
            and os.path.isfile(self.openie_results_path)
        ):
            data = json.load(open(self.openie_results_path))
            all_info = data.get("docs", [])
            # Standardise indices
            for item in all_info:
                item["idx"] = compute_mdhash_id(item["passage"], "chunk-")
            # Only consider a doc "done" if it has at least some extracted content
            existing_keys = set()
            for info in all_info:
                has_content = (
                    len(info.get("extracted_entities", [])) > 0
                    or len(info.get("extracted_triples", [])) > 0
                )
                if has_content:
                    existing_keys.add(info["idx"])
            for ck in chunk_keys:
                if ck not in existing_keys:
                    chunk_keys_to_save.add(ck)
        else:
            all_info = []
            chunk_keys_to_save = set(chunk_keys)
        return all_info, chunk_keys_to_save

    def _merge_openie_results(self, all_info, chunks, ner_dict, triple_dict):
        for ck, row in chunks.items():
            passage = row["content"]
            try:
                info = {
                    "idx": ck,
                    "passage": passage,
                    "extracted_entities": ner_dict[ck].unique_entities,
                    "extracted_triples": triple_dict[ck].triples,
                }
            except Exception as e:
                logger.error(f"Error merging chunk {ck}: {e}")
                info = {
                    "idx": ck,
                    "passage": passage,
                    "extracted_entities": [],
                    "extracted_triples": [],
                }
            all_info.append(info)

    def _save_openie_results(self, all_info: list):
        num_phrases = sum(len(c["extracted_entities"]) for c in all_info)
        if num_phrases > 0:
            avg_chars = round(
                sum(len(e) for c in all_info for e in c["extracted_entities"]) / num_phrases, 4
            )
            avg_words = round(
                sum(len(e.split()) for c in all_info for e in c["extracted_entities"]) / num_phrases, 4
            )
        else:
            avg_chars, avg_words = 0, 0

        os.makedirs(os.path.dirname(self.openie_results_path), exist_ok=True)
        with open(self.openie_results_path, "w") as f:
            json.dump(
                {"docs": all_info, "avg_ent_chars": avg_chars, "avg_ent_words": avg_words},
                f,
            )
        logger.info(f"OpenIE results saved to {self.openie_results_path}")
