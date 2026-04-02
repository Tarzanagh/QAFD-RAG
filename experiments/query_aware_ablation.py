"""
Ablation: Query-Aware vs Query-Agnostic Flow Diffusion
=======================================================

Condition A (Query-Agnostic): b=0 → w̄ = H_sim(h(u), h(v)) · a
Condition B (Query-Aware):    b=0.25 → w̄ = H_sim(h(u), h(v)) · (a + b·(sim(u,q)+sim(v,q)))

Measures:
  - Downstream task quality (Recall@K, F1, EM)
  - Subgraph size (nodes with nonzero flow)
  - Leakage ratio (mass at irrelevant vs relevant nodes)
  - Convergence iterations
  - Per-query qualitative diagnostics
"""

import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project_root)

import types as _types
for _pkg_path in ["src", "src.retrievers", "src.hipporag_pipeline"]:
    if _pkg_path not in sys.modules:
        _m = _types.ModuleType(_pkg_path)
        _m.__path__ = [os.path.join(_project_root, *_pkg_path.split("."))]
        _m.__package__ = _pkg_path
        sys.modules[_pkg_path] = _m

import importlib.util as _ilu
def _load_mod(fqn, filepath):
    spec = _ilu.spec_from_file_location(fqn, filepath)
    mod = _ilu.module_from_spec(spec)
    sys.modules[fqn] = mod
    spec.loader.exec_module(mod)
    return mod

_src = os.path.join(_project_root, "src")
_load_mod("src.retrievers.base", os.path.join(_src, "retrievers", "base.py"))
_load_mod("src.retrievers.flow_diffusion", os.path.join(_src, "retrievers", "flow_diffusion.py"))

from src.hipporag_pipeline.config import HippoRAGConfig
from src.hipporag_pipeline.embedding_store import EmbeddingModelWrapper
from src.hipporag_pipeline.kg_builder import KGBuilder
from src.hipporag_pipeline.openie import OpenIE
from src.hipporag_pipeline.reranker import FactReranker
from src.hipporag_pipeline.retriever import HippoRAGRetriever
from src.hipporag_pipeline.graph_adapter import IGraphQAFD
from src.hipporag_pipeline.benchmark_runner import (
    get_gold_docs, get_gold_answers, recall_at_k,
    exact_match, f1_score, run_qa, _openai_embed, _openai_complete,
)
from src.hipporag_pipeline.utils import compute_mdhash_id, min_max_normalize

logging.basicConfig(level=logging.WARNING)


# ===========================================================================
# Data + KG loading (once per dataset)
# ===========================================================================

def load_dataset_and_kg(dataset="musique", num_queries=10):
    api_key = os.environ.get("OPENAI_API_KEY", "")
    config = HippoRAGConfig(
        llm_model="gpt-4o-mini",
        embedding_model_key="openai-small",
        dataset=dataset,
        save_dir="outputs",
    )
    import asyncio

    async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return await _openai_complete(
            model="gpt-4o-mini", prompt=prompt,
            system_prompt=system_prompt, history_messages=history_messages,
            api_key=api_key, **kwargs
        )

    async def embed_func(texts):
        return await _openai_embed(texts, model="text-embedding-3-small", api_key=api_key)

    embedding_model = EmbeddingModelWrapper(embed_func, batch_size=16)
    openie = OpenIE(llm_func)
    builder = KGBuilder(config, embedding_model, openie)

    data_dir = os.path.join(_project_root, "data", "multihop")
    with open(os.path.join(data_dir, f"{dataset}_corpus.json")) as f:
        corpus = json.load(f)
    docs = [f"{d['title']}\n{d['text']}" for d in corpus]

    with open(os.path.join(data_dir, f"{dataset}.json")) as f:
        samples = json.load(f)[:num_queries]

    queries = [s["question"] for s in samples]
    gold_answers = get_gold_answers(samples)
    gold_docs = get_gold_docs(samples, dataset)

    builder.index(docs)

    reranker = FactReranker(llm_func)
    retriever = HippoRAGRetriever(
        config=config, embedding_model=embedding_model, reranker=reranker,
        graph=builder.graph,
        chunk_embedding_store=builder.chunk_embedding_store,
        entity_embedding_store=builder.entity_embedding_store,
        fact_embedding_store=builder.fact_embedding_store,
        openie_results_path=builder.openie_results_path,
    )
    retriever.prepare()

    # Cache node embeddings
    retriever._node_emb_dict = {}
    for i, nk in enumerate(retriever.entity_node_keys):
        if i < len(retriever.entity_embeddings):
            retriever._node_emb_dict[nk] = retriever.entity_embeddings[i]
    for i, nk in enumerate(retriever.passage_node_keys):
        if i < len(retriever.passage_embeddings):
            retriever._node_emb_dict[nk] = retriever.passage_embeddings[i]

    return config, retriever, queries, gold_answers, gold_docs, llm_func


# ===========================================================================
# Core: run QAFD with detailed diagnostics
# ===========================================================================

def run_qafd_detailed(
    retriever: HippoRAGRetriever,
    query: str,
    gold_doc_set: set,
    alpha: float,
    weight_scheme: str,
    hybrid_a: float,
    hybrid_b: float,
    query_aware: bool,
) -> Dict:
    """Run QAFD on a single query, return detailed metrics."""

    retriever._encode_queries([query])
    fact_scores = retriever._get_fact_scores(query)
    top_indices, top_facts, _ = retriever._rerank_facts(query, fact_scores)

    # DPR fallback
    if len(top_facts) == 0:
        sorted_ids, sorted_scores = retriever._dense_passage_retrieval(query)
        top_docs = [
            retriever.chunk_store.get_row(retriever.passage_node_keys[idx])["content"]
            for idx in sorted_ids[:200]
        ]
        return {
            "method": "DPR_fallback",
            "top_docs": top_docs,
            "subgraph_size": 0,
            "leakage_ratio": 1.0,
            "convergence_iters": -1,
            "qafd_time": 0.0,
            "flow_at_relevant": 0.0,
            "flow_at_irrelevant": 0.0,
        }

    # Compute seed weights
    n_nodes = retriever.graph.vcount()
    phrase_weights = np.zeros(n_nodes)
    passage_weights = np.zeros(n_nodes)
    number_of_occurs = np.zeros(n_nodes)

    for rank, f in enumerate(top_facts):
        subj, obj = f[0].lower(), f[2].lower()
        fs = fact_scores[top_indices[rank]] if fact_scores.ndim > 0 else float(fact_scores)
        for phrase in [subj, obj]:
            pk = compute_mdhash_id(phrase, prefix="entity-")
            pid = retriever.node_name_to_vertex_idx.get(pk)
            if pid is not None:
                wfs = fs
                nc = len(retriever.ent_node_to_chunk_ids.get(pk, set()))
                if nc > 0:
                    wfs /= nc
                phrase_weights[pid] += wfs
                number_of_occurs[pid] += 1

    nonzero = number_of_occurs > 0
    phrase_weights[nonzero] /= number_of_occurs[nonzero]

    dpr_ids, dpr_scores = retriever._dense_passage_retrieval(query)
    norm_dpr = min_max_normalize(dpr_scores)
    pw = retriever.config.passage_node_weight
    for i, did in enumerate(dpr_ids.tolist()):
        pk = retriever.passage_node_keys[did]
        pid = retriever.node_name_to_vertex_idx.get(pk)
        if pid is not None:
            passage_weights[pid] = norm_dpr[i] * pw

    node_weights = phrase_weights + passage_weights
    if np.sum(node_weights) == 0:
        sorted_ids, sorted_scores = dpr_ids, dpr_scores
        top_docs = [
            retriever.chunk_store.get_row(retriever.passage_node_keys[idx])["content"]
            for idx in sorted_ids[:200]
        ]
        return {
            "method": "DPR_fallback_zero_seeds",
            "top_docs": top_docs,
            "subgraph_size": 0,
            "leakage_ratio": 1.0,
            "convergence_iters": -1,
            "qafd_time": 0.0,
            "flow_at_relevant": 0.0,
            "flow_at_irrelevant": 0.0,
        }

    query_emb = retriever._query_emb_fact.get(query) if query_aware else None
    node_embs = retriever._node_emb_dict if query_aware else {}

    # Run QAFD directly to get raw node scores
    t0 = time.time()
    qafd = IGraphQAFD(
        graph=retriever.graph,
        node_name_to_idx=retriever.node_name_to_vertex_idx,
        source_weights=node_weights,
        node_embeddings=node_embs,
        query_embedding=query_emb,
        alpha=alpha,
        epsilon=retriever.config.qafd_epsilon,
        max_iterations=retriever.config.qafd_max_iterations,
        step_size=retriever.config.qafd_step_size,
        weight_scheme=weight_scheme,
        hybrid_a=hybrid_a,
        hybrid_b=hybrid_b,
        use_node_degree=retriever.config.qafd_use_node_degree,
        random_seed=retriever.config.qafd_random_seed,
    )
    raw_scores = qafd.run()
    elapsed = time.time() - t0

    # Extract passage scores
    doc_scores = np.array([raw_scores[idx] for idx in retriever.passage_node_idxs])
    total = np.sum(doc_scores)
    if total > 0:
        doc_scores_norm = doc_scores / total
    else:
        doc_scores_norm = np.ones(len(doc_scores)) / max(len(doc_scores), 1)

    sorted_ids = np.argsort(doc_scores_norm)[::-1]
    sorted_scores = doc_scores_norm[sorted_ids]

    top_docs = [
        retriever.chunk_store.get_row(retriever.passage_node_keys[idx])["content"]
        for idx in sorted_ids[:200]
    ]

    # --- Compute detailed metrics ---

    # Subgraph size: nodes with nonzero flow
    subgraph_size = int(np.sum(raw_scores > 1e-10))

    # Leakage ratio: flow at irrelevant vs relevant passage nodes
    flow_relevant = 0.0
    flow_irrelevant = 0.0
    relevant_count = 0
    irrelevant_count = 0

    for i, pk in enumerate(retriever.passage_node_keys):
        content = retriever.chunk_store.get_row(pk)["content"]
        score = doc_scores[i]
        is_gold = any(g in content or content in g for g in gold_doc_set)
        if is_gold:
            flow_relevant += score
            relevant_count += 1
        else:
            flow_irrelevant += score
            irrelevant_count += 1

    total_flow = flow_relevant + flow_irrelevant
    leakage = flow_irrelevant / total_flow if total_flow > 0 else 1.0

    return {
        "method": "QAFD",
        "top_docs": top_docs,
        "subgraph_size": subgraph_size,
        "leakage_ratio": round(leakage, 4),
        "flow_at_relevant": round(flow_relevant, 6),
        "flow_at_irrelevant": round(flow_irrelevant, 6),
        "relevant_passages": relevant_count,
        "qafd_time": round(elapsed, 4),
    }


# ===========================================================================
# Run full experiment
# ===========================================================================

def run_experiment(
    retriever, queries, gold_docs, gold_answers,
    alpha: float, hybrid_a: float, hybrid_b: float,
    query_aware: bool, label: str,
) -> Dict:
    """Run all queries with given params, return aggregate metrics."""
    all_docs = []
    subgraph_sizes = []
    leakage_ratios = []
    flow_relevants = []
    flow_irrelevants = []
    qafd_times = []
    per_query = []

    for qi, q in enumerate(queries):
        gold_set = set(gold_docs[qi]) if gold_docs else set()
        r = run_qafd_detailed(
            retriever, q, gold_set,
            alpha=alpha,
            weight_scheme="original",
            hybrid_a=hybrid_a,
            hybrid_b=hybrid_b,
            query_aware=query_aware,
        )
        all_docs.append(r["top_docs"])
        subgraph_sizes.append(r["subgraph_size"])
        leakage_ratios.append(r["leakage_ratio"])
        flow_relevants.append(r["flow_at_relevant"])
        flow_irrelevants.append(r["flow_at_irrelevant"])
        qafd_times.append(r["qafd_time"])
        per_query.append({
            "query": q[:100],
            "method": r["method"],
            "subgraph_size": r["subgraph_size"],
            "leakage_ratio": r["leakage_ratio"],
            "flow_relevant": r["flow_at_relevant"],
            "flow_irrelevant": r["flow_at_irrelevant"],
        })

    # Recall
    k_list = [1, 2, 5, 10, 20, 50, 100, 200]
    recall_metrics = recall_at_k(gold_docs, all_docs, k_list) if gold_docs else {}

    return {
        "label": label,
        "alpha": alpha,
        "hybrid_a": hybrid_a,
        "hybrid_b": hybrid_b,
        "query_aware": query_aware,
        "recall": recall_metrics,
        "avg_subgraph_size": round(np.mean(subgraph_sizes), 1),
        "avg_leakage_ratio": round(np.mean(leakage_ratios), 4),
        "avg_flow_relevant": round(np.mean(flow_relevants), 6),
        "avg_flow_irrelevant": round(np.mean(flow_irrelevants), 6),
        "avg_qafd_time": round(np.mean(qafd_times), 4),
        "per_query": per_query,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--num_queries", type=int, default=10)
    args = parser.parse_args()

    print("=" * 70)
    print("  Query-Aware vs Query-Agnostic Ablation Study")
    print(f"  Dataset: {args.dataset}, Queries: {args.num_queries}")
    print("=" * 70)

    config, retriever, queries, gold_answers, gold_docs, llm_func = \
        load_dataset_and_kg(args.dataset, args.num_queries)

    print(f"  Graph: {retriever.graph.vcount()} nodes, {retriever.graph.ecount()} edges")
    print(f"  Entities: {len(retriever.entity_node_keys)}, Passages: {len(retriever.passage_node_keys)}")

    all_results = {}

    # ── Run across multiple alpha values ─────────────────────────────
    alpha_values = [2.0, 10.0, 50.0]

    for alpha in alpha_values:
        print(f"\n{'─' * 70}")
        print(f"  Alpha = {alpha}")
        print(f"{'─' * 70}")

        # Condition A: Query-Agnostic (b=0)
        print(f"  Running Query-Agnostic (b=0) ...")
        r_agnostic = run_experiment(
            retriever, queries, gold_docs, gold_answers,
            alpha=alpha, hybrid_a=1.0, hybrid_b=0.0,
            query_aware=False, label=f"agnostic_a{alpha}",
        )
        all_results[f"agnostic_a{alpha}"] = r_agnostic

        # Condition B: Query-Aware (a=1, b=0.25)
        print(f"  Running Query-Aware (a=1, b=0.25) ...")
        r_aware_025 = run_experiment(
            retriever, queries, gold_docs, gold_answers,
            alpha=alpha, hybrid_a=1.0, hybrid_b=0.25,
            query_aware=True, label=f"aware025_a{alpha}",
        )
        all_results[f"aware025_a{alpha}"] = r_aware_025

        # Condition C: Query-Aware (a=1, b=0.5) — current default
        print(f"  Running Query-Aware (a=1, b=0.5) ...")
        r_aware_050 = run_experiment(
            retriever, queries, gold_docs, gold_answers,
            alpha=alpha, hybrid_a=1.0, hybrid_b=0.5,
            query_aware=True, label=f"aware050_a{alpha}",
        )
        all_results[f"aware050_a{alpha}"] = r_aware_050

        # Print comparison table
        print(f"\n  {'Metric':<25} {'Agnostic(b=0)':>15} {'Aware(b=0.25)':>15} {'Aware(b=0.5)':>15} {'Δ(0.25 vs 0)':>15}")
        print(f"  {'─' * 85}")

        for k in [10, 50, 100]:
            key = f"Recall@{k}"
            va = r_agnostic['recall'].get(key, 0)
            vb = r_aware_025['recall'].get(key, 0)
            vc = r_aware_050['recall'].get(key, 0)
            d = vb - va
            marker = " ↑" if d > 0 else (" ↓" if d < 0 else "")
            print(f"  {key:<25} {va:>15.4f} {vb:>15.4f} {vc:>15.4f} {d:>+14.4f}{marker}")

        print(f"  {'Subgraph size':<25} {r_agnostic['avg_subgraph_size']:>15.1f} {r_aware_025['avg_subgraph_size']:>15.1f} {r_aware_050['avg_subgraph_size']:>15.1f} {r_aware_025['avg_subgraph_size'] - r_agnostic['avg_subgraph_size']:>+14.1f}")
        print(f"  {'Leakage ratio':<25} {r_agnostic['avg_leakage_ratio']:>15.4f} {r_aware_025['avg_leakage_ratio']:>15.4f} {r_aware_050['avg_leakage_ratio']:>15.4f} {r_aware_025['avg_leakage_ratio'] - r_agnostic['avg_leakage_ratio']:>+14.4f}")
        print(f"  {'Flow@relevant':<25} {r_agnostic['avg_flow_relevant']:>15.6f} {r_aware_025['avg_flow_relevant']:>15.6f} {r_aware_050['avg_flow_relevant']:>15.6f} {r_aware_025['avg_flow_relevant'] - r_agnostic['avg_flow_relevant']:>+14.6f}")
        print(f"  {'Flow@irrelevant':<25} {r_agnostic['avg_flow_irrelevant']:>15.6f} {r_aware_025['avg_flow_irrelevant']:>15.6f} {r_aware_050['avg_flow_irrelevant']:>15.6f} {r_aware_025['avg_flow_irrelevant'] - r_agnostic['avg_flow_irrelevant']:>+14.6f}")
        print(f"  {'QAFD time (s)':<25} {r_agnostic['avg_qafd_time']:>15.4f} {r_aware_025['avg_qafd_time']:>15.4f} {r_aware_050['avg_qafd_time']:>15.4f}")

    # ── Per-query diagnostics (alpha=10, first 5 queries) ────────────
    print(f"\n{'=' * 70}")
    print("  Per-Query Diagnostics (alpha=10.0)")
    print(f"{'=' * 70}")

    r_ag = all_results.get("agnostic_a10.0", {})
    r_aw = all_results.get("aware025_a10.0", {})

    if r_ag and r_aw:
        ag_pq = r_ag.get("per_query", [])
        aw_pq = r_aw.get("per_query", [])
        for qi in range(min(5, len(ag_pq))):
            ag = ag_pq[qi]
            aw = aw_pq[qi]
            print(f"\n  Q{qi}: {ag['query']}")
            print(f"    {'':>20} {'Agnostic':>12} {'Aware':>12} {'Delta':>12}")
            print(f"    {'Subgraph size':>20} {ag['subgraph_size']:>12} {aw['subgraph_size']:>12} {aw['subgraph_size']-ag['subgraph_size']:>+12}")
            print(f"    {'Leakage ratio':>20} {ag['leakage_ratio']:>12.4f} {aw['leakage_ratio']:>12.4f} {aw['leakage_ratio']-ag['leakage_ratio']:>+12.4f}")
            print(f"    {'Flow@relevant':>20} {ag['flow_relevant']:>12.6f} {aw['flow_relevant']:>12.6f} {aw['flow_relevant']-ag['flow_relevant']:>+12.6f}")
            print(f"    {'Flow@irrelevant':>20} {ag['flow_irrelevant']:>12.6f} {aw['flow_irrelevant']:>12.6f} {aw['flow_irrelevant']-ag['flow_irrelevant']:>+12.6f}")

            status = "HELPS" if aw['leakage_ratio'] < ag['leakage_ratio'] else (
                "HURTS" if aw['leakage_ratio'] > ag['leakage_ratio'] else "SAME"
            )
            print(f"    → Query awareness {status} (leakage {'decreased' if status == 'HELPS' else 'increased' if status == 'HURTS' else 'unchanged'})")

    # ── Save ─────────────────────────────────────────────────────────
    out_dir = os.path.join(_project_root, "experiments", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"query_aware_ablation_{args.dataset}.json")

    # Remove top_docs from saved output (too large)
    save_results = {}
    for k, v in all_results.items():
        sv = dict(v)
        sv.pop("per_query", None)
        save_results[k] = sv

    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
