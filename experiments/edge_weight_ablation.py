"""
Edge Weight Ablation Study for QAFD-RAG
========================================

Runs all experiments from a single script to avoid reloading the KG each time.

Experiments:
1. Query-aware vs query-agnostic
2. Weight scheme comparison (original/multiply/add)
3. (a, b) sensitivity sweep for Hybrid
4. Qualitative per-query diagnostics
"""

import json
import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Setup paths
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
from src.hipporag_pipeline.graph_adapter import run_igraph_qafd
from src.hipporag_pipeline.benchmark_runner import (
    get_gold_docs, get_gold_answers, recall_at_k,
    exact_match, f1_score, run_qa, _openai_embed, _openai_complete,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load KG once
# ---------------------------------------------------------------------------

def load_everything(dataset="musique", num_queries=10):
    """Load KG, embeddings, data — once for all experiments."""
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

    # Load data
    data_dir = os.path.join(_project_root, "data", "multihop")
    with open(os.path.join(data_dir, f"{dataset}_corpus.json")) as f:
        corpus = json.load(f)
    docs = [f"{d['title']}\n{d['text']}" for d in corpus]

    with open(os.path.join(data_dir, f"{dataset}.json")) as f:
        samples = json.load(f)[:num_queries]

    queries = [s["question"] for s in samples]
    gold_answers = get_gold_answers(samples)
    gold_docs = get_gold_docs(samples, dataset)

    # Build / load KG
    builder.index(docs)

    # Build retriever
    reranker = FactReranker(llm_func)
    retriever = HippoRAGRetriever(
        config=config,
        embedding_model=embedding_model,
        reranker=reranker,
        graph=builder.graph,
        chunk_embedding_store=builder.chunk_embedding_store,
        entity_embedding_store=builder.entity_embedding_store,
        fact_embedding_store=builder.fact_embedding_store,
        openie_results_path=builder.openie_results_path,
    )
    retriever.prepare()

    return config, retriever, queries, gold_answers, gold_docs, llm_func


def run_retrieval_with_params(
    retriever: HippoRAGRetriever,
    queries: List[str],
    gold_docs: List[List[str]],
    gold_answers,
    weight_scheme: str = "original",
    hybrid_a: float = 1.0,
    hybrid_b: float = 0.5,
    query_aware: bool = True,
) -> Dict:
    """Run retrieval with specific edge weight params. Returns metrics dict."""
    from src.hipporag_pipeline.utils import compute_mdhash_id, min_max_normalize

    retriever._encode_queries(queries)
    k_list = [1, 2, 5, 10, 20, 50, 100, 200]
    all_retrieved = []
    qafd_times = []
    convergence_iters = []

    for q in queries:
        fact_scores = retriever._get_fact_scores(q)
        top_indices, top_facts, _ = retriever._rerank_facts(q, fact_scores)

        if len(top_facts) == 0:
            sorted_ids, sorted_scores = retriever._dense_passage_retrieval(q)
            all_retrieved.append([
                retriever.chunk_store.get_row(retriever.passage_node_keys[idx])["content"]
                for idx in sorted_ids[:200]
            ])
            qafd_times.append(0.0)
            convergence_iters.append(-1)  # DPR fallback
            continue

        # Compute seed weights (same as retriever._graph_search)
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
                    num_chunks = len(retriever.ent_node_to_chunk_ids.get(pk, set()))
                    if num_chunks > 0:
                        wfs /= num_chunks
                    phrase_weights[pid] += wfs
                    number_of_occurs[pid] += 1

        nonzero = number_of_occurs > 0
        phrase_weights[nonzero] /= number_of_occurs[nonzero]

        dpr_ids, dpr_scores = retriever._dense_passage_retrieval(q)
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
            all_retrieved.append([
                retriever.chunk_store.get_row(retriever.passage_node_keys[idx])["content"]
                for idx in sorted_ids[:200]
            ])
            qafd_times.append(0.0)
            convergence_iters.append(-1)
            continue

        # Build node embeddings
        if not hasattr(retriever, '_node_emb_dict') or retriever._node_emb_dict is None:
            retriever._node_emb_dict = {}
            for i, nk in enumerate(retriever.entity_node_keys):
                if i < len(retriever.entity_embeddings):
                    retriever._node_emb_dict[nk] = retriever.entity_embeddings[i]
            for i, nk in enumerate(retriever.passage_node_keys):
                if i < len(retriever.passage_embeddings):
                    retriever._node_emb_dict[nk] = retriever.passage_embeddings[i]

        query_emb = retriever._query_emb_fact.get(q) if query_aware else None

        t0 = time.time()
        sorted_ids, sorted_scores = run_igraph_qafd(
            graph=retriever.graph,
            node_name_to_idx=retriever.node_name_to_vertex_idx,
            passage_node_idxs=retriever.passage_node_idxs,
            source_weights=node_weights,
            node_embeddings=retriever._node_emb_dict if query_aware else {},
            query_embedding=query_emb,
            alpha=retriever.config.qafd_alpha,
            epsilon=retriever.config.qafd_epsilon,
            max_iterations=retriever.config.qafd_max_iterations,
            step_size=retriever.config.qafd_step_size,
            weight_scheme=weight_scheme,
            hybrid_a=hybrid_a,
            hybrid_b=hybrid_b,
            use_node_degree=retriever.config.qafd_use_node_degree,
            random_seed=retriever.config.qafd_random_seed,
        )
        elapsed = time.time() - t0
        qafd_times.append(elapsed)

        top_docs = [
            retriever.chunk_store.get_row(retriever.passage_node_keys[idx])["content"]
            for idx in sorted_ids[:200]
        ]
        all_retrieved.append(top_docs)

    # Compute metrics
    recall_metrics = recall_at_k(gold_docs, all_retrieved, k_list) if gold_docs else {}

    return {
        "recall": recall_metrics,
        "avg_qafd_time": np.mean([t for t in qafd_times if t > 0]) if any(t > 0 for t in qafd_times) else 0,
        "dpr_fallback_count": sum(1 for t in convergence_iters if t == -1),
        "qafd_queries": sum(1 for t in qafd_times if t > 0),
    }


# ===========================================================================
# Main experiments
# ===========================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--num_queries", type=int, default=10)
    args = parser.parse_args()

    print("=" * 70)
    print("  QAFD-RAG Edge Weight Ablation Study")
    print(f"  Dataset: {args.dataset}, Queries: {args.num_queries}")
    print("=" * 70)

    config, retriever, queries, gold_answers, gold_docs, llm_func = load_everything(
        args.dataset, args.num_queries
    )

    results = {}

    # ── Experiment 1: Query-aware vs Query-agnostic ──────────────────
    print("\n[1/4] Query-aware vs Query-agnostic ablation")

    print("  Running: Hybrid (a=1, b=0.5) — query-aware ...")
    r = run_retrieval_with_params(
        retriever, queries, gold_docs, gold_answers,
        weight_scheme="original", hybrid_a=1.0, hybrid_b=0.5, query_aware=True,
    )
    results["hybrid_query_aware"] = r
    print(f"    R@10={r['recall'].get('Recall@10', 0):.4f}  R@100={r['recall'].get('Recall@100', 0):.4f}  QAFD={r['avg_qafd_time']:.3f}s  DPR_fallback={r['dpr_fallback_count']}")

    print("  Running: Query-agnostic (b=0) ...")
    r = run_retrieval_with_params(
        retriever, queries, gold_docs, gold_answers,
        weight_scheme="original", hybrid_a=1.0, hybrid_b=0.0, query_aware=False,
    )
    results["query_agnostic"] = r
    print(f"    R@10={r['recall'].get('Recall@10', 0):.4f}  R@100={r['recall'].get('Recall@100', 0):.4f}  QAFD={r['avg_qafd_time']:.3f}s  DPR_fallback={r['dpr_fallback_count']}")

    # ── Experiment 2: Weight scheme comparison ───────────────────────
    print("\n[2/4] Weight scheme comparison")

    for scheme in ["original", "multiply", "add"]:
        print(f"  Running: {scheme} ...")
        r = run_retrieval_with_params(
            retriever, queries, gold_docs, gold_answers,
            weight_scheme=scheme, query_aware=True,
        )
        results[f"scheme_{scheme}"] = r
        print(f"    R@10={r['recall'].get('Recall@10', 0):.4f}  R@100={r['recall'].get('Recall@100', 0):.4f}  QAFD={r['avg_qafd_time']:.3f}s")

    # ── Experiment 3: (a, b) sensitivity sweep ───────────────────────
    print("\n[3/4] (a, b) sensitivity sweep for Hybrid")

    a_values = [0.5, 1.0, 2.0]
    b_values = [0.0, 0.1, 0.25, 0.5, 1.0]

    sweep_results = {}
    for a in a_values:
        for b in b_values:
            label = f"a={a},b={b}"
            r = run_retrieval_with_params(
                retriever, queries, gold_docs, gold_answers,
                weight_scheme="original", hybrid_a=a, hybrid_b=b, query_aware=(b > 0),
            )
            sweep_results[label] = r
            r10 = r['recall'].get('Recall@10', 0)
            r100 = r['recall'].get('Recall@100', 0)
            print(f"  {label:>15}  R@10={r10:.4f}  R@100={r100:.4f}  QAFD={r['avg_qafd_time']:.3f}s")

    results["sweep"] = sweep_results

    # ── Experiment 4: Per-query diagnostics ──────────────────────────
    print("\n[4/4] Per-query diagnostics (first 5 queries)")

    # Compare query-aware vs agnostic per query
    diag_queries = queries[:5]
    diag_results = []

    for qi, q in enumerate(diag_queries):
        # Query-aware
        r_aware = run_retrieval_with_params(
            retriever, [q], [gold_docs[qi]], [gold_answers[qi]],
            weight_scheme="original", hybrid_a=1.0, hybrid_b=0.5, query_aware=True,
        )
        # Query-agnostic
        r_agnostic = run_retrieval_with_params(
            retriever, [q], [gold_docs[qi]], [gold_answers[qi]],
            weight_scheme="original", hybrid_a=1.0, hybrid_b=0.0, query_aware=False,
        )

        r10_aware = r_aware['recall'].get('Recall@10', 0)
        r10_agnostic = r_agnostic['recall'].get('Recall@10', 0)
        delta = r10_aware - r10_agnostic

        status = "HELPS" if delta > 0 else ("HURTS" if delta < 0 else "SAME")
        print(f"  Q{qi}: R@10 aware={r10_aware:.3f} agnostic={r10_agnostic:.3f}  delta={delta:+.3f} [{status}]")
        print(f"       Q: {q[:80]}...")

        diag_results.append({
            "query": q,
            "r10_aware": r10_aware,
            "r10_agnostic": r10_agnostic,
            "delta": delta,
            "status": status,
        })

    results["diagnostics"] = diag_results

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    print("\n  Ablation: Query-aware vs Query-agnostic")
    print(f"    Query-aware (Hybrid a=1,b=0.5):  R@10={results['hybrid_query_aware']['recall'].get('Recall@10',0):.4f}  R@100={results['hybrid_query_aware']['recall'].get('Recall@100',0):.4f}")
    print(f"    Query-agnostic (b=0):             R@10={results['query_agnostic']['recall'].get('Recall@10',0):.4f}  R@100={results['query_agnostic']['recall'].get('Recall@100',0):.4f}")

    print("\n  Weight scheme comparison")
    for scheme in ["original", "multiply", "add"]:
        r = results[f"scheme_{scheme}"]
        print(f"    {scheme:>10}:  R@10={r['recall'].get('Recall@10',0):.4f}  R@100={r['recall'].get('Recall@100',0):.4f}")

    print("\n  Best (a,b) from sweep:")
    best_label = max(sweep_results, key=lambda k: sweep_results[k]['recall'].get('Recall@10', 0))
    best = sweep_results[best_label]
    print(f"    {best_label}:  R@10={best['recall'].get('Recall@10',0):.4f}  R@100={best['recall'].get('Recall@100',0):.4f}")

    print("\n  Per-query diagnostics:")
    helps = sum(1 for d in diag_results if d["status"] == "HELPS")
    hurts = sum(1 for d in diag_results if d["status"] == "HURTS")
    same = sum(1 for d in diag_results if d["status"] == "SAME")
    print(f"    Query-awareness HELPS: {helps}/{len(diag_results)}, HURTS: {hurts}/{len(diag_results)}, SAME: {same}/{len(diag_results)}")

    # Save
    out_dir = os.path.join(_project_root, "experiments", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"edge_weight_ablation_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
