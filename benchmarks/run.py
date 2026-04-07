#!/usr/bin/env python3
"""
Unified Benchmark Runner for QAFD-RAG
======================================

Supports two graph types (always with Query-Aware Flow Diffusion):

  - **passage-entity**: Entities + passages + facts as nodes, synonymy edges.
    Flow diffusion reaches passages directly. Default for multihop.

  - **entity**: Classic KG with entity + relationship nodes.
    Passages are looked up after graph traversal. Default for other tasks.

Usage::

    # Multihop (auto-selects passage-entity graph)
    python benchmarks/run.py --task multihop --dataset musique

    # Override graph type
    python benchmarks/run.py --task multihop --dataset musique --graph_type entity

    # Ultradomain (auto-selects entity graph)
    python benchmarks/run.py --task ultradomain --dataset mix

    # Text2SQL
    python benchmarks/run.py --task text2sql --dataset spider2-lite

    # Build KG only
    python benchmarks/run.py --task multihop --dataset musique --build_only
"""

import argparse
import asyncio
import os
import sys

QAFD_RAG_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, QAFD_RAG_HOME)

# ── Task → dataset mapping ──────────────────────────────────────────────

TASK_DATASETS = {
    "multihop": ["musique", "hotpotqa", "2wikimultihopqa"],
    "ultradomain": [
        "agriculture", "biology", "cs", "finance", "legal",
        "math", "medicine", "mix", "music", "philosophy",
        "physics", "psychology",
    ],
    "text2sql": ["spider2-lite", "bird"],
    "summarization": ["squality"],
}

# ── Default graph type per task ─────────────────────────────────────────

TASK_DEFAULT_GRAPH_TYPE = {
    "multihop": "passage-entity",
    "ultradomain": "entity",
    "text2sql": "entity",
    "summarization": "entity",
}

# ── Default QAFD parameters per graph type ──────────────────────────────
# These come from proven successful runs in each pipeline.

GRAPH_TYPE_DEFAULTS = {
    "passage-entity": {
        "alpha": 2.0,
        "epsilon": 0.01,
        "max_iterations": 500,
        "step_size": 0.2,
        "weight_scheme": "original",
        "linking_top_k": 5,           # fact seeds
        "passage_node_weight": 0.05,
        "retrieval_top_k": 200,
    },
    "entity": {
        "alpha": 2.0,
        "epsilon": 0.01,
        "max_iterations": 500,
        "step_size": 0.2,
        "weight_scheme": "original",
        "max_source_nodes": 20,
        "min_flow_threshold": 0.1,
    },
}


def run_passage_entity(args):
    """Run benchmark using passage-entity graph (passage_entity)."""
    # Bypass src/__init__.py (heavy AWS deps)
    import types as _types
    for _pkg_path in ["src", "src.retrievers", "src.passage_entity"]:
        if _pkg_path not in sys.modules:
            _m = _types.ModuleType(_pkg_path)
            _m.__path__ = [os.path.join(QAFD_RAG_HOME, *_pkg_path.split("."))]
            _m.__package__ = _pkg_path
            sys.modules[_pkg_path] = _m

    import importlib.util as _ilu
    def _load_mod(fqn, filepath):
        spec = _ilu.spec_from_file_location(fqn, filepath)
        mod = _ilu.module_from_spec(spec)
        sys.modules[fqn] = mod
        spec.loader.exec_module(mod)
        return mod

    _src = os.path.join(QAFD_RAG_HOME, "src")
    _load_mod("src.retrievers.base", os.path.join(_src, "retrievers", "base.py"))
    _load_mod("src.retrievers.flow_diffusion", os.path.join(_src, "retrievers", "flow_diffusion.py"))

    # Import after module setup
    from src.passage_entity.benchmark_runner import main as pe_main

    # Build sys.argv for the sub-module
    sub_argv = [
        "benchmark_runner",
        "--task", args.task,
        "--dataset", args.dataset,
        "--data_dir", os.path.join(QAFD_RAG_HOME, "data", "multihop"),
        "--embedding_model", args.embedding,
        "--llm_model", args.llm,
        "--num_queries", str(args.questions),
        "--qafd_alpha", str(args.alpha),
        "--qafd_epsilon", str(args.epsilon),
        "--qafd_max_iterations", str(args.max_iterations),
        "--qafd_step_size", str(args.step_size),
        "--qafd_weight_scheme", str(args.weight_scheme),
        "--linking_top_k", str(args.linking_top_k),
        "--passage_node_weight", str(args.passage_node_weight),
        "--retrieval_top_k", str(args.retrieval_top_k),
    ]
    if args.skip_qa:
        sub_argv.append("--skip_qa")
    if getattr(args, 'batch_push', False):
        sub_argv.append("--batch_push")
    if args.force_build:
        sub_argv.append("--force_index")
        sub_argv.append("--force_openie")
    if args.max_documents:
        sub_argv.extend(["--max_documents", str(args.max_documents)])

    old_argv = sys.argv
    sys.argv = sub_argv
    try:
        pe_main()
    finally:
        sys.argv = old_argv


def run_entity(args):
    """Run benchmark using entity graph (original QAFD-RAG pipeline)."""
    import nest_asyncio
    nest_asyncio.apply()

    task = args.task

    if task == "multihop":
        from benchmarks.multihop.benchmark_multihop import MultiHopBenchmark

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: Set OPENAI_API_KEY environment variable")
            return

        benchmark = MultiHopBenchmark(args.dataset, api_key, args.embedding, args.llm)

        if args.build_only:
            asyncio.run(benchmark.build_kg(max_documents=args.max_documents))
            return

        result = asyncio.run(benchmark.run_benchmark(
            question_count=args.questions,
            force_build=args.force_build,
            max_documents=args.max_documents,
            mode="hybrid",
            max_source_nodes=args.max_source_nodes,
            min_flow_threshold=args.min_flow_threshold,
            alpha=args.alpha,
        ))
        benchmark.save_results(result)

    elif task == "ultradomain":
        # Delegate to ultradomain's own argparse
        # Ultradomain expects dataset as "mix.jsonl" format
        ud_dataset = args.dataset if args.dataset.endswith(".jsonl") else f"{args.dataset}.jsonl"
        sub_argv = [
            "benchmark_ultradomain",
            "--dataset", ud_dataset,
            "--questions", str(args.questions),
            "--embedding", args.embedding,
            "--llm", args.llm,
        ]
        if args.force_build:
            sub_argv.append("--force-build")
        if args.build_only:
            sub_argv.append("--build")
        if args.max_documents:
            sub_argv.extend(["--max-documents", str(args.max_documents)])

        old_argv = sys.argv
        sys.argv = sub_argv
        try:
            from benchmarks.ultradomain.benchmark_ultradomain import main as ultra_main
            asyncio.run(ultra_main())
        finally:
            sys.argv = old_argv

    elif task == "text2sql":
        sub_argv = ["benchmark_text2sql"]
        if args.max_documents:
            sub_argv.extend(["--max-documents", str(args.max_documents)])

        old_argv = sys.argv
        sys.argv = sub_argv
        try:
            from benchmarks.text2sql.benchmark_text2sql import main as text2sql_main
            asyncio.run(text2sql_main())
        finally:
            sys.argv = old_argv

    elif task == "summarization":
        sub_argv = [
            "benchmark_summarization",
            "--dataset", args.dataset,
            "--questions", str(args.questions),
            "--embedding", args.embedding,
            "--llm", args.llm,
        ]
        if args.force_build:
            sub_argv.append("--force-build")
        if args.build_only:
            sub_argv.append("--build")
        if args.max_documents:
            sub_argv.extend(["--max-documents", str(args.max_documents)])

        old_argv = sys.argv
        sys.argv = sub_argv
        try:
            from benchmarks.summarization.benchmark_summarization import main as summ_main
            asyncio.run(summ_main())
        finally:
            sys.argv = old_argv

    else:
        print(f"ERROR: Unknown task '{task}'")


def main():
    parser = argparse.ArgumentParser(
        description="QAFD-RAG Unified Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multihop with passage-entity graph (default)
  python benchmarks/run.py --task multihop --dataset musique

  # Multihop with entity graph (override)
  python benchmarks/run.py --task multihop --dataset musique --graph_type entity

  # Ultradomain with entity graph (default)
  python benchmarks/run.py --task ultradomain --dataset mix

  # Retrieval only (skip QA)
  python benchmarks/run.py --task multihop --dataset musique --skip_qa

  # Build KG only
  python benchmarks/run.py --task multihop --dataset musique --build_only
""",
    )

    # ── Required ────────────────────────────────────────────────────────
    parser.add_argument("--task", type=str, required=True,
                        choices=["multihop", "ultradomain", "text2sql", "summarization"])
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g. musique, hotpotqa, mix, spider2-lite)")

    # ── Graph type ──────────────────────────────────────────────────────
    parser.add_argument("--graph_type", type=str, default=None,
                        choices=["passage-entity", "entity"],
                        help="Graph type (default: passage-entity for multihop, entity for others)")

    # ── Model ───────────────────────────────────────────────────────────
    parser.add_argument("--llm", type=str, default="gpt-4o-mini")
    parser.add_argument("--embedding", type=str, default=None,
                        help="Embedding model (auto-selected per task: "
                             "nvidia-nv-embed-v2 for multihop, openai-small for others)")

    # ── Run control ─────────────────────────────────────────────────────
    parser.add_argument("--questions", type=int, default=100)
    parser.add_argument("--max_documents", type=int, default=None)
    parser.add_argument("--build_only", action="store_true",
                        help="Build KG only, skip benchmark")
    parser.add_argument("--force_build", action="store_true",
                        help="Rebuild KG even if it exists")
    parser.add_argument("--skip_qa", action="store_true",
                        help="Run retrieval only, skip QA (passage-entity only)")
    parser.add_argument("--batch_push", action="store_true",
                        help="Batch push-relabel (process all excess nodes per iter)")

    # ── QAFD parameters (shared) ────────────────────────────────────────
    parser.add_argument("--alpha", type=float, default=None,
                        help="QAFD alpha (default: 2.0)")
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--step_size", type=float, default=None)
    parser.add_argument("--weight_scheme", type=str, default=None,
                        choices=["original", "multiply", "add"])

    # ── Passage-entity specific ─────────────────────────────────────────
    parser.add_argument("--linking_top_k", type=int, default=None,
                        help="Number of fact seeds (passage-entity only)")
    parser.add_argument("--passage_node_weight", type=float, default=None,
                        help="Passage node weight in seed computation (passage-entity only)")
    parser.add_argument("--retrieval_top_k", type=int, default=None,
                        help="Number of passages to retrieve (passage-entity only)")

    # ── Entity graph specific ───────────────────────────────────────────
    parser.add_argument("--max_source_nodes", type=int, default=None,
                        help="Max source nodes for flow diffusion (entity only)")
    parser.add_argument("--min_flow_threshold", type=float, default=None,
                        help="Min flow threshold for clusters (entity only)")

    args = parser.parse_args()

    # ── Resolve graph type ──────────────────────────────────────────────
    if args.graph_type is None:
        args.graph_type = TASK_DEFAULT_GRAPH_TYPE[args.task]

    # ── Resolve embedding (must match pre-built KGs on HuggingFace) ────
    if args.embedding is None:
        _task_embeddings = {
            "multihop": "nvidia-nv-embed-v2",
            "ultradomain": "openai-small",
            "text2sql": "openai-small",
            "summarization": "openai-small",
        }
        args.embedding = _task_embeddings[args.task]

    # ── Validate dataset ────────────────────────────────────────────────
    valid = TASK_DATASETS.get(args.task, [])
    if args.dataset not in valid and args.dataset != "all":
        print(f"ERROR: Unknown dataset '{args.dataset}' for task '{args.task}'")
        print(f"  Valid: {valid}")
        return

    # ── Apply graph-type defaults for unset params ──────────────────────
    defaults = GRAPH_TYPE_DEFAULTS[args.graph_type]
    for key, default_val in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, default_val)

    # ── Print config ────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  QAFD-RAG Benchmark")
    print(f"{'=' * 70}")
    print(f"  Task:         {args.task}")
    print(f"  Dataset:      {args.dataset}")
    print(f"  Graph type:   {args.graph_type}")
    print(f"  LLM:          {args.llm}")
    print(f"  Embedding:    {args.embedding}")
    print(f"  QAFD alpha:   {args.alpha}")
    print(f"  Questions:    {args.questions}")
    print(f"{'=' * 70}\n")

    # ── Dispatch ────────────────────────────────────────────────────────
    if args.graph_type == "passage-entity":
        run_passage_entity(args)
    else:
        run_entity(args)


if __name__ == "__main__":
    main()
