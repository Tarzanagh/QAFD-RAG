"""
Entry point for running multihop benchmarks (MuSiQue, HotpotQA, 2WikiMultiHopQA)
Passage-entity KG pipeline benchmark runner for QAFD-RAG.

Usage::

    python -m src.passage_entity.benchmark_runner \\
        --dataset musique \\
        --llm_model gpt-4o-mini \\
        --embedding_model nvidia-nv-embed-v2 \\
        --num_queries 100 \\
        --qafd_alpha 10.0

The script will:
    1. Load corpus and questions from ``data/multihop/``.
    2. Build (or load) the knowledge graph.
    3. Run retrieval + QA.
    4. Evaluate Recall@K, Exact Match, and F1.
"""

import argparse
import asyncio
import collections
import json
import logging
import os
import re
import string
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the QAFD-RAG root is on the path so ``src.*`` imports work
# when this file is executed as ``python -m src.passage_entity.benchmark_runner``
# ---------------------------------------------------------------------------
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---------------------------------------------------------------------------
# Bypass src/__init__.py (which imports heavy AWS deps) by registering
# src as a plain namespace package before any sub-package imports.
# ---------------------------------------------------------------------------
import types as _types
for _pkg_path in ["src", "src.retrievers", "src.passage_entity"]:
    if _pkg_path not in sys.modules:
        _m = _types.ModuleType(_pkg_path)
        _m.__path__ = [os.path.join(_project_root, *_pkg_path.split("."))]
        _m.__package__ = _pkg_path
        sys.modules[_pkg_path] = _m

# Load only the modules we actually need (no aioboto3, no AWS, no SAPIEN)
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

from src.passage_entity.config import PassageEntityConfig
from src.passage_entity.embedding_store import EmbeddingModelWrapper
from src.passage_entity.kg_builder import KGBuilder
from src.passage_entity.openie import OpenIE
from src.passage_entity.reranker import FactReranker
from src.passage_entity.retriever import PassageEntityRetriever
from src.passage_entity.prompts import make_qa_messages
from src.passage_entity.utils import QuerySolution

# ---------------------------------------------------------------------------
# Minimal OpenAI LLM + Embedding (no AWS deps, no src/llm.py)
# ---------------------------------------------------------------------------
from openai import AsyncOpenAI

# Load .env file if present (for API keys)
_dotenv_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
if os.path.isfile(_dotenv_path):
    with open(_dotenv_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                _v = _v.strip().strip('"').strip("'")
                if _k.strip() not in os.environ:
                    os.environ[_k.strip()] = _v

# Shared client — avoids "Event loop is closed" errors from abandoned clients
_openai_clients: dict = {}

def _get_client(base_url="https://api.openai.com/v1", api_key=""):
    key = (base_url, api_key)
    if key not in _openai_clients:
        _openai_clients[key] = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
        )
    return _openai_clients[key]

async def _openai_complete(model, prompt, system_prompt=None, history_messages=[],
                           base_url="https://api.openai.com/v1", api_key="", **kwargs):
    client = _get_client(base_url, api_key)
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    response = await client.chat.completions.create(model=model, messages=messages, **kwargs)
    return response.choices[0].message.content

async def _openai_embed(texts, model="text-embedding-3-small", api_key=""):
    client = _get_client(api_key=api_key)
    cleaned = [t if t.strip() else " " for t in texts]
    response = await client.embeddings.create(model=model, input=cleaned, encoding_format="float")
    return np.array([dp.embedding for dp in response.data])

logger = logging.getLogger(__name__)

# ===========================================================================
# Gold extraction helpers (from the original pipeline main_qafd.py)
# ===========================================================================

def get_gold_docs(samples: List[dict], dataset_name: str = None) -> List[List[str]]:
    gold_docs = []
    for sample in samples:
        if "supporting_facts" in sample:
            gold_titles = {item[0] for item in sample["supporting_facts"]}
            pairs = [item for item in sample["context"] if item[0] in gold_titles]
            if dataset_name and dataset_name.startswith("hotpotqa"):
                gd = [item[0] + "\n" + "".join(item[1]) for item in pairs]
            else:
                gd = [item[0] + "\n" + " ".join(item[1]) for item in pairs]
        elif "contexts" in sample:
            gd = [
                item["title"] + "\n" + item["text"]
                for item in sample["contexts"]
                if item["is_supporting"]
            ]
        elif "paragraphs" in sample:
            paras = [
                p for p in sample["paragraphs"]
                if p.get("is_supporting", True)
            ]
            gd = [
                p["title"] + "\n" + p.get("text", p.get("paragraph_text", ""))
                for p in paras
            ]
        else:
            gd = []
        gold_docs.append(list(set(gd)))
    return gold_docs


def get_gold_answers(samples: List[dict]) -> List[Set[str]]:
    answers = []
    for s in samples:
        ans = s.get("answer") or s.get("gold_ans") or s.get("reference")
        if ans is None and "obj" in s:
            ans = list(
                {s["obj"], s.get("possible_answers", ""), s.get("o_wiki_title", ""), s.get("o_aliases", "")}
            )
        if ans is None:
            ans = ""
        if isinstance(ans, str):
            ans = [ans]
        ans_set = set(ans)
        if "answer_aliases" in s:
            ans_set.update(s["answer_aliases"])
        answers.append(ans_set)
    return answers


# ===========================================================================
# Evaluation metrics
# ===========================================================================

def _normalize_answer(s: str) -> str:
    """Lower-case, remove articles, punctuation, extra whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def exact_match(prediction: str, gold_answers: Set[str]) -> float:
    pred_norm = _normalize_answer(prediction)
    return float(any(_normalize_answer(g) == pred_norm for g in gold_answers))


def f1_score(prediction: str, gold_answers: Set[str]) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = _normalize_answer(gold).split()
        common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1


def recall_at_k(
    gold_docs: List[List[str]], retrieved_docs: List[List[str]], k_list: List[int]
) -> Dict[str, float]:
    """Compute Recall@K across all queries."""
    results = {}
    for k in k_list:
        recalls = []
        for gd, rd in zip(gold_docs, retrieved_docs):
            if not gd:
                continue
            retrieved_set = set(rd[:k])
            found = sum(1 for g in gd if g in retrieved_set)
            recalls.append(found / len(gd))
        results[f"Recall@{k}"] = round(np.mean(recalls), 4) if recalls else 0.0
    return results


# ===========================================================================
# QA (reading comprehension)
# ===========================================================================

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


def run_qa(
    queries: List[QuerySolution],
    llm_func,
    qa_top_k: int = 5,
) -> List[QuerySolution]:
    """Run reading-comprehension QA over retrieved passages."""
    for qs in queries:
        passages = qs.docs[:qa_top_k]
        msgs = make_qa_messages(passages, qs.question)
        # Convert messages to single call
        system_prompt = None
        history = []
        user_prompt = ""
        for msg in msgs:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "assistant":
                history.append(msg)
            elif msg["role"] == "user":
                if user_prompt:
                    history.append({"role": "user", "content": user_prompt})
                user_prompt = msg["content"]

        try:
            response = _run_sync(
                llm_func(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    history_messages=history,
                    max_tokens=512,
                )
            )
            # Extract answer
            if "Answer:" in response:
                qs.answer = response.split("Answer:")[-1].strip()
            else:
                qs.answer = response.strip()
        except Exception as e:
            logger.error(f"QA error: {e}")
            qs.answer = ""
    return queries


def run_qa_ultradomain(
    queries: List[QuerySolution],
    llm_func,
    qa_top_k: int = 5,
) -> List[QuerySolution]:
    """Generate full responses for UltraDomain (not short answers)."""
    for qs in queries:
        passages = qs.docs[:qa_top_k]
        context = "\n\n".join(passages)
        prompt = (
            f"Based on the following context, provide a comprehensive and detailed "
            f"answer to the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {qs.question}\n\n"
            f"Answer:"
        )
        try:
            response = _run_sync(
                llm_func(prompt=prompt, max_tokens=1024)
            )
            qs.answer = response.strip()
        except Exception as e:
            logger.error(f"QA error: {e}")
            qs.answer = ""
    return queries


def run_quality_eval(
    queries: List[str],
    responses: List[str],
    llm_func,
    num_eval_rounds: int = 5,
) -> Dict[str, List[float]]:
    """Evaluate response quality using LLM scoring (same as entity-graph pipeline).

    Each response is evaluated num_eval_rounds times on 5 criteria.
    Returns dict of criterion -> list of per-query average scores.
    """
    criteria = ["comprehensiveness", "diversity", "logicality", "relevance", "coherence"]
    result = {c: [] for c in criteria}

    for i, (query, response) in enumerate(zip(queries, responses)):
        if not response:
            for c in criteria:
                result[c].append(0.0)
            continue

        criterion_scores = {c: [] for c in criteria}
        for _ in range(num_eval_rounds):
            prompt = f"""Evaluate the following response to a question based on five criteria. Rate each criterion from 0-100.

Question: {query}
Response: {response}

Please evaluate based on these criteria:
- Comprehensiveness: How much detail does the answer provide to cover all aspects and details of the question?
- Diversity: How varied and rich is the answer in providing different perspectives and insights on the question?
- Logicality: How logically does the answer respond to all parts of the question?
- Relevance: How relevant is the answer to the question, staying focused and addressing the intended topic or issue?
- Coherence: How well does the answer maintain internal logical connections between its parts, ensuring a smooth and consistent structure?

Provide scores in JSON format:
{{
    "comprehensiveness": [score],
    "diversity": [score],
    "logicality": [score],
    "relevance": [score],
    "coherence": [score]
}}"""
            try:
                eval_response = _run_sync(
                    llm_func(prompt=prompt, max_tokens=200)
                )
                import re as _re
                json_match = _re.search(r'\{.*\}', eval_response, _re.DOTALL)
                if json_match:
                    scores = json.loads(json_match.group())
                    for c in criteria:
                        if c in scores:
                            val = float(scores[c])
                            if 0 <= val <= 100:
                                criterion_scores[c].append(val)
            except Exception:
                continue

        for c in criteria:
            if criterion_scores[c]:
                result[c].append(np.mean(criterion_scores[c]))
            else:
                result[c].append(0.0)

    return result


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="QAFD-RAG passage-entity benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="musique",
                        help="Dataset name (e.g. musique, hotpotqa, 2wikimultihopqa, mix)")
    parser.add_argument("--task", type=str, default="multihop",
                        choices=["multihop", "ultradomain"],
                        help="Task type (determines data loading)")
    parser.add_argument("--num_queries", type=int, default=-1,
                        help="Number of queries (-1 = all)")
    parser.add_argument("--data_dir", type=str, default="data/multihop",
                        help="Directory with corpus/question JSON files (multihop only)")
    parser.add_argument("--save_dir", type=str, default="outputs",
                        help="Output directory")

    # LLM
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--llm_base_url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--llm_api_key", type=str, default="")

    # Embedding
    parser.add_argument("--embedding_model", type=str, default="nvidia-nv-embed-v2",
                        help="Key from QAFD-RAG embedding registry")

    # Indexing
    parser.add_argument("--force_index", action="store_true")
    parser.add_argument("--force_openie", action="store_true")
    parser.add_argument("--max_documents", type=int, default=None,
                        help="Max documents for KG building (default: all)")

    # QAFD
    parser.add_argument("--qafd_alpha", type=float, default=1.5)
    parser.add_argument("--qafd_epsilon", type=float, default=0.01)
    parser.add_argument("--qafd_max_iterations", type=int, default=500)
    parser.add_argument("--qafd_weight_scheme", type=str, default="multiply")
    parser.add_argument("--qafd_step_size", type=float, default=0.2)

    # Retrieval
    parser.add_argument("--linking_top_k", type=int, default=10)
    parser.add_argument("--retrieval_top_k", type=int, default=200)
    parser.add_argument("--passage_node_weight", type=float, default=0.05)

    # QA
    parser.add_argument("--qa_top_k", type=int, default=5)
    parser.add_argument("--skip_qa", action="store_true",
                        help="Only run retrieval, skip QA step")

    # Query-aware enhancements (all default = original behaviour)
    parser.add_argument("--sim_mode", type=str, default="normalized",
                        choices=["normalized", "relu", "relu_sq"],
                        help="Similarity contrast function (default=normalized)")
    parser.add_argument("--qa_sink_gamma", type=float, default=0.0,
                        help="Query-aware sink capacity (0=off)")
    parser.add_argument("--qa_warm_delta", type=float, default=0.0,
                        help="Query-aware seed bias (0=off)")
    parser.add_argument("--qa_warm_walk", action="store_true",
                        help="Use QA edge weights in warm-start random walk")
    parser.add_argument("--qa_warm_steps", type=int, default=2,
                        help="Number of warm-start steps (default 2)")
    parser.add_argument("--qa_accum_gamma", type=float, default=0.0,
                        help="Query-aware x accumulation boost (0=off)")
    parser.add_argument("--qa_post_lambda", type=float, default=0.0,
                        help="Post-diffusion query reranking (0=off)")
    parser.add_argument("--batch_push", action="store_true",
                        help="Use batch push-relabel (all excess nodes per iter)")

    # Reranker
    parser.add_argument("--rerank_dspy_path", type=str, default=None)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ----------------------------------------------------------------
    # Config
    # ----------------------------------------------------------------
    config = PassageEntityConfig(
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        embedding_model_key=args.embedding_model,
        dataset=args.dataset,
        save_dir=args.save_dir,
        force_index_from_scratch=args.force_index,
        force_openie_from_scratch=args.force_openie,
        linking_top_k=args.linking_top_k,
        retrieval_top_k=args.retrieval_top_k,
        passage_node_weight=args.passage_node_weight,
        qa_top_k=args.qa_top_k,
        use_qafd=True,
        qafd_alpha=args.qafd_alpha,
        qafd_epsilon=args.qafd_epsilon,
        qafd_max_iterations=args.qafd_max_iterations,
        qafd_weight_scheme=args.qafd_weight_scheme,
        qafd_step_size=args.qafd_step_size,
        sim_mode=args.sim_mode,
        qa_sink_gamma=args.qa_sink_gamma,
        qa_warm_walk=args.qa_warm_walk,
        qa_warm_steps=args.qa_warm_steps,
        qa_accum_gamma=args.qa_accum_gamma,
        batch_push=args.batch_push,
        qa_warm_delta=args.qa_warm_delta,
        qa_post_lambda=args.qa_post_lambda,
        rerank_dspy_file_path=args.rerank_dspy_path,
    )

    # ----------------------------------------------------------------
    # LLM function
    # ----------------------------------------------------------------
    _api_key = config.llm_api_key or os.environ.get("OPENAI_API_KEY", "")

    async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return await _openai_complete(
            model=config.llm_model,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=config.llm_base_url,
            api_key=_api_key,
            **kwargs,
        )

    # ----------------------------------------------------------------
    # Embedding function (must match the model used to build the KG)
    # ----------------------------------------------------------------
    emb_key = config.embedding_model_key
    if emb_key in ("openai-small", "openai-large"):
        openai_model = "text-embedding-3-small" if emb_key == "openai-small" else "text-embedding-3-large"
        async def embed_func(texts):
            return await _openai_embed(texts, model=openai_model, api_key=_api_key)
    else:
        # Local embedding model — use QAFD-RAG's embedding registry
        logger.info(f"Loading local embedding model: {emb_key}")
        _emb_cfg = type("Cfg", (), {
            "embedding_model_name": {
                "nvidia-nv-embed-v2": "nvidia/NV-Embed-v2",
                "jina-v3": "jinaai/jina-embeddings-v3",
                "gritlm": "GritLM/GritLM-7B",
            }.get(emb_key, emb_key),
            "embedding_batch_size": config.embedding_batch_size,
        })()
        _emb_src = os.path.join(_project_root, "src", "embedding_models")
        if emb_key == "nvidia-nv-embed-v2":
            _mod = _load_mod("src.embedding_models.NVEmbedV2", os.path.join(_emb_src, "NVEmbedV2.py"))
            _local_model = _mod.NVEmbedV2EmbeddingModel(_emb_cfg)
        elif emb_key == "jina-v3":
            _mod = _load_mod("src.embedding_models.JinaV3", os.path.join(_emb_src, "JinaV3.py"))
            _local_model = _mod.JinaV3EmbeddingModel(_emb_cfg)
        elif emb_key == "gritlm":
            _mod = _load_mod("src.embedding_models.GritLM", os.path.join(_emb_src, "GritLM.py"))
            _local_model = _mod.GritLMEmbeddingModel(_emb_cfg)
        else:
            raise ValueError(f"Unknown embedding model: {emb_key}")
        async def embed_func(texts):
            return np.array(_local_model.batch_encode(texts))
    embedding_model = EmbeddingModelWrapper(embed_func, batch_size=config.embedding_batch_size)

    # ----------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------
    if args.task == "ultradomain":
        # UltraDomain: load from HuggingFace, each record has context + input
        from datasets import load_dataset as hf_load_dataset

        dataset_file = f"{args.dataset}.jsonl"
        logger.info(f"Loading UltraDomain dataset: {dataset_file}")
        hf_dataset = hf_load_dataset(
            "TommyChien/UltraDomain", data_files=dataset_file, split="train"
        )

        num_q = args.num_queries if args.num_queries > 0 else len(hf_dataset)
        num_q = min(num_q, len(hf_dataset))

        # Each record's context becomes the corpus.
        # UltraDomain contexts can be very long (30K+ chars), so we chunk them
        # into ~500-word passages to fit embedding model token limits.
        docs = []
        chunk_size = 500  # words per chunk
        chunk_overlap = 50  # word overlap between chunks
        for i in range(num_q):
            ctx = hf_dataset[i].get("context", "")
            if not ctx:
                continue
            words = ctx.split()
            if len(words) <= chunk_size:
                docs.append(ctx)
            else:
                for start in range(0, len(words), chunk_size - chunk_overlap):
                    chunk = " ".join(words[start : start + chunk_size])
                    if chunk.strip():
                        docs.append(chunk)

        all_queries = [hf_dataset[i]["input"] for i in range(num_q)]
        samples = [dict(hf_dataset[i]) for i in range(num_q)]
        gold_answers = [
            set(s.get("answers", [s.get("label", "")])) for s in samples
        ]
        gold_docs = None  # UltraDomain has no gold supporting docs

    else:
        # Multihop: load from local JSON files
        corpus_path = os.path.join(args.data_dir, f"{args.dataset}_corpus.json")
        questions_path = os.path.join(args.data_dir, f"{args.dataset}.json")

        logger.info(f"Loading corpus from {corpus_path}")
        with open(corpus_path) as f:
            corpus = json.load(f)
        docs = [f"{d['title']}\n{d['text']}" for d in corpus]
        if args.max_documents and args.max_documents < len(docs):
            docs = docs[:args.max_documents]
            logger.info(f"Limited to {args.max_documents} documents for KG building")

        logger.info(f"Loading questions from {questions_path}")
        with open(questions_path) as f:
            samples = json.load(f)

        all_queries = [s["question"] for s in samples]
        if args.num_queries > 0:
            all_queries = all_queries[: args.num_queries]
            samples = samples[: args.num_queries]

        gold_answers = get_gold_answers(samples)
        try:
            gold_docs = get_gold_docs(samples, args.dataset)
        except Exception:
            gold_docs = None

    print("=" * 70)
    print(f"  Graph type:  passage-entity")
    print(f"  Task:        {args.task}")
    print(f"  Dataset:     {args.dataset}")
    print(f"  Queries:     {len(all_queries)}")
    print(f"  Corpus:      {len(docs)} documents")
    print(f"  LLM:         {config.llm_model}")
    print(f"  Embedding:   {config.embedding_model_key}")
    print(f"  QAFD alpha:  {config.qafd_alpha}")
    print("=" * 70)

    # ----------------------------------------------------------------
    # Build / load KG
    # ----------------------------------------------------------------
    openie = OpenIE(llm_func)
    builder = KGBuilder(config, embedding_model, openie)

    if builder.graph.vcount() > 0 and not config.force_index_from_scratch:
        logger.info(f"Using existing KG: {config.working_dir} "
                     f"({builder.graph.vcount()} nodes, {builder.graph.ecount()} edges)")
    else:
        logger.info("Building KG from scratch ...")
        builder.index(docs)

    # ----------------------------------------------------------------
    # Retriever
    # ----------------------------------------------------------------
    reranker = FactReranker(llm_func, dspy_file_path=config.rerank_dspy_file_path)
    retriever = PassageEntityRetriever(
        config=config,
        embedding_model=embedding_model,
        reranker=reranker,
        graph=builder.graph,
        chunk_embedding_store=builder.chunk_embedding_store,
        entity_embedding_store=builder.entity_embedding_store,
        fact_embedding_store=builder.fact_embedding_store,
        openie_results_path=builder.openie_results_path,
    )

    logger.info("Running retrieval ...")
    retrieval_results = retriever.retrieve(
        queries=all_queries, num_to_retrieve=config.retrieval_top_k
    )

    # ----------------------------------------------------------------
    # Retrieval evaluation
    # ----------------------------------------------------------------
    if gold_docs is not None:
        k_list = [1, 2, 5, 10, 20, 50, 100, 200]
        retrieved = [r.docs for r in retrieval_results]
        retrieval_metrics = recall_at_k(gold_docs, retrieved, k_list)
        print("\n--- Retrieval Metrics ---")
        for metric, val in retrieval_metrics.items():
            print(f"  {metric}: {val}")
    else:
        retrieval_metrics = {}

    # ----------------------------------------------------------------
    # QA + Evaluation (task-aware)
    # ----------------------------------------------------------------
    avg_em, avg_f1 = None, None
    quality_scores = None

    if not args.skip_qa:
        logger.info("Running QA ...")

        if args.task == "ultradomain":
            # UltraDomain: generate full responses, evaluate with quality scores
            retrieval_results = run_qa_ultradomain(
                retrieval_results, llm_func, qa_top_k=config.qa_top_k
            )
            # Quality evaluation (same as entity-graph pipeline)
            quality_scores = run_quality_eval(
                all_queries, [qs.answer for qs in retrieval_results], llm_func
            )
            if quality_scores:
                print("\n--- Quality Metrics ---")
                overall = []
                for criterion, scores in quality_scores.items():
                    avg = np.mean(scores)
                    std = np.std(scores)
                    print(f"  {criterion:<25} {avg:.2f} +/- {std:.2f}")
                    overall.append(avg)
                print(f"  {'Overall Average':<25} {np.mean(overall):.2f}")
        else:
            # Multihop: generate short answers, evaluate with F1/EM
            retrieval_results = run_qa(retrieval_results, llm_func, qa_top_k=config.qa_top_k)

            em_scores, f1_scores = [], []
            for qs, ga in zip(retrieval_results, gold_answers):
                qs.gold_answers = list(ga)
                em_scores.append(exact_match(qs.answer or "", ga))
                f1_scores.append(f1_score(qs.answer or "", ga))

            avg_em = round(np.mean(em_scores), 4)
            avg_f1 = round(np.mean(f1_scores), 4)
            print("\n--- QA Metrics ---")
            print(f"  Exact Match: {avg_em}")
            print(f"  F1 Score:    {avg_f1}")

    # ----------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------
    os.makedirs(config.working_dir, exist_ok=True)
    results_path = os.path.join(config.working_dir, f"results_{args.dataset}.json")
    output = {
        "graph_type": "passage-entity",
        "dataset": args.dataset,
        "task": args.task,
        "num_queries": len(all_queries),
        "retrieval_metrics": retrieval_metrics,
        "qa_em": avg_em,
        "qa_f1": avg_f1,
        "quality_scores": quality_scores,
        "config": {
            "llm_model": config.llm_model,
            "embedding_model_key": config.embedding_model_key,
            "qafd_alpha": config.qafd_alpha,
            "qafd_epsilon": config.qafd_epsilon,
            "qafd_max_iterations": config.qafd_max_iterations,
            "qafd_weight_scheme": config.qafd_weight_scheme,
            "linking_top_k": config.linking_top_k,
            "retrieval_top_k": config.retrieval_top_k,
        },
        "per_query": [qs.to_dict() for qs in retrieval_results],
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
