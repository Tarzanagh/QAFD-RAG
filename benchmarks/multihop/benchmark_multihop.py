#!/usr/bin/env python3
"""
Unified Multi-hop QA Benchmark for QAFD-RAG
Supports: MuSiQue, HotpotQA, 2WikiMultiHopQA
"""

import os
import sys
import asyncio
import time
import json
import re
import string
import logging
from typing import List, Dict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from collections import Counter
import numpy as np

# Suppress verbose logging
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("QAFD_RAG").setLevel(logging.ERROR)
logging.getLogger("nano-vectordb").setLevel(logging.ERROR)
logging.getLogger("OpenAI").setLevel(logging.ERROR)

# Add QAFD-RAG to path
QAFD_RAG_HOME = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, QAFD_RAG_HOME)

import nest_asyncio
nest_asyncio.apply()

# Dataset configurations
DATASETS = {
    "musique": {
        "name": "MuSiQue",
        "data_file": "musique.json",
        "corpus_file": "musique_corpus.json",
        "kg_dir": "musique",
    },
    "hotpotqa": {
        "name": "HotpotQA",
        "data_file": "hotpotqa.json",
        "corpus_file": "hotpotqa_corpus.json",
        "kg_dir": "hotpotqa",
    },
    "2wikimultihopqa": {
        "name": "2WikiMultiHopQA",
        "data_file": "2wikimultihopqa.json",
        "corpus_file": "2wikimultihopqa_corpus.json",
        "kg_dir": "2wikimultihopqa",
    },
}


def print_header(title: str, width: int = 70):
    """Print a formatted header"""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_config(items: Dict[str, str], width: int = 70):
    """Print configuration items"""
    print(f"{'─' * width}")
    for key, value in items.items():
        print(f"  {key:<20} {value}")
    print(f"{'─' * width}")


def print_progress(current: int, total: int, prefix: str = "", width: int = 40):
    """Print a progress bar"""
    percent = current / total
    filled = int(width * percent)
    bar = '█' * filled + '░' * (width - filled)
    print(f"\r  {prefix} [{bar}] {current}/{total} ({percent*100:.1f}%)", end='', flush=True)
    if current == total:
        print()


def print_metric(name: str, value: float, std: float = None, width: int = 25):
    """Print a metric with optional std"""
    if std is not None:
        print(f"  {name:<{width}} {value:.4f} ± {std:.4f}")
    else:
        print(f"  {name:<{width}} {value:.4f}")


@dataclass
class BenchmarkResult:
    """Benchmark result for multi-hop QA"""
    model_name: str
    dataset_name: str
    total_questions: int
    success_count: int
    total_time: float
    kg_build_time: float
    query_time: float
    avg_time_per_question: float
    f1_score_mean: float
    f1_score_std: float
    exact_match_mean: float
    exact_match_std: float
    f1_scores: List[float] = None
    exact_match_scores: List[float] = None
    responses: List[str] = None
    questions: List[str] = None
    gold_answers: List[List[str]] = None
    error_message: str = ""


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison"""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(answer))))


def compute_f1(gold: str, predicted: str) -> float:
    """Compute F1 score between gold and predicted answers"""
    gold_tokens = normalize_answer(gold).split()
    predicted_tokens = normalize_answer(predicted).split()
    common = Counter(predicted_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(predicted_tokens) if predicted_tokens else 0.0
    recall = 1.0 * num_same / len(gold_tokens) if gold_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_exact_match(gold: str, predicted: str) -> float:
    """Compute exact match score"""
    return 1.0 if normalize_answer(gold) == normalize_answer(predicted) else 0.0


def get_gold_answers(samples):
    """Extract gold answers from samples"""
    gold_answers = []
    for sample in samples:
        if 'answer' in sample:
            gold_ans = sample['answer']
        elif 'reference' in sample:
            gold_ans = sample['reference']
        else:
            gold_ans = "Unknown"
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        elif not isinstance(gold_ans, list):
            gold_ans = [str(gold_ans)]
        gold_answers.append(gold_ans)
    return gold_answers


class MultiHopBenchmark:
    def __init__(self, dataset: str, api_key: str, embedding_model: str = "openai-small",
                 llm_model: str = "gpt-4o-mini"):
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}. Choose from: {list(DATASETS.keys())}")

        self.dataset = dataset
        self.config = DATASETS[dataset]
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"

    def _get_working_dir(self) -> str:
        return os.path.join(QAFD_RAG_HOME, "kg", "multihop", f"{self.config['kg_dir']}_{self.embedding_model}")

    def _kg_exists(self, working_dir: str) -> bool:
        kg_files = [
            os.path.join(working_dir, "vdb_entities.json"),
            os.path.join(working_dir, "vdb_chunks.json"),
            os.path.join(working_dir, "kv_store_full_docs.json"),
        ]
        return all(os.path.exists(f) for f in kg_files)

    def _get_llm_func(self):
        from src import llm
        llm_funcs = {
            "gpt-4o-mini": llm.gpt_4o_mini_complete,
            "gpt-4o": llm.gpt_4o_complete,
            "gpt-oss-120b": llm.gpt_oss_120b_complete,
            "gpt-5": llm.gpt_5_complete,
            "gpt-5-mini": llm.gpt_5_mini_complete,
            "gpt-5-nano": llm.gpt_5_nano_complete,
        }
        return llm_funcs.get(self.llm_model, llm.gpt_4o_mini_complete)

    def _ensure_data_file(self, filename: str) -> str:
        """Return path to data file, downloading from HuggingFace if missing."""
        data_dir = os.path.join(QAFD_RAG_HOME, "data", "multihop")
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"  Downloading {filename} from osunlp/HippoRAG_2...", end=" ", flush=True)
            from huggingface_hub import hf_hub_download
            os.makedirs(data_dir, exist_ok=True)
            hf_hub_download(
                repo_id="osunlp/HippoRAG_2",
                filename=filename,
                repo_type="dataset",
                local_dir=data_dir,
            )
            print("done")
        return filepath

    def _load_dataset(self) -> List[Dict]:
        dataset_path = self._ensure_data_file(self.config["data_file"])
        with open(dataset_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        return samples

    def _load_corpus(self) -> List[str]:
        corpus_path = self._ensure_data_file(self.config["corpus_file"])
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
        docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
        return docs

    async def build_kg(self, max_documents: int = None) -> bool:
        """Build KG only (no benchmark)"""
        from src.QAFD_RAG import QAFD_RAG

        print_header(f"QAFD-RAG Knowledge Graph Builder")
        print_config({
            "Dataset": self.config['name'],
            "Embedding": self.embedding_model,
            "LLM": self.llm_model,
            "Working Dir": self._get_working_dir()
        })

        working_dir = self._get_working_dir()
        os.makedirs(working_dir, exist_ok=True)

        llm_func = self._get_llm_func()
        rag = QAFD_RAG(
            working_dir=working_dir,
            llm_model_func=llm_func,
            llm_model_name=self.llm_model,
            embedding_model_key=self.embedding_model,
            enable_llm_cache=True,
        )

        print("\n  Loading corpus...", end=" ", flush=True)
        docs = self._load_corpus()
        docs_to_process = min(max_documents, len(docs)) if max_documents else len(docs)
        print(f"done ({len(docs)} documents available)")

        print(f"\n  Building KG from {docs_to_process} documents...")
        start_time = time.time()

        for i, doc in enumerate(docs[:docs_to_process]):
            print_progress(i + 1, docs_to_process, "Progress")
            await rag.ainsert(doc)

        build_time = time.time() - start_time

        print_header("Build Complete")
        print(f"  Documents processed:  {docs_to_process}")
        print(f"  Time elapsed:         {build_time:.2f}s")
        print(f"  Avg per document:     {build_time/docs_to_process:.2f}s")
        print(f"  Output directory:     {working_dir}")
        print()

        return True

    async def run_benchmark(self, question_count: int = 100, force_build: bool = False,
                           max_documents: int = None, mode: str = "hybrid",
                           max_source_nodes: int = 20, min_flow_threshold: float = 0.1,
                           alpha: float = 0.5) -> BenchmarkResult:
        """Run benchmark"""
        from src.QAFD_RAG import QAFD_RAG, QueryParam

        print_header(f"QAFD-RAG Multi-hop QA Benchmark")
        print_config({
            "Dataset": self.config['name'],
            "Questions": str(question_count),
            "Embedding": self.embedding_model,
            "LLM": self.llm_model,
            "Mode": mode,
            "Max Nodes": str(max_source_nodes),
            "Threshold": str(min_flow_threshold),
            "Alpha": str(alpha)
        })

        working_dir = self._get_working_dir()
        os.makedirs(working_dir, exist_ok=True)

        llm_func = self._get_llm_func()
        rag = QAFD_RAG(
            working_dir=working_dir,
            llm_model_func=llm_func,
            llm_model_name=self.llm_model,
            embedding_model_key=self.embedding_model,
            enable_llm_cache=True,
        )

        # Check if KG exists or needs to be built
        kg_build_time = 0.0
        if self._kg_exists(working_dir) and not force_build:
            print(f"\n  Using existing KG: {working_dir}")
        else:
            print("\n  Loading corpus...", end=" ", flush=True)
            docs = self._load_corpus()
            docs_to_process = min(max_documents, len(docs)) if max_documents else len(docs)
            print(f"done ({docs_to_process} documents)")

            print(f"  Building KG...")
            start_time = time.time()
            for i, doc in enumerate(docs[:docs_to_process]):
                print_progress(i + 1, docs_to_process, "Progress")
                await rag.ainsert(doc)
            kg_build_time = time.time() - start_time
            print(f"  KG built in {kg_build_time:.2f}s")

        # Load dataset
        print("\n  Loading dataset...", end=" ", flush=True)
        samples = self._load_dataset()
        samples = samples[:question_count]
        questions = [s['question'] for s in samples]
        gold_answers = get_gold_answers(samples)
        print(f"done ({len(questions)} questions)")

        # Run queries
        print(f"\n  Running queries...")
        start_time = time.time()
        responses = []
        success_count = 0

        for i, question in enumerate(questions):
            try:
                print_progress(i + 1, len(questions), "Progress")
                query_param = QueryParam(
                    mode=mode,
                    max_source_nodes=max_source_nodes,
                    min_flow_threshold=min_flow_threshold,
                    alpha=alpha,
                    response_type="Brief, accurate answer (maximum 14 words)."
                )
                response = await rag.aquery(question, query_param)
                if response and len(response.split()) > 14:
                    response = " ".join(response.split()[:14])
                responses.append(response)
                success_count += 1
            except Exception as e:
                responses.append("")

        query_time = time.time() - start_time

        # Calculate metrics
        f1_scores = []
        em_scores = []
        for gold_list, predicted in zip(gold_answers, responses):
            if not predicted:
                f1_scores.append(0.0)
                em_scores.append(0.0)
                continue
            f1_scores.append(max(compute_f1(g, predicted) for g in gold_list))
            em_scores.append(max(compute_exact_match(g, predicted) for g in gold_list))

        result = BenchmarkResult(
            model_name="QAFD_RAG",
            dataset_name=self.dataset,
            total_questions=len(questions),
            success_count=success_count,
            total_time=kg_build_time + query_time,
            kg_build_time=kg_build_time,
            query_time=query_time,
            avg_time_per_question=query_time / len(questions) if questions else 0,
            f1_score_mean=float(np.mean(f1_scores)) if f1_scores else 0,
            f1_score_std=float(np.std(f1_scores)) if f1_scores else 0,
            exact_match_mean=float(np.mean(em_scores)) if em_scores else 0,
            exact_match_std=float(np.std(em_scores)) if em_scores else 0,
            f1_scores=f1_scores,
            exact_match_scores=em_scores,
            responses=responses,
            questions=questions,
            gold_answers=gold_answers,
        )

        self.print_results(result)
        return result

    def print_results(self, result: BenchmarkResult):
        """Print benchmark results"""
        print_header(f"Results: {self.config['name']}")

        print("\n  PERFORMANCE")
        print(f"  {'─' * 40}")
        print(f"  {'Questions':<25} {result.total_questions}")
        print(f"  {'Successful':<25} {result.success_count}/{result.total_questions}")
        print(f"  {'KG Build Time':<25} {result.kg_build_time:.2f}s")
        print(f"  {'Query Time':<25} {result.query_time:.2f}s")
        print(f"  {'Avg per Question':<25} {result.avg_time_per_question:.2f}s")

        print("\n  ACCURACY METRICS")
        print(f"  {'─' * 40}")
        print_metric("F1 Score", result.f1_score_mean, result.f1_score_std)
        print_metric("Exact Match", result.exact_match_mean, result.exact_match_std)
        print()

    def save_results(self, result: BenchmarkResult):
        """Save results as two separate files: eval metrics and generated responses"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(QAFD_RAG_HOME, "results", "multihop")
        os.makedirs(results_dir, exist_ok=True)

        eval_file = os.path.join(results_dir, f"{self.dataset}_{timestamp}_eval.json")
        output_file = os.path.join(results_dir, f"{self.dataset}_{timestamp}_responses.json")

        # --- Eval file: metrics and timing ---
        eval_data = {
            "timestamp": datetime.now().isoformat(),
            "model": result.model_name,
            "llm": self.llm_model,
            "embedding": self.embedding_model,
            "dataset": result.dataset_name,
            "performance": {
                "total_questions": result.total_questions,
                "success_count": result.success_count,
                "kg_build_time": result.kg_build_time,
                "query_time": result.query_time,
                "total_time": result.total_time,
                "avg_time_per_question": result.avg_time_per_question,
            },
            "metrics": {
                "f1_score_mean": result.f1_score_mean,
                "f1_score_std": result.f1_score_std,
                "exact_match_mean": result.exact_match_mean,
                "exact_match_std": result.exact_match_std,
            },
            "per_question_f1": result.f1_scores,
            "per_question_em": result.exact_match_scores,
            "error": result.error_message,
        }

        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, indent=2, ensure_ascii=False)

        # --- Responses file: questions + generated answers + gold answers ---
        output_entries = []
        if result.responses:
            for i, response in enumerate(result.responses):
                entry = {
                    "id": i + 1,
                    "question": result.questions[i] if result.questions else "",
                    "generated_answer": response,
                    "gold_answers": result.gold_answers[i] if result.gold_answers else [],
                    "f1": result.f1_scores[i] if result.f1_scores else None,
                    "exact_match": result.exact_match_scores[i] if result.exact_match_scores else None,
                }
                output_entries.append(entry)

        output_data = {
            "timestamp": datetime.now().isoformat(),
            "model": result.model_name,
            "llm": self.llm_model,
            "embedding": self.embedding_model,
            "dataset": result.dataset_name,
            "num_responses": len(output_entries),
            "responses": output_entries,
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"  Eval saved:      {eval_file}")
        print(f"  Responses saved: {output_file}\n")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="QAFD_RAG Multi-hop QA Benchmark")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["musique", "hotpotqa", "2wikimultihopqa"])
    parser.add_argument("--questions", type=int, default=100)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--embedding", type=str, default="openai-small",
                        choices=["openai-small", "openai-large", "jina-v3", "gritlm", "nvidia-nv-embed-v2"])
    parser.add_argument("--llm", type=str, default="gpt-4o-mini",
                        choices=["gpt-4o-mini", "gpt-4o", "gpt-oss-120b", "gpt-5", "gpt-5-mini", "gpt-5-nano"])
    parser.add_argument("--mode", type=str, default="hybrid",
                        choices=["local", "global", "hybrid"])
    parser.add_argument("--max-source-nodes", type=int, default=20)
    parser.add_argument("--min-flow-threshold", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.5)

    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        return

    benchmark = MultiHopBenchmark(args.dataset, api_key, args.embedding, args.llm)

    if args.build:
        await benchmark.build_kg(max_documents=args.max_documents)
        return

    result = await benchmark.run_benchmark(
        question_count=args.questions,
        force_build=args.force_build,
        max_documents=args.max_documents,
        mode=args.mode,
        max_source_nodes=args.max_source_nodes,
        min_flow_threshold=args.min_flow_threshold,
        alpha=args.alpha
    )
    benchmark.save_results(result)


if __name__ == "__main__":
    asyncio.run(main())
