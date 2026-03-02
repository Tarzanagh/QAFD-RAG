#!/usr/bin/env python3
"""
UltraDomain Benchmark for QAFD-RAG
Quality evaluation using GPT-4o as judge
"""

import os
import sys
import asyncio
import time
import json
import logging
import nest_asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Suppress verbose logging
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("QAFD_RAG").setLevel(logging.ERROR)
logging.getLogger("nano-vectordb").setLevel(logging.ERROR)
logging.getLogger("OpenAI").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)

# Add QAFD-RAG to path (so 'src' becomes the package)
QAFD_RAG_HOME = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, QAFD_RAG_HOME)

nest_asyncio.apply()

# Define datasets
DATASETS = ["mix.jsonl"]


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

@dataclass
class BenchmarkResult:
    """Combined performance and quality metrics"""
    model_name: str
    dataset_name: str
    success: bool = False
    total_time: float = 0.0
    quality_scores: Dict[str, List[float]] = None
    responses: List[str] = None
    questions: List[str] = None
    error_message: str = ""

class RAGBenchmark:
    def __init__(self, api_key: str, embedding_model: str = "openai-small", llm_model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"

        # Initialize LLM for evaluation
        try:
            from src.llm import gpt_4o_mini_complete
            self.llm_func = gpt_4o_mini_complete
        except ImportError:
            print("Warning: Could not import LLM function for evaluation")
            self.llm_func = None


    def _get_working_dir(self, dataset_name: str) -> str:
        """Get working directory for a dataset"""
        return os.path.join(QAFD_RAG_HOME, "kg", "ultradomain", f"{dataset_name}_{self.embedding_model}")

    def _kg_exists(self, working_dir: str) -> bool:
        """Check if KG already exists"""
        kg_files = [
            os.path.join(working_dir, "vdb_entities.json"),
            os.path.join(working_dir, "vdb_chunks.json"),
            os.path.join(working_dir, "kv_store_full_docs.json"),
        ]
        return all(os.path.exists(f) for f in kg_files)

    def _get_llm_func(self):
        """Get LLM function based on model name"""
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

    async def _benchmark_qafd(self, dataset, dataset_name: str, question_count: int,
                              force_build: bool = False) -> BenchmarkResult:
        """Benchmark QAFD_RAG"""
        try:
            from src import QAFD_RAG, QueryParam

            working_dir = self._get_working_dir(dataset_name)
            os.makedirs(working_dir, exist_ok=True)

            # Get LLM function
            llm_func = self._get_llm_func()

            rag = QAFD_RAG(
                working_dir=working_dir,
                llm_model_func=llm_func,
                llm_model_name=self.llm_model,
                embedding_model_key=self.embedding_model,
                enable_llm_cache=True,
            )

            # Check if KG exists or needs to be built
            insertion_time = 0.0
            if self._kg_exists(working_dir) and not force_build:
                print(f"\n  Using existing KG: {working_dir}")
            else:
                print(f"\n  Building KG from {len(dataset)} documents...")
                start_time = time.time()
                for i, item in enumerate(dataset):
                    print_progress(i + 1, len(dataset), "Progress")
                    context = item.get("context", "")
                    if context:
                        await rag.ainsert(context)
                insertion_time = time.time() - start_time
                print(f"  KG built in {insertion_time:.2f}s")

            # Query phase
            print(f"\n  Running queries...")
            start_time = time.time()
            responses = []
            questions_list = []
            total_questions = min(question_count, len(dataset))
            for i in range(total_questions):
                print_progress(i + 1, total_questions, "Progress")

                question = dataset[i]["input"]
                questions_list.append(question)
                query_param = QueryParam(
                    mode="hybrid",
                    max_source_nodes=40,
                    min_flow_threshold=0.01
                )
                response = await rag.aquery(question, query_param)
                responses.append(response)

            query_time = time.time() - start_time

            return BenchmarkResult(
                model_name="QAFD_RAG",
                dataset_name=dataset_name,
                success=True,
                total_time=insertion_time + query_time,
                responses=responses,
                questions=questions_list,
            )

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return BenchmarkResult(
                model_name="QAFD_RAG",
                dataset_name=dataset_name,
                success=False,
                error_message=str(e)
            )

    async def _evaluate_quality(self, dataset, responses: List[str], model_name: str, dataset_name: str, question_count: int) -> Dict[str, List[float]]:
        """Evaluate response quality using GPT-4o"""
        if not responses:
            return {}

        try:
            # Import gpt_4o_complete specifically for evaluation
            from src.llm import gpt_4o_complete

            criteria = ["comprehensiveness", "diversity", "logicality", "relevance", "coherence"]
            response_scores = {criterion: [] for criterion in criteria}

            eval_count = min(question_count, len(dataset), len(responses))

            for i in range(eval_count):
                print_progress(i + 1, eval_count, "Evaluating")

                query = dataset[i]["input"]
                response = responses[i]

                # Evaluate this response 5 times
                response_criterion_scores = {criterion: [] for criterion in criteria}
                for eval_round in range(5):
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

                    evaluation_result = await gpt_4o_complete(prompt, max_tokens=200)

                    import re
                    json_match = re.search(r'\{.*\}', evaluation_result, re.DOTALL)
                    if json_match:
                        try:
                            scores = json.loads(json_match.group())
                            for criterion in criteria:
                                if criterion in scores:
                                    try:
                                        score_value = float(scores[criterion])
                                        if 0 <= score_value <= 100:  # Validate score range
                                            response_criterion_scores[criterion].append(score_value)
                                    except (ValueError, TypeError):
                                        continue
                        except json.JSONDecodeError:
                            continue

                # Calculate average for this response and add to the list
                for criterion in criteria:
                    if response_criterion_scores[criterion]:
                        avg_score = sum(response_criterion_scores[criterion]) / len(response_criterion_scores[criterion])
                        response_scores[criterion].append(avg_score)
                    else:
                        response_scores[criterion].append(0.0)

            return response_scores

        except Exception as e:
            print(f"Quality evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

    async def build_kg(self, max_documents: int = None) -> bool:
        """Build KG only (no benchmark)"""
        try:
            from datasets import load_dataset
            from src import QAFD_RAG
        except ImportError as e:
            print(f"ERROR: Missing dependency: {e}")
            return False

        for dataset_file in DATASETS:
            dataset_name = dataset_file.replace('.jsonl', '')

            print_header("QAFD-RAG Knowledge Graph Builder")
            print_config({
                "Dataset": f"UltraDomain ({dataset_name})",
                "Embedding": self.embedding_model,
                "LLM": self.llm_model,
                "Max Documents": str(max_documents) if max_documents else "all"
            })

            try:
                print("\n  Loading dataset...", end=" ", flush=True)
                dataset = load_dataset("TommyChien/UltraDomain", data_files=dataset_file, split="train")
                print(f"done ({len(dataset)} samples)")
            except Exception as e:
                print(f"failed: {e}")
                return False

            working_dir = self._get_working_dir(dataset_name)
            os.makedirs(working_dir, exist_ok=True)

            llm_func = self._get_llm_func()
            rag = QAFD_RAG(
                working_dir=working_dir,
                llm_model_func=llm_func,
                llm_model_name=self.llm_model,
                embedding_model_key=self.embedding_model,
                enable_llm_cache=True,
            )

            # Determine how many documents to process
            docs_to_process = len(dataset)
            if max_documents:
                docs_to_process = min(max_documents, len(dataset))

            print(f"\n  Building KG from {docs_to_process} documents...")
            start_time = time.time()

            for i, item in enumerate(dataset):
                if i >= docs_to_process:
                    break
                print_progress(i + 1, docs_to_process, "Progress")
                context = item.get("context", "")
                if context:
                    await rag.ainsert(context)

            build_time = time.time() - start_time

            print_header("Build Complete")
            print(f"  Documents processed:  {docs_to_process}")
            print(f"  Time elapsed:         {build_time:.2f}s")
            print(f"  Avg per document:     {build_time/docs_to_process:.2f}s")
            print(f"  Output directory:     {working_dir}")
            print()

        return True

    async def run_benchmark(self, question_count: int = 10, force_build: bool = False) -> Dict[str, List[BenchmarkResult]]:
        """Run benchmark on QAFD_RAG for all datasets"""
        try:
            from datasets import load_dataset
        except ImportError:
            print("ERROR: 'datasets' library not found. Install with: pip install datasets")
            return {}

        all_results = {}
        models = [("QAFD_RAG", self._benchmark_qafd)]

        for dataset_file in DATASETS:
            dataset_name = dataset_file.replace('.jsonl', '')

            print_header("QAFD-RAG UltraDomain Benchmark")
            print_config({
                "Dataset": dataset_name,
                "Questions": str(question_count),
                "Embedding": self.embedding_model,
                "LLM": self.llm_model
            })

            try:
                print("\n  Loading dataset...", end=" ", flush=True)
                dataset = load_dataset("TommyChien/UltraDomain", data_files=dataset_file, split="train")
                print(f"done ({len(dataset)} samples)")
            except Exception as e:
                print(f"failed: {e}")
                continue

            results = []

            for model_name, benchmark_func in models:
                result = await benchmark_func(dataset, dataset_name, question_count, force_build)

                # Add quality evaluation if successful
                if result.success:
                    print(f"\n  Evaluating quality ({len(result.responses)} responses)...")
                    result.quality_scores = await self._evaluate_quality(
                        dataset, result.responses, model_name, dataset_name, question_count
                    )

                results.append(result)

            all_results[dataset_name] = results
            self.print_results({dataset_name: results})

        return all_results

    def print_results(self, all_results: Dict[str, List[BenchmarkResult]]):
        """Print benchmark results"""
        import statistics

        for dataset_name, results in all_results.items():
            print_header(f"Results: UltraDomain ({dataset_name})")

            for result in results:
                if not result.success:
                    print(f"\n  Status: FAILED")
                    print(f"  Error: {result.error_message}")
                    continue

                print("\n  PERFORMANCE")
                print(f"  {'─' * 40}")
                print(f"  {'Total Time':<25} {result.total_time:.2f}s")

                if result.quality_scores:
                    scores = result.quality_scores

                    def safe_avg_std(score_list):
                        if not score_list:
                            return 0.0, 0.0
                        avg = sum(score_list) / len(score_list)
                        try:
                            std = statistics.stdev(score_list) if len(score_list) > 1 else 0.0
                        except statistics.StatisticsError:
                            std = 0.0
                        return avg, std

                    print("\n  QUALITY METRICS")
                    print(f"  {'─' * 40}")

                    compreh_avg, compreh_std = safe_avg_std(scores.get('comprehensiveness', []))
                    diversity_avg, diversity_std = safe_avg_std(scores.get('diversity', []))
                    logical_avg, logical_std = safe_avg_std(scores.get('logicality', []))
                    relevance_avg, relevance_std = safe_avg_std(scores.get('relevance', []))
                    coherence_avg, coherence_std = safe_avg_std(scores.get('coherence', []))

                    print(f"  {'Comprehensiveness':<25} {compreh_avg:.2f} ± {compreh_std:.2f}")
                    print(f"  {'Diversity':<25} {diversity_avg:.2f} ± {diversity_std:.2f}")
                    print(f"  {'Logicality':<25} {logical_avg:.2f} ± {logical_std:.2f}")
                    print(f"  {'Relevance':<25} {relevance_avg:.2f} ± {relevance_std:.2f}")
                    print(f"  {'Coherence':<25} {coherence_avg:.2f} ± {coherence_std:.2f}")

                    overall_avg = (compreh_avg + diversity_avg + logical_avg + relevance_avg + coherence_avg) / 5
                    print(f"\n  {'Overall Average':<25} {overall_avg:.2f}")
            print()

    def save_results(self, all_results: Dict[str, List[BenchmarkResult]]):
        """Save results as two separate files: eval metrics and generated responses"""
        import statistics

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(QAFD_RAG_HOME, "results", "ultradomain")
        os.makedirs(results_dir, exist_ok=True)

        for dataset_name, results in all_results.items():
            eval_file = os.path.join(results_dir, f"{dataset_name}_{timestamp}_eval.json")
            output_file = os.path.join(results_dir, f"{dataset_name}_{timestamp}_responses.json")

            for result in results:
                # --- Eval file: metrics and timing ---
                quality_summary = {}
                if result.quality_scores:
                    for criterion, scores in result.quality_scores.items():
                        if scores:
                            avg = sum(scores) / len(scores)
                            std = statistics.stdev(scores) if len(scores) > 1 else 0.0
                            quality_summary[criterion] = {"mean": avg, "std": std}

                eval_data = {
                    "timestamp": datetime.now().isoformat(),
                    "model": result.model_name,
                    "llm": self.llm_model,
                    "embedding": self.embedding_model,
                    "dataset": dataset_name,
                    "success": result.success,
                    "performance": {
                        "total_time": result.total_time,
                    },
                    "quality_scores": quality_summary,
                    "quality_raw": result.quality_scores,
                    "error": result.error_message,
                }

                with open(eval_file, 'w', encoding='utf-8') as f:
                    json.dump(eval_data, f, indent=2, ensure_ascii=False)

                # --- Responses file: questions + generated answers ---
                output_entries = []
                if result.responses:
                    for i, response in enumerate(result.responses):
                        entry = {
                            "id": i + 1,
                            "question": result.questions[i] if result.questions else "",
                            "generated_answer": response,
                        }
                        output_entries.append(entry)

                output_data = {
                    "timestamp": datetime.now().isoformat(),
                    "model": result.model_name,
                    "llm": self.llm_model,
                    "embedding": self.embedding_model,
                    "dataset": dataset_name,
                    "num_responses": len(output_entries),
                    "responses": output_entries,
                }

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)

            print(f"  Eval saved:      {eval_file}")
            print(f"  Responses saved: {output_file}\n")

async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="QAFD_RAG UltraDomain Benchmark")
    parser.add_argument("--dataset", type=str, default="mix.jsonl", help="Dataset file (default: mix.jsonl)")
    parser.add_argument("--questions", type=int, default=10, help="Number of questions to benchmark (default: 10)")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--embedding", type=str, default="openai-small",
                        choices=["openai-small", "openai-large", "jina-v3", "gritlm", "nvidia-nv-embed-v2"],
                        help="Embedding model (default: openai-small)")
    parser.add_argument("--llm", type=str, default="gpt-4o-mini",
                        choices=["gpt-4o-mini", "gpt-4o", "gpt-oss-120b", "gpt-5", "gpt-5-mini", "gpt-5-nano"],
                        help="LLM model for response generation (default: gpt-4o-mini)")
    parser.add_argument("--build", action="store_true",
                        help="Build KG only (no benchmark)")
    parser.add_argument("--force-build", action="store_true",
                        help="Force rebuild KG even if exists")
    parser.add_argument("--max-documents", type=int, default=None,
                        help="Max documents for KG building (default: all)")

    args = parser.parse_args()

    # Get API key from argument or environment
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OpenAI API key not provided. Set OPENAI_API_KEY environment variable or use --api-key")
        return

    # Update dataset if provided
    if args.dataset:
        DATASETS[0] = args.dataset

    benchmark = RAGBenchmark(api_key, embedding_model=args.embedding, llm_model=args.llm)

    # Build-only mode
    if args.build:
        await benchmark.build_kg(max_documents=args.max_documents)
        return

    all_results = await benchmark.run_benchmark(question_count=args.questions, force_build=args.force_build)
    benchmark.save_results(all_results)

if __name__ == "__main__":
    asyncio.run(main())
