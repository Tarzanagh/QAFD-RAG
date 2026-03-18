#!/usr/bin/env python3
"""
SQuALITY Benchmark for QAFD_RAG
Multi-reference BLEU/ROUGE/METEOR metrics + LLM quality evaluation
"""

import os
import sys
import asyncio
import time
import json
import logging
import nest_asyncio
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Suppress verbose logging
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("QAFD_RAG").setLevel(logging.ERROR)
logging.getLogger("nano-vectordb").setLevel(logging.ERROR)
logging.getLogger("OpenAI").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

QAFD_RAG_HOME = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, QAFD_RAG_HOME)

nest_asyncio.apply()


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
    """Benchmark results"""
    model_name: str
    dataset_name: str
    success: bool = False
    total_time: float = 0.0
    kg_build_time: float = 0.0
    query_time: float = 0.0
    num_questions: int = 0
    metrics: Dict[str, float] = None  # BLEU, ROUGE, METEOR
    quality_scores: Dict[str, List[float]] = None  # Comprehensiveness, Diversity, etc.
    error_message: str = ""
    responses: List[str] = None  # Generated summaries
    questions: List[str] = None  # Original questions
    reference_answers: List[List[str]] = None  # Reference answers per question

class SQuALITYBenchmark:
    def __init__(self, api_key: str, embedding_model: str = "openai-small", llm_model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"

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

    def _get_working_dir(self, dataset_name: str) -> str:
        """Get working directory for dataset"""
        return os.path.join(QAFD_RAG_HOME, "kg", "summarization", f"{self.llm_model}_{self.embedding_model}_{dataset_name}")

    def _kg_exists(self, working_dir: str) -> bool:
        """Check if KG already exists"""
        # Check for key files that indicate a built KG
        kg_files = [
            os.path.join(working_dir, "vdb_entities.json"),
            os.path.join(working_dir, "vdb_chunks.json"),
            os.path.join(working_dir, "kv_store_full_docs.json"),
        ]
        return all(os.path.exists(f) for f in kg_files)

    async def _benchmark_qafd(self, dataset: List[Dict], dataset_name: str,
                            question_count: int, max_documents: int = None,
                            mode: str = "hybrid", max_source_nodes: int = 40,
                            min_flow_threshold: float = 0.01, alpha: float = 5.0,
                            evaluator: str = "mini", force_build: bool = False) -> BenchmarkResult:
        """Benchmark QAFD_RAG on SQuALITY"""
        try:
            from src.QAFD_RAG import QAFD_RAG, QueryParam

            working_dir = self._get_working_dir(dataset_name)

            llm_func = self._get_llm_func()
            rag = QAFD_RAG(
                working_dir=working_dir,
                llm_model_func=llm_func,
                llm_model_name=self.llm_model,
                embedding_model_key=self.embedding_model,
                enable_llm_cache=True,
            )

            # Get unique documents and selected questions
            unique_docs = {}
            selected_questions = []

            for item in dataset:
                passage_id = item["passage_id"]

                if passage_id not in unique_docs:
                    if max_documents and len(unique_docs) >= max_documents:
                        continue
                    unique_docs[passage_id] = item["document"]

                if passage_id in unique_docs:
                    selected_questions.append(item)
                    if len(selected_questions) >= question_count:
                        break

            dataset = selected_questions

            # Check if KG exists or needs to be built
            kg_build_time = 0.0
            if self._kg_exists(working_dir) and not force_build:
                print(f"\n  Using existing KG: {working_dir}")
            else:
                print(f"\n  Building KG from {len(unique_docs)} documents...")
                start_time = time.time()

                for i, (passage_id, document) in enumerate(unique_docs.items()):
                    print_progress(i + 1, len(unique_docs), "Progress")
                    await rag.ainsert(document)

                kg_build_time = time.time() - start_time
                print(f"  KG built in {kg_build_time:.2f}s")

            # Query phase
            print(f"\n  Running queries...")
            start_time = time.time()
            responses = []

            for i in range(min(question_count, len(dataset))):
                print_progress(i + 1, min(question_count, len(dataset)), "Progress")
                question = dataset[i]["question"]

                query_param = QueryParam(
                    mode=mode,
                    max_source_nodes=max_source_nodes,
                    min_flow_threshold=min_flow_threshold,
                    alpha=alpha
                )
                response = await rag.aquery(question, query_param)
                responses.append(response)

            query_time = time.time() - start_time

            # Calculate metrics with MULTIPLE REFERENCES
            print(f"\n  Calculating reference metrics...", end=" ", flush=True)
            metrics = self._calculate_metrics(responses, dataset[:len(responses)])
            print("done")

            # Calculate quality scores
            print(f"  Evaluating quality ({len(responses)} responses)...")
            quality_scores = await self._evaluate_quality(dataset, responses, evaluator)
            
            # Collect questions and reference answers for output
            questions_list = [dataset[i]["question"] for i in range(len(responses))]
            references_list = [dataset[i].get("all_answers", [dataset[i]["answer"]]) for i in range(len(responses))]

            return BenchmarkResult(
                model_name="QAFD_RAG",
                dataset_name=dataset_name,
                success=True,
                total_time=kg_build_time + query_time,
                kg_build_time=kg_build_time,
                query_time=query_time,
                num_questions=len(responses),
                metrics=metrics,
                quality_scores=quality_scores,
                responses=responses,
                questions=questions_list,
                reference_answers=references_list,
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
    
    def _calculate_metrics(self, responses: List[str], dataset: List[Dict]) -> Dict[str, float]:
        """Calculate BLEU, ROUGE, METEOR metrics with MULTIPLE REFERENCES"""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from nltk.translate.meteor_score import meteor_score
            from nltk.tokenize import word_tokenize
            from rouge_score import rouge_scorer
            import nltk
            
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            nltk.download('punkt', quiet=True)
            
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
            smoothing = SmoothingFunction().method1
            
            bleu1_scores = []
            bleu2_scores = []
            rouge1_scores = []
            rouge2_scores = []
            meteor_scores = []
            
            for i, response in enumerate(responses):
                if not response:
                    continue
                
                # Get ALL reference answers for this question
                all_refs = dataset[i].get("all_answers", [dataset[i]["answer"]])
                
                # Tokenize all references
                try:
                    ref_tokens_list = [word_tokenize(ref.lower()) for ref in all_refs]
                    resp_tokens = word_tokenize(response.lower())
                except:
                    ref_tokens_list = [ref.lower().split() for ref in all_refs]
                    resp_tokens = response.lower().split()
                
                # BLEU with multiple references
                bleu1 = sentence_bleu(ref_tokens_list, resp_tokens, weights=(1,0,0,0), smoothing_function=smoothing)
                bleu2 = sentence_bleu(ref_tokens_list, resp_tokens, weights=(0.5,0.5,0,0), smoothing_function=smoothing)
                bleu1_scores.append(bleu1)
                bleu2_scores.append(bleu2)
                
                # ROUGE uses first reference (standard)
                primary_ref = all_refs[0]
                rouge_result = scorer.score(primary_ref, response)
                rouge1_scores.append(rouge_result['rouge1'].fmeasure)
                rouge2_scores.append(rouge_result['rouge2'].fmeasure)
                
                # METEOR with multiple references
                try:
                    meteor = meteor_score(ref_tokens_list, resp_tokens)
                    meteor_scores.append(meteor)
                except:
                    meteor_scores.append(0.0)
            
            return {
                "bleu_1": sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0,
                "bleu_2": sum(bleu2_scores) / len(bleu2_scores) if bleu2_scores else 0,
                "rouge_1_f1": sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0,
                "rouge_2_f1": sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0,
                "meteor": sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0,
            }
            
        except Exception as e:
            print(f"  Warning: Metrics calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    async def _evaluate_quality(self, dataset: List[Dict], responses: List[str],
                               evaluator: str = "mini") -> Dict[str, List[float]]:
        """Evaluate response quality using GPT-4o or GPT-4o-mini"""
        if not responses:
            return {}

        try:
            if evaluator == "4o":
                from src.llm import gpt_4o_complete as eval_complete
            else:
                from src.llm import gpt_4o_mini_complete as eval_complete

            criteria = ["comprehensiveness", "diversity", "logicality", "relevance", "coherence"]
            response_scores = {criterion: [] for criterion in criteria}

            for i in range(len(responses)):
                print_progress(i + 1, len(responses), "Evaluating")
                
                question = dataset[i]["question"]
                response = responses[i]
                
                if not response:
                    for criterion in criteria:
                        response_scores[criterion].append(0.0)
                    continue
                
                response_criterion_scores = {criterion: [] for criterion in criteria}
                for _ in range(5):
                    prompt = f"""Evaluate the following response based on five criteria. Rate each from 0-100.

Question: {question}
Response: {response}

Please evaluate based on these criteria:
- Comprehensiveness: How much detail does the answer provide to cover all aspects and details of the question?
- Diversity: How varied and rich is the answer in providing different perspectives and insights on the question?
- Logicality: How logically does the answer respond to all parts of the question?
- Relevance: How relevant is the answer to the question, staying focused and addressing the intended topic or issue?
- Coherence: How well does the answer maintain internal logical connections between its parts, ensuring a smooth and consistent structure?

JSON format:
{{
    "comprehensiveness": [score],
    "diversity": [score],
    "logicality": [score],
    "relevance": [score],
    "coherence": [score]
}}"""

                    evaluation_result = await eval_complete(prompt, max_tokens=200)
                    
                    import re
                    json_match = re.search(r'\{.*\}', evaluation_result, re.DOTALL)
                    if json_match:
                        try:
                            scores = json.loads(json_match.group())
                            for criterion in criteria:
                                if criterion in scores:
                                    score_value = float(scores[criterion])
                                    if 0 <= score_value <= 100:
                                        response_criterion_scores[criterion].append(score_value)
                        except:
                            pass
                
                for criterion in criteria:
                    if response_criterion_scores[criterion]:
                        avg = sum(response_criterion_scores[criterion]) / len(response_criterion_scores[criterion])
                        response_scores[criterion].append(avg)
                    else:
                        response_scores[criterion].append(0.0)
            
            return response_scores
        except Exception as e:
            print(f"  Warning: Quality evaluation failed: {e}")
            return {}
    
    def _download_dataset(self) -> List[Dict]:
        """Download SQuALITY dataset - COLLECTS ALL REFERENCE ANSWERS"""
        try:
            from datasets import load_dataset
            import warnings
            warnings.filterwarnings("ignore", category=FutureWarning)
            logging.getLogger("datasets").setLevel(logging.WARNING)
        except ImportError:
            print("ERROR: Install datasets with: pip install datasets")
            sys.exit(1)

        dataset = load_dataset("pszemraj/SQuALITY-v1.3", split="train")
        
        prepared_data = []
        for i, item in enumerate(dataset):
            document = item.get("document", "")
            questions = item.get("questions", [])
            metadata = item.get("metadata", {})
            passage_id = metadata.get("passage_id", f"passage_{i}") if isinstance(metadata, dict) else f"passage_{i}"
            
            if not document or not questions:
                continue
            
            for q in questions:
                question_text = q.get("question_text", "")
                responses = q.get("responses", [])
                
                if responses and len(responses) > 0:
                    # COLLECT ALL REFERENCE ANSWERS (KEY CHANGE!)
                    answer_texts = [r.get("response_text", "") for r in responses if r.get("response_text", "")]
                    
                    if question_text and answer_texts:
                        prepared_data.append({
                            "document": document,
                            "question": question_text,
                            "answer": answer_texts[0],  # Primary answer
                            "all_answers": answer_texts,  # All references for BLEU
                            "passage_id": passage_id
                        })
        
        return prepared_data
    
    async def build_kg(self, dataset_name: str = "squality", max_documents: int = None) -> bool:
        """Build KG only (no benchmark)"""
        print_header("QAFD-RAG Knowledge Graph Builder")
        print_config({
            "Dataset": "SQuALITY",
            "Embedding": self.embedding_model,
            "LLM": self.llm_model,
            "Max Documents": str(max_documents) if max_documents else "all"
        })

        print("\n  Downloading dataset...", end=" ", flush=True)
        dataset = self._download_dataset()
        print("done")

        try:
            from src.QAFD_RAG import QAFD_RAG

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

            # Get unique documents
            unique_docs = {}
            for item in dataset:
                passage_id = item["passage_id"]
                if passage_id not in unique_docs:
                    if max_documents and len(unique_docs) >= max_documents:
                        break
                    unique_docs[passage_id] = item["document"]

            print(f"\n  Building KG from {len(unique_docs)} documents...")
            start_time = time.time()

            for i, (passage_id, document) in enumerate(unique_docs.items()):
                print_progress(i + 1, len(unique_docs), "Progress")
                await rag.ainsert(document)

            build_time = time.time() - start_time

            print_header("Build Complete")
            print(f"  Documents processed:  {len(unique_docs)}")
            print(f"  Time elapsed:         {build_time:.2f}s")
            print(f"  Avg per document:     {build_time/len(unique_docs):.2f}s")
            print(f"  Output directory:     {working_dir}")
            print()
            return True

        except Exception as e:
            print(f"\n  ERROR: {e}")
            return False

    async def run_benchmark(self, dataset_name: str = "squality", question_count: int = 250, max_documents: int = None,
                           mode: str = "hybrid", max_source_nodes: int = 40,
                           min_flow_threshold: float = 0.01, alpha: float = 10.0,
                           evaluator: str = "mini", force_build: bool = False) -> BenchmarkResult:
        """Run benchmark"""
        print_header("QAFD-RAG SQuALITY Benchmark")
        print_config({
            "Questions": str(question_count),
            "Embedding": self.embedding_model,
            "LLM": self.llm_model,
            "Mode": mode,
            "Max Nodes": str(max_source_nodes),
            "Threshold": str(min_flow_threshold),
            "Alpha": str(alpha),
            "Evaluator": "GPT-4o" if evaluator == "4o" else "GPT-4o-mini"
        })

        print("\n  Downloading dataset...", end=" ", flush=True)
        dataset = self._download_dataset()
        print("done")

        if len(dataset) < question_count:
            question_count = len(dataset)

        result = await self._benchmark_qafd(dataset, dataset_name, question_count, max_documents,
                                           mode, max_source_nodes, min_flow_threshold, alpha,
                                           evaluator, force_build)
        return result
    
    def print_results(self, result: BenchmarkResult):
        """Print results"""
        print_header("Results: SQuALITY")

        if not result.success:
            print(f"\n  Status: FAILED")
            print(f"  Error: {result.error_message}")
            return

        print("\n  PERFORMANCE")
        print(f"  {'─' * 40}")
        print(f"  {'Questions':<25} {result.num_questions}")
        print(f"  {'KG Build Time':<25} {result.kg_build_time:.2f}s")
        print(f"  {'Query Time':<25} {result.query_time:.2f}s")
        print(f"  {'Total Time':<25} {result.total_time:.2f}s")
        if result.num_questions > 0:
            print(f"  {'Avg per Question':<25} {result.query_time/result.num_questions:.2f}s")

        if result.metrics:
            print("\n  REFERENCE METRICS")
            print(f"  {'─' * 40}")
            for k, v in result.metrics.items():
                print(f"  {k:<25} {v*100:.2f}%")

        if result.quality_scores:
            print("\n  QUALITY METRICS")
            print(f"  {'─' * 40}")
            import statistics
            for k, v in result.quality_scores.items():
                if v:
                    avg = sum(v) / len(v)
                    std = statistics.stdev(v) if len(v) > 1 else 0
                    print(f"  {k:<25} {avg:.2f} ± {std:.2f}")

            all_scores = []
            for scores in result.quality_scores.values():
                all_scores.extend(scores)
            if all_scores:
                overall = sum(all_scores) / len(all_scores)
                print(f"\n  {'Overall Average':<25} {overall:.2f}")
        print()
    
    def save_results(self, result: BenchmarkResult, dataset_name: str = "squality"):
        """Save results as two separate files: eval metrics and generated responses"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(QAFD_RAG_HOME, "results", "summarization", dataset_name)
        os.makedirs(results_dir, exist_ok=True)

        eval_file = os.path.join(results_dir, f"{dataset_name}_{timestamp}_eval.json")
        output_file = os.path.join(results_dir, f"{dataset_name}_{timestamp}_responses.json")

        # --- Eval file: metrics, quality scores, timing ---
        quality_averages = {}
        quality_stds = {}
        if result.quality_scores:
            import statistics
            for criterion, scores in result.quality_scores.items():
                if scores:
                    quality_averages[criterion] = sum(scores) / len(scores)
                    quality_stds[criterion] = statistics.stdev(scores) if len(scores) > 1 else 0.0

        eval_data = {
            "timestamp": datetime.now().isoformat(),
            "model": result.model_name,
            "llm": self.llm_model,
            "embedding": self.embedding_model,
            "success": result.success,
            "performance": {
                "num_questions": result.num_questions,
                "kg_build_time": result.kg_build_time,
                "query_time": result.query_time,
                "total_time": result.total_time,
                "avg_per_question": result.query_time / result.num_questions if result.num_questions > 0 else 0
            },
            "reference_metrics": result.metrics,
            "quality_scores": quality_averages,
            "quality_stds": quality_stds,
            "quality_raw": result.quality_scores,
            "error": result.error_message
        }

        with open(eval_file, 'w') as f:
            json.dump(eval_data, f, indent=2)

        # --- Responses file: questions + generated summaries + references ---
        output_entries = []
        if result.responses:
            for i, response in enumerate(result.responses):
                entry = {
                    "id": i + 1,
                    "question": result.questions[i] if result.questions else "",
                    "generated_summary": response,
                    "reference_answers": result.reference_answers[i] if result.reference_answers else [],
                }
                output_entries.append(entry)

        output_data = {
            "timestamp": datetime.now().isoformat(),
            "model": result.model_name,
            "llm": self.llm_model,
            "embedding": self.embedding_model,
            "num_responses": len(output_entries),
            "responses": output_entries,
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\nEval saved:      {eval_file}")
        print(f"Responses saved: {output_file}")

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SQuALITY Benchmark for QAFD_RAG")
    parser.add_argument("--questions", type=int, default=250,
                       help="Number of questions (default: 250)")
    parser.add_argument("--max-documents", type=int, default=None,
                       help="Max documents (default: unlimited)")
    parser.add_argument("--mode", type=str, default="hybrid",
                       choices=["local", "global", "hybrid"],
                       help="Query mode (default: hybrid)")
    parser.add_argument("--max-source-nodes", type=int, default=40,
                       help="Max source nodes (default: 40)")
    parser.add_argument("--min-flow-threshold", type=float, default=0.01,
                       help="Min flow threshold (default: 0.01)")
    parser.add_argument("--alpha", type=float, default=10.0,
                       help="Alpha (default: 10.0)")
    parser.add_argument("--evaluator", type=str, default="mini",
                       choices=["mini", "4o"],
                       help="Evaluator model: mini (gpt-4o-mini) or 4o (gpt-4o) (default: mini)")
    parser.add_argument("--api-key", type=str,
                       help="OpenAI API key")
    parser.add_argument("--build", action="store_true",
                       help="Build KG only (no benchmark)")
    parser.add_argument("--force-build", action="store_true",
                       help="Force rebuild KG even if it exists")
    parser.add_argument("--embedding", type=str, default="openai-small",
                       choices=["openai-small", "openai-large", "jina-v3", "gritlm", "nvidia-nv-embed-v2"],
                       help="Embedding model (default: openai-small)")
    parser.add_argument("--llm", type=str, default="gpt-4o-mini",
                       choices=["gpt-4o-mini", "gpt-4o", "gpt-oss-120b", "gpt-5", "gpt-5-mini", "gpt-5-nano"],
                       help="LLM model (default: gpt-4o-mini)")
    parser.add_argument("--dataset", type=str, default="squality",
                       help="Dataset name (default: squality)")

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY")
        return

    benchmark = SQuALITYBenchmark(api_key, args.embedding, args.llm)

    # Build-only mode
    if args.build:
        await benchmark.build_kg(dataset_name=args.dataset, max_documents=args.max_documents)
        return

    result = await benchmark.run_benchmark(
        dataset_name=args.dataset,
        question_count=args.questions,
        max_documents=args.max_documents,
        mode=args.mode,
        max_source_nodes=args.max_source_nodes,
        min_flow_threshold=args.min_flow_threshold,
        alpha=args.alpha,
        evaluator=args.evaluator,
        force_build=args.force_build
    )
    benchmark.print_results(result)
    benchmark.save_results(result, dataset_name=args.dataset)

if __name__ == "__main__":
    asyncio.run(main())