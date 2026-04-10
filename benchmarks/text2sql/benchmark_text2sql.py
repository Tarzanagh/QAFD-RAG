#!/usr/bin/env python3
"""
Text2SQL Benchmark using QAFD-RAG for schema context retrieval.
Auto-builds KGs from database summaries.
"""

import os
import sys
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from pathlib import Path

# Suppress verbose logging
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("QAFD_RAG").setLevel(logging.ERROR)
logging.getLogger("nano-vectordb").setLevel(logging.ERROR)
logging.getLogger("OpenAI").setLevel(logging.ERROR)

# Add QAFD-RAG to path
QAFD_RAG_HOME = Path(__file__).parent.parent.parent
sys.path.insert(0, str(QAFD_RAG_HOME))

import nest_asyncio
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
class Text2SQLResult:
    instance_id: str
    db: str
    question: str
    create_table: str
    success: bool = True
    error_message: str = ""
    retrieved_tables: List[str] = None
    retrieved_columns: List[str] = None


def parse_schema_from_create_table(create_table: str) -> tuple:
    """Extract table names and table.column pairs from CREATE TABLE output."""
    import re
    tables = []
    columns = []
    current_table = None
    for line in create_table.split('\n'):
        m = re.match(r'^CREATE TABLE\s+`?(\w+)`?\s*\(', line)
        if m:
            current_table = m.group(1)
            tables.append(current_table)
            continue
        if current_table and line.strip().startswith('`'):
            cm = re.match(r'\s*`(\w+)`', line)
            if cm:
                columns.append(f"{current_table}.{cm.group(1)}")
        elif current_table and line.strip().startswith('"'):
            cm = re.match(r'\s*"(\w+)"', line)
            if cm:
                columns.append(f"{current_table}.{cm.group(1)}")
    return tables, columns


def compute_schema_metrics(results: List[Text2SQLResult], golden_path: str) -> dict:
    """Compute table and column recall/precision/F1 against golden annotations."""
    if not os.path.exists(golden_path):
        return {}

    with open(golden_path, 'r') as f:
        golden = json.load(f)

    table_metrics = []
    column_metrics = []

    for r in results:
        if not r.success or r.instance_id not in golden:
            continue
        g = golden[r.instance_id]
        if 'schema_extraction' not in g:
            continue

        golden_tables = set(t.lower() for t in g['schema_extraction'].get('tables', []))
        golden_cols = set(c.lower() for c in g['schema_extraction'].get('columns', []))

        retrieved_tables = set(t.lower() for t in (r.retrieved_tables or []))
        retrieved_cols = set(c.lower() for c in (r.retrieved_columns or []))

        # Table metrics
        if golden_tables:
            tp = len(retrieved_tables & golden_tables)
            precision = tp / len(retrieved_tables) if retrieved_tables else 0
            recall = tp / len(golden_tables)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            table_metrics.append({"precision": precision, "recall": recall, "f1": f1})

        # Column metrics
        if golden_cols:
            tp = len(retrieved_cols & golden_cols)
            precision = tp / len(retrieved_cols) if retrieved_cols else 0
            recall = tp / len(golden_cols)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            column_metrics.append({"precision": precision, "recall": recall, "f1": f1})

    if not table_metrics:
        return {}

    avg = lambda lst, key: sum(m[key] for m in lst) / len(lst)
    return {
        "table_recall": round(avg(table_metrics, "recall"), 4),
        "table_precision": round(avg(table_metrics, "precision"), 4),
        "table_f1": round(avg(table_metrics, "f1"), 4),
        "column_recall": round(avg(column_metrics, "recall"), 4),
        "column_precision": round(avg(column_metrics, "precision"), 4),
        "column_f1": round(avg(column_metrics, "f1"), 4),
        "num_evaluated": len(table_metrics),
    }


class Text2SQLBenchmark:
    def __init__(self, api_key: str, embedding_model: str = "openai-small", llm_model: str = "gpt-4o-mini", force_build: bool = False):
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.force_build = force_build
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"

    def load_questions(self, data_path: str, db_filter: List[str] = None) -> List[Dict]:
        """Load questions from spider2-lite.jsonl"""
        questions = []
        with open(data_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                if db_filter is None or item.get("db") in db_filter:
                    questions.append(item)
        return questions

    async def run_benchmark(self, questions: List[Dict], question_count: int = 5) -> List[Text2SQLResult]:
        """Run text2sql benchmark using the generic runner."""
        from src.text2sql.runner import kg_exists, get_schema_path, ensure_kg, query_kg, get_kg_dir
        from src.text2sql.prompt_parser import parse_qafd_clusters

        results = []
        total = min(question_count, len(questions))

        print(f"\n  Running queries...")

        for i, item in enumerate(questions[:question_count]):
            instance_id = item.get("instance_id", f"q{i}")
            db = item.get("db", "unknown")
            question = item.get("question", "")

            print_progress(i + 1, total, "Progress")

            # Auto-build KG if schema summary exists (generates from .sqlite if needed)
            if self.force_build or not kg_exists(db):
                from src.text2sql.runner import ensure_db_summary
                schema_path = ensure_db_summary(db) or get_schema_path(db)
                if not schema_path:
                    results.append(Text2SQLResult(
                        instance_id=instance_id, db=db, question=question,
                        create_table="",
                        success=False, error_message=f"No DB summary or .sqlite file for {db}"
                    ))
                    continue

                try:
                    await ensure_kg(db, schema_path,
                                    embedding_model=self.embedding_model,
                                    llm_model=self.llm_model,
                                    rebuild=self.force_build)
                except Exception as e:
                    results.append(Text2SQLResult(
                        instance_id=instance_id, db=db, question=question,
                        create_table="",
                        success=False, error_message=f"KG build failed: {e}"
                    ))
                    continue

            working_dir = get_kg_dir(db, llm_model=self.llm_model, embedding_model=self.embedding_model)

            try:
                # Get raw clusters (same as CoFD-M pipeline)
                clusters = await query_kg(
                    question, working_dir,
                    embedding_model=self.embedding_model,
                    llm_model=self.llm_model,
                    return_raw=True,
                )

                # Load db_summary for types, PK/FK, sample rows
                schema_data = None
                schema_path = get_schema_path(db)
                if schema_path:
                    with open(schema_path, 'r') as sf:
                        schema_data = json.load(sf)

                # Format as CREATE TABLE (with types, PK/FK, constraints, sample rows)
                if isinstance(clusters, list):
                    create_table_str = parse_qafd_clusters(
                        clusters,
                        add_sample_rows=True,
                        schema_data=schema_data,
                        format_type="create_table",
                    )
                else:
                    create_table_str = str(clusters)

                tables, columns = parse_schema_from_create_table(create_table_str)
                results.append(Text2SQLResult(
                    instance_id=instance_id, db=db, question=question,
                    create_table=create_table_str,
                    success=True,
                    retrieved_tables=tables,
                    retrieved_columns=columns,
                ))

            except Exception as e:
                results.append(Text2SQLResult(
                    instance_id=instance_id, db=db, question=question,
                    create_table="",
                    success=False, error_message=str(e)
                ))

        return results

    def save_results(self, results: List[Text2SQLResult], db_name: str = None,
                     benchmark: str = "spider2-lite"):
        """Save results as two separate files: eval metrics and generated responses"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if db_name:
            results_dir = QAFD_RAG_HOME / "results" / "text2sql" / db_name
        else:
            results_dir = QAFD_RAG_HOME / "results" / "text2sql"
        results_dir.mkdir(parents=True, exist_ok=True)

        eval_file = str(results_dir / f"text2sql_{timestamp}_eval.json")

        success_count = sum(1 for r in results if r.success)

        # --- Schema accuracy metrics (if golden file exists) ---
        golden_paths = {
            "spider2-lite": QAFD_RAG_HOME / "data" / "text2sql" / "spider2-lite" / "golden_lite_spider_total.json",
        }
        golden_path = str(golden_paths.get(benchmark, ""))
        schema_metrics = compute_schema_metrics(results, golden_path)

        # --- Eval file: success/fail stats + schema metrics ---
        eval_data = {
            "timestamp": datetime.now().isoformat(),
            "llm": self.llm_model,
            "embedding": self.embedding_model,
            "total_questions": len(results),
            "success_count": success_count,
            "fail_count": len(results) - success_count,
            "failed_instances": [
                {"instance_id": r.instance_id, "db": r.db, "error": r.error_message}
                for r in results if not r.success
            ],
        }
        if schema_metrics:
            eval_data["schema_metrics"] = schema_metrics

        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, indent=2, ensure_ascii=False)

        # --- Responses: one prompts txt per instance ---
        for r in results:
            if not r.success or not r.create_table:
                continue
            prompt_file = results_dir / f"{r.instance_id}_prompts.txt"
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(r.create_table)

        print(f"  Eval saved:      {eval_file}")
        print(f"  Prompts saved:   {results_dir}/<instance_id>_prompts.txt\n")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Text2SQL Benchmark using QAFD_RAG")
    parser.add_argument("--questions", type=int, default=5, help="Number of questions")
    parser.add_argument("--embedding", type=str, default="openai-small",
                        choices=["openai-small", "openai-large", "jina-v3", "gritlm", "nvidia-nv-embed-v2"])
    parser.add_argument("--llm", type=str, default="gpt-4o-mini",
                        choices=["gpt-4o-mini", "gpt-4o", "gpt-oss-120b", "gpt-5", "gpt-5-mini", "gpt-5-nano"])
    parser.add_argument("--db", type=str, default=None,
                        help="Filter by database name (e.g. Pagila, superhero)")
    parser.add_argument("--benchmark", type=str, default="spider2-lite",
                        choices=["spider2-lite", "bird"],
                        help="Benchmark dataset (default: spider2-lite)")
    parser.add_argument("--force-build", action="store_true",
                        help="Force rebuild KG even if it exists")
    parser.add_argument("--build", action="store_true",
                        help="Build KG only, don't run benchmark")

    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        return

    benchmark = Text2SQLBenchmark(api_key, args.embedding, args.llm, force_build=args.force_build)

    jsonl_files = {
        "spider2-lite": QAFD_RAG_HOME / "data" / "text2sql" / "spider2-lite" / "spider2-lite.jsonl",
        "bird": QAFD_RAG_HOME / "data" / "text2sql" / "bird" / "bird.jsonl",
    }
    data_path = str(jsonl_files[args.benchmark])
    db_filter = [args.db] if args.db else None
    questions = benchmark.load_questions(data_path, db_filter)

    # Auto-detect benchmark: if --db is set but found 0 questions, try the other benchmark
    if db_filter and len(questions) == 0:
        other = "bird" if args.benchmark == "spider2-lite" else "spider2-lite"
        other_path = str(jsonl_files[other])
        if os.path.exists(other_path):
            other_questions = benchmark.load_questions(other_path, db_filter)
            if other_questions:
                args.benchmark = other
                data_path = other_path
                questions = other_questions

    print_header("QAFD-RAG Text2SQL Benchmark")
    print_config({
        "Benchmark": args.benchmark,
        "Questions": f"{min(args.questions, len(questions))} / {len(questions)}",
        "DB Filter": str(db_filter[0]) if db_filter else "all",
        "Embedding": args.embedding,
        "LLM": args.llm
    })

    results = await benchmark.run_benchmark(questions, args.questions)

    success_count = sum(1 for r in results if r.success)

    print_header("Results: Text2SQL")
    print(f"\n  SUMMARY")
    print(f"  {'─' * 40}")
    print(f"  {'Successful':<25} {success_count}/{len(results)}")
    print(f"  {'Failed':<25} {len(results) - success_count}")

    # Schema accuracy metrics
    golden_paths = {
        "spider2-lite": QAFD_RAG_HOME / "data" / "text2sql" / "spider2-lite" / "golden_lite_spider_total.json",
    }
    golden_path = str(golden_paths.get(args.benchmark, ""))
    schema_metrics = compute_schema_metrics(results, golden_path)
    if schema_metrics:
        print(f"\n  SCHEMA RETRIEVAL ACCURACY")
        print(f"  {'─' * 40}")
        print(f"  {'Table Recall':<25} {schema_metrics['table_recall']*100:.1f}%")
        print(f"  {'Table Precision':<25} {schema_metrics['table_precision']*100:.1f}%")
        print(f"  {'Table F1':<25} {schema_metrics['table_f1']*100:.1f}%")
        print(f"  {'Column Recall':<25} {schema_metrics['column_recall']*100:.1f}%")
        print(f"  {'Column Precision':<25} {schema_metrics['column_precision']*100:.1f}%")
        print(f"  {'Column F1':<25} {schema_metrics['column_f1']*100:.1f}%")

    # Show failed instances
    failed = [r for r in results if not r.success]
    if failed:
        print(f"\n  FAILED INSTANCES")
        print(f"  {'─' * 40}")
        for r in failed[:5]:
            print(f"  {r.instance_id:<20} {r.error_message[:40]}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")
    print()

    benchmark.save_results(results, db_name=args.db, benchmark=args.benchmark)

if __name__ == "__main__":
    asyncio.run(main())
