#!/usr/bin/env python3
"""
Text2SQL Runner for QAFD-RAG.

Given an instance ID from spider2-lite.jsonl, automatically:
1. Looks up the question and DB name
2. Builds the KG if it doesn't exist
3. Queries the KG
4. Formats output as CREATE TABLE statements

Usage:
    python -m src.text2sql.runner --instance-id local038
    python -m src.text2sql.runner --instance-id local038 --rebuild
    python -m src.text2sql.runner --instance-id local038 local039 --format simple
    python -m src.text2sql.runner --db Pagila --list
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

QAFD_RAG_HOME = Path(__file__).parent.parent.parent
sys.path.insert(0, str(QAFD_RAG_HOME))

DEFAULT_JSONL = QAFD_RAG_HOME / "data" / "text2sql" / "spider2-lite.jsonl"
DEFAULT_SUMMARY_DIR = QAFD_RAG_HOME / "data" / "text2sql"
DEFAULT_KG_DIR = QAFD_RAG_HOME / "kg" / "text2sql"


# ---------------------------------------------------------------------------
# Instance loading
# ---------------------------------------------------------------------------

def load_instances(
    jsonl_path: Path = DEFAULT_JSONL,
    instance_ids: List[str] = None,
    db_filter: str = None,
) -> List[Dict[str, Any]]:
    """
    Load instances from spider2-lite.jsonl.

    Args:
        jsonl_path: Path to the JSONL file
        instance_ids: Filter to these specific IDs (None = all)
        db_filter: Filter to this DB name (None = all)

    Returns:
        List of instance dicts with keys: instance_id, db, question, external_knowledge
    """
    instances = []
    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            if instance_ids and data["instance_id"] not in instance_ids:
                continue
            if db_filter and data.get("db") != db_filter:
                continue
            instances.append(data)
    return instances


def load_instance(instance_id: str, jsonl_path: Path = DEFAULT_JSONL) -> Optional[Dict]:
    """Load a single instance by ID."""
    results = load_instances(jsonl_path, instance_ids=[instance_id])
    return results[0] if results else None


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_sqlite_path(db_name: str, summary_dir: Path = DEFAULT_SUMMARY_DIR) -> Optional[Path]:
    """Get path to SQLite database file. Returns None if not found."""
    path = summary_dir / f"{db_name}.sqlite"
    return path if path.exists() else None


def get_schema_path(db_name: str, summary_dir: Path = DEFAULT_SUMMARY_DIR) -> Optional[Path]:
    """Get path to DB summary JSON. Returns None if not found."""
    path = summary_dir / f"{db_name}_db_summary.json"
    return path if path.exists() else None


def ensure_db_summary(db_name: str, summary_dir: Path = DEFAULT_SUMMARY_DIR) -> Optional[Path]:
    """
    Ensure DB summary JSON exists for db_name.
    If the summary is missing but a .sqlite file is present, auto-generates it.

    Returns:
        Path to the summary JSON, or None if it cannot be produced.
    """
    summary_path = summary_dir / f"{db_name}_db_summary.json"
    if summary_path.exists():
        return summary_path

    sqlite_path = get_sqlite_path(db_name, summary_dir)
    if not sqlite_path:
        return None

    print(f"DB summary not found. Generating from {sqlite_path}...")
    try:
        from src.indexing.extract_db_summary import extract_db_summary_for_schema, save_db_summary
        db_summary = extract_db_summary_for_schema(str(sqlite_path))
        save_db_summary(db_summary, str(summary_path))
        print(f"DB summary saved to {summary_path}")
        return summary_path
    except Exception as e:
        print(f"ERROR: Failed to generate DB summary: {e}")
        return None


def get_kg_dir(db_name: str, kg_base: Path = DEFAULT_KG_DIR) -> Path:
    """Get KG directory for a database."""
    return kg_base / f"spider_local_{db_name}"


def kg_exists(db_name: str, kg_base: Path = DEFAULT_KG_DIR) -> bool:
    """Check if a KG already exists (graphml + vector DBs)."""
    kg_dir = get_kg_dir(db_name, kg_base)
    graphml = kg_dir / "graph_chunk_entity_relation.graphml"
    entities = kg_dir / "vdb_entities.json"
    return graphml.exists() and entities.exists()


# ---------------------------------------------------------------------------
# KG build
# ---------------------------------------------------------------------------

async def ensure_kg(
    db_name: str,
    schema_path: Path,
    embedding_model: str = "jina-v3",
    llm_model: str = "gpt-4o-mini",
    rebuild: bool = False,
) -> Path:
    """
    Ensure KG exists for db_name. Build if missing or rebuild=True.

    Returns:
        Path to the KG working directory
    """
    from src import QAFD_RAG
    from src.llm import gpt_4o_mini_complete, gpt_4o_complete, gpt_oss_120b_complete

    working_dir = get_kg_dir(db_name)

    if kg_exists(db_name) and not rebuild:
        print(f"KG exists for {db_name} at {working_dir}")
        return working_dir

    print(f"Building KG for {db_name} from {schema_path}...")
    working_dir.mkdir(parents=True, exist_ok=True)

    llm_funcs = {"gpt-4o-mini": gpt_4o_mini_complete, "gpt-4o": gpt_4o_complete, "gpt-oss-120b": gpt_oss_120b_complete}
    llm_func = llm_funcs.get(llm_model, gpt_4o_mini_complete)

    rag = QAFD_RAG(
        working_dir=str(working_dir),
        llm_model_func=llm_func,
        llm_model_name=llm_model,
        embedding_model_key=embedding_model,
        enable_llm_cache=True,
    )

    start = datetime.now()
    result = await rag.abuild_from_database_schema(
        schema_file_path=str(schema_path),
        language="English",
    )
    elapsed = (datetime.now() - start).total_seconds()

    print(f"KG built: {result.get('entities_added', 0)} entities, "
          f"{result.get('relationships_added', 0)} relationships "
          f"in {elapsed:.1f}s")

    return working_dir


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

async def query_kg(
    question: str,
    working_dir: Path,
    embedding_model: str = "jina-v3",
    llm_model: str = "gpt-4o-mini",
    return_raw: bool = True,
) -> Any:
    """
    Query an existing KG with a question.

    Args:
        question: The natural language question
        working_dir: Path to the KG directory
        embedding_model: Embedding model key
        llm_model: LLM model name
        return_raw: If True, return raw clusters (for prompt_parser formatting)

    Returns:
        Raw cluster data (list) or formatted context string
    """
    from src import QAFD_RAG, QueryParam
    from src.llm import gpt_4o_mini_complete, gpt_4o_complete, gpt_oss_120b_complete

    llm_funcs = {"gpt-4o-mini": gpt_4o_mini_complete, "gpt-4o": gpt_4o_complete, "gpt-oss-120b": gpt_oss_120b_complete}
    llm_func = llm_funcs.get(llm_model, gpt_4o_mini_complete)

    rag = QAFD_RAG(
        working_dir=str(working_dir),
        llm_model_func=llm_func,
        llm_model_name=llm_model,
        embedding_model_key=embedding_model,
        enable_llm_cache=True,
    )

    param = QueryParam(
        mode="hybrid",
        only_need_context=True,
        return_raw_clusters=return_raw,
        max_source_nodes=20,
    )

    result = await rag.aquery(question, param)

    if return_raw and isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            pass

    return result


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

async def run_instance(
    instance_id: str,
    embedding_model: str = "jina-v3",
    llm_model: str = "gpt-4o-mini",
    rebuild: bool = False,
    format_type: str = "create_table",
    jsonl_path: Path = DEFAULT_JSONL,
) -> Optional[str]:
    """
    End-to-end: look up instance -> build KG if needed -> query -> format.

    Returns:
        Formatted schema context string, or None on failure
    """
    from .prompt_parser import parse_qafd_clusters

    # 1. Look up instance
    instance = load_instance(instance_id, jsonl_path)
    if not instance:
        print(f"ERROR: Instance '{instance_id}' not found in {jsonl_path}")
        return None

    db_name = instance.get("db")
    question = instance.get("question", "")

    if not db_name:
        print(f"ERROR: Instance '{instance_id}' has no 'db' field")
        return None

    print(f"Instance:  {instance_id}")
    print(f"DB:        {db_name}")
    print(f"Question:  {question[:100]}{'...' if len(question) > 100 else ''}")

    # 2. Find (or auto-generate) schema summary
    schema_path = ensure_db_summary(db_name) or get_schema_path(db_name)
    if not schema_path and not kg_exists(db_name):
        print(f"ERROR: No DB summary found for '{db_name}'.")
        print(f"  Place data/text2sql/{db_name}.sqlite or {db_name}_db_summary.json and retry.")
        return None

    # 3. Ensure KG exists
    if schema_path or kg_exists(db_name):
        working_dir = await ensure_kg(
            db_name,
            schema_path or Path(""),  # won't be used if KG exists
            embedding_model=embedding_model,
            llm_model=llm_model,
            rebuild=rebuild,
        )
    else:
        print(f"ERROR: Cannot build KG without schema summary")
        return None

    # 4. Query KG
    print(f"\nQuerying KG for: {question[:80]}...")
    clusters = await query_kg(
        question,
        working_dir,
        embedding_model=embedding_model,
        llm_model=llm_model,
        return_raw=True,
    )

    # 5. Format output
    schema_data = None
    if schema_path:
        with open(schema_path, "r") as f:
            schema_data = json.load(f)

    if isinstance(clusters, list):
        formatted = parse_qafd_clusters(
            clusters,
            add_sample_rows=True,
            schema_data=schema_data,
            format_type=format_type,
        )
    else:
        # Fallback: return raw context string
        formatted = str(clusters)

    return formatted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Text2SQL Runner - query KGs from spider2-lite instances"
    )
    parser.add_argument(
        "--instance-id", nargs="+",
        help="One or more instance IDs to process"
    )
    parser.add_argument(
        "--db", default=None,
        help="Filter instances by database name"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List matching instances instead of running"
    )
    parser.add_argument(
        "--embedding", default="jina-v3",
        choices=["openai-small", "openai-large", "jina-v3"],
        help="Embedding model (default: jina-v3)"
    )
    parser.add_argument(
        "--llm", default="gpt-4o-mini",
        choices=["gpt-4o-mini", "gpt-4o", "gpt-oss-120b"],
        help="LLM model (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--format", default="create_table",
        choices=["create_table", "simple"],
        help="Output format (default: create_table)"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force rebuild KG even if it exists"
    )
    parser.add_argument(
        "--jsonl", default=str(DEFAULT_JSONL),
        help="Path to spider2-lite.jsonl"
    )

    args = parser.parse_args()
    jsonl_path = Path(args.jsonl)

    # List mode
    if args.list:
        instances = load_instances(
            jsonl_path,
            instance_ids=args.instance_id,
            db_filter=args.db,
        )
        print(f"{'Instance ID':<25} {'DB':<25} {'KG?':<6} {'Question'}")
        print("-" * 100)
        for inst in instances:
            db = inst.get("db", "??")
            has_kg = "yes" if kg_exists(db) else "no"
            q = inst["question"][:50]
            print(f"{inst['instance_id']:<25} {db:<25} {has_kg:<6} {q}")
        print(f"\nTotal: {len(instances)} instances")
        return

    # Run mode
    if not args.instance_id:
        parser.error("--instance-id is required (or use --list)")

    for iid in args.instance_id:
        print("=" * 60)
        result = asyncio.run(run_instance(
            instance_id=iid,
            embedding_model=args.embedding,
            llm_model=args.llm,
            rebuild=args.rebuild,
            format_type=args.format,
            jsonl_path=jsonl_path,
        ))

        if result:
            print(f"\n{'=' * 60}")
            print("OUTPUT:")
            print("=" * 60)
            print(result)
        else:
            print(f"\nFailed for instance {iid}")
        print()


if __name__ == "__main__":
    main()
