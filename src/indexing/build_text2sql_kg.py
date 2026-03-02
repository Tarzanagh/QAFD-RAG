#!/usr/bin/env python3
"""
Build Knowledge Graph from Database Schema Summary (Text-to-SQL)

Builds a text2sql knowledge graph from a JSON database summary file.
The DB summary is assumed to already exist (generated separately).

Usage:
    # By schema path + DB name (direct):
    python -m src.indexing.build_text2sql_kg --schema data/text2sql/databases/sqlite/Pagila/Pagila_db_summary.json --db-name Pagila
    python -m src.indexing.build_text2sql_kg --schema data/text2sql/databases/sqlite/Pagila/Pagila_db_summary.json --db-name Pagila --rebuild

    # By instance ID (reads spider2-lite.jsonl to resolve DB name + schema):
    python -m src.indexing.build_text2sql_kg --instance-id local038
    python -m src.indexing.build_text2sql_kg --instance-id local038 --rebuild

Options:
    --schema PATH       Path to the JSON DB summary file
    --db-name NAME      Database name for the KG directory
    --instance-id ID    Instance ID from spider2-lite.jsonl (auto-resolves schema + db-name)
    --metadata PATH     Optional path to metadata file
    --embedding MODEL   Embedding model (default: jina-v3)
    --llm MODEL         LLM model name (default: gpt-4o-mini)
    --output-dir DIR    Override auto-generated output directory
    --rebuild           Force rebuild even if KG already exists
    --language LANG     Output language (default: English)
"""

import os
import sys
import asyncio
import argparse
from datetime import datetime
from pathlib import Path

# Ensure QAFD-RAG is in path
QAFD_RAG_HOME = Path(__file__).parent.parent.parent
sys.path.insert(0, str(QAFD_RAG_HOME))


def get_output_dir(db_name: str) -> Path:
    """Auto-generate output directory from database name."""
    return QAFD_RAG_HOME / "kg" / "text2sql" / f"spider_local_{db_name}"


async def build_text2sql_kg(
    schema_path: str,
    db_name: str,
    metadata_path: str = None,
    embedding_model: str = "jina-v3",
    llm_model: str = "gpt-4o-mini",
    output_dir: str = None,
    rebuild: bool = False,
    language: str = "English",
):
    """Build text2sql KG from a database schema JSON summary."""

    working_dir = Path(output_dir) if output_dir else get_output_dir(db_name)

    # Check if KG already exists
    graph_file = working_dir / "graph_chunk_entity_relation.graphml"
    if graph_file.exists() and not rebuild:
        print(f"KG already exists at {working_dir}")
        print("Use --rebuild to force regeneration.")
        return True

    print("=" * 60)
    print("QAFD-RAG: Build Text2SQL Knowledge Graph")
    print("=" * 60)
    print(f"Timestamp:  {datetime.now().isoformat()}")
    print(f"DB Name:    {db_name}")
    print(f"Schema:     {schema_path}")
    print(f"Metadata:   {metadata_path or 'none'}")
    print(f"Embedding:  {embedding_model}")
    print(f"LLM:        {llm_model}")
    print(f"Output:     {working_dir}")
    print(f"Language:   {language}")
    print(f"Rebuild:    {rebuild}")
    print("=" * 60)

    # Validate schema file exists
    if not os.path.exists(schema_path):
        print(f"ERROR: Schema file not found: {schema_path}")
        return False

    # Create output directory
    working_dir.mkdir(parents=True, exist_ok=True)

    # Initialize QAFD_RAG
    print("\n[Step 1] Initializing QAFD_RAG...")
    try:
        from src import QAFD_RAG
        from src.llm import (gpt_4o_mini_complete, gpt_4o_complete, gpt_oss_120b_complete,
                              gpt_5_complete, gpt_5_mini_complete, gpt_5_nano_complete)

        llm_funcs = {
            "gpt-4o-mini": gpt_4o_mini_complete,
            "gpt-4o": gpt_4o_complete,
            "gpt-oss-120b": gpt_oss_120b_complete,
            "gpt-5": gpt_5_complete,
            "gpt-5-mini": gpt_5_mini_complete,
            "gpt-5-nano": gpt_5_nano_complete,
        }
        llm_func = llm_funcs.get(llm_model, gpt_4o_mini_complete)

        rag = QAFD_RAG(
            working_dir=str(working_dir),
            llm_model_func=llm_func,
            llm_model_name=llm_model,
            embedding_model_key=embedding_model,
            enable_llm_cache=True,
        )
        print("  QAFD_RAG initialized successfully")
    except Exception as e:
        print(f"  ERROR: Failed to initialize QAFD_RAG: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Build KG from schema
    print("\n[Step 2] Building knowledge graph from database schema...")
    start_time = datetime.now()

    try:
        result = await rag.abuild_from_database_schema(
            schema_file_path=schema_path,
            metadata_file_path=metadata_path,
            language=language,
        )
        total_time = (datetime.now() - start_time).total_seconds()

        print("\n" + "=" * 60)
        print("BUILD COMPLETE")
        print("=" * 60)
        print(f"  Schema type:          {result.get('schema_type', 'unknown')}")
        print(f"  Tables added:         {result.get('tables_added', 0)}")
        print(f"  Entities added:       {result.get('entities_added', 0)}")
        print(f"  Relationships added:  {result.get('relationships_added', 0)}")
        print(f"  Duplicates removed:   {result.get('duplicates_removed', 0)}")
        graph_stats = result.get('graph_stats', {})
        print(f"  Graph nodes:          {graph_stats.get('total_nodes', '?')}")
        print(f"  Graph edges:          {graph_stats.get('total_edges', '?')}")
        print(f"  Total time:           {total_time:.2f} seconds")
        print(f"  Output:               {working_dir}")
        print("=" * 60)

    except Exception as e:
        print(f"  ERROR: Failed to build KG: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify output files
    print("\n[Verification] Created files:")
    for f in sorted(working_dir.iterdir()):
        size = f.stat().st_size
        print(f"    {f.name}: {size/1024:.1f} KB")

    return True


def main():
    parser = argparse.ArgumentParser(description="Build Text2SQL KG from database schema")

    # Two modes: direct (--schema + --db-name) or instance-based (--instance-id)
    parser.add_argument("--schema", default=None, help="Path to JSON DB summary file")
    parser.add_argument("--db-name", default=None, help="Database name (e.g., Pagila)")
    parser.add_argument("--instance-id", default=None,
                        help="Instance ID from spider2-lite.jsonl (auto-resolves schema + db-name)")
    parser.add_argument("--metadata", default=None, help="Optional metadata file path")
    parser.add_argument("--embedding", default="jina-v3",
                        choices=["openai-small", "openai-large", "jina-v3"],
                        help="Embedding model (default: jina-v3)")
    parser.add_argument("--llm", default="gpt-4o-mini",
                        choices=["gpt-4o-mini", "gpt-4o", "gpt-oss-120b", "gpt-5", "gpt-5-mini", "gpt-5-nano"],
                        help="LLM model (default: gpt-4o-mini)")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild")
    parser.add_argument("--language", default="English", help="Output language")

    args = parser.parse_args()

    # Resolve schema + db-name from instance ID if provided
    schema_path = args.schema
    db_name = args.db_name

    if args.instance_id:
        from src.text2sql.runner import load_instance, get_schema_path

        instance = load_instance(args.instance_id)
        if not instance:
            print(f"ERROR: Instance '{args.instance_id}' not found in spider2-lite.jsonl")
            sys.exit(1)

        db_name = db_name or instance.get("db")
        if not db_name:
            print(f"ERROR: Instance '{args.instance_id}' has no 'db' field")
            sys.exit(1)

        if not schema_path:
            resolved = get_schema_path(db_name)
            if not resolved:
                print(f"ERROR: No DB summary found: data/text2sql/databases/sqlite/{db_name}/{db_name}_db_summary.json")
                sys.exit(1)
            schema_path = str(resolved)

        print(f"Resolved from instance '{args.instance_id}': db={db_name}, schema={schema_path}")

    if not schema_path or not db_name:
        parser.error("Either --instance-id or both --schema and --db-name are required")

    success = asyncio.run(build_text2sql_kg(
        schema_path=schema_path,
        db_name=db_name,
        metadata_path=args.metadata,
        embedding_model=args.embedding,
        llm_model=args.llm,
        output_dir=args.output_dir,
        rebuild=args.rebuild,
        language=args.language,
    ))

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
