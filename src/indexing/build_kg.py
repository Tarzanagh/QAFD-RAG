#!/usr/bin/env python3
"""
Build Knowledge Graph from HuggingFace Dataset

Generic script to build knowledge graphs from any HuggingFace dataset.
Output directory is auto-generated based on dataset name.

Usage:
    python -m src.indexing.build_kg --dataset TommyChien/UltraDomain --file mix.jsonl --field context
    python -m src.indexing.build_kg --dataset hotpotqa --split train --field context

Options:
    --dataset NAME      HuggingFace dataset name (required)
    --file FILE         Data file within dataset (e.g., mix.jsonl)
    --split SPLIT       Dataset split (default: train)
    --field FIELD       Text field to extract (default: context)
    --max-docs N        Limit number of documents (default: all)
    --embedding MODEL   Embedding model (default: openai-large)
    --output-dir DIR    Override auto-generated output directory
"""

import os
import sys
import asyncio
import argparse
import re
from datetime import datetime
from pathlib import Path

# Ensure QAFD-RAG is in path
QAFD_RAG_HOME = Path(__file__).parent.parent.parent
sys.path.insert(0, str(QAFD_RAG_HOME))


def slugify(name: str) -> str:
    """Convert dataset name to directory-safe slug."""
    # Remove owner prefix (e.g., "TommyChien/UltraDomain" -> "ultradomain")
    name = name.split("/")[-1].lower()
    # Remove special characters
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def get_output_dir(dataset: str, data_file: str = None) -> Path:
    """Auto-generate output directory from dataset name."""
    dataset_slug = slugify(dataset)

    if data_file:
        # e.g., "mix.jsonl" -> "mix"
        file_slug = slugify(Path(data_file).stem)
        return QAFD_RAG_HOME / "kg" / dataset_slug / file_slug

    return QAFD_RAG_HOME / "kg" / dataset_slug


async def build_kg(
    dataset: str,
    data_file: str = None,
    split: str = "train",
    text_field: str = "context",
    max_docs: int = None,
    embedding_model: str = "openai-large",
    output_dir: str = None,
):
    """Build KG from any HuggingFace dataset."""

    # Auto-generate output dir if not specified
    working_dir = Path(output_dir) if output_dir else get_output_dir(dataset, data_file)

    print("=" * 60)
    print("QAFD-RAG: Build Knowledge Graph")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Dataset: {dataset}")
    print(f"Data file: {data_file or 'default'}")
    print(f"Split: {split}")
    print(f"Text field: {text_field}")
    print(f"Embedding model: {embedding_model}")
    print(f"Output directory: {working_dir}")
    print(f"Max documents: {max_docs if max_docs else 'all'}")
    print("=" * 60)

    # Step 1: Load dataset
    print("\n[Step 1] Loading dataset...")
    try:
        from datasets import load_dataset

        load_kwargs = {"split": split}
        if data_file:
            load_kwargs["data_files"] = data_file

        ds = load_dataset(dataset, **load_kwargs)
        print(f"  Loaded {len(ds)} samples")
    except Exception as e:
        print(f"  ERROR: Failed to load dataset: {e}")
        return False

    # Step 2: Extract unique texts
    print(f"\n[Step 2] Extracting unique texts from '{text_field}' field...")
    try:
        all_texts = ds[text_field]
    except KeyError:
        print(f"  ERROR: Field '{text_field}' not found. Available: {ds.column_names}")
        return False

    unique_texts = list(set(all_texts))
    print(f"  Total samples: {len(all_texts)}")
    print(f"  Unique texts: {len(unique_texts)}")

    if max_docs and max_docs < len(unique_texts):
        unique_texts = unique_texts[:max_docs]
        print(f"  Limited to: {len(unique_texts)} documents")

    # Step 3: Initialize QAFD_RAG
    print("\n[Step 3] Initializing QAFD_RAG...")
    working_dir.mkdir(parents=True, exist_ok=True)

    try:
        from src import QAFD_RAG
        from src.llm import gpt_4o_mini_complete

        rag = QAFD_RAG(
            working_dir=str(working_dir),
            llm_model_func=gpt_4o_mini_complete,
            llm_model_name="gpt-4o-mini",
            embedding_model_key=embedding_model,
            enable_llm_cache=True,
        )
        print("  QAFD_RAG initialized successfully")
    except Exception as e:
        print(f"  ERROR: Failed to initialize QAFD_RAG: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 4: Insert documents
    print(f"\n[Step 4] Inserting {len(unique_texts)} documents into KG...")
    start_time = datetime.now()
    success_count = 0
    error_count = 0

    for i, doc in enumerate(unique_texts):
        try:
            if i % 10 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = i / elapsed if elapsed > 0 else 0
                print(f"    [{i+1}/{len(unique_texts)}] - {rate:.2f} docs/sec")

            await rag.ainsert(doc)
            success_count += 1

        except Exception as e:
            error_count += 1
            print(f"    ERROR at doc {i+1}: {str(e)[:100]}")
            if error_count > 10:
                print("    Too many errors, stopping...")
                break

    total_time = (datetime.now() - start_time).total_seconds()

    # Step 5: Summary
    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"  Documents processed: {success_count}/{len(unique_texts)}")
    print(f"  Errors: {error_count}")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Average: {total_time/max(success_count,1):.2f} sec/doc")
    print(f"  Output: {working_dir}")
    print("=" * 60)

    # Verify files
    print("\n[Verification] Created files:")
    for f in sorted(working_dir.iterdir()):
        size = f.stat().st_size
        print(f"    {f.name}: {size/1024/1024:.2f} MB")

    return success_count > 0


def main():
    parser = argparse.ArgumentParser(description="Build KG from HuggingFace dataset")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name")
    parser.add_argument("--file", default=None, help="Data file within dataset")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--field", default="context", help="Text field to extract (default: context)")
    parser.add_argument("--max-docs", type=int, default=None, help="Max documents to insert")
    parser.add_argument("--embedding", default="openai-large",
                        choices=["openai-small", "openai-large", "jina-v3"],
                        help="Embedding model (default: openai-large)")
    parser.add_argument("--output-dir", default=None, help="Override output directory")

    args = parser.parse_args()

    success = asyncio.run(build_kg(
        dataset=args.dataset,
        data_file=args.file,
        split=args.split,
        text_field=args.field,
        max_docs=args.max_docs,
        embedding_model=args.embedding,
        output_dir=args.output_dir,
    ))

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
