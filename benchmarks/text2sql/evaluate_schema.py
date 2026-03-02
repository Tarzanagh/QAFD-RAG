#!/usr/bin/env python3
"""
Schema Retrieval Precision/Recall Evaluator for QAFD-RAG Text2SQL.

Compares retrieved tables/columns against gold standard.

Usage:
    python benchmarks/text2sql/evaluate_schema.py --results <benchmark_results.json>
    python benchmarks/text2sql/evaluate_schema.py --results <results.json> --gold <gold.json>
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import Dict, Set, Tuple

QAFD_RAG_HOME = Path(__file__).parent.parent.parent
DEFAULT_GOLD = QAFD_RAG_HOME / "data" / "text2sql" / "spider2-lite" / "golden_lite_spider_total.json"


def load_json_file(file_path: str) -> Dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_columns_from_gold(data: Dict) -> Dict[str, Set[str]]:
    """Extract columns from gold standard (CoFD-M format)."""
    columns_by_instance = {}
    for instance_id, instance_data in data.items():
        if isinstance(instance_data, dict) and 'schema_extraction' in instance_data:
            schema = instance_data['schema_extraction']
            columns_by_instance[instance_id] = set(schema.get('columns', []))
        else:
            columns_by_instance[instance_id] = set()
    return columns_by_instance


def _parse_create_table(create_table_text: str) -> Tuple[Set[str], Set[str]]:
    """Parse CREATE TABLE statements to extract table and column names."""
    tables = set()
    columns = set()
    current_table = None
    for line in create_table_text.splitlines():
        stripped = line.strip()
        table_match = re.match(r'CREATE\s+TABLE\s+`?(\w+)`?\s*\(', stripped, re.IGNORECASE)
        if table_match:
            current_table = table_match.group(1)
            tables.add(current_table)
            continue
        if current_table and stripped.startswith('`'):
            col_match = re.match(r'`(\w+)`\s+\w+', stripped)
            if col_match:
                columns.add(f"{current_table}.{col_match.group(1)}")
        if stripped.startswith(');'):
            current_table = None
    return tables, columns


def extract_columns_from_results(results_data: Dict) -> Dict[str, Set[str]]:
    """Extract columns from QAFD-RAG benchmark results by parsing create_table field."""
    columns_by_instance = {}
    for result in results_data.get('results', []):
        instance_id = result.get('instance_id', '')
        create_table = result.get('create_table', '')
        if create_table:
            _, cols = _parse_create_table(create_table)
            columns_by_instance[instance_id] = cols
        else:
            columns_by_instance[instance_id] = set()
    return columns_by_instance


def extract_tables_from_gold(data: Dict) -> Dict[str, Set[str]]:
    """Extract tables from gold standard."""
    tables_by_instance = {}
    for instance_id, instance_data in data.items():
        if isinstance(instance_data, dict) and 'schema_extraction' in instance_data:
            schema = instance_data['schema_extraction']
            tables_by_instance[instance_id] = set(schema.get('tables', []))
        else:
            tables_by_instance[instance_id] = set()
    return tables_by_instance


def extract_tables_from_results(results_data: Dict) -> Dict[str, Set[str]]:
    """Extract tables from QAFD-RAG benchmark results by parsing create_table field."""
    tables_by_instance = {}
    for result in results_data.get('results', []):
        instance_id = result.get('instance_id', '')
        create_table = result.get('create_table', '')
        if create_table:
            tbls, _ = _parse_create_table(create_table)
            tables_by_instance[instance_id] = tbls
        else:
            tables_by_instance[instance_id] = set()
    return tables_by_instance


def calculate_precision_recall(predicted: Set[str], actual: Set[str]) -> Tuple[float, float]:
    """Calculate precision and recall."""
    if not predicted and not actual:
        return 1.0, 1.0
    if not predicted:
        return 0.0, 0.0
    if not actual:
        return 0.0, 0.0

    tp = len(predicted & actual)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(actual) if actual else 0.0
    return precision, recall


def evaluate(gold: Dict[str, Set[str]], predicted: Dict[str, Set[str]]) -> Dict:
    """Evaluate predicted against gold for all common instances."""
    common = set(gold.keys()) & set(predicted.keys())
    if not common:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'instances': 0, 'details': {}}

    precisions, recalls = [], []
    details = {}

    for iid in sorted(common):
        p, r = calculate_precision_recall(predicted[iid], gold[iid])
        precisions.append(p)
        recalls.append(r)
        details[iid] = {
            'precision': round(p, 4),
            'recall': round(r, 4),
            'predicted': sorted(predicted[iid]),
            'gold': sorted(gold[iid]),
            'missing': sorted(gold[iid] - predicted[iid]),
            'extra': sorted(predicted[iid] - gold[iid]),
        }

    avg_p = sum(precisions) / len(precisions)
    avg_r = sum(recalls) / len(recalls)
    f1 = 2 * avg_p * avg_r / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0.0

    return {
        'precision': round(avg_p, 4),
        'recall': round(avg_r, 4),
        'f1': round(f1, 4),
        'instances': len(common),
        'details': details,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate text2sql schema retrieval")
    parser.add_argument("--results", required=True, help="QAFD-RAG benchmark results JSON")
    parser.add_argument("--gold", default=str(DEFAULT_GOLD), help="Gold standard JSON")
    parser.add_argument("--output", default=None, help="Save detailed results to JSON")
    args = parser.parse_args()

    print("Loading files...")
    results_data = load_json_file(args.results)
    gold_data = load_json_file(args.gold)

    # Extract columns and tables
    gold_cols = extract_columns_from_gold(gold_data)
    pred_cols = extract_columns_from_results(results_data)
    gold_tbls = extract_tables_from_gold(gold_data)
    pred_tbls = extract_tables_from_results(results_data)

    # Evaluate
    col_eval = evaluate(gold_cols, pred_cols)
    tbl_eval = evaluate(gold_tbls, pred_tbls)

    # Print results
    print(f"\n{'=' * 50}")
    print("SCHEMA RETRIEVAL EVALUATION")
    print(f"{'=' * 50}")
    print(f"  Instances evaluated: {col_eval['instances']}")

    print(f"\n  COLUMN RETRIEVAL")
    print(f"  {'─' * 40}")
    print(f"  Precision:  {col_eval['precision']:.4f}")
    print(f"  Recall:     {col_eval['recall']:.4f}")
    print(f"  F1:         {col_eval['f1']:.4f}")

    print(f"\n  TABLE RETRIEVAL")
    print(f"  {'─' * 40}")
    print(f"  Precision:  {tbl_eval['precision']:.4f}")
    print(f"  Recall:     {tbl_eval['recall']:.4f}")
    print(f"  F1:         {tbl_eval['f1']:.4f}")

    # Per-instance details
    print(f"\n  PER-INSTANCE COLUMN DETAILS")
    print(f"  {'─' * 40}")
    for iid, d in col_eval['details'].items():
        status = "OK" if d['recall'] == 1.0 else f"missing: {d['missing']}"
        print(f"  {iid:<15} P={d['precision']:.2f} R={d['recall']:.2f}  {status}")

    # Save
    output_path = args.output
    if not output_path:
        output_path = args.results.replace('.json', '_eval.json')

    output = {
        'column_evaluation': col_eval,
        'table_evaluation': tbl_eval,
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
