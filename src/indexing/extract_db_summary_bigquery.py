#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BigQuery Database Analyzer
- Lists tables, schema, sample rows, and metadata
- Groups similarly-patterned tables to reduce redundancy
- Heuristically detects potential primary/foreign keys
- Produces a simplified JSON summary and optional enriched output

Requirements:
  pip install google-cloud-bigquery google-api-core
"""

import argparse
import datetime
import json
import os
import re
import fnmatch
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional, Union

try:
    from google.oauth2 import service_account
    from google.cloud import bigquery
except ImportError:
    print("Please install required packages: pip install google-cloud-bigquery google-api-core")
    raise


# ===============================
# Table Pattern Analyzer
# ===============================

class BigQueryTablePatternAnalyzer:
    """Analyzes table name patterns and selects representative tables."""
    def __init__(self, client: Optional["bigquery.Client"] = None):
        self.client = client
        self.table_groups: Dict[str, Any] = {}
        self.representative_tables: Dict[str, Any] = {}
        self.group_metadata: Dict[str, Any] = {}

    def analyze_and_group_tables(
        self,
        tables_dict: Dict[str, List[str]],
        table_column_info: Optional[Dict[str, Dict[str, List[str]]]] = None
    ) -> Dict[str, Any]:
        print("Analyzing table patterns and grouping similar tables...")

        grouped_analysis = {
            'filtered_tables': {},
            'group_info': {},
            'table_column_inf': {}
        }

        for dataset_key, table_list in tables_dict.items():
            if not table_list:
                continue

            print(f"Processing dataset: {dataset_key}")
            project_id, dataset_id = dataset_key.split('.', 1)

            pattern_groups = self._group_tables_by_pattern(table_list, project_id, dataset_id)

            filtered_tables: List[str] = []
            for _, tables_in_group in pattern_groups.items():
                if len(tables_in_group) == 1:
                    table_name = tables_in_group[0]['table_name']
                    filtered_tables.append(table_name)
                    full_table_name = f"{project_id}.{dataset_id}.{table_name}"
                    grouped_analysis['group_info'][full_table_name] = None
                    grouped_analysis['table_column_inf'][full_table_name] = None
                else:
                    representative = self._select_representative_table(
                        tables_in_group, dataset_key, table_column_info
                    )
                    filtered_tables.append(representative['table_name'])

                    full_table_name = f"{project_id}.{dataset_id}.{representative['table_name']}"
                    group_info_string = self._generate_group_info_string(tables_in_group, representative)
                    grouped_analysis['group_info'][full_table_name] = group_info_string

                    column_info_string = self._generate_table_column_info_string(
                        tables_in_group, dataset_key, table_column_info
                    )
                    grouped_analysis['table_column_inf'][full_table_name] = column_info_string

            grouped_analysis['filtered_tables'][dataset_key] = filtered_tables

        print("Table grouping complete.")
        return grouped_analysis

    def create_grouped_structure(
        self,
        original_structure: Dict[str, Any],
        grouped_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        print("\n=== CREATING GROUPED STRUCTURE ===")
        grouped_structure = {"datasets": {}}

        for dataset_key, dataset_data in original_structure.get("datasets", {}).items():
            if dataset_key not in grouped_analysis['filtered_tables']:
                continue

            representative_tables = grouped_analysis['filtered_tables'][dataset_key]
            grouped_structure["datasets"][dataset_key] = {
                "project_id": dataset_data["project_id"],
                "dataset_id": dataset_data["dataset_id"],
                "tables": {}
            }

            for table_name in representative_tables:
                if table_name not in dataset_data["tables"]:
                    continue

                original_table_info = dict(dataset_data["tables"][table_name])
                project_id = dataset_data["project_id"]
                dataset_id = dataset_data["dataset_id"]
                full_table_name = f"{project_id}.{dataset_id}.{table_name}"

                group_info = grouped_analysis['group_info'].get(full_table_name)
                table_column_inf = grouped_analysis['table_column_inf'].get(full_table_name)

                if group_info:
                    original_table_info["table_info"] = group_info
                    print(f"    🔗 {table_name}: {group_info}")
                else:
                    print(f"    📋 {table_name}: Standalone table")

                if table_column_inf:
                    original_table_info["table_column_inf"] = table_column_inf

                grouped_structure["datasets"][dataset_key]["tables"][table_name] = original_table_info

        return grouped_structure

    def apply_pkfk_detection(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        print("\n=== APPLYING PK/FK DETECTION ===")
        if self.client is None:
            raise RuntimeError("BigQuery client is not set in BigQueryTablePatternAnalyzer.")

        project_datasets: List[Tuple[str, str]] = []
        for dataset_key in structure.get("datasets", {}).keys():
            project_id, dataset_id = dataset_key.split('.', 1)
            project_datasets.append((project_id, dataset_id))

        key_finder = BigQueryKeyFinder(self.client, project_datasets)

        key_finder.tables = []
        key_finder.table_columns = {}
        key_finder.table_sample_data = {}

        for _, dataset_data in structure.get("datasets", {}).items():
            for table_name, table_data in dataset_data.get("tables", {}).items():
                project_id = dataset_data["project_id"]
                dataset_id = dataset_data["dataset_id"]
                full_table_name = f"{project_id}.{dataset_id}.{table_name}"

                key_finder.tables.append(full_table_name)

                col_names = table_data.get("column_names", [])
                col_types = table_data.get("column_types", [])
                key_finder.table_columns[full_table_name] = list(zip(col_names, col_types))

                sample_rows = table_data.get("sample_rows", [])
                sample_data: Dict[str, List[Any]] = {c: [] for c in col_names}
                for row in sample_rows:
                    if isinstance(row, list):
                        for i, c in enumerate(col_names):
                            if i < len(row):
                                sample_data[c].append(row[i])
                    elif isinstance(row, dict):
                        for c in col_names:
                            if c in row:
                                sample_data[c].append(row[c])
                key_finder.table_sample_data[full_table_name] = sample_data

        print("\nFinding potential primary keys...")
        pk_results = key_finder.find_potential_primary_keys()

        print("\nFinding potential foreign keys...")
        fk_results = key_finder.find_potential_foreign_keys()

        print("PK/FK detection completed:")
        print(f"  Tables with primary keys: {len(pk_results)}")
        print(f"  Tables with foreign keys: {len(fk_results)}")
        print(f"  Total FK relationships: {sum(len(v) for v in fk_results.values())}")

        # Return full table names instead of simple names
        return {
            'tables': [t for t in key_finder.tables],
            'columns': {t: cols for t, cols in key_finder.table_columns.items()},
            'primary_keys': pk_results,
            'foreign_keys': fk_results
        }

    # ---- helpers ----

    def _group_tables_by_pattern(
        self,
        table_list: List[str],
        project_id: str,
        dataset_id: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for table_name in table_list:
            pattern_type, base_name, suffix = self._classify_table_pattern(table_name)
            info = {
                'table_name': table_name,
                'base_name': base_name,
                'suffix': suffix,
                'pattern_type': pattern_type
            }
            group_key = f"{pattern_type}_{base_name}" if base_name else f"STANDALONE_{table_name}"
            patterns[group_key].append(info)
        return dict(patterns)

    def _classify_table_pattern(self, table_name: str) -> Tuple[str, str, str]:
        m = re.search(r'^(.+?)(\d{4})$', table_name)
        if m:
            base, year = m.group(1), m.group(2)
            try:
                y = int(year)
                if 1900 <= y <= 2100:
                    return 'YEARLY', base, year
            except Exception:
                pass

        m = re.search(r'^(.+?)_(\d{4})_(.+)$', table_name)
        if m:
            base, year, suffix = m.group(1), m.group(2), m.group(3)
            try:
                y = int(year)
                if 1900 <= y <= 2100:
                    return 'YEARLY_SUFFIX', base, f"{year}_{suffix}"
            except Exception:
                pass

        m = re.search(r'^(.+)_(\d{8})$', table_name)
        if m:
            return 'DATE_STREAMING', m.group(1), m.group(2)

        m = re.search(r'^(.+)_(\d{4}_\d{2}_\d{2})$', table_name)
        if m:
            return 'DATE_STREAMING', m.group(1), m.group(2)

        m = re.search(r'^(.+)_(Q[1-4]_\d{4})$', table_name)
        if m:
            return 'QUARTERLY', m.group(1), m.group(2)

        m = re.search(r'^(.+)_((JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)_?\d{4})$', table_name)
        if m:
            return 'MONTHLY', m.group(1), m.group(2)

        m = re.search(r'^([A-Z]+[A-Z])(\d{2,4})$', table_name)
        if m:
            return 'NUMERIC_SUFFIX', m.group(1), m.group(2)

        m = re.search(r'^(.+)_(\d{3,})$', table_name)
        if m:
            return 'SEQUENTIAL', m.group(1), m.group(2)

        if re.search(r'_(ARCHIVE|BACKUP|BAK|HIST|HISTORICAL)$', table_name):
            base = re.sub(r'_(ARCHIVE|BACKUP|BAK|HIST|HISTORICAL)$', '', table_name)
            return 'ARCHIVE', base, 'ARCHIVE'

        return 'STANDALONE', table_name, ''

    def _select_representative_table(
        self,
        tables_in_group: List[Dict[str, Any]],
        dataset_key: str,
        table_column_info: Optional[Dict[str, Dict[str, List[str]]]] = None
    ) -> Dict[str, Any]:
        scores = []
        for t in tables_in_group:
            score = 0
            table_name = t['table_name']

            column_count = 0
            if table_column_info and dataset_key in table_column_info:
                column_count = len(table_column_info[dataset_key].get(table_name, []))
            score += column_count * 1000

            if t['pattern_type'] in ['YEARLY', 'DATE_STREAMING', 'QUARTERLY', 'MONTHLY'] and t['suffix']:
                year_digits = re.search(r'(\d{4})', t['suffix'])
                if year_digits:
                    try:
                        y = int(year_digits.group(1))
                        if y >= 2020:
                            score += 30
                        elif y >= 2015:
                            score += 20
                        elif y >= 2010:
                            score += 10
                    except Exception:
                        pass

            scores.append({'table_info': t, 'score': score, 'column_count': column_count})

        return max(scores, key=lambda x: x['score'])['table_info']

    def _generate_group_info_string(self, tables_in_group: List[Dict[str, Any]], representative: Dict[str, Any]) -> str:
        names = sorted([t['table_name'] for t in tables_in_group])
        others = [n for n in names if n != representative['table_name']]
        if len(names) <= 10:
            return f"{representative['table_name']} represents a group of tables containing {', '.join(others)}"
        first3, last3 = others[:3], others[-3:]
        return f"{representative['table_name']} represents a group of {len(names)} tables containing {', '.join(first3)}, ..., {', '.join(last3)}"

    def _generate_table_column_info_string(
        self,
        tables_in_group: List[Dict[str, Any]],
        dataset_key: str,
        table_column_info: Optional[Dict[str, Dict[str, List[str]]]] = None
    ) -> str:
        if not table_column_info or dataset_key not in table_column_info:
            return "Column information not available"

        table_column_mappings: List[str] = []
        for t in tables_in_group:
            table_name = t['table_name']
            cols = table_column_info[dataset_key].get(table_name, [])
            for c in cols:
                table_column_mappings.append(f"{table_name}.{c}")

        if len(table_column_mappings) <= 20:
            return f"Group columns are {', '.join(sorted(table_column_mappings))}"
        sorted_map = sorted(table_column_mappings)
        return f"Group columns are {', '.join(sorted_map[:10])}, ..., {', '.join(sorted_map[-10:])} (total: {len(table_column_mappings)} columns)"


# ===============================
# Key Finder
# ===============================

class BigQueryKeyFinder:
    """Heuristic PK/FK detector using table samples and naming patterns."""
    def __init__(self, client: "bigquery.Client", project_datasets: List[Tuple[str, str]]):
        self.client = client
        self.project_datasets = project_datasets
        self.tables: List[str] = []
        self.table_columns: Dict[str, List[Tuple[str, str]]] = {}
        self.table_sample_data: Dict[str, Dict[str, List[Any]]] = {}
        self.primary_keys: Dict[str, Dict[str, Any]] = {}
        self.foreign_keys: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def _extract_database_structure(self):
        for project_id, dataset_id in self.project_datasets:
            try:
                dataset_ref = self.client.dataset(dataset_id, project=project_id)
                tables = list(self.client.list_tables(dataset_ref))
                for table in tables:
                    full = f"{project_id}.{dataset_id}.{table.table_id}"
                    self.tables.append(full)

                    table_ref = dataset_ref.table(table.table_id)
                    table_obj = self.client.get_table(table_ref)

                    cols = [(f.name, f.field_type) for f in table_obj.schema]
                    self.table_columns[full] = cols
                    self._get_sample_data(full, table_obj)

            except Exception as e:
                print(f"Error accessing {project_id}.{dataset_id}: {e}")
                continue

        return self.tables, self.table_columns

    def _get_sample_data(self, full_table_name: str, table_obj: "bigquery.table.Table", sample_size: int = 100):
        try:
            q = f"SELECT * FROM `{full_table_name}` LIMIT {sample_size}"
            rows = self.client.query(q).result()
            sample: Dict[str, List[Any]] = {f.name: [] for f in table_obj.schema}
            for r in rows:
                for f in table_obj.schema:
                    sample[f.name].append(r.get(f.name))
            self.table_sample_data[full_table_name] = sample
        except Exception as e:
            print(f"Error getting sample data for {full_table_name}: {e}")
            self.table_sample_data[full_table_name] = {}

    def _get_sample_values(self, table_name: str, column_name: str) -> List[Any]:
        return self.table_sample_data.get(table_name, {}).get(column_name, [])

    @staticmethod
    def _analyze_sample_values(sample_values: List[Any]) -> Dict[str, Union[int, float]]:
        if not sample_values:
            return {"total_count": 0, "null_count": 0, "distinct_count": 0, "uniqueness_ratio": 0.0, "null_ratio": 0.0}
        non_null = [v for v in sample_values if v is not None]
        uniq = set(str(v) for v in non_null)
        total = len(sample_values)
        nulls = total - len(non_null)
        return {
            "total_count": total,
            "null_count": nulls,
            "distinct_count": len(uniq),
            "uniqueness_ratio": (len(uniq) / max(1, len(non_null))) if non_null else 0.0,
            "null_ratio": nulls / max(1, total)
        }
    
    def find_potential_primary_keys(self) -> Dict[str, Dict[str, Any]]:
        print(f"Finding potential primary keys for {len(self.tables)} tables...")
        for table in self.tables:
            pk_candidates: Dict[str, Dict[str, Any]] = {}
            for col_name, data_type in self.table_columns.get(table, []):
                stats = self._analyze_sample_values(self._get_sample_values(table, col_name))
                if stats["uniqueness_ratio"] < 0.9:
                    continue
                if stats["null_ratio"] > 0.1:
                    continue

                score = 0
                if stats["uniqueness_ratio"] == 1.0:
                    score += 30
                elif stats["uniqueness_ratio"] > 0.98:
                    score += 20
                if stats["null_count"] == 0:
                    score += 20
                typ = (data_type or "").upper()
                if typ in ['INTEGER', 'INT64', 'NUMERIC', 'BIGNUMERIC']:
                    score += 15
                elif typ in ['STRING', 'BYTES']:
                    score += 5
                
                # ✅ FIX: Ensure col_name is a string before regex operations
                col_name_str = str(col_name) if col_name is not None else ""
                base = table.split('.')[-1].lower()
                
                patterns = [
                    (r'^id$', 15),
                    (rf'^{base}_id$', 15),
                    (rf'^{base}_key$', 15),
                    (r'^pk_', 15),
                    (r'^key$', 10),
                    (r'^code$', 8),
                    (r'^uuid$', 15),
                    (r'^guid$', 15),
                    (r'id$', 5),
                ]
                for pat, pts in patterns:
                    if re.search(pat, col_name_str, re.IGNORECASE):
                        score += pts
                        break

                if score >= 25:
                    pk_candidates[col_name] = {
                        'score': score,
                        'data_type': data_type,
                        'uniqueness': stats["uniqueness_ratio"],
                        'null_ratio': stats["null_ratio"],
                        'sample_count': stats["total_count"]
                    }

            if pk_candidates:
                sorted_cands = sorted(pk_candidates.items(), key=lambda x: x[1]['score'], reverse=True)
                top_score = sorted_cands[0][1]['score']
                threshold = 0.8 * top_score
                pk_cols = [c for c, info in sorted_cands if info['score'] >= threshold]
                self.primary_keys[table] = {'columns': pk_cols, 'origin': 'potential'}

        return self.primary_keys

    def find_potential_foreign_keys(self) -> Dict[str, List[Dict[str, Any]]]:
        print(f"Finding potential foreign keys for {len(self.tables)} tables...")
        for src in self.tables:
            seen: set = set()
            for src_col, src_type in self.table_columns.get(src, []):
                # ✅ FIX: Ensure src_col is a string
                if src_col is None:
                    continue
                src_col_str = str(src_col)
                
                for ref in self.tables:
                    if ref == src:
                        continue
                    if ref not in self.primary_keys:
                        continue
                    ref_pk_cols = self.primary_keys[ref].get('columns', [])
                    if not ref_pk_cols:
                        continue

                    ref_base = ref.split('.')[-1].lower()
                    for ref_col in ref_pk_cols:
                        # ✅ FIX: Ensure ref_col is a string
                        if ref_col is None:
                            continue
                        ref_col_str = str(ref_col)
                        
                        if (src_col_str, ref, ref_col_str) in seen:
                            continue

                        ref_col_type = None
                        for c, t in self.table_columns.get(ref, []):
                            if c == ref_col:
                                ref_col_type = t
                                break

                        patterns = [
                            rf'^{ref_base}_{ref_col_str}$',
                            rf'^{ref_base}{ref_col_str.capitalize()}$',
                            rf'^{ref_base}_id$',
                            rf'^{ref_col_str}$'
                        ]
                        name_ok = any(re.match(p, src_col_str, re.IGNORECASE) for p in patterns)
                        if name_ok:
                            confidence = "medium" if (ref_col_type and src_type and ref_col_type.upper() == (src_type or '').upper()) else "low"
                            self.foreign_keys[src].append({
                                'from': src_col_str,
                                'to_table': ref,
                                'to_column': ref_col_str,
                                'origin': 'potential',
                                'confidence': confidence
                            })
                            seen.add((src_col_str, ref, ref_col_str))

        return dict(self.foreign_keys)

    def analyze(self) -> Dict[str, Any]:
        print("Analyzing BigQuery database structure")
        self._extract_database_structure()
        self.find_potential_primary_keys()
        self.find_potential_foreign_keys()
        return {
            'tables': [t for t in self.tables],
            'columns': {t: cols for t, cols in self.table_columns.items() },
            'primary_keys': self.primary_keys,
            'foreign_keys': self.foreign_keys
        }


# ===============================
# Database Analyzer
# ===============================

class BigQueryDatabaseAnalyzer:
    """Orchestrates loading datasets, grouping tables, and PK/FK detection."""
    def __init__(self, credentials_path: Optional[str] = None, project_id: Optional[str] = None):
        self.credentials_path = credentials_path
        self.project_id = project_id
        self.client = self._create_client()
        self.pattern_analyzer = BigQueryTablePatternAnalyzer(self.client)

    def _create_client(self) -> "bigquery.Client":
        try:
            if self.credentials_path:
                if not os.path.exists(self.credentials_path):
                    raise FileNotFoundError(
                        f"BigQuery credentials file not found at {self.credentials_path}. "
                        f"Please check the path."
                    )
                print(f"✅ Using service account credentials from {self.credentials_path}")
                creds = service_account.Credentials.from_service_account_file(self.credentials_path)
                return bigquery.Client(credentials=creds, project=self.project_id)

            # If no credentials_path provided at all, fallback to ADC
            print("⚠️ No credentials_path provided, falling back to Application Default Credentials (ADC)")
            return bigquery.Client(project=self.project_id)

        except Exception as e:
            print(f"❌ Error creating BigQuery client: {e}")
            raise



    def list_available_datasets(self, project_ids: Optional[List[str]] = None) -> Dict[str, List[str]]:
        if project_ids is None:
            project_ids = [self.project_id] if self.project_id else ['bigquery-public-data']
        out: Dict[str, List[str]] = {}
        for pid in project_ids:
            try:
                ds = list(self.client.list_datasets(project=pid))
                out[pid] = [d.dataset_id for d in ds]
                print(f"Found {len(out[pid])} datasets in {pid}")
            except Exception as e:
                print(f"Error listing datasets for {pid}: {e}")
                out[pid] = []
        return out

    def expand_datasets(self, project_datasets: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Expand (project, dataset_pattern) into concrete (project, dataset_id) pairs.
        Supports:
          - glob patterns: 'austin_*'
          - plain prefix convenience: 'austin' -> matches 'austin' and 'austin_*'
        """
        expanded: List[Tuple[str, str]] = []
        for project_id, pattern in project_datasets:
            try:
                all_ds = [d.dataset_id for d in self.client.list_datasets(project=project_id)]
            except Exception as e:
                print(f"Error listing datasets for {project_id}: {e}")
                continue

            if "*" not in pattern and "?" not in pattern and "[" not in pattern:
                candidates = [ds for ds in all_ds if ds == pattern or ds.startswith(pattern + "_")]
            else:
                candidates = [ds for ds in all_ds if fnmatch.fnmatch(ds, pattern)]

            if not candidates:
                print(f"  No datasets matched '{project_id}.{pattern}'")
            for ds in candidates:
                expanded.append((project_id, ds))
        return expanded

    def load_dataset_structure(self, project_datasets: List[Tuple[str, str]], sample_size: int = 10) -> Dict[str, Any]:
        print("Loading dataset structures from BigQuery...")
        structure = {"datasets": {}}
        for project_id, dataset_id in project_datasets:
            key = f"{project_id}.{dataset_id}"
            print(f"Loading dataset: {key}")
            ds = self._load_dataset_tables(project_id, dataset_id, sample_size=sample_size)
            if ds["tables"]:
                structure["datasets"][key] = ds
                print(f"  Loaded {len(ds['tables'])} tables")
        return structure

    def _load_dataset_tables(self, project_id: str, dataset_id: str, sample_size: int = 10) -> Dict[str, Any]:
        """
        Hybrid loader: combines local schema (DDL.csv) with live BigQuery table info.
        - Reads schema definitions from spider2-lite/resource/databases/bigquery/{dataset_id}/DDL.csv
        - Enriches with live samples & metadata from BigQuery
        """
        out = {"project_id": project_id, "dataset_id": dataset_id, "tables": {}}

        # --- 1. Try to load schema from local DDL.csv (FIXED: handle nested structures) ---
        local_db_root = os.path.join(os.environ.get("BQ_LOCAL_DB_ROOT", "./bigquery_dbs"), dataset_id)
        local_schemas: Dict[str, Dict[str, Any]] = {}
        ddl_paths = []

        # Check for nested structure (e.g., TCGA_bioclin_v0/isb-cgc.GDC_metadata/DDL.csv)
        if os.path.exists(local_db_root):
            try:
                items = os.listdir(local_db_root)
                for item in items:
                    item_path = os.path.join(local_db_root, item)
                    # Look for subdirectories matching project.dataset pattern
                    if os.path.isdir(item_path) and '.' in item:
                        nested_ddl = os.path.join(item_path, "DDL.csv")
                        if os.path.exists(nested_ddl):
                            ddl_paths.append(nested_ddl)
                            print(f"  📁 Found nested DDL: {item}/DDL.csv")
            except Exception as e:
                print(f"⚠️ Error scanning for nested DDL files: {e}")

        # Also check for direct DDL.csv at root level
        ddl_path = os.path.join(local_db_root, "DDL.csv")
        if os.path.exists(ddl_path):
            ddl_paths.insert(0, ddl_path)  # Prioritize root-level DDL

        # Load from all found DDL.csv files
        for ddl_file in ddl_paths:
            try:
                import csv
                with open(ddl_file, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        tname = row.get("table_name")
                        cname = row.get("column_name")
                        ctype = row.get("data_type")
                        if not tname:
                            continue
                        if tname not in local_schemas:
                            local_schemas[tname] = {
                                "table_name": tname,
                                "table_fullname": f"{project_id}.{dataset_id}.{tname}",
                                "column_names": [],
                                "column_types": [],
                                "sample_rows": [],
                                "row_count": None,
                                "size_bytes": None,
                                "created": None,
                                "modified": None
                            }
                        local_schemas[tname]["column_names"].append(cname)
                        local_schemas[tname]["column_types"].append(ctype)
                dir_name = os.path.basename(os.path.dirname(ddl_file))
                print(f"  ✅ Loaded {len([t for t in local_schemas.keys()])} tables from {dir_name}/DDL.csv")
            except Exception as e:
                print(f"⚠️ Error reading DDL.csv from {ddl_file}: {e}")

        # --- 2. Try to load tables from BigQuery API ---
        try:
            dataset_ref = self.client.dataset(dataset_id, project=project_id)
            tables = list(self.client.list_tables(dataset_ref))

            for table in tables:
                tname = table.table_id
                table_ref = dataset_ref.table(tname)

                try:
                    table_obj = self.client.get_table(table_ref)

                    # Prefer local schema if available
                    if tname in local_schemas:
                        col_names = local_schemas[tname]["column_names"]
                        col_types = local_schemas[tname]["column_types"]
                    else:
                        col_names = [f.name for f in table_obj.schema]
                        col_types = [f.field_type for f in table_obj.schema]

                    # Fetch online sample rows
                    samples = self._get_sample_rows(f"{project_id}.{dataset_id}.{tname}", sample_size)

                    out["tables"][tname] = {
                        "table_name": tname,
                        "table_fullname": f"{project_id}.{dataset_id}.{tname}",
                        "column_names": col_names,
                        "column_types": col_types,
                        "sample_rows": samples,
                        "row_count": table_obj.num_rows,
                        "size_bytes": table_obj.num_bytes,
                        "created": table_obj.created.isoformat() if table_obj.created else None,
                        "modified": table_obj.modified.isoformat() if table_obj.modified else None
                    }

                except Exception as e:
                    print(f"⚠️ Error loading table {tname} from BigQuery: {e}")
                    if tname in local_schemas:
                        out["tables"][tname] = local_schemas[tname]

        except Exception as e:
            print(f"❌ Error accessing dataset {project_id}.{dataset_id}: {e}")
            # If BigQuery fails entirely, fallback to local schema only
            if local_schemas:
                out["tables"].update(local_schemas)

        return out


    def _get_sample_rows(self, table_name: str, sample_size: int = 10) -> list:
        project_id = "western-trilogy-473322-n2"  # your project where jobUser permission exists
        query = f"SELECT * FROM `{table_name}` LIMIT {sample_size}"
        try:
            # Run the query using your own project as the job execution project
            rows = self.client.query(query, project=project_id).result()
            return [list(row.values()) for row in rows]
        except Exception as e:
            # Return empty list so PK/FK detection doesn't fail
            print(f"⚠️ Skipping samples for {table_name}: {str(e)[:100]}")
            return []  




    def _extract_table_column_info(self, structure: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
        out: Dict[str, Dict[str, List[str]]] = {}
        for dataset_key, dataset_data in structure.get("datasets", {}).items():
            out[dataset_key] = {}
            for table_name, t in dataset_data.get("tables", {}).items():
                out[dataset_key][table_name] = t.get("column_names", [])
        return out

    def apply_table_grouping(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        print("\n=== APPLYING TABLE GROUPING ===")
        table_column_info = self._extract_table_column_info(structure)
        tables_dict = {dk: list(dd["tables"].keys()) for dk, dd in structure.get("datasets", {}).items()}
        total = sum(len(v) for v in tables_dict.values())
        print(f"Total tables before grouping: {total}")
        grouped = self.pattern_analyzer.analyze_and_group_tables(tables_dict, table_column_info)
        reps = sum(len(v) for v in grouped['filtered_tables'].values())
        red = ((total - reps) / total * 100.0) if total else 0.0
        print(f"Tables after grouping: {reps}")
        print(f"Reduction: {red:.1f}%")
        return grouped

    def run_simple_analysis(self, project_datasets: List[Tuple[str, str]], output_dir: str = "./bigquery_analysis_results", sample_size: int = 10) -> Dict[str, Any]:
        print("STARTING BIGQUERY DATABASE ANALYSIS")
        print("=" * 60)

        # EXPAND nested/pattern datasets here
        project_datasets = self.expand_datasets(project_datasets)

        dataset_names = "_".join([f"{p}_{d}" for p, d in project_datasets])
        print(f"Processing datasets: {project_datasets}")

        try:
            print("Loading dataset structures...")
            structure = self.load_dataset_structure(project_datasets, sample_size=sample_size)
            if not structure["datasets"]:
                print("No datasets loaded successfully")
                return {"status": "error", "error": "No datasets loaded"}

            grouped_analysis = self.apply_table_grouping(structure)
            grouped_structure = self.pattern_analyzer.create_grouped_structure(structure, grouped_analysis)
            key_analysis = self.pattern_analyzer.apply_pkfk_detection(grouped_structure)
            simple_output = self._generate_simple_output(grouped_structure, key_analysis)

            print("\nSaving results...")
            saved = self._save_simple_output(simple_output, dataset_names, output_dir)

            result = {"output": simple_output, "saved_file": saved, "status": "success"}
            self._print_simple_summary(dataset_names, simple_output)
            return result

        except Exception as e:
            print(f"Error processing datasets: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)}

    # ----- helpers for simple output -----

    def _generate_simple_output(
        self,
        structure: Dict[str, Any],
        key_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate simplified JSON output (Snowflake-style structure):
        {
          "tables": {
            "project.dataset.table": {
              "name": "project.dataset.table",
              "columns": [...],
              "samples": {...},
              "foreign_keys": [...],
              "row_count": ...,
              "size_bytes": ...,
              "created": "...",
              "modified": "..."
            }
          },
          "relationships": [...]
        }
        """
        print("\n=== GENERATING SIMPLE OUTPUT ===")
        output: Dict[str, Any] = {"tables": {}, "relationships": []}

        for _, dataset_data in structure.get("datasets", {}).items():
            for table_name, t in dataset_data.get("tables", {}).items():
                project_id = dataset_data["project_id"]
                dataset_id = dataset_data["dataset_id"]
                full_table_name = f"{project_id}.{dataset_id}.{table_name}"

                col_names = t.get("column_names", [])
                col_types = t.get("column_types", [])
                sample_rows = t.get("sample_rows", [])

                # build columns metadata
                cols: List[Dict[str, Any]] = []
                for i, c in enumerate(col_names):
                    ctype = col_types[i] if i < len(col_types) else "UNKNOWN"
                    col_info = {
                        "name": c,
                        "type": ctype,
                        "is_primary_key": False,
                        "is_foreign_key": False
                    }

                    pk_info = key_analysis.get("primary_keys", {}).get(full_table_name)
                    if pk_info and c in pk_info.get("columns", []):
                        col_info["is_primary_key"] = True
                        col_info["pk_origin"] = pk_info.get("origin")

                    fk_list = key_analysis.get("foreign_keys", {}).get(full_table_name, [])
                    for fk in fk_list:
                        if fk.get("from") == c:
                            col_info["is_foreign_key"] = True
                            col_info["fk_origin"] = fk.get("origin")
                            col_info["references_table"] = fk.get("to_table")
                            col_info["references_column"] = fk.get("to_column")
                            break

                    cols.append(col_info)

                # sample rows per column
                samples: Dict[str, List[Any]] = {c: [] for c in col_names}
                for row in sample_rows:
                    if isinstance(row, list):
                        for i, c in enumerate(col_names):
                            if i < len(row):
                                samples[c].append(row[i])
                    elif isinstance(row, dict):
                        for c in col_names:
                            if c in row:
                                samples[c].append(row[c])

                # foreign keys info
                fks: List[Dict[str, Any]] = []
                for fk in key_analysis.get("foreign_keys", {}).get(full_table_name, []):
                    fks.append({
                        "column": fk['from'],
                        "references": {
                            "table": fk['to_table'],
                            "column": fk['to_column']
                        },
                        "fk_origin": fk.get('origin'),
                        "confidence": fk.get('confidence', 'medium')
                    })

                # final entry for this table
                entry = {
                    "name": full_table_name,          # ✅ keep full name inside
                    "columns": cols,
                    "samples": samples,
                    "foreign_keys": fks,
                    "row_count": t.get("row_count"),
                    "size_bytes": t.get("size_bytes"),
                    "created": t.get("created"),
                    "modified": t.get("modified")
                }

                if "table_info" in t:
                    entry["table_info"] = t["table_info"]
                if "table_column_inf" in t:
                    entry["table_column_inf"] = t["table_column_inf"]

                # ✅ ensure consistent key = full table name
                output["tables"][full_table_name] = entry

        # relationships block
        for full_table_name, fk_list in key_analysis.get("foreign_keys", {}).items():
            for fk in fk_list:
                output["relationships"].append({
                    "from_table": full_table_name,
                    "from_column": fk['from'],
                    "to_table": fk['to_table'],
                    "to_column": fk['to_column'],
                    "type": "simple"
                })

        return output


    def _save_simple_output(self, output: Dict[str, Any], dataset_names: str, output_dir: str) -> Optional[str]:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{dataset_names}_bigquery_summary.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False, default=str)
            print(f"Simple output saved: {path}")
            return path
        except Exception as e:
            print(f"Error saving simple output: {e}")
            return None

    def _print_simple_summary(self, dataset_names: str, output: Dict[str, Any]):
        print(f"\nSUMMARY FOR {dataset_names}")
        print("-" * 40)
        tables = output.get("tables", {})
        rels = output.get("relationships", [])
        print(f"Tables: {len(tables)}")
        print(f"Relationships: {len(rels)}")

        grouped = sum(1 for t in tables.values() if "table_info" in t)
        print(f"Grouped Tables: {grouped}")
        print(f"Standalone Tables: {len(tables) - grouped}")

        tables_with_pk = 0
        tables_with_fk = 0
        for t in tables.values():
            cols = t.get("columns", [])
            if any(c.get("is_primary_key") for c in cols):
                tables_with_pk += 1
            if any(c.get("is_foreign_key") for c in cols):
                tables_with_fk += 1
        print(f"Tables with PK: {tables_with_pk}")
        print(f"Tables with FK: {tables_with_fk}")

        total_size = sum((t.get("size_bytes") or 0) for t in tables.values())
        total_rows = sum((t.get("row_count") or 0) for t in tables.values())
        print(f"Total estimated rows: {total_rows:,}")
        if total_size > 0:
            print(f"Total estimated size: {total_size / (1024**3):.2f} GB")


def save_bigquery_db_summary(db_summary: Dict[str, Any], output_path: str, indent: int = 2) -> None:
    """Save JSON to disk (handles datetimes via isoformat)."""
    def json_serialize(obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        return str(obj)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(db_summary, f, default=json_serialize, indent=indent, ensure_ascii=False)
    print(f"BigQuery database summary saved to {output_path}")


# ===============================
# CLI
# ===============================

def main():
    # ===============================
    # CLI defaults
    # ===============================

    DEFAULT_CREDENTIALS = os.environ.get("BQ_SERVICE_ACCOUNT_JSON", None)
    DEFAULT_PROJECT = os.environ.get("BQ_PROJECT_ID", None)

    parser = argparse.ArgumentParser(description='BigQuery Database Analysis with Grouping and PK/FK Detection')
    parser.add_argument('--credentials', default=DEFAULT_CREDENTIALS, help='Path to service account JSON file')
    parser.add_argument('--project', default=DEFAULT_PROJECT, help='Default project ID')
    parser.add_argument('--datasets', nargs='+',
                        default=['bigquery-public-data.usa_names', 'bigquery-public-data.census_bureau_usa'],
                        help='Datasets to process (format: project.dataset | supports glob/prefix like austin_*)')
    parser.add_argument('--output-dir', default='./bigquery_results', help='Output directory')
    parser.add_argument('--sample-size', type=int, default=10, help='Sample rows per table')
    args = parser.parse_args()

    # parse incoming dataset strings
    project_datasets: List[Tuple[str, str]] = []
    for ds in args.datasets:
        if '.' in ds:
            p, d = ds.split('.', 1)
            project_datasets.append((p, d))
        else:
            project_datasets.append((args.project or 'bigquery-public-data', ds))

    analyzer = BigQueryDatabaseAnalyzer(args.credentials, args.project)

    # Use the analyzer approach with expansion
    expanded = analyzer.expand_datasets(project_datasets)
    analyzer.run_simple_analysis(project_datasets=expanded,
                                 output_dir=args.output_dir,
                                 sample_size=args.sample_size)

def extract_bigquery_db_summary(
    credentials_path: str,
    project_id: str,
    project_datasets: List[Tuple[str, str]],
    sample_limit: int = 10,
    detect_primary_keys: bool = True,
    detect_foreign_keys: bool = True,
    apply_table_grouping: bool = True
) -> Dict[str, Any]:
    """
    Wrapper to match db_summary_gen.py expectations.
    Uses BigQueryDatabaseAnalyzer to build a summary.
    """
    analyzer = BigQueryDatabaseAnalyzer(credentials_path, project_id)

    # Expand dataset patterns (wildcards, prefixes)
    expanded = analyzer.expand_datasets(project_datasets)

    # Load structure
    structure = analyzer.load_dataset_structure(expanded, sample_size=sample_limit)
    if not structure.get("datasets"):
        print("No datasets loaded successfully")
        return {}

    grouped_analysis = analyzer.apply_table_grouping(structure) if apply_table_grouping else None
    grouped_structure = (
        analyzer.pattern_analyzer.create_grouped_structure(structure, grouped_analysis)
        if grouped_analysis
        else structure
    )

    key_analysis = analyzer.pattern_analyzer.apply_pkfk_detection(grouped_structure) \
        if (detect_primary_keys or detect_foreign_keys) else {}

    return analyzer._generate_simple_output(grouped_structure, key_analysis)


if __name__ == "__main__":
    main()