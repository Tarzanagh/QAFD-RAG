import json
import os
import datetime
import statistics
from collections import defaultdict
import re
from typing import Dict, Any, List, Tuple, Optional, Union
import argparse
import glob

import snowflake.connector

SNOWFLAKE_USER = os.environ.get("SNOWFLAKE_USER", "")
SNOWFLAKE_PASSWORD = os.environ.get("SNOWFLAKE_PASSWORD", "")
SNOWFLAKE_ACCOUNT = os.environ.get("SNOWFLAKE_ACCOUNT", "")

def connect_snowflake():
    if not SNOWFLAKE_USER or not SNOWFLAKE_PASSWORD or not SNOWFLAKE_ACCOUNT:
        raise ValueError(
            "Snowflake credentials not set. Export SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, SNOWFLAKE_ACCOUNT."
        )
    return snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT
    )


class LocalTablePatternAnalyzer:
    """
    Analyzes table patterns and groups similar tables for local JSON files
    """
    
    def __init__(self):
        self.table_groups = {}
        self.representative_tables = {}
        self.group_metadata = {}
        
    def analyze_and_group_tables(self, tables_dict: Dict[str, List[str]], table_column_info: Dict[str, Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Analyze table patterns and group similar tables locally
        
        Args:
            tables_dict: Dictionary of schema_key -> list of table names
            table_column_info: Dictionary of schema_key -> table_name -> list of column names
            
        Returns:
            Dictionary containing grouped table analysis
        """
        print("Analyzing table patterns and grouping similar tables...")
        
        grouped_analysis = {
            'filtered_tables': {},
            'group_info': {},
            'table_column_inf': {}
        }
        
        for schema_key, table_list in tables_dict.items():
            if not table_list:
                continue
                
            print(f"Processing schema: {schema_key}")
            db_name, schema_name = schema_key.split('.', 1)
            
            # Group tables by pattern
            pattern_groups = self._group_tables_by_pattern(table_list, db_name, schema_name)
            
            # Select representative tables and generate group info
            filtered_tables = []
            
            for pattern_type, tables_in_group in pattern_groups.items():
                if len(tables_in_group) == 1:
                    # Single table - no grouping needed
                    table_name = tables_in_group[0]['table_name']
                    filtered_tables.append(table_name)
                    
                    full_table_name = f"{db_name}.{schema_name}.{table_name}"
                    grouped_analysis['group_info'][full_table_name] = None
                    grouped_analysis['table_column_inf'][full_table_name] = None
                    
                else:
                    # Multiple tables - select representative and create group info
                    representative = self._select_representative_table(tables_in_group, schema_key, table_column_info)
                    
                    filtered_tables.append(representative['table_name'])
                    
                    # Generate group info string
                    full_table_name = f"{db_name}.{schema_name}.{representative['table_name']}"
                    group_info_string = self._generate_group_info_string(tables_in_group, representative)
                    grouped_analysis['group_info'][full_table_name] = group_info_string
                    
                    # Generate table column info string
                    column_info_string = self._generate_table_column_info_string(tables_in_group, schema_key, table_column_info)
                    grouped_analysis['table_column_inf'][full_table_name] = column_info_string
            
            # Store filtered table list (only representatives)
            grouped_analysis['filtered_tables'][schema_key] = filtered_tables
            
        print(f"Table grouping complete.")
        return grouped_analysis
    
    def _group_tables_by_pattern(self, table_list: List[str], db_name: str, schema_name: str) -> Dict[str, List[Dict]]:
        """Group tables by detected patterns"""
        
        patterns = defaultdict(list)
        
        for table_name in table_list:
            pattern_type, base_name, suffix = self._classify_table_pattern(table_name)
            
            table_info = {
                'table_name': table_name,
                'base_name': base_name,
                'suffix': suffix,
                'pattern_type': pattern_type
            }
            
            # Group key combines pattern type and base name for better grouping
            if base_name:
                group_key = f"{pattern_type}_{base_name}"
            else:
                group_key = f"STANDALONE_{table_name}"
                
            patterns[group_key].append(table_info)
        
        return dict(patterns)
    
    def _classify_table_pattern(self, table_name: str) -> Tuple[str, str, str]:
        """Classify table pattern and return (pattern_type, base_name, identifier)"""
        
        # Year at end pattern (GSOD1929, GSOD2024, etc.)
        year_match = re.search(r'^(.+?)(\d{4})$', table_name)
        if year_match:
            base_name = year_match.group(1)
            year = year_match.group(2)
            if 1900 <= int(year) <= 2100:
                return 'YEARLY', base_name, year
        
        # Year in middle with suffix pattern (BLOCKGROUP_2010_5YR, CBSA_2007_1YR, etc.)
        year_middle_match = re.search(r'^(.+?)_(\d{4})_(.+)$', table_name)
        if year_middle_match:
            base_name = year_middle_match.group(1)
            year = year_middle_match.group(2)
            suffix = year_middle_match.group(3)
            if 1900 <= int(year) <= 2100:
                return 'YEARLY_SUFFIX', base_name, f"{year}_{suffix}"
        
        # Date streaming pattern (YYYYMMDD)
        date_match = re.search(r'^(.+)_(\d{8})$', table_name)
        if date_match:
            return 'DATE_STREAMING', date_match.group(1), date_match.group(2)
        
        # Date pattern with separators (YYYY_MM_DD)
        date_sep_match = re.search(r'^(.+)_(\d{4}_\d{2}_\d{2})$', table_name)
        if date_sep_match:
            return 'DATE_STREAMING', date_sep_match.group(1), date_sep_match.group(2)
        
        # Quarterly pattern (standard format: BASE_Q1_2024)
        quarterly_match = re.search(r'^(.+)_(Q[1-4]_\d{4})$', table_name)
        if quarterly_match:
            return 'QUARTERLY', quarterly_match.group(1), quarterly_match.group(2)
        
        # Quarterly pattern (underscore format: _1990_Q2)
        quarterly_underscore_match = re.search(r'^_(\d{4})_(Q[1-4])$', table_name)
        if quarterly_underscore_match:
            year = quarterly_underscore_match.group(1)
            quarter = quarterly_underscore_match.group(2)
            if 1900 <= int(year) <= 2100:
                return 'QUARTERLY', 'BLS_QCEW', f"{year}_{quarter}"
        
        # Monthly pattern
        monthly_match = re.search(r'^(.+)_((JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)_?\d{4})$', table_name)
        if monthly_match:
            return 'MONTHLY', monthly_match.group(1), monthly_match.group(2)
        
        # Numeric suffix without underscore (INDIV04, OPPEXP06, OTH00, etc.)
        numeric_suffix_match = re.search(r'^([A-Z]+[A-Z])(\d{2,4})$', table_name)
        if numeric_suffix_match:
            base_name = numeric_suffix_match.group(1)
            suffix = numeric_suffix_match.group(2)
            return 'NUMERIC_SUFFIX', base_name, suffix
        
        # Sequential numeric pattern with underscore
        seq_match = re.search(r'^(.+)_(\d{3,})$', table_name)
        if seq_match:
            return 'SEQUENTIAL', seq_match.group(1), seq_match.group(2)
        
        # Release/Version pattern with number in middle (REL17_DESCRIPTION, REL18_DESCRIPTION)
        release_match = re.search(r'^([A-Z]+)(\d{2,3})_(.+)$', table_name)
        if release_match:
            base_name = release_match.group(1)
            version = release_match.group(2)
            suffix = release_match.group(3)
            return 'RELEASE_VERSIONED', base_name, f"{version}_{suffix}"
        
        # Version pattern
        version_match = re.search(r'^(.+)_(V\d+|VER_\d+|VERSION_\d+)$', table_name)
        if version_match:
            return 'VERSIONED', version_match.group(1), version_match.group(2)
        
        # Archive pattern
        if re.search(r'_(ARCHIVE|BACKUP|BAK|HIST|HISTORICAL)$', table_name):
            base = re.sub(r'_(ARCHIVE|BACKUP|BAK|HIST|HISTORICAL)$', '', table_name)
            return 'ARCHIVE', base, 'ARCHIVE'
        
        # Default case - standalone table
        return 'STANDALONE', table_name, ''
    
    def _select_representative_table(self, tables_in_group: List[Dict], schema_key: str, table_column_info: Dict[str, Dict[str, List[str]]] = None) -> Dict:
        """Select the best representative table from a group based on column count (highest priority) and other factors"""
        
        table_scores = []
        
        for table_info in tables_in_group:
            score = 0
            table_name = table_info['table_name']
            
            # Get column count for this table
            column_count = 0
            if table_column_info and schema_key in table_column_info and table_name in table_column_info[schema_key]:
                column_count = len(table_column_info[schema_key][table_name])
            
            # Primary scoring: highest column count gets the highest score
            # Use a large multiplier to ensure column count dominates other factors
            score += column_count * 1000
            
            # Secondary scoring: other factors (much lower weight)
            # Prefer more recent tables for temporal patterns
            if table_info['pattern_type'] in ['YEARLY', 'DATE_STREAMING', 'QUARTERLY', 'MONTHLY']:
                if table_info['suffix']:
                    # For yearly patterns, prefer more recent years
                    if table_info['pattern_type'] == 'YEARLY' and table_info['suffix'].isdigit():
                        try:
                            year_val = int(table_info['suffix'])
                            if 2015 <= year_val <= 2022:  # Sweet spot for complete recent data
                                score += 25
                            elif 2010 <= year_val <= 2024:  # Still good recent data
                                score += 15
                            elif year_val >= 2000:  # Decent recent data
                                score += 10
                        except:
                            pass
                    
                    # For date patterns, prefer more recent dates
                    elif table_info['pattern_type'] == 'DATE_STREAMING' and re.match(r'\d{8}', table_info['suffix']):
                        try:
                            date_val = int(table_info['suffix'])
                            score += date_val / 100000  # Recent dates get higher scores
                        except:
                            pass
            
            # Prefer middle values for sequential patterns
            elif table_info['pattern_type'] == 'SEQUENTIAL':
                if table_info['suffix'] and table_info['suffix'].isdigit():
                    seq_val = int(table_info['suffix'])
                    if 50 <= seq_val <= 500:  # Middle range
                        score += 30
            
            table_scores.append({
                'table_info': table_info,
                'score': score,
                'column_count': column_count
            })
        
        # Select table with highest score (which will be dominated by column count)
        best_table = max(table_scores, key=lambda x: x['score'])
        return best_table['table_info']
    
    def _generate_group_info_string(self, tables_in_group: List[Dict], representative: Dict) -> str:
        """Generate a group info string showing all table names that this represents"""
        
        table_names = sorted([t['table_name'] for t in tables_in_group])
        
        if len(table_names) <= 10:
            other_tables = [name for name in table_names if name != representative['table_name']]
            return f"{representative['table_name']} represents a group of tables containing {', '.join(other_tables)}"
        else:
            other_tables = [name for name in table_names if name != representative['table_name']]
            first_few = other_tables[:3]
            last_few = other_tables[-3:]
            return f"{representative['table_name']} represents a group of {len(table_names)} tables containing {', '.join(first_few)}, ..., {', '.join(last_few)}"
    
    def _generate_table_column_info_string(self, tables_in_group: List[Dict], schema_key: str, table_column_info: Dict[str, Dict[str, List[str]]] = None) -> str:
        """Generate a table column info string showing all possible columns in the group"""
        
        if not table_column_info or schema_key not in table_column_info:
            return "Column information not available"
        
        all_columns = set()
        table_column_mappings = []
        
        for table_info in tables_in_group:
            table_name = table_info['table_name']
            if table_name in table_column_info[schema_key]:
                columns = table_column_info[schema_key][table_name]
                for column in columns:
                    all_columns.add(column)
                    table_column_mappings.append(f"{table_name}.{column}")
        
        if len(table_column_mappings) <= 20:
            return f"Group columns are {', '.join(sorted(table_column_mappings))}"
        else:
            sorted_mappings = sorted(table_column_mappings)
            first_few = sorted_mappings[:10]
            last_few = sorted_mappings[-10:]
            return f"Group columns are {', '.join(first_few)}, ..., {', '.join(last_few)} (total: {len(table_column_mappings)} columns)"


class LocalJSONDatabaseKeyFinder:
    """
    Enhanced key finder for local JSON database structures
    """
    
    def __init__(self, database_structure, db_name=None):
        self.database_structure = database_structure
        self.db_name = db_name or "unknown_db"
        self.tables = []
        self.table_columns = {}
        self.table_data = {}
        self.primary_keys = {}
        self.foreign_keys = defaultdict(list)
        
    def _extract_database_structure(self):
        """Extract tables and columns from database structure"""
        
        if "databases" in self.database_structure:
            # Multi-database structure
            for db_name, db_data in self.database_structure["databases"].items():
                if not self.db_name or self.db_name == "unknown_db":
                    self.db_name = db_name
                schemas = db_data.get("schemas", {})
                break  # Take first database
        elif "database" in self.database_structure:
            # Single database structure
            schemas = self.database_structure["database"]["schemas"]
            if not self.db_name or self.db_name == "unknown_db":
                self.db_name = self.database_structure["database"].get("name", "unknown_db")
        else:
            raise ValueError("Unsupported database structure format")
        
        # Extract tables from all schemas
        for schema_name, schema_data in schemas.items():
            if "tables" not in schema_data:
                continue
                
            for table_name, table_info in schema_data["tables"].items():
                full_table_name = f"{schema_name}.{table_name}"
                self.tables.append(full_table_name)
                
                # Extract column information
                columns = []
                column_names = table_info.get("column_names", [])
                column_types = table_info.get("column_types", [])
                
                for i, col_name in enumerate(column_names):
                    col_type = column_types[i] if i < len(column_types) else "UNKNOWN"
                    columns.append((col_name, col_type))
                
                self.table_columns[full_table_name] = columns
                self.table_data[full_table_name] = table_info
        
        return self.tables, self.table_columns

    def _get_sample_values(self, table_name, column_name):
        """Get sample values for a column from table data"""
        if table_name not in self.table_data:
            return []
        
        table_info = self.table_data[table_name]
        sample_rows = table_info.get("sample_rows", [])
        column_names = table_info.get("column_names", [])
        
        if column_name not in column_names:
            return []
        
        col_index = column_names.index(column_name)
        sample_values = []
        
        for row in sample_rows:
            if isinstance(row, list) and col_index < len(row):
                sample_values.append(row[col_index])
            elif isinstance(row, dict) and column_name in row:
                sample_values.append(row[column_name])
        
        return sample_values

    def _analyze_sample_values(self, sample_values):
        """Analyze sample values to determine characteristics"""
        if not sample_values:
            return {
                "total_count": 0,
                "null_count": 0,
                "distinct_count": 0,
                "uniqueness_ratio": 0,
                "null_ratio": 0
            }
        
        non_null_values = [v for v in sample_values if v is not None]
        unique_values = set(str(v) for v in non_null_values)
        
        total_count = len(sample_values)
        null_count = total_count - len(non_null_values)
        distinct_count = len(unique_values)
        
        return {
            "total_count": total_count,
            "null_count": null_count,
            "distinct_count": distinct_count,
            "uniqueness_ratio": distinct_count / max(1, len(non_null_values)) if non_null_values else 0,
            "null_ratio": null_count / max(1, total_count)
        }

    def find_potential_primary_keys(self):
        """Identify columns that are likely to be primary keys"""
        print(f"Finding potential primary keys for {len(self.tables)} tables...")
        
        for table_name in self.tables:
            pk_candidates = {}
            
            for col_name, data_type in self.table_columns[table_name]:
                sample_values = self._get_sample_values(table_name, col_name)
                stats = self._analyze_sample_values(sample_values)
                
                if stats["uniqueness_ratio"] < 0.9:
                    continue
                
                if stats["null_ratio"] > 0.1:
                    continue
                
                pk_score = 0
                
                # Score uniqueness
                if stats["uniqueness_ratio"] == 1.0:
                    pk_score += 30
                elif stats["uniqueness_ratio"] > 0.98:
                    pk_score += 20
                
                # No nulls is good for PKs
                if stats["null_count"] == 0:
                    pk_score += 20
                
                # Data type scoring
                if any(int_type in data_type.upper() for int_type in ['NUMBER', 'INTEGER', 'BIGINT', 'SMALLINT']):
                    pk_score += 15
                elif any(text_type in data_type.upper() for text_type in ['VARCHAR', 'CHAR', 'STRING', 'TEXT']):
                    pk_score += 5
                
                # Check for naming patterns
                table_base_name = table_name.split('.')[-1]
                name_patterns = [
                    (r'^id$', 15),
                    (r'^{}_id$'.format(table_base_name.lower()), 15),
                    (r'^{}_key$'.format(table_base_name.lower()), 15),
                    (r'^pk_', 15),
                    (r'^key$', 10),
                    (r'^code$', 8),
                    (r'^uuid$', 15),
                    (r'^guid$', 15),
                    (r'^serial$', 15),
                    (r'^seq', 10),
                    (r'id$', 5),
                    (r'uuid$', 10),
                    (r'code$', 5),
                    (r'num$', 5),
                    (r'no$', 5),
                    (r'^record', 8),
                    (r'^pid$', 15),
                    (r'^mid$', 15),
                    (r'^uid$', 15),
                    (r'^eid$', 15),
                    (r'^[a-z]+id$', 10),
                    (r'^[a-z]+_id$', 10),
                ]
                
                for pattern, score in name_patterns:
                    if re.search(pattern, col_name, re.IGNORECASE):
                        pk_score += score
                        break
                
                # Check for auto-increment indication in sample values
                try:
                    non_null_values = [v for v in sample_values if v is not None]
                    if non_null_values and all(isinstance(v, (int, float)) for v in non_null_values):
                        sorted_values = sorted([int(v) for v in non_null_values])
                        min_val = sorted_values[0]
                        
                        if min_val in [0, 1] and len(sorted_values) > 1:
                            expected_values = list(range(int(min_val), int(min_val) + len(sorted_values)))
                            if sorted_values == expected_values:
                                pk_score += 15
                except:
                    pass
                
                if pk_score >= 25:
                    pk_candidates[col_name] = {
                        'score': pk_score,
                        'data_type': data_type,
                        'uniqueness': stats["uniqueness_ratio"],
                        'null_ratio': stats["null_ratio"],
                        'sample_count': stats["total_count"]
                    }
            
            # Select the best primary key candidate(s)
            if pk_candidates:
                sorted_candidates = sorted(
                    pk_candidates.items(), 
                    key=lambda x: x[1]['score'], 
                    reverse=True
                )
                
                pk_columns = []
                threshold_score = sorted_candidates[0][1]['score'] * 0.8
                
                for col_name, info in sorted_candidates:
                    if info['score'] >= threshold_score:
                        pk_columns.append(col_name)
                
                self.primary_keys[table_name] = {
                    'columns': pk_columns,
                    'origin': 'potential'
                }
                
        return self.primary_keys

    def find_potential_foreign_keys(self):
        """Identify columns that are likely to be foreign keys"""
        print(f"Finding potential foreign keys for {len(self.tables)} tables...")
        
        for src_table in self.tables:
            processed_relationships = set()
            
            for src_col_name, src_data_type in self.table_columns[src_table]:
                for ref_table in self.tables:
                    if src_table == ref_table:
                        continue
                    
                    if ref_table not in self.primary_keys:
                        continue
                    
                    ref_col_list = self.primary_keys[ref_table].get('columns', [])
                    if not ref_col_list:
                        continue
                    
                    for ref_col in ref_col_list:
                        if (src_col_name, ref_table, ref_col) in processed_relationships:
                            continue
                        
                        # Get reference column data type
                        ref_col_type = None
                        for col_name, col_type in self.table_columns[ref_table]:
                            if col_name == ref_col:
                                ref_col_type = col_type
                                break
                        
                        # Define naming patterns for foreign keys
                        ref_table_base = ref_table.split('.')[-1]
                        fk_patterns = [
                            r'^{}_{}$'.format(ref_table_base.lower(), ref_col.lower()),
                            r'^{}{}$'.format(ref_table_base.lower(), ref_col.capitalize()),
                            r'^{}_id$'.format(ref_table_base.lower()),
                            r'^{}Id$'.format(ref_table_base.lower()),
                            r'^{}_key$'.format(ref_table_base.lower()),
                            r'^fk_{}_'.format(ref_table_base.lower()),
                            r'^{}$'.format(ref_col.lower())
                        ]
                        
                        name_pattern_match = False
                        for pattern in fk_patterns:
                            if re.match(pattern, src_col_name, re.IGNORECASE):
                                name_pattern_match = True
                                break
                        
                        if name_pattern_match:
                            confidence = "medium" if ref_col_type and src_data_type and ref_col_type.upper() == src_data_type.upper() else "low"
                            
                            self.foreign_keys[src_table].append({
                                'from': src_col_name,
                                'to_table': ref_table,
                                'to_column': ref_col,
                                'origin': 'potential',
                                'confidence': confidence
                            })
                            
                            processed_relationships.add((src_col_name, ref_table, ref_col))
                        
                        # Check data value matching
                        else:
                            try:
                                src_sample_values = self._get_sample_values(src_table, src_col_name)
                                ref_sample_values = self._get_sample_values(ref_table, ref_col)
                                
                                if not src_sample_values or not ref_sample_values:
                                    continue
                                
                                src_set = set(str(v) for v in src_sample_values if v is not None)
                                ref_set = set(str(v) for v in ref_sample_values if v is not None)
                                
                                if not src_set:
                                    continue
                                
                                invalid_refs = src_set - ref_set
                                
                                if len(invalid_refs) == 0:
                                    coverage_ratio = len(src_set) / max(1, len(ref_set))
                                    
                                    if (coverage_ratio > 0.01 and 
                                        (not ref_col_type or not src_data_type or 
                                         ref_col_type.upper() == src_data_type.upper())):
                                        
                                        self.foreign_keys[src_table].append({
                                            'from': src_col_name,
                                            'to_table': ref_table,
                                            'to_column': ref_col,
                                            'origin': 'potential',
                                            'confidence': 'high' if coverage_ratio > 0.3 else 'medium',
                                            'evidence': 'data_match',
                                            'coverage_ratio': round(coverage_ratio, 3)
                                        })
                                        
                                        processed_relationships.add((src_col_name, ref_table, ref_col))
                            except Exception as e:
                                continue
        
        return dict(self.foreign_keys)

    def analyze(self):
        """Run the full analysis to find potential primary and foreign keys"""
        print(f"Analyzing database structure for {self.db_name}")
        
        tables, table_columns = self._extract_database_structure()
        print(f"Found {len(tables)} tables: {', '.join([t.split('.')[-1] for t in tables])}")
        
        print("\nFinding potential primary keys...")
        pk_results = self.find_potential_primary_keys()
        
        print("\nFinding potential foreign keys...")
        fk_results = self.find_potential_foreign_keys()
        
        return {
            'tables': [t.split('.')[-1] for t in tables],
            'columns': {t.split('.')[-1]: cols for t, cols in table_columns.items()},
            'primary_keys': {t.split('.')[-1]: pk for t, pk in pk_results.items()},
            'foreign_keys': {t.split('.')[-1]: fk for t, fk in fk_results.items()}
        }


class LocalDatabaseAnalyzer:
    """
    Combined analyzer for local JSON database files with grouping and PK/FK detection
    """
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.pattern_analyzer = LocalTablePatternAnalyzer()
        
    def load_database_structure(self, database_name: str) -> Dict[str, Any]:
        """Load database structure from local JSON files"""
        
        db_path = os.path.join(self.base_path, database_name)
        
        if not os.path.exists(db_path):
            available_dbs = [d for d in os.listdir(self.base_path) 
                           if os.path.isdir(os.path.join(self.base_path, d))]
            raise ValueError(f"Database '{database_name}' not found. Available: {available_dbs}")
        
        print(f"Loading database structure from: {db_path}")
        
        structure = {
            "databases": {
                database_name: self._load_database_directory(db_path, database_name)
            }
        }
        
        return structure
    
    def _load_database_directory(self, db_path: str, db_name: str) -> Dict[str, Any]:
        """Load a database directory structure"""
        
        db_structure = {
            "name": db_name,
            "schemas": {}
        }
        
        items = os.listdir(db_path)
        json_files = [f for f in items if f.endswith('.json')]
        subdirs = [d for d in items if os.path.isdir(os.path.join(db_path, d))]
        
        if json_files and not subdirs:
            # Direct table files in database directory
            db_structure["schemas"]["PUBLIC"] = self._load_schema_directory(db_path, "PUBLIC")
        elif subdirs:
            # Schema directories
            for schema_name in subdirs:
                schema_path = os.path.join(db_path, schema_name)
                schema_structure = self._load_schema_directory(schema_path, schema_name)
                if schema_structure["tables"]:
                    db_structure["schemas"][schema_name] = schema_structure
        
        return db_structure
    
    def _load_schema_directory(self, schema_path: str, schema_name: str) -> Dict[str, Any]:
        """Load a schema directory structure"""
        
        schema_structure = {
            "name": schema_name,
            "tables": {}
        }
        
        json_files = glob.glob(os.path.join(schema_path, "*.json"))
        
        for json_file in json_files:
            table_name = os.path.splitext(os.path.basename(json_file))[0]
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    table_data = json.load(f)
                
                # Ensure required fields
                if "table_name" not in table_data:
                    table_data["table_name"] = table_name
                if "table_fullname" not in table_data:
                    table_data["table_fullname"] = f"{db_name}.{schema_name}.{table_name}"

                
                schema_structure["tables"][table_name] = table_data
                
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        return schema_structure
    
    def _extract_table_column_info(self, structure: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
        """Extract column information for all tables to help with representative selection"""
        
        table_column_info = {}
        
        for db_name, db_data in structure["databases"].items():
            for schema_name, schema_data in db_data["schemas"].items():
                schema_key = f"{db_name}.{schema_name}"
                table_column_info[schema_key] = {}
                
                for table_name, table_data in schema_data["tables"].items():
                    column_names = table_data.get("column_names", [])
                    table_column_info[schema_key][table_name] = column_names
        
        return table_column_info
    
    def apply_table_grouping(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Apply table grouping to reduce the number of tables"""
        
        print("\n=== APPLYING TABLE GROUPING ===")
        
        # Extract table column information first
        table_column_info = self._extract_table_column_info(structure)
        
        # Convert structure to the format expected by pattern analyzer
        tables_dict = {}
        
        for db_name, db_data in structure["databases"].items():
            for schema_name, schema_data in db_data["schemas"].items():
                schema_key = f"{db_name}.{schema_name}"
                table_list = list(schema_data["tables"].keys())
                tables_dict[schema_key] = table_list
        
        total_tables = sum(len(table_list) for table_list in tables_dict.values())
        print(f"Total tables before grouping: {total_tables}")
        
        # Apply grouping with column information
        grouped_analysis = self.pattern_analyzer.analyze_and_group_tables(tables_dict, table_column_info)
        
        total_representative_tables = sum(len(filtered_list) for filtered_list in grouped_analysis['filtered_tables'].values())
        reduction_percentage = ((total_tables - total_representative_tables) / total_tables * 100) if total_tables > 0 else 0
        
        print(f"Tables after grouping: {total_representative_tables}")
        print(f"Reduction: {reduction_percentage:.1f}%")
        
        return grouped_analysis
    
    def create_grouped_structure(self, original_structure: Dict[str, Any], 
                                grouped_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create structure with only representative tables"""
        
        print("\n=== CREATING GROUPED STRUCTURE ===")
        
        grouped_structure = {
            "databases": {}
        }
        
        for db_name, db_data in original_structure["databases"].items():
            grouped_structure["databases"][db_name] = {
                "name": db_name,
                "schemas": {}
            }
            
            for schema_name, schema_data in db_data["schemas"].items():
                schema_key = f"{db_name}.{schema_name}"
                
                if schema_key not in grouped_analysis['filtered_tables']:
                    continue
                
                representative_tables = grouped_analysis['filtered_tables'][schema_key]
                
                grouped_structure["databases"][db_name]["schemas"][schema_name] = {
                    "name": schema_name,
                    "tables": {}
                }
                
                for table_name in representative_tables:
                    if table_name not in schema_data["tables"]:
                        continue
                    
                    original_table_info = schema_data["tables"][table_name].copy()
                    
                    # Add grouping information
                    full_table_name = f"{db_name}.{schema_name}.{table_name}"
                    group_info = grouped_analysis['group_info'].get(full_table_name)
                    table_column_inf = grouped_analysis['table_column_inf'].get(full_table_name)
                    
                    if group_info:
                        original_table_info["table_info"] = group_info
                        print(f"    🔗 {table_name}: {group_info}")
                    else:
                        print(f"    📋 {table_name}: Standalone table")
                    
                    if table_column_inf:
                        original_table_info["table_column_inf"] = table_column_inf
                        print(f"    📊 {table_name}: {table_column_inf}")
                    
                    grouped_structure["databases"][db_name]["schemas"][schema_name]["tables"][table_name] = original_table_info
        
        return grouped_structure
    
    def apply_pkfk_detection(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Apply PK/FK detection to the structure"""
        
        print("\n=== APPLYING PK/FK DETECTION ===")
        
        # Create key finder for the structure
        key_finder = LocalJSONDatabaseKeyFinder(structure)
        
        # Run PK/FK analysis
        key_analysis = key_finder.analyze()
        
        print(f"PK/FK detection completed:")
        print(f"  Tables with primary keys: {len(key_analysis['primary_keys'])}")
        print(f"  Tables with foreign keys: {len(key_analysis['foreign_keys'])}")
        print(f"  Total FK relationships: {sum(len(fks) for fks in key_analysis['foreign_keys'].values())}")
        
        return key_analysis
    
    def generate_simple_output(self, structure: Dict[str, Any], key_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate simple output format matching the sample"""
        
        print("\n=== GENERATING SIMPLE OUTPUT ===")
        
        output = {
            "tables": {},
            "relationships": []
        }
        
        # Process each table
        for db_name, db_data in structure["databases"].items():
            for schema_name, schema_data in db_data["schemas"].items():
                for table_name, table_data in schema_data["tables"].items():
                    
                    # Get column information
                    column_names = table_data.get("column_names", [])
                    column_types = table_data.get("column_types", [])
                    sample_rows = table_data.get("sample_rows", [])
                    
                    # Create columns list
                    columns = []
                    for i, col_name in enumerate(column_names):
                        col_type = column_types[i] if i < len(column_types) else "UNKNOWN"
                        
                        column_info = {
                            "name": col_name,
                            "type": col_type,
                            "is_primary_key": False,
                            "is_foreign_key": False
                        }
                        
                        # Check if this column is a primary key
                        if table_name in key_analysis['primary_keys']:
                            pk_info = key_analysis['primary_keys'][table_name]
                            if col_name in pk_info['columns']:
                                column_info["is_primary_key"] = True
                                column_info["pk_origin"] = pk_info['origin']
                        
                        # Check if this column is a foreign key
                        if table_name in key_analysis['foreign_keys']:
                            for fk in key_analysis['foreign_keys'][table_name]:
                                if fk['from'] == col_name:
                                    column_info["is_foreign_key"] = True
                                    column_info["fk_origin"] = fk['origin']
                                    to_full = fk['to_table']
                                    if to_full.count('.') == 0:
                                        to_full = f"{db_name}.{schema_name}.{to_full}"
                                    elif to_full.count('.') == 1:
                                        to_full = f"{db_name}.{to_full}"
                                    # else already fully qualified
                                    column_info["references_table"] = to_full
                                    column_info["references_column"] = fk['to_column']
                                    break
                        
                        columns.append(column_info)
                    
                    # Create samples dictionary
                    samples = {}
                    for i, col_name in enumerate(column_names):
                        col_samples = []
                        for row in sample_rows:
                            if isinstance(row, list) and i < len(row):
                                col_samples.append(row[i])
                            elif isinstance(row, dict) and col_name in row:
                                col_samples.append(row[col_name])
                        samples[col_name] = col_samples
                    
                    # Store the full table name for this table (we'll need it for FK lookup)
                    full_table_name = f"{db_name}.{schema_name}.{table_name}"
                    
                    # Create foreign keys list for the table
                    foreign_keys = []
                    if table_name in key_analysis['foreign_keys']:
                        for fk in key_analysis['foreign_keys'][table_name]:
                            # Find the actual full table name for the reference
                            ref_table_full = None
                            for table_key in [f"{db_name}.{schema_name}.{table_name}" for db_name, db_data in structure["databases"].items() for schema_name, schema_data in db_data["schemas"].items() for table_name in schema_data["tables"].keys()]:
                                if table_key.endswith(f".{fk['to_table']}"):
                                    ref_table_full = table_key
                                    break
                            
                            # Fallback if not found
                            if not ref_table_full:
                                ref_table_full = fk['to_table']
                            
                            fk_entry = {
                                "column": fk['from'],
                                "references": {
                                    "table": ref_table_full,  # Now uses the full table name
                                    "column": fk['to_column']
                                },
                                "fk_origin": fk['origin'],
                                "confidence": fk.get('confidence', 'medium')
                            }
                            foreign_keys.append(fk_entry)
                    
                    # Build table entry
                    table_entry = {
                        "name": full_table_name,
                        "columns": columns,
                        "samples": samples,
                        "foreign_keys": foreign_keys
                    }
                    
                    # Add table_info if it exists (for grouped tables)
                    if "table_info" in table_data:
                        table_entry["table_info"] = table_data["table_info"]
                    
                    # Add table_column_inf if it exists (for grouped tables)
                    if "table_column_inf" in table_data:
                        table_entry["table_column_inf"] = table_data["table_column_inf"]
                    
                    output["tables"][full_table_name] = table_entry

        
        # Process relationships
        for src_table, fks in key_analysis['foreign_keys'].items():
            for fk in fks:
                # Use table names as they appear in the tables section keys
                # Find the actual full table name from the tables dictionary
                from_full = None
                to_full = None
                
                # Find source table full name
                for table_key in output["tables"].keys():
                    if table_key.endswith(f".{src_table}"):
                        from_full = table_key
                        break
                
                # Find destination table full name  
                for table_key in output["tables"].keys():
                    if table_key.endswith(f".{fk['to_table']}"):
                        to_full = table_key
                        break
                
                # Fallback if not found (shouldn't happen but safety check)
                if not from_full:
                    from_full = src_table
                if not to_full:
                    to_full = fk['to_table']

                relationship = {
                    "from_table": from_full,
                    "from_column": fk['from'],
                    "to_table": to_full,
                    "to_column": fk['to_column'],
                    "type": "simple"
                }
                output["relationships"].append(relationship)
        return output



    
    def save_simple_output(self, output: Dict[str, Any], database_name: str, 
                          output_dir: str = "./analysis_results") -> str:
        """Save simple output to JSON file"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"{database_name}_db_summary.json")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"Simple output saved: {output_file}")
            return output_file
        except Exception as e:
            print(f"Error saving simple output: {e}")
            return None
    
    def run_simple_analysis(self, database_names: Union[str, List[str]], 
                           output_dir: str = "./analysis_results") -> Dict[str, Any]:
        """Run complete analysis and generate simple output format"""
        
        print("STARTING SIMPLE DATABASE ANALYSIS")
        print("=" * 60)
        
        if database_names == "all":
            # Get all available databases
            available_dbs = [d for d in os.listdir(self.base_path) 
                           if os.path.isdir(os.path.join(self.base_path, d))]
            database_names = available_dbs
            print(f"Processing ALL databases: {database_names}")
        elif isinstance(database_names, str):
            database_names = [database_names]
            print(f"Processing specific database: {database_names}")
        else:
            print(f"Processing specific databases: {database_names}")
        
        results = {}
        
        for db_name in database_names:
            print(f"\n{'='*20} PROCESSING {db_name} {'='*20}")
            
            try:
                # Step 1: Load database structure
                print(f"Loading database structure for {db_name}...")
                structure = self.load_database_structure(db_name)
                
                # Step 2: Apply table grouping
                grouped_analysis = self.apply_table_grouping(structure)
                
                # Step 3: Create grouped structure
                grouped_structure = self.create_grouped_structure(structure, grouped_analysis)
                
                # Step 4: Apply PK/FK detection
                key_analysis = self.apply_pkfk_detection(grouped_structure)
                
                # Step 5: Generate simple output
                simple_output = self.generate_simple_output(grouped_structure, key_analysis)
                
                # Step 6: Save simple output
                print(f"\nSaving results for {db_name}...")
                saved_file = self.save_simple_output(simple_output, db_name, output_dir)
                
                results[db_name] = {
                    "output": simple_output,
                    "saved_file": saved_file,
                    "status": "success"
                }
                
                # Print summary for this database
                self._print_simple_summary(db_name, simple_output)
                
            except Exception as e:
                print(f"Error processing database {db_name}: {e}")
                results[db_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Print final summary
        self._print_final_simple_summary(results, output_dir)
        
        return results
    
    def _print_simple_summary(self, db_name: str, output: Dict[str, Any]):
        """Print summary for a single database"""
        
        print(f"\nSUMMARY FOR {db_name}")
        print("-" * 40)
        
        tables = output.get("tables", {})
        relationships = output.get("relationships", [])
        
        print(f"Tables: {len(tables)}")
        print(f"Relationships: {len(relationships)}")
        
        # Count tables with grouping info
        grouped_tables = sum(1 for table in tables.values() if "table_info" in table)
        print(f"Grouped Tables: {grouped_tables}")
        print(f"Standalone Tables: {len(tables) - grouped_tables}")
        
        # Count primary and foreign keys
        tables_with_pk = 0
        tables_with_fk = 0
        
        for table in tables.values():
            has_pk = any(col["is_primary_key"] for col in table["columns"])
            has_fk = any(col["is_foreign_key"] for col in table["columns"])
            
            if has_pk:
                tables_with_pk += 1
            if has_fk:
                tables_with_fk += 1
        
        print(f"Tables with PK: {tables_with_pk}")
        print(f"Tables with FK: {tables_with_fk}")
    
    def _print_final_simple_summary(self, results: Dict[str, Any], output_dir: str):
        """Print final summary of all processed databases"""
        
        print(f"\n{'='*60}")
        print("SIMPLE DATABASE ANALYSIS COMPLETED!")
        print(f"{'='*60}")
        
        successful_dbs = [db for db, result in results.items() if result.get("status") == "success"]
        failed_dbs = [db for db, result in results.items() if result.get("status") == "error"]
        
        print(f"Successfully processed: {len(successful_dbs)} databases")
        if successful_dbs:
            print(f"   {', '.join(successful_dbs)}")
        
        if failed_dbs:
            print(f"Failed to process: {len(failed_dbs)} databases")
            print(f"   {', '.join(failed_dbs)}")
        
        print(f"\nResults saved to: {output_dir}")
        
        # Show sample files created
        if successful_dbs:
            sample_db = successful_dbs[0]
            sample_file = results[sample_db].get("saved_file")
            if sample_file:
                print(f"\nSample file created: {os.path.basename(sample_file)}")


def extract_local_db_summary(base_path: str,
                            database_name: str = None,
                            sample_limit: int = 10,
                            include_samples: bool = True,
                            include_column_names: bool = True,
                            include_data_types: bool = True,
                            detect_primary_keys: bool = True,
                            detect_foreign_keys: bool = True,
                            include_key_confidence: bool = True,
                            include_row_count: bool = False,
                            include_column_count: bool = True,
                            skip_empty_tables: bool = False,
                            include_table_relationships: bool = True,
                            apply_table_grouping: bool = True,
                            max_string_display_length: int = 100):
    """
    Extract comprehensive local database summary including schema, statistics, and sample data.
    
    Args:
        base_path: Base path to database directories
        database_name: Specific database to analyze (None for all accessible)
        sample_limit: Number of sample rows per table
        include_samples: Include sample data
        include_column_names: Include column names
        include_data_types: Include data types
        detect_primary_keys: Detect primary keys
        detect_foreign_keys: Detect foreign keys
        include_key_confidence: Include key detection confidence
        include_row_count: Include row counts
        include_column_count: Include column counts
        skip_empty_tables: Skip empty tables
        include_table_relationships: Include relationship summary
        apply_table_grouping: Apply table grouping to reduce table count
        max_string_display_length: String truncation length
        
    Returns:
        Dictionary containing database summary in simple format
    """
    
    try:
        # Create analyzer
        analyzer = LocalDatabaseAnalyzer(base_path)
        
        # Determine which databases to process
        if database_name:
            databases_to_process = [database_name]
        else:
            # Get all available databases
            available_dbs = [d for d in os.listdir(base_path) 
                           if os.path.isdir(os.path.join(base_path, d))]
            databases_to_process = available_dbs
        
        all_results = {}
        
        for db_name in databases_to_process:
            print(f"Processing database: {db_name}")
            
            # Step 1: Load database structure
            structure = analyzer.load_database_structure(db_name)
            
            # Step 2: Apply table grouping if requested
            if apply_table_grouping:
                grouped_analysis = analyzer.apply_table_grouping(structure)
                grouped_structure = analyzer.create_grouped_structure(structure, grouped_analysis)
            else:
                # Create a dummy grouped_analysis but still normalize the structure
                grouped_analysis = {'filtered_tables': {}, 'group_info': {}, 'table_column_inf': {}}
                # Populate filtered_tables with all tables (no filtering)
                for db_name, db_data in structure["databases"].items():
                    for schema_name, schema_data in db_data["schemas"].items():
                        schema_key = f"{db_name}.{schema_name}"
                        grouped_analysis['filtered_tables'][schema_key] = list(schema_data["tables"].keys())
                # Use create_grouped_structure to normalize even without grouping
                grouped_structure = analyzer.create_grouped_structure(structure, grouped_analysis)
            # Step 3: Apply PK/FK detection
            key_analysis = analyzer.apply_pkfk_detection(grouped_structure)
            
            # Step 4: Generate enhanced output with additional statistics
            db_summary = _generate_enhanced_output(
                grouped_structure, 
                key_analysis, 
                grouped_analysis,
                sample_limit=sample_limit,
                include_samples=include_samples,
                include_column_names=include_column_names,
                include_data_types=include_data_types,
                detect_primary_keys=detect_primary_keys,
                detect_foreign_keys=detect_foreign_keys,
                include_key_confidence=include_key_confidence,
                include_row_count=include_row_count,
                include_column_count=include_column_count,
                skip_empty_tables=skip_empty_tables,
                include_table_relationships=include_table_relationships,
                max_string_display_length=max_string_display_length
            )
            
            all_results[db_name] = db_summary
        
        # If only one database, return it directly, otherwise return all
        if len(all_results) == 1:
            return list(all_results.values())[0]
        else:
            return all_results
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def _generate_enhanced_output(structure: Dict[str, Any], 
                            key_analysis: Dict[str, Any],
                            grouped_analysis: Dict[str, Any],
                            sample_limit: int = 10,
                            include_samples: bool = True,
                            include_column_names: bool = True,
                            include_data_types: bool = True,
                            detect_primary_keys: bool = True,
                            detect_foreign_keys: bool = True,
                            include_key_confidence: bool = True,
                            include_row_count: bool = False,
                            include_column_count: bool = True,
                            skip_empty_tables: bool = False,
                            include_table_relationships: bool = True,
                            max_string_display_length: int = 100) -> Dict[str, Any]:
    """Generate enhanced output with additional statistics similar to Snowflake function"""
    
    output = {
        "tables": {},
        "relationships": []
    }
    
    # Process each table
    for db_name, db_data in structure["databases"].items():
        for schema_name, schema_data in db_data["schemas"].items():
            for table_name, table_data in schema_data["tables"].items():
                
                # Define full_table_name at the beginning of the loop
                full_table_name = f"{db_name}.{schema_name}.{table_name}"
                
                # Skip empty tables if requested
                if skip_empty_tables:
                    sample_rows = table_data.get("sample_rows", [])
                    if not sample_rows:
                        continue
                
                # Get column information
                column_names = table_data.get("column_names", [])
                column_types = table_data.get("column_types", [])
                sample_rows = table_data.get("sample_rows", [])
                
                # Create columns list
                columns = []
                for i, col_name in enumerate(column_names):
                    if not include_column_names and not include_data_types:
                        continue
                        
                    col_type = column_types[i] if i < len(column_types) else "UNKNOWN"
                    
                    column_info = {}
                    
                    if include_column_names:
                        column_info["name"] = col_name
                    
                    if include_data_types:
                        column_info["type"] = col_type
                    
                    # Initialize key flags
                    if detect_primary_keys:
                        column_info["is_primary_key"] = False
                    
                    if detect_foreign_keys:
                        column_info["is_foreign_key"] = False
                    
                    # Check if this column is a primary key
                    if detect_primary_keys and table_name in key_analysis['primary_keys']:
                        pk_info = key_analysis['primary_keys'][table_name]
                        if col_name in pk_info['columns']:
                            column_info["is_primary_key"] = True
                            if include_key_confidence:
                                column_info["pk_origin"] = pk_info['origin']
                    
                    # Check if this column is a foreign key
                    if detect_foreign_keys and table_name in key_analysis['foreign_keys']:
                        for fk in key_analysis['foreign_keys'][table_name]:
                            if fk['from'] == col_name:
                                column_info["is_foreign_key"] = True
                                if include_key_confidence:
                                    column_info["fk_origin"] = fk['origin']

                                    # Find the actual full table name for the reference
                                    to_full = None
                                    for table_key in [f"{db_name}.{schema_name}.{table_name}" for db_name, db_data in structure["databases"].items() for schema_name, schema_data in db_data["schemas"].items() for table_name in schema_data["tables"].keys()]:
                                        if table_key.endswith(f".{fk['to_table']}"):
                                            to_full = table_key
                                            break
                                    
                                    # Fallback if not found
                                    if not to_full:
                                        to_full = fk['to_table']
                                    
                                    column_info["references_table"] = to_full
                                    column_info["references_column"] = fk['to_column']
                                break

                    columns.append(column_info)

                
                # Create samples dictionary
                samples = {}
                if include_samples:
                    for i, col_name in enumerate(column_names):
                        col_samples = []
                        sample_count = 0
                        
                        for row in sample_rows:
                            if sample_count >= sample_limit:
                                break
                                
                            value = None
                            if isinstance(row, list) and i < len(row):
                                value = row[i]
                            elif isinstance(row, dict) and col_name in row:
                                value = row[col_name]
                            
                            # Handle string truncation
                            if isinstance(value, str) and len(value) > max_string_display_length:
                                value = value[:max_string_display_length-3] + "..."
                            
                            col_samples.append(value)
                            sample_count += 1
                        
                        samples[col_name] = col_samples

                # Create foreign keys list for the table
                foreign_keys = []
                if detect_foreign_keys and table_name in key_analysis['foreign_keys']:
                    for fk in key_analysis['foreign_keys'][table_name]:
                        # Find the actual full table name for the reference
                        ref_table_full = None
                        for table_key in [f"{db_name}.{schema_name}.{table_name}" for db_name, db_data in structure["databases"].items() for schema_name, schema_data in db_data["schemas"].items() for table_name in schema_data["tables"].keys()]:
                            if table_key.endswith(f".{fk['to_table']}"):
                                ref_table_full = table_key
                                break
                        
                        # Fallback if not found
                        if not ref_table_full:
                            ref_table_full = fk['to_table']
                        
                        fk_entry = {
                            "column": fk['from'],
                            "references": {
                                "table": ref_table_full,  # Now uses the full table name
                                "column": fk['to_column']
                            }
                        }
                        
                        if include_key_confidence:
                            fk_entry["fk_origin"] = fk['origin']
                            fk_entry["confidence"] = fk.get('confidence', 'medium')
                        
                        foreign_keys.append(fk_entry)
                
                # Build table entry (full_table_name is already defined above)
                table_entry = {
                    "name": full_table_name
                }
                
                if columns:
                    table_entry["columns"] = columns
                
                if include_samples and samples:
                    table_entry["samples"] = samples
                
                if foreign_keys:
                    table_entry["foreign_keys"] = foreign_keys
                
                # Add column count if requested
                if include_column_count:
                    table_entry["column_count"] = len(column_names)
                
                # Add row count if requested (estimate from samples)
                if include_row_count:
                    table_entry["row_count"] = len(sample_rows)  # This is just sample count, not actual row count
                
                # Add table_info if it exists (for grouped tables)
                group_info = grouped_analysis.get('group_info', {}).get(full_table_name)
                if group_info:
                    table_entry["table_info"] = group_info
                
                # Add table_column_inf if it exists (for grouped tables)
                table_column_inf = grouped_analysis.get('table_column_inf', {}).get(full_table_name)
                if table_column_inf:
                    table_entry["table_column_inf"] = table_column_inf
                
                # Store the table entry
                output["tables"][full_table_name] = table_entry

    
    # Process relationships
    if include_table_relationships:
        for src_table, fks in key_analysis['foreign_keys'].items():
            for fk in fks:
                # Use table names as they appear in the tables section keys
                # Find the actual full table name from the tables dictionary
                from_full = None
                to_full = None
                
                # Find source table full name
                for table_key in output["tables"].keys():
                    if table_key.endswith(f".{src_table}"):
                        from_full = table_key
                        break
                
                # Find destination table full name  
                for table_key in output["tables"].keys():
                    if table_key.endswith(f".{fk['to_table']}"):
                        to_full = table_key
                        break
                
                # Fallback if not found (shouldn't happen but safety check)
                if not from_full:
                    from_full = src_table
                if not to_full:
                    to_full = fk['to_table']

                relationship = {
                    "from_table": from_full,
                    "from_column": fk['from'],
                    "to_table": to_full,
                    "to_column": fk['to_column'],
                    "type": "simple"
                }
                output["relationships"].append(relationship)

    return output



def save_local_db_summary(db_summary, output_path, indent=2):
    """
    Save local database summary to JSON file
    """
    def json_serialize(obj):
        if hasattr(obj, 'isoformat'):  # Handle datetime objects
            return obj.isoformat()
        return str(obj)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(db_summary, f, default=json_serialize, indent=indent, ensure_ascii=False)
    
    print(f"Local database summary saved to {output_path}")


def main():
    """Main function with command line interface"""
    
    # Simple configuration
    BASE_PATH = os.environ.get("SNOWFLAKE_LOCAL_DB_ROOT", "./snowflake_dbs")
    DATABASES_TO_PROCESS = ['NOAA_GSOD'] #'all' #or ['AUSTIN'] for specific database
    
    parser = argparse.ArgumentParser(description='Simple Local Database Analysis with Grouping and PK/FK Detection')
    
    parser.add_argument('--base-path', help='Base path to database directories', default=BASE_PATH)
    parser.add_argument('--databases', nargs='+', help='Specific databases to process', default=DATABASES_TO_PROCESS)
    parser.add_argument('--output-dir', help='Output directory for results', default='./results_snow_d_v2')
    parser.add_argument('--use-extract-function', action='store_true', help='Use extract_local_db_summary function instead')
    
    args = parser.parse_args()
    
    try:
        if args.use_extract_function:
            # Use the new extract function (similar to Snowflake version)
            if len(args.databases) == 1 and args.databases[0] != 'all':
                db_summary = extract_local_db_summary(
                    base_path=args.base_path,
                    database_name=args.databases[0],
                    sample_limit=10,
                    include_samples=True,
                    detect_primary_keys=True,
                    detect_foreign_keys=True,
                    apply_table_grouping=True
                )
                
                # Save the result
                output_file = os.path.join(args.output_dir, f"{args.databases[0]}_extract_summary.json")
                save_local_db_summary(db_summary, output_file)
                
                return db_summary
            else:
                print("Extract function works best with single database. Use --databases DBNAME")
                return None
        else:
            # Use the original analyzer approach
            analyzer = LocalDatabaseAnalyzer(args.base_path)
            
            # Run simple analysis
            results = analyzer.run_simple_analysis(
                database_names=args.databases,
                output_dir=args.output_dir
            )
            
            return results
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()