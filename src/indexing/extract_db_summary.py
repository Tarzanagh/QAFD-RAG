import sqlite3
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import os
import datetime
import statistics
import re
from collections import defaultdict

# DB-specific table-specific columns to skip
# Format: {'database_name': {'table_name': ['column1', 'column2', ...]}}
DB_TABLE_SPECIFIC_SKIP_COLUMNS = {
    'Db-IMDB.sqlite': {
        'Movie': ['index'],
        'Genre': ['index'],
        'Language': ['index'],
        'Country': ['index'],
        'Location': ['index'],
        'M_Location': ['index', 'ID'],
        'M_Country': ['index', 'ID'],
        'M_Language': ['index', 'ID'],
        'M_Genre': ['index', 'ID'],
        'Person': ['index'],
        'M_Producer': ['index', 'ID'],
        'M_Director': ['index', 'ID'],
        'M_Cast': ['index', 'ID']
    }
    # Add more database and table-specific skip rules as needed
}

def quote_column_name(col_name):
    """Properly quote column names that need it"""
    if not col_name:
        return '""'
    
    # Special characters that require quoting
    special_chars = ['%', '-', '+', '*', '/', '(', ')', '[', ']', ' ', '.', '&', '|', '!', '@', '#', '$', '^', '~', '`', '=', '<', '>', '?', ',', ';', ':', "'", '"']
    
    # SQL reserved words (common ones)
    reserved_words = [
        'index', 'key', 'order', 'group', 'from', 'select', 'where', 'table', 'column',
        'and', 'or', 'not', 'in', 'is', 'null', 'like', 'between', 'exists', 'case',
        'when', 'then', 'else', 'end', 'union', 'all', 'distinct', 'having', 'limit',
        'offset', 'join', 'inner', 'outer', 'left', 'right', 'full', 'cross', 'on',
        'using', 'natural', 'as', 'desc', 'asc', 'primary', 'foreign', 'references',
        'constraint', 'unique', 'check', 'default', 'create', 'drop', 'alter', 'insert',
        'update', 'delete', 'into', 'values', 'set', 'truncate', 'commit', 'rollback',
        'transaction', 'begin', 'savepoint', 'release', 'pragma', 'vacuum', 'analyze',
        'explain', 'view', 'trigger', 'procedure', 'function', 'database', 'schema'
    ]
    
    # Check if quoting is needed
    needs_quoting = (
        any(char in col_name for char in special_chars) or
        col_name.lower() in reserved_words or
        col_name.isdigit() or
        (col_name and col_name[0].isdigit())
    )
    
    if needs_quoting:
        # Escape any existing double quotes by doubling them
        escaped_name = col_name.replace('"', '""')
        return f'"{escaped_name}"'
    return col_name

def quote_table_name(table_name):
    """Properly quote table names that need it"""
    return quote_column_name(table_name)  # Same logic applies

class SQLiteKeyFinder:
    """
    A tool to identify potential primary and foreign keys in SQLite databases
    where they are not explicitly defined.
    """
    
    def __init__(self, db_path):
        """Initialize with the path to the SQLite database."""
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        self.db_path = db_path
        self.db_name = os.path.basename(db_path)
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.tables = []
        self.table_columns = {}
        self.primary_keys = {}
        self.foreign_keys = defaultdict(list)
        
    def should_skip_column(self, table_name, column_name):
        """
        Determine if a column should be skipped based on database-specific table rules.
        """
        if self.db_name in DB_TABLE_SPECIFIC_SKIP_COLUMNS and \
           table_name in DB_TABLE_SPECIFIC_SKIP_COLUMNS[self.db_name] and \
           column_name.lower() in [col.lower() for col in DB_TABLE_SPECIFIC_SKIP_COLUMNS[self.db_name][table_name]]:
            return True
        return False

    def _get_tables(self):
        """Get all tables in the database."""
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        self.tables = [row[0] for row in self.cursor.fetchall() 
                       if not row[0].startswith('sqlite_')]
        return self.tables
    
    def _get_table_columns(self):
        """Get all columns for each table."""
        for table in self.tables:
            quoted_table = quote_table_name(table)
            self.cursor.execute(f"PRAGMA table_info({quoted_table});")
            columns = []
            for col_info in self.cursor.fetchall():
                col_name = col_info[1]
                data_type = col_info[2]
                columns.append((col_name, data_type))
            self.table_columns[table] = columns
        return self.table_columns
    
    def _check_defined_keys(self):
        """Check for already defined primary and foreign keys."""
        defined_pk = {}
        defined_fk = defaultdict(list)
        
        # Check primary keys
        for table in self.tables:
            quoted_table = quote_table_name(table)
            self.cursor.execute(f"PRAGMA table_info({quoted_table});")
            pk_columns = []
            for col_info in self.cursor.fetchall():
                if col_info[5] > 0:  # pk column
                    pk_columns.append(col_info[1])
            if pk_columns:
                defined_pk[table] = {'columns': pk_columns, 'origin': 'db'}
                
        # Check foreign keys
        for table in self.tables:
            quoted_table = quote_table_name(table)
            self.cursor.execute(f"PRAGMA foreign_key_list({quoted_table});")
            for fk_info in self.cursor.fetchall():
                defined_fk[table].append({
                    'from': fk_info[3],
                    'to_table': fk_info[2],
                    'to_column': fk_info[4],
                    'origin': 'db'
                })
                
        return defined_pk, defined_fk

    def find_potential_primary_keys(self):
        """Identify columns that are likely to be primary keys."""
        # First check for explicitly defined keys
        defined_pk, _ = self._check_defined_keys()
        if defined_pk:
            print("Found defined primary keys:", defined_pk)
            self.primary_keys.update(defined_pk)
        
        # For tables without defined primary keys, try to identify them
        for table in self.tables:
            if table in self.primary_keys:
                continue
            
            quoted_table = quote_table_name(table)
            self.cursor.execute(f"SELECT COUNT(*) FROM {quoted_table};")
            total_rows = self.cursor.fetchone()[0]
            
            if total_rows == 0:
                continue
            
            pk_candidates = {}
            
            for col_name, data_type in self.table_columns[table]:
                if self.should_skip_column(table, col_name):
                    continue
                
                try:
                    pk_score = 0
                    
                    # Handle reserved keyword column names
                    quoted_col_name = quote_column_name(col_name)
                    
                    # Check uniqueness
                    self.cursor.execute(f"SELECT COUNT(DISTINCT {quoted_col_name}) FROM {quoted_table};")
                    distinct_values = self.cursor.fetchone()[0]
                    uniqueness_ratio = distinct_values / max(1, total_rows)
                    
                    if uniqueness_ratio < 0.9:
                        continue
                    
                    # Check for NULL values
                    self.cursor.execute(f"SELECT COUNT(*) FROM {quoted_table} WHERE {quoted_col_name} IS NULL;")
                    null_count = self.cursor.fetchone()[0]
                    null_ratio = null_count / max(1, total_rows)
                    
                    if null_ratio > 0.1:
                        continue
                    
                    # Score uniqueness
                    if uniqueness_ratio == 1.0:
                        pk_score += 30
                    elif uniqueness_ratio > 0.98:
                        pk_score += 20
                    
                    # No nulls is good for PKs
                    if null_count == 0:
                        pk_score += 20
                    
                    # Data type scoring
                    if 'int' in data_type.lower():
                        pk_score += 15
                    elif data_type.lower() in ['text', 'varchar', 'char', 'string']:
                        pk_score += 5
                    
                    # Check for naming patterns
                    name_patterns = [
                        (r'^id$', 15),
                        (r'^{}_id$'.format(table), 15),
                        (r'^{}_key$'.format(table), 15),
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
                    
                    # Check for auto-increment indication
                    try:
                        self.cursor.execute(f"SELECT MIN({quoted_col_name}), MAX({quoted_col_name}) FROM {quoted_table};")
                        min_val, max_val = self.cursor.fetchone()
                        
                        if (isinstance(min_val, int) and isinstance(max_val, int) and 
                            min_val in [0, 1] and max_val == total_rows):
                            pk_score += 15
                            
                        if total_rows <= 1000:
                            self.cursor.execute(f"""
                                WITH numbers AS (
                                    SELECT ROW_NUMBER() OVER (ORDER BY {quoted_col_name}) + {min_val} - 1 as expected,
                                    {quoted_col_name} as actual
                                    FROM {quoted_table}
                                    ORDER BY {quoted_col_name}
                                )
                                SELECT COUNT(*) FROM numbers
                                WHERE expected != actual;
                            """)
                            gaps = self.cursor.fetchone()[0]
                            if gaps == 0:
                                pk_score += 10
                    except (sqlite3.OperationalError, TypeError):
                        pass
                    
                    # Check for index presence
                    self.cursor.execute(f"PRAGMA index_list({quoted_table});")
                    indexes = self.cursor.fetchall()
                    
                    for idx_info in indexes:
                        idx_name = idx_info[1]
                        is_unique = idx_info[2]
                        
                        self.cursor.execute(f"PRAGMA index_info({quote_column_name(idx_name)});")
                        index_columns = [info[2] for info in self.cursor.fetchall()]
                        
                        if col_name in index_columns and len(index_columns) == 1:
                            pk_score += 10
                            if is_unique:
                                pk_score += 10
                    
                    # Add to candidates if score is high enough
                    if pk_score >= 25:
                        pk_candidates[col_name] = {
                            'score': pk_score,
                            'data_type': data_type,
                            'uniqueness': uniqueness_ratio,
                            'null_ratio': null_ratio
                        }
                        
                except sqlite3.OperationalError as e:
                    print(f"Error analyzing column {table}.{col_name}: {e}")
                    continue
            
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
                
                self.primary_keys[table] = {
                    'columns': pk_columns,
                    'origin': 'potential'
                }
                    
        return self.primary_keys

    def find_potential_foreign_keys(self):
        """Identify columns that are likely to be foreign keys."""
        # First check for explicitly defined foreign keys
        _, defined_fk = self._check_defined_keys()
        if defined_fk:
            print("Found defined foreign keys:", dict(defined_fk))
            self.foreign_keys.update(defined_fk)
        
        # Find potential foreign keys
        for src_table in self.tables:
            defined_fks_for_table = self.foreign_keys.get(src_table, [])
            defined_fk_cols = set(fk['from'] for fk in defined_fks_for_table)
            
            processed_relationships = set()
            for fk in defined_fks_for_table:
                processed_relationships.add((fk['from'], fk['to_table'], fk['to_column']))
            
            quoted_src_table = quote_table_name(src_table)
            
            for src_col_name, src_data_type in self.table_columns[src_table]:
                if self.should_skip_column(src_table, src_col_name):
                    continue
                    
                if src_col_name in defined_fk_cols:
                    continue
                
                quoted_src_col_name = quote_column_name(src_col_name)
                
                for ref_table in self.tables:
                    if src_table == ref_table:
                        continue
                    
                    if ref_table not in self.primary_keys:
                        continue
                    
                    ref_col_list = self.primary_keys[ref_table].get('columns', [])
                    if not ref_col_list:
                        continue
                    
                    quoted_ref_table = quote_table_name(ref_table)
                    
                    for ref_col in ref_col_list:
                        if self.should_skip_column(ref_table, ref_col):
                            continue
                            
                        quoted_ref_col = quote_column_name(ref_col)
                        
                        if (src_col_name, ref_table, ref_col) in processed_relationships:
                            continue
                        
                        # Define naming patterns for foreign keys
                        fk_patterns = [
                            r'^{}_{}$'.format(ref_table, ref_col),
                            r'^{}{}$'.format(ref_table, ref_col.capitalize()),
                            r'^{}_id$'.format(ref_table),
                            r'^{}Id$'.format(ref_table),
                            r'^{}_key$'.format(ref_table),
                            r'^fk_{}_'.format(ref_table),
                            r'^{}$'.format(ref_col)
                        ]
                        
                        name_pattern_match = False
                        for pattern in fk_patterns:
                            if re.match(pattern, src_col_name, re.IGNORECASE):
                                name_pattern_match = True
                                break
                        
                        ref_col_type = next((dtype for col, dtype in self.table_columns[ref_table] 
                                           if col == ref_col), None)
                        
                        if name_pattern_match:
                            if ref_col_type and src_data_type and ref_col_type.lower() != src_data_type.lower():
                                confidence = "low"
                            else:
                                confidence = "medium"
                            
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
                                if self.get_row_count(src_table) > 10000 or self.get_row_count(ref_table) > 10000:
                                    continue
                                
                                self.cursor.execute(f"""
                                    SELECT COUNT(*) FROM {quoted_src_table} 
                                    WHERE {quoted_src_col_name} IS NOT NULL
                                    AND {quoted_src_col_name} NOT IN (
                                        SELECT {quoted_ref_col} FROM {quoted_ref_table}
                                    )
                                """)
                                invalid_refs = self.cursor.fetchone()[0]
                                
                                if invalid_refs == 0:
                                    self.cursor.execute(f"""
                                        SELECT COUNT(DISTINCT {quoted_src_col_name}) 
                                        FROM {quoted_src_table}
                                        WHERE {quoted_src_col_name} IS NOT NULL
                                    """)
                                    distinct_values = self.cursor.fetchone()[0]
                                    
                                    self.cursor.execute(f"""
                                        SELECT COUNT(DISTINCT {quoted_ref_col}) 
                                        FROM {quoted_ref_table}
                                        WHERE {quoted_ref_col} IS NOT NULL
                                    """)
                                    ref_distinct = self.cursor.fetchone()[0]
                                    
                                    coverage_ratio = distinct_values / max(1, ref_distinct)
                                    
                                    if (coverage_ratio > 0.01 and 
                                        (not ref_col_type or not src_data_type or 
                                         ref_col_type.lower() == src_data_type.lower())):
                                        
                                        self.foreign_keys[src_table].append({
                                            'from': src_col_name,
                                            'to_table': ref_table,
                                            'to_column': ref_col,
                                            'origin': 'potential',
                                            'confidence': 'high' if coverage_ratio > 0.3 else 'medium',
                                            'evidence': 'data_match'
                                        })
                                        
                                        processed_relationships.add((src_col_name, ref_table, ref_col))
                            except sqlite3.OperationalError as e:
                                print(f"Error checking foreign key relationship {src_table}.{src_col_name} -> {ref_table}.{ref_col}: {e}")
                                continue
                    
        return dict(self.foreign_keys)

    def get_row_count(self, table):
        """Get the number of rows in a table."""
        quoted_table = quote_table_name(table)
        self.cursor.execute(f"SELECT COUNT(*) FROM {quoted_table};")
        return self.cursor.fetchone()[0]
    
    def analyze(self):
        """Run the full analysis to find potential primary and foreign keys."""
        print(f"Analyzing database: {self.db_path}")
        
        self._get_tables()
        print(f"Found {len(self.tables)} tables: {', '.join(self.tables)}")
        
        self._get_table_columns()
        
        print("\nFinding potential primary keys...")
        pk_results = self.find_potential_primary_keys()
        
        print("\nFinding potential foreign keys...")
        fk_results = self.find_potential_foreign_keys()
        
        return {
            'tables': self.tables,
            'columns': self.table_columns,
            'primary_keys': pk_results,
            'foreign_keys': fk_results
        }
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()


def extract_db_summary_for_schema(db_path: str, 
                                sample_limit=5000,
                                include_samples=True,
                                include_column_names=True,
                                include_data_types=True,
                                detect_primary_keys=True,
                                detect_foreign_keys=True,
                                include_key_confidence=True,
                                include_row_count=True,                    # Enable: Critical for JOIN optimization
                                include_column_count=True,
                                include_distinct_count=True,               # Enable: Cardinality for indexing decisions
                                include_null_count=True,                   # Enable: NULL handling in queries
                                include_cardinality=True,                  # Enable: Index selectivity
                                include_nullability=True,                  # Enable: NULL safety validation
                                include_min_max=True,                      # Enable: Range query optimization
                                include_average=False,                     # Optional: Less critical for SQL
                                include_median=False,                      # Optional: Less critical for SQL
                                include_stddev=False,                      # Optional: Less critical for SQL
                                include_avg_length=True,                   # Enable: String column sizing
                                include_common_values=True,                # Enable: Pattern recognition
                                common_values_limit=5,
                                common_values_threshold=100,
                                include_date_range=True,                   # Enable: Time-based query optimization
                                include_date_range_days=True,              # Enable: Date range insights
                                include_not_null_constraint=True,          # Enable: Data quality validation
                                include_default_values=True,               # Enable: INSERT statement help
                                include_indexes=True,                      # Enable: Performance optimization
                                max_rows_for_expensive_stats=10000,
                                max_string_display_length=100,
                                max_binary_display_bytes=50,
                                include_db_metadata=True,                  # Enable: Database-level info
                                include_table_metadata=True,               # Enable: Table-level info
                                include_extraction_timestamp=True,         # Enable: Freshness tracking
                                skip_empty_tables=True,                    # Enable: Focus on relevant tables
                                include_table_relationships=True,
                                include_schema_summary=True):
    """
    Extract comprehensive database summary including schema, statistics, and sample data.
    
    Args:
        db_path: Path to SQLite database file
        sample_limit: Number of sample rows per table
        include_samples: Include sample data
        include_column_names: Include column names
        include_data_types: Include data types
        detect_primary_keys: Detect primary keys
        detect_foreign_keys: Detect foreign keys
        include_key_confidence: Include key detection confidence
        include_row_count: Include row counts
        include_column_count: Include column counts
        include_distinct_count: Include distinct value counts
        include_null_count: Include NULL value counts
        include_cardinality: Include uniqueness ratio
        include_nullability: Include NULL ratio
        include_min_max: Include min/max values
        include_average: Include average values
        include_median: Include median (expensive)
        include_stddev: Include standard deviation (expensive)
        include_avg_length: Include average text length
        include_common_values: Include most frequent values
        common_values_limit: How many common values to show
        common_values_threshold: Only for columns with <= N distinct values
        include_date_range: Include min/max dates
        include_date_range_days: Include date range in days
        include_not_null_constraint: Include NOT NULL constraints
        include_default_values: Include default values
        include_indexes: Include index information
        max_rows_for_expensive_stats: Limit for median/stddev calculation
        max_string_display_length: String truncation length
        max_binary_display_bytes: Show binary data info up to N bytes
        include_db_metadata: Include database metadata
        include_table_metadata: Include table metadata
        include_extraction_timestamp: Include timestamp
        skip_empty_tables: Skip empty tables
        include_table_relationships: Include relationship summary
        include_schema_summary: Include overall schema statistics
        
    Returns:
        Dictionary containing database summary
    """
    
    # Initialize key finder
    key_finder = None
    
    # Use SQLiteKeyFinder for enhanced PK/FK detection if requested
    if detect_primary_keys or detect_foreign_keys:
        key_finder = SQLiteKeyFinder(db_path)
        key_analysis = key_finder.analyze()
    else:
        key_analysis = {'tables': [], 'columns': {}, 'primary_keys': {}, 'foreign_keys': {}}
        # Still need to get basic table info
        conn_temp = sqlite3.connect(db_path)
        cursor_temp = conn_temp.cursor()
        cursor_temp.execute("SELECT name FROM sqlite_master WHERE type='table';")
        key_analysis['tables'] = [row[0] for row in cursor_temp.fetchall() 
                                  if not row[0].startswith('sqlite_')]
        conn_temp.close()
    
    # Connect to database for additional info extraction
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get database size if metadata is requested
    db_size = None
    if include_db_metadata:
        db_size = os.path.getsize(db_path) if os.path.exists(db_path) else None
    
    tables = key_analysis['tables']
    
    # Create summary structure
    db_summary = {}
    
    # Add metadata if requested
    if include_db_metadata:
        db_summary["metadata"] = {
            "db_file": db_path,
            "db_size_bytes": db_size,
            "table_count": len(tables)
        }
        
        if include_extraction_timestamp:
            db_summary["metadata"]["extracted_at"] = datetime.datetime.now().isoformat()
    
    # Initialize schema summary if requested
    schema_summary = {}
    if include_schema_summary:
        schema_summary = {
            "total_tables": len(tables),
            "total_columns": 0,
            "tables_with_primary_keys": 0,
            "tables_with_foreign_keys": 0,
            "total_relationships": 0
        }
    
    db_summary["tables"] = {}
    
    # Extract information for each table
    for table in tables:
        # Skip empty tables if requested
        quoted_table = quote_table_name(table)
        if skip_empty_tables:
            cursor.execute(f"SELECT COUNT(*) FROM {quoted_table};")
            if cursor.fetchone()[0] == 0:
                continue

        # Table schema
        cursor.execute(f"PRAGMA table_info({quoted_table});")
        columns_info = cursor.fetchall()
        
        # Table structure
        table_info = {"name": table}
        
        # Add table metadata if requested
        if include_table_metadata:
            table_info["column_count"] = 0  # We'll count only non-skipped columns
        
        # Get row count if requested
        if include_row_count:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {quoted_table};")
                table_info["row_count"] = cursor.fetchone()[0]
            except sqlite3.Error as e:
                print(f"Error getting row count for table {table}: {e}")
                table_info["row_count"] = -1
        
        # Initialize columns list
        table_info["columns"] = []
        
        # Initialize samples if requested
        if include_samples:
            table_info["samples"] = {}
        
        # Get column information - with enhanced PK detection
        if detect_primary_keys:
            pk_info = key_analysis['primary_keys'].get(table, {})
            primary_keys = pk_info.get('columns', [])
            pk_origin = pk_info.get('origin', 'potential')
        else:
            primary_keys = []
            pk_origin = 'unknown'
        
        # Track foreign key columns
        fk_columns = set()
        
        # Process columns, skipping those that should be excluded
        for col in columns_info:
            column_name = col["name"]
            
            # Skip columns that should be excluded based on DB_TABLE_SPECIFIC_SKIP_COLUMNS
            if key_finder and key_finder.should_skip_column(table, column_name):
                continue
            
            # Count this column for metadata
            if include_table_metadata:
                table_info["column_count"] += 1
            
            # Update schema summary
            if include_schema_summary:
                schema_summary["total_columns"] += 1
            
            # Basic column information
            column = {}
            
            if include_column_names:
                column["name"] = column_name
            
            if include_data_types:
                column["type"] = col["type"]
            
            # Primary key information
            if detect_primary_keys:
                is_primary_key = column_name in primary_keys
                column["is_primary_key"] = is_primary_key
                
                if is_primary_key and include_key_confidence:
                    column["pk_origin"] = pk_origin
            
            # NOT NULL constraint
            if include_not_null_constraint:
                column["not_null"] = bool(col["notnull"] == 1)
            
            # Default values
            if include_default_values:
                column["default"] = col["dflt_value"]
            
            # Extract column statistics
            try:
                # Handle reserved keyword column names
                quoted_col_name = quote_column_name(column_name)
                
                # Get distinct value count
                if include_distinct_count:
                    cursor.execute(f"SELECT COUNT(DISTINCT {quoted_col_name}) FROM {quoted_table};")
                    column["distinct_count"] = cursor.fetchone()[0]
                
                # Get null count
                if include_null_count:
                    cursor.execute(f"SELECT COUNT(*) FROM {quoted_table} WHERE {quoted_col_name} IS NULL;")
                    column["null_count"] = cursor.fetchone()[0]
                
                # Calculate derived statistics
                if include_row_count and table_info.get("row_count", 0) > 0:
                    if include_nullability and include_null_count:
                        column["nullability"] = round(column["null_count"] / table_info["row_count"], 4)
                    
                    if include_cardinality and include_distinct_count:
                        column["cardinality"] = round(column["distinct_count"] / table_info["row_count"], 4)
                
                # For numeric columns, get min, max, avg
                if any(num_type in col["type"].lower() for num_type in ["int", "real", "float", "double", "number", "decimal"]):
                    try:
                        if include_min_max or include_average:
                            cursor.execute(f"SELECT MIN({quoted_col_name}), MAX({quoted_col_name}), AVG({quoted_col_name}) FROM {quoted_table} WHERE {quoted_col_name} IS NOT NULL;")
                            min_val, max_val, avg_val = cursor.fetchone()
                            
                            if include_min_max:
                                column["min"] = min_val
                                column["max"] = max_val
                            
                            if include_average:
                                column["avg"] = round(avg_val, 4) if avg_val is not None else None
                        
                        # For small to medium tables, get median and stddev
                        if (include_median or include_stddev) and table_info.get("row_count", 0) <= max_rows_for_expensive_stats:
                            cursor.execute(f"SELECT {quoted_col_name} FROM {quoted_table} WHERE {quoted_col_name} IS NOT NULL;")
                            values = [row[0] for row in cursor.fetchall() if row[0] is not None]
                            
                            if values:
                                try:
                                    if include_median:
                                        column["median"] = round(statistics.median(values), 4)
                                    if include_stddev and len(values) > 1:
                                        column["stddev"] = round(statistics.stdev(values), 4)
                                except:
                                    pass  # Skip if not truly numeric
                    except:
                        pass  # Skip if not truly numeric
                
                # For text columns, get average length
                if any(text_type in col["type"].lower() for text_type in ["text", "char", "varchar", "string"]):
                    try:
                        if include_avg_length:
                            cursor.execute(f"SELECT AVG(LENGTH({quoted_col_name})) FROM {quoted_table} WHERE {quoted_col_name} IS NOT NULL;")
                            avg_length = cursor.fetchone()[0]
                            column["avg_length"] = round(avg_length, 2) if avg_length is not None else None
                        
                        # Get most common values for columns with reasonable cardinality
                        if (include_common_values and 
                            include_distinct_count and 
                            column.get("distinct_count", float('inf')) <= common_values_threshold):
                            cursor.execute(f"SELECT {quoted_col_name}, COUNT(*) as cnt FROM {quoted_table} WHERE {quoted_col_name} IS NOT NULL GROUP BY {quoted_col_name} ORDER BY cnt DESC LIMIT {common_values_limit};")
                            common_values = []
                            for row in cursor.fetchall():
                                value, count = row
                                if isinstance(value, str) and len(value) > max_string_display_length:
                                    value = value[:max_string_display_length-3] + "..."
                                common_values.append({"value": value, "count": count})
                            column["common_values"] = common_values
                    except:
                        pass
                
                # For date/time columns, get min and max dates
                if any(date_type in col["type"].lower() for date_type in ["date", "time", "timestamp", "datetime"]):
                    try:
                        if include_date_range:
                            cursor.execute(f"SELECT MIN({quoted_col_name}), MAX({quoted_col_name}) FROM {quoted_table} WHERE {quoted_col_name} IS NOT NULL;")
                            min_date, max_date = cursor.fetchone()
                            column["min_date"] = min_date
                            column["max_date"] = max_date
                        
                        # Try to calculate date range in days
                        if include_date_range_days:
                            try:
                                cursor.execute(f"SELECT julianday(MAX({quoted_col_name})) - julianday(MIN({quoted_col_name})) FROM {quoted_table} WHERE {quoted_col_name} IS NOT NULL;")
                                date_range = cursor.fetchone()[0]
                                if date_range is not None:
                                    column["date_range_days"] = round(date_range, 2)
                            except:
                                pass
                    except:
                        pass
                
            except sqlite3.Error as e:
                print(f"Error getting statistics for column {table}.{column_name}: {e}")
            
            # Add column to table
            table_info["columns"].append(column)
            
            # Prepare for sample data
            if include_samples:
                table_info["samples"][column_name] = []
        
        # Get foreign keys using enhanced detection with categorization
        if detect_foreign_keys:
            table_info["foreign_keys"] = []
            fk_list = key_analysis['foreign_keys'].get(table, [])
            
            for fk in fk_list:
                # Handle both regular and composite foreign keys
                if 'type' in fk and fk['type'] == 'composite':
                    # Check if any columns in this composite key should be skipped
                    skip_columns = [col for col in fk['from_columns'] if key_finder and key_finder.should_skip_column(table, col)]
                    skip_ref_columns = [col for col in fk['to_columns'] if key_finder and key_finder.should_skip_column(fk['to_table'], col)]
                    
                    # Skip this foreign key if it involves columns that should be excluded
                    if skip_columns or skip_ref_columns:
                        continue
                        
                    # Composite foreign key
                    foreign_key = {
                        "columns": fk['from_columns'],
                        "references": {
                            "table": fk['to_table'],
                            "columns": fk['to_columns']
                        },
                        "type": "composite"
                    }
                    
                    if include_key_confidence:
                        foreign_key["fk_origin"] = fk.get('origin', 'potential')
                        foreign_key["confidence"] = fk.get('confidence', 'low')
                    
                    # Update the column records to mark them as foreign keys
                    for col_name in fk['from_columns']:
                        for column in table_info["columns"]:
                            if column.get("name") == col_name:
                                column["is_foreign_key"] = True
                                if include_key_confidence:
                                    column["fk_origin"] = fk.get('origin', 'potential')
                                    column["references_table"] = fk['to_table']
                                    column["references_column"] = fk['to_columns'][fk['from_columns'].index(col_name)]
                                    column["composite_fk"] = True
                                fk_columns.add(col_name)
                else:
                    # Skip this foreign key if it involves columns that should be excluded
                    if key_finder and (key_finder.should_skip_column(table, fk['from']) or key_finder.should_skip_column(fk['to_table'], fk['to_column'])):
                        continue
                        
                    # Regular foreign key
                    foreign_key = {
                        "column": fk['from'],
                        "references": {
                            "table": fk['to_table'],
                            "column": fk['to_column']
                        }
                    }
                    
                    if include_key_confidence:
                        foreign_key["fk_origin"] = fk.get('origin', 'potential')
                        foreign_key["confidence"] = fk.get('confidence', 'medium')
                    
                    # Update the column record to mark it as a foreign key
                    for column in table_info["columns"]:
                        if column.get("name") == fk['from']:
                            column["is_foreign_key"] = True
                            if include_key_confidence:
                                column["fk_origin"] = fk.get('origin', 'potential')
                                column["references_table"] = fk['to_table']
                                column["references_column"] = fk['to_column']
                            fk_columns.add(fk['from'])
                
                # Add to table foreign keys
                table_info["foreign_keys"].append(foreign_key)
        
        # Get indexes if requested
        if include_indexes:
            table_info["indexes"] = []
            cursor.execute(f"PRAGMA index_list({quoted_table});")
            idx_list = cursor.fetchall()
            
            for idx in idx_list:
                # idx fields: (seq, name, unique, origin, partial)
                index_name = idx["name"]
                index_unique = (idx["unique"] == 1)
                
                # Look up the columns used by this index
                cursor.execute(f"PRAGMA index_info({quote_column_name(index_name)});")
                idx_cols_info = cursor.fetchall()
                col_names = [ic["name"] for ic in idx_cols_info]
                
                # Filter out columns that should be excluded
                if key_finder:
                    col_names = [col for col in col_names if not key_finder.should_skip_column(table, col)]
                
                # Skip empty indexes (after filtering)
                if not col_names:
                    continue
                    
                index = {
                    "name": index_name,
                    "unique": index_unique,
                    "columns": col_names
                }
                
                table_info["indexes"].append(index)
                
                # Update column to indicate indexing
                for col_name in col_names:
                    for column in table_info["columns"]:
                        if column.get("name") == col_name:
                            column["indexed"] = True
                            if "indexes" not in column:
                                column["indexes"] = []
                            column["indexes"].append(index_name)
        
        # Get sample data if requested
        if include_samples:
            try:
                # Debug info
                cursor.execute(f"SELECT COUNT(*) FROM {quoted_table};")
                row_count = cursor.fetchone()[0]
                print(f"Table {table} has {row_count} rows")
                
                # Get column names for SELECT, excluding those that should be skipped
                valid_column_names = []
                for col in columns_info:
                    col_name = col["name"]
                    if key_finder:
                        if not key_finder.should_skip_column(table, col_name):
                            valid_column_names.append(col_name)
                    else:
                        valid_column_names.append(col_name)
                
                print(f"Valid columns for {table}: {valid_column_names}")
                
                if valid_column_names and row_count > 0:
                    # Convert column names to quoted form for SQL
                    quoted_col_names = [quote_column_name(col) for col in valid_column_names]
                    select_cols = ', '.join(quoted_col_names)
                    
                    # Try different ordering strategies
                    sample_rows = []
                    try:
                        # First try ordering by rowid
                        select_sql = f"SELECT {select_cols} FROM {quoted_table} ORDER BY rowid ASC LIMIT ?"
                        cursor.execute(select_sql, (sample_limit,))
                        sample_rows = cursor.fetchall()
                        print(f"Retrieved {len(sample_rows)} rows using rowid ordering")
                    except sqlite3.OperationalError as e1:
                        print(f"Rowid ordering failed: {e1}")
                        try:
                            # Fallback: order by first column
                            first_col_quoted = quote_column_name(valid_column_names[0])
                            select_sql = f"SELECT {select_cols} FROM {quoted_table} ORDER BY {first_col_quoted} ASC LIMIT ?"
                            cursor.execute(select_sql, (sample_limit,))
                            sample_rows = cursor.fetchall()
                            print(f"Retrieved {len(sample_rows)} rows using first column ordering")
                        except sqlite3.OperationalError as e2:
                            print(f"First column ordering failed: {e2}")
                            try:
                                # Last resort: no ordering
                                select_sql = f"SELECT {select_cols} FROM {quoted_table} LIMIT ?"
                                cursor.execute(select_sql, (sample_limit,))
                                sample_rows = cursor.fetchall()
                                print(f"Retrieved {len(sample_rows)} rows with no ordering")
                            except sqlite3.OperationalError as e3:
                                print(f"All query methods failed: {e3}")
                                sample_rows = []
                    
                    # Process each sample row
                    if sample_rows:
                        print(f"Processing {len(sample_rows)} sample rows for table {table}")
                        if hasattr(sample_rows[0], 'keys'):
                            print(f"Sample row keys: {list(sample_rows[0].keys())}")
                        
                        for row_idx, row in enumerate(sample_rows):
                            # Convert sqlite3.Row to dict for easier access
                            if hasattr(row, 'keys'):
                                row_dict = dict(row)
                            else:
                                # Fallback: assume it's a tuple and map to column names
                                row_dict = dict(zip(valid_column_names, row))
                            
                            for col_name in valid_column_names:
                                # Skip columns not in our table_info (already filtered)
                                if col_name not in table_info["samples"]:
                                    continue
                                
                                # Get value - try different access methods
                                value = None
                                try:
                                    # Method 1: Direct access by column name
                                    value = row_dict.get(col_name)
                                    if value is None:
                                        # Method 2: Try case-insensitive lookup
                                        for key in row_dict.keys():
                                            if key.lower() == col_name.lower():
                                                value = row_dict[key]
                                                break
                                except (KeyError, IndexError, AttributeError) as e:
                                    print(f"Error accessing column {col_name} in row {row_idx}: {e}")
                                    print(f"Available keys: {list(row_dict.keys()) if isinstance(row_dict, dict) else 'Not a dict'}")
                                    continue
                                
                                # Handle special data types for JSON serialization
                                if isinstance(value, bytes):
                                    if len(value) <= max_binary_display_bytes:
                                        value = f"<binary data: {len(value)} bytes>"
                                    else:
                                        value = f"<binary data: {len(value)} bytes (truncated)>"
                                elif isinstance(value, str) and len(value) > max_string_display_length:
                                    value = value[:max_string_display_length-3] + "..."  # Truncate long strings
                                
                                table_info["samples"][col_name].append(value)
                    else:
                        print(f"No sample rows retrieved for table {table}")
                
                # Debug: Print final sample counts
                for col_name, samples in table_info["samples"].items():
                    print(f"Column {col_name}: {len(samples)} samples")
            
            except sqlite3.Error as e:
                print(f"SQLite error getting sample data for table {table}: {e}")
                print(f"Query attempted: {select_sql if 'select_sql' in locals() else 'No query built'}")
                # Try to get actual column names from database for debugging
                try:
                    cursor.execute(f"PRAGMA table_info({quoted_table});")
                    actual_columns = [col[1] for col in cursor.fetchall()]
                    print(f"Actual columns in {table}: {actual_columns}")
                    print(f"Valid columns we tried to use: {valid_column_names if 'valid_column_names' in locals() else 'None'}")
                except Exception as debug_e:
                    print(f"Could not get debug info: {debug_e}")
            except Exception as e:
                print(f"Unexpected error getting sample data for table {table}: {e}")
                import traceback
                traceback.print_exc()
        
        # Update schema summary
        if include_schema_summary:
            if detect_primary_keys and table in key_analysis['primary_keys']:
                schema_summary["tables_with_primary_keys"] += 1
            if detect_foreign_keys and table in key_analysis['foreign_keys']:
                schema_summary["tables_with_foreign_keys"] += 1
                schema_summary["total_relationships"] += len(key_analysis['foreign_keys'][table])
                    
        # Add table to db summary
        db_summary["tables"][table] = table_info
    
    # Add schema summary if requested
    if include_schema_summary:
        db_summary["schema_summary"] = schema_summary
    
    # Add table relationships summary if requested
    if include_table_relationships and detect_foreign_keys:
        relationships = []
        for table, fk_list in key_analysis['foreign_keys'].items():
            for fk in fk_list:
                if 'type' in fk and fk['type'] == 'composite':
                    relationships.append({
                        "from_table": table,
                        "from_columns": fk['from_columns'],
                        "to_table": fk['to_table'],
                        "to_columns": fk['to_columns'],
                        "type": "composite"
                    })
                else:
                    relationships.append({
                        "from_table": table,
                        "from_column": fk['from'],
                        "to_table": fk['to_table'],
                        "to_column": fk['to_column'],
                        "type": "simple"
                    })
        db_summary["relationships"] = relationships
    
    # Close database connection
    conn.close()
    if key_finder:
        key_finder.close()
    
    return db_summary


def save_db_summary(db_summary, output_path, indent=2):
    """
    Save database summary to JSON file
    """
    def json_serialize(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return str(obj)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(db_summary, f, default=json_serialize, indent=indent)
    
    print(f"Database summary saved to {output_path}")


def main():
    """
    Main function to run the schema analysis
    """
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Analyze a SQLite database and generate schema summary')
    parser.add_argument('--db-path', required=True, help='Path to the SQLite database file (.sqlite)')
    parser.add_argument('--output', help='Directory to save the JSON output',
                        default=str(Path(__file__).parent.parent.parent / 'data' / 'text2sql'))

    args = parser.parse_args()
    database_path = args.db_path

    if not os.path.exists(database_path):
        print(f"Error: Database file not found: {database_path}")
        return None
    
    print(f"Extracting schema summary from {database_path}...")
    
    try:
        db_summary = extract_db_summary_for_schema(database_path)
    except Exception as e:
        print(f"Error extracting database summary: {e}")
        return None
    
    # Save the summary
    db_basename = os.path.splitext(os.path.basename(database_path))[0]
    filename = f'{db_basename}_db_summary.json'
    output_path = os.path.join(args.output, filename)
    
    try:
        save_db_summary(db_summary, output_path)
    except Exception as e:
        print(f"Error saving database summary: {e}")
        return None
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    if 'metadata' in db_summary:
        metadata = db_summary['metadata']
        db_size = metadata.get('db_size_bytes', 'unknown')
        if isinstance(db_size, int):
            db_size_mb = round(db_size / (1024 * 1024), 2)
            print(f"  Database size: {db_size:,} bytes ({db_size_mb} MB)")
        else:
            print(f"  Database size: {db_size}")
        print(f"  Total tables: {metadata.get('table_count', 'unknown')}")
    
    if 'schema_summary' in db_summary:
        schema = db_summary['schema_summary']
        print(f"  Total columns: {schema.get('total_columns', 'unknown')}")
        print(f"  Tables with PKs: {schema.get('tables_with_primary_keys', 'unknown')}")
        print(f"  Tables with FKs: {schema.get('tables_with_foreign_keys', 'unknown')}")
        print(f"  Total relationships: {schema.get('total_relationships', 'unknown')}")
    
    total_tables = len(db_summary.get('tables', {}))
    print(f"  Processed tables: {total_tables}")
    
    # Calculate output file size
    try:
        output_size = os.path.getsize(output_path)
        output_size_kb = round(output_size / 1024, 2)
        print(f"  Output file size: {output_size:,} bytes ({output_size_kb} KB)")
    except:
        pass
    
    print(f"\nDatabase summary extraction completed and saved to {output_path}")
    return db_summary


if __name__ == "__main__":
    main()