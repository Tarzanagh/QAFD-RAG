"""
Prompt parser for QAFD cluster output.

Converts KG query clusters into formatted schema documentation
(CREATE TABLE statements with constraints and sample rows).

Matches the CoFD-M prompt.txt format for text2sql pipelines.
"""

import json
from typing import List, Dict, Any, Optional


def parse_qafd_clusters(
    clusters_data: List[Dict[str, Any]],
    add_sample_rows: bool = True,
    schema_data: Optional[Dict] = None,
    format_type: str = "create_table",
) -> str:
    """
    Parse QAFD clusters into formatted schema documentation.

    Args:
        clusters_data: List of cluster dicts from KG query (return_raw_clusters=True)
        add_sample_rows: Whether to include sample rows from schema
        schema_data: Original DB summary JSON for sample rows and column types
        format_type: "create_table" or "simple"

    Returns:
        Formatted schema string
    """
    if not clusters_data:
        return "No schema information available."

    table_info = extract_tables_from_clusters(clusters_data)

    if not table_info:
        return "No tables found in cluster data."

    if format_type == "create_table":
        return format_as_create_table(table_info, schema_data, add_sample_rows)
    else:
        return format_as_simple(table_info, schema_data, add_sample_rows)


def extract_tables_from_clusters(clusters_data: List[Dict[str, Any]]) -> Dict[str, Dict]:
    """Extract table/column info from cluster entities."""
    table_info = {}

    for cluster in clusters_data:
        entities = cluster.get("entities", cluster.get("nodes", []))

        for entity in entities:
            entity_name = entity.get("entity", "").strip('"')
            entity_type = entity.get("entity_type", entity.get("type", ""))
            description = entity.get("description", "")

            if not entity_name:
                continue

            if entity_type == "complete_table":
                if entity_name not in table_info:
                    table_info[entity_name] = {
                        "columns": {},
                        "description": description,
                        "rank": entity.get("rank", 0),
                    }
                elif description and not table_info[entity_name].get("description"):
                    table_info[entity_name]["description"] = description

            elif entity_type == "column" and "." in entity_name:
                parts = entity_name.split(".", 1)
                if len(parts) == 2:
                    table_name, column_name = parts

                    if table_name not in table_info:
                        table_info[table_name] = {
                            "columns": {},
                            "description": "",
                            "rank": 0,
                        }

                    table_info[table_name]["columns"][column_name] = {
                        "description": description,
                        "rank": entity.get("rank", 0),
                    }

    return table_info


def format_as_simple(
    table_info: Dict[str, Dict],
    schema_data: Optional[Dict] = None,
    add_sample_rows: bool = True,
) -> str:
    """Format as simple bullet-point list."""
    output = []

    for table_name, info in sorted(table_info.items()):
        output.append(f"\n{'=' * 60}")
        output.append(f"Table: {table_name}")

        if info.get("description"):
            output.append(f"Description: {info['description']}")

        if info.get("columns"):
            output.append("\nColumns:")
            for col_name, col_info in sorted(info["columns"].items()):
                desc = col_info.get("description", "")
                output.append(f"  - {col_name}: {desc}")

        if add_sample_rows and schema_data:
            samples = _get_sample_rows_json(table_name, schema_data)
            if samples:
                output.append(samples)

    return "\n".join(output)


def format_as_create_table(
    table_info: Dict[str, Dict],
    schema_data: Optional[Dict] = None,
    add_sample_rows: bool = True,
) -> str:
    """Format as CREATE TABLE statements (CoFD-M prompt.txt format)."""
    output = []

    sorted_tables = sorted(
        table_info.items(),
        key=lambda x: x[1].get("rank", 0),
        reverse=True,
    )

    for table_name, info in sorted_tables:
        create_stmt = _generate_create_table(table_name, info, schema_data)
        output.append(create_stmt)

        if add_sample_rows and schema_data:
            samples = _get_sample_rows_json(table_name, schema_data)
            if samples:
                output.append(samples)

        output.append("")

    return "\n".join(output)


def _generate_create_table(
    table_name: str,
    table_info: Dict,
    schema_data: Optional[Dict] = None,
) -> str:
    """
    Generate CREATE TABLE matching CoFD-M prompt.txt format:

    -- table_name
    CREATE TABLE `table_name` (
      `col` TYPE NOT NULL,
      `col2` TYPE PRIMARY KEY,
      ,FOREIGN KEY (`col`) REFERENCES `other_table`(`other_col`),
      ,CHECK (`col` BETWEEN min AND max)
    );
    """
    rag_columns = table_info.get("columns", {})
    table_desc = table_info.get("description", "")

    col_defs = []
    constraints = []

    if schema_data and "tables" in schema_data:
        schema_table = _find_table_in_schema(table_name, schema_data)

        if schema_table and "columns" in schema_table:
            schema_col_lookup = {
                col.get("name", ""): col for col in schema_table["columns"]
            }

            for col_name in rag_columns:
                schema_col = schema_col_lookup.get(col_name, {})
                col_type = schema_col.get("type", "TEXT")

                col_line = f"  `{col_name}` {col_type}"
                if schema_col.get("is_primary_key"):
                    col_line += " PRIMARY KEY"
                elif schema_col.get("not_null"):
                    col_line += " NOT NULL"

                col_defs.append(col_line)

                # FK constraint
                if schema_col.get("is_foreign_key") and schema_col.get("references_table"):
                    ref_table = schema_col["references_table"]
                    ref_col = schema_col.get("references_column", col_name)
                    constraints.append(
                        f"  ,FOREIGN KEY (`{col_name}`) REFERENCES `{ref_table}`(`{ref_col}`)"
                    )

                # CHECK constraint with min/max range
                col_min = schema_col.get("min")
                col_max = schema_col.get("max")
                if col_min is not None and col_max is not None and col_min != col_max:
                    if isinstance(col_min, str):
                        constraints.append(
                            f"  ,CHECK (`{col_name}` BETWEEN '{col_min}' AND '{col_max}')"
                        )
                    else:
                        constraints.append(
                            f"  ,CHECK (`{col_name}` BETWEEN {col_min} AND {col_max})"
                        )
    else:
        # No schema_data — use RAG info only
        for col_name in rag_columns:
            col_defs.append(f"  `{col_name}` TEXT")

    lines = [f"-- {table_name}"]
    lines.append(f"CREATE TABLE `{table_name}` (")

    if col_defs or constraints:
        all_lines = col_defs + constraints
        lines.append(",\n".join(all_lines))
    else:
        lines.append("  -- No columns found")

    lines.append(");")

    return "\n".join(lines)


def _find_table_in_schema(table_name: str, schema_data: Dict) -> Optional[Dict]:
    """Find table in schema, handling qualified names."""
    tables = schema_data.get("tables", {})

    if table_name in tables:
        return tables[table_name]

    short_name = table_name.split(".")[-1]
    for key, value in tables.items():
        if key.split(".")[-1] == short_name:
            return value

    return None


def _get_sample_rows_json(table_name: str, schema_data: Dict) -> str:
    """
    Get sample rows in JSON format (CoFD-M prompt.txt style).

    /* Sample rows:
    [
      {"col1": val1, "col2": val2},
      {"col1": val3, "col2": val4}
    ]
    */
    """
    table_data = _find_table_in_schema(table_name, schema_data)
    if not table_data:
        return ""

    sample_rows = None
    for key in ["sample_rows", "samples", "example_rows", "rows"]:
        if key in table_data:
            sample_rows = table_data[key]
            break

    if not sample_rows:
        return ""

    rows_as_dicts = []

    # Dict-of-lists format
    if isinstance(sample_rows, dict):
        col_names = list(sample_rows.keys())
        max_rows = max(
            (len(v) for v in sample_rows.values() if isinstance(v, list)),
            default=0,
        )
        for i in range(min(3, max_rows)):
            row = {}
            for col in col_names:
                col_data = sample_rows.get(col, [])
                if isinstance(col_data, list) and i < len(col_data):
                    val = col_data[i]
                    if isinstance(val, str) and len(val) > 80:
                        val = val[:77] + "..."
                    row[col] = val
                else:
                    row[col] = None
            rows_as_dicts.append(row)

    # List-of-dicts format
    elif isinstance(sample_rows, list) and sample_rows:
        for row in sample_rows[:3]:
            if isinstance(row, dict):
                cleaned = {}
                for k, v in row.items():
                    if isinstance(v, str) and len(v) > 80:
                        v = v[:77] + "..."
                    cleaned[k] = v
                rows_as_dicts.append(cleaned)

    if not rows_as_dicts:
        return ""

    json_str = json.dumps(rows_as_dicts, indent=2, ensure_ascii=False, default=str)
    return f"/* Sample rows:\n{json_str}\n*/"
