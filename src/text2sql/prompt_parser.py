"""
Prompt parser for QAFD cluster output.

Converts KG query clusters into formatted schema documentation
(CREATE TABLE statements with sample rows).

Ported from CoFD-M/methods/CoFD/prompt_parser.py
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
            samples = get_sample_rows(table_name, schema_data)
            if samples:
                output.append(samples)

    return "\n".join(output)


def format_as_create_table(
    table_info: Dict[str, Dict],
    schema_data: Optional[Dict] = None,
    add_sample_rows: bool = True,
) -> str:
    """Format as CREATE TABLE statements."""
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
            samples = get_sample_rows(table_name, schema_data)
            if samples:
                output.append(samples)

        output.append("")

    return "\n".join(output)


def _generate_create_table(
    table_name: str,
    table_info: Dict,
    schema_data: Optional[Dict] = None,
) -> str:
    """Generate a single CREATE TABLE statement."""
    columns = []
    table_desc = table_info.get("description", "")

    if schema_data and "tables" in schema_data:
        schema_table = _find_table_in_schema(table_name, schema_data)

        if schema_table and "columns" in schema_table:
            for col in schema_table["columns"]:
                col_name = col.get("name", "")
                col_type = col.get("type", "TEXT")

                cluster_desc = ""
                if col_name in table_info.get("columns", {}):
                    cluster_desc = table_info["columns"][col_name].get("description", "")

                columns.append(
                    {
                        "name": col_name,
                        "type": col_type,
                        "description": cluster_desc or col.get("description", ""),
                        "is_pk": col.get("is_primary_key", False),
                        "not_null": col.get("not_null", False),
                    }
                )

            if not table_desc and schema_table.get("description"):
                table_desc = schema_table["description"]

    if not columns:
        for col_name, col_info in table_info.get("columns", {}).items():
            columns.append(
                {
                    "name": col_name,
                    "type": "TEXT",
                    "description": col_info.get("description", ""),
                    "is_pk": False,
                    "not_null": False,
                }
            )

    lines = []

    if table_desc:
        lines.append(f"-- {table_name}: {table_desc}")

    lines.append(f"CREATE TABLE {table_name} (")

    if columns:
        col_lines = []
        for col in columns:
            col_line = f"    {col['name']} {col['type']}"

            if col.get("is_pk"):
                col_line += " PRIMARY KEY"
            elif col.get("not_null"):
                col_line += " NOT NULL"

            if col.get("description"):
                col_line += f"  -- {col['description']}"

            col_lines.append(col_line)

        lines.append(",\n".join(col_lines))
    else:
        lines.append("    -- No columns found")

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


def get_sample_rows(table_name: str, schema_data: Dict) -> str:
    """Get formatted sample rows from schema data."""
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

    lines = [f"/* Sample rows from {table_name}: */"]

    # Dict-of-lists format (common in db_summary.json)
    if isinstance(sample_rows, dict):
        col_names = list(sample_rows.keys())
        max_rows = max(
            (len(v) for v in sample_rows.values() if isinstance(v, list)),
            default=0,
        )
        if max_rows == 0:
            return ""

        lines.append(f"/* {' | '.join(col_names[:10])} */")

        for i in range(min(3, max_rows)):
            row_values = []
            for col in col_names[:10]:
                col_data = sample_rows.get(col, [])
                if isinstance(col_data, list) and i < len(col_data):
                    val = col_data[i]
                    if val is None:
                        row_values.append("NULL")
                    elif isinstance(val, str):
                        row_values.append(val[:50] if len(val) > 50 else val)
                    else:
                        row_values.append(str(val))
                else:
                    row_values.append("NULL")
            lines.append(f"/* {' | '.join(row_values)} */")

        return "\n".join(lines)

    # List-of-dicts or list-of-tuples format
    elif isinstance(sample_rows, list) and sample_rows:
        first_row = sample_rows[0]

        if isinstance(first_row, dict):
            col_names = list(first_row.keys())
            lines.append(f"/* {' | '.join(col_names[:10])} */")

            for row in sample_rows[:3]:
                values = []
                for col in col_names[:10]:
                    val = row.get(col, "NULL")
                    if val is None:
                        values.append("NULL")
                    elif isinstance(val, str):
                        values.append(val[:50] if len(val) > 50 else val)
                    else:
                        values.append(str(val))
                lines.append(f"/* {' | '.join(values)} */")

        elif isinstance(first_row, (list, tuple)):
            if "columns" in table_data:
                col_names = [c.get("name", f"col{i}") for i, c in enumerate(table_data["columns"])]
            else:
                col_names = [f"col{i}" for i in range(len(first_row))]

            lines.append(f"/* {' | '.join(col_names[:10])} */")

            for row in sample_rows[:3]:
                values = [str(v) if v is not None else "NULL" for v in row[:10]]
                lines.append(f"/* {' | '.join(values)} */")

        return "\n".join(lines)

    return ""
