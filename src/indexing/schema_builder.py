import json
import asyncio
import re
import time as _time
from typing import Dict, List, Any, Optional
from ..base import BaseGraphStorage, BaseVectorStorage
from ..utils import logger, compute_mdhash_id
from ..prompts import get_prompts

PROMPTS = get_prompts("text2sql")
import os


def detect_schema_type(schema_data: Dict[str, Any]) -> str:
    """
    Detect if schema has pre-built relationships or needs inference
    
    Args:
        schema_data: Parsed JSON schema data
        
    Returns:
        "db_summary" if has explicit relationships, "reconstruct" if needs inference
    """
    # Check for explicit relationship indicators
    if "relationships" in schema_data:
        return "db_summary"
    
    if "foreign_keys" in schema_data:
        return "db_summary"
    
    # Check if tables have relationship information
    tables = schema_data.get("tables", {})
    for table_data in tables.values():
        if "foreign_keys" in table_data:
            return "db_summary"
        if "relationships" in table_data:
            return "db_summary"
        # Check if columns are in DB summary format (list with detailed FK info)
        columns = table_data.get("columns", [])
        if columns and isinstance(columns, list):
            for col in columns[:3]:  # Check first few columns
                if isinstance(col, dict) and ("references_table" in col or "is_foreign_key" in col):
                    return "db_summary"
    
    # If no explicit relationships found, assume reconstruct data
    return "reconstruct"


class DatabaseSchemaBuilder:
    """
    Database Schema Builder for QAFD_RAG with LLM Enhancement
    
    This class handles the manual construction of knowledge graphs from database schema JSON files,
    with LLM-powered description enhancement and relationship weight optimization.
    """
    
    def __init__(self, 
                 graph_storage: BaseGraphStorage,
                 entities_vdb: BaseVectorStorage,
                 relationships_vdb: BaseVectorStorage,
                 llm_model_func: callable,
                 schema_file_path: Optional[str] = None):
        self.graph_storage = graph_storage
        self.entities_vdb = entities_vdb
        self.relationships_vdb = relationships_vdb
        self.llm_model_func = llm_model_func
        self.schema_file_path = schema_file_path
        self._tables_info_cache = {}  # Cache for tables_info
    
    async def build_from_json_schema(self, 
                                schema_file_path: str, 
                                metadata_file_path: Optional[str] = None,
                                language: str = "English") -> Dict[str, Any]:
        """
        Build knowledge graph from JSON schema file with LLM enhancement
        
        Args:
            schema_file_path: Path to the JSON schema file
            metadata_file_path: Optional path to metadata file
            language: Output language for descriptions
            
        Returns:
            Dictionary containing build statistics
        """
        logger.info(f"Building knowledge graph from schema: {schema_file_path}")
        
        # Update instance schema file path for validation methods
        self.schema_file_path = schema_file_path
        
        # Load JSON schema
        with open(schema_file_path, 'r', encoding='utf-8') as f:
            schema_data = json.load(f)
        
        # Detect schema type
        schema_type = detect_schema_type(schema_data)
        logger.info(f"Detected schema type: {schema_type}")
        
        # Load metadata if provided
        metadata_content = None
        if metadata_file_path:
            with open(metadata_file_path, 'r', encoding='utf-8') as f:
                metadata_content = f.read()
        
        # Step 1: Extract tables and columns based on schema type
        if schema_type == "db_summary":
            logger.info("Processing DB summary with existing relationships")
            tables_info = self._extract_tables_from_schema(schema_data)
        else:
            logger.info("Processing reconstruct data - will infer relationships from schema structure")
            tables_info = self._extract_tables_from_reconstruct_schema(schema_data)
        
        self._tables_info_cache = tables_info  # Cache for weight enhancement
        
        # Step 2: Insert tables and columns as entities WITH MINIMAL DESCRIPTIONS
        _t0 = _time.time()
        _build_start = _t0
        print(f"  [1/4] Inserting entities for {len(tables_info)} tables...", flush=True)
        entities_added = await self._insert_schema_entities_minimal(tables_info)
        print(f"  [1/4] Done ({entities_added} entities, {_time.time()-_t0:.1f}s)", flush=True)

        # Step 3: Create relationships between tables and columns
        _t0 = _time.time()
        print(f"  [2/4] Creating relationships...", flush=True)
        if schema_type == "db_summary":
            relationships_added = await self._create_schema_relationships(tables_info)
        else:
            relationships_added = await self._create_inferred_relationships(tables_info, schema_data)
        print(f"  [2/4] Done ({relationships_added} relationships, {_time.time()-_t0:.1f}s)", flush=True)

        # Step 3.5: Clean up any duplicate nodes that might have been created
        duplicates_removed = await self.graph_storage.remove_duplicate_nodes()
        if duplicates_removed > 0:
            logger.info(f"Cleaned up {duplicates_removed} duplicate nodes")

        # Step 3.6: Log graph statistics
        graph_stats = await self.graph_storage.get_graph_stats()
        logger.info(f"Graph statistics: {graph_stats['total_nodes']} nodes, {graph_stats['total_edges']} edges")
        logger.info(f"Node types: {graph_stats['node_types']}")

        # Step 4: LLM ENHANCEMENT (MOVED EARLIER AND IMPROVED)
        if self.llm_model_func:
            logger.info("Starting LLM enhancement phase...")

            # by Wenjun
            self.edge_descriptions = {}
            _t0 = _time.time()
            print(f"  [3/4] Enhancing descriptions with LLM...", flush=True)
            await self._enhance_descriptions_with_llm_chunked(tables_info, metadata_content, language)
            print(f"  [3/4] Done ({_time.time()-_t0:.1f}s)", flush=True)
            _t0 = _time.time()
            print(f"  [4/4] Enhancing relationship weights with LLM...", flush=True)
            await self._enhance_relationship_weights_with_llm_chunked(metadata_content, language)
            print(f"  [4/4] Done ({_time.time()-_t0:.1f}s, total: {_time.time()-_build_start:.1f}s)", flush=True)
        else:
            logger.warning("No LLM function provided - skipping description enhancement")
        return {
            "schema_type": schema_type,
            "tables_added": len(tables_info),
            "entities_added": entities_added,
            "relationships_added": relationships_added,
            "duplicates_removed": duplicates_removed,
            "graph_stats": graph_stats
        }

    def _extract_tables_from_reconstruct_schema(self, schema_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract table and column information from reconstruct-style JSON schema"""
        tables_info = {}
        
        # Extract metadata
        metadata = schema_data.get("metadata", {})
        
        # Extract tables - reconstruct format has tables as dict with column lists
        tables = schema_data.get("tables", {})
        
        for table_name, table_data in tables.items():
            table_info = {
                "name": table_name,
                "column_count": len(table_data.get("columns", [])),
                "row_count": len(table_data.get("sample_data", [])),
                "columns": []
            }
            
            # Extract columns - reconstruct format has columns as list of dicts
            columns = table_data.get("columns", [])
            for col in columns:
                column_info = {
                    "name": col.get("name", ""),
                    "type": col.get("type", ""),
                    "description": col.get("description", ""),
                    "is_primary_key": False,  # Will be inferred
                    "is_foreign_key": False,  # Will be inferred
                    "not_null": False,
                    "default": None,
                    "references_table": None,  # Will be inferred
                    "references_column": None  # Will be inferred
                }
                
                # Infer primary key from naming patterns
                if col.get("name", "").endswith("_id") and "id" in col.get("name", ""):
                    if col.get("name", "") == f"{table_name.split('.')[-1]}_id":
                        column_info["is_primary_key"] = True
                
                # Infer foreign keys from naming patterns
                if col.get("name", "").endswith("_id") and not column_info["is_primary_key"]:
                    potential_ref_table = col.get("name", "").replace("_id", "")
                    # Look for matching table
                    for other_table in tables.keys():
                        if potential_ref_table in other_table:
                            column_info["is_foreign_key"] = True
                            column_info["references_table"] = other_table
                            column_info["references_column"] = col.get("name", "")
                            break
                
                table_info["columns"].append(column_info)
            
            # Add sample data info
            if "sample_data" in table_data:
                table_info["sample_rows"] = table_data["sample_data"]
            
            # Add project structure info if available
            if "project" in table_data:
                table_info["project"] = table_data["project"]
            if "dataset" in table_data:
                table_info["dataset"] = table_data["dataset"]
            
            tables_info[table_name] = table_info
        
        return tables_info

    async def _create_inferred_relationships(self, tables_info: Dict[str, Dict[str, Any]], 
                                        schema_data: Dict[str, Any]) -> int:
        """Create inferred relationships for reconstruct data based on schema analysis"""
        relationships_added = 0
        
        # Create table-column relationships (same as existing)
        for table_name, table_data in tables_info.items():
            table_id = f'"{table_name}"'
            
            for column in table_data.get("columns", []):
                column_name = column.get("name", "")
                if not column_name:
                    continue
                    
                col_id = f'"{table_name}.{column_name}"'
                
                # Table contains column relationship
                edge_data = {
                    "weight": 8.0,
                    "description": f"Table '{table_name}' contains column '{column_name}'",
                    "keywords": "table_structure, contains_column",
                    "source_id": "reconstruct_inference"
                }
                
                await self.graph_storage.upsert_edge(table_id, col_id, edge_data=edge_data)
                relationships_added += 1
        
        # Create inferred table-to-table relationships
        table_names = list(tables_info.keys())
        for i, table1 in enumerate(table_names):
            for j, table2 in enumerate(table_names):
                if i >= j:  # Avoid duplicates and self-references
                    continue
                
                table1_data = tables_info[table1]
                table2_data = tables_info[table2]
                
                relationship_weight = 0.0
                relationship_type = "related_to"
                
                # Check for same project/dataset (for BigQuery style)
                if (table1_data.get("project") == table2_data.get("project") and 
                    table1_data.get("dataset") == table2_data.get("dataset")):
                    relationship_weight = 6.0
                    relationship_type = "same_dataset"
                
                # Check for naming similarity
                table1_base = table1.split('.')[-1]
                table2_base = table2.split('.')[-1]
                if any(word in table1_base for word in table2_base.split('_')) or \
                any(word in table2_base for word in table1_base.split('_')):
                    if relationship_weight < 5.0:
                        relationship_weight = 5.0
                        relationship_type = "semantically_related"
                
                # Add relationship if weight is significant
                if relationship_weight > 4.0:
                    table1_id = f'"{table1}"'
                    table2_id = f'"{table2}"'
                    
                    edge_data = {
                        "weight": relationship_weight,
                        "description": f"Inferred {relationship_type} relationship between '{table1}' and '{table2}'",
                        "keywords": f"inferred_relationship, {relationship_type}",
                        "source_id": "reconstruct_inference"
                    }
                    
                    await self.graph_storage.upsert_edge(table1_id, table2_id, edge_data=edge_data)
                    relationships_added += 1
        
        logger.info(f"Created {relationships_added} inferred relationships for reconstruct data")
        return relationships_added
    
    def _extract_tables_from_schema(self, schema_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract table and column information from JSON schema"""
        tables_info = {}
        
        # Extract tables
        tables = schema_data.get("tables", {})
        
        for table_name, table_data in tables.items():
            table_info = {
                "name": table_name,
                "column_count": table_data.get("column_count", 0),
                "row_count": table_data.get("row_count", 0),
                "columns": []
            }
            
            # Extract columns
            columns = table_data.get("columns", [])
            for col in columns:
                column_info = {
                    "name": col.get("name", ""),
                    "type": col.get("type", ""),
                    "is_primary_key": col.get("is_primary_key", False),
                    "is_foreign_key": col.get("is_foreign_key", False),
                    "not_null": col.get("not_null", False),
                    "default": col.get("default", None),
                    "references_table": col.get("references_table", None),
                    "references_column": col.get("references_column", None)
                }
                table_info["columns"].append(column_info)
            
            # ✅ NEW: Read table-level foreign_keys array and merge into columns
            table_level_fks = table_data.get("foreign_keys", [])
            if table_level_fks:
                logger.info(f"Processing {len(table_level_fks)} table-level FKs for {table_name}")
                
                for fk in table_level_fks:
                    col_name = fk.get("column")
                    ref_info = fk.get("references", {})
                    ref_table = ref_info.get("table")
                    ref_column = ref_info.get("column")
                    
                    if not col_name or not ref_table or not ref_column:
                        continue
                    
                    # Find the matching column and update its FK info
                    for col_info in table_info["columns"]:
                        if col_info["name"] == col_name:
                            col_info["is_foreign_key"] = True
                            # Only set if not already populated from column-level data
                            if not col_info["references_table"]:
                                col_info["references_table"] = ref_table
                                logger.info(f"Set FK: {table_name}.{col_name} -> {ref_table}.{ref_column}")
                            if not col_info["references_column"]:
                                col_info["references_column"] = ref_column
                            break
            
            tables_info[table_name] = table_info
        
        return tables_info
    
    async def _insert_schema_entities_minimal(self, tables_info: Dict[str, Dict[str, Any]]) -> int:
        """
        Insert tables and columns as entities with MINIMAL descriptions for LLM enhancement
        
        Args:
            tables_info: Dictionary of table information
            
        Returns:
            Number of entities added
        """
        entities_added = 0
        entities_for_vdb = {}
        
        for table_name, table_data in tables_info.items():
            # Add table entity with MINIMAL description
            table_id = f'"{table_name}"'
            table_node_data = {
                "entity_type": "complete_table",
                "description": f"Table: {table_name}",  # MINIMAL - will be enhanced by LLM
                "source_id": "schema_extraction",
                "table_name": table_name,
                "column_count": table_data["column_count"],
                "row_count": table_data["row_count"]
            }
            
            await self.graph_storage.upsert_node(table_id, node_data=table_node_data)
            
            # Prepare for vector database   ????????????????????????????????
            table_vdb_id = compute_mdhash_id(table_id, prefix="ent-")
            entities_for_vdb[table_vdb_id] = {
                "content": table_id + " " + table_node_data["description"],
                "entity_name": table_id
            }
            
            entities_added += 1
            
            # Add column entities with MINIMAL descriptions
            for col in table_data["columns"]:
                col_name = col["name"]
                if not col_name:
                    continue
                    
                col_id = f'"{table_name}.{col_name}"'
                col_node_data = {
                    "entity_type": "column",
                    "description": f"Column: {table_name}.{col_name}",  # MINIMAL - will be enhanced by LLM
                    "source_id": "schema_extraction",
                    "table_name": table_name,
                    "column_name": col_name,
                    "data_type": col["type"],
                    "is_primary_key": col["is_primary_key"],
                    "is_foreign_key": col["is_foreign_key"],
                    "not_null": col["not_null"]
                }
                
                # Add optional fields only if they are not None
                if col["default"] is not None:
                    col_node_data["default"] = col["default"]
                if col["references_table"] is not None:
                    col_node_data["references_table"] = col["references_table"]
                if col["references_column"] is not None:
                    col_node_data["references_column"] = col["references_column"]
                
                await self.graph_storage.upsert_node(col_id, node_data=col_node_data)
                
                # Prepare for vector database
                col_vdb_id = compute_mdhash_id(col_id, prefix="ent-")
                entities_for_vdb[col_vdb_id] = {
                    "content": col_id + " " + col_node_data["description"],
                    "entity_name": col_id
                }
                
                entities_added += 1
        
        # Insert entities into vector database
        if entities_for_vdb and self.entities_vdb:
            await self.entities_vdb.upsert(entities_for_vdb)
            logger.info(f"Inserted {len(entities_for_vdb)} entities into vector database")
        
        return entities_added
    
    async def _create_schema_relationships(self, tables_info: Dict[str, Dict[str, Any]]) -> int:
        """Create relationships between tables and columns with fully qualified names"""
        relationships_added = 0
        relationships_for_vdb = {}

        # Add debug logging
        logger.info(f"Creating relationships for {len(tables_info)} tables")
        available_tables = set(tables_info.keys())
        logger.info(f"Available tables: {sorted(available_tables)}")

        for table_name, table_data in tables_info.items():
            table_id = f'"{table_name}"'
            logger.info(f"Processing table: {table_name}")

            for col in table_data["columns"]:
                col_name = col["name"]
                if not col_name:
                    continue

                col_id = f'"{table_name}.{col_name}"'

                # Create table-to-column relationship
                edge_data = {
                    "weight": 10.0,
                    "description": f"Table '{table_name}' contains column '{col_name}'",
                    "keywords": "table_structure, contains_column",
                    "source_id": "schema_extraction"
                }

                await self.graph_storage.upsert_edge(table_id, col_id, edge_data=edge_data)

                # Prepare for vector database   ???????????????????????????????
                edge_vdb_id = compute_mdhash_id(f"{table_id}->{col_id}", prefix="rel-")
                relationships_for_vdb[edge_vdb_id] = {
                    "src_id": table_id,
                    "tgt_id": col_id,
                    "content": edge_data["keywords"] + " " + table_id + " " + col_id + " " + edge_data["description"]
                }

                relationships_added += 1

                # Create foreign key relationships if applicable
                if col["is_foreign_key"] and col["references_table"] and col["references_column"]:
                    ref_table_full = col["references_table"]
                    ref_column = col["references_column"]

                    logger.info(f"Processing FK: {col_name} -> {ref_table_full}.{ref_column}")

                    # ✅ Use fully qualified names consistently
                    if ref_table_full in available_tables:
                        ref_table_id = f'"{ref_table_full}"'
                        ref_col_id = f'"{ref_table_full}.{ref_column}"'

                        logger.info(f"Creating FK edge: {col_id} -> {ref_col_id}")

                        fk_edge_data = {
                            "weight": 15.0,
                            "description": f"Foreign key: '{col_name}' in '{table_name}' "
                                           f"references '{ref_column}' in '{ref_table_full}'",
                            "keywords": "foreign_key, references, data_integrity",
                            "source_id": "schema_extraction"
                        }

                        await self.graph_storage.upsert_edge(col_id, ref_col_id, edge_data=fk_edge_data)

                        # Prepare for vector database
                        fk_edge_vdb_id = compute_mdhash_id(f"{col_id}->{ref_col_id}", prefix="rel-")
                        relationships_for_vdb[fk_edge_vdb_id] = {
                            "src_id": col_id,
                            "tgt_id": ref_col_id,
                            "content": fk_edge_data["keywords"] + " " + col_id + " " +
                                       ref_col_id + " " + fk_edge_data["description"]
                        }

                        relationships_added += 1
                        logger.info(f"Successfully created FK relationship: {col_id} -> {ref_col_id}")
                    else:
                        logger.warning(f"Referenced table '{ref_table_full}' not found in schema for FK {col_name}")
                        logger.warning(f"Available tables: {sorted(available_tables)}")

                elif col["is_foreign_key"]:
                    logger.warning(
                        f"FK column {col_name} missing reference info: "
                        f"references_table={col.get('references_table')}, "
                        f"references_column={col.get('references_column')}"
                    )

        # ✅ Insert into vector DB once, after all loops
        if relationships_for_vdb and self.relationships_vdb:
            await self.relationships_vdb.upsert(relationships_for_vdb)
            logger.info(f"Inserted {len(relationships_for_vdb)} relationships into vector database")

        logger.info(f"Total relationships created: {relationships_added}")
        return relationships_added

    
    async def _enhance_descriptions_with_llm_chunked(self, 
                                        tables_info: Dict[str, Dict[str, Any]], 
                                        metadata_content: Optional[str],
                                        language: str) -> None:
        """
        Use LLM to enhance entity descriptions with chunking to avoid token limits
        """
        logger.info("Enhancing descriptions with LLM (chunked)...")
        
        from ..utils import encode_string_by_tiktoken
        
        # Configuration that actually works
        max_tokens_per_chunk = 8000   # Reasonable chunk size
        base_prompt_tokens = 1000     # Reduce prompt overhead
        response_buffer_tokens = 1000 # Reduce response buffer  
        available_tokens = max_tokens_per_chunk - base_prompt_tokens - response_buffer_tokens
         
        # Split tables into chunks
        table_chunks = self._chunk_tables_by_tokens(tables_info, available_tokens)
        
        logger.info(f"Processing {len(tables_info)} tables in {len(table_chunks)} chunks")
        _chunk_start = _time.time()

        # Process each chunk
        for i, chunk_tables in enumerate(table_chunks, 1):
            logger.info(f"Processing chunk {i}/{len(table_chunks)} with {len(chunk_tables)} tables")
            elapsed = _time.time() - _chunk_start
            if i > 1:
                eta = elapsed / (i - 1) * (len(table_chunks) - i + 1)
                print(f"\r  [3/4] Enhancing descriptions: chunk {i}/{len(table_chunks)} (ETA: {eta:.0f}s)", end='', flush=True)
            else:
                print(f"\r  [3/4] Enhancing descriptions: chunk {i}/{len(table_chunks)}", end='', flush=True)

            try:
                # Create prompt for this chunk
                chunk_schema_text = self._format_schema_for_llm({"tables": chunk_tables})
                prompt = self._create_description_enhancement_prompt(chunk_schema_text, metadata_content, language)
                
                # Verify token count
                prompt_tokens = len(encode_string_by_tiktoken(prompt))
                
                if prompt_tokens > max_tokens_per_chunk:
                    logger.warning(f"Chunk {i} still exceeds token limit ({prompt_tokens} tokens), skipping")
                    continue
                
                logger.info(f"Chunk {i}: Sending {prompt_tokens} tokens to LLM...")
                
                # Get enhanced descriptions from LLM
                enhanced_result = await self.llm_model_func(prompt)
                
                logger.info(f"Chunk {i}: Received LLM response, applying enhancements...")
                
                # Parse and apply enhanced descriptions
                await self._apply_enhanced_descriptions(enhanced_result, chunk_tables)
                
            except Exception as e:
                logger.error(f"Error enhancing descriptions for chunk {i}: {e}")
                continue
        print(flush=True)

    async def _enhance_relationship_weights_with_llm_chunked(self, 
                                                metadata_content: Optional[str],
                                                language: str = "English") -> None:
        """
        Use LLM to enhance relationship weights with chunking to avoid token limits
        """
        logger.info("Enhancing relationship weights with LLM (chunked)...")
        
        from ..utils import encode_string_by_tiktoken
      
        # Configuration
        max_tokens_per_chunk = 100000
        base_prompt_tokens = 3000
        response_buffer_tokens = 2000
        available_tokens = max_tokens_per_chunk - base_prompt_tokens - response_buffer_tokens
        
        # Get all relationships
        relationships_list = await self._get_all_relationships()
        if not relationships_list.strip():
            logger.warning("No relationships found for weight enhancement")
            return
        
        # Split relationships into chunks
        relationship_chunks = self._chunk_relationships_by_tokens(relationships_list, available_tokens)
        
        logger.info(f"Processing relationships in {len(relationship_chunks)} chunks")
        _chunk_start = _time.time()

        # Process each chunk
        for i, chunk_relationships in enumerate(relationship_chunks, 1):
            logger.info(f"Processing relationship chunk {i}/{len(relationship_chunks)}")
            elapsed = _time.time() - _chunk_start
            if i > 1:
                eta = elapsed / (i - 1) * (len(relationship_chunks) - i + 1)
                print(f"\r  [4/4] Enhancing weights: chunk {i}/{len(relationship_chunks)} (ETA: {eta:.0f}s)", end='', flush=True)
            else:
                print(f"\r  [4/4] Enhancing weights: chunk {i}/{len(relationship_chunks)}", end='', flush=True)
            
            try:
                # Create minimal schema for this chunk
                involved_tables = self._extract_tables_from_relationships(chunk_relationships)
                minimal_schema = self._create_minimal_schema_for_tables(involved_tables)
                
                # Create prompt for this chunk
                prompt = self._create_weight_enhancement_prompt(minimal_schema, metadata_content, chunk_relationships, language)
                
                # Verify token count
                prompt_tokens = len(encode_string_by_tiktoken(prompt))
                if prompt_tokens > max_tokens_per_chunk:
                    logger.warning(f"Relationship chunk {i} exceeds token limit ({prompt_tokens} tokens), skipping")
                    continue
                
                logger.info(f"Relationship chunk {i}: Sending {prompt_tokens} tokens to LLM...")
                
                # Get enhanced weights from LLM
                enhanced_result = await self.llm_model_func(prompt)
                
                logger.info(f"Relationship chunk {i}: Received LLM response, applying weight enhancements...")
                
                # Parse and apply enhanced weights
                await self._apply_enhanced_weights(enhanced_result)
                
            except Exception as e:
                logger.error(f"Error enhancing weights for chunk {i}: {e}")
                continue
    
    async def _get_all_relationships(self) -> str:
        """Get all relationships from the graph for LLM processing"""
        relationships = []
        
        try:
            # Get all edges from the actual graph
            all_edges = await self.graph_storage.edges()
            
            if all_edges:
                # Convert EdgeView to list for easier handling
                edges_list = list(all_edges)
                logger.info(f"Found {len(edges_list)} edges in graph")
                
                for source, target in edges_list:
                    # Get edge data to get current weight and description
                    edge_data = await self.graph_storage.get_edge(source, target)
                    if edge_data:
                        current_weight = edge_data.get('weight', 1.0)
                        description = edge_data.get('description', 'unknown')
                        relationship_info = f"- {source} -> {target} (current_weight: {current_weight}, description: {description})"
                        relationships.append(relationship_info)
                    else:
                        logger.warning(f"Could not get edge data for {source} -> {target}")
            else:
                # Fallback to schema-based relationships if graph is empty
                logger.warning("Graph is empty, falling back to schema-based relationships")
                for table_name, table_data in self._tables_info_cache.items():
                    table_id = f'"{table_name}"'
                    
                    for col in table_data.get("columns", []):
                        col_name = col.get("name", "")
                        if col_name:
                            col_id = f'"{table_name}.{col_name}"'
                            
                            # Table-to-column relationship
                            relationship_info = f"- {table_id} -> {col_id} (current_weight: 10.0, description: table_structure)"
                            relationships.append(relationship_info)
                            
                            # Foreign key relationships if applicable
                            if col.get("is_foreign_key") and col.get("references_table") and col.get("references_column"):
                                ref_table_id = f'"{col["references_table"]}"'
                                ref_col_id = f'"{col["references_table"]}.{col["references_column"]}"'
                                
                                fk_relationship_info = f"- {col_id} -> {ref_col_id} (current_weight: 9.0, description: foreign_key)"
                                relationships.append(fk_relationship_info)
        
        except Exception as e:
            logger.error(f"Error getting relationships from graph: {e}")
            # Fallback to empty list
            relationships = []
        
        return "\n".join(relationships)
    
    def _create_weight_enhancement_prompt(self, 
                                        schema_text: str, 
                                        metadata_content: Optional[str],
                                        relationships_list: str,
                                        language: str) -> str:
        """Create prompt for LLM weight enhancement"""
        # Load the complete JSON file to include sample data
        try:
            # Use the instance schema file path if available
            if self.schema_file_path and os.path.exists(self.schema_file_path):
                with open(self.schema_file_path, 'r', encoding='utf-8') as f:
                    complete_schema_data = json.load(f)
                complete_schema_text = json.dumps(complete_schema_data, indent=2, ensure_ascii=False)
            else:
                complete_schema_text = schema_text
        except Exception as e:
            logger.warning(f"Could not load complete schema file: {e}")
            complete_schema_text = schema_text
        
        # Use the enhanced weight assignment prompt template
        prompt_template = PROMPTS["enhanced_graph_weight_assignment"]
        
        # Format the prompt with the complete schema data
        prompt = prompt_template.format(
            language=language,
            schema_text=complete_schema_text,
            metadata_content=metadata_content or "No additional metadata provided.",
            edge_list=relationships_list
        )
        
        return prompt
    
    async def _apply_enhanced_weights(self, enhanced_result: str) -> None:
        """Apply enhanced weights from LLM to graph relationships with improved debugging"""
        try:
            logger.info(f"Raw LLM weight response (first 500 chars): {enhanced_result[:500]}...")
            
            # Extract JSON from the response with improved parsing
            import re
            
            # Try multiple patterns to find JSON
            json_patterns = [
                r'\{.*\}',  # Basic JSON object
                r'```json\s*(\{.*?\})\s*```',  # JSON in code blocks
                r'```\s*(\{.*?\})\s*```',  # JSON in generic code blocks
                r'```\s*(\{.*\})\s*```',  # JSON in code blocks (non-greedy)
            ]

            enhanced_data = None
            for pattern in json_patterns:
                json_match = re.search(pattern, enhanced_result, re.DOTALL)

                if json_match:
                    try:
                        json_str = json_match.group(1) if len(json_match.groups()) > 0 else json_match.group(0)
                        enhanced_data = json.loads(json_str)
                        logger.info(f"Successfully parsed JSON with keys: {list(enhanced_data.keys())}")
                        break
                    except json.JSONDecodeError:
                        continue

            if not enhanced_data:
                logger.warning("No valid JSON found in enhanced result, skipping weight enhancement")
                return

            # Apply relationship weights (CoFD-style multiplication)
            relationship_weights = enhanced_data.get("relationship_weights", {})
            weighting_rationale = enhanced_data.get("weighting_rationale", {})
            
            logger.info(f"Found {len(relationship_weights)} relationship weights to apply")
            
            weights_updated = 0
            for relationship_key, llm_score in relationship_weights.items():
                if '->' in relationship_key:
                    source, target = relationship_key.split('->', 1)
                    
                    # Clean up the source and target IDs (remove quotes if present)
                    source = source.strip().strip('"')
                    target = target.strip().strip('"')
                    
                    # Add quotes to match the graph format
                    source_with_quotes = f'"{source}"'
                    target_with_quotes = f'"{target}"'
                    
                    # Try both formats: with and without quotes
                    edge_data = await self.graph_storage.get_edge(source_with_quotes, target_with_quotes)
                    actual_source = source_with_quotes
                    actual_target = target_with_quotes
                    
                    if not edge_data:
                        # Try without quotes as fallback
                        edge_data = await self.graph_storage.get_edge(source, target)
                        actual_source = source
                        actual_target = target
                    
                    # ✅ FIXED: Only proceed if edge exists
                    if edge_data:
                        # Apply description from self.edge_descriptions if available
                        # (This is redundant since Phase 3 already applied it, but kept for backward compatibility)
                        if relationship_key in self.edge_descriptions:
                            if 'original_description' not in edge_data:
                                edge_data['original_description'] = edge_data.get('description', '')
                            edge_data['description'] = self.edge_descriptions[relationship_key]
                        
                        # Get original weight (CoFD-style)
                        original_weight = edge_data.get('weight', 1.0)
                        
                        # Multiply original weight with LLM score (CoFD approach)
                        enhanced_weight = original_weight * llm_score
                        
                        # Update edge with new weight
                        edge_data['weight'] = enhanced_weight
                        edge_data['llm_enhanced'] = True
                        edge_data['llm_score'] = llm_score
                        edge_data['original_weight'] = original_weight
                        
                        # Add rationale if available
                        if relationship_key in weighting_rationale:
                            edge_data['weighting_rationale'] = weighting_rationale[relationship_key]
                        
                        await self.graph_storage.upsert_edge(actual_source, actual_target, edge_data=edge_data)
                        weights_updated += 1
                        logger.info(f"Updated weight for {relationship_key}: {original_weight} * {llm_score} = {enhanced_weight}")
                    else:
                        logger.warning(f"Edge not found in graph: {source} -> {target} (tried both quoted and unquoted formats)")
            
            logger.info(f"Successfully updated {weights_updated} relationship weights with LLM enhancement")
        
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from enhanced result: {e}")
        except Exception as e:
            logger.error(f"Error applying enhanced weights: {e}")
    
    def _format_schema_for_llm(self, tables_info: Dict[str, Dict[str, Any]]) -> str:
        """Format schema information for LLM processing"""
        schema_text = "JSON Schema:\n"
        schema_text += json.dumps({"tables": tables_info}, indent=2, ensure_ascii=False)
        return schema_text
    
    def _create_description_enhancement_prompt(self, 
                                            schema_text: str, 
                                            metadata_content: Optional[str],
                                            language: str) -> str:
        """Create prompt for LLM description enhancement using only chunked data"""
        
        # Use the provided chunked schema_text instead of loading complete file
        prompt_template = PROMPTS["enhanced_graph_description"]
        
        prompt = prompt_template.format(
            language=language,
            schema_text=schema_text,  # Use chunked data passed in
            metadata_content=metadata_content or "No additional metadata provided."
        )
        
        return prompt
    
    async def _apply_enhanced_descriptions(self, 
                                        enhanced_result: str, 
                                        tables_info: Dict[str, Dict[str, Any]]) -> None:
        """Apply enhanced descriptions from LLM to graph entities and edges"""
        try:
            logger.info(f"Raw LLM description response (first 500 chars): {enhanced_result[:500]}...")
            
            # Also check what node IDs actually exist:
            all_nodes = await self.graph_storage.nodes()
            existing_node_ids = list(all_nodes) if all_nodes else []
            logger.info(f"Total nodes in graph: {len(existing_node_ids)}")
            
            # Extract JSON from the response with improved parsing
            import re
            
            # Try multiple patterns to find JSON
            json_patterns = [
                r'\{.*\}',  # Basic JSON object
                r'```json\s*(\{.*?\})\s*```',  # JSON in code blocks
                r'```\s*(\{.*?\})\s*```',  # JSON in generic code blocks
                r'```\s*(\{.*\})\s*```',  # JSON in code blocks (non-greedy)
            ]

            enhanced_data = None
            for pattern in json_patterns:
                json_match = re.search(pattern, enhanced_result, re.DOTALL)

                if json_match:
                    try:
                        json_str = json_match.group(1) if len(json_match.groups()) > 0 else json_match.group(0)
                        enhanced_data = json.loads(json_str)
                        logger.info(f"Successfully parsed JSON with keys: {list(enhanced_data.keys())}")
                        break
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON parse failed for pattern {pattern}: {e}")
                        continue
            
            if not enhanced_data:
                logger.warning("No valid JSON found in enhanced result, skipping description enhancement")
                return
            
            # ====================================================================
            # PHASE 1: ENHANCE NODES (Tables and Columns)
            # ====================================================================
            
            # Apply table descriptions
            table_descriptions = enhanced_data.get("table_descriptions", {})
            logger.info(f"Found {len(table_descriptions)} table descriptions to apply")
            
            for table_name, description in table_descriptions.items():
                # Try different table name formats
                table_ids_to_try = [
                    f'"{table_name}"',
                    f'"{table_name.lower()}"',
                    f'"{table_name.upper()}"'
                ]
                
                description_applied = False
                for table_id in table_ids_to_try:
                    if await self.graph_storage.has_node(table_id):
                        # Get existing node data and update description
                        existing_data = await self.graph_storage.get_node(table_id)
                        if existing_data:
                            existing_data["description"] = description
                            existing_data["llm_enhanced"] = True
                            await self.graph_storage.upsert_node(table_id, existing_data)
                            logger.info(f"Updated table description for {table_name} (ID: {table_id})")
                            description_applied = True
                            break
                
                if not description_applied:
                    logger.warning(f"Could not find table node for: {table_name}")
                    
            # Apply column descriptions
            column_descriptions = enhanced_data.get("column_descriptions", {})
            logger.info(f"Found {len(column_descriptions)} column descriptions to apply")
            
            for column_name, description in column_descriptions.items():
                # Try different column name formats
                column_ids_to_try = [
                    f'"{column_name}"',
                    f'"{column_name.lower()}"',
                    f'"{column_name.upper()}"'
                ]
                
                description_applied = False
                for column_id in column_ids_to_try:
                    if await self.graph_storage.has_node(column_id):
                        # Get existing node data and update description
                        existing_data = await self.graph_storage.get_node(column_id)
                        if existing_data:
                            existing_data["description"] = description
                            existing_data["llm_enhanced"] = True
                            await self.graph_storage.upsert_node(column_id, existing_data)
                            logger.info(f"Updated column description for {column_name} (ID: {column_id})")
                            description_applied = True
                            break
                
                if not description_applied:
                    logger.warning(f"Could not find column node for: {column_name}")
            
            # ====================================================================
            # PHASE 2: UPDATE VECTOR DATABASE WITH ENHANCED NODE DESCRIPTIONS
            # ====================================================================
            
            entities_added = 0
            entities_for_vdb = {}
            
            for table_name, description in table_descriptions.items():
                table_id = f'"{table_name}"'
                # Prepare for vector database
                table_vdb_id = compute_mdhash_id(table_id, prefix="ent-")
                entities_for_vdb[table_vdb_id] = {
                    "content": table_id + " " + description,
                    "entity_name": table_id
                }
                entities_added += 1
                
            for col_name, description in column_descriptions.items():
                col_id = f'"{col_name}"'
                # Prepare for vector database
                col_vdb_id = compute_mdhash_id(col_id, prefix="ent-")
                entities_for_vdb[col_vdb_id] = {
                    "content": col_id + " " + description,
                    "entity_name": col_id
                }
                entities_added += 1
                    
            if entities_for_vdb and self.entities_vdb:
                await self.entities_vdb.upsert(entities_for_vdb)
                logger.info(f"Inserted llm enhanced {len(entities_for_vdb)} entities into vector database")
            
            # ====================================================================
            # PHASE 3: ENHANCE GRAPH EDGES (Table-Column & Column-Column)
            # ====================================================================
            
            # Get relationship descriptions
            table_column_descriptions = enhanced_data.get("table_column_relationship_descriptions", {})
            column_column_descriptions = enhanced_data.get("column_relationship_descriptions", {})
            
            logger.info(f"Found {len(table_column_descriptions)} table-column relationship descriptions")
            logger.info(f"Found {len(column_column_descriptions)} column-column relationship descriptions")
            
            # Apply table-column relationship descriptions to GRAPH EDGES
            edges_enhanced = 0
            for rel_key, description in table_column_descriptions.items():
                if '->' in rel_key:
                    source, target = rel_key.split('->', 1)
                    source = source.strip().strip('"')
                    target = target.strip().strip('"')
                    
                    source_quoted = f'"{source}"'
                    target_quoted = f'"{target}"'
                    
                    # Try to find and update the edge in the graph
                    edge_data = await self.graph_storage.get_edge(source_quoted, target_quoted)
                    actual_source = source_quoted
                    actual_target = target_quoted
                    
                    if not edge_data:
                        edge_data = await self.graph_storage.get_edge(source, target)
                        actual_source = source
                        actual_target = target
                    
                    if edge_data:
                        # Store original description before overwriting
                        if 'description' in edge_data and 'original_description' not in edge_data:
                            edge_data['original_description'] = edge_data['description']
                        
                        edge_data['description'] = description
                        edge_data['llm_enhanced'] = True
                        await self.graph_storage.upsert_edge(actual_source, actual_target, edge_data=edge_data)
                        edges_enhanced += 1
                        logger.info(f"Enhanced table-column edge: {source} -> {target}")
                    else:
                        logger.warning(f"Could not find graph edge for: {source} -> {target}")
            
            # Apply column-column relationship descriptions to GRAPH EDGES
            for rel_key, description in column_column_descriptions.items():
                if '->' in rel_key:
                    source, target = rel_key.split('->', 1)
                    source = source.strip().strip('"')
                    target = target.strip().strip('"')
                    
                    source_quoted = f'"{source}"'
                    target_quoted = f'"{target}"'
                    
                    edge_data = await self.graph_storage.get_edge(source_quoted, target_quoted)
                    actual_source = source_quoted
                    actual_target = target_quoted
                    
                    if not edge_data:
                        edge_data = await self.graph_storage.get_edge(source, target)
                        actual_source = source
                        actual_target = target
                    
                    if edge_data:
                        # Store original description before overwriting
                        if 'description' in edge_data and 'original_description' not in edge_data:
                            edge_data['original_description'] = edge_data['description']
                        
                        edge_data['description'] = description
                        edge_data['llm_enhanced'] = True
                        await self.graph_storage.upsert_edge(actual_source, actual_target, edge_data=edge_data)
                        edges_enhanced += 1
                        logger.info(f"Enhanced col-col edge: {source} -> {target}")
                    else:
                        logger.warning(f"Could not find graph edge for: {source} -> {target}")
            
            logger.info(f"Enhanced {edges_enhanced} graph edges with LLM descriptions")
            
            # ====================================================================
            # PHASE 4: UPDATE VECTOR DATABASE WITH ENHANCED EDGE DESCRIPTIONS
            # ====================================================================
            
            relationships_added = 0
            relationships_for_vdb = {}

            for relationship_key, description in table_column_descriptions.items():
                if '->' in relationship_key:
                    table_name, column_name = relationship_key.split('->')
                    table_name = table_name.strip()
                    column_name = column_name.strip()
                
                    table_id = f'"{table_name}"'
                    col_id = f'"{column_name}"'

                    # Prepare for vector database
                    edge_vdb_id = compute_mdhash_id(f"{table_id}->{col_id}", prefix="rel-")
                    relationships_for_vdb[edge_vdb_id] = {
                        "src_id": table_id,
                        "tgt_id": col_id,
                        "content": "table_structure, contains_column" + " " + table_id + " " + col_id + " " + description
                    }
                    relationships_added += 1

            for relationship_key, description in column_column_descriptions.items():
                if '->' in relationship_key:
                    col_name, ref_col_name = relationship_key.split('->')
                    col_name = col_name.strip()
                    ref_col_name = ref_col_name.strip()
                    
                    col_id = f'"{col_name}"'
                    ref_col_id = f'"{ref_col_name}"'

                    # Prepare for vector database
                    fk_edge_vdb_id = compute_mdhash_id(f"{col_id}->{ref_col_id}", prefix="rel-")
                    relationships_for_vdb[fk_edge_vdb_id] = {
                        "src_id": col_id,
                        "tgt_id": ref_col_id,
                        "content": "foreign_key, references, data_integrity" + " " + col_id + " " + ref_col_id + " " + description
                    }
                    relationships_added += 1

            if relationships_for_vdb and self.relationships_vdb:
                await self.relationships_vdb.upsert(relationships_for_vdb)
                logger.info(f"Inserted llm enhanced {len(relationships_for_vdb)} relationships into vector database")
            
            # ====================================================================
            # PHASE 5: STORE EDGE DESCRIPTIONS FOR WEIGHT ENHANCEMENT PHASE
            # ====================================================================
            
            # Store descriptions from the LLM response for use in weight enhancement
            temp_d = enhanced_data.get("relationship_descriptions", {})
            if len(temp_d) > 0:
                self.edge_descriptions = self.edge_descriptions | temp_d
            
            # Merge all edge descriptions for weight phase
            self.edge_descriptions = self.edge_descriptions | table_column_descriptions
            self.edge_descriptions = self.edge_descriptions | column_column_descriptions
            
            logger.info(f"Stored {len(self.edge_descriptions)} edge descriptions for weight enhancement phase")
            
            # ====================================================================
            # PHASE 6: LOG DATA INSIGHTS IF AVAILABLE
            # ====================================================================
            
            # Log data insights if available
            data_insights = enhanced_data.get("data_insights", {})
            if data_insights:
                logger.info(f"Data insights extracted: {data_insights}")
            
            logger.info("Finished applying enhanced descriptions")
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from enhanced result: {e}")
        except Exception as e:
            logger.error(f"Error applying enhanced descriptions: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
    
    # Utility methods for chunking
    def _chunk_tables_by_tokens(self, tables_info: Dict[str, Dict[str, Any]], max_tokens: int) -> List[Dict[str, Dict[str, Any]]]:
        """Split tables into chunks based on token limits, with column-level chunking for large tables"""
        from ..utils import encode_string_by_tiktoken
        
        all_chunks = []
        current_chunk = {}
        current_tokens = 0
        
        for table_name, table_data in tables_info.items():
            # Estimate tokens for this table
            table_text = json.dumps({table_name: table_data}, indent=2)
            table_tokens = len(encode_string_by_tiktoken(table_text))
            
            # If single table exceeds limit, split by columns
            if table_tokens > max_tokens:
                logger.info(f"Table {table_name} too large ({table_tokens} tokens), splitting by columns")
                
                # Split this large table by columns
                columns = table_data.get("columns", [])
                max_columns_per_chunk = 15  # Start with 15 columns per chunk
                
                for i in range(0, len(columns), max_columns_per_chunk):
                    chunk_columns = columns[i:i + max_columns_per_chunk]
                    
                    # Create table data for this column chunk
                    chunk_table_data = table_data.copy()
                    chunk_table_data["columns"] = chunk_columns
                    chunk_table_data["column_count"] = len(chunk_columns)
                    
                    # Remove sample data to reduce size
                    chunk_table_data.pop("sample_data", None)
                    chunk_table_data.pop("sample_rows", None)
                    
                    # Test if this chunk is small enough
                    chunk_text = json.dumps({table_name: chunk_table_data}, indent=2)
                    chunk_tokens = len(encode_string_by_tiktoken(chunk_text))
                    
                    if chunk_tokens > max_tokens:
                        # Still too large, try smaller chunks
                        smaller_max = 5
                        for j in range(i, min(i + max_columns_per_chunk, len(columns)), smaller_max):
                            smaller_chunk_columns = columns[j:j + smaller_max]
                            smaller_chunk_data = table_data.copy()
                            smaller_chunk_data["columns"] = smaller_chunk_columns
                            smaller_chunk_data["column_count"] = len(smaller_chunk_columns)
                            smaller_chunk_data.pop("sample_data", None)
                            smaller_chunk_data.pop("sample_rows", None)
                            
                            all_chunks.append({table_name: smaller_chunk_data})
                    else:
                        # Chunk is acceptable size
                        all_chunks.append({table_name: chunk_table_data})
            
            else:
                # Normal chunking logic for manageable tables
                if current_tokens + table_tokens > max_tokens and current_chunk:
                    all_chunks.append(current_chunk)
                    current_chunk = {table_name: table_data}
                    current_tokens = table_tokens
                else:
                    current_chunk[table_name] = table_data
                    current_tokens += table_tokens
        
        # Add the last normal chunk if it has content
        if current_chunk:
            all_chunks.append(current_chunk)
        
        return all_chunks

    def _chunk_relationships_by_tokens(self, relationships_list: str, max_tokens: int) -> List[str]:
        """Split relationships into chunks based on token limits"""
        from ..utils import encode_string_by_tiktoken
        
        relationships = relationships_list.strip().split('\n')
        chunks = []
        current_chunk_lines = []
        current_tokens = 0
        
        for relationship in relationships:
            if not relationship.strip():
                continue
                
            relationship_tokens = len(encode_string_by_tiktoken(relationship))
            
            # If adding this relationship would exceed the limit and current chunk is not empty
            if current_tokens + relationship_tokens > max_tokens and current_chunk_lines:
                chunks.append('\n'.join(current_chunk_lines))
                current_chunk_lines = [relationship]
                current_tokens = relationship_tokens
            else:
                current_chunk_lines.append(relationship)
                current_tokens += relationship_tokens
        
        # Add the last chunk if it has content
        if current_chunk_lines:
            chunks.append('\n'.join(current_chunk_lines))
        
        return chunks

    def _extract_tables_from_relationships(self, relationships_chunk: str) -> List[str]:
        """Extract fully qualified table names from this chunk of relationships"""
        import re
        
        tables = set()
        
        for line in relationships_chunk.split('\n'):
            # Find quoted strings (tables or columns)
            matches = re.findall(r'"([^"]+)"', line)
            for match in matches:
                if '.' in match:
                    parts = match.split('.')
                    if len(parts) > 1:
                        # ✅ Keep all but the last as table (e.g., project.dataset.table)
                        table_name = '.'.join(parts[:-1])
                        tables.add(table_name)
                    else:
                        tables.add(match)
                else:
                    tables.add(match)
        
        return list(tables)


    def _create_minimal_schema_for_tables(self, table_names: List[str]) -> str:
        """Create minimal schema containing only the specified tables"""
        minimal_tables = {}
        
        # Get table info from cache
        for table_name in table_names:
            if table_name in self._tables_info_cache:
                # Include only essential info to minimize tokens
                table_data = self._tables_info_cache[table_name]
                minimal_tables[table_name] = {
                    "name": table_data["name"],
                    "column_count": table_data["column_count"],
                    "columns": [
                        {
                            "name": col["name"],
                            "type": col["type"],
                            "is_primary_key": col.get("is_primary_key", False),
                            "is_foreign_key": col.get("is_foreign_key", False)
                        }
                        for col in table_data.get("columns", [])
                    ]
                }
        
        return json.dumps({"tables": minimal_tables}, indent=2, ensure_ascii=False)