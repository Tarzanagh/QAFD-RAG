import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from ..base import BaseGraphStorage, BaseVectorStorage
from ..utils import logger, compute_mdhash_id

class ExcelSchemaBuilder:
    """Build KG from Excel files: File → Sheet → Column hierarchy"""
    
    def __init__(self, 
                 graph_storage: BaseGraphStorage,
                 entities_vdb: BaseVectorStorage,
                 relationships_vdb: BaseVectorStorage):
        self.graph_storage = graph_storage
        self.entities_vdb = entities_vdb
        self.relationships_vdb = relationships_vdb
    
    async def build_from_excel_files(self, excel_paths: List[str]) -> Dict[str, Any]:
        """Build KG from multiple Excel files"""
        stats = {
            "files_processed": 0,
            "sheets_processed": 0,
            "columns_processed": 0,
            "nodes_added": 0,
            "edges_added": 0
        }
        
        for excel_path in excel_paths:
            file_stats = await self._process_excel_file(excel_path)
            stats["files_processed"] += 1
            stats["sheets_processed"] += file_stats["sheets"]
            stats["columns_processed"] += file_stats["columns"]
            stats["nodes_added"] += file_stats["nodes"]
            stats["edges_added"] += file_stats["edges"]
        
        logger.info(f"Excel KG build complete: {stats}")
        return stats
    
    async def _process_excel_file(self, excel_path: str) -> Dict[str, int]:
        """Process single Excel file"""
        file_name = Path(excel_path).stem
        logger.info(f"Processing Excel file: {file_name}")
        
        xl_file = pd.ExcelFile(excel_path)
        stats = {"sheets": 0, "columns": 0, "nodes": 0, "edges": 0}
        
        # Create FILE node (master/parent)
        file_id = f'"{file_name}"'
        file_node = {
            "entity_type": "excel_file",
            "description": f"Excel file: {file_name}",
            "sheet_count": len(xl_file.sheet_names),
            "source_id": "excel_extraction",
            "path": excel_path
        }
        await self.graph_storage.upsert_node(file_id, node_data=file_node)
        stats["nodes"] += 1
        
        # Add to vector DB
        file_vdb_id = compute_mdhash_id(file_id, prefix="ent-")
        await self.entities_vdb.upsert({
            file_vdb_id: {
                "content": f"{file_id} {file_node['description']}",
                "entity_name": file_id
            }
        })
        
        # Process each sheet
        for sheet_name in xl_file.sheet_names:
            sheet_stats = await self._process_sheet(
                file_id, file_name, sheet_name, excel_path
            )
            stats["sheets"] += 1
            stats["columns"] += sheet_stats["columns"]
            stats["nodes"] += sheet_stats["nodes"]
            stats["edges"] += sheet_stats["edges"]
        
        return stats
    
    async def _process_sheet(self, file_id: str, file_name: str, 
                            sheet_name: str, excel_path: str) -> Dict[str, int]:
        """Process single sheet (sub-parent)"""
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        stats = {"columns": 0, "nodes": 0, "edges": 0}
        
        # Create SHEET node
        sheet_id = f'"{file_name}.{sheet_name}"'
        sheet_node = {
            "entity_type": "sheet",
            "description": f"Sheet: {sheet_name} in {file_name}",
            "row_count": len(df),
            "column_count": len(df.columns),
            "source_id": "excel_extraction"
        }
        await self.graph_storage.upsert_node(sheet_id, node_data=sheet_node)
        stats["nodes"] += 1
        
        # Add to vector DB
        sheet_vdb_id = compute_mdhash_id(sheet_id, prefix="ent-")
        await self.entities_vdb.upsert({
            sheet_vdb_id: {
                "content": f"{sheet_id} {sheet_node['description']}",
                "entity_name": sheet_id
            }
        })
        
        # Create FILE → SHEET edge
        await self.graph_storage.upsert_edge(file_id, sheet_id, edge_data={
            "weight": 10.0,
            "description": f"File {file_name} contains sheet {sheet_name}",
            "keywords": "contains_sheet, hierarchy",
            "source_id": "excel_extraction"
        })
        stats["edges"] += 1
        
        # Process columns
        for col_name in df.columns:
            col_stats = await self._process_column(
                sheet_id, file_name, sheet_name, col_name, df
            )
            stats["columns"] += 1
            stats["nodes"] += col_stats["nodes"]
            stats["edges"] += col_stats["edges"]
        
        return stats
    
    async def _process_column(self, sheet_id: str, file_name: str,
                             sheet_name: str, col_name: str, 
                             df: pd.DataFrame) -> Dict[str, int]:
        """Process single column (child/leaf)"""
        stats = {"nodes": 0, "edges": 0}
        
        # Extract column metadata
        col_data = df[col_name]
        dtype = str(col_data.dtype)
        null_count = int(col_data.isnull().sum())
        
        # Get sample values (first 3 non-null)
        sample_values = col_data.dropna().head(3).tolist()
        sample_str = str(sample_values)[:100]  # Limit length
        
        # Create COLUMN node
        col_id = f'"{file_name}.{sheet_name}.{col_name}"'
        col_node = {
            "entity_type": "column",
            "description": f"Column: {col_name} (type: {dtype})",
            "data_type": dtype,
            "null_count": null_count,
            "sample_values": sample_str,
            "source_id": "excel_extraction"
        }
        await self.graph_storage.upsert_node(col_id, node_data=col_node)
        stats["nodes"] += 1
        
        # Add to vector DB
        col_vdb_id = compute_mdhash_id(col_id, prefix="ent-")
        await self.entities_vdb.upsert({
            col_vdb_id: {
                "content": f"{col_id} {col_node['description']} samples: {sample_str}",
                "entity_name": col_id
            }
        })
        
        # Create SHEET → COLUMN edge
        await self.graph_storage.upsert_edge(sheet_id, col_id, edge_data={
            "weight": 8.0,
            "description": f"Sheet {sheet_name} contains column {col_name}",
            "keywords": "contains_column, hierarchy",
            "source_id": "excel_extraction"
        })
        stats["edges"] += 1
        
        return stats