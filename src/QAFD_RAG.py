import asyncio
import os
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Type, cast, List, Dict, Any, Optional

from .llm import (
    gpt_4o_mini_complete,
    gpt_oss_120b_complete,
    local_sentence_embedding,
    openai_cloud_embedding,
    is_local_model,
    is_cloud_model,
)

# Try to import the new functions, but don't fail if they don't exist
try:
    from .llm import get_embedding_func_for_model, EMBEDDING_CONFIGS
    HAS_EMBEDDING_CONFIGS = True
except ImportError:
    HAS_EMBEDDING_CONFIGS = False
    print("Warning: get_embedding_func_for_model not found in llm.py")
    print("Please update your llm.py file with the new version")

from .operate import (
    chunking_by_token_size,
    extract_entities,
    kg_query,
)
from .indexing import DatabaseSchemaBuilder

from .utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    logger,
    set_logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)

from .storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)


async def abuild_from_excel_files(self, excel_paths: List[str]) -> Dict[str, Any]:
    """Build KG from Excel files"""
    from .indexing import ExcelSchemaBuilder
    
    builder = ExcelSchemaBuilder(
        graph_storage=self.chunk_entity_relation_graph,
        entities_vdb=self.entities_vdb,
        relationships_vdb=self.relationships_vdb
    )
    
    result = await builder.build_from_excel_files(excel_paths)
    await self._insert_done()
    
    logger.info(f"Excel KG build completed: {result}")
    return result


def build_from_excel_files(self, excel_paths: List[str]):
    """Sync wrapper"""
    loop = always_get_an_event_loop()
    return loop.run_until_complete(self.abuild_from_excel_files(excel_paths))


def lazy_external_import(module_name: str, class_name: str):
    """Lazily import a class from an external module based on the package of the caller."""
    import inspect

    caller_frame = inspect.currentframe().f_back
    module = inspect.getmodule(caller_frame)
    package = module.__package__ if module else None

    def import_class(*args, **kwargs):
        import importlib
        module = importlib.import_module(module_name, package=package)
        cls = getattr(module, class_name)
        return cls(*args, **kwargs)

    return import_class


Neo4JStorage = lazy_external_import(".kg.neo4j_impl", "Neo4JStorage")
OracleKVStorage = lazy_external_import(".kg.oracle_impl", "OracleKVStorage")
OracleGraphStorage = lazy_external_import(".kg.oracle_impl", "OracleGraphStorage")
OracleVectorDBStorage = lazy_external_import(".kg.oracle_impl", "OracleVectorDBStorage")
MilvusVectorDBStorge = lazy_external_import(".kg.milvus_impl", "MilvusVectorDBStorge")
MongoKVStorage = lazy_external_import(".kg.mongo_impl", "MongoKVStorage")
ChromaVectorDBStorage = lazy_external_import(".kg.chroma_impl", "ChromaVectorDBStorage")
TiDBKVStorage = lazy_external_import(".kg.tidb_impl", "TiDBKVStorage")
TiDBVectorDBStorage = lazy_external_import(".kg.tidb_impl", "TiDBVectorDBStorage")
AGEStorage = lazy_external_import(".kg.age_impl", "AGEStorage")


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop
    except RuntimeError:
        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


@dataclass
class QAFD_RAG:
    working_dir: str = field(
        default_factory=lambda: f"./QAFD_RAG_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )

    embedding_cache_config: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "similarity_threshold": 0.95,
            "use_llm_check": False,
        }
    )
    kv_storage: str = field(default="JsonKVStorage")
    vector_storage: str = field(default="NanoVectorDBStorage")
    graph_storage: str = field(default="NetworkXStorage")

    current_log_level = logger.level
    log_level: str = field(default=current_log_level)

    # Chunking parameters
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"

    # Entity extraction parameters
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 5000

    # Node embedding algorithm
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # ============================================================================
    # EMBEDDING CONFIGURATION (NEW: Configurable embedding models)
    # ============================================================================
    
    # Embedding model key (from EMBEDDING_CONFIGS in llm.py)
    embedding_model_key: Optional[str] = None
    
    # Embedding function and dimensions (auto-configured from embedding_model_key)
    embedding_func: Optional[EmbeddingFunc] = None
    embedding_dim: Optional[int] = None
    
    # Embedding parameters
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16
    max_embed_tokens: int = 8192  # Maximum tokens per embedding request

    # ============================================================================
    # LLM CONFIGURATION
    # ============================================================================
    
    llm_model_func: callable = gpt_4o_mini_complete
    llm_model_name: str = "gpt-4o-mini"
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 16
    llm_model_kwargs: dict = field(default_factory=dict)

    # Vector DB storage parameters
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)

    enable_llm_cache: bool = True

    # Additional parameters
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    def __post_init__(self):
        log_file = os.path.join("QAFD_RAG.log")
        set_logger(log_file)
        logger.setLevel(self.log_level)

        logger.info(f"Logger initialized for working directory: {self.working_dir}")

        # ============================================================================
        # EMBEDDING MODEL CONFIGURATION LOGIC (NEW)
        # ============================================================================
        
        if HAS_EMBEDDING_CONFIGS:
            # Priority 1: Use explicitly passed embedding_model_key
            if self.embedding_model_key:
                logger.info(f"[Embedding Config] Using explicit embedding_model_key: {self.embedding_model_key}")
                embedding_func, embedding_dim, emb_config = get_embedding_func_for_model(self.embedding_model_key)
                self.embedding_func = embedding_func
                self.embedding_dim = embedding_dim
                logger.info(f"[Embedding Config] {emb_config['description']}")
                logger.info(f"[Embedding Config] Dimensions: {embedding_dim}, Max tokens: {emb_config['max_tokens']}")
            
            # Priority 2: Check environment variable EMBEDDING_MODEL_KEY
            elif os.environ.get("EMBEDDING_MODEL_KEY"):
                embedding_key = os.environ.get("EMBEDDING_MODEL_KEY")
                logger.info(f"[Embedding Config] Using env EMBEDDING_MODEL_KEY: {embedding_key}")
                embedding_func, embedding_dim, emb_config = get_embedding_func_for_model(embedding_key)
                self.embedding_func = embedding_func
                self.embedding_dim = embedding_dim
                self.embedding_model_key = embedding_key
                logger.info(f"[Embedding Config] {emb_config['description']}")
            
            # Priority 3: Check environment variable USE_OPENAI_EMBEDDINGS (legacy)
            elif os.environ.get("USE_OPENAI_EMBEDDINGS") == "1":
                logger.info(f"[Embedding Config] Using legacy USE_OPENAI_EMBEDDINGS=1")
                self.embedding_func = openai_cloud_embedding
                self.embedding_dim = 1024
                self.embedding_model_key = "openai-large"
                logger.info(f"[Embedding Config] OpenAI cloud embeddings (1024-dim)")
            
            elif os.environ.get("USE_OPENAI_EMBEDDINGS") == "0":
                logger.info(f"[Embedding Config] Using legacy USE_OPENAI_EMBEDDINGS=0")
                self.embedding_func = local_sentence_embedding
                self.embedding_dim = 1024
                self.embedding_model_key = "jina-v3"
                logger.info(f"[Embedding Config] Local Jina v3 embeddings (1024-dim)")
            
            # Priority 4: Auto-detect based on LLM model name
            elif self.llm_model_name:
                model_name = self.llm_model_name.lower()
                if is_local_model(model_name):
                    logger.info(f"[Embedding Config] Local LLM detected ({model_name}) → using local embeddings")
                    self.embedding_func = local_sentence_embedding
                    self.embedding_dim = 1024
                    self.embedding_model_key = "jina-v3"
                else:
                    logger.info(f"[Embedding Config] Cloud LLM detected ({model_name}) → using OpenAI embeddings")
                    self.embedding_func = openai_cloud_embedding
                    self.embedding_dim = 1024
                    self.embedding_model_key = "openai-large"
            
            # Priority 5: Default to local Jina v3
            else:
                logger.info(f"[Embedding Config] No configuration found → defaulting to Jina v3 (local)")
                self.embedding_func = local_sentence_embedding
                self.embedding_dim = 1024
                self.embedding_model_key = "jina-v3"
        else:
            # Fallback to old behavior if new functions not available
            logger.warning("[Embedding Config] Using legacy embedding configuration")
            env_embedding_setting = os.environ.get("USE_OPENAI_EMBEDDINGS")
            
            if env_embedding_setting == "0":
                self.embedding_func = local_sentence_embedding
                self.embedding_dim = 1024
                logger.info(f"[Embedding Override] Using local embeddings (1024-dim) - forced by USE_OPENAI_EMBEDDINGS=0")
            elif env_embedding_setting == "1":
                self.embedding_func = openai_cloud_embedding
                self.embedding_dim = 1024
                logger.info(f"[Embedding Override] Using OpenAI embeddings (1024-dim) - forced by USE_OPENAI_EMBEDDINGS=1")
            elif is_local_model(self.llm_model_name.lower() if self.llm_model_name else ""):
                self.embedding_func = local_sentence_embedding
                self.embedding_dim = 1024
                logger.info(f"[Embedding] Local model detected → using local embeddings (1024-dim)")
            else:
                self.embedding_func = openai_cloud_embedding
                self.embedding_dim = 1024
                logger.info(f"[Embedding] Cloud model detected → using OpenAI embeddings (1024-dim)")
        
        # Validate embedding function was set
        if self.embedding_func is None:
            logger.error("[Embedding Config] Failed to configure embedding function!")
            raise ValueError("Embedding function not configured")
        
        if self.embedding_dim is None:
            self.embedding_dim = 1024  # Default
            logger.warning(f"[Embedding Config] embedding_dim not set, defaulting to 1024")
        
        logger.info(f"[Embedding Config] ✅ Final: {self.embedding_model_key if self.embedding_model_key else 'auto'} ({self.embedding_dim}-dim)")

        # ============================================================================
        # STORAGE INITIALIZATION
        # ============================================================================
        
        self.key_string_value_json_storage_cls: Type[BaseKVStorage] = (
            self._get_storage_class()[self.kv_storage]
        )
        self.vector_db_storage_cls: Type[BaseVectorStorage] = self._get_storage_class()[
            self.vector_storage
        ]
        self.graph_storage_cls: Type[BaseGraphStorage] = self._get_storage_class()[
            self.graph_storage
        ]

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache",
                global_config=asdict(self),
                embedding_func=None,
            )
            if self.enable_llm_cache
            else None
        )
        
        # Limit async calls for embedding function
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )

        # Initialize storage components with embedding function
        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )

        # Vector databases for entities, relationships, and chunks
        self.entities_vdb = self.vector_db_storage_cls(
            namespace="entities",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )
        self.relationships_vdb = self.vector_db_storage_cls(
            namespace="relationships",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id"},
        )
        self.chunks_vdb = self.vector_db_storage_cls(
            namespace="chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )

        # Configure LLM function
        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache
                and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    global_config=asdict(self),
                ),
                **self.llm_model_kwargs,
            )
        )
        
        # Initialize database schema builder
        self.schema_builder = DatabaseSchemaBuilder(
            graph_storage=self.chunk_entity_relation_graph,
            entities_vdb=self.entities_vdb,
            relationships_vdb=self.relationships_vdb,
            llm_model_func=self.llm_model_func
        )

    def _get_storage_class(self) -> dict[str, Type]:
        return {
            # Key-Value Storage
            "JsonKVStorage": JsonKVStorage,
            "OracleKVStorage": OracleKVStorage,
            "MongoKVStorage": MongoKVStorage,
            "TiDBKVStorage": TiDBKVStorage,
            # Vector Storage
            "NanoVectorDBStorage": NanoVectorDBStorage,
            "OracleVectorDBStorage": OracleVectorDBStorage,
            "MilvusVectorDBStorge": MilvusVectorDBStorge,
            "ChromaVectorDBStorage": ChromaVectorDBStorage,
            "TiDBVectorDBStorage": TiDBVectorDBStorage,
            # Graph Storage
            "NetworkXStorage": NetworkXStorage,
            "Neo4JStorage": Neo4JStorage,
            "OracleGraphStorage": OracleGraphStorage,
            "AGEStorage": AGEStorage,
        }

    def insert(self, string_or_strings, addon_params=None):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings, addon_params))

    async def ainsert(self, string_or_strings, addon_params=None):
        update_storage = False
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]

            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("All docs are already in the storage")
                return
            update_storage = True
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            inserting_chunks = {}
            for doc_key, doc in tqdm_async(
                new_docs.items(), desc="Chunking documents", unit="doc"
            ):
                chunks = {
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                        **dp,
                        "full_doc_id": doc_key,
                    }
                    for dp in chunking_by_token_size(
                        doc["content"],
                        overlap_token_size=self.chunk_overlap_token_size,
                        max_token_size=self.chunk_token_size,
                        tiktoken_model=self.tiktoken_model_name,
                    )
                }
                inserting_chunks.update(chunks)
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

            await self.chunks_vdb.upsert(inserting_chunks)

            logger.info("[Entity Extraction]...")
            
            # Create a temporary config with custom addon_params if provided
            temp_config = asdict(self)
            if addon_params is not None:
                temp_config["addon_params"] = addon_params
            
            maybe_new_kg = await extract_entities(
                inserting_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                global_config=temp_config,
            )
            if maybe_new_kg is None:
                logger.warning("No new entities and relationships found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg

            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            if update_storage:
                await self._insert_done()

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.relationships_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def insert_custom_kg(self, custom_kg: dict):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert_custom_kg(custom_kg))

    async def ainsert_custom_kg(self, custom_kg: dict):
        update_storage = False
        try:
            all_chunks_data = {}
            chunk_to_source_map = {}
            for chunk_data in custom_kg.get("chunks", []):
                chunk_content = chunk_data["content"]
                source_id = chunk_data["source_id"]
                chunk_id = compute_mdhash_id(chunk_content.strip(), prefix="chunk-")

                chunk_entry = {"content": chunk_content.strip(), "source_id": source_id}
                all_chunks_data[chunk_id] = chunk_entry
                chunk_to_source_map[source_id] = chunk_id
                update_storage = True

            if self.chunks_vdb is not None and all_chunks_data:
                await self.chunks_vdb.upsert(all_chunks_data)
            if self.text_chunks is not None and all_chunks_data:
                await self.text_chunks.upsert(all_chunks_data)

            all_entities_data = []
            for entity_data in custom_kg.get("entities", []):
                entity_name = f'"{entity_data["entity_name"].lower()}"'
                entity_type = entity_data.get("entity_type", "UNKNOWN")
                description = entity_data.get("description", "No description provided")

                source_chunk_id = entity_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")

                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Entity '{entity_name}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                node_data = {
                    "entity_type": entity_type,
                    "description": description,
                    "source_id": source_id,
                }

                await self.chunk_entity_relation_graph.upsert_node(
                    entity_name, node_data=node_data
                )
                node_data["entity_name"] = entity_name
                all_entities_data.append(node_data)
                update_storage = True

            all_relationships_data = []
            for relationship_data in custom_kg.get("relationships", []):
                src_id = f'"{relationship_data["src_id"].lower()}"'
                tgt_id = f'"{relationship_data["tgt_id"].lower()}"'
                description = relationship_data["description"]
                keywords = relationship_data["keywords"]
                weight = relationship_data.get("weight", 1.0)

                source_chunk_id = relationship_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")

                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Relationship from '{src_id}' to '{tgt_id}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                for need_insert_id in [src_id, tgt_id]:
                    if not (
                        await self.chunk_entity_relation_graph.has_node(need_insert_id)
                    ):
                        await self.chunk_entity_relation_graph.upsert_node(
                            need_insert_id,
                            node_data={
                                "source_id": source_id,
                                "description": "UNKNOWN",
                                "entity_type": "UNKNOWN",
                            },
                        )

                await self.chunk_entity_relation_graph.upsert_edge(
                    src_id,
                    tgt_id,
                    edge_data={
                        "weight": weight,
                        "description": description,
                        "keywords": keywords,
                        "source_id": source_id,
                    },
                )
                edge_data = {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "description": description,
                    "keywords": keywords,
                }
                all_relationships_data.append(edge_data)
                update_storage = True

            if self.entities_vdb is not None:
                data_for_vdb = {
                    compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                        "content": dp["entity_name"] + dp["description"],
                        "entity_name": dp["entity_name"],
                    }
                    for dp in all_entities_data
                }
                await self.entities_vdb.upsert(data_for_vdb)

            if self.relationships_vdb is not None:
                data_for_vdb = {
                    compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                        "src_id": dp["src_id"],
                        "tgt_id": dp["tgt_id"],
                        "content": dp["keywords"]
                        + dp["src_id"]
                        + dp["tgt_id"]
                        + dp["description"],
                    }
                    for dp in all_relationships_data
                }
                await self.relationships_vdb.upsert(data_for_vdb)
        finally:
            if update_storage:
                await self._insert_done()
    
    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))
    
    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        if param.mode in ["local", "global", "hybrid"]:
            response = await kg_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache
                and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    global_config=asdict(self),
                ),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def delete_by_entity(self, entity_name: str):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_entity(self, entity_name: str):
        entity_name = f'"{entity_name.lower()}"'

        try:
            await self.entities_vdb.delete_entity(entity_name)
            await self.relationships_vdb.delete_relation(entity_name)
            await self.chunk_entity_relation_graph.delete_node(entity_name)

            logger.info(
                f"Entity '{entity_name}' and its relationships have been deleted."
            )
            await self._delete_by_entity_done()
        except Exception as e:
            logger.error(f"Error while deleting entity '{entity_name}': {e}")

    async def _delete_by_entity_done(self):
        tasks = []
        for storage_inst in [
            self.entities_vdb,
            self.relationships_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def build_from_database_schema(self, 
                                  schema_file_path: str, 
                                  metadata_file_path: str = None,
                                  language: str = "English"):
        """
        Build knowledge graph from database schema JSON file
        
        This method manually constructs the knowledge graph from a JSON schema file,
        avoiding the chunking issues that can cause LLM errors. It follows the approach
        used in CoFD for database schema processing.
        
        Args:
            schema_file_path: Path to the JSON schema file
            metadata_file_path: Optional path to metadata file
            language: Output language for descriptions
            
        Returns:
            Dictionary containing build statistics
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.abuild_from_database_schema(
            schema_file_path, metadata_file_path, language
        ))

    async def abuild_from_database_schema(self, 
                                        schema_file_path: str, 
                                        metadata_file_path: str = None,
                                        language: str = "English"):
        """
        Async version of build_from_database_schema
        """
        try:
            # Use the schema builder to construct the knowledge graph
            result = await self.schema_builder.build_from_json_schema(
                schema_file_path, metadata_file_path, language
            )
            
            # Update storage after building
            await self._insert_done()
            
            logger.info(f"Database schema build completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error building from database schema: {e}")
            raise