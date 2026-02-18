import asyncio
import html
import os
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import dataclass
from typing import Any, Union, cast
import networkx as nx
import numpy as np
from nano_vectordb import NanoVectorDB

from .utils import (
    logger,
    load_json,
    write_json,
    compute_mdhash_id,
)

from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
)


@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        write_json(self._data, self._file_name)

    async def get_by_id(self, id):
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        left_data = {k: v for k, v in data.items() if k not in self._data}
        self._data.update(left_data)
        return left_data

    async def drop(self):
        self._data = {}


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        async def wrapped_task(batch):
            result = await self.embedding_func(batch)
            pbar.update(1)
            return result

        embedding_tasks = [wrapped_task(batch) for batch in batches]
        pbar = tqdm_async(
            total=len(embedding_tasks), desc="Generating embeddings", unit="batch"
        )
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)
        if len(embeddings) == len(list_data):
            for i, d in enumerate(list_data):
                d["__vector__"] = embeddings[i]
            results = self._client.upsert(datas=list_data)
            return results
        else:
            # sometimes the embedding is not returned correctly. just log it.
            logger.error(
                f"embedding is not 1-1 with data, {len(embeddings)} != {len(list_data)}"
            )

    async def query(self, query: str, top_k=5):
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    @property
    def client_storage(self):
        return getattr(self._client, "_NanoVectorDB__storage")

    async def delete_entity(self, entity_name: str):
        try:
            entity_id = [compute_mdhash_id(entity_name, prefix="ent-")]

            if self._client.get(entity_id):
                self._client.delete(entity_id)
                logger.info(f"Entity {entity_name} have been deleted.")
            else:
                logger.info(f"No entity found with name {entity_name}.")
        except Exception as e:
            logger.error(f"Error while deleting entity {entity_name}: {e}")

    async def delete_relation(self, entity_name: str):
        try:
            relations = [
                dp
                for dp in self.client_storage["data"]
                if dp["src_id"] == entity_name or dp["tgt_id"] == entity_name
            ]
            ids_to_delete = [relation["__id__"] for relation in relations]

            if ids_to_delete:
                self._client.delete(ids_to_delete)
                logger.info(
                    f"All relations related to entity {entity_name} have been deleted."
                )
            else:
                logger.info(f"No relations found for entity {entity_name}.")
        except Exception as e:
            logger.error(
                f"Error while deleting relations for entity {entity_name}: {e}"
            )

    async def index_done_callback(self):
        self._client.save()


@dataclass
class NetworkXStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name) -> nx.DiGraph:
        if os.path.exists(file_name):
            graph = nx.read_graphml(file_name)
            
            # Clean the graph by filtering out None values
            clean_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()
            
            # Add nodes with filtered data
            for node, data in graph.nodes(data=True):
                filtered_data = {k: v for k, v in data.items() if v is not None}
                clean_graph.add_node(node, **filtered_data)
            
            # Add edges with filtered data
            for u, v, data in graph.edges(data=True):
                filtered_data = {k: v_val for k, v_val in data.items() if v_val is not None}
                clean_graph.add_edge(u, v, **filtered_data)
            
            return clean_graph
        return None
    # def load_nx_graph(file_name) -> nx.Graph:
    #     if os.path.exists(file_name):
    #         return nx.read_graphml(file_name)
    #     return None

    @staticmethod
    def write_nx_graph(graph: nx.DiGraph, file_name):
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        
        # Create a clean copy of the graph with None values filtered out
        clean_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()
        
        # Add nodes with filtered data
        for node, data in graph.nodes(data=True):
            filtered_data = {k: v for k, v in data.items() if v is not None}
            clean_graph.add_node(node, **filtered_data)
        
        # Add edges with filtered data
        for u, v, data in graph.edges(data=True):
            filtered_data = {k: v_val for k, v_val in data.items() if v_val is not None}
            clean_graph.add_edge(u, v, **filtered_data)
        
        nx.write_graphml(clean_graph, file_name)

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        """
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        graph = cast(nx.Graph, largest_connected_component(graph))
        node_mapping = {
            node: html.unescape(node.lower().strip()) for node in graph.nodes()
        }  # type: ignore
        graph = nx.relabel_nodes(graph, node_mapping)
        
        # Clean the graph by filtering out None values before stabilizing
        clean_graph = nx.Graph()
        
        # Add nodes with filtered data
        for node, data in graph.nodes(data=True):
            filtered_data = {k: v for k, v in data.items() if v is not None}
            clean_graph.add_node(node, **filtered_data)
        
        # Add edges with filtered data
        for u, v, data in graph.edges(data=True):
            filtered_data = {k: v_val for k, v_val in data.items() if v_val is not None}
            clean_graph.add_edge(u, v, **filtered_data)
        
        return NetworkXStorage._stabilize_graph(clean_graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        # Add nodes with filtered data
        for node, data in sorted_nodes:
            filtered_data = {k: v for k, v in data.items() if v is not None}
            fixed_graph.add_node(node, **filtered_data)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        # Add edges with filtered data
        for u, v, data in edges:
            filtered_data = {k: v_val for k, v_val in data.items() if v_val is not None}
            fixed_graph.add_edge(u, v, **filtered_data)
        return fixed_graph

    def __post_init__(self):
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        # Persistent embeddings file (store node embeddings on disk under working dir)
        self._node_embeddings_file = os.path.join(
            self.global_config["working_dir"], f"kg_node_embeddings_{self.namespace}.json"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
            # The load_nx_graph method already cleans the graph, so we can use it directly
            self._graph = preloaded_graph
        else:
            self._graph = nx.DiGraph()
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }
        # Add cached NetworkX graph instance
        self._cached_nx_graph = None
        # In-memory cache is kept empty by default; embeddings are persisted on disk
        self._node_embeddings_cache = {}
        # Add query embeddings cache
        self._query_embeddings_cache = {}

    def _invalidate_node_embeddings_file(self):
        """Remove persisted embeddings file to avoid stale data."""
        try:
            if os.path.exists(self._node_embeddings_file):
                os.remove(self._node_embeddings_file)
                logger.info(
                    f"Removed stale node embeddings file: {self._node_embeddings_file}"
                )
        except Exception as e:
            logger.warning(
                f"Failed to remove node embeddings file {self._node_embeddings_file}: {e}"
            )

    async def index_done_callback(self):
        # Clean the graph before writing to ensure no None values
        clean_graph = nx.DiGraph() if self._graph.is_directed() else nx.Graph()
        
        # Add nodes with filtered data
        for node, data in self._graph.nodes(data=True):
            filtered_data = {k: v for k, v in data.items() if v is not None}
            clean_graph.add_node(node, **filtered_data)
        
        # Add edges with filtered data
        for u, v, data in self._graph.edges(data=True):
            filtered_data = {k: v_val for k, v_val in data.items() if v_val is not None}
            clean_graph.add_edge(u, v, **filtered_data)
        
        NetworkXStorage.write_nx_graph(clean_graph, self._graphml_xml_file)
        # Clear cache as graph may have been updated
        self._cached_nx_graph = None
        self._node_embeddings_cache = {}
        # Remove persisted node embeddings file to avoid stale embeddings
        self._invalidate_node_embeddings_file()
        self._query_embeddings_cache = {}

    async def get_cached_nx_graph(self) -> nx.Graph:
        """Get cached NetworkX graph instance to avoid repeated construction"""
        if self._cached_nx_graph is None:
            logger.info("Building cached NetworkX graph...")
            # Build undirected graph for flow diffusion
            G = nx.Graph()
            
            # Add all edges
            edge_count = 0
            for u, v in self._graph.edges():
                edge_data = self._graph.edges[u, v]
                # Filter out None values to avoid GraphML writer issues
                filtered_edge_data = {k: v_val for k, v_val in edge_data.items() if v_val is not None} if edge_data else {}
                weight = filtered_edge_data.get('weight', 1.0)
                G.add_edge(u, v, weight=weight, **{k: v_val for k, v_val in filtered_edge_data.items() if k != 'weight'})
                edge_count += 1
            
            # Add all nodes
            node_count = 0
            for node, data in self._graph.nodes(data=True):
                # Filter out None values to avoid GraphML writer issues
                filtered_data = {k: v for k, v in data.items() if v is not None}
                G.add_node(node, **filtered_data)
                node_count += 1
            
            self._cached_nx_graph = G
            logger.info(f"Successfully cached NetworkX graph with {node_count} nodes and {edge_count} edges")
        else:
            logger.info(f"Using cached NetworkX graph with {self._cached_nx_graph.number_of_nodes()} nodes and {self._cached_nx_graph.number_of_edges()} edges")
        
        return self._cached_nx_graph

    async def get_cached_node_embeddings(self, global_config: dict) -> dict:
        """Get node embeddings; prioritize loading from working directory file, compute and persist if not exists.

        File path: working_dir/kg_node_embeddings_{namespace}.json
        """
        # 1) Try to load from persisted file
        try:
            if os.path.exists(self._node_embeddings_file):
                persisted = load_json(self._node_embeddings_file) or {}
                if isinstance(persisted, dict) and len(persisted) > 0:
                    logger.info(
                        f"Loaded node embeddings from file ({len(persisted)} nodes): {self._node_embeddings_file}"
                    )
                    return persisted
        except Exception as e:
            logger.warning(
                f"Failed to load node embeddings file {self._node_embeddings_file}: {e}"
            )

        # 2) Compute if not available on disk
        logger.info("No persisted node embeddings found, computing new ones...")
        if "embedding_func" not in global_config or not global_config["embedding_func"]:
            logger.warning("No embedding function available in global config")
            return {}

        try:
            all_nodes = list(self._graph.nodes())
            if not all_nodes:
                logger.warning("No nodes found in graph for embedding computation")
                return {}

            logger.info(
                f"Preparing to compute embeddings for {len(all_nodes)} nodes..."
            )

            # Get batch size and token limits
            batch_size = global_config.get("embedding_batch_num", 32)
            max_tokens_per_request = global_config.get("max_embed_tokens", 8192)

            # Prepare node texts
            node_texts = []
            for node in all_nodes:
                node_data = self._graph.nodes.get(node, {})
                if node_data and "description" in node_data:
                    node_text = f"{node} {node_data['description']}"
                else:
                    node_text = node
                node_texts.append(node_text)

            # Token counting
            import tiktoken
            tiktoken_model = global_config.get("tiktoken_model_name", "gpt-4o-mini")
            try:
                encoding = tiktoken.encoding_for_model(tiktoken_model)
            except Exception:
                encoding = tiktoken.get_encoding("cl100k_base")

            all_embeddings = []
            for i in range(0, len(node_texts), batch_size):
                batch_texts = node_texts[i : i + batch_size]
                total_tokens = sum(len(encoding.encode(text)) for text in batch_texts)

                if total_tokens > max_tokens_per_request:
                    logger.warning(
                        f"Batch {i//batch_size + 1} exceeds token limit ({total_tokens} > {max_tokens_per_request}), reducing batch size..."
                    )
                    sub_batch_size = max(1, batch_size // 2)
                    for j in range(0, len(batch_texts), sub_batch_size):
                        sub_batch_texts = batch_texts[j : j + sub_batch_size]
                        sub_total_tokens = sum(
                            len(encoding.encode(text)) for text in sub_batch_texts
                        )
                        if sub_total_tokens > max_tokens_per_request:
                            logger.warning(
                                f"Sub-batch still exceeds token limit ({sub_total_tokens} > {max_tokens_per_request}), processing one by one..."
                            )
                            for text in sub_batch_texts:
                                try:
                                    embedding_array = await global_config["embedding_func"](
                                        [text]
                                    )
                                    all_embeddings.append(embedding_array[0])
                                except Exception as e:
                                    logger.error(
                                        f"Failed to process text for embedding: {e}"
                                    )
                                    all_embeddings.append([0.0] * 1536)
                        else:
                            try:
                                embedding_array = await global_config["embedding_func"](
                                    sub_batch_texts
                                )
                                all_embeddings.extend(embedding_array)
                            except Exception as e:
                                logger.error(f"Failed to process sub-batch: {e}")
                                for _ in sub_batch_texts:
                                    all_embeddings.append([0.0] * 1536)
                else:
                    try:
                        embedding_array = await global_config["embedding_func"](
                            batch_texts
                        )
                        all_embeddings.extend(embedding_array)
                    except Exception as e:
                        logger.error(f"Failed to process batch: {e}")
                        for _ in batch_texts:
                            all_embeddings.append([0.0] * 1536)

            # Map embeddings back to node ids
            node_embeddings = {}
            for i, node in enumerate(all_nodes):
                if i < len(all_embeddings):
                    emb = all_embeddings[i]
                    node_embeddings[node] = (
                        emb.tolist() if hasattr(emb, "tolist") else emb
                    )
                else:
                    logger.warning(f"Missing embedding for node {node}")

            # Persist to disk
            try:
                write_json(node_embeddings, self._node_embeddings_file)
                logger.info(
                    f"Persisted node embeddings to file: {self._node_embeddings_file} ({len(node_embeddings)} nodes)"
                )
            except Exception as e:
                logger.error(
                    f"Failed to persist node embeddings to {self._node_embeddings_file}: {e}"
                )

            return node_embeddings

        except Exception as e:
            logger.error(f"Failed to compute node embeddings: {e}")
            return {}

    async def get_cached_query_embedding(self, query: str, global_config: dict) -> list:
        """Get cached query embedding to avoid repeated computation"""
        if query not in self._query_embeddings_cache:
            logger.info(f"Computing new query embedding for: {query[:50]}...")
            if "embedding_func" in global_config and global_config["embedding_func"]:
                try:
                    query_embedding_array = await global_config["embedding_func"]([query])
                    self._query_embeddings_cache[query] = query_embedding_array[0].tolist()
                    logger.info(f"Successfully cached query embedding for: {query[:50]}...")
                    
                    # Limit cache size to avoid excessive memory usage
                    max_cache_size = 100  # Cache at most 100 query embeddings
                    if len(self._query_embeddings_cache) > max_cache_size:
                        # Remove oldest cache entry
                        oldest_query = next(iter(self._query_embeddings_cache))
                        del self._query_embeddings_cache[oldest_query]
                        logger.info(f"Removed oldest query embedding from cache to maintain size limit")
                        
                except Exception as e:
                    logger.error(f"Failed to compute query embedding: {e}")
                    return None
            else:
                logger.warning("No embedding function available in global config")
                return None
        else:
            logger.info(f"Using cached query embedding for: {query[:50]}...")
        
        return self._query_embeddings_cache[query]

    def get_node_embeddings_file_path(self) -> str:
        """Return the path of persisted node embeddings file for external extraction."""
        return self._node_embeddings_file

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        return self._graph.degree(node_id)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return self._graph.degree(src_id) + self._graph.degree(tgt_id)

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        if self._graph.has_edge(source_node_id, target_node_id):
            return self._graph.edges[source_node_id, target_node_id]
        return None

    async def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None
    async def get_node_in_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.in_edges(source_node_id))
        return None
    async def get_node_out_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.out_edges(source_node_id))
        return None
    
    async def get_pagerank(self,source_node_id:str):
        pagerank_list=nx.pagerank(self._graph)
        if source_node_id in pagerank_list:
            return pagerank_list[source_node_id]
        else:
            print("pagerank failed")

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        # Filter out None values to avoid GraphML writer issues
        filtered_node_data = {k: v for k, v in node_data.items() if v is not None}
        self._graph.add_node(node_id, **filtered_node_data)
        # Clear cache as graph has been updated
        self._cached_nx_graph = None
        self._node_embeddings_cache = {}
        self._invalidate_node_embeddings_file()
        self._query_embeddings_cache = {}

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        # Filter out None values to avoid GraphML writer issues
        filtered_edge_data = {k: v for k, v in edge_data.items() if v is not None}
        
        # Check if both nodes exist before adding edge
        if not self._graph.has_node(source_node_id):
            logger.warning(f"Source node {source_node_id} does not exist, skipping edge creation")
            return
        if not self._graph.has_node(target_node_id):
            logger.warning(f"Target node {target_node_id} does not exist, skipping edge creation")
            return
            
        self._graph.add_edge(source_node_id, target_node_id, **filtered_edge_data)
        # Clear cache as graph has been updated
        self._cached_nx_graph = None
        self._node_embeddings_cache = {}
        self._invalidate_node_embeddings_file()
        self._query_embeddings_cache = {}

    async def delete_node(self, node_id: str):
        """
        Delete a node from the graph based on the specified node_id.

        :param node_id: The node_id to delete
        """
        if self._graph.has_node(node_id):
            self._graph.remove_node(node_id)
            logger.info(f"Node {node_id} deleted from the graph.")
            # Clear cache as graph has been updated
            self._cached_nx_graph = None
            self._node_embeddings_cache = {}
            self._invalidate_node_embeddings_file()
        else:
            logger.warning(f"Node {node_id} not found in the graph for deletion.")

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    # @TODO: NOT USED
    async def _node2vec_embed(self):
        from graspologic import embed

        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids
    
    async def edges(self):
        return self._graph.edges()
    async def nodes(self):
        return self._graph.nodes()
    
    async def get_all_nodes(self):
        """Get all nodes with their data"""
        return dict(self._graph.nodes(data=True))
    
    async def remove_duplicate_nodes(self):
        """Remove duplicate nodes that might have been created accidentally"""
        nodes_data = await self.get_all_nodes()
        seen_nodes = set()
        duplicates = []
        
        for node_id, node_data in nodes_data.items():
            if node_id in seen_nodes:
                duplicates.append(node_id)
            else:
                seen_nodes.add(node_id)
        
        if duplicates:
            logger.warning(f"Found {len(duplicates)} duplicate nodes: {duplicates}")
            for node_id in duplicates:
                if self._graph.has_node(node_id):
                    self._graph.remove_node(node_id)
                    logger.info(f"Removed duplicate node: {node_id}")
            
            # Clear cache as graph has been updated
            self._cached_nx_graph = None
            self._node_embeddings_cache = {}
            self._invalidate_node_embeddings_file()
            self._query_embeddings_cache = {}
        
        return len(duplicates)
    
    async def get_graph_stats(self):
        """Get statistics about the graph"""
        nodes_data = await self.get_all_nodes()
        edges = await self.edges()
        edges_list = list(edges) if edges else []
        
        # Count nodes by type
        node_types = {}
        for node_id, node_data in nodes_data.items():
            node_type = node_data.get('entity_type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        stats = {
            'total_nodes': len(nodes_data),
            'total_edges': len(edges_list),
            'node_types': node_types,
            'sample_nodes': list(nodes_data.keys())[:5] if nodes_data else [],
            'sample_edges': edges_list[:5] if edges_list else []
        }
        
        return stats