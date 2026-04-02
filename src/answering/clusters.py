"""
Flow diffusion cluster operations for knowledge graph traversal.

This module provides functions to find clusters using flow diffusion,
convert subgraphs to JSON format and summarize clusters using LLM.
"""

import asyncio
import tiktoken
import networkx as nx
from ..base import (
    BaseGraphStorage,
    QueryParam,
)
from ..retrievers import QueryAwareWeightedFlowDiffusion
from ..utils import (
    logger,
    truncate_list_by_token_size,
)


async def get_embeddings_for_flow_diffusion(
    graph: nx.Graph,
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
    query_param: QueryParam = None,
) -> tuple[dict, list]:
    """
    Get embeddings for nodes and query for query-aware flow diffusion.

    Returns:
    --------
    tuple
        (node_embeddings, subquery_embedding)
    """
    node_embeddings = {}
    subquery_embedding = None

    # Check if query-aware flow diffusion is enabled
    if query_param and not query_param.enable_query_aware_flow_diffusion:
        logger.info("Query-aware flow diffusion is disabled, skipping embedding calculation")
        return {}, None

    # Try to get embeddings from global config if available
    if "embedding_func" in global_config and global_config["embedding_func"]:
        try:
            # Prioritize using cached node embeddings
            if hasattr(knowledge_graph_inst, 'get_cached_node_embeddings'):
                logger.info("Attempting to use cached node embeddings...")
                node_embeddings = await knowledge_graph_inst.get_cached_node_embeddings(global_config)
                if node_embeddings:
                    logger.info(f"Successfully using cached embeddings for {len(node_embeddings)} nodes")
                else:
                    logger.info("No cached embeddings found, will compute new ones")
            else:
                logger.info("Storage class does not support cached embeddings, computing new ones...")
                # Get embeddings for all nodes in the graph with batching
                all_nodes = list(graph.nodes())
                if all_nodes:
                    # Get batch size and token limit from config
                    batch_size = global_config.get("embedding_batch_num", 32)
                    max_tokens_per_request = global_config.get("max_embed_tokens", 8192)

                    # Prepare node texts
                    node_texts = []
                    for node in all_nodes:
                        node_info = await knowledge_graph_inst.get_node(node)
                        if node_info and "description" in node_info:
                            node_texts.append(f"{node} {node_info['description']}")
                        else:
                            node_texts.append(node)

                    # Process in batches to avoid token limit
                    logger.info(f"Computing embeddings for {len(node_texts)} node texts in batches of {batch_size}...")

                    # Import tiktoken for token counting
                    tiktoken_model = global_config.get("tiktoken_model_name", "gpt-4o-mini")
                    try:
                        encoding = tiktoken.encoding_for_model(tiktoken_model)
                    except:
                        encoding = tiktoken.get_encoding("cl100k_base")  # fallback

                    # Process batches
                    all_embeddings = []
                    for i in range(0, len(node_texts), batch_size):
                        batch_texts = node_texts[i:i + batch_size]
                        batch_nodes = all_nodes[i:i + batch_size]

                        # Check token count for this batch
                        total_tokens = sum(len(encoding.encode(text)) for text in batch_texts)

                        if total_tokens > max_tokens_per_request:
                            logger.warning(f"Batch {i//batch_size + 1} exceeds token limit ({total_tokens} > {max_tokens_per_request}), reducing batch size...")
                            # Process this batch with smaller chunks
                            sub_batch_size = max(1, batch_size // 2)
                            for j in range(0, len(batch_texts), sub_batch_size):
                                sub_batch_texts = batch_texts[j:j + sub_batch_size]
                                sub_batch_nodes = batch_nodes[j:j + sub_batch_size]

                                # Check sub-batch token count
                                sub_total_tokens = sum(len(encoding.encode(text)) for text in sub_batch_texts)
                                if sub_total_tokens > max_tokens_per_request:
                                    logger.warning(f"Sub-batch still exceeds token limit ({sub_total_tokens} > {max_tokens_per_request}), processing one by one...")
                                    # Process one by one
                                    for k, (text, node) in enumerate(zip(sub_batch_texts, sub_batch_nodes)):
                                        try:
                                            embedding_array = await global_config["embedding_func"]([text])
                                            all_embeddings.append(embedding_array[0])
                                            logger.debug(f"Processed node {k+1}/{len(sub_batch_texts)} in sub-batch")
                                        except Exception as e:
                                            logger.error(f"Failed to process node {node}: {e}")
                                            # Add zero embedding as fallback
                                            embedding_dim = 1536  # default dimension
                                            all_embeddings.append([0.0] * embedding_dim)
                                else:
                                    try:
                                        embedding_array = await global_config["embedding_func"](sub_batch_texts)
                                        all_embeddings.extend(embedding_array)
                                        logger.debug(f"Processed sub-batch {j//sub_batch_size + 1} with {len(sub_batch_texts)} nodes")
                                    except Exception as e:
                                        logger.error(f"Failed to process sub-batch: {e}")
                                        # Add zero embeddings as fallback
                                        embedding_dim = 1536  # default dimension
                                        for _ in sub_batch_texts:
                                            all_embeddings.append([0.0] * embedding_dim)
                        else:
                            try:
                                embedding_array = await global_config["embedding_func"](batch_texts)
                                all_embeddings.extend(embedding_array)
                                logger.debug(f"Processed batch {i//batch_size + 1} with {len(batch_texts)} nodes")
                            except Exception as e:
                                logger.error(f"Failed to process batch: {e}")
                                # Add zero embeddings as fallback
                                embedding_dim = 1536  # default dimension
                                for _ in batch_texts:
                                    all_embeddings.append([0.0] * embedding_dim)

                    # Store embeddings
                    for i, node in enumerate(all_nodes):
                        if i < len(all_embeddings):
                            node_embeddings[node] = all_embeddings[i].tolist()
                        else:
                            logger.warning(f"Missing embedding for node {node}")

                    logger.info(f"Computed embeddings for {len(node_embeddings)} nodes")

            # Get query embedding
            if query:
                # Prioritize using cached query embedding
                if hasattr(knowledge_graph_inst, 'get_cached_query_embedding'):
                    logger.info("Attempting to use cached query embedding...")
                    subquery_embedding = await knowledge_graph_inst.get_cached_query_embedding(query, global_config)
                    if subquery_embedding is None:
                        logger.warning("Failed to get cached query embedding, computing new one...")
                        query_embedding_array = await global_config["embedding_func"]([query])
                        subquery_embedding = query_embedding_array[0].tolist()
                        logger.info("Query embedding computed successfully")
                else:
                    logger.info("Storage class does not support cached query embeddings, computing new one...")
                    query_embedding_array = await global_config["embedding_func"]([query])
                    subquery_embedding = query_embedding_array[0].tolist()
                    logger.info("Query embedding computed successfully")

        except Exception as e:
            logger.warning(f"Failed to get embeddings for query-aware flow diffusion: {e}")
            # Fallback to non-query-aware mode
            node_embeddings = {}
            subquery_embedding = None
    else:
        logger.warning("No embedding function available in global config")

    return node_embeddings, subquery_embedding


def convert_subgraph_to_json(G: nx.Graph, cluster_nodes: list, diffused_nodes: dict, source_node: str) -> dict:
    """
    Convert a subgraph to JSON format with nodes and edges information.

    Parameters:
    -----------
    G : nx.Graph
        The original graph
    cluster_nodes : list
        List of nodes in the cluster
    diffused_nodes : dict
        Dictionary mapping nodes to their flow values
    source_node : str
        The source node for this cluster

    Returns:
    --------
    dict
        JSON representation of the subgraph
    """
    # Create subgraph from cluster nodes
    support_nodes = set(cluster_nodes)
    subgraph = G.subgraph(support_nodes)

    # Convert nodes to JSON format
    nodes_json = []
    # Build a local index map so node "id" matches the CSV-style index
    node_index_map = {node: idx for idx, node in enumerate(cluster_nodes)}
    for node in cluster_nodes:
        node_attrs = G.nodes[node] if node in G.nodes else {}
        node_degree = subgraph.degree(node) if node in subgraph else 0
        nodes_json.append({
            "id": node_index_map[node],
            "entity": node,
            "entity_type": node_attrs.get("entity_type", "UNKNOWN"),
            "description": node_attrs.get("description", "UNKNOWN"),
            "rank": node_degree,
        })

    # Convert edges to JSON format
    edges_json = []
    for u, v, data in subgraph.edges(data=True):
        edges_json.append({
            "source": u,
            "target": v,
            "source_id": node_index_map.get(u),
            "target_id": node_index_map.get(v),
            "weight": data.get('weight', 1.0)
        })

    return {
        "source_node": source_node,
        "cluster_size": len(cluster_nodes),
        "max_flow_value": max(diffused_nodes.values()) if diffused_nodes else 0.0,
        "nodes": nodes_json,
        "edges": edges_json,
        "total_edges": len(edges_json)
    }


async def find_flow_diffusion_clusters_and_summarize(
    node_datas: list[dict],
    query: str,
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """
    Apply flow diffusion to find clusters and summarize them using LLM.

    Parameters:
    -----------
    node_datas : list[dict]
        List of node data dictionaries
    query : str
        The original query that provides context for the relationship analysis
    query_param : QueryParam
        Query parameters (includes flow diffusion configuration)
    knowledge_graph_inst : BaseGraphStorage
        Knowledge graph storage instance
    global_config : dict
        Global configuration

    Returns:
    --------
    list
        If return_raw_clusters=True: List of cluster JSON objects with subgraph data
        If return_raw_clusters=False: List of summarized cluster relationships
    """
    # Use cached NetworkX graph to avoid repeated construction
    if hasattr(knowledge_graph_inst, 'get_cached_nx_graph'):
        logger.info("Attempting to use cached NetworkX graph...")
        G = await knowledge_graph_inst.get_cached_nx_graph()
        logger.info(f"Successfully obtained NetworkX graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    else:
        logger.info("Storage class does not support cached graphs, building new one...")
        # Compatibility handling: if no cache method, use original approach
        G = nx.Graph()
        edges = await knowledge_graph_inst.edges()
        nodes = await knowledge_graph_inst.nodes()

        # Add edges with weights
        for u, v in edges:
            edge_data = await knowledge_graph_inst.get_edge(u, v)
            if edge_data and 'weight' in edge_data:
                G.add_edge(u, v, weight=edge_data['weight'])
            else:
                G.add_edge(u, v, weight=1.0)  # Default weight: 1.0 if not specified

        G.add_nodes_from(nodes)
        logger.info(f"Built new NetworkX graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Get source nodes from node_datas (limit to configured maximum)
    source_nodes = [dp["entity_name"] for dp in node_datas[:query_param.max_source_nodes]]

    # Apply flow diffusion from each source node independently
    all_clusters = []
    use_llm_func = global_config["llm_model_func"]

    logger.info(f"Starting flow diffusion from {len(source_nodes)} source nodes independently")

    # Pre-compute node embeddings and query embeddings to avoid repeated computation in loops
    logger.info("Pre-computing embeddings for flow diffusion...")
    node_embeddings, subquery_embedding = await get_embeddings_for_flow_diffusion(
        G, query, knowledge_graph_inst, global_config, query_param
    )
    logger.info(f"Pre-computed embeddings: {len(node_embeddings)} nodes, query embedding: {'Yes' if subquery_embedding else 'No'}")

    # Initialize cluster processing based on configuration
    all_clusters = []

    if query_param.use_batch_cluster_summarization:
        logger.info("Using batch cluster summarization mode (more efficient)")
    else:
        logger.info("Using individual cluster summarization mode (original approach)")

    if query_param.use_batch_cluster_summarization:
        # Collect all clusters first for batch processing
        clusters_to_summarize = []
        raw_clusters = []

        # Run flow diffusion from each source node
        for source_node in source_nodes:

            if not G.has_node(source_node):
                continue

            # Get source node information
            source_node_info = await knowledge_graph_inst.get_node(source_node)
            if not source_node_info:
                source_node_info = {"entity_type": "UNKNOWN", "description": "No description available"}

            # Calculate confidence based on source node's relevance to the query
            confidence = 0.7  # Default confidence

            # Apply flow diffusion from this source node
            wfd = QueryAwareWeightedFlowDiffusion(
                G, source_node, source_node, confidence,
                node_embeddings=node_embeddings,
                subquery_embedding=subquery_embedding,
                weight_func=query_param.weight_func
            )
            wfd.initialize(alpha=query_param.alpha)
            diffused_nodes = wfd.flow_diffusion()

            if len(diffused_nodes) > 1:  # Multi-node cluster
                # Get cluster nodes and their flow values
                cluster_nodes = list(diffused_nodes.keys())
                cluster_flow_values = list(diffused_nodes.values())

                # Only process if cluster has significant flow (using configured threshold)
                if cluster_flow_values:
                    max_flow = max(cluster_flow_values)
                    if max_flow < query_param.min_flow_threshold:
                        continue
                else:
                    continue  # Skip if no flow values

                # Get node information for the cluster
                cluster_node_data = []
                for node in cluster_nodes:
                    node_info = await knowledge_graph_inst.get_node(node)
                    if node_info:
                        cluster_node_data.append({
                            'name': node,
                            'entity_type': node_info.get('entity_type', 'UNKNOWN'),
                            'description': node_info.get('description', 'UNKNOWN'),
                            'flow_value': diffused_nodes.get(node, 0.0)
                        })

                # Sort by flow value
                cluster_node_data.sort(key=lambda x: x['flow_value'], reverse=True)

                if query_param.return_raw_clusters:
                    # Convert subgraph to JSON format
                    cluster_json = convert_subgraph_to_json(G, cluster_nodes, diffused_nodes, source_node)
                    # Add node details to the JSON
                    cluster_json["node_details"] = cluster_node_data
                    # Enrich source_node with type + description
                    for nd in cluster_node_data:
                        if nd["name"] == source_node:
                            cluster_json["source_node"] = {
                                "entity": source_node,
                                "entity_type": nd.get("entity_type", "UNKNOWN"),
                                "description": nd.get("description", "UNKNOWN")
                            }
                            break
                    # Remove edges before appending
                    cluster_json.pop("edges", None)
                    cluster_json.pop("total_edges", None)
                    cluster_json.pop("node_details", None)
                    raw_clusters.append(cluster_json)
                else:
                    # Collect cluster data for batch processing
                    clusters_to_summarize.append((cluster_node_data, source_node))

            else:  # --- NEW FALLBACK: single-node cluster ---
                node_info = await knowledge_graph_inst.get_node(source_node)
                cluster_node_data = [{
                    'name': source_node,
                    'entity_type': node_info.get('entity_type', 'UNKNOWN') if node_info else 'UNKNOWN',
                    'description': node_info.get('description', 'UNKNOWN') if node_info else 'UNKNOWN',
                    'flow_value': diffused_nodes.get(source_node, 0.0) if diffused_nodes else 0.0
                }]

                if query_param.return_raw_clusters:
                    cluster_json = {
                        "source_node": {
                            "entity": source_node,
                            "entity_type": cluster_node_data[0]["entity_type"],
                            "description": cluster_node_data[0]["description"]
                        },
                        "cluster_size": 1,
                        "max_flow_value": 0.0,
                        "nodes": [{
                            "id": 0,
                            "entity": source_node,
                            "entity_type": cluster_node_data[0]["entity_type"],
                            "description": cluster_node_data[0]["description"],
                            "rank": 0
                        }],
                        "edges": [],
                        "total_edges": 0,
                        "node_details": cluster_node_data
                    }
                    # Remove edges before appending
                    cluster_json.pop("edges", None)
                    cluster_json.pop("total_edges", None)
                    cluster_json.pop("node_details", None)
                    raw_clusters.append(cluster_json)
                else:
                    clusters_to_summarize.append((cluster_node_data, source_node))

        # Process clusters based on return type
        if query_param.return_raw_clusters:
            all_clusters = raw_clusters
        else:
            # Batch process clusters in chunks to avoid token limits
            if clusters_to_summarize:
                all_clusters = []
                batch_size = query_param.batch_cluster_size

                # Process clusters in batches
                for i in range(0, len(clusters_to_summarize), batch_size):
                    batch_clusters = clusters_to_summarize[i:i + batch_size]
                    logger.info(f"Processing batch {i//batch_size + 1}/{(len(clusters_to_summarize) + batch_size - 1)//batch_size} with {len(batch_clusters)} clusters")

                    cluster_summaries = await summarize_clusters_batch_with_llm(
                        batch_clusters, use_llm_func, global_config
                    )
                    all_clusters.extend([summary for summary in cluster_summaries if summary])
            else:
                all_clusters = []

    else:
        # Process clusters individually (original approach)
        for source_node in source_nodes:

            if not G.has_node(source_node):
                continue

            if G.nodes[source_node].get("entity_type", "").lower() == "complete_table":
                continue

            # Get source node information
            source_node_info = await knowledge_graph_inst.get_node(source_node)
            if not source_node_info:
                source_node_info = {"entity_type": "UNKNOWN", "description": "No description available"}

            # Calculate confidence based on source node's relevance to the query
            confidence = 0.7  # Default confidence

            # Apply flow diffusion from this source node
            wfd = QueryAwareWeightedFlowDiffusion(
                G, source_node, source_node, confidence,
                node_embeddings=node_embeddings,
                subquery_embedding=subquery_embedding,
                weight_func=query_param.weight_func
            )
            wfd.initialize(alpha=query_param.alpha)
            diffused_nodes = wfd.flow_diffusion()

            if len(diffused_nodes) > 1:  # Only consider clusters with multiple nodes
                # Get cluster nodes and their flow values
                cluster_nodes = list(diffused_nodes.keys())
                cluster_flow_values = list(diffused_nodes.values())

                # Only process if cluster has significant flow (using configured threshold)
                if cluster_flow_values:
                    max_flow = max(cluster_flow_values)
                    if max_flow < query_param.min_flow_threshold:
                        continue
                else:
                    continue  # Skip if no flow values

                # Get node information for the cluster
                cluster_node_data = []
                for node in cluster_nodes:
                    node_info = await knowledge_graph_inst.get_node(node)
                    if node_info:
                        cluster_node_data.append({
                            'name': node,
                            'entity_type': node_info.get('entity_type', 'UNKNOWN'),
                            'description': node_info.get('description', 'UNKNOWN'),
                            'flow_value': diffused_nodes.get(node, 0.0)
                        })

                # Sort by flow value
                cluster_node_data.sort(key=lambda x: x['flow_value'], reverse=True)

                # Check if we should return raw cluster data instead of LLM summaries
                if query_param.return_raw_clusters:
                    # Convert subgraph to JSON format
                    cluster_json = convert_subgraph_to_json(G, cluster_nodes, diffused_nodes, source_node)
                    # Add node details to the JSON
                    cluster_json["node_details"] = cluster_node_data
                    all_clusters.append(cluster_json)
                else:
                    # Create cluster summary using LLM (individual processing)
                    cluster_summary = await summarize_cluster_with_llm(
                        cluster_node_data, source_node, use_llm_func, global_config
                    )

                    if cluster_summary:
                        all_clusters.append(cluster_summary)

    logger.info(f"Flow diffusion completed, {len(all_clusters)} clusters found")

    # Only apply token constraints if we're returning LLM summaries
    if not query_param.return_raw_clusters:
        # Limit the number of clusters based on token constraints
        original_cluster_count = len(all_clusters)
        all_clusters = truncate_list_by_token_size(
            all_clusters,
            key=lambda x: x,
            max_token_size=query_param.max_token_for_local_context,
        )

        if original_cluster_count != len(all_clusters):
            logger.info(f"Clusters truncated from {original_cluster_count} to {len(all_clusters)} due to token limit")

    return all_clusters


async def summarize_clusters_batch_with_llm(
    clusters_data: list[tuple[list[dict], str]],
    use_llm_func: callable,
    global_config: dict = None,
) -> list[str]:
    """
    Summarize multiple clusters using a single LLM call.

    Parameters:
    -----------
    clusters_data : list[tuple[list[dict], str]]
        List of tuples containing (cluster_node_data, source_node) for each cluster
    use_llm_func : callable
        LLM function to use for summarization
    global_config : dict
        Global configuration (optional)

    Returns:
    --------
    list[str]
        List of summarized cluster relationships
    """
    if not clusters_data:
        return []

    # Filter out clusters with less than 2 nodes
    valid_clusters = []
    for cluster_node_data, source_node in clusters_data:
        if len(cluster_node_data) >= 2:
            valid_clusters.append((cluster_node_data, source_node))

    if not valid_clusters:
        return []

    # Create batch prompt for all clusters
    clusters_info = []
    for i, (cluster_node_data, source_node) in enumerate(valid_clusters, 1):
        nodes_info = []
        for node_data in cluster_node_data:
            nodes_info.append(
                f"- {node_data['name']} ({node_data['entity_type']}): {node_data['description']} "
                f"(flow strength: {node_data['flow_value']:.3f})"
            )

        nodes_text = "\n".join(nodes_info)
        clusters_info.append(f"""Cluster {i}:
Source Node: {source_node}
Cluster Nodes:
{nodes_text}""")

    all_clusters_text = "\n\n".join(clusters_info)

    prompt = f"""Please analyze the following clusters of entities and their relationships, then provide a concise summary for each cluster.

Clusters Information:
{all_clusters_text}

For each cluster, please provide a summary that explains:
1. How these entities are related to each other
2. What concept or theme this cluster represents
3. The significance of the connections between these entities
4. How the source node connects to the other entities in this cluster

Please format your response as follows:
Cluster 1: [summary for cluster 1]
Cluster 2: [summary for cluster 2]
...
Cluster N: [summary for cluster N]

Keep each summary concise but informative, focusing on the relationships and thematic connections."""

    # Use configuration-based max_tokens if available, otherwise use a reasonable default
    max_tokens = 2000  # Increased for batch processing
    if global_config and "entity_summary_to_max_tokens" in global_config:
        max_tokens = min(global_config["entity_summary_to_max_tokens"] * len(valid_clusters), 16384)

    logger.info(f"Processing batch of {len(valid_clusters)} clusters with max_tokens={max_tokens}")

    try:
        batch_summary = await use_llm_func(prompt, max_tokens=max_tokens)
        batch_summary = batch_summary.strip()

        # Parse the response to extract individual cluster summaries
        summaries = []
        lines = batch_summary.split('\n')
        current_summary = ""

        for line in lines:
            line = line.strip()
            if line.startswith('Cluster ') and ':' in line:
                # Save previous summary if exists
                if current_summary:
                    summaries.append(current_summary.strip())
                # Start new summary
                current_summary = line.split(':', 1)[1].strip()
            elif line and current_summary:
                current_summary += " " + line

        # Add the last summary
        if current_summary:
            summaries.append(current_summary.strip())

        # Ensure we have the right number of summaries
        while len(summaries) < len(valid_clusters):
            summaries.append("Cluster summary not available")

        return summaries[:len(valid_clusters)]

    except Exception as e:
        logger.error(f"Error summarizing clusters batch: {e}")
        # Fallback: create individual summaries for each cluster
        fallback_summaries = []
        for cluster_node_data, source_node in valid_clusters:
            if len(cluster_node_data) > 0:
                top_nodes = cluster_node_data[:3]  # Get top 3 nodes by flow value
                node_names = [node['name'] for node in top_nodes]
                if len(cluster_node_data) > 1:
                    fallback_summary = f"Cluster centered around {source_node} connecting to {', '.join(node_names)} and {len(cluster_node_data)-1} other entities with flow values ranging from {cluster_node_data[0]['flow_value']:.3f} to {cluster_node_data[-1]['flow_value']:.3f}"
                else:
                    fallback_summary = f"Cluster centered around {source_node} connecting to {', '.join(node_names)}"
                fallback_summaries.append(fallback_summary)
            else:
                fallback_summaries.append(f"Cluster connecting {source_node} with related entities")

        return fallback_summaries


async def summarize_cluster_with_llm(
    cluster_node_data: list[dict],
    source_node: str,
    use_llm_func: callable,
    global_config: dict = None,
) -> str:
    """
    Summarize a cluster of nodes using LLM.

    Parameters:
    -----------
    cluster_node_data : list[dict]
        List of node data in the cluster
    source_node : str
        Source node name
    use_llm_func : callable
        LLM function to use for summarization
    global_config : dict
        Global configuration (optional)

    Returns:
    --------
    str
        Summarized cluster relationship
    """
    if len(cluster_node_data) < 2:
        return None

    # Create prompt for cluster summarization
    nodes_info = []
    for node_data in cluster_node_data:
        nodes_info.append(
            f"- {node_data['name']} ({node_data['entity_type']}): {node_data['description']} "
            f"(flow strength: {node_data['flow_value']:.3f})"
        )

    nodes_text = "\n".join(nodes_info)

    prompt = f"""Please analyze the following cluster of entities and their relationships, then provide a concise summary of how they are connected and what this cluster represents.

Cluster Information:
Source Node: {source_node}
Cluster Nodes:
{nodes_text}

Please provide a summary that explains:
1. How these entities are related to each other
2. What concept or theme this cluster represents
3. The significance of the connections between these entities
4. How the source node connects to the other entities in this cluster

Keep the summary concise but informative, focusing on the relationships and thematic connections."""

    # Use configuration-based max_tokens if available, otherwise use a reasonable default
    max_tokens = 1000  # Default value
    if global_config and "entity_summary_to_max_tokens" in global_config:
        max_tokens = global_config["entity_summary_to_max_tokens"]

    try:
        summary = await use_llm_func(prompt, max_tokens=max_tokens)
        return summary.strip()
    except Exception as e:
        logger.error(f"Error summarizing cluster: {e}")
        # Fallback to more informative description
        if len(cluster_node_data) > 0:
            top_nodes = cluster_node_data[:3]  # Get top 3 nodes by flow value
            node_names = [node['name'] for node in top_nodes]
            if len(cluster_node_data) > 1:
                return f"Cluster centered around {source_node} connecting to {', '.join(node_names)} and {len(cluster_node_data)-1} other entities with flow values ranging from {cluster_node_data[0]['flow_value']:.3f} to {cluster_node_data[-1]['flow_value']:.3f}"
            else:
                return f"Cluster centered around {source_node} connecting to {', '.join(node_names)}"
        else:
            return f"Cluster connecting {source_node} with related entities"
