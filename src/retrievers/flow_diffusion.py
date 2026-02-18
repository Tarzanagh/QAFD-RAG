"""Query-Aware Weighted Flow Diffusion retriever.

This module implements the Query-Aware Weighted Flow Diffusion algorithm
for knowledge graph traversal. The algorithm uses flow diffusion to find
relevant nodes and paths, with optional query-aware edge weighting based
on node embeddings and query embeddings.
"""

import random
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

import networkx as nx

from ..utils import logger
from .base import BaseRetriever, RetrieverResult


class FlowDiffusionRetriever(BaseRetriever):
    """Query-Aware Weighted Flow Diffusion algorithm for graph retrieval.

    This retriever uses flow diffusion to find relevant nodes and paths
    in a knowledge graph, with optional query-aware edge weighting.

    The algorithm works by:
    1. Initializing mass at the source node
    2. Iteratively pushing excess mass to neighbors
    3. Using flow values to find optimal paths

    Attributes:
        graph: NetworkX graph to traverse
        source: Source node ID
        target: Target node ID
        confidence: Confidence level (0.0-1.0)
        epsilon: Convergence threshold
        node_embeddings: Optional dict of node embeddings
        subquery_embedding: Optional query embedding vector
        weight_func: Weight function type ("multiply", "add", or None)
    """

    def __init__(
        self,
        graph,
        source_node: str,
        target_node: str,
        confidence: float = 0.5,
        epsilon: float = 0.05,
        node_embeddings: Optional[Dict] = None,
        subquery_embedding: Optional[List[float]] = None,
        weight_func: Optional[str] = None
    ):
        """Initialize the Query-Aware Weighted Flow Diffusion algorithm.

        Args:
            graph: NetworkX graph to traverse
            source_node: Starting node for pathfinding
            target_node: Destination node for pathfinding
            confidence: Confidence level (0.0-1.0, default 0.5)
            epsilon: Convergence threshold (default 0.05)
            node_embeddings: Optional dict mapping node IDs to embeddings
            subquery_embedding: Optional query embedding vector
            weight_func: Optional weight function ("multiply", "add", or None)
        """
        super().__init__(graph)
        self.source = source_node
        self.target = target_node
        self.confidence = max(0.0, min(1.0, confidence))
        self.epsilon = epsilon
        self.mass = defaultdict(float)
        self.x = defaultdict(float)
        self.sink_capacity = defaultdict(float)

        # Query-aware components
        self.node_embeddings = node_embeddings or {}
        # Use actual embedding dimension or default to 1536
        embedding_dim = len(subquery_embedding) if subquery_embedding else 1536
        self.subquery_embedding = subquery_embedding or [0.0] * embedding_dim
        self.weight_func = weight_func
        self.edge_weights_cache = {}

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Fast cosine similarity calculation.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score normalized to [0, 1]
        """
        if not vec1 or not vec2:
            return 0.0

        try:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            mag1 = sum(a * a for a in vec1) ** 0.5
            mag2 = sum(b * b for b in vec2) ** 0.5

            if mag1 == 0 or mag2 == 0:
                return 0.0

            similarity = dot_product / (mag1 * mag2)
            return max(0.0, (similarity + 1.0) / 2.0)  # Normalize to [0, 1]
        except:
            return 0.0

    def get_edge_weight(self, node1: str, node2: str) -> float:
        """Get smart edge weight with optional query-aware weighting.

        This is the key query-aware modification that adjusts edge weights
        based on node similarity to the query embedding.

        Args:
            node1: Source node ID
            node2: Target node ID

        Returns:
            Edge weight (possibly adjusted for query relevance)
        """
        # Use cache if available
        cache_key = (node1, node2)
        if cache_key in self.edge_weights_cache:
            return self.edge_weights_cache[cache_key]

        # Get original edge weight
        edge_data = self.graph[node1][node2]
        original_weight = edge_data.get('weight', 0.0)

        # If no embeddings available, return original weight
        if not self.node_embeddings or not self.subquery_embedding:
            self.edge_weights_cache[cache_key] = original_weight
            return original_weight

        # If original weight is 0, keep it 0
        if original_weight <= 0:
            self.edge_weights_cache[cache_key] = 0.0
            return 0.0

        # Get node embeddings
        node1_emb = self.node_embeddings.get(node1, [0.0] * len(self.subquery_embedding))
        node2_emb = self.node_embeddings.get(node2, [0.0] * len(self.subquery_embedding))

        # Calculate query similarity factor
        node1_query_sim = self.cosine_similarity(node1_emb, self.subquery_embedding)
        node2_query_sim = self.cosine_similarity(node2_emb, self.subquery_embedding)

        if self.weight_func == "multiply":
            smart_weight = original_weight * node1_query_sim * node2_query_sim
        elif self.weight_func == "add":
            smart_weight = (original_weight + node1_query_sim + node2_query_sim) / 3.0
        else:
            # Apply modest boost: original_weight * (1 + small_boost)
            query_factor = (node1_query_sim + node2_query_sim) / 2.0
            smart_weight = original_weight * (1.0 + query_factor * 0.5)

        # Cache and return
        self.edge_weights_cache[cache_key] = smart_weight
        return smart_weight

    def initialize(self, alpha: float = 50, use_node_degree: bool = True):
        """Initialize the flow diffusion algorithm.

        Sets up sink capacities and initial mass at the source node.

        Args:
            alpha: Mass initialization factor (default 50)
            use_node_degree: Whether to use node degree for sink capacity (default True)
        """
        # Set sink capacity for all nodes
        for node in self.graph.nodes():
            if use_node_degree:
                # Set sink capacity to node degree (number of connections)
                self.sink_capacity[node] = self.graph.degree(node)
            else:
                # Set sink capacity to 1 (original behavior)
                self.sink_capacity[node] = 1

        # Set all masses to 0 initially
        for node in self.graph.nodes():
            self.mass[node] = 0

        # Set source mass - boost based on confidence
        try:
            path = nx.shortest_path(self.graph, self.source, self.target)
            total_sink_on_path = sum(self.sink_capacity[node] for node in path)
            confidence_boost = 1.0 + self.confidence
            self.mass[self.source] = alpha * total_sink_on_path * confidence_boost
        except nx.NetworkXNoPath:
            avg_sink = sum(self.sink_capacity.values()) / len(self.sink_capacity)
            confidence_boost = 1.0 + self.confidence
            self.mass[self.source] = alpha * avg_sink * len(self.graph.nodes()) / 10 * confidence_boost

    def push(self, node: str) -> bool:
        """Push operation - distribute excess mass to neighbors.

        Args:
            node: Node to push mass from

        Returns:
            True if push occurred, False otherwise
        """
        # Get neighbors
        neighbors = list(self.graph.neighbors(node))
        if not neighbors:
            return False

        # Calculate w_i using query-aware edge weights
        w_i = 0
        for neighbor in neighbors:
            weight = self.get_edge_weight(node, neighbor)
            w_i += weight

        if w_i == 0:
            return False

        # Calculate excess mass
        excess = self.mass[node] - self.sink_capacity[node]
        if excess <= 0:
            return False

        # Update x_i
        self.x[node] += excess / w_i

        # Update mass at node i
        self.mass[node] = self.sink_capacity[node]

        # Distribute excess mass to neighbors
        for neighbor in neighbors:
            w_ij = self.get_edge_weight(node, neighbor)
            if w_ij > 0:
                self.mass[neighbor] += excess * w_ij / w_i

        return True

    def flow_diffusion(self, max_iterations: int = 500) -> Dict[str, float]:
        """Run the flow diffusion algorithm.

        Iteratively pushes mass from nodes with excess capacity until
        convergence or max iterations is reached.

        Args:
            max_iterations: Maximum iterations to run (default 500)

        Returns:
            Dictionary of nodes with diffused flow values
        """
        iterations = 0
        pushes = 0

        while iterations < max_iterations:
            iterations += 1

            # Find nodes with excess mass
            excess_nodes = [node for node in self.graph.nodes()
                           if self.mass[node] > self.sink_capacity[node]]

            if not excess_nodes:
                break

            # Pick a node uniformly at random
            node = random.choice(excess_nodes)

            # Apply push operation
            if self.push(node):
                pushes += 1

            # Check for convergence periodically
            if iterations % 100 == 0:
                remaining_excess = sum(max(0, self.mass[node] - self.sink_capacity[node])
                                     for node in self.graph.nodes())
                if remaining_excess < self.epsilon:
                    break

        return {node: val for node, val in self.x.items() if val > 0}

    def find_furthest_reachable_node(
        self,
        diffused_nodes: Dict[str, float]
    ) -> Tuple[Optional[str], Optional[List[str]]]:
        """Find the furthest reachable node in the diffused subgraph.

        Used as a fallback when the target node is not reachable.

        Args:
            diffused_nodes: Dictionary of nodes with diffused flow values

        Returns:
            Tuple of (furthest node ID, path to that node) or (None, None)
        """
        if self.source not in diffused_nodes:
            return None, None

        support_nodes = set(diffused_nodes.keys())
        subgraph = self.graph.subgraph(support_nodes)

        queue = deque([(self.source, 0, [self.source])])
        visited = {self.source}
        furthest_nodes = []
        max_distance = 0
        paths_to_nodes = {self.source: [self.source]}

        while queue:
            current_node, distance, path = queue.popleft()

            if distance > max_distance:
                max_distance = distance
                furthest_nodes = [current_node]
            elif distance == max_distance and current_node not in furthest_nodes:
                furthest_nodes.append(current_node)

            for neighbor in subgraph.neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    new_distance = distance + 1
                    queue.append((neighbor, new_distance, new_path))
                    paths_to_nodes[neighbor] = new_path

        if furthest_nodes:
            chosen_node = max(furthest_nodes, key=lambda n: diffused_nodes.get(n, 0)) if len(furthest_nodes) > 1 else furthest_nodes[0]
            logger.info(f"Furthest reachable node: {chosen_node} at distance {max_distance} from source")
            return chosen_node, paths_to_nodes[chosen_node]

        return None, None

    def calculate_path_score(
        self,
        path: Optional[List[str]],
        confidence_weight: float = 0.7,
        flow_weight: float = 0.3
    ) -> float:
        """Calculate a score for a path based on confidence and flow.

        Args:
            path: List of node IDs representing the path
            confidence_weight: Weight for confidence component (default 0.7)
            flow_weight: Weight for flow component (default 0.3)

        Returns:
            Combined score for the path
        """
        if not path or len(path) < 2:
            return 0.0

        # Calculate flow strength
        flow_values = [self.x[node] for node in path if node in self.x and self.x[node] > 0]
        if not flow_values:
            flow_strength = 0.0
        else:
            avg_flow = sum(flow_values) / len(flow_values)
            flow_strength = avg_flow / (avg_flow + 1.0)

        # Calculate weighted combined score
        combined_score = (confidence_weight * self.confidence) + (flow_weight * flow_strength)
        return combined_score

    def find_path(self) -> Tuple[Optional[List[str]], float]:
        """Find optimal path from source to target using flow diffusion.

        This is the main entry point that orchestrates:
        1. Initialization
        2. Flow diffusion
        3. Path finding in the diffused subgraph

        Returns:
            Tuple of (path as list of node IDs or None, score)
        """
        # Initialize
        self.initialize()

        # Run flow diffusion
        diffused_nodes = self.flow_diffusion()

        # Check if diffusion reached the target
        if self.target not in diffused_nodes:
            logger.info(f"Flow diffusion did not reach target node {self.target}")

            if len(diffused_nodes) > 1:
                furthest_node, furthest_path = self.find_furthest_reachable_node(diffused_nodes)
                if furthest_path:
                    score = self.calculate_path_score(furthest_path)
                    logger.info(f"Path found to furthest reachable node: {' -> '.join(furthest_path)} (score: {score:.3f})")
                    return furthest_path, score

            fallback_path = [self.source] if self.source in diffused_nodes else None
            fallback_score = self.calculate_path_score(fallback_path) if fallback_path else 0.0
            return fallback_path, fallback_score

        # Create subgraph and find path
        support_nodes = set(diffused_nodes.keys())
        subgraph = self.graph.subgraph(support_nodes)

        # Reweight edges based on x values
        weighted_subgraph = nx.Graph()
        for u, v, data in subgraph.edges(data=True):
            new_weight = 1.0 / (self.x[u] + self.x[v] + 1e-10)
            weighted_subgraph.add_edge(u, v, weight=new_weight)

        # Find shortest path
        try:
            path = nx.shortest_path(weighted_subgraph, self.source, self.target, weight='weight')
            score = self.calculate_path_score(path)
            logger.info(f"Path found: {' -> '.join(path)} (score: {score:.3f})")
            return path, score
        except nx.NetworkXNoPath:
            logger.info(f"No path found from {self.source} to {self.target} in diffused subgraph")
            return None, 0.0

    def retrieve(
        self,
        source_node: Optional[str] = None,
        target_node: Optional[str] = None,
        **kwargs
    ) -> RetrieverResult:
        """Retrieve nodes using flow diffusion.

        This method provides a standardized interface for retrieval that
        conforms to the BaseRetriever interface.

        Args:
            source_node: Optional override for source node (uses self.source if None)
            target_node: Optional override for target node (uses self.target if None)
            **kwargs: Additional parameters:
                - alpha: Mass initialization factor (default 50)
                - max_iterations: Max diffusion iterations (default 500)

        Returns:
            RetrieverResult with diffused nodes and optional path
        """
        # Initialize with optional alpha parameter
        alpha = kwargs.get('alpha', 50)
        max_iterations = kwargs.get('max_iterations', 500)

        self.initialize(alpha=alpha)
        diffused_nodes = self.flow_diffusion(max_iterations=max_iterations)

        # Optionally find path if target is set
        path = None
        score = 0.0
        if self.target:
            path, score = self.find_path()

        return RetrieverResult(
            nodes=diffused_nodes,
            path=path,
            score=score,
            metadata={
                'source': self.source,
                'target': self.target,
                'confidence': self.confidence,
                'weight_func': self.weight_func
            }
        )

    def get_node_scores(self) -> Dict[str, float]:
        """Get the flow values for all processed nodes.

        Returns:
            Dictionary mapping node IDs to their flow values
        """
        return dict(self.x)


# Alias for backward compatibility
QueryAwareWeightedFlowDiffusion = FlowDiffusionRetriever
