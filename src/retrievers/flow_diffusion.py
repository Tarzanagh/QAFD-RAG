"""Query-Aware Weighted Flow Diffusion retriever.

This module implements the Query-Aware Weighted Flow Diffusion algorithm
for knowledge graph traversal. The algorithm uses flow diffusion to rank
nodes by relevance, with query-aware edge weighting based on node
embeddings and query embeddings.

Algorithm (push-relabel flow diffusion):
1. Initialize mass at source node(s), warm-start x via lazy random walk
2. Iteratively push excess mass to neighbors via query-aware edges
3. Accumulate importance scores in x with configurable step size
4. Return node flow values as relevance ranking
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional

import networkx as nx

from ..utils import logger
from .base import BaseRetriever, RetrieverResult


class FlowDiffusionRetriever(BaseRetriever):
    """Query-Aware Weighted Flow Diffusion algorithm for graph retrieval.

    Uses push-relabel flow diffusion to rank nodes by relevance.
    Each source node diffuses mass through the graph; nodes accumulate
    importance scores proportional to flow passing through them.

    Attributes:
        graph: NetworkX graph to traverse
        source: Source node ID
        confidence: Confidence level (0.0-1.0)
        epsilon: Convergence threshold
        step_size: Learning rate for flow accumulation
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
        epsilon: float = 0.02,
        node_embeddings: Optional[Dict] = None,
        subquery_embedding: Optional[List[float]] = None,
        weight_func: Optional[str] = None,
        step_size: float = 0.2,
        random_seed: int = 42
    ):
        """Initialize the Query-Aware Weighted Flow Diffusion algorithm.

        Args:
            graph: NetworkX graph to traverse
            source_node: Starting node for flow diffusion
            target_node: Target node (unused in ranking mode)
            confidence: Confidence level (0.0-1.0, default 0.5)
            epsilon: Convergence threshold (default 0.02)
            node_embeddings: Optional dict mapping node IDs to embeddings
            subquery_embedding: Optional query embedding vector
            weight_func: Optional weight function ("multiply", "add", or None)
            step_size: Learning rate for flow accumulation (default 0.2)
            random_seed: Random seed for reproducibility (default 42)
        """
        super().__init__(graph)
        self.source = source_node
        self.target = target_node
        self.confidence = max(0.0, min(1.0, confidence))
        self.epsilon = epsilon
        self.step_size = step_size
        self.mass = defaultdict(float)
        self.x = defaultdict(float)
        self.sink_capacity = defaultdict(float)

        # Query-aware components
        self.node_embeddings = node_embeddings or {}
        embedding_dim = len(subquery_embedding) if subquery_embedding else 1536
        self.subquery_embedding = subquery_embedding or [0.0] * embedding_dim
        self.weight_func = weight_func
        self.edge_weights_cache = {}

        # Deterministic seeding for reproducibility
        random.seed(random_seed)

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Cosine similarity normalized to [0, 1]."""
        if not vec1 or not vec2:
            return 0.0

        try:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            mag1 = sum(a * a for a in vec1) ** 0.5
            mag2 = sum(b * b for b in vec2) ** 0.5

            if mag1 == 0 or mag2 == 0:
                return 0.0

            similarity = dot_product / (mag1 * mag2)
            return max(0.0, (similarity + 1.0) / 2.0)
        except Exception:
            return 0.0

    def get_edge_weight(self, node1: str, node2: str) -> float:
        """Get query-aware edge weight: w'(u,v) = w(u,v) * f(sim(u,q), sim(v,q))."""
        cache_key = (node1, node2)
        if cache_key in self.edge_weights_cache:
            return self.edge_weights_cache[cache_key]

        edge_data = self.graph[node1][node2]
        original_weight = edge_data.get('weight', 1.0)

        if not self.node_embeddings or not self.subquery_embedding:
            self.edge_weights_cache[cache_key] = original_weight
            return original_weight

        if original_weight <= 0:
            self.edge_weights_cache[cache_key] = 0.0
            return 0.0

        node1_emb = self.node_embeddings.get(node1, [0.0] * len(self.subquery_embedding))
        node2_emb = self.node_embeddings.get(node2, [0.0] * len(self.subquery_embedding))

        node1_query_sim = self.cosine_similarity(node1_emb, self.subquery_embedding)
        node2_query_sim = self.cosine_similarity(node2_emb, self.subquery_embedding)

        if self.weight_func == "multiply":
            smart_weight = original_weight * node1_query_sim * node2_query_sim
        elif self.weight_func == "add":
            smart_weight = (original_weight + node1_query_sim + node2_query_sim) / 3.0
        else:
            query_factor = (node1_query_sim + node2_query_sim) / 2.0
            smart_weight = original_weight * (1.0 + query_factor * 0.5)

        self.edge_weights_cache[cache_key] = smart_weight
        return smart_weight

    def initialize(self, alpha: float = 50, use_node_degree: bool = True):
        """Initialize sink capacities, source mass, and warm-start x.

        Args:
            alpha: Mass initialization factor (default 50)
            use_node_degree: Whether to use node degree for sink capacity (default True)
        """
        # Set sink capacity proportional to node degree
        for node in self.graph.nodes():
            if use_node_degree:
                self.sink_capacity[node] = max(self.graph.degree(node), 1)
            else:
                self.sink_capacity[node] = 1

        # Normalize sink capacity to sum to 10
        total_sink = sum(self.sink_capacity.values())
        if total_sink > 0:
            for node in self.sink_capacity:
                self.sink_capacity[node] = 10.0 * self.sink_capacity[node] / total_sink
        total_sink = sum(self.sink_capacity.values())

        # Set all masses to 0
        for node in self.graph.nodes():
            self.mass[node] = 0

        # Inject mass at source: alpha * total_sink * confidence_boost
        confidence_boost = 1.0 + self.confidence
        self.mass[self.source] = alpha * total_sink * confidence_boost

        # Warm-start x: 2-step lazy random walk from source
        self.x = defaultdict(float)
        self.x[self.source] = 1.0
        for _ in range(2):
            x_new = defaultdict(float)
            for node, val in self.x.items():
                if val > 0:
                    neighbors = list(self.graph.neighbors(node))
                    if neighbors:
                        for neighbor in neighbors:
                            x_new[neighbor] += val / len(neighbors)
            # Lazy update: average diffused values with source-anchored distribution
            x_combined = defaultdict(float)
            x_combined[self.source] = 1.0
            for node in set(list(x_new.keys()) + [self.source]):
                x_combined[node] = (x_combined.get(node, 0.0) + x_new.get(node, 0.0)) / 2.0
            self.x = x_combined

    def push(self, node: str) -> bool:
        """Push excess mass from node to neighbors via query-aware edges."""
        neighbors = list(self.graph.neighbors(node))
        if not neighbors:
            return False

        # Total outgoing weight
        w_i = 0
        for neighbor in neighbors:
            weight = self.get_edge_weight(node, neighbor)
            w_i += weight

        if w_i == 0:
            return False

        excess = self.mass[node] - self.sink_capacity[node]
        if excess <= 0:
            return False

        # Accumulate importance score with step size
        self.x[node] += self.step_size * excess / (w_i + 1e-8)

        # Absorb what sink can hold
        self.mass[node] = self.sink_capacity[node]

        # Push remaining excess to neighbors proportional to edge weights
        for neighbor in neighbors:
            w_ij = self.get_edge_weight(node, neighbor)
            if w_ij > 0:
                self.mass[neighbor] += excess * w_ij / (w_i + 1e-8)

        return True

    def flow_diffusion(self, max_iterations: int = 500) -> Dict[str, float]:
        """Run push-relabel flow diffusion until convergence.

        Args:
            max_iterations: Maximum iterations to run (default 500)

        Returns:
            Dictionary of nodes with positive flow values
        """
        iterations = 0
        pushes = 0

        while iterations < max_iterations:
            iterations += 1

            # Find nodes with excess mass
            excess_nodes = [node for node in self.graph.nodes()
                           if self.mass[node] > self.sink_capacity[node] + self.epsilon]

            if not excess_nodes:
                logger.debug(f"QAFD converged in {iterations} iterations ({pushes} pushes)")
                break

            node = random.choice(excess_nodes)

            if self.push(node):
                pushes += 1

            # Check convergence every 10 iterations
            if iterations % 10 == 0:
                remaining_excess = sum(max(0, self.mass[node] - self.sink_capacity[node])
                                     for node in self.graph.nodes())
                if remaining_excess < self.epsilon:
                    logger.debug(f"QAFD converged in {iterations} iterations ({pushes} pushes)")
                    break

        if iterations >= max_iterations:
            logger.warning(f"QAFD did not converge after {max_iterations} iterations")

        return {node: val for node, val in self.x.items() if val > 0}

    def retrieve(
        self,
        source_node: Optional[str] = None,
        target_node: Optional[str] = None,
        **kwargs
    ) -> RetrieverResult:
        """Retrieve nodes using flow diffusion.

        Args:
            source_node: Optional override for source node
            target_node: Optional override for target node
            **kwargs: Additional parameters:
                - alpha: Mass initialization factor (default 10)
                - max_iterations: Max diffusion iterations (default 500)

        Returns:
            RetrieverResult with diffused nodes and scores
        """
        alpha = kwargs.get('alpha', 50)
        max_iterations = kwargs.get('max_iterations', 500)

        self.initialize(alpha=alpha)
        diffused_nodes = self.flow_diffusion(max_iterations=max_iterations)

        return RetrieverResult(
            nodes=diffused_nodes,
            path=None,
            score=0.0,
            metadata={
                'source': self.source,
                'target': self.target,
                'confidence': self.confidence,
                'weight_func': self.weight_func
            }
        )

    def get_node_scores(self) -> Dict[str, float]:
        """Get the flow values for all processed nodes."""
        return dict(self.x)


# Alias for backward compatibility
QueryAwareWeightedFlowDiffusion = FlowDiffusionRetriever
