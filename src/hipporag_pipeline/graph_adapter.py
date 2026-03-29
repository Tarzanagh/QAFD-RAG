"""
Bridge between igraph (HippoRAG's graph format) and QAFD-RAG's flow diffusion.

Provides:
    - ``igraph_to_networkx``: convert an igraph.Graph to NetworkX (kept for
      compatibility, but no longer used in the main retrieval path).
    - ``IGraphQAFD``: igraph-native QAFD that matches HippoRAG's
      ``QueryAwareFlowDiffusion`` exactly — numpy arrays, C-based neighbor
      lookups, no NetworkX conversion overhead.
"""

import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ===========================================================================
# igraph  -->  NetworkX  (kept for compatibility; not used in hot path)
# ===========================================================================

def igraph_to_networkx(ig_graph):
    """Convert an igraph.Graph to a NetworkX (undirected) graph."""
    import networkx as nx

    G = nx.Graph()
    name_attr = ig_graph.vs.attribute_names()
    has_name = "name" in name_attr

    for v in ig_graph.vs:
        node_id = v["name"] if has_name else v.index
        G.add_node(node_id)

    has_weight = "weight" in ig_graph.es.attribute_names()

    for e in ig_graph.es:
        src = ig_graph.vs[e.source]["name"] if has_name else e.source
        tgt = ig_graph.vs[e.target]["name"] if has_name else e.target
        w = e["weight"] if has_weight else 1.0
        G.add_edge(src, tgt, weight=w)

    return G


# ===========================================================================
# igraph-native Query-Aware Flow Diffusion
# ===========================================================================

def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Cosine similarity normalised to [0, 1]."""
    if len(vec1) == 0 or len(vec2) == 0:
        return 0.0
    dot = np.dot(vec1, vec2)
    m1 = np.linalg.norm(vec1)
    m2 = np.linalg.norm(vec2)
    if m1 == 0 or m2 == 0:
        return 0.0
    return max(0.0, (dot / (m1 * m2) + 1.0) / 2.0)


class IGraphQAFD:
    """Query-Aware Flow Diffusion directly on igraph — matches HippoRAG exactly.

    Uses numpy arrays for mass/x/sink_capacity and igraph's C-based
    ``graph.neighbors()`` for fast neighbour lookups.

    Parameters
    ----------
    graph : igraph.Graph
    node_name_to_idx : dict
        Mapping from node name (str) -> vertex index (int).
    source_weights : np.ndarray
        Per-node seed weights (length = number of nodes). Will be normalised.
    node_embeddings : dict
        Mapping node_name -> np.ndarray embedding.
    query_embedding : np.ndarray
        Query embedding vector.
    alpha, epsilon, max_iterations, step_size : float / int
        Algorithm parameters.
    weight_scheme : str
        "original", "multiply", or "add".
    random_seed : int
    """

    def __init__(
        self,
        graph,
        node_name_to_idx: Dict[str, int],
        source_weights: np.ndarray,
        node_embeddings: Dict[str, np.ndarray],
        query_embedding: Optional[np.ndarray],
        alpha: float = 10.0,
        epsilon: float = 1e-6,
        max_iterations: int = 10000,
        step_size: float = 0.2,
        weight_scheme: str = "original",
        use_node_degree: bool = True,
        random_seed: int = 42,
        threshold: float = 1e-5,
    ):
        self.graph = graph
        self.node_name_to_idx = node_name_to_idx
        self.idx_to_node_name = {v: k for k, v in node_name_to_idx.items()}
        self.node_embeddings = node_embeddings or {}
        self.query_embedding = query_embedding
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.weight_scheme = weight_scheme
        self.use_node_degree = use_node_degree

        n = len(node_name_to_idx)

        # Normalise source weights (threshold small values, then normalise)
        sw = np.copy(source_weights).astype(np.float64)
        sw[sw < threshold] = 0.0
        sw_sum = np.sum(sw)
        if sw_sum > 0:
            sw /= sw_sum
        else:
            sw = np.ones(n) / n
        self.source_weights = sw

        # State arrays
        self.mass = np.zeros(n)
        self.sink_capacity = np.zeros(n)
        self.x = np.zeros(n)

        # Edge weight cache
        self._edge_weight_cache: Dict[Tuple[int, int], float] = {}

        random.seed(random_seed)

        # Warm-start x (2-step lazy random walk from seed distribution)
        x = self.source_weights.copy()
        for _ in range(2):
            x_new = np.zeros(n)
            for i in range(n):
                if x[i] > 0:
                    neighbors = self.graph.neighbors(i)
                    if neighbors:
                        spread = x[i] / len(neighbors)
                        for j in neighbors:
                            x_new[j] += spread
            x = (self.source_weights + x_new) / 2.0
        self.x = x

    # ------------------------------------------------------------------
    def _get_edge_weight(self, i: int, j: int) -> float:
        """Get (cached) query-aware edge weight between node indices i and j."""
        key = (i, j)
        if key in self._edge_weight_cache:
            return self._edge_weight_cache[key]

        try:
            eid = self.graph.get_eid(i, j)
            attrs = self.graph.es[eid].attributes()
            w = attrs.get("weight", 1.0)
        except Exception:
            self._edge_weight_cache[key] = 0.0
            return 0.0

        if w <= 0:
            self._edge_weight_cache[key] = 0.0
            return 0.0

        # Query-aware modulation
        if not self.node_embeddings or self.query_embedding is None:
            self._edge_weight_cache[key] = w
            return w

        n1 = self.idx_to_node_name.get(i)
        n2 = self.idx_to_node_name.get(j)
        if n1 is None or n2 is None:
            self._edge_weight_cache[key] = w
            return w

        e1 = self.node_embeddings.get(n1)
        e2 = self.node_embeddings.get(n2)
        if e1 is None and e2 is None:
            self._edge_weight_cache[key] = w
            return w

        zero = np.zeros_like(self.query_embedding)
        s1 = _cosine_similarity(e1 if e1 is not None else zero, self.query_embedding)
        s2 = _cosine_similarity(e2 if e2 is not None else zero, self.query_embedding)

        if self.weight_scheme == "multiply":
            qw = w * s1 * s2
        elif self.weight_scheme == "add":
            qw = (w + s1 + s2) / 3.0
        else:  # "original"
            qf = (s1 + s2) / 2.0
            qw = w * (1.0 + qf * 0.5)

        self._edge_weight_cache[key] = qw
        return qw

    # ------------------------------------------------------------------
    def _initialize(self):
        """Set sink capacities and inject mass at seeds."""
        n = len(self.source_weights)

        if self.use_node_degree:
            for i in range(n):
                self.sink_capacity[i] = max(self.graph.degree(i), 1.0)
        else:
            self.sink_capacity[:] = 1.0

        total_sink = np.sum(self.sink_capacity)
        self.sink_capacity = 10.0 * self.sink_capacity / total_sink
        total_sink = np.sum(self.sink_capacity)

        # Inject mass at seeds
        self.mass[:] = 0.0
        for i in range(n):
            if self.source_weights[i] > 0:
                self.mass[i] = self.alpha * total_sink * self.source_weights[i]

    # ------------------------------------------------------------------
    def _push(self, node_idx: int) -> bool:
        """Push excess mass from node to neighbours."""
        neighbors = self.graph.neighbors(node_idx)
        if not neighbors:
            return False

        w_i = 0.0
        for j in neighbors:
            w_i += self._get_edge_weight(node_idx, j)

        if w_i == 0:
            return False

        excess = self.mass[node_idx] - self.sink_capacity[node_idx]
        if excess <= 0:
            return False

        self.x[node_idx] += self.step_size * excess / (w_i + 1e-8)
        self.mass[node_idx] = self.sink_capacity[node_idx]

        for j in neighbors:
            w_ij = self._get_edge_weight(node_idx, j)
            if w_ij > 0:
                self.mass[j] += excess * w_ij / (w_i + 1e-8)

        return True

    # ------------------------------------------------------------------
    def run(self) -> np.ndarray:
        """Run push-relabel flow diffusion. Returns per-node scores (np.ndarray)."""
        self._initialize()

        iterations = 0
        pushes = 0

        while iterations < self.max_iterations:
            iterations += 1

            # Find nodes with excess mass (vectorised)
            excess_mask = self.mass > (self.sink_capacity + self.epsilon)
            excess_indices = np.nonzero(excess_mask)[0]

            if len(excess_indices) == 0:
                logger.info(f"QAFD converged in {iterations} iters ({pushes} pushes)")
                break

            node_idx = int(random.choice(excess_indices))
            if self._push(node_idx):
                pushes += 1

            if iterations % 10 == 0:
                remaining = np.sum(np.maximum(0, self.mass - self.sink_capacity))
                if remaining < self.epsilon:
                    logger.info(f"QAFD converged in {iterations} iters ({pushes} pushes)")
                    break

        if iterations >= self.max_iterations:
            logger.warning(f"QAFD did not converge after {self.max_iterations} iterations")

        return self.x


# ===========================================================================
# Convenience wrapper matching the interface used by retriever.py
# ===========================================================================

def run_igraph_qafd(
    graph,
    node_name_to_idx: Dict[str, int],
    passage_node_idxs: List[int],
    source_weights: np.ndarray,
    node_embeddings: Dict[str, np.ndarray],
    query_embedding: Optional[np.ndarray],
    alpha: float = 10.0,
    epsilon: float = 1e-6,
    max_iterations: int = 10000,
    step_size: float = 0.2,
    weight_scheme: str = "original",
    use_node_degree: bool = True,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run QAFD on igraph and return (sorted_doc_ids, sorted_doc_scores).

    This is a drop-in replacement for HippoRAG's ``run_qafd()``.
    """
    qafd = IGraphQAFD(
        graph=graph,
        node_name_to_idx=node_name_to_idx,
        source_weights=source_weights,
        node_embeddings=node_embeddings,
        query_embedding=query_embedding,
        alpha=alpha,
        epsilon=epsilon,
        max_iterations=max_iterations,
        step_size=step_size,
        weight_scheme=weight_scheme,
        use_node_degree=use_node_degree,
        random_seed=random_seed,
    )

    node_scores = qafd.run()

    # Extract passage scores
    doc_scores = np.array([node_scores[idx] for idx in passage_node_idxs])

    total = np.sum(doc_scores)
    if total > 0:
        doc_scores = doc_scores / total
    else:
        doc_scores = np.ones(len(doc_scores)) / max(len(doc_scores), 1)

    sorted_ids = np.argsort(doc_scores)[::-1]
    sorted_scores = doc_scores[sorted_ids]

    return sorted_ids, sorted_scores


# ===========================================================================
# Fast PPR via igraph (matches HippoRAG's actual benchmark method)
# ===========================================================================

def run_ppr(
    graph,
    node_name_to_idx: Dict[str, int],
    passage_node_idxs: List[int],
    reset_prob: np.ndarray,
    damping: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run Personalized PageRank on igraph and return (sorted_doc_ids, sorted_doc_scores).

    This matches HippoRAG's ``run_ppr()`` with ``use_qafd=False``.
    Uses igraph's C-based prpack implementation — converges instantly.
    """
    reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)

    pagerank_scores = graph.personalized_pagerank(
        vertices=range(len(node_name_to_idx)),
        damping=damping,
        directed=False,
        weights="weight",
        reset=reset_prob,
        implementation="prpack",
    )

    doc_scores = np.array([pagerank_scores[idx] for idx in passage_node_idxs])

    sorted_ids = np.argsort(doc_scores)[::-1]
    sorted_scores = doc_scores[sorted_ids]

    return sorted_ids, sorted_scores
