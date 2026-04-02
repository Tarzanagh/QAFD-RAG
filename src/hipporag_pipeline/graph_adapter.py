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

def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray, mode: str = "normalized") -> float:
    """Cosine similarity with configurable contrast.

    Modes:
        "normalized": (cos+1)/2 → [0, 1] (original, low contrast)
        "relu":       max(0, cos) → [0, 1] (natural contrast)
        "relu_sq":    max(0, cos)² → [0, 1] (sharpest contrast)
    """
    if len(vec1) == 0 or len(vec2) == 0:
        return 0.0
    dot = np.dot(vec1, vec2)
    m1 = np.linalg.norm(vec1)
    m2 = np.linalg.norm(vec2)
    if m1 == 0 or m2 == 0:
        return 0.0
    raw = dot / (m1 * m2)
    if mode == "relu":
        return max(0.0, raw)
    elif mode == "relu_sq":
        r = max(0.0, raw)
        return r * r
    else:  # "normalized" — original
        return max(0.0, (raw + 1.0) / 2.0)


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
        hybrid_a: float = 1.0,
        hybrid_b: float = 0.5,
        use_node_degree: bool = True,
        random_seed: int = 42,
        threshold: float = 1e-5,
        # ── Query-aware enhancements (all default OFF = original behaviour) ──
        sim_mode: str = "normalized",   # Similarity contrast: "normalized", "relu", "relu_sq"
        qa_sink_gamma: float = 0.0,     # query-aware sink capacity
        qa_warm_delta: float = 0.0,     # query-aware seed bias
        qa_warm_walk: bool = False,     # query-aware warm-start random walk (uses edge weights)
        qa_warm_steps: int = 2,         # number of warm-start steps (default 2)
        qa_accum_gamma: float = 0.0,    # query-aware x accumulation boost
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
        self.hybrid_a = hybrid_a
        self.hybrid_b = hybrid_b
        self.use_node_degree = use_node_degree
        self.sim_mode = sim_mode
        self.qa_sink_gamma = qa_sink_gamma
        self.qa_warm_delta = qa_warm_delta
        self.qa_warm_walk = qa_warm_walk
        self.qa_accum_gamma = qa_accum_gamma

        n = len(node_name_to_idx)

        # Precompute per-node query similarity (used by sink/warm QA)
        self._node_query_sim = np.zeros(n)
        if (qa_sink_gamma > 0 or qa_warm_delta > 0) and query_embedding is not None:
            for i in range(n):
                name = self.idx_to_node_name.get(i)
                if name:
                    emb = self.node_embeddings.get(name)
                    if emb is not None:
                        self._node_query_sim[i] = _cosine_similarity(emb, query_embedding, mode=sim_mode)

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

        # Warm-start x: multi-step lazy random walk from seed distribution
        if qa_warm_delta > 0:
            x = self.source_weights * (1.0 + qa_warm_delta * self._node_query_sim)
            x_sum = np.sum(x)
            if x_sum > 0:
                x /= x_sum
        else:
            x = self.source_weights.copy()

        for _ in range(qa_warm_steps):
            x_new = np.zeros(n)
            for i in range(n):
                if x[i] > 0:
                    neighbors = self.graph.neighbors(i)
                    if not neighbors:
                        continue
                    if qa_warm_walk and query_embedding is not None:
                        # Query-aware walk: spread proportional to edge weights
                        weights = []
                        for j in neighbors:
                            w = self._get_edge_weight(i, j)
                            weights.append(w)
                        total_w = sum(weights)
                        if total_w > 0:
                            for j, w in zip(neighbors, weights):
                                x_new[j] += x[i] * w / total_w
                        else:
                            spread = x[i] / len(neighbors)
                            for j in neighbors:
                                x_new[j] += spread
                    else:
                        # Original: uniform spread
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
        s1 = _cosine_similarity(e1 if e1 is not None else zero, self.query_embedding, mode=self.sim_mode)
        s2 = _cosine_similarity(e2 if e2 is not None else zero, self.query_embedding, mode=self.sim_mode)

        if self.weight_scheme == "multiply":
            # Product (Eq. 5b): w * sim(u,q) * sim(v,q)
            qw = w * s1 * s2
        elif self.weight_scheme == "add":
            # Mean (Eq. 5a): (w + sim(u,q) + sim(v,q)) / 3
            qw = (w + s1 + s2) / 3.0
        else:  # "original" = Hybrid (Eq. 5c)
            # w * (a + b * avg_query_sim)
            qf = (s1 + s2) / 2.0
            qw = w * (self.hybrid_a + self.hybrid_b * qf)

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

        # Phase 1: query-aware sink capacity — relevant nodes absorb more
        if self.qa_sink_gamma > 0:
            self.sink_capacity *= (1.0 + self.qa_sink_gamma * self._node_query_sim)

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

        # Accumulate importance — optionally boosted by query relevance
        accum = self.step_size * excess / (w_i + 1e-8)
        if self.qa_accum_gamma > 0:
            accum *= (1.0 + self.qa_accum_gamma * self._node_query_sim[node_idx])
        self.x[node_idx] += accum
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
    hybrid_a: float = 1.0,
    hybrid_b: float = 0.5,
    use_node_degree: bool = True,
    random_seed: int = 42,
    sim_mode: str = "normalized",
    qa_sink_gamma: float = 0.0,
    qa_warm_delta: float = 0.0,
    qa_warm_walk: bool = False,
    qa_warm_steps: int = 2,
    qa_accum_gamma: float = 0.0,
    qa_post_lambda: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run QAFD on igraph and return (sorted_doc_ids, sorted_doc_scores).

    sim_mode: Similarity contrast function ("normalized", "relu", "relu_sq")
    Query-aware enhancement flags (all default 0.0 = original behaviour):
        qa_sink_gamma:  Scale sink capacity by (1 + gamma * sim(node, query))
        qa_warm_delta:  Bias warm-start x toward query-relevant seeds
        qa_post_lambda: Rerank output by (1 + lambda * sim(passage, query))
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
        hybrid_a=hybrid_a,
        hybrid_b=hybrid_b,
        use_node_degree=use_node_degree,
        random_seed=random_seed,
        sim_mode=sim_mode,
        qa_sink_gamma=qa_sink_gamma,
        qa_warm_delta=qa_warm_delta,
        qa_warm_walk=qa_warm_walk,
        qa_warm_steps=qa_warm_steps,
        qa_accum_gamma=qa_accum_gamma,
    )

    node_scores = qafd.run()

    # Extract passage scores
    doc_scores = np.array([node_scores[idx] for idx in passage_node_idxs])

    # Phase 3: post-diffusion query-aware reranking
    if qa_post_lambda > 0 and query_embedding is not None and node_embeddings:
        idx_to_name = qafd.idx_to_node_name
        for pi, pidx in enumerate(passage_node_idxs):
            name = idx_to_name.get(pidx)
            if name:
                emb = node_embeddings.get(name)
                if emb is not None:
                    sim = _cosine_similarity(emb, query_embedding, mode=sim_mode)
                    doc_scores[pi] *= (1.0 + qa_post_lambda * sim)

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
