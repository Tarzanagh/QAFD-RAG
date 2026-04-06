"""
Configuration dataclass for the passage-entity KG pipeline in QAFD-RAG.

Combines passage-entity BaseConfig fields with QAFD algorithm parameters.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class PassageEntityConfig:
    """Unified configuration for indexing, retrieval, and QAFD."""

    # ── LLM ────────────────────────────────────────────────────────────────
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""                    # falls back to OPENAI_API_KEY env
    max_new_tokens: Optional[int] = 2048
    temperature: float = 0.0

    # ── Embedding ──────────────────────────────────────────────────────────
    embedding_model_key: str = "nvidia-nv-embed-v2"   # key in QAFD-RAG registry
    embedding_batch_size: int = 16

    # ── Dataset / paths ────────────────────────────────────────────────────
    dataset: Optional[str] = None                     # musique, hotpotqa, 2wikimultihopqa
    save_dir: str = "outputs"
    force_index_from_scratch: bool = False
    force_openie_from_scratch: bool = False
    save_openie: bool = True

    # ── Graph construction ─────────────────────────────────────────────────
    is_directed_graph: bool = False
    synonymy_edge_topk: int = 2047
    synonymy_edge_query_batch_size: int = 1000
    synonymy_edge_key_batch_size: int = 10000
    synonymy_edge_sim_threshold: float = 0.8

    # ── Retrieval ──────────────────────────────────────────────────────────
    linking_top_k: int = 10
    retrieval_top_k: int = 200
    passage_node_weight: float = 0.05
    damping: float = 0.5

    # ── QA ─────────────────────────────────────────────────────────────────
    qa_top_k: int = 5

    # ── QAFD algorithm parameters ──────────────────────────────────────────
    use_qafd: bool = True
    qafd_alpha: float = 2.0
    qafd_epsilon: float = 0.01
    qafd_max_iterations: int = 500
    qafd_weight_scheme: str = "multiply"     # "multiply", "add", "original"
    qafd_use_node_degree: bool = True
    qafd_step_size: float = 0.2
    qafd_random_seed: int = 42

    # ── Query-aware enhancements (all default = original behaviour) ────────
    sim_mode: str = "normalized"     # Similarity contrast: "normalized", "relu", "relu_sq"
    qa_sink_gamma: float = 0.0       # Query-aware sink capacity (0=off)
    qa_warm_delta: float = 0.0       # Query-aware seed bias (0=off)
    qa_warm_walk: bool = False       # Use QA edge weights in warm-start walk
    qa_warm_steps: int = 2           # Number of warm-start steps (default 2)
    qa_accum_gamma: float = 0.0      # Query-aware x accumulation boost (0=off)
    qa_post_lambda: float = 0.0      # Post-diffusion reranking (0=off)
    batch_push: bool = False         # Batch push-relabel (process all excess nodes per iter)

    # ── Reranker ───────────────────────────────────────────────────────────
    rerank_dspy_file_path: Optional[str] = None      # path to DSPy JSON; None → built-in prompt

    def __post_init__(self):
        if self.save_dir == "outputs" and self.dataset:
            self.save_dir = f"outputs/{self.dataset}"

    @property
    def working_dir(self) -> str:
        """Model-specific sub-directory under save_dir.

        Also checks kg/multihop/ for pre-downloaded KGs from HuggingFace.
        If found there, uses that path instead of outputs/.
        """
        import os
        llm_label = self.llm_model.replace("/", "_")
        emb_label = self.embedding_model_key.replace("/", "_")

        # Check HuggingFace download location (kg/multihop/{llm}_{emb}_{dataset}/)
        if self.dataset:
            for task_dir in ["multihop", "ultradomain"]:
                hf_path = os.path.join("kg", task_dir, f"{llm_label}_{emb_label}_{self.dataset}")
                if os.path.isdir(hf_path) and os.path.exists(os.path.join(hf_path, "graph.pickle")):
                    return hf_path

        return f"{self.save_dir}/{llm_label}_{emb_label}"
