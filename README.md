# QAFD-RAG

Official code for ICLR 2026 paper: **[Query-Aware Flow Diffusion for Graph-Based RAG with Retrieval Guarantees](https://openreview.net/forum?id=n28wnc2QTc)**

QAFD-RAG uses **query-aware flow diffusion** to retrieve contextually relevant subgraphs from a knowledge graph with retrieval guarantees.

<p align="center"><em>Query: "Introduce Steve Jobs's products in Apple."</em></p>

<table align="center"><tr>
  <td align="center"><img src="docs/figs/GraphRAG.png" width="260" alt="GraphRAG"/><br/><em>GraphRAG</em></td>
  <td align="center"><img src="docs/figs/LightRAG.png" width="260" alt="LightRAG"/><br/><em>LightRAG</em></td>
  <td align="center"><img src="docs/figs/QAFD-RAG.png" width="260" alt="QAFD-RAG"/><br/><em>QAFD-RAG</em></td>
</tr></table>
<p align="center"><em>QAFD-RAG reweights edges by query relevance, suppressing irrelevant neighborhoods.</em></p>

## Quick Start

```bash
conda create -n qafd-rag python=3.10 -y
conda activate qafd-rag
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."

python benchmarks/run.py --task multihop --dataset musique --questions 10
```

## Graph Types

QAFD-RAG supports two knowledge graph types. Both use Query-Aware Flow Diffusion.

| | **Entity Graph** | **Passage-Entity Graph** ([Gutiérrez et al., 2024](https://arxiv.org/abs/2405.14831)) |
|---|---|---|
| **Nodes** | Entities + relationships | Entities + passages + facts |
| **Edges** | Entity-entity relationships | Fact + passage + synonymy edges |
| **How passages are found** | Looked up after graph traversal | Ranked directly as graph nodes |
| **Default for** | ultradomain, text2sql, summarization | multihop |

## Usage

```bash
python benchmarks/run.py --task <task> --dataset <dataset> [--graph_type <type>] [options]
```

### Multihop QA

```bash
# Uses passage-entity graph by default
python benchmarks/run.py --task multihop --dataset musique --questions 100
python benchmarks/run.py --task multihop --dataset hotpotqa --questions 100
python benchmarks/run.py --task multihop --dataset 2wikimultihopqa --questions 100

# Use entity graph instead
python benchmarks/run.py --task multihop --dataset musique --graph_type entity --questions 100
```

### UltraDomain QA

```bash
# Uses entity graph by default
python benchmarks/run.py --task ultradomain --dataset mix --questions 10

# Use passage-entity graph instead
python benchmarks/run.py --task ultradomain --dataset mix --graph_type passage-entity --questions 10
```

Available UltraDomain datasets: agriculture, biology, cs, finance, legal, math, medicine, mix, music, philosophy, physics

### Text-to-SQL & Summarization

```bash
python benchmarks/run.py --task text2sql --dataset spider2-lite
python benchmarks/run.py --task summarization --dataset narrativeqa
```

### Options

| Option | Description |
|--------|-------------|
| `--task` | `multihop`, `ultradomain`, `text2sql`, `summarization` |
| `--dataset` | Dataset name (e.g., `musique`, `mix`, `spider2-lite`) |
| `--graph_type` | `passage-entity` or `entity` (auto-selected per task) |
| `--questions N` | Number of questions |
| `--embedding` | `openai-small` (default), `openai-large`, `jina-v3`, `gritlm`, `nvidia-nv-embed-v2` |
| `--llm` | `gpt-4o-mini` (default), `gpt-4o`, `gpt-5-nano`, `gpt-5-mini`, `gpt-5`, `gpt-oss-120b` |
| `--alpha` | QAFD alpha (default: 3.0) |
| `--skip_qa` | Run retrieval only, skip answer generation |
| `--build_only` | Build KG only |
| `--batch_push` | Use batch push-relabel |

### Pre-built Knowledge Graphs

KGs are auto-generated on first run. To skip building, download from [HuggingFace](https://huggingface.co/datasets/qafd/kg):

```bash
# Single KG (quick test)
huggingface-cli download qafd/kg --repo-type dataset \
    --include "ultradomain/gpt-4o-mini_openai-small_mix/*" --local-dir ./kg

# All KGs for one task
huggingface-cli download qafd/kg --repo-type dataset --include "multihop/*" --local-dir ./kg

# Everything
huggingface-cli download qafd/kg --repo-type dataset --local-dir ./kg
```

### Local Embeddings

Local models run on GPU without an API key for embeddings:

```bash
python benchmarks/run.py --task multihop --dataset musique --embedding jina-v3 --questions 10
python benchmarks/run.py --task ultradomain --dataset mix --embedding nvidia-nv-embed-v2 --questions 10
```

> An LLM API key is still needed for KG building and answer generation.

## How It Works

**Stage 1 — Knowledge Graph Construction**

*Entity Graph:* LLM extracts entities and relationships from document chunks. Entities become nodes, relationships become edges.

*Passage-Entity Graph:* OpenIE extracts named entities and triples. Both entities and passages are graph nodes, connected by fact edges, passage-entity edges, and synonymy edges.

**Stage 2 — Query-Aware Flow Diffusion**

1. **Seed Selection**: Query matched to entities/facts via embedding similarity + LLM reranking
2. **Flow Diffusion**: Mass injected at seeds, propagated via push-relabel with query-aware edge weights
3. **Passage Ranking**: Nodes accumulate scores proportional to flow; passages ranked directly (passage-entity) or looked up from top entities (entity graph)
4. **Answer Generation**: Top passages assembled into context for LLM

## Citation

```bibtex
@inproceedings{zhou2026qafd,
  title={Query-Aware Flow Diffusion for Graph-Based RAG with Retrieval Guarantees},
  author={Zhou, Zhuoping and Ataee Tarzanagh, Davoud and Didari, Sima and Hu, Wenjun and Gutow, Baruch and Verkholyak, Oxana and Faraki, Masoud and Hao, Heng and Moon, Hankyu and Min, Seungjai},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```
