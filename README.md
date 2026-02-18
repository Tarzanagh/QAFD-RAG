# QAFD-RAG

Official code for the paper:

**[Query-Aware Flow Diffusion for Graph-Based RAG with Retrieval Guarantees](https://openreview.net/forum?id=n28wnc2QTc)**
*Zhuoping Zhou, Davoud Ataee Tarzanagh, Sima Didari, Wenjun Hu, Baruch Gutow, Oxana Verkholyak, Masoud Faraki, Heng Hao, Hankyu Moon, Seungjai Min*
**ICLR 2026**

QAFD-RAG uses **query-aware flow diffusion** to retrieve contextually relevant subgraphs from a knowledge graph. Unlike community-based (GraphRAG) or entity-centric (LightRAG) approaches, QAFD-RAG dynamically re-weights edges based on query relevance and propagates flow through the graph to discover multi-hop context with retrieval guarantees.

<p align="center">
  <img src="docs/figs/GraphRAG.png" width="260" alt="GraphRAG"/>
  <img src="docs/figs/LightRAG.png" width="260" alt="LightRAG"/>
  <img src="docs/figs/QAFD-RAG.png" width="260" alt="QAFD-RAG"/>
</p>
<p align="center">
  <em>Left:</em> GraphRAG &nbsp;&nbsp;|&nbsp;&nbsp; <em>Center:</em> LightRAG &nbsp;&nbsp;|&nbsp;&nbsp; <em>Right:</em> QAFD-RAG

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# 3. Build a knowledge graph
./run.sh ultradomain --build --max-documents 100

# 4. Run a benchmark
./run.sh ultradomain --questions 10
```

## Project Structure

```
QAFD-RAG/
├── run.sh                           # CLI entry point
├── requirements.txt
├── src/
│   ├── QAFD_RAG.py                  # Main class
│   ├── base.py                      # Storage interfaces, QueryParam
│   ├── llm.py                       # LLM and embedding functions
│   ├── storage.py                   # KV, vector, and graph storage
│   ├── operate.py                   # Query operations
│   ├── evaluation.py                # Answer evaluation
│   ├── answering/                   # Query processing pipeline
│   │   ├── handler.py               # kg_query entry point
│   │   ├── context.py               # Context building (local/global/hybrid)
│   │   ├── clusters.py              # Flow diffusion clustering
│   │   └── text_units.py            # Text chunk retrieval
│   ├── retrievers/
│   │   ├── flow_diffusion.py        # Flow diffusion algorithm
│   │   └── base.py                  # Retriever interface
│   ├── indexing/                     # KG construction
│   │   ├── chunkers.py              # Token-based chunking
│   │   ├── extractors.py            # Entity/relationship extraction
│   │   ├── schema_builder.py        # Database schema KG builder
│   │   └── build_kg.py              # CLI KG builder
│   ├── embedding_models/            # Embedding model implementations
│   │   ├── JinaV3.py
│   │   ├── GritLM.py
│   │   ├── NVEmbedV2.py
│   │   └── OpenAI.py
│   ├── prompts/                     # LLM prompt templates
│   ├── text2sql/                    # Text-to-SQL support
│   └── utils/                       # Helpers
├── benchmarks/
│   ├── ultradomain/                 # General domain QA
│   ├── multihop/                    # Multi-hop reasoning
│   ├── text2sql/                    # Natural language to SQL
│   └── summarization/               # Document summarization
├── data/                            # Datasets
├── docs/figs/                       # Figures for README
├── kg/                              # Built knowledge graphs
├── ICLR2026/                        # Paper source (LaTeX)
└── results/                         # Benchmark results (JSON)
```

## Benchmarks

| Task | Datasets | Metrics |
|------|----------|---------|
| **UltraDomain** | mix.jsonl | Quality scores (comprehensiveness, diversity, relevance, logicality, coherence) |
| **Multi-hop QA** | MuSiQue, HotpotQA, 2WikiMultiHopQA | F1, Exact Match |
| **Text-to-SQL** | Spider2-lite (Pagila, etc.) | Schema retrieval accuracy |
| **Summarization** | SQuALITY | BLEU, ROUGE, METEOR, quality scores |

## Usage

All benchmarks are run through `./run.sh`:

```bash
./run.sh <task> [options]
```

### Build Knowledge Graphs

```bash
./run.sh ultradomain --build --max-documents 100
./run.sh summarization --build --max-documents 10
./run.sh multihop --dataset musique --build --max-documents 500
./run.sh multihop --dataset hotpotqa --build --max-documents 500
```

### Run Benchmarks

```bash
./run.sh ultradomain --questions 10
./run.sh summarization --questions 50
./run.sh multihop --dataset musique --questions 100
./run.sh multihop --dataset hotpotqa --questions 100
./run.sh multihop --dataset 2wikimultihopqa --questions 100
./run.sh text2sql --questions 5 --db Pagila
```

### Common Options

| Option | Description |
|--------|-------------|
| `--questions N` | Number of questions to evaluate |
| `--build` | Build KG only (skip benchmark) |
| `--force-build` | Rebuild KG even if it exists |
| `--max-documents N` | Limit documents for KG construction |
| `--embedding MODEL` | Embedding model (see table below) |
| `--llm MODEL` | LLM model (`gpt-4o-mini`, `gpt-4o`, `gpt-oss-120b`) |

### Task-Specific Options

| Option | Task | Description |
|--------|------|-------------|
| `--dataset NAME` | multihop | `musique`, `hotpotqa`, `2wikimultihopqa` |
| `--dataset NAME` | summarization | `squality` (default) |
| `--db NAME` | text2sql | Database name (e.g., `Pagila`) |
| `--mode MODE` | multihop, summarization | `local`, `global`, `hybrid` (default: `hybrid`) |

## Configuration

### Embedding Models

| Key | Provider | Dimensions | Cost |
|-----|----------|-----------|------|
| `openai-small` | OpenAI API | 1536 | $0.02/1M tokens |
| `openai-large` | OpenAI API | 3072 | $0.13/1M tokens |
| `jina-v3` | Local (GPU) | 1024 | Free |
| `gritlm` | Local (GPU) | 4096 | Free |
| `nvidia-nv-embed-v2` | Local (GPU) | 4096 | Free |

Use with `--embedding <key>`, e.g.:

```bash
./run.sh ultradomain --questions 10 --embedding jina-v3
```

### LLM Models

| Key | Description |
|-----|-------------|
| `gpt-4o-mini` | Default. Fast, cost-effective |
| `gpt-4o` | Higher quality, slower |
| `gpt-oss-120b` | Free, open-source |

## Python API

```python
from src import QAFD_RAG, QueryParam

# Initialize
rag = QAFD_RAG(
    working_dir="./my_kg",
    llm_model_name="gpt-4o-mini",
    embedding_model_key="jina-v3",
)

# Index documents
rag.insert(["Document text 1...", "Document text 2..."])

# Query
answer = rag.query(
    "What is X?",
    param=QueryParam(mode="hybrid"),
)
print(answer)
```

### QueryParam Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `"hybrid"` | Retrieval mode: `local`, `global`, `hybrid` |
| `top_k` | 40 | Number of entities to retrieve |
| `max_source_nodes` | 40 | Max source nodes for flow diffusion |
| `min_flow_threshold` | 0.1 | Minimum flow value to include a node |
| `enable_query_aware_flow_diffusion` | `True` | Use query-aware edge weighting |
| `alpha` | 50.0 | Flow diffusion initialization factor |
| `response_type` | `"Multiple Paragraphs"` | Output format |

## How It Works

QAFD-RAG processes documents through a two-stage pipeline:

**Stage 1 -- Knowledge Graph Construction**

1. **Chunking**: Documents are split into overlapping token-based chunks.
2. **Entity Extraction**: An LLM identifies entities (people, organizations, events, etc.) and their relationships from each chunk.
3. **Graph Assembly**: Entities become nodes, relationships become weighted edges, and all are embedded into vector space.

**Stage 2 -- Query-Aware Retrieval and Answering**

1. **Entity Matching**: The query is parsed for keywords, which are matched to KG entities via vector similarity.
2. **Flow Diffusion**: Starting from matched entities, a flow diffusion algorithm propagates "mass" through the graph. Edge weights are dynamically adjusted based on each node's similarity to the query (query-aware weighting).
3. **Context Assembly**: Top-ranked nodes, relationships, and their associated text chunks are assembled into a context window.
4. **LLM Answering**: The context is passed to an LLM along with the original query to generate the final answer.

The flow diffusion algorithm is the key differentiator: rather than simple graph traversal or vector search alone, it combines graph structure with query relevance to find contextually important information that may be several hops away from the initial match.
