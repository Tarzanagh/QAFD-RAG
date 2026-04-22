# QAFD-RAG

Official code for ICLR 2026 paper: **[Query-Aware Flow Diffusion for Graph-Based RAG with Retrieval Guarantees](https://openreview.net/forum?id=n28wnc2QTc)**

QAFD-RAG uses **query-aware flow diffusion** to retrieve contextually relevant subgraphs from a knowledge graph. Unlike community-based (GraphRAG) or entity-centric (LightRAG) approaches, QAFD-RAG dynamically re-weights edges based on query relevance and propagates flow through the graph to discover multi-hop context with retrieval guarantees.

<p align="center"><em>Query: "Introduce Steve Jobs's products in Apple."</em></p>

<table align="center"><tr>
  <td align="center"><img src="docs/figs/GraphRAG.png" width="260" alt="GraphRAG"/><br/><em>GraphRAG</em></td>
  <td align="center"><img src="docs/figs/LightRAG.png" width="260" alt="LightRAG"/><br/><em>LightRAG</em></td>
  <td align="center"><img src="docs/figs/QAFD-RAG.png" width="260" alt="QAFD-RAG"/><br/><em>QAFD-RAG</em></td>
</tr></table>
<p align="center"><em>QAFD-RAG reweights edges by query relevance, suppressing irrelevant neighborhoods (e.g., Amazon, Apple fruit).</em></p>

## Quick Start

```bash
# 1. Create conda environment and install dependencies
conda create -n qafd-rag python=3.10 -y
conda activate qafd-rag
pip install -r requirements.txt

# 2. Set your OpenAI API key (required for LLM and answer generation)
export OPENAI_API_KEY="sk-..."

# 3. Download pre-built KGs (recommended — saves hours of build time + API costs)
huggingface-cli download tarzanagh/QAFD-RAG --include "kg/multihop/*" --local-dir .
huggingface-cli download tarzanagh/QAFD-RAG --include "kg/ultradomain/*" --local-dir .

# 4. Run a benchmark
#    Note: First run for multihop will auto-download the nvidia/NV-Embed-v2
#    embedding model (~8GB, one-time, requires GPU with 16GB+ VRAM).
#    No GPU? See "Rebuild with openai-small" below.
python benchmarks/run.py --task multihop --dataset musique --questions 10
python benchmarks/run.py --task ultradomain --dataset mix --questions 10
```

> **Embeddings:** For best multihop results, download the pre-built KGs which use `nvidia-nv-embed-v2` (requires GPU with 16GB+ VRAM, auto-downloaded on first run). If you don't have a GPU, you can rebuild the KGs from scratch with `openai-small` instead: `python benchmarks/run.py --task multihop --dataset musique --force_build --embedding openai-small`.
>
> **Graph Types:** Multihop defaults to **passage-entity** graph (entities + passages + facts as nodes). UltraDomain/text2sql/summarization default to **entity** graph (classic KG). Override with `--graph_type`.

## Two Graph Types

QAFD-RAG supports two knowledge graph representations, both using the same Query-Aware Flow Diffusion algorithm:

| | **Entity Graph** | **Passage-Entity Graph** |
|---|---|---|
| **Nodes** | Entities + relationships | Entities + passages + facts (Gutiérrez et al., 2024) |
| **Extraction** | LLM entity/relationship extraction | OpenIE (NER + triple extraction) |
| **Edges** | Entity-entity relationships | Fact edges + passage edges + synonymy edges |
| **QAFD traversal** | Flow reaches entities, passages looked up after | Flow reaches passages directly as graph nodes |
| **Best for** | General QA, text2sql, summarization | Multi-hop reasoning |
| **Default tasks** | ultradomain, text2sql, summarization | multihop |

Both graph types use query-aware flow diffusion with the same parameters (alpha=1.5, epsilon=0.01, step_size=0.2, weight_scheme=multiply, linking_top_k=10).

## Benchmarks

QAFD-RAG is evaluated on four tasks. Each subsection below covers data setup, pre-built KGs, and how to run.

All benchmarks use the unified runner:

```bash
python benchmarks/run.py --task <task> --dataset <dataset> [options]
```

The runner automatically selects the appropriate graph type (passage-entity for multihop, entity for others). Override with `--graph_type`. A legacy CLI (`./run.sh`) is also available for entity-graph benchmarks.

Pre-built KGs for all tasks are available at [huggingface.co/tarzanagh/QAFD-RAG](https://huggingface.co/tarzanagh/QAFD-RAG). Downloading is **recommended** to avoid hours of build time and API costs.

### Multi-hop QA

| | |
|---|---|
| **Datasets** | MuSiQue, HotpotQA, 2WikiMultiHopQA |
| **Graph type** | passage-entity (default) |
| **Metrics** | F1, Exact Match |
| **Data** | Included in `data/multihop/` (from [osunlp](https://huggingface.co/osunlp)) |
| **Embedding** | `nvidia-nv-embed-v2` (GPU, 16GB+ VRAM) |

```bash
# Download pre-built KGs (recommended — saves hours + API costs)
huggingface-cli download tarzanagh/QAFD-RAG --include "kg/multihop/*" --local-dir .

# Run benchmark
python benchmarks/run.py --task multihop --dataset musique --questions 100
python benchmarks/run.py --task multihop --dataset hotpotqa --questions 100
python benchmarks/run.py --task multihop --dataset 2wikimultihopqa --questions 100

# No GPU? Rebuild KGs with openai-small instead
python benchmarks/run.py --task multihop --dataset musique --force_build --embedding openai-small

# Retrieval only (skip answer generation)
python benchmarks/run.py --task multihop --dataset musique --skip_qa
```

### UltraDomain

| | |
|---|---|
| **Datasets** | agriculture, biology, cs, finance, legal, math, medicine, mix, music, philosophy, physics |
| **Graph type** | entity (default) |
| **Metrics** | Quality scores (comprehensiveness, diversity, relevance, logicality, coherence) |
| **Data** | Auto-downloaded from [TommyChien/UltraDomain](https://huggingface.co/datasets/TommyChien/UltraDomain) at runtime |
| **Embedding** | `openai-small` (no GPU needed) |

```bash
# Download pre-built KGs (recommended)
huggingface-cli download tarzanagh/QAFD-RAG --include "kg/ultradomain/*" --local-dir .

# Run benchmark
python benchmarks/run.py --task ultradomain --dataset mix --questions 10

# Use passage-entity graph instead of entity graph
python benchmarks/run.py --task ultradomain --dataset mix --graph_type passage-entity --questions 10
```

### Text-to-SQL

| | |
|---|---|
| **Datasets** | Spider2-lite, Bird |
| **Graph type** | entity |
| **Metrics** | Schema retrieval precision / recall |
| **Data** | Example databases included in `data/text2sql/` (see [Data](#data) section) |
| **Embedding** | `openai-small` (no GPU needed) |

Two example databases are included with pre-generated DB summaries:
- **Pagila** (Spider2-lite) -- DVD rental store, 16 tables, SQLite
- **superhero** (Bird) -- superhero database, 10 tables, SQLite

For the full Spider2-lite benchmark, clone [Spider2](https://github.com/xlang-ai/Spider2) and copy databases into `data/text2sql/spider2-lite/sqlite/`.

```bash
# Run on included example databases
./run.sh text2sql --questions 5 --db Pagila                          # Spider2-lite (auto-detected)
./run.sh text2sql --questions 5 --db superhero                       # Bird (auto-detected)
./run.sh text2sql --benchmark bird --questions 5 --db superhero      # explicit benchmark selection

# Build KG only (no benchmark)
python benchmarks/run.py --task text2sql --dataset spider2-lite --build_only --db Pagila
```

> **Adding a new database:** Place your `.sqlite` file in `data/text2sql/spider2-lite/sqlite/<DB_Name>/`, then generate a DB summary (see [Generating DB Summaries](#generating-db-summaries)). The benchmark will auto-build the KG on first run.

### Summarization

| | |
|---|---|
| **Dataset** | SQuALITY |
| **Graph type** | entity |
| **Metrics** | BLEU, ROUGE, METEOR, quality scores |
| **Data** | Auto-downloaded from [pszemraj/SQuALITY-v1.3](https://huggingface.co/datasets/pszemraj/SQuALITY-v1.3) at runtime |

```bash
python benchmarks/run.py --task summarization --dataset squality --questions 50
```

### Common Options

| Option | Description |
|--------|-------------|
| `--task TASK` | `multihop`, `ultradomain`, `text2sql`, `summarization` |
| `--dataset NAME` | Dataset name (e.g., `musique`, `mix`, `spider2-lite`) |
| `--graph_type TYPE` | `passage-entity` or `entity` (auto-selected by task) |
| `--questions N` | Number of questions to evaluate |
| `--build_only` | Build KG only, skip benchmark |
| `--force_build` | Rebuild KG even if it exists |
| `--skip_qa` | Run retrieval only, skip QA (passage-entity) |
| `--embedding MODEL` | Embedding model (see [Configuration](#configuration)) |
| `--llm MODEL` | LLM model (`gpt-4o-mini`, `gpt-4o`, `gpt-5-nano`, `gpt-5-mini`, `gpt-5`, `gpt-oss-120b`) |
| `--alpha FLOAT` | QAFD alpha parameter (default: 2.0) |
| `--epsilon FLOAT` | QAFD convergence threshold (default: 0.01) |
| `--weight_scheme` | Query-aware edge weighting: `original`, `multiply`, `add` |

## Data

### Included Data

| Task | Directory | Contents | Size |
|------|-----------|----------|------|
| **Multi-hop QA** | `data/multihop/` | MuSiQue, HotpotQA, 2WikiMultiHopQA (full datasets from [osunlp](https://huggingface.co/osunlp)) | 134 MB |
| **UltraDomain** | `data/ultradomain/` | Auto-downloaded from [HuggingFace](https://huggingface.co/datasets/TommyChien/UltraDomain) at runtime | -- |
| **Text-to-SQL** | `data/text2sql/` | Example databases + question files (see below) | 9.6 MB |
| **Summarization** | `data/summarization/` | Auto-downloaded from [HuggingFace](https://huggingface.co/datasets/pszemraj/SQuALITY-v1.3) at runtime | -- |

### Text-to-SQL Data

Example databases with pre-generated DB summaries are included so you can run the text2sql benchmark out of the box:

| Backend | Database | Files | Description |
|---------|----------|-------|-------------|
| SQLite | **Pagila** | `Pagila.sqlite` + `Pagila_db_summary.json` | DVD rental store (16 tables) |
| SQLite | **superhero** | `superhero.sqlite` + `superhero_db_summary.json` | Superhero database (10 tables) |
| BigQuery | **san_francisco** | `san_francisco_bigquery_summary.json` | SF city data: bikeshare, crime, film locations, fire dept, street trees (8 tables, 118 columns, schema only) |
| Snowflake | AUSTIN | `AUSTIN_db_summary.json` | Austin 311 service requests (schema only) |

Question files: `spider2-lite.jsonl` (546 questions) and `bird.jsonl` (129 questions).

For the full Spider2-lite benchmark with all databases, clone [Spider2](https://github.com/xlang-ai/Spider2) and copy the SQLite files into `data/text2sql/spider2-lite/sqlite/`.

### Generating DB Summaries

To add a new SQLite database to the text2sql benchmark, generate a DB summary using the included CLI tool:

```bash
# Generate DB summary for a SQLite database
python -m src.indexing.extract_db_summary \
    --db-path data/text2sql/spider2-lite/sqlite/MyDB/MyDB.sqlite

# Specify custom output directory
python -m src.indexing.extract_db_summary \
    --db-path path/to/database.sqlite \
    --output data/text2sql/spider2-lite/sqlite/MyDB/
```

This produces a `<DB_Name>_db_summary.json` file containing:
- Table and column metadata (types, primary/foreign keys, nullable)
- Cardinality and distinct value counts per column
- Sample values for each column
- Inferred foreign key relationships

For BigQuery and Snowflake databases (schema summaries only, no raw data):

```bash
# BigQuery
python -m src.indexing.extract_db_summary_bigquery \
    --datasets project_id.dataset_id \
    --output-dir data/text2sql/spider2-lite/bigquery/my_dataset/

# Snowflake
python -m src.indexing.extract_db_summary_snowflake \
    --databases MY_DATABASE \
    --output-dir data/text2sql/spider2-lite/snowflake/MY_DATABASE/
```

## Project Structure

```
QAFD-RAG/
├── benchmarks/
│   ├── run.py                       # Unified benchmark runner (--graph_type, --task)
│   ├── multihop/                    # Multi-hop reasoning
│   ├── ultradomain/                 # General domain QA
│   ├── text2sql/                    # Natural language to SQL
│   └── summarization/               # Document summarization
├── src/
│   ├── QAFD_RAG.py                  # Entity graph pipeline (main class)
│   ├── base.py                      # Storage interfaces, QueryParam
│   ├── llm.py                       # LLM and embedding functions
│   ├── storage.py                   # KV, vector, and graph storage
│   ├── operate.py                   # Query operations
│   ├── evaluation.py                # Answer evaluation
│   ├── answering/                   # Query processing (entity graph)
│   │   ├── handler.py               # kg_query entry point
│   │   ├── context.py               # Context building (local/global/hybrid)
│   │   ├── clusters.py              # Flow diffusion clustering
│   │   └── text_units.py            # Text chunk retrieval
│   ├── retrievers/
│   │   ├── flow_diffusion.py        # QAFD on NetworkX (entity graph)
│   │   └── base.py                  # Retriever interface
│   ├── passage_entity/           # Passage-entity graph pipeline
│   │   ├── kg_builder.py            # OpenIE → igraph (entities + passages + facts)
│   │   ├── graph_adapter.py         # QAFD on igraph (passage-entity graph)
│   │   ├── retriever.py             # Fact reranking → seed selection → QAFD
│   │   ├── config.py                # Pipeline configuration
│   │   ├── openie.py                # NER + triple extraction
│   │   ├── embedding_store.py       # Parquet-backed vector store
│   │   ├── reranker.py              # LLM-based fact reranker
│   │   ├── prompts.py               # Prompt templates
│   │   ├── benchmark_runner.py      # Standalone runner
│   │   └── utils.py                 # Helpers
│   ├── indexing/                     # KG construction (entity graph)
│   │   ├── chunkers.py              # Token-based chunking
│   │   ├── extractors.py            # Entity/relationship extraction
│   │   ├── schema_builder.py        # Database schema KG builder
│   │   ├── extract_db_summary.py    # SQLite DB summary generator (CLI)
│   │   └── build_kg.py              # CLI KG builder
│   ├── embedding_models/            # Embedding model implementations
│   ├── prompts/                     # LLM prompt templates
│   ├── text2sql/                    # Text-to-SQL support
│   └── utils/                       # Helpers
├── data/                            # Datasets (included or auto-downloaded)
│   ├── multihop/                    # MuSiQue, HotpotQA, 2WikiMultiHopQA
│   ├── ultradomain/                 # Auto-downloaded at runtime
│   ├── text2sql/                    # Example DBs (Pagila, superhero) + questions
│   └── summarization/               # Auto-downloaded at runtime
├── kg/                              # Knowledge graphs (auto-generated or downloaded)
├── docs/figs/                       # Figures for README
└── results/                         # Benchmark results (JSON)
```

## Configuration

### Embedding Models

| Key | Model | Dimensions |
|-----|-------|-----------|
| `openai-small` | text-embedding-3-small | 1536 |
| `openai-large` | text-embedding-3-large | 3072 |
| `jina-v3` | jinaai/jina-embeddings-v3 | 1024 |
| `gritlm` | GritLM/GritLM-7B | 4096 |
| `nvidia-nv-embed-v2` | nvidia/NV-Embed-v2 | 4096 |

The default embedding is **`openai-small`** which requires `OPENAI_API_KEY`.

Use `--embedding <key>` to select a model:

```bash
./run.sh ultradomain --questions 10 --embedding jina-v3
```

### Local Embedding Models

Local embeddings (`jina-v3`, `gritlm`, `nvidia-nv-embed-v2`) run on your GPU and do not require an API key for embeddings. They are downloaded automatically from HuggingFace on first use.

**Requirements:**
- CUDA-capable GPU with sufficient VRAM (8GB+ recommended)
- Models are cached in `~/.cache/huggingface/`

**Usage with local embeddings (no OpenAI API needed for embeddings):**

```bash
# Use Jina v3 (1024-dim, lightweight, good quality)
./run.sh ultradomain --questions 10 --embedding jina-v3

# Use GritLM (4096-dim, unified embedding+generation)
./run.sh multihop --dataset musique --questions 10 --embedding gritlm

# Use NVIDIA NV-Embed-v2 (4096-dim, 32K context, high quality)
./run.sh ultradomain --questions 10 --embedding nvidia-nv-embed-v2
```

> **Note:** Even with local embeddings, an LLM API key is still required for entity extraction (KG building) and answer generation. Set `OPENAI_API_KEY` or use `--llm gpt-oss-120b` for a free open-source LLM.

### LLM Models

`gpt-4o-mini` (default), `gpt-4o`, `gpt-5-nano`, `gpt-5-mini`, `gpt-5`, `gpt-oss-120b` (free, local)


## How It Works

QAFD-RAG processes documents through a two-stage pipeline:

**Stage 1 -- Knowledge Graph Construction**

*Entity Graph:*
1. Documents are split into token-based chunks.
2. An LLM extracts entities and relationships from each chunk.
3. Entities become nodes, relationships become weighted edges.

*Passage-Entity Graph:*
1. Documents are split into passages (chunks).
2. OpenIE extracts named entities and (subject, predicate, object) triples.
3. Entities and passages are both graph nodes, connected by fact edges, passage-entity edges, and synonymy edges (entity pairs with cosine similarity > 0.8).

**Stage 2 -- Query-Aware Flow Diffusion**

1. **Seed Selection**: Query is matched to entities/facts via embedding similarity. An LLM reranker filters the most relevant facts.
2. **Flow Diffusion**: Mass is injected at seed nodes and propagated through the graph via push-relabel. Edge weights are dynamically adjusted based on each node's similarity to the query.
3. **Passage Ranking**: Nodes accumulate importance scores proportional to flow. In the passage-entity graph, passages are ranked directly. In the entity graph, associated text chunks are retrieved from top-ranked entities.
4. **LLM Answering**: Top passages are assembled into context and passed to an LLM for answer generation.

The flow diffusion algorithm is the key differentiator: rather than simple graph traversal or vector search alone, it combines graph structure with query relevance to find contextually important information that may be several hops away from the initial match.

## BibTeX

```bibtex
@inproceedings{zhou2026qafd,
  title={Query-Aware Flow Diffusion for Graph-Based RAG with Retrieval Guarantees},
  author={Zhou, Zhuoping and Ataee Tarzanagh, Davoud and Didari, Sima and Hu, Wenjun and Gutow, Baruch and Verkholyak, Oxana and Faraki, Masoud and Hao, Heng and Moon, Hankyu and Min, Seungjai},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```
