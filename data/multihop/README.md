# Multi-hop QA Data

Datasets: MuSiQue, HotpotQA, 2WikiMultiHopQA (included in this directory).

**Graph type:** passage-entity (default) or entity (`--graph_type entity`)

## Pre-built KGs

Pre-built KGs use `nvidia-nv-embed-v2` (requires GPU, 16GB+ VRAM, auto-downloaded):

```bash
huggingface-cli download qafd/kg --repo-type dataset --include "multihop/*" --local-dir ./kg
```

## Usage

```bash
# Uses downloaded KG, or builds from scratch if not found
python benchmarks/run.py --task multihop --dataset musique --questions 100

# Rebuild with openai-small (no GPU needed)
python benchmarks/run.py --task multihop --dataset musique --force_build --embedding openai-small
```
