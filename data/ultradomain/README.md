# UltraDomain Data

Datasets: agriculture, biology, cooking, cs, finance, legal, literature, mathematics, mix, music, philosophy, physics

Loaded from [TommyChien/UltraDomain](https://huggingface.co/datasets/TommyChien/UltraDomain) at runtime.

**Graph type:** entity (default) or passage-entity (`--graph_type passage-entity`)

## Pre-built KGs

Pre-built KGs use `openai-small` embedding (requires `OPENAI_API_KEY`):

```bash
huggingface-cli download qafd/kg --repo-type dataset --include "ultradomain/*" --local-dir ./kg
```

## Usage

```bash
# Uses downloaded KG, or builds from scratch if not found
python benchmarks/run.py --task ultradomain --dataset mix --questions 10

# Use passage-entity graph instead
python benchmarks/run.py --task ultradomain --dataset mix --graph_type passage-entity --questions 10
```
