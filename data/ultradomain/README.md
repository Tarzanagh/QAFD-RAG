# UltraDomain Data

The UltraDomain benchmark data is loaded automatically from HuggingFace at runtime:

```
Dataset: TommyChien/UltraDomain
File:    mix.jsonl
```

No manual download required. The benchmark will fetch the data on first run via the `datasets` library.

## Format

Each item contains:
- `context`: The source document text used to build the knowledge graph
- `input`: The question to answer

## Usage

```bash
./run.sh ultradomain --questions 10
./run.sh ultradomain --build --max-documents 100
```
