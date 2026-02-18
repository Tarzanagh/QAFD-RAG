# Summarization Data

The summarization benchmark (SQuALITY) data is loaded automatically from HuggingFace at runtime:

```
Dataset: pszemraj/SQuALITY-v1.3
Split:   train
```

No manual download required. The benchmark will fetch the data on first run via the `datasets` library.

## Format

Each item contains:
- `document`: The full source document text
- `questions`: List of question objects, each with:
  - `question_text`: The question
  - `responses`: List of response objects with `response_text`
- `metadata.passage_id`: Unique passage identifier

## Usage

```bash
./run.sh summarization --questions 50
./run.sh summarization --build --max-documents 10
```
