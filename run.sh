#!/bin/bash
# QAFD-RAG Benchmark Runner
# Unified interface for all benchmarks

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")" || { echo "Error: Cannot cd to script directory"; exit 1; }

# Load API key from .env if not already set
if [ -z "${OPENAI_API_KEY:-}" ]; then
    if [ -f .env ]; then
        eval "$(grep -E '^OPENAI_API_KEY=' .env | head -1)"
        export OPENAI_API_KEY
    fi
    if [ -z "${OPENAI_API_KEY:-}" ]; then
        echo "Error: OPENAI_API_KEY not set. Export it or add it to a .env file."
        exit 1
    fi
fi

if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found. Activate your conda/venv environment first."
    exit 1
fi

# Show help
show_help() {
    echo "Usage: ./run.sh <task> [options]"
    echo ""
    echo "Tasks:"
    echo "  ultradomain      QA benchmark on UltraDomain dataset"
    echo "  text2sql         Text-to-SQL benchmark"
    echo "  summarization    Document summarization QA benchmark (e.g., squality)"
    echo "  multihop         Multi-hop QA benchmark (MuSiQue, HotpotQA, 2WikiMultiHopQA)"
    echo ""
    echo "Common Options (all tasks):"
    echo "  --questions N        Number of questions to benchmark"
    echo "  --build              Build KG only, don't run benchmark"
    echo "  --force-build        Force rebuild KG even if exists"
    echo "  --max-documents N    Max documents for KG building"
    echo "  --embedding MODEL    Embedding model (openai-small, openai-large, jina-v3)"
    echo "  --llm MODEL          LLM model (gpt-4o-mini, gpt-4o, gpt-oss-120b)"
    echo ""
    echo "Summarization-specific Options:"
    echo "  --dataset NAME       Dataset: squality (default: squality)"
    echo ""
    echo "Multihop-specific Options:"
    echo "  --dataset NAME       Dataset: musique, hotpotqa, 2wikimultihopqa"
    echo ""
    echo "Text2SQL-specific Options:"
    echo "  --db NAME            Database name (e.g., Pagila)"
    echo ""
    echo "Text2SQL End-to-End Pipeline:"
    echo "  Place your .sqlite file in data/text2sql/<DB>.sqlite"
    echo "  The DB summary (JSON) is auto-generated on first run."
    echo "  Or generate it manually:"
    echo "  python -m src.indexing.extract_db_summary --db-path data/text2sql/Pagila.sqlite"
    echo ""
    echo "Examples:"
    echo "  # Build KG"
    echo "  ./run.sh ultradomain --build --max-documents 100"
    echo "  ./run.sh summarization --build --max-documents 10"
    echo "  ./run.sh multihop --dataset musique --build --max-documents 500"
    echo "  ./run.sh multihop --dataset hotpotqa --build --max-documents 500"
    echo ""
    echo "  # Run benchmark"
    echo "  ./run.sh ultradomain --questions 10"
    echo "  ./run.sh summarization --questions 50"
    echo "  ./run.sh multihop --dataset musique --questions 100"
    echo "  ./run.sh multihop --dataset hotpotqa --questions 100"
    echo "  ./run.sh multihop --dataset 2wikimultihopqa --questions 100"
    echo "  ./run.sh text2sql --questions 5 --db Pagila"
    echo ""
    echo "  # Use local GPT-OSS model (free)"
    echo "  ./run.sh ultradomain --questions 10 --llm gpt-oss-120b"
    echo "  ./run.sh multihop --dataset musique --questions 10 --llm gpt-oss-120b"
}

case "${1:-help}" in
    ultradomain)
        shift
        python3 benchmarks/ultradomain/benchmark_ultradomain.py "$@"
        ;;
    text2sql)
        shift
        python3 benchmarks/text2sql/benchmark_text2sql.py "$@"
        ;;
    summarization)
        shift
        python3 benchmarks/summarization/benchmark_summarization.py "$@"
        ;;
    multihop)
        shift
        python3 benchmarks/multihop/benchmark_multihop.py "$@"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Error: Unknown task '${1}'. Run './run.sh --help' for usage."
        exit 1
        ;;
esac
