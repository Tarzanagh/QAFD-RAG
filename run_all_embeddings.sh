#!/bin/bash
# Run all benchmarks across all embedding models with gpt-oss-120b LLM
# Usage: nohup bash run_all_embeddings.sh > run_all_embeddings.log 2>&1 &

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

# Use cofd_sql conda env (has torch, transformers, sentence-transformers, etc.)
CONDA_ENV="cofd_sql"
PYTHON="conda run --live-stream -n ${CONDA_ENV} python"

# Set NVIDIA library paths (needed for torch with pip-installed CUDA libs)
NVIDIA_LIB="/home/davoud/miniconda3/envs/${CONDA_ENV}/lib/python3.10/site-packages/nvidia"
if [ -d "$NVIDIA_LIB" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_LIB}/cudnn/lib:${NVIDIA_LIB}/cublas/lib:${NVIDIA_LIB}/cuda_runtime/lib:${NVIDIA_LIB}/cuda_cupti/lib:${NVIDIA_LIB}/cuda_nvrtc/lib:${NVIDIA_LIB}/cufft/lib:${NVIDIA_LIB}/curand/lib:${NVIDIA_LIB}/cusolver/lib:${NVIDIA_LIB}/cusparse/lib:${NVIDIA_LIB}/nccl/lib:${NVIDIA_LIB}/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
fi

# Ensure OPENAI_API_KEY is set (needed even for gpt-oss, some embeddings use OpenAI)
if [ -z "${OPENAI_API_KEY:-}" ]; then
    if [ -f .env ]; then
        eval "$(grep -E '^OPENAI_API_KEY=' .env | head -1)"
        export OPENAI_API_KEY
    fi
fi

LLM="gpt-oss-120b"
QUESTIONS=3
MAX_DOCS=10
EMBEDDINGS=("openai-small" "openai-large" "jina-v3" "gritlm" "nvidia-nv-embed-v2")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="results/embedding_sweep_${TIMESTAMP}"
mkdir -p "$LOGDIR"

echo "============================================================"
echo "  QAFD-RAG Embedding Sweep"
echo "  LLM: ${LLM}"
echo "  Questions: ${QUESTIONS}"
echo "  Embeddings: ${EMBEDDINGS[*]}"
echo "  Conda env: ${CONDA_ENV}"
echo "  Log dir: ${LOGDIR}"
echo "  Started: $(date)"
echo "============================================================"

run_benchmark() {
    local task="$1"
    local embedding="$2"
    local flags="$3"   # comma-separated: "force,maxdocs" or "none"
    shift 3
    local extra_args=("$@")

    local label="${task}"
    if [ ${#extra_args[@]} -gt 0 ]; then
        label="${task}_${extra_args[*]}"
        label="${label// /_}"
        label="${label//--/}"
    fi

    local logfile="${LOGDIR}/${label}_${embedding}.log"

    local opt_flags=()
    if [[ "$flags" == *"force"* ]]; then
        opt_flags+=("--force-build")
    fi
    if [[ "$flags" == *"maxdocs"* ]]; then
        opt_flags+=("--max-documents" "$MAX_DOCS")
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "  [$(date +%H:%M:%S)] ${label} | embedding=${embedding}"
    echo "------------------------------------------------------------"

    if $PYTHON benchmarks/"${task}"/benchmark_"${task}".py \
        --questions "$QUESTIONS" \
        --embedding "$embedding" \
        --llm "$LLM" \
        "${opt_flags[@]}" \
        "${extra_args[@]}" \
        > "$logfile" 2>&1; then
        echo "  OK  (log: ${logfile})"
    else
        echo "  FAILED  (log: ${logfile})"
        echo "  Last 5 lines:"
        tail -5 "$logfile" | sed 's/^/    /'
    fi
}

for emb in "${EMBEDDINGS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  EMBEDDING: ${emb}"
    echo "  $(date)"
    echo "============================================================"

    # UltraDomain
    run_benchmark ultradomain "$emb" force,maxdocs

    # Multi-hop: MuSiQue
    run_benchmark multihop "$emb" force,maxdocs --dataset musique

    # Multi-hop: HotpotQA
    run_benchmark multihop "$emb" force,maxdocs --dataset hotpotqa

    # Multi-hop: 2WikiMultiHopQA
    run_benchmark multihop "$emb" force,maxdocs --dataset 2wikimultihopqa

    # Summarization
    run_benchmark summarization "$emb" force,maxdocs

    # Text2SQL (no --force-build or --max-documents, it auto-builds per DB)
    run_benchmark text2sql "$emb" none --db Pagila
done

echo ""
echo "============================================================"
echo "  ALL DONE: $(date)"
echo "  Results in: results/"
echo "  Logs in: ${LOGDIR}/"
echo "============================================================"
