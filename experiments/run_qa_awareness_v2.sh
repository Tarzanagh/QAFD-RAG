#!/bin/bash
# Query-Awareness Ablation V2: warm walk, accumulation, more steps
cd "$(dirname "$0")/.."
N=100
BASE="python src/hipporag_pipeline/benchmark_runner.py --task multihop --num_queries $N --embedding_model openai-small --skip_qa"

echo "======================================================================"
echo "  Query-Awareness V2 Ablation ($N queries per dataset)"
echo "======================================================================"

for DATASET in musique hotpotqa 2wikimultihopqa; do
    echo ""
    echo "====== Dataset: $DATASET ======"

    echo "  [1] Baseline (original)"
    $BASE --dataset $DATASET 2>&1 | grep -E "Recall@|Retrieval done"

    echo "  [2] QA Warm Walk (edge weights in warm-start)"
    $BASE --dataset $DATASET --qa_warm_walk 2>&1 | grep -E "Recall@|Retrieval done"

    echo "  [3] QA Warm Walk + 5 steps"
    $BASE --dataset $DATASET --qa_warm_walk --qa_warm_steps 5 2>&1 | grep -E "Recall@|Retrieval done"

    echo "  [4] QA Accumulation gamma=0.5"
    $BASE --dataset $DATASET --qa_accum_gamma 0.5 2>&1 | grep -E "Recall@|Retrieval done"

    echo "  [5] QA Warm Walk + Accumulation gamma=0.5"
    $BASE --dataset $DATASET --qa_warm_walk --qa_accum_gamma 0.5 2>&1 | grep -E "Recall@|Retrieval done"
done

echo ""
echo "======================================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "======================================================================"
