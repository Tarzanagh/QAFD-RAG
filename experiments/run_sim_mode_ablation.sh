#!/bin/bash
# Edge weight ablation: similarity contrast functions across 3 datasets
# Tests whether sharper similarity contrast makes query-aware edge weights effective

cd "$(dirname "$0")/.."
N=100
BASE="python src/hipporag_pipeline/benchmark_runner.py --task multihop --num_queries $N --embedding_model openai-small --skip_qa"

echo "======================================================================"
echo "  Similarity Mode Ablation ($N queries per dataset)"
echo "======================================================================"

for DATASET in musique hotpotqa 2wikimultihopqa; do
    echo ""
    echo "======================================================================"
    echo "  Dataset: $DATASET"
    echo "======================================================================"

    for SIM in normalized relu relu_sq; do
        echo ""
        echo "  [$DATASET] sim_mode=$SIM"
        $BASE --dataset $DATASET --sim_mode $SIM 2>&1 | grep -E "Recall@|Retrieval done"
    done
done

echo ""
echo "======================================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "======================================================================"
