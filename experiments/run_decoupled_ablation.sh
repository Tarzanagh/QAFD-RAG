#!/bin/bash
# Decoupled push ablation: does separating accumulation from routing
# make query-aware edge weights effective?
cd "$(dirname "$0")/.."
N=100
BASE="python src/hipporag_pipeline/benchmark_runner.py --task multihop --num_queries $N --embedding_model openai-small --skip_qa"

echo "======================================================================"
echo "  Decoupled Push Ablation ($N queries per dataset)"
echo "======================================================================"

for DATASET in musique hotpotqa 2wikimultihopqa; do
    echo ""
    echo "====== Dataset: $DATASET ======"

    echo "  [1] Single push, query-aware (decoupled)"
    $BASE --dataset $DATASET 2>&1 | grep -E "Recall@|QAFD:"

    echo "  [2] Single push, query-agnostic (decoupled, b=0 baseline)"
    $BASE --dataset $DATASET --qafd_weight_scheme none 2>&1 | grep -E "Recall@|QAFD:"

    echo "  [3] Batch push, query-aware (decoupled)"
    $BASE --dataset $DATASET --batch_push 2>&1 | grep -E "Recall@|QAFD:"

    echo "  [4] Batch push, query-agnostic (decoupled, b=0 baseline)"
    $BASE --dataset $DATASET --batch_push --qafd_weight_scheme none 2>&1 | grep -E "Recall@|QAFD:"
done

echo ""
echo "======================================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "======================================================================"
