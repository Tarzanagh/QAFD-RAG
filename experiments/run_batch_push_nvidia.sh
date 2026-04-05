#!/bin/bash
# Batch push sweep on nvidia KG (dense, 1.6M edges)
# Goal: find config where QA-aware beats agnostic AND recall is high
cd "$(dirname "$0")/.."
N=100
BASE="python src/passage_entity/benchmark_runner.py --task multihop --num_queries $N --embedding_model nvidia-nv-embed-v2 --skip_qa"
LOG=experiments/results/batch_push_nvidia.log
mkdir -p experiments/results
exec > >(tee -a $LOG) 2>&1

echo "======================================================================"
echo "  Batch Push Sweep on nvidia KG ($N queries)"
echo "  Started: $(date)"
echo "======================================================================"

for DATASET in musique hotpotqa; do
    echo ""
    echo "====== $DATASET ======"

    # Baseline: single push, alpha=2, eps=0.01
    echo "  [1] single, a=2, eps=0.01, QA-aware"
    $BASE --dataset $DATASET --qafd_alpha 2 --qafd_epsilon 0.01 2>&1 | grep "Recall@"
    echo "  [2] single, a=2, eps=0.01, agnostic"
    $BASE --dataset $DATASET --qafd_alpha 2 --qafd_epsilon 0.01 --qafd_weight_scheme none 2>&1 | grep "Recall@"

    # Batch push, alpha=2, eps=0.01
    echo "  [3] batch, a=2, eps=0.01, QA-aware"
    $BASE --dataset $DATASET --qafd_alpha 2 --qafd_epsilon 0.01 --batch_push 2>&1 | grep "Recall@"
    echo "  [4] batch, a=2, eps=0.01, agnostic"
    $BASE --dataset $DATASET --qafd_alpha 2 --qafd_epsilon 0.01 --batch_push --qafd_weight_scheme none 2>&1 | grep "Recall@"

    # Batch push, alpha=2, eps=0.005
    echo "  [5] batch, a=2, eps=0.005, QA-aware"
    $BASE --dataset $DATASET --qafd_alpha 2 --qafd_epsilon 0.005 --batch_push 2>&1 | grep "Recall@"
    echo "  [6] batch, a=2, eps=0.005, agnostic"
    $BASE --dataset $DATASET --qafd_alpha 2 --qafd_epsilon 0.005 --batch_push --qafd_weight_scheme none 2>&1 | grep "Recall@"

    # Batch push, alpha=2, eps=0.003
    echo "  [7] batch, a=2, eps=0.003, QA-aware"
    $BASE --dataset $DATASET --qafd_alpha 2 --qafd_epsilon 0.003 --batch_push 2>&1 | grep "Recall@"
    echo "  [8] batch, a=2, eps=0.003, agnostic"
    $BASE --dataset $DATASET --qafd_alpha 2 --qafd_epsilon 0.003 --batch_push --qafd_weight_scheme none 2>&1 | grep "Recall@"

    # Batch push, alpha=3, eps=0.005
    echo "  [9] batch, a=3, eps=0.005, QA-aware"
    $BASE --dataset $DATASET --qafd_alpha 3 --qafd_epsilon 0.005 --batch_push 2>&1 | grep "Recall@"
    echo "  [10] batch, a=3, eps=0.005, agnostic"
    $BASE --dataset $DATASET --qafd_alpha 3 --qafd_epsilon 0.005 --batch_push --qafd_weight_scheme none 2>&1 | grep "Recall@"
done

echo ""
echo "======================================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Finished: $(date)"
echo "======================================================================"
