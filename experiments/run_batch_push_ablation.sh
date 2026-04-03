#!/bin/bash
# Batch push ablation: does batch push make edge weights matter?
# Compare: single vs batch, agnostic vs aware
cd /home/davoud/QAFD-RAG
N=100
BASE="python src/hipporag_pipeline/benchmark_runner.py --task multihop --num_queries $N --embedding_model openai-small --skip_qa"

echo "======================================================================"
echo "  Batch Push Ablation ($N queries per dataset)"
echo "======================================================================"

for DATASET in musique hotpotqa 2wikimultihopqa; do
    echo ""
    echo "====== Dataset: $DATASET ======"

    echo "  [1] Single push, query-aware (current default)"
    $BASE --dataset $DATASET 2>&1 | grep -E "Recall@|QAFD:"

    echo "  [2] Batch push, query-aware"
    $BASE --dataset $DATASET --batch_push 2>&1 | grep -E "Recall@|QAFD:"

    echo "  [3] Batch push, query-agnostic (b=0)"
    $BASE --dataset $DATASET --batch_push --qafd_weight_scheme none 2>&1 | grep -E "Recall@|QAFD:"
done

echo ""
echo "======================================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "======================================================================"
