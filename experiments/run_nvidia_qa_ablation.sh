#!/bin/bash
# Query-awareness ablation on nvidia-nv-embed-v2 KGs (dense, 1.6M edges)
# Tests: QA-aware vs agnostic, different alpha/epsilon settings
cd "$(dirname "$0")/.."
N=100
BASE="python src/hipporag_pipeline/benchmark_runner.py --task multihop --num_queries $N --embedding_model nvidia-nv-embed-v2 --skip_qa"

echo "======================================================================"
echo "  Query-Awareness on nvidia KG (dense graph, $N queries)"
echo "  Started: $(date)"
echo "======================================================================"

for DATASET in musique hotpotqa 2wikimultihopqa; do
    echo ""
    echo "====== $DATASET ======"

    # Alpha=2, epsilon=0.01 (original settings)
    echo "  --- alpha=2, eps=0.01 ---"
    echo "  [QA-aware]"
    $BASE --dataset $DATASET --qafd_alpha 2.0 --qafd_epsilon 0.01 2>&1 | grep "Recall@10:\|Recall@100:\|QAFD:"
    echo "  [QA-agnostic]"
    $BASE --dataset $DATASET --qafd_alpha 2.0 --qafd_epsilon 0.01 --qafd_weight_scheme none 2>&1 | grep "Recall@10:\|Recall@100:\|QAFD:"

    # Alpha=3, epsilon=0.01
    echo "  --- alpha=3, eps=0.01 ---"
    echo "  [QA-aware]"
    $BASE --dataset $DATASET --qafd_alpha 3.0 --qafd_epsilon 0.01 2>&1 | grep "Recall@10:\|Recall@100:\|QAFD:"
    echo "  [QA-agnostic]"
    $BASE --dataset $DATASET --qafd_alpha 3.0 --qafd_epsilon 0.01 --qafd_weight_scheme none 2>&1 | grep "Recall@10:\|Recall@100:\|QAFD:"

    # Alpha=3, epsilon=0.001 (tighter convergence)
    echo "  --- alpha=3, eps=0.001 ---"
    echo "  [QA-aware]"
    $BASE --dataset $DATASET --qafd_alpha 3.0 --qafd_epsilon 0.001 2>&1 | grep "Recall@10:\|Recall@100:\|QAFD:"
    echo "  [QA-agnostic]"
    $BASE --dataset $DATASET --qafd_alpha 3.0 --qafd_epsilon 0.001 --qafd_weight_scheme none 2>&1 | grep "Recall@10:\|Recall@100:\|QAFD:"
done

echo ""
echo "======================================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Finished: $(date)"
echo "======================================================================"
