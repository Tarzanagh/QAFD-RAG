#!/bin/bash
# Alpha × QA sweep: find the sweet spot where decoupled QA edge weights help
cd "$(dirname "$0")/.."
N=100
BASE="python src/passage_entity/benchmark_runner.py --task multihop --num_queries $N --embedding_model openai-small --skip_qa"

echo "======================================================================"
echo "  Alpha x Query-Awareness Sweep (decoupled push, $N queries)"
echo "======================================================================"

for DATASET in musique hotpotqa 2wikimultihopqa; do
    echo ""
    echo "====== Dataset: $DATASET ======"

    for ALPHA in 2 3 5 8; do
        echo "  --- alpha=$ALPHA ---"

        echo "    QA-aware (original, a=1, b=0.5)"
        $BASE --dataset $DATASET --qafd_alpha $ALPHA 2>&1 | grep "Recall@10:\|Recall@100:"

        echo "    QA-agnostic (b=0)"
        $BASE --dataset $DATASET --qafd_alpha $ALPHA --qafd_weight_scheme none 2>&1 | grep "Recall@10:\|Recall@100:"
    done
done

echo ""
echo "======================================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "======================================================================"
