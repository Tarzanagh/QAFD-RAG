#!/bin/bash
# Run all query-awareness ablation experiments on musique (100 queries)
# Each run reuses the pre-built KG — only retrieval is re-run.

DATASET="musique"
N=100
BASE="python src/hipporag_pipeline/benchmark_runner.py --task multihop --dataset $DATASET --num_queries $N --embedding_model openai-small --skip_qa"

echo "======================================================================"
echo "  QAFD-RAG Query-Awareness Ablation (${DATASET}, ${N} queries)"
echo "======================================================================"

# Baseline (all flags = 0)
echo ""
echo "[1/7] Baseline (original, all QA flags = 0)"
$BASE --save_dir outputs_ablation/baseline 2>&1 | grep -E "Recall@|QAFD completed|Retrieval done"

# Phase 1: Query-aware sink capacity
echo ""
echo "[2/7] Phase 1: qa_sink_gamma=0.5"
$BASE --qa_sink_gamma 0.5 --save_dir outputs_ablation/sink_05 2>&1 | grep -E "Recall@|QAFD completed|Retrieval done"

echo ""
echo "[3/7] Phase 1: qa_sink_gamma=1.0"
$BASE --qa_sink_gamma 1.0 --save_dir outputs_ablation/sink_10 2>&1 | grep -E "Recall@|QAFD completed|Retrieval done"

# Phase 2: Query-aware warm start
echo ""
echo "[4/7] Phase 2: qa_warm_delta=0.5"
$BASE --qa_warm_delta 0.5 --save_dir outputs_ablation/warm_05 2>&1 | grep -E "Recall@|QAFD completed|Retrieval done"

# Phase 3: Post-diffusion reranking
echo ""
echo "[5/7] Phase 3: qa_post_lambda=0.5"
$BASE --qa_post_lambda 0.5 --save_dir outputs_ablation/post_05 2>&1 | grep -E "Recall@|QAFD completed|Retrieval done"

echo ""
echo "[6/7] Phase 3: qa_post_lambda=1.0"
$BASE --qa_post_lambda 1.0 --save_dir outputs_ablation/post_10 2>&1 | grep -E "Recall@|QAFD completed|Retrieval done"

# Combined best
echo ""
echo "[7/7] Combined: sink=0.5 + warm=0.5 + post=0.5"
$BASE --qa_sink_gamma 0.5 --qa_warm_delta 0.5 --qa_post_lambda 0.5 --save_dir outputs_ablation/combined 2>&1 | grep -E "Recall@|QAFD completed|Retrieval done"

echo ""
echo "======================================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "======================================================================"
