#!/bin/bash
# Overnight query-awareness experiments
# Tests decoupled push with alpha=3: QA-aware vs agnostic
# Across tasks, graph types, and embedding models
cd "$(dirname "$0")/.."
LOG=./experiments/results/overnight_qa_awareness.log
mkdir -p experiments/results
exec > >(tee -a $LOG) 2>&1
echo "======================================================================"
echo "  Overnight Query-Awareness Experiments"
echo "  Started: $(date)"
echo "======================================================================"

BASE_PE="python src/hipporag_pipeline/benchmark_runner.py"

# =====================================================================
# 1. PASSAGE-ENTITY: Multihop with OpenAI embedding (100q, alpha=3)
# =====================================================================
echo ""
echo "############ PASSAGE-ENTITY: MULTIHOP (openai-small) ############"

for DATASET in musique hotpotqa 2wikimultihopqa; do
    echo ""
    echo "====== PE / $DATASET / openai-small ======"

    echo "  [QA-aware] alpha=3"
    $BASE_PE --task multihop --dataset $DATASET --num_queries 100 --skip_qa \
        --embedding_model openai-small --qafd_alpha 3.0 2>&1 | grep "Recall@"

    echo "  [QA-agnostic] alpha=3"
    $BASE_PE --task multihop --dataset $DATASET --num_queries 100 --skip_qa \
        --embedding_model openai-small --qafd_alpha 3.0 --qafd_weight_scheme none 2>&1 | grep "Recall@"
done

# =====================================================================
# 2. PASSAGE-ENTITY: Multihop with LOCAL embeddings (50q, alpha=3)
# =====================================================================
echo ""
echo "############ PASSAGE-ENTITY: MULTIHOP (local embeddings) ############"

echo ""
echo "====== PE / musique / jina-v3 ======"
echo "  [QA-aware]"
$BASE_PE --task multihop --dataset musique --num_queries 50 --skip_qa \
    --embedding_model jina-v3 --qafd_alpha 3.0 2>&1 | grep "Recall@"
echo "  [QA-agnostic]"
$BASE_PE --task multihop --dataset musique --num_queries 50 --skip_qa \
    --embedding_model jina-v3 --qafd_alpha 3.0 --qafd_weight_scheme none 2>&1 | grep "Recall@"

echo ""
echo "====== PE / musique / nvidia-nv-embed-v2 ======"
echo "  [QA-aware]"
$BASE_PE --task multihop --dataset musique --num_queries 50 --skip_qa \
    --embedding_model nvidia-nv-embed-v2 --qafd_alpha 3.0 2>&1 | grep "Recall@"
echo "  [QA-agnostic]"
$BASE_PE --task multihop --dataset musique --num_queries 50 --skip_qa \
    --embedding_model nvidia-nv-embed-v2 --qafd_alpha 3.0 --qafd_weight_scheme none 2>&1 | grep "Recall@"

echo ""
echo "====== PE / hotpotqa / gritlm ======"
echo "  [QA-aware]"
$BASE_PE --task multihop --dataset hotpotqa --num_queries 50 --skip_qa \
    --embedding_model gritlm --qafd_alpha 3.0 2>&1 | grep "Recall@"
echo "  [QA-agnostic]"
$BASE_PE --task multihop --dataset hotpotqa --num_queries 50 --skip_qa \
    --embedding_model gritlm --qafd_alpha 3.0 --qafd_weight_scheme none 2>&1 | grep "Recall@"

# =====================================================================
# 3. PASSAGE-ENTITY: UltraDomain (10q, quality scores)
# =====================================================================
echo ""
echo "############ PASSAGE-ENTITY: ULTRADOMAIN ############"

for DATASET in mix agriculture cs; do
    echo ""
    echo "====== PE / ultradomain / $DATASET / openai-small ======"

    echo "  [QA-aware]"
    $BASE_PE --task ultradomain --dataset $DATASET --num_queries 10 \
        --embedding_model openai-small --qafd_alpha 3.0 2>&1 | grep -E "comprehensiveness|diversity|logicality|relevance|coherence|Overall"

    echo "  [QA-agnostic]"
    $BASE_PE --task ultradomain --dataset $DATASET --num_queries 10 \
        --embedding_model openai-small --qafd_alpha 3.0 --qafd_weight_scheme none 2>&1 | grep -E "comprehensiveness|diversity|logicality|relevance|coherence|Overall"
done

echo ""
echo "====== PE / ultradomain / mix / jina-v3 ======"
echo "  [QA-aware]"
$BASE_PE --task ultradomain --dataset mix --num_queries 10 \
    --embedding_model jina-v3 --qafd_alpha 3.0 2>&1 | grep -E "comprehensiveness|diversity|logicality|relevance|coherence|Overall"
echo "  [QA-agnostic]"
$BASE_PE --task ultradomain --dataset mix --num_queries 10 \
    --embedding_model jina-v3 --qafd_alpha 3.0 --qafd_weight_scheme none 2>&1 | grep -E "comprehensiveness|diversity|logicality|relevance|coherence|Overall"

# =====================================================================
# 4. ENTITY GRAPH: Multihop (100q, alpha=3 vs alpha=2)
# =====================================================================
echo ""
echo "############ ENTITY GRAPH: MULTIHOP ############"

for DATASET in musique hotpotqa; do
    echo ""
    echo "====== Entity / $DATASET ======"

    echo "  [alpha=3]"
    QAFD_CONDA_ENV=qafd-rag ./run.sh multihop --dataset $DATASET --questions 100 --alpha 3.0 2>&1 | grep -E "F1 Score|Exact Match"

    echo "  [alpha=2]"
    QAFD_CONDA_ENV=qafd-rag ./run.sh multihop --dataset $DATASET --questions 100 --alpha 2.0 2>&1 | grep -E "F1 Score|Exact Match"
done

# =====================================================================
# 5. BATCH PUSH + ALPHA=3 (passage-entity, 100q)
# =====================================================================
echo ""
echo "############ BATCH PUSH + ALPHA=3 ############"

for DATASET in musique hotpotqa; do
    echo ""
    echo "====== Batch / $DATASET / openai-small ======"

    echo "  [QA-aware, batch]"
    $BASE_PE --task multihop --dataset $DATASET --num_queries 100 --skip_qa \
        --embedding_model openai-small --qafd_alpha 3.0 --batch_push 2>&1 | grep "Recall@"

    echo "  [QA-agnostic, batch]"
    $BASE_PE --task multihop --dataset $DATASET --num_queries 100 --skip_qa \
        --embedding_model openai-small --qafd_alpha 3.0 --batch_push --qafd_weight_scheme none 2>&1 | grep "Recall@"
done

echo ""
echo "======================================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Finished: $(date)"
echo "======================================================================"
