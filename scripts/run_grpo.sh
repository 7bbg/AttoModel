#!/usr/bin/env bash
# ==============================================================================
# AttoModel: Post-Training Policy Alignment via GRPO + Verifiable Rewards (RLVR)
# ==============================================================================

set -eo pipefail

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

CHECKPOINT_PATH=${1:-"./checkpoints/pretrain_final.pt"}
OUTPUT_DIR="./checkpoints/grpo_aligned"

# Auto-detect RLVR prompts in root or data/
if [ -z "${PROMPTS_FILE}" ] || [ ! -f "${PROMPTS_FILE}" ]; then
    if [ -f "./rlvr_prompts.json" ]; then
        PROMPTS_FILE="./rlvr_prompts.json"
    elif [ -f "./data/rlvr_prompts.json" ]; then
        PROMPTS_FILE="./data/rlvr_prompts.json"
    fi
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "./logs"

PROMPT_ARGS=""
if [ -n "${PROMPTS_FILE}" ] && [ -f "${PROMPTS_FILE}" ]; then
    PROMPT_ARGS="--prompts_file ${PROMPTS_FILE}"
fi

echo "=============================================================================="
echo " Starting AttoModel GRPO RLVR Alignment"
echo " Initial Checkpoint: ${CHECKPOINT_PATH}"
echo " Group Size: G = 8 Candidate Samples"
echo " Verifiable Rewards: Python AST (+1.0), JSON/Tools (+2.0), Reflection (+1.0)"
if [ -f "${PROMPTS_FILE}" ]; then
    echo " Prompts File: ${PROMPTS_FILE}"
fi
echo "=============================================================================="

python -m training.grpo_rlvr \
    --checkpoint "${CHECKPOINT_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    ${PROMPT_ARGS} \
    --group_size 8 \
    --lr 5e-5 \
    --kl_beta 0.04 \
    --clip_eps 0.2 \
    "$@" 2>&1 | tee -a "./logs/grpo_$(date +%Y%m%d_%H%M%S).log"

echo "[*] GRPO Alignment Complete."
