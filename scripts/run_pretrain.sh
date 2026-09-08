#!/usr/bin/env bash
# ==============================================================================
# AttoModel: 18.5M Micro-Agent Language Model Pre-Training Launch Script
# 6-Hour Single RTX PRO 6000 Sprint & Multi-GPU Cluster Configuration
# ==============================================================================

set -eo pipefail

# Environment Defaults
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

# Configuration paths
CONFIG_MODEL="configs/model_18m.yaml"
CONFIG_TRAIN="configs/training_wsd.yaml"
CONFIG_DATA="configs/data_mix.yaml"
CHECKPOINT_DIR="./checkpoints"

# Auto-detect binary dataset in root or data/ directory
if [ -z "${DATA_BIN}" ]; then
    if [ -f "./curriculum_train.bin" ]; then
        DATA_BIN="./curriculum_train.bin"
    elif [ -f "./data/curriculum_train.bin" ]; then
        DATA_BIN="./data/curriculum_train.bin"
    fi
fi

mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "./logs"

DATA_BIN_ARG=""
if [ -n "${DATA_BIN}" ] && [ -f "${DATA_BIN}" ]; then
    DATA_BIN_ARG="--data_bin ${DATA_BIN}"
fi

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "0")

echo "=============================================================================="
echo " Starting AttoModel (18.5M) Pre-Training Sprint"
echo " Detected GPUs: ${NUM_GPUS}"
echo " Model Config: ${CONFIG_MODEL}"
echo " Schedule: Warmup-Stable-Decay (WSD) with Muon (2D) + AdamW (1D)"
if [ -n "${DATA_BIN}" ] && [ -f "${DATA_BIN}" ]; then
    echo " Dataset: High-Speed Binary Streaming (${DATA_BIN})"
else
    echo " Dataset: Dynamic Synthesis from ${CONFIG_DATA}"
fi
echo "=============================================================================="

if [ "${NUM_GPUS}" -gt "1" ]; then
    echo "[*] Launching Distributed Pretraining via torchrun (${NUM_GPUS} GPUs)..."
    torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port=29500 \
        -m training.pretrain \
        --model_config "${CONFIG_MODEL}" \
        --train_config "${CONFIG_TRAIN}" \
        --data_config "${CONFIG_DATA}" \
        --checkpoint_dir "${CHECKPOINT_DIR}" \
        ${DATA_BIN_ARG} \
        "$@" 2>&1 | tee -a "./logs/pretrain_$(date +%Y%m%d_%H%M%S).log"
else
    echo "[*] Launching Single-GPU High-Throughput Sprint..."
    python -m training.pretrain \
        --model_config "${CONFIG_MODEL}" \
        --train_config "${CONFIG_TRAIN}" \
        --data_config "${CONFIG_DATA}" \
        --checkpoint_dir "${CHECKPOINT_DIR}" \
        ${DATA_BIN_ARG} \
        "$@" 2>&1 | tee -a "./logs/pretrain_$(date +%Y%m%d_%H%M%S).log"
fi

echo "[*] Pre-Training Pipeline Finished."
