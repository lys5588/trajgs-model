#!/bin/bash
# Stage 2a: T2M + M2T + Pred multi-task training (fixed prompts)
# Usage: bash train_stage2a.sh [GPU_ID...] [-- EXTRA_ARGS...]
# Example: bash train_stage2a.sh 0 1 2 5 6

# === Log backup ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
LOG_DIR="${PROJECT_ROOT}/wandb/all_log"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/stage2a_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
# ==================

GPU_IDS=()
EXTRA_ARGS=()
PARSING_EXTRA_ARGS=0
for arg in "$@"; do
    if [ "$arg" = "--" ]; then
        PARSING_EXTRA_ARGS=1
        continue
    fi
    if [ "$PARSING_EXTRA_ARGS" -eq 1 ]; then
        EXTRA_ARGS+=("$arg")
    else
        GPU_IDS+=("$arg")
    fi
done
if [ ${#GPU_IDS[@]} -eq 0 ]; then
    GPU_IDS=(0 1 2 3 4 5)
fi

for gpu_id in "${GPU_IDS[@]}"; do
    if ! [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
        echo "Error: invalid GPU id '$gpu_id'."
        echo "Usage: bash train_stage2a.sh [GPU_ID...]"
        exit 1
    fi
done

CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}
export CUDA_VISIBLE_DEVICES

echo "=========================================="
echo "Stage 2a: Multi-task Training (Fixed Prompts)"
echo "Task ratios: T2M=0.4, M2T=0.4, Pred=0.2"
echo "Learning rate: 5e-5"
echo "Template: template_pretrain.json (fixed prompts)"
echo "LoRA: A=ON (trainable), B=ON (trainable)"
echo "GPUs: ${NUM_GPUS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="

if [ "$NUM_GPUS" -eq 1 ]; then
    python train_motion.py config=configs/showo2_motion_stage2a.yaml "${EXTRA_ARGS[@]}"
else
    accelerate launch --multi_gpu \
        --num_processes=$NUM_GPUS \
        --mixed_precision=bf16 \
        --main_process_port=29511 \
        train_motion.py \
        config=configs/showo2_motion_stage2a.yaml "${EXTRA_ARGS[@]}"
fi
