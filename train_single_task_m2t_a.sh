#!/bin/bash
# Single-task M2T (a): M2T-only training from the Show-o2 base model (fixed template)
# Usage: bash train_single_task_m2t_a.sh [NUM_GPUS] [-- EXTRA_ARGS...]
# Example: bash train_single_task_m2t_a.sh 6

set -e

# === Log backup ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
LOG_DIR="${PROJECT_ROOT}/wandb/all_log"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/single_task_m2t_a_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
# ==================

NUM_GPUS=${1:-6}
shift 2>/dev/null || true
EXTRA_ARGS=()
if [ "${1:-}" = "--" ]; then
    shift
    EXTRA_ARGS=("$@")
fi
PORT=${PORT:-29526}
CONFIG="configs/showo2_motion_single_task_m2t_a.yaml"

# Select GPUs via the CUDA_VISIBLE_DEVICES environment variable, e.g.:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash train_single_task_m2t_a.sh 6

echo "=========================================="
echo "Single-task M2T (a)"
echo "Task: m2t only"
echo "Template: instructions/template_m2t_fixed.json"
echo "Start from: showlab/show-o2-1.5B"
echo "Learning rate: 5e-5 | Batch(m2t): 2"
echo "LoRA: A=ON (trainable), B=ON (trainable)"
echo "Config: ${CONFIG}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-(not set, all GPUs visible)}"
echo "NUM_GPUS: ${NUM_GPUS}"
echo "PORT: ${PORT}"
echo "=========================================="

if [ "$NUM_GPUS" -eq 1 ]; then
    python train_motion.py config=$CONFIG "${EXTRA_ARGS[@]}"
else
    accelerate launch --multi_gpu \
        --num_processes=$NUM_GPUS \
        --mixed_precision=bf16 \
        --main_process_port=$PORT \
        train_motion.py \
        config=$CONFIG "${EXTRA_ARGS[@]}"
fi
