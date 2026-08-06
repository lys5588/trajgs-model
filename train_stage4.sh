#!/bin/bash
# Stage 4: Joint finetuning on all downstream tasks (m2m excluded)
# Usage: bash train_stage4.sh [GPU_ID...] [-- EXTRA_ARGS...] [LAUNCH_MODE]
#   LAUNCH_MODE:
#     - zero2 (default): Accelerate + DeepSpeed ZeRO-2
#     - zero3          : Accelerate + DeepSpeed ZeRO-3
#     - ddp            : Plain Accelerate DDP (no DeepSpeed)
# bash train_stage4.sh 2 5 6 zero2
# bash train_stage4.sh 2 5 6 zero3
# bash train_stage4.sh 0 1 2 3 4 ddp

set -e

# === Log backup ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
LOG_DIR="${PROJECT_ROOT}/wandb/all_log"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/stage4_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# ================= Configuration =================
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
LAUNCH_MODE=zero2
if [ ${#GPU_IDS[@]} -gt 0 ]; then
    LAST_ARG_INDEX=$((${#GPU_IDS[@]} - 1))
    LAST_ARG="${GPU_IDS[$LAST_ARG_INDEX]}"
    case "$LAST_ARG" in
        zero2|zero3|ddp)
            LAUNCH_MODE="$LAST_ARG"
            unset 'GPU_IDS[$LAST_ARG_INDEX]'
            GPU_IDS=("${GPU_IDS[@]}")
            ;;
    esac
fi

if [ ${#GPU_IDS[@]} -eq 0 ]; then
    GPU_IDS=(0 1 2 3 4 5)
fi

for gpu_id in "${GPU_IDS[@]}"; do
    if ! [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
        echo "Error: invalid GPU id '$gpu_id'."
        echo "Usage: bash train_stage4.sh [GPU_ID...] [LAUNCH_MODE]"
        exit 1
    fi
done

CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}
export CUDA_VISIBLE_DEVICES

PORT=29506
CONFIG="configs/showo2_motion_stage4.yaml"
ZERO3_CONFIG="accelerate_configs/deepspeed_zero3.yaml"
ZERO2_CONFIG="accelerate_configs/deepspeed_zero2.yaml"
# =================================================

echo "=========================================="
echo "Stage 4: All Downstream Tasks"
echo "Task ratios: t2m=0.28, m2t=0.14, pred=0.08, t2t=0.10, edit=0.15, img2pose=0.17, img2text=0.08"
echo "Training stage: lm_finetune (with t2t)"
echo "Learning rate: 3e-5"
echo "LoRA: A=ON (trainable), B=ON (frozen, preserved)"
echo "Launch mode: ${LAUNCH_MODE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "NUM_GPUS: ${NUM_GPUS}"
echo "PORT: ${PORT}"
echo "=========================================="

if [ "$NUM_GPUS" -eq 1 ]; then
    python train_motion.py \
        config=$CONFIG "${EXTRA_ARGS[@]}"
else
    ACCELERATE_ARGS=(
        --num_processes=$NUM_GPUS
        --mixed_precision=bf16
        --main_process_port=$PORT
    )

    case "$LAUNCH_MODE" in
        zero3)
            ACCELERATE_ARGS+=(--config_file "$ZERO3_CONFIG")
            ;;
        zero2)
            ACCELERATE_ARGS+=(--config_file "$ZERO2_CONFIG")
            ;;
        ddp)
            ACCELERATE_ARGS+=(--multi_gpu)
            ;;
        *)
            echo "Error: unsupported LAUNCH_MODE='$LAUNCH_MODE'. Use one of: zero3, zero2, ddp"
            exit 1
            ;;
    esac

    accelerate launch "${ACCELERATE_ARGS[@]}" \
        train_motion.py \
        config=$CONFIG "${EXTRA_ARGS[@]}"
fi
