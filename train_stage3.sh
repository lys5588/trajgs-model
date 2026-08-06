#!/bin/bash
# Stage 3: Image-to-Pose + Image-to-Text training (H36M dataset)
# Usage: bash train_stage3.sh [GPU_ID...] [-- EXTRA_ARGS...]
# Example: bash train_stage3.sh 0 1 2 5 6

# === Log backup ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Optional: use a HuggingFace mirror if huggingface.co is unreachable from your network.
# export HF_ENDPOINT="https://hf-mirror.com"

LOG_DIR="${PROJECT_ROOT}/wandb/all_log"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/stage3_$(date +%Y%m%d_%H%M%S).log"
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
        echo "Usage: bash train_stage3.sh [GPU_ID...]"
        exit 1
    fi
done

CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}
export CUDA_VISIBLE_DEVICES

echo "=========================================="
echo "Stage 3: Image-related Tasks (H36M)"
echo "Task ratios: Image-to-Pose=0.8, Image-to-Text=0.2"
echo "Learning rate: 6e-5"
echo "LoRA: A=ON (trainable), B=ON (trainable)"
echo "Trainable: motion layers, image_embedder_und, fusion_proj, LoRA-A/B"
echo "Frozen: Qwen, SigLIP encoder, image_embedder_gen, image diffusion head"
echo "GPUs: ${NUM_GPUS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="

if [ "$NUM_GPUS" -eq 1 ]; then
    python train_motion.py config=configs/showo2_motion_stage3.yaml "${EXTRA_ARGS[@]}"
else
    accelerate launch --multi_gpu \
        --num_processes=$NUM_GPUS \
        --num_machines=1 \
        --dynamo_backend=no \
        --mixed_precision=bf16 \
        --main_process_port=29504 \
        train_motion.py \
        config=configs/showo2_motion_stage3.yaml "${EXTRA_ARGS[@]}"
fi
