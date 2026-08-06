#!/bin/bash
# Stage 1 T2M Evaluation
#
# Usage:
#   bash test_stage1.sh                    # single GPU, full evaluation
#   bash test_stage1.sh --quick            # single GPU, quick test (random subset)
#   bash test_stage1.sh --quick --quick-samples 512
#   bash test_stage1.sh --multi            # multi-GPU, full evaluation
#   bash test_stage1.sh --multi --quick    # multi-GPU, quick test
#
# Environment variables:
#   GPUS=0,1,2,3   GPUs to use (default: 0; multi-GPU mode uses all listed GPUs)

# === Log backup ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
LOG_DIR="${PROJECT_ROOT}/wandb/all_log"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/test_stage1_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
# ==================

# Default arguments
MULTI_GPU=false
QUICK_TEST=false
QUICK_SAMPLES=512

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --multi)
            MULTI_GPU=true
            shift
            ;;
        --quick)
            QUICK_TEST=true
            shift
            ;;
        --quick-samples)
            QUICK_SAMPLES="$2"
            shift 2
            ;;
        --quick-samples=*)
            QUICK_SAMPLES="${1#*=}"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# GPU selection (script parameter; default: GPU 0)
GPUS=${GPUS:-"0"}
export CUDA_VISIBLE_DEVICES=$GPUS
NUM_GPUS=$(echo "$GPUS" | awk -F',' '{print NF}')

echo "=========================================="
echo "Stage 1 T2M Evaluation"
echo "=========================================="
echo "Multi-GPU: $MULTI_GPU"
echo "Quick test: $QUICK_TEST"
echo "Quick samples: $QUICK_SAMPLES"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Build the command
CONFIG_FILE="configs/showo2_motion_test_stage1.yaml"
EXTRA_ARGS=""

if [ "$QUICK_TEST" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS test.quick=True test.num_quick_samples=$QUICK_SAMPLES"
fi

# Run the evaluation
if [ "$MULTI_GPU" = true ]; then
    echo "Running multi-GPU evaluation with $NUM_GPUS GPUs..."
    torchrun --nproc_per_node=$NUM_GPUS test_stage1.py \
        config=$CONFIG_FILE \
        $EXTRA_ARGS
else
    echo "Running single-GPU evaluation..."
    python test_stage1.py \
        config=$CONFIG_FILE \
        $EXTRA_ARGS
fi
