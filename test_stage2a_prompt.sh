#!/bin/bash
# UniMotion Stage 2a Evaluation Script (with instruction template)
# Stage 2a uses fixed prompts from template_pretrain.json
# 269-dim representation, downsample_rate=4, LoRA-A+B
#
# Default replication times: M2T=3, T2M=1, Pred=1 (configured in the yaml/Python)
# The REP environment variable globally overrides the replication times of all tasks.
#
# Usage:
#   ./test_stage2a_prompt.sh                       # full evaluation of all tasks
#   ./test_stage2a_prompt.sh m2t                   # M2T only (default 3 reps)
#   ./test_stage2a_prompt.sh t2m                   # T2M only (default 1 rep)
#   ./test_stage2a_prompt.sh quick 128             # quick test (128 random samples, 1 rep per task)
#   ./test_stage2a_prompt.sh t2m,m2t               # T2M + M2T
#   ./test_stage2a_prompt.sh t2m /path/to/ckpt     # custom checkpoint
#   ./test_stage2a_prompt.sh all 6                 # full evaluation on 6 GPUs
#   REP=5 ./test_stage2a_prompt.sh                 # global override: 5 reps for all tasks
#   GPUS=1,2,3,4,5,6 NGPUS=6 BATCH_SIZE=16 ./test_stage2a_prompt.sh m2t  # multi-GPU
#
# Environment variables:
#   GPUS=0,1,2,3,4,5   GPUs to use (default: 0)
#   NGPUS=6             number of GPUs (when set, torchrun is used automatically)
#   BATCH_SIZE=32       batch size (default: 32)
#   REP                 global override of replication times (unset = per-task defaults: M2T=3, others=1)

# === Log backup ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
LOG_DIR="${PROJECT_ROOT}/wandb/all_log"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/test_stage2a_prompt_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
# ==================

set -e

# Default arguments
MODE=${1:-"all"}        # all, quick, t2m, m2t, pred, t2m,m2t, etc.
ARG2=${2:-""}           # checkpoint path / number of GPUs / quick-sample count
GPUS=${GPUS:-"0"}
NGPUS=${NGPUS:-""}
BATCH_SIZE=${BATCH_SIZE:-32}

CONFIG="configs/showo2_motion_test_stage2a.yaml"

# Parse mode
EXTRA_ARGS=""
QUICK_FLAG=""

if [ "$MODE" == "quick" ]; then
    QUICK_FLAG="test.quick=True"
    if [ -n "$ARG2" ] && [[ "$ARG2" =~ ^[0-9]+$ ]]; then
        QUICK_FLAG="$QUICK_FLAG test.num_quick_samples=$ARG2"
    fi
    echo "Mode: Quick Test"
elif [ "$MODE" == "all" ]; then
    if [ -n "$ARG2" ] && [[ "$ARG2" =~ ^[0-9]+$ ]]; then
        NGPUS="$ARG2"
    elif [ -n "$ARG2" ]; then
        EXTRA_ARGS="test.model_path=$ARG2"
    fi
    echo "Mode: Full Evaluation (all tasks)"
else
    # Either a task list (possibly comma-separated) or a checkpoint path
    if [[ "$MODE" =~ ^(t2m|m2t|pred|t2m,m2t|t2m,pred|m2t,pred|t2m,m2t,pred)$ ]]; then
        EXTRA_ARGS="test.tasks=[$MODE]"
        if [ -n "$ARG2" ]; then
            if [[ "$ARG2" =~ ^[0-9]+$ ]]; then
                NGPUS="$ARG2"
            else
                EXTRA_ARGS="$EXTRA_ARGS test.model_path=$ARG2"
            fi
        fi
        echo "Mode: Tasks [$MODE]"
    else
        # Treat it as a checkpoint path
        EXTRA_ARGS="test.model_path=$MODE"
        echo "Mode: Full Evaluation with custom checkpoint"
    fi
fi

# Build the command (global replication_times is only passed when REP is explicitly set)
CMD_ARGS="config=$CONFIG test.batch_size=$BATCH_SIZE $QUICK_FLAG $EXTRA_ARGS"
if [ -n "${REP+x}" ]; then
    CMD_ARGS="$CMD_ARGS test.replication_times=$REP"
    REP_INFO="$REP (global override)"
else
    REP_INFO="per-task default (M2T=3, T2M=1, Pred=1)"
fi

echo "=========================================="
echo "UniMotion Stage 2a Evaluation (with prompt)"
echo "Template: template_pretrain.json (fixed prompts)"
echo "=========================================="
echo "Config: $CONFIG"
echo "GPUs: $GPUS"
echo "Batch size: $BATCH_SIZE"
echo "Replication times: $REP_INFO"
echo "Args: $CMD_ARGS"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=$GPUS

if [ -n "$NGPUS" ] && [ "$NGPUS" -gt 1 ] 2>/dev/null; then
    echo "Running with torchrun ($NGPUS GPUs)..."
    torchrun --nproc_per_node=$NGPUS test_stage2_prompt.py $CMD_ARGS
else
    echo "Running with single GPU..."
    python test_stage2_prompt.py $CMD_ARGS
fi

echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
