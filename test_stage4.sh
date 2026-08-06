#!/bin/bash
# UniMotion Stage 4 Evaluation Script
# Supported tasks: t2m,m2t,pred,t2t,edit,img2pose,img2text,mgie
# Note: the img2pose/img2text/mgie evaluators are not included in this release;
# running those tasks exits with a clear error message.
#
# Usage:
#   ./test_stage4.sh
#   ./test_stage4.sh quick 128    # quick test (128 random samples)
#   ./test_stage4.sh t2t
#   ./test_stage4.sh t2m,m2t
#   ./test_stage4.sh all /path/to/checkpoint
#   GPUS=0,1,2,3 NGPUS=4 ./test_stage4.sh all
#   REP=3 ./test_stage4.sh t2t

# === Log backup ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
LOG_DIR="${PROJECT_ROOT}/wandb/all_log"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/test_stage4_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
# ==================

set -e

MODE=${1:-"all"}
ARG2=${2:-""}
GPUS=${GPUS:-"0"}
NGPUS=${NGPUS:-""}
BATCH_SIZE=${BATCH_SIZE:-8}

CONFIG="configs/showo2_motion_test_stage4.yaml"

is_task_list() {
    local task_csv="$1"
    IFS=',' read -ra parts <<< "$task_csv"
    if [ "${#parts[@]}" -eq 0 ]; then
        return 1
    fi
    for t in "${parts[@]}"; do
        case "$t" in
            t2m|m2t|pred|t2t|edit|img2pose|img2text|mgie)
                ;;
            *)
                return 1
                ;;
        esac
    done
    return 0
}

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
elif is_task_list "$MODE"; then
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
    EXTRA_ARGS="test.model_path=$MODE"
    echo "Mode: Full Evaluation with custom checkpoint"
fi

CMD_ARGS="config=$CONFIG test.batch_size=$BATCH_SIZE $QUICK_FLAG $EXTRA_ARGS"
if [ -n "${REP+x}" ]; then
    CMD_ARGS="$CMD_ARGS test.replication_times=$REP"
    REP_INFO="$REP (global override)"
else
    REP_INFO="per-task defaults"
fi

echo "=========================================="
echo "UniMotion Stage 4 Evaluation"
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
    torchrun --nproc_per_node=$NGPUS test_stage4.py $CMD_ARGS
else
    echo "Running with single GPU..."
    python test_stage4.py $CMD_ARGS
fi

echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
