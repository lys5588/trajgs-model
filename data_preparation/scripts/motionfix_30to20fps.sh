#!/bin/bash
# MotionFix frame-rate conversion script (30fps -> 20fps)
# ============================================================
# Two-step pipeline:
# 1. First downsample the raw SMPL parameters (joint_positions, rots, trans)
# 2. Then run convert_motionfix_to_269.py to produce the 269-dim representation
#
# This way velocities, foot contacts, etc. are naturally computed at 20fps.
#
# Usage:
#   bash data_preparation/scripts/motionfix_30to20fps.sh [MOTIONFIX_DIR]
# Default MOTIONFIX_DIR: data/MotionFix/motionfix-dataset (relative to the repo
# root), which must contain the official motionfix.pth.tar (30fps).

set -e

# Run from the repository root
cd "$(dirname "$0")/../.."

MOTIONFIX_DIR="${1:-data/MotionFix/motionfix-dataset}"

# Raw MotionFix data (30fps)
INPUT_ORIG="${MOTIONFIX_DIR}/motionfix.pth.tar"

# Intermediate file: downsampled SMPL data (20fps)
OUTPUT_SMPL_20FPS="${MOTIONFIX_DIR}/motionfix_20fps.pth.tar"

# Final output: 20fps 269-dim representation
OUTPUT_269_20FPS="${MOTIONFIX_DIR}/motionfix_20fps_269.pth.tar"

SRC_FPS=30
TGT_FPS=20

# Number of parallel workers (adjust to your CPU core count)
NUM_WORKERS=1

# Limit the number of samples for debugging (e.g. "10"); empty = all samples
MAX_SAMPLES=""

echo "=========================================="
echo "MotionFix frame-rate conversion: ${SRC_FPS}fps -> ${TGT_FPS}fps"
echo "Method: downsample SMPL first, then compute 269"
echo "=========================================="

# ============================================
# Step 1: downsample the SMPL parameters
# ============================================
echo ""
echo "[Step 1/2] Downsampling the SMPL parameters..."
echo "Input:  $INPUT_ORIG"
echo "Output: $OUTPUT_SMPL_20FPS"

EXTRA_ARGS=""
if [ -n "$MAX_SAMPLES" ]; then
    EXTRA_ARGS="--max-samples $MAX_SAMPLES"
fi

python data_preparation/scripts/resample_motionfix_smpl.py \
    --input "$INPUT_ORIG" \
    --output "$OUTPUT_SMPL_20FPS" \
    --src-fps $SRC_FPS \
    --tgt-fps $TGT_FPS \
    --num-workers $NUM_WORKERS \
    --overwrite \
    $EXTRA_ARGS

if [ $? -ne 0 ]; then
    echo "Error in Step 1!"
    exit 1
fi

# ============================================
# Step 2: compute the 269-dim representation
# ============================================
echo ""
echo "[Step 2/2] Computing the 269-dim representation..."
echo "Input:  $OUTPUT_SMPL_20FPS"
echo "Output: $OUTPUT_269_20FPS"

# --target-offset-mode humanml3d_default retargets to the HumanML3D 000021.npy
# skeleton so all three datasets share the same bone-length semantics (this is
# the configuration used for the released UNI_MOTION_269 data).
python data_preparation/scripts/convert_motionfix_to_269.py \
    --in-joblib "$OUTPUT_SMPL_20FPS" \
    --out-joblib "$OUTPUT_269_20FPS" \
    --out-mode smpl_semantic_t \
    --coord zup_to_yup \
    --reorder-joints none \
    --target-offset-mode humanml3d_default \
    --num-workers $NUM_WORKERS \
    --overwrite

if [ $? -ne 0 ]; then
    echo "Error in Step 2!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Conversion done!"
echo "=========================================="
echo "SMPL 20fps data: $OUTPUT_SMPL_20FPS"
echo "269 20fps data:  $OUTPUT_269_20FPS"
