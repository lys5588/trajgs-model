#!/bin/bash
# Human3.6M dataset conversion script
# Converts WHAM/TCMR-format Human3.6M data into the 269-dim unified
# representation (single-frame mode).
#
# Usage:
#   bash data_preparation/scripts/h36m_smpl_to_269.sh <train|test> [DATA_ROOT]
# DATA_ROOT defaults to "data" (relative to the repo root) and must contain:
#   train: ${DATA_ROOT}/_DATA_for_WHAM/human36m_train_vit.pth   (WHAM parsed data)
#   test:  ${DATA_ROOT}/_DATA_for_TCMR/h36m_test_25fps_db.pt    (TCMR preprocessed data)
# plus the SMPLH body models at ${DATA_ROOT}/body_models/smplh/.

set -e

# Run from the repository root
cd "$(dirname "$0")/../.."

split="${1:?Usage: h36m_smpl_to_269.sh <train|test> [DATA_ROOT]}"
DATA_ROOT="${2:-data}"

# Rotation writing strategy (pixel alignment / fair comparison vs semantic consistency)
# - none:   do not keep SMPL rotations (269 fully uses the normalized semantics
#           of process_mesh_reconstruction)
# - global: only write the SMPL global_orient into 263:269; rot_data (67:193)
#           keeps the normalized semantics (a reasonable compromise)
# - full:   write both SMPL global_orient + body_pose (overrides 263:269 and
#           67:193; best for "recoverable SMPL pose", largest semantic gap)
# NOTE: "full" is the configuration used for the released UNI_MOTION_269/H36M data.
ROTATION_MODE="full"

SMPL_MODEL_DIR="${DATA_ROOT}/body_models/smplh"

if [ "$split" == "train" ]; then
    INPUT="${DATA_ROOT}/_DATA_for_WHAM/human36m_train_vit.pth"
    OUTPUT="${DATA_ROOT}/_DATA_for_WHAM/human36m_train_269_${ROTATION_MODE}.pth"
elif [ "$split" == "test" ]; then
    INPUT="${DATA_ROOT}/_DATA_for_TCMR/h36m_test_25fps_db.pt"
    OUTPUT="${DATA_ROOT}/_DATA_for_TCMR/h36m_test_25fps_db_269_${ROTATION_MODE}.pt"
else
    echo "Unknown split=$split (expected: train|test)"
    exit 1
fi

echo "=========================================="
echo "Human3.6M -> 269-dim unified representation"
echo "=========================================="
echo "Split:  $split"
echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo "Rotation mode: $ROTATION_MODE"
echo "=========================================="

if [ "$ROTATION_MODE" = "none" ]; then
    EXTRA_FLAGS="--no-preserve-rotation"
elif [ "$ROTATION_MODE" = "global" ]; then
    EXTRA_FLAGS="--preserve-global-orient-only"
elif [ "$ROTATION_MODE" = "full" ]; then
    EXTRA_FLAGS=""
else
    echo "Unknown ROTATION_MODE=$ROTATION_MODE (expected: none|global|full)"
    exit 1
fi

python data_preparation/scripts/convert_human36m_to_269.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --smpl-model "$SMPL_MODEL_DIR" \
    $EXTRA_FLAGS \
    --batch-size 512

echo ""
echo "Conversion done!"
echo "Output file: $OUTPUT"
