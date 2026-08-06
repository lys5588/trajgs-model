#!/bin/bash
# HumanML3D dataset conversion script
# Combines HumanML3D joints + official new_joint_vecs (263) into
# new_joint_vecs_269 (269 = 263 + 6).
#
# Usage:
#   bash data_preparation/scripts/humanml3d_263_to_269.sh [HUMANML3D_ROOT]
# Default HUMANML3D_ROOT: data/HumanML3D (relative to the repo root)

set -e

# Run from the repository root
cd "$(dirname "$0")/../.."

HUMANML3D_ROOT="${1:-data/HumanML3D}"

echo "=========================================="
echo "HumanML3D -> 269 (263 + 6) conversion"
echo "=========================================="
echo "HUMANML3D_ROOT: $HUMANML3D_ROOT"
echo "=========================================="

# Tip: add "--max-files 20" for a quick debug run.
python data_preparation/scripts/convert_humanml3d_to_269.py \
  --humanml3d-root "$HUMANML3D_ROOT" \
  --overwrite \
  --strict

echo ""
echo "Conversion done!"
echo "Output directory: ${HUMANML3D_ROOT}/HumanML3D/new_joint_vecs_269"

# Reference: in our full run all 29228 files passed the allclose consistency
# check against the official 263-dim files (ok_allclose = 29228, failed = 0).
