#!/bin/bash
# Compute Mean_269/Std_269 for HumanML3D.
# The first 263 dims are taken directly from the official Mean.npy/Std.npy;
# only the last 6 dims (global_orient_6d) are computed from
# new_joint_vecs_269/*.npy. --smooth-last6 replaces the last-6 std values by
# their mean (this is the configuration used for the released statistics).
#
# Usage:
#   bash data_preparation/scripts/humanml3d_cal_mean_var.sh [HUMANML3D_ROOT]
# Default HUMANML3D_ROOT: data/HumanML3D (relative to the repo root)

set -e

# Run from the repository root
cd "$(dirname "$0")/../.."

HUMANML3D_ROOT="${1:-data/HumanML3D}"

python data_preparation/scripts/humanml3d_cal_mean_variance_269.py \
  --data-dir "${HUMANML3D_ROOT}/HumanML3D/new_joint_vecs_269" \
  --official-stats-dir "${HUMANML3D_ROOT}/HumanML3D" \
  --save-dir "${HUMANML3D_ROOT}/HumanML3D" \
  --smooth-last6 \
  --overwrite

# Reference output of our run (with --smooth-last6):
#   used_files:       29228
#   last6 mean range: [0, 1]
#   last6 std range:  [0.151159, 0.151159]
# The resulting Mean_269.npy/Std_269.npy are renamed to Mean.npy/Std.npy when
# assembling data/UNI_MOTION_269/HumanML3D (see data_preparation/README.md).
# Precomputed copies ship in data_preparation/precomputed_stats/HumanML3D/.
