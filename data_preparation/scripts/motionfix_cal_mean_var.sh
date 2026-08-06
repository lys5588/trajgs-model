#!/bin/bash
# Compute Mean_269/Std_269 for MotionFix over all frames of the source+target
# sequences listed in train.txt/test.txt. --smooth-std applies HumanML3D-style
# per-group std smoothing (this is the configuration used for the released
# statistics).
#
# Usage:
#   bash data_preparation/scripts/motionfix_cal_mean_var.sh [MOTIONFIX_269_DIR]
# Default MOTIONFIX_269_DIR: data/UNI_MOTION_269/MotionFix (relative to the
# repo root; must contain new_joint_vecs/{source,target}/ and train.txt/test.txt)

set -e

# Run from the repository root
cd "$(dirname "$0")/../.."

DATA_DIR="${1:-data/UNI_MOTION_269/MotionFix}"

python data_preparation/scripts/motionfix_cal_mean_variance_269.py \
  --data-dir "$DATA_DIR" \
  --save-dir "$DATA_DIR" \
  --smooth-std \
  --overwrite

# Reference output of our run (with --smooth-std):
#   loaded sequences: 12800 (source + target)
#   used_frames:      1416337
#   mean range:       [-0.64125, 1.43296]
#   std range:        [0.00683833, 0.24307]
# The resulting Mean_269.npy/Std_269.npy are renamed to Mean.npy/Std.npy when
# assembling data/UNI_MOTION_269/MotionFix (see data_preparation/README.md).
# Precomputed copies ship in data_preparation/precomputed_stats/MotionFix/.
