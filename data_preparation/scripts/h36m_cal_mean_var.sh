#!/bin/bash
# Compute Mean_269/Std_269 for H36M by combining the train and test frame
# arrays. --smooth-std applies HumanML3D-style per-group std smoothing (this
# is the configuration used for the released statistics).
#
# Usage:
#   bash data_preparation/scripts/h36m_cal_mean_var.sh [H36M_269_DIR]
# Default H36M_269_DIR: data/UNI_MOTION_269/H36M (relative to the repo root;
# must contain new_joint_vecs/h36m_{train,test}_269.npy)

set -e

# Run from the repository root
cd "$(dirname "$0")/../.."

DATA_DIR="${1:-data/UNI_MOTION_269/H36M}"

python data_preparation/scripts/h36m_cal_mean_variance_269.py \
  --train-data "${DATA_DIR}/new_joint_vecs/h36m_train_269.npy" \
  --test-data "${DATA_DIR}/new_joint_vecs/h36m_test_269.npy" \
  --save-dir "$DATA_DIR" \
  --smooth-std \
  --overwrite

# Reference output of our run (with --smooth-std), computed on the full-fps25
# arrays (780040 train + 274471 test = 1054511 frames) BEFORE the stride-5
# subsampling used at training time:
#   used_frames: 1054511
#   mean range:  [-0.664618, 1.31443]
#   std range:   [0, 0.38651]
# Note: the velocity dims have std=0 in single-frame mode; the training-side
# dataloaders guard against the resulting division by zero.
# The resulting Mean_269.npy/Std_269.npy are renamed to Mean.npy/Std.npy when
# assembling data/UNI_MOTION_269/H36M (see data_preparation/README.md).
# Precomputed copies ship in data_preparation/precomputed_stats/H36M/.
