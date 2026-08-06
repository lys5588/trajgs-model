"""
Compute Mean/Std for H36M 269-dim unified representation.

This script computes statistics for all 269 dimensions by combining train and test data,
similar to how HumanML3D computes its statistics from all available samples.

Unlike HumanML3D 269-dim script, this computes statistics for all 269 dims from scratch
since H36M doesn't have official 263-dim statistics.

Usage example (run from the repository root):
  python data_preparation/scripts/h36m_cal_mean_variance_269.py \
    --train-data data/UNI_MOTION_269/H36M/new_joint_vecs/h36m_train_269.npy \
    --test-data data/UNI_MOTION_269/H36M/new_joint_vecs/h36m_test_269.npy \
    --save-dir data/UNI_MOTION_269/H36M \
    --smooth-std
"""

from __future__ import annotations

import os
import argparse
from os.path import join as pjoin
from typing import Optional

import numpy as np
from tqdm import tqdm


def compute_mean_std_from_files(
    train_path: str,
    test_path: str,
    max_frames: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Compute mean/std for all 269 dims by combining train and test data.

    Args:
        train_path: Path to train data .npy file (frames, 269)
        test_path: Path to test data .npy file (frames, 269)
        max_frames: Maximum frames to use (for debugging)

    Returns:
        (mean269, std269, total_frames_used, total_frames_skipped)
    """
    print("Loading train data...")
    train_data = np.load(train_path)
    print(f"Train data shape: {train_data.shape}")

    print("Loading test data...")
    test_data = np.load(test_path)
    print(f"Test data shape: {test_data.shape}")

    # Check shapes
    if train_data.ndim != 2 or train_data.shape[-1] != 269:
        raise ValueError(f"Expected train data shape (T, 269), got {train_data.shape}")
    if test_data.ndim != 2 or test_data.shape[-1] != 269:
        raise ValueError(f"Expected test data shape (T, 269), got {test_data.shape}")

    # Concatenate train and test data
    print("Concatenating train and test data...")
    all_data = np.concatenate([train_data, test_data], axis=0)
    print(f"Combined data shape: {all_data.shape}")

    # Check for NaN values
    nan_mask = np.isnan(all_data).any(axis=1)
    if nan_mask.any():
        print(f"Warning: Found {nan_mask.sum()} frames with NaN values, skipping them")
        all_data = all_data[~nan_mask]
        print(f"Data shape after removing NaN: {all_data.shape}")

    # Limit frames if specified
    if max_frames is not None and all_data.shape[0] > max_frames:
        print(f"Limiting to {max_frames} frames for debugging")
        all_data = all_data[:max_frames]

    total_frames = all_data.shape[0]
    skipped_frames = nan_mask.sum()

    print(f"Computing mean and std over {total_frames} frames...")

    # Compute mean and std
    mean269 = all_data.mean(axis=0)
    std269 = all_data.std(axis=0)

    # Unlike HumanML3D (which smooths certain std dimensions by group means),
    # no std smoothing is applied for H36M; the raw std is used.

    return mean269.astype(np.float32), std269.astype(np.float32), total_frames, skipped_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Mean/Std for H36M 269-dim representation.")
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to train data .npy file (shape: frames, 269).",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test data .npy file (shape: frames, 269).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Output directory for Mean_269.npy and Std_269.npy.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Process at most N frames (debug).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output files if they exist.")
    parser.add_argument(
        "--smooth-std",
        action="store_true",
        help="Apply std smoothing similar to HumanML3D (set std to mean within groups).",
    )
    args = parser.parse_args()

    train_path = args.train_data
    test_path = args.test_data
    save_dir = args.save_dir

    # Check input files
    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Train data not found: {train_path}")
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"Test data not found: {test_path}")

    os.makedirs(save_dir, exist_ok=True)

    out_mean = pjoin(save_dir, "Mean_269.npy")
    out_std = pjoin(save_dir, "Std_269.npy")
    if (not args.overwrite) and (os.path.exists(out_mean) or os.path.exists(out_std)):
        raise FileExistsError(
            f"Output exists. Use --overwrite to replace.\n"
            f"- {out_mean}\n"
            f"- {out_std}"
        )

    # Compute statistics
    mean269, std269, used_frames, skipped_frames = compute_mean_std_from_files(
        train_path=train_path,
        test_path=test_path,
        max_frames=args.max_frames,
    )

    # Optional: apply smoothing like HumanML3D
    # Apply the same smoothing logic as HumanML3D cal_mean_variance.ipynb
    if args.smooth_std:
        print("Applying std smoothing (HumanML3D style)...")

        # HumanML3D smoothing for first 263 dims (joints_num = 22)
        joints_num = 22

        # root_rot_velocity (1 dim)
        std269[0:1] = std269[0:1].mean() / 1.0

        # root_linear_velocity (2 dims)
        std269[1:3] = std269[1:3].mean() / 1.0

        # root_y (1 dim)
        std269[3:4] = std269[3:4].mean() / 1.0

        # ric_data ((joints_num-1) * 3 dims)
        ric_start = 4
        ric_end = ric_start + (joints_num - 1) * 3
        std269[ric_start:ric_end] = std269[ric_start:ric_end].mean() / 1.0

        # rot_data ((joints_num-1) * 6 dims)
        rot_start = ric_end
        rot_end = rot_start + (joints_num - 1) * 6
        std269[rot_start:rot_end] = std269[rot_start:rot_end].mean() / 1.0

        # local_velocity (joints_num * 3 dims)
        vel_start = rot_end
        vel_end = vel_start + joints_num * 3
        std269[vel_start:vel_end] = std269[vel_start:vel_end].mean() / 1.0

        # foot contact (4 dims)
        foot_start = vel_end
        foot_end = foot_start + 4
        std269[foot_start:foot_end] = std269[foot_start:foot_end].mean() / 1.0

        # For the last 6 dims (global_orient_6d), apply smoothing like foot contact
        std269[263:269] = std269[263:269].mean() / 1.0

        # Verify dimensions
        expected_total = 8 + (joints_num - 1) * 9 + joints_num * 3 + 6  # 263 + 6 = 269
        assert expected_total == std269.shape[-1], f"Expected {expected_total} dims, got {std269.shape[-1]}"

    # Save results
    np.save(out_mean, mean269)
    np.save(out_std, std269)

    print("==========================================")
    print("H36M Mean/Std (269-dim)")
    print("==========================================")
    print(f"train_data:          {train_path}")
    print(f"test_data:           {test_path}")
    print(f"save_dir:            {save_dir}")
    print(f"used_frames:         {used_frames}")
    print(f"skipped_frames:      {skipped_frames}")
    print(f"max_frames:          {args.max_frames}")
    print(f"smooth_std:          {args.smooth_std}")
    print("------------------------------------------")
    print(f"Mean_269:            {out_mean}")
    print(f"Std_269:             {out_std}")
    print(f"shape(mean/std):     {mean269.shape} / {std269.shape}")
    print(f"mean range:          [{float(mean269.min()):.6g}, {float(mean269.max()):.6g}]")
    print(f"std range:           [{float(std269.min()):.6g}, {float(std269.max()):.6g}]")
    print("==========================================")


if __name__ == "__main__":
    main()