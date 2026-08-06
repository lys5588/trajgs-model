"""
Compute Mean/Std for MotionFix 269-dim unified representation.

This script computes statistics for all 269 dimensions by combining all motion sequences
from train and test splits. Unlike H36M which has single .npy files, MotionFix data is
stored as individual .npy files for each motion sequence.

Usage example (run from the repository root):
  python data_preparation/scripts/motionfix_cal_mean_variance_269.py \
    --data-dir data/UNI_MOTION_269/MotionFix \
    --save-dir data/UNI_MOTION_269/MotionFix \
    --smooth-std
"""

from __future__ import annotations

import os
import argparse
from os.path import join as pjoin
from typing import Optional, List

import numpy as np
from tqdm import tqdm


def load_motionfix_data(
    data_dir: str,
    split_files: List[str],
    max_frames: Optional[int] = None,
) -> tuple[np.ndarray, int, int]:
    """
    Load MotionFix data from multiple .npy files.

    Args:
        data_dir: Root directory containing new_joint_vecs/ and split files
        split_files: List of split files (e.g., ['train.txt', 'test.txt'])
        max_frames: Maximum frames to use (for debugging)

    Returns:
        (all_data, total_frames_used, total_frames_skipped)
        all_data: Shape (total_frames, 269)
    """
    all_sequences = []

    # Read all split files to get list of motion files
    motion_files = []
    for split_file in split_files:
        split_path = pjoin(data_dir, split_file)
        with open(split_path, 'r') as f:
            for line in f:
                motion_files.append(line.strip())

    print(f"Found {len(motion_files)} motion sequences")

    # Load all motion sequences
    total_frames = 0
    skipped_frames = 0

    for motion_file in tqdm(motion_files, desc="Loading motion sequences"):
        # Convert "source/000362" to "new_joint_vecs/source/000362.npy"
        file_path = pjoin(data_dir, "new_joint_vecs", motion_file + ".npy")

        try:
            motion_data = np.load(file_path)
            all_sequences.append(motion_data)
            total_frames += motion_data.shape[0]
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            skipped_frames += 1

    print(f"Successfully loaded {len(all_sequences)} sequences")
    print(f"Total frames: {total_frames}, skipped: {skipped_frames}")

    # Concatenate all sequences
    if all_sequences:
        all_data = np.concatenate(all_sequences, axis=0)
    else:
        raise ValueError("No valid motion sequences found")

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

    total_frames_used = all_data.shape[0]
    total_frames_skipped = skipped_frames + nan_mask.sum()

    return all_data.astype(np.float32), total_frames_used, total_frames_skipped


def compute_mean_std_from_motionfix(
    data_dir: str,
    max_frames: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Compute mean/std for all 269 dims by combining train and test data from MotionFix.

    Args:
        data_dir: Directory containing new_joint_vecs/ and train.txt/test.txt
        max_frames: Maximum frames to use (for debugging)

    Returns:
        (mean269, std269, total_frames_used, total_frames_skipped)
    """
    print("Loading MotionFix data...")

    # Load data from both train and test splits
    all_data, total_frames_used, total_frames_skipped = load_motionfix_data(
        data_dir=data_dir,
        split_files=['train.txt', 'test.txt'],
        max_frames=max_frames,
    )

    print(f"Computing mean and std over {total_frames_used} frames...")

    # Compute mean and std
    mean269 = all_data.mean(axis=0)
    std269 = all_data.std(axis=0)

    # Unlike HumanML3D (which smooths certain std dimensions by group means),
    # no std smoothing is applied for MotionFix; the raw std is used.

    return mean269.astype(np.float32), std269.astype(np.float32), total_frames_used, total_frames_skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Mean/Std for MotionFix 269-dim representation.")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing new_joint_vecs/ and train.txt/test.txt.",
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

    data_dir = args.data_dir
    save_dir = args.save_dir

    # Check input directory
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    required_files = ['train.txt', 'test.txt']
    required_dirs = ['new_joint_vecs/source', 'new_joint_vecs/target']

    for file in required_files:
        if not os.path.isfile(pjoin(data_dir, file)):
            raise FileNotFoundError(f"Required file not found: {pjoin(data_dir, file)}")

    for dir_path in required_dirs:
        if not os.path.isdir(pjoin(data_dir, dir_path)):
            raise FileNotFoundError(f"Required directory not found: {pjoin(data_dir, dir_path)}")

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
    mean269, std269, used_frames, skipped_frames = compute_mean_std_from_motionfix(
        data_dir=data_dir,
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
    print("MotionFix Mean/Std (269-dim)")
    print("==========================================")
    print(f"data_dir:           {data_dir}")
    print(f"save_dir:           {save_dir}")
    print(f"used_frames:        {used_frames}")
    print(f"skipped_frames:     {skipped_frames}")
    print(f"max_frames:         {args.max_frames}")
    print(f"smooth_std:         {args.smooth_std}")
    print("------------------------------------------")
    print(f"Mean_269:           {out_mean}")
    print(f"Std_269:            {out_std}")
    print(f"shape(mean/std):    {mean269.shape} / {std269.shape}")
    print(f"mean range:         [{float(mean269.min()):.6g}, {float(mean269.max()):.6g}]")
    print(f"std range:          [{float(std269.min()):.6g}, {float(std269.max()):.6g}]")
    print("==========================================")


if __name__ == "__main__":
    main()