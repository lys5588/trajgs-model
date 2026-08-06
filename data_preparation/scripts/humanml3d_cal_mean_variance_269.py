"""
Compute Mean/Std for HumanML3D 269-dim unified representation (263 + 6).
========================================================================

Key requirement (as requested):
- The first 263 dims are *exactly* the official HumanML3D 263-dim representation,
  so their Mean/Std must be taken directly from the official files:
  `DATASETS/HumanML3D/HumanML3D/Mean.npy` and `Std.npy`.
- Only the last 6 dims (global_orient_6d) are computed from `new_joint_vecs_269/*.npy`.

This script computes statistics for the last 6 dims using a streaming accumulator
(sum / sumsq) to avoid loading everything into memory, and then concatenates:

  Mean_269 = [Mean_263, Mean_last6]
  Std_269  = [Std_263,  Std_last6]

Usage example (run from the repository root):
  python data_preparation/scripts/humanml3d_cal_mean_variance_269.py \
    --data-dir data/HumanML3D/HumanML3D/new_joint_vecs_269 \
    --official-stats-dir data/HumanML3D/HumanML3D \
    --save-dir data/HumanML3D/HumanML3D \
    --smooth-last6
"""

from __future__ import annotations

import os
import argparse
from os.path import join as pjoin
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm


def _stream_mean_std_last6(
    data_dir: str,
    max_files: Optional[int] = None,
    skip_nan: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Compute mean/std for the last 6 dims over all frames in `data_dir`.

    Returns:
      (mean6, std6, used_files, skipped_files)
    """
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])
    if max_files is not None:
        files = files[: max_files]

    cnt = 0
    sum_ = np.zeros((6,), dtype=np.float64)
    sumsq = np.zeros((6,), dtype=np.float64)
    used_files = 0
    skipped_files = 0

    for f in tqdm(files, desc="Mean/Std last6", dynamic_ncols=True):
        path = pjoin(data_dir, f)
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[-1] != 269:
            raise ValueError(f"Expected (T, 269) npy, got {arr.shape} @ {path}")

        tail = arr[:, 263:269]
        if skip_nan and np.isnan(tail).any():
            skipped_files += 1
            continue

        tail64 = tail.astype(np.float64, copy=False)
        cnt += int(tail64.shape[0])
        sum_ += tail64.sum(axis=0)
        sumsq += np.square(tail64).sum(axis=0)
        used_files += 1

    if cnt == 0:
        raise RuntimeError("No valid frames found to compute last-6 stats (cnt == 0).")

    mean6 = sum_ / float(cnt)
    var6 = sumsq / float(cnt) - np.square(mean6)
    var6 = np.maximum(var6, 0.0)  # numerical safety
    std6 = np.sqrt(var6)
    return mean6.astype(np.float32), std6.astype(np.float32), used_files, skipped_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Mean/Std for HumanML3D 269-dim (263 + 6).")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing `new_joint_vecs_269/*.npy` (each of shape (T, 269)).",
    )
    parser.add_argument(
        "--official-stats-dir",
        type=str,
        required=True,
        help="Directory containing official `Mean.npy` and `Std.npy` for 263 dims.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Output directory for `Mean_269.npy` and `Std_269.npy`.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Process at most N files (debug).")
    parser.add_argument("--no-skip-nan", action="store_true", help="Do not skip files containing NaNs.")
    parser.add_argument(
        "--smooth-last6",
        action="store_true",
        help="If set, replace last 6 std values by their mean (optional scale smoothing).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output files if they exist.")
    args = parser.parse_args()

    data_dir = args.data_dir
    official_stats_dir = args.official_stats_dir
    save_dir = args.save_dir

    mean263_path = pjoin(official_stats_dir, "Mean.npy")
    std263_path = pjoin(official_stats_dir, "Std.npy")
    if not os.path.isfile(mean263_path):
        raise FileNotFoundError(f"Official Mean.npy not found: {mean263_path}")
    if not os.path.isfile(std263_path):
        raise FileNotFoundError(f"Official Std.npy not found: {std263_path}")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data-dir not found: {data_dir}")
    os.makedirs(save_dir, exist_ok=True)

    out_mean = pjoin(save_dir, "Mean_269.npy")
    out_std = pjoin(save_dir, "Std_269.npy")
    if (not args.overwrite) and (os.path.exists(out_mean) or os.path.exists(out_std)):
        raise FileExistsError(
            f"Output exists. Use --overwrite to replace.\n"
            f"- {out_mean}\n"
            f"- {out_std}"
        )

    mean263 = np.load(mean263_path)
    std263 = np.load(std263_path)
    if mean263.shape != (263,) or std263.shape != (263,):
        raise ValueError(f"Expected Mean/Std shape (263,), got mean={mean263.shape}, std={std263.shape}")

    mean6, std6, used_files, skipped_files = _stream_mean_std_last6(
        data_dir=data_dir,
        max_files=args.max_files,
        skip_nan=(not args.no_skip_nan),
    )

    if args.smooth_last6:
        std6 = np.full_like(std6, float(std6.mean()))

    mean269 = np.concatenate([mean263.astype(np.float32, copy=False), mean6], axis=0)
    std269 = np.concatenate([std263.astype(np.float32, copy=False), std6], axis=0)
    assert mean269.shape == (269,) and std269.shape == (269,)

    np.save(out_mean, mean269)
    np.save(out_std, std269)

    print("==========================================")
    print("HumanML3D Mean/Std (269 = 263 + 6)")
    print("==========================================")
    print(f"data_dir:            {data_dir}")
    print(f"official_stats_dir:  {official_stats_dir}")
    print(f"save_dir:            {save_dir}")
    print(f"used_files:          {used_files}")
    print(f"skipped_files:       {skipped_files}")
    print(f"max_files:           {args.max_files}")
    print(f"skip_nan:            {not args.no_skip_nan}")
    print(f"smooth_last6:        {args.smooth_last6}")
    print("------------------------------------------")
    print(f"Mean_269:            {out_mean}")
    print(f"Std_269:             {out_std}")
    print(f"shape(mean/std):     {mean269.shape} / {std269.shape}")
    print(f"last6 mean range:    [{float(mean6.min()):.6g}, {float(mean6.max()):.6g}]")
    print(f"last6 std  range:    [{float(std6.min()):.6g}, {float(std6.max()):.6g}]")
    print("==========================================")


if __name__ == "__main__":
    main()


