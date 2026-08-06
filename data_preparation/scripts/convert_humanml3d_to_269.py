"""
Convert the HumanML3D dataset to the 269-dim unified representation (263 + 6)
=============================================================================

Goal:
- First 263 dims: read the official HumanML3D preprocessing output
  `new_joint_vecs` directly (avoids any numerical drift).
- Last 6 dims: global_orient_6d computed from `joints` via
  unified_representation_269.process_motion_generation.

Consistency is verified during conversion:
- For every sample, check whether our_269[:, :263] matches official_263
  (exact / allclose). In our full run all 29228 files passed allclose.

Inputs:
- joints dir:       HumanML3D/joints/*.npy       (each (T, J, 3), (T, 22, 3) or (T, 66))
- official 263 dir: HumanML3D/HumanML3D/new_joint_vecs/*.npy  (each (T-1, 263))

Output:
- HumanML3D/HumanML3D/new_joint_vecs_269/*.npy   (each (T-1, 269))

Usage example (run from the repository root):
  python data_preparation/scripts/convert_humanml3d_to_269.py \
    --humanml3d-root data/HumanML3D \
    --overwrite
"""

from __future__ import annotations

import os
import sys
import argparse
from os.path import join as pjoin
from typing import Dict, Tuple, Optional, List

import numpy as np
from tqdm import tqdm

# Add the data_preparation directory (which contains unified_representation_269.py)
# to sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from unified_representation_269 import process_motion_generation, set_target_skeleton_path  # noqa: E402


def _load_joints_22(path: str) -> np.ndarray:
    """
    Load a HumanML3D joints file and return it as (T, 22, 3).
    Handles several common storage formats:
    - (T, J>=22, 3) -> take the first 22 joints
    - (T, 22, 3)
    - (T, 66) -> reshape to (T, 22, 3)

    Important: the dtype is intentionally NOT forced here; the original dtype
    is preserved so we can do strict / high-precision alignment checks against
    the official HumanML3D new_joint_vecs (263).
    """
    arr = np.load(path)

    # (T, 66)
    if arr.ndim == 2 and arr.shape[-1] == 66:
        arr = arr.reshape(arr.shape[0], 22, 3)
    # (T, J, 3)
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        if arr.shape[1] < 22:
            raise ValueError(f"joints file has <22 joints: shape={arr.shape}")
        arr = arr[:, :22, :]
    else:
        raise ValueError(f"Unsupported joints format: shape={arr.shape}")

    return arr


def _compare_arrays(a: np.ndarray, b: np.ndarray, rtol: float, atol: float) -> Dict[str, object]:
    if a.shape != b.shape:
        return {
            "ok_allclose": False,
            "ok_exact": False,
            "max_diff": float("inf"),
            "mean_diff": float("inf"),
            "shape_a": a.shape,
            "shape_b": b.shape,
        }
    diff = np.abs(a - b)
    return {
        "ok_allclose": bool(np.allclose(a, b, rtol=rtol, atol=atol)),
        "ok_exact": bool(diff.max() == 0),
        "max_diff": float(diff.max()),
        "mean_diff": float(diff.mean()),
        "shape_a": a.shape,
        "shape_b": b.shape,
    }


def convert_humanml3d_dataset(
    joints_dir: str,
    official_263_dir: str,
    out_dir: str,
    target_skeleton_path: str,
    feet_thre: float = 0.002,
    overwrite: bool = False,
    max_files: Optional[int] = None,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    strict: bool = False,
    log_inconsistent: int = 20,
) -> Dict[str, object]:
    """
    Returns summary dict.
    """
    if not os.path.isdir(joints_dir):
        raise FileNotFoundError(f"JOINTS_DIR not found: {joints_dir}")
    if not os.path.isdir(official_263_dir):
        raise FileNotFoundError(f"OFFICIAL_263_DIR not found: {official_263_dir}")
    if not os.path.exists(target_skeleton_path):
        raise FileNotFoundError(f"Target skeleton file not found: {target_skeleton_path}")

    os.makedirs(out_dir, exist_ok=True)

    # Use the HumanML3D reference target skeleton sample (000021.npy) consistently
    set_target_skeleton_path(target_skeleton_path)

    joints_files = sorted([f for f in os.listdir(joints_dir) if f.endswith(".npy")])
    official_files = set([f for f in os.listdir(official_263_dir) if f.endswith(".npy")])

    common_files = [f for f in joints_files if f in official_files]
    if max_files is not None:
        common_files = common_files[: max_files]

    stats = {
        "total": len(common_files),
        "written": 0,
        "skipped_exists": 0,
        "failed": 0,
        "ok_exact": 0,
        "ok_allclose": 0,
        "inconsistent": 0,
        "max_diff_overall": 0.0,
        "mean_diff_overall": 0.0,
        "inconsistent_examples": [],  # type: ignore[list-item]
    }

    max_diff_list: List[float] = []
    mean_diff_list: List[float] = []

    for fname in tqdm(common_files, desc="HumanML3D->269", dynamic_ncols=True):
        in_joints = pjoin(joints_dir, fname)
        in_official = pjoin(official_263_dir, fname)
        out_path = pjoin(out_dir, fname)

        if (not overwrite) and os.path.exists(out_path):
            stats["skipped_exists"] += 1
            continue

        try:
            joints_22 = _load_joints_22(in_joints)  # (T, 22, 3)
            official_263 = np.load(in_official)     # (T-1, 263)

            # Compute our_269: only the last 6 dims are used for output;
            # the first 263 are used for verification
            our_269 = process_motion_generation(joints_22, feet_thre=feet_thre)  # (T-1, 269)

            if our_269.shape[0] != official_263.shape[0] or official_263.shape[-1] != 263:
                raise ValueError(
                    f"Shape mismatch: our_269={our_269.shape}, official_263={official_263.shape} for {fname}"
                )

            cmp = _compare_arrays(our_269[:, :263], official_263, rtol=rtol, atol=atol)
            if cmp["ok_exact"]:
                stats["ok_exact"] += 1
            if cmp["ok_allclose"]:
                stats["ok_allclose"] += 1
            if not cmp["ok_allclose"]:
                stats["inconsistent"] += 1
                if len(stats["inconsistent_examples"]) < log_inconsistent:
                    stats["inconsistent_examples"].append(
                        {
                            "file": fname,
                            "max_diff": cmp["max_diff"],
                            "mean_diff": cmp["mean_diff"],
                            "shape": {"our": cmp["shape_a"], "official": cmp["shape_b"]},
                        }
                    )
                if strict:
                    raise RuntimeError(
                        f"Inconsistent 263 for {fname}: max_diff={cmp['max_diff']:.3e}, mean_diff={cmp['mean_diff']:.3e}"
                    )

            # Output: official 263 + our last 6
            tail_6 = our_269[:, 263:269]
            merged_269 = np.concatenate([official_263, tail_6], axis=-1)

            # Keep the dtype consistent with the official file (usually float32)
            if merged_269.dtype != official_263.dtype:
                merged_269 = merged_269.astype(official_263.dtype, copy=False)

            np.save(out_path, merged_269)
            stats["written"] += 1

            # Aggregate statistics
            if np.isfinite(cmp["max_diff"]):
                max_diff_list.append(float(cmp["max_diff"]))
            if np.isfinite(cmp["mean_diff"]):
                mean_diff_list.append(float(cmp["mean_diff"]))

        except Exception as e:
            stats["failed"] += 1
            if len(stats["inconsistent_examples"]) < log_inconsistent:
                stats["inconsistent_examples"].append({"file": fname, "error": str(e)})
            if strict:
                raise

    if max_diff_list:
        stats["max_diff_overall"] = float(np.max(max_diff_list))
        stats["mean_diff_overall"] = float(np.mean(max_diff_list))
    if mean_diff_list:
        stats["mean_diff_overall"] = float(np.mean(mean_diff_list))

    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert HumanML3D joints + official 263 -> 269 (263+6).")
    parser.add_argument(
        "--humanml3d-root",
        type=str,
        default="data/HumanML3D",
        help="HumanML3D dataset root (containing joints/ and HumanML3D/new_joint_vecs/)",
    )
    parser.add_argument(
        "--joints-dir",
        type=str,
        default=None,
        help="joints directory (inferred from --humanml3d-root by default)",
    )
    parser.add_argument(
        "--official-263-dir",
        type=str,
        default=None,
        help="official 263 directory new_joint_vecs (inferred from --humanml3d-root by default)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="output directory new_joint_vecs_269 (inferred from --humanml3d-root by default)",
    )
    parser.add_argument(
        "--target-skeleton",
        type=str,
        default=None,
        help="target skeleton reference sample joints/000021.npy (inferred from --humanml3d-root by default)",
    )
    parser.add_argument("--feet-thre", type=float, default=0.002, help="foot_detect threshold (same as HumanML3D)")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing output files")
    parser.add_argument("--max-files", type=int, default=None, help="process at most N files (debug)")
    parser.add_argument("--rtol", type=float, default=1e-5, help="allclose rtol")
    parser.add_argument("--atol", type=float, default=1e-6, help="allclose atol")
    parser.add_argument("--strict", action="store_true", help="stop immediately on inconsistency or error")
    parser.add_argument("--log-inconsistent", type=int, default=20, help="max number of inconsistent/failed samples recorded in the summary")
    args = parser.parse_args()

    root = args.humanml3d_root
    joints_dir = args.joints_dir or pjoin(root, "joints")
    official_263_dir = args.official_263_dir or pjoin(root, "HumanML3D", "new_joint_vecs")
    out_dir = args.out_dir or pjoin(root, "HumanML3D", "new_joint_vecs_269")
    target_skeleton = args.target_skeleton or pjoin(root, "joints", "000021.npy")

    print("==========================================")
    print("HumanML3D -> 269 (263 + 6) conversion")
    print("==========================================")
    print(f"JOINTS_DIR:         {joints_dir}")
    print(f"OFFICIAL_263_DIR:   {official_263_dir}")
    print(f"OUT_DIR:            {out_dir}")
    print(f"TGT_SKELETON:       {target_skeleton}")
    print(f"feet_thre:          {args.feet_thre}")
    print(f"overwrite:          {args.overwrite}")
    print(f"max_files:          {args.max_files}")
    print(f"allclose:           rtol={args.rtol} atol={args.atol}")
    print(f"strict:             {args.strict}")
    print("==========================================")

    summary = convert_humanml3d_dataset(
        joints_dir=joints_dir,
        official_263_dir=official_263_dir,
        out_dir=out_dir,
        target_skeleton_path=target_skeleton,
        feet_thre=args.feet_thre,
        overwrite=args.overwrite,
        max_files=args.max_files,
        rtol=args.rtol,
        atol=args.atol,
        strict=args.strict,
        log_inconsistent=args.log_inconsistent,
    )

    print("\n========== Summary ==========")
    for k in [
        "total",
        "written",
        "skipped_exists",
        "failed",
        "ok_allclose",
        "ok_exact",
        "inconsistent",
        "max_diff_overall",
        "mean_diff_overall",
    ]:
        print(f"{k:16s}: {summary.get(k)}")

    examples = summary.get("inconsistent_examples", [])
    if examples:
        print("\n--- Examples (inconsistent/failed) ---")
        for ex in examples:
            print(ex)


if __name__ == "__main__":
    main()
