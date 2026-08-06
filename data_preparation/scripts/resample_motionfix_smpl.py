"""
MotionFix SMPL-parameter frame-rate resampling script
=====================================================

Performs the frame-rate conversion in the original SMPL parameter space
(joint_positions, rots, trans) BEFORE computing the 269-dim representation
with convert_motionfix_to_269.py.

This is the more principled approach:
- Interpolating directly in the original space avoids feature-coupling issues
  in the 269 representation
- Velocities, contacts, etc. are then naturally computed at the new frame rate
- Stays consistent with the original processing pipeline

Processing steps:
1. Load the original MotionFix data (30fps)
2. Linearly interpolate joint_positions
3. Convert rots (axis-angle) to quaternions and SLERP-interpolate
4. Linearly interpolate trans
5. Save the resampled SMPL data
6. Optional: run convert_motionfix_to_269.py to produce the 269 representation

Usage (run from the repository root):
    python data_preparation/scripts/resample_motionfix_smpl.py \
        --input /path/to/motionfix.pth.tar \
        --output /path/to/motionfix_20fps.pth.tar \
        --src-fps 30 \
        --tgt-fps 20
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import numpy as np

try:
    import joblib
except ImportError as e:
    raise ImportError("This script requires joblib. Please `pip install joblib`.") from e

try:
    from tqdm import tqdm
except ImportError:
    # Simple fallback when tqdm is not installed
    def tqdm(iterable, **kwargs):
        total = kwargs.get('total', len(iterable) if hasattr(iterable, '__len__') else None)
        desc = kwargs.get('desc', '')
        for i, item in enumerate(iterable):
            if (i + 1) % 100 == 0 or i == 0:
                print(f"\r{desc}: {i + 1}/{total if total else '?'}", end='', flush=True)
            yield item
        print()

# Add the data_preparation directory (which contains unified_representation_269.py)
# to sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from unified_representation_269 import (
    axis_angle_to_matrix_np,
    matrix_to_quaternion_np,
    quaternion_to_matrix_np,
    matrix_to_axis_angle_np,
)


# ============================================================================
# Quaternion SLERP
# ============================================================================

def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    Quaternion spherical linear interpolation (SLERP).

    Args:
        q0: start quaternion (..., 4), format [w, x, y, z]
        q1: end quaternion (..., 4)
        t: interpolation parameter in [0, 1]

    Returns:
        interpolated quaternion (..., 4)
    """
    # Make sure the quaternions are in the same hemisphere
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)

    # Handle nearly identical quaternions
    DOT_THRESHOLD = 0.9995
    close_mask = (dot > DOT_THRESHOLD).squeeze(-1)

    # SLERP
    theta_0 = np.arccos(np.clip(dot, -1, 1))
    sin_theta_0 = np.sin(theta_0)
    sin_theta_0 = np.where(np.abs(sin_theta_0) < 1e-10, 1e-10, sin_theta_0)

    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    result = s0 * q0 + s1 * q1

    # Linear-interpolation fallback
    if np.any(close_mask):
        linear_result = (1 - t) * q0 + t * q1
        linear_result = linear_result / (np.linalg.norm(linear_result, axis=-1, keepdims=True) + 1e-10)
        result = np.where(close_mask[..., None], linear_result, result)

    # Normalize
    result = result / (np.linalg.norm(result, axis=-1, keepdims=True) + 1e-10)

    return result


def interpolate_axis_angle(aa0: np.ndarray, aa1: np.ndarray, t: float) -> np.ndarray:
    """
    SLERP interpolation on the axis-angle representation.

    Args:
        aa0: start axis-angle (..., 3)
        aa1: end axis-angle (..., 3)
        t: interpolation parameter in [0, 1]

    Returns:
        interpolated axis-angle (..., 3)
    """
    # axis-angle -> rotation matrix -> quaternion
    mat0 = axis_angle_to_matrix_np(aa0)
    mat1 = axis_angle_to_matrix_np(aa1)

    quat0 = matrix_to_quaternion_np(mat0)
    quat1 = matrix_to_quaternion_np(mat1)

    # SLERP
    quat_interp = quaternion_slerp(quat0, quat1, t)

    # quaternion -> rotation matrix -> axis-angle
    mat_interp = quaternion_to_matrix_np(quat_interp)
    aa_interp = matrix_to_axis_angle_np(mat_interp)

    return aa_interp


# ============================================================================
# Resampling core
# ============================================================================

def resample_motion_dict(
    motion_dict: Dict[str, Any],
    src_fps: float,
    tgt_fps: float,
) -> Dict[str, Any]:
    """
    Resample the SMPL parameters of a single motion dict.

    Handled fields:
    - joint_positions: (T, 22, 3) or (T, N, 3) - linear interpolation
    - rots: (T, 66) or (T, 22, 3) - SLERP interpolation
    - trans: (T, 3) - linear interpolation
    """
    result = dict(motion_dict)

    # Get the original frame count
    if "joint_positions" in motion_dict:
        joints = np.asarray(motion_dict["joint_positions"])
        T = joints.shape[0]
    elif "rots" in motion_dict:
        rots = np.asarray(motion_dict["rots"])
        T = rots.shape[0]
    else:
        print("Warning: No joint_positions or rots found, skipping motion_dict")
        return result

    if T <= 1:
        return result

    # Compute the new frame count
    duration = (T - 1) / src_fps
    T_new = int(np.floor(duration * tgt_fps)) + 1

    if T_new <= 1:
        return result

    # Compute the original-frame positions corresponding to the new frames
    new_indices = np.arange(T_new) * (src_fps / tgt_fps)
    new_indices = np.clip(new_indices, 0, T - 1)

    # ============================================
    # 1. Resample joint_positions (linear interpolation)
    # ============================================
    if "joint_positions" in motion_dict:
        joints = np.asarray(motion_dict["joint_positions"]).astype(np.float32)

        joints_new = np.zeros((T_new,) + joints.shape[1:], dtype=np.float32)
        for i in range(T_new):
            pos = new_indices[i]
            idx_low = int(np.floor(pos))
            idx_high = min(idx_low + 1, T - 1)
            t = pos - idx_low
            joints_new[i] = (1 - t) * joints[idx_low] + t * joints[idx_high]

        result["joint_positions"] = joints_new

    # ============================================
    # 2. Resample rots (SLERP interpolation)
    # ============================================
    if "rots" in motion_dict:
        rots = np.asarray(motion_dict["rots"]).astype(np.float32)

        # Handle different rots formats
        if rots.ndim == 2 and rots.shape[-1] == 66:
            # (T, 66) -> (T, 22, 3)
            rots_reshaped = rots.reshape(T, 22, 3)
            need_reshape_back = True
        elif rots.ndim == 3 and rots.shape[-1] == 3:
            # (T, N, 3)
            rots_reshaped = rots
            need_reshape_back = False
        elif rots.ndim == 2 and rots.shape[-1] == 72:
            # (T, 72) -> (T, 24, 3)
            rots_reshaped = rots.reshape(T, 24, 3)
            need_reshape_back = True
        else:
            print(f"Warning: Unsupported rots format {rots.shape}, using linear interpolation")
            rots_new = np.zeros((T_new,) + rots.shape[1:], dtype=np.float32)
            for i in range(T_new):
                pos = new_indices[i]
                idx_low = int(np.floor(pos))
                idx_high = min(idx_low + 1, T - 1)
                t = pos - idx_low
                rots_new[i] = (1 - t) * rots[idx_low] + t * rots[idx_high]
            result["rots"] = rots_new
            rots_reshaped = None

        if rots_reshaped is not None:
            num_joints = rots_reshaped.shape[1]
            rots_new = np.zeros((T_new, num_joints, 3), dtype=np.float32)

            for i in range(T_new):
                pos = new_indices[i]
                idx_low = int(np.floor(pos))
                idx_high = min(idx_low + 1, T - 1)
                t = pos - idx_low

                if t < 1e-6 or idx_low == idx_high:
                    rots_new[i] = rots_reshaped[idx_low]
                else:
                    # SLERP each joint's axis-angle
                    rots_new[i] = interpolate_axis_angle(
                        rots_reshaped[idx_low], rots_reshaped[idx_high], t
                    )

            if need_reshape_back:
                result["rots"] = rots_new.reshape(T_new, -1)
            else:
                result["rots"] = rots_new

    # ============================================
    # 3. Resample trans (linear interpolation)
    # ============================================
    if "trans" in motion_dict:
        trans = np.asarray(motion_dict["trans"]).astype(np.float32)

        trans_new = np.zeros((T_new,) + trans.shape[1:], dtype=np.float32)
        for i in range(T_new):
            pos = new_indices[i]
            idx_low = int(np.floor(pos))
            idx_high = min(idx_low + 1, T - 1)
            t = pos - idx_low
            trans_new[i] = (1 - t) * trans[idx_low] + t * trans[idx_high]

        result["trans"] = trans_new

    # Record frame-rate information
    result["original_fps"] = src_fps
    result["fps"] = tgt_fps
    result["original_frames"] = T
    result["frames"] = T_new

    return result


def resample_sample(
    sample: Dict[str, Any],
    src_fps: float,
    tgt_fps: float,
) -> Dict[str, Any]:
    """
    Resample a single sample.
    """
    result = dict(sample)

    if "motion_source" in sample:
        result["motion_source"] = resample_motion_dict(
            sample["motion_source"], src_fps, tgt_fps
        )

    if "motion_target" in sample:
        result["motion_target"] = resample_motion_dict(
            sample["motion_target"], src_fps, tgt_fps
        )

    return result


def _worker(args_tuple):
    """
    Multiprocessing worker function.
    """
    key, sample, src_fps, tgt_fps = args_tuple
    resampled_sample = resample_sample(sample, src_fps, tgt_fps)
    return (key, resampled_sample)


def main():
    parser = argparse.ArgumentParser(
        description="Resample MotionFix SMPL parameters from source FPS to target FPS."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input MotionFix joblib file (original SMPL data)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output joblib file (resampled SMPL data)"
    )
    parser.add_argument(
        "--src-fps", type=float, default=30.0,
        help="Source frame rate (default: 30)"
    )
    parser.add_argument(
        "--tgt-fps", type=float, default=20.0,
        help="Target frame rate (default: 20)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Only process first N samples (for debugging)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite output file if exists"
    )
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help="Number of parallel workers (default: 1, single process)"
    )
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}. Use --overwrite to overwrite.")

    print(f"[resample_motionfix_smpl] Loading: {args.input}")
    data = joblib.load(args.input)

    # Handle the data format
    if isinstance(data, dict):
        items = list(data.items())
        out_is_dict = True
    elif isinstance(data, list):
        items = list(enumerate(data))
        out_is_dict = False
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    if args.max_samples is not None:
        items = items[:args.max_samples]

    print(f"[resample_motionfix_smpl] Processing {len(items)} samples...")
    print(f"[resample_motionfix_smpl] {args.src_fps} fps -> {args.tgt_fps} fps")
    print(f"[resample_motionfix_smpl] Workers: {args.num_workers}")

    out_items = []

    if args.num_workers <= 1:
        # Single-process mode
        for key, sample in tqdm(items, desc="Resampling SMPL", unit="sample"):
            resampled_sample = resample_sample(sample, args.src_fps, args.tgt_fps)
            out_items.append((key, resampled_sample))
    else:
        # Multi-process mode
        # Prepare the task arguments
        task_args = [(key, sample, args.src_fps, args.tgt_fps) for key, sample in items]

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(_worker, arg): i for i, arg in enumerate(task_args)}

            # Collect results (in completion order)
            results = [None] * len(items)
            for future in tqdm(as_completed(futures), total=len(futures),
                              desc=f"Resampling SMPL ({args.num_workers} workers)", unit="sample"):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"\nError processing sample {idx}: {e}")
                    raise

            out_items = results

    # Rebuild the output
    if out_is_dict:
        out_data = {k: v for k, v in out_items}
    else:
        out_data = [v for _, v in out_items]

    print(f"[resample_motionfix_smpl] Saving: {args.output}")
    joblib.dump(out_data, args.output)
    print("[resample_motionfix_smpl] Done.")
    print()
    print("Next step: Run convert_motionfix_to_269.py on the resampled data:")
    print(f"  python data_preparation/scripts/convert_motionfix_to_269.py \\")
    print(f"    --in-joblib {args.output} \\")
    print(f"    --out-joblib {args.output.replace('.pth.tar', '_269.pth.tar')} \\")
    print(f"    --out-mode smpl_semantic_t \\")
    print(f"    --coord zup_to_yup")


if __name__ == "__main__":
    main()
