"""
Convert MotionFix data to the 269-dim unified representation (HumanML3D semantics)
==================================================================================

Background
----------
Native MotionFix data typically contains (per sample):
- motion_source / motion_target:
  - joint_positions: (T, 22, 3)  (usually the first 22 joints in SMPL/SMPLH order)
  - rots: (T, 66) or equivalent (global_orient(3) + body_pose(21*3))
  - trans: (T, 3)

This repository's `unified_representation_269.process_motion_generation` builds
the 269 representation using the HumanML3D/T2M 22-joint order and the "Y-up"
coordinate convention.

Therefore, before feeding MotionFix joints to `process_motion_generation`, you
usually need:
1) A *basis* change of coordinates: if the MotionFix data is Z-up (MotionFix
   code commonly uses yaw around Z), convert Z-up -> Y-up.

About joint order:
- The key indices in `unified_representation_269.py` (e.g.
  `FACE_JOINT_INDX=[2,1,17,16]`, foot indices, ...) are consistent with the
  **first 22 joints of the SMPL order**.
- Hence this script does **no joint reordering** by default (unless explicitly
  requested).

Important limitation (fair-comparison related)
----------------------------------------------
The 269 produced by the default HumanML3D pipeline has canonical/IK semantics:
- Suitable for training a unified input (T2M/M2T/motion modeling, ...)
- But it is **not possible to losslessly recover** the native MotionFix
  (trans, rots, joint_positions) from it:
  - the 269 drops the absolute world translation reference and the exact
    MotionFix global-orientation semantics;
  - `process_motion_generation` additionally applies uniform_skeleton +
    canonical facing.
If you need reversibility for strict fair comparison (e.g. mapping model output
back to the native MotionFix rot/trans space), additionally store the original
rots/trans (or design a separate SMPL-semantic 269 packing).

Update: "SMPL semantics, length-preserving T" 269 output mode
-------------------------------------------------------------
When you want a strict fair comparison on MotionFix against methods operating
in "SMPL/rots/trans space", use `--out-mode smpl_semantic_t`:
- The output repr_269 has shape (T, 269) (no minus-1)
- Every frame stores the 6D rotations of the original MotionFix rots
  (global_orient + 21 body joints)
- Velocity-like dims use finite differences with padding (spec below)

Padding spec (out-mode=smpl_semantic_t)
---------------------------------------
- root_rot_velocity[0] = 0
- root_linear_velocity[0] = (0, 0)
- local_vel[0] = 0 (66 dims)
- foot_contact[0] = 0.5 (4 dims, meaning unknown)
- foot_contact[t] (t>=1) uses the (t-1)->t foot-speed threshold test (same
  style as HumanML3D foot_detect)

Usage (run from the repository root)
------------------------------------
python data_preparation/scripts/convert_motionfix_to_269.py \
  --in-joblib /path/to/motionfix_dataset.pth.tar \
  --out-joblib /path/to/motionfix_dataset_with269.pth.tar \
  --coord zup_to_yup \
  --preserve-rotation \
  --overwrite
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import numpy as np
try:
    import joblib
except ImportError as e:  # pragma: no cover
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

from unified_representation_269 import process_motion_generation, Skeleton, T2M_KINEMATIC_CHAIN, T2M_RAW_OFFSETS  # noqa: E402
from unified_representation_269 import axis_angle_to_matrix_np, matrix_to_cont6d_np  # noqa: E402
from unified_representation_269 import Repr269  # noqa: E402
from unified_representation_269 import (  # noqa: E402
    FACE_JOINT_INDX,
    get_target_offsets,
    qbetween_np,
    qrot_np,
    uniform_skeleton,
)


# -----------------------------------------------------------------------------
# Joint order mapping (optional)
# -----------------------------------------------------------------------------
# Default: no reordering (assumes the input joint_positions is already in the
# standard SMPL-22 order). If your MotionFix variant stores joint_positions in a
# different order, enable the remapping explicitly on the CLI.
SMPL22_TO_T2M22 = np.array(
    [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    # torch tensor
    try:
        import torch

        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _reorder_smpl22_to_t2m22(joints: np.ndarray) -> np.ndarray:
    if joints.shape[-2] != 22 or joints.shape[-1] != 3:
        raise ValueError(f"Expected joints shape (..., 22, 3), got {joints.shape}")
    return joints[..., SMPL22_TO_T2M22, :]


def _reorder_pose_smpl22_to_t2m22(pose_aa: np.ndarray) -> np.ndarray:
    """
    Reorder SMPL22 axis-angle pose to match the joint order used for joints.
    pose_aa: (..., 22, 3)
    """
    if pose_aa.shape[-2] != 22 or pose_aa.shape[-1] != 3:
        raise ValueError(f"Expected pose shape (..., 22, 3), got {pose_aa.shape}")
    return pose_aa[..., SMPL22_TO_T2M22, :]


def _coord_transform(joints: np.ndarray, mode: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply a *basis* transform to positions (and translations if you apply it there).
    """
    if mode == "none":
        return joints, {"mode": "none"}

    if mode == "zup_to_yup":
        # Old basis (Z-up, right-handed): X right, Y forward, Z up
        # New basis (Y-up, right-handed): X right, Y up, Z forward
        # Choose: Xn=Xo, Yn=Zo, Zn=-Yo  => v_new = C v_old
        C = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        )
        joints_new = np.einsum("ij,...j->...i", C, joints)
        return joints_new, {"mode": mode, "C": C}

    raise ValueError(f"Unsupported --coord: {mode}")


def _convert_rotmat_world_basis(rotmat_22: np.ndarray, coord: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply the same *world basis* change as `_coord_transform` to the SMPL root rotation.

    Important:
    - For a basis change v_new = C v_old, a world rotation matrix should transform as:
        R_new = C @ R_old @ C^{-1}
      Here C is orthonormal so C^{-1} = C^T.
    - We only apply this conjugation to the **root** (global_orient). Body pose is local (in parent joint frames)
      and is left unchanged.
    """
    if coord == "none":
        return rotmat_22, {"mode": "none"}
    if coord == "zup_to_yup":
        # Same C as `_coord_transform` for positions: v_new = C v_old
        C = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        )
        if rotmat_22.shape[-3:] != (22, 3, 3):
            raise ValueError(f"Expected rotmat shape (...,22,3,3), got {rotmat_22.shape}")
        rotmat_body = rotmat_22[..., 1:, :, :]
        rotmat_root = rotmat_22[..., :1, :, :]
        rotmat_root = C @ rotmat_root @ C.T
        return np.concatenate([rotmat_root, rotmat_body], axis=-3), {"mode": coord, "C": C, "note": "root only: R'=C@R@C^T"}
    raise ValueError(f"Unsupported coord for rotmat basis: {coord}")


def _extract_pose_aa_22(motion_dict: Dict[str, Any]) -> np.ndarray:
    """
    Extract SMPL22 axis-angle pose from MotionFix sample.

    Supported formats:
    - rots: (T, 66) where 66 = 22*3 (global_orient + 21*body_pose)
    - rots: (T, 22, 3)
    - rots: (T, 72) or (T, 24, 3): will take first 22 joints
    """
    if "rots" not in motion_dict:
        raise KeyError("Missing 'rots' in motion dict (needed for --preserve-rotation).")
    rots = _as_numpy(motion_dict["rots"])
    if rots.ndim == 2 and rots.shape[-1] == 66:
        pose_aa = rots.reshape(rots.shape[0], 22, 3)
    elif rots.ndim == 3 and rots.shape[-2:] == (22, 3):
        pose_aa = rots
    elif rots.ndim == 2 and rots.shape[-1] == 72:
        pose_aa = rots.reshape(rots.shape[0], 24, 3)[:, :22]
    elif rots.ndim == 3 and rots.shape[-2:] == (24, 3):
        pose_aa = rots[:, :22]
    else:
        raise ValueError(f"Unsupported rots format: shape={rots.shape} (expected (T,66) or (T,22,3))")
    return pose_aa.astype(np.float32)


def _compute_target_offset_from_first_frame(joints_t2m_yup: np.ndarray) -> np.ndarray:
    """
    Compute target offsets from the first frame so `uniform_skeleton` becomes (almost) identity.
    This reduces unintended cross-dataset skeleton scaling when converting MotionFix.
    """
    if joints_t2m_yup.ndim != 3 or joints_t2m_yup.shape[1:] != (22, 3):
        raise ValueError(f"Expected joints (T,22,3), got {joints_t2m_yup.shape}")
    skel = Skeleton(
        offset=torch.from_numpy(T2M_RAW_OFFSETS).float(),
        kinematic_tree=T2M_KINEMATIC_CHAIN,
        device="cpu",
    )
    # `Skeleton.get_offsets_joints_batch` expects a torch Tensor (N, 22, 3)
    j0 = torch.from_numpy(joints_t2m_yup[:1].astype(np.float32))
    tgt = skel.get_offsets_joints_batch(j0)
    # tgt: (1,22,3) torch.Tensor or numpy-like; force numpy
    return _as_numpy(tgt[0])


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angles (radians) to (-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def _yaw_from_rotmat_yup(R: np.ndarray) -> np.ndarray:
    """
    Extract yaw (rotation about +Y) from rotation matrices in a Y-up convention.
    R: (..., 3, 3)
    Returns: (...) yaw in radians
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotmat (...,3,3), got {R.shape}")
    # Standard Y-up yaw: yaw = atan2(R[0,2], R[2,2])
    return np.arctan2(R[..., 0, 2], R[..., 2, 2])


def _foot_contact_t_from_positions(
    positions: np.ndarray,
    thres: float,
    fid_l: Tuple[int, int] = (7, 10),
    fid_r: Tuple[int, int] = (8, 11),
    pad_value: float = 0.5,
) -> np.ndarray:
    """
    Build per-frame (T,4) foot contact where contact[t] (t>=1) is computed from (t-1)->t speed,
    and contact[0] is padded as `pad_value`.
    """
    if positions.ndim != 3 or positions.shape[1:] != (22, 3):
        raise ValueError(f"Expected positions (T,22,3), got {positions.shape}")
    T = positions.shape[0]
    if T == 0:
        return np.zeros((0, 4), dtype=np.float32)
    if T == 1:
        return np.full((1, 4), pad_value, dtype=np.float32)

    velfactor = np.array([thres, thres], dtype=np.float32)

    dl = positions[1:, list(fid_l), :] - positions[:-1, list(fid_l), :]
    feet_l = ((dl[..., 0] ** 2 + dl[..., 1] ** 2 + dl[..., 2] ** 2) < velfactor).astype(np.float32)  # (T-1,2)

    dr = positions[1:, list(fid_r), :] - positions[:-1, list(fid_r), :]
    feet_r = ((dr[..., 0] ** 2 + dr[..., 1] ** 2 + dr[..., 2] ** 2) < velfactor).astype(np.float32)  # (T-1,2)

    contact = np.concatenate([feet_l, feet_r], axis=-1)  # (T-1,4)
    contact = np.concatenate([np.full((1, 4), pad_value, dtype=np.float32), contact], axis=0)  # (T,4)
    return contact


def _canonical_root_y_and_ric_all_frames(
    joints_yup: np.ndarray,
    target_offset: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reproduce the canonical pipeline inside `process_motion_generation` but keep outputs for all T frames.

    Returns:
        root_y:   (T, 1)
        ric_data: (T, 63)   positions[:,1:] in RIFKE/local coordinates
    """
    if joints_yup.ndim != 3 or joints_yup.shape[1:] != (22, 3):
        raise ValueError(f"Expected joints (T,22,3), got {joints_yup.shape}")
    T = joints_yup.shape[0]
    if T == 0:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0, 63), dtype=np.float32)

    positions = joints_yup.copy()

    # 1) uniform skeleton (HumanML3D)
    positions = uniform_skeleton(positions, target_offset)  # (T,22,3)

    # 2) put on floor (Y-up)
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    # 3) XZ origin align using first frame
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1], dtype=np.float32)
    positions = positions - root_pose_init_xz

    # 4) rotate so initial facing is Z+
    r_hip, l_hip, sdr_r, sdr_l = FACE_JOINT_INDX
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    forward_init = np.cross(np.array([[0, 1, 0]], dtype=np.float32), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]], dtype=np.float32)
    root_quat_init = qbetween_np(forward_init, target)  # (1,4)
    root_quat_init = np.ones(positions.shape[:-1] + (4,), dtype=np.float32) * root_quat_init  # (T,22,4)
    positions = qrot_np(root_quat_init, positions)

    # 5) IK to get r_rot (smooth_forward=True matches process_motion_generation)
    skel = Skeleton(torch.from_numpy(T2M_RAW_OFFSETS).float(), T2M_KINEMATIC_CHAIN, "cpu")
    quat_params = skel.inverse_kinematics_np(positions, FACE_JOINT_INDX, smooth_forward=True)  # (T,22,4)
    r_rot = quat_params[:, 0].copy()  # (T,4)

    # 6) RIFKE: subtract root XZ and rotate into root local frame
    positions[..., 0] -= positions[:, 0:1, 0]
    positions[..., 2] -= positions[:, 0:1, 2]
    positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)

    root_y = positions[:, 0, 1:2].astype(np.float32)         # (T,1)
    ric_data = positions[:, 1:].reshape(T, -1).astype(np.float32)  # (T,63)
    return root_y, ric_data


def _convert_one_motion_smpl_semantic_t(
    motion_dict: Dict[str, Any],
    coord: str,
    reorder_joints: str,
    target_offset_mode: str,
    feet_thre: float,
    preserve_global_orient_only: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Output (T,269) SMPL-semantic representation (append-1-frame strategy):
    - First compute base features with `process_motion_generation(positions)` which outputs (T-1,269)
    - Then append 1 frame at the end to make (T,269)
    - Finally override per-frame rotations (global_orient_6d + rot_data) from MotionFix original SMPL `rots`

    Padding convention for the appended frame (t=T-1):
    - root_rot_velocity[T-1] = 0
    - root_linear_velocity[T-1] = (0,0)
    - local_vel[T-1] = 0 (66 dims)
    - foot_contact[T-1] = 0.5 (unknown)
    """
    joints = _as_numpy(motion_dict["joint_positions"])
    if joints.ndim != 3:
        raise ValueError(f"joint_positions must be (T,22,3), got {joints.shape}")
    if joints.shape[-1] == 66 and joints.ndim == 2:
        joints = joints.reshape(joints.shape[0], 22, 3)
    if joints.shape[1:] != (22, 3):
        raise ValueError(f"joint_positions must be (T,22,3), got {joints.shape}")

    if reorder_joints == "none":
        joint_order_meta = {"mode": "none", "note": "assume input is SMPL22 standard order"}
    elif reorder_joints == "smpl22_to_t2m22":
        joints = _reorder_smpl22_to_t2m22(joints)
        joint_order_meta = {"mode": "smpl22_to_t2m22", "map": SMPL22_TO_T2M22.tolist()}
    else:
        raise ValueError(f"Unsupported reorder_joints: {reorder_joints}")

    joints_yup, coord_meta = _coord_transform(joints, coord)

    T = joints_yup.shape[0]

    # ------------------------------------------------------------------
    # 1) Base features from HumanML3D semantic pipeline: (T-1,269)
    # ------------------------------------------------------------------
    if target_offset_mode == "from_first_frame":
        target_offset = torch.from_numpy(_compute_target_offset_from_first_frame(joints_yup)).float()
    elif target_offset_mode == "humanml3d_default":
        target_offset = get_target_offsets()
    else:
        raise ValueError(f"Unsupported --target-offset-mode: {target_offset_mode}")

    if T < 2:
        # Degenerate case: cannot run process_motion_generation (needs at least 2 frames).
        # Create placeholders and rely on rotation override for SMPL semantics.
        repr_T = np.zeros((T, 269), dtype=np.float32)
        if T == 1:
            repr_T[0, 3] = float(joints_yup[0, 0, 1])  # root_y
            repr_T[0, 259:263] = 0.5                   # foot_contact unknown
        base_meta = {"note": "T<2 fallback; no process_motion_generation"}
    else:
        base = process_motion_generation(
            positions=joints_yup,
            feet_thre=feet_thre,
            target_offset=target_offset,
        ).astype(np.float32)  # (T-1,269)
        if base.shape[0] != T - 1:
            raise RuntimeError(f"Expected base length T-1={T-1}, got {base.shape[0]}")

        # append 1 frame at the end to make (T,269)
        # Base row t corresponds to original frame t (static) + t->t+1 deltas.
        repr_T = np.zeros((T, 269), dtype=np.float32)
        repr_T[: T - 1] = base

        # Initialize last frame by copying the last available base frame (which corresponds to original t=T-2),
        # then apply explicit padding on delta/contact channels for the appended frame (no future frame).
        repr_T[T - 1] = repr_T[T - 2].copy()
        repr_T[T - 1, 0:3] = 0.0              # root_rot_vel + root_lin_vel
        repr_T[T - 1, 193:259] = 0.0          # local_vel
        repr_T[T - 1, 259:263] = 0.5          # foot_contact unknown

        # Make the appended frame's root_y and ric_data strictly match the canonical pipeline at frame T-1
        # (instead of copying from T-2).
        root_y_all, ric_all = _canonical_root_y_and_ric_all_frames(joints_yup, target_offset=target_offset)
        repr_T[T - 1, 3] = float(root_y_all[T - 1, 0])
        repr_T[T - 1, 4:67] = ric_all[T - 1]

        base_meta = {
            "base_source": "process_motion_generation",
            "base_shape": list(base.shape),
            "append_strategy": "frame(T-1) = copy(base[T-2]) then pad deltas/contact",
            "alignment": "row t aligns to frame t (start-aligned deltas)",
            "last_frame_rooty_ric": "computed via canonical pipeline (uniform_skeleton/floor/xz/faceZ+/IK/RIFKE)",
        }

    # ------------------------------------------------------------------
    # 2) Override rotations using MotionFix SMPL rots (per-frame, length T)
    # ------------------------------------------------------------------
    pose_aa_22 = _extract_pose_aa_22(motion_dict)  # (T,22,3)
    if reorder_joints == "smpl22_to_t2m22":
        pose_aa_22 = _reorder_pose_smpl22_to_t2m22(pose_aa_22)
    rotmat_22 = axis_angle_to_matrix_np(pose_aa_22)  # (T,22,3,3)
    rotmat_22, rot_basis_meta = _convert_rotmat_world_basis(rotmat_22, coord=coord)
    pose_6d_22 = matrix_to_cont6d_np(rotmat_22).astype(np.float32)  # (T,22,6)

    if pose_6d_22.shape[0] != T:
        raise RuntimeError(f"Length mismatch: joints={T} vs rots={pose_6d_22.shape[0]}")

    global_orient_6d = pose_6d_22[:, 0, :]  # (T,6)
    rot_data = pose_6d_22[:, 1:, :].reshape(T, -1)  # (T,126)

    if preserve_global_orient_only:
        # Only store global_orient; body pose set to identity (6D)
        rot_data = None  # keep whatever base provides

    repr_269 = repr_T.copy()
    # global orient always overridden (strict per-frame alignment)
    repr_269[:, 263:269] = global_orient_6d
    # body pose optionally overridden (strict per-frame alignment)
    if rot_data is not None:
        repr_269[:, 67:193] = rot_data

    meta = {
        "out_mode": "smpl_semantic_t",
        "coord": coord_meta,
        "joint_order": joint_order_meta,
        "target_offset_mode": target_offset_mode,
        "feet_thre": float(feet_thre),
        "padding": {
            "t_last_root_rot_velocity": 0.0,
            "t_last_root_linear_velocity": [0.0, 0.0],
            "t_last_local_vel": "all zeros",
            "t_last_foot_contact": 0.5,
        },
        "base": base_meta,
        "rotation": {
            "source": "motionfix_rots_axis_angle",
            "converted_to": "6d",
            "root_basis": rot_basis_meta,
            "preserve_global_orient_only": bool(preserve_global_orient_only),
        },
    }
    return repr_269, meta


def _convert_one_motion(
    motion_dict: Dict[str, Any],
    out_mode: str,
    coord: str,
    reorder_joints: str,
    target_offset_mode: str,
    feet_thre: float,
    preserve_rotation: bool,
    preserve_global_orient_only: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if out_mode == "smpl_semantic_t":
        # In SMPL semantic mode we output (T,269) and always take rotations from MotionFix rots.
        return _convert_one_motion_smpl_semantic_t(
            motion_dict=motion_dict,
            coord=coord,
            reorder_joints=reorder_joints,
            target_offset_mode=target_offset_mode,
            feet_thre=feet_thre,
            preserve_global_orient_only=preserve_global_orient_only,
        )

    joints = _as_numpy(motion_dict["joint_positions"])
    if joints.ndim != 3:
        raise ValueError(f"joint_positions must be (T,22,3), got {joints.shape}")
    # ensure (T,22,3)
    if joints.shape[-1] == 66 and joints.ndim == 2:
        joints = joints.reshape(joints.shape[0], 22, 3)
    if joints.shape[1:] != (22, 3):
        raise ValueError(f"joint_positions must be (T,22,3), got {joints.shape}")

    if reorder_joints == "none":
        joint_order_meta = {"mode": "none", "note": "assume input is SMPL22 standard order"}
    elif reorder_joints == "smpl22_to_t2m22":
        joints = _reorder_smpl22_to_t2m22(joints)
        joint_order_meta = {"mode": "smpl22_to_t2m22", "map": SMPL22_TO_T2M22.tolist()}
    else:
        raise ValueError(f"Unsupported reorder_joints: {reorder_joints}")

    joints, coord_meta = _coord_transform(joints, coord)

    if target_offset_mode == "from_first_frame":
        target_offset = torch.from_numpy(_compute_target_offset_from_first_frame(joints)).float()
    elif target_offset_mode == "humanml3d_default":
        target_offset = None
    else:
        raise ValueError(f"Unsupported --target-offset-mode: {target_offset_mode}")

    repr_269 = process_motion_generation(
        positions=joints,
        feet_thre=feet_thre,
        target_offset=target_offset,
    )

    rot_meta: Dict[str, Any] = {"preserve_rotation": bool(preserve_rotation), "preserve_global_orient_only": bool(preserve_global_orient_only)}
    if preserve_rotation:
        # Compute 6D from original SMPL axis-angle, then overwrite:
        # - rot_data (67:193) from joints 1..21
        # - global_orient_6d (263:269) from joint 0
        pose_aa_22 = _extract_pose_aa_22(motion_dict)  # (T,22,3)
        if reorder_joints == "smpl22_to_t2m22":
            pose_aa_22 = _reorder_pose_smpl22_to_t2m22(pose_aa_22)

        rotmat_22 = axis_angle_to_matrix_np(pose_aa_22)  # (T,22,3,3)
        rotmat_22, rot_basis_meta = _convert_rotmat_world_basis(rotmat_22, coord=coord)
        pose_6d_22 = matrix_to_cont6d_np(rotmat_22)  # (T,22,6)

        # Align to process_motion_generation output length (T-1)
        pose_6d_22 = pose_6d_22[:-1]
        if pose_6d_22.shape[0] != repr_269.shape[0]:
            raise RuntimeError(f"Length mismatch: pose_6d_22={pose_6d_22.shape[0]} vs repr_269={repr_269.shape[0]}")

        repr_269 = repr_269.copy()
        # global orient
        repr_269[:, 263:269] = pose_6d_22[:, 0]
        # body pose
        if not preserve_global_orient_only:
            repr_269[:, 67:193] = pose_6d_22[:, 1:].reshape(repr_269.shape[0], -1)

        rot_meta.update(
            {
                "rots_shape": list(_as_numpy(motion_dict["rots"]).shape),
                "rot_basis": rot_basis_meta,
            }
        )

    meta = {
        "coord": coord_meta,
        "joint_order": joint_order_meta,
        "target_offset_mode": target_offset_mode,
        "feet_thre": feet_thre,
        "rotation_override": rot_meta,
    }
    return repr_269, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MotionFix joblib dataset to 269 (HumanML3D semantic).")
    parser.add_argument("--in-joblib", type=str, required=True, help="Input MotionFix dataset joblib file (.pth.tar / .pkl)")
    parser.add_argument("--out-joblib", type=str, required=True, help="Output joblib file")
    parser.add_argument(
        "--out-mode",
        type=str,
        default="humanml3d_tminus1",
        choices=["humanml3d_tminus1", "smpl_semantic_t"],
        help="Output representation mode. humanml3d_tminus1 outputs (T-1,269). smpl_semantic_t outputs (T,269) with SMPL rotation semantics + t=0 padding for deltas.",
    )
    parser.add_argument("--coord", type=str, default="zup_to_yup", choices=["none", "zup_to_yup"], help="Coordinate basis transform")
    parser.add_argument(
        "--reorder-joints",
        type=str,
        default="none",
        choices=["none", "smpl22_to_t2m22"],
        help="Optional joint order remapping. Default assumes MotionFix joint_positions is SMPL22 standard order.",
    )
    parser.add_argument(
        "--target-offset-mode",
        type=str,
        default="from_first_frame",
        choices=["from_first_frame", "humanml3d_default"],
        help="How to choose target skeleton offsets inside process_motion_generation",
    )
    parser.add_argument("--feet-thre", type=float, default=0.002, help="Foot contact threshold (same meaning as HumanML3D)")
    parser.add_argument(
        "--preserve-rotation",
        action="store_true",
        help="(Only for out-mode=humanml3d_tminus1) Override rot_data (67:193) and global_orient_6d (263:269) using MotionFix original SMPL axis-angle 'rots' (6D after conversion).",
    )
    parser.add_argument(
        "--preserve-global-orient-only",
        action="store_true",
        help="If set, only override global_orient_6d (263:269); keep rot_data (67:193) from process_motion_generation (HumanML3D IK semantic).",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Only convert first N samples (debug)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite out file if exists")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    args = parser.parse_args()

    if os.path.exists(args.out_joblib) and (not args.overwrite):
        raise FileExistsError(f"Output exists: {args.out_joblib}. Use --overwrite to overwrite.")

    print(f"[convert_motionfix_to_269] Loading: {args.in_joblib}")
    data = joblib.load(args.in_joblib)

    # allow dict or list
    if isinstance(data, dict):
        items = list(data.items())
        out_is_dict = True
    elif isinstance(data, list):
        items = list(enumerate(data))
        out_is_dict = False
    else:
        raise TypeError(f"Unsupported dataset type: {type(data)} (expected dict or list)")

    if args.max_samples is not None:
        items = items[: args.max_samples]

    print(f"[convert_motionfix_to_269] Workers: {args.num_workers}")

    def process_one_sample(key, sample, out_mode, coord, reorder_joints, target_offset_mode,
                           feet_thre, preserve_rotation, preserve_global_orient_only):
        """Process a single sample"""
        if "motion_source" not in sample or "motion_target" not in sample:
            raise KeyError(f"Sample {key} missing motion_source/motion_target keys")

        repr_src, meta_src = _convert_one_motion(
            sample["motion_source"],
            out_mode=out_mode,
            coord=coord,
            reorder_joints=reorder_joints,
            target_offset_mode=target_offset_mode,
            feet_thre=feet_thre,
            preserve_rotation=preserve_rotation,
            preserve_global_orient_only=preserve_global_orient_only,
        )
        repr_tgt, meta_tgt = _convert_one_motion(
            sample["motion_target"],
            out_mode=out_mode,
            coord=coord,
            reorder_joints=reorder_joints,
            target_offset_mode=target_offset_mode,
            feet_thre=feet_thre,
            preserve_rotation=preserve_rotation,
            preserve_global_orient_only=preserve_global_orient_only,
        )

        sample_out = dict(sample)
        sample_out["motion_source"] = dict(sample_out["motion_source"])
        sample_out["motion_target"] = dict(sample_out["motion_target"])
        sample_out["motion_source"]["repr_269"] = repr_src
        sample_out["motion_target"]["repr_269"] = repr_tgt

        return key, sample_out, {"source": meta_src, "target": meta_tgt}

    out_items = []
    per_sample_meta = {}

    if args.num_workers <= 1:
        # Single-process mode
        for key, sample in tqdm(items, desc="Converting to 269", unit="sample"):
            key, sample_out, meta = process_one_sample(
                key, sample, args.out_mode, args.coord, args.reorder_joints,
                args.target_offset_mode, args.feet_thre,
                bool(args.preserve_rotation), bool(args.preserve_global_orient_only)
            )
            out_items.append((key, sample_out))
            per_sample_meta[str(key)] = meta
    else:
        # Multi-process mode
        def worker_fn(task_args):
            idx, key, sample, out_mode, coord, reorder_joints, target_offset_mode, \
                feet_thre, preserve_rotation, preserve_global_orient_only = task_args
            key, sample_out, meta = process_one_sample(
                key, sample, out_mode, coord, reorder_joints,
                target_offset_mode, feet_thre, preserve_rotation, preserve_global_orient_only
            )
            return idx, key, sample_out, meta

        task_args_list = [
            (i, key, sample, args.out_mode, args.coord, args.reorder_joints,
             args.target_offset_mode, args.feet_thre,
             bool(args.preserve_rotation), bool(args.preserve_global_orient_only))
            for i, (key, sample) in enumerate(items)
        ]

        results = [None] * len(items)
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(worker_fn, arg): arg[0] for arg in task_args_list}
            for future in tqdm(as_completed(futures), total=len(futures),
                              desc=f"Converting to 269 ({args.num_workers} workers)", unit="sample"):
                idx = futures[future]
                try:
                    res_idx, key, sample_out, meta = future.result()
                    results[res_idx] = (key, sample_out, meta)
                except Exception as e:
                    print(f"\nError processing sample {idx}: {e}")
                    raise

        for key, sample_out, meta in results:
            out_items.append((key, sample_out))
            per_sample_meta[str(key)] = meta

    if out_is_dict:
        out_data = {k: v for k, v in out_items}
    else:
        # list: restore ordering by integer keys
        out_data = [v for _, v in out_items]

    out_bundle = {
        "data": out_data,
        "meta": {
            "converter": "scripts/convert_motionfix_to_269.py",
            "out_mode": args.out_mode,
            "coord": args.coord,
        "reorder_joints": args.reorder_joints,
            "target_offset_mode": args.target_offset_mode,
            "feet_thre": args.feet_thre,
            "preserve_rotation": bool(args.preserve_rotation),
            "preserve_global_orient_only": bool(args.preserve_global_orient_only),
            "per_sample": per_sample_meta,
        },
    }

    print(f"[convert_motionfix_to_269] Saving: {args.out_joblib}")
    joblib.dump(out_bundle, args.out_joblib)
    print("[convert_motionfix_to_269] Done.")


if __name__ == "__main__":
    main()
