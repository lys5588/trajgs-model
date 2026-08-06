"""
Convert the Human3.6M dataset to the 269-dim unified representation
===================================================================

This script converts Human3.6M data from the WHAM format into the 269-dim
unified human representation. Every frame is processed independently
(single-frame mode), which is suitable for image-based single-frame mesh
reconstruction tasks.

Input:  WHAM-preprocessed train file (e.g. human36m_train_vit.pth)
        or TCMR-preprocessed test file (e.g. h36m_test_25fps_db.pt)
Output: joblib .pth file with a per-frame repr_269 array

Data format notes:
- The input contains: pose (SMPL axis-angle), betas, trans, joints3D_smpl,
  features, etc.
- Note: joints3D_smpl is in a 31-joint format (COCO 17 + common J14), NOT the
  native SMPL joints!
- The correct SMPL 22 joints must be computed from pose/betas through the SMPL
  model.
- Output: one 269-dim vector per frame, produced in
  process_mesh_reconstruction mode.
"""

import os
import sys
import argparse
import joblib
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# Add the data_preparation directory (which contains unified_representation_269.py)
# to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_representation_269 import (
    process_mesh_reconstruction,
    set_target_skeleton_path,
    axis_angle_to_matrix_np,
    matrix_to_cont6d_np,
    Repr269,
    matrix_to_axis_angle_np,
    cont6d_to_matrix_np,
)

# Try to import smplx
try:
    import smplx
    SMPLX_AVAILABLE = True
except ImportError:
    SMPLX_AVAILABLE = False
    print("Warning: smplx is not installed; joint positions cannot be computed from SMPL parameters")


# SMPL 24-joint to HumanML3D 22-joint mapping
# HumanML3D uses the first 22 SMPL joints (drops hand joints 22 and 23)
SMPL_TO_HUMANML3D_22 = list(range(22))


def convert_to_humanml3d_coords(joints: np.ndarray) -> np.ndarray:
    """
    Convert joint positions to the HumanML3D "Y-up right-handed" convention.

    The transform used here is:
      C = diag([1, -1, -1]) = R_x(pi)

    which is equivalent to a **180-degree rotation about the X axis**. It maps
    the common "Y-down" coordinate system to "Y-up" while keeping the frame
    right-handed.

    Note: flipping only Y (Y=-Y) would turn the right-handed frame into a
    left-handed one and break rotations/cross products; therefore both Y and Z
    must be flipped.

    Args:
        joints: (N, 22, 3) or (22, 3) joint positions

    Returns:
        joints_converted: converted joint positions (head pointing to +Y)
    """
    if joints.shape[-1] != 3:
        raise ValueError(f"Expected joints[..., 3], got {joints.shape}")
    joints_converted = joints.copy()
    joints_converted[..., 1] = -joints[..., 1]  # Y = -Y
    joints_converted[..., 2] = -joints[..., 2]  # Z = -Z
    return joints_converted


# Used to convert a "Y-down" coordinate system to "Y-up" (keeping the frame
# right-handed). In practice this equals a 180-degree rotation about the X
# axis: x unchanged, y/z negated.
ROT_X_PI = np.array([
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
], dtype=np.float32)


def convert_rotmat_to_humanml3d_coords(rotmat: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrices to the same coordinate convention as
    `convert_to_humanml3d_coords()`.

    We apply the same right-handed basis change C = R_x(pi) to the whole 3D
    space. For any rotation matrix R, its representation in the new basis is:

      R' = C . R . C^{-1} = C . R . C^T

    which guarantees, for any vector v:
      C (R v) = (C R C^T) (C v)

    Args:
        rotmat: (..., 3, 3)
    Returns:
        rotmat_converted: (..., 3, 3)
    """
    C = ROT_X_PI
    if rotmat.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotmat[..., 3, 3], got {rotmat.shape}")
    # numpy.matmul broadcasts automatically over any leading dims.
    # NOTE: only the root (global_orient) is left-multiplied by C here; the
    # body pose is local (expressed in parent-joint frames) and left unchanged.
    # return C @ rotmat @ C.T
    rotmat_body = rotmat[..., 1:, :, :]
    rotmat_root = rotmat[..., :1, :, :]
    rotmat_root = C @ rotmat_root
    return np.concatenate([rotmat_root, rotmat_body], axis=-3)


# SMPL 24-joint names (for reference)
SMPL_24_JOINTS = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
]


def create_smpl_model(model_path: str, gender: str = 'neutral', batch_size: int = 1):
    """
    Create the SMPL model.

    Args:
        model_path: directory containing the SMPL model files
        gender: gender ('male', 'female', 'neutral')
        batch_size: batch size

    Returns:
        SMPL model instance
    """
    if not SMPLX_AVAILABLE:
        raise RuntimeError("smplx is not installed. Please run: pip install smplx")

    smpl = smplx.SMPLH(
        model_path,
        num_betas=10,
        ext='pkl',
        use_pca=False,  # use full axis-angle instead of PCA
        flat_hand_mean=True,  # default hand pose is a flat hand
    )
    return smpl


def compute_smpl_joints(smpl_model,
                        pose_aa: np.ndarray,
                        betas: np.ndarray = None,
                        trans: np.ndarray = None,
                        device: str = 'cpu') -> np.ndarray:
    """
    Compute the 22 joint positions through the SMPL model.

    Args:
        smpl_model: SMPL model instance
        pose_aa: (N, 24, 3) or (N, 72) axis-angle pose parameters
        betas: (N, 10) shape parameters; defaults used if None
        trans: (N, 3) translation parameters; defaults used if None
        device: computation device

    Returns:
        joints_22: (N, 22, 3) first 22 SMPL joint positions
    """
    smpl_model = smpl_model.to(device)

    # Handle pose parameters
    if pose_aa.ndim == 3:
        # (N, 24, 3) -> (N, 72)
        pose_aa = pose_aa.reshape(-1, 72)

    N = pose_aa.shape[0]

    # Convert to tensors
    pose_tensor = torch.from_numpy(pose_aa).float().to(device)
    global_orient = pose_tensor[:, :3]  # (N, 3)
    body_pose = pose_tensor[:, 3:66]  # (N, 63)

    if betas is not None:
        betas_tensor = torch.from_numpy(betas).float().to(device)
    else:
        betas_tensor = torch.zeros(N, 10).float().to(device)

    if trans is not None:
        trans_tensor = torch.from_numpy(trans).float().to(device)
    else:
        trans_tensor = torch.zeros(N, 3).float().to(device)

    # SMPLH requires hand pose parameters; use zeros (flat hands)
    left_hand_pose = torch.zeros(N, 45).float().to(device)   # 15 joints * 3
    right_hand_pose = torch.zeros(N, 45).float().to(device)  # 15 joints * 3

    # Run the SMPL model
    with torch.no_grad():
        output = smpl_model(
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            betas=betas_tensor,
            transl=trans_tensor,
        )

    # Get the joint positions (native SMPL output has 24 joints)
    joints = output.joints.cpu().numpy()  # (N, 24, 3) or (N, 45, 3)

    # Take the first 22 joints (drop the hand joints)
    joints_22 = joints[:, :22, :]  # (N, 22, 3)

    return joints_22


def compute_smpl_joints_batch(smpl_model,
                               poses: np.ndarray,
                               betas: np.ndarray = None,
                               trans: np.ndarray = None,
                               batch_size: int = 512,
                               device: str = 'cpu') -> np.ndarray:
    """
    Compute SMPL joint positions in batches.

    Args:
        smpl_model: SMPL model instance
        poses: (N, 24, 3) or (N, 72) axis-angle pose parameters
        betas: (N, 10) shape parameters
        trans: (N, 3) translation parameters
        batch_size: number of samples per batch
        device: computation device

    Returns:
        joints_22: (N, 22, 3) first 22 SMPL joint positions
    """
    N = poses.shape[0]
    all_joints = []

    for start in tqdm(range(0, N, batch_size), desc="Computing SMPL joints"):
        end = min(start + batch_size, N)

        pose_batch = poses[start:end]
        betas_batch = betas[start:end] if betas is not None else None
        trans_batch = trans[start:end] if trans is not None else None

        joints_batch = compute_smpl_joints(
            smpl_model, pose_batch, betas_batch, trans_batch, device
        )
        all_joints.append(joints_batch)

    return np.concatenate(all_joints, axis=0)


def convert_single_frame_from_joints(
    joints_22: np.ndarray,
    pose_aa: np.ndarray = None,
    preserve_rotation: bool = True,
    preserve_global_orient_only: bool = False,
    flip_coords: bool = True,
) -> np.ndarray:
    """
    Convert 22 joint positions to the 269-dim representation (single frame).

    Args:
        joints_22: (22, 3) joint positions
        pose_aa: (24, 3) optional SMPL axis-angle used to preserve the
                 original rotations
        preserve_rotation: whether to keep the original SMPL rotation parameters
        preserve_global_orient_only: if True, only global_orient_6d (263:269)
                                     is overridden; rot_data (67:193) keeps the
                                     normalized semantics of
                                     process_mesh_reconstruction.

    Returns:
        repr_269: (269,) 269-dim representation
    """
    # Process the single frame with process_mesh_reconstruction (applies
    # facing normalization)
    result = process_mesh_reconstruction(joints_22)  # (1, 269)
    repr_269 = result[0]  # (269,)

    # If the original rotation parameters are provided and should be kept,
    # override the rotation-related dims
    if preserve_rotation and pose_aa is not None:
        # Convert axis-angle to rotation matrices, then to the 6D representation
        pose_rotmat = axis_angle_to_matrix_np(pose_aa[..., :22, :])  # (22, 3, 3)
        # If the joints were coordinate-flipped (180 degrees about X), the
        # rotations must undergo the same basis change
        if flip_coords:
            pose_rotmat = convert_rotmat_to_humanml3d_coords(pose_rotmat)
        pose_6d = matrix_to_cont6d_np(pose_rotmat)  # (22, 6)

        # Update global_orient (indices 263-269)
        repr_269[263:269] = pose_6d[0]  # global_orient

        # Optional: whether to also override rot_data (indices 67-193).
        # If the goal is "stay as close as possible to the motion-generation
        # 269 semantics", it is recommended to only override global_orient and
        # leave rot_data untouched.
        if not preserve_global_orient_only:
            repr_269[67:193] = pose_6d[1:].reshape(-1)  # 21 joints * 6 = 126

    return repr_269


def convert_single_frame_from_smpl(
    pose_aa: np.ndarray,
    joints_22: np.ndarray = None,
    preserve_global_orient_only: bool = False,
    flip_coords: bool = True,
) -> np.ndarray:
    """
    Convert SMPL parameters to the 269-dim representation (single frame).

    Args:
        pose_aa: (24, 3) SMPL axis-angle rotation parameters
        joints_22: (22, 3) optional joint positions; if not provided a default
                   standard body is used

    Returns:
        repr_269: (269,) 269-dim representation
    """
    if joints_22 is not None:
        # Use the provided joint positions
        result = process_mesh_reconstruction(joints_22)
        repr_269 = result[0]  # (269,)

        # Replace the rotation data with the actual SMPL rotations
        # Convert axis-angle to the 6D representation
        pose_rotmat = axis_angle_to_matrix_np(pose_aa[:22])  # (22, 3, 3)
        if flip_coords:
            pose_rotmat = convert_rotmat_to_humanml3d_coords(pose_rotmat)
        pose_6d = matrix_to_cont6d_np(pose_rotmat)  # (22, 6)

        # Update global_orient (indices 263-269)
        repr_269[263:269] = pose_6d[0]  # global_orient

        if not preserve_global_orient_only:
            # Update rot_data (indices 67-193)
            repr_269[67:193] = pose_6d[1:].reshape(-1)  # 21 joints * 6 = 126

        return repr_269
    else:
        raise NotImplementedError("No joint positions available; the simplified path is not supported")


def convert_human36m_dataset(input_path: str,
                              output_path: str,
                              smpl_model_path: str = None,
                              preserve_rotation: bool = True,
                              preserve_global_orient_only: bool = False,
                              flip_coords: bool = True,
                              batch_size: int = 512,
                              max_samples: int = None,
                              device: str = 'cpu'):
    """
    Convert the whole Human3.6M dataset.

    Args:
        input_path: input file path (e.g. human36m_train_vit.pth)
        output_path: output file path
        smpl_model_path: SMPL model path (directory with SMPL_NEUTRAL.pkl etc.)
        preserve_rotation: whether to keep the original SMPL rotation
                          parameters (overrides rot_data/global_orient in the
                          output).
                          Note: with flip_coords=True this script applies the
                          same coordinate basis change to the SMPL rotations so
                          they stay consistent with the flipped joints
                          (preserve_rotation is no longer force-disabled).
        preserve_global_orient_only: whether to keep only the SMPL global
                                     rotation in global_orient_6d (263:269);
                                     rot_data (67:193) keeps the normalized
                                     semantics of process_mesh_reconstruction.
        flip_coords: whether to apply the coordinate change (180-degree
                     rotation about X so the head points to +Y, consistent
                     with HumanML3D)
        batch_size: batch size (for the SMPL computation)
        max_samples: max number of samples to process (for testing)
        device: computation device ('cpu' or 'cuda')
    """
    print(f"Loading data: {input_path}")
    data = joblib.load(input_path)

    total_samples = len(data['pose'])
    if max_samples is not None:
        total_samples = min(total_samples, max_samples)

    # Previously flip_coords=True forced preserve_rotation off.
    # Now the same basis change (180 degrees about X) is applied to the SMPL
    # rotation matrices, so preserving rotations and flipping coordinates can
    # coexist.
    if flip_coords and preserve_rotation:
        print("Note: flip_coords=True and preserve_rotation=True; the SMPL rotations will also undergo the Y-down -> Y-up basis change (180 degrees about X).")

    # Extra note: preserve_rotation replaces the rotation-related dims of the
    # output with the SMPL pose parameters. This changes the semantics of the
    # 269 representation (especially global_orient_6d) and is not strictly
    # consistent with the facing normalization inside
    # process_mesh_reconstruction. If the goal is to stay close to the
    # motion-generation 269 physics, preserve_rotation should usually be off.
    if preserve_rotation:
        print("\nNote: preserve_rotation=True replaces the rotation-related dims of repr_269 with the original SMPL pose parameters:")
        print("  - global_orient_6d (263:269) comes from the SMPL global orientation")
        if preserve_global_orient_only:
            print("  - rot_data (67:193) keeps the normalized semantics of process_mesh_reconstruction (not overridden)")
        else:
            print("  - rot_data (67:193) comes from the SMPL body_pose (local joint rotations)")
        print("This differs from the facing normalization of process_mesh_reconstruction (each frame facing Z+, root aligned)")
        print("and may introduce a cross-domain difference similar to a constant ~180-degree yaw bias; for training a unified model consider --no-preserve-rotation.\n")

    print(f"Total samples: {total_samples}")
    print(f"Coordinate flip (head to +Y): {flip_coords}")
    print(f"Preserve original rotations: {preserve_rotation}")
    if preserve_rotation:
        print(f"Preserve global_orient only: {preserve_global_orient_only}")
    print(f"Device: {device}")

    # Extract the required data
    poses = data['pose'][:total_samples]  # (N, 24, 3) axis-angle
    if isinstance(poses, torch.Tensor):
        poses = poses.numpy()

    # betas/trans:
    # For a fair comparison against methods that "use only the SMPL pose
    # parameters", betas/trans are deliberately NOT used to compute the joints
    # here. Even if the input contains them, they are ignored (equivalent to
    # betas=0, trans=0).
    betas = None
    trans = None
    if 'betas' in data:
        print("Note: input contains betas, but for fair comparison (pose only) they are ignored (equivalent to 0).")
    else:
        print("Note: input has no betas; default shape parameters (0) are used.")
    if 'trans' in data:
        print("Note: input contains trans, but for fair comparison (pose only) it is ignored (equivalent to 0).")
    else:
        print("Note: input has no trans; default translation (0) is used.")

    # Compute the correct 22 joint positions through the SMPL model.
    # Note: joints3D_smpl has 31 joints (COCO 17 + common J14) and cannot be
    # used directly!
    if smpl_model_path is not None:
        print(f"\nComputing joint positions with the SMPL model: {smpl_model_path}")
        smpl_model = create_smpl_model(smpl_model_path)
        joints_22 = compute_smpl_joints_batch(
            smpl_model, poses, betas, trans,
            batch_size=batch_size, device=device
        )
        print(f"Done. Joints shape: {joints_22.shape}")
        print(f"Original joints Y range: [{joints_22[:, :, 1].min():.3f}, {joints_22[:, :, 1].max():.3f}]")

        # Coordinate change: convert head-to-negative-Y into head-to-positive-Y
        # (consistent with HumanML3D)
        if flip_coords:
            print("Coordinate change: rotating 180 degrees about the X axis (head to +Y)...")
            joints_22 = convert_to_humanml3d_coords(joints_22)
            print(f"Converted joints Y range: [{joints_22[:, :, 1].min():.3f}, {joints_22[:, :, 1].max():.3f}]")
    else:
        joints_22 = None
        print("Warning: no SMPL model path provided; only rotation parameters will be used (no joint positions)")

    # Store the conversion results
    repr_269_list = []

    print("Starting conversion...")
    for i in tqdm(range(total_samples), desc="Converting"):
        pose_aa = poses[i]  # (24, 3)
        if pose_aa.shape[-1] != 3:
            pose_aa = pose_aa.reshape(*pose_aa.shape[:-1], pose_aa.shape[-1] // 3, 3)
        j22 = joints_22[i] if joints_22 is not None else None  # (22, 3) or None

        try:
            if j22 is not None:
                repr_269 = convert_single_frame_from_joints(
                    j22,
                    pose_aa=pose_aa,
                    preserve_rotation=preserve_rotation,
                    preserve_global_orient_only=preserve_global_orient_only,
                    flip_coords=flip_coords,
                )
            else:
                raise NotImplementedError

            repr_269_list.append(repr_269)
        except Exception as e:
            print(f"Warning: sample {i} conversion failed: {e}")
            # Use a zero vector as fallback
            repr_269_list.append(np.zeros(269))

    # Convert to a tensor
    repr_269_array = np.stack(repr_269_list, axis=0)  # (N, 269)
    repr_269_tensor = torch.from_numpy(repr_269_array).float()

    print(f"\nConversion done!")
    print(f"Output shape: {repr_269_tensor.shape}")

    # Create the output data dict
    output_data = defaultdict(lambda: None)

    # Copy the fields of the original data that should be kept
    keys_to_copy = ['bbox', 'joints2D', 'res', 'vid', 'frame_id',
                    'features', 'betas', 'cam_poses']
    for key in keys_to_copy:
        if key in data:
            output_data[key] = data[key][:total_samples]

    # Add the 269-dim representation
    output_data['repr_269'] = repr_269_tensor

    # Record the coordinate/rotation transform metadata so evaluation can
    # invert it
    output_data['coord_transform'] = {
        'flip_coords': bool(flip_coords),
        'rotation_basis_change': 'Rx(pi) / diag([1,-1,-1])' if flip_coords else 'none',
        'preserve_rotation': bool(preserve_rotation),
        'preserve_global_orient_only': bool(preserve_global_orient_only),
        'note': 'If flip_coords=True, joints and (optionally preserved) SMPL rotations are represented in Y-up via a basis change. '
                'To compare with methods expecting original Y-down, apply the inverse (same Rx(pi)) to rotations/3D joints.',
    }

    # Keep the original pose for verification (trans may not exist, and this
    # script ignores trans by default)
    output_data['pose_original'] = data['pose'][:total_samples]
    if 'trans' in data:
        output_data['trans_original'] = data['trans'][:total_samples]

    # Save
    print(f"\nSaving to: {output_path}")

    # Convert to a plain dict
    output_dict = dict(output_data)

    # Save with joblib (supports large files)
    joblib.dump(output_dict, output_path)

    print("Saved!")

    # Verify
    print("\nVerifying output...")
    loaded = joblib.load(output_path)
    print(f"Loaded repr_269 shape: {loaded['repr_269'].shape}")

    # Show some statistics
    r269 = loaded['repr_269'].numpy()
    print(f"\n269-dim representation statistics:")
    print(f"  mean: {r269.mean():.4f}")
    print(f"  std: {r269.std():.4f}")
    print(f"  min: {r269.min():.4f}")
    print(f"  max: {r269.max():.4f}")

    # Check for NaN
    nan_count = np.isnan(r269).sum()
    if nan_count > 0:
        print(f"  Warning: {nan_count} NaN values found")
    else:
        print(f"  No NaN values")

    return output_dict


def validate_conversion(original_data: dict, converted_data: dict,
                        sample_indices: list = None):
    """
    Validate the conversion results.

    Args:
        original_data: original data
        converted_data: converted data
        sample_indices: list of sample indices to validate
    """
    if sample_indices is None:
        sample_indices = [0, 100, 1000]

    repr_269 = converted_data['repr_269'].numpy()

    print("\nConversion validation:")
    print("=" * 60)

    for idx in sample_indices:
        if idx >= len(repr_269):
            continue

        r269 = repr_269[idx]

        print(f"\nSample {idx}:")

        # Validate the data structure
        validation = Repr269.validate(r269)
        print(f"  validation: {'passed' if validation['valid'] else 'failed'}")
        if validation['errors']:
            print(f"  errors: {validation['errors']}")
        if validation['warnings']:
            print(f"  warnings: {validation['warnings']}")

        # Inspect the individual components
        print(f"  root_rot_vel: {r269[0]:.4f}")
        print(f"  root_lin_vel: {r269[1:3]}")
        print(f"  root_y: {r269[3]:.4f}")
        print(f"  ric_data range: [{r269[4:67].min():.4f}, {r269[4:67].max():.4f}]")
        print(f"  rot_data range: [{r269[67:193].min():.4f}, {r269[67:193].max():.4f}]")
        print(f"  foot_contact: {r269[259:263]}")
        print(f"  global_orient: {r269[263:269]}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert the Human3.6M dataset to the 269-dim unified representation'
    )
    parser.add_argument(
        '--input', '-i', type=str, required=True,
        help='input file path (WHAM train .pth or TCMR test .pt)'
    )
    parser.add_argument(
        '--output', '-o', type=str, required=True,
        help='output file path'
    )
    parser.add_argument(
        '--smpl-model', '-s', type=str, default="data/body_models/smplh",
        help='SMPL model directory (containing the SMPLH .pkl files)'
    )
    parser.add_argument(
        '--max-samples', '-n', type=int, default=None, help='max number of samples to process (for testing)'
    )
    parser.add_argument(
        '--batch-size', '-b', type=int, default=512, help='batch size for the SMPL computation'
    )
    parser.add_argument(
        '--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='computation device'
    )
    parser.add_argument(
        '--no-preserve-rotation', action='store_true', help='do not keep the original SMPL rotations (use normalized facing)'
    )
    parser.add_argument(
        '--preserve-global-orient-only', action='store_true',
        help='keep only the SMPL global rotation in global_orient_6d (263:269); rot_data (67:193) keeps the normalized semantics'
    )
    parser.add_argument(
        '--no-flip-coords', action='store_true',
        help='do not flip coordinates (by default the head is flipped to +Y, consistent with HumanML3D)'
    )
    parser.add_argument(
        '--target-skeleton', type=str, default=None, help='target skeleton reference file path (HumanML3D joints/000021.npy)'
    )
    parser.add_argument(
        '--validate-only', action='store_true', help='only validate already-converted data'
    )

    args = parser.parse_args()

    # Set the target skeleton path if provided
    if args.target_skeleton:
        set_target_skeleton_path(args.target_skeleton)

    if args.validate_only:
        # Validation-only mode
        print("Validation mode")
        print(f"Loading: {args.output}")
        converted_data = joblib.load(args.output)

        print(f"Loading: {args.input}")
        original_data = joblib.load(args.input)

        validate_conversion(original_data, converted_data)
    else:
        # Conversion mode
        if args.smpl_model is None:
            print("Warning: no --smpl-model given; joint positions cannot be computed")
            print("Suggested: --smpl-model /path/to/smpl/models")

        converted_data = convert_human36m_dataset(
            input_path=args.input,
            output_path=args.output,
            smpl_model_path=args.smpl_model,
            preserve_rotation=not args.no_preserve_rotation,
            preserve_global_orient_only=bool(args.preserve_global_orient_only),
            flip_coords=not args.no_flip_coords,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            device=args.device
        )

        # Validate
        original_data = joblib.load(args.input)
        validate_conversion(original_data, converted_data)


if __name__ == '__main__':
    main()
