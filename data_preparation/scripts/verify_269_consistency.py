"""
Verify the consistency of the HumanML3D and Human3.6M 269-dim representations.

This script analyzes the differences between the two processing modes and
checks whether their physical semantics are truly unified.

Analysis focus:
1. Whether global_orient_6d is a normalized rotation in both modes
2. Whether rot_data (joint rotations) is computed the same way
3. Whether ric_data (joint positions) uses a unified reference frame
4. Whether the coordinate systems are unified (head to +Y, facing +Z)

See also docs/269_consistency_analysis.md for the written conclusions.

Usage (run from the repository root):
  python data_preparation/scripts/verify_269_consistency.py \
    [--target-skeleton data/HumanML3D/joints/000021.npy] \
    [--humanml3d-sample data/HumanML3D/HumanML3D/new_joint_vecs/000021.npy] \
    [--h36m-269 path/to/human36m_train_269_full.pth]
"""

import os
import sys
import argparse
import numpy as np
import torch
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_representation_269 import (
    process_motion_generation,
    process_mesh_reconstruction,
    Skeleton,
    T2M_RAW_OFFSETS,
    T2M_KINEMATIC_CHAIN,
    FACE_JOINT_INDX,
    set_target_skeleton_path,
    quaternion_to_cont6d_np,
    cont6d_to_matrix_np,
    matrix_to_axis_angle_np,
    qrot_np,
    qbetween_np,
    _get_skeleton_and_offsets,
    _n_raw_offsets,
    _kinematic_chain,
    Repr269,
)


def create_test_motion(T: int = 30, rotation_range: float = np.pi/4) -> np.ndarray:
    """
    Create a test motion sequence.

    Generates a person starting facing +X and gradually turning towards +Y.

    Args:
        T: number of frames
        rotation_range: rotation range (radians)

    Returns:
        positions: (T, 22, 3) joint position sequence
    """
    # Get the skeleton
    _, target_offset = _get_skeleton_and_offsets()
    skel = Skeleton(_n_raw_offsets, _kinematic_chain, "cpu")
    skel.set_offset(target_offset)

    # T-pose quaternions
    default_quat = np.zeros((1, 22, 4))
    default_quat[..., 0] = 1.0
    default_root_pos = np.array([[0.0, 0.9, 0.0]])

    # Generate the T-pose joint positions
    t_pose = skel.forward_kinematics_np(default_quat, default_root_pos)[0]  # (22, 3)

    positions = []
    for t in range(T):
        # Rotate the person about the Y axis
        angle = rotation_range * t / (T - 1) if T > 1 else 0
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        # Y-axis rotation matrix
        rot_mat = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])

        # Rotate all joints
        pos_t = (rot_mat @ t_pose.T).T  # (22, 3)

        # Add some translation
        pos_t[:, 0] += t * 0.02  # move along X

        positions.append(pos_t)

    return np.stack(positions, axis=0)  # (T, 22, 3)


def analyze_global_orient(repr_269: np.ndarray, name: str):
    """Analyze the statistics of global_orient_6d"""
    print(f"\n{'='*60}")
    print(f"Analyzing global_orient_6d of {name}")
    print(f"{'='*60}")

    if len(repr_269.shape) == 1:
        repr_269 = repr_269[np.newaxis, :]

    global_orient_6d = repr_269[:, 263:269]  # (N, 6)

    # Convert to rotation matrices
    rotmat = cont6d_to_matrix_np(global_orient_6d)  # (N, 3, 3)

    # Check whether they are close to the identity
    identity = np.eye(3)
    diff_from_identity = np.abs(rotmat - identity[np.newaxis, :, :])

    print(f"Samples: {len(global_orient_6d)}")
    print(f"\n6D rotation statistics:")
    print(f"  mean: {global_orient_6d.mean(axis=0)}")
    print(f"  std: {global_orient_6d.std(axis=0)}")

    print(f"\nDifference from the identity rotation:")
    print(f"  mean diff: {diff_from_identity.mean():.6f}")
    print(f"  max diff: {diff_from_identity.max():.6f}")

    # Extract the rotation angles (axis-angle representation)
    axis_angle = matrix_to_axis_angle_np(rotmat)  # (N, 3)
    rotation_angles = np.linalg.norm(axis_angle, axis=-1)  # (N,)
    rotation_angles_deg = np.degrees(rotation_angles)

    print(f"\nRotation angles (degrees):")
    print(f"  mean: {rotation_angles_deg.mean():.2f}")
    print(f"  std: {rotation_angles_deg.std():.2f}")
    print(f"  min: {rotation_angles_deg.min():.2f}")
    print(f"  max: {rotation_angles_deg.max():.2f}")

    # Analyze the dominant rotation axis
    if rotation_angles.max() > 0.01:  # significant rotation present
        # Find the frame with the largest rotation angle
        max_idx = np.argmax(rotation_angles)
        axis = axis_angle[max_idx] / (rotation_angles[max_idx] + 1e-8)
        print(f"\nFrame with the largest rotation ({max_idx}):")
        print(f"  axis: {axis}")
        print(f"  angle: {rotation_angles_deg[max_idx]:.2f} degrees")

    return {
        'mean_6d': global_orient_6d.mean(axis=0),
        'std_6d': global_orient_6d.std(axis=0),
        'mean_angle_deg': rotation_angles_deg.mean(),
        'max_angle_deg': rotation_angles_deg.max(),
    }


def analyze_rot_data(repr_269: np.ndarray, name: str):
    """Analyze the statistics of rot_data (21 joint rotations)"""
    print(f"\n{'='*60}")
    print(f"Analyzing rot_data (joint rotations) of {name}")
    print(f"{'='*60}")

    if len(repr_269.shape) == 1:
        repr_269 = repr_269[np.newaxis, :]

    rot_data = repr_269[:, 67:193].reshape(-1, 21, 6)  # (N, 21, 6)

    # Convert to rotation matrices
    rotmat = cont6d_to_matrix_np(rot_data)  # (N, 21, 3, 3)

    # Extract the rotation angles
    axis_angle = matrix_to_axis_angle_np(rotmat)  # (N, 21, 3)
    rotation_angles = np.linalg.norm(axis_angle, axis=-1)  # (N, 21)
    rotation_angles_deg = np.degrees(rotation_angles)

    print(f"Samples: {rot_data.shape[0]}")
    print(f"\nMean rotation angle per joint (degrees):")
    for j in range(21):
        mean_angle = rotation_angles_deg[:, j].mean()
        std_angle = rotation_angles_deg[:, j].std()
        print(f"  joint {j:2d}: {mean_angle:6.2f} +/- {std_angle:5.2f}")

    return {
        'mean_angle_per_joint': rotation_angles_deg.mean(axis=0),
        'std_angle_per_joint': rotation_angles_deg.std(axis=0),
    }


def analyze_ric_data(repr_269: np.ndarray, name: str):
    """Analyze the statistics of ric_data (joint positions)"""
    print(f"\n{'='*60}")
    print(f"Analyzing ric_data (joint positions) of {name}")
    print(f"{'='*60}")

    if len(repr_269.shape) == 1:
        repr_269 = repr_269[np.newaxis, :]

    ric_data = repr_269[:, 4:67].reshape(-1, 21, 3)  # (N, 21, 3)

    print(f"Samples: {ric_data.shape[0]}")
    print(f"\nJoint position ranges:")
    print(f"  X: [{ric_data[:,:,0].min():.4f}, {ric_data[:,:,0].max():.4f}]")
    print(f"  Y: [{ric_data[:,:,1].min():.4f}, {ric_data[:,:,1].max():.4f}]")
    print(f"  Z: [{ric_data[:,:,2].min():.4f}, {ric_data[:,:,2].max():.4f}]")

    # Check whether facing +Z.
    # In the normalized frame the chest direction should point to +Z.
    # Joint 9 is spine3 (chest); this is its position relative to the root.
    if ric_data.shape[1] > 8:  # ensure enough joints
        spine3_rel = ric_data[:, 8, :]  # spine3 is joint 9 (index 8, since ric starts at joint 1)
        print(f"\nChest (spine3) relative position:")
        print(f"  mean: {spine3_rel.mean(axis=0)}")
        print(f"  fraction of frames with Z > 0: {(spine3_rel[:, 2] > 0).mean()*100:.1f}%")

    return {
        'x_range': (ric_data[:,:,0].min(), ric_data[:,:,0].max()),
        'y_range': (ric_data[:,:,1].min(), ric_data[:,:,1].max()),
        'z_range': (ric_data[:,:,2].min(), ric_data[:,:,2].max()),
    }


def compare_processing_modes():
    """Compare the differences between the two processing modes"""
    print("\n" + "="*80)
    print("Comparing process_motion_generation and process_mesh_reconstruction")
    print("="*80)

    # Create a test motion (person turning from one facing direction to another)
    T = 30
    positions = create_test_motion(T, rotation_range=np.pi/2)  # 90-degree turn

    print(f"\nInput motion sequence: shape={positions.shape}")
    print(f"  initial facing: some direction")
    print(f"  motion: 90-degree rotation about the Y axis")

    # Mode 1: process_motion_generation (whole sequence processed together)
    print("\n" + "-"*60)
    print("Mode 1: process_motion_generation (motion generation)")
    print("-"*60)
    repr_motion = process_motion_generation(positions)  # (T-1, 269)
    print(f"Output shape: {repr_motion.shape}")

    motion_orient = analyze_global_orient(repr_motion, "motion generation mode")

    # Mode 2: process_mesh_reconstruction (each frame independent)
    print("\n" + "-"*60)
    print("Mode 2: process_mesh_reconstruction (mesh reconstruction, per-frame)")
    print("-"*60)
    repr_mesh = process_mesh_reconstruction(positions)  # (T, 269)
    print(f"Output shape: {repr_mesh.shape}")

    mesh_orient = analyze_global_orient(repr_mesh, "mesh reconstruction mode")

    # Key difference analysis
    print("\n" + "="*80)
    print("Key difference analysis")
    print("="*80)

    print(f"""
1. Difference in the physical meaning of global_orient_6d:

   Motion generation mode (HumanML3D):
   - max rotation angle: {motion_orient['max_angle_deg']:.2f} degrees
   - the whole sequence is rotated once using the first frame's facing
   - global_orient_6d reflects the dynamic facing change (relative to the first frame)

   Mesh reconstruction mode (Human3.6M):
   - max rotation angle: {mesh_orient['max_angle_deg']:.2f} degrees
   - every frame is rotated independently to face +Z
   - global_orient_6d should theoretically be close to the identity rotation

2. Are the semantics unified: {"NO!" if mesh_orient['max_angle_deg'] < motion_orient['max_angle_deg'] / 2 else "there may be differences"}

   - the motion generation mode keeps the facing changes over the motion
   - the mesh reconstruction mode normalizes every frame independently and loses the facing information
    """)

    return repr_motion, repr_mesh


def verify_humanml3d_data(data_path: str = None):
    """Verify actual HumanML3D data"""
    if data_path is None:
        # Default path (relative to the repo root)
        data_path = "data/HumanML3D/HumanML3D/new_joint_vecs/000021.npy"

    if not os.path.exists(data_path):
        print(f"HumanML3D data file not found: {data_path}")
        return None

    print("\n" + "="*80)
    print(f"Verifying HumanML3D data: {data_path}")
    print("="*80)

    data = np.load(data_path)
    print(f"Original data shape: {data.shape}")

    if data.shape[-1] == 263:
        print("Note: this is the original 263-dim HumanML3D format; appending a zero-padded global_orient")
        data_269 = np.concatenate([data, np.zeros((*data.shape[:-1], 6))], axis=-1)
    else:
        data_269 = data

    analyze_global_orient(data_269, "actual HumanML3D data")
    analyze_ric_data(data_269, "actual HumanML3D data")

    return data_269


def verify_human36m_data(data_path: str = None):
    """Verify actual Human3.6M 269-dim data"""
    import joblib

    if data_path is None:
        print("No H36M 269 data path given (--h36m-269); skipping the H36M data check.")
        return None

    if not os.path.exists(data_path):
        print(f"Human3.6M 269-dim data file not found: {data_path}")
        return None

    print("\n" + "="*80)
    print(f"Verifying Human3.6M 269-dim data: {data_path}")
    print("="*80)

    data = joblib.load(data_path)
    repr_269 = data['repr_269'].numpy()

    print(f"Data shape: {repr_269.shape}")

    analyze_global_orient(repr_269, "actual Human3.6M data")
    analyze_ric_data(repr_269, "actual Human3.6M data")

    return repr_269


def main():
    """Main entry"""
    parser = argparse.ArgumentParser(description="Verify 269-dim representation consistency across datasets.")
    parser.add_argument(
        "--target-skeleton", type=str, default="data/HumanML3D/joints/000021.npy",
        help="HumanML3D reference skeleton file (joints/000021.npy)"
    )
    parser.add_argument(
        "--humanml3d-sample", type=str, default=None,
        help="A HumanML3D new_joint_vecs .npy sample to inspect (optional)"
    )
    parser.add_argument(
        "--h36m-269", type=str, default=None,
        help="A converted H36M 269 .pth file to inspect (optional)"
    )
    args = parser.parse_args()

    print("="*80)
    print("269-dim unified representation consistency verification")
    print("="*80)

    # Set the target skeleton path
    if os.path.exists(args.target_skeleton):
        set_target_skeleton_path(args.target_skeleton)
    else:
        print(f"Warning: target skeleton file not found: {args.target_skeleton}")

    # 1. Compare the two processing modes
    repr_motion, repr_mesh = compare_processing_modes()

    # 2. Verify actual data (if present)
    h36m_data = verify_human36m_data(args.h36m_269)
    humanml3d_data = verify_humanml3d_data(args.humanml3d_sample)

    # 3. Summary
    print("\n" + "="*80)
    print("Summary and recommendations")
    print("="*80)

    print("""
Key findings:

1. **The physical meaning of global_orient_6d is not unified**:
   - HumanML3D (motion generation mode): reflects the dynamic facing change relative to the first frame
   - Human3.6M (mesh reconstruction mode): every frame is aligned to +Z independently; facing changes are flattened

2. **rot_data and ric_data**:
   - both modes apply skeleton normalization and foot-on-floor
   - but the reference frames differ: motion generation uses the first frame's facing, mesh reconstruction is per-frame

3. **Coordinate systems**:
   - both are unified to head-to-+Y (Human3.6M via flip_coords)
   - but the facing-+Z handling differs

Recommendations:

1. If the goal is training a unified motion generation model:
   - Human3.6M could be processed as continuous sequences with process_motion_generation
   - or, under the current design, accept that global_orient_6d of single-frame data has different semantics

2. If the goal is training a unified mesh reconstruction model:
   - ric_data and rot_data are consistent at the single-frame level between the two sources
   - global_orient_6d in mesh reconstruction mode should be read as "current frame's rotation relative to the canonical facing"

3. Be explicit about the design intent:
   - mesh reconstruction mode is per-frame by design and needs no temporal coherence
   - motion generation mode needs temporal coherence; global_orient_6d records the facing evolution

Note: in the released training data, H36M is converted with ROTATION_MODE=full,
which overrides rot_data + global_orient with the real SMPL rotations and thus
partially mitigates the global_orient inconsistency described above. See
docs/269_consistency_analysis.md.
    """)


if __name__ == '__main__':
    main()
