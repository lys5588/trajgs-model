"""
Unified Human Motion Representation (269-dim)
=============================================

Paper: Unified 269-dim motion representation (Appendix D.1).

This module implements the unified human body representation with two modes:
- Mode A: motion generation (HumanML3D-compatible) - outputs (T-1, 269)
- Mode B: mesh reconstruction (single frame, UniPose-compatible) - outputs (N, 269)

269-dim layout:
| index | dims | content | motion generation | mesh reconstruction |
|-------|------|---------|-------------------|---------------------|
| 0 | 1 | root_rot_velocity | actual value | 0 |
| 1-2 | 2 | root_linear_velocity | actual value | 0 |
| 3 | 1 | root_y | yes | yes |
| 4-66 | 63 | ric_data (21x3) | yes | yes |
| 67-192 | 126 | rot_data (21x6) | yes | yes |
| 193-258 | 66 | local_vel (22x3) | actual value | 0 |
| 259-262 | 4 | foot_contact | 0/1 | 0.5 (unknown) |
| 263-268 | 6 | global_orient_6d | dynamic | yes |

The first 263 dims are bit-exact with the official HumanML3D representation;
the last 6 dims append the root joint rotation in continuous 6D form.
"""

import os
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from typing import Optional, Union, Tuple, Dict, Any


# ============================================================================
# Constants
# ============================================================================

# HumanML3D skeleton parameters (22 joints)
T2M_RAW_OFFSETS = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, -1, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, -1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [-1, 0, 0],
    [0, 0, 1],
    [0, -1, 0],
    [0, -1, 0],
    [0, -1, 0],
    [0, -1, 0],
    [0, -1, 0],
    [0, -1, 0]
])

# Kinematic chains
T2M_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],   # left leg
    [0, 1, 4, 7, 10],   # right leg
    [0, 3, 6, 9, 12, 15],  # spine and head
    [9, 14, 17, 19, 21],   # right arm
    [9, 13, 16, 18, 20]    # left arm
]

# Joint indices used for facing-direction detection: r_hip, l_hip, sdr_r, sdr_l
FACE_JOINT_INDX = [2, 1, 17, 16]

# Leg indices (used for skeleton scaling)
L_IDX1, L_IDX2 = 5, 8

# Foot joint indices
FID_R, FID_L = [8, 11], [7, 10]

# Number of joints
JOINTS_NUM = 22


# ============================================================================
# Quaternion and rotation utility functions
# ============================================================================

def qinv(q: torch.Tensor) -> torch.Tensor:
    """Quaternion inverse"""
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def qinv_np(q: np.ndarray) -> np.ndarray:
    """Quaternion inverse (numpy version)"""
    return qinv(torch.from_numpy(q).float()).numpy()


def qnormalize(q: torch.Tensor) -> torch.Tensor:
    """Quaternion normalization"""
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    return q / torch.norm(q, dim=-1, keepdim=True)


def qmul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Quaternion multiplication"""
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def qmul_np(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Quaternion multiplication (numpy version)"""
    q = torch.from_numpy(q).contiguous().float()
    r = torch.from_numpy(r).contiguous().float()
    return qmul(q, r).numpy()


def qrot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by a quaternion"""
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def qrot_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by a quaternion (numpy version)"""
    q = torch.from_numpy(q).contiguous().float()
    v = torch.from_numpy(v).contiguous().float()
    return qrot(q, v).numpy()


def qbetween(v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    """Find the quaternion rotating v0 to v1"""
    assert v0.shape[-1] == 3, 'v0 must be of the shape (*, 3)'
    assert v1.shape[-1] == 3, 'v1 must be of the shape (*, 3)'

    v = torch.cross(v0, v1)
    w = torch.sqrt((v0 ** 2).sum(dim=-1, keepdim=True) * (v1 ** 2).sum(dim=-1, keepdim=True)) + \
        (v0 * v1).sum(dim=-1, keepdim=True)
    return qnormalize(torch.cat([w, v], dim=-1))


def qbetween_np(v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    """Find the quaternion rotating v0 to v1 (numpy version)"""
    v0 = torch.from_numpy(v0).float()
    v1 = torch.from_numpy(v1).float()
    return qbetween(v0, v1).numpy()


def qfix(q: np.ndarray) -> np.ndarray:
    """
    Fix quaternion discontinuities by choosing the representation that minimizes
    the distance between consecutive frames.
    Input: (L, J, 4)
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4

    result = q.copy()
    dot_products = np.sum(q[1:] * q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Quaternion to rotation matrix"""
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def quaternion_to_matrix_np(quaternions: np.ndarray) -> np.ndarray:
    """Quaternion to rotation matrix (numpy version)"""
    q = torch.from_numpy(quaternions).contiguous().float()
    return quaternion_to_matrix(q).numpy()


def quaternion_to_cont6d(quaternions: torch.Tensor) -> torch.Tensor:
    """Quaternion to continuous 6D rotation representation"""
    rotation_mat = quaternion_to_matrix(quaternions)
    cont_6d = torch.cat([rotation_mat[..., 0], rotation_mat[..., 1]], dim=-1)
    return cont_6d


def quaternion_to_cont6d_np(quaternions: np.ndarray) -> np.ndarray:
    """Quaternion to continuous 6D rotation representation (numpy version)"""
    rotation_mat = quaternion_to_matrix_np(quaternions)
    cont_6d = np.concatenate([rotation_mat[..., 0], rotation_mat[..., 1]], axis=-1)
    return cont_6d


def cont6d_to_matrix(cont6d: torch.Tensor) -> torch.Tensor:
    """Continuous 6D rotation representation to rotation matrix"""
    assert cont6d.shape[-1] == 6, "The last dimension must be 6"
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]

    x = x_raw / torch.norm(x_raw, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, dim=-1, keepdim=True)
    y = torch.cross(z, x, dim=-1)

    x = x[..., None]
    y = y[..., None]
    z = z[..., None]

    mat = torch.cat([x, y, z], dim=-1)
    return mat


def cont6d_to_matrix_np(cont6d: np.ndarray) -> np.ndarray:
    """Continuous 6D rotation representation to rotation matrix (numpy version)"""
    q = torch.from_numpy(cont6d).contiguous().float()
    return cont6d_to_matrix(q).numpy()


def matrix_to_cont6d(matrix: torch.Tensor) -> torch.Tensor:
    """Rotation matrix to continuous 6D rotation representation"""
    return torch.cat([matrix[..., 0], matrix[..., 1]], dim=-1)


def matrix_to_cont6d_np(matrix: np.ndarray) -> np.ndarray:
    """Rotation matrix to continuous 6D rotation representation (numpy version)"""
    return np.concatenate([matrix[..., 0], matrix[..., 1]], axis=-1)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Rotation matrix to quaternion (uses Shepperd's method for numerical stability)
    """
    R = matrix
    w2 = (1 + R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2])
    x2 = (1 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2])
    y2 = (1 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2])
    z2 = (1 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2])

    yz = (R[..., 1, 2] + R[..., 2, 1])
    xz = (R[..., 2, 0] + R[..., 0, 2])
    xy = (R[..., 0, 1] + R[..., 1, 0])

    wx = (R[..., 2, 1] - R[..., 1, 2])
    wy = (R[..., 0, 2] - R[..., 2, 0])
    wz = (R[..., 1, 0] - R[..., 0, 1])

    w = torch.empty_like(x2)
    x = torch.empty_like(x2)
    y = torch.empty_like(x2)
    z = torch.empty_like(x2)

    flagA = (R[..., 2, 2] < 0) * (R[..., 0, 0] > R[..., 1, 1])
    flagB = (R[..., 2, 2] < 0) * (R[..., 0, 0] <= R[..., 1, 1])
    flagC = (R[..., 2, 2] >= 0) * (R[..., 0, 0] < -R[..., 1, 1])
    flagD = (R[..., 2, 2] >= 0) * (R[..., 0, 0] >= -R[..., 1, 1])

    x[flagA] = torch.sqrt(x2[flagA])
    w[flagA] = wx[flagA] / x[flagA]
    y[flagA] = xy[flagA] / x[flagA]
    z[flagA] = xz[flagA] / x[flagA]

    y[flagB] = torch.sqrt(y2[flagB])
    w[flagB] = wy[flagB] / y[flagB]
    x[flagB] = xy[flagB] / y[flagB]
    z[flagB] = yz[flagB] / y[flagB]

    z[flagC] = torch.sqrt(z2[flagC])
    w[flagC] = wz[flagC] / z[flagC]
    x[flagC] = xz[flagC] / z[flagC]
    y[flagC] = yz[flagC] / z[flagC]

    w[flagD] = torch.sqrt(w2[flagD])
    x[flagD] = wx[flagD] / w[flagD]
    y[flagD] = wy[flagD] / w[flagD]
    z[flagD] = wz[flagD] / w[flagD]

    res = [w, x, y, z]
    res = [elem.unsqueeze(-1) for elem in res]

    return torch.cat(res, dim=-1) / 2


def matrix_to_quaternion_np(matrix: np.ndarray) -> np.ndarray:
    """Rotation matrix to quaternion (numpy version)"""
    return matrix_to_quaternion(torch.from_numpy(matrix).float()).numpy()


def cont6d_to_quaternion(cont6d: torch.Tensor) -> torch.Tensor:
    """Continuous 6D rotation representation to quaternion"""
    return matrix_to_quaternion(cont6d_to_matrix(cont6d))


def cont6d_to_quaternion_np(cont6d: np.ndarray) -> np.ndarray:
    """Continuous 6D rotation representation to quaternion (numpy version)"""
    return cont6d_to_quaternion(torch.from_numpy(cont6d).float()).numpy()


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Rotation matrix to axis-angle representation
    (implementation follows pytorch3d)
    """
    quat = matrix_to_quaternion(matrix)
    # Ensure w is positive
    quat = torch.where(quat[..., 0:1] < 0, -quat, quat)

    # Extract axis and angle
    w = quat[..., 0:1]
    xyz = quat[..., 1:]

    # sin and cos of the half angle
    sin_half_angle = torch.norm(xyz, dim=-1, keepdim=True)
    cos_half_angle = w

    # Compute the angle
    two_theta = 2.0 * torch.atan2(sin_half_angle, cos_half_angle)

    # Handle small angles
    near_zero = sin_half_angle.abs() < 1e-6

    # Axis-angle representation
    axis_angle = torch.where(
        near_zero,
        2.0 * xyz,
        xyz / sin_half_angle * two_theta
    )

    return axis_angle


def matrix_to_axis_angle_np(matrix: np.ndarray) -> np.ndarray:
    """Rotation matrix to axis-angle representation (numpy version)"""
    return matrix_to_axis_angle(torch.from_numpy(matrix).float()).numpy()


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Axis-angle representation to rotation matrix (Rodrigues' formula)
    """
    theta = torch.norm(axis_angle, dim=-1, keepdim=True)
    near_zero = theta.abs() < 1e-8

    # Normalize the axis
    axis = torch.where(near_zero, axis_angle, axis_angle / theta)

    # Rodrigues' formula
    cos_theta = torch.cos(theta)[..., 0]
    sin_theta = torch.sin(theta)[..., 0]

    # Cross-product matrix
    K = torch.zeros(axis.shape[:-1] + (3, 3), dtype=axis.dtype, device=axis.device)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]

    # R = I + sin(theta)*K + (1-cos(theta))*K^2
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device)
    R = eye + sin_theta[..., None, None] * K + \
        (1 - cos_theta)[..., None, None] * torch.matmul(K, K)

    return R


def axis_angle_to_matrix_np(axis_angle: np.ndarray) -> np.ndarray:
    """Axis-angle representation to rotation matrix (numpy version)"""
    return axis_angle_to_matrix(torch.from_numpy(axis_angle).float()).numpy()


# ============================================================================
# Skeleton class
# ============================================================================

class Skeleton:
    """Skeleton class for forward/inverse kinematics computation"""

    def __init__(self, offset: torch.Tensor, kinematic_tree: list, device: str):
        self.device = device
        self._raw_offset_np = offset.numpy()
        self._raw_offset = offset.clone().detach().to(device).float()
        self._kinematic_tree = kinematic_tree
        self._offset = None
        self._parents = [0] * len(self._raw_offset)
        self._parents[0] = -1
        for chain in self._kinematic_tree:
            for j in range(1, len(chain)):
                self._parents[chain[j]] = chain[j-1]

    def njoints(self) -> int:
        return len(self._raw_offset)

    def offset(self) -> torch.Tensor:
        return self._offset

    def set_offset(self, offsets: torch.Tensor):
        self._offset = offsets.clone().detach().to(self.device).float()

    def kinematic_tree(self) -> list:
        return self._kinematic_tree

    def parents(self) -> list:
        return self._parents

    def get_offsets_joints_batch(self, joints: torch.Tensor) -> torch.Tensor:
        """Compute skeleton offsets in batch"""
        assert len(joints.shape) == 3
        _offsets = self._raw_offset.expand(joints.shape[0], -1, -1).clone()
        for i in range(1, self._raw_offset.shape[0]):
            _offsets[:, i] = torch.norm(joints[:, i] - joints[:, self._parents[i]], p=2, dim=1)[:, None] * _offsets[:, i]

        self._offset = _offsets.detach()
        return _offsets

    def get_offsets_joints(self, joints: torch.Tensor) -> torch.Tensor:
        """Compute skeleton offsets"""
        assert len(joints.shape) == 2
        _offsets = self._raw_offset.clone()
        for i in range(1, self._raw_offset.shape[0]):
            _offsets[i] = torch.norm(joints[i] - joints[self._parents[i]], p=2, dim=0) * _offsets[i]

        self._offset = _offsets.detach()
        return _offsets

    def inverse_kinematics_np(self, joints: np.ndarray, face_joint_idx: list,
                               smooth_forward: bool = False) -> np.ndarray:
        """
        Inverse kinematics: compute quaternion parameters from joint positions

        Args:
            joints: (batch_size, joints_num, 3) joint positions
            face_joint_idx: [r_hip, l_hip, sdr_r, sdr_l] joints used for facing detection
            smooth_forward: whether to smooth the forward direction

        Returns:
            quat_params: (batch_size, joints_num, 4) quaternion parameters
        """
        assert len(face_joint_idx) == 4
        l_hip, r_hip, sdr_r, sdr_l = face_joint_idx

        # Compute the forward direction
        across1 = joints[:, r_hip] - joints[:, l_hip]
        across2 = joints[:, sdr_r] - joints[:, sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across**2).sum(axis=-1))[:, np.newaxis]

        # forward (batch_size, 3)
        forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        if smooth_forward:
            forward = gaussian_filter1d(forward, 20, axis=0, mode='nearest')
        forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]

        # Compute the root rotation
        target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
        root_quat = qbetween_np(forward, target)

        # Inverse kinematics
        quat_params = np.zeros(joints.shape[:-1] + (4,))
        root_quat[0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        quat_params[:, 0] = root_quat

        for chain in self._kinematic_tree:
            R = root_quat
            for j in range(len(chain) - 1):
                u = self._raw_offset_np[chain[j+1]][np.newaxis, ...].repeat(len(joints), axis=0)
                v = joints[:, chain[j+1]] - joints[:, chain[j]]
                v = v / np.sqrt((v**2).sum(axis=-1))[:, np.newaxis]
                rot_u_v = qbetween_np(u, v)
                R_loc = qmul_np(qinv_np(R), rot_u_v)
                quat_params[:, chain[j + 1], :] = R_loc
                R = qmul_np(R, R_loc)

        return quat_params

    def forward_kinematics_np(self, quat_params: np.ndarray, root_pos: np.ndarray,
                               skel_joints: np.ndarray = None, do_root_R: bool = True) -> np.ndarray:
        """Forward kinematics: compute joint positions from quaternion parameters (numpy version)"""
        if skel_joints is not None:
            skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(quat_params.shape[0], -1, -1)
        offsets = offsets.numpy()

        joints = np.zeros(quat_params.shape[:-1] + (3,))
        joints[:, 0] = root_pos

        for chain in self._kinematic_tree:
            if do_root_R:
                R = quat_params[:, 0]
            else:
                R = np.array([[1.0, 0.0, 0.0, 0.0]]).repeat(len(quat_params), axis=0)
            for i in range(1, len(chain)):
                R = qmul_np(R, quat_params[:, chain[i]])
                offset_vec = offsets[:, chain[i]]
                joints[:, chain[i]] = qrot_np(R, offset_vec) + joints[:, chain[i - 1]]
        return joints

    def forward_kinematics_cont6d_np(self, cont6d_params: np.ndarray, root_pos: np.ndarray,
                                      skel_joints: np.ndarray = None, do_root_R: bool = True) -> np.ndarray:
        """Forward kinematics: compute joint positions from 6D rotation parameters (numpy version)"""
        if skel_joints is not None:
            skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(cont6d_params.shape[0], -1, -1)
        offsets = offsets.numpy()

        joints = np.zeros(cont6d_params.shape[:-1] + (3,))
        joints[:, 0] = root_pos

        for chain in self._kinematic_tree:
            if do_root_R:
                matR = cont6d_to_matrix_np(cont6d_params[:, 0])
            else:
                matR = np.eye(3)[np.newaxis, :].repeat(len(cont6d_params), axis=0)
            for i in range(1, len(chain)):
                matR = np.matmul(matR, cont6d_to_matrix_np(cont6d_params[:, chain[i]]))
                offset_vec = offsets[:, chain[i]][..., np.newaxis]
                joints[:, chain[i]] = np.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i-1]]
        return joints

    def forward_kinematics_cont6d(self, cont6d_params: torch.Tensor, root_pos: torch.Tensor,
                                   skel_joints: torch.Tensor = None, do_root_R: bool = True) -> torch.Tensor:
        """Forward kinematics: compute joint positions from 6D rotation parameters"""
        if skel_joints is not None:
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(cont6d_params.shape[0], -1, -1)

        joints = torch.zeros(cont6d_params.shape[:-1] + (3,)).to(cont6d_params.device)
        joints[..., 0, :] = root_pos

        for chain in self._kinematic_tree:
            if do_root_R:
                matR = cont6d_to_matrix(cont6d_params[:, 0])
            else:
                matR = torch.eye(3).expand((len(cont6d_params), -1, -1)).detach().to(cont6d_params.device)
            for i in range(1, len(chain)):
                matR = torch.matmul(matR, cont6d_to_matrix(cont6d_params[:, chain[i]]))
                offset_vec = offsets[:, chain[i]].unsqueeze(-1)
                joints[:, chain[i]] = torch.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i-1]]
        return joints


# ============================================================================
# Global skeleton object and target offsets (lazy initialization)
# ============================================================================

_n_raw_offsets = torch.from_numpy(T2M_RAW_OFFSETS).float()
_kinematic_chain = T2M_KINEMATIC_CHAIN

# Reference sample id and path used by the official HumanML3D pipeline.
# Consistent with `t2m_tgt_skel_id = '000021'` in HumanML3D/paramUtil.py.
# The default path is relative to the repository root; override it with
# `set_target_skeleton_path()` if your HumanML3D data lives elsewhere.
T2M_TGT_SKEL_ID = '000021'
T2M_TGT_SKEL_PATH = 'data/HumanML3D/joints/000021.npy'

# Cached target offsets (lazy initialization; computed from file on first use)
_cached_target_offsets = None


def _compute_target_offsets_from_file(filepath: str) -> torch.Tensor:
    """
    Compute target skeleton offsets from the reference sample file.
    Identical to the official HumanML3D processing.

    Args:
        filepath: path to the reference sample (e.g. 000021.npy)

    Returns:
        target_offsets: (22, 3) target skeleton offsets
    """
    # Load the reference sample
    example_data = np.load(filepath)
    example_data = example_data.reshape(len(example_data), -1, 3)
    example_data = torch.from_numpy(example_data).float()

    # Compute target offsets (identical to HumanML3D/motion_representation.py)
    tgt_skel = Skeleton(_n_raw_offsets, _kinematic_chain, 'cpu')
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])

    return tgt_offsets


def _get_skeleton_and_offsets(example_joints: np.ndarray = None) -> Tuple[Skeleton, torch.Tensor]:
    """
    Get the skeleton object and target offsets.
    Consistent with the official HumanML3D processing: target offsets are
    computed from 000021.npy.

    Args:
        example_joints: optional example joint positions (22, 3) used to compute
                        target offsets. If not provided, the reference sample
                        loaded from T2M_TGT_SKEL_PATH is used.

    Returns:
        skeleton: Skeleton object
        target_offsets: target skeleton offsets
    """
    global _cached_target_offsets

    skeleton = Skeleton(_n_raw_offsets, _kinematic_chain, "cpu")

    if example_joints is not None:
        # Use custom example joints
        target_offsets = skeleton.get_offsets_joints(torch.from_numpy(example_joints).float())
    else:
        # Use the cache or load from file (consistent with official HumanML3D)
        if _cached_target_offsets is None:
            if os.path.exists(T2M_TGT_SKEL_PATH):
                _cached_target_offsets = _compute_target_offsets_from_file(T2M_TGT_SKEL_PATH)
                print(f"[unified_representation_269] Loaded target offsets from: {T2M_TGT_SKEL_PATH}")
            else:
                raise FileNotFoundError(
                    f"HumanML3D reference skeleton file not found: {T2M_TGT_SKEL_PATH}\n"
                    f"This should be the official HumanML3D 'joints/000021.npy' file. "
                    f"Make sure it exists at the path above (relative to the repo root), "
                    f"or call set_target_skeleton_path() to point to the correct location."
                )
        target_offsets = _cached_target_offsets
        skeleton.set_offset(target_offsets)

    return skeleton, target_offsets


def set_target_skeleton_path(filepath: str) -> None:
    """
    Set the target skeleton reference sample path.

    Args:
        filepath: reference sample path (should be HumanML3D's joints/000021.npy)
    """
    global T2M_TGT_SKEL_PATH, _cached_target_offsets
    T2M_TGT_SKEL_PATH = filepath
    _cached_target_offsets = None  # clear the cache; reloaded on next use
    print(f"[unified_representation_269] Target skeleton path set to: {filepath}")


def get_target_offsets() -> torch.Tensor:
    """
    Get the current target skeleton offsets.

    Returns:
        target_offsets: (22, 3) target skeleton offsets
    """
    _, target_offsets = _get_skeleton_and_offsets()
    return target_offsets


# ============================================================================
# Core processing functions
# ============================================================================

def uniform_skeleton(positions: np.ndarray, target_offset: torch.Tensor) -> np.ndarray:
    """
    Retarget the skeleton to the standard (target) skeleton.

    Args:
        positions: (T, 22, 3) joint position sequence
        target_offset: target skeleton offsets

    Returns:
        new_joints: (T, 22, 3) retargeted joint positions
    """
    src_skel = Skeleton(_n_raw_offsets, _kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()

    # Compute the leg-length ratio
    src_leg_len = np.abs(src_offset[L_IDX1]).max() + np.abs(src_offset[L_IDX2]).max()
    tgt_leg_len = np.abs(tgt_offset[L_IDX1]).max() + np.abs(tgt_offset[L_IDX2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    # Inverse kinematics
    quat_params = src_skel.inverse_kinematics_np(positions, FACE_JOINT_INDX)

    # Forward kinematics
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints


def foot_detect(positions: np.ndarray, thres: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect foot contacts.

    Args:
        positions: (T, 22, 3) joint positions
        thres: velocity threshold

    Returns:
        feet_l: (T-1, 2) left-foot contact
        feet_r: (T-1, 2) right-foot contact
    """
    velfactor = np.array([thres, thres])

    feet_l_x = (positions[1:, FID_L, 0] - positions[:-1, FID_L, 0]) ** 2
    feet_l_y = (positions[1:, FID_L, 1] - positions[:-1, FID_L, 1]) ** 2
    feet_l_z = (positions[1:, FID_L, 2] - positions[:-1, FID_L, 2]) ** 2
    feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

    feet_r_x = (positions[1:, FID_R, 0] - positions[:-1, FID_R, 0]) ** 2
    feet_r_y = (positions[1:, FID_R, 1] - positions[:-1, FID_R, 1]) ** 2
    feet_r_z = (positions[1:, FID_R, 2] - positions[:-1, FID_R, 2]) ** 2
    feet_r = ((feet_r_x + feet_r_y + feet_r_z) < velfactor).astype(np.float32)

    return feet_l, feet_r


def process_motion_generation(positions: np.ndarray,
                               feet_thre: float = 0.002,
                               target_offset: torch.Tensor = None) -> np.ndarray:
    """
    Motion generation mode: fully HumanML3D-compatible + global_orient_6d.

    This function performs the full HumanML3D preprocessing pipeline and appends
    the 6-dim global_orient_6d at the end.

    Args:
        positions: (T, 22, 3) raw joint position sequence
        feet_thre: foot contact detection threshold
        target_offset: optional target skeleton offsets. If not provided, the
                       default is used.

    Returns:
        data: (T-1, 269) unified representation
            - the first 263 dims are identical to the original HumanML3D output
            - the last 6 dims are global_orient_6d (root joint rotation in 6D)
    """
    # Get the skeleton and target offsets
    if target_offset is None:
        _, target_offset = _get_skeleton_and_offsets()

    # ====== Original HumanML3D preprocessing (bit-exact) ======

    # 1. Uniform skeleton
    positions = uniform_skeleton(positions, target_offset)

    # 2. Put on floor
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    # 3. XZ-plane origin alignment
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # 4. Rotate so the initial facing direction is Z+
    r_hip, l_hip, sdr_r, sdr_l = FACE_JOINT_INDX
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions = qrot_np(root_quat_init, positions)

    # Keep the global positions for velocity computation
    global_positions = positions.copy()

    # ====== Foot contact detection ======
    feet_l, feet_r = foot_detect(positions, feet_thre)

    # ====== Compute 6D rotation parameters ======
    skel = Skeleton(_n_raw_offsets, _kinematic_chain, "cpu")
    quat_params = skel.inverse_kinematics_np(positions, FACE_JOINT_INDX, smooth_forward=True)

    # Quaternion to 6D rotation
    cont_6d_params = quaternion_to_cont6d_np(quat_params)  # (T, 22, 6)

    # Keep the root rotation quaternion (used later)
    r_rot = quat_params[:, 0].copy()  # (T, 4)

    # Root linear velocity (in the local frame)
    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()  # (T-1, 3)
    velocity = qrot_np(r_rot[1:], velocity)  # rotate into the local frame

    # Root angular velocity
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))  # (T-1, 4)

    # ====== Compute RIFKE (Root-Invariant Forward Kinematics) positions ======
    # Subtract the root XZ position
    positions[..., 0] -= positions[:, 0:1, 0]
    positions[..., 2] -= positions[:, 0:1, 2]
    # Rotate into the local frame
    positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)

    # ====== Extract features ======

    # Root height
    root_y = positions[:, 0, 1:2]  # (T, 1)

    # Root rotation velocity (Y-axis component only)
    r_velocity_y = np.arcsin(r_velocity[:, 2:3])  # (T-1, 1)

    # Root linear velocity (XZ plane)
    l_velocity = velocity[:, [0, 2]]  # (T-1, 2)

    # Root features
    root_data = np.concatenate([r_velocity_y, l_velocity, root_y[:-1]], axis=-1)  # (T-1, 4)

    # Joint rotations (excluding the root)
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)  # (T, 126)

    # Joint positions (excluding the root, in the local frame)
    ric_data = positions[:, 1:].reshape(len(positions), -1)  # (T, 63)

    # Local joint velocities
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)  # (T-1, 66)

    # Root joint rotation in 6D representation
    global_orient_6d = cont_6d_params[:, 0, :]  # (T, 6)

    # ====== Concatenate 269 dims (T-1 frames) ======
    data = np.concatenate([
        root_data,                     # (T-1, 4): r_velocity, l_velocity, root_y
        ric_data[:-1],                 # (T-1, 63)
        rot_data[:-1],                 # (T-1, 126)
        local_vel,                     # (T-1, 66)
        feet_l, feet_r,                # (T-1, 4)
        global_orient_6d[:-1],         # (T-1, 6) <- appended
    ], axis=-1)

    return data  # (T-1, 269)


def process_mesh_reconstruction(positions: np.ndarray,
                                  target_offset: torch.Tensor = None) -> np.ndarray:
    """
    Mesh reconstruction mode: per-frame processing, information content
    consistent with UniPose.

    Each frame is processed independently; velocity dims are set to 0 and foot
    contact is set to 0.5 (unknown).

    Args:
        positions: (22, 3) single-frame joint positions
                   or (N, 22, 3) multiple frames (each processed independently)
        target_offset: optional target skeleton offsets

    Returns:
        data: (1, 269) or (N, 269)
    """
    # Handle input shape
    single_frame = len(positions.shape) == 2
    if single_frame:
        positions = positions[np.newaxis, ...]  # (1, 22, 3)

    N = len(positions)

    # Get the skeleton and target offsets
    if target_offset is None:
        _, target_offset = _get_skeleton_and_offsets()

    data_list = []
    for i in range(N):
        pos = positions[i:i+1].copy()  # (1, 22, 3)
        # ====== Preprocessing (same as motion generation) ======
        # 1. Uniform skeleton
        pos = uniform_skeleton(pos, target_offset)
        # 2. Put on floor
        floor_height = pos.min(axis=0).min(axis=0)[1]
        pos[:, :, 1] -= floor_height

        # 3. XZ-plane origin alignment
        root_pos_init = pos[0]
        root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
        pos = pos - root_pose_init_xz

        # 4. Rotate so the initial facing direction is Z+
        r_hip, l_hip, sdr_r, sdr_l = FACE_JOINT_INDX
        across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
        across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

        forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

        root_quat_init = qbetween_np(forward_init, np.array([[0, 0, 1]]))
        # Expand root_quat_init to match the shape of pos (1, 22, 4)
        root_quat_init_expanded = np.ones((1, 22, 4)) * root_quat_init  # (1, 22, 4)
        pos = qrot_np(root_quat_init_expanded, pos)

        # ====== Compute 6D rotations ======
        skel = Skeleton(_n_raw_offsets, _kinematic_chain, "cpu")
        quat_params = skel.inverse_kinematics_np(pos, FACE_JOINT_INDX, smooth_forward=False)
        cont_6d = quaternion_to_cont6d_np(quat_params)  # (1, 22, 6)

        global_orient_6d = cont_6d[0, 0, :]  # (6,)
        rot_data = cont_6d[0, 1:].reshape(-1)  # (126,)

        # ====== Compute position features (consistent with get_rifke) ======
        r_rot = quat_params[0, 0]  # (4,) root quaternion

        # Subtract the root XZ position
        pos_local = pos.copy()
        pos_local[..., 0] -= pos[:, 0:1, 0]
        pos_local[..., 2] -= pos[:, 0:1, 2]

        # Rotate into the local frame (crucial! consistent with get_rifke)
        pos_local = qrot_np(np.repeat(r_rot[np.newaxis, np.newaxis, :], 22, axis=1), pos_local)

        root_y = pos_local[0, 0, 1]  # scalar
        ric_data = pos_local[0, 1:].reshape(-1)  # (63,)

        # ====== Concatenate 269 dims (velocities 0, foot contact 0.5) ======
        frame_data = np.concatenate([
            np.zeros(1),            # root_rot_velocity
            np.zeros(2),            # root_linear_velocity
            np.array([root_y]),     # root_y
            ric_data,               # (63,) - local joint positions
            rot_data,               # (126,) - joint rotations
            np.zeros(66),           # local_vel - velocities are 0
            np.full(4, 0.5),        # foot_contact - 0.5 means unknown
            global_orient_6d,       # (6,) - root rotation
        ])  # (269,)

        data_list.append(frame_data)

    data = np.stack(data_list, axis=0)  # (N, 269)

    if single_frame:
        return data  # (1, 269)
    return data  # (N, 269)


def process_from_smpl(pose_rotmat: np.ndarray,
                       betas: np.ndarray = None,
                       body_model: Any = None,
                       global_orient_rotmat: np.ndarray = None) -> np.ndarray:
    """
    Generate the 269-dim representation directly from SMPL rotation matrices
    (mesh reconstruction mode).

    Args:
        pose_rotmat: (21, 3, 3) or (N, 21, 3, 3) body_pose joint rotation matrices,
                     or (22, 3, 3) or (N, 22, 3, 3) full rotation matrices
                     including global_orient
        betas: (10,) shape parameters (optional, used to generate joint positions)
        body_model: SMPL/SMPLH body model (optional, used to generate joint positions)
        global_orient_rotmat: (3, 3) or (N, 3, 3) global orientation rotation matrices.
                              Required if pose_rotmat has only 21 joints.

    Returns:
        data: (1, 269) or (N, 269)

    Note:
        If body_model and betas are provided, they are used to generate joint
        positions, and process_mesh_reconstruction is called.
        Otherwise the representation is built directly from the rotation matrices
        (ric_data uses default standard-body values).
    """
    # Handle input shape
    single_frame = len(pose_rotmat.shape) == 3
    if single_frame:
        pose_rotmat = pose_rotmat[np.newaxis, ...]  # (1, J, 3, 3)
        if global_orient_rotmat is not None:
            global_orient_rotmat = global_orient_rotmat[np.newaxis, ...]  # (1, 3, 3)

    N = len(pose_rotmat)

    # Handle rotation matrices
    if pose_rotmat.shape[1] == 21:
        # body_pose only; global_orient is required
        if global_orient_rotmat is None:
            raise ValueError("pose_rotmat has only 21 joints; global_orient_rotmat must be provided")
        full_rotmat = np.concatenate([
            global_orient_rotmat[:, np.newaxis, :, :],  # (N, 1, 3, 3)
            pose_rotmat  # (N, 21, 3, 3)
        ], axis=1)  # (N, 22, 3, 3)
    elif pose_rotmat.shape[1] == 22:
        full_rotmat = pose_rotmat
    else:
        raise ValueError(f"pose_rotmat should have 21 or 22 joints, got {pose_rotmat.shape[1]}")

    # If a body_model is available, generate joint positions and call the
    # standard processing function
    if body_model is not None:
        # Requires pytorch3d- or smplx-style rotation conversion
        pose_rotmat_torch = torch.from_numpy(full_rotmat).float()

        # Convert to axis-angle
        pose_aa = matrix_to_axis_angle(pose_rotmat_torch)  # (N, 22, 3)

        if betas is None:
            betas = np.zeros(10)
        betas_torch = torch.from_numpy(betas).float().unsqueeze(0).expand(N, -1)

        # Use the body_model to generate joint positions
        output = body_model(
            global_orient=pose_aa[:, 0],
            body_pose=pose_aa[:, 1:].flatten(1),
            betas=betas_torch,
        )
        positions = output.joints[:, :22].numpy()  # (N, 22, 3)

        return process_mesh_reconstruction(positions)

    # Without a body_model, use ric_data of the default standard body
    # First generate ric_data of a standard pose as a fallback
    _, target_offset = _get_skeleton_and_offsets()

    # Use the T-pose as the default position
    skel = Skeleton(_n_raw_offsets, _kinematic_chain, "cpu")
    skel.set_offset(target_offset)

    # T-pose quaternion parameters (identity quaternion for all joints)
    default_quat = np.zeros((1, 22, 4))
    default_quat[..., 0] = 1.0  # identity quaternion
    default_root_pos = np.zeros((1, 3))
    default_root_pos[:, 1] = 0.9  # approximate root height

    default_positions = skel.forward_kinematics_np(default_quat, default_root_pos)

    # Preprocess the default positions to get default ric_data
    default_pos = default_positions[0]  # (22, 3)
    default_ric = default_pos[1:].reshape(-1)  # (63,)
    default_root_y = default_pos[0, 1]

    # Convert to 6D representation
    pose_6d = matrix_to_cont6d_np(full_rotmat)  # (N, 22, 6)

    data_list = []
    for i in range(N):
        global_orient_6d = pose_6d[i, 0]  # (6,)
        rot_data = pose_6d[i, 1:].reshape(-1)  # (126,)

        frame_data = np.concatenate([
            np.zeros(1),            # root_rot_velocity
            np.zeros(2),            # root_linear_velocity
            np.array([default_root_y]),  # root_y (default value)
            default_ric,            # (63,) ric_data (default values)
            rot_data,               # (126,)
            np.zeros(66),           # local_vel
            np.full(4, 0.5),        # foot_contact (0.5 means unknown)
            global_orient_6d,       # (6,)
        ])
        data_list.append(frame_data)

    data = np.stack(data_list, axis=0)
    return data[0:1] if single_frame else data


# ============================================================================
# Recovery functions
# ============================================================================

def recover_root_rot_pos(data: Union[np.ndarray, torch.Tensor]) -> Tuple:
    """
    Recover the root rotation and position from the 263/269-dim representation.

    Args:
        data: (..., 263) or (..., 269) motion representation

    Returns:
        r_rot_quat: (..., 4) root rotation quaternion
        r_pos: (..., 3) root position
    """
    is_numpy = isinstance(data, np.ndarray)
    if is_numpy:
        data = torch.from_numpy(data).float()

    # Check whether global_orient_6d should be used (269-dim feature)
    use_global_orient = False
    if data.shape[-1] == 269:
        # Check for single-frame mode (velocities close to 0).
        # This check matches Repr269.is_single_frame.
        root_vel = data[..., 0:3]
        local_vel = data[..., 193:259]
        if torch.all(torch.abs(root_vel) < 1e-5) and torch.all(torch.abs(local_vel) < 1e-5):
            use_global_orient = True

    if use_global_orient:
        # === Mode B: single frame / mesh reconstruction ===
        # Recover the rotation directly from global_orient_6d, no integration
        global_orient_6d = data[..., 263:269]
        r_rot_quat = cont6d_to_quaternion(global_orient_6d)

        # Recover the position.
        # In single-frame mode there is no velocity integration; the XZ position
        # defaults to 0 (it would have to be provided externally, which is not
        # possible here). The Y position comes from root_y.
        r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
        r_pos[..., 1] = data[..., 3]  # root_y

    else:
        # === Mode A: motion generation ===
        # Standard HumanML3D logic: recover by integrating velocities
        rot_vel = data[..., 0]
        r_rot_ang = torch.zeros_like(rot_vel).to(data.device)

        # Integrate the rotation velocity to get the Y-axis rotation angle
        r_rot_ang[..., 1:] = rot_vel[..., :-1]
        r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

        # Convert to quaternion
        r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
        r_rot_quat[..., 0] = torch.cos(r_rot_ang)
        r_rot_quat[..., 2] = torch.sin(r_rot_ang)

        # Integrate the linear velocity to get the root position
        r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
        r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]

        # Apply the Y-axis rotation
        r_pos = qrot(qinv(r_rot_quat), r_pos)
        r_pos = torch.cumsum(r_pos, dim=-2)

        # The Y coordinate comes directly from the data
        r_pos[..., 1] = data[..., 3]

    if is_numpy:
        return r_rot_quat.numpy(), r_pos.numpy()
    return r_rot_quat, r_pos


def recover_from_ric(data: Union[np.ndarray, torch.Tensor],
                      joints_num: int = 22) -> Union[np.ndarray, torch.Tensor]:
    """
    Recover global joint positions from RIC (Root-Invariant Coordinates).

    Args:
        data: (..., 263) or (..., 269) motion representation
        joints_num: number of joints

    Returns:
        positions: (..., joints_num, 3) global joint positions
    """
    is_numpy = isinstance(data, np.ndarray)
    if is_numpy:
        data = torch.from_numpy(data).float()

    r_rot_quat, r_pos = recover_root_rot_pos(data)

    # Extract local joint positions
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    # Apply the Y-axis rotation
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    # Add the root XZ position
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    # Concatenate the root joint
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    if is_numpy:
        return positions.numpy()
    return positions


def recover_from_rot(data: Union[np.ndarray, torch.Tensor],
                      joints_num: int = 22,
                      skeleton: Skeleton = None) -> Union[np.ndarray, torch.Tensor]:
    """
    Recover joint positions from the rotation representation.

    Args:
        data: (..., 263) or (..., 269) motion representation
        joints_num: number of joints
        skeleton: Skeleton object. If not provided, the default skeleton is used.

    Returns:
        positions: (..., joints_num, 3) joint positions
    """
    is_numpy = isinstance(data, np.ndarray)
    if is_numpy:
        data = torch.from_numpy(data).float()

    if skeleton is None:
        skeleton, target_offset = _get_skeleton_and_offsets()
        skeleton.set_offset(target_offset)

    r_rot_quat, r_pos = recover_root_rot_pos(data)
    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    # Extract joint rotations
    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]

    # Concatenate the root rotation
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    # Forward kinematics
    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos.view(-1, 3))

    if is_numpy:
        return positions.numpy()
    return positions


def recover_smpl_params_from_269(
    data: Union[np.ndarray, torch.Tensor],
    joints_num: int = 22,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract "SMPL semantic" pose parameters from the 269-dim representation
    (for fair comparison with mesh/SMPL-based methods).

    Important:
    - This function only "recovers SMPL pose" in a meaningful sense when the
      rot_data/global_orient of your 269 data actually come from SMPL poses
      (e.g. the H36M conversion script with ROTATION_MODE=full/global).
    - If the 269 comes from the HumanML3D canonical/IK semantics (the native
      output of process_motion_generation / process_mesh_reconstruction), these
      rotations are HumanML3D IK results, not SMPL body_pose.

    Args:
        data: (..., 269) or (..., 263)
        joints_num: number of joints (default 22, consistent with the 269
                    definition in this repo)

    Returns:
        pose_aa_22: (N, 22, 3) axis-angle (numpy)
        rotmat_22:  (N, 22, 3, 3) rotation matrices (numpy)
    """
    # Convert to numpy and flatten the batch
    if isinstance(data, torch.Tensor):
        data_np = data.detach().cpu().numpy()
    else:
        data_np = np.asarray(data)

    if data_np.shape[-1] == 263:
        raise ValueError("Input has 263 dims (no global_orient_6d). Cannot recover 22-joint SMPL params from 263.")
    if data_np.shape[-1] != 269:
        raise ValueError(f"Expected last dim 269, got {data_np.shape[-1]}")

    pose_6d = Repr269.to_pose_6d_22(data_np)  # (..., 22, 6)
    pose_6d = pose_6d.reshape(-1, joints_num, 6)
    rotmat_22 = cont6d_to_matrix_np(pose_6d)  # (N, 22, 3, 3)
    pose_aa_22 = matrix_to_axis_angle_np(rotmat_22)  # (N, 22, 3)
    return pose_aa_22, rotmat_22


def recover_smpl_joints_from_269(
    data: Union[np.ndarray, torch.Tensor],
    body_model: Any,
    betas: Optional[Union[np.ndarray, torch.Tensor]] = None,
    transl: Optional[Union[np.ndarray, torch.Tensor]] = None,
    joints_num: int = 22,
    device: str = "cpu",
) -> np.ndarray:
    """
    Recover joint positions from 269 (storing SMPL rotation semantics) using an
    SMPL/SMPLH body model.

    This is the recovery path that stays "as close as possible to the original
    SMPL/H36M" data:
    - 269 -> (22, axis-angle) -> SMPL forward model -> joints
    - Does not depend on the HumanML3D target skeleton offsets and does not
      impose canonical facing constraints.

    Args:
        data: (..., 269). rot_data/global_orient must have SMPL pose semantics
              (e.g. H36M conversion script with ROTATION_MODE=full/global)
        body_model: smplx.create(...) / smplx.SMPL / smplx.SMPLH instance
        betas: (10,) or (N, 10). None means zeros
        transl: (3,) or (N, 3). None means zeros
        joints_num: 22 (per this repo's 269 definition)
        device: cpu/cuda

    Returns:
        joints_22: (N, 22, 3) numpy
    """
    pose_aa_22, _ = recover_smpl_params_from_269(data, joints_num=joints_num)  # (N, 22, 3)
    N = pose_aa_22.shape[0]

    # Handle betas/transl
    if betas is None:
        betas_np = np.zeros((N, 10), dtype=np.float32)
    else:
        if isinstance(betas, torch.Tensor):
            betas_np = betas.detach().cpu().numpy()
        else:
            betas_np = np.asarray(betas)
        if betas_np.ndim == 1:
            betas_np = np.repeat(betas_np[None, :], N, axis=0)

    if transl is None:
        transl_np = np.zeros((N, 3), dtype=np.float32)
    else:
        if isinstance(transl, torch.Tensor):
            transl_np = transl.detach().cpu().numpy()
        else:
            transl_np = np.asarray(transl)
        if transl_np.ndim == 1:
            transl_np = np.repeat(transl_np[None, :], N, axis=0)

    # SMPLX/SMPLH call
    body_model = body_model.to(device)
    pose_t = torch.from_numpy(pose_aa_22).float().to(device)  # (N, 22, 3)
    betas_t = torch.from_numpy(betas_np).float().to(device)   # (N, 10)
    transl_t = torch.from_numpy(transl_np).float().to(device) # (N, 3)

    global_orient = pose_t[:, 0]                      # (N, 3)
    body_pose = pose_t[:, 1:].reshape(N, -1)          # (N, 63)

    # SMPLH usually requires hand poses; for robustness pass zeros (flat hands)
    # by default and fall back when the model does not support them.
    model_kwargs = dict(
        global_orient=global_orient,
        body_pose=body_pose,
        betas=betas_t,
        transl=transl_t,
    )
    with torch.no_grad():
        try:
            # SMPLH compatibility: fill in hand pose parameters
            model_kwargs_with_hands = dict(model_kwargs)
            model_kwargs_with_hands["left_hand_pose"] = torch.zeros(N, 45, device=device)
            model_kwargs_with_hands["right_hand_pose"] = torch.zeros(N, 45, device=device)
            out = body_model(**model_kwargs_with_hands)
        except TypeError:
            out = body_model(**model_kwargs)

    joints = out.joints[:, :joints_num].detach().cpu().numpy()  # (N, 22, 3)
    return joints


# ============================================================================
# Repr269 utility class
# ============================================================================

class Repr269:
    """Utility class for the 269-dim unified representation"""

    # Index definitions
    IDX = {
        'root_rot_vel': 0,
        'root_lin_vel': slice(1, 3),
        'root_y': 3,
        'ric_data': slice(4, 67),
        'rot_data': slice(67, 193),
        'local_vel': slice(193, 259),
        'foot_contact': slice(259, 263),
        'global_orient': slice(263, 269),
    }

    # Dimension definitions
    DIMS = {
        'root_rot_vel': 1,
        'root_lin_vel': 2,
        'root_y': 1,
        'ric_data': 63,
        'rot_data': 126,
        'local_vel': 66,
        'foot_contact': 4,
        'global_orient': 6,
        'total': 269,
        'humanml3d': 263,
    }

    @staticmethod
    def to_humanml3d_263(data: np.ndarray) -> np.ndarray:
        """
        Extract the HumanML3D-compatible 263 dims.

        Args:
            data: (..., 269)

        Returns:
            data_263: (..., 263)
        """
        return data[..., :263]

    @staticmethod
    def to_pose_6d_22(data: np.ndarray) -> np.ndarray:
        """
        Extract the 22-joint 6D rotations (for mesh reconstruction).

        Args:
            data: (..., 269)

        Returns:
            pose_6d: (..., 22, 6)
        """
        rot_data = data[..., 67:193].reshape(*data.shape[:-1], 21, 6)
        global_orient = data[..., 263:269].reshape(*data.shape[:-1], 1, 6)
        return np.concatenate([global_orient, rot_data], axis=-2)  # (..., 22, 6)

    @staticmethod
    def to_smpl_rotmat(data: np.ndarray) -> np.ndarray:
        """
        Convert to SMPL rotation matrices.

        Args:
            data: (..., 269)

        Returns:
            rotmat: (..., 22, 3, 3)
        """
        pose_6d = Repr269.to_pose_6d_22(data)
        return cont6d_to_matrix_np(pose_6d)

    @staticmethod
    def to_smpl_axis_angle(data: np.ndarray) -> np.ndarray:
        """
        Convert to the SMPL axis-angle representation.

        Args:
            data: (..., 269)

        Returns:
            axis_angle: (..., 22, 3)
        """
        rotmat = Repr269.to_smpl_rotmat(data)
        return matrix_to_axis_angle_np(rotmat)

    @staticmethod
    def is_single_frame(data: np.ndarray) -> bool:
        """
        Check whether the data is single-frame (all velocities zero).

        Args:
            data: (..., 269)

        Returns:
            is_single: True if the velocity dims are all zero (or near zero)
        """
        velocity_dims = np.concatenate([
            data[..., 0:3],      # root velocities
            data[..., 193:259]   # local velocities
        ], axis=-1)
        return np.allclose(velocity_dims, 0, atol=1e-6)

    @staticmethod
    def get_root_position(data: np.ndarray) -> np.ndarray:
        """
        Recover the root trajectory from 269 (valid for sequence data only).

        Args:
            data: (..., T, 269)

        Returns:
            root_pos: (..., T, 3)
        """
        _, r_pos = recover_root_rot_pos(data)
        return r_pos

    @staticmethod
    def get_root_rotation(data: np.ndarray) -> np.ndarray:
        """
        Get the root rotation quaternion.

        For sequence data: obtained by integrating velocities.
        For single-frame data: converted directly from global_orient_6d.

        Args:
            data: (..., 269)

        Returns:
            root_rot: (..., 4) quaternion
        """
        if Repr269.is_single_frame(data):
            # Single frame: convert directly from 6D
            global_orient_6d = data[..., 263:269]
            return cont6d_to_quaternion_np(global_orient_6d)
        else:
            # Sequence: integrate velocities
            r_rot_quat, _ = recover_root_rot_pos(data)
            return r_rot_quat

    @staticmethod
    def get_joint_positions(data: np.ndarray, use_rot: bool = False) -> np.ndarray:
        """
        Get joint positions.

        Args:
            data: (..., 269)
            use_rot: whether to recover via rotations (more precise but slower)

        Returns:
            positions: (..., 22, 3)
        """
        if use_rot:
            return recover_from_rot(data)
        else:
            return recover_from_ric(data)

    @staticmethod
    def get_foot_contact(data: np.ndarray) -> np.ndarray:
        """
        Get the foot contact information.

        Args:
            data: (..., 269)

        Returns:
            foot_contact: (..., 4) [l_heel, l_toe, r_heel, r_toe]
        """
        return data[..., 259:263]

    @staticmethod
    def validate(data: np.ndarray, strict: bool = True) -> Dict[str, Any]:
        """
        Validate the 269-dim representation.

        Args:
            data: (..., 269)
            strict: whether to validate strictly

        Returns:
            result: dict with validation results
                - valid: bool, whether the data is valid
                - errors: list of errors
                - warnings: list of warnings
        """
        errors = []
        warnings = []

        # Check the shape
        if data.shape[-1] != 269:
            errors.append(f"Expected 269 dimensions, got {data.shape[-1]}")

        # Check for NaN
        if np.any(np.isnan(data)):
            errors.append("Data contains NaN values")

        # Check for Inf
        if np.any(np.isinf(data)):
            errors.append("Data contains Inf values")

        # Check orthogonality of the 6D rotations
        pose_6d = Repr269.to_pose_6d_22(data)

        # Check whether the first column is a unit vector
        col1 = pose_6d[..., :3]
        col1_norm = np.linalg.norm(col1, axis=-1)
        if not np.allclose(col1_norm, 1.0, atol=0.1):
            warnings.append("6D rotation first column not normalized")

        # Check that foot contact is within [0, 1]
        foot_contact = data[..., 259:263]
        if np.any(foot_contact < -0.1) or np.any(foot_contact > 1.1):
            warnings.append("Foot contact values out of expected range [0, 1]")

        # Check velocities in single-frame mode
        is_single = Repr269.is_single_frame(data)
        if is_single:
            if not np.allclose(data[..., 0:3], 0, atol=1e-6):
                warnings.append("Single frame mode but root velocity not zero")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'is_single_frame': is_single,
        }

    @staticmethod
    def create_from_components(
        root_rot_vel: np.ndarray = None,
        root_lin_vel: np.ndarray = None,
        root_y: np.ndarray = None,
        ric_data: np.ndarray = None,
        rot_data: np.ndarray = None,
        local_vel: np.ndarray = None,
        foot_contact: np.ndarray = None,
        global_orient: np.ndarray = None,
        batch_shape: Tuple = None,
    ) -> np.ndarray:
        """
        Create the 269-dim representation from individual components.

        Args:
            components; zeros are used for any that are not provided
            batch_shape: batch shape used to determine the output shape

        Returns:
            data: (..., 269)
        """
        if batch_shape is None:
            # Try to infer from the provided arguments
            for arr in [root_rot_vel, root_lin_vel, root_y, ric_data,
                       rot_data, local_vel, foot_contact, global_orient]:
                if arr is not None:
                    batch_shape = arr.shape[:-1]
                    break
            if batch_shape is None:
                batch_shape = ()

        # Create default values
        if root_rot_vel is None:
            root_rot_vel = np.zeros(batch_shape + (1,))
        if root_lin_vel is None:
            root_lin_vel = np.zeros(batch_shape + (2,))
        if root_y is None:
            root_y = np.zeros(batch_shape + (1,))
        if ric_data is None:
            ric_data = np.zeros(batch_shape + (63,))
        if rot_data is None:
            rot_data = np.zeros(batch_shape + (126,))
        if local_vel is None:
            local_vel = np.zeros(batch_shape + (66,))
        if foot_contact is None:
            foot_contact = np.full(batch_shape + (4,), 0.5)
        if global_orient is None:
            # Default to the 6D representation of the identity rotation
            global_orient = np.zeros(batch_shape + (6,))
            global_orient[..., 0] = 1.0  # first column, x
            global_orient[..., 4] = 1.0  # second column, y

        return np.concatenate([
            root_rot_vel,
            root_lin_vel,
            root_y,
            ric_data,
            rot_data,
            local_vel,
            foot_contact,
            global_orient,
        ], axis=-1)


# ============================================================================
# Normalization
# ============================================================================

class Normalizer:
    """Normalization utility class"""

    def __init__(self, mean: np.ndarray = None, std: np.ndarray = None):
        """
        Initialize the normalizer.

        Args:
            mean: (269,) or (263,) mean
            std: (269,) or (263,) standard deviation
        """
        self.mean = mean
        self.std = std

        # If only 263-dim statistics are provided, extend them to 269 dims
        if mean is not None and len(mean) == 263:
            self.mean_263 = mean
            self.std_263 = std
            # The extra 6 dims (global_orient_6d) get special treatment:
            # 6D rotations are usually within [-1, 1] and can be left
            # unnormalized or use fixed values.
            self.mean = np.concatenate([mean, np.zeros(6)])
            self.std = np.concatenate([std, np.ones(6)])
        else:
            self.mean_263 = mean[:263] if mean is not None else None
            self.std_263 = std[:263] if std is not None else None

    def normalize(self, data: np.ndarray, keep_global_orient: bool = True) -> np.ndarray:
        """
        Normalize the data.

        Args:
            data: (..., 269) or (..., 263)
            keep_global_orient: whether to leave global_orient unnormalized

        Returns:
            normalized: normalized data
        """
        if self.mean is None:
            raise ValueError("Normalizer not initialized with mean/std")

        dim = data.shape[-1]

        if dim == 263:
            return (data - self.mean_263) / self.std_263
        elif dim == 269:
            if keep_global_orient:
                # Normalize the first 263 dims, leave the last 6 unchanged
                normalized = data.copy()
                normalized[..., :263] = (data[..., :263] - self.mean_263) / self.std_263
                return normalized
            else:
                return (data - self.mean) / self.std
        else:
            raise ValueError(f"Expected 263 or 269 dimensions, got {dim}")

    def denormalize(self, data: np.ndarray, keep_global_orient: bool = True) -> np.ndarray:
        """
        Denormalize the data.

        Args:
            data: (..., 269) or (..., 263)
            keep_global_orient: whether global_orient was left unnormalized

        Returns:
            denormalized: denormalized data
        """
        if self.mean is None:
            raise ValueError("Normalizer not initialized with mean/std")

        dim = data.shape[-1]

        if dim == 263:
            return data * self.std_263 + self.mean_263
        elif dim == 269:
            if keep_global_orient:
                denormalized = data.copy()
                denormalized[..., :263] = data[..., :263] * self.std_263 + self.mean_263
                return denormalized
            else:
                return data * self.std + self.mean
        else:
            raise ValueError(f"Expected 263 or 269 dimensions, got {dim}")

    @classmethod
    def compute_statistics(cls, data_list: list) -> 'Normalizer':
        """
        Compute statistics from a list of data arrays.

        Args:
            data_list: list of (T, 269) data arrays

        Returns:
            normalizer: initialized normalizer
        """
        all_data = np.concatenate(data_list, axis=0)
        mean = np.mean(all_data, axis=0)
        std = np.std(all_data, axis=0)
        std[std < 1e-6] = 1.0  # avoid division by zero
        return cls(mean, std)

    def save(self, save_dir: str, prefix: str = ''):
        """Save the statistics"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, f'{prefix}Mean.npy'), self.mean)
        np.save(os.path.join(save_dir, f'{prefix}Std.npy'), self.std)

    @classmethod
    def load(cls, save_dir: str, prefix: str = '') -> 'Normalizer':
        """Load the statistics"""
        mean = np.load(os.path.join(save_dir, f'{prefix}Mean.npy'))
        std = np.load(os.path.join(save_dir, f'{prefix}Std.npy'))
        return cls(mean, std)


# ============================================================================
# Consistency checking and conversion
# ============================================================================

def check_global_orient_consistency(data: np.ndarray, tolerance: float = 0.1) -> Dict[str, Any]:
    """
    Check the consistency between global_orient_6d and root_rot_velocity.

    For a motion sequence, global_orient should satisfy:
    global_orient[t] ~= integrate(root_rot_velocity[0:t])

    Args:
        data: (T, 269) motion sequence
        tolerance: tolerance (radians)

    Returns:
        result: dict with the consistency check results
    """
    if Repr269.is_single_frame(data):
        return {'consistent': True, 'message': 'Single frame mode, no velocity to check'}

    # Rotation recovered from velocity
    r_rot_quat, _ = recover_root_rot_pos(data)  # (T, 4)
    recovered_6d = quaternion_to_cont6d_np(r_rot_quat)  # (T, 6)

    # global_orient stored in the data
    stored_6d = data[:, 263:269]  # (T, 6)

    # Compare
    diff = np.abs(recovered_6d - stored_6d)
    max_diff = diff.max()
    mean_diff = diff.mean()

    consistent = max_diff < tolerance

    return {
        'consistent': consistent,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'message': 'Consistent' if consistent else f'Inconsistent: max_diff={max_diff:.4f}',
    }


def make_consistent(data: np.ndarray, use_velocity: bool = True) -> np.ndarray:
    """
    Make global_orient_6d consistent with root_rot_velocity.

    Args:
        data: (T, 269) motion sequence
        use_velocity: True to recover orient from velocity, False to compute
                      velocity from orient

    Returns:
        consistent_data: (T, 269) consistent data
    """
    data = data.copy()

    if Repr269.is_single_frame(data):
        return data

    if use_velocity:
        # Recover orient from velocity
        r_rot_quat, _ = recover_root_rot_pos(data)
        recovered_6d = quaternion_to_cont6d_np(r_rot_quat)
        data[:, 263:269] = recovered_6d
    else:
        # Compute velocity from orient
        stored_6d = data[:, 263:269]
        stored_quat = cont6d_to_quaternion_np(stored_6d)

        # Compute the angular velocity
        r_velocity = qmul_np(stored_quat[1:], qinv_np(stored_quat[:-1]))
        r_velocity_y = np.arcsin(r_velocity[:, 2:3])

        # Update velocity (first frame velocity set to 0)
        data[0, 0] = 0
        data[1:, 0:1] = r_velocity_y

    return data


# ============================================================================
# Exported symbols
# ============================================================================

__all__ = [
    # Constants
    'T2M_RAW_OFFSETS',
    'T2M_KINEMATIC_CHAIN',
    'FACE_JOINT_INDX',
    'JOINTS_NUM',
    'T2M_TGT_SKEL_ID',
    'T2M_TGT_SKEL_PATH',

    # Quaternion and rotation functions
    'qinv', 'qinv_np',
    'qnormalize',
    'qmul', 'qmul_np',
    'qrot', 'qrot_np',
    'qbetween', 'qbetween_np',
    'qfix',
    'quaternion_to_matrix', 'quaternion_to_matrix_np',
    'quaternion_to_cont6d', 'quaternion_to_cont6d_np',
    'cont6d_to_matrix', 'cont6d_to_matrix_np',
    'matrix_to_cont6d', 'matrix_to_cont6d_np',
    'matrix_to_quaternion', 'matrix_to_quaternion_np',
    'cont6d_to_quaternion', 'cont6d_to_quaternion_np',
    'matrix_to_axis_angle', 'matrix_to_axis_angle_np',
    'axis_angle_to_matrix', 'axis_angle_to_matrix_np',

    # Skeleton class
    'Skeleton',

    # Core processing functions
    'uniform_skeleton',
    'foot_detect',
    'process_motion_generation',
    'process_mesh_reconstruction',
    'process_from_smpl',

    # Target skeleton configuration
    'set_target_skeleton_path',
    'get_target_offsets',

    # Recovery functions
    'recover_root_rot_pos',
    'recover_from_ric',
    'recover_from_rot',

    # Utility classes
    'Repr269',
    'Normalizer',

    # Consistency checking
    'check_global_orient_consistency',
    'make_consistent',
]


# ============================================================================
# Self-test code
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Unified human motion representation (269-dim) - self test")
    print("=" * 60)

    # Test 1: create a simple T-pose and process it
    print("\nTest 1: mesh reconstruction mode (single frame)")
    print("-" * 40)

    # Create a simple T-pose
    _, target_offset = _get_skeleton_and_offsets()
    skel = Skeleton(_n_raw_offsets, _kinematic_chain, "cpu")
    skel.set_offset(target_offset)

    # T-pose quaternions
    t_pose_quat = np.zeros((1, 22, 4))
    t_pose_quat[..., 0] = 1.0
    t_pose_root = np.array([[0.0, 0.9, 0.0]])
    t_pose_joints = skel.forward_kinematics_np(t_pose_quat, t_pose_root)

    print(f"Input joint positions shape: {t_pose_joints.shape}")

    # Process the single frame
    single_frame_data = process_mesh_reconstruction(t_pose_joints[0])
    print(f"Output data shape: {single_frame_data.shape}")
    print(f"Is single-frame mode: {Repr269.is_single_frame(single_frame_data)}")

    # Validate
    validation = Repr269.validate(single_frame_data)
    print(f"Validation result: {validation}")

    # Test 2: motion generation mode
    print("\nTest 2: motion generation mode (sequence)")
    print("-" * 40)

    # Create a simple motion sequence
    T = 30
    motion_joints = np.tile(t_pose_joints, (T, 1, 1))

    # Add some motion (arm swing)
    for t in range(T):
        angle = np.sin(t * 0.2) * 0.5
        # Left arm
        motion_joints[t, 18, 0] += angle * 0.2
        # Root translation
        motion_joints[t, :, 0] += t * 0.01

    print(f"Input joint positions shape: {motion_joints.shape}")

    # Process the sequence
    motion_data = process_motion_generation(motion_joints)
    print(f"Output data shape: {motion_data.shape}")
    print(f"Is single-frame mode: {Repr269.is_single_frame(motion_data)}")

    # Validate
    validation = Repr269.validate(motion_data)
    print(f"Validation result: {validation}")

    # Consistency check
    consistency = check_global_orient_consistency(motion_data)
    print(f"Consistency check: {consistency}")

    # Test 3: recover joint positions
    print("\nTest 3: recover joint positions")
    print("-" * 40)

    recovered_positions = recover_from_ric(motion_data)
    print(f"Recovered joint positions shape: {recovered_positions.shape}")

    # Test 4: extract SMPL parameters
    print("\nTest 4: extract SMPL parameters")
    print("-" * 40)

    pose_6d = Repr269.to_pose_6d_22(motion_data)
    print(f"6D rotation shape: {pose_6d.shape}")

    rotmat = Repr269.to_smpl_rotmat(motion_data)
    print(f"Rotation matrix shape: {rotmat.shape}")

    axis_angle = Repr269.to_smpl_axis_angle(motion_data)
    print(f"Axis-angle shape: {axis_angle.shape}")

    # Test 5: create from components
    print("\nTest 5: create from components")
    print("-" * 40)

    created_data = Repr269.create_from_components(
        root_y=np.array([[0.9]]),
        batch_shape=(1,)
    )
    print(f"Created data shape: {created_data.shape}")
    print(f"Validation: {Repr269.validate(created_data)}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
