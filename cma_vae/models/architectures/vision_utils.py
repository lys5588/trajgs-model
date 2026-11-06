"""
Utility functions for visual feature extraction.

Used by the image-guided MldVaeV2Vision model:
1. Recover 3D coordinates of 22 SMPL joints from 269-dim motion features
2. Weak-perspective projection onto the 2D image plane
3. Coordinate normalization for grid_sample
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict


def recover_root_rot_pos(data: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Recover root rotation and position from 263/269-dim data.
    
    Args:
        data: [B, T, 263/269] or [T, 263/269] motion features
    
    Returns:
        r_rot_quat: root Y-axis rotation quaternion
        r_pos: root position [x, y, z]
    """
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    
    # Accumulate angular velocity to get the rotation angle
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)
    
    # Convert to a quaternion (Y-axis rotation only)
    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)
    
    # Recover the root position
    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    
    # Apply the Y-axis rotation to the position
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    
    # The Y coordinate is taken directly from the data
    r_pos[..., 1] = data[..., 3]
    
    return r_rot_quat, r_pos


def qinv(q: Tensor) -> Tensor:
    """
    Quaternion inverse.
    
    Args:
        q: [..., 4] quaternion (w, x, y, z)
    
    Returns:
        q_inv: [..., 4] inverse quaternion
    """
    assert q.shape[-1] == 4
    return q * torch.tensor([1, -1, -1, -1], dtype=q.dtype, device=q.device)


def qrot(q: Tensor, v: Tensor) -> Tensor:
    """
    Rotate a vector by a quaternion.
    
    Args:
        q: [..., 4] quaternion (w, x, y, z)
        v: [..., 3] vector
    
    Returns:
        v_rot: [..., 3] rotated vector
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    
    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    
    return v + 2 * (q[..., :1] * uv + uuv)


def recover_from_ric(data: Tensor, joints_num: int = 22) -> Tensor:
    """
    Recover 3D joint coordinates from the 263/269-dim representation.
    
    This is the core routine for recovering joint positions from motion features.
    
    Args:
        data: [B, T, 263/269] or [T, 263/269] motion features
            - index 0: root_rot_velocity (Y-axis angular velocity)
            - indices 1-2: root_linear_velocity (linear velocity on the XZ plane)
            - index 3: root_y (root height)
            - indices 4-66: ric_data (local positions of 21 non-root joints)
            - ...
        joints_num: number of joints (default 22 for HumanML3D)
    
    Returns:
        positions: [B, T, joints_num, 3] joint 3D coordinates (Y-up coordinate system)
    """
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    
    # Extract ric_data (local positions of 21 non-root joints),
    # indices 4 to 4 + (joints_num - 1) * 3
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    # positions: [B, T, 21, 3]
    
    # Apply the Y-axis rotation to the local joint positions
    positions = qrot(
        qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), 
        positions
    )
    
    # Add the root XZ translation
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    
    # Concatenate the root joint with the other joints
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    # positions: [B, T, 22, 3]
    
    return positions


def weak_perspective_project(
    joints_3d: Tensor,
    fx: Tensor,
    fy: Tensor, 
    cx: Tensor,
    cy: Tensor,
) -> Tensor:
    """
    Weak-perspective projection: project 3D joint coordinates onto the 2D image plane.
    
    Note: joints_3d is assumed to be in camera coordinates. If joints_3d is in
    the HumanML3D Y-up world coordinate system, convert it to camera coordinates first.
    
    Args:
        joints_3d: [B, T, J, 3] 3D joint coordinates (camera coordinates)
        fx: focal length x (scalar or [B] tensor)
        fy: focal length y
        cx: principal point x
        cy: principal point y
    
    Returns:
        joints_2d: [B, T, J, 2] 2D pixel coordinates
    """
    # Perspective division
    z = joints_3d[..., 2:3].clamp(min=1e-6)  # avoid division by zero
    joints_2d = joints_3d[..., :2] / z
    
    # Handle scalar focal lengths and principal points
    if isinstance(fx, (int, float)):
        fx = torch.tensor(fx, dtype=joints_3d.dtype, device=joints_3d.device)
    if isinstance(fy, (int, float)):
        fy = torch.tensor(fy, dtype=joints_3d.dtype, device=joints_3d.device)
    if isinstance(cx, (int, float)):
        cx = torch.tensor(cx, dtype=joints_3d.dtype, device=joints_3d.device)
    if isinstance(cy, (int, float)):
        cy = torch.tensor(cy, dtype=joints_3d.dtype, device=joints_3d.device)
    
    # Expand dimensions for broadcasting
    while fx.dim() < joints_2d.dim() - 1:
        fx = fx.unsqueeze(-1)
        fy = fy.unsqueeze(-1)
        cx = cx.unsqueeze(-1)
        cy = cy.unsqueeze(-1)
    
    # Apply camera intrinsics
    joints_2d_pixel = torch.zeros_like(joints_2d)
    joints_2d_pixel[..., 0] = joints_2d[..., 0] * fx.squeeze(-1) + cx.squeeze(-1)
    joints_2d_pixel[..., 1] = joints_2d[..., 1] * fy.squeeze(-1) + cy.squeeze(-1)
    
    return joints_2d_pixel


def normalize_coords_for_grid_sample(
    joints_2d: Tensor,
    img_h: int,
    img_w: int,
) -> Tensor:
    """
    Normalize pixel coordinates to [-1, 1] for F.grid_sample.
    
    Args:
        joints_2d: [B, T, J, 2] pixel coordinates (x, y)
        img_h: image height
        img_w: image width
    
    Returns:
        joints_normalized: [B, T, J, 2] normalized coordinates in [-1, 1]
    """
    joints_normalized = joints_2d.clone()
    # X: [0, W] -> [-1, 1]
    joints_normalized[..., 0] = 2 * joints_2d[..., 0] / img_w - 1
    # Y: [0, H] -> [-1, 1]
    joints_normalized[..., 1] = 2 * joints_2d[..., 1] / img_h - 1
    
    # Clamp to avoid out-of-range coordinates
    joints_normalized = joints_normalized.clamp(-1, 1)
    
    return joints_normalized


def world_to_camera(
    joints_3d: Tensor,
    R: Optional[Tensor] = None,
    T: Optional[Tensor] = None,
) -> Tensor:
    """
    Transform joints from world coordinates to camera coordinates.
    
    HumanML3D uses a Y-up coordinate system; joints must be converted to
    camera coordinates before projection.
    
    Args:
        joints_3d: [B, T, J, 3] joints in world coordinates (Y-up)
        R: [B, 3, 3] or [3, 3] rotation matrix (world to camera)
        T: [B, 3] or [3] translation vector
    
    Returns:
        joints_cam: [B, T, J, 3] joints in camera coordinates
    """
    if R is None and T is None:
        # Default transform: Y-up world coordinates -> standard camera coordinates.
        # Assumes the camera faces the Z+ direction with Y pointing down.
        # This is a simplified transform; adjust it based on actual camera parameters.
        joints_cam = joints_3d.clone()
        # Flip the Y axis (Y-up -> Y-down for camera)
        joints_cam[..., 1] = -joints_3d[..., 1]
        # Add a default depth offset so the subject is in front of the camera
        joints_cam[..., 2] = joints_3d[..., 2] + 3.0  # assume the subject is 3m in front of the camera
        return joints_cam
    
    B, T, J, _ = joints_3d.shape
    
    # Apply rotation
    if R is not None:
        if R.dim() == 2:
            R = R.unsqueeze(0).expand(B, -1, -1)
        # [B, T, J, 3] @ [B, 3, 3]^T -> [B, T, J, 3]
        joints_flat = joints_3d.view(B, T * J, 3)
        joints_rot = torch.bmm(joints_flat, R.transpose(1, 2))
        joints_cam = joints_rot.view(B, T, J, 3)
    else:
        joints_cam = joints_3d.clone()
    
    # Apply translation
    if T is not None:
        if T.dim() == 1:
            T = T.unsqueeze(0).expand(B, -1)
        joints_cam = joints_cam + T.view(B, 1, 1, 3)
    
    return joints_cam


def get_joints_for_vision(
    motion_269: Tensor,
    camera_params: Optional[Dict] = None,
    img_h: int = 256,
    img_w: int = 192,
    joints_num: int = 22,
    use_joint3d_image: bool = True,
    joints_2d_precomputed: Optional[Tensor] = None,
) -> Tensor:
    """
    Get normalized joint coordinates for grid_sample from 269-dim motion features.
    
    Following the MTVCrafter implementation, several input sources are
    supported (in priority order):
    1. Precomputed 2D joints (produced by generate_2d_skeleton_from_gt.py) - recommended
    2. 'joint3d_image' in camera_params (H36M format)
    3. Recover 3D joints from motion and project them (computed at runtime)
    
    Args:
        motion_269: [B, T, 269] motion features
        camera_params: camera parameter dict, may contain:
            - 'joint3d_image': [B, T, J, 3] projected joint coordinates (XY in pixels)
            - 'fx', 'fy', 'cx', 'cy', 'R', 'T': camera parameters
        img_h: image height
        img_w: image width
        joints_num: number of joints
        use_joint3d_image: whether to prefer joint3d_image
        joints_2d_precomputed: [B, T, J, 2] precomputed pixel coordinates (recommended)
    
    Returns:
        joints_normalized: [B, T, J, 2] normalized coordinates in [-1, 1]
    """
    # ===== Option 0: precomputed 2D joints (most recommended) =====
    if joints_2d_precomputed is not None:
        # Precomputed joints are already in pixel coordinates; just normalize.
        joints_normalized = normalize_coords_for_grid_sample(
            joints_2d_precomputed, img_h, img_w
        )
        return joints_normalized
    # ===== Option 1: precomputed joint3d_image (recommended for H36M) =====
    if use_joint3d_image and camera_params is not None and 'joint3d_image' in camera_params:
        joint3d_image = camera_params['joint3d_image']  # [B, T, J, 3]
        # XY of joint3d_image are pixel coordinates; Z is relative depth.
        # Following MTVCrafter: grid_sample_ref = joint3d_images[..., :2]
        joints_2d = joint3d_image[..., :2]  # [B, T, J, 2]
        
        joints_normalized = normalize_coords_for_grid_sample(joints_2d, img_h, img_w)
        return joints_normalized
    
    # ===== Option 2: recover joints from motion and project =====
    # 1. Recover 3D joint coordinates
    joints_3d = recover_from_ric(motion_269, joints_num)
    # joints_3d: [B, T, 22, 3] Y-up world coordinates
    
    if camera_params is not None and 'R' in camera_params:
        # 2. World coordinates -> camera coordinates
        R = camera_params.get('R', None)
        T = camera_params.get('T', None)
        joints_cam = world_to_camera(joints_3d, R, T)
        
        # 3. Weak-perspective projection
        fx = camera_params.get('fx', img_w)
        fy = camera_params.get('fy', img_h)
        cx = camera_params.get('cx', img_w / 2)
        cy = camera_params.get('cy', img_h / 2)
        joints_2d = weak_perspective_project(joints_cam, fx, fy, cx, cy)
    else:
        # Simplified orthographic projection: assume the camera directly faces
        # the subject and map XZ coordinates to image XY. This is an
        # approximation; provide camera parameters or joint3d_image in practice.
        
        # Center the joints
        joints_centered = joints_3d.clone()
        center = joints_3d.mean(dim=2, keepdim=True)
        joints_centered = joints_centered - center
        
        # Scale to the image size (assume the joint range is roughly +/-1m)
        scale = min(img_h, img_w) / 2.0 / 1.0
        
        # Use X and -Y as image coordinates (Y flipped since image Y points down)
        joints_2d = torch.zeros(joints_3d.shape[:-1] + (2,), 
                                dtype=joints_3d.dtype, device=joints_3d.device)
        joints_2d[..., 0] = joints_centered[..., 0] * scale + img_w / 2
        joints_2d[..., 1] = -joints_centered[..., 1] * scale + img_h / 2
    
    # 4. Normalize to [-1, 1]
    joints_normalized = normalize_coords_for_grid_sample(joints_2d, img_h, img_w)
    
    return joints_normalized


def sample_features_by_joints(
    features: Tensor,
    joints_normalized: Tensor,
) -> Tensor:
    """
    Sample a feature map at joint locations using grid_sample.
    
    Args:
        features: [B*T, C, H, W] image feature maps
        joints_normalized: [B*T, J, 2] normalized joint coordinates in [-1, 1]
    
    Returns:
        sampled_features: [B*T, J, C] sampled features
    """
    BT, C, H, W = features.shape
    J = joints_normalized.shape[1]
    
    # grid_sample expects a sampling grid of shape [B, H_out, W_out, 2];
    # for point sampling we use [B, J, 1, 2].
    grid = joints_normalized.unsqueeze(2)  # [B*T, J, 1, 2]
    
    sampled = F.grid_sample(
        features, 
        grid, 
        mode='bilinear', 
        padding_mode='zeros',  # out-of-range points are zero-filled
        align_corners=True
    )
    # sampled: [B*T, C, J, 1]
    
    sampled = sampled.squeeze(-1).permute(0, 2, 1)  # [B*T, J, C]
    
    return sampled


# ===================== VisionEncoder modules =====================

import torch.nn as nn


class ResBlock2d(nn.Module):
    """
    2D residual block for visual features in [B, C, T, J] layout.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        
        # A shortcut projection is needed when the channel count changes
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return self.act(h + self.shortcut(x))


class Downsample2d(nn.Module):
    """
    2D downsampling module with independent strides for the time and joint dimensions.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        time_stride: int = 2,
        joint_stride: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3,
            stride=(time_stride, joint_stride),
            padding=1
        )
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class VisionEncoder(nn.Module):
    """
    Visual feature encoder.
    
    Compresses HRNet features sampled at joint locations into a latent space
    aligned with the skeleton features.
    
    Input: [B, in_channels, T, J] (HRNet features sampled at joints)
    Output: [B, out_channels, T/downsample_rate, J] (compressed visual features)
    
    Design follows MTVCrafter's VisionEncoder, adapted to match MldVaeV2's
    temporal downsampling.
    """
    
    def __init__(
        self,
        in_channels: int = 256,        # HRNet stage-4 output channels
        mid_channels: int = 256,       # intermediate channels
        out_channels: int = 128,       # output channels (visual feature dimension)
        downsample_rate: int = 4,      # temporal downsampling rate (aligned with MldVaeV2)
        num_joints: int = 22,          # number of joints
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample_rate = downsample_rate
        self.num_joints = num_joints
        
        self.conv_in = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.norm_in = nn.GroupNorm(8, mid_channels)
        self.act = nn.SiLU()
        
        layers = []
        
        if downsample_rate == 1:
            # No downsampling; feature transformation only
            layers.append(ResBlock2d(mid_channels, mid_channels))
            layers.append(ResBlock2d(mid_channels, mid_channels))
        elif downsample_rate == 2:
            # 1 downsampling step
            layers.append(ResBlock2d(mid_channels, mid_channels))
            layers.append(Downsample2d(mid_channels, mid_channels, time_stride=2, joint_stride=1))
            layers.append(ResBlock2d(mid_channels, mid_channels))
        elif downsample_rate == 4:
            # 2 downsampling steps (most common)
            layers.append(ResBlock2d(mid_channels, mid_channels))
            layers.append(Downsample2d(mid_channels, mid_channels, time_stride=2, joint_stride=1))
            layers.append(ResBlock2d(mid_channels, mid_channels))
            layers.append(Downsample2d(mid_channels, mid_channels, time_stride=2, joint_stride=1))
            layers.append(ResBlock2d(mid_channels, mid_channels))
        elif downsample_rate == 8:
            # 3 downsampling steps
            layers.append(ResBlock2d(mid_channels, mid_channels))
            layers.append(Downsample2d(mid_channels, mid_channels, time_stride=2, joint_stride=1))
            layers.append(ResBlock2d(mid_channels, mid_channels))
            layers.append(Downsample2d(mid_channels, mid_channels, time_stride=2, joint_stride=1))
            layers.append(ResBlock2d(mid_channels, mid_channels))
            layers.append(Downsample2d(mid_channels, mid_channels, time_stride=2, joint_stride=1))
            layers.append(ResBlock2d(mid_channels, mid_channels))
        else:
            raise ValueError(f"Unsupported downsample_rate: {downsample_rate}. Use 1, 2, 4, or 8.")
        
        self.layers = nn.Sequential(*layers)
        
        self.conv_out = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.norm_out = nn.GroupNorm(8, out_channels)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, in_channels, T, J] HRNet features sampled at joints
        
        Returns:
            out: [B, out_channels, T/downsample_rate, J] compressed visual features
        """
        x = self.conv_in(x)
        x = self.norm_in(x)
        x = self.act(x)
        
        x = self.layers(x)
        
        x = self.conv_out(x)
        x = self.norm_out(x)
        x = self.act(x)
        
        return x


class VisionFeaturePooler(nn.Module):
    """
    Visual feature pooler.
    
    Pools visual features over the joint dimension into a global representation.
    
    Input: [B, C, T, J]
    Output: [B, T, C] (for concatenation with skeleton features)
    """
    
    def __init__(self, pool_type: str = 'mean'):
        """
        Args:
            pool_type: pooling type ('mean', 'max')
        """
        super().__init__()
        self.pool_type = pool_type
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, T, J] visual features
        
        Returns:
            out: [B, T, C] pooled features
        """
        if self.pool_type == 'mean':
            out = x.mean(dim=-1)  # [B, C, T]
        elif self.pool_type == 'max':
            out = x.max(dim=-1)[0]  # [B, C, T]
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")
        
        out = out.permute(0, 2, 1)  # [B, T, C]
        return out
