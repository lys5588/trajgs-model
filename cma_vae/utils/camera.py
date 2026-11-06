# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# Modifications for TrajGS: dual camera representations, explicit conventions,
# column-based 6D rotations, validation, per-view image sizes and FP32 codecs.
# Licensed under the Apache License, Version 2.0. See cma_vae/LICENSE.
# Quaternion conversion is adapted from Depth Anything 3 model/utils/transform.py.
"""Differentiable camera representation codecs, independent of neural encoders.

9D:  [C2W center(3), quaternion XYZW(4), vertical/horizontal FOV(2)].
11D: [C2W center(3), rotation column 1(3), column 2(3), FOV(2)].

The public codecs accept/return W2C extrinsics by default. Select
``extrinsics_type="c2w"`` to match DA3's low-level transform helpers.
Input extrinsics: [B,S,3,4] or [B,S,4,4]; decoded extrinsics: [B,S,3,4].
Intrinsics: [B,S,3,3]. Geometry and outputs are FP32 on the input device,
including under autocast. Casts preserve autograd. No reference/scale
normalization, resizing, temporal resampling or model layers occur here.

Only fx/fy are encoded. Decoding reconstructs a centered principal point and
zero skew, not arbitrary original intrinsics. image_size_hw describes the
valid image extent corresponding to the intrinsics, excluding extra padding.
"""

import math
from typing import Literal, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn.functional as F

PoseEncodingType = Literal["quaternion_9d", "rotation_6d_11d"]
ExtrinsicsType = Literal["w2c", "c2w"]
ImageSize = Union[Tuple[int, int], Tensor]
_ENCODING_DIMS = {"quaternion_9d": 9, "rotation_6d_11d": 11}
_EPS = 1e-6

__all__ = [
    "extri_intri_to_pose_encoding", "pose_encoding_to_extri_intri",
    "matrix_to_rotation_6d", "rotation_6d_to_matrix",
]


def _require(condition, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _float_tensor(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point torch.Tensor")
    value = value.float()
    _require(torch.isfinite(value).all(), f"{name} must contain finite FP32 values")
    return value


def _encoding_dim(pose_encoding_type: str) -> int:
    if pose_encoding_type not in _ENCODING_DIMS:
        raise ValueError(f"pose_encoding_type must be one of {tuple(_ENCODING_DIMS)}")
    return _ENCODING_DIMS[pose_encoding_type]


def _check_convention(extrinsics_type: str) -> None:
    if extrinsics_type not in ("w2c", "c2w"):
        raise ValueError("extrinsics_type must be 'w2c' or 'c2w'")


def _image_hw(image_size_hw: ImageSize, batch: int, steps: int, device):
    if image_size_hw is None:
        raise ValueError("image_size_hw=(H,W) is required to convert FOV and focal length")
    hw = torch.as_tensor(image_size_hw, dtype=torch.float32, device=device)
    if hw.shape == (2,):
        hw = hw.expand(batch, steps, 2)
    elif hw.shape == (batch, 2):
        hw = hw[:, None, :].expand(batch, steps, 2)
    elif hw.shape != (batch, steps, 2):
        raise ValueError("image_size_hw must have shape [2], [B,2] or [B,S,2], ordered (H,W)")
    _require(torch.isfinite(hw).all() & (hw > 0).all(), "image sizes must be finite and positive")
    return hw[..., 0], hw[..., 1]


def _check_rotation(rotation: Tensor) -> None:
    eye = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    # Accommodate rounding from half/bfloat16 inputs; do not silently repair a
    # scale, reflection or arbitrary non-rigid matrix using a rigid inverse.
    _require(torch.allclose(rotation.mT @ rotation, eye.expand_as(rotation), atol=1e-2, rtol=0),
             "extrinsics rotation must be orthonormal")
    _require(torch.allclose(torch.linalg.det(rotation), torch.ones_like(rotation[..., 0, 0]),
                            atol=1e-2, rtol=0), "extrinsics rotation must have determinant +1")


def matrix_to_rotation_6d(matrix: Tensor) -> Tensor:
    """Pack the FIRST TWO COLUMNS of [...,3,3] into [...,6].

    Column convention from the supplied G2VLM 11D analysis. This differs from
    row-based 6D helpers: [R00,R10,R20,R01,R11,R21].
    """
    matrix = _float_tensor(matrix, "matrix")
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("matrix must end in [3,3]")
    return matrix[..., :, :2].transpose(-1, -2).reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(rotation_6d: Tensor) -> Tensor:
    """Gram-Schmidt construct [...,3,3] with orthonormal COLUMNS.

    Zero/near-zero first vectors and collinear second vectors are undefined
    and rejected. Nondegenerate network outputs need not be orthogonal.
    """
    rotation_6d = _float_tensor(rotation_6d, "rotation_6d")
    if rotation_6d.ndim < 1 or rotation_6d.shape[-1] != 6:
        raise ValueError("rotation_6d must end in 6")
    a1, a2 = rotation_6d[..., :3], rotation_6d[..., 3:]
    norm1 = torch.linalg.vector_norm(a1, dim=-1, keepdim=True)
    _require(torch.isfinite(norm1).all() & (norm1 > _EPS).all(),
             "rotation_6d first vector is degenerate")
    b1 = a1 / norm1
    # Normalize a2 before projection: a large collinear vector must not
    # acquire a spurious residual direction from FP32 cancellation.
    norm_a2 = torch.linalg.vector_norm(a2, dim=-1, keepdim=True)
    _require(torch.isfinite(norm_a2).all() & (norm_a2 > _EPS).all(),
             "rotation_6d second vector is degenerate")
    a2 = a2 / norm_a2
    u2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    norm2 = torch.linalg.vector_norm(u2, dim=-1, keepdim=True)
    _require(torch.isfinite(norm2).all() & (norm2 > _EPS).all(),
             "rotation_6d vectors are collinear or degenerate")
    b2 = u2 / norm2
    b3 = F.normalize(torch.cross(b1, b2, dim=-1), dim=-1)
    # Remove the remaining roundoff component along b1 near collinearity.
    b2 = torch.cross(b3, b1, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def _sqrt_positive_part(x: Tensor) -> Tensor:
    # Avoid sqrt(0)'s infinite gradient in inactive quaternion candidates.
    result = torch.zeros_like(x)
    positive = x > 0
    result[positive] = torch.sqrt(x[positive])
    return result


def _matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """DA3 convention: XYZW, standardized to nonnegative W."""
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = matrix.flatten(-2).unbind(-1)
    q_abs = _sqrt_positive_part(torch.stack((
        1 + m00 + m11 + m22, 1 + m00 - m11 - m22,
        1 - m00 + m11 - m22, 1 - m00 - m11 + m22,
    ), dim=-1))
    candidates = torch.stack((
        torch.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), -1),
        torch.stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), -1),
        torch.stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21), -1),
        torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2), -1),
    ), dim=-2)
    candidates = candidates / (2 * q_abs[..., None].clamp_min(0.1))
    selected = candidates[F.one_hot(q_abs.argmax(-1), num_classes=4).bool()]
    quaternion = selected.reshape(*matrix.shape[:-2], 4)[..., [1, 2, 3, 0]]
    return torch.where(quaternion[..., 3:4] < 0, -quaternion, quaternion)


def _quaternion_to_matrix(quaternion: Tensor) -> Tensor:
    # Rescale before squaring so finite large predictions need not overflow.
    scale = quaternion.abs().amax(dim=-1, keepdim=True)
    _require((scale > _EPS).all(), "quaternion is zero or degenerate")
    quaternion = quaternion / scale
    x, y, z, w = quaternion.unbind(-1)
    two_s = 2 / quaternion.square().sum(-1)
    return torch.stack((
        1 - two_s * (y*y + z*z), two_s * (x*y - z*w), two_s * (x*z + y*w),
        two_s * (x*y + z*w), 1 - two_s * (x*x + z*z), two_s * (y*z - x*w),
        two_s * (x*z - y*w), two_s * (y*z + x*w), 1 - two_s * (x*x + y*y),
    ), dim=-1).reshape(*quaternion.shape[:-1], 3, 3)


def extri_intri_to_pose_encoding(
    extrinsics: Tensor,
    intrinsics: Tensor,
    image_size_hw: Optional[ImageSize] = None,
    *,
    pose_encoding_type: PoseEncodingType = "quaternion_9d",
    extrinsics_type: ExtrinsicsType = "w2c",
) -> Tensor:
    """Encode cameras into [B,S,9] or [B,S,11] float32 pose_encoding.

    extrinsics: [B,S,3,4] or [B,S,4,4], W2C by default.
    intrinsics: [B,S,3,3], in pixels, on the same device.
    image_size_hw: (H,W), or tensor [2], [B,2], [B,S,2].

    Like DA3 CameraEnc, W2C is inverted before parameter packing. Unlike
    DA3's low-level function of the same name (which expects C2W), this
    wrapper accepts raw W2C by default. Explicit c2w mode matches that helper.
    """
    _encoding_dim(pose_encoding_type)
    _check_convention(extrinsics_type)
    ext = _float_tensor(extrinsics, "extrinsics")
    k = _float_tensor(intrinsics, "intrinsics")
    if ext.ndim != 4 or ext.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError("extrinsics must have shape [B,S,3,4] or [B,S,4,4]")
    batch, steps = ext.shape[:2]
    if batch == 0 or steps == 0 or k.shape != (batch, steps, 3, 3):
        raise ValueError("intrinsics must have shape [B,S,3,3] with matching nonempty B,S")
    if ext.device != k.device:
        raise ValueError("extrinsics and intrinsics must be on the same device")
    with torch.autocast(device_type=ext.device.type, enabled=False):
        if ext.shape[-2] == 4:
            bottom = ext.new_tensor([0, 0, 0, 1]).expand(batch, steps, 4)
            _require(torch.allclose(ext[..., 3, :], bottom, atol=1e-6, rtol=0),
                     "homogeneous extrinsics must end in [0,0,0,1]")
        rotation, translation = ext[..., :3, :3], ext[..., :3, 3]
        _check_rotation(rotation)
        _require((k[..., 0, 0] > 0).all() & (k[..., 1, 1] > 0).all(),
                 "fx and fy must be positive")
        if extrinsics_type == "w2c":
            rotation = rotation.mT
            translation = -(rotation @ translation[..., None]).squeeze(-1)
        if pose_encoding_type == "quaternion_9d":
            orientation = _matrix_to_quaternion(rotation)
        else:
            orientation = matrix_to_rotation_6d(rotation)
        h, w = _image_hw(image_size_hw, batch, steps, ext.device)
        fov = torch.stack((2 * torch.atan((h / 2) / k[..., 1, 1]),
                           2 * torch.atan((w / 2) / k[..., 0, 0])), dim=-1)
        _require(((fov > 0) & (fov < math.pi)).all(),
                 "focal length and image size produce FOV outside the FP32 open interval (0,pi)")
        encoded = torch.cat((translation, orientation, fov), dim=-1)
        _require(torch.isfinite(encoded).all(), "pose_encoding overflowed FP32")
        return encoded


def pose_encoding_to_extri_intri(
    pose_encoding: Tensor,
    image_size_hw: Optional[ImageSize] = None,
    *,
    pose_encoding_type: Optional[PoseEncodingType] = None,
    extrinsics_type: ExtrinsicsType = "w2c",
) -> Tuple[Tensor, Tensor]:
    """Decode to (extrinsics [B,S,3,4], intrinsics [B,S,3,3]), float32.

    Format is inferred from the final dimension if pose_encoding_type is
    omitted. Extrinsics default to W2C; use c2w for DA3 low-level outputs.
    Both representations internally store C2W center and orientation.
    FOV must already be in radians in (0,pi), not logits. No sigmoid/clipping
    is applied. K has fx/fy recovered at image_size_hw, centered cx/cy and
    zero skew. Rotation normalization is many-to-one for network outputs.
    """
    _check_convention(extrinsics_type)
    pose = _float_tensor(pose_encoding, "pose_encoding")
    if pose.ndim != 3 or pose.shape[-1] not in (9, 11) or 0 in pose.shape[:2]:
        raise ValueError("pose_encoding must have nonempty shape [B,S,9] or [B,S,11]")
    if pose_encoding_type is None:
        pose_encoding_type = "quaternion_9d" if pose.shape[-1] == 9 else "rotation_6d_11d"
    if _encoding_dim(pose_encoding_type) != pose.shape[-1]:
        raise ValueError("pose_encoding_type does not match pose_encoding's final dimension")
    with torch.autocast(device_type=pose.device.type, enabled=False):
        batch, steps = pose.shape[:2]
        center, fov = pose[..., :3], pose[..., -2:]
        _require(((fov > 0) & (fov < math.pi)).all(), "FOV must be in radians strictly between 0 and pi")
        if pose_encoding_type == "quaternion_9d":
            rotation = _quaternion_to_matrix(pose[..., 3:7])
        else:
            rotation = rotation_6d_to_matrix(pose[..., 3:9])
        translation = center
        if extrinsics_type == "w2c":
            rotation = rotation.mT
            translation = -(rotation @ center[..., None]).squeeze(-1)
        ext = torch.cat((rotation, translation[..., None]), dim=-1)
        h, w = _image_hw(image_size_hw, batch, steps, pose.device)
        fy = (h / 2) / torch.tan(fov[..., 0] / 2)
        fx = (w / 2) / torch.tan(fov[..., 1] / 2)
        zero, one = torch.zeros_like(fx), torch.ones_like(fx)
        k = torch.stack((fx, zero, w/2, zero, fy, h/2, zero, zero, one), -1).reshape(batch, steps, 3, 3)
        _require(torch.isfinite(ext).all() & torch.isfinite(k).all(), "decoded cameras overflowed FP32")
        _require((fx > 0).all() & (fy > 0).all(), "decoded focal lengths must be positive")
        return ext, k
