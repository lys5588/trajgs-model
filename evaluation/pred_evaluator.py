# coding=utf-8
# Copyright 2026 UniMotion Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Motion Prediction Evaluation Module for UniMotion.
Fully consistent with the MotionGPT/MotionGPT3 evaluation protocol; supports the following metrics:
- ADE (Average Displacement Error)
- FDE (Final Displacement Error)
- APD (Average Pairwise Distance) - used to evaluate diversity

Motion Prediction task:
- Input: the first 20% of the motion
- Output: full motion (source concatenated with the predicted target)
- Evaluation: compare the generated last 80% with the GT last 80% (in meters)

Key consistency requirements (aligned with MotionGPT3):
1. Pass the full motion (source+target concatenated); the evaluator skips the source part according to predict_ratio
2. Inputs must be denormalized to raw space (meters) beforehand
3. Use recover_from_ric to recover joints from HumanML3D features (includes root trajectory integration and rotation)
"""

import numpy as np
import torch
from typing import List, Dict, Optional


# ==================== Quaternion Utilities (from HumanML3D) ====================

def qinv(q):
    """Invert quaternions."""
    assert q.shape[-1] == 4
    mask = torch.tensor([1, -1, -1, -1], dtype=q.dtype, device=q.device)
    return q * mask


def qrot(q, v):
    """
    Rotate vectors by quaternions.
    q: [..., 4] quaternions (w, x, y, z)
    v: [..., 3] vectors
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return v + 2 * (q[..., :1] * uv + uuv)


# ==================== HumanML3D Features -> Joints (identical to MotionGPT) ====================

def recover_root_rot_pos(data):
    """
    Recover the root rotation and position from HumanML3D features.
    Copied verbatim from MotionGPT: mGPT/data/humanml/scripts/motion_process.py
    
    First 4 dims of HumanML3D features:
    - dim 0: root rotation velocity (around the Y axis)
    - dim 1-2: root linear velocity (in the XZ plane)
    - dim 3: root height (along the Y axis)
    
    Args:
        data: [..., 263] denormalized HumanML3D features
    Returns:
        r_rot_quat: [..., 4] root rotation quaternions
        r_pos: [..., 3] root positions
    """
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    # Integrate angular velocity to get rotation angles
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    # Convert rotation angles to quaternions (rotation around the Y axis)
    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    # Integrate linear velocity to get the root position
    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    # Apply the Y-axis rotation to the linear velocity
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    # Integrate to get absolute positions
    r_pos = torch.cumsum(r_pos, dim=-2)
    # The Y position comes directly from dim 3
    r_pos[..., 1] = data[..., 3]

    return r_rot_quat, r_pos


def recover_from_ric(data, joints_num):
    """
    Recover global joint coordinates from HumanML3D ric (relative in coordinate) features.
    Copied verbatim from MotionGPT: mGPT/data/humanml/scripts/motion_process.py
    
    Args:
        data: [..., seq_len, 263] denormalized HumanML3D features
        joints_num: number of joints (22 for HumanML3D)
    Returns:
        [..., seq_len, joints_num, 3] global joint coordinates (meters)
    """
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    
    # Extract ric_data: joint positions relative to the root
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    # Apply the inverse root rotation to local joint positions -> relative positions in the global frame
    positions = qrot(
        qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)),
        positions
    )

    # Add the root XZ position to all joints
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    # Concatenate the root position and the other joints
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


def feats2joints_humanml3d(feats: torch.Tensor, njoints: int = 22,
                           mean: torch.Tensor = None, std: torch.Tensor = None) -> torch.Tensor:
    """
    Convert HumanML3D features to joint coordinates (identical to MotionGPT).
    
    Pipeline:
    1. Denormalize: feats * std + mean (if mean/std are provided)
    2. Use recover_from_ric to recover global joint coordinates
    
    Args:
        feats: [bs, seq_len, 263+] HumanML3D features (normalized or raw space)
        njoints: number of joints (default 22)
        mean: [263] normalization mean. If None, feats are assumed to be already denormalized
        std: [263] normalization std. If None, feats are assumed to be already denormalized
    Returns:
        [bs, seq_len, njoints, 3] global joint coordinates
    """
    # Handle 2-D input
    input_is_2d = feats.dim() == 2
    if input_is_2d:
        feats = feats.unsqueeze(0)
    
    # Truncate to 263 dims (compatible with 269-dim input)
    if feats.shape[-1] > 263:
        feats = feats[..., :263]
    
    # Denormalize (consistent with MotionGPT3's feats2joints)
    if mean is not None and std is not None:
        mean = mean.to(feats.device).to(feats.dtype)
        std = std.to(feats.device).to(feats.dtype)
        # Make sure mean/std are truncated to 263 dims as well
        if mean.shape[-1] > 263:
            mean = mean[..., :263]
        if std.shape[-1] > 263:
            std = std[..., :263]
        feats = feats * std + mean
    
    # Use recover_from_ric to recover joints (identical to MotionGPT)
    joints = recover_from_ric(feats, njoints)
    
    if input_is_2d:
        joints = joints.squeeze(0)
    
    return joints


# ==================== PredEvaluator ====================

class PredEvaluator:
    """
    Evaluator for the Motion Prediction task.
    Identical to PredMetrics in MotionGPT/MotionGPT3.
    
    Important: update() should receive the full motion (source + predicted/GT target);
    The evaluator skips the source part internally according to predict_ratio.
    This is fully aligned with the MotionGPT3 evaluation protocol.
    
    Evaluation metrics:
    - ADE: Average Displacement Error (meters)
    - FDE: Final Displacement Error (meters)
    - APD: Average Pairwise Distance (used for diversity evaluation)
    """
    
    def __init__(
        self,
        njoints: int = 22,
        predict_ratio: float = 0.2,
        task: str = 'pred',
        force_in_meter: bool = True,
        device: str = 'cuda',
    ):
        """
        Args:
            njoints: number of joints
            predict_ratio: input fraction (leading fraction used as the condition)
            task: 'pred' or 'inbetween'
            force_in_meter: whether to force meters as the unit
            device: compute device
        """
        self.njoints = njoints
        self.predict_ratio = predict_ratio
        self.task = task
        self.force_in_meter = force_in_meter
        self.device = device
        
        self.reset()
    
    def reset(self):
        """Reset storage."""
        self.count = 0
        self.count_seq = 0
        self.ADE_sum = 0.0
        self.FDE_sum = 0.0
        self.APD_sum = 0.0
    
    def update(
        self,
        joints_rst: torch.Tensor,
        joints_ref: torch.Tensor,
        lengths: List[int],
        inbetween_ratio: float = 0.25,
    ):
        """
        Update evaluation data.
        
        Important: joints_rst and joints_ref should be the full motion (including source and target parts);
        The evaluator automatically skips the source part according to predict_ratio during evaluation.
        Identical to PredMetrics.update in MotionGPT/MotionGPT3.
        
        Inputs must be denormalized and converted to joints via recover_from_ric beforehand.
        
        Args:
            joints_rst: [bs, seq_len, njoints, 3] generated full-motion joints (source_GT + predicted_target)
            joints_ref: [bs, seq_len, njoints, 3] GT full-motion joints
            lengths: [bs] list of full-motion lengths
            inbetween_ratio: head/tail fraction for the inbetween task
        """
        assert joints_rst.shape == joints_ref.shape
        assert joints_rst.dim() == 4  # (bs, seq, njoints, 3)
        
        self.count += sum(lengths)
        self.count_seq += len(lengths)
        
        # Flatten joints: [bs, seq, njoints, 3] -> [bs, seq, njoints*3]
        rst = torch.flatten(joints_rst, start_dim=2)
        ref = torch.flatten(joints_ref, start_dim=2)
        
        for i, l in enumerate(lengths):
            if self.task == "pred":
                # Prediction: skip the source (first predict_ratio) and evaluate the target (last 1-predict_ratio)
                # Identical to MotionGPT3 PredMetrics
                pred_start = int(l * self.predict_ratio)
                diff = rst[i, pred_start:l] - ref[i, pred_start:l]
            elif self.task == "inbetween":
                inbetween_start = int(l * inbetween_ratio)
                inbetween_end = l - int(l * inbetween_ratio)
                diff = rst[i, inbetween_start:inbetween_end] - ref[i, inbetween_start:inbetween_end]
            else:
                print(f"Task {self.task} not implemented.")
                diff = rst[i, :l] - ref[i, :l]
            
            # Compute distances: [seq_len, njoints*3] -> [seq_len]
            # Consistent with MotionGPT3: L2 norm over all joint dimensions
            dist = torch.linalg.norm(diff, dim=-1)
            
            # ADE: average displacement error (consistent with MotionGPT3: dist.mean(dim=1), then [None], then mean)
            ade = dist.mean()
            
            # FDE: final-frame displacement error
            fde = dist[-1] if len(dist) > 0 else torch.tensor(0.0)
            
            self.ADE_sum += ade.item()
            self.FDE_sum += fde.item()
    
    def update_diversity(
        self,
        joints_list: List[torch.Tensor],
        lengths: List[int],
    ):
        """
        Update diversity evaluation data (APD).
        
        Args:
            joints_list: list of [seq_len, njoints, 3] tensors
            lengths: list of corresponding motion lengths
        """
        if len(joints_list) < 2:
            return
        
        min_len = min(len(j) for j in joints_list)
        flattened = [torch.flatten(j[:min_len], start_dim=1).cpu().numpy() 
                     for j in joints_list]
        
        n = len(flattened)
        total_dist = 0.0
        pair_count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(flattened[i] - flattened[j], axis=-1).mean()
                total_dist += dist
                pair_count += 1
        
        if pair_count > 0:
            self.APD_sum += total_dist / pair_count
    
    def compute(self) -> Dict[str, float]:
        """Compute all evaluation metrics."""
        metrics = {}
        
        if self.count_seq > 0:
            metrics['ADE'] = self.ADE_sum / self.count_seq
            metrics['FDE'] = self.FDE_sum / self.count_seq
            metrics['APD'] = self.APD_sum / max(self.count_seq, 1)
        else:
            metrics['ADE'] = 0.0
            metrics['FDE'] = 0.0
            metrics['APD'] = 0.0
        
        self.reset()
        
        return metrics


def load_pred_evaluator(
    njoints: int = 22,
    predict_ratio: float = 0.2,
    task: str = 'pred',
    device: str = 'cuda',
) -> PredEvaluator:
    """Convenience function: load the Pred evaluator."""
    return PredEvaluator(
        njoints=njoints,
        predict_ratio=predict_ratio,
        task=task,
        device=device,
    )
