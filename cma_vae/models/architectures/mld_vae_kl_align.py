"""
CMA-VAE (Cross-Modal Aligned Motion VAE).

Paper: CMA-VAE, trained with DPA (Dual-Posterior KL Alignment).
Historical code names: "MLD VAE with KL Alignment" / `MldVaeKLAlign`
(JMVAE-KL style). The file name is kept for checkpoint compatibility.

Core design:
1. Three components are trained:
   - Motion Encoder (unimodal, used at inference)
   - Joint Encoder (fuses motion and image features during training)
   - A KL divergence term aligns the Motion Encoder with the Joint Encoder

2. Loss functions:
   - L_recon: reconstruction loss (uses z from the Joint Encoder)
   - L_KL: KL regularization (pushes z towards a standard normal)
   - L_align: KL alignment loss (Motion Encoder learns the Joint Encoder's distribution)

3. Inference:
   - Only the Motion Encoder and the Decoder are used
   - No image input is required
"""

import math
from functools import reduce
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.distribution import Distribution

try:
    from .tools.embeddings import TimestepEmbedding, Timesteps
    from ..operator import PositionalEncoding
    from ..operator.cross_attention import (
        SkipTransformerEncoder,
        SkipTransformerDecoder,
        TransformerDecoder,
        TransformerDecoderLayer,
        TransformerEncoder,
        TransformerEncoderLayer,
    )
    from ..operator.position_encoding import build_position_encoding
    from ...utils.temos_utils import lengths_to_mask
    from .vision_utils import (
        VisionEncoder, VisionFeaturePooler,
        recover_from_ric, get_joints_for_vision, sample_features_by_joints
    )
except ImportError:
    from models.architectures.tools.embeddings import TimestepEmbedding, Timesteps
    from models.operator import PositionalEncoding
    from models.operator.cross_attention import (
        SkipTransformerEncoder,
        SkipTransformerDecoder,
        TransformerDecoder,
        TransformerDecoderLayer,
        TransformerEncoder,
        TransformerEncoderLayer,
    )
    from models.operator.position_encoding import build_position_encoding
    from utils.temos_utils import lengths_to_mask
    from models.architectures.vision_utils import (
        VisionEncoder, VisionFeaturePooler,
        recover_from_ric, get_joints_for_vision, sample_features_by_joints
    )


# ===================== Helper modules =====================

class TemporalDownsample(nn.Module):
    """Temporal downsampling module."""
    def __init__(self, in_channels: int, out_channels: int, downsample_rate: int = 4):
        super().__init__()
        self.downsample_rate = downsample_rate
        
        if downsample_rate == 1:
            self.conv = nn.Identity()
        elif downsample_rate == 2:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, out_channels),
                nn.GELU(),
            )
        elif downsample_rate == 4:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, out_channels),
                nn.GELU(),
                nn.Conv1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, out_channels),
                nn.GELU(),
            )
        elif downsample_rate == 8:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, out_channels),
                nn.GELU(),
                nn.Conv1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, out_channels),
                nn.GELU(),
                nn.Conv1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, out_channels),
                nn.GELU(),
            )
        else:
            raise ValueError(f"Unsupported downsample_rate: {downsample_rate}")
    
    def forward(self, x: Tensor) -> Tensor:
        if self.downsample_rate == 1:
            return x
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        return x


class TemporalUpsample(nn.Module):
    """Temporal upsampling module."""
    def __init__(self, in_channels: int, out_channels: int, downsample_rate: int = 4):
        super().__init__()
        self.downsample_rate = downsample_rate
        
        if downsample_rate == 1:
            self.upsample = nn.Identity()
        elif downsample_rate == 2:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='linear', align_corners=True),
                nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, out_channels),
                nn.GELU(),
            )
        elif downsample_rate == 4:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='linear', align_corners=True),
                nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, in_channels),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode='linear', align_corners=True),
                nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, out_channels),
                nn.GELU(),
            )
        elif downsample_rate == 8:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='linear', align_corners=True),
                nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, in_channels),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode='linear', align_corners=True),
                nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, in_channels),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode='linear', align_corners=True),
                nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, out_channels),
                nn.GELU(),
            )
        else:
            raise ValueError(f"Unsupported downsample_rate: {downsample_rate}")
    
    def forward(self, x: Tensor) -> Tensor:
        if self.downsample_rate == 1:
            return x
        x = x.permute(0, 2, 1)
        x = self.upsample(x)
        x = x.permute(0, 2, 1)
        return x


# ===================== Main model: CMAVAE =====================

class CMAVAE(nn.Module):
    """
    CMA-VAE: Cross-Modal Aligned Motion VAE.

    Paper: CMA-VAE, trained with DPA (Dual-Posterior KL Alignment).
    During training, a KL alignment term (JMVAE-KL style) teaches the
    unimodal encoder to match the vision-fused joint encoder's distribution.

    Core components:
    1. Motion Encoder: q(z|motion) - used at inference
    2. Vision-Fused Motion Encoder ("Joint Encoder" in code): q(z|motion, image)
       - used during training, fuses visual information
    3. Motion Decoder: p(motion|z)
    4. Vision Feature Extractor: HRNet + joint-based sampling
    
    Training strategy:
    - Samples with images: reconstruct via the Joint Encoder + KL alignment loss
    - Samples without images: reconstruct via the Motion Encoder (standard VAE)
    
    Inference strategy:
    - Only the Motion Encoder and the Decoder are used
    """
    
    def __init__(self,
                 ablation,
                 nfeats: int,
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 downsample_rate: int = 4,
                 arch: str = "all_encoder",
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 position_embedding: str = "learned",
                 # ===== KL alignment parameters =====
                 kl_align_weight: float = 0.1,      # weight of the KL alignment loss
                 # ===== Vision guidance parameters =====
                 vision_dim: int = 128,
                 hrnet_channels: int = 256,
                 hrnet_checkpoint: str = None,
                 freeze_hrnet: bool = True,
                 num_joints: int = 22,
                 img_size: Tuple[int, int] = (256, 192),
                 **kwargs) -> None:
        
        super().__init__()
        
        # ==================== Basic parameters ====================
        self.nfeats = nfeats
        self.latent_dim_config = latent_dim
        self.latent_dim = latent_dim[-1]  # 256
        self.vision_dim = vision_dim      # 128
        self.downsample_rate = downsample_rate
        self.pe_type = ablation.PE_TYPE
        self.arch = arch
        self.num_joints = num_joints
        self.img_h, self.img_w = img_size
        self.hrnet_checkpoint = hrnet_checkpoint
        self.freeze_hrnet = freeze_hrnet
        self.kl_align_weight = kl_align_weight
        
        # ==================== Motion Encoder (unimodal) ====================
        # Used at inference; takes motion input only.
        
        self.motion_embedding = nn.Linear(nfeats, self.latent_dim)
        
        self.motion_downsample = TemporalDownsample(
            self.latent_dim, self.latent_dim, downsample_rate)
        
        if self.pe_type == "actor":
            self.motion_pos_encoder = PositionalEncoding(self.latent_dim, dropout)
        elif self.pe_type == "mld":
            self.motion_pos_encoder = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
        else:
            raise ValueError("Not Support PE type")
        
        motion_encoder_layer = TransformerEncoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        motion_encoder_norm = nn.LayerNorm(self.latent_dim)
        self.motion_encoder = SkipTransformerEncoder(
            motion_encoder_layer, num_layers, motion_encoder_norm)
        
        # Distribution prediction head of the Motion Encoder
        self.motion_dist_layer = nn.Linear(self.latent_dim, 2 * self.latent_dim)
        
        # ==================== Joint Encoder ====================
        # Used during training; fuses motion and image features.
        
        # Joint feature dimension = motion_dim + vision_dim
        self.joint_dim = self.latent_dim + self.vision_dim  # 256 + 128 = 384
        
        joint_encoder_layer = TransformerEncoderLayer(
            self.joint_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        joint_encoder_norm = nn.LayerNorm(self.joint_dim)
        self.joint_encoder = SkipTransformerEncoder(
            joint_encoder_layer, num_layers, joint_encoder_norm)
        
        # Distribution prediction head of the Joint Encoder
        self.joint_dist_layer = nn.Linear(self.joint_dim, 2 * self.latent_dim)
        
        # ==================== Vision branch ====================
        # HRNet backbone (lazily initialized)
        self.vision_backbone = None
        self.hrnet_channels = hrnet_channels
        
        self.vision_encoder = VisionEncoder(
            in_channels=hrnet_channels,
            mid_channels=256,
            out_channels=vision_dim,
            downsample_rate=downsample_rate,
            num_joints=num_joints,
        )
        
        self.vision_pooler = VisionFeaturePooler(pool_type='mean')
        
        # ==================== Decoder (shared) ====================
        # The decoder consumes z of size latent_dim (256).
        
        if self.pe_type == "actor":
            self.decoder_pos_encoder = PositionalEncoding(self.latent_dim, dropout)
        elif self.pe_type == "mld":
            self.decoder_pos_encoder = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
        
        if self.arch == "all_encoder":
            decoder_layer = TransformerEncoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = SkipTransformerEncoder(decoder_layer, num_layers, decoder_norm)
        elif self.arch == "encoder_decoder":
            decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = SkipTransformerDecoder(decoder_layer, num_layers, decoder_norm)
        else:
            raise ValueError("Not support architecture!")
        
        self.decoder_upsample = TemporalUpsample(
            self.latent_dim, self.latent_dim, downsample_rate)
        
        self.final_layer = nn.Linear(self.latent_dim, nfeats)
    
    def _init_vision_backbone(self):
        """Lazily initialize the HRNet backbone."""
        if self.vision_backbone is not None:
            return
        
        try:
            try:
                from .hrnet_wrapper import (
                    HRNetFeatureExtractor, create_hrnet_config
                )
            except ImportError:
                from models.architectures.hrnet_wrapper import (
                    HRNetFeatureExtractor, create_hrnet_config
                )
            
            hrnet_config = create_hrnet_config()
            checkpoint = self.hrnet_checkpoint
            if checkpoint is None:
                checkpoint = hrnet_config.model.backbone.checkpoint
            
            self.vision_backbone = HRNetFeatureExtractor(
                config=hrnet_config.model.backbone,
                checkpoint_path=checkpoint,
                freeze=self.freeze_hrnet,
                output_level=3,  # use the 4th stage (256 channels)
            )
            
            if hasattr(self, 'motion_embedding'):
                device = next(self.motion_embedding.parameters()).device
                self.vision_backbone = self.vision_backbone.to(device)
            
            print(f"Initialized HRNet backbone (freeze={self.freeze_hrnet})")
            
        except Exception as e:
            print(f"Warning: Failed to initialize HRNet: {e}")
            print("Vision guidance will not be available.")
            self.vision_backbone = None
    
    def _pad_to_multiple(self, x: Tensor, lengths: List[int]) -> Tuple[Tensor, int]:
        """Pad the input to a multiple of downsample_rate."""
        bs, seq_len, dim = x.shape
        remainder = seq_len % self.downsample_rate
        pad_len = (self.downsample_rate - remainder) if remainder != 0 else 0
        padded_len = seq_len + pad_len
        
        if pad_len > 0:
            padding = torch.zeros(bs, pad_len, dim, device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)
        
        return x, padded_len
    
    def _compute_latent_lengths(self, lengths: List[int], padded_len: int) -> Tuple[List[int], int]:
        """Compute valid lengths in the latent space."""
        latent_len = padded_len // self.downsample_rate
        latent_lengths = [math.ceil(l / self.downsample_rate) for l in lengths]
        return latent_lengths, latent_len
    
    def get_vision_feats(
        self,
        motion_features: Tensor,
        video_rgb: Tensor,
        camera_params: Optional[dict] = None,
        joints_2d_precomputed: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Extract vision features (following MTVCrafter, supports precomputed joints).
        
        Args:
            motion_features: [B, T, 269] motion features
            video_rgb: [B, T, H, W, 3] video frames
            camera_params: camera parameters (optional)
            joints_2d_precomputed: [B, T, 22, 2] precomputed pixel coordinates (recommended)
        
        Returns:
            vision_feats: [B, vision_dim, T/downsample_rate, num_joints]
        """
        B, T = motion_features.shape[:2]
        device = motion_features.device
        
        # Make sure HRNet is initialized
        if self.vision_backbone is None:
            self._init_vision_backbone()
        
        if self.vision_backbone is None:
            latent_t = T // self.downsample_rate
            if T % self.downsample_rate != 0:
                latent_t += 1
            return torch.zeros(B, self.vision_dim, latent_t, self.num_joints, 
                             device=device, dtype=motion_features.dtype)
        
        # ===== 1. Get 2D joint coordinates from motion (normalized to [-1, 1]) =====
        # Following MTVCrafter: joint3d_images[..., :2].unsqueeze(-2)
        # Precomputed joint coordinates take precedence when provided.
        joints_normalized = get_joints_for_vision(
            motion_features, 
            camera_params, 
            self.img_h, 
            self.img_w, 
            self.num_joints,
            joints_2d_precomputed=joints_2d_precomputed
        )
        # joints_normalized: [B, T, num_joints, 2]
        
        # ===== 2. Process video frames =====
        # video_rgb: [B, T, H, W, 3] -> [B*T, 3, H, W]
        video_rgb = video_rgb.permute(0, 1, 4, 2, 3).contiguous()  # [B, T, 3, H, W]
        images = video_rgb.reshape(-1, *video_rgb.shape[2:])  # [B*T, 3, H, W]
        
        # Normalize images to [0, 1]
        if images.max() > 1.0:
            images = images / 255.0
        
        # ===== 3. Extract image features with HRNet =====
        # Following MTVCrafter: image_feature_list = self.vision_backbone(images)
        # Returns either a list of multi-scale features or a single-scale feature.
        image_features = self.vision_backbone(images)
        
        # Wrap single-scale HRNet output in a list
        if not isinstance(image_features, (list, tuple)):
            image_features = [image_features]
        
        # ===== 4. Sample at joint locations (grid_sample) =====
        # Following MTVCrafter: grid_sample_ref = joint3d_images[..., :2].unsqueeze(-2)
        joints_flat = joints_normalized.reshape(B * T, self.num_joints, 2)  # [B*T, J, 2]
        grid_sample_ref = joints_flat.unsqueeze(-2)  # [B*T, J, 1, 2]
        
        # Sample features at each scale
        features_ref_list = []
        for features in image_features:
            # features: [BT, C, H, W]
            # grid_sample_ref: [BT, J, 1, 2]
            features_ref = F.grid_sample(features, grid_sample_ref, align_corners=True)
            # features_ref: [BT, C, J, 1]
            
            features_ref = features_ref.squeeze(-1).permute(0, 2, 1).contiguous()  # [BT, J, C]
            features_ref_list.append(features_ref)
        
        # ===== 5. Concatenate multi-scale features =====
        # Following MTVCrafter: video_ref_features = torch.cat(features_ref_list, dim=-1)
        video_ref_features = torch.cat(features_ref_list, dim=-1)  # [BT, J, C_total]
        video_ref_features = video_ref_features.reshape(B, T, self.num_joints, -1)  # [B, T, J, C_total]
        
        # Convert to [B, C_total, T, J] layout
        video_ref_features = video_ref_features.permute(0, 3, 1, 2).contiguous()
        # video_ref_features: [B, C_total, T, J]
        
        # ===== 6. Compress to the target dimension with VisionEncoder =====
        # Input: [B, C_total, T, J]
        # Output: [B, vision_dim, T/downsample_rate, J]
        vision_feats = self.vision_encoder(video_ref_features)
        
        return vision_feats
    
    def encode_motion(
        self,
        features: Tensor,
        lengths: Optional[List[int]] = None,
    ) -> Tuple[Tensor, Distribution]:
        """
        Motion Encoder: q(z|motion)
        
        Unimodal encoding, used at inference.
        
        Args:
            features: [B, T, nfeats] motion features
            lengths: actual length of each sample
        
        Returns:
            latent: [latent_len, B, latent_dim] latent codes
            dist: distribution object
        """
        if lengths is None:
            lengths = [features.shape[1]] * features.shape[0]
        
        bs, seq_len, _ = features.shape
        device = features.device
        
        # 1. Pad to a multiple of downsample_rate
        x = features
        x, padded_len = self._pad_to_multiple(x, lengths)
        
        # 2. Feature embedding
        x = self.motion_embedding(x)
        
        # 3. Temporal downsampling
        x = self.motion_downsample(x)
        
        # 4. Compute the latent mask
        latent_lengths, latent_len = self._compute_latent_lengths(lengths, padded_len)
        # NOTE:
        # Input features may already be padded by dataloader collate (e.g. unit_length alignment),
        # so mask length must match actual latent sequence length after downsample.
        latent_mask = lengths_to_mask(latent_lengths, device, max_len=x.shape[1])
        
        # 5. Convert to Transformer layout
        x = x.permute(1, 0, 2)  # [latent_len, B, latent_dim]
        
        # 6. Positional encoding
        x = self.motion_pos_encoder(x)
        
        # 7. SkipTransformer encoding
        x = self.motion_encoder(x, src_key_padding_mask=~latent_mask)
        
        # 8. Predict distribution parameters
        dist_params = self.motion_dist_layer(x)
        mu = dist_params[..., :self.latent_dim]
        logvar = dist_params[..., self.latent_dim:]
        
        # 9. Reparameterized sampling
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()
        
        # Zero out latents at padded positions
        latent = latent * latent_mask.T.unsqueeze(-1)
        
        return latent, dist
    
    def encode_joint(
        self,
        features: Tensor,
        lengths: Optional[List[int]] = None,
        video_rgb: Optional[Tensor] = None,
        camera_params: Optional[dict] = None,
        joints_2d_precomputed: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Distribution]:
        """
        Joint Encoder: q(z|motion, image)
        
        Joint encoding, used during training; fuses visual information.
        
        Args:
            features: [B, T, nfeats] motion features
            lengths: actual length of each sample
            video_rgb: [B, T, H, W, 3] video frames
            camera_params: camera parameters
            joints_2d_precomputed: [B, T, 22, 2] precomputed pixel coordinates (optional)
        
        Returns:
            latent: [latent_len, B, latent_dim] latent codes
            dist: distribution object
        """
        if lengths is None:
            lengths = [features.shape[1]] * features.shape[0]
        
        bs, seq_len, _ = features.shape
        device = features.device
        
        # ==================== Motion branch ====================
        # 1. Pad to a multiple of downsample_rate
        x = features
        x, padded_len = self._pad_to_multiple(x, lengths)
        
        # 2. Feature embedding
        x = self.motion_embedding(x)
        
        # 3. Temporal downsampling
        motion_feats = self.motion_downsample(x)
        # motion_feats: [B, latent_len, latent_dim]
        
        # 4. Compute the latent mask
        latent_lengths, latent_len = self._compute_latent_lengths(lengths, padded_len)
        # Keep mask length consistent with actual latent temporal length.
        latent_mask = lengths_to_mask(latent_lengths, device, max_len=motion_feats.shape[1])
        
        # ==================== Vision branch ====================
        if video_rgb is not None:
            # Extract vision features (a single frame is broadcast across the sequence)
            first_frame_motion = features[:, :1, :]
            first_frame_image = video_rgb
            
            # If precomputed joints are provided, also keep only the first frame
            first_frame_joints = None
            if joints_2d_precomputed is not None:
                first_frame_joints = joints_2d_precomputed[:, :1, :, :]  # [B, 1, 22, 2]
            
            vision_feats_single = self.get_vision_feats(
                first_frame_motion,
                first_frame_image,
                camera_params,
                joints_2d_precomputed=first_frame_joints
            )
            # vision_feats_single: [B, vision_dim, 1, J]
            
            # Pool over the joint dimension
            vision_feats_single = self.vision_pooler(vision_feats_single)
            # vision_feats_single: [B, 1, vision_dim]
            
            # Repeat across the whole sequence in feature space
            vision_feats = vision_feats_single.repeat(1, latent_len, 1)
            # vision_feats: [B, latent_len, vision_dim]
        else:
            # No video: zero out all vision features
            vision_feats = torch.zeros(
                bs, latent_len, self.vision_dim,
                device=device, dtype=motion_feats.dtype
            )
        
        # ==================== Feature fusion ====================
        # Concatenate motion and vision features
        fused_feats = torch.cat([motion_feats, vision_feats], dim=-1)
        # fused_feats: [B, latent_len, joint_dim]
        
        # Convert to Transformer layout
        fused_feats = fused_feats.permute(1, 0, 2)  # [latent_len, B, joint_dim]
        
        # Positional encoding (reuses the motion positional encoder)
        fused_feats_pos = self.motion_pos_encoder(fused_feats[..., :self.latent_dim])
        # Re-attach the vision part
        fused_feats = torch.cat([fused_feats_pos, fused_feats[..., self.latent_dim:]], dim=-1)
        
        # SkipTransformer encoding
        fused_feats = self.joint_encoder(fused_feats, src_key_padding_mask=~latent_mask)
        
        # Predict distribution parameters
        dist_params = self.joint_dist_layer(fused_feats)
        mu = dist_params[..., :self.latent_dim]
        logvar = dist_params[..., self.latent_dim:]
        
        # Reparameterized sampling
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()
        
        # Zero out latents at padded positions
        latent = latent * latent_mask.T.unsqueeze(-1)
        
        return latent, dist
    
    def decode(self, z: Tensor, lengths: List[int]) -> Tensor:
        """
        Decoder: p(motion|z)
        
        Args:
            z: [latent_len, B, latent_dim] latent codes
            lengths: list of original target lengths
        
        Returns:
            feats: [B, max_len, nfeats] reconstructed features
        """
        latent_len, bs, _ = z.shape
        device = z.device
        
        padded_len = latent_len * self.downsample_rate
        max_orig_len = max(lengths)
        
        # Compute the latent mask
        latent_lengths = [math.ceil(l / self.downsample_rate) for l in lengths]
        # z may include padded latent steps from batched collate; mask must match latent_len.
        latent_mask = lengths_to_mask(latent_lengths, device, max_len=latent_len)
        
        if self.arch == "all_encoder":
            z_pos = self.decoder_pos_encoder(z)
            
            output = self.decoder(z_pos, src_key_padding_mask=~latent_mask)
        
        elif self.arch == "encoder_decoder":
            queries = torch.zeros(latent_len, bs, self.latent_dim, device=device)
            queries = self.decoder_pos_encoder(queries)
            output = self.decoder(
                tgt=queries,
                memory=z,
                tgt_key_padding_mask=~latent_mask,
            ).squeeze(0)
        
        output = output.permute(1, 0, 2)
        
        # Temporal upsampling
        output = self.decoder_upsample(output)
        
        output = self.final_layer(output)
        
        # Slice to the original maximum length
        feats = output[:, :max_orig_len, :]
        
        # Mask out padded positions
        output_mask = lengths_to_mask(lengths, device)
        feats[~output_mask] = 0
        
        return feats
    
    def forward(
        self,
        features: Tensor,
        lengths: Optional[List[int]] = None,
        video_rgb: Optional[Tensor] = None,
        camera_params: Optional[dict] = None,
        has_video: Optional[Tensor] = None,
        joints_2d_precomputed: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Full forward pass.
        
        Args:
            features: [B, T, 269] motion features
            lengths: actual length of each sample
            video_rgb: [B, T, H, W, 3] video frames (optional)
            camera_params: camera parameters (optional)
            has_video: [B] bool, marks which samples have video
            joints_2d_precomputed: [B, T, 22, 2] precomputed pixel coordinates (optional, recommended)
        
        Returns:
            outputs: dict with the following keys
                - motion_recon: [B, T, 269] reconstructed motion
                - latent: [latent_len, B, latent_dim] latent codes
                - dist_motion: distribution from the Motion Encoder
                - dist_joint: distribution from the Joint Encoder
                - has_video: [B] bool, marks which samples have video
        """
        if lengths is None:
            lengths = [features.shape[1]] * features.shape[0]
        
        bs = features.shape[0]
        device = features.device
        
        # ==================== Encoding ====================
        # 1. Motion Encoder (always runs)
        latent_motion, dist_motion = self.encode_motion(features, lengths)
        
        # 2. Joint Encoder (only runs for samples with images)
        latent_joint = None
        dist_joint = None
        
        if has_video is not None and has_video.any():
            latent_joint, dist_joint = self.encode_joint(
                features, lengths, video_rgb, camera_params,
                joints_2d_precomputed=joints_2d_precomputed
            )
        
        # ==================== Decoding ====================
        # Training strategy:
        #   - Samples with images: decode from the Joint Encoder latent (forces visual guidance)
        #   - Samples without images: decode from the Motion Encoder latent
        # Inference strategy: always use the Motion Encoder (images may be unavailable)
        
        if self.training and latent_joint is not None:
            # During training, mix latents: joint latent for samples with video,
            # motion latent otherwise.
            latent_for_decode = latent_motion.clone()
            has_video_indices = torch.where(has_video)[0]
            latent_for_decode[:, has_video_indices, :] = latent_joint[:, has_video_indices, :]
            
            motion_recon = self.decode(latent_for_decode, lengths)
        else:
            # At inference, or when no sample has video, use the motion latent
            motion_recon = self.decode(latent_motion, lengths)
        
        return {
            'motion_recon': motion_recon,
            'latent': latent_joint if (self.training and latent_joint is not None) else latent_motion,
            'latent_motion': latent_motion,
            'latent_joint': latent_joint,  # None when no image is available
            'dist_motion': dist_motion,
            'dist_joint': dist_joint,  # None when no image is available (KL alignment loss skipped)
            'has_video': has_video if has_video is not None else torch.zeros(bs, dtype=torch.bool, device=device),
        }
    
    def get_latent_shape(self, seq_len: int) -> int:
        """Compute the latent length for a given input sequence length."""
        remainder = seq_len % self.downsample_rate
        if remainder != 0:
            padded_len = seq_len + (self.downsample_rate - remainder)
        else:
            padded_len = seq_len
        return padded_len // self.downsample_rate
    
    @staticmethod
    def kl_between_gaussians(mu1: Tensor, logvar1: Tensor, 
                            mu2: Tensor, logvar2: Tensor) -> Tensor:
        """
        Compute the KL divergence between two Gaussian distributions.
        
        KL(N(mu1, var1) || N(mu2, var2))
        
        Args:
            mu1, logvar1: parameters of the first distribution
            mu2, logvar2: parameters of the second distribution
        
        Returns:
            kl: KL divergence
        """
        return 0.5 * torch.mean(
            logvar2 - logvar1 + 
            (logvar1.exp() + (mu1 - mu2).pow(2)) / logvar2.exp() - 1
        )


# Backward-compatibility alias: historical checkpoints (and their pickled
# `hyper_parameters`) may reference the original class name. Do not remove.
MldVaeKLAlign = CMAVAE
