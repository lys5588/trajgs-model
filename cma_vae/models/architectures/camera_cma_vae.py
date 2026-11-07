"""Camera CMA-VAE: 9D/11D, twenty time steps, and paired bbox images.

This module has no dependency on the original human-motion loader or loss.
The decoder retains the baseline all_encoder topology, but predicts raw
camera vectors. Rotation/FOV constraints and training losses are out of scope.
"""

from typing import Optional

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from ..operator.cross_attention import SkipTransformerEncoder, TransformerEncoderLayer
from ..operator.position_encoding import PositionEmbeddingLearned1D
from .camera_vision import (
    DEFAULT_DINOV3_REPO, DEFAULT_DINOV3_WEIGHTS, FrozenDINOv3, crop_bbox_images,
)


class CameraCMAVAE(nn.Module):
    """Batch-first inputs; sequence-first posterior and latent outputs.

    Images and pixel-xyxy boxes use full-batch layouts [B,20,3,H,W] and
    [B,20,4]. With has_images[B], inactive rows may contain placeholders;
    active rows must provide all twenty valid frames and boxes.
    """

    def __init__(
        self, camera_dim: int = 11, sequence_length: int = 20,
        downsample_rate: int = 1, latent_dim: int = 256, vision_dim: int = 256,
        camera_ff_size: int = 1024, joint_ff_size: int = 2048,
        num_layers: int = 9, num_heads: int = 4, dropout: float = 0.1,
        crop_size: int = 224, freeze_dinov3: bool = True,
        dinov3_repo=DEFAULT_DINOV3_REPO, dinov3_weights=DEFAULT_DINOV3_WEIGHTS,
        dinov3_chunk_size: int = 20, vision_backbone: Optional[nn.Module] = None,
    ):
        super().__init__()
        if camera_dim not in (9, 11):
            raise ValueError("camera_dim must be 9 or 11")
        if sequence_length != 20 or downsample_rate != 1:
            raise ValueError("CameraCMAVAE requires sequence_length=20 and downsample_rate=1")
        if crop_size != 224 or freeze_dinov3 is not True:
            raise ValueError("This version requires crop_size=224 and freeze_dinov3=True")
        if (latent_dim, vision_dim, camera_ff_size, joint_ff_size, num_layers, num_heads) != (
                256, 256, 1024, 2048, 9, 4):
            raise ValueError("Use the agreed 256/512-channel, 9-layer, 4-head camera architecture")
        self.camera_dim = camera_dim
        self.pose_encoding_type = "quaternion_9d" if camera_dim == 9 else "rotation_6d_11d"
        self.sequence_length = sequence_length
        self.downsample_rate = downsample_rate
        self.latent_dim = latent_dim
        self.vision_dim = vision_dim
        self.joint_dim = latent_dim + vision_dim

        def encoder(width, ff_size):
            layer = TransformerEncoderLayer(
                width, num_heads, ff_size, dropout,
                activation="gelu", normalize_before=False,
            )
            return SkipTransformerEncoder(layer, num_layers, nn.LayerNorm(width))

        self.camera_embedding = nn.Linear(camera_dim, latent_dim)
        self.camera_downsample = nn.Identity()
        self.camera_pos_encoder = PositionEmbeddingLearned1D(latent_dim)
        self.camera_encoder = encoder(latent_dim, camera_ff_size)
        self.camera_dist_layer = nn.Linear(latent_dim, 2 * latent_dim)

        self.vision_backbone = FrozenDINOv3(
            dinov3_repo, dinov3_weights, dinov3_chunk_size, vision_backbone,
        )
        self.vision_projection = nn.Linear(768, vision_dim)
        self.joint_encoder = encoder(self.joint_dim, joint_ff_size)
        self.joint_dist_layer = nn.Linear(self.joint_dim, 2 * latent_dim)

        # Same all_encoder decoder topology as CMAVAE with r=1. Only nfeats
        # becomes camera_dim; no nonlinear camera-specific output head.
        self.decoder_pos_encoder = PositionEmbeddingLearned1D(latent_dim)
        self.decoder = encoder(latent_dim, camera_ff_size)
        self.decoder_upsample = nn.Identity()
        self.final_layer = nn.Linear(latent_dim, camera_dim)

    def _camera_tokens(self, camera_features: Tensor) -> Tensor:
        if (not isinstance(camera_features, Tensor) or camera_features.ndim != 3
                or camera_features.shape[0] < 1
                or camera_features.shape[1:] != (20, self.camera_dim)):
            raise ValueError(f"camera_features must have shape [B,20,{self.camera_dim}]")
        if not camera_features.is_floating_point():
            raise TypeError("camera_features must be a floating point tensor")
        if not torch.isfinite(camera_features).all():
            raise ValueError("camera_features must contain finite values")
        features = camera_features.to(dtype=self.camera_embedding.weight.dtype)
        embedded = self.camera_downsample(self.camera_embedding(features))
        return self.camera_pos_encoder(embedded.transpose(0, 1))

    def _posterior(self, params: Tensor):
        mu, logvar = params.chunk(2, dim=-1)
        # Distribution construction in FP32 also keeps autocast exp stable.
        dist = Normal(mu.float(), torch.exp(0.5 * logvar.float()))
        return dist.rsample(), dist

    def encode_camera(self, camera_features: Tensor):
        """Return z and q_camera, each shaped [20,B,256]."""
        tokens = self._camera_tokens(camera_features)
        return self._posterior(self.camera_dist_layer(self.camera_encoder(tokens)))

    def _encode_joint_tokens(self, tokens: Tensor, images: Tensor, bboxes: Tensor):
        crops = crop_bbox_images(images, bboxes)
        if crops.shape[0] != tokens.shape[1]:
            raise ValueError("camera_features, images and bboxes must have the same batch size")
        cls = self.vision_backbone(crops)
        # Keep this trainable operation outside the frozen backbone's no_grad.
        visual = self.vision_projection(cls.to(dtype=self.vision_projection.weight.dtype))
        fused = torch.cat((tokens, visual.transpose(0, 1)), dim=-1)
        return self._posterior(self.joint_dist_layer(self.joint_encoder(fused)))

    def encode_joint(self, camera_features: Tensor, images: Tensor, bboxes: Tensor):
        """Return z and q_joint for a complete visual batch, [20,B,256]."""
        return self._encode_joint_tokens(self._camera_tokens(camera_features), images, bboxes)

    def forward(self, camera_features: Tensor, images: Optional[Tensor] = None,
                bboxes: Optional[Tensor] = None, has_images: Optional[Tensor] = None):
        """Encode only. Joint outputs use the subset selected by joint_indices.

        latent_camera/dist_camera cover all B rows. latent_joint/dist_joint
        cover B_visual rows, or are None. latent selects joint rows where
        available, regardless of train/eval mode. Call decode explicitly.
        """
        tokens = self._camera_tokens(camera_features)
        batch = camera_features.shape[0]
        if (images is None) != (bboxes is None):
            raise ValueError("images and bboxes must be provided together")
        if has_images is None:
            has_images = torch.full((batch,), images is not None, dtype=torch.bool,
                                    device=camera_features.device)
        if (not isinstance(has_images, Tensor) or has_images.dtype != torch.bool
                or has_images.shape != (batch,)):
            raise ValueError("has_images must be a bool tensor [B]; partial-frame masks are unsupported")
        has_images = has_images.to(camera_features.device)
        joint_indices = has_images.nonzero(as_tuple=True)[0]
        if joint_indices.numel():
            if images is None:
                raise ValueError("has_images=True requires all 20 images and bboxes")
            if (not isinstance(images, Tensor) or images.ndim != 5
                    or images.shape[:3] != (batch, 20, 3)):
                raise ValueError("images must have full-batch shape [B,20,3,H,W]")
            if not isinstance(bboxes, Tensor) or bboxes.shape != (batch, 20, 4):
                raise ValueError("bboxes must have full-batch shape [B,20,4]")

        latent_camera, dist_camera = self._posterior(
            self.camera_dist_layer(self.camera_encoder(tokens)))
        latent_joint = dist_joint = None
        latent = latent_camera
        if joint_indices.numel():
            latent_joint, dist_joint = self._encode_joint_tokens(
                tokens.index_select(1, joint_indices),
                images.index_select(0, joint_indices.to(images.device)),
                bboxes.index_select(0, joint_indices.to(bboxes.device)),
            )
            latent = latent_camera.index_copy(1, joint_indices, latent_joint)
        return {
            "latent_camera": latent_camera, "dist_camera": dist_camera,
            "latent_joint": latent_joint, "dist_joint": dist_joint,
            "joint_indices": joint_indices, "has_images": has_images, "latent": latent,
        }

    def decode(self, z: Tensor) -> Tensor:
        """[20,B,256] -> [B,20,camera_dim], raw predictions (no geometry repair)."""
        if (not isinstance(z, Tensor) or z.ndim != 3 or z.shape[0] != 20
                or z.shape[1] < 1 or z.shape[2] != self.latent_dim):
            raise ValueError("z must have shape [20,B,256]")
        if not z.is_floating_point() or not torch.isfinite(z).all():
            raise ValueError("z must contain finite floating point values")
        z = z.to(dtype=self.final_layer.weight.dtype)
        output = self.decoder(self.decoder_pos_encoder(z)).transpose(0, 1)
        return self.final_layer(self.decoder_upsample(output))
