# coding=utf-8
# Copyright 2025 NUS Show Lab.
# Modifications Copyright 2026 UniMotion Authors.
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
Showo2 Model with Motion Modality Support.
Adds human Motion modality support on top of the original image/video
understanding and generation capabilities.

Paper note: the class Showo2WithMotion in this file is the UniMotion model.

Motion data flow:
- Input: [bs, seq_len, nfeats=269] - raw human motion features (269-dim version)
- Motion VAE Encode: [bs, seq_len, 269] -> [latent_len, bs, 256]
- Motion Embedder: [latent_len, bs, 256] -> [bs, latent_len, hidden_size]
- After LLM: [bs, latent_len, hidden_size]
- Diffusion Head: [bs, latent_len, hidden_size] -> [bs, latent_len, 256]
- Motion VAE Decode: [latent_len, bs, 256] -> [bs, seq_len, 269]
- Note: when downsample_rate=1, latent_len=seq_len (no downsampling)
"""

import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoConfig
from torch.nn.attention.flex_attention import BlockMask

from .misc import velocity_prediction, next_token_prediction, interpolate_pos_encoding
from .modeling_siglip import SiglipModel
from .modeling_utils import ConfigMixin, ModelMixin, register_to_config
from .modules import DiffusionHeadConfig
from .modules import ModulatedAttentionBlock, RMSNorm, PatchEmbed, TimestepEmbedder, FinalLayer
from .qwen2 import Qwen2ForCausalLM
from .lora import apply_token_wise_lora, get_lora_state_dict, load_lora_state_dict
from .pose_vit_backbone import PoseViTBackbone


# ===================== Motion-specific Modules =====================

class MotionPositionalEncoding(nn.Module):
    """
    1D Sinusoidal Positional Encoding for Motion sequences.
    Adds positional information to variable-length Motion latent sequences.
    """
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build the positional encoding table
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # [max_len, d_model] -> [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [bs, seq_len, d_model]
        Returns:
            [bs, seq_len, d_model] with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class MotionEmbedderDualPath(nn.Module):
    """
    Dual-path Motion Embedder (mirrors the image-processing design of the original Show-o2):

    Paper: Dual-Path Embedder (Sec. 3.3); und = Semantic branch, gen = Generation branch.

    Design principle: stay consistent with the image modality -- all tasks go through the fused path!

    Paths:
    - und path (Understanding): extracts semantic features
      - input motion latent -> Transformer semantic layers -> semantic features [motion_semantic_dim]
    - gen path (Generation): preserves detail features
      - input motion latent -> MLP projection -> generation features [hidden_size]
    - fused path: fuses und+gen features (all tasks uniformly use this path)
      - output: [bs, latent_len, hidden_size]

    Usage (consistent with images, always via the fused path):
    - T2M: forward_fused (noisy latents also need semantic information to aid denoising)
    - M2T: forward_fused (understand the semantics of clean motion)
    - T+M2M: both Source and Target go through forward_fused
    - M2M Prediction: both Source and Target go through forward_fused
    - M2Image: Motion as condition, forward_fused
    - Image2M: Motion as generation target, forward_fused

    Motion VAE output format: [latent_len, bs, 256]
    Needs to be converted to: [bs, latent_len, hidden_size]
    """
    def __init__(
        self,
        motion_latent_dim: int = 256,
        hidden_size: int = 1536,
        motion_semantic_dim: int = 512,       # semantic feature dim (similar to clip_latent_dim)
        num_motion_semantic_layers: int = 4,  # number of semantic Transformer layers
        max_motion_latent_len: int = 128,     # max motion latent sequence length
    ):
        super().__init__()
        self.motion_latent_dim = motion_latent_dim
        self.hidden_size = hidden_size
        self.motion_semantic_dim = motion_semantic_dim
        
        # ===== Understanding Path (similar to image_embedder_und + und_trans) =====
        # For understanding tasks: extracts the semantic information of motion
        # Input: clean motion latent from VAE encoder
        self.motion_embedder_und = nn.Sequential(
            nn.Linear(motion_latent_dim, motion_semantic_dim),
            nn.GELU(),
            nn.Linear(motion_semantic_dim, motion_semantic_dim)
        )
        
        # Positional encoding (for the und-path Transformer)
        self.motion_position_embedding = nn.Embedding(max_motion_latent_len, motion_semantic_dim)
        self.register_buffer(
            "motion_position_ids",
            torch.arange(max_motion_latent_len).expand((1, -1)),
            persistent=False
        )
        
        # Semantic Transformer layers (similar to the SigLIP encoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=motion_semantic_dim,
            nhead=8,
            dim_feedforward=motion_semantic_dim * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.motion_semantic_trans = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_motion_semantic_layers
        )
        
        # ===== Generation Path (similar to image_embedder_gen) =====
        # For generation tasks: maps noisy latents into the hidden space
        # Input: noisy motion latent (to be denoised)
        self.motion_embedder_gen = nn.Sequential(
            nn.Linear(motion_latent_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        # The gen branch in the fused path uses a learnable positional embedding consistent with und
        self.motion_position_embedding_gen = nn.Embedding(max_motion_latent_len, hidden_size)
        
        # ===== Fusion Path (similar to fusion_proj) =====
        # Fuses und and gen features, for tasks that need both semantic understanding and detail preservation
        self.motion_fusion_proj = nn.Sequential(
            RMSNorm(motion_semantic_dim + hidden_size),
            nn.Linear(motion_semantic_dim + hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward_und(self, motion_latents: torch.Tensor) -> torch.Tensor:
        """
        Understanding path: clean motion -> semantic features

        Paper: Semantic branch of the Dual-Path Embedder (Sec. 3.3).

        Args:
            motion_latents: [latent_len, bs, motion_latent_dim] clean motion latents from VAE
        Returns:
            [bs, latent_len, motion_semantic_dim] semantic features
        """
        # [latent_len, bs, dim] -> [bs, latent_len, dim]
        x = motion_latents.permute(1, 0, 2)
        bs, seq_len, _ = x.shape

        # Project to the semantic dimension
        x = self.motion_embedder_und(x)

        # Add positional encoding
        position_ids = self.motion_position_ids[:, :seq_len]
        x = x + self.motion_position_embedding(position_ids)

        # Extract semantics through the semantic Transformer
        x = self.motion_semantic_trans(x)
        
        return x
    
    def forward_gen(self, motion_latents: torch.Tensor) -> torch.Tensor:
        """
        Generation path: motion latent -> detail features (internal use only)

        Paper: Generation branch of the Dual-Path Embedder (Sec. 3.3).

        Note: this method is only for internal use by forward_fused and should not be called directly.
        All tasks should use forward_fused to stay consistent with the image-modality design.

        Args:
            motion_latents: [latent_len, bs, motion_latent_dim] motion latents
        Returns:
            [bs, latent_len, hidden_size] detail features
        """
        # [latent_len, bs, dim] -> [bs, latent_len, dim]
        x = motion_latents.permute(1, 0, 2)

        # Project directly to the hidden size
        x = self.motion_embedder_gen(x)
        
        return x
    
    def forward_fused(self, motion_latents: torch.Tensor) -> torch.Tensor:
        """
        Fused path: extracts semantics and preserves details at the same time
        (all tasks uniformly use this path)

        Consistent with the image modality: both understanding and generation tasks use the fused path.
        - Semantic features (und): help the model understand the high-level semantics of motion
        - Detail features (gen): preserve the low-level detail information of motion

        Args:
            motion_latents: [latent_len, bs, motion_latent_dim] motion latents
                           (can be clean or noisy; the model handles both adaptively)
        Returns:
            [bs, latent_len, hidden_size] fused features
        """
        # Extract und features (semantics)
        x_und = self.forward_und(motion_latents)  # [bs, seq_len, motion_semantic_dim]

        # Extract gen features (details)
        x_gen = self.forward_gen(motion_latents)  # [bs, seq_len, hidden_size]
        position_ids = self.motion_position_ids[:, :x_gen.size(1)]
        x_gen = x_gen + self.motion_position_embedding_gen(position_ids)

        # Fuse the two feature streams
        x_fused = self.motion_fusion_proj(torch.cat([x_und, x_gen], dim=-1))
        
        return x_fused
    
    def forward(self, motion_latents: torch.Tensor, mode: str = 'fused') -> torch.Tensor:
        """
        Unified forward interface (uses the fused path by default)

        Recommended: all tasks use mode='fused' to stay consistent with the image modality.

        Args:
            motion_latents: [latent_len, bs, motion_latent_dim]
            mode: 'fused' (recommended), 'und', or 'gen' (internal use only)
        Returns:
            [bs, latent_len, hidden_size] (or motion_semantic_dim for 'und')
        """
        if mode == 'fused':
            return self.forward_fused(motion_latents)
        elif mode == 'und':
            return self.forward_und(motion_latents)
        elif mode == 'gen':
            # Internal use only; not recommended to call directly
            return self.forward_gen(motion_latents)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'fused', 'und', or 'gen'")


class MotionFinalLayer(nn.Module):
    """
    Motion Final Layer: maps LLM outputs back to the Motion VAE latent space.

    Paper: final projection of the Motion Flow Head (used as motion_diffusion_head_b).

    Uses AdaLN modulation, similar to the FinalLayer in image generation.
    """
    def __init__(self, hidden_size: int, motion_latent_dim: int = 256):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, motion_latent_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        
        # Zero-initialize
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        adaln_input: torch.Tensor,
        modality_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [bs, seq_len, hidden_size] - LLM hidden states
            adaln_input: [num_motions, hidden_size] - timestep embeddings
            modality_positions: [bs, num_motions, 2] - (offset, length) pairs
        Returns:
            [bs, seq_len, motion_latent_dim] - predicted velocity
        """
        shift, scale = self.adaLN_modulation(adaln_input).chunk(2, dim=1)
        
        # Build tensors for modulation
        shift_new = torch.zeros_like(x)
        scale_new = torch.zeros_like(x)
        
        for i, modality_batch in enumerate(modality_positions):
            for j, (offset, length) in enumerate(modality_batch):
                if length == 0:
                    continue
                idx = i * modality_positions.size(1) + j
                shift_new[i, offset:offset + length] = shift[idx]
                scale_new[i, offset:offset + length] = scale[idx]
        
        # Modulate and project
        x = self._modulate(self.norm_final(x), shift_new, scale_new)
        x = self.linear(x)
        return x
    
    def _modulate(self, x, shift, scale):
        """Apply modulation with shift and scale."""
        input_dtype = x.dtype
        x = x.to(torch.float32)
        shift = shift.to(torch.float32)
        scale = scale.to(torch.float32)
        return (x * (1 + scale) + shift).to(input_dtype)


def motion_velocity_prediction(latents: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Compute the MSE loss for Motion velocity prediction.

    Args:
        latents: [bs, latent_len, dim] predicted velocity
        labels: [bs, latent_len, dim] target velocity
        mask: [bs, latent_len] motion mask
    Returns:
        scalar loss
    """
    if mask is not None:
        # Expand mask for all dimensions
        mask = mask.unsqueeze(-1).expand_as(latents)
        loss = F.mse_loss(latents, labels, reduction='none')
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        return loss
    else:
        return F.mse_loss(latents, labels)


# ===================== Main Model =====================

class Showo2WithMotion(ModelMixin, ConfigMixin):
    """
    Showo2 Model with Motion Modality Support.

    Paper: this class is the UniMotion model.

    Supported capabilities:
    1. Original image/video understanding and generation
    2. New Motion understanding (Motion -> Text)
    3. New Motion generation (Text -> Motion)

    Motion processing flow:
    - Generation: text tokens -> LLM -> Diffusion Head -> Motion latent -> VAE decode -> Motion
    - Understanding: Motion -> VAE encode -> Motion latent -> Embedder -> LLM -> text tokens
    """
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(
        self,
        # LLM config
        llm_vocab_size=None,
        llm_model_path='',
        load_from_showo=False,
        hidden_size=1536,

        # Image/Video config (kept from the original)
        image_latent_dim=16,
        image_latent_height=16,
        image_latent_width=16,
        video_latent_height=16,
        video_latent_width=16,
        patch_size=2,
        clip_latent_dim=1152,
        num_diffusion_layers=10,
        add_time_embeds=True,
        add_qk_norm=False,
        clip_pretrained_model_path="google/siglip-so400m-patch14-384",
        
        # Motion config (new)
        motion_latent_dim=256,         # Motion VAE latent dimension
        motion_nfeats=269,             # Motion feature dimension (269-dim version, includes 6 extra dims)
        motion_downsample_rate=1,      # Motion VAE downsample rate (1=no downsampling)
        max_motion_length=500,         # Maximum motion sequence length
        num_motion_diffusion_layers=6, # Motion specific diffusion layers
        motion_semantic_dim=512,       # Motion semantic feature dim (similar to clip_latent_dim)
        num_motion_semantic_layers=4,  # Number of motion semantic Transformer layers
        pose_branch=None,              # Optional TokenHMR pose branch for img2pose only
        
        **kwargs,
    ):
        super().__init__()

        pose_branch_cfg = {}
        if pose_branch is None:
            pose_branch_cfg = {}
        elif isinstance(pose_branch, dict):
            pose_branch_cfg = pose_branch
        else:
            pose_branch_cfg = dict(pose_branch)
        if bool(pose_branch_cfg.get("enabled", False)) and bool(pose_branch_cfg.get("strict_ckpt", True)):
            ckpt_path = str(pose_branch_cfg.get("ckpt_path", ""))
            if not ckpt_path or not os.path.exists(ckpt_path):
                raise RuntimeError(
                    "pose_branch is enabled but checkpoint is missing: "
                    f"{ckpt_path if ckpt_path else '<empty>'}"
                )
        
        # ==================== LLM Backbone ====================
        llm_config = AutoConfig.from_pretrained(llm_model_path)
        if load_from_showo:
            self.showo = Qwen2ForCausalLM(llm_config)
        else:
            self.showo = Qwen2ForCausalLM.from_pretrained(llm_model_path, attn_implementation='sdpa')
        self.showo.resize_token_embeddings(llm_vocab_size)
        
        # ==================== Image/Video Modules (kept from the original) ====================
        # Patch embedding for understanding path
        self.image_embedder_und = PatchEmbed(
            patch_size=patch_size,
            in_chans=image_latent_dim,
            embed_dim=clip_latent_dim,
        )
        
        # Patch embedding for generation path
        self.image_embedder_gen = PatchEmbed(
            patch_size=patch_size,
            in_chans=image_latent_dim,
            embed_dim=hidden_size,
        )
        
        # Semantic layers from SigLIP
        siglip_model = SiglipModel.from_pretrained(clip_pretrained_model_path)
        self.position_embedding = siglip_model.vision_model.embeddings.position_embedding
        self.und_trans = siglip_model.vision_model.encoder
        del self.und_trans.layers[-1]
        self.register_buffer(
            "image_position_ids",
            torch.arange(image_latent_height * image_latent_width).expand((1, -1)),
            persistent=False
        )
        
        # Image fusion projection
        self.fusion_proj = nn.Sequential(
            RMSNorm(clip_latent_dim + hidden_size),
            nn.Linear(clip_latent_dim + hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # ==================== Optional Pose Branch (img2pose only) ====================
        pose_branch = pose_branch_cfg

        self.pose_branch_enabled = False
        self.pose_branch_apply_tasks = set(
            [str(x) for x in pose_branch.get("apply_tasks", ["img2pose"])]
        )
        self.pose_branch_run_with_no_grad = bool(pose_branch.get("run_with_no_grad", True))
        self.pose_branch_target_tokens_hw = tuple(
            int(x) for x in pose_branch.get("target_tokens_hw", [27, 27])
        )
        self.pose_feature_dim = int(pose_branch.get("feature_dim", 1280))
        self.pose_vit_backbone = None
        self.pose_fusion_proj = None

        pose_branch_enabled = bool(pose_branch.get("enabled", False))
        if pose_branch_enabled:
            ckpt_path = str(pose_branch.get("ckpt_path", ""))
            strict_ckpt = bool(pose_branch.get("strict_ckpt", True))
            if not ckpt_path or not os.path.exists(ckpt_path):
                msg = (
                    "pose_branch is enabled but checkpoint is missing: "
                    f"{ckpt_path if ckpt_path else '<empty>'}"
                )
                if strict_ckpt:
                    raise RuntimeError(msg)
                pose_branch_enabled = False

            if pose_branch_enabled:
                self.pose_vit_backbone = PoseViTBackbone(
                    ckpt_path=ckpt_path,
                    freeze_backbone=bool(pose_branch.get("freeze_backbone", True)),
                    input_size=int(pose_branch.get("input_size", 256)),
                    crop_width=int(pose_branch.get("crop_width", 192)),
                )
                self.pose_fusion_proj = nn.Sequential(
                    RMSNorm(hidden_size + self.pose_feature_dim),
                    nn.Linear(hidden_size + self.pose_feature_dim, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size),
                )
                self.pose_branch_enabled = True
        
        # ==================== Motion Modules (dual-path design) ====================
        # Dual-path Motion Embedder (mirrors the image processing of the original Show-o2)
        # Paper: Dual-Path Embedder (Sec. 3.3)
        # - und path: for understanding tasks (M2T, etc.), extracts semantic features
        # - gen path: for generation tasks (T2M, etc.), processes noisy latents
        # - fused path: fuses und+gen, for tasks that need full understanding
        self.motion_embedder = MotionEmbedderDualPath(
            motion_latent_dim=motion_latent_dim,
            hidden_size=hidden_size,
            motion_semantic_dim=motion_semantic_dim,
            num_motion_semantic_layers=num_motion_semantic_layers,
            max_motion_latent_len=max_motion_length // motion_downsample_rate + 10,
        )
        
        # [DEPRECATED] Motion positional encoding - deprecated in favor of the learnable
        # positional embeddings inside motion_embedder.
        # This module is kept for compatibility with loading old checkpoints.
        self.motion_pos_encoding = MotionPositionalEncoding(
            d_model=hidden_size,
            max_len=max_motion_length // motion_downsample_rate + 10,
            dropout=0.0
        )
        
        # ==================== Diffusion Head (shared + Motion-specific) ====================
        # Paper: motion_diffusion_head_a/b below form the Motion Flow Head;
        # diffusion_head_a/b form the Vision Flow Head.
        self.diffusion_head_config = DiffusionHeadConfig()
        self.time_embed = TimestepEmbedder(self.diffusion_head_config.hidden_size)
        
        if hidden_size != self.diffusion_head_config.hidden_size:
            self.diff_proj = nn.Sequential(
                nn.Linear(hidden_size, self.diffusion_head_config.hidden_size),
                nn.GELU(),
                nn.Linear(self.diffusion_head_config.hidden_size, self.diffusion_head_config.hidden_size)
            )
            self.time_embed_proj = nn.Linear(self.diffusion_head_config.hidden_size, hidden_size)
        
        # Shared Diffusion attention layers (for Image/Video)
        self.diffusion_head_a = nn.ModuleList([
            ModulatedAttentionBlock(self.diffusion_head_config, layer_idx)
            for layer_idx in range(num_diffusion_layers)
        ])
        
        # Image/Video final layer
        self.diffusion_head_b = FinalLayer(
            self.diffusion_head_config.hidden_size,
            patch_size,
            image_latent_dim
        )
        
        # Motion-specific diffusion layers
        # Paper: Motion Flow Head (attention blocks)
        self.motion_diffusion_head_a = nn.ModuleList([
            ModulatedAttentionBlock(self.diffusion_head_config, layer_idx)
            for layer_idx in range(num_motion_diffusion_layers)
        ])
        
        # Motion final layer
        # Paper: Motion Flow Head (final projection)
        self.motion_diffusion_head_b = MotionFinalLayer(
            self.diffusion_head_config.hidden_size,
            motion_latent_dim
        )
        
        # Initialize parameters
        self.reset_parameters()
        
        # ==================== Token-Level LoRA ====================
        # LoRA is disabled by default; the training script enables it via apply_lora()
        self.lora_config = None
        self.lora_enabled = False
        # For sequential cross-stage training: cache LoRA branch parameters that exist
        # in the checkpoint but are not instantiated in the current model
        self._preserved_lora_state: Dict[str, torch.Tensor] = {}
    
    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True
    
    def reset_parameters(self):
        """Initialize model parameters."""
        # Initialize image embedders
        w1 = self.image_embedder_und.proj.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        nn.init.constant_(self.image_embedder_und.proj.bias, 0)
        
        w2 = self.image_embedder_gen.proj.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.image_embedder_gen.proj.bias, 0)
        
        # Initialize transformer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        _basic_init(self.und_trans)
        _basic_init(self.fusion_proj)
        _basic_init(self.diffusion_head_a)
        _basic_init(self.motion_diffusion_head_a)
        
        # Initialize motion embedder (dual-path)
        _basic_init(self.motion_embedder.motion_embedder_und)
        _basic_init(self.motion_embedder.motion_embedder_gen)
        _basic_init(self.motion_embedder.motion_fusion_proj)
        _basic_init(self.motion_embedder.motion_semantic_trans)
        
        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        
        # Zero-out output layers
        nn.init.constant_(self.diffusion_head_b.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.diffusion_head_b.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.diffusion_head_b.linear.weight, 0)
        nn.init.constant_(self.diffusion_head_b.linear.bias, 0)
    
    def unpatchify(self, x, h, w, T=0):
        """Unpatchify image/video latents."""
        c = self.config.image_latent_dim
        p = self.image_embedder_gen.patch_size[0]
        if T == 0:
            x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
            imgs = x.reshape(shape=(x.shape[0], h * p * w * p, c))
        else:
            x = x.reshape(shape=(x.shape[0], T, h, w, p, p, c))
            imgs = x.reshape(shape=(x.shape[0], T, h * p * w * p, c))
        return imgs
    
    # ==================== Token-Level LoRA Methods ====================
    
    def apply_lora(self, lora_config: dict) -> None:
        """
        Apply Token-Level LoRA to the attention layers of the LLM backbone.

        Paper: Modality-Routed LoRA (LoRA-A: text/RGB, LoRA-B: motion).

        Args:
            lora_config: dict with keys:
                - rank_A: rank of LoRA-A (Text/Image path)
                - rank_B: rank of LoRA-B (Motion path)
                - alpha: LoRA alpha
                - dropout: dropout rate
                - enable_A: whether to enable LoRA-A
                - enable_B: whether to enable LoRA-B
                - target_modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        """
        apply_token_wise_lora(self.showo, lora_config)
        self.lora_config = lora_config
        self.lora_enabled = True
    
    def get_lora_parameters(self) -> list:
        """Get all LoRA parameters for the optimizer."""
        lora_params = []
        for name, param in self.named_parameters():
            if 'lora_' in name:
                lora_params.append(param)
        return lora_params
    
    def save_lora(self, save_path: str) -> None:
        """Save LoRA weights."""
        state_dict = get_lora_state_dict(self)
        active_count = len(state_dict)
        preserved_count = len(self._preserved_lora_state)
        if preserved_count > 0:
            merged_state_dict = dict(self._preserved_lora_state)
            # Keys already in the current model take priority, to avoid being overwritten by the old cache
            merged_state_dict.update(state_dict)
            state_dict = merged_state_dict
        torch.save(state_dict, save_path)
        print(
            f"[LoRA] Saved LoRA weights to {save_path} "
            f"(active={active_count}, preserved={preserved_count}, total={len(state_dict)})"
        )
    
    def load_lora(
        self,
        load_path: str,
        strict: bool = False,
        preserve_unmatched: bool = True,
    ) -> Dict[str, object]:
        """Load LoRA weights."""
        try:
            state_dict = torch.load(load_path, map_location='cpu', weights_only=True)
        except TypeError:
            # Compatibility with older PyTorch versions
            state_dict = torch.load(load_path, map_location='cpu')
        load_info = load_lora_state_dict(self, state_dict, strict=strict)
        unmatched_state = load_info.get('unmatched_state_dict', {})
        if preserve_unmatched and unmatched_state:
            # Keep on CPU so checkpoints can be chained across devices
            for key, value in unmatched_state.items():
                self._preserved_lora_state[key] = value.clone().cpu()
        print(
            f"[LoRA] Loaded LoRA weights from {load_path} "
            f"(loaded={len(load_info.get('loaded_keys', []))}, "
            f"unmatched={len(load_info.get('missing_keys_in_model', []))}, "
            f"preserved={len(self._preserved_lora_state)})"
        )
        return load_info
    
    def _create_modality_mask(
        self,
        batch_size: int,
        seq_len: int,
        modality_positions: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create the modality_mask from modality_positions.

        Used by Token-Level LoRA to indicate which LoRA path each position uses
        (Paper: the deterministic routing of Modality-Routed LoRA).

        Args:
            batch_size: batch size
            seq_len: sequence length
            modality_positions: [bs, num_modalities, 2] (offset, length) pairs
                                indicating the positions of motion tokens
            device: torch device

        Returns:
            modality_mask: [bs, seq_len]
                - 0: text/image tokens (use LoRA-A)
                - 1: motion tokens (use LoRA-B)
        """
        modality_mask = torch.zeros(batch_size, seq_len, device=device)

        for i in range(batch_size):
            for j in range(modality_positions.size(1)):
                offset = modality_positions[i, j, 0].item()
                length = modality_positions[i, j, 1].item()
                if length > 0:
                    # Mark motion token positions as 1
                    end_pos = min(offset + length, seq_len)
                    modality_mask[i, offset:end_pos] = 1.0
        
        return modality_mask
    
    # ==================== Motion-specific Methods ====================
    
    def forward_motion_gen(
        self,
        text_tokens: torch.Tensor,
        motion_latents: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor = None,
        motion_lengths: List[int] = None,
        text_labels: torch.Tensor = None,
        motion_labels: torch.Tensor = None,
        modality_positions: torch.Tensor = None,
        text_masks: torch.Tensor = None,
        motion_masks: torch.Tensor = None,
        output_hidden_states: bool = True,
        max_seq_len: int = None,
        device: str = 'cuda:0',
        **kwargs,
    ):
        """
        Forward pass for Motion Generation (Text-to-Motion).
        
        Args:
            text_tokens: [bs, seq_len] text token ids
            motion_latents: [latent_len, bs, motion_latent_dim] noisy motion latents
            t: [num_motions] diffusion timesteps
            attention_mask: attention mask for the sequence
            motion_lengths: list of original motion lengths
            text_labels: [bs, seq_len] text labels for NTP loss
            motion_labels: [bs, latent_len, motion_latent_dim] target motion velocities
            modality_positions: [bs, num_motions, 2] (offset, length) pairs
            text_masks: [bs, seq_len] text token mask
            motion_masks: [bs, seq_len] motion token mask
            output_hidden_states: whether to output hidden states
            max_seq_len: maximum sequence length
            device: device string
            
        Returns:
            If motion_labels is not None: (logits, loss_ntp, loss_motion)
            Else: (logits, v_pred)
        """
        # Get text embeddings
        input_embeds = self.showo.model.embed_tokens(text_tokens)
        dtype = input_embeds.dtype
        bs = text_tokens.shape[0]
        device = input_embeds.device
        
        # Get motion embeddings using FUSED path (consistent with the image modality)
        # [latent_len, num_motions, dim] -> [num_motions, latent_len, hidden_size]
        # Note: num_motions = bs when not using CFG, 2*bs when using CFG
        num_motions = motion_latents.shape[1]
        # T2M task: use the fused path, fusing semantic and detail features just like images
        # The fused path already includes positional encoding internally; no extra needed
        motion_embeds = self.motion_embedder.forward_fused(motion_latents.to(dtype))
        
        # Get time embeddings
        # t might come from ODE solver with wrong batch size, need to expand/fix
        if t.numel() == 1:
            # t is a scalar, expand to num_motions
            t = t.expand(num_motions)
        elif t.shape[0] != num_motions:
            # t has wrong size (e.g., latent_len instead of num_motions), fix it
            t = t[0].expand(num_motions)
        
        time_embeds = self.time_embed(t, dtype)
        if hasattr(self, 'time_embed_proj'):
            time_embeds_proj = self.time_embed_proj(time_embeds)
        else:
            time_embeds_proj = time_embeds
        
        # Prepare motion labels for loss calculation
        if motion_labels is not None:
            latent_len = motion_latents.shape[0]
            # motion_labels: [bs, latent_len, motion_latent_dim]
            new_motion_labels = torch.zeros(
                [bs, max_seq_len, self.config.motion_latent_dim],
                device=device, dtype=dtype
            )
            motion_masks_expanded = motion_masks.unsqueeze(-1).expand(
                -1, -1, self.config.motion_latent_dim
            )
        
        # Structure text and motion embeddings into sequences
        for i, modality_batch in enumerate(modality_positions):
            for j, (offset, length) in enumerate(modality_batch):
                if length == 0:
                    continue
                idx = i * modality_positions.size(1) + j
                
                if self.config.add_time_embeds:
                    # First position is time embedding
                    input_embeds[i, offset] = time_embeds_proj[idx]
                    # Remaining positions are motion embeddings
                    input_embeds[i, offset + 1:offset + 1 + length - 1] = motion_embeds[idx, :max(length - 1, 0)]
                    
                    if motion_labels is not None:
                        motion_masks[i, offset] = 0  # Mask time embedding position
                        new_motion_labels[i, offset + 1:offset + 1 + length - 1] = motion_labels[idx, :max(length - 1, 0)]
                else:
                    input_embeds[i, offset:offset + length] = motion_embeds[idx, :length]
                    if motion_labels is not None:
                        new_motion_labels[i, offset:offset + length] = motion_labels[idx, :length]
        
        # Create modality_mask for Token-Level LoRA
        modality_mask = None
        if self.lora_enabled:
            modality_mask = self._create_modality_mask(
                bs, input_embeds.shape[1], modality_positions, device
            )

        # Forward through LLM
        outputs = self.showo(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            modality_mask=modality_mask,  # Token-Level LoRA mask
        )
        
        logits = outputs['logits']
        last_hidden_states = outputs['hidden_states'][-1]
        
        # Forward through Motion Diffusion Head
        if hasattr(self, 'diff_proj'):
            last_hidden_states = self.diff_proj(last_hidden_states)
        
        position_ids = torch.arange(
            last_hidden_states.shape[1], 
            device=last_hidden_states.device
        ).unsqueeze(0)
        
        # Motion-specific diffusion layers
        for layer in self.motion_diffusion_head_a:
            last_hidden_states = layer(
                hidden_states=last_hidden_states,
                adaln_input=time_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                modality_positions=modality_positions,
            )[0]
        
        # Motion final layer
        v_pred = self.motion_diffusion_head_b(last_hidden_states, time_embeds, modality_positions)
        
        # Calculate losses or return predictions
        if text_labels is not None and motion_labels is not None:
            loss_ntp = next_token_prediction(logits, text_labels, self.config.llm_vocab_size)
            loss_motion = motion_velocity_prediction(
                v_pred[:bs], new_motion_labels[:bs], motion_masks
            )
            return logits, loss_ntp, loss_motion
        
        elif motion_labels is not None:
            loss_motion = motion_velocity_prediction(
                v_pred[:bs], new_motion_labels[:bs], motion_masks
            )
            return logits, loss_motion
        
        elif text_labels is not None:
            loss_ntp = next_token_prediction(logits, text_labels, self.config.llm_vocab_size)
            return logits, loss_ntp
        
        else:
            # Inference mode: extract motion predictions
            v_pred_list = []
            for i, modality_batch in enumerate(modality_positions):
                for j, (offset, length) in enumerate(modality_batch):
                    if length == 0:
                        break
                    v_pred_list.append(v_pred[i, offset:offset + length])
            
            if len(v_pred_list) > 0:
                v_pred_out = torch.stack(v_pred_list)
                # Remove time embedding position if needed
                if self.config.add_time_embeds:
                    v_pred_out = v_pred_out[:, 1:, :]
                
                # [num_motions, latent_len, motion_latent_dim] -> [latent_len, num_motions, motion_latent_dim]
                v_pred_out = v_pred_out.permute(1, 0, 2)
            else:
                v_pred_out = None
            
            return logits, v_pred_out
    
    def forward_motion_und(
        self,
        text_tokens: torch.Tensor,
        motion_latents: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor = None,
        motion_lengths: List[int] = None,
        text_labels: torch.Tensor = None,
        modality_positions: torch.Tensor = None,
        output_hidden_states: bool = True,
        max_seq_len: int = None,
        device: str = 'cuda:0',
        **kwargs,
    ):
        """
        Forward pass for Motion Understanding (Motion-to-Text).
        Uses the FUSED path: combines semantic understanding (und) and detail preservation (gen)

        Args:
            text_tokens: [bs, seq_len] text token ids
            motion_latents: [latent_len, bs, motion_latent_dim] clean motion latents from VAE
            t: [num_motions] timesteps (usually set to 1.0 for understanding)
            attention_mask: attention mask for the sequence
            motion_lengths: list of original motion lengths
            text_labels: [bs, seq_len] text labels for NTP loss
            modality_positions: [bs, num_motions, 2] (offset, length) pairs
            output_hidden_states: whether to output hidden states
            max_seq_len: maximum sequence length
            device: device string
            
        Returns:
            If text_labels is not None: (logits, loss_ntp)
            Else: logits
        """
        # Get text embeddings
        input_embeds = self.showo.model.embed_tokens(text_tokens)
        dtype = input_embeds.dtype
        bs = text_tokens.shape[0]
        
        # Get motion embeddings using FUSED path (for clean motion latents in understanding)
        # M2T task: the input is a clean motion latent; use the fused path (semantics + details)
        # [latent_len, num_motions, dim] -> [num_motions, latent_len, hidden_size]
        num_motions = motion_latents.shape[1]
        motion_embeds = self.motion_embedder.forward_fused(motion_latents.to(dtype))
        # Note: the und path inside the fused path already has positional encoding;
        # no extra motion_pos_encoding is needed
        
        # Get time embeddings (for consistency, though not strictly needed for understanding)
        # Fix t batch size if needed
        if t.numel() == 1:
            t = t.expand(num_motions)
        elif t.shape[0] != num_motions:
            t = t[0].expand(num_motions)
        
        time_embeds = self.time_embed(t, dtype)
        if hasattr(self, 'time_embed_proj'):
            time_embeds_proj = self.time_embed_proj(time_embeds)
        else:
            time_embeds_proj = time_embeds
        
        # Structure text and motion embeddings into sequences
        for i, modality_batch in enumerate(modality_positions):
            for j, (offset, length) in enumerate(modality_batch):
                if length == 0:
                    continue
                idx = i * modality_positions.size(1) + j
                
                if self.config.add_time_embeds:
                    input_embeds[i, offset] = time_embeds_proj[idx]
                    input_embeds[i, offset + 1:offset + 1 + length - 1] = motion_embeds[idx, :max(length - 1, 0)]
                else:
                    input_embeds[i, offset:offset + length] = motion_embeds[idx, :length]
        
        # Create modality_mask for Token-Level LoRA
        modality_mask = None
        if self.lora_enabled:
            modality_mask = self._create_modality_mask(
                bs, input_embeds.shape[1], modality_positions, input_embeds.device
            )
        
        # Forward through LLM
        outputs = self.showo(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            modality_mask=modality_mask,  # Token-Level LoRA mask
        )
        
        logits = outputs['logits']
        
        if text_labels is not None:
            loss_ntp = next_token_prediction(logits, text_labels, self.config.llm_vocab_size)
            return logits, loss_ntp
        else:
            return logits
    
    @torch.no_grad()
    def t2m_generate(
        self,
        motion_latents: torch.Tensor,
        t: torch.Tensor,
        text_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        modality_positions: torch.Tensor,
        motion_lengths: List[int] = None,
        max_seq_len: int = None,
        guidance_scale: float = 0.0,
        **kwargs,
    ):
        """
        Generate motion velocity prediction for text-to-motion generation.
        
        Args:
            motion_latents: [latent_len, bs, motion_latent_dim] noisy motion latents
            t: [num_motions] diffusion timesteps
            text_tokens: [bs, seq_len] text token ids
            attention_mask: attention mask
            modality_positions: [bs, num_motions, 2] (offset, length) pairs
            motion_lengths: list of original motion lengths
            max_seq_len: maximum sequence length
            guidance_scale: CFG guidance scale
            
        Returns:
            v_pred: [latent_len, num_motions, motion_latent_dim] velocity predictions
        """
        if guidance_scale > 0.0:
            # Classifier-free guidance
            _, v = self.forward_motion_gen(
                text_tokens=text_tokens,
                motion_latents=motion_latents,
                t=t,
                attention_mask=attention_mask,
                modality_positions=modality_positions,
                motion_lengths=motion_lengths,
                output_hidden_states=True,
                max_seq_len=max_seq_len,
            )
            
            # Split conditional and unconditional
            v_cond, v_uncond = torch.chunk(v, 2, dim=1)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
            return torch.cat([v, v], dim=1)
        else:
            _, v = self.forward_motion_gen(
                text_tokens=text_tokens,
                motion_latents=motion_latents,
                t=t,
                attention_mask=attention_mask,
                modality_positions=modality_positions,
                motion_lengths=motion_lengths,
                output_hidden_states=True,
                max_seq_len=max_seq_len,
            )
            return v
    
    @torch.no_grad()
    def m2t_generate(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        modality_mask: torch.Tensor = None,  # New: LoRA mask
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = None,
        do_sample: bool = False,  # New: whether to sample (False=greedy decoding)
        eos_token: int = None,
    ):
        """
        Generate text from motion (Motion Understanding).
        
        Similar to mmu_generate but for motion modality.
        Consistent with MotionGPT3, greedy decoding is used by default.
        
        Args:
            input_embeds: [bs, seq_len, hidden_size] input embeddings
            attention_mask: attention mask
            modality_mask: [bs, seq_len] mask for Token-Level LoRA
                          - 0: text tokens (use LoRA-A)
                          - 1: motion tokens (use LoRA-B)
            max_new_tokens: maximum number of tokens to generate
            temperature: sampling temperature (only used when do_sample=True)
            top_k: top-k sampling (only used when do_sample=True)
            do_sample: whether to use sampling (False=greedy decoding, like MotionGPT3)
            eos_token: end of sequence token
        """
        device = input_embeds.device
        dtype = input_embeds.dtype
        result = []
        
        for _ in range(max_new_tokens):
            # Key fix: pass modality_mask so that LoRA-B is used correctly
            logits = self.showo(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask.to(dtype),
                modality_mask=modality_mask,  # Token-Level LoRA mask
            )['logits']
            
            L = attention_mask.shape[-1]
            attention_mask = attention_mask.squeeze()
            attention_mask_a = torch.hstack([
                attention_mask,
                torch.zeros((L, 1)).to(device) + torch.finfo(logits.dtype).min,
            ])
            attention_mask_b = torch.vstack([
                attention_mask_a,
                torch.hstack([attention_mask[-1, :], torch.tensor([0]).to(device)]).unsqueeze(0),
            ])
            attention_mask = attention_mask_b
            
            # Extend modality_mask (newly generated tokens are text, use LoRA-A)
            if modality_mask is not None:
                modality_mask = torch.cat([
                    modality_mask, 
                    torch.zeros(modality_mask.shape[0], 1, device=device, dtype=modality_mask.dtype)
                ], dim=1)
            
            # Get next token
            logits = logits[:, -1, :]
            
            if do_sample:
                # Sampling mode (original behavior)
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding (like MotionGPT3)
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
            result.append(idx_next[0][0])
            
            # Append to sequence
            idx_next_embeds = self.showo.model.embed_tokens(idx_next)
            input_embeds = torch.cat([input_embeds, idx_next_embeds], dim=1).to(dtype)
            
            if eos_token is not None and idx_next.cpu() == eos_token:
                break
        
        return result
    
    def forward_motion_edit(
        self,
        text_tokens: torch.Tensor,
        source_motion_latents: torch.Tensor,
        target_motion_latents: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor = None,
        source_motion_lengths: List[int] = None,
        target_motion_lengths: List[int] = None,
        text_labels: torch.Tensor = None,
        motion_labels: torch.Tensor = None,
        source_modality_positions: torch.Tensor = None,
        target_modality_positions: torch.Tensor = None,
        text_masks: torch.Tensor = None,
        motion_masks: torch.Tensor = None,
        output_hidden_states: bool = True,
        max_seq_len: int = None,
        device: str = 'cuda:0',
        **kwargs,
    ):
        """
        Forward pass for Motion Editing (T+M2M) / Motion Prediction.
        Consistent with the image modality: both Source and Target use the fused path.

        Note: this routine is also reused by the M2M self-reconstruction task
        (Paper: LRA pre-training, Stage 0 in code).

        Args:
            text_tokens: [bs, seq_len] text token ids
            source_motion_latents: [latent_len, bs, dim] clean source motion (condition)
            target_motion_latents: [latent_len, bs, dim] noisy target motion (generation target, also fused)
            t: [num_motions] diffusion timesteps for target
            attention_mask: attention mask for the sequence
            source_motion_lengths: list of source motion lengths
            target_motion_lengths: list of target motion lengths
            text_labels: [bs, seq_len] text labels for NTP loss
            motion_labels: [bs, latent_len, dim] target motion velocity labels
            source_modality_positions: [bs, 1, 2] (offset, length) for source motion
            target_modality_positions: [bs, 1, 2] (offset, length) for target motion
            text_masks: [bs, seq_len] text token mask
            motion_masks: [bs, seq_len] target motion token mask
            output_hidden_states: whether to output hidden states
            max_seq_len: maximum sequence length
            device: device string
            
        Returns:
            If motion_labels is not None: (logits, loss_ntp, loss_motion)
            Else: (logits, v_pred)
        """
        # Get text embeddings
        input_embeds = self.showo.model.embed_tokens(text_tokens)
        dtype = input_embeds.dtype
        bs = text_tokens.shape[0]
        device = input_embeds.device
        
        # Get source motion embeddings using FUSED path (clean motion as condition)
        # Source motion needs to be understood by the model, so use the fused path
        num_source = source_motion_latents.shape[1]
        source_embeds = self.motion_embedder.forward_fused(source_motion_latents.to(dtype))
        # The fused path already includes positional encoding internally; no extra needed

        # Get target motion embeddings using FUSED path (consistent with the image modality)
        # Target motion also uses the fused path; fusing semantic and detail features aids generation
        num_target = target_motion_latents.shape[1]
        target_embeds = self.motion_embedder.forward_fused(target_motion_latents.to(dtype))
        # The fused path already includes positional encoding internally; no extra needed
        
        # Get time embeddings (for target motion)
        if t.numel() == 1:
            t = t.expand(num_target)
        elif t.shape[0] != num_target:
            t = t[0].expand(num_target)
        
        time_embeds = self.time_embed(t, dtype)
        if hasattr(self, 'time_embed_proj'):
            time_embeds_proj = self.time_embed_proj(time_embeds)
        else:
            time_embeds_proj = time_embeds
        
        # Prepare motion labels for loss calculation
        if motion_labels is not None:
            new_motion_labels = torch.zeros(
                [bs, max_seq_len, self.config.motion_latent_dim],
                device=device, dtype=dtype
            )
        
        # Structure embeddings: [text][source_motion][target_motion]
        # Source motion positions (no time embedding needed for condition)
        for i, modality_batch in enumerate(source_modality_positions):
            for j, (offset, length) in enumerate(modality_batch):
                if length == 0:
                    continue
                idx = i * source_modality_positions.size(1) + j
                input_embeds[i, offset:offset + length] = source_embeds[idx, :length]
        
        # Target motion positions (with time embedding)
        for i, modality_batch in enumerate(target_modality_positions):
            for j, (offset, length) in enumerate(modality_batch):
                if length == 0:
                    continue
                idx = i * target_modality_positions.size(1) + j
                
                if self.config.add_time_embeds:
                    input_embeds[i, offset] = time_embeds_proj[idx]
                    input_embeds[i, offset + 1:offset + 1 + length - 1] = target_embeds[idx, :max(length - 1, 0)]
                    
                    if motion_labels is not None:
                        motion_masks[i, offset] = 0  # Mask time embedding position
                        new_motion_labels[i, offset + 1:offset + 1 + length - 1] = motion_labels[idx, :max(length - 1, 0)]
                else:
                    input_embeds[i, offset:offset + length] = target_embeds[idx, :length]
                    if motion_labels is not None:
                        new_motion_labels[i, offset:offset + length] = motion_labels[idx, :length]
        
        # Create modality_mask for Token-Level LoRA
        # In the Motion Edit task, both source and target motion are motion tokens and use LoRA-B
        modality_mask = None
        if self.lora_enabled:
            # First mark the source motion positions
            modality_mask = self._create_modality_mask(
                bs, input_embeds.shape[1], source_modality_positions, device
            )
            # Then mark the target motion positions (overlaid)
            target_mask = self._create_modality_mask(
                bs, input_embeds.shape[1], target_modality_positions, device
            )
            modality_mask = torch.clamp(modality_mask + target_mask, max=1.0)
        
        # Forward through LLM
        outputs = self.showo(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            modality_mask=modality_mask,  # Token-Level LoRA mask
        )
        
        logits = outputs['logits']
        last_hidden_states = outputs['hidden_states'][-1]
        
        # Forward through Motion Diffusion Head
        if hasattr(self, 'diff_proj'):
            last_hidden_states = self.diff_proj(last_hidden_states)
        
        position_ids = torch.arange(
            last_hidden_states.shape[1], 
            device=last_hidden_states.device
        ).unsqueeze(0)
        
        for layer in self.motion_diffusion_head_a:
            last_hidden_states = layer(
                hidden_states=last_hidden_states,
                adaln_input=time_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                modality_positions=target_modality_positions,  # diffusion only on target motion
            )[0]
        
        v_pred = self.motion_diffusion_head_b(last_hidden_states, time_embeds, target_modality_positions)
        
        # Calculate losses or return predictions
        if text_labels is not None and motion_labels is not None:
            from .misc import next_token_prediction
            loss_ntp = next_token_prediction(logits, text_labels, self.config.llm_vocab_size)
            loss_motion = motion_velocity_prediction(
                v_pred[:bs], new_motion_labels[:bs], motion_masks
            )
            return logits, loss_ntp, loss_motion
        
        elif motion_labels is not None:
            loss_motion = motion_velocity_prediction(
                v_pred[:bs], new_motion_labels[:bs], motion_masks
            )
            return logits, loss_motion
        
        else:
            # Inference mode: extract motion predictions
            v_pred_list = []
            for i, modality_batch in enumerate(target_modality_positions):
                for j, (offset, length) in enumerate(modality_batch):
                    if length == 0:
                        break
                    v_pred_list.append(v_pred[i, offset:offset + length])
            
            if len(v_pred_list) > 0:
                # Edit/Pred may contain variable-length targets; pad to the same length before stacking
                max_len = max(item.shape[0] for item in v_pred_list)
                if any(item.shape[0] != max_len for item in v_pred_list):
                    padded = []
                    for item in v_pred_list:
                        if item.shape[0] < max_len:
                            pad = torch.zeros(
                                max_len - item.shape[0], item.shape[1],
                                device=item.device, dtype=item.dtype
                            )
                            item = torch.cat([item, pad], dim=0)
                        padded.append(item)
                    v_pred_out = torch.stack(padded)
                else:
                    v_pred_out = torch.stack(v_pred_list)
                if self.config.add_time_embeds:
                    v_pred_out = v_pred_out[:, 1:, :]
                v_pred_out = v_pred_out.permute(1, 0, 2)
            else:
                v_pred_out = None
            
            return logits, v_pred_out
    
    @torch.no_grad()
    def pred_generate(
        self,
        target_motion_latents: torch.Tensor,
        t: torch.Tensor,
        text_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        source_motion_latents: torch.Tensor,
        source_modality_positions: torch.Tensor,
        target_modality_positions: torch.Tensor,
        source_motion_lengths: List[int] = None,
        target_motion_lengths: List[int] = None,
        max_seq_len: int = None,
        guidance_scale: float = 0.0,
        **kwargs,
    ):
        """
        Motion Prediction generate method (wrapper for forward_motion_edit in inference mode).
        
        Note: target_motion_latents and t are the first two arguments because ODE sampler
        calls model(x, t, **kwargs) where x is the noisy target latents.
        
        Args:
            target_motion_latents: [latent_len, bs, dim] noisy target motion for diffusion
            t: [bs] diffusion timesteps
            text_tokens: [bs, seq_len] text token ids
            attention_mask: attention mask
            source_motion_latents: [latent_len, bs, dim] clean source motion (condition)
            source_modality_positions: [bs, 1, 2] (offset, length) for source motion
            target_modality_positions: [bs, 1, 2] (offset, length) for target motion
            source_motion_lengths: list of source motion lengths
            target_motion_lengths: list of target motion lengths
            max_seq_len: maximum sequence length
            guidance_scale: classifier-free guidance scale (0.0 = no guidance)
            
        Returns:
            v_pred: [latent_len, bs, dim] velocity prediction for target motion
        """
        if guidance_scale > 1.0:
            # Classifier-free guidance: expects duplicated inputs (conditional + unconditional)
            bs = text_tokens.shape[0] // 2
            
            # Forward pass
            _, v = self.forward_motion_edit(
                text_tokens=text_tokens,
                source_motion_latents=source_motion_latents,
                target_motion_latents=target_motion_latents,
                t=t,
                attention_mask=attention_mask,
                source_modality_positions=source_modality_positions,
                target_modality_positions=target_modality_positions,
                source_motion_lengths=source_motion_lengths,
                target_motion_lengths=target_motion_lengths,
                output_hidden_states=True,
                max_seq_len=max_seq_len,
            )
            
            # Split conditional and unconditional
            v_cond, v_uncond = torch.chunk(v, 2, dim=1)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
            return torch.cat([v, v], dim=1)
        else:
            _, v = self.forward_motion_edit(
                text_tokens=text_tokens,
                source_motion_latents=source_motion_latents,
                target_motion_latents=target_motion_latents,
                t=t,
                attention_mask=attention_mask,
                source_modality_positions=source_modality_positions,
                target_modality_positions=target_modality_positions,
                source_motion_lengths=source_motion_lengths,
                target_motion_lengths=target_motion_lengths,
                output_hidden_states=True,
                max_seq_len=max_seq_len,
            )
            return v
    
    # ==================== Text-to-Text Methods (Stage 3) ====================
    
    def t2t_forward(
        self,
        text_tokens: torch.Tensor,
        text_labels: torch.Tensor,
        attention_mask: torch.Tensor = None,
        text_masks: torch.Tensor = None,
        **kwargs,
    ):
        """
        Forward pass for Text-to-Text (T2T) tasks in Stage 3.
        
        T2T is a text-only task used to strengthen the model's instruction following. Task types:
        - caption-to-framelen: predict the number of motion frames from a caption
        - framelen-to-caption: generate a plausible motion description from a frame count
        - second-to-caption: generate a description from a duration in seconds
        - n2t (random-caption): unconditionally generate a motion description
        
        Args:
            text_tokens: [bs, seq_len] text token ids
            text_labels: [bs, seq_len] labels for NTP loss (-100 for masked positions)
            attention_mask: [bs, 1, seq_len, seq_len] causal attention mask
            text_masks: [bs, seq_len] mask for valid tokens (optional)
            
        Returns:
            logits: [bs, seq_len, vocab_size] model predictions
            loss_ntp: scalar NTP loss
        """
        # Get text embeddings
        input_embeds = self.showo.model.embed_tokens(text_tokens)
        dtype = input_embeds.dtype
        bs = text_tokens.shape[0]
        device = input_embeds.device
        
        # The T2T task involves no motion tokens; use an all-zero modality_mask
        # This means everything goes through the LoRA-A path (Text/Image path)
        modality_mask = None
        if self.lora_enabled:
            modality_mask = torch.zeros(
                bs, input_embeds.shape[1],
                device=device, dtype=dtype
            )
        
        # Forward through LLM
        outputs = self.showo(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=False,  # hidden states not needed
            modality_mask=modality_mask,  # Token-Level LoRA mask
        )
        
        logits = outputs['logits']
        
        # Calculate NTP loss
        loss_ntp = next_token_prediction(logits, text_labels, self.config.llm_vocab_size)
        
        return logits, loss_ntp
    
    # ==================== Image + Motion Methods (Stage 3) ====================
    
    def _encode_pose_tokens(
        self,
        pose_images: torch.Tensor,
        target_hw: Tuple[int, int],
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        if pose_images is None:
            raise RuntimeError(
                "pose_branch is enabled for img2pose, but pose_images was not provided."
            )
        if self.pose_vit_backbone is None:
            raise RuntimeError("pose_branch is enabled but pose_vit_backbone is not initialized.")

        if self.pose_branch_run_with_no_grad:
            with torch.no_grad():
                pose_feats = self.pose_vit_backbone(pose_images)
        else:
            pose_feats = self.pose_vit_backbone(pose_images)

        if pose_feats.shape[1] != self.pose_feature_dim:
            raise RuntimeError(
                f"Unexpected pose feature dim={pose_feats.shape[1]}, expected={self.pose_feature_dim}."
            )

        h_target, w_target = target_hw
        if pose_feats.shape[-2:] != (h_target, w_target):
            pose_feats = F.interpolate(
                pose_feats,
                size=(h_target, w_target),
                mode="bilinear",
                align_corners=False,
            )

        pose_tokens = pose_feats.flatten(2).transpose(1, 2).to(dtype=out_dtype)
        return pose_tokens

    def _encode_image_features(
        self,
        image_latents: torch.Tensor,
        task_name: Optional[str] = None,
        pose_images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode image VAE latents into image embeddings usable by the LLM.

        Flow: image_latents → PatchEmbed(und+gen) → SigLIP encoder → fusion_proj

        WAN21 VAE 432x432 → 54x54 latent, PatchEmbed(patch_size=2) → 27x27=729 tokens.
        The SigLIP (patch14, 384) positional embedding is also 27x27=729, an exact match
        requiring no interpolation.

        Args:
            image_latents: [B, 16, H, W] - image latents encoded by the WAN21 VAE (typically 54x54)

        Returns:
            image_embeds: [B, num_tokens, hidden_size] - fused image features (typically 729 tokens)
        """
        dtype = image_latents.dtype
        b, c, h, w = image_latents.shape
        p = self.config.patch_size
        h_, w_ = h // p, w // p  # 27, 27 for 54x54 input with patch_size=2
        
        # Understanding path
        image_embeds_und = self.image_embedder_und(image_latents)  # [B, h_*w_, clip_latent_dim]
        
        # Add position encoding (consistent with forward_und_only)
        if self.position_embedding.weight.shape[0] == h_ * w_:
            # When sizes match, use position_embedding directly (SigLIP 729 == 27*27)
            image_embeds_und = image_embeds_und + self.position_embedding(self.image_position_ids)
        else:
            # When sizes do not match, use interpolation
            image_embeds_und = image_embeds_und + interpolate_pos_encoding(
                self.config.clip_latent_dim, self.position_embedding, h_, w_, 1,
            )
        
        # SigLIP encoder
        image_embeds_und = self.und_trans(image_embeds_und)['last_hidden_state']  # [B, h_*w_, clip_latent_dim]
        
        # Generation path (frozen, but still provides features)
        image_embeds_gen = self.image_embedder_gen(image_latents)  # [B, h_*w_, hidden_size]
        
        # Fusion
        image_embeds = self.fusion_proj(
            torch.cat([image_embeds_und, image_embeds_gen], dim=-1)
        )  # [B, h_*w_, hidden_size]  (e.g., 729 for 432x432 input)
        
        if self.pose_branch_enabled and task_name in self.pose_branch_apply_tasks:
            pose_tokens = self._encode_pose_tokens(
                pose_images=pose_images,
                target_hw=(h_, w_),
                out_dtype=image_embeds.dtype,
            )
            if pose_tokens.shape[:2] != image_embeds.shape[:2]:
                raise RuntimeError(
                    f"Pose token shape mismatch: pose={tuple(pose_tokens.shape)}, "
                    f"image={tuple(image_embeds.shape)}"
                )
            image_embeds = self.pose_fusion_proj(
                torch.cat([image_embeds, pose_tokens], dim=-1)
            )

        return image_embeds
    
    def forward_img2pose(
        self,
        text_tokens: torch.Tensor,
        image_latents: torch.Tensor,
        motion_latents: torch.Tensor,
        t: torch.Tensor,
        pose_images: Optional[torch.Tensor] = None,
        return_motion_pred: bool = False,
        attention_mask: torch.Tensor = None,
        motion_lengths: List[int] = None,
        text_labels: torch.Tensor = None,
        motion_labels: torch.Tensor = None,
        image_modality_positions: torch.Tensor = None,
        motion_modality_positions: torch.Tensor = None,
        text_masks: torch.Tensor = None,
        image_masks: torch.Tensor = None,
        motion_masks: torch.Tensor = None,
        output_hidden_states: bool = True,
        max_seq_len: int = None,
        device: str = 'cuda:0',
        **kwargs,
    ):
        """
        Forward pass for Image-to-Pose (img2pose) task.

        The image is the conditioning input (similar to source motion); motion is
        the generation target. Image features get no time embedding; motion
        features get time embeddings.

        Args:
            text_tokens: [bs, seq_len] text token ids
            image_latents: [bs, 16, H, W] image latents from WAN21 VAE
            motion_latents: [latent_len, bs, motion_latent_dim] noisy motion latents
            t: [bs] diffusion timesteps for motion
            attention_mask: attention mask
            motion_lengths: list of motion lengths
            text_labels: [bs, seq_len] text labels (-100 for all in img2pose)
            motion_labels: [bs, latent_len, motion_latent_dim] target velocity
            image_modality_positions: [bs, 1, 2] (offset, length) for image
            motion_modality_positions: [bs, 1, 2] (offset, length) for motion
            text_masks: [bs, seq_len] text token mask
            image_masks: [bs, seq_len] image token mask
            motion_masks: [bs, seq_len] motion token mask
            
        Returns:
            (logits, loss_ntp, loss_motion) during training
        """
        input_embeds = self.showo.model.embed_tokens(text_tokens)
        dtype = input_embeds.dtype
        bs = text_tokens.shape[0]
        device = input_embeds.device
        
        # 1. Encode image features (frozen pipeline)
        image_embeds = self._encode_image_features(
            image_latents.to(dtype),
            task_name='img2pose',
            pose_images=pose_images,
        )
        
        # 2. Encode motion features using fused path
        num_motions = motion_latents.shape[1]
        motion_embeds = self.motion_embedder.forward_fused(motion_latents.to(dtype))  # [num_motions, latent_len, hidden_size]
        
        # 3. Get time embeddings (for motion only)
        if t.numel() == 1:
            t = t.expand(num_motions)
        elif t.shape[0] != num_motions:
            t = t[0].expand(num_motions)
        
        time_embeds = self.time_embed(t, dtype)
        if hasattr(self, 'time_embed_proj'):
            time_embeds_proj = self.time_embed_proj(time_embeds)
        else:
            time_embeds_proj = time_embeds
        
        # 4. Prepare motion labels
        if motion_labels is not None:
            new_motion_labels = torch.zeros(
                [bs, max_seq_len, self.config.motion_latent_dim],
                device=device, dtype=dtype
            )
        
        # 5. Place image features (NO time embed, the image is the condition)
        for i, modality_batch in enumerate(image_modality_positions):
            for j, (offset, length) in enumerate(modality_batch):
                if length == 0:
                    continue
                actual_len = min(length.item(), image_embeds.shape[1])
                input_embeds[i, offset:offset + actual_len] = image_embeds[i, :actual_len]
        
        # 6. Place motion features (WITH time embed, motion is the generation target)
        for i, modality_batch in enumerate(motion_modality_positions):
            for j, (offset, length) in enumerate(modality_batch):
                if length == 0:
                    continue
                idx = i * motion_modality_positions.size(1) + j
                
                if self.config.add_time_embeds:
                    input_embeds[i, offset] = time_embeds_proj[idx]
                    input_embeds[i, offset + 1:offset + 1 + length - 1] = motion_embeds[idx, :max(length - 1, 0)]
                    
                    if motion_labels is not None:
                        motion_masks[i, offset] = 0  # Mask time embedding position
                        new_motion_labels[i, offset + 1:offset + 1 + length - 1] = motion_labels[idx, :max(length - 1, 0)]
                else:
                    input_embeds[i, offset:offset + length] = motion_embeds[idx, :length]
                    if motion_labels is not None:
                        new_motion_labels[i, offset:offset + length] = motion_labels[idx, :length]
        
        # 7. Create modality_mask for LoRA routing
        # image tokens → 0 (LoRA-A), motion tokens → 1 (LoRA-B), text → 0 (LoRA-A)
        modality_mask = None
        if self.lora_enabled:
            modality_mask = self._create_modality_mask(
                bs, input_embeds.shape[1], motion_modality_positions, device
            )
        
        # 8. LLM forward
        outputs = self.showo(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            modality_mask=modality_mask,
        )
        
        logits = outputs['logits']
        last_hidden_states = outputs['hidden_states'][-1]
        
        # 9. Motion Diffusion Head
        if hasattr(self, 'diff_proj'):
            last_hidden_states = self.diff_proj(last_hidden_states)
        
        position_ids = torch.arange(
            last_hidden_states.shape[1],
            device=last_hidden_states.device
        ).unsqueeze(0)
        
        for layer in self.motion_diffusion_head_a:
            last_hidden_states = layer(
                hidden_states=last_hidden_states,
                adaln_input=time_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                modality_positions=motion_modality_positions,
            )[0]
        
        v_pred = self.motion_diffusion_head_b(last_hidden_states, time_embeds, motion_modality_positions)

        v_pred_out = None
        if return_motion_pred or (text_labels is None and motion_labels is None):
            v_pred_list = []
            for i, modality_batch in enumerate(motion_modality_positions):
                for j, (offset, length) in enumerate(modality_batch):
                    if length == 0:
                        break
                    v_pred_list.append(v_pred[i, offset:offset + length])
            if len(v_pred_list) > 0:
                v_pred_out = torch.stack(v_pred_list)
                if self.config.add_time_embeds:
                    v_pred_out = v_pred_out[:, 1:, :]
                v_pred_out = v_pred_out.permute(1, 0, 2)

        # 10. Calculate losses
        if text_labels is not None and motion_labels is not None:
            loss_ntp = next_token_prediction(logits, text_labels, self.config.llm_vocab_size)
            loss_motion = motion_velocity_prediction(
                v_pred[:bs], new_motion_labels[:bs], motion_masks
            )
            if return_motion_pred:
                return logits, loss_ntp, loss_motion, v_pred_out
            return logits, loss_ntp, loss_motion
        elif motion_labels is not None:
            loss_motion = motion_velocity_prediction(
                v_pred[:bs], new_motion_labels[:bs], motion_masks
            )
            if return_motion_pred:
                return logits, loss_motion, v_pred_out
            return logits, loss_motion
        else:
            # Inference mode
            return logits, v_pred_out
    
    @torch.no_grad()
    def img2pose_generate(
        self,
        motion_latents: torch.Tensor,
        t: torch.Tensor,
        text_tokens: torch.Tensor,
        image_latents: torch.Tensor,
        attention_mask: torch.Tensor,
        image_modality_positions: torch.Tensor,
        motion_modality_positions: torch.Tensor,
        pose_images: Optional[torch.Tensor] = None,
        max_seq_len: int = None,
        guidance_scale: float = 0.0,
        **kwargs,
    ):
        """
        Generate motion velocity prediction for image-to-pose (img2pose).
        Called by the ODE sampler for flow matching generation.
        
        For CFG (guidance_scale > 0):
        - First bs samples in batch: conditional (real image features)
        - Last bs samples in batch: unconditional (zero image features)
        
        Args:
            motion_latents: [latent_len, 2*bs, motion_latent_dim] noisy motion latents
            t: diffusion timesteps
            text_tokens: [2*bs, seq_len] (cond + uncond, same token structure)
            image_latents: [2*bs, 16, H, W] (real images + zero images)
            attention_mask: [2*bs, ...] attention mask
            image_modality_positions: [2*bs, 1, 2] (offset, length)
            motion_modality_positions: [2*bs, 1, 2] (offset, length)
            max_seq_len: maximum sequence length
            guidance_scale: CFG guidance scale
            
        Returns:
            v: [latent_len, 2*bs, motion_latent_dim] velocity predictions
        """
        if guidance_scale > 0.0:
            # Single forward pass with doubled batch (cond + uncond)
            _, v = self.forward_img2pose(
                text_tokens=text_tokens,
                image_latents=image_latents,
                motion_latents=motion_latents,
                pose_images=pose_images,
                t=t,
                attention_mask=attention_mask,
                image_modality_positions=image_modality_positions,
                motion_modality_positions=motion_modality_positions,
                output_hidden_states=True,
                max_seq_len=max_seq_len,
            )
            
            # Split conditional and unconditional
            v_cond, v_uncond = torch.chunk(v, 2, dim=1)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
            return torch.cat([v, v], dim=1)
        else:
            _, v = self.forward_img2pose(
                text_tokens=text_tokens,
                image_latents=image_latents,
                motion_latents=motion_latents,
                pose_images=pose_images,
                t=t,
                attention_mask=attention_mask,
                image_modality_positions=image_modality_positions,
                motion_modality_positions=motion_modality_positions,
                output_hidden_states=True,
                max_seq_len=max_seq_len,
            )
            return v
    
    def forward_img2text(
        self,
        text_tokens: torch.Tensor,
        image_latents: torch.Tensor,
        pose_images: Optional[torch.Tensor] = None,
        attention_mask: torch.Tensor = None,
        text_labels: torch.Tensor = None,
        image_modality_positions: torch.Tensor = None,
        text_masks: torch.Tensor = None,
        image_masks: torch.Tensor = None,
        output_hidden_states: bool = False,
        max_seq_len: int = None,
        device: str = 'cuda:0',
        **kwargs,
    ):
        """
        Forward pass for Image-to-Text (img2text) task.

        The image is the conditioning input; the text description is the generation
        target (NTP loss).

        Args:
            text_tokens: [bs, seq_len] text token ids
            image_latents: [bs, 16, H, W] image latents from WAN21 VAE
            attention_mask: attention mask
            text_labels: [bs, seq_len] text labels for NTP loss
            image_modality_positions: [bs, 1, 2] (offset, length) for image
            text_masks: [bs, seq_len] text token mask
            image_masks: [bs, seq_len] image token mask
            
        Returns:
            (logits, loss_ntp) during training
        """
        input_embeds = self.showo.model.embed_tokens(text_tokens)
        dtype = input_embeds.dtype
        bs = text_tokens.shape[0]
        device = input_embeds.device
        
        # 1. Encode image features
        image_embeds = self._encode_image_features(
            image_latents.to(dtype),
            task_name='img2text',
            pose_images=pose_images,
        )
        
        # 2. Place image features (NO time embed for img2text condition)
        for i, modality_batch in enumerate(image_modality_positions):
            for j, (offset, length) in enumerate(modality_batch):
                if length == 0:
                    continue
                actual_len = min(length.item(), image_embeds.shape[1])
                input_embeds[i, offset:offset + actual_len] = image_embeds[i, :actual_len]
        
        # 3. Create modality_mask (all 0 for LoRA-A, no motion tokens)
        modality_mask = None
        if self.lora_enabled:
            modality_mask = torch.zeros(
                bs, input_embeds.shape[1],
                device=device, dtype=dtype
            )
        
        # 4. LLM forward
        outputs = self.showo(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            modality_mask=modality_mask,
        )
        
        logits = outputs['logits']
        
        # 5. Calculate NTP loss
        if text_labels is not None:
            loss_ntp = next_token_prediction(logits, text_labels, self.config.llm_vocab_size)
            return logits, loss_ntp
        else:
            return logits

    def forward_mgie(
        self,
        text_tokens: torch.Tensor,
        source_image_latents: torch.Tensor,
        target_image_latents: torch.Tensor,
        ref_motion_latents: torch.Tensor,
        t: torch.Tensor,
        source_pose_images: Optional[torch.Tensor] = None,
        attention_mask: torch.Tensor = None,
        text_labels: torch.Tensor = None,
        image_labels: torch.Tensor = None,
        source_image_modality_positions: torch.Tensor = None,
        ref_motion_modality_positions: torch.Tensor = None,
        target_image_modality_positions: torch.Tensor = None,
        text_masks: torch.Tensor = None,
        target_image_masks: torch.Tensor = None,
        output_hidden_states: bool = True,
        max_seq_len: int = None,
        device: str = 'cuda:0',
        **kwargs,
    ):
        """
        Motion-guided image editing:
          text + source image + reference motion -> target image (flow loss).
        """
        input_embeds = self.showo.model.embed_tokens(text_tokens)
        dtype = input_embeds.dtype
        bs = text_tokens.shape[0]
        device = input_embeds.device

        source_image_embeds = self._encode_image_features(
            source_image_latents.to(dtype),
            task_name='mgie',
            pose_images=source_pose_images,
        )
        target_image_embeds = self._encode_image_features(
            target_image_latents.to(dtype),
            task_name='mgie_target',
        )
        ref_motion_embeds = self.motion_embedder.forward_fused(ref_motion_latents.to(dtype))

        b, c, h, w = target_image_latents.shape
        p = self.config.patch_size
        h_, w_ = h // p, w // p

        if t.numel() == 1:
            t = t.expand(bs)
        elif t.shape[0] != bs:
            t = t[0].expand(bs)
        time_embeds = self.time_embed(t, dtype)
        if hasattr(self, 'time_embed_proj'):
            time_embeds_proj = self.time_embed_proj(time_embeds)
        else:
            time_embeds_proj = time_embeds

        if image_labels is not None:
            img_labels = rearrange(image_labels, 'b c h w -> b (h w) c')
            img_labels = img_labels.reshape(shape=(b, h_, w_, p, p, c))
            img_labels = img_labels.reshape(shape=(b, h_ * w_, p * p * c))
            new_image_labels = torch.zeros(
                [bs, max_seq_len, p * p * c], device=device, dtype=dtype
            )
            if target_image_masks is None:
                raise ValueError("target_image_masks is required when image_labels is provided.")
            target_image_masks = target_image_masks[:, :, None].repeat(1, 1, p * p * c)

        # Place source image condition tokens.
        for i, modality_batch in enumerate(source_image_modality_positions):
            for _, (offset, length) in enumerate(modality_batch):
                if length == 0:
                    continue
                actual_len = min(length.item(), source_image_embeds.shape[1])
                input_embeds[i, offset:offset + actual_len] = source_image_embeds[i, :actual_len]

        # Place reference motion condition tokens (no time embedding).
        for i, modality_batch in enumerate(ref_motion_modality_positions):
            for _, (offset, length) in enumerate(modality_batch):
                if length == 0:
                    continue
                actual_len = min(length.item(), ref_motion_embeds.shape[1])
                input_embeds[i, offset:offset + actual_len] = ref_motion_embeds[i, :actual_len]

        # Place target image generation tokens (with time embedding).
        for i, modality_batch in enumerate(target_image_modality_positions):
            for j, (offset, length) in enumerate(modality_batch):
                if length == 0:
                    continue
                idx = i * target_image_modality_positions.size(1) + j
                if self.config.add_time_embeds:
                    input_embeds[i, offset] = time_embeds_proj[idx]
                    input_embeds[i, offset + 1:offset + 1 + length - 1] = target_image_embeds[idx, :max(length - 1, 0)]
                    if image_labels is not None:
                        target_image_masks[i, offset] = 0
                        new_image_labels[i, offset + 1:offset + 1 + length - 1] = img_labels[idx, :max(length - 1, 0)]
                else:
                    input_embeds[i, offset:offset + length] = target_image_embeds[idx, :length]
                    if image_labels is not None:
                        new_image_labels[i, offset:offset + length] = img_labels[idx, :length]

        # LoRA routing: only reference motion tokens go to LoRA-B.
        modality_mask = None
        if self.lora_enabled:
            modality_mask = self._create_modality_mask(
                bs, input_embeds.shape[1], ref_motion_modality_positions, device
            )

        outputs = self.showo(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            modality_mask=modality_mask,
        )

        logits = outputs['logits']
        last_hidden_states = outputs['hidden_states'][-1]

        if hasattr(self, 'diff_proj'):
            last_hidden_states = self.diff_proj(last_hidden_states)
        position_ids = torch.arange(last_hidden_states.shape[1], device=last_hidden_states.device).unsqueeze(0)
        for layer in self.diffusion_head_a:
            last_hidden_states = layer(
                hidden_states=last_hidden_states,
                adaln_input=time_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                modality_positions=target_image_modality_positions,
            )[0]
        v_pred = self.diffusion_head_b(last_hidden_states, time_embeds, target_image_modality_positions)

        if text_labels is not None and image_labels is not None:
            loss_ntp = next_token_prediction(logits, text_labels, self.config.llm_vocab_size)
            loss_flow = velocity_prediction(v_pred, new_image_labels[:v_pred.shape[0]], target_image_masks)
            return logits, loss_ntp, loss_flow
        elif image_labels is not None:
            loss_flow = velocity_prediction(v_pred, new_image_labels[:v_pred.shape[0]], target_image_masks)
            return logits, loss_flow
        elif text_labels is not None:
            loss_ntp = next_token_prediction(logits, text_labels, self.config.llm_vocab_size)
            return logits, loss_ntp
        else:
            v_pred_ = []
            for i, modality_batch in enumerate(target_image_modality_positions):
                for _, (offset, length) in enumerate(modality_batch):
                    if length == 0:
                        break
                    v_pred_.append(v_pred[i, offset:offset + length])
            v_pred_ = torch.stack(v_pred_)

            if self.config.add_time_embeds:
                v_pred_ = v_pred_[:, 1:, :]

            v_pred_ = self.unpatchify(v_pred_, h_, w_, T=0)
            v_pred_ = rearrange(v_pred_, 'i j k -> i k j')
            v_pred_ = v_pred_.reshape(v_pred_.shape[0], c, h, w)
            return logits, v_pred_

    @torch.no_grad()
    def mgie_generate(
        self,
        target_image_latents=None,
        t=None,
        text_tokens=None,
        attention_mask=None,
        source_image_latents=None,
        ref_motion_latents=None,
        source_image_modality_positions=None,
        ref_motion_modality_positions=None,
        target_image_modality_positions=None,
        max_seq_len=None,
        guidance_scale: float = 0.0,
        **kwargs,
    ):
        if guidance_scale > 0.0:
            _, v = self.forward_mgie(
                text_tokens=text_tokens,
                source_image_latents=source_image_latents,
                target_image_latents=target_image_latents,
                ref_motion_latents=ref_motion_latents,
                t=t,
                source_pose_images=kwargs.get("source_pose_images"),
                attention_mask=attention_mask,
                source_image_modality_positions=source_image_modality_positions,
                ref_motion_modality_positions=ref_motion_modality_positions,
                target_image_modality_positions=target_image_modality_positions,
                output_hidden_states=True,
                max_seq_len=max_seq_len,
            )
            v_cond, v_uncond = torch.chunk(v, 2, dim=0)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
            return torch.cat([v, v], dim=0)
        else:
            _, v = self.forward_mgie(
                text_tokens=text_tokens,
                source_image_latents=source_image_latents,
                target_image_latents=target_image_latents,
                ref_motion_latents=ref_motion_latents,
                t=t,
                source_pose_images=kwargs.get("source_pose_images"),
                attention_mask=attention_mask,
                source_image_modality_positions=source_image_modality_positions,
                ref_motion_modality_positions=ref_motion_modality_positions,
                target_image_modality_positions=target_image_modality_positions,
                output_hidden_states=True,
                max_seq_len=max_seq_len,
            )
            return v
    
    # ==================== Original Image/Video Methods (unchanged) ====================
    
    def forward_und_only(
        self,
        text_tokens=None,
        image_latents=None,
        t=None,
        attention_mask=None,
        text_masks=None,
        image_masks=None,
        text_labels=None,
        image_labels=None,
        modality_positions=None,
        output_hidden_states=True,
        max_seq_len=None,
        device='cuda:0',
        **kwargs,
    ):
        """Forward pass for image/video understanding only."""
        T = 0
        input_embeds = self.showo.model.embed_tokens(text_tokens)
        dtype = input_embeds.dtype
        if len(image_latents.shape) != 4:
            b, c, T, h, w = image_latents.shape
        else:
            b, c, h, w = image_latents.shape

        if T == 0:
            image_embeds_und = self.image_embedder_und(image_latents.to(dtype))
            image_embeds_gen = self.image_embedder_gen(image_latents.to(dtype))
        else:
            image_latents = rearrange(image_latents, 'b c t h w -> (b t) c h w')
            image_embeds_und = self.image_embedder_und(image_latents.to(dtype))
            image_embeds_und = image_embeds_und.reshape(b, T, -1, self.config.clip_latent_dim)
            image_embeds_und = rearrange(image_embeds_und, 'b t l d -> (b t) l d')

            image_embeds_gen = self.image_embedder_gen(image_latents.to(dtype))
            image_embeds_gen = image_embeds_gen.reshape(b, T, -1, self.config.hidden_size)
            image_embeds_gen = rearrange(image_embeds_gen, 'b t l d -> b (t l) d')

        # Semantic layers
        p = self.config.patch_size
        h_, w_ = h // p, w // p
        if self.position_embedding.weight.shape[0] == self.image_position_ids.shape[-1]:
            image_embeds_und = image_embeds_und + self.position_embedding(self.image_position_ids)
            image_embeds_und = self.und_trans(image_embeds_und)['last_hidden_state']
        else:
            image_embeds_und = image_embeds_und + interpolate_pos_encoding(
                self.config.clip_latent_dim, self.position_embedding, h_, w_, 1,
            )
            image_embeds_und = self.und_trans(image_embeds_und)['last_hidden_state']
        
        if T != 0:
            image_embeds_und = image_embeds_und.reshape(b, T, image_embeds_und.shape[1], -1)
            image_embeds_und = rearrange(image_embeds_und, 'b t l d -> b (t l) d')

        image_embeds = self.fusion_proj(torch.cat([image_embeds_und, image_embeds_gen], dim=-1))

        time_embeds = self.time_embed(t, dtype)
        if hasattr(self, 'time_embed_proj'):
            time_embeds_proj = self.time_embed_proj(time_embeds)
        else:
            time_embeds_proj = time_embeds

        for i, modality_batch in enumerate(modality_positions):
            for j, (offset, length) in enumerate(modality_batch):
                if self.config.add_time_embeds:
                    input_embeds[i, offset] = time_embeds_proj[i * modality_positions.size(1) + j]
                    input_embeds[i, offset + 1:offset + 1 + length - 1] = \
                        image_embeds[i * modality_positions.size(1) + j, :max(length - 1, 0)]
                else:
                    input_embeds[i, offset:offset + length] = image_embeds[i * modality_positions.size(1) + j, :length]

        # Image/video understanding tasks involve no motion tokens
        # If LoRA is enabled, pass an all-zero mask to enable LoRA-A (Text/Image path)
        modality_mask = None
        if self.lora_enabled:
            modality_mask = torch.zeros(
                input_embeds.shape[0],
                input_embeds.shape[1],
                device=input_embeds.device,
                dtype=input_embeds.dtype
            )
        
        outputs = self.showo(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            modality_mask=modality_mask,  # all zeros means everything uses LoRA-A
        )

        logits, last_hidden_states = outputs['logits'], outputs['hidden_states'][-1]

        if text_labels is not None:
            loss_ntp = next_token_prediction(logits, text_labels, self.config.llm_vocab_size)
            return logits, loss_ntp
        else:
            return logits

    def forward(
        self,
        text_tokens=None,
        image_latents=None,
        motion_latents=None,
        t=None,
        attention_mask=None,
        text_masks=None,
        image_masks=None,
        motion_masks=None,
        text_labels=None,
        image_labels=None,
        motion_labels=None,
        motion_lengths=None,
        modality_positions=None,
        first_frame_as_cond=False,
        only_denoise_last_image=False,
        guidance_scale=0.0,
        output_hidden_states=True,
        max_seq_len=None,
        device='cuda:0',
        data_type=None,
        **kwargs,
    ):
        """
        Unified forward pass supporting image, video, and motion modalities.
        
        Args:
            text_tokens: text token ids
            image_latents: image/video latents from image VAE
            motion_latents: motion latents from motion VAE
            t: diffusion timesteps
            attention_mask: attention mask
            text_masks: text token mask
            image_masks: image token mask  
            motion_masks: motion token mask
            text_labels: text labels for NTP loss
            image_labels: image labels for flow loss
            motion_labels: motion labels for flow loss
            motion_lengths: list of motion lengths
            modality_positions: (offset, length) pairs for each modality
            first_frame_as_cond: whether to use first frame as condition (video)
            only_denoise_last_image: only denoise last image (interleaved)
            guidance_scale: CFG guidance scale
            output_hidden_states: whether to output hidden states
            max_seq_len: maximum sequence length
            device: device string
            data_type: type of data ('t2m_motion', 'm2t', etc.)
        """
        # Route to appropriate forward function based on data type
        if data_type == 'mgie':
            return self.forward_mgie(
                text_tokens=text_tokens,
                source_image_latents=kwargs.get('source_image_latents'),
                target_image_latents=kwargs.get('target_image_latents', image_latents),
                ref_motion_latents=kwargs.get('ref_motion_latents'),
                t=t,
                source_pose_images=kwargs.get('source_pose_images'),
                attention_mask=attention_mask,
                text_labels=text_labels,
                image_labels=image_labels,
                source_image_modality_positions=kwargs.get('source_image_modality_positions'),
                ref_motion_modality_positions=kwargs.get('ref_motion_modality_positions'),
                target_image_modality_positions=kwargs.get('target_image_modality_positions'),
                text_masks=text_masks,
                target_image_masks=kwargs.get('target_image_masks'),
                output_hidden_states=output_hidden_states,
                max_seq_len=max_seq_len,
                device=device,
            )
        if data_type == 'img2pose':
            return self.forward_img2pose(
                text_tokens=text_tokens,
                image_latents=image_latents,
                motion_latents=motion_latents,
                pose_images=kwargs.get('pose_images'),
                return_motion_pred=bool(kwargs.get('return_motion_pred', False)),
                t=t,
                attention_mask=attention_mask,
                motion_lengths=motion_lengths,
                text_labels=text_labels,
                motion_labels=motion_labels,
                image_modality_positions=kwargs.get('image_modality_positions'),
                motion_modality_positions=kwargs.get('motion_modality_positions'),
                text_masks=text_masks,
                image_masks=image_masks,
                motion_masks=motion_masks,
                output_hidden_states=output_hidden_states,
                max_seq_len=max_seq_len,
                device=device,
            )
        elif data_type == 'img2text':
            return self.forward_img2text(
                text_tokens=text_tokens,
                image_latents=image_latents,
                pose_images=kwargs.get('pose_images'),
                attention_mask=attention_mask,
                text_labels=text_labels,
                image_modality_positions=kwargs.get('image_modality_positions'),
                text_masks=text_masks,
                image_masks=image_masks,
                output_hidden_states=output_hidden_states,
                max_seq_len=max_seq_len,
                device=device,
            )
        elif data_type == 't2m_motion':
            return self.forward_motion_gen(
                text_tokens=text_tokens,
                motion_latents=motion_latents,
                t=t,
                attention_mask=attention_mask,
                motion_lengths=motion_lengths,
                text_labels=text_labels,
                motion_labels=motion_labels,
                modality_positions=modality_positions,
                text_masks=text_masks,
                motion_masks=motion_masks,
                output_hidden_states=output_hidden_states,
                max_seq_len=max_seq_len,
                device=device,
                **kwargs,
            )
        elif data_type == 'm2t':
            return self.forward_motion_und(
                text_tokens=text_tokens,
                motion_latents=motion_latents,
                t=t,
                attention_mask=attention_mask,
                motion_lengths=motion_lengths,
                text_labels=text_labels,
                modality_positions=modality_positions,
                output_hidden_states=output_hidden_states,
                max_seq_len=max_seq_len,
                device=device,
                **kwargs,
            )
        elif data_type == 'motion_edit':
            # Motion Editing: source motion uses fused path, target motion uses gen path
            return self.forward_motion_edit(
                text_tokens=text_tokens,
                source_motion_latents=kwargs.get('source_motion_latents'),
                target_motion_latents=motion_latents,  # motion_latents is target
                t=t,
                attention_mask=attention_mask,
                source_motion_lengths=kwargs.get('source_motion_lengths'),
                target_motion_lengths=motion_lengths,
                text_labels=text_labels,
                motion_labels=motion_labels,
                source_modality_positions=kwargs.get('source_modality_positions'),
                target_modality_positions=modality_positions,
                text_masks=text_masks,
                motion_masks=motion_masks,
                output_hidden_states=output_hidden_states,
                max_seq_len=max_seq_len,
                device=device,
            )
        elif motion_latents is not None:
            # Default to motion generation if motion_latents is provided
            return self.forward_motion_gen(
                text_tokens=text_tokens,
                motion_latents=motion_latents,
                t=t,
                attention_mask=attention_mask,
                motion_lengths=motion_lengths,
                text_labels=text_labels,
                motion_labels=motion_labels,
                modality_positions=modality_positions,
                text_masks=text_masks,
                motion_masks=motion_masks,
                output_hidden_states=output_hidden_states,
                max_seq_len=max_seq_len,
                device=device,
                **kwargs,
            )
        elif image_latents is None:
            # Text-only
            logits = self.showo(input_ids=text_tokens, attention_mask=attention_mask)
            return logits
        else:
            # Image/Video processing (original logic)
            return self._forward_image_video(
                text_tokens=text_tokens,
                image_latents=image_latents,
                t=t,
                attention_mask=attention_mask,
                text_masks=text_masks,
                image_masks=image_masks,
                text_labels=text_labels,
                image_labels=image_labels,
                modality_positions=modality_positions,
                first_frame_as_cond=first_frame_as_cond,
                only_denoise_last_image=only_denoise_last_image,
                guidance_scale=guidance_scale,
                output_hidden_states=output_hidden_states,
                max_seq_len=max_seq_len,
                device=device,
                **kwargs,
            )
    
    def _forward_image_video(
        self,
        text_tokens=None,
        image_latents=None,
        t=None,
        attention_mask=None,
        text_masks=None,
        image_masks=None,
        text_labels=None,
        image_labels=None,
        modality_positions=None,
        first_frame_as_cond=False,
        only_denoise_last_image=False,
        guidance_scale=0.0,
        output_hidden_states=True,
        max_seq_len=None,
        device='cuda:0',
        **kwargs,
    ):
        """Original image/video forward logic (extracted for clarity)."""
        T = 0
        input_embeds = self.showo.model.embed_tokens(text_tokens)
        dtype = input_embeds.dtype
        if len(image_latents.shape) != 4:
            b, c, T, h, w = image_latents.shape
        else:
            b, c, h, w = image_latents.shape

        # Dual-path extraction
        if T == 0:
            image_embeds_und = self.image_embedder_und(image_latents.to(dtype))
            image_embeds_gen = self.image_embedder_gen(image_latents.to(dtype))
        else:
            image_latents = rearrange(image_latents, 'b c t h w -> (b t) c h w')
            image_embeds_und = self.image_embedder_und(image_latents.to(dtype))
            image_embeds_und = image_embeds_und.reshape(b, T, -1, self.config.clip_latent_dim)
            image_embeds_und = rearrange(image_embeds_und, 'b t l d -> (b t) l d')

            image_embeds_gen = self.image_embedder_gen(image_latents.to(dtype))
            image_embeds_gen = image_embeds_gen.reshape(b, T, -1, self.config.hidden_size)
            image_embeds_gen = rearrange(image_embeds_gen, 'b t l d -> b (t l) d')

        # Semantic layers
        p = self.config.patch_size
        h_, w_ = h // p, w // p
        if self.position_embedding.weight.shape[0] == self.image_position_ids.shape[-1]:
            image_embeds_und = image_embeds_und + self.position_embedding(self.image_position_ids)
            image_embeds_und = self.und_trans(image_embeds_und)['last_hidden_state']
        else:
            image_embeds_und = image_embeds_und + interpolate_pos_encoding(
                self.config.clip_latent_dim, self.position_embedding, h_, w_, 1,
            )
            image_embeds_und = self.und_trans(image_embeds_und)['last_hidden_state']
        
        if T != 0:
            image_embeds_und = image_embeds_und.reshape(b, T, image_embeds_und.shape[1], -1)
            image_embeds_und = rearrange(image_embeds_und, 'b t l d -> b (t l) d')

        # Fusion
        image_embeds = self.fusion_proj(torch.cat([image_embeds_und, image_embeds_gen], dim=-1))

        if image_labels is not None:
            if T == 0:
                image_labels = rearrange(image_labels, 'b c h w -> b (h w) c')
                image_labels = image_labels.reshape(shape=(b, h_, w_, p, p, c))
                image_labels = image_labels.reshape(shape=(b, h_ * w_, p * p * c))
            else:
                image_labels = rearrange(image_labels, 'b c t h w -> b (t h w) c')
                image_labels = image_labels.reshape(shape=(b, T, h_, w_, p, p, c))
                image_labels = image_labels.reshape(shape=(b, T * h_ * w_, p * p * c))

        time_embeds = self.time_embed(t, dtype)
        if hasattr(self, 'time_embed_proj'):
            time_embeds_proj = self.time_embed_proj(time_embeds)
        else:
            time_embeds_proj = time_embeds

        # Structure embeddings
        if image_labels is not None:
            new_image_labels = torch.zeros([b, max_seq_len, p * p * c], device=device, dtype=dtype)
            image_masks = image_masks[:, :, None].repeat(1, 1, p * p * c)

        for i, modality_batch in enumerate(modality_positions):
            for j, (offset, length) in enumerate(modality_batch):
                if self.config.add_time_embeds:
                    input_embeds[i, offset] = time_embeds_proj[i * modality_positions.size(1) + j]
                    input_embeds[i, offset + 1:offset + 1 + length - 1] = image_embeds[
                        i * modality_positions.size(1) + j, :max(length - 1, 0)]
                    if image_labels is not None:
                        image_masks[i, offset] = 0
                        new_image_labels[i, offset + 1:offset + 1 + length - 1] = image_labels[
                            i * modality_positions.size(1) + j, :max(length - 1, 0)]
                else:
                    input_embeds[i, offset:offset + length] = image_embeds[i * modality_positions.size(1) + j, :length]
                    if image_labels is not None:
                        new_image_labels[i, offset:offset + length] = image_labels[
                            i * modality_positions.size(1) + j, :length]

        # Image/video generation tasks involve no motion tokens
        # If LoRA is enabled, pass an all-zero mask to enable LoRA-A (Text/Image path)
        modality_mask = None
        if self.lora_enabled:
            modality_mask = torch.zeros(
                input_embeds.shape[0],
                input_embeds.shape[1],
                device=input_embeds.device,
                dtype=input_embeds.dtype
            )
        
        outputs = self.showo(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            modality_mask=modality_mask,  # all zeros means everything uses LoRA-A
        )

        logits, last_hidden_states = outputs['logits'], outputs['hidden_states'][-1]

        # Diffusion head
        if hasattr(self, 'diff_proj'):
            last_hidden_states = self.diff_proj(last_hidden_states)
        position_ids = torch.arange(last_hidden_states.shape[1], device=last_hidden_states.device).unsqueeze(0)
        for layer in self.diffusion_head_a:
            last_hidden_states = layer(
                hidden_states=last_hidden_states,
                adaln_input=time_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                modality_positions=modality_positions,
            )[0]
        v_pred = self.diffusion_head_b(last_hidden_states, time_embeds, modality_positions)

        # Calculate losses
        if text_labels is not None and image_labels is not None:
            loss_ntp = next_token_prediction(logits, text_labels, self.config.llm_vocab_size)
            loss_flow = velocity_prediction(v_pred, new_image_labels[:v_pred.shape[0]], image_masks)
            return logits, loss_ntp, loss_flow
        elif image_labels is not None:
            loss_flow = velocity_prediction(v_pred, new_image_labels[:v_pred.shape[0]], image_masks)
            return logits, loss_flow
        elif text_labels is not None:
            loss_ntp = next_token_prediction(logits, text_labels, self.config.llm_vocab_size)
            return logits, loss_ntp
        else:
            # Inference
            v_pred_ = []
            num_imgs = 0
            for i, modality_batch in enumerate(modality_positions):
                for j, (offset, length) in enumerate(modality_batch):
                    if length == 0:
                        break
                    else:
                        v_pred_.append(v_pred[i, offset:offset + length])
                        num_imgs += 1
            v_pred_ = torch.stack(v_pred_)

            if self.config.add_time_embeds:
                v_pred_ = v_pred_[:, 1:, :]

            v_pred_ = self.unpatchify(v_pred_, h_, w_, T=T)

            if T == 0:
                v_pred_ = rearrange(v_pred_, 'i j k -> i k j')
                v_pred_ = v_pred_.reshape(num_imgs, c, h, w)
            else:
                v_pred_ = rearrange(v_pred_, 'b t l c -> b c t l')
                v_pred_ = v_pred_.reshape(num_imgs, c, T, h, w)

            if first_frame_as_cond:
                v_pred_ = torch.cat([
                    torch.zeros_like(v_pred_)[:, :, :1],
                    v_pred_[:, :, 1:]
                ], dim=2)

            if only_denoise_last_image:
                if guidance_scale > 0:
                    v_pred_cond, v_pred_uncond = torch.chunk(v_pred_, 2)
                    v_pred_cond = torch.cat([
                        torch.zeros_like(v_pred_cond)[:-1, :, :],
                        v_pred_cond[-1:, :, :]
                    ], dim=0)
                    v_pred_uncond = torch.cat([
                        torch.zeros_like(v_pred_uncond)[:-1, :, :],
                        v_pred_uncond[-1:, :, :]
                    ], dim=0)
                    v_pred_ = torch.cat([v_pred_cond, v_pred_uncond], dim=0)
                else:
                    v_pred_ = torch.cat([
                        torch.zeros_like(v_pred_)[:-1, :, :],
                        v_pred_[-1:, :, :]
                    ], dim=0)

            return logits, v_pred_

    @torch.no_grad()
    def t2i_generate(
        self,
        image_latents=None,
        t=None,
        text_tokens=None,
        attention_mask=None,
        modality_positions=None,
        first_frame_as_cond=False,
        only_denoise_last_image=False,
        max_seq_len=None,
        guidance_scale=0.0,
        **kwargs,
    ):
        """Generate image/video (original method)."""
        if guidance_scale > 0.0:
            if t.shape[-1] != text_tokens.shape[0]:
                t_cond, t_uncond = torch.chunk(t, 2)
                t_cond[:-1] = 1.0
                t_uncond[:-1] = 1.0
                t = torch.cat([t_cond, t_uncond])
            _, v = self(text_tokens,
                        image_latents=image_latents,
                        t=t,
                        attention_mask=attention_mask,
                        modality_positions=modality_positions,
                        first_frame_as_cond=first_frame_as_cond,
                        only_denoise_last_image=only_denoise_last_image,
                        guidance_scale=guidance_scale,
                        output_hidden_states=True,
                        max_seq_len=max_seq_len)
            v_cond, v_uncond = torch.chunk(v, 2)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
            return torch.cat([v, v], dim=0)
        else:
            if t.shape[-1] != text_tokens.shape[0]:
                t[:-1] = 1.0
            _, v = self(text_tokens,
                        image_latents=image_latents,
                        t=t,
                        attention_mask=attention_mask,
                        modality_positions=modality_positions,
                        first_frame_as_cond=first_frame_as_cond,
                        only_denoise_last_image=only_denoise_last_image,
                        guidance_scale=guidance_scale,
                        output_hidden_states=True,
                        max_seq_len=max_seq_len)
            return v

    @torch.no_grad()
    def mmu_generate(
        self,
        input_embeds=None,
        attention_mask=None,
        max_new_tokens=100,
        temperature=1.0,
        top_k=None,
        eos_token=None
    ):
        """Generate text from image/video (original method)."""
        device = input_embeds.device
        dtype = input_embeds.dtype
        result = []
        for _ in range(max_new_tokens):
            logits = self.showo(inputs_embeds=input_embeds, attention_mask=attention_mask.to(dtype))['logits']

            L = attention_mask.shape[-1]
            attention_mask = attention_mask.squeeze()
            attention_mask_a = torch.hstack([
                attention_mask,
                torch.zeros((L, 1)).to(device) + torch.finfo(logits.dtype).min,
            ])
            attention_mask_b = torch.vstack([
                attention_mask_a,
                torch.hstack([attention_mask[-1, :], torch.tensor([0]).to(device)]).unsqueeze(0),
            ])
            attention_mask = attention_mask_b

            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            result.append(idx_next[0][0])
            idx_next_embeds = self.showo.model.embed_tokens(idx_next)
            input_embeds = torch.cat([input_embeds, idx_next_embeds], dim=1).to(dtype)

            if eos_token is not None and idx_next.cpu() == eos_token:
                break

        return result
