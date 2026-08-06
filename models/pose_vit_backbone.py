# coding=utf-8
"""
Minimal TokenHMR ViT backbone adapter for img2pose.

This module mirrors the UniPose/TokenHMR ViT structure closely enough to
load TokenHMR checkpoints and output dense pose-aware feature maps.
"""

import math
import os
import warnings
from functools import partial
from itertools import repeat
import collections.abc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        bsz, n_tokens, _ = x.shape
        qkv = self.qkv(x).reshape(bsz, n_tokens, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(bsz, n_tokens, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path_prob=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_head_dim=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_head_dim=attn_head_dim,
        )
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patch_shape = (int(img_size[0] // patch_size[0] * ratio), int(img_size[1] // patch_size[1] * ratio))
        self.origin_patch_shape = (int(img_size[0] // patch_size[0]), int(img_size[1] // patch_size[1]))
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=(patch_size[0] // ratio),
            padding=4 + 2 * (ratio // 2 - 1),
        )

    def forward(self, x, **kwargs):
        x = self.proj(x)
        h_out, w_out = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, (h_out, w_out)


class ViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        use_checkpoint=False,
        ratio=1,
        last_norm=True,
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.use_checkpoint = use_checkpoint
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            ratio=ratio,
        )
        num_patches = self.patch_embed.patch_shape[0] * self.patch_embed.patch_shape[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path_prob=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()
        trunc_normal_(self.pos_embed, std=0.02)

    def forward_features(self, x):
        bsz = x.shape[0]
        x, (h_out, w_out) = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        x = self.last_norm(x)
        x = x.permute(0, 2, 1).reshape(bsz, -1, h_out, w_out).contiguous()
        return x

    def forward(self, x):
        return self.forward_features(x)


def _trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_.")

    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    with torch.no_grad():
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _trunc_normal_(tensor, mean, std, a, b)


def build_and_load_tokenhmr_vit(ckpt_path: str, freeze: bool = True) -> ViT:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"TokenHMR checkpoint not found: {ckpt_path}")

    model = ViT(
        img_size=(256, 192),
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.55,
    )

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load TokenHMR checkpoint: {ckpt_path}. "
            f"Please verify the file is a complete PyTorch checkpoint. Original error: {exc}"
        ) from exc
    state_dict = ckpt.get("state_dict", ckpt)

    adapted = {}
    for key, value in state_dict.items():
        if "backbone" in key:
            adapted[key.split("backbone.", 1)[-1]] = value

    if len(adapted) == 0:
        raise RuntimeError(
            f"Checkpoint has no 'backbone.*' keys: {ckpt_path}. "
            "This does not look like a TokenHMR ViT checkpoint."
        )

    model.load_state_dict(adapted, strict=False)

    if freeze:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    return model


class PoseViTBackbone(nn.Module):
    """Paper: pose-aware vision backbone (frozen TokenHMR ViT-H)."""

    def __init__(
        self,
        ckpt_path: str,
        freeze_backbone: bool = True,
        input_size: int = 256,
        crop_width: int = 192,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.crop_width = int(crop_width)

        self.backbone = build_and_load_tokenhmr_vit(
            ckpt_path=ckpt_path,
            freeze=freeze_backbone,
        )

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("imagenet_mean", mean, persistent=False)
        self.register_buffer("imagenet_std", std, persistent=False)

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        # Input images are in [-1, 1] from dataset pipeline.
        x = ((images.float() + 1.0) * 0.5).clamp(0.0, 1.0)
        x = F.interpolate(
            x,
            size=(self.input_size, self.input_size),
            mode="bicubic",
            align_corners=False,
        )

        if self.crop_width < self.input_size:
            left = (self.input_size - self.crop_width) // 2
            right = left + self.crop_width
            x = x[:, :, :, left:right]

        x = (x - self.imagenet_mean) / self.imagenet_std
        return x

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(images)
        # Keep the patch-embed input dtype aligned with the backbone weights.
        # Preprocess intentionally uses fp32 for normalization, but inference may run
        # the backbone in bf16; mixed dtypes would fail at the first conv.
        backbone_param = next(self.backbone.parameters())
        x = x.to(device=backbone_param.device, dtype=backbone_param.dtype)
        return self.backbone(x)
