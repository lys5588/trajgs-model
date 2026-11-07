"""Per-frame bbox crops and a frozen, locally loaded DINOv3 ViT-B/16."""

from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DINOV3_REPO = ROOT / "submodules" / "DINOv3"
DEFAULT_DINOV3_WEIGHTS = Path(
    "/mnt/d/model/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
)


def crop_bbox_images(images: Tensor, bboxes: Tensor) -> Tensor:
    """RGB [B,20,3,H,W] + pixel xyxy [B,20,4] -> [B,20,3,224,224].

    Returns ImageNet-normalized FP32 crops. Box coordinates are annotations,
    not differentiable inputs. Pixel slicing uses floor/ceil, then clipping.
    Resize stretches both dimensions directly; it does not preserve aspect.
    """
    if not isinstance(images, Tensor) or images.ndim != 5:
        raise ValueError("images must be a tensor shaped [B,20,3,H,W]")
    batch, steps, channels, height, width = images.shape
    if batch < 1 or steps != 20 or channels != 3 or min(height, width) < 1:
        raise ValueError("images must have nonempty B,H,W and shape [B,20,3,H,W]")
    if not isinstance(bboxes, Tensor) or bboxes.shape != (batch, steps, 4):
        raise ValueError("bboxes must have shape [B,20,4]; all 20 frames are required")
    if bboxes.dtype == torch.bool or bboxes.is_complex():
        raise TypeError("bboxes must contain real pixel xyxy coordinates")
    boxes = bboxes.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(boxes).all():
        raise ValueError("bboxes must contain finite coordinates")
    if (boxes[..., 2:] <= boxes[..., :2]).any():
        raise ValueError("bboxes must have x_max > x_min and y_max > y_min")
    if images.dtype == torch.uint8:
        pixels = images.float().div(255)
    elif images.is_floating_point():
        if not torch.isfinite(images).all() or (images < 0).any() or (images > 1).any():
            raise ValueError("floating RGB images must contain finite values in [0,1]")
        pixels = images.float()
    else:
        raise TypeError("images must be uint8 or floating point RGB in [0,1]")

    starts = boxes[..., :2].floor().clamp_min(0)
    ends = boxes[..., 2:].ceil().clamp_min(0)
    limits = torch.tensor([width, height])
    starts = torch.minimum(starts, limits).long()
    ends = torch.minimum(ends, limits).long()
    if (ends <= starts).any():
        raise ValueError("bbox is empty after clipping to image boundaries")
    crops = []
    for b in range(batch):
        for t in range(steps):
            x1, y1 = starts[b, t].tolist()
            x2, y2 = ends[b, t].tolist()
            crop = pixels[b, t, :, y1:y2, x1:x2].unsqueeze(0)
            crops.append(F.interpolate(
                crop, size=(224, 224), mode="bilinear",
                align_corners=False, antialias=True,
            ))
    crops = torch.cat(crops).reshape(batch, steps, 3, 224, 224)
    mean = crops.new_tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    std = crops.new_tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    return (crops - mean) / std


class FrozenDINOv3(nn.Module):
    """Load pretrained weights lazily, so camera-only batches need no backbone.

    No implicit download or random-weight fallback. An injected backbone is
    supported for explicitly labelled architecture tests. It must implement
    forward_features(...)["x_norm_clstoken"] with width 768.
    """

    def __init__(self, repo_path=DEFAULT_DINOV3_REPO,
                 weights_path=DEFAULT_DINOV3_WEIGHTS, chunk_size: int = 20,
                 backbone: Optional[nn.Module] = None):
        super().__init__()
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("dinov3_chunk_size must be a positive integer")
        self.repo_path = Path(repo_path)
        if not self.repo_path.is_absolute():
            self.repo_path = ROOT / self.repo_path
        self.weights_path = Path(weights_path)
        self.chunk_size = chunk_size
        # Tracks .to(device/dtype) even before lazy backbone construction.
        self.register_buffer("_placement", torch.empty(0), persistent=False)
        self.backbone = backbone
        if backbone is not None:
            self._freeze()

    def _freeze(self):
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone is not None:
            self.backbone.eval()
        return self

    def load_backbone(self):
        if self.backbone is None:
            if not (self.repo_path / "hubconf.py").is_file():
                raise FileNotFoundError(
                    f"DINOv3 source missing: {self.repo_path}. "
                    "Run git submodule update --init submodules/DINOv3."
                )
            if not self.weights_path.is_file():
                raise FileNotFoundError(
                    f"DINOv3 pretrained weights missing: {self.weights_path}. "
                    "Download the official ViT-B/16 LVD-1689M .pth weights or "
                    "pass --dinov3_weights PATH. No random fallback is used."
                )
            self.backbone = torch.hub.load(
                str(self.repo_path), "dinov3_vitb16", source="local",
                weights=str(self.weights_path), pretrained=True,
            ).to(device=self._placement.device, dtype=self._placement.dtype)
            self._freeze()
        return self.backbone

    def forward(self, crops: Tensor) -> Tensor:
        if crops.ndim != 5 or crops.shape[1:] != (20, 3, 224, 224):
            raise ValueError("DINOv3 crops must have shape [B,20,3,224,224]")
        backbone = self.load_backbone()
        backbone.eval()
        flat = crops.flatten(0, 1)
        chunks = []
        # no_grad rather than inference_mode: the downstream trainable Linear
        # must be able to save these ordinary tensors for its weight gradient.
        with torch.no_grad():
            for chunk in flat.split(self.chunk_size):
                chunk = chunk.to(device=self._placement.device, dtype=self._placement.dtype)
                cls = backbone.forward_features(chunk)["x_norm_clstoken"]
                if cls.shape != (chunk.shape[0], 768):
                    raise ValueError("Expected DINOv3 ViT-B/16 CLS features [N,768]")
                chunks.append(cls)
        features = torch.cat(chunks)
        if not torch.isfinite(features).all():
            raise ValueError("DINOv3 returned non-finite CLS features")
        return features.reshape(crops.shape[0], 20, 768)
