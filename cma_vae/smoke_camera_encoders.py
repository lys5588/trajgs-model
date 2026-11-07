"""Run CameraCMAVAE forward/backward with the actual DINOv3 ViT-B/16.

Run from the repository root:
    python -m cma_vae.smoke_camera_encoders --camera_dim 11

Pretrained weights are required by default. --random_backbone explicitly
tests architecture only and is recorded as such in the output report.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from omegaconf import OmegaConf
import torch

from .models.architectures.camera_cma_vae import CameraCMAVAE
from .models.architectures.camera_vision import ROOT
from .utils.camera import extri_intri_to_pose_encoding


def make_inputs(camera_dim, device):
    batch, steps, height, width = 2, 20, 48, 64
    c2w = torch.eye(4).repeat(batch, steps, 1, 1)
    angle = torch.linspace(0, 0.8, steps)
    c2w[:, :, 0, 0] = c2w[:, :, 1, 1] = angle.cos()
    c2w[:, :, 0, 1] = -angle.sin()
    c2w[:, :, 1, 0] = angle.sin()
    c2w[:, :, 0, 3] = torch.linspace(-1, 1, steps)
    c2w[1, :, 1, 3] = 1
    intrinsics = torch.tensor([[50., 0., 32.], [0., 48., 24.], [0., 0., 1.]])
    intrinsics = intrinsics.expand(batch, steps, 3, 3)
    kind = "quaternion_9d" if camera_dim == 9 else "rotation_6d_11d"
    cameras = extri_intri_to_pose_encoding(
        c2w, intrinsics, (height, width), pose_encoding_type=kind, extrinsics_type="c2w",
    ).to(device)
    images = torch.rand(batch, steps, 3, height, width)
    images[:, :, 0] *= torch.linspace(0.1, 1, steps)[None, :, None, None]
    boxes = torch.tensor([4., 8., 60., 44.]).repeat(batch, steps, 1)
    return cameras, images, boxes


def check_distribution(latent, distribution, batch):
    assert latent.shape == distribution.loc.shape == distribution.scale.shape == (20, batch, 256)
    assert torch.isfinite(latent).all()
    assert torch.isfinite(distribution.loc).all() and torch.isfinite(distribution.scale).all()
    assert (distribution.scale > 0).all()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, default=ROOT / "cma_vae/configs/config_camera_vae.yaml")
    parser.add_argument("--camera_dim", type=int, choices=(9, 11))
    parser.add_argument("--dinov3_weights", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--random_backbone", action="store_true",
                        help="Explicit architecture-only test: random DINOv3 weights, not pretrained validation")
    parser.add_argument("--output", type=Path, help="Optional JSON verification report")
    args = parser.parse_args()
    torch.manual_seed(2026)
    torch.set_num_threads(4)
    config = OmegaConf.to_container(OmegaConf.load(args.cfg), resolve=True)
    if args.camera_dim is not None:
        config["camera_dim"] = args.camera_dim
    if args.dinov3_weights is not None:
        config["dinov3_weights"] = str(args.dinov3_weights)
    repo = Path(config["dinov3_repo"])
    if not repo.is_absolute():
        repo = ROOT / repo
    config["dinov3_repo"] = str(repo)
    if args.random_backbone:
        config["vision_backbone"] = torch.hub.load(
            str(repo), "dinov3_vitb16", source="local", pretrained=False,
        )
    model = CameraCMAVAE(**config).to(args.device)
    model.vision_backbone.load_backbone()  # Fail clearly before any smoke claims.
    cameras, images, boxes = make_inputs(model.camera_dim, args.device)
    model.train()
    latent_c, qc = model.encode_camera(cameras)
    latent_j, qj = model.encode_joint(cameras, images, boxes)
    check_distribution(latent_c, qc, 2)
    check_distribution(latent_j, qj, 2)
    loss = sum(q.loc.square().mean() + q.scale.log().square().mean() for q in (qc, qj))
    loss.backward()
    gradients = {}
    for name in ("camera_embedding", "camera_encoder", "vision_projection", "joint_encoder"):
        grads = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), name
        assert any(g.abs().sum() > 0 for g in grads), name
        gradients[name] = "finite_nonzero"
    assert all(not p.requires_grad and p.grad is None for p in model.vision_backbone.parameters())
    assert not model.vision_backbone.backbone.training

    model.eval()
    with torch.no_grad():
        no_images = model(cameras)
        assert no_images["latent_joint"] is None and no_images["joint_indices"].numel() == 0
        has_images = torch.tensor([False, True], device=args.device)
        mixed = model(cameras, images, boxes, has_images)
        assert mixed["joint_indices"].tolist() == [1]
        check_distribution(mixed["latent_joint"], mixed["dist_joint"], 1)
        torch.testing.assert_close(mixed["latent"][:, 0], mixed["latent_camera"][:, 0])
        torch.testing.assert_close(mixed["latent"][:, 1], mixed["latent_joint"][:, 0])
        decoded = model.decode(mixed["latent"])
        assert decoded.shape == (2, 20, model.camera_dim) and torch.isfinite(decoded).all()
    weights = Path(config["dinov3_weights"])
    weights_sha256 = None
    if not args.random_backbone:
        with weights.open("rb") as checkpoint:
            weights_sha256 = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    report = {
        "camera_dim": model.camera_dim, "device": args.device,
        "backbone": "dinov3_vitb16", "backbone_revision": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "weights_status": "random_architecture_only" if args.random_backbone else "pretrained_loaded",
        "pretrained_validation": "NOT_RUN" if args.random_backbone else "PASS",
        "weights_path": None if args.random_backbone else str(weights.resolve()),
        "weights_sha256": weights_sha256,
        "camera_latent_shape": list(latent_c.shape), "joint_latent_shape": list(latent_j.shape),
        "decoder_shape": list(decoded.shape), "gradients": gradients,
        "dinov3_frozen_eval": True, "camera_only": "PASS", "mixed_batch": "PASS",
        "input_data": "synthetic calibrated cameras and frame-varying RGB images",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
