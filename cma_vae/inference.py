"""Inference script for CMA-VAE (Cross-Modal Aligned Motion VAE).

Run from the cma_vae/ directory. See README.md for usage.
"""
import argparse
import os
import sys
from collections import OrderedDict
from os.path import join as pjoin

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.architectures.mld_vae_kl_align import CMAVAE
from models.architectures.vision_utils import recover_from_ric


class _AblationCfg:
    """Minimal ablation config required by CMAVAE."""
    PE_TYPE = "mld"


def detect_downsample_rate(vae_state_dict: dict) -> int:
    """Infer downsample_rate from weights (rate=1 uses parameter-free Identity)."""
    conv_keys = [k for k in vae_state_dict if k.startswith("motion_downsample.conv")
                 and k.endswith(".weight") and vae_state_dict[k].dim() == 3]
    if len(conv_keys) == 0:
        return 1
    # each stride-2 Conv1d halves the temporal resolution
    return 2 ** len(conv_keys)


def load_pretrained_vae(checkpoint_path: str, device: str = "cpu") -> CMAVAE:
    """
    Load CMAVAE from a checkpoint. The HRNet vision backbone is dropped
    since it is only needed during training.
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        # Checkpoints saved by train.py embed config objects in their
        # hyper_parameters, which weights_only=True refuses to unpickle.
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    vae_sd = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("vae.") and not k.startswith("vae.vision_backbone."):
            vae_sd[k[len("vae."):]] = v

    if len(vae_sd) == 0:
        raise ValueError(f"No 'vae.*' weights found in {checkpoint_path}")

    downsample_rate = detect_downsample_rate(vae_sd)
    nfeats = vae_sd["motion_embedding.weight"].shape[1]
    latent_dim = vae_sd["motion_embedding.weight"].shape[0]
    vision_dim = vae_sd["joint_dist_layer.weight"].shape[1] - latent_dim

    print(f"[load_vae] {checkpoint_path}")
    print(f"[load_vae] nfeats={nfeats} latent_dim={latent_dim} "
          f"vision_dim={vision_dim} downsample_rate={downsample_rate} "
          f"(epoch={ckpt.get('epoch', '?')})")

    vae = CMAVAE(
        ablation=_AblationCfg(),
        nfeats=nfeats,
        latent_dim=[1, latent_dim],
        ff_size=1024,
        num_layers=9,
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        position_embedding="learned",
        arch="all_encoder",
        downsample_rate=downsample_rate,
        vision_dim=vision_dim,
        hrnet_checkpoint=None,
    )

    missing, unexpected = vae.load_state_dict(vae_sd, strict=False)
    # vision_backbone is intentionally dropped; everything else must match
    real_missing = [k for k in missing if not k.startswith("vision_backbone.")]
    assert not real_missing, f"Missing keys: {real_missing}"
    assert not unexpected, f"Unexpected keys: {unexpected}"

    vae.eval().to(device)
    n_params = sum(p.numel() for p in vae.parameters()) / 1e6
    print(f"[load_vae] loaded OK, {n_params:.1f}M params")
    return vae


@torch.no_grad()
def reconstruct(vae: CMAVAE,
                motion_raw: np.ndarray,
                mean: np.ndarray,
                std: np.ndarray,
                device: str = "cpu",
                use_dist_mean: bool = True):
    """
    Encode-decode reconstruction of a single motion sequence.

    Args:
        motion_raw: [T, 269] un-normalized features (new_joint_vecs format)
        mean/std:   [269] dataset Mean.npy / Std.npy
        use_dist_mean: use the posterior mean (deterministic) instead of sampling
    Returns:
        recon_raw:  [T, 269] reconstructed features (de-normalized)
        latent:     [L, latent_dim] latent codes
    """
    T = motion_raw.shape[0]
    # Same zero-std handling as the training datasets
    std = np.where(std == 0, 1.0, std)
    x = (motion_raw - mean) / std
    x = torch.from_numpy(x).float().unsqueeze(0).to(device)
    lengths = [T]

    latent, dist = vae.encode_motion(x, lengths)
    if use_dist_mean:
        latent = dist.loc

    recon = vae.decode(latent, lengths)
    recon_raw = recon.squeeze(0).cpu().numpy() * std + mean
    return recon_raw, latent.squeeze(1).cpu().numpy()


def feats_to_joints(feats_raw: np.ndarray, njoints: int = 22) -> np.ndarray:
    return recover_from_ric(torch.from_numpy(feats_raw).float(), njoints).numpy()


def main():
    parser = argparse.ArgumentParser(description="CMA-VAE inference")
    parser.add_argument("--ckpt", required=True, help="checkpoint path (.ckpt)")
    parser.add_argument("--motion", required=True,
                        help="motion npy file [T, 269] (new_joint_vecs format)")
    parser.add_argument("--stats_dir", required=True,
                        help="directory containing Mean.npy / Std.npy of the dataset")
    parser.add_argument("--out_dir", default="./infer_out", help="output directory")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample", action="store_true",
                        help="sample from the posterior instead of using its mean")
    args = parser.parse_args()

    vae = load_pretrained_vae(args.ckpt, device=args.device)

    motion_raw = np.load(args.motion).astype(np.float32)
    if motion_raw.ndim != 2:
        raise ValueError(f"Expect [T, D] motion, got shape {motion_raw.shape}")
    mean = np.load(pjoin(args.stats_dir, "Mean.npy")).astype(np.float32)
    std = np.load(pjoin(args.stats_dir, "Std.npy")).astype(np.float32)
    print(f"[data] motion={motion_raw.shape} from {args.motion}")

    recon_raw, latent = reconstruct(
        vae, motion_raw, mean, std,
        device=args.device, use_dist_mean=not args.sample)

    # report errors in normalized feature space and joint space
    std_safe = np.where(std == 0, 1.0, std)
    mse_feats = float(np.mean(((recon_raw - motion_raw) / std_safe) ** 2))
    joints_gt = feats_to_joints(motion_raw)
    joints_rc = feats_to_joints(recon_raw)
    mpjpe = float(np.mean(np.linalg.norm(joints_gt - joints_rc, axis=-1)))
    print(f"[result] latent={latent.shape}  feats MSE(norm)={mse_feats:.6f}  "
          f"MPJPE={mpjpe * 1000:.2f} mm")

    os.makedirs(args.out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.motion))[0]
    np.save(pjoin(args.out_dir, f"{name}_recon_feats.npy"), recon_raw)
    np.save(pjoin(args.out_dir, f"{name}_recon_joints.npy"), joints_rc)
    np.save(pjoin(args.out_dir, f"{name}_latent.npy"), latent)
    print(f"[save] results saved to {args.out_dir}/{name}_*.npy")


if __name__ == "__main__":
    main()
