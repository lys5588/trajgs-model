"""
CMA-VAE (Cross-Modal Aligned Motion VAE) package.

Full training + inference code for the CMA-VAE described in the UniMotion
paper (pre-trained with DPA, Dual-Posterior KL Alignment). The main UniMotion
pipeline only needs the checkpoint this package produces
(`pretrained/cma_vae.ckpt`) — see README.md in this directory.

Historical code names (kept for checkpoint compatibility):
- `CMAVAE` was `MldVaeKLAlign` ("KL-Align Motion VAE", JMVAE-KL style)
- `CMAVAELosses` was `MLDLossesKLAlign`
- internal module file names (`mld_vae_kl_align.py`, `mld_kl_align.py`)
  are unchanged so that pickled references in old checkpoints stay valid.

Main components:
- VAE architecture with dual-posterior KL alignment
- Multi-modal data loading (motion + image)
- KL divergence losses with alignment
- Training (train.py) and inference (inference.py) entry points
"""

__version__ = "1.0.0"
__description__ = "CMA-VAE (Cross-Modal Aligned Motion VAE) for UniMotion"

__all__ = [
    # preferred (paper) names
    "CMAVAE",
    "CMAVAELosses",
    # legacy aliases (checkpoint compatibility)
    "MldVaeKLAlign",
    "MLDLossesKLAlign",
    "KLLoss",
    "KLAlignLoss",
]


def __getattr__(name):
    if name in {"CMAVAE", "MldVaeKLAlign"}:
        from .models.architectures.mld_vae_kl_align import CMAVAE

        return CMAVAE

    if name in {"CMAVAELosses", "MLDLossesKLAlign", "KLLoss", "KLAlignLoss"}:
        from .models.losses.mld_kl_align import (
            CMAVAELosses,
            KLLoss,
            KLAlignLoss,
        )

        return {
            "CMAVAELosses": CMAVAELosses,
            "MLDLossesKLAlign": CMAVAELosses,
            "KLLoss": KLLoss,
            "KLAlignLoss": KLAlignLoss,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
