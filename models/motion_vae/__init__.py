from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import MethodType
from typing import Any, Dict, Mapping

from cma_vae.models.architectures.mld_vae_kl_align import CMAVAE, MldVaeKLAlign

__all__ = ["CMAVAE", "MldVaeKLAlign", "load_motion_vae", "load_motion_vae_state"]


class _Ablation:
    def __init__(self, pe_type: str = "mld"):
        self.PE_TYPE = pe_type


def _safe_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, Mapping):
        return node.get(key, default)
    return getattr(node, key, default)


def _safe_get_path(node: Any, path: str, default: Any = None) -> Any:
    cur = node
    for key in path.split("."):
        cur = _safe_get(cur, key, None)
        if cur is None:
            return default
    return cur


def _unique_int_indices(state_dict: Mapping[str, Any], prefix: str) -> int:
    indices = set()
    for key in state_dict.keys():
        if not key.startswith(prefix):
            continue
        parts = key.split(".")
        if len(parts) < 4:
            continue
        try:
            indices.add(int(parts[3]))
        except ValueError:
            continue
    return len(indices)


def _infer_downsample_rate_from_state(vae_state: Mapping[str, Any]) -> int:
    """
    Infer temporal downsample rate from module parameter structure when
    checkpoint metadata is unavailable.

    Mapping by parameterized stages:
    - rate=1: no learnable params in motion_downsample / decoder_upsample
    - rate=2: 1 stage => 2 parameterized submodules
    - rate=4: 2 stages => 4 parameterized submodules
    - rate=8: 3 stages => 6 parameterized submodules
    """
    downsample_indices = set()
    for key in vae_state.keys():
        if not key.startswith("motion_downsample.conv."):
            continue
        parts = key.split(".")
        if len(parts) < 4:
            continue
        try:
            downsample_indices.add(int(parts[2]))
        except ValueError:
            continue

    decoder_indices = set()
    for key in vae_state.keys():
        if not key.startswith("decoder_upsample.upsample."):
            continue
        parts = key.split(".")
        if len(parts) < 4:
            continue
        try:
            decoder_indices.add(int(parts[2]))
        except ValueError:
            continue

    param_stage_count = max(len(downsample_indices), len(decoder_indices))
    if param_stage_count == 0:
        return 1
    if param_stage_count == 2:
        return 2
    if param_stage_count == 4:
        return 4
    if param_stage_count == 6:
        return 8

    raise ValueError(
        "Unable to infer Motion VAE downsample_rate from checkpoint structure. "
        f"motion_downsample_indices={sorted(downsample_indices)}, "
        f"decoder_upsample_indices={sorted(decoder_indices)}"
    )


def _extract_vae_state_dict(ckpt: Mapping[str, Any]) -> Dict[str, Any]:
    state_dict = ckpt.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint missing `state_dict`.")

    vae_state = {k[4:]: v for k, v in state_dict.items() if k.startswith("vae.")}
    if vae_state:
        return vae_state

    if "motion_embedding.weight" in state_dict:
        return dict(state_dict)

    raise ValueError("No VAE weights found (expected keys starting with `vae.`).")


def _extract_model_config(ckpt: Mapping[str, Any], vae_state: Mapping[str, Any]) -> Dict[str, Any]:
    hp = ckpt.get("hyper_parameters", {})
    cfg = _safe_get(hp, "cfg", hp)

    model_cfg = _safe_get_path(cfg, "model", {})
    vae_cfg = _safe_get(model_cfg, "vae_config", {})
    train_ablation = _safe_get_path(cfg, "TRAIN.ABLATION", {})
    dataset_cfg = _safe_get_path(cfg, "DATASET", {})

    motion_embed = vae_state.get("motion_embedding.weight")
    inferred_latent_dim = int(motion_embed.shape[0]) if motion_embed is not None else 256
    inferred_nfeats = int(motion_embed.shape[1]) if motion_embed is not None else 269

    ff_w = vae_state.get("motion_encoder.input_blocks.0.linear1.weight")
    inferred_ff_size = int(ff_w.shape[0]) if ff_w is not None else 1024

    input_block_count = _unique_int_indices(vae_state, "motion_encoder.input_blocks.")
    inferred_num_layers = 2 * input_block_count + 1 if input_block_count > 0 else 9

    latent_dim = _safe_get(model_cfg, "latent_dim", [1, inferred_latent_dim])
    if isinstance(latent_dim, int):
        latent_dim = [1, latent_dim]
    elif not isinstance(latent_dim, (list, tuple)):
        latent_dim = [1, inferred_latent_dim]
    latent_dim = list(latent_dim)
    inferred_downsample_rate = _infer_downsample_rate_from_state(vae_state)

    model_out = {
        "pe_type": _safe_get(train_ablation, "PE_TYPE", "mld"),
        "nfeats": int(_safe_get(dataset_cfg, "NFEATS", inferred_nfeats)),
        "latent_dim": latent_dim,
        "ff_size": int(_safe_get(model_cfg, "ff_size", inferred_ff_size)),
        "num_layers": int(_safe_get(model_cfg, "num_layers", inferred_num_layers)),
        "num_heads": int(_safe_get(model_cfg, "num_head", _safe_get(model_cfg, "num_heads", 4))),
        "dropout": float(_safe_get(model_cfg, "droupout", _safe_get(model_cfg, "dropout", 0.1))),
        "activation": _safe_get(model_cfg, "activation", "gelu"),
        "normalize_before": bool(_safe_get(vae_cfg, "normalize_before", False)),
        "position_embedding": _safe_get(vae_cfg, "position_embedding", "learned"),
        "arch": _safe_get(vae_cfg, "arch", "all_encoder"),
        "downsample_rate": int(_safe_get(vae_cfg, "downsample_rate", inferred_downsample_rate)),
        "kl_align_weight": float(_safe_get(vae_cfg, "kl_align_weight", 0.001)),
        "vision_dim": int(_safe_get(vae_cfg, "vision_dim", 128)),
        "hrnet_channels": int(_safe_get(vae_cfg, "hrnet_channels", 256)),
        "hrnet_checkpoint": _safe_get(vae_cfg, "hrnet_checkpoint", None),
        "freeze_hrnet": bool(_safe_get(vae_cfg, "freeze_hrnet", True)),
        "num_joints": int(_safe_get(vae_cfg, "num_joints", 22)),
        "img_size": tuple(_safe_get(vae_cfg, "img_size", (448, 448))),
    }
    return model_out


def _validate_load_result(missing_keys: list[str], unexpected_keys: list[str]) -> None:
    allow_prefixes = ("vision_backbone.",)
    missing_bad = [k for k in missing_keys if not k.startswith(allow_prefixes)]
    unexpected_bad = [k for k in unexpected_keys if not k.startswith(allow_prefixes)]
    if missing_bad or unexpected_bad:
        raise RuntimeError(
            "Failed to load CMA-VAE strictly enough. "
            f"missing={missing_bad[:20]}, unexpected={unexpected_bad[:20]}"
        )


def _attach_encode_alias(vae: CMAVAE) -> None:
    if hasattr(vae, "encode"):
        return

    def _encode(self, features, lengths=None):
        return self.encode_motion(features, lengths)

    vae.encode = MethodType(_encode, vae)


def _register_legacy_mld_aliases() -> None:
    """
    Register legacy module aliases used by historical checkpoints.

    Old checkpoints may have been pickled against either of two historical
    package layouts:
    - `mld.*` (original motion-latent-diffusion training repo)
    - `vae_kl_align.*` (previous in-repo package name of CMA-VAE)
    Both are aliased to the current `cma_vae.*` modules so that
    `torch.load(..., weights_only=False)` can resolve pickled references
    (e.g. in `hyper_parameters`).
    """
    # Module path suffixes shared by all layouts, relative to the package root.
    suffixes = [
        "data",
        "data.utils",
        "data.get_data",
        "data.humanml",
        "data.humanml.dataset",
        "models",
        "models.architectures",
        "models.architectures.mld_vae_kl_align",
        "models.operator",
        "models.operator.cross_attention",
        "models.operator.position_encoding",
        "models.losses",
        "models.losses.mld_kl_align",
        "models.modeltype",
        "models.modeltype.mld_kl_align",
        "utils",
        "utils.temos_utils",
    ]

    for legacy_root in ("mld", "vae_kl_align"):
        root = sys.modules.get(legacy_root)
        if root is None:
            try:
                target_root = importlib.import_module("cma_vae")
            except Exception:
                target_root = None
            if legacy_root == "vae_kl_align" and target_root is not None:
                # Alias the package root itself to cma_vae.
                sys.modules[legacy_root] = target_root
                root = target_root
            else:
                root = types.ModuleType(legacy_root)
                root.__path__ = []  # mark as package
                sys.modules[legacy_root] = root

        for suffix in suffixes:
            legacy_name = f"{legacy_root}.{suffix}"
            target_name = f"cma_vae.{suffix}"
            if legacy_name in sys.modules:
                continue
            try:
                module = importlib.import_module(target_name)
            except Exception:
                continue

            sys.modules[legacy_name] = module
            parent_name, _, child_name = legacy_name.rpartition(".")
            if parent_name in sys.modules and child_name:
                setattr(sys.modules[parent_name], child_name, module)

    # Checkpoints trained from the standalone `cma_vae/` working directory
    # pickle BARE top-level references (e.g. `data.utils`, because cma_vae's
    # standalone mode uses absolute imports). Alias the `data.*` family too —
    # but never shadow a top-level module that is already imported (e.g. the
    # main repo's `models` / `utils`); such references are handled by the
    # tolerant unpickler fallback in `load_motion_vae` instead.
    for suffix in suffixes:
        top = suffix.split(".")[0]
        if top != "data":
            continue
        if suffix in sys.modules:
            continue
        if top in sys.modules and not sys.modules[top].__name__.startswith("cma_vae"):
            continue
        try:
            module = importlib.import_module(f"cma_vae.{suffix}")
        except Exception:
            continue
        sys.modules[suffix] = module
        parent_name, _, child_name = suffix.rpartition(".")
        if parent_name and parent_name in sys.modules and child_name:
            setattr(sys.modules[parent_name], child_name, module)


def _torch_load_tolerant(ckpt_path: str, device: str):
    """
    `torch.load` with a fallback unpickler that substitutes unresolvable
    pickled references (e.g. bare `models.*` module paths from standalone
    cma_vae training runs) with an inert stub. Only `hyper_parameters`
    entries can be affected; the weight tensors are unaffected, and the
    loader falls back to weight-shape inference for any missing config.
    """
    import torch

    try:
        try:
            return torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(ckpt_path, map_location=device)
    except (ModuleNotFoundError, AttributeError) as err:
        import pickle

        class _MissingStub:
            def __init__(self, *args, **kwargs):
                pass

            def __setstate__(self, state):
                if isinstance(state, dict):
                    self.__dict__.update(state)

        class _TolerantUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                try:
                    return super().find_class(module, name)
                except (ModuleNotFoundError, AttributeError):
                    return _MissingStub

        print(
            f"[motion_vae] Warning: checkpoint references unavailable module "
            f"({err}); retrying with a tolerant unpickler. Model config will "
            f"be inferred from weight shapes if needed."
        )
        shim = types.SimpleNamespace(Unpickler=_TolerantUnpickler)
        try:
            return torch.load(
                ckpt_path, map_location=device, weights_only=False, pickle_module=shim
            )
        except TypeError:
            return torch.load(ckpt_path, map_location=device, pickle_module=shim)


def load_motion_vae(checkpoint_path: str, device: str = "cpu", freeze: bool = True):
    """
    Load CMA-VAE (Cross-Modal Aligned Motion VAE) from Lightning checkpoint.

    Code class: `CMAVAE` in cma_vae/ (historical name `MldVaeKLAlign`).
    Release checkpoint convention: `pretrained/cma_vae.ckpt`.

    Args:
        checkpoint_path: Path to `.ckpt` file.
        device: Device string.
        freeze: Whether to freeze all VAE parameters.
    """
    import torch

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Motion VAE checkpoint not found: {checkpoint_path}")

    _register_legacy_mld_aliases()
    ckpt = _torch_load_tolerant(str(ckpt_path), device)

    if not isinstance(ckpt, Mapping):
        raise ValueError("Unsupported checkpoint format; expected mapping-like checkpoint.")

    vae_state = _extract_vae_state_dict(ckpt)
    cfg = _extract_model_config(ckpt, vae_state)

    vae = CMAVAE(
        ablation=_Ablation(pe_type=cfg["pe_type"]),
        nfeats=cfg["nfeats"],
        latent_dim=cfg["latent_dim"],
        ff_size=cfg["ff_size"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        dropout=cfg["dropout"],
        downsample_rate=cfg["downsample_rate"],
        arch=cfg["arch"],
        normalize_before=cfg["normalize_before"],
        activation=cfg["activation"],
        position_embedding=cfg["position_embedding"],
        kl_align_weight=cfg["kl_align_weight"],
        vision_dim=cfg["vision_dim"],
        hrnet_channels=cfg["hrnet_channels"],
        hrnet_checkpoint=cfg["hrnet_checkpoint"],
        freeze_hrnet=cfg["freeze_hrnet"],
        num_joints=cfg["num_joints"],
        img_size=cfg["img_size"],
    )

    incompatible = vae.load_state_dict(vae_state, strict=False)
    _validate_load_result(incompatible.missing_keys, incompatible.unexpected_keys)
    _attach_encode_alias(vae)

    vae = vae.to(device)
    vae.eval()
    if freeze:
        for p in vae.parameters():
            p.requires_grad = False

    motion_vae_config = {
        "nfeats": cfg["nfeats"],
        "latent_dim": cfg["latent_dim"][-1] if isinstance(cfg["latent_dim"], (list, tuple)) else cfg["latent_dim"],
        "downsample_rate": cfg["downsample_rate"],
        "ff_size": cfg["ff_size"],
        "num_layers": cfg["num_layers"],
        "num_heads": cfg["num_heads"],
    }
    return vae, motion_vae_config


def load_motion_vae_state(vae: CMAVAE, checkpoint_path: str, device: str = "cpu") -> Dict[str, Any]:
    """
    Load state dict into an existing CMA-VAE instance.

    Paper: CMA-VAE (Cross-Modal Aligned Motion VAE).

    Args:
        vae: Existing VAE module instance.
        checkpoint_path: Path to checkpoint that contains VAE state.
        device: Map location device string for checkpoint loading.

    Returns:
        Metadata dict (contains at least `motion_vae_config` if present).
    """
    import torch

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Motion VAE checkpoint not found: {checkpoint_path}")

    _register_legacy_mld_aliases()
    ckpt = _torch_load_tolerant(str(ckpt_path), device)

    if not isinstance(ckpt, Mapping):
        raise ValueError("Unsupported checkpoint format; expected mapping-like checkpoint.")

    vae_state = _extract_vae_state_dict(ckpt)
    incompatible = vae.load_state_dict(vae_state, strict=False)
    _validate_load_result(incompatible.missing_keys, incompatible.unexpected_keys)
    _attach_encode_alias(vae)

    return {
        "motion_vae_config": ckpt.get("motion_vae_config"),
        "global_step": ckpt.get("global_step"),
    }
