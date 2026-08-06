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
Training script for Show-o2 with Motion modality (UniMotion).

Unified training entry for all stages; see configs/ for per-stage task mixes.

Code stage <-> paper stage mapping:

    Code stage (config)                              | Paper stage
    -------------------------------------------------|--------------------------------------------
    stage0  (showo2_motion_stage0.yaml)      | Stage 0: LRA pre-alignment (M2M self-reconstruction)
    stage1  (showo2_motion_stage1.yaml)      | Stage 1a: T2M warmup
    stage2a (showo2_motion_stage2a.yaml)     | Stage 1b (part 1): T2M+M2T+Pred, fixed prompts
    stage2b (showo2_motion_stage2b.yaml)     | Stage 1b (part 2): +Edit, diverse instruction templates
    stage3  (showo2_motion_stage3.yaml)      | Stage 2: cross-modal image tasks (img2pose + img2text)
    stage4  (showo2_motion_stage4.yaml)      | Stage 3: joint finetuning on all downstream tasks
    single_task_m2t_a/b (showo2_motion_single_task_m2t_*.yaml) | Single-task M2T variants (template ablation)

    The CMA-VAE (motion VAE) is pretrained separately (see cma_vae/) and stays frozen here.

Usage:
    # Stage 1: T2M pretraining
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train_motion.py config=configs/showo2_motion_stage1.yaml

    # Stage 2a: multi-task training (fixed prompts)
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train_motion.py config=configs/showo2_motion_stage2a.yaml

    # Stage 2b: multi-task training (diverse prompts + Motion Edit)
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train_motion.py config=configs/showo2_motion_stage2b.yaml
"""

import os
import json
import logging
import math
import shutil
import time
from pathlib import Path
import numpy as np
import wandb
import random
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed
from omegaconf import OmegaConf
from models import Showo2WithMotion, omni_attn_mask_naive
from models.lr_schedulers import get_scheduler
from models.my_logging import set_verbosity_info, set_verbosity_error
from models.misc import get_text_tokenizer, get_weight_type
from models.motion_vae import load_motion_vae, load_motion_vae_state
from models.lora import count_lora_parameters, freeze_non_lora_params
from datasets import create_motion_dataloader
from datasets import create_h36m_dataloader
from datasets import create_h36m_mgie_dataloader
from models import WanVAE

from utils import (
    get_config, flatten_omega_conf, AverageMeter, path_to_llm_name, 
    _freeze_params, _freeze_original_embeddings,
    # Motion visualization utils
    recover_from_ric, visualize_motion_to_png, create_motion_grid, load_motion_stats
)

from transport import Sampler, create_transport
from motion_length_utils import (
    latent_len_from_motion_length,
    latent_lens_from_motion_lengths,
    motion_token_slots,
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = get_logger(__name__, log_level="INFO")


def _assert_no_meta_tensors(module, module_name: str):
    """
    Fail fast if any parameter/buffer is still on meta device.
    Meta tensors indicate weights were not materialized (common source of silent bad init).
    """
    meta_params = [name for name, p in module.named_parameters() if getattr(p, "is_meta", False)]
    meta_buffers = [name for name, b in module.named_buffers() if getattr(b, "is_meta", False)]
    if meta_params or meta_buffers:
        preview = (meta_params + meta_buffers)[:10]
        raise RuntimeError(
            f"{module_name} has meta tensors after loading "
            f"(params={len(meta_params)}, buffers={len(meta_buffers)}). "
            f"Examples: {preview}"
        )


def _set_motion_vae_trainability(motion_vae, motion_only: bool, logger=None):
    train_prefixes = (
        "motion_embedding.",
        "motion_downsample.",
        "motion_pos_encoder.",
        "motion_encoder.",
        "motion_dist_layer.",
        "decoder_",
        "final_layer.",
    )
    freeze_prefixes = (
        "joint_encoder.",
        "joint_dist_layer.",
        "vision_",
        "vision_backbone.",
    )

    trainable_count = 0
    frozen_count = 0
    for name, param in motion_vae.named_parameters():
        if not motion_only:
            param.requires_grad = True
            trainable_count += param.numel()
            continue

        should_train = name.startswith(train_prefixes)
        if name.startswith(freeze_prefixes):
            should_train = False

        param.requires_grad = should_train
        if should_train:
            trainable_count += param.numel()
        else:
            frozen_count += param.numel()

    if logger is not None:
        logger.info(
            f"Motion VAE trainability set (motion_only={motion_only}): "
            f"trainable={trainable_count:,}, frozen={frozen_count:,}"
        )


def _build_lengths_mask(lengths, max_len, device):
    lengths_t = torch.as_tensor(lengths, device=device, dtype=torch.long)
    return torch.arange(max_len, device=device).unsqueeze(0) < lengths_t.unsqueeze(1)


def _compute_motion_vae_losses(
    recon_motion,
    target_motion,
    lengths,
    dist_motion,
    downsample_rate,
):
    # Reconstruction loss with valid-frame mask
    valid_mask = _build_lengths_mask(lengths, recon_motion.shape[1], recon_motion.device).unsqueeze(-1)
    recon_err = F.smooth_l1_loss(recon_motion, target_motion, reduction="none")
    recon_mask = valid_mask.expand_as(recon_err)
    recon_loss = (recon_err * recon_mask).sum() / recon_mask.sum().clamp_min(1.0)

    # KL(q(z|motion) || N(0, I)) with valid-latent mask
    latent_lengths = latent_lens_from_motion_lengths(lengths, max(1, int(downsample_rate)))
    mu = dist_motion.mean
    std = dist_motion.stddev
    var = std.pow(2).clamp_min(1e-8)
    kl_elem = 0.5 * (mu.pow(2) + var - torch.log(var) - 1.0)
    latent_mask = _build_lengths_mask(latent_lengths, mu.shape[0], mu.device).transpose(0, 1).unsqueeze(-1)
    kl_mask = latent_mask.expand_as(kl_elem)
    kl_loss = (kl_elem * kl_mask).sum() / kl_mask.sum().clamp_min(1.0)

    return recon_loss, kl_loss


def _load_h36m_constant_projection(data_root: str, expected_dim: int = 269):
    """
    Build a canonical projection for H36M normalized motion features.

    Dims with Std==0 are treated as constant and projected to canonical normalized
    values (0.0) to reduce noise on non-informative channels.
    """
    mean_path = os.path.join(data_root, "Mean.npy")
    std_path = os.path.join(data_root, "Std.npy")
    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        raise FileNotFoundError(f"Missing Mean/Std in {data_root}")

    mean = np.load(mean_path)
    std = np.load(std_path)
    if mean.ndim != 1 or std.ndim != 1:
        raise ValueError(f"Unexpected Mean/Std shape: mean={mean.shape}, std={std.shape}")
    if mean.shape[0] < expected_dim or std.shape[0] < expected_dim:
        raise ValueError(
            f"Mean/Std dims smaller than expected_dim={expected_dim}: "
            f"mean={mean.shape[0]}, std={std.shape[0]}"
        )

    std = std[:expected_dim].astype(np.float32)
    const_mask = np.isclose(std, 0.0)
    const_values = np.zeros((expected_dim,), dtype=np.float32)
    return torch.from_numpy(const_mask), torch.from_numpy(const_values)


def _project_motion_constant_dims(
    motion_norm: torch.Tensor,
    const_mask: torch.Tensor,
    const_values: torch.Tensor,
) -> torch.Tensor:
    if const_mask is None or const_values is None:
        return motion_norm
    if motion_norm.shape[-1] != const_mask.numel():
        raise RuntimeError(
            f"Motion dim mismatch for constant projection: "
            f"motion={motion_norm.shape[-1]}, mask={const_mask.numel()}"
        )

    projected = motion_norm.clone()
    mask = const_mask.to(device=motion_norm.device)
    values = const_values.to(device=motion_norm.device, dtype=motion_norm.dtype)
    projected[..., mask] = values[mask]
    return projected


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    
    # For multi-task training (Stage 2a/2b/4, etc.), different tasks may exercise different module paths.
    # Single-task training (e.g. Stage 1) can disable find_unused_parameters to avoid the extra graph traversal.
    from accelerate import DistributedDataParallelKwargs
    ddp_tasks = list(config.dataset.get('tasks', ['t2m', 'm2t']))
    ddp_find_unused_parameters = bool(
        config.training.get('ddp_find_unused_parameters', len(ddp_tasks) > 1)
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=ddp_find_unused_parameters)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
        kwargs_handlers=[ddp_kwargs],
    )
    logger.info(
        f"DDP find_unused_parameters={ddp_find_unused_parameters} "
        f"(tasks={ddp_tasks if ddp_tasks else 'unknown'})"
    )

    batch_size_t2m = config.training.batch_size_t2m
    batch_size_m2t = config.training.batch_size_m2t
    batch_size_m2m = config.training.get('batch_size_m2m', 0)
    batch_size_pred = config.training.get('batch_size_pred', 0)
    batch_size_t2t = config.training.get('batch_size_t2t', 0)
    batch_size_edit = config.training.get('batch_size_edit', 0)
    batch_size_img2pose = config.training.get('batch_size_img2pose', 0)
    batch_size_img2text = config.training.get('batch_size_img2text', 0)
    batch_size_mgie = config.training.get('batch_size_mgie', 0)
    batch_size_candidates = [
        bs for bs in [batch_size_t2m, batch_size_m2t, batch_size_m2m, batch_size_pred, batch_size_t2t, 
                      batch_size_edit, batch_size_img2pose, batch_size_img2text, batch_size_mgie] if bs > 0
    ]
    total_batch_size_per_gpu = max(batch_size_candidates) if batch_size_candidates else 0
    total_batch_size_without_accum = total_batch_size_per_gpu * accelerator.num_processes
    total_batch_size = total_batch_size_without_accum * config.training.gradient_accumulation_steps

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            total_batch_size_per_gpu
        )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # Initialize wandb
    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint", None)

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # Set seed
    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    weight_type = get_weight_type(config)

    motion_vae_train_cfg = config.model.motion_vae.get("train", {})
    motion_vae_train_enabled = bool(motion_vae_train_cfg.get("enabled", False))
    motion_vae_motion_only = bool(motion_vae_train_cfg.get("motion_only", True))
    motion_vae_keep_fp32 = bool(motion_vae_train_cfg.get("keep_fp32", True))
    motion_vae_task_whitelist = set(motion_vae_train_cfg.get("task_whitelist", ["m2m"]))

    vae_loss_cfg = config.training.get("vae_loss", {})
    vae_loss_enabled = bool(vae_loss_cfg.get("enabled", False)) and motion_vae_train_enabled
    vae_loss_coeff = float(vae_loss_cfg.get("coeff", 1.0))
    vae_recon_weight = float(vae_loss_cfg.get("recon_weight", 1.0))
    vae_kl_weight = float(vae_loss_cfg.get("kl_weight", 1.0e-4))
    vae_kl_warmup_steps = max(1, int(vae_loss_cfg.get("kl_warmup_steps", 3000)))

    # Load Motion VAE
    logger.info(f"Loading Motion VAE from {config.model.motion_vae.pretrained_model_path}")
    motion_vae, motion_vae_config = load_motion_vae(
        config.model.motion_vae.pretrained_model_path,
        device=accelerator.device,
        freeze=not motion_vae_train_enabled,
    )
    motion_vae_downsample_rate = int(motion_vae_config["downsample_rate"])
    config_motion_vae_downsample = int(config.model.motion_vae.downsample_rate)
    config_showo_downsample = int(config.model.showo.motion_downsample_rate)
    config_dataset_downsample = int(config.dataset.motion.downsample_rate)
    mismatch_values = {
        "config.model.motion_vae.downsample_rate": config_motion_vae_downsample,
        "config.model.showo.motion_downsample_rate": config_showo_downsample,
        "config.dataset.motion.downsample_rate": config_dataset_downsample,
    }
    mismatch_fields = [
        f"{name}={value}"
        for name, value in mismatch_values.items()
        if value != motion_vae_downsample_rate
    ]
    if mismatch_fields:
        raise ValueError(
            "Motion downsample_rate must match the loaded Motion VAE checkpoint "
            f"(checkpoint={motion_vae_downsample_rate}). Mismatched fields: {', '.join(mismatch_fields)}"
        )
    add_time_embeds = bool(config.model.showo.add_time_embeds)
    base_max_motion_tokens = latent_len_from_motion_length(
        config.dataset.motion.max_motion_length,
        motion_vae_downsample_rate,
    )
    runtime_max_motion_tokens = motion_token_slots(
        config.dataset.motion.max_motion_length,
        motion_vae_downsample_rate,
        add_time_embeds=add_time_embeds,
    )
    configured_max_motion_tokens = int(config.dataset.preprocessing.max_motion_tokens)
    if configured_max_motion_tokens != base_max_motion_tokens:
        logger.warning(
            "config.dataset.preprocessing.max_motion_tokens=%s does not match "
            "the runtime base motion token count=%s derived from max_motion_length=%s "
            "and downsample_rate=%s. Runtime will use the derived value.",
            configured_max_motion_tokens,
            base_max_motion_tokens,
            int(config.dataset.motion.max_motion_length),
            motion_vae_downsample_rate,
        )
    if motion_vae_train_enabled:
        _set_motion_vae_trainability(
            motion_vae,
            motion_only=motion_vae_motion_only,
            logger=logger,
        )
        if motion_vae_keep_fp32:
            motion_vae = motion_vae.float()
        else:
            motion_vae = motion_vae.to(weight_type)
        motion_vae.train()
        logger.info(
            f"Motion VAE train enabled: motion_only={motion_vae_motion_only}, "
            f"keep_fp32={motion_vae_keep_fp32}, task_whitelist={sorted(motion_vae_task_whitelist)}"
        )
        if vae_loss_enabled:
            logger.info(
                f"Motion VAE loss enabled: coeff={vae_loss_coeff}, "
                f"recon_weight={vae_recon_weight}, kl_weight={vae_kl_weight}, "
                f"kl_warmup_steps={vae_kl_warmup_steps}"
            )
        else:
            logger.warning("Motion VAE training enabled but training.vae_loss.enabled=false.")
    else:
        motion_vae = motion_vae.to(weight_type)
        motion_vae.eval()
    motion_vae_runtime_dtype = (
        torch.float32 if (motion_vae_train_enabled and motion_vae_keep_fp32) else weight_type
    )

    # Initialize tokenizer with motion tokens
    # First get the vocab size without motion tokens (the original Show-o2 vocabulary).
    # This determines which embedding rows are the newly added motion tokens that need training.
    showo_tokenizer_without_motion, _ = get_text_tokenizer(
        config.model.showo.llm_model_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[config.model.showo.llm_model_path],
        add_motion_tokens=False  # do not add motion tokens
    )
    original_vocab_size = len(showo_tokenizer_without_motion)  # original Show-o2 vocab size (incl. showo tokens)
    del showo_tokenizer_without_motion

    # Get the full tokenizer (with motion tokens)
    text_tokenizer, showo_token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[config.model.showo.llm_model_path],
        add_motion_tokens=True
    )
    config.model.showo.llm_vocab_size = len(text_tokenizer)
    num_new_motion_tokens = len(text_tokenizer) - original_vocab_size
    logger.info(f"Original vocab size (Show-o2): {original_vocab_size}, "
                f"Final vocab size (with motion tokens): {len(text_tokenizer)}, "
                f"New motion tokens: {num_new_motion_tokens}")

    # ==================== Load WAN21 VAE (for Stage 3 img2pose/img2text) ====================
    image_vae = None
    h36m_config = config.dataset.get('h36m', None)
    mgie_config_pre = config.dataset.get('mgie', None)
    if (h36m_config is not None) or (mgie_config_pre is not None):
        vae_path = config.model.vae_model.pretrained_model_path
        logger.info(f"Loading WAN21 VAE from {vae_path} for image encoding...")
        image_vae = WanVAE(
            vae_pth=vae_path,
            dtype=weight_type,
            device=accelerator.device,
        )
        logger.info("WAN21 VAE loaded (frozen)")
    
    # Load motion stats for visualization
    motion_mean, motion_std = load_motion_stats(config.dataset.motion.data_root, logger=logger)
    if motion_mean is None:
        logger.warning("Could not load motion mean/std. Visualization may not be accurate.")

    # Initialize Showo2WithMotion model
    # Convert OmegaConf containers to plain python types so model config can be JSON-serialized on save.
    showo_init_kwargs = OmegaConf.to_container(config.model.showo, resolve=True)
    if config.model.showo.load_from_showo:
        logger.info(f"Loading pretrained model from {config.model.showo.pretrained_model_path}")
        # First load the original Show-o2 model weights
        from models import Showo2Qwen2_5
        base_model = Showo2Qwen2_5.from_pretrained(
            config.model.showo.pretrained_model_path,
            use_safetensors=False,
            # Avoid meta-init path here; nested from_pretrained calls can otherwise
            # leave meta tensors and trigger non-meta->meta no-op warnings.
            low_cpu_mem_usage=False,
            device_map=None,
            torch_dtype=weight_type,
        )
        _assert_no_meta_tensors(base_model, "Showo2Qwen2_5 (base_model)")
        
        # Create the Showo2WithMotion model
        model = Showo2WithMotion(**showo_init_kwargs).to(accelerator.device)
        _assert_no_meta_tensors(model, "Showo2WithMotion (before base weight copy)")
        
        # Copy weights from base_model into the new model
        base_state_dict = base_model.state_dict()
        model_state_dict = model.state_dict()
        
        # Handle the vocab-size mismatch (motion tokens were added):
        # embed_tokens and lm_head need special treatment
        embed_key = 'showo.model.embed_tokens.weight'
        lm_head_key = 'showo.lm_head.weight'
        
        if embed_key in base_state_dict and embed_key in model_state_dict:
            old_embed = base_state_dict[embed_key]
            new_embed = model_state_dict[embed_key]
            if old_embed.shape[0] != new_embed.shape[0]:
                logger.info(f"Resizing embed_tokens: {old_embed.shape} -> {new_embed.shape}")
                # Copy embeddings of existing tokens; new tokens keep their random init
                new_embed[:old_embed.shape[0], :] = old_embed
                base_state_dict[embed_key] = new_embed
        
        if lm_head_key in base_state_dict and lm_head_key in model_state_dict:
            old_lm_head = base_state_dict[lm_head_key]
            new_lm_head = model_state_dict[lm_head_key]
            if old_lm_head.shape[0] != new_lm_head.shape[0]:
                logger.info(f"Resizing lm_head: {old_lm_head.shape} -> {new_lm_head.shape}")
                # Copy weights of existing tokens; new tokens keep their random init
                new_lm_head[:old_lm_head.shape[0], :] = old_lm_head
                base_state_dict[lm_head_key] = new_lm_head
        
        # Only copy weights that exist in the new model
        filtered_state_dict = {k: v for k, v in base_state_dict.items() if k in model_state_dict}
        model.load_state_dict(filtered_state_dict, strict=False)
        _assert_no_meta_tensors(model, "Showo2WithMotion (after base weight copy)")
        
        del base_model
        logger.info(f"Loaded {len(filtered_state_dict)} parameters from pretrained model")
        previous_stage_lora_path = None  # Stage 1 does not load LoRA weights from a previous stage
    else:
        # Stage 2a/2b: load from the previous stage's checkpoint
        model = Showo2WithMotion(**showo_init_kwargs).to(accelerator.device)
        
        # Load previous-stage weights from pretrained_model_path (if a path is given)
        pretrained_path = config.model.showo.get('pretrained_model_path', None)
        if pretrained_path and os.path.exists(pretrained_path):
            logger.info(f"Loading previous stage weights from {pretrained_path}")
            
            # Auto-resolve the checkpoint path: check whether the unwrapped_model subdirectory is needed
            if os.path.isdir(pretrained_path):
                bin_files = [f for f in os.listdir(pretrained_path) if f.endswith('.bin')]
                if not bin_files:
                    unwrapped_path = os.path.join(pretrained_path, 'unwrapped_model')
                    if os.path.exists(unwrapped_path):
                        pretrained_path = unwrapped_path
                        logger.info(f"  Auto-detected unwrapped_model path: {pretrained_path}")
            
            # Load the state_dict
            from utils import load_state_dict
            state_dict = load_state_dict(pretrained_path)
            
            # Filter out parameters that should not be loaded
            if config.model.showo.params_not_load is not None:
                params_to_delete = []
                for k in state_dict:
                    for n in config.model.showo.params_not_load:
                        if n in k:
                            params_to_delete.append(k)
                for k in params_to_delete:
                    del state_dict[k]
            
            # Filter out weights that do not exist in the current model
            model_state_dict = model.state_dict()
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
            model.load_state_dict(filtered_state_dict, strict=False)
            
            logger.info(f"  Loaded {len(filtered_state_dict)} parameters from previous stage")
            del state_dict
            
            # Record the previous stage's LoRA weight path; it is loaded after LoRA init,
            # because the LoRA modules must be created before their weights can be loaded
            previous_stage_lora_path = None
            original_pretrained_path = config.model.showo.get('pretrained_model_path', None)
            if original_pretrained_path:
                # Check whether lora_weights.pt exists in the checkpoint directory
                if 'unwrapped_model' in pretrained_path:
                    potential_lora_path = os.path.join(os.path.dirname(pretrained_path), 'lora_weights.pt')
                else:
                    potential_lora_path = os.path.join(original_pretrained_path, 'lora_weights.pt')
                if os.path.exists(potential_lora_path):
                    previous_stage_lora_path = potential_lora_path
                    logger.info(f"  Found previous stage LoRA weights: {previous_stage_lora_path}")
        elif pretrained_path:
            previous_stage_lora_path = None
            logger.warning(f"pretrained_model_path specified but not found: {pretrained_path}")
            logger.warning("Starting with randomly initialized model!")
        else:
            previous_stage_lora_path = None

    # Freeze specified parameters
    _freeze_params(model, config.model.showo.frozen_params)
    
    # Embedding training policy (recommended to specify explicitly in each stage config):
    #   - required: training of the newly added motion token embeddings must be enabled, otherwise error out
    #   - disabled: disable embedding training (keep frozen)
    #   - auto: try to enable; on failure only warn (backward-compatible behavior)
    embedding_train_policy = str(
        config.model.showo.get("embedding_train_policy", "auto")
    ).lower()
    if embedding_train_policy not in {"required", "disabled", "auto"}:
        raise ValueError(
            f"Invalid model.showo.embedding_train_policy={embedding_train_policy}. "
            f"Expected one of: required, disabled, auto."
        )
    logger.info(f"Embedding train policy: {embedding_train_policy}")
    if embedding_train_policy == "disabled":
        logger.info("Embedding new-token training is disabled by config.")
    else:
        _freeze_original_embeddings(
            model,
            original_vocab_size,
            logger=logger,
            strict=(embedding_train_policy == "required"),
        )

    # ==================== Apply Token-Level LoRA ====================
    lora_config = config.get('lora', None)
    if lora_config is not None and lora_config.get('enabled', False):
        logger.info("=" * 50)
        logger.info("Applying Token-Level LoRA...")
        
        lora_params_dict = {
            'rank_A': lora_config.get('rank_A', 16),
            'rank_B': lora_config.get('rank_B', 16),
            'alpha': lora_config.get('alpha', 16),
            'dropout': lora_config.get('dropout', 0.0),
            'enable_A': lora_config.get('enable_A', False),  # text/image path
            'enable_B': lora_config.get('enable_B', True),   # motion path
            'target_modules': lora_config.get('target_modules', ['q_proj', 'k_proj', 'v_proj', 'o_proj']),
        }
        model.apply_lora(lora_params_dict)
        
        # Count LoRA parameters
        lora_stats = count_lora_parameters(model)
        logger.info(f"LoRA-A (Text/Image) enabled: {lora_params_dict['enable_A']}")
        logger.info(f"LoRA-B (Motion) enabled: {lora_params_dict['enable_B']}")
        logger.info(f"LoRA-A parameters: {lora_stats['text_image_path_params']:,}")
        logger.info(f"LoRA-B parameters: {lora_stats['motion_path_params']:,}")
        logger.info(f"Total LoRA parameters: {lora_stats['total_lora_params']:,}")
        
        # Load LoRA weights from the previous stage (when Stage 2a/2b continues from Stage 1/2a)
        if previous_stage_lora_path and os.path.exists(previous_stage_lora_path):
            lora_load_info = model.load_lora(
                previous_stage_lora_path,
                strict=False,
                preserve_unmatched=True,
            )
            logger.info(
                f"Loaded previous stage LoRA weights from {previous_stage_lora_path} "
                f"(loaded={len(lora_load_info.get('loaded_keys', []))}, "
                f"unmatched={len(lora_load_info.get('missing_keys_in_model', []))}, "
                f"missing_in_ckpt={len(lora_load_info.get('missing_keys_in_state', []))})"
            )
        
        # Stage 3: freeze LoRA-B (motion path) but keep its weights for forward and save.
        # With train_B=false, LoRA-B is still created and loaded, but does not participate in training.
        #
        # Parameter naming convention:
        #   TokenWiseLoRALinear.lora_A (text/image path):
        #     ...q_proj.lora_A.lora_A  (A matrix)
        #     ...q_proj.lora_A.lora_B  (B matrix)
        #   TokenWiseLoRALinear.lora_B (motion path):
        #     ...q_proj.lora_B.lora_A  (A matrix)
        #     ...q_proj.lora_B.lora_B  (B matrix)
        #
        # Detection rule: name.split('.')[-2] == 'lora_B' means the parameter belongs to the motion-path LoRA
        if not lora_config.get('train_B', True):
            frozen_lora_b_count = 0
            for name, param in model.named_parameters():
                parts = name.split('.')
                if len(parts) >= 2 and parts[-2] == 'lora_B':
                    # This parameter belongs to TokenWiseLoRALinear.lora_B (motion path)
                    param.requires_grad = False
                    frozen_lora_b_count += 1
            logger.info(f"Froze {frozen_lora_b_count} LoRA-B (motion path) parameters (train_B=False)")
        
        logger.info("=" * 50)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params
    optimizer_type = config.optimizer.name

    if optimizer_type == "adamw":
        param_groups = []
        base_lr = float(optimizer_config.learning_rate)

        if lora_config is not None and lora_config.get('enabled', False):
            lora_params = [p for p in model.get_lora_parameters() if p.requires_grad]
            lora_param_ids = set(id(p) for p in lora_params)
            base_params = [p for p in model.parameters() if p.requires_grad and id(p) not in lora_param_ids]
            lora_lr = float(lora_config.get('learning_rate', base_lr))
        else:
            lora_params = []
            base_params = [p for p in model.parameters() if p.requires_grad]
            lora_lr = base_lr

        if base_params:
            param_groups.append({
                'params': base_params,
                'lr': base_lr,
                'name': 'base',
            })
        if lora_params:
            param_groups.append({
                'params': lora_params,
                'lr': lora_lr,
                'name': 'lora',
            })

        motion_vae_lr = float(optimizer_config.get("motion_vae_learning_rate", 5.0e-6))
        motion_vae_weight_decay = float(optimizer_config.get("motion_vae_weight_decay", 0.0))
        motion_vae_trainable_params = []
        if motion_vae_train_enabled:
            motion_vae_trainable_params = [p for p in motion_vae.parameters() if p.requires_grad]
            if motion_vae_trainable_params:
                param_groups.append({
                    "params": motion_vae_trainable_params,
                    "lr": motion_vae_lr,
                    "weight_decay": motion_vae_weight_decay,
                    "name": "motion_vae",
                })
            else:
                logger.warning("Motion VAE train enabled but no trainable Motion VAE params were found.")

        if len(param_groups) == 0:
            raise ValueError("No trainable parameters found for optimizer.")

        logger.info("Optimizer param groups:")
        logger.info(f"  - base params: {len(base_params)} tensors, lr={base_lr}")
        if lora_params:
            logger.info(f"  - LoRA params: {len(lora_params)} tensors, lr={lora_lr}")
        if motion_vae_trainable_params:
            logger.info(
                f"  - motion_vae params: {len(motion_vae_trainable_params)} tensors, "
                f"lr={motion_vae_lr}, weight_decay={motion_vae_weight_decay}"
            )

        optimizer = AdamW(
            param_groups,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders")
    
    # Read the training-stage configuration
    training_stage = config.training.get('stage', 'lm_pretrain')
    instruction_template_path = config.training.get('instruction_template_path', None)
    t2m_ratio = config.training.get('t2m_ratio', 0.5)
    
    logger.info(f"Training stage: {training_stage}")
    logger.info(f"Instruction template: {instruction_template_path}")
    logger.info(f"T2M ratio: {t2m_ratio}")
    
    # Determine the training tasks
    tasks = config.dataset.get('tasks', None)
    if tasks is None:
        # Eval-only configs put tasks under `test.tasks`; prevent accidental misuse in training.
        if config.training.get('stage', '').lower() == 'test' and config.get('test', None) is not None:
            test_tasks = config.test.get('tasks', None)
            if test_tasks:
                raise ValueError(
                    "config.dataset.tasks is missing, but test.tasks exists. "
                    "This looks like an evaluation config and should not be used with train_motion.py. "
                    f"test.tasks={test_tasks}"
                )
        tasks = ['t2m', 'm2t']
    train_m2m = 'm2m' in tasks
    train_t2m = 't2m' in tasks
    train_m2t = 'm2t' in tasks
    train_pred = 'pred' in tasks
    train_t2t = 't2t' in tasks  # Text-to-Text task
    train_edit = 'edit' in tasks  # Motion Edit task (MotionFix)
    train_img2pose = 'img2pose' in tasks  # Image-to-Pose task
    train_img2text = 'img2text' in tasks  # Image-to-Text task
    train_mgie = 'mgie' in tasks  # Stage 3b: Motion-guided Image Editing

    # For H36M img2pose, project non-informative constant dims (Std==0) to canonical values.
    img2pose_project_constant_dims = bool(config.training.get('img2pose_project_constant_dims', True))
    img2pose_const_mask = None
    img2pose_const_values = None
    if train_img2pose and img2pose_project_constant_dims:
        if h36m_config is None:
            logger.warning(
                "img2pose_project_constant_dims=True but dataset.h36m is missing; skip projection."
            )
        else:
            try:
                img2pose_const_mask, img2pose_const_values = _load_h36m_constant_projection(
                    h36m_config.data_root, expected_dim=269
                )
                num_const_dims = int(img2pose_const_mask.sum().item())
                logger.info(
                    f"img2pose constant-dim projection enabled: {num_const_dims}/269 dims "
                    f"(from {h36m_config.data_root}/Std.npy)"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize img2pose constant-dim projection. "
                    f"Continue without it. Error: {e}"
                )
                img2pose_const_mask = None
                img2pose_const_values = None

    # Optional decoded supervision for img2pose:
    # decode x1_hat / x1_target and supervise 0:67 (root + ric) feature dims.
    img2pose_decode067_cfg = config.training.get("img2pose_decode067_supervision", {})
    img2pose_decode067_enabled = bool(img2pose_decode067_cfg.get("enabled", False))
    img2pose_decode067_weight = float(img2pose_decode067_cfg.get("weight", 0.1))
    img2pose_decode067_dims = int(img2pose_decode067_cfg.get("dims", 67))
    img2pose_decode067_loss_type = str(img2pose_decode067_cfg.get("loss_type", "smooth_l1")).lower()
    img2pose_decode067_warmup_steps = int(img2pose_decode067_cfg.get("warmup_steps", 0))
    img2pose_decode067_debug_every = int(img2pose_decode067_cfg.get("debug_every", 0))
    if img2pose_decode067_dims <= 0 or img2pose_decode067_dims > 269:
        raise ValueError(f"img2pose_decode067_supervision.dims must be in [1,269], got {img2pose_decode067_dims}")
    if img2pose_decode067_loss_type not in {"smooth_l1", "l1", "mse"}:
        raise ValueError(
            "img2pose_decode067_supervision.loss_type must be one of "
            "['smooth_l1', 'l1', 'mse']"
        )
    img2pose_decode067_effective = (
        img2pose_decode067_enabled
        and train_img2pose
        and (img2pose_decode067_weight > 0.0)
    )
    if img2pose_decode067_effective and motion_vae_train_enabled:
        logger.warning(
            "img2pose decoded 0:67 supervision disabled because motion_vae.train.enabled=True. "
            "Current implementation assumes frozen Motion VAE for this auxiliary loss."
        )
        img2pose_decode067_effective = False
    if img2pose_decode067_effective:
        logger.info(
            "img2pose decoded 0:67 supervision enabled: "
            f"weight={img2pose_decode067_weight}, dims={img2pose_decode067_dims}, "
            f"loss_type={img2pose_decode067_loss_type}, warmup_steps={img2pose_decode067_warmup_steps}, "
            f"debug_every={img2pose_decode067_debug_every}"
        )
    
    logger.info(f"Training tasks: {tasks}")
    
    # Motion Prediction parameters
    predict_ratio = config.dataset.motion.get('predict_ratio', 0.2)
    
    # Motion Edit parameters (MotionFix dataset)
    motion_edit_config = config.dataset.get('motion_edit', None)
    motion_recon_config = config.dataset.get('motion_recon', None)
    mgie_config = config.dataset.get('mgie', None)

    # Motion Reconstruction dataloaders (Stage 0)
    train_dataloader_m2m_hml = None
    train_dataloader_m2m_h36m = None
    train_dataloader_m2m_mfix = None
    if train_m2m and batch_size_m2m > 0:
        if motion_recon_config is None:
            logger.warning("Task 'm2m' enabled but dataset.motion_recon is missing. Skip m2m.")
        else:
            recon_instruction_path = motion_recon_config.get('instruction_template_path', None)
            recon_bottleneck = motion_recon_config.get('bottleneck', {})
            default_unit_length = config.dataset.motion.get('unit_length', 1)
            default_fps = config.dataset.motion.get('fps', 20)

            # HumanML3D
            hml_cfg = motion_recon_config.get('humanml3d', None)
            if hml_cfg is not None and hml_cfg.get('data_root', ''):
                train_dataloader_m2m_hml = create_motion_dataloader(
                    data_root=hml_cfg.get('data_root'),
                    batch_size=batch_size_m2m,
                    text_tokenizer=text_tokenizer,
                    accelerator=accelerator,
                    max_seq_len=config.dataset.preprocessing.max_seq_length,
                    max_motion_length=hml_cfg.get('max_motion_length', config.dataset.motion.max_motion_length),
                    min_motion_length=hml_cfg.get('min_motion_length', config.dataset.motion.min_motion_length),
                    downsample_rate=config.dataset.motion.downsample_rate,
                    motion_latent_dim=config.dataset.motion.motion_latent_dim,
                    cond_dropout_prob=0.0,
                    num_workers=config.dataset.params.num_workers,
                    showo_token_ids=showo_token_ids,
                    split='train',
                    add_time_embeds=config.model.showo.add_time_embeds,
                    instruction_template_path=recon_instruction_path,
                    task_type='m2m',
                    dataset_name='humanml3d',
                    bottleneck=recon_bottleneck,
                    unit_length=default_unit_length,
                    fps=default_fps,
                )
                logger.info(f"M2M HumanML3D dataloader created with {len(train_dataloader_m2m_hml)} batches")

            # H36M (single-frame reconstruction)
            h36m_recon_cfg = motion_recon_config.get('h36m', None)
            if h36m_recon_cfg is not None and h36m_recon_cfg.get('data_root', ''):
                train_dataloader_m2m_h36m = create_motion_dataloader(
                    data_root=h36m_recon_cfg.get('data_root'),
                    batch_size=batch_size_m2m,
                    text_tokenizer=text_tokenizer,
                    accelerator=accelerator,
                    max_seq_len=config.dataset.preprocessing.max_seq_length,
                    max_motion_length=h36m_recon_cfg.get('max_motion_length', 1),
                    min_motion_length=h36m_recon_cfg.get('min_motion_length', 1),
                    downsample_rate=config.dataset.motion.downsample_rate,
                    motion_latent_dim=config.dataset.motion.motion_latent_dim,
                    cond_dropout_prob=0.0,
                    num_workers=config.dataset.params.num_workers,
                    showo_token_ids=showo_token_ids,
                    split='train',
                    add_time_embeds=config.model.showo.add_time_embeds,
                    instruction_template_path=recon_instruction_path,
                    task_type='m2m',
                    dataset_name='h36m',
                    bottleneck=recon_bottleneck,
                    unit_length=1,
                    fps=default_fps,
                )
                logger.info(f"M2M H36M dataloader created with {len(train_dataloader_m2m_h36m)} batches")

            # MotionFix target-only reconstruction
            mfix_cfg = motion_recon_config.get('motionfix', None)
            if mfix_cfg is not None and mfix_cfg.get('data_root', ''):
                train_dataloader_m2m_mfix = create_motion_dataloader(
                    data_root=mfix_cfg.get('data_root'),
                    batch_size=batch_size_m2m,
                    text_tokenizer=text_tokenizer,
                    accelerator=accelerator,
                    max_seq_len=config.dataset.preprocessing.max_seq_length,
                    max_motion_length=mfix_cfg.get('max_motion_length', config.dataset.motion.max_motion_length),
                    min_motion_length=mfix_cfg.get('min_motion_length', config.dataset.motion.min_motion_length),
                    downsample_rate=config.dataset.motion.downsample_rate,
                    motion_latent_dim=config.dataset.motion.motion_latent_dim,
                    cond_dropout_prob=0.0,
                    num_workers=config.dataset.params.num_workers,
                    showo_token_ids=showo_token_ids,
                    split='train',
                    add_time_embeds=config.model.showo.add_time_embeds,
                    instruction_template_path=recon_instruction_path,
                    task_type='m2m',
                    dataset_name='motionfix_target',
                    bottleneck=recon_bottleneck,
                    unit_length=default_unit_length,
                    fps=default_fps,
                )
                logger.info(f"M2M MotionFix(target) dataloader created with {len(train_dataloader_m2m_mfix)} batches")

    # Text-to-Motion dataloader
    train_dataloader_t2m = None
    if train_t2m and batch_size_t2m > 0:
        train_dataloader_t2m = create_motion_dataloader(
            data_root=config.dataset.motion.data_root,
            batch_size=batch_size_t2m,
            text_tokenizer=text_tokenizer,
            accelerator=accelerator,
            max_seq_len=config.dataset.preprocessing.max_seq_length,
            max_motion_length=config.dataset.motion.max_motion_length,
            min_motion_length=config.dataset.motion.min_motion_length,
            downsample_rate=config.dataset.motion.downsample_rate,
            motion_latent_dim=config.dataset.motion.motion_latent_dim,
            cond_dropout_prob=config.training.cond_dropout_prob,
            num_workers=config.dataset.params.num_workers,
            is_understanding=False,  # T2M
            showo_token_ids=showo_token_ids,
            split='train',
            latent_dir=config.dataset.motion.get('latent_dir', None),
            add_time_embeds=config.model.showo.add_time_embeds,
            # Training-stage configuration
            stage=training_stage,
            instruction_template_path=instruction_template_path,
            task_type='t2m',
        )
        logger.info(f"T2M dataloader created with {len(train_dataloader_t2m)} batches")

    # Motion-to-Text dataloader
    train_dataloader_m2t = None
    if train_m2t and batch_size_m2t > 0:
        disable_random_crop_for_m2t = bool(
            config.dataset.motion.get('disable_random_crop_for_m2t', False)
        )
        train_dataloader_m2t = create_motion_dataloader(
            data_root=config.dataset.motion.data_root,
            batch_size=batch_size_m2t,
            text_tokenizer=text_tokenizer,
            accelerator=accelerator,
            max_seq_len=config.dataset.preprocessing.max_seq_length,
            max_motion_length=config.dataset.motion.max_motion_length,
            min_motion_length=config.dataset.motion.min_motion_length,
            downsample_rate=config.dataset.motion.downsample_rate,
            motion_latent_dim=config.dataset.motion.motion_latent_dim,
            cond_dropout_prob=0.0,  # M2T does not need cond dropout
            num_workers=config.dataset.params.num_workers,
            is_understanding=True,  # M2T
            showo_token_ids=showo_token_ids,
            split='train',
            latent_dir=config.dataset.motion.get('latent_dir', None),
            add_time_embeds=config.model.showo.add_time_embeds,
            # Training-stage configuration
            stage=training_stage,
            instruction_template_path=instruction_template_path,
            task_type='m2t',
            disable_random_crop_for_m2t=disable_random_crop_for_m2t,
        )
        logger.info(f"M2T dataloader created with {len(train_dataloader_m2t)} batches")
    
    # Motion Prediction dataloader
    train_dataloader_pred = None
    if train_pred and batch_size_pred > 0:
        train_dataloader_pred = create_motion_dataloader(
            data_root=config.dataset.motion.data_root,
            batch_size=batch_size_pred,
            text_tokenizer=text_tokenizer,
            accelerator=accelerator,
            max_seq_len=config.dataset.preprocessing.max_seq_length,
            max_motion_length=config.dataset.motion.max_motion_length,
            min_motion_length=config.dataset.motion.min_motion_length,
            downsample_rate=config.dataset.motion.downsample_rate,
            motion_latent_dim=config.dataset.motion.motion_latent_dim,
            cond_dropout_prob=0.0,  # Prediction does not need cond dropout
            num_workers=config.dataset.params.num_workers,
            showo_token_ids=showo_token_ids,
            split='train',
            add_time_embeds=config.model.showo.add_time_embeds,
            stage=training_stage,
            instruction_template_path=instruction_template_path,
            task_type='pred',
            predict_ratio=predict_ratio,
        )
        logger.info(f"Prediction dataloader created with {len(train_dataloader_pred)} batches")
    
    # Text-to-Text dataloader (Stage 3)
    train_dataloader_t2t = None
    if train_t2t and batch_size_t2t > 0:
        train_dataloader_t2t = create_motion_dataloader(
            data_root=config.dataset.motion.data_root,
            batch_size=batch_size_t2t,
            text_tokenizer=text_tokenizer,
            accelerator=accelerator,
            max_seq_len=config.dataset.preprocessing.max_seq_length,
            max_motion_length=config.dataset.motion.max_motion_length,
            min_motion_length=config.dataset.motion.min_motion_length,
            downsample_rate=config.dataset.motion.downsample_rate,
            motion_latent_dim=config.dataset.motion.motion_latent_dim,
            cond_dropout_prob=0.0,  # T2T does not need cond dropout
            num_workers=config.dataset.params.num_workers,
            showo_token_ids=showo_token_ids,
            split='train',
            add_time_embeds=config.model.showo.add_time_embeds,
            stage=training_stage,
            instruction_template_path=instruction_template_path,
            task_type='t2t',
            fps=config.dataset.motion.fps,  # used by the T2T task to compute durations in seconds
        )
        logger.info(f"T2T dataloader created with {len(train_dataloader_t2t)} batches")
    
    # Motion Edit dataloader (Stage 3 - MotionFix)
    train_dataloader_edit = None
    if train_edit and batch_size_edit > 0 and motion_edit_config is not None:
        edit_data_root = motion_edit_config.get('data_root', '')
        edit_instruction_path = motion_edit_config.get('instruction_template_path', None)
        train_dataloader_edit = create_motion_dataloader(
            data_root=edit_data_root,  # MotionFix dataset path
            batch_size=batch_size_edit,
            text_tokenizer=text_tokenizer,
            accelerator=accelerator,
            max_seq_len=config.dataset.preprocessing.max_seq_length,
            max_motion_length=config.dataset.motion.max_motion_length,
            min_motion_length=config.dataset.motion.min_motion_length,
            downsample_rate=config.dataset.motion.downsample_rate,
            motion_latent_dim=config.dataset.motion.motion_latent_dim,
            cond_dropout_prob=0.0,  # Edit does not need cond dropout
            num_workers=config.dataset.params.num_workers,
            showo_token_ids=showo_token_ids,
            split='train',
            add_time_embeds=config.model.showo.add_time_embeds,
            stage=training_stage,
            instruction_template_path=edit_instruction_path,
            task_type='edit',
            unit_length=config.dataset.motion.unit_length,
            fps=config.dataset.motion.fps,
        )
        logger.info(f"Motion Edit dataloader created with {len(train_dataloader_edit)} batches")

    # H36M Image-to-Pose dataloader (Stage 3)
    train_dataloader_img2pose = None
    if train_img2pose and batch_size_img2pose > 0 and h36m_config is not None:
        h36m_instruction_path = h36m_config.get('instruction_template_path', None)
        num_image_tokens = config.dataset.get('num_image_tokens', 169)
        image_resolution = config.dataset.get('image_resolution', 432)
        train_dataloader_img2pose = create_h36m_dataloader(
            data_root=h36m_config.data_root,
            image_root=h36m_config.image_root,
            batch_size=batch_size_img2pose,
            text_tokenizer=text_tokenizer,
            showo_token_ids=showo_token_ids,
            task_type='img2pose',
            accelerator=accelerator,
            max_seq_len=config.dataset.preprocessing.max_seq_length,
            num_image_tokens=num_image_tokens,
            image_resolution=image_resolution,
            instruction_template_path=h36m_instruction_path,
            num_workers=config.dataset.params.num_workers,
            split='train',
            add_time_embeds=config.model.showo.add_time_embeds,
        )
        logger.info(f"H36M img2pose dataloader created with {len(train_dataloader_img2pose)} batches")
    
    # H36M Image-to-Text dataloader (Stage 3)
    train_dataloader_img2text = None
    if train_img2text and batch_size_img2text > 0 and h36m_config is not None:
        h36m_instruction_path = h36m_config.get('instruction_template_path', None)
        num_image_tokens = config.dataset.get('num_image_tokens', 169)
        image_resolution = config.dataset.get('image_resolution', 432)
        train_dataloader_img2text = create_h36m_dataloader(
            data_root=h36m_config.data_root,
            image_root=h36m_config.image_root,
            batch_size=batch_size_img2text,
            text_tokenizer=text_tokenizer,
            showo_token_ids=showo_token_ids,
            task_type='img2text',
            accelerator=accelerator,
            max_seq_len=config.dataset.preprocessing.max_seq_length,
            num_image_tokens=num_image_tokens,
            image_resolution=image_resolution,
            instruction_template_path=h36m_instruction_path,
            num_workers=config.dataset.params.num_workers,
            split='train',
            add_time_embeds=config.model.showo.add_time_embeds,
        )
        logger.info(f"H36M img2text dataloader created with {len(train_dataloader_img2text)} batches")

    # H36M MGIE dataloader (Stage 3b)
    train_dataloader_mgie = None
    if train_mgie and mgie_config is None:
        logger.warning("Task 'mgie' enabled but config.dataset.mgie is missing. Skip mgie.")
    if train_mgie and batch_size_mgie > 0 and mgie_config is not None:
        mgie_instruction_path = mgie_config.get('instruction_template_path', None)
        num_image_tokens = config.dataset.get('num_image_tokens', 729)
        image_resolution = config.dataset.get('image_resolution', 432)
        train_dataloader_mgie = create_h36m_mgie_dataloader(
            data_root=mgie_config.data_root,
            image_root=mgie_config.image_root,
            batch_size=batch_size_mgie,
            text_tokenizer=text_tokenizer,
            showo_token_ids=showo_token_ids,
            accelerator=accelerator,
            max_seq_len=config.dataset.preprocessing.max_seq_length,
            num_image_tokens=num_image_tokens,
            image_resolution=image_resolution,
            instruction_template_path=mgie_instruction_path,
            num_workers=config.dataset.params.num_workers,
            split='train',
            add_time_embeds=config.model.showo.add_time_embeds,
            max_text_len=config.dataset.preprocessing.get('max_text_len', 256),
        )
        logger.info(f"H36M MGIE dataloader created with {len(train_dataloader_mgie)} batches")

    # Fail fast for Stage 3/3b image tasks to avoid silent task dropping.
    if train_img2pose and train_dataloader_img2pose is None:
        raise ValueError(
            "Task 'img2pose' is enabled but dataloader was not created. "
            "Check training.batch_size_img2pose > 0 and dataset.h36m config."
        )
    if train_img2text and train_dataloader_img2text is None:
        raise ValueError(
            "Task 'img2text' is enabled but dataloader was not created. "
            "Check training.batch_size_img2text > 0 and dataset.h36m config."
        )
    if train_mgie and train_dataloader_mgie is None:
        raise ValueError(
            "Task 'mgie' is enabled but dataloader was not created. "
            "Check training.batch_size_mgie > 0 and dataset.mgie config."
        )

    # Calculate training steps
    dataloader_lengths = []
    if train_dataloader_m2m_hml is not None:
        dataloader_lengths.append(len(train_dataloader_m2m_hml))
    if train_dataloader_m2m_h36m is not None:
        dataloader_lengths.append(len(train_dataloader_m2m_h36m))
    if train_dataloader_m2m_mfix is not None:
        dataloader_lengths.append(len(train_dataloader_m2m_mfix))
    if train_dataloader_t2m is not None:
        dataloader_lengths.append(len(train_dataloader_t2m))
    if train_dataloader_m2t is not None:
        dataloader_lengths.append(len(train_dataloader_m2t))
    if train_dataloader_pred is not None:
        dataloader_lengths.append(len(train_dataloader_pred))
    if train_dataloader_t2t is not None:
        dataloader_lengths.append(len(train_dataloader_t2t))
    if train_dataloader_edit is not None:
        dataloader_lengths.append(len(train_dataloader_edit))
    if train_dataloader_img2pose is not None:
        dataloader_lengths.append(len(train_dataloader_img2pose))
    if train_dataloader_img2text is not None:
        dataloader_lengths.append(len(train_dataloader_img2text))
    if train_dataloader_mgie is not None:
        dataloader_lengths.append(len(train_dataloader_mgie))
    
    if len(dataloader_lengths) == 0:
        raise ValueError("At least one dataloader must be provided!")
    num_update_steps_per_epoch = max(dataloader_lengths)
    
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    ##################################
    #         MODEL RESUME          #
    #################################
    global_step = 0
    first_epoch = 0

    if config.experiment.resume_from_checkpoint:
        resume_target = config.experiment.resume_from_checkpoint
        path = None

        if isinstance(resume_target, str) and resume_target.lower() != "latest":
            path = resume_target
            if not os.path.isabs(path):
                path = os.path.join(config.experiment.output_dir, path)
            if not os.path.isdir(path):
                raise FileNotFoundError(f"Resume checkpoint directory not found: {path}")
        else:
            dirs = os.listdir(config.experiment.output_dir) if os.path.exists(config.experiment.output_dir) else []
            dirs = [d for d in dirs if d.startswith("checkpoint") and d != "checkpoint-final"]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            if len(dirs) > 0:
                path = os.path.join(config.experiment.output_dir, dirs[-1])

        if path is not None:
            global_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

            unwrapped_model_path = os.path.join(path, 'unwrapped_model')
            single_weight_file = os.path.join(unwrapped_model_path, 'pytorch_model.bin')
            
            # Support both sharded and single-file checkpoint formats
            if os.path.exists(single_weight_file):
                # Single-file format
                accelerator.print(f"Resuming from checkpoint {single_weight_file}")
                state_dict = torch.load(single_weight_file, map_location="cpu")
            else:
                # Sharded format (pytorch_model-00001-of-00002.bin, etc.)
                accelerator.print(f"Resuming from sharded checkpoint {unwrapped_model_path}")
                from safetensors.torch import load_file as load_safetensors
                import glob
                import json
                
                # Check whether safetensors files are present
                safetensors_files = glob.glob(os.path.join(unwrapped_model_path, '*.safetensors'))
                pytorch_shard_files = glob.glob(os.path.join(unwrapped_model_path, 'pytorch_model-*.bin'))
                
                state_dict = {}
                if safetensors_files:
                    # Load safetensors format
                    for sf in safetensors_files:
                        state_dict.update(load_safetensors(sf))
                elif pytorch_shard_files:
                    # Load sharded pytorch format
                    pytorch_shard_files = sorted(pytorch_shard_files)
                    for shard_file in pytorch_shard_files:
                        accelerator.print(f"  Loading shard: {os.path.basename(shard_file)}")
                        shard_state_dict = torch.load(shard_file, map_location="cpu")
                        state_dict.update(shard_state_dict)
                        del shard_state_dict
                else:
                    raise FileNotFoundError(f"No checkpoint weights found in {unwrapped_model_path}")

            # Not load some parameters
            if config.model.showo.params_not_load is not None:
                params_to_delete = []
                for k in state_dict:
                    for n in config.model.showo.params_not_load:
                        if n in k:
                            params_to_delete.append(k)
                for k in params_to_delete:
                    del state_dict[k]

            model.load_state_dict(state_dict, strict=False if config.model.showo.params_not_load is not None else True)
            del state_dict
            
            # Load LoRA weights (if present and LoRA is enabled)
            if lora_config is not None and lora_config.get('enabled', False):
                lora_weights_path = os.path.join(path, 'lora_weights.pt')
                if os.path.exists(lora_weights_path):
                    lora_load_info = model.load_lora(
                        lora_weights_path,
                        strict=False,
                        preserve_unmatched=True,
                    )
                    accelerator.print(
                        f"  Loaded LoRA weights from {lora_weights_path} "
                        f"(loaded={len(lora_load_info.get('loaded_keys', []))}, "
                        f"unmatched={len(lora_load_info.get('missing_keys_in_model', []))}, "
                        f"missing_in_ckpt={len(lora_load_info.get('missing_keys_in_state', []))})"
                    )
                else:
                    accelerator.print(f"  Note: LoRA weights file not found at {lora_weights_path}, using initialized LoRA weights")

            if motion_vae_train_enabled:
                motion_vae_path = os.path.join(path, "motion_vae.ckpt")
                if os.path.exists(motion_vae_path):
                    motion_vae_meta = load_motion_vae_state(
                        motion_vae,
                        motion_vae_path,
                        device="cpu",
                    )
                    accelerator.print(
                        f"  Loaded Motion VAE weights from {motion_vae_path} "
                        f"(step={motion_vae_meta.get('global_step', 'n/a')})"
                    )
                else:
                    accelerator.print(
                        f"  Note: Motion VAE checkpoint not found at {motion_vae_path}, "
                        "using initialization from model.motion_vae.pretrained_model_path"
                    )

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps - global_step,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
    )

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    if motion_vae_train_enabled:
        model, motion_vae, optimizer, lr_scheduler = accelerator.prepare(
            model, motion_vae, optimizer, lr_scheduler
        )
    else:
        model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    # DDP wrapper does not expose custom methods like encode_motion.
    # Use the underlying module for latent preparation helpers.
    motion_vae_encode_module = motion_vae.module if hasattr(motion_vae, "module") else motion_vae
    motion_vae_decode_module = motion_vae_encode_module

    ##################################
    #             Training          #
    #################################
    # Create iterators early for logging
    m2m_hml_iter = iter(train_dataloader_m2m_hml) if train_dataloader_m2m_hml is not None else None
    m2m_h36m_iter = iter(train_dataloader_m2m_h36m) if train_dataloader_m2m_h36m is not None else None
    m2m_mfix_iter = iter(train_dataloader_m2m_mfix) if train_dataloader_m2m_mfix is not None else None
    t2m_iter = iter(train_dataloader_t2m) if train_dataloader_t2m is not None else None
    m2t_iter = iter(train_dataloader_m2t) if train_dataloader_m2t is not None else None
    pred_iter = iter(train_dataloader_pred) if train_dataloader_pred is not None else None
    t2t_iter = iter(train_dataloader_t2t) if train_dataloader_t2t is not None else None
    edit_iter = iter(train_dataloader_edit) if train_dataloader_edit is not None else None
    img2pose_iter = iter(train_dataloader_img2pose) if train_dataloader_img2pose is not None else None
    img2text_iter = iter(train_dataloader_img2text) if train_dataloader_img2text is not None else None
    mgie_iter = iter(train_dataloader_mgie) if train_dataloader_mgie is not None else None

    # Task sampling ratios (used for both logging and sampling)
    m2m_ratio_sampling = config.training.get('m2m_ratio', 0.0)
    m2t_ratio = config.training.get('m2t_ratio', 0.3)
    pred_ratio_sampling = config.training.get('pred_ratio', 0.3)
    t2t_ratio_sampling = config.training.get('t2t_ratio', 0.1)  # Stage 3/4: T2T task ratio (default 10%)
    edit_ratio_sampling = config.training.get('edit_ratio', 0.5)  # Motion Edit task ratio
    img2pose_ratio_sampling = config.training.get('img2pose_ratio', 0.7)  # Stage 3: img2pose ratio
    img2text_ratio_sampling = config.training.get('img2text_ratio', 0.3)  # Stage 3: img2text ratio
    mgie_ratio_sampling = config.training.get('mgie_ratio', 0.0)  # Stage 3b: MGIE ratio

    m2m_sample_weights_cfg = (
        motion_recon_config.get('sample_weights', {}) if motion_recon_config is not None else {}
    )
    m2m_source_weight_hml = float(m2m_sample_weights_cfg.get('humanml3d', 2.0))
    m2m_source_weight_h36m = float(m2m_sample_weights_cfg.get('h36m', 1.0))
    m2m_source_weight_mfix = float(m2m_sample_weights_cfg.get('motionfix', 1.0))

    m2m_source_counter = {"m2m_hml": 0, "m2m_h36m": 0, "m2m_mfix": 0}

    logger.info("***** Running Motion Training *****")
    logger.info(f"  Training Stage: {training_stage}")
    logger.info(f"  Tasks: {tasks}")
    logger.info(f"  Use Instructions: {training_stage in ['lm_instruct', 'lm_finetune']}")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Batch size T2M = {batch_size_t2m}")
    logger.info(f"  Batch size M2T = {batch_size_m2t}")
    if batch_size_m2m > 0:
        logger.info(f"  Batch size M2M = {batch_size_m2m}")
    if batch_size_pred > 0:
        logger.info(f"  Batch size Pred = {batch_size_pred}")
    if batch_size_t2t > 0:
        logger.info(f"  Batch size T2T = {batch_size_t2t}")
    if batch_size_edit > 0:
        logger.info(f"  Batch size Edit = {batch_size_edit}")
    if batch_size_img2pose > 0:
        logger.info(f"  Batch size Img2Pose = {batch_size_img2pose}")
    if batch_size_img2text > 0:
        logger.info(f"  Batch size Img2Text = {batch_size_img2text}")
    if batch_size_mgie > 0:
        logger.info(f"  Batch size MGIE = {batch_size_mgie}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  T2M ratio = {t2m_ratio}")
    logger.info(f"  M2T ratio = {m2t_ratio}")
    if m2m_hml_iter is not None or m2m_h36m_iter is not None or m2m_mfix_iter is not None:
        logger.info(f"  M2M ratio = {m2m_ratio_sampling}")
        logger.info(
            f"  M2M dataset ratio (HML:H36M:MFix) = "
            f"{m2m_source_weight_hml}:{m2m_source_weight_h36m}:{m2m_source_weight_mfix}"
        )
    logger.info(f"  Motion VAE train enabled = {motion_vae_train_enabled}")
    logger.info(f"  Motion VAE loss enabled = {vae_loss_enabled}")
    logger.info(f"  Pred ratio = {pred_ratio_sampling}")
    if t2t_iter is not None:
        logger.info(f"  T2T ratio = {t2t_ratio_sampling}")
    if edit_iter is not None:
        logger.info(f"  Edit ratio = {edit_ratio_sampling}")
    if img2pose_iter is not None:
        logger.info(f"  Img2Pose ratio = {img2pose_ratio_sampling}")
    if img2text_iter is not None:
        logger.info(f"  Img2Text ratio = {img2text_ratio_sampling}")
    if mgie_iter is not None:
        logger.info(f"  MGIE ratio = {mgie_ratio_sampling}")

    # Create transport for flow matching
    transport = create_transport(
        path_type=config.transport.path_type,
        prediction=config.transport.prediction,
        loss_weight=config.transport.loss_weight,
        train_eps=config.transport.train_eps,
        sample_eps=config.transport.sample_eps,
        snr_type=config.transport.snr_type,
        do_shift=config.transport.do_shift,
        seq_len=runtime_max_motion_tokens,
    )

    sampler = Sampler(transport)

    @torch.no_grad()
    def prepare_motion_latents_and_labels(
        motions: torch.Tensor,
        motion_lengths: torch.Tensor,
        data_type: str,
        motion_masks: torch.Tensor,
    ):
        """
        Prepare motion latents and training labels (encoded from raw motion).
        
        Args:
            motions: [bs, T, 269] normalized motion features
            motion_lengths: [bs] original motion lengths
            data_type: 't2m_motion' or 'm2t'
            motion_masks: [bs, seq_len] motion token mask
            
        Returns:
            motion_latents: [latent_len, bs, 256] noisy motion latents
            t: [bs] timesteps
            motion_labels: [bs, latent_len, 256] velocity targets
            motion_masks: [bs, seq_len] updated masks
        """
        bs = motions.shape[0]
        device = motions.device
        
        # Encode motion with VAE
        motion_latents, _ = motion_vae_encode_module.encode_motion(motions, motion_lengths.tolist())
        # motion_latents: [latent_len, bs, 256]
        
        latent_len = motion_latents.shape[0]
        
        # Sample timesteps for each batch element (one t per motion sequence, not per token)
        t0, t1 = 0.0, 1.0
        if data_type == 'm2t':
            t0 = config.training.und_max_t0  # understanding tasks use less noise
        
        # Sample t from lognorm distribution
        u = torch.normal(mean=0.0, std=1.0, size=(bs,), device=device)
        t = 1 / (1 + torch.exp(-u)) * (t1 - t0) + t0  # [bs]
        
        # Create noisy latents using the sampled timesteps
        # x0 is noise, x1 is clean motion latent
        x0 = torch.randn_like(motion_latents)  # [latent_len, bs, 256]
        x1 = motion_latents  # [latent_len, bs, 256]
        
        # Linear interpolation: xt = (1-t) * x0 + t * x1
        # Expand t for broadcasting: [bs] -> [1, bs, 1]
        t_expanded = t.view(1, bs, 1)
        xt = (1 - t_expanded) * x0 + t_expanded * x1  # [latent_len, bs, 256]
        
        # Velocity target: ut = x1 - x0 (for linear path)
        ut = x1 - x0  # [latent_len, bs, 256]
        
        # motion_labels: [bs, latent_len, 256]
        motion_labels = ut.permute(1, 0, 2)
        
        # The M2T task does not need the motion velocity loss
        if data_type == 'm2t' and config.training.und_max_t0 == 1.0:
            motion_masks = motion_masks * 0.0
        
        return xt.to(weight_type), t, motion_labels.to(weight_type), motion_masks

    @torch.no_grad()
    def prepare_motion_latents_precomputed(
        motion_latents: torch.Tensor,
        motion_lengths: torch.Tensor,
        data_type: str,
        motion_masks: torch.Tensor,
    ):
        """
        Prepare motion latents and training labels (using precomputed latents).
        
        Args:
            motion_latents: [latent_len, bs, 256] precomputed clean motion latents
            motion_lengths: [bs] original motion lengths
            data_type: 't2m_motion' or 'm2t'
            motion_masks: [bs, seq_len] motion token mask
            
        Returns:
            motion_latents: [latent_len, bs, 256] noisy motion latents
            t: [bs] timesteps
            motion_labels: [bs, latent_len, 256] velocity targets
            motion_masks: [bs, seq_len] updated masks
        """
        bs = motion_latents.shape[1]
        device = motion_latents.device
        
        # Sample timesteps for each batch element
        t0, t1 = 0.0, 1.0
        if data_type == 'm2t':
            t0 = config.training.und_max_t0  # understanding tasks use less noise
        
        # Sample t from lognorm distribution
        u = torch.normal(mean=0.0, std=1.0, size=(bs,), device=device)
        t = 1 / (1 + torch.exp(-u)) * (t1 - t0) + t0  # [bs]
        
        # Create noisy latents using the sampled timesteps
        # x0 is noise, x1 is clean motion latent
        x0 = torch.randn_like(motion_latents)  # [latent_len, bs, 256]
        x1 = motion_latents  # [latent_len, bs, 256]
        
        # Linear interpolation: xt = (1-t) * x0 + t * x1
        # Expand t for broadcasting: [bs] -> [1, bs, 1]
        t_expanded = t.view(1, bs, 1)
        xt = (1 - t_expanded) * x0 + t_expanded * x1  # [latent_len, bs, 256]
        
        # Velocity target: ut = x1 - x0 (for linear path)
        ut = x1 - x0  # [latent_len, bs, 256]
        
        # motion_labels: [bs, latent_len, 256]
        motion_labels = ut.permute(1, 0, 2)
        
        # The M2T task does not need the motion velocity loss
        if data_type == 'm2t' and config.training.und_max_t0 == 1.0:
            motion_masks = motion_masks * 0.0
        
        return xt.to(weight_type), t, motion_labels.to(weight_type), motion_masks

    @torch.no_grad()
    def prepare_motion_latents_pred(
        source_motions: torch.Tensor,
        target_motions: torch.Tensor,
        source_motion_lengths: torch.Tensor,
        target_motion_lengths: torch.Tensor,
        target_motion_masks: torch.Tensor,
    ):
        """
        Prepare latents for the Motion Prediction task.
        
        Args:
            source_motions: [bs, T1, 269] normalized source motion features
            target_motions: [bs, T2, 269] normalized target motion features
            source_motion_lengths: [bs] source motion lengths
            target_motion_lengths: [bs] target motion lengths
            target_motion_masks: [bs, seq_len] target motion token mask
            
        Returns:
            source_latents: [latent_len1, bs, 256] clean source motion latents
            target_latents: [latent_len2, bs, 256] noisy target motion latents
            t: [bs] timesteps
            target_labels: [bs, latent_len2, 256] velocity targets for target
            target_motion_masks: [bs, seq_len] updated masks
        """
        bs = source_motions.shape[0]
        device = source_motions.device
        
        # Encode source motion (clean, for understanding)
        source_latents, _ = motion_vae_encode_module.encode_motion(source_motions, source_motion_lengths.tolist())
        # source_latents: [latent_len1, bs, 256]
        
        # Encode target motion (will be noised, for generation)
        target_latents_clean, _ = motion_vae_encode_module.encode_motion(target_motions, target_motion_lengths.tolist())
        # target_latents_clean: [latent_len2, bs, 256]
        
        # Sample timesteps for target motion
        t0, t1 = 0.0, 1.0
        u = torch.normal(mean=0.0, std=1.0, size=(bs,), device=device)
        t = 1 / (1 + torch.exp(-u)) * (t1 - t0) + t0  # [bs]
        
        # Create noisy target latents
        x0 = torch.randn_like(target_latents_clean)
        x1 = target_latents_clean
        
        # Linear interpolation: xt = (1-t) * x0 + t * x1
        t_expanded = t.view(1, bs, 1)
        xt = (1 - t_expanded) * x0 + t_expanded * x1  # [latent_len2, bs, 256]
        
        # Velocity target: ut = x1 - x0
        ut = x1 - x0
        
        # target_labels: [bs, latent_len2, 256]
        target_labels = ut.permute(1, 0, 2)
        
        return (
            source_latents.to(weight_type),
            xt.to(weight_type),
            t,
            target_labels.to(weight_type),
            target_motion_masks,
        )

    @torch.no_grad()
    def prepare_image_latents_and_labels_mgie(
        source_images: torch.Tensor,
        target_images: torch.Tensor,
        target_image_masks: torch.Tensor,
    ):
        """
        Prepare MGIE image latents:
          source_image_latents: clean condition
          target_image_latents: noisy xt for diffusion
          image_labels: velocity target ut
        """
        bs = source_images.shape[0]
        device = source_images.device

        with torch.no_grad():
            source_images_5d = source_images.unsqueeze(2)
            target_images_5d = target_images.unsqueeze(2)
            source_image_latents = image_vae.sample(source_images_5d, deterministic=True).squeeze(2)
            target_clean_latents = image_vae.sample(target_images_5d, deterministic=True).squeeze(2)

        t0, t1 = 0.0, 1.0
        u = torch.normal(mean=0.0, std=1.0, size=(bs,), device=device)
        t = 1 / (1 + torch.exp(-u)) * (t1 - t0) + t0

        x0 = torch.randn_like(target_clean_latents)
        x1 = target_clean_latents
        t_expanded = t.view(bs, 1, 1, 1)
        xt = (1 - t_expanded) * x0 + t_expanded * x1
        ut = x1 - x0

        return (
            source_image_latents.to(weight_type),
            xt.to(weight_type),
            t,
            ut.to(weight_type),
            target_image_masks,
        )

    @torch.no_grad()
    def prepare_ref_motion_latents_mgie(ref_motion: torch.Tensor):
        """
        Encode single-frame reference motion and keep 2-token layout compatibility.
        """
        bs = ref_motion.shape[0]
        lengths = [1] * bs
        ref_latents, _ = motion_vae_encode_module.encode_motion(ref_motion, lengths)
        if config.model.showo.add_time_embeds:
            zero_slot = torch.zeros_like(ref_latents[:1])
            ref_latents = torch.cat([zero_slot, ref_latents], dim=0)
        return ref_latents.to(weight_type)

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        if motion_vae_train_enabled:
            motion_vae.train()
        
        for step in range(num_update_steps_per_epoch):
            # Decide which task to train based on task ratios
            task_rand = random.random()
            
            # Determine available tasks and select based on ratios
            # Stage 3 supports: T2M + M2T + T2T (plus optional pred)
            available_tasks = []
            task_ratios = []

            has_m2m_loader = (m2m_hml_iter is not None) or (m2m_h36m_iter is not None) or (m2m_mfix_iter is not None)
            if has_m2m_loader and m2m_ratio_sampling > 0:
                available_tasks.append('m2m')
                task_ratios.append(m2m_ratio_sampling)
            
            if t2m_iter is not None:
                available_tasks.append('t2m_motion')
                task_ratios.append(t2m_ratio)
            if m2t_iter is not None:
                available_tasks.append('m2t')
                task_ratios.append(m2t_ratio)
            if pred_iter is not None:
                available_tasks.append('pred')
                task_ratios.append(pred_ratio_sampling)
            if t2t_iter is not None:
                available_tasks.append('t2t')
                task_ratios.append(t2t_ratio_sampling)
            if edit_iter is not None:
                available_tasks.append('edit')
                task_ratios.append(edit_ratio_sampling)
            if img2pose_iter is not None:
                available_tasks.append('img2pose')
                task_ratios.append(img2pose_ratio_sampling)
            if img2text_iter is not None:
                available_tasks.append('img2text')
                task_ratios.append(img2text_ratio_sampling)
            if mgie_iter is not None:
                available_tasks.append('mgie')
                task_ratios.append(mgie_ratio_sampling)
            
            if len(available_tasks) == 0:
                raise ValueError("No dataloader available!")

            if any(r < 0 for r in task_ratios):
                raise ValueError(
                    f"Task sampling ratios must be non-negative, got: "
                    f"{dict(zip(available_tasks, task_ratios))}"
                )

            # Normalize the ratios (if all are 0, fall back to uniform sampling)
            total_ratio = sum(task_ratios)
            if total_ratio <= 0:
                logger.warning(
                    "All available task ratios are <= 0. Falling back to uniform sampling: "
                    f"{available_tasks}"
                )
                normalized_ratios = [1.0 / len(task_ratios)] * len(task_ratios)
            else:
                normalized_ratios = [r / total_ratio for r in task_ratios]
            
            # Select the task according to the ratios
            cumsum = 0
            data_type = available_tasks[-1]  # default to the last one
            for task, ratio in zip(available_tasks, normalized_ratios):
                cumsum += ratio
                if task_rand < cumsum:
                    data_type = task
                    break

            data_type_log = data_type
            
            # Get batch from corresponding dataloader
            if data_type == 'm2m':
                m2m_candidates = []
                if m2m_hml_iter is not None and m2m_source_weight_hml > 0:
                    m2m_candidates.append(('m2m_hml', m2m_source_weight_hml))
                if m2m_h36m_iter is not None and m2m_source_weight_h36m > 0:
                    m2m_candidates.append(('m2m_h36m', m2m_source_weight_h36m))
                if m2m_mfix_iter is not None and m2m_source_weight_mfix > 0:
                    m2m_candidates.append(('m2m_mfix', m2m_source_weight_mfix))

                if len(m2m_candidates) == 0:
                    raise ValueError("Task m2m selected but no valid m2m source iterator is available.")

                source_rand = random.random()
                source_cumsum = 0.0
                source_total = sum(item[1] for item in m2m_candidates)
                m2m_source = m2m_candidates[-1][0]
                for source_name, source_weight in m2m_candidates:
                    source_cumsum += source_weight / source_total
                    if source_rand < source_cumsum:
                        m2m_source = source_name
                        break

                data_type_log = m2m_source
                m2m_source_counter[m2m_source] += 1

                if m2m_source == 'm2m_hml':
                    try:
                        batch = next(m2m_hml_iter)
                    except StopIteration:
                        m2m_hml_iter = iter(train_dataloader_m2m_hml)
                        batch = next(m2m_hml_iter)
                elif m2m_source == 'm2m_h36m':
                    try:
                        batch = next(m2m_h36m_iter)
                    except StopIteration:
                        m2m_h36m_iter = iter(train_dataloader_m2m_h36m)
                        batch = next(m2m_h36m_iter)
                else:  # m2m_mfix
                    try:
                        batch = next(m2m_mfix_iter)
                    except StopIteration:
                        m2m_mfix_iter = iter(train_dataloader_m2m_mfix)
                        batch = next(m2m_mfix_iter)
            elif data_type == 't2m_motion':
                try:
                    batch = next(t2m_iter)
                except StopIteration:
                    t2m_iter = iter(train_dataloader_t2m)
                    batch = next(t2m_iter)
            elif data_type == 'm2t':
                try:
                    batch = next(m2t_iter)
                except StopIteration:
                    m2t_iter = iter(train_dataloader_m2t)
                    batch = next(m2t_iter)
            elif data_type == 'pred':
                try:
                    batch = next(pred_iter)
                except StopIteration:
                    pred_iter = iter(train_dataloader_pred)
                    batch = next(pred_iter)
            elif data_type == 't2t':
                try:
                    batch = next(t2t_iter)
                except StopIteration:
                    t2t_iter = iter(train_dataloader_t2t)
                    batch = next(t2t_iter)
            elif data_type == 'edit':
                try:
                    batch = next(edit_iter)
                except StopIteration:
                    edit_iter = iter(train_dataloader_edit)
                    batch = next(edit_iter)
            elif data_type == 'img2pose':
                try:
                    batch = next(img2pose_iter)
                except StopIteration:
                    img2pose_iter = iter(train_dataloader_img2pose)
                    batch = next(img2pose_iter)
            elif data_type == 'img2text':
                try:
                    batch = next(img2text_iter)
                except StopIteration:
                    img2text_iter = iter(train_dataloader_img2text)
                    batch = next(img2text_iter)
            elif data_type == 'mgie':
                try:
                    batch = next(mgie_iter)
                except StopIteration:
                    mgie_iter = iter(train_dataloader_mgie)
                    batch = next(mgie_iter)

            data_time_m.update(time.time() - end)

            text_tokens = batch['text_tokens'].to(accelerator.device)
            text_labels = batch['text_labels'].to(accelerator.device)
            text_masks = batch['text_masks'].to(accelerator.device)
            bs = text_tokens.shape[0]
            vae_loss_recon = torch.tensor(0.0, device=accelerator.device)
            vae_loss_kl = torch.tensor(0.0, device=accelerator.device)
            vae_loss_total = torch.tensor(0.0, device=accelerator.device)
            loss_img2pose_decode067 = torch.tensor(0.0, device=accelerator.device)

            # Handle different data types
            if data_type in ['pred', 'edit', 'm2m']:
                # Motion Prediction/Edit task: source motion → target motion
                source_motion = batch['source_motion'].to(accelerator.device).to(motion_vae_runtime_dtype)
                target_motion = batch['target_motion'].to(accelerator.device).to(motion_vae_runtime_dtype)
                source_motion_lengths = batch['source_motion_length'].to(accelerator.device)
                target_motion_lengths = batch['target_motion_length'].to(accelerator.device)
                source_modality_positions = batch['source_modality_positions'].to(accelerator.device)
                target_modality_positions = batch['target_modality_positions'].to(accelerator.device)
                source_motion_masks = batch['source_motion_masks'].to(accelerator.device)
                target_motion_masks = batch['target_motion_masks'].to(accelerator.device)
                
                # Prepare latents for prediction/edit
                source_latents, target_latents, t, motion_labels, motion_masks = prepare_motion_latents_pred(
                    source_motion, target_motion,
                    source_motion_lengths, target_motion_lengths,
                    target_motion_masks
                )
                
                if torch.sum(motion_masks).item() == 0:
                    logger.warning(
                        f"Motion masks are all zeros for task={data_type_log}. "
                        f"Check target_motion_masks and modality positions."
                    )
                
                motion_lengths = target_motion_lengths
                modality_positions = target_modality_positions  # for diffusion head
            elif data_type == 'img2pose':
                # Image-to-Pose task: image -> single-frame motion
                # Requires: image (encoded by the WAN21 VAE) + single-frame motion (encoded by the Motion VAE)
                images = batch['image'].to(accelerator.device).to(weight_type)  # [bs, 3, H, W]
                motions = batch['motions'].to(accelerator.device).to(motion_vae_runtime_dtype)  # [bs, 1, 269]
                if img2pose_const_mask is not None and img2pose_const_values is not None:
                    motions = _project_motion_constant_dims(
                        motions, img2pose_const_mask, img2pose_const_values
                    )
                motion_lengths = batch['motion_lengths'].to(accelerator.device)
                image_modality_positions = batch['image_modality_positions'].to(accelerator.device)
                motion_modality_positions = batch['motion_modality_positions'].to(accelerator.device)
                image_masks = batch['image_masks'].to(accelerator.device)
                motion_masks = batch['motion_masks'].to(accelerator.device)
                
                # Encode the image with the WAN21 VAE -> image_latents [bs, 16, H', W']
                with torch.no_grad():
                    # WAN21 VAE expects [B, 3, T, H, W], for single image T=1
                    images_5d = images.unsqueeze(2)  # [bs, 3, 1, H, W]
                    image_latents = image_vae.sample(images_5d, deterministic=True)  # [bs, 16, 1, H', W']
                    image_latents = image_latents.squeeze(2)  # [bs, 16, H', W']
                
                # Encode the motion with the Motion VAE -> motion_latents
                motion_latents_raw, t, motion_labels, motion_masks = prepare_motion_latents_and_labels(
                    motions, motion_lengths, 't2m_motion', motion_masks
                )
                
                modality_positions = motion_modality_positions  # for diffusion head
                
            elif data_type == 'img2text':
                # Image-to-Text task: image -> text description
                images = batch['image'].to(accelerator.device).to(weight_type)
                image_modality_positions = batch['image_modality_positions'].to(accelerator.device)
                image_masks = batch['image_masks'].to(accelerator.device)
                
                # Encode the image with the WAN21 VAE
                with torch.no_grad():
                    images_5d = images.unsqueeze(2)
                    image_latents = image_vae.sample(images_5d, deterministic=True)
                    image_latents = image_latents.squeeze(2)
                
                # img2text is a pure NTP task and does not need motion
                motion_lengths = None
                modality_positions = None
                motion_masks = None
                motion_latents = None
                t = None
                motion_labels = None

            elif data_type == 'mgie':
                # Motion-guided Image Editing:
                # text + source image + reference motion -> target image
                source_images = batch['source_image'].to(accelerator.device).to(weight_type)
                target_images = batch['target_image'].to(accelerator.device).to(weight_type)
                ref_motion = batch['ref_motion'].to(accelerator.device).to(motion_vae_runtime_dtype)

                source_image_modality_positions = batch['source_image_modality_positions'].to(accelerator.device)
                ref_motion_modality_positions = batch['ref_motion_modality_positions'].to(accelerator.device)
                target_image_modality_positions = batch['target_image_modality_positions'].to(accelerator.device)
                target_image_masks = batch['target_image_masks'].to(accelerator.device)
                ref_motion_masks = batch['ref_motion_masks'].to(accelerator.device)

                source_image_latents, target_image_latents, t, image_labels, target_image_masks = \
                    prepare_image_latents_and_labels_mgie(
                        source_images=source_images,
                        target_images=target_images,
                        target_image_masks=target_image_masks,
                    )
                ref_motion_latents = prepare_ref_motion_latents_mgie(ref_motion)

                motion_lengths = None
                modality_positions = target_image_modality_positions
                motion_masks = None
                motion_latents = None
                motion_labels = None
                
            elif data_type == 't2t':
                # T2T is a pure text task and does not need motion-related fields;
                # skip motion processing here, it is handled later in the forward stage
                motion_lengths = None
                modality_positions = None
                motion_masks = None
                motion_latents = None
                t = None
                motion_labels = None
            else:
                # T2M or M2T task
                motion_lengths = batch['motion_lengths'].to(accelerator.device)
                modality_positions = batch['modality_positions'].to(accelerator.device)
                motion_masks = batch['motion_masks'].to(accelerator.device)

                # Check if using precomputed latents or raw motions
                if 'motion_latents' in batch:
                    # Using precomputed latents - directly prepare with noise
                    precomputed_latents = batch['motion_latents'].to(accelerator.device).to(weight_type)
                    # precomputed_latents: [bs, latent_len, 256], need to transpose to [latent_len, bs, 256]
                    precomputed_latents = precomputed_latents.permute(1, 0, 2)
                    motion_latents, t, motion_labels, motion_masks = prepare_motion_latents_precomputed(
                        precomputed_latents, motion_lengths, data_type, motion_masks
                    )
                else:
                    # Using raw motions - need VAE encoding
                    motions = batch['motions'].to(accelerator.device).to(motion_vae_runtime_dtype)
                    motion_latents, t, motion_labels, motion_masks = prepare_motion_latents_and_labels(
                        motions, motion_lengths, data_type, motion_masks
                    )

            run_vae_branch = (
                motion_vae_train_enabled
                and vae_loss_enabled
                and data_type == "m2m"
                and "m2m" in motion_vae_task_whitelist
            )
            if run_vae_branch:
                vae_lengths = target_motion_lengths.tolist()
                vae_features = target_motion.float() if motion_vae_keep_fp32 else target_motion
                vae_out = motion_vae(
                    features=vae_features,
                    lengths=vae_lengths,
                    has_video=None,
                )
                vae_target = target_motion.float() if motion_vae_keep_fp32 else target_motion
                vae_loss_recon, vae_loss_kl = _compute_motion_vae_losses(
                    recon_motion=vae_out["motion_recon"],
                    target_motion=vae_target,
                    lengths=vae_lengths,
                    dist_motion=vae_out["dist_motion"],
                    downsample_rate=motion_vae_config["downsample_rate"],
                )
                kl_w = min(1.0, float(global_step + 1) / float(vae_kl_warmup_steps))
                vae_loss_total = vae_recon_weight * vae_loss_recon + vae_kl_weight * kl_w * vae_loss_kl

            # Forward pass based on data type
            if data_type in ['pred', 'edit', 'm2m']:
                # Motion Prediction/Edit: source + target, need special attention mask
                # Combine source and target positions for attention mask
                # Create combined modality positions for attention mask
                combined_modality_positions = torch.cat([
                    source_modality_positions, 
                    target_modality_positions
                ], dim=1)
                
                block_mask = omni_attn_mask_naive(
                    text_tokens.size(0),
                    text_tokens.size(1),
                    combined_modality_positions,
                    accelerator.device
                ).to(weight_type)
                
                # Forward pass for motion editing/prediction
                logits, loss_ntp, loss_motion = model(
                    text_tokens=text_tokens,
                    motion_latents=target_latents,  # target (noisy)
                    t=t.to(weight_type),
                    attention_mask=block_mask,
                    motion_lengths=target_motion_lengths.tolist(),
                    text_labels=text_labels,
                    motion_labels=motion_labels,
                    modality_positions=target_modality_positions,
                    text_masks=text_masks,
                    motion_masks=motion_masks,
                    output_hidden_states=True,
                    max_seq_len=text_tokens.size(1),
                    device=accelerator.device,
                    data_type='motion_edit',  # use the motion_edit mode
                    # Motion edit specific kwargs
                    source_motion_latents=source_latents,  # source (clean)
                    source_motion_lengths=source_motion_lengths.tolist(),
                    source_modality_positions=source_modality_positions,
                )
            elif data_type == 't2m_motion':
                # Create attention mask
                block_mask = omni_attn_mask_naive(
                    text_tokens.size(0),
                    text_tokens.size(1),
                    modality_positions,
                    accelerator.device
                ).to(weight_type)
                
                logits, loss_ntp, loss_motion = model(
                    text_tokens=text_tokens,
                    motion_latents=motion_latents,
                    t=t.to(weight_type),
                    attention_mask=block_mask,
                    motion_lengths=motion_lengths.tolist(),
                    text_labels=text_labels,
                    motion_labels=motion_labels,
                    modality_positions=modality_positions,
                    text_masks=text_masks,
                    motion_masks=motion_masks,
                    output_hidden_states=True,
                    max_seq_len=text_tokens.size(1),
                    device=accelerator.device,
                    data_type=data_type,
                )
            elif data_type == 'img2pose':
                # Image-to-Pose: image -> single-frame motion (diffusion loss)
                # Merge the image and motion modality positions for the attention mask
                combined_positions = torch.cat([
                    image_modality_positions, motion_modality_positions
                ], dim=1)
                block_mask = omni_attn_mask_naive(
                    text_tokens.size(0),
                    text_tokens.size(1),
                    combined_positions,
                    accelerator.device
                ).to(weight_type)
                
                # Use model(...) instead of unwrapped_model so DDP gradient sync works correctly on multiple GPUs
                if img2pose_decode067_effective:
                    logits, loss_ntp, loss_motion, img2pose_v_pred = model(
                        text_tokens=text_tokens,
                        image_latents=image_latents.to(weight_type),
                        motion_latents=motion_latents_raw,
                        pose_images=images,
                        return_motion_pred=True,
                        t=t.to(weight_type),
                        attention_mask=block_mask,
                        motion_lengths=motion_lengths.tolist(),
                        text_labels=text_labels,
                        motion_labels=motion_labels,
                        modality_positions=motion_modality_positions,
                        text_masks=text_masks,
                        motion_masks=motion_masks,
                        output_hidden_states=True,
                        max_seq_len=text_tokens.size(1),
                        device=accelerator.device,
                        data_type='img2pose',
                        image_modality_positions=image_modality_positions,
                        motion_modality_positions=motion_modality_positions,
                    )
                else:
                    img2pose_v_pred = None
                    logits, loss_ntp, loss_motion = model(
                        text_tokens=text_tokens,
                        image_latents=image_latents.to(weight_type),
                        motion_latents=motion_latents_raw,
                        pose_images=images,
                        t=t.to(weight_type),
                        attention_mask=block_mask,
                        motion_lengths=motion_lengths.tolist(),
                        text_labels=text_labels,
                        motion_labels=motion_labels,
                        modality_positions=motion_modality_positions,
                        text_masks=text_masks,
                        motion_masks=motion_masks,
                        output_hidden_states=True,
                        max_seq_len=text_tokens.size(1),
                        device=accelerator.device,
                        data_type='img2pose',
                        image_modality_positions=image_modality_positions,
                        motion_modality_positions=motion_modality_positions,
                    )
                
            elif data_type == 'img2text':
                # Image-to-Text: image -> text description (NTP loss only)
                block_mask = omni_attn_mask_naive(
                    text_tokens.size(0),
                    text_tokens.size(1),
                    image_modality_positions,
                    accelerator.device
                ).to(weight_type)
                
                # Use model(...) instead of unwrapped_model so DDP gradient sync works correctly on multiple GPUs
                logits, loss_ntp = model(
                    text_tokens=text_tokens,
                    image_latents=image_latents.to(weight_type),
                    pose_images=images,
                    attention_mask=block_mask,
                    text_labels=text_labels,
                    text_masks=text_masks,
                    output_hidden_states=False,
                    max_seq_len=text_tokens.size(1),
                    device=accelerator.device,
                    data_type='img2text',
                    image_modality_positions=image_modality_positions,
                )
                loss_motion = torch.tensor(0.0).to(accelerator.device)

            elif data_type == 'mgie':
                combined_positions = torch.cat([
                    source_image_modality_positions,
                    ref_motion_modality_positions,
                    target_image_modality_positions,
                ], dim=1)
                block_mask = omni_attn_mask_naive(
                    text_tokens.size(0),
                    text_tokens.size(1),
                    combined_positions,
                    accelerator.device
                ).to(weight_type)

                logits, loss_ntp, loss_motion = model(
                    text_tokens=text_tokens,
                    image_latents=target_image_latents,
                    t=t.to(weight_type),
                    attention_mask=block_mask,
                    text_labels=text_labels,
                    image_labels=image_labels,
                    text_masks=text_masks,
                    image_masks=target_image_masks,
                    modality_positions=target_image_modality_positions,
                    output_hidden_states=True,
                    max_seq_len=text_tokens.size(1),
                    device=accelerator.device,
                    data_type='mgie',
                    source_image_latents=source_image_latents,
                    source_pose_images=source_images,
                    ref_motion_latents=ref_motion_latents,
                    source_image_modality_positions=source_image_modality_positions,
                    ref_motion_modality_positions=ref_motion_modality_positions,
                    target_image_modality_positions=target_image_modality_positions,
                    target_image_masks=target_image_masks,
                )
                
            elif data_type == 'm2t':
                # Create attention mask
                block_mask = omni_attn_mask_naive(
                    text_tokens.size(0),
                    text_tokens.size(1),
                    modality_positions,
                    accelerator.device
                ).to(weight_type)
                
                logits, loss_ntp = model(
                    text_tokens=text_tokens,
                    motion_latents=motion_latents,
                    t=t.to(weight_type),
                    attention_mask=block_mask,
                    motion_lengths=motion_lengths.tolist(),
                    text_labels=text_labels,
                    modality_positions=modality_positions,
                    output_hidden_states=True,
                    max_seq_len=text_tokens.size(1),
                    device=accelerator.device,
                    data_type=data_type,
                )
                loss_motion = torch.tensor(0.0).to(accelerator.device)
            else:  # t2t
                # Text-to-Text task (Stage 3)
                # Pure text task: no motion latents needed, only the NTP loss is computed.
                # Uses a causal attention mask (standard language model).

                # Create the causal attention mask
                causal_mask = torch.tril(
                    torch.ones(text_tokens.size(1), text_tokens.size(1), device=accelerator.device)
                ).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
                causal_mask = causal_mask.expand(bs, 1, -1, -1).to(weight_type)
                
                # T2T task: only the language-model forward pass is needed.
                # Call the model's text forward method (pure text processing).
                # Note: under DDP, unwrap_model is required first to access custom methods.
                unwrapped_model = accelerator.unwrap_model(model)
                logits, loss_ntp = unwrapped_model.t2t_forward(
                    text_tokens=text_tokens,
                    text_labels=text_labels,
                    attention_mask=causal_mask,
                    text_masks=text_masks,
                )
                loss_motion = torch.tensor(0.0).to(accelerator.device)

            if data_type == 'img2pose' and img2pose_decode067_effective:
                if img2pose_v_pred is None:
                    raise RuntimeError("img2pose decoded 0:67 supervision expects return_motion_pred=True.")
                if img2pose_v_pred.shape != motion_latents_raw.shape:
                    raise RuntimeError(
                        "img2pose decoded supervision shape mismatch: "
                        f"v_pred={tuple(img2pose_v_pred.shape)}, xt={tuple(motion_latents_raw.shape)}"
                    )

                t_lat = t.to(dtype=img2pose_v_pred.dtype).view(1, bs, 1)
                x_t = motion_latents_raw.to(dtype=img2pose_v_pred.dtype)
                u_target = motion_labels.permute(1, 0, 2).to(dtype=img2pose_v_pred.dtype)
                x1_pred = x_t + (1.0 - t_lat) * img2pose_v_pred
                x1_target = x_t + (1.0 - t_lat) * u_target

                decode_lengths = motion_lengths.tolist()
                decode_dtype = next(motion_vae_decode_module.parameters()).dtype
                pred_feats = motion_vae_decode_module.decode(x1_pred.to(decode_dtype), decode_lengths)
                with torch.no_grad():
                    target_feats = motion_vae_decode_module.decode(
                        x1_target.detach().to(decode_dtype), decode_lengths
                    )

                pred_067 = pred_feats[..., :img2pose_decode067_dims]
                target_067 = target_feats[..., :img2pose_decode067_dims]
                valid_mask = _build_lengths_mask(
                    decode_lengths, pred_067.shape[1], pred_067.device
                ).unsqueeze(-1).to(dtype=pred_067.dtype)

                if img2pose_decode067_loss_type == "l1":
                    elem = torch.abs(pred_067 - target_067)
                elif img2pose_decode067_loss_type == "mse":
                    elem = (pred_067 - target_067).pow(2)
                else:
                    elem = F.smooth_l1_loss(pred_067, target_067, reduction="none")
                valid_mask_expanded = valid_mask.expand_as(elem)
                loss_img2pose_decode067 = (
                    (elem * valid_mask_expanded).sum()
                    / valid_mask_expanded.sum().clamp_min(1.0)
                )

                if img2pose_decode067_debug_every > 0 and accelerator.is_main_process:
                    if ((global_step + 1) % img2pose_decode067_debug_every) == 0:
                        logger.info(
                            "[img2pose_decode067] "
                            f"step={global_step + 1}, "
                            f"pred_shape={tuple(pred_067.shape)}, "
                            f"target_shape={tuple(target_067.shape)}, "
                            f"loss_raw={float(loss_img2pose_decode067.detach().item()):.6f}"
                        )

            # Calculate total loss
            # Handle NaN: in the T2M task loss_ntp is NaN (there is no text to predict); replace with 0
            loss_ntp_safe = torch.where(torch.isnan(loss_ntp), torch.zeros_like(loss_ntp), loss_ntp)
            loss_motion_safe = torch.where(torch.isnan(loss_motion), torch.zeros_like(loss_motion), loss_motion)
            vae_loss_recon_safe = torch.where(torch.isnan(vae_loss_recon), torch.zeros_like(vae_loss_recon), vae_loss_recon)
            vae_loss_kl_safe = torch.where(torch.isnan(vae_loss_kl), torch.zeros_like(vae_loss_kl), vae_loss_kl)
            vae_loss_total_safe = torch.where(torch.isnan(vae_loss_total), torch.zeros_like(vae_loss_total), vae_loss_total)
            loss_img2pose_decode067_safe = torch.where(
                torch.isnan(loss_img2pose_decode067),
                torch.zeros_like(loss_img2pose_decode067),
                loss_img2pose_decode067,
            )
            if img2pose_decode067_effective and data_type == 'img2pose':
                if img2pose_decode067_warmup_steps > 0:
                    decode_aux_weight = img2pose_decode067_weight * min(
                        1.0, float(global_step + 1) / float(img2pose_decode067_warmup_steps)
                    )
                else:
                    decode_aux_weight = img2pose_decode067_weight
            else:
                decode_aux_weight = 0.0
            loss = (
                config.training.ntp_coeff * loss_ntp_safe
                + config.training.flow_coeff * loss_motion_safe
                + vae_loss_coeff * vae_loss_total_safe
                + decode_aux_weight * loss_img2pose_decode067_safe
            )

            # Gather losses for logging
            avg_loss_ntp = accelerator.gather(loss_ntp_safe.repeat(bs)).mean()
            avg_loss_motion = accelerator.gather(loss_motion_safe.repeat(bs)).mean() if data_type in ['t2m_motion', 'pred', 'edit', 'm2m', 'img2pose', 'mgie'] else torch.tensor(0.0)
            avg_loss_vae_recon = accelerator.gather(vae_loss_recon_safe.repeat(bs)).mean()
            avg_loss_vae_kl = accelerator.gather(vae_loss_kl_safe.repeat(bs)).mean()
            avg_loss_vae_total = accelerator.gather(vae_loss_total_safe.repeat(bs)).mean()
            avg_loss_img2pose_decode067 = (
                accelerator.gather(loss_img2pose_decode067_safe.repeat(bs)).mean()
                if data_type == 'img2pose' else torch.tensor(0.0)
            )

            use_deepspeed = accelerator.distributed_type == DistributedType.DEEPSPEED
            if use_deepspeed:
                loss_for_backward = loss.to(weight_type)
            else:
                loss_for_backward = loss.to(weight_type) / config.training.gradient_accumulation_steps

            accelerator.backward(loss_for_backward)

            if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                if motion_vae_train_enabled:
                    clip_params = list(model.parameters()) + list(motion_vae.parameters())
                else:
                    clip_params = model.parameters()
                accelerator.clip_grad_norm_(clip_params, config.training.max_grad_norm)

            if accelerator.sync_gradients:
                optimizer.step()
                lr_scheduler.step()

            # Log gradient norm
            if (
                accelerator.sync_gradients
                and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                and accelerator.is_main_process
            ):
                log_grad_norm(model, accelerator, global_step + 1)

            if accelerator.sync_gradients:
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step
            if accelerator.sync_gradients:
                batch_time_m.update(time.time() - end)
                end = time.time()

                # Log metrics
                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                        config.training.gradient_accumulation_steps * bs / batch_time_m.val
                    )
                    lr_dict = {}
                    for group in optimizer.param_groups:
                        group_name = group.get("name", "unnamed")
                        lr_dict[group_name] = group["lr"]
                    
                    logs = {
                        "step_loss_ntp": avg_loss_ntp.item(),
                        # Primary metric name for motion generation / denoising loss.
                        "step_loss_motion": avg_loss_motion.item(),
                        # Historical alias kept for compatibility with existing plots.
                        "step_loss_flow": avg_loss_motion.item(),
                        "step_loss_vae_recon": avg_loss_vae_recon.item(),
                        "step_loss_vae_kl": avg_loss_vae_kl.item(),
                        "step_loss_vae_total": avg_loss_vae_total.item(),
                        "step_loss_img2pose_decode067": avg_loss_img2pose_decode067.item(),
                        "img2pose_decode067_weight": decode_aux_weight,
                        "task": data_type_log,
                        "lr": lr_dict.get("base", optimizer.param_groups[0]["lr"]),
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                    }
                    if "lora" in lr_dict:
                        logs["lr_lora"] = lr_dict["lora"]
                    if "motion_vae" in lr_dict:
                        logs["lr_motion_vae"] = lr_dict["motion_vae"]
                    accelerator.log(logs, step=global_step + 1)
                    lr_items = ", ".join([f"{k}={v:0.6f}" for k, v in sorted(lr_dict.items())])
                    logger.info(
                        f"Epoch: {epoch} "
                        f"Step: {global_step + 1} "
                        f"Task: {data_type_log} "
                        f"Loss_NTP: {avg_loss_ntp.item():0.4f} "
                        f"Loss_Motion: {avg_loss_motion.item():0.4f} "
                        f"Loss_I2P067: {avg_loss_img2pose_decode067.item():0.4f} "
                        f"Loss_VAE_Recon: {avg_loss_vae_recon.item():0.4f} "
                        f"Loss_VAE_KL: {avg_loss_vae_kl.item():0.4f} "
                        f"Loss_VAE: {avg_loss_vae_total.item():0.4f} "
                        f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_m.val:0.4f} "
                        f"LRs: {lr_items}"
                    )
                    if sum(m2m_source_counter.values()) > 0:
                        logger.info(
                            f"M2M source counters: "
                            f"hml={m2m_source_counter['m2m_hml']}, "
                            f"h36m={m2m_source_counter['m2m_h36m']}, "
                            f"mfix={m2m_source_counter['m2m_mfix']}"
                        )
                    
                    batch_time_m.reset()
                    data_time_m.reset()

                # Generate samples
                # if (global_step + 1) % config.experiment.generate_every == 0:
                #     generate_motion_samples(
                #         model, motion_vae, text_tokenizer, config, 
                #         global_step + 1, accelerator.device, weight_type, 
                #         sampler, showo_token_ids
                #     )
                if (global_step + 1) % config.experiment.generate_every == 0:
                    generate_motion_samples(
                        model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model, 
                        motion_vae, text_tokenizer, config, 
                        global_step + 1, accelerator.device, weight_type, 
                        sampler, showo_token_ids,
                        mean=motion_mean,
                        std=motion_std,
                    )

                # Save checkpoint
                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(
                        model,
                        config,
                        accelerator,
                        global_step + 1,
                        motion_vae=motion_vae,
                        motion_vae_config=motion_vae_config,
                        save_motion_vae=motion_vae_train_enabled,
                    )

                global_step += 1

            if global_step >= config.training.max_train_steps:
                break

    accelerator.wait_for_everyone()

    # Save final checkpoint
    save_checkpoint(
        model,
        config,
        accelerator,
        "final",
        motion_vae=motion_vae,
        motion_vae_config=motion_vae_config,
        save_motion_vae=motion_vae_train_enabled,
    )

    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(config.experiment.output_dir, safe_serialization=False)

    accelerator.end_training()


@torch.no_grad()
def generate_motion_samples(
    model,
    motion_vae,
    text_tokenizer,
    config,
    global_step,
    device,
    weight_type,
    sampler,
    showo_token_ids,
    mean=None,
    std=None,
):
    """Generate motion samples for visualization and save PNG sequences."""
    logger.info("Generating motion samples...")
    model.eval()
    # DDP wrapper does not expose custom decode method.
    motion_vae_decode_module = motion_vae.module if hasattr(motion_vae, "module") else motion_vae

    # Recompute runtime motion token geometry locally so this helper does not
    # depend on outer-scope variables from main().
    motion_vae_downsample_rate = int(config.model.motion_vae.downsample_rate)
    base_max_motion_tokens = latent_len_from_motion_length(
        int(config.dataset.motion.max_motion_length),
        motion_vae_downsample_rate,
    )
    num_motion_tokens = base_max_motion_tokens + int(config.model.showo.add_time_embeds)

    # Test prompts
    test_prompts = [
        "A person walks forward slowly.",
        "A person jumps up and lands.",
        "A person waves their right hand.",
        "A person sits down on a chair.",
    ]

    # Prepare inputs
    bos_id = showo_token_ids['bos_id']
    eos_id = showo_token_ids['eos_id']
    bom_id = showo_token_ids['bom_id']
    eom_id = showo_token_ids['eom_id']
    pad_id = text_tokenizer.pad_token_id
    mot_pad_id = showo_token_ids['mot_pad_id']

    max_text_len = config.dataset.preprocessing.max_text_len
    max_seq_len = config.dataset.preprocessing.max_seq_length

    batch_text_tokens = []
    batch_modality_positions = []
    
    for prompt in test_prompts:
        text_tokens = text_tokenizer(prompt, add_special_tokens=False)['input_ids'][:max_text_len]
        
        # Format: [BOS] text [BOM][MOT_PAD*N][EOM][EOS][PAD...]
        modality_positions = torch.tensor([len(text_tokens) + 2, num_motion_tokens]).unsqueeze(0)
        text_tokens = [bos_id] + text_tokens + [bom_id] + [mot_pad_id] * num_motion_tokens + \
                      [eom_id] + [eos_id] + [pad_id] * (max_seq_len - len(text_tokens) - num_motion_tokens - 4)
        
        batch_text_tokens.append(torch.tensor(text_tokens[:max_seq_len]))
        batch_modality_positions.append(modality_positions)

    batch_text_tokens = torch.stack(batch_text_tokens, dim=0).to(device)
    batch_modality_positions = torch.stack(batch_modality_positions, dim=0).to(device)

    # Create attention mask
    block_mask = omni_attn_mask_naive(
        batch_text_tokens.size(0),
        batch_text_tokens.size(1),
        batch_modality_positions,
        device
    ).to(weight_type)

    # Initialize noise
    latent_len = num_motion_tokens - 1 if config.model.showo.add_time_embeds else num_motion_tokens
    z = torch.randn((len(test_prompts), latent_len, config.model.motion_vae.motion_latent_dim)).to(weight_type).to(device)
    # [bs, latent_len, 256] -> [latent_len, bs, 256]
    z = z.permute(1, 0, 2)

    # CFG setup
    guidance_scale = config.transport.guidance_scale
    if guidance_scale > 0:
        z = torch.cat([z, z], dim=1)  # [latent_len, 2*bs, 256]
        
        # Create null condition
        batch_text_tokens_null = []
        batch_modality_positions_null = []
        for _ in test_prompts:
            modality_positions_null = torch.tensor([2, num_motion_tokens]).unsqueeze(0)
            text_tokens_null = [bos_id] + [bom_id] + [mot_pad_id] * num_motion_tokens + \
                              [eom_id] + [eos_id] + [pad_id] * (max_seq_len - num_motion_tokens - 4)
            batch_text_tokens_null.append(torch.tensor(text_tokens_null[:max_seq_len]))
            batch_modality_positions_null.append(modality_positions_null)
        
        batch_text_tokens_null = torch.stack(batch_text_tokens_null, dim=0).to(device)
        batch_modality_positions_null = torch.stack(batch_modality_positions_null, dim=0).to(device)
        
        text_tokens = torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0)
        modality_positions = torch.cat([batch_modality_positions, batch_modality_positions_null], dim=0)
        
        block_mask = omni_attn_mask_naive(
            text_tokens.size(0),
            text_tokens.size(1),
            modality_positions,
            device
        ).to(weight_type)
    else:
        text_tokens = batch_text_tokens
        modality_positions = batch_modality_positions

    # Prepare model kwargs
    model_kwargs = dict(
        text_tokens=text_tokens,
        attention_mask=block_mask,
        modality_positions=modality_positions,
        output_hidden_states=True,
        max_seq_len=max_seq_len,
        guidance_scale=guidance_scale
    )

    # ODE sampling
    sample_fn = sampler.sample_ode(
        sampling_method=config.transport.sampling_method,
        num_steps=config.transport.num_inference_steps,
        atol=config.transport.atol,
        rtol=config.transport.rtol,
        reverse=config.transport.reverse,
        time_shifting_factor=config.transport.time_shifting_factor
    )
    
    samples = sample_fn(z, model.t2m_generate, **model_kwargs)[-1]
    
    if guidance_scale > 0:
        samples = samples[:, :len(test_prompts), :]  # Take conditional samples

    # Decode with Motion VAE
    # samples: [latent_len, bs, 256]
    
    # Compute the correct motion_lengths so they match latent_len
    # latent_lengths = ceil(motion_length / downsample_rate) should equal samples.shape[0]
    actual_latent_len = samples.shape[0]
    downsample_rate = motion_vae_downsample_rate
    motion_lengths = [actual_latent_len * downsample_rate] * len(test_prompts)
    
    # Convert to float32 for decoding (motion_vae uses native PyTorch modules internally, which need float32)
    motion_vae_dtype = next(motion_vae_decode_module.parameters()).dtype
    with torch.autocast(device_type='cuda', enabled=False):
        decoded_motions = motion_vae_decode_module.float().decode(samples.float(), motion_lengths)
    motion_vae_decode_module.to(motion_vae_dtype)  # restore the original precision
    # decoded_motions: [bs, max_len, 269]

    # ==================== Visualize and Save PNGs ====================
    try:
        # Create output directory for this step
        vis_dir = os.path.join(config.experiment.output_dir, 'visualizations', f'step_{global_step}')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Convert to numpy
        if isinstance(decoded_motions, torch.Tensor):
            decoded_motions_np = decoded_motions.cpu().numpy()
        else:
            decoded_motions_np = np.array([m.cpu().numpy() for m in decoded_motions])
        
        all_joints = []
        sample_info = []  # collect sample info for saving
        
        # Generate PNG sequences for each sample
        for sample_idx, (motion, prompt) in enumerate(zip(decoded_motions_np, test_prompts)):
            sample_vis_dir = os.path.join(vis_dir, f'sample_{sample_idx:02d}')
            
            # Save PNG sequence (use actual motion length, not fixed 50 frames)
            # motion.shape[0] is the actual number of frames from VAE decoder
            actual_motion_len = motion.shape[0]
            saved_frames = visualize_motion_to_png(
                motion_features=motion,
                save_dir=sample_vis_dir,
                prompt=prompt,
                sample_idx=sample_idx,
                mean=mean,
                std=std,
                fps=config.dataset.motion.get('fps', 20),
                max_frames=actual_motion_len,  # Use actual motion length
            )
            
            # Save prompt text file for each sample
            prompt_path = os.path.join(sample_vis_dir, 'prompt.txt')
            with open(prompt_path, 'w', encoding='utf-8') as f:
                f.write(prompt)
            
            # Collect sample info
            sample_info.append({
                'sample_idx': sample_idx,
                'prompt': prompt,
                'num_frames': len(saved_frames),
                'motion_length': motion.shape[0],
            })
            
            # Also collect joints for grid visualization
            motion_tensor = torch.from_numpy(motion).float()
            if mean is not None and std is not None:
                mean_t = torch.from_numpy(mean).float() if isinstance(mean, np.ndarray) else mean
                std_t = torch.from_numpy(std).float() if isinstance(std, np.ndarray) else std
                motion_tensor = motion_tensor * std_t + mean_t
            joints = recover_from_ric(motion_tensor.unsqueeze(0), joints_num=22)
            all_joints.append(joints.squeeze(0).numpy())
            
            logger.info(f"  Saved {len(saved_frames)} frames for sample {sample_idx}: '{prompt[:30]}...'")
        
        # Create grid visualization (selected frames from all samples)
        # Use relative frame positions based on actual motion length
        grid_path = os.path.join(vis_dir, 'motion_grid.png')
        create_motion_grid(
            all_joints=all_joints,
            prompts=test_prompts,
            save_path=grid_path,
            frame_indices=None,  # Auto-select frames based on motion length
        )
        logger.info(f"  Saved motion grid to {grid_path}")
        
        # Save summary file with all prompts and metadata
        summary_path = os.path.join(vis_dir, 'samples_info.json')
        summary_data = {
            'step': global_step,
            'num_samples': len(test_prompts),
            'guidance_scale': config.transport.guidance_scale,
            'num_inference_steps': config.transport.num_inference_steps,
            'mean_std_source': 'training (data_root/Mean.npy, Std.npy)',
            'samples': sample_info,
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        logger.info(f"  Saved samples info to {summary_path}")
        
        # Log to wandb
        if wandb.run is not None:
            # Verify image is readable before passing to wandb
            def safe_load_wandb_image(path, max_retries=5):
                """Safely load image for wandb with retry logic."""
                import time
                from PIL import Image
                for attempt in range(max_retries):
                    try:
                        with Image.open(path) as img:
                            img.verify()
                        return wandb.Image(path)
                    except Exception as e:
                        if attempt < max_retries - 1:
                            time.sleep(0.2)
                        else:
                            logger.warning(f"Failed to load image {path} after {max_retries} attempts: {e}")
                            return None
            
            # Log grid image
            grid_image = safe_load_wandb_image(grid_path)
            if grid_image is not None:
                wandb.log({
                    "motion_grid": grid_image,
                    "generated_motion_lengths": motion_lengths,
                    "test_prompts": test_prompts,
                }, step=global_step)
            else:
                wandb.log({
                    "generated_motion_lengths": motion_lengths,
                    "test_prompts": test_prompts,
                }, step=global_step)
            
            # Log individual sample first frames
            for sample_idx, prompt in enumerate(test_prompts):
                first_frame_path = os.path.join(
                    vis_dir, f'sample_{sample_idx:02d}', 
                    f'sample_{sample_idx:02d}_frame_000.png'
                )
                if os.path.exists(first_frame_path):
                    try:
                        from PIL import Image
                        with Image.open(first_frame_path) as img:
                            img.verify()
                        wandb.log({
                            f"motion_sample_{sample_idx}": wandb.Image(
                                first_frame_path, 
                                caption=prompt[:50]
                            ),
                        }, step=global_step)
                    except Exception as e:
                        logger.warning(f"Failed to log sample {sample_idx}: {e}")
        
        logger.info(f"Motion visualization saved to {vis_dir}")
        
    except Exception as e:
        logger.warning(f"Failed to visualize motion: {e}")
        import traceback
        traceback.print_exc()

    model.train()
    logger.info("Motion generation complete")


def save_checkpoint(
    model,
    config,
    accelerator,
    global_step,
    motion_vae=None,
    motion_vae_config=None,
    save_motion_vae=False,
):
    """Save a model checkpoint."""
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir) if os.path.exists(output_dir) else []
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint") and d != "checkpoint-final"]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False
        )
        
        # Save LoRA weights separately (if LoRA is enabled)
        if hasattr(unwrapped_model, 'lora_enabled') and unwrapped_model.lora_enabled:
            lora_save_path = save_path / "lora_weights.pt"
            unwrapped_model.save_lora(str(lora_save_path))
            logger.info(f"  Saved LoRA weights to {lora_save_path}")

        if save_motion_vae and motion_vae is not None:
            unwrapped_motion_vae = accelerator.unwrap_model(motion_vae)
            motion_state = unwrapped_motion_vae.state_dict()
            motion_ckpt = {
                "state_dict": {f"vae.{k}": v.detach().cpu() for k, v in motion_state.items()},
                "motion_vae_config": dict(motion_vae_config or {}),
                "global_step": global_step,
            }
            motion_ckpt_path = save_path / "motion_vae.ckpt"
            torch.save(motion_ckpt, motion_ckpt_path)
            logger.info(f"  Saved Motion VAE weights to {motion_ckpt_path}")
        
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")


def log_grad_norm(model, accelerator, global_step):
    """Log gradient norms."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


if __name__ == "__main__":
    main()
