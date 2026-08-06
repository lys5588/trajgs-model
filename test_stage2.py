# coding=utf-8
# Copyright 2026 UniMotion Authors.
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
Stage 2a/2b Unified Evaluation Script for UniMotion (variant without instruction templates).

Generic evaluation script usable for both Stage 2a and Stage 2b.
Distinguished via different config yaml files:
- Stage 2a: configs/showo2_motion_test_stage2a.yaml (fixed prompts)
- Stage 2b: configs/showo2_motion_test_stage2b.yaml (diverse prompts)

Supports 3 tasks:
- T2M (Text-to-Motion): FID, R-Precision, Matching Score, Diversity
- M2T (Motion-to-Text): R-Precision, Matching Score, ROUGE-L, CIDEr, Bleu, BertScore
- Pred (Motion Prediction): FID, DIV, ADE, FDE (APD optional)

Fully consistent with the MotionGPT3 evaluation protocol: repeat
replication_times times and report mean +/- 95% CI.

Usage:
    # Stage 2a evaluation:
    python test_stage2.py config=configs/showo2_motion_test_stage2a.yaml

    # Stage 2b evaluation:
    python test_stage2.py config=configs/showo2_motion_test_stage2b.yaml

    # Quick test:
    python test_stage2.py config=configs/showo2_motion_test_stage2a.yaml \
        test.quick=True test.num_quick_samples=64

    # Test only the specified tasks:
    python test_stage2.py config=configs/showo2_motion_test_stage2a.yaml test.tasks=[t2m]

    # Multi-GPU test:
    torchrun --nproc_per_node=8 test_stage2.py \
        config=configs/showo2_motion_test_stage2a.yaml

    # Specify a checkpoint:
    python test_stage2.py config=configs/showo2_motion_test_stage2a.yaml \
        test.model_path=/path/to/checkpoint/unwrapped_model
"""

import os
import sys
import json
import hashlib
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from models import Showo2WithMotion, omni_attn_mask_naive
from models.misc import get_text_tokenizer
from models.motion_vae import load_motion_vae
from models.lora import count_lora_parameters
from utils import get_config, load_state_dict, path_to_llm_name, set_seed
from transport import Sampler, create_transport
from motion_length_utils import latent_len_from_motion_length, latent_lens_from_motion_lengths

from evaluation.t2m_evaluator import T2MEvaluator, WordVectorizer
from evaluation.m2t_evaluator import M2TEvaluator
from evaluation.pred_evaluator import PredEvaluator, feats2joints_humanml3d
from evaluation.save_vis_utils import SaveVisManager
from datasets.motion_eval_dataset import MotionEvalDataset

os.environ["TOKENIZERS_PARALLELISM"] = "true"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAVE_VIS_ROOT = os.path.join(PROJECT_ROOT, "outputs", "save_vis")

# Suppress frequent irrelevant warnings to avoid log spam in multi-GPU runs
warnings.filterwarnings(
    "ignore",
    message=".*position_embeddings.*will be mandatory.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass


# ==================== Distributed Utilities ====================

def setup_distributed():
    """Initialize the distributed environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check whether this is the main process."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def sync_run_timestamp(local_timestamp: str) -> str:
    """Broadcast a unified run timestamp across ranks in multi-GPU runs."""
    if not dist.is_initialized():
        return local_timestamp
    obj_list = [local_timestamp if is_main_process() else None]
    dist.broadcast_object_list(obj_list, src=0)
    return str(obj_list[0])


def gather_embeddings(embeddings_list, world_size):
    """Gather embeddings from all processes to the main process."""
    if world_size == 1:
        if len(embeddings_list) == 0:
            return None
        return torch.cat(embeddings_list, dim=0)
    
    if len(embeddings_list) == 0:
        local_embeddings = torch.zeros(0, 512).cuda()
    else:
        local_embeddings = torch.cat(embeddings_list, dim=0).cuda()
    
    local_count = torch.tensor([local_embeddings.shape[0]], device='cuda')
    all_counts = [torch.zeros(1, device='cuda', dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(all_counts, local_count)
    all_counts = [int(c.item()) for c in all_counts]
    
    max_count = max(all_counts)
    if max_count == 0:
        return None
    
    embed_dim = local_embeddings.shape[1] if local_embeddings.shape[0] > 0 else 512
    
    if local_embeddings.shape[0] < max_count:
        pad = torch.zeros(max_count - local_embeddings.shape[0], embed_dim, device='cuda')
        local_embeddings = torch.cat([local_embeddings, pad], dim=0)
    
    gathered = [torch.zeros(max_count, embed_dim, device='cuda') for _ in range(world_size)]
    dist.all_gather(gathered, local_embeddings)
    
    if is_main_process():
        all_embeddings = []
        for i, count in enumerate(all_counts):
            if count > 0:
                all_embeddings.append(gathered[i][:count])
        if len(all_embeddings) == 0:
            return None
        return torch.cat(all_embeddings, dim=0).cpu()
    return None


def select_deterministic_quick_indices(keys, subset_size: int, seed: int):
    """
    Deterministic "random" subset selection based on sample keys.
    Same sample list + same seed + same subset_size => identical result.
    """
    n = len(keys)
    if n == 0:
        return []
    subset_size = min(max(1, int(subset_size)), n)
    scored = []
    for idx, key in enumerate(keys):
        token = f"{seed}:{key}".encode("utf-8")
        score = int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "big")
        scored.append((score, idx))
    scored.sort(key=lambda x: (x[0], x[1]))
    return sorted([idx for _, idx in scored[:subset_size]])


# ==================== Motion Utilities ====================

def renorm4t2m(motion, ori_mean, ori_std, eval_mean, eval_std):
    """Convert motion from training norm to eval norm."""
    motion = motion * ori_std + ori_mean
    motion = (motion - eval_mean) / eval_std
    return motion


def ensure_eval_motion_dim(motion: torch.Tensor, eval_dim: int = 263) -> torch.Tensor:
    """Unify eval dims: keep only the first 263 dims (ignore the trailing 6), zero-pad if shorter."""
    if motion.shape[-1] > eval_dim:
        return motion[..., :eval_dim]
    if motion.shape[-1] < eval_dim:
        pad = torch.zeros(*motion.shape[:-1], eval_dim - motion.shape[-1], device=motion.device, dtype=motion.dtype)
        return torch.cat([motion, pad], dim=-1)
    return motion


def get_metric_statistics(values, replication_times):
    """Compute the mean and 95% confidence interval of metrics (identical to MotionGPT3)."""
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


# ==================== Generation Input Preparation ====================

def prepare_t2m_input(
    captions, text_tokenizer, num_motion_tokens,
    bos_id, eos_id, bom_id, eom_id, pad_id, mot_pad_id,
    max_text_len, max_seq_len, device,
):
    """
    Prepare inputs for T2M generation.
    Sequence layout:
    - Conditional: [BOS] text [BOM][MOT_PAD*N][EOM][EOS]
    - Unconditional: [BOS][BOM][MOT_PAD*N][EOM][EOS]
    """
    batch_cond_tokens = []
    batch_cond_positions = []
    batch_uncond_tokens = []
    batch_uncond_positions = []
    
    for caption in captions:
        text_tokens = text_tokenizer(caption, add_special_tokens=False)['input_ids'][:max_text_len]
        
        # Conditional
        modality_positions = torch.tensor([len(text_tokens) + 2, num_motion_tokens]).unsqueeze(0)
        full_seq = [bos_id] + text_tokens + [bom_id] + [mot_pad_id] * num_motion_tokens + [eom_id] + [eos_id]
        pad_len = max_seq_len - len(full_seq)
        full_seq = full_seq + [pad_id] * pad_len
        
        batch_cond_tokens.append(torch.tensor(full_seq[:max_seq_len]))
        batch_cond_positions.append(modality_positions)
        
        # Unconditional
        modality_positions_null = torch.tensor([2, num_motion_tokens]).unsqueeze(0)
        null_seq = [bos_id] + [bom_id] + [mot_pad_id] * num_motion_tokens + [eom_id] + [eos_id]
        pad_len_null = max_seq_len - len(null_seq)
        null_seq = null_seq + [pad_id] * pad_len_null
        
        batch_uncond_tokens.append(torch.tensor(null_seq[:max_seq_len]))
        batch_uncond_positions.append(modality_positions_null)
    
    return (
        torch.stack(batch_cond_tokens).to(device),
        torch.stack(batch_uncond_tokens).to(device),
        torch.stack(batch_cond_positions).to(device),
        torch.stack(batch_uncond_positions).to(device)
    )


def prepare_m2t_input(
    motion_latents, text_tokenizer, num_motion_tokens,
    bos_id, eos_id, bom_id, eom_id, pad_id, mot_pad_id,
    max_seq_len, device,
):
    """
    Prepare inputs for M2T generation.
    Format: [BOS][BOM][MOTION_TOKENS][EOM] -> generate text -> [EOS]
    """
    bs = motion_latents.shape[1]
    batch_tokens = []
    batch_positions = []
    
    for i in range(bs):
        modality_positions = torch.tensor([2, num_motion_tokens]).unsqueeze(0)
        seq = [bos_id, bom_id] + [mot_pad_id] * num_motion_tokens + [eom_id]
        pad_len = max_seq_len - len(seq)
        seq = seq + [pad_id] * pad_len
        
        batch_tokens.append(torch.tensor(seq[:max_seq_len]))
        batch_positions.append(modality_positions)
    
    return (
        torch.stack(batch_tokens).to(device),
        torch.stack(batch_positions).to(device)
    )


def prepare_pred_input(
    source_motion, target_motion, text_tokenizer,
    num_source_tokens, num_target_tokens,
    bos_id, eos_id, bom_id, eom_id, pad_id, mot_pad_id,
    max_seq_len, device, add_time_embeds=True,
):
    """
    Prepare inputs for Motion Prediction generation.
    Sequence layout: [BOS] "Predict motion:" [BOM][SOURCE_MOT*N1][EOM] [BOM][TARGET_MOT*N2][EOM][EOS]
    """
    bs = source_motion.shape[0] if source_motion.dim() == 3 else 1
    
    batch_tokens = []
    batch_source_positions = []
    batch_target_positions = []
    
    prompt = "Predict motion:"
    prompt_tokens = text_tokenizer(prompt, add_special_tokens=False)['input_ids']
    
    for i in range(bs):
        source_offset = 1 + len(prompt_tokens) + 1  # BOS + prompt + BOM
        source_modality_positions = torch.tensor([source_offset, num_source_tokens]).unsqueeze(0)
        
        target_offset = source_offset + num_source_tokens + 1 + 1  # source + EOM + BOM
        target_modality_positions = torch.tensor([target_offset, num_target_tokens]).unsqueeze(0)
        
        seq = ([bos_id] + prompt_tokens +
               [bom_id] + [mot_pad_id] * num_source_tokens + [eom_id] +
               [bom_id] + [mot_pad_id] * num_target_tokens + [eom_id] + [eos_id])
        
        pad_len = max_seq_len - len(seq)
        if pad_len > 0:
            seq = seq + [pad_id] * pad_len
        
        batch_tokens.append(torch.tensor(seq[:max_seq_len]))
        batch_source_positions.append(source_modality_positions)
        batch_target_positions.append(target_modality_positions)
    
    return (
        torch.stack(batch_tokens).to(device),
        torch.stack(batch_source_positions).to(device),
        torch.stack(batch_target_positions).to(device)
    )


# ==================== Generation Functions ====================

@torch.no_grad()
def generate_t2m(
    model, motion_vae, sampler, text_tokens, modality_positions,
    latent_len, motion_latent_dim, motion_lengths, max_seq_len,
    guidance_scale, num_inference_steps, config, device, weight_type,
):
    """Generate T2M motion."""
    bs = text_tokens.shape[0] // 2 if guidance_scale > 0 else text_tokens.shape[0]
    
    z = torch.randn((latent_len, bs, motion_latent_dim)).to(weight_type).to(device)
    if guidance_scale > 0:
        z = torch.cat([z, z], dim=1)
    
    block_mask = omni_attn_mask_naive(
        text_tokens.size(0), text_tokens.size(1), modality_positions, device
    ).to(weight_type)
    
    model_kwargs = dict(
        text_tokens=text_tokens,
        attention_mask=block_mask,
        modality_positions=modality_positions,
        output_hidden_states=True,
        max_seq_len=max_seq_len,
        guidance_scale=guidance_scale
    )
    
    sample_fn = sampler.sample_ode(
        sampling_method=config.transport.sampling_method,
        num_steps=num_inference_steps,
        atol=config.transport.atol,
        rtol=config.transport.rtol,
        reverse=config.transport.reverse,
        time_shifting_factor=config.transport.time_shifting_factor
    )
    
    samples = sample_fn(z, model.t2m_generate, **model_kwargs)[-1]
    
    if guidance_scale > 0:
        samples = samples[:, :bs, :]
    
    with torch.autocast(device_type='cuda', enabled=False):
        decoded_motions = motion_vae.float().decode(samples.float(), motion_lengths)
    
    if isinstance(decoded_motions, list):
        max_len = max(m.shape[0] for m in decoded_motions)
        padded = []
        for m in decoded_motions:
            if m.shape[0] < max_len:
                m = torch.cat([m, torch.zeros(max_len - m.shape[0], m.shape[1], device=m.device)], dim=0)
            padded.append(m)
        decoded_motions = torch.stack(padded)
    
    return decoded_motions, motion_lengths


@torch.no_grad()
def generate_m2t(
    model, motion_vae, text_tokenizer, motion_latents,
    text_tokens, modality_positions, motion_lengths,
    max_new_tokens, eos_token,
    max_seq_len, device, weight_type,
    do_sample=False,
    temperature=1.0,
    top_k=None,
    debug=False,
):
    """Generate M2T text (greedy decoding by default, consistent with MotionGPT3)."""
    bs = motion_latents.shape[1]
    dtype = weight_type
    
    motion_embeds = model.motion_embedder.forward_fused(motion_latents.to(dtype))
    input_embeds = model.showo.model.embed_tokens(text_tokens)
    
    t = torch.ones(bs, device=device)
    time_embeds = model.time_embed(t, dtype)
    if hasattr(model, 'time_embed_proj'):
        time_embeds_proj = model.time_embed_proj(time_embeds)
    else:
        time_embeds_proj = time_embeds
    
    for i, modality_batch in enumerate(modality_positions):
        for j, (offset, length) in enumerate(modality_batch):
            if length == 0:
                continue
            idx = i * modality_positions.size(1) + j
            
            if model.config.add_time_embeds:
                input_embeds[i, offset] = time_embeds_proj[idx]
                input_embeds[i, offset + 1:offset + 1 + length - 1] = motion_embeds[idx, :max(length - 1, 0)]
            else:
                input_embeds[i, offset:offset + length] = motion_embeds[idx, :length]
    
    # Create modality_mask for Token-Level LoRA
    modality_mask = None
    if hasattr(model, 'lora_enabled') and model.lora_enabled:
        modality_mask = model._create_modality_mask(
            bs, input_embeds.shape[1], modality_positions, device
        )
    
    generated_texts = []
    for i in range(bs):
        offset, length = modality_positions[i][0].tolist()
        valid_len = offset + length + 1  # +1 for EOM
        
        input_embeds_valid = input_embeds[i:i+1, :valid_len, :]
        
        sample_modality_mask = None
        if modality_mask is not None:
            sample_modality_mask = modality_mask[i:i+1, :valid_len]
        
        valid_modality_positions = modality_positions[i:i+1]
        block_mask_valid = omni_attn_mask_naive(
            1, valid_len, valid_modality_positions, device
        ).to(dtype)
        
        result = model.m2t_generate(
            input_embeds=input_embeds_valid,
            attention_mask=block_mask_valid,
            modality_mask=sample_modality_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            eos_token=eos_token,
        )
        
        token_ids = [t.item() for t in result]
        text = text_tokenizer.decode(token_ids, skip_special_tokens=True)
        generated_texts.append(text)
        
        if debug and i == 0:
            print(f"    [M2T Debug] motion_latents shape: {motion_latents.shape}")
            print(f"    [M2T Debug] motion_embeds shape: {motion_embeds.shape}")
            print(f"    [M2T Debug] modality_positions: {modality_positions[i]}")
            print(f"    [M2T Debug] valid_len: {valid_len}")
            print(f"    [M2T Debug] modality_mask sum: {sample_modality_mask.sum().item() if sample_modality_mask is not None else 'None'}")
            print(f"    [M2T Debug] Generated {len(token_ids)} tokens: {token_ids[:20]}...")
            print(f"    [M2T Debug] Text: '{text[:200]}...'")
    
    return generated_texts


@torch.no_grad()
def generate_pred(
    model, motion_vae, sampler,
    source_motion, target_motion,
    source_lengths, target_lengths,
    text_tokenizer, config, device, weight_type,
    mean_t, std_t,
    max_seq_len, downsample_rate, motion_latent_dim,
    bos_id, eos_id, bom_id, eom_id, pad_id, mot_pad_id,
    predict_ratio=0.2,
    guidance_scale=1.0,
    num_inference_steps=50,
    debug=False,
):
    """
    Generate Motion Prediction results.
    
    Args:
        source_motion: [bs, source_len, nfeats] conditioning motion (first predict_ratio part)
        target_motion: [bs, target_len, nfeats] GT target motion (used to compute lengths)
        source_lengths: list of source motion lengths
        target_lengths: list of target motion lengths
    Returns:
        predicted_motions: [bs, target_len, nfeats] predicted target motion
        target_lengths: list of target lengths
    """
    bs = source_motion.shape[0]
    dtype = weight_type
    add_time_embeds = config.model.showo.add_time_embeds
    
    source_latent_lens = latent_lens_from_motion_lengths(source_lengths, downsample_rate)
    target_latent_lens = latent_lens_from_motion_lengths(target_lengths, downsample_rate)
    
    max_source_latent_len = max(source_latent_lens)
    max_target_latent_len = max(target_latent_lens)
    
    # Encode source motion to latents
    with torch.autocast(device_type='cuda', enabled=False):
        source_latents, _ = motion_vae.float().encode_motion(
            source_motion.float(),
            source_lengths
        )
    # source_latents: [latent_len, bs, dim]
    
    num_source_tokens = max_source_latent_len
    num_target_tokens = max_target_latent_len + 1 if add_time_embeds else max_target_latent_len
    
    text_tokens, source_positions, target_positions = prepare_pred_input(
        source_motion, target_motion, text_tokenizer,
        num_source_tokens, num_target_tokens,
        bos_id, eos_id, bom_id, eom_id, pad_id, mot_pad_id,
        max_seq_len, device, add_time_embeds
    )
    
    z = torch.randn((max_target_latent_len, bs, motion_latent_dim)).to(dtype).to(device)
    
    combined_positions = torch.cat([source_positions, target_positions], dim=1)
    block_mask = omni_attn_mask_naive(
        text_tokens.size(0), text_tokens.size(1), combined_positions, device
    ).to(dtype)
    
    model_kwargs = dict(
        text_tokens=text_tokens,
        attention_mask=block_mask,
        source_motion_latents=source_latents.to(dtype),
        source_modality_positions=source_positions,
        target_modality_positions=target_positions,
        source_motion_lengths=source_lengths,
        target_motion_lengths=target_lengths,
        max_seq_len=max_seq_len,
        guidance_scale=guidance_scale,
    )
    
    sample_fn = sampler.sample_ode(
        sampling_method=config.transport.sampling_method,
        num_steps=num_inference_steps,
        atol=config.transport.atol,
        rtol=config.transport.rtol,
        reverse=config.transport.reverse,
        time_shifting_factor=config.transport.time_shifting_factor
    )
    
    samples = sample_fn(z, model.pred_generate, **model_kwargs)[-1]
    
    with torch.autocast(device_type='cuda', enabled=False):
        decoded_motions = motion_vae.float().decode(samples.float(), target_lengths)
    
    if isinstance(decoded_motions, list):
        max_len = max(m.shape[0] for m in decoded_motions)
        padded = []
        for m in decoded_motions:
            if m.shape[0] < max_len:
                m = torch.cat([m, torch.zeros(max_len - m.shape[0], m.shape[1], device=m.device)], dim=0)
            padded.append(m)
        decoded_motions = torch.stack(padded)
    
    if debug:
        print(f"    [PRED Debug] source_motion shape: {source_motion.shape}")
        print(f"    [PRED Debug] source_latents shape: {source_latents.shape}")
        print(f"    [PRED Debug] target z shape: {z.shape}")
        print(f"    [PRED Debug] samples shape: {samples.shape}")
        print(f"    [PRED Debug] decoded_motions shape: {decoded_motions.shape}")
        print(f"    [PRED Debug] source_lengths: {source_lengths[:3]}")
        print(f"    [PRED Debug] target_lengths: {target_lengths[:3]}")
    
    return decoded_motions, target_lengths


# ==================== Main Test Function ====================

@torch.no_grad()
def main():
    # Initialize distributed
    rank, world_size, local_rank = setup_distributed()
    is_distributed = world_size > 1
    
    # Parse config
    config = get_config()
    seed = config.test.get('seed', 42)
    set_seed(seed + rank)
    run_timestamp = sync_run_timestamp(datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # Device and dtype
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    weight_type = torch.bfloat16 if config.model.weight_type == "bfloat16" else torch.float32
    
    # Get task list
    tasks = config.test.get('tasks', ['t2m', 'm2t', 'pred'])
    if isinstance(tasks, str):
        tasks = [tasks]
    
    # Quick test mode
    quick_mode = config.test.get('quick', False)
    num_quick_samples = int(config.test.get('num_quick_samples', 64))
    quick_subset_seed = int(config.test.get('quick_subset_seed', seed))
    # Per-task replication times: M2T defaults to 3 (needed for NLG statistics), others default to 1
    global_rep = config.test.get('replication_times', None)  # global override (highest priority)
    _task_defaults = {'t2m': 1, 'm2t': 3, 'pred': 1}  # hard-coded defaults
    default_rep_per_task = {}
    for _task, _def in _task_defaults.items():
        _task_cfg = config.test.get(_task, {})
        default_rep_per_task[_task] = _task_cfg.get('replication_times', _def) if _task_cfg else _def
    if global_rep is not None:
        # Global override: all tasks use the same value
        for k in default_rep_per_task:
            default_rep_per_task[k] = int(global_rep)
    if quick_mode:
        for k in default_rep_per_task:
            default_rep_per_task[k] = 1  # quick test runs every task only once
    
    if is_main_process():
        print("=" * 70)
        print("UniMotion Stage 2a/2b Unified Evaluation")
        print("Motion Features: 269-dim, Downsample Rate: 1, LoRA-A+B")
        print("=" * 70)
        print(f"Tasks: {list(tasks)}")
        print(f"Quick mode: {quick_mode}" + (f" ({num_quick_samples} samples)" if quick_mode else ""))
        if quick_mode:
            print(f"Quick subset seed: {quick_subset_seed}")
        print(f"Replication times per task: { {t: default_rep_per_task.get(t, 1) for t in tasks} }")
        print(f"Distributed: {is_distributed}, World size: {world_size}")
        print(f"Device: {device}, Weight type: {weight_type}")
    
    # Create output directory
    output_dir = config.test.get('output_dir', 'outputs/stage2_eval')
    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)
    
    if is_distributed:
        dist.barrier()
    
    # ==================== [1/5] Load Models ====================
    if is_main_process():
        print("\n[1/5] Loading models...")
    
    # Motion VAE (269-dim, downsample_rate determined by the checkpoint)
    motion_vae, motion_vae_config = load_motion_vae(
        config.model.motion_vae.pretrained_model_path,
        device=device, freeze=True
    )
    motion_vae = motion_vae.to(weight_type)
    
    if is_main_process():
        print(f"  Motion VAE loaded: nfeats={motion_vae_config.get('nfeats', 269)}, "
              f"downsample_rate={motion_vae_config.get('downsample_rate', 1)}")
    
    # Tokenizer
    text_tokenizer, showo_token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[config.model.showo.llm_model_path],
        add_motion_tokens=True
    )
    config.model.showo.llm_vocab_size = len(text_tokenizer)
    
    # Model
    model = Showo2WithMotion(**config.model.showo).to(device)
    
    model_path = config.test.get('model_path', None)
    
    # Resolve checkpoint path automatically
    if model_path and os.path.exists(model_path):
        if os.path.isdir(model_path):
            bin_files = [f for f in os.listdir(model_path) if f.endswith('.bin')]
            if not bin_files:
                unwrapped_path = os.path.join(model_path, 'unwrapped_model')
                if os.path.exists(unwrapped_path):
                    model_path = unwrapped_path

    save_vis_cfg = config.test.get('save_vis', {})
    save_vis_manager = SaveVisManager(
        enabled=bool(save_vis_cfg.get('enabled', False)),
        root_dir=save_vis_cfg.get('root_dir', DEFAULT_SAVE_VIS_ROOT),
        script_tag=os.path.splitext(os.path.basename(__file__))[0],
        tasks=tasks,
        seed=seed,
        quick=bool(quick_mode),
        batch_size=int(config.test.get('batch_size', 32)),
        ckpt_path=model_path,
        run_timestamp=run_timestamp,
        rank=rank,
        world_size=world_size,
        is_main_process=is_main_process(),
        save_all_replications=bool(save_vis_cfg.get('save_all_replications', False)),
    )
    
    # Apply the LoRA structure first (must happen before loading the checkpoint)
    lora_config = config.get('lora', None)
    lora_weights_path = None
    
    if lora_config is not None and lora_config.get('enabled', False):
        if is_main_process():
            print("  Applying Token-Level LoRA structure (A+B)...")
        
        lora_params_dict = {
            'rank_A': lora_config.get('rank_A', 16),
            'rank_B': lora_config.get('rank_B', 16),
            'alpha': lora_config.get('alpha', 16),
            'dropout': 0.0,
            'enable_A': lora_config.get('enable_A', True),   # Stage 2a/2b: A enabled
            'enable_B': lora_config.get('enable_B', True),   # Stage 2a/2b: B enabled
            'target_modules': lora_config.get('target_modules', ['q_proj', 'k_proj', 'v_proj', 'o_proj']),
        }
        model.apply_lora(lora_params_dict)
        
        if is_main_process():
            lora_stats = count_lora_parameters(model)
            print(f"    LoRA-A (Text): {lora_params_dict['enable_A']}")
            print(f"    LoRA-B (Motion): {lora_params_dict['enable_B']}")
            print(f"    Total LoRA params: {lora_stats['total_lora_params']:,}")
        
        if model_path:
            original_model_path = config.test.get('model_path', None)
            if original_model_path:
                if 'unwrapped_model' in model_path:
                    lora_weights_path = os.path.join(os.path.dirname(model_path), 'lora_weights.pt')
                else:
                    lora_weights_path = os.path.join(original_model_path, 'lora_weights.pt')
    
    # Load checkpoint
    if model_path and os.path.exists(model_path):
        if is_main_process():
            print(f"  Loading checkpoint from {model_path}")
        state_dict = load_state_dict(model_path)
        model_state_dict = model.state_dict()
        
        # Handle vocabulary size mismatch
        embed_key = 'showo.model.embed_tokens.weight'
        lm_head_key = 'showo.lm_head.weight'
        
        if embed_key in state_dict and embed_key in model_state_dict:
            old_embed = state_dict[embed_key]
            new_embed = model_state_dict[embed_key]
            if old_embed.shape[0] != new_embed.shape[0]:
                new_embed[:old_embed.shape[0], :] = old_embed
                state_dict[embed_key] = new_embed
        
        if lm_head_key in state_dict and lm_head_key in model_state_dict:
            old_lm_head = state_dict[lm_head_key]
            new_lm_head = model_state_dict[lm_head_key]
            if old_lm_head.shape[0] != new_lm_head.shape[0]:
                new_lm_head[:old_lm_head.shape[0], :] = old_lm_head
                state_dict[lm_head_key] = new_lm_head
        
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
        model.load_state_dict(filtered_state_dict, strict=False)
        
        if is_main_process():
            base_linear_keys = [k for k in filtered_state_dict.keys() if 'base_linear' in k]
            print(f"    Loaded {len(filtered_state_dict)} params (base_linear: {len(base_linear_keys)})")
    
    # Load LoRA weights
    if lora_weights_path and os.path.exists(lora_weights_path):
        if is_main_process():
            print(f"  Loading LoRA weights from {lora_weights_path}")
        model.load_lora(lora_weights_path)
    
    model.to(weight_type)
    model.eval()
    
    # Transport sampler
    # Consistent with training: when add_time_embeds=True, seq_len needs +1
    transport_seq_len = config.dataset.preprocessing.max_motion_tokens
    if config.model.showo.add_time_embeds:
        transport_seq_len += 1
    transport = create_transport(
        path_type=config.transport.path_type,
        prediction=config.transport.prediction,
        loss_weight=config.transport.loss_weight,
        train_eps=config.transport.train_eps,
        sample_eps=config.transport.sample_eps,
        snr_type=config.transport.snr_type,
        do_shift=config.transport.do_shift,
        seq_len=transport_seq_len,
    )
    sampler = Sampler(transport)
    
    # ==================== [2/5] Load Evaluators ====================
    if is_main_process():
        print("\n[2/5] Loading evaluators...")
    
    t2m_evaluator_path = config.test.get('t2m_evaluator_path', 'deps/t2m/t2m')
    glove_dir = config.test.get('glove_dir', 'deps/glove/glove')
    # The evaluator always uses unit_length=4 (HumanML3D standard, 263 dims)
    eval_unit_length = 4
    
    evaluators = {}
    
    if 't2m' in tasks:
        evaluators['t2m'] = T2MEvaluator(
            t2m_evaluator_path=t2m_evaluator_path,
            glove_dir=glove_dir,
            dataname='t2m',
            device=device,
            unit_length=eval_unit_length,
        )
        if is_main_process():
            print("  T2M evaluator loaded")
    
    if 'm2t' in tasks:
        evaluators['m2t'] = M2TEvaluator(
            t2m_evaluator_path=t2m_evaluator_path,
            glove_dir=glove_dir,
            dataname='t2m',
            device=device,
            unit_length=eval_unit_length,
        )
        if is_main_process():
            print("  M2T evaluator loaded")
    
    if 'pred' in tasks:
        predict_ratio = config.test.pred.get('predict_ratio', 0.2)
        evaluators['pred'] = PredEvaluator(
            njoints=22,
            predict_ratio=predict_ratio,
            task='pred',
            device=device,
        )
        evaluators['pred_t2m'] = T2MEvaluator(
            t2m_evaluator_path=t2m_evaluator_path,
            glove_dir=glove_dir,
            dataname='t2m',
            device=device,
            unit_length=eval_unit_length,
        )
        if is_main_process():
            print("  Pred evaluator loaded")
            print("  Pred FID/DIV evaluator loaded")
    
    # ==================== [3/5] Load Datasets ====================
    if is_main_process():
        print("\n[3/5] Loading evaluation dataset...")
    
    # Always use UNI_MOTION_269 (269-dim data; the dataset truncates to 263 dims for eval normalization)
    data_root = config.dataset.motion.data_root
    
    max_motion_length = config.test.get('max_motion_length', 200)
    min_motion_length = config.test.get('min_motion_length', 20)
    batch_size = config.test.get('batch_size', 32)
    
    mean_eval_path = config.test.get('mean_eval_path', None)
    std_eval_path = config.test.get('std_eval_path', None)
    eval_max_text_len = config.dataset.preprocessing.get('eval_max_text_len', 20)
    
    if is_main_process():
        print(f"  Data root: {data_root}")
    
    w_vectorizer = None
    if os.path.exists(glove_dir):
        w_vectorizer = WordVectorizer(glove_dir)
    
    # Eval dataset: loads 269-dim data, internally truncates to 263 dims and normalizes with eval mean/std
    eval_dataset = MotionEvalDataset(
        data_root=data_root,
        split='test',
        w_vectorizer=w_vectorizer,
        max_motion_length=max_motion_length,
        min_motion_length=min_motion_length,
        mean_eval_path=mean_eval_path,
        std_eval_path=std_eval_path,
        max_text_len=eval_max_text_len,
        unit_length=eval_unit_length,
        tiny=config.test.get('tiny', False),
    )
    
    # Quick test: randomly sampled subset (fixed seed, reproducible)
    if quick_mode:
        full_size = len(eval_dataset.name_list) - eval_dataset.pointer
        subset_size = min(num_quick_samples, full_size)
        original_names = eval_dataset.name_list[eval_dataset.pointer:]
        indices = select_deterministic_quick_indices(original_names, subset_size, quick_subset_seed)
        eval_dataset.name_list = [original_names[i] for i in indices]
        eval_dataset.pointer = 0
        if is_main_process():
            print(f"  Quick mode: using {subset_size}/{full_size} samples (deterministic random subset)")
            print(f"  Quick subset seed: {quick_subset_seed}")
    
    if is_distributed:
        data_sampler = DistributedSampler(eval_dataset, shuffle=False)
    else:
        data_sampler = None
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=data_sampler,
        num_workers=config.dataset.params.get('num_workers', 4),
        collate_fn=eval_dataset.collate_fn,
        drop_last=False,
        pin_memory=True,
    )
    
    if is_main_process():
        print(f"  Dataset size: {len(eval_dataset)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Total batches: {len(eval_dataloader)}")
    
    # Load normalization parameters
    # Training mean/std: 269-dim (loaded from data_root)
    mean_train = np.load(os.path.join(data_root, 'Mean.npy'))
    std_train = np.load(os.path.join(data_root, 'Std.npy'))
    
    if is_main_process():
        print(f"  Train mean/std shape: {mean_train.shape}")
    
    # Evaluation mean/std: 263-dim (loaded from Comp_v6_KLD01, used by the T2M evaluator)
    if mean_eval_path and os.path.exists(mean_eval_path):
        mean_eval = np.load(mean_eval_path)
    else:
        mean_eval = mean_train[:263]  # fallback: truncate to 263 dims
    
    if std_eval_path and os.path.exists(std_eval_path):
        std_eval = np.load(std_eval_path)
    else:
        std_eval = std_train[:263]  # fallback: truncate to 263 dims
    
    if is_main_process():
        print(f"  Eval mean/std shape: {mean_eval.shape}")
    
    mean_train_t = torch.from_numpy(mean_train).float().to(device)
    std_train_t = torch.from_numpy(std_train).float().to(device)
    mean_eval_t = torch.from_numpy(mean_eval).float().to(device)
    std_eval_t = torch.from_numpy(std_eval).float().to(device)
    
    # Precompute: 263-dim truncated version of the 269-dim training mean/std (for T2M renorm)
    mean_train_263 = mean_train_t[:263] if mean_train_t.shape[0] > 263 else mean_train_t
    std_train_263 = std_train_t[:263] if std_train_t.shape[0] > 263 else std_train_t
    
    # Token IDs
    bos_id = showo_token_ids['bos_id']
    eos_id = showo_token_ids['eos_id']
    bom_id = showo_token_ids['bom_id']
    eom_id = showo_token_ids['eom_id']
    pad_id = text_tokenizer.pad_token_id
    mot_pad_id = showo_token_ids['mot_pad_id']
    
    max_text_len = config.dataset.preprocessing.max_text_len
    max_seq_len = config.dataset.preprocessing.max_seq_length
    downsample_rate = config.dataset.motion.downsample_rate
    motion_latent_dim = config.dataset.motion.motion_latent_dim
    
    # ==================== [4/5] Run Evaluation ====================
    if is_main_process():
        print(f"\n[4/5] Running evaluation...")
    
    all_results = {}  # task -> {metric_name: [values_per_rep]}
    m2t_debug_samples = []
    
    for task in tasks:
        replication_times = default_rep_per_task.get(task, 1)
        
        if is_main_process():
            print(f"\n{'='*60}")
            print(f"  Evaluating: {task.upper()} (replication_times={replication_times})")
            print(f"{'='*60}")
        
        # Get task-specific parameters
        if task == 't2m':
            guidance_scale = config.test.t2m.get('guidance_scale', 5.0)
            num_inference_steps = config.test.t2m.get('num_inference_steps', 50)
            if is_main_process():
                print(f"  Guidance scale: {guidance_scale}, Steps: {num_inference_steps}")
        elif task == 'm2t':
            max_new_tokens = config.test.m2t.get('max_new_tokens', 100)
            do_sample = config.test.m2t.get('do_sample', False)
            temperature = config.test.m2t.get('temperature', 1.0)
            top_k = config.test.m2t.get('top_k', 50)
            if is_main_process():
                print(f"  Max new tokens: {max_new_tokens}, Greedy: {not do_sample}")
        elif task == 'pred':
            predict_ratio = config.test.pred.get('predict_ratio', 0.2)
            pred_guidance_scale = config.test.pred.get('guidance_scale', 1.0)
            pred_num_steps = config.test.pred.get('num_inference_steps', 50)
            if is_main_process():
                print(f"  Predict ratio: {predict_ratio}, Steps: {pred_num_steps}")
        
        task_all_metrics = {}
        
        for rep_i in range(replication_times):
            if is_main_process():
                print(f"\n  --- Replication {rep_i + 1}/{replication_times} ---")
            
            # Reset evaluator
            if task in evaluators:
                if hasattr(evaluators[task], 'reset'):
                    evaluators[task].reset()
            if task == 'pred' and 'pred_t2m' in evaluators and hasattr(evaluators['pred_t2m'], 'reset'):
                evaluators['pred_t2m'].reset()
            
            # Use a different seed for each replication
            set_seed(seed + rank + rep_i * 1000)
            
            total_samples = 0
            
            pbar = tqdm(
                eval_dataloader,
                desc=f"{task} Rep {rep_i+1}/{replication_times}",
                disable=not is_main_process()
            )
            save_this_rep = save_vis_manager.should_save_replication(rep_i)
            
            for batch_idx, batch in enumerate(pbar):
                gt_motions = ensure_eval_motion_dim(batch['motions'].to(device))     # [bs, max_len, 263] eval-normalized
                gt_lengths = batch['m_lengths']
                captions = batch['captions']
                fnames = batch.get('fnames', [])
                word_embs = batch['word_embs'].to(device)
                pos_ohot = batch['pos_ohot'].to(device)
                sent_lens = batch['sent_lens'].to(device)
                all_captions = batch['all_captions']
                
                bs = gt_motions.shape[0]
                use_gt_length = config.test.get('use_gt_length', True)
                
                if use_gt_length:
                    motion_lengths = [max(1, int(l)) for l in gt_lengths]
                else:
                    fixed_len = max(1, int(max_motion_length))
                    motion_lengths = [fixed_len] * bs
                
                # Align to downsample_rate (no-op when =1)
                motion_length = max(motion_lengths)
                latent_len = latent_len_from_motion_length(motion_length, downsample_rate)
                num_motion_tokens = latent_len + 1 if config.model.showo.add_time_embeds else latent_len
                
                # ============ T2M: Text -> Motion ============
                if task == 't2m':
                    cond_tokens, uncond_tokens, cond_positions, uncond_positions = prepare_t2m_input(
                        captions, text_tokenizer, num_motion_tokens,
                        bos_id, eos_id, bom_id, eom_id, pad_id, mot_pad_id,
                        max_text_len, max_seq_len, device,
                    )
                    
                    if guidance_scale > 0:
                        text_tokens = torch.cat([cond_tokens, uncond_tokens], dim=0)
                        modality_positions = torch.cat([cond_positions, uncond_positions], dim=0)
                    else:
                        text_tokens = cond_tokens
                        modality_positions = cond_positions
                    
                    gen_motions, gen_lengths = generate_t2m(
                        model, motion_vae, sampler, text_tokens, modality_positions,
                        latent_len, motion_latent_dim, motion_lengths, max_seq_len,
                        guidance_scale, num_inference_steps, config, device, weight_type,
                    )
                    
                    gen_motions = ensure_eval_motion_dim(gen_motions)
                    
                    # Align lengths
                    gt_max_len = gt_motions.shape[1]
                    gen_max_len = gen_motions.shape[1]
                    if gen_max_len < gt_max_len:
                        pad = torch.zeros(bs, gt_max_len - gen_max_len, 263, device=device, dtype=gen_motions.dtype)
                        gen_motions = torch.cat([gen_motions, pad], dim=1)
                    elif gen_max_len > gt_max_len:
                        gen_motions = gen_motions[:, :gt_max_len, :]
                    
                    if save_this_rep:
                        for i in range(bs):
                            sample_id = fnames[i] if i < len(fnames) else f"sample_{batch_idx:05d}_{i:03d}"
                            motion_len_i = int(gen_lengths[i]) if i < len(gen_lengths) else int(gen_motions.shape[1])
                            motion_len_i = max(1, min(motion_len_i, int(gen_motions.shape[1])))
                            save_vis_manager.save_motion_sample(
                                task='t2m',
                                replication=rep_i,
                                batch_idx=batch_idx,
                                in_batch_idx=i,
                                sample_id=sample_id,
                                motion=gen_motions[i, :motion_len_i],
                                keep_dim=263,
                                extra={
                                    'fname': sample_id,
                                    'caption': captions[i] if i < len(captions) else "",
                                    'motion_length': motion_len_i,
                                },
                            )
                    
                    # Renorm: training norm (263) -> eval norm
                    gen_motions = renorm4t2m(gen_motions, mean_train_263, std_train_263, mean_eval_t, std_eval_t)
                    eval_gen_lengths = [min(int(gen_lengths[i]), int(gt_lengths[i])) for i in range(bs)]
                    
                    evaluators['t2m'].update(
                        gt_motion=gt_motions.float(),
                        gen_motion=gen_motions.float(),
                        gt_lengths=gt_lengths,
                        gen_lengths=eval_gen_lengths,
                        word_embs=word_embs.float(),
                        pos_ohot=pos_ohot.float(),
                        cap_lens=sent_lens,
                    )
                
                # ============ M2T: Motion -> Text ============
                elif task == 'm2t':
                    # Convert GT motion from eval norm to training norm before encoding
                    gt_motions_denorm = gt_motions * std_eval_t + mean_eval_t        # denormalize to raw space
                    # Raw space is 263-dim but the training mean/std are 269-dim; pad to 269 dims before normalizing
                    if mean_train_t.shape[0] > 263:
                        # Pad extra dims with the training mean (instead of 0) so they become 0 (neutral) after normalization
                        pad_dim = mean_train_t.shape[0] - 263
                        pad_values = mean_train_t[263:].unsqueeze(0).unsqueeze(0).expand(
                            gt_motions_denorm.shape[0], gt_motions_denorm.shape[1], pad_dim)
                        gt_motions_pad = torch.cat([gt_motions_denorm, pad_values], dim=-1)
                        gt_motions_train_norm = (gt_motions_pad - mean_train_t) / std_train_t
                    else:
                        gt_motions_train_norm = (gt_motions_denorm - mean_train_t) / std_train_t
                    
                    with torch.autocast(device_type='cuda', enabled=False):
                        motion_latents, _ = motion_vae.float().encode_motion(
                            gt_motions_train_norm.float(),
                            gt_lengths
                        )
                    
                    # For M2T, num_motion_tokens is based on the latent length
                    m2t_latent_len = motion_latents.shape[0]
                    m2t_num_tokens = m2t_latent_len + 1 if config.model.showo.add_time_embeds else m2t_latent_len
                    
                    text_tokens_m2t, modality_positions_m2t = prepare_m2t_input(
                        motion_latents, text_tokenizer, m2t_num_tokens,
                        bos_id, eos_id, bom_id, eom_id, pad_id, mot_pad_id,
                        max_seq_len, device,
                    )
                    
                    debug_m2t = is_main_process() and batch_idx == 0 and rep_i == 0
                    
                    gen_texts = generate_m2t(
                        model, motion_vae, text_tokenizer, motion_latents.to(weight_type),
                        text_tokens_m2t, modality_positions_m2t, gt_lengths,
                        max_new_tokens, eos_id,
                        max_seq_len, device, weight_type,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_k=top_k,
                        debug=debug_m2t,
                    )
                    
                    # Save debug samples
                    if is_main_process() and batch_idx < 5 and rep_i == 0:
                        for i, (gen, gt_list) in enumerate(zip(gen_texts, all_captions)):
                            m2t_debug_samples.append({
                                'batch_idx': batch_idx,
                                'sample_idx': i,
                                'generated': gen,
                                'gt_captions': gt_list,
                            })
                            if batch_idx < 2 and i < 2:
                                print(f"\n  [M2T] Batch {batch_idx}, Sample {i}:")
                                print(f"    Gen: '{gen[:150]}' (len={len(gen)})")
                                print(f"    GT:  '{gt_list[0][:100] if gt_list else 'N/A'}'")
                    
                    evaluators['m2t'].update(
                        gt_motion=gt_motions.float(),
                        pred_texts=gen_texts,
                        gt_texts=all_captions,
                        gt_lengths=gt_lengths,
                        word_embs=word_embs.float(),
                        pos_ohot=pos_ohot.float(),
                        cap_lens=sent_lens,
                    )
                    if save_this_rep:
                        for i, gen in enumerate(gen_texts):
                            sample_id = fnames[i] if i < len(fnames) else f"sample_{batch_idx:05d}_{i:03d}"
                            save_vis_manager.save_text_sample(
                                task='m2t',
                                replication=rep_i,
                                batch_idx=batch_idx,
                                in_batch_idx=i,
                                sample_id=sample_id,
                                text=gen,
                                extra={
                                    'fname': sample_id,
                                    'gt_texts': all_captions[i] if i < len(all_captions) else [],
                                },
                            )
                
                # ============ Pred: Motion Prediction ============
                elif task == 'pred':
                    # Convert GT motion from eval norm to training norm
                    gt_motions_denorm = gt_motions * std_eval_t + mean_eval_t
                    if mean_train_t.shape[0] > 263:
                        # Pad extra dims with the training mean (instead of 0) so they become 0 (neutral) after normalization
                        pad_dim = mean_train_t.shape[0] - 263
                        pad_values = mean_train_t[263:].unsqueeze(0).unsqueeze(0).expand(
                            gt_motions_denorm.shape[0], gt_motions_denorm.shape[1], pad_dim)
                        gt_motions_pad = torch.cat([gt_motions_denorm, pad_values], dim=-1)
                        gt_motions_train_norm = (gt_motions_pad - mean_train_t) / std_train_t
                    else:
                        gt_motions_train_norm = (gt_motions_denorm - mean_train_t) / std_train_t
                    
                    nfeats = gt_motions_train_norm.shape[-1]
                    
                    # Split motion: source (first predict_ratio) + target (last 1-predict_ratio)
                    source_motions = []
                    target_motions = []
                    source_lengths_list = []
                    target_lengths_list = []
                    full_lengths = []
                    
                    for i, length in enumerate(gt_lengths):
                        length = int(length)
                        aligned_len = int(length)
                        if aligned_len < 2:
                            continue

                        split_point = int(aligned_len * predict_ratio)
                        if split_point < 1:
                            split_point = 1
                        if aligned_len - split_point < 1:
                            split_point = aligned_len - 1
                        
                        source_len = split_point
                        target_len = aligned_len - split_point
                        
                        source_motions.append(gt_motions_train_norm[i, :source_len, :])
                        target_motions.append(gt_motions_train_norm[i, split_point:aligned_len, :])
                        source_lengths_list.append(source_len)
                        target_lengths_list.append(target_len)
                        full_lengths.append(source_len + target_len)
                    if len(source_motions) == 0:
                        continue
                    
                    # Pad to same length
                    max_source_len = max(source_lengths_list)
                    max_target_len = max(target_lengths_list)
                    
                    source_padded = []
                    target_padded = []
                    for s, t in zip(source_motions, target_motions):
                        if s.shape[0] < max_source_len:
                            s = torch.cat([s, torch.zeros(max_source_len - s.shape[0], nfeats, device=device)], dim=0)
                        if t.shape[0] < max_target_len:
                            t = torch.cat([t, torch.zeros(max_target_len - t.shape[0], nfeats, device=device)], dim=0)
                        source_padded.append(s)
                        target_padded.append(t)
                    
                    source_batch = torch.stack(source_padded)
                    target_batch = torch.stack(target_padded)
                    
                    # Generate predictions
                    predicted_motions, _ = generate_pred(
                        model, motion_vae, sampler,
                        source_batch, target_batch,
                        source_lengths_list, target_lengths_list,
                        text_tokenizer, config, device, weight_type,
                        mean_train_t, std_train_t,
                        max_seq_len, downsample_rate, motion_latent_dim,
                        bos_id, eos_id, bom_id, eom_id, pad_id, mot_pad_id,
                        predict_ratio=predict_ratio,
                        guidance_scale=pred_guidance_scale,
                        num_inference_steps=pred_num_steps,
                        debug=(batch_idx == 0 and rep_i == 0 and is_main_process()),
                    )
                    
                    # Concatenate the full motion and convert to joints (aligned with MotionGPT3)
                    max_full_len = max(full_lengths)
                    feat_dim = 263  # evaluation uses 263 dims
                    
                    full_gt_motions = []
                    full_pred_motions = []
                    for i in range(bs):
                        s_len = source_lengths_list[i]
                        t_len = target_lengths_list[i]
                        
                        # Full GT motion
                        gt_full = gt_motions_train_norm[i, :full_lengths[i], :feat_dim]
                        
                        # Full pred motion = source (GT) + predicted_target
                        pred_target = predicted_motions[i, :t_len, :feat_dim]
                        source_part = source_motions[i][:s_len, :feat_dim]
                        pred_full = torch.cat([source_part, pred_target], dim=0)
                        
                        # Pad
                        if gt_full.shape[0] < max_full_len:
                            gt_full = torch.cat([gt_full, torch.zeros(max_full_len - gt_full.shape[0], feat_dim, device=device)], dim=0)
                        if pred_full.shape[0] < max_full_len:
                            pred_full = torch.cat([pred_full, torch.zeros(max_full_len - pred_full.shape[0], feat_dim, device=device)], dim=0)
                        
                        full_gt_motions.append(gt_full)
                        full_pred_motions.append(pred_full)
                    
                    full_gt_batch = torch.stack(full_gt_motions)
                    full_pred_batch = torch.stack(full_pred_motions)
                    
                    # Denormalize + recover_from_ric -> joints
                    gt_joints = feats2joints_humanml3d(full_gt_batch, mean=mean_train_263, std=std_train_263)
                    pred_joints = feats2joints_humanml3d(full_pred_batch, mean=mean_train_263, std=std_train_263)
                    
                    evaluators['pred'].update(
                        joints_rst=pred_joints.float(),
                        joints_ref=gt_joints.float(),
                        lengths=full_lengths,
                    )
                    # MotionGPT pred alignment: additionally compute TM2TMetrics FID / Diversity
                    full_gt_eval = renorm4t2m(full_gt_batch, mean_train_263, std_train_263, mean_eval_t, std_eval_t)
                    full_pred_eval = renorm4t2m(full_pred_batch, mean_train_263, std_train_263, mean_eval_t, std_eval_t)
                    evaluators['pred_t2m'].update(
                        gt_motion=full_gt_eval.float(),
                        gen_motion=full_pred_eval.float(),
                        gt_lengths=full_lengths,
                        gen_lengths=full_lengths,
                    )
                    if save_this_rep:
                        for i in range(bs):
                            sample_id = fnames[i] if i < len(fnames) else f"sample_{batch_idx:05d}_{i:03d}"
                            motion_len_i = int(full_lengths[i]) if i < len(full_lengths) else int(full_pred_batch.shape[1])
                            motion_len_i = max(1, min(motion_len_i, int(full_pred_batch.shape[1])))
                            save_vis_manager.save_motion_sample(
                                task='pred',
                                replication=rep_i,
                                batch_idx=batch_idx,
                                in_batch_idx=i,
                                sample_id=sample_id,
                                motion=full_pred_batch[i, :motion_len_i],
                                keep_dim=263,
                                extra={
                                    'fname': sample_id,
                                    'motion_length': motion_len_i,
                                    'source_length': int(source_lengths_list[i]) if i < len(source_lengths_list) else None,
                                    'target_length': int(target_lengths_list[i]) if i < len(target_lengths_list) else None,
                                },
                            )
                    
                    if batch_idx == 0 and rep_i == 0 and is_main_process():
                        print(f"\n  [PRED] Batch 0 (aligned with MotionGPT3):")
                        print(f"    Source lengths: {source_lengths_list[:3]}")
                        print(f"    Target lengths: {target_lengths_list[:3]}")
                        print(f"    Full lengths: {full_lengths[:3]}")
                        print(f"    GT joints shape: {gt_joints.shape}")
                        print(f"    Pred joints shape: {pred_joints.shape}")
                
                total_samples += bs
            
            # Aggregate distributed results
            if is_distributed:
                dist.barrier()
                
                total_samples_tensor = torch.tensor([total_samples], device=device)
                dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
                total_samples = int(total_samples_tensor.item())
                
                # Aggregate embeddings (needed for T2M/M2T/Pred-FID)
                if task in ['t2m', 'm2t']:
                    evaluator = evaluators[task]
                    if hasattr(evaluator, 'text_embeddings'):
                        all_text_embs = gather_embeddings(evaluator.text_embeddings, world_size)
                        
                        if task == 't2m':
                            all_gen_embs = gather_embeddings(evaluator.gen_motion_embeddings, world_size)
                            all_gt_embs = gather_embeddings(evaluator.gt_motion_embeddings, world_size)
                        else:
                            all_gen_embs = gather_embeddings(evaluator.predtext_embeddings, world_size)
                            all_gt_embs = gather_embeddings(evaluator.gtmotion_embeddings, world_size)
                        
                        if is_main_process():
                            evaluator.text_embeddings = [all_text_embs] if all_text_embs is not None else []
                            if task == 't2m':
                                evaluator.gen_motion_embeddings = [all_gen_embs] if all_gen_embs is not None else []
                                evaluator.gt_motion_embeddings = [all_gt_embs] if all_gt_embs is not None else []
                            else:
                                evaluator.predtext_embeddings = [all_gen_embs] if all_gen_embs is not None else []
                                evaluator.gtmotion_embeddings = [all_gt_embs] if all_gt_embs is not None else []
                elif task == 'pred' and 'pred_t2m' in evaluators:
                    pred_t2m_eval = evaluators['pred_t2m']
                    all_gen_embs = gather_embeddings(pred_t2m_eval.gen_motion_embeddings, world_size)
                    all_gt_embs = gather_embeddings(pred_t2m_eval.gt_motion_embeddings, world_size)
                    if is_main_process():
                        pred_t2m_eval.gen_motion_embeddings = [all_gen_embs] if all_gen_embs is not None else []
                        pred_t2m_eval.gt_motion_embeddings = [all_gt_embs] if all_gt_embs is not None else []
            
            # Compute metrics for this replication
            if is_main_process():
                rep_metrics = evaluators[task].compute()
                if task == 'pred' and 'pred_t2m' in evaluators:
                    pred_t2m_metrics = evaluators['pred_t2m'].compute()
                    rep_metrics['FID'] = float(pred_t2m_metrics.get('FID', 0.0))
                    rep_metrics['DIV'] = float(pred_t2m_metrics.get('Diversity', 0.0))
                    rep_metrics['GT_DIV'] = float(pred_t2m_metrics.get('gt_Diversity', 0.0))
                
                # Print results of this replication
                key_metrics = []
                for key, value in sorted(rep_metrics.items()):
                    if isinstance(value, float):
                        key_metrics.append(f"{key}={value:.4f}")
                if key_metrics:
                    print(f"  Rep {rep_i+1}: {', '.join(key_metrics[:6])}")
                
                # Collect into all_metrics
                for key, value in rep_metrics.items():
                    if key not in task_all_metrics:
                        task_all_metrics[key] = [value]
                    else:
                        task_all_metrics[key].append(value)
            
            if is_distributed:
                dist.barrier()
        
        # Store results for this task
        all_results[task] = {
            'total_samples': total_samples,
            'replication_times': replication_times,
            'per_rep_metrics': task_all_metrics,
        }
    
    # ==================== [5/5] Compute Statistics and Save Results ====================
    if not is_main_process():
        if save_vis_manager.enabled and is_distributed:
            dist.barrier()
            dist.barrier()
        cleanup_distributed()
        return
    
    print(f"\n[5/5] Computing final statistics...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_results = {}
    
    # Print final results
    print("\n" + "=" * 70)
    mode_str = "QUICK TEST" if quick_mode else "FULL EVALUATION"
    print(f"STAGE 2 {mode_str} RESULTS (mean +/- 95% CI)")
    print("=" * 70)
    print(f"Model: {config.test.get('model_path', None)}")
    print(f"Replication times per task: { {t: default_rep_per_task.get(t, 1) for t in tasks} }")
    
    result_lines = []
    
    for task in tasks:
        if task not in all_results:
            continue
        
        task_result = all_results[task]
        task_metrics = task_result['per_rep_metrics']
        replication_times = task_result['replication_times']
        
        print(f"\n--- {task.upper()} ({task_result['total_samples']} samples, rep={replication_times}) ---")
        
        task_final = {}
        
        # Define metric display order per task type
        if task == 't2m':
            metric_order = [
                'R_Precision_Top_1', 'R_Precision_Top_2', 'R_Precision_Top_3',
                'new_fid', 'new_gt_fid',
                'FID', 'Matching_Score', 'Diversity',
                'gt_R_Precision_Top_1', 'gt_R_Precision_Top_2', 'gt_R_Precision_Top_3',
                'gt_Matching_Score', 'gt_Diversity',
            ]
            up_metrics = {'R_Precision_Top_1', 'R_Precision_Top_2', 'R_Precision_Top_3',
                         'Diversity', 'gt_Diversity',
                         'gt_R_Precision_Top_1', 'gt_R_Precision_Top_2', 'gt_R_Precision_Top_3'}
        elif task == 'm2t':
            metric_order = [
                'R_precision_top_1', 'R_precision_top_2', 'R_precision_top_3',
                'Matching_score', 'gt_Matching_score',
                'ROUGE_L', 'CIDEr', 'Bleu_1', 'Bleu_4', 'Bert_F1',
            ]
            up_metrics = {'R_precision_top_1', 'R_precision_top_2', 'R_precision_top_3',
                         'ROUGE_L', 'CIDEr', 'Bleu_1', 'Bleu_4', 'Bert_F1'}
        elif task == 'pred':
            metric_order = ['FID', 'DIV', 'GT_DIV', 'ADE', 'FDE', 'APD']
            up_metrics = {'DIV', 'GT_DIV'}  # higher is better for DIV/GT_DIV, lower is better for the rest
        else:
            metric_order = sorted(task_metrics.keys())
            up_metrics = set()
        
        # Compute mean ± CI
        for key in metric_order:
            if key not in task_metrics:
                continue
            values = np.array(task_metrics[key])
            if replication_times > 1:
                mean_val, conf_val = get_metric_statistics(values, replication_times)
            else:
                mean_val = float(values[0])
                conf_val = 0.0
            
            task_final[f"{key}/mean"] = float(mean_val)
            task_final[f"{key}/conf_interval"] = float(conf_val)
            
            arrow = "↑" if key in up_metrics else "↓"
            if replication_times > 1:
                line = f"  {key} {arrow}: {mean_val:.4f} +/- {conf_val:.4f}"
            else:
                line = f"  {key} {arrow}: {mean_val:.4f}"
            print(line)
            result_lines.append(f"[{task.upper()}] {line.strip()}")
        
        # Handle metrics not listed in the display order
        for key in sorted(task_metrics.keys()):
            if key not in metric_order and f"{key}/mean" not in task_final:
                values = np.array(task_metrics[key])
                if replication_times > 1:
                    mean_val, conf_val = get_metric_statistics(values, replication_times)
                else:
                    mean_val = float(values[0])
                    conf_val = 0.0
                task_final[f"{key}/mean"] = float(mean_val)
                task_final[f"{key}/conf_interval"] = float(conf_val)
                line = f"  {key}: {mean_val:.4f}" + (f" +/- {conf_val:.4f}" if replication_times > 1 else "")
                print(line)
                result_lines.append(f"[{task.upper()}] {line.strip()}")
        
        final_results[task] = task_final
    
    print("\n" + "=" * 70)
    
    # Save JSON results
    save_data = {
        'timestamp': timestamp,
        'mode': 'quick' if quick_mode else 'full',
        'config': {
            'model_path': config.test.get('model_path', None),
            'tasks': list(tasks),
            'replication_times_per_task': {t: default_rep_per_task.get(t, 1) for t in tasks},
            'seed': seed,
            'quick_mode': quick_mode,
            'num_quick_samples': num_quick_samples if quick_mode else None,
            'quick_subset_seed': quick_subset_seed if quick_mode else None,
            'motion_nfeats': config.model.showo.motion_nfeats,
            'downsample_rate': config.model.showo.motion_downsample_rate,
        },
        'final_metrics': final_results,
        'per_replication_metrics': {
            task: all_results[task]['per_rep_metrics']
            for task in tasks if task in all_results
        },
    }
    
    json_path = os.path.join(output_dir, f"stage2_eval_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")
    
    # Save text summary
    txt_path = os.path.join(output_dir, f"stage2_eval_{timestamp}.txt")
    with open(txt_path, 'w') as f:
        f.write("UniMotion Stage 2a/2b Evaluation Results\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Mode: {'Quick Test' if quick_mode else 'Full Evaluation'}\n")
        f.write(f"Model: {config.test.get('model_path', None)}\n")
        f.write(f"Tasks: {list(tasks)}\n")
        f.write(f"Replication times per task: { {t: default_rep_per_task.get(t, 1) for t in tasks} }\n")
        f.write(f"Motion features: 269-dim, downsample_rate={config.dataset.motion.downsample_rate}\n")
        f.write(f"LoRA: A={lora_config.get('enable_A', True)}, B={lora_config.get('enable_B', True)}\n" if lora_config else "")
        f.write(f"{'=' * 60}\n\n")
        f.write("Metrics (mean +/- 95% confidence interval):\n")
        for line in result_lines:
            f.write(f"  {line}\n")
        f.write(f"\n{'=' * 60}\n")
        f.write("Per-replication raw values:\n")
        for task in tasks:
            if task in all_results:
                f.write(f"\n  {task.upper()}:\n")
                for key, values in sorted(all_results[task]['per_rep_metrics'].items()):
                    f.write(f"    {key}: {[f'{v:.4f}' for v in values]}\n")
    print(f"Summary saved to {txt_path}")
    
    # Append: comparison against the MotionGPT3 paper format (NLG metrics x100 for easy comparison)
    if 'm2t' in tasks and 'm2t' in all_results:
        m2t_metrics = all_results['m2t']['per_rep_metrics']
        with open(txt_path, 'a') as f:
            f.write(f"\n{'=' * 60}\n")
            f.write("M2T Comparison with MotionGPT3 Paper (NLG metrics ×100):\n")
            f.write("-" * 90 + "\n")
            f.write(f"{'METHOD':<15} {'R@1':<8} {'R@2':<8} {'R@3':<8} {'MMDist↓':<10} "
                    f"{'Bleu@1↑':<10} {'Bleu@4↑':<10} {'Rouge↑':<10} {'CIDEr↑':<10} {'BertScore↑':<10}\n")
            f.write(f"{'GT':<15} ")
            for k in ['gt_R_Precision_Top_1', 'gt_R_Precision_Top_2', 'gt_R_Precision_Top_3']:
                if k in m2t_metrics:
                    f.write(f"{np.mean(m2t_metrics[k]):<8.3f} ")
                else:
                    f.write(f"{'N/A':<8} ")
            if 'gt_Matching_Score' in m2t_metrics:
                f.write(f"{np.mean(m2t_metrics['gt_Matching_Score']):<10.3f}")
            f.write("\n")
            f.write(f"{'Ours':<15} ")
            for k in ['R_Precision_Top_1', 'R_Precision_Top_2', 'R_Precision_Top_3']:
                if k in m2t_metrics:
                    f.write(f"{np.mean(m2t_metrics[k]):<8.3f} ")
                else:
                    f.write(f"{'N/A':<8} ")
            if 'Matching_Score' in m2t_metrics:
                f.write(f"{np.mean(m2t_metrics['Matching_Score']):<10.3f} ")
            for k in ['bleu_1', 'bleu_4', 'ROUGE_L', 'CIDEr', 'Bert_F1']:
                if k in m2t_metrics:
                    f.write(f"{np.mean(m2t_metrics[k]) * 100:<10.3f} ")
                else:
                    f.write(f"{'N/A':<10} ")
            f.write("\n")
            f.write(f"{'MotionGPT3':<15} {'0.553':<8} {'0.756':<8} {'0.853':<8} {'2.524':<10} "
                    f"{'56.363':<10} {'17.661':<10} {'44.997':<10} {'30.980':<10} {'35.850':<10}\n")
            f.write("-" * 90 + "\n")
        print("  M2T comparison table appended to txt file")
    
    # Save M2T debug samples
    if m2t_debug_samples:
        m2t_path = os.path.join(output_dir, f"m2t_debug_{timestamp}.json")
        with open(m2t_path, 'w', encoding='utf-8') as f:
            json.dump(m2t_debug_samples, f, indent=2, ensure_ascii=False)
        print(f"M2T debug samples saved to {m2t_path}")

    if save_vis_manager.enabled:
        finalized_tasks = [t for t in tasks if t in all_results]
        if is_distributed:
            dist.barrier()
        save_vis_manager.finalize_tasks(finalized_tasks)
        save_vis_manager.write_run_config_snapshot(
            {
                'script': os.path.basename(__file__),
                'run_timestamp': run_timestamp,
                'tasks': finalized_tasks,
                'save_vis': {
                    'enabled': True,
                    'root_dir': save_vis_cfg.get('root_dir', DEFAULT_SAVE_VIS_ROOT),
                    'save_all_replications': bool(save_vis_cfg.get('save_all_replications', False)),
                },
                'seed': seed,
                'quick': bool(quick_mode),
                'batch_size': int(config.test.get('batch_size', 32)),
                'model_path': model_path,
                'replication_times_per_task': {t: default_rep_per_task.get(t, 1) for t in tasks},
                'task_settings': {
                    't2m': dict(config.test.get('t2m', {}) or {}),
                    'm2t': dict(config.test.get('m2t', {}) or {}),
                    'pred': dict(config.test.get('pred', {}) or {}),
                },
            }
        )
        print(f"Save-vis outputs written to: {save_vis_manager.run_dir}")
        if is_distributed:
            dist.barrier()
    
    print("\nEvaluation complete!")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
