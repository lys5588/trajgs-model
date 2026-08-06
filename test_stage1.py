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
Stage 1 T2M Evaluation Script for UniMotion.

Evaluates the T2M model trained in Stage 1, using 269-dim features and
the Motion VAE downsample_rate from the current config.

Key points:
- Motion VAE: 269-dim features; downsample_rate determined by config and checkpoint
- Evaluator: uses 263-dim features (generated motion must be truncated)
- Normalization: generation uses the training norm, evaluation uses the eval norm

Usage:
    # Single-GPU test:
    python test_stage1.py config=configs/showo2_motion_test_stage1.yaml

    # Multi-GPU test:
    torchrun --nproc_per_node=8 test_stage1.py \
        config=configs/showo2_motion_test_stage1.yaml

    # Quick test (random subset, reproducible):
    torchrun --nproc_per_node=2 test_stage1.py \
        config=configs/showo2_motion_test_stage1.yaml \
        test.quick=True test.num_quick_samples=512
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
from tqdm import tqdm
from datetime import datetime

from models import Showo2WithMotion, omni_attn_mask_naive
from models.misc import get_text_tokenizer
from models.motion_vae import load_motion_vae
from models.lora import count_lora_parameters
from utils import get_config, load_state_dict, path_to_llm_name, set_seed
from transport import Sampler, create_transport
from motion_length_utils import latent_len_from_motion_length

from evaluation.t2m_evaluator import T2MEvaluator, WordVectorizer
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
        # Single-GPU mode
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
    """
    Gather embeddings from all processes to the main process.
    
    Args:
        embeddings_list: list of [N_i, D] tensors
        world_size: total number of processes
    Returns:
        all_embeddings: aggregated embeddings (valid only on the main process)
    """
    if world_size == 1:
        if len(embeddings_list) == 0:
            return None
        return torch.cat(embeddings_list, dim=0)
    
    if len(embeddings_list) == 0:
        local_embeddings = torch.zeros(0, 512).cuda()
    else:
        local_embeddings = torch.cat(embeddings_list, dim=0).cuda()
    
    # Collect the embedding count from each process
    local_count = torch.tensor([local_embeddings.shape[0]], device='cuda')
    all_counts = [torch.zeros(1, device='cuda', dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(all_counts, local_count)
    all_counts = [int(c.item()) for c in all_counts]
    
    # Pad to max size for gathering
    max_count = max(all_counts)
    if max_count == 0:
        return None
    
    embed_dim = local_embeddings.shape[1] if local_embeddings.shape[0] > 0 else 512
    
    # Pad local embeddings
    if local_embeddings.shape[0] < max_count:
        pad = torch.zeros(max_count - local_embeddings.shape[0], embed_dim, device='cuda')
        local_embeddings = torch.cat([local_embeddings, pad], dim=0)
    
    # All gather
    gathered = [torch.zeros(max_count, embed_dim, device='cuda') for _ in range(world_size)]
    dist.all_gather(gathered, local_embeddings)
    
    # Concatenate the real data only on the main process
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


# ==================== Motion Normalization Conversion ====================

def renorm4t2m(motion, ori_mean, ori_std, eval_mean, eval_std):
    """
    Convert motion from training norm to eval norm.
    
    Args:
        motion: [bs, seq_len, 263] motion in training normalization
        ori_mean, ori_std: training mean/std
        eval_mean, eval_std: evaluation mean/std
    Returns:
        [bs, seq_len, 263] motion in eval normalization
    """
    # Denormalize
    motion = motion * ori_std + ori_mean
    # Renormalize
    motion = (motion - eval_mean) / eval_std
    return motion


def ensure_eval_motion_dim(motion: torch.Tensor, eval_dim: int = 263) -> torch.Tensor:
    """
    Unify eval dims: keep only the first 263 dims (ignore the trailing 6), zero-pad if shorter.
    """
    if motion.shape[-1] > eval_dim:
        return motion[..., :eval_dim]
    if motion.shape[-1] < eval_dim:
        pad = torch.zeros(*motion.shape[:-1], eval_dim - motion.shape[-1], device=motion.device, dtype=motion.dtype)
        return torch.cat([motion, pad], dim=-1)
    return motion


def get_metric_statistics(values, replication_times):
    """
    Compute the mean and 95% confidence interval of metrics (identical to MotionGPT3).
    
    Args:
        values: np.array of metric values across replications
        replication_times: number of replications
    Returns:
        mean, conf_interval
    """
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
    
    Returns:
        cond_tokens, uncond_tokens, cond_positions, uncond_positions
    """
    batch_cond_tokens = []
    batch_cond_positions = []
    batch_uncond_tokens = []
    batch_uncond_positions = []
    
    for caption in captions:
        # Tokenize text
        text_tokens = text_tokenizer(caption, add_special_tokens=False)['input_ids'][:max_text_len]
        
        # Conditional sequence: [BOS] text [BOM][MOT_PAD*N][EOM][EOS]
        modality_positions = torch.tensor([len(text_tokens) + 2, num_motion_tokens]).unsqueeze(0)
        full_seq = [bos_id] + text_tokens + [bom_id] + [mot_pad_id] * num_motion_tokens + [eom_id] + [eos_id]
        pad_len = max_seq_len - len(full_seq)
        full_seq = full_seq + [pad_id] * pad_len
        
        batch_cond_tokens.append(torch.tensor(full_seq[:max_seq_len]))
        batch_cond_positions.append(modality_positions)
        
        # Unconditional sequence: [BOS][BOM][MOT_PAD*N][EOM][EOS]
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


# ==================== Generation Functions ====================

@torch.no_grad()
def generate_t2m(
    model, motion_vae, sampler, text_tokens, modality_positions,
    latent_len, motion_latent_dim, motion_lengths, max_seq_len,
    guidance_scale, num_inference_steps, config, device, weight_type,
):
    """
    Generate T2M motion.
    
    Args:
        model: Showo2WithMotion model
        motion_vae: Motion VAE model
        sampler: ODE sampler
        text_tokens: [bs*2, seq_len] conditional + unconditional token sequences
        modality_positions: [bs*2, 1, 2] motion positions
        latent_len: motion latent length
        motion_latent_dim: motion latent dimension
        motion_lengths: target motion length of each sample
        max_seq_len: maximum sequence length
        guidance_scale: CFG guidance scale
        num_inference_steps: number of sampling steps
        config: config object
        device: device
        weight_type: weight dtype
        
    Returns:
        decoded_motions: [bs, motion_length, nfeats] generated motion
        motion_lengths: list of motion lengths
    """
    bs = text_tokens.shape[0] // 2 if guidance_scale > 0 else text_tokens.shape[0]
    
    # Initialize noise
    z = torch.randn((latent_len, bs, motion_latent_dim)).to(weight_type).to(device)
    if guidance_scale > 0:
        z = torch.cat([z, z], dim=1)
    
    # Create the attention mask
    block_mask = omni_attn_mask_naive(
        text_tokens.size(0), text_tokens.size(1), modality_positions, device
    ).to(weight_type)
    
    # Set model kwargs
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
        num_steps=num_inference_steps,
        atol=config.transport.atol,
        rtol=config.transport.rtol,
        reverse=config.transport.reverse,
        time_shifting_factor=config.transport.time_shifting_factor
    )
    
    samples = sample_fn(z, model.t2m_generate, **model_kwargs)[-1]
    
    # If CFG is used, take the conditional part
    if guidance_scale > 0:
        samples = samples[:, :bs, :]
    
    # Decode motion
    with torch.autocast(device_type='cuda', enabled=False):
        decoded_motions = motion_vae.float().decode(samples.float(), motion_lengths)
    
    # Handle list outputs
    if isinstance(decoded_motions, list):
        max_len = max(m.shape[0] for m in decoded_motions)
        padded = []
        for m in decoded_motions:
            if m.shape[0] < max_len:
                m = torch.cat([m, torch.zeros(max_len - m.shape[0], m.shape[1], device=m.device)], dim=0)
            padded.append(m)
        decoded_motions = torch.stack(padded)
    
    return decoded_motions, motion_lengths


# ==================== Main Test Function ====================

@torch.no_grad()
def main():
    # Initialize the distributed environment
    rank, world_size, local_rank = setup_distributed()
    is_distributed = world_size > 1
    
    if is_main_process():
        print("=" * 60)
        print("Stage 1 T2M Evaluation")
        print("Motion Features: 269-dim, Downsample Rate: 1")
        print("=" * 60)
    
    # Load config
    config = get_config()
    seed = config.test.get('seed', 42)
    set_seed(seed + rank)  # different seed per process
    run_timestamp = sync_run_timestamp(datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # Set device
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    weight_type = torch.bfloat16 if config.model.weight_type == "bfloat16" else torch.float32
    
    if is_main_process():
        print(f"Device: {device}")
        print(f"Distributed: {is_distributed}, World size: {world_size}")
        print(f"Weight type: {weight_type}")
    
    # Create output directory
    output_dir = config.test.get('output_dir', 'outputs/stage1_eval')
    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)
    
    # Synchronize all processes
    if is_distributed:
        dist.barrier()
    
    # ==================== Load Models ====================
    if is_main_process():
        print("\n[1/5] Loading models...")
    
    # Load the Motion VAE (269-dim, downsample_rate determined by the checkpoint)
    motion_vae, motion_vae_config = load_motion_vae(
        config.model.motion_vae.pretrained_model_path,
        device=device, freeze=True
    )
    motion_vae = motion_vae.to(weight_type)
    
    if is_main_process():
        print(f"  Motion VAE loaded: nfeats={motion_vae_config.get('nfeats', 269)}, "
              f"downsample_rate={motion_vae_config.get('downsample_rate', 1)}")
    
    # Load tokenizer
    text_tokenizer, showo_token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[config.model.showo.llm_model_path],
        add_motion_tokens=True
    )
    config.model.showo.llm_vocab_size = len(text_tokenizer)
    
    # Load model
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
    
    # Apply the LoRA structure
    lora_config = config.get('lora', None)
    lora_weights_path = None
    
    if lora_config is not None and lora_config.get('enabled', False):
        if is_main_process():
            print("  Applying Token-Level LoRA structure...")
        
        lora_params_dict = {
            'rank_A': lora_config.get('rank_A', 16),
            'rank_B': lora_config.get('rank_B', 16),
            'alpha': lora_config.get('alpha', 16),
            'dropout': 0.0,
            'enable_A': lora_config.get('enable_A', False),
            'enable_B': lora_config.get('enable_B', True),
            'target_modules': lora_config.get('target_modules', ['q_proj', 'k_proj', 'v_proj', 'o_proj']),
        }
        model.apply_lora(lora_params_dict)
        
        if model_path:
            # Look for LoRA weights
            original_model_path = config.test.get('model_path', None)
            if original_model_path:
                if 'unwrapped_model' in model_path:
                    lora_weights_path = os.path.join(os.path.dirname(model_path), 'lora_weights.pt')
                else:
                    lora_weights_path = os.path.join(original_model_path, 'lora_weights.pt')

    save_vis_cfg = config.test.get('save_vis', {})
    save_vis_manager = SaveVisManager(
        enabled=bool(save_vis_cfg.get('enabled', False)),
        root_dir=save_vis_cfg.get('root_dir', DEFAULT_SAVE_VIS_ROOT),
        script_tag=os.path.splitext(os.path.basename(__file__))[0],
        tasks=['t2m'],
        seed=seed,
        quick=bool(config.test.get('quick', False)),
        batch_size=int(config.test.get('batch_size', 4)),
        ckpt_path=model_path,
        run_timestamp=run_timestamp,
        rank=rank,
        world_size=world_size,
        is_main_process=is_main_process(),
        save_all_replications=bool(save_vis_cfg.get('save_all_replications', False)),
    )
    
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
            print(f"    Loaded {len(filtered_state_dict)} parameters")
    
    # Load LoRA weights
    if lora_weights_path and os.path.exists(lora_weights_path):
        if is_main_process():
            print(f"  Loading LoRA weights from {lora_weights_path}")
        model.load_lora(lora_weights_path)
    
    model.to(weight_type)
    model.eval()
    
    # Create the transport sampler
    transport = create_transport(
        path_type=config.transport.path_type,
        prediction=config.transport.prediction,
        loss_weight=config.transport.loss_weight,
        train_eps=config.transport.train_eps,
        sample_eps=config.transport.sample_eps,
        snr_type=config.transport.snr_type,
        do_shift=config.transport.do_shift,
        seq_len=(
            int(config.dataset.preprocessing.max_motion_tokens)
            + (1 if config.model.showo.add_time_embeds else 0)
        ),
    )
    sampler = Sampler(transport)
    
    # ==================== Load Evaluators ====================
    if is_main_process():
        print("\n[2/5] Loading T2M evaluator...")
    
    t2m_evaluator_path = config.test.get('t2m_evaluator_path', 'deps/t2m/t2m')
    glove_dir = config.test.get('glove_dir', 'deps/glove/glove')
    unit_length = config.dataset.motion.get('unit_length', 1)
    
    # Note: the evaluator uses 263-dim features, unit_length=4
    # We convert 269 dims to 263 dims before evaluation
    evaluator = T2MEvaluator(
        t2m_evaluator_path=t2m_evaluator_path,
        glove_dir=glove_dir,
        dataname='t2m',
        device=device,
        unit_length=4,  # the evaluator always uses 4
    )
    
    if is_main_process():
        print("  T2M evaluator loaded")
    
    # ==================== Load Datasets ====================
    if is_main_process():
        print("\n[3/5] Loading evaluation dataset...")
    
    # Eval data uses raw 263-dim HumanML3D
    data_root = config.dataset.motion.data_root
    
    max_motion_length = config.test.get('max_motion_length', 200)
    min_motion_length = config.test.get('min_motion_length', 20)
    batch_size = config.test.get('batch_size', 4)
    
    mean_eval_path = config.test.get('mean_eval_path', None)
    std_eval_path = config.test.get('std_eval_path', None)
    eval_max_text_len = config.dataset.preprocessing.get('eval_max_text_len', 20)
    
    if is_main_process():
        print(f"  Data root: {data_root}")
    
    # Word vectorizer
    w_vectorizer = None
    if os.path.exists(glove_dir):
        w_vectorizer = WordVectorizer(glove_dir)
    
    # The eval dataset uses raw 263-dim HumanML3D data
    eval_dataset_full = MotionEvalDataset(
        data_root=data_root,
        split='test',
        w_vectorizer=w_vectorizer,
        max_motion_length=max_motion_length,
        min_motion_length=min_motion_length,
        mean_eval_path=mean_eval_path,
        std_eval_path=std_eval_path,
        max_text_len=eval_max_text_len,
        unit_length=4,  # the eval dataset uses unit_length=4
        tiny=config.test.get('tiny', False),
    )
    
    from torch.utils.data import DataLoader, Subset

    # Quick mode: randomly sampled eval subset (fixed seed, reproducible)
    quick_mode = bool(config.test.get('quick', False))
    num_quick_samples = int(config.test.get('num_quick_samples', 256))
    quick_subset_seed = int(config.test.get('quick_subset_seed', seed))
    if quick_mode:
        quick_count = min(len(eval_dataset_full), max(1, num_quick_samples))
        base_keys = eval_dataset_full.name_list[eval_dataset_full.pointer:]
        subset_indices = select_deterministic_quick_indices(base_keys, quick_count, quick_subset_seed)
        eval_dataset = Subset(eval_dataset_full, subset_indices)
    else:
        eval_dataset = eval_dataset_full
    
    # Distributed sampler
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
        collate_fn=eval_dataset_full.collate_fn,
        drop_last=False,
        pin_memory=True,
    )
    
    if is_main_process():
        if quick_mode:
            print(f"  Quick mode enabled (deterministic random subset): using {len(eval_dataset)} samples")
            print(f"  Quick subset seed: {quick_subset_seed}")
        print(f"  Dataset size: {len(eval_dataset)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Total batches: {len(eval_dataloader)}")
    
    # Load normalization parameters (269-dim, from data_root)
    mean_train = np.load(os.path.join(data_root, 'Mean.npy'))
    std_train = np.load(os.path.join(data_root, 'Std.npy'))
    
    if is_main_process():
        print(f"  Train mean/std shape (269-dim): {mean_train.shape}")
    
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
        print(f"  Eval mean/std shape (263-dim): {mean_eval.shape}")
    
    # Convert to tensors
    mean_train_t = torch.from_numpy(mean_train).float().to(device)
    std_train_t = torch.from_numpy(std_train).float().to(device)
    mean_eval_t = torch.from_numpy(mean_eval).float().to(device)
    std_eval_t = torch.from_numpy(std_eval).float().to(device)
    
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
    
    # ==================== Run Evaluation with Replications ====================
    # Consistent with MotionGPT3: repeat the evaluation multiple times and report mean ± 95% confidence interval
    # Stage 1 has only the T2M task, default 1 replication; set it in the config for more
    replication_times = config.test.get('replication_times', 1)
    
    guidance_scale = config.test.t2m.get('guidance_scale', 5.0)
    num_inference_steps = config.test.t2m.get('num_inference_steps', 50)
    
    if is_main_process():
        print(f"\n[4/5] Running T2M evaluation ({replication_times} replications)...")
        print(f"  Each replication generates motions for all {len(eval_dataset)} test samples")
        print(f"  Guidance scale: {guidance_scale}")
        print(f"  Inference steps: {num_inference_steps}")
    
    all_metrics = {}
    
    for rep_i in range(replication_times):
        if is_main_process():
            print(f"\n{'='*40} Replication {rep_i + 1}/{replication_times} {'='*40}")
        
        evaluator.reset()
        total_samples = 0
        
        pbar = tqdm(
            eval_dataloader,
            desc=f"Rep {rep_i+1}/{replication_times}",
            disable=not is_main_process()
        )
        save_this_rep = save_vis_manager.should_save_replication(rep_i)
        
        for batch_idx, batch in enumerate(pbar):
            gt_motions = ensure_eval_motion_dim(batch['motions'].to(device))
            gt_lengths = batch['m_lengths']
            captions = batch['captions']
            fnames = batch.get('fnames', [])
            word_embs = batch['word_embs'].to(device)
            pos_ohot = batch['pos_ohot'].to(device)
            sent_lens = batch['sent_lens'].to(device)
            
            bs = gt_motions.shape[0]
            use_gt_length = config.test.get('use_gt_length', True)
            
            if use_gt_length:
                motion_lengths = [max(1, int(l)) for l in gt_lengths]
            else:
                fixed_len = max(1, int(max_motion_length))
                motion_lengths = [fixed_len] * bs

            motion_length = max(motion_lengths)
            latent_len = latent_len_from_motion_length(motion_length, downsample_rate)
            num_motion_tokens = latent_len + 1 if config.model.showo.add_time_embeds else latent_len
            
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
            
            gt_max_len = gt_motions.shape[1]
            gen_max_len = gen_motions.shape[1]
            if gen_max_len < gt_max_len:
                pad = torch.zeros(bs, gt_max_len - gen_max_len, 263, device=device, dtype=gen_motions.dtype)
                gen_motions = torch.cat([gen_motions, pad], dim=1)
            elif gen_max_len > gt_max_len:
                gen_motions = gen_motions[:, :gt_max_len, :]
            
            mean_train_263 = mean_train_t[:263] if mean_train_t.shape[0] > 263 else mean_train_t
            std_train_263 = std_train_t[:263] if std_train_t.shape[0] > 263 else std_train_t
            
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
            gen_motions = renorm4t2m(gen_motions, mean_train_263, std_train_263, mean_eval_t, std_eval_t)
            eval_gen_lengths = [min(int(gen_lengths[i]), int(gt_lengths[i])) for i in range(bs)]
            
            evaluator.update(
                gt_motion=gt_motions.float(),
                gen_motion=gen_motions.float(),
                gt_lengths=gt_lengths,
                gen_lengths=eval_gen_lengths,
                word_embs=word_embs.float(),
                pos_ohot=pos_ohot.float(),
                cap_lens=sent_lens,
            )
            
            total_samples += bs
            
            # Debug info (only first batch of first replication)
            if batch_idx == 0 and rep_i == 0 and is_main_process():
                print(f"\n  [Debug] Batch 0:")
                print(f"    GT motion shape: {gt_motions.shape}")
                print(f"    Gen motion shape: {gen_motions.shape}")
                print(f"    GT lengths: {gt_lengths[:3]}")
                print(f"    Caption: {captions[0][:100]}...")
        
        # Aggregate distributed results
        if is_distributed:
            dist.barrier()
            
            total_samples_tensor = torch.tensor([total_samples], device=device)
            dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
            total_samples = int(total_samples_tensor.item())
            
            all_text_embs = gather_embeddings(evaluator.text_embeddings, world_size)
            all_gen_motion_embs = gather_embeddings(evaluator.gen_motion_embeddings, world_size)
            all_gt_motion_embs = gather_embeddings(evaluator.gt_motion_embeddings, world_size)
            
            if is_main_process():
                evaluator.text_embeddings = [all_text_embs] if all_text_embs is not None else []
                evaluator.gen_motion_embeddings = [all_gen_motion_embs] if all_gen_motion_embs is not None else []
                evaluator.gt_motion_embeddings = [all_gt_motion_embs] if all_gt_motion_embs is not None else []
        
        # Compute metrics for this replication
        if is_main_process():
            rep_metrics = evaluator.compute()
            
            fid_val = rep_metrics.get('FID', float('nan'))
            r3_val = rep_metrics.get('R_Precision_Top_3', float('nan'))
            mm_val = rep_metrics.get('Matching_Score', float('nan'))
            div_val = rep_metrics.get('Diversity', float('nan'))
            print(f"  Rep {rep_i+1}: FID={fid_val:.4f}, R@3={r3_val:.4f}, "
                  f"MM-Dist={mm_val:.4f}, Diversity={div_val:.4f}")
            
            for key, value in rep_metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = [value]
                else:
                    all_metrics[key].append(value)
        
        if is_distributed:
            dist.barrier()
    
    # ==================== Compute Final Statistics ====================
    if not is_main_process():
        if save_vis_manager.enabled and is_distributed:
            dist.barrier()
            dist.barrier()
        cleanup_distributed()
        return
    
    print(f"\n[5/5] Computing final statistics over {replication_times} replications...")
    
    if total_samples == 0:
        print("Warning: No samples processed. Skipping metrics computation.")
        cleanup_distributed()
        return
    
    final_metrics = {}
    for key, values in all_metrics.items():
        mean_val, conf_val = get_metric_statistics(np.array(values), replication_times)
        final_metrics[f"{key}/mean"] = float(mean_val)
        final_metrics[f"{key}/conf_interval"] = float(conf_val)
    
    # ==================== Print Results ====================
    print("\n" + "=" * 60)
    print("STAGE 1 T2M EVALUATION (mean +/- 95% confidence interval)")
    print("=" * 60)
    print(f"Samples per replication: {total_samples}")
    print(f"Replication times: {replication_times}")
    print(f"Model path: {config.test.get('model_path', None)}")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Inference steps: {num_inference_steps}")
    print("-" * 60)
    
    metric_order = [
        'R_Precision_Top_1', 'R_Precision_Top_2', 'R_Precision_Top_3',
        'new_fid', 'new_gt_fid',
        'FID', 'MMDist', 'Diversity',
        'gt_R_Precision_Top_1', 'gt_R_Precision_Top_2', 'gt_R_Precision_Top_3',
        'gt_FID', 'gt_MMDist', 'gt_Diversity',
        # Keep legacy names for backward compatibility with old logs/scripts.
        'Matching_Score', 'gt_Matching_Score',
    ]
    
    result_lines = []
    for key in metric_order:
        mean_key = f"{key}/mean"
        conf_key = f"{key}/conf_interval"
        if mean_key in final_metrics:
            mean_val = final_metrics[mean_key]
            conf_val = final_metrics[conf_key]
            arrow = "↑" if key in ['R_Precision_Top_1', 'R_Precision_Top_2', 'R_Precision_Top_3',
                                    'Diversity', 'gt_Diversity',
                                    'gt_R_Precision_Top_1', 'gt_R_Precision_Top_2', 'gt_R_Precision_Top_3'] else "↓"
            line = f"{key} {arrow}: {mean_val:.4f} +/- {conf_val:.4f}"
            print(f"  {line}")
            result_lines.append(line)
    
    for key in sorted(all_metrics.keys()):
        if key not in metric_order and f"{key}/mean" in final_metrics:
            mean_val = final_metrics[f"{key}/mean"]
            conf_val = final_metrics[f"{key}/conf_interval"]
            line = f"{key}: {mean_val:.4f} +/- {conf_val:.4f}"
            print(f"  {line}")
            result_lines.append(line)
    
    print("=" * 60)
    
    # ==================== Save Results ====================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        'timestamp': timestamp,
        'config': {
            'model_path': config.test.get('model_path', None),
            'replication_times': replication_times,
            'guidance_scale': guidance_scale,
            'num_inference_steps': num_inference_steps,
            'seed': seed,
            'motion_nfeats': config.model.showo.motion_nfeats,
            'downsample_rate': config.model.showo.motion_downsample_rate,
            'num_samples_per_rep': total_samples,
        },
        'final_metrics': final_metrics,
        'per_replication_metrics': {k: v for k, v in all_metrics.items()},
    }
    
    json_path = os.path.join(output_dir, f"stage1_eval_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {json_path}")
    
    txt_path = os.path.join(output_dir, f"stage1_eval_{timestamp}.txt")
    with open(txt_path, 'w') as f:
        f.write("UniMotion Stage 1 T2M Evaluation Results\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {config.test.get('model_path', None)}\n")
        f.write(f"Samples per replication: {total_samples}\n")
        f.write(f"Replication times: {replication_times}\n")
        f.write(f"Guidance scale: {guidance_scale}\n")
        f.write(f"Inference steps: {num_inference_steps}\n")
        f.write(f"{'=' * 50}\n\n")
        f.write("Metrics (mean +/- 95% confidence interval):\n")
        for line in result_lines:
            f.write(f"  {line}\n")
        f.write(f"\n{'=' * 50}\n")
        f.write("Per-replication raw values:\n")
        for key in metric_order:
            if key in all_metrics:
                vals = all_metrics[key]
                f.write(f"  {key}: {[f'{v:.4f}' for v in vals]}\n")
    print(f"Summary saved to {txt_path}")

    if save_vis_manager.enabled:
        if is_distributed:
            dist.barrier()
        save_vis_manager.finalize_tasks(['t2m'])
        save_vis_manager.write_run_config_snapshot(
            {
                'script': os.path.basename(__file__),
                'run_timestamp': run_timestamp,
                'tasks': ['t2m'],
                'save_vis': {
                    'enabled': True,
                    'root_dir': save_vis_cfg.get('root_dir', DEFAULT_SAVE_VIS_ROOT),
                    'save_all_replications': bool(save_vis_cfg.get('save_all_replications', False)),
                },
                'seed': seed,
                'quick': quick_mode,
                'batch_size': batch_size,
                'model_path': model_path,
                'replication_times': replication_times,
                't2m': {
                    'guidance_scale': guidance_scale,
                    'num_inference_steps': num_inference_steps,
                    'use_gt_length': bool(config.test.get('use_gt_length', True)),
                },
            }
        )
        print(f"Save-vis outputs written to: {save_vis_manager.run_dir}")
        if is_distributed:
            dist.barrier()
    
    print("\nEvaluation complete.")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
