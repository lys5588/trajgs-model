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
Stage 4 Evaluation Script for UniMotion, T2M full-caption protocol variant.

Same engine as `test_stage4.py`, plus the T2M all-captions protocol:
every caption associated with each test sample is expanded and evaluated
(`test.t2m.use_all_captions=true`). Prefer launching it via the thin wrapper
`test_t2m_allcaps.py`, which forces `test.tasks=[t2m]`.

Note: the image-to-motion (img2pose/img2text) and MGIE evaluation suite is
not included in this release; requesting those tasks raises
NotImplementedError.

Usage:
    torchrun --nproc_per_node=8 test_stage4_t2m_allcaps.py \
      config=configs/showo2_motion_test_stage4.yaml \
      test.tasks=[t2m] \
      test.t2m.use_all_captions=true

    # If fragmentation-related OOM is a concern, consider adding:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=8 \
      test_stage4_t2m_allcaps.py config=configs/showo2_motion_test_stage4.yaml

Compatible with the existing stage2b evaluation; still supports
replication_times and mean +/- 95% CI reporting.
"""

import os
import sys
import json
import hashlib
import re
import warnings
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
import torch.nn.functional as F
import torch.distributed as dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from models import Showo2WithMotion, WanVAE, omni_attn_mask_naive
from models.misc import get_text_tokenizer
from models.motion_vae import load_motion_vae
from models.lora import count_lora_parameters
from utils import get_config, load_state_dict, path_to_llm_name, set_seed
from transport import Sampler, create_transport
from motion_length_utils import latent_len_from_motion_length, latent_lens_from_motion_lengths

from evaluation.t2m_evaluator import (
    T2MEvaluator, WordVectorizer,
    euclidean_distance_matrix, calculate_top_k,
    calculate_activation_statistics, calculate_frechet_distance, calculate_diversity,
)
from evaluation.m2t_evaluator import M2TEvaluator
from evaluation.pred_evaluator import PredEvaluator, feats2joints_humanml3d
from evaluation.text_nlg_evaluator import TextNLGEvaluator
from evaluation.save_vis_utils import SaveVisManager
from datasets.motion_eval_dataset import MotionEvalDataset
from datasets.motion_edit_dataset import MotionEditDataset
from datasets.motion_dataset import TextToTextDataset

# NOTE: The img2pose/mgie evaluators and the H36M eval datasets are imported
# lazily inside their task branches; those tasks are not included in this
# release (see RELEASE_EXCLUDED_TASKS below).
RELEASE_EXCLUDED_TASKS = ('img2pose', 'img2text', 'mgie')

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


def gather_object_list(local_object, world_size):
    """Gather python objects from all ranks. Returns merged list on main rank."""
    if world_size == 1:
        return [local_object]
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_object)
    if is_main_process():
        return gathered
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


def build_t2t_entry_keys(dataset):
    """Build stable keys for TextToTextDataset instead of relying on indices only."""
    names = list(getattr(dataset, "name_list", []))
    num_tasks = max(1, len(getattr(dataset, "t2t_tasks", [])))
    total = len(dataset)
    if not names:
        return [f"idx:{i}" for i in range(total)]
    keys = []
    for idx in range(total):
        data_idx = idx // num_tasks
        task_idx = idx % num_tasks
        name = names[data_idx] if data_idx < len(names) else f"data_{data_idx}"
        keys.append(f"{name}::task{task_idx}")
    return keys


def _scale_batch_size(value, scale):
    """Scale the batch size by `scale`, keeping it >= 1."""
    return max(1, int(round(float(value) * float(scale))))


def apply_stage4_auto_profile(config, tasks):
    """
    Automatically switch the stage4 evaluation profile per task:
    - motion-only: mainly for pure-motion tasks (no image tasks)
    - full-stage4: includes the image tasks (img2pose/img2text/mgie)
    """
    auto_cfg = config.test.get("auto_profile", None)
    if auto_cfg is None or not bool(auto_cfg.get("enabled", False)):
        return None

    image_tasks = set(auto_cfg.get("image_tasks", ["img2pose", "img2text", "mgie"]))
    use_full_stage4 = any(t in image_tasks for t in tasks)
    profile_name = "full_stage4" if use_full_stage4 else "motion_only"
    profile_cfg = auto_cfg.get(profile_name, None)

    summary = {
        "enabled": True,
        "profile": profile_name,
        "image_tasks": sorted(list(image_tasks)),
        "changes": [],
    }
    if profile_cfg is None:
        summary["warning"] = f"Missing config.test.auto_profile.{profile_name}, skip applying."
        return summary

    # 1) Switch max_seq_length per task
    max_seq_len = profile_cfg.get("max_seq_length", None)
    if max_seq_len is not None:
        old_val = int(config.dataset.preprocessing.get("max_seq_length", max_seq_len))
        new_val = int(max_seq_len)
        config.dataset.preprocessing.max_seq_length = new_val
        summary["changes"].append(
            f"dataset.preprocessing.max_seq_length: {old_val} -> {new_val}"
        )

    # 2) Scale batch size by the profile ratio (scaling applies after CLI overrides)
    bs_scale = float(profile_cfg.get("batch_size_scale", 1.0))
    if abs(bs_scale - 1.0) > 1e-8:
        test_batch_size = config.test.get("batch_size", None)
        if test_batch_size is not None:
            old_bs = int(test_batch_size)
            new_bs = _scale_batch_size(old_bs, bs_scale)
            config.test.batch_size = new_bs
            summary["changes"].append(f"test.batch_size: {old_bs} -> {new_bs}")

        for task_name in ["edit", "img2pose", "img2text", "mgie", "t2t"]:
            task_cfg = config.test.get(task_name, None)
            if task_cfg is None:
                continue
            if task_cfg.get("batch_size", None) is not None:
                old_task_bs = int(task_cfg.get("batch_size"))
                new_task_bs = _scale_batch_size(old_task_bs, bs_scale)
                task_cfg.batch_size = new_task_bs
                summary["changes"].append(
                    f"test.{task_name}.batch_size: {old_task_bs} -> {new_task_bs}"
                )
            if task_name == "edit" and task_cfg.get("retrieval_batch_size", None) is not None:
                old_ret_bs = int(task_cfg.get("retrieval_batch_size"))
                new_ret_bs = _scale_batch_size(old_ret_bs, bs_scale)
                task_cfg.retrieval_batch_size = new_ret_bs
                summary["changes"].append(
                    f"test.edit.retrieval_batch_size: {old_ret_bs} -> {new_ret_bs}"
                )

    return summary


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


def zero_tail_dims(motion: torch.Tensor, keep_dim: int = 263) -> torch.Tensor:
    """Keep the first keep_dim dims and zero out the rest (to explicitly ignore trailing features)."""
    if motion.shape[-1] <= keep_dim:
        return motion
    motion = motion.clone()
    motion[..., keep_dim:] = 0
    return motion


def get_metric_statistics(values, replication_times):
    """Compute the mean and 95% confidence interval of metrics (identical to MotionGPT3)."""
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def _compute_retrieval_cols(sim_matrix: np.ndarray) -> np.ndarray:
    """
    Compute GT rank positions (0-indexed) from a similarity matrix.
    Mirrors MotionFix's contrastive ranking logic with tie-averaging.
    """
    dists = -sim_matrix
    sorted_dists = np.sort(dists, axis=1)
    gt_dists = np.diag(dists)[:, None]
    rows, cols = np.where((sorted_dists - gt_dists) == 0)
    num_queries = sim_matrix.shape[0]

    if rows.size > num_queries:
        # tie averaging, same as MotionFix metrics.py::break_ties_average
        locs = np.argwhere((sorted_dists - gt_dists) == 0)
        steps = np.diff(locs[:, 0])
        splits = np.nonzero(steps)[0] + 1
        splits = np.insert(splits, 0, 0)
        summed_cols = np.add.reduceat(locs[:, 1], splits)
        counts = np.diff(np.append(splits, locs.shape[0]))
        cols = summed_cols / counts

    return cols


def _retrieval_r_at_k(sim_matrix: np.ndarray, ks=(1, 2, 3)) -> dict:
    cols = _compute_retrieval_cols(sim_matrix)
    num_queries = max(len(cols), 1)
    out = {}
    for k in ks:
        out[f"R@{k}"] = 100.0 * float(np.sum(cols < int(k))) / num_queries
    return out


def compute_motionfix_style_retrieval_metrics(
    source_emb: torch.Tensor,
    target_emb: torch.Tensor,
    pred_emb: torch.Tensor,
    batch_eval_size: int = 32,
    seed: int = 0,
):
    """
    MotionFix-style retrieval metrics for edit task.
    - source_target: source -> GT target retrieval (R@1/2/3)
    - target_generated: GT target -> generated target retrieval (R@1/2/3)
    Reports both batch protocol (size=32) and full protocol.
    """
    if source_emb is None or target_emb is None or pred_emb is None:
        return {}

    if source_emb.shape[0] == 0:
        return {}

    # cosine sim matrix (same style as MotionFix get_sim_matrix)
    source_np = F.normalize(source_emb.float(), dim=-1).cpu().numpy()
    target_np = F.normalize(target_emb.float(), dim=-1).cpu().numpy()
    pred_np = F.normalize(pred_emb.float(), dim=-1).cpu().numpy()

    sim_s2t = source_np @ target_np.T
    sim_t2g = target_np @ pred_np.T

    full_s2t = _retrieval_r_at_k(sim_s2t)
    full_t2g = _retrieval_r_at_k(sim_t2g)

    n = sim_s2t.shape[0]
    if n >= batch_eval_size:
        rng = np.random.RandomState(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        num_batches = n // batch_eval_size
        batch_s2t_vals = []
        batch_t2g_vals = []
        for i in range(num_batches):
            cur = idx[i * batch_eval_size:(i + 1) * batch_eval_size]
            batch_s2t_vals.append(_retrieval_r_at_k(sim_s2t[np.ix_(cur, cur)]))
            batch_t2g_vals.append(_retrieval_r_at_k(sim_t2g[np.ix_(cur, cur)]))

        batch_s2t = {
            k: float(np.mean([m[k] for m in batch_s2t_vals]))
            for k in ["R@1", "R@2", "R@3"]
        }
        batch_t2g = {
            k: float(np.mean([m[k] for m in batch_t2g_vals]))
            for k in ["R@1", "R@2", "R@3"]
        }
    else:
        # fallback to full protocol when sample size < batch_eval_size
        batch_s2t = full_s2t
        batch_t2g = full_t2g

    return {
        "R@1_s2t": float(batch_s2t["R@1"]),
        "R@2_s2t": float(batch_s2t["R@2"]),
        "R@3_s2t": float(batch_s2t["R@3"]),
        "R@1": float(batch_t2g["R@1"]),
        "R@2": float(batch_t2g["R@2"]),
        "R@3": float(batch_t2g["R@3"]),
        "R@1_s2t_full": float(full_s2t["R@1"]),
        "R@2_s2t_full": float(full_s2t["R@2"]),
        "R@3_s2t_full": float(full_s2t["R@3"]),
        "R@1_full": float(full_t2g["R@1"]),
        "R@2_full": float(full_t2g["R@2"]),
        "R@3_full": float(full_t2g["R@3"]),
    }


def compute_edit_retrieval_metrics(
    source_emb: torch.Tensor,
    target_emb: torch.Tensor,
    pred_emb: torch.Tensor,
    batch_eval_size: int = 32,
    seed: int = 0,
    diversity_times: int = 300,
):
    """
    Edit retrieval metrics. Protocol follows Table 8 of the MotionFix paper:
    - Top1/2/3: retrieval R@k (%) from source -> edited_target
    - MM Dist: matching distance from source -> edited_target (trace pairwise L2 / N, lower is better)
    - FID: edited_target vs GT target (motion embedding space)
    - Diversity: embedding diversity of edited_target
    Also reports Real (= source -> GT target) reference values.
    """
    if source_emb is None or target_emb is None or pred_emb is None:
        return {}
    if source_emb.shape[0] == 0 or target_emb.shape[0] == 0 or pred_emb.shape[0] == 0:
        return {}

    n = min(source_emb.shape[0], target_emb.shape[0], pred_emb.shape[0])
    source_emb = source_emb[:n].float()
    target_emb = target_emb[:n].float()
    pred_emb = pred_emb[:n].float()

    src_np = F.normalize(source_emb, dim=-1).cpu().numpy()
    tgt_np = F.normalize(target_emb, dim=-1).cpu().numpy()
    pred_np = F.normalize(pred_emb, dim=-1).cpu().numpy()

    sim_s2g = src_np @ pred_np.T
    sim_s2t = src_np @ tgt_np.T

    # Matching distance in embedding space (same quantity style as t2m Matching Score)
    dist_s2g = euclidean_distance_matrix(source_emb, pred_emb)
    dist_s2t = euclidean_distance_matrix(source_emb, target_emb)
    full_s2g = _retrieval_r_at_k(sim_s2g, ks=(1, 2, 3))
    full_s2t = _retrieval_r_at_k(sim_s2t, ks=(1, 2, 3))
    full_mm_s2g = float(torch.trace(dist_s2g).item() / n)
    full_mm_s2t = float(torch.trace(dist_s2t).item() / n)

    # MotionFix "batches of 32" protocol for table-style reporting
    if n >= batch_eval_size:
        rng = np.random.RandomState(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        num_batches = n // batch_eval_size
        batch_s2g_vals = []
        batch_s2t_vals = []
        batch_mm_s2g = []
        batch_mm_s2t = []
        for i in range(num_batches):
            cur = idx[i * batch_eval_size:(i + 1) * batch_eval_size]
            sim_s2g_b = sim_s2g[np.ix_(cur, cur)]
            sim_s2t_b = sim_s2t[np.ix_(cur, cur)]
            batch_s2g_vals.append(_retrieval_r_at_k(sim_s2g_b, ks=(1, 2, 3)))
            batch_s2t_vals.append(_retrieval_r_at_k(sim_s2t_b, ks=(1, 2, 3)))
            dist_s2g_b = dist_s2g[np.ix_(cur, cur)]
            dist_s2t_b = dist_s2t[np.ix_(cur, cur)]
            batch_mm_s2g.append(float(torch.trace(dist_s2g_b).item() / batch_eval_size))
            batch_mm_s2t.append(float(torch.trace(dist_s2t_b).item() / batch_eval_size))

        s2g_for_table = {
            "R@1": float(np.mean([m["R@1"] for m in batch_s2g_vals])),
            "R@2": float(np.mean([m["R@2"] for m in batch_s2g_vals])),
            "R@3": float(np.mean([m["R@3"] for m in batch_s2g_vals])),
        }
        s2t_for_table = {
            "R@1": float(np.mean([m["R@1"] for m in batch_s2t_vals])),
            "R@2": float(np.mean([m["R@2"] for m in batch_s2t_vals])),
            "R@3": float(np.mean([m["R@3"] for m in batch_s2t_vals])),
        }
        mm_s2g_for_table = float(np.mean(batch_mm_s2g))
        mm_s2t_for_table = float(np.mean(batch_mm_s2t))
    else:
        s2g_for_table = full_s2g
        s2t_for_table = full_s2t
        mm_s2g_for_table = full_mm_s2g
        mm_s2t_for_table = full_mm_s2t

    metrics = {
        "edit_retr_Top1": float(s2g_for_table["R@1"]),
        "edit_retr_Top2": float(s2g_for_table["R@2"]),
        "edit_retr_Top3": float(s2g_for_table["R@3"]),
        "edit_retr_MM_Dist": float(mm_s2g_for_table),
    }

    pred_mu, pred_cov = calculate_activation_statistics(pred_np)
    tgt_mu, tgt_cov = calculate_activation_statistics(tgt_np)
    metrics["edit_retr_FID"] = float(calculate_frechet_distance(tgt_mu, tgt_cov, pred_mu, pred_cov))
    metrics["edit_retr_Diversity"] = float(
        calculate_diversity(pred_np, min(diversity_times, max(2, n)))
    )

    # Real reference row: source -> GT target
    metrics["edit_retr_real_Top1"] = float(s2t_for_table["R@1"])
    metrics["edit_retr_real_Top2"] = float(s2t_for_table["R@2"])
    metrics["edit_retr_real_Top3"] = float(s2t_for_table["R@3"])
    metrics["edit_retr_real_MM_Dist"] = float(mm_s2t_for_table)
    metrics["edit_retr_real_Diversity"] = float(
        calculate_diversity(tgt_np, min(diversity_times, max(2, n)))
    )

    # Keep full-set diagnostics to avoid ambiguity.
    metrics["edit_retr_full_Top1"] = float(full_s2g["R@1"])
    metrics["edit_retr_full_Top2"] = float(full_s2g["R@2"])
    metrics["edit_retr_full_Top3"] = float(full_s2g["R@3"])
    metrics["edit_retr_real_full_Top1"] = float(full_s2t["R@1"])
    metrics["edit_retr_real_full_Top2"] = float(full_s2t["R@2"])
    metrics["edit_retr_real_full_Top3"] = float(full_s2t["R@3"])
    metrics["edit_retr_full_MM_Dist"] = float(full_mm_s2g)
    metrics["edit_retr_real_full_MM_Dist"] = float(full_mm_s2t)
    return metrics


def load_cached_t2m_motion_records(resume_run_dir: str, task: str = "t2m", replication: int = 0):
    """
    Load cached T2M motion outputs from save_vis manifests.
    Returns: dict[sample_id] -> absolute .npy path
    """
    if not resume_run_dir:
        return {}
    task_dir = os.path.join(resume_run_dir, str(task))
    if not os.path.isdir(task_dir):
        return {}

    manifest_files = []
    merged_manifest = os.path.join(task_dir, "manifest.jsonl")
    if os.path.exists(merged_manifest):
        manifest_files.append(merged_manifest)
    for name in sorted(os.listdir(task_dir)):
        if name.startswith("manifest_rank") and name.endswith(".jsonl"):
            manifest_files.append(os.path.join(task_dir, name))

    if len(manifest_files) == 0:
        return {}

    replication = int(replication)
    cache = {}
    for manifest_path in manifest_files:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if rec.get("output_type") != "motion":
                        continue
                    if int(rec.get("replication", -1)) != replication:
                        continue
                    sample_id = str(rec.get("sample_id", "")).strip()
                    data_rel = rec.get("data_path", None)
                    if not sample_id or not data_rel:
                        continue
                    abs_path = os.path.join(task_dir, str(data_rel))
                    if os.path.exists(abs_path):
                        cache[sample_id] = abs_path
        except Exception:
            continue

    return cache


# ==================== Generation Input Preparation ====================

def _strip_motion_placeholders(prompt: str) -> str:
    """Remove motion placeholders and clean up extra spaces."""
    if prompt is None:
        return ""
    for placeholder in [
        "<Motion_Placeholder_s1>",
        "<Motion_Placeholder_s2>",
        "<Motion_Placeholder>",
        "<Motion_Placeholder_Masked>",
    ]:
        prompt = prompt.replace(placeholder, "")
    prompt = prompt.strip()
    while "  " in prompt:
        prompt = prompt.replace("  ", " ")
    return prompt


def prepare_t2m_input(
    captions, text_tokenizer, num_motion_tokens,
    bos_id, eos_id, bom_id, eom_id, pad_id, mot_pad_id,
    max_text_len, max_seq_len, device,
    instruction_template=None,
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
        if instruction_template is not None:
            processed_caption = instruction_template.replace("<Caption_Placeholder>", caption)
        else:
            processed_caption = caption
        text_tokens = text_tokenizer(processed_caption, add_special_tokens=False)['input_ids'][:max_text_len]
        
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
    prompt_tokens=None,
):
    """
    Prepare inputs for M2T generation.
    Format: [BOS][BOM][MOTION_TOKENS][EOM] -> generate text -> [EOS]
    """
    bs = motion_latents.shape[1]
    batch_tokens = []
    batch_positions = []
    prompt_len = len(prompt_tokens) if prompt_tokens is not None else 0
    
    for i in range(bs):
        modality_positions = torch.tensor([2, num_motion_tokens]).unsqueeze(0)
        if prompt_tokens is not None:
            seq = [bos_id, bom_id] + [mot_pad_id] * num_motion_tokens + [eom_id] + prompt_tokens
        else:
            seq = [bos_id, bom_id] + [mot_pad_id] * num_motion_tokens + [eom_id]
        pad_len = max_seq_len - len(seq)
        seq = seq + [pad_id] * pad_len
        
        batch_tokens.append(torch.tensor(seq[:max_seq_len]))
        batch_positions.append(modality_positions)
    
    return (
        torch.stack(batch_tokens).to(device),
        torch.stack(batch_positions).to(device),
        prompt_len,
    )


def prepare_pred_input(
    source_motion, target_motion,
    num_source_tokens, num_target_tokens,
    bos_id, eos_id, bom_id, eom_id, pad_id, mot_pad_id,
    max_seq_len, device, prompt_tokens=None, add_time_embeds=True,
):
    """
    Prepare inputs for Motion Prediction generation.
    Sequence layout: [BOS] "Predict motion:" [BOM][SOURCE_MOT*N1][EOM] [BOM][TARGET_MOT*N2][EOM][EOS]
    """
    bs = source_motion.shape[0] if source_motion.dim() == 3 else 1
    
    batch_tokens = []
    batch_source_positions = []
    batch_target_positions = []
    
    if prompt_tokens is None:
        prompt_tokens = []
    
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

def smooth_motion_latents(samples: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Apply 1-D temporal Gaussian smoothing to the motion latent sequence to reduce temporal jitter in the generated results.

    Args:
        samples: [T, bs, latent_dim]  latent sequence after ODE sampling
        sigma:   Gaussian kernel std (in frames). sigma=0 means no smoothing (returns the original tensor).

    Returns:
        Smoothed [T, bs, latent_dim] tensor (same dtype/device as the input)
    """
    if sigma <= 0.0 or samples.shape[0] <= 2:
        return samples
    arr = samples.cpu().float().numpy()          # [T, bs, dim]
    arr = gaussian_filter1d(arr, sigma=sigma, axis=0)
    return torch.from_numpy(arr).to(device=samples.device, dtype=samples.dtype)


def smooth_motion_sequence(motion: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Apply temporal Gaussian smoothing to the decoded motion sequence.

    Args:
        motion: [T, D] or [B, T, D]
        sigma: Gaussian kernel std. Returns immediately when sigma<=0.
    """
    if sigma <= 0.0:
        return motion

    squeeze_batch = False
    if motion.dim() == 2:
        motion = motion.unsqueeze(0)
        squeeze_batch = True

    if motion.dim() != 3 or motion.shape[1] <= 2:
        return motion[0] if squeeze_batch else motion

    arr = motion.detach().cpu().float().numpy()
    arr = gaussian_filter1d(arr, sigma=sigma, axis=1)
    smoothed = torch.from_numpy(arr).to(device=motion.device, dtype=motion.dtype)
    return smoothed[0] if squeeze_batch else smoothed


@torch.no_grad()
def generate_t2m(
    model, motion_vae, sampler, text_tokens, modality_positions,
    latent_len, motion_latent_dim, motion_lengths, max_seq_len,
    guidance_scale, num_inference_steps, config, device, weight_type,
    latent_smooth_sigma: float = 0.0,
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

    # Temporal Gaussian smoothing in latent space to suppress frame jitter (skipped when sigma=0)
    samples = smooth_motion_latents(samples, latent_smooth_sigma)

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
    prompt_len=0,
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
        valid_len = offset + length + 1 + prompt_len  # +1 for EOM + prompt_len
        
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
            print(f"    [M2T Debug] valid_len: {valid_len} (prompt_len={prompt_len})")
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
    pred_prompt_tokens=None,
    debug=False,
    latent_smooth_sigma: float = 0.0,
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
        source_motion, target_motion,
        num_source_tokens, num_target_tokens,
        bos_id, eos_id, bom_id, eom_id, pad_id, mot_pad_id,
        max_seq_len, device, prompt_tokens=pred_prompt_tokens, add_time_embeds=add_time_embeds
    )
    
    z = torch.randn((max_target_latent_len, bs, motion_latent_dim)).to(dtype).to(device)

    # pred_generate requires (cond + uncond) concatenated input when guidance_scale > 1
    source_latents_infer = source_latents.to(dtype)
    source_positions_infer = source_positions
    target_positions_infer = target_positions
    text_tokens_infer = text_tokens
    source_lengths_infer = source_lengths
    target_lengths_infer = target_lengths

    if guidance_scale > 1.0:
        z = torch.cat([z, z], dim=1)
        source_latents_infer = torch.cat([source_latents_infer, source_latents_infer], dim=1)
        source_positions_infer = torch.cat([source_positions, source_positions], dim=0)
        target_positions_infer = torch.cat([target_positions, target_positions], dim=0)
        text_tokens_infer = torch.cat([text_tokens, text_tokens], dim=0)
        source_lengths_infer = source_lengths + source_lengths
        target_lengths_infer = target_lengths + target_lengths

    combined_positions = torch.cat([source_positions_infer, target_positions_infer], dim=1)
    block_mask = omni_attn_mask_naive(
        text_tokens_infer.size(0), text_tokens_infer.size(1), combined_positions, device
    ).to(dtype)

    model_kwargs = dict(
        text_tokens=text_tokens_infer,
        attention_mask=block_mask,
        source_motion_latents=source_latents_infer,
        source_modality_positions=source_positions_infer,
        target_modality_positions=target_positions_infer,
        source_motion_lengths=source_lengths_infer,
        target_motion_lengths=target_lengths_infer,
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
    if guidance_scale > 1.0:
        samples = samples[:, :bs, :]

    # Temporal Gaussian smoothing in latent space to suppress frame jitter (skipped when sigma=0)
    samples = smooth_motion_latents(samples, latent_smooth_sigma)

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


@torch.no_grad()
def generate_edit(
    model, motion_vae, sampler,
    source_motion, target_motion,
    source_lengths, target_lengths,
    text_tokens, source_positions, target_positions,
    config, device, weight_type,
    max_seq_len, downsample_rate, motion_latent_dim,
    guidance_scale=1.0,
    num_inference_steps=50,
    debug=False,
    latent_smooth_sigma: float = 0.0,
):
    """
    Generate Motion Edit results (source motion + edit instruction -> target motion).
    Consistent with training: reuses the forward_motion_edit / pred_generate generation paths.
    """
    dtype = weight_type

    target_latent_lens = latent_lens_from_motion_lengths(target_lengths, downsample_rate)
    max_target_latent_len = max(target_latent_lens)

    with torch.autocast(device_type='cuda', enabled=False):
        source_latents, _ = motion_vae.float().encode_motion(
            source_motion.float(),
            source_lengths
        )

    z = torch.randn((max_target_latent_len, source_motion.shape[0], motion_latent_dim)).to(dtype).to(device)

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

    # Temporal Gaussian smoothing in latent space to suppress frame jitter (skipped when sigma=0)
    samples = smooth_motion_latents(samples, latent_smooth_sigma)

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
        print(f"    [EDIT Debug] source_motion shape: {source_motion.shape}")
        print(f"    [EDIT Debug] source_latents shape: {source_latents.shape}")
        print(f"    [EDIT Debug] target z shape: {z.shape}")
        print(f"    [EDIT Debug] samples shape: {samples.shape}")
        print(f"    [EDIT Debug] decoded_motions shape: {decoded_motions.shape}")
        print(f"    [EDIT Debug] source_lengths: {source_lengths[:3]}")
        print(f"    [EDIT Debug] target_lengths: {target_lengths[:3]}")

    return decoded_motions, target_lengths


def _strip_image_placeholder(prompt: str) -> str:
    if prompt is None:
        return ""
    prompt = prompt.replace("<Image_Placeholder>", "").strip()
    while "  " in prompt:
        prompt = prompt.replace("  ", " ")
    return prompt


def _strip_mgie_placeholders(prompt: str) -> str:
    if prompt is None:
        return ""
    prompt = prompt.replace("<Source_Image_Placeholder>", "")
    prompt = prompt.replace("<Reference_Motion_Placeholder>", "")
    prompt = prompt.replace("<Image_Placeholder>", "")
    prompt = prompt.strip()
    while "  " in prompt:
        prompt = prompt.replace("  ", " ")
    return prompt


def prepare_mgie_input(
    prompts,
    text_tokenizer,
    num_source_image_tokens,
    num_ref_motion_tokens,
    num_target_image_tokens,
    bos_id,
    eos_id,
    boi_id,
    eoi_id,
    bom_id,
    eom_id,
    pad_id,
    img_pad_id,
    mot_pad_id,
    max_seq_len,
    max_text_len,
    device,
):
    batch_tokens = []
    source_image_positions = []
    ref_motion_positions = []
    target_image_positions = []

    for prompt in prompts:
        prompt = _strip_mgie_placeholders(prompt)
        instr_tokens = text_tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_text_len,
        )["input_ids"]

        src_img_offset = len(instr_tokens) + 2
        ref_motion_offset = src_img_offset + num_source_image_tokens + 2
        tgt_img_offset = ref_motion_offset + num_ref_motion_tokens + 2

        source_image_positions.append(torch.tensor([[src_img_offset, num_source_image_tokens]]))
        ref_motion_positions.append(torch.tensor([[ref_motion_offset, num_ref_motion_tokens]]))
        target_image_positions.append(torch.tensor([[tgt_img_offset, num_target_image_tokens]]))

        seq = (
            [bos_id] + instr_tokens +
            [boi_id] + [img_pad_id] * num_source_image_tokens + [eoi_id] +
            [bom_id] + [mot_pad_id] * num_ref_motion_tokens + [eom_id] +
            [boi_id] + [img_pad_id] * num_target_image_tokens + [eoi_id] +
            [eos_id]
        )
        seq = seq[:max_seq_len] + [pad_id] * max(max_seq_len - len(seq), 0)
        batch_tokens.append(torch.tensor(seq[:max_seq_len]))

    return (
        torch.stack(batch_tokens).to(device),
        torch.stack(source_image_positions).to(device),
        torch.stack(ref_motion_positions).to(device),
        torch.stack(target_image_positions).to(device),
    )


@torch.no_grad()
def generate_mgie_image(
    model,
    image_vae,
    motion_vae,
    sampler,
    source_images,
    ref_motions_norm,
    prompts,
    text_tokenizer,
    config,
    showo_token_ids,
    max_seq_len,
    motion_latent_dim,
    device,
    weight_type,
    guidance_scale=3.0,
    num_inference_steps=50,
):
    bs = source_images.shape[0]
    num_image_tokens = int(config.dataset.get("num_image_tokens", 729))
    num_ref_motion_tokens = 2 if bool(config.model.showo.add_time_embeds) else 1
    max_text_len = int(config.dataset.preprocessing.get("max_text_len", 256))

    bos_id = showo_token_ids["bos_id"]
    eos_id = showo_token_ids["eos_id"]
    boi_id = showo_token_ids["boi_id"]
    eoi_id = showo_token_ids["eoi_id"]
    bom_id = showo_token_ids["bom_id"]
    eom_id = showo_token_ids["eom_id"]
    img_pad_id = showo_token_ids["img_pad_id"]
    mot_pad_id = showo_token_ids["mot_pad_id"]
    pad_id = text_tokenizer.pad_token_id

    (
        text_tokens,
        source_positions,
        ref_positions,
        target_positions,
    ) = prepare_mgie_input(
        prompts=prompts,
        text_tokenizer=text_tokenizer,
        num_source_image_tokens=num_image_tokens,
        num_ref_motion_tokens=num_ref_motion_tokens,
        num_target_image_tokens=num_image_tokens,
        bos_id=bos_id,
        eos_id=eos_id,
        boi_id=boi_id,
        eoi_id=eoi_id,
        bom_id=bom_id,
        eom_id=eom_id,
        pad_id=pad_id,
        img_pad_id=img_pad_id,
        mot_pad_id=mot_pad_id,
        max_seq_len=max_seq_len,
        max_text_len=max_text_len,
        device=device,
    )

    with torch.autocast(device_type="cuda", enabled=False):
        source_images_5d = source_images.unsqueeze(2)
        source_image_latents = image_vae.sample(source_images_5d.to(weight_type), deterministic=True).squeeze(2)

        ref_motion = ref_motions_norm.unsqueeze(1).float()
        ref_motion_latents, _ = motion_vae.float().encode_motion(ref_motion, [1] * bs)
        if num_ref_motion_tokens == 2:
            zero_slot = torch.zeros_like(ref_motion_latents[:1])
            ref_motion_latents = torch.cat([zero_slot, ref_motion_latents], dim=0)
        ref_motion_latents = ref_motion_latents.to(weight_type)

    latent_h = int(config.dataset.image_resolution // 8)
    latent_w = int(config.dataset.image_resolution // 8)
    z = torch.randn((bs, 16, latent_h, latent_w), device=device, dtype=weight_type)

    text_tokens_infer = text_tokens
    source_positions_infer = source_positions
    ref_positions_infer = ref_positions
    target_positions_infer = target_positions
    source_latents_infer = source_image_latents
    source_pose_images_infer = source_images.to(weight_type)
    ref_latents_infer = ref_motion_latents
    z_infer = z

    if guidance_scale > 0:
        text_tokens_infer = torch.cat([text_tokens, text_tokens], dim=0)
        source_positions_infer = torch.cat([source_positions, source_positions], dim=0)
        ref_positions_infer = torch.cat([ref_positions, ref_positions], dim=0)
        target_positions_infer = torch.cat([target_positions, target_positions], dim=0)

        source_latents_uncond = torch.zeros_like(source_image_latents)
        source_latents_infer = torch.cat([source_image_latents, source_latents_uncond], dim=0)
        source_pose_images_uncond = torch.zeros_like(source_pose_images_infer)
        source_pose_images_infer = torch.cat([source_pose_images_infer, source_pose_images_uncond], dim=0)

        ref_latents_uncond = torch.zeros_like(ref_motion_latents)
        ref_latents_infer = torch.cat([ref_motion_latents, ref_latents_uncond], dim=1)

        z_infer = torch.cat([z, z], dim=0)

    combined_positions = torch.cat(
        [source_positions_infer, ref_positions_infer, target_positions_infer], dim=1
    )
    block_mask = omni_attn_mask_naive(
        text_tokens_infer.size(0), text_tokens_infer.size(1), combined_positions, device
    ).to(weight_type)

    model_kwargs = dict(
        text_tokens=text_tokens_infer,
        attention_mask=block_mask,
        source_image_latents=source_latents_infer,
        source_pose_images=source_pose_images_infer,
        ref_motion_latents=ref_latents_infer,
        source_image_modality_positions=source_positions_infer,
        ref_motion_modality_positions=ref_positions_infer,
        target_image_modality_positions=target_positions_infer,
        max_seq_len=max_seq_len,
        guidance_scale=guidance_scale,
    )

    sample_fn = sampler.sample_ode(
        sampling_method=config.transport.sampling_method,
        num_steps=num_inference_steps,
        atol=config.transport.atol,
        rtol=config.transport.rtol,
        reverse=config.transport.reverse,
        time_shifting_factor=config.transport.time_shifting_factor,
    )
    samples = sample_fn(z_infer, model.mgie_generate, **model_kwargs)[-1]
    if guidance_scale > 0:
        samples = samples[:bs]

    with torch.autocast(device_type="cuda", enabled=False):
        decoded = image_vae.batch_decode(samples.unsqueeze(2)).squeeze(2).float()
    return decoded


def prepare_t2t_input(
    input_texts,
    text_tokenizer,
    bos_id,
    pad_id,
    max_seq_len,
    max_input_tokens,
    device,
    prompt_template="",
):
    """Build T2T inference inputs: [BOS] input [PAD...]."""
    batch_tokens = []
    valid_lens = []

    for text in input_texts:
        if prompt_template:
            if "<Input_Placeholder>" in prompt_template:
                text = prompt_template.replace("<Input_Placeholder>", text)
            else:
                text = f"{prompt_template} {text}".strip()
        tok = text_tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_input_tokens,
        )["input_ids"]
        seq = [bos_id] + tok
        valid_lens.append(min(len(seq), max_seq_len))
        seq = seq[:max_seq_len] + [pad_id] * max(max_seq_len - len(seq), 0)
        batch_tokens.append(torch.tensor(seq[:max_seq_len]))

    return torch.stack(batch_tokens).to(device), valid_lens


@torch.no_grad()
def generate_t2t_texts(
    model,
    text_tokenizer,
    text_tokens,
    valid_lens,
    max_new_tokens,
    eos_token,
    device,
    weight_type,
    do_sample=False,
    temperature=1.0,
    top_k=None,
):
    """T2T generation (explicitly passes an all-zero modality_mask to enable LoRA-A)."""
    input_embeds = model.showo.model.embed_tokens(text_tokens)
    bs = input_embeds.shape[0]
    generated_texts = []

    modality_mask = torch.zeros(
        bs,
        input_embeds.shape[1],
        device=device,
        dtype=weight_type,
    )

    for i in range(bs):
        valid_len = int(valid_lens[i])
        embeds_valid = input_embeds[i:i+1, :valid_len, :]
        mask_valid = modality_mask[i:i+1, :valid_len]
        causal_mask = torch.tril(
            torch.ones(valid_len, valid_len, device=device)
        ).unsqueeze(0).unsqueeze(0).to(weight_type)

        result = model.m2t_generate(
            input_embeds=embeds_valid,
            attention_mask=causal_mask,
            modality_mask=mask_valid,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            eos_token=eos_token,
        )
        token_ids = [t.item() for t in result]
        generated_texts.append(text_tokenizer.decode(token_ids, skip_special_tokens=True))

    return generated_texts


def prepare_img2pose_input(
    prompts,
    text_tokenizer,
    num_image_tokens,
    num_motion_tokens,
    bos_id,
    eos_id,
    boi_id,
    eoi_id,
    bom_id,
    eom_id,
    pad_id,
    img_pad_id,
    mot_pad_id,
    max_seq_len,
    max_text_len,
    device,
):
    """Build img2pose inference inputs."""
    batch_tokens = []
    image_positions = []
    motion_positions = []

    for prompt in prompts:
        prompt = _strip_image_placeholder(prompt)
        instr_tokens = text_tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_text_len,
        )["input_ids"]

        img_offset = len(instr_tokens) + 2
        mot_offset = img_offset + num_image_tokens + 2

        image_positions.append(torch.tensor([[img_offset, num_image_tokens]]))
        motion_positions.append(torch.tensor([[mot_offset, num_motion_tokens]]))

        seq = (
            [bos_id] + instr_tokens +
            [boi_id] + [img_pad_id] * num_image_tokens + [eoi_id] +
            [bom_id] + [mot_pad_id] * num_motion_tokens + [eom_id] + [eos_id]
        )
        seq = seq[:max_seq_len] + [pad_id] * max(max_seq_len - len(seq), 0)
        batch_tokens.append(torch.tensor(seq[:max_seq_len]))

    return (
        torch.stack(batch_tokens).to(device),
        torch.stack(image_positions).to(device),
        torch.stack(motion_positions).to(device),
    )


@torch.no_grad()
def generate_img2pose(
    model,
    image_vae,
    motion_vae,
    sampler,
    images,
    prompts,
    text_tokenizer,
    config,
    showo_token_ids,
    max_seq_len,
    motion_latent_dim,
    device,
    weight_type,
    guidance_scale=5.0,
    num_inference_steps=50,
):
    """Sample img2pose and decode to a 269-dim single-frame motion (normalized space)."""
    bs = images.shape[0]
    add_time_embeds = bool(config.model.showo.add_time_embeds)
    num_motion_tokens = 2 if add_time_embeds else 1
    num_image_tokens = int(config.dataset.get("num_image_tokens", 729))
    max_text_len = int(config.dataset.preprocessing.get("max_text_len", 256))

    bos_id = showo_token_ids["bos_id"]
    eos_id = showo_token_ids["eos_id"]
    boi_id = showo_token_ids["boi_id"]
    eoi_id = showo_token_ids["eoi_id"]
    bom_id = showo_token_ids["bom_id"]
    eom_id = showo_token_ids["eom_id"]
    img_pad_id = showo_token_ids["img_pad_id"]
    mot_pad_id = showo_token_ids["mot_pad_id"]
    pad_id = text_tokenizer.pad_token_id

    text_tokens, image_positions, motion_positions = prepare_img2pose_input(
        prompts,
        text_tokenizer,
        num_image_tokens,
        num_motion_tokens,
        bos_id,
        eos_id,
        boi_id,
        eoi_id,
        bom_id,
        eom_id,
        pad_id,
        img_pad_id,
        mot_pad_id,
        max_seq_len,
        max_text_len,
        device,
    )

    with torch.autocast(device_type="cuda", enabled=False):
        images_5d = images.unsqueeze(2)
        image_latents = image_vae.sample(images_5d.to(weight_type), deterministic=True).squeeze(2)

    latent_len = 1
    z = torch.randn((latent_len, bs, motion_latent_dim)).to(weight_type).to(device)
    text_tokens_infer = text_tokens
    image_positions_infer = image_positions
    motion_positions_infer = motion_positions
    image_latents_infer = image_latents.to(weight_type)

    if guidance_scale > 0:
        z = torch.cat([z, z], dim=1)
        text_tokens_infer = torch.cat([text_tokens, text_tokens], dim=0)
        image_positions_infer = torch.cat([image_positions, image_positions], dim=0)
        motion_positions_infer = torch.cat([motion_positions, motion_positions], dim=0)
        # The uncond branch zeroes the image condition
        image_latents_uncond = torch.zeros_like(image_latents_infer)
        image_latents_infer = torch.cat([image_latents_infer, image_latents_uncond], dim=0)

    combined_positions = torch.cat([image_positions_infer, motion_positions_infer], dim=1)
    block_mask = omni_attn_mask_naive(
        text_tokens_infer.size(0), text_tokens_infer.size(1), combined_positions, device
    ).to(weight_type)

    model_kwargs = dict(
        text_tokens=text_tokens_infer,
        image_latents=image_latents_infer,
        attention_mask=block_mask,
        image_modality_positions=image_positions_infer,
        motion_modality_positions=motion_positions_infer,
        max_seq_len=max_seq_len,
        guidance_scale=guidance_scale,
    )

    sample_fn = sampler.sample_ode(
        sampling_method=config.transport.sampling_method,
        num_steps=num_inference_steps,
        atol=config.transport.atol,
        rtol=config.transport.rtol,
        reverse=config.transport.reverse,
        time_shifting_factor=config.transport.time_shifting_factor,
    )
    samples = sample_fn(z, model.img2pose_generate, **model_kwargs)[-1]
    if guidance_scale > 0:
        samples = samples[:, :bs, :]

    with torch.autocast(device_type="cuda", enabled=False):
        decoded_motions = motion_vae.float().decode(samples.float(), [1] * bs)

    if isinstance(decoded_motions, list):
        decoded_motions = torch.stack(decoded_motions, dim=0)
    if decoded_motions.dim() == 3:
        decoded_motions = decoded_motions[:, 0, :]

    return decoded_motions


@torch.no_grad()
def generate_img2text(
    model,
    image_vae,
    images,
    prompts,
    text_tokenizer,
    showo_token_ids,
    num_image_tokens,
    max_seq_len,
    max_text_len,
    max_new_tokens,
    eos_id,
    device,
    weight_type,
    do_sample=False,
    temperature=1.0,
    top_k=None,
):
    """img2text generation (explicitly passes an all-zero modality_mask to enable LoRA-A)."""
    bs = images.shape[0]

    bos_id = showo_token_ids["bos_id"]
    boi_id = showo_token_ids["boi_id"]
    eoi_id = showo_token_ids["eoi_id"]
    img_pad_id = showo_token_ids["img_pad_id"]
    pad_id = text_tokenizer.pad_token_id

    batch_tokens = []
    image_positions = []
    prompt_lens = []

    for prompt in prompts:
        prompt = _strip_image_placeholder(prompt)
        prompt_tokens = text_tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_text_len,
        )["input_ids"]
        prompt_lens.append(len(prompt_tokens))
        image_positions.append(torch.tensor([[2, num_image_tokens]]))

        seq = [bos_id, boi_id] + [img_pad_id] * num_image_tokens + [eoi_id] + prompt_tokens
        seq = seq[:max_seq_len] + [pad_id] * max(max_seq_len - len(seq), 0)
        batch_tokens.append(torch.tensor(seq[:max_seq_len]))

    text_tokens = torch.stack(batch_tokens).to(device)
    image_positions = torch.stack(image_positions).to(device)

    with torch.autocast(device_type="cuda", enabled=False):
        images_5d = images.unsqueeze(2)
        image_latents = image_vae.sample(images_5d.to(weight_type), deterministic=True).squeeze(2)

    input_embeds = model.showo.model.embed_tokens(text_tokens)
    image_embeds = model._encode_image_features(
        image_latents.to(weight_type),
        task_name="img2text",
        pose_images=images.to(weight_type),
    )

    for i in range(bs):
        offset, length = image_positions[i][0].tolist()
        actual_len = min(int(length), image_embeds.shape[1])
        input_embeds[i, offset:offset + actual_len] = image_embeds[i, :actual_len]

    modality_mask = torch.zeros(bs, input_embeds.shape[1], device=device, dtype=weight_type)
    generated_texts = []

    for i in range(bs):
        offset, length = image_positions[i][0].tolist()
        valid_len = int(offset + length + 1 + prompt_lens[i])  # +1 for EOI
        embeds_valid = input_embeds[i:i+1, :valid_len, :]
        mask_valid = modality_mask[i:i+1, :valid_len]
        pos_valid = image_positions[i:i+1]
        block_mask = omni_attn_mask_naive(1, valid_len, pos_valid, device).to(weight_type)

        result = model.m2t_generate(
            input_embeds=embeds_valid,
            attention_mask=block_mask,
            modality_mask=mask_valid,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            eos_token=eos_id,
        )
        token_ids = [t.item() for t in result]
        generated_texts.append(text_tokenizer.decode(token_ids, skip_special_tokens=True))

    return generated_texts


def _project_joints_to_image(joints_xyz, image_h, image_w, margin=0.08):
    """
    Fallback camera-free projection on cropped image.
    """
    feet_idx = [10, 11]
    y = joints_xyz[:, 1]
    x = joints_xyz[:, 0]
    body_height = max(float(y.max() - y.min()), 1e-6)
    target_height = (1.0 - 2.0 * margin) * image_h
    scale = target_height / body_height
    feet_y = float(y[feet_idx].mean())
    feet_x = float(x[feet_idx].mean())
    target_feet_y = (1.0 - margin) * image_h
    target_center_x = 0.5 * image_w
    px = target_center_x + (x - feet_x) * scale
    py = target_feet_y - (y - feet_y) * scale
    px = np.clip(px, 0, image_w - 1)
    py = np.clip(py, 0, image_h - 1)
    return np.stack([px, py], axis=1)


def _fit_affine_2d(src_xy, dst_xy):
    """
    Solve 2D affine map: [x y 1] @ W = [u v].
    """
    if src_xy is None or dst_xy is None:
        return None
    if src_xy.shape[0] < 3 or dst_xy.shape[0] < 3:
        return None
    A = np.concatenate([src_xy, np.ones((src_xy.shape[0], 1), dtype=np.float32)], axis=1)
    W, _, _, _ = np.linalg.lstsq(A, dst_xy, rcond=None)
    return W


def _apply_affine_2d(xy, W):
    if W is None:
        return None
    A = np.concatenate([xy, np.ones((xy.shape[0], 1), dtype=np.float32)], axis=1)
    return A @ W


def _parse_h36m_rel_path(image_rel_path):
    """
    Parse:
      S9/S9_Directions_1.54138969/S9_Directions_1.54138969_000001.jpg
    """
    parts = image_rel_path.split("/")
    if len(parts) < 3:
        return None
    subject = parts[0]
    dir_name = parts[1]
    file_name = parts[2]

    dir_pat = rf"^{re.escape(subject)}_(.+)_(\d+)\.(\d+)$"
    m = re.match(dir_pat, dir_name)
    if not m:
        return None
    action = m.group(1)
    trial = m.group(2)
    cam = m.group(3)

    fm = re.search(r"_(\d+)\.jpg$", file_name)
    if not fm:
        return None
    frame_num = int(fm.group(1))
    return {
        "subject": subject,
        "action": action,
        "trial": trial,
        "cam": cam,
        "frame_num": frame_num,
    }


def _resolve_h36m_d2_cdf_path(parsed, extracted_root):
    if parsed is None or not extracted_root:
        return None
    base = os.path.join(parsed["subject"], "MyPoseFeatures", "D2_Positions")
    candidates = [
        f"{parsed['action']} {parsed['trial']}.{parsed['cam']}.cdf",
        f"{parsed['action']}.{parsed['cam']}.cdf",
    ]
    for c in candidates:
        p = os.path.join(extracted_root, base, c)
        if os.path.exists(p):
            return p
    return None


def _load_h36m_d2_frame(image_rel_path, extracted_root, cdf_cache):
    parsed = _parse_h36m_rel_path(image_rel_path)
    cdf_path = _resolve_h36m_d2_cdf_path(parsed, extracted_root)
    if cdf_path is None:
        return None

    if cdf_path not in cdf_cache:
        try:
            import cdflib
            cdf = cdflib.CDF(cdf_path)
            pose = cdf.varget("Pose")
            cdf_cache[cdf_path] = pose
        except Exception:
            cdf_cache[cdf_path] = None
    pose = cdf_cache[cdf_path]
    if pose is None:
        return None

    # Pose shape: [1, num_frames, 64], 64 = 32 joints * 2
    num_frames = int(pose.shape[1])
    frame_num = int(parsed["frame_num"])
    frame_idx = max(0, min(frame_num - 1, num_frames - 1))
    xy = pose[0, frame_idx].reshape(-1, 2).astype(np.float32)
    return xy


def _fit_hml22_to_h36m_d2(joints_xyz, d2_xy):
    """
    Fit HumanML3D 22 joints to H36M D2 joints in image plane via affine map.
    """
    if d2_xy is None or d2_xy.shape[0] < 17:
        return None
    # HumanML3D-22 idx -> H36M-32 idx (commonly used 17 body joints).
    pairs = [
        (0, 0),   # root
        (1, 1),   # right hip
        (2, 4),   # left hip
        (3, 7),   # spine
        (6, 8),   # thorax
        (9, 9),   # neck
        (15, 10), # head
        (4, 2),   # right knee
        (5, 5),   # left knee
        (7, 3),   # right ankle
        (8, 6),   # left ankle
        (16, 14), # right shoulder
        (17, 11), # left shoulder
        (18, 15), # right elbow
        (19, 12), # left elbow
        (20, 16), # right wrist
        (21, 13), # left wrist
    ]
    src_xy_all = joints_xyz[:, [0, 1]].astype(np.float32)
    src_xy = np.stack([src_xy_all[i] for i, _ in pairs], axis=0)
    dst_xy = np.stack([d2_xy[j] for _, j in pairs], axis=0)
    W = _fit_affine_2d(src_xy, dst_xy)
    return _apply_affine_2d(src_xy_all, W)


def save_img2pose_alignment_visualizations(
    samples,
    evaluator,
    out_dir,
    h36m_full_image_root=None,
    h36m_extracted_root=None,
    use_h36m_d2=True,
):
    os.makedirs(out_dir, exist_ok=True)
    # HumanML3D canonical kinematic tree (same as MotionGPT3).
    chains_hml22 = [
        [0, 3, 6, 9, 12, 15],   # body
        [9, 14, 17, 19, 21],    # right arm
        [9, 13, 16, 18, 20],    # left arm
        [0, 2, 5, 8, 11],       # right leg
        [0, 1, 4, 7, 10],       # left leg
    ]
    chains_h36m17 = [
        [0, 7, 8, 9, 10],  # body
        [8, 14, 15, 16],   # right arm
        [8, 11, 12, 13],   # left arm
        [0, 1, 2, 3],      # right leg
        [0, 4, 5, 6],      # left leg
    ]

    meta = []
    cdf_cache = {}
    for i, sample in enumerate(samples):
        img_t = sample["image"].cpu().float()
        img_cropped = (img_t * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()

        pred_norm = sample["gen_motion_norm"].unsqueeze(0).to(evaluator.device)
        gt_norm = sample["gt_motion_norm"].unsqueeze(0).to(evaluator.device)
        pred_denorm = evaluator._denormalize(pred_norm)
        gt_denorm = evaluator._denormalize(gt_norm)
        pred_joints = evaluator._extract_joints(pred_denorm)[0].detach().cpu().numpy()
        gt_joints = evaluator._extract_joints(gt_denorm)[0].detach().cpu().numpy()

        image_rel_path = sample.get("image_path", "")
        d2_xy = None
        img = img_cropped
        panel1_title = "Input Image (Cropped)"
        if use_h36m_d2 and h36m_extracted_root:
            d2_xy = _load_h36m_d2_frame(image_rel_path, h36m_extracted_root, cdf_cache)
            if d2_xy is not None and h36m_full_image_root:
                full_path = os.path.join(h36m_full_image_root, image_rel_path)
                if os.path.exists(full_path):
                    img = np.asarray(Image.open(full_path).convert("RGB")).astype(np.float32) / 255.0
                    panel1_title = "Input Image (Full)"
        h, w = img.shape[:2]

        if d2_xy is not None and img.shape[0] > 0:
            pred_2d = _fit_hml22_to_h36m_d2(pred_joints, d2_xy)
            if pred_2d is None:
                pred_2d = _project_joints_to_image(pred_joints, h, w)
            else:
                pred_2d[:, 0] = np.clip(pred_2d[:, 0], 0, w - 1)
                pred_2d[:, 1] = np.clip(pred_2d[:, 1], 0, h - 1)
            gt_2d = d2_xy
            pred_title = "Pred (Affine->GT2D)"
            gt_title = "GT (H36M D2)"
            gt_chains = chains_h36m17
            gt_pts = gt_2d[:17] if gt_2d.shape[0] >= 17 else gt_2d
        else:
            pred_2d = _project_joints_to_image(pred_joints, h, w)
            gt_2d = _project_joints_to_image(gt_joints, h, w)
            pred_title = "Pred (Fitted)"
            gt_title = "GT (Fitted)"
            gt_chains = chains_hml22
            gt_pts = gt_2d

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img)
        axes[0].set_title(panel1_title)
        axes[0].axis("off")

        axes[1].imshow(img)
        for chain in chains_hml22:
            axes[1].plot(pred_2d[chain, 0], pred_2d[chain, 1], c="lime", linewidth=2)
        axes[1].scatter(pred_2d[:, 0], pred_2d[:, 1], c="yellow", s=8)
        axes[1].set_title(pred_title)
        axes[1].axis("off")

        axes[2].imshow(img)
        for chain in gt_chains:
            if max(chain) < gt_2d.shape[0]:
                axes[2].plot(gt_2d[chain, 0], gt_2d[chain, 1], c="cyan", linewidth=2)
        axes[2].scatter(gt_pts[:, 0], gt_pts[:, 1], c="orange", s=8)
        axes[2].set_title(gt_title)
        axes[2].axis("off")

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"sample_{i:02d}.png")
        fig.savefig(out_path, dpi=180)
        plt.close(fig)

        meta.append(
            {
                "index": i,
                "image_path": sample.get("image_path", ""),
                "use_h36m_d2": bool(d2_xy is not None),
                "figure": out_path,
            }
        )

    with open(os.path.join(out_dir, "samples_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


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
    tasks = config.test.get('tasks', ['t2m', 'm2t', 'pred', 't2t', 'edit'])
    if isinstance(tasks, str):
        tasks = [tasks]
    excluded_requested = [t for t in tasks if t in RELEASE_EXCLUDED_TASKS]
    if excluded_requested:
        raise NotImplementedError(
            "The image-to-motion / MGIE evaluation suite is not included in this release. "
            f"Requested unavailable tasks: {excluded_requested}. "
            "Available tasks: ['t2m', 'm2t', 'pred', 't2t', 'edit']."
        )
    auto_profile_summary = apply_stage4_auto_profile(config, tasks)
    eval_title = config.test.get('eval_title', "UniMotion Stage 4 Unified Evaluation")
    eval_stage_label = config.test.get('eval_stage_label', "STAGE 4")
    eval_result_prefix = config.test.get('eval_result_prefix', "stage4_eval")
    
    # Quick test mode
    quick_mode = config.test.get('quick', False)
    num_quick_samples = int(config.test.get('num_quick_samples', 64))
    quick_subset_seed = int(config.test.get('quick_subset_seed', seed))
    # Per-task replication times: text tasks default to multiple runs, others to 1
    global_rep = config.test.get('replication_times', None)  # global override (highest priority)
    _task_defaults = {
        't2m': 1,
        'm2t': 3,
        'pred': 1,
        't2t': 5,
        'edit': 1,
        'img2pose': 1,
        'img2text': 5,
        'mgie': 1,
    }
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
        print(eval_title)
        print("Motion Features: 269-dim, Downsample Rate: 1, LoRA-A+B")
        print("=" * 70)
        print(f"Tasks: {list(tasks)}")
        if auto_profile_summary is not None:
            print(
                f"Auto profile: {auto_profile_summary.get('profile')} "
                f"(image_tasks={auto_profile_summary.get('image_tasks', [])})"
            )
            if auto_profile_summary.get("warning", None) is not None:
                print(f"  [WARN] {auto_profile_summary['warning']}")
            for line in auto_profile_summary.get("changes", []):
                print(f"  - {line}")
        print(f"Quick mode: {quick_mode}" + (f" ({num_quick_samples} samples)" if quick_mode else ""))
        if quick_mode:
            print(f"Quick subset seed: {quick_subset_seed}")
        print(f"Replication times per task: { {t: default_rep_per_task.get(t, 1) for t in tasks} }")
        print(f"Distributed: {is_distributed}, World size: {world_size}")
        print(f"Device: {device}, Weight type: {weight_type}")
    
    # Create output directory
    output_dir = config.test.get('output_dir', f'outputs/{eval_result_prefix}')
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

    need_image_tasks = ('img2pose' in tasks) or ('img2text' in tasks) or ('mgie' in tasks)
    image_vae = None
    if need_image_tasks:
        vae_path = config.model.vae_model.pretrained_model_path
        image_vae = WanVAE(
            vae_pth=vae_path,
            dtype=weight_type,
            device=device,
        )
        if is_main_process():
            print(f"  WAN21 VAE loaded from {vae_path}")
    
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
    save_t2m_sigma_smoothed_motion = bool(save_vis_cfg.get('save_t2m_sigma_smoothed_motion', False))
    t2m_sigma_smoothed_sigma = float(
        save_vis_cfg.get(
            't2m_sigma_smoothed_sigma',
            config.test.get('t2m', {}).get('latent_smooth_sigma', 0.0),
        )
    )
    t2m_sigma_smoothed_task_name = str(
        save_vis_cfg.get('t2m_sigma_smoothed_task_name', 't2m_sigma_smoothed_postdecode')
    )
    enable_t2m_sigma_smoothed_save = (
        save_t2m_sigma_smoothed_motion and t2m_sigma_smoothed_sigma > 0.0
    )
    _default_script_tag = os.path.splitext(os.path.basename(__file__))[0]
    _script_tag = save_vis_cfg.get('script_tag', _default_script_tag) or _default_script_tag
    save_vis_tasks = list(tasks)
    if enable_t2m_sigma_smoothed_save and t2m_sigma_smoothed_task_name not in save_vis_tasks:
        save_vis_tasks.append(t2m_sigma_smoothed_task_name)
    save_vis_manager = SaveVisManager(
        enabled=bool(save_vis_cfg.get('enabled', False)),
        root_dir=save_vis_cfg.get('root_dir', DEFAULT_SAVE_VIS_ROOT),
        script_tag=_script_tag,
        tasks=save_vis_tasks,
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
    resume_t2m_from_save_vis_dir = config.test.get('resume_t2m_from_save_vis_dir', None)
    resume_t2m_replication = int(config.test.get('resume_t2m_replication', 0))
    cached_t2m_motion_paths = {}
    if isinstance(resume_t2m_from_save_vis_dir, str) and resume_t2m_from_save_vis_dir.strip():
        cached_t2m_motion_paths = load_cached_t2m_motion_records(
            resume_t2m_from_save_vis_dir.strip(),
            task='t2m',
            replication=resume_t2m_replication,
        )
        if is_main_process():
            print(
                f"  Resume T2M cache: {len(cached_t2m_motion_paths)} samples "
                f"(dir={resume_t2m_from_save_vis_dir}, rep={resume_t2m_replication})"
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
            'enable_A': lora_config.get('enable_A', True),   # Text/Image path
            'enable_B': lora_config.get('enable_B', True),   # Motion path (for chained load/save)
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
        lora_info = model.load_lora(
            lora_weights_path,
            strict=False,
            preserve_unmatched=True,
        )
        if is_main_process():
            print(
                f"    LoRA loaded={len(lora_info.get('loaded_keys', []))}, "
                f"unmatched={len(lora_info.get('missing_keys_in_model', []))}, "
                f"missing_in_ckpt={len(lora_info.get('missing_keys_in_state', []))}"
            )
    
    model.to(weight_type)
    model.eval()
    
    # Transport sampler
    # Consistent with training: when add_time_embeds=True, seq_len needs +1
    transport_seq_len = config.dataset.preprocessing.max_motion_tokens
    if 'mgie' in tasks:
        transport_seq_len = max(transport_seq_len, int(config.dataset.get('num_image_tokens', 729)))
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
    edit_embedder = None
    edit_text_embedder = None
    text_nlg_evaluator = None
    mgie_pose_evaluator = None
    
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
        if is_main_process():
            print("  Pred evaluator loaded")

    if 'edit' in tasks:
        edit_predict_ratio = config.test.get('edit', {}).get('predict_ratio', 0.0)
        evaluators['edit'] = PredEvaluator(
            njoints=22,
            predict_ratio=edit_predict_ratio,
            task='pred',
            device=device,
        )
        # Reuse T2M motion encoder for retrieval-style embeddings.
        edit_embedder = T2MEvaluator(
            t2m_evaluator_path=t2m_evaluator_path,
            glove_dir=glove_dir,
            dataname='t2m',
            device=device,
            unit_length=eval_unit_length,
        )
        # Used to extract text embeddings from raw sentences (edit retrieval metrics)
        if 'm2t' in evaluators:
            edit_text_embedder = evaluators['m2t']
        else:
            edit_text_embedder = M2TEvaluator(
                t2m_evaluator_path=t2m_evaluator_path,
                glove_dir=glove_dir,
                dataname='t2m',
                device=device,
                unit_length=eval_unit_length,
            )
        if is_main_process():
            print("  Edit evaluator loaded (ADE/FDE + MotionFix-style retrieval)")

    if ('t2t' in tasks) or ('img2text' in tasks):
        text_nlg_evaluator = TextNLGEvaluator(device=device)
        if is_main_process():
            print("  Text NLG evaluator loaded (BLEU/ROUGE/CIDEr/METEOR/BertScore)")

    if 'img2pose' in tasks:
        # Lazy import: this evaluator is not shipped in this release.
        from evaluation.img2pose_evaluator import Img2PoseEvaluator
        h36m_root = config.dataset.h36m.data_root
        h36m_mean = np.load(os.path.join(h36m_root, 'Mean.npy'))
        h36m_std = np.load(os.path.join(h36m_root, 'Std.npy'))
        evaluators['img2pose'] = Img2PoseEvaluator(
            mean=h36m_mean,
            std=h36m_std,
            joints_num=22,
            device=device,
        )
        if is_main_process():
            print("  Img2Pose evaluator loaded (MPJPE / PA-MPJPE)")

    if 'mgie' in tasks:
        # Lazy import: these evaluators are not shipped in this release.
        from evaluation.img2pose_evaluator import Img2PoseEvaluator
        from evaluation.mgie_evaluator import MGIEEvaluator
        mgie_cfg = config.test.get('mgie', {})
        evaluators['mgie'] = MGIEEvaluator(
            device=device,
            lpips_net=mgie_cfg.get('lpips_net', 'alex'),
            clip_model_name=mgie_cfg.get('clip_model_name', 'ViT-B/32'),
        )
        # Pose-proxy metric: generated image -> img2pose -> compare to target pose.
        if mgie_pose_evaluator is None:
            h36m_root = config.dataset.h36m.data_root
            h36m_mean = np.load(os.path.join(h36m_root, 'Mean.npy'))
            h36m_std = np.load(os.path.join(h36m_root, 'Std.npy'))
            mgie_pose_evaluator = Img2PoseEvaluator(
                mean=h36m_mean,
                std=h36m_std,
                joints_num=22,
                device=device,
            )
        if is_main_process():
            print("  MGIE evaluator loaded (PSNR/SSIM/LPIPS/CLIP + PoseProxy)")
    
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

    eval_dataloader = None
    need_motion_eval = any(t in tasks for t in ['t2m', 'm2t', 'pred'])
    if need_motion_eval:
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
    edit_dataloader = None
    edit_mean = mean_train
    edit_std = std_train
    
    if is_main_process():
        print(f"  Train mean/std shape: {mean_train.shape}")

    if 'edit' in tasks:
        motion_edit_cfg = config.dataset.get('motion_edit', None)
        if motion_edit_cfg is None:
            raise ValueError("Task 'edit' is enabled but config.dataset.motion_edit is missing.")

        edit_data_root = motion_edit_cfg.get('data_root', '')
        edit_instruction_path = motion_edit_cfg.get('instruction_template_path', None)
        edit_batch_size = config.test.get('edit', {}).get('batch_size', batch_size)
        edit_max_motion_length = config.test.get('edit', {}).get('max_motion_length', max_motion_length)
        edit_min_motion_length = config.test.get('edit', {}).get('min_motion_length', min_motion_length)

        if is_main_process():
            print(f"  Edit data root: {edit_data_root}")

        edit_dataset = MotionEditDataset(
            data_root=edit_data_root,
            text_tokenizer=text_tokenizer,
            max_seq_len=config.dataset.preprocessing.max_seq_length,
            max_motion_length=edit_max_motion_length,
            min_motion_length=edit_min_motion_length,
            unit_length=1,
            downsample_rate=config.dataset.motion.downsample_rate,
            showo_token_ids=showo_token_ids,
            motion_latent_dim=config.dataset.motion.motion_latent_dim,
            split='test',
            tiny=config.test.get('tiny', False),
            add_time_embeds=config.model.showo.add_time_embeds,
            instruction_template_path=edit_instruction_path,
        )

        if quick_mode:
            subset_size = min(num_quick_samples, len(edit_dataset.name_list))
            keys = list(edit_dataset.name_list)
            indices = select_deterministic_quick_indices(keys, subset_size, quick_subset_seed)
            edit_dataset.name_list = [edit_dataset.name_list[i] for i in indices]
            if is_main_process():
                print(f"  Quick mode edit: using {subset_size} samples (deterministic random subset)")
                print(f"  Quick subset seed: {quick_subset_seed}")

        if len(edit_dataset) == 0:
            raise RuntimeError("Edit dataset is empty. Please check motion_edit data_root and split files.")

        edit_mean = edit_dataset.mean
        edit_std = edit_dataset.std

        if is_distributed:
            edit_sampler = DistributedSampler(edit_dataset, shuffle=False)
        else:
            edit_sampler = None

        edit_dataloader = DataLoader(
            edit_dataset,
            batch_size=edit_batch_size,
            shuffle=False,
            sampler=edit_sampler,
            num_workers=config.dataset.params.get('num_workers', 4),
            collate_fn=edit_dataset.collate_fn,
            drop_last=False,
            pin_memory=True,
        )

        if is_main_process():
            print(f"  Edit dataset size: {len(edit_dataset)}")
            print(f"  Edit batch size: {edit_batch_size}")
            print(f"  Edit total batches: {len(edit_dataloader)}")

    # T2T dataloader
    t2t_dataloader = None
    if 't2t' in tasks:
        t2t_cfg = config.test.get('t2t', {})
        t2t_batch_size = int(t2t_cfg.get('batch_size', batch_size))
        t2t_instruction_path = config.training.get('instruction_template_path', None)
        if t2t_instruction_path is None:
            t2t_instruction_path = "instructions/template_witht2t_instructions.json"

        t2t_dataset = TextToTextDataset(
            data_root=config.dataset.motion.data_root,
            text_tokenizer=text_tokenizer,
            max_seq_len=config.dataset.preprocessing.max_seq_length,
            max_motion_length=config.dataset.motion.max_motion_length,
            min_motion_length=config.dataset.motion.min_motion_length,
            fps=config.dataset.motion.fps,
            downsample_rate=config.dataset.motion.downsample_rate,
            showo_token_ids=showo_token_ids,
            split='test',
            tiny=config.test.get('tiny', False),
            instruction_template_path=t2t_instruction_path,
        )

        if quick_mode:
            from torch.utils.data import Subset
            subset_size = min(num_quick_samples, len(t2t_dataset))
            keys = build_t2t_entry_keys(t2t_dataset)
            idx = select_deterministic_quick_indices(keys, subset_size, quick_subset_seed)
            t2t_dataset = Subset(t2t_dataset, idx)
            if is_main_process():
                print(f"  Quick mode t2t: using {subset_size} samples (deterministic random subset)")
                print(f"  Quick subset seed: {quick_subset_seed}")

        if is_distributed:
            t2t_sampler = DistributedSampler(t2t_dataset, shuffle=False)
        else:
            t2t_sampler = None

        t2t_dataloader = DataLoader(
            t2t_dataset,
            batch_size=t2t_batch_size,
            shuffle=False,
            sampler=t2t_sampler,
            num_workers=config.dataset.params.get('num_workers', 4),
            collate_fn=t2t_dataset.dataset.collate_fn if hasattr(t2t_dataset, 'dataset') else t2t_dataset.collate_fn,
            drop_last=False,
            pin_memory=True,
        )
        if is_main_process():
            print(f"  T2T dataset size: {len(t2t_dataset)}")
            print(f"  T2T total batches: {len(t2t_dataloader)}")

    # H36M dataloaders (img2pose / img2text)
    img2pose_dataloader = None
    img2text_dataloader = None
    mgie_dataloader = None
    if ('img2pose' in tasks) or ('img2text' in tasks):
        # Lazy import: the H36M eval pipeline is not shipped in this release.
        from datasets.h36m_eval_dataset import H36MEvalDataset
        h36m_cfg = config.dataset.get('h36m', None)
        if h36m_cfg is None:
            raise ValueError("img2pose/img2text requires config.dataset.h36m")

        image_resolution = int(config.dataset.get('image_resolution', 432))

        if 'img2pose' in tasks:
            pose_cfg = config.test.get('img2pose', {})
            pose_max_samples = pose_cfg.get('max_samples', config.test.get('max_samples', None))
            if quick_mode:
                pose_max_samples = num_quick_samples if pose_max_samples is None else min(num_quick_samples, pose_max_samples)

            img2pose_dataset = H36MEvalDataset(
                data_root=h36m_cfg.data_root,
                image_root=h36m_cfg.image_root,
                split='test',
                image_resolution=image_resolution,
                max_samples=pose_max_samples,
                seed=quick_subset_seed if quick_mode else seed,
                require_text=False,
            )
            if is_distributed:
                img2pose_sampler = DistributedSampler(img2pose_dataset, shuffle=False)
            else:
                img2pose_sampler = None

            img2pose_dataloader = DataLoader(
                img2pose_dataset,
                batch_size=int(pose_cfg.get('batch_size', batch_size)),
                shuffle=False,
                sampler=img2pose_sampler,
                num_workers=config.dataset.params.get('num_workers', 4),
                collate_fn=img2pose_dataset.collate_fn,
                drop_last=False,
                pin_memory=True,
            )
            if is_main_process():
                print(f"  Img2Pose dataset size: {len(img2pose_dataset)}")
                print(f"  Img2Pose total batches: {len(img2pose_dataloader)}")

        if 'img2text' in tasks:
            img2text_cfg = config.test.get('img2text', {})
            img2text_max_samples = img2text_cfg.get('max_samples', config.test.get('max_samples', None))
            if quick_mode:
                img2text_max_samples = num_quick_samples if img2text_max_samples is None else min(num_quick_samples, img2text_max_samples)

            img2text_dataset = H36MEvalDataset(
                data_root=h36m_cfg.data_root,
                image_root=h36m_cfg.image_root,
                split='test',
                image_resolution=image_resolution,
                max_samples=img2text_max_samples,
                seed=quick_subset_seed if quick_mode else seed,
                require_text=bool(img2text_cfg.get('require_text', True)),
            )
            if is_distributed:
                img2text_sampler = DistributedSampler(img2text_dataset, shuffle=False)
            else:
                img2text_sampler = None

            img2text_dataloader = DataLoader(
                img2text_dataset,
                batch_size=int(img2text_cfg.get('batch_size', batch_size)),
                shuffle=False,
                sampler=img2text_sampler,
                num_workers=config.dataset.params.get('num_workers', 4),
                collate_fn=img2text_dataset.collate_fn,
                drop_last=False,
                pin_memory=True,
            )
            if is_main_process():
                print(f"  Img2Text dataset size: {len(img2text_dataset)}")
                print(f"  Img2Text total batches: {len(img2text_dataloader)}")

    if 'mgie' in tasks:
        # Lazy import: the MGIE eval pipeline is not shipped in this release.
        from datasets.h36m_mgie_eval_dataset import H36MMGIEEvalDataset
        mgie_cfg = config.dataset.get('mgie', None)
        if mgie_cfg is None:
            raise ValueError("Task 'mgie' is enabled but config.dataset.mgie is missing.")
        mgie_test_cfg = config.test.get('mgie', {})
        mgie_max_samples = mgie_test_cfg.get('max_samples', config.test.get('max_samples', None))
        if quick_mode:
            mgie_max_samples = num_quick_samples if mgie_max_samples is None else min(num_quick_samples, mgie_max_samples)

        mgie_dataset = H36MMGIEEvalDataset(
            data_root=mgie_cfg.data_root,
            image_root=mgie_cfg.image_root,
            split='test',
            image_resolution=int(config.dataset.get('image_resolution', 432)),
            instruction_template_path=mgie_cfg.get('instruction_template_path', None),
            max_samples=mgie_max_samples,
            seed=quick_subset_seed if quick_mode else seed,
        )
        if is_distributed:
            mgie_sampler = DistributedSampler(mgie_dataset, shuffle=False)
        else:
            mgie_sampler = None

        mgie_dataloader = DataLoader(
            mgie_dataset,
            batch_size=int(mgie_test_cfg.get('batch_size', batch_size)),
            shuffle=False,
            sampler=mgie_sampler,
            num_workers=config.dataset.params.get('num_workers', 4),
            collate_fn=mgie_dataset.collate_fn,
            drop_last=False,
            pin_memory=True,
        )
        if is_main_process():
            print(f"  MGIE dataset size: {len(mgie_dataset)}")
            print(f"  MGIE total batches: {len(mgie_dataloader)}")
    
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
    edit_mean_t = torch.from_numpy(edit_mean).float().to(device)
    edit_std_t = torch.from_numpy(edit_std).float().to(device)
    
    # Precompute: 263-dim truncated version of the 269-dim training mean/std (for T2M renorm)
    mean_train_263 = mean_train_t[:263] if mean_train_t.shape[0] > 263 else mean_train_t
    std_train_263 = std_train_t[:263] if std_train_t.shape[0] > 263 else std_train_t
    edit_mean_263 = edit_mean_t[:263] if edit_mean_t.shape[0] > 263 else edit_mean_t
    edit_std_263 = edit_std_t[:263] if edit_std_t.shape[0] > 263 else edit_std_t
    
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

    # Final MotionGPT3 test prompts (for fair comparison)
    t2m_prompt_template = config.test.get(
        't2m_prompt_template', "Generate motion: <Caption_Placeholder>")
    m2t_prompt_template = config.test.get(
        'm2t_prompt_template', "Generate text: <Motion_Placeholder>")
    pred_prompt_template = config.test.get(
        'pred_prompt_template', "Predict motion: <Motion_Placeholder_s1>")
    t2t_prompt_template = config.test.get(
        't2t_prompt_template', "")
    img2pose_prompt_template = config.test.get(
        'img2pose_prompt_template', "Generate pose of the image: <Image_Placeholder>.")
    img2text_prompt_template = config.test.get(
        'img2text_prompt_template', "Generate description of the pose in the image: <Image_Placeholder>.")
    mgie_prompt_template = config.test.get(
        'mgie_prompt_template',
        "Given source image <Source_Image_Placeholder> and reference motion "
        "<Reference_Motion_Placeholder>, edit the person's pose to match the motion.",
    )

    if "<Caption_Placeholder>" not in t2m_prompt_template:
        t2m_prompt_template = f"{t2m_prompt_template.strip()} <Caption_Placeholder>"

    m2t_prompt_text = _strip_motion_placeholders(m2t_prompt_template)
    m2t_prompt_tokens = None
    if len(m2t_prompt_text) > 0:
        m2t_prompt_tokens = text_tokenizer(
            m2t_prompt_text, add_special_tokens=False
        )['input_ids']

    pred_prompt_text = _strip_motion_placeholders(pred_prompt_template)
    pred_prompt_tokens = None
    if len(pred_prompt_text) > 0:
        pred_prompt_tokens = text_tokenizer(
            pred_prompt_text, add_special_tokens=False
        )['input_ids']

    if is_main_process():
        print("\n  Prompt templates (aligned with MotionGPT3 final test):")
        print(f"  T2M: '{t2m_prompt_template}'")
        print(f"  M2T: '{m2t_prompt_text}'")
        print(f"  Pred: '{pred_prompt_text}'")
        print(f"  T2T prefix: '{t2t_prompt_template}'")
        print(f"  Img2Pose: '{img2pose_prompt_template}'")
        print(f"  Img2Text: '{img2text_prompt_template}'")
        print(f"  MGIE: '{mgie_prompt_template}'")
    
    # ==================== [4/5] Run Evaluation ====================
    if is_main_process():
        print(f"\n[4/5] Running evaluation...")
    
    all_results = {}  # task -> {metric_name: [values_per_rep]}
    primary_tasks = [t for t in tasks if t in ['t2m', 'm2t', 'pred']]
    
    for task in primary_tasks:
        replication_times = default_rep_per_task.get(task, 1)
        
        if is_main_process():
            print(f"\n{'='*60}")
            print(f"  Evaluating: {task.upper()} (replication_times={replication_times})")
            print(f"{'='*60}")
        
        # Get task-specific parameters
        if task == 't2m':
            guidance_scale = config.test.t2m.get('guidance_scale', 5.0)
            num_inference_steps = config.test.t2m.get('num_inference_steps', 50)
            t2m_latent_smooth_sigma = float(config.test.t2m.get('latent_smooth_sigma', 1.5))
            t2m_gen_micro_batch_size = int(
                config.test.t2m.get('gen_micro_batch_size', config.test.get('batch_size', 32))
            )
            t2m_gen_micro_batch_size = max(1, t2m_gen_micro_batch_size)
            if is_main_process():
                print(f"  Guidance scale: {guidance_scale}, Steps: {num_inference_steps}, "
                      f"Latent smooth sigma: {t2m_latent_smooth_sigma}")
                print(f"  T2M generation micro-batch size: {t2m_gen_micro_batch_size}")
                if save_t2m_sigma_smoothed_motion:
                    if enable_t2m_sigma_smoothed_save:
                        print(
                            f"  Extra save-vis post-decode sigma smooth: "
                            f"sigma={t2m_sigma_smoothed_sigma}, task={t2m_sigma_smoothed_task_name}"
                        )
                    else:
                        print(
                            "  Extra save-vis post-decode sigma smooth requested, "
                            f"but sigma={t2m_sigma_smoothed_sigma} so it is skipped."
                        )
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
            pred_latent_smooth_sigma = float(config.test.pred.get('latent_smooth_sigma', 1.5))
            if is_main_process():
                print(f"  Predict ratio: {predict_ratio}, Steps: {pred_num_steps}, "
                      f"Latent smooth sigma: {pred_latent_smooth_sigma}")
        
        task_all_metrics = {}
        
        for rep_i in range(replication_times):
            if is_main_process():
                print(f"\n  --- Replication {rep_i + 1}/{replication_times} ---")
            
            # Reset evaluator
            if task in evaluators:
                if hasattr(evaluators[task], 'reset'):
                    evaluators[task].reset()
            
            # Use a different seed for each replication
            set_seed(seed + rank + rep_i * 1000)
            
            total_samples = 0
            
            pbar = tqdm(
                eval_dataloader,
                desc=f"{task} Rep {rep_i+1}/{replication_times}",
                disable=not is_main_process()
            )
            save_this_rep = save_vis_manager.should_save_replication(rep_i)
            t2m_mode_buffers = None
            if task == 't2m':
                t2m_mode_buffers = {
                    'all': {'text': [], 'gen': [], 'gt': []},
                    'random_one': {'text': [], 'gen': [], 'gt': []},
                    'best_of_n': {'text': [], 'gen': [], 'gt': []},
                }
            
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
                    expanded_captions = []
                    expanded_motion_lengths = []
                    expanded_gt_indices = []
                    expanded_sample_ids = []
                    expanded_word_embs = []
                    expanded_pos_ohot = []
                    expanded_sent_lens = []
                    expanded_caption_idx = []
                    expanded_caption_total = []

                    use_all_captions = config.test.t2m.get('use_all_captions', True)

                    for i in range(bs):
                        sample_id = fnames[i] if i < len(fnames) else f"sample_{batch_idx:05d}_{i:03d}"

                        if use_all_captions:
                            # Expand all texts associated with this sample (allcaps mode, default behavior)
                            sample_meta = eval_dataset.data_dict.get(sample_id, {})
                            text_entries = sample_meta.get('text', [])

                            uniq_entries = []
                            seen = set()
                            for entry in text_entries:
                                cap = str(entry.get('caption', '')).strip()
                                toks = entry.get('tokens', [])
                                if cap in seen:
                                    continue
                                seen.add(cap)
                                uniq_entries.append({'caption': cap, 'tokens': toks})

                            if not uniq_entries:
                                fallback_caption = captions[i] if i < len(captions) else ""
                                uniq_entries = [{'caption': fallback_caption, 'tokens': None}]
                        else:
                            # Use only the batch's primary caption (same behavior as test_stage4.py)
                            primary_caption = captions[i] if i < len(captions) else ""
                            uniq_entries = [{'caption': primary_caption, 'tokens': None}]

                        total_caps = len(uniq_entries)
                        for cap_idx, entry in enumerate(uniq_entries):
                            expanded_captions.append(entry['caption'])
                            expanded_motion_lengths.append(int(motion_lengths[i]))
                            expanded_gt_indices.append(i)
                            expanded_sample_ids.append(sample_id)
                            expanded_caption_idx.append(cap_idx)
                            expanded_caption_total.append(total_caps)

                            if entry['tokens'] is None:
                                expanded_word_embs.append(word_embs[i].float())
                                expanded_pos_ohot.append(pos_ohot[i].float())
                                expanded_sent_lens.append(int(sent_lens[i]))
                            else:
                                proc_word_emb, proc_pos_ohot, proc_sent_len = eval_dataset._process_tokens(entry['tokens'])
                                expanded_word_embs.append(torch.from_numpy(proc_word_emb).to(device=device, dtype=torch.float32))
                                expanded_pos_ohot.append(torch.from_numpy(proc_pos_ohot).to(device=device, dtype=torch.float32))
                                expanded_sent_lens.append(int(proc_sent_len))

                    if len(expanded_captions) == 0:
                        continue

                    expanded_bs = len(expanded_captions)
                    vis_sample_ids = [
                        f"{expanded_sample_ids[i]}__cap{int(expanded_caption_idx[i]):02d}"
                        for i in range(expanded_bs)
                    ]
                    cache_enabled_this_rep = (task == 't2m' and rep_i == resume_t2m_replication and len(cached_t2m_motion_paths) > 0)
                    cached_gen_count = 0

                    gen_motion_slots = [None for _ in range(expanded_bs)]
                    gen_lengths = [0 for _ in range(expanded_bs)]
                    need_gen_indices = []
                    for i, vis_sample_id in enumerate(vis_sample_ids):
                        cached_path = cached_t2m_motion_paths.get(vis_sample_id, None) if cache_enabled_this_rep else None
                        if cached_path is not None and os.path.exists(cached_path):
                            cached_motion = np.load(cached_path)
                            if cached_motion.ndim == 3 and cached_motion.shape[0] == 1:
                                cached_motion = cached_motion[0]
                            if cached_motion.ndim != 2:
                                need_gen_indices.append(i)
                                continue
                            cached_tensor = torch.from_numpy(cached_motion).to(device=device, dtype=torch.float32)
                            cached_tensor = ensure_eval_motion_dim(cached_tensor)
                            gen_motion_slots[i] = cached_tensor
                            gen_lengths[i] = int(cached_tensor.shape[0])
                            cached_gen_count += 1
                        else:
                            need_gen_indices.append(i)

                    if len(need_gen_indices) > 0:
                        total_need_gen = len(need_gen_indices)
                        for chunk_start in range(0, total_need_gen, t2m_gen_micro_batch_size):
                            chunk_indices = need_gen_indices[chunk_start:chunk_start + t2m_gen_micro_batch_size]
                            chunk_captions = [expanded_captions[i] for i in chunk_indices]
                            chunk_motion_lengths = [expanded_motion_lengths[i] for i in chunk_indices]
                            chunk_motion_length_max = max(chunk_motion_lengths)
                            chunk_latent_len = latent_len_from_motion_length(
                                chunk_motion_length_max,
                                downsample_rate,
                            )
                            chunk_num_motion_tokens = (
                                chunk_latent_len + 1 if config.model.showo.add_time_embeds else chunk_latent_len
                            )

                            cond_tokens, uncond_tokens, cond_positions, uncond_positions = prepare_t2m_input(
                                chunk_captions, text_tokenizer, chunk_num_motion_tokens,
                                bos_id, eos_id, bom_id, eom_id, pad_id, mot_pad_id,
                                max_text_len, max_seq_len, device,
                                instruction_template=t2m_prompt_template,
                            )

                            if guidance_scale > 0:
                                text_tokens = torch.cat([cond_tokens, uncond_tokens], dim=0)
                                modality_positions = torch.cat([cond_positions, uncond_positions], dim=0)
                            else:
                                text_tokens = cond_tokens
                                modality_positions = cond_positions

                            chunk_gen_motions, chunk_gen_lengths = generate_t2m(
                                model, motion_vae, sampler, text_tokens, modality_positions,
                                chunk_latent_len, motion_latent_dim, chunk_motion_lengths, max_seq_len,
                                guidance_scale, num_inference_steps, config, device, weight_type,
                                latent_smooth_sigma=t2m_latent_smooth_sigma,
                            )
                            chunk_gen_motions = ensure_eval_motion_dim(chunk_gen_motions)

                            for j, expanded_idx in enumerate(chunk_indices):
                                gen_motion_slots[expanded_idx] = chunk_gen_motions[j]
                                gen_lengths[expanded_idx] = int(chunk_gen_lengths[j])

                    valid_slots = [m for m in gen_motion_slots if m is not None]
                    if len(valid_slots) != expanded_bs:
                        raise RuntimeError(
                            f"T2M resume/generation mismatch: got {len(valid_slots)}/{expanded_bs} generated motions"
                        )
                    merged_max_len = max(int(m.shape[0]) for m in valid_slots)
                    merged_motions = []
                    for m in gen_motion_slots:
                        if m.shape[0] < merged_max_len:
                            pad = torch.zeros(merged_max_len - m.shape[0], m.shape[1], device=device, dtype=m.dtype)
                            m = torch.cat([m, pad], dim=0)
                        merged_motions.append(m)
                    gen_motions = torch.stack(merged_motions, dim=0)

                    if is_main_process():
                        pbar.set_postfix_str(f"cached={cached_gen_count}/{expanded_bs}")

                    expanded_gt_motions = torch.stack(
                        [gt_motions[idx] for idx in expanded_gt_indices], dim=0
                    )
                    expanded_gt_lengths = [int(gt_lengths[idx]) for idx in expanded_gt_indices]
                    gt_max_len = expanded_gt_motions.shape[1]
                    gen_max_len = gen_motions.shape[1]
                    if gen_max_len < gt_max_len:
                        pad = torch.zeros(expanded_bs, gt_max_len - gen_max_len, 263, device=device, dtype=gen_motions.dtype)
                        gen_motions = torch.cat([gen_motions, pad], dim=1)
                    elif gen_max_len > gt_max_len:
                        gen_motions = gen_motions[:, :gt_max_len, :]

                    if save_this_rep:
                        need_gen_set = set(need_gen_indices)
                        for i in range(expanded_bs):
                            if i not in need_gen_set:
                                continue
                            base_sample_id = expanded_sample_ids[i]
                            cap_idx = expanded_caption_idx[i]
                            total_caps = expanded_caption_total[i]
                            vis_sample_id = vis_sample_ids[i]
                            motion_len_i = int(gen_lengths[i]) if i < len(gen_lengths) else int(gen_motions.shape[1])
                            motion_len_i = max(1, min(motion_len_i, int(gen_motions.shape[1])))
                            save_vis_manager.save_motion_sample(
                                task='t2m',
                                replication=rep_i,
                                batch_idx=batch_idx,
                                in_batch_idx=i,
                                sample_id=vis_sample_id,
                                motion=gen_motions[i, :motion_len_i],
                                keep_dim=263,
                                extra={
                                    'fname': base_sample_id,
                                    'caption': expanded_captions[i],
                                    'caption_idx': cap_idx,
                                    'caption_total': total_caps,
                                    'motion_length': motion_len_i,
                                },
                            )
                            if enable_t2m_sigma_smoothed_save:
                                smoothed_motion = smooth_motion_sequence(
                                    gen_motions[i, :motion_len_i],
                                    t2m_sigma_smoothed_sigma,
                                )
                                save_vis_manager.save_motion_sample(
                                    task=t2m_sigma_smoothed_task_name,
                                    replication=rep_i,
                                    batch_idx=batch_idx,
                                    in_batch_idx=i,
                                    sample_id=vis_sample_id,
                                    motion=smoothed_motion,
                                    keep_dim=263,
                                    extra={
                                        'fname': base_sample_id,
                                        'caption': expanded_captions[i],
                                        'caption_idx': cap_idx,
                                        'caption_total': total_caps,
                                        'motion_length': motion_len_i,
                                        'from_task': 't2m',
                                        'postdecode_sigma_smooth': True,
                                        'postdecode_sigma': t2m_sigma_smoothed_sigma,
                                    },
                                )

                    # Renorm: training norm (263) -> eval norm
                    gen_motions = renorm4t2m(gen_motions, mean_train_263, std_train_263, mean_eval_t, std_eval_t)
                    eval_gen_lengths = [min(int(gen_lengths[i]), int(expanded_gt_lengths[i])) for i in range(expanded_bs)]

                    expanded_word_embs_t = torch.stack(expanded_word_embs, dim=0)
                    expanded_pos_ohot_t = torch.stack(expanded_pos_ohot, dim=0)
                    expanded_sent_lens_t = torch.tensor(expanded_sent_lens, device=device, dtype=torch.long)

                    text_emb = evaluators['t2m'].get_text_embedding(
                        expanded_word_embs_t,
                        expanded_pos_ohot_t,
                        expanded_sent_lens_t,
                    ).detach().cpu()
                    gen_emb = evaluators['t2m'].get_motion_embedding(
                        gen_motions.float(),
                        eval_gen_lengths,
                    ).detach().cpu()
                    gt_emb = evaluators['t2m'].get_motion_embedding(
                        expanded_gt_motions.float(),
                        expanded_gt_lengths,
                    ).detach().cpu()

                    # Matching distance for each candidate pair (text_i vs gen_i)
                    pair_dist = torch.norm(text_emb - gen_emb, dim=1)
                    sample_to_indices = {}
                    for idx, sample_idx in enumerate(expanded_gt_indices):
                        sample_to_indices.setdefault(int(sample_idx), []).append(idx)

                    chosen_random = []
                    chosen_best = []
                    for sample_idx in sorted(sample_to_indices.keys()):
                        cand = sample_to_indices[sample_idx]
                        if len(cand) == 1:
                            chosen_random.append(cand[0])
                            chosen_best.append(cand[0])
                            continue
                        chosen_random.append(cand[np.random.randint(0, len(cand))])
                        local_best = int(torch.argmin(pair_dist[cand]).item())
                        chosen_best.append(cand[local_best])

                    t2m_mode_buffers['all']['text'].append(text_emb)
                    t2m_mode_buffers['all']['gen'].append(gen_emb)
                    t2m_mode_buffers['all']['gt'].append(gt_emb)

                    random_idx = torch.tensor(chosen_random, dtype=torch.long)
                    best_idx = torch.tensor(chosen_best, dtype=torch.long)
                    t2m_mode_buffers['random_one']['text'].append(text_emb[random_idx])
                    t2m_mode_buffers['random_one']['gen'].append(gen_emb[random_idx])
                    t2m_mode_buffers['random_one']['gt'].append(gt_emb[random_idx])
                    t2m_mode_buffers['best_of_n']['text'].append(text_emb[best_idx])
                    t2m_mode_buffers['best_of_n']['gen'].append(gen_emb[best_idx])
                    t2m_mode_buffers['best_of_n']['gt'].append(gt_emb[best_idx])
                
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
                    
                    text_tokens_m2t, modality_positions_m2t, m2t_prompt_len = prepare_m2t_input(
                        motion_latents, text_tokenizer, m2t_num_tokens,
                        bos_id, eos_id, bom_id, eom_id, pad_id, mot_pad_id,
                        max_seq_len, device,
                        prompt_tokens=m2t_prompt_tokens,
                    )
                    
                    gen_texts = generate_m2t(
                        model, motion_vae, text_tokenizer, motion_latents.to(weight_type),
                        text_tokens_m2t, modality_positions_m2t, gt_lengths,
                        max_new_tokens, eos_id,
                        max_seq_len, device, weight_type,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_k=top_k,
                        prompt_len=m2t_prompt_len,
                    )
                    
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
                        pred_prompt_tokens=pred_prompt_tokens,
                        debug=(batch_idx == 0 and rep_i == 0 and is_main_process()),
                        latent_smooth_sigma=pred_latent_smooth_sigma,
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
                
                # Aggregate embeddings (needed only for T2M and M2T)
                if task in ['t2m', 'm2t']:
                    evaluator = evaluators[task]
                    if hasattr(evaluator, 'text_embeddings'):
                        if task == 'm2t':
                            all_text_embs = gather_embeddings(evaluator.text_embeddings, world_size)
                            all_gen_embs = gather_embeddings(evaluator.predtext_embeddings, world_size)
                            all_gt_embs = gather_embeddings(evaluator.gtmotion_embeddings, world_size)
                        
                        if is_main_process():
                            if task == 'm2t':
                                evaluator.text_embeddings = [all_text_embs] if all_text_embs is not None else []
                                evaluator.predtext_embeddings = [all_gen_embs] if all_gen_embs is not None else []
                                evaluator.gtmotion_embeddings = [all_gt_embs] if all_gt_embs is not None else []

            t2m_mode_gathered = None
            if task == 't2m':
                t2m_mode_gathered = {}
                for mode_name in ['all', 'random_one', 'best_of_n']:
                    mode_text = t2m_mode_buffers[mode_name]['text']
                    mode_gen = t2m_mode_buffers[mode_name]['gen']
                    mode_gt = t2m_mode_buffers[mode_name]['gt']
                    if is_distributed:
                        mode_text_g = gather_embeddings(mode_text, world_size)
                        mode_gen_g = gather_embeddings(mode_gen, world_size)
                        mode_gt_g = gather_embeddings(mode_gt, world_size)
                    else:
                        mode_text_g = torch.cat(mode_text, dim=0) if len(mode_text) > 0 else None
                        mode_gen_g = torch.cat(mode_gen, dim=0) if len(mode_gen) > 0 else None
                        mode_gt_g = torch.cat(mode_gt, dim=0) if len(mode_gt) > 0 else None
                    t2m_mode_gathered[mode_name] = (mode_text_g, mode_gen_g, mode_gt_g)
            
            # Compute metrics for this replication
            if is_main_process():
                rep_metrics = {}
                if task == 't2m':
                    for mode_name in ['all', 'random_one', 'best_of_n']:
                        mode_text_g, mode_gen_g, mode_gt_g = t2m_mode_gathered[mode_name]

                        evaluators['t2m'].text_embeddings = [mode_text_g] if mode_text_g is not None else []
                        evaluators['t2m'].gen_motion_embeddings = [mode_gen_g] if mode_gen_g is not None else []
                        evaluators['t2m'].gt_motion_embeddings = [mode_gt_g] if mode_gt_g is not None else []
                        mode_metrics = evaluators['t2m'].compute()
                        for key, value in mode_metrics.items():
                            rep_metrics[f"{mode_name}/{key}"] = value

                    for mode_name in ['all', 'random_one', 'best_of_n']:
                        mode_lines = []
                        for key in ['FID', 'Matching_Score', 'R_Precision_Top_1', 'R_Precision_Top_2', 'R_Precision_Top_3', 'Diversity']:
                            metric_key = f"{mode_name}/{key}"
                            if metric_key in rep_metrics and isinstance(rep_metrics[metric_key], float):
                                mode_lines.append(f"{key}={rep_metrics[metric_key]:.4f}")
                        if mode_lines:
                            print(f"  Rep {rep_i+1} [{mode_name}]: {', '.join(mode_lines)}")
                else:
                    rep_metrics = evaluators[task].compute()
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

    # ---------- T2T ----------
    if 't2t' in tasks:
        if t2t_dataloader is None:
            raise RuntimeError("Task 't2t' requested but t2t_dataloader is not initialized.")
        if text_nlg_evaluator is None:
            raise RuntimeError("Task 't2t' requires TextNLGEvaluator.")

        replication_times = default_rep_per_task.get('t2t', 1)
        t2t_cfg = config.test.get('t2t', {})
        t2t_max_new_tokens = int(t2t_cfg.get('max_new_tokens', 80))
        t2t_do_sample = bool(t2t_cfg.get('do_sample', False))
        t2t_temperature = float(t2t_cfg.get('temperature', 1.0))
        t2t_top_k = t2t_cfg.get('top_k', 50)
        t2t_max_input_tokens = int(t2t_cfg.get('max_input_tokens', max(16, max_text_len // 2)))

        if is_main_process():
            print(f"\n{'='*60}")
            print(f"  Evaluating: T2T (replication_times={replication_times})")
            print(f"{'='*60}")

        t2t_all_metrics = {}
        total_samples_final = 0

        for rep_i in range(replication_times):
            if is_distributed and hasattr(t2t_dataloader, 'sampler') and isinstance(t2t_dataloader.sampler, DistributedSampler):
                t2t_dataloader.sampler.set_epoch(rep_i)

            local_preds = []
            local_refs = []
            local_total = 0
            set_seed(seed + rank + rep_i * 1000)
            save_this_rep = save_vis_manager.should_save_replication(rep_i)

            pbar = tqdm(
                t2t_dataloader,
                desc=f"t2t Rep {rep_i+1}/{replication_times}",
                disable=not is_main_process(),
            )
            for batch_idx, batch in enumerate(pbar):
                input_texts = batch['input_text']
                output_texts = batch['output_text']
                sample_ids = batch.get('sample_id', [])
                task_indices = batch.get('task_index', [])
                task_classes = batch.get('task_class', [])
                text_tokens_t2t, valid_lens_t2t = prepare_t2t_input(
                    input_texts=input_texts,
                    text_tokenizer=text_tokenizer,
                    bos_id=bos_id,
                    pad_id=pad_id,
                    max_seq_len=max_seq_len,
                    max_input_tokens=t2t_max_input_tokens,
                    device=device,
                    prompt_template=t2t_prompt_template,
                )

                gen_texts = generate_t2t_texts(
                    model=model,
                    text_tokenizer=text_tokenizer,
                    text_tokens=text_tokens_t2t,
                    valid_lens=valid_lens_t2t,
                    max_new_tokens=t2t_max_new_tokens,
                    eos_token=eos_id,
                    device=device,
                    weight_type=weight_type,
                    do_sample=t2t_do_sample,
                    temperature=t2t_temperature,
                    top_k=t2t_top_k,
                )

                local_preds.extend(gen_texts)
                for ref in output_texts:
                    local_refs.append([ref] if isinstance(ref, str) else [""])
                local_total += len(gen_texts)
                if save_this_rep:
                    for i, pred_text in enumerate(gen_texts):
                        sample_id = sample_ids[i] if i < len(sample_ids) else f"sample_{batch_idx:05d}_{i:03d}"
                        save_vis_manager.save_text_sample(
                            task='t2t',
                            replication=rep_i,
                            batch_idx=batch_idx,
                            in_batch_idx=i,
                            sample_id=sample_id,
                            text=pred_text,
                            extra={
                                'output_text': output_texts[i] if i < len(output_texts) else "",
                                'input_text': input_texts[i] if i < len(input_texts) else "",
                                'task_index': int(task_indices[i]) if i < len(task_indices) else None,
                                'task_class': task_classes[i] if i < len(task_classes) else "",
                            },
                        )

            total_samples = local_total
            preds = local_preds
            refs = local_refs
            if is_distributed:
                dist.barrier()
                gathered = gather_object_list(
                    {'preds': local_preds, 'refs': local_refs, 'total': local_total},
                    world_size,
                )
                if is_main_process():
                    preds = []
                    refs = []
                    total_samples = 0
                    for item in gathered:
                        preds.extend(item['preds'])
                        refs.extend(item['refs'])
                        total_samples += int(item['total'])
                dist.barrier()

            if is_main_process():
                rep_metrics = text_nlg_evaluator.compute(preds, refs)
                total_samples_final = total_samples
                key_metrics = [f"{k}={v:.4f}" for k, v in sorted(rep_metrics.items())]
                if key_metrics:
                    print(f"  Rep {rep_i+1}: {', '.join(key_metrics)}")
                for key, value in rep_metrics.items():
                    t2t_all_metrics.setdefault(key, []).append(value)

        all_results['t2t'] = {
            'total_samples': total_samples_final,
            'replication_times': replication_times,
            'per_rep_metrics': t2t_all_metrics,
        }

    # ---------- Img2Pose ----------
    if 'img2pose' in tasks:
        if img2pose_dataloader is None:
            raise RuntimeError("Task 'img2pose' requested but img2pose_dataloader is not initialized.")
        if image_vae is None:
            raise RuntimeError("Task 'img2pose' requires WAN21 image VAE.")

        replication_times = default_rep_per_task.get('img2pose', 1)
        img2pose_cfg = config.test.get('img2pose', {})
        img2pose_guidance = float(img2pose_cfg.get('guidance_scale', 5.0))
        img2pose_steps = int(img2pose_cfg.get('num_inference_steps', 50))
        img2pose_vis_enabled = bool(img2pose_cfg.get('save_alignment_vis', False))
        img2pose_vis_num = int(img2pose_cfg.get('vis_num_samples', 10))
        img2pose_vis_dir = os.path.join(output_dir, f"img2pose_alignment_{run_timestamp}")
        img2pose_vis_use_h36m_d2 = bool(img2pose_cfg.get('vis_use_h36m_d2', True))
        h36m_cropped_root = config.dataset.h36m.image_root
        img2pose_vis_full_root = img2pose_cfg.get('vis_h36m_full_image_root', None)
        img2pose_vis_extracted_root = img2pose_cfg.get('vis_h36m_extracted_root', None)
        if img2pose_vis_full_root is None:
            img2pose_vis_full_root = h36m_cropped_root.replace(
                'images_fps50_cropped_448x448', 'images_fps50'
            )
        if img2pose_vis_extracted_root is None:
            if '/processed/' in h36m_cropped_root:
                ds_root = h36m_cropped_root.split('/processed/')[0]
                img2pose_vis_extracted_root = os.path.join(ds_root, 'extracted')
            else:
                img2pose_vis_extracted_root = ''

        if is_main_process():
            print(f"\n{'='*60}")
            print(f"  Evaluating: IMG2POSE (replication_times={replication_times})")
            print(f"{'='*60}")
            if img2pose_vis_enabled:
                print(f"  Alignment visualization enabled: {img2pose_vis_num} samples")
                print(f"  Use H36M D2 alignment: {img2pose_vis_use_h36m_d2}")
                print(f"  Full image root: {img2pose_vis_full_root}")
                print(f"  Extracted root: {img2pose_vis_extracted_root}")

        img2pose_all_metrics = {}
        total_samples_final = 0

        for rep_i in range(replication_times):
            if is_distributed and hasattr(img2pose_dataloader, 'sampler') and isinstance(img2pose_dataloader.sampler, DistributedSampler):
                img2pose_dataloader.sampler.set_epoch(rep_i)

            set_seed(seed + rank + rep_i * 1000)
            save_this_rep = save_vis_manager.should_save_replication(rep_i)
            local_gen = []
            local_gt = []
            local_gt_raw = []
            local_total = 0
            local_vis_samples = []

            pbar = tqdm(
                img2pose_dataloader,
                desc=f"img2pose Rep {rep_i+1}/{replication_times}",
                disable=not is_main_process(),
            )
            for batch_idx, batch in enumerate(pbar):
                images = batch['images'].to(device).to(weight_type)
                gt_motion_norm = batch['motions_norm'].to(device)
                gt_motion_raw = batch['motions_raw'].to(device)
                prompts = [img2pose_prompt_template] * images.shape[0]
                image_paths = batch.get('image_paths', [])

                gen_motion_norm = generate_img2pose(
                    model=model,
                    image_vae=image_vae,
                    motion_vae=motion_vae,
                    sampler=sampler,
                    images=images,
                    prompts=prompts,
                    text_tokenizer=text_tokenizer,
                    config=config,
                    showo_token_ids=showo_token_ids,
                    max_seq_len=max_seq_len,
                    motion_latent_dim=motion_latent_dim,
                    device=device,
                    weight_type=weight_type,
                    guidance_scale=img2pose_guidance,
                    num_inference_steps=img2pose_steps,
                )

                local_gen.append(gen_motion_norm.float().cpu())
                local_gt.append(gt_motion_norm.float().cpu())
                local_gt_raw.append(gt_motion_raw.float().cpu())
                local_total += gen_motion_norm.shape[0]
                if save_this_rep:
                    for i in range(gen_motion_norm.shape[0]):
                        sample_id = image_paths[i] if i < len(image_paths) else f"sample_{batch_idx:05d}_{i:03d}"
                        save_vis_manager.save_motion_sample(
                            task='img2pose',
                            replication=rep_i,
                            batch_idx=batch_idx,
                            in_batch_idx=i,
                            sample_id=sample_id,
                            motion=gen_motion_norm[i],
                            keep_dim=269,
                            extra={
                                'image_path': image_paths[i] if i < len(image_paths) else "",
                            },
                        )
                if img2pose_vis_enabled and rep_i == 0 and len(local_vis_samples) < img2pose_vis_num:
                    remain = img2pose_vis_num - len(local_vis_samples)
                    take_n = min(remain, gen_motion_norm.shape[0])
                    images_cpu = images.float().cpu()
                    gen_cpu = gen_motion_norm.float().cpu()
                    gt_cpu = gt_motion_norm.float().cpu()
                    image_paths = batch.get('image_paths', [""] * take_n)
                    for j in range(take_n):
                        local_vis_samples.append(
                            {
                                "image": images_cpu[j],
                                "gen_motion_norm": gen_cpu[j],
                                "gt_motion_norm": gt_cpu[j],
                                "image_path": image_paths[j] if j < len(image_paths) else "",
                            }
                        )

            gen_tensor = torch.cat(local_gen, dim=0) if len(local_gen) > 0 else torch.zeros(0, 269)
            gt_tensor = torch.cat(local_gt, dim=0) if len(local_gt) > 0 else torch.zeros(0, 269)
            gt_raw_tensor = torch.cat(local_gt_raw, dim=0) if len(local_gt_raw) > 0 else torch.zeros(0, 269)
            total_samples = local_total

            if is_distributed:
                dist.barrier()
                gathered = gather_object_list(
                    {
                        'gen': gen_tensor.numpy(),
                        'gt': gt_tensor.numpy(),
                        'gt_raw': gt_raw_tensor.numpy(),
                        'total': local_total,
                    },
                    world_size,
                )
                if is_main_process():
                    gen_arr = []
                    gt_arr = []
                    gt_raw_arr = []
                    total_samples = 0
                    for item in gathered:
                        if item['gen'].shape[0] > 0:
                            gen_arr.append(item['gen'])
                            gt_arr.append(item['gt'])
                            gt_raw_arr.append(item['gt_raw'])
                        total_samples += int(item['total'])
                    if len(gen_arr) > 0:
                        gen_tensor = torch.from_numpy(np.concatenate(gen_arr, axis=0))
                        gt_tensor = torch.from_numpy(np.concatenate(gt_arr, axis=0))
                        gt_raw_tensor = torch.from_numpy(np.concatenate(gt_raw_arr, axis=0))
                    else:
                        gen_tensor = torch.zeros(0, 269)
                        gt_tensor = torch.zeros(0, 269)
                        gt_raw_tensor = torch.zeros(0, 269)
                dist.barrier()

            if is_main_process():
                evaluators['img2pose'].reset()
                if gen_tensor.shape[0] > 0:
                    evaluators['img2pose'].update(
                        gen_motion_norm=gen_tensor.to(device),
                        gt_motion_norm=gt_tensor.to(device),
                        gt_motion_raw=gt_raw_tensor.to(device),
                    )
                rep_metrics = evaluators['img2pose'].compute()
                total_samples_final = total_samples
                key_metrics = [f"{k}={v:.4f}" for k, v in sorted(rep_metrics.items()) if isinstance(v, float)]
                if key_metrics:
                    print(f"  Rep {rep_i+1}: {', '.join(key_metrics)}")
                for key, value in rep_metrics.items():
                    if isinstance(value, (float, int)):
                        img2pose_all_metrics.setdefault(key, []).append(float(value))
                if img2pose_vis_enabled and rep_i == 0 and len(local_vis_samples) > 0:
                    save_img2pose_alignment_visualizations(
                        samples=local_vis_samples[:img2pose_vis_num],
                        evaluator=evaluators['img2pose'],
                        out_dir=img2pose_vis_dir,
                        h36m_full_image_root=img2pose_vis_full_root,
                        h36m_extracted_root=img2pose_vis_extracted_root,
                        use_h36m_d2=img2pose_vis_use_h36m_d2,
                    )
                    print(f"  Saved alignment visualizations to: {img2pose_vis_dir}")

        all_results['img2pose'] = {
            'total_samples': total_samples_final,
            'replication_times': replication_times,
            'per_rep_metrics': img2pose_all_metrics,
        }

    # ---------- Img2Text ----------
    if 'img2text' in tasks:
        if img2text_dataloader is None:
            raise RuntimeError("Task 'img2text' requested but img2text_dataloader is not initialized.")
        if image_vae is None:
            raise RuntimeError("Task 'img2text' requires WAN21 image VAE.")
        if text_nlg_evaluator is None:
            raise RuntimeError("Task 'img2text' requires TextNLGEvaluator.")

        replication_times = default_rep_per_task.get('img2text', 1)
        img2text_cfg = config.test.get('img2text', {})
        img2text_max_new_tokens = int(img2text_cfg.get('max_new_tokens', 80))
        img2text_do_sample = bool(img2text_cfg.get('do_sample', False))
        img2text_temperature = float(img2text_cfg.get('temperature', 1.0))
        img2text_top_k = img2text_cfg.get('top_k', 50)
        num_image_tokens_eval = int(config.dataset.get('num_image_tokens', 729))

        if is_main_process():
            print(f"\n{'='*60}")
            print(f"  Evaluating: IMG2TEXT (replication_times={replication_times})")
            print(f"{'='*60}")

        img2text_all_metrics = {}
        total_samples_final = 0

        for rep_i in range(replication_times):
            if is_distributed and hasattr(img2text_dataloader, 'sampler') and isinstance(img2text_dataloader.sampler, DistributedSampler):
                img2text_dataloader.sampler.set_epoch(rep_i)

            set_seed(seed + rank + rep_i * 1000)
            save_this_rep = save_vis_manager.should_save_replication(rep_i)
            local_preds = []
            local_refs = []
            local_total = 0

            pbar = tqdm(
                img2text_dataloader,
                desc=f"img2text Rep {rep_i+1}/{replication_times}",
                disable=not is_main_process(),
            )
            for batch_idx, batch in enumerate(pbar):
                images = batch['images'].to(device).to(weight_type)
                refs_batch = batch['all_texts']
                image_paths = batch.get('image_paths', [])
                prompts = [img2text_prompt_template] * images.shape[0]

                gen_texts = generate_img2text(
                    model=model,
                    image_vae=image_vae,
                    images=images,
                    prompts=prompts,
                    text_tokenizer=text_tokenizer,
                    showo_token_ids=showo_token_ids,
                    num_image_tokens=num_image_tokens_eval,
                    max_seq_len=max_seq_len,
                    max_text_len=max_text_len,
                    max_new_tokens=img2text_max_new_tokens,
                    eos_id=eos_id,
                    device=device,
                    weight_type=weight_type,
                    do_sample=img2text_do_sample,
                    temperature=img2text_temperature,
                    top_k=img2text_top_k,
                )

                local_preds.extend(gen_texts)
                for refs in refs_batch:
                    if isinstance(refs, list) and len(refs) > 0:
                        local_refs.append(refs)
                    else:
                        local_refs.append([""])
                local_total += len(gen_texts)
                if save_this_rep:
                    for i, pred_text in enumerate(gen_texts):
                        sample_id = image_paths[i] if i < len(image_paths) else f"sample_{batch_idx:05d}_{i:03d}"
                        save_vis_manager.save_text_sample(
                            task='img2text',
                            replication=rep_i,
                            batch_idx=batch_idx,
                            in_batch_idx=i,
                            sample_id=sample_id,
                            text=pred_text,
                            extra={
                                'image_path': image_paths[i] if i < len(image_paths) else "",
                                'gt_texts': refs_batch[i] if i < len(refs_batch) else [],
                            },
                        )

            total_samples = local_total
            preds = local_preds
            refs = local_refs
            if is_distributed:
                dist.barrier()
                gathered = gather_object_list(
                    {'preds': local_preds, 'refs': local_refs, 'total': local_total},
                    world_size,
                )
                if is_main_process():
                    preds = []
                    refs = []
                    total_samples = 0
                    for item in gathered:
                        preds.extend(item['preds'])
                        refs.extend(item['refs'])
                        total_samples += int(item['total'])
                dist.barrier()

            if is_main_process():
                rep_metrics = text_nlg_evaluator.compute(preds, refs)
                total_samples_final = total_samples
                key_metrics = [f"{k}={v:.4f}" for k, v in sorted(rep_metrics.items())]
                if key_metrics:
                    print(f"  Rep {rep_i+1}: {', '.join(key_metrics)}")
                for key, value in rep_metrics.items():
                    img2text_all_metrics.setdefault(key, []).append(value)

        all_results['img2text'] = {
            'total_samples': total_samples_final,
            'replication_times': replication_times,
            'per_rep_metrics': img2text_all_metrics,
        }

    # ---------- MGIE ----------
    if 'mgie' in tasks:
        if mgie_dataloader is None:
            raise RuntimeError("Task 'mgie' requested but mgie_dataloader is not initialized.")
        if image_vae is None:
            raise RuntimeError("Task 'mgie' requires WAN21 image VAE.")

        replication_times = default_rep_per_task.get('mgie', 1)
        mgie_cfg = config.test.get('mgie', {})
        mgie_guidance_scale = float(mgie_cfg.get('guidance_scale', 3.0))
        mgie_num_steps = int(mgie_cfg.get('num_inference_steps', 50))
        mgie_use_dataset_instruction = bool(mgie_cfg.get('use_dataset_instruction', True))
        mgie_pose_proxy_enabled = bool(mgie_cfg.get('enable_pose_proxy', True))

        if is_main_process():
            print(f"\n{'='*60}")
            print(f"  Evaluating: MGIE (replication_times={replication_times})")
            print(f"{'='*60}")
            print(f"  Guidance scale: {mgie_guidance_scale}, Steps: {mgie_num_steps}")

        mgie_all_metrics = {}
        total_samples_final = 0

        for rep_i in range(replication_times):
            if is_distributed and hasattr(mgie_dataloader, 'sampler') and isinstance(mgie_dataloader.sampler, DistributedSampler):
                mgie_dataloader.sampler.set_epoch(rep_i)

            set_seed(seed + rank + rep_i * 1000)
            save_this_rep = save_vis_manager.should_save_replication(rep_i)
            evaluators['mgie'].reset()
            if mgie_pose_proxy_enabled and (mgie_pose_evaluator is not None):
                mgie_pose_evaluator.reset()

            local_total = 0
            pbar = tqdm(
                mgie_dataloader,
                desc=f"mgie Rep {rep_i+1}/{replication_times}",
                disable=not is_main_process(),
            )
            for batch_idx, batch in enumerate(pbar):
                source_images = batch['source_images'].to(device).to(weight_type)
                target_images = batch['target_images'].to(device).to(weight_type)
                ref_motions = batch['ref_motions_norm'].to(device)
                instruction_texts = batch.get('instruction_texts', None)
                sample_ids = batch.get('sample_ids', [])
                source_image_paths = batch.get('source_image_paths', [])
                target_image_paths = batch.get('target_image_paths', [])

                if mgie_use_dataset_instruction and instruction_texts is not None:
                    prompts = list(instruction_texts)
                else:
                    prompts = [mgie_prompt_template] * source_images.shape[0]
                clip_texts = list(instruction_texts) if instruction_texts is not None else prompts

                gen_images = generate_mgie_image(
                    model=model,
                    image_vae=image_vae,
                    motion_vae=motion_vae,
                    sampler=sampler,
                    source_images=source_images,
                    ref_motions_norm=ref_motions,
                    prompts=prompts,
                    text_tokenizer=text_tokenizer,
                    config=config,
                    showo_token_ids=showo_token_ids,
                    max_seq_len=max_seq_len,
                    motion_latent_dim=motion_latent_dim,
                    device=device,
                    weight_type=weight_type,
                    guidance_scale=mgie_guidance_scale,
                    num_inference_steps=mgie_num_steps,
                )
                evaluators['mgie'].update(
                    pred_images=gen_images,
                    gt_images=target_images,
                    texts=clip_texts,
                )
                if save_this_rep:
                    for i in range(gen_images.shape[0]):
                        sample_id = sample_ids[i] if i < len(sample_ids) else f"sample_{batch_idx:05d}_{i:03d}"
                        save_vis_manager.save_image_sample(
                            task='mgie',
                            replication=rep_i,
                            batch_idx=batch_idx,
                            in_batch_idx=i,
                            sample_id=sample_id,
                            image=gen_images[i],
                            extra={
                                'sample_id': sample_id,
                                'source_image_path': source_image_paths[i] if i < len(source_image_paths) else "",
                                'target_image_path': target_image_paths[i] if i < len(target_image_paths) else "",
                                'instruction_text': instruction_texts[i] if instruction_texts is not None and i < len(instruction_texts) else "",
                            },
                        )

                if mgie_pose_proxy_enabled and (mgie_pose_evaluator is not None):
                    pose_prompts = [img2pose_prompt_template] * gen_images.shape[0]
                    pose_motion_norm = generate_img2pose(
                        model=model,
                        image_vae=image_vae,
                        motion_vae=motion_vae,
                        sampler=sampler,
                        images=gen_images.to(weight_type),
                        prompts=pose_prompts,
                        text_tokenizer=text_tokenizer,
                        config=config,
                        showo_token_ids=showo_token_ids,
                        max_seq_len=max_seq_len,
                        motion_latent_dim=motion_latent_dim,
                        device=device,
                        weight_type=weight_type,
                        guidance_scale=float(config.test.get('img2pose', {}).get('guidance_scale', 5.0)),
                        num_inference_steps=int(config.test.get('img2pose', {}).get('num_inference_steps', 50)),
                    )
                    gt_motion_norm = batch['target_motions_norm'].to(device)
                    mgie_pose_evaluator.update(
                        gen_motion_norm=pose_motion_norm,
                        gt_motion_norm=gt_motion_norm,
                    )

                local_total += gen_images.shape[0]

            mgie_eval = evaluators['mgie']
            local_stats = {
                'num_samples': int(mgie_eval.total_samples),
                'psnr_sum': float(np.sum(mgie_eval.psnr_vals)) if len(mgie_eval.psnr_vals) > 0 else 0.0,
                'ssim_sum': float(np.sum(mgie_eval.ssim_vals)) if len(mgie_eval.ssim_vals) > 0 else 0.0,
                'lpips_sum': float(np.sum(mgie_eval.lpips_vals)) if len(mgie_eval.lpips_vals) > 0 else 0.0,
                'clip_i_sum': float(np.sum(mgie_eval.clip_i_vals)) if len(mgie_eval.clip_i_vals) > 0 else 0.0,
                'clip_t_sum': float(np.sum(mgie_eval.clip_t_vals)) if len(mgie_eval.clip_t_vals) > 0 else 0.0,
                'clip_t_count': int(len(mgie_eval.clip_t_vals)),
                'pose_mpjpe_sum': 0.0,
                'pose_pa_sum': 0.0,
                'pose_count': 0,
                'local_total': int(local_total),
            }
            if mgie_pose_proxy_enabled and (mgie_pose_evaluator is not None):
                local_stats['pose_mpjpe_sum'] = float(np.sum(mgie_pose_evaluator.mpjpe_list)) if len(mgie_pose_evaluator.mpjpe_list) > 0 else 0.0
                local_stats['pose_pa_sum'] = float(np.sum(mgie_pose_evaluator.pa_mpjpe_list)) if len(mgie_pose_evaluator.pa_mpjpe_list) > 0 else 0.0
                local_stats['pose_count'] = int(len(mgie_pose_evaluator.mpjpe_list))

            if is_distributed:
                dist.barrier()
                gathered_stats = gather_object_list(local_stats, world_size)
            else:
                gathered_stats = [local_stats]

            if is_main_process():
                merged = {
                    'num_samples': 0,
                    'psnr_sum': 0.0,
                    'ssim_sum': 0.0,
                    'lpips_sum': 0.0,
                    'clip_i_sum': 0.0,
                    'clip_t_sum': 0.0,
                    'clip_t_count': 0,
                    'pose_mpjpe_sum': 0.0,
                    'pose_pa_sum': 0.0,
                    'pose_count': 0,
                    'local_total': 0,
                }
                for item in gathered_stats:
                    for k in merged.keys():
                        merged[k] += item.get(k, 0)

                total_samples_final = int(merged['local_total'])
                rep_metrics = {}
                if merged['num_samples'] > 0:
                    rep_metrics['PSNR'] = float(merged['psnr_sum'] / merged['num_samples'])
                    rep_metrics['SSIM'] = float(merged['ssim_sum'] / merged['num_samples'])
                    rep_metrics['LPIPS'] = float(merged['lpips_sum'] / merged['num_samples'])
                    rep_metrics['CLIP-I'] = float(merged['clip_i_sum'] / merged['num_samples'])
                if merged['clip_t_count'] > 0:
                    rep_metrics['CLIP-T'] = float(merged['clip_t_sum'] / merged['clip_t_count'])
                if merged['pose_count'] > 0:
                    rep_metrics['PoseProxy_MPJPE'] = float(merged['pose_mpjpe_sum'] / merged['pose_count'] * 1000.0)
                    rep_metrics['PoseProxy_PA-MPJPE'] = float(merged['pose_pa_sum'] / merged['pose_count'] * 1000.0)

                key_metrics = [f"{k}={v:.4f}" for k, v in sorted(rep_metrics.items()) if isinstance(v, float)]
                if key_metrics:
                    print(f"  Rep {rep_i+1}: {', '.join(key_metrics)}")
                for key, value in rep_metrics.items():
                    if isinstance(value, (float, int)):
                        mgie_all_metrics.setdefault(key, []).append(float(value))

            if is_distributed:
                dist.barrier()

        all_results['mgie'] = {
            'total_samples': total_samples_final,
            'replication_times': replication_times,
            'per_rep_metrics': mgie_all_metrics,
        }

    # ---------- Edit: MotionFix-style motion editing evaluation ----------
    if 'edit' in tasks:
        if edit_dataloader is None:
            raise RuntimeError("Task 'edit' requested but edit_dataloader is not initialized.")

        replication_times = default_rep_per_task.get('edit', 1)
        edit_cfg = config.test.get('edit', {})
        edit_guidance_scale = float(edit_cfg.get('guidance_scale', 1.0))
        edit_num_steps = int(edit_cfg.get('num_inference_steps', 50))
        edit_latent_smooth_sigma = float(edit_cfg.get('latent_smooth_sigma', 1.5))
        retrieval_batch_size = int(edit_cfg.get('retrieval_batch_size', 32))
        edit_retr_diversity_times = int(edit_cfg.get('edit_retr_diversity_times', 300))

        if edit_guidance_scale > 1.0:
            if is_main_process():
                print("  [EDIT] guidance_scale > 1 is currently not supported stably, fallback to 1.0")
            edit_guidance_scale = 1.0

        if is_main_process():
            print(f"\n{'='*60}")
            print(f"  Evaluating: EDIT (replication_times={replication_times})")
            print(f"{'='*60}")
            print(f"  Guidance scale: {edit_guidance_scale}, Steps: {edit_num_steps}, "
                  f"Latent smooth sigma: {edit_latent_smooth_sigma}")

        edit_task_all_metrics = {}

        for rep_i in range(replication_times):
            if is_main_process():
                print(f"\n  --- Replication {rep_i + 1}/{replication_times} ---")

            evaluators['edit'].reset()
            set_seed(seed + rank + rep_i * 1000)
            save_this_rep = save_vis_manager.should_save_replication(rep_i)

            total_samples = 0
            edit_src_embs = []
            edit_tgt_embs = []
            edit_pred_embs = []
            edit_text_embs = []

            if is_distributed and hasattr(edit_dataloader, 'sampler') and isinstance(edit_dataloader.sampler, DistributedSampler):
                edit_dataloader.sampler.set_epoch(rep_i)

            pbar = tqdm(
                edit_dataloader,
                desc=f"edit Rep {rep_i+1}/{replication_times}",
                disable=not is_main_process()
            )

            for batch_idx, batch in enumerate(pbar):
                source_motion = zero_tail_dims(batch['source_motion'].to(device))
                target_motion = zero_tail_dims(batch['target_motion'].to(device))
                edit_texts = batch.get('edit_instruction', None)
                sample_ids = batch.get('sample_id', [])
                source_lengths = [int(x) for x in batch['source_motion_length'].tolist()]
                target_lengths = [int(x) for x in batch['target_motion_length'].tolist()]

                text_tokens_edit = batch['text_tokens'].to(device)
                source_positions = batch['source_modality_positions'].to(device)
                target_positions = batch['target_modality_positions'].to(device)

                pred_target_motions, _ = generate_edit(
                    model, motion_vae, sampler,
                    source_motion, target_motion,
                    source_lengths, target_lengths,
                    text_tokens_edit, source_positions, target_positions,
                    config, device, weight_type,
                    max_seq_len, downsample_rate, motion_latent_dim,
                    guidance_scale=edit_guidance_scale,
                    num_inference_steps=edit_num_steps,
                    debug=(batch_idx == 0 and rep_i == 0 and is_main_process()),
                    latent_smooth_sigma=edit_latent_smooth_sigma,
                )

                # ADE/FDE on edited target motion
                max_tgt_len = max(target_lengths)
                gt_eval = ensure_eval_motion_dim(target_motion)
                pred_eval = ensure_eval_motion_dim(pred_target_motions)

                if gt_eval.shape[1] > max_tgt_len:
                    gt_eval = gt_eval[:, :max_tgt_len, :]
                if pred_eval.shape[1] < max_tgt_len:
                    pad = torch.zeros(
                        pred_eval.shape[0], max_tgt_len - pred_eval.shape[1], pred_eval.shape[2],
                        device=pred_eval.device, dtype=pred_eval.dtype
                    )
                    pred_eval = torch.cat([pred_eval, pad], dim=1)
                elif pred_eval.shape[1] > max_tgt_len:
                    pred_eval = pred_eval[:, :max_tgt_len, :]

                gt_joints = feats2joints_humanml3d(gt_eval, mean=edit_mean_263, std=edit_std_263)
                pred_joints = feats2joints_humanml3d(pred_eval, mean=edit_mean_263, std=edit_std_263)

                evaluators['edit'].update(
                    joints_rst=pred_joints.float(),
                    joints_ref=gt_joints.float(),
                    lengths=target_lengths,
                )
                if save_this_rep:
                    for i in range(pred_target_motions.shape[0]):
                        sample_id = sample_ids[i] if i < len(sample_ids) else f"sample_{batch_idx:05d}_{i:03d}"
                        motion_len_i = int(target_lengths[i]) if i < len(target_lengths) else int(pred_target_motions.shape[1])
                        motion_len_i = max(1, min(motion_len_i, int(pred_target_motions.shape[1])))
                        save_vis_manager.save_motion_sample(
                            task='edit',
                            replication=rep_i,
                            batch_idx=batch_idx,
                            in_batch_idx=i,
                            sample_id=sample_id,
                            motion=pred_target_motions[i, :motion_len_i],
                            keep_dim=263,
                            extra={
                                'edit_instruction': edit_texts[i] if edit_texts is not None and i < len(edit_texts) else "",
                                'target_length': motion_len_i,
                            },
                        )

                if edit_embedder is not None:
                    src_emb = edit_embedder.get_motion_embedding(
                        ensure_eval_motion_dim(source_motion).float(), source_lengths
                    ).cpu()
                    tgt_emb = edit_embedder.get_motion_embedding(
                        ensure_eval_motion_dim(target_motion).float(), target_lengths
                    ).cpu()
                    pred_emb = edit_embedder.get_motion_embedding(
                        ensure_eval_motion_dim(pred_target_motions).float(), target_lengths
                    ).cpu()
                    edit_src_embs.append(src_emb)
                    edit_tgt_embs.append(tgt_emb)
                    edit_pred_embs.append(pred_emb)
                if edit_text_embedder is not None and edit_texts is not None:
                    text_emb = edit_text_embedder._get_text_embeddings_from_sentences(list(edit_texts)).cpu()
                    edit_text_embs.append(text_emb)

                total_samples += source_motion.shape[0]
                pbar.set_postfix({'samples': total_samples})

            if is_distributed:
                dist.barrier()
                total_samples_tensor = torch.tensor([total_samples], device=device)
                dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
                total_samples = int(total_samples_tensor.item())

                all_src_embs = gather_embeddings(edit_src_embs, world_size) if len(edit_src_embs) > 0 else None
                all_tgt_embs = gather_embeddings(edit_tgt_embs, world_size) if len(edit_tgt_embs) > 0 else None
                all_pred_embs = gather_embeddings(edit_pred_embs, world_size) if len(edit_pred_embs) > 0 else None
                all_text_embs = gather_embeddings(edit_text_embs, world_size) if len(edit_text_embs) > 0 else None
            else:
                all_src_embs = torch.cat(edit_src_embs, dim=0) if len(edit_src_embs) > 0 else None
                all_tgt_embs = torch.cat(edit_tgt_embs, dim=0) if len(edit_tgt_embs) > 0 else None
                all_pred_embs = torch.cat(edit_pred_embs, dim=0) if len(edit_pred_embs) > 0 else None
                all_text_embs = torch.cat(edit_text_embs, dim=0) if len(edit_text_embs) > 0 else None

            if is_main_process():
                rep_metrics = evaluators['edit'].compute()

                if all_src_embs is not None and all_tgt_embs is not None and all_pred_embs is not None:
                    src_all = all_src_embs
                    tgt_all = all_tgt_embs
                    pred_all = all_pred_embs
                    retrieval_metrics = compute_motionfix_style_retrieval_metrics(
                        source_emb=src_all,
                        target_emb=tgt_all,
                        pred_emb=pred_all,
                        batch_eval_size=retrieval_batch_size,
                        seed=seed + rep_i,
                    )
                    rep_metrics.update(retrieval_metrics)

                if all_src_embs is not None and all_tgt_embs is not None and all_pred_embs is not None:
                    edit_retr_metrics = compute_edit_retrieval_metrics(
                        source_emb=all_src_embs,
                        target_emb=all_tgt_embs,
                        pred_emb=all_pred_embs,
                        batch_eval_size=retrieval_batch_size,
                        seed=seed + rep_i,
                        diversity_times=edit_retr_diversity_times,
                    )
                    rep_metrics.update(edit_retr_metrics)

                key_metrics = []
                for key, value in sorted(rep_metrics.items()):
                    if isinstance(value, float):
                        key_metrics.append(f"{key}={value:.4f}")
                if key_metrics:
                    print(f"  Rep {rep_i+1}: {', '.join(key_metrics[:8])}")

                for key, value in rep_metrics.items():
                    if key not in edit_task_all_metrics:
                        edit_task_all_metrics[key] = [value]
                    else:
                        edit_task_all_metrics[key].append(value)

            if is_distributed:
                dist.barrier()

        all_results['edit'] = {
            'total_samples': total_samples,
            'replication_times': replication_times,
            'per_rep_metrics': edit_task_all_metrics,
        }
    
    # ==================== [5/5] Compute Statistics and Save Results ====================
    if not is_main_process():
        if save_vis_manager.enabled and is_distributed:
            dist.barrier()
            dist.barrier()
        cleanup_distributed()
        return
    
    print(f"\n[5/5] Computing final statistics...")
    
    timestamp = run_timestamp
    final_results = {}
    
    # Print final results
    print("\n" + "=" * 70)
    mode_str = "QUICK TEST" if quick_mode else "FULL EVALUATION"
    print(f"{eval_stage_label} {mode_str} RESULTS (mean +/- 95% CI)")
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
                'R_Precision_Top_1', 'R_Precision_Top_2', 'R_Precision_Top_3',
                'Matching_Score', 'gt_Matching_Score',
                'ROUGE_L', 'CIDEr', 'METEOR', 'bleu_1', 'bleu_4', 'Bert_F1',
            ]
            up_metrics = {'R_Precision_Top_1', 'R_Precision_Top_2', 'R_Precision_Top_3',
                         'ROUGE_L', 'CIDEr', 'METEOR', 'bleu_1', 'bleu_4', 'Bert_F1'}
        elif task == 'pred':
            metric_order = ['ADE', 'FDE', 'APD']
            up_metrics = set()  # lower is better for ADE/FDE
        elif task == 'img2pose':
            metric_order = [
                'MPJPE', 'PA-MPJPE',
                'MPJPE_269GT', 'PA-MPJPE_269GT',
                'GT_RENORM_MPJPE', 'GT_RENORM_PA-MPJPE',
            ]
            up_metrics = set()
        elif task in ['t2t', 'img2text']:
            metric_order = ['Bleu_1', 'Bleu_4', 'ROUGE_L', 'CIDEr', 'METEOR', 'Bert_F1']
            up_metrics = {'Bleu_1', 'Bleu_4', 'ROUGE_L', 'CIDEr', 'METEOR', 'Bert_F1'}
        elif task == 'mgie':
            metric_order = [
                'PSNR', 'SSIM', 'LPIPS', 'CLIP-I', 'CLIP-T',
                'PoseProxy_MPJPE', 'PoseProxy_PA-MPJPE',
            ]
            up_metrics = {'PSNR', 'SSIM', 'CLIP-I', 'CLIP-T'}
        elif task == 'edit':
            metric_order = [
                'ADE', 'FDE', 'APD',
                # Edit retrieval metrics (protocol of MotionFix paper, Table 8)
                'edit_retr_Top1', 'edit_retr_Top2', 'edit_retr_Top3',
                'edit_retr_FID', 'edit_retr_MM_Dist', 'edit_retr_Diversity',
                'edit_retr_real_Top1', 'edit_retr_real_Top2', 'edit_retr_real_Top3',
                'edit_retr_real_MM_Dist', 'edit_retr_real_Diversity',
                'edit_retr_full_Top1', 'edit_retr_full_Top2', 'edit_retr_full_Top3',
                'edit_retr_full_MM_Dist',
                'edit_retr_real_full_Top1', 'edit_retr_real_full_Top2', 'edit_retr_real_full_Top3',
                'edit_retr_real_full_MM_Dist',
                # Existing MotionFix-style retrieval
                'R@1_s2t', 'R@2_s2t', 'R@3_s2t',
                'R@1', 'R@2', 'R@3',
                'R@1_s2t_full', 'R@2_s2t_full', 'R@3_s2t_full',
                'R@1_full', 'R@2_full', 'R@3_full',
            ]
            up_metrics = {
                'edit_retr_Top1', 'edit_retr_Top2', 'edit_retr_Top3',
                'edit_retr_real_Top1', 'edit_retr_real_Top2', 'edit_retr_real_Top3',
                'edit_retr_full_Top1', 'edit_retr_full_Top2', 'edit_retr_full_Top3',
                'edit_retr_real_full_Top1', 'edit_retr_real_full_Top2', 'edit_retr_real_full_Top3',
                'R@1_s2t', 'R@2_s2t', 'R@3_s2t',
                'R@1', 'R@2', 'R@3',
                'R@1_s2t_full', 'R@2_s2t_full', 'R@3_s2t_full',
                'R@1_full', 'R@2_full', 'R@3_full',
            }
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
    
    json_path = os.path.join(output_dir, f"{eval_result_prefix}_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")
    
    # Save text summary
    txt_path = os.path.join(output_dir, f"{eval_result_prefix}_{timestamp}.txt")
    with open(txt_path, 'w') as f:
        f.write(f"{eval_title} Results\n")
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
            f.write("-" * 110 + "\n")
            f.write(f"{'METHOD':<15} {'R@1':<8} {'R@2':<8} {'R@3':<8} {'MMDist↓':<10} "
                    f"{'Bleu@1↑':<10} {'Bleu@4↑':<10} {'Rouge↑':<10} {'CIDEr↑':<10} {'METEOR↑':<10} {'BertScore↑':<10}\n")
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
            for k in ['bleu_1', 'bleu_4', 'ROUGE_L', 'CIDEr', 'METEOR', 'Bert_F1']:
                if k in m2t_metrics:
                    f.write(f"{np.mean(m2t_metrics[k]) * 100:<10.3f} ")
                else:
                    f.write(f"{'N/A':<10} ")
            f.write("\n")
            f.write(f"{'MotionGPT3':<15} {'0.553':<8} {'0.756':<8} {'0.853':<8} {'2.524':<10} "
                    f"{'56.363':<10} {'17.661':<10} {'44.997':<10} {'30.980':<10} {'N/A':<10} {'35.850':<10}\n")
            f.write("-" * 110 + "\n")
        print("  M2T comparison table appended to txt file")
    
    if save_vis_manager.enabled:
        finalized_tasks = [t for t in tasks if t in all_results]
        if (
            't2m' in finalized_tasks and
            enable_t2m_sigma_smoothed_save and
            t2m_sigma_smoothed_task_name not in finalized_tasks
        ):
            finalized_tasks.append(t2m_sigma_smoothed_task_name)
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
                    'save_t2m_sigma_smoothed_motion': bool(save_t2m_sigma_smoothed_motion),
                    't2m_sigma_smoothed_sigma': float(t2m_sigma_smoothed_sigma),
                    't2m_sigma_smoothed_task_name': t2m_sigma_smoothed_task_name,
                },
                'seed': seed,
                'quick': bool(quick_mode),
                'batch_size': int(config.test.get('batch_size', 32)),
                'model_path': model_path,
                'replication_times_per_task': {t: default_rep_per_task.get(t, 1) for t in tasks},
                'task_settings': {t: dict(config.test.get(t, {}) or {}) for t in tasks},
            }
        )
        print(f"Save-vis outputs written to: {save_vis_manager.run_dir}")
        if is_distributed:
            dist.barrier()
    
    print("\nEvaluation complete!")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
