# coding=utf-8
"""
Motion reconstruction dataset for Stage0 (m2m).

This dataset provides source->target motion pairs for reconstruction:
- source_motion: bottlenecked motion condition (no time embedding token)
- target_motion: original motion to reconstruct (with time embedding token)
"""

import collections
import json
import os
import random
import re
from typing import Any, Dict, List, Optional

import codecs as cs
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.utils import format_sequence_motion_pred
from motion_length_utils import motion_token_slots


def _load_recon_instructions(template_path: Optional[str]) -> List[str]:
    if template_path and os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        prompts: List[str] = []
        for _, task_dict in data.items():
            for _, subtask in task_dict.items():
                prompts.extend(subtask.get("input", []))
        if prompts:
            return prompts

    return [
        "Reconstruct the complete motion sequence from the given motion cue: <Motion_Placeholder_s1>",
        "Recover the original full-body movement from this compressed motion signal: <Motion_Placeholder_s1>",
        "Use the provided motion hint to regenerate the full motion: <Motion_Placeholder_s1>",
        "Infer and reconstruct the full human motion from this partial motion evidence: <Motion_Placeholder_s1>",
    ]


def _cleanup_prompt(prompt: str) -> str:
    prompt = prompt.replace("<Motion_Placeholder_s1>", " ").strip()
    prompt = re.sub(r"\s+", " ", prompt)
    return prompt if prompt else "Reconstruct the full motion."


class MotionReconDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        text_tokenizer: Any,
        mean_path: str = None,
        std_path: str = None,
        max_seq_len: int = 512,
        max_motion_length: int = 200,
        min_motion_length: int = 20,
        unit_length: int = 1,
        downsample_rate: int = 1,
        showo_token_ids: Optional[Dict[str, int]] = None,
        split: str = "train",
        tiny: bool = False,
        debug: bool = False,
        add_time_embeds: bool = True,
        instruction_template_path: str = None,
        bottleneck: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        super().__init__()

        self.data_root = data_root
        self.dataset_name = dataset_name.lower()
        self.text_tokenizer = text_tokenizer
        self.max_seq_len = max_seq_len
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = max(1, unit_length)
        self.downsample_rate = max(1, downsample_rate)
        self.add_time_embeds = add_time_embeds
        self.data_type = "m2m"

        self.instructions = _load_recon_instructions(instruction_template_path)

        # Token ids
        self.pad_id = text_tokenizer.pad_token_id
        self.bos_id = showo_token_ids["bos_id"]
        self.eos_id = showo_token_ids["eos_id"]
        self.bom_id = showo_token_ids.get("bom_id", showo_token_ids["boi_id"])
        self.eom_id = showo_token_ids.get("eom_id", showo_token_ids["eoi_id"])
        self.mot_pad_id = showo_token_ids.get("mot_pad_id", showo_token_ids["img_pad_id"])

        # Normalization
        if mean_path and os.path.exists(mean_path):
            self.mean = np.load(mean_path)
        else:
            self.mean = np.load(os.path.join(data_root, "Mean.npy"))
        if std_path and os.path.exists(std_path):
            self.std = np.load(std_path)
        else:
            self.std = np.load(os.path.join(data_root, "Std.npy"))
        self.std_safe = self.std.copy()
        self.std_safe[self.std_safe == 0] = 1.0

        # Bottleneck config
        bottleneck = bottleneck or {}
        self.temporal_keep_ratio_min = float(bottleneck.get("temporal_keep_ratio_min", 0.2))
        self.temporal_keep_ratio_max = float(bottleneck.get("temporal_keep_ratio_max", 0.5))
        self.feature_drop_prob = float(bottleneck.get("feature_drop_prob", 0.15))
        self.noise_std = float(bottleneck.get("noise_std", 0.02))

        self.samples: List[Dict[str, Any]] = []
        self.h36m_motions = None
        self._load_data(split=split, tiny=tiny, debug=debug)

    def _load_data(self, split: str, tiny: bool, debug: bool):
        split_file = os.path.join(self.data_root, f"{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        if self.dataset_name == "humanml3d":
            motion_dir = os.path.join(self.data_root, "new_joint_vecs")
            with cs.open(split_file, "r") as f:
                ids = [line.strip() for line in f.readlines() if line.strip()]

            for sample_id in ids:
                motion_path = os.path.join(motion_dir, f"{sample_id}.npy")
                if not os.path.exists(motion_path):
                    continue
                try:
                    length = np.load(motion_path, mmap_mode="r").shape[0]
                except Exception:
                    continue
                if length < self.min_motion_length or length >= self.max_motion_length:
                    continue
                self.samples.append({"id": sample_id, "motion_path": motion_path})

        elif self.dataset_name in {"motionfix_target", "motionfix"}:
            target_dir = os.path.join(self.data_root, "new_joint_vecs", "target")
            with cs.open(split_file, "r") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            for line in lines:
                sample_id = line.split("/")[-1]
                motion_path = os.path.join(target_dir, f"{sample_id}.npy")
                if not os.path.exists(motion_path):
                    continue
                try:
                    length = np.load(motion_path, mmap_mode="r").shape[0]
                except Exception:
                    continue
                if length < self.min_motion_length or length >= self.max_motion_length:
                    continue
                self.samples.append({"id": sample_id, "motion_path": motion_path})

        elif self.dataset_name == "h36m":
            with cs.open(split_file, "r") as f:
                names = [line.strip() for line in f.readlines() if line.strip()]
            if not names:
                raise ValueError(f"Invalid H36M split file: {split_file}")

            data_name = names[0]
            motion_path = os.path.join(self.data_root, "new_joint_vecs", f"{data_name}.npy")
            self.h36m_motions = np.load(motion_path)
            for i in range(self.h36m_motions.shape[0]):
                self.samples.append({"id": f"{data_name}_{i}", "frame_idx": i})
        else:
            raise ValueError(f"Unsupported dataset_name: {self.dataset_name}")

        if tiny or debug:
            self.samples = self.samples[: min(1000, len(self.samples))]

        print(
            f"[MotionReconDataset] loaded {len(self.samples)} samples "
            f"from {self.dataset_name} ({split}), bottleneck="
            f"({self.temporal_keep_ratio_min:.2f},{self.temporal_keep_ratio_max:.2f})/"
            f"drop={self.feature_drop_prob:.2f}/noise={self.noise_std:.3f}"
        )
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found for MotionReconDataset: {self.dataset_name}")

    def _compute_num_motion_tokens(self, motion_length: int, add_time_embed: bool) -> int:
        return motion_token_slots(
            motion_length,
            self.downsample_rate,
            add_time_embeds=add_time_embed,
        )

    def _apply_bottleneck(self, target_motion: torch.Tensor) -> torch.Tensor:
        # target_motion: [T, D], already normalized
        source = target_motion.clone()
        t_len = source.shape[0]

        # Temporal bottleneck only for multi-frame sequences
        if t_len > 1 and self.temporal_keep_ratio_max > 0:
            keep_ratio = random.uniform(self.temporal_keep_ratio_min, self.temporal_keep_ratio_max)
            keep_len = max(1, int(round(t_len * keep_ratio)))
            if keep_len < t_len:
                keep_ids = torch.linspace(0, t_len - 1, keep_len).round().long()
                source = source[keep_ids]

        # Feature bottleneck
        if self.feature_drop_prob > 0:
            drop_mask = torch.rand_like(source) < self.feature_drop_prob
            source = source.masked_fill(drop_mask, 0.0)

        # Additive noise
        if self.noise_std > 0:
            source = source + torch.randn_like(source) * self.noise_std

        return source

    def __len__(self):
        return len(self.samples)

    def _load_target_motion(self, sample: Dict[str, Any]) -> np.ndarray:
        if self.dataset_name == "h36m":
            frame_idx = sample["frame_idx"]
            frame = self.h36m_motions[frame_idx]  # [269]
            return frame[None, :].copy()
        return np.load(sample["motion_path"])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        target_motion_np = self._load_target_motion(sample)

        # Enforce length constraints
        target_len = target_motion_np.shape[0]
        if target_len > 1:
            target_len = (target_len // self.unit_length) * self.unit_length
            target_motion_np = target_motion_np[:target_len]

        if target_motion_np.shape[0] < self.min_motion_length or target_motion_np.shape[0] > self.max_motion_length:
            return self.__getitem__((idx + 1) % len(self.samples))

        # Normalize target and construct bottlenecked source
        target_motion = ((target_motion_np - self.mean) / self.std_safe).astype(np.float32)
        target_motion = torch.from_numpy(target_motion)
        source_motion = self._apply_bottleneck(target_motion)

        num_source_tokens = self._compute_num_motion_tokens(len(source_motion), add_time_embed=False)
        num_target_tokens = self._compute_num_motion_tokens(len(target_motion), add_time_embed=self.add_time_embeds)

        prompt = _cleanup_prompt(random.choice(self.instructions))
        text_tokens = self.text_tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=96,
        ).input_ids

        (
            text_tokens_tensor,
            text_labels,
            source_modality_positions,
            target_modality_positions,
            text_mask,
            source_motion_mask,
            target_motion_mask,
        ) = format_sequence_motion_pred(
            text_tokens,
            self.bos_id,
            self.eos_id,
            self.bom_id,
            self.eom_id,
            self.pad_id,
            self.mot_pad_id,
            num_source_tokens,
            num_target_tokens,
            self.max_seq_len,
        )

        return {
            "text_tokens": text_tokens_tensor,
            "text_labels": text_labels,
            "source_motion": source_motion,
            "target_motion": target_motion,
            "source_motion_length": torch.tensor(len(source_motion)),
            "target_motion_length": torch.tensor(len(target_motion)),
            "source_modality_positions": source_modality_positions,
            "target_modality_positions": target_modality_positions,
            "text_masks": text_mask,
            "source_motion_masks": source_motion_mask,
            "target_motion_masks": target_motion_mask,
            "data_type": self.data_type,
            "dataset_name": self.dataset_name,
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batched = collections.defaultdict(list)

        max_source_len = max(sample["source_motion"].shape[0] for sample in batch)
        max_target_len = max(sample["target_motion"].shape[0] for sample in batch)
        if max_source_len % self.unit_length != 0:
            max_source_len = ((max_source_len // self.unit_length) + 1) * self.unit_length
        if max_target_len % self.unit_length != 0:
            max_target_len = ((max_target_len // self.unit_length) + 1) * self.unit_length

        for item in batch:
            for key, value in item.items():
                if key == "source_motion":
                    pad_len = max_source_len - value.shape[0]
                    if pad_len > 0:
                        value = torch.cat([value, torch.zeros(pad_len, value.shape[1])], dim=0)
                    batched[key].append(value)
                elif key == "target_motion":
                    pad_len = max_target_len - value.shape[0]
                    if pad_len > 0:
                        value = torch.cat([value, torch.zeros(pad_len, value.shape[1])], dim=0)
                    batched[key].append(value)
                else:
                    batched[key].append(value)

        result = {}
        for key, value in batched.items():
            if key in {"data_type", "dataset_name"}:
                result[key] = value
            else:
                result[key] = torch.stack(value, dim=0)
        return result
