# coding=utf-8
"""
H36M MGIE Dataset for Stage3b.

Task:
  text + source image + reference motion(target-pose) -> target image

Dataset layout:
  - manifests/{split}.jsonl
  - arrays/motions_norm_{split}.npy  # [N, 2, 269]
  - prompts/mgie_template_instructions.json
"""

import json
import os
import random
import collections
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets.h36m_dataset import get_image_transform
from datasets.utils import format_sequence_mgie


def load_mgie_instruction_templates(template_path: str) -> List[str]:
    if not template_path or not os.path.exists(template_path):
        return []
    with open(template_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    templates = []
    if isinstance(data, dict):
        for group_data in data.values():
            if not isinstance(group_data, dict):
                continue
            for task_data in group_data.values():
                if not isinstance(task_data, dict):
                    continue
                if task_data.get("class") == "motion_guided_image_edit":
                    templates.extend(task_data.get("input", []))
    return templates


def _strip_mgie_placeholders(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("<Source_Image_Placeholder>", "")
    text = text.replace("<Reference_Motion_Placeholder>", "")
    text = text.strip()
    while "  " in text:
        text = text.replace("  ", " ")
    return text


class H36MMGIEDataset(Dataset):
    """
    Training dataset for Motion-guided Image Editing (MGIE).
    """

    def __init__(
        self,
        data_root: str,
        image_root: str,
        text_tokenizer: Any,
        showo_token_ids: Dict[str, int],
        max_seq_len: int = 1024,
        num_image_tokens: int = 729,
        image_resolution: int = 432,
        instruction_template_path: Optional[str] = None,
        split: str = "train",
        add_time_embeds: bool = True,
        max_text_len: int = 256,
        tiny: bool = False,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.data_root = data_root
        self.image_root = image_root
        self.text_tokenizer = text_tokenizer
        self.max_seq_len = max_seq_len
        self.num_image_tokens = int(num_image_tokens)
        self.max_text_len = int(max_text_len)
        self.image_transform = get_image_transform(image_resolution)
        self.data_type = "mgie"

        self.pad_id = text_tokenizer.pad_token_id
        self.bos_id = showo_token_ids["bos_id"]
        self.eos_id = showo_token_ids["eos_id"]
        self.boi_id = showo_token_ids["boi_id"]
        self.eoi_id = showo_token_ids["eoi_id"]
        self.img_pad_id = showo_token_ids["img_pad_id"]
        self.bom_id = showo_token_ids["bom_id"]
        self.eom_id = showo_token_ids["eom_id"]
        self.mot_pad_id = showo_token_ids["mot_pad_id"]

        # Keep 1 latent + optional time-slot for single-frame reference motion.
        # Under downsample_rate=4, the Motion VAE still maps length-1 input to 1 latent
        # via internal zero-padding followed by decode-time slicing.
        self.num_ref_motion_tokens = 2 if add_time_embeds else 1
        # Target image generation needs one extra slot when time embeddings are enabled.
        self.num_target_image_tokens = self.num_image_tokens + 1 if add_time_embeds else self.num_image_tokens

        manifests_dir = os.path.join(data_root, "manifests")
        arrays_dir = os.path.join(data_root, "arrays")
        manifest_path = os.path.join(manifests_dir, f"{split}.jsonl")
        motion_norm_path = os.path.join(arrays_dir, f"motions_norm_{split}.npy")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"MGIE manifest not found: {manifest_path}")
        if not os.path.exists(motion_norm_path):
            raise FileNotFoundError(f"MGIE motions_norm not found: {motion_norm_path}")

        self.samples = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))
        self.motions_norm = np.load(motion_norm_path).astype(np.float32, copy=False)

        if len(self.samples) != self.motions_norm.shape[0]:
            raise ValueError(
                f"MGIE manifest/motions length mismatch: {len(self.samples)} vs {self.motions_norm.shape[0]}"
            )
        if self.motions_norm.shape[1:] != (2, 269):
            raise ValueError(f"Unexpected motions_norm shape tail: {self.motions_norm.shape[1:]}")

        if tiny or debug:
            keep_n = min(len(self.samples), 2000)
            self.samples = self.samples[:keep_n]
            self.motions_norm = self.motions_norm[:keep_n]

        if instruction_template_path is None:
            instruction_template_path = os.path.join(
                data_root, "prompts", "mgie_template_instructions.json"
            )
        self.instructions = load_mgie_instruction_templates(instruction_template_path)
        if not self.instructions:
            self.instructions = [
                "Given source image <Source_Image_Placeholder> and reference motion "
                "<Reference_Motion_Placeholder>, edit the person's pose to match the motion. "
                "Keep the same person identity, face, hair, body shape, clothing, clothing texture, "
                "camera viewpoint, lighting, and background unchanged; change only the human pose.",
            ]

        print(f"\n{'='*20} H36M MGIE Dataset Info {'='*20}")
        print(f"Split: {split}")
        print(f"Samples: {len(self.samples)}")
        print(f"Image tokens: {self.num_image_tokens}")
        print(f"Ref motion tokens: {self.num_ref_motion_tokens}")
        print(f"Target image slots: {self.num_target_image_tokens}")
        print(f"Instructions: {len(self.instructions)}")
        print(f"{'='*60}\n")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, rel_path: str) -> torch.Tensor:
        full_path = os.path.join(self.image_root, rel_path)
        img = Image.open(full_path).convert("RGB")
        return self.image_transform(img)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        try:
            source_img = self._load_image(sample["source_image_relpath"])
            target_img = self._load_image(sample["target_image_relpath"])
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

        ref_motion = self.motions_norm[idx, 1].copy()
        ref_motion = torch.from_numpy(ref_motion).float().unsqueeze(0)  # [1, 269]

        template = random.choice(self.instructions)
        instruction_text = _strip_mgie_placeholders(template)
        instr_tokens = self.text_tokenizer(
            instruction_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_text_len,
        ).input_ids

        (
            text_tokens,
            text_labels,
            src_img_pos,
            ref_mot_pos,
            tgt_img_pos,
            text_mask,
            src_img_mask,
            ref_mot_mask,
            tgt_img_mask,
        ) = format_sequence_mgie(
            instr_tokens=instr_tokens,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            boi_id=self.boi_id,
            eoi_id=self.eoi_id,
            bom_id=self.bom_id,
            eom_id=self.eom_id,
            pad_id=self.pad_id,
            img_pad_id=self.img_pad_id,
            mot_pad_id=self.mot_pad_id,
            num_source_image_tokens=self.num_image_tokens,
            num_ref_motion_tokens=self.num_ref_motion_tokens,
            num_target_image_tokens=self.num_target_image_tokens,
            max_seq_len=self.max_seq_len,
        )

        return {
            "text_tokens": text_tokens,
            "text_labels": text_labels,
            "text_masks": text_mask,
            "source_image": source_img,
            "target_image": target_img,
            "ref_motion": ref_motion,
            "source_image_masks": src_img_mask,
            "target_image_masks": tgt_img_mask,
            "ref_motion_masks": ref_mot_mask,
            "source_image_modality_positions": src_img_pos,
            "ref_motion_modality_positions": ref_mot_pos,
            "target_image_modality_positions": tgt_img_pos,
            "instruction_text": instruction_text,
            "source_image_path": sample["source_image_relpath"],
            "target_image_path": sample["target_image_relpath"],
            "data_type": self.data_type,
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batched = collections.defaultdict(list)
        for item in batch:
            for k, v in item.items():
                batched[k].append(v)

        result = {}
        for k, v in batched.items():
            if k in ("instruction_text", "source_image_path", "target_image_path", "data_type"):
                result[k] = v
            else:
                result[k] = torch.stack(v, dim=0)
        return result


def create_h36m_mgie_dataloader(
    data_root: str,
    image_root: str,
    batch_size: int,
    text_tokenizer: Any,
    showo_token_ids: Dict[str, int],
    accelerator: Optional[Any] = None,
    max_seq_len: int = 1024,
    num_image_tokens: int = 729,
    image_resolution: int = 432,
    instruction_template_path: Optional[str] = None,
    num_workers: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
    split: str = "train",
    add_time_embeds: bool = True,
    max_text_len: int = 256,
    tiny: bool = False,
    debug: bool = False,
    **kwargs,
) -> DataLoader:
    dataset = H36MMGIEDataset(
        data_root=data_root,
        image_root=image_root,
        text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids,
        max_seq_len=max_seq_len,
        num_image_tokens=num_image_tokens,
        image_resolution=image_resolution,
        instruction_template_path=instruction_template_path,
        split=split,
        add_time_embeds=add_time_embeds,
        max_text_len=max_text_len,
        tiny=tiny,
        debug=debug,
    )

    if accelerator is not None and accelerator.num_processes > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=dataset.collate_fn,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,
        persistent_workers=bool(num_workers > 0),
    )
