# coding=utf-8
"""
H36M MGIE evaluation dataset.
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.h36m_dataset import get_image_transform
from datasets.h36m_mgie_dataset import load_mgie_instruction_templates, _strip_mgie_placeholders


class H36MMGIEEvalDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        image_root: str,
        split: str = "test",
        image_resolution: int = 432,
        instruction_template_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.data_root = data_root
        self.image_root = image_root
        self.image_transform = get_image_transform(image_resolution)

        manifest_path = os.path.join(data_root, "manifests", f"{split}.jsonl")
        motion_norm_path = os.path.join(data_root, "arrays", f"motions_norm_{split}.npy")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"MGIE eval manifest not found: {manifest_path}")
        if not os.path.exists(motion_norm_path):
            raise FileNotFoundError(f"MGIE eval motions_norm not found: {motion_norm_path}")

        samples = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
        motions_norm = np.load(motion_norm_path).astype(np.float32, copy=False)
        if len(samples) != motions_norm.shape[0]:
            raise ValueError(f"MGIE eval length mismatch: {len(samples)} vs {motions_norm.shape[0]}")

        if max_samples is not None and max_samples < len(samples):
            rng = np.random.RandomState(seed)
            idx = sorted(rng.choice(len(samples), max_samples, replace=False).tolist())
            samples = [samples[i] for i in idx]
            motions_norm = motions_norm[idx]

        self.samples = samples
        self.motions_norm = motions_norm

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
                "camera viewpoint, lighting, and background unchanged; change only the human pose."
            ]

        print(f"\n{'='*20} H36M MGIE Eval Dataset Info {'='*20}")
        print(f"Split: {split}")
        print(f"Samples: {len(self.samples)}")
        print(f"Instructions: {len(self.instructions)}")
        print(f"{'='*60}\n")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, rel_path: str) -> torch.Tensor:
        full_path = os.path.join(self.image_root, rel_path)
        img = Image.open(full_path).convert("RGB")
        return self.image_transform(img)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        source_img = self._load_image(s["source_image_relpath"])
        target_img = self._load_image(s["target_image_relpath"])

        # [2, 269]: index 0=source pose, index 1=target pose
        motion_pair = self.motions_norm[idx]
        source_motion = torch.from_numpy(motion_pair[0].copy()).float()
        target_motion = torch.from_numpy(motion_pair[1].copy()).float()

        template = self.instructions[idx % len(self.instructions)]
        instruction_text = _strip_mgie_placeholders(template)

        return {
            "source_image": source_img,
            "target_image": target_img,
            "source_motion_norm": source_motion,
            "ref_motion_norm": target_motion,
            "target_motion_norm": target_motion,
            "instruction_text": instruction_text,
            "source_image_path": s["source_image_relpath"],
            "target_image_path": s["target_image_relpath"],
            "sample_id": s.get("sample_id", f"mgie_{idx:09d}"),
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = {
            "source_images": torch.stack([x["source_image"] for x in batch], dim=0),
            "target_images": torch.stack([x["target_image"] for x in batch], dim=0),
            "source_motions_norm": torch.stack([x["source_motion_norm"] for x in batch], dim=0),
            "ref_motions_norm": torch.stack([x["ref_motion_norm"] for x in batch], dim=0),
            "target_motions_norm": torch.stack([x["target_motion_norm"] for x in batch], dim=0),
            "instruction_texts": [x["instruction_text"] for x in batch],
            "source_image_paths": [x["source_image_path"] for x in batch],
            "target_image_paths": [x["target_image_path"] for x in batch],
            "sample_ids": [x["sample_id"] for x in batch],
        }
        return out
