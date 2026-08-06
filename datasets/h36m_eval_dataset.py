# coding=utf-8
"""
H36M Evaluation Dataset for Stage 3 testing.

Loads H36M test-set images, GT motions, and GT texts for img2pose and img2text evaluation.

Data layout:
- Motion data: single npy file, shape=(N_frames, 269)
- Image paths: single npy file, shape=(N_frames,), frame-aligned with the motion
- Text descriptions: nested directories text/S{id}/sequence_dir/frame.txt
- Mean.npy, Std.npy: normalization statistics

Each sample is a single-frame (image, motion_frame, text_description) triplet.
"""

import os
import json
import random
import collections
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Reuse functions from h36m_dataset
from datasets.h36m_dataset import image_path_to_text_path, load_text_description

import re


def image_path_to_text_path_fallback(image_rel_path: str, text_root: str) -> Optional[str]:
    """
    Extended text path mapping with multiple fallback strategies.
    
    The original mapping (offset +4, no trial number) works for training data.
    For test data (S9, S11), different strategies may be needed:
    - Strategy 1: Remove trial number, offset +4 (original)
    - Strategy 2: Remove trial number, offset +6 (test data)
    - Strategy 3: Keep trial number, offset 0 (direct match)
    """
    try:
        parts = image_rel_path.split('/')
        subject = parts[0]
        img_dir_name = parts[1]
        img_file_name = parts[2]
        
        frame_str = img_file_name.rsplit('_', 1)[-1].replace('.jpg', '')
        frame_num = int(frame_str)
        
        dot_pos = img_dir_name.rfind('.')
        if dot_pos == -1:
            return None
        base_part = img_dir_name[:dot_pos]
        camera_id = img_dir_name[dot_pos + 1:]
        
        match = re.match(r'^(.+)_(\d+)$', base_part)
        text_base = match.group(1) if match else base_part
        
        # Strategy 1: original (offset +4, no trial number)
        text_dir_1 = f"{text_base}.{camera_id}"
        text_file_1 = f"{text_base}.{camera_id}_{frame_num + 4:06d}.txt"
        path_1 = os.path.join(text_root, subject, text_dir_1, text_file_1)
        if os.path.exists(path_1):
            return path_1
        
        # Strategy 2: offset +6, no trial number
        text_file_2 = f"{text_base}.{camera_id}_{frame_num + 6:06d}.txt"
        path_2 = os.path.join(text_root, subject, text_dir_1, text_file_2)
        if os.path.exists(path_2):
            return path_2
        
        # Strategy 3: keep trial number, offset 0 (direct match)
        text_file_3 = f"{img_dir_name}_{frame_num:06d}.txt"
        path_3 = os.path.join(text_root, subject, img_dir_name, text_file_3)
        if os.path.exists(path_3):
            return path_3
        
        return None
    except Exception:
        return None


def get_eval_image_transform(resolution: int = 432):
    """Image preprocessing for evaluation (consistent with training)."""
    return transforms.Compose([
        transforms.Resize((resolution, resolution),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


class H36MEvalDataset(Dataset):
    """
    H36M Evaluation Dataset for Stage 3 tasks: img2pose and img2text.
    
    Used for testing/evaluation; returns raw data for the evaluators without sequence formatting.
    """
    
    def __init__(
        self,
        data_root: str,
        image_root: str,
        split: str = 'test',
        image_resolution: int = 432,
        max_samples: int = None,
        seed: int = 42,
        require_text: bool = False,
        manifest_path: Optional[str] = None,
    ):
        """
        Args:
            data_root: H36M dataset root directory
            image_root: H36M raw image root directory
            split: dataset split ('test' or 'train')
            image_resolution: target image resolution
            max_samples: maximum number of samples (None = all)
            seed: random seed (for subset sampling)
            require_text: whether every sample must have a text annotation
        """
        super().__init__()
        
        self.data_root = data_root
        self.image_root = image_root
        self.image_resolution = image_resolution
        self.image_transform = get_eval_image_transform(image_resolution)
        self.use_manifest = False
        self.has_gt_motion = True
        
        # Load mean/std for motion normalization
        self.mean = np.load(os.path.join(data_root, 'Mean.npy'))
        self.std = np.load(os.path.join(data_root, 'Std.npy'))
        self.std_safe = self.std.copy()
        self.std_safe[self.std_safe == 0] = 1.0

        if manifest_path:
            self._load_manifest_data(
                manifest_path=manifest_path,
                max_samples=max_samples,
            )
            return
        
        # Load data
        split_file = os.path.join(data_root, f'{split}.txt')
        with open(split_file, 'r') as f:
            data_name = f.readline().strip()
        
        # Motion (N_frames, 269)
        motion_path = os.path.join(data_root, 'new_joint_vecs', f'{data_name}.npy')
        self.all_motions = np.load(motion_path)
        
        # Image paths (N_frames,)
        image_paths_file = os.path.join(data_root, 'image_paths', f'{data_name}.npy')
        self.all_image_paths = np.load(image_paths_file, allow_pickle=True)
        
        assert len(self.all_motions) == len(self.all_image_paths), \
            f"Motion frames ({len(self.all_motions)}) != Image paths ({len(self.all_image_paths)})"
        
        # Text directory
        self.text_root = os.path.join(data_root, 'text')
        
        # Build valid sample indices
        total_frames = len(self.all_motions)
        self.valid_indices = []
        self.has_text = []  # Track which samples have text
        skipped_no_image = 0
        skipped_no_text = 0
        
        for i in range(total_frames):
            stored_path = str(self.all_image_paths[i])
            rel_path = self._get_image_rel_path(stored_path)
            full_path = os.path.join(self.image_root, rel_path)
            
            if not os.path.exists(full_path):
                skipped_no_image += 1
                continue
            
            # Check text availability (with fallback strategies for test set)
            text_path = image_path_to_text_path_fallback(rel_path, self.text_root)
            text_exists = text_path is not None
            
            if require_text and not text_exists:
                skipped_no_text += 1
                continue
            
            self.valid_indices.append(i)
            self.has_text.append(text_exists)
        
        # Subsample if max_samples specified
        if max_samples is not None and max_samples < len(self.valid_indices):
            rng = np.random.RandomState(seed)
            subset_idx = rng.choice(len(self.valid_indices), max_samples, replace=False)
            subset_idx = sorted(subset_idx)
            self.valid_indices = [self.valid_indices[i] for i in subset_idx]
            self.has_text = [self.has_text[i] for i in subset_idx]
        
        # Build text-available index list for img2text evaluation
        self.text_indices = [i for i, has in enumerate(self.has_text) if has]
        
        print(f"\n{'='*20} H36M Eval Dataset Info {'='*20}")
        print(f"Split:     {split}")
        print(f"Total frames: {len(self.all_motions)}")
        print(f"Valid samples: {len(self.valid_indices)}")
        print(f"Samples with text: {len(self.text_indices)}")
        if skipped_no_image > 0:
            print(f"Skipped (no image): {skipped_no_image}")
        if skipped_no_text > 0:
            print(f"Skipped (no text):  {skipped_no_text}")
        print(f"Motion shape: {self.all_motions.shape}")
        print(f"Image resolution: {image_resolution}")
        print(f"{'='*60}\n")

    def _load_manifest_data(
        self,
        manifest_path: str,
        max_samples: Optional[int],
    ) -> None:
        """Load an image-only sequential manifest for img2pose inference."""
        self.use_manifest = True
        self.has_gt_motion = False
        self.manifest_entries: List[Dict[str, Any]] = []

        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                abs_path = entry.get('abs_image_path', None)
                rel_path = entry.get('image_rel_path', entry.get('image_path', ''))
                if abs_path is None:
                    if not rel_path:
                        continue
                    abs_path = os.path.join(self.image_root, rel_path)
                elif (not rel_path) and str(abs_path).startswith(self.image_root.rstrip('/') + '/'):
                    rel_path = str(abs_path)[len(self.image_root.rstrip('/')) + 1:]

                if not os.path.exists(abs_path):
                    continue

                self.manifest_entries.append(
                    {
                        'abs_image_path': str(abs_path),
                        'image_rel_path': str(rel_path),
                        'clip_id': entry.get('clip_id', f"clip_{line_idx:05d}"),
                        'clip_index': int(entry.get('clip_index', -1)),
                        'clip_position': int(entry.get('clip_position', -1)),
                        'frame_number': int(entry.get('frame_number', -1)),
                        'sequence_id': str(entry.get('sequence_id', '')),
                    }
                )

        if max_samples is not None:
            self.manifest_entries = self.manifest_entries[: max(0, int(max_samples))]

        print(f"\n{'='*20} H36M Eval Manifest Info {'='*20}")
        print(f"Manifest:   {manifest_path}")
        print(f"Samples:    {len(self.manifest_entries)}")
        print(f"Image root: {self.image_root}")
        print(f"Image resolution: {self.image_resolution}")
        print(f"GT motion:  disabled (image-only inference mode)")
        print(f"{'='*60}\n")
    
    def _get_image_rel_path(self, stored_path: str) -> str:
        """Extract relative path from stored full path."""
        marker = 'images_fps50_cropped_448x448/'
        idx = stored_path.find(marker)
        if idx >= 0:
            return stored_path[idx + len(marker):]
        return stored_path
    
    def __len__(self) -> int:
        if self.use_manifest:
            return len(self.manifest_entries)
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        try:
            if self.use_manifest:
                entry = self.manifest_entries[idx]
                img = Image.open(entry['abs_image_path']).convert('RGB')
                image = self.image_transform(img)

                zero_motion = np.zeros_like(self.mean, dtype=np.float32)
                return {
                    'image': image,
                    'motion_raw': torch.from_numpy(zero_motion.copy()).float(),
                    'motion_norm': torch.from_numpy(zero_motion.copy()).float(),
                    'all_texts': [],
                    'has_text': False,
                    'has_gt_motion': False,
                    'image_path': entry['image_rel_path'],
                    'image_abspath': entry['abs_image_path'],
                    'clip_id': entry['clip_id'],
                    'clip_index': entry['clip_index'],
                    'clip_position': entry['clip_position'],
                    'frame_number': entry['frame_number'],
                    'sequence_id': entry['sequence_id'],
                }

            frame_idx = self.valid_indices[idx]
            
            # Load image
            stored_path = str(self.all_image_paths[frame_idx])
            rel_path = self._get_image_rel_path(stored_path)
            full_path = os.path.join(self.image_root, rel_path)
            
            img = Image.open(full_path).convert('RGB')
            image = self.image_transform(img)  # [3, H, W]
            
            # Load GT motion (raw, NOT normalized - we'll normalize in test script)
            motion_raw = self.all_motions[frame_idx].copy()  # (269,)
            
            # Also provide normalized version
            motion_norm = (motion_raw - self.mean) / self.std_safe  # (269,)
            
            # Load text (if available, with fallback strategies)
            text_path = image_path_to_text_path_fallback(rel_path, self.text_root)
            all_texts = []
            if text_path is not None and os.path.exists(text_path):
                try:
                    with open(text_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    if content:
                        all_texts = [l.strip() for l in content.split('\n') if l.strip()]
                except Exception:
                    pass
            
            return {
                'image': image,
                'motion_raw': torch.from_numpy(motion_raw).float(),     # [269] NOT normalized
                'motion_norm': torch.from_numpy(motion_norm).float(),   # [269] normalized
                'all_texts': all_texts,                                  # list of GT text descriptions
                'has_text': len(all_texts) > 0,
                'has_gt_motion': True,
                'image_path': rel_path,
            }
            
        except Exception as e:
            print(f"Error in H36MEvalDataset __getitem__ {idx}: {e}")
            # Return next valid sample
            return self.__getitem__((idx + 1) % len(self))
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """Custom collate function."""
        result = {
            'images': torch.stack([d['image'] for d in batch]),                # [bs, 3, H, W]
            'motions_raw': torch.stack([d['motion_raw'] for d in batch]),      # [bs, 269]
            'motions_norm': torch.stack([d['motion_norm'] for d in batch]),    # [bs, 269]
            'all_texts': [d['all_texts'] for d in batch],                      # list of list
            'has_text': [d['has_text'] for d in batch],                        # list of bool
            'has_gt_motion': [d['has_gt_motion'] for d in batch],              # list of bool
            'image_paths': [d['image_path'] for d in batch],                   # list of str
        }
        for key in ('image_abspath', 'clip_id', 'clip_index', 'clip_position', 'frame_number', 'sequence_id'):
            if key in batch[0]:
                result[key] = [d[key] for d in batch]
        return result
