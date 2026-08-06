#!/usr/bin/env python
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
Motion preprocessing script: encode raw motion data into latents via the VAE and save them to speed up training.

Usage:
    python datasets/preprocess_motion.py \
        --data_root data/UNI_MOTION_269/HumanML3D \
        --vae_path pretrained/cma_vae.ckpt \
        --output_dir data/UNI_MOTION_269/HumanML3D/motion_latents \
        --split train \
        --batch_size 32 \
        --clean \
        # --device cuda:0

Arguments:
    --clean: remove existing .npz files in the output directory before processing (use when regenerating with a different model)
    --verify: verify the generated latent files after processing

Output format:
    Each motion sample is saved as an {id}.npz file containing:
    - latent: [latent_len, 256] motion latent
    - length: original motion length
"""

import os
import sys
import argparse
import math
from pathlib import Path
from typing import List
from tqdm import tqdm

import numpy as np
import torch
import codecs as cs

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_motion_vae(vae_path: str, device: str = "cuda:0"):
    """Load the Motion VAE model."""
    from models.motion_vae import load_motion_vae as _load_motion_vae
    return _load_motion_vae(vae_path, device=device, freeze=True)


def load_motion_data(
    data_root: str,
    split: str,
    max_motion_length: int = 200,
    min_motion_length: int = 20,
    fps: int = 20
) -> dict:
    """
    Load motion data.

    Returns:
        data_dict: {name: {'motion': np.array, 'length': int}}
    """
    split_file = os.path.join(data_root, f'{split}.txt')
    motion_dir = os.path.join(data_root, 'new_joint_vecs')
    text_dir = os.path.join(data_root, 'texts')
    mean = np.load(os.path.join(data_root, 'Mean.npy'))
    std = np.load(os.path.join(data_root, 'Std.npy'))
    
    # Load IDs
    id_list = []
    with cs.open(split_file, 'r') as f:
        for line in f.readlines():
            id_list.append(line.strip())
    
    data_dict = {}
    
    for name in tqdm(id_list, desc=f"Loading {split} data"):
        try:
            motion_path = os.path.join(motion_dir, f'{name}.npy')
            if not os.path.exists(motion_path):
                continue
            motion = np.load(motion_path)
            
            text_path = os.path.join(text_dir, f'{name}.txt')
            if not os.path.exists(text_path):
                continue
            
            # Parse text file for sub-segments
            with cs.open(text_path, 'r') as f:
                has_full = False
                for line in f.readlines():
                    line_split = line.strip().split('#')
                    if len(line_split) < 4:
                        continue
                    
                    f_tag = float(line_split[2]) if line_split[2] else 0.0
                    to_tag = float(line_split[3]) if line_split[3] else 0.0
                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag
                    
                    if f_tag == 0.0 and to_tag == 0.0:
                        has_full = True
                    else:
                        # Sub-segment
                        start_frame = int(f_tag * fps)
                        end_frame = int(to_tag * fps)
                        if start_frame >= end_frame:
                            continue
                        
                        motion_new = motion[start_frame:end_frame]
                        if len(motion_new) < min_motion_length or \
                           len(motion_new) >= max_motion_length:
                            continue
                        
                        new_name = f'{name}_{f_tag}_{to_tag}'
                        # Normalize
                        motion_norm = (motion_new - mean) / std
                        data_dict[new_name] = {
                            'motion': motion_norm,
                            'length': len(motion_new)
                        }
                
                # Full motion
                if has_full and min_motion_length <= len(motion) < max_motion_length:
                    motion_norm = (motion - mean) / std
                    data_dict[name] = {
                        'motion': motion_norm,
                        'length': len(motion)
                    }
                    
        except Exception as e:
            print(f"Error loading {name}: {e}")
            continue
    
    return data_dict


def encode_motions_batch(
    vae,
    motions: List[np.ndarray],
    lengths: List[int],
    device: str,
    downsample_rate: int,
) -> List[np.ndarray]:
    """
    Encode motions in batches.

    Args:
        vae: Motion VAE model
        motions: [N] list of [T_i, 269] arrays
        lengths: [N] list of motion lengths
        device: compute device
        downsample_rate: VAE downsample rate (1/2/4/8)

    Returns:
        latents: [N] list of [latent_len_i, 256] arrays
    """
    # Pad to same length
    max_len = max(m.shape[0] for m in motions)
    batch = np.zeros((len(motions), max_len, motions[0].shape[1]), dtype=np.float32)
    for i, m in enumerate(motions):
        batch[i, :m.shape[0]] = m
    
    batch_tensor = torch.from_numpy(batch).to(device)
    
    with torch.no_grad():
        # VAE encode: [bs, T, 269] -> [latent_len, bs, 256]
        latent, _ = vae.encode_motion(batch_tensor, lengths)
        # [latent_len, bs, 256] -> [bs, latent_len, 256]
        latent = latent.permute(1, 0, 2).cpu().numpy()
    
    # Extract valid latents for each sample
    results = []
    for i, length in enumerate(lengths):
        latent_len = math.ceil(length / downsample_rate)
        results.append(latent[i, :latent_len])
    
    return results


def preprocess_motion(
    data_root: str,
    vae_path: str,
    output_dir: str,
    split: str = 'train',
    batch_size: int = 32,
    max_motion_length: int = 200,
    min_motion_length: int = 20,
    device: str = 'cuda:0',
    clean: bool = False,
):
    """
    Preprocess motion data: encode into VAE latents and save.

    Args:
        data_root: HumanML3D dataset root directory
        vae_path: path to the Motion VAE checkpoint
        output_dir: output directory
        split: dataset split (train/val/test)
        batch_size: batch size
        max_motion_length: maximum motion length
        min_motion_length: minimum motion length
        device: compute device
        clean: whether to clean the output directory before processing
    """
    print(f"="*60)
    print(f"Motion Preprocessing Pipeline")
    print(f"="*60)
    print(f"Data root: {data_root}")
    print(f"VAE path: {vae_path}")
    print(f"Output dir: {output_dir}")
    print(f"Split: {split}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"Clean mode: {clean}")
    print(f"="*60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean existing files if requested
    if clean:
        existing_files = list(Path(output_dir).glob("*.npz"))
        if existing_files:
            print(f"\nCleaning {len(existing_files)} existing latent files...")
            for f in existing_files:
                f.unlink()
            print("Clean complete!")
    
    # Load VAE
    print("\nLoading Motion VAE...")
    vae, config = load_motion_vae(vae_path, device=device)
    print(f"VAE config: {config}")
    downsample_rate = int(config.get("downsample_rate", 1))
    
    # Load data
    print(f"\nLoading {split} motion data...")
    data_dict = load_motion_data(
        data_root, split,
        max_motion_length=max_motion_length,
        min_motion_length=min_motion_length
    )
    print(f"Loaded {len(data_dict)} motion samples")
    
    # Process in batches
    names = list(data_dict.keys())
    num_batches = (len(names) + batch_size - 1) // batch_size
    
    processed = 0
    skipped = 0
    
    print(f"\nEncoding motions (batch_size={batch_size})...")
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(names))
        batch_names = names[start_idx:end_idx]
        
        # Prepare batch
        motions = []
        lengths = []
        valid_names = []
        
        for name in batch_names:
            data = data_dict[name]
            motions.append(data['motion'])
            lengths.append(data['length'])
            valid_names.append(name)
        
        if len(motions) == 0:
            continue
        
        try:
            # Encode batch
            latents = encode_motions_batch(
                vae,
                motions,
                lengths,
                device,
                downsample_rate=downsample_rate,
            )
            
            # Save each latent
            for name, latent, length in zip(valid_names, latents, lengths):
                # Replace '.' with '_' in name for safe file naming
                safe_name = name.replace('.', '_')
                output_path = os.path.join(output_dir, f'{safe_name}.npz')
                np.savez_compressed(
                    output_path,
                    latent=latent,
                    length=length
                )
                processed += 1
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            skipped += len(batch_names)
            continue
    
    print(f"\n{'='*60}")
    print(f"Preprocessing complete!")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*60}")


def verify_latents(
    output_dir: str,
    num_samples: int = 5
):
    """Verify the generated latent files."""
    print(f"\nVerifying latent files in {output_dir}...")
    
    files = list(Path(output_dir).glob("*.npz"))
    print(f"Found {len(files)} latent files")
    
    if len(files) == 0:
        print("No latent files found!")
        return
    
    # Check random samples
    import random
    samples = random.sample(files, min(num_samples, len(files)))
    
    print(f"\nSample verification:")
    for f in samples:
        data = np.load(f)
        print(f"  {f.name}: latent={data['latent'].shape}, length={data['length']}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess motion data for UniMotion')
    parser.add_argument('--data_root', type=str, required=True,
                        help='HumanML3D dataset root directory')
    parser.add_argument('--vae_path', type=str, required=True,
                        help='Path to Motion VAE checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for latent files')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to process')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for encoding')
    parser.add_argument('--max_motion_length', type=int, default=200,
                        help='Maximum motion length')
    parser.add_argument('--min_motion_length', type=int, default=20,
                        help='Minimum motion length')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for computation')
    parser.add_argument('--verify', action='store_true',
                        help='Verify generated latents after preprocessing')
    parser.add_argument('--clean', action='store_true',
                        help='Clean existing latent files before preprocessing')
    
    args = parser.parse_args()
    
    preprocess_motion(
        data_root=args.data_root,
        vae_path=args.vae_path,
        output_dir=args.output_dir,
        split=args.split,
        batch_size=args.batch_size,
        max_motion_length=args.max_motion_length,
        min_motion_length=args.min_motion_length,
        device=args.device,
        clean=args.clean,
    )
    
    if args.verify:
        verify_latents(args.output_dir)


if __name__ == '__main__':
    main()
