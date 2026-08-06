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
Motion Edit Dataset for the UniMotion framework.
Supports the MotionFix dataset format, used to train the Motion Editing task.

Task description:
- Input: source motion + edit instruction
- Output: target motion (the edited motion)

Data format:
- Source motion: [T, 269] - original motion
- Target motion: [T, 269] - edited motion (same frame count as the source)
- Edit instruction: textual editing instruction

Sequence format:
[BOS] <edit_instruction> [BOM][SOURCE_MOT*N][EOM] [BOM][TARGET_MOT*N][EOM][EOS]
- SOURCE_MOT: conditioning input, no loss computed
- TARGET_MOT: generation target, diffusion loss computed
"""

import os
import json
import random
import collections
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import codecs as cs
from torch.utils.data import Dataset

from datasets.utils import format_sequence_motion_edit
from motion_length_utils import motion_token_slots


def load_edit_instruction_templates(template_path: str) -> List[str]:
    """
    Load the edit instruction template JSON file.
    
    Args:
        template_path: path to the template file
        
    Returns:
        list: list of instruction templates
    """
    with open(template_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract all input templates
    templates = []
    for task_name, task_data in data.items():
        for subtask_name, subtask_data in task_data.items():
            if 'input' in subtask_data:
                templates.extend(subtask_data['input'])
    
    return templates


class MotionEditDataset(Dataset):
    """
    Dataset for Motion Editing task (Motion-to-Motion with text instruction).
    
    Given a source motion and an edit instruction, generate the edited motion.
    - Source motion: original motion (conditioning input)
    - Edit instruction: textual editing instruction
    - Target motion: edited motion (generation target)
    
    Data format:
    - Input: [BOS] <edit_instruction> [BOM][SOURCE_MOT][EOM] [BOM][TARGET_MOT][EOM][EOS]
    - Source motion uses the fused path (understanding semantics)
    - Target motion uses the gen path (generation)
    """

    def __init__(
        self,
        data_root: str,
        text_tokenizer: Any,
        mean_path: str = None,
        std_path: str = None,
        max_seq_len: int = 512,
        max_motion_length: int = 200,
        min_motion_length: int = 20,
        unit_length: int = 1,
        fps: int = 20,
        downsample_rate: int = 1,
        showo_token_ids: Optional[Dict[str, int]] = None,
        motion_latent_dim: int = 256,
        split: str = 'train',
        tiny: bool = False,
        debug: bool = False,
        add_time_embeds: bool = True,
        instruction_template_path: str = None,
        **kwargs,
    ):
        super().__init__()
        
        self.data_root = data_root
        self.text_tokenizer = text_tokenizer
        self.max_seq_len = max_seq_len
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length
        self.fps = fps
        self.downsample_rate = downsample_rate
        self.motion_latent_dim = motion_latent_dim
        self.add_time_embeds = add_time_embeds
        self.data_type = "edit"  # Motion Edit
        
        # Load edit instruction templates
        self.edit_instructions = []
        if instruction_template_path and os.path.exists(instruction_template_path):
            self.edit_instructions = load_edit_instruction_templates(instruction_template_path)
            print(f"Loaded {len(self.edit_instructions)} Edit instruction templates")
        
        if not self.edit_instructions:
            # Default template
            self.edit_instructions = [
                "Given the source motion <Source_Motion_Placeholder>, please modify it according to the following instruction: <Edit_Instruction_Placeholder>"
            ]
        
        # Token IDs
        self.pad_id = text_tokenizer.pad_token_id
        self.bos_id = showo_token_ids['bos_id']
        self.eos_id = showo_token_ids['eos_id']
        self.bom_id = showo_token_ids.get('bom_id', showo_token_ids['boi_id'])
        self.eom_id = showo_token_ids.get('eom_id', showo_token_ids['eoi_id'])
        self.mot_pad_id = showo_token_ids.get('mot_pad_id', showo_token_ids['img_pad_id'])
        
        # Load mean and std
        if mean_path and os.path.exists(mean_path):
            self.mean = np.load(mean_path)
        else:
            self.mean = np.load(os.path.join(data_root, 'Mean.npy'))
            
        if std_path and os.path.exists(std_path):
            self.std = np.load(std_path)
        else:
            self.std = np.load(os.path.join(data_root, 'Std.npy'))
        
        # Compute the maximum number of motion tokens
        self.max_motion_tokens = motion_token_slots(
            max_motion_length,
            downsample_rate,
            add_time_embeds=add_time_embeds,
        )
        
        # Load data
        self._load_data(split, tiny, debug)
    
    def _load_data(self, split: str, tiny: bool, debug: bool):
        """Load the MotionFix dataset."""
        split_file = os.path.join(self.data_root, f'{split}.txt')
        source_dir = os.path.join(self.data_root, 'new_joint_vecs', 'source')
        target_dir = os.path.join(self.data_root, 'new_joint_vecs', 'target')
        text_dir = os.path.join(self.data_root, 'texts')
        
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                # Format: source/XXXXXX
                line = line.strip()
                if line.startswith('source/'):
                    sample_id = line.split('/')[1]
                else:
                    sample_id = line
                id_list.append(sample_id)
        
        if tiny or debug:
            id_list = id_list[:100]
        
        new_name_list = []
        data_dict = {}
        
        for sample_id in id_list:
            try:
                source_path = os.path.join(source_dir, f'{sample_id}.npy')
                target_path = os.path.join(target_dir, f'{sample_id}.npy')
                text_path = os.path.join(text_dir, f'{sample_id}.txt')
                
                # Check that the files exist
                if not os.path.exists(source_path) or not os.path.exists(target_path):
                    continue
                
                source_motion = np.load(source_path)
                target_motion = np.load(target_path)
                
                # Check the length limits
                if len(source_motion) < self.min_motion_length or len(source_motion) >= self.max_motion_length:
                    continue
                
                # Load the edit instruction text
                edit_text = ""
                if os.path.exists(text_path):
                    with cs.open(text_path, 'r') as f:
                        edit_text = f.read().strip()
                
                if not edit_text:
                    continue  # Skip samples without an edit instruction
                
                data_dict[sample_id] = {
                    'source_motion': source_motion,
                    'target_motion': target_motion,
                    'edit_instruction': edit_text
                }
                new_name_list.append(sample_id)
                
            except Exception as e:
                print(f"Error loading {sample_id}: {e}")
                continue
        
        self.data_dict = data_dict
        self.name_list = new_name_list
        print(f'Motion Edit dataset [{split}] loaded: {len(self.data_dict)} samples')
        
        if len(self.name_list) > 0:
            print(f"\n{'='*20} Motion Edit Dataset Info {'='*20}")
            print(f"Task:           Motion Edit (MotionFix)")
            print(f"Num samples:    {len(self.name_list)}")
            print(f"Num templates:  {len(self.edit_instructions)}")
            print(f"Max motion len: {self.max_motion_length}")
            print(f"downsample_rate: {self.downsample_rate}")
            print(f"add_time_embeds: {self.add_time_embeds}")
            print(f"{'='*60}\n")
            
            # Print a sample example
            if len(self.name_list) > 0:
                sample_id = self.name_list[0]
                sample = self.data_dict[sample_id]
                print(f"Sample example:")
                print(f"  ID: {sample_id}")
                print(f"  Source motion shape: {sample['source_motion'].shape}")
                print(f"  Target motion shape: {sample['target_motion'].shape}")
                print(f"  Edit instruction: {sample['edit_instruction'][:100]}...")
                print()
    
    def _compute_num_motion_tokens(self, motion_length: int, add_time_embed: bool = None) -> int:
        """Compute the number of tokens for a given motion length."""
        should_add_time = add_time_embed if add_time_embed is not None else self.add_time_embeds
        return motion_token_slots(
            motion_length,
            self.downsample_rate,
            add_time_embeds=should_add_time,
        )
    
    def __len__(self) -> int:
        return len(self.name_list)
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Get a single sample.
        
        Returns:
            dict containing:
                - text_tokens: [max_seq_len] padded token sequence
                - text_labels: [max_seq_len] labels for NTP loss
                - source_motion: [T, 269] source motion (last 6 dims zeroed during testing)
                - target_motion: [T, 269] target motion (last 6 dims zeroed during testing)
                - source_motion_length: source motion length
                - target_motion_length: target motion length
                - source_modality_positions: [1, 2] source motion position
                - target_modality_positions: [1, 2] target motion position
                - text_masks: [max_seq_len] text token mask
                - source_motion_masks: [max_seq_len] source motion token mask
                - target_motion_masks: [max_seq_len] target motion token mask
                - data_type: 'edit'
        """
        try:
            sample_id = self.name_list[idx]
            data = self.data_dict[sample_id]
            
            source_motion = data['source_motion'].copy()
            target_motion = data['target_motion'].copy()
            edit_instruction = data['edit_instruction']
            
            # Align to unit_length
            m_length = (len(source_motion) // self.unit_length) * self.unit_length
            if m_length < self.min_motion_length:
                return self.__getitem__((idx + 1) % len(self))
            
            source_motion = source_motion[:m_length]
            target_motion = target_motion[:m_length]
            
            # Z Normalization
            source_motion = (source_motion - self.mean) / self.std
            target_motion = (target_motion - self.mean) / self.std
            source_motion = torch.from_numpy(source_motion).float()
            target_motion = torch.from_numpy(target_motion).float()
            # The evaluation protocol uses only the first 263 dims; the last 6 dims are explicitly ignored.
            if source_motion.shape[-1] > 263:
                source_motion[:, 263:] = 0.0
            if target_motion.shape[-1] > 263:
                target_motion[:, 263:] = 0.0
            
            # Calculate number of motion tokens
            # Source motion: conditioning input, no time embedding needed (uses the fused path)
            # Target motion: generation target, needs time embedding (uses the gen path)
            num_source_tokens = self._compute_num_motion_tokens(len(source_motion), add_time_embed=False)
            num_target_tokens = self._compute_num_motion_tokens(len(target_motion), add_time_embed=True)
            
            # Randomly pick an instruction template and fill it in
            template = random.choice(self.edit_instructions)
            # Replace placeholders
            prompt = template.replace('<Source_Motion_Placeholder>', '').replace('<Edit_Instruction_Placeholder>', edit_instruction).strip()
            
            # Tokenize prompt
            text_tokens = self.text_tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=200  # Edit instructions can be long
            ).input_ids
            
            # Format sequence
            (text_tokens_tensor, text_labels, 
             source_modality_positions, target_modality_positions,
             text_mask, source_motion_mask, target_motion_mask) = format_sequence_motion_edit(
                text_tokens, self.bos_id, self.eos_id,
                self.bom_id, self.eom_id, self.pad_id, self.mot_pad_id,
                num_source_tokens, num_target_tokens, self.max_seq_len
            )
            
            sample = {
                'text_tokens': text_tokens_tensor,
                'text_labels': text_labels,
                'source_motion': source_motion,
                'target_motion': target_motion,
                'sample_id': sample_id,
                'edit_instruction': edit_instruction,
                'source_motion_length': torch.tensor(len(source_motion)),
                'target_motion_length': torch.tensor(len(target_motion)),
                'source_modality_positions': source_modality_positions,
                'target_modality_positions': target_modality_positions,
                'text_masks': text_mask,
                'source_motion_masks': source_motion_mask,
                'target_motion_masks': target_motion_mask,
                'data_type': self.data_type
            }
            
            return sample
            
        except Exception as e:
            print(f"Error in MotionEditDataset __getitem__ {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """Custom collate function for motion edit."""
        batched = collections.defaultdict(list)
        
        # Find max motion lengths
        max_source_len = max(sample['source_motion'].shape[0] for sample in batch)
        max_target_len = max(sample['target_motion'].shape[0] for sample in batch)
        
        # Align to unit_length
        if max_source_len % self.unit_length != 0:
            max_source_len = ((max_source_len // self.unit_length) + 1) * self.unit_length
        if max_target_len % self.unit_length != 0:
            max_target_len = ((max_target_len // self.unit_length) + 1) * self.unit_length
        
        for data in batch:
            for k, v in data.items():
                if k == 'source_motion':
                    motion = v
                    pad_len = max_source_len - motion.shape[0]
                    if pad_len > 0:
                        motion = torch.cat([motion, torch.zeros(pad_len, motion.shape[1])], dim=0)
                    batched[k].append(motion)
                elif k == 'target_motion':
                    motion = v
                    pad_len = max_target_len - motion.shape[0]
                    if pad_len > 0:
                        motion = torch.cat([motion, torch.zeros(pad_len, motion.shape[1])], dim=0)
                    batched[k].append(motion)
                else:
                    batched[k].append(v)
        
        # Stack tensors
        result = {}
        for k, v in batched.items():
            if k in ('data_type', 'edit_instruction', 'sample_id'):
                result[k] = v
            else:
                result[k] = torch.stack(v, dim=0)
        
        return result
