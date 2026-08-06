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
Motion-Text Dataset for the UniMotion framework.
Supports the HumanML3D dataset format, used to train Text-to-Motion and Motion-to-Text tasks.

Training stage description (following MotionGPT3):
- Stage 1 (lm_pretrain, T2M only): train T2M only, no instruction prompts
- Stage 2a (lm_pretrain, all): train T2M+M2T, no instruction prompts
- Stage 2b (lm_instruct, all): train T2M+M2T with rich instruction templates

Motion data format:
- Input: [T, 269] - motion features in HumanML3D format (269-dim version with 6 extra dims)
- Normalized: (motion - mean) / std
- Motion VAE: [T, 269] -> encode -> [latent_len, 1, 256]
- latent_len = ceil(T / downsample_rate)
"""

import os
import json
import random
import collections
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import codecs as cs
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets.utils import (
    format_sequence_motion_gen, format_sequence_motion_und, format_sequence_motion_pred,
    format_sequence_motion_und_instruct, format_sequence_text_to_text, format_sequence_motion_edit
)
from motion_length_utils import motion_token_slots


# ==================== Instruction Template Loading ====================

def load_instruction_templates(template_path: str) -> Dict[str, Any]:
    """
    Load the instruction template JSON file.
    
    Args:
        template_path: path to the template file
        
    Returns:
        dict: parsed template dictionary
    """
    with open(template_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_all_tasks_from_templates(templates: Dict[str, Any], task_type: str = None) -> List[Dict[str, Any]]:
    """
    Extract all task configs from the templates (consistent with MotionGPT3).
    
    MotionGPT3 iterates over all tasks and subtasks, treating each subtask as an independent task config.
    This ensures every data sample is paired with every instruction template during training.
    
    Args:
        templates: template dictionary
        task_type: restrict the task type ('t2m', 'm2t', None for all)
        
    Returns:
        list: list of task configs; each element is a dict with class, input, output
    """
    tasks = []
    for task_name in templates.keys():
        for subtask_name, subtask_data in templates[task_name].items():
            task_class = subtask_data.get("class", "")
            # Filter by task_type
            if task_type is not None:
                if task_type == 't2m' and task_class not in ['t2m', 'n2m']:
                    continue
                elif task_type == 'm2t' and task_class != 'm2t':
                    continue
            # Consistent with mainstream baselines: keep only caption-style templates for M2T,
            # and do not use any template containing Frame/Second length hints.
            if task_type == 'm2t':
                input_templates = subtask_data.get("input", [])
                has_length_placeholders = any(
                    ("<Frame_Placeholder>" in t) or ("<Second_Placeholder>" in t)
                    for t in input_templates
                )
                if has_length_placeholders:
                    continue
            tasks.append(subtask_data)
    return tasks


def get_t2m_instructions(templates: Dict[str, Any]) -> List[str]:
    """
    Extract the instruction list for the T2M task from the templates.
    
    Args:
        templates: template dictionary
        
    Returns:
        list: list of T2M instruction templates
    """
    t2m_instructions = []
    if "Text-to-Motion" in templates:
        for task_name, task_data in templates["Text-to-Motion"].items():
            if task_data.get("class") == "t2m":
                t2m_instructions.extend(task_data.get("input", []))
    return t2m_instructions


def get_m2t_instructions(templates: Dict[str, Any]) -> List[str]:
    """
    Extract the instruction list for the M2T task from the templates.
    
    Args:
        templates: template dictionary
        
    Returns:
        list: list of M2T instruction templates
    """
    m2t_instructions = []
    if "Motion-to-Text" in templates:
        for task_name, task_data in templates["Motion-to-Text"].items():
            if task_data.get("class") == "m2t":
                m2t_instructions.extend(task_data.get("input", []))
    return m2t_instructions


def get_pred_instructions(templates: Dict[str, Any]) -> List[str]:
    """
    Extract the instruction list for the Motion Prediction task from the templates.
    
    Note: different template files use different key names:
    - template_pretrain.json uses "Motion Prediction" (with a space)
    - template_instructions.json uses "Motion-to-Motion"
    - template_witht2t_instructions.json uses "Motion-to-Motion"
    
    Args:
        templates: template dictionary
        
    Returns:
        list: list of Prediction instruction templates
    """
    pred_instructions = []
    
    # Check all possible key names
    possible_keys = ["Motion Prediction", "Motion-Prediction", "Motion-to-Motion"]
    
    for key in possible_keys:
        if key in templates:
            for task_name, task_data in templates[key].items():
                # Match class: "predict" or "pred"
                task_class = task_data.get("class", "")
                if task_class in ["predict", "pred"]:
                    pred_instructions.extend(task_data.get("input", []))
    
    return pred_instructions


def get_t2t_tasks(templates: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract the Text-to-Text (T2T) task config list from the templates.
    
    T2T task types (following MotionGPT3 Stage 3):
    - caption-to-framelen: predict the motion frame count from the caption
    - framelen-to-caption: generate a plausible motion description from the frame count
    - second-to-caption: generate a description from the duration in seconds
    - n2t (random-caption): generate a motion description unconditionally
    
    Args:
        templates: template dictionary
        
    Returns:
        list: list of T2T task configs; each element contains class, input, output
    """
    t2t_tasks = []
    
    # Text-to-Text tasks
    if "Text-to-Text" in templates:
        for task_name, task_data in templates["Text-to-Text"].items():
            if task_data.get("class") in ["t2t", "n2t"]:
                t2t_tasks.append(task_data)
    
    return t2t_tasks


def apply_instruction_template(caption: str, instruction: str, placeholder: str = "<Caption_Placeholder>") -> str:
    """
    Apply the caption to an instruction template.
    
    Args:
        caption: original text description
        instruction: instruction template
        placeholder: placeholder string
        
    Returns:
        str: full instruction with the placeholder replaced
    """
    return instruction.replace(placeholder, caption)


def apply_m2t_instruction_template(
    instruction: str,
    frame_length: Optional[int] = None,
    second_length: Optional[float] = None,
) -> str:
    """
    Process an instruction template for the M2T task, removing the <Motion_Placeholder> placeholder.
    
    In the M2T task, <Motion_Placeholder> is replaced by actual motion tokens,
    so the text input keeps only the instruction part without the placeholder.
    
    e.g.: "Describe the motion represented by <Motion_Placeholder> using plain English."
    -> "Describe the motion represented by using plain English."
    -> finally simplified into a guiding prompt; motion tokens are embedded directly via the model architecture
    
    Args:
        instruction: original instruction template (contains <Motion_Placeholder>)
        
    Returns:
        str: processed instruction (text with placeholders removed)
    """
    # Remove <Motion_Placeholder>; M2T does not use length placeholder hints.
    result = instruction.replace("<Motion_Placeholder>", "").strip()
    result = result.replace("<Frame_Placeholder>", "")
    result = result.replace("<Second_Placeholder>", "")
    # Clean up any double spaces produced
    while "  " in result:
        result = result.replace("  ", " ")
    return result


class MotionTextDataset(Dataset):
    """
    Dataset for loading motion-text pairs for the UniMotion framework.
    
    Data layout:
    - {data_root}/new_joint_vecs/{id}.npy - motion data [T, 263]
    - {data_root}/texts/{id}.txt - text descriptions
    - {data_root}/{split}.txt - train/test/val split
    - {data_root}/Mean.npy, Std.npy - normalization statistics
    
    Training stage description:
    - Stage 1 (lm_pretrain): T2M only, no instruction prompts
    - Stage 2a (lm_pretrain): T2M+M2T, no instruction prompts
    - Stage 2b (lm_instruct): T2M+M2T with instruction templates
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
        unit_length: int = 4,
        fps: int = 20,
        downsample_rate: int = 4,
        is_understanding: bool = False,
        cond_dropout_prob: float = 0.1,
        showo_token_ids: Optional[Dict[str, int]] = None,
        motion_latent_dim: int = 256,
        split: str = 'train',
        tiny: bool = False,
        debug: bool = False,
        add_time_embeds: bool = True,
        disable_random_crop_for_m2t: bool = False,
        # New: training stage configuration
        stage: str = 'lm_pretrain',  # 'lm_pretrain' or 'lm_instruct'
        instruction_template_path: str = None,  # Path to the instruction template file
        **kwargs,
    ):
        """
        Initialize the Motion-Text dataset.
        
        Args:
            data_root: dataset root directory (HumanML3D)
            text_tokenizer: text tokenizer
            mean_path: path to Mean.npy, defaults to data_root/Mean.npy
            std_path: path to Std.npy, defaults to data_root/Std.npy
            max_seq_len: maximum sequence length (text plus motion tokens)
            max_motion_length: maximum number of motion frames
            min_motion_length: minimum number of motion frames
            unit_length: alignment unit for motion length
            fps: frame rate
            downsample_rate: Motion VAE downsample rate
            is_understanding: whether this is the understanding task (Motion-to-Text)
            cond_dropout_prob: condition dropout probability
            showo_token_ids: dict of special token IDs
            motion_latent_dim: Motion VAE latent dimension
            split: dataset split (train/val/test)
            tiny: whether to use a tiny subset
            debug: whether to run in debug mode
            add_time_embeds: whether to prepend a time embedding slot before the motion tokens
            disable_random_crop_for_m2t: whether to disable random cropping for the M2T task only (default False, keeps the original behavior)
            stage: training stage ('lm_pretrain' or 'lm_instruct')
            instruction_template_path: path to the instruction template JSON file (only needed when stage='lm_instruct')
        """
        super().__init__()
        
        self.data_root = data_root
        self.text_tokenizer = text_tokenizer
        self.max_seq_len = max_seq_len
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length
        self.fps = fps
        self.downsample_rate = downsample_rate
        self.is_understanding = is_understanding
        self.cond_dropout_prob = cond_dropout_prob
        self.motion_latent_dim = motion_latent_dim
        self.add_time_embeds = add_time_embeds
        self.data_type = "m2t" if is_understanding else "t2m_motion"
        self.disable_random_crop_for_m2t = bool(disable_random_crop_for_m2t)
        
        # Training stage configuration
        self.stage = stage
        # Consistent with MotionGPT3: both lm_instruct and lm_finetune stages use instruction templates
        # See MotionGPT3/motGPT/data/humanml/dataset_t2m_v3.py lines 54-57:
        #   elif stage in ['lm_instruct', "lm_rl", "lm_finetune"]:
        #       instructions = pjoin(data_root, f'template{instruction_type}_instructions.json')
        self.use_instructions = (stage in ['lm_instruct', 'lm_finetune'])
        
        # Load instruction templates (in the lm_instruct and lm_finetune stages)
        # Consistent with MotionGPT3: store all task configs so each data sample is paired with every instruction template
        self.t2m_instructions = []
        self.m2t_instructions = []
        self.tasks = []  # Store all task configs (consistent with MotionGPT3)
        
        if self.use_instructions and instruction_template_path:
            if os.path.exists(instruction_template_path):
                templates = load_instruction_templates(instruction_template_path)
                self.t2m_instructions = get_t2m_instructions(templates)
                self.m2t_instructions = get_m2t_instructions(templates)
                
                # Get all task configs matching the current task type
                task_type_filter = 'm2t' if is_understanding else 't2m'
                self.tasks = get_all_tasks_from_templates(templates, task_type=task_type_filter)
                
                print(f"Loaded {len(self.t2m_instructions)} T2M instructions, {len(self.m2t_instructions)} M2T instructions")
                print(f"Total tasks for {task_type_filter}: {len(self.tasks)}")
            else:
                print(f"Warning: instruction template file not found: {instruction_template_path}")
        
        # If no tasks exist, create a default empty task (for the pretraining stage)
        if len(self.tasks) == 0:
            self.tasks = [{"class": self.data_type, "input": ["<Caption_Placeholder>"], "output": ["<Motion_Placeholder>"]}]
        
        # Token IDs - fall back to the existing image tokens as defaults
        self.pad_id = text_tokenizer.pad_token_id
        self.bos_id = showo_token_ids['bos_id']
        self.eos_id = showo_token_ids['eos_id']
        # Motion uses the same begin/end tokens as image, or dedicated tokens
        self.bom_id = showo_token_ids.get('bom_id', showo_token_ids['boi_id'])  # Begin of Motion
        self.eom_id = showo_token_ids.get('eom_id', showo_token_ids['eoi_id'])  # End of Motion
        self.mot_pad_id = showo_token_ids.get('mot_pad_id', showo_token_ids['img_pad_id'])  # Motion pad token
        
        # Load mean and std for normalization
        if mean_path and os.path.exists(mean_path):
            self.mean = np.load(mean_path)
        else:
            self.mean = np.load(os.path.join(data_root, 'Mean.npy'))
            
        if std_path and os.path.exists(std_path):
            self.std = np.load(std_path)
        else:
            self.std = np.load(os.path.join(data_root, 'Std.npy'))
        
        # Compute the maximum number of motion tokens
        # max_motion_length frames -> latent_len tokens
        self.max_motion_tokens = motion_token_slots(
            max_motion_length,
            downsample_rate,
            add_time_embeds=add_time_embeds,
        )
        
        # Max text length calculation: subtract bos, eos, bom, eom and the motion tokens
        self.max_text_len = max_seq_len - self.max_motion_tokens - 4
        
        # Load data
        self._load_data(split, tiny, debug)
        
    def _load_data(self, split: str, tiny: bool, debug: bool):
        """
        Load motion-text data.
        
        Data processing steps:
        1. Read the split file to get the list of data IDs
        2. Load motion and text data for each ID
        3. Handle time tags in the text (they may correspond to motion sub-segments)
        4. Filter out data that does not meet the length requirements
        """
        split_file = os.path.join(self.data_root, f'{split}.txt')
        motion_dir = os.path.join(self.data_root, 'new_joint_vecs')
        text_dir = os.path.join(self.data_root, 'texts')
        
        # Load data IDs
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        
        # Debug/tiny mode
        if tiny or debug:
            id_list = id_list[:100]
        
        new_name_list = []
        data_dict = {}
        
        for name in id_list:
            try:
                # Load motion
                motion_path = os.path.join(motion_dir, f'{name}.npy')
                if not os.path.exists(motion_path):
                    continue
                motion = np.load(motion_path)
                
                # Load text descriptions
                text_data = []
                flag = False
                text_path = os.path.join(text_dir, f'{name}.txt')
                if not os.path.exists(text_path):
                    continue
                
                with cs.open(text_path, 'r') as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        if len(line_split) < 4:
                            continue
                            
                        caption = line_split[0]
                        t_tokens = line_split[1].split(' ') if len(line_split) > 1 else []
                        f_tag = float(line_split[2]) if line_split[2] else 0.0
                        to_tag = float(line_split[3]) if line_split[3] else 0.0
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        
                        text_dict['caption'] = caption
                        text_dict['tokens'] = t_tokens
                        
                        if f_tag == 0.0 and to_tag == 0.0:
                            # Description for the whole motion
                            flag = True
                            text_data.append(text_dict)
                        else:
                            # Description for a motion sub-segment
                            start_frame = int(f_tag * self.fps)
                            end_frame = int(to_tag * self.fps)
                            
                            if start_frame >= end_frame:
                                continue
                            
                            motion_new = motion[start_frame:end_frame]
                            new_name = f'{name}_{f_tag}_{to_tag}'
                            
                            if len(motion_new) < self.min_motion_length or \
                               len(motion_new) >= self.max_motion_length:
                                continue
                            
                            data_dict[new_name] = {
                                'motion': motion_new,
                                'text': [text_dict]
                            }
                            new_name_list.append(new_name)
                
                # Add the whole-motion sample
                if flag and not (len(motion) < self.min_motion_length or 
                                len(motion) >= self.max_motion_length):
                    data_dict[name] = {
                        'motion': motion,
                        'text': text_data
                    }
                    new_name_list.append(name)
                    
            except Exception as e:
                print(f"Error loading {name}: {e}")
                continue
        
        self.data_dict = data_dict
        self.name_list = new_name_list
        print(f'Motion dataset [{split}] loaded: {len(self.data_dict)} samples, {len(self.name_list)} entries')

        # Print debug info
        if len(self.name_list) > 0:
            print(f"\n{'='*20} Motion Dataset Info {'='*20}")
            print(f"Stage:      {self.stage}")
            print(f"Task:       {'M2T (Understanding)' if self.is_understanding else 'T2M (Generation)'}")
            print(f"Use Instructions: {self.use_instructions}")
            print(f"Num Tasks: {len(self.tasks)} (dataset length = {len(self.name_list)} * {len(self.tasks)} = {len(self)})")
            if self.use_instructions:
                print(f"T2M Instructions: {len(self.t2m_instructions)}")
                print(f"M2T Instructions: {len(self.m2t_instructions)}")
            print(f"pad_id:     {self.pad_id}")
            print(f"bos_id:     {self.bos_id}")
            print(f"eos_id:     {self.eos_id}")
            print(f"bom_id:     {self.bom_id}")
            print(f"eom_id:     {self.eom_id}")
            print(f"mot_pad_id: {self.mot_pad_id}")
            print(f"max_motion_tokens: {self.max_motion_tokens}")
            print(f"max_text_len: {self.max_text_len}")
            print(f"downsample_rate: {self.downsample_rate}")
            print(f"add_time_embeds: {self.add_time_embeds}")

            # Print sample examples
            print(f"\n{'='*20} Sample Examples {'='*20}")
            sample_indices = random.sample(range(len(self.name_list)), min(2, len(self.name_list)))
            for idx in sample_indices:
                name = self.name_list[idx]
                data = self.data_dict[name]
                print(f"\nID: {name}")
                print(f"Motion shape: {data['motion'].shape}")
                print(f"Num captions: {len(data['text'])}")
                if len(data['text']) > 0:
                    print(f"Sample caption: {data['text'][0]['caption'][:100]}...")
            print(f"{'='*60}\n")
    
    def _compute_num_motion_tokens(self, motion_length: int) -> int:
        """
        Compute the number of tokens for a given motion length.
        
        Args:
            motion_length: number of motion frames
            
        Returns:
            num_tokens: total token count including the time embedding slot
        """
        return motion_token_slots(
            motion_length,
            self.downsample_rate,
            add_time_embeds=self.add_time_embeds,
        )
    
    def __len__(self) -> int:
        """
        Return the dataset length.
        
        Consistent with MotionGPT3: len = len(data) * len(tasks)
        This ensures that within each epoch every data sample is paired with all instruction templates.
        """
        return len(self.name_list) * len(self.tasks)
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Get a single sample.
        
        Consistent with MotionGPT3: use data_idx = idx // len(tasks), task_idx = idx % len(tasks)
        This ensures every data sample is paired with all instruction templates during training.
        
        Returns:
            dict containing:
                - text_tokens: [max_seq_len] padded token sequence
                - text_labels: [max_seq_len] labels for NTP loss
                - motions: [T, 269] normalized motion features
                - motion_lengths: scalar, original motion length
                - modality_positions: [1, 2] (offset, num_tokens) for motion
                - text_masks: [max_seq_len] text token mask
                - motion_masks: [max_seq_len] motion token mask
                - texts: caption string
                - data_type: 't2m_motion' or 'm2t'
        """
        try:
            # Index computation consistent with MotionGPT3
            # data_idx = idx // len(tasks): selects which data sample to use
            # task_idx = idx % len(tasks): selects which task template to use
            data_idx = idx // len(self.tasks)
            task_idx = idx % len(self.tasks)
            
            fname = self.name_list[data_idx]
            data = self.data_dict[fname]
            text_list = data['text']
            motion = data['motion'].copy()
            
            # Randomly select a caption
            text_data = random.choice(text_list)
            caption = text_data['caption']
            
            # Get the current task config
            current_task = self.tasks[task_idx]
            
            # Random crop motion (consistent with MotionGPT3)
            # Optional: disable random cropping for M2T only, reducing "motion segment vs caption" supervision mismatch noise
            m_length = motion.shape[0]
            skip_random_crop = self.is_understanding and self.disable_random_crop_for_m2t
            if not skip_random_crop:
                # Crop with 1/3 probability, keep the original length with 2/3 probability
                coin = np.random.choice([False, False, True])  # Crop with 1/3 probability
                if coin:
                    # Align to unit_length when cropping
                    coin2 = np.random.choice([True, False])
                    if coin2:
                        m_length = (m_length // self.unit_length - 1) * self.unit_length
                    else:
                        m_length = (m_length // self.unit_length) * self.unit_length
                    
                    if m_length <= 0:
                        m_length = motion.shape[0]
                    
                    # Random start position for cropping
                    idx_start = random.randint(0, max(0, len(motion) - m_length))
                    motion = motion[idx_start:idx_start + m_length]
            # else: random cropping is disabled for M2T, keep the original motion length
            
            # Ensure motion length is valid
            if len(motion) < self.min_motion_length:
                return self.__getitem__((idx + 1) % len(self))
            
            # Z Normalization
            motion = (motion - self.mean) / self.std
            motion = torch.from_numpy(motion).float()
            
            # Calculate number of motion tokens
            num_motion_tokens = self._compute_num_motion_tokens(len(motion))
            
            # Apply instruction templates (only in the lm_instruct stage)
            use_m2t_instruct_format = False  # Flag for whether the M2T instruction format is used
            prompt_text = ""
            
            if self.use_instructions:
                # Get instruction templates from the current task config
                input_templates = current_task.get("input", [])
                
                if self.is_understanding and len(input_templates) > 0:
                    # M2T task: pick an instruction template and handle <Motion_Placeholder>
                    # In M2T, the input is motion + instruction prompt and the output is the caption
                    instruction = random.choice(input_templates)
                    # Remove <Motion_Placeholder> and keep only the instruction text
                    # The motion is embedded into the sequence via special tokens
                    prompt_text = apply_m2t_instruction_template(
                        instruction,
                        frame_length=len(motion),
                        second_length=(len(motion) / float(self.fps)),
                    )
                    use_m2t_instruct_format = True
                    # The caption keeps its original value and serves as the target output
                elif not self.is_understanding and len(input_templates) > 0:
                    # T2M task: pick an instruction template and embed the caption into it
                    instruction = random.choice(input_templates)
                    processed_caption = apply_instruction_template(caption, instruction)
                else:
                    processed_caption = caption
            else:
                # Pretraining stage: use the raw caption directly
                processed_caption = caption
            
            # Tokenize text based on task type
            if self.is_understanding and use_m2t_instruct_format:
                # M2T with instructions: tokenize the prompt and the caption separately
                prompt_tokens = self.text_tokenizer(
                    prompt_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_text_len // 2  # The prompt takes at most half
                ).input_ids
                caption_tokens = self.text_tokenizer(
                    caption,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_text_len // 2  # The caption takes at most half
                ).input_ids
            elif random.random() < self.cond_dropout_prob and not self.is_understanding:
                # Empty text for unconditional generation (CFG) - T2M only
                text_tokens = self.text_tokenizer(
                    '',
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_text_len
                ).input_ids
            else:
                # T2M task, or M2T in the pretraining stage
                text_to_tokenize = processed_caption if not self.is_understanding else caption
                text_tokens = self.text_tokenizer(
                    text_to_tokenize,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_text_len
                ).input_ids
            
            # Format sequence based on task type
            if self.is_understanding and use_m2t_instruct_format:
                # M2T with instructions: use the new formatting function
                # Sequence: [BOS][BOM][MOT*N][EOM] prompt caption [EOS]
                # Only the caption part is predicted
                text_tokens, text_labels, modality_positions, text_mask, motion_mask = \
                    format_sequence_motion_und_instruct(
                        prompt_tokens, caption_tokens, self.bos_id, self.eos_id,
                        self.bom_id, self.eom_id, self.pad_id, self.mot_pad_id,
                        num_motion_tokens, self.max_seq_len
                    )
            elif self.is_understanding:
                # Motion Understanding: [BOS][BOM][MOT_PAD*N][EOM] text [EOS]
                text_tokens, text_labels, modality_positions, text_mask, motion_mask = \
                    format_sequence_motion_und(
                        text_tokens, self.bos_id, self.eos_id,
                        self.bom_id, self.eom_id, self.pad_id, self.mot_pad_id,
                        num_motion_tokens, self.max_seq_len
                    )
            else:
                # Text-to-Motion: [BOS] text [BOM][MOT_PAD*N][EOM][EOS]
                text_tokens, text_labels, modality_positions, text_mask, motion_mask = \
                    format_sequence_motion_gen(
                        text_tokens, self.bos_id, self.eos_id,
                        self.bom_id, self.eom_id, self.pad_id, self.mot_pad_id,
                        num_motion_tokens, self.max_seq_len
                    )
            
            sample = {
                'text_tokens': text_tokens,
                'text_labels': text_labels,
                'motions': motion,  # [T, 263] normalized motion features
                'motion_lengths': torch.tensor(len(motion)),  # Original motion length (frames)
                'modality_positions': modality_positions,
                'text_masks': text_mask,
                'motion_masks': motion_mask,
                'texts': caption,
                'data_type': self.data_type
            }
            
            return sample
            
        except Exception as e:
            print(f"Error in __getitem__ {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """
        Custom collate function to handle variable-length motions.
        
        Pad the motions in a batch to the same length.
        """
        batched = collections.defaultdict(list)
        
        # Find max motion length in batch
        max_motion_len = max(sample['motions'].shape[0] for sample in batch)
        # Align to unit_length
        if max_motion_len % self.unit_length != 0:
            max_motion_len = ((max_motion_len // self.unit_length) + 1) * self.unit_length
        
        for data in batch:
            for k, v in data.items():
                if k == 'motions':
                    # Pad motion to max length
                    motion = v
                    pad_len = max_motion_len - motion.shape[0]
                    if pad_len > 0:
                        motion = torch.cat([
                            motion,
                            torch.zeros(pad_len, motion.shape[1])
                        ], dim=0)
                    batched[k].append(motion)
                else:
                    batched[k].append(v)
        
        # Stack tensors
        result = {}
        for k, v in batched.items():
            if k in ('texts', 'data_type'):
                result[k] = v
            else:
                result[k] = torch.stack(v, dim=0)
        
        return result


class MotionPredictionDataset(Dataset):
    """
    Dataset for Motion Prediction task (Motion-to-Motion).
    
    Given the first part of a motion, predict the rest.
    - Source motion (s1): the first predict_ratio (default 20%) of the motion as the input condition
    - Target motion (s2): the remaining (1-predict_ratio) of the motion as the prediction target
    
    Data format:
    - Input: [BOS] "Predict motion:" [BOM][SOURCE_MOT][EOM] [BOM][TARGET_MOT][EOM][EOS]
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
        min_motion_length: int = 40,  # Longer motions are needed for splitting
        unit_length: int = 4,
        fps: int = 20,
        downsample_rate: int = 4,
        predict_ratio: float = 0.2,  # First 20% as input
        showo_token_ids: Optional[Dict[str, int]] = None,
        motion_latent_dim: int = 256,
        split: str = 'train',
        tiny: bool = False,
        debug: bool = False,
        add_time_embeds: bool = True,
        stage: str = 'lm_pretrain',
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
        self.predict_ratio = predict_ratio
        self.motion_latent_dim = motion_latent_dim
        self.add_time_embeds = add_time_embeds
        self.data_type = "pred"  # Motion Prediction
        
        # Training stage configuration
        self.stage = stage
        # Consistent with MotionGPT3: both lm_instruct and lm_finetune stages use instruction templates
        self.use_instructions = (stage in ['lm_instruct', 'lm_finetune'])
        
        # Load instruction templates
        self.pred_instructions = ["Predict motion:"]  # Default template
        if instruction_template_path and os.path.exists(instruction_template_path):
            templates = load_instruction_templates(instruction_template_path)
            self.pred_instructions = get_pred_instructions(templates)
            if not self.pred_instructions:
                self.pred_instructions = ["Predict motion:"]
            print(f"Loaded {len(self.pred_instructions)} Prediction instructions")
        
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
        """Load data, keeping only motions long enough to split."""
        split_file = os.path.join(self.data_root, f'{split}.txt')
        motion_dir = os.path.join(self.data_root, 'new_joint_vecs')
        text_dir = os.path.join(self.data_root, 'texts')
        
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        
        if tiny or debug:
            id_list = id_list[:100]
        
        new_name_list = []
        data_dict = {}
        
        for name in id_list:
            try:
                motion_path = os.path.join(motion_dir, f'{name}.npy')
                if not os.path.exists(motion_path):
                    continue
                motion = np.load(motion_path)
                
                # Motion Prediction needs longer motions for effective splitting
                min_len_for_pred = max(self.min_motion_length, 
                                       int(self.unit_length / self.predict_ratio) + self.unit_length)
                if len(motion) < min_len_for_pred or len(motion) >= self.max_motion_length:
                    continue
                
                # Load text (optional; the prediction task does not need text)
                text_data = []
                text_path = os.path.join(text_dir, f'{name}.txt')
                if os.path.exists(text_path):
                    with cs.open(text_path, 'r') as f:
                        for line in f.readlines():
                            line_split = line.strip().split('#')
                            if len(line_split) >= 1:
                                text_data.append({'caption': line_split[0]})
                
                data_dict[name] = {
                    'motion': motion,
                    'text': text_data if text_data else [{'caption': ''}]
                }
                new_name_list.append(name)
                
            except Exception as e:
                print(f"Error loading {name}: {e}")
                continue
        
        self.data_dict = data_dict
        self.name_list = new_name_list
        print(f'Motion Prediction dataset [{split}] loaded: {len(self.data_dict)} samples')
        
        if len(self.name_list) > 0:
            print(f"\n{'='*20} Motion Prediction Dataset Info {'='*20}")
            print(f"Stage:          {self.stage}")
            print(f"Task:           Motion Prediction")
            print(f"Predict ratio:  {self.predict_ratio}")
            print(f"Min motion len: {self.min_motion_length}")
            print(f"{'='*60}\n")
    
    def _compute_num_motion_tokens(self, motion_length: int, add_time_embed: bool = None) -> int:
        """Compute the number of tokens for a given motion length.
        
        Args:
            motion_length: number of raw motion frames
            add_time_embed: whether to add a time embedding slot. If None, use self.add_time_embeds
            
        Returns:
            number of motion tokens
        """
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
                - source_motion: [T1, 263] source motion (first part)
                - target_motion: [T2, 263] target motion (remaining part)
                - source_motion_length: source motion length
                - target_motion_length: target motion length
                - source_modality_positions: [1, 2] source motion position
                - target_modality_positions: [1, 2] target motion position
                - text_masks: [max_seq_len] text token mask
                - source_motion_masks: [max_seq_len] source motion token mask
                - target_motion_masks: [max_seq_len] target motion token mask
                - data_type: 'pred'
        """
        try:
            fname = self.name_list[idx]
            data = self.data_dict[fname]
            motion = data['motion'].copy()
            
            # Align to unit_length
            m_length = (len(motion) // self.unit_length) * self.unit_length
            if m_length < self.min_motion_length:
                return self.__getitem__((idx + 1) % len(self))
            motion = motion[:m_length]
            
            # Split the motion: source (first predict_ratio) + target (remaining 1-predict_ratio)
            split_point = int(m_length * self.predict_ratio)
            # Align to unit_length
            split_point = (split_point // self.unit_length) * self.unit_length
            min_segment_len = 1
            if split_point < min_segment_len:
                split_point = min_segment_len
            if m_length - split_point < min_segment_len:
                return self.__getitem__((idx + 1) % len(self))
            
            source_motion = motion[:split_point]
            target_motion = motion[split_point:]
            
            # Z Normalization
            source_motion = (source_motion - self.mean) / self.std
            target_motion = (target_motion - self.mean) / self.std
            source_motion = torch.from_numpy(source_motion).float()
            target_motion = torch.from_numpy(target_motion).float()
            
            # Calculate number of motion tokens
            # Source motion: conditioning input, no time embedding needed (uses the fused path)
            # Target motion: generation target, needs time embedding (uses the gen path)
            num_source_tokens = self._compute_num_motion_tokens(len(source_motion), add_time_embed=False)
            num_target_tokens = self._compute_num_motion_tokens(len(target_motion), add_time_embed=True)
            
            # Get the prompt (a simple template, or randomly chosen from the instructions)
            if self.use_instructions and len(self.pred_instructions) > 0:
                prompt_template = random.choice(self.pred_instructions)
                # Remove the placeholder and keep only the text prompt
                prompt = prompt_template.replace('<Motion_Placeholder_s1>', '').strip()
                if not prompt:
                    prompt = "Predict motion:"
            else:
                prompt = "Predict motion:"
            
            # Tokenize prompt
            text_tokens = self.text_tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=50  # Short prompt
            ).input_ids
            
            # Format sequence
            (text_tokens_tensor, text_labels, 
             source_modality_positions, target_modality_positions,
             text_mask, source_motion_mask, target_motion_mask) = format_sequence_motion_pred(
                text_tokens, self.bos_id, self.eos_id,
                self.bom_id, self.eom_id, self.pad_id, self.mot_pad_id,
                num_source_tokens, num_target_tokens, self.max_seq_len
            )
            
            sample = {
                'text_tokens': text_tokens_tensor,
                'text_labels': text_labels,
                'source_motion': source_motion,
                'target_motion': target_motion,
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
            print(f"Error in MotionPredictionDataset __getitem__ {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """Custom collate function for motion prediction."""
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
            if k == 'data_type':
                result[k] = v
            else:
                result[k] = torch.stack(v, dim=0)
        
        return result


class MotionTextDatasetPrecomputed(MotionTextDataset):
    """
    Precomputed version of the Motion-Text dataset.
    
    Loads motion latents pre-encoded by the VAE, saving computation during training.
    Run preprocess_motion.py first to generate the precomputed data.
    """
    
    def __init__(
        self,
        data_root: str,
        latent_dir: str,
        text_tokenizer: Any,
        mean_path: str = None,
        std_path: str = None,
        max_seq_len: int = 512,
        max_motion_length: int = 200,
        min_motion_length: int = 20,
        unit_length: int = 4,
        fps: int = 20,
        downsample_rate: int = 4,
        is_understanding: bool = False,
        cond_dropout_prob: float = 0.1,
        showo_token_ids: Optional[Dict[str, int]] = None,
        motion_latent_dim: int = 256,
        split: str = 'train',
        tiny: bool = False,
        debug: bool = False,
        add_time_embeds: bool = True,
        # New: training stage configuration
        stage: str = 'lm_pretrain',
        instruction_template_path: str = None,
        **kwargs,
    ):
        """
        Args:
            latent_dir: precomputed latent directory containing {id}.npz files
                       each file contains 'latent' and 'length' fields
            stage: training stage ('lm_pretrain' or 'lm_instruct')
            instruction_template_path: path to the instruction template JSON file
        """
        self.latent_dir = latent_dir
        super().__init__(
            data_root=data_root,
            text_tokenizer=text_tokenizer,
            mean_path=mean_path,
            std_path=std_path,
            max_seq_len=max_seq_len,
            max_motion_length=max_motion_length,
            min_motion_length=min_motion_length,
            unit_length=unit_length,
            fps=fps,
            downsample_rate=downsample_rate,
            is_understanding=is_understanding,
            cond_dropout_prob=cond_dropout_prob,
            showo_token_ids=showo_token_ids,
            motion_latent_dim=motion_latent_dim,
            split=split,
            tiny=tiny,
            debug=debug,
            add_time_embeds=add_time_embeds,
            stage=stage,
            instruction_template_path=instruction_template_path,
            **kwargs,
        )
    
    def _load_data(self, split: str, tiny: bool, debug: bool):
        """Load data and check that the precomputed latents exist."""
        # First call the parent class to load the raw data
        super()._load_data(split, tiny, debug)
        
        # Filter out samples without precomputed latents
        valid_names = []
        for name in self.name_list:
            # Handle sub-segment naming
            safe_name = name.replace('.', '_')
            latent_path = os.path.join(self.latent_dir, f'{safe_name}.npz')
            if os.path.exists(latent_path):
                valid_names.append(name)
        
        print(f"Precomputed latents: {len(valid_names)}/{len(self.name_list)} available")
        self.name_list = valid_names
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Get a precomputed sample containing motion_latents instead of the raw motion.
        
        Consistent with MotionGPT3: use data_idx = idx // len(tasks), task_idx = idx % len(tasks)
        """
        try:
            # Index computation consistent with MotionGPT3
            data_idx = idx // len(self.tasks)
            task_idx = idx % len(self.tasks)
            
            fname = self.name_list[data_idx]
            data = self.data_dict[fname]
            text_list = data['text']
            
            # Load precomputed latent
            safe_name = fname.replace('.', '_')
            latent_path = os.path.join(self.latent_dir, f'{safe_name}.npz')
            latent_data = np.load(latent_path)
            motion_latent = torch.from_numpy(latent_data['latent']).float()  # [latent_len, 256]
            motion_length = int(latent_data['length'])
            
            # Randomly select a caption
            text_data = random.choice(text_list)
            caption = text_data['caption']
            
            # Get the current task config
            current_task = self.tasks[task_idx]
            
            # Calculate number of motion tokens
            latent_len = motion_latent.shape[0]
            num_motion_tokens = latent_len + 1 if self.add_time_embeds else latent_len
            
            # Apply instruction templates (only in the lm_instruct stage)
            use_m2t_instruct_format = False
            prompt_text = ""
            
            if self.use_instructions:
                input_templates = current_task.get("input", [])
                
                if self.is_understanding and len(input_templates) > 0:
                    instruction = random.choice(input_templates)
                    prompt_text = apply_m2t_instruction_template(
                        instruction,
                        frame_length=motion_length,
                        second_length=(motion_length / float(self.fps)),
                    )
                    use_m2t_instruct_format = True
                elif not self.is_understanding and len(input_templates) > 0:
                    instruction = random.choice(input_templates)
                    processed_caption = apply_instruction_template(caption, instruction)
                else:
                    processed_caption = caption
            else:
                processed_caption = caption
            
            # Tokenize text based on task type
            if self.is_understanding and use_m2t_instruct_format:
                prompt_tokens = self.text_tokenizer(
                    prompt_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_text_len // 2
                ).input_ids
                caption_tokens = self.text_tokenizer(
                    caption,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_text_len // 2
                ).input_ids
            elif random.random() < self.cond_dropout_prob and not self.is_understanding:
                text_tokens = self.text_tokenizer(
                    '',
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_text_len
                ).input_ids
            else:
                text_to_tokenize = processed_caption if not self.is_understanding else caption
                text_tokens = self.text_tokenizer(
                    text_to_tokenize,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_text_len
                ).input_ids
            
            # Format sequence based on task type
            if self.is_understanding and use_m2t_instruct_format:
                # M2T with instructions
                text_tokens, text_labels, modality_positions, text_mask, motion_mask = \
                    format_sequence_motion_und_instruct(
                        prompt_tokens, caption_tokens, self.bos_id, self.eos_id,
                        self.bom_id, self.eom_id, self.pad_id, self.mot_pad_id,
                        num_motion_tokens, self.max_seq_len
                    )
            elif self.is_understanding:
                text_tokens, text_labels, modality_positions, text_mask, motion_mask = \
                    format_sequence_motion_und(
                        text_tokens, self.bos_id, self.eos_id,
                        self.bom_id, self.eom_id, self.pad_id, self.mot_pad_id,
                        num_motion_tokens, self.max_seq_len
                    )
            else:
                text_tokens, text_labels, modality_positions, text_mask, motion_mask = \
                    format_sequence_motion_gen(
                        text_tokens, self.bos_id, self.eos_id,
                        self.bom_id, self.eom_id, self.pad_id, self.mot_pad_id,
                        num_motion_tokens, self.max_seq_len
                    )
            
            sample = {
                'text_tokens': text_tokens,
                'text_labels': text_labels,
                'motion_latents': motion_latent,  # [latent_len, 256] precomputed latent
                'motion_lengths': torch.tensor(motion_length),
                'modality_positions': modality_positions,
                'text_masks': text_mask,
                'motion_masks': motion_mask,
                'texts': caption,
                'data_type': self.data_type
            }
            
            return sample
            
        except Exception as e:
            print(f"Error in __getitem__ {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """Custom collate for precomputed latents."""
        batched = collections.defaultdict(list)
        
        # Find max latent length in batch
        max_latent_len = max(sample['motion_latents'].shape[0] for sample in batch)
        
        for data in batch:
            for k, v in data.items():
                if k == 'motion_latents':
                    # Pad latent to max length
                    latent = v
                    pad_len = max_latent_len - latent.shape[0]
                    if pad_len > 0:
                        latent = torch.cat([
                            latent,
                            torch.zeros(pad_len, latent.shape[1])
                        ], dim=0)
                    batched[k].append(latent)
                else:
                    batched[k].append(v)
        
        # Stack tensors
        result = {}
        for k, v in batched.items():
            if k in ('texts', 'data_type'):
                result[k] = v
            else:
                result[k] = torch.stack(v, dim=0)
        
        return result


class TextToTextDataset(Dataset):
    """
    Dataset for Text-to-Text (T2T) tasks in Stage 3.
    
    T2T task types (fully replicating MotionGPT3 Stage 3 for a fair comparison):
    - caption-to-framelen: predict the motion frame count from the caption
    - framelen-to-caption: generate a plausible motion description from the frame count
    - second-to-caption: generate a description from the duration in seconds
    - n2t (random-caption): generate a motion description unconditionally
    
    This is a text-only task without the motion modality.
    Used to strengthen the model's instruction following and motion-related text understanding.
    
    Data source: reuses the text descriptions and motion length info of the HumanML3D dataset.
    """

    def __init__(
        self,
        data_root: str,
        text_tokenizer: Any,
        max_seq_len: int = 512,
        max_motion_length: int = 200,
        min_motion_length: int = 20,
        fps: int = 20,
        downsample_rate: int = 4,
        showo_token_ids: Optional[Dict[str, int]] = None,
        split: str = 'train',
        tiny: bool = False,
        debug: bool = False,
        instruction_template_path: str = None,
        **kwargs,
    ):
        """
        Initialize the T2T dataset.
        
        Args:
            data_root: HumanML3D dataset root directory
            text_tokenizer: text tokenizer
            max_seq_len: maximum sequence length
            max_motion_length: maximum number of motion frames (used to filter data)
            min_motion_length: minimum number of motion frames
            fps: frame rate (used to compute the duration in seconds)
            downsample_rate: downsample rate (used by some T2T tasks)
            showo_token_ids: dict of special token IDs
            split: dataset split
            tiny: whether to use a tiny subset
            debug: whether to run in debug mode
            instruction_template_path: path to the instruction template JSON file
        """
        super().__init__()
        
        self.data_root = data_root
        self.text_tokenizer = text_tokenizer
        self.max_seq_len = max_seq_len
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.fps = fps
        self.downsample_rate = downsample_rate
        self.data_type = "t2t"
        
        # Token IDs
        self.pad_id = text_tokenizer.pad_token_id
        self.bos_id = showo_token_ids['bos_id']
        self.eos_id = showo_token_ids['eos_id']
        
        # Compute the maximum text length (the T2T task needs no motion tokens)
        self.max_text_len = max_seq_len - 2  # Subtract bos, eos
        
        # Load T2T task templates
        self.t2t_tasks = []
        if instruction_template_path and os.path.exists(instruction_template_path):
            templates = load_instruction_templates(instruction_template_path)
            self.t2t_tasks = get_t2t_tasks(templates)
            print(f"Loaded {len(self.t2t_tasks)} T2T tasks")
        
        if len(self.t2t_tasks) == 0:
            print("Warning: No T2T tasks found in template, using default")
            self.t2t_tasks = [{
                "class": "t2t",
                "input": ["Predict the frame count for: <Caption_Placeholder>"],
                "output": ["The motion has approximately <Frame_Placeholder> frames."]
            }]
        
        # Load data
        self._load_data(split, tiny, debug)
        
    def _load_data(self, split: str, tiny: bool, debug: bool):
        """
        Load data: reuse the text descriptions and motion length info of HumanML3D.
        """
        split_file = os.path.join(self.data_root, f'{split}.txt')
        motion_dir = os.path.join(self.data_root, 'new_joint_vecs')
        text_dir = os.path.join(self.data_root, 'texts')
        
        # Load data IDs
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        
        if tiny or debug:
            id_list = id_list[:100]
        
        new_name_list = []
        data_dict = {}
        
        for name in id_list:
            try:
                # Load motion (only for its length)
                motion_path = os.path.join(motion_dir, f'{name}.npy')
                if not os.path.exists(motion_path):
                    continue
                motion = np.load(motion_path)
                motion_length = len(motion)
                
                # Filter by length
                if motion_length < self.min_motion_length or motion_length >= self.max_motion_length:
                    continue
                
                # Load text descriptions
                text_data = []
                text_path = os.path.join(text_dir, f'{name}.txt')
                if not os.path.exists(text_path):
                    continue
                
                with cs.open(text_path, 'r') as f:
                    for line in f.readlines():
                        line_split = line.strip().split('#')
                        if len(line_split) >= 1:
                            caption = line_split[0]
                            if caption:
                                text_data.append(caption)
                
                if len(text_data) > 0:
                    data_dict[name] = {
                        'captions': text_data,
                        'motion_length': motion_length,  # Frame count
                        'duration_seconds': motion_length / self.fps  # Duration in seconds
                    }
                    new_name_list.append(name)
                    
            except Exception as e:
                print(f"Error loading {name}: {e}")
                continue
        
        self.data_dict = data_dict
        self.name_list = new_name_list
        print(f'T2T dataset [{split}] loaded: {len(self.data_dict)} samples')
        
        if len(self.name_list) > 0:
            print(f"\n{'='*20} T2T Dataset Info {'='*20}")
            print(f"Task: Text-to-Text (Stage 3)")
            print(f"Num tasks: {len(self.t2t_tasks)}")
            print(f"Dataset length: {len(self.name_list)} * {len(self.t2t_tasks)} = {len(self)}")
            print(f"{'='*60}\n")
    
    def __len__(self) -> int:
        """
        Return the dataset length.
        Consistent with MotionGPT3: len = len(data) * len(tasks)
        """
        return len(self.name_list) * len(self.t2t_tasks)
    
    def _apply_t2t_template(
        self, 
        task_config: Dict[str, Any], 
        caption: str, 
        motion_length: int,
        duration_seconds: float
    ) -> Tuple[str, str]:
        """
        Apply a T2T task template to produce the input and output text.
        
        Args:
            task_config: task config dict containing class, input, output
            caption: original caption
            motion_length: number of motion frames
            duration_seconds: motion duration in seconds
            
        Returns:
            input_text: input text
            output_text: output text
        """
        task_class = task_config.get("class", "t2t")
        input_templates = task_config.get("input", ["<Caption_Placeholder>"])
        output_templates = task_config.get("output", ["<Caption_Placeholder>"])
        
        # Randomly pick one input template and one output template
        input_template = random.choice(input_templates)
        output_template = random.choice(output_templates)
        
        # Replace placeholders
        input_text = input_template
        output_text = output_template
        
        # Replace common placeholders
        input_text = input_text.replace("<Caption_Placeholder>", caption)
        input_text = input_text.replace("<Frame_Placeholder>", str(motion_length))
        input_text = input_text.replace("<Second_Placeholder>", f"{duration_seconds:.1f}")
        
        output_text = output_text.replace("<Caption_Placeholder>", caption)
        output_text = output_text.replace("<Frame_Placeholder>", str(motion_length))
        output_text = output_text.replace("<Second_Placeholder>", f"{duration_seconds:.1f}")
        
        return input_text, output_text
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Get a single sample.
        
        Index computation consistent with MotionGPT3.
        """
        try:
            # Index computation
            data_idx = idx // len(self.t2t_tasks)
            task_idx = idx % len(self.t2t_tasks)
            
            fname = self.name_list[data_idx]
            data = self.data_dict[fname]
            
            # Randomly pick a caption
            caption = random.choice(data['captions'])
            motion_length = data['motion_length']
            duration_seconds = data['duration_seconds']
            
            # Get the current task config and apply the template
            current_task = self.t2t_tasks[task_idx]
            input_text, output_text = self._apply_t2t_template(
                current_task, caption, motion_length, duration_seconds
            )
            
            # Tokenize
            input_tokens = self.text_tokenizer(
                input_text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_text_len // 2
            ).input_ids
            
            output_tokens = self.text_tokenizer(
                output_text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_text_len // 2
            ).input_ids
            
            # Format sequence
            text_tokens, text_labels, text_mask = format_sequence_text_to_text(
                input_tokens, output_tokens,
                self.bos_id, self.eos_id, self.pad_id, self.max_seq_len
            )
            
            sample = {
                'text_tokens': text_tokens,
                'text_labels': text_labels,
                'text_masks': text_mask,
                'texts': caption,
                'input_text': input_text,
                'output_text': output_text,
                'sample_id': fname,
                'task_index': task_idx,
                'task_class': current_task.get("class", "t2t"),
                'data_type': self.data_type
            }
            
            return sample
            
        except Exception as e:
            print(f"Error in T2TDataset __getitem__ {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """Custom collate function for T2T task."""
        batched = collections.defaultdict(list)
        
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        
        # Stack tensors
        result = {}
        for k, v in batched.items():
            if k in ('texts', 'input_text', 'output_text', 'data_type', 'sample_id', 'task_index', 'task_class'):
                result[k] = v
            else:
                result[k] = torch.stack(v, dim=0)
        
        return result


def create_motion_dataloader(
    data_root: str,
    batch_size: int,
    text_tokenizer: Any,
    accelerator: Optional[Any] = None,
    max_seq_len: int = 512,
    max_motion_length: int = 200,
    min_motion_length: int = 20,
    downsample_rate: int = 4,
    motion_latent_dim: int = 256,
    cond_dropout_prob: float = 0.1,
    num_workers: int = 4,
    is_understanding: bool = False,
    shuffle: bool = True,
    drop_last: bool = True,
    showo_token_ids: Optional[Dict[str, int]] = None,
    split: str = 'train',
    mean_path: str = None,
    std_path: str = None,
    latent_dir: str = None,
    add_time_embeds: bool = True,
    tiny: bool = False,
    debug: bool = False,
    # New: training stage configuration
    stage: str = 'lm_pretrain',
    instruction_template_path: str = None,
    # New: task type parameters
    task_type: str = 't2m',  # 't2m', 'm2t', 'pred', 'edit', 'm2m', or 't2t'
    predict_ratio: float = 0.2,
    fps: int = 20,  # Used by the T2T task to compute seconds
    **kwargs
) -> DataLoader:
    """
    Create a motion-text dataloader.
    
    Args:
        data_root: HumanML3D dataset root directory
        batch_size: batch size
        text_tokenizer: text tokenizer
        accelerator: Hugging Face Accelerator (optional)
        max_seq_len: maximum sequence length
        max_motion_length: maximum number of motion frames
        min_motion_length: minimum number of motion frames
        downsample_rate: Motion VAE downsample rate
        motion_latent_dim: Motion VAE latent dimension
        cond_dropout_prob: condition dropout probability
        num_workers: number of data loading workers
        is_understanding: whether this is the understanding task
        shuffle: whether to shuffle
        drop_last: whether to drop the last incomplete batch
        showo_token_ids: dict of special token IDs
        split: dataset split
        mean_path: path to Mean.npy
        std_path: path to Std.npy
        latent_dir: precomputed latent directory (if provided, the precomputed dataset is used)
        add_time_embeds: whether to add a time embedding slot
        tiny: whether to use a tiny subset
        debug: whether to run in debug mode
        stage: training stage ('lm_pretrain', 'lm_instruct', or 'lm_finetune')
        instruction_template_path: path to the instruction template JSON file
        task_type: task type ('t2m', 'm2t', 'pred', 'edit', 'm2m', or 't2t')
        fps: frame rate (used by the T2T task to compute seconds)
        
    Returns:
        DataLoader instance
    """
    # Select the dataset by task type
    if task_type == 'm2m':
        from datasets.motion_recon_dataset import MotionReconDataset
        dataset_name = kwargs.pop('dataset_name', 'humanml3d')
        bottleneck = kwargs.pop('bottleneck', None)
        unit_length = kwargs.pop('unit_length', 1)
        print(f"Creating Motion Reconstruction dataset: {dataset_name}")
        dataset = MotionReconDataset(
            data_root=data_root,
            dataset_name=dataset_name,
            text_tokenizer=text_tokenizer,
            mean_path=mean_path,
            std_path=std_path,
            max_seq_len=max_seq_len,
            max_motion_length=max_motion_length,
            min_motion_length=min_motion_length,
            unit_length=unit_length,
            downsample_rate=downsample_rate,
            showo_token_ids=showo_token_ids,
            split=split,
            tiny=tiny,
            debug=debug,
            add_time_embeds=add_time_embeds,
            instruction_template_path=instruction_template_path,
            bottleneck=bottleneck,
            **kwargs
        )
    elif task_type == 't2t':
        # Text-to-Text dataset (Stage 3)
        print(f"Creating Text-to-Text (T2T) dataset for Stage 3")
        dataset = TextToTextDataset(
            data_root=data_root,
            text_tokenizer=text_tokenizer,
            max_seq_len=max_seq_len,
            max_motion_length=max_motion_length,
            min_motion_length=min_motion_length,
            fps=fps,
            downsample_rate=downsample_rate,
            showo_token_ids=showo_token_ids,
            split=split,
            tiny=tiny,
            debug=debug,
            instruction_template_path=instruction_template_path,
            **kwargs
        )
    elif task_type == 'edit':
        # Motion Edit dataset (MotionFix)
        from datasets.motion_edit_dataset import MotionEditDataset
        print(f"Creating Motion Edit dataset (MotionFix)")
        # Pop unit_length from kwargs to avoid passing it twice
        unit_length = kwargs.pop('unit_length', 1)
        dataset = MotionEditDataset(
            data_root=data_root,  # MotionFix dataset root directory
            text_tokenizer=text_tokenizer,
            mean_path=mean_path,
            std_path=std_path,
            max_seq_len=max_seq_len,
            max_motion_length=max_motion_length,
            min_motion_length=min_motion_length,
            unit_length=unit_length,
            fps=fps,
            downsample_rate=downsample_rate,
            showo_token_ids=showo_token_ids,
            motion_latent_dim=motion_latent_dim,
            split=split,
            tiny=tiny,
            debug=debug,
            add_time_embeds=add_time_embeds,
            instruction_template_path=instruction_template_path,
        )
    elif task_type == 'pred':
        # Motion Prediction dataset
        print(f"Creating Motion Prediction dataset with predict_ratio={predict_ratio}")
        dataset = MotionPredictionDataset(
            data_root=data_root,
            text_tokenizer=text_tokenizer,
            mean_path=mean_path,
            std_path=std_path,
            max_seq_len=max_seq_len,
            max_motion_length=max_motion_length,
            min_motion_length=max(min_motion_length, 40),  # Prediction needs longer motions
            downsample_rate=downsample_rate,
            predict_ratio=predict_ratio,
            showo_token_ids=showo_token_ids,
            motion_latent_dim=motion_latent_dim,
            split=split,
            add_time_embeds=add_time_embeds,
            tiny=tiny,
            debug=debug,
            stage=stage,
            instruction_template_path=instruction_template_path,
            **kwargs
        )
    elif latent_dir is not None and os.path.exists(latent_dir):
        print(f"Using precomputed latents from: {latent_dir}")
        dataset = MotionTextDatasetPrecomputed(
            data_root=data_root,
            latent_dir=latent_dir,
            text_tokenizer=text_tokenizer,
            mean_path=mean_path,
            std_path=std_path,
            max_seq_len=max_seq_len,
            max_motion_length=max_motion_length,
            min_motion_length=min_motion_length,
            downsample_rate=downsample_rate,
            motion_latent_dim=motion_latent_dim,
            cond_dropout_prob=cond_dropout_prob,
            is_understanding=is_understanding,
            showo_token_ids=showo_token_ids,
            split=split,
            add_time_embeds=add_time_embeds,
            tiny=tiny,
            debug=debug,
            stage=stage,
            instruction_template_path=instruction_template_path,
            **kwargs
        )
    else:
        dataset = MotionTextDataset(
            data_root=data_root,
            text_tokenizer=text_tokenizer,
            mean_path=mean_path,
            std_path=std_path,
            max_seq_len=max_seq_len,
            max_motion_length=max_motion_length,
            min_motion_length=min_motion_length,
            downsample_rate=downsample_rate,
            motion_latent_dim=motion_latent_dim,
            cond_dropout_prob=cond_dropout_prob,
            is_understanding=is_understanding,
            showo_token_ids=showo_token_ids,
            split=split,
            add_time_embeds=add_time_embeds,
            tiny=tiny,
            debug=debug,
            stage=stage,
            instruction_template_path=instruction_template_path,
            **kwargs
        )
    
    # Distributed training support
    if accelerator is not None and accelerator.num_processes > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=shuffle,
            drop_last=drop_last
        )
        shuffle = False
    else:
        sampler = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=dataset.collate_fn,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True
    )
    
    return dataloader


# ======================= Test Script =======================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from models.misc import get_text_tokenizer
    
    print("Loading tokenizer...")
    text_tokenizer, showo_token_ids = get_text_tokenizer(
        "Qwen/Qwen2.5-7B-Instruct",
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name="qwen2_5"
    )
    
    print("\n" + "="*60)
    print("Testing MotionTextDataset (Text-to-Motion)")
    print("="*60)
    
    dataset = MotionTextDataset(
        data_root="data/UNI_MOTION_269/HumanML3D",
        text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids,
        max_seq_len=512,
        max_motion_length=200,
        min_motion_length=20,
        downsample_rate=4,
        is_understanding=False,
        split='train',
        tiny=True,
        debug=True
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test single sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"text_tokens shape: {sample['text_tokens'].shape}")
    print(f"motions shape: {sample['motions'].shape}")
    print(f"motion_lengths: {sample['motion_lengths']}")
    print(f"modality_positions: {sample['modality_positions']}")
    print(f"data_type: {sample['data_type']}")
    print(f"text: {sample['texts'][:100]}...")
    
    # Test dataloader
    print("\n" + "="*60)
    print("Testing DataLoader")
    print("="*60)
    
    dataloader = create_motion_dataloader(
        data_root="data/UNI_MOTION_269/HumanML3D",
        batch_size=4,
        text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids,
        max_seq_len=512,
        split='train',
        num_workers=0,
        tiny=True
    )
    
    for batch in dataloader:
        print(f"\nBatch keys: {batch.keys()}")
        print(f"text_tokens shape: {batch['text_tokens'].shape}")
        print(f"motions shape: {batch['motions'].shape}")
        print(f"motion_lengths: {batch['motion_lengths']}")
        print(f"modality_positions shape: {batch['modality_positions'].shape}")
        break
    
    print("\n" + "="*60)
    print("Testing MotionTextDataset (Motion-to-Text / Understanding)")
    print("="*60)
    
    dataset_und = MotionTextDataset(
        data_root="data/UNI_MOTION_269/HumanML3D",
        text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids,
        max_seq_len=512,
        is_understanding=True,
        split='train',
        tiny=True
    )
    
    sample_und = dataset_und[0]
    print(f"data_type: {sample_und['data_type']}")
    print(f"modality_positions: {sample_und['modality_positions']}")
    
    print("\nSelf-test finished.")
