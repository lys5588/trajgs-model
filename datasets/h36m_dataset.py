# coding=utf-8
"""
H36M Dataset for Stage 3: Image-to-Pose (img2pose) and Image-to-Text (img2text) tasks.

Data layout (H36M-specific structure):
- Motion data: a single npy file, shape=(N_frames, 269), all frames concatenated
- Image paths: a single npy file, shape=(N_frames,), frame-aligned with the motion
- Text descriptions: nested directories text/S{id}/sequence_dir/frame.txt
- Mean.npy, Std.npy: normalization statistics

Each sample is a single-frame (image, motion_frame, text_description) triplet.
- img2pose: input image -> output single-frame motion (1 frame still maps to 1 latent token at downsample_rate=4,
  because the Motion VAE internally zero-pads to a multiple of 4, then crops back to the original length of 1)
- img2text: input image -> output text description
"""

import os
import re
import json
import random
import collections
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from datasets.utils import format_sequence_img2pose, format_sequence_img2text


# ==================== Instruction Template Loading ====================

def load_img_instruction_templates(template_path: str) -> Dict[str, Any]:
    """Load the image-task instruction template JSON file."""
    with open(template_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_img2pose_instructions(templates: Dict[str, Any]) -> List[str]:
    """Extract the img2pose instruction list."""
    instructions = []
    if "Image-to-Pose" in templates:
        for task_name, task_data in templates["Image-to-Pose"].items():
            if task_data.get("class") == "img2pose":
                instructions.extend(task_data.get("input", []))
    return instructions


def get_img2text_instructions(templates: Dict[str, Any]) -> List[str]:
    """Extract the img2text instruction list."""
    instructions = []
    if "Image-to-Text" in templates:
        for task_name, task_data in templates["Image-to-Text"].items():
            if task_data.get("class") == "img2text":
                instructions.extend(task_data.get("input", []))
    return instructions


# ==================== Path Mapping ====================

def image_path_to_text_path(image_rel_path: str, text_root: str) -> Optional[str]:
    """
    Map a relative image path to the corresponding text file path.
    
    Image path format: S1/S1_Directions_1.54138969/S1_Directions_1.54138969_000001.jpg
    Text path format: S1/S1_Directions.54138969/S1_Directions.54138969_000005.txt
    
    Differences:
    1. The trial number is removed from the directory/file name (e.g., "_1" before the camera ID)
    2. The frame number is offset by +4 (image frame 1 -> text frame 5)
    
    Args:
        image_rel_path: relative image path (starting from images_fps50_cropped_448x448/)
        text_root: text file root directory (e.g., .../H36M/text)
    
    Returns:
        Full path to the text file, or None if it cannot be mapped
    """
    try:
        # Parse image path: S1/S1_Directions_1.54138969/S1_Directions_1.54138969_000001.jpg
        parts = image_rel_path.split('/')
        subject = parts[0]  # "S1"
        img_dir_name = parts[1]  # "S1_Directions_1.54138969"
        img_file_name = parts[2]  # "S1_Directions_1.54138969_000001.jpg"
        
        # Extract frame number from filename
        frame_str = img_file_name.rsplit('_', 1)[-1].replace('.jpg', '')
        frame_num = int(frame_str)
        text_frame_num = frame_num + 4
        text_frame_str = f"{text_frame_num:06d}"
        
        # Remove trial number from directory name
        # Pattern: S1_Action_Trial.CameraID → S1_Action.CameraID
        # e.g., S1_Directions_1.54138969 → S1_Directions.54138969
        # Split on last '.' to separate camera ID
        dot_pos = img_dir_name.rfind('.')
        if dot_pos == -1:
            return None
        base_part = img_dir_name[:dot_pos]  # "S1_Directions_1"
        camera_id = img_dir_name[dot_pos + 1:]  # "54138969"
        
        # Remove trial number: the last _N where N is a digit
        # e.g., "S1_Directions_1" → "S1_Directions"
        # e.g., "S1_SittingDown_1" → "S1_SittingDown"
        match = re.match(r'^(.+)_(\d+)$', base_part)
        if match:
            text_base = match.group(1)  # "S1_Directions"
        else:
            text_base = base_part
        
        text_dir_name = f"{text_base}.{camera_id}"
        text_file_name = f"{text_base}.{camera_id}_{text_frame_str}.txt"
        
        text_path = os.path.join(text_root, subject, text_dir_name, text_file_name)
        return text_path
        
    except Exception:
        return None


def load_text_description(text_path: str) -> Optional[str]:
    """
    Load the text description for a single frame.
    
    An H36M text file may contain multiple description lines (multiple GPT-generated versions); pick one at random.
    """
    if text_path is None or not os.path.exists(text_path):
        return None
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if not content:
            return None
        # The file may contain multiple lines, each being a different description
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        if lines:
            return random.choice(lines)
        return None
    except Exception:
        return None


# ==================== Image Preprocessing ====================

def get_image_transform(resolution: int = 432):
    """
    Get the image preprocessing transform.
    Resize H36M 448x448 images to 432x432 (matching the 27x27 VAE latent).
    """
    return transforms.Compose([
        transforms.Resize((resolution, resolution), 
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


# ==================== Dataset ====================

class H36MDataset(Dataset):
    """
    H36M Dataset for Stage 3 tasks: img2pose and img2text.
    
    Each sample is a single-frame (image, motion, text) triplet.
    - img2pose: given an image, generate the corresponding single-frame motion (via flow matching diffusion)
    - img2text: given an image, generate a text description of the pose (via NTP)
    """
    
    def __init__(
        self,
        data_root: str,
        image_root: str,
        text_tokenizer: Any,
        showo_token_ids: Dict[str, int],
        task_type: str = 'img2pose',  # 'img2pose' or 'img2text'
        max_seq_len: int = 512,
        num_image_tokens: int = 169,  # PatchEmbed(patch_size=2) on 27x27 latent: 13*13=169
        image_resolution: int = 432,
        instruction_template_path: str = None,
        split: str = 'train',
        tiny: bool = False,
        debug: bool = False,
        add_time_embeds: bool = True,  # Whether to add a time embedding slot for the motion
        **kwargs,
    ):
        """
        Args:
            data_root: H36M dataset root directory
            image_root: H36M raw image root directory
            text_tokenizer: text tokenizer
            showo_token_ids: dict of special token IDs
            task_type: 'img2pose' (image -> motion) or 'img2text' (image -> description)
            max_seq_len: maximum sequence length
            num_image_tokens: number of image tokens (PatchEmbed output)
            image_resolution: target image resolution (images are resized to this size)
            instruction_template_path: path to the instruction templates
            split: dataset split
            tiny: tiny-subset mode
            debug: debug mode
            add_time_embeds: whether to add a time embedding slot for the motion tokens
        """
        super().__init__()
        
        self.data_root = data_root
        self.image_root = image_root
        self.text_tokenizer = text_tokenizer
        self.task_type = task_type
        self.data_type = task_type  # Task identifier used in the training loop
        self.max_seq_len = max_seq_len
        self.num_image_tokens = num_image_tokens
        self.image_resolution = image_resolution
        self.add_time_embeds = add_time_embeds
        
        # Token IDs
        self.pad_id = text_tokenizer.pad_token_id
        self.bos_id = showo_token_ids['bos_id']
        self.eos_id = showo_token_ids['eos_id']
        self.boi_id = showo_token_ids['boi_id']
        self.eoi_id = showo_token_ids['eoi_id']
        self.img_pad_id = showo_token_ids['img_pad_id']
        self.bom_id = showo_token_ids.get('bom_id', showo_token_ids['boi_id'])
        self.eom_id = showo_token_ids.get('eom_id', showo_token_ids['eoi_id'])
        self.mot_pad_id = showo_token_ids.get('mot_pad_id', showo_token_ids['img_pad_id'])
        
        # Motion tokens: a single-frame motion still maps to 1 latent at downsample_rate=4, optionally +1 for the time embed
        self.num_motion_tokens = 2 if add_time_embeds else 1  # time_embed + 1 latent token
        
        # Maximum text length
        if task_type == 'img2pose':
            # img2pose: [BOS] instr [BOI][IMG*N][EOI] [BOM][MOT*M][EOM][EOS]
            self.max_text_len = max_seq_len - num_image_tokens - self.num_motion_tokens - 6
        else:
            # img2text: [BOS][BOI][IMG*N][EOI] instr caption [EOS]
            self.max_text_len = max_seq_len - num_image_tokens - 4
        
        # Image transform
        self.image_transform = get_image_transform(image_resolution)
        
        # Load instruction templates
        self.instructions = []
        if instruction_template_path and os.path.exists(instruction_template_path):
            templates = load_img_instruction_templates(instruction_template_path)
            if task_type == 'img2pose':
                self.instructions = get_img2pose_instructions(templates)
            else:
                self.instructions = get_img2text_instructions(templates)
            print(f"Loaded {len(self.instructions)} {task_type} instructions")
        
        if not self.instructions:
            if task_type == 'img2pose':
                self.instructions = ["Generate pose of the image: <Image_Placeholder>."]
            else:
                self.instructions = ["Generate description of the pose in the image: <Image_Placeholder>."]
        
        # Load mean/std for motion normalization
        self.mean = np.load(os.path.join(data_root, 'Mean.npy'))
        self.std = np.load(os.path.join(data_root, 'Std.npy'))
        # Some dims of the H36M Std are 0 (constant dims); handle them safely to avoid NaN from division by zero
        self.std_safe = self.std.copy()
        self.std_safe[self.std_safe == 0] = 1.0
        
        # Load data
        self._load_data(split, tiny, debug)
    
    def _load_data(self, split: str, tiny: bool, debug: bool):
        """Load the H36M data."""
        # Determine the data file name
        split_file = os.path.join(self.data_root, f'{split}.txt')
        with open(split_file, 'r') as f:
            data_name = f.readline().strip()  # e.g., "h36m_train_269"
        
        # Load the motion data (N_frames, 269)
        motion_path = os.path.join(self.data_root, 'new_joint_vecs', f'{data_name}.npy')
        self.all_motions = np.load(motion_path)
        
        # Load the image paths (N_frames,)
        image_paths_file = os.path.join(self.data_root, 'image_paths', f'{data_name}.npy')
        self.all_image_paths = np.load(image_paths_file, allow_pickle=True)
        
        assert len(self.all_motions) == len(self.all_image_paths), \
            f"Motion frames ({len(self.all_motions)}) != Image paths ({len(self.all_image_paths)})"
        
        # Text directory
        self.text_root = os.path.join(self.data_root, 'text')
        
        # Build valid sample indices (filter out frames whose image or text cannot be found)
        total_frames = len(self.all_motions)
        if tiny or debug:
            total_frames = min(total_frames, 2000)
        
        self.valid_indices = []
        skipped_no_image = 0
        skipped_no_text = 0
        
        for i in range(total_frames):
            stored_path = str(self.all_image_paths[i])
            rel_path = self._get_image_rel_path(stored_path)
            full_path = os.path.join(self.image_root, rel_path)
            
            # Check that the image exists
            if not os.path.exists(full_path):
                skipped_no_image += 1
                continue
            
            # img2text mode must filter out frames without a text description
            if self.task_type == 'img2text':
                text_path = image_path_to_text_path(rel_path, self.text_root)
                if text_path is None or not os.path.exists(text_path):
                    skipped_no_text += 1
                    continue
            
            self.valid_indices.append(i)
        
        # Print dataset info
        print(f"\n{'='*20} H36M Dataset Info {'='*20}")
        print(f"Task:      {self.task_type}")
        print(f"Split:     {split}")
        print(f"Total frames: {len(self.all_motions)}")
        print(f"Valid samples: {len(self.valid_indices)}")
        if skipped_no_image > 0:
            print(f"Skipped (no image): {skipped_no_image}")
        if skipped_no_text > 0:
            print(f"Skipped (no text):  {skipped_no_text}")
        print(f"Motion shape: {self.all_motions.shape}")
        print(f"Num image tokens: {self.num_image_tokens}")
        print(f"Num motion tokens: {self.num_motion_tokens}")
        print(f"Max text len: {self.max_text_len}")
        print(f"Instructions: {len(self.instructions)}")
        print(f"{'='*60}\n")
    
    def _get_image_rel_path(self, stored_path: str) -> str:
        """
        Extract the relative path from the stored full path.
        Stored format: <original_absolute_prefix>/.../images_fps50_cropped_448x448/S1/...
        Extract the S1/... part
        """
        marker = 'images_fps50_cropped_448x448/'
        idx = stored_path.find(marker)
        if idx >= 0:
            return stored_path[idx + len(marker):]
        return stored_path
    
    def _load_image(self, frame_idx: int) -> Optional[torch.Tensor]:
        """Load and preprocess a single-frame image."""
        stored_path = str(self.all_image_paths[frame_idx])
        rel_path = self._get_image_rel_path(stored_path)
        full_path = os.path.join(self.image_root, rel_path)
        
        if not os.path.exists(full_path):
            return None
        
        try:
            img = Image.open(full_path).convert('RGB')
            img_tensor = self.image_transform(img)
            return img_tensor  # [3, resolution, resolution]
        except Exception:
            return None
    
    def _get_text(self, frame_idx: int) -> Optional[str]:
        """Get the text description for a single frame."""
        stored_path = str(self.all_image_paths[frame_idx])
        rel_path = self._get_image_rel_path(stored_path)
        text_path = image_path_to_text_path(rel_path, self.text_root)
        return load_text_description(text_path)
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        try:
            frame_idx = self.valid_indices[idx]
            
            # Load the image
            image = self._load_image(frame_idx)
            if image is None:
                return self.__getitem__((idx + 1) % len(self))
            
            # Load the single-frame motion
            motion_frame = self.all_motions[frame_idx].copy()  # (269,)
            # Z Normalization (use the safe std to avoid NaN from division by zero)
            motion_frame = (motion_frame - self.mean) / self.std_safe
            # Expand to (1, 269) to match the VAE input format
            motion = torch.from_numpy(motion_frame).float().unsqueeze(0)  # [1, 269]
            
            # Randomly pick an instruction template
            instruction = random.choice(self.instructions)
            # Remove the <Image_Placeholder> placeholder (the image is embedded via special tokens)
            instruction_text = instruction.replace('<Image_Placeholder>', '').strip()
            # Clean up double spaces
            while '  ' in instruction_text:
                instruction_text = instruction_text.replace('  ', ' ')
            
            if self.task_type == 'img2pose':
                # img2pose: input image, output motion
                # Tokenize instruction
                instr_tokens = self.text_tokenizer(
                    instruction_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=min(self.max_text_len, 100)  # Instructions are not very long
                ).input_ids
                
                # Format sequence
                (text_tokens, text_labels, 
                 image_modality_positions, motion_modality_positions,
                 text_mask, image_mask, motion_mask) = format_sequence_img2pose(
                    instr_tokens,
                    self.bos_id, self.eos_id,
                    self.boi_id, self.eoi_id,
                    self.bom_id, self.eom_id,
                    self.pad_id, self.img_pad_id, self.mot_pad_id,
                    self.num_image_tokens, self.num_motion_tokens,
                    self.max_seq_len
                )
                
                sample = {
                    'text_tokens': text_tokens,
                    'text_labels': text_labels,
                    'image': image,  # [3, H, W] preprocessed image
                    'motions': motion,  # [1, 269] single frame motion
                    'motion_lengths': torch.tensor(1),  # single frame
                    'image_modality_positions': image_modality_positions,
                    'motion_modality_positions': motion_modality_positions,
                    'text_masks': text_mask,
                    'image_masks': image_mask,
                    'motion_masks': motion_mask,
                    'texts': instruction_text,
                    'data_type': 'img2pose',
                }
                
            else:  # img2text
                # Load the text description
                text_desc = self._get_text(frame_idx)
                if text_desc is None:
                    # No text description, try the next sample
                    return self.__getitem__((idx + 1) % len(self))
                
                # Tokenize instruction and caption
                instr_tokens = self.text_tokenizer(
                    instruction_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=min(self.max_text_len // 3, 80)
                ).input_ids
                
                caption_tokens = self.text_tokenizer(
                    text_desc,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_text_len - len(instr_tokens)
                ).input_ids
                
                # Format sequence
                (text_tokens, text_labels,
                 image_modality_positions,
                 text_mask, image_mask) = format_sequence_img2text(
                    instr_tokens, caption_tokens,
                    self.bos_id, self.eos_id,
                    self.boi_id, self.eoi_id,
                    self.pad_id, self.img_pad_id,
                    self.num_image_tokens,
                    self.max_seq_len
                )
                
                sample = {
                    'text_tokens': text_tokens,
                    'text_labels': text_labels,
                    'image': image,
                    'image_modality_positions': image_modality_positions,
                    'text_masks': text_mask,
                    'image_masks': image_mask,
                    'texts': text_desc,
                    'data_type': 'img2text',
                }
            
            return sample
            
        except Exception as e:
            print(f"Error in H36MDataset __getitem__ {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """Custom collate function."""
        batched = collections.defaultdict(list)
        
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        
        result = {}
        for k, v in batched.items():
            if k in ('texts', 'data_type'):
                result[k] = v
            elif k == 'motions':
                # motion: [1, 269] per sample, stack to [bs, 1, 269]
                result[k] = torch.stack(v, dim=0)
            else:
                result[k] = torch.stack(v, dim=0)
        
        return result


def create_h36m_dataloader(
    data_root: str,
    image_root: str,
    batch_size: int,
    text_tokenizer: Any,
    showo_token_ids: Dict[str, int],
    task_type: str = 'img2pose',
    accelerator: Optional[Any] = None,
    max_seq_len: int = 512,
    num_image_tokens: int = 169,
    image_resolution: int = 432,
    instruction_template_path: str = None,
    num_workers: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
    split: str = 'train',
    tiny: bool = False,
    debug: bool = False,
    add_time_embeds: bool = True,
    **kwargs,
) -> DataLoader:
    """Create the H36M dataloader."""
    dataset = H36MDataset(
        data_root=data_root,
        image_root=image_root,
        text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids,
        task_type=task_type,
        max_seq_len=max_seq_len,
        num_image_tokens=num_image_tokens,
        image_resolution=image_resolution,
        instruction_template_path=instruction_template_path,
        split=split,
        tiny=tiny,
        debug=debug,
        add_time_embeds=add_time_embeds,
    )
    
    # Distributed training support
    if accelerator is not None and accelerator.num_processes > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=shuffle,
            drop_last=drop_last,
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
        pin_memory=True,
    )
    
    return dataloader
