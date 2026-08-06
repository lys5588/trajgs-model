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
Motion Evaluation Dataset for the UniMotion T2M task.
Exactly matches the MotionGPT3 evaluation dataset format for a fair comparison.

Main features:
- Load the HumanML3D test set (including sub-clips)
- Process text word embeddings and POS one-hot encoding (using the VIP POS system)
- Provide the data format required for evaluation
- Random crop (consistent with MotionGPT3)
"""

import os
import random
import codecs as cs
import pickle
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ==================== POS Enumerator (identical to MotionGPT3) ====================
# This is a critical detail: it must stay consistent with MotionGPT3's POS system

POS_enumerator = {
    'VERB': 0,
    'NOUN': 1,
    'DET': 2,
    'ADP': 3,
    'NUM': 4,
    'AUX': 5,
    'PRON': 6,
    'ADJ': 7,
    'ADV': 8,
    'Loc_VIP': 9,
    'Body_VIP': 10,
    'Obj_VIP': 11,
    'Act_VIP': 12,
    'Desc_VIP': 13,
    'OTHER': 14,
}

# VIP vocabulary (for POS tagging of special keywords)
Loc_list = ('left', 'right', 'clockwise', 'counterclockwise', 'anticlockwise', 'forward', 'back', 'backward',
            'up', 'down', 'straight', 'curve')

Body_list = ('arm', 'chin', 'foot', 'feet', 'face', 'hand', 'mouth', 'leg', 'waist', 'eye', 'knee', 'shoulder', 'thigh')

Obj_List = ('stair', 'dumbbell', 'chair', 'window', 'floor', 'car', 'ball', 'handrail', 'baseball', 'basketball')

Act_list = ('walk', 'run', 'swing', 'pick', 'bring', 'kick', 'put', 'squat', 'throw', 'hop', 'dance', 'jump', 'turn',
            'stumble', 'dance', 'stop', 'sit', 'lift', 'lower', 'raise', 'wash', 'stand', 'kneel', 'stroll',
            'rub', 'bend', 'balance', 'flap', 'jog', 'shuffle', 'lean', 'rotate', 'spin', 'spread', 'climb')

Desc_list = ('slowly', 'carefully', 'fast', 'careful', 'slow', 'quickly', 'happy', 'angry', 'sad', 'happily', 'angrily', 'sadly')

VIP_dict = {
    'Loc_VIP': Loc_list,
    'Body_VIP': Body_list,
    'Obj_VIP': Obj_List,
    'Act_VIP': Act_list,
    'Desc_VIP': Desc_list,
}


class WordVectorizer:
    """
    Word vectorizer for converting tokens to word embeddings and POS one-hot.
    Identical to MotionGPT3's WordVectorizer.
    """
    def __init__(self, glove_dir: str, dataset_name: str = 't2m'):
        """
        Args:
            glove_dir: path to the GloVe word vector directory
            dataset_name: dataset name ('t2m' or 'kit')
        """
        self.glove_dir = glove_dir
        self.dataset_name = dataset_name
        
        # Load vocabulary and embeddings
        self._load_vocab()
    
    def _load_vocab(self):
        """Load the vocabulary and word vectors"""
        # Prefer the pkl format (HumanML3D format)
        data_path = os.path.join(self.glove_dir, 'our_vab_data.npy')
        idx_path = os.path.join(self.glove_dir, 'our_vab_idx.pkl')
        words_path = os.path.join(self.glove_dir, 'our_vab_words.pkl')
        
        if os.path.exists(data_path) and os.path.exists(idx_path):
            # Load word vectors in HumanML3D format
            vectors = np.load(data_path)
            with open(idx_path, 'rb') as f:
                word2idx = pickle.load(f)
            with open(words_path, 'rb') as f:
                words = pickle.load(f)
            
            # Build the word2vec dict (consistent with MotionGPT3)
            self.word2vec = {w: vectors[word2idx[w]] for w in words}
            self.word_dim = vectors.shape[1]
            print(f"Loaded word embeddings from {data_path}: {len(self.word2vec)} words, dim={self.word_dim}")
        else:
            raise FileNotFoundError(
                f"Word vectorizer files not found in {self.glove_dir}. "
                "Please ensure our_vab_data.npy, our_vab_idx.pkl, our_vab_words.pkl exist."
            )
    
    def _get_pos_ohot(self, pos: str) -> np.ndarray:
        """Get the POS one-hot vector"""
        pos_vec = np.zeros(len(POS_enumerator), dtype=np.float32)
        if pos in POS_enumerator:
            pos_vec[POS_enumerator[pos]] = 1
        else:
            pos_vec[POS_enumerator['OTHER']] = 1
        return pos_vec
    
    def __getitem__(self, token: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the word embedding and POS one-hot for a token.
        Identical to MotionGPT3, including the special handling of VIP words.
        
        Args:
            token: a token in "word/POS" format (e.g. "walk/VERB")
        Returns:
            (word_embedding, pos_one_hot)
        """
        word, pos = token.split('/')
        
        # Word embedding
        if word in self.word2vec:
            word_vec = self.word2vec[word]
            # Check whether the word is a VIP word
            vip_pos = None
            for key, values in VIP_dict.items():
                if word in values:
                    vip_pos = key
                    break
            if vip_pos is not None:
                pos_vec = self._get_pos_ohot(vip_pos)
            else:
                pos_vec = self._get_pos_ohot(pos)
        else:
            word_vec = self.word2vec['unk']
            pos_vec = self._get_pos_ohot('OTHER')
        
        return word_vec, pos_vec
    
    @property
    def word_size(self) -> int:
        return self.word_dim
    
    @property
    def pos_size(self) -> int:
        return len(POS_enumerator)


class MotionEvalDataset(Dataset):
    """
    Motion Evaluation Dataset for T2M task.
    
    Identical to the MotionGPT3 evaluation dataset:
    - Loads the HumanML3D test set (including sub-clips!)
    - Provides word embeddings and POS one-hot (using the VIP POS system)
    - Also performs random crop during evaluation (consistent with MotionGPT3)
    
    Notes:
    - GT motions are normalized with the eval mean/std (as expected by the T2M evaluator)
    - max_text_len defaults to 20, consistent with HumanML3D
    - unit_length=4, used to align the motion length
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'test',
        mean_path: str = None,
        std_path: str = None,
        mean_eval_path: str = None,
        std_eval_path: str = None,
        w_vectorizer: WordVectorizer = None,
        max_motion_length: int = 200,
        min_motion_length: int = 20,
        unit_length: int = 4,
        fps: int = 20,
        max_text_len: int = 20,  # HumanML3D default
        tiny: bool = False,
        debug: bool = False,
    ):
        """
        Args:
            data_root: HumanML3D dataset root directory
            split: data split ('test', 'val', 'train')
            mean_path: path to Mean.npy (for training)
            std_path: path to Std.npy (for training)
            mean_eval_path: path to the evaluation Mean.npy (for the T2M evaluator)
            std_eval_path: path to the evaluation Std.npy (for the T2M evaluator)
            w_vectorizer: WordVectorizer instance
            max_motion_length: maximum motion length
            min_motion_length: minimum motion length
            unit_length: length alignment unit
            fps: frame rate
            max_text_len: maximum number of text tokens
            tiny: whether to use a tiny subset
            debug: whether to run in debug mode
        """
        self.data_root = data_root
        self.split = split
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length
        self.fps = fps
        self.max_text_len = max_text_len
        self.w_vectorizer = w_vectorizer
        
        # Load mean and std (for training; used to de-normalize generated motions)
        if mean_path and os.path.exists(mean_path):
            self.mean = np.load(mean_path)
        else:
            self.mean = np.load(os.path.join(data_root, 'Mean.npy'))
        
        if std_path and os.path.exists(std_path):
            self.std = np.load(std_path)
        else:
            self.std = np.load(os.path.join(data_root, 'Std.npy'))
        
        # Load mean_eval and std_eval (for evaluation; the normalization expected by the T2M evaluator)
        if mean_eval_path and os.path.exists(mean_eval_path):
            self.mean_eval = np.load(mean_eval_path)
            print(f"  Loaded eval mean from {mean_eval_path}")
        else:
            self.mean_eval = self.mean
        
        if std_eval_path and os.path.exists(std_eval_path):
            self.std_eval = np.load(std_eval_path)
            print(f"  Loaded eval std from {std_eval_path}")
        else:
            self.std_eval = self.std
        
        # Load data (including sub-clips, consistent with MotionGPT3)
        self._load_data(tiny, debug)
    
    def _load_data(self, tiny: bool, debug: bool):
        """
        Load motion-text pairs.
        Identical to MotionGPT3: load all qualifying motions, including time-tagged sub-clips.
        """
        split_file = os.path.join(self.data_root, f'{self.split}.txt')
        motion_dir = os.path.join(self.data_root, 'new_joint_vecs')
        text_dir = os.path.join(self.data_root, 'texts')
        
        # Load IDs
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f:
                id_list.append(line.strip())
        
        if tiny or debug:
            id_list = id_list[:100]
        
        self.data_dict = {}
        self.name_list = []
        
        for name in id_list:
            try:
                motion_path = os.path.join(motion_dir, f'{name}.npy')
                text_path = os.path.join(text_dir, f'{name}.txt')
                
                if not os.path.exists(motion_path) or not os.path.exists(text_path):
                    continue
                
                # Load motion
                motion = np.load(motion_path)
                
                # Load text and process (consistent with MotionGPT3)
                text_data_full = []  # Full motion descriptions
                flag = False  # Flag for full motion descriptions
                
                with cs.open(text_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        text_dict = {}
                        line_split = line.strip().split('#')
                        if len(line_split) < 4:
                            continue
                        
                        caption = line_split[0]
                        t_tokens = line_split[1].split(' ') if line_split[1] else []
                        f_tag = float(line_split[2]) if line_split[2] else 0.0
                        to_tag = float(line_split[3]) if line_split[3] else 0.0
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        
                        text_dict['caption'] = caption
                        text_dict['tokens'] = t_tokens
                        
                        if f_tag == 0.0 and to_tag == 0.0:
                            # Full motion description
                            flag = True
                            text_data_full.append(text_dict)
                        else:
                            # Sub-clip with time tags
                            motion_new = motion[int(f_tag * self.fps):int(to_tag * self.fps)]
                            if self.min_motion_length <= len(motion_new) < self.max_motion_length:
                                new_name = '%s_%f_%f' % (name, f_tag, to_tag)
                                self.data_dict[new_name] = {
                                    'motion': motion_new,
                                    'length': len(motion_new),
                                    'text': [text_dict],
                                }
                                self.name_list.append(new_name)
                
                # Add full motion if it has full description and meets length requirements
                if flag and self.min_motion_length <= len(motion) < self.max_motion_length:
                    self.data_dict[name] = {
                        'motion': motion,
                        'length': len(motion),
                        'text': text_data_full,
                    }
                    self.name_list.append(name)
                    
            except Exception as e:
                print(f"Error loading {name}: {e}")
                continue
        
        # Sort by length (consistent with MotionGPT3)
        length_list = [self.data_dict[n]['length'] for n in self.name_list]
        sorted_pairs = sorted(zip(self.name_list, length_list), key=lambda x: x[1])
        self.name_list = [p[0] for p in sorted_pairs]
        self.length_arr = np.array([p[1] for p in sorted_pairs])
        
        # Reset the pointer (used to skip samples that are too short)
        self.pointer = np.searchsorted(self.length_arr, self.min_motion_length)
        
        print(f"Loaded {len(self.name_list) - self.pointer} samples for evaluation [{self.split}]")
        print(f"  (Total: {len(self.name_list)}, Pointer at: {self.pointer})")
    
    def __len__(self) -> int:
        return len(self.name_list) - self.pointer
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample for evaluation.
        Identical to MotionGPT3's __getitem__, including the random crop.
        
        Returns:
            dict containing:
                - motion: [T, 263] normalized motion
                - m_length: motion length
                - caption: text caption
                - tokens: token list
                - word_embs: [max_text_len+2, word_dim] word embeddings
                - pos_ohot: [max_text_len+2, pos_size] POS one-hot
                - sent_len: sentence length
                - all_captions: list of all captions
                - fname: sample name
        """
        # Apply the pointer offset (consistent with MotionGPT3)
        idx = self.pointer + idx
        fname = self.name_list[idx]
        data = self.data_dict[fname]
        
        motion = data['motion'].copy()
        m_length = data['length']
        text_list = data['text']
        
        # Get all captions (up to 3)
        all_captions = [t['caption'] for t in text_list]
        if len(all_captions) > 3:
            all_captions = all_captions[:3]
        elif len(all_captions) == 2:
            all_captions = all_captions + all_captions[0:1]
        elif len(all_captions) == 1:
            all_captions = all_captions * 3
        
        # Randomly select one for evaluation
        text_data = random.choice(text_list)
        caption = text_data['caption']
        tokens = text_data['tokens']
        
        # Process text tokens (max_text_len = 20, consistent with MotionGPT3)
        word_embs, pos_ohot, sent_len = self._process_tokens(tokens)
        
        # ========== Random crop (identical to MotionGPT3) ==========
        m_length = motion.shape[0]
        coin = np.random.choice([False, False, True])
        if coin:
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        else:
            m_length = (m_length // self.unit_length) * self.unit_length
        
        # Random start index
        idx_start = random.randint(0, len(motion) - m_length)
        motion = motion[idx_start:idx_start + m_length]
        
        # If the motion dim > eval mean dim (e.g. 269 > 263), truncate to the eval dim
        eval_dim = self.mean_eval.shape[0]
        if motion.shape[-1] > eval_dim:
            motion = motion[..., :eval_dim]
        
        # Z Normalization using eval mean/std (for T2M evaluator)
        motion = (motion - self.mean_eval) / self.std_eval
        
        return {
            'motion': motion.astype(np.float32),
            'm_length': m_length,
            'caption': caption,
            'tokens': tokens,
            'word_embs': word_embs,
            'pos_ohot': pos_ohot,
            'sent_len': sent_len,
            'all_captions': all_captions,
            'fname': fname,
        }
    
    def _process_tokens(self, tokens: List[str]) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Process tokens to word embeddings and POS one-hot.
        Identical to MotionGPT3.
        
        Args:
            tokens: list of "word/POS" tokens
        Returns:
            (word_embs, pos_ohot, sent_len)
        """
        if self.w_vectorizer is None:
            # Return dummy values if no vectorizer
            word_embs = np.zeros((self.max_text_len + 2, 300), dtype=np.float32)
            pos_ohot = np.zeros((self.max_text_len + 2, 15), dtype=np.float32)
            return word_embs, pos_ohot, min(len(tokens) + 2, self.max_text_len + 2)
        
        # Processing consistent with MotionGPT3
        max_text_len = 20  # Fixed value for HumanML3D
        if len(tokens) < max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        
        # Get embeddings
        word_embs = []
        pos_ohot = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            word_embs.append(word_emb[None, :])
            pos_ohot.append(pos_oh[None, :])
        
        word_embs = np.concatenate(word_embs, axis=0)
        pos_ohot = np.concatenate(pos_ohot, axis=0)
        
        return word_embs, pos_ohot, sent_len
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """
        Custom collate function for evaluation.
        """
        # Find max motion length
        max_m_length = max(item['m_length'] for item in batch)
        
        # Pad motions
        motions = []
        m_lengths = []
        word_embs = []
        pos_ohot = []
        sent_lens = []
        captions = []
        all_captions = []
        fnames = []
        
        for item in batch:
            motion = item['motion']
            pad_len = max_m_length - motion.shape[0]
            if pad_len > 0:
                motion = np.concatenate([motion, np.zeros((pad_len, motion.shape[1]), dtype=np.float32)], axis=0)
            
            motions.append(motion)
            m_lengths.append(item['m_length'])
            word_embs.append(item['word_embs'])
            pos_ohot.append(item['pos_ohot'])
            sent_lens.append(item['sent_len'])
            captions.append(item['caption'])
            all_captions.append(item['all_captions'])
            fnames.append(item['fname'])
        
        return {
            'motions': torch.from_numpy(np.stack(motions)),
            'm_lengths': m_lengths,
            'word_embs': torch.from_numpy(np.stack(word_embs)),
            'pos_ohot': torch.from_numpy(np.stack(pos_ohot)),
            'sent_lens': torch.tensor(sent_lens),
            'captions': captions,
            'all_captions': all_captions,
            'fnames': fnames,
        }


def create_eval_dataloader(
    data_root: str,
    glove_dir: str = None,
    batch_size: int = 32,
    split: str = 'test',
    max_motion_length: int = 260,
    min_motion_length: int = 20,
    num_workers: int = 4,
    tiny: bool = False,
    debug: bool = False,
) -> Tuple[DataLoader, MotionEvalDataset]:
    """
    Create evaluation dataloader.
    
    Args:
        data_root: HumanML3D dataset root
        glove_dir: GloVe directory path
        batch_size: Batch size
        split: Data split
        max_motion_length: Maximum motion length
        min_motion_length: Minimum motion length
        num_workers: Number of workers
        tiny: Use tiny dataset
        debug: Debug mode
        
    Returns:
        (dataloader, dataset)
    """
    # Create word vectorizer if glove_dir provided
    w_vectorizer = None
    if glove_dir and os.path.exists(glove_dir):
        w_vectorizer = WordVectorizer(glove_dir)
    
    # Create dataset
    dataset = MotionEvalDataset(
        data_root=data_root,
        split=split,
        w_vectorizer=w_vectorizer,
        max_motion_length=max_motion_length,
        min_motion_length=min_motion_length,
        tiny=tiny,
        debug=debug,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=False,
        pin_memory=True,
    )
    
    return dataloader, dataset


# ==================== Test Script ====================

if __name__ == '__main__':
    print("Testing MotionEvalDataset (MotionGPT3 compatible)...")
    
    glove_dir = "deps/glove/glove"
    
    # Create word vectorizer
    w_vectorizer = WordVectorizer(glove_dir) if os.path.exists(glove_dir) else None
    
    dataset = MotionEvalDataset(
        data_root="data/UNI_MOTION_269/HumanML3D",
        split='test',
        w_vectorizer=w_vectorizer,
        tiny=False,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test single sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Motion shape: {sample['motion'].shape}")
    print(f"Motion length: {sample['m_length']}")
    print(f"Caption: {sample['caption'][:100]}...")
    print(f"Sent len: {sample['sent_len']}")
    print(f"Word embs shape: {sample['word_embs'].shape}")
    print(f"POS ohot shape: {sample['pos_ohot'].shape}")
    print(f"Fname: {sample['fname']}")
    
    # Test dataloader
    dataloader, _ = create_eval_dataloader(
        data_root="data/UNI_MOTION_269/HumanML3D",
        glove_dir=glove_dir,
        batch_size=4,
        split='test',
        num_workers=0,
        tiny=False,
    )
    
    print(f"\nDataloader batches: {len(dataloader)}")
    
    for batch in dataloader:
        print(f"\nBatch keys: {batch.keys()}")
        print(f"Motions shape: {batch['motions'].shape}")
        print(f"M lengths: {batch['m_lengths']}")
        print(f"Word embs shape: {batch['word_embs'].shape}")
        print(f"POS ohot shape: {batch['pos_ohot'].shape}")
        break
    
    print("\nSelf-test finished.")
