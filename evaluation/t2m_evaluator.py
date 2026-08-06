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
T2M Evaluation Module for UniMotion.
Consistent with the MotionGPT3 evaluation protocol; supports the following metrics:
- Matching Score (MM dist)
- R-Precision Top-k (k=1,2,3)
- FID (Frechet Inception Distance)
- Diversity
- gt_Diversity

Evaluator models come from HumanML3D/T2M-GPT:
- TextEncoderBiGRUCo: text encoder
- MovementConvEncoder: movement encoder (Conv)
- MotionEncoderBiGRUCo: motion encoder (BiGRU, not Conv!)
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from scipy import linalg


# ==================== Evaluation Encoder Models ====================
# These models come from HumanML3D / T2M-GPT and evaluate T2M generation quality


class MovementConvEncoder(nn.Module):
    """
    Movement encoder (first stage).
    Encode motion features into movement representations.
    Downsample by 4x with Conv1d.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [bs, seq_len, input_size] motion features
        Returns:
            [bs, seq_len//4, output_size] movement features
        """
        inputs = inputs.permute(0, 2, 1)  # [bs, input_size, seq_len]
        outputs = self.main(inputs).permute(0, 2, 1)  # [bs, seq_len//4, output_size]
        return self.out_net(outputs)


class MotionEncoderBiGRUCo(nn.Module):
    """
    Motion encoder (second stage) - uses BiGRU!
    Further encode movement representations into a motion embedding.
    
    Key point: in the original MotionGPT3 implementation, MotionEncoder does NOT use pack_padded_sequence!
    Feed the whole sequence into the GRU directly and use the final hidden state.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, device: str = None):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size, hidden_size,
            batch_first=True, bidirectional=True
        )
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )
        
        # Learnable initial hidden state (loaded from checkpoint)
        self.hidden = nn.Parameter(
            torch.randn((2, 1, hidden_size), requires_grad=True)
        )

    def forward(self, inputs: torch.Tensor, m_lens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [bs, seq_len, input_size] movement features
            m_lens: [bs] sequence lengths (unused; kept for interface compatibility)
        Returns:
            [bs, output_size] motion embedding
        """
        num_samples = inputs.shape[0]
        
        # Input embedding
        input_embs = self.input_emb(inputs)
        
        # Initialize hidden state - use repeat instead of expand
        hidden = self.hidden.repeat(1, num_samples, 1)
        
        # Key: do NOT use pack_padded_sequence! Feed the whole sequence directly
        gru_seq, gru_last = self.gru(input_embs, hidden)
        
        # Concatenate the final hidden states of the bidirectional GRU
        # gru_last: [2, bs, hidden_size]
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)  # [bs, hidden_size*2]
        
        return self.output_net(gru_last)


class TextEncoderBiGRUCo(nn.Module):
    """
    Text encoder using Bi-directional GRU.
    Encode word embeddings into a text embedding.
    
    Consistent with the original MotionGPT3 implementation: uses pack_padded_sequence for variable-length text.
    """
    def __init__(
        self, 
        word_size: int, 
        pos_size: int, 
        hidden_size: int, 
        output_size: int,
        device: str = None
    ):
        super().__init__()
        self.device = device
        
        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size, hidden_size, 
            batch_first=True, bidirectional=True
        )
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )
        
        self.hidden_size = hidden_size
        
        # Learnable initial hidden state (loaded from checkpoint)
        self.hidden = nn.Parameter(
            torch.randn((2, 1, hidden_size), requires_grad=True)
        )

    def forward(
        self, 
        word_embs: torch.Tensor, 
        pos_ohot: torch.Tensor, 
        cap_lens: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            word_embs: [bs, max_text_len, word_size] word embeddings
            pos_ohot: [bs, max_text_len, pos_size] POS one-hot
            cap_lens: [bs] text lengths
        Returns:
            [bs, output_size] text embedding
        """
        num_samples = word_embs.shape[0]
        
        pos_embs = self.pos_emb(pos_ohot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        
        # Initialize hidden state - use repeat
        hidden = self.hidden.repeat(1, num_samples, 1)
        
        # Use pack_padded_sequence for variable-length text
        cap_lens_list = cap_lens.data.tolist()
        emb = nn.utils.rnn.pack_padded_sequence(
            input_embs, cap_lens_list, batch_first=True, enforce_sorted=False
        )
        
        gru_seq, gru_last = self.gru(emb, hidden)
        
        # Concatenate the final hidden states of the bidirectional GRU
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)
        
        return self.output_net(gru_last)


# ==================== POS Enumerator (identical to MotionGPT3) ====================
# This is the key difference! Must stay consistent with MotionGPT3's POS system

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


# ==================== Word Vectorizer ====================

class WordVectorizer:
    """
    Word vectorizer for converting tokens to word embeddings and POS one-hot.
    Identical to MotionGPT3's WordVectorizer! Uses the VIP POS system.
    """
    def __init__(self, glove_dir: str, dataset_name: str = 't2m'):
        """
        Args:
            glove_dir: path to the GloVe word-vector directory
            dataset_name: dataset name ('t2m' or 'kit')
        """
        self.glove_dir = glove_dir
        self.dataset_name = dataset_name
        
        # Load vocabulary and embeddings
        self._load_vocab()
        
    def _load_vocab(self):
        """Load the vocabulary and word vectors."""
        import pickle
        
        # Try the pkl format first (HumanML3D format)
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
        """Get the POS one-hot vector."""
        pos_vec = np.zeros(len(POS_enumerator), dtype=np.float32)
        if pos in POS_enumerator:
            pos_vec[POS_enumerator[pos]] = 1
        else:
            pos_vec[POS_enumerator['OTHER']] = 1
        return pos_vec
    
    def __getitem__(self, token: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the word embedding and POS one-hot for a token.
        Identical to MotionGPT3! Includes special handling for VIP words.
        
        Args:
            token: token in "word/POS" format (e.g. "walk/VERB")
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


# ==================== Metric Utilities ====================

def euclidean_distance_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the Euclidean distance matrix between two sets of vectors.
    
    Args:
        x: [N, D] first set of vectors
        y: [M, D] second set of vectors
    Returns:
        [N, M] distance matrix
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    
    return torch.pow(x - y, 2).sum(2).sqrt()


def calculate_top_k(mat: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Compute per-row whether the correct answer is within top-k (cumulative).
    
    Identical to the original MotionGPT3 implementation.
    
    Definition of R-Precision:
    - Top_1: fraction with the correct answer ranked first
    - Top_2: fraction with the correct answer in the top 2 (>= Top_1)
    - Top_3: fraction with the correct answer in the top 3 (>= Top_2)
    
    Args:
        mat: [N, N] sorted index matrix (each row holds indices sorted by distance)
        top_k: check top-k
    Returns:
        [N, top_k] indicator matrix of per-row hits at each top-k threshold
    """
    size = mat.shape[0]
    # gt_mat: [N, N] each row is [0,1,2,...,N-1], used to match the correct answers on the diagonal
    gt_mat = torch.unsqueeze(torch.arange(size), 1).to(mat.device).repeat_interleave(size, 1)
    # bool_mat[i,j] = True means the j-th prediction of sample i is correct (i.e. predicted index == i)
    bool_mat = mat == gt_mat
    
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        # Cumulative: whether any of the first i positions is correct
        correct_vec = correct_vec | bool_mat[:, i]
        top_k_list.append(correct_vec[:, None])
    
    top_k_mat = torch.cat(top_k_list, dim=1)
    return top_k_mat


def calculate_activation_statistics(activations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the mean and covariance matrix of the activations.
    
    Args:
        activations: [N, D] feature matrix
    Returns:
        (mu, sigma): mean and covariance
    """
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, 
                               mu2: np.ndarray, sigma2: np.ndarray,
                               eps: float = 1e-6) -> float:
    """
    Compute the Frechet distance between two Gaussian distributions.
    Identical to MotionGPT3's calculate_frechet_distance_np.
    
    Args:
        mu1, sigma1: mean and covariance of the first distribution
        mu2, sigma2: mean and covariance of the second distribution
        eps: epsilon for numerical stability
    Returns:
        FID value
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    
    # Product of covariances
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Handle numerical issues (consistent with MotionGPT3)
    if not np.isfinite(covmean).all():
        print(f"FID calculation produces singular product; adding {eps} to diagonal of cov estimates")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            print(f"Warning: Imaginary component {m} in FID calculation")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    # Compute FID
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    
    return float(fid)


def calculate_diversity(activations: np.ndarray, diversity_times: int = 300) -> float:
    """
    Compute the diversity of the features.
    
    Args:
        activations: [N, D] feature matrix
        diversity_times: number of samples
    Returns:
        Diversity value
    """
    if len(activations) < 2:
        return 0.0
    
    distances = []
    for _ in range(diversity_times):
        idx = np.random.choice(len(activations), 2, replace=False)
        dist = np.linalg.norm(activations[idx[0]] - activations[idx[1]])
        distances.append(dist)
    
    return float(np.mean(distances))


# ==================== T2M Evaluator Class ====================

class T2MEvaluator:
    """
    Evaluator for the T2M task.
    Consistent with MotionGPT3's TM2TMetrics.
    """
    
    def __init__(
        self,
        t2m_evaluator_path: str,
        glove_dir: str,
        dataname: str = 't2m',
        device: str = 'cuda',
        top_k: int = 3,
        R_size: int = 32,
        diversity_times: int = 300,
        unit_length: int = 4,
    ):
        """
        Args:
            t2m_evaluator_path: path to the T2M evaluator weights (contains finest.tar)
            glove_dir: GloVe word-vector directory
            dataname: dataset name ('t2m' or 'kit')
            device: compute device
            top_k: k value for R-Precision
            R_size: batch size for R-Precision
            diversity_times: number of samples for Diversity
            unit_length: motion length unit (4 by default for HumanML3D)
        """
        self.device = device
        self.dataname = dataname
        self.top_k = top_k
        self.R_size = R_size
        self.diversity_times = diversity_times
        self.unit_length = unit_length
        
        # Load word vectorizer
        self.w_vectorizer = WordVectorizer(glove_dir, dataname)
        
        # Load evaluation models
        self._load_evaluator(t2m_evaluator_path)
        
        # Storage for embeddings
        self.reset()
    
    def _load_evaluator(self, evaluator_path: str):
        """Load T2M evaluator models."""
        # Model parameters (HumanML3D default)
        if self.dataname == 'kit':
            input_size = 251
        else:
            input_size = 263
        
        # Subtract 4 for the root features (following HumanML3D convention)
        movement_input_size = input_size - 4
        
        # Initialize models - use the correct model types!
        self.t2m_moveencoder = MovementConvEncoder(
            input_size=movement_input_size,  # 259
            hidden_size=512,
            output_size=512
        )
        
        # Key fix: use GRU instead of Conv!
        self.t2m_motionencoder = MotionEncoderBiGRUCo(
            input_size=512,
            hidden_size=1024,
            output_size=512,
            device=self.device
        )
        
        self.t2m_textencoder = TextEncoderBiGRUCo(
            word_size=self.w_vectorizer.word_size,
            pos_size=self.w_vectorizer.pos_size,
            hidden_size=512,
            output_size=512,
            device=self.device
        )
        
        # Load pretrained weights
        checkpoint_path = os.path.join(evaluator_path, 'text_mot_match/model/finest.tar')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Load the text encoder (including hidden state)
            text_encoder_state = checkpoint['text_encoder']
            # hidden is a learnable parameter and needs special handling
            if 'hidden' in text_encoder_state:
                self.t2m_textencoder.hidden.data.copy_(text_encoder_state['hidden'])
                # Remove hidden for load_state_dict
                text_encoder_state = {k: v for k, v in text_encoder_state.items() if k != 'hidden'}
            self.t2m_textencoder.load_state_dict(text_encoder_state, strict=False)
            
            # Load the movement encoder
            self.t2m_moveencoder.load_state_dict(checkpoint['movement_encoder'], strict=True)
            
            # Load the motion encoder (including hidden state)
            motion_encoder_state = checkpoint['motion_encoder']
            if 'hidden' in motion_encoder_state:
                self.t2m_motionencoder.hidden.data.copy_(motion_encoder_state['hidden'])
                motion_encoder_state = {k: v for k, v in motion_encoder_state.items() if k != 'hidden'}
            self.t2m_motionencoder.load_state_dict(motion_encoder_state, strict=False)
            
            print(f"Loaded T2M evaluator from {checkpoint_path}")
        else:
            print(f"Warning: T2M evaluator checkpoint not found at {checkpoint_path}")
            print("Metrics will be computed with randomly initialized models!")
        
        # Move to device and freeze
        self.t2m_textencoder.to(self.device).eval()
        self.t2m_moveencoder.to(self.device).eval()
        self.t2m_motionencoder.to(self.device).eval()
        
        for model in [self.t2m_textencoder, self.t2m_moveencoder, self.t2m_motionencoder]:
            for p in model.parameters():
                p.requires_grad = False
    
    def reset(self):
        """Reset stored embeddings."""
        self.text_embeddings = []
        self.gen_motion_embeddings = []
        self.gt_motion_embeddings = []
    
    @torch.no_grad()
    def get_text_embedding(
        self, 
        word_embs: torch.Tensor, 
        pos_ohot: torch.Tensor, 
        cap_lens: torch.Tensor
    ) -> torch.Tensor:
        """Get text embeddings."""
        word_embs = word_embs.to(self.device)
        pos_ohot = pos_ohot.to(self.device)
        cap_lens = cap_lens.to(self.device)
        
        text_emb = self.t2m_textencoder(word_embs, pos_ohot, cap_lens)
        return text_emb.flatten(start_dim=1)
    
    @torch.no_grad()
    def get_motion_embedding(
        self, 
        motion: torch.Tensor, 
        m_lens: List[int],
    ) -> torch.Tensor:
        """
        Get motion embeddings.
        
        Args:
            motion: [bs, max_len, nfeats] motion features (normalized)
                   Supports 263-dim or 269-dim input; 269-dim input is automatically truncated to the first 263 dims
            m_lens: [bs] motion lengths (raw frame counts)
        Returns:
            [bs, 512] motion embedding
        """
        motion = motion.to(self.device)
        bs = motion.shape[0]
        
        # Handle 269-dim motion automatically: keep the first 263 dims for compatibility with the pretrained evaluator
        if motion.shape[-1] > 263:
            motion = motion[..., :263]
        
        # Sort for pack_padded_sequence (must be descending)
        m_lens_array = np.array(m_lens)
        align_idx = np.argsort(m_lens_array)[::-1].copy()
        motion_sorted = motion[align_idx]
        sorted_lens = m_lens_array[align_idx]
        
        # Movement encoder (input excludes the last 4 root feature dims)
        mov = self.t2m_moveencoder(motion_sorted[..., :-4])
        
        # Compute the lengths required by the motion encoder
        # In MotionGPT3: m_lens // unit_length // unit_length = m_lens // 16
        # Because the movement encoder downsamples by 4x, the motion encoder needs the valid length of mov
        m_lens_div = torch.tensor(sorted_lens).to(self.device)
        m_lens_div = torch.div(m_lens_div, self.unit_length, rounding_mode='floor')
        m_lens_div = m_lens_div // self.unit_length  # Double division like in MotionGPT3
        
        # Motion encoder
        emb = self.t2m_motionencoder(mov, m_lens_div)
        
        # Restore original order
        cache = torch.zeros_like(emb)
        for i in range(len(align_idx)):
            cache[align_idx[i]] = emb[i]
        
        return cache.flatten(start_dim=1)
    
    def update(
        self,
        gt_motion: torch.Tensor,
        gen_motion: torch.Tensor,
        gt_lengths: List[int],
        gen_lengths: List[int],
        word_embs: torch.Tensor = None,
        pos_ohot: torch.Tensor = None,
        cap_lens: torch.Tensor = None,
    ):
        """
        Update evaluation data.
        
        Args:
            gt_motion: [bs, max_len, nfeats] GT motion (normalized)
            gen_motion: [bs, max_len, nfeats] generated motion (normalized)
            gt_lengths: [bs] GT motion lengths (raw frame counts)
            gen_lengths: [bs] generated motion lengths (raw frame counts)
            word_embs: [bs, max_text_len, word_size] word embeddings
            pos_ohot: [bs, max_text_len, pos_size] POS one-hot
            cap_lens: [bs] text lengths
        """
        # Get motion embeddings
        gt_emb = self.get_motion_embedding(gt_motion, gt_lengths)
        gen_emb = self.get_motion_embedding(gen_motion, gen_lengths)
        
        self.gt_motion_embeddings.append(gt_emb.cpu())
        self.gen_motion_embeddings.append(gen_emb.cpu())
        
        # Get text embeddings if provided
        if word_embs is not None and pos_ohot is not None and cap_lens is not None:
            text_emb = self.get_text_embedding(word_embs, pos_ohot, cap_lens)
            self.text_embeddings.append(text_emb.cpu())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Returns:
            Dict containing the metrics
        """
        # Concatenate all embeddings
        all_gen_motions = torch.cat(self.gen_motion_embeddings, dim=0)
        all_gt_motions = torch.cat(self.gt_motion_embeddings, dim=0)
        
        count_seq = all_gen_motions.shape[0]
        
        # Shuffle
        shuffle_idx = torch.randperm(count_seq)
        all_gen_motions = all_gen_motions[shuffle_idx]
        all_gt_motions = all_gt_motions[shuffle_idx]
        
        metrics = {}
        
        # Text-related metrics (if text embeddings available)
        if len(self.text_embeddings) > 0:
            all_texts = torch.cat(self.text_embeddings, dim=0)[shuffle_idx]
            
            # R-Precision and Matching Score
            matching_score = 0.0
            top_k_mat = torch.zeros((self.top_k,))
            R_count = count_seq // self.R_size * self.R_size
            
            for i in range(count_seq // self.R_size):
                group_texts = all_texts[i * self.R_size:(i + 1) * self.R_size]
                group_motions = all_gen_motions[i * self.R_size:(i + 1) * self.R_size]
                
                dist_mat = euclidean_distance_matrix(group_texts, group_motions).nan_to_num()
                matching_score += dist_mat.trace()
                
                argsmax = torch.argsort(dist_mat, dim=1)
                top_k_mat += calculate_top_k(argsmax, self.top_k).sum(axis=0)
            
            metrics['Matching_Score'] = float(matching_score / R_count)
            # Paper naming compatibility
            metrics['MMDist'] = metrics['Matching_Score']
            for k in range(self.top_k):
                metrics[f'R_Precision_Top_{k+1}'] = float(top_k_mat[k] / R_count)
            
            # GT R-Precision and Matching Score
            gt_matching_score = 0.0
            gt_top_k_mat = torch.zeros((self.top_k,))
            
            for i in range(count_seq // self.R_size):
                group_texts = all_texts[i * self.R_size:(i + 1) * self.R_size]
                group_motions = all_gt_motions[i * self.R_size:(i + 1) * self.R_size]
                
                dist_mat = euclidean_distance_matrix(group_texts, group_motions).nan_to_num()
                gt_matching_score += dist_mat.trace()
                
                argsmax = torch.argsort(dist_mat, dim=1)
                gt_top_k_mat += calculate_top_k(argsmax, self.top_k).sum(axis=0)
            
            metrics['gt_Matching_Score'] = float(gt_matching_score / R_count)
            metrics['gt_MMDist'] = metrics['gt_Matching_Score']
            for k in range(self.top_k):
                metrics[f'gt_R_Precision_Top_{k+1}'] = float(gt_top_k_mat[k] / R_count)
        
        # Convert to numpy for FID and Diversity
        all_gen_motions_np = all_gen_motions.numpy()
        all_gt_motions_np = all_gt_motions.numpy()
        
        # FID
        gen_mu, gen_cov = calculate_activation_statistics(all_gen_motions_np)
        gt_mu, gt_cov = calculate_activation_statistics(all_gt_motions_np)
        metrics['FID'] = calculate_frechet_distance(gt_mu, gt_cov, gen_mu, gen_cov)
        # MotionGPT-compatible naming:
        # - new_fid: same definition as MotionGPT/TM2TMetrics FID(gen vs gt)
        # - new_gt_fid: Real-row style GT-vs-GT same-distribution FID
        metrics['new_fid'] = metrics['FID']
        metrics['new_gt_fid'] = calculate_frechet_distance(gt_mu, gt_cov, gt_mu, gt_cov)
        # GT FID (Real row style): FID between two random halves of GT embeddings.
        if count_seq >= 4:
            half = count_seq // 2
            gt_a = all_gt_motions_np[:half]
            gt_b = all_gt_motions_np[half:half * 2]
            gt_a_mu, gt_a_cov = calculate_activation_statistics(gt_a)
            gt_b_mu, gt_b_cov = calculate_activation_statistics(gt_b)
            metrics['gt_FID'] = calculate_frechet_distance(gt_a_mu, gt_a_cov, gt_b_mu, gt_b_cov)
        else:
            metrics['gt_FID'] = 0.0
        
        # Diversity
        if count_seq > self.diversity_times:
            metrics['Diversity'] = calculate_diversity(all_gen_motions_np, self.diversity_times)
            metrics['gt_Diversity'] = calculate_diversity(all_gt_motions_np, self.diversity_times)
        else:
            metrics['Diversity'] = calculate_diversity(all_gen_motions_np, min(count_seq, 100))
            metrics['gt_Diversity'] = calculate_diversity(all_gt_motions_np, min(count_seq, 100))
        
        # Reset after computation
        self.reset()
        
        return metrics


def load_t2m_evaluator(
    evaluator_path: str,
    glove_dir: str,
    dataname: str = 't2m',
    device: str = 'cuda',
) -> T2MEvaluator:
    """
    Convenience function: load the T2M evaluator.
    
    Args:
        evaluator_path: root directory of the evaluator weights
        glove_dir: GloVe word-vector directory
        dataname: dataset name
        device: compute device
    Returns:
        A T2MEvaluator instance
    """
    return T2MEvaluator(
        t2m_evaluator_path=evaluator_path,
        glove_dir=glove_dir,
        dataname=dataname,
        device=device,
    )
