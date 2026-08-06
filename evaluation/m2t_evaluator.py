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
M2T (Motion-to-Text) Evaluation Module for UniMotion.
Fully consistent with the MotionGPT3 evaluation protocol; supports the following metrics:
- R-Precision Top-1/2/3
- gt_R-Precision Top-1/2/3
- Matching_score
- gt_Matching_score
- ROUGE_L
- CIDEr
- METEOR
- Bleu_1, Bleu_4
- Bert_F1
"""

import os
import numpy as np
import torch
import torch.nn as nn
import spacy
from typing import List, Dict, Optional, Tuple

from .t2m_evaluator import (
    MovementConvEncoder,
    MotionEncoderBiGRUCo,
    TextEncoderBiGRUCo,
    WordVectorizer,
    euclidean_distance_matrix,
    calculate_top_k,
    POS_enumerator,
    VIP_dict,
)


class M2TEvaluator:
    """
    Evaluator for the M2T (Motion-to-Text) task.
    Fully consistent with MotionGPT3's M2TMetrics.
    
    Evaluation metrics:
    - R-Precision: matching precision between predicted text and GT motion
    - gt_R-Precision: matching precision between GT text and GT motion
    - Matching_score: matching score between predicted text and GT motion
    - gt_Matching_score: matching score between GT text and GT motion
    - NLG metrics: ROUGE-L, CIDEr, METEOR, Bleu-1, Bleu-4
    - Bert_F1: BertScore F1
    """
    
    def __init__(
        self,
        t2m_evaluator_path: str,
        glove_dir: str,
        dataname: str = 't2m',
        device: str = 'cuda',
        top_k: int = 3,
        R_size: int = 32,
        max_text_len: int = 40,  # M2T uses a longer text limit
        unit_length: int = 4,
    ):
        """
        Args:
            t2m_evaluator_path: path to the T2M evaluator weights
            glove_dir: GloVe word-vector directory
            dataname: dataset name
            device: compute device
            top_k: k value for R-Precision
            R_size: batch size for R-Precision
            max_text_len: maximum text length
            unit_length: motion length unit
        """
        self.device = device
        self.dataname = dataname
        self.top_k = top_k
        self.R_size = R_size
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        
        # Load word vectorizer
        self.w_vectorizer = WordVectorizer(glove_dir, dataname)
        
        # Load evaluation models
        self._load_evaluator(t2m_evaluator_path)
        
        # Load spacy for text processing
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Warning: spacy 'en_core_web_sm' not found. Downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load('en_core_web_sm')
        
        # Initialize NLG evaluator
        self._init_nlg_evaluator()
        
        # Storage
        self.reset()
    
    def _load_evaluator(self, evaluator_path: str):
        """Load T2M evaluator models."""
        if self.dataname == 'kit':
            input_size = 251
        else:
            input_size = 263
        
        movement_input_size = input_size - 4
        
        # Initialize models
        self.t2m_moveencoder = MovementConvEncoder(
            input_size=movement_input_size,
            hidden_size=512,
            output_size=512
        )
        
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
            
            # Load text encoder
            text_encoder_state = checkpoint['text_encoder']
            if 'hidden' in text_encoder_state:
                self.t2m_textencoder.hidden.data.copy_(text_encoder_state['hidden'])
                text_encoder_state = {k: v for k, v in text_encoder_state.items() if k != 'hidden'}
            self.t2m_textencoder.load_state_dict(text_encoder_state, strict=False)
            
            # Load movement encoder
            self.t2m_moveencoder.load_state_dict(checkpoint['movement_encoder'], strict=True)
            
            # Load motion encoder
            motion_encoder_state = checkpoint['motion_encoder']
            if 'hidden' in motion_encoder_state:
                self.t2m_motionencoder.hidden.data.copy_(motion_encoder_state['hidden'])
                motion_encoder_state = {k: v for k, v in motion_encoder_state.items() if k != 'hidden'}
            self.t2m_motionencoder.load_state_dict(motion_encoder_state, strict=False)
            
            print(f"Loaded M2T evaluator from {checkpoint_path}")
        else:
            print(f"Warning: M2T evaluator checkpoint not found at {checkpoint_path}")
        
        # Move to device and freeze
        self.t2m_textencoder.to(self.device).eval()
        self.t2m_moveencoder.to(self.device).eval()
        self.t2m_motionencoder.to(self.device).eval()
        
        for model in [self.t2m_textencoder, self.t2m_moveencoder, self.t2m_motionencoder]:
            for p in model.parameters():
                p.requires_grad = False
    
    def _init_nlg_evaluator(self):
        """Initialize the NLG evaluator."""
        try:
            import sys
            import os
            
            # Save original state
            original_path = sys.path.copy()
            original_modules = {k: v for k, v in sys.modules.items() if k == 'datasets' or k.startswith('datasets.')}
            
            # 1. Remove the local datasets module cache from sys.modules
            keys_to_remove = [k for k in sys.modules.keys() if k == 'datasets' or k.startswith('datasets.')]
            for key in keys_to_remove:
                del sys.modules[key]
            
            # 2. Modify sys.path to remove the project directory
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sys.path = [p for p in sys.path if not p.startswith(project_dir)]
            
            # 3. Now import nlgmetricverse (it will use Hugging Face's datasets)
            from nlgmetricverse import NLGMetricverse, load_metric
            metrics = [
                load_metric("bleu", resulting_name="bleu_1", compute_kwargs={"max_order": 1}),
                load_metric("bleu", resulting_name="bleu_4", compute_kwargs={"max_order": 4}),
                load_metric("rouge"),
                load_metric("cider"),
            ]
            self.has_meteor = False
            try:
                metrics.append(load_metric("meteor"))
                self.has_meteor = True
            except Exception as e:
                print(f"Warning: METEOR metric unavailable, skipping: {e}")
            self.nlg_evaluator = NLGMetricverse(metrics)
            self.has_nlg = True
            if self.has_meteor:
                print("NLG evaluator initialized with BLEU/ROUGE/CIDEr/METEOR")
            else:
                print("NLG evaluator initialized with BLEU/ROUGE/CIDEr")
            
            # 4. Restore the original sys.path
            sys.path = original_path
            
            # 5. Restore the local datasets module (if needed)
            # Note: no restore here, since nlgmetricverse has already loaded the correct datasets
            
        except Exception as e:
            # Restore original state
            import sys
            if 'original_path' in locals():
                sys.path = original_path
            if 'original_modules' in locals():
                for k, v in original_modules.items():
                    sys.modules[k] = v
            print(f"Warning: nlgmetricverse not available: {e}")
            print("  NLG metrics (ROUGE, CIDEr, METEOR, BLEU) will not be computed")
            self.nlg_evaluator = None
            self.has_nlg = False
            self.has_meteor = False
    
    def reset(self):
        """Reset storage."""
        self.pred_texts = []
        self.gt_texts = []
        self.predtext_embeddings = []
        self.gttext_embeddings = []
        self.gtmotion_embeddings = []
    
    def _process_text(self, sentence: str) -> Tuple[List[str], List[str]]:
        """
        Process text and extract words and POS tags.
        Identical to MotionGPT3.
        """
        sentence = sentence.replace('-', '')
        doc = self.nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list
    
    @torch.no_grad()
    def _get_text_embeddings_from_sentences(self, texts: List[str]) -> torch.Tensor:
        """
        Get text embeddings from a list of sentences.
        Used for predicted text (requires spacy tokenization first).
        Identical to MotionGPT3's M2TMetrics._get_text_embeddings.
        """
        word_embs = []
        pos_ohot = []
        text_lengths = []
        
        for sentence in texts:
            word_list, pos_list = self._process_text(sentence.strip())
            t_tokens = [
                '%s/%s' % (word_list[i], pos_list[i])
                for i in range(len(word_list))
            ]
            
            if len(t_tokens) < self.max_text_len:
                tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
            else:
                tokens = t_tokens[:self.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
            
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(torch.tensor(pos_oh).float()[None])
                word_embeddings.append(torch.tensor(word_emb).float()[None])
            
            text_lengths.append(sent_len)
            pos_ohot.append(torch.cat(pos_one_hots, dim=0)[None])
            word_embs.append(torch.cat(word_embeddings, dim=0)[None])
        
        word_embs = torch.cat(word_embs, dim=0).to(self.device)
        pos_ohot = torch.cat(pos_ohot, dim=0).to(self.device)
        text_lengths = torch.tensor(text_lengths).to(self.device)
        
        align_idx = np.argsort(text_lengths.cpu().numpy())[::-1].copy()
        
        text_embeddings = self.t2m_textencoder(
            word_embs[align_idx],
            pos_ohot[align_idx],
            text_lengths[align_idx]
        )
        
        # Restore original order
        original_text_embeddings = text_embeddings.clone()
        for idx, sort in enumerate(align_idx):
            original_text_embeddings[sort] = text_embeddings[idx]
        
        return original_text_embeddings
    
    @torch.no_grad()
    def get_text_embedding(
        self,
        word_embs: torch.Tensor,
        pos_ohot: torch.Tensor,
        cap_lens: torch.Tensor
    ) -> torch.Tensor:
        """Get GT text embeddings (word_embs and pos_ohot already available)."""
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
        
        # Handle 269-dim motion automatically: keep the first 263 dims for compatibility with the pretrained evaluator
        if motion.shape[-1] > 263:
            motion = motion[..., :263]
        
        m_lens_array = np.array(m_lens)
        align_idx = np.argsort(m_lens_array)[::-1].copy()
        motion_sorted = motion[align_idx]
        sorted_lens = m_lens_array[align_idx]
        
        mov = self.t2m_moveencoder(motion_sorted[..., :-4])
        
        m_lens_div = torch.tensor(sorted_lens).to(self.device)
        m_lens_div = torch.div(m_lens_div, self.unit_length, rounding_mode='floor')
        m_lens_div = m_lens_div // self.unit_length
        
        emb = self.t2m_motionencoder(mov, m_lens_div)
        
        cache = torch.zeros_like(emb)
        for i in range(len(align_idx)):
            cache[align_idx[i]] = emb[i]
        
        return cache.flatten(start_dim=1)
    
    def update(
        self,
        gt_motion: torch.Tensor,
        pred_texts: List[str],
        gt_texts: List[List[str]],
        gt_lengths: List[int],
        word_embs: torch.Tensor = None,
        pos_ohot: torch.Tensor = None,
        cap_lens: torch.Tensor = None,
    ):
        """
        Update evaluation data.
        
        Args:
            gt_motion: [bs, max_len, nfeats] GT motion (normalized)
            pred_texts: list of predicted text strings
            gt_texts: list of list of GT text strings (all_captions)
            gt_lengths: [bs] GT motion lengths
            word_embs: [bs, max_text_len, word_size] GT word embeddings
            pos_ohot: [bs, max_text_len, pos_size] GT POS one-hot
            cap_lens: [bs] GT text lengths
        """
        # Motion embedding
        motion_emb = self.get_motion_embedding(gt_motion, gt_lengths)
        self.gtmotion_embeddings.append(motion_emb.cpu())
        
        # GT text embedding
        if word_embs is not None and pos_ohot is not None and cap_lens is not None:
            gt_text_emb = self.get_text_embedding(word_embs, pos_ohot, cap_lens)
            self.gttext_embeddings.append(gt_text_emb.cpu())
        
        # Predicted text embedding
        pred_text_emb = self._get_text_embeddings_from_sentences(pred_texts)
        self.predtext_embeddings.append(pred_text_emb.flatten(start_dim=1).cpu())
        
        # Store texts for NLG evaluation
        self.pred_texts.extend(pred_texts)
        # gt_texts is list of list, keep all captions for multi-reference NLG evaluation
        # Format: [[cap1, cap2, cap3], [cap1, cap2, cap3], ...]
        for caps in gt_texts:
            if isinstance(caps, list):
                self.gt_texts.append(caps)  # keep all reference texts
            else:
                self.gt_texts.append([caps])  # wrap single texts in a list too
    
    def compute(self) -> Dict[str, float]:
        """Compute all evaluation metrics."""
        all_pred_texts = torch.cat(self.predtext_embeddings, dim=0)
        all_gt_motions = torch.cat(self.gtmotion_embeddings, dim=0)
        
        count_seq = all_pred_texts.shape[0]
        
        # Shuffle
        shuffle_idx = torch.randperm(count_seq)
        all_pred_texts = all_pred_texts[shuffle_idx]
        all_gt_motions = all_gt_motions[shuffle_idx]
        
        metrics = {}
        
        # R-Precision and Matching Score (pred text vs gt motion)
        if count_seq >= self.R_size:
            matching_score = 0.0
            top_k_mat = torch.zeros((self.top_k,))
            R_count = count_seq // self.R_size * self.R_size
            
            for i in range(count_seq // self.R_size):
                group_texts = all_pred_texts[i * self.R_size:(i + 1) * self.R_size]
                group_motions = all_gt_motions[i * self.R_size:(i + 1) * self.R_size]
                
                dist_mat = euclidean_distance_matrix(group_texts, group_motions).nan_to_num()
                matching_score += dist_mat.trace()
                
                argsmax = torch.argsort(dist_mat, dim=1)
                top_k_mat += calculate_top_k(argsmax, self.top_k).sum(axis=0)
            
            metrics['Matching_Score'] = float(matching_score / R_count)
            for k in range(self.top_k):
                metrics[f'R_Precision_Top_{k+1}'] = float(top_k_mat[k] / R_count)
        
        # GT R-Precision and Matching Score (gt text vs gt motion)
        if len(self.gttext_embeddings) > 0:
            all_gt_texts = torch.cat(self.gttext_embeddings, dim=0)[shuffle_idx]
            
            if count_seq >= self.R_size:
                gt_matching_score = 0.0
                gt_top_k_mat = torch.zeros((self.top_k,))
                
                for i in range(count_seq // self.R_size):
                    group_texts = all_gt_texts[i * self.R_size:(i + 1) * self.R_size]
                    group_motions = all_gt_motions[i * self.R_size:(i + 1) * self.R_size]
                    
                    dist_mat = euclidean_distance_matrix(group_texts, group_motions).nan_to_num()
                    gt_matching_score += dist_mat.trace()
                    
                    argsmax = torch.argsort(dist_mat, dim=1)
                    gt_top_k_mat += calculate_top_k(argsmax, self.top_k).sum(axis=0)
                
                metrics['gt_Matching_Score'] = float(gt_matching_score / R_count)
                for k in range(self.top_k):
                    metrics[f'gt_R_Precision_Top_{k+1}'] = float(gt_top_k_mat[k] / R_count)
        
        # NLG metrics
        if self.has_nlg and len(self.pred_texts) > 0:
            try:
                scores = self.nlg_evaluator(
                    predictions=self.pred_texts,
                    references=self.gt_texts
                )
                
                # BLEU scores
                for key in scores.keys():
                    if 'bleu' in key:
                        metrics[key] = float(scores[key]['score'])
                
                # ROUGE-L
                if 'rouge' in scores:
                    metrics['ROUGE_L'] = float(scores['rouge']['rougeL'])
                
                # CIDEr
                if 'cider' in scores:
                    metrics['CIDEr'] = float(scores['cider']['score'])
                
                # METEOR
                if 'meteor' in scores:
                    meteor_score = scores['meteor']
                    if isinstance(meteor_score, dict):
                        meteor_score = meteor_score.get('score', meteor_score.get('meteor', 0.0))
                    metrics['METEOR'] = float(meteor_score)
                    
            except Exception as e:
                print(f"Warning: NLG evaluation failed: {e}")
        
        # BertScore - identical to MotionGPT3: pass multi-reference lists
        # The bert_score library supports List[List[str]] references and automatically takes the highest F1 per prediction
        try:
            from bert_score import score as score_bert
            P, R, F1 = score_bert(
                self.pred_texts,
                self.gt_texts,  # List[List[str]], consistent with MotionGPT3
                lang='en',
                rescale_with_baseline=True,
                idf=True,
                device=self.device,
                verbose=False
            )
            metrics['Bert_F1'] = float(F1.mean())
        except Exception as e:
            print(f"Warning: BertScore evaluation failed: {e}")
        
        # Reset
        self.reset()
        
        return metrics


def load_m2t_evaluator(
    evaluator_path: str,
    glove_dir: str,
    dataname: str = 't2m',
    device: str = 'cuda',
) -> M2TEvaluator:
    """Convenience function: load the M2T evaluator."""
    return M2TEvaluator(
        t2m_evaluator_path=evaluator_path,
        glove_dir=glove_dir,
        dataname=dataname,
        device=device,
    )
