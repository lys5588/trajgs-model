"""
UniMotion DataModule for PyTorch Lightning.
Jointly trains on the HumanML3D, MotionFix, and H36M datasets.
"""
import numpy as np
import torch
from os.path import join as pjoin
from torch.utils.data import ConcatDataset, DataLoader

try:
    from .humanml.scripts.motion_process import recover_from_ric
except ImportError:
    # Absolute-import fallback for standalone entrypoints run from cma_vae/
    from data.humanml.scripts.motion_process import recover_from_ric

from .base import BASEDataModule
from .unimotion.dataset import UniMotionDataset
from .unimotion.dataset_with_image import UniMotionDatasetWithImage


class UniMotionDataModule(BASEDataModule):
    """
    UniMotion DataModule for VAE training.
    Jointly loads the HumanML3D, MotionFix, and H36M datasets.
    """

    def __init__(self,
                 cfg,
                 batch_size,
                 num_workers,
                 collate_fn=None,
                 phase="train",
                 # HumanML3D parameters
                 mean_humanml3d=None,
                 std_humanml3d=None,
                 motion_dir_humanml3d=None,
                 split_root_humanml3d=None,
                 # MotionFix parameters
                 mean_motionfix=None,
                 std_motionfix=None,
                 motion_dir_motionfix=None,
                 split_root_motionfix=None,
                 # H36M parameters
                 mean_h36m=None,
                 std_h36m=None,
                 motion_dir_h36m=None,
                 split_root_h36m=None,
                 # Shared parameters
                 max_motion_length=196,
                 min_motion_length=40,
                 unit_length=4,
                 fps=20,
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=collate_fn)
        self.save_hyperparameters(logger=False)
        self.name = "unimotion"
        self.njoints = 22  # same as HumanML3D

        self.cfg = cfg
        self.phase = phase

        self.mean_humanml3d = mean_humanml3d
        self.std_humanml3d = std_humanml3d
        self.motion_dir_humanml3d = motion_dir_humanml3d
        self.split_root_humanml3d = split_root_humanml3d

        self.mean_motionfix = mean_motionfix
        self.std_motionfix = std_motionfix
        self.motion_dir_motionfix = motion_dir_motionfix
        self.split_root_motionfix = split_root_motionfix

        self.mean_h36m = mean_h36m
        self.std_h36m = std_h36m
        self.motion_dir_h36m = motion_dir_h36m
        self.split_root_h36m = split_root_h36m

        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length
        self.fps = fps

        # Use HumanML3D mean/std as the primary normalization statistics
        self.hparams['mean'] = mean_humanml3d
        self.hparams['std'] = std_humanml3d

        # Images are only loaded during training; evaluation/inference does
        # not need images or 2D joints.
        use_images_config = getattr(cfg.DATASET.UNIMOTION, 'USE_IMAGES', False)
        self.use_images = use_images_config and (phase == "train")
        
        if self.use_images and phase != "train":
            print(f"[UniMotion] Phase='{phase}': Disabling image loading (only needed for training)")
        
        self.Dataset = UniMotionDatasetWithImage if self.use_images else UniMotionDataset

        # Build a tiny sample set to infer dataset properties
        sample_overrides = {
            "split": "val",
            "tiny": True,
            "progress_bar": False
        }
        self._sample_set = self._get_single_dataset("humanml3d", overrides=sample_overrides)

        self.nfeats = self._sample_set.nfeats

    def _get_single_dataset(self, dataset_type, split=None, overrides={}):
        """Instantiate a single sub-dataset."""
        if dataset_type == "humanml3d":
            mean = self.mean_humanml3d
            std = self.std_humanml3d
            motion_dir = self.motion_dir_humanml3d
            split_root = self.split_root_humanml3d
            dataset_source = 0
            data_type = "sequence"
            # Per-dataset MIN_MOTION_LEN, falling back to the global setting
            sub_cfg = getattr(self.cfg.DATASET.UNIMOTION, 'HUMANML3D', None)
            min_motion_len = getattr(sub_cfg, 'MIN_MOTION_LEN', self.min_motion_length) if sub_cfg else self.min_motion_length
            # Sequence data has no images
            image_params = {}
        elif dataset_type == "motionfix":
            mean = self.mean_motionfix
            std = self.std_motionfix
            motion_dir = self.motion_dir_motionfix
            split_root = self.split_root_motionfix
            dataset_source = 1
            data_type = "sequence"
            # Per-dataset MIN_MOTION_LEN, falling back to the global setting
            sub_cfg = getattr(self.cfg.DATASET.UNIMOTION, 'MOTIONFIX', None)
            min_motion_len = getattr(sub_cfg, 'MIN_MOTION_LEN', self.min_motion_length) if sub_cfg else self.min_motion_length
            # Sequence data has no images
            image_params = {}
        else:  # h36m
            mean = self.mean_h36m
            std = self.std_h36m
            motion_dir = self.motion_dir_h36m
            split_root = self.split_root_h36m
            dataset_source = 2
            data_type = "single_frame"
            min_motion_len = 1  # H36M is single-frame data; no minimum length

            image_params = {}
            if self.use_images:
                h36m_cfg = self.cfg.DATASET.UNIMOTION.H36M
                image_params = {
                    "image_root": getattr(h36m_cfg, 'IMAGE_ROOT', None),
                    "image_paths_file": getattr(h36m_cfg, 'IMAGE_PATHS_FILE', None),
                    "img_size": getattr(h36m_cfg, 'IMG_SIZE', (256, 192)),
                    "load_images": True,
                    # Precomputed 2D joint parameters
                    "joints2d_file": getattr(h36m_cfg, 'JOINTS2D_FILE', None),
                    "use_precomputed_joints": getattr(h36m_cfg, 'USE_PRECOMPUTED_JOINTS', False),
                }

        if split is None:
            split = overrides.get("split", "train")
        split_file = pjoin(split_root, split + ".txt")

        params = {
            "mean": mean,
            "std": std,
            "split_file": split_file,
            "max_motion_length": self.max_motion_length,
            "min_motion_length": min_motion_len,
            "unit_length": self.unit_length,
            "motion_dir": motion_dir,
            "fps": self.fps,
            "dataset_source": dataset_source,
            "data_type": data_type,
        }
        params.update(image_params)
        params.update(overrides)

        return self.Dataset(**params)

    def _get_combined_dataset(self, split):
        """Build the concatenated dataset over all sources."""
        datasets = []

        try:
            dataset_humanml3d = self._get_single_dataset("humanml3d", split=split)
            datasets.append(dataset_humanml3d)
            print(f"[UniMotion] {split}: HumanML3D = {len(dataset_humanml3d)}")
        except Exception as e:
            print(f"[UniMotion] Warning: Failed to load HumanML3D {split}: {e}")

        try:
            dataset_motionfix = self._get_single_dataset("motionfix", split=split)
            datasets.append(dataset_motionfix)
            print(f"[UniMotion] {split}: MotionFix = {len(dataset_motionfix)}")
        except Exception as e:
            print(f"[UniMotion] Warning: Failed to load MotionFix {split}: {e}")

        try:
            dataset_h36m = self._get_single_dataset("h36m", split=split)
            datasets.append(dataset_h36m)
            print(f"[UniMotion] {split}: H36M = {len(dataset_h36m)}")
        except Exception as e:
            print(f"[UniMotion] Warning: Failed to load H36M {split}: {e}")

        if len(datasets) == 0:
            raise ValueError(f"No datasets loaded for split {split}")

        total = sum(len(d) for d in datasets)
        print(f"[UniMotion] {split}: Total = {total}")

        return ConcatDataset(datasets)

    def __getattr__(self, item):
        # train_dataset/val_dataset etc. cached like properties
        if item.endswith("_dataset") and not item.startswith("_"):
            subset = item[:-len("_dataset")]
            item_c = "_" + item
            if item_c not in self.__dict__:
                subset_upper = subset.upper() if subset != "val" else "EVAL"
                split = eval(f"self.cfg.{subset_upper}.SPLIT")
                self.__dict__[item_c] = self._get_combined_dataset(split)
            return getattr(self, item_c)
        classname = self.__class__.__name__
        raise AttributeError(f"'{classname}' object has no attribute '{item}'")

    def setup(self, stage=None):
        self.stage = stage
        if stage in (None, "fit"):
            _ = self.train_dataset
            _ = self.val_dataset
        if stage in (None, "test"):
            _ = self.test_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            persistent_workers=True,
            **self.dataloader_options,
        )

    def val_dataloader(self):
        dataloader_options = self.dataloader_options.copy()
        dataloader_options["batch_size"] = self.cfg.EVAL.BATCH_SIZE
        dataloader_options["num_workers"] = self.cfg.EVAL.NUM_WORKERS
        dataloader_options["shuffle"] = False
        return DataLoader(
            self.val_dataset,
            persistent_workers=True,
            **dataloader_options,
        )

    def test_dataloader(self):
        dataloader_options = self.dataloader_options.copy()
        dataloader_options["batch_size"] = 1 if self.is_mm else self.cfg.TEST.BATCH_SIZE
        dataloader_options["num_workers"] = self.cfg.TEST.NUM_WORKERS
        dataloader_options["shuffle"] = False
        return DataLoader(
            self.test_dataset,
            persistent_workers=True,
            **dataloader_options,
        )

    def predict_dataloader(self):
        return self.test_dataloader()

    def feats2joints(self, features, dataset_source=None):
        """Convert features back to joint positions.

        Args:
            features: feature tensor [batch_size, seq_len, nfeats]
            dataset_source: dataset source tensor [batch_size],
                0=humanml3d, 1=motionfix, 2=h36m. If None, HumanML3D
                mean/std is used (backward compatibility).
        """
        if dataset_source is None:
            mean = torch.tensor(self.mean_humanml3d).to(features)
            std = torch.tensor(self.std_humanml3d).to(features)
            features = features * std + mean
        else:
            # Denormalize each sample with its source dataset's statistics
            mean_h3d = torch.tensor(self.mean_humanml3d).to(features)
            std_h3d = torch.tensor(self.std_humanml3d).to(features)
            mean_mf = torch.tensor(self.mean_motionfix).to(features)
            std_mf = torch.tensor(self.std_motionfix).to(features)
            mean_h36m = torch.tensor(self.mean_h36m).to(features)
            std_h36m = torch.tensor(self.std_h36m).to(features)

            mask_h3d = (dataset_source == 0).unsqueeze(-1).unsqueeze(-1)
            mask_mf = (dataset_source == 1).unsqueeze(-1).unsqueeze(-1)
            mask_h36m = (dataset_source == 2).unsqueeze(-1).unsqueeze(-1)

            features_h3d = features * std_h3d + mean_h3d
            features_mf = features * std_mf + mean_mf
            features_h36m = features * std_h36m + mean_h36m
            features = (features_h3d * mask_h3d.float() + 
                       features_mf * mask_mf.float() + 
                       features_h36m * mask_h36m.float())

        return recover_from_ric(features, self.njoints)

    def joints2feats(self, features):
        """Convert joint positions to features."""
        from data.humanml.scripts.motion_process import process_file
        features = process_file(features, self.njoints)[0]
        return features

    def renorm4t2m(self, features):
        """Renormalize features to the T2M evaluator's statistics."""
        ori_mean = torch.tensor(self.hparams['mean']).to(features)
        ori_std = torch.tensor(self.hparams['std']).to(features)
        features = features * ori_std + ori_mean
        features = (features - ori_mean) / ori_std
        return features

    def mm_mode(self, mm_on=True):
        """Multimodal evaluation mode; not supported for the combined dataset."""
        pass
