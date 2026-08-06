from os.path import join as pjoin

import numpy as np

from .UniMotion import UniMotionDataModule
from .utils import mld_collate_with_source


def get_datasets(cfg, logger=None, phase="train"):
    dataset_names = getattr(cfg, phase.upper()).DATASETS
    datasets = []
    for dataset_name in dataset_names:
        if dataset_name.lower() != "unimotion":
            raise NotImplementedError(
                f"Only the `unimotion` dataset is supported, got {dataset_name}")

        # UniMotion: HumanML3D + MotionFix + H36M, each normalized with its
        # own Mean/Std statistics.
        data_root_humanml3d = cfg.DATASET.UNIMOTION.HUMANML3D.ROOT
        data_root_motionfix = cfg.DATASET.UNIMOTION.MOTIONFIX.ROOT
        data_root_h36m = cfg.DATASET.UNIMOTION.H36M.ROOT

        mean_humanml3d = np.load(pjoin(data_root_humanml3d, "Mean.npy"))
        std_humanml3d = np.load(pjoin(data_root_humanml3d, "Std.npy"))
        mean_motionfix = np.load(pjoin(data_root_motionfix, "Mean.npy"))
        std_motionfix = np.load(pjoin(data_root_motionfix, "Std.npy"))
        mean_h36m = np.load(pjoin(data_root_h36m, "Mean.npy"))
        std_h36m = np.load(pjoin(data_root_h36m, "Std.npy"))

        dataset_cfg = cfg.DATASET.UNIMOTION
        max_motion_length = getattr(dataset_cfg, 'MAX_MOTION_LEN', cfg.DATASET.SAMPLER.MAX_LEN)
        min_motion_length = getattr(dataset_cfg, 'MIN_MOTION_LEN', cfg.DATASET.SAMPLER.MIN_LEN)
        unit_length = getattr(dataset_cfg, 'UNIT_LEN', 4)
        fps = getattr(dataset_cfg, 'FPS', 20)

        dataset = UniMotionDataModule(
            cfg=cfg,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            collate_fn=mld_collate_with_source,
            phase=phase,
            mean_humanml3d=mean_humanml3d,
            std_humanml3d=std_humanml3d,
            motion_dir_humanml3d=pjoin(data_root_humanml3d, "new_joint_vecs"),
            split_root_humanml3d=cfg.DATASET.UNIMOTION.HUMANML3D.SPLIT_ROOT,
            mean_motionfix=mean_motionfix,
            std_motionfix=std_motionfix,
            motion_dir_motionfix=pjoin(data_root_motionfix, "new_joint_vecs"),
            split_root_motionfix=cfg.DATASET.UNIMOTION.MOTIONFIX.SPLIT_ROOT,
            mean_h36m=mean_h36m,
            std_h36m=std_h36m,
            motion_dir_h36m=pjoin(data_root_h36m, "new_joint_vecs"),
            split_root_h36m=cfg.DATASET.UNIMOTION.H36M.SPLIT_ROOT,
            max_motion_length=max_motion_length,
            min_motion_length=min_motion_length,
            unit_length=unit_length,
            fps=fps,
        )
        datasets.append(dataset)
        if logger:
            logger.info("Created UniMotion dataset: HumanML3D + MotionFix + H36M")

    cfg.DATASET.NFEATS = datasets[0].nfeats
    cfg.DATASET.NJOINTS = datasets[0].njoints
    return datasets
