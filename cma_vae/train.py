import os

# Relax NCCL timeouts for long multi-GPU training runs
os.environ["NCCL_TIMEOUT"] = "3600000"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

# Prefer the unified `lightning` package (2.x); fall back to legacy pytorch_lightning.
try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
try:
    from lightning.pytorch import loggers as pl_loggers
    from lightning.pytorch.callbacks import ModelCheckpoint
except ImportError:
    from pytorch_lightning import loggers as pl_loggers
    from pytorch_lightning.callbacks import ModelCheckpoint

from callback import ProgressLogger
from config import parse_args
from data.get_data import get_datasets
from models.get_model import get_model
from utils.logger import create_logger


def main():
    # parse options
    cfg = parse_args()  # parse config file

    # create logger
    logger = create_logger(cfg, phase="train")

    # resume
    if cfg.TRAIN.RESUME:
        resume = cfg.TRAIN.RESUME
        backcfg = cfg.TRAIN.copy()
        if os.path.exists(resume):
            file_list = sorted(os.listdir(resume), reverse=True)
            for item in file_list:
                if item.endswith(".yaml"):
                    cfg = OmegaConf.load(os.path.join(resume, item))
                    cfg.TRAIN = backcfg
                    break
            checkpoints = sorted(os.listdir(os.path.join(
                resume, "checkpoints")),
                                 key=lambda x: int(x[6:-5]),
                                 reverse=True)
            for checkpoint in checkpoints:
                if "epoch=" in checkpoint:
                    cfg.TRAIN.PRETRAINED = os.path.join(
                        resume, "checkpoints", checkpoint)
                    break
            if os.path.exists(os.path.join(resume, "wandb")):
                wandb_list = sorted(os.listdir(os.path.join(resume, "wandb")),
                                    reverse=True)
                for item in wandb_list:
                    if "run-" in item:
                        cfg.LOGGER.WANDB.RESUME_ID = item.split("-")[-1]

        else:
            raise ValueError("Resume path is not right.")
    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # tensorboard logger and wandb logger
    loggers = []
    if cfg.LOGGER.WANDB.PROJECT:
        wandb_logger = pl_loggers.WandbLogger(
            project=cfg.LOGGER.WANDB.PROJECT,
            offline=cfg.LOGGER.WANDB.OFFLINE,
            id=cfg.LOGGER.WANDB.RESUME_ID,
            save_dir=cfg.FOLDER_EXP,
            version="",
            name=cfg.NAME,
            anonymous=False,
            log_model=False,
        )
        loggers.append(wandb_logger)
    if cfg.LOGGER.TENSORBOARD:
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=cfg.FOLDER_EXP,
                                                 sub_dir="tensorboard",
                                                 version="",
                                                 name="")
        loggers.append(tb_logger)
    logger.info(OmegaConf.to_yaml(cfg))

    # create dataset
    datasets = get_datasets(cfg, logger=logger)
    logger.info("datasets module {} initialized".format("".join(
        cfg.TRAIN.DATASETS)))

    # create model
    model = get_model(cfg, datasets[0])
    logger.info("model {} loaded".format(cfg.model.model_type))

    # metrics reported by the progress logger at each epoch end
    metric_monitor = {
        "Rec_feature": "train/recons_feature",
        "Rec_joints": "train/recons_joints",
        "KL_motion": "train/kl_motion",
        "KL_align": "train/kl_align",
        "Total": "train/total",
    }

    # callbacks
    callbacks = [
        ProgressLogger(metric_monitor=metric_monitor),
        ModelCheckpoint(
            dirpath=os.path.join(cfg.FOLDER_EXP, "checkpoints"),
            filename="{epoch}",
            monitor="step",
            mode="max",
            every_n_epochs=cfg.LOGGER.SAVE_CHECKPOINT_EPOCH,
            save_top_k=-1,
            save_last=False,
            save_on_train_epoch_end=True,
        ),
    ]
    logger.info("Callbacks initialized")

    if len(cfg.DEVICE) > 1:
        # Frozen HRNet parameters are unused on batches without images
        ddp_strategy = "ddp_find_unused_parameters_true"
    else:
        ddp_strategy = "auto"

    # Pin the single-node cluster environment explicitly. Lightning's automatic
    # cluster detection imports mpi4py, which crashes (RuntimeError, not
    # ImportError) on machines with a broken MPI install — common on shared
    # workstations. Training here is single-node (Lightning spawns DDP itself).
    plugins = None
    try:
        from lightning.pytorch.plugins.environments import LightningEnvironment
        plugins = [LightningEnvironment()]
    except ImportError:
        try:
            from pytorch_lightning.plugins.environments import LightningEnvironment
            plugins = [LightningEnvironment()]
        except ImportError:
            pass

    # trainer
    trainer = pl.Trainer(
        benchmark=False,
        max_epochs=cfg.TRAIN.END_EPOCH,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        strategy=ddp_strategy,
        plugins=plugins,
        default_root_dir=cfg.FOLDER_EXP,
        log_every_n_steps=cfg.LOGGER.LOG_EVERY_STEPS,
        deterministic=False,
        detect_anomaly=False,
        enable_progress_bar=True,
        logger=loggers,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.LOGGER.VAL_EVERY_STEPS,
    )
    logger.info("Trainer initialized")

    # strict load vae model
    if cfg.TRAIN.PRETRAINED_VAE:
        logger.info("Loading pretrain vae from {}".format(
            cfg.TRAIN.PRETRAINED_VAE))
        state_dict = torch.load(cfg.TRAIN.PRETRAINED_VAE,
                                map_location="cpu",
                                weights_only=False)["state_dict"]
        # Extract encoder/decoder weights. The frozen HRNet backbone is
        # created lazily and initialized from its own pretrained file, so
        # its weights are skipped here.
        from collections import OrderedDict
        vae_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("vae.") and not k.startswith("vae.vision_backbone."):
                vae_dict[k[len("vae."):]] = v
        missing, unexpected = model.vae.load_state_dict(vae_dict, strict=False)
        missing = [k for k in missing if not k.startswith("vision_backbone.")]
        assert not missing, f"Missing keys: {missing}"
        assert not unexpected, f"Unexpected keys: {unexpected}"

    if cfg.TRAIN.PRETRAINED:
        logger.info("Loading pretrained model from {}".format(
            cfg.TRAIN.PRETRAINED))
        state_dict = torch.load(cfg.TRAIN.PRETRAINED,
                                map_location="cpu",
                                weights_only=False)["state_dict"]
        model.load_state_dict(state_dict, strict=False)

    # fitting
    if cfg.TRAIN.RESUME:
        trainer.fit(model,
                    datamodule=datasets[0],
                    ckpt_path=cfg.TRAIN.PRETRAINED)
    else:
        trainer.fit(model, datamodule=datasets[0])

    # checkpoint
    checkpoint_folder = trainer.checkpoint_callback.dirpath
    logger.info(f"The checkpoints are stored in {checkpoint_folder}")
    logger.info(
        f"The outputs of this experiment are stored in {cfg.FOLDER_EXP}")

    # end
    logger.info("Training ends!")


if __name__ == "__main__":
    main()
