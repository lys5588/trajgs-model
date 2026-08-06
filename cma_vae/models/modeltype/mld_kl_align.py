"""
Lightning wrapper for CMA-VAE (Cross-Modal Aligned Motion VAE).

Paper: CMA-VAE, trained with DPA (Dual-Posterior KL Alignment); historical
code name "KL-Align" VAE (JMVAE-KL architecture). The file name and the
`MLD_KL_ALIGN` wrapper class name are kept for checkpoint compatibility
(`model_type: mld_kl_align` in the config resolves to this class).

The model consists of a Motion Encoder, a vision-fused Joint Encoder and a
shared Decoder. Unimodal encoders are aligned to the joint-modality encoder
via KL divergence. Samples with and without paired images are handled
transparently.
"""
import torch
from torch.optim import AdamW
from torchmetrics import MetricCollection
try:
    from lightning.pytorch import LightningModule
except ImportError:
    from pytorch_lightning import LightningModule

from models.architectures.mld_vae_kl_align import CMAVAE
from models.losses.mld_kl_align import CMAVAELosses


class MLD_KL_ALIGN(LightningModule):

    def __init__(self, cfg, datamodule, **kwargs):
        super().__init__()

        self.cfg = cfg
        self.stage = cfg.TRAIN.STAGE
        self.is_vae = cfg.model.vae
        self.nfeats = cfg.DATASET.NFEATS
        self.njoints = cfg.DATASET.NJOINTS
        self.debug = cfg.DEBUG
        self.latent_dim = cfg.model.latent_dim
        self.datamodule = datamodule
        self.vae_type = "kl_align"

        vae_config = cfg.model.get('vae_config', {})

        self.vae = CMAVAE(
            ablation=cfg.TRAIN.ABLATION,
            nfeats=self.nfeats,
            latent_dim=self.latent_dim,
            ff_size=cfg.model.ff_size,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_head,
            dropout=cfg.model.droupout,
            activation=cfg.model.activation,
            normalize_before=vae_config.get('normalize_before', False),
            position_embedding=vae_config.get('position_embedding', 'learned'),
            arch=vae_config.get('arch', 'all_encoder'),
            downsample_rate=vae_config.get('downsample_rate', 1),
            vision_dim=vae_config.get('vision_dim', 128),
            hrnet_channels=vae_config.get('hrnet_channels', 256),
            hrnet_checkpoint=vae_config.get('hrnet_checkpoint', None),
            freeze_hrnet=vae_config.get('freeze_hrnet', True),
            num_joints=vae_config.get('num_joints', 22),
            img_size=vae_config.get('img_size', (448, 448)),
            kl_align_weight=vae_config.get('kl_align_weight', 0.1),
        )

        self._losses = MetricCollection({
            split: CMAVAELosses(
                vae=self.is_vae,
                mode="xyz",
                cfg=cfg,
                kl_align_weight=vae_config.get('kl_align_weight', 0.1),
            )
            for split in ["losses_train", "losses_test", "losses_val"]
        })
        self.losses = {
            key: self._losses["losses_" + key]
            for key in ["train", "test", "val"]
        }

        self.feats2joints = datamodule.feats2joints
        self._datamodule_feats2joints = datamodule.feats2joints
        self._is_combined_dataset = datamodule.name in ["motionx_combined", "unimotion"]

        self._test_step_outputs = []

    def configure_optimizers(self):
        optimizer = AdamW(
            lr=self.cfg.TRAIN.OPTIM.LR,
            params=self.parameters(),
            betas=(0.9, 0.99),
            weight_decay=getattr(self.cfg.TRAIN, 'WEIGHT_DECAY', 0.0),
        )
        return optimizer

    def forward(self, batch):
        """
        Args:
            batch: dict with keys:
                - motion: [B, T, D]
                - length: [B]
                - video_rgb: [B, 1, H, W, 3] (optional)
                - has_video: [B] (optional)
                - joints_2d_normalized: [B, 1, J, 2] (optional)
        """
        return self.vae(
            batch["motion"],
            batch["length"],
            video_rgb=batch.get("video_rgb", None),
            has_video=batch.get("has_video", None),
            joints_2d_precomputed=batch.get("joints_2d_normalized", None),
        )

    def allsplit_step(self, split: str, batch, batch_idx):
        forward_dict = self.forward(batch)

        feats_ref = batch["motion"]
        feats_rst = forward_dict["motion_recon"]
        has_video = batch.get("has_video", None)
        dataset_source = batch.get("dataset_source", None)

        if self._is_combined_dataset:
            joints_rst = self._datamodule_feats2joints(feats_rst, dataset_source)
            joints_ref = self._datamodule_feats2joints(feats_ref, dataset_source)
        else:
            joints_rst = self._datamodule_feats2joints(feats_rst)
            joints_ref = self._datamodule_feats2joints(feats_ref)

        # Standard normal prior as the KL reference distribution
        dist_motion = forward_dict["dist_motion"]
        dist_ref = torch.distributions.Normal(
            torch.zeros_like(dist_motion.loc),
            torch.ones_like(dist_motion.scale),
        )

        min_len = min(feats_ref.shape[1], feats_rst.shape[1])

        rs_set = {
            "m_ref": feats_ref[:, :min_len, :],
            "m_rst": feats_rst[:, :min_len, :],
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "dist_motion": forward_dict["dist_motion"],
            "dist_joint": forward_dict.get("dist_joint", None),
            "dist_ref": dist_ref,
            "has_video": has_video,
        }

        # compute_loss() (not the torchmetrics update()) keeps the returned
        # loss differentiable on torchmetrics >= 1.0
        total_loss = self.losses[split].compute_loss(rs_set)

        if split == "train":
            if batch_idx % self.cfg.LOGGER.LOG_EVERY_STEPS == 0:
                loss_dict = self.losses[split].compute(split)
                for loss_name, loss_val in loss_dict.items():
                    self.log(f"{split}/{loss_name}", loss_val)
        else:
            loss_dict = self.losses[split].compute(split)
            for loss_name, loss_val in loss_dict.items():
                self.log(f"{split}/{loss_name}", loss_val)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self.allsplit_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.allsplit_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        output = self.allsplit_step("test", batch, batch_idx)
        self._test_step_outputs.append(output)
        return output

    def on_load_checkpoint(self, checkpoint):
        # The frozen HRNet backbone is created lazily on the first batch with
        # images; drop its weights so that resuming / fine-tuning works before
        # the backbone module exists (it is rebuilt from its own pretrained
        # file anyway).
        state_dict = checkpoint.get("state_dict", {})
        for key in list(state_dict):
            if key.startswith("vae.vision_backbone."):
                del state_dict[key]

    def on_train_epoch_start(self):
        self.losses["train"].reset()

    def on_validation_epoch_start(self):
        self.losses["val"].reset()

    def on_train_epoch_end(self):
        # ModelCheckpoint(monitor="step") relies on this metric
        self.log_dict({
            "epoch": float(self.trainer.current_epoch),
            "step": float(self.trainer.current_epoch),
        }, sync_dist=True, rank_zero_only=True)
