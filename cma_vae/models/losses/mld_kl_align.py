"""
Losses for CMA-VAE (paper: DPA, Dual-Posterior KL Alignment):
- reconstruction loss on motion features and recovered 3D joints
- KL regularization towards the standard normal prior (both encoders)
- KL alignment loss pulling the motion-only encoder towards the
  vision-fused joint encoder (with linear warm-up)

Historical code name: "KL-Align" losses / `MLDLossesKLAlign`. The file name
is kept for checkpoint compatibility.
"""

import torch
import torch.nn as nn
from torchmetrics import Metric


class KLLoss(nn.Module):
    """KL(q(z|x) || p(z)) with p(z) = N(0, I) unless a prior is given."""

    def __init__(self):
        super().__init__()

    def forward(self, q, p=None):
        if p is None:
            mu = q.mean
            logvar = 2 * torch.log(q.stddev + 1e-8)
            return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return torch.distributions.kl_divergence(q, p).mean()


class KLAlignLoss(nn.Module):
    """
    KL(q(z|motion) || q(z|motion, image).detach())

    The image-guided joint distribution serves as a fixed target; only the
    motion-only encoder receives gradients.
    """

    def __init__(self):
        super().__init__()

    def forward(self, dist_motion, dist_joint):
        mu_motion = dist_motion.mean
        logvar_motion = 2 * torch.log(dist_motion.stddev + 1e-8)

        mu_joint = dist_joint.mean.detach()
        logvar_joint = 2 * torch.log(dist_joint.stddev.detach() + 1e-8)

        kl = 0.5 * torch.mean(
            logvar_joint - logvar_motion +
            (logvar_motion.exp() + (mu_motion - mu_joint).pow(2)) / logvar_joint.exp() - 1
        )
        return kl


class CMAVAELosses(Metric):
    """CMA-VAE training losses (paper: reconstruction + DPA)."""

    def __init__(self, vae, mode, cfg, kl_align_weight=0.1):
        super().__init__(dist_sync_on_step=cfg.LOSS.DIST_SYNC_ON_STEP)

        self.vae_type = cfg.TRAIN.ABLATION.VAE_TYPE
        self.mode = mode
        self.cfg = cfg
        self.stage = cfg.TRAIN.STAGE
        self.kl_align_weight = kl_align_weight

        self.kl_align_warmup_steps = getattr(cfg.LOSS, 'KL_ALIGN_WARMUP_STEPS', 2000)
        self.current_step = 0
        self.use_kl_warmup = getattr(cfg.LOSS, 'USE_KL_WARMUP', True)

        losses = [
            "recons_feature",
            "recons_joints",
            "kl_motion",   # KL of the motion-only encoder
            "kl_joint",    # KL of the image-guided joint encoder (monitoring)
            "kl_align",    # alignment between the two encoders
            "total",
        ]

        for loss in losses:
            self.add_state(loss,
                           default=torch.tensor(0.0),
                           dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")
        self.losses = losses

        self._losses_func = {}
        self._params = {}

        for loss in losses:
            if loss == 'kl_align':
                self._losses_func[loss] = KLAlignLoss()
                self._params[loss] = getattr(cfg.LOSS, 'LAMBDA_KL_ALIGN', 0.1)
            elif loss == 'kl_joint':
                # Monitored separately; weighted together with kl_motion in update()
                self._losses_func[loss] = KLLoss()
                self._params[loss] = 0.0
            elif loss == 'kl_motion':
                if cfg.LOSS.LAMBDA_KL != 0.0:
                    self._losses_func[loss] = KLLoss()
                    self._params[loss] = cfg.LOSS.LAMBDA_KL
            elif loss.startswith('recons'):
                self._losses_func[loss] = torch.nn.SmoothL1Loss(reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_REC
            elif loss != "total":
                raise ValueError(f"Loss {loss} is not recognized.")

            if loss.endswith('joints'):
                self._params[loss] = getattr(cfg.LOSS, 'LAMBDA_JOINT', 1.0)

    def update(self, rs_set):
        # torchmetrics >= 1.0 wraps update() in no_grad and discards its
        # return value, which would silently disable training. The actual
        # loss is therefore computed in compute_loss() below.
        raise RuntimeError("Call compute_loss() instead of update().")

    def compute_loss(self, rs_set):
        """Compute the weighted training loss (differentiable) and accumulate
        the detached values into the metric states for logging."""
        total: float = 0.0

        self.current_step += 1

        total += self._update_loss("recons_feature", rs_set['m_rst'],
                                   rs_set['m_ref'])
        total += self._update_loss("recons_joints", rs_set['joints_rst'],
                                   rs_set['joints_ref'])

        # KL regularization towards N(0, I) for the motion encoder
        if "kl_motion" in self._losses_func:
            kl_motion_loss = self._losses_func["kl_motion"](rs_set['dist_motion'], rs_set['dist_ref'])
            self.kl_motion.__iadd__(kl_motion_loss.detach())
            total += self._params["kl_motion"] * kl_motion_loss

        if rs_set.get('dist_joint', None) is not None:
            # Same KL regularization for the joint encoder, so that both
            # encoders stay close to the prior and the alignment loss does
            # not fight the regularizer.
            kl_joint_loss = self._losses_func["kl_joint"](rs_set['dist_joint'], rs_set['dist_ref'])
            self.kl_joint.__iadd__(kl_joint_loss.detach())
            total += self._params.get("kl_motion", 0.0) * kl_joint_loss

            # Alignment loss with linear warm-up from 0 to full weight.
            # Note: dist_joint covers the whole batch (samples without a
            # paired image contribute zero visual features), so the
            # alignment is computed over the whole batch as well.
            if self.use_kl_warmup and self.current_step < self.kl_align_warmup_steps:
                warmup_factor = self.current_step / self.kl_align_warmup_steps
            else:
                warmup_factor = 1.0

            kl_align_loss = self._losses_func["kl_align"](
                rs_set['dist_motion'],
                rs_set['dist_joint']
            )
            self.kl_align.__iadd__(kl_align_loss.detach())
            total += self._params["kl_align"] * warmup_factor * kl_align_loss

        self.total += total.detach()
        self.count += 1

        # Keep torchmetrics' internal bookkeeping consistent even though the
        # wrapped update() is bypassed: invalidate the compute() cache and
        # mark the metric as updated (attribute names differ across versions).
        self._computed = None
        if hasattr(self, "_update_count"):  # torchmetrics >= 1.0
            self._update_count += 1
        else:                               # torchmetrics < 1.0
            self._update_called = True

        return total

    def compute(self, split):
        count = getattr(self, "count")
        return {loss: getattr(self, loss) / count for loss in self.losses}

    def _update_loss(self, loss: str, outputs, inputs):
        if loss not in self._losses_func:
            return torch.tensor(0.0, device=outputs.device if isinstance(outputs, torch.Tensor) else 'cpu')

        val = self._losses_func[loss](outputs, inputs)
        getattr(self, loss).__iadd__(val.detach())
        return self._params[loss] * val

    def loss2logname(self, loss: str, split: str):
        if loss == "total":
            log_name = f"{loss}/{split}"
        else:
            loss_type, loss_name = loss.split("_")
            log_name = f"{loss_type}/{loss_name}/{split}"
        return log_name


# Backward-compatibility alias: historical checkpoints (and their pickled
# `hyper_parameters`) may reference the original class name. Do not remove.
MLDLossesKLAlign = CMAVAELosses
