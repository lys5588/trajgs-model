"""
Paper: CMA-VAE (Cross-Modal Aligned Motion VAE).

HRNet backbone wrapper.

Provides the HRNet vision backbone for the vision-fused joint encoder.
The HRNet implementation (`pose_hrnet.py`) is bundled in this directory;
an external copy can also be supplied via the HRNET_EXTERNAL_PATH
environment variable. If neither is importable, a dummy backbone is used.

NOTE: The HRNet vision backbone is ONLY required for CMA-VAE alignment
training. Motion encode/decode at inference never touches it, so this
module can be left unconfigured.

Usage:
    from models.architectures.hrnet_wrapper import get_hrnet_backbone

    hrnet = get_hrnet_backbone(
        config=hrnet_config,
        checkpoint_path="path/to/pose_hrnet_w32_256x192.pth",
        freeze=True
    )
"""

import os
import sys
import logging
from typing import Optional, List, Union

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

# Optional override: directory containing an importable `pose_hrnet.py`
# module. Normally unnecessary — the bundled pose_hrnet.py in this
# directory is used by default.
HRNET_EXTERNAL_PATH = os.environ.get('HRNET_EXTERNAL_PATH', None)

# Default HRNet configuration
DEFAULT_HRNET_CONFIG = {
    'NUM_JOINTS': 17,
    'PRETRAINED_LAYERS': ['*'],
    'STEM_INPLANES': 64,
    'FINAL_CONV_KERNEL': 1,
    'STAGE2': {
        'NUM_MODULES': 1,
        'NUM_BRANCHES': 2,
        'NUM_BLOCKS': [4, 4],
        'NUM_CHANNELS': [32, 64],
        'BLOCK': 'BASIC',
        'FUSE_METHOD': 'SUM',
    },
    'STAGE3': {
        'NUM_MODULES': 4,
        'NUM_BRANCHES': 3,
        'NUM_BLOCKS': [4, 4, 4],
        'NUM_CHANNELS': [32, 64, 128],
        'BLOCK': 'BASIC',
        'FUSE_METHOD': 'SUM',
    },
    'STAGE4': {
        'NUM_MODULES': 3,
        'NUM_BRANCHES': 4,
        'NUM_BLOCKS': [4, 4, 4, 4],
        'NUM_CHANNELS': [32, 64, 128, 256],
        'BLOCK': 'BASIC',
        'FUSE_METHOD': 'SUM',
    },
}


def _import_hrnet():
    """
    Import the HRNet implementation.

    Resolution order:
    1. the bundled `pose_hrnet.py` next to this file (default),
    2. a directory given via the HRNET_EXTERNAL_PATH environment variable,
    3. a plain `import pose_hrnet` from the current sys.path.

    Returns:
        pose_hrnet module or None
    """
    # 1. Bundled implementation (shipped with this package)
    try:
        from . import pose_hrnet
        return pose_hrnet
    except ImportError:
        pass

    # 2. Optional external override
    if HRNET_EXTERNAL_PATH is not None:
        try:
            # Temporarily add the external path
            if HRNET_EXTERNAL_PATH not in sys.path:
                sys.path.insert(0, HRNET_EXTERNAL_PATH)

            import pose_hrnet

            # Remove the temporary path
            if HRNET_EXTERNAL_PATH in sys.path:
                sys.path.remove(HRNET_EXTERNAL_PATH)

            logger.info("Successfully imported HRNet from external path")
            return pose_hrnet

        except ImportError as e:
            logger.warning(f"Failed to import HRNet from HRNET_EXTERNAL_PATH: {e}")
            if HRNET_EXTERNAL_PATH in sys.path:
                sys.path.remove(HRNET_EXTERNAL_PATH)

    # 3. Plain import (e.g. when running with cwd == cma_vae/models/architectures)
    try:
        import pose_hrnet
        return pose_hrnet
    except ImportError as e:
        logger.warning(
            f"Failed to import HRNet: {e}. The HRNet vision backbone is "
            "unavailable; it is only required for CMA-VAE alignment "
            "training — inference does not need it."
        )
        return None


def get_hrnet_backbone(
    config=None,
    checkpoint_path: Optional[str] = None,
    freeze: bool = True,
    device: str = 'cpu',
):
    """
    Get the HRNet backbone.

    Args:
        config: HRNet config object (easydict); if None, the default config is used
        checkpoint_path: path to pretrained weights
        freeze: whether to freeze the weights
        device: device

    Returns:
        HRNet model or None
    """
    pose_hrnet = _import_hrnet()

    if pose_hrnet is None:
        logger.warning("HRNet not available. Vision guidance will be disabled.")
        return None

    # Build the config object
    if config is None:
        from easydict import EasyDict as edict
        config = edict(DEFAULT_HRNET_CONFIG)

    # Create the model
    try:
        model = pose_hrnet.get_pose_net(config)
        logger.info("Created HRNet model")
    except Exception as e:
        logger.error(f"Failed to create HRNet model: {e}")
        return None

    # Load pretrained weights
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            # Handle possible state_dict format differences
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded HRNet checkpoint from {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to load HRNet checkpoint: {e}")
    elif checkpoint_path is not None:
        logger.warning(f"HRNet checkpoint not found: {checkpoint_path}")

    # Freeze weights
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        logger.info("HRNet weights frozen")

    model = model.to(device)
    return model


class DummyHRNet(nn.Module):
    """
    Placeholder model used when HRNet is unavailable.

    Outputs all-zero features so the model can run without vision guidance.
    """

    def __init__(self, output_channels: List[int] = [32, 64, 128, 256]):
        super().__init__()
        self.output_channels = output_channels

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Args:
            x: [B, 3, H, W] input images

        Returns:
            list of features, each with shape [B, C, H/stride, W/stride]
        """
        B, _, H, W = x.shape
        device = x.device
        dtype = x.dtype

        features = []
        for i, c in enumerate(self.output_channels):
            stride = 2 ** (i + 2)  # 4, 8, 16, 32
            h = H // stride
            w = W // stride
            feat = torch.zeros(B, c, h, w, device=device, dtype=dtype)
            features.append(feat)

        return features


class HRNetFeatureExtractor(nn.Module):
    """
    HRNet feature extractor.

    Wraps the HRNet backbone with a more convenient interface.
    Supports selecting output levels and automatic weight freezing.

    Follows the MTVCrafter implementation, supporting:
    - single-level output: output_level = 3 (returns [B, 256, H/32, W/32])
    - multi-level output: output_level = [0,1,2,3] (returns a list of features for multi-scale fusion)
    """

    # ImageNet normalization parameters
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        config=None,
        checkpoint_path: Optional[str] = None,
        freeze: bool = True,
        output_level: Union[int, List[int]] = 3,  # int or list of int
    ):
        """
        Args:
            config: HRNet configuration
            checkpoint_path: path to pretrained weights
            freeze: whether to freeze the weights
            output_level: output level
                - int: single level (0-3)
                - list: multiple levels [0,1,2,3], returns concatenated features
        """
        super().__init__()

        self.output_level = output_level
        self.freeze = freeze

        # Register normalization parameters as buffers
        self.register_buffer('mean', torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1))

        # Try to load the real HRNet
        self.backbone = get_hrnet_backbone(config, checkpoint_path, freeze)

        if self.backbone is None:
            logger.warning("Using DummyHRNet as fallback")
            self.backbone = DummyHRNet()
            self.is_dummy = True
        else:
            self.is_dummy = False

        # Compute the number of output channels
        # HRNet STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
        if isinstance(output_level, int):
            self.output_channels = [32, 64, 128, 256][output_level]
        elif isinstance(output_level, list):
            # Multi-scale output: channel count is the sum over selected levels
            channels_list = [32, 64, 128, 256]
            self.output_channels = sum(channels_list[i] for i in output_level)
        else:
            raise ValueError("output_level should be int or list of int")

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, 3, H, W] input images with values in [0, 1]

        Returns:
            features: [B, C, H', W'] feature maps
                - if output_level is an int: returns single-level features
                - if output_level is a list: returns concatenated features (upsampling to a common scale is required)
        """
        # Make sure the input is in [0, 1]
        if x.max() > 1.0:
            x = x / 255.0

        # ImageNet normalization
        x = (x - self.mean) / self.std

        # Forward pass
        if self.freeze and not self.is_dummy:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)

        # Handle the output
        if isinstance(features, (list, tuple)):
            if isinstance(self.output_level, int):
                # Single-level output
                return features[self.output_level]
            elif isinstance(self.output_level, list):
                # Multi-level output (following MTVCrafter)
                # Return the selected levels
                selected_features = [features[i] for i in self.output_level]
                # Note: a list is returned here; upsampling and concatenation are handled by the caller
                return selected_features
        return features

    def train(self, mode: bool = True):
        """Override train() so a frozen backbone always stays in eval mode."""
        super().train(mode)
        if self.freeze and not self.is_dummy:
            self.backbone.eval()
        return self


def create_hrnet_config():
    """
    Create the HRNet config object.

    Returns a config compatible with MTVCrafter/config/vision_backbone.py.

    Returns:
        easydict config object
    """
    from easydict import EasyDict as edict

    config = edict()
    config.model = edict()
    config.model.image_shape = [192, 256]
    config.model.init_weights = True
    config.model.checkpoint = None

    config.model.backbone = edict()
    config.model.backbone.type = 'hrnet_32'
    config.model.backbone.num_final_layer_channel = 17
    config.model.backbone.num_joints = 17
    config.model.backbone.num_layers = 152
    config.model.backbone.init_weights = True
    config.model.backbone.fix_weights = False

    # Checkpoint path is supplied by the caller
    # (model.vae_config.hrnet_checkpoint); only needed for CMA-VAE
    # alignment training.
    config.model.backbone.checkpoint = None

    # HRNet network structure configuration
    config.model.backbone.NUM_JOINTS = 17
    config.model.backbone.PRETRAINED_LAYERS = ['*']
    config.model.backbone.STEM_INPLANES = 64
    config.model.backbone.FINAL_CONV_KERNEL = 1

    # Stage 2
    config.model.backbone.STAGE2 = edict()
    config.model.backbone.STAGE2.NUM_MODULES = 1
    config.model.backbone.STAGE2.NUM_BRANCHES = 2
    config.model.backbone.STAGE2.NUM_BLOCKS = [4, 4]
    config.model.backbone.STAGE2.NUM_CHANNELS = [32, 64]
    config.model.backbone.STAGE2.BLOCK = 'BASIC'
    config.model.backbone.STAGE2.FUSE_METHOD = 'SUM'

    # Stage 3
    config.model.backbone.STAGE3 = edict()
    config.model.backbone.STAGE3.NUM_MODULES = 4
    config.model.backbone.STAGE3.NUM_BRANCHES = 3
    config.model.backbone.STAGE3.NUM_BLOCKS = [4, 4, 4]
    config.model.backbone.STAGE3.NUM_CHANNELS = [32, 64, 128]
    config.model.backbone.STAGE3.BLOCK = 'BASIC'
    config.model.backbone.STAGE3.FUSE_METHOD = 'SUM'

    # Stage 4
    config.model.backbone.STAGE4 = edict()
    config.model.backbone.STAGE4.NUM_MODULES = 3
    config.model.backbone.STAGE4.NUM_BRANCHES = 4
    config.model.backbone.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
    config.model.backbone.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
    config.model.backbone.STAGE4.BLOCK = 'BASIC'
    config.model.backbone.STAGE4.FUSE_METHOD = 'SUM'

    return config
