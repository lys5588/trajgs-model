"""Camera contracts with a deterministic test double; no pretrained claims.

Real DINOv3 architecture/pretrained integration runs use smoke_camera_encoders.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from torch import nn

from cma_vae.models.architectures.camera_cma_vae import CameraCMAVAE
from cma_vae.models.architectures.camera_vision import FrozenDINOv3, crop_bbox_images

torch.set_num_threads(1)


class RecordingBackbone(nn.Module):
    """Test double exposes frame colors directly as 768-dimensional features."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.inputs = []

    def forward_features(self, pixels):
        self.inputs.append(pixels.detach().clone())
        return {"x_norm_clstoken": pixels.mean((-2, -1)).repeat(1, 256) * self.scale}


def visual_inputs(batch=2):
    images = torch.zeros(batch, 20, 3, 6, 8)
    for b in range(batch):
        for t in range(20):
            # A colored target on a black background, unique per frame/batch.
            images[b, t, :, 1:5, 2:7] = torch.tensor([
                (t + 1) / 21, (b + 1) / (batch + 1), 0.4,
            ])[:, None, None]
    boxes = torch.tensor([2., 1., 7., 5.]).repeat(batch, 20, 1)
    return images, boxes


class CropTests(unittest.TestCase):
    def test_region_frame_order_uint8_and_normalization(self):
        images, boxes = visual_inputs()
        crops = crop_bbox_images(images, boxes)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        rgb = crops * std + mean
        self.assertEqual(rgb.shape, (2, 20, 3, 224, 224))
        for b in range(2):
            for t in range(20):
                target = torch.tensor([(t + 1) / 21, (b + 1) / 3, 0.4])
                torch.testing.assert_close(rgb[b, t], target[:, None, None].expand(3, 224, 224))
        uint8 = (images * 255).round().to(torch.uint8)
        torch.testing.assert_close(crop_bbox_images(uint8, boxes),
                                   crop_bbox_images(uint8.float() / 255, boxes))

    def test_fractional_rounding_clipping_and_stretch(self):
        images = torch.zeros(1, 20, 3, 2, 4)
        images[..., 0] = 0.2
        images[..., -1] = 0.8
        boxes = torch.tensor([-0.2, -1., 3.1, 2.4]).repeat(1, 20, 1)
        crops = crop_bbox_images(images, boxes)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        torch.testing.assert_close(crops[0, 0, :, 0, 0] * std + mean, torch.full((3,), 0.2))
        torch.testing.assert_close(crops[0, 0, :, -1, -1] * std + mean, torch.full((3,), 0.8))
        # Both original rows reach both output edges: no letterbox padding.
        torch.testing.assert_close(crops[..., 0, :], crops[..., -1, :])

    def test_invalid_boxes_and_incomplete_frames(self):
        images, boxes = visual_inputs(1)
        for box in ([3., 1., 2., 4.], [2., 1., 2., 4.],
                    [20., 1., 21., 4.], [float("nan"), 1., 5., 4.]):
            invalid = boxes.clone()
            invalid[0, 7] = torch.tensor(box)
            with self.subTest(box=box), self.assertRaises(ValueError):
                crop_bbox_images(images, invalid)
        with self.assertRaises(ValueError):
            crop_bbox_images(images[:, :19], boxes[:, :19])
        with self.assertRaises(ValueError):
            crop_bbox_images(images, boxes[:, :19])

    def test_invalid_image_values(self):
        images, boxes = visual_inputs(1)
        for value in (float("nan"), -1., 255.):
            bad = images.clone()
            bad[0, 4, 0, 2, 3] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                crop_bbox_images(bad, boxes)
        with self.assertRaises(TypeError):
            crop_bbox_images(images.long(), boxes)


class EncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {d: CameraCMAVAE(camera_dim=d, vision_backbone=RecordingBackbone(),
                                     dinov3_chunk_size=7) for d in (9, 11)}

    def setUp(self):
        torch.manual_seed(122)
        for model in self.models.values():
            model.eval()
            model.zero_grad(set_to_none=True)
            model.vision_backbone.backbone.inputs.clear()

    def test_both_representations_encoders_and_decoder(self):
        images, boxes = visual_inputs()
        for dim, model in self.models.items():
            with self.subTest(camera_dim=dim), torch.no_grad():
                cameras = torch.randn(2, 20, dim)
                for latent, q in (model.encode_camera(cameras), model.encode_joint(cameras, images, boxes)):
                    self.assertEqual(latent.shape, (20, 2, 256))
                    self.assertEqual(q.loc.shape, latent.shape)
                    self.assertTrue(torch.isfinite(latent).all())
                    self.assertTrue(torch.isfinite(q.scale).all() and (q.scale > 0).all())
                    decoded = model.decode(latent)
                    self.assertEqual(decoded.shape, (2, 20, dim))
                    self.assertTrue(torch.isfinite(decoded).all())

    def test_temporal_alignment_before_joint_attention(self):
        model = self.models[11]
        images, boxes = visual_inputs()
        cameras = torch.randn(2, 20, 11)
        captured = []
        hook = model.joint_encoder.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().clone()))
        try:
            with torch.no_grad():
                model.encode_joint(cameras, images, boxes)
                changed = images.clone()
                changed[1, 13, :, 1:5, 2:7] = 0.9
                model.encode_joint(cameras, changed, boxes)
        finally:
            hook.remove()
        self.assertEqual(captured[0].shape, (20, 2, 512))
        torch.testing.assert_close(captured[0][..., :256], captured[1][..., :256])
        changed_positions = (captured[0][..., 256:] - captured[1][..., 256:]).abs().sum(-1) > 1e-6
        self.assertEqual(changed_positions.nonzero().tolist(), [[13, 1]])
        # Distinct frames were not replaced by a broadcast of frame zero.
        self.assertFalse(torch.equal(captured[0][0, 0, 256:], captured[0][1, 0, 256:]))
        calls = model.vision_backbone.backbone.inputs
        self.assertTrue(all(len(c) <= 7 for c in calls))

    def test_mixed_batch_index_mapping_and_no_decode(self):
        model = self.models[11]
        cameras = torch.randn(3, 20, 11)
        images, boxes = visual_inputs(3)
        # Inactive rows are placeholders; they must never reach the cropper.
        images[1] = float("nan")
        boxes[1] = float("nan")
        with patch.object(model, "decode", side_effect=AssertionError("forward must encode only")):
            with torch.no_grad():
                out = model(cameras, images, boxes, torch.tensor([True, False, True]))
        self.assertEqual(out["joint_indices"].tolist(), [0, 2])
        self.assertEqual(out["latent_joint"].shape, (20, 2, 256))
        torch.testing.assert_close(out["latent"][:, [0, 2]], out["latent_joint"])
        torch.testing.assert_close(out["latent"][:, 1], out["latent_camera"][:, 1])
        self.assertEqual(sum(len(x) for x in model.vision_backbone.backbone.inputs), 40)

    def test_no_images_does_not_load_weights(self):
        model = self.models[11]
        with patch.object(model.vision_backbone, "forward", side_effect=AssertionError("no image path")):
            with torch.no_grad():
                out = model(torch.zeros(2, 20, 11))
        self.assertIsNone(out["dist_joint"])
        self.assertIsNone(out["latent_joint"])
        self.assertEqual(out["joint_indices"].numel(), 0)
        torch.testing.assert_close(out["latent"], out["latent_camera"])

    def test_frozen_backbone_and_trainable_gradients(self):
        for dim, model in self.models.items():
            with self.subTest(camera_dim=dim):
                model.train()
                cameras = torch.randn(2, 20, dim, requires_grad=True)
                images, boxes = visual_inputs()
                out = model(cameras, images, boxes)
                loss = sum(q.loc.square().mean() + q.scale.log().square().mean()
                           for q in (out["dist_camera"], out["dist_joint"]))
                loss.backward()
                for name in ("camera_embedding", "camera_pos_encoder", "camera_encoder",
                             "vision_projection", "joint_encoder", "camera_dist_layer", "joint_dist_layer"):
                    grads = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
                    self.assertTrue(grads, name)
                    self.assertTrue(all(torch.isfinite(g).all() for g in grads), name)
                    self.assertTrue(any(g.abs().sum() > 0 for g in grads), name)
                self.assertTrue(torch.isfinite(cameras.grad).all())
                self.assertFalse(model.vision_backbone.backbone.training)
                self.assertTrue(all(not p.requires_grad and p.grad is None
                                    for p in model.vision_backbone.parameters()))

    def test_bad_camera_and_partial_visual_inputs(self):
        model = self.models[11]
        images, boxes = visual_inputs()
        for cameras in (torch.zeros(2, 20, 9), torch.zeros(2, 19, 11),
                        torch.full((2, 20, 11), float("nan"))):
            with self.subTest(shape=cameras.shape), self.assertRaises(ValueError):
                model.encode_camera(cameras)
        cameras = torch.zeros(2, 20, 11)
        with self.assertRaises(ValueError):
            model(cameras, images, None)
        with self.assertRaises(ValueError):
            model(cameras, has_images=torch.ones(2, dtype=torch.bool))
        with self.assertRaises(ValueError):
            model(cameras, images, boxes, torch.ones(2, 20, dtype=torch.bool))
        with self.assertRaises(ValueError):
            model.encode_joint(cameras, images[:, :19], boxes[:, :19])
        invalid = boxes.clone()
        invalid[1, 10] = 0
        with self.assertRaises(ValueError):
            model(cameras, images, invalid)

    def test_reject_incompatible_configuration_and_latent(self):
        for config in ({"camera_dim": 10}, {"sequence_length": 10}, {"downsample_rate": 4},
                       {"freeze_dinov3": False}, {"crop_size": 256}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                CameraCMAVAE(**config)
        with self.assertRaises(ValueError):
            self.models[11].decode(torch.zeros(5, 2, 256))


class BackboneLoadingTests(unittest.TestCase):
    def test_missing_weights_fail_without_random_fallback(self):
        with TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / 'hubconf.py').touch()
            backbone = FrozenDINOv3(repo_path=repo, weights_path=repo / 'missing.pth')
            with patch('torch.hub.load') as loader:
                with self.assertRaisesRegex(FileNotFoundError, 'No random fallback'):
                    backbone.load_backbone()
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
