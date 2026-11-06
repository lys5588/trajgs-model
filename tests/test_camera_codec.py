"""Numerical contracts for camera representations (no model/weight downloads)."""
import importlib.util
import math
from pathlib import Path
import unittest

import torch

from cma_vae.utils.camera import (
    extri_intri_to_pose_encoding as encode,
    pose_encoding_to_extri_intri as decode,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)

FORMATS = ("quaternion_9d", "rotation_6d_11d")
ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def example(device="cpu"):
    # Attachment's column-convention example: C2W Rz(90deg), center (1,2,3).
    c2w = torch.tensor([[0., -1., 0., 1.], [1., 0., 0., 2.],
                        [0., 0., 1., 3.], [0., 0., 0., 1.]], device=device)[None, None]
    w2c = torch.linalg.inv(c2w)
    k = torch.tensor([[640., 0., 320.], [0., 480., 240.], [0., 0., 1.]], device=device)[None, None]
    return w2c, c2w, k


def random_cameras(device="cpu", batch=2, steps=20):
    g = torch.Generator().manual_seed(214)
    rotation, _ = torch.linalg.qr(torch.randn(batch, steps, 3, 3, generator=g))
    rotation[..., :, 2] *= torch.linalg.det(rotation)[..., None]
    c2w = torch.eye(4).expand(batch, steps, 4, 4).clone()
    c2w[..., :3, :3] = rotation
    c2w[..., :3, 3] = torch.randn(batch, steps, 3, generator=g) * 5
    k = torch.eye(3).expand(batch, steps, 3, 3).clone()
    k[..., 0, 0] = torch.rand(batch, steps, generator=g) * 1000 + 100
    k[..., 1, 1] = torch.rand(batch, steps, generator=g) * 800 + 100
    k[..., 0, 2], k[..., 1, 2] = 320, 240
    return torch.linalg.inv(c2w).to(device), c2w.to(device), k.to(device)


class CameraCodecTests(unittest.TestCase):
    def test_attachment_example_and_column_order(self):
        w2c, c2w, k = example()
        pose = encode(w2c, k, (480, 640), pose_encoding_type=FORMATS[1])
        expected = pose.new_tensor([1, 2, 3, 0, 1, 0, -1, 0, 0,
                                    2*math.atan(.5), 2*math.atan(.5)])[None, None]
        torch.testing.assert_close(pose, expected)
        ext, ki = decode(pose, (480, 640))
        torch.testing.assert_close(ext, w2c[..., :3, :])
        torch.testing.assert_close(ki, k)
        ext, _ = decode(pose, (480, 640), extrinsics_type="c2w")
        torch.testing.assert_close(ext, c2w[..., :3, :])

    def test_quaternion_xyzw_and_camera_center(self):
        w2c, _, k = example()
        pose = encode(w2c, k, (480, 640))
        torch.testing.assert_close(pose[..., :7], pose.new_tensor(
            [1, 2, 3, 0, 0, 2**-.5, 2**-.5])[None, None])

    def test_random_batched_roundtrip_both_conventions_and_shapes(self):
        w2c, c2w, k = random_cameras()
        for fmt in FORMATS:
            for convention, original in [("w2c", w2c), ("c2w", c2w)]:
                for rows in [3, 4]:
                    with self.subTest(fmt=fmt, convention=convention, rows=rows):
                        pose = encode(original[..., :rows, :], k, (480, 640),
                                      pose_encoding_type=fmt, extrinsics_type=convention)
                        ext, ki = decode(pose, (480, 640), extrinsics_type=convention)
                        self.assertEqual(pose.shape, (2, 20, 9 if fmt == FORMATS[0] else 11))
                        torch.testing.assert_close(ext, original[..., :3, :], atol=4e-6, rtol=2e-5)
                        torch.testing.assert_close(ki, k, atol=2e-4, rtol=2e-5)

    def test_against_da3_reference(self):
        path = ROOT / "submodules/Depth-Anything-3/src/depth_anything_3/model/utils/transform.py"
        if not path.exists():
            self.skipTest("DA3 submodule not initialized")
        spec = importlib.util.spec_from_file_location("da3_transform_reference", path)
        ref = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref)
        w2c, c2w, k = random_cameras()
        actual = encode(w2c, k, (480, 640))
        expected = ref.extri_intri_to_pose_encoding(c2w, k, (480, 640))
        torch.testing.assert_close(actual, expected, atol=4e-6, rtol=2e-5)
        actual_ext, actual_k = decode(actual, (480, 640), extrinsics_type="c2w")
        expected_ext, expected_k = ref.pose_encoding_to_extri_intri(actual, (480, 640))
        torch.testing.assert_close(actual_ext, expected_ext)
        torch.testing.assert_close(actual_k, expected_k)

    def test_identity_and_half_turn_rotations(self):
        for diagonal in [[1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]]:
            ext = torch.eye(4)[None, None]
            ext[..., :3, :3] = torch.diag(torch.tensor(diagonal))
            k = example()[2]
            for fmt in FORMATS:
                with self.subTest(rotation=diagonal, fmt=fmt):
                    pose = encode(ext, k, (480, 640), pose_encoding_type=fmt)
                    result, _ = decode(pose, (480, 640))
                    torch.testing.assert_close(result, ext[..., :3, :])

    def test_quaternion_sign_and_scale_do_not_change_rotation(self):
        w2c, _, k = random_cameras()
        pose = encode(w2c, k, (480, 640))
        expected, _ = decode(pose, (480, 640))
        for factor in [-1, 3, 1e20]:
            changed = pose.clone()
            changed[..., 3:7] *= factor
            actual, _ = decode(changed, (480, 640))
            torch.testing.assert_close(actual, expected, atol=4e-6, rtol=2e-5)

    def test_gram_schmidt_nonorthogonal_predictions(self):
        axes = torch.tensor([[2., 0, 0, 2, 3, 0], [0., 2, 0, -3, 2, 0]])
        rotations = rotation_6d_to_matrix(axes)
        torch.testing.assert_close(rotations.mT @ rotations, torch.eye(3).expand(2, 3, 3))
        torch.testing.assert_close(torch.linalg.det(rotations), torch.ones(2))
        torch.testing.assert_close(rotations[0], torch.eye(3))
        torch.testing.assert_close(matrix_to_rotation_6d(rotations)[1], torch.tensor([0., 1, 0, -1, 0, 0]))

    def test_nearly_collinear_rotation_stability(self):
        with self.assertRaises(ValueError):
            rotation_6d_to_matrix(torch.tensor([1., 1, 1, 100000, 100000, 100000]))
        rotation = rotation_6d_to_matrix(torch.tensor([1., 1, 1, 1.001, 1, 1]))
        torch.testing.assert_close(rotation.mT @ rotation, torch.eye(3), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(torch.linalg.det(rotation), torch.tensor(1.))

    def test_per_scene_and_per_view_valid_image_sizes(self):
        w2c, _, k = random_cameras()
        for hw in [torch.tensor([480, 640]), torch.tensor([[480, 640], [518, 700]]),
                   torch.arange(80).reshape(2, 20, 2) + 400]:
            for fmt in FORMATS:
                pose = encode(w2c, k, hw, pose_encoding_type=fmt)
                _, actual = decode(pose, hw)
                torch.testing.assert_close(actual[..., [0, 1], [0, 1]], k[..., [0, 1], [0, 1]])
                expected_hw = hw.expand(2, 20, 2) if hw.ndim == 1 else hw[:, None].expand(2, 20, 2) if hw.ndim == 2 else hw
                torch.testing.assert_close(actual[..., 0, 2], expected_hw[..., 1].float()/2)
                torch.testing.assert_close(actual[..., 1, 2], expected_hw[..., 0].float()/2)

    def test_intrinsics_information_loss_is_explicit(self):
        w2c, _, k = example()
        k[..., 0, 2], k[..., 1, 2], k[..., 0, 1] = 310, 220, 7
        for fmt in FORMATS:
            _, actual = decode(encode(w2c, k, (480, 640), pose_encoding_type=fmt), (480, 640))
            self.assertEqual(actual[0, 0, 0, 2], 320)
            self.assertEqual(actual[0, 0, 1, 2], 240)
            self.assertEqual(actual[0, 0, 0, 1], 0)
            torch.testing.assert_close(actual[..., [0, 1], [0, 1]], k[..., [0, 1], [0, 1]])

    def test_geometry_gradients_reach_pose_and_intrinsics(self):
        for fmt in FORMATS:
            w2c, _, k = random_cameras()
            w2c.requires_grad_(); k.requires_grad_()
            pose = encode(w2c, k, (480, 640), pose_encoding_type=fmt)
            pose.square().sum().backward()
            self.assertTrue(torch.isfinite(w2c.grad).all())
            self.assertTrue(torch.isfinite(k.grad).all())
            self.assertGreater(k.grad[..., [0, 1], [0, 1]].abs().min().item(), 0)
            prediction = pose.detach().requires_grad_()
            ext, ki = decode(prediction, (480, 640))
            (ext[..., :3, 3].square().sum() + ki[..., 0, 0].sum() + ki[..., 1, 1].sum()).backward()
            self.assertTrue(torch.isfinite(prediction.grad).all())
            self.assertGreater(prediction.grad[..., -2:].abs().min().item(), 0)
            self.assertGreater(prediction.grad[..., 3:-2].abs().sum().item(), 0)

    def test_identity_quaternion_encoding_has_finite_backward(self):
        ext = torch.eye(4)[None, None].requires_grad_()
        pose = encode(ext, example()[2], (480, 640))
        pose.sum().backward()
        self.assertTrue(torch.isfinite(ext.grad).all())

    def test_dtype_and_cpu_autocast(self):
        for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            ext, _, k = example()
            for fmt in FORMATS:
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    pose = encode(ext.to(dtype), k.to(dtype), (480, 640), pose_encoding_type=fmt)
                    decoded = decode(pose, (480, 640))
                for tensor in [pose, *decoded]:
                    self.assertEqual(tensor.dtype, torch.float32)
                    self.assertEqual(tensor.device.type, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_roundtrip_autocast_and_backward(self):
        w2c, _, k = random_cameras("cuda")
        for fmt in FORMATS:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pose = encode(w2c, k, (480, 640), pose_encoding_type=fmt)
                pred = pose.detach().requires_grad_()
                ext, ki = decode(pred, (480, 640))
                loss = ext.square().sum() + ki.sum()
            torch.testing.assert_close(ext, w2c[..., :3, :], atol=1e-5, rtol=2e-5)
            self.assertEqual(ext.dtype, torch.float32)
            self.assertEqual(ext.device.type, "cuda")
            loss.backward()
            self.assertTrue(torch.isfinite(pred.grad).all())

    def test_invalid_shapes_formats_and_sizes(self):
        ext, _, k = example()
        for kwargs in [{"pose_encoding_type":"bad"}, {"extrinsics_type":"bad"}]:
            with self.assertRaises(ValueError): encode(ext, k, (480, 640), **kwargs)
        for hw in [None, (0, 640), (480, -1), (float("nan"), 640), torch.ones(1, 3, 2)]:
            with self.assertRaises(ValueError): encode(ext, k, hw)
        with self.assertRaises(ValueError): encode(ext[0], k, (480, 640))
        with self.assertRaises(ValueError): encode(ext, k.expand(2, 1, 3, 3), (480, 640))
        with self.assertRaises(TypeError): encode(ext.long(), k, (480, 640))
        with self.assertRaises(ValueError): decode(torch.zeros(1, 20, 10), (480, 640))
        with self.assertRaises(ValueError): decode(encode(ext, k, (480, 640)), (480, 640), pose_encoding_type=FORMATS[1])

    def test_invalid_rotations_focals_and_nonfinite_inputs(self):
        ext, _, k = example()
        for change in ["scale", "reflection", "bottom", "nan"]:
            bad = ext.clone()
            if change == "scale": bad[..., :3, :3] *= 2
            if change == "reflection": bad[..., :3, 0] *= -1
            if change == "bottom": bad[..., 3, 3] = 2
            if change == "nan": bad[..., 0, 3] = float("nan")
            with self.subTest(change=change), self.assertRaises(ValueError): encode(bad, k, (480, 640))
        for focal in [0, -1, float("inf")]:
            bad = k.clone(); bad[..., 0, 0] = focal
            with self.assertRaises(ValueError): encode(ext, bad, (480, 640))

    def test_degenerate_predictions_are_rejected(self):
        ext, _, k = example()
        for fmt in FORMATS:
            pose = encode(ext, k, (480, 640), pose_encoding_type=fmt)
            for fov in [0, -1, math.pi, float("nan")]:
                bad = pose.clone(); bad[..., -1] = fov
                with self.assertRaises(ValueError): decode(bad, (480, 640))
            bad = pose.clone(); bad[..., 3:-2] = 0
            with self.assertRaises(ValueError): decode(bad, (480, 640))
        with self.assertRaises(ValueError): rotation_6d_to_matrix(torch.tensor([1., 0, 0, 2, 0, 0]))


if __name__ == "__main__":
    unittest.main()
