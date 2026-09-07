from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu import TSCMDLFusionModel  # noqa: E402


class DummyEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.projection(values)


class FusionModelTests(unittest.TestCase):
    def test_frozen_backbones_and_feature_contract(self) -> None:
        point_encoder = DummyEncoder(4, 1024)
        image_encoder = DummyEncoder(5, 2048)
        model = TSCMDLFusionModel(
            point_encoder,
            image_encoder,
            3,
            freeze_backbones=True,
        )
        model.train()
        self.assertFalse(model.point_encoder.training)
        self.assertFalse(model.image_encoder.training)

        points = torch.randn(2, 4)
        images = torch.randn(2, 5)
        features = model.forward_features(points, images)
        self.assertEqual(tuple(features["point_raw"].shape), (2, 1024))
        self.assertEqual(tuple(features["image_raw"].shape), (2, 2048))
        self.assertEqual(tuple(features["fused"].shape), (2, 2048))
        logits = model(points, images)
        cached_logits = model.forward_from_features(
            features["point_raw"], features["image_raw"]
        )
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(cached_logits.shape), (2, 3))

        logits.sum().backward()
        self.assertTrue(
            all(parameter.grad is None for parameter in point_encoder.parameters())
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in image_encoder.parameters())
        )
        fusion_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith(("point_encoder.", "image_encoder."))
        ]
        self.assertTrue(any(parameter.grad is not None for parameter in fusion_parameters))

    def test_unexpected_encoder_dimension_is_rejected(self) -> None:
        model = TSCMDLFusionModel(
            DummyEncoder(4, 1000),
            DummyEncoder(5, 2048),
            3,
        )
        model.eval()
        with self.assertRaisesRegex(ValueError, "point feature shape"):
            model(torch.randn(2, 4), torch.randn(2, 5))

    def test_l2_normalization_balances_cached_feature_scales(self) -> None:
        model = TSCMDLFusionModel(
            nn.Identity(),
            nn.Identity(),
            3,
            point_dim=4,
            image_dim=5,
            modal_dim=4,
            normalization="l2",
        )
        features = model.fuse_features(
            1000.0 * torch.rand(3, 4),
            0.001 * torch.rand(3, 5),
        )
        torch.testing.assert_close(
            features["point_normalized"].norm(dim=1), torch.ones(3)
        )
        torch.testing.assert_close(
            features["image_normalized"].norm(dim=1), torch.ones(3)
        )

    def test_modality_ablation_zeros_only_the_inactive_branch(self) -> None:
        model = TSCMDLFusionModel(
            nn.Identity(),
            nn.Identity(),
            3,
            point_dim=4,
            image_dim=5,
            modal_dim=4,
            normalization="l2",
            classifier_hidden_dims=(8,),
        )
        point_raw = torch.rand(3, 4)
        image_raw = torch.rand(3, 5)
        point_only = model.fuse_features(point_raw, image_raw, "point")
        image_only = model.fuse_features(point_raw, image_raw, "image")
        self.assertTrue(torch.count_nonzero(point_only["point_normalized"]))
        self.assertFalse(torch.count_nonzero(point_only["image_normalized"]))
        self.assertFalse(torch.count_nonzero(image_only["point_normalized"]))
        self.assertTrue(torch.count_nonzero(image_only["image_normalized"]))


if __name__ == "__main__":
    unittest.main()
