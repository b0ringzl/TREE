from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tscmdl_whu.ptv2 import (  # noqa: E402
    GridPool,
    GroupedLinear,
    GroupedVectorAttention,
    PointTransformerV2Classifier,
    knn_query_packed,
    packed_point_counts,
)


class PointTransformerV2Tests(unittest.TestCase):
    def test_grouped_linear_projects_each_group_independently(self) -> None:
        layer = GroupedLinear(4, 2)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
        values = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        projected = layer(values)
        self.assertTrue(torch.equal(projected, torch.tensor([[3.0, 7.0]])))

    def test_knn_query_stays_inside_packed_cloud_boundaries(self) -> None:
        coord = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
            ]
        )
        offset = torch.tensor([2, 5])
        reference = knn_query_packed(coord, offset, 4)
        self.assertEqual(tuple(reference.shape), (5, 4))
        self.assertTrue(torch.all(reference[:2] < 2))
        self.assertTrue(torch.all(reference[2:] >= 2))
        self.assertTrue(torch.all(reference[2:] < 5))

    def test_grid_pool_uses_mean_coordinates_and_max_features(self) -> None:
        pool = GridPool(1, 1, grid_size=1.0)
        pool.eval()
        with torch.no_grad():
            pool.projection.weight.fill_(1.0)
            pool.norm.norm.weight.fill_(1.0)
            pool.norm.norm.bias.zero_()
            pool.norm.norm.running_mean.zero_()
            pool.norm.norm.running_var.fill_(1.0)
        coord = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [1.2, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ]
        )
        feat = torch.tensor([[1.0], [3.0], [2.0], [4.0]])
        pooled_coord, pooled_feat, offset = pool(coord, feat, torch.tensor([3, 4]))
        self.assertEqual(packed_point_counts(offset), [2, 1])
        self.assertTrue(torch.allclose(pooled_coord[0], torch.tensor([0.2, 0.0, 0.0])))
        self.assertAlmostEqual(float(pooled_feat[0].detach()), 3.0, places=4)

    def test_grouped_attention_preserves_shape_and_gradients(self) -> None:
        torch.manual_seed(4)
        attention = GroupedVectorAttention(8, 2)
        coord = torch.randn(12, 3)
        feat = torch.randn(12, 8, requires_grad=True)
        reference = knn_query_packed(coord, torch.tensor([6, 12]), 4)
        output = attention(feat, coord, reference)
        self.assertEqual(tuple(output.shape), (12, 8))
        output.square().mean().backward()
        self.assertIsNotNone(feat.grad)
        self.assertTrue(torch.isfinite(feat.grad).all())

    def test_small_classifier_forward_and_backward(self) -> None:
        torch.manual_seed(8)
        model = PointTransformerV2Classifier(
            3,
            patch_depth=1,
            patch_channels=8,
            patch_groups=2,
            patch_neighbours=4,
            encoder_depths=(1, 1),
            encoder_channels=(16, 32),
            encoder_groups=(4, 8),
            encoder_neighbours=(4, 4),
            grid_sizes=(0.4, 0.8),
            classifier_hidden=16,
            classifier_dropout=0.0,
        )
        points = torch.randn(2, 32, 3)
        local_reference = []
        for cloud in points:
            local_reference.append(
                knn_query_packed(cloud, torch.tensor([len(cloud)]), 4)
            )
        reference = torch.stack(local_reference)
        logits = model(points, reference)
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(len(model.last_stage_point_counts), 3)
        logits.sum().backward()
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
