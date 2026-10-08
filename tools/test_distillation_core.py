import unittest
import numpy as np
import torch
from distillation_core import class_groups, collapse_logits, kd_loss, fuse, metrics


class DistillationTests(unittest.TestCase):
    def test_merge_order_and_mass(self):
        labels, groups = class_groups(['A', 'Ficus microcarpa', 'B', 'Ficus benjamina'])
        self.assertEqual(groups, [[0], [1, 3], [2]])
        x = torch.randn(8, 4)
        p = x.softmax(-1)
        q = collapse_logits(x, groups).softmax(-1)
        torch.testing.assert_close(q[:, 1], p[:, 1] + p[:, 3])

    def test_kl_identity_and_gradient(self):
        x = torch.randn(5, 7, requires_grad=True)
        loss = kd_loss(x, x.detach().softmax(-1), 4)
        self.assertLess(abs(loss.item()), 1e-5)
        kd_loss(x, torch.randn(5, 7).softmax(-1)).backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_bad_distribution(self):
        with self.assertRaises(ValueError):
            kd_loss(torch.zeros(2, 3), torch.ones(2, 3))

    def test_fusion(self):
        p = np.array([[.2, .8], [.4, .6]])
        for method in ['mean', 'logprob', 'weighted']:
            result = fuse(p, method)
            self.assertAlmostEqual(result.sum(), 1)
            self.assertEqual(result.argmax(), 1)

    def test_metric(self):
        m = metrics([0, 1], [[.9, .1], [.2, .8]])
        self.assertEqual(m['macro_f1'], 1)


if __name__ == '__main__':
    unittest.main()
