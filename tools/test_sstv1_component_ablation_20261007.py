"""No optimizer steps, image inference, backup or production mutation."""
import unittest
import torch
import train_sstv1_component_ablation_20261007 as a


class TrainingAblationTests(unittest.TestCase):
    def test_zero_initialized_residual_preserves_teacher(self):
        for architecture in ('linear', 'mlp'):
            torch.manual_seed(7)
            h = a.ResidualHead(architecture).eval()
            x = torch.randn(2, 1024); z = torch.randn(2, 148)
            self.assertTrue(torch.equal(h(x, z), z))

    def test_disable_only_class_balance_keeps_auxiliary_multiplier(self):
        rows = [{'target': 0}, {'target': 0, 'auxiliary': True}, {'target': 1}]
        w = a.training_weights(rows, False)
        self.assertAlmostEqual(float(w[1]/w[0]), 3)
        self.assertAlmostEqual(float(w[2]/w[0]), 1)
        w = a.training_weights(rows, True)
        self.assertAlmostEqual(float(w[1]/w[0]), 3)
        self.assertAlmostEqual(float(w[2]/w[0]), 2)
        self.assertAlmostEqual(float(w.mean()), 1)

    def test_seed_initialization_is_repeatable(self):
        torch.manual_seed(20260909); x = a.ResidualHead('mlp').state_dict()
        torch.manual_seed(20260909); y = a.ResidualHead('mlp').state_dict()
        self.assertTrue(all(torch.equal(x[k], y[k]) for k in x))


if __name__ == '__main__': unittest.main()
