import unittest
import torch
from bioclip_probe import ResidualHead


class ProbeTests(unittest.TestCase):
    def test_zero_initialized_residual_preserves_teacher(self):
        x = torch.randn(7, 12); z = torch.randn(7, 4)
        for architecture in ['linear', 'mlp']:
            head = ResidualHead(architecture, 12, 4)
            torch.testing.assert_close(head(x, z), z)

    def test_gradient_updates_head(self):
        for architecture in ['linear', 'mlp']:
            torch.manual_seed(4)
            head = ResidualHead(architecture, 12, 4)
            x = torch.randn(8, 12); z = torch.randn(8, 4)
            before = head(x, z).detach().clone()
            optimizer = torch.optim.AdamW(head.parameters(), lr=.01)
            loss = torch.nn.functional.cross_entropy(head(x, z), torch.arange(8) % 4)
            loss.backward(); optimizer.step()
            head.eval()
            self.assertFalse(torch.allclose(head(x, z), before))


if __name__ == '__main__': unittest.main()
