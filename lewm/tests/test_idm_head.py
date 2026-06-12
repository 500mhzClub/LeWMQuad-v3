from __future__ import annotations

import unittest

import torch

from lewm.models.idm_head import InverseDynamicsHead, idm_loss


class IDMHeadTests(unittest.TestCase):
    def test_forward_shape(self):
        head = InverseDynamicsHead(latent_dim=4, cmd_dim=15, hidden=8)
        z = torch.randn(6, 4)
        self.assertEqual(head(z, z).shape, (6, 15))

    def test_loss_backpropagates_into_latent(self):
        torch.manual_seed(7)
        head = InverseDynamicsHead(latent_dim=4, cmd_dim=15, hidden=8)
        z_lat = torch.randn(2, 4, 4, requires_grad=True)
        cmd = torch.randn(2, 4, 15)
        loss, stats = idm_loss(head, z_lat, cmd)
        loss.backward()
        self.assertIsNotNone(z_lat.grad)
        self.assertGreater(float(z_lat.grad.abs().sum()), 0.0)
        self.assertIn("idm_action_r2", stats)
        self.assertIn("idm_err_wz", stats)

    def test_decodable_action_gives_high_r2(self):
        # A latent that linearly encodes the action should be near-perfectly decodable.
        torch.manual_seed(0)
        head = InverseDynamicsHead(latent_dim=8, cmd_dim=6, hidden=64)
        opt = torch.optim.Adam(head.parameters(), lr=1e-2)
        cmd = torch.randn(32, 5, 6)
        # z_{t+1}-z_t carries the action in its first 6 dims (a clean inverse-dyn signal).
        z = torch.zeros(32, 5, 8)
        z[:, 1:, :6] = cmd[:, :-1]
        for _ in range(300):
            opt.zero_grad()
            loss, stats = idm_loss(head, z, cmd)
            loss.backward()
            opt.step()
        self.assertGreater(stats["idm_action_r2"], 0.8)

    def test_short_sequence_is_safe(self):
        head = InverseDynamicsHead(latent_dim=4, cmd_dim=15, hidden=8)
        z_lat = torch.randn(2, 1, 4)
        cmd = torch.randn(2, 1, 15)
        loss, stats = idm_loss(head, z_lat, cmd)
        self.assertEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
