from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn

from lewm.models.pose_head import (
    RelPoseHead,
    body_relative,
    integrate_world_poses,
    ordered_pair_indices,
    pose_aux_loss,
    predicted_pose_aux_loss,
)


class _RolloutStub(nn.Module):
    def plan_rollout(self, z_start_raw: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        steps = action_seq.shape[1]
        return z_start_raw[:, None, :].expand(-1, steps, -1)


class PoseHeadTests(unittest.TestCase):
    def test_integrated_forward_motion(self):
        cmd = torch.zeros(1, 3, 15)
        cmd[:, :, :5] = 1.0
        poses = integrate_world_poses(cmd, 0.1)
        self.assertTrue(torch.allclose(poses[0, :, 0], torch.tensor([0.0, 0.5, 1.0])))

    def test_body_relative_wraps_yaw(self):
        poses = torch.tensor([[[0.0, 0.0, math.pi - 0.1], [0.0, 0.0, -math.pi + 0.1]]])
        rel = body_relative(poses, torch.tensor([0]), torch.tensor([1]))
        self.assertAlmostEqual(float(rel[0, 0, 2]), 0.2, places=5)

    def test_ordered_pairs_are_bidirectional(self):
        a, b = ordered_pair_indices(3, torch.device("cpu"))
        self.assertEqual(set(zip(a.tolist(), b.tolist())), {
            (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1),
        })

    def test_pose_losses_backpropagate(self):
        torch.manual_seed(7)
        head = RelPoseHead(latent_dim=4, hidden=8)
        z_raw = torch.randn(2, 4, 4, requires_grad=True)
        z_proj = torch.randn(2, 4, 4, requires_grad=True)
        cmd = torch.zeros(2, 4, 15)
        poses = integrate_world_poses(cmd, 0.1)

        encoded_loss, _ = pose_aux_loss(head, z_proj, cmd, 0.1, poses=poses)
        predicted_loss, _ = predicted_pose_aux_loss(
            head, _RolloutStub(), z_raw, z_proj, cmd, 0.1, poses=poses,
        )
        (encoded_loss + predicted_loss).backward()

        self.assertIsNotNone(z_raw.grad)
        self.assertIsNotNone(z_proj.grad)
        self.assertGreater(float(z_raw.grad.abs().sum()), 0.0)
        self.assertGreater(float(z_proj.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
