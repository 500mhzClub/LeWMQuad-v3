from __future__ import annotations

import unittest

import torch

from lewm.models.action_ranker import (
    GoalActionRanker,
    TaskAlignedCandidateScorer,
    action_ranker_loss,
    first_action_metrics,
    task_aligned_candidate_loss,
)


class GoalActionRankerTests(unittest.TestCase):
    def test_forward_shape_and_loss_backprop(self) -> None:
        torch.manual_seed(3)
        head = GoalActionRanker(latent_dim=8, cmd_dim=6, hidden=32)
        start = torch.randn(4, 5, 8)
        goal = torch.randn(4, 5, 8)
        action = torch.randn(4, 5, 6)
        target = torch.rand(4, 5)

        scores = head(start, goal, action)
        loss = action_ranker_loss(scores, target)
        loss.backward()

        self.assertEqual(scores.shape, (4, 5))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in head.parameters()), 0.0)

    def test_metrics_distinguish_oracle_from_random(self) -> None:
        distance = torch.tensor([[3.0, 1.0, 2.0], [2.0, 3.0, 1.0]])
        metrics = first_action_metrics(distance.clone(), distance)

        self.assertAlmostEqual(metrics["mean_first_regret_m"], 0.0)
        self.assertAlmostEqual(metrics["regret_ratio_vs_random"], 0.0)
        self.assertAlmostEqual(metrics["mean_first_spearman"], 1.0)

    def test_task_aligned_scorer_shape_and_masked_loss(self) -> None:
        torch.manual_seed(7)
        head = TaskAlignedCandidateScorer(latent_dim=8, cmd_dim=6, hidden=32)
        start = torch.randn(4, 5, 8)
        goal = torch.randn(4, 5, 8)
        goal_present = torch.tensor([True, True, False, False])
        action = torch.randn(4, 5, 6)
        targets = {
            "collision": torch.randint(0, 2, (4, 5)).float(),
            "progress": torch.randn(4, 5),
            "heading": torch.randn(4, 5),
            "clearance": torch.randn(4, 5),
        }

        predictions = head(
            start,
            goal,
            goal_present[:, None].expand(-1, 5),
            action,
        )
        loss, components = task_aligned_candidate_loss(
            predictions,
            targets,
            goal_present=goal_present,
        )
        loss.backward()

        self.assertEqual(set(predictions), {"collision_logit", "progress", "heading", "clearance"})
        self.assertTrue(all(value.shape == (4, 5) for value in predictions.values()))
        self.assertEqual(set(components), {"collision", "progress", "heading", "clearance"})
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in head.parameters()), 0.0)


if __name__ == "__main__":
    unittest.main()
