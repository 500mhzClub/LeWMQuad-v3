from __future__ import annotations

import unittest

import torch

from lewm.models.place_retrieval import (
    PlaceRetrievalHead,
    masked_supervised_contrastive_loss,
)


class PlaceRetrievalTests(unittest.TestCase):
    def test_head_returns_unit_embeddings_and_backpropagates(self) -> None:
        torch.manual_seed(7)
        head = PlaceRetrievalHead(latent_dim=8, hidden=16, embedding_dim=4)
        latent = torch.randn(6, 8)
        embedding = head(latent)
        loss = embedding[:, 0].mean()
        loss.backward()

        self.assertEqual(embedding.shape, (6, 4))
        self.assertTrue(torch.allclose(embedding.norm(dim=-1), torch.ones(6), atol=1e-6))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in head.parameters()), 0.0)

    def test_contrastive_loss_rewards_same_place_geometry(self) -> None:
        positive = torch.tensor(
            [
                [False, True, False, False],
                [True, False, False, False],
                [False, False, False, True],
                [False, False, True, False],
            ]
        )
        valid = ~torch.eye(4, dtype=torch.bool)
        good = torch.tensor(
            [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]]
        )
        bad = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.99, 0.01], [0.01, 0.99]]
        )

        good_loss = masked_supervised_contrastive_loss(
            torch.nn.functional.normalize(good, dim=-1),
            positive,
            valid,
        )
        bad_loss = masked_supervised_contrastive_loss(
            torch.nn.functional.normalize(bad, dim=-1),
            positive,
            valid,
        )

        self.assertLess(float(good_loss), float(bad_loss))


if __name__ == "__main__":
    unittest.main()
