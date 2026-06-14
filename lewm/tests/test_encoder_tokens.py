from __future__ import annotations

import unittest

import torch

from lewm.models.encoders import VisionEncoder


class VisionEncoderTokenTests(unittest.TestCase):
    def test_forward_tokens_preserves_forward_cls_contract(self) -> None:
        torch.manual_seed(5)
        encoder = VisionEncoder(
            image_size=28,
            patch_size=14,
            hidden_dim=12,
            depth=1,
            n_heads=3,
        )
        image = torch.randn(2, 3, 28, 28)

        tokens = encoder.forward_tokens(image)
        cls = encoder(image)

        self.assertEqual(tokens.shape, (2, 5, 12))
        self.assertTrue(torch.allclose(cls, tokens[:, 0]))


if __name__ == "__main__":
    unittest.main()

