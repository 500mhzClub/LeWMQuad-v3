from __future__ import annotations

import unittest

import torch

from scripts.probe_lewm_patch_retrieval import spatial_pyramid_descriptor


class PatchRetrievalTests(unittest.TestCase):
    def test_spatial_pyramid_shape_and_global_mean(self) -> None:
        patches = torch.arange(2 * 16 * 3, dtype=torch.float32).reshape(2, 16, 3)
        descriptor = spatial_pyramid_descriptor(patches, levels=(1, 2, 4))

        self.assertEqual(descriptor.shape, (2, (1 + 4 + 16) * 3))
        self.assertTrue(torch.allclose(descriptor[:, :3], patches.mean(dim=1)))


if __name__ == "__main__":
    unittest.main()

