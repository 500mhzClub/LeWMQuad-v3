from __future__ import annotations

import unittest

from lewm.planning.local_obstacles import (
    PerceptionObstacleModel,
    PrivilegedGridObstacleModel,
    require_deployment_valid_local_obstacles,
)


class _FakeGrid:
    def is_free(self, xy):
        return xy[0] >= 0.0

    def nearest_free(self, xy, *, max_radius_m=None):
        del max_radius_m
        return (max(0.0, xy[0]), xy[1])


class LocalObstacleModelTest(unittest.TestCase):
    def test_privileged_grid_adapter_is_explicitly_deployment_invalid(self):
        model = PrivilegedGridObstacleModel(_FakeGrid())
        self.assertTrue(model.is_free((1.0, 0.0)))
        self.assertFalse(model.is_free((-1.0, 0.0)))
        self.assertEqual(model.nearest_free((-1.0, 2.0)), (0.0, 2.0))
        self.assertEqual(model.contract.source, "privileged_manifest_grid")
        self.assertFalse(model.contract.deployment_valid)
        with self.assertRaisesRegex(RuntimeError, "privileged_manifest_grid"):
            require_deployment_valid_local_obstacles(model)

    def test_perception_adapter_is_deployment_valid_and_uses_provider_queries(self):
        model = PerceptionObstacleModel(
            lambda xy: xy[1] >= 0.0,
            lambda xy, radius: (
                (xy[0], max(0.0, xy[1])) if radius is None or radius > 0 else None
            ),
            source="depth_local_occupancy_v0",
        )
        self.assertTrue(model.is_free((0.0, 1.0)))
        self.assertFalse(model.is_free((0.0, -1.0)))
        self.assertEqual(
            model.nearest_free((2.0, -1.0), max_radius_m=0.5),
            (2.0, 0.0),
        )
        self.assertTrue(model.contract.deployment_valid)
        require_deployment_valid_local_obstacles(model)

    def test_perception_adapter_rejects_privileged_source_name(self):
        with self.assertRaisesRegex(ValueError, "onboard provider"):
            PerceptionObstacleModel(
                lambda _xy: True,
                lambda xy, _radius: xy,
                source="privileged_manifest_grid",
            )


if __name__ == "__main__":
    unittest.main()
