from __future__ import annotations

import unittest

import numpy as np

from lewm.planning.depth_local_obstacles import (
    DepthLocalObstacleConfig,
    DepthLocalObstacleModel,
)
from lewm.planning.local_obstacles import require_deployment_valid_local_obstacles


class DepthLocalObstacleModelTest(unittest.TestCase):
    def _model(self) -> DepthLocalObstacleModel:
        return DepthLocalObstacleModel(
            DepthLocalObstacleConfig(
                cell_size_m=0.05,
                vertical_fov_deg=90.0,
                max_range_m=3.0,
                sample_stride_px=1,
                occupied_inflation_m=0.15,
                free_query_radius_m=0.08,
                robot_free_radius_m=0.10,
            )
        )

    def test_center_depth_marks_free_ray_and_occupied_endpoint(self):
        model = self._model()
        depth = np.full((3, 3), np.inf, dtype=np.float32)
        depth[1, 1] = 1.0
        model.update(
            depth,
            camera_position_xyz=np.asarray([0.0, 0.0, 0.4]),
            camera_rotation=np.eye(3, dtype=np.float32),
            robot_xy=(0.0, 0.0),
        )
        self.assertTrue(model.is_free((0.5, 0.0)))
        self.assertFalse(model.is_free((1.0, 0.0)))
        self.assertFalse(model.is_free((0.0, 1.0)))
        require_deployment_valid_local_obstacles(model)

    def test_floor_return_is_not_marked_as_obstacle(self):
        model = self._model()
        depth = np.full((3, 3), np.inf, dtype=np.float32)
        depth[2, 1] = 1.0
        model.update(
            depth,
            camera_position_xyz=np.asarray([0.0, 0.0, 0.1]),
            camera_rotation=np.eye(3, dtype=np.float32),
            robot_xy=(0.0, 0.0),
        )
        diagnostics = model.diagnostics()
        self.assertEqual(diagnostics["recent_occupied_cells"], 0)
        self.assertEqual(diagnostics["last_low_points"], 1)
        self.assertLess(diagnostics["last_endpoint_height_min_m"], 0.08)

    def test_wide_image_grazing_floor_is_not_obstacle(self):
        # Regression for the FOV-axis bug: Genesis ``fov`` is VERTICAL. On a wide
        # (W>H) image a level camera's bottom-row ray grazes the floor ~0.5 m
        # ahead. With the correct vertical-FOV geometry that return projects to
        # z~0 (floor); treating fov as horizontal underestimates the vertical
        # angle and lifts it to z~0.10 m > the 0.08 m obstacle band, filling open
        # corridors with phantom walls. camera_z 0.42, bottom row, depth 0.527 m
        # -> z = 0.42 - 0.527*tan(39.16 deg)*~0.98 ~ 0.0 (low, not an obstacle).
        model = DepthLocalObstacleModel(
            DepthLocalObstacleConfig(
                cell_size_m=0.05,
                vertical_fov_deg=78.323,
                max_range_m=3.0,
                sample_stride_px=1,
                min_obstacle_height_m=0.08,
            )
        )
        depth = np.full((48, 64), np.inf, dtype=np.float32)
        depth[47, 32] = 0.527
        model.update(
            depth,
            camera_position_xyz=np.asarray([0.0, 0.0, 0.42]),
            camera_rotation=np.eye(3, dtype=np.float32),
            robot_xy=(0.0, 0.0),
        )
        diagnostics = model.diagnostics()
        self.assertEqual(diagnostics["recent_occupied_cells"], 0)
        self.assertEqual(diagnostics["last_low_points"], 1)
        self.assertLess(diagnostics["last_endpoint_height_max_m"], 0.08)

    def test_free_ray_clears_stale_occupied(self):
        # A cell marked occupied in one frame is cleared when a later frame sees
        # a ray THROUGH it (standard occupancy semantics); prevents spurious
        # close returns during scan rotations from sticking in a corridor.
        model = self._model()
        cam = np.asarray([0.0, 0.0, 0.4])
        near = np.full((3, 3), np.inf, dtype=np.float32)
        near[1, 1] = 1.0
        model.update(near, camera_position_xyz=cam,
                     camera_rotation=np.eye(3, dtype=np.float32), robot_xy=(0.0, 0.0))
        self.assertFalse(model.is_free((1.0, 0.0)))
        far = np.full((3, 3), np.inf, dtype=np.float32)
        far[1, 1] = 2.0
        model.update(far, camera_position_xyz=cam,
                     camera_rotation=np.eye(3, dtype=np.float32), robot_xy=(0.0, 0.0))
        self.assertTrue(model.is_free((1.0, 0.0)))
        self.assertFalse(model.is_free((2.0, 0.0)))

    def test_reset_drops_rolling_observations(self):
        model = self._model()
        model.update(
            np.ones((3, 3), dtype=np.float32),
            camera_position_xyz=np.asarray([0.0, 0.0, 0.4]),
            camera_rotation=np.eye(3, dtype=np.float32),
            robot_xy=(0.0, 0.0),
        )
        self.assertGreater(model.diagnostics()["frames_fused"], 0)
        model.reset()
        self.assertEqual(model.diagnostics()["frames_fused"], 0)
        self.assertEqual(model.diagnostics()["recent_occupied_cells"], 0)

    def test_simulator_pose_registration_is_not_deployment_valid(self):
        model = DepthLocalObstacleModel(
            odometry_detail="simulator ground-truth pose",
            odometry_deployment_valid=False,
        )
        self.assertFalse(model.contract.deployment_valid)
        with self.assertRaisesRegex(RuntimeError, "deployment-valid local obstacles required"):
            require_deployment_valid_local_obstacles(model)


if __name__ == "__main__":
    unittest.main()
