"""Prospective RGB-only logging; preserve physics and RGB, omit unused depth."""
import hashlib
import json

import numpy as np
from PIL import Image

from lewm.physical_execution_development import rotation_xyzw
from lewm.physical_semantics import world_from_optical
from lewm_genesis.bounded_scene_builder_development import check_capture_domain
from lewm_genesis.optical_camera_readback_development import check_optical_pose
from lewm_genesis.ordered_union_raster_development import verify_order
from scripts.near_field_rgbd_capture_development import array, appearance_environment_identity
from scripts.whole_task_physics_session_development import WholeTaskPhysicsSession


class RGBOnlyGoalCapture:
    def capture_fixed_rgb(self, output, name):
        robot, camera = self.ctx.build.robot, self.ctx.build.camera
        position = array(robot.get_pos()).reshape(-1, 3)[0]
        quat = array(robot.get_quat()).reshape(-1, 4)[0]
        rotation = rotation_xyzw(quat[[1, 2, 3, 0]])
        mount = self.ctx.pack.camera
        assert np.allclose(mount.rpy_body_rad, 0, atol=1e-12, rtol=0)
        camera_position = position+rotation@np.asarray(mount.xyz_body_m)
        forward, up = rotation[:, 0], rotation[:, 2]
        transform = world_from_optical(camera_position, forward, up)
        check_capture_domain(self.ctx.build, transform)
        camera.set_pose(pos=camera_position, lookat=camera_position+forward, up=up)
        check_optical_pose(camera.transform, transform)
        rendered = camera.render(rgb=True, depth=False, segmentation=False, normal=False)
        # Preserve the established renderer call sequence. This separate depth
        # render is discarded immediately, never recorded or given to control.
        camera.render(rgb=False, depth=True, segmentation=False, normal=False)
        rgb = np.asarray(self.ctx.runner._extract_rgb(rendered))
        if rgb.ndim == 4 and rgb.shape[0] == 1:rgb = rgb[0]
        rgb = rgb[..., :3]
        assert rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8
        Image.fromarray(rgb).save(output/f'{name}.png')
        ground = appearance_environment_identity(self)
        if not hasattr(self, '_floor_identity'):
            self._floor_identity = ground
            (output/'floor_visual_collision_identity.json').write_text(json.dumps(ground, indent=2)+'\n')
        else:assert ground == self._floor_identity
        verify_order(camera, self.raster_order)
        return dict(timestamp_s=float(self.samples[-1]['timestamp_s']), world_from_optical=transform.tolist(),
            rigid_mount_no_obstacle_adjustment=True, rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())

    def capture_observation(self, output):
        # Bypass only the RGBD array writer; retain ordinary RGB/body/command
        # acquisition, timestamps, physical trace and fast gyro recording.
        WholeTaskPhysicsSession.capture_observation(self, output)

    def persist_observations(self, output):
        WholeTaskPhysicsSession.persist_observations(self, output)
        (output/'recording_policy.json').write_text(json.dumps(dict(
            rgb_retained=True, physics_commands_body_gyro_retained=True,
            depth_recorded=False, depth_used_by_controller=False,
            reason='prospective RGB-only scientific comparison; unused depth is not retained',
            policy_applies_to_success_and_failure=True), indent=2)+'\n')
