"""Actual colocated rasterized depth, kept separate from RGB-only policy input."""
import hashlib
import json

import numpy as np
from PIL import Image

from lewm.causal_depth_observation_development import INTRINSICS, SCHEMA, calibration_metadata, from_native_depth
from lewm.physical_execution_development import rotation_xyzw
from lewm.physical_semantics import world_from_optical
from scripts.run_go2_contact_attributed_execution_development_v1 import array
from scripts.whole_task_physics_session_development import WholeTaskPhysicsSession


class RGBDSession(WholeTaskPhysicsSession):
    def __init__(self, spec, output):
        self.depth_manifest = []
        self.depth_audit = []
        self.latest_native_depth = None
        self.latest_depth = None
        super().__init__(spec, output)

    def capture_fixed_rgb(self, output, name):
        robot, camera = self.ctx.build.robot, self.ctx.build.camera
        position = array(robot.get_pos()).reshape(-1, 3)[0]
        quat = array(robot.get_quat()).reshape(-1, 4)[0]
        rotation = rotation_xyzw(quat[[1, 2, 3, 0]])
        mount = self.ctx.pack.camera
        if not np.allclose(mount.rpy_body_rad, 0, atol=1e-12, rtol=0):
            raise ValueError('fixed zero-RPY RGBD mount required')
        camera_position = position+rotation@np.asarray(mount.xyz_body_m)
        forward, up = rotation[:, 0], rotation[:, 2]
        camera.set_pose(pos=camera_position, lookat=camera_position+forward, up=up)
        if camera._raytracer is not None or camera._batch_renderer is not None:
            raise ValueError('reviewed single-camera rasterizer path required')
        if (tuple(camera.res) != (640, 480) or not np.allclose(camera.intrinsics, INTRINSICS, atol=1e-7, rtol=0)
                or camera.near != .05 or camera.far != 200.):
            raise ValueError('native camera intrinsics or clip differs from depth contract')
        rendered = camera.render(rgb=True, depth=True, segmentation=False, normal=False)
        rgb = np.asarray(self.ctx.runner._extract_rgb(rendered))
        if rgb.ndim == 4 and rgb.shape[0] == 1: rgb = rgb[0]
        rgb = rgb[..., :3]
        native = np.asarray(rendered[1])
        native_shape = list(native.shape)
        if native.ndim == 3 and native.shape[0] == 1: native = native[0]
        if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8 or native.shape != (480, 640) or native.dtype != np.float32:
            raise ValueError('native RGB/depth raster shape or encoding mismatch')
        if rendered[2] is not None or rendered[3] is not None:
            raise ValueError('no segmentation or normal policy input requested')
        Image.fromarray(rgb).save(output/f'{name}.png')
        index = len(self.depth_manifest)
        if name != f'rgb_{index:04d}': raise ValueError('paired RGB/depth acquisition order')
        np.savez_compressed(output/f'native_depth_{index:04d}.npz', optical_depth_m=native)
        self.latest_native_depth = native.copy()
        stamp = float(self.samples[-1]['timestamp_s'])
        self.depth_audit.append({'timestamp_s': stamp, 'physical_sample_index': len(self.samples)-1,
            'native_shape': native_shape, 'native_dtype': str(native.dtype),
            'native_intrinsics': np.asarray(camera.intrinsics).tolist(), 'native_near_m': camera.near,
            'native_far_m': camera.far, 'native_vertical_fov_deg': camera.fov,
            'native_depth_sha256': hashlib.sha256(native.tobytes()).hexdigest(),
            'same_render_call_as_rgb': True, 'renderer': 'genesis_rasterizer',
            'representation': 'optical_axis_depth_m', 'hardware_calibrated': False})
        return {'timestamp_s': stamp, 'world_from_optical': world_from_optical(camera_position, forward, up).tolist(),
                'rigid_mount_no_obstacle_adjustment': True, 'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest()}

    def capture_observation(self, output):
        super().capture_observation(output)
        index = len(self.packet_rows)-1
        now = int(self.packet_rows[index]['decision_ns'])
        with Image.open(output/f'rgb_{index:04d}.png') as image: pixels = np.array(image)
        policy = self.observations.packet(pixels, now)
        depth = from_native_depth(self.latest_native_depth, policy, measured_ns=now, available_ns=now, now_ns=now)
        self.latest_depth = depth
        np.savez_compressed(output/f'depth_{index:04d}.npz', depth_m=depth['depth_m'], valid=depth['valid'])
        metadata = {k: v for k, v in depth.items() if k not in ('depth_m', 'valid')}
        metadata['depth_file'] = f'depth_{index:04d}.npz'
        self.depth_manifest.append(metadata)

    def persist_observations(self, output):
        super().persist_observations(output)
        (output/'depth_observations.json').write_text(json.dumps({'schema': SCHEMA,
            'calibration': calibration_metadata(), 'frames': self.depth_manifest}, indent=2, allow_nan=False)+'\n')
        (output/'depth_camera_audit.json').write_text(json.dumps(self.depth_audit, indent=2, allow_nan=False)+'\n')
