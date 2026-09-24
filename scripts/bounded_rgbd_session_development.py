"""Fresh bounded/aligned scene acquisition with unchanged gait/sensor wrappers."""
import hashlib
import numpy as np
from PIL import Image

from lewm.causal_depth_observation_development import INTRINSICS
from lewm.physical_execution_development import rotation_xyzw
from lewm.physical_semantics import world_from_optical
from lewm_genesis.bounded_scene_builder_development import check_capture_domain
from lewm_genesis.floor_extent_precision_development import read_extent_identity
from scripts.single_sample_rgbd_session_development import sampling_readback
from scripts.moving_rgbd_session_development import MovingRGBDSession
from scripts.bounded_floor_physical_init_development import BoundedFloorPhysicalInit
from scripts.run_go2_contact_attributed_execution_development_v1 import array, BASE
from scripts.run_go2_successive_choice_maze_development_v1 import write_json


def bounded_floor_identity(session):
    build = session.ctx.build
    return read_extent_identity(build.collision_floor, build.visual_floor, 32.)


class BoundedRGBDSession(MovingRGBDSession, BoundedFloorPhysicalInit):
    """C3 replaces only native initialization plus explicit plane/capture roles."""

    def _build_contact_topology(self):
        topology = BASE._GenesisPhysicalSession._build_contact_topology(self)
        bounded_floor_identity(self)
        build = self.ctx.build
        ground = {int(link.idx) for link in build.collision_floor.links}
        appearance = {int(link.idx) for link in build.visual_floor.links}
        if not ground or ground & appearance or ground & topology['robot']:
            raise ValueError('distinct physical-ground and visual-only link identities required')
        return topology | {'ground': ground}

    def install_contact_identity(self):
        build = self.ctx.build
        bounded_floor_identity(self)
        self.link_names = {int(link.idx): str(link.name) for entity in build.scene.entities
                           if entity is not build.visual_floor for link in entity.links}
        self.object_ids = {}
        expected = {row['wall_id'] for row in self.geometry['wall_boxes']}
        found = set()
        for entity in build.scene.entities:
            if entity is build.robot or entity is build.visual_floor: continue
            if entity is build.collision_floor:
                name = 'ground_plane'
            else:
                name = str(entity.name)
                if name not in expected: raise ValueError('unresolved physical environment identity')
                found.add(name)
            self.object_ids.update({int(link.idx): name for link in entity.links})
        if found != expected: raise ValueError('complete physical wall identities required')
        if self._contact_topology['ground'] != {int(link.idx) for link in build.collision_floor.links}:
            raise ValueError('ground topology must contain collision plane only')

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
        check_capture_domain(self.ctx.build, world_from_optical(camera_position, forward, up))
        camera.set_pose(pos=camera_position, lookat=camera_position+forward, up=up)
        if camera._raytracer is not None or camera._batch_renderer is not None:
            raise ValueError('reviewed single-camera rasterizer path required')
        if (tuple(camera.res) != (640, 480) or not np.allclose(camera.intrinsics, INTRINSICS, atol=1e-7, rtol=0)
                or camera.near != .05 or camera.far != 200.):
            raise ValueError('native camera intrinsics or clip differs from depth contract')
        before = (int(self.ctx.runner._sim_time_ns), len(self.samples))
        transform_before = np.array(camera.transform, copy=True)
        rendered = camera.render(rgb=True, depth=False, segmentation=False, normal=False)
        depth_rendered = camera.render(rgb=False, depth=True, segmentation=False, normal=False)
        after = (int(self.ctx.runner._sim_time_ns), len(self.samples))
        if before != after or not np.array_equal(transform_before, camera.transform):
            raise ValueError('physics or camera changed between RGB and depth')
        sampling = sampling_readback(camera)
        if sampling != {'draw_framebuffer_is_single_sample_target': True,
                        'draw_framebuffer_is_multisample_target': False, 'samples': 0,
                        'sample_buffers': 0, 'multisample_enabled': False, 'pixel_scale': 1}:
            raise ValueError('native depth framebuffer is not single-sample pixel scale1')
        rgb = np.asarray(self.ctx.runner._extract_rgb(rendered))
        if rgb.ndim == 4 and rgb.shape[0] == 1: rgb = rgb[0]
        rgb = rgb[..., :3]
        native = np.asarray(depth_rendered[1])
        native_shape = list(native.shape)
        if native.ndim == 3 and native.shape[0] == 1: native = native[0]
        if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8 or native.shape != (480, 640) or native.dtype != np.float32:
            raise ValueError('native RGB/depth raster shape or encoding mismatch')
        if (any(v is not None for v in rendered[1:]) or depth_rendered[0] is not None
                or any(v is not None for v in depth_rendered[2:])):
            raise ValueError('only separate RGB and depth requested')
        Image.fromarray(rgb).save(output/f'{name}.png')
        index = len(self.depth_manifest)
        if name != f'rgb_{index:04d}': raise ValueError('paired RGB/depth acquisition order')
        ground = bounded_floor_identity(self)
        if index == 0:
            self._floor_identity = ground
            write_json(output/'floor_visual_collision_identity.json', ground)
        elif ground != self._floor_identity:
            raise ValueError('fixed native floor identity drift')
        np.savez_compressed(output/f'native_depth_{index:04d}.npz', optical_depth_m=native)
        self.latest_native_depth = native.copy()
        stamp = float(self.samples[-1]['timestamp_s'])
        self.depth_audit.append({'timestamp_s': stamp, 'physical_sample_index': len(self.samples)-1,
            'native_shape': native_shape, 'native_dtype': str(native.dtype),
            'native_intrinsics': np.asarray(camera.intrinsics).tolist(), 'native_near_m': camera.near,
            'native_far_m': camera.far, 'native_vertical_fov_deg': camera.fov,
            'native_depth_sha256': hashlib.sha256(native.tobytes()).hexdigest(),
            'same_render_call_as_rgb': False, 'same_physics_and_camera_as_rgb': True,
            'physics_clock_before_after_ns': [before[0], after[0]], 'sampling_readback': sampling,
            'renderer': 'genesis_rasterizer', 'representation': 'optical_axis_depth_m', 'hardware_calibrated': False})
        return {'timestamp_s': stamp, 'world_from_optical': world_from_optical(camera_position, forward, up).tolist(),
                'rigid_mount_no_obstacle_adjustment': True, 'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest()}

