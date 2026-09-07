"""Fresh shadow-motion RGBD acquisition; physical contact identities exclude all visual surfaces."""
import hashlib
from dataclasses import asdict
import numpy as np
from PIL import Image

from lewm.causal_depth_observation_development import INTRINSICS
from lewm.physical_execution_development import rotation_xyzw
from lewm.physical_semantics import world_from_optical
from lewm_genesis.bounded_scene_builder_development import check_capture_domain
from lewm_genesis.optical_camera_readback_development import check_optical_pose
from scripts.run_go2_appearance_information_development_v1 import native_identity
from scripts.single_sample_rgbd_session_development import sampling_readback
from scripts.rgbd_session_development import RGBDSession
from scripts.rgbd_shadow_motion_physical_init_development import AppearancePhysicalInit
from scripts.run_go2_contact_attributed_execution_development_v1 import array, BASE
from scripts.run_go2_successive_choice_maze_development_v1 import write_json
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.native_foot_geometry_evaluation_development import match_native_foot_geometries, nonfoot_ground_contact_indices
from lewm.setup_snapshot_evaluation_development import check_setup_snapshot, initial_ground_support_witness
from lewm.safety.contact_attribution import attribute_contacts
from lewm.rgbd_shadow_motion_development import priors
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.capture_native_robot_geometry_development import capture_native_robot_geometry
from scripts.audit_go2_startup_observation_turn_development_v1 import padded_body_inside_setup
from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop


def appearance_environment_identity(session):
    build = session.ctx.build
    boxes=[dict(wall_id=o.object_id,centre_xyz=o.center_xyz_m,size_xyz=o.size_xyz_m,yaw_rad=o.yaw_rad) for o in build.pack.static_objects]
    row=native_identity(build.physical_environment,build.visual_surfaces,build.expected_visual_geometry,boxes)
    return dict(physical_geometries=row['physical_geometries'],visual_surfaces=row['visual_surfaces'])


class AppearanceRGBDSession(RGBDSession, AppearancePhysicalInit):
    """C3 keeps raw 50Hz/500Hz sensing and contact/stability guards, not a controller."""

    def _build_contact_topology(self):
        topology = BASE._GenesisPhysicalSession._build_contact_topology(self)
        appearance_environment_identity(self)
        build = self.ctx.build
        ground = {int(link.idx) for link in build.collision_floor.links}
        appearance = {int(link.idx) for entity in build.visual_surfaces for link in entity.links}
        if not ground or ground & appearance or ground & topology['robot']:
            raise ValueError('distinct physical-ground and visual-only link identities required')
        return topology | {'ground': ground}

    def install_contact_identity(self):
        build = self.ctx.build
        appearance_environment_identity(self)
        self.link_names = {int(link.idx): str(link.name) for entity in build.scene.entities
                           if entity not in build.visual_surfaces for link in entity.links}
        self.object_ids = {}
        expected = {row['wall_id'] for row in self.geometry['wall_boxes']}
        found = set()
        for entity in build.scene.entities:
            if entity is build.robot or entity in build.visual_surfaces: continue
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
        check_optical_pose(camera.transform,world_from_optical(camera_position,forward,up))
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
        ground = appearance_environment_identity(self)
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


class ShadowMotionSession(AppearanceRGBDSession):
    def __init__(self,*args,**kwargs):
        self.guard=None;self.guard_rows=[]
        super().__init__(*args,**kwargs)

    def _sample(self,requested,applied,timestamp_s):
        row=super()._sample(requested,applied,timestamp_s)
        if self.guard is not None:
            geometry,region,initial,contact=self.guard
            packet={k:np.asarray(v)[0] for k,v in self.packets[-1].items()}
            nonfeet=nonfoot_ground_contact_indices(packet,**contact)
            speed=float(np.linalg.norm(row['base_twist_world'][:3]))
            inside=padded_body_inside_setup(geometry,region,initial,row['base_pose_world'],row['joint_position'])
            current=int(round(timestamp_s*1e9));active=region.anchor_ns<=current<=region.valid_until_ns
            check=dict(sample_index=len(self.samples)-1,measured_ns=current,nonfoot_ground_contact_indices=nonfeet,
                base_speed_m_s=speed,padded_body_inside_region=bool(inside),region_active=bool(active))
            self.guard_rows.append(check)
            if nonfeet or speed>.3 or not inside or not active:
                raise PhysicalStop('SHADOW_COLLECTION_NATIVE_ENVELOPE_VIOLATION')
        return row


def admit_shadow_setup(session,definition_sha256):
    raw=session.samples[-1];epoch=int(round(raw['timestamp_s']*1e9))
    if epoch!=1_500_000_000:raise ValueError('new shadow collector settling epoch required')
    build=session.ctx.build;geometry=ArticulatedCollisionGeometry(URDF);velocity,region=priors(definition_sha256)
    pose=raw['base_pose_world'];q=raw['joint_position'];R=rotation_xyzw(pose[3:])
    rows=capture_native_robot_geometry(build.robot);feet=match_native_foot_geometries(rows,geometry,q,pose)
    static=[]
    for obj,entity in zip(build.pack.static_objects,build.physical_environment[1:],strict=True):
        if entity.name!=obj.object_id or len(entity.geoms)!=1 or entity.geoms[0].type.name!='BOX':
            raise ValueError('exact complete native collision box inventory required')
        static.append(dict(pack_object=asdict(obj),native_name=entity.name,native_collision_boxes=1,
            native_box_size=np.asarray(entity.geoms[0].data).tolist(),
            native_position=array(entity.get_pos()).reshape(3).tolist(),
            native_quaternion_wxyz=array(entity.get_quat()).reshape(4).tolist(),
            fixed=bool(entity.morph.fixed),collision_enabled=bool(entity.morph.collision)))
    check=check_setup_snapshot(velocity,region,identity=(0,0,0),measured_ns=epoch,
        position_world_m=pose[:3],rotation_world_from_initial_body=R,velocity_world_m_s=raw['base_twist_world'][:3],
        native_static_boxes=static,expected_nonfloor_names=tuple(o.object_id for o in build.pack.static_objects),
        geometry=geometry,joint_position=q)
    topology=session._contact_topology
    contacts=attribute_contacts(session.packets[-1],environment_index=0,robot_link_ids=topology['robot'],
        support_link_ids=topology['support'],ground_link_ids=topology['ground'],link_names=session.link_names,
        environment_object_ids=session.object_ids)
    support=initial_ground_support_witness(contacts,expected_support_groups=['FL_calf','FR_calf','RL_calf','RR_calf'],
        ground_link_ids=sorted(topology['ground']),geometry=geometry,joint_position=q,position_world_m=pose[:3],rotation_world_from_body=R)
    report=dict(velocity_prior=asdict(velocity),region_prior=asdict(region),setup=check,support=support,feet=feet,
        sample_index=len(session.samples)-1,definition_sha256=definition_sha256,
        evidence_role='EVALUATOR_ONLY_BOUNDED_ACQUISITION_NOT_DEPLOYMENT_SENSOR')
    write_json(session.output/'static_objects.json',static)
    write_json(session.output/'startup_native_robot_geometry.json',rows)
    write_json(session.output/'setup_checks.json',report)
    if not check['velocity_and_nonfloor_setup_checks_pass'] or not support['initial_native_support_witness_present']:
        raise PhysicalStop('SHADOW_COLLECTION_INITIAL_CONDITIONS_REJECTED')
    session.guard=(geometry,region,np.asarray(pose).copy(),dict(robot_geom_ids=[r['geom_id'] for r in rows],
        foot_geom_ids=sorted(feet['native_foot_geom_to_shape']),ground_geom_ids=[int(g.idx) for g in build.collision_floor.geoms]))
    return velocity
