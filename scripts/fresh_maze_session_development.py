"""Fresh complete-maze physics with external native guard and actual RGBD."""
from dataclasses import asdict
import numpy as np
from PIL import Image

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.native_foot_geometry_evaluation_development import match_native_foot_geometries, nonfoot_ground_contact_indices
from lewm.physical_execution_development import rotation_xyzw
from lewm.physical_semantics import world_from_optical
from lewm.safety.contact_attribution import attribute_contacts
from lewm.setup_snapshot_evaluation_development import check_setup_snapshot, initial_ground_support_witness
from lewm.setup_region_prior_development import SetupRegionPrior
from lewm.setup_velocity_prior_development import SetupVelocityPrior
from lewm.rgb_marker_beacon_development import _components
from lewm_genesis.optical_camera_readback_development import check_optical_pose
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.capture_native_robot_geometry_development import capture_native_robot_geometry
from scripts.fresh_maze_physical_init_development import MissionPhysicalInit
from scripts.rgbd_session_development import RGBDSession
from scripts.rgbd_shadow_motion_session_development import AppearanceRGBDSession, appearance_environment_identity
from scripts.run_go2_contact_attributed_execution_development_v1 import array, PhysicalStop
from scripts.run_go2_successive_choice_maze_development_v1 import write_json


def priors(definition_sha256):
    # The finite initial region is checked ONLY for initial setup, not supplied
    # to navigation, propagated as a map, or used to authorize future commands.
    return (SetupVelocityPrior((0, 0, 0), 1_500_000_000, (0., 0., 0.), .02, definition_sha256),
            SetupRegionPrior((0, 0, 0), 1_500_000_000, 2_500_000_000,
                             (-.8, -.8, -.8), (.8, .8, .8), definition_sha256))


def validate_command(command):
    value = np.asarray(command, dtype=float)
    if (value.shape != (3,) or not np.isfinite(value).all() or not 0 <= value[0] <= .2
            or value[1] != 0 or abs(value[2]) > .35):
        raise ValueError('finite declared forward/yaw development command limits required')
    return value.tolist()


class MissionRGBDSession(RGBDSession, MissionPhysicalInit):
    # These frozen methods are stateless with respect to their defining class;
    # keep their exact native geometry/contact/camera checks with the NEW init.
    _build_contact_topology = AppearanceRGBDSession._build_contact_topology
    install_contact_identity = AppearanceRGBDSession.install_contact_identity
    capture_fixed_rgb = AppearanceRGBDSession.capture_fixed_rgb

    def __init__(self, *args, **kwargs):
        self.guard = None
        self.guard_rows = []
        super().__init__(*args, **kwargs)

    def command_tick(self, requested):
        return super().command_tick(validate_command(requested))

    def _sample(self, requested, applied, timestamp_s):
        row = super()._sample(requested, applied, timestamp_s)
        if self.guard is not None:
            packet = {k: np.asarray(v)[0] for k, v in self.packets[-1].items()}
            indices = nonfoot_ground_contact_indices(packet, **self.guard)
            speed = float(np.linalg.norm(row['base_twist_world'][:3]))
            self.guard_rows.append(dict(sample_index=len(self.samples)-1,
                measured_ns=int(round(timestamp_s*1e9)), nonfoot_ground_contact_indices=indices,
                base_speed_m_s=speed, evaluator_supervision_not_policy_input=True))
            if indices or speed > .3:
                raise PhysicalStop('FRESH_MISSION_NATIVE_CONTACT_OR_SPEED_GUARD')
        return row


def admit_setup(session, definition_sha256):
    raw = session.samples[-1]
    epoch = int(round(raw['timestamp_s']*1e9))
    if epoch != 1_500_000_000:
        raise ValueError('fifteen actual settling ticks required')
    build = session.ctx.build
    geometry = ArticulatedCollisionGeometry(URDF)
    velocity, region = priors(definition_sha256)
    pose, q = raw['base_pose_world'], raw['joint_position']
    rotation = rotation_xyzw(pose[3:])
    native = capture_native_robot_geometry(build.robot)
    feet = match_native_foot_geometries(native, geometry, q, pose)
    static = []
    for obj, entity in zip(build.pack.static_objects, build.physical_environment[1:], strict=True):
        if entity.name != obj.object_id or len(entity.geoms) != 1 or entity.geoms[0].type.name != 'BOX':
            raise ValueError('complete actual collision-box roster required')
        static.append(dict(pack_object=asdict(obj), native_name=entity.name, native_collision_boxes=1,
            native_box_size=np.asarray(entity.geoms[0].data).tolist(),
            native_position=array(entity.get_pos()).reshape(3).tolist(),
            native_quaternion_wxyz=array(entity.get_quat()).reshape(4).tolist(),
            fixed=bool(entity.morph.fixed), collision_enabled=bool(entity.morph.collision)))
    check = check_setup_snapshot(velocity, region, identity=(0, 0, 0), measured_ns=epoch,
        position_world_m=pose[:3], rotation_world_from_initial_body=rotation,
        velocity_world_m_s=raw['base_twist_world'][:3], native_static_boxes=static,
        expected_nonfloor_names=tuple(o.object_id for o in build.pack.static_objects),
        geometry=geometry, joint_position=q)
    topology = session._contact_topology
    contacts = attribute_contacts(session.packets[-1], environment_index=0, robot_link_ids=topology['robot'],
        support_link_ids=topology['support'], ground_link_ids=topology['ground'], link_names=session.link_names,
        environment_object_ids=session.object_ids)
    support = initial_ground_support_witness(contacts,
        expected_support_groups=['FL_calf', 'FR_calf', 'RL_calf', 'RR_calf'],
        ground_link_ids=sorted(topology['ground']), geometry=geometry, joint_position=q,
        position_world_m=pose[:3], rotation_world_from_body=rotation)
    write_json(session.output/'static_objects.json', static)
    write_json(session.output/'startup_native_robot_geometry.json', native)
    write_json(session.output/'setup_checks.json', dict(velocity_prior=asdict(velocity), region_prior=asdict(region),
        setup=check, support=support, feet=feet, sample_index=len(session.samples)-1,
        definition_sha256=definition_sha256, setup_region_used_for_navigation=False,
        evidence_role='EVALUATOR_ONLY_INITIAL_SETUP_NOT_FUTURE_GAIT_QUALIFICATION'))
    if not check['velocity_and_nonfloor_setup_checks_pass'] or not support['initial_native_support_witness_present']:
        raise PhysicalStop('FRESH_MISSION_INITIAL_SETUP_REJECTED')
    session.guard = dict(robot_geom_ids=[r['geom_id'] for r in native],
        foot_geom_ids=sorted(feet['native_foot_geom_to_shape']),
        ground_geom_ids=[int(g.idx) for g in build.collision_floor.geoms])
    return velocity


def marker_pixel_pairs(rgb):
    """Same pixel predicate as the frozen detector; no fake body sensor packet."""
    image = np.asarray(rgb)
    if image.shape != (480, 640, 3) or image.dtype != np.uint8:
        raise ValueError('native RGB pixels required')
    red, green, blue = np.moveaxis(image.astype(np.int16), -1, 0)
    reds = _components((red >= 16) & (red >= 2*green) & (red >= 2*blue))
    blues = _components((blue >= 16) & (blue >= 2*red) & (blue >= 2*green))
    matches = []
    for r in reds:
        rx0, ry0, rx1, ry1 = r['bbox_xyxy']
        rw, rh = rx1-rx0, ry1-ry0
        for b in blues:
            bx0, by0, bx1, by1 = b['bbox_xyxy']
            bw, bh = bx1-bx0, by1-by0
            overlap = max(0, min(ry1, by1)-max(ry0, by0))
            if (0 <= bx0-rx1 <= .5*max(rw, bw) and .5 <= rw/bw <= 2.
                    and .7 <= rh/bh <= 1/.7 and overlap/max(rh, bh) >= .7):
                matches.append([rx0, min(ry0, by0), bx1, max(ry1, by1)])
    return matches


def marker_visibility_assay(session):
    """Static positive render, NOT robot sensing, never sent to the controller."""
    camera = session.ctx.build.camera
    before = (int(session.ctx.build.scene.t), int(session.ctx.runner._sim_time_ns), len(session.samples))
    position = np.array([3.6, -3.6, .43])
    forward, up = np.array([0., -1., 0.]), np.array([0., 0., 1.])
    camera.set_pose(pos=position, lookat=position+forward, up=up)
    expected = world_from_optical(position, forward, up)
    check_optical_pose(camera.transform, expected)
    rendered = camera.render(rgb=True, depth=False, segmentation=False, normal=False)
    rgb = np.asarray(session.ctx.runner._extract_rgb(rendered))
    if rgb.ndim == 4:
        rgb = rgb[0]
    rgb = rgb[..., :3]
    pairs = marker_pixel_pairs(rgb)
    Image.fromarray(rgb).save(session.output/'marker_visibility_assay.png')
    after = (int(session.ctx.build.scene.t), int(session.ctx.runner._sim_time_ns), len(session.samples))
    report = dict(world_from_optical=expected.tolist(), marker_pairs=pairs,
                  clocks_before_after=[list(before), list(after)], controller_received_image=False,
                  body_mounted_sensor=False, physics_advanced=False)
    write_json(session.output/'marker_visibility_assay.json', report)
    if before != after or not pairs:
        raise ValueError('native marker visibility or zero-motion assay failed')
    return report
