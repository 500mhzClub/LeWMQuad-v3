"""Live setup admission and additional physical guards; evaluator data stays here."""
import time

import numpy as np
from PIL import Image

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.native_foot_geometry_evaluation_development import match_native_foot_geometries, nonfoot_ground_contact_indices
from lewm.physical_execution_development import rotation_xyzw
from lewm.safety.contact_attribution import attribute_contacts
from lewm.setup_snapshot_evaluation_development import check_setup_snapshot, initial_ground_support_witness
from lewm.setup_region_prior_development import SetupRegionPrior
from lewm.setup_velocity_prior_development import SetupVelocityPrior
from lewm.startup_observation_turn_development import MAX_BASE_SPEED_M_S
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.bounded_rgbd_session_development import BoundedRGBDSession, bounded_floor_identity
from scripts.capture_native_robot_geometry_development import capture_native_robot_geometry
from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop
from scripts.run_go2_marker_beacon_development_v1 import static_identity
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json


def make_priors(epoch, definition_sha256):
    return (SetupVelocityPrior((0, 0, 0), epoch, (0., 0., 0.), .02, definition_sha256),
            SetupRegionPrior((0, 0, 0), epoch, epoch + 2_000_000_000,
                             (-1., -1., -1.), (1., 1., 1.), definition_sha256))


def latest_policy(session):
    row = session.model_manifest[-1]
    with Image.open(session.output / row['rgb_file']) as image: rgb = np.array(image)
    return session.observations.packet(rgb, int(row['image_ns']))


class StartupObservationSession(BoundedRGBDSession):
    def __init__(self, spec, output):
        self.startup_guard = None; self.startup_guard_rows = []; self.capture_timings = []
        super().__init__(spec, output)

    def capture_current(self):
        before = len(self.model_manifest); start = time.perf_counter()
        try: return super().capture_current()
        finally:
            if len(self.model_manifest) > before:
                self.capture_timings.append(dict(observation_index=before,
                    acquisition_and_depth_observer_ms=1000 * (time.perf_counter() - start)))

    def _sample(self, requested, applied, timestamp_s):
        row = super()._sample(requested, applied, timestamp_s)
        if self.startup_guard is not None:
            packet = {k: np.asarray(v)[0] for k, v in self.packets[-1].items()}
            indices = nonfoot_ground_contact_indices(packet, **self.startup_guard)
            speed = float(np.linalg.norm(row['base_twist_world'][:3]))
            check = dict(sample_index=len(self.samples)-1, measured_ns=int(round(timestamp_s * 1e9)),
                         base_speed_m_s=speed, nonfoot_ground_contact_indices=indices)
            self.startup_guard_rows.append(check)
            if indices: raise PhysicalStop('STARTUP_NONFOOT_GROUND_CONTACT')
            if speed > MAX_BASE_SPEED_M_S: raise PhysicalStop('STARTUP_BASE_SPEED_ASSUMPTION_VIOLATED')
        return row


def admit_setup(session, definition_sha256):
    """Check the actual first setup; return only priors and a narrow handoff."""
    output = session.output; raw = session.samples[-1]
    epoch = int(round(float(raw['timestamp_s']) * 1e9))
    if epoch != 1_500_000_000: raise ValueError('fixed fifteen-tick settling epoch required')
    geometry = ArticulatedCollisionGeometry(URDF); pose = raw['base_pose_world']; q = raw['joint_position']
    R = rotation_xyzw(pose[3:]); velocity, region = make_priors(epoch, definition_sha256)
    floor = bounded_floor_identity(session)
    write_json(output / 'startup_floor_identity.json', floor)
    rows = capture_native_robot_geometry(session.ctx.build.robot)
    write_json(output / 'startup_native_robot_geometry.json', rows)
    feet = match_native_foot_geometries(rows, geometry, q, pose)
    static = static_identity(session); write_json(output / 'static_objects.json', static)
    topology = session._contact_topology
    names = tuple(session.object_ids[i] for i in session.object_ids if i not in topology['ground'])
    setup = check_setup_snapshot(velocity, region, identity=(0, 0, 0), measured_ns=epoch,
        position_world_m=pose[:3], rotation_world_from_initial_body=R,
        velocity_world_m_s=raw['base_twist_world'][:3], native_static_boxes=static,
        expected_nonfloor_names=names, geometry=geometry, joint_position=q)
    contacts = attribute_contacts(session.packets[-1], environment_index=0,
        robot_link_ids=topology['robot'], support_link_ids=topology['support'], ground_link_ids=topology['ground'],
        link_names=session.link_names, environment_object_ids=session.object_ids)
    support = initial_ground_support_witness(contacts,
        expected_support_groups=['FL_calf', 'FR_calf', 'RL_calf', 'RR_calf'],
        ground_link_ids=sorted(topology['ground']), geometry=geometry, joint_position=q,
        position_world_m=pose[:3], rotation_world_from_body=R)
    write_json(output / 'startup_checks.json', dict(setup=setup, support=support, feet=feet,
        sample_index=len(session.samples)-1, definition_sha256=definition_sha256,
        native_plane_identity_verified=True, scope='evaluator-only initial setup; not runtime sensor data'))
    if not setup['velocity_and_nonfloor_setup_checks_pass'] or not support['initial_native_support_witness_present']:
        raise PhysicalStop('STARTUP_ADMISSION_REJECTED')
    admission = dict(schema='startup_setup_admission_development.v1', identity=(0, 0, 0), anchor_ns=epoch,
        definition_sha256=definition_sha256, checks_sha256=digest(output / 'startup_checks.json'),
        velocity_and_nonfloor_checks_pass=True, initial_native_support_witness_present=True)
    write_json(output / 'startup_admission.json', admission)
    session.startup_guard = dict(robot_geom_ids=[r['geom_id'] for r in rows],
        foot_geom_ids=sorted(feet['native_foot_geom_to_shape']),
        ground_geom_ids=[int(g.idx) for g in session.ctx.build.collision_floor.geoms])
    return geometry, velocity, region, admission
