"""Evaluator-only full articulated setup envelope; no map or online pose input."""
from dataclasses import asdict
import numpy as np
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.native_foot_geometry_evaluation_development import match_native_foot_geometries
from lewm.physical_execution_development import rotation_xyzw
from lewm.safety.contact_attribution import attribute_contacts
from lewm.setup_snapshot_evaluation_development import check_setup_snapshot,initial_ground_support_witness
from lewm.setup_region_prior_development import SetupRegionPrior
from lewm.setup_velocity_prior_development import SetupVelocityPrior
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.capture_native_robot_geometry_development import capture_native_robot_geometry
from scripts.run_go2_contact_attributed_execution_development_v1 import array,PhysicalStop
from scripts.run_go2_successive_choice_maze_development_v1 import write_json

SETUP_ENVELOPE_PAD_M=.040001


def context_priors(definition_sha256,geometry,joint_position):
    """Enclose every actual articulated primitive plus4cm, with1um strict-interior slack.

    Construct once from evaluator-only setup joints, never from wall proximity or
    a desired pass/fail. Retain the exact native wall SAT, velocity and support
    checks. This is a setup snapshot, not a certified future swept-body envelope.
    """
    if not isinstance(geometry,ArticulatedCollisionGeometry):raise ValueError('full URDF geometry required')
    shapes=geometry.supports(joint_position,np.eye(3))['shapes']
    low=np.min([s['lower'] for s in shapes],axis=0)-SETUP_ENVELOPE_PAD_M
    high=np.max([s['upper'] for s in shapes],axis=0)+SETUP_ENVELOPE_PAD_M
    return (SetupVelocityPrior((0,0,0),1_500_000_000,(0.,0.,0.),.02,definition_sha256),
        SetupRegionPrior((0,0,0),1_500_000_000,2_500_000_000,
            tuple(float(v) for v in low),tuple(float(v) for v in high),definition_sha256))


def admit_context_setup(session, definition_sha256):
    raw = session.samples[-1]
    epoch = int(round(raw['timestamp_s']*1e9))
    if epoch != 1_500_000_000:
        raise ValueError('fifteen actual settling ticks required')
    build = session.ctx.build
    geometry = ArticulatedCollisionGeometry(URDF)
    pose, q = raw['base_pose_world'], raw['joint_position']
    velocity, region = context_priors(definition_sha256, geometry, q)
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
