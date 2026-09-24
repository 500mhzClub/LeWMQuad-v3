"""Read-only evaluator check of fixed startup priors and native support witness."""
import json

import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.physical_execution_development import rotation_xyzw
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.safety.contact_attribution import attribute_contacts
from lewm.setup_region_prior_development import SetupRegionPrior
from lewm.setup_velocity_prior_development import SetupVelocityPrior
from lewm.setup_snapshot_evaluation_development import check_setup_snapshot, initial_ground_support_witness
from lewm_genesis.floor_extent_precision_development import check_extent_identity
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings, URDF
from scripts.probe_go2_measured_plane_obstacle_memory_development import DIRECTORY, IDENTITIES, SOURCES
from scripts.run_go2_contact_attributed_execution_development_v1 import CONTACT_FIELDS
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest


PROTOCOL = 'docs/go2_setup_snapshot_recorded_check_2026-09-06.md'
EXTRA = ('lewm/setup_velocity_prior_development.py', 'lewm/setup_region_prior_development.py',
    'lewm/setup_snapshot_evaluation_development.py', 'lewm/tests/test_setup_snapshot_evaluation_development.py',
    'scripts/check_go2_recorded_setup_snapshot_development.py', PROTOCOL)


def main():
    bindings = {str((DIRECTORY / p).relative_to(ROOT)): h for p, h in IDENTITIES.items()}
    verify_bindings(bindings)
    launch = json.loads((DIRECTORY / 'launch.json').read_text())
    result = json.loads((DIRECTORY / 'result.json').read_text())
    reader = json.loads((DIRECTORY / 'native_box_reader_correction.json').read_text())
    assert reader['result']['interface_check_pass'] and result['rgbd_frames'] == 26
    bindings |= launch['source_sha256'] | launch['input_sha256'] | reader['reader_source_sha256']
    bindings |= {str((DIRECTORY / p).relative_to(ROOT)): h for p, h in result['artifact_sha256'].items()}
    development = {p: digest(ROOT / p) for p in (*SOURCES, *EXTRA)}
    verify_bindings(bindings); verify_bindings(development)
    print(json.dumps({'development_source_sha256': development}), flush=True)
    try:
        topology = json.loads((DIRECTORY / 'contact_topology.json').read_text())
        roles = json.loads((DIRECTORY / 'floor_roles.json').read_text())
        assert topology['native_environment_count'] == 1 and topology['selected_environment_index'] == 0
        ground = tuple(topology['ground_link_ids'])
        assert set(ground) == set(roles['physical_ground_link_ids'])
        objects = {int(k): v for k, v in topology['environment_object_ids'].items()}
        assert not set(roles['visual_only_link_ids']) & set(objects)
        assert {objects[i] for i in ground} == {'ground_plane'}
        expected = tuple(objects[i] for i in objects if i not in ground)
        assert len(expected) == len(set(expected))
        floor = check_extent_identity(json.loads((DIRECTORY / 'floor_visual_collision_identity.json').read_text()), 32.)
        assert floor['scene_surface_alignment_verified']
        static = json.loads((DIRECTORY / 'static_objects.json').read_text())
        policy, depth = load_rgbd_observation(DIRECTORY, 0)
        identity = tuple(policy['sensor_state']['identity']); epoch = policy['sensor_state']['decision_ns']
        assert identity == (0, 0, 0) and epoch == 1_500_000_000
        with np.load(DIRECTORY / 'physics_trace.npz', allow_pickle=False) as z:
            matches = np.flatnonzero(np.rint(z['timestamp_s'] * 1e9).astype(np.int64) == epoch)
            assert len(matches) == 1
            index = int(matches[0]); pose = z['base_pose_world'][index]; velocity = z['base_twist_world'][index, :3]
            joints = z['joint_position'][index]
        np.testing.assert_array_equal(joints, policy['sensor_state']['sensed']['joints']['values'][-1, :12])
        R = rotation_xyzw(pose[3:]); geometry = ArticulatedCollisionGeometry(URDF)
        prior = SetupVelocityPrior(identity, epoch, (0., 0., 0.), .02, development[PROTOCOL])
        region = SetupRegionPrior(identity, epoch, 3_500_000_000,
            (-1., -.75, -.5), (1., .75, .6), development[PROTOCOL])
        setup = check_setup_snapshot(prior, region, identity=identity, measured_ns=epoch,
            position_world_m=pose[:3], rotation_world_from_initial_body=R, velocity_world_m_s=velocity,
            native_static_boxes=static, expected_nonfloor_names=expected, geometry=geometry, joint_position=joints)
        with np.load(DIRECTORY / 'native_contacts.npz', allow_pickle=False) as z:
            assert int(round(float(z['frame_timestamp_s'][index]) * 1e9)) == epoch
            start, end = z['frame_offsets'][index:index+2]
            contacts = {k: z[k][start:end][None] for k in CONTACT_FIELDS}
        attributed = attribute_contacts(contacts, environment_index=0,
            robot_link_ids=topology['robot_link_ids'], support_link_ids=topology['support_link_ids'],
            ground_link_ids=ground, link_names={int(k): v for k, v in topology['link_names'].items()},
            environment_object_ids=objects)
        support = initial_ground_support_witness(attributed,
            expected_support_groups=['FL_calf', 'FR_calf', 'RL_calf', 'RR_calf'], ground_link_ids=list(ground),
            geometry=geometry, joint_position=joints, position_world_m=pose[:3], rotation_world_from_body=R)
        print(json.dumps(dict(status='RECORDED_SETUP_SNAPSHOT_CHECK_COMPLETE',
            setup=setup, initial_support_witness=support, snapshot_sample_index=index,
            native_plane_identity_checked=True, original_failures_preserved=True,
            new_physical_run=False, navigation_qualified=False)), flush=True)
    finally:
        verify_bindings(bindings); verify_bindings(development)
    print('ALL_BINDINGS_VERIFIED_BEFORE_AND_AFTER', flush=True)


if __name__ == '__main__': main()
