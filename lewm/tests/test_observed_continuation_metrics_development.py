import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.observed_continuation_scene_development import trials
from lewm.observed_continuation_metrics_development import reduce_continuation
from scripts.analyze_go2_ground_plane_development_v1 import URDF


def fixture():
    count = 4500
    poses = np.tile([0., 0., .3, 0., 0., 0., 1.], (count, 1))
    poses[750:1750, 0] = np.linspace(0., 1.2, 1000)
    poses[1750:, 0] = 1.2
    poses[3250:4250, 1] = np.linspace(0., 1.2, 1000)
    poses[4250:, 1] = 1.2
    poses[3250:, 3:] = [0., 0., 2**-.5, 2**-.5]
    raw = {'timestamp_s': .002*np.arange(1, count+1), 'base_pose_world': poses,
        'joint_position': np.zeros((count, 12)), 'base_twist_world': np.zeros((count, 6)),
        'phase': np.ones(count, int), 'requested_command': np.zeros((count, 3)),
        'physics_contact': np.zeros(count, bool)}
    def row(stage, index, status):
        return {'pre_sample_index': index, 'controller': {'stage': stage,
            'child': {'status': status, 'terminal': status == 'ARRIVAL_CANDIDATE'},
            'scan': None, 'selected_side_branch': None}}
    decisions = [row('FIRST', 749, 'WARMUP'), row('FIRST', 1749, 'ARRIVAL_CANDIDATE'),
        {'pre_sample_index': 2749, 'controller': {'stage': 'SCAN', 'child': None,
            'scan': {'status': 'COMPLETE'}, 'selected_side_branch': {'place_identity': None}}},
        row('SECOND', 3249, 'WARMUP'), row('SECOND', 4249, 'ARRIVAL_CANDIDATE')]
    return trials()[0], raw, decisions, ArticulatedCollisionGeometry(URDF)


def test_first_arrival_is_evaluated_before_later_scan_and_second_destination():
    spec, raw, decisions, model = fixture()
    before = raw['phase'].copy()
    result = reduce_continuation(spec, raw, 749, decisions, 'COMPLETE_PROVISIONAL', None, None, model)
    assert result['two_leg_integration_success'] and result['trusted_graph_edges'] == 0
    assert result['legs'][0]['end_sample_index'] == 1999
    assert result['legs'][1]['end_sample_index'] == 4499
    assert np.array_equal(raw['phase'], before)
    # The final robot is outside the first destination; using the whole trace
    # for its arrival would incorrectly erase the earlier actual crossing.
    assert raw['base_pose_world'][-1, 1] > .6


def test_later_native_stop_retains_first_crossing_but_fails_continuous_task():
    spec, raw, decisions, model = fixture()
    raw = {k: v[:2750].copy() for k, v in raw.items()}
    raw['physics_contact'][-1] = True
    result = reduce_continuation(spec, raw, 749, decisions[:3], None, 'DISALLOWED_CONTACT', None, model)
    assert result['legs'][0]['response']['integration_success']
    assert result['legs'][1]['stage_started'] is False
    assert not result['two_leg_integration_success']


def test_nonzero_release_or_early_contact_cannot_pass_leg_endpoint():
    spec, raw, decisions, model = fixture()
    raw['requested_command'][1800, 0] = .1
    result = reduce_continuation(spec, raw, 749, decisions, 'COMPLETE_PROVISIONAL', None, None, model)
    assert not result['legs'][0]['response']['actual_zero_release_requests']
    assert not result['two_leg_integration_success']
    raw = {k: v[:1850].copy() for k, v in raw.items()}
    raw['physics_contact'][-1] = True
    result = reduce_continuation(spec, raw, 749, decisions[:2], None, 'DISALLOWED_CONTACT', None, model)
    assert not result['legs'][0]['response']['physical_checks']['release_motion']
    assert not result['two_leg_integration_success']
