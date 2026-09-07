import copy
import math

import numpy as np
import pytest

from lewm.active_exit_scan_metrics_development import candidate_side, opening_accounting
from lewm.active_exit_scan_scene_development import scan_scenes
from lewm.active_gyro_scan_development import ActiveGyroScan
from lewm.gravity_feedback_ground_development import CausalGravityFeedbackGround
from lewm.causal_sensor_state import SensorContractError
from scripts.run_go2_active_exit_scan_development_v1 import sensor_decision
from lewm.tests.test_rgb_exit_candidates_development import packet, initialized


def pose(yaw=0., x=0., y=0.):
    return [x, y, .3, 0., 0., math.sin(yaw / 2), math.cos(yaw / 2)]


@pytest.mark.parametrize('yaw,bearing,expected', [(0., 0., [1, 0]), (math.pi/2, 0., [0, 1]),
    (0., math.pi, [-1, 0]), (0., -math.pi/2, [0, -1])])
def test_physical_ray_side(yaw, bearing, expected):
    assert candidate_side(pose(yaw), bearing, 1.) == expected


def test_unknown_origin_and_corner_do_not_count_as_opening():
    assert candidate_side(pose(x=.6), 0., 1.) is None
    assert candidate_side(pose(), math.pi/4, 1.) is None
    with pytest.raises(ValueError):
        candidate_side(pose(), math.nan, 1.)


def test_coverage_keeps_duplicates_false_proposals_and_missing_views():
    spec = scan_scenes()[0]
    raw = {'base_pose_world': np.array([pose()])}
    def row(selected, bearings):
        return {'pre_sample_index': 0, 'selected_view': selected,
                'observation': {'candidate_rows': [{'observation_id': str(i), 'bearing_body_rad': b} for i, b in enumerate(bearings)]}}
    rows = [row(True, [0., 0., math.pi]), row(False, [math.pi/2])]
    result = opening_accounting(spec, raw, rows, selected_views=True)
    assert result['proposals'] == 3 and result['opening_directed'] == 2
    assert result['open_sides_covered'] == 1 and result['closed_side_directed'] == 1
    assert not result['all_open_sides_covered_without_false_proposals']
    assert result['qualified_traversals'] == 0
    assert opening_accounting(spec, raw, [], selected_views=True)['open_sides_expected'] == 1
    assert opening_accounting(spec, raw, rows, selected_views=False)['proposals'] == 4


def test_runtime_boundary_uses_packet_only_and_outputs_json_compatible_rows():
    import json
    actual = packet(initialized(), 80)
    decision, state, observation = sensor_decision(ActiveGyroScan(), CausalGravityFeedbackGround('transported_feedback'),
                                                 actual, tick=0, observation_id='actual-0')
    assert decision['requested_command'] == [0., 0., .35]
    assert state['ground_plane_qualified'] is False
    assert observation['metric_clearance_qualified'] is False
    json.dumps([decision, state, observation], allow_nan=False)
    bad = copy.deepcopy(actual)
    bad['world_pose'] = pose()
    with pytest.raises(SensorContractError):
        sensor_decision(ActiveGyroScan(), CausalGravityFeedbackGround('transported_feedback'), bad, tick=0, observation_id='bad')
