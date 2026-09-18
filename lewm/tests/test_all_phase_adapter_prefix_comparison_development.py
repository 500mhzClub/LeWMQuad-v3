from copy import deepcopy
import pytest
from lewm.all_phase_adapter_prefix_comparison_development import compare_observed, FAILURE


def pair():
    old = dict(evidence={'position': [0., 0.]}, original_visual_evidence={}, memory_receipt={},
        mission_receipt={'observed_goal_distance_m': 1.3}, floor_partition_receipt={}, auxiliary_floor_partition_receipt={},
        terminal='SENSOR_OR_MODEL_FAILURE', failure=FAILURE, observed_goal_distance_m=None)
    new = deepcopy(old); new.update(terminal=None, failure=None, observed_goal_distance_m=1.3)
    return old, new


def test_original_failure_null_is_not_a_changed_observation():
    old, new = pair(); report = compare_observed(old, new, frame=3)
    assert report['observed_pose_map_contact_and_mission_exact']
    assert report['original_failure_display_distance'] is None
    assert report['candidate_display_distance'] == 1.3


@pytest.mark.parametrize('fault', ['pose', 'mission', 'distance', 'old_distance', 'terminal', 'failure', 'warmup'])
def test_actual_observed_or_contract_differences_remain_rejected(fault):
    old, new = pair(); frame = 3
    if fault == 'pose': new['evidence']['position'][0] = .01
    elif fault == 'mission': new['mission_receipt']['observed_goal_distance_m'] = 1.2
    elif fault == 'distance': new['observed_goal_distance_m'] = 1.2
    elif fault == 'old_distance': old['observed_goal_distance_m'] = 1.3
    elif fault == 'terminal': new['terminal'] = 'OTHER_FAILURE'
    elif fault == 'failure': old['failure'] = 'different cause'
    else: frame = 2
    with pytest.raises(ValueError): compare_observed(old, new, frame=frame)
