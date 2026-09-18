from copy import deepcopy
import pytest
from lewm.tests.test_floor_registered_prefix_development import records
from scripts.replay_go2_joint_floor_registered_maze_prefix_v1 import compare_current


def test_same_command_then_first_intervention_detected_without_alternative_outcome():
    old, _, candidate = records(); before = deepcopy((old, candidate))
    assert compare_current(old, candidate, tick=0) == dict(command_changed=False, terminal_changed=False)
    assert (old, candidate) == before
    candidate['requested_command'] = [.16, 0., .45]
    assert compare_current(old, candidate, tick=1) == dict(command_changed=True, terminal_changed=False)
    candidate['terminal'] = 'STOP'
    assert compare_current(old, candidate, tick=1)['terminal_changed']


@pytest.mark.parametrize('fault', ['visual', 'embedded_visual', 'forecast', 'failure'])
def test_original_visual_forecast_or_admission_failure_cannot_be_an_intervention(fault):
    old, _, candidate = records()
    if fault == 'visual': candidate['original_visual_evidence']['witness'] = 'changed'
    elif fault == 'embedded_visual': candidate['evidence']['original_visual_evidence']['witness'] = 'changed'
    elif fault == 'forecast': candidate['new_selection']['prediction'][0] = 99.
    else: candidate['failure'] = 'joint plane not observable'
    with pytest.raises(ValueError): compare_current(old, candidate, tick=3)
