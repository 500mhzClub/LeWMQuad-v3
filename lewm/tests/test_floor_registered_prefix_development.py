from copy import deepcopy
import pytest
from scripts.replay_go2_floor_registered_maze_prefix_v1 import compare_current


def records():
    raw = dict(witness='original')
    old = dict(evidence=raw, new_selection=dict(prediction=[1., 2.]),
        terminal=None, requested_command=[0., 0., 0.])
    candidate = dict(original_visual_evidence=deepcopy(raw), evidence=dict(original_visual_evidence=deepcopy(raw)),
        new_selection=dict(prediction=[1., 2.], changed_map=True), failure=None,
        terminal=None, requested_command=[0., 0., 0.])
    return old, deepcopy(old), candidate


def test_only_action_or_terminal_difference_ends_the_prefix():
    a, b, c = records()
    assert compare_current(a, b, c, tick=0) == dict(command_changed=False, terminal_changed=False)
    c['requested_command'] = [.16, 0., .45]
    assert compare_current(a, b, c, tick=1) == dict(command_changed=True, terminal_changed=False)
    c['terminal'] = 'STOP'
    assert compare_current(a, b, c, tick=1) == dict(command_changed=True, terminal_changed=True)


@pytest.mark.parametrize('fault', ['predecessor', 'visual', 'embedded_visual', 'forecast', 'failure'])
def test_reconstruction_failure_cannot_be_called_a_policy_change(fault):
    a, b, c = records()
    if fault == 'predecessor': b['requested_command'][0] = .1
    elif fault == 'visual': c['original_visual_evidence']['witness'] = 'changed'
    elif fault == 'embedded_visual': c['evidence']['original_visual_evidence']['witness'] = 'changed'
    elif fault == 'forecast': c['new_selection']['prediction'][0] = 3.
    else: c['failure'] = 'missing measured floor'
    with pytest.raises(ValueError): compare_current(a, b, c, tick=3)
