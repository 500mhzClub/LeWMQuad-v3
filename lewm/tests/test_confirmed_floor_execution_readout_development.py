from copy import deepcopy
import numpy as np
import pytest
from lewm.confirmed_floor_execution_readout_development import confirmed_floor_execution
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands


def fixture():
    action = 'forward'; command = candidate_commands(action)[0]
    old = dict(shapes=[], primary_possible_intersection=False, auxiliary_possible_intersection=True,
        possible_intersection=True, auxiliary_shapes=[{'shape_id': 'FL_foot', 'intersecting_voxels': 1}])
    new = old | dict(possible_intersection=False, auxiliary_possible_intersection=False,
        auxiliary_shapes=[{'shape_id': 'FL_foot', 'intersecting_voxels': 0}],
        original_auxiliary_floor_contact_check=deepcopy(old), current_primary_floor_confirmation={'frame': 0},
        non_foot_contacts_exempted=False, non_floor_or_unknown_contacts_exempted=False)
    selection = dict(action=action, surface_checks=[deepcopy(new) for _ in ACTIONS],
        prediction=[[[.01, 0., 0., 1., 0.]] for _ in ACTIONS])
    decision = dict(new_selection=selection, selected_action=action, requested_command=command,
        terminal=None, current_primary_floor_confirmation_enabled=True)
    poses = np.zeros((800, 7)); poses[:, 6] = 1.; poses[799, 0] = .012
    tape = [dict(tick=0, pre_sample_index=749, post_sample_index=799, completed=True, requested_command=command)]
    return poses, tape, [dict(tick=0, decision=decision)]


def test_actual_complete_and_censored_motion_and_terminal_exclusion():
    poses, tape, rows = fixture()
    result = confirmed_floor_execution(poses, tape, rows)
    assert result['completed_intervals'] == 1
    assert result['records'][0]['forecast_xy_error_m'] == pytest.approx(.002)
    tape[0]['completed'] = False
    result = confirmed_floor_execution(poses[:770], tape, rows)
    assert result['censored_intervals'] == 1 and result['records'][0]['native_body_xy_m'] is None
    rows[0]['decision']['terminal'] = 'STOP'
    assert confirmed_floor_execution(poses, tape, rows)['records'] == []


@pytest.mark.parametrize('change', ['command', 'primary', 'unknown', 'blocked', 'endpoint'])
def test_changed_policy_or_missing_execution_cannot_support_readout(change):
    poses, tape, rows = fixture(); d = rows[0]['decision']; s = d['new_selection']
    check = s['surface_checks'][ACTIONS.index('forward')]
    if change == 'command': tape[0]['requested_command'] = [0., 0., 0.]
    elif change == 'primary': check['primary_possible_intersection'] = True
    elif change == 'unknown': check['non_floor_or_unknown_contacts_exempted'] = True
    elif change == 'blocked': check['possible_intersection'] = True
    else: poses = poses[:799]
    with pytest.raises(ValueError): confirmed_floor_execution(poses, tape, rows)
