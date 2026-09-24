import math
import pytest

from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.six_action_reactive_controller_development import six_action_selection


def selection(error=0., mode='WAYPOINT', clear=True, connector=True):
    return dict(current_geometry_checked=True, learned_model_used=False,
        candidate_future_outcomes_evaluated=False, command_integrated_pose_used=False,
        current_nominal_clearance=dict(nominal_disk_connector_clear=clear),
        current_surface_check=dict(possible_intersection=False),
        view_budget_exhausted=False, heading_error_rad=error, mode=mode,
        measured_waypoint_connector=dict(nominal_disk_connector_clear=connector),
        action='forward', unknown_waypoint_connector_cells=[])


@pytest.mark.parametrize('error,expected', [(0., 'forward'), (.4, 'left_arc'),
    (-.4, 'right_arc'), (math.pi, 'left_turn'), (-math.pi, 'right_turn')])
def test_instantaneous_feedback_selects_exact_common_primitives(error, expected):
    result = six_action_selection(selection(error))
    assert result['action'] == expected
    assert result['action_bank'] == list(ACTIONS)
    assert result['requested_command'] == candidate_commands(expected)[0]
    assert result['candidate_future_outcomes_evaluated'] is False
    assert result['command_integrated_pose_used'] is False


def test_scan_and_blocked_connector_do_not_admit_translation():
    for original in (selection(.4, mode='VIEW_ACQUISITION'), selection(.4, connector=False)):
        result = six_action_selection(original)
        assert result['action'] == 'left_turn'
        assert all(not r['eligible'] for r in result['candidates'] if r['requested_command'][0]>0)


def test_current_geometry_failure_returns_zero_and_no_admitted_action():
    result = six_action_selection(selection(.4, clear=False))
    assert result['action'] is None and result['requested_command'] == [0., 0., 0.]
    assert not any(r['eligible'] for r in result['candidates'])
