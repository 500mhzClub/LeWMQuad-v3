from copy import deepcopy
import numpy as np
import pytest
from lewm.reactive_connector_execution_readout_development import connector_execution
from lewm.tests.test_reactive_nominal_maze_readout_development import (
    test_near_native_endpoint_cannot_relabel_failed_round_trip as check_failed_readout,
    test_paired_readout_preserves_individual_outcomes_and_does_not_infer_attribution as check_comparison)
from scripts import read_go2_reactive_connector_maze_pilot_v1 as readout
from lewm.tests import test_reactive_nominal_maze_readout_development as prior_tests


def fixture():
    poses = np.zeros((800, 7)); poses[:, 6] = 1.
    poses[799, :2] = [.015, -.002]
    clear = dict(radius_m=.45, nominal_disk_connector_clear=True)
    blocked = dict(radius_m=.45, nominal_disk_connector_clear=False)
    selection = dict(nearer_observed_route_target=True, action='forward', requested_command=[.2, 0., 0.],
        original_waypoint_selection=dict(action=None, measured_waypoint_connector=blocked,
            waypoint_map_xy_m=[.35, -.1]), measured_waypoint_connector=clear,
        current_nominal_clearance=clear, current_surface_check=dict(possible_intersection=False),
        original_nominal_radius_preserved=True, waypoint_map_xy_m=[.30, -.1],
        original_route_target_index=7, selected_route_target_index=6)
    row = dict(tick=0, decision=dict(new_selection=selection, terminal=None,
        selected_action='forward', requested_command=[.2, 0., 0.]))
    tape = [dict(tick=0, pre_sample_index=749, post_sample_index=799,
        requested_command=[.2, 0., 0.], completed=True)]
    return poses, tape, [row]


def test_exact_executed_endpoint_and_censored_partial_interval():
    poses, tape, rows = fixture()
    result = connector_execution(poses, tape, rows)[0]
    assert result['native_body_xy_m'] == [.015, -.002]
    assert result['complete_100ms_execution'] and not result['command_outcome_forecast_used']
    assert not result['physical_clearance_certified'] and not result['unexecuted_endpoint_inferred']
    tape[0].update(completed=False, post_sample_index=775)
    result = connector_execution(poses, tape, rows)[0]
    assert result['native_body_xy_m'] is None and not result['complete_100ms_execution']
    assert not result['unexecuted_endpoint_inferred']


@pytest.mark.parametrize('mutation', ['command', 'endpoint', 'radius', 'surface', 'terminal'])
def test_execution_evidence_rejects_mismatched_tape_and_relaxed_constraints(mutation):
    poses, tape, rows = fixture(); selection = rows[0]['decision']['new_selection']
    if mutation == 'command': tape[0]['requested_command'] = [0., 0., 0.]
    if mutation == 'endpoint': tape[0]['post_sample_index'] = 798
    if mutation == 'radius': selection['measured_waypoint_connector']['radius_m'] = .4
    if mutation == 'surface': selection['current_surface_check']['possible_intersection'] = True
    if mutation == 'terminal': rows[0]['decision']['terminal'] = 'STOP'
    with pytest.raises(ValueError): connector_execution(poses, tape, rows)


def test_new_readout_preserves_failed_audit_and_comparison_limits(tmp_path, monkeypatch):
    # Reuse the predecessor's outcome-promotion tests against the new readout.
    # Their rows have no fallback selections, hence no executed entries.
    monkeypatch.setattr(prior_tests, 'module', readout)
    check_failed_readout(tmp_path, monkeypatch)
    check_comparison()
