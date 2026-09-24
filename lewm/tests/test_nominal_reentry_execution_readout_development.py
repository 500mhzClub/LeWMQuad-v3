import numpy as np
import pytest
from lewm.nominal_reentry_execution_readout_development import executed_motion, collect_entries
from lewm.tests.test_nominal_clearance_reentry_development import fixture
from lewm.nominal_clearance_reentry_development import reenter
from lewm.tests.test_executed_horizon_final_goal_development import final_selection


def test_native_motion_uses_start_body_rotation_and_only_complete_executed_interval():
    poses = np.zeros((850, 7)); poses[:, 5:] = [np.sin(np.pi/4), np.cos(np.pi/4)]
    poses[799, :2] = [0., .01]
    tape = [dict(tick=0, pre_sample_index=749, post_sample_index=799,
        requested_command=[0., 0., .45], completed=True), dict(tick=1, pre_sample_index=799,
        post_sample_index=823, requested_command=[0., 0., .45], completed=False)]
    entries = [dict(tick=i, action='left_turn', requested_command=[0., 0., .45], predicted_body_xy_m=[.01, 0.]) for i in range(2)]
    r = executed_motion(poses, tape, entries)
    np.testing.assert_allclose(r[0]['native_body_xy_m'], [.01, 0.], atol=1e-15)
    assert r[0]['forecast_xy_error_m'] < 1e-15
    assert r[1]['native_body_xy_m'] is None and not r[1]['complete_100ms_execution']
    tape[0]['requested_command'] = [0., 0., 0.]
    with pytest.raises(ValueError): executed_motion(poses, tape, entries)


def test_only_later_original_gate_pass_counts_as_observed_nominal_reentry():
    recovery = reenter(fixture(), np.zeros(3), np.eye(3), [(8, 0)])
    def row(tick, selection):
        action = selection.get('action')
        return dict(tick=tick, decision=dict(new_selection=selection, terminal=None,
            selected_action=action, requested_command=selection.get('requested_command', [0., 0., 0.])))
    rows = [row(0, recovery), row(1, {}), row(2, recovery)]
    r = collect_entries(rows)
    assert len(r['entries']) == 2 and not r['observed_nominal_reentry_transitions']
    assert r['unresolved_reentry_start_tick'] == 0
    rows.append(row(3, final_selection()))
    r = collect_entries(rows)
    assert r['unresolved_reentry_start_tick'] is None
    assert r['observed_nominal_reentry_transitions'][0]['ordinary_selection_tick'] == 3
    assert not r['observed_nominal_reentry_transitions'][0]['physical_clearance_certified']
