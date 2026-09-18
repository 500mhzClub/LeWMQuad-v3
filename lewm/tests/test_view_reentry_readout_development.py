from copy import deepcopy
import numpy as np
import pytest
from lewm.tests.test_view_reentry_selection_development import fixture
from lewm.view_reentry_selection_development import reenter_with_translation
from lewm.nominal_reentry_execution_readout_development import collect_entries, executed_motion
from scripts.read_go2_view_reentry_maze_pilot_v1 import view_translation_execution


def test_translating_recovery_readout_uses_executed_tape_and_censors_incomplete_endpoint():
    s = reenter_with_translation(fixture(), np.zeros(3), np.eye(3), [(8, 0)])
    assert s['action'] == 'forward'
    rows = [dict(tick=i, decision=dict(new_selection=deepcopy(s), terminal=None,
        selected_action='forward', requested_command=[.2, 0., 0.])) for i in range(2)]
    entries = collect_entries(rows)
    assert not entries['observed_nominal_reentry_transitions']
    assert entries['unresolved_reentry_start_tick'] == 0
    poses = np.zeros((850, 7)); poses[:, 6] = 1.; poses[799, 0] = -.006
    tape = [dict(tick=0, pre_sample_index=749, post_sample_index=799, completed=True, requested_command=[.2, 0., 0.]),
        dict(tick=1, pre_sample_index=799, post_sample_index=823, completed=False, requested_command=[.2, 0., 0.])]
    records = executed_motion(poses, tape, entries['entries'])
    selected = view_translation_execution(records)
    assert len(selected) == 2 and selected[0]['forecast_xy_error_m'] == pytest.approx(.004)
    assert not selected[1]['complete_100ms_execution'] and selected[1]['native_body_xy_m'] is None
    assert selected[1]['forecast_xy_error_m'] is None and not selected[1]['unexecuted_endpoint_inferred']
    forged = deepcopy(records); forged[0]['recovery_candidate']['phase_allowed'] = True
    with pytest.raises(ValueError): view_translation_execution(forged)
    tape[0]['requested_command'] = [0., 0., .45]
    with pytest.raises(ValueError): executed_motion(poses, tape, entries['entries'])
