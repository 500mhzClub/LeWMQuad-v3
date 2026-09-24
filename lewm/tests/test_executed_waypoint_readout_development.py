from copy import deepcopy
import numpy as np
import pytest
from lewm.executed_waypoint_readout_development import waypoint_execution
from lewm.executed_waypoint_score_development import score_waypoint_execution
from lewm.tests.test_executed_waypoint_score_development import fixture


def test_only_executed_current_selected_motion_is_scored_and_partial_execution_censored():
    s, receipt = fixture(); receipt['correction_xy_m'] = [.004, 0.]
    s = score_waypoint_execution(s, receipt)
    assert s['action'] == 'forward'
    def row(tick):
        return dict(tick=tick, decision=dict(new_selection=deepcopy(s), terminal=None,
            selected_action='forward', requested_command=[.2, 0., 0.]))
    poses = np.zeros((850, 7)); poses[:, 6] = 1.; poses[799, 0] = .006
    tape = [dict(tick=0, pre_sample_index=749, post_sample_index=799, completed=True, requested_command=[.2, 0., 0.]),
        dict(tick=1, pre_sample_index=799, post_sample_index=823, completed=False, requested_command=[.2, 0., 0.])]
    rows = [row(0), row(1)]
    r = waypoint_execution(poses, tape, rows)
    assert r['completed_intervals'] == 1 and r['censored_intervals'] == 1
    assert r['raw_forecast_mean_xy_error_m'] == pytest.approx(.004)
    assert r['causal_scoring_mean_xy_error_m'] == pytest.approx(0.)
    assert r['records'][1]['causal_scoring_xy_error_m'] is None
    assert not r['original_controller_counterfactual_trajectory_inferred']
    rows[0]['decision']['selected_action'] = 'hold'
    with pytest.raises(ValueError): waypoint_execution(poses, tape, rows)
