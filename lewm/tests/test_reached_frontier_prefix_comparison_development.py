from copy import deepcopy
import pytest
from lewm.reached_frontier_prefix_comparison_development import SHARED, compare_step


def fixture(*, reached=False):
    old = dict(tick=3, controller='recent_qualified_direct_flow_controller_v1',
        requested_command=[.16, 0., .45], terminal=None,
        new_selection=dict(prediction=[[[.01, 0., 0.]]], action='left_arc'),
        causal_residual_receipt=dict(pending_forecast_tick=3, executed=[1, 2]))
    old.update({key: {'unchanged': key} for key in SHARED})
    new = deepcopy(old)
    new.update(controller='reached_frontier_recent_qualified_controller_v1',
        reached_frontier_transition_enabled=True, last_frontier_transition_receipt=dict(
            frame=3, measured_ns=1_800_000_000, reached_frontier_cell=[9, -2] if reached else None,
            native_state_used=False, retirement_is_obstacle_evidence=False, retired_cells_remain_traversable=True))
    return old, new


def check(old, new, *, prior=False):
    return compare_step(old, new, old['requested_command'], frame=3, frontier_previously_reached=prior)


def test_before_first_reached_frontier_complete_decision_is_exact():
    assert check(*fixture())['normalized_complete_decision_exact']


def test_reached_frontier_can_change_command_without_changing_forecasts():
    old, new = fixture(reached=True); new['requested_command'] = [0., 0., .45]
    report = check(old, new)
    assert report['requested_command_changed'] and report['raw_model_forecasts_exact']


@pytest.mark.parametrize('key', SHARED)
def test_observed_state_change_is_rejected_even_at_intervention(key):
    old, new = fixture(reached=True); new[key] = 'changed'
    with pytest.raises(ValueError, match='observed evidence'): check(old, new)


@pytest.mark.parametrize('fault', ['early_command', 'forecast', 'residual', 'stale', 'native', 'obstacle'])
def test_non_frontier_changes_rejected(fault):
    old, new = fixture(reached=fault != 'early_command')
    if fault == 'early_command': new['requested_command'] = [0., 0., .45]
    if fault == 'forecast': new['new_selection']['prediction'][0][0][0] += .001
    if fault == 'residual': new['causal_residual_receipt']['executed'].append(3)
    if fault == 'stale': new['last_frontier_transition_receipt']['frame'] = 2
    if fault == 'native': new['last_frontier_transition_receipt']['native_state_used'] = True
    if fault == 'obstacle': new['last_frontier_transition_receipt']['retirement_is_obstacle_evidence'] = True
    with pytest.raises(ValueError): check(old, new)
