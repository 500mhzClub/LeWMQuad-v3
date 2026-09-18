from copy import deepcopy

from lewm.clearance_turn_recovery_development import recover_turn
from lewm.full_reserve_heading_release_development import (
    FullReserveHeadingReleaseRuntime, release_heading_turn,
)
from lewm.arrival_entry_terminal_priority_development import ArrivalEntryTerminalPriorityRuntime
from lewm.no_early_heading_release_development import NoEarlyHeadingReleaseMixin, retain_recovery_heading
from lewm.tests.test_clearance_turn_recovery_development import selection


def releasable():
    selected = selection()
    _, state = recover_turn(selected, 0., 0, None)
    for row in selected['memory_forecast_candidates']:
        row.update(minimum_predicted_path_clearance_m=.6, segment_clearances_m=[.6]*8)
    result, state = recover_turn(selected, .5, 0, state)
    return result, state


def test_eligible_reversal_is_recorded_but_does_not_change_action_or_clearance():
    result, state = releasable()
    before, previous = deepcopy(result), deepcopy(state)
    released, cleared = release_heading_turn(result, state)
    assert released['action'] == 'right_turn' and cleared is None
    kept, retained = retain_recovery_heading(result, state)
    assert kept['action'] == 'left_turn' and retained is state
    assert kept['suppressed_early_heading_release']['selected_action'] == 'right_turn'
    assert {k:v for k,v in kept.items() if k != 'suppressed_early_heading_release'} == before
    assert result == before and state == previous


def test_blocked_recovery_still_holds_and_measured_completion_still_clears():
    selected = selection()
    _, state = recover_turn(selected, 0., 0, None)
    left = next(r for r in selected['memory_forecast_candidates'] if r['action'] == 'left_turn')
    left.update(minimum_predicted_path_clearance_m=.44, segment_clearances_m=[.44]*8)
    result, state = recover_turn(selected, .2, 0, state)
    kept, retained = retain_recovery_heading(result, state)
    assert kept is result and kept['action'] == 'hold' and retained is state
    result, state = recover_turn(selected, -.8, 0, state)
    assert state is None and result['clearance_turn']['event'] == 'MEASURED_TARGET_HEADING_REACHED'
    assert retain_recovery_heading(result, state) == (result, None)


def test_composed_runtime_replaces_only_release_layer(monkeypatch):
    class Runtime(NoEarlyHeadingReleaseMixin, FullReserveHeadingReleaseRuntime):
        def _select_clear_prediction(self, *args, **kwargs):
            self.outer_called = True
            return super()._select_clear_prediction(*args, **kwargs)

    result, state = releasable()
    def lower(self, *args, **kwargs):
        self.lower_called = True
        return result
    monkeypatch.setattr(ArrivalEntryTerminalPriorityRuntime, '_select_clear_prediction', lower)
    runtime = object.__new__(Runtime)
    runtime.terminal_position_approach = False
    runtime.clearance_turn = state
    kept = runtime._select_clear_prediction()
    assert runtime.outer_called and runtime.lower_called
    assert kept['action'] == 'left_turn' and runtime.clearance_turn is state
    runtime.terminal_position_approach = True
    assert runtime._select_clear_prediction() is result
    original = object.__new__(FullReserveHeadingReleaseRuntime)
    original.terminal_position_approach = False
    original.clearance_turn = state
    assert original._select_clear_prediction()['action'] == 'right_turn'
    assert original.clearance_turn is None
