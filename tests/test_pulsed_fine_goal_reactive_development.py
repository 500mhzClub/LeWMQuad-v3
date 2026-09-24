from types import SimpleNamespace
import numpy as np
from lewm.pulsed_fine_goal_reactive_development import PulsedFineGoalReactiveRuntime


def select(*, near, scan=None):
    runtime = object.__new__(PulsedFineGoalReactiveRuntime)
    runtime.terminal_position_approach = near
    # This object has no model, predictor, or motion-residual fit.
    selected, correction = runtime._select_action(None, None, [[0.,0.,0.]]*3,
        np.array([.05,0.]), scan, SimpleNamespace(fine_occupied=frozenset()),
        np.zeros(3), np.eye(3))
    assert correction is None
    assert selected['candidate_future_outcomes_evaluated'] is False
    return selected


def test_near_goal_reactive_translation_uses_matched_short_window():
    row = select(near=True)
    assert row['action'] == 'forward'
    assert row['command_duration_ns'] == 100_000_000


def test_far_translation_and_survey_turn_keep_full_window():
    assert select(near=False)['command_duration_ns'] == 400_000_000
    row = select(near=True, scan=.8)
    assert row['action'] == 'left_turn'
    assert row['command_duration_ns'] == 400_000_000
