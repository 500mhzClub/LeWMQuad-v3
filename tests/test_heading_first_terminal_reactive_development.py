from types import SimpleNamespace

import numpy as np

from lewm.heading_first_terminal_reactive_development import HeadingFirstTerminalReactiveMixin
from lewm.pulsed_fine_goal_reactive_development import PulsedFineGoalReactiveRuntime


class Runtime(HeadingFirstTerminalReactiveMixin, PulsedFineGoalReactiveRuntime):
    pass


def select(goal, *, near=True, scan=None, occupied=frozenset()):
    runtime = object.__new__(Runtime)
    runtime.terminal_position_approach = near
    runtime.mission = SimpleNamespace(arrival_radius_m=.02)
    row, correction = runtime._select_action(None, None, [[0.,0.,0.]]*3,
        np.array(goal), scan, SimpleNamespace(fine_occupied=occupied),
        np.zeros(3), np.eye(3))
    assert correction is None and row['candidate_future_outcomes_evaluated'] is False
    return row


def test_sideways_terminal_target_uses_full_turn_then_aligned_short_forward():
    turn = select([.023, -.052])
    assert turn['action'] == 'right_turn' and turn['command_duration_ns'] == 400_000_000
    assert not turn['terminal_translation_pulse']['selected_translation_pulse']
    forward = select([.05, 0.])
    assert forward['action'] == 'forward' and forward['command_duration_ns'] == 100_000_000
    assert forward['terminal_translation_pulse']['selected_translation_pulse']


def test_near_arrival_holds_but_does_not_declare_success():
    row = select([.015, 0.])
    assert row['action'] == 'hold' and row['command_duration_ns'] == 400_000_000
    assert row['heading_first_terminal']['arrival_still_requires_measured_quiet_dwell']


def test_current_clearance_veto_scan_and_nonterminal_selection_are_preserved():
    blocked = select([.05, 0.], occupied=frozenset({(0,0)}))
    assert blocked['action'] == 'hold' and 'heading_first_terminal' not in blocked
    survey = select([.05,0.], scan=.8)
    assert survey['action'] == 'left_turn' and 'heading_first_terminal' not in survey
    far = select([.023,-.052], near=False)
    assert far['action'] == 'right_arc' and far['command_duration_ns'] == 400_000_000
