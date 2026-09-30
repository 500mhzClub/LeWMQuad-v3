"""Synthetic check of the C2 terminal spin break (development).

A target 4 cm right-front keeps the reactive terminal rule turning right. After
TERMINAL_SPIN_TURNS consecutive right turns the fix must issue one 100-ms forward pulse, then
hand control back to the rule. A C1/C3-style selection (no reactive rule) is left untouched.
"""
from lewm.dev_harness_fixes_development import TERMINAL_SPIN_TURNS, TerminalPositionScoringMixin
from lewm.persistent_visual_baselines_development import select_reactive_terminal


class Base:
    def __init__(self, result):
        self.result = result

    def _select_clear_prediction(self, *args):
        return dict(self.result, terminal_translation_pulse=dict(self.result['terminal_translation_pulse']))


class T(TerminalPositionScoringMixin, Base):
    pass


def main():
    frozen = select_reactive_terminal([.03, -.03], scan_error=None, clearance_m=.6, pulse=True, arrival_radius_m=.02)
    assert frozen['action'] == 'right_turn', frozen['action']
    t = T(frozen)
    actions = [t._select_clear_prediction(None, None, None, None, None)['action'] for _ in range(2*TERMINAL_SPIN_TURNS)]
    assert actions[:TERMINAL_SPIN_TURNS-1] == ['right_turn']*(TERMINAL_SPIN_TURNS-1), actions
    assert actions[TERMINAL_SPIN_TURNS-1] == 'forward' and actions[TERMINAL_SPIN_TURNS] == 'right_turn', actions
    pulse = T(frozen)
    for _ in range(TERMINAL_SPIN_TURNS):
        out = pulse._select_clear_prediction(None, None, None, None, None)
    assert out['command_duration_ns'] == 100_000_000 and out['terminal_translation_pulse']['selected_translation_pulse']
    other = T(dict(action='right_turn', candidates=[], terminal_translation_pulse={}))
    assert all(other._select_clear_prediction(None, None, None, None, None)['action'] == 'right_turn' for _ in range(3*TERMINAL_SPIN_TURNS))
    print('spin break after', TERMINAL_SPIN_TURNS, 'turns; non-reactive selections unchanged')


if __name__ == '__main__':
    main()
