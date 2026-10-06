"""Synthetic checks of the terminal spin breaks (development).

C1/C3/C4: within 0.10 m, after TERMINAL_SPIN_TURNS consecutive scored turns, exactly
TERMINAL_BURST_DECISIONS decisions are chosen by position utility, then the frozen scorer
resumes; outside the radius nothing changes.


A target 4 cm right-front keeps the reactive terminal rule turning right. After
TERMINAL_SPIN_TURNS consecutive right turns the fix must issue one 100-ms forward pulse, then
hand control back to the rule. A C1/C3-style selection (no reactive rule) is left untouched.
"""
from lewm.dev_harness_fixes_development import TERMINAL_BURST_DECISIONS, TERMINAL_SPIN_TURNS, TerminalPositionScoringMixin
from lewm.persistent_visual_baselines_development import select_reactive_terminal


class Base:
    def __init__(self, result):
        self.result = result

    def _select_clear_prediction(self, *args):
        return dict(self.result, terminal_translation_pulse=dict(self.result['terminal_translation_pulse']))


class T(TerminalPositionScoringMixin, Base):
    pass


class Scorer:
    def _score(self, prediction, goal_body, **kwargs):
        return dict(action='right_turn', candidates=[dict(action='right_turn', utility_m=.01, position_contact_utility_m=0.),
                                                     dict(action='forward', utility_m=0., position_contact_utility_m=.01)])


class S(TerminalPositionScoringMixin, Scorer):
    pass


def check_planner_burst():
    s = S()
    near = [s._score(None, [.04, 0.])['action'] for _ in range(TERMINAL_SPIN_TURNS+TERMINAL_BURST_DECISIONS+2)]
    want = ['right_turn']*(TERMINAL_SPIN_TURNS-1)+['forward']*TERMINAL_BURST_DECISIONS+['right_turn']*3
    assert near == want, near
    far = S()
    assert all(far._score(None, [.2, 0.])['action'] == 'right_turn' for _ in range(3*TERMINAL_SPIN_TURNS))
    print('planner burst: position scoring for', TERMINAL_BURST_DECISIONS, 'decisions after', TERMINAL_SPIN_TURNS, 'terminal turns')


def main():
    check_planner_burst()
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
