import numpy as np
import pytest

from lewm.prefix_aware_terminal_approach_development import PrefixAwareTerminalMixin, terminal_radius


def test_prefix_expansion_counts_translation_not_rotation():
    assert terminal_radius([[.16, 0, .45]]*3) == pytest.approx(.148)
    assert terminal_radius([[0, 0, .45]]*3) == .1
    with pytest.raises(ValueError):
        terminal_radius([[float('nan'), 0, 0]]*3)


class Parent:
    def _select_action(self, *args):
        return {'mode_seen_by_forecaster': self.terminal_position_approach}, None


class Probe(PrefixAwareTerminalMixin, Parent):
    pass


@pytest.mark.parametrize('exact,scan,distance,expected', [
    (True, None, .122, True), (True, None, .2, False),
    (False, None, .122, False), (True, .5, .122, False)])
def test_only_direct_goal_approach_enters_pulse_mode_early(exact, scan, distance, expected):
    runtime = Probe()
    runtime.terminal_position_approach = False
    runtime.exact_terminal_target = exact
    runtime.terminal_target_distance = distance
    selected, _ = runtime._select_action(None, None, [[.16, 0, .45]]*3,
        None, scan, None, None, None)
    assert selected['mode_seen_by_forecaster'] is expected
    assert selected['prefix_aware_terminal_approach']['newly_enabled'] is expected
