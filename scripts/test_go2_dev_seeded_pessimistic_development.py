"""Synthetic checks for the precondition-seeded pessimistic-unknown rule (development, 2 October 2026).

The start pose with only the floor ahead observed (what the depth cameras see there). An in-place
turn whose forecast centre drifts 1 cm backwards (C1's panorama turns drift about 1 cm):
- C1 p99 bound (blocking radius 0.454 m), start disc seeded at the 0.5-m precondition: passes;
- the same with the start disc seeded at the blocking radius only: blocked (why the seed is the
  precondition, not the blocking radius);
- C3 p99 bound (blocking radius 0.497 m) at the 0.5-m precondition: blocked even without drift,
  because cells straddling the 0.5-m boundary stay unknown and start 1.4 cm inside it;
- C3 p95 bound (0.479 m) at the precondition, without drift: passes; with 1-cm drift: blocked.
Unseen cells beyond the precondition still block a translation heading into them.
"""
import types

import numpy as np

from lewm import dev_harness_fixes_development as fixes
from lewm import memory_forecast_clearance_development as memory_check
from lewm.clearance_turn_recovery_development import reserve_turns
from lewm.dev_pessimistic_unknown_seeded_development import PRECONDITION_CLEAR_M, seeded_pessimistic_unknown_mixin
from lewm.geometry_progress_pilot_development import ACTIONS

AHEAD = frozenset((i, j) for i in range(9, 40) for j in range(-15, 16))  # floor seen ahead, x >= 0.45 m


def prediction(drift_m):
    p = np.zeros((6, 8, 5))
    p[..., 3] = 1.
    p[ACTIONS.index('forward'), :, 0] = np.linspace(.0125, .1, 8)
    for turn in ('left_turn', 'right_turn'):
        p[ACTIONS.index(turn), :, 0] = -np.linspace(0, drift_m, 8)
    return p


def selection(prefer='left_turn'):
    return dict(action=prefer, action_index=ACTIONS.index(prefer),
                candidates=[dict(action=a, utility_m=1. if a == prefer else .1) for a in ACTIONS])


class Base:
    # As in the frozen chain: the memory check with reserve recovery, then the stepwise 0.03-m turn reserve (turns need 0.48 m).
    def _select_clear_prediction(self, selected, prediction_, snapshot, position, rotation):
        checked = memory_check.select_clear_prediction(selected, prediction_, snapshot.fine_occupied, position, rotation,
                                                       translation_reserve_m=.03, reserve_recovery=True)
        return reserve_turns(checked, stepwise=True)

    def _route_target(self, route, snapshot, position):
        return fixes._UNKNOWN_CELLS.get()


def run(bound, seed, drift, floor=AHEAD, prefer='left_turn'):
    runtime = type('S', (seeded_pessimistic_unknown_mixin(bound, 'test', seed), Base), {})()
    snap = types.SimpleNamespace(floor=floor, occupied=frozenset(), fine_occupied=frozenset())
    result = runtime._select_clear_prediction(selection(prefer), prediction(drift), snap, np.zeros(3), np.eye(3))
    assert fixes._UNKNOWN_CELLS.get() is None
    return result


def main():
    cases = (('C1 p99, seed 0.5 m precondition, 1-cm turn drift', .0289, PRECONDITION_CLEAR_M, .01, 'left_turn'),
             ('C1 p99, seed at blocking radius 0.454 m only, 1-cm turn drift', .0289, .454, .01, 'hold'),
             ('C3 p99, seed 0.5 m precondition, no drift', .0719, PRECONDITION_CLEAR_M, 0., 'hold'),
             ('C3 p95, seed 0.5 m precondition, no drift', .0539, PRECONDITION_CLEAR_M, 0., 'left_turn'),
             ('C3 p95, seed 0.5 m precondition, 1-cm turn drift', .0539, PRECONDITION_CLEAR_M, .01, 'hold'))
    for name, bound, seed, drift, expected in cases:
        result = run(bound, seed, drift)
        row = {r['action']: r for r in result['memory_forecast_candidates']}['left_turn']
        print(f"{name}: {result['dev_pessimistic_unknown']['unknown_cells_in_scope']} unseen cells; left-turn effective clearance "
              f"{row['minimum_predicted_path_clearance_m']:.4f} (blocking at < 0.48) -> {result['action']}")
        assert result['action'] == expected, (name, result['action'])
    # A translation into unseen space beyond the precondition is still blocked: nothing observed except a strip behind.
    behind = frozenset((i, j) for i in range(-20, 0) for j in range(-20, 21))
    result = run(.0289, PRECONDITION_CLEAR_M, 0., floor=behind, prefer='forward')
    assert result['action'] != 'forward'
    print(f"C1 p99, forward into unseen space ahead: forward blocked -> {result['action']}")


if __name__ == '__main__':
    main()
