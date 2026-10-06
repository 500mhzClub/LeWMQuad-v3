"""Synthetic checks for the look-around exemption (development, 2 October 2026; Andrew's option A).

At the start pose with only the floor ahead observed, using C3's p95 bound (blocking radius
0.479 m) and a turn whose forecast centre drifts 4 cm (C3's start-turn p95):
- during the scripted look-around the turn passes (unseen cells exempt);
- with the centre 8 cm from the start (beyond the 7-cm limit) the exemption ends and stays ended;
- after the look-around completes the rule applies in full (the same turn is blocked);
- remembered walls still block during the look-around.
"""
import types

import numpy as np

from lewm import dev_harness_fixes_development as fixes
from lewm import memory_forecast_clearance_development as memory_check
from lewm.clearance_turn_recovery_development import reserve_turns
from lewm.dev_pessimistic_unknown_lookaround_development import LOOKAROUND, lookaround_exempt_pessimistic_unknown_mixin
from lewm.geometry_progress_pilot_development import ACTIONS

AHEAD = frozenset((i, j) for i in range(9, 40) for j in range(-15, 16))


def prediction(drift_m):
    p = np.zeros((6, 8, 5))
    p[..., 3] = 1.
    p[ACTIONS.index('forward'), :, 0] = np.linspace(.0125, .1, 8)
    for turn in ('left_turn', 'right_turn'):
        p[ACTIONS.index(turn), :, 0] = -np.linspace(0, drift_m, 8)
    return p


def selection():
    return dict(action='left_turn', action_index=ACTIONS.index('left_turn'),
                candidates=[dict(action=a, utility_m=1. if a == 'left_turn' else .1) for a in ACTIONS])


class Base:
    def __init__(self):
        self.status = LOOKAROUND

    def _route(self, *args, **kwargs):
        return dict(status=self.status)

    def _select_clear_prediction(self, selected, prediction_, snapshot, position, rotation):
        checked = memory_check.select_clear_prediction(selected, prediction_, snapshot.fine_occupied, position, rotation,
                                                       translation_reserve_m=.03, reserve_recovery=True)
        return reserve_turns(checked, stepwise=True)

    def _route_target(self, route, snapshot, position):
        return fixes._UNKNOWN_CELLS.get()


def step(runtime, position, wall=frozenset(), drift=.04):
    runtime._route()
    snap = types.SimpleNamespace(floor=AHEAD, occupied=frozenset(), fine_occupied=wall)
    result = runtime._select_clear_prediction(selection(), prediction(drift), snap, np.asarray(position, float), np.eye(3))
    assert fixes._UNKNOWN_CELLS.get() is None
    return result['action'], result['dev_pessimistic_unknown']


def main():
    make = lambda: type('L', (lookaround_exempt_pessimistic_unknown_mixin(.0539, 'p95'), Base), {})()
    runtime = make()
    action, record = step(runtime, (0., 0., 0.))
    assert action == 'left_turn' and record['lookaround_exempt'], record
    print('look-around, C3 p95 bound, 4-cm turn drift: unseen cells exempt -> left_turn')
    action, record = step(runtime, (.08, 0., 0.))
    assert not record['lookaround_exempt'] and record['lookaround_exemption_ended'] == 'centre drift beyond limit', record
    action, record = step(runtime, (0., 0., 0.))
    assert not record['lookaround_exempt'] and action == 'hold', (action, record)
    print('centre 8 cm from the start: exemption ends for good; back at the start the turn is blocked -> hold')
    runtime = make()
    step(runtime, (0., 0., 0.))
    runtime.status = 'OBSERVED_FLOOR_ROUTE_TO_FRONTIER'
    action, record = step(runtime, (0., 0., 0.))
    assert not record['lookaround_exempt'] and record['lookaround_exemption_ended'] == 'look-around complete' and action == 'hold', (action, record)
    print('after the look-around completes the rule applies in full -> hold')
    runtime = make()
    action, record = step(runtime, (0., 0., 0.), wall=frozenset((44, j) for j in range(-60, 61)), drift=0.)
    assert record['lookaround_exempt'] and action == 'hold', (action, record)
    print('during the look-around a remembered wall 0.44 m away still blocks the turn -> hold')


if __name__ == '__main__':
    main()
