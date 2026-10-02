"""Synthetic checks for the calibrated clearance margin (development, 2 October 2026).

A remembered wall 0.50 m ahead of the robot. Nominal: an in-place turn (centre path at the
origin) clears the 0.45-m disc. Inside the margin context (6 cm) the same turn is blocked,
because every requirement becomes r + margin; the logged distance is reduced by the margin.
The stopping projection's and routing's bindings are untouched, and outside the context the
behaviour is the frozen one. The mixin records the margin in the selection.

Pessimistic unknown (stage 2): at the start pose with only the floor ahead observed (what the
cameras see), the never-observed ring between the seeded 0.425-m start disc and 0.5 m counts
as occupied, so an in-place turn is blocked; with the whole reach observed it passes.
"""
import types
import numpy as np

from lewm import dev_harness_fixes_development as fixes
from lewm import clearance_lookahead_development as lookahead
from lewm import fine_stored_obstacle_routing_development as routing
from lewm import memory_forecast_clearance_development as memory_check
from lewm import planned_stopping_projection_development as stopping
from lewm.geometry_progress_pilot_development import ACTIONS

WALL = frozenset((50, j) for j in range(-60, 61))  # 1-cm cells, near face at x = 0.50 m


def prediction():
    p = np.zeros((6, 8, 5))
    p[..., 3] = 1.
    p[ACTIONS.index('forward'), :, 0] = np.linspace(.0125, .1, 8)
    return p


def selection():
    return dict(action='left_turn', action_index=ACTIONS.index('left_turn'),
                candidates=[dict(action=a, utility_m=1. if a == 'left_turn' else .1) for a in ACTIONS])


def check(context_margin):
    token = fixes._CLEARANCE_MARGIN.set(context_margin)
    try:
        return memory_check.select_clear_prediction(selection(), prediction(), WALL, np.zeros(3), np.eye(3), translation_reserve_m=.03)
    finally:
        fixes._CLEARANCE_MARGIN.reset(token)


def main():
    nominal = check(0.)
    rows = {r['action']: r for r in nominal['memory_forecast_candidates']}
    assert abs(rows['left_turn']['minimum_predicted_path_clearance_m']-.50) < 1e-9 and rows['left_turn']['nominal_predicted_path_clear']
    assert nominal['action'] == 'left_turn'
    margin = check(.06)
    rows = {r['action']: r for r in margin['memory_forecast_candidates']}
    assert abs(rows['left_turn']['minimum_predicted_path_clearance_m']-.44) < 1e-9 and not rows['left_turn']['nominal_predicted_path_clear']
    assert margin['action'] == 'hold'
    print('margin 0.06 m: turn with 0.50 m remembered clearance passes nominally and is blocked inside the margin (requirement 0.45 -> 0.51 m)')
    assert stopping.cached_clearance is routing.cached_clearance and routing.cached_clearance is not fixes._margin_cached_clearance
    assert memory_check.cached_clearance is fixes._margin_cached_clearance and lookahead.cached_clearance is fixes._margin_cached_clearance
    token = fixes._CLEARANCE_MARGIN.set(.06)
    try:
        assert abs(stopping.cached_clearance(WALL).minimum(np.zeros(2), np.zeros(2))-.50) < 1e-9
        target, receipt = lookahead.clear_route_target([np.array([.0, .3]), np.array([.0, .6])], np.zeros(2), WALL)
        assert abs(receipt['start_clearance_m']-.44) < 1e-9
    finally:
        fixes._CLEARANCE_MARGIN.reset(token)
    print('stopping projection and routing bindings unchanged; route-target lookahead inflated')
    again = check(0.)
    assert again == nominal
    print('outside the margin context the check is identical to the frozen one')

    class Base:
        def _select_clear_prediction(self, selected, prediction_, snapshot, position, rotation):
            return memory_check.select_clear_prediction(selected, prediction_, WALL, position, rotation, translation_reserve_m=.03)

        def _route_target(self, *args):
            return fixes._CLEARANCE_MARGIN.get()

    runtime = type('R', (fixes.margin_mixin(.06, 'test'), Base), {})()
    result = runtime._select_clear_prediction(selection(), prediction(), None, np.zeros(3), np.eye(3))
    assert result['action'] == 'hold' and result['dev_clearance_margin']['margin_m'] == .06
    assert runtime._route_target() == .06 and fixes._CLEARANCE_MARGIN.get() == 0.
    print('margin_mixin applies the margin to the action check and route target, records it, and restores the context')
    for bad in (0., -.01, .2, float('nan')):
        try:
            fixes.margin_mixin(bad, 'bad')
        except ValueError:
            continue
        raise AssertionError(f'margin {bad} accepted')


def pessimistic():
    class Base:
        def _select_clear_prediction(self, selected, prediction_, snapshot, position, rotation):
            return memory_check.select_clear_prediction(selected, prediction_, snapshot.fine_occupied, position, rotation, translation_reserve_m=.03)

        def _route_target(self, route, snapshot, position):
            return len(fixes._UNKNOWN_CELLS.get())

    ahead = frozenset((i, j) for i in range(9, 40) for j in range(-15, 16))  # floor seen ahead, x >= 0.45 m
    everywhere = frozenset((i, j) for i in range(-20, 40) for j in range(-20, 21))
    for floor, expected in ((ahead, 'hold'), (everywhere, 'left_turn')):
        runtime = type('P', (fixes.PessimisticUnknownMixin, Base), {})()
        snap = types.SimpleNamespace(floor=floor, occupied=frozenset(), fine_occupied=frozenset())
        result = runtime._select_clear_prediction(selection(), prediction(), snap, np.zeros(3), np.eye(3))
        assert result['action'] == expected, (result['action'], result['dev_pessimistic_unknown'])
        count = result['dev_pessimistic_unknown']['unknown_cells_counted_occupied']
        print(f"pessimistic unknown, {'floor ahead only' if floor is ahead else 'whole reach observed'}: "
              f"{count} unknown cells counted occupied -> {result['action']}")
        assert fixes._UNKNOWN_CELLS.get() == frozenset()
    assert stopping.cached_clearance is routing.cached_clearance


if __name__ == '__main__':
    main()
    pessimistic()
