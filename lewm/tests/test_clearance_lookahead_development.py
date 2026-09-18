import numpy as np
from lewm.clearance_lookahead_development import clear_route_target


def test_bend_shortcut_is_stopped_before_obstacle():
    # Route initially moves away, then bends around a cell at x=.5, y=.5.
    # The old radial lookahead jumps diagonally toward that cell.
    points=[[-.1,0.],[-.1,.1],[-.1,.2],[.1,.35]]
    target,receipt=clear_route_target(points,[0.,0.],{(50,50)})
    np.testing.assert_allclose(target,[-.1,.2])
    assert receipt['original_shortcut_clearance_m']<.48
    assert receipt['chosen_shortcut_clearance_m']>=.48
    assert receipt['target_changed']


def test_available_clearance_and_empty_map_do_not_prevent_escape():
    points=[[-.1,0.],[-.2,0.],[-.4,0.]]
    target,receipt=clear_route_target(points,[0.,0.],{(47,0)})
    np.testing.assert_allclose(target,[-.4,0.])
    np.testing.assert_allclose(receipt['required_shortcut_clearance_m'],.47,rtol=0,atol=1e-12)
    target,receipt=clear_route_target(points,[0.,0.],set())
    np.testing.assert_allclose(target,[-.4,0.])
    assert receipt['chosen_shortcut_clearance_m'] is None
def test_final_route_target_uses_exact_goal_within_cell():
    from types import SimpleNamespace
    from lewm.clearance_lookahead_development import ClearanceLookaheadRuntime
    route=dict(status='OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL',route_cells=[[0,0],[0,1]],goal_map_xy_m=[.01,.06])
    runtime=object.__new__(ClearanceLookaheadRuntime)
    target=runtime._route_target(route,SimpleNamespace(fine_occupied=frozenset()),[.01,.02])
    assert target.tolist()==[.01,.06]
