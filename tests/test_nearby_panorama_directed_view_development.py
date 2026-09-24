from types import SimpleNamespace

import numpy as np

from lewm.nearby_panorama_directed_view_development import NearbyPanoramaFrontierVisits


def setup():
    state = NearbyPanoramaFrontierVisits(standoff_m=.5, panorama=True)
    snap = SimpleNamespace(floor=frozenset({(0, 0), (3, 0), (9, 0)}),
        occupied=frozenset(), measured_ns=100, frame=1)
    route = dict(status='OBSERVED_FLOOR_ROUTE_TO_FRONTIER',
        target_map_xy_m=[.025, .025], route_cells=[[0, 0]])
    position = np.array([.025, .025]); goal = np.array([1., 0.])
    return state, snap, route, position, goal


def complete_panorama(state, snap, route, position, goal):
    action, first = state.advance(snap, position, 0., goal, 200, route)
    assert action == 'view' and first['panoramic_frontier_view']
    for index in range(9):
        now = 300+200*index; heading = state.visit['heading_rad']
        assert state.advance(snap, position, heading, goal, now, route)[0] == 'view'
        snap.measured_ns = now; snap.frame += 1
        action, _ = state.advance(snap, position, heading, goal, now+100, route)
        assert action == ('replan' if index == 8 else 'view')
    assert len(state.events[-1]['completed_view_stages']) == 9
    assert first['completed_view_stages'] == []


def test_nearby_revisit_still_requires_heading_and_new_measured_map():
    state, snap, route, p, goal = setup()
    complete_panorama(state, snap, route, p, goal)
    original_floor = snap.floor
    route = route | dict(target_map_xy_m=[.175, .025], route_cells=[[0, 0], [3, 0]])
    _, receipt = state.advance(snap, p, 0., goal, 3000, route)
    assert 'nearby_completed_panorama' in receipt
    assert 'panoramic_frontier_view' not in receipt
    heading = state.visit['heading_rad']
    assert state.advance(snap, p, heading+1., goal, 3100, route)[0] == 'view'
    assert state.advance(snap, p, heading, goal, 3200, route)[0] == 'view'
    assert (3, 0) not in state.excluded
    snap.measured_ns = 3200
    assert state.advance(snap, p, heading, goal, 3300, route)[0] == 'replan'
    assert state.excluded == {(0, 0), (3, 0)}
    assert snap.floor is original_floor and (9, 0) not in state.excluded
    assert len(state.events) == 2 and state.panorama


def test_far_target_or_viewpoint_gets_full_panorama():
    for far_target in (True, False):
        state, snap, route, p, goal = setup()
        complete_panorama(state, snap, route, p, goal)
        target = [.475, .025] if far_target else [.175, .025]
        position = p if far_target else np.array([.35, .025])
        route = route | dict(target_map_xy_m=target, route_cells=[[9 if far_target else 3, 0]])
        action, receipt = state.advance(snap, position, 0., goal, 3000, route)
        assert action == 'view' and receipt['panoramic_frontier_view']
        assert 'nearby_completed_panorama' not in receipt


def test_incomplete_panorama_and_long_detour_do_not_shortcut():
    state, snap, route, p, goal = setup()
    state.advance(snap, p, 0., goal, 200, route)
    snap.measured_ns = 300
    _, receipt = state.advance(snap, p, state.visit['heading_rad'], goal, 300, route)
    assert receipt['view_index'] == 1 and not state.events
    assert 'nearby_completed_panorama' not in receipt
    state, snap, route, p, goal = setup()
    complete_panorama(state, snap, route, p, goal)
    route = route | dict(target_map_xy_m=[.175, .025],
        route_cells=[[0, 0], [0, 10], [3, 10], [3, 0]])
    assert state.advance(snap, p, 0., goal, 3000, route) == ('route', None)
    assert state.visit is None and state.excluded == {(0, 0)}
