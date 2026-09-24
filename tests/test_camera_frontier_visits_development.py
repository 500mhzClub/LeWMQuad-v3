from types import SimpleNamespace
import numpy as np

from lewm.camera_frontier_visits_development import CameraFrontierVisits


def fixture(position=(0., 0.), *, near=False):
    floor = frozenset((x, y) for x in range(11) for y in range(-5, 6))
    snapshot = SimpleNamespace(floor=floor, occupied=frozenset(), fine_occupied=frozenset(),
        floor_height=-.32, measured_ns=0, frame=0)
    route = dict(status='OBSERVED_FLOOR_ROUTE_TO_FRONTIER',
        route_cells=[[x, 0] for x in range(9 if near else 0, 11)],
        target_map_xy_m=[.525, .025], goal_map_xy_m=[2., .025])
    visits = CameraFrontierVisits()
    visits.context = (np.r_[position, 0.], np.eye(3))
    return visits, snapshot, route


def test_completed_turn_does_not_observe_unknown_patch():
    visits, snapshot, route = fixture(); floor = snapshot.floor
    action, receipt = visits.advance(snapshot, np.zeros(2), 0., np.array([2., .025]), 100, route)
    assert action == 'view' and visits.visit is not None
    assert receipt['aligned_ns'] == 100
    assert snapshot.floor is floor and (11, 0) not in floor
    snapshot.measured_ns = 100; snapshot.frame = 1
    action, _ = visits.advance(snapshot, np.zeros(2), 0., np.array([2., .025]), 200, route)
    assert action == 'replan' and visits.visit is None
    assert visits.events[-1]['completion_reason'] == 'FRESH_VIEW_PATCH_STILL_UNKNOWN'
    assert not visits.events[-1]['unknown_cell_observed'] and not visits.excluded
    assert visits.attempted[(11, 0)] and snapshot.floor is floor


def test_only_mapped_evidence_resolves_requested_patch():
    visits, snapshot, route = fixture()
    visits.advance(snapshot, np.zeros(2), 0., np.array([2., .025]), 100, route)
    snapshot.floor = snapshot.floor | {(11, 0)}; snapshot.measured_ns = 200; snapshot.frame = 2
    action, _ = visits.advance(snapshot, np.zeros(2), 0., np.array([2., .025]), 300, route)
    assert action == 'replan' and visits.events[-1]['unknown_cell_observed']
    assert visits.events[-1]['completion_reason'] == 'REQUESTED_PATCH_OBSERVED'


def test_near_body_blind_region_routes_back_to_known_viewpoint():
    p = np.array([.48, 0.]); visits, snapshot, route = fixture(p, near=True)
    action, _ = visits.advance(snapshot, p, 0., np.array([2., .025]), 100, route)
    assert action == 'route' and visits.visit is not None
    assert route['target_map_xy_m'][0] < p[0]-.2
    assert all(tuple(c) in snapshot.floor for c in route['route_cells'])
    assert visits.visit['camera_viewpoint']['known_floor_backtrack_search']
    assert (11, 0) not in snapshot.floor and not visits.excluded


def test_new_mission_goal_route_releases_unfinished_view():
    visits, snapshot, route = fixture()
    visits.advance(snapshot, np.zeros(2), 0., np.array([2., .025]), 100, route)
    route['status'] = 'OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL'
    action, _ = visits.advance(snapshot, np.zeros(2), 0., np.array([2., .025]), 200, route)
    assert action == 'route' and visits.visit is None
    assert visits.events[-1]['completion_reason'] == 'MISSION_GOAL_ROUTE_AVAILABLE'
