from types import SimpleNamespace
import numpy as np

from lewm.camera_frontier_viewpoint_development import (
    floor_cell_projection, choose_route_viewpoint)


def test_near_body_patch_is_invisible_but_distant_patch_can_be_viewed():
    near = floor_cell_projection([2, 0], np.zeros(3), np.eye(3), -.32)
    ahead = floor_cell_projection([12, 0], np.zeros(3), np.eye(3), -.32)
    behind = floor_cell_projection([-12, 0], np.zeros(3), np.eye(3), -.32)
    assert not any(r['fully_projected'] for r in near+behind)
    assert next(r for r in ahead if r['camera'] == 'auxiliary')['fully_projected']


def fixture(occupied=()):
    snapshot = SimpleNamespace(floor=frozenset((i, 0) for i in range(16)),
        occupied=frozenset(occupied), floor_height=-.32)
    route = dict(status='OBSERVED_FLOOR_ROUTE_TO_FRONTIER',
        route_cells=[[i, 0] for i in range(11)], target_map_xy_m=[.525, .025])
    return snapshot, route


def test_viewpoint_stays_on_existing_route_without_admitting_unknown_floor():
    snapshot, route = fixture(); original_floor = snapshot.floor
    view = choose_route_viewpoint(snapshot, route, np.zeros(3), np.eye(3), [16, 0])
    assert view is not None and view['route_cells'] == route['route_cells'][:len(view['route_cells'])]
    assert tuple(view['viewpoint_cell']) in original_floor
    assert snapshot.floor is original_floor and (16, 0) not in snapshot.floor
    assert view['motion_authorized'] is False and view['unknown_floor_admitted'] is False
    assert all(row['fully_projected'] for row in view['projected_cameras'])


def test_known_occluder_prevents_viewpoint_and_observed_target_needs_no_probe():
    snapshot, route = fixture(occupied=[(15, 0)])
    assert choose_route_viewpoint(snapshot, route, np.zeros(3), np.eye(3), [16, 0]) is None
    snapshot, route = fixture()
    assert choose_route_viewpoint(snapshot, route, np.zeros(3), np.eye(3), [8, 0]) is None
