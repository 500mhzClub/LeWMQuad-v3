import numpy as np
from lewm.observed_floor_waypoint_development import propose, segment_cells, inflated_cells
from lewm.joint_visual_floor_map_development import floor_coverage
from lewm.tests.test_floor_footprint_bounds_development import scene


def test_observed_detour_routes_around_inflated_wall_without_diagonal_shortcuts():
    floor = {(x, y) for x in range(-5, 40) for y in range(-20, 21)}
    wall = {(15, y) for y in range(-10, 11)}
    row = propose(floor, wall, [.025, .025], [1.525, .025], radius_m=.1)
    path = [tuple(c) for c in row['route_cells']]
    assert row['status'] == 'OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL'
    assert all(abs(a[0]-b[0])+abs(a[1]-b[1]) == 1 for a,b in zip(path,path[1:]))
    assert not set(path)&inflated_cells(wall,.1) and max(abs(y) for x,y in path) > 10
    assert row['complete_route_floor_coverage'] and not row['motion_permitted']


def test_unknown_connector_is_reported_and_never_crosses_known_obstacle():
    floor = {(20,0),(21,0)}
    row = propose(floor,set(),[.025,.025],[1.075,.025],radius_m=.1)
    assert row['initial_connector_requires_observation'] and row['unknown_connector_cells']
    assert not row['complete_route_floor_coverage'] and floor == {(20,0),(21,0)}
    blocked = propose(floor,{(10,0)},[.025,.025],[1.075,.025],radius_m=.1)
    assert blocked['status'] == 'ADDITIONAL_VIEW_REQUIRED' and not blocked['route_cells']


def test_closed_connector_contains_both_sides_of_gridline_and_corner():
    cells = segment_cells([0.,0.],[.1,.1])
    assert {(0,0),(0,1),(1,0),(1,1),(2,2)} <= cells
    assert segment_cells([0.,0.],[.1,.1]) == segment_cells([.1,.1],[0.,0.])


def test_floor_hypothesis_requires_real_complete_depth_coverage():
    depth,valid = scene();cells=np.array([[30,0]],np.int64)
    row=floor_coverage(depth,valid,np.eye(3),np.zeros(3),-.3,cells)
    assert row['covered'][0]
    a,b=row['projected_lower_xy'][0],row['projected_upper_xy'][0]
    x,y=(a+b)//2;depth[y,x]=0.;valid[y,x]=False
    assert not floor_coverage(depth,valid,np.eye(3),np.zeros(3),-.3,cells)['covered'][0]
    assert not floor_coverage(np.zeros_like(depth),np.zeros_like(valid),np.eye(3),np.zeros(3),-.3,cells)['covered'][0]


def test_wrong_plane_and_out_of_view_cells_cannot_gain_floor_coverage():
    depth,valid=scene()
    assert not floor_coverage(depth,valid,np.eye(3),np.zeros(3),-.2,np.array([[30,0]]))['covered'].any()
    assert not floor_coverage(depth,valid,np.eye(3),np.zeros(3),-.3,np.array([[-10,0],[5,0],[30,90]]))['covered'].any()
