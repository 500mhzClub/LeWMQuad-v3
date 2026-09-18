from copy import deepcopy
from types import SimpleNamespace
import numpy as np

from lewm.cached_fine_connectivity_development import fine_goal_route,nearby_candidates,search_graph
from lewm.cached_fine_goal_route_development import cached_fine_goal_route as reference
from lewm.observed_floor_waypoint_development import centre


def snapshot(floor,fine=()):
    return SimpleNamespace(floor=frozenset(floor),fine_occupied=frozenset(fine),
        occupied=frozenset((x//5,y//5) for x,y in fine))


def route(radius=.01):
    return dict(status='OBSERVED_FLOOR_ROUTE_TO_FRONTIER',nominal_radius_m=radius,
        route_cells=[[0,0]],retained_field=True)


def without_time(result):
    result=deepcopy(result)
    if 'fine_goal_route' in result:result['fine_goal_route'].pop('routing_s')
    return result


def compare(s,p,g,r):
    old=reference(s,np.asarray(p),np.asarray(g),r)
    new=fine_goal_route(s,np.asarray(p),np.asarray(g),r)
    assert without_time(old)==without_time(new)
    return new


def test_cached_failure_invalidates_when_observed_floor_connects_components():
    search_graph.cache_clear();p=[.025,.025];g=[.225,.025]
    s=snapshot([(0,0),(1,0),(3,0),(4,0)])
    assert compare(s,p,g,route())['status']=='OBSERVED_FLOOR_ROUTE_TO_FRONTIER'
    hits=search_graph.cache_info().hits
    compare(s,p,g,route());assert search_graph.cache_info().hits>hits
    s=snapshot(s.floor|{(2,0)})
    assert compare(s,p,g,route())['status']=='OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL'


def test_obstacle_change_invalidates_clear_path_and_current_connector_is_rechecked():
    cells={(x,0) for x in range(5)};p=[.025,.025];g=[.225,.025]
    assert compare(snapshot(cells),p,g,route())['status']=='OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL'
    s=snapshot(cells,[(10,2)])
    assert compare(s,p,g,route())['status']=='OBSERVED_FLOOR_ROUTE_TO_FRONTIER'
    assert compare(s,[.105,.025],g,route())['status']=='OBSERVED_FLOOR_ROUTE_TO_FRONTIER'


def test_tied_paths_and_returned_path_mutation_do_not_change_cached_results():
    s=snapshot((x,y) for x in range(6) for y in range(6));p=[.025,.025];g=[.225,.225]
    first=compare(s,p,g,route());expected=deepcopy(first['route_cells'])
    first['route_cells'].clear()
    assert compare(s,p,g,route())['route_cells']==expected
    compare(s,p,[.225,.226],route())
    compare(s,[.075,.025],g,route())
    compare(s,p,g,route(.03))


def test_nearby_index_preserves_exact_radius_membership_and_tie_order():
    floor=frozenset((x,y) for x in range(-30,31) for y in range(-30,31))
    for p in (np.array([.025,.025]),np.array([.025+1e-13,.025]),np.array([.13,-.08])):
        expected=sorted((c for c in floor if np.linalg.norm(centre(c)-p)<=1.25),
            key=lambda c:(float(np.linalg.norm(centre(c)-p)),c))
        assert nearby_candidates(floor,p)==expected
