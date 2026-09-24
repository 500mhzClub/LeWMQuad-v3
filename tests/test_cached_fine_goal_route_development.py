from types import SimpleNamespace
import numpy as np
from lewm.cached_fine_goal_route_development import cached_fine_goal_route, cached_graph_geometry
from lewm.fine_goal_route_development import fine_goal_route


def comparable(result):
    result = dict(result)
    if 'fine_goal_route' in result:
        result['fine_goal_route'] = {k:v for k,v in result['fine_goal_route'].items() if k != 'routing_s'}
    return result


def test_successful_route_is_unchanged_and_obstacle_changes_invalidate_reuse():
    route = dict(status='OBSERVED_FLOOR_ROUTE_TO_FRONTIER', nominal_radius_m=.45)
    floor = frozenset((x, 0) for x in range(41))
    for fine in (frozenset(), frozenset({(80, 0)}), frozenset()):
        snapshot = SimpleNamespace(floor=floor, fine_occupied=fine,
            occupied=frozenset((x//5,y//5) for x,y in fine))
        for position in (np.array([.1,.025]), np.array([.101,.025])):
            original = fine_goal_route(snapshot, position, np.array([1.7,.025]), route)
            cached = cached_fine_goal_route(snapshot, position, np.array([1.7,.025]), route)
            assert comparable(cached) == comparable(original)
            assert ('fine_goal_route' in cached) == (not fine)


def test_changed_segment_coordinates_are_not_rounded_to_a_cache_key():
    cells = frozenset({(0,0)})
    geometry = cached_graph_geometry(cells)
    first = geometry.minimum((.6,0.),(.7,0.))
    shifted = geometry.minimum((.60001,0.),(.7,0.))
    assert shifted != first
    assert geometry.minimum((.6,0.),(.7,0.)) == first
    assert geometry._minimum.cache_info().hits >= 1
