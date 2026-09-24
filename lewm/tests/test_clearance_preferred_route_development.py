import numpy as np
from lewm.clearance_preferred_route_development import preferred_path,clearance_costs
from lewm.observed_floor_waypoint_development import inflated_cells


def test_corridor_route_moves_away_from_wall_without_changing_target_or_graph():
    occupied={(x,y) for x in range(0,61) for y in (0,26)}
    floor={(x,y) for x in range(0,61) for y in range(1,26)}
    path,receipt=preferred_path(floor,occupied,(5,11),(55,11))
    assert path[0]==[5,11] and path[-1]==[55,11]
    assert max(y for _,y in path)>11
    assert all(tuple(c) in floor-inflated_cells(occupied) for c in path)
    assert all(abs(a[0]-b[0])+abs(a[1]-b[1])==1 for a,b in zip(path,path[1:]))
    assert receipt['route_target_unchanged'] and receipt['traversable_graph_unchanged']


def test_unobserved_floor_is_not_added_to_reach_a_cheaper_route():
    floor={(x,11) for x in range(5,56)}
    occupied={(x,0) for x in range(61)}
    path,_=preferred_path(floor,occupied,(5,11),(55,11))
    assert path==[[x,11] for x in range(5,56)]
    np.testing.assert_array_equal(clearance_costs(frozenset()),np.ones((200,200)))


def test_runtime_preserves_frontier_exclusion_and_connector_receipt():
    from types import SimpleNamespace
    from lewm.clearance_preferred_route_development import ClearancePreferredTurnRecoveryRuntime
    from lewm.fine_stored_obstacle_routing_development import proposer
    floor=frozenset((x,y) for x in range(20) for y in range(20))
    snapshot=SimpleNamespace(floor=floor,occupied=frozenset(),fine_occupied=frozenset())
    excluded={(19,y) for y in range(20)}
    runtime=object.__new__(ClearancePreferredTurnRecoveryRuntime)
    runtime.frontier_visits=SimpleNamespace(excluded=excluded);runtime.mission_latest=None
    original=proposer(snapshot)(floor,frozenset(),[.5,.5],[2.,.5],excluded_frontiers=excluded)
    actual=runtime._routing_proposer(snapshot)(floor,frozenset(),[.5,.5],[2.,.5])
    assert tuple(actual['route_cells'][-1]) not in excluded
    assert actual['route_cells'][0]==original['route_cells'][0]
    assert actual['route_cells'][-1]==original['route_cells'][-1]
    assert {k:v for k,v in actual.items() if k not in ('route_cells','clearance_preferred_route')}=={
        k:v for k,v in original.items() if k!='route_cells'}
