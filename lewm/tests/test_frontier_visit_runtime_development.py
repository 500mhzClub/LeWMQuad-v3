from types import SimpleNamespace
import numpy as np
from lewm.frontier_visit_runtime_development import FrontierVisits
from lewm.observed_floor_waypoint_development import propose


def test_frontier_requires_measured_heading_and_subsequent_map():
    state=FrontierVisits()
    snap=SimpleNamespace(floor=frozenset({(0,0)}),occupied=frozenset(),measured_ns=100,frame=1)
    route=dict(status='OBSERVED_FLOOR_ROUTE_TO_FRONTIER',target_map_xy_m=[.025,.025],route_cells=[[0,0]])
    p=np.array([.025,.025]);goal=np.array([1.,0.])
    assert state.advance(snap,p,1.,goal,200,route)[0]=='view'
    assert state.advance(snap,p,0.,goal,300,route)[0]=='view'
    assert not state.excluded
    snap.measured_ns=300;snap.frame=3
    assert state.advance(snap,p,0.,goal,400,route)[0]=='replan'
    assert state.excluded=={(0,0)}
    assert len(state.events)==1


def test_excluding_frontier_does_not_block_floor_route_or_goal():
    floor={(0,0),(1,0),(2,0)}
    result=propose(floor,set(),[.025,.025],[.125,.025],radius_m=0.,excluded_frontiers=floor)
    assert result['status']=='OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL'
    assert result['route_cells']==[[0,0],[1,0],[2,0]]


def test_standoff_views_before_arrival_and_uses_remaining_route_not_radial_distance():
    floor=frozenset((x,0) for x in range(9))
    snap=SimpleNamespace(floor=floor,occupied=frozenset(),measured_ns=100,frame=1)
    route=dict(status='OBSERVED_FLOOR_ROUTE_TO_FRONTIER',target_map_xy_m=[.425,.025],
        route_cells=[[x,0]for x in range(9)])
    p=np.array([.025,.025]);goal=np.array([1.,0.])
    assert FrontierVisits().advance(snap,p,0.,goal,200,route)[0]=='route'
    action,visit=FrontierVisits(standoff_m=.50).advance(snap,p,0.,goal,200,route)
    assert action=='view' and not visit['physical_frontier_arrival_claimed']
    np.testing.assert_allclose(visit['remaining_route_at_view_start_m'],.4)
    assert visit['heading_rad']==0.
    # Same nearby target, but a route detouring around an intervening obstacle.
    route['route_cells']=[[0,0],[0,4],[8,4],[8,0]]
    assert FrontierVisits(standoff_m=.50).advance(snap,p,0.,goal,200,route)[0]=='route'


def test_panorama_does_not_exclude_frontier_after_only_one_view():
    state=FrontierVisits(standoff_m=.5,panorama=True)
    snap=SimpleNamespace(floor=frozenset({(0,0)}),occupied=frozenset(),measured_ns=100,frame=1)
    route=dict(status='OBSERVED_FLOOR_ROUTE_TO_FRONTIER',target_map_xy_m=[.025,.025],route_cells=[[0,0]])
    p=np.array([.025,.025]);goal=np.array([1.,0.])
    action,first_receipt=state.advance(snap,p,0.,goal,200,route)
    assert action=='view'
    for index in range(9):
        heading=state.visit['heading_rad'];now=300+index*200
        # Correct heading alone is insufficient: wait for a map measured there.
        assert state.advance(snap,p,heading,goal,now,route)[0]=='view'
        assert not state.excluded
        snap.measured_ns=now;snap.frame+=1
        action,_=state.advance(snap,p,heading,goal,now+100,route)
        assert action==('replan' if index==8 else 'view')
    assert state.excluded=={(0,0)}
    assert len(state.events[0]['completed_view_stages'])==9
    assert first_receipt['completed_view_stages']==[]
