from types import SimpleNamespace
from copy import deepcopy
import numpy as np
from lewm.fine_goal_route_development import fine_goal_route
from lewm.fine_stored_obstacle_routing_development import proposer,cached_clearance
from lewm.observed_floor_waypoint_development import centre


def corridor():
    fine=frozenset((x,y) for x in range(-100,200) for y in (-51,49))
    return SimpleNamespace(floor=frozenset((x,0) for x in range(20)),
        fine_occupied=fine,occupied=frozenset((x//5,y//5) for x,y in fine))


def test_exact_route_recovers_coarse_false_closure_without_radius_change():
    s=corridor();p=np.array([.025,.025]);goal=np.array([.975,.025])
    coarse=proposer(s)(s.floor,s.occupied,p,goal);before=deepcopy(coarse)
    assert coarse['status']!='OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL'
    result=fine_goal_route(s,p,goal,coarse)
    assert result['status']=='OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL'
    assert result['nominal_radius_m']==.45 and coarse==before
    points=[p]+[centre(c) for c in result['route_cells']]+[goal]
    geometry=cached_clearance(s.fine_occupied)
    assert min(geometry.minimum(a,b) for a,b in zip(points,points[1:]))>.45
    assert all(tuple(c) in s.floor for c in result['route_cells'])


def test_fallback_does_not_invent_floor_or_cross_observed_obstacles():
    p=np.array([.025,.025]);goal=np.array([.975,.025]);s=corridor()
    coarse=proposer(s)(s.floor,s.occupied,p,goal)
    s.floor=s.floor-{(10,0)}
    assert fine_goal_route(s,p,goal,coarse)==coarse
    s=corridor();s.fine_occupied=s.fine_occupied|frozenset((50,y) for y in range(-51,50))
    s.occupied=frozenset((x//5,y//5) for x,y in s.fine_occupied)
    coarse=proposer(s)(s.floor,s.occupied,p,goal)
    assert fine_goal_route(s,p,goal,coarse)==coarse
