"""Prefer room around the robot while preserving the observed route target."""
from functools import lru_cache
import heapq
import math
import time
import numpy as np
from scipy.ndimage import distance_transform_edt
from lewm.observed_floor_waypoint_development import CELL_M,NEIGHBOURS,inflated_cells

PREFERRED_CLEARANCE_M=.60
CLEARANCE_WEIGHT=2.


@lru_cache(maxsize=2)
def clearance_costs(occupied):
    mask=np.ones((200,200),dtype=bool)
    for x,y in occupied:
        if not -100<=x<100 or not -100<=y<100:raise ValueError('bounded observed obstacle cells required')
        mask[x+100,y+100]=False
    if not occupied:return np.ones_like(mask,dtype=float)
    # Centre-to-square lower bound, used only as a soft routing preference.
    # The original inflated graph and exact fine-cell action checks remain.
    distance=np.maximum(0.,distance_transform_edt(mask)*CELL_M-math.sqrt(2.)*CELL_M/2)
    deficit=np.maximum(0.,PREFERRED_CLEARANCE_M-distance)/.15
    return 1.+CLEARANCE_WEIGHT*deficit**2


def preferred_path(floor,occupied,start,target,*,radius_m=.45):
    start,target=tuple(start),tuple(target)
    available=set(floor)-inflated_cells(occupied,radius_m)
    if start not in available or target not in available:
        raise ValueError('same observed traversable entry and target required')
    cost=clearance_costs(frozenset(occupied))
    def heuristic(c):return abs(c[0]-target[0])+abs(c[1]-target[1])
    queue=[(heuristic(start),0.,start)];distance={start:0.};parent={start:None};expanded=0
    while queue:
        _,value,cell=heapq.heappop(queue)
        if value!=distance[cell]:continue
        expanded+=1
        if cell==target:
            path=[];here=target
            while here is not None:path.append(list(here));here=parent[here]
            return path[::-1],dict(weighted_path_cost=value,expanded_cells=expanded,
                preferred_obstacle_clearance_m=PREFERRED_CLEARANCE_M,
                clearance_cost_weight=CLEARANCE_WEIGHT,route_target_unchanged=True,
                traversable_graph_unchanged=True,clearance_is_soft_preference=True)
        for dx,dy in NEIGHBOURS:
            nxt=(cell[0]+dx,cell[1]+dy)
            if nxt not in available:continue
            step=.5*(cost[cell[0]+100,cell[1]+100]+cost[nxt[0]+100,nxt[1]+100])
            candidate=value+step
            if candidate<distance.get(nxt,float('inf')):
                distance[nxt]=candidate;parent[nxt]=cell
                heapq.heappush(queue,(candidate+heuristic(nxt),candidate,nxt))
    raise ValueError('original reachable target became unreachable')


def refine_proposal(route,floor,occupied):
    original=route['route_cells']
    if not original:return route
    began=time.perf_counter()
    path,receipt=preferred_path(floor,occupied,original[0],original[-1],
        radius_m=route['nominal_radius_m'])
    return route|dict(route_cells=path,clearance_preferred_route=receipt|dict(
        added_routing_s=time.perf_counter()-began,
        original_route_steps=len(original)-1,preferred_route_steps=len(path)-1))


class ClearancePreferredRouteMixin:
    def _routing_proposer(self,snapshot):
        original=super()._routing_proposer(snapshot)
        def propose(floor,occupied,position,goal,**kwargs):
            return refine_proposal(original(floor,occupied,position,goal,**kwargs),floor,occupied)
        return propose


from lewm.clearance_turn_recovery_development import StepwiseClearanceTurnRecoveryRuntime


class ClearancePreferredTurnRecoveryRuntime(ClearancePreferredRouteMixin,StepwiseClearanceTurnRecoveryRuntime):
    pass
