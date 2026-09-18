"""Reuse fine-grid search results only for identical observed graph inputs."""
from functools import lru_cache
import heapq
import time

import numpy as np
from scipy.spatial import cKDTree

from lewm.cached_fine_goal_route_development import cached_graph_geometry, CachedFineGoalRecoveryRuntime
from lewm.clearance_preferred_route_development import clearance_costs
from lewm.fine_goal_route_development import FineGoalRouteRuntime
from lewm.observed_floor_waypoint_development import centre,NEIGHBOURS
from lewm.vectorized_connector_routing_development import segment_cells


@lru_cache(maxsize=8)
def floor_index(floor):
    cells=tuple(sorted(floor))
    points=(np.asarray(cells,dtype=float).reshape(-1,2)+.5)*.05
    return cells,cKDTree(points)


def nearby_candidates(floor,position):
    cells,tree=floor_index(floor)
    # The conservative query only shortlists; original scalar distances and
    # tuple tie-breaking decide the exact 1.25-m membership and ordering.
    candidates=[]
    for index in tree.query_ball_point(position,1.25+1e-12):
        cell=cells[index];distance=float(np.linalg.norm(centre(cell)-position))
        if distance<=1.25:candidates.append((distance,cell))
    return [cell for _,cell in sorted(candidates)]


@lru_cache(maxsize=8)
def search_graph(floor,occupied,fine_occupied,radius,seed,target):
    geometry=cached_graph_geometry(fine_occupied);costs=clearance_costs(occupied)
    def heuristic(c):return abs(c[0]-target[0])+abs(c[1]-target[1])
    queue=[(heuristic(seed),0.,seed)];distance={seed:0.};parent={seed:None};edges={}
    while queue:
        _,cost,cell=heapq.heappop(queue)
        if cost!=distance[cell]:continue
        if cell==target:
            path=[];here=cell
            while here is not None:path.append(here);here=parent[here]
            return tuple(path[::-1]),cost,len(parent)
        for dx,dy in NEIGHBOURS:
            nxt=(cell[0]+dx,cell[1]+dy)
            if nxt not in floor:continue
            step=.5*(costs[cell[0]+100,cell[1]+100]+costs[nxt[0]+100,nxt[1]+100])
            candidate=cost+step
            if candidate>=distance.get(nxt,float('inf')):continue
            edge=tuple(sorted((cell,nxt)))
            if edge not in edges:
                d=geometry.minimum(centre(cell),centre(nxt))
                edges[edge]=d is None or d>radius+1e-12
            if not edges[edge]:continue
            distance[nxt]=candidate;parent[nxt]=cell
            heapq.heappush(queue,(candidate+heuristic(nxt),candidate,nxt))
    return None


def fine_goal_route(snapshot,position,goal,route):
    if route['status']=='OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL':return route
    target=tuple(map(int,np.floor(np.asarray(goal)/.05)))
    if target not in snapshot.floor:return route
    began=time.perf_counter();geometry=cached_graph_geometry(snapshot.fine_occupied)
    radius=route['nominal_radius_m']
    def clear(a,b):
        d=geometry.minimum(a,b)
        return d is None or d>radius+1e-12
    # Continuous endpoints are always checked at their exact current values.
    if not clear(position,position) or not clear(centre(target),goal):return route
    seed=next((c for c in nearby_candidates(snapshot.floor,position)
        if clear(position,centre(c))),None)
    if seed is None:return route
    found=search_graph(snapshot.floor,snapshot.occupied,snapshot.fine_occupied,radius,seed,target)
    if found is None:return route
    path,cost,reached=found;unknown=segment_cells(position,centre(seed))-snapshot.floor
    result={k:v for k,v in route.items() if k not in ('clearance_preferred_route',
        'reachable_floor_cells','frontier_cells','nominal_route_cells')}
    return result|dict(status='OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL',route_cells=[list(c) for c in path],
        entry_map_xy_m=centre(seed).tolist(),target_map_xy_m=centre(target).tolist(),
        unknown_connector_cells=[list(c) for c in sorted(unknown)],
        complete_route_floor_coverage=not unknown,initial_connector_requires_observation=bool(unknown),
        coarse_route_inflation_unchanged=False,
        fine_goal_route=dict(original_status=route['status'],nominal_radius_m=radius,
            observed_floor_only=True,continuous_edges_checked=True,
            exact_goal_connector_checked=True,action_clearance_checks_unchanged=True,
            weighted_path_cost=cost,reached_cells=reached,routing_s=time.perf_counter()-began))


class CachedFineConnectivityMixin:
    """Insert immediately before the predecessor's cached fine-goal layer."""
    def _routing_proposer(self,snapshot):
        original=super(FineGoalRouteRuntime,self)._routing_proposer(snapshot)
        def propose(floor,occupied,position,goal,**kwargs):
            route=original(floor,occupied,position,goal,**kwargs)
            return fine_goal_route(snapshot,position,goal,route)
        return propose
