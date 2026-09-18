"""Recover observed goal connectivity lost to coarse obstacle inflation."""
import heapq
import time
import numpy as np
from lewm.fine_stored_obstacle_routing_development import cached_clearance
from lewm.observed_floor_waypoint_development import centre,NEIGHBOURS
from lewm.vectorized_connector_routing_development import segment_cells
from lewm.clearance_preferred_route_development import clearance_costs
from lewm.terminal_position_priority_development import ProgressRejoiningTerminalRuntime


def fine_goal_route(snapshot,position,goal,route):
    if route['status']=='OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL':return route
    target=tuple(map(int,np.floor(np.asarray(goal)/.05)))
    if target not in snapshot.floor:return route
    began=time.perf_counter();geometry=cached_clearance(snapshot.fine_occupied)
    radius=route['nominal_radius_m']
    def clear(a,b):
        d=geometry.minimum(a,b)
        return d is None or d>radius+1e-12
    if not clear(position,position) or not clear(centre(target),goal):return route
    candidates=sorted((c for c in snapshot.floor if np.linalg.norm(centre(c)-position)<=1.25),
        key=lambda c:(float(np.linalg.norm(centre(c)-position)),c))
    seed=next((c for c in candidates if clear(position,centre(c))),None)
    if seed is None:return route
    costs=clearance_costs(snapshot.occupied)
    def heuristic(c):return abs(c[0]-target[0])+abs(c[1]-target[1])
    queue=[(heuristic(seed),0.,seed)];distance={seed:0.};parent={seed:None};edges={}
    while queue:
        _,cost,cell=heapq.heappop(queue)
        if cost!=distance[cell]:continue
        if cell==target:
            path=[];here=cell
            while here is not None:path.append(list(here));here=parent[here]
            path.reverse();unknown=segment_cells(position,centre(seed))-snapshot.floor
            result={k:v for k,v in route.items() if k not in ('clearance_preferred_route',
                'reachable_floor_cells','frontier_cells','nominal_route_cells')}
            return result|dict(status='OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL',route_cells=path,
                entry_map_xy_m=centre(seed).tolist(),target_map_xy_m=centre(target).tolist(),
                unknown_connector_cells=[list(c) for c in sorted(unknown)],
                complete_route_floor_coverage=not unknown,initial_connector_requires_observation=bool(unknown),
                coarse_route_inflation_unchanged=False,
                fine_goal_route=dict(original_status=route['status'],nominal_radius_m=radius,
                    observed_floor_only=True,continuous_edges_checked=True,
                    exact_goal_connector_checked=True,action_clearance_checks_unchanged=True,
                    weighted_path_cost=cost,reached_cells=len(parent),routing_s=time.perf_counter()-began))
        for dx,dy in NEIGHBOURS:
            nxt=(cell[0]+dx,cell[1]+dy)
            if nxt not in snapshot.floor:continue
            step=.5*(costs[cell[0]+100,cell[1]+100]+costs[nxt[0]+100,nxt[1]+100])
            candidate=cost+step
            if candidate>=distance.get(nxt,float('inf')):continue
            edge=tuple(sorted((cell,nxt)))
            if edge not in edges:edges[edge]=clear(centre(cell),centre(nxt))
            if not edges[edge]:continue
            distance[nxt]=candidate;parent[nxt]=cell
            heapq.heappush(queue,(candidate+heuristic(nxt),candidate,nxt))
    return route


class FineGoalRouteRuntime(ProgressRejoiningTerminalRuntime):
    def _routing_proposer(self,snapshot):
        original=super()._routing_proposer(snapshot)
        def propose(floor,occupied,position,goal,**kwargs):
            route=original(floor,occupied,position,goal,**kwargs)
            return fine_goal_route(snapshot,position,goal,route)
        return propose
