"""Original conservative routing with an explicit visited-frontier exclusion."""
from collections import deque
import numpy as np
from lewm.observed_floor_waypoint_development import CELL_M,NEIGHBOURS,centre,segment_cells,inflated_cells
from lewm.observed_geometry_refinement_development import nominal_connector


def propose(floor, occupied, position, goal, *, retired_frontiers=(), radius_m=.45):
    """Connect to measured floor, then route through its four-neighbour component.

    A connector may contain unknown cells, which are returned explicitly and
    prevent complete-route coverage. Its nominal disk may not touch an occupied square.
    No inferred connector changes the map. If no entry exists request another view.
    """
    floor, occupied = set(floor), set(occupied)
    retired = set(retired_frontiers)
    if (len(retired) > 40000 or any(len(c) != 2 or
            any(type(v) is not int or not -100 <= v < 100 for v in c) for c in retired)):
        raise ValueError('bounded explicit retired frontier cells required')
    p, g = np.asarray(position, float), np.asarray(goal, float)
    if (len(floor) > 40000 or len(occupied) > 40000 or p.shape != (2,) or g.shape != (2,)
            or not np.isfinite([p, g]).all() or np.max(np.abs([p, g])) > 4.9
            or any(len(c) != 2 or any(type(v) is not int or not -100 <= v < 100 for v in c) for c in floor | occupied)):
        raise ValueError('bounded registered map cells and finite mission points required')
    blocked = inflated_cells(occupied, radius_m)
    available = floor-blocked
    candidates = sorted((c for c in available if np.linalg.norm(centre(c)-p) <= 1.25),
        key=lambda c: (float(np.linalg.norm(centre(c)-p)), c))
    seed = connector = None
    occupied_ordered = sorted(occupied)
    start_check = nominal_connector(p, p, occupied_ordered, radius_m=radius_m)
    connector_check = None
    if start_check['nominal_disk_connector_clear']:
        for c in candidates:
            check = nominal_connector(p, centre(c), occupied_ordered, radius_m=radius_m)
            if check['nominal_disk_connector_clear']:
                seed, connector = c, segment_cells(p, centre(c))
                connector_check = check
                break
    common = dict(connector_geometry='continuous_segment_to_closed_observed_squares',
        start_clearance=start_check, connector_clearance=connector_check,
        nominal_radius_m=radius_m, observed_floor_cells=len(floor),
        occupied_cells=len(occupied), nominal_route_cells=len(available),
        footprint_coverage_established=False, ground_support_approved=False,
        navigation_qualified=False, motion_permitted=False)
    if seed is None:
        return common | dict(status='ADDITIONAL_VIEW_REQUIRED', route_cells=[],
            unknown_connector_cells=[], complete_route_floor_coverage=False)
    parent = {seed: None}; queue = deque([seed]); distance = {seed: 0}
    while queue:
        cell = queue.popleft()
        for dx, dy in NEIGHBOURS:
            nxt = (cell[0]+dx, cell[1]+dy)
            if nxt in available and nxt not in parent:
                parent[nxt] = cell; distance[nxt] = distance[cell]+1; queue.append(nxt)
    goal_cell = tuple(map(int, np.floor(g/CELL_M)))
    observed = floor | occupied
    frontier = [c for c in parent if any((c[0]+dx, c[1]+dy) not in observed for dx, dy in NEIGHBOURS)]
    eligible_frontier = [c for c in frontier if c not in retired]
    if goal_cell in parent:
        target = goal_cell; status = 'OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL'
    elif eligible_frontier:
        target = min(eligible_frontier, key=lambda c: (float(np.linalg.norm(centre(c)-g))+.1*CELL_M*distance[c], c))
        status = 'OBSERVED_FLOOR_ROUTE_TO_FRONTIER'
    else:
        return common | dict(status=('OBSERVED_COMPONENT_HAS_NO_UNVISITED_FRONTIER' if frontier
            else 'OBSERVED_COMPONENT_HAS_NO_FRONTIER'), route_cells=[],
            unknown_connector_cells=sorted(connector-floor), complete_route_floor_coverage=False)
    path = []; here = target
    while here is not None:
        path.append(here); here = parent[here]
    path.reverse()
    unknown = connector-floor
    return common | dict(status=status, route_cells=[list(c) for c in path],
        entry_map_xy_m=centre(seed).tolist(), target_map_xy_m=centre(target).tolist(),
        unknown_connector_cells=[list(c) for c in sorted(unknown)],
        complete_route_floor_coverage=not unknown, reachable_floor_cells=len(parent),
        frontier_cells=len(frontier), initial_connector_requires_observation=bool(unknown))
