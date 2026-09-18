"""Floor-route proposals with explicit unknown start connectors, not motion grants."""
from collections import deque
import math
import numpy as np
from functools import lru_cache

CELL_M = .05
NEIGHBOURS = ((-1, 0), (0, -1), (0, 1), (1, 0))


def centre(cell):
    return (np.asarray(cell, float)+.5)*CELL_M


def segment_cells(start, end):
    """Closed supercover via segment/box intersection; include corner contacts."""
    a, b = np.asarray(start, float), np.asarray(end, float)
    if a.shape != (2,) or b.shape != (2,) or not np.isfinite([a, b]).all() or np.max(np.abs([a, b])) > 5.:
        raise ValueError('bounded finite map connector required')
    low = np.floor(np.minimum(a, b)/CELL_M).astype(int)-1
    high = np.floor(np.maximum(a, b)/CELL_M).astype(int)
    found = set()
    for x in range(low[0], high[0]+1):
        for y in range(low[1], high[1]+1):
            lo = np.array([x, y])*CELL_M; hi = lo+CELL_M
            entry, leave = 0., 1.
            for axis in range(2):
                delta = b[axis]-a[axis]
                if abs(delta) < 1e-15:
                    if not lo[axis]-1e-12 <= a[axis] <= hi[axis]+1e-12:
                        entry, leave = 1., 0.; break
                else:
                    u, v = sorted(((lo[axis]-a[axis])/delta, (hi[axis]-a[axis])/delta))
                    entry, leave = max(entry, u), min(leave, v)
            if entry <= leave+1e-12: found.add((x, y))
    return found


def inflated_cells(occupied, radius_m=.45):
    if not math.isfinite(radius_m) or not 0 <= radius_m <= 1.:
        raise ValueError('explicit bounded nominal footprint radius required')
    return set(_inflated_cells_cached(frozenset(occupied),radius_m))


@lru_cache(maxsize=2)
def _inflated_cells_cached(occupied,radius_m):
    if not occupied:return frozenset()
    n = math.ceil(radius_m/CELL_M)+1
    offsets = [(x, y) for x in range(-n, n+1) for y in range(-n, n+1)
        if np.hypot(max(abs(x)-1, 0), max(abs(y)-1, 0))*CELL_M <= radius_m+1e-12]
    points=np.asarray(tuple(occupied),dtype=int)
    lo=points.min(0)-n;hi=points.max(0)+n
    shape=hi-lo+1
    if np.prod(shape)>1_000_000:
        return frozenset((a+x,b+y) for a,b in occupied for x,y in offsets)
    mask=np.zeros(tuple(shape),bool);local=points-lo
    offsets=np.asarray(offsets,dtype=int)
    for start in range(0,len(local),512):
        indices=local[start:start+512,None,:]+offsets[None,:,:]
        mask[indices[:,:,0],indices[:,:,1]]=True
    return frozenset((int(x+lo[0]),int(y+lo[1])) for x,y in np.argwhere(mask))


def propose(floor, occupied, position, goal, *, radius_m=.45, connector_clear=None, excluded_frontiers=()):
    """Connect to measured floor, then route through its four-neighbour component.

    A connector may contain unknown cells, which are returned explicitly and
    prevent complete-route coverage. It may never cross a known inflated obstacle.
    No inferred connector changes the map. If no entry exists request another view.
    """
    floor, occupied = set(floor), set(occupied)
    p, g = np.asarray(position, float), np.asarray(goal, float)
    if (len(floor) > 40000 or len(occupied) > 40000 or p.shape != (2,) or g.shape != (2,)
            or not np.isfinite([p, g]).all() or np.max(np.abs([p, g])) > 4.9
            or any(len(c) != 2 or any(type(v) is not int or not -100 <= v < 100 for v in c) for c in floor | occupied)):
        raise ValueError('bounded registered map cells and finite mission points required')
    blocked = inflated_cells(occupied, radius_m)
    available = floor-blocked
    candidates = sorted((c for c in available if np.linalg.norm(centre(c)-p) <= 1.25),
        key=lambda c: (float(np.linalg.norm(centre(c)-p)), c))
    if connector_clear is not None and not connector_clear(p, p, occupied, radius_m):
        # Every connector includes its start disk; none can clear this collision.
        candidates = []
    seed = connector = None
    for c in candidates:
        cells = segment_cells(p, centre(c))
        clear = not cells & blocked if connector_clear is None else connector_clear(p, centre(c), occupied, radius_m)
        if clear:
            seed, connector = c, cells; break
    common = dict(nominal_radius_m=radius_m, observed_floor_cells=len(floor),
        goal_map_xy_m=g.tolist(),
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
    frontier = [c for c in parent if c not in excluded_frontiers
        and any((c[0]+dx, c[1]+dy) not in observed for dx, dy in NEIGHBOURS)]
    if goal_cell in parent:
        target = goal_cell; status = 'OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL'
    elif frontier:
        target = min(frontier, key=lambda c: (float(np.linalg.norm(centre(c)-g))+.1*CELL_M*distance[c], c))
        status = 'OBSERVED_FLOOR_ROUTE_TO_FRONTIER'
    else:
        return common | dict(status='OBSERVED_COMPONENT_HAS_NO_FRONTIER', route_cells=[],
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
