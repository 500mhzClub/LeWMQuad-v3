"""Observed same-cell arrival bookkeeping; retired targets remain traversable."""
from copy import deepcopy
import hashlib
import json
import numpy as np

from lewm.observed_floor_waypoint_development import CELL_M, NEIGHBOURS
from lewm.reached_frontier_waypoint_development import propose


def local_classification(cell, floor, occupied):
    cells = [cell] + [(cell[0]+dx, cell[1]+dy) for dx, dy in NEIGHBOURS]
    return tuple((candidate in floor, candidate in occupied) for candidate in cells)


class ReachedFrontierState:
    def __init__(self):
        self.goal_identity = None
        self.retired = {}
        self.last_ns = None
        self.last_input_sha256 = None
        self.last_proposal = None
        self.last_receipt = None

    def waypoint(self, floor, occupied, position, target, original, *, mission_goal, now_ns):
        p, g, identity = (np.asarray(value, float) for value in (position, target, mission_goal))
        if (any(value.shape != (2,) or not np.isfinite(value).all() for value in (p, g, identity))
                or type(now_ns) is not int or now_ns < 1_500_000_000
                or (now_ns-1_500_000_000) % 100_000_000
                or (self.last_ns is not None and now_ns < self.last_ns)):
            raise ValueError('current finite observed mission points and monotonic observation clock required')
        floor, occupied = set(floor), set(occupied)
        identity = tuple(identity.tolist())
        binding = hashlib.sha256(json.dumps(dict(floor=sorted(floor), occupied=sorted(occupied),
            position=p.tolist(), target=g.tolist(), mission_goal=identity, original=original),
            sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if now_ns == self.last_ns:
            if binding != self.last_input_sha256:
                raise ValueError('same observation cannot change frontier inputs')
            return deepcopy(self.last_proposal)
        goal_changed = self.goal_identity is not None and self.goal_identity != identity
        if goal_changed:
            self.retired.clear()
        self.goal_identity = identity
        released = []
        for cell, signature in list(self.retired.items()):
            if local_classification(cell, floor, occupied) != signature:
                released.append(cell); del self.retired[cell]
        proposal = (propose(floor, occupied, p, g, retired_frontiers=self.retired)
            if self.retired else deepcopy(original))
        reached = None
        if proposal['status'] == 'OBSERVED_FLOOR_ROUTE_TO_FRONTIER':
            route = proposal['route_cells']
            if not route:
                raise ValueError('observed frontier proposal must contain its target cell')
            cell = tuple(route[-1])
            if tuple(map(int, np.floor(p/CELL_M))) == cell:
                if cell not in floor or cell in occupied:
                    raise ValueError('reached frontier must remain observed floor without an occupied cell')
                if len(self.retired) >= 40000:
                    raise ValueError('bounded frontier ledger exhausted')
                self.retired[cell] = local_classification(cell, floor, occupied)
                reached = cell
                proposal = propose(floor, occupied, p, g, retired_frontiers=self.retired)
        self.last_receipt = dict(frame=(now_ns-1_500_000_000)//100_000_000, measured_ns=now_ns,
            current_observed_cell=list(map(int, np.floor(p/CELL_M))),
            reached_frontier_cell=None if reached is None else list(reached),
            released_frontier_cells=[list(cell) for cell in sorted(released)],
            retired_frontier_cells=[list(cell) for cell in sorted(self.retired)],
            original_proposal_status=original['status'], selected_proposal_status=proposal['status'],
            original_route_cell_count=len(original['route_cells']), selected_route_cell_count=len(proposal['route_cells']),
            mission_goal_changed=goal_changed, revisit_requires_changed_local_classification=True,
            retirement_is_obstacle_evidence=False, retired_cells_remain_traversable=True,
            native_state_used=False, model_forecasts_changed_by_frontier_helper=False,
            navigation_qualified=False)
        self.last_ns = now_ns; self.last_input_sha256 = binding; self.last_proposal = deepcopy(proposal)
        return proposal
