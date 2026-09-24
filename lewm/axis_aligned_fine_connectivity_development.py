"""Accelerate exact distances for axis-aligned observed-floor graph edges."""
from functools import lru_cache
import math

import numpy as np
from numba import njit

from lewm import cached_fine_connectivity_development as previous
from lewm.cached_fine_goal_route_development import cached_graph_geometry
from lewm.eligible_floor_registration_development import bind


@njit(cache=False)
def axis_minimum(a, b, low, high):
    # An axis-aligned segment is a degenerate rectangle. Its exact distance
    # to a closed obstacle square is the norm of the two interval gaps.
    xmin, xmax = min(a[0], b[0]), max(a[0], b[0])
    ymin, ymax = min(a[1], b[1]), max(a[1], b[1])
    best = math.inf
    for i in range(len(low)):
        dx = max(low[i, 0] - xmax, xmin - high[i, 0], 0.)
        dy = max(low[i, 1] - ymax, ymin - high[i, 1], 0.)
        squared = dx * dx + dy * dy
        if squared < best:
            best = squared
    return math.sqrt(best)


class AxisGraphClearance:
    def __init__(self, cells):
        self.original = cached_graph_geometry(cells)
        self.geometry = self.original.geometry
        self._minimum = lru_cache(maxsize=16384)(self._calculate)

    def _calculate(self, start, end):
        a, b = np.asarray(start, float), np.asarray(end, float)
        if (a.shape != (2,) or b.shape != (2,)
                or not np.isfinite([a, b]).all() or np.max(np.abs([a, b])) > 5.):
            raise ValueError('bounded finite segment required')
        if not len(self.geometry.cells):
            return None
        if a[0] != b[0] and a[1] != b[1]:
            return self.original.minimum(a, b)
        return float(axis_minimum(a, b, self.geometry.low, self.geometry.high))

    def minimum(self, start, end):
        return self._minimum(tuple(start), tuple(end))


@lru_cache(maxsize=2)
def axis_graph_geometry(cells):
    return AxisGraphClearance(cells)


# Reuse the exact search, costs, seed choice and tie-breaking. Only graph-edge
# distances use the accelerated calculation; continuous connectors retain the
# predecessor geometry and its original checks.
search_graph = lru_cache(maxsize=8)(bind(previous.search_graph.__wrapped__,
    cached_graph_geometry=axis_graph_geometry))
fine_goal_route = bind(previous.fine_goal_route, search_graph=search_graph)


def warmup():
    axis_minimum(np.zeros(2), np.zeros(2), np.zeros((1, 2)), np.ones((1, 2)))


class AxisFineConnectivityMixin:
    def _routing_proposer(self, snapshot):
        original = super(previous.FineGoalRouteRuntime, self)._routing_proposer(snapshot)
        def propose(floor, occupied, position, goal, **kwargs):
            route = original(floor, occupied, position, goal, **kwargs)
            return fine_goal_route(snapshot, position, goal, route)
        return propose
