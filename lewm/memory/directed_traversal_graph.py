"""Development routing over observed, directed, viable local traversals.

Place association and arrival verification belong to upstream components.
This graph cannot certify that their reports are correct. It never infers a
reverse edge or accepts visual similarity as successful physical traversal.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math


@dataclass(frozen=True)
class Traversal:
    event_id: str
    source: str
    target: str
    duration_s: float
    reached: bool
    viable_arrival: bool
    association_confirmed: bool

    def __post_init__(self):
        if not all(isinstance(value, str) and value for value in
                   (self.event_id, self.source, self.target)):
            raise ValueError('traversal and place identities must be nonempty strings')
        if self.source == self.target:
            raise ValueError('self transitions are not route edges')
        if (isinstance(self.duration_s, bool) or not isinstance(self.duration_s, (int, float))
                or not math.isfinite(self.duration_s) or self.duration_s <= 0):
            raise ValueError('measured traversal duration must be finite and positive')
        if not all(isinstance(value, bool) for value in
                   (self.reached, self.viable_arrival, self.association_confirmed)):
            raise ValueError('explicit boolean arrival and association evidence required')

    @property
    def usable(self):
        return self.reached and self.viable_arrival and self.association_confirmed


class DirectedTraversalGraph:
    """No oracle layout: nodes must first be observed and edges executed.

    Events are supplied in execution order. A failed/unconfirmed newest attempt
    disables its direction until a later successful, viable, confirmed traversal.
    This is a conservative routing policy, not an estimator of permanent
    infeasibility. Failed events remain in the history and never vanish from
    task accounting. Costs average successful observed durations; they are not
    calibrated future costs or safety guarantees.
    """

    def __init__(self):
        self.places = set()
        self._events = {}
        self._by_edge = {}

    def observe_place(self, place_id):
        if not isinstance(place_id, str) or not place_id:
            raise ValueError('observed place needs an explicit identity')
        self.places.add(place_id)

    def record(self, event: Traversal):
        if not isinstance(event, Traversal):
            raise ValueError('typed traversal evidence required')
        if event.source not in self.places or event.target not in self.places:
            raise ValueError('cannot add an edge between unobserved places')
        if event.event_id in self._events:
            if event != self._events[event.event_id]:
                raise ValueError('conflicting reuse of a traversal event identity')
            return False  # idempotent receipt; never count twice
        self._events[event.event_id] = event
        self._by_edge.setdefault((event.source, event.target), []).append(event)
        return True

    def edge_summary(self, source, target):
        rows = self._by_edge.get((source, target), [])
        successful = [row.duration_s for row in rows if row.usable]
        return {'attempts': len(rows), 'qualified_traversals': len(successful),
                'usable_now': bool(rows and rows[-1].usable),
                'mean_success_duration_s': sum(successful) / len(successful) if successful else None}

    def route(self, source, target, *, avoid_edges=()):
        """Minimum observed-duration directed route, or None if unsupported."""
        if source not in self.places or target not in self.places:
            raise ValueError('route endpoints must have been observed')
        if source == target:
            return [source]
        excluded = set(avoid_edges)
        outgoing = {}
        for (a, b) in self._by_edge:
            summary = self.edge_summary(a, b)
            if summary['usable_now'] and (a, b) not in excluded:
                outgoing.setdefault(a, []).append((b, summary['mean_success_duration_s']))
        queue, best = [(0.0, (source,))], {source: 0.0}
        while queue:
            cost, path = heapq.heappop(queue)
            node = path[-1]
            if cost > best[node]:
                continue
            if node == target:
                return list(path)
            for neighbor, duration in sorted(outgoing.get(node, [])):
                candidate = cost + duration
                if candidate < best.get(neighbor, float('inf')):
                    best[neighbor] = candidate
                    heapq.heappush(queue, (candidate, path + (neighbor,)))
        return None
