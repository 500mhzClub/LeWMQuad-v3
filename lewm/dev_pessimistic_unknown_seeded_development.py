"""Pessimistic unknown cells with the start disc seeded at the operating precondition (development, 2 October 2026).

Andrew (2 October). The revised rule (never-observed cells block a move within body reach
0.425 m + the controller's calibrated e_f bound of the forecast centre path) deadlocked the
initial look-around when only the robot's 0.425-m reach disc was seeded: the depth cameras
cannot see beside or behind the start pose, and the blocking radius exceeds that seed (live
smoke test, C1 maze 30: no clear move in 1197 of 1197 decisions).

The start disc is now seeded at the operating precondition: the robot is placed in a cleared
0.5-m area, the same for every controller (it covers every blocking radius, 0.45-0.50 m). It is
seeded at 1-cm resolution: a fine cell counts as known free only when its whole square lies
within 0.5 m of the first planning position, so nothing beyond the precondition is assumed. The
traversed track (cells within 0.20 m of past planning positions) is seeded as before. The rest
is the revised rule of lewm/dev_harness_fixes_development.py (pessimistic_unknown_mixin), reused
through its context variable; that module is not modified. On the physical Go2 the wide-view
lidar would observe the start surroundings and make this precondition unnecessary.
"""
import math

import numpy as np

from lewm import dev_harness_fixes_development as fixes
from lewm.fine_stored_obstacle_routing_development import FineCellClearance
from lewm.navigation_capability_map_domain_development import COARSE_CELL_M, COARSE_HALF_CELLS, FINE_CELL_M

PRECONDITION_CLEAR_M = .50
_OFFSETS = np.stack(np.meshgrid(np.arange(5), np.arange(5), indexing='ij'), axis=-1).reshape(-1, 2)


def unknown_fine_cells(snapshot, position, start, track, start_clear_m=PRECONDITION_CLEAR_M, scope_m=fixes.UNKNOWN_SCOPE_M):
    """Fine cells of never-observed coarse cells within scope, minus the track and the precondition disc."""
    p = np.asarray(position, float)[:2]
    lo, hi = np.floor((p-scope_m)/COARSE_CELL_M).astype(int), np.floor((p+scope_m)/COARSE_CELL_M).astype(int)
    ii, jj = np.meshgrid(np.arange(lo[0], hi[0]+1), np.arange(lo[1], hi[1]+1), indexing='ij')
    keys = np.stack((ii.ravel(), jj.ravel()), axis=1)
    keys = keys[np.all((keys >= -COARSE_HALF_CELLS) & (keys < COARSE_HALF_CELLS), axis=1)]
    centres = (keys+.5)*COARSE_CELL_M
    candidate = np.linalg.norm(centres-p, axis=1) <= scope_m
    track = np.asarray(track, float).reshape(-1, 2)
    if len(track):
        candidate &= np.min(np.linalg.norm(centres[:, None]-track[None], axis=2), axis=1) > fixes.TRACK_RADIUS_M
    observed = snapshot.floor | snapshot.occupied
    unknown = np.array([k for k in keys[candidate] if (int(k[0]), int(k[1])) not in observed], int).reshape(-1, 2)
    if not len(unknown):
        return frozenset(), 0
    fine = (unknown[:, None, :]*5+_OFFSETS[None]).reshape(-1, 2)
    owner = np.repeat(np.arange(len(unknown)), len(_OFFSETS))
    low = fine*FINE_CELL_M-np.asarray(start, float)[:2]
    far = np.hypot(np.maximum(np.abs(low[:, 0]), np.abs(low[:, 0]+FINE_CELL_M)), np.maximum(np.abs(low[:, 1]), np.abs(low[:, 1]+FINE_CELL_M)))
    keep = far > start_clear_m
    return frozenset(map(tuple, fine[keep].tolist())), int(len(np.unique(owner[keep])))


def seeded_pessimistic_unknown_mixin(bound_m, level, start_clear_m=PRECONDITION_CLEAR_M):
    """The revised pessimistic-unknown rule with the start disc seeded at the operating precondition."""
    if not np.isfinite(bound_m) or not 0. < bound_m <= .10:
        raise ValueError('calibrated e_f bound must be in (0, 0.10] m')
    radius = fixes.BODY_REACH_M+bound_m

    class SeededPessimisticUnknownMixin:
        pessimistic_unknown = dict(bound_m=bound_m, level=level, blocking_radius_m=radius, start_clear_m=start_clear_m,
                                   start_seed='operating precondition: robot placed in a cleared area of this radius (all controllers)',
                                   track_radius_m=fixes.TRACK_RADIUS_M, scope_m=fixes.UNKNOWN_SCOPE_M)

        def _unknown_context(self, snapshot, position):
            p = tuple(float(v) for v in np.asarray(position, float)[:2])
            state = self.__dict__.setdefault('_dev_unknown_state', dict(start=None, track=[]))
            if state['start'] is None:
                state['start'] = p
            if not state['track'] or math.dist(p, state['track'][-1]) > .005:
                state['track'].append(p)
            fine, count = unknown_fine_cells(snapshot, p, state['start'], state['track'], start_clear_m)
            context = (FineCellClearance(fine), fixes.CHECK_REQUIRED_M-radius) if fine else None
            return context, dict(unknown_cells_in_scope=count, blocking_radius_m=radius, bound_m=bound_m, level=level,
                                 start_clear_m=start_clear_m, track_radius_m=fixes.TRACK_RADIUS_M, track_points=len(state['track']))

        def _select_clear_prediction(self, selected, prediction, snapshot, position, rotation):
            context, record = self._unknown_context(snapshot, position)
            token = fixes._UNKNOWN_CELLS.set(context)
            try:
                result = super()._select_clear_prediction(selected, prediction, snapshot, position, rotation)
            finally:
                fixes._UNKNOWN_CELLS.reset(token)
            selection = result[0] if isinstance(result, tuple) else result
            selection['dev_pessimistic_unknown'] = record
            return result

        def _route_target(self, route, snapshot, position):
            context, _ = self._unknown_context(snapshot, position)
            token = fixes._UNKNOWN_CELLS.set(context)
            try:
                return super()._route_target(route, snapshot, position)
            finally:
                fixes._UNKNOWN_CELLS.reset(token)

    SeededPessimisticUnknownMixin.__name__ = f'SeededPessimisticUnknown_{level}'
    return SeededPessimisticUnknownMixin
