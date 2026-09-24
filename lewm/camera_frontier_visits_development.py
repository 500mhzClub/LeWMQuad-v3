"""Move to observable frontier viewpoints, then require fresh mapped evidence."""
import math
import numpy as np

from lewm.camera_frontier_viewpoint_development import (
    choose_route_viewpoint, directed_rotation, floor_cell_projection)
from lewm.fine_stored_obstacle_routing_development import proposer
from lewm.frontier_visit_runtime_development import wrap
from lewm.observed_floor_waypoint_development import centre, inflated_cells, NEIGHBOURS


class CameraFrontierVisits:
    def __init__(self):
        self.excluded = set(); self.visit = None; self.events = []
        self.standoff_m = None; self.panorama = False
        self.attempted = {}; self.floor_count = None; self.context = None

    def _finish(self, snapshot, now, reason, observed):
        v = self.visit
        self.events.append(v | dict(completed_ns=now, map_frame=snapshot.frame,
            completion_reason=reason, unknown_cell_observed=observed, excluded_cells=0))
        self.visit = None

    def _choose(self, snapshot, route, p, R, unknown):
        tried = self.attempted.get(unknown, set())
        view = choose_route_viewpoint(snapshot, route, p, R, unknown,
            excluded_viewpoint_cells=tried)
        if view is not None: return view
        # The robot may already be inside the camera's near-body blind region.
        # Search nearby known floor; every candidate still needs an observed route.
        available = snapshot.floor - inflated_cells(snapshot.occupied)
        candidates = sorted((c for c in available if c not in tried
            and np.linalg.norm(centre(c)-centre(unknown)) <= 1.25),
            key=lambda c:(float(np.linalg.norm(centre(c)-p[:2])), c))[:128]
        propose = proposer(snapshot)
        for cell in candidates:
            query = route | dict(route_cells=[list(cell)])
            view = choose_route_viewpoint(snapshot, query, p, R, unknown)
            if view is None: continue
            path = propose(snapshot.floor, snapshot.occupied, p[:2], centre(cell))
            if path['status'] != 'OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL': continue
            return view | dict(route_cells=path['route_cells'],
                known_floor_backtrack_search=True)
        return None

    def advance(self, snapshot, position, heading, goal, now_ns, route):
        p, R = self.context
        if self.floor_count != (len(snapshot.floor), len(snapshot.occupied)):
            self.excluded.clear()
            self.floor_count = (len(snapshot.floor), len(snapshot.occupied))
        observed = snapshot.floor | snapshot.occupied
        if self.visit is not None:
            unknown = tuple(self.visit['unknown_neighbour'])
            if unknown in observed:
                self._finish(snapshot, now_ns, 'REQUESTED_PATCH_OBSERVED', True)
                return 'replan', None
            if route['status'] == 'OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL':
                self._finish(snapshot, now_ns, 'MISSION_GOAL_ROUTE_AVAILABLE', False)
                return 'route', None
        if self.visit is None:
            if route['status'] != 'OBSERVED_FLOOR_ROUTE_TO_FRONTIER': return 'route', None
            target = tuple(route['route_cells'][-1]); blocked = inflated_cells(snapshot.occupied)
            unknowns = [(target[0]+dx, target[1]+dy) for dx, dy in NEIGHBOURS
                if (target[0]+dx, target[1]+dy) not in observed]
            unknowns.sort(key=lambda c:(c in blocked, float(np.linalg.norm(centre(c)-goal)), c))
            view = None
            for unknown in unknowns:
                view = self._choose(snapshot, route, p, R, unknown)
                if view is not None: break
            if view is None:
                # A temporary search exclusion is cleared by new mapped evidence.
                # No floor cell is removed, filled or declared observed.
                self.excluded.add(target)
                self.events.append(dict(target_cell=list(target), started_ns=now_ns,
                    completed_ns=now_ns, map_frame=snapshot.frame,
                    completion_reason='NO_REACHABLE_UNTRIED_CAMERA_VIEWPOINT',
                    unknown_cell_observed=False, excluded_cells=0,
                    temporary_search_exclusion=True))
                return 'replan', None
            self.visit = dict(target_cell=list(target), target_xy_m=route['target_map_xy_m'],
                unknown_neighbour=list(unknown), started_ns=now_ns, aligned_ns=None,
                heading_rad=view['view_heading_rad'], camera_viewpoint=view,
                view_start_map_xy_m=np.asarray(position).tolist(),
                camera_visibility_is_hypothesis=True, physical_frontier_arrival_claimed=False)
        v = self.visit; unknown = tuple(v['unknown_neighbour'])
        waypoint = np.asarray(v['camera_viewpoint']['viewpoint_map_xy_m'])
        planned_R, desired = directed_rotation(R, p[:2], unknown)
        projected = floor_cell_projection(unknown, p, planned_R, snapshot.floor_height)
        close = np.linalg.norm(position-waypoint) <= .10
        if close and any(row['fully_projected'] for row in projected):
            v['heading_rad'] = desired
            if abs(wrap(desired-heading)) <= .10:
                actual = floor_cell_projection(unknown, p, R, snapshot.floor_height)
                if any(row['fully_projected'] for row in actual):
                    if v['aligned_ns'] is None: v['aligned_ns'] = now_ns
                    if snapshot.measured_ns >= v['aligned_ns']:
                        # The patch remained unknown despite a fresh mapped view.
                        # Try another position, preserving its unknown status.
                        tried = self.attempted.setdefault(unknown, set())
                        tried.update(c for c in snapshot.floor
                            if np.linalg.norm(centre(c)-position) <= .10)
                        self._finish(snapshot, now_ns, 'FRESH_VIEW_PATCH_STILL_UNKNOWN', False)
                        return 'replan', None
                else:
                    v['aligned_ns'] = None
            else:
                v['aligned_ns'] = None
            v['current_directed_projection'] = projected
            return 'view', dict(v)
        v['aligned_ns'] = None
        path = proposer(snapshot)(snapshot.floor, snapshot.occupied, p[:2], waypoint)
        if path['status'] != 'OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL':
            self._finish(snapshot, now_ns, 'VIEWPOINT_ROUTE_NO_LONGER_AVAILABLE', False)
            return 'replan', None
        route.update(route_cells=path['route_cells'], target_map_xy_m=waypoint.tolist(),
            status='OBSERVED_FLOOR_ROUTE_TO_FRONTIER', camera_viewpoint=dict(v),
            entry_map_xy_m=path['entry_map_xy_m'],
            unknown_connector_cells=path['unknown_connector_cells'],
            complete_route_floor_coverage=path['complete_route_floor_coverage'],
            initial_connector_requires_observation=path['initial_connector_requires_observation'])
        return 'route', None


class CameraFrontierRuntimeMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.frontier_visits
        if old.visit is not None or old.events or old.excluded:
            raise ValueError('install camera viewing policy before execution')
        self.frontier_visits = CameraFrontierVisits()

    def _route(self, snapshot, evidence, goal, *, measured_ns):
        p, R, _ = self._pose(evidence, identity=(0, 0, 0), now_ns=measured_ns)
        B = np.asarray(snapshot.map_from_initial)
        self.frontier_visits.context = (B@p, B@R)
        return super()._route(snapshot, evidence, goal, measured_ns=measured_ns)
