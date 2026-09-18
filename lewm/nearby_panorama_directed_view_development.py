"""Test directed frontier revisits near an actually completed panorama."""
import numpy as np

from lewm.frontier_visit_runtime_development import FrontierVisits


class NearbyPanoramaFrontierVisits(FrontierVisits):
    nearby_radius_m = .25

    def _nearby_panorama(self, position, route):
        if route['status'] != 'OBSERVED_FLOOR_ROUTE_TO_FRONTIER':
            return None
        target = np.asarray(route['target_map_xy_m'])
        for event in reversed(self.events):
            if (not event.get('panoramic_frontier_view')
                    or len(event.get('completed_view_stages', ())) != 9
                    or 'view_completion_map_xy_m' not in event):
                continue
            # Bound both the frontier target and the physical viewing location.
            distances = [np.linalg.norm(target-event['target_xy_m']),
                np.linalg.norm(position-np.asarray(event['view_start_map_xy_m'])),
                np.linalg.norm(position-np.asarray(event['view_completion_map_xy_m']))]
            if all(d <= self.nearby_radius_m for d in distances):
                return dict(completed_panorama_started_ns=event['started_ns'],
                    completed_panorama_completed_ns=event['completed_ns'],
                    nearby_radius_m=self.nearby_radius_m,
                    target_distance_m=float(distances[0]),
                    view_start_distance_m=float(distances[1]),
                    view_completion_distance_m=float(distances[2]),
                    fresh_directed_view_required=True,
                    full_footprint_visibility_certified=False)
        return None

    def advance(self, snapshot, position, heading, goal, now_ns, route):
        started = self.visit is None
        witness = self._nearby_panorama(position, route) if started and self.panorama else None
        panorama = self.panorama
        event_count = len(self.events)
        # A directed visit remains directed until its measured view completes.
        if self.visit is not None:
            self.panorama = bool(self.visit.get('panoramic_frontier_view', False))
        elif witness is not None:
            self.panorama = False
        try:
            action, receipt = super().advance(snapshot, position, heading, goal, now_ns, route)
        finally:
            self.panorama = panorama
        if started and action == 'view' and witness is not None:
            self.visit = self.visit | dict(nearby_completed_panorama=witness)
            receipt = dict(self.visit)
        if len(self.events) > event_count:
            self.events[-1] = self.events[-1] | dict(
                view_completion_map_xy_m=np.asarray(position).tolist())
        return action, receipt


class NearbyPanoramaRuntimeMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        previous = self.frontier_visits
        if previous.visit is not None or previous.events or previous.excluded:
            raise ValueError('install frontier visit strategy before execution')
        self.frontier_visits = NearbyPanoramaFrontierVisits(
            standoff_m=previous.standoff_m, panorama=previous.panorama)
