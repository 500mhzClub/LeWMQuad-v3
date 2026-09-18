"""A distant completed view does not retire an unreached frontier target."""
import numpy as np

from lewm.nearby_panorama_directed_view_development import NearbyPanoramaFrontierVisits
from lewm.observed_floor_waypoint_development import centre


class ArrivalConditionedFrontierVisits(NearbyPanoramaFrontierVisits):
    arrival_radius_m = .10

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.approach_after_view = set()

    def advance(self, snapshot, position, heading, goal, now_ns, route):
        previous_excluded = self.excluded.copy()
        previous_events = len(self.events)
        starting_view = self.visit is None
        standoff = self.standoff_m
        if (self.visit is None and route['status'] == 'OBSERVED_FLOOR_ROUTE_TO_FRONTIER'
                and tuple(route['route_cells'][-1]) in self.approach_after_view):
            # Reuse the existing observed route and original physical-arrival
            # trigger. All downstream command/obstacle checks remain active.
            self.standoff_m = None
        try:
            action, receipt = super().advance(snapshot, position, heading, goal, now_ns, route)
        finally:
            self.standoff_m = standoff
        if starting_view and action == 'view':
            # Nearby-panorama selection needs both viewing endpoints, including
            # panoramas started after arrival with the standoff trigger disabled.
            self.visit = self.visit | dict(view_start_map_xy_m=np.asarray(position).tolist())
            receipt = dict(self.visit)
        if len(self.events) > previous_events:
            event = self.events[-1]
            distance = float(np.linalg.norm(np.asarray(position)-event['target_xy_m']))
            deferred = distance > self.arrival_radius_m
            if deferred:
                viewed = {c for c in snapshot.floor
                    if np.linalg.norm(centre(c)-event['target_xy_m']) <= .10}
                self.approach_after_view.update(viewed)
                self.excluded = previous_excluded
            self.events[-1] = event | dict(
                frontier_exclusion_requires_observed_pose_arrival=True,
                completion_distance_to_target_m=distance,
                exclusion_deferred_until_arrival=deferred,
                deferred_excluded_cells=event['excluded_cells'] if deferred else 0,
                excluded_cells=0 if deferred else event['excluded_cells'],
                arrival_radius_m=self.arrival_radius_m)
        return action, receipt


class ArrivalConditionedFrontierMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.frontier_visits
        if old.events or old.excluded or old.visit is not None:
            raise ValueError('install frontier treatment before execution')
        self.frontier_visits = ArrivalConditionedFrontierVisits(
            standoff_m=old.standoff_m, panorama=old.panorama)
