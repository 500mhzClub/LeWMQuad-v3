"""Opt-in terminal scoring in the mission's measured initial-frame XY metric."""
import numpy as np
from lewm.mission_coordinate_metric_development import planar_body_to_initial_xy


class MissionCoordinateMixin:
    def __init__(self, *args, coordinate_mode, **kwargs):
        if coordinate_mode not in ('original', 'consistent'):
            raise ValueError('explicit original or consistent coordinate mode required')
        self.coordinate_mode = coordinate_mode
        self._terminal_position_metric = None
        self._coordinate_planning_goal = None
        super().__init__(*args, **kwargs)

    def _route(self, snapshot, evidence, goal_initial_xy, **kwargs):
        # Capture the same goal snapshot that _plan uses, not a later mission
        # state that could change when an observation confirms an arrival.
        self._coordinate_planning_goal = np.asarray(goal_initial_xy, float).copy()
        return super()._route(snapshot, evidence, goal_initial_xy, **kwargs)

    def _score(self, prediction, goal_body, **kwargs):
        if self._terminal_position_metric is not None:
            kwargs['position_metric_matrix'] = self._terminal_position_metric
        return super()._score(prediction, goal_body, **kwargs)

    def _select_action(self, packet, evidence, prefix, goal_body, scan_error, snapshot, q, Q):
        active = bool(self.coordinate_mode == 'consistent'
                      and self.terminal_position_approach and scan_error is None)
        original_target = np.asarray(goal_body).copy()
        position = None
        if active:
            if self._coordinate_planning_goal is None:
                raise ValueError('observation-specific planning goal required')
            B = np.asarray(snapshot.map_from_initial)
            position = (B.T @ q)[:2]
            self._terminal_position_metric = planar_body_to_initial_xy(B, Q)
            goal_body = np.linalg.solve(self._terminal_position_metric,
                                       self._coordinate_planning_goal-position)
        try:
            selected, correction = super()._select_action(
                packet, evidence, prefix, goal_body, scan_error, snapshot, q, Q)
        finally:
            self._terminal_position_metric = None
        selected['mission_coordinate_metric'] = dict(mode=self.coordinate_mode,
            applied_to_terminal_position=active,
            original_waypoint_body_xy_m=original_target.tolist(),
            planning_goal_initial_xy_m=None if not active else self._coordinate_planning_goal.tolist(),
            current_initial_xy_m=None if position is None else position.tolist(),
            constant_gravity_height_assumed=active,
            forecast_values_and_clearance_coordinates_unchanged=True,
            observed_arrival_radius_unchanged=True)
        return selected, correction
