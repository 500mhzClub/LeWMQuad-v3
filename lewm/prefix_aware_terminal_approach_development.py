"""Enter terminal pulse planning before the queued prefix consumes its radius."""
import numpy as np

from lewm.terminal_position_priority_development import TERMINAL_APPROACH_RADIUS_M
from lewm.visual_support_recovery_development import VisualSupportRuntime


def terminal_radius(prefix):
    commands = np.asarray(prefix, float)
    if commands.shape != (3, 3) or not np.isfinite(commands).all():
        raise ValueError('three finite committed command intervals required')
    # This is command travel used to enter a slower control mode, not a pose
    # estimate or a calibrated stopping-distance bound.
    return TERMINAL_APPROACH_RADIUS_M + .1*float(np.linalg.norm(commands[:, :2], axis=1).sum())


class PrefixAwareTerminalMixin:
    def _route(self, *args, **kwargs):
        self.exact_terminal_target = False
        self.terminal_target_distance = None
        return super()._route(*args, **kwargs)

    def _route_target(self, route, snapshot, position):
        target = super()._route_target(route, snapshot, position)
        self.exact_terminal_target = bool(
            route['status'] == 'OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL'
            and np.linalg.norm(target-np.asarray(route['goal_map_xy_m'])) < 1e-9)
        self.terminal_target_distance = float(np.linalg.norm(target-np.asarray(position)))
        return target

    def _select_action(self, packet, evidence, prefix, goal_body, scan_error, snapshot, q, Q):
        radius = terminal_radius(prefix)
        previous = self.terminal_position_approach
        if scan_error is None and self.exact_terminal_target:
            self.terminal_position_approach = self.terminal_target_distance <= radius
        selected, correction = super()._select_action(
            packet, evidence, prefix, goal_body, scan_error, snapshot, q, Q)
        selected['prefix_aware_terminal_approach'] = dict(
            base_radius_m=TERMINAL_APPROACH_RADIUS_M, expanded_radius_m=radius,
            exact_goal_target=self.exact_terminal_target,
            observed_target_distance_m=self.terminal_target_distance,
            prior_terminal_mode=previous, terminal_mode=self.terminal_position_approach,
            newly_enabled=bool(self.terminal_position_approach and not previous),
            queued_command_travel_is_pose_estimate=False,
            stopping_bound_calibrated=False, measured_arrival_rules_unchanged=True)
        return selected, correction


class PrefixAwareTerminalRuntime(PrefixAwareTerminalMixin, VisualSupportRuntime):
    pass
