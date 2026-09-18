"""Non-predictive nominal-route rule with explicit unknown start connectors.

Geometry gates concern current measured pose and a measured route connector.
No candidate future pose, command integration or learned forecast is evaluated.
"""
import math
import numpy as np
from lewm.observed_round_trip_mission_development import point
from lewm.observed_floor_waypoint_development import centre, segment_cells
from lewm.exact_mission_target_development import terminal_target
from lewm.observed_geometry_refinement_development import nominal_connector
from lewm.observation_horizon_waypoint_selection_development import wrap
from lewm.joint_rgbd_rigid_pose_development import proper
from lewm.geometry_progress_pilot_development import candidate_commands

HEADING_TOLERANCE_RAD = .1
LOCAL_TARGET_RADIUS_M = .04


class ReactiveNominalRouteSelector:
    def __init__(self, *, goal_initial_body_xy_m):
        self.goal = point(goal_initial_body_xy_m)
        self.scan_sign = None; self.scan_index = 0; self.scan_target = None; self.mode = 'NEW'

    def set_goal(self, goal_initial_body_xy_m):
        goal = point(goal_initial_body_xy_m)
        if not np.array_equal(goal, self.goal):
            self.goal = goal; self.scan_sign = None; self.scan_index = 0; self.scan_target = None; self.mode = 'NEW'

    def scan(self, mapper, p, R, heading):
        if self.scan_sign is None:
            points = np.array([centre(c) for c in sorted(mapper.floor)])
            local = (np.column_stack((points, np.full(len(points), p[2])))-p)@R if len(points) else np.empty((0, 3))
            local = local[np.linalg.norm(local[:, :2], axis=1) <= 2.]
            self.scan_sign = 1 if int((local[:, 1] > 0).sum()) >= int((local[:, 1] < 0).sum()) else -1
        offsets = [self.scan_sign*math.pi/4, self.scan_sign*math.pi/2,
            -self.scan_sign*math.pi/4, -self.scan_sign*math.pi/2, math.pi]
        if self.scan_index >= len(offsets): return None, True, None
        if self.scan_target is None: self.scan_target = offsets[self.scan_index]
        error = wrap(self.scan_target-heading)
        if abs(error) <= HEADING_TOLERANCE_RAD:
            self.scan_index += 1
            if self.scan_index == len(offsets): return None, True, None
            self.scan_target = offsets[self.scan_index]; error = wrap(self.scan_target-heading)
        return ('left_turn' if error > 0 else 'right_turn'), False, error

    def choose(self, mapper, geometry, *, now_ns):
        proposal = mapper.waypoint(self.goal.copy(), now_ns=now_ns)
        B = mapper.map_from_initial; R = proper(B@mapper.surface.rotation)
        p = np.asarray(B@mapper.surface.position, float)
        if p.shape != (3,) or not np.isfinite(p).all(): raise ValueError('current observed map pose required')
        heading = math.atan2(R[1, 0], R[0, 0])
        occupied = sorted(mapper.occupied)
        current = nominal_connector(p[:2], p[:2], occupied, radius_m=.45)
        footprint = mapper.surface.footprint(geometry, [0., 0.], 0., now_ns=now_ns, persistent=True)
        target = None; target_evidence = None; goal_body = None; connector = None; unknown = []
        action = None; exhausted = False; error = None
        if proposal['route_cells']:
            points = [centre(c) for c in proposal['route_cells']]
            target = next((q for q in points if np.linalg.norm(q-p[:2]) >= .35), points[-1])
            target, target_evidence = terminal_target(proposal, target, p[:2],
                (B@np.r_[self.goal, 0.])[:2], mapper.floor, mapper.occupied)
            goal_body = (R.T@np.r_[target-p[:2], 0.])[:2]
            connector = nominal_connector(p[:2], target, occupied, radius_m=.45)
            unknown = sorted(segment_cells(p[:2], target)-set(mapper.floor))
        if target is not None and np.linalg.norm(goal_body) > LOCAL_TARGET_RADIUS_M:
            self.mode = 'WAYPOINT'; error = math.atan2(goal_body[1], goal_body[0])
            if abs(error) > HEADING_TOLERANCE_RAD:
                action = 'left_turn' if error > 0 else 'right_turn'
            elif connector['nominal_disk_connector_clear']:
                action = 'forward'
        else:
            self.mode = 'VIEW_ACQUISITION'
            action, exhausted, error = self.scan(mapper, p, R, heading)
        if not current['nominal_disk_connector_clear'] or footprint['possible_intersection']:
            action = None
        return dict(action=action, requested_command=[0., 0., 0.] if action is None else candidate_commands(action)[0],
            mode=self.mode, proposal=proposal, waypoint_map_xy_m=None if target is None else target.tolist(),
            goal_body_xy_m=None if goal_body is None else goal_body.tolist(), terminal_goal_target_evidence=target_evidence,
            intermediate_target_is_mission_goal=bool(target_evidence and target_evidence['selected']),
            view_budget_exhausted=exhausted, scan_target_map_yaw_rad=self.scan_target, scan_index=self.scan_index,
            scan_sign=self.scan_sign, measured_heading_map_rad=heading, heading_error_rad=error,
            current_nominal_clearance=current, current_surface_check=footprint,
            measured_waypoint_connector=connector, unknown_waypoint_connector_cells=[list(c) for c in unknown],
            unknown_connector_motion_permitted=bool(action == 'forward' and unknown),
            unobserved_space_certified=False, current_geometry_checked=True, learned_model_used=False, candidate_future_outcomes_evaluated=False,
            command_integrated_pose_used=False, native_state_used=False,
            actual_commitment_horizon_ns=100_000_000, predictive_surface_or_path_gates_applied=False,
            clearance_certified=False, future_articulated_motion_certified=False,
            rule='turn_to_observed_waypoint_then_forward_on_nominal_connector_unknown_recorded')
