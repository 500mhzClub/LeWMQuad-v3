"""Choose a nearer observed route point when the original connector is blocked.

This checks measured map geometry only. It does not forecast a command outcome.
"""
from copy import deepcopy
import math
import numpy as np
from lewm.geometry_progress_pilot_development import candidate_commands
from lewm.joint_rgbd_rigid_pose_development import proper
from lewm.observed_floor_waypoint_development import centre, segment_cells
from lewm.observed_geometry_refinement_development import nominal_connector
from lewm.exact_mission_target_development import terminal_target
from lewm.reactive_nominal_route_selection_development import HEADING_TOLERANCE_RAD, LOCAL_TARGET_RADIUS_M


def nearer_route_target(selection, position_map, rotation_map_from_body, floor, occupied):
    if (not selection or selection.get('mode') != 'WAYPOINT' or selection.get('action') is not None
            or selection.get('view_budget_exhausted', False)
            or selection.get('intermediate_target_is_mission_goal', False)
            or not selection['current_nominal_clearance']['nominal_disk_connector_clear']
            or selection['current_surface_check']['possible_intersection']
            or selection['measured_waypoint_connector']['nominal_disk_connector_clear']):
        return selection
    p = np.asarray(position_map, float); R = proper(rotation_map_from_body)
    cells = sorted(occupied); route = selection['proposal']['route_cells']
    target = np.asarray(selection['waypoint_map_xy_m'], float)
    if (p.shape != (3,) or target.shape != (2,) or not np.isfinite(p).all() or not np.isfinite(target).all()
            or selection.get('native_state_used') is not False or selection.get('learned_model_used') is not False
            or selection.get('candidate_future_outcomes_evaluated') is not False
            or selection.get('actual_commitment_horizon_ns') != 100_000_000
            or not route or len(route) > 40000):
        raise ValueError('original non-predictive selection and current observed map pose required')
    if (nominal_connector(p[:2], p[:2], cells, radius_m=.45) != selection['current_nominal_clearance']
            or nominal_connector(p[:2], target, cells, radius_m=.45) != selection['measured_waypoint_connector']):
        raise ValueError('original current and waypoint clearance must reconstruct exactly')
    points = [centre(c) for c in route]
    matches = [i for i, q in enumerate(points) if np.array_equal(q, target)]
    if len(matches) != 1: raise ValueError('unique original route-centre waypoint required')
    original_index = matches[0]; checks = []; chosen = None
    # Prefer the latest earlier point on the already observed route. Stop at
    # the first clear connector; do not jump past the original target/corner.
    for index in range(original_index-1, -1, -1):
        q = points[index]
        if tuple(route[index]) not in floor:
            raise ValueError('replacement route centre requires observed floor')
        distance = float(np.linalg.norm(q-p[:2]))
        check = nominal_connector(p[:2], q, cells, radius_m=.45)
        eligible = bool(distance > LOCAL_TARGET_RADIUS_M and check['nominal_disk_connector_clear'])
        checks.append(dict(route_index=index, waypoint_map_xy_m=q.tolist(),
            distance_from_current_pose_m=distance, measured_connector=check, eligible=eligible))
        if eligible:
            chosen = (index, q, check)
            break
    if chosen is None: return selection
    index, q, connector = chosen
    q, evidence = terminal_target(selection['proposal'], q, p[:2],
        selection['terminal_goal_target_evidence']['mission_goal_map_xy_m'], floor, occupied)
    if evidence['selected'] or not np.array_equal(q, points[index]):
        raise ValueError('nearer intermediate route point must not change the exact mission target')
    body = (R.T@np.r_[q-p[:2], 0.])[:2]; error = math.atan2(body[1], body[0])
    action = ('left_turn' if error > 0 else 'right_turn') if abs(error) > HEADING_TOLERANCE_RAD else 'forward'
    unknown = sorted(segment_cells(p[:2], q)-set(floor))
    result = deepcopy(selection)
    result.update(action=action, requested_command=candidate_commands(action)[0],
        waypoint_map_xy_m=q.tolist(), goal_body_xy_m=body.tolist(), heading_error_rad=error,
        terminal_goal_target_evidence=evidence, measured_waypoint_connector=connector,
        unknown_waypoint_connector_cells=[list(c) for c in unknown],
        unknown_connector_motion_permitted=bool(action == 'forward' and unknown),
        nearer_observed_route_target=True, original_waypoint_selection=deepcopy(selection),
        original_route_target_index=original_index, selected_route_target_index=index,
        nearer_route_connector_checks=checks, original_nominal_radius_preserved=True,
        rule='nearer_observed_route_point_when_original_nominal_connector_is_blocked')
    return result
