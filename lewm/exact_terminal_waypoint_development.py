"""Score the actual mission point when the terminal goal-cell waypoint is selected."""
from copy import deepcopy
import numpy as np
from lewm.observed_floor_waypoint_development import CELL_M, centre
from lewm.joint_rgbd_rigid_pose_development import proper
from lewm.commitment_pose_goal_probe_development import apply_waypoint_utility
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands


def exact_terminal_target(selection, position_map, rotation_map_body, mission_map_xy):
    proposal = selection.get('proposal') or {}
    if (selection.get('mode') != 'WAYPOINT'
            or proposal.get('status') != 'OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL'):
        return selection
    path = proposal['route_cells']
    if not path:
        raise ValueError('observed goal route requires a terminal cell')
    terminal = centre(path[-1])
    if not np.array_equal(selection['waypoint_map_xy_m'], terminal):
        return selection
    p, R, g = np.asarray(position_map, float), np.asarray(rotation_map_body, float), np.asarray(mission_map_xy, float)
    if (p.shape != (3,) or g.shape != (2,) or not np.isfinite(p).all()
            or not np.isfinite(g).all() or np.max(np.abs(np.r_[p[:2], g])) > 4.9):
        raise ValueError('finite bounded observed mission coordinates required')
    proper(R)
    if list(map(int, np.floor(g/CELL_M))) != list(path[-1]):
        raise ValueError('exact mission point must remain inside the observed goal cell')
    if (selection['phase_allowed_actions'] != list(ACTIONS)
            or selection['nominal_constraint_horizon_ns'] != 500_000_000
            or [c['action'] for c in selection['nominal_action_checks']] != list(ACTIONS)
            or len(selection['surface_checks']) != 6):
        raise ValueError('complete current six-action constraint witnesses required')
    result = deepcopy(selection)
    result['before_exact_terminal_target'] = dict(action=selection['action'],
        waypoint_map_xy_m=selection['waypoint_map_xy_m'], goal_body_xy_m=selection['goal_body_xy_m'])
    result['waypoint_map_xy_m'] = g.tolist()
    result['goal_body_xy_m'] = (R.T @ np.r_[g-p[:2], 0.])[:2].tolist()
    result = apply_waypoint_utility(result)
    result['before_nominal_constraint_action'] = result['action']
    result['before_nominal_constraint_admissible_candidates'] = result['phase_admissible_candidates']
    feasible = []
    for i, (surface, nominal) in enumerate(zip(result['surface_checks'], result['nominal_action_checks'], strict=True)):
        if type(surface['possible_intersection']) is not bool or type(nominal['nominal_disk_connector_clear']) is not bool:
            raise ValueError('explicit current constraint booleans required')
        if not surface['possible_intersection'] and nominal['nominal_disk_connector_clear']:
            feasible.append(i)
    chosen = max(feasible, key=lambda i: result['candidates'][i]['utility_m']) if feasible else None
    action = None if chosen is None else ACTIONS[chosen]
    result.update(action=action, action_index=chosen,
        requested_command=[0., 0., 0.] if action is None else candidate_commands(action)[0],
        phase_admissible_candidates=len(feasible), intermediate_target_is_mission_goal=True,
        exact_terminal_target_rule='actual_mission_point_inside_selected_observed_goal_cell',
        forecast_and_geometric_constraints_unchanged=True,
        terminal_target_offset_m=float(np.linalg.norm(g-terminal)))
    return result
