"""Actual 100 ms motion following an executed nearer-route target selection."""
import numpy as np
from lewm.geometry_progress_pilot_development import candidate_commands
from lewm.physical_execution_development import rotation_xyzw


def connector_execution(poses, tape, rows):
    poses = np.asarray(poses, float)
    if (poses.ndim != 2 or poses.shape[1] != 7 or not np.isfinite(poses).all()
            or not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1., atol=1e-6, rtol=0)):
        raise ValueError('finite normalized native pose trace required')
    result = []; previous = -1
    for row in rows:
        decision = row['decision']; selection = decision.get('new_selection') or {}
        if not selection.get('nearer_observed_route_target', False):
            continue
        tick = row['tick']; action = selection['action']
        original = selection['original_waypoint_selection']
        if (type(tick) is not int or not previous < tick < len(tape)
                or decision['terminal'] is not None or original['action'] is not None
                or action not in ('forward', 'left_turn', 'right_turn')
                or decision['selected_action'] != action
                or decision['requested_command'] != candidate_commands(action)[0]
                or selection['requested_command'] != decision['requested_command']
                or original['measured_waypoint_connector']['nominal_disk_connector_clear']
                or not selection['measured_waypoint_connector']['nominal_disk_connector_clear']
                or not selection['current_nominal_clearance']['nominal_disk_connector_clear']
                or selection['current_surface_check']['possible_intersection']
                or not selection['original_nominal_radius_preserved']
                or any(c['radius_m'] != .45 for c in
                    (original['measured_waypoint_connector'], selection['measured_waypoint_connector'],
                     selection['current_nominal_clearance']))):
            raise ValueError('ordered executed fallback with unchanged original clearance required')
        item = tape[tick]; start = 749+50*tick
        if (item['tick'] != tick or item['pre_sample_index'] != start
                or item['requested_command'] != decision['requested_command']
                or not 0 <= start <= item['post_sample_index'] < len(poses)
                or item['post_sample_index'] > start+50):
            raise ValueError('native command tape must match the selected fallback')
        previous = tick
        entry = dict(tick=tick, action=action, requested_command=decision['requested_command'],
            original_waypoint_map_xy_m=original['waypoint_map_xy_m'],
            selected_waypoint_map_xy_m=selection['waypoint_map_xy_m'],
            original_route_target_index=selection['original_route_target_index'],
            selected_route_target_index=selection['selected_route_target_index'],
            original_measured_connector=original['measured_waypoint_connector'],
            selected_measured_connector=selection['measured_waypoint_connector'],
            current_nominal_clearance=selection['current_nominal_clearance'],
            complete_100ms_execution=bool(item['completed']), native_body_xy_m=None,
            native_pose_is_evaluator_only=True, unexecuted_endpoint_inferred=False,
            command_outcome_forecast_used=False, physical_clearance_certified=False)
        if item['completed']:
            end = start+50
            if item['post_sample_index'] != end or end >= len(poses):
                raise ValueError('complete native 100ms endpoint required')
            actual = rotation_xyzw(poses[start, 3:]).T@(poses[end, :3]-poses[start, :3])
            entry['native_body_xy_m'] = actual[:2].tolist()
        result.append(entry)
    return result
