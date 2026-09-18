"""Read-only recovery forecast/execution and later observed nominal evidence."""
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.physical_execution_development import rotation_xyzw


def collect_entries(rows):
    entries = []; transitions = []; pending = None
    for row in rows:
        d = row['decision']; s = d['new_selection'] or {}
        if s.get('nominal_clearance_reentry'):
            action = s['action']; index = ACTIONS.index(action)
            check = s['reentry_candidates'][index]
            if (d['terminal'] is not None or action == 'hold' or d['selected_action'] != action
                    or not check['eligible'] or check['action'] != action
                    or d['requested_command'] != candidate_commands(action)[0]):
                raise ValueError('actual nonzero recovery request and matching candidate required')
            entries.append(dict(tick=row['tick'], action=action, requested_command=d['requested_command'],
                predicted_body_xy_m=s['prediction'][index][0][:2],
                current_clearance=s['reentry_current_clearance'], recovery_candidate=check))
            if pending is None: pending = row['tick']
        elif pending is not None and s.get('action') is not None:
            action = s['action']; path = s['nominal_path_checks'][ACTIONS.index(action)]
            if not path['all_predicted_segments_nominally_clear']:
                raise ValueError('ordinary resumed selection must retain original nominal gate')
            # The first checked segment includes the current observed point,
            # so its minimum is a lower bound on that point's map clearance.
            first = path['segments'][0]
            if first['radius_m'] != .45 or not first['nominal_disk_connector_clear']:
                raise ValueError('unchanged original nominal radius required')
            transitions.append(dict(first_reentry_tick=pending, ordinary_selection_tick=row['tick'],
                action=action, observed_current_clearance_lower_bound_m=first['minimum_observed_cell_distance_m'],
                evidence_scope='current observed map nominal gate only', physical_clearance_certified=False))
            pending = None
    return dict(entries=entries, observed_nominal_reentry_transitions=transitions,
        unresolved_reentry_start_tick=pending, physical_reentry_guaranteed=False)


def executed_motion(poses, tape, entries):
    poses = np.asarray(poses, float)
    if (poses.ndim != 2 or poses.shape[1] != 7 or not np.isfinite(poses).all()
            or not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1., atol=1e-6, rtol=0)):
        raise ValueError('finite normalized native pose trace required')
    records = []; previous = -1
    for entry in entries:
        tick = entry['tick']; prediction = np.asarray(entry['predicted_body_xy_m'], float)
        if (type(tick) is not int or not previous < tick < len(tape)
                or prediction.shape != (2,) or not np.isfinite(prediction).all()):
            raise ValueError('ordered executed recovery forecast required')
        item = tape[tick]; start = 749+50*tick
        if (item['tick'] != tick or item['pre_sample_index'] != start
                or item['requested_command'] != entry['requested_command']
                or item['requested_command'] != candidate_commands(entry['action'])[0]):
            raise ValueError('native tape must dispatch the exact forecasted recovery command')
        previous = tick
        if not item['completed']:
            records.append(entry | dict(complete_100ms_execution=False, native_body_xy_m=None,
                forecast_xy_error_m=None, unexecuted_endpoint_inferred=False))
            continue
        end = start+50
        if item['post_sample_index'] != end or end >= len(poses):
            raise ValueError('complete native 100ms endpoint required')
        actual = (rotation_xyzw(poses[start, 3:]).T@(poses[end, :3]-poses[start, :3]))[:2]
        records.append(entry | dict(complete_100ms_execution=True, native_body_xy_m=actual.tolist(),
            forecast_xy_error_m=float(np.linalg.norm(prediction-actual)),
            native_pose_is_evaluator_only=True, unexecuted_endpoint_inferred=False))
    return records
