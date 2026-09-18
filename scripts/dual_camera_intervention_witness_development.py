"""Check saved first-intervention evidence and count current camera measurements."""
from lewm.dual_camera_native_admission_development import admit_prefix
from scripts.maze_decision_stream_development import read_rows
from scripts.startup_raw_sensor_audit_development import read_json


def admit_intervention(directory, report):
    admit_prefix(report)
    intervention = read_json(directory, 'intervention.json')
    changed = report['first_auxiliary_intervention_frame']
    if (intervention['frame'] != changed or intervention['first_auxiliary_attempt'] is not True
            or intervention['following_recorded_observations_consumed'] is not False):
        raise ValueError('bound first intervention without following observations required')
    candidate = intervention['candidate']; raw = candidate['original_visual_evidence'] or {}
    evidence = candidate['evidence'] or {}
    if (candidate['terminal'] != report['final_terminal'] or candidate['failure'] != report['final_failure']
            or candidate['requested_command'] != report['final_requested_command']
            or (evidence.get('current_pose') or {}).get('frame') != changed
            or (raw.get('current_pose') or {}).get('frame') != changed
            or raw.get('camera_selection_current') is not True
            or (raw.get('camera_selection') or {}).get('auxiliary_attempted') is not True
            or (raw.get('camera_selection') or {}).get('selected_camera') != 'auxiliary'):
        raise ValueError('summary must match the active measured intervention decision')
    checked = 0
    for i, row in enumerate(read_rows(directory)):
        if i > changed or row['tick'] != i:
            raise ValueError('no later or unordered prospective decisions allowed')
        decision = row['decision']; choice = (decision['original_visual_evidence'] or {}).get('camera_selection') or {}
        if i < changed:
            if choice.get('auxiliary_attempted') or row.get('first_auxiliary_intervention'):
                raise ValueError('earlier auxiliary intervention in saved prefix')
            if decision['terminal'] is not None or decision['failure'] is not None:
                raise ValueError('active preintervention controller required')
        elif row.get('first_auxiliary_intervention') is not True or decision != candidate:
            raise ValueError('saved final decision must equal intervention evidence')
        checked += 1
    if checked != report['frames']:
        raise ValueError('complete saved prospective stream required')
    return dict(frames=checked, first_auxiliary_intervention_frame=changed,
        saved_intervention_matches_summary_and_stream=True,
        no_later_recorded_observations=True, current_registered_pose_available=True)


def camera_execution(rows):
    attempts = []; selected = []; primary = []; retained = []; seen_failures = set()
    for row in rows:
        tick = row['tick']; raw = row['decision']['original_visual_evidence'] or {}
        choice = raw.get('camera_selection') or {}; pose = raw.get('current_pose')
        current = bool(raw.get('camera_selection_current') and pose is not None and pose['frame'] == tick)
        failure = raw.get('terminal_failure'); new_failure = False
        if failure is not None:
            stamp = failure['decision_ns']; new_failure = stamp not in seen_failures
            seen_failures.add(stamp)
        if choice.get('auxiliary_attempted'):
            if current or new_failure: attempts.append(tick)
            else: retained.append(tick)
        if current and choice.get('selected_camera') == 'auxiliary': selected.append(tick)
        if current and choice.get('selected_camera') == 'primary': primary.append(tick)
    return dict(auxiliary_attempt_frames=attempts, auxiliary_selected_frames=selected,
        primary_selected_frames=primary, auxiliary_attempt_count=len(attempts),
        auxiliary_selected_count=len(selected), primary_selected_count=len(primary),
        retained_auxiliary_snapshot_frames=retained, retained_snapshots_counted_as_new_measurements=False,
        source='executed complete decisions', counterfactual_outcomes_inferred=False)
