"""Offline completed-stream and native-motion diagnosis; no policy rescoring."""
from collections import Counter
import json
import numpy as np

from lewm.geometry_progress_pilot_development import ACTIONS
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json, verify
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.maze_decision_stream_development import read_rows, NAME as DECISIONS
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

INPUT = BASE/'go2_recent_qualified_direct_flow_maze03_pilot_v1_attempt_001'
CASE = 'full_jepa_recent_qualified_direct_flow_maze_03'
RESULT_SHA = '330ae2381254538f43f5bf1d5374ba20ebea657b7d2749477468276e1d5ee723'
OUTPUT = BASE/'go2_recent_qualified_maze03_frontier_stagnation_v1_attempt_001'
SOURCE = 'scripts/diagnose_go2_recent_qualified_maze03_frontier_stagnation_v1.py'
PROTOCOL = 'docs/go2_recent_qualified_maze03_frontier_stagnation_v1_2026-09-10.md'
TEST = 'lewm/tests/test_recent_qualified_maze03_frontier_stagnation_development.py'
SNAPSHOTS = (3, 50, 100, 250, 500, 1000, 1500, 2000, 2500, 2793, 2794)


def motion_summary(poses):
    poses = np.asarray(poses, float)
    if poses.ndim != 2 or poses.shape[1] != 7 or not len(poses) or not np.isfinite(poses).all():
        raise ValueError('complete finite native poses required')
    xy = poses[:, :2]; q = poses[:, 3:]
    if not np.allclose(np.linalg.norm(q, axis=1), 1., atol=1e-6, rtol=0):
        raise ValueError('unit native xyzw quaternions required')
    yaw = np.unwrap(np.arctan2(2*(q[:, 3]*q[:, 2]+q[:, 0]*q[:, 1]), 1-2*(q[:, 1]**2+q[:, 2]**2)))
    return dict(samples=len(poses), start_xy_m=xy[0].tolist(), end_xy_m=xy[-1].tolist(),
        minimum_xy_m=xy.min(0).tolist(), maximum_xy_m=xy.max(0).tolist(),
        net_displacement_m=float(np.linalg.norm(xy[-1]-xy[0])),
        maximum_displacement_from_segment_start_m=float(np.linalg.norm(xy-xy[0], axis=1).max()),
        sampled_xy_path_length_m=float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
        signed_yaw_revolutions=float((yaw[-1]-yaw[0])/(2*np.pi)),
        absolute_sampled_yaw_revolutions=float(np.abs(np.diff(yaw)).sum()/(2*np.pi)))


def selection_summary(selection):
    if not selection:
        return None
    proposal = selection['proposal']; candidates = []
    if 'prediction' in selection:
        if ([row['action'] for row in selection['candidates']] != list(ACTIONS)
                or [row['action'] for row in selection['nominal_path_checks']] != list(ACTIONS)):
            raise ValueError('complete original ordered candidates and path checks required')
        for index, row in enumerate(selection['candidates']):
            candidates.append(dict(action=row['action'], utility_m=row['utility_m'],
                executed_distance_progress_m=row.get('executed_waypoint_distance_progress_m'),
                executed_alignment_progress_m=row.get('executed_waypoint_alignment_progress_m'),
                full_plan_contact_score=row.get('full_plan_contact_score'),
                phase_allowed=row['action'] in selection['phase_allowed_actions'],
                surface_clear=not selection['surface_checks'][index]['possible_intersection'],
                nominal_path_clear=selection['nominal_path_checks'][index]['all_predicted_segments_nominally_clear']))
        for row in candidates:
            row['eligible_by_saved_gates'] = row['phase_allowed'] and row['surface_clear'] and row['nominal_path_clear']
    waypoint = selection.get('waypoint_map_xy_m'); goal = selection.get('goal_body_xy_m')
    return dict(action=selection['action'], mode=selection['mode'], score_contract=selection.get('score_contract'),
        proposal_status=proposal['status'], route_cell_count=len(proposal['route_cells']),
        route_cells=proposal['route_cells'] if len(proposal['route_cells']) <= 2 else None,
        waypoint_map_xy_m=waypoint,
        waypoint_distance_m=None if waypoint is None else float(np.linalg.norm(goal)),
        observed_floor_cells=proposal['observed_floor_cells'], occupied_cells=proposal['occupied_cells'],
        frontier_cells=proposal.get('frontier_cells'), view_budget_exhausted=selection['view_budget_exhausted'],
        intermediate_target_is_mission_goal=selection.get('intermediate_target_is_mission_goal', False),
        nominal_clearance_reentry=selection.get('nominal_clearance_reentry', False), candidates=candidates)


def main():
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive diagnostic attempt required')
    verify_artifacts(INPUT, {'result.json': RESULT_SHA}); result = read_json(INPUT, 'result.json')
    if result['status'] != 'RECENT_QUALIFIED_DIRECT_FLOW_MAZE03_PILOT_V1_COMPLETE':
        raise ValueError('completed original native experiment required')
    names = ['launch.json', CASE+'_audit.json', CASE+'_prefix_comparison.json', CASE+'_worker_terminal.json']
    names += [CASE+'/'+name for name in ('result.json', 'physics_trace.npz', 'command_tape.json', DECISIONS)]
    ids = {name: result['artifact_sha256'][name] for name in names} | {'result.json': RESULT_SHA}
    verify_artifacts(INPUT, ids); old = read_json(INPUT, 'launch.json')
    if old['source_sha256'] != result['source_sha256']: raise ValueError('original source identity mismatch')
    sources = discover_sources((SOURCE, PROTOCOL, TEST), result['source_sha256']); verify(sources)
    resources = hardware()
    if resources['memory_available_bytes'] < 8*1024**3 or resources['artifact_free_bytes'] < 40*1024**3+64*1024**2:
        raise ValueError('bounded offline diagnosis resource allowance unavailable')
    collection = read_json(INPUT/CASE, 'result.json'); tape = read_json(INPUT/CASE, 'command_tape.json')
    audit = read_json(INPUT, CASE+'_audit.json')
    if (collection['decisions'] != 2805 or collection['completed_ticks'] != 2804
            or audit['verified_round_trip'] is not False or audit['raw_model_command_replay_pass'] is not True):
        raise ValueError('exact completed negative original episode required')
    with np.load(INPUT/CASE/'physics_trace.npz', allow_pickle=False) as saved:
        poses = saved['base_pose_world']
    if len(poses) != collection['physics_samples']: raise ValueError('complete bound native trace required')
    create_output(OUTPUT)
    launch = dict(source_sha256=sources, original_input_sha256=ids, output_root=str(OUTPUT), protocol=PROTOCOL,
        snapshots=list(SNAPSHOTS), hardware=resources, native_execution=False, model_loaded=False,
        policy_rescoring_performed=False, native_state_used_only_for_offline_diagnosis=True)
    write_json(OUTPUT/'launch.json', launch)
    print('FRONTIER_STAGNATION_DIAGNOSIS_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    try:
        modes = Counter(); statuses = Counter(); actions = Counter(); targets = Counter(); forward = Counter()
        snapshots = []; distances = []; singleton_distances = []; runs = []; run = None; frames = 0
        first_terminal = None
        with (OUTPUT/'frame_summary.jsonl').open('x') as out:
            for row in read_rows(INPUT/CASE):
                frame = row['tick']; decision = row['decision']; selected = selection_summary(decision['new_selection'])
                if frame != frames: raise ValueError('complete ordered decision population required')
                frames += 1
                if frame < len(tape) and tape[frame]['requested_command'] != decision['requested_command']:
                    raise ValueError('saved decision and actual original requested command differ')
                if decision['terminal'] is not None and first_terminal is None:
                    first_terminal = dict(frame=frame, terminal=decision['terminal'], failure=decision['failure'])
                summary = dict(frame=frame, requested_command=decision['requested_command'], terminal=decision['terminal'],
                    observed_goal_distance_m=decision['observed_goal_distance_m'], selection=selected)
                out.write(json.dumps(summary, allow_nan=False)+'\n')
                if frame in SNAPSHOTS: snapshots.append(summary)
                key = None
                if selected is not None and decision['terminal'] is None:
                    modes[selected['mode']] += 1; statuses[selected['proposal_status']] += 1; actions[str(selected['action'])] += 1
                    if selected['waypoint_map_xy_m'] is not None:
                        key = tuple(selected['waypoint_map_xy_m']); targets[str(key)] += 1
                        distances.append(selected['waypoint_distance_m'])
                        if selected['route_cell_count'] == 1: singleton_distances.append(selected['waypoint_distance_m'])
                    if selected['candidates']:
                        candidate = next(c for c in selected['candidates'] if c['action'] == 'forward')
                        forward['selection_frames'] += 1
                        forward['phase_allowed'] += int(candidate['phase_allowed'])
                        forward['surface_clear'] += int(candidate['surface_clear'])
                        forward['nominal_path_clear'] += int(candidate['nominal_path_clear'])
                        forward['all_gates_eligible'] += int(candidate['eligible_by_saved_gates'])
                        if candidate['eligible_by_saved_gates'] and selected['action'] in ('left_turn', 'right_turn'):
                            forward['turn_selected_with_forward_eligible'] += 1
                if run is not None and key != run['target']:
                    runs.append(run); run = None
                if key is not None:
                    if run is None: run = dict(target=key, first_frame=frame, last_frame=frame)
                    else: run['last_frame'] = frame
            if run is not None: runs.append(run)
        if frames != collection['decisions'] or [row['frame'] for row in snapshots] != list(SNAPSHOTS):
            raise ValueError('complete declared decision and snapshot population required')
        longest = sorted(runs, key=lambda row: (-(row['last_frame']-row['first_frame']+1), row['first_frame']))[:10]
        for run in longest:
            run['observations'] = run['last_frame']-run['first_frame']+1
            run['native_motion_between_observations'] = motion_summary(poses[
                749+50*run['first_frame']:750+50*run['last_frame']])
        def distribution(values):
            return dict(count=len(values), minimum_m=float(np.min(values)), median_m=float(np.median(values)),
                p95_m=float(np.quantile(values, .95)), maximum_m=float(np.max(values))) if values else dict(count=0)
        verify(sources); verify_artifacts(INPUT, ids)
        outputs = {name:digest(OUTPUT/name) for name in ('launch.json', 'frame_summary.jsonl')}
        write_json(OUTPUT/'result.json', dict(status='RECENT_QUALIFIED_MAZE03_FRONTIER_STAGNATION_COMPLETE',
            source_sha256=sources, original_input_sha256=ids, artifact_sha256=outputs, frames=frames,
            first_terminal=first_terminal, mode_counts=dict(modes), proposal_status_counts=dict(statuses),
            selected_actions=dict(actions), forward_gate_counts=dict(forward),
            target_counts=dict(targets), longest_identical_target_runs=longest,
            waypoint_distance=distribution(distances), singleton_route_waypoint_distance=distribution(singleton_distances),
            native_2ms_motion=motion_summary(poses[749:]), native_command_endpoint_motion=motion_summary(poses[749::50]),
            sampled_path_length_includes_gait_oscillation=True,
            completed_requested_command_counts=dict(Counter(str(t['requested_command']) for t in tape if t['completed'])),
            native_crossings=len(audit['native_evaluation']['outbound_traversal']['crossings']),
            observed_arrivals=collection['mission_receipt']['arrivals'], snapshots=snapshots,
            source_frontier_target_has_reached_or_retired_state=False,
            model_loaded=False, native_execution=False, policy_rescoring_performed=False,
            unexecuted_outcomes_inferred=False, navigation_qualified=False, goal_achieved=False))
        print('FRONTIER_STAGNATION_DIAGNOSIS_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_FRONTIER_STAGNATION_DIAGNOSIS_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
