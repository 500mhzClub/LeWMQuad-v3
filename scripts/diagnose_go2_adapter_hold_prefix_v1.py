"""Immutable observed-decision prefix diagnosis; active physics remains untouched."""
from collections import Counter
from itertools import islice
import hashlib
import json
import math
from lewm.geometry_progress_pilot_development import ACTIONS
from scripts import run_go2_all_phase_adapter_maze02_matched_native_v1 as original
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json, verify
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.maze_decision_stream_development import read_rows, writer, NAME
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_adapter_hold_prefix_v1_attempt_001'
SOURCE = 'scripts/diagnose_go2_adapter_hold_prefix_v1.py'
PROTOCOL = 'docs/go2_adapter_hold_prefix_v1_2026-09-10.md'
TEST = 'lewm/tests/test_adapter_hold_prefix_diagnosis_development.py'
CASE = 'all_phase_full_jepa_residual_maze_02'
FRAMES = 1000
LAUNCH_SHA = '97c307ce8d2178494e7484f14b3b0cdcd8d7d0ceaab2f94d94d59da44e39b17a'
QUEUE = BASE/'go2_reached_frontier_maze03_native_wait_v1_attempt_001'
QUEUE_SHA = '68c10ea5a869d6236975372a525dc4586ba7ba16cbeefb17ff0fbb2b57c07a74'


def canonical(row):
    return (json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def compact(row):
    frame = row['tick']; d = row['decision']; s = d['new_selection']; mission = d['mission_receipt']
    if (type(frame) is not int or frame < 0 or row['observation_index'] != frame
            or row['pre_sample_index'] != 749+50*frame or d['tick'] != frame):
        raise ValueError('ordered original observation endpoints required')
    hold_required = None if mission is None else mission['hold_required']
    evidence = d['evidence']; pose = None if evidence is None else evidence.get('current_pose')
    result = dict(frame=frame, requested_command=d['requested_command'], terminal=d['terminal'],
        observed_goal_distance_m=d['observed_goal_distance_m'], mission_hold_required=hold_required,
        observed_arrivals=[] if mission is None else mission['arrivals'],
        observed_position_initial_body_m=None if pose is None else pose['position_initial_body_m'],
        action=None if s is None else s['action'], mode=None if s is None else s['mode'],
        discretionary_hold=bool(frame >= 3 and d['terminal'] is None and hold_required is False
            and s is not None and s['action'] == 'hold' and d['requested_command'] == [0., 0., 0.]),
        target_map_xy_m=None if s is None else s.get('waypoint_map_xy_m'),
        target_distance_m=None if s is None or s.get('goal_body_xy_m') is None else math.hypot(*s['goal_body_xy_m']),
        proposal_status=None if s is None else s['proposal']['status'],
        route_cell_count=None if s is None else len(s['proposal']['route_cells']),
        raw_gate_candidates=None, raw_feasible_moving_actions=[], higher_utility_path_vetoed_actions=[],
        raw_best_eligible_action=None, applied_feasibility_recoveries=[])
    if s is None: return result
    result['applied_feasibility_recoveries'] = [k for k in ('residual_first_interval_feasibility',
        'residual_hold_feasibility', 'residual_anchored_continuation') if s.get(k) is not None]
    if 'prediction' not in s: return result
    if ([c['action'] for c in s['candidates']] != list(ACTIONS)
            or len(s['surface_checks']) != 6 or len(s['nominal_path_checks']) != 6):
        raise ValueError('complete original six-candidate gate evidence required')
    candidates = []
    for i, candidate in enumerate(s['candidates']):
        candidates.append(dict(action=candidate['action'], utility_m=candidate['utility_m'],
            phase_allowed=candidate['action'] in s['phase_allowed_actions'],
            surface_clear=not s['surface_checks'][i]['possible_intersection'],
            raw_path_clear=s['nominal_path_checks'][i]['all_predicted_segments_nominally_clear']))
    for c in candidates: c['raw_gate_eligible'] = c['phase_allowed'] and c['surface_clear'] and c['raw_path_clear']
    hold = candidates[0]
    if hold['action'] != 'hold' or any(not math.isfinite(c['utility_m']) for c in candidates):
        raise ValueError('finite original utilities in fixed action order required')
    eligible = [c for c in candidates if c['raw_gate_eligible']]
    result.update(raw_gate_candidates=candidates,
        raw_feasible_moving_actions=[c['action'] for c in candidates[1:] if c['raw_gate_eligible']],
        higher_utility_path_vetoed_actions=[c['action'] for c in candidates[1:] if c['utility_m'] > hold['utility_m']
            and c['phase_allowed'] and c['surface_clear'] and not c['raw_path_clear']],
        raw_best_eligible_action=None if not eligible else max(eligible, key=lambda c: c['utility_m'])['action'])
    return result


def summarize(rows):
    if not rows or any(r['frame'] != i for i, r in enumerate(rows)):
        raise ValueError('complete ordered observed prefix required')
    runs = []; first = None
    for i, row in enumerate(rows):
        if row['discretionary_hold'] and first is None: first = i
        if first is not None and (not row['discretionary_hold'] or i == len(rows)-1):
            last = i if row['discretionary_hold'] else i-1
            segment = rows[first:last+1]; p = segment[0]['observed_position_initial_body_m']; q = segment[-1]['observed_position_initial_body_m']
            distances = [r['target_distance_m'] for r in segment if r['target_distance_m'] is not None]
            runs.append(dict(first_frame=first, last_frame=last, observations=len(segment),
                observation_span_s=(last-first)*.1,
                initial_observed_goal_distance_m=segment[0]['observed_goal_distance_m'],
                final_observed_goal_distance_m=segment[-1]['observed_goal_distance_m'],
                observed_position_estimate_net_change_m=None if p is None or q is None else math.dist(p[:2], q[:2]),
                target_counts=dict(Counter(json.dumps(r['target_map_xy_m']) for r in segment)),
                target_distance_min_m=min(distances) if distances else None,
                target_distance_max_m=max(distances) if distances else None))
            first = None
    holds = [r for r in rows if r['discretionary_hold']]
    return dict(frames=len(rows), requested_commands=dict(Counter(json.dumps(r['requested_command']) for r in rows)),
        action_counts=dict(Counter(str(r['action']) for r in rows)), discretionary_hold_observations=len(holds),
        hold_with_raw_feasible_movement=sum(bool(r['raw_feasible_moving_actions']) for r in holds),
        hold_is_highest_saved_utility_among_raw_eligible=sum(r['raw_best_eligible_action'] == 'hold' for r in holds),
        hold_with_higher_utility_path_vetoed=sum(bool(r['higher_utility_path_vetoed_actions']) for r in holds),
        hold_raw_feasible_movement_sets=dict(Counter(json.dumps(r['raw_feasible_moving_actions']) for r in holds)),
        hold_higher_utility_path_vetoed_sets=dict(Counter(json.dumps(r['higher_utility_path_vetoed_actions']) for r in holds)),
        hold_applied_recovery_sets=dict(Counter(json.dumps(r['applied_feasibility_recoveries']) for r in holds)),
        longest_discretionary_hold_runs=sorted(runs, key=lambda r: (-r['observations'], r['first_frame']))[:10],
        prefix_terminal_frames=[r['frame'] for r in rows if r['terminal'] is not None],
        observed_arrivals_at_prefix_end=rows[-1]['observed_arrivals'],
        initial_observed_goal_distance_m=rows[0]['observed_goal_distance_m'],
        final_observed_goal_distance_m=rows[-1]['observed_goal_distance_m'],
        native_pose_or_contact_read=False, native_command_completion_audited=False,
        raw_eligibility_does_not_include_recovery_overrides=True, model_inference_performed=False,
        alternate_action_outcomes_inferred=False, complete_episode_outcome_claimed=False, navigation_qualified=False)


def main():
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive fixed-prefix diagnosis required')
    verify_artifacts(original.OUTPUT, {'launch.json': LAUNCH_SHA}); verify_artifacts(QUEUE, {'launch.json': QUEUE_SHA})
    launch = read_json(original.OUTPUT, 'launch.json'); queued = read_json(QUEUE, 'launch.json')
    if CASE != original.CASES[0][0] or any(queued['source_sha256'].get(n) != h for n, h in launch['source_sha256'].items()):
        raise ValueError('fixed original first adapter model and unchanged queue required')
    sources = discover_sources((SOURCE, PROTOCOL, TEST), queued['source_sha256']); verify(sources)
    resources = hardware()
    if resources['memory_available_bytes'] < 8*1024**3 or resources['artifact_free_bytes'] < 41*1024**3:
        raise ValueError('bounded diagnosis resources unavailable')
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, original_launch_sha256=LAUNCH_SHA,
        original_source_root=str(original.OUTPUT), original_case=CASE, frames=FRAMES,
        queued_launch_sha256=QUEUE_SHA, output_root=str(OUTPUT), protocol=PROTOCOL, hardware=resources,
        model_state_sha256=launch['assigned_model_states'][original.CASES[0][4]],
        full_case_completion_required=False, native_execution=False, model_inference=False,
        original_live_stream_mutated=False, fixed_prefix_only=True))
    print('ADAPTER_HOLD_PREFIX_DIAGNOSIS_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    try:
        hasher = hashlib.sha256(); rows = []
        with writer(OUTPUT) as append, (OUTPUT/'frame_summary.jsonl').open('x') as summary:
            for row in islice(read_rows(original.OUTPUT/CASE), FRAMES):
                if row['tick'] != len(rows): raise ValueError('complete first 1000 observations required')
                hasher.update(canonical(row)); append(row); reduced = compact(row); rows.append(reduced)
                summary.write(json.dumps(reduced, allow_nan=False)+'\n')
        if len(rows) != FRAMES: raise ValueError('original prefix is not complete; no replacement snapshot')
        second = hashlib.sha256(); count = 0
        for row in islice(read_rows(original.OUTPUT/CASE), FRAMES): second.update(canonical(row)); count += 1
        if count != FRAMES or second.hexdigest() != hasher.hexdigest():
            raise ValueError('original completed prefix changed during snapshot')
        report = summarize(rows); verify(sources)
        verify_artifacts(original.OUTPUT, {'launch.json': LAUNCH_SHA}); verify_artifacts(QUEUE, {'launch.json': QUEUE_SHA})
        ids = {n: digest(OUTPUT/n) for n in ('launch.json', NAME, 'frame_summary.jsonl')}; verify_artifacts(OUTPUT, ids)
        write_json(OUTPUT/'result.json', dict(status='ADAPTER_HOLD_PREFIX_DIAGNOSIS_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, canonical_original_prefix_sha256=hasher.hexdigest(),
            prefix_reread_identical=True, report=report, snapshots=[rows[i] for i in (3, 300, 600, 999)],
            original_full_episode_audit_pending=True, native_execution=False, goal_achieved=False))
        print('ADAPTER_HOLD_PREFIX_DIAGNOSIS_COMPLETE', digest(OUTPUT/'result.json'),
            'holds', report['discretionary_hold_observations'], flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_ADAPTER_HOLD_PREFIX_DIAGNOSIS_FAILURE', reason=repr(error))); raise


if __name__ == '__main__': main()
