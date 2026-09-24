"""Explain the executed hold/turn stall using the authenticated complete receipts."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import numpy as np
from scripts import diagnose_go2_completed_hold_reorientation_maze02_v1 as prior
from scripts.maze_decision_stream_development import read_rows, NAME
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, verify, digest, write_json
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from lewm.geometry_progress_pilot_development import ACTIONS

SOURCE = 'scripts/diagnose_go2_hold_reorientation_stall_v1.py'
PRIOR_SHA = '8b85fcb4a5c58706ad84604ebcfd5885d94b831ef9110e528d8a1a0991c02cf2'
OUTPUT = ROOT/'docs/go2_hold_reorientation_stall_diagnosis_2026-09-11.json'
TRANSLATIONS = {'forward', 'left_arc', 'right_arc'}


def heading_change(first, last, normal):
    a = np.asarray(first)[:, 0]; b = np.asarray(last)[:, 0]; n = np.asarray(normal)
    a = a-n*float(a@n); b = b-n*float(b@n)
    if min(np.linalg.norm(a), np.linalg.norm(b)) < .2:
        raise ValueError('nonvertical observed heading required')
    return float(np.arctan2(n@np.cross(a, b), a@b))


def distribution(values):
    x = np.asarray(values, float)
    if not len(x) or not np.isfinite(x).all(): raise ValueError('finite nonempty population required')
    return dict(count=len(x), minimum=float(x.min()), median=float(np.median(x)),
        maximum=float(x.max()), total=float(x.sum()))


def main():
    if digest(prior.OUTPUT) != PRIOR_SHA: raise ValueError('exact completed floor diagnosis required')
    previous = json.loads(prior.OUTPUT.read_text())
    bindings = previous['source_sha256'] | {str(prior.OUTPUT.relative_to(ROOT)): PRIOR_SHA}
    sources = discover_sources((SOURCE,), bindings); verify(sources)
    root = prior.wait.native.OUTPUT
    verify_artifacts(root, {'result.json': prior.NATIVE_SHA, 'launch.json': prior.NATIVE_LAUNCH})
    result = read_json(root, 'result.json'); record = result['conditions'][0]
    name = record['case']; directory = root/name
    ids = {n:result['artifact_sha256'][n] for n in (name+'/'+NAME, name+'/command_tape.json')}
    verify_artifacts(root, ids)
    tape = read_json(directory, 'command_tape.json')
    rows = []; poses = {}; events = []; stream_hash = hashlib.sha256(); reference = None
    for row in read_rows(directory):
        frame = row['tick']; d = row['decision']; sel = d.get('new_selection') or {}
        stream_hash.update(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()+b'\n')
        if frame != len(rows): raise ValueError('complete ordered original decisions required')
        e = d.get('evidence') or {}; p = e.get('current_pose')
        if p is not None:
            if p['frame'] != frame: raise ValueError('current observed pose required')
            poses[frame] = p
        if frame == 0: reference = e['floor_registration']['reference']['joint_plane']['normal_body']
        checks = sel.get('nominal_path_checks', [])
        eligible = []
        if checks:
            if len(checks) != 6 or [c['action'] for c in checks] != list(ACTIONS):
                raise ValueError('complete original nominal path checks required')
            eligible = [a for i, a in enumerate(ACTIONS) if a in sel['phase_allowed_actions']
                and not sel['surface_checks'][i]['possible_intersection']
                and checks[i]['all_predicted_segments_nominally_clear']]
        event = sel.get('hold_reorientation')
        candidates = sel.get('candidates', [])
        summary = dict(frame=frame, action=d['selected_action'], command=d['requested_command'],
            terminal=d['terminal'], mode=d['planner_mode'], eligible=eligible,
            phase_allowed=sel.get('phase_allowed_actions'),
            surface_rejected=[] if not checks else [ACTIONS[i] for i,c in enumerate(sel['surface_checks']) if c['possible_intersection']],
            nominal_path_rejected=[c['action'] for c in checks if not c['all_predicted_segments_nominally_clear']],
            utility={c['action']:c['utility_m'] for c in candidates},
            waypoint_map_xy_m=sel.get('waypoint_map_xy_m'),
            goal_distance_m=d['observed_goal_distance_m'], reorientation=event is not None)
        if event:
            prediction = np.asarray(sel['prediction'], float); i = ACTIONS.index(event['selected_action'])
            if prediction.shape != (6, 8, 5): raise ValueError('original ordered forecast bank required')
            events.append(dict(frame=frame, selected_action=event['selected_action'], eligible=eligible,
                predicted_first_interval_yaw_rad=float(math.atan2(prediction[i,0,2], prediction[i,0,3])),
                original_hold_utility_m=event['original_hold_utility_m'],
                selected_utility_m=event['selected_utility_m']))
        if frame < len(tape):
            command = tape[frame]
            if (command['tick'] != frame or command['completed'] is not True
                    or command['requested_command'] != d['requested_command']
                    or command['pre_sample_index'] != 749+50*frame
                    or command['post_sample_index'] != 799+50*frame):
                raise ValueError('complete actual command and physical boundaries required')
        rows.append(summary)
    if (len(rows) != previous['complete_decisions'] or len(tape) != len(rows)-1
            or stream_hash.hexdigest() != previous['canonical_decision_stream_sha256']
            or [r['frame'] for r in events] != [e['frame'] for e in previous['hold_reorientation_events']]):
        raise ValueError('exact complete previously authenticated stream and interventions required')
    post = rows[405:1994]
    counters = dict(actions=dict(Counter(r['action'] for r in post)),
        any_translation_feasible=sum(bool(TRANSLATIONS.intersection(r['eligible'])) for r in post),
        hold_selected_with_translation_feasible=sum(r['action']=='hold' and bool(TRANSLATIONS.intersection(r['eligible'])) for r in post),
        translation_selected=sum(r['action'] in TRANSLATIONS for r in post),
        eligibility_sets=dict(Counter(','.join(r['eligible']) for r in post)),
        translation_surface_rejection_observations={a:sum(a in r['surface_rejected'] for r in post) for a in sorted(TRANSLATIONS)},
        translation_nominal_path_rejection_observations={a:sum(a in r['nominal_path_rejected'] for r in post) for a in sorted(TRANSLATIONS)})
    windows = []
    for event in events:
        frame = event['frame']
        if frame+1 not in poses: raise ValueError('actual next observed outcome required for every intervention')
        rotation = lambda f: poses[f]['rotation_initial_body_from_current_body']
        item = event | dict(observed_first_interval_yaw_rad=heading_change(rotation(frame), rotation(frame+1), reference))
        if frame+11 in poses and all(rows[f]['command']==[0.,0.,0.] for f in range(frame+1,frame+11)):
            changes = [heading_change(rotation(f), rotation(f+1), reference) for f in range(frame,frame+11)]
            item.update(ten_following_completed_holds=True,
                observed_turn_plus_ten_holds_yaw_rad=float(sum(changes)),
                observed_following_ten_holds_yaw_rad=float(sum(changes[1:])),
                observed_net_translation_m=float(np.linalg.norm(np.asarray(poses[frame+11]['position_initial_body_m'])-
                    np.asarray(poses[frame]['position_initial_body_m']))))
        else: item['ten_following_completed_holds'] = False
        windows.append(item)
    pulse = [w for w in windows if w['ten_following_completed_holds']]
    stats = {k:distribution([w[k] for w in population]) for k, population in (
        ('predicted_first_interval_yaw_rad', windows), ('observed_first_interval_yaw_rad', windows),
        ('observed_turn_plus_ten_holds_yaw_rad', pulse), ('observed_following_ten_holds_yaw_rad', pulse),
        ('observed_net_translation_m', pulse))}
    verify(sources); verify_artifacts(root, ids | {'result.json':prior.NATIVE_SHA,'launch.json':prior.NATIVE_LAUNCH})
    write_json(OUTPUT, dict(status='AUTHENTICATED_EXECUTED_HOLD_REORIENTATION_STALL_CHARACTERIZED',
        utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources, source_count=len(sources),
        prior_diagnosis_sha256=PRIOR_SHA, native_result_sha256=prior.NATIVE_SHA,
        reread_native_artifact_sha256=ids, canonical_decision_stream_sha256=stream_hash.hexdigest(),
        complete_decisions=len(rows), completed_commands=len(tape), post_intervention_frames=[405,1993],
        post_intervention_observations=len(post), post_intervention_counts=counters,
        pulse_statistics=stats, intervention_windows=windows, decision_summary=rows,
        poses_are_saved_observation_estimates=True, native_pose_used=False,
        full_native_artifact_rehash_reexecuted=False, visual_tracking_reexecuted=False,
        model_inference_reexecuted=False, unexecuted_actions_evaluated=False,
        policy_selected=False, native_execution=False, navigation_qualified=False,
        real_time_qualified=False, hardware_qualified=False, goal_achieved=False))
    print('HOLD_STALL_CHARACTERIZED', digest(OUTPUT), len(sources), flush=True)
    print('COUNTS', json.dumps(counters), flush=True)
    print('PULSE_STATISTICS', json.dumps(stats), flush=True)


if __name__ == '__main__': main()
