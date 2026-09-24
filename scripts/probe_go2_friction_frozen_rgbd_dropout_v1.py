"""Unchanged RGB-D estimators on audited friction tapes; predictions first."""
import json
import time

import cv2
import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.contact_dropout_evaluation_development import (
    stats, support_index, contact_window, dropout_spans, displacement_error, summarize_windows)
from lewm.joint_rgbd_rigid_pose_development import RigidRGBDKeyframePose, RIGID_RULES
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.audit_go2_joint_rgbd_rigid_pose_development_v1 import quaternion_rotation, angular_distance
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.probe_go2_joint_rgbd_pose_plane_development_v1 import plain
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json, read_npz
from scripts.startup_source_inventory_development import discover_sources

INPUT = ROOT/'.generated/go2_support_friction_collection_v1_attempt_001'
OUTPUT = ROOT/'.generated/go2_friction_frozen_rgbd_dropout_v1_attempt_001'
PROTOCOL = 'docs/go2_friction_frozen_rgbd_dropout_v1_2026-09-06.md'
CONDITIONS = ('nominal', 'lower_friction')
MODES = ('joint', 'gyro')
IDENTITIES = {
    'launch.json': '1602091d4f49713495798cb1ecd354294a164ca0a79cb003217348b6753a2980',
    'result.json': '39afad5afcc546f8017d1920b79cd524566c3c51f12a0ea0b8cc819ced38f175',
    'raw_support_audit_launch.json': 'a7628861af989a145b9b4d0c69d54464c93254888d57a294c4fec4487ca23198',
    'raw_support_audit.json': '8144a8bdd84dbd5534c7c166a043fd077101c371d44b97036e6337c3a9501673',
    'prefix_identity_audit_launch.json': 'd5d4d1dbbdd5b072e2196ea2fa7bafd467cd4c9251f01710b9ca4bc2975d1931',
    'prefix_identity_audit.json': '08e5a159eff5cf6b9e8edc237d3ee0b8444a5726c7ad3667590b02b4f5caf58f',
}


def preflight():
    ids = {str((INPUT/n).relative_to(ROOT)): h for n, h in IDENTITIES.items()}
    verify_bindings(ids)
    old = read_json(INPUT, 'launch.json'); verify(old)
    inherited = old['source_sha256']; inputs = old['input_sha256'] | ids
    for name in ('raw_support_audit_launch.json', 'prefix_identity_audit_launch.json'):
        prior = read_json(INPUT, name)
        verify_bindings(prior['source_sha256'] | prior['input_sha256'])
        inherited |= prior['source_sha256']; inputs |= prior['input_sha256']
    result = read_json(INPUT, 'result.json')
    if result['absent_expected_artifacts'] or any(result['conditions'][c]['rgbd_frames'] != 226 for c in CONDITIONS):
        raise ValueError('complete fixed paired recording required')
    inputs |= {str((INPUT/n).relative_to(ROOT)): h for n, h in result['artifact_sha256'].items()}
    sources = discover_sources((PROTOCOL, 'scripts/probe_go2_friction_frozen_rgbd_dropout_v1.py',
        'lewm/tests/test_contact_dropout_evaluation_development.py'), inherited)
    launch = {k: old[k] for k in ('native_sha256', 'native_geometry_sha256', 'opencv_binary_sha256', 'opencv_version', 'rules')}
    launch |= dict(source_sha256=sources, input_sha256=inputs, protocol=PROTOCOL,
                   rigid_rules=RIGID_RULES, conditions=CONDITIONS, models=MODES,
                   physics_executed=False, model_fitting=False,
                   scope='frozen pose replay and dropout-stratified native scoring; no navigation or estimator changes')
    verify(launch)
    return launch


def observe_once(model, failure, policy, depth, fast, now, frame):
    if failure is not None:
        return dict(status='NOT_REINVOKED_AFTER_FAILURE', state=None, failure=failure), failure
    start = time.perf_counter_ns()
    try:
        state = model.observe(policy, depth, fast, now_ns=now)
        item = dict(status='CONDITIONAL_RIGID_POSE', state=state)
    except SensorContractError as error:
        chain = []; cause = error
        while cause is not None:
            chain.append(str(cause)); cause = cause.__cause__
        failure = dict(frame=frame, measured_ns=now, chain=chain)
        item = dict(status='TERMINAL_FAILURE', state=None, failure=failure)
    item['observer_wall_ms'] = (time.perf_counter_ns()-start)/1e6
    return item, failure


def replay_condition(directory):
    models = {m: RigidRGBDKeyframePose(m) for m in MODES}
    failures = dict.fromkeys(MODES); rows = []
    for frame in range(226):
        policy, depth = load_rgbd_observation(directory, frame)
        fast = load_fast_packet(directory, frame); now = policy['sensor_state']['decision_ns']
        if now != 1_500_000_000+frame*100_000_000:
            raise ValueError('fixed 10Hz acquisition required')
        members = {}
        for mode in MODES:
            members[mode], failures[mode] = observe_once(models[mode], failures[mode], policy, depth, fast, now, frame)
        rows.append(dict(frame=frame, measured_ns=now, members=members,
                         depth_valid_pixels=int(np.asarray(depth['valid']).sum())))
        if frame % 50 == 0:
            print('FRICTION_RGBD_REPLAY', directory.name, frame, flush=True)
    return plain(dict(rows=rows, keyframes={m: models[m].nodes for m in MODES}, failures=failures))


def evaluate_condition(directory, record):
    raw = read_npz(directory, 'physics_trace.npz'); camera = read_json(directory, 'camera_audit.json')
    support = support_index(read_json(directory, 'support_predictions.json')['rows'])
    poses = raw['base_pose_world']; times = np.rint(raw['timestamp_s']*1e9).astype(np.int64)
    indices = [c['physical_sample_index'] for c in camera]
    if indices != list(range(749, 12000, 50)) or len(record['rows']) != len(indices):
        raise ValueError('complete audited camera/sample population required')
    rotations = [quaternion_rotation(poses[i, [6, 3, 4, 5]]) for i in indices]
    R0 = rotations[0]; p0 = poses[indices[0], :3]
    truth = [R0.T@(poses[i, :3]-p0) for i in indices]
    details = {}; summaries = {}; spans = dropout_spans(support)
    for mode in MODES:
        frames = []; windows = []
        for j, (row, sample) in enumerate(zip(record['rows'], indices, strict=True)):
            stamp = row['measured_ns']
            if stamp != int(times[sample]): raise ValueError('prediction/native clock mismatch')
            item = row['members'][mode]; state = item['state']
            missing = support[stamp]['modes']['level_sphere_rolling']['consensus_velocity_body_m_s'] is None
            frames.append(dict(frame=j, measured_ns=stamp, phase=int(raw['phase'][sample]),
                contact_unavailable_at_capture=missing, visual_available=state is not None,
                position_error_m=None if state is None else float(np.linalg.norm(np.asarray(state['position_initial_body_m'])-truth[j])),
                orientation_error_rad=None if state is None else angular_distance(np.asarray(state['rotation_initial_body_from_current_body']), R0.T@rotations[j]),
                registration=None if state is None else state['registration']))
            if not j: continue
            previous = record['rows'][j-1]
            window = contact_window(support, previous['measured_ns'], stamp)
            phases = sorted(set(int(v) for v in raw['phase'][indices[j-1]+1:sample+1]))
            window |= dict(frame=j, phases=phases,
                displacement_error_m=displacement_error(previous['members'][mode]['state'], state, truth[j-1], truth[j]))
            windows.append(window)
        summaries[mode] = dict(frames=len(frames), visual_available=sum(r['visual_available'] for r in frames),
            contact_unavailable_at_capture=sum(r['contact_unavailable_at_capture'] for r in frames),
            visual_unavailable_at_contact_dropout=sum(r['contact_unavailable_at_capture'] and not r['visual_available'] for r in frames),
            position_error_m=stats([r['position_error_m'] for r in frames if r['visual_available']]),
            orientation_error_rad=stats([r['orientation_error_rad'] for r in frames if r['visual_available']]),
            windows=summarize_windows(windows),
            by_phase={str(p): summarize_windows([r for r in windows if r['phases']==[p]])
                      for p in sorted(set(int(v) for v in raw['phase'][750:]))},
            mixed_phase_windows=sum(len(r['phases'])!=1 for r in windows),
            observer_wall_ms=stats([r['members'][mode]['observer_wall_ms'] for r in record['rows']
                                   if 'observer_wall_ms' in r['members'][mode]]),
            keyframes=len(record['keyframes'][mode]), failure=record['failures'][mode])
        details[mode] = dict(frames=frames, windows=windows)
    span_coverage = []
    for span in spans:
        covering = [j for j in range(1, len(record['rows']))
                    if record['rows'][j-1]['measured_ns'] <= span['last_missing_ns']
                    and record['rows'][j]['measured_ns'] >= span['first_missing_ns']]
        span_coverage.append(span | dict(camera_interval_end_frames=covering,
            models={m: dict(intervals=len(covering), visual_unavailable=sum(
                details[m]['windows'][j-1]['displacement_error_m'] is None for j in covering),
                displacement_error_m=stats([details[m]['windows'][j-1]['displacement_error_m']
                    for j in covering if details[m]['windows'][j-1]['displacement_error_m'] is not None])) for m in MODES}))
    return dict(summaries=summaries, details=details, dropout_spans=span_coverage,
                support_observations=len(support), unavailable_support_observations=sum(s['missing_samples'] for s in spans),
                native_evaluator_only=True, physical_error_bound=None, navigation_qualified=False)


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive frozen RGB-D friction replay')
    cv2.setNumThreads(1); launch = preflight(); OUTPUT.mkdir(); write_json(OUTPUT/'launch.json', launch)
    try:
        predictions = {c: replay_condition(INPUT/c) for c in CONDITIONS}
        write_json(OUTPUT/'predictions.json', dict(conditions=predictions, native_pose_loaded=False,
            support_predictions_loaded=False, parameter_fitting=False))
        evaluated = {c: evaluate_condition(INPUT/c, predictions[c]) for c in CONDITIONS}
        write_json(OUTPUT/'evaluation.json', evaluated); verify(launch)
        result = dict(status='FROZEN_RGBD_FRICTION_DROPOUT_DIAGNOSTIC_COMPLETE',
            summaries={c: evaluated[c]['summaries'] for c in CONDITIONS},
            artifact_sha256={n: digest(OUTPUT/n) for n in ('predictions.json', 'evaluation.json')},
            parameter_fitting=False, physics_executed=False, independent_nominal_prefix=False,
            error_bounds_calibrated=False, navigation_qualified=False, goal_achieved=False)
        write_json(OUTPUT/'result.json', result); print(json.dumps(result), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_FRICTION_RGBD_REPLAY_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
