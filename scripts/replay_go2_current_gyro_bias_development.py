"""Fixed six-case gyro sensitivity replay of two complete recent journeys.

No controller, sensor admission rule, recording, or physics is changed. Native
pose is opened only after each case's estimated trajectory is saved.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from lewm import process_mapped_runtime_development as process
from lewm.finite_rgbd_error_members_development import ErrorMember, perturb_packets
from lewm.physical_execution_development import rotation_xyzw
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.run_go2_sparse_corner_completion_development import initialize_pose

BASE = Path('/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1')
PLAN = Path('docs/go2_current_gyro_bias_plan_2026-09-17.json')
RESULT = Path('docs/go2_current_gyro_bias_result_2026-09-17.json')
ROOTS = (
    'go2_return_routing_memory_01_persistent_return_noise_2mm_native_layout00_4800_v1_attempt_001',
    'go2_return_routing_memory_04_persistent_return_noise_2mm_native_layout01_4800_v1_attempt_001',
)
MEMBERS = (ErrorMember('nominal'),
    ErrorMember('yaw_positive', gyro_z_bias_rad_s=.001),
    ErrorMember('yaw_negative', gyro_z_bias_rad_s=-.001))
ASSIGNMENTS = [(root, member) for member in MEMBERS for root in ROOTS]
SOURCES = (__file__, 'lewm/finite_rgbd_error_members_development.py',
    'lewm/sparse_corner_completion_development.py',
    'lewm/sparse_corner_completion_runtime_development.py',
    'lewm/cadenced_view_revisit_tracking_development.py',
    'lewm/fast_gyro_development.py',
    'scripts/live_depth_noise_session_development.py',
    'scripts/in_memory_public_replay_development.py',
    'scripts/run_go2_sparse_corner_completion_development.py')


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output(root, member):
    return BASE / root / ('current_gyro_bias_' + member.name + '_v1')


def prepare():
    sources = {str(Path(p).resolve().relative_to(Path.cwd())): digest(p) for p in SOURCES}
    inputs = {}
    for root in ROOTS:
        reader = NoisyPublicReplay(BASE / root / 'native')
        del reader
        for name in ('native/in_memory_camera_observations.json', 'poses.json',
                     'pose_worker_identity.json', 'return_routing_memory_navigation_readout_v1.json'):
            inputs[str(BASE / root / name)] = digest(BASE / root / name)
    assert not any(output(root, member).exists() for root, member in ASSIGNMENTS)
    save(PLAN, dict(schema='current_gyro_bias_plan.v1',
        assignments=[dict(root=root, member=asdict(member)) for root, member in ASSIGNMENTS],
        source_sha256=sources, input_sha256=inputs,
        full_recorded_journeys=True, concurrency=2,
        cpu_groups=[[*range(0, 8), *range(16, 24)], [*range(8, 16), *range(24, 32)]],
        original_depth_noise_mm=2, gyro_bias_magnitude_rad_s=.001,
        gyro_bias_is_hardware_calibrated=False,
        bias_applied_to_shared_fast_and_slow_gyro=True,
        native_state_evaluator_only=True, new_navigation=False,
        rationale='Current-tracker sensitivity on two full recent successful journeys; same signed bias sizes as the older estimator study.'))
    print('PREPARED', len(ASSIGNMENTS), flush=True)


def score(root, rows):
    metadata = json.loads((root / 'native/in_memory_camera_observations.json').read_text())['frames']
    with np.load(root / 'native/physics_trace.npz', allow_pickle=False) as archive:
        physics = archive['base_pose_world']
    original = {row['frame']: row['raw_pose'] for row in json.loads((root / 'poses.json').read_text())}
    origin = physics[metadata[0]['physical_sample_index']]
    R0 = rotation_xyzw(origin[3:])
    scores = []
    exact = 0
    for row in rows:
        frame, pose = row['frame'], row['pose']
        actual = physics[metadata[frame]['physical_sample_index']]
        position = R0.T @ (actual[:3] - origin[:3])
        rotation = R0.T @ rotation_xyzw(actual[3:])
        delta = np.asarray(pose['position_initial_body_m']) - position
        R = np.asarray(pose['rotation_initial_body_from_current_body'])
        disagreement = R.T @ rotation
        # atan2 retains precision near zero while covering the full rotation range.
        sine = np.linalg.norm([disagreement[2, 1]-disagreement[1, 2],
            disagreement[0, 2]-disagreement[2, 0], disagreement[1, 0]-disagreement[0, 1]]) / 2
        angle = np.arctan2(sine, (np.trace(disagreement)-1)/2)
        exact += int(frame in original and all(np.array_equal(pose[k], original[frame][k])
            for k in ('position_initial_body_m', 'rotation_initial_body_from_current_body')))
        scores.append(dict(frame=frame, elapsed_sensor_s=(row['measured_ns']-rows[0]['measured_ns'])/1e9,
            position_error_mm=float(np.linalg.norm(delta)*1000),
            xy_error_mm=float(np.linalg.norm(delta[:2])*1000),
            orientation_error_deg=float(np.degrees(angle))))
    return dict(rows=scores, exactly_matching_original_poses=exact,
        maximum_position_error_mm=max(r['position_error_mm'] for r in scores),
        maximum_xy_error_mm=max(r['xy_error_mm'] for r in scores),
        maximum_orientation_error_deg=max(r['orientation_error_deg'] for r in scores),
        native_state_loaded_only_after_predictions_saved=True)


def run(number):
    plan = json.loads(PLAN.read_text())
    assert all(digest(path) == expected for path, expected in plan['source_sha256'].items())
    root_name, member = ASSIGNMENTS[number-1]
    root = BASE / root_name
    for path, expected in plan['input_sha256'].items():
        if Path(path).is_relative_to(root):
            assert digest(path) == expected
    assert sorted(os.sched_getaffinity(0)) == plan['cpu_groups'][(number-1) % 2]
    reader = NoisyPublicReplay(root / 'native')
    count = len(reader.noise_rows)
    destination = output(root_name, member)
    destination.mkdir(exist_ok=False)
    save(destination / 'assignment.json', dict(number=number, root=str(root),
        member=asdict(member), plan_sha256=digest(PLAN), cpu_affinity=sorted(os.sched_getaffinity(0))))
    initialize_pose(str(destination))
    motion = process._motion
    assert type(motion.model).__name__ == 'SparseCornerCompletionPose'
    rows = []
    failure = None
    started = time.monotonic()
    for frame in range(count):
        policy, depth, fast, rgb, auxiliary, now = reader.packet(frame)
        policy, depth, fast = perturb_packets(member, policy, depth, fast, anchor_ns=1_500_000_000)
        raw = motion.observe(policy, depth, fast, auxiliary_rgb=rgb,
            auxiliary_depth=auxiliary, now_ns=now)
        pose = raw.get('current_pose')
        if pose is None:
            failure = dict(frame=frame, measured_ns=now, terminal_failure=raw.get('terminal_failure'))
            save(destination / 'terminal_raw_snapshot.json', raw)
            break
        rows.append(dict(frame=frame, measured_ns=now, pose=pose))
        if frame % 200 == 0:
            print('GYRO_BIAS_REPLAY', number, member.name, frame,
                round(time.monotonic()-started, 1), flush=True)
    save(destination / 'predictions.json', rows)
    evaluated = score(root, rows)
    save(destination / 'accuracy.json', evaluated)
    result = dict(assignment=number, root=root_name, condition=member.name,
        bias_rad_s=member.gyro_z_bias_rad_s, recorded_frames=count, accepted_frames=len(rows),
        complete_recording_accepted=failure is None and len(rows) == count, failure=failure,
        accepted_sensor_seconds=(rows[-1]['measured_ns']-rows[0]['measured_ns'])/1e9,
        wall_seconds=time.monotonic()-started,
        accuracy={k: v for k, v in evaluated.items() if k != 'rows'},
        depth_digests_verified=True, raw_tracker_only=True,
        downstream_floor_registration_or_navigation_reexecuted=False,
        synthetic_bias_not_hardware_calibration=True,
        source_sha256=plan['source_sha256'], prediction_sha256=digest(destination/'predictions.json'))
    save(destination / 'result.json', result)
    if member.name == 'nominal':
        assert result['complete_recording_accepted']
        assert evaluated['exactly_matching_original_poses'] == count
    print('GYRO_BIAS_RESULT', json.dumps(result), flush=True)


def summarize():
    results = [json.loads((output(root, member)/'result.json').read_text())
        for root, member in ASSIGNMENTS]
    matched = []
    for root in ROOTS:
        per_case = {member.name: json.loads((output(root, member)/'accuracy.json').read_text())['rows']
            for member in MEMBERS}
        common = min(len(rows) for rows in per_case.values())
        matched.append(dict(root=root, common_frames=common, errors={name: dict(
            maximum_xy_error_mm=max(row['xy_error_mm'] for row in rows[:common]),
            maximum_orientation_error_deg=max(row['orientation_error_deg'] for row in rows[:common]))
            for name, rows in per_case.items()}))
    save(RESULT, dict(status='complete', assignments=results, common_prefix_comparisons=matched,
        native_navigation_reexecuted=False, hardware_calibrated=False,
        outcome='Full assigned population retained; accepted-prefix accuracy and tracking survival are distinct outcomes.'))
    print('GYRO_BIAS_COMPLETE', json.dumps(matched), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--assignment', type=int, choices=range(1, 7))
    modes.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.summarize:
        summarize()
    else:
        run(args.assignment)
