"""Full-trace corner feature observer comparison; no new commands or physics."""
import json
import time
import cv2
import numpy as np
import torch

from lewm.corner_support_joint_observer_development import CornerSupportVisualLedMotion
from lewm.joint_temporal_anchor_continuity_development import JointTemporalAnchorVisualLedMotion
from lewm.joint_sensor_anchored_goal_development import current_joint_pose
from lewm.joint_rgbd_rigid_pose_development import angle
from lewm.physical_execution_development import rotation_xyzw
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts.run_go2_learned_goal_bootstrap_probe_v1 import OUTPUT as INPUT, TRIALS
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_corner_support_observer_replay_v1_attempt_001'
PROTOCOL = 'docs/go2_corner_support_observer_replay_v1_2026-09-08.md'
IDS = {'launch.json': '962f9d461648d5aaa353c2a7a20e0a90245724cbf66cac223b60a89466422254',
    'result.json': '71601c660272b579db520ba631af661d76f869267cf1c4a26fb60b17cdfbb7f1'}
ARMS = ('original', 'corner_support')


def replay(trial, arm):
    reader = IntentReturnRGBDReplay(INPUT/trial)
    motion = (JointTemporalAnchorVisualLedMotion if arm == 'original' else CornerSupportVisualLedMotion)()
    rows = []
    for i in range(len(reader.frames)):
        p, d, f, now = reader.packet(i)
        start = time.perf_counter_ns()
        evidence = motion.observe(p, d, f, now_ns=now)
        elapsed = (time.perf_counter_ns()-start)/1e6
        if evidence['current_pose'] is not None:
            current_joint_pose(evidence, identity=(0, 0, 0), now_ns=now)
        rows.append(dict(frame=i, decision_ns=now, observer_wall_ms=elapsed, evidence=evidence))
    # The observer has finished before evaluator-only native state is opened.
    cameras = read_json(INPUT/trial, 'camera_audit.json')
    with np.load(INPUT/trial/'physics_trace.npz', allow_pickle=False) as archive:
        poses = archive['base_pose_world']
    origin = poses[cameras[0]['physical_sample_index']]
    R0 = rotation_xyzw(origin[3:])
    xy, rotations = [], []
    for row, camera in zip(rows, cameras, strict=True):
        e = row['evidence']; p = e['current_pose']
        error_xy = error_rotation = None
        if p is not None:
            truth = poses[camera['physical_sample_index']]
            actual = R0.T@(truth[:3]-origin[:3])
            actual_R = R0.T@rotation_xyzw(truth[3:])
            error_xy = float(np.linalg.norm(np.asarray(p['position_initial_body_m'])[:2]-actual[:2]))
            error_rotation = angle(actual_R.T@np.asarray(p['rotation_initial_body_from_current_body']))
            xy.append(error_xy); rotations.append(error_rotation)
        row.update(native_xy_error_m=error_xy, native_rotation_error_rad=error_rotation, errors_evaluator_only=True)
    failures = [r for r in rows if r['evidence']['terminal_failure'] is not None]
    accepted = len(xy)
    active_times = [r['observer_wall_ms'] for r in rows if not failures or r['frame'] <= failures[0]['frame']]
    eligible = bool(accepted == len(rows) and not failures and max(xy) <= .02 and max(rotations) <= .05)
    report = dict(trial=trial, arm=arm, frames=len(rows), current_joint_poses=accepted,
        first_failure_frame=failures[0]['frame'] if failures else None,
        terminal_failure=failures[0]['evidence']['terminal_failure'] if failures else None,
        maximum_xy_error_m=max(xy) if xy else None, maximum_rotation_error_rad=max(rotations) if rotations else None,
        active_observer_timing=dict(count=len(active_times), median_ms=float(np.median(active_times)),
            maximum_ms=float(max(active_times)), over100ms=sum(t > 100 for t in active_times)),
        diagnostic_thresholds_pass=eligible, native_state_used_for_observer=False,
        commands_generated=False, navigation_qualified=False)
    name = trial+'_'+arm+'_frames.json'
    write_json(OUTPUT/name, rows)
    report['frames_file'] = name
    report['frames_sha256'] = digest(OUTPUT/name)
    print('CORNER_SUPPORT_REPLAY', json.dumps(report, sort_keys=True), flush=True)
    return report


def main():
    if not __debug__:
        raise ValueError('enabled audit assertions required')
    cv2.setNumThreads(1); torch.set_num_threads(1)
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive replay attempt; no retry/resume')
    verify_artifacts(INPUT, IDS)
    original, result = read_json(INPUT, 'launch.json'), read_json(INPUT, 'result.json')
    assert result['status'] == 'LEARNED_GOAL_BOOTSTRAP_PROBE_COMPLETE' and result['trials'] == list(TRIALS)
    bindings = IDS | result['artifact_sha256']
    verify_artifacts(INPUT, bindings)
    sources = discover_sources((PROTOCOL, 'scripts/replay_go2_corner_support_observer_v1.py',
        'lewm/tests/test_corner_support_observer_development.py'), original['source_sha256'])
    resources = hardware()
    if resources['memory_available_bytes'] < 4*1024**3 or resources['artifact_free_bytes'] < 40*1024**3+256*1024**2:
        raise ValueError('replay RAM/storage allowance unavailable')
    launch = original | dict(source_sha256=sources, output_root=str(OUTPUT), protocol=PROTOCOL,
        probe_artifact_sha256=bindings, native_execution=False, hardware=resources,
        observer_arms=list(ARMS), native_scene_workers=0, threads=1,
        concurrency_reason='serial observer arms for comparable CPU timing',
        eligible_native_probe_thresholds=dict(all_frames_current=True, maximum_xy_error_m=.02, maximum_rotation_error_rad=.05))
    verify(launch); create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch)
    print('CORNER_SUPPORT_REPLAY_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    reports = []
    try:
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            for trial in TRIALS:
                for arm in ARMS:
                    monitor.write(json.dumps(dict(trial=trial, arm=arm, stage='before', **hardware()))+'\n'); monitor.flush()
                    reports.append(replay(trial, arm))
                    monitor.write(json.dumps(dict(trial=trial, arm=arm, stage='after', **hardware()))+'\n'); monitor.flush()
        assert reports[0]['first_failure_frame'] == 5
        assert [r['frames'] for r in reports] == [16, 16, 254, 254]
        verify(launch); verify_artifacts(INPUT, bindings)
        products = {r['frames_file']: r['frames_sha256'] for r in reports}
        products.update({n: digest(OUTPUT/n) for n in ('launch.json', 'resource_monitor.jsonl')})
        verify_artifacts(OUTPUT, products)
        write_json(OUTPUT/'result.json', dict(status='CORNER_SUPPORT_OBSERVER_REPLAY_COMPLETE',
            reports=reports, source_sha256=sources, artifact_sha256=products,
            candidate_eligible_for_separate_native_probe=all(r['diagnostic_thresholds_pass'] for r in reports if r['arm']=='corner_support'),
            probe_sha256=IDS, original_mission_outcomes_unchanged=True,
            native_execution=False, candidate_adopted=False, navigation_qualified=False, goal_achieved=False))
        print('CORNER_SUPPORT_REPLAY_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_CORNER_SUPPORT_REPLAY_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__':
    main()

