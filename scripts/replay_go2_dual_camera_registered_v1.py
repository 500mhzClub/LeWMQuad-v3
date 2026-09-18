"""Full actual-capture dual-camera witness admission and floor registration."""
import argparse
import json
import time
import cv2
import numpy as np
from lewm.dual_camera_visual_motion_development import DualCameraVisualMotion, current_dual_camera_pose
from lewm.joint_floor_registered_evidence_development import JointFloorRegistration, current_joint_floor_registered_pose
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.visual_led_motion_development import POSE_FIELDS
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.replay_go2_dual_camera_observer_v1 import (
    OUTPUT as OBSERVER, INPUT, CASE, verify_all as verify_observer, normalize, postfit_evaluation)
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware, source_check
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.maze_decision_stream_development import read_rows, writer

OUTPUT = BASE/'go2_dual_camera_registered_replay_v1_attempt_001'
PROTOCOL = 'docs/go2_dual_camera_registered_replay_v1_2026-09-09.md'
OBSERVER_SHA = '540f63243f74b63e90dadeaa8e7aebde936bb7880d49f0c77ffc5e8affc59eea'


def verify_all(launch):
    source_check(launch['source_sha256'])
    verify_artifacts(OBSERVER, launch['observer_artifact_sha256'])
    verify_observer(read_json(OBSERVER, 'launch.json'))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive registered replay required')
    cv2.setNumThreads(1)
    verify_artifacts(OBSERVER, {'result.json': OBSERVER_SHA})
    old = read_json(OBSERVER, 'result.json')
    assert old['status'] == 'DUAL_CAMERA_OBSERVER_REPLAY_COMPLETE' and old['all_frames_tracked']
    assert old['frames'] == old['accepted_poses'] == 1881 and old['first_auxiliary_attempt'] == 1870
    sources = discover_sources((PROTOCOL, 'scripts/replay_go2_dual_camera_registered_v1.py',
        'lewm/tests/test_dual_camera_visual_motion_development.py'), old['source_sha256'])
    resources = hardware()
    launch = dict(protocol=PROTOCOL, output_root=str(OUTPUT), source_sha256=sources,
        observer_artifact_sha256={'result.json': OBSERVER_SHA, **old['artifact_sha256']},
        hardware=resources, frames=1881, cpu_processes=1, numerical_threads=1, native_scene_workers=0,
        minimum_available_ram_bytes=4*1024**3, output_allowance_bytes=1024**3,
        os_resource_limits_enforced=False, native_execution=False, model_training=False,
        controller_executed=False, native_pose_used_only_after_observer_finishes=True,
        concurrency_reason='one CPU registration replay beside the existing tenth episode audit')
    verify_all(launch)
    memory_ok = resources['memory_available_bytes'] >= 4*1024**3
    storage_ok = resources['artifact_free_bytes'] >= 41*1024**3
    if args.preflight_only:
        print('DUAL_CAMERA_REGISTERED_PREFLIGHT', json.dumps(dict(source_count=len(sources), hardware=resources,
            memory_admission_pass=memory_ok, storage_admission_pass=storage_ok, output_created=False)), flush=True)
        return
    if not memory_ok or not storage_ok: raise ValueError('registered replay resources unavailable')
    create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch)
    print('DUAL_CAMERA_REGISTERED_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    start = time.perf_counter()
    try:
        reader = IntentReturnRGBDReplay(INPUT/CASE)
        acquisitions = read_json(INPUT/CASE, 'auxiliary_camera_audit.json')
        motion = DualCameraVisualMotion(); registration = JointFloorRegistration()
        failure = None; first_failure = None; poses = []; raw_poses = []; times = []
        exact_raw = 0; exact_registered_prefix = 0
        with writer(OUTPUT) as append:
            for i, (saved, expected_row) in enumerate(zip(read_rows(INPUT/CASE), read_rows(OBSERVER), strict=True)):
                if i >= 1881 or saved['tick'] != i or expected_row['tick'] != i:
                    raise ValueError('complete ordered paired population required')
                p, d, f, now = reader.packet(i)
                image, auxiliary = packet(INPUT/CASE, i, p, public_acquisition(acquisitions[i]), now_ns=now)
                raw = registered = None
                if failure is None:
                    began = time.perf_counter_ns()
                    raw = motion.observe(p, d, f, auxiliary_rgb=image, auxiliary_depth=auxiliary, now_ns=now)
                    if raw['current_pose'] is None:
                        failure = dict(frame=i, stage='motion', witness=raw['terminal_failure']); first_failure = i
                    else:
                        expected = expected_row['decision']
                        keys = (*POSE_FIELDS, 'auxiliary_rgb_sha256', 'auxiliary_depth_sha256')
                        assert normalize({k:raw['current_pose'][k] for k in keys}) == {k:expected['pose'][k] for k in keys}
                        for key, other in [('continuity_evidence','continuity'), ('camera_selection','camera_selection'),
                                ('reference_selection','reference_selection'), ('overlap_retention','overlap_retention')]:
                            assert normalize(raw[key]) == expected[other], key
                        exact_raw += 1
                        try:
                            current_dual_camera_pose(raw, p, image, auxiliary, identity=(0,0,0), now_ns=now)
                            registered = registration.observe(p, d, auxiliary, raw, now_ns=now)
                            current_joint_floor_registered_pose(registered, identity=(0,0,0), now_ns=now)
                        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
                            failure = dict(frame=i, stage='registration', reason=str(error)); first_failure = i
                            registered = None
                        if registered is not None and i < old['first_auxiliary_attempt']:
                            original = saved['decision']['evidence']
                            assert normalize(registered['floor_registration']) == original['floor_registration']
                            assert normalize({k:registered['current_pose'][k] for k in original['current_pose']}) == original['current_pose']
                            exact_registered_prefix += 1
                    times.append((time.perf_counter_ns()-began)/1e6)
                append(dict(tick=i, decision=dict(raw=raw, registered=registered, failure=failure)))
                def minimal(e):
                    return None if e is None or e['current_pose'] is None else {k:e['current_pose'][k] for k in
                        ('position_initial_body_m','rotation_initial_body_from_current_body')}
                raw_poses.append(dict(frame=i, pose=minimal(raw)))
                poses.append(dict(frame=i, pose=minimal(registered)))
                if (OUTPUT/'context_decisions.jsonl.gz').stat().st_size > 1024**3:
                    raise ValueError('registered replay output allowance exceeded')
                if i%100 == 0: print('DUAL_CAMERA_REGISTERED_FRAME', i, first_failure, flush=True)
        assert len(poses) == 1881
        errors = dict(raw=postfit_evaluation(raw_poses), registered=postfit_evaluation(poses))
        write_json(OUTPUT/'postfit_errors.json', errors)
        verify_all(launch)
        artifacts = {n:digest(OUTPUT/n) for n in ('launch.json','context_decisions.jsonl.gz','postfit_errors.json')}
        write_json(OUTPUT/'result.json', dict(status='DUAL_CAMERA_REGISTERED_REPLAY_COMPLETE',
            source_sha256=sources, artifact_sha256=artifacts, frames=len(poses),
            accepted_raw_poses=len(errors['raw']), accepted_registered_poses=len(errors['registered']),
            exact_raw_observer_frames=exact_raw, exact_original_registered_prefix_frames=exact_registered_prefix,
            first_failure_frame=first_failure, terminal_failure=failure, all_frames_registered=first_failure is None,
            observed_error_summary={view:{k:dict(maximum=max(r[k] for r in rows), mean=float(np.mean([r[k] for r in rows])))
                for k in ('translation_error_m','xy_error_m','rotation_error_rad')} if rows else {} for view,rows in errors.items()},
            processing_timing=dict(count=len(times), median_ms=float(np.median(times)), maximum_ms=max(times),
                over100ms=sum(t>100 for t in times)), wall_s=time.perf_counter()-start, hardware_after=hardware(),
            native_pose_used_only_after_observer_finishes=True, controller_executed=False, commands_generated=False,
            native_execution=False, original_navigation_outcome_unchanged=True, independent_layouts=0,
            uncertainty_calibrated=False, navigation_qualified=False, goal_achieved=False))
        print('DUAL_CAMERA_REGISTERED_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(reason=repr(error)))
        raise


if __name__ == '__main__': main()
