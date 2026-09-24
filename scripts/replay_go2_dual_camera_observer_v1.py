"""Continuous recorded-sensor dual-camera tracking; native truth is postfit only."""
import argparse
from copy import deepcopy
import json
import time
import cv2
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.dual_camera_anchor_pose_development import DualCameraAnchorPose
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.visual_led_motion_development import POSE_FIELDS
from lewm.physical_execution_development import rotation_xyzw
from lewm.joint_rgbd_rigid_pose_development import angle
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.audit_go2_auxiliary_rgb_packet_v1 import OUTPUT as PACKET, verify_all as verify_packet
from scripts.navigation_artifact_root_development import (
    BASE, create_output, validate_root, artifact_path, verify_artifacts)
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware, source_check
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.maze_decision_stream_development import read_rows, writer

INPUT = BASE/'go2_later_floor_resolution_maze_pilot_v1_attempt_001'
CASE = 'full_jepa_novel_maze_00'
OUTPUT = BASE/'go2_dual_camera_observer_replay_v1_attempt_001'
PROTOCOL = 'docs/go2_dual_camera_observer_replay_v1_2026-09-09.md'
NATIVE_SHA = '3745c0b7c45265a2fcc38caf732b6f3f4b228487f344d23502dc7395077d5755'
PACKET_SHA = '122da7713c522ca06e8c6881597d42cc2b6f49a35bae691703aaf94f20a2b5c0'


def verify_all(launch):
    source_check(launch['source_sha256'])
    verify_artifacts(INPUT, launch['native_input_sha256'])
    verify_artifacts(PACKET, launch['packet_artifact_sha256'])
    verify_packet(read_json(PACKET, 'launch.json'))


def normalize(value):
    return json.loads(json.dumps(value, allow_nan=False))


def compare_primary(pose, model, original):
    expected = original['current_pose']
    if expected is None or normalize({k: pose[k] for k in POSE_FIELDS}) != {k: expected[k] for k in POSE_FIELDS}:
        raise ValueError('unchanged primary pose prefix mismatch')
    for actual, key in [(model.last_continuity, 'continuity_evidence'),
                        (model.last_selection, 'reference_selection'),
                        (model.last_overlap_retention, 'overlap_retention')]:
        if normalize(actual) != original[key]:
            raise ValueError('unchanged primary prefix mismatch: '+key)


def postfit_evaluation(poses):
    with np.load(artifact_path(INPUT, CASE+'/physics_trace.npz'), allow_pickle=False) as a:
        native = a['base_pose_world']
    origin = native[749]; R0 = rotation_xyzw(origin[3:]); errors = []
    for row in poses:
        pose = row['pose']
        if pose is None: continue
        truth = native[749+50*row['frame']]
        position = R0.T @ (truth[:3]-origin[:3])
        rotation = R0.T @ rotation_xyzw(truth[3:])
        delta = np.asarray(pose['position_initial_body_m'])-position
        errors.append(dict(frame=row['frame'], translation_error_m=float(np.linalg.norm(delta)),
            xy_error_m=float(np.linalg.norm(delta[:2])),
            rotation_error_rad=angle(rotation.T @ np.asarray(pose['rotation_initial_body_from_current_body']))))
    return errors


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive observer replay required')
    cv2.setNumThreads(1)
    verify_artifacts(INPUT, {'result.json': NATIVE_SHA})
    native = read_json(INPUT, 'result.json')
    assert native['status'] == 'LATER_FLOOR_RESOLUTION_MAZE_PILOT_COMPLETE'
    verify_artifacts(PACKET, {'result.json': PACKET_SHA})
    packet_result = read_json(PACKET, 'result.json')
    assert packet_result['status'] == 'AUXILIARY_RGB_PACKET_AUDIT_COMPLETE'
    sources = discover_sources((PROTOCOL, 'scripts/replay_go2_dual_camera_observer_v1.py',
        'lewm/tests/test_dual_camera_anchor_pose_development.py'), packet_result['source_sha256'])
    resources = hardware()
    launch = dict(protocol=PROTOCOL, output_root=str(OUTPUT), source_sha256=sources,
        native_input_sha256={'result.json': NATIVE_SHA, **native['artifact_sha256']},
        packet_artifact_sha256={'result.json': PACKET_SHA, **packet_result['artifact_sha256']},
        hardware=resources, frames=1881, cpu_processes=1, numerical_threads=1,
        native_scene_workers=0, minimum_available_ram_bytes=4*1024**3,
        output_allowance_bytes=1024**3, os_resource_limits_enforced=False,
        native_execution=False, model_training=False, controller_installed=False,
        native_pose_used_only_after_observer_finishes=True,
        concurrency_reason='one CPU observer replay after paired timing, beside existing tenth episode audit')
    verify_all(launch)
    memory_ok = resources['memory_available_bytes'] >= 4*1024**3
    storage_ok = resources['artifact_free_bytes'] >= 41*1024**3
    if args.preflight_only:
        print('DUAL_CAMERA_OBSERVER_PREFLIGHT', json.dumps(dict(source_count=len(sources), hardware=resources,
            memory_admission_pass=memory_ok, storage_admission_pass=storage_ok, output_created=False)), flush=True)
        return
    if not memory_ok or not storage_ok: raise ValueError('observer replay resource admission failed')
    create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch)
    print('DUAL_CAMERA_OBSERVER_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    start = time.perf_counter()
    try:
        reader = IntentReturnRGBDReplay(INPUT/CASE)
        assert len(reader.frames) == 1881
        acquisitions = read_json(INPUT/CASE, 'auxiliary_camera_audit.json')
        model = DualCameraAnchorPose(); failure = None; first_aux = None
        first_failure = None; exact_prefix = 0; poses = []; times = []; selected = []
        with writer(OUTPUT) as append:
            for i, saved in enumerate(read_rows(INPUT/CASE)):
                if i >= 1881 or saved['tick'] != i: raise ValueError('complete ordered recorded population required')
                p, d, f, now = reader.packet(i)
                rgb, auxiliary = packet(INPUT/CASE, i, p, public_acquisition(acquisitions[i]), now_ns=now)
                pose = None
                if failure is None:
                    began = time.perf_counter_ns()
                    try:
                        pose = model.observe(p, d, f, auxiliary_rgb=rgb, auxiliary_depth=auxiliary, now_ns=now)
                    except SensorContractError as error:
                        chain = []; cause = error
                        while cause is not None:
                            chain.append(str(cause)); cause = cause.__cause__
                        failure = dict(frame=i, chain=chain); first_failure = i
                    times.append((time.perf_counter_ns()-began)/1e6)
                camera = model.last_camera_selection or {}
                if camera.get('auxiliary_attempted') and first_aux is None: first_aux = i
                if first_aux is None and pose is not None:
                    try:
                        compare_primary(pose, model, saved['decision']['original_visual_evidence'])
                    except ValueError:
                        write_json(OUTPUT/'prefix_mismatch.json', dict(frame=i, pose=pose,
                            continuity=model.last_continuity, reference_selection=model.last_selection,
                            overlap_retention=model.last_overlap_retention))
                        raise
                    exact_prefix += 1
                if pose is not None: selected.append(camera.get('selected_camera'))
                append(dict(tick=i, decision=dict(pose=pose, failure=failure,
                    reference_selection=deepcopy(model.last_selection), continuity=deepcopy(model.last_continuity),
                    camera_selection=deepcopy(model.last_camera_selection),
                    overlap_retention=deepcopy(model.last_overlap_retention),
                    observer_terminal_latched=failure is not None)))
                poses.append(dict(frame=i, pose=None if pose is None else {k: pose[k] for k in
                    ('position_initial_body_m', 'rotation_initial_body_from_current_body')}))
                if (OUTPUT/'context_decisions.jsonl.gz').stat().st_size > 1024**3:
                    raise ValueError('observer output allowance exceeded')
                if i%100 == 0: print('DUAL_CAMERA_OBSERVER_FRAME', i, first_aux, first_failure, flush=True)
        assert len(poses) == 1881
        errors = postfit_evaluation(poses)
        write_json(OUTPUT/'postfit_errors.json', errors)
        verify_all(launch)
        artifacts = {n: digest(OUTPUT/n) for n in ('launch.json', 'context_decisions.jsonl.gz', 'postfit_errors.json')}
        write_json(OUTPUT/'result.json', dict(status='DUAL_CAMERA_OBSERVER_REPLAY_COMPLETE',
            source_sha256=sources, artifact_sha256=artifacts, frames=len(poses),
            accepted_poses=len(errors), first_auxiliary_attempt=first_aux, first_failure_frame=first_failure,
            terminal_failure=failure, unchanged_primary_prefix_frames=exact_prefix,
            selected_primary_frames=selected.count('primary'), selected_auxiliary_frames=selected.count('auxiliary'),
            total_measured_bridge_frames=model.total_bridge_frames, retained_references=len(model.references),
            keyframes=len(model.nodes), all_frames_tracked=first_failure is None,
            observed_error_summary={k:dict(maximum=max(r[k] for r in errors), mean=float(np.mean([r[k] for r in errors])))
                for k in ('translation_error_m', 'xy_error_m', 'rotation_error_rad')} if errors else {},
            observer_timing=dict(count=len(times), median_ms=float(np.median(times)), maximum_ms=max(times),
                over100ms=sum(t > 100 for t in times)), wall_s=time.perf_counter()-start,
            hardware_after=hardware(), native_pose_used_only_after_observer_finishes=True,
            controller_installed=False, commands_generated=False, native_execution=False,
            original_navigation_outcome_unchanged=True, independent_layouts=0,
            uncertainty_calibrated=False, navigation_qualified=False, goal_achieved=False))
        print('DUAL_CAMERA_OBSERVER_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(reason=repr(error)))
        raise


if __name__ == '__main__': main()
