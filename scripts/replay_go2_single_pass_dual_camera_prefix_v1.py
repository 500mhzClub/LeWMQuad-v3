"""Exact optimized-map replay through the completed first auxiliary intervention."""
import argparse
import json
import time
import cv2
import torch
from lewm.single_pass_dual_camera_controller_development import SinglePassDualCameraController
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS, RESERVE_BYTES
from lewm.pulse_timed_training_runner_development import state_digest
from lewm.dual_camera_native_admission_development import admit_prefix, admit_predecessor
from scripts.dual_camera_intervention_witness_development import admit_intervention
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.maze_decision_stream_development import read_rows, writer, NAME as STREAM
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware, source_check
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from scripts.replay_go2_dual_camera_controller_prefix_v2 import (
    OUTPUT as PREFIX, INPUT, CASE, verify_all as verify_prefix)

PERFORMANCE = BASE/'go2_single_pass_maze_controller_replay_v1_attempt_001'
OUTPUT = BASE/'go2_single_pass_dual_camera_prefix_v1_attempt_001'
PROTOCOL = 'docs/go2_single_pass_dual_camera_prefix_v1_2026-09-09.md'
NATIVE_RESULT = 'a7a02db120b4b662cd66efee01f10784edb6f2ed6984bdc420007773a1b1b6fb'
PERFORMANCE_RESULT = '4687d67fbb53fce3b29a122e379b51805fa685b4c39e3e817e3a5862374d342e'
MAX_OUTPUT_BYTES = 1024**3


def verify_all(launch):
    source_check(launch['source_sha256'])
    verify_artifacts(PREFIX, launch['prefix_artifact_sha256'])
    verify_artifacts(INPUT, launch['native_artifact_sha256'])
    verify_artifacts(PERFORMANCE, launch['performance_artifact_sha256'])
    verify_prefix(read_json(PREFIX, 'launch.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix-result-sha256', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive dual-camera optimization replay required')
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    verify_artifacts(PREFIX, {'result.json': args.prefix_result_sha256})
    prefix = read_json(PREFIX, 'result.json'); admit_prefix(prefix)
    if prefix['frames'] != 1873 or prefix['first_auxiliary_intervention_frame'] != 1872:
        raise ValueError('the completed bounded1873-frame controller prefix is required')
    prefix_ids = {'result.json': args.prefix_result_sha256, **prefix['artifact_sha256']}
    verify_artifacts(PREFIX, prefix_ids)
    witness = admit_intervention(PREFIX, prefix)
    old = read_json(PREFIX, 'launch.json')
    verify_artifacts(INPUT, {'result.json': NATIVE_RESULT})
    native = read_json(INPUT, 'result.json')
    native_ids = {'result.json': NATIVE_RESULT, **native['artifact_sha256']}
    verify_artifacts(INPUT, native_ids)
    predecessor = admit_predecessor(native, read_json(INPUT, CASE[0]+'_audit.json'), old, case=CASE[0])
    verify_artifacts(PERFORMANCE, {'result.json': PERFORMANCE_RESULT})
    performance = read_json(PERFORMANCE, 'result.json')
    if (performance['status'] != 'SINGLE_PASS_MAZE_CONTROLLER_REPLAY_COMPLETE'
            or performance['all_complete_decisions_exact'] is not True
            or performance['model_state_unchanged'] is not True or performance['native_adoption'] is not False):
        raise ValueError('completed separate single-pass equality evidence required')
    performance_ids = {'result.json': PERFORMANCE_RESULT, **performance['artifact_sha256']}
    inherited = dict(prefix['source_sha256'])
    for name, h in performance['source_sha256'].items():
        if name in inherited and inherited[name] != h: raise ValueError('incompatible frozen source: '+name)
        inherited[name] = h
    sources = discover_sources((PROTOCOL, 'scripts/replay_go2_single_pass_dual_camera_prefix_v1.py',
        'lewm/tests/test_single_pass_dual_camera_controller_development.py'), inherited)
    resources = hardware()
    launch = dict(protocol=PROTOCOL, output_root=str(OUTPUT), source_sha256=sources,
        prefix_artifact_sha256=prefix_ids, native_artifact_sha256=native_ids,
        performance_artifact_sha256=performance_ids, correction_admission=old['correction_admission'],
        prefix_admission=witness, predecessor_admission=predecessor, hardware=resources,
        frames=1873, first_auxiliary_intervention_frame=1872,
        implementation_class='SinglePassDualCameraController', original_decision_labels_preserved=True,
        cpu_processes=1, numerical_threads=1, opencv_threads=1, native_scene_workers=0,
        memory_admission_bytes=8*1024**3, concurrent_native_memory_headroom_bytes=32*1024**3,
        output_allowance_bytes=MAX_OUTPUT_BYTES, concurrent_native_output_headroom_bytes=11*1024**3,
        minimum_free_bytes=RESERVE_BYTES, os_resource_limits_enforced=False,
        concurrency_reason='one CPU equality replay beside the one frozen dual-camera native scene',
        controlled_speed_comparison=False, native_execution=False, native_adoption=False,
        model_training=False, navigation_qualified=False, real_time_qualified=False, goal_achieved=False)
    verify_all(launch)
    memory_ok = resources['memory_available_bytes'] >= 40*1024**3
    storage_ok = resources['artifact_free_bytes'] >= RESERVE_BYTES+12*1024**3
    if args.preflight_only:
        print('SINGLE_PASS_DUAL_CAMERA_PREFLIGHT', json.dumps(dict(source_count=len(sources), hardware=resources,
            memory_admission_pass=memory_ok, storage_admission_pass=storage_ok,
            completed_inputs_and_sources_verified=True, output_created=False)), flush=True)
        return
    if not memory_ok or not storage_ok: raise ValueError('shared native/replay resource admission failed')
    create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch)
    print('SINGLE_PASS_DUAL_CAMERA_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    start = time.perf_counter()
    try:
        name, index, variant, condition, model_name = CASE
        model, c, v = load_assigned(launch['correction_admission'], model_name)
        assert (c, v) == (condition, variant)
        before = state_digest(model.state_dict()); assert before == prefix['model_state_sha256']
        controller = SinglePassDualCameraController(model, ArticulatedCollisionGeometry(URDF),
            public_mission=public_mission(index), navigation_ticks=NAVIGATION_TICKS,
            persistent=True, condition=c, variant=v)
        directory = INPUT/name; reader = IntentReturnRGBDReplay(directory)
        acquisitions = read_json(directory, 'auxiliary_camera_audit.json')
        tape = read_json(directory, 'command_tape.json'); count = 0; auxiliary_frames = []
        with writer(OUTPUT) as append:
            for i, saved in enumerate(read_rows(PREFIX)):
                if i >= 1873 or saved['tick'] != i: raise ValueError('bounded ordered prospective prefix required')
                p, d, f, now = reader.packet(i)
                image, auxiliary = packet(directory, i, p, public_acquisition(acquisitions[i]), now_ns=now)
                input_before = fingerprint((p, d, f, auxiliary, image, now))
                began = time.perf_counter_ns()
                candidate = controller.observe(p, d, f, now_ns=now, auxiliary_depth=auxiliary, auxiliary_rgb=image)
                elapsed = (time.perf_counter_ns()-began)/1e6
                normalized = json.loads(json.dumps(candidate, allow_nan=False))
                if normalized != saved['decision']:
                    write_json(OUTPUT/'mismatch.json', dict(frame=i, decision=normalized))
                    raise ValueError('complete dual-camera decision differs at frame '+str(i))
                if fingerprint((p, d, f, auxiliary, image, now)) != input_before:
                    raise ValueError('input arrays mutated at frame '+str(i))
                if i < 1872 and normalized['requested_command'] != tape[i]['requested_command']:
                    raise ValueError('preintervention command differs from actual executed tape')
                if normalized['terminal'] is not None: raise ValueError('active completed prefix required')
                raw = normalized['original_visual_evidence'] or {}
                if (raw.get('camera_selection') or {}).get('auxiliary_attempted'): auxiliary_frames.append(i)
                append(dict(tick=i, decision=normalized, complete_decision_exact=True,
                    input_arrays_unchanged=True, controller_wall_ms=elapsed)); count += 1
                if (OUTPUT/STREAM).stat().st_size > MAX_OUTPUT_BYTES: raise ValueError('compressed output allowance exceeded')
                if i%100 == 0: print('SINGLE_PASS_DUAL_CAMERA_FRAME', i, flush=True)
        assert count == 1873 and auxiliary_frames == [1872]
        assert normalized['requested_command'] == prefix['final_requested_command']
        assert state_digest(model.state_dict()) == before and all(p.grad is None for p in model.parameters())
        verify_all(launch)
        bindings = {n:digest(OUTPUT/n) for n in ('launch.json', STREAM)}
        verify_artifacts(OUTPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='SINGLE_PASS_DUAL_CAMERA_PREFIX_COMPLETE',
            source_sha256=sources, artifact_sha256=bindings, frames=count, auxiliary_intervention_frames=auxiliary_frames,
            all_complete_decisions_exact=True, all_preintervention_executed_commands_exact=True,
            final_requested_command=normalized['requested_command'], final_registered_pose_available=True,
            following_recorded_observations_consumed=False, input_arrays_unchanged=True,
            model_state_unchanged=True, model_state_sha256=before,
            controlled_speed_comparison=False, native_execution=False, native_adoption=False,
            navigation_qualified=False, real_time_qualified=False, goal_achieved=False,
            wall_s=time.perf_counter()-start, hardware_after=hardware()))
        print('SINGLE_PASS_DUAL_CAMERA_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(reason=repr(error))); raise


if __name__ == '__main__': main()
