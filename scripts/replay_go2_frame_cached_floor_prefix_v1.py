"""Exact implementation-equivalence replay through the completed intervention prefix."""
import argparse
import hashlib
import json
import time
import cv2
import torch
from lewm.frame_cached_floor_map_development import FrameCachedJointFloorRoundTripController
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS, RESERVE_BYTES
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.maze_decision_stream_development import read_rows
from scripts.novel_maze_auxiliary_packet_development import packet, public_acquisition
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.replay_go2_joint_floor_registered_maze_prefix_v1 import (
    OUTPUT as PREFIX, INPUT, READOUT, CASE, FITS, CORRECTION, FAILED, FAILED_BINDINGS)

OUTPUT = BASE/'go2_frame_cached_floor_prefix_v1_attempt_001'
PROTOCOL = 'docs/go2_frame_cached_floor_prefix_v1_2026-09-09.md'
EXPECTED = 'c1a2c347d5aea7a1e0db352a451daf4add20c9c42905886712778cbdd53166ea'
DIAGNOSIS = BASE/'go2_floor_index_reuse_diagnosis_v1_attempt_001'
DIAGNOSIS_SHA = '49de94b98890885d0c47bf78313d2383e62c0d3fc2583795bd05cc1a81291c60'


def verify_inputs(launch):
    verify(launch)
    verify_artifacts(PREFIX, launch['prefix_artifact_sha256'])
    verify_artifacts(DIAGNOSIS, {'result.json': DIAGNOSIS_SHA})
    verify_artifacts(FAILED, FAILED_BINDINGS)
    for root in (INPUT, READOUT):
        verify_artifacts(root, launch['replay_input_bindings'][str(root)])
    admission = launch['correction_admission']
    verify_artifacts(FITS, admission['base_admission']['fit_artifact_sha256'])
    verify_artifacts(CORRECTION, admission['correction_artifact_sha256'])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--preflight-only', action='store_true'); args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive implementation-equivalence replay required')
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    verify_artifacts(PREFIX, {'result.json': EXPECTED})
    result = read_json(PREFIX, 'result.json'); report = result['report']
    assert result['status'] == 'JOINT_FLOOR_REGISTERED_MAZE_PREFIX_COMPLETE'
    assert report['frames'] == 960 and report['first_requested_command_difference'] == 959
    assert report['final_terminal'] is None and report['stopped_before_unexecuted_outcome']
    bindings = {'result.json': EXPECTED, **result['artifact_sha256']}
    verify_artifacts(PREFIX, bindings)
    old = read_json(PREFIX, 'launch.json')
    sources = discover_sources((PROTOCOL, 'scripts/replay_go2_frame_cached_floor_prefix_v1.py',
        'lewm/tests/test_frame_floor_cache_development.py',
        'docs/go2_frame_cached_floor_source_derivatives_2026-09-09.json'), result['source_sha256'])
    resources = hardware()
    launch = old | dict(protocol=PROTOCOL, source_sha256=sources, output_root=str(OUTPUT),
        prefix_artifact_sha256=bindings, prefix_result_sha256=EXPECTED,
        implementation_class='FrameCachedJointFloorRoundTripController',
        diagnosis_result_sha256=DIAGNOSIS_SHA, hardware=resources, maximum_frames=960,
        native_execution=False, model_training=False, cpu_processes=1, numerical_threads=1,
        concurrency_reason='immutable prefix replay independent of single current native scene',
        minimum_available_ram_bytes=8*1024**3, output_allowance_bytes=128*1024**2,
        os_resource_limits_enforced=False, real_time_qualified=False)
    verify_inputs(launch)
    memory_ok = resources['memory_available_bytes'] >= 8*1024**3
    storage_ok = resources['artifact_free_bytes'] >= RESERVE_BYTES+128*1024**2
    if args.preflight_only:
        print('FRAME_CACHED_FLOOR_PREFLIGHT', json.dumps(dict(source_count=len(sources), hardware=resources,
            memory_admission_pass=memory_ok, storage_admission_pass=storage_ok, output_created=False)), flush=True)
        return
    if not memory_ok or not storage_ok: raise ValueError('replay resource envelope unavailable')
    create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch)
    print('FRAME_CACHED_FLOOR_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    started = time.perf_counter()
    try:
        name, index, variant, condition, model_name = CASE
        model, c, v = load_assigned(launch['correction_admission'], model_name)
        assert (c, v) == (condition, variant)
        before = state_digest(model.state_dict()); assert before == report['model_state_sha256']
        controller = FrameCachedJointFloorRoundTripController(model, ArticulatedCollisionGeometry(URDF),
            public_mission=public_mission(index), navigation_ticks=NAVIGATION_TICKS,
            persistent=True, condition=condition, variant=variant)
        directory = INPUT/name; reader = IntentReturnRGBDReplay(directory)
        acquisitions = read_json(directory, 'auxiliary_camera_audit.json')
        frames = 0
        with (OUTPUT/'timings_and_equality.jsonl').open('x') as stream:
            for i, saved in enumerate(read_rows(PREFIX)):
                if i >= 960 or saved['tick'] != i: raise ValueError('exact bound prefix population required')
                p, d, f, now = reader.packet(i)
                auxiliary = packet(directory, i, p, public_acquisition(acquisitions[i]), now_ns=now)
                begin = time.perf_counter()
                decision = controller.observe(p, d, f, now_ns=now, auxiliary_depth=auxiliary)
                elapsed = 1000*(time.perf_counter()-begin)
                normalized = json.loads(json.dumps(decision, allow_nan=False))
                if normalized != saved['decision']:
                    write_json(OUTPUT/'mismatch.json', dict(frame=i, candidate_decision=normalized))
                    raise ValueError(f'complete implementation decision differs at frame {i}')
                payload = json.dumps(normalized, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
                stream.write(json.dumps(dict(frame=i, controller_wall_ms=elapsed,
                    decision_sha256=hashlib.sha256(payload).hexdigest(), complete_decision_exact=True,
                    cache_counts=controller.mapper.last_cache_counts))+'\n'); stream.flush()
                frames += 1
                if i % 100 == 0: print('FRAME_CACHED_FLOOR_FRAME', i, flush=True)
        assert frames == 960 and state_digest(model.state_dict()) == before
        verify_inputs(launch)
        write_json(OUTPUT/'result.json', dict(status='FRAME_CACHED_FLOOR_PREFIX_COMPLETE', frames=frames,
            source_sha256=sources, artifact_sha256={n:digest(OUTPUT/n) for n in ('launch.json', 'timings_and_equality.jsonl')},
            prefix_result_sha256=EXPECTED, complete_decisions_exact=True,
            model_state_unchanged=True, model_state_sha256=before, wall_s=time.perf_counter()-started,
            hardware_after=hardware(), stopped_before_unexecuted_outcome=True,
            native_execution=False, model_training=False, navigation_qualified=False,
            controlled_speed_comparison=False, real_time_qualified=False, goal_achieved=False))
        print('FRAME_CACHED_FLOOR_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='FRAME_CACHED_FLOOR_PREFIX_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
