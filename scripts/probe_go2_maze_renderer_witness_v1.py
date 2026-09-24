"""Bounded actual maze-camera witness integration after the current native audit."""
import argparse
import json
import os
from pathlib import Path
import time
import cv2
import torch
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.pulse_timed_training_runner_development import state_digest
from lewm.novel_maze_round_trip_contract_development import RESERVE_BYTES, PERSISTENCE_HEADROOM_BYTES
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.renderer_witness_maze_probe_episode_development import collect, artifacts
from scripts.dual_camera_settled_maze_audit_development import audit
from scripts.renderer_witness_maze_probe_comparison_development import compare
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.run_go2_dual_camera_settled_maze_pilot_v1 import OUTPUT as INPUT, CASE, verify_inputs as verify_native
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_maze_renderer_witness_probe_v1_attempt_001'
PROTOCOL = 'docs/go2_maze_renderer_witness_probe_v1_2026-09-09.md'
CAMERA_PROBE = BASE/'go2_camera_renderer_identity_v1_attempt_001'
CAMERA_RESULT = 'c1636416aa33295641358bec601a6bb6202e664410104f6afb9dbcd35103a8f5'
OUTPUT_ALLOWANCE = 256*1024**2


def verify_all(launch):
    verify(launch)
    verify_artifacts(INPUT, launch['completed_native_artifact_sha256'])
    verify_native(read_json(INPUT, 'launch.json'))
    verify_artifacts(CAMERA_PROBE, launch['camera_probe_artifact_sha256'])
    for name, h in launch['additional_runtime_sha256'].items():
        if digest(Path(name)) != h: raise ValueError('actual camera runtime changed: '+name)


def admit_native(result, report):
    if (result['status'] != 'DUAL_CAMERA_SETTLED_MAZE_PILOT_V1_COMPLETE' or len(result['conditions']) != 1
            or result['conditions'][0]['case'] != CASE[0]
            or result['conditions'][0]['status'] != 'DUAL_CAMERA_SETTLED_MAZE_COLLECTED_AND_RAW_AUDITED'
            or result['conditions'][0]['prefix_comparison']['physical_and_public_prefix_exact'] is not True
            or result['conditions'][0]['prefix_comparison']['complete_candidate_decisions_match_prospective_prefix'] is not True):
        raise ValueError('completed current native collection, raw audit and prospective prefix required')
    for key in ('raw_sensor_reconstruction_pass', 'additional_auxiliary_rgb_reconstructed',
            'raw_model_command_replay_pass', 'raw_command_audit_pass', 'model_state_unchanged'):
        if report[key] is not True: raise ValueError('completed native raw invariant required: '+key)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--native-result-sha256', required=True)
    parser.add_argument('--preflight-only', action='store_true'); args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive actual maze-camera integration probe required')
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    verify_artifacts(INPUT, {'result.json': args.native_result_sha256})
    native = read_json(INPUT, 'result.json')
    native_ids = {'result.json': args.native_result_sha256, **native['artifact_sha256']}
    verify_artifacts(INPUT, native_ids)
    admit_native(native, read_json(INPUT, CASE[0]+'_audit.json'))
    old = read_json(INPUT, 'launch.json')
    verify_artifacts(CAMERA_PROBE, {'result.json': CAMERA_RESULT})
    camera = read_json(CAMERA_PROBE, 'result.json')
    if (camera['status'] != 'CAMERA_RENDERER_IDENTITY_INTEGRATION_COMPLETE'
            or camera['actual_genesis_camera_context_queried'] is not True
            or camera['repeated_rgb_depth_after_readback_exact'] is not True):
        raise ValueError('completed actual Camera API integration required')
    camera_ids = {'result.json': CAMERA_RESULT, **camera['artifact_sha256']}
    verify_artifacts(CAMERA_PROBE, camera_ids)
    camera_launch = read_json(CAMERA_PROBE, 'launch.json')
    inherited = dict(old['source_sha256'])
    for name, h in camera['source_sha256'].items():
        if name in inherited and inherited[name] != h: raise ValueError('incompatible frozen camera source: '+name)
        inherited[name] = h
    sources = discover_sources((PROTOCOL, 'scripts/probe_go2_maze_renderer_witness_v1.py',
        'lewm/tests/test_maze_renderer_witness_development.py',
        'lewm/tests/test_maze_renderer_probe_scope_development.py'), inherited)
    resources = hardware()
    launch = {k:old[k] for k in ('input_sha256', 'native_sha256', 'native_scene_sha256',
        'native_geometry_sha256', 'opencv_binary_sha256', 'opencv_version', 'rules')}
    launch.update(protocol=PROTOCOL, output_root=str(OUTPUT), source_sha256=sources,
        completed_native_artifact_sha256=native_ids, camera_probe_artifact_sha256=camera_ids,
        additional_runtime_sha256=camera_launch['additional_runtime_sha256'],
        render_environment={k:os.environ.get(k) for k in ('PYOPENGL_PLATFORM','LIBGL_ALWAYS_SOFTWARE',
            'EGL_DEVICE_ID','MESA_GL_VERSION_OVERRIDE','MESA_LOADER_DRIVER_OVERRIDE')},
        correction_admission=old['correction_admission'], model_state_sha256=old['prefix_report']['model_state_sha256'],
        robot_urdf_path=str(URDF), robot_urdf_sha256=digest(URDF), planned_case=CASE,
        hardware=resources, observations=3, expected_physics_samples=900, completed_warmup_commands=3,
        warmup_requested_command=[0., 0., 0.], context_endpoints_per_observation=2,
        cpu_processes=1, numerical_threads=1, native_scene_workers=1,
        memory_admission_bytes=32*1024**3, output_allowance_bytes=OUTPUT_ALLOWANCE,
        minimum_free_bytes=RESERVE_BYTES, persistence_headroom_bytes=PERSISTENCE_HEADROOM_BYTES,
        os_resource_limits_enforced=False, physics_paused_during_compute=True,
        concurrency_reason='one bounded startup scene after the completed dual-camera native audit; no parallel scene',
        model_training=False, native_execution=True, navigation_experiment=False,
        historical_context_inferred=False, raster_error_bound_proven=False,
        predecessor_outcome_unchanged=True, navigation_qualified=False, real_time_qualified=False, goal_achieved=False)
    verify_all(launch)
    memory_ok = resources['memory_available_bytes'] >= 32*1024**3
    storage_ok = resources['artifact_free_bytes'] >= RESERVE_BYTES+PERSISTENCE_HEADROOM_BYTES+OUTPUT_ALLOWANCE
    if args.preflight_only:
        print('MAZE_RENDERER_WITNESS_PREFLIGHT', json.dumps(dict(source_count=len(sources), hardware=resources,
            completed_inputs_and_sources_verified=True, memory_admission_pass=memory_ok,
            storage_admission_pass=storage_ok, output_created=False)), flush=True)
        return
    if not memory_ok or not storage_ok: raise ValueError('bounded actual-camera probe resources unavailable')
    create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch)
    print('MAZE_RENDERER_WITNESS_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    started = time.perf_counter(); collection = None; bindings = {}
    try:
        name, index, variant, condition, model_name = CASE
        model, c, v = load_assigned(launch['correction_admission'], model_name)
        assert (c, v) == (condition, variant)
        before = state_digest(model.state_dict()); assert before == launch['model_state_sha256']
        collection = collect(index, sources[PROTOCOL], output=OUTPUT, model=model,
            geometry=ArticulatedCollisionGeometry(URDF), episode_name=name, condition=condition, variant=variant)
        bindings = {name+'/'+n:digest(OUTPUT/name/n) for n in artifacts(index, collection)}
        verify_artifacts(OUTPUT, bindings)
        if (any(collection[k] != v for k, v in dict(decisions=3, rgbd_frames=3, auxiliary_frames=3,
                command_ticks=3, completed_ticks=3, physics_samples=900, terminal_zero_ticks=0).items())
                or collection['acquisition_stop'] != 'DECLARED_THREE_OBSERVATION_PROBE_BOUNDARY'
                or collection['physical_stop'] is not None or collection['schedule_terminal'] is not None
                or collection['renderer_witnesses_recorded'] is not True):
            raise ValueError('complete declared three-observation camera integration required')
        if sum((OUTPUT/n).stat().st_size for n in bindings) > OUTPUT_ALLOWANCE:
            raise ValueError('bounded probe output allowance exceeded')
        assert state_digest(model.state_dict()) == before
        replay_model, c, v = load_assigned(launch['correction_admission'], model_name)
        assert (c, v) == (condition, variant)
        raw = audit(index, collection, sources[PROTOCOL], input_root=OUTPUT, model=replay_model,
            robot_geometry=ArticulatedCollisionGeometry(URDF), episode_name=name, condition=condition, variant=variant)
        comparison = compare(INPUT/name, OUTPUT/name)
        for filename, value in ((name+'_audit.json', raw), ('comparison.json', comparison)):
            write_json(OUTPUT/filename, value); bindings[filename] = digest(OUTPUT/filename)
        bindings['launch.json'] = digest(OUTPUT/'launch.json')
        verify_all(launch); verify_artifacts(OUTPUT, bindings)
        if sum((OUTPUT/n).stat().st_size for n in bindings) > OUTPUT_ALLOWANCE:
            raise ValueError('bounded probe artifacts exceeded output allowance')
        write_json(OUTPUT/'result.json', dict(status='MAZE_RENDERER_WITNESS_INTEGRATION_COMPLETE',
            source_sha256=sources, artifact_sha256=bindings, collection=collection, comparison=comparison,
            raw_sensor_reconstruction_pass=raw['raw_sensor_reconstruction_pass'],
            raw_model_command_replay_pass=raw['raw_model_command_replay_pass'],
            raw_command_audit_pass=raw['raw_command_audit_pass'], model_state_unchanged=True,
            native_result_sha256=args.native_result_sha256, actual_maze_camera_endpoints_queried=True,
            predecessor_outcome_unchanged=True, new_navigation_episodes=0,
            historical_context_inferred=False, raster_error_bound_proven=False,
            model_training=False, navigation_qualified=False, real_time_qualified=False, goal_achieved=False,
            wall_s=time.perf_counter()-started, hardware_after=hardware()))
        print('MAZE_RENDERER_WITNESS_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(reason=repr(error), collection=collection, artifact_sha256=bindings))
        raise


if __name__ == '__main__': main()
