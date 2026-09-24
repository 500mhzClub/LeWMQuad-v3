"""Fresh maze1 contact-horizon intervention after the existing native queue."""
import argparse
import contextlib
import json
import multiprocessing
import resource
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import cv2
import torch
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.novel_maze_round_trip_scene_development import specification, public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS, RESERVE_BYTES, PERSISTENCE_HEADROOM_BYTES
from lewm.executed_waypoint_resource_envelope_development import COLLECTION_ALLOWANCE_BYTES
from lewm.pulse_timed_training_runner_development import state_digest
from lewm.supervised_rollout_maze_study_development import SUPERVISED_STATE, planned_cases
from lewm.independent_reactive_floor_transport_study_development import require_raw_audit, MATCHED_KEYS
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.supervised_commitment_contact_maze01_episode_development import collect, artifacts
from scripts.supervised_commitment_contact_maze01_audit_development import audit
from scripts.supervised_commitment_contact_native_prefix_development import admit_prefix, fixed_report, compare
from scripts.replay_go2_supervised_commitment_contact_prefix_v1 import (
    OUTPUT as PREFIX, INPUT as PRIOR, CASE as PRIOR_CASE, FIXED, DIAGNOSTIC_SHA,
    verify_input_context as verify_prefix_context, admit)
from scripts.supervised_commitment_contact_queue_gate_development import (
    QUEUE, QUEUE_LAUNCH_SHA, verify_queue_launch, verify_queue_completion, require_native_idle)
from scripts.scoped_verification_digest_development import verify_with_scoped_digests
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.partial_floor_height_scoped_verification_admission_development import BENCHMARK_SHA, admit_benchmark

OUTPUT = BASE/'go2_supervised_commitment_contact_maze01_pilot_v1_attempt_001'
CASE = ('full_supervised_commitment_contact_maze_01', *planned_cases()[0][1:])
PROTOCOL = 'docs/go2_supervised_commitment_contact_maze01_pilot_v1_2026-09-09.md'
SOURCE = 'scripts/run_go2_supervised_commitment_contact_maze01_pilot_v1.py'
PREFIX_SHA = '37b29828635e88fab77f81447f6a05b890911426fc3478d8aac451426a229de0'


def verify_input_context(launch):
    verify(launch); verify_artifacts(PREFIX, launch['prefix_artifact_sha256'])
    verify_artifacts(PRIOR, launch['prior_artifact_sha256'])
    prefix = read_json(PREFIX, 'launch.json')
    if (prefix['diagnostic_result_sha256'] != DIAGNOSTIC_SHA
            or prefix['verification_benchmark_result_sha256'] != BENCHMARK_SHA
            or any(prefix['replay_input_bindings'].get(k) != v for k, v in FIXED.items())
            or prefix['replay_input_bindings'] != launch['prior_artifact_sha256']
            or prefix['correction_admission'] != launch['correction_admission']
            or launch['prefix_report'] != read_json(PREFIX, 'result.json')['report']):
        raise ValueError('all original replay identities, inputs, model and result required')
    fixed_report(launch['prefix_report']); verify_prefix_context(prefix)
    native = read_json(PRIOR, 'launch.json')
    if any(launch[k] != native[k] for k in MATCHED_KEYS):
        raise ValueError('same original native environment, budget and numerical settings required')
    if (launch['planned_case'] != list(CASE) or launch['scene_specification'] != specification(1)
            or launch['public_mission'] != public_mission(1) or launch['model_state_sha256'] != SUPERVISED_STATE
            or launch['implementation_class'] != 'CommitmentContactController'
            or launch['commitment_contact_policy_enabled'] is not True
            or launch['scored_pose_horizon_ns'] != 100_000_000 or launch['scored_contact_horizon_ns'] != 100_000_000
            or launch['path_constraint_horizon_ns'] != 800_000_000 or launch['contact_penalty_coefficient_m'] != 1.2):
        raise ValueError('fixed supervised commitment-contact native definition required')
    verify_queue_launch(launch['source_sha256'])
    if launch['preflight_only'] is False:
        if verify_queue_completion(launch['queue_result_sha256'], launch['source_sha256']) != launch['queue_completion']:
            raise ValueError('same completed original queue required')
    elif launch['preflight_only'] is not True or launch['queue_completion'] is not None:
        raise ValueError('explicit read-only preflight or authenticated actual queue completion required')


def verify_inputs(launch):
    if (launch['verification_benchmark_result_sha256'] != BENCHMARK_SHA
            or launch['prospective_prefix_result_sha256'] != PREFIX_SHA
            or launch['prefix_artifact_sha256'].get('result.json') != PREFIX_SHA
            or any(launch['prior_artifact_sha256'].get(k) != v for k, v in FIXED.items())):
        raise ValueError('exact completed first worker, prefix and benchmark required')
    admit_benchmark(); before = fingerprint(launch)
    result, counters = verify_with_scoped_digests(verify_input_context, digest, launch)
    if result is not None or fingerprint(launch) != before: raise ValueError('unchanged original verification context required')
    print('COMMITMENT_CONTACT_NATIVE_INPUTS_VERIFIED', counters, flush=True)


def worker(launch_sha):
    name, index, variant, condition, model_name = CASE
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    terminal = dict(case=name, layout_index=index, model_name=model_name,
        status='SUPERVISED_COMMITMENT_CONTACT_MAZE01_WORKER_FAILED', artifact_sha256={}); started = time.perf_counter()
    with (OUTPUT/(name+'_worker.log')).open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            verify_artifacts(OUTPUT, {'launch.json': launch_sha}); launch = read_json(OUTPUT, 'launch.json')
            if launch['preflight_only'] is not False: raise ValueError('preflight cannot execute a native worker')
            verify_inputs(launch)
            if digest(URDF) != launch['robot_urdf_sha256']: raise ValueError('frozen robot required')
            model, c, v = load_assigned(launch['correction_admission'], model_name)
            if (c, v) != (condition, variant) or state_digest(model.state_dict()) != SUPERVISED_STATE:
                raise ValueError('same assigned supervised model required')
            result = collect(index, launch['source_sha256'][PROTOCOL], output=OUTPUT, model=model,
                geometry=ArticulatedCollisionGeometry(URDF), episode_name=name, condition=condition, variant=variant)
            if state_digest(model.state_dict()) != SUPERVISED_STATE: raise ValueError('collection changed model weights')
            bindings = {name+'/'+n: digest(OUTPUT/name/n) for n in artifacts(index, result)}
            verify_artifacts(OUTPUT, bindings); terminal.update(collection=result, artifact_sha256=dict(bindings))
            replay_model, c, v = load_assigned(launch['correction_admission'], model_name)
            if (c, v) != (condition, variant) or state_digest(replay_model.state_dict()) != SUPERVISED_STATE:
                raise ValueError('fresh identical supervised audit model required')
            report = audit(index, result, launch['source_sha256'][PROTOCOL], input_root=OUTPUT, model=replay_model,
                robot_geometry=ArticulatedCollisionGeometry(URDF), episode_name=name, condition=condition, variant=variant)
            audit_name = name+'_audit.json'; write_json(OUTPUT/audit_name, report); bindings[audit_name] = digest(OUTPUT/audit_name)
            terminal['artifact_sha256'] = dict(bindings)
            witness = compare(PRIOR/PRIOR_CASE, OUTPUT/name, PREFIX, launch['prefix_report'])
            prefix_name = name+'_prefix_comparison.json'; write_json(OUTPUT/prefix_name, witness); bindings[prefix_name] = digest(OUTPUT/prefix_name)
            terminal['artifact_sha256'] = dict(bindings)
            verify_inputs(launch); verify_artifacts(OUTPUT, bindings)
            terminal.update(status='SUPERVISED_COMMITMENT_CONTACT_MAZE01_COLLECTED_AND_RAW_AUDITED',
                verified_round_trip=report['verified_round_trip'], native_evaluation=report['native_evaluation'],
                strict_physical_visibility_pass=report['strict_physical_visibility_pass'],
                hard_measurement_failed_frames=report['hard_measurement_failed_frames'],
                renderer_capture_audit=report['renderer_capture_audit'], prefix_comparison=witness,
                model_state_unchanged=True, model_state_sha256=SUPERVISED_STATE, condition=condition, variant=variant,
                commitment_contact_policy_enabled=True, reused_development_layout=True)
        except Exception as error:
            import traceback
            traceback.print_exc(); terminal['failure'] = repr(error)
    terminal.update(wall_s=time.perf_counter()-started, maximum_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        worker_log_sha256=digest(OUTPUT/(name+'_worker.log')))
    write_json(OUTPUT/(name+'_worker_terminal.json'), terminal)
    return terminal


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prefix-result-sha256', required=True)
    parser.add_argument('--queue-result-sha256'); parser.add_argument('--preflight-only', action='store_true'); args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive commitment-contact native attempt required')
    if args.prefix_result_sha256 != PREFIX_SHA: raise ValueError('exact completed contact-horizon prefix required')
    if not args.preflight_only and not args.queue_result_sha256: raise ValueError('completed existing queue identity required before execution')
    resources = hardware()
    if resources['memory_available_bytes'] < 8*1024**3: raise ValueError('native preparation RAM unavailable')
    verify_artifacts(PREFIX, {'result.json': PREFIX_SHA}); prefix = read_json(PREFIX, 'result.json')
    prefix_ids = dict(prefix['artifact_sha256'], **{'result.json': PREFIX_SHA}); verify_artifacts(PREFIX, prefix_ids)
    report = admit_prefix(PREFIX, prefix); old, record, audit_report, prior_ids = admit(); benchmark = admit_benchmark()
    verify_artifacts(QUEUE, {'launch.json': QUEUE_LAUNCH_SHA}); queue_launch = read_json(QUEUE, 'launch.json')
    inherited = dict(prefix['source_sha256'])
    for source in (old, benchmark, queue_launch):
        for name, sha in source['source_sha256'].items():
            if name in inherited and inherited[name] != sha: raise ValueError('incompatible frozen source: '+name)
            inherited[name] = sha
    sources = discover_sources((PROTOCOL, SOURCE, 'lewm/tests/test_supervised_commitment_contact_native_development.py',
        'lewm/tests/test_supervised_commitment_contact_native_launcher_development.py',
        'docs/go2_supervised_commitment_contact_prefix_result_2026-09-09.md'), inherited)
    keys = ('input_sha256', 'native_sha256', 'native_scene_sha256', 'native_geometry_sha256',
        'opencv_binary_sha256', 'opencv_version', 'rules', 'renderer_environment', 'correction_admission')
    launch = {k: old[k] for k in keys}
    queue_completion = None if args.preflight_only else verify_queue_completion(args.queue_result_sha256, sources)
    launch.update(protocol=PROTOCOL, source_sha256=sources, output_root=str(OUTPUT), planned_case=list(CASE),
        scene_specification=specification(1), public_mission=public_mission(1), implementation_class='CommitmentContactController',
        robot_urdf_sha256=digest(URDF), verification_benchmark_result_sha256=BENCHMARK_SHA,
        prospective_prefix_result_sha256=PREFIX_SHA, prefix_artifact_sha256=prefix_ids, prefix_report=report,
        prior_artifact_sha256=prior_ids, model_state_sha256=SUPERVISED_STATE,
        preflight_only=args.preflight_only, queue_result_sha256=args.queue_result_sha256, queue_completion=queue_completion,
        navigation_ticks=NAVIGATION_TICKS, shared_outbound_return_budget=True, native_scene_workers=1,
        opencv_threads=1, blas_threads=1, maximum_tasks_per_process=1, minimum_free_bytes=RESERVE_BYTES,
        planned_collection_allowance_bytes=COLLECTION_ALLOWANCE_BYTES, persistence_headroom_bytes=PERSISTENCE_HEADROOM_BYTES,
        memory_admission_bytes=32*1024**3, os_resource_limits_enforced=False, hardware_before=resources,
        physics_paused_during_compute=True, native_execution=not args.preflight_only, model_training=False,
        measured_floor_transport_enabled=True, renderer_capture_witnesses_enabled=True, commitment_contact_policy_enabled=True,
        scored_pose_horizon_ns=100_000_000, scored_contact_horizon_ns=100_000_000, path_constraint_horizon_ns=800_000_000,
        contact_penalty_coefficient_m=1.2, contact_scores_calibrated=False, original_feasibility_checks_unchanged=True,
        prospective_physical_and_public_prefix_required=True, changed_command_completion_required=True,
        independent_layout_development_execution=False, reused_development_layout=True,
        navigation_qualified=False, real_time_qualified=False, hardware_qualified=False, goal_achieved=False)
    verify_inputs(launch); resources = hardware(); launch['hardware'] = resources
    memory_ok = resources['memory_available_bytes'] >= 32*1024**3
    storage_ok = resources['artifact_free_bytes'] >= RESERVE_BYTES+COLLECTION_ALLOWANCE_BYTES+PERSISTENCE_HEADROOM_BYTES
    if args.preflight_only:
        print('COMMITMENT_CONTACT_NATIVE_PREFLIGHT', json.dumps(dict(source_count=len(sources), hardware=resources,
            memory_admission_pass=memory_ok, storage_admission_pass=storage_ok, input_and_source_bindings_verified=True,
            queue_completion_required_before_execution=True, queue_completion_verified=False,
            output_created=False, native_execution=False)), flush=True); return
    if not memory_ok or not storage_ok: raise ValueError('native resource admission failed')
    require_native_idle()
    create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch); launch_sha = digest(OUTPUT/'launch.json'); started = time.perf_counter()
    print('COMMITMENT_CONTACT_NATIVE_LAUNCHED', launch_sha, flush=True)
    try:
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as pool:
                future = pool.submit(worker, launch_sha)
                while True:
                    monitor.write(json.dumps(dict(elapsed_s=time.perf_counter()-started, **hardware()))+'\n'); monitor.flush()
                    done, _ = wait([future], timeout=15, return_when=FIRST_COMPLETED)
                    if done: record = future.result(); break
        print('COMMITMENT_CONTACT_NATIVE_TERMINAL', record['status'], record.get('verified_round_trip'), record.get('failure'), flush=True)
        if record['status'] != 'SUPERVISED_COMMITMENT_CONTACT_MAZE01_COLLECTED_AND_RAW_AUDITED':
            raise ValueError('collection/raw audit/physical prefix failed; evidence retained')
        require_raw_audit(record, read_json(OUTPUT, CASE[0]+'_audit.json'), learned=True)
        bindings = dict(record['artifact_sha256'])
        for name in ('launch.json', 'resource_monitor.jsonl', CASE[0]+'_worker.log', CASE[0]+'_worker_terminal.json'):
            bindings[name] = digest(OUTPUT/name)
        verify_inputs(launch); verify_artifacts(OUTPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='SUPERVISED_COMMITMENT_CONTACT_MAZE01_PILOT_V1_COMPLETE',
            conditions=[record], source_sha256=sources, artifact_sha256=bindings, prospective_prefix_result_sha256=PREFIX_SHA,
            queue_result_sha256=args.queue_result_sha256, wall_s=time.perf_counter()-started,
            measured_round_trip_successes=int(record['verified_round_trip']), reused_layout_executions=1,
            new_independent_layout_executions=0, model_training=False, commitment_contact_policy_enabled=True,
            navigation_qualified=False, hardware_qualified=False, real_time_qualified=False, goal_achieved=False))
        print('COMMITMENT_CONTACT_NATIVE_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_COMMITMENT_CONTACT_NATIVE_FAILURE', reason=repr(error))); raise


if __name__ == '__main__': main()
