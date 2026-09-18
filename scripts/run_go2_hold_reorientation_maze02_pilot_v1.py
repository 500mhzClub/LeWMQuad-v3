"""Fresh maze2 physics after completed hold replay and scheduled hold-reorientation execution."""
import argparse
import contextlib
import multiprocessing
import resource
import time
import json
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import cv2
import numpy as np
import torch
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.novel_maze_round_trip_scene_development import specification, public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS
from lewm.pulse_timed_training_runner_development import state_digest
from lewm.all_phase_residual_maze02_study_development import resources_for as cohort_resources
from lewm.all_phase_residual_maze02_readout_development import case_readout
from lewm.independent_reactive_floor_transport_study_development import OUTCOME_KEYS, require_raw_audit
from scripts import hold_reorientation_native_inputs_development as inputs
from scripts.hold_reorientation_native_prefix_development import compare, boundary
from scripts.hold_reorientation_maze02_episode_development import collect, artifacts
from scripts.hold_reorientation_maze02_audit_development import audit
from scripts.all_phase_planner_model_admission_development import load_assigned
from lewm.all_phase_planner_model_adapter_development import AllPhasePlannerModel
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch
from scripts.supervised_commitment_contact_queue_gate_development import require_native_idle
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_hold_reorientation_maze02_pilot_v1_attempt_001'
SOURCE = 'scripts/run_go2_hold_reorientation_maze02_pilot_v1.py'
PROTOCOL = 'docs/go2_hold_reorientation_maze02_pilot_v1_2026-09-10.md'
TESTS = ('lewm/tests/test_hold_reorientation_native_launcher_development.py',
    'lewm/tests/test_hold_reorientation_native_prefix_development.py',
    'lewm/tests/test_hold_reorientation_native_source_development.py')
CASE = ('full_jepa_hold_reorientation_maze_02', *inputs.replay.CASE[1:])
WORKER_STATUS = 'HOLD_REORIENTATION_MAZE02_COLLECTED_AND_RAW_AUDITED'


def verify_inputs(launch, *, full=False):
    verify_ordered_launch(launch)
    expected = dict(planned_case=list(CASE), output_root=str(OUTPUT),
        implementation_class='HoldReorientationController',
        scene_specification=specification(2), public_mission=public_mission(2),
        model_state_sha256=inputs.replay.MODEL_SHA, robot_urdf_sha256=digest(URDF),
        navigation_ticks=NAVIGATION_TICKS, native_scene_workers=1, opencv_threads=1, blas_threads=1,
        maximum_tasks_per_process=1, physics_paused_during_compute=True,
        renderer_capture_witnesses_enabled=True, fresh_controller_and_memory=True,
        hold_reorientation_enabled=True, planner_interface_adapter_enabled=True,
        native_execution=True, model_training=False, independent_layout_development_execution=False,
        reused_development_layout=True)
    if any(launch[k] != v or type(launch[k]) is not type(v) for k, v in expected.items()):
        raise ValueError('exact prospective hold-reorientation native definition required')
    admission = launch['input_admission']; boundary(admission['prefix_report'])
    inputs.verify_bound(admission, launch['source_sha256'])
    if full and inputs.admit(admission['raw_prefix_wait_result_sha256'], admission['frontier_wait_result_sha256'],
            launch['source_sha256']) != admission:
        raise ValueError('complete original hold-reorientation input admission changed')


def assigned_model(launch):
    model, condition, variant = load_assigned(launch['input_admission']['correction_admission'], CASE[4])
    if (type(model) is not AllPhasePlannerModel or model.training
            or (condition, variant) != (CASE[3], CASE[2])
            or state_digest(model.state_dict()) != inputs.replay.MODEL_SHA):
        raise ValueError('original assigned hold-reorientation model state required')
    return model


def require_worker(record, report, prefix_report):
    if (record['status'] != WORKER_STATUS or 'failure' in record or record['case'] != CASE[0]
            or record['layout_index'] != CASE[1] or record['model_name'] != CASE[4]
            or record['condition'] != CASE[3] or record['variant'] != CASE[2]
            or record['model_state_sha256'] != inputs.replay.MODEL_SHA
            or record['model_state_unchanged'] is not True or report['layout_index'] != CASE[1]
            or report['hold_reorientation_enabled'] is not True
            or record['collection']['status'] != 'HOLD_REORIENTATION_MAZE02_TERMINAL_AUDIT_REQUIRED'
            or record['collection']['hold_reorientation_enabled'] is not True):
        raise ValueError('complete exact hold-reorientation collection and raw audit required')
    require_raw_audit(record, report, learned=True)
    expected = bool(report['native_evaluation']['native_round_trip_candidate_pass']
        and report['strict_physical_visibility_pass'] and not report['hard_measurement_failed_frames'])
    if record['verified_round_trip'] is not expected:
        raise ValueError('unchanged joint native and visibility success criteria required')
    receipt = record['prefix_comparison']; frames, changed = boundary(prefix_report)
    for key in ('physical_and_public_prefix_exact', 'all_preintervention_requested_commands_exact',
            'complete_candidate_decisions_match_prospective_prefix', 'all_compared_raw_model_forecasts_exact',
            'candidate_intervention_command_completed'):
        if receipt[key] is not True: raise ValueError('complete physical hold-reorientation intervention required: '+key)
    if (receipt['common_prefix_frames'] != frames or receipt['first_intervention_frame'] != changed
            or receipt['physical_prefix_samples'] != 750+50*changed
            or receipt['raw_model_forecast_comparisons'] != prefix_report['raw_model_forecast_comparisons']
            or receipt['original_intervention_command'] != prefix_report['original_requested_command']
            or receipt['candidate_intervention_command'] != prefix_report['candidate_requested_command']
            or receipt['following_physical_outcomes_compared'] is not False
            or receipt['unexecuted_outcomes_inferred'] is not False):
        raise ValueError('exact prospective physical boundary and forecast count required')


def worker(launch_sha):
    name, index, variant, condition, model_name = CASE
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    record = dict(case=name, layout_index=index, variant=variant, condition=condition, model_name=model_name,
        status='HOLD_REORIENTATION_MAZE02_WORKER_FAILED', artifact_sha256={}); started = time.perf_counter()
    with (OUTPUT/(name+'_worker.log')).open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            verify_artifacts(OUTPUT, {'launch.json': launch_sha}); launch = read_json(OUTPUT, 'launch.json')
            verify_inputs(launch); model = assigned_model(launch); before = state_digest(model.state_dict())
            result = collect(index, launch['source_sha256'][PROTOCOL], output=OUTPUT, model=model,
                geometry=ArticulatedCollisionGeometry(URDF), episode_name=name, condition=condition, variant=variant)
            if state_digest(model.state_dict()) != before: raise ValueError('collection changed assigned hold-reorientation model')
            ids = {name+'/'+n: digest(OUTPUT/name/n) for n in artifacts(index, result)}
            record.update(collection=result, artifact_sha256=dict(ids)); verify_artifacts(OUTPUT, ids)
            report = audit(index, result, launch['source_sha256'][PROTOCOL], input_root=OUTPUT,
                model=assigned_model(launch), robot_geometry=ArticulatedCollisionGeometry(URDF),
                episode_name=name, condition=condition, variant=variant)
            n = name+'_audit.json'; write_json(OUTPUT/n, report); ids[n] = digest(OUTPUT/n)
            record['artifact_sha256'] = dict(ids)
            receipt = compare(inputs.replay.original.OUTPUT/inputs.replay.CASE[0], OUTPUT/name,
                inputs.replay.OUTPUT, launch['input_admission']['prefix_report'])
            n = name+'_prefix_comparison.json'; write_json(OUTPUT/n, receipt); ids[n] = digest(OUTPUT/n)
            record['artifact_sha256'] = dict(ids)
            with np.load(OUTPUT/name/'physics_trace.npz', allow_pickle=False) as saved:
                readout = case_readout(report, result, saved['physics_contact'])
            n = name+'_readout.json'; write_json(OUTPUT/n, readout); ids[n] = digest(OUTPUT/n)
            record.update(status=WORKER_STATUS, artifact_sha256=ids, prefix_comparison=receipt, readout=readout,
                model_state_sha256=before, model_state_unchanged=True, hold_reorientation_enabled=True,
                **{k: report[k] for k in OUTCOME_KEYS})
            require_worker(record, report, launch['input_admission']['prefix_report'])
            verify_inputs(launch); verify_artifacts(OUTPUT, ids)
        except Exception as error:
            import traceback
            traceback.print_exc(); record.update(status='HOLD_REORIENTATION_MAZE02_WORKER_FAILED', failure=repr(error))
    record.update(wall_s=time.perf_counter()-started,
        maximum_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        worker_log_sha256=digest(OUTPUT/(name+'_worker.log')))
    write_json(OUTPUT/(name+'_worker_terminal.json'), record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-prefix-wait-result-sha256'); parser.add_argument('--frontier-wait-result-sha256')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--source-preflight-only', action='store_true'); modes.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive hold-reorientation native attempt; no retry/resume')
    sources = inputs.prepared_sources((SOURCE, PROTOCOL, *TESTS)); resources = hardware(); cohort_resources(resources, 1)
    if args.source_preflight_only:
        print('HOLD_REORIENTATION_NATIVE_SOURCE_PREFLIGHT', json.dumps(dict(source_count=len(sources), hardware=resources,
            native_execution=False, output_created=False, complete_input_admission_performed=False)), flush=True); return
    if not args.raw_prefix_wait_result_sha256 or not args.frontier_wait_result_sha256:
        raise ValueError('exact completed raw-prefix and frontier waiter identities required')
    print('HOLD_REORIENTATION_NATIVE_INPUT_ADMISSION_STARTED', len(sources), flush=True)
    admission = inputs.admit(args.raw_prefix_wait_result_sha256, args.frontier_wait_result_sha256, sources)
    old = read_json(inputs.replay.original.OUTPUT, 'launch.json')
    keys = ('input_sha256', 'native_sha256', 'native_scene_sha256', 'native_geometry_sha256',
        'opencv_binary_sha256', 'opencv_version', 'rules', 'renderer_environment')
    launch = {k: old[k] for k in keys}
    launch.update(source_sha256=sources, protocol=PROTOCOL, output_root=str(OUTPUT), input_admission=admission,
        planned_case=list(CASE), implementation_class='HoldReorientationController',
        scene_specification=specification(2), public_mission=public_mission(2), robot_urdf_sha256=digest(URDF),
        model_state_sha256=inputs.replay.MODEL_SHA, navigation_ticks=NAVIGATION_TICKS,
        native_scene_workers=1, opencv_threads=1, blas_threads=1, maximum_tasks_per_process=1,
        physics_paused_during_compute=True, renderer_capture_witnesses_enabled=True, fresh_controller_and_memory=True,
        hold_reorientation_enabled=True, planner_interface_adapter_enabled=True,
        native_execution=True, model_training=False, independent_layout_development_execution=False,
        reused_development_layout=True, navigation_qualified=False, real_time_qualified=False,
        hardware_qualified=False, goal_achieved=False)
    verify_inputs(launch); assigned_model(launch)
    resources = hardware(); launch['hardware'] = resources; launch['resource_admission'] = cohort_resources(resources, 1)
    if args.preflight_only:
        print('HOLD_REORIENTATION_NATIVE_PREFLIGHT', json.dumps(dict(source_count=len(sources), hardware=resources,
            complete_input_admission_performed=True, native_execution=False, output_created=False)), flush=True); return
    require_native_idle(); create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch); launch_sha = digest(OUTPUT/'launch.json')
    print('HOLD_REORIENTATION_NATIVE_LAUNCHED', launch_sha, flush=True); started = time.perf_counter()
    try:
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as pool:
                future = pool.submit(worker, launch_sha)
                while True:
                    monitor.write(json.dumps(dict(elapsed_s=time.perf_counter()-started, **hardware()))+'\n'); monitor.flush()
                    done, _ = wait([future], timeout=15, return_when=FIRST_COMPLETED)
                    if done: record = future.result(); break
        print('HOLD_REORIENTATION_NATIVE_TERMINAL', record['status'], record.get('verified_round_trip'), record.get('failure'), flush=True)
        if record['status'] != WORKER_STATUS:
            raise ValueError('hold-reorientation collection/audit/prefix incomplete: '+str(record.get('failure')))
        require_worker(record, read_json(OUTPUT, CASE[0]+'_audit.json'), admission['prefix_report'])
        ids = dict(record['artifact_sha256'])
        for n in ('launch.json', 'resource_monitor.jsonl', CASE[0]+'_worker.log', CASE[0]+'_worker_terminal.json'):
            ids[n] = digest(OUTPUT/n)
        verify_inputs(launch, full=True); verify_artifacts(OUTPUT, ids)
        write_json(OUTPUT/'result.json', dict(status='HOLD_REORIENTATION_MAZE02_PILOT_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, conditions=[record], wall_s=time.perf_counter()-started,
            prospective_prefix_result_sha256=admission['prefix_result_sha256'],
            raw_prefix_wait_result_sha256=args.raw_prefix_wait_result_sha256, frontier_wait_result_sha256=args.frontier_wait_result_sha256,
            measured_round_trip_successes=int(record['verified_round_trip']), reused_layout_executions=1,
            new_independent_layout_executions=0, hold_reorientation_enabled=True, model_training=False,
            navigation_qualified=False, real_time_qualified=False, hardware_qualified=False, goal_achieved=False))
        print('HOLD_REORIENTATION_NATIVE_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_HOLD_REORIENTATION_NATIVE_FAILURE', reason=repr(error), automatic_retry=False)); raise


if __name__ == '__main__': main()
