"""One fresh budget-only diagnostic after the original four-stage native queue."""
import argparse
import contextlib
import json
import multiprocessing
import resource
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import cv2
import numpy as np
import torch

from scripts import extended_budget_anchored_native_inputs_development as inputs
from scripts import extended_budget_anchored_prefix_development as prefix
from scripts import extended_budget_anchored_maze_development as pipeline
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch
from scripts.supervised_commitment_contact_queue_gate_development import require_native_idle
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.novel_maze_round_trip_scene_development import specification, public_mission
from lewm.pulse_timed_training_runner_development import state_digest
from lewm.all_phase_residual_maze02_readout_development import case_readout
from lewm.independent_reactive_floor_transport_study_development import OUTCOME_KEYS, require_raw_audit

SOURCE = 'scripts/run_go2_no_rgb_direct_extended_budget_maze02_pilot_v1.py'
PROTOCOL = 'docs/go2_no_rgb_direct_extended_budget_maze02_pilot_v1_2026-09-11.md'
TEST = 'lewm/tests/test_no_rgb_direct_extended_budget_native_development.py'
OUTPUT = BASE/'go2_no_rgb_direct_extended_budget_maze02_pilot_v1_attempt_001'
CASE = ('no_rgb_direct_extended_budget_anchored_maze_02', *inputs.CASE[1:])
WORKER_STATUS = 'NO_RGB_DIRECT_EXTENDED_BUDGET_MAZE02_COLLECTED_AND_RAW_AUDITED'


def resource_admission(resources):
    required = pipeline.COLLECTION_ALLOWANCE_BYTES+41*1024**3
    if resources['artifact_free_bytes'] < required or resources['memory_available_bytes'] < 32*1024**3:
        raise ValueError('55 GiB disk and 32 GiB available RAM required for the extended episode')
    return dict(required_free_bytes=required, minimum_available_memory_bytes=32*1024**3,
        native_scene_workers=1, os_resource_limits_enforced=False)


def definition():
    return dict(planned_case=list(CASE), output_root=str(OUTPUT),
        implementation_class='ResidualAnchoredContinuationController',
        scene_specification=specification(2), public_mission=public_mission(2),
        model_state_sha256=inputs.MODEL_SHA, robot_urdf_sha256=digest(URDF),
        navigation_ticks=4000, max_observations=4014, max_command_ticks=4013,
        collection_allowance_bytes=14*1024**3, native_scene_workers=1,
        opencv_threads=1, blas_threads=1, maximum_tasks_per_process=1,
        physics_paused_during_compute=True, renderer_capture_witnesses_enabled=True,
        fresh_controller_and_memory=True, planner_interface_adapter_enabled=True,
        budget_only_development_followup=True, queued_controller_changes_adopted=False,
        native_execution=True, model_training=False,
        independent_layout_development_execution=False, reused_development_layout=True,
        navigation_qualified=False, real_time_qualified=False, hardware_qualified=False, goal_achieved=False)


def verify_inputs(launch):
    verify_ordered_launch(launch)
    expected = definition()
    if any(type(launch[k]) is not type(v) or launch[k] != v for k, v in expected.items()):
        raise ValueError('exact prospective budget-only native definition required')
    inputs.verify_bound(launch['input_admission'], launch['source_sha256'])


def assigned_model(launch):
    original = read_json(inputs.batch.OUTPUT, 'launch.json')
    if original['input_admission']['correction_admission'] != launch['input_admission']['correction_admission']:
        raise ValueError('same original correction admission required')
    model = inputs.batch.assigned_model(original, inputs.CASE)
    if state_digest(model.state_dict()) != inputs.MODEL_SHA:
        raise ValueError('same preassigned no-RGB direct predictive model required')
    return model


def prefix_result(collection, launch, ids):
    mission = collection['mission_receipt'] or {}
    if collection['decisions'] < prefix.FRAMES or mission.get('frame', -1) < prefix.BOUNDARY:
        return dict(status='ORIGINAL_BUDGET_BOUNDARY_NOT_REACHED', decisions=collection['decisions'],
            final_mission_frame=mission.get('frame'), budget_only_preboundary_execution_supported=False,
            actual_paired_execution_compared=False, full_raw_audit_retained=True,
            unexecuted_outcomes_inferred=False)
    return dict(status='COMPLETE_ACTUAL_PREFIX_COMPARISON', actual_paired_execution_compared=True,
        **prefix.compare(inputs.batch.OUTPUT/inputs.CASE[0], OUTPUT/CASE[0],
            prior_bindings=launch['input_admission']['original_case_artifact_sha256'], current_bindings=ids))


def require_worker(record, report):
    if (record['status'] != WORKER_STATUS or 'failure' in record
            or record['case'] != CASE[0] or record['layout_index'] != 2
            or record['variant'] != CASE[2] or record['condition'] != CASE[3] or record['model_name'] != CASE[4]
            or record['model_state_sha256'] != inputs.MODEL_SHA or record['model_state_unchanged'] is not True
            or record['collection']['navigation_ticks'] != 4000
            or record['collection']['status'] != 'RESIDUAL_ANCHORED_CONTINUATION_MAZE_TERMINAL_AUDIT_REQUIRED'
            or record['collection']['storage_allowance_bytes'] != 14*1024**3):
        raise ValueError('complete exact extended-budget collection/model identity required')
    require_raw_audit(record, report, learned=True)
    success = bool(report['native_evaluation']['native_round_trip_candidate_pass']
        and report['strict_physical_visibility_pass'] and not report['hard_measurement_failed_frames'])
    if record['verified_round_trip'] is not success:
        raise ValueError('unchanged native and visibility success criteria required')
    receipt = record['prefix_comparison']; collection = record['collection']
    reached = collection['decisions'] >= prefix.FRAMES and (collection['mission_receipt'] or {}).get('frame', -1) >= prefix.BOUNDARY
    if reached:
        if (receipt['status'] != 'COMPLETE_ACTUAL_PREFIX_COMPARISON'
                or receipt['actual_paired_execution_compared'] is not True
                or receipt['frames'] != prefix.FRAMES or receipt['physical_prefix_samples'] != prefix.PHYSICS_SAMPLES
                or receipt['normalized_budget_paths'] != list(prefix.BUDGET_PATHS)
                or receipt['following_observations_compared'] is not False
                or receipt['unexecuted_outcomes_inferred'] is not False):
            raise ValueError('complete through-boundary prefix evidence required, including divergence')
        exact = bool(receipt['all_preboundary_decisions_exact'] and receipt['all_preboundary_commands_exact']
            and receipt['physical_prefix_exact'] and receipt['all_through_boundary_public_packets_exact'])
        if receipt['budget_only_preboundary_execution_supported'] is not exact:
            raise ValueError('budget-only prefix claim must follow actual paired evidence')
    elif receipt != dict(status='ORIGINAL_BUDGET_BOUNDARY_NOT_REACHED', decisions=collection['decisions'],
            final_mission_frame=(collection['mission_receipt'] or {}).get('frame'),
            budget_only_preboundary_execution_supported=False, actual_paired_execution_compared=False,
            full_raw_audit_retained=True, unexecuted_outcomes_inferred=False):
        raise ValueError('early terminal must preserve raw audit without claiming prefix completion')


def worker(launch_sha):
    name, index, variant, condition, model_name = CASE
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    started = time.perf_counter(); ids = {}
    record = dict(status='NO_RGB_DIRECT_EXTENDED_BUDGET_WORKER_FAILED', case=name, layout_index=index,
        variant=variant, condition=condition, model_name=model_name, artifact_sha256={})
    with (OUTPUT/(name+'_worker.log')).open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            verify_artifacts(OUTPUT, {'launch.json':launch_sha}); launch = read_json(OUTPUT, 'launch.json')
            verify_inputs(launch); resource_admission(hardware())
            model = assigned_model(launch); before = state_digest(model.state_dict())
            result = pipeline.collect(index, launch['source_sha256'][PROTOCOL], output=OUTPUT, model=model,
                geometry=ArticulatedCollisionGeometry(URDF), episode_name=name, condition=condition, variant=variant)
            record['collection'] = result
            if state_digest(model.state_dict()) != before: raise ValueError('collection changed original model')
            ids.update({name+'/'+n:digest(OUTPUT/name/n) for n in pipeline.artifacts(index, result)})
            record['artifact_sha256'] = dict(ids); verify_artifacts(OUTPUT, ids)
            report = pipeline.audit(index, result, launch['source_sha256'][PROTOCOL], input_root=OUTPUT,
                model=assigned_model(launch), robot_geometry=ArticulatedCollisionGeometry(URDF),
                episode_name=name, condition=condition, variant=variant)
            n = name+'_audit.json'; write_json(OUTPUT/n, report); ids[n] = digest(OUTPUT/n)
            record['artifact_sha256'] = dict(ids)
            with np.load(OUTPUT/name/'physics_trace.npz', allow_pickle=False) as raw:
                readout = case_readout(report, result, raw['physics_contact'])
            n = name+'_readout.json'; write_json(OUTPUT/n, readout); ids[n] = digest(OUTPUT/n)
            record.update(readout=readout, artifact_sha256=dict(ids), **{k:report[k] for k in OUTCOME_KEYS})
            receipt = prefix_result(result, launch, ids)
            n = name+'_prefix_comparison.json'; write_json(OUTPUT/n, receipt); ids[n] = digest(OUTPUT/n)
            record.update(status=WORKER_STATUS, artifact_sha256=dict(ids), prefix_comparison=receipt,
                model_state_sha256=before, model_state_unchanged=True)
            require_worker(record, report); verify_inputs(launch); verify_artifacts(OUTPUT, ids)
        except Exception as error:
            import traceback
            traceback.print_exc()
            record.update(status='NO_RGB_DIRECT_EXTENDED_BUDGET_WORKER_FAILED', failure=repr(error), artifact_sha256=dict(ids))
    record.update(wall_s=time.perf_counter()-started, maximum_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        worker_log_sha256=digest(OUTPUT/(name+'_worker.log')))
    write_json(OUTPUT/(name+'_worker_terminal.json'), record)
    return record


def main():
    parser = argparse.ArgumentParser()
    for name in ('adapter-batch', 'frontier-wait', 'hold-wait', 'contact-wait', 'tracking-wait'):
        parser.add_argument('--'+name+'-result-sha256')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--source-preflight-only', action='store_true'); mode.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive fresh episode; no retry/resume')
    sources = inputs.prepared_sources((SOURCE, PROTOCOL, TEST)); resources = hardware(); resource_admission(resources)
    if args.source_preflight_only:
        print('EXTENDED_BUDGET_NATIVE_SOURCE_PREFLIGHT', json.dumps(dict(source_count=len(sources), hardware=resources,
            native_execution=False, output_created=False, complete_input_admission_performed=False)), flush=True); return
    ids = dict(frontier=args.frontier_wait_result_sha256, hold=args.hold_wait_result_sha256, contact=args.contact_wait_result_sha256)
    if not all([args.adapter_batch_result_sha256, args.tracking_wait_result_sha256, *ids.values()]):
        raise ValueError('completed batch and all four original waiter result hashes required')
    admission = inputs.admit(args.adapter_batch_result_sha256, ids, args.tracking_wait_result_sha256, sources)
    old = read_json(inputs.batch.OUTPUT, 'launch.json')
    keys = ('input_sha256', 'native_sha256', 'native_scene_sha256', 'native_geometry_sha256',
        'opencv_binary_sha256', 'opencv_version', 'rules', 'renderer_environment')
    launch = {k:old[k] for k in keys} | definition() | dict(source_sha256=sources, protocol=PROTOCOL,
        input_admission=admission, hardware=hardware())
    launch['resource_admission'] = resource_admission(launch['hardware'])
    verify_inputs(launch); assigned_model(launch)
    if args.preflight_only:
        print('EXTENDED_BUDGET_NATIVE_PREFLIGHT', json.dumps(dict(source_count=len(sources),
            native_execution=False, output_created=False, complete_input_admission_performed=True)), flush=True); return
    require_native_idle(); resource_admission(hardware())
    create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch); launch_sha = digest(OUTPUT/'launch.json')
    print('EXTENDED_BUDGET_NATIVE_LAUNCHED', launch_sha, flush=True); started = time.perf_counter()
    try:
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as pool:
                future = pool.submit(worker, launch_sha)
                while True:
                    monitor.write(json.dumps(dict(elapsed_s=time.perf_counter()-started, **hardware()))+'\n'); monitor.flush()
                    done, _ = wait([future], timeout=15, return_when=FIRST_COMPLETED)
                    if done: record = future.result(); break
        require_worker(record, read_json(OUTPUT, CASE[0]+'_audit.json'))
        bindings = dict(record['artifact_sha256'])
        for n in ('launch.json', 'resource_monitor.jsonl', CASE[0]+'_worker.log', CASE[0]+'_worker_terminal.json'):
            bindings[n] = digest(OUTPUT/n)
        if record != read_json(OUTPUT, CASE[0]+'_worker_terminal.json'):
            raise ValueError('exact persisted worker terminal required')
        verify_inputs(launch); verify_artifacts(OUTPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='NO_RGB_DIRECT_EXTENDED_BUDGET_MAZE02_PILOT_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=bindings, conditions=[record], wall_s=time.perf_counter()-started,
            measured_round_trip_successes=int(record['verified_round_trip']),
            budget_only_preboundary_execution_supported=record['prefix_comparison']['budget_only_preboundary_execution_supported'],
            reused_layout_executions=1, new_independent_layout_executions=0, automatic_retry=False,
            model_training=False, navigation_qualified=False, real_time_qualified=False, hardware_qualified=False, goal_achieved=False))
        print('EXTENDED_BUDGET_NATIVE_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_EXTENDED_BUDGET_NATIVE_FAILURE', reason=repr(error), automatic_retry=False))
        raise


if __name__ == '__main__': main()
