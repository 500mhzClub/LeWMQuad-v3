"""One fully nonpredictive native episode after the matched nominal comparison."""
import argparse
import contextlib
import json
import multiprocessing
import resource
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
import psutil
import torch

from scripts import reactive_measured_plane_native_inputs_development as inputs
from scripts import reactive_measured_plane_extended_maze_development as pipeline
from scripts import reactive_measured_plane_native_prefix_development as prefix
from scripts.run_go2_nominal_measured_plane_maze02_v1 import wait_for_idle
from lewm.independent_reactive_floor_transport_study_development import MATCHED_KEYS, OUTCOME_KEYS, require_raw_audit

run = inputs.run
original = inputs.learned_inputs.learned.original
SOURCE = 'scripts/run_go2_reactive_measured_plane_maze02_v1.py'
PROTOCOL = 'docs/go2_reactive_measured_plane_maze02_v1_2026-09-11.md'
TESTS = ('lewm/tests/test_reactive_measured_plane_native_development.py',
    'lewm/tests/test_reactive_measured_plane_native_launcher_development.py')
OUTPUT = run.BASE/'go2_reactive_measured_plane_maze02_v1_attempt_001'
CASE = ('reactive_measured_plane_maze_02', 2)
WORKER_STATUS = 'REACTIVE_MEASURED_PLANE_COLLECTED_AND_RAW_AUDITED'
resources = original.resources


def definition():
    return pipeline.definition() | dict(planned_case=list(CASE), output_root=str(OUTPUT),
        scene_specification=original.specification(2), public_mission=original.public_mission(2),
        robot_urdf_sha256=run.digest(prefix.replay.job.URDF), native_scene_workers=1,
        opencv_threads=1, blas_threads=1, maximum_tasks_per_process=1,
        planner_interface_adapter_enabled=False, native_execution=True, model_training=False,
        independent_layout_development_execution=False, reused_development_layout=True)


def verify_inputs(launch):
    original.require_environment(launch); original.old.verify_ordered_launch(launch)
    if any(type(launch[k]) is not type(v) or launch[k] != v for k, v in definition().items()):
        raise ValueError('exact fully nonpredictive measured-plane definition required')
    baseline = inputs.learned_inputs.learned_launch()
    if any(type(launch[k]) is not type(baseline[k]) or launch[k] != baseline[k] for k in MATCHED_KEYS):
        raise ValueError('same physical, sensor, renderer and execution budget required')
    admission = inputs.admit(launch['input_admission']['nominal_wait_result_sha256'], launch['source_sha256'])
    if admission != launch['input_admission']: raise ValueError('complete original input admission changed')


def prefix_result(result, report):
    if result['decisions'] < prefix.FRAMES:
        return dict(status='REACTIVE_INTERVENTION_NOT_REACHED', decisions=result['decisions'],
            full_raw_audit_retained=True, actual_paired_execution_compared=False, navigation_verified=False)
    prior = inputs.learned_inputs.learned
    return dict(status='COMPLETE_REACTIVE_ACTUAL_PREFIX', actual_paired_execution_compared=True,
        **prefix.compare(prior.OUTPUT/prior.original.CASE[0], OUTPUT/CASE[0], report))


def require_worker(record, audit):
    if (record['status'] != WORKER_STATUS or 'failure' in record
            or (record['case'], record['layout_index']) != CASE
            or record['high_level_world_model_loaded'] is not False
            or record['measured_plane_constrained_estimator'] is not True
            or record['collection']['navigation_ticks'] != 4000
            or record['collection']['status'] != 'REACTIVE_FLOOR_TRANSPORT_MAZE_TERMINAL_AUDIT_REQUIRED'
            or record['collection']['learned_model_used'] is not False
            or record['collection']['candidate_future_outcomes_evaluated'] is not False):
        raise ValueError('complete fixed reactive collection without a high-level model required')
    require_raw_audit(record, audit, learned=False)
    success = bool(audit['native_evaluation']['native_round_trip_candidate_pass']
        and audit['strict_physical_visibility_pass'] and not audit['hard_measurement_failed_frames'])
    if record['verified_round_trip'] is not success:
        raise ValueError('same physical and strict sensing success criteria required')
    receipt = record['prefix_comparison']
    if record['collection']['decisions'] < prefix.FRAMES:
        if receipt != prefix_result(record['collection'], {}) or success:
            raise ValueError('early negative cannot claim an executed intervention or success')
    else:
        expected = dict(status='COMPLETE_REACTIVE_ACTUAL_PREFIX', actual_paired_execution_compared=True,
            common_prefix_frames=4, first_changed_command_frame=3, physical_prefix_samples=900,
            physical_and_public_prefix_exact=True, all_preintervention_requested_commands_exact=True,
            complete_candidate_decisions_match_prospective_prefix=True,
            complete_baseline_decisions_match_prospective_prefix=True,
            baseline_intervention_command=[.16, 0., .45], candidate_intervention_command=[.2, 0., 0.],
            candidate_intervention_command_completed=True, boundary_command_samples_present=50,
            following_physical_outcomes_compared=False, navigation_verified=False, unexecuted_outcomes_inferred=False,
            both_arms_predictive=False, fully_nonpredictive_candidate=True, reactive_is_whole_method_comparison=True,
            isolated_prediction_ranking_ablation=False, future_constraint_gates_matched=False)
        if any(type(receipt.get(k)) is not type(v) or receipt[k] != v for k, v in expected.items()):
            raise ValueError('complete exact physically executed reactive boundary required')


def worker(launch_sha):
    name, index = CASE
    run.cv2.setNumThreads(1); run.cv2.ocl.setUseOpenCL(False)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    start = time.perf_counter(); ids = {}
    record = dict(status='REACTIVE_MEASURED_PLANE_WORKER_FAILED', case=name, layout_index=index,
        high_level_world_model_loaded=False, artifact_sha256={})
    with (OUTPUT/(name+'_worker.log')).open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            run.verify_artifacts(OUTPUT, {'launch.json': launch_sha}); launch = run.read_json(OUTPUT, 'launch.json')
            verify_inputs(launch); resources()
            result = pipeline.collect(index, launch['source_sha256'][PROTOCOL], output=OUTPUT,
                geometry=original.ArticulatedCollisionGeometry(prefix.replay.job.URDF), episode_name=name)
            record['collection'] = result
            ids.update({name+'/'+n: run.digest(OUTPUT/name/n) for n in pipeline.artifacts(index, result)})
            record['artifact_sha256'] = dict(ids); run.verify_artifacts(OUTPUT, ids)
            audit = pipeline.audit(index, result, launch['source_sha256'][PROTOCOL], input_root=OUTPUT,
                robot_geometry=original.ArticulatedCollisionGeometry(prefix.replay.job.URDF), episode_name=name)
            n = name+'_audit.json'; run.write_json(OUTPUT/n, audit); ids[n] = run.digest(OUTPUT/n)
            record['artifact_sha256'] = dict(ids)
            receipt = prefix_result(result, launch['input_admission']['prefix_report'])
            n = name+'_prefix_comparison.json'; run.write_json(OUTPUT/n, receipt); ids[n] = run.digest(OUTPUT/n)
            record['artifact_sha256'] = dict(ids)
            with np.load(OUTPUT/name/'physics_trace.npz', allow_pickle=False) as raw:
                readout = original.case_readout(audit, result, raw['physics_contact'])
            n = name+'_readout.json'; run.write_json(OUTPUT/n, readout); ids[n] = run.digest(OUTPUT/n)
            record.update(status=WORKER_STATUS, artifact_sha256=dict(ids), prefix_comparison=receipt,
                readout=readout, measured_plane_constrained_estimator=True, **{k: audit[k] for k in OUTCOME_KEYS})
            require_worker(record, audit); verify_inputs(launch); run.verify_artifacts(OUTPUT, ids)
        except Exception as error:
            import traceback
            traceback.print_exc(); record.update(status='REACTIVE_MEASURED_PLANE_WORKER_FAILED', failure=repr(error))
    record.update(wall_s=time.perf_counter()-start,
        maximum_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        worker_log_sha256=run.digest(OUTPUT/(name+'_worker.log')))
    run.write_json(OUTPUT/(name+'_worker_terminal.json'), record)
    return record


def main(wait_sha=None, source_only=False, preflight=False):
    if not __debug__: raise ValueError('assertions required')
    run.validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive reactive native attempt; no retry or resume')
    sources = inputs.prepared_sources((SOURCE, PROTOCOL, *TESTS)); hw = resources()
    if source_only:
        print('REACTIVE_MEASURED_PLANE_SOURCE_PREFLIGHT', len(sources), json.dumps(hw), flush=True); return
    if not wait_sha: raise ValueError('actual completed nominal waiter result SHA required')
    admission = inputs.admit(wait_sha, sources); baseline = inputs.learned_inputs.learned_launch()
    launch = {k: baseline[k] for k in ('input_sha256', 'native_sha256', 'native_scene_sha256',
        'native_geometry_sha256', 'opencv_binary_sha256', 'opencv_version', 'rules', 'renderer_environment')}
    launch.update(definition()); launch.update(source_sha256=sources, protocol=PROTOCOL, input_admission=admission)
    verify_inputs(launch); launch['hardware'] = resources()
    if preflight:
        print('REACTIVE_MEASURED_PLANE_PREFLIGHT', len(sources), flush=True); return
    wait_for_idle(); original.require_native_idle(); verify_inputs(launch); launch['hardware'] = resources()
    run.create_output(OUTPUT); process = psutil.Process()
    launch.update(boot_id=run.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        owner=dict(pid=process.pid, created=process.create_time(), command=process.cmdline()), automatic_retry=False)
    run.write_json(OUTPUT/'launch.json', launch); sha = run.digest(OUTPUT/'launch.json')
    print('REACTIVE_MEASURED_PLANE_NATIVE_LAUNCHED', sha, flush=True); start = time.perf_counter()
    try:
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as pool:
                future = pool.submit(worker, sha)
                while True:
                    monitor.write(json.dumps(dict(elapsed_s=time.perf_counter()-start, **run.hardware()))+'\n'); monitor.flush()
                    done, _ = wait([future], timeout=15, return_when=FIRST_COMPLETED)
                    if done: record = future.result(); break
        if record['status'] != WORKER_STATUS:
            raise ValueError('reactive collection or audit failed: '+str(record.get('failure')))
        require_worker(record, run.read_json(OUTPUT, CASE[0]+'_audit.json'))
        ids = dict(record['artifact_sha256'])
        for name in ('launch.json', 'resource_monitor.jsonl', CASE[0]+'_worker.log', CASE[0]+'_worker_terminal.json'):
            ids[name] = run.digest(OUTPUT/name)
        verify_inputs(launch); run.verify_artifacts(OUTPUT, ids)
        run.write_json(OUTPUT/'result.json', dict(status='REACTIVE_MEASURED_PLANE_MAZE02_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, conditions=[record], wall_s=time.perf_counter()-start,
            nominal_wait_result_sha256=wait_sha, learned_result_sha256=admission['learned_result_sha256'],
            nominal_result_sha256=admission['nominal_result_sha256'], reactive_prefix_result_sha256=prefix.RESULT_SHA,
            measured_round_trip_successes=int(record['verified_round_trip']), reused_layout_executions=1,
            new_independent_layout_executions=0, high_level_world_model_loaded=False,
            fully_nonpredictive_controller=True, reactive_is_whole_method_comparison=True,
            isolated_prediction_ranking_ablation=False, automatic_retry=False, model_training=False,
            navigation_qualified=False, real_time_qualified=False, hardware_qualified=False, goal_achieved=False))
        print('REACTIVE_MEASURED_PLANE_NATIVE_COMPLETE', run.digest(OUTPUT/'result.json'), flush=True)
    except BaseException as error:
        run.write_json(OUTPUT/'failure.json', dict(status='TERMINAL_REACTIVE_MEASURED_PLANE_NATIVE_FAILURE',
            reason=repr(error), automatic_retry=False, original_evidence_preserved=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--nominal-wait-result-sha256')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--source-preflight-only', action='store_true'); mode.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(); main(args.nominal_wait_result_sha256, args.source_preflight_only, args.preflight_only)
