"""First nominal predictive episode matched to the completed learned episode."""
import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import psutil

from scripts import nominal_measured_plane_native_inputs_development as inputs
from scripts import nominal_measured_plane_maze_development as pipeline
from scripts import nominal_measured_plane_native_prefix_development as prefix
from scripts.run_go2_measured_plane_dispatch_recovery_v1 import fork, wait_for_idle as _original_wait_for_idle
from lewm.independent_reactive_floor_transport_study_development import MATCHED_KEYS, require_raw_audit

original = inputs.learned.original
run = inputs.run
SOURCE = 'scripts/run_go2_nominal_measured_plane_maze02_v1.py'
PROTOCOL = 'docs/go2_nominal_measured_plane_maze02_v1_2026-09-11.md'
TESTS = ('lewm/tests/test_nominal_measured_plane_maze_development.py',
    'lewm/tests/test_nominal_measured_plane_native_prefix_development.py',
    'lewm/tests/test_nominal_measured_plane_native_inputs_development.py',
    'lewm/tests/test_nominal_measured_plane_native_launcher_development.py')
OUTPUT = run.BASE/'go2_nominal_measured_plane_maze02_v1_attempt_001'
CASE = ('no_rgb_nominal_measured_plane_maze_02', *original.CASE[1:])
WORKER_STATUS = 'NOMINAL_MEASURED_PLANE_COLLECTED_AND_RAW_AUDITED'
definition = fork(original.definition, pipeline=pipeline, OUTPUT=OUTPUT, CASE=CASE)
resources = original.resources
assigned_model = original.assigned_model


def wait_for_idle():
    while True:
        try:
            return _original_wait_for_idle()
        except (OSError, psutil.AccessDenied) as error:
            print('NOMINAL_NATIVE_IDLE_OBSERVATION_RETRY', repr(error), flush=True)
            time.sleep(30)


def verify_inputs(launch):
    original.require_environment(launch); original.old.verify_ordered_launch(launch)
    if any(type(launch[k]) is not type(v) or launch[k] != v for k, v in definition().items()):
        raise ValueError('exact nominal prediction definition required')
    baseline = inputs.learned_launch()
    if any(type(launch[k]) is not type(baseline[k]) or launch[k] != baseline[k] for k in MATCHED_KEYS):
        raise ValueError('same learned/nominal environment and physical execution budget required')
    admission = inputs.admit(launch['input_admission']['learned_result_sha256'], launch['source_sha256'])
    if admission != launch['input_admission']: raise ValueError('complete nominal input admission changed')


def prefix_result(result, report):
    if result['decisions'] < prefix.FRAMES:
        return dict(status='NOMINAL_INTERVENTION_NOT_REACHED', decisions=result['decisions'],
            full_raw_audit_retained=True, actual_paired_execution_compared=False, navigation_verified=False)
    return dict(status='COMPLETE_NOMINAL_ACTUAL_PREFIX', actual_paired_execution_compared=True,
        **prefix.compare(inputs.learned.OUTPUT/original.CASE[0], OUTPUT/CASE[0], report))


def require_worker(record, audit):
    if (record['status'] != WORKER_STATUS or 'failure' in record
            or (record['case'], record['layout_index'], record['variant'], record['condition'], record['model_name']) != CASE
            or record['model_state_sha256'] != inputs.job.MODEL_SHA or record['model_state_unchanged'] is not True
            or record['measured_plane_constrained_estimator'] is not True
            or record['collection']['navigation_ticks'] != 4000
            or record['collection']['status'] != 'RESIDUAL_ANCHORED_CONTINUATION_MAZE_TERMINAL_AUDIT_REQUIRED'):
        raise ValueError('complete nominal controller collection and unchanged assigned model required')
    require_raw_audit(record, audit, learned=False)
    if (audit['raw_model_command_replay_pass'] is not False
            or audit['raw_nominal_forecast_command_replay_pass'] is not True
            or type(audit['actual_learned_model_forward_calls']) is not int
            or audit['actual_learned_model_forward_calls'] != 0
            or audit['nominal_predictive_controller'] is not True
            or audit['fully_nonpredictive_controller'] is not False
            or audit['model_state_unchanged'] is not True):
        raise ValueError('complete nominal predictive replay without learned inference required')
    success = bool(audit['native_evaluation']['native_round_trip_candidate_pass']
        and audit['strict_physical_visibility_pass'] and not audit['hard_measurement_failed_frames'])
    if record['verified_round_trip'] is not success:
        raise ValueError('same physical and strict sensing success criteria required')
    receipt = record['prefix_comparison']
    if record['collection']['decisions'] < prefix.FRAMES:
        if receipt != prefix_result(record['collection'], {}) or success:
            raise ValueError('early negative must retain raw audit without an intervention claim')
    else:
        expected = dict(status='COMPLETE_NOMINAL_ACTUAL_PREFIX', actual_paired_execution_compared=True,
            common_prefix_frames=4, first_changed_command_frame=3, physical_prefix_samples=900,
            physical_and_public_prefix_exact=True, all_preintervention_requested_commands_exact=True,
            complete_candidate_decisions_match_prospective_prefix=True,
            complete_baseline_decisions_match_prospective_prefix=True,
            baseline_intervention_command=[.16, 0., .45], candidate_intervention_command=[.2, 0., 0.],
            candidate_intervention_command_completed=True, boundary_command_samples_present=50,
            following_physical_outcomes_compared=False, navigation_verified=False,
            unexecuted_outcomes_inferred=False, both_arms_predictive=True)
        if any(type(receipt.get(k)) is not type(v) or receipt[k] != v for k, v in expected.items()):
            raise ValueError('complete exact nominal physical intervention required')


_worker = fork(original.worker, OUTPUT=OUTPUT, PROTOCOL=PROTOCOL, CASE=CASE, WORKER_STATUS=WORKER_STATUS,
    inputs=inputs, pipeline=pipeline, verify_inputs=verify_inputs, prefix_result=prefix_result,
    require_worker=require_worker, assigned_model=assigned_model)


def worker(launch_sha):
    return _worker(launch_sha)


def main(learned_sha=None, source_only=False, preflight=False):
    if not __debug__: raise ValueError('assertions required')
    run.validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive nominal native attempt; no retry or resume')
    sources = inputs.prepared_sources((SOURCE, PROTOCOL, *TESTS)); hw = resources()
    if source_only:
        print('NOMINAL_MEASURED_PLANE_SOURCE_PREFLIGHT', len(sources), json.dumps(hw), flush=True); return
    if not learned_sha: raise ValueError('actual completed learned native result SHA required')
    admission = inputs.admit(learned_sha, sources); baseline = inputs.learned_launch()
    launch = {k: baseline[k] for k in ('input_sha256', 'native_sha256', 'native_scene_sha256',
        'native_geometry_sha256', 'opencv_binary_sha256', 'opencv_version', 'rules', 'renderer_environment')}
    launch.update(definition()); launch.update(source_sha256=sources, protocol=PROTOCOL, input_admission=admission)
    verify_inputs(launch); assigned_model(); launch['hardware'] = resources()
    if preflight:
        print('NOMINAL_MEASURED_PLANE_PREFLIGHT', len(sources), flush=True); return
    wait_for_idle(); original.require_native_idle(); verify_inputs(launch); launch['hardware'] = resources()
    run.create_output(OUTPUT); process = psutil.Process()
    launch.update(boot_id=run.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        owner=dict(pid=process.pid, created=process.create_time(), command=process.cmdline()), automatic_retry=False)
    run.write_json(OUTPUT/'launch.json', launch); sha = run.digest(OUTPUT/'launch.json')
    print('NOMINAL_MEASURED_PLANE_NATIVE_LAUNCHED', sha, flush=True); start = time.perf_counter()
    try:
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as pool:
                future = pool.submit(worker, sha)
                while True:
                    monitor.write(json.dumps(dict(elapsed_s=time.perf_counter()-start, **run.hardware()))+'\n')
                    monitor.flush(); done, _ = wait([future], timeout=15, return_when=FIRST_COMPLETED)
                    if done: record = future.result(); break
        if record['status'] != WORKER_STATUS:
            raise ValueError('nominal native collection/audit failed: '+str(record.get('failure')))
        require_worker(record, run.read_json(OUTPUT, CASE[0]+'_audit.json'))
        ids = dict(record['artifact_sha256'])
        for name in ('launch.json', 'resource_monitor.jsonl', CASE[0]+'_worker.log', CASE[0]+'_worker_terminal.json'):
            ids[name] = run.digest(OUTPUT/name)
        verify_inputs(launch); run.verify_artifacts(OUTPUT, ids)
        run.write_json(OUTPUT/'result.json', dict(status='NOMINAL_MEASURED_PLANE_MAZE02_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, conditions=[record], wall_s=time.perf_counter()-start,
            learned_result_sha256=learned_sha, forecast_prefix_result_sha256=prefix.RESULT_SHA,
            measured_round_trip_successes=int(record['verified_round_trip']), reused_layout_executions=1,
            new_independent_layout_executions=0, assigned_forecast_source='nominal_requested_twist',
            actual_learned_model_forward_calls=0, both_comparison_arms_predictive=True,
            isolated_planning_on_off_comparison=False, automatic_retry=False, model_training=False,
            navigation_qualified=False, real_time_qualified=False, hardware_qualified=False, goal_achieved=False))
        print('NOMINAL_MEASURED_PLANE_NATIVE_COMPLETE', run.digest(OUTPUT/'result.json'), flush=True)
    except BaseException as error:
        run.write_json(OUTPUT/'failure.json', dict(status='TERMINAL_NOMINAL_MEASURED_PLANE_NATIVE_FAILURE',
            reason=repr(error), automatic_retry=False, original_evidence_preserved=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--learned-result-sha256')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--source-preflight-only', action='store_true'); mode.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(); main(args.learned_result_sha256, args.source_preflight_only, args.preflight_only)
