"""Fresh chained-tracking native episode after its completed controller replay."""
import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from types import SimpleNamespace

import psutil

from scripts import measured_plane_chained_native_inputs_development as inputs
from scripts import measured_plane_chained_native_pipeline_development as pipeline
from scripts import measured_plane_chained_native_result_development as results
from scripts.run_go2_measured_plane_dispatch_recovery_v1 import fork
from lewm.independent_reactive_floor_transport_study_development import MATCHED_KEYS

prefix = results.prefix
learned = prefix.replay.native
original = learned.original
run = inputs.run
SOURCE = 'scripts/run_go2_measured_plane_chained_maze02_v1.py'
PROTOCOL = 'docs/go2_measured_plane_chained_maze02_v1_2026-09-12.md'
TESTS = (results.TEST, inputs.TEST,
    'lewm/tests/test_measured_plane_chained_native_prefix_development.py',
    'lewm/tests/test_measured_plane_chained_native_launcher_development.py')
OUTPUT = run.BASE/'go2_measured_plane_chained_maze02_v1_attempt_001'
CASE = ('no_rgb_direct_measured_plane_chained_maze_02', *original.CASE[1:])
WORKER_STATUS = 'MEASURED_PLANE_CHAINED_MAZE02_COLLECTED_AND_RAW_AUDITED'
worker_inputs = SimpleNamespace(job=prefix.replay.inputs.job)
resources = original.resources
assigned_model = original.assigned_model
_definition = fork(original.definition, pipeline=pipeline, OUTPUT=OUTPUT, CASE=CASE, inputs=worker_inputs)


def definition():
    return _definition() | dict(native_execution_protocol_frozen=True,
        actual_completed_replay_boundary_required=True, single_pass_timing_change_adopted=False)


def verify_inputs(launch):
    original.require_environment(launch)
    original.old.verify_ordered_launch(launch)
    expected = definition()
    if any(type(launch.get(k)) is not type(v) or launch[k] != v for k, v in expected.items()):
        raise ValueError('exact prospective chained native definition required')
    baseline = prefix.replay.inputs.learned_launch()
    if any(type(launch.get(k)) is not type(baseline[k]) or launch[k] != baseline[k] for k in MATCHED_KEYS):
        raise ValueError('same learned scene, public sensing, commands and physical budget required')
    admission = inputs.admit(launch['input_admission']['chained_wait_result_sha256'], launch['source_sha256'])
    if admission != launch['input_admission']:
        raise ValueError('completed chained intervention and predecessor admission changed')


def prefix_result(collection, report):
    return results.prefix_result(collection, report,
        prior=learned.OUTPUT/original.CASE[0], current=OUTPUT/CASE[0])


def require_worker(record, audit):
    launch = run.read_json(OUTPUT, 'launch.json')
    return results.require_worker(record, audit, launch['input_admission']['prefix_report'],
        case=CASE, worker_status=WORKER_STATUS,
        prior=learned.OUTPUT/original.CASE[0], current=OUTPUT/CASE[0])


_worker = fork(original.worker, OUTPUT=OUTPUT, PROTOCOL=PROTOCOL, CASE=CASE,
    WORKER_STATUS=WORKER_STATUS, inputs=worker_inputs, pipeline=pipeline,
    verify_inputs=verify_inputs, prefix_result=prefix_result, require_worker=require_worker,
    assigned_model=assigned_model)


def worker(launch_sha):
    return _worker(launch_sha)


def wait_for_idle():
    while True:
        try:
            return learned.wait_for_idle()
        except (OSError, psutil.AccessDenied) as error:
            print('CHAINED_NATIVE_IDLE_OBSERVATION_RETRY', repr(error), flush=True)
            time.sleep(30)


def main(chained_wait_sha=None, source_only=False, preflight=False):
    if not __debug__: raise ValueError('assertions required')
    run.validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive chained native attempt; no retry, resume or overwrite')
    sources = inputs.prepared_sources((SOURCE, PROTOCOL, *TESTS))
    hardware = resources()
    if source_only:
        print('CHAINED_NATIVE_SOURCE_PREFLIGHT', len(sources), json.dumps(hardware), flush=True)
        return
    if not chained_wait_sha: raise ValueError('actual completed chained waiter result SHA required')
    admission = inputs.admit(chained_wait_sha, sources)
    baseline = prefix.replay.inputs.learned_launch()
    launch = {k:baseline[k] for k in ('input_sha256', 'native_sha256', 'native_scene_sha256',
        'native_geometry_sha256', 'opencv_binary_sha256', 'opencv_version', 'rules', 'renderer_environment')}
    launch.update(definition())
    launch.update(source_sha256=sources, protocol=PROTOCOL, input_admission=admission)
    verify_inputs(launch)
    assigned_model()
    launch['hardware'] = resources()
    if preflight:
        print('CHAINED_NATIVE_PREFLIGHT', len(sources), flush=True)
        return
    wait_for_idle()
    original.require_native_idle()
    verify_inputs(launch)
    launch['hardware'] = resources()
    run.create_output(OUTPUT)
    process = psutil.Process()
    launch.update(boot_id=run.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        owner=dict(pid=process.pid, created=process.create_time(), command=process.cmdline()), automatic_retry=False)
    run.write_json(OUTPUT/'launch.json', launch)
    sha = run.digest(OUTPUT/'launch.json')
    print('CHAINED_NATIVE_LAUNCHED', sha, flush=True)
    start = time.perf_counter()
    try:
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'),
                    max_tasks_per_child=1) as pool:
                future = pool.submit(worker, sha)
                while True:
                    monitor.write(json.dumps(dict(elapsed_s=time.perf_counter()-start, **run.hardware()))+'\n')
                    monitor.flush()
                    done, _ = wait([future], timeout=15, return_when=FIRST_COMPLETED)
                    if done:
                        record = future.result()
                        break
        if record['status'] != WORKER_STATUS:
            raise ValueError('chained native collection or raw audit failed: '+str(record.get('failure')))
        require_worker(record, run.read_json(OUTPUT, CASE[0]+'_audit.json'))
        ids = dict(record['artifact_sha256'])
        for name in ('launch.json', 'resource_monitor.jsonl', CASE[0]+'_worker.log', CASE[0]+'_worker_terminal.json'):
            ids[name] = run.digest(OUTPUT/name)
        verify_inputs(launch)
        run.verify_artifacts(OUTPUT, ids)
        run.write_json(OUTPUT/'result.json', dict(status='MEASURED_PLANE_CHAINED_MAZE02_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, conditions=[record], wall_s=time.perf_counter()-start,
            chained_wait_result_sha256=chained_wait_sha,
            controller_replay_result_sha256=admission['controller_replay_result_sha256'],
            learned_result_sha256=inputs.LEARNED_RESULT_SHA,
            measured_round_trip_successes=int(record['verified_round_trip']), reused_layout_executions=1,
            new_independent_layout_executions=0, measured_plane_constrained_estimator=True,
            chained_anchor_reacquisition_enabled=True, original_bridge_allowance_unchanged=True,
            single_pass_timing_change_adopted=False, automatic_retry=False, model_training=False,
            navigation_qualified=False, real_time_qualified=False, hardware_qualified=False, goal_achieved=False))
        print('CHAINED_NATIVE_COMPLETE', run.digest(OUTPUT/'result.json'), flush=True)
    except BaseException as error:
        run.write_json(OUTPUT/'failure.json', dict(status='TERMINAL_MEASURED_PLANE_CHAINED_NATIVE_FAILURE',
            reason=repr(error), automatic_retry=False, original_failures_preserved=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chained-wait-result-sha256')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--source-preflight-only', action='store_true')
    mode.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    main(args.chained_wait_result_sha256, args.source_preflight_only, args.preflight_only)
