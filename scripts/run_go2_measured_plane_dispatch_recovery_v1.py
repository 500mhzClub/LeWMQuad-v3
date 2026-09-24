"""Fresh measured-plane episode after four waiters aborted before dispatch."""
import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from types import FunctionType

import psutil

from scripts import run_go2_measured_plane_maze02_pilot_v1 as original
from scripts import native_waiter_dispatch_abort_development as aborted
from scripts.replay_go2_measured_plane_single_pass_prefix_v1 import cpu_slot
from scripts.run_go2_prepared_native_queue_v1 import competitors
from scripts.startup_source_inventory_development import discover_sources

run = original.run
SOURCE = 'scripts/run_go2_measured_plane_dispatch_recovery_v1.py'
PROTOCOL = 'docs/go2_measured_plane_dispatch_recovery_v1_2026-09-11.md'
TEST = 'lewm/tests/test_measured_plane_dispatch_recovery_development.py'
OUTPUT = run.BASE/'go2_measured_plane_dispatch_recovery_v1_attempt_001'


def fork(function, **replacements):
    if function.__closure__ is not None:
        raise ValueError('closure-free original native function required')
    result = FunctionType(function.__code__, function.__globals__ | replacements,
        function.__name__, function.__defaults__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


definition = fork(original.definition, OUTPUT=OUTPUT)
prefix_result = fork(original.prefix_result, OUTPUT=OUTPUT)


def prepared_sources():
    failed = aborted.admit()
    proof = original.inputs.completed_prefix()
    inherited = dict(failed['source_sha256'])
    for name, sha in proof['source_sha256'].items():
        if name in inherited and inherited[name] != sha: raise ValueError('source ancestry conflict')
        inherited[name] = sha
    sources = discover_sources((SOURCE, PROTOCOL, TEST,
        str(original.prefix.completed.OUTPUT.relative_to(run.ROOT))), inherited)
    run.verify(sources)
    return sources


def admit(sources):
    failed = aborted.admit()
    proof = original.inputs.completed_prefix()
    name = str(original.inputs.job.worker.OUTPUT.relative_to(run.ROOT))
    run.verify({name: original.inputs.job.WORKER_ADMISSION_SHA})
    worker = json.loads(original.inputs.job.worker.OUTPUT.read_text())
    if (worker['status'] != 'EXTENDED_BUDGET_COMPLETED_WORKER_ADMITTED'
            or worker['model_state_sha256'] != original.inputs.job.MODEL_SHA
            or worker['model_state_unchanged'] is not True
            or run.owner_live(worker['original_worker'])):
        raise ValueError('same ended original raw worker and corrected model required')
    run.verify_artifacts(original.old.OUTPUT, worker['artifact_sha256'])
    run.verify(sources)
    failed = {k: v for k, v in failed.items() if k != 'source_sha256'}
    return dict(scheduler_abort=failed, cpu_completion=cpu_slot(),
        controller_completion_sha256=original.prefix.COMPLETION_SHA,
        controller_result_sha256=original.prefix.RESULT_SHA, prefix_report=proof['report'],
        prefix_artifact_sha256=proof['artifact_sha256'], worker_admission_sha256=original.inputs.job.WORKER_ADMISSION_SHA,
        worker_artifact_sha256=worker['artifact_sha256'], model_state_sha256=original.inputs.job.MODEL_SHA,
        original_native_child_retried=False, failed_waiter_restarted=False,
        unexecuted_predecessor_diagnostics_required_for_new_scientific_claim=False,
        full_training_ancestry_reexecuted=False)


def verify_inputs(launch):
    original.require_environment(launch)
    original.old.verify_ordered_launch(launch)
    if any(type(launch[k]) is not type(v) or launch[k] != v for k, v in definition().items()):
        raise ValueError('same measured-plane science with explicit new output root required')
    if admit(launch['source_sha256']) != launch['input_admission']:
        raise ValueError('preserved scheduler failures and original scientific inputs changed')


_worker = fork(original.worker, OUTPUT=OUTPUT, PROTOCOL=PROTOCOL,
    verify_inputs=verify_inputs, prefix_result=prefix_result)


def worker(launch_sha):
    # The public wrapper is picklable by a fresh spawn process. The private
    # implementation retains the original collection, audit and readout code.
    return _worker(launch_sha)


def wait_for_idle():
    # An occupied slot is a wait before output creation, not a terminal native
    # attempt. Keep the conservative existing process predicate unchanged.
    while True:
        pending = competitors()
        if not pending: return
        print('MEASURED_PLANE_DISPATCH_WAITING_FOR_IDLE', json.dumps(pending), flush=True)
        time.sleep(30)


def main(source_only=False, preflight=False):
    if not __debug__: raise ValueError('assertions required')
    run.validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive fresh native attempt; no retry')
    sources = prepared_sources(); hardware = original.resources()
    if source_only:
        print('MEASURED_PLANE_DISPATCH_SOURCE_PREFLIGHT', len(sources), json.dumps(hardware), flush=True)
        return
    admission = admit(sources)
    old = run.read_json(original.old.OUTPUT, 'launch.json')
    launch = {k: old[k] for k in ('input_sha256', 'native_sha256', 'native_scene_sha256',
        'native_geometry_sha256', 'opencv_binary_sha256', 'opencv_version', 'rules', 'renderer_environment')}
    launch.update(definition())
    launch.update(source_sha256=sources, protocol=PROTOCOL, input_admission=admission)
    verify_inputs(launch); original.assigned_model(); launch['hardware'] = original.resources()
    if preflight:
        print('MEASURED_PLANE_DISPATCH_PREFLIGHT', len(sources), flush=True)
        return
    wait_for_idle(); original.require_native_idle(); verify_inputs(launch)
    launch['hardware'] = original.resources()
    run.create_output(OUTPUT); process = psutil.Process()
    launch.update(boot_id=run.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        owner=dict(pid=process.pid, created=process.create_time(), command=process.cmdline()),
        automatic_retry=False, scheduler_recovery=True)
    run.write_json(OUTPUT/'launch.json', launch); sha = run.digest(OUTPUT/'launch.json')
    print('MEASURED_PLANE_DISPATCH_NATIVE_LAUNCHED', sha, flush=True)
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
                    if done: record = future.result(); break
        print('MEASURED_PLANE_DISPATCH_NATIVE_TERMINAL', record['status'], record.get('failure'), flush=True)
        if record['status'] != original.WORKER_STATUS:
            raise ValueError('native collection/audit failed: '+str(record.get('failure')))
        original.require_worker(record, run.read_json(OUTPUT, original.CASE[0]+'_audit.json'))
        ids = dict(record['artifact_sha256'])
        for name in ('launch.json', 'resource_monitor.jsonl', original.CASE[0]+'_worker.log',
                     original.CASE[0]+'_worker_terminal.json'):
            ids[name] = run.digest(OUTPUT/name)
        verify_inputs(launch); run.verify_artifacts(OUTPUT, ids)
        run.write_json(OUTPUT/'result.json', dict(status='MEASURED_PLANE_DISPATCH_RECOVERY_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, conditions=[record], wall_s=time.perf_counter()-start,
            controller_completion_sha256=original.prefix.COMPLETION_SHA,
            scheduler_abort_evidence_sha256=run.fingerprint(admission['scheduler_abort']),
            measured_round_trip_successes=int(record['verified_round_trip']), reused_layout_executions=1,
            new_independent_layout_executions=0, measured_plane_constrained_estimator=True,
            original_failed_waiters_preserved=True, automatic_retry=False, model_training=False,
            navigation_qualified=False, real_time_qualified=False, hardware_qualified=False, goal_achieved=False))
        print('MEASURED_PLANE_DISPATCH_NATIVE_COMPLETE', run.digest(OUTPUT/'result.json'), flush=True)
    except BaseException as error:
        run.write_json(OUTPUT/'failure.json', dict(status='TERMINAL_MEASURED_PLANE_DISPATCH_RECOVERY_FAILURE',
            reason=repr(error), automatic_retry=False, original_failures_preserved=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--source-preflight-only', action='store_true')
    mode.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(); main(args.source_preflight_only, args.preflight_only)
