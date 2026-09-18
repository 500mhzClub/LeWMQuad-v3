"""Exclusive full-cache CPU benchmark and eighteen fresh expanded-data fits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from lewm.all_phase_training_schedule_development import schedule as build_schedule, SEEDS
from scripts.all_phase_study_inputs_development import authenticate, stream, CORRECTION
from scripts.all_phase_fit_execution_development import (BENCH, OUTPUT, PROTOCOL, SOURCE, TEST,
    ROSTER, RESERVE, OUTPUT_ALLOWANCE, WORKER_RAM, NATIVE_RAM, PARENT_RAM,
    science, make_request, benchmark_decision)
from scripts.all_phase_fit_worker_development import worker
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json, verify as verify_sources
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

ENVIRONMENT_FIELDS = ('input_sha256', 'native_sha256', 'native_geometry_sha256',
    'native_scene_sha256', 'opencv_binary_sha256', 'opencv_version', 'rules')


def capacity(resources, workers):
    if (resources['memory_available_bytes'] < workers*WORKER_RAM+NATIVE_RAM+PARENT_RAM
            or resources['artifact_free_bytes'] < RESERVE+OUTPUT_ALLOWANCE):
        raise ValueError('measured training-worker/native/parent memory and artifact headroom required')


def verify_owner_sources():
    sources = {}
    for name in ('go2_prepared_native_queue_v1_attempt_001', 'go2_supervised_commitment_contact_native_wait_v1_attempt_001'):
        for path, sha in read_json(BASE/name, 'launch.json')['source_sha256'].items():
            if path in sources and sources[path] != sha: raise ValueError('original native owners disagree on source identity')
            sources[path] = sha
    verify_sources(sources)
    return sources


def dispatch(output, phase, workers, launch_sha, monitor):
    slots = list(range(workers if phase == 'fits' else 3)); records = []; files = []; started = time.perf_counter()
    for at in range(0, len(slots), workers):
        group = slots[at:at+workers]; capacity(hardware(), len(group)); children = []
        try:
            for slot in group:
                request = make_request(phase, slot, workers, launch_sha); name = request['name']
                request_name = name+'_request.json'; write_json(output/request_name, request)
                log_name = name+'_worker.log'; log = (output/log_name).open('xb')
                env = os.environ.copy(); env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
                    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
                process = subprocess.Popen([sys.executable, SOURCE, '--phase', 'fits' if phase=='fits' else 'benchmark',
                    '--worker-request', request_name, '--worker-request-sha256', digest(output/request_name)],
                    stdout=log, stderr=subprocess.STDOUT, env=env)
                children.append((name, process, log)); files.extend([request_name, log_name, name+'_terminal.json'])
                print('ALL_PHASE_SUBPROCESS', name, process.pid, flush=True)
            last_monitor = 0.
            while any(p.poll() is None for _,p,_ in children):
                if time.monotonic()-last_monitor >= 15:
                    monitor.write(json.dumps(dict(phase=phase, elapsed_s=time.perf_counter()-started,
                        children=[dict(name=n, pid=p.pid, returncode=p.poll()) for n,p,_ in children], **hardware()))+'\n')
                    monitor.flush(); last_monitor = time.monotonic()
                time.sleep(1)
            batch = []
            for name, process, log in children:
                log.close(); terminal = read_json(output, name+'_terminal.json')
                if terminal['name'] != name or terminal['worker_pid'] != process.pid:
                    raise ValueError('exact original subprocess terminal identity required')
                verify_artifacts(output, terminal['artifact_sha256'])
                if process.returncode != 0 or terminal['status'] != 'ALL_PHASE_FIT_WORKER_COMPLETE':
                    raise ValueError('worker failed; preserve partial ledgers and do not retry: '+name)
                batch.append(terminal)
                print('ALL_PHASE_PARENT_WORKER_COMPLETE', name, terminal['actual_updates'], flush=True)
            records.extend(batch)
        finally:
            # Never abandon or replace a launched worker on a parent-side error.
            for _, process, log in children:
                if process.poll() is None: process.wait()
                if not log.closed: log.close()
    return dict(wall_s=time.perf_counter()-started, records=sorted(records, key=lambda r:r['name'])), files


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=('benchmark', 'fits'), required=True)
    parser.add_argument('--benchmark-result-sha256'); parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--worker-request'); parser.add_argument('--worker-request-sha256'); args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    if args.worker_request:
        if args.preflight_only or args.benchmark_result_sha256 or not args.worker_request_sha256:
            raise ValueError('exact direct-subprocess invocation required')
        return worker(phase=args.phase, request_name=args.worker_request, request_sha256=args.worker_request_sha256)
    if args.worker_request_sha256: raise ValueError('worker receipt requires an assigned request')
    output = BENCH if args.phase=='benchmark' else OUTPUT
    validate_root(output, must_exist=False)
    if output.exists() or output.is_symlink(): raise ValueError('exclusive phase; no retry/resume')
    initial_hardware = hardware(); capacity(initial_hardware, 3 if args.phase=='benchmark' else 1)
    print('ALL_PHASE_PHASE_AUTHENTICATING', args.phase, flush=True)
    definition, _, _ = authenticate(); owner_sources = verify_owner_sources()
    data = stream(maximum_cache_bytes=0)
    schedules = {str(seed):build_schedule(data.view, seed=seed) for seed in SEEDS}; del data
    sources = discover_sources((PROTOCOL, SOURCE, TEST, str(CORRECTION),
        'lewm/tests/test_all_phase_training_fit_development.py',
        'docs/go2_all_phase_training_schedules_preparation_2026-09-10.json',
        'docs/go2_all_phase_study_stream_preparation_result_2026-09-10.json'), definition['source_sha256'])
    scientific = science(schedules); workers = 3; benchmark_ids = None
    if args.phase == 'fits':
        if not args.benchmark_result_sha256: raise ValueError('exact successful full-cache benchmark required')
        verify_artifacts(BENCH, {'result.json':args.benchmark_result_sha256})
        bench = read_json(BENCH, 'result.json'); verify_artifacts(BENCH, bench['artifact_sha256'])
        verify_ordered_launch(read_json(BENCH, 'launch.json'))
        if (bench['status'] != 'ALL_PHASE_FIT_BENCHMARK_COMPLETE' or bench['science'] != scientific
                or bench['source_sha256'] != sources or bench['decision'] != benchmark_decision(bench['serial'], bench['parallel'])):
            raise ValueError('exact completed benchmark/source/science/decision required')
        workers = bench['decision']['selected_workers']
        benchmark_ids = bench['artifact_sha256'] | {'result.json':args.benchmark_result_sha256}
    schedule_bytes = (json.dumps(schedules, indent=2, allow_nan=False)+'\n').encode()
    launch = {k:definition[k] for k in ENVIRONMENT_FIELDS}
    launch.update(source_sha256=sources, protocol=PROTOCOL, science=scientific, output_root=str(output),
        phase=args.phase, selected_workers=workers, hardware=hardware(),
        training_schedules_file_sha256=hashlib.sha256(schedule_bytes).hexdigest(),
        input_scope_correction_sha256=scientific['input_identity']['scope_correction'],
        private_training_future_materialization=True, future_rgb_materialization=True,
        geometry_transfer_future_materialization=False, inference_future_materialization=False,
        training_worker_ram_bytes=WORKER_RAM, concurrent_native_ram_bytes=NATIVE_RAM,
        parent_ram_bytes=PARENT_RAM, full_cache_reused_across_fresh_models=True,
        original_native_owner_source_sha256=owner_sources, native_scene_workers=0,
        benchmark_result_sha256=args.benchmark_result_sha256, benchmark_artifact_sha256=benchmark_ids,
        native_execution=False, navigation_qualified=False, real_time_qualified=False)
    capacity(launch['hardware'], workers); verify_ordered_launch(launch)
    if args.preflight_only:
        print('ALL_PHASE_FIT_PREFLIGHT_PASS', json.dumps(dict(phase=args.phase, sources=len(sources),
            workers=workers, science=scientific, hardware=launch['hardware']), sort_keys=True), flush=True)
        return 0
    create_output(output); write_json(output/'training_schedules.json', schedules); write_json(output/'launch.json', launch)
    verify_artifacts(output, {'training_schedules.json':launch['training_schedules_file_sha256']})
    launch_sha = digest(output/'launch.json'); print('ALL_PHASE_PHASE_LAUNCHED', args.phase, launch_sha, flush=True)
    try:
        files = ['launch.json', 'training_schedules.json', 'resource_monitor.jsonl']
        with (output/'resource_monitor.jsonl').open('x') as monitor:
            if args.phase == 'benchmark':
                phases = {}
                for phase, count in (('serial', 1), ('parallel', 3)):
                    phases[phase], created = dispatch(output, phase, count, launch_sha, monitor); files.extend(created)
                    phase_file = phase+'.json'; write_json(output/phase_file, phases[phase]); files.append(phase_file)
                decision = benchmark_decision(phases['serial'], phases['parallel'])
                result = dict(status='ALL_PHASE_FIT_BENCHMARK_COMPLETE', **phases, decision=decision)
                records = phases['serial']['records']+phases['parallel']['records']
            else:
                phase, created = dispatch(output, 'fits', workers, launch_sha, monitor); files.extend(created)
                records = phase['records']; jobs = [j for r in records for j in r['jobs']]
                if (sorted(j['name'] for j in jobs) != sorted(r['name'] for r in ROSTER)
                        or sum(j['actual_updates'] for j in jobs) != 21600):
                    raise ValueError('all eighteen exact fresh fits and21600 optimizer updates required')
                for seed in SEEDS:
                    paired = [j['fit'] for j in jobs if j['fit']['seed']==seed]
                    if len(paired)!=6 or len({j['initial_sha256'] for j in paired})!=1:
                        raise ValueError('six identical fresh initializations per seed required')
                result = dict(status='ALL_PHASE_EIGHTEEN_FITS_COMPLETE', **phase, selected_workers=workers)
        authenticate(); verify_ordered_launch(launch); verify_sources(owner_sources)
        if benchmark_ids is not None: verify_artifacts(BENCH, benchmark_ids)
        bindings = {n:h for r in records for n,h in r['artifact_sha256'].items()}
        bindings.update({n:digest(output/n) for n in files}); verify_artifacts(output, bindings)
        if sum((output/n).stat().st_size for n in bindings) > OUTPUT_ALLOWANCE:
            raise ValueError('phase artifact allowance exceeded; evidence retained')
        result.update(science=scientific, source_sha256=sources, artifact_sha256=bindings,
            optimizer_updates=sum(r['actual_updates'] for r in records),
            scientific_models_trained=0 if args.phase=='benchmark' else 18,
            benchmark_weights_reused=False, checkpoint_selection_performed=False,
            native_execution=False, navigation_qualified=False, goal_achieved=False, hardware_after=hardware())
        write_json(output/'result.json', result)
        print('ALL_PHASE_PHASE_COMPLETE', args.phase, digest(output/'result.json'), flush=True)
        return 0
    except Exception as error:
        write_json(output/'failure.json', dict(status='TERMINAL_ALL_PHASE_FIT_PHASE_FAILURE', phase=args.phase, reason=repr(error)))
        raise


if __name__ == '__main__': sys.exit(main())
