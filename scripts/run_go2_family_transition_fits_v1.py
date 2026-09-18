"""Separate fitting benchmark and six fresh matched transition-model fits."""
import argparse
import contextlib
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import resource
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, wait
import cv2
import numpy as np
import torch

from lewm.cumulative_pulse_learning_development import CumulativePulseTrainer
from lewm.family_transition_fit_development import train, predict, score
from scripts.family_transition_fit_inputs_development import authenticate, stream, CHECK, CHECK_SHA
from scripts.cumulative_pulse_snapshot_development import save_snapshot, load_snapshot
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

BENCH = BASE/'go2_family_transition_fit_benchmark_v1_attempt_001'
OUTPUT = BASE/'go2_family_transition_fits_v1_attempt_001'
PROTOCOL = 'docs/go2_family_transition_fits_v1_2026-09-08.md'
VARIANTS = ('full', 'no_rgb')
CONDITIONS = ('direct', 'supervised_rollout', 'jepa')
ROSTER = tuple(f'seed_2026091001_{v}_{c}' for v in VARIANTS for c in CONDITIONS)
RESERVE = 40*1024**3


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def worker(request):
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    output = Path(request['output']); validate_root(output)
    name = request['name']; benchmark = request['benchmark']
    expected = (f"{request['phase']}_{request['case']}" if benchmark else
        f"seed_{request['seed']}_{request['variant']}_{request['condition']}")
    if name != expected or output != (BENCH if benchmark else OUTPUT):
        raise ValueError('exact assigned worker/root required')
    terminal = dict(name=name, status='FAMILY_TRANSITION_WORKER_FAILED', actual_updates=0, artifact_sha256={})
    started = time.perf_counter(); trainer = None
    with (output/(name+'_worker.log')).open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            verify_artifacts(output, {'launch.json': request['launch_sha256']})
            launch = read_json(output, 'launch.json'); verify(launch)
            if request['science'] != launch['science']:
                raise ValueError('worker scientific definition mismatch')
            _, _, schedule = authenticate()
            dataset = stream()
            if not benchmark and (name not in ROSTER or request['seed'] != 2026091001):
                raise ValueError('exact six-fit scientific roster required')
            if benchmark and (request['phase'] not in ('serial', 'parallel') or request['case'] not in range(4)
                    or request['seed'] != 2026091010+request['case'] or request['variant'] != 'full' or request['condition'] != 'jepa'):
                raise ValueError('exact separate benchmark case required')
            write_json(output/(name+'_request.json'), request)
            trainer = CumulativePulseTrainer(request['condition'], seed=request['seed'], latent_dim=32)
            ledger_name = name+'_updates.jsonl'; ledger_content = hashlib.sha256(); count = 0
            with (output/ledger_name).open('xb') as ledger:
                def record(row):
                    nonlocal count
                    count += 1
                    if row['update'] != count or shutil.disk_usage(BASE).free < RESERVE:
                        raise ValueError('exact ledger accounting/storage reserve required')
                    raw = canonical(row)+b'\n'; ledger.write(raw); ledger.flush(); os.fsync(ledger.fileno()); ledger_content.update(raw)
                    if count == 1 or count % 100 == 0:
                        print('FAMILY_TRANSITION_UPDATE', name, count, flush=True)
                fit = train(trainer, dataset, schedule, input_variant=request['variant'], on_update=record, benchmark=benchmark)
            if count != (20 if benchmark else 1200) or digest(output/ledger_name) != ledger_content.hexdigest():
                raise ValueError('complete durable update ledger required')
            artifacts = [name+'_request.json', ledger_name]
            if not benchmark:
                binding = dict(experiment_sha256=request['launch_sha256'], dataset_sha256=CHECK_SHA,
                    schedule_sha256=schedule['schedule_sha256'], input_variant=request['variant'])
                snapshot = save_snapshot(output, name+'.pt', trainer, binding)
                clone = load_snapshot(output, name+'.pt', sha256=snapshot['sha256'],
                    expected_binding=binding, expected_config=snapshot['configuration'])
                write_json(output/(name+'_fit.json'), dict(fit=fit, snapshot=snapshot, ledger_sha256=ledger_content.hexdigest()))
                artifacts += [name+'.pt', name+'_fit.json']
                for role in ('train', 'geometry_transfer'):
                    arrays = predict(clone, dataset, role=role, input_variant=request['variant'])
                    file = name+'_'+role+'.npz'
                    with (output/file).open('xb') as target:
                        np.savez_compressed(target, **arrays)
                    head = 'direct_outcomes' if request['condition'] == 'direct' else 'rollout_outcomes'
                    scores = score(dataset.view, arrays, role=role, head=head)
                    with np.load(output/file, allow_pickle=False) as saved:
                        replay = {k: saved[k] for k in saved.files}
                    if score(dataset.view, replay, role=role, head=head) != scores:
                        raise ValueError('saved raw predictions must reproduce scores exactly')
                    scored = name+'_'+role+'_scores.json'; write_json(output/scored, scores)
                    artifacts += [file, scored]
            else:
                write_json(output/(name+'_benchmark_fit.json'), fit); artifacts += [name+'_benchmark_fit.json']
            authenticate(); verify(launch)
            bindings = {n: digest(output/n) for n in artifacts}; verify_artifacts(output, bindings)
            terminal.update(status='FAMILY_TRANSITION_WORKER_COMPLETE', actual_updates=count,
                fit=fit, ledger_sha256=ledger_content.hexdigest(), artifact_sha256=bindings,
                artifact_bytes=sum((output/n).stat().st_size for n in artifacts),
                benchmark_weights_reused=False, navigation_qualified=False)
        except Exception as error:
            import traceback
            traceback.print_exc(); terminal['failure'] = repr(error)
            terminal['actual_updates'] = trainer.updates if trainer is not None else 0
    terminal.update(wall_s=time.perf_counter()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        worker_pid=os.getpid(), worker_log_sha256=digest(output/(name+'_worker.log')))
    write_json(output/(name+'_terminal.json'), terminal)
    return terminal


def dispatch(requests, *, workers, monitor):
    records = []; started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as pool:
        for at in range(0, len(requests), workers):
            pending = {pool.submit(worker, r): r['name'] for r in requests[at:at+workers]}
            batch = []
            while pending:
                monitor.write(json.dumps(dict(elapsed_s=time.perf_counter()-started, **hardware()))+'\n'); monitor.flush()
                done, _ = wait(pending, timeout=15)
                for future in done:
                    name = pending.pop(future); result = future.result(); batch.append(result)
                    print('FAMILY_TRANSITION_TERMINAL', name, result['status'], result.get('failure'), flush=True)
            records += batch
            if any(r['status'] != 'FAMILY_TRANSITION_WORKER_COMPLETE' for r in batch):
                raise ValueError('worker failed; later batches unlaunched, running siblings retained')
    return dict(wall_s=time.perf_counter()-started, records=sorted(records, key=lambda r: r['name']))


def benchmark_decision(serial, parallel):
    a = serial['records']; b = parallel['records']
    if len(a) != 4 or len(b) != 4:
        raise ValueError('four complete benchmark cases per execution mode required')
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        if (x['name'] != f'serial_{i}' or y['name'] != f'parallel_{i}'
                or x['status'] != y['status'] or x['status'] != 'FAMILY_TRANSITION_WORKER_COMPLETE'
                or x['actual_updates'] != y['actual_updates'] or x['actual_updates'] != 20
                or x['fit'] != y['fit'] or x['ledger_sha256'] != y['ledger_sha256']):
            raise ValueError('every benchmark update and final model must match exactly')
    if any(not math.isfinite(p['wall_s']) or p['wall_s'] <= 0 for p in (serial, parallel)):
        raise ValueError('positive finite measured phase durations required')
    speedup = serial['wall_s']/parallel['wall_s']
    memory_ok = max(r['peak_rss_bytes'] for r in b) <= 8*1024**3
    return dict(measured_speedup=speedup, exact_update_and_model_equality=True,
        parallel_memory_allowance_pass=memory_ok, selected_workers=4 if speedup >= 1.25 and memory_ok else 1)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=('benchmark', 'fits'), required=True)
    parser.add_argument('--benchmark-result-sha256'); args = parser.parse_args()
    if not __debug__: raise ValueError('audit assertions required')
    output = BENCH if args.phase == 'benchmark' else OUTPUT
    validate_root(output, must_exist=False)
    if output.exists() or output.is_symlink(): raise ValueError('exclusive phase; no retry/resume')
    definition, checked, _ = authenticate()
    sources = discover_sources((PROTOCOL, 'scripts/run_go2_family_transition_fits_v1.py',
        'lewm/tests/test_family_transition_fit_development.py'), definition['source_sha256'])
    resources = hardware()
    if resources['memory_available_bytes'] < 40*1024**3 or resources['artifact_free_bytes'] < RESERVE+2*1024**3:
        raise ValueError('40GiB RAM and2GiB output plus storage reserve required')
    science = dict(seed=2026091001, variants=list(VARIANTS), conditions=list(CONDITIONS), updates=1200,
        batch_size=6, latent_dim=32, learning_rate=.001, ema_momentum=.99, input_check_sha256=CHECK_SHA,
        benchmark_seeds=list(range(2026091010, 2026091014)), benchmark_updates=20,
        primary_native_candidate='seed_2026091001_full_jepa', checkpoint_selection_performed=False)
    launch = definition | dict(source_sha256=sources, protocol=PROTOCOL, hardware=resources,
        output_root=str(output), phase=args.phase, science=science, input_check_result_sha256=CHECK_SHA,
        physics_paused_during_compute=None, native_execution=False, real_time_qualified=False)
    workers = None
    if args.phase == 'fits':
        if args.benchmark_result_sha256 is None: raise ValueError('exact completed fitting benchmark required')
        verify_artifacts(BENCH, {'result.json': args.benchmark_result_sha256})
        bench = read_json(BENCH, 'result.json'); verify_artifacts(BENCH, bench['artifact_sha256'])
        if bench['status'] != 'FAMILY_TRANSITION_FIT_BENCHMARK_COMPLETE' or bench['science'] != science:
            raise ValueError('unchanged scientific settings required')
        if benchmark_decision(bench['serial'], bench['parallel']) != bench['decision']:
            raise ValueError('benchmark decision must reproduce')
        if bench['source_sha256'] != sources: raise ValueError('fit sources must equal benchmarked sources')
        workers = bench['decision']['selected_workers']; launch['benchmark_result_sha256'] = args.benchmark_result_sha256
        launch['selected_workers'] = workers
    verify(launch); create_output(output); write_json(output/'launch.json', launch)
    launch_sha = digest(output/'launch.json'); print('FAMILY_TRANSITION_PHASE_LAUNCHED', args.phase, launch_sha, flush=True)
    def request(name, *, seed, variant, condition, phase=None, case=None):
        return dict(name=name, output=str(output), launch_sha256=launch_sha, science=science,
            seed=seed, variant=variant, condition=condition, benchmark=args.phase=='benchmark', phase=phase, case=case)
    try:
        with (output/'resource_monitor.jsonl').open('x') as monitor:
            if args.phase == 'benchmark':
                phases = {}
                for phase, n in (('serial', 1), ('parallel', 4)):
                    jobs = [request(f'{phase}_{i}', seed=2026091010+i, variant='full', condition='jepa', phase=phase, case=i) for i in range(4)]
                    phases[phase] = dispatch(jobs, workers=n, monitor=monitor)
                decision = benchmark_decision(phases['serial'], phases['parallel'])
                result = dict(status='FAMILY_TRANSITION_FIT_BENCHMARK_COMPLETE', **phases, decision=decision)
                records = phases['serial']['records']+phases['parallel']['records']
            else:
                jobs = [request(f'seed_2026091001_{v}_{c}', seed=2026091001, variant=v, condition=c) for v in VARIANTS for c in CONDITIONS]
                phase = dispatch(jobs, workers=workers, monitor=monitor); records = phase['records']
                if set(r['name'] for r in records) != set(ROSTER) or sum(r['actual_updates'] for r in records) != 7200:
                    raise ValueError('all six exact fresh fits required')
                if len({r['fit']['initial_sha256'] for r in records}) != 1:
                    raise ValueError('matched common initialization required')
                result = dict(status='FAMILY_TRANSITION_SIX_FITS_COMPLETE', **phase, selected_workers=workers, optimizer_updates=7200)
        bindings = {n: h for r in records for n, h in r['artifact_sha256'].items()}
        for name in ('launch.json', 'resource_monitor.jsonl', *(r['name']+s for r in records for s in ('_worker.log', '_terminal.json'))):
            bindings[name] = digest(output/name)
        authenticate(); verify(launch); verify_artifacts(output, bindings)
        result.update(science=science, source_sha256=sources, artifact_sha256=bindings,
            benchmark_weights_reused=False, checkpoint_selection_performed=False, navigation_qualified=False, goal_achieved=False)
        write_json(output/'result.json', result); print('FAMILY_TRANSITION_PHASE_COMPLETE', args.phase, digest(output/'result.json'), flush=True)
    except Exception as error:
        write_json(output/'failure.json', dict(status='TERMINAL_FAMILY_TRANSITION_PHASE_FAILURE', phase=args.phase, reason=repr(error)))
        raise


if __name__ == '__main__': main()
