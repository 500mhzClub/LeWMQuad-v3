"""Isolated CPU fits with bounded dispatch, exclusive artifacts and no retries.

Not an experiment launcher. The caller authenticates the complete study, fixes
the roster and initializes each spawned process with its own policy stream.
Only scheduling changes: the existing trainer, schedule runner, snapshot loader
and inference functions are reused without changing their implementation.
"""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from copy import deepcopy
import hashlib
import multiprocessing
import os
import resource
import shutil
import time

import torch

from scripts import run_go2_independent_pulse_matched_study_v1 as reference
from scripts.cumulative_pulse_snapshot_development import validate_binding
from scripts.navigation_artifact_root_development import verify_artifacts

FIT_BUDGET = 192 * 1024**2
FAILURE_ALLOWANCE = 1024**2
PARENT_BUDGET = 1024**3
MAX_WORKERS = 4
_worker = None


def fit_name(seed, variant, condition):
    reference.require(type(seed) is int and seed >= 0 and variant in reference.VARIANTS
        and condition in reference.CONDITIONS, 'exact named fit required')
    return f'seed_{seed}_{variant}_{condition}'


def make_job(seed, variant, condition, schedule, binding, initial_sha256, *, latent_dim):
    job = dict(name=fit_name(seed, variant, condition), seed=seed, variant=variant,
        condition=condition, schedule=deepcopy(schedule), binding=deepcopy(binding),
        initial_sha256=initial_sha256, latent_dim=latent_dim)
    validate_job(job)
    return job


def validate_job(job):
    reference.require(isinstance(job, dict) and set(job) == {'name', 'seed', 'variant',
        'condition', 'schedule', 'binding', 'initial_sha256', 'latent_dim'}, 'exact fit request schema required')
    reference.require(job['name'] == fit_name(job['seed'], job['variant'], job['condition']),
        'fit identity mismatch')
    validate_binding(job['binding'])
    schedule = job['schedule']
    reference.require(schedule['role'] == 'train' and schedule['seed'] == job['seed']
        and schedule['schedule_sha256'] == job['binding']['schedule_sha256']
        and job['binding']['input_variant'] == job['variant']
        and type(schedule['updates']) is int and schedule['updates'] > 0
        and type(job['latent_dim']) is int and job['latent_dim'] > 0
        and type(job['initial_sha256']) is str and len(job['initial_sha256']) == 64
        and all(c in '0123456789abcdef' for c in job['initial_sha256']),
        'bound schedule, treatment, initialization and positive fit budget required')


class FitArtifacts(reference.Artifacts):
    """Disjoint filename ownership and static allowance, not a shared counter.

36 * 192 MiB + 1 GiB parent allowance = 7.75 GiB, below the original
8 GiB total. The final MiB of each fit allowance is reserved for its failure.
These byte bounds do not bound process memory or filesystem metadata.
"""
    def __init__(self, output, prefix):
        super().__init__(output)
        self.prefix = prefix

    def path(self, name):
        reference.require(name.startswith(self.prefix + '_') or name == self.prefix + '.pt',
            'worker cannot write another fit or parent artifact')
        return super().path(name)

    def capacity(self, amount):
        reference.require(type(amount) is int and amount >= 0
            and self.used + amount <= FIT_BUDGET - FAILURE_ALLOWANCE
            and shutil.disk_usage(self.output).free >= reference.RESERVE + amount,
            'per-fit artifact allowance or storage reserve exhausted')

    def failure(self, value):
        raw = reference.encode(value)
        reference.require(len(raw) <= FAILURE_ALLOWANCE, 'bounded failure record required')
        # Reserved terminal evidence must survive ordinary budget exhaustion.
        # Exclusive creation also prevents a duplicate dispatch overwriting it.
        with self.path(self.prefix + '_failure.json').open('xb') as target:
            target.write(raw); target.flush(); os.fsync(target.fileno())


def execute_fit(job, stream, output, verify):
    """One fresh fit, with all records durable before returning its receipt."""
    validate_job(job)
    reference.require(torch.get_num_threads() == 1 and torch.are_deterministic_algorithms_enabled(),
        'single-thread deterministic worker required')
    files = FitArtifacts(output, job['name']); trainer = None; ledger_path = None
    started = time.perf_counter(); name = job['name']
    cpu_started = time.process_time()
    # Fail duplicate requests before entering failure handling for the owner.
    for suffix in ('_request.json', '_complete.json', '_failure.json'):
        files.path(name + suffix)
    try:
        verify()
        trainer = reference.CumulativePulseTrainer(job['condition'], seed=job['seed'], latent_dim=job['latent_dim'])
        reference.require(trainer.initial_sha256 == job['initial_sha256'], 'same seeded initial model required')
        files.save(name + '_request.json', dict(binding=job['binding'], config=reference.config(trainer),
            planned_updates=job['schedule']['updates'], initial_sha256=trainer.initial_sha256))
        ledger_name = name + '_updates.jsonl'; ledger_path = files.path(ledger_name)
        ledger_hash = hashlib.sha256(); ledger_bytes = 0; count = 0
        with ledger_path.open('xb') as ledger:
            def on_update(record):
                nonlocal ledger_bytes, count
                reference.require(record['input_variant'] == job['variant']
                    and record['schedule_sha256'] == job['binding']['schedule_sha256']
                    and record['update'] == count + 1, 'exact treatment and optimizer accounting required')
                raw = reference.encode(record | dict(elapsed_seconds=time.perf_counter() - started))
                reference.require(ledger_bytes + len(raw) <= reference.MAX_METADATA, 'bounded per-fit update ledger required')
                files.capacity(len(raw)); ledger.write(raw); ledger.flush(); os.fsync(ledger.fileno())
                files.used += len(raw); ledger_bytes += len(raw); ledger_hash.update(raw); count += 1
                if count == 1 or count % 25 == 0:
                    print('PARALLEL_MATCHED_UPDATE', name, count, flush=True)
            fit = reference.train_schedule(trainer, stream, job['schedule'], on_update=on_update,
                input_variant=job['variant'])
        reference.require(count == job['schedule']['updates'] and ledger_path.stat().st_size == ledger_bytes,
            'complete exact update ledger required')
        files.hashes[ledger_name] = ledger_hash.hexdigest()
        verify_artifacts(output, {ledger_name: files.hashes[ledger_name]})
        snapshot, clone = files.snapshot(name + '.pt', trainer, job['binding'])
        reference.require(fit['input_variant'] == snapshot['binding']['input_variant'] == job['variant'],
            'fit/snapshot/input treatment identity mismatch')
        files.save(name + '_fit.json', dict(fit=fit, snapshot=snapshot,
            elapsed_seconds=time.perf_counter() - started, ledger_sha256=files.hashes[ledger_name]))
        for role in reference.ROLES:
            prediction = reference.predict_heads(clone, stream, role=role,
                batch_size=job['schedule']['batch_size'], input_variant=job['variant'])
            reference.require(prediction['input_variant'] == job['variant']
                and prediction['model_sha256'] == fit['model_sha256'] and prediction['updates'] == count,
                'score only exact verified final snapshot and treatment')
            files.save_predictions(name + '_' + role + '.npz', prediction)
            files.save(name + '_' + role + '_prediction.json',
                {k: v for k, v in prediction.items() if k != 'heads'})
        verify(); verify_artifacts(output, files.hashes)
        terminal = dict(status='PARALLEL_MATCHED_FIT_COMPLETE', name=name, job_sha256=reference.identity(job),
            initial_sha256=trainer.initial_sha256, model_sha256=fit['model_sha256'], updates=count,
            output_sha256=dict(files.hashes), artifact_bytes_before_terminal=files.used,
            worker_pid=os.getpid(), elapsed_seconds=time.perf_counter() - started,
            worker_cpu_seconds=time.process_time() - cpu_started,
            process_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            cpu_threads=torch.get_num_threads(), deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            retry_performed=False, navigation_qualified=False, goal_achieved=False)
        files.save(name + '_complete.json', terminal)
        return dict(name=name, complete_sha256=files.hashes[name + '_complete.json'],
            output_sha256=dict(files.hashes), artifact_bytes=files.used, worker_pid=os.getpid())
    except BaseException as error:
        files.failure(dict(status='TERMINAL_PARALLEL_MATCHED_FIT_FAILURE', name=name, reason=repr(error)[:8192],
            actual_optimizer_updates=trainer.updates if trainer is not None else None,
            partial_ledger=ledger_path.name if ledger_path is not None else None,
            output_sha256=dict(files.hashes), retry_performed=False, goal_achieved=False))
        raise


def initialize_worker(stream_factory, stream_argument, output, verifier, verifier_arguments):
    global _worker
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    _worker = (stream_factory(stream_argument), output, verifier, verifier_arguments)


def run_worker(job):
    reference.require(_worker is not None, 'explicit authenticated worker initialization required')
    stream, output, verifier, arguments = _worker
    return execute_fit(job, stream, output, lambda: verifier(*arguments))


def dispatch(jobs, *, initializer=initialize_worker, initargs, on_result, workers=MAX_WORKERS):
    """At most N submitted fits, spawn (never fork), no automatic resubmission.

On an observed failure, no further fits are submitted. Already running fits
are drained and their artifacts retained; this is not a fresh attempt/retry.
The caller must not publish a complete comparison if any future fails.
"""
    reference.require(type(workers) is int and 1 <= workers <= MAX_WORKERS and callable(on_result),
        'bounded workers and explicit result consumer required')
    jobs = list(jobs)
    for job in jobs: validate_job(job)
    reference.require(jobs and len({j['name'] for j in jobs}) == len(jobs), 'nonempty unique fit roster required')
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
        initializer=initializer, initargs=initargs)
    active = {}; completed = {}; next_index = 0
    try:
        while next_index < len(jobs) or active:
            while next_index < len(jobs) and len(active) < workers:
                job = jobs[next_index]; active[pool.submit(run_worker, job)] = job['name']; next_index += 1
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            # Inspect the entire returned group before submitting replacements.
            # A completion callback failure also stops subsequent dispatch.
            results = [(future, future.result()) for future in done]
            for future, receipt in results:
                name = active.pop(future)
                reference.require(receipt['name'] == name and name not in completed, 'worker result identity mismatch')
                on_result(receipt); completed[name] = receipt
        return [completed[j['name']] for j in jobs]
    finally:
        for future in active: future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
