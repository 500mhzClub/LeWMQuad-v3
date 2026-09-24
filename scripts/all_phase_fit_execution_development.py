"""Fixed fit assignments and exact serial/parallel benchmark comparison."""
import hashlib
import json
import math
from lewm.all_phase_training_schedule_development import SEEDS
from scripts.all_phase_study_inputs_development import CHECK_SHA, CORRECTION_SHA, TARGET_SHA
from scripts.navigation_artifact_root_development import BASE

BENCH = BASE/'go2_all_phase_fit_benchmark_v1_attempt_001'
OUTPUT = BASE/'go2_all_phase_matched_fits_v1_attempt_001'
PROTOCOL = 'docs/go2_all_phase_matched_fits_v1_2026-09-10.md'
SOURCE = 'scripts/fit_go2_all_phase_models_v1.py'
TEST = 'lewm/tests/test_all_phase_fit_execution_development.py'
VARIANTS = ('full', 'no_rgb')
CONDITIONS = ('direct', 'supervised_rollout', 'jepa')
BENCH_SEEDS = tuple(range(2026091810, 2026091816))
ROSTER = tuple(dict(name=f'seed_{seed}_{variant}_{condition}', seed=seed,
    variant=variant, condition=condition) for seed in SEEDS for variant in VARIANTS for condition in CONDITIONS)
RESERVE = 40*1024**3
OUTPUT_ALLOWANCE = 2*1024**3
WORKER_RAM = 10*1024**3
NATIVE_RAM = 32*1024**3
PARENT_RAM = 2*1024**3
CACHE_BYTES = 7_359_601_120


def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def science(schedules):
    pair = dict(targets=TARGET_SHA, inputs=CHECK_SHA, scope_correction=CORRECTION_SHA)
    return dict(optimization_seeds=list(SEEDS), variants=list(VARIANTS), conditions=list(CONDITIONS),
        updates=1200, batch_size=6, latent_dim=32, learning_rate=.001, ema_momentum=.99,
        device='cpu', dtype='float32', input_identity=pair,
        dataset_pair_sha256=hashlib.sha256(canonical(pair)).hexdigest(),
        schedule_sha256={str(seed): schedules[str(seed)]['schedule_sha256'] for seed in SEEDS},
        benchmark_seeds=list(BENCH_SEEDS), benchmark_updates=20, benchmark_schedule_seed=SEEDS[0],
        benchmark_cases=6, worker_full_training_cache_bytes=CACHE_BYTES,
        primary_native_candidate='seed_2026091001_full_jepa',
        matched_native_training_controls=['seed_2026091001_full_supervised_rollout', 'seed_2026091001_full_direct'],
        sensor_ablation_controls=[r['name'] for r in ROSTER if r['seed']==SEEDS[0] and r['variant']=='no_rgb'],
        checkpoint_selection_performed=False, native_launch_in_this_phase=False)


def assignments(phase, slot, workers):
    if type(slot) is not int or type(workers) is not int or workers not in (1, 3):
        raise ValueError('bounded exact worker assignment required')
    if phase in ('serial', 'parallel'):
        if workers != (1 if phase == 'serial' else 3) or slot not in range(3):
            raise ValueError('three serial or three concurrent benchmark assignments required')
        return [dict(name=f'benchmark_{i}', seed=BENCH_SEEDS[i], variant='full', condition='jepa')
            for i in range(2*slot, 2*slot+2)]
    if phase != 'fits' or slot not in range(workers):
        raise ValueError('exact fitting phase and slot required')
    return [dict(r) for r in ROSTER[slot::workers]]


def make_request(phase, slot, workers, launch_sha256):
    return dict(phase=phase, slot=slot, workers=workers, launch_sha256=launch_sha256,
        name=f'{phase}_{slot}', jobs=assignments(phase, slot, workers))


def benchmark_decision(serial, parallel):
    jobs = []
    for phase, value in (('serial', serial), ('parallel', parallel)):
        if len(value['records']) != 3 or not math.isfinite(value['wall_s']) or value['wall_s'] <= 0:
            raise ValueError('three complete assignments and positive measured phase duration required')
        flattened = []
        for slot, row in enumerate(value['records']):
            if (row['name'] != f'{phase}_{slot}' or row['status'] != 'ALL_PHASE_FIT_WORKER_COMPLETE'
                    or row['warmed_contexts'] != 4010 or row['cache_bytes'] != CACHE_BYTES
                    or len(row['jobs']) != 2):
                raise ValueError('every complete full-cache benchmark worker required')
            for actual, expected in zip(row['jobs'], assignments(phase, slot, 1 if phase=='serial' else 3), strict=True):
                if (actual['name'] != expected['name'] or actual['fit']['seed'] != expected['seed']
                        or actual['fit']['condition'] != 'jepa' or actual['fit']['input_variant'] != 'full'
                        or actual['fit']['benchmark'] is not True or actual['fit']['updates'] != 20
                        or actual['actual_updates'] != 20):
                    raise ValueError('every exact assigned twenty-update benchmark fit required')
                flattened.append(actual)
        jobs.append(flattened)
    for a, b in zip(*jobs, strict=True):
        if a['name'] != b['name'] or a['fit'] != b['fit'] or a['ledger_sha256'] != b['ledger_sha256']:
            raise ValueError('every optimizer receipt and final model must match exactly')
    speedup = serial['wall_s']/parallel['wall_s']
    memory_ok = all(type(r['peak_rss_bytes']) is int and 0 < r['peak_rss_bytes'] <= WORKER_RAM
        for phase in (serial, parallel) for r in phase['records'])
    return dict(measured_speedup=speedup, exact_update_and_model_equality=True,
        parallel_memory_allowance_pass=memory_ok, selected_workers=3 if speedup >= 1.25 and memory_ok else 1,
        complete_full_cache_warmups=6, compared_benchmark_fits=6, optimizer_updates_per_phase=120)
