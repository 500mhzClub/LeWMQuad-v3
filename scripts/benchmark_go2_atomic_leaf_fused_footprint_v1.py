"""Balanced synthetic component timings; no controller-speed or navigation claim."""
from datetime import datetime, timezone
import gc
import json
import math
import os
import statistics
import time

from lewm import atomic_leaf_fused_footprint_development as candidate
from lewm import fused_scoped_footprint_development as baseline
from lewm.tests.test_atomic_leaf_fused_footprint_development import graph_signature
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources

SOURCE = 'scripts/benchmark_go2_atomic_leaf_fused_footprint_v1.py'
TEST = 'lewm/tests/test_atomic_leaf_fused_footprint_development.py'
PROTOCOL = 'docs/go2_atomic_leaf_fused_footprint_component_v1_2026-09-11.md'
PROFILE = 'docs/go2_tiled_density_progressive_floor_profile_v2_completion_verification_2026-09-11.json'
PROFILE_SHA = '1c3351b8647566f2d9745502aa8d43bb050101d27dfb2991536e25189cd522cc'
OUTPUT = ROOT/'docs/go2_atomic_leaf_fused_footprint_component_benchmark_2026-09-11.json'


def workloads():
    flat = {'leaves':[None, True, 17, -.0, 'surface']*2000}
    nested = {'shapes':[{'name':str(i), 'vertices':[[float(j), float(i), 0.] for j in range(16)],
        'flags':[True, False], 'witness':None} for i in range(128)]}
    shared = {'leaves':[1., 2., 3.]}
    for _ in range(18): shared = {'left':shared, 'right':shared}
    return {'flat_primitives':flat, 'nested_geometry_shape':nested, 'shared_dag':shared,
        'late_unsupported_leaf':{'valid_prefix':flat, 'unsupported':b'preserve original input'}}


def measure(old, new, source, *, fallback=False):
    before = graph_signature(source)
    expected = old(source)
    actual = new(source)
    if graph_signature(actual) != graph_signature(expected) or (fallback and actual is not source):
        raise ValueError('same complete output graph and fallback identity required')
    for _ in range(3): old(source); new(source)
    samples = []
    # Keep normal GC behavior in both arms, avoiding a special timing advantage.
    for repeat in range(30):
        order = [0, 1] if repeat % 2 == 0 else [1, 0]
        times = [None, None]
        for index in order:
            start = time.perf_counter()
            output = (old, new)[index](source)
            elapsed = time.perf_counter()-start
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError('finite positive component timing required')
            times[index] = elapsed
            if graph_signature(output) != graph_signature(expected) or (fallback and output is not source):
                raise ValueError('every timed complete graph must match')
            del output
        samples.append(dict(repetition=repeat, execution_order=order, baseline_s=times[0], candidate_s=times[1]))
    if graph_signature(source) != before:
        raise ValueError('component benchmark mutated input graph')
    totals = [sum(row[key] for row in samples) for key in ('baseline_s', 'candidate_s')]
    return dict(paired_repetitions=30, alternating_order=True, complete_graph_and_aliases_exact=True,
        source_unchanged=True, unsupported_input_identity_preserved=fallback,
        baseline_total_s=totals[0], candidate_total_s=totals[1],
        total_time_reduction_percent=100*(1-totals[1]/totals[0]),
        baseline_median_s=statistics.median(row['baseline_s'] for row in samples),
        candidate_median_s=statistics.median(row['candidate_s'] for row in samples), samples=samples)


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive original component benchmark required')
    if not __debug__ or os.environ.get('PYTHONHASHSEED') != '0' or not gc.isenabled():
        raise ValueError('assertions, fixed hash seed and normal garbage collection required')
    verify({PROFILE:PROFILE_SHA})
    witness = json.loads((ROOT/PROFILE).read_text())
    if witness['status'] != 'TILED_DENSITY_PROGRESSIVE_FLOOR_PROFILE_COMPLETION_VERIFIED':
        raise ValueError('completed timing profile required')
    sources = discover_sources((SOURCE, TEST, PROTOCOL, PROFILE), witness['source_sha256'])
    verify(sources)
    results = {}
    for name, source in workloads().items():
        entry = dict(freeze=measure(baseline.freeze_ordinary_footprint, candidate.freeze_ordinary_footprint,
            source, fallback=name == 'late_unsupported_leaf'))
        if name != 'late_unsupported_leaf':
            frozen = baseline.freeze_ordinary_footprint(source)
            entry['cached_clone'] = measure(baseline._clone_cached_receipt, candidate._clone_cached_receipt, frozen)
        results[name] = entry
    verify(sources)
    write_json(OUTPUT, dict(status='ATOMIC_LEAF_FUSED_FOOTPRINT_SYNTHETIC_COMPONENT_BENCHMARK_COMPLETE',
        utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources, source_count=len(sources),
        motivating_profile_completion_sha256=PROFILE_SHA, workloads=results,
        synthetic_graphs_only=True, model_or_raw_sensor_inputs_loaded=False, gc_enabled=True,
        component_only=True, isolated_benchmark=False, whole_controller_speedup_established=False,
        native_execution=False, real_time_qualified=False, navigation_qualified=False, goal_achieved=False))
    print('ATOMIC_LEAF_FUSED_FOOTPRINT_COMPONENT_COMPLETE', digest(OUTPUT), len(sources), flush=True)
    for name, entry in results.items():
        print(name, {key:value['total_time_reduction_percent'] for key,value in entry.items()}, flush=True)


if __name__ == '__main__':
    main()
