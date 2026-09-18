"""Synthetic mapping-projection reuse timings, including cache construction."""
from datetime import datetime, timezone
import json
import os
import statistics
import time

import numpy as np

from lewm.frame_body_projection_cache_development import FrameBodyProjectionCache, project_body
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources

SOURCE = 'scripts/benchmark_go2_frame_body_projection_cache_v1.py'
TEST = 'lewm/tests/test_frame_body_projection_cache_development.py'
PROTOCOL = 'docs/go2_frame_body_projection_cache_component_v1_2026-09-11.md'
PROFILE = 'docs/go2_tiled_density_progressive_floor_profile_v2_completion_verification_2026-09-11.json'
PROFILE_SHA = '1c3351b8647566f2d9745502aa8d43bb050101d27dfb2991536e25189cd522cc'
OUTPUT = ROOT/'docs/go2_frame_body_projection_cache_component_benchmark_2026-09-11.json'


def sequence(images, order, expected, *, cached):
    start = time.perf_counter()
    cache = FrameBodyProjectionCache() if cached else None
    elapsed = time.perf_counter()-start if cached else 0.
    for index in order:
        start = time.perf_counter()
        value = cache.body(images[index]) if cached else project_body(images[index])
        elapsed += time.perf_counter()-start
        if value.tobytes() != expected[index]:
            raise ValueError('every complete projected grid must remain byte-exact')
        del value
    if cached:
        if cache.counts() != dict(hits=8, misses=2, uncached=0):
            raise ValueError('same two input grids and eight exact reuses required')
        start = time.perf_counter(); cache.close(); elapsed += time.perf_counter()-start
        if cache._entries or not cache.closed: raise ValueError('no projection retained across observations')
    return elapsed


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive original component benchmark required')
    env = dict(PYTHONHASHSEED='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    if not __debug__ or any(os.environ.get(k) != v for k,v in env.items()):
        raise ValueError('assertions and fixed hash/threads required')
    verify({PROFILE:PROFILE_SHA})
    witness = json.loads((ROOT/PROFILE).read_text())
    if witness['status'] != 'TILED_DENSITY_PROGRESSIVE_FLOOR_PROFILE_COMPLETION_VERIFIED':
        raise ValueError('completed motivating controller profile required')
    sources = discover_sources((SOURCE, TEST, PROTOCOL, PROFILE), witness['source_sha256']); verify(sources)
    results = {}
    rng = np.random.default_rng(20260911)
    for dtype in (np.float32, np.float64):
        images = [rng.uniform(.2, 5., size=(480,640)).astype(dtype) for _ in range(2)]
        for image in images: image[::13, ::11] = 0.
        before = [image.tobytes() for image in images]
        expected = [project_body(image).tobytes() for image in images]
        for name, order in [('grouped', [0]*5+[1]*5), ('interleaved', [0,1]*5)]:
            for _ in range(2):
                sequence(images, order, expected, cached=False)
                sequence(images, order, expected, cached=True)
            samples = []
            for repeat in range(20):
                timing = [None, None]; execution_order = [0,1] if repeat%2 == 0 else [1,0]
                for index in execution_order:
                    timing[index] = sequence(images, order, expected, cached=bool(index))
                samples.append(dict(repetition=repeat, execution_order=execution_order,
                    baseline_s=timing[0], candidate_s=timing[1]))
            totals = [sum(row[key] for row in samples) for key in ('baseline_s','candidate_s')]
            results[np.dtype(dtype).name+'_'+name] = dict(paired_repetitions=20,
                calls_per_observation=10, distinct_depth_grids=2, exact_cache_hits_per_observation=8,
                baseline_total_s=totals[0], candidate_total_s=totals[1],
                baseline_median_s=statistics.median(row['baseline_s'] for row in samples),
                candidate_median_s=statistics.median(row['candidate_s'] for row in samples),
                total_time_reduction_percent=100*(1-totals[1]/totals[0]), samples=samples)
            if [image.tobytes() for image in images] != before: raise ValueError('source grids changed')
    verify(sources)
    write_json(OUTPUT, dict(status='FRAME_BODY_PROJECTION_CACHE_SYNTHETIC_COMPONENT_BENCHMARK_COMPLETE',
        utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources, source_count=len(sources),
        motivating_profile_completion_sha256=PROFILE_SHA, workloads=results,
        synthetic_depth_grids_only=True, every_projected_grid_byte_exact=True, original_inputs_unchanged=True,
        initialization_key_copy_immutable_backing_and_close_timed=True,
        verification_outside_timing=True, warmup_pairs_per_workload=2,
        up_vector_floor_classification_and_height_checks_cached=False,
        controller_integration_performed=False, whole_controller_speedup_established=False,
        model_or_raw_sensor_inputs_loaded=False, native_execution=False, isolated_benchmark=False,
        real_time_qualified=False, navigation_qualified=False, goal_achieved=False))
    print('FRAME_BODY_PROJECTION_CACHE_COMPONENT_COMPLETE', digest(OUTPUT), len(sources), flush=True)
    for name,result in results.items(): print(name,result['total_time_reduction_percent'],flush=True)


if __name__ == '__main__': main()
