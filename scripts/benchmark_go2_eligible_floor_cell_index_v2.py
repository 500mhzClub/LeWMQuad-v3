"""Fixed alternating synthetic benchmark of identical measured floor masks."""
import json
import os
from pathlib import Path
import time
import numpy as np

from lewm.floor_footprint_bounds_development import observed_floor_cell_index as original
from lewm.eligible_floor_cell_index_development import observed_floor_cell_index as candidate
from lewm.tests.test_floor_footprint_bounds_development import scene
from scripts.run_go2_successive_choice_maze_development_v1 import digest,verify,write_json
from scripts.startup_source_inventory_development import discover_sources

SOURCE = 'scripts/benchmark_go2_eligible_floor_cell_index_v2.py'
TEST = 'lewm/tests/test_eligible_floor_cell_index_development.py'
OUTPUT = Path('docs/go2_eligible_floor_cell_index_microbenchmark_v2_2026-09-11.json')
PAIRS = 20


def fixtures():
    d,v=scene();up=np.array([0.,0.,1.]);rng=np.random.default_rng(2026091103)
    yield 'room',d,v,up
    mask=v.copy();mask[rng.random(mask.shape)<.3]=False
    yield 'missing_returns',np.where(mask,d,0.),mask,up
    yield 'all_missing',np.zeros_like(d),np.zeros_like(v),up
    yield 'unstructured_returns',rng.uniform(.2,5.,d.shape),np.ones_like(v),up
    tilted=np.array([.15,-.10,1.]);tilted/=np.linalg.norm(tilted)
    yield 'tilted_room',d.copy(),v.copy(),tilted


def exact(old,new):
    if set(old)!=set(new) or any(old[k].dtype!=new[k].dtype or old[k].shape!=new[k].shape
            or old[k].tobytes()!=new[k].tobytes() for k in old):
        raise ValueError('complete floor mask and prefix bytes must match')


def main():
    env=dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0')
    if not __debug__ or any(os.environ.get(k)!=v for k,v in env.items()):
        raise ValueError('assertions and fixed CPU environment required')
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive microbenchmark required')
    sources=discover_sources((SOURCE,TEST,'scripts/benchmark_go2_eligible_floor_cell_index_v1.py',
        'docs/go2_eligible_floor_cell_index_microbenchmark_v1_preflight_failure_2026-09-11.json'),{});verify(sources)
    reports={}
    for name,d,v,up in fixtures():
        before=(d.tobytes(),v.tobytes(),up.tobytes())
        exact(original(d,v,up),candidate(d,v,up))
        rows=[]
        for pair in range(PAIRS):
            order=('original','candidate') if pair%2==0 else ('candidate','original')
            outputs={};row=dict(pair=pair,order=list(order))
            for key in order:
                start=time.perf_counter_ns()
                outputs[key]=(original if key=='original' else candidate)(d,v,up)
                row[key+'_wall_ms']=(time.perf_counter_ns()-start)/1e6
            exact(outputs['original'],outputs['candidate'])
            rows.append(row)
        if before!=(d.tobytes(),v.tobytes(),up.tobytes()):raise ValueError('benchmark mutated input arrays')
        old=sum(r['original_wall_ms'] for r in rows);new=sum(r['candidate_wall_ms'] for r in rows)
        reports[name]=dict(pairs=rows,original_total_ms=old,candidate_total_ms=new,
            total_time_reduction_percent=100*(old-new)/old,all_output_arrays_byte_exact=True,inputs_unchanged=True)
        print('ELIGIBLE_INDEX_BENCHMARK',name,old,new,100*(old-new)/old,flush=True)
    verify(sources)
    write_json(OUTPUT,dict(status='ELIGIBLE_FLOOR_CELL_INDEX_SYNTHETIC_MICROBENCHMARK_COMPLETE',
        source_sha256=sources,environment=env,paired_repetitions=PAIRS,fixtures=reports,
        synthetic_inputs=True,controller_replay=False,real_sensor_inputs=False,isolated_benchmark=False,
        shared_host_timings=True,raw_controller_decisions_compared=False,native_execution=False,
        real_time_qualified=False,navigation_qualified=False,goal_achieved=False))
    print('ELIGIBLE_INDEX_BENCHMARK_COMPLETE',digest(OUTPUT),len(sources),flush=True)


if __name__=='__main__':main()
