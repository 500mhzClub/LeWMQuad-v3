"""Fixed synthetic paired calibration; no controller or runtime input access."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
from lewm.wall_stack_sampler_development import WallStackSampler

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'scripts/calibrate_go2_wall_stack_sampler_v1.py'
PATHS = (SOURCE, 'lewm/wall_stack_sampler_development.py',
    'lewm/tests/test_wall_stack_sampler_development.py',
    'docs/go2_wall_stack_sampler_calibration_v1_2026-09-11.md')
OUTPUT = ROOT/'docs/go2_wall_stack_sampler_calibration_2026-09-11.json'
FAILURE = ROOT/'docs/go2_wall_stack_sampler_calibration_failure_2026-09-11.json'


def identities():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in PATHS}


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def python_integer_work():
    value = 17
    for i in range(800_000):
        value = ((value*33)^i)&0xffffffff
    return value


def array_work(values):
    result = 0.
    for i in range(12):
        result += float(np.sin(values+i*.01).sum())
    return result


def waiting_work():
    time.sleep(.05)
    return 1


def measured(function, sampled):
    if sampled:
        sampler = WallStackSampler(interval_s=.01)
        start = time.perf_counter_ns()
        with sampler:
            value = function()
        elapsed = time.perf_counter_ns()-start
        return dict(elapsed_ns=elapsed, value=value, sampler=sampler.report())
    start = time.perf_counter_ns()
    value = function()
    return dict(elapsed_ns=time.perf_counter_ns()-start, value=value)


def main():
    if any(path.exists() or path.is_symlink() for path in (OUTPUT, FAILURE)):
        raise ValueError('exclusive calibration output required')
    threads = ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')
    if any(os.environ.get(key) != '1' for key in threads):
        raise ValueError('fixed single-thread numeric environment required')
    sources = identities()
    values = np.linspace(-1.,1.,1_000_000,dtype=np.float64)
    input_sha = hashlib.sha256(values.tobytes()).hexdigest()
    rows = []
    try:
        for name, function in (('python_integer',python_integer_work),
                ('native_array',lambda:array_work(values)),('waiting',waiting_work)):
            for pair in range(22):
                order = [False,True] if pair%2 == 0 else [True,False]
                results = {sampled:measured(function,sampled) for sampled in order}
                if results[False]['value'] != results[True]['value']:
                    raise ValueError('sampling changed the exact workload return value')
                rows.append(dict(workload=name,phase='warmup' if pair<2 else 'measurement',
                    pair=pair,execution_order=order,original=results[False],sampled=results[True],
                    return_value_exact=True))
        if identities() != sources or hashlib.sha256(values.tobytes()).hexdigest() != input_sha:
            raise ValueError('original source or synthetic array bytes changed')
        summary = {}
        for name in ('python_integer','native_array','waiting'):
            selected = [row for row in rows if row['workload']==name and row['phase']=='measurement']
            original = [row['original']['elapsed_ns'] for row in selected]
            sampled = [row['sampled']['elapsed_ns'] for row in selected]
            counts = [row['sampled']['sampler']['sample_count'] for row in selected]
            summary[name] = dict(pairs=len(selected),original_total_ns=sum(original),
                sampled_total_ns=sum(sampled),original_median_ns=statistics.median(original),
                sampled_median_ns=statistics.median(sampled),
                total_elapsed_overhead_percent=100*(sum(sampled)/sum(original)-1),
                sample_counts=counts,zero_sample_repetitions=sum(count==0 for count in counts))
        write(OUTPUT,dict(status='WALL_STACK_SAMPLER_SYNTHETIC_CALIBRATION_COMPLETE',
            utc=datetime.now(timezone.utc).isoformat(),explicit_source_sha256=sources,
            recursive_source_closure_claimed=False,python=sys.version,numpy=np.__version__,
            platform=platform.platform(),environment={key:os.environ[key] for key in threads},
            synthetic_array_sha256=input_sha,synthetic_array_unchanged=True,
            rows=rows,summary=summary,shared_host=True,controller_instrumented=False,
            raw_sensor_or_checkpoint_access=False,production_overhead_established=False,
            native_execution=False,navigation_qualified=False,real_time_qualified=False))
        print('CALIBRATION_COMPLETE',hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),json.dumps(summary),flush=True)
    except BaseException as error:
        write(FAILURE,dict(status='TERMINAL_WALL_STACK_SAMPLER_CALIBRATION_FAILURE',
            reason=repr(error),explicit_source_sha256=sources,completed_rows=rows,
            automatic_retry=False,evidence_preserved=True))
        raise


if __name__ == '__main__':
    main()
