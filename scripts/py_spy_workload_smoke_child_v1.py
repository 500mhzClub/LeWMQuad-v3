"""Three fixed synthetic phases for an external-profiler availability check."""
import json
import time
import numpy as np
from scripts.calibrate_go2_wall_stack_sampler_v1 import python_integer_work, array_work, waiting_work


def main():
    values = np.linspace(-1.,1.,1_000_000,dtype=np.float64)
    rows = []
    for name, function, repetitions in (
            ('python_integer_work',python_integer_work,25),
            ('array_work',lambda:array_work(values),12),
            ('waiting_work',waiting_work,20)):
        start = time.perf_counter_ns()
        results = [function() for _ in range(repetitions)]
        rows.append(dict(function=name,repetitions=repetitions,
            elapsed_ns=time.perf_counter_ns()-start,results=results))
    print(json.dumps(dict(status='SYNTHETIC_PROFILER_CHILD_COMPLETE',phases=rows),allow_nan=False))


if __name__ == '__main__':
    main()
