"""Bounded Python stack samples from the calling thread's wall-time activity.

No tracing/profiling hooks, frame locals, source reads or process attachment.
Samples include waiting and native-call boundaries; they are not CPU self time,
native stacks, function call counts or evidence of negligible perturbation.
"""
from dataclasses import dataclass
import math
import sys
import threading
import time


def capture_stack(thread_ident, maximum_depth):
    frame = sys._current_frames().get(thread_ident)
    if frame is None:
        raise RuntimeError('original owner frame is unavailable')
    stack = []
    try:
        while frame is not None:
            if len(stack) >= maximum_depth:
                raise RuntimeError('stack depth capacity exceeded; no truncated profile')
            filename = frame.f_code.co_filename
            if any(part in ('sealed', 'sealed_test.json') or part.startswith('sealed_')
                    for part in filename.replace('\\', '/').split('/')):
                raise RuntimeError('protected frame cannot be recorded')
            stack.append((filename, frame.f_lineno, frame.f_code.co_name))
            frame = frame.f_back
    finally:
        del frame
    return tuple(stack)


@dataclass(frozen=True)
class Sample:
    offset_ns: int
    capture_wall_ns: int
    stack: tuple


class WallStackSampler:
    def __init__(self, *, interval_s=.01, maximum_samples=10_000, maximum_depth=128):
        if (type(interval_s) not in (float, int) or not math.isfinite(interval_s)
                or not .005 <= interval_s <= .1
                or type(maximum_samples) is not int or not 1 <= maximum_samples <= 10_000
                or type(maximum_depth) is not int or not 1 <= maximum_depth <= 128):
            raise ValueError('bounded interval, sample count and stack depth required')
        self.interval_s = float(interval_s)
        self.maximum_samples = maximum_samples
        self.maximum_depth = maximum_depth
        self.samples = []
        self.error = None
        self._owner = None
        self._thread = None
        self._stop = threading.Event()
        self._sampled = threading.Event()
        self._closed = False
        self._start_ns = None
        self._end_ns = None
        self._sampler_cpu_ns = 0

    def __enter__(self):
        if self._owner is not None or self._closed:
            raise RuntimeError('one original sampling scope required')
        self._owner = threading.current_thread()
        self._start_ns = time.perf_counter_ns()
        self._thread = threading.Thread(target=self._run, name='bounded_wall_stack_sampler', daemon=True)
        self._thread.start()
        return self

    def _run(self):
        cpu_start = time.thread_time_ns()
        try:
            while not self._stop.wait(self.interval_s):
                if not self._owner.is_alive():
                    raise RuntimeError('original sampling owner ended')
                if len(self.samples) >= self.maximum_samples:
                    raise RuntimeError('sample capacity exceeded; no partial success')
                start = time.perf_counter_ns()
                stack = capture_stack(self._owner.ident, self.maximum_depth)
                end = time.perf_counter_ns()
                self.samples.append(Sample(start-self._start_ns, end-start, stack))
                self._sampled.set()
        except Exception as error:
            self.error = str(error)
            self._stop.set()
            self._sampled.set()
        finally:
            self._sampler_cpu_ns = time.thread_time_ns()-cpu_start

    def close(self):
        if self._owner is None or threading.current_thread() is not self._owner:
            raise RuntimeError('original owner must close its sampling scope')
        if not self._closed:
            self._stop.set()
            self._thread.join(timeout=2.)
            if self._thread.is_alive():
                raise RuntimeError('sampler did not terminate; profile unavailable')
            self._end_ns = time.perf_counter_ns()
            self._closed = True
        if self.error is not None:
            raise RuntimeError(self.error)

    def __exit__(self, kind, error, traceback):
        try:
            self.close()
        except RuntimeError as sampling_error:
            if error is None:
                raise
            error.add_note('Stack sampler also failed: '+str(sampling_error))
        return False

    def report(self):
        if not self._closed:
            raise RuntimeError('closed sampling scope required')
        return dict(schema='python_owner_wall_stack_samples_development.v1',
            complete=self.error is None, error=self.error, interval_s=self.interval_s,
            maximum_samples=self.maximum_samples, maximum_depth=self.maximum_depth,
            scope_wall_ns=self._end_ns-self._start_ns, sampler_thread_cpu_ns=self._sampler_cpu_ns,
            sample_count=len(self.samples),
            samples=[dict(offset_ns=s.offset_ns, capture_wall_ns=s.capture_wall_ns,
                stack=[dict(filename=f,line=n,function=name) for f,n,name in s.stack]) for s in self.samples],
            attribution='sampled_owner_python_stack_wall_occupancy',
            fixed_rate_sampling_guaranteed=False, gil_scheduling_bias_excluded=False,
            target_cpu_self_time_measured=False, native_stack_captured=False,
            sampler_perturbation_calibrated=False, source_files_or_frame_locals_read=False,
            navigation_qualified=False, real_time_qualified=False)
