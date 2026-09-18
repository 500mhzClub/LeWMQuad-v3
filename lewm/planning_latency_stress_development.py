"""Add a declared simulation-time delay to each planning result publication."""
import time


class PlanningLatencyClock:
    def __init__(self, base, extra_ns=20_000_000):
        if type(extra_ns) is not int or extra_ns <= 0:
            raise ValueError('positive explicit planning delay required')
        self.base = base
        self.extra_ns = extra_ns
        self.rows = []

    def begin(self, stage):
        self.base.begin(stage)
        self.base.local.planning_delay_applied = False

    def end(self):
        self.base.end()

    def __call__(self):
        ready = self.base()
        if (getattr(self.base.local, 'stage', None) != 'planning'
                or self.base.local.planning_delay_applied):
            return ready
        target = ready+self.extra_ns
        started = time.perf_counter_ns()
        with self.base.condition:
            while self.base.ns < target and not self.base.closed:
                self.base.condition.wait(timeout=.05)
            if self.base.closed:
                raise RuntimeError('clock closed before delayed planning publication')
            released = self.base.ns
        # Deliberate waiting is not extra measured CPU service on a later read.
        self.base.local.waited += time.perf_counter_ns()-started
        self.base.local.planning_delay_applied = True
        self.rows.append(dict(stage_start_ns=self.base.local.start_sim,
            base_ready_ns=ready, extra_ns=self.extra_ns,
            earliest_release_ns=target, released_ns=released))
        return released
