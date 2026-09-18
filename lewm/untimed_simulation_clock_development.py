"""Physics-time clock for explicit synchronous simulation experiments."""
import time
from lewm.measured_latency_simulation_development import MeasuredLatencyClock


class UntimedSimulationClock(MeasuredLatencyClock):
    def __call__(self):
        if self.closed:
            raise RuntimeError('untimed clock is closed')
        return self.ns

    def end(self):
        self.releases.append(dict(stage=self.local.stage, start_sim_ns=self.local.start_sim,
            end_sim_ns=self.ns, wall_ns=time.perf_counter_ns()-self.local.start_wall,
            service_cost_charged_to_simulation=False))
        super().end()
