"""Low-overhead elapsed/thread-CPU timings around unchanged planning methods."""
import time
from lewm.persistent_local_visual_recovery_development import PersistentLocalVisualRuntime


class LivePlanningProfileRuntime(PersistentLocalVisualRuntime):
    def __init__(self, *args, **kwargs):
        self.plan_profile_rows = []
        self.profile_current = None
        super().__init__(*args, **kwargs)

    def _measure(self, label, function, *args, **kwargs):
        wall = time.perf_counter_ns(); cpu = time.thread_time_ns()
        try:
            return function(*args, **kwargs)
        finally:
            cpu = time.thread_time_ns()-cpu; wall = time.perf_counter_ns()-wall
            if self.profile_current is not None:
                row = self.profile_current['components'].setdefault(label,
                    dict(calls=0, wall_ns=0, thread_cpu_ns=0))
                row['calls'] += 1; row['wall_ns'] += wall; row['thread_cpu_ns'] += cpu

    def _plan(self, item):
        packet, _ = item
        self.profile_current = dict(frame=packet.frame, measured_ns=packet.measured_ns, components={})
        try:
            return self._measure('whole_plan_including_release_wait', super()._plan, item)
        finally:
            self.plan_profile_rows.append(self.profile_current)
            self.profile_current = None

    def _route(self, *args, **kwargs):
        return self._measure('route', super()._route, *args, **kwargs)

    def _select_action(self, *args, **kwargs):
        return self._measure('action_selection', super()._select_action, *args, **kwargs)

    def _correct_prediction(self, *args, **kwargs):
        return self._measure('alternative_forecasts', super()._correct_prediction, *args, **kwargs)

    def _pose(self, *args, **kwargs):
        return self._measure('pose_read', super()._pose, *args, **kwargs)
