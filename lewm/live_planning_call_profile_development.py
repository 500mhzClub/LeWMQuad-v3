"""Sparse live call profiles to locate action-selection elapsed/CPU gaps."""
import cProfile
import io
import pstats
import time

from lewm.live_planning_profile_development import LivePlanningProfileRuntime


class LivePlanningCallProfileRuntime(LivePlanningProfileRuntime):
    def __init__(self, *args, **kwargs):
        self.call_profiles = []
        super().__init__(*args, **kwargs)

    def _select_action(self, packet, *args, **kwargs):
        # Alternate timer types over twelve consecutive post-survey plans.
        # Profiling can alter deadlines: these are diagnostic trajectories.
        if not 300 <= packet.frame < 348:
            return super()._select_action(packet, *args, **kwargs)
        cpu_timer = (packet.frame // 4) % 2 == 0
        profile = cProfile.Profile(time.thread_time if cpu_timer else time.perf_counter)
        try:
            return profile.runcall(super()._select_action, packet, *args, **kwargs)
        finally:
            self.call_profiles.append((packet.frame, 'thread_cpu' if cpu_timer else 'elapsed', profile))

    def save_call_profiles(self, output):
        for frame, timer, profile in self.call_profiles:
            name = f'calls_frame{frame}_{timer}'
            profile.dump_stats(str(output / (name + '.pstats')))
            stream = io.StringIO()
            pstats.Stats(profile, stream=stream).sort_stats('cumtime').print_stats(70)
            (output / (name + '.txt')).write_text(stream.getvalue())
