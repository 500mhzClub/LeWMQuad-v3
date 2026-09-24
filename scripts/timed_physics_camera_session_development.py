"""Attribute native command-service cost without changing execution order."""
import time
from scripts.asynchronous_camera_session_development import AsynchronousCameraSession


class TimedPhysicsCameraSession(AsynchronousCameraSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.command_component_timings = []
        self._component_timing = None
        for owner, attribute, label in (
                (self.ctx.build.scene, 'step', 'physics_step'),
                (self, '_sample', 'sensor_guard_recording'),
                (self.ctx.policy, 'act', 'gait_inference')):
            original = getattr(owner, attribute)
            def timed(*args, _original=original, _label=label, **kwargs):
                row = self._component_timing
                if row is None:
                    return _original(*args, **kwargs)
                start = time.perf_counter_ns()
                try:
                    return _original(*args, **kwargs)
                finally:
                    row[_label+'_wall_ns'] += time.perf_counter_ns()-start
                    row[_label+'_calls'] += 1
            setattr(owner, attribute, timed)

    def command_policy_step(self, requested):
        row = dict(simulator_ns=int(self.ctx.runner._sim_time_ns),
            physics_step_wall_ns=0, physics_step_calls=0,
            sensor_guard_recording_wall_ns=0, sensor_guard_recording_calls=0,
            gait_inference_wall_ns=0, gait_inference_calls=0)
        wall = time.perf_counter_ns()
        cpu = time.thread_time_ns()
        self._component_timing = row
        try:
            return super().command_policy_step(requested)
        finally:
            self._component_timing = None
            row['owner_thread_cpu_ns'] = time.thread_time_ns()-cpu
            row['total_wall_ns'] = time.perf_counter_ns()-wall
            self.command_component_timings.append(row)
