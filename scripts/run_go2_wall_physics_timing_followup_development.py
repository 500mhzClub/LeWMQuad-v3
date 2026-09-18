"""Full fitted-motion follow-up with native service component timing."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_async_wall_fitted_control_development as fitted
from scripts.timed_physics_camera_session_development import TimedPhysicsCameraSession

ROOT = 'go2_wall_physics_timing_pose_command_layout01_4800_v1_attempt_001'


def finish(name, value):
    if name == 'launch.json':
        paths = (__file__, 'scripts/timed_physics_camera_session_development.py')
        value = value | dict(experiment='full_wall_fitted_physics_cost_followup_v1',
            prior_attempt_preserved=fitted.ROOT,
            diagnostic_followup_not_replacement=True,
            command_service_component_timing=True,
            component_timers_change_runtime_cost=True,
            controller_and_physical_limits_unchanged=True,
            extra_sources=value['extra_sources'] | {
                p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths})
    RAW_WRITE(name, value)


def annotate(name, value):
    bind(fitted.annotate, RAW_WRITE=bind(finish, RAW_WRITE=RAW_WRITE))(name, value)


def main():
    probe = fitted.probe
    base = probe.cohort.stable.source.BASE
    failure = json.loads((base/fitted.ROOT/'failure.json').read_text())
    if 'two in-flight camera acquisitions' not in failure['reason']:
        raise ValueError('expected original recorded queue failure')
    holder = {}
    def camera_session(*args, **kwargs):
        session = TimedPhysicsCameraSession(*args, **kwargs)
        holder['session'] = session
        return session
    try:
        bind(probe.main, ROOT=ROOT, NAVIGATION_TICKS=4800, annotate=annotate,
            AsynchronousCameraSession=camera_session,
            previous=SimpleNamespace(WallDeadlineRuntime=fitted.fitted_runtime))()
    finally:
        if 'session' in holder:
            with (base/ROOT/'command_component_timings.json').open('x') as f:
                json.dump(holder['session'].command_component_timings, f, indent=2)


if __name__ == '__main__':
    main()
