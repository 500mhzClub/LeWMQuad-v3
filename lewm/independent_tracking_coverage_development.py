"""Evaluation-only measured coverage; commanded motion is never ground truth.

Call only after the complete paired sensor phase has been persisted and bound.
This numerical helper does not authenticate files or grant native-output access;
the future experiment runner must enforce that boundary before loading arrays.
"""
import math
import numpy as np

from lewm.independent_tracking_challenge_development import (
    DIRECTIONS, SEGMENTS, MAX_TICKS, MAX_PHYSICS_SAMPLES, SETTLE_SAMPLES, PHYSICS_PER_TICK)

MINIMUM_TURN_RAD = 5 * math.pi / 6
MINIMUM_TRANSLATION_M = .20
STOP_LINEAR_M_S = .02
STOP_ANGULAR_RAD_S = .05
FINAL_STOP_SAMPLES = 500  # Entire final second, not one favorable endpoint.


def measured_coverage(direction, timestamps_s, base_pose_world, base_twist_world, *,
                      completed_ticks, schedule_complete, physical_stop, acquisition_stop):
    """All recorded physics samples from settling onward, including stopped trials.

    Reports incomplete segments explicitly. Complete metadata with absent samples,
    or a partial-stop prefix presented as a complete experiment, is rejected.
    Quaternion is XYZW. Signed yaw is unwrapped at500Hz, not from commands/gyro.
    """
    if direction not in DIRECTIONS or type(completed_ticks) is not int or not 0 <= completed_ticks <= MAX_TICKS:
        raise ValueError('fixed direction and bounded completed-tick count required')
    if type(schedule_complete) is not bool or any(v is not None and (not isinstance(v, str) or not v)
                                               for v in (physical_stop, acquisition_stop)):
        raise ValueError('explicit completion and nullable nonempty stop reasons required')
    t = np.asarray(timestamps_s, float)
    p = np.asarray(base_pose_world, float)
    v = np.asarray(base_twist_world, float)
    n = len(t) if t.ndim == 1 else 0
    if (t.ndim != 1 or not 0 <= n <= MAX_PHYSICS_SAMPLES or p.shape != (n, 7) or v.shape != (n, 6)
            or not all(np.isfinite(a).all() for a in (t, p, v))
            or not np.allclose(t, np.arange(1, n + 1) * .002, atol=1e-9, rtol=0)):
        raise ValueError('complete finite500Hz physics prefix from settling required')
    minimum = SETTLE_SAMPLES + completed_ticks * PHYSICS_PER_TICK
    # A native stop can interrupt the final sample callback before the command
    # returns, so all50 samples may exist for a tick not marked completed.
    maximum = minimum + (PHYSICS_PER_TICK if physical_stop is not None else 0)
    early_stop = n < SETTLE_SAMPLES and completed_ticks == 0 and not schedule_complete and (
        physical_stop is not None or acquisition_stop is not None)
    if not early_stop and not minimum <= n <= maximum:
        raise ValueError('sample prefix does not match completed commands and at most one partial tick')
    if schedule_complete != (completed_ticks == MAX_TICKS and physical_stop is None and acquisition_stop is None):
        raise ValueError('schedule completion contradicts command/stop accounting')
    if not schedule_complete and physical_stop is None and acquisition_stop is None:
        raise ValueError('incomplete terminal prefix requires a recorded stop')
    q = p[:, 3:]
    if not np.allclose(np.linalg.norm(q, axis=1), 1., atol=1e-7, rtol=0):
        raise ValueError('unit native quaternions required')
    x, y, z, w = q.T
    yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y*y + z*z)))
    sign = 1 if direction == 'left' else -1
    reports = {}
    offset = 0
    for name, count, _ in SEGMENTS:
        first = SETTLE_SAMPLES - 1 + offset * PHYSICS_PER_TICK
        last = first + count * PHYSICS_PER_TICK
        available = first < n
        end = min(last, n - 1)
        complete = last < n
        reports[name] = dict(complete=complete, available=available,
            first_sample=first if available else None, last_sample=end if available else None,
            planar_displacement_m=float(np.linalg.norm(p[end, :2] - p[first, :2])) if available else None,
            planar_path_m=float(np.linalg.norm(np.diff(p[first:end+1, :2], axis=0), axis=1).sum()) if available else None,
            signed_yaw_rad=float(yaw[end] - yaw[first]) if available else None)
        offset += count
    turns = {name: bool(reports[name]['complete'] and sign * multiplier * reports[name]['signed_yaw_rad'] >= MINIMUM_TURN_RAD)
             for name, multiplier in (('turn_out', 1), ('turn_back', -1))}
    translations = {name: bool(reports[name]['complete'] and reports[name]['planar_displacement_m'] >= MINIMUM_TRANSLATION_M)
                    for name in ('approach', 'translated_view')}
    stop_window_present = bool(reports['final_hold']['complete'])
    stop_linear = float(np.linalg.norm(v[-FINAL_STOP_SAMPLES:, :3], axis=1).max()) if stop_window_present else None
    stop_angular = float(np.linalg.norm(v[-FINAL_STOP_SAMPLES:, 3:], axis=1).max()) if stop_window_present else None
    stopped = bool(stop_window_present and stop_linear <= STOP_LINEAR_M_S and stop_angular <= STOP_ANGULAR_RAD_S)
    return dict(status='MEASURED_TRACKING_CHALLENGE_COVERAGE_ONLY', segments=reports,
        direction=direction, physics_samples=n, settling_complete=bool(n >= SETTLE_SAMPLES),
        completed_ticks=completed_ticks, schedule_complete=schedule_complete,
        physical_stop=physical_stop, acquisition_stop=acquisition_stop,
        turn_coverage=turns, translation_coverage=translations,
        final_second_linear_speed_max_m_s=stop_linear, final_second_angular_speed_max_rad_s=stop_angular,
        final_second_stop_covered=stopped,
        intended_motion_covered=bool(schedule_complete and all(turns.values()) and all(translations.values()) and stopped),
        observer_accuracy_evaluated=False, independent_observations_verified=False,
        real_time_qualified=False, navigation_qualified=False)
