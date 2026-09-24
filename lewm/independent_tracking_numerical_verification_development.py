"""Independent numerical reconstruction of the fixed tracking challenge.

No file access, simulator, observer, source export or execution authority.
Caller must first authenticate the complete eight-trial base/stress sensor
phase before supplying native arrays. This does not verify that access gate,
raw contacts/geometry/sensors, stress application or predecessor independence.
"""
import itertools
import math

import numpy as np
from scipy.spatial.transform import Rotation

# Independent transcription of the reviewed V1 schedule; not imported from the
# production coverage/scoring routines that this module checks.
BOUNDARIES = (0, 20, 60, 80, 206, 226, 256, 276, 402, 442)
SEGMENTS = ('initial_hold', 'approach', 'approach_brake', 'turn_out',
    'turn_out_brake', 'translated_view', 'translation_brake', 'turn_back', 'final_hold')
ARMS = ('original', 'temporal_anchor')
METRICS = ('position_m', 'orientation_rad', 'incremental_position_m', 'incremental_orientation_rad')


def require(value, message):
    if not value:
        raise ValueError(message)


def same(actual, expected, *, tolerance=1e-10):
    """Strict keys, populations and booleans; tolerance only for finite floats."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), 'different report fields')
        for k in expected: same(actual[k], expected[k], tolerance=tolerance)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), 'different report population')
        for a, b in zip(actual, expected, strict=True): same(a, b, tolerance=tolerance)
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected)
            and abs(actual - expected) <= tolerance, 'different numerical result')
    else:
        require(type(actual) is type(expected) and actual == expected, 'different exact result')


def _native(raw):
    t = np.asarray(raw['timestamp_s'], dtype=float)
    p = np.asarray(raw['base_pose_world'], dtype=float)
    require(t.ndim == 1 and len(t) <= 22850 and p.shape == (len(t), 7)
        and np.isfinite(t).all() and np.isfinite(p).all(), 'bounded finite native prefix required')
    require(np.allclose(t * 500, np.arange(1, len(t) + 1), atol=5e-7, rtol=0),
        'native timestamps must begin at 2ms and include every 500Hz sample')
    require(np.allclose(np.sum(p[:, 3:] ** 2, axis=1) ** .5, 1., atol=1e-7, rtol=0),
        'unit native XYZW quaternions required')
    return t, p


def reconstruct_coverage(raw, *, direction, completed_ticks, schedule_complete, physical_stop, acquisition_stop):
    t, p = _native(raw); n = len(t)
    v = np.asarray(raw['base_twist_world'], dtype=float)
    require(v.shape == (n, 6) and np.isfinite(v).all(), 'complete finite world twist prefix required')
    require(direction in ('left', 'right') and type(completed_ticks) is int and 0 <= completed_ticks <= 442
        and type(schedule_complete) is bool, 'exact bounded schedule metadata required')
    require(all(x is None or type(x) is str and bool(x) for x in (physical_stop, acquisition_stop)),
        'explicit nullable nonempty stop reasons required')
    complete = completed_ticks == 442 and physical_stop is None and acquisition_stop is None
    require(schedule_complete == complete and (complete or physical_stop is not None or acquisition_stop is not None),
        'terminal completion and stop metadata disagree')
    early = n < 750 and completed_ticks == 0 and not complete
    require(early or 750 + 50 * completed_ticks <= n <= 750 + 50 * completed_ticks +
        (50 if physical_stop is not None else 0), 'sample population differs from completed/partial command accounting')
    # Project body-forward into the world plane, then integrate principal
    # heading changes. This does not call the production Euler-yaw/unwrap path.
    if n:
        forward = Rotation.from_quat(p[:, 3:]).apply(np.tile([1., 0., 0.], (n, 1)))
        angles = np.arctan2(forward[:, 1], forward[:, 0])
        changes = np.diff(angles)
        yaw = np.r_[0., np.cumsum(np.arctan2(np.sin(changes), np.cos(changes)))]
    else:
        yaw = np.empty(0)
    segments = {}
    for name, a, b in zip(SEGMENTS, BOUNDARIES[:-1], BOUNDARIES[1:], strict=True):
        start = 749 + 50 * a; stop = 749 + 50 * b; present = n > start; last = min(n - 1, stop)
        xy = p[start:last + 1, :2] if present else None
        segments[name] = dict(complete=n > stop, available=present,
            first_sample=start if present else None, last_sample=last if present else None,
            planar_displacement_m=float(math.dist(xy[0], xy[-1])) if present else None,
            planar_path_m=float(math.fsum(math.dist(a, b) for a, b in zip(xy[:-1], xy[1:]))) if present else None,
            signed_yaw_rad=float(yaw[last] - yaw[start]) if present else None)
    sign = 1 if direction == 'left' else -1
    turns = {name: bool(segments[name]['complete'] and
        sign * multiplier * segments[name]['signed_yaw_rad'] >= math.radians(150))
        for name, multiplier in (('turn_out', 1), ('turn_back', -1))}
    translations = {name: bool(segments[name]['complete'] and segments[name]['planar_displacement_m'] >= .20)
        for name in ('approach', 'translated_view')}
    last_second = segments['final_hold']['complete']
    linear = max(math.hypot(*row[:3]) for row in v[-500:]) if last_second else None
    angular = max(math.hypot(*row[3:]) for row in v[-500:]) if last_second else None
    stopped = bool(last_second and linear <= .02 and angular <= .05)
    return dict(status='MEASURED_TRACKING_CHALLENGE_COVERAGE_ONLY', segments=segments,
        direction=direction, physics_samples=n, settling_complete=n >= 750,
        completed_ticks=completed_ticks, schedule_complete=schedule_complete,
        physical_stop=physical_stop, acquisition_stop=acquisition_stop,
        turn_coverage=turns, translation_coverage=translations,
        final_second_linear_speed_max_m_s=linear, final_second_angular_speed_max_rad_s=angular,
        final_second_stop_covered=stopped,
        intended_motion_covered=bool(complete and all(turns.values()) and all(translations.values()) and stopped),
        observer_accuracy_evaluated=False, independent_observations_verified=False,
        real_time_qualified=False, navigation_qualified=False)


def verify_coverage(raw, result, saved, *, direction):
    expected = reconstruct_coverage(raw, direction=direction, **{k: result[k] for k in
        ('completed_ticks', 'schedule_complete', 'physical_stop', 'acquisition_stop')})
    # Native quaternions permit 1e-7 norm roundoff. SciPy normalizes them whereas
    # the production yaw formula uses raw components. Do not tolerate a change
    # to any pass/fail boolean at a threshold, even within the scalar tolerance.
    same(saved, expected, tolerance=1e-6)
    return dict(numerical_coverage_reconstruction_verified=True, coverage=expected,
        raw_artifacts_authenticated_by_interface=False, full_challenge_pass=False, navigation_qualified=False)


def _rotation(matrix):
    r = np.asarray(matrix, dtype=float)
    require(r.shape == (3, 3) and np.isfinite(r).all()
        and np.allclose(r.T @ r, np.eye(3), atol=1e-7, rtol=0)
        and abs(np.linalg.det(r) - 1.) <= 1e-7, 'proper finite pose rotation required')
    return r


def _angle(r):
    # Matrix skew/trace atan2, independent of the scorer's quaternion products.
    sine = math.hypot(r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]) / 2
    return math.atan2(sine, float(np.clip((np.trace(r) - 1) / 2, -1., 1.)))


def _statistics(values):
    if not values: return dict(count=0, mean=None, median=None, p95=None, maximum=None)
    ordered = sorted(values)
    def quantile(q):
        index = (len(ordered) - 1) * q; a = math.floor(index); b = math.ceil(index)
        return float(ordered[a] + (ordered[b] - ordered[a]) * (index - a))
    return dict(count=len(values), mean=float(math.fsum(values) / len(values)),
        median=quantile(.5), p95=quantile(.95), maximum=float(ordered[-1]))


def verify_pose_stream(raw, estimates, evaluation_rows, phase_report, saved_score):
    """One base OR stress stream; iterators allow bounded non-materializing use.

    Validates numerical errors, actual clocks, common/conditional denominators,
    bridge-error subset and terminal absence. It does NOT reconstruct observer
    inference, full bridge/rejoin semantics, stress interventions or raw audit.
    """
    t, native = _native(raw)
    count = phase_report['frames']
    require(type(count) is int and 0 <= count <= 443, 'bounded exact frame count required')
    require(count == 0 or len(t) >= 750 + 50 * (count - 1), 'every recorded observation needs native truth')
    origin = Rotation.from_quat(native[749, 3:]) if count else None
    initial = origin.as_matrix() if count else None
    fields = (*METRICS, 'paired_position_m', 'paired_orientation_rad', 'observer_wall_ms')
    values = {arm: {k: [] for k in fields} for arm in ARMS}
    bridges = {k: [] for k in METRICS}; previous = dict.fromkeys(ARMS); first_failure = dict.fromkeys(ARMS)
    availability = dict(both=0, original_only=0, temporal_anchor_only=0, neither=0)
    seen = 0; sentinel = object()
    for row, evaluated in itertools.zip_longest(estimates, evaluation_rows, fillvalue=sentinel):
        require(row is not sentinel and evaluated is not sentinel and seen < count, 'complete paired stream lengths required')
        frame = seen; sample = 749 + 50 * frame; now = 1_500_000_000 + 100_000_000 * frame
        require(type(row['frame']) is int and row['frame'] == frame and type(row['measured_ns']) is int
            and row['measured_ns'] == now and abs(t[sample] * 1e9 - now) < 1., 'exact observation/native clock join required')
        require(set(row['arms']) == set(ARMS), 'both fixed observers required')
        present = {a: row['arms'][a]['pose'] is not None for a in ARMS}
        category = 'both' if all(present.values()) else 'original_only' if present['original'] else (
            'temporal_anchor_only' if present['temporal_anchor'] else 'neither')
        require(row['availability'] == category, 'availability must match actual supplied poses')
        availability[category] += 1
        truth_p = initial.T @ (native[sample, :3] - native[749, :3])
        truth_r = initial.T @ Rotation.from_quat(native[sample, 3:]).as_matrix()
        errors = {}
        for arm in ARMS:
            data = row['arms'][arm]; pose = data['pose']; ms = data['observer_wall_ms']
            require(type(ms) in (int, float) and math.isfinite(ms) and ms >= 0, 'finite nonnegative observer timing required')
            values[arm]['observer_wall_ms'].append(float(ms))
            if pose is None:
                if first_failure[arm] is None: first_failure[arm] = frame
                previous[arm] = None; errors[arm] = None; continue
            require(first_failure[arm] is None, 'terminally unavailable observer cannot resume')
            require(pose['frame'] == frame and pose['measured_ns'] == now
                and pose['global_history_reset'] is False and pose['position_error_bound'] is None,
                'same-frame measured pose without invented reset or bound required')
            p = np.asarray(pose['position_initial_body_m'], dtype=float)
            require(p.shape == (3,) and np.isfinite(p).all(), 'finite three-dimensional estimate required')
            r = _rotation(pose['rotation_initial_body_from_current_body'])
            e = dict(position_m=float(math.dist(p, truth_p)), orientation_rad=_angle(r.T @ truth_r),
                incremental_position_m=None, incremental_orientation_rad=None)
            if previous[arm] is not None:
                op, oR, ot, otR = previous[arm]
                e['incremental_position_m'] = float(math.dist(p - op, truth_p - ot))
                e['incremental_orientation_rad'] = _angle((oR.T @ r).T @ (otR.T @ truth_r))
            previous[arm] = p, r, truth_p, truth_r
            errors[arm] = e
            for key, value in e.items():
                if value is not None: values[arm][key].append(value)
            if category == 'both':
                values[arm]['paired_position_m'].append(e['position_m'])
                values[arm]['paired_orientation_rad'].append(e['orientation_rad'])
            if arm == 'temporal_anchor' and data['continuity']['status'] == 'MEASURED_INCREMENT_BRIDGE':
                require(all(v is not None for v in e.values()), 'bridge needs consecutive actual estimates')
                for key, value in e.items(): bridges[key].append(value)
        same(evaluated, dict(frame=frame, measured_ns=now, native_sample_index=sample,
            availability=category, errors=errors, evaluator_only=True, navigation_qualified=False), tolerance=1e-7)
        seen += 1
    require(seen == count, 'missing frames cannot improve a complete stream score')
    same(phase_report['availability'], availability)
    for arm in ARMS:
        same(phase_report['arms'][arm], dict(frames=count, available=len(values[arm]['position_m']), first_failure=first_failure[arm]))
    same(phase_report['continuity']['total_bridged_frames'], len(bridges['position_m']))
    stats = {a: {k: _statistics(v) for k, v in d.items()} for a, d in values.items()}
    within = {a: bool(count and stats[a]['position_m']['count'] == count
        and stats[a]['position_m']['maximum'] <= .02 and stats[a]['orientation_rad']['maximum'] <= math.radians(2)) for a in ARMS}
    expected = dict(frames=count, arms=stats, bridged_frame_errors={k: _statistics(v) for k, v in bridges.items()},
        empirical_local_pose_allocation_met=within, availability=availability,
        empirical_allocation_is_calibrated_bound=False, incremental_error_is_consecutive_estimate_difference=True,
        observer_timing_excludes_acquisition_and_control=True, navigation_qualified=False)
    same(saved_score, expected, tolerance=1e-7)
    return dict(numerical_pose_score_reconstruction_verified=True, frames=count, availability=availability,
        score=expected, raw_artifacts_authenticated_by_interface=False, observer_inference_recomputed=False,
        stress_application_verified=False, full_challenge_pass=False, navigation_qualified=False, real_time_qualified=False)
