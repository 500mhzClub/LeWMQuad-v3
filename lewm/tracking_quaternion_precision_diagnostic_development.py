"""Descriptive quaternion precision checks, never an acceptance-rule change.

No file access. The caller authenticates and admits the complete sensor phase
before supplying native arrays. Normalized copies quantify representation
sensitivity only; they are never substituted for recorded data or scored poses.
"""
import math

import numpy as np


def summarize_quaternions(quaternions, timestamps_s):
    original = np.asarray(quaternions)
    if original.dtype.kind != 'f':
        raise ValueError('floating quaternion array required')
    q = np.asarray(original, dtype=np.float64)
    t = np.asarray(timestamps_s, dtype=np.float64)
    if (q.ndim != 2 or q.shape[1] != 4 or not 1 <= len(q) <= 22850
            or t.shape != (len(q),) or not np.isfinite(q).all()
            or not np.isfinite(t).all()):
        raise ValueError('bounded finite XYZW quaternion/time population required')
    if not np.allclose(t, np.arange(1, len(q) + 1) * .002, atol=1e-9, rtol=0):
        raise ValueError('complete 500Hz clock required')
    # fsum/hypot avoid relying solely on the reduction that the failed gate used.
    norms = np.array([math.hypot(*row) for row in q])
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError('nonzero finite quaternion norms required')
    numpy_norms = np.linalg.norm(q, axis=1)
    errors = np.abs(norms - 1.)
    rejected = np.flatnonzero(np.abs(numpy_norms - 1.) > 1e-7)
    normalized = q / norms[:, None]

    def yaw(a):
        x, y, z, w = a.T
        return np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))

    raw_yaw, unit_yaw = yaw(q), yaw(normalized)
    yaw_difference = np.arctan2(np.sin(raw_yaw - unit_yaw), np.cos(raw_yaw - unit_yaw))
    raw_unwrapped, unit_unwrapped = np.unwrap(raw_yaw), np.unwrap(unit_yaw)
    worst = int(np.argmax(errors))
    float32_roundtrip = q.astype(np.float32).astype(np.float64)
    capture_indices = np.arange(749, len(q), 50, dtype=int)
    return dict(
        status='RECORDED_QUATERNION_PRECISION_DIAGNOSTIC_ONLY', samples=len(q),
        stored_dtype=str(original.dtype), component_order='xyzw_from_recording_contract',
        binary32_epsilon=float(np.finfo(np.float32).eps),
        components_exactly_representable_as_binary32=int(np.sum(q == float32_roundtrip)),
        component_count=int(q.size), norm_min=float(norms.min()), norm_max=float(norms.max()),
        absolute_norm_error_mean=float(errors.mean()),
        absolute_norm_error_p99=float(np.quantile(errors, .99)),
        absolute_norm_error_max=float(errors[worst]),
        numpy_vs_hypot_norm_max_difference=float(np.max(np.abs(norms - numpy_norms))),
        frozen_norm_tolerance=1e-7, frozen_norm_gate_passes=len(rejected) == 0,
        frozen_gate_rejected_sample_count=int(len(rejected)),
        rejected_settle_samples=int(np.sum(rejected < 750)),
        rejected_capture_samples=int(np.sum(np.abs(numpy_norms[capture_indices] - 1.) > 1e-7)),
        first_rejected_sample_indices=rejected[:20].tolist(),
        first_rejected_time_s=float(t[rejected[0]]) if len(rejected) else None,
        worst_sample=dict(index=worst, time_s=float(t[worst]), xyzw=q[worst].tolist(),
                          norm=float(norms[worst])),
        diagnostic_normalized_copy_yaw_max_difference_rad=float(np.max(np.abs(yaw_difference))),
        diagnostic_normalized_copy_accumulated_yaw_max_difference_rad=float(np.max(np.abs(
            (raw_unwrapped - raw_unwrapped[0]) - (unit_unwrapped - unit_unwrapped[0])))),
        quaternion_permutation_cannot_change_norm=True,
        original_arrays_modified=False, normalized_copies_used_for_scoring=False,
        simulator_precision_cause_proved=False, native_trajectory_validated=False,
        tracking_accuracy_verified=False, full_challenge_pass=False,
        navigation_qualified=False, goal_achieved=False)
