"""Offline integration diagnostic; not a new deployed sensor or controller."""
import math

import numpy as np

from lewm.relative_gyro_turn_development import rotation_increment

METHODS = ('midpoint', 'coning', 'right_endpoint')


def integrate_rates(measured_ns, body_rates, method):
    times, rates = np.asarray(measured_ns), np.asarray(body_rates, dtype=float)
    if (method not in METHODS or times.ndim != 1 or not len(times) or times.dtype.kind not in 'iu'
            or np.any(times < 0) or np.any(np.diff(times.astype(np.int64)) <= 0)
            or rates.shape != (len(times), 3) or not np.isfinite(rates).all()):
        raise ValueError('ordered actual sample times and finite body rates required')
    result = [np.eye(3)]
    for i in range(1, len(times)):
        dt = int(times[i] - times[i - 1]) * 1e-9
        previous, current = rates[i - 1], rates[i]
        vector = current * dt if method == 'right_endpoint' else (previous + current) * (dt / 2)
        if method == 'coning':
            # First noncommuting correction for linearly varying body rates:
            # earlier right-multiplied rotations precede later rotations.
            vector = vector + np.cross(previous, current) * (dt * dt / 12)
        result.append(result[-1] @ rotation_increment(vector))
    return np.stack(result)


def orientation_errors(estimated, reference):
    estimated, reference = np.asarray(estimated), np.asarray(reference)
    if estimated.shape != reference.shape or estimated.ndim != 3 or estimated.shape[1:] != (3, 3):
        raise ValueError('paired rotations required')
    relative = np.einsum('nji,njk->nik', reference, estimated)
    skew = np.stack([relative[:, 2, 1] - relative[:, 1, 2], relative[:, 0, 2] - relative[:, 2, 0],
                     relative[:, 1, 0] - relative[:, 0, 1]], axis=1) / 2
    angle = np.arctan2(np.linalg.norm(skew, axis=1), (np.trace(relative, axis1=1, axis2=2) - 1) / 2)
    delta = np.arctan2(estimated[:, 1, 0], estimated[:, 0, 0]) - np.arctan2(reference[:, 1, 0], reference[:, 0, 0])
    yaw = np.arctan2(np.sin(delta), np.cos(delta))
    return {'rotation_error_mean_rad': float(angle.mean()), 'rotation_error_max_rad': float(angle.max()),
            'heading_error_mae_rad': float(np.abs(yaw).mean()), 'heading_error_max_rad': float(np.abs(yaw).max()),
            'terminal_signed_heading_error_rad': float(yaw[-1]), 'terminal_rotation_error_rad': float(angle[-1])}
