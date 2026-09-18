"""Express body-XY forecasts in the mission's initial-frame XY metric.

The existing model predicts body XY, not body Z. This projection assumes zero
displacement along the gravity-aligned map's vertical axis. It uses measured
orientations only and does not change the mission target or arrival radius.
"""
import numpy as np


def position_distance(delta_xy, matrix=None):
    """Distance in body XY by default, or in the supplied position metric."""
    delta = np.asarray(delta_xy)
    if matrix is None:
        # Preserve the original scalar-vector norm's floating-point path.
        return np.linalg.norm(delta) if delta.ndim == 1 else np.linalg.norm(delta, axis=-1)
    if matrix is not None:
        A = np.asarray(matrix, dtype=float)
        if A.shape != (2, 2) or not np.isfinite(A).all():
            raise ValueError('finite 2x2 position metric required')
        delta = delta @ A.T
    return np.linalg.norm(delta, axis=-1)


def planar_body_to_initial_xy(map_from_initial, body_to_map):
    """Return A such that initial displacement XY = A @ predicted body XY.

    A body-Z displacement generally accompanies body-XY motion on a horizontal
    plane when the robot pitches or rolls. Eliminate that unknown using the
    map-height constraint before projecting into the mission frame.
    """
    B = np.asarray(map_from_initial, dtype=float)
    Q = np.asarray(body_to_map, dtype=float)
    if B.shape != (3, 3) or Q.shape != (3, 3) or not np.isfinite([B, Q]).all():
        raise ValueError('finite measured 3x3 orientations required')
    if abs(Q[2, 2]) <= 1e-8:
        raise ValueError('body XY cannot parameterize the gravity-horizontal plane')
    R = B.T @ Q
    return R[:2, :2] - np.outer(R[:2, 2], Q[2, :2] / Q[2, 2])
