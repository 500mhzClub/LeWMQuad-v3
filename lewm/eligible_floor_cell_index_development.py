"""Skip triangle work for cells rejected by unchanged measured-depth gates.

The dense implementation remains the reference, including near numerical
decision boundaries. No threshold is widened and no rejected cell is restored.
"""
from types import MappingProxyType
import numpy as np

from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.causal_sensor_state import SensorContractError
from lewm.floor_footprint_bounds_development import observed_floor_cell_index as dense_index


def observed_floor_cell_index(depth, valid, up):
    d, valid, up = np.asarray(depth), np.asarray(valid), np.asarray(up, dtype=float)
    if (d.shape != (480, 640) or valid.shape != d.shape or valid.dtype != bool
            or not np.isfinite(d).all() or np.any(d[~valid] != 0.)
            or np.any((d[valid] < .2) | (d[valid] > 5.)) or up.shape != (3,)
            or not np.isfinite(up).all() or abs(np.linalg.norm(up) - 1) > 1e-6):
        raise SensorContractError('measured depth grid and unit up required')
    transform = np.asarray(BODY_FROM_OPTICAL)
    u = (np.arange(640) + .5 - 320) / FOCAL
    v = (np.arange(480) + .5 - 240) / FOCAL
    optical = np.stack((d * u[None], d * v[:, None], d), axis=2)
    points = optical @ transform[:3, :3].T + transform[:3, 3]
    a, b, c, e = points[:-1, :-1], points[:-1, 1:], points[1:, 1:], points[1:, :-1]
    good = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, 1:] & valid[1:, :-1]
    for p in (a, b, c, e): good &= p @ up < -.15
    for pair, (left, right) in enumerate(((b, e), (b, c), (c, e))):
        rows, columns = np.nonzero(good)
        if not len(rows): break
        anchor = a[rows, columns]
        normal = np.cross(left[rows, columns]-anchor, right[rows, columns]-anchor)
        length = np.linalg.norm(normal, axis=1)
        alignment = np.abs(normal @ up)
        # Gathering may select a different NumPy dot kernel. Ambiguous last-bit
        # decisions use the original dense path, rather than changing a gate.
        tolerance = 64*np.finfo(float).eps*length
        if (np.any(np.abs(length-1e-10) <= 64*np.finfo(float).eps*1e-10)
                or np.any(np.abs(alignment-.97*length) <= tolerance)):
            return dense_index(depth, valid, up)
        accepted = (length > 1e-10) & (alignment >= .97*length)
        if pair == 0:
            error = np.abs(np.sum((c[rows, columns]-anchor)*normal, axis=1))
            if np.any(np.abs(error-.003*length) <= tolerance):
                return dense_index(depth, valid, up)
            accepted &= error <= .003*length
        good[rows, columns] = accepted
    prefix = np.zeros((480, 640), np.int64)
    prefix[1:, 1:] = (~good).cumsum(axis=0).cumsum(axis=1)
    result = {'ground_cells': good.copy(), 'invalid_cell_prefix': prefix, 'up': up.copy()}
    for array in result.values(): array.flags.writeable = False
    return MappingProxyType(result)
