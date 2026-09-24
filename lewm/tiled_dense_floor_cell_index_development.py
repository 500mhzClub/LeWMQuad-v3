"""Tile dense triangle arithmetic while retaining the original full-grid gates."""
from types import MappingProxyType
import numpy as np
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.causal_sensor_state import SensorContractError

ROW_BATCH=32


def observed_floor_cell_index(depth, valid, up):
    """Immutable summed invalid-cell index for the measured two-triangle mesh."""
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
    for start in range(0, 479, ROW_BATCH):
        end = min(start+ROW_BATCH, 479)
        a, b, c, e = points[start:end, :-1], points[start:end, 1:], points[start+1:end+1, 1:], points[start+1:end+1, :-1]
        cell_good = good[start:end]
        for left, right in ((b, e), (b, c), (c, e)):
            normal = np.cross(left - a, right - a)
            length = np.linalg.norm(normal, axis=2)
            cell_good &= (length > 1e-10) & (np.abs(normal @ up) >= .97 * length)
            if left is b and right is e:
                # Same fourth-point planarity gate as patch_relation.
                cell_good &= np.abs(np.sum((c - a) * normal, axis=2)) <= .003 * length
    prefix = np.zeros((480, 640), np.int64)
    prefix[1:, 1:] = (~good).cumsum(axis=0).cumsum(axis=1)
    result = {'ground_cells': good.copy(), 'invalid_cell_prefix': prefix, 'up': up.copy()}
    for array in result.values(): array.flags.writeable = False
    return MappingProxyType(result)

