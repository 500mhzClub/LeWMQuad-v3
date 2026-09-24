"""Reuse exact depth-mesh arithmetic within one observation, across up vectors.

The original validation, array expressions, thresholds and prefix arithmetic
are preserved. No pose rounding, approximate geometry or cross-frame cache.
"""
from types import MappingProxyType

import numpy as np

from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.causal_sensor_state import SensorContractError
from lewm.floor_footprint_bounds_development import observed_floor_cell_index
from lewm.frame_floor_index_cache_development import FrameFloorIndexCache


def _validate(d, valid, up):
    if (d.shape != (480, 640) or valid.shape != d.shape or valid.dtype != bool
            or not np.isfinite(d).all() or np.any(d[~valid] != 0.)
            or np.any((d[valid] < .2) | (d[valid] > 5.)) or up.shape != (3,)
            or not np.isfinite(up).all() or abs(np.linalg.norm(up) - 1) > 1e-6):
        raise SensorContractError('measured depth grid and unit up required')


def _mesh(d, valid):
    transform = np.asarray(BODY_FROM_OPTICAL)
    u = (np.arange(640) + .5 - 320) / FOCAL
    v = (np.arange(480) + .5 - 240) / FOCAL
    optical = np.stack((d * u[None], d * v[:, None], d), axis=2)
    points = optical @ transform[:3, :3].T + transform[:3, 3]
    a, b, c, e = points[:-1, :-1], points[:-1, 1:], points[1:, 1:], points[1:, :-1]
    good = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, 1:] & valid[1:, :-1]
    triangles = []
    for left, right in ((b, e), (b, c), (c, e)):
        normal = np.cross(left - a, right - a)
        length = np.linalg.norm(normal, axis=2)
        planar = None
        if left is b and right is e:
            planar = np.abs(np.sum((c - a) * normal, axis=2)) <= .003 * length
        triangles.append((normal, length, planar))
    # These private arrays own their data and are never returned to consumers.
    # They need no copies of public inputs and survive only this observation.
    points.flags.writeable = False
    good.flags.writeable = False
    for normal, length, planar in triangles:
        normal.flags.writeable = length.flags.writeable = False
        if planar is not None:
            planar.flags.writeable = False
    return points, good, triangles


def _index(mesh, up):
    points, valid_cells, triangles = mesh
    a, b, c, e = points[:-1, :-1], points[:-1, 1:], points[1:, 1:], points[1:, :-1]
    good = valid_cells.copy()
    for p in (a, b, c, e):
        good &= p @ up < -.15
    for normal, length, planar in triangles:
        good &= (length > 1e-10) & (np.abs(normal @ up) >= .97 * length)
        if planar is not None:
            good &= planar
    prefix = np.zeros((480, 640), np.int64)
    prefix[1:, 1:] = (~good).cumsum(axis=0).cumsum(axis=1)
    result = {'ground_cells': good.copy(), 'invalid_cell_prefix': prefix, 'up': up.copy()}
    for array in result.values():
        array.flags.writeable = False
    return MappingProxyType(result)


class ReusedFloorMeshCache(FrameFloorIndexCache):
    """Keep the original exact index keys/counts; additionally reuse mesh work."""

    def __init__(self):
        super().__init__()
        self._meshes = {}

    def index(self, depth, valid, up):
        if self.closed:
            raise ValueError('floor-index observation scope already closed')
        d, v, u = np.asarray(depth), np.asarray(valid), np.asarray(up, dtype=float)
        if (d.shape != (480, 640) or v.shape != d.shape or v.dtype != bool
                or u.shape != (3,) or d.dtype.hasobject):
            return observed_floor_cell_index(depth, valid, up)
        key = (d.dtype.str, d.shape, d.tobytes(), v.dtype.str, v.shape, v.tobytes(), u.tobytes())
        if key in self._entries:
            self.hits += 1
            return self._entries[key]
        self.misses += 1
        _validate(d, v, u)
        mesh_key = key[:-1]
        mesh = self._meshes.get(mesh_key)
        if mesh is None:
            mesh = _mesh(d, v)
            if len(self._meshes) < 8:
                self._meshes[mesh_key] = mesh
        result = _index(mesh, u)
        if len(self._entries) >= 8:
            self.uncached += 1
            return result
        frozen = MappingProxyType({k: np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)
            for k, a in result.items()})
        self._entries[key] = frozen
        return frozen

    def close(self):
        self._meshes.clear()
        super().close()
