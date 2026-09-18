"""Explicit single-observation reuse of exact measured floor-index inputs.

No module-global cache, function replacement, numerical change or persisted
sensor result. Keys use exact input bytes, not hashes or rounded poses.
"""
from types import MappingProxyType
import numpy as np
from lewm.floor_footprint_bounds_development import observed_floor_cell_index


class FrameFloorIndexCache:
    def __init__(self):
        self._entries = {}
        self.closed = False
        self.hits = self.misses = self.uncached = 0

    def index(self, depth, valid, up):
        if self.closed:
            raise ValueError('floor-index observation scope already closed')
        d, v, u = np.asarray(depth), np.asarray(valid), np.asarray(up, dtype=float)
        # Let the original validator reject malformed requests; do not retain
        # their bytes or bypass its validation with a previous successful key.
        if (d.shape != (480, 640) or v.shape != d.shape or v.dtype != bool
                or u.shape != (3,) or d.dtype.hasobject):
            return observed_floor_cell_index(depth, valid, up)
        key = (d.dtype.str, d.shape, d.tobytes(), v.dtype.str, v.shape, v.tobytes(), u.tobytes())
        if key in self._entries:
            self.hits += 1
            return self._entries[key]
        self.misses += 1
        result = observed_floor_cell_index(depth, valid, up)
        if len(self._entries) >= 8:
            self.uncached += 1
            return result
        # Immutable byte backing prevents consumers from re-enabling writeability.
        frozen = MappingProxyType({k: np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)
            for k, a in result.items()})
        self._entries[key] = frozen
        return frozen

    def close(self):
        self._entries.clear()
        self.closed = True

    def counts(self):
        return dict(hits=self.hits, misses=self.misses, uncached=self.uncached)
