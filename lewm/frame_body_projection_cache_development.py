"""Bounded observation-local reuse of the exact mapping projection arithmetic.

This does not cache up vectors, floor classifications, heights or feasibility.
Callers remain responsible for their existing depth and pose validation.
"""
import numpy as np

from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL


def project_body(depth):
    """Preserve multiply-then-divide order from FloorFrameGeometry exactly."""
    yy, xx = np.indices((480, 640))
    transform = np.asarray(BODY_FROM_OPTICAL)
    optical = np.stack((depth*(xx+.5-320)/FOCAL, depth*(yy+.5-240)/FOCAL, depth), axis=-1)
    return optical@transform[:3, :3].T+transform[:3, 3]


class FrameBodyProjectionCache:
    MAX_ENTRIES = 2

    def __init__(self):
        self._entries = {}
        self.closed = False
        self.hits = self.misses = self.uncached = 0

    def body(self, depth):
        if self.closed:
            raise ValueError('body-projection observation scope already closed')
        # Unsupported representations use the original expression directly.
        # No input is validated or made acceptable by this component.
        if (type(depth) is not np.ndarray or depth.shape != (480, 640)
                or depth.dtype.kind not in 'biuf'):
            self.uncached += 1
            return project_body(depth)
        transform = np.asarray(BODY_FROM_OPTICAL)
        key = (depth.dtype.str, depth.shape, depth.tobytes(), transform.dtype.str,
            transform.shape, transform.tobytes(), FOCAL)
        if key in self._entries:
            self.hits += 1
            return self._entries[key]
        self.misses += 1
        value = project_body(depth)
        if len(self._entries) >= self.MAX_ENTRIES:
            self.uncached += 1
            return value
        # Byte backing prevents consumers from re-enabling writeability.
        frozen = np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
        self._entries[key] = frozen
        return frozen

    def close(self):
        self._entries.clear()
        self.closed = True

    def counts(self):
        return dict(hits=self.hits, misses=self.misses, uncached=self.uncached)
