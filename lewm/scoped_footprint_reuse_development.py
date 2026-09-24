"""Reuse pure footprint computations within one owned, synchronous selection.

The caller holds the observed map and geometry fixed throughout this scope.
This is not a concurrent cache or a map-mutation detector. Every returned
receipt has independent containers, while aliases inside each are retained.
"""
import numpy as np

from lewm.frozen_footprint_receipts_development import (
    freeze_footprint, detach_receipts, _ReceiptDict)


class ScopedFootprintReuse:
    MAX_ENTRIES = 18

    def __init__(self, memory, geometry):
        self._memory = memory
        self._geometry = geometry
        self._cache = {}
        self._active = False
        self._closed = False
        self.requests = self.computations = self.hits = 0

    def __enter__(self):
        if self._active or self._closed:
            raise ValueError('fresh nonnested footprint reuse scope required')
        self._active = True
        return self

    def __exit__(self, *args):
        self._cache.clear()
        self._memory = self._geometry = None
        self._active = False
        self._closed = True

    def __getattr__(self, name):
        if not self._active:
            raise ValueError('footprint memory may only be used inside its scope')
        return getattr(self._memory, name)

    def footprint(self, geometry, displacement_body_xy, yaw_rad, *, now_ns, persistent=True):
        if not self._active:
            raise ValueError('footprint query may only be used inside its scope')
        self.requests += 1
        # Do not bypass a latched failure or stale-observation rejection on hits.
        # Other observed state is fixed by the owning synchronous selector.
        current = getattr(self._memory, '_current', None)
        if current is not None:
            current(now_ns)
        key = None
        if (geometry is self._geometry and type(now_ns) is int and type(persistent) is bool
                and type(yaw_rad) is float and type(displacement_body_xy) in (list, tuple, np.ndarray)):
            try:
                xy = np.asarray(displacement_body_xy, dtype=float)
                yaw = np.asarray(yaw_rad, dtype=float)
                if xy.shape == (2,) and yaw.shape == () and np.isfinite(xy).all() and np.isfinite(yaw):
                    key = (xy.tobytes(), yaw.tobytes(), now_ns, persistent)
            except (TypeError, ValueError, OverflowError):
                # The original method retains its own invalid-input behavior.
                pass
        if key is not None and key in self._cache:
            self.hits += 1
            # Reusing the same frozen object here would create aliases between
            # separate public contact receipts after final detachment.
            return freeze_footprint(detach_receipts(self._cache[key]))
        self.computations += 1
        result = freeze_footprint(self._memory.footprint(
            geometry, displacement_body_xy, yaw_rad, now_ns=now_ns, persistent=persistent))
        # Unsupported/custom/cyclic graphs keep the existing forwarding and
        # copying behavior. Errors are never cached. A full cache recomputes.
        if key is not None and type(result) is _ReceiptDict and len(self._cache) < self.MAX_ENTRIES:
            self._cache[key] = result
        return result

    def counts(self):
        return dict(requests=self.requests, computations=self.computations, hits=self.hits,
                    retained_entries=len(self._cache), scope_closed=self._closed)


class ScopedFootprintMap:
    """Keep original bound map methods; substitute only the pure query view."""
    def __init__(self, mapper, surface):
        self._mapper = mapper
        self.surface = surface

    def __getattr__(self, name):
        return getattr(self._mapper, name)
