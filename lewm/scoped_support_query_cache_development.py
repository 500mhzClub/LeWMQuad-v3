"""Reuse identical pure support queries only within one footprint evaluation."""
import numpy as np
from lewm.receipt_copy_development import copy_receipt


class SupportQueryCache:
    """The supplied geometry definition stays fixed throughout the owned scope.

    Return fresh receipt containers on every call. Never retain a query across
    candidate footprints or observations, and never cache failed computations.
    """
    def __init__(self, geometry):
        self._target = geometry; self._cache = {}; self._active = False; self._closed = False
        self.requests = 0; self.computations = 0; self.hits = 0

    def __enter__(self):
        if self._active or self._closed: raise ValueError('fresh nonnested support query scope required')
        self._active = True
        return self

    def __exit__(self, *args):
        self._cache.clear(); self._target = None; self._active = False; self._closed = True

    def __getattr__(self, name):
        if not self._active: raise ValueError('support geometry may only be used inside its scope')
        return getattr(self._target, name)

    def supports(self, joint_position, directions_body):
        if not self._active: raise ValueError('support query may only be used inside its scope')
        # Match the original numeric conversion; byte keys distinguish signed
        # zero and every exact posture/orientation. No quantization or tolerance.
        normals = np.asarray(directions_body, dtype=float)
        joints = np.asarray(joint_position, dtype=float)
        key = (joints.shape, joints.tobytes(), normals.shape, normals.tobytes())
        self.requests += 1
        if key in self._cache:
            self.hits += 1
            return copy_receipt(self._cache[key])
        self.computations += 1
        result = self._target.supports(joint_position, directions_body)
        # A bounded cache miss always retains the original computation. Capacity
        # never changes a decision or creates a new rejection.
        if len(self._cache) < 8: self._cache[key] = result
        return copy_receipt(result)

    def counts(self):
        return dict(requests=self.requests, computations=self.computations, hits=self.hits,
            retained_entries=len(self._cache), scope_closed=self._closed)
