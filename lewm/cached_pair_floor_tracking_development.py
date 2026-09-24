"""Reuse identical paired-floor fits within one observation, for replay study."""
from copy import deepcopy
from types import FunctionType
import numpy as np
from lewm.batched_consensus_tracking_development import BatchedConsensusPose, BatchedConsensusMotion
from lewm.gyro_coherent_floor_tracking_development import GyroCoherentFloorPose
from lewm.gyro_coherent_floor_constraint_development import fit_pair


class ObservationPairFits:
    """Private cache for the tracker's owned, read-only raw floor arrays.

    Keep owners alive until clear, and key every other fit input by exact
    value and shape. Callers receive independent receipts because the tracker
    adds acquisition metadata to each returned constraint.
    """
    def __init__(self):
        self.clear()

    def clear(self):
        self.entries = {}
        self.hits = self.misses = 0

    def __call__(self, reference, current, gyro, *, reference_pool_count,
                 current_pool_count, reference_up, current_up,
                 minimum_second_eigenvalue_m2):
        if any(p.flags.writeable or not p.flags.owndata for p in (reference, current)):
            raise ValueError('owned read-only raw floor arrays required')

        def exact_array(value):
            array = np.asarray(value, float)
            return array.shape, array.tobytes()

        key = (id(reference), id(current), exact_array(gyro),
            type(reference_pool_count), reference_pool_count,
            type(current_pool_count), current_pool_count,
            exact_array(reference_up), exact_array(current_up),
            type(minimum_second_eigenvalue_m2),
            float(minimum_second_eigenvalue_m2).hex())
        if key in self.entries:
            self.hits += 1
            return deepcopy(self.entries[key][2])
        self.misses += 1
        result = fit_pair(reference, current, gyro,
            reference_pool_count=reference_pool_count, current_pool_count=current_pool_count,
            reference_up=reference_up, current_up=current_up,
            minimum_second_eigenvalue_m2=minimum_second_eigenvalue_m2)
        self.entries[key] = (reference, current, deepcopy(result))
        return result


class CachedPairFloorPose(BatchedConsensusPose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pair_fits = ObservationPairFits()
        original = GyroCoherentFloorPose._refine_candidate
        # Preserve the original super() closure and all candidate refinements.
        copied = FunctionType(original.__code__,
            original.__globals__ | dict(fit_pair=self._pair_fits),
            original.__name__, original.__defaults__, original.__closure__)
        copied.__kwdefaults__ = original.__kwdefaults__
        self._refine_candidate = copied.__get__(self, type(self))

    def _raw_floor(self, features, receipt):
        points = super()._raw_floor(features, receipt)
        points.flags.writeable = False
        return points

    def observe(self, *args, **kwargs):
        self._pair_fits.clear()
        try:
            return super().observe(*args, **kwargs)
        finally:
            self.last_pair_fit_cache = dict(hits=self._pair_fits.hits,
                misses=self._pair_fits.misses, entries=len(self._pair_fits.entries))
            self._pair_fits.clear()


class CachedPairFloorMotion(BatchedConsensusMotion):
    def __init__(self, *, identity=(0, 0, 0), activation_frame=0):
        super().__init__(identity=identity, activation_frame=activation_frame)
        self.model = CachedPairFloorPose(activation_frame=activation_frame)
