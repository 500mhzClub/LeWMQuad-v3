"""Equivalent occupied-cell grouping, isolated from existing navigation runs.

Only the obstacle extractor's NumPy binding changes. Camera sampling, plane
selection, obstacle thresholds, cropping and dispatch rules stay inherited.
This candidate is not installed by the short-pulse comparison launcher.
"""
from types import SimpleNamespace

import numpy as np

from lewm.eligible_floor_registration_development import bind
from lewm.fine_obstacle_round_trip_development import FineDepthObstacles
from lewm.partial_height_round_trip_development import _partial_observe
from lewm.auxiliary_only_turn_recovery_development import auxiliary_only_obstacles
from lewm.gyro_conditioned_partial_floor_consumers_development import (
    DETAILS, GyroConditionedAuxiliaryObstacles)
from lewm.gyro_conditioned_partial_floor_candidates_development import (
    GyroConditionedPartialFloorCandidates)


def integer_xy_unique(a, *args, **kwargs):
    """Match np.unique, using scalar keys only for bounded signed XY rows."""
    if (args or kwargs != {'axis': 0} or not isinstance(a, np.ndarray)
            or a.ndim != 2 or a.shape[1] != 2 or a.dtype.kind != 'i' or not len(a)):
        return np.unique(a, *args, **kwargs)
    low, high = a.min(axis=0), a.max(axis=0)
    width = int(high[1])-int(low[1])+1
    height = int(high[0])-int(low[0])+1
    if (height*width > np.iinfo(np.int64).max
            or max(abs(int(v)) for v in (*low, *high)) > 2**60):
        return np.unique(a, *args, **kwargs)
    shifted = a.astype(np.int64)-low
    keys = shifted[:, 0]*width+shifted[:, 1]
    values = np.unique(keys)
    return np.column_stack((values//width+low[0], values % width+low[1])).astype(a.dtype)


# Private function globals avoid changing NumPy or other sensor consumers.
_cell_numpy = SimpleNamespace(**(vars(np) | {'unique': integer_xy_unique}))
_initial = bind(FineDepthObstacles._observe, np=_cell_numpy)
_partial = bind(_partial_observe, np=_cell_numpy)
_auxiliary = bind(auxiliary_only_obstacles, np=_cell_numpy)


class PackedCellObstacles(GyroConditionedAuxiliaryObstacles):
    def _observe(self, policy, depth, fast, auxiliary, now):
        selector = GyroConditionedPartialFloorCandidates(depth, auxiliary)
        extract = _initial if self.frames == 0 else _partial
        current = bind(extract, measured_candidates=selector)(
            self, policy, depth, fast, auxiliary, now)
        receipt = self.receipts[-1]
        receipt.update(DETAILS, floor_candidate_selection=selector.receipt,
            obstacle_points_use_original_depth=True)
        if current is None:
            current = _auxiliary(policy, depth, auxiliary, receipt, now_ns=now)
            if current is not None:
                receipt.update(primary_depth_unavailable=True,
                    auxiliary_only_current_obstacles=True,
                    translation_requires_both_cameras=True,
                    missing_primary_rays_inferred_free=False)
        return current
