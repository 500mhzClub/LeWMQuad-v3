"""Group measured voxel bounds by stable sort and contiguous reductions."""
from copy import deepcopy
import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.joint_visual_surface_memory_development import MAX_COORDINATE_M, MAX_VOXELS, VOXEL_M
from lewm.single_pass_sample_bounds_development import SinglePassMeasuredSampleBoundsIndex


def grouped_bounds(points):
    coordinates = np.floor(points/VOXEL_M).astype(np.int64)
    if np.any(coordinates < -2048) or np.any(coordinates >= 2048):
        raise SensorContractError('declared signed 12-bit voxel coordinates required')
    biased = coordinates+2048
    packed = (biased[:, 0]<<24) | (biased[:, 1]<<12) | biased[:, 2]
    order = np.argsort(packed, kind='stable')
    ordered = packed[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]]) if len(ordered) else np.empty(0, int)
    unique = ordered[starts]
    keys = np.column_stack((unique>>24, (unique>>12)&4095, unique&4095))-2048
    counts = np.diff(np.r_[starts, len(points)])
    # Stability retains original point order even at signed-zero ties.
    lower = np.minimum.reduceat(points[order], starts, axis=0) if len(starts) else np.empty((0, 3))
    upper = np.maximum.reduceat(points[order], starts, axis=0) if len(starts) else np.empty((0, 3))
    return keys, counts, np.nextafter(lower, -np.inf), np.nextafter(upper, np.inf)


class SortedSampleBoundsIndex(SinglePassMeasuredSampleBoundsIndex):
    def insert(self, points, witness):
        p = np.asarray(points, float)
        if (p.ndim != 2 or p.shape[1:] != (3,) or len(p) > 19200
                or not np.isfinite(p).all() or np.any(np.abs(p) > MAX_COORDINATE_M)):
            raise SensorContractError('bounded finite sampled surfaces required')
        coordinates, counts, lower, upper = grouped_bounds(p)
        keys = [tuple(map(int, k)) for k in coordinates]
        new = [k for k in keys if k not in self.cells]
        if len(self.cells)+len(new) > MAX_VOXELS:
            raise SensorContractError('surface memory capacity exhausted; no evidence eviction')
        for key in new: self.cells[key] = deepcopy(witness)
        bounds = np.stack((lower, upper), axis=1)
        present = [i for i, k in enumerate(keys) if k in self.bounds]
        if present:
            old = np.stack([self.bounds[keys[i]] for i in present])
            bounds[present, 0] = np.minimum(old[:, 0], bounds[present, 0])
            bounds[present, 1] = np.maximum(old[:, 1], bounds[present, 1])
        for i, key in enumerate(keys):
            self.bounds[key] = bounds[i].copy()
            self.sample_counts[key] = self.sample_counts.get(key, 0)+int(counts[i])
            self.latest_frames[key] = witness['frame']
