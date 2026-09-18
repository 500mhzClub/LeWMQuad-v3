"""Exact bounded voxel grouping with independent per-cell bounds storage."""
from copy import deepcopy
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.joint_visual_surface_memory_development import MAX_COORDINATE_M, MAX_VOXELS, VOXEL_M
from lewm.measured_sample_bounds_development import MeasuredSampleBoundsIndex


def grouped_voxels(points):
    coordinates = np.floor(points/VOXEL_M).astype(np.int64)
    # The frozen +/-50m, 25mm grid lies inside signed 12-bit coordinates.
    # Biased big-endian packing preserves the original lexicographic ordering.
    if np.any(coordinates < -2048) or np.any(coordinates >= 2048):
        raise SensorContractError('declared signed 12-bit voxel coordinates required')
    biased = coordinates+2048
    packed = (biased[:,0] << 24) | (biased[:,1] << 12) | biased[:,2]
    unique, inverse, counts = np.unique(packed, return_inverse=True, return_counts=True)
    coordinates = np.column_stack((unique >> 24, (unique >> 12) & 4095, unique & 4095))-2048
    return coordinates, inverse, counts


class PackedOwnedMeasuredSampleBoundsIndex(MeasuredSampleBoundsIndex):
    def insert(self, points, witness):
        p = np.asarray(points, float)
        if (p.ndim != 2 or p.shape[1:] != (3,) or len(p) > 19200
                or not np.isfinite(p).all() or np.any(np.abs(p) > MAX_COORDINATE_M)):
            raise SensorContractError('bounded finite sampled surfaces required')
        coordinates, inverse, counts = grouped_voxels(p)
        keys = [tuple(map(int,k)) for k in coordinates]
        new = [k for k in keys if k not in self.cells]
        if len(self.cells)+len(new) > MAX_VOXELS:
            raise SensorContractError('surface memory capacity exhausted; no evidence eviction')
        for key in new:
            self.cells[key] = deepcopy(witness)
        lower = np.full((len(keys),3),np.inf); upper = np.full((len(keys),3),-np.inf)
        np.minimum.at(lower,inverse,p); np.maximum.at(upper,inverse,p)
        lower = np.nextafter(lower,-np.inf); upper = np.nextafter(upper,np.inf)
        bounds = np.stack((lower,upper),axis=1)
        present = [i for i,k in enumerate(keys) if k in self.bounds]
        if present:
            old = np.stack([self.bounds[keys[i]] for i in present])
            bounds[present,0] = np.minimum(old[:,0],bounds[present,0])
            bounds[present,1] = np.maximum(old[:,1],bounds[present,1])
        for i,key in enumerate(keys):
            self.bounds[key] = bounds[i].copy()
            self.sample_counts[key] = self.sample_counts.get(key,0)+int(counts[i])
            self.latest_frames[key] = witness['frame']
