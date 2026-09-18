"""Single broad-phase enumeration with exact measured-bound query receipts."""
from copy import deepcopy
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.joint_visual_surface_memory_development import MAX_COORDINATE_M, VOXEL_M
from lewm.packed_owned_sample_bounds_development import PackedOwnedMeasuredSampleBoundsIndex


class SinglePassMeasuredSampleBoundsIndex(PackedOwnedMeasuredSampleBoundsIndex):
    def _intersect(self, lower, upper, sphere=None):
        low, high = np.asarray(lower, float), np.asarray(upper, float)
        if (low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all()
                or np.any(low > high) or np.any(np.abs([low, high]) > MAX_COORDINATE_M)):
            raise SensorContractError('finite bounded ordered query box required')
        a = np.ceil(low/VOXEL_M).astype(int)-1
        b = np.floor(high/VOXEL_M).astype(int)
        if int(np.prod(b-a+1)) <= len(self.cells):
            keys = ((x,y,z) for x in range(a[0],b[0]+1)
                for y in range(a[1],b[1]+1) for z in range(a[2],b[2]+1))
            broad = [k for k in keys if k in self.cells]
        else:
            broad = [k for k in self.cells if all(a[i] <= k[i] <= b[i] for i in range(3))]
        available = [k for k in broad if k in self.bounds]
        hits = []
        if available:
            bounds = np.stack([self.bounds[k] for k in available])
            overlaps = np.all(bounds[:,0] <= high, axis=1) & np.all(bounds[:,1] >= low, axis=1)
            hits = [k for k,overlap in zip(available,overlaps,strict=True) if overlap]
        if sphere is not None:
            c,r = sphere
            # Keep the original scalar norm and tolerance, including its
            # rounding at the sphere boundary; only box comparisons batch.
            hits = [k for k in hits if np.linalg.norm(np.maximum(
                np.maximum(self.bounds[k][0]-c,c-self.bounds[k][1]),0.)) <= r+1e-12]
        key = min(hits) if hits else None
        assert len(hits) <= len(broad)
        return dict(status='POSSIBLE_MEASURED_SAMPLE_BOUNDS_INTERSECTION' if hits else 'UNKNOWN',
            intersecting_voxels=len(hits),first_cell=None if key is None else list(key),
            witness=None if key is None else deepcopy(self.cells[key]),
            first_bounds_m=None if key is None else self.bounds[key].tolist(),
            first_bounds_sample_count=None if key is None else self.sample_counts[key],
            first_bounds_latest_frame=None if key is None else self.latest_frames[key],
            whole_voxel_intersections=len(broad),
            query_geometry='sphere' if sphere is not None else 'axis_aligned_box',
            all_inserted_points_enclosed=True,sample_bounds_are_uncertainty_bounds=False,
            unobserved_surface_coverage=False,free_space_established=False,motion_permitted=False)
