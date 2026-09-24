"""Monotone within-voxel enclosures of all measured returns, without free space.

Original raw observations remain the source of truth. Bounds enclose every
inserted point, with outward floating-point rounding; they are not a sensor or
pose uncertainty envelope and do not cover unobserved surfaces between rays.
"""
from copy import deepcopy
import numpy as np
from lewm.joint_visual_surface_memory_development import SurfaceIndex,VOXEL_M


class MeasuredSampleBoundsIndex(SurfaceIndex):
    def __init__(self):
        super().__init__()
        self.bounds={};self.sample_counts={};self.latest_frames={}

    def insert(self,points,witness):
        super().insert(points,witness)
        p=np.asarray(points,float)
        keys,inverse,counts=np.unique(np.floor(p/VOXEL_M).astype(np.int64),axis=0,return_inverse=True,return_counts=True)
        lower=np.full((len(keys),3),np.inf);upper=np.full((len(keys),3),-np.inf)
        np.minimum.at(lower,inverse,p);np.maximum.at(upper,inverse,p)
        lower=np.nextafter(lower,-np.inf);upper=np.nextafter(upper,np.inf)
        for i,k in enumerate(keys):
            key=tuple(map(int,k))
            if key in self.bounds:
                old=self.bounds[key]
                self.bounds[key]=np.stack((np.minimum(old[0],lower[i]),np.maximum(old[1],upper[i])))
            else:self.bounds[key]=np.stack((lower[i],upper[i]))
            self.sample_counts[key]=self.sample_counts.get(key,0)+int(counts[i])
            self.latest_frames[key]=witness['frame']

    def intersect(self,lower,upper):
        return self._intersect(lower,upper)

    def intersect_sphere(self,center,radius):
        c=np.asarray(center,float)
        if c.shape!=(3,) or not np.isfinite(c).all() or not np.isfinite(radius) or not 0<radius<=1.:
            raise ValueError('bounded finite sphere required')
        return self._intersect(c-radius,c+radius,sphere=(c,radius))

    def _intersect(self,lower,upper,sphere=None):
        broad=super().intersect(lower,upper)
        low,high=np.asarray(lower,float),np.asarray(upper,float)
        a=np.ceil(low/VOXEL_M).astype(int)-1;b=np.floor(high/VOXEL_M).astype(int)
        if int(np.prod(b-a+1))<=len(self.cells):
            keys=((x,y,z) for x in range(a[0],b[0]+1) for y in range(a[1],b[1]+1) for z in range(a[2],b[2]+1))
        else:keys=(k for k in self.cells if all(a[i]<=k[i]<=b[i] for i in range(3)))
        hits=[k for k in keys if k in self.bounds and (self.bounds[k][0]<=high).all() and (self.bounds[k][1]>=low).all()]
        if sphere is not None:
            c,r=sphere
            hits=[k for k in hits if np.linalg.norm(np.maximum(np.maximum(self.bounds[k][0]-c,c-self.bounds[k][1]),0.))<=r+1e-12]
        key=min(hits) if hits else None
        assert len(hits)<=broad['intersecting_voxels']
        return dict(status='POSSIBLE_MEASURED_SAMPLE_BOUNDS_INTERSECTION' if hits else 'UNKNOWN',
            intersecting_voxels=len(hits),first_cell=None if key is None else list(key),
            witness=None if key is None else deepcopy(self.cells[key]),
            first_bounds_m=None if key is None else self.bounds[key].tolist(),
            first_bounds_sample_count=None if key is None else self.sample_counts[key],
            first_bounds_latest_frame=None if key is None else self.latest_frames[key],
            whole_voxel_intersections=broad['intersecting_voxels'],
            query_geometry='sphere' if sphere is not None else 'axis_aligned_box',
            all_inserted_points_enclosed=True,sample_bounds_are_uncertainty_bounds=False,
            unobserved_surface_coverage=False,free_space_established=False,motion_permitted=False)
