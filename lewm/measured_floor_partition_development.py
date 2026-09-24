"""Retain every return in measured-floor or non-floor/unknown bound indices."""
import numpy as np
from lewm.measured_sample_bounds_development import MeasuredSampleBoundsIndex
from lewm.observed_floor_waypoint_development import CELL_M
from lewm.observed_geometry_refinement_development import segment_cell_distances


FOOT_IDS=('FL_foot:0','FR_foot:0','RL_foot:0','RR_foot:0')


class MeasuredFloorPartition:
    def __init__(self):
        self.floor=MeasuredSampleBoundsIndex();self.other=MeasuredSampleBoundsIndex()
        self.total_returns=self.floor_returns=self.other_returns=0

    def insert(self,points,measured_floor,witness):
        p=np.asarray(points,float);mask=np.asarray(measured_floor)
        if (p.ndim!=2 or p.shape[1:]!=(3,) or len(p)>19200 or not np.isfinite(p).all()
                or np.any(np.abs(p)>50.) or mask.shape!=(len(p),) or mask.dtype!=bool):
            raise ValueError('bounded returns and explicit per-return measured-floor classification required')
        self.floor.insert(p[mask],witness);self.other.insert(p[~mask],witness)
        self.total_returns+=len(p);self.floor_returns+=int(mask.sum());self.other_returns+=int((~mask).sum())
        assert self.total_returns==self.floor_returns+self.other_returns


def foot_projection_coverage(center_map_xy,radius,floor_cells):
    """Closed nominal disk coverage by previously measured complete floor cells."""
    p=np.asarray(center_map_xy,float)
    if p.shape!=(2,) or not np.isfinite(p).all() or np.max(np.abs(p))>4.9 or not 0<radius<=.1:
        raise ValueError('bounded nominal foot disk required')
    a=np.floor((p-radius)/CELL_M).astype(int)-1;b=np.floor((p+radius)/CELL_M).astype(int)
    keys=[(x,y) for x in range(a[0],b[0]+1) for y in range(a[1],b[1]+1)]
    distances=segment_cell_distances(p,p,keys)
    touched=[k for k,d in zip(keys,distances,strict=True) if d<=radius+1e-12]
    missing=[k for k in touched if k not in floor_cells]
    return dict(projected_cells=[list(k) for k in touched],unobserved_projected_cells=[list(k) for k in missing],
        entire_nominal_projection_on_measured_floor=not missing,
        terrain_interpolation_or_pose_uncertainty_certified=False,ground_support_approved=False)
