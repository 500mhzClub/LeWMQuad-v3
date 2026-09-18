"""Continuous nominal clearance and measured floor-return classification.

Neither calculation certifies articulated motion or ground support. Both retain
the supplied observed geometry; no scene labels or native poses enter here.
"""
import math
import numpy as np
from lewm.observed_floor_waypoint_development import CELL_M
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.floor_footprint_bounds_development import observed_floor_cell_index
from lewm.joint_rgbd_rigid_pose_development import proper


def segment_cell_distances(start,end,cells):
    """Euclidean distances between a closed 2-D segment and observed cell squares."""
    a,b=np.asarray(start,float),np.asarray(end,float);keys=np.asarray(cells)
    if a.shape!=(2,) or b.shape!=(2,) or not np.isfinite([a,b]).all() or np.max(np.abs([a,b]))>5.:
        raise ValueError('bounded finite segment required')
    if keys.size==0:return np.empty(0,float)
    cell_limit=int(round(5./CELL_M))
    if keys.ndim!=2 or keys.shape[1]!=2 or keys.dtype.kind not in 'iu' or len(keys)>40000 or np.any(keys < -cell_limit) or np.any(keys>=cell_limit):
        raise ValueError('bounded integer observed cells required')
    low=keys*CELL_M;high=low+CELL_M;delta=b-a;length2=float(delta@delta)
    distances=np.minimum(np.linalg.norm(np.maximum(np.maximum(low-a,a-high),0.),axis=1),
        np.linalg.norm(np.maximum(np.maximum(low-b,b-high),0.),axis=1))
    corners=low[:,None]+np.array([[0,0],[0,1],[1,0],[1,1]])*CELL_M
    t=np.clip(((corners-a)@delta)/length2,0.,1.) if length2 else np.zeros(corners.shape[:2])
    distances=np.minimum(distances,np.linalg.norm(corners-(a+t[...,None]*delta),axis=2).min(axis=1))
    entry=np.zeros(len(keys));leave=np.ones(len(keys));cross=np.ones(len(keys),bool)
    for axis in range(2):
        if delta[axis]==0.:
            cross&=(a[axis]>=low[:,axis])&(a[axis]<=high[:,axis])
        else:
            u=(low[:,axis]-a[axis])/delta[axis];v=(high[:,axis]-a[axis])/delta[axis]
            entry=np.maximum(entry,np.minimum(u,v));leave=np.minimum(leave,np.maximum(u,v))
    distances[cross&(entry<=leave)]=0.
    return distances


def nominal_connector(start,end,cells,*,radius_m=.45):
    if not math.isfinite(radius_m) or not 0<=radius_m<=1.:
        raise ValueError('same bounded nominal radius required')
    distances=segment_cell_distances(start,end,cells)
    nearest=int(np.argmin(distances)) if len(distances) else None
    minimum=None if nearest is None else float(distances[nearest])
    return dict(radius_m=radius_m,minimum_observed_cell_distance_m=minimum,
        nearest_observed_cell=None if nearest is None else list(map(int,cells[nearest])),
        nominal_disk_connector_clear=minimum is None or minimum>radius_m+1e-12,
        numerical_clearance_allowance_m=1e-12,articulated_motion_certified=False,
        unobserved_space_certified=False)


def sampled_floor_patch(depth,valid,rotation_map_from_body,position_map,floor_height,rows,columns):
    """All four adjacent mesh quads plus all nine measured pixel heights required."""
    R=proper(rotation_map_from_body);p=np.asarray(position_map,float)
    rr,cc=np.asarray(rows),np.asarray(columns)
    if (p.shape!=(3,) or not np.isfinite(p).all() or not math.isfinite(floor_height)
            or rr.ndim!=1 or cc.ndim!=1 or rr.dtype.kind not in 'iu' or cc.dtype.kind not in 'iu'
            or np.any(rr<1) or np.any(rr>=479) or np.any(cc<1) or np.any(cc>=639)):
        raise ValueError('finite map pose/plane and interior integer sample pixels required')
    index=observed_floor_cell_index(depth,valid,R[2]);T=np.asarray(BODY_FROM_OPTICAL)
    yy,xx=np.indices((480,640))
    optical=np.stack((depth*(xx+.5-320)/FOCAL,depth*(yy+.5-240)/FOCAL,depth),axis=-1)
    heights=(optical@T[:3,:3].T+T[:3,3])@R[2]+p[2]
    near=valid&(np.abs(heights-floor_height)<=.01)
    good=np.ones((len(rr),len(cc)),bool)
    for dr in (-1,0):
        for dc in (-1,0):good&=index['ground_cells'][np.ix_(rr+dr,cc+dc)]
    for dr in (-1,0,1):
        for dc in (-1,0,1):good&=near[np.ix_(rr+dr,cc+dc)]
    return dict(measured_floor_patch=good,point_map_height_m=heights[np.ix_(rr,cc)],
        plane_height_band_m=.01,ground_support_approved=False,continuous_coverage_established=False)
