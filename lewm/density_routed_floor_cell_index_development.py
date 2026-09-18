"""Choose an exact dense or eligible-cell kernel from a cheap workload estimate.

The estimate affects computation only. Both kernels independently apply the
unchanged full-resolution input checks and measured-floor eligibility gates.
"""
import numpy as np

from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.floor_footprint_bounds_development import observed_floor_cell_index as dense_index
from lewm.eligible_floor_cell_index_development import observed_floor_cell_index as eligible_index

ROWS=np.arange(0,480,16)
COLUMNS=np.arange(0,640,16)


def prefer_dense(depth,valid,up):
    d,v,u=np.asarray(depth),np.asarray(valid),np.asarray(up,dtype=float)
    if (d.shape!=(480,640) or v.shape!=d.shape or v.dtype!=bool
            or d.dtype.kind not in 'fiu' or u.shape!=(3,) or not np.isfinite(u).all()
            or abs(np.linalg.norm(u)-1)>1e-6):
        return True
    z=d[np.ix_(ROWS,COLUMNS)];mask=v[np.ix_(ROWS,COLUMNS)]
    if not np.isfinite(z).all() or np.any(z[mask]<.2) or np.any(z[mask]>5):return True
    transform=np.asarray(BODY_FROM_OPTICAL)
    optical=np.stack((z*((COLUMNS+.5-320)/FOCAL)[None],
        z*((ROWS+.5-240)/FOCAL)[:,None],z),axis=2)
    points=optical@transform[:3,:3].T+transform[:3,3]
    return bool(np.count_nonzero(mask&(points@u<-.15))>.5*mask.size)


def observed_floor_cell_index(depth,valid,up):
    kernel=dense_index if prefer_dense(depth,valid,up) else eligible_index
    return kernel(depth,valid,up)
