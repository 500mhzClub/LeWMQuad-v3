"""Compile the same measured-quad predicates without full-image temporaries."""
import numpy as np
from numba import njit
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.causal_sensor_state import SensorContractError
from lewm.floor_pose_registration_development import (
    ROWS, COLUMNS, unit, proper, measured_candidates as dense_candidates)
from lewm.sampled_plane_candidates_development import CELL_ROWS, CELL_COLUMNS

U = (np.arange(640)+.5-320)/FOCAL
W = (np.arange(480)+.5-240)/FOCAL
EPS = 64*np.finfo(float).eps


@njit(inline='always', fastmath=False)
def _point(depth, row, column):
    z = np.float64(depth[row, column])
    return z+.326, -(z*U[column]), -(z*W[row])+.043


@njit(fastmath=False)
def quad_mask(depth, valid, up):
    good = np.zeros((len(CELL_ROWS), len(CELL_COLUMNS)), dtype=np.bool_)
    for i in range(len(CELL_ROWS)):
        row = CELL_ROWS[i]
        for j in range(len(CELL_COLUMNS)):
            col = CELL_COLUMNS[j]
            points = (_point(depth,row,col), _point(depth,row,col+1),
                _point(depth,row+1,col+1), _point(depth,row+1,col))
            offsets = ((0,0),(0,1),(1,1),(1,0))
            accepted = True
            for k in range(4):
                dr, dc = offsets[k]
                if not valid[row+dr,col+dc]:
                    accepted = False; break
                p = points[k]
                height = p[0]*up[0]+p[1]*up[1]+p[2]*up[2]
                if abs(height+.15) <= EPS:
                    return good, True
                if height >= -.15:
                    accepted = False; break
            if not accepted:
                continue
            a, b, c, e = points
            for pair in range(3):
                left, right = (b,e) if pair==0 else (b,c) if pair==1 else (c,e)
                x0,x1,x2 = left[0]-a[0],left[1]-a[1],left[2]-a[2]
                y0,y1,y2 = right[0]-a[0],right[1]-a[1],right[2]-a[2]
                n0,n1,n2 = x1*y2-x2*y1,x2*y0-x0*y2,x0*y1-x1*y0
                length = np.sqrt(n0*n0+n1*n1+n2*n2)
                alignment = abs(n0*up[0]+n1*up[1]+n2*up[2])
                if abs(length-1e-10)<=EPS*1e-10 or abs(alignment-.97*length)<=EPS*length:
                    return good, True
                if length<=1e-10 or alignment<.97*length:
                    accepted=False; break
                if pair==0:
                    error=abs((c[0]-a[0])*n0+(c[1]-a[1])*n1+(c[2]-a[2])*n2)
                    if abs(error-.003*length)<=EPS*length:
                        return good, True
                    if error>.003*length:
                        accepted=False; break
            good[i,j]=accepted
    return good, False


def measured_candidates(depth, valid, body_from_optical, up_body):
    E = np.asarray(body_from_optical, float)
    if (E.shape != (4,4) or not np.isfinite(E).all()
            or not np.array_equal(E[3],[0.,0.,0.,1.])):
        raise ValueError('finite rigid camera mount required')
    proper(E[:3,:3]);up=unit(up_body);T=np.asarray(BODY_FROM_OPTICAL)
    if not np.array_equal(T,((0.,0.,1.,.326),(-1.,0.,0.,0.),(0.,-1.,0.,.043),(0.,0.,0.,1.))):
        raise ValueError('compiled projection requires the declared optical adapter')
    reference_up=(E[:3,:3]@T[:3,:3].T).T@up
    d,v=np.asarray(depth),np.asarray(valid)
    if (d.shape!=(480,640) or v.shape!=d.shape or v.dtype!=bool
            or not np.isfinite(d).all() or np.any(d[~v]!=0.)
            or np.any((d[v]<.2)|(d[v]>5.)) or not np.isfinite(reference_up).all()
            or abs(np.linalg.norm(reference_up)-1)>1e-6):
        raise SensorContractError('measured depth grid and unit up required')
    good,boundary=quad_mask(d,v,reference_up)
    if boundary:
        return dense_candidates(depth,valid,E,up)
    accepted=good.reshape(len(ROWS),2,len(COLUMNS),2).all(axis=(1,3))
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            accepted &= v[np.ix_(ROWS+dr,COLUMNS+dc)]
    yy,xx=np.meshgrid(ROWS+.5,COLUMNS+.5,indexing='ij');z=d[np.ix_(ROWS,COLUMNS)]
    optical=np.stack((z*(xx-320)/FOCAL,z*(yy-240)/FOCAL,z),axis=-1)
    candidates=optical@E[:3,:3].T+E[:3,3]
    accepted &= candidates@up<-.15
    return candidates[accepted],accepted


def warmup():
    for dtype in (np.float32,np.float64):
        quad_mask(np.zeros((480,640),dtype=dtype),np.zeros((480,640),bool),np.array([0.,0.,1.]))
