"""Fixed hypothetical mounts and evaluator-only startup visibility geometry."""
from itertools import product
import math

import numpy as np

from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL,FOCAL

CANDIDATES=(('forward',(.326,0.,.043),0.),('front_down60',(.326,0.,.043),60.),
    ('front_down90',(.326,0.,.043),90.),('overhead_down90',(0.,0.,.20),90.),
    ('left_outboard_down90',(0.,.30,.20),90.),('right_outboard_down90',(0.,-.30,.20),90.))
DEPTH_TOLERANCE_M=.001


def checked_transform(T):
    T=np.asarray(T,float)
    if (T.shape!=(4,4) or not np.isfinite(T).all() or
            not np.allclose(T[3],[0,0,0,1],atol=1e-12,rtol=0) or
            not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-10,rtol=0) or
            abs(np.linalg.det(T[:3,:3])-1)>1e-10):
        raise ValueError('finite proper optical-to-world transform required')
    return T


def body_from_optical(name):
    matches=[r for r in CANDIDATES if r[0]==name]
    if len(matches)!=1: raise ValueError('fixed predeclared camera name required')
    _,position,pitch=matches[0]; a=math.radians(pitch); c,s=math.cos(a),math.sin(a)
    rotation=np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])
    T=np.eye(4); T[:3,:3]=rotation@np.asarray(BODY_FROM_OPTICAL)[:3,:3]; T[:3,3]=position
    return T


def floor_depth(T):
    T=checked_transform(T); v,u=np.mgrid[:480,:640]
    rays=np.stack(((u+.5-320)/FOCAL,(v+.5-240)/FOCAL,np.ones_like(u)),axis=-1)@T[:3,:3].T
    z=np.zeros((480,640)); down=rays[:,:,2]<-1e-12
    z[down]=-T[2,3]/rays[:,:,2][down]
    valid=down&(z>=.2)&(z<=5.)
    return z,valid


def classify_depth(visible,background,T):
    visible,background=np.asarray(visible),np.asarray(background)
    if visible.shape!=(480,640) or background.shape!=visible.shape:
        raise ValueError('paired native optical-depth rasters required')
    expected,ray_valid=floor_depth(T)
    available=np.isfinite(visible)&(visible>=.2)&(visible<=5.)
    unobstructed_ground=ray_valid&np.isfinite(background)&(abs(background-expected)<=DEPTH_TOLERANCE_M)
    floor=unobstructed_ground&available&(abs(visible-expected)<=DEPTH_TOLERANCE_M)
    # Raw near returns must count as occlusion even if invalid as a depth sensor.
    occluded=unobstructed_ground&np.isfinite(visible)&(visible>0)&(visible<background-DEPTH_TOLERANCE_M)
    if np.any(floor&occluded): raise ValueError('floor and robot occlusion labels overlap')
    return dict(floor=floor,self_occluded=occluded,background_floor=unobstructed_ground,
        unavailable=~available,other=~floor&~occluded&available)


def project(points,T):
    T=checked_transform(T)
    p=(np.asarray(points,float)-T[:3,3])@T[:3,:3]; z=p[...,2]
    positive=(z>=.2)&(z<=5.)
    uv=np.zeros(p.shape[:-1]+(2,)); np.divide(p[...,:2]*FOCAL,z[...,None],out=uv,where=positive[...,None])
    uv+=np.array([319.5,239.5])
    valid=positive&(uv[...,0]>=0)&(uv[...,0]<639)&(uv[...,1]>=0)&(uv[...,1]<479)
    return uv,valid


def footprint_visibility(lower_world,upper_world,T,masks):
    lo,hi=np.asarray(lower_world,float),np.asarray(upper_world,float)
    if lo.shape!=(3,) or hi.shape!=(3,) or not np.isfinite([lo,hi]).all() or np.any(lo>hi):
        raise ValueError('finite ordered physical shape bounds required')
    corners=np.array([[x,y,0.] for x,y in product((lo[0],hi[0]),(lo[1],hi[1]))])
    uv,good=project(corners,T)
    result=dict(complete_frustum=bool(good.all()),complete_rectangle_floor=False,
        rectangle_pixels=0,rectangle_floor_pixels=0,rectangle_self_occluded_pixels=0)
    if good.all():
        lower=np.floor(uv.min(0)-1e-9).astype(int); upper=np.ceil(uv.max(0)+1e-9).astype(int)
        if (lower>=0).all() and upper[0]<640 and upper[1]<480:
            x0,y0=lower; x1,y1=upper; region=np.s_[y0:y1+1,x0:x1+1]
            count=int((x1-x0+1)*(y1-y0+1)); n=int(masks['floor'][region].sum())
            result|=dict(rectangle_pixels=count,rectangle_floor_pixels=n,complete_rectangle_floor=n==count,
                rectangle_self_occluded_pixels=int(masks['self_occluded'][region].sum()))
    # Fixed 5x5 samples permit a multi-view DESIGN diagnostic only. Not a bound.
    points=np.array([[x,y,0.] for x in np.linspace(lo[0],hi[0],5) for y in np.linspace(lo[1],hi[1],5)])
    pixels,valid=project(points,T); observed=np.zeros(25,bool)
    for i in np.flatnonzero(valid):
        x,y=np.floor(pixels[i]).astype(int)
        observed[i]=bool(masks['floor'][y:y+2,x:x+2].all())
    return result|dict(sampled_floor_visibility=observed.tolist(),sampled_visibility_is_not_complete_coverage=True)
