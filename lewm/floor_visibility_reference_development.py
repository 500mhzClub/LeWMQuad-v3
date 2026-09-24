"""Privileged EVALUATION/TRAINING labels, never imported by RGB runtime code."""
import math

import numpy as np


def pixel_rays(*,stride=8):
    if type(stride) is not int or stride<1 or 480%stride or 640%stride: raise ValueError('native-grid divisor required')
    rows=np.arange(stride//2,480,stride); columns=np.arange(stride//2,640,stride)
    u,v=np.meshgrid(columns+.5,rows+.5)
    focal=320/math.tan(math.radians(78.323)/2)
    return rows,columns,np.stack(((u-320)/focal,(v-240)/focal,np.ones_like(u)),axis=-1)


def box_depth(origin,directions,box):
    centre=np.asarray(box['centre_xyz'],dtype=float); size=np.asarray(box['size_xyz'],dtype=float)
    yaw=float(box['yaw_rad']); c,s=math.cos(yaw),math.sin(yaw)
    rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    origin=(np.asarray(origin)-centre)@rotation; directions=np.asarray(directions)@rotation
    if centre.shape!=(3,) or size.shape!=(3,) or np.any(size<=0): raise ValueError('positive box dimensions required')
    lower=np.full(directions.shape,-np.inf); upper=np.full(directions.shape,np.inf)
    for axis in range(3):
        moving=np.abs(directions[...,axis])>1e-12
        a=np.divide(-size[axis]/2-origin[axis],directions[...,axis],out=np.zeros(moving.shape),where=moving)
        b=np.divide(size[axis]/2-origin[axis],directions[...,axis],out=np.zeros(moving.shape),where=moving)
        lower[...,axis]=np.where(moving,np.minimum(a,b),-np.inf)
        upper[...,axis]=np.where(moving,np.maximum(a,b),np.inf)
        outside=not (-size[axis]/2<=origin[axis]<=size[axis]/2)
        if outside: upper[...,axis]=np.where(moving,upper[...,axis],-np.inf)
    enter=lower.max(-1); leave=upper.min(-1)
    hit=(leave>=np.maximum(enter,.05))&(enter<=200)
    # A camera/near plane inside an obstacle is marked ambiguous by the caller;
    # it must not be counted as a confidently visible floor ray.
    return np.where(hit,np.maximum(enter,.05),np.inf),hit&(enter<.05)


def visible_floor(world_from_optical,wall_boxes,*,stride=8):
    transform=np.asarray(world_from_optical,dtype=float)
    if (transform.shape!=(4,4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3],[0,0,0,1],atol=1e-12,rtol=0)
            or not np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-8,rtol=0)
            or abs(np.linalg.det(transform[:3,:3])-1)>1e-8): raise ValueError('proper optical pose required')
    rows,columns,optical=pixel_rays(stride=stride)
    origin=transform[:3,3]; direction=optical@transform[:3,:3].T
    if origin[2]<=0: raise ValueError('camera above ground plane required')
    downward=direction[...,2]<-1e-12
    ground=np.divide(-origin[2],direction[...,2],out=np.full(downward.shape,np.inf),where=downward)
    ground=np.where((ground>=.05)&(ground<=200),ground,np.inf)
    wall=np.full(ground.shape,np.inf); ambiguous=np.zeros(ground.shape,dtype=bool)
    for box in wall_boxes:
        distance,uncertain=box_depth(origin,direction,box); wall=np.minimum(wall,distance); ambiguous|=uncertain
    floor=np.isfinite(ground)&(ground<wall-1e-9)
    # Report all rays as well as a fixed one-grid-cell interior; no adaptive
    # exclusion based on whether a prediction happened to be correct.
    interior=np.zeros_like(floor)
    interior[1:-1,1:-1]=(floor[1:-1,1:-1]==floor[:-2,1:-1])&(floor[1:-1,1:-1]==floor[2:,1:-1])&(
        floor[1:-1,1:-1]==floor[1:-1,:-2])&(floor[1:-1,1:-1]==floor[1:-1,2:])
    return {'rows':rows,'columns':columns,'visible_floor':floor,'valid':~ambiguous,
        'interior':interior&~ambiguous,'ground_optical_depth_m':ground}


def confusion(predicted,truth,valid):
    predicted,truth,valid=map(np.asarray,(predicted,truth,valid))
    if predicted.shape!=truth.shape or truth.shape!=valid.shape or any(v.dtype!=bool for v in (predicted,truth,valid)):
        raise ValueError('matching boolean pixel populations required')
    return {'true_positive':int(np.count_nonzero(valid&truth&predicted)),
        'false_positive':int(np.count_nonzero(valid&~truth&predicted)),
        'false_negative':int(np.count_nonzero(valid&truth&~predicted)),
        'true_negative':int(np.count_nonzero(valid&~truth&~predicted))}
