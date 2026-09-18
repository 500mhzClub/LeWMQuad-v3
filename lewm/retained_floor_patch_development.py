"""Causal complete-foot coverage from retained measured pixel patches.

Each positive result requires one entire enclosing nominal foot square to pass
all existing floor-pixel tests in a single past/current observation. No carving,
union interpolation, pose calibration or physical support certification.
"""
from copy import deepcopy
import numpy as np
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL,FOCAL
from lewm.floor_footprint_bounds_development import observed_floor_cell_index
from lewm.joint_rgbd_rigid_pose_development import proper

T=np.asarray(BODY_FROM_OPTICAL)


class RetainedFloorPatches:
    def __init__(self):self.frames=[]

    def append(self,depth,valid,rotation_map_from_body,position_map,floor_height,witness):
        R=proper(rotation_map_from_body);p=np.asarray(position_map,float)
        if (p.shape!=(3,) or not np.isfinite(p).all() or not np.isfinite(floor_height)
                or type(witness['frame']) is not int or witness['frame']!=len(self.frames) or len(self.frames)>=4096
                or (self.frames and (floor_height!=self.frames[0]['floor_height'] or
                    witness['measured_ns']-self.frames[-1]['witness']['measured_ns']!=100_000_000))):
            raise ValueError('bounded uninterrupted patch history and fixed measured floor required')
        index=observed_floor_cell_index(depth,valid,R[2]);yy,xx=np.indices((480,640))
        optical=np.stack((depth*(xx+.5-320)/FOCAL,depth*(yy+.5-240)/FOCAL,depth),axis=-1)
        heights=(optical@T[:3,:3].T+T[:3,3])@R[2]+p[2]
        near=np.abs(heights-floor_height)<=.01
        good=index['ground_cells']&near[:-1,:-1]&near[:-1,1:]&near[1:,:-1]&near[1:,1:]
        # At most 479*639 invalid pixels: exact int32 sums cannot overflow.
        prefix=np.zeros((480,640),np.int32);prefix[1:,1:]=(~good).cumsum(0,dtype=np.int32).cumsum(1,dtype=np.int32)
        prefix.flags.writeable=False
        self.frames.append(dict(R=R.copy(),p=p.copy(),floor_height=float(floor_height),prefix=prefix,witness=deepcopy(witness)))

    def coverage(self,centres,radius=.022):
        xy=np.asarray(centres,float)
        if (xy.ndim!=2 or xy.shape[1:]!=(2,) or len(xy)>128 or not np.isfinite(xy).all()
                or (xy.size and np.max(np.abs(xy))>4.9) or not np.isfinite(radius) or not 0<radius<=.1):
            raise ValueError('bounded nominal foot centres and radius required')
        found=[None]*len(xy)
        for frame in self.frames:
            ids=np.asarray([i for i,v in enumerate(found) if v is None],int)
            if not len(ids):break
            corners=xy[ids,None,:]+radius*np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
            world=np.concatenate((corners,np.full((*corners.shape[:-1],1),frame['floor_height'])),axis=-1)
            points=(world-frame['p'])@frame['R'];camera=(points-T[:3,3])@T[:3,:3];z=camera[...,2]
            uv=camera[...,:2]/np.maximum(z[...,None],1e-12)*FOCAL+[319.5,239.5]
            lo=uv.min(1);hi=uv.max(1);margin=1e-9+64*np.finfo(float).eps*np.maximum(np.abs(lo),np.abs(hi))
            lo-=margin;hi+=margin
            visible=((z>=.2)&(z<=5.)).all(1)&(lo>=0).all(1)&(hi<[639,479]).all(1)
            a=np.floor(np.clip(lo,-1,640)).astype(int);b=np.floor(np.clip(hi,-1,640)).astype(int)
            prefix=frame['prefix']
            for j in np.flatnonzero(visible):
                x0,y0=a[j];x1,y1=b[j]+1
                bad=int(prefix[y1,x1]-prefix[y0,x1]-prefix[y1,x0]+prefix[y0,x0])
                if bad==0:
                    found[int(ids[j])]=dict(witness=deepcopy(frame['witness']),pixel_rectangle=[a[j].tolist(),b[j].tolist()],
                        nominal_enclosing_square_side_m=2*radius,all_projected_pixel_quads_measured_floor=True)
        return [dict(complete_nominal_foot_patch=v is not None,coverage_witness=v,retained_frames=len(self.frames),
            no_unobserved_pixel_inference=True,pose_or_terrain_uncertainty_certified=False,ground_support_approved=False) for v in found]
