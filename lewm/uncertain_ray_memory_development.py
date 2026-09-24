"""Fusion-aware sampled ray evidence with explicit historical pose envelopes.

Envelope radii use uncalibrated development proxies. Pixel coverage is not a
continuous-scene, future-gait, probability, or hardware safety certificate.
"""
from copy import deepcopy
import hashlib
from types import MappingProxyType

import numpy as np

from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL,FOCAL,validate_depth
from lewm.causal_sensor_state import SensorContractError
from lewm.depth_inertial_moment_fusion_development import MomentWeakSubspaceIntegrator
from lewm.observed_turn_region_development import observed_corners
from lewm.ray_rectangle_kernel_development import reduce_rectangles,warm_rectangle_kernel


def depth_evidence(depth,valid,up):
    """Static per-pixel evidence; changing a query radius cannot change normals."""
    d=np.asarray(depth); valid=np.asarray(valid); up=np.asarray(up,dtype=float)
    if (d.shape!=(480,640) or valid.shape!=d.shape or valid.dtype!=bool
            or not np.isfinite(d).all() or np.any(d[~valid]!=0)
            or np.any((d[valid]<.2)|(d[valid]>5.)) or up.shape!=(3,)
            or not np.isfinite(up).all() or abs(np.linalg.norm(up)-1)>1e-6):
        raise SensorContractError('finite calibrated-shape depth, validity and unit up required')
    transform=np.asarray(BODY_FROM_OPTICAL); optical_up=transform[:3,:3].T@up
    u=(np.arange(640)+.5-320)/FOCAL; v=(np.arange(480)+.5-240)/FOCAL
    points=np.stack((d*u[None,:],d*v[:,None],d),axis=-1)
    height=points@optical_up+transform[:3,3]@up
    a,b,c,e=points[:-1,:-1],points[:-1,1:],points[1:,:-1],points[1:,1:]
    normal=np.cross(b-a,c-a); norm=np.linalg.norm(normal,axis=-1)
    normal=np.divide(normal,norm[...,None],out=np.zeros_like(normal),where=norm[...,None]>1e-8)
    good=(valid[:-1,:-1]&valid[:-1,1:]&valid[1:,:-1]&valid[1:,1:]
        &(np.abs(normal@optical_up)>=.97)
        &(np.abs(np.sum((e-a)*normal,axis=-1))<=.003)
        &(height[:-1,:-1]<-.15)&(height[:-1,1:]<-.15)&(height[1:,:-1]<-.15)&(height[1:,1:]<-.15))
    ground=np.zeros(d.shape,bool); ground[:-1,:-1]=good
    tile_shape=(60,8,80,8)
    tiles=np.stack((np.where(valid,d,np.inf).reshape(tile_shape).min(axis=(1,3)),
        np.where(valid,d,-np.inf).reshape(tile_shape).max(axis=(1,3)),
        valid.reshape(tile_shape).all(axis=(1,3)),ground.reshape(tile_shape).all(axis=(1,3)),
        height.reshape(tile_shape).min(axis=(1,3)),height.reshape(tile_shape).max(axis=(1,3)))).astype(np.float64)
    result={'depth':d.copy(),'valid':valid.copy(),'height':height,'ground':ground,'up':up.copy(),'tiles':tiles}
    # Cached extrema and raw arrays are one immutable observation. Accidental
    # mutation must not leave stale summaries that could approve an obstacle.
    for value in result.values(): value.flags.writeable=False
    return MappingProxyType(result)


def query_envelopes(evidence,points_body,radii,ground_roles,*,backend='compiled'):
    """Inspect every sampled pixel in a conservative projected pose-ball box.

    Unknown pixels prevent this view from approving the entire envelope.
    Observed near surfaces remain conflicts even when part of the box is out
    of view or invalid, so increasing radius cannot erase an old conflict.
    """
    if backend not in ('compiled','reference'): raise ValueError('explicit rectangle backend required')
    points=np.asarray(points_body,dtype=float); radii=np.asarray(radii,dtype=float); roles=np.asarray(ground_roles)
    if (points.ndim!=2 or points.shape[1:]!=(3,) or radii.shape!=(len(points),)
            or roles.shape!=(len(points),) or roles.dtype!=bool
            or not np.isfinite(points).all() or not np.isfinite(radii).all() or np.any(radii<0)):
        raise SensorContractError('finite queries, nonnegative radii and explicit support roles required')
    transform=np.asarray(BODY_FROM_OPTICAL)
    optical=(points-transform[:3,3])@transform[:3,:3]
    free=np.zeros(len(points),bool); support=free.copy(); conflict=free.copy(); pixels=np.zeros(len(points),int)
    low=optical[:,2]-radii; high=optical[:,2]+radii
    # Vectorize projection and discard envelopes too close/behind to intersect
    # any valid depth return even with the existing 4-cm conflict margin.
    eligible=np.flatnonzero(high+.04>=.2)
    crossing=low<=0
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        denominators=np.maximum(np.stack((low,high),axis=1),1e-12)
        ux=FOCAL*(optical[:,0,None,None]+radii[:,None,None]*np.array([-1,1])[None,:,None])/denominators[:,None,:]+319.5
        vy=FOCAL*(optical[:,1,None,None]+radii[:,None,None]*np.array([-1,1])[None,:,None])/denominators[:,None,:]+239.5
    lefts=np.floor(np.clip(ux.min(axis=(1,2)),-2,641)).astype(int)
    rights=np.floor(np.clip(ux.max(axis=(1,2)),-2,641)).astype(int)+1
    tops=np.floor(np.clip(vy.min(axis=(1,2)),-2,481)).astype(int)
    bottoms=np.floor(np.clip(vy.max(axis=(1,2)),-2,481)).astype(int)+1
    lefts[crossing]=0; rights[crossing]=639; tops[crossing]=0; bottoms[crossing]=479
    if backend=='compiled':
        heights=points@evidence['up']; heights[~roles]=np.nan
        free,support,conflict,visited,area=reduce_rectangles(evidence['depth'],evidence['valid'],
            evidence['height'],evidence['ground'],heights,radii,low,high,lefts,rights,tops,bottoms,evidence['tiles'])
        return {'free':free,'observed_ground_support':support,'contradictory_or_near_surface':conflict,
                'examined_pixels':visited,'projected_window_pixels':area}
    for i in eligible:
        r=radii[i]; left,right,top,bottom=lefts[i],rights[i],tops[i],bottoms[i]
        complete=low[i]>=.2 and high[i]<=5. and left>=1 and right<=638 and top>=1 and bottom<=478
        left,right=max(0,left),min(639,right); top,bottom=max(0,top),min(479,bottom)
        if right<left or bottom<top: continue
        sl=np.s_[top:bottom+1,left:right+1]; valid=evidence['valid'][sl]; values=evidence['depth'][sl]
        pixels[i]=values.size
        all_valid=bool(complete and valid.all())
        free[i]=bool(all_valid and values.min()-high[i]>=.04)
        heights=evidence['height'][sl]; query_height=points[i]@evidence['up']
        support[i]=bool(all_valid and roles[i] and evidence['ground'][sl].all()
            and np.ptp(heights)<=.006
            and max(abs(query_height-heights.min()),abs(query_height-heights.max()))+r<=.06)
        # Ground remains support, not an obstacle, only when the whole
        # uncertainty envelope satisfies the separate support predicate.
        conflict[i]=bool(np.any(valid&(values>=low[i]-.04)&(values<=high[i]+.04)) and not support[i])
    return {'free':free&~conflict,'observed_ground_support':support&~conflict,
        'contradictory_or_near_surface':conflict,'examined_pixels':pixels,'projected_window_pixels':pixels.copy()}


def transport_radius(points,current,stored):
    if current['measured_ns']==stored['measured_ns']:
        return np.zeros(len(points))  # Identical current-view pose cancels exactly.
    translation=current['position_scale_m']+stored['position_scale_m']
    angle=current['orientation_scale_rad']+stored['orientation_scale_rad']
    lever=np.linalg.norm(points,axis=1)+np.linalg.norm(current['position']-stored['position'])+translation
    return translation+angle*lever


class FusedRayEvidenceMemory:
    """Own fusion computation; never relabel a predicted component as depth."""
    def __init__(self):
        warm_rectangle_kernel()
        self.integrator=MomentWeakSubspaceIntegrator(); self.frames=[]; self.latest_frame=None
        self.last_ns=self.identity=self.up_initial=self.position=self.rotation=self.latest_surface=None
        self.fusion=None; self.failed=False

    def observe(self,policy,depth,relative,*,now_ns):
        if self.failed: raise SensorContractError('fused ray memory fault latched')
        try:
            validate_depth(depth,policy,now_ns=now_ns)
            expected={'measured_ns','local_surfaces','relative_orientation','motion','surface_points',
                'position_initial_body_m','position_is_observed_anchor','arrival_verified','turn_clearance_qualified','scope'}
            if set(relative)!=expected or relative['measured_ns']!=now_ns:
                raise SensorContractError('exact current original depth evidence required')
            surface=relative['local_surfaces']
            if (surface['rgb_sha256']!=depth['rgb_sha256'] or surface['measured_ns']!=now_ns
                    or tuple(surface['identity'])!=tuple(depth['identity'])
                    or surface['depth_sha256']!=hashlib.sha256(depth['depth_m'].tobytes()+depth['valid'].tobytes()).hexdigest()):
                raise SensorContractError('original depth evidence must bind current observations')
            fusion=self.integrator.observe(policy,relative)
            if not fusion['usable_under_declared_proxy_budget']:
                raise SensorContractError('conditional pose-error budget exhausted; stop')
            position=np.asarray(fusion['position_initial_body_m'])
            rotation=np.asarray(relative['relative_orientation']['rotation_initial_body_from_current_body'])
            if self.last_ns is None: self.up_initial=self.integrator.gravity/9.81
            current={'position':position.copy(),'rotation':rotation.copy(),'measured_ns':now_ns,
                'position_scale_m':fusion['position_error_scale_m'],
                'orientation_scale_rad':fusion['assumptions']['scale_multiplier']*np.sqrt(fusion['orientation_variance_proxy_rad2']),
                'evidence':depth_evidence(depth['depth_m'],depth['valid'],rotation.T@self.up_initial)}
            add=not self.frames or np.linalg.norm(position-self.frames[-1]['position'])>=.04 or np.linalg.norm(rotation-self.frames[-1]['rotation'])>=.08
            if add: self.frames=[*self.frames[-63:],current]
            self.latest_frame=current; self.position=position.copy(); self.rotation=rotation.copy()
            self.last_ns=now_ns; self.identity=tuple(depth['identity']); self.latest_surface=deepcopy(surface); self.fusion=deepcopy(fusion)
            return {'retained_views':len(self.frames),'oldest_ns':self.frames[0]['measured_ns'],'new_view_added':bool(add),
                'corners':observed_corners(surface),'motion_kind':fusion['kind'],'depth_rank':fusion['depth_rank'],
                'position_error_scale_m':fusion['position_error_scale_m'],'static_scene_assumed':True,
                'calibrated_uncertainty':False,'navigation_qualified':False}
        except (ValueError,TypeError,KeyError,IndexError,RuntimeError) as error:
            self.failed=True
            raise SensorContractError('fused ray evidence unavailable; stop geometric control') from error

    def query(self,points_body,ground_support_allowed,*,now_ns,backend='compiled'):
        if self.failed or self.last_ns is None or now_ns!=self.last_ns:
            raise SensorContractError('current valid fused ray memory required')
        points=np.asarray(points_body,dtype=float); roles=np.asarray(ground_support_allowed)
        if points.ndim!=2 or points.shape[1:]!=(3,) or not np.isfinite(points).all() or roles.shape!=(len(points),) or roles.dtype!=bool:
            raise SensorContractError('finite body points and explicit ground roles required')
        reference=points@self.rotation.T+self.position
        free=np.zeros(len(points),bool); support=free.copy(); conflict=free.copy(); examined=np.zeros(len(points),int)
        covered=np.zeros(len(points),int)
        max_radius=np.zeros(len(points))
        frames=self.frames if self.frames[-1]['measured_ns']==now_ns else [*self.frames,self.latest_frame]
        for stored in frames:
            body=points if stored['measured_ns']==now_ns else (reference-stored['position'])@stored['rotation']
            radius=transport_radius(points,self.latest_frame,stored)
            row=query_envelopes(stored['evidence'],body,radius,roles,backend=backend)
            free|=row['free']; support|=row['observed_ground_support']; conflict|=row['contradictory_or_near_surface']
            examined+=row['examined_pixels']; max_radius=np.maximum(max_radius,radius)
            covered+=row['projected_window_pixels']
        free&=~conflict; support&=~conflict
        return {'free':free,'observed_ground_support':support,'contradictory_or_near_surface':conflict,
            'unknown_or_blocked':~(free|support),'all_samples_supported':bool(len(points) and (free|support).all()),
            'retained_views':len(self.frames),'maximum_transport_radius_m':max_radius,'examined_pixels':examined,
            'projected_window_pixels':covered,
            'conditional_on_declared_pose_envelopes':True,'calibrated_uncertainty':False,
            'continuous_volume_qualified':False,'future_gait_qualified':False,'hardware_qualified':False}
