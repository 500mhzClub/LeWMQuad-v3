"""Observed blockers and floor-view-compatible approach limits.

This is a conservative development constraint, not a free-space certificate.
Unverified corner hypotheses never override a nearer measured front surface.
"""
from copy import deepcopy

import numpy as np

from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.causal_sensor_state import SensorContractError
from lewm.depth_relative_motion_development import surface_cloud
from lewm.measured_region_navigation_development import RegionObservation
from lewm.measured_line_integral_navigation_development import MeasuredLineTraversal


def floor_view_standoff(points, normals, up):
    points, normals, up = [np.asarray(x,dtype=float) for x in (points,normals,up)]
    if (points.ndim!=2 or points.shape[1:]!=(3,) or normals.shape!=points.shape
            or up.shape!=(3,) or not all(np.isfinite(x).all() for x in (points,normals,up))
            or abs(np.linalg.norm(up)-1.)>1e-7):
        raise SensorContractError('finite observed surface cloud and unit gravity required')
    selected=(np.abs(normals@up)>=.97)&(points@up<-.15)
    floor=points[selected]
    if len(floor)<100: return None
    centre=np.median(floor,axis=0)
    _,singular,v=np.linalg.svd(floor-centre,full_matrices=False)
    if singular[1]<.05: return None
    normal=v[-1]
    if normal@up<0: normal=-normal
    offset=float(np.median(floor@normal))
    residual=float(np.max(np.abs(floor@normal-offset)))
    if normal@up<.97 or residual>.01 or offset>=-.15: return None
    transform=np.asarray(BODY_FROM_OPTICAL)
    # Leave the lower80 pixel rows available for floor support, rather than
    # planning to the last visible floor ray. This is an explicit view margin.
    direction=transform[:3,:3]@np.array([0.,(400.5-240.)/FOCAL,1.])
    origin=transform[:3,3]; denominator=float(normal@direction)
    if denominator>=-1e-6: return None
    depth=float((offset-normal@origin)/denominator)
    point=origin+depth*direction
    if not .2<=depth<=5. or point[0]<=0.: return None
    return {'minimum_front_standoff_m':float(point[0]+.15),
            'floor_probe_body_m':point.tolist(),'probe_row':400,'probe_column_coordinate':319.5,
            'floor_normal_body':normal.tolist(),'floor_offset_body_m':offset,
            'floor_points':len(floor),'maximum_plane_residual_m':residual,
            'view_margin_m':.15,'hardware_calibrated':False}


def blocking_limit(surface, radius, floor_view):
    if floor_view is None: raise SensorContractError('current observed floor-view support required')
    if not np.isfinite(radius) or radius<=0.: raise SensorContractError('positive nominal radius required')
    standoff=max(radius+.15,floor_view['minimum_front_standoff_m'])
    rows=[]
    for s in surface['surface_segments']:
        n=np.asarray(s['normal_body_xy'],dtype=float); p=np.asarray(s['endpoints_body_xy_m'],dtype=float)
        d=float(s['offset_body_m'])
        if n.shape!=(2,) or p.shape!=(2,2) or not np.isfinite(n).all() or not np.isfinite(p).all() or not np.isfinite(d):
            raise SensorContractError('finite observed segment required')
        # Only observed support intersecting the nominal transverse footprint.
        # This may stop conservatively on a partial blocker; it never bridges
        # missing support into an inferred opening or wall endpoint.
        if n[0]>.8 and d>0. and p[:,1].min()<=radius and p[:,1].max()>=-radius:
            distance=d/n[0]
            rows.append({'surface_distance_forward_m':float(distance),
                         'maximum_approach_m':float(distance-standoff),
                         'normal_body_xy':n.tolist(),'offset_body_m':d})
    return {'maximum_approach_m':min(r['maximum_approach_m'] for r in rows) if rows else None,
            'blocking_surfaces':rows,'minimum_standoff_m':float(standoff),
            'floor_view':deepcopy(floor_view),'free_volume_qualified':False}


class ObservableApproachRegions(RegionObservation):
    def __init__(self,geometry):
        super().__init__(geometry); self.floor_view=None

    def observe(self,packet,depth,relative,*,now_ns):
        row=super().observe(packet,depth,relative,now_ns=now_ns)
        points,normals=surface_cloud(depth,packet,now_ns=now_ns)
        self.floor_view=floor_view_standoff(points,normals,self.memory.rotation.T@self.memory.up_initial)
        return {**row,'floor_view':deepcopy(self.floor_view)}

    def limit(self,packet,*,now_ns):
        if now_ns!=self.memory.last_ns: raise SensorContractError('current approach observation required')
        return blocking_limit(self.memory.latest_surface,self.volume(packet)['maximum_radius_m'],self.floor_view)

    def target(self,packet,*,now_ns):
        candidate=super().target(packet,now_ns=now_ns)
        if candidate is None: return None
        limit=self.limit(packet,now_ns=now_ns)
        value=limit['maximum_approach_m']
        if value is not None:
            if value<=.15: return None
            if candidate['target_body_m'][0]>value:
                candidate['unconstrained_target_body_m']=candidate['target_body_m']
                candidate['target_body_m']=[value,0.,0.]
                candidate['target_initial_body_m']=(self.memory.position+self.memory.rotation@candidate['target_body_m']).tolist()
                candidate['kind']='OBSERVED_BLOCKER_LIMITED_REGION'
        candidate['approach_limit']=limit
        return candidate


class ObservableApproachTraversal(MeasuredLineTraversal):
    def _observe(self,packet,fast_packet,*,now_ns):
        update=None
        if self.target is not None and self.status in ('TRAVERSING','BRAKING'):
            obs=self.observations; m=obs.memory
            limit=obs.limit(packet,now_ns=now_ns)
            value=limit['maximum_approach_m']
            remaining=float((m.rotation.T@(np.asarray(self.target['target_initial_body_m'])-m.position))[0])
            if value is not None and value<remaining-.01:
                # Tighten only, never expand an old target as support disappears.
                # A newly observed blocker behind the stopping region is a
                # failure, not a fabricated successful arrival.
                if value<0.: raise SensorContractError('observed blocker inside required stopping/view region')
                self.target['target_initial_body_m']=(m.position+m.rotation@np.array([value,0.,0.])).tolist()
                self.target['kind']='OBSERVED_BLOCKER_LIMITED_REGION'
                self.target['approach_limit']=limit
                update={'decision_ns':now_ns,'previous_remaining_m':remaining,'new_remaining_m':value,'limit':limit}
        row=super()._observe(packet,fast_packet,now_ns=now_ns)
        row['approach_constraint_update']=update
        return row
