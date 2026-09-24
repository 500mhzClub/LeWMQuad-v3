"""Causal diagnostic stance kinematics; no support or clearance permission."""
import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.foot_load_sensor_development import FEET,IdealFootForceSample
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.simulated_body_observation_development import JOINT_NAMES,CALIBRATION


class FootJacobians:
    def __init__(self,urdf):
        self.geometry=ArticulatedCollisionGeometry(urdf)
        self.parents={j['child']:j for j in self.geometry._joints}
        self.foot_shapes={s['link']:s for s in self.geometry._shapes if s['shape_id'] in {f+':0' for f in FEET}}
        if set(self.foot_shapes)!=set(FEET): raise ValueError('four declared spherical feet required')

    def calculate(self,q):
        links,_=self.geometry.transforms(q); positions=[]; rotations=[]; linear=[]; angular=[]
        for foot in FEET:
            shape=self.foot_shapes[foot]
            if shape['kind']!='sphere' or abs(shape['dimensions'][0]-.022)>1e-12: raise ValueError('22mm spherical foot required')
            p=(links[foot]@shape['origin'])[:3,3]; J=np.zeros((3,12)); A=J.copy(); link=foot
            while link!='base':
                joint=self.parents[link]; pivot=links[joint['parent']]@joint['origin']
                if joint['kind']=='revolute':
                    axis=pivot[:3,:3]@joint['axis']; i=JOINT_NAMES.index(joint['name'])
                    J[:,i]=np.cross(axis,p-pivot[:3,3]); A[:,i]=axis
                link=joint['parent']
            positions.append(p); rotations.append(links[foot][:3,:3]); linear.append(J); angular.append(A)
        return dict(position_body_m=np.array(positions),rotation_body_from_foot=np.array(rotations),
            linear_jacobian=np.array(linear),angular_jacobian=np.array(angular))


class CausalQuietUp:
    """Fixed1.3..1.5s quiet-gravity HYPOTHESIS, transported with500Hz gyro.

    Translational acceleration contaminates gravity; error remains unknown.
    Initialization averages co-rotated specific-force samples, not body axes.
    """
    def __init__(self):
        self.last=None; self.previous=None; self.Q=np.eye(3); self.forces=[]; self.up_anchor=None

    def update(self,stamp,gyro,acceleration=None):
        g=np.asarray(gyro,float)
        if (type(stamp) is not int or g.shape!=(3,) or not np.isfinite(g).all() or
                (self.last is None and stamp!=1_300_000_000) or
                (self.last is not None and stamp-self.last!=2_000_000)):
            raise ValueError('exact causal500Hz initialization/update required')
        if acceleration is not None:
            a=np.asarray(acceleration,float)
            if stamp%20_000_000 or a.shape!=(3,) or not np.isfinite(a).all(): raise ValueError('co-timed50Hz specific force required')
        if self.last is not None: self.Q=self.Q@rotation_increment((self.previous+g)*.001)
        if acceleration is not None and stamp<=1_500_000_000: self.forces.append(self.Q@a)
        if stamp==1_500_000_000:
            if len(self.forces)!=11: raise ValueError('complete11sample quiet initialization required')
            mean=np.mean(self.forces,axis=0); norm=np.linalg.norm(mean)
            if not np.isfinite(norm) or norm<1: raise ValueError('nondegenerate quiet gravity hypothesis required')
            self.up_anchor=mean/norm
        self.last=stamp; self.previous=g.copy()
        if self.up_anchor is None: return None
        up=self.Q.T@self.up_anchor
        return up/np.linalg.norm(up)


BODY_KEYS=frozenset(('identity','calibration_id','measured_ns','available_ns','q','dq','gyro','specific_force','valid'))


def predict_support_motion(kinematics,body,loads,*,identity,up_body):
    if not isinstance(body,dict) or set(body)!=BODY_KEYS: raise ValueError('strict sensor-only body packet required')
    now=body['measured_ns']
    if (type(now) is not int or now<0 or body['available_ns']!=now or body['identity']!=identity or
            body['calibration_id']!=CALIBRATION or type(body['valid']) is not bool or not body['valid']):
        raise ValueError('same-identity co-timed valid declared body sensing required')
    vectors={k:np.asarray(body[k],float) for k in ('q','dq','gyro','specific_force')}
    if any(v.shape!=((12,) if k in ('q','dq') else (3,)) or not np.isfinite(v).all() for k,v in vectors.items()):
        raise ValueError('finite ordered q/dq/IMU required')
    if not isinstance(loads,(list,tuple)) or len(loads)!=11: raise ValueError('complete causal20ms load dwell required')
    for i,sample in enumerate(loads):
        if (not isinstance(sample,IdealFootForceSample) or sample.acquisition_identity!=identity or
                sample.measured_ns!=now-20_000_000+2_000_000*i or sample.available_ns>now):
            raise ValueError('same-stream exact500Hz load history required')
    up=np.asarray(up_body,float)
    if up.shape!=(3,) or not np.isfinite(up).all() or abs(np.linalg.norm(up)-1)>1e-10:
        raise ValueError('explicit conditional unit IMU up hypothesis required')
    data=kinematics.calculate(vectors['q']); r=data['position_body_m']
    rdot=np.einsum('fij,j->fi',data['linear_jacobian'],vectors['dq'])
    angular=vectors['gyro']+np.einsum('fij,j->fi',data['angular_jacobian'],vectors['dq'])
    centre=-(np.cross(vectors['gyro'],r)+rdot)
    rolling=centre+.022*np.cross(angular,up)
    load_valid=np.array([s.valid&~s.saturated for s in loads]).all(axis=0)
    sustained=load_valid&(np.linalg.norm(np.array([s.force_foot_n for s in loads]),axis=2)>5).all(axis=0)
    current_valid=loads[-1].valid&~loads[-1].saturated
    fbody=np.einsum('fij,fj->fi',data['rotation_body_from_foot'],np.where(current_valid[:,None],loads[-1].force_foot_n,0))
    modes={}
    for name,per_foot in [('stationary_centre',centre),('level_sphere_rolling',rolling)]:
        chosen=per_foot[sustained]; mean=chosen.mean(axis=0) if len(chosen)>=2 else None
        modes[name]=dict(per_foot_velocity_body_m_s=per_foot.tolist(),
            consensus_velocity_body_m_s=mean.tolist() if mean is not None else None,
            maximum_disagreement_m_s=float(np.linalg.norm(chosen-mean,axis=1).max()) if mean is not None else None)
    heights=r@up-.022; selected_heights=heights[sustained]
    return dict(measured_ns=now,selected_loaded_feet=sustained.tolist(),load_valid=load_valid.tolist(),
        foot_position_body_m=r.tolist(),force_body_n=[v.tolist() if current_valid[i] else None for i,v in enumerate(fbody)],
        force_along_conditional_up_n=[float(v) if current_valid[i] else None for i,v in enumerate(fbody@up)],
        conditional_up_body=up.tolist(),selected_foot_height_spread_m=float(np.ptp(selected_heights)) if len(selected_heights)>=2 else None,
        modes=modes,minimum_consensus_feet=2,quiet_gravity_and_force_error_hypotheses_unvalidated=True,
        physical_velocity_error_bound_m_s=None,support_established=False,slip_excluded=False,
        continuous_floor_established=False,future_motion_qualified=False,navigation_qualified=False)
