"""Time-resolved inertial moments alongside unchanged depth constraints.

The error scales below are declared development assumptions, not calibrated
covariances, guaranteed bounds, probabilities or hardware safety certificates.
Predicted components never change the depth registration's rank or status.
"""
from copy import deepcopy
import hashlib

import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.depth_relative_motion_development import DepthRelativeState
from lewm.gravity_feedback_ground_development import force_in_current_frame
from lewm.simulated_body_observation_development import validate_policy_packet


SCHEMA='depth_inertial_moment_weak_subspace_development.v1'
ASSUMPTIONS={'depth_step_scale_m':.0005,'velocity_initialization_scale_m_s':.005,
    'unmodelled_acceleration_scale_m_s2':.02,'orientation_step_scale_rad':.00001,
    'scale_multiplier':3.,'maximum_position_scale_m':.08,
    'calibrated_covariance':False,'hardware_qualified':False}



def interval_moments(acceleration):
    """Integrate five causal 20-ms bin-average accelerations.

    The within-bin constant-acceleration approximation is explicit. This
    retains timing across bins, but cannot reconstruct intra-bin jerk.
    """
    a=np.asarray(acceleration,dtype=float)
    if a.shape!=(5,3) or not np.isfinite(a).all():
        raise SensorContractError('five finite causal acceleration vectors required')
    return {'velocity_increment_m_s':(.02*a.sum(axis=0)).tolist(),
        'displacement_from_initial_velocity_m':(np.array([.0018,.0014,.0010,.0006,.0002])@a).tolist(),
        'endpoint_velocity_correction_m_s':(np.array([.002,.006,.010,.014,.018])@a).tolist(),
        'acceleration_initial_body_m_s2':a.tolist(),
        'approximation':'piecewise_constant_within_five_causal_20ms_bins',
        'bias_calibrated':False,'intra_bin_jerk_bounded':False}


class MomentWeakSubspaceIntegrator:
    def __init__(self):
        self.last_ns=self.identity=self.rotation=self.gravity=None
        self.position=np.zeros(3); self.velocity=None
        self.position_proxy=0.; self.orientation_proxy=0.
        self.weak_seconds=0.; self.weak_proxy=0.; self.weak_intervals=0
        self.force_history=None; self.gyro_history=None; self.failed=False

    def _acceleration(self,policy,rotation):
        sensed=policy['sensor_state']['sensed']; f=sensed['specific_force']; g=sensed['gyro']
        if (not f['valid'].all() or not g['valid'].all()
                or not np.array_equal(f['measured_ns'],g['measured_ns'])
                or not np.all(np.diff(f['measured_ns'])==20_000_000)):
            raise SensorContractError('complete co-timed measured force and gyro required')
        if self.force_history is not None:
            old=self.force_history; lookup={int(t):i for i,t in enumerate(old['measured_ns'])}
            for i,t in enumerate(f['measured_ns']):
                if int(t) in lookup:
                    for key in ('values','valid','available_ns'):
                        if not np.array_equal(f[key][i],old[key][lookup[int(t)]]):
                            raise SensorContractError('specific-force history rewritten')
        if self.gyro_history is not None:
            old=self.gyro_history; lookup={int(t):i for i,t in enumerate(old['measured_ns'])}
            for i,t in enumerate(g['measured_ns']):
                if int(t) in lookup:
                    for key in ('values','valid','available_ns'):
                        if not np.array_equal(g[key][i],old[key][lookup[int(t)]]):
                            raise SensorContractError('gyro history rewritten')
        self.force_history=deepcopy(f)
        self.gyro_history=deepcopy(g)
        if f['measured_ns'][-1]!=policy['sensor_state']['decision_ns']:
            raise SensorContractError('current endpoint force sample required')
        force=force_in_current_frame(f['values'],g['values'])[-5:]
        acceleration=force@rotation.T-self.gravity
        self.moments=interval_moments(acceleration)
        return acceleration.mean(axis=0)

    def observe(self,policy,depth_state):
        if self.failed: raise SensorContractError('inertial fusion fault latched')
        try: return self._observe(policy,depth_state)
        except (ValueError,TypeError,KeyError,IndexError,RuntimeError) as error:
            self.failed=True
            raise SensorContractError('invalid inertial/depth history; apply zero') from error

    def _observe(self,policy,state):
        validate_policy_packet(policy)
        now=policy['sensor_state']['decision_ns']; surface=state['local_surfaces']
        identity=tuple(surface['identity'])
        rotation=np.asarray(state['relative_orientation']['rotation_initial_body_from_current_body'],dtype=float)
        if (identity!=tuple(policy['sensor_state']['identity'])
                or surface['rgb_sha256']!=hashlib.sha256(policy['image']['rgb'].tobytes()).hexdigest()
                or state['measured_ns']!=now or state['relative_orientation']['decision_ns']!=now
                or surface['measured_ns']!=now
                or rotation.shape!=(3,3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-7,rtol=0)
                or abs(np.linalg.det(rotation)-1)>1e-7):
            raise SensorContractError('current proper observed attitude required')
        if self.last_ns is not None and (now-self.last_ns!=100_000_000 or identity!=self.identity):
            raise SensorContractError('fusion episode/clock discontinuity')
        motion=state['motion']; delta=None; acceleration=None
        kind='INITIAL_RELATIVE_ANCHOR'
        if self.last_ns is None:
            if motion is not None or not np.array_equal(rotation,np.eye(3)):
                raise SensorContractError('initial depth/gyro anchor required')
            force=policy['sensor_state']['sensed']['specific_force']
            if not force['valid'].all(): raise SensorContractError('initial gravity hypothesis unavailable')
            mean=force['values'].mean(axis=0); norm=float(np.linalg.norm(mean))
            if abs(norm-9.81)>.75: raise SensorContractError('initial gravity hypothesis not quiet')
            self.gravity=9.81*mean/norm
            self._acceleration(policy,rotation)
        else:
            acceleration=self._acceleration(policy,rotation)
            rank=motion['rank']; status=motion['status']
            if type(rank) is not int or rank not in (1,2,3) or status not in ('OBSERVED_TRANSLATION','PARTIALLY_OBSERVED_TRANSLATION'):
                raise SensorContractError('accepted explicit depth constraints required')
            if (rank==3)!=(status=='OBSERVED_TRANSLATION'):
                raise SensorContractError('depth rank/status mismatch')
            projected=np.asarray(motion['observable_projection_previous_body_m'],dtype=float)
            weak=np.asarray(motion['weak_directions_previous_body'],dtype=float).reshape(-1,3)
            if (projected.shape!=(3,) or not np.isfinite(projected).all()
                    or weak.shape!=(3-rank,3) or not np.isfinite(weak).all()
                    or not np.allclose(weak@weak.T,np.eye(3-rank),atol=1e-7,rtol=0)
                    or not np.allclose(weak@projected,0.,atol=1e-7,rtol=0)):
                raise SensorContractError('finite independent depth subspaces required')
            observed=self.rotation@projected
            if rank==3:
                full=np.asarray(motion['translation_previous_body_m'],dtype=float)
                if full.shape!=(3,) or not np.array_equal(full,projected):
                    raise SensorContractError('full depth displacement must match its projection')
                initial_delta=observed
                self.velocity=observed/.1+np.asarray(self.moments['endpoint_velocity_correction_m_s'])
                self.weak_seconds=0.; self.weak_proxy=0.
                kind='DEPTH_CONSTRAINED_TRANSLATION'
            else:
                if motion['translation_previous_body_m'] is not None or self.velocity is None:
                    raise SensorContractError('weak prediction requires a previously observed velocity')
                w=weak@self.rotation.T; weak_projector=w.T@w
                inertial=self.velocity*.1+np.asarray(self.moments['displacement_from_initial_velocity_m'])
                initial_delta=observed+weak_projector@inertial
                self.velocity=(np.eye(3)-weak_projector)@(observed/.1+np.asarray(self.moments['endpoint_velocity_correction_m_s']))+weak_projector@(self.velocity+np.asarray(self.moments['velocity_increment_m_s']))
                self.weak_seconds+=.1; self.weak_intervals+=1
                scale=(ASSUMPTIONS['velocity_initialization_scale_m_s']*self.weak_seconds
                    +.5*ASSUMPTIONS['unmodelled_acceleration_scale_m_s2']*self.weak_seconds**2)
                additional=scale**2-self.weak_proxy
                self.position_proxy+=max(0.,additional); self.weak_proxy=scale**2
                kind='INERTIALLY_PREDICTED_WEAK_COMPONENT'
            delta=self.rotation.T@initial_delta
            if not np.isfinite(delta).all() or np.linalg.norm(delta)>.15 or not np.isfinite(self.velocity).all():
                raise SensorContractError('fusion increment outside fixed local envelope')
            self.position+=initial_delta
            self.position_proxy+=ASSUMPTIONS['depth_step_scale_m']**2
            self.orientation_proxy+=ASSUMPTIONS['orientation_step_scale_rad']**2
        # Recovery of full depth constraints does not erase accumulated pose
        # error or relocalize the initial reference; these proxies never shrink.
        scale=ASSUMPTIONS['scale_multiplier']*np.sqrt(self.position_proxy)
        usable=bool(scale<=ASSUMPTIONS['maximum_position_scale_m'])
        self.rotation=rotation.copy(); self.last_ns=now; self.identity=identity
        return {'schema':SCHEMA,'measured_ns':now,'kind':kind,
            'translation_previous_body_m':delta.tolist() if delta is not None else None,
            'position_initial_body_m':self.position.tolist(),
            'velocity_initial_body_m_s':self.velocity.tolist() if self.velocity is not None else None,
            'acceleration_initial_body_m_s2':acceleration.tolist() if acceleration is not None else None,
            'inertial_interval':deepcopy(self.moments) if acceleration is not None else None,
            'uncertainty_model_validated':False,
            'depth_rank':motion['rank'] if motion is not None else None,
            'consecutive_weak_seconds':self.weak_seconds,'weak_intervals':self.weak_intervals,
            'position_variance_proxy_m2':self.position_proxy,
            'orientation_variance_proxy_rad2':self.orientation_proxy,
            'position_error_scale_m':float(scale),'usable_under_declared_proxy_budget':usable,
            'assumptions':deepcopy(ASSUMPTIONS),'ground_plane_qualified':False,
            'scope':'causal development fusion; inertial prediction is not a full depth observation'}


class MomentDepthInertialState:
    """Live sensor observer with the original depth evidence kept verbatim."""
    def __init__(self):
        self.depth=DepthRelativeState(); self.integrator=MomentWeakSubspaceIntegrator()

    def observe(self,policy,depth,fast,*,now_ns):
        observed=self.depth.observe(policy,depth,fast,now_ns=now_ns)
        fusion=self.integrator.observe(policy,observed)
        return {'schema':SCHEMA,'measured_ns':now_ns,'depth_state':observed,'fusion':fusion,
                'navigation_qualified':False,'hardware_qualified':False}
