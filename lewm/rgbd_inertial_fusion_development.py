"""Point/plane complementary constraints with one causal gyro owner.

No covariance independence, calibrated uncertainty, relocalization, or action
permission follows from point acceptance. Frozen predecessor observers remain
unchanged. Explicit scales are hypotheses, never fitted sample-error guarantees.
"""
from copy import deepcopy
from dataclasses import dataclass, asdict
import hashlib

import cv2
import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.depth_inertial_moment_fusion_development import MomentWeakSubspaceIntegrator, ASSUMPTIONS
from lewm.depth_relative_motion_development import DepthRelativeState
from lewm.rgbd_correspondence_motion_development import track
from lewm.setup_velocity_prior_development import SetupVelocityPrior, PriorVelocitySensitivity
from lewm.simulated_body_observation_development import validate_policy_packet


@dataclass(frozen=True)
class PointFusionHypotheses:
    point_step_scale_m: float
    projection_agreement_scale_m: float

    def __post_init__(self):
        for value in asdict(self).values():
            if type(value) not in (int,float) or not np.isfinite(value) or value<=0:
                raise SensorContractError('explicit positive finite uncalibrated point hypotheses required')


def complementary_constraints(motion, point, hypotheses):
    """Return effective constraints without modifying either source observation."""
    rank=motion['rank']; status=motion['status']
    if (type(rank) is not int or rank not in (1,2,3)
            or status != ('OBSERVED_TRANSLATION' if rank==3 else 'PARTIALLY_OBSERVED_TRANSLATION')):
        raise SensorContractError('accepted explicit original plane constraints required')
    projected=np.asarray(motion['observable_projection_previous_body_m'],float)
    weak=np.asarray(motion['weak_directions_previous_body'],float).reshape(-1,3)
    if (projected.shape!=(3,) or not np.isfinite(projected).all() or weak.shape!=(3-rank,3)
            or not np.isfinite(weak).all() or not np.allclose(weak@weak.T,np.eye(3-rank),atol=1e-7,rtol=0)
            or not np.allclose(weak@projected,0,atol=1e-7,rtol=0)):
        raise SensorContractError('finite independent plane subspaces required')
    if rank==3:
        if not np.array_equal(motion['translation_previous_body_m'],projected):
            raise SensorContractError('full plane translation differs from projection')
    elif motion['translation_previous_body_m'] is not None:
        raise SensorContractError('partial plane observation must retain missing translation')
    accepted=point['status']=='CONDITIONAL_RGBD_POINT_TRANSLATION'
    if point['status'] not in ('CONDITIONAL_RGBD_POINT_TRANSLATION','INSUFFICIENT_POINT_SUPPORT','POINT_CONSISTENCY_REJECTED'):
        raise SensorContractError('explicit point acceptance or rejection required')
    if type(point['conditional_point_correspondence_rank']) is not int or point['conditional_point_correspondence_rank']!=(3 if accepted else 0):
        raise SensorContractError('point rank/status disagreement')
    full=point['translation_previous_body_m']; disagreement=None; used=False
    if accepted:
        full=np.asarray(full,float)
        if full.shape!=(3,) or not np.isfinite(full).all() or np.linalg.norm(full)>.15:
            raise SensorContractError('finite local accepted point displacement required')
        P=weak.T@weak
        disagreement=float(np.linalg.norm((np.eye(3)-P)@full-projected))
        if disagreement>hypotheses.projection_agreement_scale_m:
            raise SensorContractError('point/plane observed-subspace conflict')
        if rank<3:
            projected=projected+P@full; used=True
    elif full is not None:
        raise SensorContractError('rejected point observation cannot supply displacement')
    return dict(projected_previous_body_m=projected.tolist(),
        effective_weak_directions_previous_body=[] if used else weak.tolist(),
        original_depth_rank=rank,point_used_for_weak_directions=used,
        point_accepted=accepted,observed_subspace_disagreement_m=disagreement,
        conditional_combined_rank=3 if used else rank,independent_measurements_assumed=False)


class ComplementaryRGBDIntegrator(MomentWeakSubspaceIntegrator):
    """One accumulated pose; explicit prior and correlated point-error transport."""
    def __init__(self, prior, hypotheses):
        if not isinstance(prior,SetupVelocityPrior) or not isinstance(hypotheses,PointFusionHypotheses):
            raise SensorContractError('explicit setup prior and point hypotheses required')
        super().__init__(); self.prior=prior; self.hypotheses=hypotheses
        self.prior_sensitivity=PriorVelocitySensitivity()
        self.point_position_scale=0.; self.point_velocity_scale=0.; self.previous_rgb=None

    def observe(self, policy, relative, point):
        if self.failed:raise SensorContractError('complementary fusion fault latched')
        try:return self._fuse(policy,relative,point)
        except (ValueError,TypeError,KeyError,IndexError,RuntimeError) as error:
            self.failed=True
            raise SensorContractError('invalid complementary RGBD/inertial evidence; stop') from error

    def _fuse(self, policy, state, point):
        validate_policy_packet(policy)
        now=policy['sensor_state']['decision_ns']; surface=state['local_surfaces']
        identity=tuple(surface['identity']); rgb=hashlib.sha256(policy['image']['rgb'].tobytes()).hexdigest()
        rotation=np.asarray(state['relative_orientation']['rotation_initial_body_from_current_body'],float)
        if (identity!=tuple(policy['sensor_state']['identity']) or surface['rgb_sha256']!=rgb
                or state['measured_ns']!=now or state['relative_orientation']['decision_ns']!=now
                or surface['measured_ns']!=now or rotation.shape!=(3,3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-7,rtol=0)
                or abs(np.linalg.det(rotation)-1)>1e-7):
            raise SensorContractError('current proper observed attitude and sensor identity required')
        keys={'measured_ns','identity','rgb_sha256','depth_sha256','previous_rgb_sha256',
              'relative_orientation','motion','static_point_correspondences_assumed','native_pose_input',
              'plane_depth_rank_modified','navigation_qualified'}
        if (set(point)!=keys or point['measured_ns']!=now or tuple(point['identity'])!=identity
                or point['rgb_sha256']!=rgb or point['depth_sha256']!=surface['depth_sha256']
                or point['previous_rgb_sha256']!=self.previous_rgb
                or point['relative_orientation']!=state['relative_orientation']
                or point['static_point_correspondences_assumed'] is not True
                or any(point[k] is not False for k in ('native_pose_input','plane_depth_rank_modified','navigation_qualified'))):
            raise SensorContractError('same-interval point/depth/gyro provenance required')
        if self.last_ns is not None and (now-self.last_ns!=100_000_000 or identity!=self.identity):
            raise SensorContractError('fusion episode/clock discontinuity')
        motion=state['motion']; delta=None; acceleration=None; constraints=None
        kind='INITIAL_RELATIVE_ANCHOR'
        if self.last_ns is None:
            if (motion is not None or not np.array_equal(rotation,np.eye(3)) or now!=self.prior.anchor_ns
                    or identity!=self.prior.identity or point['motion']!={'status':'INITIAL_RGBD_ANCHOR',
                        'translation_previous_body_m':None,'conditional_point_correspondence_rank':0}):
                raise SensorContractError('single explicit setup and observed sensor anchor required')
            self.velocity=np.asarray(self.prior.mean_initial_body_m_s,float).copy()
            force=policy['sensor_state']['sensed']['specific_force']
            if not force['valid'].all():raise SensorContractError('initial gravity hypothesis unavailable')
            mean=force['values'].mean(0); norm=float(np.linalg.norm(mean))
            if abs(norm-9.81)>.75:raise SensorContractError('initial gravity hypothesis not quiet')
            self.gravity=9.81*mean/norm; self._acceleration(policy,rotation)
        else:
            constraints=complementary_constraints(motion,point['motion'],self.hypotheses)
            acceleration=self._acceleration(policy,rotation)
            weak=np.asarray(constraints['effective_weak_directions_previous_body'],float).reshape(-1,3)
            projected=np.asarray(constraints['projected_previous_body_m'])
            observed=self.rotation@projected
            w=weak@self.rotation.T; P=w.T@w
            if not len(weak):
                initial_delta=observed
                self.velocity=observed/.1+np.asarray(self.moments['endpoint_velocity_correction_m_s'])
                self.weak_seconds=0.; self.weak_proxy=0.
                kind='POINT_COMPLEMENTED_PLANE_TRANSLATION' if constraints['point_used_for_weak_directions'] else 'DEPTH_CONSTRAINED_TRANSLATION'
                if constraints['point_used_for_weak_directions']:
                    self.point_position_scale+=self.hypotheses.point_step_scale_m
                    self.point_velocity_scale=self.hypotheses.point_step_scale_m/.1
                else:self.point_velocity_scale=0.
            else:
                inertial=self.velocity*.1+np.asarray(self.moments['displacement_from_initial_velocity_m'])
                initial_delta=observed+P@inertial
                self.velocity=(np.eye(3)-P)@(observed/.1+np.asarray(self.moments['endpoint_velocity_correction_m_s']))+P@(self.velocity+np.asarray(self.moments['velocity_increment_m_s']))
                self.weak_seconds+=.1; self.weak_intervals+=1
                scale=ASSUMPTIONS['velocity_initialization_scale_m_s']*self.weak_seconds+.5*ASSUMPTIONS['unmodelled_acceleration_scale_m_s2']*self.weak_seconds**2
                self.position_proxy+=max(0.,scale**2-self.weak_proxy); self.weak_proxy=scale**2
                # Worst-case transport of the preceding point-derived velocity
                # ball. Correlation with its position error cannot cancel it.
                self.point_velocity_scale*=min(1.,float(np.linalg.norm(P,ord=2)))
                self.point_position_scale+=.1*self.point_velocity_scale
                kind='INERTIALLY_PREDICTED_WEAK_COMPONENT'
            delta=self.rotation.T@initial_delta
            if not np.isfinite(delta).all() or np.linalg.norm(delta)>.15 or not np.isfinite(self.velocity).all():
                raise SensorContractError('fusion increment outside fixed local envelope')
            self.position+=initial_delta
            self.position_proxy+=ASSUMPTIONS['depth_step_scale_m']**2
            self.orientation_proxy+=ASSUMPTIONS['orientation_step_scale_rad']**2
            self.prior_sensitivity.advance(P,.1)
        sensitivity=self.prior_sensitivity.snapshot(self.prior.radius_m_s)
        inherited=float(ASSUMPTIONS['scale_multiplier']*np.sqrt(self.position_proxy))
        combined=inherited+sensitivity['position_radius_m']+self.point_position_scale
        if not np.isfinite(combined):raise SensorContractError('finite accumulated error accounting required')
        self.rotation=rotation.copy(); self.last_ns=now; self.identity=identity; self.previous_rgb=rgb
        return dict(schema='complementary_rgbd_inertial_fusion_development.v1',measured_ns=now,kind=kind,
            translation_previous_body_m=None if delta is None else delta.tolist(),
            position_initial_body_m=self.position.tolist(),velocity_initial_body_m_s=self.velocity.tolist(),
            acceleration_initial_body_m_s2=None if acceleration is None else acceleration.tolist(),
            inertial_interval=None if acceleration is None else deepcopy(self.moments),
            depth_rank=None if motion is None else motion['rank'],constraints=constraints,
            consecutive_weak_seconds=self.weak_seconds,weak_intervals=self.weak_intervals,
            position_variance_proxy_m2=self.position_proxy,orientation_variance_proxy_rad2=self.orientation_proxy,
            inherited_position_error_scale_m=inherited,point_position_scale_m=self.point_position_scale,
            point_velocity_scale_m_s=self.point_velocity_scale,initial_velocity_prior=asdict(self.prior),
            initial_velocity_source='SUPPLIED_SETUP_PRIOR_NOT_SENSOR',initial_velocity_prior_transport=sensitivity,
            point_hypotheses=asdict(self.hypotheses),position_error_scale_m=combined,
            usable_under_declared_proxy_budget=bool(combined<=ASSUMPTIONS['maximum_position_scale_m']),
            assumptions=deepcopy(ASSUMPTIONS),independent_measurements_assumed=False,
            scale_composition='inherited_uncalibrated_proxy_plus_prior_radius_plus_correlated_point_scale',
            uncertainty_model_validated=False,ground_plane_qualified=False,navigation_qualified=False,
            point_scale_covers_calibration_and_mismatches=False,setup_condition_assumed=True,
            scope='conditional development fusion; no relocalization or calibrated probability')


class RGBDInertialState:
    """One gyro integration shared by raw plane and point observations."""
    def __init__(self, *, prior, hypotheses):
        self.depth=DepthRelativeState(); self.integrator=ComplementaryRGBDIntegrator(prior,hypotheses)
        self.previous=self.previous_rotation=None; self.failed=False

    def observe(self, policy, depth, fast, *, now_ns):
        if self.failed:raise SensorContractError('RGBD inertial state fault latched')
        try:
            relative=self.depth.observe(policy,depth,fast,now_ns=now_ns)
            attitude=relative['relative_orientation']; R=np.asarray(attitude['rotation_initial_body_from_current_body'])
            if self.previous is None:
                motion=dict(status='INITIAL_RGBD_ANCHOR',translation_previous_body_m=None,conditional_point_correspondence_rank=0)
            else:
                rgb,old_depth=self.previous
                motion=track(rgb,policy['image']['rgb'],old_depth,depth,self.previous_rotation.T@R)
            point=dict(measured_ns=now_ns,identity=list(depth['identity']),rgb_sha256=depth['rgb_sha256'],
                depth_sha256=hashlib.sha256(depth['depth_m'].tobytes()+depth['valid'].tobytes()).hexdigest(),
                previous_rgb_sha256=self.previous[1]['rgb_sha256'] if self.previous else None,
                relative_orientation=deepcopy(attitude),motion=motion,static_point_correspondences_assumed=True,
                native_pose_input=False,plane_depth_rank_modified=False,navigation_qualified=False)
            fusion=self.integrator.observe(policy,relative,point)
            self.previous=(policy['image']['rgb'].copy(),deepcopy(depth)); self.previous_rotation=R.copy()
            return dict(measured_ns=now_ns,depth_state=relative,point_state=point,fusion=fusion,
                navigation_qualified=False,hardware_qualified=False)
        except (ValueError,TypeError,KeyError,IndexError,RuntimeError,cv2.error) as error:
            self.failed=True
            raise SensorContractError('RGBD inertial state unavailable; no stale continuation') from error
