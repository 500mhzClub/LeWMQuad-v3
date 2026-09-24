"""Causal visual-led motion evidence, deliberately separate from permission.

The frozen RGB-D/gyro observer owns pose. Optional contact odometry reports
agreement or missingness; it never changes pose or grants terrain clearance.
"""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from lewm.causal_sensor_state import SensorContractError, _identity, _ns
from lewm.joint_rgbd_rigid_pose_development import RigidRGBDKeyframePose, proper

CONTACT_CALIBRATION = 'ideal-vector-load-URDF-rolling-conditional-up-diagnostic-v1'
POSE_FIELDS = ('frame', 'measured_ns', 'mode', 'reference_frame',
    'position_initial_body_m', 'rotation_initial_body_from_current_body',
    'gyro_rotation_initial_body_from_current_body', 'promoted_keyframe',
    'promotion_reason', 'keyframe_count', 'rgb_sha256', 'depth_sha256',
    'position_error_bound', 'orientation_error_bound', 'uncertainty_model_validated',
    'gyro_role', 'native_pose_input', 'global_history_reset', 'navigation_qualified')


@dataclass(frozen=True)
class ContactMotionSample:
    """Explicit ideal-sensor hypothesis, not vendor foot counts or safe support.

    The caller binds episode and acquisition identity at its acquisition boundary.
    Gyro rotations share one fixed contact-history anchor, not the visual anchor.
    """
    identity: tuple
    acquisition_identity: str
    measured_ns: int
    available_ns: int
    velocity_body_m_s: tuple | None
    rotation_contact_anchor_from_body: tuple
    calibration_id: str = CONTACT_CALIBRATION

    def __post_init__(self):
        object.__setattr__(self, 'identity', _identity(self.identity))
        for name in ('measured_ns', 'available_ns'):
            object.__setattr__(self, name, _ns(getattr(self, name), name))
        if (not isinstance(self.acquisition_identity, str) or not self.acquisition_identity
                or self.available_ns < self.measured_ns or self.calibration_id != CONTACT_CALIBRATION):
            raise SensorContractError('declared contact acquisition/calibration and causal clocks required')
        R = proper(self.rotation_contact_anchor_from_body)
        object.__setattr__(self, 'rotation_contact_anchor_from_body', tuple(tuple(float(v) for v in r) for r in R))
        if self.velocity_body_m_s is not None:
            v = np.asarray(self.velocity_body_m_s, float)
            if v.shape != (3,) or not np.isfinite(v).all():
                raise SensorContractError('finite contact velocity or explicit None required')
            object.__setattr__(self, 'velocity_body_m_s', tuple(float(x) for x in v))


def contact_comparison(samples, *, identity, acquisition_identity, before, after, now_ns):
    """Six-sample trapezoid in a fixed contact anchor, compared in current body.

    Rotating each measured body velocity before integration avoids integrating
    changing body axes as though they were a fixed frame. No time extrapolation.
    """
    if samples is None:
        return dict(status='CONTACT_NOT_SUPPLIED', displacement_disagreement_m=None)
    try:
        if before is None:
            return dict(status='NO_PREVIOUS_VISUAL_INTERVAL', displacement_disagreement_m=None)
        start, end = before['measured_ns'], after['measured_ns']
        if (acquisition_identity is None or end-start != 100_000_000
                or not isinstance(samples, (tuple, list)) or len(samples) != 6):
            raise SensorContractError('declared six-sample 100ms contact window required')
        for i, sample in enumerate(samples):
            if (not isinstance(sample, ContactMotionSample) or sample.identity != identity
                    or sample.acquisition_identity != acquisition_identity
                    or sample.measured_ns != start+i*20_000_000 or sample.available_ns > now_ns):
                raise SensorContractError('same-episode acquired contact window with exact causal clocks required')
        missing = [s.measured_ns for s in samples if s.velocity_body_m_s is None]
        common = dict(start_ns=start, end_ns=end, unavailable_measured_ns=missing,
                      physical_disagreement_bound_m=None, slip_excluded=False,
                      calibration_id=CONTACT_CALIBRATION, contact_used_to_update_pose=False)
        if missing:
            return common | dict(status='CONTACT_VELOCITY_UNAVAILABLE', displacement_disagreement_m=None)
        rotations = [np.asarray(s.rotation_contact_anchor_from_body) for s in samples]
        velocities = np.array([R@s.velocity_body_m_s for R, s in zip(rotations, samples, strict=True)])
        displacement = rotations[-1].T@((velocities[:-1]+velocities[1:]).sum(axis=0)*.01)
        visual = np.asarray(after['rotation_initial_body_from_current_body']).T@(
            np.asarray(after['position_initial_body_m'])-before['position_initial_body_m'])
        return common | dict(status='CONTACT_VISUAL_COMPARISON_AVAILABLE',
            contact_displacement_current_body_m=displacement.tolist(),
            visual_displacement_current_body_m=visual.tolist(),
            displacement_disagreement_m=float(np.linalg.norm(displacement-visual)))
    except (ValueError, TypeError, KeyError) as error:
        return dict(status='CONTACT_CONTRACT_REJECTED', reason=str(error), displacement_disagreement_m=None,
                    contact_used_to_update_pose=False)


class VisualLedMotion:
    """One explicit episode, no reset or contact-dependent visual update.

    Snapshot returns a current pose only at its measured timestamp. At later
    times the old pose is historical, not an asserted stationary robot. This
    implementation does not extrapolate translation or asynchronously integrate
    gyro between RGB-D updates; both current components are then unobserved.
    """
    def __init__(self, mode, *, identity, contact_acquisition_identity=None):
        self.identity = _identity(identity)
        if contact_acquisition_identity is not None and (
                not isinstance(contact_acquisition_identity, str) or not contact_acquisition_identity):
            raise SensorContractError('explicit optional contact acquisition identity required')
        self.contact_acquisition_identity = contact_acquisition_identity
        self.model = RigidRGBDKeyframePose(mode)
        self.last_visual = self.calibrations = self.failure = self.last_decision_ns = None
        self.contact = dict(status='CONTACT_NOT_SUPPLIED', displacement_disagreement_m=None)

    def observe(self, policy, depth, fast, *, now_ns, contact_samples=None):
        now = _ns(now_ns, 'motion decision')
        if self.last_decision_ns is not None and now < self.last_decision_ns:
            raise SensorContractError('motion decision cannot move backwards')
        self.last_decision_ns = now
        if self.failure is not None:
            return self.snapshot(now_ns=now)
        try:
            if _identity(policy['sensor_state']['identity']) != self.identity:
                raise SensorContractError('motion episode mismatch; use a distinct episode instance')
            if depth['measured_ns'] != now or policy['image']['measured_ns'] != now:
                raise SensorContractError('frozen observer requires current co-timed RGB-D')
            state = self.model.observe(policy, depth, fast, now_ns=now)
            metadata = dict(rgb=policy['image']['calibration_id'], depth=depth['calibration_id'],
                fast_gyro=fast['calibration_id'], body=policy['sensor_state']['sensed']['gyro']['calibration_id'])
            if self.calibrations is not None and metadata != self.calibrations:
                raise SensorContractError('motion calibration changed within episode')
            previous = self.last_visual
            self.last_visual = {k: deepcopy(state[k]) for k in POSE_FIELDS}
            self.last_visual['available_ns'] = now
            self.last_visual['processing_latency_accounted'] = False
            self.last_visual['acquisition_available_ns'] = max(policy['image']['available_ns'], depth['available_ns'],
                int(np.max(fast['available_ns'])))
            self.calibrations = metadata
            self.contact = contact_comparison(contact_samples, identity=self.identity,
                acquisition_identity=self.contact_acquisition_identity, before=previous,
                after=self.last_visual, now_ns=now)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            chain = []; cause = error
            while cause is not None:
                chain.append(str(cause)); cause = cause.__cause__
            self.failure = dict(decision_ns=now, chain=chain)
            self.contact = dict(status='VISUAL_TERMINAL_NO_COMPARISON', displacement_disagreement_m=None)
        return self.snapshot(now_ns=now)

    def snapshot(self, *, now_ns):
        now = _ns(now_ns, 'motion query')
        if self.last_decision_ns is not None and now < self.last_decision_ns:
            raise SensorContractError('motion query cannot move backwards')
        self.last_decision_ns = now
        age = None if self.last_visual is None else now-self.last_visual['measured_ns']
        current = self.failure is None and age == 0
        status = ('VISUAL_TERMINAL_FAILURE' if self.failure is not None else
                  'NO_VISUAL_OBSERVATION' if age is None else
                  'CURRENT_VISUAL_POSE' if current else 'TRANSLATION_AND_ORIENTATION_UNOBSERVED_SINCE_LAST_VISUAL')
        return deepcopy(dict(schema='visual_led_motion_evidence_development.v1', identity=self.identity,
            decision_ns=now, status=status, current_pose=self.last_visual if current else None,
            last_visual=self.last_visual, visual_age_ns=age, calibration_ids=self.calibrations,
            contact_diagnostic=self.contact, terminal_failure=self.failure,
            contact_diagnostic_current=current,
            pose_updated_from_contact=False, command_integration_used=False,
            contact_hardware_calibrated=False, physical_pose_error_bound=None,
            motion_permission='NOT_EVALUATED', support_established=False,
            future_clearance_established=False, navigation_qualified=False))
