"""Separate causal floor-constrained estimator with original camera/continuity gates.

Each pair first passes the original unconstrained robust image registration.
Its complete accepted point set is refined against two measured joint planes.
All original points must still pass both-image reprojection and residual gates.
No prior pose, reference, or map is reset when a measurement is rejected.
"""
from copy import deepcopy
import hashlib
import cv2
import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical
from lewm.auxiliary_reference_pose_adapter_development import body_from_reference
from lewm.dual_camera_anchor_pose_development import DualCameraAnchorPose
from lewm.floor_pose_registration_development import measured_candidates
from lewm.joint_measured_floor_plane_development import fit_joint_plane, validate_joint_plane
from lewm.joint_floor_registered_evidence_development import depth_hash
from lewm.joint_rgbd_rigid_pose_development import inliers, angle, RIGID_RULES
from lewm.temporal_anchor_continuity_development import CONTINUITY_RULES
from lewm.measured_plane_rigid_fit_development import fit
from lewm.measured_floor_transport_development import MISSING


class PlaneImageConflict(SensorContractError):
    """Qualified image and plane-constrained estimates cannot be treated as missing."""


def refine(candidate, reference_plane, current_plane, *, camera, gyro, last_p, last_R):
    original = candidate['registration']
    a = np.asarray(original['reference_inlier_points_body_m'])
    b = np.asarray(original['current_inlier_points_body_m'])
    ua = np.asarray(original['reference_inlier_pixels'])
    ub = np.asarray(original['current_inlier_pixels'])
    if len(a) != original['inliers']:
        raise SensorContractError('complete original accepted point population required')
    R, t, fitted = fit(a, b, reference_plane, current_plane)
    # The image projection helpers use the primary optical adapter coordinate
    # system, even for auxiliary images. Convert the body fit back for checking.
    if camera == 'primary':
        image_a, image_b, image_R, image_t = a, b, R, t
    elif camera == 'auxiliary':
        A, offset = body_from_reference()
        image_a, image_b = (a-offset)@A, (b-offset)@A
        image_R, image_t = A.T@R@A, A.T@(t-offset+R@offset)
    else:
        raise SensorContractError('explicit primary or auxiliary pair required')
    keep, residual = inliers(image_a, image_b, ua, ub, image_R, image_t)
    if not keep.all():
        raise SensorContractError('plane refinement loses an original image inlier')
    ref = candidate['reference']
    global_R, global_p = ref.rotation@R, ref.position+ref.rotation@t
    disagreement = angle(gyro.T@R)
    if (np.linalg.norm(t) > RIGID_RULES['maximum_reference_translation_m']
            or np.linalg.norm(global_p-last_p) > RIGID_RULES['maximum_increment_translation_m']
            or angle(last_R.T@global_R) > RIGID_RULES['maximum_increment_rotation_rad']
            or disagreement > RIGID_RULES['maximum_gyro_disagreement_rad']):
        raise SensorContractError('plane-refined pose fails original motion or gyro gates')
    displacement = float(np.linalg.norm(global_p-candidate['p']))
    rotation = angle(candidate['R'].T@global_R)
    if (displacement > CONTINUITY_RULES['maximum_measured_disagreement_m']
            or rotation > CONTINUITY_RULES['maximum_measured_disagreement_rad']):
        raise PlaneImageConflict('qualified image and measured-plane estimates conflict')
    reg = deepcopy(original)
    reg.update(translation_reference_body_m=t.tolist(), relative_rotation=R.tolist(),
        residual_rms_m=float(np.sqrt(np.mean(residual**2))), gyro_disagreement_rad=disagreement,
        measured_plane_refinement=dict(fit=fitted, original_inliers_preserved=True,
            original_inlier_array_bindings={name: dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes()).hexdigest())
                for name, value in (('reference_body_points', a), ('current_body_points', b),
                    ('reference_pixels', ua), ('current_pixels', ub))},
            original_unconstrained_relative_rotation=candidate['local_R'].tolist(),
            original_unconstrained_translation_m=candidate['t'].tolist(),
            original_unconstrained_residual_rms_m=original['residual_rms_m'],
            original_unconstrained_gyro_disagreement_rad=original['gyro_disagreement_rad'],
            original_image_pose_disagreement_m=displacement,
            original_image_pose_disagreement_rad=rotation,
            original_robust_consensus_only=True, original_image_gate_values_unchanged=True))
    return dict(reference=ref, R=global_R, p=global_p, local_R=R, t=t, registration=reg)


class MeasuredPlaneDualCameraPose(DualCameraAnchorPose):
    _refine_candidate = staticmethod(refine)

    def __init__(self):
        super().__init__()
        self._initial_up = None
        self._origin_ns = None
        self._planes = {}
        self._pending_plane = None
        self._plane_conflict = None
        self.last_measured_plane = None
        self.last_measured_plane_refinement = None

    def _prepare_plane(self, current, G, now):
        if self._origin_ns is None or now != self._origin_ns+self.frame*100_000_000:
            raise SensorContractError('exact original floor acquisition sequence required')
        if self._pending_plane is not None:
            frame, features, receipt = self._pending_plane
            if frame == self.frame:
                if features is not current or receipt['measured_ns'] != now:
                    raise SensorContractError('one paired floor acquisition per frame required')
                return receipt
        # Candidate floor extraction uses the preceding admitted visual attitude,
        # not a new gyro-derived pose. Current depth supplies the new normal.
        up = self.last_R.T@self._initial_up
        clouds = []
        hashes = {}
        for camera, E in (('primary', np.asarray(BODY_FROM_OPTICAL)), ('auxiliary', body_from_optical())):
            depth = current[camera].depth
            if depth['measured_ns'] != now or depth['available_ns'] > now:
                raise SensorContractError('current available paired floor depth required')
            clouds.append(measured_candidates(depth['depth_m'], depth['valid'], E, up)[0])
            hashes[camera] = depth_hash(depth)
        plane = fit_joint_plane(*clouds, up)
        if plane['available']:
            validate_joint_plane(plane, up)
        elif self.frame == 0 or plane['reason'] not in MISSING:
            raise SensorContractError('initial or conflicting measured floor plane unavailable')
        receipt = dict(frame=self.frame, measured_ns=now, joint_plane=plane, depth_sha256=hashes,
            initial_up_body=self._initial_up.tolist(), current_up_uses_public_gyro=False,
            up_reference_visual_frame=None if self.frame == 0 else self.frame-1,
            up_reference_rotation=self.last_R.tolist(),
            initial_up_uses_quiet_specific_force=True, static_same_floor_hypothesis=True,
            raw_native_pose_used=False, floor_identity_certified=False)
        self._pending_plane = (self.frame, current, receipt)
        return receipt

    def _remember(self, current, R, G, p, now):
        receipt = self._prepare_plane(current, G, now)
        super()._remember(current, R, G, p, now)
        self._planes[self.frame] = (current, deepcopy(receipt))

    def _candidate(self, ref, current, G):
        if self._plane_conflict is not None:
            raise PlaneImageConflict(self._plane_conflict)
        if ref.frame not in self._planes:
            raise SensorContractError('measured plane for original retained reference required')
        features, receipt = self._planes[ref.frame]
        if features is not ref.features or receipt['measured_ns'] != ref.measured_ns:
            raise SensorContractError('exact retained feature and floor acquisition ownership required')
        pending = self._prepare_plane(current, G, self.previous.measured_ns+100_000_000)
        start = len(self.rotation_measurements)
        original = super()._candidate(ref, current, G)
        try:
            if len(self.rotation_measurements) != start+1:
                raise SensorContractError('one original qualified rotation witness required')
            if not receipt['joint_plane']['available'] or not pending['joint_plane']['available']:
                original['registration']['measured_plane_refinement'] = dict(
                    applied=False, reason='reference_or_current_floor_support_missing',
                    reference_floor=deepcopy(receipt), current_floor=deepcopy(pending),
                    original_image_fit_retained=True, missing_plane_not_admitted=True)
                self.rotation_measurements[-1]['measured_plane_constrained'] = False
                return original
            candidate = self._refine_candidate(original, receipt['joint_plane'], pending['joint_plane'],
                camera=self.camera, gyro=ref.gyro.T@G, last_p=self.last_p, last_R=self.last_R)
            reg = candidate['registration']
            applied = reg['measured_plane_refinement'].get('applied', True)
            reg['measured_plane_refinement'].update(applied=applied, reference_floor=deepcopy(receipt),
                current_floor=deepcopy(pending))
            witness = deepcopy(self.rotation_measurements[-1])
            witness.update(fitted_rotation_reference_body_from_current_body=candidate['local_R'].tolist(),
                composed_rotation_initial_body_from_current_body=candidate['R'].tolist(),
                position_initial_body_m=candidate['p'].tolist(), residual_rms_m=reg['residual_rms_m'],
                gyro_disagreement_rad=reg['gyro_disagreement_rad'], measured_plane_constrained=applied)
            self.rotation_measurements[-1] = witness
            return candidate
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
            # Never expose the preliminary unconstrained witness as if it were
            # an admitted constrained fit after refinement failed.
            del self.rotation_measurements[start:]
            if isinstance(error, PlaneImageConflict): self._plane_conflict = str(error)
            raise SensorContractError('measured-plane pair refinement rejected') from error

    def _measure(self, current, G, now):
        self._prepare_plane(current, G, now)
        try:
            return super()._measure(current, G, now)
        finally:
            if self._plane_conflict is not None:
                self.last_continuity['status'] = 'MEASURED_PLANE_IMAGE_CONFLICT'
                raise PlaneImageConflict(self._plane_conflict)

    def observe(self, policy, depth, fast, *, auxiliary_rgb, auxiliary_depth, now_ns):
        if self.failed:
            raise SensorContractError('measured-plane observer terminal; no recovery or reinitialization')
        try:
            if self._initial_up is None:
                force = policy['sensor_state']['sensed']['specific_force']
                command = policy['sensor_state']['control']['applied_command']
                up = force['values'].mean(0)
                magnitude = np.linalg.norm(up)
                if (not force['valid'].all() or not command['valid'].all()
                        or np.any(np.abs(command['values']) > 1e-8) or not 8 <= magnitude <= 12):
                    raise SensorContractError('quiet initial public gravity reference required')
                self._initial_up = up/magnitude
                self._origin_ns = now_ns
            result = super().observe(policy, depth, fast, auxiliary_rgb=auxiliary_rgb,
                auxiliary_depth=auxiliary_depth, now_ns=now_ns)
            frame, current, receipt = self._pending_plane
            if frame != self.frame or current is not self.previous.features:
                raise SensorContractError('accepted floor and visual frames must share current features')
            self._planes[frame] = (current, deepcopy(receipt))
            retained = {r.frame for r in self.references} | {self.previous.frame}
            self._planes = {f:v for f,v in self._planes.items() if f in retained}
            if len(self._planes) > 9 or set(self._planes) != retained:
                raise SensorContractError('exact eight retained plus previous floor population required')
            self.last_measured_plane = deepcopy(receipt)
            self.last_measured_plane_refinement = deepcopy(
                (result['registration'] or {}).get('measured_plane_refinement'))
            return result | dict(measured_plane_constrained_estimator=True,
                measured_plane_evidence=deepcopy(receipt), original_global_floor_gate_unchanged=True,
                original_temporal_gate_values_unchanged=True, reference_history_reset=False)
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError, cv2.error, np.linalg.LinAlgError):
            self.failed = True
            raise
