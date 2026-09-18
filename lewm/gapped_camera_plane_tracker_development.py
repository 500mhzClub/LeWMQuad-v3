"""Bounded camera gaps with continuous gyro and unchanged image-fit limits.

Frame numbers count processed visual observations; acquisition timestamps remain
real. This raw tracker is not admitted by the fixed-100-ms controller interface.
Existing direct/chained flow fallbacks still require their original image times.
"""
from copy import deepcopy
from types import FunctionType

import numpy as np

from lewm.camera_independent_gyro_development import CameraIndependentGyro
from lewm.causal_sensor_state import SensorContractError, _ns
from lewm.fast_gyro_development import validate_fast_packet
from lewm.dual_camera_anchor_pose_development import DualCameraAnchorPose
from lewm.temporal_anchor_continuity_development import TemporalAnchorRGBDPose, CONTINUITY_RULES
from lewm.sampled_plane_stop_conditioned_controller_development import SampledPlaneChainedPose
from lewm.sampled_plane_candidates_development import measured_candidates
from lewm.measured_plane_dual_camera_pose_development import (
    MeasuredPlaneDualCameraPose, PlaneImageConflict, refine, depth_hash,
    BODY_FROM_OPTICAL, body_from_optical, fit_joint_plane, validate_joint_plane, MISSING)

MAX_CAMERA_INTERVAL_NS = 500_000_000
MAX_BRIDGE_NS = 1_000_000_000


def _interval(model, now):
    _ns(now, 'camera acquisition')
    if model.previous is None:
        return 100_000_000
    elapsed = now-model.previous.measured_ns
    if elapsed % 100_000_000 or not 100_000_000 <= elapsed <= MAX_CAMERA_INTERVAL_NS:
        raise SensorContractError('camera interval must be 100–500 ms on the acquisition clock')
    return elapsed


def _with_interval(method, model, interval, *args, **kwargs):
    # Keep the method's original __class__ closure so cooperative super calls
    # retain their original place in the MRO. Only this call's clock rule varies.
    rules = CONTINUITY_RULES | dict(sample_interval_ns=interval)
    function = FunctionType(method.__code__, method.__globals__ | dict(CONTINUITY_RULES=rules),
        method.__name__, method.__defaults__, method.__closure__)
    function.__kwdefaults__ = method.__kwdefaults__
    return function(model, *args, **kwargs)


class _PreparedGyro:
    def __init__(self, stream):
        self.stream = stream

    def step(self, policy, fast, *, now_ns):
        identity = validate_fast_packet(fast, policy, now_ns=now_ns)
        if identity != self.stream.integrator.identity:
            raise SensorContractError('camera and integrated gyro episode must match')
        return self.stream.for_camera(measured_ns=now_ns)

    def begin(self, policy, fast, *, now_ns):
        result = self.step(policy, fast, now_ns=now_ns)
        if result['start_ns'] != now_ns:
            raise SensorContractError('first camera must establish the original gyro origin')
        return result


class _GappedTemporalAnchor(TemporalAnchorRGBDPose):
    def _measure(self, current, G, now):
        before = {k:getattr(self, k) for k in ('bridge_frames', 'total_bridge_frames', 'bridge_path_m')}
        result = _with_interval(TemporalAnchorRGBDPose._measure, self,
            _interval(self, now), current, G, now)
        if result[2] and now-self._last_anchored_visual_ns > MAX_BRIDGE_NS:
            for key, value in before.items():
                setattr(self, key, value)
            self.last_continuity.update(status='MEASURED_BRIDGE_BUDGET_EXHAUSTED',
                elapsed_since_anchor_ns=now-self._last_anchored_visual_ns)
            raise SensorContractError('measured bridge exceeds original one-second elapsed allowance')
        return result


class _GappedDualCamera(DualCameraAnchorPose, _GappedTemporalAnchor):
    def observe(self, policy, depth, fast, *, auxiliary_rgb, auxiliary_depth, now_ns):
        return _with_interval(DualCameraAnchorPose.observe, self, _interval(self, now_ns),
            policy, depth, fast, auxiliary_rgb=auxiliary_rgb,
            auxiliary_depth=auxiliary_depth, now_ns=now_ns)


class GappedCameraPlaneTracker(SampledPlaneChainedPose, _GappedDualCamera):
    def __init__(self):
        super().__init__()
        self.gyro_stream = CameraIndependentGyro()
        self.gyro = _PreparedGyro(self.gyro_stream)
        self._last_anchored_visual_ns = None

    def ingest_gyro(self, policy, fast, *, now_ns):
        if self.failed:
            raise SensorContractError('gapped camera tracker terminal')
        try:
            return self.gyro_stream.observe(policy, fast, now_ns=now_ns)
        except (ValueError, TypeError, KeyError, IndexError):
            self.failed = True
            raise

    def observe(self, *args, **kwargs):
        prior = None if self.previous is None else self.previous.measured_ns
        result = super().observe(*args, **kwargs)
        now = result['measured_ns']
        if self.last_continuity['status'] in ('INITIAL_REFERENCE', 'ANCHOR_MEASUREMENT'):
            self._last_anchored_visual_ns = now
        return result | dict(camera_gap_tracker=True,
            acquisition_frame=(now-self._origin_ns)//100_000_000,
            previous_visual_measured_ns=prior,
            visual_interval_ns=None if prior is None else now-prior,
            maximum_visual_interval_ns=MAX_CAMERA_INTERVAL_NS,
            frame_counts_processed_visual_observations=True,
            continuous_gyro_intervals=self.gyro_stream.for_camera(measured_ns=now)['samples_integrated'],
            original_temporal_gate_values_unchanged=False,
            image_fit_limits_unchanged=True, maximum_bridge_elapsed_ns=MAX_BRIDGE_NS,
            fixed_cadence_controller_compatible=False, processing_latency_accounted=False)

    def _prepare_plane(self, current, G, now):
        if self._origin_ns is None or current['primary'].depth['measured_ns'] != now:
            raise SensorContractError('actual current camera timestamp required for plane association')
        if self._pending_plane is not None:
            frame, features, receipt = self._pending_plane
            if frame == self.frame:
                if features is not current or receipt['measured_ns'] != now:
                    raise SensorContractError('one paired floor acquisition per visual observation required')
                return receipt
        up = self.last_R.T@self._initial_up
        clouds = []; hashes = {}
        for camera, E in (('primary', np.asarray(BODY_FROM_OPTICAL)), ('auxiliary', body_from_optical())):
            depth = current[camera].depth
            if depth['measured_ns'] != now or depth['available_ns'] > now:
                raise SensorContractError('co-timed available paired floor depth required')
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

    def _candidate(self, ref, current, G):
        # Original measured-plane refinement with actual current acquisition time.
        if self._plane_conflict is not None:
            raise PlaneImageConflict(self._plane_conflict)
        if ref.frame not in self._planes:
            raise SensorContractError('measured plane for retained reference required')
        features, receipt = self._planes[ref.frame]
        if features is not ref.features or receipt['measured_ns'] != ref.measured_ns:
            raise SensorContractError('retained feature and floor acquisition ownership required')
        pending = self._prepare_plane(current, G, current['primary'].depth['measured_ns'])
        start = len(self.rotation_measurements)
        original = super(MeasuredPlaneDualCameraPose, self)._candidate(ref, current, G)
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
            candidate = refine(original, receipt['joint_plane'], pending['joint_plane'],
                camera=self.camera, gyro=ref.gyro.T@G, last_p=self.last_p, last_R=self.last_R)
            reg = candidate['registration']
            reg['measured_plane_refinement'].update(applied=True,
                reference_floor=deepcopy(receipt), current_floor=deepcopy(pending))
            witness = deepcopy(self.rotation_measurements[-1])
            witness.update(fitted_rotation_reference_body_from_current_body=candidate['local_R'].tolist(),
                composed_rotation_initial_body_from_current_body=candidate['R'].tolist(),
                position_initial_body_m=candidate['p'].tolist(), residual_rms_m=reg['residual_rms_m'],
                gyro_disagreement_rad=reg['gyro_disagreement_rad'], measured_plane_constrained=True)
            self.rotation_measurements[-1] = witness
            return candidate
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
            del self.rotation_measurements[start:]
            if isinstance(error, PlaneImageConflict):
                self._plane_conflict = str(error)
            raise SensorContractError('measured-plane pair refinement rejected') from error
