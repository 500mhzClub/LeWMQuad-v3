"""Separate retained-anchor reacquisition candidate, with original conflict vetoes.

Try the existing direct-flow observer first. Only its measured bridge or missing
pose can trigger chained image association. Original successful anchor poses,
qualified conflicts, reference promotion and the bridge allowance are retained.
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.direct_flow_dual_camera_pose_development import DirectFlowDualCameraAnchorPose, MISSING
from lewm.chained_corner_flow_association_development import chained_points, CHAIN_RULES
from lewm.joint_rgbd_rigid_pose_development import register, angle, RIGID_RULES
from lewm.auxiliary_reference_pose_adapter_development import body_from_reference, gyro_in_reference, pose_in_body
from lewm.temporal_anchor_continuity_development import CONTINUITY_RULES

TRACE_FIELDS = ('camera', 'last_camera_selection', 'last_continuity', 'last_selection',
                'rotation_measurements', 'last_direct_flow_fallback')
BRIDGE_FIELDS = ('bridge_frames', 'total_bridge_frames', 'bridge_path_m')


class ChainedAnchorDualCameraPose(DirectFlowDualCameraAnchorPose):
    def __init__(self):
        super().__init__()
        self._chain_mode = False
        self._image_history = {}
        self.last_chained_anchor_fallback = None

    def _cache_images(self, current, now):
        if self.frame in self._image_history:
            if self._image_history[self.frame][0] != now:
                raise SensorContractError('same cached image frame requires same measured clock')
            return
        views = {}
        for camera in ('primary', 'auxiliary'):
            source = current[camera]
            gray = source.gray.copy()
            depth = {k: source.depth[k].copy() for k in ('depth_m', 'valid')}
            gray.setflags(write=False)
            for value in depth.values(): value.setflags(write=False)
            views[camera] = SimpleNamespace(gray=gray, depth=depth)
        self._image_history[self.frame] = (now, views)
        oldest = self.frame-CHAIN_RULES['maximum_intervals']
        self._image_history = {f: v for f, v in self._image_history.items() if f >= oldest}

    def observe(self, *args, **kwargs):
        self._chain_mode = False
        self.last_chained_anchor_fallback = None
        result = super().observe(*args, **kwargs)
        # Frame zero has no _measure call; later frames were cached on entry.
        self._cache_images(self.previous.features, self.previous.measured_ns)
        return result

    def _candidate(self, ref, current, G):
        try:
            return super()._candidate(ref, current, G)
        except SensorContractError as error:
            if (not self._chain_mode or not any(ref is r for r in self.references)
                    or not 2 <= self.frame-ref.frame <= CHAIN_RULES['maximum_intervals']):
                raise
            original_failure = str(error)
        needed = tuple(range(ref.frame, self.frame+1))
        if any(f not in self._image_history for f in needed):
            raise SensorContractError('complete retained-reference image chain required')
        sequence = [(f, self._image_history[f][0], self._image_history[f][1][self.camera]) for f in needed]
        if sequence[0][1] != ref.measured_ns:
            raise SensorContractError('retained reference clock must match image chain')
        sequence[0] = (ref.frame, ref.measured_ns, ref.features[self.camera])
        values, association = chained_points(sequence)
        a, b, ua, ub = values
        relative_gyro = ref.gyro.T@G
        attempt = dict(camera=self.camera, reference_frame=ref.frame, current_frame=self.frame,
            original_pair_failure=original_failure, association=association, qualified=False, failure=None)
        self.last_chained_anchor_fallback['pair_attempts'].append(attempt)
        try:
            local_R, t, mask, reg = register(a, b, ua, ub,
                gyro_rotation=relative_gyro if self.camera == 'primary' else gyro_in_reference(relative_gyro),
                mode='joint', frame=self.frame)
            if self.camera == 'auxiliary':
                local_R, t = pose_in_body(local_R, t)
                A, offset = body_from_reference()
                body_a, body_b = a@A.T+offset, b@A.T+offset
            else:
                body_a, body_b = a, b
            R, p = ref.rotation@local_R, ref.position+ref.rotation@t
            if (np.linalg.norm(t) > RIGID_RULES['maximum_reference_translation_m'] or
                    np.linalg.norm(p-self.last_p) > RIGID_RULES['maximum_increment_translation_m'] or
                    angle(self.last_R.T@R) > RIGID_RULES['maximum_increment_rotation_rad']):
                raise SensorContractError('chained-anchor body-frame displacement envelope rejected')
            reg |= dict(reference_frame=ref.frame, reference_measured_ns=ref.measured_ns,
                translation_reference_body_m=t.tolist(), relative_rotation=local_R.tolist(),
                gyro_relative_rotation=relative_gyro.tolist(), reference_inlier_pixels=ua[mask].tolist(),
                current_inlier_pixels=ub[mask].tolist(), reference_inlier_points_body_m=body_a[mask].tolist(),
                current_inlier_points_body_m=body_b[mask].tolist(), camera=self.camera,
                chained_corner_flow_association=deepcopy(association))
            if self.camera == 'auxiliary': reg['fixed_reference_frame_adapter_used'] = True
            if len(self.rotation_measurements) >= 9:
                raise SensorContractError('bounded eight anchors plus one increment required per camera')
            self.rotation_measurements.append(dict(reference_frame=ref.frame, reference_measured_ns=ref.measured_ns,
                current_frame=self.frame, reference_rotation_initial_body_from_reference_body=ref.rotation.tolist(),
                fitted_rotation_reference_body_from_current_body=local_R.tolist(),
                composed_rotation_initial_body_from_current_body=R.tolist(),
                gyro_rotation_reference_body_from_current_body=relative_gyro.tolist(),
                position_initial_body_m=p.tolist(), fitting_mode=reg['mode'], inliers=reg['inliers'],
                inlier_fraction=reg['inlier_fraction'], reference_grid_cells=reg['reference_grid_cells'],
                current_grid_cells=reg['current_grid_cells'], residual_rms_m=reg['residual_rms_m'],
                gyro_disagreement_rad=reg['gyro_disagreement_rad'], candidate_envelope_passed=True,
                witness_alone_grants_pose=False, camera=self.camera))
            attempt['qualified'] = True
            return dict(reference=ref, R=R, p=p, local_R=local_R, t=t, registration=reg)
        except SensorContractError as error:
            attempt['failure'] = str(error)
            raise

    def _measure(self, current, G, now):
        self._cache_images(current, now)
        self.last_chained_anchor_fallback = None
        before = {k: getattr(self, k) for k in BRIDGE_FIELDS}
        original, original_error = None, None
        try:
            original = super()._measure(current, G, now)
            if not original[2]: return original
        except SensorContractError as error:
            if (self.last_continuity or {}).get('status') not in MISSING: raise
            original_error = error
        saved = {k: deepcopy(getattr(self, k)) for k in (*TRACE_FIELDS, *BRIDGE_FIELDS)}
        evidence = [saved['last_continuity'], (saved['last_camera_selection'] or {}).get('primary_continuity')]
        direct = saved['last_direct_flow_fallback'] or {}
        evidence += [direct.get('original_auxiliary_continuity'),
                     (direct.get('original_camera_selection') or {}).get('primary_continuity')]
        witnesses = [w for item in evidence if item for w in item.get('rotation_measurement_witnesses', [])]
        receipt = dict(frame=self.frame, measured_ns=now, original_bridge_available=original is not None,
            original_failure=None if original_error is None else str(original_error),
            original_continuity=deepcopy(saved['last_continuity']),
            original_camera_selection=deepcopy(saved['last_camera_selection']),
            original_direct_flow_fallback=deepcopy(saved['last_direct_flow_fallback']),
            bridge_state_before=before, pair_attempts=[], accepted=False,
            original_qualified_measurements_checked=0, bridge_budget_unchanged=True,
            rigid_geometry_thresholds_unchanged=True, temporal_continuity_thresholds_unchanged=True,
            reference_or_pose_history_reset=False, pose_increments_composed=False)
        self.last_chained_anchor_fallback = receipt
        for k, v in before.items(): setattr(self, k, v)
        self._chain_mode = True
        try:
            try:
                result = super()._measure(current, G, now)
            except SensorContractError as error:
                receipt['failure'] = str(error)
                if (self.last_continuity or {}).get('status') not in MISSING: raise
                result = None
            if result is None or result[2]:
                # An unhelpful search must not spend the bridge allowance twice
                # or replace the original terminal evidence with a new failure.
                for k, v in saved.items(): setattr(self, k, v)
                receipt['original_result_restored'] = True
                if original is not None: return original
                raise original_error
            candidate = result[0]
            if 'chained_corner_flow_association' not in candidate['registration']:
                raise SensorContractError('reacquisition requires an actual chained retained-anchor fit')
            for witness in witnesses:
                distance = float(np.linalg.norm(candidate['p']-np.asarray(witness['position_initial_body_m'])))
                rotation = angle(np.asarray(witness['composed_rotation_initial_body_from_current_body']).T@candidate['R'])
                receipt['original_qualified_measurements_checked'] += 1
                if (distance > CONTINUITY_RULES['maximum_measured_disagreement_m'] or
                        rotation > CONTINUITY_RULES['maximum_measured_disagreement_rad']):
                    self.last_continuity['status'] = 'CHAINED_ANCHOR_ORIGINAL_MEASUREMENT_CONFLICT'
                    raise SensorContractError('chained anchor conflicts with a qualified original measurement')
            self.last_camera_selection.update(thresholds_unchanged=False, association_rule_changed=True,
                                              rigid_geometry_thresholds_unchanged=True)
            receipt.update(accepted=True, selected_camera=self.camera,
                selected_reference=candidate['reference'].frame,
                selected_continuity_status=self.last_continuity['status'])
            return result
        except SensorContractError as error:
            receipt['failure'] = str(error)
            raise
        finally:
            self._chain_mode = False
