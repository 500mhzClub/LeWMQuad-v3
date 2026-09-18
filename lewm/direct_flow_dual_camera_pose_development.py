"""Short-interval association fallback inside the existing continuity observer.

Try the complete original two-camera policy first. Only missing measured pose
support permits a second pass, and every qualified original witness remains a
conflict veto. No reference, pose, gyro or bridge history is reset.
"""
from copy import deepcopy
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.dual_camera_anchor_pose_development import DualCameraAnchorPose
from lewm.direct_corner_flow_association_development import tracked_points
from lewm.joint_rgbd_rigid_pose_development import register, angle, RIGID_RULES
from lewm.auxiliary_reference_pose_adapter_development import body_from_reference, gyro_in_reference, pose_in_body
from lewm.temporal_anchor_continuity_development import CONTINUITY_RULES

MISSING = ('NO_CURRENT_MEASURED_TRANSLATION', 'MEASURED_BRIDGE_BUDGET_EXHAUSTED')


class DirectFlowDualCameraAnchorPose(DualCameraAnchorPose):
    def __init__(self):
        super().__init__()
        self.direct_flow_mode = False
        self.direct_flow_now = None
        self.last_direct_flow_fallback = None

    def observe(self, *args, **kwargs):
        self.direct_flow_mode = False
        self.last_direct_flow_fallback = None
        return super().observe(*args, **kwargs)

    def _candidate(self, ref, current, G):
        try:
            return super()._candidate(ref,current,G)
        except SensorContractError as original_error:
            if (not self.direct_flow_mode or self.previous is None
                    or ref.frame != self.frame-1 or ref.frame != self.previous.frame
                    or self.direct_flow_now-ref.measured_ns != CONTINUITY_RULES['sample_interval_ns']):
                raise
            prior_reason=str(original_error)
        values, association=tracked_points(ref.features[self.camera],current[self.camera])
        a,b,ua,ub=values; relative_gyro=ref.gyro.T@G
        attempt=dict(camera=self.camera,reference_frame=ref.frame,current_frame=self.frame,
            original_pair_failure=prior_reason,association=association,qualified=False,failure=None)
        self.last_direct_flow_fallback['pair_attempts'].append(attempt)
        try:
            local_R,t,mask,reg=register(a,b,ua,ub,
                gyro_rotation=relative_gyro if self.camera=='primary' else gyro_in_reference(relative_gyro),
                mode='joint',frame=self.frame)
            if self.camera=='auxiliary':
                local_R,t=pose_in_body(local_R,t)
                A,offset=body_from_reference(); body_a=a@A.T+offset; body_b=b@A.T+offset
            else: body_a=a; body_b=b
            R=ref.rotation@local_R; p=ref.position+ref.rotation@t
            if (np.linalg.norm(t)>RIGID_RULES['maximum_reference_translation_m']
                    or np.linalg.norm(p-self.last_p)>RIGID_RULES['maximum_increment_translation_m']
                    or angle(self.last_R.T@R)>RIGID_RULES['maximum_increment_rotation_rad']):
                raise SensorContractError('direct-flow body-frame displacement envelope rejected')
            reg |= dict(reference_frame=ref.frame,reference_measured_ns=ref.measured_ns,
                translation_reference_body_m=t.tolist(),relative_rotation=local_R.tolist(),
                gyro_relative_rotation=relative_gyro.tolist(),reference_inlier_pixels=ua[mask].tolist(),
                current_inlier_pixels=ub[mask].tolist(),reference_inlier_points_body_m=body_a[mask].tolist(),
                current_inlier_points_body_m=body_b[mask].tolist(),camera=self.camera,
                direct_corner_flow_association=deepcopy(association))
            if self.camera=='auxiliary': reg['fixed_reference_frame_adapter_used']=True
            if len(self.rotation_measurements)>=9:
                raise SensorContractError('bounded eight anchors plus one increment required per camera')
            self.rotation_measurements.append(dict(reference_frame=ref.frame,reference_measured_ns=ref.measured_ns,
                current_frame=self.frame,reference_rotation_initial_body_from_reference_body=ref.rotation.tolist(),
                fitted_rotation_reference_body_from_current_body=local_R.tolist(),
                composed_rotation_initial_body_from_current_body=R.tolist(),gyro_rotation_reference_body_from_current_body=relative_gyro.tolist(),
                position_initial_body_m=p.tolist(),fitting_mode=reg['mode'],inliers=reg['inliers'],
                inlier_fraction=reg['inlier_fraction'],reference_grid_cells=reg['reference_grid_cells'],
                current_grid_cells=reg['current_grid_cells'],residual_rms_m=reg['residual_rms_m'],
                gyro_disagreement_rad=reg['gyro_disagreement_rad'],candidate_envelope_passed=True,
                witness_alone_grants_pose=False,camera=self.camera))
            attempt['qualified']=True
            return dict(reference=ref,R=R,p=p,local_R=local_R,t=t,registration=reg)
        except SensorContractError as error:
            attempt['failure']=str(error)
            raise

    def _measure(self, current, G, now):
        self.last_direct_flow_fallback=None
        try:
            return super()._measure(current,G,now)
        except SensorContractError as error:
            original_choice=deepcopy(self.last_camera_selection)
            original_continuity=deepcopy(self.last_continuity)
            primary=(original_choice or {}).get('primary_continuity') or {}
            if (not original_choice or original_choice.get('auxiliary_attempted') is not True
                    or primary.get('status') not in MISSING
                    or (original_continuity or {}).get('status') not in MISSING):
                raise
            original_failure=str(error)
        receipt=dict(frame=self.frame,measured_ns=now,original_failure=original_failure,
            original_camera_selection=original_choice,original_auxiliary_continuity=original_continuity,
            original_reference_selection=deepcopy(self.last_selection),pair_attempts=[],
            accepted=False,association_rule_changed=True,rigid_geometry_thresholds_unchanged=True,
            temporal_continuity_thresholds_unchanged=True,bridge_budget_unchanged=True,
            reference_or_pose_history_reset=False,original_qualified_measurements_checked=0)
        self.last_direct_flow_fallback=receipt
        self.direct_flow_mode=True; self.direct_flow_now=now
        try:
            candidate,alternative,bridge=super()._measure(current,G,now)
            for evidence in (primary,original_continuity):
                for witness in evidence.get('rotation_measurement_witnesses',[]):
                    distance=float(np.linalg.norm(candidate['p']-np.asarray(witness['position_initial_body_m'])))
                    rotation=angle(np.asarray(witness['composed_rotation_initial_body_from_current_body']).T@candidate['R'])
                    receipt['original_qualified_measurements_checked']+=1
                    if (distance>CONTINUITY_RULES['maximum_measured_disagreement_m']
                            or rotation>CONTINUITY_RULES['maximum_measured_disagreement_rad']):
                        self.last_continuity['status']='DIRECT_FLOW_ORIGINAL_MEASUREMENT_CONFLICT'
                        raise SensorContractError('direct-flow candidate conflicts with a qualified original measurement')
            self.last_camera_selection.update(thresholds_unchanged=False,
                association_rule_changed=True,rigid_geometry_thresholds_unchanged=True)
            receipt.update(accepted=True,selected_camera=self.camera,selected_reference=candidate['reference'].frame,
                selected_continuity_status=self.last_continuity['status'])
            return candidate,alternative,bridge
        except SensorContractError as error:
            receipt['failure']=str(error)
            raise
        finally:
            self.direct_flow_mode=False; self.direct_flow_now=None
