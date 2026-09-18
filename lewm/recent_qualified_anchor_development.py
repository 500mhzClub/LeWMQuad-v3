"""One causal qualified previous view, after the original retained references.

The additional reference is saved only after an accepted anchor measurement.
It is never created from a bridge or floor-transport pose. Original reference
search, rigid fits, conflict vetoes, camera priority and bridge gates remain.
Frequent qualified chains still have uncalibrated accumulated pose error.
"""
from copy import deepcopy
import numpy as np
from lewm.joint_rgbd_rigid_pose_development import angle
from lewm.causal_sensor_state import SensorContractError
from lewm.direct_flow_dual_camera_pose_development import DirectFlowDualCameraAnchorPose
from lewm.temporal_anchor_continuity_development import CONTINUITY_RULES,TemporalAnchorRGBDPose


class _RecentReferenceSearch(TemporalAnchorRGBDPose):
    # The original TemporalAnchor method calls super()._choose directly.
    # This explicit copy changes only that call to instance dispatch; the
    # original JointTemporal wrapper still owns every rotation witness.
    def _measure(self, current, G, now):
        """Both hypotheses use the unchanged descriptor matcher and rigid gates."""
        if self.previous is None or now - self.previous.measured_ns != CONTINUITY_RULES['sample_interval_ns']:
            raise SensorContractError('immediately preceding accepted visual observation required')
        evidence = dict(status='MEASURING', incremental_available=False,
            anchor_available=False, measurements_independent=False,
            previous_frame=self.previous.frame, previous_measured_ns=self.previous.measured_ns,
            disagreement_m=None, disagreement_rad=None, anchor_failure=None,
            incremental_failure=None, error_bound_m=None, uncertainty_calibrated=False)
        self.last_continuity = evidence
        anchor = None
        try:
            anchor, alternative = self._choose(current, G)
            evidence['anchor_available'] = True
        except SensorContractError as error:
            evidence['anchor_failure'] = str(error)
            # Contradictory qualified anchors are not mere missingness.
            if self.last_selection['status'] != 'NO_QUALIFIED_REFERENCE':
                evidence['status'] = 'ANCHOR_CONFLICT_OR_INVALID'
                raise
        incremental = None
        if anchor is not None and anchor['reference'].frame == self.previous.frame:
            incremental = anchor
            evidence['same_reference_measurement_reused'] = True
        else:
            evidence['same_reference_measurement_reused'] = False
            try:
                incremental = self._candidate(self.previous, current, G)
            except SensorContractError as error:
                evidence['incremental_failure'] = str(error)
        evidence['incremental_available'] = incremental is not None
        if incremental is not None:
            evidence['incremental_position_initial_body_m'] = incremental['p'].tolist()
        if anchor is not None:
            if incremental is not None:
                distance = float(np.linalg.norm(anchor['p'] - incremental['p']))
                rotation = angle(anchor['R'].T @ incremental['R'])
                evidence.update(disagreement_m=distance, disagreement_rad=rotation)
                if (distance > CONTINUITY_RULES['maximum_measured_disagreement_m']
                        or rotation > CONTINUITY_RULES['maximum_measured_disagreement_rad']):
                    evidence['status'] = 'ANCHOR_INCREMENT_CONFLICT'
                    self.last_selection = self.last_selection | dict(status='ANCHOR_INCREMENT_CONFLICT')
                    raise SensorContractError('qualified anchor and incremental measurements conflict')
            evidence.update(status='ANCHOR_MEASUREMENT', preceding_bridge_frames=self.bridge_frames,
                preceding_bridge_path_m=self.bridge_path_m, bridge_frames=0, bridge_path_m=0.)
            self.bridge_frames = 0
            self.bridge_path_m = 0.
            return anchor, alternative, False
        if incremental is None:
            evidence['status'] = 'NO_CURRENT_MEASURED_TRANSLATION'
            raise SensorContractError('neither retained anchor nor previous frame supports current pose')
        if self.bridge_frames >= CONTINUITY_RULES['maximum_bridge_frames']:
            evidence['status'] = 'MEASURED_BRIDGE_BUDGET_EXHAUSTED'
            raise SensorContractError('bounded measured bridge exhausted without anchor observation')
        self.bridge_frames += 1
        self.total_bridge_frames += 1
        self.bridge_path_m += float(np.linalg.norm(incremental['p'] - self.last_p))
        evidence.update(status='MEASURED_INCREMENT_BRIDGE', bridge_frames=self.bridge_frames,
            bridge_path_m=self.bridge_path_m, total_bridge_frames=self.total_bridge_frames,
            anchored_error_accumulates=True)
        self.last_selection = self.last_selection | dict(status='MEASURED_INCREMENT_BRIDGE',
            anchor_status='NO_QUALIFIED_REFERENCE', selected_reference=self.previous.frame,
            selected_reference_retained_anchor=False)
        return incremental, False, True


class RecentQualifiedAnchorPose(DirectFlowDualCameraAnchorPose, _RecentReferenceSearch):
    def __init__(self):
        super().__init__()
        self.recent_qualified_reference = None
        self.recent_query_ns = None
        self.last_recent_qualified_anchor = None

    def _choose(self, current, G):
        try:
            return super()._choose(current,G)
        except SensorContractError as original_error:
            # All original alternatives must have failed; a conflict is final.
            if (self.last_selection or {}).get('status') != 'NO_QUALIFIED_REFERENCE':
                raise
            ref = self.recent_qualified_reference
            if (ref is None or ref is not self.previous or ref.frame != self.frame-1
                    or self.recent_query_ns is None
                    or ref.measured_ns != self.recent_query_ns-CONTINUITY_RULES['sample_interval_ns']
                    or any(r.frame == ref.frame for r in self.references)):
                raise
            original = deepcopy(self.last_selection)
            receipt = dict(camera=self.camera,current_frame=self.frame,
                reference_frame=ref.frame,reference_measured_ns=ref.measured_ns,
                reference_was_anchor_qualified=True,reference_is_immediately_previous=True,
                original_reference_selection=original,qualified=False,failure=None,
                direct_flow_mode=self.direct_flow_mode,rigid_thresholds_unchanged=True)
            self.last_recent_qualified_anchor['attempts'].append(receipt)
            try:
                candidate = self._candidate(ref,current,G)
            except SensorContractError as error:
                receipt['failure'] = str(error)
                # Preserve the original no-anchor result and normal increment path.
                raise original_error
            receipt['qualified'] = True
            self.last_selection = original|dict(status='RECENT_QUALIFIED_REFERENCE_ACCEPTED',
                selected_reference=ref.frame,selected_reference_retained_anchor=True,
                recent_qualified_reference_used=True,additional_retained_references=1)
            return candidate,True

    def _measure(self, current, G, now):
        self.recent_query_ns = now
        try:
            return super()._measure(current,G,now)
        finally:
            self.recent_query_ns = None

    def observe(self, *args, **kwargs):
        retained = self.recent_qualified_reference
        self.last_recent_qualified_anchor = dict(
            retained_from_frame=None if retained is None else retained.frame,
            attempts=[],retained_current_frame=None,maximum_additional_references=1,
            original_reference_population_preserved_before_search=True,
            reference_retained_from_bridge=False,reference_retained_from_floor_transport=False,
            bridge_limit_unchanged=True,pose_uncertainty_calibrated=False)
        try:
            result = super().observe(*args,**kwargs)
        except Exception:
            self.recent_qualified_reference = None
            raise
        if self.last_continuity['status'] in ('INITIAL_REFERENCE','ANCHOR_MEASUREMENT'):
            self.recent_qualified_reference = self.previous
            self.last_recent_qualified_anchor['retained_current_frame'] = self.previous.frame
        else:
            self.recent_qualified_reference = None
        return result
