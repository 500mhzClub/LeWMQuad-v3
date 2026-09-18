"""Use explicit paired height-selected floor points in three sensor consumers."""
from lewm.eligible_floor_registration_development import bind
from lewm.robust_height_floor_candidates_development import PairedHeightCandidates
from lewm.measured_plane_dual_camera_pose_development import MeasuredPlaneDualCameraPose
from lewm.local_feature_depth_consensus_development import (
    LocalFeatureDepthConsensusPose, LocalFeatureDepthConsensusMotion)
from lewm.partial_floor_height_development import PartialHeightRegistration
from lewm.partial_height_round_trip_development import PartialHeightObstacles, _partial_observe
from lewm.fine_obstacle_round_trip_development import FineDepthObstacles

DESCRIPTION = dict(floor_candidate_depth_source='paired_raw_depth_dominant_height_cluster',
    floor_candidates_are_raw_pixel_depth=True, floor_candidate_selection_changed=True,
    original_mesh_normal_candidate_rule_used=False, raw_pool_outliers_explicitly_excluded=True,
    invalid_depth_filled=False, downstream_plane_acceptance_thresholds_changed=False)


class RobustHeightFloorPose(LocalFeatureDepthConsensusPose):
    def _prepare_plane(self, current, G, now):
        selector = PairedHeightCandidates(current['primary'].depth, current['auxiliary'].depth)
        receipt = bind(MeasuredPlaneDualCameraPose._prepare_plane,
            measured_candidates=selector)(self,current,G,now)
        receipt.update(DESCRIPTION, image_feature_depth_changed=True)
        if selector.receipt is not None:
            receipt['floor_candidate_selection'] = selector.receipt
        return receipt


class RobustHeightFloorMotion(LocalFeatureDepthConsensusMotion):
    def __init__(self, *, identity=(0,0,0), activation_frame=0):
        super().__init__(identity=identity, activation_frame=activation_frame)
        self.model = RobustHeightFloorPose(activation_frame=activation_frame)

    def snapshot(self, *, now_ns):
        return super().snapshot(now_ns=now_ns) | DESCRIPTION


class RobustHeightFloorRegistration(PartialHeightRegistration):
    def observe(self, policy, primary, auxiliary, raw, *, now_ns):
        selector = PairedHeightCandidates(primary,auxiliary)
        result = bind(PartialHeightRegistration.observe, measured_candidates=selector)(
            self,policy,primary,auxiliary,raw,now_ns=now_ns)
        return result | DESCRIPTION | dict(image_feature_depth_changed=True,
            floor_candidate_selection=selector.receipt)


class RobustHeightIndependentObstacles(PartialHeightObstacles):
    def _observe(self, policy, depth, fast, auxiliary, now):
        selector = PairedHeightCandidates(depth,auxiliary)
        extract = FineDepthObstacles._observe if self.frames == 0 else _partial_observe
        current = bind(extract, measured_candidates=selector)(self,policy,depth,fast,auxiliary,now)
        self.receipts[-1].update(DESCRIPTION, floor_candidate_selection=selector.receipt,
            obstacle_points_use_original_depth=True)
        return current
