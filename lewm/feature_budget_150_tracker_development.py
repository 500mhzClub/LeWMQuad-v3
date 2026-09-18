"""150 spatially distributed corners per camera, retaining geometry gates."""
from lewm.corner_support_features_development import CornerSupportFeatureFrame,CORNER_RULES
from lewm.eligible_floor_registration_development import bind
from lewm.dual_camera_anchor_pose_development import DualCameraAnchorPose
from lewm.batched_patch_tracker_development import BatchedPatchPose,BatchedPatchVisualMotion,use_batch
from lewm.spatial_support_features_development import cell_index

RULES=CORNER_RULES|dict(per_cell=13,maximum_selected=156)
_construct=bind(CornerSupportFeatureFrame.__init__,CORNER_RULES=RULES)


class FeatureFrame150(CornerSupportFeatureFrame):
    def __init__(self,rgb,depth):
        _construct(self,rgb,depth)
        self.descriptors_evaluated=len(self.keypoints)
        # The original constructor caps each cell, not the total. Its selection
        # remains strength-ordered; trim the final population after describing
        # at most 156 candidates, preserving descriptor values exactly.
        self.keypoints=self.keypoints[:150]
        if self.descriptors is not None:self.descriptors=self.descriptors[:150].copy()
        self.cell_counts=[0]*12
        for keypoint in self.keypoints:self.cell_counts[cell_index(*keypoint.pt)]+=1

    def witness(self):
        return super().witness()|dict(feature_selection='corner_upright_sift_support_budget150_v1',
            maximum_features=150,maximum_features_per_cell=13,
            descriptors_evaluated_before_total_cap=self.descriptors_evaluated)


class _BudgetDual(DualCameraAnchorPose):
    observe=use_batch(DualCameraAnchorPose.observe,CornerSupportFeatureFrame=FeatureFrame150)


class FeatureBudget150Pose(BatchedPatchPose,_BudgetDual):pass


class FeatureBudget150VisualMotion(BatchedPatchVisualMotion):
    def __init__(self,*,identity=(0,0,0)):
        super().__init__(identity=identity);self.model=FeatureBudget150Pose()

    def snapshot(self,*,now_ns):
        return super().snapshot(now_ns=now_ns)|dict(corner_budget_per_camera=150,
            corner_budget_per_cell=13,geometric_and_temporal_gate_values_unchanged=True,
            original_600_corner_pose_equivalence_claimed=False)


def initialize_pose_150():
    import cv2
    import torch
    from lewm import process_mapped_runtime_development as process
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    process._motion=FeatureBudget150VisualMotion()
