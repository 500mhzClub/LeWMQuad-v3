"""100 spatially distributed corners per camera with unchanged pose gates."""
from lewm.corner_support_features_development import CornerSupportFeatureFrame,CORNER_RULES
from lewm.eligible_floor_registration_development import bind
from lewm.dual_camera_anchor_pose_development import DualCameraAnchorPose
from lewm.batched_patch_tracker_development import BatchedPatchPose,BatchedPatchVisualMotion,use_batch
from lewm.spatial_support_features_development import cell_index

RULES=CORNER_RULES|dict(per_cell=9,maximum_selected=108)
_construct=bind(CornerSupportFeatureFrame.__init__,CORNER_RULES=RULES)


class FeatureFrame100(CornerSupportFeatureFrame):
    def __init__(self,rgb,depth):
        _construct(self,rgb,depth)
        self.descriptors_evaluated=len(self.keypoints)
        self.keypoints=self.keypoints[:100]
        if self.descriptors is not None:self.descriptors=self.descriptors[:100].copy()
        self.cell_counts=[0]*12
        for keypoint in self.keypoints:self.cell_counts[cell_index(*keypoint.pt)]+=1

    def witness(self):
        return super().witness()|dict(feature_selection='corner_upright_sift_support_budget100_v1',
            maximum_features=100,maximum_features_per_cell=9,
            descriptors_evaluated_before_total_cap=self.descriptors_evaluated)


class _BudgetDual(DualCameraAnchorPose):
    observe=use_batch(DualCameraAnchorPose.observe,CornerSupportFeatureFrame=FeatureFrame100)


class FeatureBudget100Pose(BatchedPatchPose,_BudgetDual):pass


class FeatureBudget100VisualMotion(BatchedPatchVisualMotion):
    def __init__(self,*,identity=(0,0,0)):
        super().__init__(identity=identity);self.model=FeatureBudget100Pose()

    def snapshot(self,*,now_ns):
        return super().snapshot(now_ns=now_ns)|dict(corner_budget_per_camera=100,
            corner_budget_per_cell=9,geometric_and_temporal_gate_values_unchanged=True,
            original_600_corner_pose_equivalence_claimed=False)


def initialize_pose_100():
    import cv2
    import torch
    from lewm import process_mapped_runtime_development as process
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    process._motion=FeatureBudget100VisualMotion()
