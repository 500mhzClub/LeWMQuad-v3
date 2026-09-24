"""Combine measured 3-D support conditioning with the 150-feature budget."""
from lewm.conditioned_support_tracker_development import ConditionedSupportPose,_Dual,use
from lewm.feature_budget_150_tracker_development import FeatureFrame150,FeatureBudget150VisualMotion


class _BudgetDual(_Dual):
    observe=use(_Dual.observe,CornerSupportFeatureFrame=FeatureFrame150)


class ConditionedSupport150Pose(ConditionedSupportPose,_BudgetDual):pass


class ConditionedSupport150VisualMotion(FeatureBudget150VisualMotion):
    def __init__(self,*,identity=(0,0,0)):
        super().__init__(identity=identity);self.model=ConditionedSupport150Pose()

    def snapshot(self,*,now_ns):
        return super().snapshot(now_ns=now_ns)|dict(image_bin_count_gate_used=False,
            strict_inlier_majority_required=True,original_measured_3d_conditioning_required=True,
            spatial_and_inlier_fraction_thresholds_changed=True,
            geometric_and_temporal_gate_values_unchanged=False,
            remaining_reprojection_conditioning_motion_and_temporal_thresholds_unchanged=True,
            accepted_support_promotion_threshold_cells=7)


def initialize_conditioned_support_pose_150():
    import cv2
    import torch
    from lewm import process_mapped_runtime_development as process
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    process._motion=ConditionedSupport150VisualMotion()
