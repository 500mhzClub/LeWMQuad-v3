"""300 spatially distributed corners per camera; original admission thresholds."""
from lewm.corner_support_features_development import CornerSupportFeatureFrame, CORNER_RULES
from lewm.eligible_floor_registration_development import bind
from lewm.dual_camera_anchor_pose_development import DualCameraAnchorPose
from lewm.batched_patch_tracker_development import BatchedPatchPose, BatchedPatchVisualMotion, use_batch

RULES = CORNER_RULES | dict(per_cell=25, maximum_selected=300)


class FeatureFrame300(CornerSupportFeatureFrame):
    __init__ = bind(CornerSupportFeatureFrame.__init__, CORNER_RULES=RULES)

    def witness(self):
        return super().witness() | dict(feature_selection='corner_upright_sift_support_budget300_v1',
            maximum_features=300, maximum_features_per_cell=25)


class _BudgetDual(DualCameraAnchorPose):
    observe = use_batch(DualCameraAnchorPose.observe, CornerSupportFeatureFrame=FeatureFrame300)


class FeatureBudget300Pose(BatchedPatchPose, _BudgetDual):
    pass


class FeatureBudget300VisualMotion(BatchedPatchVisualMotion):
    def __init__(self, *, identity=(0, 0, 0)):
        super().__init__(identity=identity)
        self.model=FeatureBudget300Pose()

    def snapshot(self, *, now_ns):
        return super().snapshot(now_ns=now_ns) | dict(corner_budget_per_camera=300,
            corner_budget_per_cell=25, geometric_and_temporal_gate_values_unchanged=True,
            original_600_corner_pose_equivalence_claimed=False)
