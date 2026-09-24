"""Keep all measured corners when they already fit within the total budget."""
from lewm.corner_support_features_development import CornerSupportFeatureFrame, CORNER_RULES
from lewm.feature_budget_150_tracker_development import FeatureFrame150
from lewm.conditioned_support_150_tracker_development import _BudgetDual
from lewm.development_support_tracker_development import use
from lewm.eligible_floor_registration_development import bind
from lewm.cadenced_view_revisit_tracking_development import CadencedViewRevisitPose

_sparse_construct = bind(CornerSupportFeatureFrame.__init__,
    CORNER_RULES=CORNER_RULES | dict(per_cell=150, maximum_selected=150))


class SparseFeatureFrame150(FeatureFrame150):
    def __init__(self, rgb, depth):
        super().__init__(rgb, depth)
        self.sparse_budget_used = self.liftable_count <= 150 and len(self.keypoints) < self.liftable_count
        self.original_selected_count = len(self.keypoints)
        if self.sparse_budget_used:
            _sparse_construct(self, rgb, depth)
            self.descriptors_evaluated = len(self.keypoints)

    def witness(self):
        return super().witness() | dict(
            feature_selection='sparse_total_budget150_v1',
            sparse_budget_used=self.sparse_budget_used,
            original_selected_count=self.original_selected_count,
            maximum_features_per_cell=150 if self.sparse_budget_used else 13,
            total_feature_budget_unchanged=True)


class _SparseBudgetDual(_BudgetDual):
    observe = use(_BudgetDual.observe, CornerSupportFeatureFrame=SparseFeatureFrame150)


class SparseFeatureBudgetPose(CadencedViewRevisitPose, _SparseBudgetDual):
    pass
