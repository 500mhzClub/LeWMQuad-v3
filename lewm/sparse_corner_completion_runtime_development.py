"""Use completed corner tracks while retaining strong-corner recovery counts."""
from lewm.cadenced_view_revisit_tracking_development import CadencedViewRevisitMotion
from lewm.sparse_corner_completion_development import SparseCornerCompletionPose
from lewm.visual_support_recovery_development import SupportRegistration, support
from lewm.framewise_visual_support_recovery_development import FramewiseSupportRegistration
from lewm.visual_recovery_dispatch_hold_development import PublishingSupportRegistration
from lewm.eligible_floor_registration_development import bind


class SparseCornerCompletionMotion(CadencedViewRevisitMotion):
    def __init__(self, *, identity=(0, 0, 0), activation_frame=0):
        super().__init__(identity=identity, activation_frame=activation_frame)
        self.model = SparseCornerCompletionPose(activation_frame=activation_frame)


def strong_corner_support(raw):
    receipt = support(raw)
    if receipt is None:
        return None
    witnesses = [raw[k] for k in
        ('last_accepted_feature_witness', 'auxiliary_feature_witness')]
    return receipt | dict(tracking_selected_features=receipt['selected_features'],
        selected_features=[w['original_selected_count'] for w in witnesses],
        recovery_uses_original_strong_corner_counts=True)


class _StrongCornerSupportRegistration(SupportRegistration):
    observe = bind(SupportRegistration.observe, support=strong_corner_support)


class StrongCornerFramewiseRegistration(FramewiseSupportRegistration,
        _StrongCornerSupportRegistration):
    pass


class SparseCornerCompletionRuntimeMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.registration, PublishingSupportRegistration)
        previous = self.registration.original
        assert isinstance(previous, FramewiseSupportRegistration)
        replacement = StrongCornerFramewiseRegistration(previous.original)
        replacement.views = previous.views
        self.registration.original = replacement
