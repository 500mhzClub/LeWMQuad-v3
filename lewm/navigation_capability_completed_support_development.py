"""Use the unchanged tracker's actual selected corners for recovery counts.

Counts remain an uncalibrated recovery heuristic. Pose estimation/admission,
feature extraction, sensors and physical-clearance rules are not changed.
"""
from lewm.eligible_floor_registration_development import bind
from lewm.visual_support_recovery_development import SupportRegistration
from lewm.framewise_visual_support_recovery_development import FramewiseSupportRegistration
from lewm.sparse_corner_completion_runtime_development import strong_corner_support, StrongCornerFramewiseRegistration
from lewm.visual_recovery_dispatch_hold_development import PublishingSupportRegistration
from lewm.navigation_capability_live_turn_binding_c1_development import LiveTurnMemoryRuntimeMixin


def completed_corner_support(raw):
    receipt = strong_corner_support(raw)
    if receipt is None:
        return None
    return receipt | dict(original_strong_selected_features=receipt['selected_features'],
        selected_features=receipt['tracking_selected_features'],
        recovery_uses_original_strong_corner_counts=False,
        recovery_count_source='unchanged_tracker_selected_completed_features')


class CompletedSupportRegistration(SupportRegistration):
    observe = bind(SupportRegistration.observe, support=completed_corner_support)


class CompletedFramewiseRegistration(FramewiseSupportRegistration, CompletedSupportRegistration):
    pass


class CompletedSupportRuntimeMixin(LiveTurnMemoryRuntimeMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.registration, PublishingSupportRegistration)
        previous = self.registration.original
        assert type(previous) is StrongCornerFramewiseRegistration
        replacement = CompletedFramewiseRegistration(previous.original)
        replacement.views = previous.views  # Retain V2 reference exhaustion.
        self.registration.original = replacement
