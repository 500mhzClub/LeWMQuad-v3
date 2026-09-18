"""Single-threshold development intervention for earlier visual recovery."""
from lewm.eligible_floor_registration_development import bind
from lewm.visual_support_recovery_development import LocalSupportedView
from lewm.framewise_visual_support_recovery_development import FramewiseSupportRegistration
from lewm.visual_recovery_dispatch_hold_development import PublishingSupportRegistration

EARLIER_LOW_FEATURES = 72


class EarlierSupportedView(LocalSupportedView):
    # This changes both weak-view onset and the aligned-view release floor.
    # Strong-reference acceptance remains 96; pose fitting is untouched.
    advance = bind(LocalSupportedView.advance, LOW_FEATURES=EARLIER_LOW_FEATURES)


class EarlierVisualRecoveryMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.registration, PublishingSupportRegistration)
        registration = self.registration.original
        assert isinstance(registration, FramewiseSupportRegistration)
        assert registration.views.good is None and registration.views.active is None
        registration.views = EarlierSupportedView(maximum_view_age_ns=None)
