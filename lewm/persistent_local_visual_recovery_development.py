"""Keep a nearby observed view as a recovery objective beyond ten seconds."""
from lewm.framewise_visual_support_recovery_development import FramewiseVisualSupportRuntime
from lewm.visual_support_recovery_development import LocalSupportedView


class PersistentLocalVisualRuntime(FramewiseVisualSupportRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.registration.views = LocalSupportedView(maximum_view_age_ns=None)
