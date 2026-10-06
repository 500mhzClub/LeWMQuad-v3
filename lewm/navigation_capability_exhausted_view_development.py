"""Retire an attained view reference that does not restore strong support.

Recovery behaviour only. Ten-Hz accepted visual poses must keep the requested
heading within the existing 0.1-rad tolerance for a full second. This is not
tracking confidence or permission to bypass any movement/dispatch guard.
"""
import math
from lewm.visual_support_recovery_development import LocalSupportedView
from lewm.navigation_capability_paired_floor_start_development import PairedFloorRuntimeMixin
from lewm.sparse_corner_completion_runtime_development import StrongCornerFramewiseRegistration
from lewm.visual_recovery_dispatch_hold_development import PublishingSupportRegistration

ATTAINED_VIEW_DWELL_NS = 1_000_000_000
MAX_OBSERVATION_GAP_NS = 100_000_000


class ExhaustibleSupportedView(LocalSupportedView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.aligned_since_ns = None
        self.aligned_trigger_ns = None
        self.last_observation_ns = None
        self.retirements = []

    def advance(self, counts, position, rotation, now_ns, generation):
        reset = (generation != self.generation or self.last_observation_ns is None
            or not 0 < now_ns-self.last_observation_ns <= MAX_OBSERVATION_GAP_NS)
        if reset:
            self.aligned_since_ns = self.aligned_trigger_ns = None
        self.last_observation_ns = now_ns
        active = super().advance(counts, position, rotation, now_ns, generation)
        if active is None:
            self.aligned_since_ns = self.aligned_trigger_ns = None
            return None
        target = active['rotation']
        difference = math.atan2(target[1, 0], target[0, 0])-math.atan2(rotation[1, 0], rotation[0, 0])
        error = math.atan2(math.sin(difference), math.cos(difference))
        if abs(error) > .1:
            self.aligned_since_ns = self.aligned_trigger_ns = None
            return active
        if self.aligned_trigger_ns != active['trigger_ns']:
            self.aligned_since_ns = now_ns
            self.aligned_trigger_ns = active['trigger_ns']
        if now_ns-self.aligned_since_ns < ATTAINED_VIEW_DWELL_NS:
            return active
        self.retirements.append(dict(measured_ns=now_ns,trigger_ns=active['trigger_ns'],
            reference_measured_ns=active['measured_ns'],aligned_since_ns=self.aligned_since_ns,
            heading_error_rad=error,selected_features=list(counts),
            reason='ATTAINED_VIEW_DID_NOT_RESTORE_STRONG_SUPPORT',
            pose_admission_unchanged=True,clearance_and_dispatch_unchanged=True))
        # Consume the old reference; a later genuinely strong observation can
        # establish a new one through the inherited rule.
        self.active = self.good = None
        self.aligned_since_ns = self.aligned_trigger_ns = None
        return None


class ExhaustedViewRuntimeMixin(PairedFloorRuntimeMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.registration, PublishingSupportRegistration)
        registration = self.registration.original
        assert isinstance(registration, StrongCornerFramewiseRegistration)
        previous = registration.views
        assert type(previous) is LocalSupportedView
        assert previous.active is None and previous.good is None
        registration.views = ExhaustibleSupportedView(maximum_view_age_ns=previous.maximum_view_age_ns)
        self.exhausted_view_retirements = registration.views.retirements
