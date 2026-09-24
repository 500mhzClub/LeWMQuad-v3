"""Leave an interrupted startup sweep incomplete and use observed-floor routing."""
from lewm.initial_panorama_development import InitialSurveyMixin


class RecoveryLimitedInitialSurveyMixin:
    """Insert immediately before InitialSurveyMixin, beneath recovery routing."""

    def __init__(self, *args, **kwargs):
        self.initial_survey_deferral = None
        super().__init__(*args, **kwargs)

    def _route(self, snapshot, evidence, goal, *, measured_ns):
        receipt = evidence.get('visual_support')
        active = None if receipt is None else receipt.get('recovery_state_at_observation')
        if (self.initial_survey_deferral is None and self.initial_panorama.state is not None
                and not self.initial_panorama.complete
                and active is not None):
            if (receipt.get('camera_cadence_recovery') is not True
                    or receipt['measured_ns'] != measured_ns
                    or active['trigger_ns'] > measured_ns):
                raise ValueError('same-observation recovery required to defer startup sweep')
            self.initial_survey_deferral = dict(frame=receipt['frame'],
                measured_ns=measured_ns, recovery_trigger_ns=active['trigger_ns'],
                reason='WEAK_VIEW_INTERRUPTED_INITIAL_SURVEY',
                outstanding_views_remain_unobserved=True,
                observed_floor_routing_enabled_after_recovery=True)
            state = self.initial_panorama.state
            self.initial_panorama.state = state | dict(
                deferred=True, deferral=self.initial_survey_deferral)
        if self.initial_survey_deferral is None:
            return super()._route(snapshot, evidence, goal, measured_ns=measured_ns)
        # This mixin sits at the startup-survey layer. Only that layer is
        # bypassed; outer visual recovery, deadline, forecast and dispatch
        # methods remain in the runtime's normal call chain.
        route = super(InitialSurveyMixin, self)._route(
            snapshot, evidence, goal, measured_ns=measured_ns)
        return route | dict(initial_survey=dict(self.initial_panorama.state))
