"""Separate prospective first-interval feasibility policy; frozen baseline intact."""
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportController
from lewm.view_reentry_round_trip_controller_development import ViewReentrySelector
from lewm.residual_first_interval_feasibility_development import correct_first_interval_feasibility


class ResidualFirstIntervalSelector(ViewReentrySelector):
    def choose(self,model,history,mapper,geometry,*,now_ns):
        original=super().choose(model,history,mapper,geometry,now_ns=now_ns)
        return correct_first_interval_feasibility(original,self.residual.snapshot(),mapper,geometry,now_ns=now_ns)


class ResidualFirstIntervalController(MeasuredFloorTransportController):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.selector=ResidualFirstIntervalSelector(residual=self.residual,condition=self.selector.condition,
            variant=self.selector.variant,goal_initial_body_xy_m=self.mission.target())

    def _result(self,*args,**kwargs):
        return super()._result(*args,**kwargs)|dict(controller='residual_first_interval_feasibility_controller_v1',
            residual_first_interval_feasibility_fallback_enabled=True)
