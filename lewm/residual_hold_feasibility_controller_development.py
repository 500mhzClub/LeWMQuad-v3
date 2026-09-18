"""Separate candidate retaining original no-action recovery and observed state."""
from lewm.residual_first_interval_controller_development import ResidualFirstIntervalController, ResidualFirstIntervalSelector
from lewm.residual_hold_feasibility_development import reconsider_hold_feasibility


class ResidualHoldFeasibilitySelector(ResidualFirstIntervalSelector):
    def choose(self,model,history,mapper,geometry,*,now_ns):
        original=super().choose(model,history,mapper,geometry,now_ns=now_ns)
        return reconsider_hold_feasibility(original,self.residual.snapshot(),mapper,geometry,now_ns=now_ns)


class ResidualHoldFeasibilityController(ResidualFirstIntervalController):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.selector=ResidualHoldFeasibilitySelector(residual=self.residual,
            condition=self.selector.condition,variant=self.selector.variant,
            goal_initial_body_xy_m=self.mission.target())

    def _result(self,*args,**kwargs):
        return super()._result(*args,**kwargs)|dict(controller='residual_hold_feasibility_controller_v1',
            residual_hold_feasibility_enabled=True)
