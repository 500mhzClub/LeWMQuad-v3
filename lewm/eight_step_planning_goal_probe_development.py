"""Full nominal forecast-path planning with one-observation command execution."""
from lewm.eight_step_planning_development import plan
from lewm.observation_horizon_goal_probe_development import ObservationHorizonGoalProbe,ObservationHorizonConstrainedSelector


class EightStepPlanningSelector(ObservationHorizonConstrainedSelector):
    def choose(self,model,history,mapper,geometry,*,now_ns):
        result=super().choose(model,history,mapper,geometry,now_ns=now_ns)
        if 'prediction' not in result:return result
        B=mapper.map_from_initial
        return plan(result,B@mapper.surface.position,B@mapper.surface.rotation,mapper.occupied)


class EightStepPlanningGoalProbe(ObservationHorizonGoalProbe):
    def __init__(self,model,geometry,*,condition,variant,persistent):
        super().__init__(model,geometry,condition=condition,variant=variant,persistent=persistent)
        self.selector=EightStepPlanningSelector(condition=condition,variant=variant)

    def _result(self,command,selection,distance):
        return super()._result(command,selection,distance)|dict(controller='eight_step_planning_goal_probe_v1',
            planning_horizon_ns=800_000_000,actual_commitment_horizon_ns=100_000_000)
