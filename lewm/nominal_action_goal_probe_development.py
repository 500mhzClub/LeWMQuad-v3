"""Prospective nominal forecast feasibility in the unchanged mission and map."""
from lewm.continuous_connector_goal_probe_development import ContinuousConnectorGoalProbe
from lewm.commitment_pose_goal_probe_development import CommitmentPoseWaypointSelector
from lewm.nominal_action_constraint_development import constrain


class NominalActionWaypointSelector(CommitmentPoseWaypointSelector):
    def choose(self,model,history,mapper,geometry,*,now_ns):
        selected=super().choose(model,history,mapper,geometry,now_ns=now_ns)
        if 'prediction' not in selected:return selected
        B=mapper.map_from_initial
        return constrain(selected,B@mapper.surface.position,B@mapper.surface.rotation,mapper.occupied)


class NominalActionGoalProbe(ContinuousConnectorGoalProbe):
    def __init__(self,model,geometry,*,condition,variant,persistent):
        super().__init__(model,geometry,condition=condition,variant=variant,persistent=persistent)
        self.selector=NominalActionWaypointSelector(condition=condition,variant=variant)

    def _result(self,command,selection,distance):
        if self.terminal=='NO_PHASE_CANDIDATE_WITHOUT_SURFACE_INTERSECTION':
            self.terminal='NO_PHASE_CANDIDATE_SATISFYING_SURFACE_AND_NOMINAL_CONSTRAINTS'
        return super()._result(command,selection,distance)|dict(controller='nominal_action_goal_probe_v1')
