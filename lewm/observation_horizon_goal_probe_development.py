"""Replan each actual observation using its trained 100-ms first forecast."""
from lewm.observation_replan_goal_probe_development import ObservationReplanGoalProbe
from lewm.observation_horizon_waypoint_selection_development import ObservationHorizonWaypointSelector,restrict
from lewm.observation_horizon_waypoint_utility_development import score_commitment_pose
from lewm.observation_horizon_nominal_constraint_development import constrain
from lewm.geometry_progress_pilot_development import ACTIONS


class ObservationHorizonConstrainedSelector(ObservationHorizonWaypointSelector):
    def choose(self,model,history,mapper,geometry,*,now_ns):
        result=super().choose(model,history,mapper,geometry,now_ns=now_ns)
        if 'prediction' not in result:return result
        if result['mode']=='WAYPOINT':
            result=restrict(score_commitment_pose(result),ACTIONS)
            result['surface_conflict_filter_still_required']=False
            result['original_surface_conflicts_preserved']=True
        B=mapper.map_from_initial
        return constrain(result,B@mapper.surface.position,B@mapper.surface.rotation,mapper.occupied)


class ObservationHorizonGoalProbe(ObservationReplanGoalProbe):
    def __init__(self,model,geometry,*,condition,variant,persistent):
        super().__init__(model,geometry,condition=condition,variant=variant,persistent=persistent)
        self.selector=ObservationHorizonConstrainedSelector(condition=condition,variant=variant)

    def _result(self,command,selection,distance):
        return super()._result(command,selection,distance)|dict(controller='observation_horizon_goal_probe_v1',
            model_forecast_horizon_ns=100_000_000,maximum_model_horizon_ns=800_000_000,
            planned_command_interval_matches_first_prediction_horizon=True)
