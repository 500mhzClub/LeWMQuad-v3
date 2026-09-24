"""Same eight-step planner, with training-only translated forecast heads."""
from lewm.training_bias_waypoint_selection_development import TrainingBiasWaypointSelector
from lewm.observation_horizon_waypoint_selection_development import restrict
from lewm.observation_horizon_waypoint_utility_development import score_commitment_pose
from lewm.observation_horizon_nominal_constraint_development import constrain
from lewm.eight_step_planning_development import plan
from lewm.eight_step_planning_goal_probe_development import EightStepPlanningGoalProbe
from lewm.geometry_progress_pilot_development import ACTIONS


class TrainingBiasEightStepSelector(TrainingBiasWaypointSelector):
    def choose(self,model,history,mapper,geometry,*,now_ns):
        result=super().choose(model,history,mapper,geometry,now_ns=now_ns)
        if 'prediction' not in result:return result
        if result['mode']=='WAYPOINT':
            result=restrict(score_commitment_pose(result),ACTIONS)
            result['surface_conflict_filter_still_required']=False
            result['original_surface_conflicts_preserved']=True
        B=mapper.map_from_initial;p=B@mapper.surface.position;R=B@mapper.surface.rotation
        result=constrain(result,p,R,mapper.occupied)
        return plan(result,p,R,mapper.occupied)|dict(model_prediction_corrected=True,
            translation_bias_training_only=True)


class TrainingBiasGoalProbe(EightStepPlanningGoalProbe):
    def __init__(self,model,geometry,*,condition,variant,persistent):
        super().__init__(model,geometry,condition=condition,variant=variant,persistent=persistent)
        self.selector=TrainingBiasEightStepSelector(condition=condition,variant=variant)

    def _result(self,command,selection,distance):
        return super()._result(command,selection,distance)|dict(controller='training_bias_goal_probe_v1',
            training_translation_bias_enabled=True)
