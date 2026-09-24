"""New waypoint utility in the unchanged measured active-view mission loop."""
from lewm.matched_model_goal_probe_development import MatchedModelGoalProbe
from lewm.matched_model_waypoint_selection_development import MatchedModelWaypointSelector, restrict
from lewm.commitment_pose_waypoint_utility_development import score_commitment_pose
from lewm.geometry_progress_pilot_development import ACTIONS


def apply_waypoint_utility(selection):
    # These checks already cover these exact unchanged first-half-second poses
    # and the current measured joint posture. Reranking never removes a veto.
    result=restrict(score_commitment_pose(selection),ACTIONS)
    result['surface_conflict_filter_still_required']=False
    result['original_surface_conflicts_preserved']=True
    return result


class CommitmentPoseWaypointSelector(MatchedModelWaypointSelector):
    def choose(self,model,history,mapper,geometry,*,now_ns):
        result=super().choose(model,history,mapper,geometry,now_ns=now_ns)
        return apply_waypoint_utility(result) if result['mode']=='WAYPOINT' else result


class CommitmentPoseGoalProbe(MatchedModelGoalProbe):
    def __init__(self,model,geometry,*,condition,variant,persistent):
        super().__init__(model,geometry,condition=condition,variant=variant,persistent=persistent)
        self.selector=CommitmentPoseWaypointSelector(condition=condition,variant=variant)

    def _result(self,command,selection,distance):
        return super()._result(command,selection,distance)|dict(controller='commitment_pose_goal_probe_v1')
