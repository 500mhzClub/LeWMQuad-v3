"""Current-waypoint feedback with the continuous arm's perception and memory."""
from lewm.initial_panorama_development import InitialSurveyMixin
from lewm.clearance_lookahead_development import StandOffFrontierRuntime
from lewm.fine_stored_obstacle_routing_development import cached_clearance
from lewm.continuous_reactive_selection_development import select_reactive


class ContinuousReactiveRuntime(InitialSurveyMixin,StandOffFrontierRuntime):
    frontier_panorama=True

    def __init__(self,model,**kwargs):
        if model is not None or kwargs.get('condition')!='reactive':
            raise ValueError('explicit model-free reactive treatment required')
        super().__init__(model,**kwargs)

    def _select_action(self,packet,evidence,prefix,goal_body,scan_error,snapshot,q,Q):
        clearance=cached_clearance(snapshot.fine_occupied).minimum(q[:2],q[:2])
        selected=select_reactive(goal_body,scan_error=scan_error,current_clearance_m=clearance)
        return selected,None

    def _select_clear_prediction(self,*args,**kwargs):
        raise RuntimeError('reactive action selection cannot consume predicted outcomes')
