"""A rejected translation requests a measured new view before retrying its route."""
import math
import numpy as np
from lewm.stopping_margin_dispatch_development import StoppingMarginRoundTripRuntime
from lewm.extended_return_budget_transport_development import current_measured_floor_pose


def wrapped(angle):return math.atan2(math.sin(angle),math.cos(angle))


class VetoViewRoundTripRuntime(StoppingMarginRoundTripRuntime):
    def __init__(self,*args,**kwargs):
        self.view_recovery=None
        super().__init__(*args,**kwargs)

    def request(self,*,now_ns):
        result=super().request(now_ns=now_ns)
        with self.lock:
            if result['reason'] in ('CURRENT_STOPPING_MARGIN_VETO','CURRENT_OBSERVED_OBSTACLE_VETO'):
                plans=[p for p in self.plans if p.observed_ns==result.get('command_observation_ns')]
                if plans and any(plans[0].command[:2]) and self.view_recovery is None:
                    self.view_recovery=self._translation_veto_recovery(now_ns,plans[0])
            return result|dict(view_recovery=None if self.view_recovery is None else dict(self.view_recovery))

    def _translation_veto_recovery(self,now_ns,plan):
        return dict(trigger_ns=now_ns,mission_generation=self.mission_generation,
            target_heading_rad=None,source='actual_translation_veto')

    def _route(self,snapshot,evidence,goal,*,measured_ns):
        result=super()._route(snapshot,evidence,goal,measured_ns=measured_ns)
        with self.lock:
            recovery=self.view_recovery
            if recovery is None:return result
            if recovery['mission_generation']!=self.mission_generation:
                self.view_recovery=None;return result
        _,R,_=self._pose(evidence,identity=(0,0,0),now_ns=measured_ns)
        Q=np.asarray(snapshot.map_from_initial)@R
        heading=math.atan2(Q[1,0],Q[0,0])
        with self.lock:
            if measured_ns<recovery['trigger_ns']:
                # No recovery completion may be inferred from a pre-veto pose.
                return result|dict(route_cells=[],status='WAITING_FOR_POST_VETO_VIEW')
            if recovery['target_heading_rad'] is None:
                recovery['target_heading_rad']=wrapped(heading+recovery.get('view_angle_rad',math.pi/4))
            target=recovery['target_heading_rad']
            if abs(wrapped(target-heading))<=.1:
                self.view_recovery=None
                return result
            self.scan_target=target
        return result|dict(route_cells=[],status='TRANSLATION_VETO_REQUIRES_NEW_VIEW')
