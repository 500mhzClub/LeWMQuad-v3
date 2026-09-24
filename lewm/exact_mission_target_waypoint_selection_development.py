"""Observed final-goal connector, with original intermediate route/view logic."""
import math
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.exact_mission_target_development import terminal_target
from lewm.observed_floor_waypoint_development import centre
from lewm.observation_horizon_waypoint_selection_development import wrap,scan_rank,restrict
from lewm.observation_horizon_surface_filter_development import filter_selection
from lewm.training_bias_predictive_selection_development import select

class ExactMissionTargetWaypointSelector:
    def __init__(self, *, condition, variant):
        if condition not in ('direct', 'supervised_rollout', 'jepa') or variant not in ('full', 'no_rgb'):
            raise ValueError('explicit trained model condition and input variant required')
        self.condition = condition
        self.variant = variant
        self.head = 'direct_outcomes' if condition == 'direct' else 'rollout_outcomes'
        self.scan_sign = None
        self.scan_index = 0
        self.scan_target = None
        self.mode = 'NEW'

    def choose(self, model, history, mapper, geometry, *, now_ns):
        proposal = mapper.waypoint([1.2,0.],now_ns=now_ns)
        B=mapper.map_from_initial; R=B@mapper.surface.rotation; p=B@mapper.surface.position
        heading=math.atan2(R[1,0],R[0,0])
        if proposal['route_cells']:
            self.mode='WAYPOINT'
            points=[centre(c) for c in proposal['route_cells']]
            target=next((q for q in points if np.linalg.norm(q-p[:2])>=.35),points[-1])
            target,target_evidence=terminal_target(proposal,target,p[:2],
                (B@np.array([1.2,0.,0.]))[:2],mapper.floor,mapper.occupied)
            goal_body=(R.T@np.r_[target-p[:2],0.])[:2]
            selected=select(model,history,head=self.head,input_variant=self.variant,
                goal_body_xy_m=goal_body,contact_penalty_m=1.2)
            allowed=ACTIONS
            selected['waypoint_map_xy_m']=target.tolist()
            selected['terminal_goal_target_evidence']=target_evidence
        else:
            self.mode='VIEW_ACQUISITION'
            if self.scan_sign is None:
                points=np.array([centre(c) for c in sorted(mapper.floor)])
                local=(np.column_stack((points,np.full(len(points),p[2])))-p)@R if len(points) else np.empty((0,3))
                local=local[np.linalg.norm(local[:,:2],axis=1)<=2.]
                self.scan_sign=1 if int((local[:,1]>0).sum())>=int((local[:,1]<0).sum()) else -1
            offsets=[self.scan_sign*math.pi/4,self.scan_sign*math.pi/2,
                -self.scan_sign*math.pi/4,-self.scan_sign*math.pi/2,math.pi]
            if self.scan_target is None: self.scan_target=offsets[self.scan_index]
            if abs(wrap(self.scan_target-heading))<=.1:
                self.scan_index+=1
                if self.scan_index==len(offsets):
                    return dict(view_budget_exhausted=True,action=None,proposal=proposal,mode=self.mode)
                self.scan_target=offsets[self.scan_index]
            selected=select(model,history,head=self.head,input_variant=self.variant,
                goal_body_xy_m=[0.,0.],contact_penalty_m=1.2)
            selected=scan_rank(selected,wrap(self.scan_target-heading))
            allowed=('hold','left_turn','right_turn')
            selected.update(scan_target_map_yaw_rad=self.scan_target,scan_index=self.scan_index,
                scan_sign=self.scan_sign,measured_heading_map_rad=heading)
        selected.pop('selection_wall_ms')
        selected=filter_selection(selected,mapper.surface,geometry,now_ns=now_ns,persistent=True)
        selected=restrict(selected,allowed)
        return selected|dict(proposal=proposal,mode=self.mode,view_budget_exhausted=False,
            intermediate_target_is_mission_goal=bool(selected.get('terminal_goal_target_evidence',{}).get('selected',False)),exploratory_unknown_connector=bool(proposal.get('unknown_connector_cells')))
