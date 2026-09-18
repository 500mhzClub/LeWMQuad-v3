"""View acquisition and observed-floor waypoint selection with learned outcomes."""
from copy import deepcopy
import math
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.geometry_progress_predictive_selection_development import select
from lewm.surface_memory_candidate_filter_development import filter_selection
from lewm.observed_floor_waypoint_development import centre


def wrap(value):
    return (value+math.pi)%(2*math.pi)-math.pi


def scan_rank(selection, delta):
    """Fixed half-second heading improvement, drift and learned contact cost."""
    if not math.isfinite(delta) or abs(delta) > math.pi: raise ValueError('wrapped scan heading required')
    result = deepcopy(selection); prediction = np.asarray(result['prediction'], float)
    if prediction.shape != (6,8,5) or not np.isfinite(prediction).all(): raise ValueError('complete predictions required')
    result['original_candidates'] = deepcopy(result['candidates'])
    for i, row in enumerate(result['candidates']):
        dx,dy,sy,cy,logit = prediction[i,0]
        if np.hypot(sy,cy) <= 1e-8: raise ValueError('defined predicted scan yaw required')
        yaw = math.atan2(sy,cy); improvement = abs(delta)-abs(wrap(delta-yaw))
        contact = float(np.exp(-np.logaddexp(0.,-logit)))
        row.update(utility_m=.4*improvement-float(np.hypot(dx,dy))-1.2*contact,
            heading_improvement_rad=improvement, scan_contact_score=contact)
    result['score_contract']='half_second_heading_improvement_minus_drift_and_contact'
    result['scan_heading_delta_rad']=delta
    result['original_action']=result['action']
    chosen=max((0,4,5),key=lambda i:result['candidates'][i]['utility_m'])
    result.update(action=ACTIONS[chosen],action_index=chosen,requested_command=candidate_commands(ACTIONS[chosen])[0])
    return result


def restrict(selection, allowed):
    result = deepcopy(selection)
    feasible = [i for i,check in enumerate(result['surface_checks'])
        if ACTIONS[i] in allowed and not check['possible_intersection']]
    chosen = max(feasible,key=lambda i:result['candidates'][i]['utility_m']) if feasible else None
    result.update(action=None if chosen is None else ACTIONS[chosen], action_index=chosen,
        requested_command=[0.,0.,0.] if chosen is None else candidate_commands(ACTIONS[chosen])[0],
        phase_allowed_actions=list(allowed),phase_admissible_candidates=len(feasible))
    return result


class ActiveViewWaypointSelector:
    def __init__(self):
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
            goal_body=(R.T@np.r_[target-p[:2],0.])[:2]
            selected=select(model,history,head='rollout_outcomes',input_variant='full',
                goal_body_xy_m=goal_body,contact_penalty_m=1.2)
            allowed=ACTIONS
            selected['waypoint_map_xy_m']=target.tolist()
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
            selected=select(model,history,head='rollout_outcomes',input_variant='full',
                goal_body_xy_m=[0.,0.],contact_penalty_m=1.2)
            selected=scan_rank(selected,wrap(self.scan_target-heading))
            allowed=('hold','left_turn','right_turn')
            selected.update(scan_target_map_yaw_rad=self.scan_target,scan_index=self.scan_index,
                scan_sign=self.scan_sign,measured_heading_map_rad=heading)
        selected.pop('selection_wall_ms')
        selected=filter_selection(selected,mapper.surface,geometry,now_ns=now_ns,persistent=True)
        selected=restrict(selected,allowed)
        return selected|dict(proposal=proposal,mode=self.mode,view_budget_exhausted=False,
            intermediate_target_is_mission_goal=False,exploratory_unknown_connector=bool(proposal.get('unknown_connector_cells')))
