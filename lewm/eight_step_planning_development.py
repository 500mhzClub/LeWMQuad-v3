"""Use all trained forecast segments without lengthening actual commitment."""
from copy import deepcopy
import math
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS,candidate_commands
from lewm.joint_rgbd_rigid_pose_development import proper
from lewm.observation_horizon_predictive_selection_development import require_short_forecast
from lewm.observation_horizon_waypoint_utility_development import potential,CONTACT_PENALTY_M
from lewm.observed_geometry_refinement_development import nominal_connector


def plan(selection,position_map,rotation_map_from_body,occupied):
    require_short_forecast(selection)
    result=deepcopy(selection);prediction=np.asarray(result['prediction'],float)
    p=np.asarray(position_map,float);R=proper(rotation_map_from_body)
    if (p.shape!=(3,) or not np.isfinite(p).all() or prediction.shape!=(6,8,5)
            or not np.isfinite(prediction).all() or (np.diff(prediction[:,:,4],axis=1)<-1e-6).any()
            or np.any(np.hypot(prediction[:,:,2],prediction[:,:,3])<=1e-8)
            or [c['action'] for c in result['candidates']]!=list(ACTIONS)
            or len(result['surface_checks'])!=6 or len(result['nominal_action_checks'])!=6
            or result['mode'] not in ('WAYPOINT','VIEW_ACQUISITION')):
        raise ValueError('complete ordered forecast bank, first-step checks and observed map pose required')
    allowed=result['phase_allowed_actions']
    if not isinstance(allowed,(list,tuple)) or not allowed or any(a not in ACTIONS for a in allowed):
        raise ValueError('explicit original phase action allowance required')
    result['first_step_candidates']=deepcopy(result['candidates'])
    result['first_step_action']=result['action']
    if result['mode']=='WAYPOINT':
        goal=np.asarray(result['goal_body_xy_m'],float);initial,distance,alignment=potential(goal,0.)
        for i,row in enumerate(result['candidates']):
            dx,dy,sy,cy,logit=prediction[i,-1]
            final,end_distance,end_alignment=potential(goal-[dx,dy],math.atan2(sy,cy))
            contact=float(np.exp(-np.logaddexp(0.,-logit)))
            row.update(utility_m=initial-final-CONTACT_PENALTY_M*contact,
                planning_distance_progress_m=distance-end_distance,
                planning_alignment_progress_m=alignment-end_alignment,planning_contact_score=contact)
        result.update(scored_horizon_ns=800_000_000,
            score_contract='eight_step_waypoint_distance_and_bearing_potential_minus_contact')
    utilities=np.asarray([r['utility_m'] for r in result['candidates']],float)
    if utilities.shape!=(6,) or not np.isfinite(utilities).all():raise ValueError('finite planning utilities required')
    cells=sorted(occupied);checks=[]
    for i in range(6):
        points=[p[:2]]+[(p+R@np.r_[xy,0.])[:2] for xy in prediction[i,:,:2]]
        segments=[dict(start_offset_ns=h*100_000_000,end_offset_ns=(h+1)*100_000_000,
            predicted_start_map_xy_m=points[h].tolist(),predicted_end_map_xy_m=points[h+1].tolist(),
            **nominal_connector(points[h],points[h+1],cells,radius_m=.45)) for h in range(8)]
        original=result['nominal_action_checks'][i]
        if (original['action']!=ACTIONS[i] or original['radius_m']!=.45
                or original['minimum_observed_cell_distance_m']!=segments[0]['minimum_observed_cell_distance_m']
                or original['nominal_disk_connector_clear']!=segments[0]['nominal_disk_connector_clear']):
            raise ValueError('exact original first-segment clearance check required')
        checks.append(dict(action=ACTIONS[i],segments=segments,
            all_predicted_segments_nominally_clear=all(s['nominal_disk_connector_clear'] for s in segments)))
    feasible=[i for i in range(6) if ACTIONS[i] in allowed
        and not result['surface_checks'][i]['possible_intersection']
        and checks[i]['all_predicted_segments_nominally_clear']]
    chosen=max(feasible,key=lambda i:utilities[i]) if feasible else None
    result.update(nominal_path_checks=checks,path_constraint_horizon_ns=800_000_000,
        planned_segment_count=8,phase_admissible_candidates=len(feasible),
        action=None if chosen is None else ACTIONS[chosen],action_index=chosen,
        requested_command=[0.,0.,0.] if chosen is None else candidate_commands(ACTIONS[chosen])[0],
        actual_commitment_horizon_ns=100_000_000,all_current_occupied_squares_checked=True,
        first_step_surface_conflicts_preserved=True,later_articulated_surface_checks_performed=False,
        model_error_bound_applied=False,intermediate_motion_certified=False,
        terminal_viability_certified=False,model_prediction_corrected=False)
    return result
