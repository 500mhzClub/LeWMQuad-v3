"""Development waypoint value over the actually committed 100-ms horizon.

This component grants no native execution or clearance authority. A caller must
admit model predictions and retain measured-surface conflict filtering. Value
includes turning toward the target, whose benefit displacement-only scoring
cannot represent for an in-place turn. Prediction errors remain uncorrected.
"""
from lewm.observation_horizon_predictive_selection_development import require_short_forecast
from copy import deepcopy
import math
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands

TURN_LENGTH_M = .4
ALIGNMENT_FADE_DISTANCE_M = .35
CONTACT_PENALTY_M = 1.2


def potential(target_xy, yaw):
    """Distance plus bounded bearing cost, fading continuously near the target."""
    target=np.asarray(target_xy,float)
    if target.shape!=(2,) or not np.isfinite(target).all() or not math.isfinite(yaw):
        raise ValueError('finite relative waypoint and heading required')
    distance=float(np.linalg.norm(target))
    bearing=math.atan2(target[1],target[0])-yaw if distance else 0.
    alignment=TURN_LENGTH_M*min(distance/ALIGNMENT_FADE_DISTANCE_M,1.)*(1.-math.cos(bearing))
    return distance+alignment, distance, alignment


def score_commitment_pose(selection):
    require_short_forecast(selection)
    result=deepcopy(selection);p=np.asarray(result['prediction'],float)
    if (p.shape!=(6,8,5) or not np.isfinite(p).all()
            or (np.diff(p[:,:,4],axis=1)<-1e-6).any()
            or [c['action'] for c in result['candidates']]!=list(ACTIONS)):
        raise ValueError('complete ordered six-action predictions required')
    goal=np.asarray(result['goal_body_xy_m'],float)
    initial,distance,alignment=potential(goal,0.)
    result['original_candidates']=deepcopy(result['candidates'])
    result['original_action']=result['action']
    for i,row in enumerate(result['candidates']):
        dx,dy,sy,cy,logit=p[i,0]
        if np.hypot(sy,cy)<=1e-8:raise ValueError('defined committed-horizon yaw required')
        final,end_distance,end_alignment=potential(goal-[dx,dy],math.atan2(sy,cy))
        contact=float(np.exp(-np.logaddexp(0.,-logit)))
        row.update(utility_m=initial-final-CONTACT_PENALTY_M*contact,
            commitment_distance_progress_m=distance-end_distance,
            commitment_alignment_progress_m=alignment-end_alignment,
            commitment_contact_score=contact)
    chosen=max(range(6),key=lambda i:result['candidates'][i]['utility_m'])
    result.update(action=ACTIONS[chosen],action_index=chosen,requested_command=candidate_commands(ACTIONS[chosen])[0],
        score_contract='one_observation_waypoint_distance_and_bearing_potential_minus_contact',
        scored_horizon_ns=100_000_000,turn_length_m=TURN_LENGTH_M,
        alignment_fade_distance_m=ALIGNMENT_FADE_DISTANCE_M,contact_penalty_m=CONTACT_PENALTY_M,
        model_prediction_corrected=False,surface_conflict_filter_still_required=True)
    return result
