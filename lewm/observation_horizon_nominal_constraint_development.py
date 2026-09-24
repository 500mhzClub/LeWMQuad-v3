"""Make selected forecast chords consistent with the nominal route radius.

This tests all observed occupied squares. It does not bound model error,
intermediate curved motion, future gait or unobserved space.
"""
from lewm.observation_horizon_predictive_selection_development import require_short_forecast
from copy import deepcopy
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS,candidate_commands
from lewm.joint_rgbd_rigid_pose_development import proper
from lewm.observed_geometry_refinement_development import nominal_connector


def constrain(selection,position_map,rotation_map_from_body,occupied):
    require_short_forecast(selection)
    result=deepcopy(selection);p=np.asarray(position_map,float);R=proper(rotation_map_from_body)
    prediction=np.asarray(result['prediction'],float)
    if (p.shape!=(3,) or not np.isfinite(p).all() or prediction.shape!=(6,8,5)
            or not np.isfinite(prediction).all()
            or [c['action'] for c in result['candidates']]!=list(ACTIONS)
            or len(result['surface_checks'])!=6):
        raise ValueError('complete ordered forecast bank and finite current map pose required')
    occupied=sorted(occupied);checks=[]
    for i in range(6):
        endpoint=(p+R@np.r_[prediction[i,0,:2],0.])[:2]
        check=nominal_connector(p[:2],endpoint,occupied,radius_m=.45)
        checks.append(dict(action=ACTIONS[i],predicted_endpoint_map_xy_m=endpoint.tolist(),**check))
    allowed=result['phase_allowed_actions']
    feasible=[i for i in range(6) if ACTIONS[i] in allowed
        and not result['surface_checks'][i]['possible_intersection']
        and checks[i]['nominal_disk_connector_clear']]
    chosen=max(feasible,key=lambda i:result['candidates'][i]['utility_m']) if feasible else None
    result.update(before_nominal_constraint_action=result['action'],nominal_action_checks=checks,
        before_nominal_constraint_admissible_candidates=result['phase_admissible_candidates'],
        phase_admissible_candidates=len(feasible),action=None if chosen is None else ACTIONS[chosen],
        action_index=chosen,requested_command=[0.,0.,0.] if chosen is None else candidate_commands(ACTIONS[chosen])[0],
        nominal_constraint_horizon_ns=100_000_000,all_current_occupied_squares_checked=True,
        model_error_bound_applied=False,intermediate_motion_certified=False,
        original_surface_conflicts_preserved=True)
    return result
