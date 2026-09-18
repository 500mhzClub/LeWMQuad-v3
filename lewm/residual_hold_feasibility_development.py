"""Reconsider a feasible waypoint hold only for a better corrected-feasible action.

The original no-action fallback remains separate and unchanged. No new utility,
threshold, hold timeout, forecast, residual target or physical outcome is added.
"""
from copy import deepcopy
import numpy as np
from lewm.causal_executed_residual_diagnosis_development import WINDOW_TICKS
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.joint_rgbd_rigid_pose_development import proper
from lewm.observed_geometry_refinement_development import nominal_connector
from lewm.observation_horizon_predictive_selection_development import require_short_forecast
from lewm.observation_horizon_surface_filter_development import filter_selection
from lewm.observation_horizon_nominal_constraint_development import constrain
from lewm.eight_step_planning_development import plan


from lewm.residual_first_interval_feasibility_development import causal_correction


def reconsider_hold_feasibility(selection, receipt, mapper, geometry, *, now_ns):
    if (selection.get('action') != 'hold'
            or selection.get('residual_first_interval_feasibility') is not None or 'prediction' not in selection
            or selection.get('mode')!='WAYPOINT' or selection.get('view_budget_exhausted',False)
            or selection.get('intermediate_target_is_mission_goal',False)
            or selection.get('nominal_clearance_reentry',False)):
        return selection
    require_short_forecast(selection)
    hold=ACTIONS.index('hold')
    if (selection.get('action_index')!=hold
            or not selection['nominal_path_checks'][hold]['all_predicted_segments_nominally_clear']
            or selection['surface_checks'][hold]['possible_intersection']
            or 'hold' not in selection['phase_allowed_actions']):
        raise ValueError('original feasible selected hold required')
    bias=causal_correction(receipt,now_ns=now_ns)
    if not np.any(bias):return selection
    prediction=np.asarray(selection['prediction'],float)
    if (prediction.shape!=(6,8,5) or not np.isfinite(prediction).all()
            or selection.get('score_contract')!='causal_executed_waypoint_potential_minus_full_plan_contact'
            or selection.get('executed_waypoint_scoring') is not True
            or selection.get('actual_commitment_horizon_ns')!=100_000_000
            or selection.get('path_constraint_horizon_ns')!=800_000_000
            or selection.get('native_state_used') is not False
            or selection['causal_score_residual_receipt']!=receipt
            or [r['action'] for r in selection['candidates']]!=list(ACTIONS)
            or len(selection['surface_checks'])!=6
            or mapper.failed or mapper.surface.failed
            or mapper.surface.last_ns!=now_ns or len(mapper.surface.route)-1!=receipt['frame']):
        raise ValueError('current original scored forecast and same observed map required')
    corrected=prediction.copy();corrected[:,0,:2]-=bias
    utilities=np.asarray([r['utility_m'] for r in selection['candidates']],float)
    if not np.isfinite(utilities).all() or any(
            r['causal_scoring_body_xy_m']!=corrected[i,0,:2].tolist()
            for i,r in enumerate(selection['candidates'])):
        raise ValueError('exact original corrected scoring positions and utilities required')
    B=proper(mapper.map_from_initial);p=B@mapper.surface.position;R=proper(B@mapper.surface.rotation)
    current=nominal_connector(p[:2],p[:2],sorted(mapper.occupied),radius_m=.45)
    if not current['nominal_disk_connector_clear']:return selection
    # Reconstruct the frozen raw path receipts before evaluating an alternative.
    original=plan(constrain(selection,p,R,mapper.occupied),p,R,mapper.occupied)
    if (original['nominal_action_checks']!=selection['nominal_action_checks']
            or original['nominal_path_checks']!=selection['nominal_path_checks']):
        raise ValueError('original all-cell nominal veto evidence must reconstruct exactly')
    view=deepcopy(selection);view['prediction']=corrected.tolist()
    view=filter_selection(view,mapper.surface,geometry,now_ns=now_ns,persistent=True)
    checked=plan(constrain(view,p,R,mapper.occupied),p,R,mapper.occupied)
    allowed=selection['phase_allowed_actions']
    eligible=[i for i,a in enumerate(ACTIONS) if a in allowed
        and not selection['surface_checks'][i]['possible_intersection']
        and not checked['surface_checks'][i]['possible_intersection']
        and checked['nominal_path_checks'][i]['all_predicted_segments_nominally_clear']]
    # plan() computes its own 800ms utility; selection uses the original already
    # corrected 100ms waypoint utility, never that incidental recomputation.
    chosen=max(eligible,key=lambda i:utilities[i]) if eligible else None
    if chosen is None or utilities[chosen]<=utilities[hold]: return selection
    result=deepcopy(selection)
    result.update(action=None if chosen is None else ACTIONS[chosen],action_index=chosen,
        requested_command=[0.,0.,0.] if chosen is None else list(candidate_commands(ACTIONS[chosen])[0]),
        phase_admissible_candidates=len(eligible),
        residual_hold_feasibility=dict(frame=receipt['frame'],measured_ns=now_ns,
            correction_xy_m=bias.tolist(),residual_source_ticks=deepcopy(receipt['residual_source_ticks']),
            original_hold_utility_m=float(utilities[hold]),selected_utility_m=float(utilities[chosen]),
            strictly_higher_original_utility_required=True,
            original_action=selection['action'],original_phase_admissible_candidates=selection['phase_admissible_candidates'],
            current_nominal_clearance=current,corrected_first_body_xy_m=corrected[:,0,:2].tolist(),
            corrected_surface_checks=checked['surface_checks'],corrected_nominal_path_checks=checked['nominal_path_checks'],
            eligible_actions=[ACTIONS[i] for i in eligible],selected_action=None if chosen is None else ACTIONS[chosen],
            correction_horizon_ns=100_000_000,all_eight_segments_checked=True,
            later_predicted_points_unchanged=True,yaw_and_contact_forecasts_unchanged=True,
            original_forecasts_and_veto_receipts_preserved=True,original_surface_vetoes_preserved=True,
            raw_predictions_remain_residual_targets=True,nominal_policy_exception=True,
            model_error_bound_applied=False,physical_clearance_certified=False,goal_achieved=False))
    return result
