"""Explicit model-based recovery from an already violated nominal radius.

The original 0.45 m path veto stays recorded. Recovery is an experimental
exception, not a clearance certificate or a guarantee of measured improvement.
"""
from copy import deepcopy
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.joint_rgbd_rigid_pose_development import proper
from lewm.observed_geometry_refinement_development import nominal_connector
from lewm.observation_horizon_predictive_selection_development import require_short_forecast
from lewm.observation_horizon_waypoint_utility_development import CONTACT_PENALTY_M


def reenter(selection, position_map, rotation_map_from_body, occupied):
    if ('prediction' not in selection or selection.get('action') is not None
            or selection.get('view_budget_exhausted', False)):
        return selection
    require_short_forecast(selection)
    p = np.asarray(position_map, float); R = proper(rotation_map_from_body)
    forecast = np.asarray(selection['prediction'], float); cells = sorted(occupied)
    if (p.shape != (3,) or not np.isfinite(p).all() or forecast.shape != (6, 8, 5)
            or not np.isfinite(forecast).all() or selection.get('actual_commitment_horizon_ns') != 100_000_000
            or selection.get('path_constraint_horizon_ns') != 800_000_000
            or selection.get('native_state_used') is not False
            or [r['action'] for r in selection['candidates']] != list(ACTIONS)
            or [r['action'] for r in selection['nominal_path_checks']] != list(ACTIONS)
            or len(selection['surface_checks']) != 6):
        raise ValueError('original ordered checked forecast and observed map pose required')
    current = nominal_connector(p[:2], p[:2], cells, radius_m=.45)
    floor = current['minimum_observed_cell_distance_m']; tolerance = current['numerical_clearance_allowance_m']
    if current['nominal_disk_connector_clear'] or floor is None or floor <= 0.:
        return selection
    allowed = selection['phase_allowed_actions']
    if not allowed or any(a not in ACTIONS for a in allowed): raise ValueError('original phase allowance required')
    checks = []
    for i, action in enumerate(ACTIONS):
        points = [p[:2]]+[(p+R@np.r_[xy, 0.])[:2] for xy in forecast[i, :, :2]]
        segments = [nominal_connector(a, b, cells, radius_m=.45) for a, b in zip(points, points[1:])]
        recorded = selection['nominal_path_checks'][i]['segments']
        if len(recorded) != 8: raise ValueError('all eight original nominal segments required')
        for old, new, a, b in zip(recorded, segments, points[:-1], points[1:], strict=True):
            if (old['predicted_start_map_xy_m'] != a.tolist() or old['predicted_end_map_xy_m'] != b.tolist()
                    or any(old[k] != v for k, v in new.items())):
                raise ValueError('original raw forecast path and all-cell checks must reconstruct exactly')
        endpoint = nominal_connector(points[1], points[1], cells, radius_m=.45)['minimum_observed_cell_distance_m']
        minimum = min(s['minimum_observed_cell_distance_m'] for s in segments)
        nonworsening = minimum >= floor-tolerance
        gain = endpoint-floor
        eligible = bool(action != 'hold' and action in allowed and not selection['surface_checks'][i]['possible_intersection']
            and nonworsening and gain > tolerance)
        contact = float(np.exp(-np.logaddexp(0., -forecast[i, -1, 4])))
        checks.append(dict(action=action, eligible=eligible, phase_allowed=action in allowed,
            surface_clear=not selection['surface_checks'][i]['possible_intersection'],
            all_eight_segments_at_least_current_clearance=nonworsening,
            minimum_forecast_path_clearance_m=minimum, first_endpoint_clearance_m=endpoint,
            first_endpoint_clearance_gain_m=gain, full_plan_contact_score=contact,
            recovery_utility_m=gain-CONTACT_PENALTY_M*contact))
    eligible = [i for i, r in enumerate(checks) if r['eligible']]
    if not eligible: return selection
    chosen = max(eligible, key=lambda i: checks[i]['recovery_utility_m'])
    result = deepcopy(selection)
    result.update(action=ACTIONS[chosen], action_index=chosen,
        requested_command=list(candidate_commands(ACTIONS[chosen])[0]), phase_admissible_candidates=len(eligible),
        original_score_contract=selection['score_contract'],
        score_contract='current_clearance_reentry_gain_minus_full_plan_contact',
        nominal_clearance_reentry=True, reentry_current_clearance=current,
        reentry_candidates=checks, original_nominal_radius_m=.45,
        original_nominal_path_veto_preserved=True, original_candidate_utilities_preserved=True,
        recovery_is_nominal_policy_exception=True, reentry_guaranteed=False,
        model_error_bound_applied=False, articulated_motion_certified=False)
    return result
