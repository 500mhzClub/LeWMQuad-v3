"""Compare a prospective anchored continuation before any later old observation."""
from copy import deepcopy
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.residual_first_interval_feasibility_development import causal_correction

MAX_FRAMES = 3004
CHOICE = ('action', 'action_index', 'requested_command', 'phase_admissible_candidates')


def _compare_anchored_step(original, candidate, actual_command, *, frame):
    if (type(frame) is not int or not 0 <= frame < MAX_FRAMES
            or original['tick'] != frame or candidate['tick'] != frame
            or original['controller'] != 'residual_first_interval_feasibility_controller_v1'
            or original['residual_first_interval_feasibility_fallback_enabled'] is not True
            or candidate['controller'] != 'residual_anchored_continuation_controller_v1'
            or candidate['residual_hold_feasibility_enabled'] is not True
            or candidate['residual_anchored_continuation_enabled'] is not True
            or original['requested_command'] != actual_command
            or candidate['failure'] is not None):
        raise ValueError('exact original residual controller and explicit nonfailed hold successor required')
    old = original['new_selection']; new = candidate['new_selection']
    changed = bool(new and 'residual_anchored_continuation' in new)
    restored = deepcopy(new)
    if changed:
        receipt = restored.pop('residual_anchored_continuation')
        if (not old or old['action'] != 'hold' or old['action_index'] != 0
                or 'residual_first_interval_feasibility' in old
                or old['mode'] != 'WAYPOINT' or old['intermediate_target_is_mission_goal']
                or old.get('nominal_clearance_reentry', False) or old['view_budget_exhausted']
                or original['terminal'] is not None or candidate['terminal'] is not None
                or receipt['original_action'] != 'hold' or receipt['frame'] != frame
                or receipt['measured_ns'] != 1_500_000_000+frame*100_000_000
                or receipt['selected_action'] != new['action'] or new['action'] in (None, 'hold')):
            raise ValueError('current feasible waypoint hold intervention required')
        for key in ('strictly_higher_original_utility_required', 'all_eight_segments_checked',
                'yaw_and_contact_forecasts_unchanged',
                'original_forecasts_and_veto_receipts_preserved', 'original_surface_vetoes_preserved',
                'raw_predictions_remain_residual_targets', 'nominal_policy_exception'):
            if receipt[key] is not True: raise ValueError('declared unchanged forecast constraint required: '+key)
        for key in ('model_error_bound_applied', 'physical_clearance_certified', 'goal_achieved',
                'later_predicted_points_unchanged', 'later_prediction_correction_calibrated'):
            if receipt[key] is not False: raise ValueError('unsupported claim: '+key)
        bias = causal_correction(old['causal_score_residual_receipt'], now_ns=receipt['measured_ns'])
        raw = np.asarray(old['prediction'], float)[:,:,:2]
        points = raw[:,0]-bias
        corrected = raw.copy(); corrected[:,0] = points
        corrected[:,1:] = points[:,None,:]+(raw[:,1:]-raw[:,:1])
        paths = receipt['corrected_nominal_path_checks']
        if len(paths) != 6 or len(receipt['corrected_surface_checks']) != 6:
            raise ValueError('complete corrected action-bank checks required')
        for action, path in zip(ACTIONS, paths, strict=True):
            segments = path['segments']
            if path['action'] != action or len(segments) != 8:
                raise ValueError('all eight ordered corrected path segments required')
            for h, segment in enumerate(segments):
                if (segment['radius_m'] != .45
                        or segment['start_offset_ns'] != h*100_000_000
                        or segment['end_offset_ns'] != (h+1)*100_000_000
                        or type(segment['nominal_disk_connector_clear']) is not bool
                        or (h and segment['predicted_start_map_xy_m'] != segments[h-1]['predicted_end_map_xy_m'])):
                    raise ValueError('continuous full-radius corrected path required')
            if path['all_predicted_segments_nominally_clear'] is not all(
                    segment['nominal_disk_connector_clear'] for segment in segments):
                raise ValueError('corrected path summary must retain every segment veto')
        utilities = [r['utility_m'] for r in old['candidates']]
        eligible = [i for i,a in enumerate(ACTIONS) if a in old['phase_allowed_actions']
            and not old['surface_checks'][i]['possible_intersection']
            and not receipt['corrected_surface_checks'][i]['possible_intersection']
            and receipt['corrected_nominal_path_checks'][i]['all_predicted_segments_nominally_clear']]
        selected = new['action_index']
        if (not old['nominal_path_checks'][0]['all_predicted_segments_nominally_clear']
                or old['surface_checks'][0]['possible_intersection'] or 'hold' not in old['phase_allowed_actions']
                or receipt['correction_xy_m'] != bias.tolist()
                or receipt['residual_source_ticks'] != old['causal_score_residual_receipt']['residual_source_ticks']
                or receipt['corrected_first_body_xy_m'] != points.tolist()
                or receipt['correction_horizon_ns'] != 800_000_000
                or receipt['residual_measurement_horizon_ns'] != 100_000_000
                or receipt['corrected_body_xy_m'] != corrected.tolist()
                or receipt['anchoring_equation'] != 'corrected_first_xy + (raw_future_xy - raw_first_xy)'
                or not receipt['current_nominal_clearance']['nominal_disk_connector_clear']
                or receipt['eligible_actions'] != [ACTIONS[i] for i in eligible]
                or not eligible or selected != max(eligible, key=lambda i:utilities[i])
                or new['action'] != ACTIONS[selected] or utilities[selected] <= utilities[0]
                or receipt['original_hold_utility_m'] != utilities[0]
                or receipt['selected_utility_m'] != utilities[selected]
                or receipt['original_phase_admissible_candidates'] != old['phase_admissible_candidates']
                or new['phase_admissible_candidates'] != len(eligible)
                or new['requested_command'] != candidate_commands(new['action'])[0]
                or candidate['requested_command'] != new['requested_command']
                or candidate['selected_action'] != new['action']):
            raise ValueError('strictly better original utility and complete corrected feasibility required')
        for key in CHOICE: restored[key] = deepcopy(old[key])
    if restored != old:
        raise ValueError('complete original raw forecast, utilities, vetoes and fallback must remain exact')
    normalized = deepcopy(candidate)
    normalized.pop('residual_hold_feasibility_enabled')
    normalized.pop('residual_anchored_continuation_enabled')
    normalized['controller'] = original['controller']; normalized['new_selection'] = restored
    if changed:
        # Both policies select a feasible menu action; wait, terminal, plan and
        # all observed/residual/mission state must therefore remain identical.
        for key in ('requested_command', 'selected_action'):
            normalized[key] = deepcopy(original[key])
    if normalized != original:
        raise ValueError('complete unchanged observed, mission, residual and execution state required')
    command_changed = candidate['requested_command'] != actual_command
    if changed != command_changed:
        raise ValueError('hold intervention must stop at its first changed physical request')
    return dict(anchored_continuation_changed_action=changed, hold_reconsideration_changed_action=changed, requested_command_changed=command_changed,
        raw_model_forecasts_compared=bool(old and 'prediction' in old),
        complete_original_selection_preserved=True, unchanged_observed_mission_and_residual_state_exact=True,
        terminal_changed=False)


from lewm.residual_hold_prefix_development import compare_step as compare_first_point


def compare_step(original, candidate, actual_command, *, frame):
    if (candidate.get('controller') != 'residual_anchored_continuation_controller_v1'
            or candidate.get('residual_anchored_continuation_enabled') is not True):
        raise ValueError('explicit anchored continuation controller required')
    selection = candidate.get('new_selection')
    if selection and 'residual_anchored_continuation' in selection:
        return _compare_anchored_step(original, candidate, actual_command, frame=frame)
    normalized = deepcopy(candidate)
    normalized.pop('residual_anchored_continuation_enabled')
    normalized['controller'] = 'residual_hold_feasibility_controller_v1'
    return compare_first_point(original, normalized, actual_command, frame=frame) | dict(
        anchored_continuation_changed_action=False)
