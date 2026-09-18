"""Explain saved feasible holds without inventing alternative physical outcomes."""
from collections import Counter
import math
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.observation_horizon_waypoint_utility_development import potential, CONTACT_PENALTY_M
from lewm.residual_first_interval_feasibility_development import causal_correction


def classify_hold(selection, *, frame):
    s = selection; candidates = s['candidates']; paths = s['nominal_path_checks']
    if (s['action'] != 'hold' or s['action_index'] != ACTIONS.index('hold')
            or s['score_contract'] != 'causal_executed_waypoint_potential_minus_full_plan_contact'
            or s['mode'] != 'WAYPOINT' or s['intermediate_target_is_mission_goal']
            or s.get('nominal_clearance_reentry', False)
            or s['scored_pose_horizon_ns'] != 100_000_000 or s['scored_contact_horizon_ns'] != 800_000_000
            or [c['action'] for c in candidates] != list(ACTIONS)
            or [p['action'] for p in paths] != list(ACTIONS) or len(s['surface_checks']) != 6):
        raise ValueError('complete original intermediate-waypoint hold required')
    prediction = np.asarray(s['prediction'], float)
    if (prediction.shape != (6, 8, 5) or not np.isfinite(prediction).all()
            or np.any(np.hypot(prediction[:, :, 2], prediction[:, :, 3]) <= 1e-8)
            or (np.diff(prediction[:, :, 4], axis=1) < -1e-6).any()):
        raise ValueError('complete finite raw forecast required')
    receipt = s['causal_score_residual_receipt']
    if receipt['frame'] != frame: raise ValueError('same current causal residual frame required')
    bias = causal_correction(receipt, now_ns=1_500_000_000+frame*100_000_000)
    initial, distance, alignment = potential(s['goal_body_xy_m'], 0.)
    allowed = s['phase_allowed_actions']
    if not allowed or any(a not in ACTIONS for a in allowed): raise ValueError('original phase allowance required')
    alternatives = []; hold = candidates[ACTIONS.index('hold')]
    for i, candidate in enumerate(candidates):
        xy = prediction[i, 0, :2]-bias
        final, end_distance, end_alignment = potential(np.asarray(s['goal_body_xy_m'])-xy,
            math.atan2(prediction[i, 0, 2], prediction[i, 0, 3]))
        contact = float(np.exp(-np.logaddexp(0., -prediction[i, -1, 4])))
        expected = dict(utility_m=initial-final-CONTACT_PENALTY_M*contact,
            causal_scoring_body_xy_m=xy.tolist(), executed_waypoint_distance_progress_m=distance-end_distance,
            executed_waypoint_alignment_progress_m=alignment-end_alignment, full_plan_contact_score=contact)
        if any(candidate[k] != v for k,v in expected.items()):
            raise ValueError('saved utility and components must reconstruct exactly from current forecasts')
        segments = paths[i]['segments']
        if len(segments) != 8: raise ValueError('all eight original path segments required')
        for h, segment in enumerate(segments):
            if (segment['start_offset_ns'] != h*100_000_000 or segment['end_offset_ns'] != (h+1)*100_000_000
                    or segment['radius_m'] != .45 or type(segment['nominal_disk_connector_clear']) is not bool):
                raise ValueError('original ordered 45 cm nominal segment checks required')
        blocked = [h for h,segment in enumerate(segments) if not segment['nominal_disk_connector_clear']]
        if paths[i]['all_predicted_segments_nominally_clear'] is not (not blocked):
            raise ValueError('complete nominal path summary must match all segments')
        surface = s['surface_checks'][i]['possible_intersection']
        if type(surface) is not bool: raise ValueError('explicit original surface veto required')
        if candidate['action'] == 'hold':
            if surface or blocked or 'hold' not in allowed: raise ValueError('original selected hold must be feasible')
        elif candidate['action'] in allowed and candidate['utility_m'] > hold['utility_m']:
            if not surface and not blocked: raise ValueError('original hold lost to an originally feasible alternative')
            alternatives.append(dict(action=candidate['action'], utility_m=candidate['utility_m'],
                utility_above_hold_m=candidate['utility_m']-hold['utility_m'],
                original_surface_veto=surface, blocked_segment_indices=blocked,
                unchanged_suffix_veto=any(h >= 2 for h in blocked),
                blocked_by_first_point_invariant_checks=surface or any(h >= 2 for h in blocked)))
    if not alternatives: reason = 'no_strictly_better_allowed_nonhold'
    elif all(a['blocked_by_first_point_invariant_checks'] for a in alternatives):
        reason = 'every_better_nonhold_has_a_first_point_invariant_veto'
    else: reason = 'at_least_one_better_nonhold_blocked_only_in_first_two_segments'
    return dict(frame=frame, reason=reason, hold_utility_m=hold['utility_m'], correction_xy_m=bias.tolist(),
        bias_nonzero=bool(np.any(bias)), better_allowed_nonholds=alternatives,
        original_surface_vetoes_preserved=True, first_point_only_correction_changes_segments=[0, 1],
        segments_2_through_7_endpoints_unchanged=True, corrected_path_feasibility_not_recomputed=True,
        alternative_physical_outcomes_not_observed=True)


def summarize_hold_vetoes(rows, tape):
    frames = holds = 0; counts = Counter(); action_blocks = Counter(); first = {}; latest = None
    first_terminal = None
    for row in rows:
        i = frames; d = row['decision']; frames += 1
        if row['tick'] != i or row['observation_index'] != i or row['pre_sample_index'] != 749+50*i:
            raise ValueError('complete ordered original observations required')
        if i < len(tape) and (tape[i]['completed'] is not True or tape[i]['requested_command'] != d['requested_command']):
            raise ValueError('matching completed original command required')
        if first_terminal is None and d['terminal'] is not None: first_terminal = dict(frame=i, terminal=d['terminal'])
        s = d['new_selection']
        if s is None or s.get('action') != 'hold': continue
        if d['terminal'] is not None or d['requested_command'] != [0., 0., 0.]:
            raise ValueError('nonterminal actual hold required')
        holds += 1; classified = classify_hold(s, frame=i); reason = classified['reason']
        counts[reason] += 1; first.setdefault(reason, classified); latest = classified
        for alternative in classified['better_allowed_nonholds']:
            if alternative['original_surface_veto']: action_blocks['original_surface_veto'] += 1
            for h in alternative['blocked_segment_indices']: action_blocks['nominal_segment_'+str(h)] += 1
    if frames != len(tape)+1: raise ValueError('complete final observation and all actual requests required')
    return dict(observations=frames, selected_holds=holds, first_terminal=first_terminal,
        hold_reason_counts=dict(counts), better_alternative_veto_counts=dict(action_blocks),
        first_examples=first, latest_hold=latest, every_hold_utility_reconstructed=True,
        original_controller_reexecuted=False, model_loaded=False, native_pose_used=False,
        corrected_paths_recomputed=False, alternative_physical_outcomes_inferred=False,
        controller_or_weights_changed=False, navigation_qualified=False)
