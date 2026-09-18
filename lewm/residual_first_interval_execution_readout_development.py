"""Score only executed residual-feasibility choices, with native outcomes post hoc."""
from collections import Counter
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.nominal_reentry_execution_readout_development import executed_motion
from lewm.residual_first_interval_feasibility_development import causal_correction


def summarize_execution(poses,tape,rows):
    entries=[]; attempts=[]; actions=Counter(); terminals=Counter(); frames=0; first_terminal=None
    for row in rows:
        tick=row['tick']; d=row['decision']; s=d['new_selection'] or {}
        if tick!=frames: raise ValueError('complete ordered native decision stream required')
        frames+=1
        if d['terminal'] is not None:
            terminals[d['terminal']]+=1
            if first_terminal is None:
                first_terminal=dict(tick=tick,terminal=d['terminal'],failure=d['failure'])
        if s.get('action') is not None: actions[s['action']]+=1
        r=s.get('residual_first_interval_feasibility')
        if r is None: continue
        correction=causal_correction(s['causal_score_residual_receipt'],now_ns=1_500_000_000+tick*100_000_000)
        if (r['frame']!=tick or r['measured_ns']!=1_500_000_000+tick*100_000_000
                or r['original_action'] is not None or r['selected_action']!=s['action']
                or r['correction_xy_m']!=correction.tolist()
                or r['residual_source_ticks']!=s['causal_score_residual_receipt']['residual_source_ticks']
                or any(r[k] is not True for k in ('all_eight_segments_checked',
                    'original_forecasts_and_veto_receipts_preserved','original_surface_vetoes_preserved',
                    'later_predicted_points_unchanged','yaw_and_contact_forecasts_unchanged',
                    'raw_predictions_remain_residual_targets'))
                or r['model_error_bound_applied'] is not False or r['physical_clearance_certified'] is not False):
            raise ValueError('exact declared causal feasibility exception required')
        attempts.append(dict(tick=tick,selected_action=s['action'],eligible_actions=r['eligible_actions']))
        if s['action'] is None: continue
        action=s['action']; index=ACTIONS.index(action); path=r['corrected_nominal_path_checks'][index]
        prediction=np.asarray(s['prediction'],float)
        if prediction.shape!=(6,8,5) or not np.isfinite(prediction).all():
            raise ValueError('full finite original forecast required')
        xy=prediction[index,0,:2]-correction
        if (d['terminal'] is not None or d['selected_action']!=action
                or d['requested_command']!=candidate_commands(action)[0]
                or action not in r['eligible_actions'] or action not in s['phase_allowed_actions']
                or s['surface_checks'][index]['possible_intersection']
                or r['corrected_surface_checks'][index]['possible_intersection']
                or not path['all_predicted_segments_nominally_clear'] or len(path['segments'])!=8
                or any(segment['radius_m']!=.45 or not segment['nominal_disk_connector_clear'] for segment in path['segments'])
                or r['corrected_first_body_xy_m'][index]!=xy.tolist()
                or s['candidates'][index]['causal_scoring_body_xy_m']!=xy.tolist()):
            raise ValueError('actual selected corrected-feasible command and original vetoes required')
        entries.append(dict(tick=tick,action=action,requested_command=d['requested_command'],
            predicted_body_xy_m=prediction[index,0,:2].tolist(),corrected_first_body_xy_m=xy.tolist(),
            correction_xy_m=correction.tolist(),residual_source_ticks=r['residual_source_ticks'],
            original_selected_path_nominally_clear=s['nominal_path_checks'][index]['all_predicted_segments_nominally_clear'],
            corrected_minimum_observed_clearance_m=min(segment['minimum_observed_cell_distance_m'] for segment in path['segments'])))
    records=executed_motion(poses,tape,entries)
    for r in records:
        r['corrected_xy_error_m']=(float(np.linalg.norm(np.asarray(r['corrected_first_body_xy_m'])-r['native_body_xy_m']))
            if r['complete_100ms_execution'] else None)
    complete=[r for r in records if r['complete_100ms_execution']]
    return dict(observations=frames,first_terminal=first_terminal,terminal_counts=dict(terminals),selected_actions=dict(actions),
        fallback_attempts=attempts,fallback_selected_intervals=len(records),completed_fallback_intervals=len(complete),
        censored_fallback_intervals=len(records)-len(complete),fallback_execution=records,
        raw_forecast_mean_xy_error_m=float(np.mean([r['forecast_xy_error_m'] for r in complete])) if complete else None,
        corrected_forecast_mean_xy_error_m=float(np.mean([r['corrected_xy_error_m'] for r in complete])) if complete else None,
        native_outcomes_used_for_policy=False,unexecuted_alternative_outcomes_inferred=False,
        original_controller_counterfactual_trajectory_inferred=False,physical_clearance_certified=False)
