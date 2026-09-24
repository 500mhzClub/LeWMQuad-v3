"""Compare combined-perception motion arms using final scored forecasts."""
import argparse
from collections import Counter
import json
import numpy as np

from lewm.commanded_planar_motion_development import forecast
from lewm.matched_motion_residual_runtime_development import FITS
from lewm.pose_command_xy_control_development import FIT_SHA256
from scripts.compare_continuous_navigation_arms_development import path, read, summarize
from scripts.compare_go2_post_training_transfer_development import SHARED_FIELDS

MOTION_SHARED_FIELDS = SHARED_FIELDS + (
    'camera_projection_selects_viewpoint', 'unresolved_viewpoints_remembered',
    'recent_reference_refresh_from_accepted_anchor', 'maximum_recent_reference_age_ns',
    'bridge_measurements_promoted', 'fixed_first_source_by_layout',
    'planned_native_assignments', 'planned_layout_indices', 'contact_score_mode')


def behavior(root, plans):
    requests=read(root,'requests.json')
    times=np.asarray([r['simulator_ns'] for r in requests])
    if len(times)>1 and not np.all(np.diff(times)==20_000_000):
        raise ValueError('complete requested 20 ms command population required')
    complete=[r for r in requests if 'applied_command' in r]
    incomplete=[i for i,r in enumerate(requests) if 'applied_command' not in r]
    if incomplete and (incomplete!=[len(requests)-1] or not (root/'failure.json').exists()):
        raise ValueError('only a terminal failed command may lack its completed-step receipt')
    commands=np.asarray([r['applied_command'] for r in complete],float).reshape(-1,3)
    translating=np.any(commands[:,:2]!=0,axis=1)
    turning=(~translating)&(commands[:,2]!=0)
    return dict(selected_action_counts=dict(Counter(p['action'] for p in plans)),
        selected_plans=len(plans),plans_on_time=sum(bool(p['on_time']) for p in plans),
        requested_command_intervals=len(requests),completed_command_intervals=len(complete),
        incomplete_terminal_command_intervals=len(incomplete),
        translation_command_seconds=float(translating.sum()*.02),
        turn_only_command_seconds=float(turning.sum()*.02),
        zero_command_seconds=float((~(translating|turning)).sum()*.02),
        command_reason_counts=dict(Counter(r['reason'] for r in requests)),
        command_durations_cover_completed_intervals_only=True,
        partial_terminal_execution_retained_in_native_physics=True,
        zero_command_is_not_a_physical_stationarity_claim=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--layout-index', type=int, choices=range(4), required=True)
    i=parser.parse_args().layout_index
    output=path(f'go2_combined_perception_motion_comparison_layout{i:02d}_v1_attempt_001')
    if output.exists(): raise ValueError('preserve completed comparison')
    summaries={}; launches={}; bindings={}; hashes={}; behaviors={}; yaw={}; xy={}
    for source in ('learned','pose_command'):
        root=path(f'go2_combined_perception_motion_{source}_noise_2mm_native_layout{i:02d}_4800_v1_attempt_001')
        launch=read(root,'launch.json')
        yaw_source='learned' if source=='learned' else 'command'
        expected_fit=FITS['supervised_rollout'][1] if source=='learned' else FIT_SHA256
        if (launch['experiment']!='combined_perception_motion_comparison_v1'
                or launch['forecast_yaw_source']!=yaw_source or launch['layout_index']!=i
                or launch['forecast_xy_source']!=source or launch['contact_score_mode']!='disabled'
                or launch['pose_command_fit_sha256']!=FIT_SHA256
                or launch['model_assignment']!='seed_2026091001_full_supervised_rollout'
                or launch['planned_native_assignments']!=8 or launch['planned_layout_indices']!=[0,1,2,3]
                or launch['motion_prediction_source']!=source
                or not launch['camera_projection_selects_viewpoint']
                or not launch['recent_reference_refresh_from_accepted_anchor']
                or launch['frozen_layout_inventory_sha256']!='b1ab545f649d0efc8a9f406418a5cc454da70a18aa434c029ce99ae45d0cef4e'):
            raise ValueError('fixed combined-motion assignment differs')
        plans=[r for r in read(root,'planning.json') if 'selection' in r]
        for plan in plans:
            c=plan['motion_correction']; nested=c['learned_motion_correction']
            if (c['forecast_yaw_source']!=yaw_source or c['contact_score_mode']!='disabled'
                    or c['forecast_xy_source']!=source or c['fit_sha256']!=expected_fit
                    or (nested['correction_root'],nested['fit_sha256'])!=FITS['supervised_rollout']
                    or c['learned_yaw_retained']!=(source=='learned')):
                raise ValueError('actual model/fit/yaw assignment differs')
            upstream=np.asarray(c['upstream_prediction_for_yaw_ablation'])
            final=np.asarray(c['applied_prediction_after_yaw_ablation'])
            commanded=forecast(plan['committed_prefix'],pulse=bool(c['terminal_translation_pulse']))
            if (upstream.shape!=(6,8,5) or final.shape!=(6,8,5)
                    or not np.isfinite(final).all() or not np.isfinite(upstream).all()
                    or not np.array_equal(upstream,c['applied_prediction_after_contact_ablation'])
                    or not np.array_equal(final[:,:,[0,1,4]],upstream[:,:,[0,1,4]])
                    or not np.array_equal(final[:,:,:2],c['corrected_forecast_xy_m'])
                    or not np.allclose(final[:,:,:2],c['learned_corrected_forecast_xy_m' if source=='learned' else 'pose_command_forecast_xy_m'],rtol=0,atol=1e-7)
                    or not np.all(final[:,:,4]==-1000)):
                raise ValueError('final forecasts differ outside the yaw intervention')
            expected=upstream[:,:,2:4] if source=='learned' else commanded[:,:,2:4]
            if (not np.allclose(final[:,:,2:4],expected,rtol=0,atol=1e-7)
                    or not np.array_equal(commanded[:,:,2:4],c['commanded_yaw_sin_cos'])
                    or any(r['predicted_contact_by_commit_end']!=0 for r in plan['selection']['candidates'])):
                raise ValueError('applied yaw or scored contact differs')
        launches[source]=launch; hashes[source]=launch['source_sha256']|launch['extra_sources']
        bindings[source]=dict(selected_plans=len(plans),final_scoring_forecasts_checked=bool(plans))
        summaries[source]=summarize(root); behaviors[source]=behavior(root,plans)
        yaw[source]={k:v for k,v in read(root,'saved_neural_command_yaw_evaluation_v1.json').items() if k!='rows'}
        xy[source]={k:v for k,v in read(root,'saved_executed_motion_forecast_evaluation_v1.json').items() if k!='rows'}
    differences=[k for k in MOTION_SHARED_FIELDS if launches['learned'].get(k)!=launches['pose_command'].get(k)]
    if differences: raise ValueError(f'shared scientific settings differ: {differences}')
    common=hashes['learned'].keys() & hashes['pose_command'].keys()
    if any(hashes['learned'][k]!=hashes['pose_command'][k] for k in common):
        raise ValueError('common runtime source changed during the pair')
    report=dict(layout_index=i, conditions=summaries, actual_treatment_bindings=bindings,
        behavior_metrics=behaviors, executed_yaw_metrics=yaw, executed_xy_metrics=xy,
        common_sources={k:hashes['learned'][k] for k in sorted(common)},
        matched_settings={k:launches['learned'].get(k) for k in MOTION_SHARED_FIELDS},
        comparison='learned_corrected_xy_and_yaw_vs_fitted_pose_command_xy_and_integrated_yaw',
        camera_frontier_policy_shared=True, accepted_reference_refresh_shared=True,
        fresh_development_layout=True, prior_development_layouts_excluded=68,
        learned_network_computed_in_both_arms=True, frozen_motion_corrections=True, contact_score_disabled=True,
        learned_xy_includes_frozen_residual_correction=True,
        predictive_planning_present_in_both_arms=True, fully_model_free_comparison=False,
        statistical_advantage_established=False, host_real_time_qualified=False, hardware_validated=False)
    output.mkdir()
    with (output/'result.json').open('x') as stream: json.dump(report,stream,indent=2)
    print(json.dumps({source:dict(round_trip=s['independent_arrival_evaluation']['round_trip_arrival_checks_passed'],
        contacts=s['independent_arrival_evaluation']['disallowed_contact_samples']) for source,s in summaries.items()}))


if __name__=='__main__': main()
