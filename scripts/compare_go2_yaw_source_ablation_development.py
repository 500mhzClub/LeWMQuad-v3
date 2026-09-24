"""Compare completed yaw arms using the final forecasts returned to the scorer."""
import argparse
import json
import numpy as np

from lewm.commanded_planar_motion_development import forecast
from lewm.matched_motion_residual_runtime_development import FITS
from lewm.pose_command_xy_control_development import FIT_SHA256
from scripts.compare_continuous_navigation_arms_development import path, read, summarize
from scripts.compare_go2_contact_score_ablation_development import behavior
from scripts.compare_go2_post_training_transfer_development import SHARED_FIELDS


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--layout-index', type=int, choices=(0,1), required=True)
    i=parser.parse_args().layout_index
    output=path(f'go2_yaw_source_ablation_comparison_layout{i:02d}_v1_attempt_001')
    if output.exists(): raise ValueError('preserve completed comparison')
    summaries={}; launches={}; bindings={}; hashes={}; behaviors={}; yaw={}; xy={}
    for source in ('learned','command'):
        root=path(f'go2_yaw_source_ablation_{source}_noise_2mm_native_layout{i:02d}_4800_v1_attempt_001')
        launch=read(root,'launch.json')
        if (launch['experiment']!='yaw_source_ablation_arrival_frontier_v1'
                or launch['forecast_yaw_source']!=source or launch['layout_index']!=i
                or launch['forecast_xy_source']!='pose_command' or launch['contact_score_mode']!='disabled'
                or launch['pose_command_fit_sha256']!=FIT_SHA256
                or launch['model_assignment']!='seed_2026091001_full_supervised_rollout'
                or launch['planned_native_assignments']!=4 or launch['planned_layout_indices']!=[0,1]
                or not launch['arrival_panorama_view_start_position_recorded']):
            raise ValueError('fixed yaw-study assignment differs')
        plans=[r for r in read(root,'planning.json') if 'selection' in r]
        for plan in plans:
            c=plan['motion_correction']; nested=c['learned_motion_correction']
            if (c['forecast_yaw_source']!=source or c['contact_score_mode']!='disabled'
                    or c['forecast_xy_source']!='pose_command' or c['fit_sha256']!=FIT_SHA256
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
                    or not np.allclose(final[:,:,:2],c['pose_command_forecast_xy_m'],rtol=0,atol=1e-7)
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
    differences=[k for k in SHARED_FIELDS if launches['learned'].get(k)!=launches['command'].get(k)]
    if differences: raise ValueError(f'shared scientific settings differ: {differences}')
    common=hashes['learned'].keys() & hashes['command'].keys()
    if any(hashes['learned'][k]!=hashes['command'][k] for k in common):
        raise ValueError('common runtime source changed during the pair')
    report=dict(layout_index=i, conditions=summaries, actual_treatment_bindings=bindings,
        behavior_metrics=behaviors, executed_yaw_metrics=yaw, executed_xy_metrics=xy,
        common_sources={k:hashes['learned'][k] for k in sorted(common)},
        matched_settings={k:launches['learned'].get(k) for k in SHARED_FIELDS},
        comparison='learned_vs_command_yaw_with_pose_command_xy_and_disabled_contact',
        repaired_frontier_policy_shared=True, exposed_development_layout=True,
        learned_network_computed_in_both_arms=True, fitted_xy_predictor_retained=True,
        predictive_planning_present_in_both_arms=True, fully_model_free_comparison=False,
        statistical_advantage_established=False, host_real_time_qualified=False, hardware_validated=False)
    output.mkdir()
    with (output/'result.json').open('x') as stream: json.dump(report,stream,indent=2)
    print(json.dumps({source:dict(round_trip=s['independent_arrival_evaluation']['round_trip_arrival_checks_passed'],
        contacts=s['independent_arrival_evaluation']['disallowed_contact_samples']) for source,s in summaries.items()}))


if __name__=='__main__': main()
