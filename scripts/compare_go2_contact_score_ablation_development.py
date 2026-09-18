"""Compare one completed pair in the fixed contact-score development pilot."""
import argparse
from collections import Counter
import json
import numpy as np
from lewm.contact_score_ablation_development import MODES, DISABLED_LOGIT
from lewm.matched_motion_residual_runtime_development import FITS
from lewm.pose_command_xy_control_development import FIT_SHA256
from scripts.compare_continuous_navigation_arms_development import path,read,summarize
from scripts.compare_go2_post_training_transfer_development import SHARED_FIELDS


def behavior(root, plans):
    requests=read(root,'requests.json')
    times=np.asarray([r['simulator_ns'] for r in requests])
    if len(times)>1 and not np.all(np.diff(times)==20_000_000):
        raise ValueError('complete 20 ms command population required')
    commands=np.asarray([r['applied_command'] for r in requests],float)
    if commands.shape!=(len(requests),3) or not np.isfinite(commands).all():
        raise ValueError('complete finite applied commands required')
    translating=np.any(commands[:,:2]!=0,axis=1)
    turning=(~translating)&(commands[:,2]!=0)
    zero=~(translating|turning)
    return dict(selected_action_counts=dict(Counter(p['action'] for p in plans)),
        selected_plans=len(plans),plans_on_time=sum(bool(p['on_time']) for p in plans),
        command_intervals=len(requests),translation_command_seconds=float(translating.sum()*.02),
        turn_only_command_seconds=float(turning.sum()*.02),zero_command_seconds=float(zero.sum()*.02),
        command_reason_counts=dict(Counter(r['reason'] for r in requests)),
        zero_command_is_not_a_physical_stationarity_claim=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--layout-index',type=int,choices=(0,1),required=True)
    i=parser.parse_args().layout_index
    output=path(f'go2_contact_score_ablation_pose_command_xy_comparison_layout{i:02d}_v1_attempt_001')
    if output.exists():raise ValueError('preserve completed pilot comparison')
    launches={};summaries={};bindings={};hashes={};forecasts={};behaviors={}
    for mode in MODES:
        root=path(f'go2_contact_score_ablation_pose_command_xy_{mode}_noise_2mm_native_layout{i:02d}_4800_v1_attempt_001')
        launch=read(root,'launch.json')
        if (launch['experiment']!='contact_score_ablation_pose_command_xy_pilot_v1' or
                launch['contact_score_mode']!=mode or launch['layout_index']!=i or
                launch['model_assignment']!='seed_2026091001_full_supervised_rollout' or
                launch['forecast_xy_source']!='pose_command' or
                launch['pose_command_fit_sha256']!=FIT_SHA256 or
                launch['planned_layout_indices']!=[0,1] or launch['planned_native_assignments']!=4 or
                launch['new_independent_development_layout'] is not False):
            raise ValueError('fixed pilot assignment required')
        plans=[p for p in read(root,'planning.json') if 'selection' in p]
        for p in plans:
            c=p['motion_correction'];original=c['learned_motion_correction']
            if (c['contact_score_mode']!=mode or c['forecast_xy_source']!='pose_command' or
                    c['fit_sha256']!=FIT_SHA256 or c['correction_base_model']!='pose_command_only' or
                    (original['correction_root'],original['fit_sha256'])!=FITS['supervised_rollout'] or
                    original['correction_base_model']!='seed_2026091001_full_supervised_rollout'):
                raise ValueError('actual model, fit or treatment differs')
            upstream=np.asarray(c['upstream_prediction_for_contact_ablation'])
            applied=np.asarray(c['applied_prediction_after_contact_ablation'])
            if (upstream.shape!=(6,8,5) or applied.shape!=(6,8,5) or
                    not np.isfinite(upstream).all() or not np.isfinite(applied).all() or
                    not np.array_equal(upstream[:,:,:4],applied[:,:,:4]) or
                    not np.array_equal(applied[:,:,:2],c['corrected_forecast_xy_m']) or
                    not np.allclose(applied[:,:,:2],c['pose_command_forecast_xy_m'],rtol=0,atol=1e-7)):
                raise ValueError('non-contact channels or fixed pose-command XY differ')
            expected=upstream[:,:,4] if mode=='learned' else np.full((6,8),DISABLED_LOGIT)
            if not np.array_equal(applied[:,:,4],expected):raise ValueError('actual contact channel differs')
            probability=np.exp(-np.logaddexp(0.,-applied[:,6,4]))
            if not np.allclose(probability,[r['predicted_contact_by_commit_end'] for r in p['selection']['candidates']],rtol=0,atol=1e-12):
                raise ValueError('scored probabilities differ from applied contact channel')
        bindings[mode]=dict(selected_plans=len(plans),actual_channels_checked=bool(plans))
        launches[mode]=launch;hashes[mode]=launch['source_sha256']|launch['extra_sources']
        summaries[mode]=summarize(root)
        behaviors[mode]=behavior(root,plans)
        forecasts[mode]={k:v for k,v in read(root,'saved_executed_motion_forecast_evaluation_v1.json').items() if k!='rows'}
    differences=[k for k in SHARED_FIELDS if launches['learned'].get(k)!=launches['disabled'].get(k)]
    if differences:raise ValueError(f'shared pilot settings differ: {differences}')
    common=set(hashes['learned'])&set(hashes['disabled'])
    if any(hashes['learned'][k]!=hashes['disabled'][k] for k in common):raise ValueError('common runtime source changed')
    report=dict(layout_index=i,conditions=summaries,actual_treatment_bindings=bindings,
        executed_forecast_metrics=forecasts,behavior_metrics=behaviors,
        common_sources={k:hashes['learned'][k] for k in sorted(common)},
        matched_settings={k:launches['learned'].get(k) for k in SHARED_FIELDS},
        comparison='learned_contact_score_vs_disabled_with_pose_command_xy_and_learned_yaw',
        development_layout_revisits=True,zero_score_is_not_contact_free_prediction=True,
        learned_yaw_retained=True,fully_model_free_comparison=False,
        statistical_advantage_established=False,host_real_time_qualified=False,hardware_validated=False)
    output.mkdir()
    with (output/'result.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({mode:dict(round_trip=s['independent_arrival_evaluation']['round_trip_arrival_checks_passed'],
        disallowed_contacts=s['independent_arrival_evaluation']['disallowed_contact_samples']) for mode,s in summaries.items()}))


if __name__=='__main__':main()
