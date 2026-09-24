"""All-model short-horizon readout and matching 500-ms predecessor comparisons."""
import argparse
import time
import numpy as np
from scripts.observation_horizon_model_admission_development import admit
from scripts.run_go2_observation_horizon_fits_v1 import OUTPUT as FITS,OPTIMIZATION_SEEDS
from scripts.run_go2_augmented_family_switch_fits_v1 import OUTPUT as PRIOR_FITS
from scripts.read_go2_augmented_family_switch_fits_v1 import OUTPUT as PRIOR_READOUT,metrics
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT=BASE/'go2_observation_horizon_fit_readout_v1_attempt_001'
PROTOCOL='docs/go2_observation_horizon_fit_readout_v1_2026-09-08.md'
PRIOR_FIT_SHA='d692829f385c2ba89c198cb0e7292d78f439969a384d47779f9258a609e38eda'
PRIOR_READOUT_SHA='335e673e02c85bb67a83bf9dd0d74a97710d5bfef9a2c43d6d31f5508ff860ab'


def baseline_scope(scope):
    if scope=='half_second':return 'first_half_second'
    for stratum in ('initial','moving','repeat','switch'):
        if scope==stratum+'_half_second':return stratum+'_first_half_second'
    return None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fit-result-sha256',required=True);args=parser.parse_args()
    if not __debug__:raise ValueError('audit assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive full short-horizon fit readout')
    admission=admit(args.fit_result_sha256);study=read_json(FITS,'result.json');old=read_json(FITS,'launch.json')
    verify_artifacts(PRIOR_READOUT,{'result.json':PRIOR_READOUT_SHA});reference=read_json(PRIOR_READOUT,'result.json')
    assert reference['status']=='AUGMENTED_FAMILY_SWITCH_FIT_READOUT_COMPLETE' and reference['final_models_admitted']==18
    reference_ids={'result.json':PRIOR_READOUT_SHA,**reference['artifact_sha256']};verify_artifacts(PRIOR_READOUT,reference_ids)
    reference_launch=read_json(PRIOR_READOUT,'launch.json');verify(reference_launch)
    prior_admission=reference_launch['all_eighteen_admission']
    assert prior_admission['study_result_sha256']==PRIOR_FIT_SHA and prior_admission['all_eighteen_ledgers_raw_scores_and_snapshots_reconstructed']
    prior_ids={'result.json':PRIOR_FIT_SHA,**prior_admission['fit_artifact_sha256']};verify_artifacts(PRIOR_FITS,prior_ids)
    prior=read_json(PRIOR_FITS,'result.json')
    if study['science']['schedule_sha256']!=prior['science']['schedule_sha256']:
        raise ValueError('same exact three context schedules required for descriptive comparison')
    sources=discover_sources((PROTOCOL,'scripts/read_go2_observation_horizon_fits_v1.py',
        'lewm/tests/test_observation_horizon_fit_readout_development.py',*reference['source_sha256']),study['source_sha256'])
    for name,h in reference['source_sha256'].items():assert sources[name]==h,name
    resources=hardware()
    if resources['memory_available_bytes']<8*1024**3 or resources['artifact_free_bytes']<40*1024**3+256*1024**2:
        raise ValueError('readout resource allowance unavailable')
    launch=old|dict(source_sha256=sources,protocol=PROTOCOL,output_root=str(OUTPUT),hardware=resources,
        all_eighteen_admission=admission,fit_result_sha256=args.fit_result_sha256,
        prior_fit_artifact_sha256=prior_ids,prior_readout_artifact_sha256=reference_ids,
        model_training=False,native_execution=False,optimizer_steps=0)
    verify(launch);create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch);started=time.perf_counter()
    try:
        rows=[];paired=[];groups={}
        for record in study['records']:
            name=record['name'];fit=record['fit']
            for role in ('train','geometry_transfer'):
                previous=metrics(read_json(PRIOR_FITS,name+'_'+role+'_scores.json'))
                for (source,scope),value in metrics(read_json(FITS,name+'_'+role+'_scores.json')).items():
                    row=dict(model=name,seed=fit['seed'],variant=fit['input_variant'],condition=fit['condition'],
                        role=role,source=source,scope=scope,**value);rows.append(row)
                    groups.setdefault((fit['input_variant'],fit['condition'],role,source,scope),[]).append(row)
                    old_scope=baseline_scope(scope)
                    if old_scope is not None:
                        baseline=previous[source,old_scope]
                        if any(value[k]!=baseline[k] for k in ('motion_targets','contact_targets','contact_positives')):
                            raise ValueError('shared 500-ms target denominators changed')
                        paired.append(dict(model=name,role=role,source=source,scope=scope,prior=baseline,current=value,
                            current_minus_prior={k:value[k]-baseline[k] if value[k] is not None and baseline[k] is not None else None
                                for k in ('position_error_m','yaw_error_rad','contact_brier')},
                            horizon_ns=500_000_000,same_context_schedule_and_seed=True,
                            whole_model_initialization_identity_claimed=False,temporal_architecture_changed=True))
        grouped=[]
        for (variant,condition,role,source,scope),members in sorted(groups.items()):
            if len(members)!=3 or {r['seed'] for r in members}!=set(OPTIMIZATION_SEEDS):raise ValueError('all three optimization seeds required')
            values={}
            for metric in ('position_error_m','yaw_error_rad','contact_brier'):
                numbers=[r[metric] for r in members]
                values[metric]=dict(mean=float(np.mean(numbers)),optimization_seed_sd=float(np.std(numbers,ddof=1))) if all(v is not None for v in numbers) else None
            grouped.append(dict(variant=variant,condition=condition,role=role,source=source,scope=scope,
                seeds=list(OPTIMIZATION_SEEDS),metrics=values,independent_maze_confidence_interval=False))
        write_json(OUTPUT/'metrics.json',dict(per_model=rows,shared_half_second_comparisons=paired,three_seed_descriptive=grouped))
        verify(launch);verify_artifacts(FITS,{'result.json':args.fit_result_sha256,**study['artifact_sha256']})
        verify_artifacts(PRIOR_FITS,prior_ids);verify_artifacts(PRIOR_READOUT,reference_ids)
        write_json(OUTPUT/'result.json',dict(status='OBSERVATION_HORIZON_FIT_READOUT_COMPLETE',
            optimizer_updates_admitted=21600,final_models_admitted=18,
            primary_native_candidate=study['science']['primary_native_candidate'],checkpoint_selection_performed=False,
            optimization_seed_variation_is_not_independent_maze_replication=True,
            shared_horizon_comparisons_preserve_exact_denominators=True,wall_s=time.perf_counter()-started,
            source_sha256=sources,artifact_sha256={n:digest(OUTPUT/n) for n in ('launch.json','metrics.json')},
            native_execution=False,navigation_qualified=False,goal_achieved=False))
        print('OBSERVATION_HORIZON_FIT_READOUT_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_OBSERVATION_HORIZON_FIT_READOUT_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
