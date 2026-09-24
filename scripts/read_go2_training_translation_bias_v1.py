"""Evaluate all frozen training-only corrections without selection."""
import argparse
from copy import deepcopy
import time
import numpy as np
from lewm.training_translation_bias_development import correct_arrays
from lewm.observation_horizon_fit_development import score
from scripts.training_translation_bias_model_admission_development import admit
from scripts.fit_go2_training_translation_bias_v1 import OUTPUT as CORRECTION,FITS,FIT_SHA,load_arrays
from scripts.observation_horizon_fit_inputs_development import stream
from scripts.read_go2_augmented_family_switch_fits_v1 import metrics
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT=BASE/'go2_training_translation_bias_readout_v1_attempt_001'
PROTOCOL='docs/go2_training_translation_bias_readout_v1_2026-09-08.md'


def assert_only_position_score_changed(before,after):
    def without_position(value):
        value=deepcopy(value)
        value['clusters']=[{k:v for k,v in r.items() if k!='position_error_m'} for r in value['clusters']]
        return value
    if without_position(before)!=without_position(after):raise ValueError('XY correction changed non-position scores or population')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--correction-result-sha256',required=True);args=parser.parse_args()
    if not __debug__:raise ValueError('audit assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive complete corrected readout')
    admission=admit(args.correction_result_sha256);prior=read_json(CORRECTION,'result.json');old=read_json(CORRECTION,'launch.json')
    study=read_json(FITS,'result.json');dataset=stream(study['science']['input_result_sha256'])
    sources=discover_sources((PROTOCOL,'scripts/read_go2_training_translation_bias_v1.py',
        'lewm/tests/test_training_translation_bias_readout_development.py'),prior['source_sha256'])
    resources=hardware()
    if resources['memory_available_bytes']<8*1024**3 or resources['artifact_free_bytes']<40*1024**3+256*1024**2:
        raise ValueError('readout resource allowance unavailable')
    launch=old|dict(protocol=PROTOCOL,source_sha256=sources,output_root=str(OUTPUT),hardware=resources,
        correction_admission=admission,corrected_transfer_evaluation=True,coefficient_fitting=False,native_execution=False)
    verify(launch);create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch);started=time.perf_counter()
    print('TRAINING_TRANSLATION_BIAS_READOUT_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    artifacts=['launch.json'];rows=[];groups={}
    try:
        for name,record in admission['coefficients'].items():
            predictions={};originals={};primary='direct_outcomes' if record['condition']=='direct' else 'rollout_outcomes'
            for role in ('train','geometry_transfer'):
                original=load_arrays(name,role);corrected=correct_arrays(original,record['heads']);originals[role]=original
                for head in record['heads']:
                    if not np.array_equal(original[head][...,2:],corrected[head][...,2:]):raise ValueError('yaw/contact values changed')
                filename=name+'_'+role+'.npz'
                with (OUTPUT/filename).open('xb') as f:np.savez_compressed(f,**corrected)
                predictions[role]=dict(filename=filename,sha256=digest(OUTPUT/filename));artifacts.append(filename)
            complete=name+'_prediction_phase_complete.json'
            write_json(OUTPUT/complete,dict(correction_result_sha256=args.correction_result_sha256,
                base_model_sha256=record['base_model_sha256'],predictions=predictions));artifacts.append(complete)
            for role,receipt in predictions.items():
                with np.load(OUTPUT/receipt['filename'],allow_pickle=False) as saved:corrected={k:saved[k] for k in saved.files}
                scores={}
                for head in record['heads']:
                    before=score(dataset.view,originals[role],role=role,head=head)
                    after=score(dataset.view,corrected,role=role,head=head);assert_only_position_score_changed(before,after)
                    if head==primary and before!=read_json(FITS,name+'_'+role+'_scores.json'):
                        raise ValueError('original primary scores must reproduce exactly')
                    scores[head]=dict(before=before,after=after)
                    old_metrics,new_metrics=metrics(before),metrics(after)
                    if old_metrics.keys()!=new_metrics.keys():raise ValueError('unchanged complete strata required')
                    for (source,scope),value in new_metrics.items():
                        row=dict(model=name,seed=record['seed'],variant=record['variant'],condition=record['condition'],
                            role=role,head=head,primary_head=head==primary,source=source,scope=scope,
                            before=old_metrics[source,scope],after=value);rows.append(row)
                        if head==primary:groups.setdefault((record['variant'],record['condition'],role,source,scope),[]).append(row)
                filename=name+'_'+role+'_scores.json';write_json(OUTPUT/filename,scores);artifacts.append(filename)
        grouped=[]
        for key,members in sorted(groups.items()):
            if len(members)!=3 or {m['seed'] for m in members}!={2026091001,2026091401,2026091402}:
                raise ValueError('all three fixed seeds required')
            values={}
            for phase in ('before','after'):
                numbers=[m[phase]['position_error_m'] for m in members]
                values[phase]=None if any(v is None for v in numbers) else dict(mean=float(np.mean(numbers)),optimization_seed_sd=float(np.std(numbers,ddof=1)))
            grouped.append(dict(zip(('variant','condition','role','source','scope'),key))|dict(position_error_m=values,
                independent_maze_confidence_interval=False))
        write_json(OUTPUT/'metrics.json',dict(per_model=rows,three_seed_primary_descriptive=grouped));artifacts.append('metrics.json')
        verify(launch);verify_artifacts(CORRECTION,admission['correction_artifact_sha256'])
        verify_artifacts(FITS,{'result.json':FIT_SHA,**study['artifact_sha256']})
        write_json(OUTPUT/'result.json',dict(status='TRAINING_TRANSLATION_BIAS_READOUT_COMPLETE',models=18,
            trained_heads=30,roles=2,coefficient_fitting=False,neural_weight_updates=0,optimizer_updates=0,
            source_sha256=sources,artifact_sha256={n:digest(OUTPUT/n) for n in artifacts},
            correction_result_sha256=args.correction_result_sha256,all_coefficients_reconstructed=True,
            yaw_contact_clocks_and_populations_unchanged=True,wall_s=time.perf_counter()-started,
            checkpoint_selection_performed=False,native_execution=False,navigation_qualified=False,goal_achieved=False))
        print('TRAINING_TRANSLATION_BIAS_READOUT_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_TRANSLATION_BIAS_READOUT_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
