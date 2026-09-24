"""Complete fixed expanded-model correction readout; no native selection."""
import time
import numpy as np
from lewm.all_phase_translation_bias_development import correct_arrays
from lewm.all_phase_corrected_readout_development import verify_corrected_arrays,difference,summarize_primary
from lewm.observation_horizon_fit_development import score
from lewm.all_phase_training_schedule_development import SEEDS
from scripts.all_phase_translation_bias_model_admission_development import admit
from scripts.fit_go2_all_phase_training_translation_bias_v1 import OUTPUT as CORRECTION,FITS
from scripts.all_phase_study_inputs_development import stream,authenticate
from scripts.all_phase_fit_execution_development import ROSTER
from scripts.read_go2_augmented_family_switch_fits_v1 import metrics
from scripts.read_go2_training_translation_bias_v1 import assert_only_position_score_changed
from scripts.navigation_artifact_root_development import BASE,validate_root,create_output,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT=BASE/'go2_all_phase_corrected_prediction_readout_v1_attempt_001'
SOURCE='scripts/read_go2_all_phase_corrected_predictions_v1.py'
PROTOCOL='docs/go2_all_phase_corrected_prediction_readout_v1_2026-09-10.md'
TEST='lewm/tests/test_all_phase_corrected_readout_development.py'
FIT_SHA='44c4cd65812b021b29cfb0aff33e2058cfaaa71dcc627d36eced30dfac17ea35'
CORRECTION_SHA='1b36dc77ca51d342e45d73da027ebdbdbd1263ab5be4142766948fd8258dd460'
ALLOWANCE=512*1024**2


def resources():
    value=hardware()
    if value['memory_available_bytes']<8*1024**3 or value['artifact_free_bytes']<40*1024**3+ALLOWANCE:
        raise ValueError('complete corrected readout resource allowance unavailable')
    return value


def arrays(root,name,sha):
    verify_artifacts(root,{name:sha})
    with np.load(root/name,allow_pickle=False) as saved:return {k:saved[k] for k in saved.files}


def main():
    if not __debug__:raise ValueError('audit assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive complete readout; no retry/resume')
    initial_hardware=resources();admission=admit(CORRECTION_SHA)
    if admission['base_admission']['study_result_sha256']!=FIT_SHA:raise ValueError('exact completed expanded fits required')
    prior=read_json(CORRECTION,'result.json');old=read_json(CORRECTION,'launch.json')
    study=read_json(FITS,'result.json');data=stream(maximum_cache_bytes=0)
    sources=discover_sources((SOURCE,PROTOCOL,TEST),prior['source_sha256'])
    keys=('input_sha256','native_sha256','native_scene_sha256','native_geometry_sha256',
        'opencv_binary_sha256','opencv_version','rules')
    launch={k:old[k] for k in keys}
    launch.update(source_sha256=sources,protocol=PROTOCOL,output_root=str(OUTPUT),initial_hardware=initial_hardware,
        hardware=resources(),correction_admission=admission,fit_result_sha256=FIT_SHA,correction_result_sha256=CORRECTION_SHA,
        coefficient_fitting=False,corrected_transfer_evaluation=True,private_training_future_materialization=False,
        geometry_transfer_future_materialization=False,future_rgb_materialization=False,neural_inference=False,
        model_training=False,optimizer_updates=0,native_execution=False,checkpoint_selection_performed=False,
        native_assignments_changed=False,all_corrected_predictions_before_scoring=True)
    verify_ordered_launch(launch);create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    print('ALL_PHASE_CORRECTED_READOUT_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    started=time.perf_counter();files=['launch.json'];predictions={};rows=[]
    try:
        for assignment in ROSTER:
            name=assignment['name'];record=admission['coefficients'][name]
            snapshot=admission['base_admission']['snapshots'][name];prefix=snapshot['filename'][:-3]
            if (record['base_model_sha256']!=snapshot['model_sha256']
                    or any(record[k]!=assignment[k] for k in ('seed','condition','variant'))):
                raise ValueError('exact assigned corrected-model identity required')
            for role in ('train','geometry_transfer'):
                original_name=prefix+'_'+role+'.npz';original_sha=study['artifact_sha256'][original_name]
                original=arrays(FITS,original_name,original_sha);corrected=correct_arrays(original,record['heads'])
                verify_corrected_arrays(original,corrected,record['heads'])
                if original['indices'].tolist()!=data.view.indices(role):raise ValueError('complete original role population required')
                filename=name+'_'+role+'.npz'
                with (OUTPUT/filename).open('xb') as target:np.savez_compressed(target,**corrected)
                files.append(filename)
                predictions[name,role]=dict(filename=filename,sha256=digest(OUTPUT/filename),
                    original_filename=original_name,original_sha256=original_sha)
        if len(predictions)!=36:raise ValueError('all36 corrected model/role predictions required')
        complete='all_prediction_roles_complete.json'
        write_json(OUTPUT/complete,dict(correction_result_sha256=CORRECTION_SHA,fit_result_sha256=FIT_SHA,
            models=18,trained_heads=30,roles=2,predictions=[dict(model=n,role=r,**v) for (n,r),v in predictions.items()],
            corrected_scoring_started=False));files.append(complete)
        for assignment in ROSTER:
            name=assignment['name'];record=admission['coefficients'][name]
            primary='direct_outcomes' if assignment['condition']=='direct' else 'rollout_outcomes'
            for role in ('train','geometry_transfer'):
                receipt=predictions[name,role]
                original=arrays(FITS,receipt['original_filename'],receipt['original_sha256'])
                corrected=arrays(OUTPUT,receipt['filename'],receipt['sha256'])
                expected=correct_arrays(original,record['heads'])
                if any(not np.array_equal(corrected[k],v) for k,v in expected.items()):
                    raise ValueError('saved corrected arrays must reproduce exactly')
                verify_corrected_arrays(original,corrected,record['heads']);scores={}
                for head in record['heads']:
                    before=score(data.view,original,role=role,head=head)
                    after=score(data.view,corrected,role=role,head=head);assert_only_position_score_changed(before,after)
                    if head==primary and before!=read_json(FITS,receipt['original_filename'][:-4]+'_scores.json'):
                        raise ValueError('all original primary scores must reproduce exactly')
                    scores[head]=dict(before=before,after=after)
                    old_metrics,new_metrics=metrics(before),metrics(after)
                    if old_metrics.keys()!=new_metrics.keys():raise ValueError('unchanged full source/stratum population required')
                    for (source,scope),value in new_metrics.items():
                        previous=old_metrics[source,scope]
                        rows.append(dict(model=name,seed=assignment['seed'],variant=assignment['variant'],condition=assignment['condition'],
                            role=role,head=head,primary_head=head==primary,source=source,scope=scope,
                            before=previous,after=value,after_minus_before=difference(value,previous)))
                filename=name+'_'+role+'_scores.json';write_json(OUTPUT/filename,scores);files.append(filename)
                if read_json(OUTPUT,filename)!=scores:raise ValueError('saved complete scores changed')
            print('ALL_PHASE_CORRECTED_MODEL_SCORED',name,flush=True)
        summary=summarize_primary(rows,ROSTER,SEEDS)
        write_json(OUTPUT/'metrics.json',dict(per_model=rows,**summary));files.append('metrics.json')
        authenticate();verify_ordered_launch(launch)
        verify_artifacts(CORRECTION,admission['correction_artifact_sha256'])
        verify_artifacts(FITS,study['artifact_sha256']|{'result.json':FIT_SHA})
        bindings={n:digest(OUTPUT/n) for n in files};verify_artifacts(OUTPUT,bindings)
        if sum((OUTPUT/n).stat().st_size for n in files)>ALLOWANCE:raise ValueError('readout output allowance exceeded')
        write_json(OUTPUT/'result.json',dict(status='ALL_PHASE_CORRECTED_PREDICTION_READOUT_COMPLETE',
            source_sha256=sources,artifact_sha256=bindings,fit_result_sha256=FIT_SHA,correction_result_sha256=CORRECTION_SHA,
            models=18,trained_heads=30,roles=2,corrected_model_role_arrays=36,head_role_score_pairs=60,
            all_original_primary_scores_reproduced=True,all_saved_corrections_reproduced=True,
            yaw_contact_clocks_and_populations_unchanged=True,wall_s=time.perf_counter()-started,
            coefficient_fitting=False,neural_weight_updates=0,optimizer_updates=0,neural_inference=False,
            checkpoint_selection_performed=False,native_assignments_changed=False,native_execution=False,
            navigation_qualified=False,real_time_qualified=False,hardware_qualified=False,goal_achieved=False))
        print('ALL_PHASE_CORRECTED_READOUT_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_ALL_PHASE_CORRECTED_PREDICTION_READOUT_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
