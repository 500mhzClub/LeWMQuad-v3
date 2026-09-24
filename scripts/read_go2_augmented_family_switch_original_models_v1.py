"""All original six models on both fixed populations, without optimization."""
import argparse
import json
import time
import cv2
import numpy as np
import torch
from lewm.augmented_family_switch_fit_development import predict,score
from scripts.augmented_family_switch_fit_inputs_development import authenticate,stream
from scripts.family_transition_model_admission_development import admit
from scripts.run_go2_family_transition_fits_v1 import OUTPUT as FITS,ROSTER
from scripts.cumulative_pulse_snapshot_development import load_snapshot
from scripts.moving_action_switch_runtime_development import verify,hardware
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

OUTPUT=BASE/'go2_augmented_family_switch_original_models_v1_attempt_001'
PROTOCOL='docs/go2_augmented_family_switch_original_models_v1_2026-09-08.md'
FIT_SHA='ed3e2f6385991439fd390ffc64e647f6763fb3576b35f7c767fab19e4a29398c'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--switch-input-result-sha256',required=True);args=parser.parse_args()
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive complete old-model prediction comparison; no retry/resume')
    definition,checked,_=authenticate(args.switch_input_result_sha256)
    unused,admission=admit(FIT_SHA);del unused
    old=read_json(FITS,'result.json');sources=dict(definition['source_sha256'])
    for name,h in old['source_sha256'].items():
        if name in sources and sources[name]!=h:raise ValueError('old fit and current inputs disagree on source identity')
        sources[name]=h
    sources=discover_sources((PROTOCOL,'scripts/read_go2_augmented_family_switch_original_models_v1.py',
        'lewm/tests/test_augmented_family_switch_view_development.py'),sources)
    launch=definition|dict(source_sha256=sources,protocol=PROTOCOL,output_root=str(OUTPUT),
        fit_result_sha256=FIT_SHA,all_six_admission=admission,
        fit_artifact_sha256=old['artifact_sha256'],switch_input_result_sha256=args.switch_input_result_sha256,
        hardware=hardware(),optimizer_steps=0,model_roster=list(ROSTER),native_execution=False)
    if launch['hardware']['memory_available_bytes']<8*1024**3 or launch['hardware']['artifact_free_bytes']<40*1024**3+256*1024**2:
        raise ValueError('bounded comparison RAM and storage allowance required')
    verify(launch);cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    start=time.perf_counter();bindings={};models={}
    try:
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            for name in ROSTER:
                request=read_json(FITS,name+'_request.json');snapshot=read_json(FITS,name+'_fit.json')['snapshot']
                clone=load_snapshot(FITS,snapshot['filename'],sha256=snapshot['sha256'],
                    expected_binding=snapshot['binding'],expected_config=snapshot['configuration'])
                dataset=stream(args.switch_input_result_sha256);files={};matches={}
                for role in ('train','geometry_transfer'):
                    arrays=predict(clone,dataset,role=role,input_variant=request['variant'])
                    # The original source remains byte-exact at the array level.
                    count=len(dataset.view.indices(role,source='family'))
                    with np.load(FITS/(name+'_'+role+'.npz'),allow_pickle=False) as previous:
                        if set(arrays)!=set(previous.files) or not all(np.array_equal(arrays[k][:count],previous[k]) for k in previous.files):
                            raise ValueError('unchanged original population forecasts must reproduce exactly')
                    file=name+'_'+role+'.npz'
                    with (OUTPUT/file).open('xb') as target:np.savez_compressed(target,**arrays)
                    bindings[file]=digest(OUTPUT/file);files[role]=file;matches[role]=dict(samples=count,all_fields_exact=True)
                models[name]=dict(condition=request['condition'],variant=request['variant'],snapshot=snapshot,
                    files=files,unchanged_original_predictions=matches,model_state_unchanged=True)
                monitor.write(json.dumps(dict(completed_model=name,**hardware()))+'\n');monitor.flush()
                print('ORIGINAL_AUGMENTED_PREDICTED',name,flush=True);del clone,dataset
        # Scoring starts only after every model/role prediction has been saved.
        verify_artifacts(OUTPUT,bindings)
        write_json(OUTPUT/'prediction_phase_complete.json',dict(models=models,prediction_sha256=bindings,
            fit_result_sha256=FIT_SHA,optimizer_steps=0,source_sha256=sources))
        bindings['prediction_phase_complete.json']=digest(OUTPUT/'prediction_phase_complete.json')
        dataset=stream(args.switch_input_result_sha256);scores={}
        for name,row in models.items():
            scores[name]={};head='direct_outcomes' if row['condition']=='direct' else 'rollout_outcomes'
            for role,file in row['files'].items():
                with np.load(OUTPUT/file,allow_pickle=False) as saved:arrays={k:saved[k] for k in saved.files}
                scores[name][role]=score(dataset.view,arrays,role=role,head=head)
        write_json(OUTPUT/'scores.json',scores);bindings['scores.json']=digest(OUTPUT/'scores.json')
        for name in ('launch.json','resource_monitor.jsonl'):bindings[name]=digest(OUTPUT/name)
        authenticate(args.switch_input_result_sha256);verify(launch);verify_artifacts(FITS,{'result.json':FIT_SHA,**old['artifact_sha256']})
        verify_artifacts(OUTPUT,bindings)
        result=dict(status='AUGMENTED_FAMILY_SWITCH_ORIGINAL_MODELS_COMPLETE',models=models,
            source_sha256=sources,artifact_sha256=bindings,wall_s=time.perf_counter()-start,
            optimizer_steps=0,new_model_trained=False,checkpoint_selection_performed=False,
            geometry_transfer_is_independent_maze_evaluation=False,native_execution=False,navigation_qualified=False,goal_achieved=False)
        write_json(OUTPUT/'result.json',result);print('ORIGINAL_AUGMENTED_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_ORIGINAL_AUGMENTED_PREDICTION_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
