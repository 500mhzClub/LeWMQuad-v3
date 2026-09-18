"""All-eighteen ledger/raw-score admission, followed by explicit assigned reloads."""
import json
import numpy as np
from lewm.augmented_family_switch_fit_development import score
from scripts.run_go2_augmented_family_switch_fits_v1 import OUTPUT,ROSTER,OPTIMIZATION_SEEDS,canonical
from scripts.augmented_family_switch_fit_inputs_development import authenticate,stream,FAMILY_CHECK_SHA
from scripts.cumulative_pulse_snapshot_development import load_snapshot
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.moving_action_switch_runtime_development import verify


def admit(result_sha256):
    verify_artifacts(OUTPUT,{'result.json':result_sha256});result=read_json(OUTPUT,'result.json')
    if (result['status']!='AUGMENTED_FAMILY_SWITCH_EIGHTEEN_FITS_COMPLETE' or result['optimizer_updates']!=21600
            or result['scientific_models_trained']!=18 or len(result['records'])!=18
            or {r['name'] for r in result['records']}!=set(ROSTER)
            or result['checkpoint_selection_performed'] is not False or result['benchmark_weights_reused'] is not False):
        raise ValueError('all eighteen complete unselected fresh fits required')
    verify_artifacts(OUTPUT,result['artifact_sha256']);launch=read_json(OUTPUT,'launch.json');verify(launch)
    if launch['science']!=result['science'] or launch['source_sha256']!=result['source_sha256']:
        raise ValueError('unchanged complete scientific source and settings required')
    check_sha=launch['science']['switch_input_result_sha256'];_,_,schedules=authenticate(check_sha)
    if read_json(OUTPUT,'training_schedules.json')!=schedules:raise ValueError('all three exact mixed schedules required')
    pair=dict(family=FAMILY_CHECK_SHA,switch=check_sha)
    pair_sha=__import__('hashlib').sha256(canonical(pair)).hexdigest()
    if read_json(OUTPUT,'input_pair.json')!=pair or launch['science']['dataset_pair_sha256']!=pair_sha:
        raise ValueError('exact pair of authenticated dataset identities required')
    dataset=stream(check_sha);initial={seed:set() for seed in OPTIMIZATION_SEEDS};snapshots={}
    for record in result['records']:
        name=record['name'];request=read_json(OUTPUT,name+'_request.json');fitted=read_json(OUTPUT,name+'_fit.json')
        snapshot=fitted['snapshot'];seed=request['seed'];variant=request['variant'];condition=request['condition']
        if seed not in OPTIMIZATION_SEEDS:raise ValueError('exact optimization seed required')
        schedule=schedules[str(seed)]
        if (record['status']!='AUGMENTED_FAMILY_SWITCH_WORKER_COMPLETE' or record['actual_updates']!=1200
                or request['benchmark'] is not False or name!=f'seed_{seed}_{variant}_{condition}'
                or request['science']!=launch['science'] or fitted['fit']!=record['fit']
                or snapshot['binding']!=dict(experiment_sha256=result['artifact_sha256']['launch.json'],
                    dataset_sha256=pair_sha,schedule_sha256=schedule['schedule_sha256'],input_variant=variant)
                or snapshot['filename']!=name+'.pt'
                or snapshot['configuration']!=dict(condition=condition,seed=seed,latent_dim=32,
                    learning_rate=.001,ema_momentum=.99,updates=1200)):
            raise ValueError('complete matched fixed-configuration accounting required')
        initial[seed].add(fitted['fit']['initial_sha256']);count=0
        with (OUTPUT/(name+'_updates.jsonl')).open() as ledger:
            for count,line in enumerate(ledger,1):
                row=json.loads(line)
                if (count>1200 or row['update']!=count or row['sample_indices']!=schedule['batches'][count-1]
                        or row['schedule_sha256']!=schedule['schedule_sha256'] or row['input_variant']!=variant):
                    raise ValueError('exact every-step sample/treatment ledger required')
        if count!=1200 or row['model_sha256']!=fitted['fit']['model_sha256'] or snapshot['model_sha256']!=row['model_sha256']:
            raise ValueError('final ledger/snapshot identity mismatch')
        complete=read_json(OUTPUT,name+'_prediction_phase_complete.json')
        if complete['model_sha256']!=snapshot['model_sha256'] or set(complete['predictions'])!={'train','geometry_transfer'}:
            raise ValueError('both roles must have a complete final-model prediction receipt')
        for role in ('train','geometry_transfer'):
            filename=name+'_'+role+'.npz'
            if complete['predictions'][role]!=dict(filename=filename,sha256=result['artifact_sha256'][filename]):
                raise ValueError('raw prediction receipt changed')
            with np.load(OUTPUT/filename,allow_pickle=False) as saved:arrays={k:saved[k] for k in saved.files}
            head='direct_outcomes' if condition=='direct' else 'rollout_outcomes'
            if score(dataset.view,arrays,role=role,head=head)!=read_json(OUTPUT,name+'_'+role+'_scores.json'):
                raise ValueError('whole-population raw scores must reproduce')
        clone=load_snapshot(OUTPUT,name+'.pt',sha256=snapshot['sha256'],
            expected_binding=snapshot['binding'],expected_config=snapshot['configuration']);del clone
        snapshots[name]=snapshot
    if any(len(v)!=1 for v in initial.values()) or len(set.union(*initial.values()))!=3:
        raise ValueError('one paired initial state for each of three distinct seeds required')
    verify_artifacts(OUTPUT,{'result.json':result_sha256,**result['artifact_sha256']});verify(launch)
    return dict(study_result_sha256=result_sha256,fit_artifact_sha256=result['artifact_sha256'],snapshots=snapshots,
        all_eighteen_ledgers_raw_scores_and_snapshots_reconstructed=True,optimizer_updates=21600,
        primary_native_candidate=launch['science']['primary_native_candidate'],checkpoint_selection_performed=False,
        final_evaluation=False,navigation_qualified=False)


def load_assigned(admission,name):
    if (name not in ROSTER or set(admission['snapshots'])!=set(ROSTER)
            or admission['all_eighteen_ledgers_raw_scores_and_snapshots_reconstructed'] is not True):
        raise ValueError('complete eighteen-fit admission and explicit assigned model required')
    verify_artifacts(OUTPUT,{'result.json':admission['study_result_sha256'],**admission['fit_artifact_sha256']})
    snapshot=read_json(OUTPUT,name+'_fit.json')['snapshot'];request=read_json(OUTPUT,name+'_request.json')
    if (snapshot!=admission['snapshots'][name]
            or name!=f"seed_{request['seed']}_{request['variant']}_{request['condition']}"):
        raise ValueError('assigned final snapshot identity changed')
    clone=load_snapshot(OUTPUT,snapshot['filename'],sha256=snapshot['sha256'],
        expected_binding=snapshot['binding'],expected_config=snapshot['configuration'])
    return clone.model,request['condition'],request['variant']
