"""Matched six-model executed-transition diagnostic; no native run or fit."""
import argparse
from collections import deque, Counter
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
import torch
import cv2
from lewm.causal_subtrajectory_learning_development import causal_history_tensors
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.executed_half_second_diagnostic_development import predict, executed_outcome, error
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.family_transition_model_admission_development import admit, NAME
from scripts.run_go2_family_transition_fits_v1 import OUTPUT as FITS, ROSTER
from scripts.cumulative_pulse_snapshot_development import load_snapshot
from scripts.run_go2_active_view_goal_probe_v1 import OUTPUT as INPUT, CASES
from scripts.read_go2_active_view_goal_probe_v1 import OUTPUT as READOUT
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.near_field_sensor_audit_development import read_npz
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT=BASE/'go2_active_view_six_model_transitions_v1_attempt_001'
PROTOCOL='docs/go2_active_view_six_model_transitions_v1_2026-09-08.md'
FIT_SHA='ed3e2f6385991439fd390ffc64e647f6763fb3576b35f7c767fab19e4a29398c'
PROBE_SHA='bdd1b21cec0d02413f5833f5ff2a27bfde9a95af72c5aa8c03056f593c5db3bb'
READOUT_SHA='83bdeaa21939bb9cc2c2a3cec351bf5445e2f4eb0c1b8225d195c039f01cf840'


def contexts():
    """Only past public packets feed history; native labels are loaded later."""
    population=[]; histories=[]
    for name, _, _ in CASES:
        rows=read_json(INPUT/name,'context_decisions.json')
        reader=IntentReturnRGBDReplay(INPUT/name); history=deque(maxlen=4)
        for tick,row in enumerate(rows):
            assert row['tick']==tick
            policy,_,_,now=reader.packet(tick);history.append(policy)
            selection=row['decision']['new_selection']
            if selection is None or 'prediction' not in selection:continue
            assert len(history)==4 and selection['action'] in ACTIONS
            tensors=causal_history_tensors(list(history),now)
            histories.append(tensors)
            population.append(dict(case=name,tick=tick,decision_ns=now,
                action=selection['action'],phase=selection['mode'],
                history_sha256=state_digest(tensors),recorded_prediction=selection['prediction']))
    assert len(population)==94
    return population,histories


def model_predictions(job, histories):
    name,model,request=job;started=time.perf_counter()
    values=np.stack([predict(model,h,condition=request['condition'],variant=request['variant']) for h in histories])
    return name,values,time.perf_counter()-started


def execute(jobs,histories,width):
    with ThreadPoolExecutor(max_workers=width) as pool:
        futures=[pool.submit(model_predictions,j,histories) for j in jobs]
        return [f.result() for f in futures]


def statistics(rows):
    result=dict(n=len(rows),undefined_yaw=sum(r['yaw_error_rad'] is None for r in rows))
    for key in ('xy_error_m','yaw_error_rad','contact_brier'):
        values=[r[key] for r in rows if r[key] is not None]
        result[key]=dict(mean=float(np.mean(values)),median=float(np.median(values)),
            maximum=float(np.max(values))) if values else None
    return result


def summarize(rows):
    reports=[]
    for case,_,_ in CASES:
        for group in ('all','hold','nonzero'):
            chosen=[r for r in rows if r['case']==case and r['outcome']['eligible'] and
                (group=='all' or (r['action']=='hold')==(group=='hold'))]
            for name in (*ROSTER,'zero_motion_reference'):
                reports.append(dict(case=case,group=group,model=name,
                    **statistics([r['errors'][name] for r in chosen])))
    return reports


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    if not __debug__:raise ValueError('audit assertions required')
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive diagnostic attempt')
    preflight=hardware()
    assert preflight['memory_available_bytes']>=8*1024**3 and preflight['artifact_free_bytes']>=41*1024**3
    verify_artifacts(INPUT,{'result.json':PROBE_SHA});native=read_json(INPUT,'result.json')
    assert native['status']=='ACTIVE_VIEW_GOAL_PROBE_COMPLETE'
    bindings={'result.json':PROBE_SHA}|native['artifact_sha256'];verify_artifacts(INPUT,bindings)
    verify_artifacts(READOUT,{'result.json':READOUT_SHA});readout=read_json(READOUT,'result.json')
    assert readout['status']=='ACTIVE_VIEW_GOAL_READOUT_COMPLETE' and readout['probe_result_sha256']==PROBE_SHA
    readout_ids={'result.json':READOUT_SHA,'launch.json':readout['launch_sha256']}
    verify_artifacts(READOUT,readout_ids)
    fixed,receipt=admit(FIT_SHA)
    fit=read_json(FITS,'result.json');fit_ids={'result.json':FIT_SHA}|fit['artifact_sha256']
    models=[];snapshots={}
    for name in ROSTER:
        request=read_json(FITS,name+'_request.json');snapshot=read_json(FITS,name+'_fit.json')['snapshot']
        model=fixed if name==NAME else load_snapshot(FITS,snapshot['filename'],sha256=snapshot['sha256'],
            expected_binding=snapshot['binding'],expected_config=snapshot['configuration']).model
        models.append((name,model,request));snapshots[name]=snapshot
    original=read_json(READOUT,'launch.json')
    sources=discover_sources((PROTOCOL,'scripts/read_go2_active_view_six_model_transitions_v1.py',
        'lewm/tests/test_executed_half_second_diagnostic_development.py'),original['source_sha256'])
    launch=original|dict(source_sha256=sources,protocol=PROTOCOL,output_root=str(OUTPUT),
        native_input_artifact_sha256=bindings,readout_sha256=readout_ids,fit_artifact_sha256=fit_ids,
        snapshots=snapshots,all_six_admission=receipt,hardware=preflight,
        inference_workers_candidates=[1,2],intraop_threads=1,maximum_output_bytes=1024**3,
        minimum_free_bytes=40*1024**3,resource_envelope_os_enforced=False,
        native_execution=False,model_training=False,checkpoint_selection_performed=False)
    verify(launch);create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    print('SIX_MODEL_TRANSITIONS_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    started=time.perf_counter()
    try:
        population,histories=contexts();print('CAUSAL_CONTEXTS',len(population),flush=True)
        measurements=[];reference=None
        # Warm each model on the same two already recorded contexts, then compare
        # two repeated serial/concurrent units. These weights are never updated.
        execute(models,histories[:2],1)
        for width in (1,2,1,2):
            begin=time.perf_counter();values=execute(models,histories[:2],width)
            if reference is None:reference={n:v for n,v,_ in values}
            for n,v,_ in values:np.testing.assert_array_equal(v,reference[n])
            measurements.append(dict(workers=width,wall_s=time.perf_counter()-begin,
                model_wall_s={n:s for n,_,s in values}))
        width=min((1,2),key=lambda w:np.mean([m['wall_s'] for m in measurements if m['workers']==w]))
        write_json(OUTPUT/'workload.json',dict(measurements=measurements,selected_workers=width,
            exact_predictions_equal=True,contexts_per_model=2,models=6,hardware=hardware()))
        print('INFERENCE_WORKERS',width,flush=True)
        values=execute(models,histories,width);by_model={n:v for n,v,_ in values}
        for name,model,_ in models:
            assert state_digest(model.state_dict())==snapshots[name]['model_sha256']
            assert all(p.grad is None for p in model.parameters())
        for i,(row,history) in enumerate(zip(population,histories,strict=True)):
            assert state_digest(history)==row['history_sha256']
            np.testing.assert_array_equal(by_model[NAME][i],np.asarray(row.pop('recorded_prediction')))
        # Actual outcomes are first read after all six sets of predictions exist.
        raw_by_case={name:read_npz(INPUT/name,'physics_trace.npz') for name,_,_ in CASES}
        tapes={name:read_json(INPUT/name,'command_tape.json') for name,_,_ in CASES}
        records=[]
        for i,row in enumerate(population):
            case=row['case'];outcome=executed_outcome(raw_by_case[case],tapes[case],tick=row['tick'],action=row['action'])
            errors={};action_index=ACTIONS.index(row['action'])
            if outcome['eligible']:
                errors={name:error(by_model[name][i,action_index,0],outcome) for name in ROSTER}
                errors['zero_motion_reference']=dict(predicted_xy_m=[0.,0.],predicted_yaw_rad=0.,
                    predicted_contact_score=0.,xy_error_m=float(np.linalg.norm(outcome['xy_m'])),
                    yaw_error_rad=abs(outcome['yaw_rad']),contact_brier=float(outcome['contact']))
            records.append(row|dict(outcome=outcome,errors=errors,
                predictions={name:by_model[name][i].tolist() for name in ROSTER}))
        write_json(OUTPUT/'transitions.json',records)
        reports=summarize(records);write_json(OUTPUT/'scores.json',reports)
        verify(launch);verify_artifacts(INPUT,bindings);verify_artifacts(FITS,fit_ids);verify_artifacts(READOUT,readout_ids)
        artifacts={n:digest(OUTPUT/n) for n in ('launch.json','workload.json','transitions.json','scores.json')}
        verify_artifacts(OUTPUT,artifacts)
        assert sum((OUTPUT/n).stat().st_size for n in artifacts)<1024**3
        write_json(OUTPUT/'result.json',dict(status='ACTIVE_VIEW_SIX_MODEL_EXECUTED_TRANSITIONS_COMPLETE',
            source_sha256=sources,artifact_sha256=artifacts,probe_result_sha256=PROBE_SHA,fit_result_sha256=FIT_SHA,
            contexts=len(records),eligible_contexts=sum(r['outcome']['eligible'] for r in records),
            exclusions=[{k:r[k] for k in ('case','tick','action','outcome')} for r in records if not r['outcome']['eligible']],
            actual_action_counts=dict(Counter(r['action'] for r in records if r['outcome']['eligible'])),
            model_wall_s={n:s for n,_,s in values},wall_s=time.perf_counter()-started,hardware_after=hardware(),
            recorded_full_jepa_predictions_exactly_reconstructed=True,all_models_and_inputs_unchanged=True,
            counterfactual_outcomes_scored=False,checkpoint_selection_performed=False,model_training=False,
            native_execution=False,independent_maze_evaluation=False,navigation_qualified=False,goal_achieved=False))
        print('SIX_MODEL_TRANSITIONS_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as exc:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_SIX_MODEL_TRANSITION_DIAGNOSTIC_FAILURE',reason=repr(exc)))
        raise


if __name__=='__main__':main()
