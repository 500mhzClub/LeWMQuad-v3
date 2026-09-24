"""Fit a fixed motion layer on frozen JEPA, supervised and random features."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from lewm.command_history_residual_learning_development import CommandHistoryResidualTrainer
from lewm.frozen_motion_readout_development import features,fit_readout,predict_readout,install_readout
from lewm.pulse_timed_dataset_development import stack_samples
from lewm.pulse_timed_training_runner_development import state_digest
from lewm.eligible_floor_registration_development import bind
from scripts import pre_switch_training_data_development as source
from scripts import train_go2_command_history_residual_development as training
from scripts import run_go2_command_residual_lower_rate_development as previous
from scripts import command_history_residual_snapshot_development as snapshot
from scripts import diagnose_go2_command_residual_training_error_development as diagnostic

BASE=training.BASE
OUTPUT=BASE/'go2_frozen_motion_readout_v1_attempt_001'
PLAN=Path('docs/go2_frozen_motion_readout_plan_2026-09-17.json')
ARMS=('jepa','supervised_rollout','untrained')
NAMES=('reference_only',)+ARMS
write=training.write
metrics=bind(diagnostic.metrics,NAMES=NAMES)


def prepare():
    if OUTPUT.exists():raise ValueError('preserve every attempt')
    predecessor=json.loads(previous.PLAN.read_text())
    records={c:json.loads((previous.OUTPUT/c/'fit.json').read_text()) for c in ARMS[:2]}
    plan=dict(schema='frozen_motion_readout_plan.v1',feature_sources=records,
        source_directory=str(previous.OUTPUT),seed=predecessor['seed'],latent_dim=32,
        feature_arms=ARMS,untrained_reference_is_common_initialization=True,
        training_contexts=4694,scheduled_draws=7200,schedule_path=str(training.SCHEDULE),
        schedule_sha256=predecessor['schedule_sha256'],reference_sha256=predecessor['reference_sha256'],
        evaluation_roots=predecessor['evaluation_roots'],ridge_penalty=1.,
        fit_features='256 frozen activations immediately before the final rollout linear layer',
        fit_targets='XY residual and sin/cos of yaw residual relative to the frozen command-history fit',
        weights='original context draw count divided by its number of valid motion horizons',
        feature_standardization='weighted training means and standard deviations; scale below 1e-8 replaced by 1',
        fit_objective='weighted squared residual error plus fixed ridge penalty; unpenalized intercept',
        change='replace only four outputs of final rollout motion layer; no encoder, transition or hidden-layer update',
        inference_standardization='explicit float64 standardization and linear fit, cast motion outputs to model float32',
        contact_head_unchanged=True,contact_scores_not_evaluated_or_used_for_this_motion_comparison=True,
        training_only_labels=True,new_navigation_labels_in_training=False,hyperparameter_search=False,
        original_heads_and_reference_remain_comparators=True,
        primary='same 3726 executed windows in six exposed recordings; per-run/action prefix, increment, whole XY and yaw error',
        native_navigation_started=False,independent_generalization_claim=False,
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
            'scripts/fit_go2_frozen_motion_readout_development.py','lewm/frozen_motion_readout_development.py',
            'lewm/command_history_residual_learning_development.py','scripts/pre_switch_training_data_development.py')})
    write(PLAN,plan);OUTPUT.mkdir();print('PREPARED frozen JEPA/supervised/untrained readout comparison',flush=True)


def load_parent(name,plan):
    if name=='untrained':
        trainer=CommandHistoryResidualTrainer('jepa',seed=plan['seed'],latent_dim=plan['latent_dim'],learning_rate=1e-4)
        if trainer.initial_sha256!=plan['feature_sources']['jepa']['initial_sha256']:raise ValueError('same random initialization required')
        return trainer.model.eval()
    record=plan['feature_sources'][name]
    return snapshot.load_snapshot(Path(plan['source_directory'])/name,record['filename'],sha256=record['sha256'],
        expected_binding=record['binding'],expected_config=record['configuration']).model


def load_readout(name):
    plan=json.loads(PLAN.read_text());record=json.loads((OUTPUT/name/'fit.json').read_text())
    model=load_parent(name,plan);path=OUTPUT/name/'readout.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=record['readout_sha256']:raise ValueError('readout changed')
    if state_digest(model.state_dict())!=record['parent_model_sha256']:raise ValueError('parent model changed')
    with np.load(path,allow_pickle=False) as archive:fit={k:archive[k].copy() for k in archive.files}
    install_readout(model,fit)
    if state_digest(model.state_dict())!=record['model_sha256']:raise ValueError('installed readout state changed')
    return model


@torch.inference_mode()
def fit():
    torch.set_num_threads(1);cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    plan=json.loads(PLAN.read_text());start=time.monotonic()
    for p,sha in plan['source_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha
    assert hashlib.sha256(training.SCHEDULE.read_bytes()).hexdigest()==plan['schedule_sha256']
    if (OUTPUT/'feature_extraction_started.json').exists():raise ValueError('preserve partial or complete attempt')
    write(OUTPUT/'feature_extraction_started.json',dict(feature_arms=ARMS))
    rows=training.data.load_training_rows();schedule=json.loads(training.SCHEDULE.read_text())
    assert len(rows)==4694 and all(r['data_role']=='train' for r in rows)
    models={name:load_parent(name,plan) for name in ARMS};parent_hash={name:state_digest(m.state_dict()) for name,m in models.items()}
    identities=json.loads((previous.OUTPUT/'jepa/consumed_policy_sha256.json').read_text())
    groups=defaultdict(list)
    for row in rows:groups[row['source'],row['trial']].append(row)
    xs={name:[] for name in ARMS};references=[];truths=[];masks=[];metadata=[];examples=None
    for ordinal,((family,trial),selected) in enumerate(sorted(groups.items()),1):
        directory=training.data.ROOTS[family]/trial;past={i for r in selected for i in r['history_observation_indices']}
        for filename in ['policy_observations.json','policy_histories.npz']+[f'rgb_{i:04d}.png' for i in sorted(past)]:
            path=directory/filename;assert hashlib.sha256(path.read_bytes()).hexdigest()==identities[str(path.relative_to(training.data.BASE))]
        packets={i:source.load_route_observation(directory,i) for i in sorted(past)}
        for begin in range(0,len(selected),6):
            chunk=selected[begin:begin+6]
            inputs=stack_samples([source.inputs(r,source.PacketReader(packets,r['history_observation_indices'])) for r in chunk])
            reference=models['jepa'].reference_motion(inputs['observation_history'],inputs['known_action_blocks'],inputs['known_action_valid']).numpy()
            reference_copy=models['supervised_rollout'].reference_motion(inputs['observation_history'],inputs['known_action_blocks'],inputs['known_action_valid']).numpy()
            np.testing.assert_array_equal(reference,reference_copy)
            for name,model in models.items():xs[name].append(features(model,inputs).numpy())
            references.extend(reference)
            if examples is None:examples=inputs
            for r in chunk:
                masks.append([bool(t['motion_valid']) for t in r['targets']])
                truths.append([t['motion'] if t['motion_valid'] else [0.,0.,0.] for t in r['targets']])
                metadata.append(dict(sample_id=r['sample_id'],source=family,trial=trial,action=r['action'],
                    draw_weight=schedule['context_draw_counts'][r['sample_id']]))
        if ordinal%20==0:print('FROZEN_FEATURES',len(metadata),'of',len(rows),flush=True)
    reference=np.asarray(references);truth=np.asarray(truths);mask=np.asarray(masks,dtype=bool)
    weights=np.asarray([r['draw_weight'] for r in metadata],dtype=float)
    weights=np.broadcast_to((weights/np.maximum(mask.sum(1),1))[:,None],mask.shape)
    delta=truth-reference
    target=np.concatenate((delta[:,:,:2],np.sin(delta[:,:,2:3]),np.cos(delta[:,:,2:3])),axis=-1)
    predictions=dict(reference_only=reference);records={}
    for name,model in models.items():
        x=np.concatenate(xs[name]);fitted=fit_readout(x[mask],target[mask],weights[mask],penalty=plan['ridge_penalty'])
        assert state_digest(model.state_dict())==parent_hash[name]
        before={k:v.clone() for k,v in model.state_dict().items()};install_readout(model,fitted)
        for key,value in before.items():
            after_key=key.replace('rollout_decode.2.','rollout_decode.2.original.')
            assert torch.equal(value,model.state_dict()[after_key])
        # Verify the fitted numerical head and its actual float32 installed form.
        installed=model.rollout_decode[-1](torch.as_tensor(x)).numpy()[:,:,:4]
        expected=predict_readout(x,fitted)
        max_difference=float(np.max(np.abs(installed[mask]-expected[mask])))
        np.testing.assert_allclose(installed[mask],expected[mask],rtol=0,atol=2e-6)
        absolute=np.concatenate((installed[:,:,:2]+reference[:,:,:2],
            (np.arctan2(installed[:,:,2],installed[:,:,3])+reference[:,:,2])[:,:,None]),axis=-1)
        predictions[name]=absolute;directory=OUTPUT/name;directory.mkdir()
        np.savez_compressed(directory/'readout.npz',**fitted)
        records[name]=dict(feature_source=name,parent_model_sha256=parent_hash[name],
            model_sha256=state_digest(model.state_dict()),readout_sha256=hashlib.sha256((directory/'readout.npz').read_bytes()).hexdigest(),
            feature_dim=x.shape[-1],training_contexts=len(metadata),motion_rows=int(mask.sum()),
            contexts_without_motion_labels=int((~mask.any(1)).sum()),fit_weight_sum=float(weights[mask].sum()),
            only_four_final_motion_outputs_changed=True,maximum_float32_install_difference=max_difference,
            optimizer_updates=0,ridge_penalty=plan['ridge_penalty'])
        write(directory/'fit.json',records[name]);clone=load_readout(name)
        torch.testing.assert_close(clone(**examples)['rollout_outcomes'],model(**examples)['rollout_outcomes'],rtol=0,atol=0)
        print('READOUT_FIT_COMPLETE',name,flush=True)
    scored=[]
    for i,row in enumerate(metadata):
        errors={name:[] for name in NAMES}
        for h in range(8):
            for name in NAMES:
                if not mask[i,h]:errors[name].append(None);continue
                error=predictions[name][i,h]-truth[i,h];error[2]=np.arctan2(np.sin(error[2]),np.cos(error[2]))
                errors[name].append(error.tolist())
        scored.append(row|dict(errors=errors))
    assert len(scored)==4694 and sum(r['draw_weight'] for r in scored)==7200
    write(OUTPUT/'training_rows.json',scored)
    result=dict(schema='frozen_motion_readout_fit.v1',status='COMPLETE',records=records,total=metrics(scored),
        by_source={s:metrics([r for r in scored if r['source']==s]) for s in sorted({r['source'] for r in scored})},
        fit_before_navigation_evaluation=True,feature_models_unchanged=True,all_training_contexts_retained=True,
        feature_arrays_not_persisted=True,wall_s=time.monotonic()-start)
    write(OUTPUT/'result.json',result);print(json.dumps(result['total'],indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    else:fit()
