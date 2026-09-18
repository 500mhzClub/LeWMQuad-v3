"""All native horizons on retained branches; no fitting or native simulation."""
from collections import defaultdict
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from lewm_genesis.lewm_contract import SafetyLimits, apply_safety_limits_single
from lewm.dense_visual_motion_readout_development import pool_tokens
from lewm.dense_native_observation_development import dense_native_context
from lewm.route_rgb_dataset_development import load_route_observation
from lewm.terminal_translation_pulse_development import command_sequences
from scripts import train_go2_horizon_dense_predictor_development as fit
from scripts import train_go2_dense_visual_motion_readout_development as motion_fit
from scripts import probe_go2_jepa_latent_branch_science_development as retained
from scripts.evaluate_go2_dense_visual_motion_readout_development import metrics as motion_metrics

reference=fit.parent.reference
OUTPUT=fit.OUTPUT/'branch_evaluation'
RESULT=Path('docs/go2_horizon_dense_predictor_evaluation_2026-09-18.json')


def interaction_metrics(predicted,truth):
    """Remove scene-only and action-only main effects; score their interaction."""
    if predicted.shape!=truth.shape or predicted.shape[:2]!=(2,3):
        raise ValueError('matched two-scene by three-action tensors required')
    def centered(value):
        value=value.double()
        return value-value.mean(0,keepdim=True)-value.mean(1,keepdim=True)+value.mean((0,1),keepdim=True)
    p,t=centered(predicted),centered(truth)
    energy=float(t.square().mean());error=float((p-t).square().mean())
    return dict(target_interaction_mse=energy,predicted_interaction_mse=float(p.square().mean()),
        interaction_error_mse=error,error_over_zero_interaction=error/energy if energy>1e-12 else None)


def scene_action_interactions(rows,controls,actions,forecasts,targets,parent_predictions):
    panels=defaultdict(list)
    for i,row in enumerate(rows):panels[row['data_role'],row['prefix_action']].append(i)
    records=[]
    for (role,prefix),indices in panels.items():
        clusters=sorted({rows[i]['cluster'] for i in indices})
        names=sorted({rows[i]['pulse_action'] for i in indices})
        assert len(clusters)==2 and len(names)==3
        grid=[[next(i for i in indices if rows[i]['cluster']==s and rows[i]['pulse_action']==a) for a in names] for s in clusters]
        assert torch.equal(actions[grid[0]],actions[grid[1]]) and torch.equal(controls[grid[0]],controls[grid[1]])
        for h in range(1,9):
            truth=torch.stack([targets[g,h-1] for g in grid])
            models={a:interaction_metrics(torch.stack([v[g,h-1] for g in grid]),truth) for a,v in forecasts.items()}
            if h==5:models['parent_500ms']=interaction_metrics(torch.stack([parent_predictions[g] for g in grid]),truth)
            records.append(dict(role=role,prefix_action=prefix,clusters=clusters,actions=names,horizon_ms=h*100,models=models))
    return records


@torch.inference_mode()
def navigation_inference_timing(encoder,model,head,row,prepared,limits,mean,std):
    """Measure a six-candidate computation; no commands execute and no targets load."""
    directory,_,_=prepared
    packets=[load_route_observation(directory,i) for i in (3,8,13)]
    now=packets[-1]['image']['measured_ns']
    last=packets[-1]['sensor_state']['control']['applied_command']['values'][-1]
    requested=command_sequences(row['known_commands'][:3],pulse=False)
    applied=np.asarray([apply_safety_limits_single(r.tolist(),tuple(last),limits)[0] for r in requested],np.float32)
    actions=torch.from_numpy(applied[:,:,[0,2]]).cuda()
    mask=torch.ones(6,768,dtype=torch.bool,device='cuda')
    timings=[]
    for repetition in range(3):
        torch.cuda.synchronize();started=time.perf_counter()
        observation=dense_native_context(packets,observed_ns=now)
        control=((observation['past_applied_commands'][:,[0,2]].reshape(3,5,2)-torch.from_numpy(mean))/torch.from_numpy(std)).cuda()[None]
        pixels=observation['pixels'].cuda()
        context=F.layer_norm(encoder.tokens(pixels).float(),(1024,))[None]
        torch.cuda.synchronize();encoded=time.perf_counter()
        current=pool_tokens(context[:,-1]);outputs=[];prefix_counts=[]
        for h in range(1,9):
            unique,inverse=torch.unique(actions[:,:h].reshape(6,-1),dim=0,return_inverse=True)
            indices=torch.stack([(inverse==j).nonzero()[0,0] for j in range(len(unique))])
            n=len(unique);prefix_counts.append(n)
            pred=model(context.expand(n,-1,-1,-1),actions[indices],
                torch.full((n,),h,dtype=torch.long,device='cuda'),mask[:n],control=control.expand(n,-1,-1,-1))
            pred=F.layer_norm(pred.float(),(1024,))
            outputs.append(head(current.expand(n,-1,-1),pool_tokens(pred))[inverse])
        motion=torch.stack(outputs,dim=1).cpu()
        assert motion.shape==(6,8,3) and torch.isfinite(motion).all()
        torch.cuda.synchronize();finished=time.perf_counter()
        timings.append(dict(repetition=repetition,encoder_context_ms=(encoded-started)*1000,
            prediction_and_readout_ms=(finished-encoded)*1000,total_ms=(finished-started)*1000,
            distinct_applied_prefixes=prefix_counts))
    return dict(repetitions=timings,first_repetition_is_warmup=True,
        median_warm_total_ms=float(np.median([r['total_ms'] for r in timings[1:]])),
        old_dispatch_budget_ms=300,candidate_actions=6,horizons=8,context_images=3,
        precision='float32 without autocast',context_feature_cache_reused=False,
        includes='native array preprocessing, three-image encoder batch, per-horizon unique-prefix prediction, frozen motion decoding and CPU output copy',
        excludes='acquisition, tracking, mapping, routing, clearance, scheduling and physical execution',
        navigation_executed=False,real_time_qualified=False)


@torch.inference_mode()
def main():
    assert not OUTPUT.exists() and not RESULT.exists()
    terminal=json.loads(fit.RESULT.read_text());assert terminal['status']=='COMPLETE'
    OUTPUT.mkdir();start=time.monotonic();torch.set_num_threads(4)
    rows=reference.selected_rows();prepared=[reference.inputs(r) for r in rows]
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[row['data_role'],row['cluster'],row['prefix_action']].append(i)
    assert len(groups)==12 and all(len(v)==3 for v in groups.values())
    stats=json.loads((reference.CACHE/'proprio_v1/proprio_norm_stats.json').read_text())
    mean,std=(np.asarray(stats[k],np.float32) for k in ('control_mean','control_std'))
    limits=SafetyLimits.from_manifest(yaml.safe_load(Path('config/go2_platform_manifest.yaml').read_text()))
    actions=[]
    for row,(directory,control,_) in zip(rows,prepared,strict=True):
        assert len(row['known_commands'])>=8 and all(t['future_image_valid'] for t in row['targets'][:8])
        spec=json.loads((directory/'branch_specification.json').read_text())
        np.testing.assert_allclose(row['known_commands'][:8],spec['prospective_commands'][13:21],rtol=0,atol=1e-7)
        last=control[-1,-1]
        applied=np.asarray(apply_safety_limits_single(row['known_commands'][:8],(float(last[0]),0.,float(last[1])),limits)[0],np.float32)
        assert (applied[:,1]==0).all();actions.append(applied[:,[0,2]])
    action=torch.from_numpy(np.stack(actions));control=torch.from_numpy(np.stack([(c-mean)/std for _,c,_ in prepared]))
    fit.save(OUTPUT/'plan.json',dict(source_sha256=fit.digest(__file__),fit_result_sha256=fit.digest(fit.RESULT),
        windows_sha256=fit.digest(reference.data.PULSE/'windows.json'),horizons_ms=list(range(100,801,100)),
        primary_role='geometry_transfer',training_role_diagnostic_only=True,contexts_per_role=18,
        precision='float32 encoder and predictor, no autocast',blind_forecast='one forecast per common context and horizon, expanded across branches',
        equal_action_prefixes='Forecast and decode each distinct applied prefix once, then broadcast; identical pre-branch actions cannot acquire numerical preferences.',
        comparison='matched horizon action/blind predictors plus persistence; retained supplemented parent at its supported 500-ms horizon',
        additional_diagnostic='Scene-by-action double centering removes static scene and scene-independent action effects; visual interaction is not physical collision prediction.',
        motion_probe='Same frozen 500-ms-trained head at every horizon, including observed-future oracle; non-500-ms scores are temporal transfer, not newly fitted probes.',
        motion_readout_sha256=fit.digest(motion_fit.OUTPUT/'readout.pt'),
        future_rgb_loaded_after_all_forecasts=True,new_navigation=False))
    encoder=reference.encoders.VJepa21Arm();encoder.build(torch.device('cuda:0'),torch.float32)
    cache={}

    def encode(path):
        key=fit.digest(path)
        if key not in cache:
            pixel=encoder.preprocess(str(path))[None].cuda()
            value=F.layer_norm(encoder.tokens(pixel).float(),(1024,))[0]
            assert value.shape==(768,1024) and torch.isfinite(value).all()
            cache[key]=value.cpu()
        return cache[key]

    contexts=torch.stack([torch.stack([encode(d/f'rgb_{f:04d}.png') for f in (3,8,13)]) for d,_,_ in prepared])
    models={a:fit.load(a).cuda() for a in fit.ARMS}
    parent=fit.previous.load('mixed_action').cuda()
    head=motion_fit.load().cuda()
    forecasts={a:torch.empty(len(rows),8,768,1024) for a in models}
    motions={a:np.empty((len(rows),8,3)) for a in (*models,'persistence','observed_future','zero_motion')}
    motions['zero_motion'].fill(0)
    parent_motion=np.empty((len(rows),3))
    parent_predictions=torch.empty(len(rows),768,1024)
    for key,indices in groups.items():
        for i in indices[1:]:
            assert torch.equal(contexts[i],contexts[indices[0]]) and torch.equal(control[i],control[indices[0]])
        x=contexts[indices].cuda();c=control[indices].cuda();a=action[indices].cuda()
        mask=torch.ones(3,768,dtype=torch.bool,device='cuda')
        current=pool_tokens(x[:,-1])
        persistent=head(current[:1],current[:1]).expand(3,-1).cpu().numpy()
        for h in range(1,9):
            unique,inverse=torch.unique(a[:,:h].reshape(3,-1),dim=0,return_inverse=True)
            representatives=torch.stack([(inverse==j).nonzero()[0,0] for j in range(len(unique))])
            for arm,model in models.items():
                blind=arm=='no_future_action'
                n=1 if blind else len(unique)
                chosen=a[:1] if blind else a[representatives]
                expansion=torch.zeros(3,dtype=torch.long,device='cuda') if blind else inverse
                value=model(x[:n],chosen,torch.full((n,),h,device='cuda',dtype=torch.long),mask[:n],control=c[:n])
                value=F.layer_norm(value.float(),(1024,));assert torch.isfinite(value).all()
                forecasts[arm][indices,h-1]=value[expansion].cpu()
                motions[arm][indices,h-1]=head(current[:n],pool_tokens(value))[expansion].cpu().numpy()
            motions['persistence'][indices,h-1]=persistent
        value=parent(x,a[:,:5].reshape(3,10),mask,control=c)
        value=F.layer_norm(value.float(),(1024,))
        parent_predictions[indices]=value.cpu();parent_motion[indices]=head(current,pool_tokens(value)).cpu().numpy()
        print('HORIZON_FORECASTS',key,flush=True)
    fit.save(OUTPUT/'forecasts_complete.json',dict(future_rgb_loaded=False,contexts=len(rows),seconds=time.monotonic()-start))
    timing=navigation_inference_timing(encoder,models['action'],head,rows[0],prepared[0],limits,mean,std)
    fit.save(OUTPUT/'navigation_inference_timing.json',timing)
    del models,parent;torch.cuda.empty_cache()
    old_plan=json.loads(retained.PLAN.read_text());old_result=json.loads(retained.RESULT.read_text())
    assert old_plan['sample_ids']==[r['sample_id'] for r in rows]
    baseline_path=retained.OUTPUT/'causal_predictions.npz'
    assert fit.digest(baseline_path)==old_result['prediction_sha256']
    with np.load(baseline_path,allow_pickle=False) as archive:motions['command_history']=archive['jepa_reference'].copy()
    assert motions['command_history'].shape==(len(rows),8,3)
    targets=torch.empty(len(rows),8,768,1024)
    for i,(directory,_,_) in enumerate(prepared):
        meta=json.loads((directory/'policy_observations.json').read_text())
        with np.load(directory/'policy_histories.npz',allow_pickle=False) as archive:
            for h in range(1,9):
                assert rows[i]['targets'][h-1]['future_observation_index']==13+h
                assert meta['frames'][13+h]['image_ns']-meta['frames'][13]['image_ns']==h*100_000_000
                np.testing.assert_allclose(archive['applied_command_values'][13+h][-h:][:,[0,2]],actions[i][:h],rtol=0,atol=1e-6)
                targets[i,h-1]=encode(directory/f'rgb_{13+h:04d}.png')
                motions['observed_future'][i,h-1]=head(pool_tokens(contexts[i,-1:,:,:].cuda()),pool_tokens(targets[i,h-1:h].cuda()))[0].cpu().numpy()
    assert all(np.isfinite(v).all() for v in motions.values())
    records=[]
    for key,indices in groups.items():
        for h in range(1,9):
            truth=targets[indices,h-1]
            predictions={a:v[indices,h-1] for a,v in forecasts.items()}
            predictions['persistence']=contexts[indices,-1]
            if h==5:predictions['parent_500ms']=parent_predictions[indices]
            result={}
            for arm,pred in predictions.items():
                distances=(pred[:,None]-truth[None]).square().mean((-1,-2))
                wins=[bool(distances[j,j]<torch.cat((distances[:j,j],distances[j+1:,j])).min()) for j in range(3)]
                centered=((pred-pred.mean(0))-(truth-truth.mean(0))).square().mean()
                baseline=(truth-truth.mean(0)).square().mean()
                result[arm]=dict(factual_mse=distances.diag().tolist(),mse_matrix=distances.tolist(),
                    correct_action_beats_both_wrong=wins,centered_effect_mse=float(centered),centered_zero_mse=float(baseline))
                if arm in ('no_future_action','persistence'):
                    assert torch.equal(pred[0],pred[1]) and torch.equal(pred[1],pred[2]) and not any(wins)
            records.append(dict(role=key[0],cluster=key[1],prefix_action=key[2],horizon_ms=h*100,
                distinct_applied_prefixes=len(torch.unique(action[indices,:h].reshape(3,-1),dim=0)),models=result))
    summaries={}
    for role in ('train','geometry_transfer'):
        summaries[role]={}
        for h in range(1,9):
            chosen=[r for r in records if r['role']==role and r['horizon_ms']==h*100]
            by_arm={}
            for arm in chosen[0]['models']:
                values=[r['models'][arm] for r in chosen]
                mse=float(np.mean([v['factual_mse'] for v in values]))
                base=float(np.mean([r['models']['persistence']['factual_mse'] for r in chosen]))
                by_arm[arm]=dict(contexts=3*len(chosen),mse=mse,mse_over_persistence=mse/base,
                    correct_action_wins=sum(sum(v['correct_action_beats_both_wrong']) for v in values),
                    centered_effect_ratio=float(np.mean([v['centered_effect_mse'] for v in values])/np.mean([v['centered_zero_mse'] for v in values])))
            summaries[role][str(h*100)]=by_arm
    interaction=scene_action_interactions(rows,control,action,forecasts,targets,parent_predictions)
    physical=np.asarray([[r['targets'][h]['motion'] for h in range(8)] for r in rows])
    assert all(t['motion_valid'] for r in rows for t in r['targets'][:8])
    motion_summaries={}
    for role in ('train','geometry_transfer'):
        indices=[i for i,r in enumerate(rows) if r['data_role']==role];motion_summaries[role]={}
        for h in range(8):
            values={a:motion_metrics(v[indices,h],physical[indices,h]) for a,v in motions.items()}
            if h==4:values['parent_500ms']=motion_metrics(parent_motion[indices],physical[indices,h])
            motion_summaries[role][str((h+1)*100)]=values
    report=dict(status='COMPLETE',summaries=summaries,groups=records,scene_action_interactions=interaction,
        motion_summaries=motion_summaries,motion_predictions={a:v.tolist() for a,v in motions.items()},
        parent_500ms_motion=parent_motion.tolist(),physical_targets=physical.tolist(),trials=[r['trial'] for r in rows],
        command_reference_sha256=old_result['prediction_sha256'],source_sha256=fit.digest(__file__),
        navigation_inference_timing=timing,
        fit_result_sha256=fit.digest(fit.RESULT),wall_s=time.monotonic()-start,unique_encoded_frames=len(cache),
        new_navigation=False,no_fitting=True,dense_cache_retained=False,
        limitations=['retained exposed development branches','one training seed','no independent maze test',
                     'no JEPA encoder-objective isolation','visual forecast quality does not establish physical motion or collision accuracy',
                     'motion readout fitted only at 500 ms; other horizons are temporal transfer',
                     'command baseline used its original larger training population and four command histories'])
    fit.save(RESULT,report);fit.save(OUTPUT/'result.json',report)
    print('HORIZON_EVALUATION_COMPLETE',json.dumps(summaries['geometry_transfer']),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUTPUT.exists() and not (OUTPUT/'failure.json').exists():
            fit.save(OUTPUT/'failure.json',dict(reason=repr(error),traceback=traceback.format_exc()))
        raise
