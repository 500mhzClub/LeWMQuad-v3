"""Action-specific prediction in a common frozen visual target space."""
from collections import defaultdict
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from scripts import train_go2_anchored_visual_dynamics_development as training
from scripts import probe_go2_jepa_latent_branch_science_development as probe

OUTPUT=training.OUTPUT/'branch_evaluation'
RESULT=Path('docs/go2_anchored_visual_dynamics_result_2026-09-17.json')


def mse(value):return float(np.mean(np.square(np.asarray(value,dtype=np.float64))))


def score(rows,prediction,target,persistence,scene_target,scale,training_rows,training_targets):
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[row['cluster'],row['prefix_action']].append(i)
    curve=[]
    for h in range(8):
        p,t=prediction[:,h],target[:,h];cp=[];ct=[];wins=ties=0;wrong=[]
        for indices in groups.values():
            assert len(indices)==3
            pp,tt=p[indices].astype(float),t[indices].astype(float)
            cp.extend(pp-pp.mean(0));ct.extend(tt-tt.mean(0))
            matrix=((pp[:,None]-tt[None])**2).mean(-1)
            for j in range(3):
                other=[k for k in range(3) if k!=j]
                wins+=int(matrix[j,j]<matrix[j,other].min())
                ties+=int(matrix[j,j]==matrix[j,other].min())
                wrong.extend(matrix[other,j].tolist())
        error=mse(p-t);base=mse(persistence-t)
        zero=mse(ct);effect=mse(np.asarray(cp)-ct)
        mean=training_targets[:,h].mean(0)
        prefix=np.stack([training_targets[[j for j,r in enumerate(training_rows)
            if r['prefix_action']==row['prefix_action']],h].mean(0) for row in rows])
        scene_error=((p-scene_target[:,h])**2).mean(-1)
        correct=((p-t)**2).mean(-1)
        curve.append(dict(horizon_ms=100*(h+1),contexts=len(rows),prediction_mse=error,
            persistence_mse=base,prediction_to_persistence_ratio=error/base if base else None,
            normalized_prediction_mse=mse((p-t)/scale),normalized_persistence_mse=mse((persistence-t)/scale),
            action_retrieval_wins=wins,action_retrieval_ties=ties,wrong_action_prediction_mse=float(np.mean(wrong)),
            centered_action_prediction_mse=effect,action_independent_effect_mse=zero,
            centered_action_error_ratio=effect/zero if zero else None,
            scene_retrieval_wins=int((correct<scene_error).sum()),
            training_horizon_mean_mse=mse(mean-t),training_prefix_mean_mse=mse(prefix-t)))
    return dict(curve=curve,primary_800ms=curve[-1])


@torch.inference_mode()
def main():
    torch.set_num_threads(1);cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    assert json.loads((training.OUTPUT/'result.json').read_text())['status']=='complete'
    OUTPUT.mkdir(exist_ok=False);started=time.monotonic();rows=probe.selected_rows()
    model=training.representation.load();predictors={a:training.load(a) for a in training.ARMS}
    inputs=[]
    for row in rows:
        root=probe.data.PULSE/row['trial'];past=row['history_observation_indices']
        packets={i:probe.load_route_observation(root,i) for i in past}
        inputs.append(probe.inputs(row,probe.PacketReader(packets,past)))
    batch=probe.stack_samples(inputs);history=batch['observation_history']
    past=model.encoder({k:v.flatten(0,1) for k,v in history.items()}).reshape(len(rows),4,32)
    anchor=model.target({'rgb':history['rgb'][:,-1]})
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[row['cluster'],row['prefix_action']].append(i)
    for indices in groups.values():
        for key,tensor in history.items():
            assert all(torch.equal(tensor[indices[0]],tensor[i]) for i in indices[1:]),key
    args=(past,anchor,batch['known_action_blocks'],batch['known_action_valid'])
    predictions={a:p(*args).numpy() for a,p in predictors.items()}
    # The matched no-action model must ignore command values exactly.
    changed=batch['known_action_blocks'].flip(0)
    alternate=predictors['no_future_action'](past,anchor,changed,batch['known_action_valid']).numpy()
    np.testing.assert_array_equal(alternate,predictions['no_future_action'])
    predictions['persistence']=np.broadcast_to(anchor[:,None].numpy(),(len(rows),8,32)).copy()
    np.savez_compressed(OUTPUT/'causal_predictions.npz',**predictions)
    # Previously recorded future encodings are opened only after forecasts are saved.
    previous=training.representation.OUTPUT/'branch_evaluation'
    with np.load(previous/'encoded_targets.npz',allow_pickle=False) as a:
        target=a['visual_jepa_targets'].copy();scene=a['visual_jepa_scene_targets'].copy()
    with np.load(previous/'causal_predictions.npz',allow_pickle=False) as a:
        np.testing.assert_array_equal(anchor.numpy(),a['visual_jepa_persistence'])
        predictions['original_visual_jepa']=a['visual_jepa_predictions'].copy()
    train=[i for i,r in enumerate(rows) if r['data_role']=='train'];training_rows=[rows[i] for i in train]
    scale=predictors['action'].innovation_scale.numpy()
    scores={}
    for name,prediction in predictions.items():
        scores[name]={}
        for role in ('train','geometry_transfer'):
            idx=[i for i,r in enumerate(rows) if r['data_role']==role]
            scores[name][role]=score([rows[i] for i in idx],prediction[idx],target[idx],anchor.numpy()[idx],scene[idx],
                                    scale,training_rows,target[train])
    result=dict(status='complete',plan_sha256=probe.digest(training.PLAN),source_sha256=probe.digest(__file__),
        scores=scores,same_frozen_target_space_for_all_predictors=True,
        causal_forecasts_saved_before_future_targets_opened=True,no_action_forecast_ignores_commands_exactly=True,
        current_target_matches_preceding_assay_exactly=True,
        target_source=str(previous),target_sha256=probe.digest(previous/'encoded_targets.npz'),
        exposed_transfer_geometries=2,training_seed_count=1,branch_groups_per_role=6,
        navigation_executed=False,motion_readout_not_adapted=True,
        centered_action_metric_not_deployable=True,wall_s=time.monotonic()-started)
    probe.save(OUTPUT/'result.json',result);probe.save(RESULT,result)
    for name,s in scores.items():print(name,json.dumps(s['geometry_transfer']['primary_800ms']),flush=True)


if __name__=='__main__':main()
