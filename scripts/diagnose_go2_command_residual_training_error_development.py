"""Measure fixed final predictions on all original training inputs, without updates."""
from collections import defaultdict
import hashlib
import json
import time
import cv2
import numpy as np
import torch
from lewm.pulse_timed_dataset_development import stack_samples
from lewm.route_rgb_dataset_development import load_route_observation
from scripts import pre_switch_training_data_development as source
from scripts import train_go2_command_history_residual_development as train
from scripts import command_history_residual_snapshot_development as snapshot

OUTPUT=train.OUTPUT/'training_prediction_diagnosis'
NAMES=('reference_only','jepa','supervised_rollout')


def metrics(rows):
    result={}
    for h in (2,6,7):
        valid=[r for r in rows if r['errors'][NAMES[0]][h] is not None]
        if not valid:continue
        weights=np.asarray([r['draw_weight'] for r in valid]);models={}
        for name in NAMES:
            error=np.asarray([r['errors'][name][h] for r in valid])
            models[name]=dict(xy_rmse_mm=1000*float(np.sqrt(np.average(np.sum(error[:,:2]**2,axis=1),weights=weights))),
                mean_signed_xy_mm=(1000*np.average(error[:,:2],axis=0,weights=weights)).tolist(),
                yaw_rmse_deg=float(np.degrees(np.sqrt(np.average(error[:,2]**2,weights=weights)))))
        result[str((h+1)*100)]=dict(contexts=len(valid),weighted_draws=int(weights.sum()),models=models)
    return result


@torch.inference_mode()
def main():
    if OUTPUT.exists():raise ValueError('preserve training diagnosis')
    torch.set_num_threads(1);cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    began=time.monotonic();schedule=json.loads(train.SCHEDULE.read_text());rows=train.data.load_training_rows()
    assert len(rows)==4694 and all(r['data_role']=='train' for r in rows)
    models={}
    for name in train.CONDITIONS:
        directory=train.OUTPUT/name;record=json.loads((directory/'fit.json').read_text())
        models[name]=snapshot.load_snapshot(directory,record['filename'],sha256=record['sha256'],
            expected_binding=record['binding'],expected_config=record['configuration']).model
    identities=json.loads((train.OUTPUT/'jepa/consumed_policy_sha256.json').read_text())
    groups=defaultdict(list)
    for row in rows:groups[row['source'],row['trial']].append(row)
    OUTPUT.mkdir();scores=[]
    for ordinal,((family,trial),selected) in enumerate(sorted(groups.items()),1):
        directory=train.data.ROOTS[family]/trial
        past={i for row in selected for i in row['history_observation_indices']}
        for filename in ['policy_observations.json','policy_histories.npz']+[f'rgb_{i:04d}.png' for i in sorted(past)]:
            path=directory/filename
            assert hashlib.sha256(path.read_bytes()).hexdigest()==identities[str(path.relative_to(train.data.BASE))]
        packets={i:load_route_observation(directory,i) for i in sorted(past)}
        for start in range(0,len(selected),6):
            chunk=selected[start:start+6]
            inputs=stack_samples([source.inputs(r,source.PacketReader(packets,r['history_observation_indices'])) for r in chunk])
            predictions={name:model(**inputs)['rollout_outcomes'].numpy() for name,model in models.items()}
            reference=models['jepa'].reference_motion(inputs['observation_history'],inputs['known_action_blocks'],inputs['known_action_valid']).numpy()
            motions={name:np.concatenate((p[:,:,:2],np.arctan2(p[:,:,2:3],p[:,:,3:4])),axis=-1) for name,p in predictions.items()}
            motions['reference_only']=reference
            for i,row in enumerate(chunk):
                errors={name:[] for name in NAMES}
                for h,target in enumerate(row['targets']):
                    for name in NAMES:
                        if not target['motion_valid']:errors[name].append(None);continue
                        delta=motions[name][i,h]-np.asarray(target['motion'])
                        delta[2]=np.arctan2(np.sin(delta[2]),np.cos(delta[2]))
                        assert np.isfinite(delta).all();errors[name].append(delta.tolist())
                scores.append(dict(sample_id=row['sample_id'],source=family,trial=trial,action=row['action'],
                    draw_weight=schedule['context_draw_counts'][row['sample_id']],errors=errors))
        if ordinal%20==0:print('TRAINING_PREDICTION_DIAGNOSIS',len(scores),'of',len(rows),flush=True)
    assert len(scores)==4694 and sum(r['draw_weight'] for r in scores)==7200
    train.write(OUTPUT/'rows.json',scores)
    result=dict(schema='command_residual_training_error_diagnosis.v1',contexts=len(scores),draws=7200,
        source_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest(),
        total=metrics(scores),by_source={s:metrics([r for r in scores if r['source']==s]) for s in sorted({r['source'] for r in scores})},
        by_action={a:metrics([r for r in scores if r['action']==a]) for a in sorted({r['action'] for r in scores})},
        uses_original_sampling_weights=True,all_original_training_contexts_included=True,
        inference_uses_only_original_causal_inputs=True,target_images_not_loaded=True,
        training_labels_evaluator_only=True,training_updates=0,model_weights_unchanged=True,
        generalization_evidence=False,wall_s=time.monotonic()-began)
    train.write(OUTPUT/'result.json',result);print(json.dumps(result['total'],indent=2),flush=True)


if __name__=='__main__':main()
