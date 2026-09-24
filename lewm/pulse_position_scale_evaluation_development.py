"""Same exact targets/reducer, with explicit action/condition stratification."""
import numpy as np
import torch
from lewm.pulse_timed_dataset_development import stack_samples
from lewm.pulse_timed_training_runner_development import state_digest
from lewm.temporal_prediction_metrics_development import reduce_predictions


@torch.no_grad()
def evaluate(trainer,dataset,readers):
    indices=[i for i,w in enumerate(dataset.windows) if w['history_ready']]
    if not indices or any(dataset.episode_roles[dataset.windows[i]['condition']]['role']!='train' for i in indices):
        raise ValueError('explicit train-only diagnostic corpus required')
    predictions=[];targets=[];active=[];times=[]
    training=trainer.model.training;before=state_digest(trainer.model.state_dict());trainer.model.eval()
    try:
        for start in range(0,len(indices),6):
            batch=stack_samples([dataset.sample(i,readers) for i in indices[start:start+6]])
            output=trainer.model(**batch['inputs'])
            predictions.append({k:output[k].cpu().numpy() for k in ('direct_outcomes','rollout_outcomes')})
            targets.append({k:batch['targets'][k].cpu().numpy() for k in ('motion','motion_valid','contact','contact_valid')})
            active.append(output['prediction_valid'].cpu().numpy());times.append(output['target_offsets_ns'].cpu().numpy())
    finally:trainer.model.train(training)
    if state_digest(trainer.model.state_dict())!=before:raise ValueError('scoring mutated model')
    p={k:np.concatenate([x[k] for x in predictions]) for k in predictions[0]}
    y={k:np.concatenate([x[k] for x in targets]) for k in targets[0]}
    valid=np.concatenate(active);offsets=np.concatenate(times)
    metadata=[dataset.episode_roles[dataset.windows[i]['condition']] for i in indices]
    actions=np.array([dataset.windows[i]['action_index'] for i in indices])
    conditions=[dataset.windows[i]['condition'] for i in indices]
    def score(pred):
        return dict(all=reduce_predictions(pred,y,metadata,valid),
            by_actual_offset_ns={str(int(ns)):reduce_predictions(pred,y,metadata,valid,horizon_selection=offsets==ns)
                for ns in sorted(set(offsets[valid]))},
            by_action={str(a):reduce_predictions(pred,y,metadata,valid,row_selection=actions==a) for a in range(6)},
            by_condition={c:reduce_predictions(pred,y,metadata,valid,row_selection=np.array([x==c for x in conditions]))
                for c in sorted(set(conditions))})
    heads=('direct_outcomes',) if trainer.condition=='direct' else ('direct_outcomes','rollout_outcomes')
    zero=np.zeros_like(p['direct_outcomes']);zero[...,3]=1.;zero[...,4]=-30.
    report=dict(role='train',resubstitution=True,model_sha256=before,windows=len(indices),
        metrics={k:score(p[k]) for k in heads},zero_motion_no_contact=score(zero),
        checkpoint_selection_performed=False,navigation_qualified=False,independent_generalization_established=False)
    arrays={k:p[k] for k in heads} | y | dict(active=valid,offsets_ns=offsets,actions=actions,indices=np.array(indices))
    return report,arrays
