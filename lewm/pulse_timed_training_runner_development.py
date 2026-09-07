"""CPU matched pulse training and resubstitution diagnostics, not navigation.

No implicit split, best-checkpoint selection, directory discovery or resume.
The caller binds the full dataset, schedule and prospective experiment limits.
"""
from copy import deepcopy
import hashlib
import math
import numpy as np
import torch
from lewm.pulse_timed_rgb_body_jepa_development import PulseTimedRGBBodyJEPA
from lewm.pulse_timed_learning_development import training_loss,active_parameters,CONDITIONS
from lewm.pulse_timed_dataset_development import stack_samples
from lewm.temporal_prediction_metrics_development import reduce_predictions


def state_digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        a=value.detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


class PulseTrainer:
    def __init__(self,condition,*,seed,latent_dim=32,learning_rate=1e-3,ema_momentum=.99):
        if condition not in CONDITIONS or type(seed) is not int or seed<0:
            raise ValueError('explicit condition and nonnegative seed required')
        if type(latent_dim) is not int or not 8<=latent_dim<=128:
            raise ValueError('bounded latent width required')
        if not math.isfinite(learning_rate) or not 0<learning_rate<=.01 or not 0<=ema_momentum<=1:
            raise ValueError('bounded learning rate and EMA required')
        self.condition=condition;self.seed=seed;self.latent_dim=latent_dim
        self.learning_rate=learning_rate;self.ema_momentum=ema_momentum
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);self.model=PulseTimedRGBBodyJEPA(latent_dim).cpu()
        self.initial_sha256=state_digest(self.model.state_dict())
        self.parameters=active_parameters(self.model,condition)
        self.optimizer=torch.optim.AdamW(self.parameters,lr=learning_rate,weight_decay=0.)
        self.updates=0;self.failed=False

    def step(self,batch):
        if self.failed:raise ValueError('training failure latched; no implicit retry')
        try:
            self.model.train();self.optimizer.zero_grad(set_to_none=True)
            loss,parts=training_loss(self.model,batch,self.condition);loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.parameters):
                raise ValueError('finite gradients required for every active parameter')
            norm=torch.nn.utils.clip_grad_norm_(self.parameters,1.,error_if_nonfinite=True)
            self.optimizer.step()
            if not all(torch.isfinite(p).all() for p in self.model.parameters()):
                raise ValueError('nonfinite model after optimizer step')
            # Equal EMA maintenance for every arm; only JEPA consumes the target
            # in its latent-prediction objective. Update strictly after optimizer.
            self.model.update_target(self.ema_momentum)
            if any(p.grad is not None for p in self.model.target_encoder.parameters()):
                raise ValueError('EMA encoder must stay gradient-free')
            self.updates+=1
            return dict(update=self.updates,loss=float(loss.detach()),parts=parts,
                gradient_norm_before_clip=float(norm),model_sha256=state_digest(self.model.state_dict()))
        except Exception:
            self.failed=True;raise

    def checkpoint(self):
        return deepcopy(dict(schema='pulse_timed_training_checkpoint_development.v1',
            condition=self.condition,seed=self.seed,latent_dim=self.latent_dim,
            learning_rate=self.learning_rate,ema_momentum=self.ema_momentum,
            updates=self.updates,failed=self.failed,initial_sha256=self.initial_sha256,
            model_sha256=state_digest(self.model.state_dict()),model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),navigation_qualified=False))


@torch.no_grad()
def evaluate(trainer,dataset,readers,*,role,batch_size=6):
    """Report only an explicitly named role; train-role scoring is resubstitution.

    Stream images in bounded batches; retain small prediction/target arrays.
    Target offsets remain actual2.2/2.5s partial endpoints, never rounded to2.5s.
    """
    if type(batch_size) is not int or not 1<=batch_size<=64:raise ValueError('bounded evaluation batch required')
    if role not in dataset.groups:raise ValueError('explicit dataset role required')
    indices=[i for i,w in enumerate(dataset.windows) if w['history_ready'] and dataset.episode_roles[w['condition']]['role']==role]
    if not indices:raise ValueError('no eligible role samples')
    training=trainer.model.training;trainer.model.eval();predictions=[];targets=[];active=[];offsets=[]
    try:
        for start in range(0,len(indices),batch_size):
            ids=indices[start:start+batch_size]
            batch=stack_samples([dataset.sample(i,readers) for i in ids])
            output=trainer.model(**batch['inputs'])
            predictions.append({k:output[k].cpu().numpy() for k in ('direct_outcomes','rollout_outcomes')})
            targets.append({k:batch['targets'][k].cpu().numpy() for k in ('motion','motion_valid','contact','contact_valid')})
            active.append(output['prediction_valid'].cpu().numpy());offsets.append(output['target_offsets_ns'].cpu().numpy())
    finally:trainer.model.train(training)
    p={k:np.concatenate([x[k] for x in predictions]) for k in predictions[0]}
    t={k:np.concatenate([x[k] for x in targets]) for k in targets[0]}
    mask=np.concatenate(active);times=np.concatenate(offsets)
    metadata=[dict(layout_id=dataset.episode_roles[dataset.windows[i]['condition']]['layout_id']) for i in indices]
    zero=np.zeros_like(p['direct_outcomes']);zero[...,3]=1.;zero[...,4]=-30.
    heads=['direct_outcomes'] if trainer.condition=='direct' else ['direct_outcomes','rollout_outcomes']
    def score(pred):
        return dict(all=reduce_predictions(pred,t,metadata,mask),
            by_actual_offset_ns={str(int(ns)):reduce_predictions(pred,t,metadata,mask,horizon_selection=times==ns)
                for ns in sorted(set(times[mask]))})
    return dict(role=role,resubstitution=role=='train',windows=len(indices),
        metrics={k:score(p[k]) for k in heads},zero_motion_no_contact=score(zero),
        model_sha256=state_digest(trainer.model.state_dict()),
        checkpoint_selection_performed=False,navigation_qualified=False,
        independent_generalization_established=False)
