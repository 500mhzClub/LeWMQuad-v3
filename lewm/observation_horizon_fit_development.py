"""Fixed augmented-data fits and source-stratified policy-only predictions."""
from copy import deepcopy
import numpy as np
import torch
from lewm.observation_horizon_learning_development import ObservationHorizonTrainer
from lewm.observation_horizon_plan_development import plan,validate_plan
from lewm.observation_horizon_view_development import ObservationHorizonView
from lewm.observation_horizon_input_ablation_development import transform_inputs, transform_training_batch
from lewm.pulse_timed_training_runner_development import state_digest


def verified_plan(view, indices, inputs):
    if not isinstance(view, ObservationHorizonView):
        raise ValueError('actual complete family view required')
    values, valid = zip(*((plan(view.rows[i]['action'], offset_ticks=view.rows[i]['offset_ticks']) if view.rows[i]['source']=='family'
        else plan(view.rows[i]['action'])) for i in indices), strict=True)
    if not torch.equal(inputs['known_action_blocks'], torch.stack(values)) or not torch.equal(inputs['known_action_valid'], torch.stack(valid)):
        raise ValueError('exact family suffix action values and masks required before ablation')
    return validate_plan(inputs['known_action_blocks'], inputs['known_action_valid'], len(indices))


def train(trainer, stream, schedule, *, input_variant, on_update, benchmark=False):
    if (not isinstance(trainer, ObservationHorizonTrainer) or trainer.failed or trainer.updates != 0
            or type(benchmark) is not bool or not callable(on_update)):
        raise ValueError('fresh nonfailed trainer and explicit durable accounting required')
    try:
        expected = stream.view.schedule(updates=1200, batch_size=6, seed=schedule['seed'])
        if not benchmark and trainer.seed!=schedule['seed']:raise ValueError('fit seed and matched data schedule must agree')
        if schedule != expected:
            raise ValueError('exact complete prospective family schedule required')
        count = 20 if benchmark else 1200
        for number, indices in enumerate(schedule['batches'][:count], 1):
            batch = stream.training_batch(list(indices))
            verified_plan(stream.view, indices, batch['inputs'])
            record = trainer.step(transform_training_batch(batch, input_variant=input_variant))
            if record['update'] != number:
                raise ValueError('exact optimizer accounting required')
            on_update(deepcopy(record | dict(sample_indices=list(indices),
                schedule_sha256=schedule['schedule_sha256'], input_variant=input_variant)))
        if trainer.updates != count:
            raise ValueError('complete fixed update count required')
        trainer.model.eval()
        return dict(updates=count, initial_sha256=trainer.initial_sha256,
            model_sha256=state_digest(trainer.model.state_dict()), benchmark=benchmark,
            input_variant=input_variant, condition=trainer.condition, seed=trainer.seed)
    except Exception:
        trainer.failed = True
        raise


@torch.inference_mode()
def predict(trainer, stream, *, role, input_variant):
    if not isinstance(trainer, ObservationHorizonTrainer) or trainer.failed or trainer.updates != 1200:
        raise ValueError('complete nonfailed final family fit required')
    indices = stream.view.indices(role)
    if len(stream.view.indices(role,source='family')) != (336 if role=='train' else 348):
        raise ValueError('unchanged original family role population required')
    if role=='train' and len(stream.view.indices(role,source='switch'))!=72:
        raise ValueError('every new training cell required')
    names = ('direct_outcomes',) if trainer.condition == 'direct' else ('direct_outcomes', 'rollout_outcomes')
    heads = {k: np.empty((len(indices), 8, 5), np.float32) for k in names}
    masks = np.empty((len(indices), 8), bool); clocks = np.empty((len(indices), 8), np.int64)
    before = state_digest(trainer.model.state_dict()); trainer.model.eval()
    start=0
    for source in ('family','switch'):
        source_ids=stream.view.indices(role,source=source)
        for at in range(0,len(source_ids),6):
            ids=source_ids[at:at+6]
            if ids!=indices[start:start+len(ids)]:raise ValueError('complete source-grouped prediction order required')
            inputs=stream.inference_batch(ids,role=role)
            active,offsets=verified_plan(stream.view,ids,inputs)
            out=trainer.model(**transform_inputs(inputs,input_variant=input_variant))
            if not torch.equal(out['prediction_valid'],active) or not torch.equal(out['target_offsets_ns'],offsets):
                raise ValueError('complete actual inference clocks/masks required')
            masks[start:start+len(ids)]=active.numpy();clocks[start:start+len(ids)]=offsets.numpy()
            for name in names:
                value=out[name]
                if value.shape!=(len(ids),8,5) or value.dtype!=torch.float32 or not torch.isfinite(value[active]).all():
                    raise ValueError('finite complete active five-component predictions required')
                heads[name][start:start+len(ids)]=value.numpy()
            start+=len(ids)
    if start!=len(indices):raise ValueError('no inference rows omitted')
    if state_digest(trainer.model.state_dict()) != before:
        raise ValueError('prediction changed final model state')
    return dict(indices=np.asarray(indices, np.int64), prediction_valid=masks, target_offsets_ns=clocks, **heads)


def score(view, arrays, *, role, head):
    indices = view.indices(role)
    if not np.array_equal(arrays['indices'], indices) or arrays[head].shape != (len(indices), 8, 5):
        raise ValueError('complete ordered role predictions required for scoring')
    cells = {}
    for row, index in enumerate(indices):
        window = view.rows[index]
        for h, target in enumerate(window['targets']):
            if bool(arrays['prediction_valid'][row, h]) != target['in_plan'] or arrays['target_offsets_ns'][row, h] != target['offset_ns']:
                raise ValueError('raw predictions must match all original target clocks')
            p = arrays[head][row, h].astype(float)
            for scope in ('all', window['stratum'], *(['first_observation',window['stratum']+'_first_observation'] if h==0 else []),
                    *(['half_second',window['stratum']+'_half_second'] if h==4 else [])):
                cell = cells.setdefault((window['source'], scope, window['cluster']), dict(position=[], yaw=[], brier=[],
                    motion_targets=0, contact_targets=0, contact_positives=0, undefined_yaw=0))
                if target['in_plan'] and not np.isfinite(p).all():
                    raise ValueError('finite active prediction required')
                if target['motion_valid']:
                    m = np.asarray(target['motion']); cell['motion_targets'] += 1
                    cell['position'].append(float(np.linalg.norm(p[:2]-m[:2])))
                    if np.linalg.norm(p[2:4]) <= 1e-8:
                        cell['undefined_yaw'] += 1
                    else:
                        delta = np.arctan2(p[2], p[3])-m[2]
                        cell['yaw'].append(float(abs(np.arctan2(np.sin(delta), np.cos(delta)))))
                if target['contact_valid']:
                    probability = float(np.exp(-np.logaddexp(0., -p[4])))
                    cell['brier'].append((probability-target['contact'])**2)
                    cell['contact_targets'] += 1; cell['contact_positives'] += int(target['contact'])
    rows = []
    for (source, scope, cluster), c in sorted(cells.items()):
        rows.append(dict(source=source, scope=scope, cluster=cluster, **{k: c[k] for k in ('motion_targets', 'contact_targets', 'contact_positives', 'undefined_yaw')},
            position_error_m=float(np.mean(c['position'])) if c['position'] else None,
            yaw_error_rad=float(np.mean(c['yaw'])) if c['yaw'] and not c['undefined_yaw'] else None,
            contact_brier=float(np.mean(c['brier'])) if c['brier'] else None))
    return dict(role=role, primary_head=head, samples=len(indices), clusters=rows,
        planned_family_windows=sum(r['data_role']==role and r['source']=='family' for r in view.rows),
        planned_switch_cells=sum(r['data_role']==role and r['source']=='switch' for r in view.rows),
        available_source_samples={s:len(view.indices(role,source=s)) for s in ('family','switch')},
        resubstitution=role=='train', window_horizons_are_independent=False,
        final_evaluation=False, navigation_qualified=False, calibrated_probability=False)
