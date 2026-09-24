"""Original matched learner over the explicitly admitted expanded study.

No native launch, model selection, calibration or checkpoint resume is provided.
The caller fixes a prospective roster and performs resource/throughput checks.
"""
from copy import deepcopy
import numpy as np
import torch
from lewm.observation_horizon_learning_development import ObservationHorizonTrainer
from lewm.observation_horizon_input_ablation_development import transform_inputs, transform_training_batch
from lewm.all_phase_training_schedule_development import schedule as expected_schedule, UPDATES
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.all_phase_study_stream_development import AllPhaseStudyStream, verified_plan


def train(trainer, stream, schedule, *, input_variant, on_update, benchmark=False):
    if (not isinstance(trainer, ObservationHorizonTrainer) or trainer.failed or trainer.updates != 0
            or not isinstance(stream, AllPhaseStudyStream) or stream.failed
            or type(benchmark) is not bool or not callable(on_update)):
        raise ValueError('fresh nonfailed original trainer, admitted study and durable accounting required')
    try:
        if (trainer.latent_dim != 32 or trainer.learning_rate != .001 or trainer.ema_momentum != .99
                or input_variant not in ('full', 'no_rgb')):
            raise ValueError('unchanged width32/lr0.001/EMA0.99 and full/no-RGB treatment required')
        expected = expected_schedule(stream.view, seed=schedule['seed'])
        if schedule != expected or not benchmark and trainer.seed != schedule['seed']:
            raise ValueError('exact expanded schedule and matched fitting seed required')
        count = 20 if benchmark else UPDATES
        for number, indices in enumerate(schedule['batches'][:count], 1):
            batch = stream.training_batch(list(indices))
            verified_plan(stream.view, indices, batch['inputs'])
            record = trainer.step(transform_training_batch(batch, input_variant=input_variant))
            if record['update'] != number: raise ValueError('exact optimizer accounting required')
            on_update(deepcopy(record | dict(sample_indices=list(indices),
                schedule_sha256=schedule['schedule_sha256'], input_variant=input_variant)))
        if trainer.updates != count: raise ValueError('complete fixed update count required')
        trainer.model.eval()
        return dict(updates=count, initial_sha256=trainer.initial_sha256,
            model_sha256=state_digest(trainer.model.state_dict()), benchmark=benchmark,
            input_variant=input_variant, condition=trainer.condition, seed=trainer.seed,
            schedule_sha256=schedule['schedule_sha256'])
    except Exception:
        trainer.failed = True
        raise


@torch.inference_mode()
def predict(trainer, stream, *, role, input_variant):
    if (not isinstance(trainer, ObservationHorizonTrainer) or trainer.failed or trainer.updates != UPDATES
            or not isinstance(stream, AllPhaseStudyStream) or stream.failed
            or role not in ('train', 'geometry_transfer') or input_variant not in ('full', 'no_rgb')):
        raise ValueError('complete nonfailed final fit and explicit admitted role/treatment required')
    expected = dict(train=dict(family=1664, switch=2346), geometry_transfer=dict(family=348, switch=72))
    if any(len(stream.view.indices(role, source=s)) != n for s, n in expected[role].items()):
        raise ValueError('complete exact expanded training or original transfer population required')
    indices = stream.view.indices(role)
    names = ('direct_outcomes',) if trainer.condition == 'direct' else ('direct_outcomes', 'rollout_outcomes')
    heads = {k: np.empty((len(indices), 8, 5), np.float32) for k in names}
    masks = np.empty((len(indices), 8), bool); clocks = np.empty((len(indices), 8), np.int64)
    before = state_digest(trainer.model.state_dict()); trainer.model.eval(); start = 0
    for source in ('family', 'switch'):
        source_ids = stream.view.indices(role, source=source)
        for at in range(0, len(source_ids), 6):
            ids = source_ids[at:at+6]
            if ids != indices[start:start+len(ids)]: raise ValueError('complete source-grouped prediction order required')
            inputs = stream.inference_batch(ids, role=role)
            active, offsets = verified_plan(stream.view, ids, inputs)
            out = trainer.model(**transform_inputs(inputs, input_variant=input_variant))
            if not torch.equal(out['prediction_valid'], active) or not torch.equal(out['target_offsets_ns'], offsets):
                raise ValueError('actual inference clocks and masks required')
            masks[start:start+len(ids)] = active.numpy(); clocks[start:start+len(ids)] = offsets.numpy()
            for name in names:
                value = out[name]
                if value.shape != (len(ids), 8, 5) or value.dtype != torch.float32 or not torch.isfinite(value[active]).all():
                    raise ValueError('finite complete active five-component predictions required')
                heads[name][start:start+len(ids)] = value.numpy()
            start += len(ids)
    if start != len(indices) or state_digest(trainer.model.state_dict()) != before:
        raise ValueError('all inference rows and unchanged final model state required')
    return dict(indices=np.asarray(indices, np.int64), prediction_valid=masks, target_offsets_ns=clocks, **heads)
