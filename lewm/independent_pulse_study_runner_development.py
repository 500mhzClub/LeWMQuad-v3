"""Bounded matched training/inference mechanics, not a launched experiment.

Caller freezes the dataset receipts, schedule, model arms, seeds and resource
budget. No implicit scheduling, checkpoint selection, file writes or physics.
"""
from copy import deepcopy
import numpy as np
import torch

from lewm.cumulative_pulse_learning_development import CumulativePulseTrainer
from lewm.independent_pulse_evaluation_development import IndependentPulseEvaluation
from lewm.independent_pulse_input_ablation_development import (
    transform_inputs, transform_training_batch, validate_variant)
from lewm.pulse_timed_training_runner_development import state_digest
from lewm.pulse_timed_rgb_body_jepa_development import pulse_brake_plan, validate_timed_plan

MAX_BATCH = 16


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verified_plan(view, ids, inputs):
    """Authenticate original prospective values before any information removal."""
    require(isinstance(inputs, dict) and set(inputs) ==
        {'observation_history', 'known_action_blocks', 'known_action_valid'}, 'policy-only inference fields required')
    blocks, valid = zip(*(pulse_brake_plan(tuple(view.dataset.windows[i]['command']),
        view.dataset.windows[i]['pulse_ticks']) for i in ids), strict=True)
    require(torch.equal(inputs['known_action_blocks'], torch.stack(blocks))
        and torch.equal(inputs['known_action_valid'], torch.stack(valid)), 'exact prescribed prospective action plans required')
    return validate_timed_plan(inputs['known_action_blocks'], inputs['known_action_valid'], len(ids))


def train_schedule(trainer, stream, schedule, *, on_update, input_variant='full'):
    """Fresh fit using exactly the published train schedule, including repeats.

    The callback must persist any required accounting before returning. Callback
    failures latch this attempt after its actual completed optimizer update.
    This helper neither selects a budget nor supplies a complete experiment gate.
    """
    require(isinstance(trainer, CumulativePulseTrainer) and not trainer.failed and trainer.updates == 0,
        'fresh cumulative-event trainer required; no retry/resume')
    require(isinstance(stream.evaluation, IndependentPulseEvaluation) and callable(on_update),
        'validated evaluation and explicit update accounting callback required')
    try:
        validate_variant(input_variant)
        schedule = deepcopy(schedule)
        require(isinstance(schedule, dict) and schedule.get('role') == 'train'
            and type(schedule.get('batch_size')) is int and 1 <= schedule['batch_size'] <= MAX_BATCH,
            'bounded explicit training schedule required')
        expected = stream.evaluation.dataset.schedule('train', updates=schedule['updates'],
            batch_size=schedule['batch_size'], seed=schedule['seed'], require_all_actions=schedule['require_all_actions'])
        require(schedule == expected and schedule['require_all_actions'] is True,
            'exact layout/action-balanced schedule with unchanged identities required')
        population = stream.evaluation.population('train')
        require(all(r['eligible'] > 0 for r in population['layouts'].values()),
            'cannot silently drop an entirely absent planned training layout')
        initial = state_digest(trainer.model.state_dict()); completed = 0
        for expected_update, indices in enumerate(schedule['batches'], start=1):
            batch = stream.training_batch(list(indices))
            verified_plan(stream.evaluation, indices, batch['inputs'])
            record = trainer.step(transform_training_batch(batch, input_variant=input_variant))
            record = record | dict(sample_indices=list(indices), schedule_sha256=schedule['schedule_sha256'],
                input_variant=input_variant)
            require(record['update'] == expected_update, 'optimizer accounting mismatch')
            on_update(deepcopy(record))
            # The caller stores full records; retain only fixed-size accounting
            # here, not image batches or one duplicate log per update.
            completed += 1
        require(completed == schedule['updates'], 'optimizer accounting mismatch')
        return dict(condition=trainer.condition, seed=trainer.seed, updates=trainer.updates, input_variant=input_variant,
            schedule_sha256=schedule['schedule_sha256'], initial_sha256=initial,
            model_sha256=state_digest(trainer.model.state_dict()),
            training_draw_indices=[i for b in schedule['batches'] for i in b],
            inference_performed=False, checkpoint_selection_performed=False, navigation_qualified=False)
    except Exception:
        trainer.failed = True
        raise


@torch.inference_mode()
def predict_heads(trainer, stream, *, role, batch_size, input_variant='full'):
    """Return raw predictions with exact row bindings for both existing scorers.

    Forward receives only past observation histories and prospective actions.
    Native labels remain in the scoring view, never in the input materializer.
    Reject wrong clocks/masks or nonfinite active outputs; never score a subset.
    """
    require(isinstance(trainer, CumulativePulseTrainer) and not trainer.failed,
        'nonfailed cumulative-event trainer required')
    require(isinstance(stream.evaluation, IndependentPulseEvaluation)
        and type(batch_size) is int and 1 <= batch_size <= MAX_BATCH,
        'validated evaluation and bounded inference batch required')
    view = stream.evaluation; data = view.arrays(role)
    indices = data['indices']; model = trainer.model; mode = model.training
    before = state_digest(model.state_dict()); updates = trainer.updates
    names = ('direct_outcomes',) if trainer.condition == 'direct' else ('direct_outcomes', 'rollout_outcomes')
    prediction = {k: np.empty((len(indices), 8, 5), dtype=np.float32) for k in names}
    model.eval()
    try:
        validate_variant(input_variant)
        for start in range(0, len(indices), batch_size):
            ids = indices[start:start + batch_size].tolist()
            inputs = stream.inference_batch(ids, role=role)
            active, offsets = verified_plan(view, ids, inputs)
            inputs = transform_inputs(inputs, input_variant=input_variant)
            output = model(**inputs)
            require(output['prediction_valid'].dtype == torch.bool and output['target_offsets_ns'].dtype == torch.int64
                and torch.equal(output['prediction_valid'], active) and torch.equal(output['target_offsets_ns'], offsets),
                'model output must use exact actual horizons and masks')
            for name in names:
                p = output[name]
                require(p.shape == (len(ids), 8, 5) and p.dtype == torch.float32
                    and torch.isfinite(p[active]).all().item(), 'finite complete active five-component predictions required')
                prediction[name][start:start + len(ids)] = p.cpu().numpy()
        require(trainer.updates == updates and state_digest(model.state_dict()) == before,
            'inference cannot update model parameters or buffers')
    except Exception:
        trainer.failed = True
        raise
    finally:
        model.train(mode)
    primary = 'direct_outcomes' if trainer.condition == 'direct' else 'rollout_outcomes'
    return dict(role=role, condition=trainer.condition, seed=trainer.seed, model_sha256=before,
        updates=updates, primary_head=primary, input_variant=input_variant,
        heads={name: dict(indices=indices.copy(), prediction=p) for name, p in prediction.items()},
        checkpoint_selection_performed=False, navigation_qualified=False,
        provenance_note='caller must use receipt-authenticated stream; this numerical runner does not verify files')
