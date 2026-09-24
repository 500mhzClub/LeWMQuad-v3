from copy import deepcopy
import pytest
from lewm.all_phase_training_schedule_development import schedule, SEEDS
from lewm.all_phase_training_fit_development import train, predict
from lewm.observation_horizon_learning_development import ObservationHorizonTrainer
from lewm.tests.test_all_phase_study_stream_development import routed_stream


@pytest.mark.parametrize('fault', ['transfer', 'order', 'seed', 'variant', 'width', 'learning_rate', 'ema'])
def test_changed_schedule_or_model_definition_rejected_before_any_input_read(fault):
    stream, calls = routed_stream(); plan = schedule(stream.view, seed=SEEDS[0])
    kwargs = dict(seed=SEEDS[0]); variant = 'full'
    if fault == 'transfer': plan['batches'][0][0] = stream.view.indices('geometry_transfer')[0]
    elif fault == 'order': plan['batches'][0], plan['batches'][1] = plan['batches'][1], plan['batches'][0]
    elif fault == 'seed': kwargs['seed'] = SEEDS[1]
    elif fault == 'variant': variant = 'unassigned'
    elif fault == 'width': kwargs['latent_dim'] = 8
    elif fault == 'learning_rate': kwargs['learning_rate'] = .002
    elif fault == 'ema': kwargs['ema_momentum'] = .98
    trainer = ObservationHorizonTrainer('jepa', **kwargs)
    with pytest.raises(ValueError): train(trainer, stream, plan, input_variant=variant, on_update=lambda _: pytest.fail('unexpected optimizer receipt'))
    assert trainer.failed and trainer.updates == 0 and calls == []


def test_untrained_model_cannot_enter_final_prediction():
    stream, calls = routed_stream(); trainer = ObservationHorizonTrainer('direct', seed=SEEDS[0])
    with pytest.raises(ValueError, match='complete'): predict(trainer, stream, role='geometry_transfer', input_variant='full')
    assert calls == [] and trainer.updates == 0
