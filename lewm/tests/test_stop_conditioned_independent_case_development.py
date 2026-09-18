import pytest
import torch

from scripts import run_go2_stop_conditioned_independent_case_v1 as trial


def test_assignment_retains_training_seed_and_sensor_treatment():
    row = trial.assignment(0, 'frozen_reference', 'seed_2026091402_no_rgb_supervised_rollout')
    assert row['training_seed'] == 2026091402
    assert (row['condition'], row['variant']) == ('supervised_rollout', 'no_rgb')
    reference = trial.assignment(0, 'frozen_reference', 'seed_2026091001_full_jepa')
    memory = trial.assignment(0, 'current_planning', 'seed_2026091001_full_jepa')
    assert {k:v for k,v in reference.items() if k != 'mode'} == {k:v for k,v in memory.items() if k != 'mode'}


def test_reactive_assignment_excludes_a_high_level_model():
    row = trial.assignment(7, 'reactive', None)
    assert row['model_name'] is row['condition'] is row['variant'] is row['training_seed'] is None
    with pytest.raises(ValueError):
        trial.assignment(7, 'reactive', 'seed_2026091001_full_jepa')


@pytest.mark.parametrize('layout,mode,model', [
    (True, 'reactive', None), (8, 'reactive', None),
    (0, 'unknown', None), (0, 'frozen_reference', None),
    (0, 'frozen_reference', 'arbitrary_checkpoint')])
def test_unregistered_assignment_is_rejected_before_execution(layout, mode, model):
    with pytest.raises(ValueError):
        trial.assignment(layout, mode, model)


def test_model_identity_rejects_training_or_retained_gradients():
    model = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError):
        trial.model_identity(model)
    model.eval(); before = trial.model_identity(model)
    model(torch.zeros(1, 2)).sum().backward()
    with pytest.raises(ValueError):
        trial.model_identity(model)
    model.zero_grad(set_to_none=True)
    assert trial.model_identity(model) == before
    assert trial.model_identity(None) is None
