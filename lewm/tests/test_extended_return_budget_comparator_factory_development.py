"""Real fresh controller construction with synthetic admitted tensor loaders."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
import numpy as np

from scripts import extended_return_budget_comparator_factory_development as factory
from lewm.tests.test_measured_plane_comparator_controllers_development import MISSION
from lewm.tests.test_all_phase_planner_model_adapter_development import model as corrected_fixture
from scripts import all_phase_planner_model_admission_development as planner_loader
from lewm.training_bias_predictive_selection_development import select
from lewm.tests.test_observation_horizon_goal_selection_development import history


class TinyModel(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.child = torch.nn.Linear(2, 2)
        with torch.no_grad():
            self.child.weight.fill_(value)
            self.child.bias.fill_(-value)
        self.eval()

    def forward(self, x):
        return self.child(x)


def adapted_model(row):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(row['seed'])
        heads = ('direct_outcomes',) if row['condition'] == 'direct' else ('direct_outcomes', 'rollout_outcomes')
        return factory.AllPhasePlannerModel(corrected_fixture(heads))


def assignment(row):
    model = adapted_model(row)
    return factory.ModelAssignment(row['name'], factory.state_digest(model.state_dict()))


@pytest.fixture
def loader(monkeypatch):
    calls = []
    def load(admission, name):
        row = next(row for row in factory.ROSTER if row['name'] == name)
        model = adapted_model(row)
        calls.append((name, model))
        return model, row['condition'], row['variant']
    monkeypatch.setattr(factory, 'load_assigned', load)
    admission = dict(correction_result_sha256=factory.CORRECTION_RESULT_SHA256,
        all_coefficients_reconstructed=True, all_models=18, all_trained_heads=30)
    return calls, admission


@pytest.mark.parametrize('row', factory.ROSTER, ids=lambda row: row['name'])
def test_every_original_training_assignment_builds_all_three_predictive_controls(row, loader):
    calls, admission = loader
    selected = assignment(row)
    results = [factory.create(mode, object(), public_mission=deepcopy(MISSION),
        assignment=selected, correction_admission=admission)
        for mode in ('frozen_reference', 'nominal', 'current_planning')]
    assert [name for name, _ in calls] == [row['name']]*3
    for controller, model, receipt in results:
        assert controller.model is model
        assert factory.state_digest(model.state_dict()) == selected.model_state_sha256
        assert controller.mission.navigation_ticks == 8000
        assert controller.memory is controller.mapper.surface
        assert controller.selector.residual is controller.residual
        assert controller.selector.condition == row['condition']
        assert controller.selector.variant == row['variant']
        assert receipt['training_seed'] == row['seed']
        assert receipt['model_unchanged_after_construction']
        assert not receipt['population_assignment_authenticated']
        assert not receipt['model_forward_during_construction']
        assert not receipt['native_execution']
        assert all(not module._forward_pre_hooks for module in model.modules())
    assert results[0][0].selector.forecast_source == 'frozen_world_model'
    assert results[1][0].selector.forecast_source == 'nominal_requested_twist'
    assert results[2][0].mapper.current_planning_view is None
    for field in ('model', 'mapper', 'memory', 'motion', 'registration', 'mission',
                  'selector', 'residual', 'history'):
        assert len({id(getattr(controller, field)) for controller, _, _ in results}) == 3
    models = [model for _, model, _ in results]
    assert len({next(model.parameters()).data_ptr() for model in models}) == 3


def test_reactive_is_fresh_and_never_loads_any_model(loader):
    calls, _ = loader
    results = [factory.create('reactive', object(), public_mission=deepcopy(MISSION)) for _ in range(2)]
    assert not calls
    for controller, model, receipt in results:
        assert model is None and not hasattr(controller, 'model') and not hasattr(controller, 'residual')
        assert controller.mission.navigation_ticks == 8000
        assert controller.memory is controller.mapper.surface
        assert not receipt['learned_model_loaded']
    for field in ('mapper', 'memory', 'motion', 'registration', 'mission'):
        assert getattr(results[0][0], field) is not getattr(results[1][0], field)


@pytest.mark.parametrize('fault', ['assignment', 'admission', 'both'])
def test_reactive_rejects_hidden_model_treatment_before_loading(loader, fault):
    calls, admission = loader
    kwargs = {}
    if fault in ('assignment', 'both'): kwargs['assignment'] = assignment(factory.ROSTER[0])
    if fault in ('admission', 'both'): kwargs['correction_admission'] = admission
    with pytest.raises(ValueError, match='reactive construction'):
        factory.create('reactive', object(), public_mission=MISSION, **kwargs)
    assert not calls


@pytest.mark.parametrize('fault', ['unknown_name', 'dictionary', 'absent', 'short_sha', 'upper_sha', 'bool_name'])
def test_invalid_model_assignment_rejected_before_loading(loader, fault):
    calls, admission = loader
    selected = assignment(factory.ROSTER[0])
    if fault == 'unknown_name': selected = replace(selected, name='transfer_selected_checkpoint')
    if fault == 'dictionary': selected = vars(selected)
    if fault == 'absent': selected = None
    if fault == 'short_sha': selected = replace(selected, model_state_sha256='abc')
    if fault == 'upper_sha': selected = replace(selected, model_state_sha256='A'*64)
    if fault == 'bool_name': selected = replace(selected, name=True)
    with pytest.raises(ValueError):
        factory.create('frozen_reference', object(), public_mission=MISSION,
            assignment=selected, correction_admission=admission)
    assert not calls


@pytest.mark.parametrize('key,value', [
    ('correction_result_sha256', 'substitute'), ('all_coefficients_reconstructed', 1),
    ('all_models', 17), ('all_models', 18.), ('all_trained_heads', 29), ('all_trained_heads', 30.)])
def test_incomplete_correction_admission_rejected_before_loading(loader, key, value):
    calls, admission = loader
    admission[key] = value
    with pytest.raises(ValueError, match='correction admission'):
        factory.create('nominal', object(), public_mission=MISSION,
            assignment=assignment(factory.ROSTER[0]), correction_admission=admission)
    assert not calls


@pytest.mark.parametrize('fault', ['state', 'condition', 'variant', 'child_training', 'gradient'])
def test_changed_loaded_model_rejected_before_constructor(monkeypatch, loader, fault):
    _, admission = loader
    original = factory.load_assigned
    def load(*args):
        model, condition, variant = original(*args)
        if fault == 'state':
            with torch.no_grad(): next(model.parameters()).add_(1)
        if fault == 'condition': condition = 'jepa'
        if fault == 'variant': variant = 'no_rgb'
        if fault == 'child_training': model.base.train()
        if fault == 'gradient':
            parameter = next(model.parameters()); parameter.grad = torch.zeros_like(parameter)
        return model, condition, variant
    monkeypatch.setattr(factory, 'load_assigned', load)
    def forbidden(*args, **kwargs): raise AssertionError('constructor reached')
    monkeypatch.setitem(factory.CONTROLLERS, 'frozen_reference', forbidden)
    with pytest.raises(ValueError):
        factory.create('frozen_reference', object(), public_mission=MISSION,
            assignment=assignment(factory.ROSTER[0]), correction_admission=admission)


@pytest.mark.parametrize('fault', ['mutate', 'train_child', 'replace', 'root_forward', 'child_forward'])
def test_constructor_mutation_substitution_or_swallowed_inference_is_rejected(monkeypatch, loader, fault):
    calls, admission = loader
    original = factory.CONTROLLERS['nominal']
    def construct(model, *args, **kwargs):
        controller = original(model, *args, **kwargs)
        if fault == 'mutate':
            with torch.no_grad(): next(model.parameters()).add_(1)
        if fault == 'train_child': model.base.train()
        if fault == 'replace': controller.model = TinyModel(99)
        if fault in ('root_forward', 'child_forward'):
            try:
                (model if fault == 'root_forward' else model.base)(torch.zeros(1, 2))
            except RuntimeError:
                pass
        return controller
    monkeypatch.setitem(factory.CONTROLLERS, 'nominal', construct)
    with pytest.raises(ValueError):
        factory.create('nominal', object(), public_mission=MISSION,
            assignment=assignment(factory.ROSTER[0]), correction_admission=admission)
    assert all(not module._forward_pre_hooks for _, model in calls for module in model.modules())


@pytest.mark.parametrize('mode', ['best', None, True])
def test_unknown_modes_cannot_trigger_model_loading(loader, mode):
    calls, admission = loader
    with pytest.raises(ValueError, match='mode'):
        factory.create(mode, object(), public_mission=MISSION,
            assignment=assignment(factory.ROSTER[0]), correction_admission=admission)
    assert not calls


def test_factory_rejects_unadapted_corrected_model_even_with_exact_tensor_identity(monkeypatch, loader):
    _, admission = loader
    source = corrected_fixture(('direct_outcomes',))
    selected = factory.ModelAssignment(factory.ROSTER[0]['name'], factory.state_digest(source.state_dict()))
    monkeypatch.setattr(factory, 'load_assigned', lambda *args: (source, 'direct', 'full'))
    with pytest.raises(ValueError, match='planner'):
        factory.create('frozen_reference', object(), public_mission=MISSION,
            assignment=selected, correction_admission=admission)


@pytest.mark.parametrize('row', factory.ROSTER[:6], ids=lambda row: row['name'])
def test_production_loader_adapter_and_real_forecast_selector_are_integrated(monkeypatch, row):
    assert factory.load_assigned is planner_loader.load_assigned
    heads = ('direct_outcomes',) if row['condition'] == 'direct' else ('direct_outcomes', 'rollout_outcomes')
    source = corrected_fixture(heads)
    selected = factory.ModelAssignment(row['name'], factory.state_digest(source.state_dict()))
    loaded = []
    def original_load(admission, name):
        assert name == row['name']
        loaded.append(name)
        return deepcopy(source), row['condition'], row['variant']
    monkeypatch.setattr(planner_loader, 'original_load', original_load)
    admission = dict(correction_result_sha256=factory.CORRECTION_RESULT_SHA256,
        all_coefficients_reconstructed=True, all_models=18, all_trained_heads=30)
    controller, model, receipt = factory.create('frozen_reference', object(), public_mission=MISSION,
        assignment=selected, correction_admission=admission)
    assert loaded == [row['name']] and type(model) is factory.AllPhasePlannerModel
    assert controller.model is model and receipt['model_state_sha256'] == selected.model_state_sha256
    kwargs = dict(head=heads[-1], input_variant=row['variant'], goal_body_xy_m=[1., 0.], contact_penalty_m=1.2)
    with pytest.raises(ValueError, match='wrapper'):
        select(source, history(), **kwargs)
    calls = []
    handle = model.register_forward_pre_hook(lambda *args: calls.append(True))
    try:
        result = select(model, history(), **kwargs)
    finally:
        handle.remove()
    assert calls == [True] and np.isfinite(result['prediction']).all()
    assert factory.state_digest(model.state_dict()) == selected.model_state_sha256
    assert all(parameter.grad is None for parameter in model.parameters())
