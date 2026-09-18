import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
from lewm.independent_round_trip_comparison_study_development import ARMS, CASES, CORRECTION_RESULT_SHA256
from scripts import independent_round_trip_adapter_controller_factory_development as factory
from scripts import independent_round_trip_adapter_multiarm_episode_development as episode
from scripts import independent_round_trip_adapter_multiarm_audit_development as audit


@pytest.mark.parametrize('kind,name', [('episode', 'collect'), ('episode', 'artifacts'), ('audit', 'audit')])
def test_complete_multiarm_collection_and_audit_calculations_unchanged(kind, name):
    def function(path):
        return next(n for n in ast.parse(Path(path).read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == name)
    old = function('scripts/independent_round_trip_multiarm_'+kind+'_development.py')
    new = function('scripts/independent_round_trip_adapter_multiarm_'+kind+'_development.py')
    assert ast.dump(old) == ast.dump(new)
    assert episode.create is audit.create is factory.create


def fake_loader(monkeypatch, *, fault=None):
    class Model:
        def __init__(self, state): self.state = state; self.training = fault == 'training'
        def state_dict(self): return self.state
    def load(admission, name):
        arm = next(a for a in ARMS if a.model_name == name)
        model = Model('changed' if fault == 'state' else arm.model_state_sha256)
        return model, 'direct' if fault == 'condition' else arm.condition, 'no_rgb' if fault == 'variant' else arm.variant
    monkeypatch.setattr(factory, 'AllPhasePlannerModel', Model)
    monkeypatch.setattr(factory, 'load_assigned', load)
    monkeypatch.setattr(factory, 'state_digest', lambda state: state)
    return dict(correction_result_sha256=CORRECTION_RESULT_SHA256)


@pytest.mark.parametrize('name', ['persistent_jepa', 'persistent_supervised', 'current_pair_jepa'])
def test_assigned_learned_controllers_are_fresh_and_keep_the_same_scope(monkeypatch, name):
    admission = fake_loader(monkeypatch); case = next(c for c in CASES if c.arm_name == name)
    arm = next(a for a in ARMS if a.name == name)
    (first, m1), (second, m2) = [factory.create(case, object(), correction_admission=admission) for _ in range(2)]
    assert type(first).__name__ == type(second).__name__ == arm.implementation
    assert m1 is not m2 and m1.state_dict() == m2.state_dict() == arm.model_state_sha256
    for field in ('mapper', 'memory', 'residual', 'mission', 'registration', 'motion', 'selector', 'history'):
        assert getattr(first, field) is not getattr(second, field)
    assert first.selector.condition == arm.condition


def test_reactive_arm_never_loads_adapter_or_world_model(monkeypatch):
    monkeypatch.setattr(factory, 'load_assigned', lambda *a: pytest.fail('reactive model load'))
    case = next(c for c in CASES if c.arm_name == 'reactive')
    controller, model = factory.create(case, object())
    assert model is None and not hasattr(controller, 'model') and not hasattr(controller, 'residual')


@pytest.mark.parametrize('fault', ['state', 'condition', 'variant', 'training'])
def test_changed_model_treatment_or_mode_rejected(monkeypatch, fault):
    admission = fake_loader(monkeypatch, fault=fault)
    with pytest.raises(ValueError, match='exact preassigned'):
        factory.create(CASES[0], object(), correction_admission=admission)


def test_incompatible_old_wrapper_cannot_pass_factory_even_with_correct_identity(monkeypatch):
    arm = ARMS[0]
    monkeypatch.setattr(factory, 'load_assigned', lambda *a: (SimpleNamespace(training=False,
        state_dict=lambda: arm.model_state_sha256), arm.condition, arm.variant))
    with pytest.raises(ValueError, match='exact preassigned'):
        factory.create(CASES[0], object(), correction_admission={'correction_result_sha256': CORRECTION_RESULT_SHA256})
