"""New roster, preserved predecessor and exact fresh factory assignments."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from lewm import measured_plane_independent_round_trip_study_development as study
from lewm.measured_plane_visual_motion_development import MeasuredPlaneVisualMotion
from scripts import measured_plane_independent_controller_factory_development as factory


def test_new_roster_keeps_models_layout_units_and_balanced_order_without_mutating_old():
    assert len(study.CASES) == 32 and len({c.name for c in study.CASES}) == 32
    assert not {c.name for c in study.CASES} & {c.name for c in study.previous.CASES}
    assert [c.layout_index for c in study.CASES] == [c.layout_index for c in study.previous.CASES]
    assert [c.arm_name for c in study.CASES] == [c.arm_name for c in study.previous.CASES]
    for new, old in zip(study.ARMS, study.previous.ARMS, strict=True):
        assert replace(new, implementation=old.implementation) == old
        assert new.implementation != old.implementation
    assert study.previous.ARMS[0].implementation == 'ResidualAnchoredContinuationController'
    assert study.NAVIGATION_TICKS == 4000 and study.MAX_OBSERVATIONS == 4014
    for arm in study.ARMS:
        positions = [i % 4 for i, case in enumerate(study.CASES) if case.arm_name == arm.name]
        assert sorted(positions) == [0, 0, 1, 1, 2, 2, 3, 3]


@pytest.mark.parametrize('case', [study.previous.CASES[0], replace(study.CASES[0], layout_index=True),
    replace(study.CASES[0], name='other'), replace(study.CASES[0], arm_name='other')])
def test_previous_or_unassigned_case_cannot_enter_new_factory(case):
    with pytest.raises(ValueError, match='exact case'): factory.create(case, object())


def fake_loader(monkeypatch, fault=None):
    class Model:
        def __init__(self, state): self.state = state; self.training = fault == 'training'
        def state_dict(self): return self.state
        def parameters(self): return [SimpleNamespace(grad=1 if fault == 'gradient' else None)]
    def load(admission, name):
        arm = next(a for a in study.ARMS if a.model_name == name)
        model = Model('changed' if fault == 'state' else arm.model_state_sha256)
        return model, 'direct' if fault == 'condition' else arm.condition, 'no_rgb' if fault == 'variant' else arm.variant
    monkeypatch.setattr(factory, 'AllPhasePlannerModel', Model)
    monkeypatch.setattr(factory, 'load_assigned', load)
    monkeypatch.setattr(study, 'state_digest', lambda state: state)
    return dict(correction_result_sha256=study.CORRECTION_RESULT_SHA256)


@pytest.mark.parametrize('arm_name', ['persistent_jepa', 'persistent_supervised', 'current_pair_jepa'])
def test_preassigned_revised_controllers_and_model_states_are_fresh(monkeypatch, arm_name):
    admission = fake_loader(monkeypatch); case = next(c for c in study.CASES if c.arm_name == arm_name)
    arm = study.require_case(case)
    first, m1 = factory.create(case, object(), correction_admission=admission)
    second, m2 = factory.create(case, object(), correction_admission=admission)
    assert type(first).__name__ == type(second).__name__ == arm.implementation
    assert type(first.motion) is type(second.motion) is MeasuredPlaneVisualMotion
    assert m1 is not m2 and m1.state_dict() == m2.state_dict() == arm.model_state_sha256
    for field in ('mapper', 'memory', 'residual', 'mission', 'registration', 'motion', 'selector', 'history'):
        assert getattr(first, field) is not getattr(second, field)
    assert first.selector.condition == arm.condition


def test_reactive_factory_loads_no_world_model(monkeypatch):
    def forbidden(*args): pytest.fail('reactive factory attempted a model load')
    monkeypatch.setattr(factory, 'load_assigned', forbidden)
    case = next(c for c in study.CASES if c.arm_name == 'reactive')
    controller, model = factory.create(case, object())
    assert model is None and not hasattr(controller, 'model') and not hasattr(controller, 'residual')
    assert type(controller.motion) is MeasuredPlaneVisualMotion
    assert type(controller).__name__ == study.require_case(case).implementation


@pytest.mark.parametrize('fault', ['state', 'condition', 'variant', 'training', 'gradient'])
def test_changed_model_assignment_evaluation_mode_or_gradient_is_rejected(monkeypatch, fault):
    admission = fake_loader(monkeypatch, fault)
    with pytest.raises(ValueError): factory.create(study.CASES[0], object(), correction_admission=admission)


def test_exact_adapter_and_original_correction_are_required(monkeypatch):
    admission = fake_loader(monkeypatch)
    with pytest.raises(ValueError, match='correction'): factory.create(study.CASES[0], object())
    with pytest.raises(ValueError, match='correction'):
        factory.create(study.CASES[0], object(), correction_admission={'correction_result_sha256': 'changed'})
    arm = study.ARMS[0]
    monkeypatch.setattr(factory, 'load_assigned', lambda *args: (SimpleNamespace(training=False,
        state_dict=lambda: arm.model_state_sha256, parameters=lambda: []), arm.condition, arm.variant))
    with pytest.raises(ValueError, match='model adapter'): factory.create(study.CASES[0], object(), correction_admission=admission)


def test_collection_receipts_cannot_claim_old_budget_or_perception():
    case = study.CASES[0]
    result = dict(status=study.COLLECTION_STATUS, navigation_ticks=4000, **study.treatment(case))
    study.require_collection(case, result)
    for changed in (dict(navigation_ticks=3000), dict(measured_plane_constrained_estimator=False), dict(status='OLD')):
        with pytest.raises(ValueError): study.require_collection(case, result | changed)


def test_whole_remaining_population_storage_is_reserved_and_order_cannot_be_skipped():
    gib = 1024**3; resources = dict(memory_available_bytes=32*gib, artifact_free_bytes=520*gib)
    receipt = study.resources_for(resources)
    assert receipt['remaining_cases'] == 32 and receipt['required_free_bytes'] == 520*gib
    with pytest.raises(ValueError): study.resources_for(resources | {'artifact_free_bytes': 520*gib-1})
    with pytest.raises(ValueError): study.resources_for(resources, [study.CASES[1].name])
    receipt = study.resources_for(resources, [study.CASES[0].name])
    assert receipt['remaining_cases'] == 31 and receipt['required_free_bytes'] == 505*gib


def test_manifest_exposes_current_scope_without_execution_or_generalization_claim(monkeypatch):
    monkeypatch.setattr(study, 'specification', lambda i: dict(scene_id=str(i), procedural_seed=100+i, appearance_seed=200+i))
    manifest = study.manifest()
    assert len(manifest['ordered_cases']) == 32 and manifest['independent_layout_units'] == 8
    assert manifest['navigation_ticks'] == 4000 and manifest['training_seed_replications'] == 1
    assert manifest['completed_measured_plane_comparison_review_required_before_launch']
    for key in ('current_pair_arm_is_fully_memoryless', 'isolated_prediction_ranking_ablation',
            'nominal_predictive_arm_included', 'no_rgb_direct_development_reference_included',
            'population_execution_permitted', 'previous_32_case_definition_modified',
            'native_execution', 'navigation_qualified', 'goal_achieved'):
        assert manifest[key] is False
