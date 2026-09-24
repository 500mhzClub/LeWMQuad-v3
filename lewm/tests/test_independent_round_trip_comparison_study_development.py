"""Treatment integrity, paired population, resource boundaries and state isolation."""
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest

from lewm import independent_round_trip_comparison_study_development as study
from scripts import independent_round_trip_controller_factory_development as factory
from lewm.current_observation_planning_map_development import CurrentObservationPlanningMap
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportMap


def test_complete_paired_population_and_balanced_prospective_order():
    manifest = study.manifest()
    assert len({case.name for case in study.CASES}) == 32
    assert manifest['independent_layout_units'] == 8
    positions = {arm.name: Counter() for arm in study.ARMS}
    for index in range(8):
        rows = manifest['ordered_cases'][4*index:4*index+4]
        assert {row['arm_name'] for row in rows} == {arm.name for arm in study.ARMS}
        assert {row['layout_index'] for row in rows} == {index}
        assert {row['physics_seed'] for row in rows} == {2026092700 + index}
        assert {row['appearance_seed'] for row in rows} == {2026092800 + index}
        for position, row in enumerate(rows):
            positions[row['arm_name']][position] += 1
            # Factory/roster metadata must not pass private topology to policy.
            assert not {'evaluation_layout', 'links', 'route'} & row.keys()
    assert all(count == Counter({0:2, 1:2, 2:2, 3:2}) for count in positions.values())
    assert not manifest['native_execution'] and not manifest['execution_protocol_frozen']
    assert manifest['retain_scientific_failures'] and not manifest['outcome_based_layout_replacement']


@pytest.mark.parametrize('fault', ['layout', 'boolean_layout', 'name', 'arm', 'dictionary'])
def test_misassigned_or_out_of_roster_case_rejected(fault):
    case = study.CASES[0]
    if fault == 'layout': case = replace(case, layout_index=8)
    elif fault == 'boolean_layout': case = replace(case, layout_index=False)
    elif fault == 'name': case = replace(case, name='replacement_episode')
    elif fault == 'arm': case = replace(case, arm_name='best_transfer_model')
    else: case = vars(case)
    with pytest.raises(ValueError, match='roster'): study.require_case(case)


def test_whole_population_capacity_and_ordered_prefix():
    resources = dict(memory_available_bytes=32*1024**3, artifact_free_bytes=392*1024**3)
    assert study.resources_for(resources)['required_free_bytes'] == 392*1024**3
    resources['artifact_free_bytes'] -= 1
    with pytest.raises(ValueError, match='resource'): study.resources_for(resources)
    names = [case.name for case in study.CASES]
    assert study.resources_for(resources, names[:1])['remaining_cases'] == 31
    for prefix in (names[1:2], names[:1]*2, names, ['unplanned']):
        with pytest.raises(ValueError, match='prefix'): study.resources_for(resources, prefix)
    resources['memory_available_bytes'] -= 1
    with pytest.raises(ValueError, match='resource'): study.resources_for(resources, names[:31])


def test_reactive_factory_never_loads_model_and_instances_are_fresh(monkeypatch):
    def forbidden(*args): raise AssertionError('reactive model load')
    monkeypatch.setattr(factory, 'load_assigned', forbidden)
    case = next(case for case in study.CASES if case.arm_name == 'reactive')
    first, model = factory.create(case, object())
    second, _ = factory.create(case, object())
    assert model is None and not hasattr(first, 'model') and not hasattr(first, 'residual')
    assert first is not second and first.mapper is not second.mapper and first.memory is not second.memory
    assert type(first.mapper) is MeasuredFloorTransportMap


@pytest.mark.parametrize('value', [float('nan'), float('inf'), True, -1])
def test_invalid_resource_measurement_cannot_admit_population(value):
    with pytest.raises(ValueError, match='resource'):
        study.resources_for(dict(memory_available_bytes=value, artifact_free_bytes=392*1024**3))


def admitted_loader(monkeypatch, *, fault=None):
    calls = []
    def load(admission, name):
        calls.append(name)
        arm = next(arm for arm in study.ARMS if arm.model_name == name)
        state = arm.model_state_sha256 if fault != 'state' else 'wrong_state'
        model = SimpleNamespace(state_dict=lambda: state)
        return model, arm.condition if fault != 'condition' else 'direct', arm.variant if fault != 'variant' else 'no_rgb'
    monkeypatch.setattr(factory, 'load_assigned', load)
    monkeypatch.setattr(factory, 'state_digest', lambda state: state)
    return calls, dict(correction_result_sha256=study.CORRECTION_RESULT_SHA256)


def test_jepa_pair_shares_only_model_identity_and_retains_contact_scope(monkeypatch):
    calls, admission = admitted_loader(monkeypatch)
    cases = [next(case for case in study.CASES if case.arm_name == name)
        for name in ('persistent_jepa', 'current_pair_jepa')]
    (old, first_model), (new, second_model) = [factory.create(case, object(), correction_admission=admission) for case in cases]
    assert calls == ['seed_2026091001_full_jepa']*2 and first_model is not second_model
    assert type(old.mapper) is MeasuredFloorTransportMap and type(new.mapper) is CurrentObservationPlanningMap
    for field in ('mapper', 'memory', 'residual', 'mission', 'registration', 'motion', 'selector', 'history'):
        assert getattr(old, field) is not getattr(new, field)
    assert new.memory is new.mapper.surface and new.selector.residual is new.residual
    assert old.selector.condition == new.selector.condition == 'jepa'


def test_supervised_uses_same_persistent_controller_with_assigned_training_objective(monkeypatch):
    calls, admission = admitted_loader(monkeypatch)
    case = next(case for case in study.CASES if case.arm_name == 'persistent_supervised')
    controller, _ = factory.create(case, object(), correction_admission=admission)
    assert calls == ['seed_2026091001_full_supervised_rollout']
    assert type(controller) is factory.ResidualAnchoredContinuationController
    assert controller.selector.condition == 'supervised_rollout'


@pytest.mark.parametrize('fault', ['state', 'condition', 'variant', 'correction', 'absent'])
def test_wrong_model_or_admission_rejected_before_controller_use(monkeypatch, fault):
    _, admission = admitted_loader(monkeypatch, fault=fault)
    if fault == 'correction': admission['correction_result_sha256'] = 'substitute'
    if fault == 'absent': admission = None
    with pytest.raises(ValueError): factory.create(study.CASES[0], object(), correction_admission=admission)


def test_model_mutation_during_construction_rejected(monkeypatch):
    _, admission = admitted_loader(monkeypatch)
    original = factory.ResidualAnchoredContinuationController
    def mutate(model, *args, **kwargs):
        controller = original(model, *args, **kwargs)
        model.state_dict = lambda: 'mutated_state'
        return controller
    monkeypatch.setattr(factory, 'ResidualAnchoredContinuationController', mutate)
    with pytest.raises(ValueError, match='construction changed'):
        factory.create(study.CASES[0], object(), correction_admission=admission)
