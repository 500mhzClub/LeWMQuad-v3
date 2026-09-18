"""No-inference enforcement and accurate full-audit attribution."""
import pytest
import torch

from scripts import nominal_measured_plane_maze_development as pipeline


def test_forward_guard_rejects_root_and_child_calls_and_restores_hooks():
    model = torch.nn.Sequential(torch.nn.Linear(1, 1)).eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    for target in (model, model[0]):
        with pytest.raises(RuntimeError, match='must not execute'):
            with pipeline.forbid_model_forward(model): target(torch.ones(1, 1))
        assert all(not module._forward_pre_hooks for module in model.modules())
    with pytest.raises(ValueError, match='attempted model inference'):
        with pipeline.forbid_model_forward(model):
            try: model(torch.ones(1, 1))
            except RuntimeError: pass
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    assert all(not module._forward_pre_hooks for module in model.modules())


def test_forward_guard_preserves_preexisting_hooks_and_unrelated_exceptions():
    model = torch.nn.Linear(1, 1)
    handle = model.register_forward_pre_hook(lambda *args: None)
    before = dict(model._forward_pre_hooks)
    with pytest.raises(KeyError):
        with pipeline.forbid_model_forward(model): raise KeyError('existing failure')
    assert dict(model._forward_pre_hooks) == before
    handle.remove()


def test_private_collection_and_audit_keep_all_original_function_code():
    for new, old in ((pipeline._collect, pipeline.original.collect), (pipeline._audit, pipeline.original.audit)):
        assert new.__code__ is old.__code__
        assert new.__globals__['ResidualAnchoredContinuationController'] is pipeline.NominalMeasuredPlaneController
        assert all(new.__globals__[key] is value for key, value in old.__globals__.items()
            if key != 'ResidualAnchoredContinuationController')


def test_collected_persistent_result_is_unchanged_and_audit_does_not_claim_model_replay(monkeypatch):
    model = torch.nn.Linear(1, 1)
    persisted = {'collection': 'fixed'}
    monkeypatch.setattr(pipeline, '_collect', lambda *args, **kwargs: persisted)
    assert pipeline.collect(model=model) is persisted
    monkeypatch.setattr(pipeline, '_audit', lambda *args, **kwargs: {'raw_model_command_replay_pass': True,
        'raw_sensor_reconstruction_pass': True, 'raw_command_audit_pass': True, 'model_state_unchanged': True})
    audit = pipeline.audit(model=model)
    assert not audit['raw_model_command_replay_pass']
    assert audit['raw_controller_command_replay_pass'] and audit['raw_nominal_forecast_command_replay_pass']
    assert not audit['high_level_world_model_used'] and audit['actual_learned_model_forward_calls'] == 0
    assert audit['nominal_predictive_controller'] and not audit['fully_nonpredictive_controller']


def test_constructor_cannot_select_learned_mode():
    from lewm.tests.test_measured_plane_comparator_controllers_development import MISSION
    with pytest.raises(TypeError):
        pipeline.NominalMeasuredPlaneController(object(), object(), forecast_source='frozen_world_model',
            public_mission=MISSION, navigation_ticks=40, condition='direct', variant='no_rgb', persistent=True)
