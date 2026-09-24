from __future__ import annotations

import hashlib
import json
import math

import pytest
import torch

from lewm.models import (
    direct_egocentric_bev_state_jepa_v3_coordinate_aware_film_unet_predictor
    as canonical_v3,
)
from lewm.models import (
    direct_egocentric_bev_state_jepa_v6_phase_separated_frozen_state_prediction
    as v6,
)
from lewm.models.encoders import VisionEncoder


PREDICTOR_PARAMETER_COUNT = 317_107
PREDICTOR_PARAMETER_TENSOR_COUNT = 79
MODEL_PARAMETER_COUNT = 6_552_249
MODEL_PARAMETER_TENSOR_COUNT = 277


@pytest.fixture(scope="module")
def n320_encoder_state() -> dict[str, torch.Tensor]:
    caller_state = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(1701)
        encoder = VisionEncoder(
            image_size=112,
            patch_size=7,
            hidden_dim=192,
            depth=6,
            n_heads=6,
            mlp_ratio=4,
            dropout=0.0,
        )
        return {
            name: value.detach().clone()
            for name, value in encoder.state_dict().items()
        }
    finally:
        torch.random.set_rng_state(caller_state)


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _module_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
    }


def _assert_module_state_equal(
    module: torch.nn.Module,
    expected: dict[str, torch.Tensor],
) -> None:
    actual = module.state_dict()
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[name], value) for name, value in expected.items())


def _objective_fixture() -> v6.DirectBevStateObjectiveV1:
    state = torch.zeros(2, 3, 4, 5)
    predictions = state[:, None].expand(-1, 9, -1, -1, -1).clone()
    candidates = torch.zeros(2, 11)
    return v6.DirectBevStateObjectiveV1(
        total=torch.tensor(99.0),
        G=torch.tensor(0.75),
        J=torch.tensor(0.5),
        C=torch.tensor(1.25),
        G_current=torch.tensor(0.7),
        G_next=torch.tensor(0.8),
        current_state_logits=state,
        next_online_state_logits=state.clone(),
        executed_prediction_logits=state.clone(),
        all_action_prediction_logits=predictions,
        target_next_logits=state.clone(),
        target_current_logits=state.clone(),
        target_mapped_negative_logits=state.clone(),
        action_energies=torch.zeros(2, 9),
        action_logits=torch.zeros(2, 9),
        action_nll_per_row=torch.zeros(2),
        executed_energy=torch.zeros(2),
        mapped_negative_energy=torch.zeros(2),
        current_target_energy=torch.zeros(2),
        candidate_energies=candidates,
        candidate_logits=candidates.clone(),
        candidate_mask=torch.ones(2, 11, dtype=torch.bool),
        candidate_counts=torch.full((2,), 11, dtype=torch.long),
        candidate_energy_scale=torch.ones(2),
        conditional_nce_per_row=torch.ones(2),
    )


def _unused_objective_arguments() -> dict[str, torch.Tensor]:
    actions = torch.tensor([0, 8])
    return {
        "current_rgb": torch.zeros(2, 3, 2, 2),
        "next_rgb": torch.zeros(2, 3, 2, 2),
        "fixed_negative_rgb": torch.zeros(2, 3, 2, 2),
        "action_one_hot": torch.nn.functional.one_hot(
            actions, num_classes=9
        ).to(torch.float32),
        "non_hold_mask": actions != v6.HOLD_ACTION_INDEX_V1,
        "current_labels": torch.zeros(2, 2, 2, dtype=torch.long),
        "next_labels": torch.zeros(2, 2, 2, dtype=torch.long),
    }


def test_construction_is_exact_frozen_v3_state_and_inventory(
    n320_encoder_state,
) -> None:
    caller_rng = torch.random.get_rng_state().clone()
    model_v3 = canonical_v3.DirectEgocentricBevStateJepaV1(
        n320_encoder_state
    )
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    model_v6 = v6.DirectEgocentricBevStateJepaV1(n320_encoder_state)
    assert torch.equal(torch.random.get_rng_state(), caller_rng)

    assert issubclass(
        v6.DirectEgocentricBevStateJepaV1,
        v6._v3.DirectEgocentricBevStateJepaV1,
    )
    assert not model_v6.phase_policy_armed_v6
    assert [name for name, _value in model_v6.named_parameters()] == [
        name for name, _value in model_v3.named_parameters()
    ]
    assert [name for name, _value in model_v6.named_buffers()] == [
        name for name, _value in model_v3.named_buffers()
    ]
    assert [name for name, _value in model_v6.named_modules()] == [
        name for name, _value in model_v3.named_modules()
    ]
    parameters = tuple(model_v6.named_parameters())
    assert len(parameters) == MODEL_PARAMETER_TENSOR_COUNT
    assert sum(value.numel() for _name, value in parameters) == (
        MODEL_PARAMETER_COUNT
    )
    predictor = tuple(model_v6.predictor.named_parameters())
    assert len(predictor) == PREDICTOR_PARAMETER_TENSOR_COUNT
    assert sum(value.numel() for _name, value in predictor) == (
        PREDICTOR_PARAMETER_COUNT
    )
    assert _state_digest(model_v6) == _state_digest(model_v3)
    assert all(parameter.requires_grad for parameter in model_v6.predictor.parameters())

    model_v6.arm_phase_schedule_v6()
    assert _state_digest(model_v6) == _state_digest(model_v3)
    assert model_v6.phase_counters_v6() == {
        "phase_policy_armed": True,
        "global_target_update_callback_count": 0,
        "target_update_callback_count": 0,
        "ema_arithmetic_update_count": 0,
        "boundary_hard_sync_count": 0,
        "phase_two_target_noop_count": 0,
        "perception_optimizer_update_count": 0,
        "predictor_optimizer_update_count": 0,
    }


def test_phase_totals_replace_only_total_and_probe_override_is_nonpersistent(
    n320_encoder_state,
    monkeypatch,
) -> None:
    model = v6.DirectEgocentricBevStateJepaV1(n320_encoder_state)
    base = _objective_fixture()
    calls = 0

    def frozen_training_objective(_self, **_kwargs):
        nonlocal calls
        calls += 1
        return base

    monkeypatch.setattr(
        v6._v3.DirectEgocentricBevStateJepaV1,
        "training_objective",
        frozen_training_objective,
    )
    with pytest.raises(RuntimeError, match="before phase policy was armed"):
        model.training_objective(**_unused_objective_arguments())

    state_before = _state_digest(model)
    model.arm_phase_schedule_v6()
    phase_one = model.training_objective(**_unused_objective_arguments())
    torch.testing.assert_close(
        phase_one.total,
        base.G / math.log(2.0),
        rtol=0.0,
        atol=0.0,
    )
    model.set_phase_override_for_integrity_probe_v6(v6.PHASE_TWO_V6)
    phase_two = model.training_objective(**_unused_objective_arguments())
    torch.testing.assert_close(
        phase_two.total,
        base.J / math.log(2.0) + base.C,
        rtol=0.0,
        atol=0.0,
    )
    assert calls == 2
    for result in (phase_one, phase_two):
        for field in base._fields:
            if field == "total":
                continue
            original = getattr(base, field)
            updated = getattr(result, field)
            if isinstance(original, torch.Tensor):
                assert updated is original
            else:
                assert updated == original
    assert model.phase_counters_v6()["target_update_callback_count"] == 0
    assert _state_digest(model) == state_before
    model.set_phase_override_for_integrity_probe_v6(None)
    assert model.active_phase_v6 == v6.PHASE_ONE_V6


def test_phase_trainability_and_modes_survive_observation_round_trips(
    n320_encoder_state,
) -> None:
    model = v6.DirectEgocentricBevStateJepaV1(n320_encoder_state)
    model.train(True)
    model.arm_phase_schedule_v6()

    assert model.training
    assert all(module.training for module in model._online_modules())
    assert not model.predictor.training
    assert all(
        parameter.requires_grad
        for module in model._online_modules()
        for parameter in module.parameters()
    )
    assert all(not parameter.requires_grad for parameter in model.predictor.parameters())
    assert all(not module.training for module in model._target_modules())

    model.eval()
    assert not model.training
    assert all(not module.training for module in model._online_modules())
    assert not model.predictor.training

    model.set_phase_override_for_integrity_probe_v6(v6.PHASE_TWO_V6)
    assert all(not module.training for module in model._online_modules())
    assert model.predictor.training
    assert all(
        not parameter.requires_grad
        for module in model._online_modules()
        for parameter in module.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.predictor.parameters())
    assert all(not module.training for module in model._target_modules())
    assert all(
        not parameter.requires_grad
        for module in model._target_modules()
        for parameter in module.parameters()
    )

    model.train(True)
    assert model.training
    assert all(not module.training for module in model._online_modules())
    assert model.predictor.training
    model.eval()
    assert not model.training
    assert all(not module.training for module in model._online_modules())
    assert model.predictor.training

    model.set_phase_override_for_integrity_probe_v6(None)
    assert model.active_phase_v6 == v6.PHASE_ONE_V6
    assert all(not module.training for module in model._online_modules())
    assert not model.predictor.training


def test_callback_400_syncs_once_and_later_callbacks_are_target_noops(
    n320_encoder_state,
) -> None:
    model = v6.DirectEgocentricBevStateJepaV1(n320_encoder_state)
    model.arm_phase_schedule_v6()
    with torch.no_grad():
        model.ema_update_count.fill_(399)
        next(model.encoder.parameters()).add_(0.125)
    object.__setattr__(model, "_v6_target_update_callback_count", 399)
    object.__setattr__(model, "_v6_ema_arithmetic_update_count", 399)
    model.apply_phase_policy_v6()

    model.update_target_ema_after_optimizer_step()
    counters_400 = model.phase_counters_v6()
    assert counters_400 == {
        "phase_policy_armed": True,
        "global_target_update_callback_count": 400,
        "target_update_callback_count": 400,
        "ema_arithmetic_update_count": 400,
        "boundary_hard_sync_count": 1,
        "phase_two_target_noop_count": 0,
        "perception_optimizer_update_count": 400,
        "predictor_optimizer_update_count": 0,
    }
    assert model.active_phase_v6 == v6.PHASE_TWO_V6
    for online, target in zip(
        model._online_modules(), model._target_modules(), strict=True
    ):
        _assert_module_state_equal(target, _module_state(online))
    assert all(not module.training for module in model._online_modules())
    assert model.predictor.training

    target_before = [_module_state(module) for module in model._target_modules()]
    model.update_target_ema_after_optimizer_step()
    assert model.phase_counters_v6() == {
        "phase_policy_armed": True,
        "global_target_update_callback_count": 401,
        "target_update_callback_count": 401,
        "ema_arithmetic_update_count": 400,
        "boundary_hard_sync_count": 1,
        "phase_two_target_noop_count": 1,
        "perception_optimizer_update_count": 400,
        "predictor_optimizer_update_count": 1,
    }
    for target, expected in zip(
        model._target_modules(), target_before, strict=True
    ):
        _assert_module_state_equal(target, expected)

    model.set_phase_override_for_integrity_probe_v6(v6.PHASE_ONE_V6)
    with pytest.raises(RuntimeError, match="forbidden during a probe"):
        model.update_target_ema_after_optimizer_step()
