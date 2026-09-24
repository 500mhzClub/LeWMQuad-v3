from __future__ import annotations

import hashlib
import json
import math

import pytest
import torch
import torch.nn.functional as F

from lewm.models import (
    direct_egocentric_bev_state_jepa_v3_coordinate_aware_film_unet_predictor
    as canonical_v3,
)
from lewm.models import (
    direct_egocentric_bev_state_jepa_v5_all_actions_state_delta_contrast
    as v5,
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


@pytest.fixture(scope="module")
def frozen_models(n320_encoder_state):
    caller_state = torch.random.get_rng_state().clone()
    model_v3 = canonical_v3.DirectEgocentricBevStateJepaV1(
        n320_encoder_state
    )
    assert torch.equal(torch.random.get_rng_state(), caller_state)
    model_v5 = v5.DirectEgocentricBevStateJepaV1(n320_encoder_state)
    assert torch.equal(torch.random.get_rng_state(), caller_state)
    return model_v3, model_v5


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _inputs(
    *,
    batch: int = 2,
    height: int = 4,
    width: int = 5,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(9921)
    current = torch.randn(
        batch,
        3,
        height,
        width,
        generator=generator,
    )
    return {
        "current_state_logits": current,
        "all_action_prediction_logits": torch.randn(
            batch,
            9,
            3,
            height,
            width,
            generator=generator,
        ),
        "target_current_logits": torch.randn(
            batch,
            3,
            height,
            width,
            generator=generator,
        ),
        "target_next_logits": torch.randn(
            batch,
            3,
            height,
            width,
            generator=generator,
        ),
        "executed_action_indices": torch.arange(batch) % 9,
    }


def _reference_auxiliary(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    current_probability = torch.softmax(
        inputs["current_state_logits"], dim=1
    )
    prediction_probability = torch.softmax(
        inputs["all_action_prediction_logits"], dim=2
    )
    predicted_delta = prediction_probability - current_probability[:, None]
    target_delta = (
        torch.softmax(inputs["target_next_logits"].detach(), dim=1)
        - torch.softmax(inputs["target_current_logits"].detach(), dim=1)
    ).detach()
    distances = (
        predicted_delta - target_delta[:, None]
    ).square().mean(dim=(2, 3, 4))
    scale = distances.mean(dim=1).detach().clamp_min(1e-4)
    logits = -distances / scale[:, None]
    return F.cross_entropy(
        logits,
        inputs["executed_action_indices"],
        reduction="mean",
    ) / math.log(9.0)


def test_exact_preregistered_formula_is_pure_and_parameter_free() -> None:
    inputs = _inputs()
    before = {name: value.clone() for name, value in inputs.items()}
    rng_before = torch.random.get_rng_state().clone()

    actual = v5.all_actions_state_delta_contrast_v5(**inputs)
    expected = _reference_auxiliary(inputs)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert actual.ndim == 0
    assert bool(torch.isfinite(actual))
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert all(torch.equal(inputs[name], value) for name, value in before.items())
    assert v5.STATE_DELTA_SCALE_FLOOR_V5 == 1e-4
    assert v5.STATE_DELTA_CONTRAST_WEIGHT_V5 == 1.0


def test_update_zero_auxiliary_is_exact_chance_and_c_increment() -> None:
    inputs = _inputs(batch=3)
    current = inputs["current_state_logits"]
    inputs["all_action_prediction_logits"] = (
        current[:, None].expand(-1, 9, -1, -1, -1).clone()
    )
    inputs["executed_action_indices"] = torch.tensor([0, 4, 8])

    auxiliary = v5.all_actions_state_delta_contrast_v5(**inputs)
    torch.testing.assert_close(
        auxiliary,
        torch.ones_like(auxiliary),
        rtol=1e-7,
        atol=1e-7,
    )
    frozen_v4_update_zero_c = auxiliary.new_tensor(0.9996682393430459)
    c_v5 = frozen_v4_update_zero_c + auxiliary
    assert 1.99 <= float(c_v5) <= 2.01
    torch.testing.assert_close(
        c_v5 - frozen_v4_update_zero_c,
        auxiliary,
        rtol=0.0,
        atol=0.0,
    )


def test_override_calls_frozen_objective_once_and_changes_only_total_and_c(
    frozen_models,
    monkeypatch,
) -> None:
    _model_v3, model = frozen_models
    inputs = _inputs(batch=2)
    current = inputs["current_state_logits"]
    predictions = current[:, None].expand(-1, 9, -1, -1, -1).clone()
    zero = current.new_zeros(())
    state = current.clone()
    candidate = current.new_zeros(2, 11)
    base = v5.DirectBevStateObjectiveV1(
        total=current.new_tensor(3.25),
        G=current.new_tensor(0.75),
        J=current.new_tensor(0.5),
        C=current.new_tensor(1.0),
        G_current=current.new_tensor(0.7),
        G_next=current.new_tensor(0.8),
        current_state_logits=state,
        next_online_state_logits=state.clone(),
        executed_prediction_logits=state.clone(),
        all_action_prediction_logits=predictions,
        target_next_logits=inputs["target_next_logits"],
        target_current_logits=inputs["target_current_logits"],
        target_mapped_negative_logits=state.clone(),
        action_energies=current.new_zeros(2, 9),
        action_logits=current.new_zeros(2, 9),
        action_nll_per_row=current.new_zeros(2),
        executed_energy=current.new_zeros(2),
        mapped_negative_energy=current.new_zeros(2),
        current_target_energy=current.new_zeros(2),
        candidate_energies=candidate,
        candidate_logits=candidate.clone(),
        candidate_mask=torch.ones(2, 11, dtype=torch.bool),
        candidate_counts=torch.full((2,), 11, dtype=torch.long),
        candidate_energy_scale=current.new_ones(2),
        conditional_nce_per_row=current.new_ones(2),
    )
    calls = 0

    def frozen_training_objective(_self, **_kwargs):
        nonlocal calls
        calls += 1
        return base

    monkeypatch.setattr(
        v5._v3.DirectEgocentricBevStateJepaV1,
        "training_objective",
        frozen_training_objective,
    )
    action_indices = inputs["executed_action_indices"]
    action_one_hot = F.one_hot(action_indices, num_classes=9).to(torch.float32)
    state_before = _state_digest(model)
    rng_before = torch.random.get_rng_state().clone()
    result = model.training_objective(
        current_rgb=torch.zeros(2, 3, 2, 2),
        next_rgb=torch.zeros(2, 3, 2, 2),
        fixed_negative_rgb=torch.zeros(2, 3, 2, 2),
        action_one_hot=action_one_hot,
        non_hold_mask=action_indices != v5.HOLD_ACTION_INDEX_V1,
        current_labels=torch.zeros(2, 2, 2, dtype=torch.long),
        next_labels=torch.zeros(2, 2, 2, dtype=torch.long),
    )
    auxiliary = v5.all_actions_state_delta_contrast_v5(
        current_state_logits=base.current_state_logits,
        all_action_prediction_logits=base.all_action_prediction_logits,
        target_current_logits=base.target_current_logits,
        target_next_logits=base.target_next_logits,
        executed_action_indices=action_indices,
    )

    assert calls == 1
    torch.testing.assert_close(result.total, base.total + auxiliary)
    torch.testing.assert_close(result.C, base.C + auxiliary)
    for field in base._fields:
        if field not in {"total", "C"}:
            original = getattr(base, field)
            updated = getattr(result, field)
            if isinstance(original, torch.Tensor):
                assert updated is original
            else:
                assert updated == original
    assert _state_digest(model) == state_before
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_initial_gradient_enters_action_conditioned_residual_not_identity(
    frozen_models,
) -> None:
    _model_v3, model = frozen_models
    predictor = model.predictor
    predictor.zero_grad(set_to_none=True)
    generator = torch.Generator().manual_seed(6205)
    current = torch.randn(
        1, 3, 64, 64, generator=generator, requires_grad=True
    )
    target_current = torch.randn(
        1, 3, 64, 64, generator=generator, requires_grad=True
    )
    target_next = torch.randn(
        1, 3, 64, 64, generator=generator, requires_grad=True
    )
    predictions = predictor.predict_all_actions(current)
    predictions.retain_grad()
    auxiliary = v5.all_actions_state_delta_contrast_v5(
        current_state_logits=current,
        all_action_prediction_logits=predictions,
        target_current_logits=target_current,
        target_next_logits=target_next,
        executed_action_indices=torch.tensor([4]),
    )
    auxiliary.backward()

    head_gradient = predictor.residual_head.weight.grad
    assert head_gradient is not None
    assert bool(torch.isfinite(head_gradient).all())
    assert float(head_gradient.abs().sum()) > 0.0
    assert predictions.grad is not None
    assert float(predictions.grad.abs().sum()) > 0.0
    assert current.grad is not None
    torch.testing.assert_close(
        current.grad,
        torch.zeros_like(current.grad),
        rtol=0.0,
        atol=2e-9,
    )
    assert target_current.grad is None
    assert target_next.grad is None


def test_v5_construction_is_state_and_inventory_identical_to_frozen_v3(
    frozen_models,
) -> None:
    model_v3, model_v5 = frozen_models
    assert issubclass(
        v5.DirectEgocentricBevStateJepaV1,
        v5._v3.DirectEgocentricBevStateJepaV1,
    )
    parameters_v3 = tuple(model_v3.named_parameters())
    parameters_v5 = tuple(model_v5.named_parameters())
    assert [name for name, _value in parameters_v5] == [
        name for name, _value in parameters_v3
    ]
    assert len(parameters_v5) == MODEL_PARAMETER_TENSOR_COUNT
    assert sum(value.numel() for _name, value in parameters_v5) == (
        MODEL_PARAMETER_COUNT
    )
    predictor_parameters = tuple(model_v5.predictor.named_parameters())
    assert len(predictor_parameters) == PREDICTOR_PARAMETER_TENSOR_COUNT
    assert sum(value.numel() for _name, value in predictor_parameters) == (
        PREDICTOR_PARAMETER_COUNT
    )
    assert [name for name, _value in model_v5.named_buffers()] == [
        name for name, _value in model_v3.named_buffers()
    ]
    assert [name for name, _value in model_v5.named_modules()] == [
        name for name, _value in model_v3.named_modules()
    ]
    state_v3 = model_v3.state_dict()
    state_v5 = model_v5.state_dict()
    assert state_v5.keys() == state_v3.keys()
    assert all(
        torch.equal(value, state_v3[name])
        for name, value in state_v5.items()
    )
    assert _state_digest(model_v5) == _state_digest(model_v3)
    assert torch.count_nonzero(model_v5.predictor.residual_head.weight) == 0
    assert torch.count_nonzero(model_v5.predictor.residual_head.bias) == 0


def test_helper_rejects_invalid_shapes_types_indices_and_nonfinite() -> None:
    valid = _inputs(batch=2)

    changed = dict(valid)
    changed["current_state_logits"] = valid["current_state_logits"][:, :2]
    with pytest.raises(ValueError, match="current_state_logits"):
        v5.all_actions_state_delta_contrast_v5(**changed)

    changed = dict(valid)
    changed["all_action_prediction_logits"] = (
        valid["all_action_prediction_logits"][:, :8]
    )
    with pytest.raises(ValueError, match="all_action_prediction_logits"):
        v5.all_actions_state_delta_contrast_v5(**changed)

    changed = dict(valid)
    changed["target_next_logits"] = valid["target_next_logits"].to(
        torch.float64
    )
    with pytest.raises(TypeError, match="floating dtype"):
        v5.all_actions_state_delta_contrast_v5(**changed)

    changed = dict(valid)
    changed["executed_action_indices"] = torch.tensor([0.0, 1.0])
    with pytest.raises(TypeError, match="executed_action_indices"):
        v5.all_actions_state_delta_contrast_v5(**changed)

    changed = dict(valid)
    changed["executed_action_indices"] = torch.tensor([0, 9])
    with pytest.raises(ValueError, match=r"\[0,8\]"):
        v5.all_actions_state_delta_contrast_v5(**changed)

    changed = dict(valid)
    changed["target_current_logits"] = valid[
        "target_current_logits"
    ].clone()
    changed["target_current_logits"][0, 0, 0, 0] = torch.nan
    with pytest.raises(FloatingPointError, match="target_current_logits"):
        v5.all_actions_state_delta_contrast_v5(**changed)
