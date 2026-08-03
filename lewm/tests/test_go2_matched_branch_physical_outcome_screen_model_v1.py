from __future__ import annotations

from collections import OrderedDict
import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.go2_matched_branch_physical_outcome_screen_v1 import (
    HIDDEN_WIDTH,
    INITIALIZATION_ORDER,
    INPUT_WIDTH,
    OUTPUT_WIDTH,
    PARAMETER_COUNT,
    PARAMETER_TENSOR_COUNT,
    PhysicalOutcomeMLPV1,
    initialize_physical_outcome_mlp_v1,
    physical_outcome_state_identity_v1,
)


SEED = 2_026_080_311


def _random_features(
    batch: int = 7,
    *,
    requires_grad: bool = False,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(91_731)
    return torch.randn(
        (batch, INPUT_WIDTH),
        generator=generator,
        dtype=torch.float32,
    ).requires_grad_(requires_grad)


def test_exact_inventory_and_dedicated_generator_initialization() -> None:
    caller_rng = torch.random.get_rng_state().clone()
    model = initialize_physical_outcome_mlp_v1(SEED)
    assert torch.equal(torch.random.get_rng_state(), caller_rng)

    parameters = dict(model.named_parameters())
    assert len(parameters) == PARAMETER_TENSOR_COUNT == 4
    assert PARAMETER_COUNT == 532
    assert sum(value.numel() for value in parameters.values()) == PARAMETER_COUNT
    assert {name: tuple(value.shape) for name, value in parameters.items()} == {
        "input_weight": (HIDDEN_WIDTH, INPUT_WIDTH),
        "input_bias": (HIDDEN_WIDTH,),
        "output_weight": (OUTPUT_WIDTH, HIDDEN_WIDTH),
        "output_bias": (OUTPUT_WIDTH,),
    }
    assert all(
        value.dtype == torch.float32 and value.requires_grad
        for value in parameters.values()
    )

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    for name in INITIALIZATION_ORDER:
        actual = getattr(model, name)
        expected = torch.empty_like(actual)
        nn.init.xavier_uniform_(expected, gain=1.0, generator=generator)
        assert torch.equal(actual, expected)
    assert torch.count_nonzero(model.input_bias) == 0
    assert torch.count_nonzero(model.output_bias) == 0


def test_direct_construction_is_zero_and_forward_matches_fixed_equation() -> None:
    zero_model = PhysicalOutcomeMLPV1()
    features = _random_features()
    assert torch.equal(
        zero_model(features),
        torch.zeros((features.shape[0], OUTPUT_WIDTH), dtype=torch.float32),
    )

    model = initialize_physical_outcome_mlp_v1(SEED)
    expected = F.linear(
        torch.tanh(F.linear(features, model.input_weight, model.input_bias)),
        model.output_weight,
        model.output_bias,
    )
    first = model(features)
    assert first.shape == (features.shape[0], OUTPUT_WIDTH)
    assert torch.equal(first, expected)
    assert torch.equal(first, model(features.clone()))


def test_all_parameters_and_input_receive_finite_gradients() -> None:
    model = initialize_physical_outcome_mlp_v1(SEED)
    features = _random_features(requires_grad=True)
    model(features).square().mean().backward()

    assert features.grad is not None
    assert bool(torch.isfinite(features.grad).all())
    assert torch.count_nonzero(features.grad) > 0
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
        assert torch.count_nonzero(parameter.grad) > 0, name


def test_seed_and_state_identity_are_exact_and_mapping_order_independent() -> None:
    first = initialize_physical_outcome_mlp_v1(SEED)
    replay = initialize_physical_outcome_mlp_v1(SEED)
    different = initialize_physical_outcome_mlp_v1(SEED + 1)

    identity = physical_outcome_state_identity_v1(first)
    assert len(identity) == 64
    assert identity == physical_outcome_state_identity_v1(replay)
    assert identity != physical_outcome_state_identity_v1(different)
    assert all(
        torch.equal(value, replay.state_dict()[name])
        for name, value in first.state_dict().items()
    )

    reversed_state = OrderedDict(reversed(tuple(first.state_dict().items())))
    assert physical_outcome_state_identity_v1(reversed_state) == identity
    mutated_state = copy.deepcopy(first.state_dict())
    mutated_state["input_weight"][0, 0] += 1.0
    assert physical_outcome_state_identity_v1(mutated_state) != identity


@pytest.mark.parametrize(
    ("features", "error"),
    (
        (torch.zeros((1, INPUT_WIDTH - 1)), ValueError),
        (torch.zeros((1, INPUT_WIDTH + 1)), ValueError),
        (torch.zeros((0, INPUT_WIDTH)), ValueError),
        (torch.zeros((INPUT_WIDTH,)), ValueError),
        (torch.zeros((1, INPUT_WIDTH, 1)), ValueError),
        (torch.zeros((1, INPUT_WIDTH), dtype=torch.float64), TypeError),
    ),
)
def test_input_contract_rejects_wrong_shape_and_dtype(
    features: torch.Tensor,
    error: type[Exception],
) -> None:
    model = initialize_physical_outcome_mlp_v1(SEED)
    with pytest.raises(error):
        model(features)


def test_nonfinite_inputs_parameters_and_invalid_states_fail_closed() -> None:
    model = initialize_physical_outcome_mlp_v1(SEED)
    features = _random_features(batch=1)

    bad_features = features.clone()
    bad_features[0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        model(bad_features)

    with torch.no_grad():
        model.output_weight[0, 0] = float("inf")
    with pytest.raises(FloatingPointError):
        model(features)
    with pytest.raises(FloatingPointError):
        physical_outcome_state_identity_v1(model)

    valid = initialize_physical_outcome_mlp_v1(SEED).state_dict()
    missing = dict(valid)
    missing.pop("input_bias")
    with pytest.raises(ValueError):
        physical_outcome_state_identity_v1(missing)
    wrong_shape = copy.deepcopy(valid)
    wrong_shape["output_bias"] = torch.zeros((OUTPUT_WIDTH + 1,))
    with pytest.raises(ValueError):
        physical_outcome_state_identity_v1(wrong_shape)
    wrong_dtype = copy.deepcopy(valid)
    wrong_dtype["input_bias"] = wrong_dtype["input_bias"].double()
    with pytest.raises(TypeError):
        physical_outcome_state_identity_v1(wrong_dtype)


def test_seed_validation_is_strict() -> None:
    with pytest.raises(TypeError):
        initialize_physical_outcome_mlp_v1(True)
    with pytest.raises(TypeError):
        initialize_physical_outcome_mlp_v1(1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        initialize_physical_outcome_mlp_v1(-1)
    with pytest.raises(ValueError):
        initialize_physical_outcome_mlp_v1(2**63)
