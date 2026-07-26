from __future__ import annotations

import copy
import hashlib
import inspect
import json

import pytest
import torch
import torch.nn as nn

from lewm.models import direct_egocentric_bev_state_jepa_v2_integrity as v2
from lewm.models import (
    direct_egocentric_bev_state_jepa_v3_coordinate_aware_film_unet_predictor
    as v3,
)
from lewm.models.encoders import VisionEncoder


PREDICTOR_PARAMETER_COUNT = 317_107
PREDICTOR_PARAMETER_TENSOR_COUNT = 79
MODEL_PARAMETER_COUNT = 6_552_249
MODEL_PARAMETER_TENSOR_COUNT = 277
PREDICTOR_ORDERED_PARAMETER_NAME_SHA256 = (
    "ebbd0bb384b09862c867338b39b4ffcfa4072e43730451f0eee337be3167fad2"
)
PREDICTOR_ORDERED_INVENTORY_SHA256 = (
    "5c8cac4bb77b3669894b04a7def61fe8f35ee2f7cb84bb2e38c0efdb8ab35665"
)
PREDICTOR_FULLY_QUALIFIED_ORDERED_NAME_SHA256 = (
    "0398031cb776c10a23b14c7935d2566f4a3087175213e87b49c2a05cadf6e1dd"
)


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
def models(n320_encoder_state):
    return (
        v2.DirectEgocentricBevStateJepaV1(n320_encoder_state),
        v3.DirectEgocentricBevStateJepaV1(n320_encoder_state),
    )


def _block_names(prefix: str) -> list[str]:
    return [
        f"{prefix}.conv1.weight",
        f"{prefix}.conv1.bias",
        f"{prefix}.norm1.weight",
        f"{prefix}.norm1.bias",
        f"{prefix}.conv2.weight",
        f"{prefix}.conv2.bias",
        f"{prefix}.norm2.weight",
        f"{prefix}.norm2.bias",
    ]


def _down_names(prefix: str) -> list[str]:
    return [
        f"{prefix}.0.weight",
        f"{prefix}.0.bias",
        f"{prefix}.1.weight",
        f"{prefix}.1.bias",
    ]


def _expected_predictor_names() -> tuple[str, ...]:
    names = ["action_embedding.weight"]
    names.extend(_block_names("enc64"))
    names.extend(_down_names("down32"))
    names.extend(_block_names("enc32"))
    names.extend(_down_names("down16"))
    names.extend(_block_names("enc16"))
    names.extend(_down_names("down8"))
    names.extend(_block_names("bottleneck"))
    names.extend(("film64.weight", "film64.bias"))
    names.extend(_block_names("dec16"))
    names.extend(("film48.weight", "film48.bias"))
    names.extend(_block_names("dec32"))
    names.extend(("film32.weight", "film32.bias"))
    names.extend(_block_names("dec64"))
    names.extend(("film16.weight", "film16.bias"))
    names.extend(("residual_head.weight", "residual_head.bias"))
    return tuple(names)


def _ordered_inventory(module: nn.Module) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "shape": list(parameter.shape),
            "numel": parameter.numel(),
        }
        for name, parameter in module.named_parameters()
    ]


def _inventory_sha256(module: nn.Module) -> str:
    raw = json.dumps(
        _ordered_inventory(module),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _ordered_name_sha256(module: nn.Module) -> str:
    raw = json.dumps(
        [name for name, _parameter in module.named_parameters()],
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def test_exact_predictor_and_model_parameter_inventories(models) -> None:
    _model_v2, model = models
    predictor_parameters = tuple(model.predictor.named_parameters())
    assert tuple(name for name, _parameter in predictor_parameters) == (
        _expected_predictor_names()
    )
    assert len(predictor_parameters) == PREDICTOR_PARAMETER_TENSOR_COUNT
    assert sum(
        parameter.numel() for _name, parameter in predictor_parameters
    ) == PREDICTOR_PARAMETER_COUNT

    model_parameters = tuple(model.named_parameters())
    assert len(model_parameters) == MODEL_PARAMETER_TENSOR_COUNT
    assert sum(
        parameter.numel() for _name, parameter in model_parameters
    ) == MODEL_PARAMETER_COUNT
    assert model.predictor.net[-1] is model.predictor.residual_head
    assert list(model.predictor._modules).count("residual_head") == 1
    assert (
        _ordered_name_sha256(model.predictor)
        == PREDICTOR_ORDERED_PARAMETER_NAME_SHA256
    )
    assert (
        _inventory_sha256(model.predictor)
        == PREDICTOR_ORDERED_INVENTORY_SHA256
    )
    fully_qualified_predictor_names = [
        name for name, _parameter in model_parameters
        if name.startswith("predictor.")
    ]
    assert hashlib.sha256(json.dumps(
        fully_qualified_predictor_names,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")).hexdigest() == (
        PREDICTOR_FULLY_QUALIFIED_ORDERED_NAME_SHA256
    )


def test_shared_perception_is_bitwise_identical_to_v2(models) -> None:
    model_v2, model_v3 = models
    for name in (
        "encoder",
        "bev_decoder",
        "state_head",
        "target_encoder",
        "target_bev_decoder",
        "target_state_head",
    ):
        state_v2 = getattr(model_v2, name).state_dict()
        state_v3 = getattr(model_v3, name).state_dict()
        assert state_v3.keys() == state_v2.keys()
        assert all(
            torch.equal(value, state_v2[key])
            for key, value in state_v3.items()
        )


def test_coordinate_planes_have_exact_row_column_orientation(models) -> None:
    _model_v2, model = models
    state = torch.zeros(2, 3, 64, 64)
    row, column = model.predictor._coordinate_planes(state)
    expected = torch.linspace(-1.0, 1.0, 64)
    assert row.shape == (2, 1, 64, 64)
    assert column.shape == (2, 1, 64, 64)
    assert torch.equal(row[0, 0, :, 0], expected)
    assert torch.equal(row[0, 0, :, 0], row[1, 0, :, -1])
    assert torch.equal(column[0, 0, 0, :], expected)
    assert torch.equal(column[0, 0, 0, :], column[1, 0, -1, :])
    assert tuple(model.predictor.named_buffers()) == ()


def test_zero_head_is_exact_persistence_for_every_action(models) -> None:
    _model_v2, model = models
    generator = torch.Generator().manual_seed(45)
    state = torch.randn(2, 3, 64, 64, generator=generator)
    predictions = model.predict_all_actions_from_state(state)
    assert predictions.shape == (2, 9, 3, 64, 64)
    assert all(
        torch.equal(predictions[:, action], state)
        for action in range(9)
    )
    assert torch.count_nonzero(model.predictor.residual_head.weight) == 0
    assert torch.count_nonzero(model.predictor.residual_head.bias) == 0
    assert model.action_vocabulary == (
        "arc_left",
        "arc_right",
        "backward",
        "forward_fast",
        "forward_medium",
        "forward_slow",
        "hold",
        "yaw_left",
        "yaw_right",
    )
    assert model.action_vocabulary == v3.ACTION_VOCABULARY_V1


def test_all_actions_encodes_once_and_matches_stacked_candidates(
    models,
    monkeypatch,
) -> None:
    _model_v2, model = models
    predictor = copy.deepcopy(model.predictor)
    with torch.no_grad():
        predictor.residual_head.weight.fill_(0.0001)
        predictor.residual_head.bias.copy_(torch.tensor([-0.01, 0.0, 0.01]))
    state = torch.linspace(-1.0, 1.0, 3 * 64 * 64).reshape(1, 3, 64, 64)

    original_encode = predictor._encode_shared
    encode_calls = 0

    def counted_encode(value: torch.Tensor):
        nonlocal encode_calls
        encode_calls += 1
        return original_encode(value)

    monkeypatch.setattr(predictor, "_encode_shared", counted_encode)
    all_actions = predictor.predict_all_actions(state)
    assert encode_calls == 1
    monkeypatch.setattr(predictor, "_encode_shared", original_encode)

    individual = []
    for action in range(9):
        action_one_hot = torch.eye(9)[action : action + 1]
        individual.append(predictor(state, action_one_hot))
    stacked = torch.stack(individual, dim=1)
    torch.testing.assert_close(all_actions, stacked, rtol=1e-6, atol=1e-6)
    assert any(
        not torch.equal(all_actions[:, 0], all_actions[:, action])
        for action in range(1, 9)
    )


def test_constructor_preserves_cpu_rng_and_never_seeds_cuda(
    n320_encoder_state,
    monkeypatch,
) -> None:
    accelerator_seeds: list[int] = []
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda seed: accelerator_seeds.append(int(seed)),
    )
    torch.random.default_generator.manual_seed(9001)
    caller_state = torch.random.get_rng_state().clone()
    v3.DirectEgocentricBevStateJepaV1(n320_encoder_state)
    assert torch.equal(torch.random.get_rng_state(), caller_state)
    assert accelerator_seeds == []


def test_predictor_has_no_forbidden_spatial_operator(models) -> None:
    _model_v2, model = models
    assert not any(
        isinstance(module, nn.MultiheadAttention)
        for module in model.predictor.modules()
    )
    source = inspect.getsource(v3._CoordinateAwareFilmUnetPredictorV3)
    assert "grid_sample" not in source
    assert "MultiheadAttention" not in source
