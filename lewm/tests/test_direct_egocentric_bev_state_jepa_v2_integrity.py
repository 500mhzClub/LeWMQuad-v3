from __future__ import annotations

import pytest
import torch

from lewm.models import direct_egocentric_bev_state_jepa_v1 as v1
from lewm.models import direct_egocentric_bev_state_jepa_v2_integrity as v2
from lewm.models.encoders import VisionEncoder


@pytest.fixture(scope="module")
def n320_encoder_state() -> dict[str, torch.Tensor]:
    generator_state = torch.random.get_rng_state().clone()
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
        torch.random.set_rng_state(generator_state)


def test_v2_is_bitwise_identical_to_v1_and_preserves_inventories(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    model_v1 = v1.DirectEgocentricBevStateJepaV1(n320_encoder_state)
    model_v2 = v2.DirectEgocentricBevStateJepaV1(n320_encoder_state)

    state_v1 = model_v1.state_dict()
    state_v2 = model_v2.state_dict()
    assert list(state_v2) == list(state_v1)
    assert list(state_v2._metadata) == list(state_v1._metadata)
    assert state_v2._metadata == state_v1._metadata
    assert all(
        torch.equal(value, state_v1[name])
        for name, value in state_v2.items()
    )

    parameter_rows_v1 = list(model_v1.named_parameters())
    parameter_rows_v2 = list(model_v2.named_parameters())
    assert [name for name, _ in parameter_rows_v2] == [
        name for name, _ in parameter_rows_v1
    ]
    parameters_v1 = dict(parameter_rows_v1)
    parameters_v2 = dict(parameter_rows_v2)
    assert {
        name: value.numel() for name, value in parameters_v2.items()
    } == {
        name: value.numel() for name, value in parameters_v1.items()
    }
    assert [name for name, _ in model_v2.named_buffers()] == [
        name for name, _ in model_v1.named_buffers()
    ]
    assert [name for name, _ in model_v2.named_modules()] == [
        name for name, _ in model_v1.named_modules()
    ]
    assert torch.count_nonzero(model_v2.predictor.net[-1].weight) == 0
    assert torch.count_nonzero(model_v2.predictor.net[-1].bias) == 0
    assert int(model_v2.ema_update_count) == 0
    for target, online in zip(
        model_v2._target_modules(),
        model_v2._online_modules(),
        strict=True,
    ):
        assert target.state_dict().keys() == online.state_dict().keys()
        assert all(
            torch.equal(value, online.state_dict()[name])
            for name, value in target.state_dict().items()
        )
        assert not any(parameter.requires_grad for parameter in target.parameters())


def test_v2_preserves_caller_cpu_rng(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    torch.random.default_generator.manual_seed(9001)
    caller_state = torch.random.get_rng_state().clone()
    v2.DirectEgocentricBevStateJepaV1(n320_encoder_state)
    assert torch.equal(torch.random.get_rng_state(), caller_state)


def test_v2_avoids_accelerator_seed_while_v1_exposes_the_integrity_failure(
    n320_encoder_state: dict[str, torch.Tensor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accelerator_seeds: list[int] = []
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda seed: accelerator_seeds.append(int(seed)),
    )

    v1.DirectEgocentricBevStateJepaV1(n320_encoder_state)
    assert accelerator_seeds == [20260712]
    accelerator_seeds.clear()

    v2.DirectEgocentricBevStateJepaV1(n320_encoder_state)
    assert accelerator_seeds == []


def test_v2_reexports_runner_model_api() -> None:
    assert (
        v2.DirectEgocentricBevStateJepaV1Config
        is v1.DirectEgocentricBevStateJepaV1Config
    )
    assert v2.direct_bev_state_objective_v1 is v1.direct_bev_state_objective_v1
    assert v2._hard_hierarchical_loss_per_row is v1._hard_hierarchical_loss_per_row
    assert set(v1.__all__).issubset(v2.__all__)
