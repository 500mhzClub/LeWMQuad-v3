from __future__ import annotations

import inspect
import math

import pytest
import torch

from lewm.models.direct_egocentric_bev_state_jepa_v1 import (
    ACTION_VOCABULARY_V1,
    DirectEgocentricBevStateJepaV1,
    DirectEgocentricBevStateJepaV1Config,
    direct_bev_state_objective_v1,
    hard_hierarchical_raster_loss_v1,
)
from lewm.models.encoders import VisionEncoder


@pytest.fixture(scope="module")
def n320_encoder_state() -> dict[str, torch.Tensor]:
    torch.manual_seed(1701)
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


@pytest.fixture(scope="module")
def model(
    n320_encoder_state: dict[str, torch.Tensor],
) -> DirectEgocentricBevStateJepaV1:
    return DirectEgocentricBevStateJepaV1(n320_encoder_state).eval()


def test_exact_shape_and_three_channel_bottleneck(
    model: DirectEgocentricBevStateJepaV1,
) -> None:
    assert model.config == DirectEgocentricBevStateJepaV1Config()
    assert model.action_vocabulary == ACTION_VOCABULARY_V1
    assert model.encoder.image_size == 112
    assert model.encoder.patch_size == 7
    assert len(model.encoder.blocks) == 6
    assert model.bev_decoder.cross_attention.embed_dim == 64
    assert model.bev_decoder.bev_size == (64, 64)
    assert model.state_head.in_channels == 64
    assert model.state_head.out_channels == 3

    with torch.no_grad():
        logits = model.online_state(torch.zeros(1, 3, 112, 112))
    assert logits.shape == (1, 3, 64, 64)

    # The transition sees two three-channel tensors after the action condition;
    # there is no decoder-feature input or hidden-state skip.
    assert model.predictor.condition[0].in_features == 12
    assert model.predictor.condition[-1].out_features == 3
    assert model.predictor.net[0].in_channels == 6
    assert model.predictor.net[-1].out_channels == 3
    assert list(inspect.signature(model.predict_state).parameters) == [
        "current_state_logits",
        "action_one_hot",
    ]
    assert list(inspect.signature(model.predict_from_rgb).parameters) == [
        "current_rgb",
        "action_one_hot",
    ]
    with pytest.raises(TypeError):
        model.predict_state(  # type: ignore[call-arg]
            logits,
            torch.eye(9)[:1],
            commanded_delta_pose_current=torch.zeros(1, 3),
        )
    assert not any(
        term in name.lower()
        for name, _module in model.named_modules()
        for term in ("camera", "ray", "depth", "raster", "warp", "pose")
    )


def test_update_zero_is_exact_persistence_and_action_symmetric(
    model: DirectEgocentricBevStateJepaV1,
) -> None:
    current = torch.randn(2, 3, 5, 7)
    predictions = model.predict_all_actions_from_state(current)
    assert predictions.shape == (2, 9, 3, 5, 7)
    for action in range(9):
        assert torch.equal(predictions[:, action], current)
    assert torch.count_nonzero(model.predictor.net[-1].weight) == 0
    assert torch.count_nonzero(model.predictor.net[-1].bias) == 0


def test_n320_validation_and_fresh_initialization_preserve_rng(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    torch.manual_seed(9001)
    caller_rng = torch.random.get_rng_state().clone()
    first = DirectEgocentricBevStateJepaV1(n320_encoder_state)
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    first_fresh = {
        name: value.detach().clone()
        for name, value in (
            list(first.bev_decoder.state_dict().items())
            + [(f"state_head.{k}", v) for k, v in first.state_head.state_dict().items()]
            + [(f"predictor.{k}", v) for k, v in first.predictor.state_dict().items()]
        )
    }

    torch.manual_seed(42)
    second = DirectEgocentricBevStateJepaV1(n320_encoder_state)
    second_fresh = {
        name: value
        for name, value in (
            list(second.bev_decoder.state_dict().items())
            + [(f"state_head.{k}", v) for k, v in second.state_head.state_dict().items()]
            + [(f"predictor.{k}", v) for k, v in second.predictor.state_dict().items()]
        )
    }
    assert first_fresh.keys() == second_fresh.keys()
    assert all(
        torch.equal(value, second_fresh[name])
        for name, value in first_fresh.items()
    )

    missing = dict(n320_encoder_state)
    missing.pop(next(iter(missing)))
    before_failure = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="N320 encoder state keys changed"):
        DirectEgocentricBevStateJepaV1(missing)
    assert torch.equal(torch.random.get_rng_state(), before_failure)


def test_target_inventory_isolated_hard_synced_and_ema_updated(
    n320_encoder_state: dict[str, torch.Tensor],
) -> None:
    model = DirectEgocentricBevStateJepaV1(n320_encoder_state)
    for target, online in zip(
        model._target_modules(), model._online_modules(), strict=True
    ):
        assert target.state_dict().keys() == online.state_dict().keys()
        assert all(
            torch.equal(value, online.state_dict()[name])
            for name, value in target.state_dict().items()
        )
        assert not any(parameter.requires_grad for parameter in target.parameters())

    old_target = model.target_state_head.bias.detach().clone()
    with torch.no_grad():
        model.state_head.bias.add_(2.0)
    changed_online = model.state_head.bias.detach().clone()
    assert torch.equal(model.target_state_head.bias, old_target)
    model.update_target_ema_after_optimizer_step()
    expected = 0.996 * old_target + 0.004 * changed_online
    assert torch.allclose(model.target_state_head.bias, expected, rtol=0.0, atol=1e-7)
    assert int(model.ema_update_count) == 1
    model.train()
    assert all(not module.training for module in model._target_modules())
    assert not any(
        name.startswith("target_") and "predictor" in name
        for name, _module in model.named_modules()
    )


def _small_objective_inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1234)
    current = torch.randn(2, 3, 2, 3, generator=generator, requires_grad=True)
    next_online = torch.randn(
        2, 3, 2, 3, generator=generator, requires_grad=True
    )
    target_next = torch.randn(
        2, 3, 2, 3, generator=generator, requires_grad=True
    )
    target_current = torch.randn(
        2, 3, 2, 3, generator=generator, requires_grad=True
    )
    target_mapped = torch.randn(
        2, 3, 2, 3, generator=generator, requires_grad=True
    )
    return {
        "current": current,
        "next_online": next_online,
        "target_next": target_next,
        "target_current": target_current,
        "target_mapped": target_mapped,
    }


def test_objective_arithmetic_candidate_counts_and_intended_gradients(
    model: DirectEgocentricBevStateJepaV1,
) -> None:
    values = _small_objective_inputs()
    model.zero_grad(set_to_none=True)
    predictions = model.predict_all_actions_from_state(values["current"])
    labels = torch.tensor(
        [
            [[0, 1, 2], [2, 1, 0]],
            [[2, 0, 1], [1, 2, 0]],
        ],
        dtype=torch.uint8,
    )
    executed = torch.tensor([6, 2], dtype=torch.long)
    non_hold = torch.tensor([False, True])
    objective = direct_bev_state_objective_v1(
        current_state_logits=values["current"],
        next_online_state_logits=values["next_online"],
        all_action_prediction_logits=predictions,
        target_next_logits=values["target_next"],
        target_current_logits=values["target_current"],
        target_mapped_negative_logits=values["target_mapped"],
        current_labels=labels,
        next_labels=labels.flip(0),
        executed_action_indices=executed,
        non_hold_mask=non_hold,
    )
    assert torch.equal(objective.candidate_counts, torch.tensor([10, 11]))
    assert torch.equal(
        objective.candidate_mask.sum(dim=1), objective.candidate_counts
    )
    expected_scale = (
        (
            objective.candidate_energies
            * objective.candidate_mask.to(objective.candidate_energies.dtype)
        ).sum(dim=1)
        / objective.candidate_counts
    ).detach().clamp_min(1e-6)
    assert torch.equal(objective.candidate_energy_scale, expected_scale)
    expected_c = (
        (
            torch.logsumexp(objective.candidate_logits, dim=1)
            - objective.candidate_logits[:, 0]
        )
        / torch.log(objective.candidate_counts.float())
    ).mean()
    assert torch.allclose(objective.C, expected_c)
    assert torch.allclose(
        objective.total,
        objective.G / math.log(2.0)
        + objective.J / math.log(2.0)
        + objective.C,
    )
    assert objective.action_energies.shape == (2, 9)
    assert objective.action_logits.shape == (2, 9)
    assert objective.action_nll_per_row.shape == (2,)
    rows = torch.arange(2)
    assert torch.equal(
        objective.executed_energy,
        objective.action_energies[rows, executed],
    )
    assert torch.equal(
        objective.mapped_negative_energy,
        objective.candidate_energies[:, 9],
    )
    assert torch.equal(
        objective.current_target_energy,
        objective.candidate_energies[:, 10],
    )
    assert not objective.target_next_logits.requires_grad
    assert not objective.target_current_logits.requires_grad
    assert not objective.target_mapped_negative_logits.requires_grad

    objective.total.backward()
    assert values["current"].grad is not None
    assert torch.count_nonzero(values["current"].grad) > 0
    assert values["next_online"].grad is not None
    assert torch.count_nonzero(values["next_online"].grad) > 0
    assert values["target_next"].grad is None
    assert values["target_current"].grad is None
    assert values["target_mapped"].grad is None
    assert model.predictor.net[-1].weight.grad is not None
    assert torch.count_nonzero(model.predictor.net[-1].weight.grad) > 0


def test_training_and_diagnostic_calls_are_isolated(
    model: DirectEgocentricBevStateJepaV1,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_rgb = torch.zeros(1, 3, 112, 112)
    next_rgb = torch.ones(1, 3, 112, 112)
    fixed_rgb = torch.full((1, 3, 112, 112), 2.0)
    labels = torch.tensor([[[0, 1], [2, 0]]], dtype=torch.uint8)
    action = torch.eye(9)[6:7]
    non_hold = torch.tensor([False])
    online_calls: list[tuple[torch.Tensor, bool]] = []
    target_calls: list[tuple[torch.Tensor, bool]] = []

    def fake_online(rgb: torch.Tensor) -> torch.Tensor:
        online_calls.append((rgb, torch.is_grad_enabled()))
        value = 0.1 if rgb is current_rgb else 0.2
        return torch.full((1, 3, 2, 2), value, requires_grad=torch.is_grad_enabled())

    @torch.no_grad()
    def fake_target(rgb: torch.Tensor) -> torch.Tensor:
        target_calls.append((rgb, torch.is_grad_enabled()))
        return torch.full((1, 3, 2, 2), 0.3)

    monkeypatch.setattr(model, "online_state", fake_online)
    monkeypatch.setattr(model, "target_state", fake_target)
    objective = model.training_objective(
        current_rgb=current_rgb,
        next_rgb=next_rgb,
        fixed_negative_rgb=fixed_rgb,
        action_one_hot=action,
        non_hold_mask=non_hold,
        current_labels=labels,
        next_labels=labels,
    )
    assert [rgb for rgb, _enabled in online_calls] == [current_rgb, next_rgb]
    assert all(enabled for _rgb, enabled in online_calls)
    assert [rgb for rgb, _enabled in target_calls] == [
        next_rgb,
        current_rgb,
        fixed_rgb,
    ]
    assert not any(enabled for _rgb, enabled in target_calls)
    assert objective.candidate_counts.tolist() == [10]

    online_calls.clear()
    control = model.wrong_rgb_grounding_control(
        next_rgb=next_rgb,
        fixed_negative_rgb=fixed_rgb,
        next_labels=labels,
    )
    assert [rgb for rgb, _enabled in online_calls] == [next_rgb, fixed_rgb]
    assert not any(enabled for _rgb, enabled in online_calls)
    assert control.correct_next_loss_per_row.shape == (1,)
    assert control.mapped_negative_loss_per_row.shape == (1,)


def test_full_frozen_training_call_integrates_on_cpu(
    model: DirectEgocentricBevStateJepaV1,
) -> None:
    current = torch.zeros(1, 3, 112, 112)
    next_rgb = torch.full_like(current, 0.25)
    mapped = torch.full_like(current, 0.75)
    labels = (torch.arange(64 * 64).reshape(1, 64, 64) % 3).to(torch.uint8)
    with torch.no_grad():
        objective = model.training_objective(
            current_rgb=current,
            next_rgb=next_rgb,
            fixed_negative_rgb=mapped,
            action_one_hot=torch.eye(9)[6:7],
            non_hold_mask=torch.tensor([False]),
            current_labels=labels,
            next_labels=labels.flip(-1),
        )
    assert objective.current_state_logits.shape == (1, 3, 64, 64)
    assert objective.next_online_state_logits.shape == (1, 3, 64, 64)
    assert objective.all_action_prediction_logits.shape == (1, 9, 3, 64, 64)
    assert objective.candidate_counts.tolist() == [10]
    assert objective.candidate_energy_scale.shape == (1,)
    assert torch.isfinite(objective.total)


def test_hard_grounding_uses_occupied_then_conditional_free() -> None:
    logits = torch.zeros(1, 3, 1, 3)
    labels = torch.tensor([[[0, 1, 2]]], dtype=torch.uint8)
    loss = hard_hierarchical_raster_loss_v1(logits, labels)
    expected_occupied = 0.5 * (-math.log(2.0 / 3.0) - math.log(1.0 / 3.0))
    expected_free = math.log(2.0)
    assert loss.occupied.item() == pytest.approx(expected_occupied)
    assert loss.free_given_not_occupied.item() == pytest.approx(expected_free)
    assert loss.total.item() == pytest.approx(
        0.5 * expected_occupied + 0.5 * expected_free
    )
