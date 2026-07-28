from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    ROOT
    / "scripts/run_go2_rgb_swept_progress_survival_joint_jepa_v2_occupied_safety_aux.py"
)


def _load_runner() -> Any:
    name = "_test_go2_swept_progress_survival_v2_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _direct_present_binary_rows(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    binary = logits[:, 2] - torch.logsumexp(logits[:, :2], dim=1)
    target = labels == 2
    elements = F.binary_cross_entropy_with_logits(
        binary, target.to(logits.dtype), reduction="none"
    )
    rows = []
    for row in range(logits.shape[0]):
        groups = [
            elements[row][target[row] == state].mean()
            for state in (False, True)
            if bool((target[row] == state).any())
        ]
        rows.append(torch.stack(groups).mean())
    return torch.stack(rows)


def test_occupied_aux_matches_direct_row_balanced_current_next_math() -> None:
    current = torch.tensor(
        [
            [
                [[0.2, -0.4, 1.1]],
                [[-0.7, 0.3, 0.2]],
                [[0.8, -0.2, -0.5]],
            ],
            [
                [[0.9, -0.3, 0.0]],
                [[0.1, 0.6, -0.2]],
                [[-0.4, 0.4, 1.2]],
            ],
        ],
        dtype=torch.float64,
    )
    current_labels = torch.tensor([[[2, 0, 1]], [[2, 2, 2]]])
    next_logits = current.flip(0).mul(0.7).add(0.15)
    next_labels = torch.tensor([[[0, 1, 0]], [[1, 2, 0]]])

    observed = runner.occupied_safety_aux_loss_v2(
        current, current_labels, next_logits, next_labels
    )
    expected_current = _direct_present_binary_rows(current, current_labels)
    expected_next = _direct_present_binary_rows(next_logits, next_labels)
    expected = (
        0.5 * expected_current.mean() + 0.5 * expected_next.mean()
    ) / math.log(2.0)

    assert torch.equal(observed.current_per_row, expected_current)
    assert torch.equal(observed.next_per_row, expected_next)
    assert torch.equal(observed.loss, expected)
    # Equal weighting is per present binary class, not per pixel: the all-
    # occupied and all-rest rows remain valid one-class rows.
    assert observed.current_per_row.shape == (2,)
    assert observed.next_per_row.shape == (2,)


def test_occupied_aux_is_common_shift_invariant_and_reaches_all_three_logits() -> None:
    generator = torch.Generator().manual_seed(20260728)
    current = torch.randn((2, 3, 2, 3), generator=generator, requires_grad=True)
    next_logits = torch.randn((2, 3, 2, 3), generator=generator, requires_grad=True)
    labels = torch.tensor(
        [
            [[0, 1, 2], [2, 1, 0]],
            [[2, 2, 0], [1, 0, 2]],
        ]
    )
    shifted = runner.occupied_safety_aux_loss_v2(
        current + 19.0, labels, next_logits - 7.0, labels.roll(1, dims=2)
    ).loss
    unshifted = runner.occupied_safety_aux_loss_v2(
        current, labels, next_logits, labels.roll(1, dims=2)
    ).loss
    assert torch.allclose(shifted, unshifted, atol=2e-6, rtol=2e-6)

    unshifted.backward()
    assert current.grad is not None and next_logits.grad is not None
    assert bool((current.grad.abs().sum(dim=(0, 2, 3)) > 0).all())
    assert bool((next_logits.grad.abs().sum(dim=(0, 2, 3)) > 0).all())
    assert torch.allclose(
        current.grad.sum(dim=1), torch.zeros_like(current.grad[:, 0]), atol=1e-7
    )


@pytest.mark.parametrize(
    ("logits", "labels", "error"),
    [
        (torch.zeros((1, 2, 2, 2)), torch.zeros((1, 2, 2), dtype=torch.long), ValueError),
        (torch.zeros((1, 3, 2, 2)), torch.zeros((1, 2, 2)), TypeError),
        (
            torch.zeros((1, 3, 2, 2)),
            torch.full((1, 2, 2), 3, dtype=torch.long),
            ValueError,
        ),
    ],
)
def test_occupied_aux_rejects_invalid_semantic_contract(
    logits: torch.Tensor, labels: torch.Tensor, error: type[Exception]
) -> None:
    with pytest.raises(error):
        runner.occupied_safety_aux_loss_v2(logits, labels, logits, labels)


class _ChannelVector(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter(torch.randn(64, generator=generator) * 0.1)


class _SurvivalHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output = nn.Linear(64, 16)

    def forward(self, predicted: torch.Tensor) -> torch.Tensor:
        return self.output(predicted.mean(dim=(-2, -1)))


class _Predictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(43)
        self.action = nn.Parameter(torch.randn(9, 64, generator=generator) * 0.05)
        self.swept_progress_head = _SurvivalHead()


class _TinyJointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _ChannelVector(11)
        self.bev_lift = _ChannelVector(17)
        self.semantic_head = nn.Linear(64, 3)
        self.predictor = _Predictor()
        self.target_encoder = _ChannelVector(11)
        self.target_bev_lift = _ChannelVector(17)
        for module in (self.target_encoder, self.target_bev_lift):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.register_buffer("ema_update_count", torch.zeros((), dtype=torch.int64))
        self.register_buffer("channel_template", torch.linspace(-1.0, 1.0, 64))

    def _latent(
        self,
        rgb: torch.Tensor,
        encoder: _ChannelVector,
        lift: _ChannelVector,
    ) -> torch.Tensor:
        observation = rgb.mean(dim=(1, 2, 3))[:, None]
        channels = (
            encoder.weight[None]
            + torch.tanh(lift.weight)[None]
            + observation * self.channel_template[None]
        )
        return channels[:, :, None, None].expand(-1, -1, 2, 2)

    def encode_online(self, rgb: torch.Tensor) -> torch.Tensor:
        return self._latent(rgb, self.encoder, self.bev_lift)

    @torch.no_grad()
    def encode_target(self, rgb: torch.Tensor) -> torch.Tensor:
        return self._latent(rgb, self.target_encoder, self.target_bev_lift).detach()

    def semantic_logits_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return F.linear(
            latent.movedim(1, -1),
            self.semantic_head.weight,
            self.semantic_head.bias,
        ).movedim(-1, 1)

    def predict_all_actions_with_survival(self, current: torch.Tensor) -> Any:
        predicted = current[:, None] + self.predictor.action[None, :, :, None, None]
        return SimpleNamespace(
            predicted_latents=predicted,
            survival_logits=self.predictor.swept_progress_head(predicted),
        )

    @torch.no_grad()
    def update_target_ema_after_optimizer_step(self) -> None:
        for target, online in (
            (self.target_encoder.weight, self.encoder.weight),
            (self.target_bev_lift.weight, self.bev_lift.weight),
        ):
            target.mul_(0.996).add_(online, alpha=0.004)
        self.ema_update_count.add_(1)


def _semantic_labels() -> torch.Tensor:
    return torch.tensor(((0, 1), (2, 0)), dtype=torch.long)[None].expand(4, -1, -1)


def _microbatches() -> list[dict[str, torch.Tensor]]:
    result = []
    for microbatch in range(4):
        generator = torch.Generator().manual_seed(100 + microbatch)
        prefixes = torch.tensor(
            [
                [(3 * action + row + microbatch) % 16 for action in range(9)]
                for row in range(4)
            ],
            dtype=torch.long,
        )
        immediate = torch.ones((4, 9), dtype=torch.bool)
        immediate[:, 6] = False
        prefixes[:, 6] = 0
        result.append(
            {
                runner.CURRENT_RGB_KEY: torch.randn(
                    (4, 3, 2, 2), generator=generator
                ),
                runner.NEXT_RGB_KEY: torch.randn(
                    (4, 3, 2, 2), generator=generator
                ),
                runner.CURRENT_LABELS_KEY: _semantic_labels().clone(),
                runner.NEXT_LABELS_KEY: _semantic_labels().roll(1, dims=1),
                runner.EXECUTED_ACTION_KEY: torch.tensor((0, 3, 6, 8)),
                runner.IMMEDIATE_FEASIBLE_KEY: immediate,
                runner.PREFIX_LENGTHS_KEY: prefixes,
            }
        )
    return result


def test_joint_update_adds_only_o_and_preserves_v1_accounting_and_partitions() -> None:
    model = _TinyJointModel()
    parameter_names_before = tuple(name for name, _ in model.named_parameters())
    partition = runner.partition_parameters_v1(model)
    optimizer = runner.build_frozen_optimizer_v1(partition)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    result = runner.joint_training_update_v2(model, optimizer, _microbatches())

    assert result.accounting == runner.JointTrainingAccountingV1(
        updates=1,
        presentations=16,
        microbatch_graphs=4,
        backward_calls=4,
        optimizer_steps=1,
        ema_steps=1,
        predictor_forwards=4,
        predictor_objectives=4,
    )
    assert set(result.mean_losses) == {"S", "P", "U", "R", "O", "L"}
    assert result.mean_losses["O"] > 0.0
    assert math.isclose(
        result.mean_losses["L"],
        sum(result.mean_losses[name] for name in ("S", "P", "U", "R", "O")),
        rel_tol=2e-6,
        abs_tol=2e-6,
    )
    assert all(math.isfinite(value) for value in result.mean_losses.values())
    assert all(value > 0.0 for value in result.gradient_l2.values())
    assert int(model.ema_update_count) == 1
    assert tuple(name for name, _ in model.named_parameters()) == parameter_names_before
    assert all(parameter.grad is None for parameter in partition.target)
    for role in (partition.encoder, partition.lift_semantic, partition.predictor):
        ids = set(map(id, role))
        assert any(
            not torch.equal(before[name], parameter)
            for name, parameter in model.named_parameters()
            if id(parameter) in ids
        )
    assert model.semantic_head.weight.grad is not None
    assert float(model.semantic_head.weight.grad.abs().sum()) > 0.0


def _label_rows(role: str = "train") -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for action_index, action in enumerate(runner.ACTION_ORDER):
        rows.append(
            {
                "dataset_role": role,
                "role_state_index": 0,
                "pair_content_sha256": "a" * 64,
                "current_endpoint_sha256": "b" * 64,
                "scene_id": "scene-a",
                "family": "small_enclosed_maze",
                "action_index": action_index,
                "action": action,
                "swept_progress_prefix_length": (
                    0 if action_index == 6 else action_index + 1
                ),
                "immediate_feasible": action_index != 6,
                "provenance": {"executed_pair_primitive": "arc_left"},
            }
        )
    return tuple(rows)


def test_fixed_v2_driver_consumes_exact_v1_schedule_and_traces_o_from_update_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = runner.freeze_role_labels_v1(
        _label_rows(), role="train", np=__import__("numpy")
    )
    pair = {
        "dataset_role": "train",
        "content_sha256": "a" * 64,
        "current_endpoint_sha256": "b" * 64,
        "next_endpoint_sha256": "c" * 64,
        "scene_id": "scene-a",
        "family": "small_enclosed_maze",
        "primitive": "arc_left",
    }
    built = 0

    def fake_build(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal built
        del args, kwargs
        built += 1
        return {}

    def fake_update(
        model: Any,
        optimizer: Any,
        microbatches: Any,
        *,
        accounting: runner.JointTrainingAccountingV1,
    ) -> runner.JointUpdateResultV2:
        del model, optimizer
        assert len(microbatches) == 4
        losses = {name: 1.0 for name in ("S", "P", "U", "R", "O")}
        losses["L"] = 5.0
        return runner.JointUpdateResultV2(
            accounting=runner._v1._base._advanced_accounting(accounting),
            mean_losses=losses,
            gradient_l2={
                name: 1.0 for name in ("encoder", "lift_semantic", "predictor")
            },
            representation_clip_pre_l2=1.0,
            predictor_clip_pre_l2=1.0,
            ranking_active_microbatches=2,
            ranking_eligible_pairs=3,
            survival_supervised_decisions=4,
        )

    monkeypatch.setattr(runner, "build_microbatch_v1", fake_build)
    monkeypatch.setattr(runner, "joint_training_update_v2", fake_update)
    accounting, trace, diagnostics = runner.run_fixed_training_v2(
        object(), object(), object(), (pair,), labels, (0,) * 16_000, object()
    )
    assert accounting.updates == runner._v1.MAXIMUM_UPDATES == 1_000
    assert accounting.presentations == runner._v1.MAXIMUM_PRESENTATIONS == 16_000
    assert len(trace) == 1_000
    assert built == 4_000
    assert trace[0]["losses"] == {
        "S": 1.0,
        "P": 1.0,
        "U": 1.0,
        "R": 1.0,
        "O": 1.0,
        "L": 5.0,
    }
    assert diagnostics["ranking_active_microbatch_count"] == 2_000
    with pytest.raises(PermissionError, match="cap"):
        runner.run_fixed_training_v2(
            object(),
            object(),
            object(),
            (pair,),
            labels,
            (0,) * 16_000,
            object(),
            maximum_updates=999,
        )


def test_v2_reuses_v1_optimizer_data_controls_and_caps_by_identity() -> None:
    assert runner.ACTION_ORDER is runner._v1.ACTION_ORDER
    assert runner.REQUIRED_BATCH_KEYS is runner._v1.REQUIRED_BATCH_KEYS
    assert runner.build_microbatch_v1 is runner._v1.build_microbatch_v1
    assert runner.build_frozen_optimizer_v1 is runner._v1.build_frozen_optimizer_v1
    assert runner.partition_parameters_v1 is runner._v1.partition_parameters_v1
    assert runner.score_full_control_v1 is runner._v1.score_full_control_v1
    assert runner.MAXIMUM_UPDATES == runner._v1.MAXIMUM_UPDATES
    assert runner.MAXIMUM_PRESENTATIONS == runner._v1.MAXIMUM_PRESENTATIONS
    assert runner.OCCUPIED_SAFETY_AUX_COEFFICIENT == 1.0
    assert runner.OCCUPIED_SAFETY_AUX_NORMALIZATION == math.log(2.0)
