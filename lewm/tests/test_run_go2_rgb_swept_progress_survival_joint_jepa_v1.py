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
RUNNER_PATH = ROOT / "scripts/run_go2_rgb_swept_progress_survival_joint_jepa_v1.py"


def _load_runner() -> Any:
    name = "_test_go2_swept_progress_survival_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


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
        pooled = predicted.mean(dim=(-2, -1))
        return self.output(pooled)


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
                [
                    (3 * action + row + microbatch) % 16
                    for action in range(9)
                ]
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


def _label_rows(role: str = "train") -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for action_index, action in enumerate(runner.ACTION_ORDER):
        prefix = 0 if action_index == 6 else action_index + 1
        row: dict[str, Any] = {
            "dataset_role": role,
            "role_state_index": 0,
            "pair_content_sha256": "a" * 64,
            "current_endpoint_sha256": "b" * 64,
            "scene_id": "scene-a",
            "family": "small_enclosed_maze",
            "action_index": action_index,
            "action": action,
            "swept_progress_prefix_length": prefix,
            "provenance": {"executed_pair_primitive": "arc_left"},
        }
        feasible = action_index != 6
        if action_index % 3 == 0:
            row["immediate_primitive"] = {"feasible": feasible}
        elif action_index % 3 == 1:
            row["immediate_feasible"] = feasible
        else:
            row["immediate_primitive_feasible"] = feasible
        rows.append(row)
    return tuple(rows)


def test_label_adapter_pair_join_and_narrow_microbatch() -> None:
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
    runner.validate_pairs_against_labels_v1((pair,), labels)

    class Loader:
        runtime = SimpleNamespace(torch=torch)

        def __init__(self) -> None:
            self.image_kinds: list[str] = []

        def image(self, identity: str, **kwargs: Any) -> torch.Tensor:
            del identity
            self.image_kinds.append(kwargs["kind"])
            return torch.zeros((3, 2, 2))

        def raster_label(self, identity: str, **kwargs: Any) -> torch.Tensor:
            del identity, kwargs
            return torch.tensor(((0, 1), (2, 0)), dtype=torch.uint8)

    loader = Loader()
    batch = runner.build_microbatch_v1(
        loader,
        (pair,),
        labels,
        (0, 0, 0, 0),
        torch.device("cpu"),
        stage="train_update_1",
    )
    assert set(batch) == set(runner.REQUIRED_BATCH_KEYS)
    assert loader.image_kinds == ["current"] * 4 + ["next"] * 4
    assert batch[runner.IMMEDIATE_FEASIBLE_KEY].shape == (4, 9)
    assert batch[runner.IMMEDIATE_FEASIBLE_KEY].dtype == torch.bool
    assert batch[runner.PREFIX_LENGTHS_KEY].shape == (4, 9)
    assert batch[runner.PREFIX_LENGTHS_KEY].dtype == torch.long


def test_one_joint_update_trains_every_online_role_and_scores_controls() -> None:
    model = _TinyJointModel()
    partition = runner.partition_parameters_v1(model)
    optimizer = runner.build_frozen_optimizer_v1(partition)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    result = runner.joint_training_update_v1(model, optimizer, _microbatches())

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
    assert set(result.mean_losses) == {"S", "P", "U", "R", "L"}
    assert all(math.isfinite(value) for value in result.mean_losses.values())
    assert all(value > 0.0 for value in result.gradient_l2.values())
    assert result.ranking_active_microbatches == 4
    assert result.ranking_eligible_pairs > 0
    assert result.survival_supervised_decisions > 0
    assert int(model.ema_update_count) == 1
    assert all(parameter.grad is None for parameter in partition.target)
    for role in (partition.encoder, partition.lift_semantic, partition.predictor):
        assert any(
            not torch.equal(before[name], parameter)
            for name, parameter in model.named_parameters()
            if id(parameter) in set(map(id, role))
        )
    assert model.predictor.swept_progress_head.output.weight.grad is not None
    assert float(model.predictor.swept_progress_head.output.weight.grad.abs().sum()) > 0

    rgb = torch.randn((2, 3, 2, 2))
    latent = model.encode_online(rgb)
    full = runner.score_full_control_v1(model, latent)
    assert full.expected_progress_m.shape == (2, 9)
    shuffled = runner.score_shuffled_action_control_v1(
        model, full.predicted_latents
    )
    assert torch.equal(
        shuffled.predicted_latents[:, 0], full.predicted_latents[:, 1]
    )
    persistence = runner.score_persistence_control_v1(model, latent)
    assert persistence.predicted_latents is None
    assert persistence.expected_progress_m.shape == (2, 9)
    wrong = runner.score_wrong_rgb_control_v1(model, rgb.roll(1, dims=0))
    assert wrong.expected_progress_m.shape == (2, 9)


def test_fixed_driver_consumes_exactly_one_capped_schedule(
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
    ) -> runner.JointUpdateResultV1:
        del model, optimizer
        assert len(microbatches) == 4
        return runner.JointUpdateResultV1(
            accounting=runner._base._advanced_accounting(accounting),
            mean_losses={name: 1.0 for name in ("S", "P", "U", "R", "L")},
            gradient_l2={name: 1.0 for name in ("encoder", "lift_semantic", "predictor")},
            representation_clip_pre_l2=1.0,
            predictor_clip_pre_l2=1.0,
            ranking_active_microbatches=2,
            ranking_eligible_pairs=3,
            survival_supervised_decisions=4,
        )

    monkeypatch.setattr(runner, "build_microbatch_v1", fake_build)
    monkeypatch.setattr(runner, "joint_training_update_v1", fake_update)
    accounting, trace, diagnostics = runner.run_fixed_training_v1(
        object(),
        object(),
        object(),
        (pair,),
        labels,
        (0,) * 16_000,
        object(),
    )
    assert accounting.updates == 1_000
    assert accounting.presentations == 16_000
    assert len(trace) == 1_000
    assert built == 4_000
    assert diagnostics["ranking_active_microbatch_count"] == 2_000
    with pytest.raises(PermissionError, match="cap"):
        runner.run_fixed_training_v1(
            object(),
            object(),
            object(),
            (pair,),
            labels,
            (0,) * 16_000,
            object(),
            maximum_updates=999,
        )
