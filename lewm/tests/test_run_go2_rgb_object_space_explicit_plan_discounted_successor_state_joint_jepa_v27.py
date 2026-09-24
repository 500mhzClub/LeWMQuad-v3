from __future__ import annotations

from dataclasses import replace
import importlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runner = importlib.import_module(
    "scripts.run_go2_rgb_object_space_explicit_plan_discounted_"
    "successor_state_joint_jepa_v27"
)


class _TargetModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.25), requires_grad=False)
        self.seen_shape: tuple[int, ...] | None = None

    def encode_target(self, rgb: torch.Tensor) -> torch.Tensor:
        self.seen_shape = tuple(rgb.shape)
        values = rgb.mean(dim=(1, 2, 3)) + self.bias
        return values[:, None, None, None].expand(-1, 64, 64, 64)


def test_discounted_target_is_the_exact_normalized_four_step_ema_sum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.v25._tensor_core,
        "_runtime_apis",
        lambda: (torch, None, None, None, None, None, None),
    )
    model = _TargetModel()
    future = torch.stack(
        tuple(torch.full((3, 112, 112), value) for value in (1.0, 2.0, 4.0, 8.0))
    )[None]

    actual = runner.discounted_future_target_v27(model, future)
    weights = torch.tensor((1.0, 0.9, 0.9**2, 0.9**3))
    expected = ((weights * torch.tensor((1.25, 2.25, 4.25, 8.25))).sum()
                / weights.sum())

    assert model.seen_shape == (4, 3, 112, 112)
    assert tuple(actual.shape) == (1, 64, 64, 64)
    assert torch.allclose(actual, torch.full_like(actual, expected))
    assert actual.requires_grad is False
    assert model.bias.grad is None


def _accounting(updates: int) -> Any:
    return runner.JointTrainingAccountingV27(
        updates=updates,
        presentations=32 * updates,
        physical_presentations=16 * updates,
        plan_presentations=16 * updates,
        physical_microbatch_graphs=4 * updates,
        plan_microbatch_graphs=4 * updates,
        autograd_grad_calls=16 * updates,
        optimizer_steps=updates,
        ema_steps=updates,
    )


def test_accounting_and_400_update_12800_presentation_cap_are_exact() -> None:
    runner.validate_accounting_v27(_accounting(0))
    runner.validate_accounting_v27(_accounting(400))
    runner._validate_capacity_v27(_accounting(399))
    with pytest.raises(PermissionError, match="no complete update"):
        runner._validate_capacity_v27(_accounting(400))
    with pytest.raises(RuntimeError, match="accounting is inconsistent"):
        runner.validate_accounting_v27(
            replace(_accounting(7), plan_microbatch_graphs=27)
        )


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Parameter(torch.tensor(0.30))
        self.spatial = nn.Parameter(torch.tensor(0.20))
        self.semantic = nn.Parameter(torch.tensor(0.10))
        self.old_transition = nn.Parameter(torch.tensor(0.04))
        self.old_survival = nn.Parameter(torch.tensor(0.05))
        self.plan = nn.Parameter(torch.tensor(0.06))
        self.target = nn.Parameter(torch.tensor(0.25), requires_grad=False)
        self.register_buffer("ema_update_count", torch.zeros((), dtype=torch.long))
        self.physical_forwards = 0
        self.plan_forwards = 0
        self.ema_calls = 0

    def _latent(self, rgb: torch.Tensor) -> torch.Tensor:
        signal = rgb.float().reshape(rgb.shape[0], -1).mean(dim=1)
        return (signal + self.shared + self.spatial)[:, None, None, None]

    def encode_online_training(self, rgb: torch.Tensor, **_: Any) -> Any:
        self.physical_forwards += 1
        latent = self._latent(rgb)
        return SimpleNamespace(
            latent=latent,
            auxiliary_evidence=latent.mean() + self.shared,
        )

    def encode_online(self, rgb: torch.Tensor) -> torch.Tensor:
        return self._latent(rgb)

    def semantic_logits_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        value = latent + self.semantic
        return torch.cat((value, -value, 0.25 * value), dim=1)

    def predict_all_actions_with_survival(self, latent: torch.Tensor) -> Any:
        value = (latent + self.old_transition) * self.old_survival
        logits = value[:, None].expand(-1, 9, -1, -1, 16).reshape(-1, 9, 16)
        return SimpleNamespace(survival_logits=logits)

    def predict_plan_successor(
        self, current: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        self.plan_forwards += 1
        action_signal = 0.01 * actions.float().mean(dim=1)[:, None, None, None]
        return current + self.plan + action_signal

    def update_target_ema_after_optimizer_step(self) -> None:
        with torch.no_grad():
            self.target.mul_(0.9).add_(self.shared, alpha=0.1)
            self.ema_update_count.add_(1)
        self.ema_calls += 1


class _CountingSgd(torch.optim.SGD):
    def __init__(self, parameters: list[nn.Parameter]) -> None:
        super().__init__(parameters, lr=1.0e-3)
        self.step_calls = 0

    def step(self, closure: Any = None) -> Any:
        self.step_calls += 1
        return super().step(closure)


def _partition(model: _TinyModel) -> Any:
    return runner.ParameterPartitionV27(
        encoder=(model.shared,),
        evidence_head=(),
        representation=(model.spatial, model.semantic),
        predictor=(model.old_transition, model.old_survival),
        plan=(model.plan,),
        target=(model.target,),
        names={
            "encoder": ("encoder.synthetic",),
            "evidence_head": (),
            "representation": (
                "bev_lift.point_projection.synthetic",
                "semantic_head.synthetic",
            ),
            "predictor": (
                "predictor.action_embedding.weight",
                "predictor.swept_progress_head.output.weight",
            ),
            "plan": ("plan_predictor.synthetic",),
            "target": ("target_encoder.synthetic",),
        },
    )


def _physical_batch() -> dict[str, torch.Tensor]:
    v25 = runner.v25
    batch: dict[str, torch.Tensor] = {
        v25.CURRENT_RGB_KEY: torch.full((4, 1), 0.2),
        v25.NEXT_RGB_KEY: torch.full((4, 1), 0.4),
        v25.CURRENT_LABELS_KEY: torch.zeros((4, 1, 1), dtype=torch.long),
        v25.NEXT_LABELS_KEY: torch.ones((4, 1, 1), dtype=torch.long),
        v25.IMMEDIATE_FEASIBLE_KEY: torch.ones((4, 9), dtype=torch.bool),
        v25.PREFIX_LENGTHS_KEY: torch.ones((4, 9), dtype=torch.long),
        v25.SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21: torch.tensor((1, 2, 3, 0)),
        v25.ACTION_PRIOR_M_KEY_V23: torch.full((9,), 0.55),
    }
    for key in (v25.CURRENT_CAMERA_ORIGIN_KEY, v25.NEXT_CAMERA_ORIGIN_KEY):
        batch[key] = torch.zeros((4, 3))
    for key in (v25.CURRENT_CAMERA_BASIS_KEY, v25.NEXT_CAMERA_BASIS_KEY):
        batch[key] = torch.eye(3).expand(4, -1, -1).clone()
    for key in (
        v25.CURRENT_GROUND_PLANE_Z_KEY,
        v25.NEXT_GROUND_PLANE_Z_KEY,
        v25.CURRENT_PIXEL_HIT_KEY,
        v25.CURRENT_PIXEL_DISTANCE_KEY,
        v25.CURRENT_GROUND_IN_FRUSTUM_KEY,
        v25.CURRENT_GROUND_CLEAR_KEY,
        v25.NEXT_PIXEL_HIT_KEY,
        v25.NEXT_PIXEL_DISTANCE_KEY,
        v25.NEXT_GROUND_IN_FRUSTUM_KEY,
        v25.NEXT_GROUND_CLEAR_KEY,
    ):
        batch[key] = torch.zeros((4,))
    return batch


def _plan_batch() -> dict[str, torch.Tensor]:
    return {
        runner.H6_CURRENT_RGB_KEY_V27: torch.zeros((4, 3, 112, 112)),
        runner.H6_FUTURE_RGB_KEY_V27: torch.full((4, 4, 3, 112, 112), 0.7),
        runner.H6_FUTURE_ACTIONS_KEY_V27: torch.tensor(
            ((0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6)),
            dtype=torch.long,
        ),
    }


def test_one_mixed_cpu_update_has_exact_graphs_and_disjoint_plan_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyModel()
    partition = _partition(model)
    p25_calls: list[Any] = []
    zero_temporal_terms: list[float] = []
    target_calls = 0

    class _SemanticApi:
        @staticmethod
        def semantic_loss_v1(current_logits, current_labels, next_logits, next_labels):
            loss = 0.5 * (
                F.cross_entropy(current_logits, current_labels)
                + F.cross_entropy(next_logits, next_labels)
            ) / math.log(3.0)
            return SimpleNamespace(loss=loss)

        @staticmethod
        def latent_energy_per_row(predicted, target):
            return (predicted - target.detach()).square().mean(dim=(1, 2, 3))

        @staticmethod
        def microbatch_persistence_loss_v1(*args: Any) -> Any:
            p25_calls.append(args)
            raise AssertionError("rejected P25 objective was called")

    class _SurvivalApi:
        @staticmethod
        def joint_survival_loss_v1(**values: Any) -> Any:
            temporal = values["executed_action_ema_latent_loss"]
            zero_temporal_terms.append(float(temporal.detach()))
            survival = 0.05 * values["survival_logits"].square().mean()
            ranking = 0.02 * values["survival_logits"].square().mean()
            semantic = values["semantic_loss"]
            return SimpleNamespace(
                loss=semantic + temporal + survival + ranking,
                semantic=semantic,
                survival=survival,
                progress_ranking=ranking,
                ranking_terms=SimpleNamespace(eligible_pair_count=torch.tensor(3)),
                survival_terms=SimpleNamespace(
                    supervised_decision_count=torch.tensor(9)
                ),
            )

    def auxiliary_objective(_torch, _survival_api, logits, *_args):
        base = logits.square().mean(dim=(1, 2))
        return SimpleNamespace(
            loss=base.mean(),
            positive_energy=base[:, None].expand(-1, 8),
            scene_negative_energy=base,
            prior_negative_energy=base,
            scene_eligible=torch.ones(4, dtype=torch.bool),
            prior_eligible=torch.ones(4, dtype=torch.bool),
            scene_advantage_sum=base.sum(),
            prior_advantage_sum=base.sum(),
            scene_rank_sum=base.sum(),
            prior_rank_sum=base.sum(),
            scene_eligible_count=4,
            prior_eligible_count=4,
        )

    subset = SimpleNamespace(
        parameters=(model.shared, model.old_survival),
        protected_predictor_core_parameters=(model.old_transition,),
    )
    monkeypatch.setattr(
        runner.v25._tensor_core,
        "_runtime_apis",
        lambda: (torch, _SemanticApi, _SurvivalApi, None, None, None, None),
    )
    monkeypatch.setattr(runner.v25, "_validate_microbatches_v25", lambda *_: None)
    monkeypatch.setattr(runner, "partition_parameters_v27", lambda _: partition)
    monkeypatch.setattr(runner, "validate_optimizer_v27", lambda *_: None)
    monkeypatch.setattr(
        runner.v25._v24,
        "predictor_core_protected_survival_parameter_subset_v24",
        lambda _: subset,
    )
    monkeypatch.setattr(
        runner.v25._v24,
        "predictor_core_protected_survival_objective_v24",
        auxiliary_objective,
    )
    monkeypatch.setattr(
        runner.v25._tensor_core._v3,
        "occupied_safety_aux_loss_v3",
        lambda current, _a, next_, _b: SimpleNamespace(
            loss=0.01 * (current.square().mean() + next_.square().mean())
        ),
    )
    monkeypatch.setattr(
        runner.v25._tensor_core._v3._v2._v1,
        "_prediction_parts",
        lambda prediction: (None, prediction.survival_logits),
    )
    monkeypatch.setattr(
        runner.v25._base,
        "camera_evidence_pair_loss_v13",
        lambda current, next_, *_: SimpleNamespace(
            total=current.square() + next_.square()
        ),
    )

    def tiny_discounted_target(_model, future):
        nonlocal target_calls
        target_calls += 1
        weights = future.new_tensor((1.0, 0.9, 0.9**2, 0.9**3))
        rows = (future.mean(dim=(2, 3, 4)) * weights).sum(dim=1) / weights.sum()
        return (rows + model.target.detach())[:, None, None, None]

    monkeypatch.setattr(runner, "discounted_future_target_v27", tiny_discounted_target)
    original_grad = torch.autograd.grad
    grad_parameter_ids: list[tuple[int, ...]] = []

    def counted_grad(*args: Any, **kwargs: Any) -> Any:
        grad_parameter_ids.append(tuple(id(value) for value in args[1]))
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", counted_grad)
    optimizer = _CountingSgd(list(partition.online))
    result = runner.joint_training_update_v27(
        model,
        optimizer,
        (_physical_batch(),) * 4,
        (_plan_batch(),) * 4,
    )

    assert [len(values) for values in grad_parameter_ids] == [1, 5, 2] * 4 + [3] * 4
    plan_ids = tuple(id(value) for value in partition.plan_recipients)
    assert grad_parameter_ids[12:] == [plan_ids] * 4
    assert id(model.semantic) not in plan_ids
    assert id(model.old_transition) not in plan_ids
    assert id(model.old_survival) not in plan_ids
    assert optimizer.step_calls == 1
    assert model.ema_calls == 1
    assert int(model.ema_update_count) == 1
    assert model.physical_forwards == 8
    assert model.plan_forwards == 4
    assert target_calls == 4
    assert model.target.grad is None
    assert result.target_gradient_tensor_count == 0
    assert result.optimizer_steps_this_update == 1
    assert result.ema_steps_this_update == 1
    assert result.accounting == _accounting(1)
    assert result.plan_diagnostics["p25_evaluation_count"] == 0
    assert "P25" not in result.mean_losses
    assert p25_calls == []
    assert zero_temporal_terms == [0.0] * 4

