from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    ROOT
    / "scripts/run_go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16.py"
)


def _load_runner() -> Any:
    name = "_test_go2_ego_motion_aligned_ray_consistency_v16_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    __import__("sys").modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _one_microbatch(offset: float = 0.0) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for key in runner.REQUIRED_BATCH_KEYS:
        values[key] = torch.zeros((4,), dtype=torch.float32)
    values[runner.CURRENT_RGB_KEY] = torch.full((4, 1), 0.2 + offset)
    values[runner.NEXT_RGB_KEY] = torch.full((4, 1), 0.4 + offset)
    values[runner.CURRENT_LABELS_KEY] = torch.zeros((4,), dtype=torch.long)
    values[runner.NEXT_LABELS_KEY] = torch.ones((4,), dtype=torch.long)
    values[runner.EXECUTED_ACTION_KEY] = torch.arange(4, dtype=torch.long)
    values[runner.IMMEDIATE_FEASIBLE_KEY] = torch.ones((4, 9), dtype=torch.bool)
    values[runner.PREFIX_LENGTHS_KEY] = torch.zeros((4, 9), dtype=torch.long)
    for key in (runner.CURRENT_CAMERA_ORIGIN_KEY, runner.NEXT_CAMERA_ORIGIN_KEY):
        values[key] = torch.zeros((4, 3), dtype=torch.float32)
    for key in (runner.CURRENT_CAMERA_BASIS_KEY, runner.NEXT_CAMERA_BASIS_KEY):
        values[key] = torch.eye(3).expand(4, -1, -1).clone()
    for key in (
        runner.CURRENT_GROUND_IN_FRUSTUM_KEY,
        runner.NEXT_GROUND_IN_FRUSTUM_KEY,
        runner.CURRENT_GROUND_CLEAR_KEY,
        runner.NEXT_GROUND_CLEAR_KEY,
        runner.CURRENT_PIXEL_HIT_KEY,
        runner.NEXT_PIXEL_HIT_KEY,
    ):
        values[key] = torch.ones((4, 1), dtype=torch.bool)
    values[runner.REALIZED_RELATIVE_SE2_KEY] = torch.tensor(
        [[0.05 + offset, 0.01, -0.02]] * 4,
        dtype=torch.float32,
    )
    assert tuple(values) == runner.REQUIRED_BATCH_KEYS
    return values


def _microbatches() -> tuple[dict[str, torch.Tensor], ...]:
    return tuple(_one_microbatch(0.01 * index) for index in range(4))


def _accounting_after(completed_updates: int) -> Any:
    return runner.JointTrainingAccountingV13(
        updates=completed_updates,
        presentations=16 * completed_updates,
        microbatch_graphs=4 * completed_updates,
        backward_calls=8 * completed_updates,
        camera_route_grad_calls=4 * completed_updates,
        joint_route_grad_calls=4 * completed_updates,
        camera_frame_objectives=32 * completed_updates,
        optimizer_steps=completed_updates,
        ema_steps=completed_updates,
        predictor_forwards=4 * completed_updates,
        predictor_objectives=4 * completed_updates,
    )


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared_parameter = nn.Parameter(torch.tensor(0.7))
        self.representation_parameter = nn.Parameter(torch.tensor(0.4))
        self.predictor_parameter = nn.Parameter(torch.tensor(0.3))
        self.target_parameter = nn.Parameter(
            torch.tensor(0.2), requires_grad=False
        )
        self.register_buffer("ema_update_count", torch.zeros((), dtype=torch.long))

    def encode_online_training(self, rgb: torch.Tensor, **_: Any) -> Any:
        signal = rgb.float().mean()
        latent = self.shared_parameter + self.representation_parameter + signal
        evidence = self.shared_parameter + 0.5 * signal
        return SimpleNamespace(latent=latent, auxiliary_evidence=evidence)

    def semantic_logits_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return latent

    def predict_all_actions_with_survival(self, latent: torch.Tensor) -> Any:
        predicted = latent + self.predictor_parameter
        return SimpleNamespace(
            predicted_latents=predicted,
            survival_logits=0.5 * predicted,
        )

    def encode_target(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.target_parameter + rgb.float().mean()

    def update_target_ema_after_optimizer_step(self) -> None:
        with torch.no_grad():
            self.target_parameter.mul_(0.9).add_(self.shared_parameter, alpha=0.1)
            self.ema_update_count.add_(1)


class _CountingSgd(torch.optim.SGD):
    def __init__(self, parameters: list[nn.Parameter]) -> None:
        super().__init__(parameters, lr=1e-3)
        self.step_calls = 0

    def step(self, closure: Any = None) -> Any:
        self.step_calls += 1
        return super().step(closure)


def _install_tiny_update_apis(
    monkeypatch: pytest.MonkeyPatch,
    model: _TinyModel,
) -> tuple[Any, list[torch.Tensor]]:
    partition = runner.ParameterPartitionV13(
        encoder=(model.shared_parameter,),
        evidence_head=(),
        representation=(model.representation_parameter,),
        predictor=(model.predictor_parameter,),
        target=(model.target_parameter,),
        names={
            "encoder": ("shared_parameter",),
            "evidence_head": (),
            "representation": ("representation_parameter",),
            "predictor": ("predictor_parameter",),
            "target": ("target_parameter",),
        },
    )

    class _SemanticApi:
        @staticmethod
        def semantic_loss_v1(current: torch.Tensor, *_: Any) -> Any:
            next_logits = _[1]
            return SimpleNamespace(loss=current.square() + next_logits.square())

        @staticmethod
        def microbatch_persistence_loss_v1(
            predicted: torch.Tensor, *_: Any
        ) -> Any:
            return SimpleNamespace(loss=0.2 * predicted.square())

    class _SurvivalApi:
        @staticmethod
        def joint_survival_loss_v1(**values: Any) -> Any:
            semantic = values["semantic_loss"]
            persistence = values["executed_action_ema_latent_loss"]
            survival = 0.3 * values["survival_logits"].square()
            ranking = 0.1 * values["survival_logits"].square()
            return SimpleNamespace(
                loss=semantic + persistence + survival + ranking,
                semantic=semantic,
                executed_action_ema_latent=persistence,
                survival=survival,
                progress_ranking=ranking,
                ranking_terms=SimpleNamespace(
                    eligible_pair_count=torch.tensor(2, dtype=torch.long)
                ),
                survival_terms=SimpleNamespace(
                    supervised_decision_count=torch.tensor(9, dtype=torch.long)
                ),
            )

    monkeypatch.setattr(
        runner._base,
        "_runtime_apis",
        lambda: (torch, _SemanticApi, _SurvivalApi, None, None, None, None),
    )
    monkeypatch.setattr(runner, "partition_parameters_v13", lambda _: partition)
    monkeypatch.setattr(runner, "validate_optimizer_v13", lambda *_: None)
    monkeypatch.setattr(
        runner._base._v3,
        "occupied_safety_aux_loss_v3",
        lambda current, _current_labels, next_, _next_labels: SimpleNamespace(
            loss=0.05 * (current.square() + next_.square())
        ),
    )
    monkeypatch.setattr(
        runner._base._v3._v2._v1,
        "_prediction_parts",
        lambda prediction: (
            prediction.predicted_latents,
            prediction.survival_logits,
        ),
    )
    monkeypatch.setattr(
        runner,
        "camera_evidence_pair_loss_v13",
        lambda current, next_, *_: SimpleNamespace(
            total=current.square() + next_.square()
        ),
    )
    realized_rows: list[torch.Tensor] = []

    def consistency(current: torch.Tensor, next_: torch.Tensor, **values: Any) -> Any:
        realized_rows.append(values["relative_se2_current_frame"].detach().clone())
        return runner.EgoMotionAlignedRayConsistencyReceiptV16(
            loss=0.5 * (current.square() + next_.square()),
            shared_valid_cell_count=7,
            positive_weight_cell_count=5,
            weight_sum=3.25,
        )

    monkeypatch.setattr(runner, "ego_motion_aligned_ray_consistency_v16", consistency)
    return partition, realized_rows


def test_exact_realized_se2_schema_is_required_and_finite() -> None:
    batches = list(_microbatches())
    runner._validate_microbatches_v16(torch, batches)

    for malformed in (
        torch.zeros((4, 2), dtype=torch.float32),
        torch.zeros((4, 3), dtype=torch.float64),
        torch.tensor([[float("nan"), 0.0, 0.0]] * 4),
    ):
        changed = [dict(batch) for batch in batches]
        changed[0][runner.REALIZED_RELATIVE_SE2_KEY] = malformed
        with pytest.raises((ValueError, FloatingPointError)):
            runner._validate_microbatches_v16(torch, changed)

    changed = [dict(batch) for batch in batches]
    changed[0]["unexpected"] = torch.zeros((4,))
    with pytest.raises(ValueError, match="key order or membership"):
        runner._validate_microbatches_v16(torch, changed)


def test_adapter_preserves_caps_and_executor_compatibility_hooks() -> None:
    assert runner.MAXIMUM_UPDATES == 1_000
    assert runner.MAXIMUM_PRESENTATIONS == 16_000
    assert runner.PRESENTATIONS_PER_UPDATE == 16
    assert runner.EGO_MOTION_ALIGNED_RAY_CONSISTENCY_WEIGHT_V16 == 0.1
    assert runner.joint_training_update_v13 is runner.joint_training_update_v16
    assert runner._validate_microbatches_v13 is runner._validate_microbatches_v16


def test_delayed_onset_weight_has_one_exact_boundary() -> None:
    assert runner.RAY_CONSISTENCY_ONSET_UPDATE_V17 == 101
    assert runner.ray_consistency_weight_v17(1) == 0.0
    assert runner.ray_consistency_weight_v17(100) == 0.0
    assert runner.ray_consistency_weight_v17(101) == 0.1
    assert runner.ray_consistency_weight_v17(1_000) == 0.1
    for update in (0, 1_001, True):
        with pytest.raises(ValueError, match="integer in"):
            runner.ray_consistency_weight_v17(update)


def test_update100_computes_m_but_does_not_add_it_to_camera_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyModel()
    partition, _ = _install_tiny_update_apis(monkeypatch, model)
    optimizer = _CountingSgd(list(partition.online))
    model.ema_update_count.fill_(99)
    result = runner.joint_training_update_v16(
        model,
        optimizer,
        _microbatches(),
        accounting=_accounting_after(99),
    )

    assert result.accounting.updates == 100
    assert result.mean_losses["M"] > 0.0
    assert result.mean_losses["C"] == pytest.approx(
        result.mean_losses["C_base"]
    )


def test_one_update_routes_weighted_m_only_through_camera_and_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyModel()
    partition, realized_rows = _install_tiny_update_apis(monkeypatch, model)
    optimizer = _CountingSgd(list(partition.online))
    original_grad = torch.autograd.grad
    grad_calls: list[int] = []

    def counted_grad(*args: Any, **kwargs: Any) -> Any:
        grad_calls.append(len(args[1]))
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", counted_grad)
    model.ema_update_count.fill_(100)
    result = runner.joint_training_update_v16(
        model,
        optimizer,
        _microbatches(),
        accounting=_accounting_after(100),
    )

    assert grad_calls == [1, 3] * 4
    assert optimizer.step_calls == 1
    assert int(model.ema_update_count.item()) == 101
    assert result.accounting.updates == 101
    assert result.optimizer_steps_this_update == result.ema_steps_this_update == 1
    assert result.accounting.camera_route_grad_calls == 404
    assert result.accounting.joint_route_grad_calls == 404
    assert result.target_gradient_tensor_count == 0
    assert model.target_parameter.grad is None
    assert len(realized_rows) == 4
    assert all(tuple(value.shape) == (4, 3) for value in realized_rows)
    assert result.ray_consistency_shared_valid_cell_count == 28
    assert result.ray_consistency_positive_weight_cell_count == 20
    assert result.ray_consistency_weight_sum == 13.0
    assert set(result.mean_losses) == {
        "S",
        "P",
        "U",
        "R",
        "O",
        "N",
        "C_base",
        "M",
        "C",
        "L",
    }
    assert math.isclose(
        result.mean_losses["C"],
        result.mean_losses["C_base"] + 0.1 * result.mean_losses["M"],
        rel_tol=1e-6,
        abs_tol=1e-6,
    )
    assert math.isclose(
        result.mean_losses["L"],
        result.mean_losses["N"] + result.mean_losses["C"],
        rel_tol=1e-6,
        abs_tol=1e-6,
    )
    assert math.isclose(
        result.mean_losses["N"],
        sum(result.mean_losses[name] for name in ("S", "P", "U", "R", "O")),
        rel_tol=1e-6,
        abs_tol=1e-6,
    )
    assert all(
        result.gradient_routes[name].preclip_l2 > 0.0
        for name in ("camera_shared", "joint_shared", "predictor")
    )
