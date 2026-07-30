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

V18_NAME = "scripts.run_go2_rgb_object_space_height_volume_joint_jepa_v18"
V19_NAME = (
    "scripts.run_go2_rgb_object_space_height_volume_executed_successor_"
    "semantic_grounding_joint_jepa_v19"
)
runner = importlib.import_module(V19_NAME)


TRANSITION_NAMES = (
    "predictor.action_embedding.weight",
    "predictor.input_projection.weight",
    "predictor.input_projection.bias",
    "predictor.residual_blocks.0.conv1.weight",
    "predictor.residual_blocks.0.conv1.bias",
    "predictor.residual_blocks.0.conv2.weight",
    "predictor.residual_blocks.0.conv2.bias",
    "predictor.residual_blocks.1.conv1.weight",
    "predictor.residual_blocks.1.conv1.bias",
    "predictor.residual_blocks.1.conv2.weight",
    "predictor.residual_blocks.1.conv2.bias",
    "predictor.residual_head.weight",
    "predictor.residual_head.bias",
)
TRANSITION_SIZES = (
    576,
    73_728,
    64,
    36_864,
    64,
    36_864,
    64,
    36_864,
    64,
    36_864,
    64,
    36_864,
    64,
)
SURVIVAL_NAMES = (
    "predictor.swept_progress_head.output.weight",
    "predictor.swept_progress_head.output.bias",
)
SURVIVAL_SIZES = (64, 1)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))
        self.representation = nn.Parameter(
            torch.tensor(0.15, dtype=torch.float32)
        )
        self.transition = nn.ParameterList(
            [
                nn.Parameter(torch.full((size,), 0.01, dtype=torch.float32))
                for size in TRANSITION_SIZES
            ]
        )
        self.survival = nn.ParameterList(
            [
                nn.Parameter(torch.full((size,), 0.02, dtype=torch.float32))
                for size in SURVIVAL_SIZES
            ]
        )
        self.target = nn.Parameter(
            torch.tensor(0.25, dtype=torch.float32), requires_grad=False
        )
        self.register_buffer("ema_update_count", torch.zeros((), dtype=torch.long))
        self.predictor_forward_count = 0

    def encode_online_training(self, rgb: torch.Tensor, **_: Any) -> Any:
        batch = rgb.shape[0]
        signal = rgb.float().reshape(batch, -1).mean(dim=1)[:, None, None, None]
        latent = signal + self.shared + self.representation
        return SimpleNamespace(
            latent=latent,
            auxiliary_evidence=signal.mean() + self.shared,
        )

    def semantic_logits_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.cat((latent, -latent, 0.25 * latent), dim=1)

    def predict_all_actions_with_survival(self, latent: torch.Tensor) -> Any:
        self.predictor_forward_count += 1
        transition_signal = sum(
            parameter.reshape(-1)[0] for parameter in self.transition
        )
        action_offsets = torch.arange(
            9, dtype=latent.dtype, device=latent.device
        )[None, :, None, None, None]
        predicted = (
            latent[:, None]
            + 1.0e-3 * transition_signal
            + 1.0e-2 * action_offsets
        )
        pooled = predicted.mean(dim=(2, 3, 4))
        survival_logits = (
            pooled * self.survival[0][0] + self.survival[1][0]
        )[:, :, None].expand(-1, -1, 16)
        return SimpleNamespace(
            predicted_latents=predicted,
            survival_logits=survival_logits,
        )

    def encode_target(self, rgb: torch.Tensor) -> torch.Tensor:
        batch = rgb.shape[0]
        signal = rgb.float().reshape(batch, -1).mean(dim=1)[:, None, None, None]
        return signal + self.target

    def update_target_ema_after_optimizer_step(self) -> None:
        with torch.no_grad():
            self.target.mul_(0.9).add_(self.shared, alpha=0.1)
            self.ema_update_count.add_(1)


class _CountingSgd(torch.optim.SGD):
    def __init__(self, parameters: list[nn.Parameter]) -> None:
        super().__init__(parameters, lr=1.0e-3)
        self.step_calls = 0

    def step(self, closure: Any = None) -> Any:
        self.step_calls += 1
        return super().step(closure)


def _partition(model: _TinyModel) -> Any:
    predictor = tuple(model.transition) + tuple(model.survival)
    return runner.ParameterPartitionV13(
        encoder=(model.shared,),
        evidence_head=(),
        representation=(model.representation,),
        predictor=predictor,
        target=(model.target,),
        names={
            "encoder": ("encoder.synthetic",),
            "evidence_head": (),
            "representation": ("semantic_head.synthetic",),
            "predictor": TRANSITION_NAMES + SURVIVAL_NAMES,
            "target": ("target_encoder.synthetic",),
        },
    )


def _one_microbatch(offset: float) -> dict[str, torch.Tensor]:
    values = {
        key: torch.zeros((4,), dtype=torch.float32)
        for key in runner.REQUIRED_BATCH_KEYS
    }
    values[runner.CURRENT_RGB_KEY] = torch.full((4, 1), 0.2 + offset)
    values[runner.NEXT_RGB_KEY] = torch.full((4, 1), 0.4 + offset)
    values[runner.CURRENT_LABELS_KEY] = torch.zeros((4, 1, 1), dtype=torch.long)
    values[runner.NEXT_LABELS_KEY] = torch.ones((4, 1, 1), dtype=torch.long)
    values[runner.EXECUTED_ACTION_KEY] = torch.tensor(
        (0, runner.HOLD_ACTION_INDEX_V19, 2, 3), dtype=torch.long
    )
    values[runner.IMMEDIATE_FEASIBLE_KEY] = torch.ones(
        (4, 9), dtype=torch.bool
    )
    values[runner.PREFIX_LENGTHS_KEY] = torch.ones((4, 9), dtype=torch.long)
    for key in (runner.CURRENT_CAMERA_ORIGIN_KEY, runner.NEXT_CAMERA_ORIGIN_KEY):
        values[key] = torch.zeros((4, 3), dtype=torch.float32)
    for key in (runner.CURRENT_CAMERA_BASIS_KEY, runner.NEXT_CAMERA_BASIS_KEY):
        values[key] = torch.eye(3).expand(4, -1, -1).clone()
    return values


def _microbatches() -> tuple[dict[str, torch.Tensor], ...]:
    return tuple(_one_microbatch(0.01 * index) for index in range(4))


def _accounting_after(updates: int) -> runner.JointTrainingAccountingV19:
    return runner.JointTrainingAccountingV19(
        updates=updates,
        presentations=16 * updates,
        microbatch_graphs=4 * updates,
        backward_calls=12 * updates,
        camera_route_grad_calls=4 * updates,
        joint_route_grad_calls=4 * updates,
        factual_successor_grad_calls=4 * updates,
        camera_frame_objectives=32 * updates,
        optimizer_steps=updates,
        ema_steps=updates,
        predictor_forwards=4 * updates,
        predictor_objectives=8 * updates,
        factual_successor_objectives=4 * updates,
    )


def _macro_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    per_cell = F.cross_entropy(logits, labels, reduction="none")
    return per_cell.flatten(start_dim=1).mean(dim=1)


def _install_tiny_apis(
    monkeypatch: pytest.MonkeyPatch,
    model: _TinyModel,
    partition: Any,
) -> None:
    class _SemanticApi:
        final_class_macro_nll_per_row = staticmethod(_macro_nll)

        @staticmethod
        def semantic_loss_v1(
            current_logits: torch.Tensor,
            current_labels: torch.Tensor,
            next_logits: torch.Tensor,
            next_labels: torch.Tensor,
        ) -> Any:
            loss = 0.5 * (
                _macro_nll(current_logits, current_labels).mean()
                + _macro_nll(next_logits, next_labels).mean()
            ) / math.log(3.0)
            return SimpleNamespace(loss=loss)

        @staticmethod
        def microbatch_persistence_loss_v1(
            predicted: torch.Tensor,
            executed: torch.Tensor,
            _ema_current: torch.Tensor,
            ema_next: torch.Tensor,
        ) -> Any:
            rows = torch.arange(4)
            factual = predicted[rows, executed]
            return SimpleNamespace(loss=(factual - ema_next).square().mean())

    class _SurvivalApi:
        @staticmethod
        def joint_survival_loss_v1(**values: Any) -> Any:
            semantic = values["semantic_loss"]
            persistence = values["executed_action_ema_latent_loss"]
            survival = 0.05 * values["survival_logits"].square().mean()
            ranking = 0.02 * values["survival_logits"].square().mean()
            return SimpleNamespace(
                loss=semantic + persistence + survival + ranking,
                semantic=semantic,
                executed_action_ema_latent=persistence,
                survival=survival,
                progress_ranking=ranking,
                ranking_terms=SimpleNamespace(
                    eligible_pair_count=torch.tensor(3, dtype=torch.long)
                ),
                survival_terms=SimpleNamespace(
                    supervised_decision_count=torch.tensor(9, dtype=torch.long)
                ),
            )

    monkeypatch.setattr(
        runner._tensor_core,
        "_runtime_apis",
        lambda: (torch, _SemanticApi, _SurvivalApi, None, None, None, None),
    )
    monkeypatch.setattr(
        runner._base, "partition_parameters_v18", lambda _: partition
    )
    monkeypatch.setattr(runner._base, "validate_optimizer_v18", lambda *_: None)
    monkeypatch.setattr(
        runner._tensor_core._v3,
        "occupied_safety_aux_loss_v3",
        lambda current, _current_labels, next_, _next_labels: SimpleNamespace(
            loss=0.01 * (current.square().mean() + next_.square().mean())
        ),
    )
    monkeypatch.setattr(
        runner._tensor_core._v3._v2._v1,
        "_prediction_parts",
        lambda prediction: (
            prediction.predicted_latents,
            prediction.survival_logits,
        ),
    )
    monkeypatch.setattr(
        runner._base,
        "camera_evidence_pair_loss_v13",
        lambda current, next_, *_: SimpleNamespace(
            total=current.square() + next_.square()
        ),
    )


def test_private_adapter_preserves_v18_surface_caps_and_executor_aliases() -> None:
    receipt = runner.private_training_adapter_receipt_v19()
    assert isinstance(receipt.pop("public_base_was_loaded_before_adapter"), bool)
    assert receipt == {
        "schema": (
            "lewm_go2_rgb_object_space_height_volume_executed_successor_"
            "semantic_grounding_joint_jepa_v19_training_adapter_v1"
        ),
        "base_training": (
            "scripts/run_go2_rgb_object_space_height_volume_joint_jepa_v18.py"
        ),
        "public_base_loaded_by_adapter": False,
        "private_module_registered": False,
        "factual_successor_gradient_norm_cap": 1.0,
        "factual_successor_predictor_parameter_tensor_count": 13,
        "factual_successor_predictor_parameter_count": 259_008,
        "excluded_predictor_prefix": "predictor.swept_progress_head.",
        "maximum_updates": 1_000,
        "maximum_presentations": 16_000,
    }
    assert runner._base.__name__ != V18_NAME
    assert runner.PRIVATE_BASE_MODULE_NAME not in sys.modules
    assert runner.MAXIMUM_UPDATES == 1_000
    assert runner.MAXIMUM_PRESENTATIONS == 16_000
    assert runner.PRESENTATIONS_PER_UPDATE == 16
    assert runner.joint_training_update_v13 is runner.joint_training_update_v19
    assert runner.validate_accounting_v13 is runner.validate_accounting_v19
    assert runner._validate_microbatches_v13 is runner._base._validate_microbatches_v13


def test_registered_factual_subset_is_exact_and_excludes_only_survival_head() -> None:
    model = _TinyModel()
    partition = _partition(model)
    subset = runner.factual_successor_predictor_subset_v19(partition)

    assert subset.names == TRANSITION_NAMES
    assert subset.parameters == tuple(model.transition)
    assert subset.predictor_indices == tuple(range(13))
    assert subset.parameter_count == 259_008
    assert sum(parameter.numel() for parameter in partition.predictor[13:]) == 65

    changed_names = dict(partition.names)
    changed_names["predictor"] = (
        *TRANSITION_NAMES[:-1],
        "predictor.swept_progress_head.accidentally_excluded",
        *SURVIVAL_NAMES,
    )
    malformed = replace(partition, names=changed_names)
    with pytest.raises(RuntimeError, match="subset changed"):
        runner.factual_successor_predictor_subset_v19(malformed)


def test_v19_accounting_is_exact_and_cap_is_fail_closed() -> None:
    runner.validate_accounting_v19(_accounting_after(0))
    runner.validate_accounting_v19(_accounting_after(400))
    runner.validate_accounting_v19(_accounting_after(1_000))

    malformed = replace(_accounting_after(3), backward_calls=35)
    with pytest.raises(RuntimeError, match="accounting is inconsistent"):
        runner.validate_accounting_v19(malformed)
    with pytest.raises(PermissionError, match="no complete update"):
        runner._validate_update_capacity_v19(_accounting_after(1_000))


def test_one_update_adds_only_the_exact_norm_capped_factual_predictor_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyModel()
    partition = _partition(model)
    _install_tiny_apis(monkeypatch, model, partition)
    optimizer = _CountingSgd(list(partition.online))
    original_grad = torch.autograd.grad
    grad_parameter_ids: list[tuple[int, ...]] = []

    def counted_grad(*args: Any, **kwargs: Any) -> Any:
        grad_parameter_ids.append(tuple(id(value) for value in args[1]))
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", counted_grad)
    result = runner.joint_training_update_v19(
        model, optimizer, _microbatches()
    )

    assert [len(value) for value in grad_parameter_ids] == [1, 17, 13] * 4
    transition_ids = tuple(id(value) for value in model.transition)
    assert grad_parameter_ids[2::3] == [transition_ids] * 4
    assert all(
        id(value) not in grad_parameter_ids[2]
        for value in (
            model.shared,
            model.representation,
            model.target,
            *model.survival,
        )
    )
    assert optimizer.step_calls == 1
    assert model.predictor_forward_count == 4
    assert int(model.ema_update_count.item()) == 1
    assert model.target.grad is None
    assert result.target_gradient_tensor_count == 0
    assert result.optimizer_steps_this_update == 1
    assert result.ema_steps_this_update == 1

    assert result.accounting == _accounting_after(1)
    assert set(result.gradient_routes) == {
        "camera_shared",
        "joint_shared",
        "representation",
        "predictor",
        "factual_successor_predictor",
    }
    factual_route = result.gradient_routes["factual_successor_predictor"]
    assert factual_route.parameter_tensor_count == 13
    assert factual_route.absent_tensor_gradient_count == 0
    assert factual_route.preclip_l2 > 0.0
    assert factual_route.applied_scale == pytest.approx(
        min(1.0, 1.0 / factual_route.preclip_l2), rel=1.0e-6
    )
    assert set(result.mean_losses) == {"S", "P", "U", "R", "O", "Q", "N", "C", "L"}
    assert result.mean_losses["N"] == pytest.approx(
        sum(result.mean_losses[name] for name in ("S", "P", "U", "R", "O")),
        rel=2.0e-6,
    )
    assert result.mean_losses["L"] == pytest.approx(
        result.mean_losses["N"]
        + result.mean_losses["C"]
        + result.mean_losses["Q"],
        rel=2.0e-6,
    )

    diagnostics = result.factual_successor_diagnostics
    assert set(diagnostics) == {
        "successor_semantic_nll_normalized",
        "persistence_semantic_nll_normalized",
        "successor_minus_persistence_nll_normalized",
        "changed_cell_fraction",
        "non_hold_row_count",
        "matching_predictor_gradient_cosine",
    }
    assert diagnostics["successor_semantic_nll_normalized"] == pytest.approx(
        result.mean_losses["Q"]
    )
    assert diagnostics["successor_minus_persistence_nll_normalized"] == pytest.approx(
        diagnostics["successor_semantic_nll_normalized"]
        - diagnostics["persistence_semantic_nll_normalized"]
    )
    assert diagnostics["changed_cell_fraction"] == 1.0
    assert diagnostics["non_hold_row_count"] == 12
    assert math.isfinite(float(diagnostics["matching_predictor_gradient_cosine"]))
    assert -1.0 <= diagnostics["matching_predictor_gradient_cosine"] <= 1.0


def test_factual_route_rejects_any_absent_transition_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyModel()
    partition = _partition(model)
    _install_tiny_apis(monkeypatch, model, partition)
    optimizer = _CountingSgd(list(partition.online))
    original_grad = torch.autograd.grad
    call = 0

    def drop_one_factual_gradient(*args: Any, **kwargs: Any) -> Any:
        nonlocal call
        call += 1
        gradients = original_grad(*args, **kwargs)
        if call == 3:
            return (None, *gradients[1:])
        return gradients

    monkeypatch.setattr(torch.autograd, "grad", drop_one_factual_gradient)
    with pytest.raises(RuntimeError, match="absent gradient"):
        runner.joint_training_update_v19(model, optimizer, _microbatches())
    assert optimizer.step_calls == 0
    assert int(model.ema_update_count.item()) == 0
