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

V21_NAME = (
    "scripts.run_go2_rgb_same_action_cross_scene_contrastive_innovation_"
    "joint_jepa_v21"
)
runner = importlib.import_module(V21_NAME)

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


def _valid_mask(count: int = 512) -> torch.Tensor:
    mask = torch.zeros((64 * 64,), dtype=torch.bool)
    mask[:count] = True
    return mask.reshape(64, 64)


def _objective_inputs(*, scene_conditioned: bool) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(20_260_730)
    current = 0.1 * torch.randn((4, 64, 64, 64), generator=generator)
    target = torch.zeros_like(current)
    target[0, :, 0:2, :] = 0.4
    target[1, :, 2:4, :] = -0.3
    target[2, :, 4:6, :] = 0.2
    target[3, :, 6:8, :] = -0.1
    actions = torch.tensor((0, 1, 2, 3), dtype=torch.int64)
    negatives = torch.tensor((1, 2, 3, 0), dtype=torch.int64)
    if scene_conditioned:
        residual = target
        predicted = current[:, None] + residual[:, None]
        predicted = predicted.expand(-1, 9, -1, -1, -1).contiguous()
    else:
        action_residual = torch.linspace(-0.2, 0.2, 9)[None, :, None, None, None]
        predicted = current[:, None] + action_residual
        predicted = predicted.expand(-1, -1, 64, 64, 64).contiguous()
    ema_current = torch.zeros_like(current)
    ema_next = target.clone()
    return (
        predicted.float(),
        current.float(),
        ema_current,
        ema_next,
        actions,
        negatives,
        _valid_mask(),
    )


def test_action_only_is_tied_and_scene_conditioning_has_material_advantage() -> None:
    tied = runner.scene_innovation_objective_v21(
        torch, *_objective_inputs(scene_conditioned=False)
    )
    assert torch.allclose(
        tied.positive_energy,
        tied.negative_energy,
        atol=runner.ACTION_ONLY_EQUALITY_TOLERANCE_V21,
        rtol=0.0,
    )
    assert float(tied.advantage.abs().max()) <= 1.0e-6
    assert float(tied.rank) == pytest.approx(1.0, abs=1.0e-6)

    conditioned = runner.scene_innovation_objective_v21(
        torch, *_objective_inputs(scene_conditioned=True)
    )
    assert float(conditioned.advantage.mean()) > runner.MATERIAL_SCENE_ADVANTAGE_V21
    assert float(conditioned.fit) < float(tied.fit)


def test_valid_only_scale_and_stable_flattened_salience_tie_order() -> None:
    current = torch.zeros((4, 64, 64, 64), dtype=torch.float32)
    predicted = current[:, None].expand(-1, 9, -1, -1, -1).clone()
    ema_current = torch.zeros_like(current)
    ema_next = torch.full_like(current, 100.0)
    valid = _valid_mask(256)
    ema_next[:, :, valid] = 1.0
    receipt = runner.scene_innovation_objective_v21(
        torch,
        predicted,
        current,
        ema_current,
        ema_next,
        torch.tensor((0, 1, 2, 3), dtype=torch.int64),
        torch.tensor((1, 2, 3, 0), dtype=torch.int64),
        valid,
    )
    assert receipt.valid_cell_count == 256
    assert torch.equal(receipt.scale, torch.ones(4))
    assert torch.equal(
        receipt.low_flat_indices,
        torch.arange(128, dtype=torch.int64)[None].expand(4, -1),
    )
    assert torch.equal(
        receipt.high_flat_indices,
        torch.arange(128, 256, dtype=torch.int64)[None].expand(4, -1),
    )
    assert not bool(
        (
            receipt.low_flat_indices[:, :, None]
            == receipt.high_flat_indices[:, None, :]
        ).any()
    )

    too_small = _valid_mask(255)
    with pytest.raises(ValueError, match="fewer than 256"):
        runner.scene_innovation_objective_v21(
            torch,
            predicted,
            current,
            ema_current,
            ema_next,
            torch.tensor((0, 1, 2, 3), dtype=torch.int64),
            torch.tensor((1, 2, 3, 0), dtype=torch.int64),
            too_small,
        )


def test_negative_rows_and_projected_batch_schema_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inherited_calls: list[tuple[dict[str, Any], ...]] = []
    monkeypatch.setattr(
        runner._base,
        "_validate_microbatches_v13",
        lambda _torch, values: inherited_calls.append(values),
    )
    batches = []
    for _ in range(4):
        batch = {
            name: torch.zeros((4,), dtype=torch.float32)
            for name in runner.REQUIRED_BATCH_KEYS_V21
        }
        batch[runner.CURRENT_RGB_KEY] = torch.zeros((4, 1), dtype=torch.float32)
        batch[runner.SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21] = torch.tensor(
            (1, 2, 3, 0), dtype=torch.int64
        )
        batches.append(batch)
    runner._validate_microbatches_v21(torch, tuple(batches))
    assert len(inherited_calls) == 1
    assert all(
        tuple(batch) == runner.INHERITED_REQUIRED_BATCH_KEYS_V21
        for batch in inherited_calls[0]
    )
    assert all(
        runner.SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21 not in batch
        for batch in inherited_calls[0]
    )

    malformed = [dict(batch) for batch in batches]
    malformed[0][runner.SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21] = torch.arange(4)
    with pytest.raises(ValueError, match="self match"):
        runner._validate_microbatches_v21(torch, tuple(malformed))
    wrong_dtype = [dict(batch) for batch in batches]
    wrong_dtype[0][runner.SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21] = torch.tensor(
        (1, 2, 3, 0), dtype=torch.int32
    )
    with pytest.raises(ValueError, match="int64"):
        runner._validate_microbatches_v21(torch, tuple(wrong_dtype))


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))
        self.representation = nn.Parameter(torch.tensor(0.15, dtype=torch.float32))
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
        self.bev_lift = SimpleNamespace(cell_valid_mask=_valid_mask(512))
        self.predictor_forward_count = 0

    def _latent(self, rgb: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        signal = rgb.float().reshape(4, -1).mean(dim=1)[:, None, None, None]
        return (signal + offset).expand(-1, 64, 64, 64)

    def encode_online_training(self, rgb: torch.Tensor, **_: Any) -> Any:
        latent = self._latent(rgb, self.shared + self.representation)
        return SimpleNamespace(
            latent=latent,
            auxiliary_evidence=latent.mean(),
        )

    def semantic_logits_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        value = latent.mean(dim=1, keepdim=True)
        return torch.cat((value, -value, 0.25 * value), dim=1)

    def predict_all_actions_with_survival(self, latent: torch.Tensor) -> Any:
        self.predictor_forward_count += 1
        transition_signal = sum(
            parameter.reshape(-1)[0] for parameter in self.transition
        )
        action_offsets = torch.arange(
            9, dtype=latent.dtype, device=latent.device
        )[None, :, None, None, None]
        predicted = latent[:, None] + 1.0e-3 * transition_signal + 1.0e-2 * action_offsets
        pooled = predicted.mean(dim=(2, 3, 4))
        survival_logits = (
            pooled * self.survival[0][0] + self.survival[1][0]
        )[:, :, None].expand(-1, -1, 16)
        return SimpleNamespace(
            predicted_latents=predicted,
            survival_logits=survival_logits,
        )

    def encode_target(self, rgb: torch.Tensor) -> torch.Tensor:
        return self._latent(rgb, self.target)

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
        for key in runner.REQUIRED_BATCH_KEYS_V21
    }
    rows = torch.tensor((0.0, 0.1, 0.2, 0.3), dtype=torch.float32)[:, None]
    values[runner.CURRENT_RGB_KEY] = rows + 0.2 + offset
    values[runner.NEXT_RGB_KEY] = rows + 0.4 + offset
    values[runner.CURRENT_LABELS_KEY] = torch.zeros((4, 64, 64), dtype=torch.long)
    values[runner.NEXT_LABELS_KEY] = torch.ones((4, 64, 64), dtype=torch.long)
    values[runner.EXECUTED_ACTION_KEY] = torch.tensor(
        (0, runner.HOLD_ACTION_INDEX_V21, 2, 3), dtype=torch.long
    )
    values[runner.IMMEDIATE_FEASIBLE_KEY] = torch.ones((4, 9), dtype=torch.bool)
    values[runner.PREFIX_LENGTHS_KEY] = torch.ones((4, 9), dtype=torch.long)
    for key in (runner.CURRENT_CAMERA_ORIGIN_KEY, runner.NEXT_CAMERA_ORIGIN_KEY):
        values[key] = torch.zeros((4, 3), dtype=torch.float32)
    for key in (runner.CURRENT_CAMERA_BASIS_KEY, runner.NEXT_CAMERA_BASIS_KEY):
        values[key] = torch.eye(3).expand(4, -1, -1).clone()
    values[runner.SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21] = torch.tensor(
        (1, 2, 3, 0), dtype=torch.int64
    )
    return values


def _microbatches() -> tuple[dict[str, torch.Tensor], ...]:
    return tuple(_one_microbatch(0.01 * index) for index in range(4))


def _accounting_after(updates: int) -> runner.JointTrainingAccountingV21:
    return runner.JointTrainingAccountingV21(
        updates=updates,
        presentations=16 * updates,
        microbatch_graphs=4 * updates,
        backward_calls=12 * updates,
        camera_route_grad_calls=4 * updates,
        joint_route_grad_calls=4 * updates,
        scene_innovation_grad_calls=4 * updates,
        camera_frame_objectives=32 * updates,
        optimizer_steps=updates,
        ema_steps=updates,
        predictor_forwards=4 * updates,
        predictor_objectives=8 * updates,
        scene_innovation_objectives=4 * updates,
    )


def _macro_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels, reduction="none").mean(dim=(1, 2))


def _install_tiny_apis(
    monkeypatch: pytest.MonkeyPatch, model: _TinyModel, partition: Any
) -> None:
    class _SemanticApi:
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
            return SimpleNamespace(
                loss=(predicted[rows, executed] - ema_next).square().mean()
            )

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
    monkeypatch.setattr(runner._base, "_validate_microbatches_v13", lambda *_: None)
    monkeypatch.setattr(runner._base, "partition_parameters_v18", lambda _: partition)
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
        lambda current, next_, *_: SimpleNamespace(total=current.square() + next_.square()),
    )


def test_adapter_subset_accounting_and_aliases_are_exact() -> None:
    receipt = runner.private_training_adapter_receipt_v21()
    assert receipt["preregistration_commit"] == "c2bbce067175dd980c9ed2511dc14db5a222afe4"
    assert receipt["preregistration_file_sha256"] == (
        "f4ff1453e5cb63677dad66253d568c9204bd5504b3b3871e2b0c341402b1850e"
    )
    assert receipt["preregistration_byte_count"] == 11_594
    assert receipt["negative_row_key"] == "scene_innovation_negative_row"
    assert runner.REQUIRED_BATCH_KEYS_V21 == (
        *runner.INHERITED_REQUIRED_BATCH_KEYS_V21,
        runner.SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21,
    )
    assert runner.joint_training_update_v13 is runner.joint_training_update_v21
    assert runner.validate_accounting_v13 is runner.validate_accounting_v21

    model = _TinyModel()
    partition = _partition(model)
    subset = runner.scene_innovation_predictor_subset_v21(partition)
    assert subset.names == TRANSITION_NAMES
    assert subset.parameters == tuple(model.transition)
    assert subset.predictor_indices == tuple(range(13))
    assert subset.parameter_count == 259_008
    changed_names = dict(partition.names)
    changed_names["predictor"] = (
        *TRANSITION_NAMES[:-1],
        "predictor.swept_progress_head.accidentally_excluded",
        *SURVIVAL_NAMES,
    )
    with pytest.raises(RuntimeError, match="subset changed"):
        runner.scene_innovation_predictor_subset_v21(
            replace(partition, names=changed_names)
        )

    runner.validate_accounting_v21(_accounting_after(0))
    runner.validate_accounting_v21(_accounting_after(1_000))
    with pytest.raises(RuntimeError, match="accounting is inconsistent"):
        runner.validate_accounting_v21(
            replace(_accounting_after(3), backward_calls=35)
        )
    with pytest.raises(PermissionError, match="no complete update"):
        runner._validate_update_capacity_v21(_accounting_after(1_000))


def test_one_update_routes_only_the_exact_norm_capped_predictor_subset(
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
    result = runner.joint_training_update_v21(model, optimizer, _microbatches())

    assert [len(value) for value in grad_parameter_ids] == [1, 17, 13] * 4
    transition_ids = tuple(id(value) for value in model.transition)
    assert grad_parameter_ids[2::3] == [transition_ids] * 4
    assert optimizer.step_calls == 1
    assert model.predictor_forward_count == 4
    assert int(model.ema_update_count.item()) == 1
    assert model.target.grad is None
    assert result.target_gradient_tensor_count == 0
    assert result.accounting == _accounting_after(1)
    assert set(result.gradient_routes) == {
        "camera_shared",
        "joint_shared",
        "representation",
        "predictor",
        "scene_innovation_predictor",
    }
    route = result.gradient_routes["scene_innovation_predictor"]
    assert route.parameter_tensor_count == 13
    assert route.absent_tensor_gradient_count == 0
    assert route.preclip_l2 > 0.0
    assert route.applied_scale == pytest.approx(
        min(1.0, 1.0 / route.preclip_l2), rel=1.0e-6
    )
    assert set(result.mean_losses) == {
        "S",
        "P",
        "U",
        "R",
        "O",
        "I_fit",
        "I_rank",
        "I_scene",
        "N",
        "C",
        "L",
    }
    assert result.mean_losses["I_scene"] == pytest.approx(
        result.mean_losses["I_fit"] + result.mean_losses["I_rank"], rel=2.0e-6
    )
    assert result.mean_losses["L"] == pytest.approx(
        result.mean_losses["N"]
        + result.mean_losses["C"]
        + result.mean_losses["I_scene"],
        rel=2.0e-6,
    )
    diagnostics = result.scene_innovation_diagnostics
    assert set(diagnostics) == {
        "positive_energy_mean",
        "negative_energy_mean",
        "advantage_sum",
        "advantage_count",
        "advantage_mean",
        "matching_predictor_gradient_cosine",
        "valid_cell_count",
        "high_salience_cell_count",
        "low_salience_cell_count",
    }
    assert diagnostics["advantage_count"] == 16
    assert diagnostics["advantage_mean"] == pytest.approx(
        diagnostics["advantage_sum"] / 16.0
    )
    assert diagnostics["valid_cell_count"] == 512
    assert diagnostics["high_salience_cell_count"] == 128
    assert diagnostics["low_salience_cell_count"] == 128
    assert math.isfinite(float(diagnostics["matching_predictor_gradient_cosine"]))


def test_scene_innovation_route_rejects_an_absent_transition_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyModel()
    partition = _partition(model)
    _install_tiny_apis(monkeypatch, model, partition)
    optimizer = _CountingSgd(list(partition.online))
    original_grad = torch.autograd.grad
    call = 0

    def drop_one_innovation_gradient(*args: Any, **kwargs: Any) -> Any:
        nonlocal call
        call += 1
        gradients = original_grad(*args, **kwargs)
        if call == 3:
            return (None, *gradients[1:])
        return gradients

    monkeypatch.setattr(torch.autograd, "grad", drop_one_innovation_gradient)
    with pytest.raises(RuntimeError, match="absent gradient"):
        runner.joint_training_update_v21(model, optimizer, _microbatches())
    assert optimizer.step_calls == 0
    assert int(model.ema_update_count.item()) == 0
