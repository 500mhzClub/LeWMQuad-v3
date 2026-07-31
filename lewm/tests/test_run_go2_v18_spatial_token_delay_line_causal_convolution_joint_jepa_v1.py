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
    "scripts.run_go2_v18_spatial_token_delay_line_causal_convolution_"
    "joint_jepa_v1"
)


def _accounting(updates: int) -> Any:
    return runner.JointTrainingAccountingV1(
        updates=updates,
        presentations=24 * updates,
        physical_presentations=8 * updates,
        memory_presentations=16 * updates,
        physical_microbatch_graphs=2 * updates,
        memory_microbatch_graphs=8 * updates,
        autograd_grad_calls=14 * updates,
        optimizer_steps=updates,
        ema_steps=updates,
    )


def test_accounting_and_1000_update_24000_presentation_cap_are_exact() -> None:
    runner.validate_accounting_v1(_accounting(0))
    runner.validate_accounting_v1(_accounting(1_000))
    runner._validate_capacity_v1(_accounting(999))
    with pytest.raises(PermissionError, match="no complete update"):
        runner._validate_capacity_v1(_accounting(1_000))
    with pytest.raises(RuntimeError, match="accounting is inconsistent"):
        runner.validate_accounting_v1(
            replace(_accounting(7), memory_presentations=111)
        )


def test_nested_physical_builder_schema_aliases_are_exact() -> None:
    assert runner.REQUIRED_BATCH_KEYS_V21 == (
        *runner.REQUIRED_BATCH_KEYS,
        runner.SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21,
    )
    assert runner.REQUIRED_BATCH_KEYS_V23 == (
        *runner.REQUIRED_BATCH_KEYS_V21,
        runner.ACTION_PRIOR_M_KEY_V23,
    )
    assert runner.REQUIRED_BATCH_KEYS_V24 == runner.REQUIRED_BATCH_KEYS_V23
    assert runner.REQUIRED_BATCH_KEYS_V25 == runner.REQUIRED_BATCH_KEYS_V24
    assert runner.REQUIRED_BATCH_KEYS_V21 == runner.v25.REQUIRED_BATCH_KEYS_V21
    assert runner.REQUIRED_BATCH_KEYS_V23 == runner.v25.REQUIRED_BATCH_KEYS_V23
    assert runner.REQUIRED_BATCH_KEYS_V24 == runner.v25.REQUIRED_BATCH_KEYS_V24


def test_overflow_safe_route_norm_matches_legacy_in_normal_and_zero_ranges() -> None:
    normal_gradients = (torch.tensor((3.0, 4.0), dtype=torch.float32),)
    legacy_norm, legacy_scale = (
        runner.v25._tensor_core._route_norm_and_scale_v13(
            torch, normal_gradients
        )
    )
    receipt, applied_scale = runner._route_norm_and_scale_v1(
        torch,
        "normal_reference",
        normal_gradients,
        parameter_tensor_count=1,
        absent_tensor_gradient_count=0,
    )
    assert receipt == runner.GradientRouteReceiptV1(
        route_name="normal_reference",
        raw_gradients_finite=True,
        maximum_absolute_raw_gradient=4.0,
        preclip_l2=pytest.approx(float(legacy_norm), abs=1.0e-7),
        applied_scale=pytest.approx(float(legacy_scale), abs=1.0e-7),
        parameter_tensor_count=1,
        absent_tensor_gradient_count=0,
    )
    assert torch.allclose(
        applied_scale, legacy_scale, rtol=1.0e-7, atol=1.0e-7
    )

    zero_gradients = (torch.zeros(5, dtype=torch.float32),)
    zero_legacy_norm, zero_legacy_scale = (
        runner.v25._tensor_core._route_norm_and_scale_v13(
            torch, zero_gradients
        )
    )
    zero_receipt, zero_scale = runner._route_norm_and_scale_v1(
        torch,
        "zero_reference",
        zero_gradients,
        parameter_tensor_count=1,
        absent_tensor_gradient_count=0,
    )
    assert zero_receipt.preclip_l2 == float(zero_legacy_norm) == 0.0
    assert zero_receipt.applied_scale == float(zero_legacy_scale) == 1.0
    assert float(zero_scale) == 1.0


@pytest.mark.parametrize("value", (float("nan"), float("inf")))
def test_overflow_safe_route_norm_rejects_named_nonfinite_raw_gradient(
    value: float,
) -> None:
    route_name = "injected_nonfinite_route"
    with pytest.raises(FloatingPointError) as caught:
        runner._route_norm_and_scale_v1(
            torch,
            route_name,
            (torch.tensor((1.0, value), dtype=torch.float32),),
            parameter_tensor_count=1,
            absent_tensor_gradient_count=0,
        )
    message = str(caught.value)
    assert f"route={route_name!r}" in message
    assert "stage=raw_gradient_finiteness" in message
    assert "raw_gradients_finite=False" in message
    assert "maximum_absolute_raw_gradient=" in message
    assert "preclip_l2=not_computed" in message
    assert "applied_scale=not_computed" in message


def _rollout_persistence_predictor(
    predictor: nn.Module,
    tokens: torch.Tensor,
    validity: torch.Tensor,
    action_tape: torch.Tensor,
) -> torch.Tensor:
    predictions = []
    current_tokens = tokens
    current_validity = validity
    for _ in range(4):
        prediction = predictor(
            current_tokens,
            current_validity,
            action_tape,
        )
        predictions.append(prediction)
        current_tokens = torch.cat(
            (prediction[:, None], current_tokens[:, :-1]),
            dim=1,
        )
        current_validity = torch.cat(
            (
                torch.ones(
                    (tokens.shape[0], 1),
                    dtype=torch.bool,
                    device=tokens.device,
                ),
                current_validity[:, :-1],
            ),
            dim=1,
        )
    return torch.stack(predictions, dim=1)


def test_registered_h4_masked_loss_overflow_is_clipped_to_unit_route() -> None:
    model_api = importlib.import_module(
        "lewm.models.v18_spatial_token_delay_line_causal_convolution_"
        "joint_jepa_v1"
    )
    torch.manual_seed(7)
    predictor = model_api.SpatialTokenDelayLineCausalConvolutionPredictorV1(
        model_api.V18SpatialTokenDelayLineCausalConvolutionJointJepaV1Config()
    )
    batch_size = 2
    tokens = F.normalize(
        torch.randn(batch_size, 4, 64, 16, 16),
        dim=2,
    )
    validity = torch.tensor(
        ((True, True, True, False),) * batch_size,
        dtype=torch.bool,
    )
    action_tape = F.one_hot(
        torch.zeros((batch_size, 4), dtype=torch.long),
        num_classes=9,
    ).to(dtype=torch.float32)
    block_rows = torch.arange(16) // 4
    keep = (
        block_rows[:, None] + block_rows[None, :]
    ).remainder(2).eq(1)
    masked_tokens = tokens.clone()
    masked_tokens[:, 0] = (
        masked_tokens[:, 0]
        * keep[None, None].to(dtype=masked_tokens.dtype)
    )
    target = F.normalize(
        torch.randn(batch_size, 4, 64, 16, 16),
        dim=2,
    )

    full_predictions = _rollout_persistence_predictor(
        predictor, tokens, validity, action_tape
    )
    masked_predictions = _rollout_persistence_predictor(
        predictor, masked_tokens, validity, action_tape
    )
    zero_counts = [
        int(
            (
                masked_predictions[:, horizon]
                .square()
                .sum(dim=1)
                .eq(0)
            )
            .sum()
            .item()
        )
        for horizon in range(4)
    ]
    assert zero_counts == [256, 256, 256, 256]

    full_loss = (
        1.0 - (full_predictions * target).sum(dim=2)
    ).mean()
    masked_loss = (
        1.0 - (masked_predictions * target).sum(dim=2)
    ).mean()
    registered_loss = (
        full_loss + 0.5 * masked_loss
    ) / runner.MEMORY_MICROBATCHES_PER_UPDATE_V1
    gradients = torch.autograd.grad(
        registered_loss,
        tuple(predictor.parameters()),
    )
    assert len(gradients) == 8
    assert all(bool(torch.isfinite(value).all()) for value in gradients)
    maximum = max(float(value.abs().max()) for value in gradients)
    assert float(full_loss.detach()) == pytest.approx(
        1.0020451545715332, abs=1.0e-7
    )
    assert float(masked_loss.detach()) == pytest.approx(
        1.0025625228881836, abs=1.0e-7
    )
    assert float(registered_loss.detach()) == pytest.approx(
        0.18791580200195312, abs=1.0e-7
    )
    assert maximum == pytest.approx(
        2.003031701502789e20, rel=1.0e-6
    )

    legacy_total = gradients[0].new_zeros((), dtype=torch.float32)
    for gradient in gradients:
        legacy_total = legacy_total + (
            gradient.float() * gradient.float()
        ).sum(dtype=torch.float32)
    assert bool(torch.isinf(torch.sqrt(legacy_total)))
    with pytest.raises(
        FloatingPointError, match="gradient norm or scale is nonfinite"
    ):
        runner.v25._tensor_core._route_norm_and_scale_v13(torch, gradients)

    receipt, applied_scale = runner._route_norm_and_scale_v1(
        torch,
        runner.MEMORY_ROUTE_NAME_V1,
        gradients,
        parameter_tensor_count=len(gradients),
        absent_tensor_gradient_count=0,
    )
    assert receipt.raw_gradients_finite is True
    assert receipt.maximum_absolute_raw_gradient == pytest.approx(
        maximum, rel=1.0e-6
    )
    assert receipt.preclip_l2 == pytest.approx(
        1.2587330137152002e21, rel=1.0e-6
    )
    assert math.isfinite(receipt.applied_scale)
    assert receipt.applied_scale > 0.0
    assert receipt.applied_scale == pytest.approx(
        1.0 / receipt.preclip_l2,
        rel=1.0e-6,
    )
    clipped = tuple(applied_scale * gradient for gradient in gradients)
    assert all(bool(torch.isfinite(value).all()) for value in clipped)
    clipped_total = clipped[0].new_zeros((), dtype=torch.float64)
    for gradient in clipped:
        value = gradient.to(dtype=torch.float64)
        clipped_total = clipped_total + (value * value).sum(
            dtype=torch.float64
        )
    assert float(torch.sqrt(clipped_total)) == pytest.approx(
        1.0, abs=1.0e-6
    )


def _strict_memory_batch() -> dict[str, torch.Tensor]:
    return {
        runner.MEMORY_HISTORY_RGB_KEY_V1: torch.zeros(
            (2, 3, 3, 112, 112), dtype=torch.float32
        ),
        runner.MEMORY_HISTORY_ACTIONS_KEY_V1: torch.tensor(
            ((0, 1), (2, 3)), dtype=torch.long
        ),
        runner.MEMORY_FUTURE_RGB_KEY_V1: torch.zeros(
            (2, 4, 3, 112, 112), dtype=torch.float32
        ),
        runner.MEMORY_FUTURE_ACTIONS_KEY_V1: torch.tensor(
            ((2, 3, 4, 5), (4, 5, 6, 7)), dtype=torch.long
        ),
    }


def test_memory_schema_is_exact_eight_b2_sequences_with_six_integer_actions() -> None:
    batch = _strict_memory_batch()
    runner._validate_memory_microbatches_v1(torch, (batch,) * 8)
    with pytest.raises(ValueError, match="exactly eight"):
        runner._validate_memory_microbatches_v1(torch, (batch,) * 7)
    invalid = dict(batch)
    invalid[runner.MEMORY_FUTURE_ACTIONS_KEY_V1] = torch.full(
        (2, 4), 9, dtype=torch.long
    )
    with pytest.raises(ValueError, match="future_actions"):
        runner._validate_memory_microbatches_v1(
            torch, (invalid,) + (batch,) * 7
        )


class _PartitionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(1, 1)
        self.bev_lift = nn.Module()
        self.bev_lift.evidence_head = nn.Linear(1, 1)
        self.bev_lift.point_projection = nn.Linear(1, 1)
        self.bev_lift.volume_block = nn.Linear(1, 1)
        self.semantic_head = nn.Linear(1, 1)
        self.predictor = nn.Linear(1, 1)
        self.memory_predictor = nn.Linear(1, 1)
        self.role_factorizer = nn.Linear(1, 1)
        self.place_predictor = nn.Linear(1, 1)
        self.local_predictor = nn.Linear(1, 1)
        self.target_encoder = nn.Linear(1, 1)
        self.target_bev_lift = nn.Module()
        self.target_bev_lift.evidence_head = nn.Linear(1, 1)
        self.target_bev_lift.point_projection = nn.Linear(1, 1)
        self.target_bev_lift.volume_block = nn.Linear(1, 1)
        self.target_role_factorizer = nn.Linear(1, 1)
        for module in (
            self.role_factorizer,
            self.place_predictor,
            self.local_predictor,
            self.target_encoder,
            self.target_bev_lift,
            self.target_role_factorizer,
        ):
            module.requires_grad_(False)


def test_partition_and_optimizer_exclude_target_and_diagnostic_heads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _PartitionModel()
    partition = runner.partition_parameters_v1(model)
    assert {id(value) for value in partition.online} == {
        id(value) for value in model.parameters() if value.requires_grad
    }
    assert {id(value) for value in partition.memory_predictor} == {
        id(value) for value in model.memory_predictor.parameters()
    }
    assert not (
        {id(value) for value in partition.online}
        & {
            id(value)
            for module in (
                model.role_factorizer,
                model.place_predictor,
                model.local_predictor,
                model.target_encoder,
                model.target_bev_lift,
                model.target_role_factorizer,
            )
            for value in module.parameters()
        }
    )
    monkeypatch.setattr(
        runner.v25._tensor_core,
        "_runtime_apis",
        lambda: (torch, None, None, None, None, None, None),
    )
    optimizer = runner.build_optimizer_v1(partition)
    runner.validate_optimizer_v1(optimizer, partition)
    assert [group["name"] for group in optimizer.param_groups] == [
        "encoder",
        "evidence_projection_semantic",
        "physical_and_memory_predictors",
    ]


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Parameter(torch.tensor(0.30))
        self.spatial = nn.Parameter(torch.tensor(0.20))
        self.semantic = nn.Parameter(torch.tensor(0.10))
        self.old_transition = nn.Parameter(torch.tensor(0.04))
        self.old_survival = nn.Parameter(torch.tensor(0.05))
        self.memory = nn.Parameter(torch.tensor(0.06))
        self.target = nn.Parameter(torch.tensor(0.25), requires_grad=False)
        self.register_buffer("ema_update_count", torch.zeros((), dtype=torch.long))
        self.physical_forwards = 0
        self.memory_forwards = 0
        self.ema_calls = 0
        self.seen_actions: list[torch.Tensor] = []

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

    def semantic_logits_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        value = latent + self.semantic
        return torch.cat((value, -value, 0.25 * value), dim=1)

    def predict_all_actions_with_survival(self, latent: torch.Tensor) -> Any:
        value = (latent + self.old_transition) * self.old_survival
        logits = value[:, None].expand(-1, 9, -1, -1, 16).reshape(-1, 9, 16)
        return SimpleNamespace(survival_logits=logits)

    def forward_memory(
        self,
        history_rgb: torch.Tensor,
        action_sequence: torch.Tensor,
        future_rgb: torch.Tensor,
    ) -> Any:
        self.memory_forwards += 1
        self.seen_actions.append(action_sequence.detach().clone())
        batch = history_rgb.shape[0]
        online = (
            history_rgb.reshape(batch, -1).mean(dim=1)
            + self.shared
            + self.spatial
            + self.memory
        )
        horizon = torch.arange(
            4, dtype=online.dtype, device=online.device
        )[None]
        action_signal = 0.01 * (
            action_sequence[:, 2:].argmax(dim=2).to(online.dtype)
        )
        values = online[:, None] + horizon + action_signal
        predicted = values[:, :, None, None, None].expand(-1, -1, 64, 16, 16)
        masked = predicted + 0.1 * self.memory
        target_values = (
            future_rgb.reshape(batch, 4, -1).mean(dim=2)
            + self.target.detach()
        )
        target = target_values[:, :, None, None, None].expand_as(predicted)
        full_loss = (predicted - target).square().mean()
        masked_loss = (masked - target).square().mean()
        keep = torch.zeros((batch, 1, 16, 16), dtype=torch.bool)
        keep[:, :, :8] = True
        return SimpleNamespace(
            target_future_tokens=target.detach(),
            full_predictions=predicted,
            masked_current_predictions=masked,
            newest_keep_mask=keep,
            full_loss=full_loss,
            masked_current_loss=masked_loss,
            loss=full_loss + 0.5 * masked_loss,
        )

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
    return runner.ParameterPartitionV1(
        encoder=(model.shared,),
        evidence_head=(),
        representation=(model.spatial, model.semantic),
        predictor=(model.old_transition, model.old_survival),
        memory_predictor=(model.memory,),
        target=(model.target,),
        frozen_diagnostics=(),
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
            "memory_predictor": ("memory_predictor.synthetic",),
            "target": ("target_encoder.synthetic",),
            "frozen_diagnostics": (),
        },
    )


def _physical_batch() -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {
        runner.CURRENT_RGB_KEY: torch.full((4, 1), 0.2),
        runner.NEXT_RGB_KEY: torch.full((4, 1), 0.4),
        runner.CURRENT_LABELS_KEY: torch.zeros((4, 1, 1), dtype=torch.long),
        runner.NEXT_LABELS_KEY: torch.ones((4, 1, 1), dtype=torch.long),
        runner.IMMEDIATE_FEASIBLE_KEY: torch.ones((4, 9), dtype=torch.bool),
        runner.PREFIX_LENGTHS_KEY: torch.ones((4, 9), dtype=torch.long),
        runner.SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21: torch.tensor((1, 2, 3, 0)),
        runner.ACTION_PRIOR_M_KEY_V23: torch.full((9,), 0.55),
    }
    for key in (
        runner.CURRENT_CAMERA_ORIGIN_KEY,
        runner.NEXT_CAMERA_ORIGIN_KEY,
    ):
        batch[key] = torch.zeros((4, 3))
    for key in (
        runner.CURRENT_CAMERA_BASIS_KEY,
        runner.NEXT_CAMERA_BASIS_KEY,
    ):
        batch[key] = torch.eye(3).expand(4, -1, -1).clone()
    for key in (
        runner.CURRENT_GROUND_PLANE_Z_KEY,
        runner.NEXT_GROUND_PLANE_Z_KEY,
        runner.CURRENT_PIXEL_HIT_KEY,
        runner.CURRENT_PIXEL_DISTANCE_KEY,
        runner.CURRENT_GROUND_IN_FRUSTUM_KEY,
        runner.CURRENT_GROUND_CLEAR_KEY,
        runner.NEXT_PIXEL_HIT_KEY,
        runner.NEXT_PIXEL_DISTANCE_KEY,
        runner.NEXT_GROUND_IN_FRUSTUM_KEY,
        runner.NEXT_GROUND_CLEAR_KEY,
    ):
        batch[key] = torch.zeros((4,))
    return batch


def _tiny_memory_batch() -> dict[str, torch.Tensor]:
    return {
        runner.MEMORY_HISTORY_RGB_KEY_V1: torch.tensor(
            (((0.1,), (0.2,), (0.3,)), ((0.4,), (0.5,), (0.6,))),
            dtype=torch.float32,
        ),
        runner.MEMORY_HISTORY_ACTIONS_KEY_V1: torch.tensor(
            ((0, 1), (2, 3)), dtype=torch.long
        ),
        runner.MEMORY_FUTURE_RGB_KEY_V1: torch.tensor(
            (((0.4,), (0.5,), (0.6,), (0.7,)),
             ((0.7,), (0.8,), (0.9,), (1.0,))),
            dtype=torch.float32,
        ),
        runner.MEMORY_FUTURE_ACTIONS_KEY_V1: torch.tensor(
            ((2, 3, 4, 5), (4, 5, 6, 7)), dtype=torch.long
        ),
    }


def test_one_mixed_cpu_update_has_exact_routes_step_ema_and_action_tape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyModel()
    partition = _partition(model)

    class _SemanticApi:
        @staticmethod
        def semantic_loss_v1(current_logits, current_labels, next_logits, next_labels):
            loss = 0.5 * (
                F.cross_entropy(current_logits, current_labels)
                + F.cross_entropy(next_logits, next_labels)
            ) / math.log(3.0)
            return SimpleNamespace(loss=loss)

    class _SurvivalApi:
        @staticmethod
        def joint_survival_loss_v1(**values: Any) -> Any:
            survival = 0.05 * values["survival_logits"].square().mean()
            ranking = 0.02 * values["survival_logits"].square().mean()
            semantic = values["semantic_loss"]
            return SimpleNamespace(
                loss=semantic + values["executed_action_ema_latent_loss"]
                + survival
                + ranking,
                semantic=semantic,
                survival=survival,
                progress_ranking=ranking,
                ranking_terms=SimpleNamespace(eligible_pair_count=torch.tensor(3)),
                survival_terms=SimpleNamespace(
                    supervised_decision_count=torch.tensor(9)
                ),
            )

    def auxiliary_objective(_torch, _survival_api, logits, *_args):
        base = logits.square().mean()
        return SimpleNamespace(loss=base)

    subset = SimpleNamespace(
        parameters=(model.shared, model.old_survival),
        protected_predictor_core_parameters=(model.old_transition,),
    )
    monkeypatch.setattr(
        runner.v25._tensor_core,
        "_runtime_apis",
        lambda: (torch, _SemanticApi, _SurvivalApi, None, None, None, None),
    )
    monkeypatch.setattr(runner, "_validate_physical_microbatches_v1", lambda *_: None)
    monkeypatch.setattr(runner, "_validate_memory_microbatches_v1", lambda *_: None)
    monkeypatch.setattr(runner, "partition_parameters_v1", lambda _: partition)
    monkeypatch.setattr(runner, "validate_optimizer_v1", lambda *_: None)
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

    original_grad = torch.autograd.grad
    gradient_parameter_counts: list[int] = []

    def counted_grad(*args: Any, **kwargs: Any) -> Any:
        gradient_parameter_counts.append(len(args[1]))
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", counted_grad)
    optimizer = _CountingSgd(list(partition.online))
    result = runner.joint_training_update_v1(
        model,
        optimizer,
        (_physical_batch(),) * 2,
        (_tiny_memory_batch(),) * 8,
    )

    assert gradient_parameter_counts == [1, 5, 2] * 2 + [3] * 8
    assert optimizer.step_calls == 1
    assert model.ema_calls == 1
    assert int(model.ema_update_count) == 1
    assert model.physical_forwards == 4
    assert model.memory_forwards == 8
    assert all(tuple(value.shape) == (2, 6, 9) for value in model.seen_actions)
    expected_actions = torch.tensor(
        ((0, 1, 2, 3, 4, 5), (2, 3, 4, 5, 6, 7))
    )
    assert all(
        torch.equal(value.argmax(dim=2), expected_actions)
        for value in model.seen_actions
    )
    assert model.target.grad is None
    assert result.target_gradient_tensor_count == 0
    assert result.optimizer_steps_this_update == 1
    assert result.ema_steps_this_update == 1
    assert result.accounting == _accounting(1)
    assert result.memory_diagnostics["newest_keep_fraction"] == 0.5
    assert result.memory_diagnostics["future_online_access_count"] == 0
    assert set(result.gradient_routes) == {
        "camera_shared",
        "joint_shared",
        "representation",
        "predictor",
        runner.PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25,
        runner.MEMORY_ROUTE_NAME_V1,
    }
    assert all(
        isinstance(receipt, runner.GradientRouteReceiptV1)
        and receipt.route_name == name
        and receipt.raw_gradients_finite is True
        and math.isfinite(receipt.maximum_absolute_raw_gradient)
        and math.isfinite(receipt.preclip_l2)
        and math.isfinite(receipt.applied_scale)
        for name, receipt in result.gradient_routes.items()
    )
