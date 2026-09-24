from __future__ import annotations

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
    "scripts.run_go2_rgb_memory_role_factorized_joint_jepa_v1"
)


def _accounting(updates: int) -> Any:
    return runner.JointTrainingAccountingV1(
        updates=updates,
        presentations=32 * updates,
        physical_presentations=16 * updates,
        local_presentations=8 * updates,
        place_presentations=8 * updates,
        rgb_decodes=72 * updates,
        physical_rgb_decodes=32 * updates,
        local_rgb_decodes=16 * updates,
        place_rgb_decodes=24 * updates,
        online_rgb_encodings=48 * updates,
        ema_target_rgb_encodings=24 * updates,
        physical_microbatch_graphs=4 * updates,
        local_microbatch_graphs=2 * updates,
        place_microbatch_graphs=updates,
        autograd_grad_calls=15 * updates,
        optimizer_steps=updates,
        ema_steps=updates,
    )


def _local_batch() -> dict[str, torch.Tensor]:
    return {
        runner.LOCAL_CURRENT_RGB_KEY_V1: torch.full((4, 3, 112, 112), 0.2),
        runner.LOCAL_NEXT_RGB_KEY_V1: torch.full((4, 3, 112, 112), 0.4),
        runner.LOCAL_ACTION_KEY_V1: torch.tensor((0, 2, 5, 8)),
    }


def _place_batch() -> dict[str, torch.Tensor]:
    return {
        runner.PLACE_ANCHOR_RGB_KEY_V1: torch.full((4, 3, 112, 112), 0.2),
        runner.PLACE_POSITIVE_RGB_KEY_V1: torch.full((4, 3, 112, 112), 0.3),
        runner.PLACE_NEGATIVE_RGB_KEY_V1: torch.full((4, 3, 112, 112), 0.8),
    }


class _PlaceHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.07))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        basis = torch.linspace(-1.0, 1.0, 64, device=value.device)
        return value + self.weight * basis


class _LocalHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.08))

    def forward(
        self, value: torch.Tensor, action_one_hot: torch.Tensor
    ) -> torch.Tensor:
        indices = torch.arange(9, dtype=value.dtype, device=value.device)
        action = (action_one_hot * indices).sum(dim=1) / 8.0
        return value + self.weight * action[:, None, None, None]


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Parameter(torch.tensor(0.30))
        self.spatial = nn.Parameter(torch.tensor(0.20))
        self.semantic = nn.Parameter(torch.tensor(0.10))
        self.old_transition = nn.Parameter(torch.tensor(0.04))
        self.old_survival = nn.Parameter(torch.tensor(0.05))
        self.place_factor = nn.Parameter(torch.tensor(0.06))
        self.local_factor = nn.Parameter(torch.tensor(0.09))
        self.place_predictor = _PlaceHead()
        self.local_predictor = _LocalHead()
        self.target = nn.Parameter(torch.tensor(0.25), requires_grad=False)
        self.register_buffer("ema_update_count", torch.zeros((), dtype=torch.long))
        self.physical_forwards = 0
        self.role_forwards = 0
        self.target_calls = 0
        self.target_rows = 0
        self.ema_calls = 0

    def _signal(self, rgb: torch.Tensor) -> torch.Tensor:
        return rgb.float().reshape(rgb.shape[0], -1).mean(dim=1)

    def _latent(self, rgb: torch.Tensor) -> torch.Tensor:
        return (self._signal(rgb) + self.shared + self.spatial)[:, None, None, None]

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

    def encode_online_roles(self, rgb: torch.Tensor) -> Any:
        self.role_forwards += 1
        signal = self._signal(rgb) + self.shared + self.spatial
        basis = torch.linspace(-1.0, 1.0, 64, device=rgb.device)
        place = signal[:, None] + self.place_factor * basis
        local = (signal + self.local_factor)[:, None, None, None].expand(
            -1, 32, 16, 16
        )
        return SimpleNamespace(place_key=place, local_control=local)

    def encode_target_roles(self, rgb: torch.Tensor) -> Any:
        self.target_calls += 1
        self.target_rows += rgb.shape[0]
        signal = self._signal(rgb) + self.target
        basis = torch.linspace(-0.5, 1.0, 64, device=rgb.device)
        return SimpleNamespace(
            place_key=signal[:, None] + self.target * basis,
            local_control=signal[:, None, None, None].expand(-1, 32, 16, 16),
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
        role_factorizer=(model.place_factor, model.local_factor),
        place_predictor=(model.place_predictor.weight,),
        local_predictor=(model.local_predictor.weight,),
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
            "role_factorizer": (
                "role_factorizer.place_projection.synthetic",
                "role_factorizer.local_projection.synthetic",
            ),
            "place_predictor": ("place_predictor.synthetic",),
            "local_predictor": ("local_predictor.synthetic",),
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


def test_v3_place_objective_matches_exact_registered_formula() -> None:
    generator = torch.Generator().manual_seed(19)
    online_anchor_keys = torch.randn(
        8, 64, generator=generator, dtype=torch.float32
    ).requires_grad_()
    predictions = F.normalize(online_anchor_keys + 0.07, dim=1)
    positive_targets = F.normalize(
        torch.randn(8, 64, generator=generator, dtype=torch.float32), dim=1
    )
    negative_targets = F.normalize(
        torch.randn(8, 64, generator=generator, dtype=torch.float32), dim=1
    )

    terms = runner.place_objective_v3(
        torch,
        online_anchor_keys,
        predictions,
        positive_targets,
        negative_targets,
    )

    candidates = torch.cat((positive_targets, negative_targets), dim=0)
    expected_logits = predictions @ candidates.T / 0.10
    expected_alignment = (
        1.0 - F.cosine_similarity(predictions, positive_targets, dim=1, eps=1e-6)
    ).mean()
    expected_contrast = F.cross_entropy(expected_logits, torch.arange(8))
    centered = online_anchor_keys - online_anchor_keys.mean(dim=0, keepdim=True)
    expected_covariance_matrix = centered.T @ centered / 7.0
    expected_variance = F.relu(
        0.05 - torch.sqrt(expected_covariance_matrix.diagonal() + 1e-4)
    ).mean()
    off_diagonal = ~torch.eye(64, dtype=torch.bool)
    expected_covariance = (
        expected_covariance_matrix.square().masked_select(off_diagonal).sum()
        / 64.0
    )
    expected_loss = (
        expected_alignment
        + expected_contrast
        + expected_variance
        + 0.10 * expected_covariance
    )

    torch.testing.assert_close(terms.logits, expected_logits)
    torch.testing.assert_close(terms.alignment, expected_alignment)
    torch.testing.assert_close(terms.contrast, expected_contrast)
    torch.testing.assert_close(terms.variance, expected_variance)
    torch.testing.assert_close(terms.covariance, expected_covariance)
    torch.testing.assert_close(terms.loss, expected_loss)
    torch.testing.assert_close(
        terms.logits[:, :8], predictions @ positive_targets.T / 0.10
    )
    torch.testing.assert_close(
        terms.logits[:, 8:], predictions @ negative_targets.T / 0.10
    )
    terms.loss.backward()
    assert online_anchor_keys.grad is not None
    assert bool(torch.isfinite(online_anchor_keys.grad).all())


def test_one_mixed_update_has_exact_routes_counts_and_one_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner.validate_accounting_v1(_accounting(400))
    runner._validate_capacity_v1(_accounting(399))
    with pytest.raises(PermissionError, match="no complete update"):
        runner._validate_capacity_v1(_accounting(400))
    runner._validate_role_microbatches_v1(
        torch, (_local_batch(),) * 2, (_place_batch(),) * 2
    )
    local_with_metadata = dict(_local_batch())
    local_with_metadata["cell_id"] = torch.zeros(4, dtype=torch.long)
    with pytest.raises(ValueError, match="local batch keys changed"):
        runner._validate_role_microbatches_v1(
            torch, (local_with_metadata,) * 2, (_place_batch(),) * 2
        )

    model = _TinyModel()
    partition = _partition(model)
    zero_temporal_terms: list[float] = []

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
    gradient_parameter_ids: list[tuple[int, ...]] = []

    def counted_grad(*args: Any, **kwargs: Any) -> Any:
        gradient_parameter_ids.append(tuple(id(value) for value in args[1]))
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", counted_grad)
    optimizer = _CountingSgd(list(partition.online))
    result = runner.joint_training_update_v1(
        model,
        optimizer,
        (_physical_batch(),) * 4,
        (_local_batch(),) * 2,
        (_place_batch(),) * 2,
    )

    assert [len(values) for values in gradient_parameter_ids] == (
        [1, 5, 2] * 4 + [4] * 2 + [4]
    )
    assert gradient_parameter_ids[12:14] == [
        tuple(map(id, partition.local_recipients))
    ] * 2
    assert gradient_parameter_ids[14:] == [
        tuple(map(id, partition.place_recipients))
    ]
    assert optimizer.step_calls == 1
    assert model.ema_calls == 1
    assert int(model.ema_update_count) == 1
    assert model.physical_forwards == 8
    assert model.role_forwards == 3
    assert model.target_calls == 3
    assert model.target_rows == 24
    assert model.target.grad is None
    assert result.target_gradient_tensor_count == 0
    assert result.optimizer_steps_this_update == 1
    assert result.ema_steps_this_update == 1
    assert result.accounting == _accounting(1)
    assert result.mean_losses.keys() >= {"C", "N", "J24", "local", "place", "total"}
    assert result.local_diagnostics["mechanism"] == runner.LOCAL_ROUTE_NAME_V1
    assert result.place_diagnostics["mechanism"] == runner.PLACE_ROUTE_NAME_V1
    assert result.local_diagnostics["correct_energy"]["count"] == 8
    assert result.place_diagnostics["positive_energy"]["count"] == 8
    assert result.place_diagnostics["objective_version"] == 3
    assert result.place_diagnostics["candidate_count"] == 16
    assert result.place_diagnostics["contrast_temperature"] == 0.10
    assert result.place_diagnostics["variance_floor"] == 0.05
    assert result.place_diagnostics["covariance_weight"] == 0.10
    assert "margin_loss_per_row" not in result.place_diagnostics
    assert "hard_negative_margin" not in result.place_diagnostics
    assert zero_temporal_terms == [0.0] * 4
    assert {id(model.semantic), id(model.old_transition)}.isdisjoint(
        map(id, partition.local_recipients)
    )
    assert {id(model.semantic), id(model.old_survival)}.isdisjoint(
        map(id, partition.place_recipients)
    )
