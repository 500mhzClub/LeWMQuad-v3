from __future__ import annotations

import importlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runner = importlib.import_module(
    "scripts.run_go2_rgb_scene_action_contrastive_innovation_joint_jepa_v22"
)
v21_fixture = importlib.import_module(
    "lewm.tests.test_run_go2_rgb_same_action_cross_scene_contrastive_"
    "innovation_joint_jepa_v21"
)


def _valid_mask(count: int = 256) -> torch.Tensor:
    mask = torch.zeros(64 * 64, dtype=torch.bool)
    mask[:count] = True
    return mask.reshape(64, 64)


def _constant_objective(kind: str) -> Any:
    actions = torch.tensor((0, 1, 2, 3), dtype=torch.int64)
    negatives = torch.tensor((1, 2, 3, 0), dtype=torch.int64)
    current = torch.zeros((4, 64, 64, 64), dtype=torch.float32)
    scene = torch.tensor((0.2, -0.35, 0.55, -0.7), dtype=torch.float32)
    action = torch.linspace(-0.4, 0.4, 9)
    if kind == "scene":
        target_value = scene
        residual = scene[:, None].expand(-1, 9)
    elif kind == "action":
        target_value = action[actions]
        residual = action[None].expand(4, -1)
    elif kind == "both":
        target_value = scene + action[actions]
        residual = scene[:, None] + action[None]
    else:
        raise AssertionError(kind)
    predicted = residual[:, :, None, None, None].expand(-1, -1, 64, 64, 64).clone()
    ema_current = torch.zeros_like(current)
    ema_next = target_value[:, None, None, None].expand_as(current).clone()
    return runner.two_axis_innovation_objective_v22(
        torch,
        predicted,
        current,
        ema_current,
        ema_next,
        actions,
        negatives,
        _valid_mask(),
    )


def test_scene_action_and_joint_synthetic_falsifiers_are_axis_specific() -> None:
    scene_only = _constant_objective("scene")
    assert float(scene_only.scene_advantage.mean()) > 1.0e-4
    assert float(scene_only.action_advantage.abs().max()) < 1.0e-6
    assert float(scene_only.action_rank) == pytest.approx(1.0, abs=1.0e-6)

    action_only = _constant_objective("action")
    assert float(action_only.scene_advantage.abs().max()) < 1.0e-6
    assert float(action_only.scene_rank) == pytest.approx(1.0, abs=1.0e-6)
    assert float(action_only.action_advantage.mean()) > 1.0e-4

    joint = _constant_objective("both")
    assert float(joint.scene_advantage.mean()) > 1.0e-4
    assert float(joint.action_advantage.mean()) > 1.0e-4
    assert float(joint.loss) == pytest.approx(
        float(joint.fit + 0.5 * (joint.scene_rank + joint.action_rank)),
        rel=1.0e-6,
    )


def test_action_energy_is_mean_after_evaluation_and_excludes_requested() -> None:
    current = torch.zeros((4, 64, 64, 64), dtype=torch.float32)
    predicted = torch.zeros((4, 9, 64, 64, 64), dtype=torch.float32)
    values = torch.tensor((0.0, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0))
    predicted[:] = values[None, :, None, None, None]
    receipt = runner.two_axis_innovation_objective_v22(
        torch,
        predicted,
        current,
        torch.zeros_like(current),
        torch.zeros_like(current),
        torch.zeros(4, dtype=torch.int64),
        torch.tensor((1, 2, 3, 0), dtype=torch.int64),
        _valid_mask(),
    )
    assert torch.equal(
        receipt.nonrequested_actions,
        torch.arange(1, 9, dtype=torch.int64)[None].expand(4, -1),
    )
    assert torch.allclose(
        receipt.action_negative_energy,
        receipt.action_candidate_energy.mean(dim=1),
        atol=0.0,
        rtol=0.0,
    )
    expected_candidates = F.smooth_l1_loss(
        values[1:] / 1.0e-3,
        torch.zeros(8),
        beta=1.0,
        reduction="none",
    )
    assert torch.allclose(receipt.action_candidate_energy[0], expected_candidates)
    assert float(values[1:].mean()) == 0.0
    assert float(receipt.action_negative_energy[0]) > 0.0


def _install_tiny_apis(
    monkeypatch: pytest.MonkeyPatch,
    model: Any,
    partition: Any,
) -> None:
    class _SemanticApi:
        @staticmethod
        def semantic_loss_v1(current_logits, current_labels, next_logits, next_labels):
            current = F.cross_entropy(
                current_logits, current_labels, reduction="none"
            ).mean(dim=(1, 2))
            next_ = F.cross_entropy(
                next_logits, next_labels, reduction="none"
            ).mean(dim=(1, 2))
            return SimpleNamespace(loss=0.5 * (current.mean() + next_.mean()) / math.log(3.0))

        @staticmethod
        def microbatch_persistence_loss_v1(predicted, executed, _ema_current, ema_next):
            rows = torch.arange(4)
            return SimpleNamespace(loss=(predicted[rows, executed] - ema_next).square().mean())

    class _SurvivalApi:
        @staticmethod
        def joint_survival_loss_v1(**values):
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
                ranking_terms=SimpleNamespace(eligible_pair_count=torch.tensor(3)),
                survival_terms=SimpleNamespace(supervised_decision_count=torch.tensor(9)),
            )

    monkeypatch.setattr(
        runner._tensor_core,
        "_runtime_apis",
        lambda: (torch, _SemanticApi, _SurvivalApi, None, None, None, None),
    )
    monkeypatch.setattr(runner._v21, "_validate_microbatches_v21", lambda *_: None)
    monkeypatch.setattr(runner._base, "partition_parameters_v18", lambda _: partition)
    monkeypatch.setattr(runner._base, "validate_optimizer_v18", lambda *_: None)
    monkeypatch.setattr(
        runner._tensor_core._v3,
        "occupied_safety_aux_loss_v3",
        lambda current, _a, next_, _b: SimpleNamespace(
            loss=0.01 * (current.square().mean() + next_.square().mean())
        ),
    )
    monkeypatch.setattr(
        runner._tensor_core._v3._v2._v1,
        "_prediction_parts",
        lambda prediction: (prediction.predicted_latents, prediction.survival_logits),
    )
    monkeypatch.setattr(
        runner._base,
        "camera_evidence_pair_loss_v13",
        lambda current, next_, *_: SimpleNamespace(total=current.square() + next_.square()),
    )


def test_one_cpu_update_preserves_exact_forward_gradient_and_step_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = v21_fixture._TinyModel()
    partition = v21_fixture._partition(model)
    _install_tiny_apis(monkeypatch, model, partition)
    optimizer = v21_fixture._CountingSgd(list(partition.online))
    original_grad = torch.autograd.grad
    grad_parameter_ids: list[tuple[int, ...]] = []

    def counted_grad(*args: Any, **kwargs: Any) -> Any:
        grad_parameter_ids.append(tuple(id(value) for value in args[1]))
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", counted_grad)
    result = runner.joint_training_update_v22(
        model,
        optimizer,
        v21_fixture._microbatches(),
    )
    assert [len(value) for value in grad_parameter_ids] == [1, 17, 13] * 4
    assert model.predictor_forward_count == 4
    assert optimizer.step_calls == 1
    assert int(model.ema_update_count.item()) == 1
    assert result.target_gradient_tensor_count == 0
    assert model.target.grad is None
    assert result.accounting == runner.JointTrainingAccountingV22(
        updates=1,
        presentations=16,
        microbatch_graphs=4,
        backward_calls=12,
        camera_route_grad_calls=4,
        joint_route_grad_calls=4,
        two_axis_innovation_grad_calls=4,
        camera_frame_objectives=32,
        optimizer_steps=1,
        ema_steps=1,
        predictor_forwards=4,
        predictor_objectives=8,
        two_axis_innovation_objectives=4,
    )
    route = result.gradient_routes["two_axis_innovation_predictor"]
    assert route.parameter_tensor_count == 13
    assert route.absent_tensor_gradient_count == 0
    assert route.preclip_l2 > 0.0
    assert route.applied_scale == pytest.approx(min(1.0, 1.0 / route.preclip_l2))
    losses = result.mean_losses
    assert losses["I_two_axis"] == pytest.approx(
        losses["I_fit"] + 0.5 * (losses["I_scene_rank"] + losses["I_action_rank"]),
        rel=2.0e-6,
    )
    diagnostics = result.two_axis_innovation_diagnostics
    assert diagnostics["scene_advantage_count"] == 16
    assert diagnostics["action_advantage_count"] == 16
    assert diagnostics["nonrequested_action_count_per_row"] == 8
    assert diagnostics["action_candidate_energy_count"] == 128
    assert math.isfinite(float(diagnostics["matching_predictor_gradient_cosine"]))
