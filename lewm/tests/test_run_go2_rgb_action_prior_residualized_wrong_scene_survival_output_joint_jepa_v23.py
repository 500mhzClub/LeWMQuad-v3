from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / (
    "scripts/run_go2_rgb_action_prior_residualized_wrong_scene_survival_"
    "output_joint_jepa_v23.py"
)
V21_FIXTURE_PATH = ROOT / (
    "lewm/tests/test_run_go2_rgb_same_action_cross_scene_contrastive_"
    "innovation_joint_jepa_v21.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("_v23_training_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_v21_fixture():
    spec = importlib.util.spec_from_file_location(
        "_v21_training_fixture_for_v23", V21_FIXTURE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _DirectScores:
    @staticmethod
    def survival_scores_v1(logits: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(expected_progress_m=logits[..., 0])


def _fixture():
    module = _load_module()
    prefix = torch.tensor(
        [
            [0, 2, 4, 6, 8, 10, 7, 12, 14],
            [14, 12, 10, 8, 6, 4, 7, 2, 0],
            [2, 4, 6, 8, 10, 12, 7, 14, 0],
            [12, 10, 8, 6, 4, 2, 7, 0, 14],
        ],
        dtype=torch.int64,
    )
    negative = torch.tensor([1, 0, 3, 2], dtype=torch.int64)
    prior = torch.full((9,), 0.55, dtype=torch.float32)
    logits = torch.zeros((4, 9, 16), dtype=torch.float32, requires_grad=True)
    return module, logits, prefix, negative, prior


def test_objective_matches_registered_count_weighted_equations_and_gradients() -> None:
    module, logits, prefix, negative, prior = _fixture()
    actions = torch.tensor(module.NON_HOLD_ACTION_INDICES_V23)
    target = prefix.float() * 0.1
    with torch.no_grad():
        logits[..., 0].copy_(target)
    result = module.state_residual_survival_objective_v23(
        torch, _DirectScores, logits, prefix, negative, prior
    )
    q = logits[..., 0].index_select(1, actions)
    t = target.index_select(1, actions)
    q_scene = logits[..., 0].index_select(0, negative).index_select(1, actions)
    t_scene = target.index_select(0, negative).index_select(1, actions)
    prior_grid = prior.index_select(0, actions)[None].expand(4, -1)
    positive = torch.nn.functional.smooth_l1_loss(
        q / 1.5, t / 1.5, reduction="none"
    )
    scene = torch.nn.functional.smooth_l1_loss(
        q_scene / 1.5, t / 1.5, reduction="none"
    )
    prior_energy = torch.nn.functional.smooth_l1_loss(
        prior_grid / 1.5, t / 1.5, reduction="none"
    )
    scene_mask = t != t_scene
    prior_mask = prior_energy > 0
    expected_rank = torch.cat(
        (
            torch.nn.functional.softplus(positive[scene_mask] - scene[scene_mask]),
            torch.nn.functional.softplus(
                positive[prior_mask] - prior_energy[prior_mask]
            ),
        )
    ).mean() / math.log(2.0)
    assert torch.allclose(result.fit, positive.mean())
    assert torch.allclose(result.rank, expected_rank)
    assert torch.allclose(result.loss, result.fit + result.rank)
    assert result.scene_eligible_count == int(scene_mask.sum())
    assert result.prior_eligible_count == int(prior_mask.sum())
    assert float(result.scene_advantage_sum.detach()) > 0.0
    assert float(result.prior_advantage_sum.detach()) > 0.0
    assert float(result.rank.detach()) < 1.0
    result.loss.backward()
    assert logits.grad is not None
    assert bool((logits.grad[..., 0].abs().sum(dim=1) > 0).all())
    assert float(logits.grad[..., 1:].abs().sum()) == 0.0


def test_prior_template_is_an_exact_rank_tie_and_hold_is_excluded() -> None:
    module, logits, prefix, negative, prior = _fixture()
    with torch.no_grad():
        logits[..., 0].copy_(prior[None].expand(4, -1))
    result = module.state_residual_survival_objective_v23(
        torch, _DirectScores, logits, prefix, negative, prior
    )
    assert float(result.rank.detach()) == pytest.approx(1.0, abs=1e-7)
    assert float(result.scene_advantage_sum.detach()) == pytest.approx(0.0, abs=1e-7)
    assert float(result.prior_advantage_sum.detach()) == pytest.approx(0.0, abs=1e-7)

    changed_prefix = prefix.clone()
    changed_prefix[:, 6] = torch.tensor([0, 15, 1, 14])
    changed_logits = logits.detach().clone().requires_grad_(True)
    with torch.no_grad():
        changed_logits[:, 6, 0] = torch.tensor([1.5, 0.0, 1.4, 0.1])
    changed = module.state_residual_survival_objective_v23(
        torch, _DirectScores, changed_logits, changed_prefix, negative, prior
    )
    assert torch.equal(result.loss.detach(), changed.loss.detach())


def test_each_empty_comparison_axis_fails_closed() -> None:
    module, logits, prefix, negative, prior = _fixture()
    same_scene_targets = prefix[0].repeat(4, 1)
    with pytest.raises(ValueError, match="both scene and prior"):
        module.state_residual_survival_objective_v23(
            torch, _DirectScores, logits, same_scene_targets, negative, prior
        )

    exact_prior = prefix[0].float() * 0.1
    varied = prefix.clone()
    varied[:, :] = prefix[0]
    varied[1, 0] = 1
    varied[3, 1] = 3
    exact_prior[0] = 0.0
    exact_prior[1] = 0.2
    # Make every target equal to the per-action prior while retaining a scene
    # mismatch only in HOLD, which is deliberately excluded.
    prior_prefix = prefix[0].repeat(4, 1)
    prior_prefix[1, 6] = 8
    with pytest.raises(ValueError, match="both scene and prior"):
        module.state_residual_survival_objective_v23(
            torch,
            _DirectScores,
            logits,
            prior_prefix,
            negative,
            prefix[0].float() * 0.1,
        )


def _parameters(counts: list[int]) -> tuple[torch.nn.Parameter, ...]:
    return tuple(torch.nn.Parameter(torch.zeros(count)) for count in counts)


def test_parameter_subset_is_exact_full_output_path_and_all_groups_receive_gradient() -> None:
    module = _load_module()
    encoder = _parameters([1] * 79 + [3_102_737])
    evidence = _parameters([1] * 7 + [0])
    # Move one parameter from encoder to evidence without changing the total.
    evidence = (*evidence[:-1], torch.nn.Parameter(torch.zeros(8)))
    encoder = (*encoder[:-1], torch.nn.Parameter(torch.zeros(3_102_730)))
    representation_live = _parameters([1] * 5 + [3_515])
    semantic = _parameters([1] * 5 + [73_981])
    representation = (*representation_live, *semantic)
    predictor = _parameters([1] * 13 + [64, 1])
    predictor = (*predictor[:-3], torch.nn.Parameter(torch.zeros(258_996)), *predictor[-2:])
    names = {
        "encoder": tuple(f"encoder.p{i}" for i in range(len(encoder))),
        "evidence_head": tuple(
            f"bev_lift.evidence_head.p{i}" for i in range(len(evidence))
        ),
        "representation": (
            *(f"bev_lift.point_projection.p{i}" for i in range(3)),
            *(f"bev_lift.volume_block.p{i}" for i in range(3)),
            *(f"semantic_head.p{i}" for i in range(6)),
        ),
        "predictor": (
            *(f"predictor.core.p{i}" for i in range(len(predictor) - 2)),
            "predictor.swept_progress_head.output.weight",
            "predictor.swept_progress_head.output.bias",
        ),
    }
    partition = SimpleNamespace(
        encoder=encoder,
        evidence_head=evidence,
        representation=representation,
        predictor=predictor,
        names=names,
    )
    subset = module.state_residual_survival_parameter_subset_v23(partition)
    assert len(subset.parameters) == 109
    assert subset.parameter_count == 3_365_417
    assert all(not name.startswith("semantic_head.") for name in subset.names)
    assert subset.names[-2:] == (
        "predictor.swept_progress_head.output.weight",
        "predictor.swept_progress_head.output.bias",
    )
    signal = torch.stack(
        tuple(parameter.reshape(-1)[0] for parameter in subset.parameters)
    ).sum()
    survival_logits = signal.expand(4, 9, 16)
    prefix = torch.tensor(
        (
            (0, 2, 4, 6, 8, 10, 7, 12, 14),
            (14, 12, 10, 8, 6, 4, 7, 2, 0),
            (2, 4, 6, 8, 10, 12, 7, 14, 0),
            (12, 10, 8, 6, 4, 2, 7, 0, 14),
        ),
        dtype=torch.int64,
    )
    objective = module.state_residual_survival_objective_v23(
        torch,
        _DirectScores,
        survival_logits,
        prefix,
        torch.tensor((1, 0, 3, 2), dtype=torch.int64),
        torch.full((9,), 0.55, dtype=torch.float32),
    )
    gradients = torch.autograd.grad(
        objective.loss, subset.parameters, allow_unused=True
    )
    assert len(gradients) == 109
    assert all(
        gradient is not None
        and bool(torch.isfinite(gradient).all())
        and float(gradient.abs().sum()) > 0.0
        for gradient in gradients
    )
    for prefix_name in (
        "encoder.",
        "bev_lift.evidence_head.",
        "bev_lift.point_projection.",
        "bev_lift.volume_block.",
        "predictor.",
    ):
        assert any(
            name.startswith(prefix_name) and float(gradient.abs().sum()) > 0.0
            for name, gradient in zip(subset.names, gradients, strict=True)
        )


def test_accounting_remains_sixteen_presentations_and_twelve_grad_calls() -> None:
    module = _load_module()
    value = module.JointTrainingAccountingV23(
        updates=2,
        presentations=32,
        microbatch_graphs=8,
        backward_calls=24,
        camera_route_grad_calls=8,
        joint_route_grad_calls=8,
        state_residual_survival_grad_calls=8,
        camera_frame_objectives=64,
        optimizer_steps=2,
        ema_steps=2,
        predictor_forwards=8,
        predictor_objectives=16,
        state_residual_survival_objectives=8,
    )
    module.validate_accounting_v23(value)


def test_one_cpu_update_exercises_v23_route_and_exact_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    v21_fixture = _load_v21_fixture()
    model = v21_fixture._TinyModel()
    partition = v21_fixture._partition(model)

    class _SemanticApi:
        @staticmethod
        def semantic_loss_v1(current_logits, current_labels, next_logits, next_labels):
            current = F.cross_entropy(
                current_logits, current_labels, reduction="none"
            ).mean(dim=(1, 2))
            next_ = F.cross_entropy(
                next_logits, next_labels, reduction="none"
            ).mean(dim=(1, 2))
            return SimpleNamespace(
                loss=0.5 * (current.mean() + next_.mean()) / math.log(3.0)
            )

        @staticmethod
        def microbatch_persistence_loss_v1(predicted, executed, _ema_current, ema_next):
            rows = torch.arange(4)
            return SimpleNamespace(
                loss=(predicted[rows, executed] - ema_next).square().mean()
            )

    class _SurvivalApi:
        @staticmethod
        def survival_scores_v1(logits):
            probabilities = torch.sigmoid(logits)
            survival = probabilities[..., :1] * probabilities[..., 1:].cumprod(
                dim=-1
            )
            return SimpleNamespace(expected_progress_m=0.1 * survival.sum(dim=-1))

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
                survival_terms=SimpleNamespace(
                    supervised_decision_count=torch.tensor(9)
                ),
            )

    monkeypatch.setattr(
        module._tensor_core,
        "_runtime_apis",
        lambda: (torch, _SemanticApi, _SurvivalApi, None, None, None, None),
    )
    monkeypatch.setattr(module._v21, "_validate_microbatches_v21", lambda *_: None)
    monkeypatch.setattr(module._base, "partition_parameters_v18", lambda _: partition)
    monkeypatch.setattr(module._base, "validate_optimizer_v18", lambda *_: None)
    monkeypatch.setattr(
        module._tensor_core._v3,
        "occupied_safety_aux_loss_v3",
        lambda current, _a, next_, _b: SimpleNamespace(
            loss=0.01 * (current.square().mean() + next_.square().mean())
        ),
    )
    monkeypatch.setattr(
        module._tensor_core._v3._v2._v1,
        "_prediction_parts",
        lambda prediction: (prediction.predicted_latents, prediction.survival_logits),
    )
    monkeypatch.setattr(
        module._base,
        "camera_evidence_pair_loss_v13",
        lambda current, next_, *_: SimpleNamespace(total=current.square() + next_.square()),
    )

    auxiliary_parameters = partition.shared + partition.predictor
    auxiliary_names = partition.names["encoder"] + partition.names["predictor"]
    auxiliary_subset = module.StateResidualSurvivalParameterSubsetV23(
        parameters=auxiliary_parameters,
        names=auxiliary_names,
        parameter_count=sum(value.numel() for value in auxiliary_parameters),
    )
    monkeypatch.setattr(
        module,
        "state_residual_survival_parameter_subset_v23",
        lambda _: auxiliary_subset,
    )

    prefix = torch.tensor(
        (
            (0, 2, 4, 6, 8, 10, 7, 12, 14),
            (14, 12, 10, 8, 6, 4, 7, 2, 0),
            (2, 4, 6, 8, 10, 12, 7, 14, 0),
            (12, 10, 8, 6, 4, 2, 7, 0, 14),
        ),
        dtype=torch.int64,
    )
    microbatches = []
    for batch in v21_fixture._microbatches():
        value = dict(batch)
        value[module.PREFIX_LENGTHS_KEY] = prefix.clone()
        value[module.ACTION_PRIOR_M_KEY_V23] = torch.full(
            (9,), 0.55, dtype=torch.float32
        )
        microbatches.append(value)

    optimizer = v21_fixture._CountingSgd(list(partition.online))
    original_grad = torch.autograd.grad
    grad_parameter_ids: list[tuple[int, ...]] = []

    def counted_grad(*args: Any, **kwargs: Any) -> Any:
        grad_parameter_ids.append(tuple(id(value) for value in args[1]))
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", counted_grad)
    result = module.joint_training_update_v23(
        model, optimizer, tuple(microbatches)
    )

    assert len(grad_parameter_ids) == 12
    assert [len(value) for value in grad_parameter_ids] == [1, 17, 16] * 4
    assert grad_parameter_ids[2::3] == [
        tuple(id(value) for value in auxiliary_parameters)
    ] * 4
    assert model.predictor_forward_count == 4
    assert optimizer.step_calls == 1
    assert int(model.ema_update_count.item()) == 1
    assert result.target_gradient_tensor_count == 0
    assert model.target.grad is None
    assert result.accounting == module.JointTrainingAccountingV23(
        updates=1,
        presentations=16,
        microbatch_graphs=4,
        backward_calls=12,
        camera_route_grad_calls=4,
        joint_route_grad_calls=4,
        state_residual_survival_grad_calls=4,
        camera_frame_objectives=32,
        optimizer_steps=1,
        ema_steps=1,
        predictor_forwards=4,
        predictor_objectives=8,
        state_residual_survival_objectives=4,
    )
    route = result.gradient_routes[module.STATE_RESIDUAL_SURVIVAL_ROUTE_NAME_V23]
    assert route.parameter_tensor_count == 16
    assert route.absent_tensor_gradient_count == 0
    assert route.preclip_l2 > 0.0
    assert route.applied_scale == pytest.approx(min(1.0, 1.0 / route.preclip_l2))
    assert result.mean_losses["J23"] == pytest.approx(
        result.mean_losses["F"] + result.mean_losses["J_rank"], rel=2.0e-6
    )
    assert result.mean_losses["L"] == pytest.approx(
        result.mean_losses["N"]
        + result.mean_losses["C"]
        + result.mean_losses["J23"],
        rel=2.0e-6,
    )
    diagnostics = result.state_residual_survival_diagnostics
    assert diagnostics["positive_energy_count"] == 128
    negative_rows = microbatches[0][module.SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21]
    non_hold = torch.tensor(module.NON_HOLD_ACTION_INDICES_V23)
    expected_scene_per_microbatch = int(
        (
            prefix.index_select(1, non_hold)
            != prefix.index_select(0, negative_rows).index_select(1, non_hold)
        ).sum()
    )
    assert diagnostics["scene_eligible_count"] == 4 * expected_scene_per_microbatch
    assert diagnostics["prior_eligible_count"] == 128
    assert math.isfinite(float(diagnostics["scene_advantage_mean"]))
    assert math.isfinite(float(diagnostics["prior_advantage_mean"]))
    assert math.isfinite(float(diagnostics["scene_rank_sum"]))
    assert math.isfinite(float(diagnostics["prior_rank_sum"]))
