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
    "scripts/run_go2_rgb_predictor_core_protected_survival_output_joint_jepa_v24.py"
)
V21_FIXTURE_PATH = ROOT / (
    "lewm/tests/test_run_go2_rgb_same_action_cross_scene_contrastive_"
    "innovation_joint_jepa_v21.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("_v24_training_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_v21_fixture():
    spec = importlib.util.spec_from_file_location(
        "_v21_training_fixture_for_v24", V21_FIXTURE_PATH
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


def _objective_inputs() -> tuple[torch.Tensor, ...]:
    logits = torch.zeros((4, 9, 16), dtype=torch.float32, requires_grad=True)
    prefix = torch.tensor(
        (
            (0, 2, 4, 6, 8, 10, 7, 12, 14),
            (14, 12, 10, 8, 6, 4, 7, 2, 0),
            (2, 4, 6, 8, 10, 12, 7, 14, 0),
            (12, 10, 8, 6, 4, 2, 7, 0, 14),
        ),
        dtype=torch.int64,
    )
    negatives = torch.tensor((1, 0, 3, 2), dtype=torch.int64)
    prior = torch.full((9,), 0.55, dtype=torch.float32)
    return logits, prefix, negatives, prior


def test_private_adapter_binds_v23_without_public_import() -> None:
    module = _load_module()
    receipt = module.private_training_adapter_receipt_v24()
    assert receipt["base_training_file_sha256"] == module.BASE_TRAINING_FILE_SHA256
    assert receipt["objective_bit_identical_to_v23"] is True
    assert receipt["j24_parameter_tensor_count"] == 96
    assert receipt["j24_parameter_count"] == 3_106_409
    assert receipt["protected_predictor_core_parameter_tensor_count"] == 13
    assert receipt["protected_predictor_core_parameter_count"] == 259_008
    assert receipt["inherited_joint_predictor_parameter_tensor_count"] == 15
    assert receipt["public_base_loaded_by_adapter"] is False
    assert receipt["private_module_registered"] is False


def test_v23_v24_objectives_are_bit_identical() -> None:
    module = _load_module()
    logits, prefix, negatives, prior = _objective_inputs()
    v23 = module._v23.state_residual_survival_objective_v23(
        torch, _DirectScores, logits, prefix, negatives, prior
    )
    v24 = module.predictor_core_protected_survival_objective_v24(
        torch, _DirectScores, logits, prefix, negatives, prior
    )
    for name in (
        "loss",
        "fit",
        "rank",
        "positive_energy",
        "scene_negative_energy",
        "prior_negative_energy",
        "scene_rank_sum",
        "prior_rank_sum",
        "scene_advantage_sum",
        "prior_advantage_sum",
    ):
        assert torch.equal(getattr(v23, name), getattr(v24, name))
    assert v23.scene_eligible_count == v24.scene_eligible_count
    assert v23.prior_eligible_count == v24.prior_eligible_count


def _parameters(counts: list[int]) -> tuple[torch.nn.Parameter, ...]:
    return tuple(torch.nn.Parameter(torch.zeros(count)) for count in counts)


def _exact_partition() -> SimpleNamespace:
    encoder = _parameters([1] * 79 + [3_102_730])
    evidence = _parameters([1] * 7 + [8])
    representation_live = _parameters([1] * 5 + [3_515])
    semantic = _parameters([1] * 5 + [73_981])
    representation = (*representation_live, *semantic)
    transition_sizes = [
        576, 73_728, 64, 36_864, 64, 36_864, 64,
        36_864, 64, 36_864, 64, 36_864, 64,
    ]
    predictor = (*_parameters(transition_sizes), *_parameters([64, 1]))
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
            "predictor.swept_progress_head.output.weight",
            "predictor.swept_progress_head.output.bias",
        ),
    }
    return SimpleNamespace(
        encoder=encoder,
        evidence_head=evidence,
        representation=representation,
        predictor=predictor,
        names=names,
    )


def test_exact_96_tensor_route_excludes_core_and_reaches_every_intended_group() -> None:
    module = _load_module()
    subset = module.predictor_core_protected_survival_parameter_subset_v24(
        _exact_partition()
    )
    assert len(subset.parameters) == 96
    assert subset.parameter_count == 3_106_409
    assert len(subset.protected_predictor_core_parameters) == 13
    assert subset.protected_predictor_core_parameter_count == 259_008
    assert subset.names[-2:] == module.SWEPT_PROGRESS_OUTPUT_PARAMETER_NAMES_V24
    assert not set(subset.names) & set(subset.protected_predictor_core_names)
    assert subset.protected_predictor_core_names == (
        module.PROTECTED_PREDICTOR_CORE_PARAMETER_NAMES_V24
    )

    signal = torch.stack(
        tuple(parameter.reshape(-1)[0] for parameter in subset.parameters)
    ).sum()
    logits = signal.expand(4, 9, 16)
    _, prefix, negatives, prior = _objective_inputs()
    objective = module.predictor_core_protected_survival_objective_v24(
        torch, _DirectScores, logits, prefix, negatives, prior
    )
    gradients = torch.autograd.grad(objective.loss, subset.parameters)
    assert len(gradients) == 96
    assert all(
        bool(torch.isfinite(gradient).all()) and float(gradient.abs().sum()) > 0.0
        for gradient in gradients
    )
    for prefix_name in (
        "encoder.",
        "bev_lift.evidence_head.",
        "bev_lift.point_projection.",
        "bev_lift.volume_block.",
        "predictor.swept_progress_head.output.",
    ):
        assert any(
            name.startswith(prefix_name) and float(gradient.abs().sum()) > 0.0
            for name, gradient in zip(subset.names, gradients, strict=True)
        )
    assert all(parameter.grad is None for parameter in subset.protected_predictor_core_parameters)


def test_accounting_preserves_exact_v23_work() -> None:
    module = _load_module()
    value = module.JointTrainingAccountingV24(
        updates=2,
        presentations=32,
        microbatch_graphs=8,
        backward_calls=24,
        camera_route_grad_calls=8,
        joint_route_grad_calls=8,
        predictor_core_protected_survival_grad_calls=8,
        camera_frame_objectives=64,
        optimizer_steps=2,
        ema_steps=2,
        predictor_forwards=8,
        predictor_objectives=16,
        predictor_core_protected_survival_objectives=8,
    )
    module.validate_accounting_v24(value)


def test_one_cpu_update_keeps_core_on_inherited_route_and_adds_j24_only_to_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    fixture = _load_v21_fixture()
    model = fixture._TinyModel()
    partition = fixture._partition(model)

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
    monkeypatch.setattr(module._v23, "_validate_microbatches_v23", lambda *_: None)
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

    auxiliary_parameters = (model.shared, *tuple(model.survival))
    subset = module.PredictorCoreProtectedSurvivalParameterSubsetV24(
        parameters=auxiliary_parameters,
        names=("encoder.synthetic", *fixture.SURVIVAL_NAMES),
        parameter_count=sum(value.numel() for value in auxiliary_parameters),
        protected_predictor_core_parameters=tuple(model.transition),
        protected_predictor_core_names=fixture.TRANSITION_NAMES,
        protected_predictor_core_parameter_count=sum(
            value.numel() for value in model.transition
        ),
    )
    monkeypatch.setattr(
        module,
        "predictor_core_protected_survival_parameter_subset_v24",
        lambda _: subset,
    )

    prefix = _objective_inputs()[1]
    microbatches = []
    for batch in fixture._microbatches():
        value = dict(batch)
        value[module.PREFIX_LENGTHS_KEY] = prefix.clone()
        value[module.ACTION_PRIOR_M_KEY_V23] = torch.full(
            (9,), 0.55, dtype=torch.float32
        )
        microbatches.append(value)

    captured_applied: dict[int, torch.Tensor] = {}

    class _CapturingSgd(fixture._CountingSgd):
        def step(self, closure: Any = None) -> Any:
            for parameter in partition.online:
                if parameter.grad is not None:
                    captured_applied[id(parameter)] = parameter.grad.detach().clone()
            return super().step(closure)

    optimizer = _CapturingSgd(list(partition.online))
    original_grad = torch.autograd.grad
    calls: list[tuple[tuple[int, ...], tuple[torch.Tensor | None, ...]]] = []

    def counted_grad(*args: Any, **kwargs: Any) -> Any:
        result = original_grad(*args, **kwargs)
        calls.append(
            (
                tuple(id(value) for value in args[1]),
                tuple(
                    None if value is None else value.detach().clone()
                    for value in result
                ),
            )
        )
        return result

    monkeypatch.setattr(torch.autograd, "grad", counted_grad)
    result = module.joint_training_update_v24(
        model, optimizer, tuple(microbatches)
    )

    assert len(calls) == 12
    assert [len(names) for names, _ in calls] == [1, 17, 3] * 4
    assert all(
        names == tuple(id(value) for value in auxiliary_parameters)
        for names, _ in calls[2::3]
    )
    predictor_scale = result.gradient_routes["predictor"].applied_scale
    auxiliary_scale = result.gradient_routes[
        module.PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V24
    ].applied_scale
    predictor_ids = tuple(id(value) for value in partition.predictor)
    inherited_sums = {
        parameter_id: torch.zeros_like(parameter)
        for parameter_id, parameter in zip(predictor_ids, partition.predictor, strict=True)
    }
    for names, gradients in calls[1::3]:
        for parameter_id, gradient in zip(names, gradients, strict=True):
            if parameter_id in inherited_sums:
                assert gradient is not None
                inherited_sums[parameter_id].add_(gradient)
    auxiliary_sums = {
        id(parameter): torch.zeros_like(parameter) for parameter in auxiliary_parameters
    }
    for names, gradients in calls[2::3]:
        for parameter_id, gradient in zip(names, gradients, strict=True):
            assert gradient is not None
            auxiliary_sums[parameter_id].add_(gradient)

    for parameter in model.transition:
        expected = predictor_scale * inherited_sums[id(parameter)]
        assert torch.equal(captured_applied[id(parameter)], expected)
        assert id(parameter) not in auxiliary_sums
    for parameter in model.survival:
        expected = (
            predictor_scale * inherited_sums[id(parameter)]
            + auxiliary_scale * auxiliary_sums[id(parameter)]
        )
        assert torch.equal(captured_applied[id(parameter)], expected)
        assert float(auxiliary_sums[id(parameter)].abs().sum()) > 0.0

    assert model.predictor_forward_count == 4
    assert optimizer.step_calls == 1
    assert int(model.ema_update_count.item()) == 1
    assert result.target_gradient_tensor_count == 0
    assert model.target.grad is None
    assert result.accounting == module.JointTrainingAccountingV24(
        updates=1,
        presentations=16,
        microbatch_graphs=4,
        backward_calls=12,
        camera_route_grad_calls=4,
        joint_route_grad_calls=4,
        predictor_core_protected_survival_grad_calls=4,
        camera_frame_objectives=32,
        optimizer_steps=1,
        ema_steps=1,
        predictor_forwards=4,
        predictor_objectives=8,
        predictor_core_protected_survival_objectives=4,
    )
    assert result.mean_losses["J24"] == pytest.approx(
        result.mean_losses["F"] + result.mean_losses["J_rank"], rel=2.0e-6
    )
    assert result.mean_losses["L"] == pytest.approx(
        result.mean_losses["N"]
        + result.mean_losses["C"]
        + result.mean_losses["J24"],
        rel=2.0e-6,
    )
