from __future__ import annotations

import ast
from dataclasses import replace
import importlib.util
import inspect
import math
from pathlib import Path
import sys
from typing import Any

import pytest
import torch

from lewm.tests import (
    test_run_go2_rgb_swept_progress_survival_joint_jepa_v2_occupied_safety_aux
    as v2_fixtures,
)


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    ROOT
    / "scripts"
    / "run_go2_rgb_swept_progress_survival_joint_jepa_v4_matched_no_persistence.py"
)


def _load_runner() -> Any:
    name = "_test_go2_swept_progress_survival_v4_matched_no_persistence_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


class _CountingTinyJointModel(v2_fixtures._TinyJointModel):
    def __init__(self) -> None:
        super().__init__()
        self.forward_counts = {
            "encode_online": 0,
            "semantic_logits": 0,
            "predict_all_actions_with_survival": 0,
            "encode_target": 0,
            "target_ema": 0,
        }

    def encode_online(self, rgb: torch.Tensor) -> torch.Tensor:
        self.forward_counts["encode_online"] += 1
        return super().encode_online(rgb)

    def semantic_logits_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        self.forward_counts["semantic_logits"] += 1
        return super().semantic_logits_from_latent(latent)

    def predict_all_actions_with_survival(self, current: torch.Tensor) -> Any:
        self.forward_counts["predict_all_actions_with_survival"] += 1
        return super().predict_all_actions_with_survival(current)

    def encode_target(self, rgb: torch.Tensor) -> torch.Tensor:
        self.forward_counts["encode_target"] += 1
        return super().encode_target(rgb)

    @torch.no_grad()
    def update_target_ema_after_optimizer_step(self) -> None:
        self.forward_counts["target_ema"] += 1
        super().update_target_ema_after_optimizer_step()


def _matched_models() -> tuple[Any, Any]:
    torch.manual_seed(71)
    full = v2_fixtures._TinyJointModel()
    control = _CountingTinyJointModel()
    control.load_state_dict(full.state_dict())
    return full, control


def _optimizer(model: Any) -> Any:
    return runner.build_frozen_optimizer_v1(runner.partition_parameters_v1(model))


def test_no_p_update_matches_v4_full_forward_components_and_accounting() -> None:
    full, control = _matched_models()
    full_result = runner._v3.joint_training_update_v3(
        full, _optimizer(full), v2_fixtures._microbatches()
    )
    control_optimizer = _optimizer(control)
    control_result = runner.joint_training_update_v4_matched_no_persistence(
        control, control_optimizer, v2_fixtures._microbatches()
    )

    assert tuple(control_result.mean_losses) == runner.TRACE_LOSS_KEYS == (
        "S",
        "P_diagnostic",
        "U",
        "R",
        "O",
        "L_full_diagnostic",
        "L_backward",
    )
    for full_name, control_name in (
        ("S", "S"),
        ("P", "P_diagnostic"),
        ("U", "U"),
        ("R", "R"),
        ("O", "O"),
        ("L", "L_full_diagnostic"),
    ):
        assert control_result.mean_losses[control_name] == full_result.mean_losses[
            full_name
        ]
    assert math.isclose(
        control_result.mean_losses["L_backward"],
        sum(control_result.mean_losses[name] for name in ("S", "U", "R", "O")),
        rel_tol=2e-6,
        abs_tol=2e-6,
    )

    expected_accounting = runner.JointTrainingAccountingV1(
        updates=1,
        presentations=16,
        microbatch_graphs=4,
        backward_calls=4,
        optimizer_steps=1,
        ema_steps=1,
        predictor_forwards=4,
        predictor_objectives=4,
    )
    assert control_result.accounting == full_result.accounting == expected_accounting
    assert control.forward_counts == {
        "encode_online": 8,
        "semantic_logits": 8,
        "predict_all_actions_with_survival": 4,
        "encode_target": 8,
        "target_ema": 1,
    }
    assert tuple(control_result.gradient_l2) == runner.GRADIENT_GROUPS
    assert all(value > 0.0 for value in control_result.gradient_l2.values())
    assert control_result.gradient_l2["predictor"] > 0.0
    partition = runner.partition_parameters_v1(control)
    assert all(parameter.grad is None for parameter in partition.target)
    assert [group["name"] for group in control_optimizer.param_groups] == list(
        runner.GRADIENT_GROUPS
    )
    assert int(control.ema_update_count) == 1


def test_direct_backward_scalar_has_only_s_u_r_o_members() -> None:
    source = inspect.getsource(
        runner.joint_training_update_v4_matched_no_persistence
    )
    function = ast.parse(source).body[0]
    assignments = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "backward_loss"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1

    def flattened_add(node: ast.AST) -> list[str]:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return flattened_add(node.left) + flattened_add(node.right)
        return [ast.unparse(node)]

    assert flattened_add(assignments[0].value) == [
        "joint.semantic",
        "joint.survival",
        "joint.progress_ranking",
        "occupied.loss",
    ]


def test_differentiable_p_perturbation_changes_only_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, perturbed = _matched_models()
    baseline_result = runner.joint_training_update_v4_matched_no_persistence(
        baseline, _optimizer(baseline), v2_fixtures._microbatches()
    )

    _, semantic_api, _ = runner._v3._v2._v1._runtime_apis()
    original_persistence = semantic_api.microbatch_persistence_loss_v1

    def changed_persistence(
        predicted_latents: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        terms = original_persistence(predicted_latents, *args, **kwargs)
        differentiable_delta = 0.125 * predicted_latents.square().mean()
        return replace(terms, loss=terms.loss + differentiable_delta)

    monkeypatch.setattr(
        semantic_api, "microbatch_persistence_loss_v1", changed_persistence
    )
    perturbed_result = runner.joint_training_update_v4_matched_no_persistence(
        perturbed, _optimizer(perturbed), v2_fixtures._microbatches()
    )

    assert (
        perturbed_result.mean_losses["P_diagnostic"]
        != baseline_result.mean_losses["P_diagnostic"]
    )
    assert (
        perturbed_result.mean_losses["L_full_diagnostic"]
        != baseline_result.mean_losses["L_full_diagnostic"]
    )
    for name in ("S", "U", "R", "O", "L_backward"):
        assert perturbed_result.mean_losses[name] == baseline_result.mean_losses[name]
    assert perturbed_result.gradient_l2 == baseline_result.gradient_l2
    for (_, baseline_parameter), (_, perturbed_parameter) in zip(
        baseline.named_parameters(), perturbed.named_parameters(), strict=True
    ):
        assert torch.equal(baseline_parameter, perturbed_parameter)
        if baseline_parameter.requires_grad:
            assert baseline_parameter.grad is not None
            assert perturbed_parameter.grad is not None
            assert torch.equal(baseline_parameter.grad, perturbed_parameter.grad)
        else:
            assert baseline_parameter.grad is None
            assert perturbed_parameter.grad is None
    baseline_state = baseline.state_dict()
    perturbed_state = perturbed.state_dict()
    assert baseline_state.keys() == perturbed_state.keys()
    assert all(
        torch.equal(baseline_state[name], perturbed_state[name])
        for name in baseline_state
    )


def test_first_update_witness_mismatch_blocks_before_step_and_ema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _CountingTinyJointModel()
    optimizer = _optimizer(model)
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    step_calls = 0
    original_step = optimizer.step

    def counted_step(*args: Any, **kwargs: Any) -> Any:
        nonlocal step_calls
        step_calls += 1
        return original_step(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", counted_step)
    backward_calls = 0

    def count_backward(gradient: torch.Tensor) -> torch.Tensor:
        nonlocal backward_calls
        backward_calls += 1
        return gradient

    hook = model.predictor.action.register_hook(count_backward)
    wrong = dict(runner.FIRST_UPDATE_COMPONENT_MEANS)
    wrong["S"] += 1.0
    try:
        with pytest.raises(
            runner.FirstUpdateComponentWitnessMismatchV1,
            match="component witness mismatch",
        ) as caught:
            runner.joint_training_update_v4_matched_no_persistence(
                model,
                optimizer,
                v2_fixtures._microbatches(),
                expected_component_means=wrong,
            )
    finally:
        hook.remove()

    error = caught.value
    assert dict(error.expected) == wrong
    assert tuple(error.observed) == runner.COMPONENT_KEYS
    expected_mismatch = {
        name
        for name in runner.COMPONENT_KEYS
        if error.expected[name] != error.observed[name]
    }
    assert set(error.mismatch) == expected_mismatch
    for name in error.mismatch:
        assert dict(error.mismatch[name]) == {
            "expected": error.expected[name],
            "observed": error.observed[name],
        }
    assert dict(error.pre_step_operation_counts) == {
        "presentations_consumed": 16,
        "microbatch_graphs_completed": 4,
        "backward_calls_completed": 4,
        "optimizer_steps_completed": 0,
        "ema_steps_completed": 0,
        "predictor_forwards_completed": 4,
        "predictor_objectives_evaluated": 4,
    }
    assert backward_calls == runner.MICROBATCHES_PER_UPDATE == 4
    assert step_calls == 0
    assert int(model.ema_update_count) == 0
    assert len(optimizer.state) == 0
    after = model.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert all(
        parameter.grad is None
        for parameter in runner.partition_parameters_v1(model).target
    )


def test_clip_step_and_ema_order_remains_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, model = _matched_models()
    optimizer = _optimizer(model)
    partition = runner.partition_parameters_v1(model)
    representation_ids = tuple(map(id, partition.representation))
    predictor_ids = tuple(map(id, partition.predictor))
    events: list[str] = []
    original_clip = torch.nn.utils.clip_grad_norm_
    original_step = optimizer.step
    original_ema = model.update_target_ema_after_optimizer_step

    def observed_clip(
        parameters: Any,
        max_norm: float,
        error_if_nonfinite: bool,
    ) -> torch.Tensor:
        values = tuple(parameters)
        ids = tuple(map(id, values))
        if ids == representation_ids:
            events.append("clip_representation")
        elif ids == predictor_ids:
            events.append("clip_predictor")
        else:
            raise AssertionError("unexpected clipping parameter group")
        assert max_norm == 1.0
        assert error_if_nonfinite is True
        return original_clip(
            values, max_norm=max_norm, error_if_nonfinite=error_if_nonfinite
        )

    def observed_step(*args: Any, **kwargs: Any) -> Any:
        events.append("optimizer_step")
        return original_step(*args, **kwargs)

    @torch.no_grad()
    def observed_ema() -> None:
        events.append("target_ema")
        original_ema()

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", observed_clip)
    monkeypatch.setattr(optimizer, "step", observed_step)
    monkeypatch.setattr(model, "update_target_ema_after_optimizer_step", observed_ema)
    runner.joint_training_update_v4_matched_no_persistence(
        model, optimizer, v2_fixtures._microbatches()
    )

    assert events == [
        "clip_representation",
        "clip_predictor",
        "optimizer_step",
        "target_ema",
    ]


def test_fixed_driver_enforces_witness_cap_trace_and_gradient_diagnostics() -> None:
    labels = runner.freeze_role_labels_v1(
        v2_fixtures._label_rows(), role="train", np=__import__("numpy")
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
    updates = 0
    witness_calls = 0

    def build(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal built
        del args, kwargs
        built += 1
        return {}

    def update(
        model: Any,
        optimizer: Any,
        microbatches: Any,
        *,
        accounting: runner.JointTrainingAccountingV1,
        expected_component_means: Any,
    ) -> runner.JointUpdateResultV4MatchedNoPersistence:
        nonlocal updates, witness_calls
        del model, optimizer
        assert len(microbatches) == 4
        updates += 1
        if expected_component_means is not None:
            witness_calls += 1
            components = dict(expected_component_means)
        else:
            components = {
                "S": 1.0,
                "P_diagnostic": 2.0,
                "U": 3.0,
                "R": 4.0,
                "O": 0.5,
            }
        losses = {
            **components,
            "L_full_diagnostic": sum(components.values()),
            "L_backward": sum(components[name] for name in ("S", "U", "R", "O")),
        }
        return runner.JointUpdateResultV4MatchedNoPersistence(
            accounting=runner._v3._v2._v1._base._advanced_accounting(accounting),
            mean_losses=losses,
            gradient_l2={name: 1.0 for name in runner.GRADIENT_GROUPS},
            representation_clip_pre_l2=1.0,
            predictor_clip_pre_l2=1.0,
            ranking_active_microbatches=2,
            ranking_eligible_pairs=3,
            survival_supervised_decisions=4,
            first_update_component_witness_checked=(
                expected_component_means is not None
            ),
        )

    accounting, trace, diagnostics = (
        runner._run_fixed_training_core_v4_matched_no_persistence(
            object(),
            object(),
            object(),
            (pair,),
            labels,
            (0,) * runner.MAXIMUM_PRESENTATIONS,
            object(),
            microbatch_builder=build,
            joint_update=update,
        )
    )
    assert accounting == runner.JointTrainingAccountingV1(
        updates=1_000,
        presentations=16_000,
        microbatch_graphs=4_000,
        backward_calls=4_000,
        optimizer_steps=1_000,
        ema_steps=1_000,
        predictor_forwards=4_000,
        predictor_objectives=4_000,
    )
    assert built == 4_000
    assert updates == len(trace) == 1_000
    assert witness_calls == 1
    assert tuple(trace[0]["losses"]) == runner.TRACE_LOSS_KEYS
    assert trace[0]["losses"] == {
        **dict(runner.FIRST_UPDATE_COMPONENT_MEANS),
        "L_full_diagnostic": sum(runner.FIRST_UPDATE_COMPONENT_MEANS.values()),
        "L_backward": sum(
            runner.FIRST_UPDATE_COMPONENT_MEANS[name]
            for name in ("S", "U", "R", "O")
        ),
    }
    assert diagnostics["gradient_groups"] == list(runner.GRADIENT_GROUPS)
    assert diagnostics["minimum_gradient_l2"] == {
        name: 1.0 for name in runner.GRADIENT_GROUPS
    }
    assert diagnostics["maximum_gradient_l2"] == {
        name: 1.0 for name in runner.GRADIENT_GROUPS
    }
    assert diagnostics["first_update_component_witness"] == {
        "expected": dict(runner.FIRST_UPDATE_COMPONENT_MEANS),
        "observed": dict(runner.FIRST_UPDATE_COMPONENT_MEANS),
        "exact_match": True,
        "checked_after_backward_calls": 4,
        "checked_before_optimizer_step": True,
    }

    with pytest.raises(PermissionError, match="cap"):
        runner.run_fixed_training_v4_matched_no_persistence(
            object(),
            object(),
            object(),
            (pair,),
            labels,
            (0,) * runner.MAXIMUM_PRESENTATIONS,
            object(),
            maximum_updates=999,
        )


def test_control_reuses_all_non_treatment_v3_identities() -> None:
    assert runner.ACTION_ORDER is runner._v3.ACTION_ORDER
    assert runner.REQUIRED_BATCH_KEYS is runner._v3.REQUIRED_BATCH_KEYS
    assert runner.build_microbatch_v1 is runner._v3.build_microbatch_v1
    assert runner.build_frozen_optimizer_v1 is runner._v3.build_frozen_optimizer_v1
    assert runner.partition_parameters_v1 is runner._v3.partition_parameters_v1
    assert runner.score_full_control_v1 is runner._v3.score_full_control_v1
    assert runner.MAXIMUM_UPDATES == runner._v3.MAXIMUM_UPDATES == 1_000
    assert runner.MAXIMUM_PRESENTATIONS == runner._v3.MAXIMUM_PRESENTATIONS == 16_000
    assert (
        runner.OCCUPIED_SAFETY_AUX_COEFFICIENT
        == runner._v3.OCCUPIED_SAFETY_AUX_COEFFICIENT
        == 0.5
    )
    assert dict(runner.FIRST_UPDATE_COMPONENT_MEANS) == {
        "S": 1.313827022910118,
        "P_diagnostic": 1.0,
        "U": 0.9792981296777725,
        "R": 1.0,
        "O": 1.026371382176876,
    }
    with pytest.raises(TypeError):
        runner.FIRST_UPDATE_COMPONENT_MEANS["S"] = 0.0
