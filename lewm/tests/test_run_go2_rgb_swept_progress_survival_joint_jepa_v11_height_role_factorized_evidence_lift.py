from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
import torch

from scripts import (
    run_go2_rgb_swept_progress_survival_joint_jepa_v11_height_role_factorized_evidence_lift
    as runner,
)


class _HeightRoleLift(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.floor_query_projection = torch.nn.Linear(64, 32, bias=True)
        self.floor_key_projection = torch.nn.Linear(64, 32, bias=False)
        self.floor_value_projection = torch.nn.Linear(64, 32, bias=True)
        self.floor_output_projection = torch.nn.Linear(32, 32, bias=True)
        self.elevated_query_projection = torch.nn.Linear(64, 32, bias=True)
        self.elevated_key_projection = torch.nn.Linear(64, 32, bias=False)
        self.elevated_value_projection = torch.nn.Linear(64, 32, bias=True)
        self.elevated_output_projection = torch.nn.Linear(32, 32, bias=True)


class _EvidenceAxis(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Conv2d(32, 1, kernel_size=1, bias=True)
        self.local = torch.nn.Conv2d(
            32,
            32,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.residual_output = torch.nn.Conv2d(
            32,
            1,
            kernel_size=1,
            bias=True,
        )


class _FactorizedSemanticHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.free_axis = _EvidenceAxis()
        self.occupied_axis = _EvidenceAxis()


class _MockV11(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bev_lift = _HeightRoleLift()
        self.target_bev_lift = copy.deepcopy(self.bev_lift)
        self.target_bev_lift.requires_grad_(False)
        self.semantic_head = _FactorizedSemanticHead()
        self.register_buffer("ema_update_count", torch.tensor(1, dtype=torch.int64))


def _model_with_online_gradients() -> _MockV11:
    model = _MockV11()
    for name, parameter in model.named_parameters():
        if not name.startswith(runner.TARGET_LIFT_PREFIX_V11):
            parameter.grad = torch.ones_like(parameter)
    return model


def _receipt(
    group: str,
    update: int,
    *,
    inactive: tuple[str, ...] = (),
) -> dict[str, Any]:
    if group == "attention":
        suffixes = runner.BRANCH_ATTENTION_PARAMETER_SUFFIXES_V11
        schema = runner._ATTENTION_RECEIPT_SCHEMA_V11
        count = runner.HEIGHT_ROLE_BRANCH_ATTENTION_PARAMETER_COUNT_V11
        target_count = runner.HEIGHT_ROLE_BRANCH_ATTENTION_PARAMETER_TENSOR_COUNT_V11
    elif group == "semantic":
        suffixes = runner.SEMANTIC_AXIS_PARAMETER_SUFFIXES_V11
        schema = runner._SEMANTIC_RECEIPT_SCHEMA_V11
        count = runner.FACTORIZED_SEMANTIC_AXIS_PARAMETER_COUNT_V11
        target_count = 0
    else:
        raise AssertionError(group)
    return {
        "schema": schema,
        "update": update,
        "measurement": "post_clip_post_optimizer_step_retained_gradient",
        "gradient_l2": 1.0,
        "minimum_parameter_gradient_l2": 0.0 if inactive else 1.0,
        "maximum_parameter_gradient_l2": 1.0,
        "online_parameter_count": count,
        "online_parameter_tensor_count": len(suffixes),
        "gradient_tensor_count": len(suffixes),
        "active_parameter_tensor_count": len(suffixes) - len(inactive),
        "inactive_parameter_suffixes": list(inactive),
        "parameter_suffix_inventory_sha256": runner._inventory_sha256_v11(
            suffixes
        ),
        "target_parameter_tensor_count": target_count,
        "target_gradient_tensor_count": 0,
    }


def _terminal_accounting() -> runner.JointTrainingAccountingV1:
    return runner.JointTrainingAccountingV1(
        updates=1_000,
        presentations=16_000,
        microbatch_graphs=4_000,
        backward_calls=4_000,
        optimizer_steps=1_000,
        ema_steps=1_000,
        predictor_forwards=4_000,
        predictor_objectives=4_000,
    )


def _inherited_trace(losses: dict[str, float]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "update": update,
            "presentations": update * 16,
            "losses": dict(losses),
            "gradient_l2": {
                "encoder": 1.0,
                "lift_semantic": 2.0,
                "predictor": 3.0,
            },
        }
        for update in range(1, 1_001)
    )


def test_v11_wrapper_inherits_exact_v3_science_schedule_and_cap() -> None:
    for name in (
        "ACTION_ORDER",
        "MICROBATCH_SIZE",
        "MICROBATCHES_PER_UPDATE",
        "PRESENTATIONS_PER_UPDATE",
        "MAXIMUM_UPDATES",
        "MAXIMUM_PRESENTATIONS",
        "OCCUPIED_CLASS_INDEX",
        "OCCUPIED_SAFETY_AUX_COEFFICIENT",
        "OCCUPIED_SAFETY_AUX_NORMALIZATION",
    ):
        assert getattr(runner, name) == getattr(runner._v3, name)
    assert runner.OCCUPIED_SAFETY_AUX_COEFFICIENT == 0.5
    assert runner.MAXIMUM_UPDATES == 1_000
    assert runner.MAXIMUM_PRESENTATIONS == 16_000
    assert runner.HEIGHT_ROLE_BRANCH_ATTENTION_PARAMETER_COUNT_V11 == 14_528
    assert runner.HEIGHT_ROLE_BRANCH_ATTENTION_PARAMETER_TENSOR_COUNT_V11 == 14
    assert runner.FACTORIZED_SEMANTIC_AXIS_PARAMETER_COUNT_V11 == 18_628
    assert runner.FACTORIZED_SEMANTIC_AXIS_PARAMETER_TENSOR_COUNT_V11 == 12


def test_exact_registered_parameter_name_contract() -> None:
    assert runner.BRANCH_ATTENTION_PARAMETER_SUFFIXES_V11 == tuple(
        runner._model_v11.HEIGHT_ROLE_ATTENTION_PARAMETER_SUFFIXES_V11
    )
    assert runner.SEMANTIC_AXIS_PARAMETER_SUFFIXES_V11 == tuple(
        runner._model_v11.HEIGHT_ROLE_SEMANTIC_PARAMETER_SUFFIXES_V11
    )
    assert runner.BRANCH_ATTENTION_PARAMETER_SUFFIXES_V11 == (
        "floor_query_projection.weight",
        "floor_query_projection.bias",
        "floor_key_projection.weight",
        "floor_value_projection.weight",
        "floor_value_projection.bias",
        "floor_output_projection.weight",
        "floor_output_projection.bias",
        "elevated_query_projection.weight",
        "elevated_query_projection.bias",
        "elevated_key_projection.weight",
        "elevated_value_projection.weight",
        "elevated_value_projection.bias",
        "elevated_output_projection.weight",
        "elevated_output_projection.bias",
    )
    assert runner.SEMANTIC_AXIS_PARAMETER_SUFFIXES_V11 == (
        "free_axis.base.weight",
        "free_axis.base.bias",
        "free_axis.local.weight",
        "free_axis.local.bias",
        "free_axis.residual_output.weight",
        "free_axis.residual_output.bias",
        "occupied_axis.base.weight",
        "occupied_axis.base.bias",
        "occupied_axis.local.weight",
        "occupied_axis.local.bias",
        "occupied_axis.residual_output.weight",
        "occupied_axis.residual_output.bias",
    )


def test_public_inventory_validates_order_shapes_counts_and_target_freeze() -> None:
    online, target, semantic = runner.v11_parameter_inventories(_MockV11())
    assert tuple(name for name, _ in online) == (
        runner.BRANCH_ATTENTION_PARAMETER_SUFFIXES_V11
    )
    assert tuple(name for name, _ in target) == (
        runner.BRANCH_ATTENTION_PARAMETER_SUFFIXES_V11
    )
    assert tuple(name for name, _ in semantic) == (
        runner.SEMANTIC_AXIS_PARAMETER_SUFFIXES_V11
    )
    assert sum(parameter.numel() for _, parameter in online) == 14_528
    assert sum(parameter.numel() for _, parameter in target) == 14_528
    assert sum(parameter.numel() for _, parameter in semantic) == 18_628
    assert all(parameter.requires_grad for _, parameter in online)
    assert all(not parameter.requires_grad for _, parameter in target)
    assert all(parameter.requires_grad for _, parameter in semantic)


def test_inventory_rejects_extra_semantic_tensor() -> None:
    model = _MockV11()
    model.semantic_head.extra = torch.nn.Parameter(torch.zeros(()))
    with pytest.raises(RuntimeError, match="semantic-axis inventory"):
        runner.v11_parameter_inventories(model)


def test_inventory_rejects_trainable_target_attention() -> None:
    model = _MockV11()
    next(model.target_bev_lift.parameters()).requires_grad_(True)
    with pytest.raises(RuntimeError, match="target height-role attention.*trainable"):
        runner.v11_parameter_inventories(model)


def test_gradient_receipts_cover_all_online_tensors_and_frozen_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model_with_online_gradients()
    monkeypatch.setattr(
        runner._v3._v2._v1,
        "_runtime_apis",
        lambda: (torch, None, None),
    )
    attention = runner.branch_attention_gradient_receipt_v11(model)
    semantic = runner.semantic_axis_gradient_receipt_v11(model)
    assert attention["schema"] == runner._ATTENTION_RECEIPT_SCHEMA_V11
    assert attention["online_parameter_count"] == 14_528
    assert attention["online_parameter_tensor_count"] == 14
    assert attention["active_parameter_tensor_count"] == 14
    assert attention["target_parameter_tensor_count"] == 14
    assert attention["target_gradient_tensor_count"] == 0
    assert semantic["schema"] == runner._SEMANTIC_RECEIPT_SCHEMA_V11
    assert semantic["online_parameter_count"] == 18_628
    assert semantic["online_parameter_tensor_count"] == 12
    assert semantic["active_parameter_tensor_count"] == 12
    assert semantic["target_parameter_tensor_count"] == 0


def test_gradient_receipt_rejects_target_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model_with_online_gradients()
    target = next(model.target_bev_lift.parameters())
    target.grad = torch.ones_like(target)
    monkeypatch.setattr(
        runner._v3._v2._v1,
        "_runtime_apis",
        lambda: (torch, None, None),
    )
    with pytest.raises(RuntimeError, match="target height-role attention"):
        runner.branch_attention_gradient_receipt_v11(model)


def test_gradient_receipt_rejects_missing_semantic_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model_with_online_gradients()
    model.semantic_head.free_axis.local.weight.grad = None
    monkeypatch.setattr(
        runner._v3._v2._v1,
        "_runtime_apis",
        lambda: (torch, None, None),
    )
    with pytest.raises(FloatingPointError, match="absent or nonfinite"):
        runner.semantic_axis_gradient_receipt_v11(model)


@pytest.mark.parametrize(
    ("group", "suffixes", "summarize", "expected_schema"),
    (
        (
            "attention",
            runner.BRANCH_ATTENTION_PARAMETER_SUFFIXES_V11,
            runner._summarize_branch_attention_activity_v11,
            runner._ATTENTION_ACTIVITY_SCHEMA_V11,
        ),
        (
            "semantic",
            runner.SEMANTIC_AXIS_PARAMETER_SUFFIXES_V11,
            runner._summarize_semantic_axis_activity_v11,
            runner._SEMANTIC_ACTIVITY_SCHEMA_V11,
        ),
    ),
)
def test_activity_requires_every_new_tensor_by_update_two(
    group: str,
    suffixes: tuple[str, ...],
    summarize: Callable[[Any, Any], dict[str, Any]],
    expected_schema: str,
) -> None:
    receipts = [_receipt(group, 1, inactive=(suffixes[-1],))]
    receipts.extend(_receipt(group, update) for update in range(2, 1_001))
    observed = summarize(receipts, suffixes)
    assert observed["schema"] == expected_schema
    assert observed["update_count"] == 1_000
    assert observed["all_online_parameter_tensors_active_by_update_2"] is True
    assert observed["latest_first_active_update"] == 2
    assert observed["first_active_update"][suffixes[-1]] == 2
    assert observed["target_gradient_tensor_count"] == 0


@pytest.mark.parametrize(
    ("group", "suffixes", "summarize"),
    (
        (
            "attention",
            runner.BRANCH_ATTENTION_PARAMETER_SUFFIXES_V11,
            runner._summarize_branch_attention_activity_v11,
        ),
        (
            "semantic",
            runner.SEMANTIC_AXIS_PARAMETER_SUFFIXES_V11,
            runner._summarize_semantic_axis_activity_v11,
        ),
    ),
)
def test_activity_fails_if_any_tensor_activates_after_update_two(
    group: str,
    suffixes: tuple[str, ...],
    summarize: Callable[[Any, Any], dict[str, Any]],
) -> None:
    suffix = suffixes[-1]
    receipts = [
        _receipt(group, update, inactive=(suffix,) if update < 3 else ())
        for update in range(1, 1_001)
    ]
    with pytest.raises(RuntimeError, match="not active by update 2"):
        summarize(receipts, suffixes)


def test_driver_delegates_exact_v3_update_and_preserves_trace_and_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    losses = {
        name: float(index)
        for index, name in enumerate(("S", "P", "U", "R", "O", "L"), start=1)
    }
    inherited_result = SimpleNamespace(
        accounting=runner.JointTrainingAccountingV1(),
        mean_losses=losses,
        gradient_l2={
            "encoder": 1.0,
            "lift_semantic": 2.0,
            "predictor": 3.0,
        },
        ranking_active_microbatches=0,
        ranking_eligible_pairs=0,
        survival_supervised_decisions=0,
    )

    def fake_joint(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        del args, kwargs
        calls += 1
        return inherited_result

    def fake_core(*args: Any, joint_update: Any, **kwargs: Any) -> Any:
        del args, kwargs
        for _ in range(1_000):
            joint_update(
                None,
                None,
                (),
                accounting=runner.JointTrainingAccountingV1(),
            )
        return _terminal_accounting(), _inherited_trace(losses), {"inherited": True}

    attention_inventory = tuple(
        (suffix, object())
        for suffix in runner.BRANCH_ATTENTION_PARAMETER_SUFFIXES_V11
    )
    semantic_inventory = tuple(
        (suffix, object()) for suffix in runner.SEMANTIC_AXIS_PARAMETER_SUFFIXES_V11
    )
    monkeypatch.setattr(
        runner,
        "v11_parameter_inventories",
        lambda model: (attention_inventory, (), semantic_inventory),
    )
    monkeypatch.setattr(runner._v3, "joint_training_update_v3", fake_joint)
    monkeypatch.setattr(
        runner._v3._v2,
        "_run_fixed_training_core_v2",
        fake_core,
    )
    monkeypatch.setattr(
        runner,
        "branch_attention_gradient_receipt_v11",
        lambda model: _receipt("attention", calls),
    )
    monkeypatch.setattr(
        runner,
        "semantic_axis_gradient_receipt_v11",
        lambda model: _receipt("semantic", calls),
    )

    accounting, trace, diagnostics = runner.run_fixed_training_v11(
        None,
        None,
        None,
        (),
        None,
        (),
        None,
    )
    assert calls == 1_000
    assert accounting == _terminal_accounting()
    assert trace[0]["losses"] == losses
    assert trace[-1]["losses"] == losses
    assert tuple(trace[0]["losses"]) == ("S", "P", "U", "R", "O", "L")
    assert trace[0]["height_role_branch_attention"]["update"] == 1
    assert trace[-1]["height_role_branch_attention"]["update"] == 1_000
    assert trace[0]["factorized_semantic_axes"]["update"] == 1
    assert trace[-1]["factorized_semantic_axes"]["update"] == 1_000
    assert diagnostics["inherited"] is True
    assert diagnostics["height_role_branch_attention"][
        "latest_first_active_update"
    ] == 1
    assert diagnostics["factorized_semantic_axes"][
        "latest_first_active_update"
    ] == 1


def test_terminal_accounting_and_inherited_loss_schema_are_fail_closed() -> None:
    losses = {name: 1.0 for name in ("S", "P", "U", "R", "O", "L")}
    trace = list(_inherited_trace(losses))
    runner._validate_inherited_training_result_v11(_terminal_accounting(), trace)

    bad_accounting = copy.copy(_terminal_accounting())
    object.__setattr__(bad_accounting, "updates", 999)
    with pytest.raises((RuntimeError, ValueError)):
        runner._validate_inherited_training_result_v11(bad_accounting, trace)

    bad_trace = list(trace)
    bad_trace[0] = {**bad_trace[0], "losses": {"S": 1.0, "L": 1.0}}
    with pytest.raises(RuntimeError, match=r"S\+P\+U\+R\+O"):
        runner._validate_inherited_training_result_v11(
            _terminal_accounting(),
            bad_trace,
        )


def test_driver_rejects_any_non_preregistered_training_cap() -> None:
    with pytest.raises(RuntimeError, match="1,000-update/16,000-presentation"):
        runner.run_fixed_training_v11(
            None,
            None,
            None,
            (),
            None,
            (),
            None,
            maximum_updates=999,
        )


def test_source_delegates_science_without_reimplemented_loss_or_backward() -> None:
    source = Path(runner.__file__).read_text()
    assert source.count("_v3.joint_training_update_v3(") == 1
    assert source.count("_v3._v2._run_fixed_training_core_v2(") == 1
    assert ".backward(" not in source
    assert "semantic_loss_v1(" not in source
    assert "binary_cross_entropy" not in source
    assert "cross_entropy" not in source
