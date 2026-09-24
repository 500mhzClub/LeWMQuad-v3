#!/usr/bin/env python3
"""Lean V11 wrapper around the unchanged V3/V4 joint-training core.

V11 changes the V10 lift aggregation and semantic readout, but deliberately
retains the exact V3 ``S+P+U+R+O`` update with ``O=0.5``.  This module adds no
loss or backward implementation.  It delegates every update to V3 through the
fixed V2 driver and records post-clip retained gradients for every new online
height-role attention and semantic-axis tensor while proving that the matching
EMA-target attentions stay gradient-free.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
_inserted_root = str(ROOT) not in sys.path
if _inserted_root:
    sys.path.insert(0, str(ROOT))
try:
    from lewm.models import (
        geometry_anchored_swept_progress_survival_joint_jepa_v11_height_role_factorized_evidence_lift
        as _model_v11,
    )
    from scripts import (
        run_go2_rgb_swept_progress_survival_joint_jepa_v3_half_occupied_safety_aux
        as _v3,
    )
finally:
    if _inserted_root:
        sys.path.remove(str(ROOT))


# Exact inherited scientific and execution identities.
ACTION_ORDER = _v3.ACTION_ORDER
MICROBATCHES_PER_UPDATE = _v3.MICROBATCHES_PER_UPDATE
MICROBATCH_SIZE = _v3.MICROBATCH_SIZE
PRESENTATIONS_PER_UPDATE = _v3.PRESENTATIONS_PER_UPDATE
MAXIMUM_UPDATES = _v3.MAXIMUM_UPDATES
MAXIMUM_PRESENTATIONS = _v3.MAXIMUM_PRESENTATIONS
OCCUPIED_CLASS_INDEX = _v3.OCCUPIED_CLASS_INDEX
OCCUPIED_SAFETY_AUX_COEFFICIENT = _v3.OCCUPIED_SAFETY_AUX_COEFFICIENT
OCCUPIED_SAFETY_AUX_NORMALIZATION = _v3.OCCUPIED_SAFETY_AUX_NORMALIZATION

FrozenSurvivalRoleLabelsV1 = _v3.FrozenSurvivalRoleLabelsV1
JointTrainingAccountingV1 = _v3.JointTrainingAccountingV1
build_microbatch_v1 = _v3.build_microbatch_v1
partition_parameters_v1 = _v3.partition_parameters_v1
build_frozen_optimizer_v1 = _v3.build_frozen_optimizer_v1
freeze_role_labels_v1 = _v3.freeze_role_labels_v1
validate_pairs_against_labels_v1 = _v3.validate_pairs_against_labels_v1


HEIGHT_ROLE_BRANCH_ATTENTION_PARAMETER_COUNT_V11 = (
    _model_v11.HEIGHT_ROLE_ATTENTION_PARAMETER_COUNT_V11
)
HEIGHT_ROLE_BRANCH_ATTENTION_PARAMETER_TENSOR_COUNT_V11 = (
    _model_v11.HEIGHT_ROLE_ATTENTION_PARAMETER_TENSOR_COUNT_V11
)
FACTORIZED_SEMANTIC_AXIS_PARAMETER_COUNT_V11 = (
    _model_v11.HEIGHT_ROLE_SEMANTIC_PARAMETER_COUNT_V11
)
FACTORIZED_SEMANTIC_AXIS_PARAMETER_TENSOR_COUNT_V11 = (
    _model_v11.HEIGHT_ROLE_SEMANTIC_PARAMETER_TENSOR_COUNT_V11
)

ONLINE_LIFT_PREFIX_V11 = "bev_lift."
TARGET_LIFT_PREFIX_V11 = "target_bev_lift."
ONLINE_SEMANTIC_PREFIX_V11 = "semantic_head."
TARGET_SEMANTIC_PREFIX_V11 = "target_semantic_head."

BRANCH_ATTENTION_MODULE_STEMS_V11 = (
    "floor_query_projection",
    "floor_key_projection",
    "floor_value_projection",
    "floor_output_projection",
    "elevated_query_projection",
    "elevated_key_projection",
    "elevated_value_projection",
    "elevated_output_projection",
)
_FROZEN_BRANCH_ATTENTION_PARAMETER_SUFFIXES_V11 = (
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
_FROZEN_SEMANTIC_AXIS_PARAMETER_SUFFIXES_V11 = (
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
BRANCH_ATTENTION_PARAMETER_SUFFIXES_V11 = tuple(
    _model_v11.HEIGHT_ROLE_ATTENTION_PARAMETER_SUFFIXES_V11
)
SEMANTIC_AXIS_PARAMETER_SUFFIXES_V11 = tuple(
    _model_v11.HEIGHT_ROLE_SEMANTIC_PARAMETER_SUFFIXES_V11
)
if (
    HEIGHT_ROLE_BRANCH_ATTENTION_PARAMETER_COUNT_V11 != 14_528
    or HEIGHT_ROLE_BRANCH_ATTENTION_PARAMETER_TENSOR_COUNT_V11 != 14
    or FACTORIZED_SEMANTIC_AXIS_PARAMETER_COUNT_V11 != 18_628
    or FACTORIZED_SEMANTIC_AXIS_PARAMETER_TENSOR_COUNT_V11 != 12
    or BRANCH_ATTENTION_PARAMETER_SUFFIXES_V11
    != _FROZEN_BRANCH_ATTENTION_PARAMETER_SUFFIXES_V11
    or SEMANTIC_AXIS_PARAMETER_SUFFIXES_V11
    != _FROZEN_SEMANTIC_AXIS_PARAMETER_SUFFIXES_V11
):
    raise RuntimeError("V11 model/exported new-parameter contract changed")

HEIGHT_ROLE_BRANCH_ATTENTION_ONLINE_PARAMETER_NAMES_V11 = tuple(
    ONLINE_LIFT_PREFIX_V11 + suffix
    for suffix in BRANCH_ATTENTION_PARAMETER_SUFFIXES_V11
)
HEIGHT_ROLE_BRANCH_ATTENTION_TARGET_PARAMETER_NAMES_V11 = tuple(
    TARGET_LIFT_PREFIX_V11 + suffix
    for suffix in BRANCH_ATTENTION_PARAMETER_SUFFIXES_V11
)
FACTORIZED_SEMANTIC_AXIS_ONLINE_PARAMETER_NAMES_V11 = tuple(
    ONLINE_SEMANTIC_PREFIX_V11 + suffix
    for suffix in SEMANTIC_AXIS_PARAMETER_SUFFIXES_V11
)

_ATTENTION_EXPECTED_SHAPES_V11 = {
    suffix: (
        (32, 32)
        if suffix.endswith("output_projection.weight")
        else (32,)
        if suffix.endswith(".bias")
        else (32, 64)
    )
    for suffix in BRANCH_ATTENTION_PARAMETER_SUFFIXES_V11
}
_SEMANTIC_EXPECTED_SHAPES_V11 = {
    suffix: (
        (32, 32, 3, 3)
        if suffix.endswith("local.weight")
        else (32,)
        if suffix.endswith("local.bias")
        else (1, 32, 1, 1)
        if suffix.endswith(".weight")
        else (1,)
    )
    for suffix in SEMANTIC_AXIS_PARAMETER_SUFFIXES_V11
}

_ATTENTION_RECEIPT_SCHEMA_V11 = (
    "lewm_v11_height_role_branch_attention_post_backward_gradient_v1"
)
_ATTENTION_ACTIVITY_SCHEMA_V11 = (
    "lewm_v11_height_role_branch_attention_training_activity_v1"
)
_SEMANTIC_RECEIPT_SCHEMA_V11 = (
    "lewm_v11_factorized_semantic_axes_post_backward_gradient_v1"
)
_SEMANTIC_ACTIVITY_SCHEMA_V11 = (
    "lewm_v11_factorized_semantic_axes_training_activity_v1"
)


def _inventory_sha256_v11(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _is_branch_attention_suffix_v11(suffix: str) -> bool:
    return any(
        suffix.startswith(stem + ".")
        for stem in BRANCH_ATTENTION_MODULE_STEMS_V11
    )


def _validate_parameter_shapes_v11(
    inventory: Sequence[tuple[str, Any]],
    expected: Mapping[str, tuple[int, ...]],
    *,
    label: str,
) -> None:
    for suffix, parameter in inventory:
        if tuple(parameter.shape) != expected[suffix]:
            raise RuntimeError(
                f"V11 {label} shape changed for {suffix}: "
                f"{tuple(parameter.shape)} != {expected[suffix]}"
            )
        if not parameter.is_floating_point():
            raise TypeError(f"V11 {label} parameter is not floating: {suffix}")


def v11_parameter_inventories(
    model: Any,
) -> tuple[
    tuple[tuple[str, Any], ...],
    tuple[tuple[str, Any], ...],
    tuple[tuple[str, Any], ...],
]:
    """Validate and return online attention, target attention, semantic axes.

    Names in each returned tuple are relative to their registered module
    prefix.  Registration order is scientific state: floor precedes elevated,
    and FREE precedes OCCUPIED.
    """

    named = tuple(model.named_parameters())
    names = tuple(name for name, _ in named)
    if len(names) != len(set(names)):
        raise RuntimeError("V11 model exposes duplicate parameter names")

    online_attention = tuple(
        (name.removeprefix(ONLINE_LIFT_PREFIX_V11), parameter)
        for name, parameter in named
        if name.startswith(ONLINE_LIFT_PREFIX_V11)
        and _is_branch_attention_suffix_v11(
            name.removeprefix(ONLINE_LIFT_PREFIX_V11)
        )
    )
    target_attention = tuple(
        (name.removeprefix(TARGET_LIFT_PREFIX_V11), parameter)
        for name, parameter in named
        if name.startswith(TARGET_LIFT_PREFIX_V11)
        and _is_branch_attention_suffix_v11(
            name.removeprefix(TARGET_LIFT_PREFIX_V11)
        )
    )
    semantic_axes = tuple(
        (name.removeprefix(ONLINE_SEMANTIC_PREFIX_V11), parameter)
        for name, parameter in named
        if name.startswith(ONLINE_SEMANTIC_PREFIX_V11)
    )
    target_semantic = tuple(
        name for name in names if name.startswith(TARGET_SEMANTIC_PREFIX_V11)
    )

    if tuple(name for name, _ in online_attention) != (
        BRANCH_ATTENTION_PARAMETER_SUFFIXES_V11
    ):
        raise RuntimeError("V11 online height-role attention inventory changed")
    if tuple(name for name, _ in target_attention) != (
        BRANCH_ATTENTION_PARAMETER_SUFFIXES_V11
    ):
        raise RuntimeError("V11 target height-role attention inventory changed")
    if tuple(name for name, _ in semantic_axes) != (
        SEMANTIC_AXIS_PARAMETER_SUFFIXES_V11
    ):
        raise RuntimeError("V11 factorized semantic-axis inventory changed")
    if target_semantic:
        raise RuntimeError("V11 unexpectedly added a target semantic head")

    _validate_parameter_shapes_v11(
        online_attention,
        _ATTENTION_EXPECTED_SHAPES_V11,
        label="online height-role attention",
    )
    _validate_parameter_shapes_v11(
        target_attention,
        _ATTENTION_EXPECTED_SHAPES_V11,
        label="target height-role attention",
    )
    _validate_parameter_shapes_v11(
        semantic_axes,
        _SEMANTIC_EXPECTED_SHAPES_V11,
        label="factorized semantic axis",
    )
    if any(
        online_parameter.shape != target_parameter.shape
        or online_parameter.dtype != target_parameter.dtype
        for (_, online_parameter), (_, target_parameter) in zip(
            online_attention, target_attention, strict=True
        )
    ):
        raise RuntimeError("V11 online/target attention tensor metadata differ")

    online_attention_count = sum(
        parameter.numel() for _, parameter in online_attention
    )
    target_attention_count = sum(
        parameter.numel() for _, parameter in target_attention
    )
    semantic_count = sum(parameter.numel() for _, parameter in semantic_axes)
    if (
        len(online_attention)
        != HEIGHT_ROLE_BRANCH_ATTENTION_PARAMETER_TENSOR_COUNT_V11
        or len(target_attention)
        != HEIGHT_ROLE_BRANCH_ATTENTION_PARAMETER_TENSOR_COUNT_V11
        or online_attention_count
        != HEIGHT_ROLE_BRANCH_ATTENTION_PARAMETER_COUNT_V11
        or target_attention_count
        != HEIGHT_ROLE_BRANCH_ATTENTION_PARAMETER_COUNT_V11
    ):
        raise RuntimeError("V11 height-role attention parameter count changed")
    if (
        len(semantic_axes) != FACTORIZED_SEMANTIC_AXIS_PARAMETER_TENSOR_COUNT_V11
        or semantic_count != FACTORIZED_SEMANTIC_AXIS_PARAMETER_COUNT_V11
    ):
        raise RuntimeError("V11 factorized semantic parameter count changed")
    if any(not parameter.requires_grad for _, parameter in online_attention):
        raise RuntimeError("V11 online height-role attention tensor is frozen")
    if any(parameter.requires_grad for _, parameter in target_attention):
        raise RuntimeError("V11 target height-role attention tensor is trainable")
    if any(not parameter.requires_grad for _, parameter in semantic_axes):
        raise RuntimeError("V11 factorized semantic-axis tensor is frozen")

    all_parameters = tuple(
        parameter
        for inventory in (online_attention, target_attention, semantic_axes)
        for _, parameter in inventory
    )
    if len({id(parameter) for parameter in all_parameters}) != len(all_parameters):
        raise RuntimeError("V11 new parameter inventories alias one another")
    return online_attention, target_attention, semantic_axes


def _gradient_receipt_v11(
    model: Any,
    *,
    group: str,
) -> dict[str, Any]:
    torch = _v3._v2._v1._runtime_apis()[0]
    online_attention, target_attention, semantic_axes = (
        v11_parameter_inventories(model)
    )
    if any(parameter.grad is not None for _, parameter in target_attention):
        raise RuntimeError("V11 target height-role attention received a gradient")

    if group == "height_role_branch_attention":
        online = online_attention
        expected_count = HEIGHT_ROLE_BRANCH_ATTENTION_PARAMETER_COUNT_V11
        expected_tensor_count = (
            HEIGHT_ROLE_BRANCH_ATTENTION_PARAMETER_TENSOR_COUNT_V11
        )
        schema = _ATTENTION_RECEIPT_SCHEMA_V11
        target_parameter_tensor_count = len(target_attention)
    elif group == "factorized_semantic_axes":
        online = semantic_axes
        expected_count = FACTORIZED_SEMANTIC_AXIS_PARAMETER_COUNT_V11
        expected_tensor_count = FACTORIZED_SEMANTIC_AXIS_PARAMETER_TENSOR_COUNT_V11
        schema = _SEMANTIC_RECEIPT_SCHEMA_V11
        target_parameter_tensor_count = 0
    else:
        raise ValueError("unknown V11 gradient-receipt group")

    values: list[float] = []
    for suffix, parameter in online:
        gradient = parameter.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(
                f"V11 {group} gradient is absent or nonfinite for {suffix}"
            )
        value = float(
            gradient.detach().double().square().sum().sqrt().cpu().item()
        )
        if not math.isfinite(value):
            raise FloatingPointError(f"V11 {group} gradient norm is nonfinite")
        values.append(value)

    suffixes = tuple(name for name, _ in online)
    inactive = tuple(
        name
        for name, value in zip(suffixes, values, strict=True)
        if value == 0.0
    )
    update = int(model.ema_update_count.item())
    if update < 1:
        raise RuntimeError("V11 gradient receipt precedes the first EMA update")
    if sum(parameter.numel() for _, parameter in online) != expected_count or len(
        online
    ) != expected_tensor_count:
        raise RuntimeError(f"V11 {group} receipt inventory changed")
    return {
        "schema": schema,
        "update": update,
        "measurement": "post_clip_post_optimizer_step_retained_gradient",
        "gradient_l2": math.sqrt(sum(value * value for value in values)),
        "minimum_parameter_gradient_l2": min(values),
        "maximum_parameter_gradient_l2": max(values),
        "online_parameter_count": expected_count,
        "online_parameter_tensor_count": expected_tensor_count,
        "gradient_tensor_count": len(values),
        "active_parameter_tensor_count": len(values) - len(inactive),
        "inactive_parameter_suffixes": list(inactive),
        "parameter_suffix_inventory_sha256": _inventory_sha256_v11(suffixes),
        "target_parameter_tensor_count": target_parameter_tensor_count,
        "target_gradient_tensor_count": 0,
    }


def branch_attention_gradient_receipt_v11(model: Any) -> dict[str, Any]:
    """Receipt all 14 online role-attention gradients and frozen targets."""

    return _gradient_receipt_v11(model, group="height_role_branch_attention")


def semantic_axis_gradient_receipt_v11(model: Any) -> dict[str, Any]:
    """Receipt all 12 online factorized-semantic-axis gradients."""

    return _gradient_receipt_v11(model, group="factorized_semantic_axes")


def _summarize_parameter_activity_v11(
    receipts: Sequence[Mapping[str, Any]],
    parameter_suffixes: Sequence[str],
    *,
    group: str,
    expected_parameter_count: int,
    receipt_schema: str,
    activity_schema: str,
    target_parameter_tensor_count: int,
) -> dict[str, Any]:
    if len(receipts) != MAXIMUM_UPDATES:
        raise RuntimeError(f"V11 {group} receipt count changed")
    suffixes = tuple(parameter_suffixes)
    inventory_hash = _inventory_sha256_v11(suffixes)
    first_active: dict[str, int | None] = {name: None for name in suffixes}
    minimum_gradient = math.inf
    maximum_gradient = 0.0
    minimum_active = len(suffixes)
    maximum_active = 0
    active_update_count = 0

    for expected_update, receipt in enumerate(receipts, start=1):
        if receipt.get("schema") != receipt_schema:
            raise RuntimeError(f"V11 {group} receipt schema changed")
        if int(receipt.get("update", -1)) != expected_update:
            raise RuntimeError(f"V11 {group} receipt order changed")
        if receipt.get("measurement") != (
            "post_clip_post_optimizer_step_retained_gradient"
        ):
            raise RuntimeError(f"V11 {group} gradient measurement changed")
        if receipt.get("parameter_suffix_inventory_sha256") != inventory_hash:
            raise RuntimeError(f"V11 {group} gradient inventory changed")
        if (
            int(receipt.get("online_parameter_count", -1))
            != expected_parameter_count
            or int(receipt.get("online_parameter_tensor_count", -1))
            != len(suffixes)
            or int(receipt.get("gradient_tensor_count", -1)) != len(suffixes)
            or int(receipt.get("target_parameter_tensor_count", -1))
            != target_parameter_tensor_count
            or int(receipt.get("target_gradient_tensor_count", -1)) != 0
        ):
            raise RuntimeError(f"V11 {group} receipt counts changed")

        raw_inactive = tuple(receipt.get("inactive_parameter_suffixes", ()))
        inactive = set(raw_inactive)
        if len(inactive) != len(raw_inactive) or not inactive <= set(suffixes):
            raise RuntimeError(f"V11 {group} receipt names changed")
        active_count = int(receipt.get("active_parameter_tensor_count", -1))
        if active_count != len(suffixes) - len(inactive):
            raise RuntimeError(f"V11 {group} active tensor count changed")

        gradient_l2 = float(receipt.get("gradient_l2", math.nan))
        minimum_parameter = float(
            receipt.get("minimum_parameter_gradient_l2", math.nan)
        )
        maximum_parameter = float(
            receipt.get("maximum_parameter_gradient_l2", math.nan)
        )
        if (
            not math.isfinite(gradient_l2)
            or not math.isfinite(minimum_parameter)
            or not math.isfinite(maximum_parameter)
            or min(gradient_l2, minimum_parameter, maximum_parameter) < 0.0
            or minimum_parameter > maximum_parameter
        ):
            raise FloatingPointError(f"V11 {group} receipt norm is invalid")
        if gradient_l2 > 0.0:
            active_update_count += 1
        minimum_gradient = min(minimum_gradient, gradient_l2)
        maximum_gradient = max(maximum_gradient, gradient_l2)
        minimum_active = min(minimum_active, active_count)
        maximum_active = max(maximum_active, active_count)
        for suffix in suffixes:
            if first_active[suffix] is None and suffix not in inactive:
                first_active[suffix] = expected_update

    late_or_inactive = {
        name: update
        for name, update in first_active.items()
        if update is None or update > 2
    }
    if late_or_inactive:
        raise RuntimeError(
            f"V11 {group} tensors were not active by update 2: "
            + ", ".join(
                f"{name}={update}" for name, update in late_or_inactive.items()
            )
        )
    first_active_updates = {
        name: int(update)
        for name, update in first_active.items()
        if update is not None
    }
    return {
        "schema": activity_schema,
        "update_count": len(receipts),
        "online_parameter_count": expected_parameter_count,
        "online_parameter_tensor_count": len(suffixes),
        "parameter_suffix_inventory_sha256": inventory_hash,
        "all_online_parameter_tensors_active_by_update_2": True,
        "first_active_update": first_active_updates,
        "latest_first_active_update": max(first_active_updates.values()),
        "active_update_count": active_update_count,
        "minimum_active_parameter_tensor_count": minimum_active,
        "maximum_active_parameter_tensor_count": maximum_active,
        "minimum_gradient_l2": minimum_gradient,
        "maximum_gradient_l2": maximum_gradient,
        "target_parameter_tensor_count": target_parameter_tensor_count,
        "target_gradient_tensor_count": 0,
    }


def _summarize_branch_attention_activity_v11(
    receipts: Sequence[Mapping[str, Any]],
    parameter_suffixes: Sequence[str],
) -> dict[str, Any]:
    return _summarize_parameter_activity_v11(
        receipts,
        parameter_suffixes,
        group="height-role branch-attention",
        expected_parameter_count=HEIGHT_ROLE_BRANCH_ATTENTION_PARAMETER_COUNT_V11,
        receipt_schema=_ATTENTION_RECEIPT_SCHEMA_V11,
        activity_schema=_ATTENTION_ACTIVITY_SCHEMA_V11,
        target_parameter_tensor_count=(
            HEIGHT_ROLE_BRANCH_ATTENTION_PARAMETER_TENSOR_COUNT_V11
        ),
    )


def _summarize_semantic_axis_activity_v11(
    receipts: Sequence[Mapping[str, Any]],
    parameter_suffixes: Sequence[str],
) -> dict[str, Any]:
    return _summarize_parameter_activity_v11(
        receipts,
        parameter_suffixes,
        group="factorized semantic-axis",
        expected_parameter_count=FACTORIZED_SEMANTIC_AXIS_PARAMETER_COUNT_V11,
        receipt_schema=_SEMANTIC_RECEIPT_SCHEMA_V11,
        activity_schema=_SEMANTIC_ACTIVITY_SCHEMA_V11,
        target_parameter_tensor_count=0,
    )


def _validate_training_cap_v11(maximum_updates: int) -> None:
    if (
        MICROBATCH_SIZE != 4
        or MICROBATCHES_PER_UPDATE != 4
        or PRESENTATIONS_PER_UPDATE != 16
        or MAXIMUM_UPDATES != 1_000
        or MAXIMUM_PRESENTATIONS != 16_000
        or maximum_updates != MAXIMUM_UPDATES
    ):
        raise RuntimeError("V11 frozen 1,000-update/16,000-presentation cap changed")


def _validate_inherited_training_result_v11(
    accounting: JointTrainingAccountingV1,
    trace: Sequence[Mapping[str, Any]],
) -> None:
    _v3.validate_accounting_v1(accounting)
    expected_accounting = JointTrainingAccountingV1(
        updates=1_000,
        presentations=16_000,
        microbatch_graphs=4_000,
        backward_calls=4_000,
        optimizer_steps=1_000,
        ema_steps=1_000,
        predictor_forwards=4_000,
        predictor_objectives=4_000,
    )
    if accounting != expected_accounting:
        raise RuntimeError("V11 inherited terminal training accounting changed")
    if len(trace) != MAXIMUM_UPDATES:
        raise RuntimeError("V11 inherited trace length changed")
    for expected_update, row in enumerate(trace, start=1):
        if tuple(row) != ("update", "presentations", "losses", "gradient_l2"):
            raise RuntimeError("V11 inherited trace fields changed")
        if int(row["update"]) != expected_update or int(row["presentations"]) != (
            expected_update * PRESENTATIONS_PER_UPDATE
        ):
            raise RuntimeError("V11 inherited trace accounting changed")
        if tuple(row["losses"]) != ("S", "P", "U", "R", "O", "L"):
            raise RuntimeError("V11 inherited S+P+U+R+O loss schema changed")
        if tuple(row["gradient_l2"]) != (
            "encoder",
            "lift_semantic",
            "predictor",
        ):
            raise RuntimeError("V11 inherited gradient-group schema changed")


def run_fixed_training_v11(
    model: Any,
    optimizer: Any,
    loader: Any,
    train_pairs: Sequence[Mapping[str, Any]],
    train_labels: FrozenSurvivalRoleLabelsV1,
    schedule: Sequence[int],
    device: Any,
    *,
    action_order: Sequence[str] = ACTION_ORDER,
    maximum_updates: int = MAXIMUM_UPDATES,
) -> tuple[JointTrainingAccountingV1, tuple[dict[str, Any], ...], dict[str, Any]]:
    """Run exact V3 science and append V11 new-parameter evidence."""

    _validate_training_cap_v11(maximum_updates)
    online_attention, _, semantic_axes = v11_parameter_inventories(model)
    attention_suffixes = tuple(name for name, _ in online_attention)
    semantic_suffixes = tuple(name for name, _ in semantic_axes)
    attention_receipts: list[Mapping[str, Any]] = []
    semantic_receipts: list[Mapping[str, Any]] = []

    def tracked_update(
        candidate: Any,
        candidate_optimizer: Any,
        microbatches: Sequence[Mapping[str, Any]],
        *,
        accounting: JointTrainingAccountingV1,
    ) -> Any:
        result = _v3.joint_training_update_v3(
            candidate,
            candidate_optimizer,
            microbatches,
            accounting=accounting,
        )
        attention_receipts.append(branch_attention_gradient_receipt_v11(candidate))
        semantic_receipts.append(semantic_axis_gradient_receipt_v11(candidate))
        return result

    accounting, inherited_trace, inherited_diagnostics = (
        _v3._v2._run_fixed_training_core_v2(
            model,
            optimizer,
            loader,
            train_pairs,
            train_labels,
            schedule,
            device,
            action_order=action_order,
            maximum_updates=maximum_updates,
            microbatch_builder=build_microbatch_v1,
            joint_update=tracked_update,
        )
    )
    _validate_inherited_training_result_v11(accounting, inherited_trace)
    attention_activity = _summarize_branch_attention_activity_v11(
        attention_receipts,
        attention_suffixes,
    )
    semantic_activity = _summarize_semantic_axis_activity_v11(
        semantic_receipts,
        semantic_suffixes,
    )
    trace = tuple(
        {
            **row,
            "height_role_branch_attention": dict(attention_receipt),
            "factorized_semantic_axes": dict(semantic_receipt),
        }
        for row, attention_receipt, semantic_receipt in zip(
            inherited_trace,
            attention_receipts,
            semantic_receipts,
            strict=True,
        )
    )
    diagnostics = {
        **inherited_diagnostics,
        "height_role_branch_attention": attention_activity,
        "factorized_semantic_axes": semantic_activity,
    }
    return accounting, trace, diagnostics


__all__ = [
    "ACTION_ORDER",
    "BRANCH_ATTENTION_MODULE_STEMS_V11",
    "BRANCH_ATTENTION_PARAMETER_SUFFIXES_V11",
    "FACTORIZED_SEMANTIC_AXIS_ONLINE_PARAMETER_NAMES_V11",
    "FACTORIZED_SEMANTIC_AXIS_PARAMETER_COUNT_V11",
    "FACTORIZED_SEMANTIC_AXIS_PARAMETER_TENSOR_COUNT_V11",
    "FrozenSurvivalRoleLabelsV1",
    "HEIGHT_ROLE_BRANCH_ATTENTION_ONLINE_PARAMETER_NAMES_V11",
    "HEIGHT_ROLE_BRANCH_ATTENTION_PARAMETER_COUNT_V11",
    "HEIGHT_ROLE_BRANCH_ATTENTION_PARAMETER_TENSOR_COUNT_V11",
    "HEIGHT_ROLE_BRANCH_ATTENTION_TARGET_PARAMETER_NAMES_V11",
    "JointTrainingAccountingV1",
    "MAXIMUM_PRESENTATIONS",
    "MAXIMUM_UPDATES",
    "MICROBATCHES_PER_UPDATE",
    "MICROBATCH_SIZE",
    "OCCUPIED_CLASS_INDEX",
    "OCCUPIED_SAFETY_AUX_COEFFICIENT",
    "OCCUPIED_SAFETY_AUX_NORMALIZATION",
    "ONLINE_LIFT_PREFIX_V11",
    "ONLINE_SEMANTIC_PREFIX_V11",
    "PRESENTATIONS_PER_UPDATE",
    "SEMANTIC_AXIS_PARAMETER_SUFFIXES_V11",
    "TARGET_LIFT_PREFIX_V11",
    "branch_attention_gradient_receipt_v11",
    "build_frozen_optimizer_v1",
    "build_microbatch_v1",
    "freeze_role_labels_v1",
    "partition_parameters_v1",
    "run_fixed_training_v11",
    "semantic_axis_gradient_receipt_v11",
    "v11_parameter_inventories",
    "validate_pairs_against_labels_v1",
]
