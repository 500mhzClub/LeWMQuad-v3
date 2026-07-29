#!/usr/bin/env python3
"""Lean V9 wrapper around the unchanged V3/V4 joint-training core.

V9 changes only the image-token aggregation inside the BEV lift.  Losses,
backward, clipping, optimizer validation, EMA, schedule, and accounting remain
delegated to the frozen V3 implementation; this wrapper receipts every new
online attention gradient and proves that the matching target stays frozen.
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
    from scripts import (
        run_go2_rgb_swept_progress_survival_joint_jepa_v3_half_occupied_safety_aux
        as _v3,
    )
finally:
    if _inserted_root:
        sys.path.remove(str(ROOT))


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

DENSE_LOCAL_ATTENTION_ADDED_PARAMETER_COUNT_V9 = 16_576
ONLINE_LIFT_PREFIX_V9 = "bev_lift."
TARGET_LIFT_PREFIX_V9 = "target_bev_lift."
ATTENTION_PARAMETER_SUFFIXES_V9 = (
    "query_projection.weight",
    "query_projection.bias",
    "key_projection.weight",
    "value_projection.weight",
    "value_projection.bias",
    "output_projection.weight",
    "output_projection.bias",
)
DENSE_LOCAL_ATTENTION_ONLINE_PARAMETER_NAMES_V9 = tuple(
    ONLINE_LIFT_PREFIX_V9 + suffix for suffix in ATTENTION_PARAMETER_SUFFIXES_V9
)
DENSE_LOCAL_ATTENTION_TARGET_PARAMETER_NAMES_V9 = tuple(
    TARGET_LIFT_PREFIX_V9 + suffix for suffix in ATTENTION_PARAMETER_SUFFIXES_V9
)


def _attention_parameter_inventory_v9(
    model: Any,
) -> tuple[tuple[tuple[str, Any], ...], tuple[tuple[str, Any], ...]]:
    named = tuple(model.named_parameters())
    online = tuple(
        (name.removeprefix(ONLINE_LIFT_PREFIX_V9), parameter)
        for name, parameter in named
        if name in DENSE_LOCAL_ATTENTION_ONLINE_PARAMETER_NAMES_V9
    )
    target = tuple(
        (name.removeprefix(TARGET_LIFT_PREFIX_V9), parameter)
        for name, parameter in named
        if name in DENSE_LOCAL_ATTENTION_TARGET_PARAMETER_NAMES_V9
    )
    if tuple(name for name, _ in online) != ATTENTION_PARAMETER_SUFFIXES_V9:
        raise RuntimeError("V9 online dense-local attention inventory changed")
    if tuple(name for name, _ in target) != ATTENTION_PARAMETER_SUFFIXES_V9:
        raise RuntimeError("V9 target dense-local attention inventory changed")
    if any(
        online_parameter.shape != target_parameter.shape
        for (_, online_parameter), (_, target_parameter) in zip(
            online, target, strict=True
        )
    ):
        raise RuntimeError("V9 online/target attention tensor shapes differ")
    if {
        sum(parameter.numel() for _, parameter in online),
        sum(parameter.numel() for _, parameter in target),
    } != {DENSE_LOCAL_ATTENTION_ADDED_PARAMETER_COUNT_V9}:
        raise RuntimeError("V9 dense-local attention parameter count changed")
    if any(not parameter.requires_grad for _, parameter in online):
        raise RuntimeError("V9 online dense-local attention parameter is frozen")
    if any(parameter.requires_grad for _, parameter in target):
        raise RuntimeError("V9 target dense-local attention parameter is trainable")
    return online, target


def _inventory_sha256_v9(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def attention_gradient_receipt_v9(model: Any) -> dict[str, Any]:
    """Receipt retained post-clip gradients for every new attention tensor."""

    torch = _v3._v2._v1._runtime_apis()[0]
    online, target = _attention_parameter_inventory_v9(model)
    if any(parameter.grad is not None for _, parameter in target):
        raise RuntimeError("V9 target dense-local attention received a gradient")

    squared_norms = []
    for suffix, parameter in online:
        gradient = parameter.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(
                f"V9 attention gradient is absent or nonfinite for {suffix}"
            )
        squared_norms.append(gradient.detach().double().square().sum())
    values = torch.stack(squared_norms).sqrt().detach().cpu().tolist()
    if any(not math.isfinite(float(value)) for value in values):
        raise FloatingPointError("V9 attention gradient norm is nonfinite")

    suffixes = tuple(name for name, _ in online)
    inactive = tuple(
        name
        for name, value in zip(suffixes, values, strict=True)
        if float(value) == 0.0
    )
    aggregate = math.sqrt(sum(float(value) ** 2 for value in values))
    update = int(model.ema_update_count.item())
    if update < 1:
        raise RuntimeError("V9 gradient receipt precedes the first EMA update")
    return {
        "schema": "lewm_v9_dense_local_attention_post_backward_gradient_v1",
        "update": update,
        "measurement": "post_clip_post_optimizer_step_retained_gradient",
        "gradient_l2": aggregate,
        "minimum_parameter_gradient_l2": min(float(value) for value in values),
        "maximum_parameter_gradient_l2": max(float(value) for value in values),
        "online_parameter_count": sum(parameter.numel() for _, parameter in online),
        "online_parameter_tensor_count": len(online),
        "gradient_tensor_count": len(values),
        "active_parameter_tensor_count": len(values) - len(inactive),
        "inactive_parameter_suffixes": list(inactive),
        "parameter_suffix_inventory_sha256": _inventory_sha256_v9(suffixes),
        "target_gradient_tensor_count": 0,
    }


def _summarize_attention_activity_v9(
    receipts: Sequence[Mapping[str, Any]], parameter_suffixes: Sequence[str]
) -> dict[str, Any]:
    if len(receipts) != MAXIMUM_UPDATES:
        raise RuntimeError("V9 dense-local attention receipt count changed")
    suffixes = tuple(parameter_suffixes)
    if suffixes != ATTENTION_PARAMETER_SUFFIXES_V9:
        raise RuntimeError("V9 dense-local attention summary inventory changed")
    inventory_hash = _inventory_sha256_v9(suffixes)
    first_active: dict[str, int | None] = {name: None for name in suffixes}
    for expected, receipt in enumerate(receipts, start=1):
        if int(receipt.get("update", -1)) != expected:
            raise RuntimeError("V9 dense-local attention receipt order changed")
        if receipt.get("parameter_suffix_inventory_sha256") != inventory_hash:
            raise RuntimeError("V9 dense-local attention gradient inventory changed")
        if int(receipt.get("online_parameter_count", -1)) != (
            DENSE_LOCAL_ATTENTION_ADDED_PARAMETER_COUNT_V9
        ) or int(receipt.get("online_parameter_tensor_count", -1)) != len(suffixes):
            raise RuntimeError("V9 dense-local attention receipt count changed")
        if int(receipt.get("target_gradient_tensor_count", -1)) != 0:
            raise RuntimeError("V9 target dense-local attention gradient was recorded")
        inactive = set(receipt["inactive_parameter_suffixes"])
        if not inactive <= set(suffixes):
            raise RuntimeError("V9 dense-local attention receipt names changed")
        for suffix in suffixes:
            if first_active[suffix] is None and suffix not in inactive:
                first_active[suffix] = expected

    late_or_inactive = {
        name: update
        for name, update in first_active.items()
        if update is None or update > 2
    }
    if late_or_inactive:
        raise RuntimeError(
            "V9 attention tensors were not active by update 2: "
            + ", ".join(
                f"{name}={update}" for name, update in late_or_inactive.items()
            )
        )
    first_active_updates = {
        name: int(update) for name, update in first_active.items() if update is not None
    }
    return {
        "schema": "lewm_v9_dense_local_attention_training_activity_v1",
        "update_count": len(receipts),
        "online_parameter_count": DENSE_LOCAL_ATTENTION_ADDED_PARAMETER_COUNT_V9,
        "online_parameter_tensor_count": len(suffixes),
        "parameter_suffix_inventory_sha256": inventory_hash,
        "all_online_parameter_tensors_active_by_update_2": True,
        "first_active_update": first_active_updates,
        "latest_first_active_update": max(first_active_updates.values()),
        "active_update_count": sum(
            float(receipt["gradient_l2"]) > 0.0 for receipt in receipts
        ),
        "minimum_active_parameter_tensor_count": min(
            int(receipt["active_parameter_tensor_count"]) for receipt in receipts
        ),
        "maximum_active_parameter_tensor_count": max(
            int(receipt["active_parameter_tensor_count"]) for receipt in receipts
        ),
        "minimum_gradient_l2": min(
            float(receipt["gradient_l2"]) for receipt in receipts
        ),
        "maximum_gradient_l2": max(
            float(receipt["gradient_l2"]) for receipt in receipts
        ),
        "target_gradient_tensor_count": 0,
    }


def _validate_training_cap_v9(maximum_updates: int) -> None:
    if (
        MICROBATCH_SIZE != 4
        or MICROBATCHES_PER_UPDATE != 4
        or PRESENTATIONS_PER_UPDATE != 16
        or MAXIMUM_UPDATES != 1_000
        or MAXIMUM_PRESENTATIONS != 16_000
        or maximum_updates != MAXIMUM_UPDATES
    ):
        raise RuntimeError("V9 frozen 1,000-update/16,000-presentation cap changed")


def run_fixed_training_v9(
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
    """Run exact V3/V4 training and append V9 attention evidence."""

    _validate_training_cap_v9(maximum_updates)
    online, _ = _attention_parameter_inventory_v9(model)
    parameter_suffixes = tuple(name for name, _ in online)
    receipts: list[Mapping[str, Any]] = []

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
        receipts.append(attention_gradient_receipt_v9(candidate))
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
    activity = _summarize_attention_activity_v9(receipts, parameter_suffixes)
    trace = tuple(
        {**row, "dense_local_attention": dict(receipt)}
        for row, receipt in zip(inherited_trace, receipts, strict=True)
    )
    diagnostics = {**inherited_diagnostics, "dense_local_attention": activity}
    return accounting, trace, diagnostics


__all__ = [
    "ACTION_ORDER",
    "ATTENTION_PARAMETER_SUFFIXES_V9",
    "DENSE_LOCAL_ATTENTION_ADDED_PARAMETER_COUNT_V9",
    "DENSE_LOCAL_ATTENTION_ONLINE_PARAMETER_NAMES_V9",
    "DENSE_LOCAL_ATTENTION_TARGET_PARAMETER_NAMES_V9",
    "MAXIMUM_PRESENTATIONS",
    "MAXIMUM_UPDATES",
    "MICROBATCHES_PER_UPDATE",
    "MICROBATCH_SIZE",
    "OCCUPIED_CLASS_INDEX",
    "OCCUPIED_SAFETY_AUX_COEFFICIENT",
    "OCCUPIED_SAFETY_AUX_NORMALIZATION",
    "ONLINE_LIFT_PREFIX_V9",
    "PRESENTATIONS_PER_UPDATE",
    "TARGET_LIFT_PREFIX_V9",
    "attention_gradient_receipt_v9",
    "build_frozen_optimizer_v1",
    "build_microbatch_v1",
    "freeze_role_labels_v1",
    "partition_parameters_v1",
    "run_fixed_training_v9",
    "validate_pairs_against_labels_v1",
]
