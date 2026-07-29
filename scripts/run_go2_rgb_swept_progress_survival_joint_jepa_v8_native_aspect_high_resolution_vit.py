#!/usr/bin/env python3
"""Lean V8 wrapper around the unchanged V3/V4 joint-training core.

V8 changes only native RGB/encoder token geometry and its positional migration.
Losses, backward, clipping, optimizer, EMA, schedule, and accounting remain
delegated to the frozen V3 implementation; this wrapper receipts gradients for
the complete online high-resolution ViT.
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

NATIVE_ASPECT_HIGH_RESOLUTION_VIT_TRAINABLE_PARAMETER_COUNT_V8 = 2_845_824
NATIVE_ENCODER_TRAINABLE_PARAMETER_COUNT_V8 = (
    NATIVE_ASPECT_HIGH_RESOLUTION_VIT_TRAINABLE_PARAMETER_COUNT_V8
)
ONLINE_ENCODER_PREFIX_V8 = "encoder."
TARGET_ENCODER_PREFIX_V8 = "target_encoder."


def _encoder_parameter_inventory_v8(
    model: Any,
) -> tuple[tuple[tuple[str, Any], ...], tuple[tuple[str, Any], ...]]:
    named = tuple(model.named_parameters())
    online = tuple(
        (name.removeprefix(ONLINE_ENCODER_PREFIX_V8), parameter)
        for name, parameter in named
        if name.startswith(ONLINE_ENCODER_PREFIX_V8)
    )
    target = tuple(
        (name.removeprefix(TARGET_ENCODER_PREFIX_V8), parameter)
        for name, parameter in named
        if name.startswith(TARGET_ENCODER_PREFIX_V8)
    )
    if not online or tuple(name for name, _ in online) != tuple(
        name for name, _ in target
    ):
        raise RuntimeError("V8 online/target ViT parameter inventories differ")
    if sum(parameter.numel() for _, parameter in online) != (
        NATIVE_ASPECT_HIGH_RESOLUTION_VIT_TRAINABLE_PARAMETER_COUNT_V8
    ):
        raise RuntimeError("V8 native-aspect high-resolution ViT parameter count changed")
    if any(not parameter.requires_grad for _, parameter in online):
        raise RuntimeError("V8 online high-resolution ViT parameter is frozen")
    if any(parameter.requires_grad for _, parameter in target):
        raise RuntimeError("V8 target high-resolution ViT parameter is trainable")
    return online, target


def _inventory_sha256_v8(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def vit_encoder_gradient_receipt_v8(model: Any) -> dict[str, Any]:
    """Receipt retained post-clip gradients for every online ViT tensor."""

    torch = _v3._v2._v1._runtime_apis()[0]
    online, target = _encoder_parameter_inventory_v8(model)
    if any(parameter.grad is not None for _, parameter in target):
        raise RuntimeError("V8 target high-resolution ViT received a gradient")
    squared_norms = []
    for suffix, parameter in online:
        gradient = parameter.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(
                f"V8 ViT gradient is absent or nonfinite for {suffix}"
            )
        squared_norms.append(gradient.detach().double().square().sum())
    values = torch.stack(squared_norms).sqrt().detach().cpu().tolist()
    if any(not math.isfinite(float(value)) for value in values):
        raise FloatingPointError("V8 ViT gradient norm is nonfinite")
    aggregate = math.sqrt(sum(float(value) ** 2 for value in values))
    if aggregate <= 0.0:
        raise RuntimeError("V8 native-aspect high-resolution ViT received no gradient")
    suffixes = tuple(name for name, _ in online)
    inactive = tuple(
        name
        for name, value in zip(suffixes, values, strict=True)
        if float(value) == 0.0
    )
    update = int(model.ema_update_count.item())
    if update < 1:
        raise RuntimeError("V8 gradient receipt precedes the first EMA update")
    return {
        "schema": "lewm_v8_native_aspect_high_resolution_vit_post_backward_gradient_v1",
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
        "parameter_suffix_inventory_sha256": _inventory_sha256_v8(suffixes),
        "target_gradient_tensor_count": 0,
    }


def _summarize_encoder_activity_v8(
    receipts: Sequence[Mapping[str, Any]], parameter_suffixes: Sequence[str]
) -> dict[str, Any]:
    if len(receipts) != MAXIMUM_UPDATES:
        raise RuntimeError("V8 high-resolution ViT receipt count changed")
    suffixes = tuple(parameter_suffixes)
    inventory_hash = _inventory_sha256_v8(suffixes)
    first_active = {name: None for name in suffixes}
    for expected, receipt in enumerate(receipts, start=1):
        if int(receipt.get("update", -1)) != expected:
            raise RuntimeError("V8 high-resolution ViT receipt order changed")
        if receipt.get("parameter_suffix_inventory_sha256") != inventory_hash:
            raise RuntimeError("V8 high-resolution ViT gradient inventory changed")
        inactive = set(receipt["inactive_parameter_suffixes"])
        if not inactive <= set(suffixes):
            raise RuntimeError("V8 high-resolution ViT receipt names changed")
        for suffix in suffixes:
            if first_active[suffix] is None and suffix not in inactive:
                first_active[suffix] = expected
    never_active = [name for name, update in first_active.items() if update is None]
    if never_active:
        raise RuntimeError(
            "V8 high-resolution ViT tensors never received a nonzero gradient: "
            + ", ".join(never_active)
        )
    return {
        "schema": "lewm_v8_native_aspect_high_resolution_vit_training_activity_v1",
        "update_count": len(receipts),
        "online_parameter_count": int(receipts[0]["online_parameter_count"]),
        "online_parameter_tensor_count": len(suffixes),
        "parameter_suffix_inventory_sha256": inventory_hash,
        "all_online_parameter_tensors_received_gradient": True,
        "latest_first_active_update": max(int(value) for value in first_active.values()),
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
        "target_gradient_tensor_count": sum(
            int(receipt["target_gradient_tensor_count"]) for receipt in receipts
        ),
    }


def run_fixed_training_v8(
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
    """Run exact V3/V4 training and append V8 encoder evidence."""

    online, _ = _encoder_parameter_inventory_v8(model)
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
        receipts.append(vit_encoder_gradient_receipt_v8(candidate))
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
    activity = _summarize_encoder_activity_v8(receipts, parameter_suffixes)
    trace = tuple(
        {**row, "native_aspect_high_resolution_vit_encoder": dict(receipt)}
        for row, receipt in zip(inherited_trace, receipts, strict=True)
    )
    diagnostics = {
        **inherited_diagnostics,
        "native_aspect_high_resolution_vit_encoder": activity,
    }
    return accounting, trace, diagnostics


__all__ = [
    "ACTION_ORDER",
    "MAXIMUM_PRESENTATIONS",
    "MAXIMUM_UPDATES",
    "MICROBATCHES_PER_UPDATE",
    "MICROBATCH_SIZE",
    "NATIVE_ASPECT_HIGH_RESOLUTION_VIT_TRAINABLE_PARAMETER_COUNT_V8",
    "NATIVE_ENCODER_TRAINABLE_PARAMETER_COUNT_V8",
    "OCCUPIED_CLASS_INDEX",
    "OCCUPIED_SAFETY_AUX_COEFFICIENT",
    "OCCUPIED_SAFETY_AUX_NORMALIZATION",
    "ONLINE_ENCODER_PREFIX_V8",
    "PRESENTATIONS_PER_UPDATE",
    "TARGET_ENCODER_PREFIX_V8",
    "build_frozen_optimizer_v1",
    "freeze_role_labels_v1",
    "partition_parameters_v1",
    "run_fixed_training_v8",
    "validate_pairs_against_labels_v1",
    "vit_encoder_gradient_receipt_v8",
]
