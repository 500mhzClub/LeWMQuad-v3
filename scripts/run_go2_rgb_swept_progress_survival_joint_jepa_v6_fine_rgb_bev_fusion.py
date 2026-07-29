#!/usr/bin/env python3
"""Lean V6 wrapper around the unchanged V3/V4 joint-training core.

The V6 scientific delta lives entirely in the model.  This module delegates
every loss, backward, clip, optimizer, EMA, schedule, and accounting operation
to the frozen V3 implementation and only records the retained post-backward
gradients of the new fine-RGB branch.
"""
from __future__ import annotations

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

FINE_RGB_BRANCH_ADDED_TRAINABLE_PARAMETER_COUNT_V6 = 12_256
FINE_RGB_ONLINE_PARAMETER_NAMES_V6 = (
    "bev_lift.fine_rgb_branch.conv1.weight",
    "bev_lift.fine_rgb_branch.conv1.bias",
    "bev_lift.fine_rgb_branch.conv2.weight",
    "bev_lift.fine_rgb_branch.conv2.bias",
    "bev_lift.fine_rgb_branch.output.weight",
    "bev_lift.fine_rgb_branch.output.bias",
)
FINE_RGB_TARGET_PARAMETER_NAMES_V6 = tuple(
    name.replace("bev_lift.", "target_bev_lift.", 1)
    for name in FINE_RGB_ONLINE_PARAMETER_NAMES_V6
)


def _finite_gradient_l2_v6(torch: Any, parameters: Sequence[Any], name: str) -> float:
    if not parameters:
        raise RuntimeError(f"V6 {name} parameter group is empty")
    total = torch.zeros((), dtype=torch.float64, device=parameters[0].device)
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f"V6 {name} gradient is absent or nonfinite")
        total = total + gradient.detach().double().square().sum()
    result = float(total.sqrt().detach().cpu())
    if not math.isfinite(result):
        raise FloatingPointError(f"V6 {name} gradient norm is nonfinite")
    return result


def fine_branch_gradient_receipt_v6(model: Any) -> dict[str, Any]:
    """Receipt the clipped gradients retained after the inherited optimizer step."""

    torch = _v3._v2._v1._runtime_apis()[0]
    named = dict(model.named_parameters())
    if not all(name in named for name in FINE_RGB_ONLINE_PARAMETER_NAMES_V6):
        raise RuntimeError("V6 online fine-RGB parameter identity changed")
    if not all(name in named for name in FINE_RGB_TARGET_PARAMETER_NAMES_V6):
        raise RuntimeError("V6 target fine-RGB parameter identity changed")
    online = tuple(named[name] for name in FINE_RGB_ONLINE_PARAMETER_NAMES_V6)
    target = tuple(named[name] for name in FINE_RGB_TARGET_PARAMETER_NAMES_V6)
    if sum(parameter.numel() for parameter in online) != (
        FINE_RGB_BRANCH_ADDED_TRAINABLE_PARAMETER_COUNT_V6
    ):
        raise RuntimeError("V6 fine-RGB parameter count changed")
    if any(parameter.grad is not None for parameter in target):
        raise RuntimeError("V6 target fine-RGB branch received a gradient")

    groups = {
        "branch": online,
        "conv1": online[0:2],
        "conv2": online[2:4],
        "output": online[4:6],
    }
    norms = {
        name: _finite_gradient_l2_v6(torch, parameters, name)
        for name, parameters in groups.items()
    }
    update = int(model.ema_update_count.item())
    if update < 1:
        raise RuntimeError("V6 gradient receipt precedes the first EMA update")
    if update == 1:
        if norms["branch"] <= 0.0 or norms["output"] <= 0.0:
            raise RuntimeError("V6 zero projection did not receive a first gradient")
        if norms["conv1"] != 0.0 or norms["conv2"] != 0.0:
            raise RuntimeError("V6 pre-projection convolutions activated before unlock")
    return {
        "schema": "lewm_v6_fine_rgb_post_backward_gradient_v1",
        "update": update,
        "measurement": "post_clip_post_optimizer_step_retained_gradient",
        "gradient_l2": norms,
        "active": {name: value > 0.0 for name, value in norms.items()},
        "online_parameter_count": sum(parameter.numel() for parameter in online),
        "online_parameter_tensor_count": len(online),
        "target_gradient_tensor_count": 0,
    }


def _summarize_branch_activity_v6(
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(receipts) != MAXIMUM_UPDATES:
        raise RuntimeError("V6 fine-RGB receipt count changed")
    names = ("branch", "conv1", "conv2", "output")
    for expected, receipt in enumerate(receipts, start=1):
        if int(receipt.get("update", -1)) != expected:
            raise RuntimeError("V6 fine-RGB receipt order changed")
    first_active: dict[str, int] = {}
    for name in names:
        active_updates = [
            index
            for index, receipt in enumerate(receipts, start=1)
            if bool(receipt["active"][name])
        ]
        if not active_updates:
            raise RuntimeError(f"V6 fine-RGB {name} never received a gradient")
        first_active[name] = active_updates[0]
    if first_active["conv1"] <= 1 or first_active["conv2"] <= 1:
        raise RuntimeError("V6 fine-RGB pre-projection unlock timing changed")
    return {
        "schema": "lewm_v6_fine_rgb_training_activity_v1",
        "update_count": len(receipts),
        "first_active_update": first_active,
        "active_update_count": {
            name: sum(bool(receipt["active"][name]) for receipt in receipts)
            for name in names
        },
        "minimum_gradient_l2": {
            name: min(float(receipt["gradient_l2"][name]) for receipt in receipts)
            for name in names
        },
        "maximum_gradient_l2": {
            name: max(float(receipt["gradient_l2"][name]) for receipt in receipts)
            for name in names
        },
        "target_gradient_tensor_count": sum(
            int(receipt["target_gradient_tensor_count"]) for receipt in receipts
        ),
    }


def run_fixed_training_v6(
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
    """Run the exact V3/V4 training loop and append branch-only evidence."""

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
        receipts.append(fine_branch_gradient_receipt_v6(candidate))
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
    activity = _summarize_branch_activity_v6(receipts)
    trace = tuple(
        {**row, "fine_rgb_branch": dict(receipt)}
        for row, receipt in zip(inherited_trace, receipts, strict=True)
    )
    diagnostics = {**inherited_diagnostics, "fine_rgb_branch": activity}
    return accounting, trace, diagnostics


__all__ = [
    "ACTION_ORDER",
    "FINE_RGB_BRANCH_ADDED_TRAINABLE_PARAMETER_COUNT_V6",
    "FINE_RGB_ONLINE_PARAMETER_NAMES_V6",
    "FINE_RGB_TARGET_PARAMETER_NAMES_V6",
    "MAXIMUM_PRESENTATIONS",
    "MAXIMUM_UPDATES",
    "MICROBATCHES_PER_UPDATE",
    "MICROBATCH_SIZE",
    "OCCUPIED_CLASS_INDEX",
    "OCCUPIED_SAFETY_AUX_COEFFICIENT",
    "OCCUPIED_SAFETY_AUX_NORMALIZATION",
    "PRESENTATIONS_PER_UPDATE",
    "build_frozen_optimizer_v1",
    "fine_branch_gradient_receipt_v6",
    "freeze_role_labels_v1",
    "partition_parameters_v1",
    "run_fixed_training_v6",
    "validate_pairs_against_labels_v1",
]
