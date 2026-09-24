#!/usr/bin/env python3
"""Lean training core for recurrent patch-memory temporal JEPA V1.

This module owns optimizer partitioning, exact presentation accounting, one
joint JEPA update, and checkpoint payload construction. Dataset access,
evaluation, publication, and lifecycle policy remain outside this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]

MICROBATCH_SIZE_V1 = 2
MICROBATCHES_PER_UPDATE_V1 = 5
SEQUENCES_PER_UPDATE_V1 = 10
LOGICAL_RGB_PRESENTATIONS_PER_UPDATE_V1 = 40
ONLINE_FRAME_ENCODINGS_PER_UPDATE_V1 = 30
EMA_TARGET_FRAME_ENCODINGS_PER_UPDATE_V1 = 10
MAXIMUM_UPDATES_V1 = 400
MAXIMUM_PRESENTATIONS_V1 = 16_000
ENCODER_LEARNING_RATE_V1 = 3.0e-5
PREDICTOR_LEARNING_RATE_V1 = 1.0e-4
MEMORY_LEARNING_RATE_V1 = 3.0e-4
WEIGHT_DECAY_V1 = 1.0e-4
GLOBAL_GRADIENT_CLIP_V1 = 1.0
EMA_MOMENTUM_V1 = 0.996

_PREDICTOR_EXACT_NAMES = ("predictor_position", "predictor_mask_token")
_PREDICTOR_PREFIXES = (
    "predictor_blocks.",
    "predictor_norm.",
    "predictor_output.",
)
_MEMORY_PREFIXES = ("action_embedding.", "time_embedding.", "temporal_gru.")


def _runtime_apis() -> tuple[Any, Any]:
    torch = importlib.import_module("torch")
    original_path = list(sys.path)
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        metrics = importlib.import_module(
            "lewm.benchmarks."
            "go2_rgb_recurrent_patch_memory_temporal_jepa_v1"
        )
    finally:
        sys.path[:] = original_path
    return torch, metrics


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class TemporalTrainingAccountingV1:
    updates: int = 0
    sequence_rows: int = 0
    logical_rgb_presentations: int = 0
    online_frame_encodings: int = 0
    ema_target_frame_encodings: int = 0
    microbatch_graphs: int = 0
    backward_calls: int = 0
    global_gradient_clips: int = 0
    optimizer_steps: int = 0
    ema_steps: int = 0


@dataclass(frozen=True)
class ParameterPartitionV1:
    encoder: tuple[Any, ...]
    predictor: tuple[Any, ...]
    memory: tuple[Any, ...]
    target: tuple[Any, ...]
    encoder_names: tuple[str, ...]
    predictor_names: tuple[str, ...]
    memory_names: tuple[str, ...]
    target_names: tuple[str, ...]

    @property
    def online(self) -> tuple[Any, ...]:
        return self.encoder + self.predictor + self.memory


@dataclass(frozen=True)
class TemporalUpdateResultV1:
    accounting: TemporalTrainingAccountingV1
    mean_jepa_loss: float
    microbatch_jepa_losses: tuple[float, ...]
    row_indices_sha256: str
    target_indices_sha256: str
    gradient_receipt: Mapping[str, Any]
    target_gradient_tensor_count: int
    optimizer_steps_this_update: int
    ema_steps_this_update: int


def accounting_for_completed_updates_v1(updates: int) -> TemporalTrainingAccountingV1:
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
        raise ValueError("update count must be a nonnegative integer")
    return TemporalTrainingAccountingV1(
        updates=updates,
        sequence_rows=updates * SEQUENCES_PER_UPDATE_V1,
        logical_rgb_presentations=(
            updates * LOGICAL_RGB_PRESENTATIONS_PER_UPDATE_V1
        ),
        online_frame_encodings=updates * ONLINE_FRAME_ENCODINGS_PER_UPDATE_V1,
        ema_target_frame_encodings=(
            updates * EMA_TARGET_FRAME_ENCODINGS_PER_UPDATE_V1
        ),
        microbatch_graphs=updates * MICROBATCHES_PER_UPDATE_V1,
        backward_calls=updates * MICROBATCHES_PER_UPDATE_V1,
        global_gradient_clips=updates,
        optimizer_steps=updates,
        ema_steps=updates,
    )


def validate_accounting_v1(accounting: TemporalTrainingAccountingV1) -> None:
    if not isinstance(accounting, TemporalTrainingAccountingV1):
        raise TypeError("temporal accounting has the wrong type")
    values = asdict(accounting).values()
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
        raise ValueError("temporal accounting values must be nonnegative integers")
    if accounting != accounting_for_completed_updates_v1(accounting.updates):
        raise RuntimeError("temporal accounting is inconsistent")
    if (
        accounting.updates > MAXIMUM_UPDATES_V1
        or accounting.logical_rgb_presentations > MAXIMUM_PRESENTATIONS_V1
    ):
        raise PermissionError("temporal training cap exceeded")


def partition_parameters_v1(model: Any) -> ParameterPartitionV1:
    torch, _ = _runtime_apis()
    groups: dict[str, list[Any]] = {
        "encoder": [],
        "predictor": [],
        "memory": [],
        "target": [],
    }
    names: dict[str, list[str]] = {key: [] for key in groups}
    for name, parameter in model.named_parameters():
        if name.startswith("encoder."):
            role = "encoder"
        elif name in _PREDICTOR_EXACT_NAMES or name.startswith(_PREDICTOR_PREFIXES):
            role = "predictor"
        elif name.startswith(_MEMORY_PREFIXES):
            role = "memory"
        elif name.startswith("target_encoder."):
            role = "target"
        else:
            raise RuntimeError(f"unregistered temporal model parameter {name!r}")
        groups[role].append(parameter)
        names[role].append(name)
    if any(not group for group in groups.values()):
        raise RuntimeError("temporal parameter partition contains an empty role")
    partitioned = tuple(p for group in groups.values() for p in group)
    all_parameters = tuple(model.parameters())
    if (
        len({id(p) for p in partitioned}) != len(partitioned)
        or {id(p) for p in partitioned} != {id(p) for p in all_parameters}
    ):
        raise RuntimeError("temporal parameter partition is incomplete or overlapping")
    for role in ("encoder", "predictor", "memory"):
        if any((not p.requires_grad) or p.dtype != torch.float32 for p in groups[role]):
            raise RuntimeError(f"{role} parameters must be trainable float32")
    if any(p.requires_grad or p.dtype != torch.float32 for p in groups["target"]):
        raise RuntimeError("target parameters must be frozen float32")
    if bool(model.target_encoder.training):
        raise RuntimeError("target encoder must stay in evaluation mode")
    return ParameterPartitionV1(
        encoder=tuple(groups["encoder"]),
        predictor=tuple(groups["predictor"]),
        memory=tuple(groups["memory"]),
        target=tuple(groups["target"]),
        encoder_names=tuple(names["encoder"]),
        predictor_names=tuple(names["predictor"]),
        memory_names=tuple(names["memory"]),
        target_names=tuple(names["target"]),
    )


def parameter_inventory_v1(model: Any) -> Mapping[str, Any]:
    partition = partition_parameters_v1(model)
    result: dict[str, Any] = {
        "schema": "lewm_temporal_patch_memory_parameter_inventory_v1",
        "target_optimizer_excluded": True,
    }
    for role in ("encoder", "predictor", "memory", "target"):
        parameters = getattr(partition, role)
        role_names = getattr(partition, f"{role}_names")
        result[f"{role}_tensor_count"] = len(parameters)
        result[f"{role}_parameter_count"] = sum(p.numel() for p in parameters)
        result[f"{role}_names_sha256"] = _canonical_sha256(role_names)
    return result


def build_optimizer_v1(model_or_partition: Any) -> Any:
    torch, _ = _runtime_apis()
    partition = (
        model_or_partition
        if isinstance(model_or_partition, ParameterPartitionV1)
        else partition_parameters_v1(model_or_partition)
    )
    optimizer = torch.optim.AdamW(
        [
            {"group_name": "encoder", "params": list(partition.encoder), "lr": ENCODER_LEARNING_RATE_V1},
            {"group_name": "predictor", "params": list(partition.predictor), "lr": PREDICTOR_LEARNING_RATE_V1},
            {"group_name": "memory", "params": list(partition.memory), "lr": MEMORY_LEARNING_RATE_V1},
        ],
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=WEIGHT_DECAY_V1,
        amsgrad=False,
    )
    validate_optimizer_v1(optimizer, partition)
    return optimizer


def validate_optimizer_v1(optimizer: Any, partition: ParameterPartitionV1) -> None:
    expected = (
        ("encoder", partition.encoder, ENCODER_LEARNING_RATE_V1),
        ("predictor", partition.predictor, PREDICTOR_LEARNING_RATE_V1),
        ("memory", partition.memory, MEMORY_LEARNING_RATE_V1),
    )
    if optimizer.__class__.__name__ != "AdamW" or len(optimizer.param_groups) != 3:
        raise RuntimeError("temporal optimizer must be three-group AdamW")
    observed_ids: list[int] = []
    for group, (name, parameters, lr) in zip(optimizer.param_groups, expected, strict=True):
        values = tuple(group["params"])
        observed_ids.extend(id(p) for p in values)
        if (
            group.get("group_name") != name
            or tuple(map(id, values)) != tuple(map(id, parameters))
            or float(group["lr"]) != lr
            or tuple(group["betas"]) != (0.9, 0.999)
            or float(group["eps"]) != 1.0e-8
            or float(group["weight_decay"]) != WEIGHT_DECAY_V1
            or bool(group["amsgrad"])
        ):
            raise RuntimeError(f"temporal optimizer group {name!r} changed")
    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != {
        id(p) for p in partition.online
    }:
        raise RuntimeError("temporal optimizer membership changed")


def _validate_microbatches(
    torch: Any,
    context_rgb: Sequence[Any],
    actions: Sequence[Any],
    future_rgb: Sequence[Any],
    row_indices: Sequence[Sequence[int]],
    expected_row_indices: Sequence[int],
) -> tuple[int, ...]:
    if not all(len(values) == MICROBATCHES_PER_UPDATE_V1 for values in (context_rgb, actions, future_rgb, row_indices)):
        raise ValueError("a temporal update requires exactly five microbatches")
    flattened: list[int] = []
    device = None
    for microbatch, (context, action, future, indices) in enumerate(
        zip(context_rgb, actions, future_rgb, row_indices, strict=True)
    ):
        if (
            not isinstance(context, torch.Tensor)
            or tuple(context.shape) != (2, 3, 3, 112, 112)
            or context.dtype != torch.float32
            or context.requires_grad
            or not bool(torch.isfinite(context).all())
        ):
            raise ValueError(f"context microbatch {microbatch} is invalid")
        if (
            not isinstance(future, torch.Tensor)
            or tuple(future.shape) != (2, 3, 112, 112)
            or future.dtype != torch.float32
            or future.requires_grad
            or not bool(torch.isfinite(future).all())
        ):
            raise ValueError(f"future microbatch {microbatch} is invalid")
        if (
            not isinstance(action, torch.Tensor)
            or tuple(action.shape) != (2, 3)
            or action.dtype != torch.long
            or action.requires_grad
            or bool((action < 0).any())
            or bool((action > 8).any())
        ):
            raise ValueError(f"action microbatch {microbatch} is invalid")
        if context.device != future.device or action.device != context.device:
            raise ValueError("all tensors in a temporal microbatch need one device")
        if device is None:
            device = context.device
        elif context.device != device:
            raise ValueError("all temporal microbatches need one device")
        if len(indices) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in indices):
            raise TypeError("each row-index microbatch must contain two integers")
        flattened.extend(indices)
    observed = tuple(flattened)
    if observed != tuple(expected_row_indices) or len(observed) != SEQUENCES_PER_UPDATE_V1:
        raise PermissionError("temporal rows left the frozen schedule slice")
    return observed


def _gradient_norm(torch: Any, parameters: Sequence[Any]) -> float:
    values = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    if not values:
        return 0.0
    result = float(torch.stack(values).sum().sqrt().detach().cpu())
    if not math.isfinite(result):
        raise FloatingPointError("gradient norm is nonfinite")
    return result


def _normalized_tokens_valid(torch: Any, value: Any) -> bool:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != (2, 64, 192)
        or value.dtype != torch.float32
        or not bool(torch.isfinite(value).all())
    ):
        return False
    norms = value.detach().norm(dim=-1)
    return bool(torch.allclose(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-4))


def training_update_v1(
    model: Any,
    optimizer: Any,
    context_rgb_microbatches: Sequence[Any],
    action_microbatches: Sequence[Any],
    future_rgb_microbatches: Sequence[Any],
    row_index_microbatches: Sequence[Sequence[int]],
    *,
    expected_row_indices: Sequence[int],
    schedule_offset: int,
    accounting: TemporalTrainingAccountingV1 | None = None,
) -> TemporalUpdateResultV1:
    """Accumulate five B2 future-JEPA graphs, then step once and update EMA."""

    torch, metrics = _runtime_apis()
    state = TemporalTrainingAccountingV1() if accounting is None else accounting
    validate_accounting_v1(state)
    if (
        isinstance(schedule_offset, bool)
        or not isinstance(schedule_offset, int)
        or schedule_offset != state.sequence_rows
    ):
        raise PermissionError(
            "temporal schedule offset disagrees with cumulative accounting"
        )
    if state.updates >= MAXIMUM_UPDATES_V1:
        raise PermissionError("temporal cap leaves no complete update")
    rows = _validate_microbatches(
        torch,
        context_rgb_microbatches,
        action_microbatches,
        future_rgb_microbatches,
        row_index_microbatches,
        expected_row_indices,
    )
    partition = partition_parameters_v1(model)
    validate_optimizer_v1(optimizer, partition)
    ema_before = int(model.ema_update_count.detach().cpu().item())
    if ema_before != state.ema_steps:
        raise RuntimeError("model EMA count disagrees with temporal accounting")
    if any(p.grad is not None for p in partition.target):
        raise RuntimeError("target encoder already has gradients")

    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    mask_rows: list[list[int]] = []
    for context, actions, future, indices in zip(
        context_rgb_microbatches,
        action_microbatches,
        future_rgb_microbatches,
        row_index_microbatches,
        strict=True,
    ):
        target_indices, _ = metrics.batched_mask_indices(
            "train", indices, device=context.device
        )
        mask_rows.extend(target_indices.detach().cpu().tolist())
        output = model(context, actions, future, target_indices)
        prediction = output.prediction.normalized_predicted_target_tokens
        target = output.target.normalized_target_tokens
        if not _normalized_tokens_valid(torch, prediction) or not _normalized_tokens_valid(torch, target):
            raise RuntimeError("temporal prediction/target normalization changed")
        if not prediction.requires_grad or target.requires_grad or target.grad_fn is not None:
            raise RuntimeError("future JEPA gradient boundary changed")
        registered_loss = 0.5 * (prediction - target).square().sum(dim=-1).mean()
        loss = output.loss
        if (
            loss.ndim != 0
            or not loss.requires_grad
            or not bool(torch.isfinite(loss))
            or not bool(torch.allclose(loss, registered_loss, rtol=1e-6, atol=1e-7))
        ):
            raise RuntimeError("sole future JEPA objective changed")
        (loss / MICROBATCHES_PER_UPDATE_V1).backward()
        if any(p.grad is not None for p in partition.target):
            raise RuntimeError("target encoder received a gradient")
        losses.append(float(loss.detach().cpu()))

    role_norms: dict[str, float] = {}
    role_nonzero_tensors: dict[str, int] = {}
    role_missing_tensors: dict[str, int] = {}
    for role in ("encoder", "predictor", "memory"):
        parameters = getattr(partition, role)
        missing = sum(p.grad is None for p in parameters)
        if missing:
            raise RuntimeError(f"{role} parameter missed the sole JEPA route")
        if any(not bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise FloatingPointError(f"{role} gradient became nonfinite")
        nonzero = sum(bool(torch.count_nonzero(p.grad).item()) for p in parameters)
        norm = _gradient_norm(torch, parameters)
        if nonzero < 1 or norm <= 0.0:
            raise RuntimeError(f"{role} JEPA gradient route is zero")
        role_norms[role] = norm
        role_nonzero_tensors[role] = nonzero
        role_missing_tensors[role] = missing

    before = float(
        torch.nn.utils.clip_grad_norm_(partition.online, GLOBAL_GRADIENT_CLIP_V1)
        .detach()
        .cpu()
    )
    after = _gradient_norm(torch, partition.online)
    if not math.isfinite(before) or before <= 0.0 or after > 1.0 + 1e-5:
        raise FloatingPointError("global gradient clip receipt is invalid")
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in partition.online):
        raise FloatingPointError("online parameter became nonfinite")
    model.update_target_ema()
    ema_after = int(model.ema_update_count.detach().cpu().item())
    if ema_after != ema_before + 1 or bool(model.target_encoder.training):
        raise RuntimeError("EMA target did not advance exactly once")
    target_gradient_count = sum(p.grad is not None for p in partition.target)
    if target_gradient_count:
        raise RuntimeError("target encoder received gradient tensors")

    advanced = accounting_for_completed_updates_v1(state.updates + 1)
    validate_accounting_v1(advanced)
    receipt = {
        "sole_future_jepa_route": True,
        "encoder_gradient_norm_before_clip": role_norms["encoder"],
        "predictor_gradient_norm_before_clip": role_norms["predictor"],
        "memory_gradient_norm_before_clip": role_norms["memory"],
        "encoder_nonzero_gradient_tensor_count": role_nonzero_tensors["encoder"],
        "predictor_nonzero_gradient_tensor_count": role_nonzero_tensors["predictor"],
        "memory_nonzero_gradient_tensor_count": role_nonzero_tensors["memory"],
        "encoder_missing_gradient_tensor_count": role_missing_tensors["encoder"],
        "predictor_missing_gradient_tensor_count": role_missing_tensors["predictor"],
        "memory_missing_gradient_tensor_count": role_missing_tensors["memory"],
        "global_gradient_norm_before_clip": before,
        "global_gradient_norm_after_clip": after,
        "global_gradient_clip": GLOBAL_GRADIENT_CLIP_V1,
        "all_gradient_receipts_finite": True,
    }
    return TemporalUpdateResultV1(
        accounting=advanced,
        mean_jepa_loss=sum(losses) / len(losses),
        microbatch_jepa_losses=tuple(losses),
        row_indices_sha256=_canonical_sha256(rows),
        target_indices_sha256=_canonical_sha256(mask_rows),
        gradient_receipt=receipt,
        target_gradient_tensor_count=target_gradient_count,
        optimizer_steps_this_update=1,
        ema_steps_this_update=1,
    )


def _clone_to_cpu(value: Any) -> Any:
    torch, _ = _runtime_apis()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    return value


def checkpoint_payload_v1(
    model: Any,
    optimizer: Any,
    accounting: TemporalTrainingAccountingV1,
) -> Mapping[str, Any]:
    validate_accounting_v1(accounting)
    if accounting.updates not in (200, 400):
        raise PermissionError(
            "complete temporal checkpoints are registered only at updates 200/400"
        )
    partition = partition_parameters_v1(model)
    validate_optimizer_v1(optimizer, partition)
    if int(model.ema_update_count.detach().cpu().item()) != accounting.ema_steps:
        raise RuntimeError("checkpoint EMA count disagrees with accounting")
    return {
        "schema": "lewm_go2_rgb_recurrent_patch_memory_temporal_jepa_v1_checkpoint_v1",
        "model_state_dict": _clone_to_cpu(model.state_dict()),
        "optimizer_state_dict": _clone_to_cpu(optimizer.state_dict()),
        "accounting": asdict(accounting),
        "parameter_inventory": parameter_inventory_v1(model),
        "training_contract": {
            "microbatch_size": MICROBATCH_SIZE_V1,
            "microbatches_per_update": MICROBATCHES_PER_UPDATE_V1,
            "sequences_per_update": SEQUENCES_PER_UPDATE_V1,
            "logical_rgb_presentations_per_update": LOGICAL_RGB_PRESENTATIONS_PER_UPDATE_V1,
            "maximum_updates": MAXIMUM_UPDATES_V1,
            "maximum_presentations": MAXIMUM_PRESENTATIONS_V1,
            "encoder_learning_rate": ENCODER_LEARNING_RATE_V1,
            "predictor_learning_rate": PREDICTOR_LEARNING_RATE_V1,
            "memory_learning_rate": MEMORY_LEARNING_RATE_V1,
            "weight_decay": WEIGHT_DECAY_V1,
            "global_gradient_clip": GLOBAL_GRADIENT_CLIP_V1,
            "ema_momentum": EMA_MOMENTUM_V1,
            "sole_objective": "normalized_half_squared_future_latent_jepa",
        },
    }
