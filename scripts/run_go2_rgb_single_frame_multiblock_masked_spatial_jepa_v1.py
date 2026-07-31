#!/usr/bin/env python3
"""Lean training core for the single-frame masked spatial JEPA V1.

Importing this module is source-only.  Torch and the benchmark mask contract
are imported only when a tensor-bearing helper is called.  Dataset loading,
observations, checkpoint publication, reservation, and lifecycle policy live
outside this module.
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

MICROBATCH_SIZE_V1 = 4
MICROBATCHES_PER_UPDATE_V1 = 4
PRESENTATIONS_PER_UPDATE_V1 = 16
MAXIMUM_UPDATES_V1 = 1_000
MAXIMUM_PRESENTATIONS_V1 = 16_000
TARGET_TOKEN_COUNT_V1 = 64
VISIBLE_TOKEN_COUNT_V1 = 192
SPATIAL_TOKEN_COUNT_V1 = 256
ENCODER_LEARNING_RATE_V1 = 1.0e-4
PREDICTOR_LEARNING_RATE_V1 = 3.0e-4
WEIGHT_DECAY_V1 = 1.0e-4
GLOBAL_GRADIENT_CLIP_V1 = 1.0
EMA_MOMENTUM_V1 = 0.996
TRAIN_MASK_ROLE_V1 = "train"

_PREDICTOR_EXACT_PARAMETER_NAMES_V1 = (
    "predictor_position",
    "predictor_mask_token",
)
_PREDICTOR_PARAMETER_PREFIXES_V1 = (
    "predictor_blocks.",
    "predictor_norm.",
    "predictor_output.",
)


def _runtime_apis() -> tuple[Any, Any]:
    """Import tensor and mask APIs only at the tensor-bearing boundary."""

    torch = importlib.import_module("torch")
    original_path = list(sys.path)
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        masks = importlib.import_module(
            "lewm.benchmarks."
            "go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1"
        )
    finally:
        sys.path[:] = original_path
    return torch, masks


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class MaskedSpatialTrainingAccountingV1:
    updates: int = 0
    presentations: int = 0
    mask_rows: int = 0
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
    target: tuple[Any, ...]
    encoder_names: tuple[str, ...]
    predictor_names: tuple[str, ...]
    target_names: tuple[str, ...]

    @property
    def online(self) -> tuple[Any, ...]:
        return self.encoder + self.predictor


@dataclass(frozen=True)
class MaskedSpatialUpdateResultV1:
    accounting: MaskedSpatialTrainingAccountingV1
    mean_jepa_loss: float
    microbatch_jepa_losses: tuple[float, ...]
    row_indices_sha256: str
    target_indices_sha256: str
    visible_indices_sha256: str
    gradient_receipt: Mapping[str, float | int | bool]
    target_gradient_tensor_count: int
    optimizer_steps_this_update: int
    ema_steps_this_update: int


def accounting_for_completed_updates_v1(
    updates: int,
) -> MaskedSpatialTrainingAccountingV1:
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
        raise ValueError("completed update count must be a nonnegative integer")
    return MaskedSpatialTrainingAccountingV1(
        updates=updates,
        presentations=updates * PRESENTATIONS_PER_UPDATE_V1,
        mask_rows=updates * PRESENTATIONS_PER_UPDATE_V1,
        online_frame_encodings=updates * PRESENTATIONS_PER_UPDATE_V1,
        ema_target_frame_encodings=updates * PRESENTATIONS_PER_UPDATE_V1,
        microbatch_graphs=updates * MICROBATCHES_PER_UPDATE_V1,
        backward_calls=updates * MICROBATCHES_PER_UPDATE_V1,
        global_gradient_clips=updates,
        optimizer_steps=updates,
        ema_steps=updates,
    )


def validate_accounting_v1(
    accounting: MaskedSpatialTrainingAccountingV1,
) -> None:
    if not isinstance(accounting, MaskedSpatialTrainingAccountingV1):
        raise TypeError("masked-spatial accounting has the wrong type")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in asdict(accounting).values()
    ):
        raise ValueError(
            "masked-spatial accounting values must be nonnegative integers"
        )
    expected = accounting_for_completed_updates_v1(accounting.updates)
    if accounting != expected:
        raise RuntimeError("masked-spatial accounting is inconsistent")
    if (
        accounting.updates > MAXIMUM_UPDATES_V1
        or accounting.presentations > MAXIMUM_PRESENTATIONS_V1
    ):
        raise PermissionError("masked-spatial accounting exceeds the frozen cap")


def _validate_capacity_v1(
    accounting: MaskedSpatialTrainingAccountingV1,
) -> None:
    validate_accounting_v1(accounting)
    if (
        accounting.updates >= MAXIMUM_UPDATES_V1
        or accounting.presentations + PRESENTATIONS_PER_UPDATE_V1
        > MAXIMUM_PRESENTATIONS_V1
    ):
        raise PermissionError(
            "masked-spatial training cap leaves no complete update"
        )


def _advance_accounting_v1(
    accounting: MaskedSpatialTrainingAccountingV1,
) -> MaskedSpatialTrainingAccountingV1:
    advanced = accounting_for_completed_updates_v1(accounting.updates + 1)
    validate_accounting_v1(advanced)
    return advanced


def partition_parameters_v1(model: Any) -> ParameterPartitionV1:
    """Bind every model parameter to one and only one scientific role."""

    torch, _ = _runtime_apis()
    groups: dict[str, list[Any]] = {
        "encoder": [],
        "predictor": [],
        "target": [],
    }
    names: dict[str, list[str]] = {name: [] for name in groups}
    for name, parameter in model.named_parameters():
        if name.startswith("encoder."):
            group = "encoder"
        elif name in _PREDICTOR_EXACT_PARAMETER_NAMES_V1 or name.startswith(
            _PREDICTOR_PARAMETER_PREFIXES_V1
        ):
            group = "predictor"
        elif name.startswith("target_encoder."):
            group = "target"
        else:
            raise RuntimeError(
                f"unregistered masked-spatial model parameter {name!r}"
            )
        groups[group].append(parameter)
        names[group].append(name)
    if any(not values for values in groups.values()):
        raise RuntimeError("masked-spatial parameter partition has an empty role")
    all_parameters = tuple(model.parameters())
    partitioned = tuple(
        parameter for values in groups.values() for parameter in values
    )
    if (
        len({id(value) for value in partitioned}) != len(partitioned)
        or {id(value) for value in partitioned}
        != {id(value) for value in all_parameters}
    ):
        raise RuntimeError(
            "masked-spatial parameter partition is incomplete or overlapping"
        )
    if any(parameter.requires_grad for parameter in groups["target"]):
        raise RuntimeError("EMA target parameters must remain frozen")
    if any(
        not parameter.requires_grad or parameter.dtype != torch.float32
        for group in ("encoder", "predictor")
        for parameter in groups[group]
    ):
        raise RuntimeError(
            "all masked-spatial online parameters must be trainable float32"
        )
    if any(parameter.dtype != torch.float32 for parameter in groups["target"]):
        raise RuntimeError("all EMA target parameters must be float32")
    target_encoder = getattr(model, "target_encoder", None)
    if target_encoder is None or bool(target_encoder.training):
        raise RuntimeError("EMA target encoder must remain in evaluation mode")
    return ParameterPartitionV1(
        encoder=tuple(groups["encoder"]),
        predictor=tuple(groups["predictor"]),
        target=tuple(groups["target"]),
        encoder_names=tuple(names["encoder"]),
        predictor_names=tuple(names["predictor"]),
        target_names=tuple(names["target"]),
    )


def parameter_inventory_v1(model: Any) -> Mapping[str, Any]:
    partition = partition_parameters_v1(model)
    return {
        "schema": "lewm_masked_spatial_jepa_v1_parameter_inventory_v1",
        "encoder_tensor_count": len(partition.encoder),
        "encoder_parameter_count": sum(
            parameter.numel() for parameter in partition.encoder
        ),
        "predictor_tensor_count": len(partition.predictor),
        "predictor_parameter_count": sum(
            parameter.numel() for parameter in partition.predictor
        ),
        "target_tensor_count": len(partition.target),
        "target_parameter_count": sum(
            parameter.numel() for parameter in partition.target
        ),
        "encoder_names_sha256": _canonical_json_sha256(
            partition.encoder_names
        ),
        "predictor_names_sha256": _canonical_json_sha256(
            partition.predictor_names
        ),
        "target_names_sha256": _canonical_json_sha256(partition.target_names),
        "target_optimizer_excluded": True,
    }


def build_optimizer_v1(model_or_partition: Any) -> Any:
    torch, _ = _runtime_apis()
    partition = (
        model_or_partition
        if isinstance(model_or_partition, ParameterPartitionV1)
        else partition_parameters_v1(model_or_partition)
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "group_name": "encoder",
                "params": list(partition.encoder),
                "lr": ENCODER_LEARNING_RATE_V1,
            },
            {
                "group_name": "predictor",
                "params": list(partition.predictor),
                "lr": PREDICTOR_LEARNING_RATE_V1,
            },
        ],
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=WEIGHT_DECAY_V1,
        amsgrad=False,
    )
    validate_optimizer_v1(optimizer, partition)
    return optimizer


def validate_optimizer_v1(
    optimizer: Any,
    partition: ParameterPartitionV1,
) -> None:
    expected = (
        ("encoder", partition.encoder, ENCODER_LEARNING_RATE_V1),
        ("predictor", partition.predictor, PREDICTOR_LEARNING_RATE_V1),
    )
    if optimizer.__class__.__name__ != "AdamW":
        raise RuntimeError("masked-spatial optimizer must be AdamW")
    if len(optimizer.param_groups) != len(expected):
        raise RuntimeError(
            "masked-spatial optimizer must contain exactly two groups"
        )
    observed_ids: list[int] = []
    for observed, (name, parameters, learning_rate) in zip(
        optimizer.param_groups, expected, strict=True
    ):
        values = tuple(observed["params"])
        observed_ids.extend(id(value) for value in values)
        if (
            observed.get("group_name") != name
            or tuple(map(id, values)) != tuple(map(id, parameters))
            or float(observed["lr"]) != learning_rate
            or tuple(observed["betas"]) != (0.9, 0.999)
            or float(observed["eps"]) != 1.0e-8
            or float(observed["weight_decay"]) != WEIGHT_DECAY_V1
            or bool(observed["amsgrad"])
        ):
            raise RuntimeError(
                f"masked-spatial optimizer group {name!r} changed"
            )
    if (
        len(observed_ids) != len(set(observed_ids))
        or set(observed_ids) != {id(value) for value in partition.online}
    ):
        raise RuntimeError(
            "masked-spatial optimizer membership is incomplete or overlapping"
        )


def _validate_rgb_microbatches_v1(
    torch: Any,
    rgb_microbatches: Sequence[Any],
) -> None:
    if len(rgb_microbatches) != MICROBATCHES_PER_UPDATE_V1:
        raise ValueError(
            "masked-spatial update requires exactly four RGB microbatches"
        )
    device = None
    for index, rgb in enumerate(rgb_microbatches):
        if (
            not isinstance(rgb, torch.Tensor)
            or tuple(rgb.shape) != (MICROBATCH_SIZE_V1, 3, 112, 112)
            or rgb.dtype != torch.float32
            or rgb.requires_grad
            or not bool(torch.isfinite(rgb).all())
        ):
            raise ValueError(
                f"RGB microbatch {index} must be finite non-gradient float32 "
                "with shape (4,3,112,112)"
            )
        if device is None:
            device = rgb.device
        elif rgb.device != device:
            raise ValueError("all masked-spatial RGB microbatches need one device")


def _validate_row_index_microbatches_v1(
    row_index_microbatches: Sequence[Sequence[int]],
    accounting: MaskedSpatialTrainingAccountingV1,
) -> tuple[int, ...]:
    if len(row_index_microbatches) != MICROBATCHES_PER_UPDATE_V1:
        raise ValueError(
            "masked-spatial update requires exactly four row-index microbatches"
        )
    flattened: list[int] = []
    for microbatch, values in enumerate(row_index_microbatches):
        if len(values) != MICROBATCH_SIZE_V1:
            raise ValueError(
                f"row-index microbatch {microbatch} must contain exactly four rows"
            )
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("masked-spatial row indices must be exact integers")
            flattened.append(value)
    expected = tuple(
        range(
            accounting.presentations,
            accounting.presentations + PRESENTATIONS_PER_UPDATE_V1,
        )
    )
    observed = tuple(flattened)
    if observed != expected:
        raise PermissionError(
            "masked-spatial row indices left the frozen corrected-H6 order"
        )
    return observed


def _mask_indices_for_microbatch_v1(
    torch: Any,
    masks: Any,
    row_indices: Sequence[int],
    device: Any,
) -> tuple[Any, tuple[tuple[int, ...], ...]]:
    target_rows: list[tuple[int, ...]] = []
    visible_rows: list[tuple[int, ...]] = []
    universe = set(range(SPATIAL_TOKEN_COUNT_V1))
    for row_index in row_indices:
        target, visible = masks.mask_indices(TRAIN_MASK_ROLE_V1, row_index)
        target = tuple(target)
        visible = tuple(visible)
        if (
            len(target) != TARGET_TOKEN_COUNT_V1
            or target != tuple(sorted(target))
            or len(set(target)) != TARGET_TOKEN_COUNT_V1
            or any(value < 0 or value >= SPATIAL_TOKEN_COUNT_V1 for value in target)
        ):
            raise RuntimeError("benchmark returned invalid target mask indices")
        if (
            len(visible) != VISIBLE_TOKEN_COUNT_V1
            or visible != tuple(sorted(visible))
            or len(set(visible)) != VISIBLE_TOKEN_COUNT_V1
            or set(visible) != universe - set(target)
        ):
            raise RuntimeError("benchmark returned invalid visible mask indices")
        target_rows.append(target)
        visible_rows.append(visible)
    target_tensor = torch.tensor(
        target_rows,
        dtype=torch.long,
        device=device,
    )
    if tuple(target_tensor.shape) != (
        MICROBATCH_SIZE_V1,
        TARGET_TOKEN_COUNT_V1,
    ):
        raise RuntimeError("batched target mask shape changed")
    return target_tensor, tuple(visible_rows)


def _validate_normalized_tokens_v1(
    torch: Any,
    value: Any,
    *,
    name: str,
) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 3
        or tuple(value.shape[:2])
        != (MICROBATCH_SIZE_V1, TARGET_TOKEN_COUNT_V1)
        or value.dtype != torch.float32
        or not bool(torch.isfinite(value).all())
    ):
        raise RuntimeError(
            f"{name} must be finite float32 with shape (4,64,D)"
        )
    norms = value.detach().float().norm(dim=-1)
    if not bool(
        torch.allclose(
            norms,
            torch.ones_like(norms),
            rtol=1.0e-4,
            atol=1.0e-4,
        )
    ):
        raise RuntimeError(f"{name} stopped being L2 normalized")


def _gradient_norm_v1(torch: Any, parameters: Sequence[Any]) -> float:
    contributions = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not contributions:
        return 0.0
    value = float(torch.stack(contributions).sum().sqrt().detach().cpu())
    if not math.isfinite(value):
        raise FloatingPointError("masked-spatial gradient norm is nonfinite")
    return value


def _ema_update_count_v1(model: Any) -> int:
    value = getattr(model, "ema_update_count", None)
    if value is None or value.numel() != 1:
        raise RuntimeError("masked-spatial model lacks scalar EMA accounting")
    count = int(value.detach().cpu().item())
    if count < 0:
        raise RuntimeError("masked-spatial EMA accounting is negative")
    return count


def training_update_v1(
    model: Any,
    optimizer: Any,
    rgb_microbatches: Sequence[Any],
    row_index_microbatches: Sequence[Sequence[int]],
    *,
    accounting: MaskedSpatialTrainingAccountingV1 | None = None,
) -> MaskedSpatialUpdateResultV1:
    """Accumulate four same-image JEPA graphs, then step optimizer and EMA."""

    torch, masks = _runtime_apis()
    state = (
        MaskedSpatialTrainingAccountingV1()
        if accounting is None
        else accounting
    )
    _validate_capacity_v1(state)
    _validate_rgb_microbatches_v1(torch, rgb_microbatches)
    row_indices = _validate_row_index_microbatches_v1(
        row_index_microbatches,
        state,
    )
    partition = partition_parameters_v1(model)
    validate_optimizer_v1(optimizer, partition)
    ema_before = _ema_update_count_v1(model)
    if ema_before != state.ema_steps:
        raise RuntimeError(
            "model EMA count disagrees with masked-spatial accounting"
        )
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("EMA target already has a gradient tensor")

    optimizer.zero_grad(set_to_none=True)
    microbatch_losses: list[float] = []
    all_target_rows: list[tuple[int, ...]] = []
    all_visible_rows: list[tuple[int, ...]] = []
    for rgb, indices in zip(
        rgb_microbatches,
        row_index_microbatches,
        strict=True,
    ):
        target_indices, visible_rows = _mask_indices_for_microbatch_v1(
            torch,
            masks,
            indices,
            rgb.device,
        )
        all_target_rows.extend(
            tuple(int(value) for value in row)
            for row in target_indices.detach().cpu().tolist()
        )
        all_visible_rows.extend(visible_rows)
        output = model(rgb, target_indices)
        prediction = output.prediction.normalized_predicted_target_tokens
        target_value = output.target.normalized_target_tokens
        expected_visible = torch.tensor(
            visible_rows,
            dtype=torch.long,
            device=rgb.device,
        )
        if (
            not torch.equal(output.prediction.target_indices, target_indices)
            or not torch.equal(output.target.target_indices, target_indices)
            or not torch.equal(
                output.prediction.visible_indices,
                expected_visible,
            )
        ):
            raise RuntimeError(
                "model-visible mask indices differ from the benchmark schedule"
            )
        _validate_normalized_tokens_v1(
            torch,
            prediction,
            name="normalized online prediction",
        )
        _validate_normalized_tokens_v1(
            torch,
            target_value,
            name="normalized EMA target",
        )
        if prediction.shape != target_value.shape:
            raise RuntimeError("online prediction and EMA target shapes differ")
        if not prediction.requires_grad:
            raise RuntimeError("masked-spatial prediction lost its gradient graph")
        if target_value.requires_grad or target_value.grad_fn is not None:
            raise RuntimeError("EMA target retained an autograd graph")
        loss = output.loss
        registered_loss = (
            0.5
            * (prediction - target_value).square().sum(dim=-1).mean()
        )
        if not bool(
            torch.allclose(
                loss,
                registered_loss,
                rtol=1.0e-6,
                atol=1.0e-7,
            )
        ):
            raise RuntimeError("masked-spatial sole JEPA objective changed")
        if (
            loss.ndim != 0
            or not loss.requires_grad
            or not bool(torch.isfinite(loss).item())
        ):
            raise FloatingPointError(
                "masked-spatial sole JEPA objective became invalid"
            )
        (loss / MICROBATCHES_PER_UPDATE_V1).backward()
        if any(parameter.grad is not None for parameter in partition.target):
            raise RuntimeError("EMA target received a gradient tensor")
        microbatch_losses.append(float(loss.detach().cpu()))

    if len(microbatch_losses) != MICROBATCHES_PER_UPDATE_V1:
        raise RuntimeError("masked-spatial microbatch accounting changed")
    missing_encoder = sum(
        parameter.grad is None for parameter in partition.encoder
    )
    missing_predictor = sum(
        parameter.grad is None for parameter in partition.predictor
    )
    if missing_encoder or missing_predictor:
        raise RuntimeError(
            "masked-spatial online parameter missed the sole JEPA route"
        )
    for parameter in partition.online:
        if not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError(
                "masked-spatial online gradient became nonfinite"
            )
    encoder_norm = _gradient_norm_v1(torch, partition.encoder)
    predictor_norm = _gradient_norm_v1(torch, partition.predictor)
    if encoder_norm <= 0.0 or predictor_norm <= 0.0:
        raise RuntimeError("masked-spatial sole JEPA gradient route is zero")
    global_before_tensor = torch.nn.utils.clip_grad_norm_(
        partition.online,
        max_norm=GLOBAL_GRADIENT_CLIP_V1,
    )
    global_before = float(global_before_tensor.detach().cpu())
    global_after = _gradient_norm_v1(torch, partition.online)
    if (
        not math.isfinite(global_before)
        or global_before <= 0.0
        or global_after > GLOBAL_GRADIENT_CLIP_V1 + 1.0e-5
    ):
        raise FloatingPointError(
            "masked-spatial global gradient clipping receipt is invalid"
        )

    optimizer.step()
    for parameter in partition.online:
        if not bool(torch.isfinite(parameter).all()):
            raise FloatingPointError(
                "masked-spatial online parameter became nonfinite"
            )
    model.update_target_ema()
    ema_after = _ema_update_count_v1(model)
    if ema_after != ema_before + 1:
        raise RuntimeError("masked-spatial EMA did not advance exactly once")
    if bool(model.target_encoder.training):
        raise RuntimeError("masked-spatial EMA target left evaluation mode")
    target_gradient_count = sum(
        parameter.grad is not None for parameter in partition.target
    )
    if target_gradient_count:
        raise RuntimeError("masked-spatial EMA target received a gradient tensor")
    for parameter in partition.target:
        if not bool(torch.isfinite(parameter).all()):
            raise FloatingPointError(
                "masked-spatial EMA target parameter became nonfinite"
            )

    advanced = _advance_accounting_v1(state)
    if advanced.ema_steps != ema_after:
        raise RuntimeError(
            "post-update EMA count disagrees with masked-spatial accounting"
        )
    mean_loss = sum(microbatch_losses) / MICROBATCHES_PER_UPDATE_V1
    gradient_receipt: Mapping[str, float | int | bool] = {
        "sole_jepa_route": True,
        "encoder_gradient_tensor_count": len(partition.encoder),
        "predictor_gradient_tensor_count": len(partition.predictor),
        "missing_encoder_gradient_tensor_count": missing_encoder,
        "missing_predictor_gradient_tensor_count": missing_predictor,
        "encoder_gradient_norm_before_global_clip": encoder_norm,
        "predictor_gradient_norm_before_global_clip": predictor_norm,
        "global_gradient_norm_before_clip": global_before,
        "global_gradient_norm_after_clip": global_after,
        "global_gradient_clip": GLOBAL_GRADIENT_CLIP_V1,
        "all_gradient_receipts_finite": all(
            math.isfinite(value)
            for value in (
                encoder_norm,
                predictor_norm,
                global_before,
                global_after,
                mean_loss,
            )
        ),
    }
    return MaskedSpatialUpdateResultV1(
        accounting=advanced,
        mean_jepa_loss=mean_loss,
        microbatch_jepa_losses=tuple(microbatch_losses),
        row_indices_sha256=_canonical_json_sha256(row_indices),
        target_indices_sha256=_canonical_json_sha256(all_target_rows),
        visible_indices_sha256=_canonical_json_sha256(all_visible_rows),
        gradient_receipt=gradient_receipt,
        target_gradient_tensor_count=target_gradient_count,
        optimizer_steps_this_update=1,
        ema_steps_this_update=1,
    )


def model_state_inventory_v1(model: Any) -> Mapping[str, Any]:
    torch, _ = _runtime_apis()
    parameter_inventory = parameter_inventory_v1(model)
    state = model.state_dict()
    if not state:
        raise RuntimeError("masked-spatial model state is empty")
    if any(
        not isinstance(value, torch.Tensor)
        or (
            value.is_floating_point()
            and not bool(torch.isfinite(value).all())
        )
        for value in state.values()
    ):
        raise FloatingPointError("masked-spatial model state is invalid")
    return {
        **parameter_inventory,
        "state_tensor_count": len(state),
        "state_names_sha256": _canonical_json_sha256(tuple(state)),
        "ema_update_count": _ema_update_count_v1(model),
    }


def _clone_checkpoint_value_to_cpu_v1(value: Any) -> Any:
    torch, _ = _runtime_apis()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {
            key: _clone_checkpoint_value_to_cpu_v1(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_clone_checkpoint_value_to_cpu_v1(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_checkpoint_value_to_cpu_v1(item) for item in value)
    return value


def checkpoint_payload_v1(
    model: Any,
    optimizer: Any,
    accounting: MaskedSpatialTrainingAccountingV1,
) -> Mapping[str, Any]:
    """Build a complete immutable CPU payload; publication stays elsewhere."""

    validate_accounting_v1(accounting)
    partition = partition_parameters_v1(model)
    validate_optimizer_v1(optimizer, partition)
    if accounting.ema_steps != _ema_update_count_v1(model):
        raise RuntimeError(
            "checkpoint EMA count disagrees with masked-spatial accounting"
        )
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("cannot checkpoint a target with gradient tensors")
    return {
        "schema": "lewm_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1_checkpoint_v1",
        "model_state_dict": _clone_checkpoint_value_to_cpu_v1(
            model.state_dict()
        ),
        "optimizer_state_dict": _clone_checkpoint_value_to_cpu_v1(
            optimizer.state_dict()
        ),
        "accounting": asdict(accounting),
        "model_state_inventory": model_state_inventory_v1(model),
        "training_contract": {
            "microbatch_size": MICROBATCH_SIZE_V1,
            "microbatches_per_update": MICROBATCHES_PER_UPDATE_V1,
            "presentations_per_update": PRESENTATIONS_PER_UPDATE_V1,
            "maximum_updates": MAXIMUM_UPDATES_V1,
            "maximum_presentations": MAXIMUM_PRESENTATIONS_V1,
            "encoder_learning_rate": ENCODER_LEARNING_RATE_V1,
            "predictor_learning_rate": PREDICTOR_LEARNING_RATE_V1,
            "weight_decay": WEIGHT_DECAY_V1,
            "global_gradient_clip": GLOBAL_GRADIENT_CLIP_V1,
            "ema_momentum": EMA_MOMENTUM_V1,
            "mask_role": TRAIN_MASK_ROLE_V1,
            "sole_objective": "normalized_half_squared_masked_spatial_jepa",
        },
    }
