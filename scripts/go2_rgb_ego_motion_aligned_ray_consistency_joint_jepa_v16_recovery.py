#!/usr/bin/env python3
"""Narrow full-state recovery seam for the preregistered V16 joint JEPA.

This module performs no discovery, data access, accelerator selection, or
execution.  The reviewed caller supplies the live Torch API, exact V13
parameter partition, source identity, schedule, controller state, and custody
ledger.  Only passing update-400 and eligible update-1,000 milestones are
representable.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import io
import json
from typing import Any, Callable, Mapping, Sequence

from scripts import (
    run_go2_rgb_swept_progress_survival_joint_jepa_v13_camera_evidence_bottleneck
    as v13_training,
)


SCHEMA_PREFIX_V16 = (
    "lewm_go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16_"
    "integrity_replacement_v1"
)
PREREGISTRATION_COMMIT_V16 = (
    "2ac4b08d94ee249ae42194b3c737190d39fd2396"
)
RECOVERY_UPDATES_V16 = (400, 1_000)
PRESENTATIONS_PER_UPDATE_V16 = 16
MAXIMUM_PRESENTATIONS_V16 = 16_000
TRAIN_PAIR_COUNT_V16 = 4_262

PAYLOAD_SCHEMA_V16 = f"{SCHEMA_PREFIX_V16}_full_state_recovery_payload_v1"
BINDING_SCHEMA_V16 = f"{SCHEMA_PREFIX_V16}_full_state_recovery_binding_v1"

SOURCE_IDENTITY_KEYS_V16 = {
    "preregistration_commit",
    "frozen_source_and_review_commit",
    "recursive_source_closure_manifest_sha256",
    "execution_binding_commit",
    "authority_sha256",
    "attempt_identity",
}
CONTROLLER_STATE_KEYS_V16 = {"trace", "metric_bindings"}
DOWNSTREAM_DENIALS_V16 = {
    "promotable_model": False,
    "probability_calibration_authorized": False,
    "g2_authorized": False,
    "navigation_authorized": False,
    "held_out_authorized": False,
    "production_authorized": False,
    "deployment_authorized": False,
    "automatic_retry_authorized": False,
    "automatic_extension_authorized": False,
}

_PAYLOAD_KEYS_V16 = {
    "schema",
    "update",
    "next_update",
    "presentation_cursor",
    "metadata",
    "metadata_sha256",
    "model_state_dict",
    "optimizer_state_dict",
    "accounting",
    "torch_cpu_rng_state",
    "torch_cuda_rng_states",
    "controller_state",
    "consumed_input_ledger",
    "access_receipt",
}
_BINDING_KEYS_V16 = {
    "schema",
    "update",
    "next_update",
    "presentation_cursor",
    "payload",
    "metadata_sha256",
    "source_identity_sha256",
    "schedule_full_sha256",
    "consumed_prefix_sha256",
    "recovery_only",
    "content_sha256",
}
_METADATA_KEYS_V16 = {
    "schema",
    "update",
    "next_update",
    "presentation_cursor",
    "presentations_per_update",
    "schedule",
    "authority",
    "authority_sha256",
    "source_identity",
    "source_identity_sha256",
    "model_config",
    "model_state_manifest",
    "model_state_manifest_sha256",
    "state_key_count",
    "optimizer_state_sha256",
    "optimizer_group_names",
    "optimizer_step",
    "ema_update_count",
    "target_hard_sync_count",
    "controller_state_sha256",
    "trace_row_count",
    "metric_binding_count",
    "consumed_input_ledger_sha256",
    "access_receipt_sha256",
    "torch_cpu_rng_state_sha256",
    "torch_cuda_rng_states_sha256",
    "torch_cuda_rng_state_count",
    "development_only",
    "recovery_only",
    "downstream_authority",
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_hex(value: object, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _content_bound(core: Mapping[str, Any]) -> dict[str, Any]:
    if type(core) is not dict or "content_sha256" in core:
        raise ValueError("V16 binding core must be a plain unbound dict")
    value = dict(core)
    value["content_sha256"] = _canonical_sha256(core)
    return value


def _validate_content_bound(value: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _BINDING_KEYS_V16:
        raise PermissionError("V16 recovery binding schema changed")
    core = dict(value)
    declared = core.pop("content_sha256")
    if declared != _canonical_sha256(core):
        raise PermissionError("V16 recovery binding content hash changed")
    return dict(value)


def _json_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be a plain dict")
    try:
        normalized = json.loads(_canonical_json_bytes(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not finite canonical JSON") from error
    if type(normalized) is not dict:
        raise TypeError(f"{name} normalization changed its mapping type")
    return normalized


def _tensor_raw(value: Any) -> bytes:
    tensor = value.detach().to(device="cpu").contiguous()
    try:
        return tensor.numpy().tobytes(order="C")
    except TypeError as error:
        raise TypeError(f"unsupported V16 checkpoint tensor dtype: {tensor.dtype}") from error


def _tensor_receipt(value: Any) -> dict[str, Any]:
    tensor = value.detach().to(device="cpu").contiguous()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "numel": int(tensor.numel()),
        "tensor_sha256": hashlib.sha256(_tensor_raw(tensor)).hexdigest(),
    }


def _clone_to_cpu(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "shape"):
        return value.detach().to(device="cpu").contiguous().clone()
    if isinstance(value, Mapping):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    if value is None or type(value) in {bool, int, float, str}:
        return value
    raise TypeError(f"unsupported V16 checkpoint value: {type(value)!r}")


def _nested_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def add(item: Any, path: str) -> None:
        digest.update(path.encode("utf-8"))
        if hasattr(item, "detach") and hasattr(item, "shape"):
            receipt = _tensor_receipt(item)
            digest.update(b"tensor")
            digest.update(_canonical_json_bytes(receipt))
            return
        if isinstance(item, Mapping):
            digest.update(b"mapping")
            for key in sorted(item, key=repr):
                add(item[key], f"{path}/{key!r}")
            return
        if isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode("ascii"))
            for index, child in enumerate(item):
                add(child, f"{path}/{index}")
            return
        if item is None or type(item) in {bool, int, float, str}:
            if isinstance(item, float) and not (float("-inf") < item < float("inf")):
                raise ValueError("nonfinite scalar in V16 checkpoint identity")
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(repr(item).encode("utf-8"))
            return
        raise TypeError(f"unsupported V16 identity value: {type(item)!r}")

    add(value, "root")
    return digest.hexdigest()


def _normalized_model_state(model: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observed = model.state_dict()
    if not isinstance(observed, Mapping) or not observed:
        raise ValueError("V16 model state is empty or not a mapping")
    state = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in sorted(observed.items())
    }
    if any(type(name) is not str for name in state):
        raise TypeError("V16 model state keys must be strings")
    manifest = [{"name": name, **_tensor_receipt(value)} for name, value in state.items()]
    return state, manifest


def _validate_update(update: int) -> None:
    if type(update) is not int or update not in RECOVERY_UPDATES_V16:
        raise PermissionError("V16 recovery update must be exactly 400 or 1000")


def _schedule_identity(
    schedule: Sequence[int],
    schedule_receipt: Mapping[str, Any],
    update: int,
) -> dict[str, Any]:
    _validate_update(update)
    if (
        isinstance(schedule, (str, bytes))
        or len(schedule) != MAXIMUM_PRESENTATIONS_V16
    ):
        raise PermissionError("V16 recovery schedule must contain exactly 16000 indices")
    values = list(schedule)
    if any(
        type(index) is not int or not 0 <= index < TRAIN_PAIR_COUNT_V16
        for index in values
    ):
        raise PermissionError("V16 recovery schedule contains an invalid index")
    receipt = _json_mapping(schedule_receipt, name="V16 schedule receipt")
    if not receipt:
        raise PermissionError("V16 schedule receipt is empty")
    if "presentation_count" in receipt and (
        receipt["presentation_count"] != MAXIMUM_PRESENTATIONS_V16
    ):
        raise PermissionError("V16 schedule receipt presentation count changed")
    cursor = update * PRESENTATIONS_PER_UPDATE_V16
    return {
        "presentation_count": len(values),
        "train_pair_count": TRAIN_PAIR_COUNT_V16,
        "full_schedule_sha256": _canonical_sha256(values),
        "consumed_prefix_sha256": _canonical_sha256(values[:cursor]),
        "schedule_receipt": receipt,
        "schedule_receipt_sha256": _canonical_sha256(receipt),
    }


def _source_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _json_mapping(value, name="V16 source identity")
    if set(result) != SOURCE_IDENTITY_KEYS_V16:
        raise PermissionError("V16 source identity fields changed")
    if result["preregistration_commit"] != PREREGISTRATION_COMMIT_V16:
        raise PermissionError("V16 preregistration commit changed")
    if not all(
        _is_hex(result[name], 40)
        for name in (
            "preregistration_commit",
            "frozen_source_and_review_commit",
            "execution_binding_commit",
        )
    ) or not all(
        _is_hex(result[name], 64)
        for name in (
            "recursive_source_closure_manifest_sha256",
            "authority_sha256",
        )
    ):
        raise PermissionError("V16 source identity has a malformed hash")
    if type(result["attempt_identity"]) is not str or not result["attempt_identity"]:
        raise PermissionError("V16 attempt identity is absent")
    return result


def _authority_identity(
    authority: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = _json_mapping(authority, name="V16 authority")
    required = {
        "preregistration_commit",
        "frozen_source_and_review_commit",
        "recursive_source_closure_manifest_sha256",
        "execution_binding_commit",
    }
    if not required <= set(value):
        raise PermissionError("V16 authority lacks source-binding fields")
    attempt = value.get("attempt_identity", value.get("output_root"))
    source = _source_identity(
        {
            "preregistration_commit": value["preregistration_commit"],
            "frozen_source_and_review_commit": value[
                "frozen_source_and_review_commit"
            ],
            "recursive_source_closure_manifest_sha256": value[
                "recursive_source_closure_manifest_sha256"
            ],
            "execution_binding_commit": value["execution_binding_commit"],
            "authority_sha256": _canonical_sha256(value),
            "attempt_identity": attempt,
        }
    )
    return value, source


def _model_config(model: Any) -> dict[str, Any]:
    value = getattr(model, "config", None)
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    elif hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    if type(value) is not dict:
        raise PermissionError("V16 model config is not an exact mapping")
    return _json_mapping(value, name="V16 model config")


def _json_sequence(value: Sequence[Any], *, name: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    try:
        normalized = json.loads(_canonical_json_bytes(list(value)))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not finite canonical JSON") from error
    if type(normalized) is not list:
        raise TypeError(f"{name} normalization changed its sequence type")
    return normalized


def _controller_state(
    trace: Sequence[Mapping[str, Any]],
    metric_bindings: Sequence[Mapping[str, Any]],
    *,
    update: int,
) -> dict[str, Any]:
    trace_value = _json_sequence(trace, name="V16 controller trace")
    metric_value = _json_sequence(
        metric_bindings, name="V16 controller metric bindings"
    )
    if (
        not trace_value
        or type(trace_value[-1]) is not dict
        or trace_value[-1].get("update") != update
        or not metric_value
        or any(type(binding) is not dict for binding in metric_value)
    ):
        raise PermissionError("V16 controller trace or metric bindings changed")
    return {"trace": trace_value, "metric_bindings": metric_value}


def _accounting(value: Any, *, update: int) -> tuple[Any, dict[str, int]]:
    if isinstance(value, v13_training.JointTrainingAccountingV13):
        accounting = value
    elif type(value) is dict:
        try:
            accounting = v13_training.JointTrainingAccountingV13(**value)
        except TypeError as error:
            raise PermissionError("V16 accounting fields changed") from error
    else:
        raise TypeError("V16 accounting must be V13 accounting or a plain dict")
    try:
        v13_training.validate_accounting_v13(accounting)
    except (TypeError, ValueError, RuntimeError) as error:
        raise PermissionError("V16 accounting identity changed") from error
    normalized = asdict(accounting)
    if normalized["updates"] != update:
        raise PermissionError("V16 accounting update differs from checkpoint update")
    return accounting, normalized


def _model_counter(model: Any, name: str) -> int:
    value = getattr(model, name, None)
    if value is None or not hasattr(value, "detach") or int(value.numel()) != 1:
        raise PermissionError(f"V16 model lacks scalar {name}")
    return int(value.detach().to(device="cpu").item())


def _state_counter(state: Mapping[str, Any], name: str) -> int:
    value = state.get(name)
    if value is None or not hasattr(value, "detach") or int(value.numel()) != 1:
        raise PermissionError(f"V16 checkpoint lacks scalar state {name}")
    return int(value.detach().to(device="cpu").item())


def _step_value(value: Any) -> int:
    if hasattr(value, "detach") and hasattr(value, "numel"):
        if int(value.numel()) != 1:
            raise PermissionError("V16 AdamW step is not scalar")
        scalar = float(value.detach().to(device="cpu").item())
    elif type(value) in {int, float}:
        scalar = float(value)
    else:
        raise PermissionError("V16 AdamW step has an invalid type")
    integer = int(scalar)
    if scalar != float(integer):
        raise PermissionError("V16 AdamW step is not integral")
    return integer


def _validate_serialized_optimizer_v16(torch: Any, value: Any, *, update: int) -> None:
    if type(value) is not dict or set(value) != {"state", "param_groups"}:
        raise PermissionError("V16 serialized optimizer fields changed")
    groups = value["param_groups"]
    state = value["state"]
    if type(groups) is not list or len(groups) != 3 or not isinstance(state, Mapping):
        raise PermissionError("V16 serialized optimizer group count changed")
    expected = (
        ("encoder", v13_training.ENCODER_LEARNING_RATE),
        ("evidence_projection_semantic", v13_training.OTHER_ONLINE_LEARNING_RATE),
        ("predictor", v13_training.OTHER_ONLINE_LEARNING_RATE),
    )
    parameter_ids: list[Any] = []
    for group, (name, learning_rate) in zip(groups, expected, strict=True):
        if (
            type(group) is not dict
            or group.get("name") != name
            or float(group.get("lr", -1.0)) != learning_rate
            or tuple(group.get("betas", ())) != v13_training.ADAMW_BETAS
            or float(group.get("eps", -1.0)) != v13_training.ADAMW_EPSILON
            or float(group.get("weight_decay", -1.0))
            != v13_training.ADAMW_WEIGHT_DECAY
            or group.get("amsgrad") is not False
            or type(group.get("params")) is not list
            or not group["params"]
        ):
            raise PermissionError(f"V16 serialized optimizer group changed: {name}")
        parameter_ids.extend(group["params"])
    if len(parameter_ids) != len(set(parameter_ids)) or set(state) != set(parameter_ids):
        raise PermissionError("V16 serialized optimizer state coverage changed")
    for parameter_id in parameter_ids:
        row = state[parameter_id]
        if type(row) is not dict or set(row) != {"step", "exp_avg", "exp_avg_sq"}:
            raise PermissionError("V16 AdamW state fields changed")
        if _step_value(row["step"]) != update:
            raise PermissionError("V16 AdamW step differs from checkpoint update")
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = row[name]
            if not isinstance(tensor, torch.Tensor) or not bool(
                torch.isfinite(tensor).all().item()
            ):
                raise PermissionError(f"V16 AdamW {name} is absent or nonfinite")


def _capture_cuda_rng_states(torch: Any) -> list[Any]:
    if not bool(torch.cuda.is_initialized()):
        return []
    return [
        state.detach().to(device="cpu").contiguous().clone()
        for state in torch.cuda.get_rng_state_all()
    ]


def _validate_rng_state(torch: Any, value: Any, *, name: str) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.uint8
        or value.ndim != 1
        or value.numel() <= 0
    ):
        raise PermissionError(f"{name} is not a CPU uint8 RNG vector")


def _metadata_v16(
    *,
    update: int,
    schedule_identity: Mapping[str, Any],
    authority: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    model_config: Mapping[str, Any],
    model_manifest: Sequence[Mapping[str, Any]],
    optimizer_state: Mapping[str, Any],
    controller_state: Mapping[str, Any],
    consumed_input_ledger: Mapping[str, Any],
    access_receipt: Mapping[str, Any],
    cpu_rng_state: Any,
    cuda_rng_states: Sequence[Any],
    ema_update_count: int,
    target_hard_sync_count: int,
) -> dict[str, Any]:
    cursor = update * PRESENTATIONS_PER_UPDATE_V16
    manifest = [dict(row) for row in model_manifest]
    return {
        "schema": f"{SCHEMA_PREFIX_V16}_full_state_recovery_metadata_v1",
        "update": update,
        "next_update": update + 1,
        "presentation_cursor": cursor,
        "presentations_per_update": PRESENTATIONS_PER_UPDATE_V16,
        "schedule": dict(schedule_identity),
        "authority": dict(authority),
        "authority_sha256": _canonical_sha256(authority),
        "source_identity": dict(source_identity),
        "source_identity_sha256": _canonical_sha256(source_identity),
        "model_config": dict(model_config),
        "model_state_manifest": manifest,
        "model_state_manifest_sha256": _canonical_sha256(manifest),
        "state_key_count": len(manifest),
        "optimizer_state_sha256": _nested_sha256(optimizer_state),
        "optimizer_group_names": [
            "encoder",
            "evidence_projection_semantic",
            "predictor",
        ],
        "optimizer_step": update,
        "ema_update_count": ema_update_count,
        "target_hard_sync_count": target_hard_sync_count,
        "controller_state_sha256": _canonical_sha256(controller_state),
        "trace_row_count": len(controller_state["trace"]),
        "metric_binding_count": len(controller_state["metric_bindings"]),
        "consumed_input_ledger_sha256": _canonical_sha256(consumed_input_ledger),
        "access_receipt_sha256": _canonical_sha256(access_receipt),
        "torch_cpu_rng_state_sha256": _nested_sha256(cpu_rng_state),
        "torch_cuda_rng_states_sha256": _nested_sha256(cuda_rng_states),
        "torch_cuda_rng_state_count": len(cuda_rng_states),
        "development_only": True,
        "recovery_only": True,
        "downstream_authority": dict(DOWNSTREAM_DENIALS_V16),
    }


def build_recovery_binding_v16(raw: bytes, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Bind one already serialized payload; this grants no write authority."""

    if not isinstance(raw, bytes) or not raw:
        raise ValueError("V16 recovery payload bytes are empty")
    if type(metadata) is not dict:
        raise TypeError("V16 recovery metadata must be a plain dict")
    update = metadata.get("update")
    _validate_update(update)
    path = f"recovery/checkpoint_update_{update}.pt"
    core = {
        "schema": BINDING_SCHEMA_V16,
        "update": update,
        "next_update": update + 1,
        "presentation_cursor": update * PRESENTATIONS_PER_UPDATE_V16,
        "payload": {
            "path": path,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
        },
        "metadata_sha256": _canonical_sha256(metadata),
        "source_identity_sha256": metadata.get("source_identity_sha256"),
        "schedule_full_sha256": metadata.get("schedule", {}).get(
            "full_schedule_sha256"
        ),
        "consumed_prefix_sha256": metadata.get("schedule", {}).get(
            "consumed_prefix_sha256"
        ),
        "recovery_only": True,
    }
    return _content_bound(core)


def serialize_recovery_checkpoint_v16(
    torch: Any,
    model: Any,
    optimizer: Any,
    accounting: Any,
    *,
    update: int,
    schedule: Sequence[int],
    schedule_receipt: Mapping[str, Any],
    authority: Mapping[str, Any],
    trace: Sequence[Mapping[str, Any]],
    metric_bindings: Sequence[Mapping[str, Any]],
    access_receipt: Mapping[str, Any],
    consumed_inputs: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    """Serialize one passing V16 milestone without changing training state."""

    _validate_update(update)
    schedule_value = _schedule_identity(schedule, schedule_receipt, update)
    authority_value, source_value = _authority_identity(authority)
    config_value = _model_config(model)
    controller_value = _controller_state(
        trace, metric_bindings, update=update
    )
    ledger_value = _json_mapping(
        consumed_inputs, name="V16 consumed-input ledger"
    )
    if not ledger_value:
        raise PermissionError("V16 consumed-input ledger is empty")
    access_value = _json_mapping(access_receipt, name="V16 access receipt")
    if not access_value:
        raise PermissionError("V16 access receipt is empty")
    _, accounting_value = _accounting(accounting, update=update)
    partition = v13_training.partition_parameters_v13(model)
    v13_training.validate_optimizer_v13(optimizer, partition)
    _validate_serialized_optimizer_v16(
        torch, optimizer.state_dict(), update=update
    )

    model_before, manifest_before = _normalized_model_state(model)
    model_sha_before = _canonical_sha256(manifest_before)
    optimizer_sha_before = _nested_sha256(optimizer.state_dict())
    cpu_rng_before = torch.random.get_rng_state().detach().cpu().clone()
    cuda_rng_before = _capture_cuda_rng_states(torch)
    ema_count = _model_counter(model, "ema_update_count")
    hard_sync_count = _model_counter(model, "target_hard_sync_count")
    if ema_count != update or hard_sync_count != 1:
        raise PermissionError("V16 model EMA or hard-sync count changed")

    optimizer_state = _clone_to_cpu(optimizer.state_dict())
    metadata = _metadata_v16(
        update=update,
        schedule_identity=schedule_value,
        authority=authority_value,
        source_identity=source_value,
        model_config=config_value,
        model_manifest=manifest_before,
        optimizer_state=optimizer_state,
        controller_state=controller_value,
        consumed_input_ledger=ledger_value,
        access_receipt=access_value,
        cpu_rng_state=cpu_rng_before,
        cuda_rng_states=cuda_rng_before,
        ema_update_count=ema_count,
        target_hard_sync_count=hard_sync_count,
    )
    payload = {
        "schema": PAYLOAD_SCHEMA_V16,
        "update": update,
        "next_update": update + 1,
        "presentation_cursor": update * PRESENTATIONS_PER_UPDATE_V16,
        "metadata": metadata,
        "metadata_sha256": _canonical_sha256(metadata),
        "model_state_dict": model_before,
        "optimizer_state_dict": optimizer_state,
        "accounting": accounting_value,
        "torch_cpu_rng_state": cpu_rng_before,
        "torch_cuda_rng_states": cuda_rng_before,
        "controller_state": controller_value,
        "consumed_input_ledger": ledger_value,
        "access_receipt": access_value,
    }
    stream = io.BytesIO()
    torch.save(payload, stream)
    raw = stream.getvalue()
    if not raw:
        raise RuntimeError("V16 recovery serialization produced no bytes")

    _, manifest_after = _normalized_model_state(model)
    if (
        _canonical_sha256(manifest_after) != model_sha_before
        or _nested_sha256(optimizer.state_dict()) != optimizer_sha_before
        or not torch.equal(torch.random.get_rng_state(), cpu_rng_before)
    ):
        raise RuntimeError("V16 recovery serialization mutated training state")
    cuda_rng_after = _capture_cuda_rng_states(torch)
    if len(cuda_rng_after) != len(cuda_rng_before) or any(
        not torch.equal(before, after)
        for before, after in zip(cuda_rng_before, cuda_rng_after, strict=True)
    ):
        raise RuntimeError("V16 recovery serialization mutated CUDA RNG state")
    return raw, build_recovery_binding_v16(raw, metadata)


def _load_and_validate_payload_v16(
    torch: Any,
    raw: bytes,
    binding: Mapping[str, Any] | None,
    *,
    expected_update: int,
    schedule: Sequence[int],
    schedule_receipt: Mapping[str, Any],
    authority: Mapping[str, Any],
    expected_model_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_update(expected_update)
    if binding is None:
        raise PermissionError("orphan V16 recovery payload has no exact binding")
    bound = _validate_content_bound(binding)
    expected_path = f"recovery/checkpoint_update_{expected_update}.pt"
    payload_binding = bound["payload"]
    if (
        type(payload_binding) is not dict
        or set(payload_binding) != {"path", "file_sha256", "byte_count"}
        or payload_binding["path"] != expected_path
        or payload_binding["file_sha256"] != hashlib.sha256(raw).hexdigest()
        or payload_binding["byte_count"] != len(raw)
    ):
        raise PermissionError("V16 recovery payload file binding changed")
    try:
        payload = torch.load(
            io.BytesIO(raw), map_location="cpu", weights_only=True
        )
    except Exception as error:
        raise PermissionError("V16 recovery payload cannot be safely loaded") from error
    if type(payload) is not dict or set(payload) != _PAYLOAD_KEYS_V16:
        raise PermissionError("V16 recovery payload fields changed")
    if (
        payload["schema"] != PAYLOAD_SCHEMA_V16
        or payload["update"] != expected_update
        or payload["next_update"] != expected_update + 1
        or payload["presentation_cursor"]
        != expected_update * PRESENTATIONS_PER_UPDATE_V16
        or bound["schema"] != BINDING_SCHEMA_V16
        or bound["update"] != expected_update
        or bound["next_update"] != expected_update + 1
        or bound["presentation_cursor"] != payload["presentation_cursor"]
        or bound["recovery_only"] is not True
    ):
        raise PermissionError("V16 recovery cursor or identity changed")

    metadata = payload["metadata"]
    if (
        type(metadata) is not dict
        or set(metadata) != _METADATA_KEYS_V16
        or payload["metadata_sha256"] != _canonical_sha256(metadata)
        or bound["metadata_sha256"] != payload["metadata_sha256"]
        or metadata.get("update") != expected_update
        or metadata.get("next_update") != expected_update + 1
        or metadata.get("presentation_cursor") != payload["presentation_cursor"]
        or metadata.get("presentations_per_update")
        != PRESENTATIONS_PER_UPDATE_V16
        or metadata.get("development_only") is not True
        or metadata.get("recovery_only") is not True
        or metadata.get("downstream_authority") != DOWNSTREAM_DENIALS_V16
    ):
        raise PermissionError("V16 recovery metadata changed")

    schedule_value = _schedule_identity(
        schedule, schedule_receipt, expected_update
    )
    authority_value, source_value = _authority_identity(authority)
    config_value = (
        None
        if expected_model_config is None
        else _json_mapping(
            expected_model_config, name="V16 expected model config"
        )
    )
    if (
        metadata.get("schedule") != schedule_value
        or bound["schedule_full_sha256"]
        != schedule_value["full_schedule_sha256"]
        or bound["consumed_prefix_sha256"]
        != schedule_value["consumed_prefix_sha256"]
        or metadata.get("source_identity") != source_value
        or metadata.get("source_identity_sha256")
        != _canonical_sha256(source_value)
        or bound["source_identity_sha256"]
        != metadata["source_identity_sha256"]
        or metadata.get("authority") != authority_value
        or metadata.get("authority_sha256")
        != _canonical_sha256(authority_value)
        or (
            config_value is not None
            and metadata.get("model_config") != config_value
        )
    ):
        raise PermissionError("V16 recovery source, schedule, or config changed")

    state = payload["model_state_dict"]
    if not isinstance(state, Mapping) or not state:
        raise PermissionError("V16 recovery model state is absent")
    state_manifest = [
        {"name": name, **_tensor_receipt(value)}
        for name, value in sorted(state.items())
    ]
    if (
        metadata.get("model_state_manifest") != state_manifest
        or metadata.get("model_state_manifest_sha256")
        != _canonical_sha256(state_manifest)
        or metadata.get("state_key_count") != len(state_manifest)
        or _state_counter(state, "ema_update_count") != expected_update
        or _state_counter(state, "target_hard_sync_count") != 1
        or metadata.get("ema_update_count") != expected_update
        or metadata.get("target_hard_sync_count") != 1
    ):
        raise PermissionError("V16 recovery model or EMA identity changed")

    _accounting(payload["accounting"], update=expected_update)
    _validate_serialized_optimizer_v16(
        torch, payload["optimizer_state_dict"], update=expected_update
    )
    if (
        metadata.get("optimizer_state_sha256")
        != _nested_sha256(payload["optimizer_state_dict"])
        or metadata.get("optimizer_step") != expected_update
        or metadata.get("optimizer_group_names")
        != ["encoder", "evidence_projection_semantic", "predictor"]
    ):
        raise PermissionError("V16 recovery optimizer identity changed")

    controller_payload = payload["controller_state"]
    if (
        type(controller_payload) is not dict
        or set(controller_payload) != CONTROLLER_STATE_KEYS_V16
    ):
        raise PermissionError("V16 recovery controller fields changed")
    controller = _controller_state(
        controller_payload["trace"],
        controller_payload["metric_bindings"],
        update=expected_update,
    )
    ledger = _json_mapping(
        payload["consumed_input_ledger"], name="V16 recovery consumed ledger"
    )
    access = _json_mapping(
        payload["access_receipt"], name="V16 recovery access receipt"
    )
    if (
        not ledger
        or not access
        or metadata.get("controller_state_sha256")
        != _canonical_sha256(controller)
        or metadata.get("trace_row_count") != len(controller["trace"])
        or metadata.get("metric_binding_count")
        != len(controller["metric_bindings"])
        or metadata.get("consumed_input_ledger_sha256")
        != _canonical_sha256(ledger)
        or metadata.get("access_receipt_sha256") != _canonical_sha256(access)
    ):
        raise PermissionError("V16 recovery controller or custody identity changed")

    cpu_rng = payload["torch_cpu_rng_state"]
    cuda_rng = payload["torch_cuda_rng_states"]
    _validate_rng_state(torch, cpu_rng, name="V16 CPU RNG state")
    if type(cuda_rng) is not list:
        raise PermissionError("V16 CUDA RNG state collection changed")
    for index, state_value in enumerate(cuda_rng):
        _validate_rng_state(
            torch, state_value, name=f"V16 CUDA RNG state {index}"
        )
    if (
        metadata.get("torch_cpu_rng_state_sha256") != _nested_sha256(cpu_rng)
        or metadata.get("torch_cuda_rng_states_sha256")
        != _nested_sha256(cuda_rng)
        or metadata.get("torch_cuda_rng_state_count") != len(cuda_rng)
    ):
        raise PermissionError("V16 recovery RNG identity changed")
    return payload


def validate_recovery_checkpoint_v16(
    raw_or_payload: bytes | Mapping[str, Any],
    *,
    torch_module: Any,
    binding: Mapping[str, Any] | None,
    expected_update: int,
    schedule: Sequence[int],
    schedule_receipt: Mapping[str, Any],
    authority: Mapping[str, Any],
    expected_model_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate exact artifact bytes, or a prepublication payload mapping.

    Recovery callers must pass the original bytes and immutable binding.  The
    mapping form exists only for source-only construction tests: it receives a
    fresh local binding and therefore does not authenticate an external file.
    """

    if isinstance(raw_or_payload, bytes):
        raw = raw_or_payload
        selected_binding = binding
    elif type(raw_or_payload) is dict:
        if binding is not None:
            raise PermissionError(
                "a loaded V16 payload cannot authenticate external file bytes"
            )
        stream = io.BytesIO()
        torch_module.save(raw_or_payload, stream)
        raw = stream.getvalue()
        metadata = raw_or_payload.get("metadata")
        if type(metadata) is not dict:
            raise PermissionError("V16 prepublication payload metadata is absent")
        selected_binding = build_recovery_binding_v16(raw, metadata)
    else:
        raise TypeError("V16 recovery input must be payload bytes or a plain dict")
    return _load_and_validate_payload_v16(
        torch_module,
        raw,
        selected_binding,
        expected_update=expected_update,
        schedule=schedule,
        schedule_receipt=schedule_receipt,
        authority=authority,
        expected_model_config=expected_model_config,
    )


def restore_recovery_checkpoint_v16(
    raw: bytes,
    model: Any,
    optimizer: Any,
    *,
    torch_module: Any,
    binding: Mapping[str, Any] | None,
    expected_update: int,
    schedule: Sequence[int],
    schedule_receipt: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Strict-load a fresh exact topology, then restore RNG state last."""

    if not isinstance(raw, bytes):
        raise TypeError("V16 restoration requires original immutable payload bytes")
    payload = validate_recovery_checkpoint_v16(
        raw,
        torch_module=torch_module,
        binding=binding,
        expected_update=expected_update,
        schedule=schedule,
        schedule_receipt=schedule_receipt,
        authority=authority,
        expected_model_config=_model_config(model),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    partition = v13_training.partition_parameters_v13(model)
    v13_training.validate_optimizer_v13(optimizer, partition)
    _validate_serialized_optimizer_v16(
        torch_module, optimizer.state_dict(), update=expected_update
    )
    if (
        _model_counter(model, "ema_update_count") != expected_update
        or _model_counter(model, "target_hard_sync_count") != 1
    ):
        raise PermissionError("restored V16 model counter identity changed")
    _, manifest = _normalized_model_state(model)
    if _canonical_sha256(manifest) != payload["metadata"][
        "model_state_manifest_sha256"
    ]:
        raise PermissionError("restored V16 model state hash changed")
    accounting, accounting_value = _accounting(
        payload["accounting"], update=expected_update
    )

    saved_cuda = payload["torch_cuda_rng_states"]
    current_cuda_count = (
        len(torch_module.cuda.get_rng_state_all())
        if torch_module.cuda.is_initialized()
        else 0
    )
    if current_cuda_count != len(saved_cuda):
        raise PermissionError("V16 CUDA RNG device count changed across recovery")
    torch_module.random.set_rng_state(payload["torch_cpu_rng_state"])
    if saved_cuda:
        torch_module.cuda.set_rng_state_all(saved_cuda)
    if not torch_module.equal(
        torch_module.random.get_rng_state(), payload["torch_cpu_rng_state"]
    ):
        raise RuntimeError("V16 CPU RNG state was not restored exactly")
    if saved_cuda and any(
        not torch_module.equal(before, after)
        for before, after in zip(
            saved_cuda, torch_module.cuda.get_rng_state_all(), strict=True
        )
    ):
        raise RuntimeError("V16 CUDA RNG state was not restored exactly")
    return {
        "accounting": accounting,
        "accounting_dict": accounting_value,
        "completed_update": expected_update,
        "next_update": expected_update + 1,
        "presentation_cursor": expected_update * PRESENTATIONS_PER_UPDATE_V16,
        "controller_state": dict(payload["controller_state"]),
        "consumed_input_ledger": dict(payload["consumed_input_ledger"]),
        "access_receipt": dict(payload["access_receipt"]),
        "metadata": dict(payload["metadata"]),
    }


def publish_recovery_checkpoint_v16(
    raw: bytes,
    binding: Mapping[str, Any],
    publish_bytes: Callable[[str, bytes], Mapping[str, Any]],
) -> dict[str, Any]:
    """Publish payload first and its exact binding second via a write-once callback."""

    bound = _validate_content_bound(binding)
    expected_payload = bound["payload"]
    if (
        not isinstance(raw, bytes)
        or expected_payload.get("file_sha256")
        != hashlib.sha256(raw).hexdigest()
        or expected_payload.get("byte_count") != len(raw)
    ):
        raise PermissionError("V16 payload bytes differ from their binding")
    payload_receipt = dict(publish_bytes(expected_payload["path"], raw))
    if any(payload_receipt.get(name) != expected_payload[name] for name in expected_payload):
        raise RuntimeError("V16 payload publisher receipt changed")
    binding_path = f"recovery/checkpoint_update_{bound['update']}.binding.json"
    binding_raw = _canonical_json_bytes(bound) + b"\n"
    binding_receipt = dict(publish_bytes(binding_path, binding_raw))
    expected_binding = {
        "path": binding_path,
        "file_sha256": hashlib.sha256(binding_raw).hexdigest(),
        "byte_count": len(binding_raw),
    }
    if any(binding_receipt.get(name) != value for name, value in expected_binding.items()):
        raise RuntimeError("V16 binding publisher receipt changed")
    return {"payload": payload_receipt, "binding": binding_receipt, "value": bound}


__all__ = [
    "BINDING_SCHEMA_V16",
    "DOWNSTREAM_DENIALS_V16",
    "MAXIMUM_PRESENTATIONS_V16",
    "PAYLOAD_SCHEMA_V16",
    "PRESENTATIONS_PER_UPDATE_V16",
    "PREREGISTRATION_COMMIT_V16",
    "RECOVERY_UPDATES_V16",
    "SCHEMA_PREFIX_V16",
    "build_recovery_binding_v16",
    "publish_recovery_checkpoint_v16",
    "restore_recovery_checkpoint_v16",
    "serialize_recovery_checkpoint_v16",
    "validate_recovery_checkpoint_v16",
    "v13_training",
]
