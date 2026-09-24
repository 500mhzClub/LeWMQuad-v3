#!/usr/bin/env python3
"""Canonical caller-hashed finalizer for the V4 development-fit ladder."""
from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import types
import uuid
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DEVELOPMENT_ROOT = (
    ROOT / ".generated/go2_observable_camera_ray_fit_v4/development_fit_v2"
)
CANONICAL_ATTEMPT_ROOT = CANONICAL_DEVELOPMENT_ROOT / "attempts"
CANONICAL_GATE_ROOT = CANONICAL_DEVELOPMENT_ROOT / "gates"
CANONICAL_METRIC_RECEIPT_ROOT = CANONICAL_DEVELOPMENT_ROOT / "metric_verifications"
CANONICAL_TRAINER_AUTHORIZATION_PATH = (
    ROOT
    / "docs/lewm_go2_observable_camera_ray_fit_v4_trainer_authorization_bound_2026-07-12.json"
).resolve()
CANONICAL_TRAINER_REVIEW_PATH = (
    ROOT
    / "docs/lewm_go2_observable_camera_ray_fit_v4_trainer_review_record_2026-07-12.json"
).resolve()
LADDER_FIT_SIZES = (5, 16, 32, 320)
EXPECTED_SEEDS = (20260710, 20260711)
LAUNCHER_LOGICAL_NAME = "scripts.launch_go2_observable_camera_ray_fit_v4"
LAUNCHER_RELATIVE_PATH = "scripts/launch_go2_observable_camera_ray_fit_v4.py"


gate: Any = None
metric_verifier: Any = None
if __name__.startswith("_lewm_v4_ca_"):
    from lewm.benchmarks import (
        go2_observable_camera_ray_fit_v4_ladder_gate as gate,
    )
    from scripts import verify_go2_observable_camera_ray_fit_v4_metrics as metric_verifier


def _require_captured_private_finalizer() -> None:
    logical_name = "scripts.finalize_go2_observable_camera_ray_fit_v4_ladder"
    if (
        __name__ == logical_name
        or not __name__.startswith("_lewm_v4_ca_")
        or globals().get("__verified_logical_name__") != logical_name
    ):
        raise PermissionError("V4 finalizer library computation is unsupported")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    ).encode("utf-8")


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _read_caller_hashed_bytes(
    path: Path,
    expected_file_sha256: str,
    *,
    name: str,
) -> bytes:
    if not _is_sha256(expected_file_sha256):
        raise ValueError(f"caller {name} SHA-256 is malformed")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PermissionError(f"{name} is not a regular file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
        )
        if identity(before) != identity(after):
            raise RuntimeError(f"{name} changed while read")
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if hashlib.sha256(raw).hexdigest() != expected_file_sha256:
        raise ValueError(f"{name} caller SHA-256 changed")
    return raw


def load_caller_hashed_json(
    path: Path,
    expected_file_sha256: str,
    *,
    name: str,
) -> dict[str, Any]:
    raw = _read_caller_hashed_bytes(path, expected_file_sha256, name=name)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict JSON") from exc
    if not isinstance(value, dict) or raw != _canonical_json_bytes(value) + b"\n":
        raise ValueError(f"{name} is not canonical newline-terminated JSON")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_gate_exclusive(
    path: Path,
    value: Mapping[str, Any],
    *,
    enforce_canonical_root: bool = True,
) -> dict[str, Any]:
    if enforce_canonical_root:
        _require_captured_private_finalizer()
    destination = path.resolve()
    if enforce_canonical_root:
        if CANONICAL_GATE_ROOT.is_symlink():
            raise PermissionError("V4 canonical gate root may not be a symlink")
        root = CANONICAL_GATE_ROOT.resolve()
        if destination.parent != root:
            raise PermissionError("V4 gate output is not a canonical direct child")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise PermissionError("V4 gate root is not a real directory")
    payload = _canonical_json_bytes(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    created = False
    try:
        descriptor = os.open(destination, flags, 0o644)
        created = True
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(destination.parent)
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
            try:
                _fsync_directory(destination.parent)
            except OSError:
                pass
        raise
    return {
        "path": str(destination),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "content_sha256": value["content_sha256"],
    }


def _parse_bound_path(value: str) -> tuple[Path, str]:
    path_text, separator, digest = value.rpartition(":")
    if not separator or not path_text or not _is_sha256(digest):
        raise ValueError("bound input must be PATH:SHA256")
    return Path(path_text), digest


def _load_bound(value: str, *, name: str) -> tuple[Path, str, dict[str, Any]]:
    path, digest = _parse_bound_path(value)
    return path, digest, load_caller_hashed_json(path, digest, name=name)


def canonical_stage_gate_path(seed: int, fit_size: int) -> Path:
    if seed not in EXPECTED_SEEDS or fit_size not in LADDER_FIT_SIZES:
        raise ValueError("V4 stage gate seed/rung is outside the ladder")
    return CANONICAL_GATE_ROOT / f"stage_seed_{seed}_n{fit_size}.json"


def canonical_seed_gate_path(seed: int) -> Path:
    if seed not in EXPECTED_SEEDS:
        raise ValueError("V4 seed gate seed is outside the ladder")
    return CANONICAL_GATE_ROOT / f"seed_{seed}.json"


def canonical_two_seed_gate_path() -> Path:
    return CANONICAL_GATE_ROOT / "two_seed.json"


def canonical_metric_receipt_path(seed: int, fit_size: int) -> Path:
    if seed not in EXPECTED_SEEDS or fit_size not in LADDER_FIT_SIZES:
        raise ValueError("V4 metric receipt seed/rung is outside the ladder")
    return CANONICAL_METRIC_RECEIPT_ROOT / f"seed_{seed}_n{fit_size}.json"


def _require_path(actual: Path, expected: Path, *, name: str) -> None:
    if str(actual) != str(expected) or actual.resolve(strict=True) != expected:
        raise PermissionError(f"{name} path is not canonical")


def _preflight_stage_authorization(
    *,
    authorization_bound: str,
    review_bound: str,
) -> dict[str, Any]:
    """Revalidate both fixed trainer-authority files before artifact reads."""

    _require_captured_private_finalizer()
    authorization_path, authorization_sha = _parse_bound_path(authorization_bound)
    review_path, review_sha = _parse_bound_path(review_bound)
    _require_path(
        authorization_path,
        CANONICAL_TRAINER_AUTHORIZATION_PATH,
        name="V4 trainer authorization",
    )
    _require_path(
        review_path,
        CANONICAL_TRAINER_REVIEW_PATH,
        name="V4 trainer review record",
    )
    from scripts import launch_go2_observable_camera_ray_fit_v4 as launcher

    return launcher.preflight_exact_authorization(
        dataset_path=launcher.CANONICAL_DATASET_PATH,
        dataset_file_sha256=launcher.DATASET_MANIFEST_FILE_SHA256,
        audit_path=launcher.CANONICAL_AUDIT_PATH,
        audit_file_sha256=launcher.AUDIT_RECEIPT_FILE_SHA256,
        authorization_path=authorization_path,
        authorization_file_sha256=authorization_sha,
        review_record_path=review_path,
        review_record_file_sha256=review_sha,
    )


def _file_binding(path: str, raw: bytes, content_sha256: str) -> dict[str, Any]:
    return {
        "path": path,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": content_sha256,
    }


def _validate_content_hash(value: Mapping[str, Any], *, name: str) -> None:
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if not _is_sha256(declared) or declared != gate.canonical_json_sha256(core):
        raise ValueError(f"{name} content SHA-256 changed")


def _validate_checkpoint(
    raw: bytes,
    *,
    expected_content_sha256: str,
    expected_metadata: Mapping[str, Any],
) -> None:
    _require_captured_private_finalizer()
    import torch
    from lewm.models.observable_camera_ray_evidence_v4 import (
        ObservableCameraRayEvidenceV4Model,
    )

    payload = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema",
        "model_class",
        "state_manifest",
        "metadata",
        "authoritative",
        "aggregation_eligible",
        "promotion_eligible",
        "state_dict",
        "content_sha256",
    } or (
        payload.get("schema")
        != "lewm_go2_observable_camera_ray_fit_v4_development_checkpoint_v2"
        or payload.get("model_class") != "ObservableCameraRayEvidenceV4Model"
        or payload.get("metadata") != dict(expected_metadata)
        or payload.get("authoritative") is not False
        or payload.get("aggregation_eligible") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("content_sha256") != expected_content_sha256
    ):
        raise PermissionError("V4 checkpoint metadata/scope changed")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("V4 checkpoint state is malformed")
    model = ObservableCameraRayEvidenceV4Model()
    if sum(parameter.numel() for parameter in model.parameters()) != gate.MODEL_PARAMETER_COUNT:
        raise ValueError("V4 canonical model parameter count changed")
    canonical_state = model.state_dict()
    if set(state) != set(canonical_state):
        raise ValueError("V4 checkpoint state keys changed")
    canonical_buffers = dict(model.named_buffers())
    manifest = []
    for name in sorted(state):
        tensor = state[name]
        expected = canonical_state[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.layout != torch.strided
            or tuple(tensor.shape) != tuple(expected.shape)
            or tensor.dtype != expected.dtype
            or not bool(torch.isfinite(tensor).all().item())
        ):
            raise ValueError(f"V4 checkpoint tensor contract changed: {name}")
        tensor = tensor.detach().cpu().contiguous()
        if name in canonical_buffers and not torch.equal(tensor, expected.cpu()):
            raise ValueError(f"V4 deterministic checkpoint buffer changed: {name}")
        manifest.append(
            {
                "name": name,
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
                "sha256": hashlib.sha256(
                    tensor.numpy().tobytes(order="C")
                ).hexdigest(),
            }
        )
    if payload.get("state_manifest") != manifest:
        raise ValueError("V4 checkpoint state manifest changed")
    semantic_core = {
        key: payload[key]
        for key in (
            "schema",
            "model_class",
            "state_manifest",
            "metadata",
            "authoritative",
            "aggregation_eligible",
            "promotion_eligible",
        )
    }
    if gate.canonical_json_sha256(semantic_core) != expected_content_sha256:
        raise ValueError("V4 checkpoint semantic content hash changed")
    model.load_state_dict(dict(state), strict=True)


def _validate_stage_artifact_bundle(
    *,
    seed: int,
    fit_size: int,
    reservation_bound: str,
    result_bound: str,
    checkpoint_bound: str,
    completion_bound: str,
    metric_verification_bound: str,
    authorization_bound: str,
    review_bound: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _require_captured_private_finalizer()
    expected_directory = CANONICAL_ATTEMPT_ROOT / f"seed_{seed}" / f"n{fit_size}"
    constrained = {
        "reservation": (*_parse_bound_path(reservation_bound), "reservation.json"),
        "result": (*_parse_bound_path(result_bound), "result.json"),
        "checkpoint": (*_parse_bound_path(checkpoint_bound), "checkpoint.pt"),
        "completion": (*_parse_bound_path(completion_bound), "completed.json"),
    }
    for name, (path, _digest, filename) in constrained.items():
        _require_path(path, expected_directory / filename, name=f"V4 {name}")
    metric_path, metric_sha = _parse_bound_path(metric_verification_bound)
    _require_path(
        metric_path,
        canonical_metric_receipt_path(seed, fit_size),
        name="V4 metric verification receipt",
    )
    preauth = _preflight_stage_authorization(
        authorization_bound=authorization_bound,
        review_bound=review_bound,
    )
    metric_authorization_path = metric_verifier.CANONICAL_AUTHORIZATION_PATH
    metric_authorization_sha = metric_verifier.AUTHORIZATION_FILE_SHA256
    metric_verifier.preflight_metric_verifier_authorization(
        metric_authorization_path,
        metric_authorization_sha,
    )
    authorization_path, authorization_sha = _parse_bound_path(authorization_bound)
    review_path, review_sha = _parse_bound_path(review_bound)
    reservation_path, reservation_sha, reservation = _load_bound(
        reservation_bound, name="V4 attempt reservation"
    )
    result_path, result_sha, result = _load_bound(
        result_bound, name="V4 development fit result"
    )
    checkpoint_path, checkpoint_sha = _parse_bound_path(checkpoint_bound)
    checkpoint_raw = _read_caller_hashed_bytes(
        checkpoint_path, checkpoint_sha, name="V4 development checkpoint"
    )
    completion_path, completion_sha, completion = _load_bound(
        completion_bound, name="V4 attempt completion"
    )
    if result.get("fit_size") != fit_size:
        raise ValueError("V4 result rung changed")
    if expected_directory.is_symlink() or not expected_directory.is_dir():
        raise PermissionError("V4 attempt directory is not real")
    entries = list(expected_directory.iterdir())
    if {entry.name for entry in entries} != {
        "reservation.json",
        "result.json",
        "checkpoint.pt",
        "completed.json",
    } or any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise PermissionError("V4 completed attempt inventory changed")
    _validate_content_hash(reservation, name="V4 attempt reservation")
    _validate_content_hash(result, name="V4 development result")
    _validate_content_hash(completion, name="V4 attempt completion")
    reservation_binding = _file_binding(
        "reservation.json", _canonical_json_bytes(reservation) + b"\n", reservation["content_sha256"]
    )
    result_binding = _file_binding(
        "result.json", _canonical_json_bytes(result) + b"\n", result["content_sha256"]
    )
    checkpoint_receipt = result.get("model", {}).get("checkpoint", {})
    if not isinstance(checkpoint_receipt, Mapping) or set(checkpoint_receipt) != {
        "path",
        "file_sha256",
        "content_sha256",
        "byte_count",
        "development_only",
    } or (
        checkpoint_receipt.get("path") != "checkpoint.pt"
        or checkpoint_receipt.get("file_sha256") != checkpoint_sha
        or not _is_sha256(checkpoint_receipt.get("content_sha256"))
        or checkpoint_receipt.get("byte_count") != len(checkpoint_raw)
        or checkpoint_receipt.get("development_only") is not True
    ):
        raise PermissionError("V4 result checkpoint receipt changed")
    checkpoint_binding = {
        "path": "checkpoint.pt",
        "file_sha256": checkpoint_sha,
        "content_sha256": checkpoint_receipt.get("content_sha256"),
        "byte_count": len(checkpoint_raw),
    }
    completion_binding = _file_binding(
        "completed.json", _canonical_json_bytes(completion) + b"\n", completion["content_sha256"]
    )
    expected_reservation_fields = {
        "schema",
        "contract",
        "predecessor_failure",
        "seed",
        "fit_size",
        "attempt_index",
        "maximum_attempts",
        "scope",
        "inputs",
        "prerequisite_gates",
        "licenses",
        "content_sha256",
    }
    expected_reservation_input_fields = {
        "dataset_manifest_file_sha256",
        "dataset_manifest_content_sha256",
        "audit_receipt_file_sha256",
        "audit_receipt_content_sha256",
        "trainer_authorization_file_sha256",
        "trainer_authorization_content_sha256",
        "trainer_review_record_file_sha256",
        "trainer_review_record_content_sha256",
        "rgb_receipt_content_sha256",
        "subset_content_sha256",
        "target_partition",
        "source_map_sha256",
    }
    reservation_inputs = reservation.get("inputs")
    prerequisites = reservation.get("prerequisite_gates")
    if set(reservation) != expected_reservation_fields or (
        reservation.get("schema") != gate.ATTEMPT_RESERVATION_SCHEMA
        or reservation.get("contract") != gate.LADDER_CONTRACT
        or reservation.get("predecessor_failure") != gate.V1_FAILURE_LINEAGE
        or reservation.get("seed") != seed
        or reservation.get("fit_size") != fit_size
        or reservation.get("attempt_index") != 1
        or reservation.get("maximum_attempts") != 1
        or reservation.get("scope") != "one_frozen_attempt_per_seed_and_fit_size"
        or not isinstance(reservation_inputs, Mapping)
        or set(reservation_inputs) != expected_reservation_input_fields
        or reservation_inputs.get("dataset_manifest_file_sha256")
        != gate.DATASET_MANIFEST_FILE_SHA256
        or reservation_inputs.get("dataset_manifest_content_sha256")
        != gate.DATASET_MANIFEST_CONTENT_SHA256
        or reservation_inputs.get("audit_receipt_file_sha256")
        != gate.AUDIT_RECEIPT_FILE_SHA256
        or reservation_inputs.get("audit_receipt_content_sha256")
        != gate.AUDIT_RECEIPT_CONTENT_SHA256
        or reservation_inputs.get("rgb_receipt_content_sha256")
        != gate.RGB_RECEIPT_CONTENT_SHA256
        or reservation_inputs.get("subset_content_sha256")
        != gate.EXPECTED_SUBSET_CONTENT_SHA256[fit_size]
        or reservation_inputs.get("target_partition")
        != gate.target_partition_binding_v4(fit_size)
        or not all(
            _is_sha256(reservation_inputs.get(key))
            for key in (
                "trainer_authorization_file_sha256",
                "trainer_authorization_content_sha256",
                "trainer_review_record_file_sha256",
                "trainer_review_record_content_sha256",
                "source_map_sha256",
            )
        )
        or not isinstance(prerequisites, Mapping)
        or set(prerequisites) != {"previous_stage_gate", "seed_20260710_gate"}
        or reservation.get("licenses")
        != {
            "development_checkpoint_creation_authorized": True,
            "checkpoint_use_authorized": False,
            "holdout_authorized": False,
            "g2_authorized": False,
            "runtime_authorized": False,
            "promotion_authorized": False,
        }
        or result.get("attempt", {}).get("reservation") != reservation_binding
        or result.get("attempt", {}).get("predecessor_failure")
        != reservation["predecessor_failure"]
        or result_sha != result_binding["file_sha256"]
        or reservation_sha != reservation_binding["file_sha256"]
    ):
        raise PermissionError("V4 attempt reservation/result chain changed")
    expected_completion_fields = {
        "schema",
        "status",
        "reservation",
        "checkpoint",
        "result",
        "inventory",
        "licenses",
        "content_sha256",
    }
    if set(completion) != expected_completion_fields or (
        completion.get("schema") != gate.ATTEMPT_COMPLETION_SCHEMA
        or completion.get("status") != "completed"
        or completion.get("reservation") != reservation_binding
        or completion.get("checkpoint") != checkpoint_binding
        or completion.get("result")
        != {**result_binding, "byte_count": len(_canonical_json_bytes(result) + b"\n")}
        or completion.get("inventory")
        != ["checkpoint.pt", "completed.json", "reservation.json", "result.json"]
        or completion.get("licenses")
        != {
            "development_checkpoint_creation_authorized": True,
            "checkpoint_use_authorized": False,
            "holdout_authorized": False,
            "g2_authorized": False,
            "runtime_authorized": False,
            "promotion_authorized": False,
        }
        or completion_sha != completion_binding["file_sha256"]
    ):
        raise PermissionError("V4 attempt completion chain changed")
    result_inputs = result.get("inputs", {})
    reservation_inputs = reservation.get("inputs", {})
    if (
        result_inputs.get("trainer_authorization_file_sha256") != authorization_sha
        or result_inputs.get("trainer_authorization_content_sha256")
        != preauth["authorization_content_sha256"]
        or result_inputs.get("trainer_review_record_file_sha256") != review_sha
        or result_inputs.get("trainer_review_record_content_sha256")
        != preauth["review_record_content_sha256"]
        or result_inputs.get("rgb_receipt_content_sha256")
        != gate.RGB_RECEIPT_CONTENT_SHA256
        or reservation_inputs.get("source_map_sha256")
        != preauth["source_map_sha256"]
        or reservation_inputs.get("trainer_authorization_file_sha256")
        != authorization_sha
        or reservation_inputs.get("trainer_authorization_content_sha256")
        != preauth["authorization_content_sha256"]
        or reservation_inputs.get("trainer_review_record_file_sha256") != review_sha
        or reservation_inputs.get("trainer_review_record_content_sha256")
        != preauth["review_record_content_sha256"]
        or reservation_inputs.get("subset_content_sha256")
        != gate.EXPECTED_SUBSET_CONTENT_SHA256[fit_size]
        or result_inputs.get("target_partition_content_sha256")
        != gate.target_partition_binding_v4(fit_size)["content_sha256"]
        or result.get("target_partition")
        != gate.target_partition_binding_v4(fit_size)
        or reservation_inputs.get("target_partition")
        != gate.target_partition_binding_v4(fit_size)
    ):
        raise PermissionError("V4 reviewed authorization/result bindings changed")
    result_previous = result_inputs.get("previous_stage_gate")
    result_seed = result_inputs.get("seed_20260710_gate")
    if prerequisites != {
        "previous_stage_gate": result_previous,
        "seed_20260710_gate": result_seed,
    }:
        raise PermissionError("V4 reservation prerequisite bindings changed")
    expected_metadata = {
        **reservation_inputs,
        "fit_size": fit_size,
        "seed": seed,
        "training_schedule_sha256": result.get("training", {}).get("schedule_sha256"),
        "attempt_reservation": reservation_binding,
        "predecessor_failure": reservation["predecessor_failure"],
        "prerequisite_gates": prerequisites,
    }
    _validate_checkpoint(
        checkpoint_raw,
        expected_content_sha256=checkpoint_binding["content_sha256"],
        expected_metadata=expected_metadata,
    )
    metric_receipt = metric_verifier.reverify_canonical_metric_receipt(
        receipt_path=metric_path,
        receipt_file_sha256=metric_sha,
        metric_authorization_path=metric_authorization_path,
        metric_authorization_file_sha256=metric_authorization_sha,
        trainer_authorization_path=authorization_path,
        trainer_authorization_file_sha256=authorization_sha,
        trainer_review_path=review_path,
        trainer_review_file_sha256=review_sha,
        reservation_path=reservation_path,
        reservation_file_sha256=reservation_sha,
        result_path=result_path,
        result_file_sha256=result_sha,
        checkpoint_path=checkpoint_path,
        checkpoint_file_sha256=checkpoint_sha,
        seed=seed,
        fit_size=fit_size,
    )
    metric_raw = _canonical_json_bytes(metric_receipt) + b"\n"
    artifact_binding = {
        "attempt_directory": f"attempts/seed_{seed}/n{fit_size}",
        "reservation": reservation_binding,
        "result": result_binding,
        "checkpoint": checkpoint_binding,
        "completion": completion_binding,
        "metric_verification": _file_binding(
            f"metric_verifications/seed_{seed}_n{fit_size}.json",
            metric_raw,
            metric_receipt["content_sha256"],
        ),
    }
    if artifact_binding["metric_verification"]["file_sha256"] != metric_sha:
        raise ValueError("V4 metric verification caller SHA-256 changed")
    return result, artifact_binding, metric_receipt


def _bound(path: Path, binding: Mapping[str, Any]) -> str:
    return f"{path}:{binding['file_sha256']}"


def _load_canonical_gate_record(
    path: Path,
    expected_file_sha256: str,
    *,
    expected_path: Path,
    name: str,
) -> tuple[dict[str, Any], bytes]:
    _require_path(path, expected_path, name=name)
    raw = _read_caller_hashed_bytes(path, expected_file_sha256, name=name)
    value = load_caller_hashed_json(path, expected_file_sha256, name=name)
    return value, raw


def _preflight_reviewed_inputs(reviewed: object) -> None:
    if not isinstance(reviewed, Mapping):
        raise ValueError("V4 gate reviewed-input binding is malformed")
    _preflight_stage_authorization(
        authorization_bound=(
            f"{CANONICAL_TRAINER_AUTHORIZATION_PATH}:"
            f"{reviewed.get('trainer_authorization_file_sha256')}"
        ),
        review_bound=(
            f"{CANONICAL_TRAINER_REVIEW_PATH}:"
            f"{reviewed.get('trainer_review_record_file_sha256')}"
        ),
    )
    if (
        reviewed.get("metric_verifier_authorization_file_sha256")
        != metric_verifier.AUTHORIZATION_FILE_SHA256
        or reviewed.get("metric_verifier_authorization_content_sha256")
        != metric_verifier.AUTHORIZATION_CONTENT_SHA256
    ):
        raise PermissionError("V4 metric verifier authorization binding changed")
    metric_verifier.preflight_metric_verifier_authorization(
        metric_verifier.CANONICAL_AUTHORIZATION_PATH,
        metric_verifier.AUTHORIZATION_FILE_SHA256,
    )


def verify_canonical_stage_gate(
    path: Path,
    expected_file_sha256: str,
    *,
    expected_seed: int,
    expected_fit_size: int,
    _memo: dict[tuple[str, str], dict[str, Any]] | None = None,
    _active: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Reopen, fully reverify, and byte-recompute one canonical stage chain."""

    _require_captured_private_finalizer()

    memo = {} if _memo is None else _memo
    active = set() if _active is None else _active
    expected_path = canonical_stage_gate_path(expected_seed, expected_fit_size)
    key = (str(expected_path), expected_file_sha256)
    if key in memo:
        return memo[key]
    if key in active:
        raise PermissionError("V4 stage-gate dependency cycle detected")
    active.add(key)
    try:
        stage, raw = _load_canonical_gate_record(
            path,
            expected_file_sha256,
            expected_path=expected_path,
            name=f"V4 seed-{expected_seed} N{expected_fit_size} stage gate",
        )
        if (
            stage.get("seed") != expected_seed
            or stage.get("fit_size") != expected_fit_size
        ):
            raise ValueError("V4 canonical stage gate seed/rung changed")
        _preflight_reviewed_inputs(stage.get("reviewed_inputs"))
        gate._validate_stage_gate_schema(stage)

        stage_index = LADDER_FIT_SIZES.index(expected_fit_size)
        previous = None
        if stage_index:
            previous_size = LADDER_FIT_SIZES[stage_index - 1]
            previous_binding = stage.get("previous_stage_gate")
            if not isinstance(previous_binding, Mapping):
                raise PermissionError("V4 canonical stage lost its predecessor")
            previous = verify_canonical_stage_gate(
                canonical_stage_gate_path(expected_seed, previous_size),
                str(previous_binding.get("file_sha256")),
                expected_seed=expected_seed,
                expected_fit_size=previous_size,
                _memo=memo,
                _active=active,
            )
            if gate._stage_gate_binding(previous) != previous_binding:
                raise PermissionError("V4 canonical predecessor binding changed")

        first_seed = None
        if expected_seed == 20260711:
            first_binding = stage.get("seed_20260710_gate")
            if not isinstance(first_binding, Mapping):
                raise PermissionError("V4 second-seed stage lost the first seed")
            first_seed = verify_canonical_seed_gate(
                canonical_seed_gate_path(20260710),
                str(first_binding.get("file_sha256")),
                expected_seed=20260710,
                _memo=memo,
                _active=active,
            )
            if gate._seed_gate_binding(first_seed) != first_binding:
                raise PermissionError("V4 canonical first-seed binding changed")

        artifacts = stage["artifacts"]
        attempt = CANONICAL_ATTEMPT_ROOT / f"seed_{expected_seed}" / f"n{expected_fit_size}"
        reviewed = stage["reviewed_inputs"]
        result, artifact_binding, metric_receipt = _validate_stage_artifact_bundle(
            seed=expected_seed,
            fit_size=expected_fit_size,
            reservation_bound=_bound(attempt / "reservation.json", artifacts["reservation"]),
            result_bound=_bound(attempt / "result.json", artifacts["result"]),
            checkpoint_bound=_bound(attempt / "checkpoint.pt", artifacts["checkpoint"]),
            completion_bound=_bound(attempt / "completed.json", artifacts["completion"]),
            metric_verification_bound=_bound(
                canonical_metric_receipt_path(expected_seed, expected_fit_size),
                artifacts["metric_verification"],
            ),
            authorization_bound=(
                f"{CANONICAL_TRAINER_AUTHORIZATION_PATH}:"
                f"{reviewed['trainer_authorization_file_sha256']}"
            ),
            review_bound=(
                f"{CANONICAL_TRAINER_REVIEW_PATH}:"
                f"{reviewed['trainer_review_record_file_sha256']}"
            ),
        )
        if artifact_binding != artifacts:
            raise ValueError("V4 canonical stage artifact binding changed")
        recomputed = gate.finalize_development_fit_stage_v4(
            result,
            expected_seed=expected_seed,
            artifact_binding=artifact_binding,
            metric_verification_receipt=metric_receipt,
            previous_stage_gate=previous,
            seed_20260710_gate=first_seed,
        )
        if raw != _canonical_json_bytes(recomputed) + b"\n":
            raise ValueError("V4 canonical stage gate is not byte-reproducible")
        memo[key] = stage
        return stage
    finally:
        active.discard(key)


def verify_canonical_seed_gate(
    path: Path,
    expected_file_sha256: str,
    *,
    expected_seed: int,
    _memo: dict[tuple[str, str], dict[str, Any]] | None = None,
    _active: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Reopen and reverify all four canonical stage chains for one seed."""

    _require_captured_private_finalizer()

    memo = {} if _memo is None else _memo
    active = set() if _active is None else _active
    expected_path = canonical_seed_gate_path(expected_seed)
    key = (str(expected_path), expected_file_sha256)
    if key in memo:
        return memo[key]
    if key in active:
        raise PermissionError("V4 seed-gate dependency cycle detected")
    active.add(key)
    try:
        seed_gate, raw = _load_canonical_gate_record(
            path,
            expected_file_sha256,
            expected_path=expected_path,
            name=f"V4 seed-{expected_seed} gate",
        )
        if seed_gate.get("seed") != expected_seed:
            raise ValueError("V4 canonical seed gate changed seed")
        _preflight_reviewed_inputs(seed_gate.get("reviewed_inputs"))
        gate._validate_seed_gate_schema(seed_gate)
        stage_hashes = seed_gate["stage_gate_file_sha256"]
        stages = [
            verify_canonical_stage_gate(
                canonical_stage_gate_path(expected_seed, fit_size),
                str(stage_hash),
                expected_seed=expected_seed,
                expected_fit_size=fit_size,
                _memo=memo,
                _active=active,
            )
            for fit_size, stage_hash in zip(LADDER_FIT_SIZES, stage_hashes)
        ]
        first_seed = None
        if expected_seed == 20260711:
            first_binding = seed_gate.get("seed_20260710_gate")
            if not isinstance(first_binding, Mapping):
                raise PermissionError("V4 second seed gate lost the first seed")
            first_seed = verify_canonical_seed_gate(
                canonical_seed_gate_path(20260710),
                str(first_binding.get("file_sha256")),
                expected_seed=20260710,
                _memo=memo,
                _active=active,
            )
            if gate._seed_gate_binding(first_seed) != first_binding:
                raise PermissionError("V4 first-seed gate binding changed")
        recomputed = gate.finalize_development_fit_seed_v4(
            stages,
            expected_seed=expected_seed,
            seed_20260710_gate=first_seed,
        )
        if raw != _canonical_json_bytes(recomputed) + b"\n":
            raise ValueError("V4 canonical seed gate is not byte-reproducible")
        memo[key] = seed_gate
        return seed_gate
    finally:
        active.discard(key)


def validate_canonical_stage_gate_for_execution(
    path: Path,
    expected_file_sha256: str,
    *,
    expected_seed: int,
    expected_next_fit_size: int,
) -> dict[str, Any]:
    _require_captured_private_finalizer()
    previous_index = LADDER_FIT_SIZES.index(expected_next_fit_size) - 1
    if previous_index < 0:
        raise ValueError("N5 cannot consume a previous-stage gate")
    expected_fit_size = LADDER_FIT_SIZES[previous_index]
    value = verify_canonical_stage_gate(
        path,
        expected_file_sha256,
        expected_seed=expected_seed,
        expected_fit_size=expected_fit_size,
    )
    return gate._validate_stage_execution_fields(
        value,
        gate_file_sha256=expected_file_sha256,
        expected_seed=expected_seed,
        expected_next_fit_size=expected_next_fit_size,
    )


def validate_canonical_seed_gate_for_execution(
    path: Path,
    expected_file_sha256: str,
) -> dict[str, Any]:
    _require_captured_private_finalizer()
    value = verify_canonical_seed_gate(
        path,
        expected_file_sha256,
        expected_seed=20260710,
    )
    return gate._validate_seed_execution_fields(
        value,
        gate_file_sha256=expected_file_sha256,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    stage = subparsers.add_parser("stage")
    stage.add_argument("--reservation", required=True, help="PATH:SHA256")
    stage.add_argument("--result", required=True, help="PATH:SHA256")
    stage.add_argument("--checkpoint", required=True, help="PATH:SHA256")
    stage.add_argument("--completion", required=True, help="PATH:SHA256")
    stage.add_argument("--metric-verification", required=True, help="PATH:SHA256")
    stage.add_argument("--trainer-authorization", required=True, help="PATH:SHA256")
    stage.add_argument("--trainer-review-record", required=True, help="PATH:SHA256")
    stage.add_argument("--seed", type=int, choices=EXPECTED_SEEDS, required=True)
    stage.add_argument("--fit-size", type=int, choices=LADDER_FIT_SIZES, required=True)
    stage.add_argument("--previous-stage-gate")
    stage.add_argument("--seed-20260710-gate")

    seed = subparsers.add_parser("seed")
    seed.add_argument("--stage-gate", action="append", required=True, help="PATH:SHA256")
    seed.add_argument("--seed", type=int, choices=EXPECTED_SEEDS, required=True)
    seed.add_argument("--seed-20260710-gate")

    combined = subparsers.add_parser("two-seed")
    combined.add_argument("--seed-20260710-gate", required=True)
    combined.add_argument("--seed-20260711-gate", required=True)
    return parser.parse_args(argv)


def _load_canonical_gate(
    bound: str,
    *,
    expected_path: Path,
    name: str,
) -> dict[str, Any]:
    path, _digest, value = _load_bound(bound, name=name)
    _require_path(path, expected_path, name=name)
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    _require_captured_private_finalizer()
    if args.mode == "stage":
        result, artifacts, metric_receipt = _validate_stage_artifact_bundle(
            seed=args.seed,
            fit_size=args.fit_size,
            reservation_bound=args.reservation,
            result_bound=args.result,
            checkpoint_bound=args.checkpoint,
            completion_bound=args.completion,
            metric_verification_bound=args.metric_verification,
            authorization_bound=args.trainer_authorization,
            review_bound=args.trainer_review_record,
        )
        fit_size = args.fit_size
        previous = None
        if fit_size != LADDER_FIT_SIZES[0]:
            previous_size = LADDER_FIT_SIZES[
                LADDER_FIT_SIZES.index(fit_size) - 1
            ]
            if args.previous_stage_gate is None:
                raise PermissionError("larger V4 rung lacks its canonical predecessor")
            previous_path, previous_sha = _parse_bound_path(args.previous_stage_gate)
            previous = verify_canonical_stage_gate(
                previous_path,
                previous_sha,
                expected_seed=args.seed,
                expected_fit_size=previous_size,
            )
        elif args.previous_stage_gate is not None:
            raise ValueError("V4 N5 may not bind a predecessor")
        first_seed = None
        if args.seed == 20260711:
            if args.seed_20260710_gate is None:
                raise PermissionError("second V4 seed lacks the first-seed gate")
            first_path, first_sha = _parse_bound_path(args.seed_20260710_gate)
            first_seed = verify_canonical_seed_gate(
                first_path,
                first_sha,
                expected_seed=20260710,
            )
        elif args.seed_20260710_gate is not None:
            raise ValueError("first V4 seed may not bind itself")
        finalized = gate.finalize_development_fit_stage_v4(
            result,
            expected_seed=args.seed,
            artifact_binding=artifacts,
            metric_verification_receipt=metric_receipt,
            previous_stage_gate=previous,
            seed_20260710_gate=first_seed,
        )
        output = canonical_stage_gate_path(args.seed, fit_size)
    elif args.mode == "seed":
        if len(args.stage_gate) != len(LADDER_FIT_SIZES):
            raise ValueError("V4 seed finalizer requires exactly four stage gates")
        stages = []
        for bound, fit_size in zip(args.stage_gate, LADDER_FIT_SIZES):
            stage_path, stage_sha = _parse_bound_path(bound)
            stages.append(
                verify_canonical_stage_gate(
                    stage_path,
                    stage_sha,
                    expected_seed=args.seed,
                    expected_fit_size=fit_size,
                )
            )
        first_seed = None
        if args.seed == 20260711:
            if args.seed_20260710_gate is None:
                raise PermissionError("second V4 seed lacks the first-seed gate")
            first_path, first_sha = _parse_bound_path(args.seed_20260710_gate)
            first_seed = verify_canonical_seed_gate(
                first_path,
                first_sha,
                expected_seed=20260710,
            )
        elif args.seed_20260710_gate is not None:
            raise ValueError("first V4 seed may not bind itself")
        finalized = gate.finalize_development_fit_seed_v4(
            stages,
            expected_seed=args.seed,
            seed_20260710_gate=first_seed,
        )
        output = canonical_seed_gate_path(args.seed)
    elif args.mode == "two-seed":
        first_path, first_sha = _parse_bound_path(args.seed_20260710_gate)
        first = verify_canonical_seed_gate(
            first_path,
            first_sha,
            expected_seed=20260710,
        )
        second_path, second_sha = _parse_bound_path(args.seed_20260711_gate)
        second = verify_canonical_seed_gate(
            second_path,
            second_sha,
            expected_seed=20260711,
        )
        finalized = gate.finalize_development_fit_two_seed_v4(first, second)
        output = canonical_two_seed_gate_path()
    else:
        raise AssertionError("unreachable V4 finalizer mode")
    return write_gate_exclusive(output, finalized, enforce_canonical_root=True)


def _cli_authority_bounds(args: argparse.Namespace) -> tuple[str, str]:
    if args.mode == "stage":
        return str(args.trainer_authorization), str(args.trainer_review_record)
    if args.mode == "seed":
        first_path, first_sha = _parse_bound_path(args.stage_gate[0])
        _require_path(
            first_path,
            canonical_stage_gate_path(args.seed, LADDER_FIT_SIZES[0]),
            name="V4 first stage gate",
        )
        first = load_caller_hashed_json(first_path, first_sha, name="V4 first stage gate")
    elif args.mode == "two-seed":
        first_path, first_sha = _parse_bound_path(args.seed_20260710_gate)
        _require_path(
            first_path,
            canonical_seed_gate_path(EXPECTED_SEEDS[0]),
            name="V4 first seed gate",
        )
        first = load_caller_hashed_json(first_path, first_sha, name="V4 first seed gate")
    else:
        raise AssertionError("unreachable V4 finalizer mode")
    reviewed = first.get("reviewed_inputs")
    if not isinstance(reviewed, Mapping):
        raise PermissionError("V4 gate lacks reviewed authorization bindings")
    return (
        f"{CANONICAL_TRAINER_AUTHORIZATION_PATH}:"
        f"{reviewed.get('trainer_authorization_file_sha256')}",
        f"{CANONICAL_TRAINER_REVIEW_PATH}:"
        f"{reviewed.get('trainer_review_record_file_sha256')}",
    )


def _captured_finalizer_cli(args: argparse.Namespace) -> int:
    _require_captured_private_finalizer()
    receipt = run(args)
    print((_canonical_json_bytes(receipt) + b"\n").decode("ascii"), end="")
    return 0


def _dispatch_captured_finalizer_cli(args: argparse.Namespace) -> int:
    if __name__ != "__main__":
        raise PermissionError("V4 finalizer library execution is unsupported")
    authorization_bound, review_bound = _cli_authority_bounds(args)
    authorization_path, authorization_sha = _parse_bound_path(authorization_bound)
    review_path, review_sha = _parse_bound_path(review_bound)
    _require_path(
        authorization_path,
        CANONICAL_TRAINER_AUTHORIZATION_PATH,
        name="V4 trainer authorization",
    )
    _require_path(
        review_path,
        CANONICAL_TRAINER_REVIEW_PATH,
        name="V4 trainer review record",
    )
    authorization = load_caller_hashed_json(
        authorization_path,
        authorization_sha,
        name="V4 trainer authorization",
    )
    source_map = authorization.get("source_map")
    entries = source_map.get("entries") if isinstance(source_map, Mapping) else None
    if not isinstance(entries, list):
        raise PermissionError("V4 trainer source map is unavailable")
    normalized = []
    launcher_sha = None
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "role", "sha256"}:
            raise ValueError("V4 trainer source-map entry is malformed")
        normalized.append(dict(entry))
        if entry.get("path") == LAUNCHER_RELATIVE_PATH:
            launcher_sha = entry.get("sha256")
    if (
        not _is_sha256(launcher_sha)
        or source_map.get("source_map_sha256")
        != hashlib.sha256(_canonical_json_bytes(normalized)).hexdigest()
    ):
        raise ValueError("V4 trainer source-map binding changed")
    launcher_path = (ROOT / LAUNCHER_RELATIVE_PATH).resolve(strict=True)
    launcher_raw = _read_caller_hashed_bytes(
        launcher_path,
        str(launcher_sha),
        name="V4 launcher source",
    )
    launcher_name = f"_lewm_v4_ca_launcher_{launcher_sha}_{uuid.uuid4().hex}"
    launcher = types.ModuleType(launcher_name)
    launcher.__file__ = str(launcher_path)
    launcher.__cached__ = None
    launcher.__verified_logical_name__ = LAUNCHER_LOGICAL_NAME
    launcher.__verified_source_sha256__ = launcher_sha
    sys.modules[launcher_name] = launcher
    try:
        exec(
            compile(
                launcher_raw,
                f"v4ca://{launcher_sha}/{LAUNCHER_RELATIVE_PATH}",
                "exec",
            ),
            launcher.__dict__,
        )
    except BaseException:
        sys.modules.pop(launcher_name, None)
        raise
    preauth = launcher.preflight_exact_authorization(
        dataset_path=launcher.CANONICAL_DATASET_PATH,
        dataset_file_sha256=launcher.DATASET_MANIFEST_FILE_SHA256,
        audit_path=launcher.CANONICAL_AUDIT_PATH,
        audit_file_sha256=launcher.AUDIT_RECEIPT_FILE_SHA256,
        authorization_path=authorization_path,
        authorization_file_sha256=authorization_sha,
        review_record_path=review_path,
        review_record_file_sha256=review_sha,
    )

    # The complete canonical authority has passed. Runtime classes and source
    # capture remain local to this terminal and are never attached or returned.
    import builtins
    import importlib
    import importlib.abc
    import importlib.util

    runtime_relatives = (
        "lewm/__init__.py",
        "lewm/benchmarks/__init__.py",
        "lewm/benchmarks/counterfactual.py",
        "lewm/benchmarks/go2_observable_camera_ray_evidence_v4.py",
        "lewm/benchmarks/go2_observable_camera_ray_fit_v4_ladder_gate.py",
        "lewm/benchmarks/go2_observable_camera_ray_fit_v4_metrics.py",
        "lewm/models/__init__.py",
        "lewm/models/encoders.py",
        "lewm/models/lewm.py",
        "lewm/models/observable_camera_ray_evidence_v4.py",
        "lewm/models/observable_camera_ray_evidence_v4_training.py",
        "lewm/models/phase2d_spatial_lewm.py",
        "lewm/models/predictor.py",
        "lewm/models/primitive_affordance.py",
        "lewm/models/sigreg.py",
        "lewm/models/source_action_utility.py",
        "lewm/models/spatial_lewm.py",
        "lewm/models/spatial_predictor.py",
        "scripts/finalize_go2_observable_camera_ray_fit_v4_ladder.py",
        "scripts/launch_go2_observable_camera_ray_fit_v4.py",
        "scripts/train_go2_observable_camera_ray_fit_v4.py",
        "scripts/verify_go2_observable_camera_ray_fit_v4_metrics.py",
        "scripts/verify_go2_observable_camera_ray_fit_v4_target_partitions.py",
    )

    def logical_name(relative: str) -> str:
        value = relative[:-3].replace("/", ".")
        return value[:-9] if value.endswith(".__init__") else value

    entry_hashes = {
        str(row["path"]): str(row["sha256"])
        for row in preauth["source_map"]["entries"]
    }
    captured: dict[str, tuple[Path, bytes, str]] = {}
    for relative in runtime_relatives:
        path = (ROOT / relative).resolve(strict=True)
        source = _read_caller_hashed_bytes(
            path,
            entry_hashes.get(relative, ""),
            name=f"V4 finalizer runtime source {relative}",
        )
        captured[logical_name(relative)] = (
            path,
            source,
            hashlib.sha256(source).hexdigest(),
        )
    preloaded = sorted(name for name in captured if name in sys.modules)
    if preloaded:
        raise PermissionError(f"V4 finalizer runtime modules were preloaded: {preloaded}")
    allowed_roots = frozenset({*sys.stdlib_module_names, "PIL", "numpy", "torch"})
    namespace = f"_lewm_v4_ca_{preauth['source_map_sha256'][:12]}_{uuid.uuid4().hex}"

    class Loader(importlib.abc.Loader):
        def __init__(self, logical: str, record: tuple[Path, bytes, str]) -> None:
            self.logical = logical
            self.path, self.source, self.digest = record

        def create_module(self, spec: Any) -> None:
            return None

        def exec_module(self, module: types.ModuleType) -> None:
            module.__file__ = str(self.path)
            module.__cached__ = None
            module.__verified_source_sha256__ = self.digest
            module.__verified_logical_name__ = self.logical
            verified_builtins = dict(vars(builtins))
            verified_builtins["__import__"] = finder.verified_import
            module.__builtins__ = verified_builtins
            filename = f"v4ca://{self.digest}/{self.logical.replace('.', '/')}"
            exec(compile(self.source, filename, "exec", dont_inherit=True), module.__dict__)

    class Finder(importlib.abc.MetaPathFinder):
        def synthetic(self, logical: str) -> str:
            return f"{namespace}.{logical}"

        def verified_import(
            self,
            name: str,
            globals: Mapping[str, Any] | None = None,
            locals: Mapping[str, Any] | None = None,
            fromlist: Sequence[str] = (),
            level: int = 0,
        ) -> Any:
            if level:
                return builtins.__import__(name, globals, locals, fromlist, level)
            tracked = name in captured or any(key.startswith(f"{name}.") for key in captured)
            if not tracked:
                if name.split(".", 1)[0] not in allowed_roots:
                    raise ImportError(f"V4 finalizer runtime import is not whitelisted: {name}")
                return builtins.__import__(name, globals, locals, fromlist, level)
            translated = self.synthetic(name)
            builtins.__import__(translated, globals, locals, fromlist, 0)
            if fromlist:
                return sys.modules[translated]
            return sys.modules[self.synthetic(name.split(".", 1)[0])]

        def find_spec(self, fullname: str, path: object = None, target: object = None) -> Any:
            prefix = f"{namespace}."
            logical = fullname[len(prefix) :] if fullname.startswith(prefix) else None
            record = captured.get(logical or "")
            if record is None:
                return None
            return importlib.util.spec_from_loader(
                fullname,
                Loader(str(logical), record),
                origin=str(record[0]),
                is_package=record[0].name == "__init__.py",
            )

    finder = Finder()
    root = types.ModuleType(namespace)
    root.__path__ = []
    root.__package__ = namespace
    sys.modules[namespace] = root
    scripts_name = finder.synthetic("scripts")
    scripts_package = types.ModuleType(scripts_name)
    scripts_package.__path__ = []
    scripts_package.__package__ = scripts_name
    sys.modules[scripts_name] = scripts_package
    sys.meta_path.insert(0, finder)
    fingerprints: dict[str, str] = {}

    def load(logical: str) -> types.ModuleType:
        module = importlib.import_module(finder.synthetic(logical))
        path, source, digest = captured[logical]
        if (
            module is not sys.modules.get(finder.synthetic(logical))
            or getattr(module, "__verified_logical_name__", None) != logical
            or getattr(module, "__verified_source_sha256__", None) != digest
            or Path(str(getattr(module, "__file__", ""))).resolve(strict=True) != path
            or hashlib.sha256(source).hexdigest() != digest
        ):
            raise PermissionError(f"V4 finalizer loaded module identity changed: {logical}")
        fingerprint = launcher._module_code_sha256(module)
        if fingerprints.setdefault(logical, fingerprint) != fingerprint:
            raise PermissionError(f"V4 finalizer loaded module code changed: {logical}")
        return module

    logical_self = "scripts.finalize_go2_observable_camera_ray_fit_v4_ladder"
    try:
        private = load(logical_self)
        live = sys.modules.get(__name__)
        if not isinstance(live, types.ModuleType) or (
            launcher._module_code_sha256(live)
            != launcher._module_code_sha256(private)
        ):
            raise PermissionError("V4 live finalizer differs from captured source")
        result = int(private._captured_finalizer_cli(args))
        for logical in tuple(fingerprints):
            load(logical)
        return result
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        for name in tuple(sys.modules):
            if name == namespace or name.startswith(f"{namespace}."):
                sys.modules.pop(name, None)
        sys.modules.pop(launcher_name, None)


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not sys.flags.isolated:
        if __name__ != "__main__":
            raise PermissionError("V4 finalizer library execution is unsupported")
        environment = dict(os.environ)
        for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE"):
            environment.pop(name, None)
        environment["PYTHONNOUSERSITE"] = "1"
        return int(
            subprocess.run(
                [sys.executable, "-I", "-B", str(Path(__file__).resolve()), *raw_argv],
                cwd=ROOT,
                env=environment,
                check=False,
            ).returncode
        )
    return _dispatch_captured_finalizer_cli(parse_args(raw_argv))


if __name__ == "__main__":
    raise SystemExit(main())
