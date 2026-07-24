"""Bound runner and immutable raw-outcome contract for Shared JEPA V5."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping, Sequence
import weakref


RAW_SCENE_OUTCOME_SCHEMA = "lewm_go2_shared_jepa_raw_scene_outcome_v6"
RUNNER_LEDGER_SCHEMA = "lewm_go2_shared_jepa_runner_access_ledger_v6"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _canonical_relative_path(value: object, *, name: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError(f"{name} must be a canonical relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must be a canonical relative path")
    if path.as_posix() != value:
        raise ValueError(f"{name} must be a canonical relative path")
    return value


def _parse_canonical_json(encoded: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict) or encoded != _canonical_bytes(value) + b"\n":
        raise ValueError(f"{name} is not canonical newline-terminated JSON")
    claimed = _require_sha256(value.get("content_sha256"), name=f"{name} content")
    core = dict(value)
    del core["content_sha256"]
    if _canonical_sha256(core) != claimed:
        raise ValueError(f"{name} content hash changed")
    return value


def _read_fixed_repository_file(
    relative_path: str,
    *,
    expected_file_sha256: str,
    name: str,
) -> bytes:
    from lewm.models.shared_observable_camera_ray_jepa_v5_authority import (
        require_frozen_production_authority,
    )

    authority = require_frozen_production_authority()
    root = Path(authority["repository_root"])
    relative = _canonical_relative_path(relative_path, name=f"{name} path")
    _require_sha256(expected_file_sha256, name=f"{name} file hash")
    root_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    root_fd = os.open(root, root_flags)
    current_fd = root_fd
    try:
        parts = PurePosixPath(relative).parts
        for part in parts[:-1]:
            next_fd = os.open(part, root_flags, dir_fd=current_fd)
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = next_fd
        file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        file_fd = os.open(parts[-1], file_flags, dir_fd=current_fd)
        try:
            metadata = os.fstat(file_fd)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise PermissionError(f"{name} is not a singly-linked regular file")
            chunks = []
            while True:
                chunk = os.read(file_fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            os.close(file_fd)
    finally:
        if current_fd != root_fd:
            os.close(current_fd)
        os.close(root_fd)
    encoded = b"".join(chunks)
    if hashlib.sha256(encoded).hexdigest() != expected_file_sha256:
        raise ValueError(f"{name} file hash changed")
    return encoded


@dataclass(frozen=True, init=False, slots=True, weakref_slot=True, eq=False)
class _RemovedCanonicalRunnerBatchV6:
    """Immutable bytes reopened from the canonical production runner files."""

    gate: str
    role_manifest_bytes: bytes
    scene_outcome_bytes: tuple[bytes, ...]
    scene_file_bindings: tuple[tuple[str, str, str], ...]
    runner_ledger_bytes: bytes
    expected_model_state_sha256: str
    expected_checkpoint_file_sha256: str
    expected_runner_source_sha256: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise PermissionError("canonical runner batches are fixed-file issued")


@dataclass(frozen=True, init=False, slots=True, weakref_slot=True, eq=False)
class SyntheticRunnerBatchV6:
    """Irreversibly non-production batch used only by CPU unit tests."""

    gate: str
    role_manifest_bytes: bytes
    scene_outcome_bytes: tuple[bytes, ...]
    scene_file_bindings: tuple[tuple[str, str, str], ...]
    runner_ledger_bytes: bytes
    expected_model_state_sha256: str
    expected_checkpoint_file_sha256: str
    expected_runner_source_sha256: str
    _synthetic_fixture_marker: bytes

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise PermissionError("synthetic runner batches are test-fixture issued")


@dataclass(frozen=True, slots=True)
class _NormalizedBatchMaterial:
    gate: str
    role_manifest_bytes: bytes
    scene_outcome_bytes: tuple[bytes, ...]
    scene_file_bindings: tuple[tuple[str, str, str], ...]
    runner_ledger_bytes: bytes
    expected_model_state_sha256: str
    expected_checkpoint_file_sha256: str
    expected_runner_source_sha256: str


def _batch_commitment(batch: _RemovedCanonicalRunnerBatchV6 | SyntheticRunnerBatchV6) -> str:
    digest = hashlib.sha256(b"lewm_go2_v5_runner_batch_issuance_v2")
    for value in (
        type(batch).__name__.encode("ascii"),
        batch.gate.encode("ascii"),
        batch.role_manifest_bytes,
        *batch.scene_outcome_bytes,
        _canonical_bytes(batch.scene_file_bindings),
        batch.runner_ledger_bytes,
        batch.expected_model_state_sha256.encode("ascii"),
        batch.expected_checkpoint_file_sha256.encode("ascii"),
        batch.expected_runner_source_sha256.encode("ascii"),
        (
            batch._synthetic_fixture_marker
            if type(batch) is SyntheticRunnerBatchV6
            else b"canonical-fixed-file"
        ),
    ):
        digest.update(len(value).to_bytes(8, "little"))
        digest.update(value)
    return digest.hexdigest()


def _set_batch_material(
    batch: _RemovedCanonicalRunnerBatchV6 | SyntheticRunnerBatchV6,
    material: _NormalizedBatchMaterial,
) -> None:
    for field_name in material.__dataclass_fields__:
        object.__setattr__(batch, field_name, getattr(material, field_name))


def _decode_batch_payload(
    batch: _RemovedCanonicalRunnerBatchV6 | SyntheticRunnerBatchV6,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    role_manifest = _parse_canonical_json(
        batch.role_manifest_bytes,
        name="issued dataset role manifest",
    )
    outcomes = [
        _parse_canonical_json(encoded, name="issued raw scene outcome")
        for encoded in batch.scene_outcome_bytes
    ]
    ledger = _parse_canonical_json(
        batch.runner_ledger_bytes,
        name="issued runner ledger",
    )
    bindings = {
        scene_id: {"path": path, "file_sha256": file_sha256}
        for scene_id, path, file_sha256 in batch.scene_file_bindings
    }
    return role_manifest, outcomes, ledger, bindings


def source_file_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _normalize_scene_outcome(
    value: Mapping[str, object],
    *,
    gate: str,
    metric_names: Sequence[str],
    expected_model_state_sha256: str,
    expected_checkpoint_file_sha256: str,
    expected_runner_source_sha256: str,
) -> dict[str, Any]:
    encoded = _canonical_bytes(dict(value)) + b"\n"
    outcome = _parse_canonical_json(encoded, name="raw scene outcome")
    if set(outcome) != {
        "schema",
        "gate",
        "scene_id",
        "family",
        "model_state_sha256",
        "evaluated_checkpoint_file_sha256",
        "runner_source_sha256",
        "instances",
        "content_sha256",
    } or outcome.get("schema") != RAW_SCENE_OUTCOME_SCHEMA:
        raise ValueError("raw scene outcome fields changed")
    if (
        outcome.get("gate") != gate
        or outcome.get("model_state_sha256") != expected_model_state_sha256
        or outcome.get("evaluated_checkpoint_file_sha256")
        != expected_checkpoint_file_sha256
        or outcome.get("runner_source_sha256") != expected_runner_source_sha256
        or type(outcome.get("scene_id")) is not str
        or not outcome["scene_id"]
        or type(outcome.get("family")) is not str
        or not outcome["family"]
    ):
        raise PermissionError("raw scene outcome authority changed")
    instances = outcome.get("instances")
    if not isinstance(instances, list) or not instances:
        raise ValueError("raw scene outcome instances are empty")
    identifiers = []
    for instance in instances:
        if not isinstance(instance, Mapping) or set(instance) != {
            "instance_id",
            "inference_output_sha256",
            "metric_outcomes",
        }:
            raise ValueError("raw inference event fields changed")
        identifier = instance.get("instance_id")
        metrics = instance.get("metric_outcomes")
        _require_sha256(
            instance.get("inference_output_sha256"),
            name="inference output",
        )
        if (
            type(identifier) is not str
            or not identifier
            or not isinstance(metrics, Mapping)
            or set(metrics) != set(metric_names)
            or any(type(item) is not bool for item in metrics.values())
        ):
            raise ValueError("raw inference metric event changed")
        identifiers.append(identifier)
    if identifiers != sorted(identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("raw inference instances are not sorted and unique")
    return outcome


def _normalize_batch_material(
    *,
    gate: str,
    metric_names: Sequence[str],
    role_manifest: Mapping[str, object],
    scene_outcomes: Sequence[Mapping[str, object]],
    scene_paths: Sequence[str],
    expected_model_state_sha256: str,
    expected_checkpoint_file_sha256: str,
    expected_runner_source_sha256: str,
    expected_ledger: Mapping[str, object] | None,
) -> _NormalizedBatchMaterial:
    if gate not in {"g2", "g3"}:
        raise ValueError("gate must be g2 or g3")
    for value, name in (
        (expected_model_state_sha256, "model state"),
        (expected_checkpoint_file_sha256, "checkpoint"),
        (expected_runner_source_sha256, "runner source"),
    ):
        _require_sha256(value, name=name)
    role_bytes = _canonical_bytes(dict(role_manifest)) + b"\n"
    _parse_canonical_json(role_bytes, name="dataset role manifest")
    normalized = [
        _normalize_scene_outcome(
            item,
            gate=gate,
            metric_names=metric_names,
            expected_model_state_sha256=expected_model_state_sha256,
            expected_checkpoint_file_sha256=expected_checkpoint_file_sha256,
            expected_runner_source_sha256=expected_runner_source_sha256,
        )
        for item in scene_outcomes
    ]
    if len(scene_paths) != len(normalized):
        raise ValueError("scene path/outcome cardinality changed")
    scene_bytes = tuple(_canonical_bytes(item) + b"\n" for item in normalized)
    bindings = tuple(
        (
            str(item["scene_id"]),
            _canonical_relative_path(path, name="scene outcome path"),
            hashlib.sha256(encoded).hexdigest(),
        )
        for item, path, encoded in zip(normalized, scene_paths, scene_bytes, strict=True)
    )
    events = [
        {
            "sequence": sequence,
            "scene_id": scene_id,
            "role": gate,
            "operation": "read_and_evaluate_canonical_scene",
            "path": path,
            "file_sha256": file_sha256,
            "instance_count": len(normalized[sequence - 1]["instances"]),
            "forbidden": False,
        }
        for sequence, (scene_id, path, file_sha256) in enumerate(bindings, start=1)
    ]
    role_value = _parse_canonical_json(role_bytes, name="dataset role manifest")
    ledger_core = {
        "schema": RUNNER_LEDGER_SCHEMA,
        "gate": gate,
        "dataset_role_manifest_content_sha256": role_value["content_sha256"],
        "runner_source_sha256": expected_runner_source_sha256,
        "events": events,
    }
    ledger = {**ledger_core, "content_sha256": _canonical_sha256(ledger_core)}
    if expected_ledger is not None and dict(expected_ledger) != ledger:
        raise PermissionError("bound runner ledger did not reproduce actual opens")
    return _NormalizedBatchMaterial(
        gate=gate,
        role_manifest_bytes=role_bytes,
        scene_outcome_bytes=scene_bytes,
        scene_file_bindings=bindings,
        runner_ledger_bytes=_canonical_bytes(ledger) + b"\n",
        expected_model_state_sha256=expected_model_state_sha256,
        expected_checkpoint_file_sha256=expected_checkpoint_file_sha256,
        expected_runner_source_sha256=expected_runner_source_sha256,
    )


def _removed_reopen_canonical_runner_batch(*args: object, **kwargs: object) -> None:
    """Removed: production evidence is emitted only by the one-shot runner CLI."""

    raise PermissionError("production runner batches were removed; use the one-shot CLI")


def _removed_validated_runner_batch_payload(*args: object, **kwargs: object) -> None:
    """Removed with the in-process production batch capability."""

    raise PermissionError("production runner batches were removed; use the one-shot CLI")


def _install_synthetic_runner_test_api():
    issued: weakref.WeakKeyDictionary[SyntheticRunnerBatchV6, str] = (
        weakref.WeakKeyDictionary()
    )

    def issue_synthetic_runner_batch_for_tests(
        *,
        gate: str,
        metric_names: Sequence[str],
        role_manifest: Mapping[str, object],
        scene_outcomes: Sequence[Mapping[str, object]],
        expected_model_state_sha256: str,
        expected_checkpoint_file_sha256: str,
    ) -> SyntheticRunnerBatchV6:
        """Issue an exact synthetic type that no production finalizer accepts."""

        paths = [
            f"synthetic/{gate}/{item.get('scene_id')}.json"
            for item in scene_outcomes
        ]
        material = _normalize_batch_material(
            gate=gate,
            metric_names=metric_names,
            role_manifest=role_manifest,
            scene_outcomes=scene_outcomes,
            scene_paths=paths,
            expected_model_state_sha256=expected_model_state_sha256,
            expected_checkpoint_file_sha256=expected_checkpoint_file_sha256,
            expected_runner_source_sha256=source_file_sha256(),
            expected_ledger=None,
        )
        batch = object.__new__(SyntheticRunnerBatchV6)
        _set_batch_material(batch, material)
        object.__setattr__(
            batch,
            "_synthetic_fixture_marker",
            b"lewm-go2-v5-test-fixture-never-production",
        )
        issued[batch] = _batch_commitment(batch)
        return batch

    def validated_synthetic_runner_batch_payload(
        value: object,
    ) -> tuple[
        dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]
    ]:
        if type(value) is not SyntheticRunnerBatchV6 or value not in issued:
            raise PermissionError("synthetic raw outcome batch was not test-issued")
        if issued[value] != _batch_commitment(value):
            raise PermissionError("issued synthetic batch changed after issuance")
        return _decode_batch_payload(value)

    return (
        issue_synthetic_runner_batch_for_tests,
        validated_synthetic_runner_batch_payload,
    )


(
    _issue_synthetic_runner_batch_for_tests,
    _validated_synthetic_runner_batch_payload,
) = _install_synthetic_runner_test_api()
del _install_synthetic_runner_test_api


__all__ = [
    "RAW_SCENE_OUTCOME_SCHEMA",
    "RUNNER_LEDGER_SCHEMA",
    "source_file_sha256",
]
