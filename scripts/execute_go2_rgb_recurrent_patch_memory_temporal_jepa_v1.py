#!/usr/bin/env python3
"""One-shot lifecycle controller for recurrent patch-memory temporal JEPA V1.

The controller owns authority validation, attempt reservation, the exact
observation/continuation schedule, immutable publication, and terminal
receipts.  Tensor loading, training, temporal evaluation, and predecessor
retention evaluation remain in their separately reviewed modules.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


SCHEMA_PREFIX = "lewm_go2_rgb_recurrent_patch_memory_temporal_jepa_v1"
PREREGISTRATION_COMMIT = "1ac341cd97ab7a7d1a1b8c46695cf2fd3382ed60"
SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_recurrent_patch_memory_temporal_jepa_v1_"
    "source_manifest_2026-07-31.json"
)
CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_recurrent_patch_memory_temporal_jepa_v1_"
    "clean_export_certification_2026-07-31.json"
)
AUTHORITY_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_recurrent_patch_memory_temporal_jepa_v1_"
    "execution_authorization_2026-07-31.json"
)
CERTIFIED_SOURCE_ROOT = (
    "/home/andrewknowles/Workspace/"
    "LeWMQuad-v3-rgb-recurrent-patch-memory-temporal-jepa-v1-source"
)
RUNTIME_DATA_ROOT = "/home/andrewknowles/Workspace/LeWMQuad-v3"
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_recurrent_patch_memory_temporal_jepa_v1/attempt_v1"
)
METADATA_PREFLIGHT_RECEIPT_RELATIVE_PATH = (
    ".generated/go2_rgb_recurrent_patch_memory_temporal_jepa_v1_"
    "metadata_preflight/attempt_v1/receipt.json"
)
RGB_ROOT_RELATIVE_PATH = ".generated/datagen_full/render_textured_v03"

MODEL_MODULE_NAME = (
    "lewm.models.rgb_recurrent_patch_memory_temporal_jepa_v1"
)
MODEL_CLASS_NAME = "RGBRecurrentPatchMemoryTemporalJepaV1"
TRAINING_MODULE_NAME = (
    "scripts.run_go2_rgb_recurrent_patch_memory_temporal_jepa_v1"
)
EVALUATION_MODULE_NAME = (
    "scripts.evaluate_go2_rgb_recurrent_patch_memory_temporal_jepa_v1"
)
METRICS_MODULE_NAME = (
    "lewm.benchmarks.go2_rgb_recurrent_patch_memory_temporal_jepa_v1"
)
PREFLIGHT_MODULE_NAME = (
    "scripts.preflight_go2_rgb_recurrent_patch_memory_temporal_jepa_v1_metadata"
)

MAXIMUM_UPDATES = 400
MAXIMUM_PRESENTATIONS = 16_000
OBSERVATION_UPDATES = (0, 50, 100, 200, 400)
FULL_OBSERVATION_UPDATES = (0, 200, 400)
SENTINEL_OBSERVATION_UPDATES = (0, 50, 100)
CHECKPOINT_UPDATES = (200, 400)
PREDECESSOR_CONTROL_NAMES = (
    "wrong_target",
    "wrong_context",
    "position_mean",
)
RAW_HEALTH_NAMES = (
    "effective_rank",
    "cross_sample_variance",
    "within_image_spatial_diversity",
)

RUNTIME_INPUT_BINDINGS: Mapping[str, Mapping[str, Any]] = {
    "h6_train_index": {
        "path": (
            ".generated/go2_recurrent_h4_rgb_sequence_index_v2_"
            "schedule_integrity/train.jsonl"
        ),
        "file_sha256": (
            "aee2a54cddd849162648f9b8cfd54a0a28a25bd0705b6482e6af7435c85f4d77"
        ),
        "byte_count": 10_328_000,
    },
    "h6_validation_index": {
        "path": (
            ".generated/go2_recurrent_h4_rgb_sequence_index_v2_"
            "schedule_integrity/val.jsonl"
        ),
        "file_sha256": (
            "83592e2fea5927802881f076a58a9710100bea017d658c1b978ba651369beac6"
        ),
        "byte_count": 1_317_888,
    },
    "place_triplet_manifest": {
        "path": ".generated/go2_memory_role_place_triplet_index_v1/manifest.json",
        "file_sha256": (
            "a5997d93838419cabaaf8e262db70ed51f6f928195f1a312cadc4768f74ca6ca"
        ),
        "byte_count": 42_308,
    },
    "place_triplet_checkpoint_selection_index": {
        "path": (
            ".generated/go2_memory_role_place_triplet_index_v1/"
            "checkpoint_selection.jsonl"
        ),
        "file_sha256": (
            "a628a1047b6f15223a4fd7d30c5c87fa1914efef0955d70d9bd2f5330c77dcb0"
        ),
        "byte_count": 473_508,
    },
    "predecessor_scientific_result": {
        "path": (
            "docs/lewm_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1_"
            "scientific_result_2026-07-31.json"
        ),
        "file_sha256": (
            "92b2c23545b5f2b42f81b392359e7ad1c58e4773053d02df39a2df54509228ef"
        ),
        "content_sha256": (
            "59e55e2ef718c670c1251062572ef4d6bb76f7ff51582fe599d7ebe534ebefba"
        ),
        "byte_count": 7_685,
    },
    "predecessor_success": {
        "path": (
            ".generated/go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1/"
            "attempt_v1/success.json"
        ),
        "file_sha256": (
            "40e4d2e595609d4dff379634880caf122a9b1523ee555c3d7c3ae683d5e687b2"
        ),
        "content_sha256": (
            "f93162027bf4ea0f2c76c29708bd4342119c4ece010308b5d1164f591986f2fc"
        ),
        "byte_count": 3_100,
    },
    "predecessor_checkpoint": {
        "path": (
            ".generated/go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1/"
            "attempt_v1/snapshots/update_1000.pt"
        ),
        "file_sha256": (
            "f5aac23cf275d73b92ce5609a583dea89f6686a624d4889d9762740535aab873"
        ),
        "byte_count": 52_282_877,
    },
}
RUNTIME_INPUT_BINDING_NAMES = tuple(RUNTIME_INPUT_BINDINGS)

_MIGRATED_PREFIXES = (
    "encoder.",
    "predictor_blocks.",
    "predictor_norm.",
    "predictor_output.",
)
_MIGRATED_EXACT = ("predictor_position", "predictor_mask_token")
_REJECTED_PREFIXES = ("target_encoder.",)
_REJECTED_EXACT = ("ema_update_count",)

_DIR_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _content_bound(core: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(_jsonable(core))
    value.pop("content_sha256", None)
    value["content_sha256"] = hashlib.sha256(
        _canonical_json_bytes(value)
    ).hexdigest()
    return value


def validate_content_bound_v1(value: Any) -> dict[str, Any]:
    if type(value) is not dict or type(value.get("content_sha256")) is not str:
        raise TypeError("temporal V1 content-bound value must be an exact object")
    core = dict(value)
    observed = core.pop("content_sha256")
    if observed != hashlib.sha256(_canonical_json_bytes(core)).hexdigest():
        raise PermissionError("temporal V1 content binding changed")
    return dict(value)


def _safe_relative_path(value: Any) -> PurePosixPath:
    path = PurePosixPath(value) if type(value) is str else None
    folded = tuple(part.casefold() for part in path.parts) if path else ()
    if (
        path is None
        or path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(
            part == "sealed"
            or part.startswith(("sealed_", "heldout_", "held_out_"))
            for part in folded
        )
        or "sealed_test" in path.name.casefold()
    ):
        raise PermissionError("bound relative path is unsafe")
    return path


def _binding(value: Any, *, content: bool = False) -> dict[str, Any]:
    expected = {"path", "file_sha256", "byte_count"}
    if content:
        expected.add("content_sha256")
    if type(value) is not dict or set(value) != expected:
        raise TypeError("temporal V1 binding fields changed")
    result = dict(value)
    _safe_relative_path(result["path"])
    hashes = ("file_sha256", "content_sha256") if content else ("file_sha256",)
    if (
        any(
            type(result[name]) is not str
            or len(result[name]) != 64
            or any(character not in "0123456789abcdef" for character in result[name])
            for name in hashes
        )
        or type(result["byte_count"]) is not int
        or result["byte_count"] <= 0
    ):
        raise TypeError("temporal V1 binding values changed")
    return result


def _safe_certified_source_path(relative: str) -> PurePosixPath:
    value = PurePosixPath(relative)
    folded = tuple(part.casefold() for part in value.parts)
    if (
        value.is_absolute()
        or not value.parts
        or any(part in {"", ".", ".."} for part in value.parts)
        or value.suffix not in {".py", ".md", ".json"}
        or any(
            part in {".generated", "sealed", "heldout", "held_out"}
            or part.startswith(("sealed_", "heldout_", "held_out_"))
            for part in folded
        )
        or "sealed_test" in value.name.casefold()
    ):
        raise PermissionError(f"unsafe certified source path: {relative}")
    return value


def _validate_certified_source_binding(
    source_root: Path,
    value: Any,
) -> None:
    binding = _binding(value)
    relative = _safe_certified_source_path(binding["path"])
    path = source_root.joinpath(*relative.parts)
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise PermissionError(
            f"certified source is absent: {relative}"
        ) from error
    if (
        resolved != path.absolute()
        or not resolved.is_relative_to(source_root)
        or path.is_symlink()
        or not path.is_file()
    ):
        raise PermissionError(f"certified source escaped: {relative}")
    raw = path.read_bytes()
    if (
        len(raw) != binding["byte_count"]
        or hashlib.sha256(raw).hexdigest() != binding["file_sha256"]
    ):
        raise PermissionError(f"certified source changed: {relative}")


def validate_certified_source_v1(
    source_root: Path,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Rehash every certified narrow-source binding before reservation."""

    root = Path(source_root).resolve(strict=True)
    certification_path = root / CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH
    info = os.lstat(certification_path)
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise PermissionError(
            "clean-export certification must be a regular non-symlink"
        )
    raw = certification_path.read_bytes()
    certification = _decode_content_bound_json(
        raw,
        name="clean-export certification",
    )
    identity = _binding(
        authority.get("clean_export_certification"),
        content=True,
    )
    bindings = certification.get("source_bindings")
    if (
        identity["path"] != CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH
        or identity["byte_count"] != len(raw)
        or identity["file_sha256"] != hashlib.sha256(raw).hexdigest()
        or identity["content_sha256"] != certification["content_sha256"]
        or certification.get("schema")
        != f"{SCHEMA_PREFIX}_clean_export_certification_v1"
        or certification.get("status")
        != "PASS_NARROW_CLEAN_EXPORT_CERTIFIED"
        or certification.get("certified_source_root") != str(root)
        or certification.get("pinned_source_and_review_commit")
        != authority.get("pinned_source_and_review_commit")
        or type(bindings) is not list
        or not bindings
    ):
        raise PermissionError("clean-export certification identity changed")
    paths = [dict(item).get("path") for item in bindings]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise PermissionError("certified source inventory order changed")
    bindings_sha256 = hashlib.sha256(
        _canonical_json_bytes(bindings)
    ).hexdigest()
    if certification.get("bindings_sha256") != bindings_sha256:
        raise PermissionError("certified source inventory binding changed")
    for binding in bindings:
        _validate_certified_source_binding(root, binding)
    return {
        "status": "PASS_CERTIFIED_SOURCE_REHASH",
        "validated_path_count": len(bindings),
        "bindings_sha256": bindings_sha256,
        "certification_content_sha256": certification["content_sha256"],
    }


def validate_gpu_v1(torch: Any) -> dict[str, Any]:
    """Require exactly one visible HIP AMD R9700 without tensor allocation."""

    if (
        not bool(torch.cuda.is_available())
        or int(torch.cuda.device_count()) != 1
    ):
        raise RuntimeError(
            "temporal V1 requires exactly one visible AMD GPU"
        )
    hip = getattr(getattr(torch, "version", None), "hip", None)
    name = str(torch.cuda.get_device_name(0))
    normalized = name.replace(" ", "").upper()
    if (
        type(hip) is not str
        or not hip
        or "AMD" not in normalized
        or "R9700" not in normalized
    ):
        raise RuntimeError("visible GPU is not the registered AMD R9700")
    return {
        "status": "PASS_EXACTLY_ONE_VISIBLE_AMD_R9700",
        "visible_device_count": 1,
        "visible_device_name": name,
        "torch_hip_version": hip,
        "tensor_allocation_count": 0,
        "dataset_open_count": 0,
        "checkpoint_open_count": 0,
    }


_EXECUTION_ENVIRONMENT_ATTESTATION_SEAL = object()


class _ExecutionEnvironmentAttestationV1:
    __slots__ = ("receipt", "_seal")

    def __init__(self, receipt: Mapping[str, Any], seal: object) -> None:
        if seal is not _EXECUTION_ENVIRONMENT_ATTESTATION_SEAL:
            raise PermissionError(
                "execution-environment attestation was not guard-issued"
            )
        self.receipt = dict(receipt)
        self._seal = seal


def _build_execution_environment_attestation_v1(
    source_root: Path | str,
    authority: Mapping[str, Any],
    *,
    source_receipt: Mapping[str, Any],
    gpu_receipt: Mapping[str, Any],
) -> _ExecutionEnvironmentAttestationV1:
    source = dict(source_receipt)
    gpu = dict(gpu_receipt)
    if (
        source.get("status") != "PASS_CERTIFIED_SOURCE_REHASH"
        or gpu.get("status") != "PASS_EXACTLY_ONE_VISIBLE_AMD_R9700"
        or gpu.get("tensor_allocation_count") != 0
        or gpu.get("dataset_open_count") != 0
        or gpu.get("checkpoint_open_count") != 0
    ):
        raise PermissionError("execution-environment guard receipt changed")
    receipt = _content_bound(
        {
            "schema": f"{SCHEMA_PREFIX}_execution_environment_attestation_v1",
            "status": "PASS_PRE_RESERVATION_ENVIRONMENT_GUARDS",
            "authority_sha256": hashlib.sha256(
                _canonical_json_bytes(dict(authority))
            ).hexdigest(),
            "certified_source_root": str(source_root),
            "source": source,
            "gpu": gpu,
        }
    )
    return _ExecutionEnvironmentAttestationV1(
        receipt,
        _EXECUTION_ENVIRONMENT_ATTESTATION_SEAL,
    )


def _validate_execution_environment_attestation_v1(
    attestation: Any,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        type(attestation) is not _ExecutionEnvironmentAttestationV1
        or attestation._seal is not _EXECUTION_ENVIRONMENT_ATTESTATION_SEAL
    ):
        raise PermissionError(
            "execution requires a guard-issued environment attestation"
        )
    receipt = validate_content_bound_v1(attestation.receipt)
    source = receipt.get("source")
    gpu = receipt.get("gpu")
    if (
        set(receipt)
        != {
            "schema",
            "status",
            "authority_sha256",
            "certified_source_root",
            "source",
            "gpu",
            "content_sha256",
        }
        or receipt["schema"]
        != f"{SCHEMA_PREFIX}_execution_environment_attestation_v1"
        or receipt["status"]
        != "PASS_PRE_RESERVATION_ENVIRONMENT_GUARDS"
        or receipt["authority_sha256"]
        != hashlib.sha256(
            _canonical_json_bytes(dict(authority))
        ).hexdigest()
        or receipt["certified_source_root"]
        != authority.get("certified_source_root")
        or type(source) is not dict
        or source.get("status") != "PASS_CERTIFIED_SOURCE_REHASH"
        or type(gpu) is not dict
        or gpu.get("status") != "PASS_EXACTLY_ONE_VISIBLE_AMD_R9700"
        or gpu.get("tensor_allocation_count") != 0
        or gpu.get("dataset_open_count") != 0
        or gpu.get("checkpoint_open_count") != 0
    ):
        raise PermissionError(
            "execution-environment attestation identity changed"
        )
    return receipt


def validate_execution_environment_v1(
    source_root: Path,
    authority: Mapping[str, Any],
    *,
    torch_module: Any | None = None,
) -> _ExecutionEnvironmentAttestationV1:
    """Enforce both pre-reservation guards for every execution entry path."""

    source = validate_certified_source_v1(source_root, authority)
    torch = (
        importlib.import_module("torch")
        if torch_module is None
        else torch_module
    )
    gpu = validate_gpu_v1(torch)
    return _build_execution_environment_attestation_v1(
        Path(source_root).resolve(strict=True),
        authority,
        source_receipt=source,
        gpu_receipt=gpu,
    )


def validate_future_execution_prerequisites_v1(
    authority: Any,
) -> dict[str, Any]:
    value = validate_content_bound_v1(authority)
    expected_fields = {
        "schema",
        "status",
        "scientific_payload_authorized",
        "one_shot",
        "maximum_updates",
        "maximum_presentations",
        "retry_authorized",
        "resume_authorized",
        "preregistration_commit",
        "pinned_source_and_review_commit",
        "certified_source_root",
        "output_root",
        "rgb_root_relative_path",
        "output_root_absent_at_authorization",
        "device",
        "runtime_data_root",
        "selectors",
        "clean_export_certification",
        "metadata_preflight_receipt",
        "runtime_inputs",
        "content_sha256",
    }
    required = {
        "schema": f"{SCHEMA_PREFIX}_future_execution_authority_v1",
        "status": "AUTHORIZED_CERTIFIED_NARROW_EXPORT_ONE_SHOT",
        "scientific_payload_authorized": True,
        "one_shot": True,
        "maximum_updates": MAXIMUM_UPDATES,
        "maximum_presentations": MAXIMUM_PRESENTATIONS,
        "retry_authorized": False,
        "resume_authorized": False,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "certified_source_root": CERTIFIED_SOURCE_ROOT,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "rgb_root_relative_path": RGB_ROOT_RELATIVE_PATH,
        "output_root_absent_at_authorization": True,
        "device": "cuda:0",
        "runtime_data_root": RUNTIME_DATA_ROOT,
    }
    pinned = value.get("pinned_source_and_review_commit")
    if (
        set(value) != expected_fields
        or any(value.get(name) != expected for name, expected in required.items())
        or type(pinned) is not str
        or len(pinned) != 40
        or any(character not in "0123456789abcdef" for character in pinned)
    ):
        raise PermissionError("temporal V1 authority identity, scope, or cap changed")
    if value.get("selectors") != {
        "executor_module": __name__,
        "model_module": MODEL_MODULE_NAME,
        "model_class": MODEL_CLASS_NAME,
        "training_module": TRAINING_MODULE_NAME,
        "evaluation_module": EVALUATION_MODULE_NAME,
        "metrics_module": METRICS_MODULE_NAME,
        "metadata_preflight_module": PREFLIGHT_MODULE_NAME,
    }:
        raise PermissionError("temporal V1 runtime selectors changed")
    certification = _binding(
        value.get("clean_export_certification"),
        content=True,
    )
    if certification["path"] != CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH:
        raise PermissionError("temporal V1 clean-export certification changed")
    preflight = _binding(
        value.get("metadata_preflight_receipt"),
        content=True,
    )
    if preflight["path"] != METADATA_PREFLIGHT_RECEIPT_RELATIVE_PATH:
        raise PermissionError("temporal V1 metadata-preflight binding changed")
    inputs = value.get("runtime_inputs")
    if type(inputs) is not dict or set(inputs) != set(RUNTIME_INPUT_BINDING_NAMES):
        raise PermissionError("temporal V1 runtime input inventory changed")
    for name, expected in RUNTIME_INPUT_BINDINGS.items():
        if _binding(
            inputs[name],
            content="content_sha256" in expected,
        ) != expected:
            raise PermissionError(f"temporal V1 runtime binding changed: {name}")
    return value


def _mkdir_beneath(root: Path, relative: PurePosixPath) -> Path:
    current = root
    for part in relative.parts:
        current = current / part
        created = False
        try:
            os.mkdir(current, 0o700)
            created = True
        except FileExistsError:
            info = os.lstat(current)
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise PermissionError("temporal output containment changed")
        if created:
            os.chmod(current, 0o700, follow_symlinks=False)
    return current


def _write_immutable_bytes(path: Path, raw: bytes) -> dict[str, Any]:
    parent = os.lstat(path.parent)
    if not stat.S_ISDIR(parent.st_mode) or stat.S_ISLNK(parent.st_mode):
        raise PermissionError("immutable publication parent changed type")
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444, follow_symlinks=False)
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444:
        raise PermissionError("immutable publication mode changed")
    return {
        "path": path.name,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }


def _publish_bytes(output: Path, relative: str, raw: bytes) -> dict[str, Any]:
    path = _safe_relative_path(relative)
    directory = _mkdir_beneath(output, PurePosixPath(*path.parts[:-1]))
    binding = _write_immutable_bytes(directory / path.name, raw)
    return {**binding, "path": path.as_posix()}


def _publish_json(
    output: Path,
    relative: str,
    core: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = _content_bound(core)
    raw = _canonical_json_bytes(value) + b"\n"
    return value, _publish_bytes(output, relative, raw)


def reserve_attempt_v1(
    repository_root: Path,
    authority: Mapping[str, Any],
    *,
    environment_attestation: _ExecutionEnvironmentAttestationV1,
    created_utc: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve(strict=True)
    validated = validate_future_execution_prerequisites_v1(dict(authority))
    environment = _validate_execution_environment_attestation_v1(
        environment_attestation,
        validated,
    )
    output_relative = _safe_relative_path(OUTPUT_ROOT_RELATIVE_PATH)
    output = root.joinpath(*output_relative.parts)
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            "temporal V1 output root must be absent before reservation"
        )
    parent = _mkdir_beneath(root, PurePosixPath(*output_relative.parts[:-1]))
    os.mkdir(parent / output_relative.name, 0o700)
    output = parent / output_relative.name
    reservation = _content_bound(
        {
            "schema": f"{SCHEMA_PREFIX}_attempt_reservation_v1",
            "status": "RESERVED_ONE_SHOT",
            "created_utc": created_utc,
            "authority_sha256": hashlib.sha256(
                _canonical_json_bytes(validated)
            ).hexdigest(),
            "environment_attestation_content_sha256": environment[
                "content_sha256"
            ],
            "output_root": OUTPUT_ROOT_RELATIVE_PATH,
            "attempt": 1,
            "maximum_updates": MAXIMUM_UPDATES,
            "maximum_presentations": MAXIMUM_PRESENTATIONS,
            "retry_authorized": False,
            "resume_authorized": False,
        }
    )
    _publish_bytes(
        output,
        "reservation.json",
        _canonical_json_bytes(reservation) + b"\n",
    )
    return reservation


def validate_attempt_reservation_v1(value: Any) -> dict[str, Any]:
    result = validate_content_bound_v1(value)
    required = {
        "schema": f"{SCHEMA_PREFIX}_attempt_reservation_v1",
        "status": "RESERVED_ONE_SHOT",
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "attempt": 1,
        "maximum_updates": MAXIMUM_UPDATES,
        "maximum_presentations": MAXIMUM_PRESENTATIONS,
        "retry_authorized": False,
        "resume_authorized": False,
    }
    environment_sha256 = result.get(
        "environment_attestation_content_sha256"
    )
    authority_sha256 = result.get("authority_sha256")
    if (
        set(result)
        != set(required)
        | {
            "created_utc",
            "authority_sha256",
            "environment_attestation_content_sha256",
            "content_sha256",
        }
        or any(
            result.get(name) != expected
            for name, expected in required.items()
        )
        or type(result.get("created_utc")) is not str
        or not result["created_utc"]
        or type(authority_sha256) is not str
        or len(authority_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in authority_sha256
        )
        or type(environment_sha256) is not str
        or len(environment_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in environment_sha256
        )
    ):
        raise PermissionError("temporal V1 reservation changed")
    return result


def _strict_json(raw: bytes) -> Any:
    def unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise PermissionError("JSON repeats a key")
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=unique,
            parse_constant=lambda value: (_ for _ in ()).throw(
                PermissionError(f"nonfinite JSON constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PermissionError("bound JSON is invalid") from error


def _decode_content_bound_json(raw: bytes, *, name: str) -> dict[str, Any]:
    if (
        type(raw) is not bytes
        or not raw.endswith(b"\n")
        or raw.count(b"\n") != 1
        or b"\r" in raw
    ):
        raise PermissionError(f"{name} is not one canonical JSON line")
    body = raw[:-1]
    value = _strict_json(body)
    if type(value) is not dict or _canonical_json_bytes(value) != body:
        raise PermissionError(f"{name} is not canonical JSON")
    return validate_content_bound_v1(value)


def _read_bound_file(root: Path, binding: Mapping[str, Any]) -> bytes:
    expected = _binding(
        dict(binding),
        content="content_sha256" in binding,
    )
    relative = _safe_relative_path(expected["path"])
    repository = Path(root).resolve(strict=True)
    descriptor = os.open(repository, _DIR_FLAGS)
    file_descriptor: int | None = None
    before: os.stat_result | None = None
    after: os.stat_result | None = None
    try:
        for component in relative.parts[:-1]:
            child = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        file_descriptor = os.open(relative.name, _READ_FLAGS, dir_fd=descriptor)
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PermissionError("bound runtime input is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_descriptor)
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        os.close(descriptor)
    raw = b"".join(chunks)
    if (
        before is None
        or after is None
        or (before.st_dev, before.st_ino, before.st_size)
        != (after.st_dev, after.st_ino, after.st_size)
        or len(raw) != expected["byte_count"]
        or hashlib.sha256(raw).hexdigest() != expected["file_sha256"]
    ):
        raise PermissionError("bound runtime input bytes changed")
    return raw


def load_metadata_preflight_receipt_v1(
    runtime_data_root: Path,
    authority: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = authority["metadata_preflight_receipt"]
    raw = _read_bound_file(runtime_data_root, binding)
    receipt = _decode_content_bound_json(raw, name="metadata-preflight receipt")
    checks = receipt.get("checks")
    access = receipt.get("access")
    validation = receipt.get("validation")
    inputs = receipt.get("inputs")
    expected_inputs = authority["runtime_inputs"]
    if (
        receipt.get("schema") != f"{SCHEMA_PREFIX}_metadata_preflight_receipt_v1"
        or receipt.get("status") != "PASS_METADATA_PREFLIGHT"
        or receipt.get("preregistration_commit") != PREREGISTRATION_COMMIT
        or type(checks) is not dict
        or not checks
        or not all(value is True for value in checks.values())
        or type(access) is not dict
        or access.get("metadata_index_open_count") != 2
        or access.get("rgb_open_count") != 0
        or access.get("checkpoint_open_count") != 0
        or access.get("navigation_open_count") != 0
        or access.get("held_out_or_sealed_opened") is not False
        or type(validation) is not dict
        or validation.get("sentinel_indices_sha256")
        != "615287ba03169cfb390626d38163836d92ad1750fd5a74885e9105e56f5152ee"
        or validation.get("full_wrong_history_donors_sha256")
        != "7bab828cc1170edc39b13e8277d3a739f97106eba4d88bed5631b27a5111823c"
        or validation.get("sentinel_wrong_history_donors_sha256")
        != "6d8978266e466ed191c978819d2aaa79e17773d32e4e17ac0a2542c0bb542dd4"
        or type(validation.get("full_panel_identity_sha256")) is not str
        or len(validation["full_panel_identity_sha256"]) != 64
        or type(validation.get("sentinel_panel_identity_sha256")) is not str
        or len(validation["sentinel_panel_identity_sha256"]) != 64
        or type(inputs) is not dict
    ):
        raise PermissionError("metadata-preflight receipt did not pass exactly")
    for role, authority_name in (
        ("train", "h6_train_index"),
        ("validation", "h6_validation_index"),
    ):
        observed = inputs.get(role)
        expected = expected_inputs[authority_name]
        if type(observed) is not dict or any(
            observed.get(name) != expected[name]
            for name in ("path", "file_sha256", "byte_count")
        ):
            raise PermissionError("metadata-preflight input binding changed")
    return receipt, {
        "receipt_open_count": 1,
        "receipt_file_sha256": binding["file_sha256"],
        "receipt_content_sha256": binding["content_sha256"],
        "rgb_open_count": 0,
        "checkpoint_open_count": 0,
        "passed": True,
    }


def _accepted_predecessor_key(name: str) -> bool:
    return name in _MIGRATED_EXACT or name.startswith(_MIGRATED_PREFIXES)


def _rejected_predecessor_key(name: str) -> bool:
    return name in _REJECTED_EXACT or name.startswith(_REJECTED_PREFIXES)


def _tensor_manifest(torch: Any, state: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name, value in sorted(state.items()):
        if type(name) is not str or not isinstance(value, torch.Tensor):
            raise PermissionError("predecessor state inventory changed")
        tensor = value.detach().to(device="cpu").contiguous()
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise PermissionError("predecessor state contains a nonfinite tensor")
        result.append(
            {
                "name": name,
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
                "sha256": hashlib.sha256(
                    tensor.reshape(-1)
                    .view(torch.uint8)
                    .numpy()
                    .tobytes(order="C")
                ).hexdigest(),
            }
        )
    if not result:
        raise PermissionError("predecessor model state is empty")
    return result


def extract_predecessor_model_state_v1(
    torch: Any,
    checkpoint: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_fields = {
        "schema",
        "model_state_dict",
        "optimizer_state_dict",
        "accounting",
        "model_state_inventory",
        "training_contract",
        "update",
        "authority_sha256",
        "rng",
        "complete_continuation_state",
        "same_attempt_reopen_count",
        "retry_authorized",
        "resume_authorized",
    }
    expected_accounting = {
        "updates": 1_000,
        "presentations": 16_000,
        "mask_rows": 16_000,
        "online_frame_encodings": 16_000,
        "ema_target_frame_encodings": 16_000,
        "microbatch_graphs": 4_000,
        "backward_calls": 4_000,
        "global_gradient_clips": 1_000,
        "optimizer_steps": 1_000,
        "ema_steps": 1_000,
    }
    if (
        type(checkpoint) is not dict
        or set(checkpoint) != expected_fields
        or checkpoint.get("schema")
        != (
            "lewm_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1_"
            "checkpoint_v1"
        )
        or checkpoint.get("update") != 1_000
        or checkpoint.get("accounting") != expected_accounting
        or checkpoint.get("complete_continuation_state") is not True
        or checkpoint.get("same_attempt_reopen_count") != 0
        or checkpoint.get("retry_authorized") is not False
        or checkpoint.get("resume_authorized") is not False
        or type(checkpoint.get("model_state_dict")) is not dict
        or type(checkpoint.get("model_state_inventory")) is not dict
    ):
        raise PermissionError("predecessor checkpoint schema or scope changed")
    state = checkpoint["model_state_dict"]
    manifest = _tensor_manifest(torch, state)
    names = tuple(state)
    inventory = checkpoint["model_state_inventory"]
    names_sha256 = hashlib.sha256(_canonical_json_bytes(names)).hexdigest()
    if (
        inventory.get("state_tensor_count") != len(state)
        or inventory.get("state_names_sha256") != names_sha256
        or inventory.get("ema_update_count") != 1_000
    ):
        raise PermissionError("predecessor checkpoint state inventory changed")
    accepted = tuple(sorted(name for name in state if _accepted_predecessor_key(name)))
    rejected = tuple(sorted(name for name in state if _rejected_predecessor_key(name)))
    if (
        not accepted
        or not rejected
        or len(accepted) + len(rejected) != len(state)
        or "ema_update_count" not in rejected
        or not any(name.startswith("target_encoder.") for name in rejected)
    ):
        raise PermissionError("predecessor accepted/rejected migration split changed")
    detached = {name: value.detach() for name, value in state.items()}
    return detached, {
        "schema": f"{SCHEMA_PREFIX}_predecessor_migration_input_v1",
        "checkpoint_update": 1_000,
        "state_tensor_count": len(state),
        "accepted_state_tensor_count": len(accepted),
        "rejected_state_tensor_count": len(rejected),
        "accepted_state_names_sha256": hashlib.sha256(
            _canonical_json_bytes(accepted)
        ).hexdigest(),
        "rejected_state_names_sha256": hashlib.sha256(
            _canonical_json_bytes(rejected)
        ).hexdigest(),
        "tensor_manifest_sha256": hashlib.sha256(
            _canonical_json_bytes(manifest)
        ).hexdigest(),
        "target_and_ema_state_rejected": True,
        "passed": True,
    }


def load_predecessor_model_state_v1(
    runtime_data_root: Path,
    authority: Mapping[str, Any],
    torch: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = authority["runtime_inputs"]
    result_raw = _read_bound_file(
        runtime_data_root,
        inputs["predecessor_scientific_result"],
    )
    result = _decode_content_bound_json(
        result_raw,
        name="predecessor scientific result",
    )
    success_raw = _read_bound_file(
        runtime_data_root,
        inputs["predecessor_success"],
    )
    success = _decode_content_bound_json(
        success_raw,
        name="predecessor terminal success",
    )
    expected_checkpoint = inputs["predecessor_checkpoint"]
    selected_result = result.get("terminal_artifacts", {}).get(
        "selected_checkpoint"
    )
    selected_success = success.get("selected_checkpoint")
    expected_relative = "snapshots/update_1000.pt"
    if (
        result.get("schema")
        != (
            "lewm_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1_"
            "scientific_result_v1"
        )
        or result.get("status") != "PASS_PERCEPTION_QUALIFIED"
        or result.get("passed") is not True
        or result.get("selected_update") != 1_000
        or success.get("status") != "PASS_PERCEPTION_QUALIFIED"
        or success.get("selected_update") != 1_000
        or type(selected_result) is not dict
        or type(selected_success) is not dict
        or any(
            selected.get("path") != expected_relative
            or selected.get("file_sha256") != expected_checkpoint["file_sha256"]
            or selected.get("byte_count") != expected_checkpoint["byte_count"]
            or selected.get("update") != 1_000
            or selected.get("retry_authorized") is not False
            or selected.get("resume_authorized") is not False
            for selected in (selected_result, selected_success)
        )
    ):
        raise PermissionError("predecessor selection evidence changed")

    # The exact selected checkpoint is opened once.  Deserialization is from
    # these already-bound bytes and never reopens the path.
    checkpoint_raw = _read_bound_file(
        runtime_data_root,
        expected_checkpoint,
    )
    checkpoint = torch.load(
        io.BytesIO(checkpoint_raw),
        map_location="cpu",
        weights_only=True,
    )
    state, migration = extract_predecessor_model_state_v1(torch, checkpoint)
    return state, {
        "schema": f"{SCHEMA_PREFIX}_predecessor_open_receipt_v1",
        "scientific_result_open_count": 1,
        "terminal_success_open_count": 1,
        "checkpoint_open_count": 1,
        "checkpoint_deserialize_count": 1,
        "checkpoint_bytes_read": len(checkpoint_raw),
        "checkpoint_file_sha256": hashlib.sha256(checkpoint_raw).hexdigest(),
        "post_initialization_checkpoint_reopen_count": 0,
        "migration": migration,
        "passed": True,
    }


def _default_apis() -> Any:
    torch = importlib.import_module("torch")
    model_module = importlib.import_module(MODEL_MODULE_NAME)
    training = importlib.import_module(TRAINING_MODULE_NAME)
    evaluation = importlib.import_module(EVALUATION_MODULE_NAME)
    metrics = importlib.import_module(METRICS_MODULE_NAME)
    return SimpleNamespace(
        torch=torch,
        model_class=getattr(model_module, MODEL_CLASS_NAME),
        training=training,
        evaluation=evaluation,
        metrics=metrics,
        open_runtime=evaluation.open_bound_runtime,
        evaluate_checkpoint=evaluation.evaluate_checkpoint,
        evaluate_update_zero=(
            evaluation.evaluate_update_zero_full_and_sentinel_v1
        ),
        evaluate_predecessor=(
            evaluation.evaluate_predecessor_retention_panel_v1
        ),
        load_preflight=load_metadata_preflight_receipt_v1,
        load_predecessor=load_predecessor_model_state_v1,
    )


def _finite_tree(value: Any) -> bool:
    if (
        hasattr(value, "is_floating_point")
        and callable(value.is_floating_point)
        and hasattr(value, "detach")
    ):
        tensor = value.detach()
        return (
            not tensor.is_floating_point()
            or bool(tensor.isfinite().all())
        )
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_tree(item) for item in value)
    return True


def _tensor_tree_finite(value: Any, torch: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return (
            not value.is_floating_point()
            or bool(torch.isfinite(value.detach()).all())
        )
    if isinstance(value, Mapping):
        return all(_tensor_tree_finite(item, torch) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_tensor_tree_finite(item, torch) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _accounting_dict(value: Any) -> dict[str, Any]:
    return dict(asdict(value) if is_dataclass(value) else value)


def _expected_training_accounting(update: int) -> dict[str, int]:
    return {
        "updates": update,
        "sequence_rows": 10 * update,
        "logical_rgb_presentations": 40 * update,
        "online_frame_encodings": 30 * update,
        "ema_target_frame_encodings": 10 * update,
        "microbatch_graphs": 5 * update,
        "backward_calls": 5 * update,
        "global_gradient_clips": update,
        "optimizer_steps": update,
        "ema_steps": update,
    }


def _schedule_prefix_receipt(
    consumed_rows: Sequence[int],
    training_schedule: Sequence[int],
) -> dict[str, Any]:
    consumed = tuple(consumed_rows)
    schedule = tuple(training_schedule)
    return {
        "consumed_schedule_row_count": len(consumed),
        "consumed_schedule_rows_canonical_sha256": hashlib.sha256(
            _canonical_json_bytes(consumed)
        ).hexdigest(),
        "consumed_schedule_rows_equal_runtime_prefix": (
            consumed == schedule[: len(consumed)]
        ),
        "consumed_schedule_rows_unique": len(set(consumed)) == len(consumed),
        "runtime_training_schedule_row_count": len(schedule),
        "runtime_training_schedule_unique": (
            len(set(schedule)) == len(schedule)
        ),
    }


def _migration_integrity(
    model: Any,
    predecessor_state: Mapping[str, Any],
    torch: Any,
) -> dict[str, Any]:
    state = model.state_dict()
    accepted = tuple(
        sorted(name for name in predecessor_state if _accepted_predecessor_key(name))
    )
    inherited_exact = bool(accepted) and all(
        name in state and torch.equal(state[name], predecessor_state[name])
        for name in accepted
    )
    online = dict(model.encoder.named_parameters())
    target = dict(model.target_encoder.named_parameters())
    checks = {
        "accepted_inventory_nonempty": bool(accepted),
        "accepted_values_bit_exact": inherited_exact,
        "target_inventory_matches_online": online.keys() == target.keys(),
        "target_hard_synchronized": (
            online.keys() == target.keys()
            and all(torch.equal(value, target[name]) for name, value in online.items())
        ),
        "target_frozen_eval": (
            not model.target_encoder.training
            and all(not value.requires_grad for value in target.values())
        ),
        "target_zero_grad": all(value.grad is None for value in target.values()),
        "ema_update_count_zero": int(model.ema_update_count.detach().cpu()) == 0,
        "new_memory_modules_present": all(
            hasattr(model, name)
            for name in ("action_embedding", "time_embedding", "temporal_gru")
        ),
    }
    return {
        "schema": f"{SCHEMA_PREFIX}_initial_migration_integrity_v1",
        "accepted_state_tensor_count": len(accepted),
        "accepted_state_names_sha256": hashlib.sha256(
            _canonical_json_bytes(accepted)
        ).hexdigest(),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _initial_integrity(
    model: Any,
    optimizer: Any,
    training: Any,
    torch: Any,
) -> dict[str, Any]:
    inventory = training.parameter_inventory_v1(model)
    partition = training.partition_parameters_v1(model)
    training.validate_optimizer_v1(optimizer, partition)
    optimizer_state = optimizer.state_dict()
    checks = {
        "target_optimizer_excluded": inventory.get(
            "target_optimizer_excluded"
        )
        is True,
        "all_online_roles_nonempty": all(
            bool(getattr(partition, role))
            for role in ("encoder", "predictor", "memory")
        ),
        "target_role_nonempty": bool(partition.target),
        "target_frozen_eval": (
            not model.target_encoder.training
            and all(not value.requires_grad for value in partition.target)
        ),
        "target_zero_grad": all(value.grad is None for value in partition.target),
        "ema_update_count_zero": int(model.ema_update_count.detach().cpu()) == 0,
        "model_and_optimizer_state_finite": (
            _tensor_tree_finite(model.state_dict(), torch)
            and _tensor_tree_finite(optimizer_state, torch)
        ),
        "finite_nonzero_model": _tensor_tree_finite(
            model.state_dict(),
            torch,
        )
        and any(
            value.is_floating_point() and bool(value.detach().abs().sum() > 0)
            for value in model.state_dict().values()
        ),
    }
    return {
        "schema": f"{SCHEMA_PREFIX}_initial_integrity_v1",
        "parameter_inventory": dict(inventory),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _access_delta(
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> dict[str, int]:
    if set(before) != set(after):
        raise RuntimeError("temporal RGB access counter schema changed")
    result = {name: int(after[name]) - int(before[name]) for name in before}
    if any(value < 0 for value in result.values()):
        raise RuntimeError("temporal RGB access counters moved backwards")
    return result


def _training_access_exact(receipt: Mapping[str, int], *, rows: int) -> bool:
    expected_total = 4 * rows
    if (
        receipt.get("rgb_tensor_request_count") != expected_total
        or receipt.get("rgb_open_attempt_count") != expected_total
        or receipt.get("rgb_open_success_count") != expected_total
        or receipt.get("rgb_decode_success_count") != expected_total
        or receipt.get("denied_rgb_position_request_count") != 0
    ):
        return False
    for role in ("train", "val"):
        for access_kind in ("factual", "donor"):
            for position in range(7):
                expected = (
                    rows
                    if role == "train"
                    and access_kind == "factual"
                    and position in (0, 1, 2, 3)
                    else 0
                )
                for operation in (
                    "request",
                    "open_attempt",
                    "open_success",
                    "decode_success",
                ):
                    name = (
                        f"{role}_{access_kind}_rgb_position_{position}_"
                        f"{operation}_count"
                    )
                    if receipt.get(name) != expected:
                        return False
    return True


def _validate_update_result(
    result: Any,
    *,
    update: int,
    model: Any,
    optimizer: Any,
    access: Mapping[str, int],
    torch: Any,
) -> dict[str, Any]:
    accounting = _accounting_dict(result.accounting)
    receipt = dict(result.gradient_receipt)
    required_receipt = {
        "sole_future_jepa_route": True,
        "all_gradient_receipts_finite": True,
        "encoder_missing_gradient_tensor_count": 0,
        "predictor_missing_gradient_tensor_count": 0,
        "memory_missing_gradient_tensor_count": 0,
    }
    role_norms = (
        "encoder_gradient_norm_before_clip",
        "predictor_gradient_norm_before_clip",
        "memory_gradient_norm_before_clip",
    )
    if (
        accounting != _expected_training_accounting(update)
        or result.target_gradient_tensor_count != 0
        or result.optimizer_steps_this_update != 1
        or result.ema_steps_this_update != 1
        or int(model.ema_update_count.detach().cpu()) != update
        or any(receipt.get(name) != expected for name, expected in required_receipt.items())
        or any(
            not math.isfinite(float(receipt.get(name, 0.0)))
            or float(receipt.get(name, 0.0)) <= 0.0
            for name in role_norms
        )
        or not math.isfinite(float(result.mean_jepa_loss))
        or not _training_access_exact(access, rows=10)
        or any(
            parameter.grad is not None
            for parameter in model.target_encoder.parameters()
        )
        or not _tensor_tree_finite(model.state_dict(), torch)
        or not _tensor_tree_finite(optimizer.state_dict(), torch)
    ):
        raise RuntimeError(
            "temporal update, gradient, access, or accounting integrity failed"
        )
    return {
        "update": update,
        "accounting": accounting,
        "mean_jepa_loss": float(result.mean_jepa_loss),
        "gradient_receipt": receipt,
        "row_indices_sha256": result.row_indices_sha256,
        "target_indices_sha256": result.target_indices_sha256,
        "training_rgb_access": dict(access),
        "passed": True,
    }


def _validate_runtime_audit(
    audit: Mapping[str, Any],
    authority: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    expected = authority["runtime_inputs"]
    train = audit.get("train")
    validation = audit.get("validation")
    panels = audit.get("panels")
    if (
        type(train) is not dict
        or type(validation) is not dict
        or type(panels) is not dict
        or any(
            train.get(name) != expected["h6_train_index"][name]
            for name in ("path", "file_sha256", "byte_count")
        )
        or any(
            validation.get(name) != expected["h6_validation_index"][name]
            for name in ("path", "file_sha256", "byte_count")
        )
        or panels.get("training_schedule_indices_sha256")
        != preflight["train"]["schedule_indices_sha256"]
        or panels.get("sentinel_indices_sha256")
        != preflight["validation"]["sentinel_indices_sha256"]
        or panels.get("wrong_history_donor_indices_sha256")
        != preflight["validation"]["full_wrong_history_donors_sha256"]
        or panels.get("sentinel_wrong_history_donor_indices_sha256")
        != preflight["validation"]["sentinel_wrong_history_donors_sha256"]
        or audit.get("rgb_open_count") != 0
        or audit.get("checkpoint_open_count") != 0
        or audit.get("gpu_tensor_allocation_count") != 0
    ):
        raise PermissionError("temporal runtime preflight differs from authority")
    return {
        "schema": f"{SCHEMA_PREFIX}_runtime_input_audit_v1",
        "audit": dict(audit),
        "passed": True,
    }


def _validate_predecessor_panel(
    panel: Mapping[str, Any],
    *,
    temporal_update: int,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        type(panel) is not dict
        or panel.get("schema")
        != f"{SCHEMA_PREFIX}_predecessor_retention_panel_v1"
        or panel.get("temporal_update") != temporal_update
        or panel.get("underlying_spatial_evaluator_update") != 0
        or type(panel.get("runtime_audit")) is not dict
        or type(panel.get("evaluation")) is not dict
        or not _finite_tree(panel)
    ):
        raise RuntimeError("predecessor retention panel identity changed")
    audit = panel["runtime_audit"]
    evaluation = panel["evaluation"]
    expected = authority["runtime_inputs"]
    train = audit.get("train")
    validation = audit.get("validation")
    place = audit.get("place")
    controls = evaluation.get("controls")
    raw_health = evaluation.get("raw_health")
    place_result = evaluation.get("place")
    if (
        type(train) is not dict
        or type(validation) is not dict
        or type(place) is not dict
        or any(
            train.get(name) != expected["h6_train_index"][name]
            for name in ("path", "file_sha256", "byte_count")
        )
        or any(
            validation.get(name) != expected["h6_validation_index"][name]
            for name in ("path", "file_sha256", "byte_count")
        )
        or place.get("manifest_file_sha256")
        != expected["place_triplet_manifest"]["file_sha256"]
        or place.get("index_file_sha256")
        != expected["place_triplet_checkpoint_selection_index"]["file_sha256"]
        or evaluation.get("update") != 0
        or type(controls) is not dict
        or set(controls) != set(PREDECESSOR_CONTROL_NAMES)
        or type(raw_health) is not dict
        or set(raw_health) != {"online", "target"}
        or any(
            type(raw_health[branch]) is not dict
            or any(
                name not in raw_health[branch]
                or not math.isfinite(float(raw_health[branch][name]))
                or float(raw_health[branch][name]) <= 0.0
                for name in RAW_HEALTH_NAMES
            )
            for branch in ("online", "target")
        )
        or type(place_result) is not dict
        or evaluation.get("integrity", {}).get("passed") is not True
        or evaluation.get("access", {}).get("future_rgb_tensor_count") != 0
        or evaluation.get("access", {}).get("action_tensor_count") != 0
    ):
        raise PermissionError("predecessor retention evidence changed")
    retrieval = place_result.get("retrieval")
    if (
        type(retrieval) is not dict
        or not math.isfinite(float(retrieval.get("chance_multiple", 0.0)))
        or float(retrieval.get("chance_multiple", 0.0)) <= 0.0
        or not math.isfinite(
            float(place_result.get("target_place_key_effective_rank", 0.0))
        )
        or float(place_result.get("target_place_key_effective_rank", 0.0))
        <= 0.0
    ):
        raise RuntimeError("predecessor place health is invalid")
    return dict(panel)


def _predecessor_bridge(
    panel: Mapping[str, Any],
    *,
    baseline_panel: Mapping[str, Any] | None,
    training_accounting_exact: bool,
    latest_training_receipt_pass: bool | None,
    model_and_optimizer_state_finite: bool,
) -> dict[str, Any]:
    if model_and_optimizer_state_finite is not True:
        raise FloatingPointError(
            "predecessor bridge requires verified finite runtime state"
        )
    evaluation = panel["evaluation"]
    bridge: dict[str, Any] = {
        "predecessor_controls": dict(evaluation["controls"]),
        "training_accounting_exact": training_accounting_exact,
        "latest_training_receipt_pass": latest_training_receipt_pass,
        "baseline_health_noncollapsed": True,
        "model_and_optimizer_state_finite": model_and_optimizer_state_finite,
    }
    if baseline_panel is None:
        return bridge
    baseline = baseline_panel["evaluation"]
    retentions: dict[str, float] = {}
    for branch in ("online", "target"):
        for name in RAW_HEALTH_NAMES:
            denominator = float(baseline["raw_health"][branch][name])
            current = float(evaluation["raw_health"][branch][name])
            if denominator <= 0.0:
                raise RuntimeError("predecessor raw-health baseline is nonpositive")
            retentions[f"{branch}.{name}"] = current / denominator
    baseline_chance = float(baseline["place"]["retrieval"]["chance_multiple"])
    current_chance = float(evaluation["place"]["retrieval"]["chance_multiple"])
    baseline_rank = float(
        baseline["place"]["target_place_key_effective_rank"]
    )
    current_rank = float(
        evaluation["place"]["target_place_key_effective_rank"]
    )
    if baseline_chance <= 0.0 or baseline_rank <= 0.0:
        raise RuntimeError("predecessor place baseline is nonpositive")
    bridge.update(
        {
            "raw_health_retentions": retentions,
            "place_chance_multiple_retention": (
                current_chance / baseline_chance
            ),
            "target_place_rank_retention": current_rank / baseline_rank,
        }
    )
    return bridge


def _summary_from_dict(metrics: Any, value: Mapping[str, Any]) -> Any:
    return metrics.ControlSummary(**dict(value))


def _observation_from_record(
    record: Mapping[str, Any],
    *,
    bridge: Mapping[str, Any],
    metrics: Any,
    expected_panel_identity: str,
) -> tuple[Any, dict[str, Any]]:
    update = record.get("update")
    panel_kind = record.get("panel_kind")
    expected_rows = 2_048 if panel_kind == "full" else 256
    if (
        type(record) is not dict
        or record.get("schema") != f"{SCHEMA_PREFIX}_checkpoint_evaluation_v1"
        or update not in OBSERVATION_UPDATES
        or panel_kind not in {"full", "sentinel"}
        or record.get("row_count") != expected_rows
        or record.get("panel_identity_sha256") != expected_panel_identity
        or not _finite_tree(record)
    ):
        raise RuntimeError("temporal observation record identity changed")
    controls_raw = record.get("controls")
    health = record.get("health")
    integrity_raw = record.get("integrity")
    predecessor_raw = bridge.get("predecessor_controls")
    if (
        type(controls_raw) is not dict
        or set(controls_raw) != set(metrics.CONTROL_NAMES)
        or type(health) is not dict
        or set(health) != {"recurrent", "prediction", "target"}
        or type(integrity_raw) is not dict
    ):
        raise RuntimeError("temporal observation payload changed")
    predecessor = (
        None
        if predecessor_raw is None
        else {
            name: _summary_from_dict(metrics, value)
            for name, value in dict(predecessor_raw).items()
        }
    )
    observation = metrics.TemporalObservation(
        update=update,
        panel_kind=panel_kind,
        panel_identity_sha256=expected_panel_identity,
        controls={
            name: _summary_from_dict(metrics, controls_raw[name])
            for name in metrics.CONTROL_NAMES
        },
        recurrent_health=metrics.RepresentationHealth(**health["recurrent"]),
        prediction_health=metrics.RepresentationHealth(**health["prediction"]),
        target_health=metrics.RepresentationHealth(**health["target"]),
        integrity=metrics.IntegrityFacts(**integrity_raw),
        predecessor_controls=predecessor,
        raw_health_retentions=bridge.get("raw_health_retentions"),
        place_chance_multiple_retention=bridge.get(
            "place_chance_multiple_retention"
        ),
        target_place_rank_retention=bridge.get("target_place_rank_retention"),
    )
    expected_checks = metrics.observation_survival_checks(observation)
    if panel_kind == "full" and update in (200, 400):
        expected_checks = metrics.qualification_checks(observation)
    evaluator_gate = record.get("gate")
    if (
        type(evaluator_gate) is not dict
        or evaluator_gate.get("checks") != expected_checks
        or evaluator_gate.get("passed") != all(expected_checks.values())
    ):
        raise RuntimeError("evaluator and pure temporal gates disagree")
    gate = metrics.observation_gate(observation)
    return observation, gate


def _serialize_checkpoint(
    torch: Any,
    payload: Mapping[str, Any],
    *,
    update: int,
    authority_sha256: str,
) -> bytes:
    gpu_rng: tuple[Any, ...] = ()
    if bool(torch.cuda.is_available()):
        gpu_rng = tuple(value.clone() for value in torch.cuda.get_rng_state_all())
    complete = {
        **dict(payload),
        "update": update,
        "authority_sha256": authority_sha256,
        "rng": {
            "torch_cpu": torch.random.get_rng_state().clone(),
            "visible_gpu": gpu_rng,
        },
        "complete_continuation_state": True,
        "same_attempt_reopen_count": 0,
        "predecessor_checkpoint_reopen_count": 0,
        "retry_authorized": False,
        "resume_authorized": False,
    }
    buffer = io.BytesIO()
    torch.save(complete, buffer)
    return buffer.getvalue()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def terminalize_failure_v1(
    output: Path,
    reservation: Mapping[str, Any],
    *,
    stage: str,
    error: BaseException,
    accounting: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    metrics_bindings: Sequence[Mapping[str, Any]],
    access: Mapping[str, Any],
    last_completed_update: int,
) -> dict[str, Any]:
    validate_attempt_reservation_v1(dict(reservation))
    value, _ = _publish_json(
        output,
        "failure.json",
        {
            "schema": f"{SCHEMA_PREFIX}_exception_failure_v1",
            "status": "FAIL_EXCEPTION_TERMINAL_NO_RETRY_NO_RESUME",
            "stage": stage,
            "created_utc": _utc_now(),
            "exception_type": type(error).__name__,
            "exception_message_sha256": hashlib.sha256(
                str(error).encode("utf-8")
            ).hexdigest(),
            "last_completed_update": last_completed_update,
            "accounting": dict(accounting),
            "checkpoints": list(checkpoints),
            "metrics": list(metrics_bindings),
            "access": dict(access),
            "predecessor_checkpoint_reopen_count": 0,
            "attempt_consumed": True,
            "retry_authorized": False,
            "resume_authorized": False,
            "held_out_or_sealed_opened": False,
            "navigation_executed": False,
            "g2_executed": False,
        },
    )
    return value


def _publish_checkpoint(
    output: Path,
    *,
    api: Any,
    model: Any,
    optimizer: Any,
    accounting: Any,
    update: int,
    authority_sha256: str,
) -> dict[str, Any]:
    if update not in CHECKPOINT_UPDATES:
        raise PermissionError("temporal checkpoint update is not registered")
    payload = api.training.checkpoint_payload_v1(
        model,
        optimizer,
        accounting,
    )
    raw = _serialize_checkpoint(
        api.torch,
        payload,
        update=update,
        authority_sha256=authority_sha256,
    )
    binding = _publish_bytes(
        output,
        f"snapshots/update_{update}.pt",
        raw,
    )
    checkpoint = {
        **binding,
        "update": update,
        "complete_continuation_state": True,
        "same_attempt_reopen_count": 0,
        "predecessor_checkpoint_reopen_count": 0,
        "retry_authorized": False,
        "resume_authorized": False,
    }
    metadata, metadata_binding = _publish_json(
        output,
        f"snapshots/update_{update}.binding.json",
        {
            "schema": f"{SCHEMA_PREFIX}_checkpoint_binding_v1",
            **checkpoint,
        },
    )
    return {
        **checkpoint,
        "metadata": metadata_binding,
        "metadata_content_sha256": metadata["content_sha256"],
    }


def _close_runtime(runtime: Any) -> None:
    close = getattr(runtime, "close", None)
    if callable(close):
        close()
        return
    loader = getattr(runtime, "loader", None)
    close = getattr(loader, "close", None)
    if callable(close):
        close()


def run_authorized_engine_v1(
    *,
    authority: Mapping[str, Any],
    reservation: Mapping[str, Any],
    environment_attestation: _ExecutionEnvironmentAttestationV1,
    repository_root: Path,
    runtime_data_root: Path,
    device: Any,
    apis: Any | None = None,
) -> dict[str, Any]:
    """Execute the exact one-shot temporal schedule and terminalize once."""

    validated = validate_future_execution_prerequisites_v1(dict(authority))
    environment = _validate_execution_environment_attestation_v1(
        environment_attestation,
        validated,
    )
    reserved = validate_attempt_reservation_v1(dict(reservation))
    authority_sha256 = hashlib.sha256(
        _canonical_json_bytes(validated)
    ).hexdigest()
    if reserved["authority_sha256"] != authority_sha256:
        raise PermissionError(
            "temporal reservation does not bind the supplied authority"
        )
    if (
        reserved["environment_attestation_content_sha256"]
        != environment["content_sha256"]
    ):
        raise PermissionError(
            "temporal reservation does not bind the guarded environment"
        )
    root = Path(repository_root).resolve(strict=True)
    data_root = Path(runtime_data_root).resolve(strict=True)
    if (
        root != data_root
        or str(data_root) != validated["runtime_data_root"]
        or str(device) != validated["device"]
    ):
        raise PermissionError("temporal runtime/output root or device changed")
    output = root / OUTPUT_ROOT_RELATIVE_PATH
    if not output.is_dir() or output.is_symlink():
        raise PermissionError(
            "temporal reserved output root is absent or changed type"
        )

    api = _default_apis() if apis is None else apis
    runtime = None
    model = optimizer = accounting = None
    preflight_receipt: dict[str, Any] = {}
    preflight_access: dict[str, Any] = {}
    predecessor_open: dict[str, Any] = {}
    runtime_audit: dict[str, Any] = {}
    checkpoints: list[dict[str, Any]] = []
    metric_bindings: list[dict[str, Any]] = []
    predecessor_bindings: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    training_schedule: tuple[int, ...] = ()
    consumed_schedule_rows: list[int] = []
    last_completed_update = 0
    latest_training_receipt: dict[str, Any] | None = None
    stage = "load_metadata_preflight"
    try:
        preflight_receipt, preflight_access = api.load_preflight(
            data_root,
            validated,
        )

        stage = "load_predecessor_once"
        predecessor_state, predecessor_open = api.load_predecessor(
            data_root,
            validated,
            api.torch,
        )
        stage = "initialize_model"
        model = api.model_class(predecessor_state)
        migration_integrity = _migration_integrity(
            model,
            predecessor_state,
            api.torch,
        )
        if not migration_integrity["passed"]:
            raise RuntimeError("temporal predecessor migration integrity failed")
        model = model.to(device)
        model.train()
        optimizer = api.training.build_optimizer_v1(model)
        initial_integrity = _initial_integrity(
            model,
            optimizer,
            api.training,
            api.torch,
        )
        if not initial_integrity["passed"]:
            raise RuntimeError("temporal initial model integrity failed")

        stage = "open_temporal_runtime"
        runtime, raw_runtime_audit = api.open_runtime(
            data_root,
            device=device,
        )
        runtime_audit = _validate_runtime_audit(
            raw_runtime_audit,
            validated,
            preflight_receipt,
        )
        training_schedule = tuple(runtime.training_schedule)
        expected_schedule_rows = 10 * MAXIMUM_UPDATES
        if (
            len(training_schedule) != expected_schedule_rows
            or any(type(row) is not int for row in training_schedule)
            or len(set(training_schedule)) != expected_schedule_rows
        ):
            raise RuntimeError(
                "temporal runtime training schedule is not 4,000 unique rows"
            )

        stage = "observe_update_0_predecessor"
        update_zero_state_finite = (
            _tensor_tree_finite(model.state_dict(), api.torch)
            and _tensor_tree_finite(optimizer.state_dict(), api.torch)
        )
        if not update_zero_state_finite:
            raise FloatingPointError(
                "update-zero model or optimizer state is nonfinite"
            )
        predecessor_zero = _validate_predecessor_panel(
            api.evaluate_predecessor(model, data_root, 0, device),
            temporal_update=0,
            authority=validated,
        )
        _, predecessor_binding = _publish_json(
            output,
            "metrics/update_0_predecessor.json",
            predecessor_zero,
        )
        predecessor_bindings.append(predecessor_binding)
        update_zero_bridge = _predecessor_bridge(
            predecessor_zero,
            baseline_panel=None,
            training_accounting_exact=True,
            latest_training_receipt_pass=None,
            model_and_optimizer_state_finite=update_zero_state_finite,
        )

        stage = "observe_update_0_temporal_once"
        paired = api.evaluate_update_zero(
            model,
            runtime,
            device,
            baseline=update_zero_bridge,
        )
        if (
            type(paired) is not dict
            or paired.get("schema")
            != f"{SCHEMA_PREFIX}_update_zero_full_and_sentinel_v1"
            or paired.get("update") != 0
            or paired.get("single_temporal_rgb_and_model_pass") is not True
            or type(paired.get("full")) is not dict
            or type(paired.get("sentinel")) is not dict
        ):
            raise RuntimeError("update-zero paired temporal evaluation changed")
        full_identity = preflight_receipt["validation"][
            "full_panel_identity_sha256"
        ]
        sentinel_identity = preflight_receipt["validation"][
            "sentinel_panel_identity_sha256"
        ]
        zero_full, zero_full_gate = _observation_from_record(
            paired["full"],
            bridge=update_zero_bridge,
            metrics=api.metrics,
            expected_panel_identity=full_identity,
        )
        zero_sentinel, zero_sentinel_gate = _observation_from_record(
            paired["sentinel"],
            bridge=update_zero_bridge,
            metrics=api.metrics,
            expected_panel_identity=sentinel_identity,
        )
        provenance = paired["sentinel"].get("access_provenance", {})
        if (
            provenance.get("derived_from_source_panel") is not True
            or provenance.get("source_panel_row_count") != 2_048
            or provenance.get("additional_rgb_open_count") != 0
            or provenance.get("additional_model_call_count") != 0
        ):
            raise RuntimeError("update-zero sentinel was not an exact zero-I/O slice")
        zero_full_decision = api.metrics.continuation_gate(
            zero_full,
            update_zero=zero_full,
        )
        zero_sentinel_decision = api.metrics.continuation_gate(
            zero_sentinel,
            update_zero=zero_sentinel,
        )
        for panel_name, raw, observation, gate, decision in (
            (
                "full",
                paired["full"],
                zero_full,
                zero_full_gate,
                zero_full_decision,
            ),
            (
                "sentinel",
                paired["sentinel"],
                zero_sentinel,
                zero_sentinel_gate,
                zero_sentinel_decision,
            ),
        ):
            record = {
                **dict(raw),
                "observation_gate": gate,
                "continuation_gate": decision,
                "accounting": _expected_training_accounting(0),
                "initial_integrity": initial_integrity,
                "migration_integrity": migration_integrity,
                "predecessor_panel": predecessor_binding,
                "executor_model_and_optimizer_state_finite": True,
                "held_out_or_sealed_opened": False,
                "navigation_executed": False,
            }
            _, binding = _publish_json(
                output,
                f"metrics/update_0_{panel_name}.json",
                record,
            )
            metric_bindings.append(binding)
        baseline_full = zero_full
        baseline_sentinel = zero_sentinel
        previous_sentinel = zero_sentinel
        terminal_decision: Mapping[str, Any] | None = None
        if (
            zero_full_decision["action"] != "CONTINUE"
            or zero_sentinel_decision["action"] != "CONTINUE"
        ):
            terminal_decision = {
                "status": "FAIL_UPDATE_ZERO_OBSERVATION_SURVIVAL",
                "action": "TERMINAL",
                "full": zero_full_decision,
                "sentinel": zero_sentinel_decision,
            }

        for update in range(1, MAXIMUM_UPDATES + 1):
            if terminal_decision is not None:
                break
            stage = f"train_update_{update}"
            start = 10 * (update - 1)
            schedule_slice = training_schedule[start : start + 10]
            if len(schedule_slice) != 10:
                raise RuntimeError("temporal training schedule ended early")
            access_before = runtime.access_snapshot()
            batches = runtime.load_training_microbatches(schedule_slice, device)
            access_after = runtime.access_snapshot()
            training_access = _access_delta(access_before, access_after)
            if (
                len(batches) != 5
                or any(len(batch.row_indices) != 2 for batch in batches)
                or tuple(
                    row
                    for batch in batches
                    for row in batch.row_indices
                )
                != schedule_slice
            ):
                raise RuntimeError("temporal runtime microbatch schedule changed")
            consumed_schedule_rows.extend(schedule_slice)
            if (
                tuple(consumed_schedule_rows)
                != training_schedule[: 10 * update]
                or len(set(consumed_schedule_rows))
                != len(consumed_schedule_rows)
            ):
                raise RuntimeError(
                    "temporal consumed schedule rows are not the unique prefix"
                )
            result = api.training.training_update_v1(
                model,
                optimizer,
                tuple(batch.context_rgb for batch in batches),
                tuple(batch.action_sequence for batch in batches),
                tuple(batch.target_rgb for batch in batches),
                tuple(batch.row_indices for batch in batches),
                expected_row_indices=schedule_slice,
                schedule_offset=start,
                accounting=accounting,
            )
            accounting = result.accounting
            latest_training_receipt = _validate_update_result(
                result,
                update=update,
                model=model,
                optimizer=optimizer,
                access=training_access,
                torch=api.torch,
            )
            trace.append(latest_training_receipt)
            last_completed_update = update
            if update not in OBSERVATION_UPDATES:
                continue

            is_full = update in FULL_OBSERVATION_UPDATES
            observation_state_finite = (
                _tensor_tree_finite(model.state_dict(), api.torch)
                and _tensor_tree_finite(optimizer.state_dict(), api.torch)
            )
            if not observation_state_finite:
                raise FloatingPointError(
                    "observation model or optimizer state is nonfinite"
                )
            predecessor_panel = None
            predecessor_binding_for_update = None
            if is_full:
                stage = f"observe_update_{update}_predecessor"
                predecessor_panel = _validate_predecessor_panel(
                    api.evaluate_predecessor(
                        model,
                        data_root,
                        update,
                        device,
                    ),
                    temporal_update=update,
                    authority=validated,
                )
                _, predecessor_binding_for_update = _publish_json(
                    output,
                    f"metrics/update_{update}_predecessor.json",
                    predecessor_panel,
                )
                predecessor_bindings.append(
                    predecessor_binding_for_update
                )
                bridge = _predecessor_bridge(
                    predecessor_panel,
                    baseline_panel=predecessor_zero,
                    training_accounting_exact=(
                        _accounting_dict(accounting)
                        == _expected_training_accounting(update)
                    ),
                    latest_training_receipt_pass=(
                        latest_training_receipt["passed"]
                    ),
                    model_and_optimizer_state_finite=(
                        observation_state_finite
                    ),
                )
            else:
                bridge = {
                    "training_accounting_exact": (
                        _accounting_dict(accounting)
                        == _expected_training_accounting(update)
                    ),
                    "latest_training_receipt_pass": (
                        latest_training_receipt["passed"]
                    ),
                    "baseline_health_noncollapsed": True,
                    "model_and_optimizer_state_finite": (
                        observation_state_finite
                    ),
                }

            stage = f"observe_update_{update}_temporal"
            raw_observation = api.evaluate_checkpoint(
                model,
                runtime,
                update,
                device,
                full=is_full,
                baseline=bridge,
            )
            expected_identity = full_identity if is_full else sentinel_identity
            observation, observation_gate = _observation_from_record(
                raw_observation,
                bridge=bridge,
                metrics=api.metrics,
                expected_panel_identity=expected_identity,
            )
            continuation = api.metrics.continuation_gate(
                observation,
                update_zero=(
                    baseline_full if is_full else baseline_sentinel
                ),
                previous=(
                    previous_sentinel if update == 100 else None
                ),
            )
            if update in CHECKPOINT_UPDATES:
                stage = f"publish_checkpoint_update_{update}"
                checkpoint = _publish_checkpoint(
                    output,
                    api=api,
                    model=model,
                    optimizer=optimizer,
                    accounting=accounting,
                    update=update,
                    authority_sha256=authority_sha256,
                )
                checkpoints.append(checkpoint)
            else:
                checkpoint = None
            record = {
                **dict(raw_observation),
                "observation_gate": observation_gate,
                "continuation_gate": continuation,
                "accounting": _accounting_dict(accounting),
                "latest_training_receipt": latest_training_receipt,
                "predecessor_panel": predecessor_binding_for_update,
                "checkpoint": checkpoint,
                "executor_model_and_optimizer_state_finite": True,
                "held_out_or_sealed_opened": False,
                "navigation_executed": False,
            }
            stage = f"publish_metrics_update_{update}"
            _, binding = _publish_json(
                output,
                (
                    f"metrics/update_{update}_"
                    f"{'full' if is_full else 'sentinel'}.json"
                ),
                record,
            )
            metric_bindings.append(binding)
            if not is_full:
                previous_sentinel = observation
            if continuation["action"] != "CONTINUE":
                terminal_decision = continuation
                break

        if terminal_decision is None:
            raise RuntimeError("temporal schedule ended without a terminal decision")
        terminal_update = last_completed_update
        terminal_accounting = (
            _expected_training_accounting(0)
            if accounting is None
            else _accounting_dict(accounting)
        )
        if (
            terminal_accounting != _expected_training_accounting(terminal_update)
            or terminal_accounting["logical_rgb_presentations"]
            > MAXIMUM_PRESENTATIONS
            or len(consumed_schedule_rows) != 10 * terminal_update
            or tuple(consumed_schedule_rows)
            != training_schedule[: 10 * terminal_update]
        ):
            raise RuntimeError("terminal temporal accounting changed")
        schedule_prefix_receipt = _schedule_prefix_receipt(
            consumed_schedule_rows,
            training_schedule,
        )
        if (
            schedule_prefix_receipt[
                "consumed_schedule_rows_equal_runtime_prefix"
            ]
            is not True
            or schedule_prefix_receipt["consumed_schedule_rows_unique"]
            is not True
            or schedule_prefix_receipt["runtime_training_schedule_row_count"]
            != 4_000
            or schedule_prefix_receipt["runtime_training_schedule_unique"]
            is not True
        ):
            raise RuntimeError("terminal temporal schedule-prefix receipt failed")

        stage = "publish_trace"
        trace_raw = b"".join(
            _canonical_json_bytes(_content_bound(row)) + b"\n"
            for row in trace
        )
        trace_binding = _publish_bytes(output, "trace.jsonl", trace_raw)
        stage = "publish_terminal_access"
        final_runtime_access = runtime.access_audit()
        if (
            final_runtime_access.get("passed") is not True
            or final_runtime_access.get("forbidden_rgb_open_count") != 0
        ):
            raise RuntimeError("terminal temporal RGB access audit failed")
        terminal_access, terminal_access_binding = _publish_json(
            output,
            "receipts/terminal_access.json",
            {
                "schema": f"{SCHEMA_PREFIX}_terminal_access_receipt_v1",
                "metadata_preflight": preflight_access,
                "predecessor": predecessor_open,
                "runtime_preflight": runtime_audit,
                "runtime_terminal": final_runtime_access,
                "training_trace_row_count": len(trace),
                "schedule_prefix": schedule_prefix_receipt,
                "predecessor_panel_bindings": predecessor_bindings,
                "predecessor_checkpoint_reopen_count": 0,
                "held_out_or_sealed_opened": False,
                "navigation_executed": False,
                "g2_executed": False,
            },
        )
        common = {
            "terminal_update": terminal_update,
            "accounting": terminal_accounting,
            "terminal_decision": dict(terminal_decision),
            "metrics": metric_bindings,
            "predecessor_panels": predecessor_bindings,
            "checkpoints": checkpoints,
            "trace": trace_binding,
            "terminal_access": terminal_access_binding,
            "terminal_access_content_sha256": terminal_access[
                "content_sha256"
            ],
            "attempt_consumed": True,
            "retry_authorized": False,
            "resume_authorized": False,
            "held_out_or_sealed_opened": False,
            "navigation_executed": False,
            "g2_executed": False,
        }
        if terminal_decision.get("status") == "PASS_TEMPORAL_QUALIFIED":
            selected_update = int(terminal_decision["selected_update"])
            selected = next(
                item for item in checkpoints if item["update"] == selected_update
            )
            value, _ = _publish_json(
                output,
                "success.json",
                {
                    "schema": f"{SCHEMA_PREFIX}_success_v1",
                    "status": "PASS_TEMPORAL_PERCEPTION_QUALIFIED",
                    **common,
                    "selected_update": selected_update,
                    "selected_checkpoint": selected,
                    "learned_navigation_memory_integration_may_be_preregistered": True,
                    "navigation_authorized": False,
                    "g2_authorized": False,
                },
            )
            return value
        value, _ = _publish_json(
            output,
            "failure.json",
            {
                "schema": f"{SCHEMA_PREFIX}_scientific_failure_v1",
                "status": (
                    "FAIL_SCIENTIFIC_NO_QUALIFYING_CHECKPOINT"
                    if terminal_update == MAXIMUM_UPDATES
                    else "FAIL_SCIENTIFIC_CONTINUATION_GATE_NOT_MET"
                ),
                **common,
                "checkpoint_selected": False,
            },
        )
        return value
    except BaseException as error:
        return terminalize_failure_v1(
            output,
            reserved,
            stage=stage,
            error=error,
            accounting=(
                {}
                if accounting is None
                else _accounting_dict(accounting)
            ),
            checkpoints=checkpoints,
            metrics_bindings=metric_bindings,
            access={
                "metadata_preflight": preflight_access,
                "predecessor": predecessor_open,
                "runtime_preflight": runtime_audit,
                "schedule_prefix": _schedule_prefix_receipt(
                    consumed_schedule_rows,
                    training_schedule,
                ),
                "predecessor_checkpoint_reopen_count": 0,
                "held_out_or_sealed_opened": False,
                "navigation_executed": False,
            },
            last_completed_update=last_completed_update,
        )
    finally:
        if runtime is not None:
            _close_runtime(runtime)


def execute_authorized_v1(
    repository_root: Path,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Reserve the absent runtime attempt root, then consume the one shot."""

    validated = validate_future_execution_prerequisites_v1(dict(authority))
    source_root = Path(repository_root).resolve(strict=True)
    if str(source_root) != validated["certified_source_root"]:
        raise PermissionError(
            "temporal executor is outside its certified source root"
        )
    environment = validate_execution_environment_v1(source_root, validated)
    data_root = Path(validated["runtime_data_root"]).resolve(strict=True)
    reservation = reserve_attempt_v1(
        data_root,
        validated,
        environment_attestation=environment,
        created_utc=_utc_now(),
    )
    return run_authorized_engine_v1(
        authority=validated,
        reservation=reservation,
        environment_attestation=environment,
        repository_root=data_root,
        runtime_data_root=data_root,
        device=validated["device"],
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    arguments = parser.parse_args()
    info = os.lstat(arguments.authority)
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise PermissionError("temporal authority path changed type")
    authority = _decode_content_bound_json(
        arguments.authority.read_bytes(),
        name="temporal execution authority",
    )
    result = execute_authorized_v1(arguments.repository_root, authority)
    print(_canonical_json_bytes(result).decode("ascii"))
    return (
        0
        if result.get("status") == "PASS_TEMPORAL_PERCEPTION_QUALIFIED"
        else 3
    )


__all__ = [
    "AUTHORITY_RELATIVE_PATH",
    "CERTIFIED_SOURCE_ROOT",
    "CHECKPOINT_UPDATES",
    "CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH",
    "EVALUATION_MODULE_NAME",
    "FULL_OBSERVATION_UPDATES",
    "MAXIMUM_PRESENTATIONS",
    "MAXIMUM_UPDATES",
    "METADATA_PREFLIGHT_RECEIPT_RELATIVE_PATH",
    "METRICS_MODULE_NAME",
    "MODEL_CLASS_NAME",
    "MODEL_MODULE_NAME",
    "OBSERVATION_UPDATES",
    "OUTPUT_ROOT_RELATIVE_PATH",
    "PREFLIGHT_MODULE_NAME",
    "PREREGISTRATION_COMMIT",
    "RUNTIME_DATA_ROOT",
    "RUNTIME_INPUT_BINDINGS",
    "RUNTIME_INPUT_BINDING_NAMES",
    "SCHEMA_PREFIX",
    "SENTINEL_OBSERVATION_UPDATES",
    "TRAINING_MODULE_NAME",
    "execute_authorized_v1",
    "extract_predecessor_model_state_v1",
    "load_metadata_preflight_receipt_v1",
    "load_predecessor_model_state_v1",
    "validate_attempt_reservation_v1",
    "validate_certified_source_v1",
    "validate_content_bound_v1",
    "validate_execution_environment_v1",
    "validate_future_execution_prerequisites_v1",
    "validate_gpu_v1",
]


if __name__ == "__main__":
    raise SystemExit(main())
