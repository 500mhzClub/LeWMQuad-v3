#!/usr/bin/env python3
"""One-shot controller for the memory-role factorized joint-JEPA V1 probe.

The controller keeps V25's reviewed physical route and adds two RGB-only
four-row routes: corrected-H6 immediate control and manifest-bound place
triplets.  It owns lifecycle and accounting only; tensor work and evaluation
remain in their dedicated modules.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping, Sequence

from scripts import (
    execute_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v26
    as v26,
)
from scripts import (
    execute_go2_rgb_swept_progress_survival_joint_jepa_v13_camera_evidence_bottleneck
    as physical_executor,
)


SCHEMA_PREFIX = "lewm_go2_rgb_memory_role_factorized_joint_jepa_v1"
PREREGISTRATION_PATH = (
    "docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v1_"
    "preregistration_2026-07-30.md"
)
PREREGISTRATION_COMMIT = "01d78284a22a52816a41f31a78411491714b4f9c"
SPLIT_INTEGRITY_AMENDMENT_PATH = (
    "docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v1_"
    "split_integrity_amendment_2026-07-30.md"
)
SPLIT_INTEGRITY_AMENDMENT_COMMIT = (
    "5a1535567bf00b8e47d67d8966ef42a52726bd5b"
)
SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v1_"
    "source_manifest_2026-07-30.json"
)
SOURCE_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v1_"
    "source_review_2026-07-30.json"
)
CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v1_"
    "clean_export_certification_2026-07-30.json"
)
AUTHORITY_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v1_"
    "execution_authorization_2026-07-30.json"
)
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_memory_role_factorized_joint_jepa_v1/attempt_v1"
)
CERTIFIED_SOURCE_ROOT = (
    "/home/andrewknowles/Workspace/"
    "LeWMQuad-v3-memory-role-factorized-joint-jepa-v1-source"
)
MODEL_CLASS_NAME = "MemoryRoleFactorizedJointJepaV1"
MODEL_MODULE_NAME = "lewm.models.memory_role_factorized_joint_jepa_v1"
TRAINING_MODULE_NAME = (
    "scripts.run_go2_rgb_memory_role_factorized_joint_jepa_v1"
)
EVALUATION_MODULE_NAME = (
    "scripts.evaluate_go2_rgb_memory_role_factorized_joint_jepa_v1"
)

MAXIMUM_UPDATES = 400
MAXIMUM_PRESENTATIONS = 12_800
PHYSICAL_PRESENTATIONS_PER_UPDATE = 16
LOCAL_PRESENTATIONS_PER_UPDATE = 8
PLACE_PRESENTATIONS_PER_UPDATE = 8
PRESENTATIONS_PER_UPDATE = 32
OBSERVATION_UPDATES = (0, 100, 400)
TERMINAL_UPDATES = (400,)
RGB_ROOT_RELATIVE_PATH = Path(".generated/datagen_full/render_textured_v03")
PLACE_TRIPLET_ROOT_RELATIVE_PATH = Path(
    ".generated/go2_memory_role_place_triplet_index_v1"
)
LOCAL_TRAIN_ROWS_PER_UPDATE = 8
PLACE_TRAIN_ROWS_PER_UPDATE = 8
ROLE_MICROBATCH_SIZE = 4
MAXIMUM_RGB_FILE_BYTES = 4 * 1024 * 1024
LOCAL_TRAIN_ROW_COUNT = 3_200
LOCAL_SELECTION_ROW_COUNT = 1_994
LOCAL_TRAIN_SOURCE_INDEX_SHA256 = (
    "263e72b1bfff24b059d1d46f0ec1859dbc497602e82c3f5e02f628e4f26809a5"
)
LOCAL_SELECTION_SOURCE_INDEX_SHA256 = (
    "a9344429cdafca23cbce8e26ef18756423ac364c12bb1c4d3af78e1ab4a533b9"
)

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
    | getattr(os, "O_NONBLOCK", 0)
)

# Exact inherited physical runtime API consumed by the V13 composer.
PHYSICAL_RUNTIME_INPUT_BINDING_NAMES = v26.RUNTIME_INPUT_BINDING_NAMES
ROLE_RUNTIME_INPUT_BINDING_NAMES = (
    "h6_train_index",
    "h6_checkpoint_selection_index",
    "place_triplet_manifest",
    "place_triplet_train_index",
    "place_triplet_checkpoint_selection_index",
)
RUNTIME_INPUT_BINDING_NAMES = (
    *PHYSICAL_RUNTIME_INPUT_BINDING_NAMES,
    *ROLE_RUNTIME_INPUT_BINDING_NAMES,
)
CHECKPOINT_SCHEDULE_PREFIX_SHA256 = v26.CHECKPOINT_SCHEDULE_PREFIX_SHA256
REGISTERED_FAMILIES = v26.REGISTERED_FAMILIES
SCOPES = v26.SCOPES
V12_GATE_CHECK_NAMES = v26.V12_GATE_CHECK_NAMES
CONTROL_NAMES = v26.CONTROL_NAMES
MATCHED_UPDATE400_THRESHOLDS = v26.MATCHED_UPDATE400_THRESHOLDS
EXPECTED_RUNTIME_FINGERPRINT = v26.EXPECTED_RUNTIME_FINGERPRINT
flatten_physical_metrics_v13 = v26.flatten_physical_metrics_v26
registered_wrong_rgb_mapping_v13 = v26.registered_wrong_rgb_mapping_v26
_canonical_json_bytes = v26._canonical_json_bytes
_write_immutable_json_v13 = v26._write_immutable_json_v13


def _content_bound(core: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(core)
    value.pop("content_sha256", None)
    value["content_sha256"] = hashlib.sha256(
        _canonical_json_bytes(value)
    ).hexdigest()
    return value


def validate_content_bound_v1(value: Any) -> dict[str, Any]:
    if type(value) is not dict or type(value.get("content_sha256")) is not str:
        raise TypeError("memory-role content-bound value must be an exact object")
    observed = value["content_sha256"]
    core = dict(value)
    core.pop("content_sha256")
    if observed != hashlib.sha256(_canonical_json_bytes(core)).hexdigest():
        raise RuntimeError("memory-role content binding changed")
    return dict(value)


def _binding(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"memory-role {name} binding is absent")
    result = dict(value)
    if (
        set(result) != {"path", "file_sha256", "byte_count"}
        or type(result["path"]) is not str
        or not result["path"]
        or type(result["file_sha256"]) is not str
        or len(result["file_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in result["file_sha256"])
        or type(result["byte_count"]) is not int
        or result["byte_count"] <= 0
    ):
        raise TypeError(f"memory-role {name} binding changed")
    return result


def validate_future_execution_prerequisites_v1(authority: Any) -> dict[str, Any]:
    value = validate_content_bound_v1(authority)
    required = {
        "schema": f"{SCHEMA_PREFIX}_future_execution_authority_v1",
        "status": "AUTHORIZED_CERTIFIED_NARROW_EXPORT_ONE_SHOT",
        "scientific_payload_authorized": True,
        "one_shot": True,
        "maximum_updates": MAXIMUM_UPDATES,
        "maximum_presentations": MAXIMUM_PRESENTATIONS,
        "retry_authorized": False,
        "resume_authorized": False,
        "certified_source_root": CERTIFIED_SOURCE_ROOT,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "split_integrity_amendment_commit": SPLIT_INTEGRITY_AMENDMENT_COMMIT,
    }
    if any(value.get(name) != expected for name, expected in required.items()):
        raise PermissionError("memory-role authority identity or cap changed")
    selectors = value.get("selectors")
    if selectors != {
        "executor_module": __name__,
        "model_module": MODEL_MODULE_NAME,
        "model_class": MODEL_CLASS_NAME,
        "training_module": TRAINING_MODULE_NAME,
        "evaluation_module": EVALUATION_MODULE_NAME,
    }:
        raise PermissionError("memory-role runtime selectors changed")
    expected_data_root = "/home/andrewknowles/Workspace/LeWMQuad-v3"
    if value.get("runtime_data_root") not in {
        str(Path(__file__).resolve().parents[1]),
        expected_data_root,
    }:
        raise PermissionError("memory-role runtime-data root changed")
    runtime_inputs = value.get("runtime_inputs")
    if type(runtime_inputs) is not dict or set(runtime_inputs) != set(
        RUNTIME_INPUT_BINDING_NAMES
    ):
        raise PermissionError("memory-role runtime input inventory changed")
    for name in RUNTIME_INPUT_BINDING_NAMES:
        _binding(runtime_inputs[name], name=name)
    certification = value.get("clean_export_certification")
    if not isinstance(certification, Mapping) or set(certification) != {
        "path",
        "file_sha256",
        "byte_count",
        "content_sha256",
    }:
        raise PermissionError("memory-role authority lacks certification")
    if certification.get("path") != CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH:
        raise PermissionError("memory-role certification path changed")
    return value


# Facade expected by the frozen V13 physical composer.
validate_content_bound_v13 = validate_content_bound_v1
validate_future_execution_prerequisites_v13 = (
    validate_future_execution_prerequisites_v1
)


def _read_certification(repository_root: Path) -> dict[str, Any]:
    path = repository_root / CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH
    if path.is_symlink() or not path.is_file():
        raise PermissionError("memory-role clean-export certification is absent")
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PermissionError("memory-role certification is invalid") from error
    value = validate_content_bound_v1(value)
    if (
        value.get("schema") != f"{SCHEMA_PREFIX}_clean_export_certification_v1"
        or value.get("status") != "PASS_CLEAN_EXPORT_CERTIFIED"
        or value.get("passed") is not True
        or value.get("certified_source_root") != str(repository_root)
    ):
        raise PermissionError("memory-role certification identity changed")
    return value


def _protected_source_path(relative: str) -> bool:
    pure = PurePosixPath(relative)
    folded = tuple(part.casefold() for part in pure.parts)
    return bool(
        pure.is_absolute()
        or not pure.parts
        or relative != pure.as_posix()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or any(part == "sealed_test.json" for part in folded)
        or any(part == "sealed" or part.startswith("sealed_") for part in folded)
        or any(
            part in {"heldout", "held_out"}
            or part.startswith("heldout_")
            or part.startswith("held_out_")
            for part in folded
        )
        or any(part in {".generated", "data"} for part in folded)
    )


def validate_bound_sources_v1(repository_root: Path) -> dict[str, Any]:
    root = Path(repository_root).resolve(strict=True)
    certification = _read_certification(root)
    bindings = certification.get("bindings")
    if not isinstance(bindings, list) or not bindings:
        raise PermissionError("memory-role certified source inventory is absent")
    canonical = hashlib.sha256(_canonical_json_bytes(bindings)).hexdigest()
    if canonical != certification.get("bindings_sha256"):
        raise PermissionError("memory-role source inventory binding changed")
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_binding in bindings:
        binding = _binding(raw_binding, name="certified source")
        relative = binding["path"]
        if _protected_source_path(relative) or relative in seen:
            raise PermissionError("memory-role certified source path is protected")
        path = root.joinpath(*PurePosixPath(relative).parts)
        try:
            resolved = path.resolve(strict=True)
        except (FileNotFoundError, OSError) as error:
            raise PermissionError("memory-role certified source is absent") from error
        if (
            resolved != path.absolute()
            or not resolved.is_relative_to(root)
            or path.is_symlink()
            or not path.is_file()
        ):
            raise PermissionError("memory-role certified source escaped")
        payload = path.read_bytes()
        if (
            len(payload) != binding["byte_count"]
            or hashlib.sha256(payload).hexdigest() != binding["file_sha256"]
        ):
            raise PermissionError(f"memory-role certified source changed: {relative}")
        seen.add(relative)
        validated.append(binding)
    return {
        "validated_path_count": len(validated),
        "bindings_sha256": canonical,
        "certification_content_sha256": certification["content_sha256"],
        "passed": True,
    }


validate_bound_sources_v13 = validate_bound_sources_v1


def reserve_attempt_v1(
    repository_root: Path,
    authority: Mapping[str, Any],
    *,
    created_utc: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve(strict=True)
    validated = validate_future_execution_prerequisites_v1(dict(authority))
    output = root / OUTPUT_ROOT_RELATIVE_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    os.chmod(output, 0o700, follow_symlinks=False)
    info = os.lstat(output)
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700:
        raise PermissionError("memory-role attempt root mode changed")
    reservation = _content_bound(
        {
            "schema": f"{SCHEMA_PREFIX}_attempt_reservation_v1",
            "status": "RESERVED_ONE_SHOT",
            "created_utc": created_utc,
            "authority_sha256": hashlib.sha256(
                _canonical_json_bytes(validated)
            ).hexdigest(),
            "output_root": OUTPUT_ROOT_RELATIVE_PATH,
            "maximum_updates": MAXIMUM_UPDATES,
            "maximum_presentations": MAXIMUM_PRESENTATIONS,
            "attempt": 1,
        }
    )
    path = output / "reservation.json"
    with path.open("xb") as handle:
        handle.write(_canonical_json_bytes(reservation) + b"\n")
    os.chmod(path, 0o444, follow_symlinks=False)
    if stat.S_IMODE(os.lstat(path).st_mode) != 0o444:
        raise PermissionError("memory-role reservation file mode changed")
    return reservation


def validate_attempt_reservation_v1(value: Any) -> dict[str, Any]:
    reservation = validate_content_bound_v1(value)
    required = {
        "schema": f"{SCHEMA_PREFIX}_attempt_reservation_v1",
        "status": "RESERVED_ONE_SHOT",
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "maximum_updates": MAXIMUM_UPDATES,
        "maximum_presentations": MAXIMUM_PRESENTATIONS,
        "attempt": 1,
    }
    if any(reservation.get(name) != expected for name, expected in required.items()):
        raise PermissionError("memory-role reservation identity changed")
    return reservation


reserve_attempt_v13 = reserve_attempt_v1
validate_attempt_reservation_v13 = validate_attempt_reservation_v1


def terminalize_failure_v1(
    output_root: Path,
    reservation: Mapping[str, Any],
    *,
    stage: str,
    error: BaseException,
    created_utc: str,
    partial_checkpoint_binding: Mapping[str, Any] | None = None,
    failure_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_attempt_reservation_v1(dict(reservation))
    checkpoint_path = Path(output_root) / "checkpoint_update_400.pt"
    checkpoint_binding: dict[str, Any] | None = None
    checkpoint_quarantined = False
    hash_reads = 0
    if checkpoint_path.exists() or checkpoint_path.is_symlink():
        info = os.lstat(checkpoint_path)
        if checkpoint_path.is_symlink() or not stat.S_ISREG(info.st_mode):
            raise PermissionError("memory-role partial checkpoint changed type")
        raw = checkpoint_path.read_bytes()
        hash_reads = 1
        observed = {
            "path": "checkpoint_update_400.pt",
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
        }
        if partial_checkpoint_binding is not None and _binding(
            partial_checkpoint_binding, name="partial checkpoint"
        ) != observed:
            raise PermissionError("memory-role partial checkpoint binding changed")
        checkpoint_binding = observed
        os.chmod(checkpoint_path, 0o000, follow_symlinks=False)
        checkpoint_quarantined = True
    elif partial_checkpoint_binding is not None:
        raise PermissionError("memory-role bound partial checkpoint is absent")
    core = {
        "schema": f"{SCHEMA_PREFIX}_exception_failure_v1",
        "status": "FAIL_EXCEPTION_TERMINAL_NO_RETRY_NO_RESUME",
        "stage": stage,
        "created_utc": created_utc,
        "exception_type": type(error).__name__,
        "exception_message_sha256": hashlib.sha256(
            str(error).encode("utf-8")
        ).hexdigest(),
        "attempt_consumed": True,
        "checkpoint_published": checkpoint_binding is not None,
        "checkpoint_present_at_terminal": checkpoint_binding is not None,
        "checkpoint_quarantined": checkpoint_quarantined,
        "checkpoint": checkpoint_binding,
        "checkpoint_binding_available": checkpoint_binding is not None,
        "checkpoint_terminalization_hash_read_count": hash_reads,
        "checkpoint_deserialized": False,
        "checkpoint_access_authorized": False,
        "failure_context": (
            None if failure_context is None else dict(failure_context)
        ),
        "retry_authorized": False,
        "resume_authorized": False,
    }
    return _write_immutable_json_v13(Path(output_root) / "failure.json", core)


terminalize_failure_v13 = terminalize_failure_v1


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"memory-role {name} must be a dataclass or mapping")


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"memory-role {name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"memory-role {name} is nonfinite")
    return result


def validate_update_integrity_v1(
    runtime: Any,
    model: Any,
    result: Any,
    *,
    update: int,
) -> dict[str, Any]:
    accounting = _mapping(result.accounting, name="accounting")
    multipliers = {
        "updates": 1,
        "presentations": 32,
        "physical_presentations": 16,
        "local_presentations": 8,
        "place_presentations": 8,
        "rgb_decodes": 72,
        "physical_rgb_decodes": 32,
        "local_rgb_decodes": 16,
        "place_rgb_decodes": 24,
        "online_rgb_encodings": 48,
        "ema_target_rgb_encodings": 24,
        "physical_microbatch_graphs": 4,
        "local_microbatch_graphs": 2,
        "place_microbatch_graphs": 2,
        "autograd_grad_calls": 16,
        "optimizer_steps": 1,
        "ema_steps": 1,
    }
    expected = {name: multiplier * update for name, multiplier in multipliers.items()}
    if accounting != expected:
        raise RuntimeError("memory-role update accounting changed")
    route_names = (
        "camera_shared",
        "joint_shared",
        "representation",
        "predictor",
        "predictor_core_protected_survival_output",
        "immediate_action_local_control",
        "same_place_retrieval_key",
    )
    routes = {
        name: _mapping(value, name=f"route {name}")
        for name, value in result.gradient_routes.items()
    }
    if tuple(routes) != route_names:
        raise RuntimeError("memory-role gradient route order or membership changed")
    for name, route in routes.items():
        preclip = _finite(route.get("preclip_l2"), name=f"{name} preclip L2")
        scale = _finite(route.get("applied_scale"), name=f"{name} scale")
        if route.get("absent_tensor_gradient_count") != 0 or preclip < 0.0:
            raise RuntimeError(f"memory-role route {name} has an invalid gradient")
        if name != "representation" and not preclip > 0.0:
            raise RuntimeError(f"memory-role required route {name} is zero")
        if not 0.0 < scale <= 1.0:
            raise RuntimeError(f"memory-role route {name} clipping scale changed")
    losses = {
        name: _finite(value, name=f"loss {name}")
        for name, value in result.mean_losses.items()
    }
    required_losses = {"S", "U", "R", "O", "N", "C", "J24", "L", "local", "place", "total"}
    if not required_losses.issubset(losses):
        raise RuntimeError("memory-role required loss receipt is incomplete")

    local = _mapping(result.local_diagnostics, name="local diagnostics")
    place = _mapping(result.place_diagnostics, name="place diagnostics")
    if (
        local.get("mechanism") != "immediate_action_local_control"
        or place.get("mechanism") != "same_place_retrieval_key"
        or len(local.get("correct_energy_per_row", ())) != 8
        or len(local.get("wrong_energy_per_row", ())) != 8
        or len(place.get("positive_energy_per_row", ())) != 8
        or len(place.get("negative_energy_per_row", ())) != 8
    ):
        raise RuntimeError("memory-role route diagnostics changed")
    for values in (
        local["correct_energy_per_row"],
        local["wrong_energy_per_row"],
        place["positive_energy_per_row"],
        place["negative_energy_per_row"],
    ):
        if any(_finite(value, name="diagnostic energy") < 0.0 for value in values):
            raise RuntimeError("memory-role diagnostic energy is negative")
    if (
        result.target_gradient_tensor_count != 0
        or result.optimizer_steps_this_update != 1
        or result.ema_steps_this_update != 1
        or int(model.ema_update_count.item()) != update
        or any(
            parameter.grad is not None
            for module in model.target_modules()
            for parameter in module.parameters()
        )
    ):
        raise RuntimeError("memory-role target, optimizer, or EMA integrity failed")
    for value in model.state_dict().values():
        if value.is_floating_point() and not bool(runtime.torch.isfinite(value).all()):
            raise FloatingPointError("memory-role model state became nonfinite")
    return {
        "schema": f"{SCHEMA_PREFIX}_update_integrity_v1",
        "update": update,
        "accounting": accounting,
        "gradient_routes": routes,
        "mean_losses": losses,
        "local_diagnostics": local,
        "place_diagnostics": place,
        "target_gradient_tensor_count": 0,
        "passed": True,
    }


def _publish_json(
    publisher: Any, relative_path: str, core: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = publisher.publish_json(relative_path, dict(core))
    if type(result) is not dict or set(result) != {"value", "binding"}:
        raise RuntimeError("memory-role publisher JSON result changed")
    return validate_content_bound_v1(result["value"]), dict(result["binding"])


def _serialize_checkpoint_v1(
    runtime: Any,
    model: Any,
    optimizer: Any,
    accounting: Any,
    authority: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    state = {
        "schema": f"{SCHEMA_PREFIX}_development_scale_seed_v1",
        "update": 400,
        "model_module": MODEL_MODULE_NAME,
        "model_class": MODEL_CLASS_NAME,
        "model_state_dict": {
            name: value.detach().cpu().contiguous().clone()
            for name, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "accounting": _mapping(accounting, name="checkpoint accounting"),
        "rng": {
            "torch_cpu": runtime.torch.random.get_rng_state().clone(),
            "visible_gpu": tuple(
                value.clone() for value in runtime.torch.cuda.get_rng_state_all()
            ),
        },
        "authority_sha256": hashlib.sha256(
            _canonical_json_bytes(authority)
        ).hexdigest(),
        "resume_authorized": False,
    }
    buffer = io.BytesIO()
    runtime.torch.save(state, buffer)
    raw = buffer.getvalue()
    return raw, {
        "schema": f"{SCHEMA_PREFIX}_development_scale_seed_binding_v1",
        "update": 400,
        "byte_count": len(raw),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "resume_authorized": False,
        "navigation_authorized": False,
        "held_out_authorized": False,
    }


def _file_fingerprint(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _open_absolute_directory(path: Path) -> int:
    value = Path(path)
    if (
        not value.is_absolute()
        or any(part in {"", ".", ".."} for part in value.parts[1:])
        or not getattr(os, "O_NOFOLLOW", 0)
        or not getattr(os, "O_DIRECTORY", 0)
    ):
        raise PermissionError("memory-role RGB directory is not canonical no-follow")
    descriptor = os.open(value.anchor, _DIR_FLAGS)
    try:
        for component in value.parts[1:]:
            child = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


class _SafeLocalRGBLoaderV1:
    """No-follow RGB loader restricted to registered H6 e2/e3 leaves."""

    def __init__(self, runtime_data_root: Path, rows: Sequence[Any], data: Any) -> None:
        root = Path(runtime_data_root)
        if not root.is_absolute():
            raise PermissionError("memory-role local RGB root must be absolute")
        self._rgb_root = root / RGB_ROOT_RELATIVE_PATH
        self._data = data
        self._rows: dict[tuple[str, int], Any] = {}
        for row in rows:
            key = (str(row.role), int(row.index))
            if key in self._rows:
                raise RuntimeError("memory-role registered H6 row repeats")
            self._rows[key] = row
        self._consumed: dict[str, dict[str, Any]] = {}
        self._tensor_requests = 0
        self._open_attempts = 0
        self._open_successes = 0
        self._decode_successes = 0
        self._byte_count = 0
        self._closed = False

    def _registered(self, row: Any) -> Any:
        if self._closed:
            raise RuntimeError("memory-role local RGB loader is closed")
        registered = self._rows.get((str(getattr(row, "role", "")), getattr(row, "index", -1)))
        if registered is None or registered != row:
            raise PermissionError("memory-role H6 row is not registered")
        return registered

    def _read_raw(self, row: Any, leaf: str, *, record: bool) -> bytes:
        registered = self._registered(row)
        if leaf not in (registered.rgb[2], registered.rgb[3]):
            raise PermissionError("memory-role local RGB leaf is not e2 or e3")
        canonical, _frame, _environment = self._data._validate_rgb_leaf(
            leaf, scene_id=registered.scene_id
        )
        parts = PurePosixPath(canonical).parts
        descriptor = _open_absolute_directory(self._rgb_root)
        image_fd: int | None = None
        try:
            for component in parts[:-1]:
                child = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            if record:
                self._open_attempts += 1
            image_fd = os.open(parts[-1], _READ_FLAGS, dir_fd=descriptor)
            before = os.fstat(image_fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or not 0 < before.st_size <= MAXIMUM_RGB_FILE_BYTES
            ):
                raise PermissionError("memory-role local RGB leaf is unsafe")
            chunks: list[bytes] = []
            consumed = 0
            while True:
                chunk = os.read(
                    image_fd, min(1024 * 1024, MAXIMUM_RGB_FILE_BYTES)
                )
                if not chunk:
                    break
                consumed += len(chunk)
                if consumed > MAXIMUM_RGB_FILE_BYTES:
                    raise PermissionError("memory-role local RGB exceeded byte cap")
                chunks.append(chunk)
            after = os.fstat(image_fd)
            raw = b"".join(chunks)
            if _file_fingerprint(before) != _file_fingerprint(after):
                raise PermissionError("memory-role local RGB changed while reading")
            digest = hashlib.sha256(raw).hexdigest()
            previous = self._consumed.get(canonical)
            binding = {
                "path": canonical,
                "file_sha256": digest,
                "byte_count": len(raw),
                "role": str(registered.role),
                "row_index": int(registered.index),
                "leaf": leaf,
            }
            if previous is not None and any(
                previous[name] != binding[name]
                for name in ("path", "file_sha256", "byte_count")
            ):
                raise PermissionError("memory-role repeated local RGB identity changed")
            if record:
                if previous is None:
                    self._consumed[canonical] = binding
                self._open_successes += 1
                self._byte_count += len(raw)
            return raw
        finally:
            if image_fd is not None:
                os.close(image_fd)
            os.close(descriptor)

    def load_pair(self, row: Any) -> dict[str, Any]:
        registered = self._registered(row)
        current_raw = self._read_raw(registered, registered.rgb[2], record=True)
        next_raw = self._read_raw(registered, registered.rgb[3], record=True)
        current = self._data.rectify_h6_rgb_bytes(current_raw)
        next_rgb = self._data.rectify_h6_rgb_bytes(next_raw)
        self._tensor_requests += 2
        self._decode_successes += 2
        return {
            "current_rgb": current,
            "next_rgb": next_rgb,
            "action": int(registered.actions[2]),
        }

    def access_receipt(self) -> dict[str, int]:
        return {
            "rgb_tensor_request_count": self._tensor_requests,
            "rgb_open_attempt_count": self._open_attempts,
            "rgb_open_success_count": self._open_successes,
            "rgb_decode_success_count": self._decode_successes,
            "rgb_byte_count": self._byte_count,
            "unique_rgb_leaf_count": len(self._consumed),
        }

    def terminal_rehash(self) -> dict[str, Any]:
        consumed = tuple(sorted(self._consumed.values(), key=lambda item: item["path"]))
        for binding in consumed:
            row = self._rows[(binding["role"], binding["row_index"])]
            raw = self._read_raw(row, binding["leaf"], record=False)
            if (
                len(raw) != binding["byte_count"]
                or hashlib.sha256(raw).hexdigest() != binding["file_sha256"]
            ):
                raise PermissionError("memory-role local RGB terminal rehash changed")
        public = [
            {
                "path": binding["path"],
                "file_sha256": binding["file_sha256"],
                "byte_count": binding["byte_count"],
            }
            for binding in consumed
        ]
        return {
            "terminal_rgb_rehash_count": len(public),
            "consumed_rgb_bindings_sha256": hashlib.sha256(
                _canonical_json_bytes(public)
            ).hexdigest(),
            "all_consumed_local_rgb_rehashed": True,
        }

    def close(self) -> None:
        self._closed = True


class MemoryRoleRuntimeV1:
    """Controller-facing owner of exact local/place indexes and RGB access."""

    def __init__(
        self,
        *,
        runtime_data_root: Path,
        runtime_inputs: Mapping[str, Any],
        h6_data: Any,
        place_data: Any,
        evaluation: Any,
        training: Any,
        h6_train_rows: Sequence[Any],
        h6_selection_rows: Sequence[Any],
        place_train_rows: Sequence[Any],
        place_selection_rows: Sequence[Any],
        physical_train_scene_ids: Sequence[str],
        physical_selection_scene_ids: Sequence[str],
        audits: Mapping[str, Any],
    ) -> None:
        self.runtime_data_root = Path(runtime_data_root)
        self.runtime_inputs = dict(runtime_inputs)
        self.h6_data = h6_data
        self.place_data = place_data
        self.evaluation = evaluation
        self.training = training
        self.h6_train_rows = tuple(h6_train_rows)
        self.h6_selection_rows = tuple(h6_selection_rows)
        self.place_train_rows = tuple(place_train_rows)
        self.place_selection_rows = tuple(place_selection_rows)
        self.audits = {name: dict(value) for name, value in audits.items()}

        physical_train_scenes = frozenset(physical_train_scene_ids)
        physical_selection_scenes = frozenset(physical_selection_scene_ids)
        place_training_scenes = {row.scene_id for row in self.place_train_rows}
        place_selection_scenes = {
            row.scene_id for row in self.place_selection_rows
        }
        if (
            len(physical_train_scenes) != 72
            or len(physical_selection_scenes) != 8
            or physical_train_scenes & physical_selection_scenes
            or place_selection_scenes != physical_selection_scenes
            or not place_training_scenes.issubset(physical_train_scenes)
        ):
            raise PermissionError("memory-role physical/place role split changed")
        eligible_local_train = tuple(
            row
            for row in self.h6_train_rows
            if row.scene_id not in physical_selection_scenes
        )
        self.local_train_rows = eligible_local_train[:LOCAL_TRAIN_ROW_COUNT]
        self.local_selection_source_rows = tuple(
            row
            for row in self.h6_selection_rows
            if row.scene_id not in physical_train_scenes
        )

        def ordered_index_sha(rows: Sequence[Any]) -> str:
            return hashlib.sha256(
                _canonical_json_bytes([int(row.index) for row in rows])
            ).hexdigest()

        if (
            len(self.local_train_rows) != LOCAL_TRAIN_ROW_COUNT
            or len(self.local_selection_source_rows) != LOCAL_SELECTION_ROW_COUNT
            or ordered_index_sha(self.local_train_rows)
            != LOCAL_TRAIN_SOURCE_INDEX_SHA256
            or ordered_index_sha(self.local_selection_source_rows)
            != LOCAL_SELECTION_SOURCE_INDEX_SHA256
        ):
            raise PermissionError("memory-role split-integrity adapter changed")
        self.training_scene_ids = frozenset(
            physical_train_scenes
            | {row.scene_id for row in self.local_train_rows}
        )
        selection_scenes = physical_selection_scenes | {
            row.scene_id for row in self.local_selection_source_rows
        }
        if not self.training_scene_ids or self.training_scene_ids & selection_scenes:
            raise PermissionError("memory-role train/selection scene custody changed")
        self.local_selection_rows = tuple(
            evaluation.LocalSelectionRowV1(
                index=index,
                role="checkpoint_selection",
                family=row.family,
                scene_id=row.scene_id,
                action=int(row.actions[2]),
            )
            for index, row in enumerate(self.local_selection_source_rows)
        )
        place_family_counts = {
            family: sum(row.family == family for row in self.place_selection_rows)
            for family in self.evaluation.FAMILIES_V1
        }
        if (
            len(self.place_selection_rows)
            != self.evaluation.PLACE_SELECTION_ROW_COUNT_V1
            or place_family_counts != self.evaluation.PLACE_FAMILY_ROW_COUNTS_V1
        ):
            raise PermissionError("memory-role place selection quota contract changed")
        self._local_loader = _SafeLocalRGBLoaderV1(
            self.runtime_data_root,
            (*self.local_train_rows, *self.local_selection_source_rows),
            self.h6_data,
        )
        self._place_rows = {
            (row.role, row.index): row
            for row in (*self.place_train_rows, *self.place_selection_rows)
        }
        self._place_loader_calls = 0
        self._place_loaded_row_keys: set[tuple[str, int]] = set()
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("memory-role role runtime is closed")

    def _load_place_triplet(self, row: Any) -> Any:
        self._require_open()
        registered = self._place_rows.get((getattr(row, "role", ""), getattr(row, "index", -1)))
        if registered is None or registered != row:
            raise PermissionError("memory-role place row is not registered")
        result = self.place_data.load_rgb_triplet(self.runtime_data_root, registered)
        self._place_loader_calls += 1
        self._place_loaded_row_keys.add((registered.role, registered.index))
        return result

    def _load_local_selection_pair(self, row: Any) -> dict[str, Any]:
        self._require_open()
        if (
            not isinstance(row, self.evaluation.LocalSelectionRowV1)
            or not 0 <= row.index < len(self.local_selection_source_rows)
        ):
            raise PermissionError("memory-role local selection row is not registered")
        source = self.local_selection_source_rows[row.index]
        expected = self.local_selection_rows[row.index]
        if row != expected:
            raise PermissionError("memory-role local selection identity changed")
        return self._local_loader.load_pair(source)

    def preflight_receipt(self) -> dict[str, Any]:
        self._require_open()
        if any(self._local_loader.access_receipt().values()) or self._place_loader_calls:
            raise RuntimeError("memory-role preflight must precede RGB access")
        return {
            "schema": f"{SCHEMA_PREFIX}_role_runtime_preflight_v1",
            "status": "PASS_METADATA_ONLY_PREFLIGHT",
            "audits": self.audits,
            "h6_train_row_count": len(self.h6_train_rows),
            "h6_checkpoint_selection_row_count": len(self.h6_selection_rows),
            "effective_local_train_row_count": len(self.local_train_rows),
            "effective_local_checkpoint_selection_row_count": len(
                self.local_selection_source_rows
            ),
            "local_train_source_index_sha256": LOCAL_TRAIN_SOURCE_INDEX_SHA256,
            "local_checkpoint_selection_source_index_sha256": (
                LOCAL_SELECTION_SOURCE_INDEX_SHA256
            ),
            "place_train_row_count": len(self.place_train_rows),
            "place_checkpoint_selection_row_count": len(self.place_selection_rows),
            "training_scene_count": len(self.training_scene_ids),
            "probability_calibration_opened": False,
            "held_out_or_sealed_opened": False,
            "rgb_open_count": 0,
            "gpu_use_count": 0,
        }

    @staticmethod
    def _stack(torch: Any, values: Sequence[Any], device: Any) -> Any:
        result = torch.stack(tuple(values), dim=0).to(
            device=torch.device(device), non_blocking=False
        )
        if (
            tuple(result.shape) != (ROLE_MICROBATCH_SIZE, 3, 112, 112)
            or result.dtype != torch.float32
            or not bool(torch.isfinite(result).all())
        ):
            raise RuntimeError("memory-role stacked RGB microbatch changed")
        return result

    def build_local_train_microbatches(
        self, update: int, device: Any
    ) -> tuple[dict[str, Any], ...]:
        self._require_open()
        if type(update) is not int or not 1 <= update <= MAXIMUM_UPDATES:
            raise PermissionError("memory-role local update left 1..400")
        torch = self.training.v25._tensor_core._runtime_apis()[0]
        first = LOCAL_TRAIN_ROWS_PER_UPDATE * (update - 1)
        selected = self.local_train_rows[first : first + LOCAL_TRAIN_ROWS_PER_UPDATE]
        if len(selected) != LOCAL_TRAIN_ROWS_PER_UPDATE:
            raise RuntimeError("memory-role local train schedule exhausted")
        batches: list[dict[str, Any]] = []
        for start in range(0, len(selected), ROLE_MICROBATCH_SIZE):
            rows = selected[start : start + ROLE_MICROBATCH_SIZE]
            pairs = tuple(self._local_loader.load_pair(row) for row in rows)
            batch = {
                self.training.LOCAL_CURRENT_RGB_KEY_V1: self._stack(
                    torch, [pair["current_rgb"] for pair in pairs], device
                ),
                self.training.LOCAL_NEXT_RGB_KEY_V1: self._stack(
                    torch, [pair["next_rgb"] for pair in pairs], device
                ),
                self.training.LOCAL_ACTION_KEY_V1: torch.tensor(
                    [pair["action"] for pair in pairs],
                    dtype=torch.long,
                    device=torch.device(device),
                ),
            }
            if tuple(batch) != self.training.REQUIRED_LOCAL_BATCH_KEYS_V1:
                raise RuntimeError("memory-role local batch key order changed")
            batches.append(batch)
        return tuple(batches)

    def build_place_train_microbatches(
        self, update: int, device: Any
    ) -> tuple[dict[str, Any], ...]:
        self._require_open()
        if type(update) is not int or not 1 <= update <= MAXIMUM_UPDATES:
            raise PermissionError("memory-role place update left 1..400")
        torch = self.training.v25._tensor_core._runtime_apis()[0]
        first = PLACE_TRAIN_ROWS_PER_UPDATE * (update - 1)
        selected = self.place_train_rows[first : first + PLACE_TRAIN_ROWS_PER_UPDATE]
        if len(selected) != PLACE_TRAIN_ROWS_PER_UPDATE:
            raise RuntimeError("memory-role place train schedule exhausted")
        batches: list[dict[str, Any]] = []
        for start in range(0, len(selected), ROLE_MICROBATCH_SIZE):
            rows = selected[start : start + ROLE_MICROBATCH_SIZE]
            triplets = tuple(self._load_place_triplet(row) for row in rows)
            batch = {
                self.training.PLACE_ANCHOR_RGB_KEY_V1: self._stack(
                    torch, [item.anchor_rgb for item in triplets], device
                ),
                self.training.PLACE_POSITIVE_RGB_KEY_V1: self._stack(
                    torch, [item.positive_rgb for item in triplets], device
                ),
                self.training.PLACE_NEGATIVE_RGB_KEY_V1: self._stack(
                    torch, [item.negative_rgb for item in triplets], device
                ),
            }
            if tuple(batch) != self.training.REQUIRED_PLACE_BATCH_KEYS_V1:
                raise RuntimeError("memory-role place batch key order changed")
            batches.append(batch)
        return tuple(batches)

    def evaluate_role_metrics(
        self, model: Any, *, update: int, device: Any
    ) -> dict[str, Any]:
        self._require_open()
        result = self.evaluation.evaluate_checkpoint_selection_v1(
            model,
            place_rows=self.place_selection_rows,
            load_triplet=self._load_place_triplet,
            local_rows=self.local_selection_rows,
            load_local_pair=self._load_local_selection_pair,
            device=device,
            training_scene_ids=self.training_scene_ids,
            update=update,
        )
        integrity_checks = {
            "place_target_integrity": result["place"]["target_integrity"]["passed"] is True,
            "local_target_integrity": result["local"]["target_integrity"]["passed"] is True,
            "probability_calibration_opened_false": True,
            "held_out_or_sealed_opened_false": True,
        }
        return {
            **result,
            "integrity": {
                "checks": integrity_checks,
                "passed": all(integrity_checks.values()),
            },
        }

    def terminal_access_receipt(self) -> dict[str, Any]:
        self._require_open()
        h6_train, h6_train_audit = self.h6_data.load_bound_index(
            self.runtime_data_root, role="train"
        )
        h6_selection, h6_selection_audit = self.h6_data.load_bound_index(
            self.runtime_data_root, role="val"
        )
        manifest_sha = self.runtime_inputs["place_triplet_manifest"]["file_sha256"]
        place_train, place_train_audit = self.place_data.load_index(
            self.runtime_data_root,
            PLACE_TRIPLET_ROOT_RELATIVE_PATH,
            role="train",
            expected_manifest_sha256=manifest_sha,
        )
        place_selection, place_selection_audit = self.place_data.load_index(
            self.runtime_data_root,
            PLACE_TRIPLET_ROOT_RELATIVE_PATH,
            role="checkpoint_selection",
            expected_manifest_sha256=manifest_sha,
        )
        if (
            h6_train != self.h6_train_rows
            or h6_selection != self.h6_selection_rows
            or place_train != self.place_train_rows
            or place_selection != self.place_selection_rows
            or h6_train_audit != self.audits["h6_train"]
            or h6_selection_audit != self.audits["h6_checkpoint_selection"]
            or place_train_audit != self.audits["place_train"]
            or place_selection_audit != self.audits["place_checkpoint_selection"]
        ):
            raise PermissionError("memory-role terminal index rehash changed")
        local_rehash = self._local_loader.terminal_rehash()
        return {
            "schema": f"{SCHEMA_PREFIX}_role_terminal_access_v1",
            "terminal_index_rehash_count": 4,
            "local_rgb": self._local_loader.access_receipt(),
            "local_terminal_rehash": local_rehash,
            "place_triplet_loader_call_count": self._place_loader_calls,
            "place_rgb_sha256_verified_per_access_count": 3 * self._place_loader_calls,
            "place_unique_row_count_opened": len(self._place_loaded_row_keys),
            "place_index_terminal_rehash_count": 2,
            "place_rgb_terminal_rehash_count": 0,
            "place_rgb_terminal_rehash_not_required_reason": (
                "every RGB read was checked against the manifest-bound per-leaf SHA-256"
            ),
            "probability_calibration_opened": False,
            "navigation_executed": False,
            "held_out_or_sealed_opened": False,
        }

    def failure_access_snapshot(self) -> dict[str, Any]:
        """Return in-memory access counters without new file opens or rehashes."""

        self._require_open()
        return {
            "schema": f"{SCHEMA_PREFIX}_role_failure_access_snapshot_v1",
            "local_rgb": self._local_loader.access_receipt(),
            "place_triplet_loader_call_count": self._place_loader_calls,
            "place_rgb_sha256_verified_per_access_count": (
                3 * self._place_loader_calls
            ),
            "place_unique_row_count_opened": len(self._place_loaded_row_keys),
            "new_file_open_count": 0,
            "terminal_rehash_count": 0,
            "probability_calibration_opened": False,
            "navigation_executed": False,
            "held_out_or_sealed_opened": False,
        }

    def close(self) -> None:
        if not self._closed:
            self._local_loader.close()
            self._closed = True


def load_memory_role_runtime_v1(
    runtime_data_root: Path,
    *,
    runtime_inputs: Mapping[str, Any],
    physical_train_scene_ids: Sequence[str],
    physical_selection_scene_ids: Sequence[str],
) -> MemoryRoleRuntimeV1:
    """Open only exact train/checkpoint-selection metadata before RGB access."""

    root = Path(runtime_data_root)
    if not root.is_absolute() or type(runtime_inputs) is not dict:
        raise PermissionError("memory-role role runtime inputs are invalid")
    h6_data = __import__(
        "lewm.datasets.go2_explicit_plan_discounted_successor_state_v27",
        fromlist=["*"],
    )
    place_data = __import__(
        "lewm.datasets.go2_memory_role_place_triplets_v1", fromlist=["*"]
    )
    evaluation = __import__(EVALUATION_MODULE_NAME, fromlist=["*"])
    training = __import__(TRAINING_MODULE_NAME, fromlist=["*"])
    h6_train_rows, h6_train_audit = h6_data.load_bound_index(root, role="train")
    h6_selection_rows, h6_selection_audit = h6_data.load_bound_index(
        root, role="val"
    )
    manifest_binding = _binding(
        runtime_inputs.get("place_triplet_manifest"), name="place triplet manifest"
    )
    place_train_rows, place_train_audit = place_data.load_index(
        root,
        PLACE_TRIPLET_ROOT_RELATIVE_PATH,
        role="train",
        expected_manifest_sha256=manifest_binding["file_sha256"],
    )
    place_selection_rows, place_selection_audit = place_data.load_index(
        root,
        PLACE_TRIPLET_ROOT_RELATIVE_PATH,
        role="checkpoint_selection",
        expected_manifest_sha256=manifest_binding["file_sha256"],
    )
    expected_audits = {
        "h6_train_index": h6_train_audit,
        "h6_checkpoint_selection_index": h6_selection_audit,
        "place_triplet_train_index": {
            "path": f"{PLACE_TRIPLET_ROOT_RELATIVE_PATH.as_posix()}/train.jsonl",
            "file_sha256": place_train_audit["index_file_sha256"],
        },
        "place_triplet_checkpoint_selection_index": {
            "path": (
                f"{PLACE_TRIPLET_ROOT_RELATIVE_PATH.as_posix()}/"
                "checkpoint_selection.jsonl"
            ),
            "file_sha256": place_selection_audit["index_file_sha256"],
        },
    }
    for name, audit in expected_audits.items():
        authority_binding = _binding(runtime_inputs.get(name), name=name)
        if authority_binding["path"] != audit["path"] or authority_binding[
            "file_sha256"
        ] != audit["file_sha256"]:
            raise PermissionError(f"memory-role {name} does not match loaded index")
        if "byte_count" in audit and authority_binding["byte_count"] != audit["byte_count"]:
            raise PermissionError(f"memory-role {name} byte count changed")
    if len(place_train_rows) != MAXIMUM_UPDATES * PLACE_TRAIN_ROWS_PER_UPDATE:
        raise PermissionError("memory-role place train index is not exact 3,200 rows")
    return MemoryRoleRuntimeV1(
        runtime_data_root=root,
        runtime_inputs=runtime_inputs,
        h6_data=h6_data,
        place_data=place_data,
        evaluation=evaluation,
        training=training,
        h6_train_rows=h6_train_rows,
        h6_selection_rows=h6_selection_rows,
        place_train_rows=place_train_rows,
        place_selection_rows=place_selection_rows,
        physical_train_scene_ids=physical_train_scene_ids,
        physical_selection_scene_ids=physical_selection_scene_ids,
        audits={
            "h6_train": h6_train_audit,
            "h6_checkpoint_selection": h6_selection_audit,
            "place_train": place_train_audit,
            "place_checkpoint_selection": place_selection_audit,
        },
    )


def run_future_authorized_engine_v1(
    *,
    authority: Mapping[str, Any],
    reservation: Mapping[str, Any],
    runtime: Any,
    publisher: Any,
) -> dict[str, Any]:
    """Execute observations 0/100/400 and exactly 400 mixed updates."""

    validated_authority = validate_future_execution_prerequisites_v1(dict(authority))
    validated_reservation = validate_attempt_reservation_v1(dict(reservation))
    if validated_reservation["authority_sha256"] != hashlib.sha256(
        _canonical_json_bytes(validated_authority)
    ).hexdigest():
        raise PermissionError("memory-role reservation does not bind authority")

    evaluation = __import__(EVALUATION_MODULE_NAME, fromlist=["*"])
    physical_train_scene_ids = tuple(
        sorted({str(pair["scene_id"]) for pair in runtime.pairs["train"]})
    )
    physical_selection_scene_ids = tuple(
        sorted(
            {
                str(pair["scene_id"])
                for pair in runtime.pairs["checkpoint_selection"]
            }
        )
    )
    role_runtime = load_memory_role_runtime_v1(
        runtime.runtime_data_root,
        runtime_inputs=validated_authority["runtime_inputs"],
        physical_train_scene_ids=physical_train_scene_ids,
        physical_selection_scene_ids=physical_selection_scene_ids,
    )
    trace: list[dict[str, Any]] = []
    metric_bindings: list[dict[str, Any]] = []
    observations: dict[int, dict[str, Any]] = {}
    accounting: Any = None
    model: Any = None
    optimizer: Any = None
    partial_checkpoint_binding: Mapping[str, Any] | None = None
    stage = "initialize"
    try:
        model, optimizer, initialization = runtime.initialize_model_v13()
        initial_structural = physical_executor._derive_initial_structural_integrity_v13(
            runtime, model
        )
        trace.append(
            {
                "schema": f"{SCHEMA_PREFIX}_trace_row_v1",
                "event": "initialized",
                "update": 0,
                "initialization": dict(initialization),
                "initial_structural_integrity": initial_structural,
                "role_preflight": role_runtime.preflight_receipt(),
            }
        )
        integrity_pass = initial_structural.get("passed") is True
        for update in range(MAXIMUM_UPDATES + 1):
            if update:
                stage = f"train_update_{update}"
                start = (update - 1) * PHYSICAL_PRESENTATIONS_PER_UPDATE
                physical_indices = tuple(
                    runtime.schedule[start : start + PHYSICAL_PRESENTATIONS_PER_UPDATE]
                )
                physical_batches = runtime.build_microbatches_v13(
                    physical_indices, update=update
                )
                local_batches = role_runtime.build_local_train_microbatches(
                    update, runtime.device
                )
                place_batches = role_runtime.build_place_train_microbatches(
                    update, runtime.device
                )
                result = runtime.training_module.joint_training_update_v1(
                    model,
                    optimizer,
                    physical_batches,
                    local_batches,
                    place_batches,
                    accounting=accounting,
                )
                accounting = result.accounting
                integrity = validate_update_integrity_v1(
                    runtime, model, result, update=update
                )
                integrity_pass = integrity_pass and integrity["passed"]
                trace.append(
                    {
                        "schema": f"{SCHEMA_PREFIX}_trace_row_v1",
                        "event": "optimizer_ema_update",
                        **integrity,
                    }
                )
            if update not in OBSERVATION_UPDATES:
                continue
            stage = f"observe_update_{update}"
            physical = physical_executor._observation_v13(
                runtime,
                model,
                update=update,
                integrity_pass=integrity_pass,
            )
            roles = role_runtime.evaluate_role_metrics(
                model, update=update, device=runtime.device
            )
            observation = {
                "schema": f"{SCHEMA_PREFIX}_observation_v1",
                "update": update,
                "physical": physical,
                "roles": roles,
                "integrity_pass": bool(
                    integrity_pass
                    and physical.get("integrity_pass") is True
                    and roles.get("integrity", {}).get("passed") is True
                ),
                "state_mutation_count": 0,
                "probability_calibration_opened": False,
                "navigation_executed": False,
                "held_out_or_sealed_opened": False,
            }
            observations[update] = observation
            _, binding = _publish_json(
                publisher, f"metrics/update_{update}.json", observation
            )
            metric_bindings.append(binding)

        if accounting is None or set(observations) != set(OBSERVATION_UPDATES):
            raise RuntimeError("memory-role controller did not complete exact schedule")
        stage = "classify_update400"
        gate = evaluation.evaluate_terminal_gate_v1(
            update0_place=observations[0]["roles"]["place"],
            update400_place=observations[400]["roles"]["place"],
            update400_local=observations[400]["roles"]["local"],
            physical_summary=observations[400]["physical"]["physical"],
            controls=observations[400]["physical"]["controls"],
            integrity_pass=all(
                observation["integrity_pass"] for observation in observations.values()
            ),
        )
        trace.append(
            {
                "schema": f"{SCHEMA_PREFIX}_trace_row_v1",
                "event": "update400_terminal_gate",
                "update": 400,
                "decision": gate,
            }
        )
        trace_raw = b"".join(
            _canonical_json_bytes(_content_bound(row)) + b"\n" for row in trace
        )
        trace_binding = publisher.publish_bytes("trace.jsonl", trace_raw)
        terminal_access = {
            "schema": f"{SCHEMA_PREFIX}_terminal_access_receipt_v1",
            "physical": runtime.terminal_access_receipt_v13(),
            "roles": role_runtime.terminal_access_receipt(),
            "probability_calibration_opened": False,
            "navigation_executed": False,
            "held_out_or_sealed_opened": False,
        }
        access_value, access_binding = _publish_json(
            publisher, "receipts/terminal_access.json", terminal_access
        )
        common = {
            "terminal_update": 400,
            "decision": gate,
            "accounting": _mapping(accounting, name="terminal accounting"),
            "metrics": metric_bindings,
            "trace": trace_binding,
            "terminal_access": access_binding,
            "terminal_access_content_sha256": access_value["content_sha256"],
            "attempt_consumed": True,
            "probability_calibration_opened": False,
            "navigation_executed": False,
            "held_out_or_sealed_opened": False,
        }
        if gate.get("passed") is not True:
            value, _ = _publish_json(
                publisher,
                "failure.json",
                {
                    "schema": f"{SCHEMA_PREFIX}_scientific_failure_v1",
                    "status": "FAIL_SCIENTIFIC_UPDATE400_GATE_TERMINAL",
                    **common,
                    "checkpoint_published": False,
                    "retry_authorized": False,
                    "resume_authorized": False,
                },
            )
            return value

        stage = "publish_pass_checkpoint"
        checkpoint_raw, checkpoint_core = _serialize_checkpoint_v1(
            runtime, model, optimizer, accounting, validated_authority
        )
        checkpoint_binding = publisher.publish_bytes(
            "checkpoint_update_400.pt", checkpoint_raw
        )
        partial_checkpoint_binding = checkpoint_binding
        checkpoint_value, checkpoint_metadata_binding = _publish_json(
            publisher,
            "checkpoint_update_400.binding.json",
            {**checkpoint_core, "checkpoint": checkpoint_binding},
        )
        value, _ = _publish_json(
            publisher,
            "success.json",
            {
                "schema": f"{SCHEMA_PREFIX}_success_v1",
                "status": "PASS_DEVELOPMENT_UPDATE400_TERMINAL",
                **common,
                "checkpoint_published": True,
                "checkpoint": checkpoint_binding,
                "checkpoint_metadata": checkpoint_metadata_binding,
                "checkpoint_metadata_content_sha256": checkpoint_value[
                    "content_sha256"
                ],
                "memory_integration_requires_separate_authority": True,
                "resume_authorized": False,
                "navigation_authorized": False,
                "held_out_authorized": False,
            },
        )
        return value
    except BaseException as error:
        def access_snapshot(name: str, callback: Any) -> dict[str, Any]:
            try:
                return {"status": "CAPTURED", "receipt": dict(callback())}
            except BaseException as snapshot_error:
                return {
                    "status": "UNAVAILABLE",
                    "source": name,
                    "exception_type": type(snapshot_error).__name__,
                    "exception_message_sha256": hashlib.sha256(
                        str(snapshot_error).encode("utf-8")
                    ).hexdigest(),
                }

        accounting_snapshot = (
            None if accounting is None else _mapping(accounting, name="failure accounting")
        )
        failure_context = {
            "schema": f"{SCHEMA_PREFIX}_failure_context_v1",
            "last_completed_update": (
                0 if accounting_snapshot is None else accounting_snapshot["updates"]
            ),
            "accounting": accounting_snapshot,
            "completed_observation_updates": sorted(observations),
            "published_metric_bindings": [dict(value) for value in metric_bindings],
            "trace_event_count": len(trace),
            "trace_rows_content_sha256": hashlib.sha256(
                _canonical_json_bytes(trace)
            ).hexdigest(),
            "physical_access": access_snapshot(
                "physical", runtime.access_receipt_v13
            ),
            "role_access": access_snapshot(
                "roles", role_runtime.failure_access_snapshot
            ),
            "probability_calibration_opened": False,
            "navigation_executed": False,
            "held_out_or_sealed_opened": False,
        }
        return terminalize_failure_v1(
            Path(publisher.output_root),
            validated_reservation,
            stage=stage,
            error=error,
            created_utc=datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            partial_checkpoint_binding=partial_checkpoint_binding,
            failure_context=failure_context,
        )
    finally:
        role_runtime.close()


run_future_authorized_engine_v13 = run_future_authorized_engine_v1


__all__ = [
    "AUTHORITY_RELATIVE_PATH",
    "CERTIFIED_SOURCE_ROOT",
    "CHECKPOINT_SCHEDULE_PREFIX_SHA256",
    "CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH",
    "CONTROL_NAMES",
    "EVALUATION_MODULE_NAME",
    "MATCHED_UPDATE400_THRESHOLDS",
    "MAXIMUM_PRESENTATIONS",
    "MAXIMUM_UPDATES",
    "MODEL_CLASS_NAME",
    "MODEL_MODULE_NAME",
    "OBSERVATION_UPDATES",
    "OUTPUT_ROOT_RELATIVE_PATH",
    "REGISTERED_FAMILIES",
    "RUNTIME_INPUT_BINDING_NAMES",
    "SCHEMA_PREFIX",
    "SCOPES",
    "TRAINING_MODULE_NAME",
    "V12_GATE_CHECK_NAMES",
    "reserve_attempt_v1",
    "load_memory_role_runtime_v1",
    "run_future_authorized_engine_v1",
    "terminalize_failure_v1",
    "validate_bound_sources_v1",
    "validate_content_bound_v1",
    "validate_future_execution_prerequisites_v1",
    "validate_update_integrity_v1",
]
