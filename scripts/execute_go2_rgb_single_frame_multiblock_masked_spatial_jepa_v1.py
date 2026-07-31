#!/usr/bin/env python3
"""One-shot controller for the single-frame masked spatial JEPA V1.

This module owns only authority, reservation, fixed update/observation policy,
immutable publication, and terminal receipts.  RGB decoding, tensor training,
and checkpoint evaluation remain in their reviewed runtime modules.
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


SCHEMA_PREFIX = (
    "lewm_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1"
)
PREREGISTRATION_COMMIT = "55d07f54408237723db641f95a22eb113a7965ad"
SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1_"
    "source_manifest_2026-07-31.json"
)
CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1_"
    "clean_export_certification_2026-07-31.json"
)
AUTHORITY_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1_"
    "execution_authorization_2026-07-31.json"
)
CERTIFIED_SOURCE_ROOT = (
    "/home/andrewknowles/Workspace/"
    "LeWMQuad-v3-rgb-single-frame-multiblock-masked-spatial-jepa-v1-source"
)
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1/"
    "attempt_v1"
)
RGB_ROOT_RELATIVE_PATH = ".generated/datagen_full/render_textured_v03"
MODEL_MODULE_NAME = (
    "lewm.models.rgb_single_frame_multiblock_masked_spatial_jepa_v1"
)
MODEL_CLASS_NAME = "SingleFrameMultiblockMaskedSpatialJepaV1"
TRAINING_MODULE_NAME = (
    "scripts.run_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1"
)
EVALUATION_MODULE_NAME = (
    "scripts.evaluate_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1"
)

MAXIMUM_UPDATES = 1_000
MAXIMUM_PRESENTATIONS = 16_000
PRESENTATIONS_PER_UPDATE = 16
OBSERVATION_UPDATES = (0, 250, 500, 750, 1_000)
CONTROL_NAMES = ("wrong_target", "wrong_context", "position_mean")
HEALTH_NAMES = (
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
    "n320_gate": {
        "path": (
            ".generated/go2_observable_camera_ray_fit_v4/"
            "n320_compute_scaled_v1/gate.json"
        ),
        "file_sha256": (
            "4943b4060e88296503c09fc714e55e40fd762527cfccb70a3a341f0df800efe6"
        ),
        "content_sha256": (
            "76ce5ab703560d171f7c84684b90eed18e8b4cdcc2d8ed3eff6d48496f4de67b"
        ),
        "byte_count": 7_960,
    },
    "n320_checkpoint": {
        "path": (
            ".generated/go2_observable_camera_ray_fit_v4/"
            "n320_compute_scaled_v1/checkpoint.pt"
        ),
        "file_sha256": (
            "ece874b53941e841fffc61b724a86d4383b881549afa453b746dd5d68aba11b0"
        ),
        "content_sha256": (
            "9dcca536943f89acfd7d463fdab591e19a030ef3dc8f3f19a050b1b10025fc2b"
        ),
        "byte_count": 13_777_100,
    },
}
RUNTIME_INPUT_BINDING_NAMES = tuple(RUNTIME_INPUT_BINDINGS)

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
    result = dict(_jsonable(core))
    result.pop("content_sha256", None)
    result["content_sha256"] = hashlib.sha256(
        _canonical_json_bytes(result)
    ).hexdigest()
    return result


def validate_content_bound_v1(value: Any) -> dict[str, Any]:
    if type(value) is not dict or type(value.get("content_sha256")) is not str:
        raise TypeError("V1 content-bound value must be an exact object")
    core = dict(value)
    observed = core.pop("content_sha256")
    if observed != hashlib.sha256(_canonical_json_bytes(core)).hexdigest():
        raise RuntimeError("V1 content binding changed")
    return dict(value)


def _binding(value: Any, *, content: bool = False) -> dict[str, Any]:
    expected = {"path", "file_sha256", "byte_count"}
    if content:
        expected.add("content_sha256")
    if type(value) is not dict or set(value) != expected:
        raise TypeError("runtime binding fields changed")
    result = dict(value)
    path = PurePosixPath(result["path"]) if type(result["path"]) is str else None
    hashes = ("file_sha256", "content_sha256") if content else ("file_sha256",)
    if (
        path is None
        or path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(
            type(result[name]) is not str
            or len(result[name]) != 64
            or any(character not in "0123456789abcdef" for character in result[name])
            for name in hashes
        )
        or type(result["byte_count"]) is not int
        or result["byte_count"] <= 0
    ):
        raise TypeError("runtime binding values changed")
    return result


def validate_future_execution_prerequisites_v1(
    authority: Any,
) -> dict[str, Any]:
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
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "certified_source_root": CERTIFIED_SOURCE_ROOT,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "rgb_root_relative_path": RGB_ROOT_RELATIVE_PATH,
        "output_root_absent_at_authorization": True,
        "device": "cuda:0",
    }
    if any(value.get(name) != expected for name, expected in required.items()):
        raise PermissionError("V1 authority identity, scope, or cap changed")
    if value.get("selectors") != {
        "executor_module": __name__,
        "model_module": MODEL_MODULE_NAME,
        "model_class": MODEL_CLASS_NAME,
        "training_module": TRAINING_MODULE_NAME,
        "evaluation_module": EVALUATION_MODULE_NAME,
    }:
        raise PermissionError("V1 runtime selectors changed")
    data_root = value.get("runtime_data_root")
    if data_root != "/home/andrewknowles/Workspace/LeWMQuad-v3":
        raise PermissionError("V1 runtime data root changed")
    certification = value.get("clean_export_certification")
    if (
        type(certification) is not dict
        or _binding(certification, content=True)["path"]
        != CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH
    ):
        raise PermissionError("V1 clean-export certification changed")
    inputs = value.get("runtime_inputs")
    if type(inputs) is not dict or tuple(inputs) != RUNTIME_INPUT_BINDING_NAMES:
        raise PermissionError("V1 runtime input order or inventory changed")
    for name, expected in RUNTIME_INPUT_BINDINGS.items():
        if _binding(inputs[name], content="content_sha256" in expected) != expected:
            raise PermissionError(f"V1 runtime binding changed: {name}")
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
                raise PermissionError("output directory containment changed")
        if created:
            os.chmod(current, 0o700, follow_symlinks=False)
    return current


def _write_immutable_bytes(path: Path, raw: bytes) -> dict[str, Any]:
    parent_info = os.lstat(path.parent)
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or stat.S_ISLNK(parent_info.st_mode)
    ):
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
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PermissionError("publication path escaped the attempt root")
    directory = _mkdir_beneath(output, PurePosixPath(*path.parts[:-1]))
    binding = _write_immutable_bytes(directory / path.name, raw)
    return {**binding, "path": path.as_posix()}


def _publish_json(
    output: Path, relative: str, core: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = _content_bound(core)
    raw = _canonical_json_bytes(value) + b"\n"
    return value, _publish_bytes(output, relative, raw)


def reserve_attempt_v1(
    repository_root: Path,
    authority: Mapping[str, Any],
    *,
    created_utc: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve(strict=True)
    validated = validate_future_execution_prerequisites_v1(dict(authority))
    output_relative = PurePosixPath(OUTPUT_ROOT_RELATIVE_PATH)
    output = root.joinpath(*output_relative.parts)
    if output.exists() or output.is_symlink():
        raise FileExistsError("V1 output root must be absent before reservation")
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
            "output_root": OUTPUT_ROOT_RELATIVE_PATH,
            "attempt": 1,
            "maximum_updates": MAXIMUM_UPDATES,
            "maximum_presentations": MAXIMUM_PRESENTATIONS,
        }
    )
    _publish_bytes(output, "reservation.json", _canonical_json_bytes(reservation) + b"\n")
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
    }
    if any(result.get(name) != expected for name, expected in required.items()):
        raise PermissionError("V1 reservation changed")
    return result


def _read_bound_file(root: Path, binding: Mapping[str, Any]) -> bytes:
    expected = _binding(
        dict(binding), content="content_sha256" in binding
    )
    relative = PurePosixPath(expected["path"])
    descriptor = os.open(root, _DIR_FLAGS)
    file_descriptor: int | None = None
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
        (before.st_dev, before.st_ino, before.st_size)
        != (after.st_dev, after.st_ino, after.st_size)
        or len(raw) != expected["byte_count"]
        or hashlib.sha256(raw).hexdigest() != expected["file_sha256"]
    ):
        raise PermissionError("bound runtime input bytes changed")
    return raw


def _strict_json(raw: bytes) -> Any:
    def unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise PermissionError("JSON repeats a key")
            result[key] = value
        return result

    return json.loads(
        raw,
        object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(
            PermissionError(f"nonfinite JSON constant {value}")
        ),
    )


def _tensor_manifest(torch: Any, state: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name, value in sorted(state.items()):
        if type(name) is not str or not isinstance(value, torch.Tensor):
            raise PermissionError("N320 state inventory changed")
        tensor = value.detach().to(device="cpu").contiguous()
        result.append(
            {
                "name": name,
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
                "sha256": hashlib.sha256(
                    tensor.view(torch.uint8).numpy().tobytes(order="C")
                ).hexdigest(),
            }
        )
    if not result:
        raise PermissionError("N320 state is empty")
    return result


def extract_n320_encoder_state_v1(
    torch: Any,
    checkpoint: Any,
    *,
    expected_content_sha256: str,
) -> dict[str, Any]:
    fields = {
        "schema",
        "model_class",
        "state_manifest",
        "metadata",
        "authoritative",
        "aggregation_eligible",
        "promotion_eligible",
        "state_dict",
        "content_sha256",
    }
    if (
        type(checkpoint) is not dict
        or set(checkpoint) != fields
        or checkpoint["schema"]
        != "lewm_go2_observable_camera_ray_fit_v4_development_checkpoint_v2"
        or checkpoint["model_class"] != "ObservableCameraRayEvidenceV4Model"
        or checkpoint["authoritative"] is not False
        or checkpoint["aggregation_eligible"] is not False
        or checkpoint["promotion_eligible"] is not False
        or type(checkpoint["state_dict"]) is not dict
    ):
        raise PermissionError("N320 checkpoint schema or scope changed")
    manifest = _tensor_manifest(torch, checkpoint["state_dict"])
    semantic = {
        name: checkpoint[name]
        for name in (
            "schema",
            "model_class",
            "state_manifest",
            "metadata",
            "authoritative",
            "aggregation_eligible",
            "promotion_eligible",
        )
    }
    if (
        checkpoint["state_manifest"] != manifest
        or checkpoint["content_sha256"]
        != hashlib.sha256(_canonical_json_bytes(semantic)).hexdigest()
        or checkpoint["content_sha256"] != expected_content_sha256
    ):
        raise PermissionError("N320 checkpoint tensor or semantic binding changed")
    encoder = {
        name.removeprefix("encoder."): value.detach()
        for name, value in checkpoint["state_dict"].items()
        if name.startswith("encoder.")
    }
    if not encoder or len(encoder) == len(checkpoint["state_dict"]):
        raise PermissionError("N320 encoder-only extraction changed")
    return encoder


def load_n320_encoder_state_v1(
    runtime_data_root: Path,
    authority: Mapping[str, Any],
    torch: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(runtime_data_root).resolve(strict=True)
    inputs = authority["runtime_inputs"]
    gate_raw = _read_bound_file(root, inputs["n320_gate"])
    gate = _strict_json(gate_raw)
    artifact = gate.get("artifacts", {}).get("checkpoint")
    if (
        type(gate) is not dict
        or gate.get("schema")
        != "lewm_go2_observable_camera_ray_fit_v4_n320_compute_scaled_v1_gate_v1"
        or gate.get("status") != "passed"
        or gate.get("passes") is not True
        or gate.get("check_count") != 26
        or gate.get("failure_count") != 0
        or gate.get("row", {}).get("seed") != 20_260_710
        or gate.get("row", {}).get("fit_size") != 320
        or gate.get("row", {}).get("updates") != 40_000
        or gate.get("numeric_gate", {}).get("passes") is not True
        or gate.get("numeric_gate", {}).get("failure_count") != 0
        or gate.get("retry_authorized") is not False
        or type(artifact) is not dict
        or artifact.get("file_sha256")
        != inputs["n320_checkpoint"]["file_sha256"]
        or artifact.get("byte_count") != inputs["n320_checkpoint"]["byte_count"]
        or artifact.get("content_sha256")
        != inputs["n320_checkpoint"]["content_sha256"]
    ):
        raise PermissionError("N320 gate did not pass its exact 26 checks")
    checkpoint_raw = _read_bound_file(root, inputs["n320_checkpoint"])
    checkpoint = torch.load(
        io.BytesIO(checkpoint_raw), map_location="cpu", weights_only=True
    )
    encoder = extract_n320_encoder_state_v1(
        torch,
        checkpoint,
        expected_content_sha256=inputs["n320_checkpoint"]["content_sha256"],
    )
    return encoder, {
        "gate_open_count": 1,
        "checkpoint_open_count": 1,
        "checkpoint_deserialize_count": 1,
        "encoder_only_initialization": True,
        "evidence_head_migrated": False,
        "passed": True,
    }


def _default_apis() -> Any:
    torch = importlib.import_module("torch")
    model_module = importlib.import_module(MODEL_MODULE_NAME)
    training = importlib.import_module(TRAINING_MODULE_NAME)
    evaluation = importlib.import_module(EVALUATION_MODULE_NAME)
    return SimpleNamespace(
        torch=torch,
        model_class=getattr(model_module, MODEL_CLASS_NAME),
        training=training,
        open_runtime=evaluation.open_bound_runtime,
        evaluate=evaluation.evaluate_checkpoint,
        load_n320=load_n320_encoder_state_v1,
    )


def _finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_tree(item) for item in value)
    return True


def _health_retention(
    observation: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for branch in ("online", "target"):
        result[branch] = {}
        for name in HEALTH_NAMES:
            denominator = float(baseline["raw_health"][branch][name])
            current = float(observation["raw_health"][branch][name])
            if not denominator > 0.0:
                raise RuntimeError("update-zero raw-health baseline is nonpositive")
            result[branch][name] = current / denominator
    return result


def _place_retention(
    observation: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, float]:
    current = observation["place"]
    initial = baseline["place"]
    multiple0 = float(initial["retrieval"]["chance_multiple"])
    rank0 = float(initial["target_place_key_effective_rank"])
    if not multiple0 > 0.0 or not rank0 > 0.0:
        raise RuntimeError("update-zero place baseline is nonpositive")
    return {
        "chance_multiple": (
            float(current["retrieval"]["chance_multiple"]) / multiple0
        ),
        "target_place_key_effective_rank": (
            float(current["target_place_key_effective_rank"]) / rank0
        ),
    }


def evaluate_observation_gate_v1(
    observation: Mapping[str, Any],
    baseline: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
) -> dict[str, Any]:
    update = int(observation["update"])
    retention = _health_retention(observation, baseline)
    place = _place_retention(observation, baseline)
    integrity = (
        observation.get("integrity", {}).get("passed") is True
        and _finite_tree(observation)
    )
    catastrophic = update >= 250 and any(
        retention[branch][name] < 0.25
        for branch in retention
        for name in HEALTH_NAMES
    )
    controls = observation["controls"]
    maximum_ratio = max(float(controls[name]["primary_ratio"]) for name in CONTROL_NAMES)
    minimum_families = min(
        int(controls[name]["positive_family_count"]) for name in CONTROL_NAMES
    )
    improves = False
    already_separating = all(
        float(controls[name]["primary_ratio"]) < 1.0
        and int(controls[name]["positive_family_count"]) >= 4
        for name in CONTROL_NAMES
    )
    if update in (500, 750):
        if previous is None:
            raise RuntimeError("continuation gate lacks its preceding observation")
        old = previous["controls"]
        improves = (
            max(float(old[name]["primary_ratio"]) for name in CONTROL_NAMES)
            - maximum_ratio
            >= 0.001
            or minimum_families
            > min(int(old[name]["positive_family_count"]) for name in CONTROL_NAMES)
        )
    qualified = bool(
        integrity
        and not catastrophic
        and all(
            float(controls[name]["primary_ratio"]) <= 0.90
            and float(controls[name]["advantage_bootstrap_lower_95"]) > 0.0
            and int(controls[name]["positive_family_count"]) >= 6
            for name in CONTROL_NAMES
        )
        and all(
            retention[branch][name] >= 0.50
            for branch in retention
            for name in HEALTH_NAMES
        )
        and place["chance_multiple"] >= 0.80
        and int(
            observation["place"]["retrieval"][
                "scene_count_at_least_1_5x_chance"
            ]
        )
        >= 6
        and float(observation["place"]["target_place_key_effective_rank"]) >= 2.0
        and place["target_place_key_effective_rank"] >= 0.80
    )
    if update in (0, 250):
        continue_training = integrity and not catastrophic
    elif update in (500, 750):
        continue_training = integrity and not catastrophic and (
            improves or already_separating
        )
    else:
        continue_training = False
    return {
        "schema": f"{SCHEMA_PREFIX}_observation_gate_v1",
        "update": update,
        "integrity_pass": integrity,
        "raw_health_retention": retention,
        "place_retention": place,
        "catastrophic_representation_collapse": catastrophic,
        "maximum_primary_ratio": maximum_ratio,
        "minimum_positive_family_count": minimum_families,
        "improves_from_preceding_observation": improves,
        "already_separating": already_separating,
        "continue_training": continue_training,
        "perception_qualified": qualified,
    }


def select_qualified_checkpoint_v1(
    observations: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    qualified = [
        value
        for value in observations
        if value["gate"]["perception_qualified"] is True and value["update"] > 0
    ]
    if not qualified:
        return None
    return min(
        qualified,
        key=lambda value: (
            value["gate"]["maximum_primary_ratio"],
            value["controls"]["wrong_target"]["correct_macro_mean"],
            value["update"],
        ),
    )


def _accounting_dict(value: Any) -> dict[str, Any]:
    return dict(asdict(value) if is_dataclass(value) else value)


def _validate_update_result(
    result: Any, *, update: int, model: Any, torch: Any
) -> dict[str, Any]:
    expected = {
        "updates": update,
        "presentations": 16 * update,
        "mask_rows": 16 * update,
        "online_frame_encodings": 16 * update,
        "ema_target_frame_encodings": 16 * update,
        "microbatch_graphs": 4 * update,
        "backward_calls": 4 * update,
        "global_gradient_clips": update,
        "optimizer_steps": update,
        "ema_steps": update,
    }
    accounting = _accounting_dict(result.accounting)
    receipt = dict(result.gradient_receipt)
    if (
        accounting != expected
        or result.target_gradient_tensor_count != 0
        or result.optimizer_steps_this_update != 1
        or result.ema_steps_this_update != 1
        or int(model.ema_update_count.detach().cpu()) != update
        or receipt.get("sole_jepa_route") is not True
        or receipt.get("all_gradient_receipts_finite") is not True
        or not math.isfinite(float(result.mean_jepa_loss))
        or any(
            parameter.grad is not None
            for parameter in model.target_encoder.parameters()
        )
        or any(
            value.is_floating_point() and not bool(torch.isfinite(value).all())
            for value in model.state_dict().values()
        )
    ):
        raise RuntimeError("V1 update, target, gradient, or accounting integrity failed")
    return {
        "update": update,
        "accounting": accounting,
        "mean_jepa_loss": float(result.mean_jepa_loss),
        "gradient_receipt": receipt,
        "row_indices_sha256": result.row_indices_sha256,
        "target_indices_sha256": result.target_indices_sha256,
        "visible_indices_sha256": result.visible_indices_sha256,
        "passed": True,
    }


def _initial_integrity(model: Any, optimizer: Any, training: Any, torch: Any) -> dict[str, Any]:
    inventory = training.parameter_inventory_v1(model)
    online = tuple(model.encoder.named_parameters())
    target = dict(model.target_encoder.named_parameters())
    checks = {
        "hard_synced_target": bool(
            online
            and set(name for name, _ in online) == set(target)
            and all(torch.equal(value, target[name]) for name, value in online)
        ),
        "target_frozen_eval": (
            not model.target_encoder.training
            and all(not value.requires_grad for value in target.values())
        ),
        "target_zero_grad": all(value.grad is None for value in target.values()),
        "target_optimizer_excluded": inventory["target_optimizer_excluded"] is True,
        "ema_update_count_zero": int(model.ema_update_count.detach().cpu()) == 0,
        "finite_nonzero_model": all(
            not value.is_floating_point() or bool(torch.isfinite(value).all())
            for value in model.state_dict().values()
        )
        and any(
            value.is_floating_point() and bool(value.detach().abs().sum() > 0)
            for value in model.state_dict().values()
        ),
    }
    return {
        "schema": f"{SCHEMA_PREFIX}_initial_integrity_v1",
        "checks": checks,
        "parameter_inventory": inventory,
        "passed": all(checks.values()),
    }


def _serialize_checkpoint(
    torch: Any,
    payload: Mapping[str, Any],
    *,
    update: int,
    authority_sha256: str,
) -> bytes:
    complete = {
        **dict(payload),
        "update": update,
        "authority_sha256": authority_sha256,
        "rng": {
            "torch_cpu": torch.random.get_rng_state().clone(),
            "visible_gpu": tuple(
                value.clone() for value in torch.cuda.get_rng_state_all()
            ),
        },
        "complete_continuation_state": True,
        "same_attempt_reopen_count": 0,
        "retry_authorized": False,
        "resume_authorized": False,
    }
    buffer = io.BytesIO()
    torch.save(complete, buffer)
    return buffer.getvalue()


def _runtime_input_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    train = audit["train"]
    validation = audit["validation"]
    place = audit["place"]
    expected = RUNTIME_INPUT_BINDINGS
    checks = {
        "train_index": all(
            train.get(key) == expected["h6_train_index"][key]
            for key in ("path", "file_sha256", "byte_count")
        ),
        "validation_index": all(
            validation.get(key) == expected["h6_validation_index"][key]
            for key in ("path", "file_sha256", "byte_count")
        ),
        "place_manifest": (
            place.get("manifest_file_sha256")
            == expected["place_triplet_manifest"]["file_sha256"]
        ),
        "place_index": (
            place.get("index_file_sha256")
            == expected["place_triplet_checkpoint_selection_index"]["file_sha256"]
        ),
        "no_future_rgb": audit.get("future_rgb_tensor_count") == 0,
        "no_actions": audit.get("action_tensor_count") == 0,
    }
    if not all(checks.values()):
        raise PermissionError("V1 opened runtime inputs differ from authority")
    return {"checks": checks, "audit": dict(audit), "passed": True}


def terminalize_failure_v1(
    output: Path,
    reservation: Mapping[str, Any],
    *,
    stage: str,
    error: BaseException,
    accounting: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    metric_bindings: Sequence[Mapping[str, Any]],
    access: Mapping[str, Any],
) -> dict[str, Any]:
    validate_attempt_reservation_v1(dict(reservation))
    value, _binding_value = _publish_json(
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
            "accounting": dict(accounting),
            "checkpoints": list(checkpoints),
            "metrics": list(metric_bindings),
            "access": dict(access),
            "checkpoint_deserialize_count_after_initialization": 0,
            "attempt_consumed": True,
            "retry_authorized": False,
            "resume_authorized": False,
            "held_out_or_sealed_opened": False,
            "navigation_executed": False,
        },
    )
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def run_authorized_engine_v1(
    *,
    authority: Mapping[str, Any],
    reservation: Mapping[str, Any],
    repository_root: Path,
    runtime_data_root: Path,
    device: Any,
    apis: Any | None = None,
) -> dict[str, Any]:
    """Execute the exact one-shot schedule, including update-zero evaluation."""

    validated = validate_future_execution_prerequisites_v1(dict(authority))
    reserved = validate_attempt_reservation_v1(dict(reservation))
    authority_sha256 = hashlib.sha256(_canonical_json_bytes(validated)).hexdigest()
    if reserved["authority_sha256"] != authority_sha256:
        raise PermissionError("V1 reservation does not bind the supplied authority")
    root = Path(repository_root).resolve(strict=True)
    output = root / OUTPUT_ROOT_RELATIVE_PATH
    if not output.is_dir() or output.is_symlink():
        raise PermissionError("V1 reserved output root is absent or changed type")

    api = _default_apis() if apis is None else apis
    runtime = None
    model = optimizer = accounting = None
    checkpoints: list[dict[str, Any]] = []
    metric_bindings: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    input_audit: dict[str, Any] = {}
    n320_receipt: dict[str, Any] = {}
    stage = "load_n320"
    try:
        state, n320_receipt = api.load_n320(
            Path(runtime_data_root), validated, api.torch
        )
        stage = "initialize_model"
        model = api.model_class(state).to(device)
        model.train()
        optimizer = api.training.build_optimizer_v1(model)
        initial = _initial_integrity(model, optimizer, api.training, api.torch)
        if not initial["passed"]:
            raise RuntimeError("V1 initial model integrity failed")

        stage = "open_current_frame_runtime"
        runtime, audit = api.open_runtime(
            Path(runtime_data_root), device=device, include_place=True
        )
        input_audit = _runtime_input_audit(audit)
        baseline: dict[str, Any] | None = None
        previous: dict[str, Any] | None = None
        stop_reason: str | None = None

        for update in range(MAXIMUM_UPDATES + 1):
            if update:
                stage = f"train_update_{update}"
                batches = runtime.train_rows_for_update(update)
                if len(batches) != 4 or any(
                    len(batch.row_indices) != 4 for batch in batches
                ):
                    raise RuntimeError("V1 runtime did not provide four B4 batches")
                result = api.training.training_update_v1(
                    model,
                    optimizer,
                    tuple(batch.rgb for batch in batches),
                    tuple(batch.row_indices for batch in batches),
                    accounting=accounting,
                )
                accounting = result.accounting
                integrity = _validate_update_result(
                    result, update=update, model=model, torch=api.torch
                )
                trace.append(integrity)
            if update not in OBSERVATION_UPDATES:
                continue

            stage = f"observe_update_{update}"
            raw = dict(api.evaluate(model, runtime, update, device))
            if raw.get("update") != update or not _finite_tree(raw):
                raise RuntimeError("V1 observation identity or finiteness changed")
            if update == 0:
                raw["initial_integrity"] = initial
                baseline = raw
            if baseline is None:
                raise RuntimeError("V1 update-zero baseline was not recorded")
            gate = evaluate_observation_gate_v1(raw, baseline, previous)
            observation = {
                **raw,
                "accounting": (
                    {
                        "updates": 0,
                        "presentations": 0,
                        "mask_rows": 0,
                        "online_frame_encodings": 0,
                        "ema_target_frame_encodings": 0,
                        "microbatch_graphs": 0,
                        "backward_calls": 0,
                        "global_gradient_clips": 0,
                        "optimizer_steps": 0,
                        "ema_steps": 0,
                    }
                    if accounting is None
                    else _accounting_dict(accounting)
                ),
                "gate": gate,
                "state_mutation_count": 0,
                "checkpoint_deserialize_count": 0,
                "held_out_or_sealed_opened": False,
                "navigation_executed": False,
            }
            if update:
                stage = f"publish_checkpoint_update_{update}"
                payload = api.training.checkpoint_payload_v1(
                    model, optimizer, accounting
                )
                raw_checkpoint = _serialize_checkpoint(
                    api.torch,
                    payload,
                    update=update,
                    authority_sha256=authority_sha256,
                )
                binding = _publish_bytes(
                    output, f"snapshots/update_{update}.pt", raw_checkpoint
                )
                checkpoint = {
                    **binding,
                    "update": update,
                    "complete_continuation_state": True,
                    "same_attempt_reopen_count": 0,
                    "retry_authorized": False,
                    "resume_authorized": False,
                }
                checkpoints.append(checkpoint)
                metadata, metadata_binding = _publish_json(
                    output,
                    f"snapshots/update_{update}.binding.json",
                    {
                        "schema": f"{SCHEMA_PREFIX}_checkpoint_binding_v1",
                        **checkpoint,
                    },
                )
                observation["checkpoint"] = checkpoint
                observation["checkpoint_metadata"] = metadata_binding
                observation["checkpoint_metadata_content_sha256"] = metadata[
                    "content_sha256"
                ]
            stage = f"publish_metrics_update_{update}"
            published, binding = _publish_json(
                output, f"metrics/update_{update}.json", observation
            )
            metric_bindings.append(binding)
            observations.append(published)
            previous = published
            if update < MAXIMUM_UPDATES and not gate["continue_training"]:
                stop_reason = (
                    "CATASTROPHIC_REPRESENTATION_COLLAPSE"
                    if gate["catastrophic_representation_collapse"]
                    else "CONTINUATION_GATE_NOT_MET"
                )
                break

        if not observations or observations[0]["update"] != 0:
            raise RuntimeError("V1 update-zero observation did not complete")
        terminal_update = int(observations[-1]["update"])
        if accounting is None:
            terminal_accounting = observations[-1]["accounting"]
        else:
            terminal_accounting = _accounting_dict(accounting)
        if terminal_accounting["updates"] != terminal_update:
            raise RuntimeError("V1 terminal accounting disagrees with observation")
        selected = select_qualified_checkpoint_v1(observations)
        stage = "publish_trace"
        trace_raw = b"".join(
            _canonical_json_bytes(_content_bound(row)) + b"\n" for row in trace
        )
        trace_binding = _publish_bytes(output, "trace.jsonl", trace_raw)
        access_core = {
            "schema": f"{SCHEMA_PREFIX}_terminal_access_receipt_v1",
            "runtime_inputs": input_audit,
            "n320": n320_receipt,
            "observations": [
                {"update": value["update"], "access": value.get("access", {})}
                for value in observations
            ],
            "h6_training_current_rgb_presentations": terminal_update * 16,
            "place_rgb_presentations": len(observations) * 320 * 3,
            "future_rgb_tensor_count": 0,
            "action_tensor_count": 0,
            "checkpoint_deserialize_count_after_initialization": 0,
            "held_out_or_sealed_opened": False,
            "navigation_executed": False,
        }
        _access, access_binding = _publish_json(
            output, "receipts/terminal_access.json", access_core
        )
        common = {
            "terminal_update": terminal_update,
            "accounting": terminal_accounting,
            "metrics": metric_bindings,
            "checkpoints": checkpoints,
            "trace": trace_binding,
            "terminal_access": access_binding,
            "attempt_consumed": True,
            "retry_authorized": False,
            "resume_authorized": False,
            "held_out_or_sealed_opened": False,
            "navigation_executed": False,
        }
        if selected is not None:
            value, _ = _publish_json(
                output,
                "success.json",
                {
                    "schema": f"{SCHEMA_PREFIX}_success_v1",
                    "status": "PASS_PERCEPTION_QUALIFIED",
                    **common,
                    "selected_update": selected["update"],
                    "selected_checkpoint": selected["checkpoint"],
                    "learned_memory_experiment_may_be_preregistered": True,
                    "g2_authorized": False,
                    "navigation_authorized": False,
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
                    if stop_reason is None
                    else f"FAIL_SCIENTIFIC_{stop_reason}"
                ),
                **common,
                "checkpoint_selected": False,
            },
        )
        return value
    except BaseException as error:
        access = {
            "runtime_inputs": input_audit,
            "n320": n320_receipt,
            "held_out_or_sealed_opened": False,
            "navigation_executed": False,
        }
        return terminalize_failure_v1(
            output,
            reserved,
            stage=stage,
            error=error,
            accounting=(
                {} if accounting is None else _accounting_dict(accounting)
            ),
            checkpoints=checkpoints,
            metric_bindings=metric_bindings,
            access=access,
        )
    finally:
        if runtime is not None:
            loader = getattr(
                runtime,
                "loader",
                getattr(runtime, "_loader", None),
            )
            close = getattr(loader, "close", None)
            if callable(close):
                close()


def execute_authorized_v1(
    repository_root: Path,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Reserve the absent attempt root, then execute the bound one-shot probe."""

    validated = validate_future_execution_prerequisites_v1(dict(authority))
    source_root = Path(repository_root).resolve(strict=True)
    if str(source_root) != validated["certified_source_root"]:
        raise PermissionError("V1 executor was not called from its certified source root")
    data_root = Path(validated["runtime_data_root"]).resolve(strict=True)
    reservation = reserve_attempt_v1(
        data_root,
        validated,
        created_utc=_utc_now(),
    )
    return run_authorized_engine_v1(
        authority=validated,
        reservation=reservation,
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
        raise PermissionError("V1 authority path changed type")
    authority = _strict_json(arguments.authority.read_bytes())
    result = execute_authorized_v1(arguments.repository_root, authority)
    print(_canonical_json_bytes(result).decode("ascii"))
    return 0 if result.get("status") == "PASS_PERCEPTION_QUALIFIED" else 3


__all__ = [
    "AUTHORITY_RELATIVE_PATH",
    "CERTIFIED_SOURCE_ROOT",
    "CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH",
    "CONTROL_NAMES",
    "EVALUATION_MODULE_NAME",
    "MAXIMUM_PRESENTATIONS",
    "MAXIMUM_UPDATES",
    "MODEL_CLASS_NAME",
    "MODEL_MODULE_NAME",
    "OBSERVATION_UPDATES",
    "OUTPUT_ROOT_RELATIVE_PATH",
    "PREREGISTRATION_COMMIT",
    "RUNTIME_INPUT_BINDINGS",
    "RUNTIME_INPUT_BINDING_NAMES",
    "SCHEMA_PREFIX",
    "TRAINING_MODULE_NAME",
    "evaluate_observation_gate_v1",
    "execute_authorized_v1",
    "extract_n320_encoder_state_v1",
    "load_n320_encoder_state_v1",
    "reserve_attempt_v1",
    "run_authorized_engine_v1",
    "select_qualified_checkpoint_v1",
    "terminalize_failure_v1",
    "validate_attempt_reservation_v1",
    "validate_content_bound_v1",
    "validate_future_execution_prerequisites_v1",
]


if __name__ == "__main__":
    raise SystemExit(main())
