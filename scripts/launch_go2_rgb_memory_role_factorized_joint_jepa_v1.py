#!/usr/bin/env python3
"""Denied-by-default launcher for memory-role factorized joint-JEPA V1.

Import and the no-argument path open no scientific payload.  A run requires
the exact future authority, exact clean-source certification, an unused output
root, and the reviewed GPU selector before any runtime data is composed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE_LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_rgb_per_row_persistence_contrastive_temporal_joint_"
    "jepa_v25.py"
)
BASE_LAUNCHER_COMMIT = "43231c689547b66de83f3cafbfac270455a7a234"
BASE_LAUNCHER_FILE_SHA256 = (
    "dac097313656e7dd77b93fcf9433ece72c180b1edfbe6b9339ac4b4b8a1ceb0a"
)
BASE_LAUNCHER_BYTE_COUNT = 14_818
PRIVATE_BASE_MODULE_NAME = "_lewm_memory_role_v1_private_v25_launcher"

SCHEMA_PREFIX = (
    "lewm_go2_rgb_memory_role_factorized_joint_jepa_v2"
)
LAUNCHER_SCHEMA = f"{SCHEMA_PREFIX}_launcher_v1"
AUTHORITY_SCHEMA = f"{SCHEMA_PREFIX}_future_execution_authority_v1"
CERTIFICATION_SCHEMA = f"{SCHEMA_PREFIX}_clean_export_certification_v1"
EXPERIMENT_ARM_NAME = (
    "memory_role_factorized_joint_jepa_v2"
)
PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v2_"
    "preregistration_2026-07-30.md"
)
PREREGISTRATION_COMMIT = "429cb57bd89348502cd5b695a25ae864d33fdfa7"
RETRIEVAL_METADATA_PREFLIGHT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v2_"
    "retrieval_metadata_preflight_2026-07-30.json"
)
ORIGINAL_PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v1_"
    "preregistration_2026-07-30.md"
)
INTEGRITY_REPLACEMENT_PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v1_"
    "integrity_replacement_v1_preregistration_2026-07-30.md"
)
TERMINAL_FAILURE_RESULT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v1_"
    "terminal_infrastructure_failure_result_2026-07-30.json"
)
INTEGRITY_REPLACEMENT_TERMINAL_FAILURE_RESULT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v1_"
    "integrity_replacement_v1_terminal_infrastructure_failure_result_"
    "2026-07-30.json"
)
SPLIT_INTEGRITY_AMENDMENT_PATH = (
    "docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v1_"
    "split_integrity_amendment_2026-07-30.md"
)
SPLIT_INTEGRITY_AMENDMENT_COMMIT = (
    "5a1535567bf00b8e47d67d8966ef42a52726bd5b"
)
CERTIFIED_SOURCE_ROOT = (
    "/home/andrewknowles/Workspace/"
    "LeWMQuad-v3-memory-role-factorized-joint-jepa-v2-source"
)
AUTHORITY_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v2_"
    "execution_authorization_2026-07-30.json"
)
CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v2_"
    "clean_export_certification_2026-07-30.json"
)
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_memory_role_factorized_joint_jepa_v2/attempt_v1"
)
EXECUTOR_MODULE_NAME = (
    "scripts.execute_go2_rgb_memory_role_factorized_joint_jepa_v1"
)
MODEL_MODULE_NAME = "lewm.models.memory_role_factorized_joint_jepa_v1"
MODEL_CLASS_NAME = "MemoryRoleFactorizedJointJepaV1"
TRAINING_MODULE_NAME = (
    "scripts.run_go2_rgb_memory_role_factorized_joint_jepa_v1"
)
EVALUATION_MODULE_NAME = (
    "scripts.evaluate_go2_rgb_memory_role_factorized_joint_jepa_v1"
)
MAXIMUM_UPDATES = 400
MAXIMUM_PRESENTATIONS = 12_800
RUNTIME_DATA_ROOT = "/home/andrewknowles/Workspace/LeWMQuad-v3"
RGB_ROOT_RELATIVE_PATH = ".generated/datagen_full/render_textured_v03"
PLACE_TRIPLET_ROOT_RELATIVE_PATH = (
    ".generated/go2_memory_role_place_triplet_index_v1"
)
REQUIRED_HIP_VISIBLE_DEVICES = "0"
CONFLICTING_GPU_VISIBILITY_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "HSA_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
    "NVIDIA_VISIBLE_DEVICES",
    "ONEAPI_DEVICE_SELECTOR",
    "ZE_AFFINITY_MASK",
)

PHYSICAL_RUNTIME_INPUT_NAMES = (
    "raw_manifest",
    "raw_audit",
    "n320_gate",
    "n320_checkpoint",
    "schedule",
    "swept_label_manifest",
    "train_labels",
    "checkpoint_selection_labels",
)
H6_TRAIN_BINDING = {
    "path": (
        ".generated/go2_recurrent_h4_rgb_sequence_index_v2_schedule_integrity/"
        "train.jsonl"
    ),
    "file_sha256": (
        "aee2a54cddd849162648f9b8cfd54a0a28a25bd0705b6482e6af7435c85f4d77"
    ),
    "byte_count": 10_328_000,
}
H6_CHECKPOINT_SELECTION_BINDING = {
    "path": (
        ".generated/go2_recurrent_h4_rgb_sequence_index_v2_schedule_integrity/"
        "val.jsonl"
    ),
    "file_sha256": (
        "83592e2fea5927802881f076a58a9710100bea017d658c1b978ba651369beac6"
    ),
    "byte_count": 1_317_888,
}
PLACE_TRIPLET_MANIFEST_BINDING = {
    "path": f"{PLACE_TRIPLET_ROOT_RELATIVE_PATH}/manifest.json",
    "file_sha256": (
        "a5997d93838419cabaaf8e262db70ed51f6f928195f1a312cadc4768f74ca6ca"
    ),
    "byte_count": 42_308,
}
PLACE_TRIPLET_TRAIN_BINDING = {
    "path": f"{PLACE_TRIPLET_ROOT_RELATIVE_PATH}/train.jsonl",
    "file_sha256": (
        "72044c597286631be6133b45663ef975e222cd10d3f0cee1d0a9c038f0d422b6"
    ),
    "byte_count": 4_687_348,
}
PLACE_TRIPLET_CHECKPOINT_SELECTION_BINDING = {
    "path": f"{PLACE_TRIPLET_ROOT_RELATIVE_PATH}/checkpoint_selection.jsonl",
    "file_sha256": (
        "a628a1047b6f15223a4fd7d30c5c87fa1914efef0955d70d9bd2f5330c77dcb0"
    ),
    "byte_count": 473_508,
}

ROLE_RUNTIME_BINDINGS = {
    "h6_train_index": H6_TRAIN_BINDING,
    "h6_checkpoint_selection_index": H6_CHECKPOINT_SELECTION_BINDING,
    "place_triplet_manifest": PLACE_TRIPLET_MANIFEST_BINDING,
    "place_triplet_train_index": PLACE_TRIPLET_TRAIN_BINDING,
    "place_triplet_checkpoint_selection_index": (
        PLACE_TRIPLET_CHECKPOINT_SELECTION_BINDING
    ),
}
PHYSICAL_RUNTIME_INPUTS_SHA256 = (
    "7be0ab1978d0c7ba1fa2540362eb4220cc688389cd26d1c0dd6b84d85299b496"
)
HARDWARE_SHA256 = (
    "174f62b4fe4024611c8ea5b5ab9860defd9b12b8d4d4485d8ae78d0189aa6538"
)
RUNTIME_SHA256 = (
    "41f30aeb58abd1b0e5a0006f6187138e7b8ce8bcc6591ed476fa14ebba8d8741"
)
AUTHORIZED_ROLES_SHA256 = (
    "854d30b43eceb1f3308316fca7b2a068dfa08bab4b036f8ed3842c244b4daf9d"
)

REQUIRED_CERTIFIED_SOURCE_PATHS = frozenset(
    {
        "scripts/launch_go2_rgb_memory_role_factorized_joint_jepa_v1.py",
        PREREGISTRATION_RELATIVE_PATH,
        RETRIEVAL_METADATA_PREFLIGHT_RELATIVE_PATH,
        ORIGINAL_PREREGISTRATION_RELATIVE_PATH,
        INTEGRITY_REPLACEMENT_PREREGISTRATION_RELATIVE_PATH,
        TERMINAL_FAILURE_RESULT_RELATIVE_PATH,
        INTEGRITY_REPLACEMENT_TERMINAL_FAILURE_RESULT_RELATIVE_PATH,
        SPLIT_INTEGRITY_AMENDMENT_PATH,
        "scripts/execute_go2_rgb_memory_role_factorized_joint_jepa_v1.py",
        "scripts/run_go2_rgb_memory_role_factorized_joint_jepa_v1.py",
        "scripts/evaluate_go2_rgb_memory_role_factorized_joint_jepa_v1.py",
        "lewm/models/memory_role_factorized_joint_jepa_v1.py",
        "lewm/datasets/go2_memory_role_place_triplets_v1.py",
        "lewm/datasets/go2_explicit_plan_discounted_successor_state_v27.py",
        BASE_LAUNCHER_RELATIVE_PATH,
        "scripts/launch_go2_rgb_predictor_core_protected_survival_output_"
        "joint_jepa_v24.py",
        "scripts/launch_go2_rgb_action_prior_residualized_wrong_scene_survival_"
        "output_joint_jepa_v23.py",
        "scripts/launch_go2_rgb_same_action_cross_scene_contrastive_innovation_"
        "joint_jepa_v21.py",
        "scripts/launch_go2_rgb_object_space_height_volume_executed_successor_"
        "semantic_grounding_joint_jepa_v19.py",
        "scripts/launch_go2_rgb_object_space_height_volume_joint_jepa_v18.py",
        "scripts/launch_go2_rgb_swept_progress_survival_joint_jepa_v13_"
        "camera_evidence_bottleneck.py",
    }
)

AUTHORITY_KEYS = frozenset(
    {
        "schema",
        "status",
        "scientific_payload_authorized",
        "one_shot",
        "maximum_updates",
        "maximum_presentations",
        "retry_authorized",
        "resume_authorized",
        "certified_source_root",
        "output_root",
        "preregistration_commit",
        "split_integrity_amendment_commit",
        "pinned_source_and_review_commit",
        "implementation_commit",
        "selectors",
        "runtime_data_root",
        "runtime_inputs",
        "rgb_root_relative_path",
        "hardware",
        "runtime",
        "authorized_roles",
        "clean_export_certification",
        "content_sha256",
    }
)
CERTIFICATION_KEYS = frozenset(
    {
        "schema",
        "status",
        "passed",
        "certified_source_root",
        "pinned_source_and_review_commit",
        "bindings_sha256",
        "bindings",
        "content_sha256",
    }
)


def _canonical_json_bytes_v1(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _strict_json_object_v1(raw: bytes, *, name: str) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict JSON") from error
    if type(value) is not dict:
        raise ValueError(f"{name} must be one JSON object")
    return value


def _is_lower_sha256_v1(value: Any) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_commit_v1(value: Any) -> bool:
    return bool(
        type(value) is str
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_canonical_v1(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes_v1(value)).hexdigest()


def _validate_content_bound_v1(
    value: Any,
    *,
    name: str,
    exact_keys: frozenset[str] | None = None,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact object")
    if exact_keys is not None and set(value) != exact_keys:
        raise PermissionError(f"{name} keys changed")
    observed = value.get("content_sha256")
    core = dict(value)
    core.pop("content_sha256", None)
    if not _is_lower_sha256_v1(observed) or _sha256_canonical_v1(core) != observed:
        raise PermissionError(f"{name} content binding changed")
    return dict(value)


def validate_pre_reservation_gpu_visibility_v1(
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    observed = os.environ if environment is None else environment
    if observed.get("HIP_VISIBLE_DEVICES") != REQUIRED_HIP_VISIBLE_DEVICES:
        raise PermissionError("memory-role V1 requires HIP_VISIBLE_DEVICES=0")
    conflicting = tuple(
        name for name in CONFLICTING_GPU_VISIBILITY_ENVIRONMENT_KEYS if name in observed
    )
    if conflicting:
        raise PermissionError("memory-role V1 GPU visibility has a conflicting selector")
    return {
        "schema": f"{SCHEMA_PREFIX}_pre_reservation_gpu_visibility_v1",
        "hip_visible_devices": REQUIRED_HIP_VISIBLE_DEVICES,
        "conflicting_selectors_present": [],
        "hardware_queried": False,
        "passed": True,
    }


def _load_private_v25_launcher() -> Any:
    path = ROOT / BASE_LAUNCHER_RELATIVE_PATH
    root = ROOT.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if (
        resolved != path.absolute()
        or not resolved.is_relative_to(root)
        or path.is_symlink()
        or not path.is_file()
    ):
        raise PermissionError("frozen V25 launcher escaped or is absent")
    source = path.read_bytes()
    if (
        len(source) != BASE_LAUNCHER_BYTE_COUNT
        or hashlib.sha256(source).hexdigest() != BASE_LAUNCHER_FILE_SHA256
    ):
        raise PermissionError("frozen V25 launcher binding changed")
    if PRIVATE_BASE_MODULE_NAME in sys.modules:
        raise RuntimeError("private V25 launcher module name is occupied")
    spec = importlib.util.spec_from_file_location(PRIVATE_BASE_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen V25 launcher")
    module = importlib.util.module_from_spec(spec)
    sys.modules[PRIVATE_BASE_MODULE_NAME] = module
    inherited_names = (
        "_lewm_v25_per_row_temporal_private_v24_launcher",
        "_lewm_v24_core_protected_private_v23_launcher",
        "_lewm_v23_scene_action_private_v21_launcher",
        "_lewm_v21_scene_innovation_private_v20_launcher",
    )
    previous = {name: sys.modules.pop(name, None) for name in inherited_names}
    try:
        spec.loader.exec_module(module)
        if Path(module.ROOT).resolve(strict=True) != root:
            raise PermissionError("frozen V25 launcher resolved another root")
        module._assert_configured_base_v25()
        return module
    except BaseException:
        sys.modules.pop(PRIVATE_BASE_MODULE_NAME, None)
        for name, value in previous.items():
            if value is not None:
                sys.modules[name] = value
        raise


_V25_LAUNCHER = _load_private_v25_launcher()
_BASE_LAUNCHER = _V25_LAUNCHER._BASE_LAUNCHER
_BASE_OVERRIDES = {
    "AUTHORITY_RELATIVE_PATH": AUTHORITY_RELATIVE_PATH,
    "CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH": (
        CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH
    ),
    "EXECUTOR_MODULE_NAME": EXECUTOR_MODULE_NAME,
    "MODEL_MODULE_NAME": MODEL_MODULE_NAME,
    "TRAINING_MODULE_NAME": TRAINING_MODULE_NAME,
    "SOURCE_EVIDENCE_SCHEMA_PREFIX": SCHEMA_PREFIX,
    "EXPERIMENT_ARM_NAME": EXPERIMENT_ARM_NAME,
    "LAUNCHER_SCHEMA": LAUNCHER_SCHEMA,
}
for _name, _value in _BASE_OVERRIDES.items():
    setattr(_BASE_LAUNCHER, _name, _value)


def _validate_certified_source_binding_for_base_v1(
    source_root: Path, binding: Mapping[str, Any]
) -> str:
    relative, _ = _validate_certified_path_v1(
        Path(source_root).resolve(strict=True), binding
    )
    return relative


_BASE_LAUNCHER._validate_certified_source_binding_v13 = (
    _validate_certified_source_binding_for_base_v1
)


def _assert_runtime_adapter_v1() -> None:
    if any(
        getattr(_BASE_LAUNCHER, name, None) != value
        for name, value in _BASE_OVERRIDES.items()
    ):
        raise PermissionError("memory-role inherited runtime selectors changed")
    if (
        _BASE_LAUNCHER._build_one_microbatch_v13
        is not _V25_LAUNCHER._build_one_microbatch_v25
    ):
        raise PermissionError("memory-role physical builder is not exact V25")
    if (
        _BASE_LAUNCHER._validate_certified_source_binding_v13
        is not _validate_certified_source_binding_for_base_v1
    ):
        raise PermissionError("memory-role terminal source validator changed")


def _load_authority_file_v1(path: Path) -> dict[str, Any]:
    candidate = Path(path)
    expected = ROOT / AUTHORITY_RELATIVE_PATH
    try:
        root = ROOT.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        expected_resolved = expected.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise PermissionError("fixed memory-role authority is absent") from error
    if (
        resolved != expected_resolved
        or resolved != expected.absolute()
        or not resolved.is_relative_to(root)
        or candidate.is_symlink()
        or not candidate.is_file()
        or expected.is_symlink()
    ):
        raise PermissionError("memory-role authority must be its fixed regular file")
    raw = candidate.read_bytes()
    value = _strict_json_object_v1(raw, name="memory-role execution authority")
    if raw != _canonical_json_bytes_v1(value) + b"\n":
        raise PermissionError("memory-role authority must be canonical JSON")
    return value


def _validate_binding_shape_v1(value: Any, *, name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) not in (
        {"path", "file_sha256", "byte_count"},
        {"path", "file_sha256", "byte_count", "content_sha256"},
    ):
        raise PermissionError(f"memory-role {name} binding shape changed")
    if (
        type(value.get("path")) is not str
        or not value["path"]
        or not _is_lower_sha256_v1(value.get("file_sha256"))
        or type(value.get("byte_count")) is not int
        or value["byte_count"] <= 0
        or (
            "content_sha256" in value
            and not _is_lower_sha256_v1(value["content_sha256"])
        )
    ):
        raise PermissionError(f"memory-role {name} binding is malformed")
    return dict(value)


def validate_authority_v1(value: Any) -> dict[str, Any]:
    authority = _validate_content_bound_v1(
        value,
        name="memory-role execution authority",
        exact_keys=AUTHORITY_KEYS,
    )
    fixed = {
        "schema": AUTHORITY_SCHEMA,
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
        "runtime_data_root": RUNTIME_DATA_ROOT,
        "rgb_root_relative_path": RGB_ROOT_RELATIVE_PATH,
    }
    if any(authority.get(name) != expected for name, expected in fixed.items()):
        raise PermissionError("memory-role authority identity or cap changed")
    if not _is_commit_v1(authority.get("pinned_source_and_review_commit")):
        raise PermissionError("memory-role source-and-review commit is malformed")
    if not _is_commit_v1(authority.get("implementation_commit")):
        raise PermissionError("memory-role implementation commit is malformed")
    if authority.get("selectors") != {
        "executor_module": EXECUTOR_MODULE_NAME,
        "model_module": MODEL_MODULE_NAME,
        "model_class": MODEL_CLASS_NAME,
        "training_module": TRAINING_MODULE_NAME,
        "evaluation_module": EVALUATION_MODULE_NAME,
    }:
        raise PermissionError("memory-role selectors changed")
    if (
        _sha256_canonical_v1(authority.get("hardware")) != HARDWARE_SHA256
        or _sha256_canonical_v1(authority.get("runtime")) != RUNTIME_SHA256
        or _sha256_canonical_v1(authority.get("authorized_roles"))
        != AUTHORIZED_ROLES_SHA256
    ):
        raise PermissionError("memory-role inherited physical runtime changed")
    runtime_inputs = authority.get("runtime_inputs")
    expected_names = {*PHYSICAL_RUNTIME_INPUT_NAMES, *ROLE_RUNTIME_BINDINGS}
    if type(runtime_inputs) is not dict or set(runtime_inputs) != expected_names:
        raise PermissionError("memory-role runtime input inventory changed")
    for name, binding in runtime_inputs.items():
        _validate_binding_shape_v1(binding, name=name)
    physical = {name: runtime_inputs[name] for name in PHYSICAL_RUNTIME_INPUT_NAMES}
    if _sha256_canonical_v1(physical) != PHYSICAL_RUNTIME_INPUTS_SHA256:
        raise PermissionError("memory-role physical input bindings changed")
    if any(runtime_inputs[name] != binding for name, binding in ROLE_RUNTIME_BINDINGS.items()):
        raise PermissionError("memory-role local/place input bindings changed")
    certification = authority.get("clean_export_certification")
    if type(certification) is not dict or set(certification) != {
        "path",
        "file_sha256",
        "byte_count",
        "content_sha256",
    }:
        raise PermissionError("memory-role certification binding changed")
    _validate_binding_shape_v1(certification, name="certification")
    if certification["path"] != CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH:
        raise PermissionError("memory-role certification path changed")
    return authority


def _protected_source_path_v1(relative: str) -> bool:
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


def _validate_certified_path_v1(
    source_root: Path, binding: Any
) -> tuple[str, bytes]:
    value = _validate_binding_shape_v1(binding, name="certified source")
    if set(value) != {"path", "file_sha256", "byte_count"}:
        raise PermissionError("memory-role source binding gained metadata")
    relative = value["path"]
    if _protected_source_path_v1(relative):
        raise PermissionError("memory-role certification contains a protected path")
    path = source_root.joinpath(*PurePosixPath(relative).parts)
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise PermissionError("memory-role certified source is absent") from error
    if (
        resolved != path.absolute()
        or not resolved.is_relative_to(source_root)
        or path.is_symlink()
        or not path.is_file()
    ):
        raise PermissionError("memory-role certified source escaped")
    raw = path.read_bytes()
    if (
        len(raw) != value["byte_count"]
        or hashlib.sha256(raw).hexdigest() != value["file_sha256"]
    ):
        raise PermissionError(f"memory-role source changed: {relative}")
    return relative, raw


def validate_source_certification_v1(
    repository_root: Path, authority: Mapping[str, Any]
) -> dict[str, Any]:
    root = Path(repository_root).resolve(strict=True)
    path = root / CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH
    if path.is_symlink() or not path.is_file():
        raise PermissionError("memory-role clean-export certification is absent")
    raw = path.read_bytes()
    certification = _strict_json_object_v1(raw, name="memory-role certification")
    if raw != _canonical_json_bytes_v1(certification) + b"\n":
        raise PermissionError("memory-role certification must be canonical JSON")
    certification = _validate_content_bound_v1(
        certification,
        name="memory-role certification",
        exact_keys=CERTIFICATION_KEYS,
    )
    identity = authority.get("clean_export_certification")
    if identity != {
        "path": CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "content_sha256": certification["content_sha256"],
    }:
        raise PermissionError("memory-role authority does not bind certification")
    if (
        certification["schema"] != CERTIFICATION_SCHEMA
        or certification["status"] != "PASS_CLEAN_EXPORT_CERTIFIED"
        or certification["passed"] is not True
        or certification["certified_source_root"] != str(root)
        or certification["certified_source_root"]
        != authority.get("certified_source_root")
        or certification["pinned_source_and_review_commit"]
        != authority.get("pinned_source_and_review_commit")
    ):
        raise PermissionError("memory-role certification identity changed")
    bindings = certification["bindings"]
    if type(bindings) is not list or not bindings:
        raise PermissionError("memory-role source inventory is absent")
    if _sha256_canonical_v1(bindings) != certification["bindings_sha256"]:
        raise PermissionError("memory-role source inventory binding changed")
    paths = [_validate_certified_path_v1(root, binding)[0] for binding in bindings]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise PermissionError("memory-role source inventory order changed")
    if not REQUIRED_CERTIFIED_SOURCE_PATHS.issubset(paths):
        raise PermissionError("memory-role certification omits required source")
    return {
        "schema": f"{SCHEMA_PREFIX}_source_validation_receipt_v1",
        "passed": True,
        "validated_path_count": len(paths),
        "bindings_sha256": certification["bindings_sha256"],
        "certification_content_sha256": certification["content_sha256"],
        "certified_export_binding_count": len(bindings),
        "certified_source_bindings_sha256": certification["bindings_sha256"],
        "certified_source_bindings": [dict(binding) for binding in bindings],
    }


def _terminal_exists_v1(output_root: Path) -> bool:
    return any(
        path.exists() or path.is_symlink()
        for path in (output_root / "success.json", output_root / "failure.json")
    )


def execute_future_authorized_v1(
    *, repository_root: Path, authority: Mapping[str, Any]
) -> Mapping[str, Any]:
    _assert_runtime_adapter_v1()
    repository = Path(repository_root).resolve(strict=True)
    if repository != ROOT.resolve(strict=True):
        raise PermissionError("memory-role execution requires its certified export")
    fixed_authority = _load_authority_file_v1(ROOT / AUTHORITY_RELATIVE_PATH)
    if type(authority) is not dict or authority != fixed_authority:
        raise PermissionError("supplied memory-role authority differs from fixed file")
    validated_authority = validate_authority_v1(fixed_authority)
    validate_pre_reservation_gpu_visibility_v1()
    _BASE_LAUNCHER._validate_certified_source_root_v13(
        repository, validated_authority
    )
    _BASE_LAUNCHER._activate_certified_source_root_v13(repository)
    source_evidence = validate_source_certification_v1(
        repository, validated_authority
    )
    executor = importlib.import_module(EXECUTOR_MODULE_NAME)
    executor.validate_future_execution_prerequisites_v1(validated_authority)

    def validate_bound_sources_for_runtime_v1(source_root: Path) -> dict[str, Any]:
        return validate_source_certification_v1(source_root, validated_authority)

    executor.validate_bound_sources_v13 = validate_bound_sources_for_runtime_v1
    _BASE_LAUNCHER._validate_runtime_data_root_v13(repository, validated_authority)
    _BASE_LAUNCHER._ensure_output_parent_v13(
        repository, OUTPUT_ROOT_RELATIVE_PATH
    )
    reservation = executor.reserve_attempt_v1(
        repository,
        validated_authority,
        created_utc=_BASE_LAUNCHER._utc_now_v13(),
    )
    output_root = repository / OUTPUT_ROOT_RELATIVE_PATH
    runtime: Any = None
    stage = "post_reservation_runtime_composition"
    try:
        runtime = _BASE_LAUNCHER.compose_runtime_v13(
            repository_root=repository,
            authority=validated_authority,
            reservation=reservation,
            source_evidence=source_evidence,
        )
        publisher = _BASE_LAUNCHER.V13WriteOncePublisher(output_root, executor)
        stage = "future_authorized_engine_v1"
        result = executor.run_future_authorized_engine_v1(
            authority=validated_authority,
            reservation=reservation,
            runtime=runtime,
            publisher=publisher,
        )
        if not isinstance(result, Mapping):
            raise RuntimeError("memory-role controller omitted terminal receipt")
        return dict(result)
    except BaseException as error:
        if not _terminal_exists_v1(output_root):
            executor.terminalize_failure_v1(
                output_root,
                reservation,
                stage=stage,
                error=error,
                created_utc=_BASE_LAUNCHER._utc_now_v13(),
            )
        raise
    finally:
        if runtime is not None:
            runtime.close_v13()


def private_launcher_adapter_receipt_v1() -> dict[str, Any]:
    _assert_runtime_adapter_v1()
    return {
        "schema": f"{SCHEMA_PREFIX}_private_launcher_adapter_v1",
        "base_launcher": {
            "path": BASE_LAUNCHER_RELATIVE_PATH,
            "commit": BASE_LAUNCHER_COMMIT,
            "file_sha256": BASE_LAUNCHER_FILE_SHA256,
            "byte_count": BASE_LAUNCHER_BYTE_COUNT,
        },
        "physical_microbatch_builder": "v25_exact",
        "physical_runtime_composer": "v13_exact",
        "physical_presentations_per_update": 16,
        "local_presentations_per_update": 8,
        "place_presentations_per_update": 8,
        "model_module": MODEL_MODULE_NAME,
        "training_module": TRAINING_MODULE_NAME,
        "evaluation_module": EVALUATION_MODULE_NAME,
        "executor_module": EXECUTOR_MODULE_NAME,
        "maximum_updates": MAXIMUM_UPDATES,
        "maximum_presentations": MAXIMUM_PRESENTATIONS,
        "one_shot_attempt_count": 1,
        "retry_authorized": False,
        "resume_authorized": False,
        "execution_authorized_by_source": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--future-authority", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if not arguments:
        print(
            json.dumps(
                {
                    "schema": LAUNCHER_SCHEMA,
                    "status": "DENIED_NO_FUTURE_AUTHORITY",
                    "scientific_payload_opened": False,
                    "reservation_created": False,
                },
                sort_keys=True,
            )
        )
        return 4
    parsed = _parser().parse_args(arguments)
    authority = _load_authority_file_v1(parsed.future_authority)
    result = execute_future_authorized_v1(repository_root=ROOT, authority=authority)
    print(json.dumps(dict(result), sort_keys=True))
    status = result.get("status")
    if status == "PASS_DEVELOPMENT_UPDATE400_TERMINAL":
        return 0
    if isinstance(status, str) and status.startswith("FAIL_"):
        return 2
    raise RuntimeError("memory-role controller returned a nonterminal status")


if __name__ == "__main__":
    raise SystemExit(main())
