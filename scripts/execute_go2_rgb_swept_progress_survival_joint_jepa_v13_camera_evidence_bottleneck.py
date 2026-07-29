#!/usr/bin/env python3
"""Denied-by-default executor contracts for Camera-evidence-bottleneck V13.

This file intentionally contains only source-verifiable contracts: custody
receipts, frozen metric adapters, gates, accounting checks, and runtime API
validation.  It does not discover or open scientific inputs.  Execution stays
closed until a later reviewed source closure, clean-export certification, and
execution binding are all supplied by the custodian-owned launcher.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence


SCHEMA_PREFIX = "lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13"
PREREGISTRATION_COMMIT = "ba735b4c2a66168c6dd058fcfb0ed3095d350ac3"
EXPECTED_RUNTIME_FINGERPRINT = {
    "executable": (
        "/home/andrewknowles/.local/share/lewmquad-v12-runtime-rocm711/bin/python"
    ),
    "python": "3.12.3",
    "torch": "2.14.0.dev20260726+rocm7.1",
    "torch_hip": "7.1.52802",
    "numpy": "1.26.4",
    "pillow": "10.2.0",
}
PREREGISTRATION_PATH = (
    "docs/lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_"
    "integrity_replacement_v1_preregistration_2026-07-29.md"
)
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_"
    "integrity_replacement_v1/attempt_v1"
)

# A later committed binding is necessary but cannot silently change this
# source-only checkout into an executor.
CURRENT_EXECUTION_AUTHORIZED = False
CURRENT_EXECUTION_DENIAL = (
    "V13 scientific execution is denied until its recursive source closure, "
    "independent exact-binding review, custody clean-export exception and "
    "certification, frozen-export validation, and one-shot execution binding "
    "are committed and validated by the custodian-owned launcher"
)

CONSTRUCTOR_INITIALIZATION_SEED = 20_260_712
SCHEDULE_SEED = 20_260_713
EXPERIMENT_SEED = 20_260_728
BOOTSTRAP_SEED = 20_260_728
PROJECTION_INITIALIZATION_SEED = 20_260_729

MICROBATCH_SIZE = 4
MICROBATCHES_PER_UPDATE = 4
PRESENTATIONS_PER_UPDATE = 16
MAXIMUM_UPDATES = 1_000
MAXIMUM_PRESENTATIONS = 16_000
OBSERVATION_UPDATES = (0, 100, 400, 1_000)
TERMINAL_UPDATES = (400, 1_000)
CHECKPOINT_SCHEDULE_PREFIX_SHA256 = {
    100: "9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51",
    400: "6e7e5cc766c0a768b5771181cfaf2583598c1c22e5d4fc19e6ff1b245a5c8f92",
    1_000: "3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528",
}

CONTROL_NAMES = (
    "coordinate_matched_persistence",
    "shuffled_action",
    "wrong_rgb",
    "train_action_mean_prior",
)
CONTROL_CHECK_NAMES = (
    "positive_equal_scene_delta",
    "positive_bootstrap_lower_95",
    "positive_family_count",
)
V12_GATE_CHECK_NAMES = (
    "semantic_balanced_accuracy",
    "semantic_free_recall",
    "semantic_occupied_recall",
    "semantic_unknown_recall",
    "semantic_rough_occupied_recall",
    "selection_registered_families",
    "selection_informative_utility",
    "selection_zero_prefix_rate",
    "selection_pair_concordance",
    "all_family_utility",
    "all_family_zero_prefix_rate",
    "all_family_pair_concordance",
    *tuple(
        f"{control}:{check}"
        for control in CONTROL_NAMES
        for check in CONTROL_CHECK_NAMES
    ),
)

PHYSICAL_LOWER_THRESHOLDS = {
    "pixel_first_hit_balanced_accuracy": 0.95,
    "ground_clear_balanced_accuracy": 0.95,
    "derived_raster_balanced_accuracy": 0.95,
    "wrong_rgb_pixel_balanced_accuracy_drop": 0.12,
    "wrong_rgb_depth_median_error_increase_m": 0.12,
    "wrong_rgb_depth_p95_error_increase_m": 0.20,
    "wrong_rgb_ground_balanced_accuracy_drop": 0.12,
    "wrong_rgb_raster_nll_increase": 0.12,
    "wrong_rgb_raster_balanced_accuracy_drop": 0.12,
}
PHYSICAL_UPPER_THRESHOLDS = {
    "depth_median_error_m": 0.10,
    "depth_p95_error_m": 0.25,
    "derived_raster_nll": 0.15,
}
DISTANCE_GROUP_NAMES = (
    "0.0_to_1.0",
    "1.0_to_2.0",
    "2.0_to_3.0",
    "3.0_to_4.0",
    "4.0_to_5.0",
    "5.0_plus",
)
PRESENT_CLASS_NAMES = ("free", "occupied", "unknown")
SCOPES = (
    "aggregate",
    "large_enclosed_maze",
    "local_composite_motifs",
    "loop_alias_stress",
    "medium_enclosed_maze",
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "visual_sensor_stress",
)
REGISTERED_FAMILIES = SCOPES[1:]
MARGINS_PER_SCOPE = 21
PHYSICAL_MARGIN_COUNT = 189

FINAL_PHYSICAL_THRESHOLDS = {
    "passed_margin_count_minimum": 112,
    "total_shortfall_strictly_less_than": 33.05143763708337,
    "complete_physical_scope_count_minimum": 1,
    "rough_pixel_balanced_accuracy_strictly_greater_than":
        0.8198594673963917,
    "rough_ground_balanced_accuracy_strictly_greater_than":
        0.647134926562893,
    "rough_depth_p95_m_strictly_less_than": 0.9777327477931971,
}

# These are the preregistered parent implementations that define the fine
# losses and physical metric arithmetic.  V13's own recursive closure must be
# independently bound later; these entries do not grant execution authority.
BOUND_PARENT_SOURCES = {
    PREREGISTRATION_PATH: (
        "3721e937106f837fa7877dd18d8899779f9a0c747b92d7d177850af1de92ea54",
        3_571,
    ),
    "lewm/models/observable_camera_ray_evidence_v4_hierarchical_first_hit_v9.py": (
        "52bc99f0ba59c2cf7444221931169ba57af61f343308b85625877c7a257adffd",
        9_240,
    ),
    "lewm/models/shared_observable_camera_ray_jepa_v5.py": (
        "ee3fd612bb0a40d615fda7f7110091a330c849f8b1ec2b48cc0af3e406c928fc",
        183_593,
    ),
    "lewm/models/observable_camera_ray_evidence_v4_training.py": (
        "c0f3f944883987950edb7579a9e108171486122a9a3ae9d84d2a1abb6ac015ed",
        32_751,
    ),
    "scripts/run_go2_shared_jepa_v5_matched_training_v1.py": (
        "e98bd8cceed26288ebcbf8a02eac03c72be6d06a539953927754353e049a5578",
        103_456,
    ),
    "lewm/benchmarks/go2_observable_camera_ray_fit_v4_metrics.py": (
        "6a0e40f9dcb496831553dc5bbc6d1efcdf6d82676d6f18aa20e417f8de4fa6a0",
        14_068,
    ),
    "lewm/benchmarks/go2_shared_jepa_v5_matched_training_v1.py": (
        "53a7fac793a1b46764d49e7259fd637ec02b20111927effd01cdcd09682c206a",
        41_189,
    ),
}

MODEL_CLASS_NAME = "GeometryAnchoredSweptProgressSurvivalJointJepaV13"
MODEL_REQUIRED_METHODS = (
    "encode_online",
    "encode_target",
    "encode_online_with_evidence",
    "encode_online_with_auxiliary_evidence",
    "encode_online_training",
    "semantic_logits_from_latent",
    "online_state",
    "predict_all_actions_with_survival",
    "update_target_ema_after_optimizer_step",
    "trainable_parameter_groups_v13",
)
MODEL_REQUIRED_CONSTANTS = {
    "SHARED_ROUTE_PARAMETER_COUNT_V13": 3_105_513,
    "REPRESENTATION_GROUP_PARAMETER_COUNT_V13": 22_020,
    "PREDICTOR_GROUP_PARAMETER_COUNT_V13": 259_073,
    "ONLINE_TRAINABLE_PARAMETER_COUNT_V13": 3_386_606,
    "TARGET_BOTTLENECK_PARAMETER_COUNT_V13": 3_108_905,
    "CAMERA_EVIDENCE_PROJECTION_PARAMETER_COUNT_V13": 3_392,
}
TRAINING_REQUIRED_FUNCTIONS = (
    "partition_parameters_v13",
    "build_frozen_optimizer_v13",
    "validate_optimizer_v13",
    "joint_training_update_v13",
    "validate_accounting_v13",
)
TRAINING_REQUIRED_BATCH_KEYS = (
    "current_rgb",
    "next_rgb",
    "current_labels",
    "next_labels",
    "executed_action_indices",
    "immediate_feasible",
    "swept_progress_prefix_lengths",
    "current_camera_origin_body_m",
    "next_camera_origin_body_m",
    "current_camera_basis_body_fru",
    "next_camera_basis_body_fru",
    "current_ground_plane_z_body_m",
    "next_ground_plane_z_body_m",
    "current_pixel_hit_mask",
    "next_pixel_hit_mask",
    "current_pixel_first_hit_distance_m",
    "next_pixel_first_hit_distance_m",
    "current_ground_support_in_frustum",
    "next_ground_support_in_frustum",
    "current_ground_support_clear_to_target",
    "next_ground_support_clear_to_target",
)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _content_bound(value: Mapping[str, Any]) -> dict[str, Any]:
    core = dict(value)
    if "content_sha256" in core:
        raise ValueError("content_sha256 is computed, not supplied")
    result = dict(core)
    result["content_sha256"] = hashlib.sha256(_canonical_json_bytes(core)).hexdigest()
    return result


def validate_content_bound_v13(value: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict or "content_sha256" not in value:
        raise ValueError("receipt must be a content-bound plain dict")
    core = dict(value)
    declared = core.pop("content_sha256")
    expected = hashlib.sha256(_canonical_json_bytes(core)).hexdigest()
    if declared != expected:
        raise ValueError("receipt content binding differs")
    return dict(value)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_bound_sources_v13(
    repository_root: Path,
    bindings: Mapping[str, tuple[str, int]] = BOUND_PARENT_SOURCES,
) -> dict[str, Any]:
    """Validate only explicitly named source files; never discover the tree."""

    root = Path(repository_root)
    if type(bindings) is not dict or not bindings:
        raise ValueError("source bindings must be a nonempty plain dict")
    validated = []
    for relative_path, binding in bindings.items():
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts or len(binding) != 2:
            raise ValueError("source binding path or tuple is malformed")
        expected_sha256, expected_bytes = binding
        if not _is_sha256(expected_sha256) or type(expected_bytes) is not int:
            raise ValueError("source binding digest or byte count is malformed")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"bound source is absent or not regular: {relative_path}")
        raw = path.read_bytes()
        if len(raw) != expected_bytes or hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise PermissionError(f"bound source identity differs: {relative_path}")
        validated.append(relative_path)
    return {
        "schema": f"{SCHEMA_PREFIX}_parent_source_validation_v1",
        "validated_paths": validated,
        "validated_path_count": len(validated),
        "execution_authority_granted": False,
    }


def validate_model_api_v13(module: Any) -> dict[str, Any]:
    model_class = getattr(module, MODEL_CLASS_NAME, None)
    if not isinstance(model_class, type):
        raise RuntimeError("V13 model class is absent")
    missing = [name for name in MODEL_REQUIRED_METHODS if not callable(getattr(model_class, name, None))]
    if missing:
        raise RuntimeError(f"V13 model API is incomplete: {missing}")
    for name, expected in MODEL_REQUIRED_CONSTANTS.items():
        if getattr(module, name, None) != expected:
            raise RuntimeError(f"V13 model constant changed: {name}")
    expected_prefixes = {
        "SHARED_PARAMETER_PREFIXES_V13": ("encoder.", "bev_lift.evidence_head."),
        "REPRESENTATION_PARAMETER_PREFIXES_V13": (
            "bev_lift.free_projection.",
            "bev_lift.occupied_projection.",
            "semantic_head.",
        ),
        "PREDICTOR_PARAMETER_PREFIXES_V13": ("predictor.",),
        "TARGET_PARAMETER_PREFIXES_V13": (
            "target_encoder.",
            "target_bev_lift.evidence_head.",
            "target_bev_lift.free_projection.",
            "target_bev_lift.occupied_projection.",
        ),
    }
    for name, expected in expected_prefixes.items():
        if tuple(getattr(module, name, ())) != expected:
            raise RuntimeError(f"V13 model parameter prefixes changed: {name}")
    return {
        "model_class": MODEL_CLASS_NAME,
        "method_count": len(MODEL_REQUIRED_METHODS),
        "online_trainable_parameter_count": MODEL_REQUIRED_CONSTANTS[
            "ONLINE_TRAINABLE_PARAMETER_COUNT_V13"
        ],
    }


def validate_training_api_v13(module: Any) -> dict[str, Any]:
    missing = [name for name in TRAINING_REQUIRED_FUNCTIONS if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError(f"V13 training API is incomplete: {missing}")
    if (
        getattr(module, "MICROBATCH_SIZE", None) != MICROBATCH_SIZE
        or getattr(module, "MICROBATCHES_PER_UPDATE", None) != MICROBATCHES_PER_UPDATE
        or getattr(module, "PRESENTATIONS_PER_UPDATE", None) != PRESENTATIONS_PER_UPDATE
        or getattr(module, "MAXIMUM_UPDATES", None) != MAXIMUM_UPDATES
        or getattr(module, "MAXIMUM_PRESENTATIONS", None) != MAXIMUM_PRESENTATIONS
    ):
        raise RuntimeError("V13 training caps changed")
    if tuple(getattr(module, "REQUIRED_BATCH_KEYS", ())) != TRAINING_REQUIRED_BATCH_KEYS:
        raise RuntimeError("V13 training batch schema changed")
    return {
        "required_function_count": len(TRAINING_REQUIRED_FUNCTIONS),
        "required_batch_key_count": len(TRAINING_REQUIRED_BATCH_KEYS),
        "presentations_per_update": PRESENTATIONS_PER_UPDATE,
    }


FUTURE_AUTHORITY_FIELDS = {
    "schema",
    "status",
    "preregistration_commit",
    "frozen_source_and_review_commit",
    "recursive_source_closure_manifest_sha256",
    "independent_source_review_sha256",
    "recursive_source_closure_reviewed",
    "exact_path_sha256_byte_count_review_passed",
    "custody_clean_export_exception_committed",
    "clean_export_certification_sha256",
    "exported_paths_frozen_commit_validated",
    "execution_binding_commit",
    "scientific_payload_authorized",
    "one_shot",
    "retry_authorized",
    "resume_authorized",
    "maximum_updates",
    "maximum_presentations",
    "output_root",
    "certified_source_root",
    "runtime_data_root",
    "runtime_inputs",
    "authorized_roles",
    "hardware",
    "runtime",
}

RUNTIME_INPUT_BINDING_NAMES = (
    "raw_manifest",
    "raw_audit",
    "n320_gate",
    "n320_checkpoint",
    "schedule",
    "swept_label_manifest",
    "train_labels",
    "checkpoint_selection_labels",
)


def _validate_runtime_binding_v13(value: object, *, name: str) -> dict[str, Any]:
    allowed_fields = {"path", "file_sha256", "byte_count"}
    if type(value) is not dict or set(value) not in (
        allowed_fields,
        allowed_fields | {"content_sha256"},
    ):
        raise PermissionError(f"future V13 runtime binding is incomplete: {name}")
    path = Path(value["path"])
    if (
        path.is_absolute()
        or ".." in path.parts
        or not _is_sha256(value["file_sha256"])
        or type(value["byte_count"]) is not int
        or value["byte_count"] <= 0
        or (
            "content_sha256" in value
            and not _is_sha256(value["content_sha256"])
        )
    ):
        raise PermissionError(f"future V13 runtime binding is malformed: {name}")
    return dict(value)


def validate_future_execution_prerequisites_v13(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a future receipt; this function itself grants no authority."""

    if type(value) is not dict or set(value) != FUTURE_AUTHORITY_FIELDS:
        raise PermissionError("future V13 authority receipt fields changed")
    digest_fields = (
        "recursive_source_closure_manifest_sha256",
        "independent_source_review_sha256",
        "clean_export_certification_sha256",
    )
    commit_fields = ("frozen_source_and_review_commit", "execution_binding_commit")
    if any(not _is_sha256(value[name]) for name in digest_fields) or any(
        not _is_git_commit(value[name]) for name in commit_fields
    ):
        raise PermissionError("future V13 authority binding is malformed")
    required_true = (
        "recursive_source_closure_reviewed",
        "exact_path_sha256_byte_count_review_passed",
        "custody_clean_export_exception_committed",
        "exported_paths_frozen_commit_validated",
        "scientific_payload_authorized",
        "one_shot",
    )
    runtime_inputs = value.get("runtime_inputs")
    certified_source_root = value.get("certified_source_root")
    runtime_data_root = value.get("runtime_data_root")
    roles = value.get("authorized_roles")
    hardware = value.get("hardware")
    if type(runtime_inputs) is not dict or set(runtime_inputs) != set(RUNTIME_INPUT_BINDING_NAMES):
        raise PermissionError("future V13 runtime-input binding set changed")
    roots = (certified_source_root, runtime_data_root)
    if any(
        not isinstance(root, str)
        or not root
        or not Path(root).is_absolute()
        or ".." in Path(root).parts
        or str(Path(root)) != root
        or Path(root) == Path("/")
        for root in roots
    ) or (
        Path(certified_source_root).is_relative_to(Path(runtime_data_root))
        or Path(runtime_data_root).is_relative_to(Path(certified_source_root))
    ):
        raise PermissionError(
            "future V13 certified source and runtime data roots are not canonical and disjoint"
        )
    for name in RUNTIME_INPUT_BINDING_NAMES:
        _validate_runtime_binding_v13(runtime_inputs[name], name=name)
    expected_roles = {
        "train": {"pairs": 4_262, "scenes": 72, "unique_endpoints": 7_777},
        "checkpoint_selection": {
            "pairs": 495,
            "scenes": 8,
            "unique_endpoints": 924,
        },
        "probability_calibration_open_count": 0,
    }
    expected_hardware = {
        "visible_device_count": 1,
        "name": "AMD Radeon AI PRO R9700",
        "total_memory_bytes": 34_208_743_424,
        "isolated_python": True,
    }
    if (
        roles != expected_roles
        or hardware != expected_hardware
        or value.get("runtime") != EXPECTED_RUNTIME_FINGERPRINT
    ):
        raise PermissionError(
            "future V13 role population, hardware, or runtime binding changed"
        )
    if (
        value["schema"] != f"{SCHEMA_PREFIX}_future_execution_authority_v1"
        or value["status"] != "AUTHORIZED_CERTIFIED_NARROW_EXPORT_ONE_SHOT"
        or value["preregistration_commit"] != PREREGISTRATION_COMMIT
        or any(value[name] is not True for name in required_true)
        or value["retry_authorized"] is not False
        or value["resume_authorized"] is not False
        or value["maximum_updates"] != MAXIMUM_UPDATES
        or value["maximum_presentations"] != MAXIMUM_PRESENTATIONS
        or value["output_root"] != OUTPUT_ROOT_RELATIVE_PATH
    ):
        raise PermissionError("future V13 authority prerequisites are not conjunctively met")
    return dict(value)


def execution_denial_receipt_v13() -> dict[str, Any]:
    return _content_bound(
        {
            "schema": f"{SCHEMA_PREFIX}_current_execution_denial_v1",
            "status": "DENIED_SOURCE_ONLY",
            "reason": CURRENT_EXECUTION_DENIAL,
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "output_root": OUTPUT_ROOT_RELATIVE_PATH,
            "scientific_payload_opened": False,
            "reservation_created": False,
            "attempt_consumed": False,
            "retry_authorized": False,
            "resume_authorized": False,
        }
    )


def _write_immutable_json_v13(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Publish one content-bound JSON file without a replace-overwrite path."""

    bound = _content_bound(value)
    raw = _canonical_json_bytes(bound) + b"\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"write-once receipt already exists: {path.name}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
        os.unlink(temporary)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        raise
    return bound


def reserve_attempt_v13(
    repository_root: Path,
    future_authority: Mapping[str, Any],
    *,
    created_utc: str,
) -> dict[str, Any]:
    """Create the future one-shot root and immutable pre-payload reservation."""

    authority = validate_future_execution_prerequisites_v13(future_authority)
    repository = Path(repository_root)
    if repository.is_symlink() or not repository.is_dir():
        raise PermissionError("certified export repository root is absent or not regular")
    root = repository / OUTPUT_ROOT_RELATIVE_PATH
    if root.parent.is_symlink() or not root.parent.is_dir():
        raise PermissionError("custodian must create the exact V13 output parent first")
    if root.exists() or root.is_symlink():
        raise FileExistsError("V13 output root must be absent before reservation")
    if not isinstance(created_utc, str) or not created_utc:
        raise ValueError("created_utc must be a nonempty caller-bound timestamp")
    os.mkdir(root, 0o700)
    os.chmod(root, 0o700)
    reservation_core = {
        "schema": f"{SCHEMA_PREFIX}_attempt_reservation_v1",
        "status": "RESERVED_BEFORE_SCIENTIFIC_PAYLOAD",
        "created_utc": created_utc,
        "authority_sha256": hashlib.sha256(_canonical_json_bytes(authority)).hexdigest(),
        "one_shot_attempt": 1,
        "attempt_consumed": True,
        "maximum_updates": MAXIMUM_UPDATES,
        "maximum_presentations": MAXIMUM_PRESENTATIONS,
        "retry_authorized": False,
        "resume_authorized": False,
    }
    try:
        return _write_immutable_json_v13(root / "reservation.json", reservation_core)
    except BaseException:
        # The directory itself is already the fail-closed consumption marker.
        # Never remove it after reservation begins.
        raise


def terminalize_failure_v13(
    output_root: Path,
    reservation: Mapping[str, Any],
    *,
    stage: str,
    error: BaseException,
    created_utc: str,
) -> dict[str, Any]:
    """Publish a terminal, non-retryable failure without leaking error text."""

    root = Path(output_root)
    validated_reservation = validate_content_bound_v13(reservation)
    reservation_path = root / "reservation.json"
    if (
        not root.is_dir()
        or root.is_symlink()
        or stat.S_IMODE(root.stat().st_mode) != 0o700
        or reservation_path.is_symlink()
        or not reservation_path.is_file()
        or stat.S_IMODE(reservation_path.stat().st_mode) != 0o444
    ):
        raise PermissionError("V13 reserved root integrity differs")
    raw_reservation = reservation_path.read_bytes()
    if raw_reservation != _canonical_json_bytes(validated_reservation) + b"\n":
        raise PermissionError("V13 on-disk reservation differs from supplied receipt")
    if not isinstance(stage, str) or not stage or not isinstance(error, BaseException):
        raise ValueError("terminal failure stage or exception is malformed")
    if not isinstance(created_utc, str) or not created_utc:
        raise ValueError("created_utc must be nonempty")
    if (root / "success.json").exists() or (root / "failure.json").exists():
        raise FileExistsError("V13 attempt already has a terminal receipt")
    return _write_immutable_json_v13(
        root / "failure.json",
        {
            "schema": f"{SCHEMA_PREFIX}_terminal_failure_v1",
            "status": "FAIL_TERMINAL_NO_RETRY_NO_RESUME",
            "created_utc": created_utc,
            "stage": stage,
            "exception_type": type(error).__name__,
            "exception_message_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
            "reservation_content_sha256": validated_reservation["content_sha256"],
            "attempt_consumed": True,
            "retry_authorized": False,
            "resume_authorized": False,
        },
    )


@dataclass(frozen=True)
class SemanticProbabilityRasterV13:
    """The only raster object admitted to the inherited metric accumulator."""

    class_probabilities: Any

    def __post_init__(self) -> None:
        import torch

        value = self.class_probabilities
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 4
            or value.shape[1] != 3
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).all().item())
            or bool((value < 0).any().item())
        ):
            raise ValueError("V13 semantic probabilities must be finite nonnegative (B,3,H,W)")
        sums = value.sum(dim=1)
        if not torch.allclose(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-6):
            raise ValueError("V13 semantic probabilities must sum to one")


def adapt_nominal_logits_with_target_metadata_v13(
    nominal_output: Any,
    *,
    target_ground_in_frustum: Any,
    target_ground_distance_m: Any,
) -> Any:
    """Attach evaluator metadata while preserving every nominal learned tensor."""

    import torch
    from lewm.models.observable_camera_ray_evidence_v4 import (
        ObservableCameraRayEvidenceV4RawOutput,
    )

    if not isinstance(nominal_output, ObservableCameraRayEvidenceV4RawOutput):
        raise TypeError("nominal_output must be the V4 raw evidence schema")
    shape = tuple(nominal_output.ground_clear_to_target_logits.shape)
    if (
        not isinstance(target_ground_in_frustum, torch.Tensor)
        or target_ground_in_frustum.dtype != torch.bool
        or tuple(target_ground_in_frustum.shape) != shape
        or not isinstance(target_ground_distance_m, torch.Tensor)
        or not target_ground_distance_m.is_floating_point()
        or tuple(target_ground_distance_m.shape) != shape
        or target_ground_in_frustum.device
        != nominal_output.ground_clear_to_target_logits.device
        or target_ground_distance_m.device
        != nominal_output.ground_clear_to_target_logits.device
        or not bool(torch.isfinite(target_ground_distance_m).all().item())
    ):
        raise ValueError("target-derived ground metadata shape, dtype, device, or finiteness differs")
    return ObservableCameraRayEvidenceV4RawOutput(
        pixel_first_hit_hazard_logits=nominal_output.pixel_first_hit_hazard_logits,
        pixel_within_bin_offset_m=nominal_output.pixel_within_bin_offset_m,
        ground_clear_to_target_logits=nominal_output.ground_clear_to_target_logits,
        ground_query_in_frustum=target_ground_in_frustum,
        ground_query_uv_px=nominal_output.ground_query_uv_px,
        ground_target_distance_m=target_ground_distance_m,
    )


def update_physical_accumulator_from_rgb_v13(
    model: Any,
    accumulator: Any,
    *,
    selected_rgb: Any,
    target_camera_origin_body_m: Any,
    target_camera_basis_body_fru: Any,
    target_ground_plane_z_body_m: Any,
    targets: Any,
    target_raster_labels: Any,
    families: Sequence[str],
) -> dict[str, Any]:
    """Qualifying metric update with learned values derived internally.

    The caller may supply target context and labels, but cannot supply learned
    evidence or raster probabilities.  This call site therefore cannot be
    redirected to auxiliary Camera logits or the rejected old rasterizer.
    """

    import torch

    encoding = model.encode_online_with_evidence(selected_rgb)
    nominal_output = encoding.nominal_evidence
    query = model.bev_lift.evidence_head.ground_query_geometry(
        target_camera_origin_body_m,
        target_camera_basis_body_fru,
        target_ground_plane_z_body_m,
    )
    if not torch.equal(query.in_frustum, targets.ground_in_frustum):
        raise PermissionError(
            "target-derived query visibility differs from bound supervision"
        )
    semantic_logits = model.semantic_logits_from_latent(encoding.latent)
    semantic_class_probabilities = torch.softmax(semantic_logits, dim=1)

    adapted = adapt_nominal_logits_with_target_metadata_v13(
        nominal_output,
        target_ground_in_frustum=targets.ground_in_frustum,
        target_ground_distance_m=query.target_distance_m,
    )
    accumulator.update(
        raw_output=adapted,
        targets=targets,
        soft_raster=SemanticProbabilityRasterV13(semantic_class_probabilities),
        target_raster_labels=target_raster_labels,
        families=families,
    )
    return {
        "model_entrypoint": "encode_online_with_evidence",
        "learned_evidence_source": "encoding.nominal_evidence",
        "semantic_probability_source": (
            "softmax(model.semantic_logits_from_latent(encoding.latent),dim=1)"
        ),
        "target_metadata_only": ("ground_query_in_frustum", "ground_target_distance_m"),
        "auxiliary_logits_used": False,
        "old_camera_raster_used": False,
        "batch_size": int(selected_rgb.shape[0]),
    }


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _required_path(value: Mapping[str, Any], *names: str) -> Any:
    current: Any = value
    for name in names:
        if not isinstance(current, Mapping) or name not in current:
            raise ValueError(f"physical metric path is absent: {'.'.join(names)}")
        current = current[name]
    return current


def flatten_physical_metrics_v13(
    correct: Mapping[str, Any],
    wrong: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen Shared-V5 matched/wrong metric flattening."""

    distance_rows = _required_path(correct, "ground_clear", "by_distance_m")
    class_recalls = _required_path(correct, "derived_raster", "class_recalls")
    if tuple(distance_rows) != DISTANCE_GROUP_NAMES or tuple(class_recalls) != (
        "unknown",
        "free",
        "occupied",
    ):
        raise ValueError("physical distance or raster class schema changed")
    if any(_required_path(distance_rows, name, "count") <= 0 for name in DISTANCE_GROUP_NAMES):
        raise ValueError("every registered distance group must be present")
    if any(class_recalls[name] is None for name in class_recalls):
        raise ValueError("every registered raster class must be present")

    correct_depth = _required_path(correct, "pixel_hit_depth")
    wrong_depth = _required_path(wrong, "pixel_hit_depth")
    correct_raster = _required_path(correct, "derived_raster")
    wrong_raster = _required_path(wrong, "derived_raster")
    return {
        "pixel_first_hit_balanced_accuracy": _required_path(
            correct, "pixel_hit_no_hit", "balanced_accuracy"
        ),
        "depth_median_error_m": correct_depth["median_absolute_error_m"],
        "depth_p95_error_m": correct_depth["p95_absolute_error_m"],
        "ground_clear_balanced_accuracy": _required_path(
            correct, "ground_clear", "overall", "balanced_accuracy"
        ),
        "distance_group_balanced_accuracy": [
            distance_rows[name]["balanced_accuracy"] for name in DISTANCE_GROUP_NAMES
        ],
        "derived_raster_nll": correct_raster["nll"],
        "derived_raster_balanced_accuracy": correct_raster["balanced_accuracy"],
        "present_class_recall": {
            name: class_recalls[name] for name in PRESENT_CLASS_NAMES
        },
        "wrong_rgb_pixel_balanced_accuracy_drop": _required_path(
            correct, "pixel_hit_no_hit", "balanced_accuracy"
        )
        - _required_path(wrong, "pixel_hit_no_hit", "balanced_accuracy"),
        "wrong_rgb_depth_median_error_increase_m": wrong_depth[
            "median_absolute_error_m"
        ]
        - correct_depth["median_absolute_error_m"],
        "wrong_rgb_depth_p95_error_increase_m": wrong_depth["p95_absolute_error_m"]
        - correct_depth["p95_absolute_error_m"],
        "wrong_rgb_ground_balanced_accuracy_drop": _required_path(
            correct, "ground_clear", "overall", "balanced_accuracy"
        )
        - _required_path(wrong, "ground_clear", "overall", "balanced_accuracy"),
        "wrong_rgb_raster_nll_increase": wrong_raster["nll"] - correct_raster["nll"],
        "wrong_rgb_raster_balanced_accuracy_drop": correct_raster["balanced_accuracy"]
        - wrong_raster["balanced_accuracy"],
    }


def physical_margins_v13(metrics: Mapping[str, Any]) -> list[float]:
    if type(metrics) is not dict:
        raise TypeError("physical metrics must be a plain dict")
    margins = [
        (_finite(metrics.get(name), name=name) - threshold) / threshold
        for name, threshold in PHYSICAL_LOWER_THRESHOLDS.items()
    ]
    margins.extend(
        (threshold - _finite(metrics.get(name), name=name)) / threshold
        for name, threshold in PHYSICAL_UPPER_THRESHOLDS.items()
    )
    distance = metrics.get("distance_group_balanced_accuracy")
    recalls = metrics.get("present_class_recall")
    if (
        not isinstance(distance, Sequence)
        or isinstance(distance, (str, bytes))
        or len(distance) != len(DISTANCE_GROUP_NAMES)
        or type(recalls) is not dict
        or tuple(sorted(recalls)) != PRESENT_CLASS_NAMES
    ):
        raise ValueError("physical distance or recall groups changed")
    margins.extend(
        (_finite(value, name=f"distance group {index}") - 0.92) / 0.92
        for index, value in enumerate(distance)
    )
    margins.extend(
        (_finite(recalls[name], name=f"{name} recall") - 0.95) / 0.95
        for name in sorted(recalls)
    )
    if len(margins) != MARGINS_PER_SCOPE:
        raise RuntimeError("V13 physical margin count per scope changed")
    return margins


def evaluate_physical_scopes_v13(scopes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if type(scopes) is not dict or tuple(scopes) != SCOPES:
        raise ValueError("physical scope order changed")
    rows: dict[str, Any] = {}
    flat: list[float] = []
    complete = 0
    for scope in SCOPES:
        margins = physical_margins_v13(scopes[scope])
        passed = all(value >= 0.0 for value in margins)
        rows[scope] = {"physical_margins": margins, "passes": passed}
        flat.extend(margins)
        complete += int(passed)
    if len(flat) != PHYSICAL_MARGIN_COUNT or any(not math.isfinite(value) for value in flat):
        raise PermissionError("V13 physical evaluator did not produce exactly 189 finite margins")
    rough = scopes["rough_local_dynamics"]
    return {
        "scope_evaluations": rows,
        "complete_physical_scope_count": complete,
        "margin_count": len(flat),
        "passed_margin_count": sum(value >= 0.0 for value in flat),
        "total_shortfall": sum(max(0.0, -value) for value in flat),
        "worst_margin": min(flat),
        "rough_motion": {
            "pixel_balanced_accuracy": _finite(
                rough["pixel_first_hit_balanced_accuracy"], name="rough pixel balanced accuracy"
            ),
            "ground_balanced_accuracy": _finite(
                rough["ground_clear_balanced_accuracy"], name="rough ground balanced accuracy"
            ),
            "depth_p95_m": _finite(rough["depth_p95_error_m"], name="rough depth p95"),
        },
    }


def registered_wrong_rgb_mapping_v13(
    endpoints: Sequence[Mapping[str, str]],
) -> dict[str, str]:
    """Return the exact within-family sorted cyclic one-step endpoint mapping."""

    by_family: dict[str, list[str]] = {}
    seen: set[str] = set()
    for row in endpoints:
        if type(row) is not dict or set(row) != {"endpoint_sha256", "family"}:
            raise ValueError("endpoint registration row changed")
        endpoint = row["endpoint_sha256"]
        family = row["family"]
        if not _is_sha256(endpoint) or not isinstance(family, str) or not family:
            raise ValueError("endpoint SHA-256 or family is malformed")
        if endpoint in seen:
            raise ValueError("endpoint registration is not unique")
        seen.add(endpoint)
        by_family.setdefault(family, []).append(endpoint)
    if len(seen) != 924:
        raise ValueError("registered endpoint count changed")
    if set(by_family) != set(REGISTERED_FAMILIES):
        raise ValueError("registered selection family set changed")
    mapping: dict[str, str] = {}
    for family in sorted(by_family):
        identifiers = sorted(by_family[family])
        if len(identifiers) < 2:
            raise ValueError(f"selection family has insufficient endpoints: {family}")
        rotated = identifiers[1:] + identifiers[:1]
        mapping.update(zip(identifiers, rotated, strict=True))
    return mapping


def _validate_physical_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "complete_physical_scope_count",
        "margin_count",
        "passed_margin_count",
        "total_shortfall",
        "worst_margin",
        "rough_motion",
    }
    if not isinstance(value, Mapping) or not required.issubset(value):
        raise ValueError("physical summary fields changed")
    complete = value["complete_physical_scope_count"]
    passed = value["passed_margin_count"]
    rough = value["rough_motion"]
    if (
        type(complete) is not int
        or not 0 <= complete <= len(SCOPES)
        or value["margin_count"] != PHYSICAL_MARGIN_COUNT
        or type(passed) is not int
        or not 0 <= passed <= PHYSICAL_MARGIN_COUNT
        or not isinstance(rough, Mapping)
        or set(rough) != {"pixel_balanced_accuracy", "ground_balanced_accuracy", "depth_p95_m"}
    ):
        raise ValueError("physical summary counts or rough schema changed")
    result = {
        "complete_physical_scope_count": complete,
        "margin_count": PHYSICAL_MARGIN_COUNT,
        "passed_margin_count": passed,
        "total_shortfall": _finite(value["total_shortfall"], name="total shortfall"),
        "worst_margin": _finite(value["worst_margin"], name="worst margin"),
        "rough_motion": {
            "pixel_balanced_accuracy": _finite(
                rough["pixel_balanced_accuracy"], name="rough pixel balanced accuracy"
            ),
            "ground_balanced_accuracy": _finite(
                rough["ground_balanced_accuracy"], name="rough ground balanced accuracy"
            ),
            "depth_p95_m": _finite(rough["depth_p95_m"], name="rough depth p95"),
        },
    }
    if result["total_shortfall"] < 0.0:
        raise ValueError("physical total shortfall cannot be negative")
    return result


def evaluate_update400_gate_v13(
    update100: Mapping[str, Any],
    update400: Mapping[str, Any],
    controls: Mapping[str, Mapping[str, bool]],
    *,
    integrity_pass: bool,
) -> dict[str, Any]:
    if type(integrity_pass) is not bool:
        raise TypeError("update-400 structural-integrity decision must be Boolean")
    before = _validate_physical_summary(update100)
    after = _validate_physical_summary(update400)
    if type(controls) is not dict or set(controls) != set(CONTROL_NAMES):
        raise ValueError("update-400 causal control set changed")
    control_checks: dict[str, bool] = {}
    for name in CONTROL_NAMES:
        row = controls[name]
        if type(row) is not dict or set(row) != set(CONTROL_CHECK_NAMES):
            raise ValueError(f"update-400 causal control schema changed: {name}")
        for check in CONTROL_CHECK_NAMES:
            if type(row[check]) is not bool:
                raise TypeError("causal control decisions must be Boolean")
            control_checks[f"{name}:{check}"] = row[check]
    rough_before = before["rough_motion"]
    rough_after = after["rough_motion"]
    rough_checks = {
        "rough_pixel_balanced_accuracy_strictly_increased":
            rough_after["pixel_balanced_accuracy"] > rough_before["pixel_balanced_accuracy"],
        "rough_ground_balanced_accuracy_strictly_increased":
            rough_after["ground_balanced_accuracy"] > rough_before["ground_balanced_accuracy"],
        "rough_depth_p95_strictly_decreased":
            rough_after["depth_p95_m"] < rough_before["depth_p95_m"],
    }
    checks = {
        "structural_integrity_pass": integrity_pass,
        "passed_physical_margin_count_strictly_increased":
            after["passed_margin_count"] > before["passed_margin_count"],
        "total_physical_shortfall_strictly_decreased":
            after["total_shortfall"] < before["total_shortfall"],
        "at_least_two_rough_metrics_strictly_improved":
            sum(rough_checks.values()) >= 2,
        "all_twelve_causal_control_checks_true": all(control_checks.values()),
    }
    passed = all(checks.values())
    return {
        "schema": f"{SCHEMA_PREFIX}_update400_directional_gate_v1",
        "checks": checks,
        "rough_direction_checks": rough_checks,
        "causal_control_checks": control_checks,
        "passed": passed,
        "action": "CONTINUE_TO_UPDATE_1000" if passed else "FAIL_TERMINAL_NO_RETRY_NO_RESUME",
        "next_update": 1_000 if passed else None,
        "retry_authorized": False,
        "resume_authorized": False,
    }


def evaluate_final_gate_v13(
    v12_gate: Mapping[str, Any],
    physical_summary: Mapping[str, Any],
    *,
    integrity_pass: bool,
) -> dict[str, Any]:
    if type(integrity_pass) is not bool:
        raise TypeError("final structural-integrity decision must be Boolean")
    if (
        not isinstance(v12_gate, Mapping)
        or type(v12_gate.get("passed")) is not bool
        or type(v12_gate.get("checks")) is not dict
        or tuple(v12_gate["checks"]) != V12_GATE_CHECK_NAMES
        or any(type(value) is not bool for value in v12_gate["checks"].values())
        or v12_gate["passed"] != all(v12_gate["checks"].values())
    ):
        raise ValueError("inherited V12 24-check gate changed or is inconsistent")
    physical = _validate_physical_summary(physical_summary)
    rough = physical["rough_motion"]
    checks = {
        "structural_integrity_pass": integrity_pass,
        "inherited_v12_full_arm_24_of_24": v12_gate["passed"],
        "passed_physical_margin_count_at_least_112":
            physical["passed_margin_count"]
            >= FINAL_PHYSICAL_THRESHOLDS["passed_margin_count_minimum"],
        "total_physical_shortfall_strictly_below_threshold":
            physical["total_shortfall"]
            < FINAL_PHYSICAL_THRESHOLDS["total_shortfall_strictly_less_than"],
        "complete_physical_scope_count_at_least_1":
            physical["complete_physical_scope_count"]
            >= FINAL_PHYSICAL_THRESHOLDS["complete_physical_scope_count_minimum"],
        "rough_pixel_balanced_accuracy_strictly_above_threshold":
            rough["pixel_balanced_accuracy"]
            > FINAL_PHYSICAL_THRESHOLDS[
                "rough_pixel_balanced_accuracy_strictly_greater_than"
            ],
        "rough_ground_balanced_accuracy_strictly_above_threshold":
            rough["ground_balanced_accuracy"]
            > FINAL_PHYSICAL_THRESHOLDS[
                "rough_ground_balanced_accuracy_strictly_greater_than"
            ],
        "rough_depth_p95_strictly_below_threshold":
            rough["depth_p95_m"]
            < FINAL_PHYSICAL_THRESHOLDS["rough_depth_p95_m_strictly_less_than"],
    }
    passed = all(checks.values())
    return {
        "schema": f"{SCHEMA_PREFIX}_final_development_gate_v1",
        "update": 1_000,
        "checks": checks,
        "passed": passed,
        "action": (
            "PASS_DEVELOPMENT_EARNS_PHYSICAL_ADAPTER_PREREGISTRATION_ONLY"
            if passed
            else "FAIL_TERMINAL_NO_RETRY_NO_RESUME"
        ),
        "physical_adapter_preregistration_eligible": passed,
        "probability_calibration_authorized": False,
        "g2_authorized": False,
        "navigation_authorized": False,
        "held_out_authorized": False,
        "retry_authorized": False,
        "resume_authorized": False,
    }


ACCOUNTING_MULTIPLIERS = {
    "updates": 1,
    "presentations": 16,
    "microbatch_graphs": 4,
    "backward_calls": 8,
    "camera_route_grad_calls": 4,
    "joint_route_grad_calls": 4,
    "camera_frame_objectives": 32,
    "optimizer_steps": 1,
    "ema_steps": 1,
    "predictor_forwards": 4,
    "predictor_objectives": 4,
}

TRACE_RELATIVE_PATH = "trace.jsonl"
METRIC_RELATIVE_PATHS = {
    update: f"metrics/update_{update}.json" for update in OBSERVATION_UPDATES
}
DEVELOPMENT_CHECKPOINT_RELATIVE_PATH = "checkpoints/update_1000.pt"
DEVELOPMENT_CHECKPOINT_BINDING_RELATIVE_PATH = (
    "checkpoints/update_1000.binding.json"
)
TERMINAL_ACCESS_RECEIPT_RELATIVE_PATH = "receipts/terminal_access.json"
SUCCESS_RELATIVE_PATH = "success.json"
SCIENTIFIC_FAILURE_RELATIVE_PATH = "failure.json"


def _canonical_value_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def validate_schedule_v13(
    schedule: Sequence[int],
    *,
    train_pair_count: int = 4_262,
) -> dict[str, Any]:
    if (
        not isinstance(schedule, Sequence)
        or isinstance(schedule, (str, bytes))
        or len(schedule) != MAXIMUM_PRESENTATIONS
        or type(train_pair_count) is not int
        or train_pair_count != 4_262
    ):
        raise PermissionError("V13 schedule length or train population changed")
    indices = list(schedule)
    if any(type(index) is not int or not 0 <= index < train_pair_count for index in indices):
        raise PermissionError("V13 schedule contains an invalid train-pair index")
    observed = {
        update: _canonical_value_sha256(indices[: update * PRESENTATIONS_PER_UPDATE])
        for update in (100, 400, 1_000)
    }
    if observed != CHECKPOINT_SCHEDULE_PREFIX_SHA256:
        raise PermissionError("V13 frozen schedule prefix SHA-256 changed")
    return {
        "presentation_count": len(indices),
        "train_pair_count": train_pair_count,
        "prefix_sha256": observed,
        "schedule_regeneration_count": 0,
    }


def validate_attempt_reservation_v13(value: Mapping[str, Any]) -> dict[str, Any]:
    reservation = validate_content_bound_v13(value)
    required = {
        "schema",
        "status",
        "created_utc",
        "authority_sha256",
        "one_shot_attempt",
        "attempt_consumed",
        "maximum_updates",
        "maximum_presentations",
        "retry_authorized",
        "resume_authorized",
        "content_sha256",
    }
    if (
        set(reservation) != required
        or reservation["schema"] != f"{SCHEMA_PREFIX}_attempt_reservation_v1"
        or reservation["status"] != "RESERVED_BEFORE_SCIENTIFIC_PAYLOAD"
        or not _is_sha256(reservation["authority_sha256"])
        or reservation["one_shot_attempt"] != 1
        or reservation["attempt_consumed"] is not True
        or reservation["maximum_updates"] != MAXIMUM_UPDATES
        or reservation["maximum_presentations"] != MAXIMUM_PRESENTATIONS
        or reservation["retry_authorized"] is not False
        or reservation["resume_authorized"] is not False
    ):
        raise PermissionError("V13 attempt reservation contract changed")
    return reservation


def _receipt_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        result = asdict(value)
    elif isinstance(value, Mapping):
        result = dict(value)
    else:
        raise TypeError(f"{name} must be a dataclass or mapping")
    return result


def _validate_access_receipt_v13(
    value: Mapping[str, Any],
    *,
    terminal: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("V13 access receipt must be a mapping")
    receipt = dict(value)
    required = {
        "forbidden_input_count",
        "probability_calibration_open_count",
        "opened_roles",
    }
    if not required.issubset(receipt):
        raise PermissionError("V13 access receipt is incomplete")
    roles = receipt["opened_roles"]
    if (
        type(receipt["forbidden_input_count"]) is not int
        or receipt["forbidden_input_count"] != 0
        or type(receipt["probability_calibration_open_count"]) is not int
        or receipt["probability_calibration_open_count"] != 0
        or not isinstance(roles, Sequence)
        or isinstance(roles, (str, bytes))
        or "probability_calibration" in roles
        or not set(roles).issubset(
            {"authority", "index", "train", "checkpoint_selection"}
        )
        or (
            terminal
            and not {"train", "checkpoint_selection"}.issubset(set(roles))
        )
    ):
        raise PermissionError("V13 access receipt records a forbidden role or input")
    if terminal:
        terminal_required = {
            "terminal_full_rehash_count",
            "raw_consumed_inputs_rehashed",
            "raw_consumed_file_rehash_count",
            "raw_consumed_records_sha256",
            "label_source_rehash_count",
            "label_sources_rehashed",
            "bound_parent_source_rehash_count",
            "bound_parent_sources",
            "certified_source_rehash_count",
            "certified_source_bindings_sha256",
            "certified_source_bindings",
            "all_consumed_inputs_rehashed",
            "source_root",
            "runtime_data_root",
            "runtime_fingerprint",
            "raw_inputs",
        }
        if not terminal_required.issubset(receipt):
            raise PermissionError("V13 terminal full-rehash receipt is incomplete")
        raw = receipt["raw_inputs"]
        bound_sources = receipt["bound_parent_sources"]
        certified_bindings = receipt["certified_source_bindings"]
        label_sources = receipt["label_sources_rehashed"]
        source_root = receipt["source_root"]
        data_root = receipt["runtime_data_root"]
        runtime_fingerprint = receipt["runtime_fingerprint"]
        roots_are_canonical = all(
            isinstance(root, str)
            and bool(root)
            and Path(root).is_absolute()
            and ".." not in Path(root).parts
            and str(Path(root)) == root
            and Path(root) != Path("/")
            for root in (source_root, data_root)
        )
        if (
            type(receipt["terminal_full_rehash_count"]) is not int
            or receipt["terminal_full_rehash_count"] != 1
            or receipt["raw_consumed_inputs_rehashed"] is not True
            or type(receipt["raw_consumed_file_rehash_count"]) is not int
            or receipt["raw_consumed_file_rehash_count"] <= 0
            or not _is_sha256(receipt["raw_consumed_records_sha256"])
            or type(receipt["label_source_rehash_count"]) is not int
            or receipt["label_source_rehash_count"] != 3
            or not isinstance(label_sources, Sequence)
            or isinstance(label_sources, (str, bytes))
            or len(label_sources) != 3
            or any(not isinstance(path, str) or not path for path in label_sources)
            or len(set(label_sources)) != 3
            or type(receipt["bound_parent_source_rehash_count"]) is not int
            or receipt["bound_parent_source_rehash_count"] <= 0
            or receipt["all_consumed_inputs_rehashed"] is not True
            or not roots_are_canonical
            or source_root == data_root
            or runtime_fingerprint != EXPECTED_RUNTIME_FINGERPRINT
            or type(raw) is not dict
            or type(raw.get("unique_file_count")) is not int
            or raw.get("unique_file_count")
            != receipt["raw_consumed_file_rehash_count"]
            or raw.get("records_sha256")
            != receipt["raw_consumed_records_sha256"]
            or raw.get("all_consumed_files_rehashed") is not True
            or not isinstance(bound_sources, Mapping)
            or bound_sources.get("validated_path_count")
            != receipt["bound_parent_source_rehash_count"]
            or bound_sources.get("execution_authority_granted") is not False
            or type(receipt["certified_source_rehash_count"]) is not int
            or receipt["certified_source_rehash_count"] <= 0
            or not _is_sha256(receipt["certified_source_bindings_sha256"])
            or type(certified_bindings) is not list
            or len(certified_bindings)
            != receipt["certified_source_rehash_count"]
            or len(certified_bindings)
            != len(
                {
                    binding.get("path")
                    for binding in certified_bindings
                    if type(binding) is dict
                }
            )
            or any(
                type(binding) is not dict
                or set(binding) != {"path", "file_sha256", "byte_count"}
                or not isinstance(binding["path"], str)
                or not binding["path"]
                or not _is_sha256(binding["file_sha256"])
                or type(binding["byte_count"]) is not int
                or binding["byte_count"] <= 0
                for binding in certified_bindings
            )
            or _canonical_value_sha256(certified_bindings)
            != receipt["certified_source_bindings_sha256"]
        ):
            raise PermissionError("V13 terminal full-rehash receipt is inconsistent")
    return receipt


def _derive_initial_structural_integrity_v13(runtime: Any, model: Any) -> dict[str, Any]:
    probe = runtime.structural_probe_inputs_v13()
    required = {
        "rgb",
        "wrong_rgb",
        "camera_origin_a",
        "camera_origin_b",
        "camera_basis",
        "ground_plane_z",
    }
    if type(probe) is not dict or set(probe) != required:
        raise PermissionError("V13 structural probe input schema changed")
    torch = runtime.torch
    if torch.equal(probe["rgb"], probe["wrong_rgb"]):
        raise PermissionError("V13 structural wrong-RGB probe is not distinct")
    first = model.encode_online_with_auxiliary_evidence(
        probe["rgb"],
        camera_origin_body_m=probe["camera_origin_a"],
        camera_basis_body_fru=probe["camera_basis"],
        ground_plane_z_body_m=probe["ground_plane_z"],
    )
    second = model.encode_online_with_auxiliary_evidence(
        probe["rgb"],
        camera_origin_body_m=probe["camera_origin_b"],
        camera_basis_body_fru=probe["camera_basis"],
        ground_plane_z_body_m=probe["ground_plane_z"],
    )
    nominal = model.encode_online_with_evidence(probe["rgb"])
    direct = model.encode_online(probe["rgb"])
    target = model.encode_target(probe["rgb"])
    wrong = model.encode_online_with_evidence(probe["wrong_rgb"])
    raw_fields = (
        "pixel_first_hit_hazard_logits",
        "pixel_within_bin_offset_m",
        "ground_clear_to_target_logits",
        "ground_query_in_frustum",
        "ground_query_uv_px",
        "ground_target_distance_m",
    )
    checks = {
        "auxiliary_geometry_was_changed": not torch.equal(
            probe["camera_origin_a"], probe["camera_origin_b"]
        ),
        "auxiliary_geometry_leaves_latent_bit_identical": torch.equal(
            first.latent, second.latent
        ),
        "auxiliary_geometry_leaves_nominal_evidence_bit_identical": all(
            torch.equal(
                getattr(first.nominal_evidence, field),
                getattr(second.nominal_evidence, field),
            )
            for field in raw_fields
        ),
        "public_nominal_path_matches_auxiliary_nominal_state": torch.equal(
            nominal.latent, first.latent
        ),
        "rgb_only_encode_matches_nominal_latent": torch.equal(direct, nominal.latent),
        "initial_target_matches_online_latent": torch.equal(target, nominal.latent),
        "target_is_finite": bool(torch.isfinite(target).all().item()),
        "target_is_noncollapsed": bool(
            target.numel() > 1
            and torch.std(target.float(), unbiased=False).item() > 0.0
        ),
        "wrong_rgb_changes_nominal_latent": not torch.equal(
            wrong.latent, nominal.latent
        ),
        "target_has_no_gradient": not target.requires_grad,
        "nominal_latent_is_finite": bool(torch.isfinite(nominal.latent).all().item()),
        "auxiliary_logits_not_used_for_state": torch.equal(
            first.latent, nominal.latent
        ),
        "old_raster_or_direct_token_bypass_absent": not any(
            fragment in name
            for name, _ in model.named_parameters()
            for fragment in (
                "soft_raster",
                "hard_raster",
                "direct_token",
                "deformable_lift",
            )
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"V13 initial structural integrity failed: {failed}")
    return {
        "schema": f"{SCHEMA_PREFIX}_initial_structural_integrity_v1",
        "checks": checks,
        "passed": True,
    }


def _validate_initialization_v13(
    runtime: Any,
    model: Any,
    initialization: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "n320_gate_open_count": 1,
        "n320_checkpoint_open_count": 1,
        "n320_gate_passed": True,
        "payload_access_after_reservation": True,
        "probability_calibration_open_count": 0,
        "constructor_initialization_seed": CONSTRUCTOR_INITIALIZATION_SEED,
        "projection_initialization_seed": PROJECTION_INITIALIZATION_SEED,
    }
    if not isinstance(initialization, Mapping) or any(
        initialization.get(name) != expected for name, expected in required.items()
    ):
        raise PermissionError("V13 N320/model initialization receipt changed")
    validate_model_api_v13(runtime.model_module)
    validate_training_api_v13(runtime.training_module)
    model_class = getattr(runtime.model_module, MODEL_CLASS_NAME)
    if not isinstance(model, model_class):
        raise TypeError("future V13 runtime returned the wrong model class")
    if (
        int(model.target_hard_sync_count.item()) != 1
        or int(model.ema_update_count.item()) != 0
    ):
        raise RuntimeError("V13 initial hard-sync/EMA accounting changed")
    groups = model.trainable_parameter_groups_v13()
    counts = tuple(sum(parameter.numel() for _, parameter in group) for group in groups)
    if counts != (
        MODEL_REQUIRED_CONSTANTS["SHARED_ROUTE_PARAMETER_COUNT_V13"],
        MODEL_REQUIRED_CONSTANTS["REPRESENTATION_GROUP_PARAMETER_COUNT_V13"],
        MODEL_REQUIRED_CONSTANTS["PREDICTOR_GROUP_PARAMETER_COUNT_V13"],
    ):
        raise RuntimeError("V13 initial online parameter partition changed")
    target_parameters = tuple(
        parameter
        for module in model.target_modules()
        for parameter in module.parameters()
    )
    if (
        sum(parameter.numel() for parameter in target_parameters)
        != MODEL_REQUIRED_CONSTANTS["TARGET_BOTTLENECK_PARAMETER_COUNT_V13"]
        or any(parameter.requires_grad or parameter.grad is not None for parameter in target_parameters)
        or any(module.training for module in model.target_modules())
    ):
        raise RuntimeError("V13 initial EMA target partition changed")
    return {**dict(initialization), "parameter_group_counts": counts}


def _validate_batch_query_identity_v13(model: Any, batch: Mapping[str, Any]) -> None:
    import torch

    for prefix in ("current", "next"):
        query = model.bev_lift.evidence_head.ground_query_geometry(
            batch[f"{prefix}_camera_origin_body_m"],
            batch[f"{prefix}_camera_basis_body_fru"],
            batch[f"{prefix}_ground_plane_z_body_m"],
        )
        target = batch[f"{prefix}_ground_support_in_frustum"]
        if not torch.equal(query.in_frustum, target):
            raise PermissionError(
                f"V13 {prefix} auxiliary query visibility differs from supervision"
            )


def _validate_microbatches_for_engine_v13(
    runtime: Any,
    model: Any,
    microbatches: Sequence[Mapping[str, Any]],
) -> None:
    if len(microbatches) != MICROBATCHES_PER_UPDATE:
        raise PermissionError("V13 engine did not receive exactly four microbatches")
    for batch in microbatches:
        if type(batch) is not dict or set(batch) != set(TRAINING_REQUIRED_BATCH_KEYS):
            raise PermissionError("V13 engine microbatch schema changed")
        _validate_batch_query_identity_v13(model, batch)
    validator = getattr(runtime.training_module, "_validate_microbatches_v13", None)
    if not callable(validator):
        raise RuntimeError("V13 training microbatch validator is absent")
    validator(runtime.torch, microbatches)


def _validate_update_integrity_v13(
    runtime: Any,
    model: Any,
    result: Any,
    *,
    update: int,
    access_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if type(update) is not int or not 1 <= update <= MAXIMUM_UPDATES:
        raise ValueError("V13 update integrity index escaped the cap")
    accounting = _receipt_mapping(result.accounting, name="V13 accounting")
    expected_accounting = {
        name: update * multiplier for name, multiplier in ACCOUNTING_MULTIPLIERS.items()
    }
    if accounting != expected_accounting:
        raise RuntimeError("V13 per-update accounting changed")
    routes = result.gradient_routes
    if not isinstance(routes, Mapping) or set(routes) != {
        "camera_shared",
        "joint_shared",
        "representation",
        "predictor",
    }:
        raise RuntimeError("V13 gradient-route receipt set changed")
    route_receipts: dict[str, Any] = {}
    for name in ("camera_shared", "joint_shared", "representation", "predictor"):
        receipt = _receipt_mapping(routes[name], name=f"{name} gradient receipt")
        if set(receipt) != {
            "preclip_l2",
            "applied_scale",
            "parameter_tensor_count",
            "absent_tensor_gradient_count",
        }:
            raise RuntimeError(f"V13 gradient-route fields changed: {name}")
        norm = _finite(receipt["preclip_l2"], name=f"{name} preclip norm")
        scale = _finite(receipt["applied_scale"], name=f"{name} scale")
        expected_scale = float(
            runtime.torch.minimum(
                runtime.torch.tensor(1.0, dtype=runtime.torch.float32),
                runtime.torch.reciprocal(
                    runtime.torch.maximum(
                        runtime.torch.tensor(norm, dtype=runtime.torch.float32),
                        runtime.torch.tensor(
                            runtime.torch.finfo(runtime.torch.float32).tiny,
                            dtype=runtime.torch.float32,
                        ),
                    )
                ),
            ).item()
        )
        if (
            scale != expected_scale
            or type(receipt["parameter_tensor_count"]) is not int
            or receipt["parameter_tensor_count"] <= 0
            or type(receipt["absent_tensor_gradient_count"]) is not int
            or receipt["absent_tensor_gradient_count"] < 0
            or (name in ("camera_shared", "joint_shared", "predictor") and norm <= 0.0)
        ):
            raise RuntimeError(f"V13 gradient-route integrity failed: {name}")
        route_receipts[name] = receipt
    losses = dict(result.mean_losses)
    if set(losses) != {"S", "P", "U", "R", "O", "N", "C", "L"}:
        raise RuntimeError("V13 per-update mean-loss receipt set changed")
    losses = {name: _finite(value, name=f"mean loss {name}") for name, value in losses.items()}
    if not math.isclose(
        losses["N"],
        sum(losses[name] for name in ("S", "P", "U", "R", "O")),
        rel_tol=2e-6,
        abs_tol=2e-6,
    ) or not math.isclose(
        losses["L"],
        losses["N"] + losses["C"],
        rel_tol=2e-6,
        abs_tol=2e-6,
    ):
        raise RuntimeError("V13 N/L loss equations changed")
    count_fields = {
        "ranking_active_microbatches": result.ranking_active_microbatches,
        "ranking_eligible_pairs": result.ranking_eligible_pairs,
        "survival_supervised_decisions": result.survival_supervised_decisions,
    }
    if any(type(value) is not int or value < 0 for value in count_fields.values()) or (
        count_fields["survival_supervised_decisions"] <= 0
    ):
        raise RuntimeError("V13 ranking/survival accounting changed")
    if (
        result.target_gradient_tensor_count != 0
        or result.optimizer_steps_this_update != 1
        or result.ema_steps_this_update != 1
        or int(model.target_hard_sync_count.item()) != 1
        or int(model.ema_update_count.item()) != update
        or any(parameter.grad is not None for module in model.target_modules() for parameter in module.parameters())
        or any(module.training for module in model.target_modules())
    ):
        raise RuntimeError("V13 target/optimizer/EMA structural integrity failed")
    for value in model.state_dict().values():
        if hasattr(value, "is_floating_point") and value.is_floating_point() and not bool(
            runtime.torch.isfinite(value).all().item()
        ):
            raise FloatingPointError("V13 model state became nonfinite")
    access = _validate_access_receipt_v13(access_receipt)
    return {
        "update": update,
        "accounting": accounting,
        "gradient_routes": route_receipts,
        "mean_losses": losses,
        **count_fields,
        "target_gradient_tensor_count": 0,
        "optimizer_steps_this_update": 1,
        "ema_steps_this_update": 1,
        "hard_sync_count": 1,
        "ema_count": update,
        "access_receipt_sha256": _canonical_value_sha256(access),
        "passed": True,
    }


def _publisher_json_v13(
    publisher: Any,
    relative_path: str,
    core: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = publisher.publish_json(relative_path, dict(core))
    if type(result) is not dict or set(result) != {"value", "binding"}:
        raise RuntimeError("V13 JSON publisher result changed")
    value = validate_content_bound_v13(result["value"])
    binding = dict(result["binding"])
    raw = _canonical_json_bytes(value) + b"\n"
    if (
        binding.get("path") != relative_path
        or binding.get("file_sha256") != hashlib.sha256(raw).hexdigest()
        or binding.get("byte_count") != len(raw)
    ):
        raise RuntimeError("V13 JSON publisher binding changed")
    return value, binding


def _publisher_bytes_v13(
    publisher: Any,
    relative_path: str,
    raw: bytes,
) -> dict[str, Any]:
    binding = publisher.publish_bytes(relative_path, raw)
    if (
        type(binding) is not dict
        or binding.get("path") != relative_path
        or binding.get("file_sha256") != hashlib.sha256(raw).hexdigest()
        or binding.get("byte_count") != len(raw)
    ):
        raise RuntimeError("V13 binary publisher binding changed")
    return dict(binding)


def _observation_v13(
    runtime: Any,
    model: Any,
    *,
    update: int,
    integrity_pass: bool,
) -> dict[str, Any]:
    probe = runtime.structural_probe_inputs_v13()
    probe_rgb = probe.get("rgb") if isinstance(probe, Mapping) else None
    if not isinstance(probe_rgb, runtime.torch.Tensor):
        raise PermissionError("V13 fixed structural RGB probe is absent")
    with runtime.torch.no_grad():
        target = model.encode_target(probe_rgb)
    if not isinstance(target, runtime.torch.Tensor):
        raise RuntimeError("V13 EMA target probe did not return a tensor")
    target_checks = {
        "target_is_finite": bool(runtime.torch.isfinite(target).all().item()),
        "target_is_noncollapsed": bool(
            target.numel() > 1
            and runtime.torch.std(
                target.float(), unbiased=False
            ).item() > 0.0
        ),
        "target_has_no_gradient": not target.requires_grad,
        "target_modules_are_frozen": all(
            not parameter.requires_grad and parameter.grad is None
            for module in model.target_modules()
            for parameter in module.parameters()
        ),
        "target_modules_are_eval": all(
            not module.training for module in model.target_modules()
        ),
    }
    target_integrity = {
        "schema": f"{SCHEMA_PREFIX}_observation_target_integrity_v1",
        "update": update,
        "checks": target_checks,
        "passed": all(target_checks.values()),
    }
    combined_integrity_pass = integrity_pass and target_integrity["passed"]
    observed = runtime.observe_v13(
        model,
        update=update,
        physical_endpoint_updater=update_physical_accumulator_from_rgb_v13,
    )
    if type(observed) is not dict or not {
        "physical_scopes",
        "v12_gate",
        "controls",
        "physical_provenance",
    }.issubset(observed):
        raise RuntimeError("V13 observation result schema changed")
    provenance = observed["physical_provenance"]
    expected_provenance = {
        "target_endpoint_count": 924,
        "matched_nominal_call_count": 924,
        "wrong_nominal_call_count": 924,
        "qualifying_updater_call_count": 1_848,
        "qualifying_updater_name": "update_physical_accumulator_from_rgb_v13",
        "auxiliary_logits_used": False,
        "old_camera_raster_used": False,
        "target_query_identity_pass": True,
        "wrong_rgb_dependence_nonzero": True,
    }
    if provenance != expected_provenance:
        raise PermissionError("V13 physical evaluator provenance receipt changed")
    aggregate = observed["physical_scopes"].get("aggregate")
    wrong_names = (
        "wrong_rgb_pixel_balanced_accuracy_drop",
        "wrong_rgb_depth_median_error_increase_m",
        "wrong_rgb_depth_p95_error_increase_m",
        "wrong_rgb_ground_balanced_accuracy_drop",
        "wrong_rgb_raster_nll_increase",
        "wrong_rgb_raster_balanced_accuracy_drop",
    )
    if not isinstance(aggregate, Mapping) or not any(
        abs(_finite(aggregate.get(name), name=name)) > 0.0 for name in wrong_names
    ):
        raise PermissionError("V13 physical observation has zero wrong-RGB dependence")
    physical = evaluate_physical_scopes_v13(observed["physical_scopes"])
    return {
        "schema": f"{SCHEMA_PREFIX}_observation_v1",
        "update": update,
        "physical": physical,
        "v12_gate": observed["v12_gate"],
        "controls": observed["controls"],
        "physical_provenance": dict(provenance),
        "target_integrity": target_integrity,
        "integrity_pass": combined_integrity_pass,
        "probability_calibration_opened": False,
        "state_mutation_count": 0,
    }


def _serialize_development_checkpoint_v13(
    runtime: Any,
    model: Any,
    authority: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    state = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in model.state_dict().items()
    }
    state_manifest = []
    for name, value in state.items():
        raw_tensor = value.numpy().tobytes(order="C")
        state_manifest.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
                "numel": int(value.numel()),
                "tensor_sha256": hashlib.sha256(raw_tensor).hexdigest(),
            }
        )
    groups = model.trainable_parameter_groups_v13()
    online_counts = {
        name: sum(parameter.numel() for _, parameter in group)
        for name, group in zip(
            ("shared", "representation", "predictor"), groups, strict=True
        )
    }
    config = model.config
    if not is_dataclass(config):
        raise RuntimeError("V13 model config is not the frozen dataclass")
    metadata = {
        "schema": f"{SCHEMA_PREFIX}_development_checkpoint_binding_v1",
        "update": 1_000,
        "model_module": type(model).__module__,
        "model_class": type(model).__name__,
        "model_config": asdict(config),
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "frozen_source_and_review_commit": authority[
            "frozen_source_and_review_commit"
        ],
        "recursive_source_closure_manifest_sha256": authority[
            "recursive_source_closure_manifest_sha256"
        ],
        "execution_binding_commit": authority["execution_binding_commit"],
        "authority_sha256": hashlib.sha256(
            _canonical_json_bytes(authority)
        ).hexdigest(),
        "state_manifest": state_manifest,
        "state_manifest_sha256": _canonical_value_sha256(state_manifest),
        "state_key_count": len(state),
        "online_parameter_counts": online_counts,
        "target_parameter_count": sum(
            parameter.numel()
            for module in model.target_modules()
            for parameter in module.parameters()
        ),
        "promotable_state": "update_1000_only",
        "development_pass_required": True,
        "probability_calibration_used": False,
    }
    payload = {
        "schema": f"{SCHEMA_PREFIX}_development_checkpoint_v1",
        "update": 1_000,
        "model_state_dict": state,
        "metadata": metadata,
        "probability_calibration_used": False,
    }
    stream = io.BytesIO()
    runtime.torch.save(payload, stream)
    raw = stream.getvalue()
    if not raw:
        raise RuntimeError("V13 development checkpoint serialization is empty")
    return raw, metadata


def run_future_authorized_engine_v13(
    *,
    authority: Mapping[str, Any],
    reservation: Mapping[str, Any],
    runtime: Any,
    publisher: Any,
) -> dict[str, Any]:
    """Run the frozen one-shot lifecycle after a custodian reservation.

    The public launcher remains denied until real authority artifacts exist.
    This engine assumes the reviewed launcher composed ``runtime`` only after
    publishing ``reservation`` and never performs pre-reservation payload I/O.
    """

    stage = "validate_post_reservation_authority"
    trace: list[dict[str, Any]] = []
    metric_bindings: list[dict[str, Any]] = []
    trace_binding: dict[str, Any] | None = None
    terminal_access_binding: dict[str, Any] | None = None
    terminal_access_content_sha256: str | None = None
    terminal_published = False
    validated_authority: dict[str, Any] | None = None

    def publish_trace() -> dict[str, Any]:
        nonlocal trace_binding
        if trace_binding is None:
            raw = b"".join(
                _canonical_json_bytes(_content_bound(row)) + b"\n" for row in trace
            )
            trace_binding = _publisher_bytes_v13(
                publisher, TRACE_RELATIVE_PATH, raw
            )
        return trace_binding

    def publish_terminal_access(receipt: Mapping[str, Any]) -> dict[str, Any]:
        nonlocal terminal_access_binding, terminal_access_content_sha256
        if terminal_access_binding is None:
            value, terminal_access_binding = _publisher_json_v13(
                publisher,
                TERMINAL_ACCESS_RECEIPT_RELATIVE_PATH,
                {
                    "schema": f"{SCHEMA_PREFIX}_terminal_access_receipt_v1",
                    "receipt": dict(receipt),
                },
            )
            terminal_access_content_sha256 = value["content_sha256"]
        return terminal_access_binding

    try:
        validated_authority = validate_future_execution_prerequisites_v13(authority)
        validated_reservation = validate_attempt_reservation_v13(reservation)
        if validated_reservation["authority_sha256"] != hashlib.sha256(
            _canonical_json_bytes(validated_authority)
        ).hexdigest():
            raise PermissionError("V13 reservation does not bind supplied authority")

        stage = "validate_deferred_runtime_and_schedule"
        schedule_receipt = validate_schedule_v13(
            runtime.schedule,
            train_pair_count=int(runtime.train_pair_count),
        )
        stage = "initialize_n320_v13_model_optimizer"
        model, optimizer, initialization = runtime.initialize_model_v13()
        initialization_receipt = _validate_initialization_v13(
            runtime, model, initialization
        )
        initial_structural = _derive_initial_structural_integrity_v13(runtime, model)
        access = _validate_access_receipt_v13(runtime.access_receipt_v13())
        trace.append(
            {
                "schema": f"{SCHEMA_PREFIX}_trace_row_v1",
                "event": "initialized",
                "update": 0,
                "initialization": initialization_receipt,
                "structural_integrity": initial_structural,
                "schedule": schedule_receipt,
                "access_receipt_sha256": _canonical_value_sha256(access),
            }
        )

        stage = "observe_update_0"
        observations: dict[int, dict[str, Any]] = {}
        observations[0] = _observation_v13(
            runtime,
            model,
            update=0,
            integrity_pass=bool(initial_structural["passed"]),
        )
        _, binding = _publisher_json_v13(
            publisher, METRIC_RELATIVE_PATHS[0], observations[0]
        )
        metric_bindings.append(binding)

        accounting: Any = None
        structural_pass = bool(observations[0]["integrity_pass"])
        terminal_update: int | None = None
        scientific_decision: dict[str, Any] | None = None
        for update in range(1, MAXIMUM_UPDATES + 1):
            stage = f"train_update_{update}"
            start = (update - 1) * PRESENTATIONS_PER_UPDATE
            indices = list(runtime.schedule[start : start + PRESENTATIONS_PER_UPDATE])
            if len(indices) != PRESENTATIONS_PER_UPDATE:
                raise PermissionError("V13 frozen schedule ended early")
            microbatches = runtime.build_microbatches_v13(indices, update=update)
            _validate_microbatches_for_engine_v13(runtime, model, microbatches)
            result = runtime.training_module.joint_training_update_v13(
                model,
                optimizer,
                microbatches,
                accounting=accounting,
            )
            accounting = result.accounting
            integrity = _validate_update_integrity_v13(
                runtime,
                model,
                result,
                update=update,
                access_receipt=runtime.access_receipt_v13(),
            )
            structural_pass = structural_pass and bool(integrity["passed"])
            trace.append(
                {
                    "schema": f"{SCHEMA_PREFIX}_trace_row_v1",
                    "event": "optimizer_ema_update",
                    **integrity,
                }
            )

            if update not in (100, 400, 1_000):
                continue
            stage = f"observe_update_{update}"
            observations[update] = _observation_v13(
                runtime,
                model,
                update=update,
                integrity_pass=structural_pass,
            )
            structural_pass = bool(observations[update]["integrity_pass"])
            _, binding = _publisher_json_v13(
                publisher,
                METRIC_RELATIVE_PATHS[update],
                observations[update],
            )
            metric_bindings.append(binding)
            if update == 400:
                controls = observations[400]["controls"]
                scientific_decision = evaluate_update400_gate_v13(
                    observations[100]["physical"],
                    observations[400]["physical"],
                    controls,
                    integrity_pass=structural_pass,
                )
                trace.append(
                    {
                        "schema": f"{SCHEMA_PREFIX}_trace_row_v1",
                        "event": "update400_control",
                        "update": 400,
                        "decision": scientific_decision,
                    }
                )
                if not scientific_decision["passed"]:
                    terminal_update = 400
                    break
            elif update == 1_000:
                scientific_decision = evaluate_final_gate_v13(
                    observations[1_000]["v12_gate"],
                    observations[1_000]["physical"],
                    integrity_pass=structural_pass,
                )
                terminal_update = 1_000
                trace.append(
                    {
                        "schema": f"{SCHEMA_PREFIX}_trace_row_v1",
                        "event": "update1000_final_gate",
                        "update": 1_000,
                        "decision": scientific_decision,
                    }
                )

        if terminal_update not in TERMINAL_UPDATES or scientific_decision is None:
            raise RuntimeError("V13 engine did not reach one frozen terminal update")
        terminal_accounting = validate_terminal_accounting_v13(
            accounting, terminal_update=terminal_update
        )
        terminal_access_reader = getattr(
            runtime, "terminal_access_receipt_v13", None
        )
        final_access = _validate_access_receipt_v13(
            (
                terminal_access_reader()
                if callable(terminal_access_reader)
                else runtime.access_receipt_v13()
            ),
            terminal=True,
        )
        if final_access["runtime_data_root"] != validated_authority["runtime_data_root"]:
            raise PermissionError(
                "V13 terminal rehash used a different authority-bound runtime data root"
            )
        if final_access["source_root"] != validated_authority["certified_source_root"]:
            raise PermissionError(
                "V13 terminal rehash used a different certified source root"
            )
        if final_access["runtime_fingerprint"] != validated_authority["runtime"]:
            raise PermissionError("V13 terminal rehash used a different runtime stack")
        final_access_artifact = publish_terminal_access(final_access)
        trace_record = publish_trace()

        if not scientific_decision["passed"]:
            stage = "publish_terminal_scientific_failure"
            if callable(getattr(runtime, "close_v13", None)):
                runtime.close_v13()
            failure_core = {
                "schema": f"{SCHEMA_PREFIX}_scientific_failure_v1",
                "status": (
                    "FAIL_SCIENTIFIC_UPDATE400_GATE_TERMINAL"
                    if terminal_update == 400
                    else "FAIL_SCIENTIFIC_UPDATE1000_GATE_TERMINAL"
                ),
                "terminal_update": terminal_update,
                "decision": scientific_decision,
                "accounting": terminal_accounting,
                "metrics": metric_bindings,
                "trace": trace_record,
                "access_receipt_sha256": _canonical_value_sha256(final_access),
                "terminal_access_receipt": final_access_artifact,
                "terminal_access_receipt_content_sha256": (
                    terminal_access_content_sha256
                ),
                "checkpoint_published": False,
                "probability_calibration_opened": False,
                "attempt_consumed": True,
                "retry_authorized": False,
                "resume_authorized": False,
            }
            value, _ = _publisher_json_v13(
                publisher, SCIENTIFIC_FAILURE_RELATIVE_PATH, failure_core
            )
            terminal_published = True
            return value

        stage = "publish_pass1000_development_checkpoint"
        checkpoint_raw, checkpoint_core = _serialize_development_checkpoint_v13(
            runtime, model, validated_authority
        )
        checkpoint_binding = _publisher_bytes_v13(
            publisher,
            DEVELOPMENT_CHECKPOINT_RELATIVE_PATH,
            checkpoint_raw,
        )
        checkpoint_core["checkpoint"] = checkpoint_binding
        checkpoint_value, checkpoint_metadata_binding = _publisher_json_v13(
            publisher,
            DEVELOPMENT_CHECKPOINT_BINDING_RELATIVE_PATH,
            checkpoint_core,
        )
        if callable(getattr(runtime, "close_v13", None)):
            runtime.close_v13()
        stage = "publish_terminal_success"
        success_core = {
            "schema": f"{SCHEMA_PREFIX}_success_v1",
            "status": "PASS_DEVELOPMENT_UPDATE1000_TERMINAL",
            "terminal_update": 1_000,
            "decision": scientific_decision,
            "accounting": terminal_accounting,
            "metrics": metric_bindings,
            "trace": trace_record,
            "checkpoint": checkpoint_binding,
            "checkpoint_metadata": checkpoint_metadata_binding,
            "checkpoint_metadata_content_sha256": checkpoint_value[
                "content_sha256"
            ],
            "access_receipt_sha256": _canonical_value_sha256(final_access),
            "terminal_access_receipt": final_access_artifact,
            "terminal_access_receipt_content_sha256": (
                terminal_access_content_sha256
            ),
            "physical_adapter_preregistration_eligible": True,
            "probability_calibration_authorized": False,
            "probability_calibration_opened": False,
            "g2_authorized": False,
            "navigation_authorized": False,
            "held_out_authorized": False,
            "attempt_consumed": True,
            "retry_authorized": False,
            "resume_authorized": False,
        }
        value, _ = _publisher_json_v13(
            publisher, SUCCESS_RELATIVE_PATH, success_core
        )
        terminal_published = True
        return value
    except BaseException as error:
        if terminal_published:
            raise
        try:
            if callable(getattr(runtime, "close_v13", None)):
                runtime.close_v13()
            terminal_reader = getattr(runtime, "terminal_access_receipt_v13", None)
            exception_access = _validate_access_receipt_v13(
                (
                    terminal_reader()
                    if callable(terminal_reader)
                    else runtime.access_receipt_v13()
                ),
                terminal=True,
            )
            if validated_authority is not None and (
                exception_access["runtime_data_root"]
                != validated_authority["runtime_data_root"]
                or exception_access["source_root"]
                != validated_authority["certified_source_root"]
                or exception_access["runtime_fingerprint"]
                != validated_authority["runtime"]
            ):
                raise PermissionError(
                    "V13 exception access receipt used an unbound source or data root"
                )
            exception_access_artifact = publish_terminal_access(exception_access)
            trace_record = publish_trace()
            failure_core = {
                "schema": f"{SCHEMA_PREFIX}_exception_failure_v1",
                "status": "FAIL_EXCEPTION_TERMINAL_NO_RETRY_NO_RESUME",
                "stage": stage,
                "exception_type": type(error).__name__,
                "exception_message_sha256": hashlib.sha256(
                    str(error).encode("utf-8")
                ).hexdigest(),
                "trace": trace_record,
                "access_receipt_sha256": _canonical_value_sha256(
                    exception_access
                ),
                "terminal_access_receipt": exception_access_artifact,
                "terminal_access_receipt_content_sha256": (
                    terminal_access_content_sha256
                ),
                "checkpoint_published": False,
                "probability_calibration_opened": False,
                "attempt_consumed": True,
                "retry_authorized": False,
                "resume_authorized": False,
            }
            value, _ = _publisher_json_v13(
                publisher, SCIENTIFIC_FAILURE_RELATIVE_PATH, failure_core
            )
            return value
        except BaseException:
            raise error


def validate_terminal_accounting_v13(accounting: Any, *, terminal_update: int) -> dict[str, int]:
    if terminal_update not in TERMINAL_UPDATES:
        raise ValueError("V13 terminal update must be exactly 400 or 1000")
    if is_dataclass(accounting) and not isinstance(accounting, type):
        value = asdict(accounting)
    elif type(accounting) is dict:
        value = dict(accounting)
    else:
        raise TypeError("V13 accounting must be a dataclass instance or plain dict")
    if set(value) != set(ACCOUNTING_MULTIPLIERS):
        raise ValueError("V13 terminal accounting fields changed")
    expected = {
        name: terminal_update * multiplier
        for name, multiplier in ACCOUNTING_MULTIPLIERS.items()
    }
    if any(type(value[name]) is not int for name in value) or value != expected:
        raise RuntimeError("V13 terminal accounting is inconsistent with the frozen cap")
    if value["updates"] > MAXIMUM_UPDATES or value["presentations"] > MAXIMUM_PRESENTATIONS:
        raise PermissionError("V13 scientific cap was exceeded")
    return expected


def execute_v13(*_args: Any, **_kwargs: Any) -> None:
    """Fail before root creation, scientific I/O, module loading, or GPU query."""

    raise PermissionError(CURRENT_EXECUTION_DENIAL)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments:
        raise ValueError("the denied V13 source shell accepts no arguments")
    print(json.dumps(execution_denial_receipt_v13(), sort_keys=True))
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
