"""Source-only contract for the causal motion-alignment V1 falsification.

This module is deliberately a narrow successor adapter.  It loads the frozen
causal-temporal V1 standard-library contract under a private module identity,
rebinds that contract to the new source/science identities, and changes only
the runtime-open boundary needed to admit the already-authorized development
RGB leaves.  Importing it reads no generated input, RGB, checkpoint, tensor,
or accelerator state.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
TEMPORAL_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_rgb_causal_temporal_perception_v1.py"
)
TEMPORAL_CONTRACT_FILE_SHA256 = (
    "ba3fd9cda5c1d3d4b3383b192bfb3ccafa6e5bd08e581c0e1d147c34d0c9e949"
)
TEMPORAL_RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_rgb_causal_temporal_perception_v1.py"
)
TEMPORAL_RUNNER_FILE_SHA256 = (
    "941db26b14a956aac89b0d762e64448e6efdbf1ca1a4d79741eb305d9096200b"
)
TEMPORAL_LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_rgb_causal_temporal_perception_v1.py"
)
TEMPORAL_LAUNCHER_FILE_SHA256 = (
    "c381e9d9158bc4d9559566b63e9dcc9f7b860c6be5884ad56ad6bcc0b90482f6"
)


def _load_temporal_contract() -> Any:
    source = ROOT / TEMPORAL_CONTRACT_RELATIVE_PATH
    spec = importlib.util.spec_from_file_location(
        "_lewm_motion_alignment_v1_frozen_temporal_contract",
        source,
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen temporal perception contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TEMPORAL = _load_temporal_contract()
_STRUCTURAL_TEMPORAL = _load_temporal_contract()
_BASE_SCIENCE_CONTRACT = _TEMPORAL.science_contract
_BASE_LIFECYCLE_CONTRACT = _TEMPORAL.lifecycle_contract
_BASE_VALIDATE_RUNTIME_INPUTS = _TEMPORAL.validate_runtime_inputs
_BASE_PARSE_PARTIAL_ACCESS_LEDGER = _TEMPORAL.parse_partial_access_ledger
_BASE_SOURCE_FREEZE_STATUS = _TEMPORAL.source_freeze_status
_BASE_CURRENT_SOURCE_BINDINGS = _TEMPORAL.current_source_bindings
_BASE_VALIDATE_METRIC_SIDECAR = _TEMPORAL.validate_metric_sidecar
_BASE_VALIDATE_FAILURE_RECEIPT = _TEMPORAL.validate_failure_receipt
_BASE_VALIDATE_PRE_LEDGER_FAILURE_RECEIPT = (
    _TEMPORAL.validate_pre_ledger_failure_receipt
)
_STRUCTURAL_PARSE_PARTIAL_ACCESS_LEDGER = (
    _STRUCTURAL_TEMPORAL.parse_partial_access_ledger
)
_STRUCTURAL_VALIDATE_FAILURE_RECEIPT = (
    _STRUCTURAL_TEMPORAL.validate_failure_receipt
)
TEMPORAL_MODEL_RELATIVE_PATH = _TEMPORAL.MODEL_RELATIVE_PATH
TEMPORAL_MODEL_FILE_SHA256 = (
    "2bc62999ae6aa2f1d52275dc5f25edcf9755cc4b15b7d786bd06c5817673e19d"
)
TEMPORAL_SOURCE_MANIFEST_RELATIVE_PATH = (
    _TEMPORAL.SOURCE_MANIFEST_RELATIVE_PATH
)
TEMPORAL_SOURCE_MANIFEST_FILE_SHA256 = (
    "3c4e82a6e0b24a30ee7c8a6d7564953f14e6611e6e87afe5f7c458969d2c9dc3"
)
TEMPORAL_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "486367e67b411d08575db0e63982c2f0b5b4b525bf63daf22927b4364f62e518"
)
TEMPORAL_SOURCE_MANIFEST_BYTE_COUNT = 12_294

# Start from the frozen standard-library surface, then replace only successor
# identities and science constants below.
for _name in dir(_TEMPORAL):
    if _name.isupper():
        globals()[_name] = getattr(_TEMPORAL, _name)

canonical_json_bytes = _TEMPORAL.canonical_json_bytes
canonical_json_sha256 = _TEMPORAL.canonical_json_sha256
is_sha256 = _TEMPORAL.is_sha256
with_content_sha256 = _TEMPORAL.with_content_sha256
parse_canonical_json = _TEMPORAL.parse_canonical_json
safe_relative_path = _TEMPORAL.safe_relative_path
artifact_binding = _TEMPORAL.artifact_binding
validate_binding = _TEMPORAL.validate_binding


def _structural_expected_open_binding(value: object) -> dict[str, Any]:
    """Validate binding shape/path syntax without scientific admission."""

    if (
        type(value) is not dict
        or set(value)
        != {"path", "file_sha256", "content_sha256", "byte_count"}
        or not is_sha256(value["file_sha256"])
        or (
            value["content_sha256"] is not None
            and not is_sha256(value["content_sha256"])
        )
        or (
            value["byte_count"] is not None
            and (
                type(value["byte_count"]) is not int
                or value["byte_count"] <= 0
            )
        )
    ):
        raise PermissionError("structural attempted runtime binding changed")
    safe_relative_path(value["path"], name="structural attempted runtime input")
    return dict(value)


_STRUCTURAL_TEMPORAL._validated_expected_open_binding = (
    _structural_expected_open_binding
)


CONTRACT_AUTHOR = "/root/execution_readiness_audit"
SCHEMA_PREFIX = "lewm_go2_rgb_causal_motion_alignment_v1"

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_rgb_causal_motion_alignment_v1.py"
)
MODEL_RELATIVE_PATH = (
    "lewm/models/"
    "shared_observable_camera_ray_jepa_v5_multires_motion_alignment_v1.py"
)
RUNNER_RELATIVE_PATH = "scripts/run_go2_rgb_causal_motion_alignment_v1.py"
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_rgb_causal_motion_alignment_v1.py"
)
CONTRACT_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_rgb_causal_motion_alignment_v1_contract.py"
)
MODEL_TEST_RELATIVE_PATH = (
    "lewm/tests/"
    "test_shared_observable_camera_ray_jepa_v5_multires_motion_alignment_v1.py"
)
RUNNER_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_rgb_causal_motion_alignment_v1_runner.py"
)
EVALUATOR_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_rgb_causal_motion_alignment_v1_evaluator.py"
)
RECEIPT_BOUNDARY_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_rgb_causal_motion_alignment_v1_receipt_boundary.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_rgb_causal_motion_alignment_v1_source_closure.py"
)
SOURCE_CLOSURE_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_rgb_causal_motion_alignment_v1_source_closure.py"
)
SOURCE_CLOSURE_BASE_CHECKER_RELATIVE_PATH = (
    _TEMPORAL.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH
)
TEST_RELATIVE_PATH = RUNNER_TEST_RELATIVE_PATH

ADDITIVE_SOURCE_PATHS = (
    CONTRACT_RELATIVE_PATH,
    MODEL_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    CONTRACT_TEST_RELATIVE_PATH,
    MODEL_TEST_RELATIVE_PATH,
    RUNNER_TEST_RELATIVE_PATH,
    EVALUATOR_TEST_RELATIVE_PATH,
    RECEIPT_BOUNDARY_TEST_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    SOURCE_CLOSURE_TEST_RELATIVE_PATH,
)
REUSED_SOURCE_PATHS = (
    TEMPORAL_CONTRACT_RELATIVE_PATH,
    TEMPORAL_MODEL_RELATIVE_PATH,
    TEMPORAL_RUNNER_RELATIVE_PATH,
    TEMPORAL_LAUNCHER_RELATIVE_PATH,
    *_TEMPORAL.REUSED_SOURCE_PATHS,
)
SOURCE_PATHS = tuple(dict.fromkeys((*ADDITIVE_SOURCE_PATHS, *REUSED_SOURCE_PATHS)))

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_causal_motion_alignment_v1_"
    "preregistration_2026-07-24.md"
)
PREREGISTRATION_COMMIT = "a3cea116e5cdf6cfec3801624c51306742e0f0f5"
PREREGISTRATION_FILE_SHA256 = (
    "500362b928b009d6c79487a3842441fff5e1708f6e2e5d7e0d0a39c698db3293"
)
PREREGISTRATION_BYTE_COUNT = 15_309
PREREGISTRATION_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_causal_motion_alignment_v1_"
    "preregistration_independent_review_v2_2026-07-24.json"
)
PREREGISTRATION_REVIEW_FILE_SHA256 = (
    "27b750c4d1eb1efccc97565483939a93f1006c7147f24908aa003f33c5fce35a"
)
PREREGISTRATION_REVIEW_CONTENT_SHA256 = (
    "daf18364b1d965d8f5f9739bbe65f940b7f6ec70fb6aa2afe9deabe8245a42f9"
)
PREREGISTRATION_REVIEW_BYTE_COUNT = 7_286

SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_causal_motion_alignment_v1_"
    "source_manifest_2026-07-24.json"
)
SOURCE_MANIFEST_SCHEMA = (
    "lewm_go2_rgb_causal_motion_alignment_v1_source_manifest"
)
SOURCE_MANIFEST_ENTRYPOINTS = (LAUNCHER_RELATIVE_PATH, RUNNER_RELATIVE_PATH)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = tuple(dict.fromkeys((
    CONTRACT_RELATIVE_PATH,
    MODEL_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    TEMPORAL_CONTRACT_RELATIVE_PATH,
    TEMPORAL_RUNNER_RELATIVE_PATH,
    TEMPORAL_LAUNCHER_RELATIVE_PATH,
    CONTRACT_TEST_RELATIVE_PATH,
    MODEL_TEST_RELATIVE_PATH,
    RUNNER_TEST_RELATIVE_PATH,
    EVALUATOR_TEST_RELATIVE_PATH,
    RECEIPT_BOUNDARY_TEST_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    SOURCE_CLOSURE_TEST_RELATIVE_PATH,
    SOURCE_CLOSURE_BASE_CHECKER_RELATIVE_PATH,
    *_TEMPORAL.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES,
)))
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_causal_motion_alignment_v1_source_review_2026-07-24.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_causal_motion_alignment_v1_"
    "execution_authorization_2026-07-24.json"
)
SOURCE_REVIEW_ADDITIONAL_PATHS = (
    SOURCE_MANIFEST_RELATIVE_PATH,
    PREREGISTRATION_RELATIVE_PATH,
    PREREGISTRATION_REVIEW_RELATIVE_PATH,
    TEMPORAL_SOURCE_MANIFEST_RELATIVE_PATH,
)
FROZEN_SOURCE_SHA256 = {
    **dict(_TEMPORAL.FROZEN_SOURCE_SHA256),
    TEMPORAL_CONTRACT_RELATIVE_PATH: TEMPORAL_CONTRACT_FILE_SHA256,
    TEMPORAL_MODEL_RELATIVE_PATH: TEMPORAL_MODEL_FILE_SHA256,
    TEMPORAL_RUNNER_RELATIVE_PATH: TEMPORAL_RUNNER_FILE_SHA256,
    TEMPORAL_LAUNCHER_RELATIVE_PATH: TEMPORAL_LAUNCHER_FILE_SHA256,
    TEMPORAL_SOURCE_MANIFEST_RELATIVE_PATH:
        TEMPORAL_SOURCE_MANIFEST_FILE_SHA256,
}

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "rgb_causal_motion_alignment_probe_v1"
)
PROHIBITED_RUNTIME_OUTPUT_ROOTS = tuple(dict.fromkeys(
    (
        *_TEMPORAL.PROHIBITED_RUNTIME_OUTPUT_ROOTS,
        _TEMPORAL.OUTPUT_ROOT_RELATIVE_PATH,
    )
))

MODEL_FAMILY = (
    "shared_observable_camera_ray_jepa_v5_multires_motion_alignment_v1"
)
MODEL_RUNTIME_VERSION = (
    "lewm_go2_rgb_causal_motion_alignment_v1_model_runtime_v1"
)
ALIGNMENT_INITIALIZATION_SEED = 20_260_726
ALIGNMENT_PARAMETER_COUNT = 12_832
ALIGNMENT_PARAMETER_TENSOR_COUNT = 4
CHANGED_POST_ENCODER_PARAMETER_COUNT = 15_992
CHANGED_POST_ENCODER_PARAMETER_TENSOR_COUNT = 9
ALIGNMENT_STATE_PREFIX = "evidence_head.motion_alignment."
PREDECESSOR_EVIDENCE_HEAD_PARAMETER_COUNT = 355_849
PREDECESSOR_EVIDENCE_HEAD_PARAMETER_TENSOR_COUNT = 31
EVIDENCE_HEAD_PARAMETER_CEILING = 368_681
EXPECTED_PARAMETER_COUNTS = {
    "evidence_head": 368_681,
    "encoder": 2_747_520,
}
EXPECTED_PARAMETER_TENSOR_COUNTS = {
    "evidence_head": 35,
    "encoder": 78,
}
EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT = 3_116_201
EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT = 113

MOTION_CONDITION_COMPONENTS = (
    "nominal_forward_m",
    "nominal_left_m",
    "nominal_yaw_rad",
    "relative_roll_rad",
    "relative_pitch_rad",
)
MODEL_INPUTS = (
    "previous_raw_visual_tokens_at_fixed_lag",
    "current_raw_visual_tokens",
    "causal_motion_condition_5d",
    "history_valid",
)
FORBIDDEN_MODEL_INPUTS = (
    "primitive_id_or_string",
    "outgoing_primitive",
    "per_sample_realized_relative_se2_current_frame",
    "exact_simulator_pose",
    "simulator_position_velocity_or_world_transform",
    "future_realized_motion",
    "requested_target_or_evaluator_feedback",
    "scene_geometry_labels_depth_or_ground_truth",
    "calibration_role_g2_navigation_or_heldout_input",
    "prior_run_output",
    "failed_temporal_or_multiresolution_checkpoint",
)

REVIEW_SCHEMA = f"{SCHEMA_PREFIX}_source_review_v1"
AUTHORIZATION_SCHEMA = f"{SCHEMA_PREFIX}_execution_authorization_v1"
RESERVATION_SCHEMA = f"{SCHEMA_PREFIX}_reservation_v1"
SNAPSHOT_SCHEMA = f"{SCHEMA_PREFIX}_snapshot_v1"
METRIC_SIDECAR_SCHEMA = f"{SCHEMA_PREFIX}_metric_sidecar_v1"
CHECKPOINT_METRICS_SCHEMA = f"{SCHEMA_PREFIX}_checkpoint_metrics_v1"
ACCESS_SCHEMA = f"{SCHEMA_PREFIX}_access_v1"
RESULT_SCHEMA = f"{SCHEMA_PREFIX}_result_v1"
COMPLETION_SCHEMA = f"{SCHEMA_PREFIX}_completion_v1"
FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v1"
PRE_LEDGER_FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_pre_ledger_failure_v1"
CONTRACT_INVALID_LEDGER_FAILURE_SCHEMA = (
    f"{SCHEMA_PREFIX}_contract_invalid_ledger_failure_v1"
)
PARTIAL_ACCESS_RECORD_SCHEMA = f"{SCHEMA_PREFIX}_partial_access_record_v1"
PARTIAL_ACCESS_LEDGER_SCHEMA = f"{SCHEMA_PREFIX}_partial_access_ledger_v1"

SOURCE_ONLY_AUTHORITY = dict(_TEMPORAL.SOURCE_ONLY_AUTHORITY)
REVIEW_AUTHORITY = dict(SOURCE_ONLY_AUTHORITY)
EXECUTION_AUTHORITY = {
    "one_exact_probe_attempt_authorized": True,
    "one_discrete_r9700_authorized": True,
    "n320_initialization_only_authorized": True,
    "train_and_checkpoint_selection_roles_only_authorized": True,
    "generated_mutation_scope": OUTPUT_ROOT_RELATIVE_PATH,
    "output_root_must_be_absent": True,
    **dict(DOWNSTREAM_DENIALS),
}

DEVELOPMENT_RGB_PATH_PATTERN = (
    r"\A\.generated/go2_render_selected_v04/scenes/"
    r"scene_[0-9a-f]{16}/rgb/frame_[0-9]{6}_env_[0-9]{2}\.png\Z"
)
_DEVELOPMENT_RGB_PATH_RE = re.compile(DEVELOPMENT_RGB_PATH_PATTERN)


def is_development_rgb_path(value: object) -> bool:
    """Return true only for the exact preregistered regular-path spelling."""

    try:
        path = safe_relative_path(value, name="development RGB path")
    except (TypeError, ValueError, PermissionError):
        return False
    return _DEVELOPMENT_RGB_PATH_RE.fullmatch(path) is not None


def _validated_expected_open_binding(value: object) -> dict[str, Any]:
    """Validate one runtime binding with the sole render-RGB allowlist repair."""

    if (
        type(value) is not dict
        or set(value)
        != {"path", "file_sha256", "content_sha256", "byte_count"}
        or not is_sha256(value["file_sha256"])
        or (
            value["content_sha256"] is not None
            and not is_sha256(value["content_sha256"])
        )
        or (
            value["byte_count"] is not None
            and (
                type(value["byte_count"]) is not int
                or value["byte_count"] <= 0
            )
        )
    ):
        raise PermissionError("attempted runtime binding changed")
    path = safe_relative_path(value["path"], name="attempted runtime input")
    fixed = {
        RAW_MANIFEST_RELATIVE_PATH,
        RAW_AUDIT_RELATIVE_PATH,
        N320_GATE_RELATIVE_PATH,
        N320_CHECKPOINT_RELATIVE_PATH,
        SCHEDULE_RELATIVE_PATH,
    }
    allowed = (
        path in fixed
        or path.startswith(RAW_ROOT_RELATIVE_PATH + "/")
        or is_development_rgb_path(path)
    )
    if not allowed:
        raise PermissionError("attempted input escaped frozen runtime roots")
    if any(
        path == root or path.startswith(root + "/")
        for root in PROHIBITED_RUNTIME_OUTPUT_ROOTS
    ):
        raise PermissionError("prior runtime output was attempted")
    parts = PurePosixPath(path).parts
    if (
        path.endswith("sealed_test.json")
        or any(part == "sealed" or part.startswith("sealed_") for part in parts)
    ):
        raise PermissionError("protected input was attempted")
    return dict(value)


def validate_runtime_inputs(value: object) -> dict[str, Any]:
    """Retain the frozen leaves while naming the successor training operation."""

    if type(value) is not dict:
        raise PermissionError("runtime input groups changed")
    translated = deepcopy(value)
    try:
        grant = translated["raw"]["grant"]
    except (KeyError, TypeError):
        raise PermissionError("raw runtime authority changed") from None
    expected_operations = [
        "development_rgb_decode",
        "causal_motion_alignment_training",
        "physical_checkpoint_selection",
    ]
    if (
        type(grant) is not dict
        or grant.get("allowed_operations") != expected_operations
    ):
        raise PermissionError("raw runtime authority changed")
    grant["allowed_operations"] = [
        "development_rgb_decode",
        "causal_temporal_perception_training",
        "physical_checkpoint_selection",
    ]
    _BASE_VALIDATE_RUNTIME_INPUTS(translated)
    return deepcopy(value)


def _validate_frozen_temporal_source_manifest(root: Path) -> None:
    raw = _TEMPORAL._read_regular_source(
        root / TEMPORAL_SOURCE_MANIFEST_RELATIVE_PATH
    )
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PermissionError(
            "frozen temporal source manifest is malformed"
        ) from error
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if (
        len(raw) != TEMPORAL_SOURCE_MANIFEST_BYTE_COUNT
        or hashlib.sha256(raw).hexdigest()
        != TEMPORAL_SOURCE_MANIFEST_FILE_SHA256
        or declared != TEMPORAL_SOURCE_MANIFEST_CONTENT_SHA256
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("frozen temporal source manifest changed")


def source_freeze_status(root: Path = ROOT) -> dict[str, Any]:
    """Require the new closure and the exact reviewed temporal source freeze."""

    status = dict(_BASE_SOURCE_FREEZE_STATUS(root))
    inconsistent = set(status["inconsistent_freeze_fields"])
    try:
        _validate_frozen_temporal_source_manifest(root)
    except (OSError, PermissionError, RuntimeError, TypeError, ValueError):
        inconsistent.add(TEMPORAL_SOURCE_MANIFEST_RELATIVE_PATH)
    else:
        inconsistent.discard(TEMPORAL_SOURCE_MANIFEST_RELATIVE_PATH)
    status["inconsistent_freeze_fields"] = sorted(inconsistent)
    status["ready"] = bool(
        status["manifest_present"]
        and status["manifest_valid"]
        and not status["missing_source_paths"]
        and not inconsistent
        and not status["malformed_source_bindings"]
        and not status["unset_freeze_fields"]
        and status["manifest_error"] is None
    )
    return status


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    """Rehash the reviewed closure and exact frozen temporal manifest."""

    result = _BASE_CURRENT_SOURCE_BINDINGS(root)
    _validate_frozen_temporal_source_manifest(root)
    if (
        result.get(TEMPORAL_SOURCE_MANIFEST_RELATIVE_PATH)
        != TEMPORAL_SOURCE_MANIFEST_FILE_SHA256
    ):
        raise PermissionError("frozen temporal source manifest binding changed")
    return result


def science_contract() -> dict[str, Any]:
    """Return the exact one-mechanism motion-alignment science identity."""

    value = _BASE_SCIENCE_CONTRACT()
    value["schema"] = f"{SCHEMA_PREFIX}_science_contract_v1"
    value["one_science_delta"] = (
        "causal_motion_conditioned_dense_previous_token_alignment_before_"
        "retained_temporal_residual"
    )
    value.pop("temporal_mechanism")
    value["motion_alignment_mechanism"] = {
        "inputs": list(MODEL_INPUTS),
        "condition_components": list(MOTION_CONDITION_COMPONENTS),
        "forbidden_inputs": list(FORBIDDEN_MODEL_INPUTS),
        "lag_s": HISTORY_LAG_S,
        "tick_count": HISTORY_TICK_COUNT,
        "tick_s": HISTORY_TICK_S,
        "incoming_predecessor_primitive_only": True,
        "outgoing_primitive_forbidden": True,
        "per_sample_realized_se2_materialized_to_model": False,
        "cold_condition": [0.0, 0.0, 0.0, 0.0, 0.0],
        "cold_history_exact_bypass": True,
        "alignment_state_prefix": ALIGNMENT_STATE_PREFIX,
        "temporal_state_prefix": TEMPORAL_STATE_PREFIX,
        "formula": [
            "concat_previous_current_and_broadcast_5d_condition_389_channels",
            "conv1x1_389_to_32_with_bias",
            "gelu_approximate_none",
            "depthwise_conv3x3_32_groups_32_padding_1_no_bias",
            "gelu_approximate_none",
            "zero_initialized_conv1x1_32_to_2_no_bias",
            "offset_tokens=2*tanh(raw_offset)",
            "sample_grid=identity_grid+xy_offset_tokens*(2/15)",
            "bilinear_border_grid_sample_align_corners_true",
            "delta=current_tokens-aligned_previous_tokens",
            "retained_temporal_residual_192_to_8_to_192",
            "fused=current_tokens+history_valid*residual",
        ],
    }
    value["initialization"] = {
        **value["initialization"],
        "alignment_local_cpu_seed": ALIGNMENT_INITIALIZATION_SEED,
        "alignment_entry_copy_count": 0,
        "alignment_hidden_xavier_uniform_gain": 1.0,
        "alignment_input_bias_zero": True,
        "alignment_offset_projection_zero": True,
    }
    value["parameter_counts"] = {
        **dict(EXPECTED_PARAMETER_COUNTS),
        "alignment": ALIGNMENT_PARAMETER_COUNT,
        "temporal": TEMPORAL_PARAMETER_COUNT,
        "changed_post_encoder": CHANGED_POST_ENCODER_PARAMETER_COUNT,
        "total_trainable": EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT,
    }
    value["parameter_tensor_counts"] = {
        **dict(EXPECTED_PARAMETER_TENSOR_COUNTS),
        "alignment": ALIGNMENT_PARAMETER_TENSOR_COUNT,
        "temporal": TEMPORAL_PARAMETER_TENSOR_COUNT,
        "changed_post_encoder": CHANGED_POST_ENCODER_PARAMETER_TENSOR_COUNT,
        "total_trainable": EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT,
    }
    return value


def lifecycle_contract() -> dict[str, Any]:
    value = _BASE_LIFECYCLE_CONTRACT()
    value.update({
        "receipt_validator_repair_science_change": False,
        "render_rgb_allowlist_pattern": DEVELOPMENT_RGB_PATH_PATTERN,
        "scientific_admissibility_requires_full_corrected_parser": True,
        "full_parser_must_include_terminal_rehashes_and_terminal_record": True,
        "parser_failure_consumes_attempt_as_contract_invalid": True,
        "preledger_metric_artifacts_scientifically_admissible": False,
        "preledger_integrity_or_pass_fail_control_emitted": False,
        "terminal_control_materialized_after_finalized_parser_only": True,
    })
    return value


def validate_runtime_open_context(
    *,
    path: object,
    purpose: object,
    role: object,
    kind: object,
) -> dict[str, str]:
    """Validate the exact context labels emitted by the frozen runner."""

    try:
        relative = safe_relative_path(path, name="attempted runtime input")
    except (TypeError, ValueError) as error:
        raise PermissionError(
            "runtime input path/purpose/role/kind context changed"
        ) from error
    fixed_kinds = {
        SCHEDULE_RELATIVE_PATH: "bound_schedule",
        N320_GATE_RELATIVE_PATH: "n320_gate",
        N320_CHECKPOINT_RELATIVE_PATH: "n320_checkpoint",
        RAW_MANIFEST_RELATIVE_PATH: "raw_authority_manifest",
        RAW_AUDIT_RELATIVE_PATH: "raw_authority_audit",
        RAW_PAIRS_RELATIVE_PATH: "raw_pairs_index",
        RAW_ENDPOINTS_RELATIVE_PATH: "raw_endpoints_index",
    }
    if relative in fixed_kinds:
        expected_kind = fixed_kinds[relative]
        context_allowed = (
            purpose in {"runtime_load", "terminal_rehash"}
            and role == "authority"
        )
    elif relative.startswith(RAW_ROOT_RELATIVE_PATH + "/"):
        expected_kind = "raw_supervision"
        context_allowed = (
            purpose == "runtime_load"
            and role in {"train", "checkpoint_selection"}
        ) or (
            purpose == "terminal_rehash"
            and role == "authority"
        )
    elif is_development_rgb_path(relative):
        expected_kind = "development_rgb"
        context_allowed = (
            purpose == "runtime_load"
            and role in {"train", "checkpoint_selection"}
        ) or (
            purpose == "terminal_rehash"
            and role == "authority"
        )
    else:
        raise PermissionError(
            "runtime input path/purpose/role/kind context changed"
        )
    if (
        type(purpose) is not str
        or type(role) is not str
        or type(kind) is not str
        or kind != expected_kind
        or not context_allowed
    ):
        raise PermissionError(
            "runtime input path/purpose/role/kind context changed"
        )
    return {
        "path": relative,
        "purpose": purpose,
        "role": role,
        "kind": kind,
    }


def parse_structural_partial_access_ledger(
    raw: bytes,
) -> list[dict[str, Any]]:
    """Validate canonical chain/outcomes while deliberately skipping admission."""

    _STRUCTURAL_TEMPORAL.PARTIAL_ACCESS_RECORD_SCHEMA = (
        PARTIAL_ACCESS_RECORD_SCHEMA
    )
    return _STRUCTURAL_PARSE_PARTIAL_ACCESS_LEDGER(raw)


def parse_partial_access_ledger(raw: bytes) -> list[dict[str, Any]]:
    """Run the inherited parser plus exact runtime-open context closure."""

    try:
        records = _BASE_PARSE_PARTIAL_ACCESS_LEDGER(raw)
    except (TypeError, ValueError) as error:
        raise PermissionError(
            "partial-access ledger failed canonical fail-closed validation"
        ) from error
    for record in records:
        if record["record_type"] != "OPEN_ATTEMPTED":
            continue
        validate_runtime_open_context(
            path=record["expected_binding"]["path"],
            purpose=record["purpose"],
            role=record["role"],
            kind=record["kind"],
        )
    return records


def validate_finalized_access_ledger(raw: bytes) -> list[dict[str, Any]]:
    """Require full corrected-parser success before real result admissibility."""

    records = parse_partial_access_ledger(raw)
    if records[-1]["record_type"] != "RUNTIME_INPUT_ACCESS_FINALIZED":
        raise PermissionError(
            "scientific admissibility requires a finalized access ledger"
        )
    return records


def validate_no_preledger_scientific_control(value: object) -> None:
    """Reject any nested finalized-integrity or scientific control claim."""

    forbidden_values = {
        CONTROL_CONTINUE,
        CONTROL_PASS,
        CONTROL_FAIL,
        CONTROL_INTEGRITY_FAIL,
    }

    def walk(item: object) -> None:
        if isinstance(item, dict):
            for key, nested in item.items():
                if key == "integrity_pass":
                    raise PermissionError(
                        "preledger artifact contains integrity_pass"
                    )
                walk(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                walk(nested)
        elif isinstance(item, str) and item in forbidden_values:
            raise PermissionError(
                "preledger artifact contains a scientific control token"
            )

    walk(value)


def validate_provisional_metric(
    value: object,
    *,
    update: int,
) -> dict[str, Any]:
    """Validate the exact local metric shape without claiming access integrity."""

    metric_fields = {
        "update",
        "role",
        "pair_count",
        "unique_endpoint_count",
        "temporal_population",
        "scopes",
        "warm_scopes_informational_only",
        "aggregate_complete_v4_tail_depth_loss",
        "evaluation",
        "preledger_model_state_checks_pass",
        "state_sha256_before",
        "state_sha256_after",
        "frozen_state_sha256_before_and_after",
        "state_mutation_count",
    }
    evaluation_fields = {
        "scope_evaluations",
        "complete_physical_scope_count",
        "margin_count",
        "passed_margin_count",
        "total_shortfall",
        "worst_margin",
        "rough_motion",
    }
    if (
        type(value) is not dict
        or set(value) != metric_fields
        or value["update"] != update
        or type(value.get("evaluation")) is not dict
        or set(value["evaluation"]) != evaluation_fields
        or type(value["evaluation"]["scope_evaluations"]) is not dict
        or tuple(value["evaluation"]["scope_evaluations"]) != SCOPES
        or any(
            type(row) is not dict
            or set(row) != {"physical_margins", "passes"}
            or type(row["physical_margins"]) is not list
            or type(row["passes"]) is not bool
            for row in value["evaluation"]["scope_evaluations"].values()
        )
        or type(value["evaluation"]["rough_motion"]) is not dict
        or set(value["evaluation"]["rough_motion"])
        != {
            "pixel_balanced_accuracy",
            "ground_balanced_accuracy",
            "depth_p95_m",
        }
        or value["preledger_model_state_checks_pass"] is not True
    ):
        raise PermissionError("provisional checkpoint metric changed")
    validate_no_preledger_scientific_control(value)
    return dict(value)


def provisional_checkpoint_control(update: int) -> dict[str, Any]:
    if update not in CHECKPOINT_UPDATES:
        raise ValueError("provisional checkpoint update is not fixed")
    return {
        "schema": f"{SCHEMA_PREFIX}_provisional_checkpoint_flow_v1",
        "update": update,
        "status": "PENDING_FINALIZED_ACCESS_LEDGER_PARSE",
        "training_flow": (
            "CONTINUE_TO_NEXT_FIXED_CHECKPOINT"
            if update in (100, 400)
            else "FIXED_TRAINING_COMPLETE"
        ),
        "next_update": {100: 400, 400: 1_000}.get(update),
        "scientifically_admissible": False,
        "finalized_access_ledger_parser_pass": None,
    }


def validate_metric_sidecar(
    value: object,
    *,
    update: int | None = None,
) -> dict[str, Any]:
    """Validate a durable pre-ledger sidecar as explicitly inadmissible."""

    fields = {
        "schema",
        "status",
        "update",
        "checkpoint",
        "metric",
        "inline_evaluation_count",
        "state_mutation_count",
        "publication_order",
        "continuation",
        "scientifically_admissible",
        "authority",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("provisional metric-sidecar fields changed")
    core = dict(value)
    declared = core.pop("content_sha256")
    metric = value["metric"]
    if (
        value["status"]
        != "PROVISIONAL_INADMISSIBLE_PENDING_FINALIZED_LEDGER_PARSE"
        or value["continuation"]
        != provisional_checkpoint_control(value["update"])
        or value["scientifically_admissible"] is not False
        or value["publication_order"]
        != [
            "cpu_snapshot",
            "inline_nonmutating_selection_evaluation",
            "atomic_mode_0444_provisional_sidecar",
            "internal_fixed_training_flow_only",
        ]
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("provisional metric sidecar changed")
    validate_provisional_metric(metric, update=int(value["update"]))
    validate_no_preledger_scientific_control(value)

    translated = deepcopy(value)
    translated.pop("scientifically_admissible")
    translated["status"] = (
        "PUBLISHED_0444_AFTER_INLINE_EVALUATION_BEFORE_CONTROL"
    )
    translated["publication_order"] = [
        "cpu_snapshot",
        "inline_nonmutating_selection_evaluation",
        "atomic_mode_0444_sidecar",
        "control_branch",
    ]
    translated_metric = dict(translated["metric"])
    translated_metric["integrity_pass"] = translated_metric.pop(
        "preledger_model_state_checks_pass"
    )
    translated["metric"] = translated_metric
    translated["continuation"] = checkpoint_control_decision(
        update=int(translated["update"]),
        evaluation=translated_metric["evaluation"],
        integrity_pass=True,
    )
    translated.pop("content_sha256")
    translated = with_content_sha256(translated)
    _BASE_VALIDATE_METRIC_SIDECAR(translated, update=update)
    return dict(value)


NORMAL_FAILURE_STATUS = (
    "TERMINAL_CAUSAL_MOTION_ALIGNMENT_V1_OPERATIONAL_OR_INTEGRITY_"
    "FAILURE_NO_RETRY"
)
PRE_LEDGER_FAILURE_STATUS = (
    "TERMINAL_CAUSAL_MOTION_ALIGNMENT_V1_POST_RESERVATION_PRE_LEDGER_"
    "FAILURE_NO_RETRY"
)
CONTRACT_INVALID_LEDGER_FAILURE_STATUS = (
    "TERMINAL_CAUSAL_MOTION_ALIGNMENT_V1_CONTRACT_INVALID_ACCESS_LEDGER_"
    "NO_RETRY"
)
_TEMPORAL_NORMAL_FAILURE_STATUS = (
    "TERMINAL_CAUSAL_TEMPORAL_V1_OPERATIONAL_OR_INTEGRITY_FAILURE_NO_RETRY"
)
_TEMPORAL_PRE_LEDGER_FAILURE_STATUS = (
    "TERMINAL_CAUSAL_TEMPORAL_V1_POST_RESERVATION_PRE_LEDGER_FAILURE_NO_RETRY"
)


def _validate_runtime_open_rows_context(rows: object) -> None:
    if type(rows) is not list:
        raise PermissionError("failure runtime-open rows changed")
    for row in rows:
        if type(row) is not dict:
            raise PermissionError("failure runtime-open row changed")
        validate_runtime_open_context(
            path=row.get("expected_binding", {}).get("path"),
            purpose=row.get("purpose"),
            role=row.get("role"),
            kind=row.get("kind"),
        )


def validate_failure_receipt(
    value: object,
    *,
    reservation_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Retain the frozen receipt structure with the successor status identity."""

    if type(value) is not dict:
        raise PermissionError("motion-alignment failure receipt changed")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if (
        value.get("schema") != FAILURE_SCHEMA
        or value.get("status") != NORMAL_FAILURE_STATUS
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("motion-alignment failure receipt changed")
    _validate_runtime_open_rows_context(value.get("runtime_opens"))
    translated = deepcopy(value)
    translated["status"] = _TEMPORAL_NORMAL_FAILURE_STATUS
    translated.pop("content_sha256")
    translated = with_content_sha256(translated)
    _BASE_VALIDATE_FAILURE_RECEIPT(
        translated,
        reservation_binding=reservation_binding,
    )
    return dict(value)


def validate_pre_ledger_failure_receipt(
    value: object,
    *,
    reservation_binding: Mapping[str, Any] | None = None,
    attempt_identity: str | None = None,
) -> dict[str, Any]:
    """Retain the frozen pre-ledger structure with the successor identity."""

    if type(value) is not dict:
        raise PermissionError("motion-alignment pre-ledger receipt changed")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if (
        value.get("schema") != PRE_LEDGER_FAILURE_SCHEMA
        or value.get("status") != PRE_LEDGER_FAILURE_STATUS
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("motion-alignment pre-ledger receipt changed")
    translated = deepcopy(value)
    translated["status"] = _TEMPORAL_PRE_LEDGER_FAILURE_STATUS
    translated.pop("content_sha256")
    translated = with_content_sha256(translated)
    _BASE_VALIDATE_PRE_LEDGER_FAILURE_RECEIPT(
        translated,
        reservation_binding=reservation_binding,
        attempt_identity=attempt_identity,
    )
    return dict(value)


def _runtime_opens_from_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    attempts = {
        int(row["open_id"]): row
        for row in records
        if row["record_type"] == "OPEN_ATTEMPTED"
    }
    outcomes = {
        int(row["open_id"]): row
        for row in records
        if row["record_type"] == "OPEN_OUTCOME"
    }
    if set(attempts) != set(outcomes):
        raise PermissionError("contract-invalid ledger has an unpaired open")
    return [
        {
            "open_id": open_id,
            "stage": attempts[open_id]["stage"],
            "kind": attempts[open_id]["kind"],
            "role": attempts[open_id]["role"],
            "purpose": attempts[open_id]["purpose"],
            "expected_binding": attempts[open_id]["expected_binding"],
            "outcome": outcomes[open_id]["outcome"],
            "descriptor_opened": outcomes[open_id]["descriptor_opened"],
            "read_completed": outcomes[open_id]["read_completed"],
            "binding_accepted": outcomes[open_id]["binding_accepted"],
            "observed_binding": outcomes[open_id]["observed_binding"],
            "partial_byte_count": outcomes[open_id]["partial_byte_count"],
            "error": outcomes[open_id]["error"],
        }
        for open_id in sorted(attempts)
    ]


def validate_contract_invalid_ledger_failure_receipt(
    value: object,
    *,
    reservation_binding: Mapping[str, Any],
    ledger_raw: bytes,
) -> dict[str, Any]:
    """Prove an exact on-disk ledger is structural but context-invalid."""

    if type(value) is not dict or type(ledger_raw) is not bytes:
        raise PermissionError("contract-invalid failure receipt changed")
    fields = {
        "schema",
        "status",
        "attempt_identity",
        "reservation",
        "partial_access_ledger",
        "runtime_opens",
        "runtime_opens_sha256",
        "failure_stage",
        "operation_counts",
        "published_prefix",
        "published_prefix_sha256",
        "directories_including_root",
        "error",
        "ledger_parser_failure",
        "scientific_result",
        "scientific_result_status",
        "retry_authorized",
        "g2_navigation_or_heldout_attempted",
        "prior_runtime_output_open_count",
        "authority",
        "terminalization",
        "content_sha256",
    }
    if set(value) != fields:
        raise PermissionError("contract-invalid failure fields changed")
    core = dict(value)
    declared = core.pop("content_sha256")
    ledger = value["partial_access_ledger"]
    parser_failure = value["ledger_parser_failure"]
    if (
        value["schema"] != CONTRACT_INVALID_LEDGER_FAILURE_SCHEMA
        or value["status"] != CONTRACT_INVALID_LEDGER_FAILURE_STATUS
        or value["scientific_result_status"]
        != "NOT_OBSERVED_CONTRACT_INVALID_ACCESS_LEDGER"
        or type(ledger) is not dict
        or ledger.get("file_sha256") != hashlib.sha256(ledger_raw).hexdigest()
        or ledger.get("byte_count") != len(ledger_raw)
        or type(parser_failure) is not dict
        or set(parser_failure)
        != {
            "validator",
            "full_on_disk_ledger_checked",
            "accepted",
            "ledger_file_sha256",
            "error",
        }
        or parser_failure["validator"] != "parse_partial_access_ledger"
        or parser_failure["full_on_disk_ledger_checked"] is not True
        or parser_failure["accepted"] is not False
        or parser_failure["ledger_file_sha256"] != ledger["file_sha256"]
        or _TEMPORAL._validated_error(
            parser_failure["error"], allow_none=False
        )
        is None
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("contract-invalid failure receipt changed")

    structural_records = parse_structural_partial_access_ledger(ledger_raw)
    runtime_opens = _runtime_opens_from_records(structural_records)
    if (
        structural_records[-1]["record_type"] not in {
            "ATTEMPT_TERMINATING",
            "RUNTIME_INPUT_ACCESS_FINALIZED",
        }
        or ledger.get("record_count") != len(structural_records)
        or ledger.get("records_content_sha256")
        != canonical_json_sha256(structural_records)
        or ledger.get("last_record_content_sha256")
        != structural_records[-1]["content_sha256"]
        or value["runtime_opens"] != runtime_opens
    ):
        raise PermissionError("contract-invalid raw ledger binding changed")
    try:
        parse_partial_access_ledger(ledger_raw)
    except BaseException as error:
        observed_parser_error = _TEMPORAL._validated_error(
            {
                "type": type(error).__name__,
                "message": str(error),
                "message_sha256": hashlib.sha256(
                    str(error).encode("utf-8")
                ).hexdigest(),
            },
            allow_none=False,
        )
    else:
        raise PermissionError("contract-invalid ledger unexpectedly parsed")
    if parser_failure["error"] != observed_parser_error:
        raise PermissionError("contract-invalid parser evidence changed")

    translated = deepcopy(value)
    translated.pop("ledger_parser_failure")
    translated["schema"] = _STRUCTURAL_TEMPORAL.FAILURE_SCHEMA
    translated["status"] = _TEMPORAL_NORMAL_FAILURE_STATUS
    translated["scientific_result_status"] = (
        "NOT_OBSERVED_TERMINAL_OPERATIONAL_OR_INTEGRITY_FAILURE"
    )
    translated.pop("content_sha256")
    translated = with_content_sha256(translated)
    _STRUCTURAL_VALIDATE_FAILURE_RECEIPT(
        translated,
        reservation_binding=reservation_binding,
    )
    return dict(value)


# Every inherited validator/function resolves globals in the private temporal
# module.  Rebind that namespace before re-exporting its public functions so
# there is no split identity between direct calls and nested validator calls.
for _name, _value in tuple(globals().items()):
    if _name.isupper():
        setattr(_TEMPORAL, _name, _value)
_TEMPORAL._validated_expected_open_binding = _validated_expected_open_binding
_TEMPORAL.current_source_bindings = current_source_bindings
_TEMPORAL.source_freeze_status = source_freeze_status
_TEMPORAL.validate_runtime_inputs = validate_runtime_inputs
_TEMPORAL.science_contract = science_contract
_TEMPORAL.lifecycle_contract = lifecycle_contract
_TEMPORAL.parse_partial_access_ledger = parse_partial_access_ledger
_TEMPORAL.validate_failure_receipt = validate_failure_receipt
_TEMPORAL.validate_metric_sidecar = validate_metric_sidecar
_TEMPORAL.validate_pre_ledger_failure_receipt = (
    validate_pre_ledger_failure_receipt
)

_OVERRIDDEN_PUBLIC = {
    "current_source_bindings",
    "is_development_rgb_path",
    "lifecycle_contract",
    "parse_partial_access_ledger",
    "parse_structural_partial_access_ledger",
    "provisional_checkpoint_control",
    "science_contract",
    "source_freeze_status",
    "validate_contract_invalid_ledger_failure_receipt",
    "validate_failure_receipt",
    "validate_finalized_access_ledger",
    "validate_metric_sidecar",
    "validate_no_preledger_scientific_control",
    "validate_pre_ledger_failure_receipt",
    "validate_provisional_metric",
    "validate_runtime_open_context",
    "validate_runtime_inputs",
}
for _name in _TEMPORAL.__all__:
    if _name not in _OVERRIDDEN_PUBLIC and not _name.isupper():
        globals()[_name] = getattr(_TEMPORAL, _name)

__all__ = [name for name in globals() if name.isupper()] + sorted(
    _OVERRIDDEN_PUBLIC
    | {
        name
        for name in _TEMPORAL.__all__
        if not name.isupper()
    }
)
