"""Source-only contract for RGB overlapping tokenization V1.

This is a deliberately thin adapter.  Scientific identity comes from the
exact frozen static multiresolution V3 contract; terminal-ledger and failure
receipt semantics come from the corrected causal-motion V1 lifecycle.  The
only new science mechanism is the 11x11, stride-seven, padding-two RGB patch
projection.  Importing this module reads source files only and imports no
tensor, image, dataset, generated-input, or accelerator library.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

STATIC_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v3.py"
)
STATIC_CONTRACT_FILE_SHA256 = (
    "3553810c79686f642a30fdfd0d2ff6ae047a97ea65c1366cae4cb3231e44e669"
)
STATIC_MODEL_RELATIVE_PATH = (
    "lewm/models/shared_observable_camera_ray_jepa_v5_multires_v1.py"
)
STATIC_MODEL_FILE_SHA256 = (
    "a63da1137539953b2f40d184def1652ae05f63d7b434084b1a91787e1fc83d0b"
)
STATIC_SCIENCE_COMMIT = "97824b29ce9f4789b18e7a0cb5bc36f2feac1704"

CORRECTED_LIFECYCLE_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_rgb_causal_motion_alignment_v1.py"
)
CORRECTED_LIFECYCLE_CONTRACT_FILE_SHA256 = (
    "3e1f5dc1bde9f4235b01ceab7621c8db8ad66f2fc8b51d3f0fd5354a7f5b3ce9"
)
CORRECTED_LIFECYCLE_RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_rgb_causal_motion_alignment_v1.py"
)
CORRECTED_LIFECYCLE_RUNNER_FILE_SHA256 = (
    "99dd6de6d65f53c1d81af786cd9c453b541639e07cebab6a6d83681396931ab9"
)


def _load_source_module(relative_path: str, module_name: str) -> Any:
    source = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source-only contract {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_STATIC = _load_source_module(
    STATIC_CONTRACT_RELATIVE_PATH,
    "_lewm_overlap_v1_frozen_static_v3_contract",
)
_MOTION = _load_source_module(
    CORRECTED_LIFECYCLE_CONTRACT_RELATIVE_PATH,
    "_lewm_overlap_v1_corrected_lifecycle_contract",
)

_base_static_science_contract = _STATIC.science_contract
_base_corrected_lifecycle_contract = _MOTION.lifecycle_contract
_base_validate_runtime_inputs = _MOTION.validate_runtime_inputs

# Re-export the corrected standard-library contract surface, then replace the
# successor identities below.  This preserves the reviewed ledger validators
# without copying another audit framework.
for _name in dir(_MOTION):
    if _name.isupper():
        globals()[_name] = getattr(_MOTION, _name)

canonical_json_bytes = _MOTION.canonical_json_bytes
canonical_json_sha256 = _MOTION.canonical_json_sha256
is_sha256 = _MOTION.is_sha256
with_content_sha256 = _MOTION.with_content_sha256
parse_canonical_json = _MOTION.parse_canonical_json
safe_relative_path = _MOTION.safe_relative_path
artifact_binding = _MOTION.artifact_binding
validate_binding = _MOTION.validate_binding


CONTRACT_AUTHOR = "/root/overlap_contract_impl"
SCHEMA_PREFIX = "lewm_go2_rgb_overlapping_tokenization_v1"

CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_rgb_overlapping_tokenization_v1.py"
)
MODEL_RELATIVE_PATH = (
    "lewm/models/"
    "shared_observable_camera_ray_jepa_v5_multires_"
    "overlapping_tokenization_v1.py"
)
RUNNER_RELATIVE_PATH = "scripts/run_go2_rgb_overlapping_tokenization_v1.py"
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_rgb_overlapping_tokenization_v1.py"
)
CONTRACT_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_rgb_overlapping_tokenization_v1_contract.py"
)
MODEL_TEST_RELATIVE_PATH = (
    "lewm/tests/test_shared_observable_camera_ray_jepa_v5_multires_"
    "overlapping_tokenization_v1.py"
)
RUNNER_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_rgb_overlapping_tokenization_v1_runner.py"
)
EVALUATOR_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_rgb_overlapping_tokenization_v1_evaluator.py"
)
RECEIPT_BOUNDARY_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_rgb_overlapping_tokenization_v1_receipt_boundary.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_rgb_overlapping_tokenization_v1_source_closure.py"
)
SOURCE_CLOSURE_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_rgb_overlapping_tokenization_v1_source_closure.py"
)
TEST_RELATIVE_PATH = RUNNER_TEST_RELATIVE_PATH

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_overlapping_tokenization_v1_"
    "preregistration_2026-07-25.md"
)
PREREGISTRATION_COMMIT = "c88eadf269d9acc8c4ca87576fea48ce14721ee5"
PREREGISTRATION_FILE_SHA256 = (
    "d7b9ae265efb54422ecb116d199cbfc4e8d36118da1c8924b3e0ede8e93c0086"
)
PREREGISTRATION_BYTE_COUNT = 19_860
PREREGISTRATION_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_overlapping_tokenization_v1_"
    "preregistration_independent_review_2026-07-25.json"
)
PREREGISTRATION_REVIEW_FILE_SHA256 = (
    "05b0dc96f02d19d265686017d78bec5ddc325a8d58dc67b57008a78e67ec7ce9"
)
PREREGISTRATION_REVIEW_CONTENT_SHA256 = (
    "bdf11948618ec07132e3b9986447ea50ead7bc04019c28b4760720fb3cd96710"
)
PREREGISTRATION_REVIEW_BYTE_COUNT = 6_324

SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_overlapping_tokenization_v1_"
    "source_manifest_2026-07-25.json"
)
SOURCE_MANIFEST_SCHEMA = (
    "lewm_go2_rgb_overlapping_tokenization_v1_source_manifest"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_overlapping_tokenization_v1_"
    "source_review_2026-07-25.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_overlapping_tokenization_v1_"
    "execution_authorization_2026-07-25.json"
)

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
REUSED_SOURCE_PATHS = tuple(dict.fromkeys((
    CORRECTED_LIFECYCLE_CONTRACT_RELATIVE_PATH,
    CORRECTED_LIFECYCLE_RUNNER_RELATIVE_PATH,
    STATIC_CONTRACT_RELATIVE_PATH,
    STATIC_MODEL_RELATIVE_PATH,
    *_MOTION.REUSED_SOURCE_PATHS,
)))
SOURCE_PATHS = tuple(dict.fromkeys((*ADDITIVE_SOURCE_PATHS, *REUSED_SOURCE_PATHS)))
SOURCE_MANIFEST_ENTRYPOINTS = (LAUNCHER_RELATIVE_PATH, RUNNER_RELATIVE_PATH)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = tuple(dict.fromkeys((
    *ADDITIVE_SOURCE_PATHS,
    CORRECTED_LIFECYCLE_CONTRACT_RELATIVE_PATH,
    CORRECTED_LIFECYCLE_RUNNER_RELATIVE_PATH,
    *_MOTION.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES,
)))
SOURCE_REVIEW_ADDITIONAL_PATHS = (
    SOURCE_MANIFEST_RELATIVE_PATH,
    PREREGISTRATION_RELATIVE_PATH,
    PREREGISTRATION_REVIEW_RELATIVE_PATH,
    TEMPORAL_SOURCE_MANIFEST_RELATIVE_PATH,
)
FROZEN_SOURCE_SHA256 = {
    **dict(_MOTION.FROZEN_SOURCE_SHA256),
    STATIC_CONTRACT_RELATIVE_PATH: STATIC_CONTRACT_FILE_SHA256,
    STATIC_MODEL_RELATIVE_PATH: STATIC_MODEL_FILE_SHA256,
    CORRECTED_LIFECYCLE_CONTRACT_RELATIVE_PATH:
        CORRECTED_LIFECYCLE_CONTRACT_FILE_SHA256,
    CORRECTED_LIFECYCLE_RUNNER_RELATIVE_PATH:
        CORRECTED_LIFECYCLE_RUNNER_FILE_SHA256,
}

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "rgb_overlapping_tokenization_probe_v1"
)
PROHIBITED_RUNTIME_OUTPUT_ROOTS = tuple(dict.fromkeys((
    *_MOTION.PROHIBITED_RUNTIME_OUTPUT_ROOTS,
    _MOTION.OUTPUT_ROOT_RELATIVE_PATH,
)))

MODEL_FAMILY = (
    "shared_observable_camera_ray_jepa_v5_multires_"
    "overlapping_tokenization_v1"
)
MODEL_RUNTIME_VERSION = (
    "lewm_go2_rgb_overlapping_tokenization_v1_model_runtime_v1"
)
ARCHITECTURE_SCHEMA = (
    "lewm_go2_shared_jepa_v5_multires_"
    "overlapping_tokenization_v1_architecture"
)
INITIALIZATION_SCHEMA = (
    "lewm_go2_shared_jepa_v5_multires_"
    "overlapping_tokenization_v1_initialization"
)
ONE_SCIENCE_DELTA = (
    "overlapping_rgb_patch_tokenization_relative_to_static_multires_v3_only"
)

PATCH_INPUT_CHANNELS = 3
PATCH_OUTPUT_CHANNELS = 192
PATCH_KERNEL_SIZE = (11, 11)
PATCH_STRIDE = (7, 7)
PATCH_PADDING = (2, 2)
PATCH_DILATION = (1, 1)
PATCH_GROUPS = 1
PATCH_BIAS = True
PATCH_PADDING_MODE = "zeros"
PATCH_CENTER_COPY_SLICE = (2, 9, 2, 9)
PATCH_CENTRAL_WEIGHT_SCALAR_COUNT = 28_224
PATCH_OUTER_RING_SCALAR_COUNT = 41_472
PATCH_BIAS_SCALAR_COUNT = 192
PATCH_WEIGHT_PARAMETER_COUNT = 69_696
PATCH_PROJECTION_PARAMETER_COUNT = 69_888
PATCH_ADJACENT_OVERLAP_PIXELS = 4

N320_EXACT_COPY_ENTRY_COUNT = 83
N320_TRANSFORMED_ENTRY_COUNT = 1
N320_DERIVED_ENTRY_COUNT = 84
N320_TRANSFORMED_ENTRY = "encoder.patch_embed.weight"
N320_SOURCE_PATCH_WEIGHT_SHAPE = (192, 3, 7, 7)
N320_DESTINATION_PATCH_WEIGHT_SHAPE = (192, 3, 11, 11)

EXPECTED_PARAMETER_COUNTS = {
    "evidence_head": 352_689,
    "encoder": 2_788_992,
}
EXPECTED_PARAMETER_TENSOR_COUNTS = {
    "evidence_head": 26,
    "encoder": 78,
}
EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT = 3_141_681
EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT = 104
EXPECTED_COMPLETE_MODEL_PARAMETER_COUNT = 7_049_460
EXPECTED_COMPLETE_MODEL_PARAMETER_TENSOR_COUNT = 232
TRAINABLE_PARAMETER_PREFIXES = ("evidence_head.", "encoder.")

_BASE_ARCHITECTURE = {
    "schema": ARCHITECTURE_SCHEMA,
    "model_family": MODEL_FAMILY,
    "scientific_delta": ONE_SCIENCE_DELTA,
    "one_science_delta": ONE_SCIENCE_DELTA,
    "decoder": {
        "input_channels": 192,
        "stage_channels": [112, 80, 56, 36, 36],
        "stage_sizes": [
            [16, 16],
            [28, 28],
            [56, 56],
            [112, 112],
            [112, 112],
        ],
        "group_counts": [8, 8, 8, 4, 4],
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "convolution_bias": True,
        "resize_mode": "bilinear",
        "align_corners": False,
        "antialias": False,
        "gelu_approximate": "none",
        "group_norm_eps": 1e-5,
        "group_norm_affine": True,
        "initialization_seed": 20_260_724,
        "parameter_count": 345_264,
        "parameter_tensor_count": 20,
    },
    "evidence_head": {
        "parameter_count": 352_689,
        "parameter_tensor_count": 26,
        "predecessor_parameter_ceiling": 357_993,
    },
    "patch_projection": {
        "input_channels": PATCH_INPUT_CHANNELS,
        "output_channels": PATCH_OUTPUT_CHANNELS,
        "predecessor_kernel_size": [7, 7],
        "kernel_size": list(PATCH_KERNEL_SIZE),
        "stride": list(PATCH_STRIDE),
        "padding": list(PATCH_PADDING),
        "dilation": list(PATCH_DILATION),
        "groups": PATCH_GROUPS,
        "bias": PATCH_BIAS,
        "padding_mode": PATCH_PADDING_MODE,
        "center_copy_slice": list(PATCH_CENTER_COPY_SLICE),
        "central_weight_scalar_count": PATCH_CENTRAL_WEIGHT_SCALAR_COUNT,
        "outer_ring_scalar_count": PATCH_OUTER_RING_SCALAR_COUNT,
        "bias_scalar_count": PATCH_BIAS_SCALAR_COUNT,
        "weight_parameter_count": PATCH_WEIGHT_PARAMETER_COUNT,
        "adjacent_overlap_pixels": PATCH_ADJACENT_OVERLAP_PIXELS,
        "configured_patch_size": 7,
    },
    "token_geometry": {
        "input_shape": [3, 112, 112],
        "patch_map_shape": [192, 16, 16],
        "patch_token_count": 256,
        "patch_token_width": 192,
        "cls_plus_patch_token_count": 257,
        "positional_embedding_shape": [1, 257, 192],
        "token_center_formula": "7*i+3",
    },
    "trainable": {
        "encoder_parameter_count": EXPECTED_PARAMETER_COUNTS["encoder"],
        "encoder_parameter_tensor_count":
            EXPECTED_PARAMETER_TENSOR_COUNTS["encoder"],
        "evidence_head_parameter_count":
            EXPECTED_PARAMETER_COUNTS["evidence_head"],
        "evidence_head_parameter_tensor_count":
            EXPECTED_PARAMETER_TENSOR_COUNTS["evidence_head"],
        "total_parameter_count": EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT,
        "total_parameter_tensor_count":
            EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT,
    },
    "complete_model": {
        "parameter_count": EXPECTED_COMPLETE_MODEL_PARAMETER_COUNT,
        "parameter_tensor_count":
            EXPECTED_COMPLETE_MODEL_PARAMETER_TENSOR_COUNT,
    },
    "jepa_tensor_interface": {
        "online_patch_tokens_shape": [256, 192],
        "target_patch_tokens_shape": [256, 192],
        "same_shape_ema_target": True,
    },
    "unchanged_consumers": [
        "pixel_head",
        "ground_head",
        "camera_geometry",
        "ray_depth_ground_output_contract",
        "rasterization",
    ],
    "intermediate_encoder_features_used": False,
    "temporal_or_motion_module_present": False,
}


def overlapping_tokenization_architecture_contract_v1() -> dict[str, Any]:
    """Return the literal source-bound overlap architecture."""

    return deepcopy(_BASE_ARCHITECTURE)


OVERLAPPING_TOKENIZATION_ARCHITECTURE_CONTRACT = (
    overlapping_tokenization_architecture_contract_v1()
)
ARCHITECTURE_CONTRACT_SHA256 = canonical_json_sha256(
    OVERLAPPING_TOKENIZATION_ARCHITECTURE_CONTRACT
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

NORMAL_FAILURE_STATUS = (
    "TERMINAL_OVERLAPPING_TOKENIZATION_V1_OPERATIONAL_OR_INTEGRITY_"
    "FAILURE_NO_RETRY"
)
PRE_LEDGER_FAILURE_STATUS = (
    "TERMINAL_OVERLAPPING_TOKENIZATION_V1_POST_RESERVATION_PRE_LEDGER_"
    "FAILURE_NO_RETRY"
)
CONTRACT_INVALID_LEDGER_FAILURE_STATUS = (
    "TERMINAL_OVERLAPPING_TOKENIZATION_V1_CONTRACT_INVALID_ACCESS_LEDGER_"
    "NO_RETRY"
)

# Source/review artifacts never authorize runtime access.  EXECUTION_AUTHORITY
# describes the exact future authorization payload; it is inert until a
# separately reviewed, committed authorization validates.
SOURCE_ONLY_AUTHORITY = dict(_MOTION.SOURCE_ONLY_AUTHORITY)
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


def science_contract() -> dict[str, Any]:
    """Return static V3 science with only the preregistered overlap delta."""

    value = deepcopy(_base_static_science_contract())
    value["schema"] = f"{SCHEMA_PREFIX}_science_contract_v1"
    value["model_family"] = MODEL_FAMILY
    value["model_runtime_version"] = MODEL_RUNTIME_VERSION
    value["one_science_delta"] = ONE_SCIENCE_DELTA
    value["architecture_contract"] = (
        overlapping_tokenization_architecture_contract_v1()
    )
    value["architecture_contract_sha256"] = ARCHITECTURE_CONTRACT_SHA256
    value["initialization"] = {
        **value["initialization"],
        "migration_schema": INITIALIZATION_SCHEMA,
        "exact_copy_entry_count": N320_EXACT_COPY_ENTRY_COUNT,
        "transformed_entry_count": N320_TRANSFORMED_ENTRY_COUNT,
        "n320_derived_entry_count": N320_DERIVED_ENTRY_COUNT,
        "transformed_entry": N320_TRANSFORMED_ENTRY,
        "source_patch_weight_shape":
            list(N320_SOURCE_PATCH_WEIGHT_SHAPE),
        "destination_patch_weight_shape":
            list(N320_DESTINATION_PATCH_WEIGHT_SHAPE),
        "center_copy_slice": list(PATCH_CENTER_COPY_SLICE),
        "central_weight_scalar_count":
            PATCH_CENTRAL_WEIGHT_SCALAR_COUNT,
        "outer_ring_exact_zero_scalar_count":
            PATCH_OUTER_RING_SCALAR_COUNT,
        "caller_cpu_rng_restored": True,
        "fresh_optimizer_required": True,
    }
    value["parameter_counts"] = {
        **dict(EXPECTED_PARAMETER_COUNTS),
        "total_trainable": EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT,
        "complete_model": EXPECTED_COMPLETE_MODEL_PARAMETER_COUNT,
    }
    value["parameter_tensor_counts"] = {
        **dict(EXPECTED_PARAMETER_TENSOR_COUNTS),
        "total_trainable": EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT,
        "complete_model": EXPECTED_COMPLETE_MODEL_PARAMETER_TENSOR_COUNT,
    }
    return value


def lifecycle_contract() -> dict[str, Any]:
    """Retain the corrected motion lifecycle without its science mechanism."""

    value = deepcopy(_base_corrected_lifecycle_contract())
    value.update({
        "science_identity_source": "frozen_static_multiresolution_v3",
        "corrected_lifecycle_source": "causal_motion_alignment_v1",
        "execution_requires_future_exact_authorization": True,
        "prior_runtime_output_open_authorized": False,
    })
    return value


def validate_runtime_inputs(value: object) -> dict[str, Any]:
    """Admit only the frozen leaves under the overlap operation name."""

    if type(value) is not dict:
        raise PermissionError("runtime input groups changed")
    translated = deepcopy(value)
    try:
        grant = translated["raw"]["grant"]
    except (KeyError, TypeError):
        raise PermissionError("raw runtime authority changed") from None
    expected = [
        "development_rgb_decode",
        "overlapping_tokenization_training",
        "physical_checkpoint_selection",
    ]
    if type(grant) is not dict or grant.get("allowed_operations") != expected:
        raise PermissionError("raw runtime authority changed")
    grant["allowed_operations"] = [
        "development_rgb_decode",
        "causal_motion_alignment_training",
        "physical_checkpoint_selection",
    ]
    _base_validate_runtime_inputs(translated)
    return deepcopy(value)


def _validate_noncontrol_evaluation_summary(
    value: object,
) -> dict[str, Any]:
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
        or set(value) != evaluation_fields
        or type(value["scope_evaluations"]) is not dict
        or tuple(value["scope_evaluations"]) != SCOPES
    ):
        raise PermissionError("provisional evaluation summary changed")
    flat: list[float] = []
    complete = 0
    for row in value["scope_evaluations"].values():
        if (
            type(row) is not dict
            or set(row) != {"physical_margins", "passes"}
            or type(row["physical_margins"]) is not list
            or type(row["passes"]) is not bool
        ):
            raise PermissionError("provisional scope evaluation changed")
        margins: list[float] = []
        for margin in row["physical_margins"]:
            if (
                type(margin) not in (int, float)
                or not math.isfinite(float(margin))
            ):
                raise PermissionError(
                    "provisional physical margin changed"
                )
            margins.append(float(margin))
        passes = all(margin >= 0.0 for margin in margins)
        if not margins or row["passes"] is not passes:
            raise PermissionError("provisional scope pass summary changed")
        flat.extend(margins)
        complete += int(passes)
    rough = value["rough_motion"]
    if (
        len(flat) != MARGIN_COUNT
        or type(value["complete_physical_scope_count"]) is not int
        or value["complete_physical_scope_count"] != complete
        or type(value["margin_count"]) is not int
        or value["margin_count"] != len(flat)
        or type(value["passed_margin_count"]) is not int
        or value["passed_margin_count"]
        != sum(margin >= 0.0 for margin in flat)
        or type(value["total_shortfall"]) not in (int, float)
        or not math.isfinite(float(value["total_shortfall"]))
        or float(value["total_shortfall"])
        != sum(max(0.0, -margin) for margin in flat)
        or type(value["worst_margin"]) not in (int, float)
        or not math.isfinite(float(value["worst_margin"]))
        or float(value["worst_margin"]) != min(flat)
        or type(rough) is not dict
        or set(rough)
        != {
            "pixel_balanced_accuracy",
            "ground_balanced_accuracy",
            "depth_p95_m",
        }
        or any(
            type(rough[name]) not in (int, float)
            or not math.isfinite(float(rough[name]))
            for name in rough
        )
        or not 0.0 <= float(rough["pixel_balanced_accuracy"]) <= 1.0
        or not 0.0 <= float(rough["ground_balanced_accuracy"]) <= 1.0
        or float(rough["depth_p95_m"]) < 0.0
    ):
        raise PermissionError("provisional evaluation values changed")
    return dict(value)


def validate_provisional_metric(
    value: object,
    *,
    update: int,
) -> dict[str, Any]:
    """Validate a static metric without granting pre-ledger control."""

    metric_fields = {
        "update",
        "role",
        "pair_count",
        "unique_endpoint_count",
        "scopes",
        "aggregate_complete_v4_tail_depth_loss",
        "evaluation",
        "preledger_model_state_checks_pass",
        "state_sha256_before",
        "state_sha256_after",
        "frozen_state_sha256_before_and_after",
        "state_mutation_count",
    }
    if (
        type(value) is not dict
        or set(value) != metric_fields
        or value["update"] != update
        or value["role"] != "checkpoint_selection"
        or value["pair_count"] != SELECTION_ROLE_COUNTS["pairs"]
        or value["unique_endpoint_count"]
        != SELECTION_ROLE_COUNTS["unique_endpoints"]
        or type(value["scopes"]) is not dict
        or tuple(value["scopes"]) != SCOPES
        or type(value["aggregate_complete_v4_tail_depth_loss"])
        not in (int, float)
        or not math.isfinite(
            float(value["aggregate_complete_v4_tail_depth_loss"])
        )
        or float(value["aggregate_complete_v4_tail_depth_loss"]) < 0.0
        or value["preledger_model_state_checks_pass"] is not True
        or not all(
            is_sha256(value[name])
            for name in (
                "state_sha256_before",
                "state_sha256_after",
                "frozen_state_sha256_before_and_after",
            )
        )
        or value["state_sha256_before"] != value["state_sha256_after"]
        or value["state_mutation_count"] != 0
    ):
        raise PermissionError("provisional checkpoint metric changed")
    _validate_noncontrol_evaluation_summary(value["evaluation"])
    validate_no_preledger_scientific_control(value)
    return deepcopy(value)


def validate_metric_sidecar(
    value: object,
    *,
    update: int | None = None,
) -> dict[str, Any]:
    """Validate one exact static sidecar as pre-ledger inadmissible evidence."""

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
    checkpoint_fields = {
        "path",
        "file_sha256",
        "content_sha256",
        "byte_count",
        "state_sha256",
        "frozen_state_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("provisional metric-sidecar fields changed")
    core = dict(value)
    declared = core.pop("content_sha256")
    observed_update = value["update"]
    checkpoint = value["checkpoint"]
    metric = value["metric"]
    if (
        type(observed_update) is not int
        or observed_update not in CHECKPOINT_UPDATES
        or (update is not None and observed_update != update)
        or type(checkpoint) is not dict
        or set(checkpoint) != checkpoint_fields
        or checkpoint["path"] != f"checkpoints/update_{observed_update}.pt"
        or not all(
            is_sha256(checkpoint[name])
            for name in (
                "file_sha256",
                "content_sha256",
                "state_sha256",
                "frozen_state_sha256",
            )
        )
        or type(checkpoint["byte_count"]) is not int
        or checkpoint["byte_count"] <= 0
        or value["schema"] != METRIC_SIDECAR_SCHEMA
        or value["status"]
        != "PROVISIONAL_INADMISSIBLE_PENDING_FINALIZED_LEDGER_PARSE"
        or validate_provisional_metric(metric, update=observed_update)
        != metric
        or value["inline_evaluation_count"] != 1
        or value["state_mutation_count"] != 0
        or value["publication_order"]
        != [
            "cpu_snapshot",
            "inline_nonmutating_selection_evaluation",
            "atomic_mode_0444_provisional_sidecar",
            "internal_fixed_training_flow_only",
        ]
        or value["continuation"]
        != provisional_checkpoint_control(observed_update)
        or value["scientifically_admissible"] is not False
        or value["authority"] != DOWNSTREAM_DENIALS
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("provisional metric sidecar changed")
    validate_no_preledger_scientific_control(value)
    return dict(value)


# Functions inherited through the private temporal module resolve constants in
# that module.  Rebind both layers so nested validation cannot split identity.
for _target in (_MOTION, _MOTION._TEMPORAL):
    for _name, _value in tuple(globals().items()):
        if _name.isupper():
            setattr(_target, _name, _value)

_MOTION.science_contract = science_contract
_MOTION.lifecycle_contract = lifecycle_contract
_MOTION.validate_runtime_inputs = validate_runtime_inputs
_MOTION.validate_provisional_metric = validate_provisional_metric
_MOTION.validate_metric_sidecar = validate_metric_sidecar
_MOTION._TEMPORAL.science_contract = science_contract
_MOTION._TEMPORAL.lifecycle_contract = lifecycle_contract
_MOTION._TEMPORAL.validate_runtime_inputs = validate_runtime_inputs
_MOTION._TEMPORAL.validate_provisional_metric = validate_provisional_metric
_MOTION._TEMPORAL.validate_metric_sidecar = validate_metric_sidecar

_OVERRIDDEN_PUBLIC = {
    "lifecycle_contract",
    "overlapping_tokenization_architecture_contract_v1",
    "science_contract",
    "validate_metric_sidecar",
    "validate_provisional_metric",
    "validate_runtime_inputs",
}
for _name in _MOTION.__all__:
    if _name not in _OVERRIDDEN_PUBLIC and not _name.isupper():
        globals()[_name] = getattr(_MOTION, _name)

__all__ = [name for name in globals() if name.isupper()] + sorted(
    _OVERRIDDEN_PUBLIC
    | {
        name
        for name in _MOTION.__all__
        if not name.isupper()
    }
)
