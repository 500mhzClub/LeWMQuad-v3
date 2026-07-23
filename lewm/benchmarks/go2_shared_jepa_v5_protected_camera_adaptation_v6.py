"""Frozen contract for one final fresh-update-zero Camera V6 attempt."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_AUTHOR = "/root/camera_v6_implement"
SCHEMA_PREFIX = (
    "lewm_go2_shared_jepa_v5_protected_camera_adaptation_v6_"
    "final_fresh_update0_tail_depth_8k"
)
CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_shared_jepa_v5_protected_camera_adaptation_v6.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_shared_jepa_v5_protected_camera_adaptation_v6.py"
)
TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_shared_jepa_v5_protected_camera_adaptation_v6.py"
)
PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v6_"
    "final_fresh_update0_tail_depth_8k_preregistration_2026-07-23.md"
)
PREREGISTRATION_COMMIT = "48712ffe5379324847f027d10c2305e82b351397"

V5_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/go2_shared_jepa_v5_protected_camera_adaptation_v5.py"
)
V5_RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_shared_jepa_v5_protected_camera_adaptation_v5.py"
)
V5_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_shared_jepa_v5_protected_camera_adaptation_v5.py"
)
V5_SOURCE_COMMIT = "6d001171d3f79fd8703e449272416191aae0c8b5"
V5_SOURCE_SHA256 = {
    V5_CONTRACT_RELATIVE_PATH:
        "ee732e692823b3bd9e3ac1c36611c976f8961cf6f6cc694cd82d05652351b582",
    V5_RUNNER_RELATIVE_PATH:
        "3640ca35300ca36485487d6529dd352c76900c47018f7043cb165a1a078d72c4",
    V5_TEST_RELATIVE_PATH:
        "b835207f046c099f6a2450c51fe55c4a8bcf730d3f486ed1c9866e55e39cb767",
}


def _load_exact(path: str, name: str, digest: str) -> ModuleType:
    source = ROOT / path
    raw = source.read_bytes()
    if (
        source.is_symlink()
        or not source.is_file()
        or hashlib.sha256(raw).hexdigest() != digest
    ):
        raise PermissionError(f"frozen Camera V5 source changed: {path}")
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen Camera V5 source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_v5_contract = _load_exact(
    V5_CONTRACT_RELATIVE_PATH,
    "_lewm_protected_camera_adaptation_v5_contract_for_v6",
    V5_SOURCE_SHA256[V5_CONTRACT_RELATIVE_PATH],
)
_v1 = _v5_contract._v1

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "protected_camera_adaptation_v6_final_fresh_update0_tail_depth_8k"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v6_"
    "independent_review_2026-07-23.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v6_"
    "execution_authorization_2026-07-23.json"
)
REVIEW_SCHEMA = f"{SCHEMA_PREFIX}_independent_review_v1"
AUTHORIZATION_SCHEMA = f"{SCHEMA_PREFIX}_execution_authorization_v1"
RESERVATION_SCHEMA = f"{SCHEMA_PREFIX}_reservation_v1"
SNAPSHOT_SCHEMA = f"{SCHEMA_PREFIX}_camera_snapshot_v1"
METRIC_SIDECAR_SCHEMA = f"{SCHEMA_PREFIX}_checkpoint_metric_sidecar_v1"
METRICS_SCHEMA = f"{SCHEMA_PREFIX}_checkpoint_metrics_v1"
ACCESS_SCHEMA = f"{SCHEMA_PREFIX}_access_v1"
RESULT_SCHEMA = f"{SCHEMA_PREFIX}_result_v1"
COMPLETION_SCHEMA = f"{SCHEMA_PREFIX}_completion_v1"
FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v1"

TAIL_DEPTH_LOSS_RELATIVE_PATH = (
    "lewm/models/"
    "shared_observable_camera_ray_jepa_v5_protected_camera_adaptation_v4_tail_depth.py"
)
V4_PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v4_"
    "tail_depth_successor_preregistration_2026-07-15.md"
)
V4_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v4_"
    "independent_review_2026-07-15.json"
)
V4_AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v4_"
    "execution_authorization_2026-07-15.json"
)
V4_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v4_"
    "terminal_audit_2026-07-16.json"
)
V5_RECOVERY_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v5_"
    "environment_recovery_v1_terminal_audit_2026-07-23.json"
)
FIXED_EVIDENCE_SHA256 = {
    PREREGISTRATION_RELATIVE_PATH:
        "651beaacb5476e1ad0ce5d3d489acd0dba63f1ddb03df57b93d7d267e561d73b",
    V4_PREREGISTRATION_RELATIVE_PATH:
        "cada72599abfec257583986a8fb08254f9d16b8644b4062e17323da3004c81c8",
    V4_REVIEW_RELATIVE_PATH:
        "c8ef0dc4ab2f415bc757fde963094eb163315b217cc0baf2770c5421bfdf8d93",
    V4_AUTHORIZATION_RELATIVE_PATH:
        "749ab396723422b16f919bd6b8838d9dba1ce160cc2cbaa315da04bf01c80502",
    TAIL_DEPTH_LOSS_RELATIVE_PATH:
        "6fc0a114386ee2fb0ae98704a970d38a7194db192283b904138015498fb02384",
    V4_TERMINAL_AUDIT_RELATIVE_PATH:
        "5d0d4a1cf966e5f612e15da9cacbc705ace4f629183038c6743f0e2fac1b355f",
    V5_RECOVERY_TERMINAL_AUDIT_RELATIVE_PATH:
        "4284014d283a94d4a45decb9aee5164a45f35a93c36afd2e31a93685564ad5de",
}
FIXED_EVIDENCE_CONTENT_SHA256 = {
    V4_REVIEW_RELATIVE_PATH:
        "52f7b233ffbf03abdc6743954b2529f89aae5054e7034b9cbea497bb36ea8f12",
    V4_AUTHORIZATION_RELATIVE_PATH:
        "f0d1aaf0226977a6865ea86c3fc91a3f6bc3644712671234cfab2ab850f5e5a6",
    V4_TERMINAL_AUDIT_RELATIVE_PATH:
        "246e50b986316f7dc8c806960e8661cf83417fd34c0baa269d83b221cf98d5e2",
    V5_RECOVERY_TERMINAL_AUDIT_RELATIVE_PATH:
        "582325e20fa4622c9f9be1c46ae67011c4df1fc43ad2cdcc396fb5a1df6c671f",
}
FIXED_EVIDENCE_BYTE_COUNT = {
    PREREGISTRATION_RELATIVE_PATH: 13_064,
    V4_PREREGISTRATION_RELATIVE_PATH: 7_427,
    V4_REVIEW_RELATIVE_PATH: 29_717,
    V4_AUTHORIZATION_RELATIVE_PATH: 23_660,
    TAIL_DEPTH_LOSS_RELATIVE_PATH: 7_374,
    V4_TERMINAL_AUDIT_RELATIVE_PATH: 20_077,
    V5_RECOVERY_TERMINAL_AUDIT_RELATIVE_PATH: 20_914,
}

UPDATE0_STATE_SHA256 = _v5_contract.UPDATE0_STATE_SHA256
N320_CHECKPOINT_CONTENT_SHA256 = _v5_contract.N320_CHECKPOINT_CONTENT_SHA256
MAXIMUM_UPDATE = _v5_contract.MAXIMUM_UPDATE
CHECKPOINT_UPDATES = tuple(_v5_contract.CHECKPOINT_UPDATES)
CHECKPOINT_SCHEDULE_PREFIX_SHA256 = dict(
    _v5_contract.CHECKPOINT_SCHEDULE_PREFIX_SHA256
)
SCHEDULE_PREFIX_INDICES_SHA256 = _v5_contract.SCHEDULE_PREFIX_INDICES_SHA256
PRESENTATION_COUNT = _v5_contract.PRESENTATION_COUNT
MARGIN_COUNT = _v5_contract.MARGIN_COUNT
ENCODER_LR_SCALE = _v5_contract.ENCODER_LR_SCALE
TRAINABLE_PARAMETER_PREFIXES = tuple(_v5_contract.TRAINABLE_PARAMETER_PREFIXES)
FROZEN_STATE_PREFIXES = tuple(_v5_contract.FROZEN_STATE_PREFIXES)
EXPECTED_PARAMETER_COUNTS = dict(_v5_contract.EXPECTED_PARAMETER_COUNTS)
EXPECTED_PARAMETER_TENSOR_COUNTS = dict(
    _v5_contract.EXPECTED_PARAMETER_TENSOR_COUNTS
)
OPTIMIZER_CONTRACT = copy.deepcopy(_v5_contract.OPTIMIZER_CONTRACT)
POST_CLIP_NORM_ASSERTION_TOLERANCE = (
    _v5_contract.POST_CLIP_NORM_ASSERTION_TOLERANCE
)
DOWNSTREAM_DENIALS = dict(_v5_contract.DOWNSTREAM_DENIALS)
CONTROL_ACTION_CONTINUE = _v5_contract.CONTROL_ACTION_CONTINUE
CONTROL_ACTION_QUALIFY = _v5_contract.CONTROL_ACTION_QUALIFY
CONTROL_ACTION_STOP_PROGRESS = _v5_contract.CONTROL_ACTION_STOP_PROGRESS
CONTROL_ACTION_STOP_MAXIMUM = _v5_contract.CONTROL_ACTION_STOP_MAXIMUM
METRIC_SIDECAR_DIRECTORY = _v5_contract.METRIC_SIDECAR_DIRECTORY
TERMINAL_DIRECTORIES_INCLUDING_ROOT = tuple(
    _v5_contract.TERMINAL_DIRECTORIES_INCLUDING_ROOT
)

TAIL_DEPTH_DEFINITION = {
    "finite_hit_bin_count": 64,
    "conditional_first_hit_mass": True,
    "predicted_depth": "frozen_bin_center_plus_existing_per_bin_offset",
    "target_rays": "represented_in_range_hits",
    "normalized_by_depth_p95_ceiling_m": 0.25,
    "reduction":
        "mean_largest_ceil_0.05_times_N_per_real_B4_current_or_next_frame",
    "objective_slot_weight": 0.25,
}
REVIEW_AUTHORITY = {
    **dict(_v5_contract.REVIEW_AUTHORITY),
    "final_fresh_update0_tail_depth_8k_authorized": False,
    "visibility_preflight_authorized": False,
}
EXECUTION_AUTHORITY = {
    **dict(_v5_contract.EXECUTION_AUTHORITY),
    "mutation_scope": OUTPUT_ROOT_RELATIVE_PATH,
    "final_fresh_update0_tail_depth_8k_authorized": True,
    "visibility_preflight_authorized": True,
}
SOURCE_PATHS = tuple(dict.fromkeys((
    CONTRACT_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    *_v5_contract.SOURCE_PATHS,
    *FIXED_EVIDENCE_SHA256,
)))


def canonical_json_bytes(value: object) -> bytes:
    return _v5_contract.canonical_json_bytes(value)


def canonical_json_sha256(value: object) -> str:
    return _v5_contract.canonical_json_sha256(value)


def with_content_sha256(core: Mapping[str, Any]) -> dict[str, Any]:
    return _v5_contract.with_content_sha256(core)


def parse_canonical_json(raw: bytes, *, name: str) -> dict[str, Any]:
    return _v5_contract.parse_canonical_json(raw, name=name)


def artifact_binding(
    path: str, raw: bytes, *, content_sha256: str
) -> dict[str, Any]:
    return _v5_contract.artifact_binding(
        path, raw, content_sha256=content_sha256
    )


def validate_binding(
    value: object, *, path: str | None = None
) -> dict[str, Any]:
    return _v5_contract.validate_binding(value, path=path)


def visibility_preflight_contract() -> dict[str, Any]:
    return {
        "status": "PASS_exactly_one_visible_discrete_R9700",
        "python_flags": ["-I", "-B"],
        "selector_environment": {
            "HIP_VISIBLE_DEVICES": "0",
            "ROCR_VISIBLE_DEVICES": "absent",
            "CUDA_VISIBLE_DEVICES": "absent",
            "HSA_OVERRIDE_GFX_VERSION": "absent",
        },
        "native_thread_environment": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
        },
        "torch_cuda_available": True,
        "visible_device_count": 1,
        "visible_device_index": 0,
        "visible_device_name": "AMD Radeon AI PRO R9700",
        "normalized_device_name_must_contain": "r9700",
        "tensor_allocation_count": 0,
        "checkpoint_open_count": 0,
        "dataset_open_count": 0,
        "rgb_open_count": 0,
        "selection_navigation_or_heldout_open_count": 0,
        "output_root_absent": True,
        "other_generated_mutator_active": False,
        "other_kfd_training_process_active": False,
        "competing_gpu_work_active": False,
        "gpu_management_query_between_probe_and_launch_authorized": False,
    }


def expected_visibility_preflight() -> dict[str, Any]:
    return visibility_preflight_contract()


def _fixed_evidence_bindings() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path, file_sha256 in FIXED_EVIDENCE_SHA256.items():
        value: dict[str, Any] = {
            "path": path,
            "file_sha256": file_sha256,
            "byte_count": FIXED_EVIDENCE_BYTE_COUNT[path],
        }
        if path in FIXED_EVIDENCE_CONTENT_SHA256:
            value["content_sha256"] = FIXED_EVIDENCE_CONTENT_SHA256[path]
        result[path] = value
    return result


def evidence_contract() -> dict[str, Any]:
    return {
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "v5_source_commit": V5_SOURCE_COMMIT,
        "v5_source_sha256": dict(V5_SOURCE_SHA256),
        "fixed_artifacts": _fixed_evidence_bindings(),
        "v5_recovery_terminal": {
            "status": "failed_predeclared_numeric_progress_cutoff",
            "last_completed_update": 1_000,
            "passed_margin_count": 106,
            "total_shortfall": 49.13255561472496,
            "worst_margin": -7.945521640777587,
            "qualified_checkpoint_exists": False,
            "g2_navigation_or_heldout_attempted": False,
        },
        "v4_tail_depth_progress": {
            "100": {
                "passed_margin_count": 61,
                "total_shortfall": 112.38092829435729,
                "worst_margin": -8.20910987854004,
            },
            "400": {
                "passed_margin_count": 84,
                "total_shortfall": 63.3565430408583,
                "worst_margin": -5.343397927284242,
            },
            "1000": {
                "passed_margin_count": 97,
                "total_shortfall": 41.00174362036205,
                "worst_margin": -5.476026201248172,
            },
        },
        "visibility_preflight": visibility_preflight_contract(),
        "checkpoint_deserialization_during_source_review_authorized": False,
    }


def predecessor_contract() -> dict[str, Any]:
    return {
        "fresh_update0_initialization_not_v1_through_v5_continuation": True,
        "update0_state_sha256": UPDATE0_STATE_SHA256,
        "n320_checkpoint_file_sha256":
            _v5_contract.FIXED_EVIDENCE_SHA256[
                _v5_contract.N320_CHECKPOINT_RELATIVE_PATH
            ],
        "n320_checkpoint_content_sha256": N320_CHECKPOINT_CONTENT_SHA256,
        "v5_predecessor": _v5_contract.predecessor_contract(),
        "v5_recovery_terminal_audit":
            _fixed_evidence_bindings()[
                V5_RECOVERY_TERMINAL_AUDIT_RELATIVE_PATH
            ],
        "v1_through_v5_camera_checkpoint_or_optimizer_load_authorized": False,
        "retry_resume_recovery_or_optimizer_reconstruction_authorized": False,
        "evidence": evidence_contract(),
    }


def science_contract() -> dict[str, Any]:
    value = copy.deepcopy(_v5_contract.science_contract())
    value["camera_loss"] = {
        **value["camera_loss"],
        "source": TAIL_DEPTH_LOSS_RELATIVE_PATH,
        "terms": [
            "hierarchical_first_hit_nll",
            "tail_depth_p95_cvar",
            "ground_clear_distance_state_balanced_bce",
            "derived_raster_hierarchical_bce",
            "derived_raster_cell_nll",
        ],
        "tail_depth_p95_cvar": copy.deepcopy(TAIL_DEPTH_DEFINITION),
    }
    return value


def science_delta() -> dict[str, Any]:
    base = _v5_contract.science_contract()
    return {
        "base_v5_science_contract_sha256": canonical_json_sha256(base),
        "training_science_change_count": 1,
        "training_science_change":
            "exact_reuse_of_previously_executed_v4_tail_depth_objective_slot",
        "changed_objective_slot": "camera_loss.target_bin_offset_smooth_l1",
        "before": "target_bin_offset_smooth_l1_at_target_bin",
        "after": "tail_depth_p95_cvar_over_conditional_finite_hit_distribution",
        "slot_weight_before_and_after": 0.25,
        "contract_leaf_changes_encoding_that_one_slot_replacement": [
            {
                "path": "camera_loss.source",
                "before": base["camera_loss"]["source"],
                "after": TAIL_DEPTH_LOSS_RELATIVE_PATH,
            },
            {
                "path": "camera_loss.terms[1]",
                "before": "target_bin_offset_smooth_l1",
                "after": "tail_depth_p95_cvar",
            },
            {
                "path": "camera_loss.tail_depth_p95_cvar",
                "before": None,
                "after": copy.deepcopy(TAIL_DEPTH_DEFINITION),
            },
        ],
        "schedule_change_relative_to_v5": False,
        "schedule_change_relative_to_v4":
            "use_existing_v5_8000_update_128000_presentation_schedule",
        "architecture_data_sampling_seed_initialization_optimizer_or_coefficient_changes": [],
        "other_training_science_changes": [],
    }


def control_contract() -> dict[str, Any]:
    return {
        "precedence": [
            "integrity_failure_is_terminal",
            "earliest_all_nine_physical_pass_qualifies",
            "fixed_checkpoint_control",
        ],
        "margin_statistics": {
            "count": MARGIN_COUNT,
            "P": "count(m>=0)",
            "S": "sum(max(0,-m))",
            "W": "min(m)",
        },
        "loss_policy":
            "finite_integrity_value_only_never_compared_for_continuation",
        "update_100":
            "finite_state_metrics_92_gradients_frozen_hash_trainable_movement_"
            "and_189_margins_required_then_qualify_or_continue",
        "update_400_continue_if":
            "P400>=P100+5_or_S400<=0.90*S100_from_immutable_same_run_sidecar",
        "update_1000_continue_if":
            "P1000>=P400+5_or_S1000<=0.90*S400_from_immutable_same_run_sidecar",
        "updates_4000_and_6000":
            "informational_then_qualify_or_continue_without_numeric_cutoff",
        "update_8000":
            "only_exact_all_nine_and_189_of_189_nonnegative_margins_qualifies_"
            "otherwise_stop",
        "equality_continues": True,
        "observer_policy":
            "read_only_completed_mode_0444_sidecars_only_no_checkpoint_load_or_"
            "evaluation_rerun",
        "retry_resume_extension_threshold_relaxation_or_soft_promotion_authorized":
            False,
    }


def metric_sidecar_path(update: int) -> str:
    return _v5_contract.metric_sidecar_path(update)


def validate_checkpoint_prefix(updates: Sequence[int]) -> tuple[int, ...]:
    return _v5_contract.validate_checkpoint_prefix(updates)


def expected_metric_sidecar_paths(
    updates: Sequence[int],
) -> tuple[str, ...]:
    return _v5_contract.expected_metric_sidecar_paths(updates)


def reporting_contract() -> dict[str, Any]:
    value = copy.deepcopy(_v5_contract.reporting_contract())
    value["numeric_continuation_rule"] = (
        "V6_same_run_P_gain_or_S_reduction_health_checks_at_updates_400_and_1000"
    )
    value["numeric_progress_cutoff_updates"] = [400, 1_000]
    value["terminal_physical_gate_update"] = 8_000
    return value


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    inherited = _v5_contract.current_source_bindings(root)
    if any(inherited.get(path) != digest for path, digest in V5_SOURCE_SHA256.items()):
        raise PermissionError("frozen Camera V5 source binding changed")
    result = dict(inherited)
    for path in SOURCE_PATHS:
        if path in result:
            continue
        source = root / path
        if source.is_symlink() or not source.is_file():
            raise PermissionError(f"reviewed V6 input is not one regular file: {path}")
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if path in FIXED_EVIDENCE_SHA256:
            if (
                digest != FIXED_EVIDENCE_SHA256[path]
                or len(raw) != FIXED_EVIDENCE_BYTE_COUNT[path]
            ):
                raise PermissionError(f"fixed Camera V6 evidence changed: {path}")
            expected_content = FIXED_EVIDENCE_CONTENT_SHA256.get(path)
            if expected_content is not None:
                parsed = parse_canonical_json(
                    raw, name=f"fixed Camera V6 evidence {path}"
                )
                if parsed.get("content_sha256") != expected_content:
                    raise PermissionError(
                        f"fixed Camera V6 evidence content changed: {path}"
                    )
        result[path] = digest
    if set(result) != set(SOURCE_PATHS):
        raise PermissionError("protected Camera V6 source closure changed")
    return result


def validate_review(
    value: object, *, expected_sources: Mapping[str, str]
) -> dict[str, Any]:
    fields = {
        "schema", "status", "implementation_author", "reviewer",
        "reviewed_sources", "predecessor", "science_contract", "science_delta",
        "evidence", "visibility_preflight", "reporting_contract",
        "control_contract", "source_only", "findings", "authority",
        "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("Camera V6 independent-review fields changed")
    core, declared = dict(value), value["content_sha256"]
    core.pop("content_sha256")
    reviewer = value["reviewer"]
    if (
        value["schema"] != REVIEW_SCHEMA
        or value["status"] != "PASS"
        or value["implementation_author"] != IMPLEMENTATION_AUTHOR
        or type(reviewer) is not str
        or not reviewer.startswith("/root/")
        or reviewer == IMPLEMENTATION_AUTHOR
        or value["reviewed_sources"] != dict(expected_sources)
        or value["predecessor"] != predecessor_contract()
        or value["science_contract"] != science_contract()
        or value["science_delta"] != science_delta()
        or value["evidence"] != evidence_contract()
        or value["visibility_preflight"] != visibility_preflight_contract()
        or value["reporting_contract"] != reporting_contract()
        or value["control_contract"] != control_contract()
        or value["source_only"] is not True
        or value["findings"] != []
        or value["authority"] != REVIEW_AUTHORITY
        or not _v1.is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError(
            "independent review did not pass exact Camera V6 sources"
        )
    return dict(value)


def validate_authorization(
    value: object,
    *,
    review_binding: Mapping[str, Any],
    reviewer: str,
) -> dict[str, Any]:
    fields = {
        "schema", "status", "authorizer", "independent_review", "predecessor",
        "raw", "camera", "experiment", "science_delta", "evidence",
        "visibility_preflight", "reporting_contract", "control_contract",
        "authority", "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("Camera V6 execution-authorization fields changed")
    core, declared = dict(value), value["content_sha256"]
    core.pop("content_sha256")
    authorizer = value["authorizer"]
    if (
        value["schema"] != AUTHORIZATION_SCHEMA
        or value["status"]
        != "authorized_one_exact_camera_v6_final_fresh_update0_tail_depth_8k_attempt"
        or type(authorizer) is not str
        or not authorizer.startswith("/root/")
        or authorizer in {IMPLEMENTATION_AUTHOR, reviewer}
        or value["independent_review"] != dict(review_binding)
        or value["predecessor"] != predecessor_contract()
        or value["raw"] != expected_raw_authority()
        or value["camera"] != expected_camera_authority()
        or value["experiment"] != science_contract()
        or value["science_delta"] != science_delta()
        or value["evidence"] != evidence_contract()
        or value["visibility_preflight"] != visibility_preflight_contract()
        or value["reporting_contract"] != reporting_contract()
        or value["control_contract"] != control_contract()
        or value["authority"] != EXECUTION_AUTHORITY
        or not _v1.is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("Camera V6 execution authorization changed")
    return dict(value)


def learning_rates(update: int) -> tuple[float, float]:
    return _v5_contract.learning_rates(update)


def parameter_partition(name: str) -> str:
    return _v5_contract.parameter_partition(name)


def evaluate_physical_scopes(
    scopes: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    return _v5_contract.evaluate_physical_scopes(scopes)


def _validate_health_baseline(
    value: object, *, expected_update: int
) -> dict[str, Any]:
    fields = {
        "update", "path", "file_sha256", "content_sha256",
        "passed_margin_count", "total_shortfall",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("same-run health baseline fields changed")
    if (
        value["update"] != expected_update
        or value["path"] != metric_sidecar_path(expected_update)
        or not _v1.is_sha256(value["file_sha256"])
        or not _v1.is_sha256(value["content_sha256"])
        or type(value["passed_margin_count"]) is not int
        or not 0 <= value["passed_margin_count"] <= MARGIN_COUNT
        or type(value["total_shortfall"]) not in (int, float)
        or not math.isfinite(float(value["total_shortfall"]))
        or float(value["total_shortfall"]) < 0.0
    ):
        raise PermissionError("same-run health baseline changed")
    return {
        **dict(value),
        "total_shortfall": float(value["total_shortfall"]),
    }


def control_decision_from_progress(
    *,
    update: int,
    passed_margin_count: int,
    total_shortfall: float,
    worst_margin: float,
    aggregate_complete_v4_loss: float,
    all_nine_physical_pass: bool,
    same_run_health_baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if type(update) is not int or update not in CHECKPOINT_UPDATES:
        raise ValueError("checkpoint control update is not fixed")
    if (
        type(passed_margin_count) is not int
        or not 0 <= passed_margin_count <= MARGIN_COUNT
    ):
        raise ValueError("passed physical margin count is invalid")
    if type(all_nine_physical_pass) is not bool:
        raise TypeError("all-nine physical decision is not Boolean")
    scalars = (total_shortfall, worst_margin, aggregate_complete_v4_loss)
    if any(
        type(value) not in (int, float) or not math.isfinite(float(value))
        for value in scalars
    ):
        raise FloatingPointError("checkpoint progress scalar became nonfinite")
    shortfall, worst, loss = map(float, scalars)
    if shortfall < 0.0:
        raise ValueError("checkpoint total shortfall became negative")
    expected_baseline = {400: 100, 1_000: 400}.get(update)
    baseline = (
        _validate_health_baseline(
            same_run_health_baseline, expected_update=expected_baseline
        )
        if expected_baseline is not None
        else None
    )
    if expected_baseline is None and same_run_health_baseline is not None:
        raise PermissionError(
            "same-run health baseline appeared at the wrong checkpoint"
        )
    statistics = {
        "margin_count": MARGIN_COUNT,
        "passed_margin_count": passed_margin_count,
        "total_shortfall": shortfall,
        "worst_margin": worst,
        "aggregate_complete_v4_loss": loss,
        "all_nine_physical_pass": all_nine_physical_pass,
        "same_run_health_baseline": baseline,
    }
    if all_nine_physical_pass:
        action = CONTROL_ACTION_QUALIFY
        reason = (
            "earliest fixed checkpoint passed every physical margin in all "
            "nine scopes"
        )
        terminal_stage = "earliest_all_nine_physical_pass"
        next_update = None
    elif update == 100:
        action = CONTROL_ACTION_CONTINUE
        reason = "update 100 functionality spotcheck passed integrity"
        terminal_stage = None
        next_update = 400
    elif update in (400, 1_000):
        assert baseline is not None
        p_progress = (
            passed_margin_count >= baseline["passed_margin_count"] + 5
        )
        s_progress = shortfall <= 0.90 * baseline["total_shortfall"]
        keep = p_progress or s_progress
        action = (
            CONTROL_ACTION_CONTINUE if keep else CONTROL_ACTION_STOP_PROGRESS
        )
        reason = (
            f"update {update} met the same-run P-gain or S-reduction health check"
            if keep
            else f"update {update} failed both same-run coarse health checks"
        )
        terminal_stage = (
            None
            if keep
            else f"predeclared_numeric_progress_cutoff_at_update_{update}"
        )
        next_update = (
            CHECKPOINT_UPDATES[CHECKPOINT_UPDATES.index(update) + 1]
            if keep
            else None
        )
    elif update in (4_000, 6_000):
        action = CONTROL_ACTION_CONTINUE
        reason = "informational fixed checkpoint continues because it did not qualify"
        terminal_stage = None
        next_update = CHECKPOINT_UPDATES[CHECKPOINT_UPDATES.index(update) + 1]
    else:
        action = CONTROL_ACTION_STOP_MAXIMUM
        reason = "update 8000 did not pass the unchanged all-nine physical gate"
        terminal_stage = "scientific_numeric_physical_gate_at_update_8000"
        next_update = None
    return {
        "schema": f"{SCHEMA_PREFIX}_checkpoint_control_decision_v1",
        "update": update,
        "statistics": statistics,
        "action": action,
        "reason": reason,
        "qualifies": action == CONTROL_ACTION_QUALIFY,
        "terminal_stage": terminal_stage,
        "next_checkpoint_update": next_update,
        "control_contract_sha256": canonical_json_sha256(control_contract()),
    }


def checkpoint_progress(metric: Mapping[str, Any]) -> dict[str, Any]:
    progress = _v5_contract._v3_contract.checkpoint_progress(metric)
    progress["same_run_health_baseline"] = metric.get(
        "same_run_health_baseline"
    )
    return progress


def checkpoint_control_decision(
    metric: Mapping[str, Any]
) -> dict[str, Any]:
    progress = checkpoint_progress(metric)
    if progress["update"] == 100:
        before = metric.get("state_sha256_before")
        after = metric.get("state_sha256_after")
        frozen = metric.get("frozen_state_sha256_before_and_after")
        if (
            not _v1.is_sha256(before)
            or after != before
            or before == UPDATE0_STATE_SHA256
            or not _v1.is_sha256(frozen)
            or metric.get("state_mutation_count") != 0
        ):
            raise PermissionError(
                "update 100 integrity or trainable-state movement failed"
            )
    return control_decision_from_progress(**progress)


def validate_metric_sidecar(
    value: object,
    *,
    update: int | None = None,
    checkpoint: Mapping[str, Any] | None = None,
    metric: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fields = {
        "schema", "status", "update", "checkpoint", "metric",
        "inline_evaluation_count", "state_mutation_count", "publication",
        "continuation", "authority", "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("checkpoint metric sidecar fields changed")
    core, declared = dict(value), value["content_sha256"]
    core.pop("content_sha256")
    observed_update = value["update"]
    observed_metric = value["metric"]
    if (
        type(observed_update) is not int
        or observed_update not in CHECKPOINT_UPDATES
        or value["schema"] != METRIC_SIDECAR_SCHEMA
        or value["status"]
        != "published_after_inline_nonmutating_physical_evaluation_before_control_branch"
        or type(value["checkpoint"]) is not dict
        or type(observed_metric) is not dict
        or observed_metric.get("update") != observed_update
        or observed_metric.get("role") != "checkpoint_selection"
        or observed_metric.get("state_mutation_count") != 0
        or value["inline_evaluation_count"] != 1
        or value["state_mutation_count"] != 0
        or value["publication"] != reporting_contract()["publication_order"]
        or value["continuation"] != checkpoint_control_decision(observed_metric)
        or value["authority"] != {
            "read_only_observation_authorized": True,
            "observer_evaluation_rerun_authorized": False,
            "only_predeclared_metric_control_authorized": True,
            "g2_navigation_or_heldout_use_authorized": False,
        }
        or (update is not None and observed_update != update)
        or (
            checkpoint is not None
            and value["checkpoint"] != dict(checkpoint)
        )
        or (metric is not None and observed_metric != dict(metric))
        or not _v1.is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("checkpoint metric sidecar changed")
    return dict(value)


def expected_raw_authority() -> dict[str, Any]:
    return _v5_contract.expected_raw_authority()


def expected_camera_authority() -> dict[str, Any]:
    return _v5_contract.expected_camera_authority()


def __getattr__(name: str) -> Any:
    """Delegate unchanged V5 training and lifecycle constants."""
    return getattr(_v5_contract, name)


__all__ = [name for name in globals() if name.isupper()] + [
    "artifact_binding", "canonical_json_bytes", "canonical_json_sha256",
    "checkpoint_control_decision", "checkpoint_progress", "control_contract",
    "control_decision_from_progress", "current_source_bindings",
    "evidence_contract", "evaluate_physical_scopes",
    "expected_camera_authority", "expected_metric_sidecar_paths",
    "expected_raw_authority", "expected_visibility_preflight",
    "learning_rates", "metric_sidecar_path", "parameter_partition",
    "parse_canonical_json", "predecessor_contract", "reporting_contract",
    "science_contract", "science_delta", "validate_authorization",
    "validate_binding", "validate_checkpoint_prefix",
    "validate_metric_sidecar", "validate_review",
    "visibility_preflight_contract", "with_content_sha256",
]
