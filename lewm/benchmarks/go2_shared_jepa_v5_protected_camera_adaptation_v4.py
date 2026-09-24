"""Frozen contract for one preregistered Camera V4 tail-depth attempt."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_AUTHOR = "/root/camera_v4_implement"
SCHEMA_PREFIX = "lewm_go2_shared_jepa_v5_protected_camera_adaptation_v4"
CONTRACT_RELATIVE_PATH = "lewm/benchmarks/go2_shared_jepa_v5_protected_camera_adaptation_v4.py"
RUNNER_RELATIVE_PATH = "scripts/run_go2_shared_jepa_v5_protected_camera_adaptation_v4.py"
TEST_RELATIVE_PATH = "lewm/tests/test_go2_shared_jepa_v5_protected_camera_adaptation_v4.py"
LOSS_RELATIVE_PATH = "lewm/models/shared_observable_camera_ray_jepa_v5_protected_camera_adaptation_v4_tail_depth.py"
PREREGISTRATION_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v4_tail_depth_successor_preregistration_2026-07-15.md"

V3_CONTRACT_RELATIVE_PATH = "lewm/benchmarks/go2_shared_jepa_v5_protected_camera_adaptation_v3.py"
V3_RUNNER_RELATIVE_PATH = "scripts/run_go2_shared_jepa_v5_protected_camera_adaptation_v3.py"
V3_TEST_RELATIVE_PATH = "lewm/tests/test_go2_shared_jepa_v5_protected_camera_adaptation_v3.py"
V3_SOURCE_SHA256 = {
    V3_CONTRACT_RELATIVE_PATH: "9fd912538d94944881bd8a2789023470345208abb55a94466fbecc9d82afa0be",
    V3_RUNNER_RELATIVE_PATH: "921f88a149940adb9df79684bdc810f87a97fe657b4d321a4df76c69d3f4af61",
    V3_TEST_RELATIVE_PATH: "06f0cf6d1fe74b59f9de880d9ee9adcc27d8863537c3170f3163946c1eb9d656",
}


def _load_exact(path: str, name: str, digest: str) -> ModuleType:
    source = ROOT / path
    raw = source.read_bytes()
    if source.is_symlink() or not source.is_file() or hashlib.sha256(raw).hexdigest() != digest:
        raise PermissionError(f"frozen Camera V3 source changed: {path}")
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen Camera V3 source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_v3_contract = _load_exact(
    V3_CONTRACT_RELATIVE_PATH,
    "_lewm_protected_camera_adaptation_v3_contract_for_v4",
    V3_SOURCE_SHA256[V3_CONTRACT_RELATIVE_PATH],
)
_v1 = _v3_contract._v1

OUTPUT_ROOT_RELATIVE_PATH = ".generated/go2_shared_observable_camera_ray_jepa_v5/protected_camera_adaptation_v4"
REVIEW_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v4_independent_review_2026-07-15.json"
AUTHORIZATION_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v4_execution_authorization_2026-07-15.json"
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

V3_TERMINAL_AUDIT_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v3_terminal_audit_2026-07-15.json"
V3_WARMSTART_BLOCK_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v3_warmstart_science_review_2026-07-15.json"
ORACLE_AUDIT_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_protected_camera_physical_gate_oracle_v1_terminal_audit_2026-07-15.json"
ORACLE_RESULT_RELATIVE_PATH = ".generated/go2_shared_observable_camera_ray_jepa_v5/protected_camera_physical_gate_oracle_v1/result.json"
V3_SIDECAR_ROOT = ".generated/go2_shared_observable_camera_ray_jepa_v5/protected_camera_adaptation_v3/checkpoints"
FIXED_EVIDENCE_SHA256 = {
    PREREGISTRATION_RELATIVE_PATH: "cada72599abfec257583986a8fb08254f9d16b8644b4062e17323da3004c81c8",
    V3_TERMINAL_AUDIT_RELATIVE_PATH: "3eb77a83ede536680e03363521f73f41205ac17d845a0e28251a40dcf82f77ab",
    V3_WARMSTART_BLOCK_RELATIVE_PATH: "b37829a2c311533240f6191c099d79411d453adbde43cd0304f1e5c74bd676d7",
    "docs/lewm_go2_shared_jepa_v5_protected_camera_physical_gate_oracle_v1_independent_review_v2_2026-07-15.json": "1b4345911f51bcb60e472a366b1b3b68858e9d82673a357a0a75cd81d72c41d6",
    "docs/lewm_go2_shared_jepa_v5_protected_camera_physical_gate_oracle_v1_execution_authorization_2026-07-15.json": "5a5b9e5bb04e1218614ce84d54ca286989d111a090f2f0d1f634b4c112fd6246",
    ORACLE_AUDIT_RELATIVE_PATH: "a899c199cb03be09c795eac0203747e6e1e507cca6d2e4f5a5b9db3b41435dee",
    "lewm/benchmarks/go2_shared_jepa_v5_protected_camera_physical_gate_oracle_v1.py": "d34d9475eb79e228f3d7d3b1511e93c2c31c9900a16d2b792a910874766be773",
    "scripts/run_go2_shared_jepa_v5_protected_camera_physical_gate_oracle_v1.py": "1873df123b5d4b48fc5bdb0e24a05b596b96537c9b58f0b6e4fd5a1ae2ac0084",
    ORACLE_RESULT_RELATIVE_PATH: "ce23d00fab6b5be3b222b837cc70635ebdc5955ca82bf738ba7ef0d9731e24f8",
    f"{V3_SIDECAR_ROOT}/update_100.metrics.json": "33104dcfa12bd90cc3db0366059a06b5adf84b6b440deb6181b0a618221d930d",
    f"{V3_SIDECAR_ROOT}/update_400.metrics.json": "c53711248f70482ed790484591503741255c7a5a9d2429d165d7c9c42f0be31a",
    f"{V3_SIDECAR_ROOT}/update_1000.metrics.json": "26f5e06d141b974b335d7f056b5392bd308342082bf832acfbd83f70b451e926",
    f"{V3_SIDECAR_ROOT}/update_2000.metrics.json": "28fb55ed2c679d8af84ecdff4159e52832a2337c95fc8c10db60a683567f4b7a",
    f"{V3_SIDECAR_ROOT}/update_4000.metrics.json": "5b83a880d13983c398083525fb05d939673cad2a86ec38596a7f279670cf1a05",
}
BASELINE_PROGRESS = {
    1_000: {"passed_margin_count": 106, "total_shortfall": 49.09939462151839, "worst_margin": -7.944758415222166},
    2_000: {"passed_margin_count": 121, "total_shortfall": 30.06221418748834, "worst_margin": -5.833248805999755},
}
PERSISTED_BASELINE_PROGRESS = {
    str(update): dict(values) for update, values in BASELINE_PROGRESS.items()
}
TAIL_DEPTH_DEFINITION = {
    "finite_hit_bin_count": 64,
    "conditional_first_hit_mass": True,
    "predicted_depth": "frozen_bin_center_plus_existing_per_bin_offset",
    "target_rays": "represented_in_range_hits",
    "normalized_by_depth_p95_ceiling_m": 0.25,
    "reduction": "mean_largest_ceil_0.05_times_N_per_real_B4_current_or_next_frame",
    "objective_slot_weight": 0.25,
}
UPDATE0_STATE_SHA256 = "e03613bf5da2d93910630a0e2b98799a907f9a2b4767a0c2c36b1fa942cd2a87"
MARGIN_COUNT = 189
MAXIMUM_UPDATE = _v3_contract.MAXIMUM_UPDATE
CHECKPOINT_UPDATES = tuple(_v3_contract.CHECKPOINT_UPDATES)
TRAINABLE_PARAMETER_PREFIXES = tuple(_v3_contract.TRAINABLE_PARAMETER_PREFIXES)
FROZEN_STATE_PREFIXES = tuple(_v3_contract.FROZEN_STATE_PREFIXES)
EXPECTED_PARAMETER_COUNTS = dict(_v3_contract.EXPECTED_PARAMETER_COUNTS)
EXPECTED_PARAMETER_TENSOR_COUNTS = dict(_v3_contract.EXPECTED_PARAMETER_TENSOR_COUNTS)
OPTIMIZER_CONTRACT = copy.deepcopy(_v3_contract.OPTIMIZER_CONTRACT)
POST_CLIP_NORM_ASSERTION_TOLERANCE = _v3_contract.POST_CLIP_NORM_ASSERTION_TOLERANCE
DOWNSTREAM_DENIALS = dict(_v3_contract.DOWNSTREAM_DENIALS)
REVIEW_AUTHORITY = dict(_v3_contract.REVIEW_AUTHORITY)
EXECUTION_AUTHORITY = {**dict(_v3_contract.EXECUTION_AUTHORITY), "mutation_scope": OUTPUT_ROOT_RELATIVE_PATH}
CONTROL_ACTION_CONTINUE = _v3_contract.CONTROL_ACTION_CONTINUE
CONTROL_ACTION_QUALIFY = _v3_contract.CONTROL_ACTION_QUALIFY
CONTROL_ACTION_STOP_PROGRESS = _v3_contract.CONTROL_ACTION_STOP_PROGRESS
CONTROL_ACTION_STOP_MAXIMUM = _v3_contract.CONTROL_ACTION_STOP_MAXIMUM

SOURCE_PATHS = tuple(dict.fromkeys((
    CONTRACT_RELATIVE_PATH, RUNNER_RELATIVE_PATH, TEST_RELATIVE_PATH, LOSS_RELATIVE_PATH,
    *_v3_contract.SOURCE_PATHS, *FIXED_EVIDENCE_SHA256,
)))


def canonical_json_bytes(value: object) -> bytes:
    return _v3_contract.canonical_json_bytes(value)


def canonical_json_sha256(value: object) -> str:
    return _v3_contract.canonical_json_sha256(value)


def with_content_sha256(core: Mapping[str, Any]) -> dict[str, Any]:
    return _v3_contract.with_content_sha256(core)


def parse_canonical_json(raw: bytes, *, name: str) -> dict[str, Any]:
    return _v3_contract.parse_canonical_json(raw, name=name)


def artifact_binding(path: str, raw: bytes, *, content_sha256: str) -> dict[str, Any]:
    return _v3_contract.artifact_binding(path, raw, content_sha256=content_sha256)


def evidence_contract() -> dict[str, Any]:
    return {
        "preregistration_commit": "0fdf1b163394aefa1a0a3731f9609ba4fa314f77",
        "fixed_file_sha256": dict(FIXED_EVIDENCE_SHA256),
        "v3_progress_baselines": copy.deepcopy(PERSISTED_BASELINE_PROGRESS),
        "v3_warm_start_authorized": False,
        "unchanged_physical_gate_attainable_positive_control_only": True,
        "learned_checkpoint_qualified_by_oracle": False,
    }


def predecessor_contract() -> dict[str, Any]:
    return {
        "fresh_update0_initialization_not_v1_v2_v3_continuation": True,
        "update0_state_sha256": UPDATE0_STATE_SHA256,
        "v3_predecessor": _v3_contract.predecessor_contract(),
        "v3_terminal_audit_file_sha256": FIXED_EVIDENCE_SHA256[V3_TERMINAL_AUDIT_RELATIVE_PATH],
        "v3_qualified_checkpoint_exists": False,
        "v3_warm_start_authorized": False,
        "evidence": evidence_contract(),
    }


def science_contract() -> dict[str, Any]:
    value = copy.deepcopy(_v3_contract.science_contract())
    value["camera_loss"] = {
        **value["camera_loss"],
        "source": LOSS_RELATIVE_PATH,
        "terms": [
            "hierarchical_first_hit_nll", "tail_depth_p95_cvar",
            "ground_clear_distance_state_balanced_bce",
            "derived_raster_hierarchical_bce", "derived_raster_cell_nll",
        ],
        "tail_depth_p95_cvar": copy.deepcopy(TAIL_DEPTH_DEFINITION),
    }
    return value


def science_delta() -> dict[str, Any]:
    return {
        "base_science_contract_sha256": canonical_json_sha256(_v3_contract.science_contract()),
        "scientific_change_count": 1,
        "scientific_change": "replace_one_existing_camera_objective_slot",
        "changed_objective_slot": "camera_loss.target_bin_offset_smooth_l1",
        "before": "target_bin_offset_smooth_l1_at_target_bin",
        "after": "tail_depth_p95_cvar_over_conditional_finite_hit_distribution",
        "slot_weight_before_and_after": 0.25,
        "contract_leaf_changes_encoding_that_one_slot_replacement": [
            {"path": "camera_loss.source", "before": _v3_contract.science_contract()["camera_loss"]["source"], "after": LOSS_RELATIVE_PATH},
            {"path": "camera_loss.terms[1]", "before": "target_bin_offset_smooth_l1", "after": "tail_depth_p95_cvar"},
            {"path": "camera_loss.tail_depth_p95_cvar", "before": None, "after": copy.deepcopy(TAIL_DEPTH_DEFINITION)},
        ],
        "other_science_changes": [],
    }


def control_contract() -> dict[str, Any]:
    return {
        "precedence": ["integrity_failure_is_terminal", "earliest_all_nine_physical_pass_qualifies", "fixed_checkpoint_control"],
        "margin_statistics": {"count": MARGIN_COUNT, "P": "count(m>=0)", "S": "sum(max(0,-m))", "W": "min(m)"},
        "update_100": "finite_state_metrics_92_gradients_frozen_hash_trainable_movement_and_189_margins_required_then_qualify_or_continue",
        "update_400": "qualify_or_continue_without_numeric_cutoff",
        "pareto_baselines": copy.deepcopy(PERSISTED_BASELINE_PROGRESS),
        "pareto_rule": "P>=P3_and_S<=S3_and_W>=W3_with_at_least_one_strict; loss_is_not_compared",
        "update_4000": "only_all_nine_qualifies_otherwise_stop_unqualified",
        "retry_resume_extension_threshold_relaxation_or_soft_promotion_authorized": False,
    }


def reporting_contract() -> dict[str, Any]:
    value = copy.deepcopy(_v3_contract.reporting_contract())
    value["numeric_progress_cutoff_updates"] = [1_000, 2_000]
    value["numeric_continuation_rule"] = "V4_P_S_W_only_Pareto_control_bound_inside_each_immutable_sidecar"
    return value


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    inherited = _v3_contract.current_source_bindings(root)
    if any(inherited.get(path) != digest for path, digest in V3_SOURCE_SHA256.items()):
        raise PermissionError("protected Camera V3 source binding changed")
    result = dict(inherited)
    for path in SOURCE_PATHS:
        if path in result:
            continue
        source = root / path
        if source.is_symlink() or not source.is_file():
            raise PermissionError(f"reviewed V4 input is not one regular file: {path}")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if path in FIXED_EVIDENCE_SHA256 and digest != FIXED_EVIDENCE_SHA256[path]:
            raise PermissionError(f"preregistered V4 evidence changed: {path}")
        result[path] = digest
    if set(result) != set(SOURCE_PATHS):
        raise PermissionError("protected Camera V4 source closure changed")
    return result


def validate_review(value: object, *, expected_sources: Mapping[str, str]) -> dict[str, Any]:
    fields = {"schema", "status", "implementation_author", "reviewer", "reviewed_sources", "predecessor", "science_contract", "science_delta", "evidence", "reporting_contract", "control_contract", "source_only", "findings", "authority", "content_sha256"}
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("independent review fields changed")
    core, declared = dict(value), value["content_sha256"]
    core.pop("content_sha256")
    reviewer = value["reviewer"]
    if (
        value["schema"] != REVIEW_SCHEMA or value["status"] != "PASS"
        or value["implementation_author"] != IMPLEMENTATION_AUTHOR
        or type(reviewer) is not str or not reviewer.startswith("/root/") or reviewer == IMPLEMENTATION_AUTHOR
        or value["reviewed_sources"] != dict(expected_sources)
        or value["predecessor"] != predecessor_contract() or value["science_contract"] != science_contract()
        or value["science_delta"] != science_delta() or value["evidence"] != evidence_contract()
        or value["reporting_contract"] != reporting_contract() or value["control_contract"] != control_contract()
        or value["source_only"] is not True or value["findings"] != []
        or value["authority"] != REVIEW_AUTHORITY or not _v1.is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("independent review did not pass exact V4 sources")
    return dict(value)


def validate_authorization(value: object, *, review_binding: Mapping[str, Any], reviewer: str) -> dict[str, Any]:
    fields = {"schema", "status", "authorizer", "independent_review", "predecessor", "raw", "camera", "experiment", "science_delta", "evidence", "reporting_contract", "control_contract", "authority", "content_sha256"}
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("execution authorization fields changed")
    core, declared = dict(value), value["content_sha256"]
    core.pop("content_sha256")
    authorizer = value["authorizer"]
    if (
        value["schema"] != AUTHORIZATION_SCHEMA
        or value["status"] != "authorized_one_exact_protected_camera_adaptation_v4_tail_depth_attempt"
        or type(authorizer) is not str or not authorizer.startswith("/root/")
        or authorizer in {IMPLEMENTATION_AUTHOR, reviewer}
        or value["independent_review"] != dict(review_binding) or value["predecessor"] != predecessor_contract()
        or value["raw"] != expected_raw_authority() or value["camera"] != expected_camera_authority()
        or value["experiment"] != science_contract() or value["science_delta"] != science_delta()
        or value["evidence"] != evidence_contract() or value["reporting_contract"] != reporting_contract()
        or value["control_contract"] != control_contract() or value["authority"] != EXECUTION_AUTHORITY
        or not _v1.is_sha256(declared) or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("execution authorization changed")
    return dict(value)


def control_decision_from_progress(*, update: int, passed_margin_count: int, total_shortfall: float, worst_margin: float, aggregate_complete_v4_loss: float, all_nine_physical_pass: bool) -> dict[str, Any]:
    if type(update) is not int or update not in CHECKPOINT_UPDATES:
        raise ValueError("checkpoint control update is not fixed")
    if type(passed_margin_count) is not int or not 0 <= passed_margin_count <= MARGIN_COUNT:
        raise ValueError("passed physical margin count is invalid")
    scalars = (total_shortfall, worst_margin, aggregate_complete_v4_loss)
    if type(all_nine_physical_pass) is not bool or any(type(x) not in (int, float) or not math.isfinite(float(x)) for x in scalars):
        raise FloatingPointError("checkpoint progress value became invalid or nonfinite")
    shortfall, worst, loss = map(float, scalars)
    statistics = {"margin_count": MARGIN_COUNT, "passed_margin_count": passed_margin_count, "total_shortfall": shortfall, "worst_margin": worst, "aggregate_complete_v4_loss": loss, "all_nine_physical_pass": all_nine_physical_pass}
    if all_nine_physical_pass:
        action, reason, terminal, next_update = CONTROL_ACTION_QUALIFY, "earliest fixed checkpoint passed all nine physical scopes", "earliest_all_nine_physical_pass", None
    elif update in (100, 400):
        action, reason, terminal, next_update = CONTROL_ACTION_CONTINUE, "fixed checkpoint continues because it did not qualify", None, CHECKPOINT_UPDATES[CHECKPOINT_UPDATES.index(update) + 1]
    elif update in BASELINE_PROGRESS:
        base = BASELINE_PROGRESS[update]
        weak = passed_margin_count >= base["passed_margin_count"] and shortfall <= base["total_shortfall"] and worst >= base["worst_margin"]
        strict = passed_margin_count > base["passed_margin_count"] or shortfall < base["total_shortfall"] or worst > base["worst_margin"]
        keep = weak and strict
        action = CONTROL_ACTION_CONTINUE if keep else CONTROL_ACTION_STOP_PROGRESS
        reason = "P/S/W Pareto-dominated V3 with one strict improvement" if keep else "checkpoint missed preregistered P/S/W Pareto continuation"
        terminal, next_update = (None, 2_000 if update == 1_000 else 4_000) if keep else (f"predeclared_numeric_progress_cutoff_at_update_{update}", None)
    else:
        action, reason, terminal, next_update = CONTROL_ACTION_STOP_MAXIMUM, "update 4000 did not pass the unchanged all-nine physical gate", "scientific_numeric_physical_gate_at_update_4000", None
    return {"schema": f"{SCHEMA_PREFIX}_checkpoint_control_decision_v1", "update": update, "statistics": statistics, "action": action, "reason": reason, "qualifies": action == CONTROL_ACTION_QUALIFY, "terminal_stage": terminal, "next_checkpoint_update": next_update, "control_contract_sha256": canonical_json_sha256(control_contract())}


def checkpoint_progress(metric: Mapping[str, Any]) -> dict[str, Any]:
    return _v3_contract.checkpoint_progress(metric)


def checkpoint_control_decision(metric: Mapping[str, Any]) -> dict[str, Any]:
    progress = checkpoint_progress(metric)
    if progress["update"] == 100:
        before, after = metric.get("state_sha256_before"), metric.get("state_sha256_after")
        frozen = metric.get("frozen_state_sha256_before_and_after")
        if (not _v1.is_sha256(before) or after != before or before == UPDATE0_STATE_SHA256
                or not _v1.is_sha256(frozen) or metric.get("state_mutation_count") != 0):
            raise PermissionError("update 100 integrity or trainable-state movement failed")
    return control_decision_from_progress(**progress)


def metric_sidecar_path(update: int) -> str:
    return _v3_contract.metric_sidecar_path(update)


def validate_metric_sidecar(value: object, *, update: int | None = None, checkpoint: Mapping[str, Any] | None = None, metric: Mapping[str, Any] | None = None) -> dict[str, Any]:
    fields = {"schema", "status", "update", "checkpoint", "metric", "inline_evaluation_count", "state_mutation_count", "publication", "continuation", "authority", "content_sha256"}
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("checkpoint metric sidecar fields changed")
    core, declared = dict(value), value["content_sha256"]
    core.pop("content_sha256")
    observed_update, observed_metric = value["update"], value["metric"]
    if (
        type(observed_update) is not int or observed_update not in CHECKPOINT_UPDATES
        or value["schema"] != METRIC_SIDECAR_SCHEMA
        or value["status"] != "published_after_inline_nonmutating_physical_evaluation_before_control_branch"
        or type(value["checkpoint"]) is not dict or type(observed_metric) is not dict
        or observed_metric.get("update") != observed_update or observed_metric.get("role") != "checkpoint_selection"
        or observed_metric.get("state_mutation_count") != 0 or value["inline_evaluation_count"] != 1 or value["state_mutation_count"] != 0
        or value["publication"] != reporting_contract()["publication_order"]
        or value["continuation"] != checkpoint_control_decision(observed_metric)
        or value["authority"] != {"read_only_observation_authorized": True, "observer_evaluation_rerun_authorized": False, "only_predeclared_metric_control_authorized": True, "g2_navigation_or_heldout_use_authorized": False}
        or (update is not None and observed_update != update) or (checkpoint is not None and value["checkpoint"] != dict(checkpoint))
        or (metric is not None and observed_metric != dict(metric)) or not _v1.is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("checkpoint metric sidecar changed")
    return dict(value)


def expected_raw_authority() -> dict[str, Any]:
    return _v3_contract.expected_raw_authority()


def expected_camera_authority() -> dict[str, Any]:
    return _v3_contract.expected_camera_authority()


def __getattr__(name: str) -> Any:
    return getattr(_v3_contract, name)


__all__ = [name for name in globals() if name.isupper()] + [
    "artifact_binding", "canonical_json_bytes", "canonical_json_sha256", "checkpoint_control_decision",
    "checkpoint_progress", "control_contract", "control_decision_from_progress", "current_source_bindings",
    "evidence_contract", "expected_camera_authority", "expected_raw_authority", "metric_sidecar_path",
    "parse_canonical_json", "predecessor_contract", "reporting_contract", "science_contract", "science_delta",
    "validate_authorization", "validate_metric_sidecar", "validate_review", "with_content_sha256",
]
