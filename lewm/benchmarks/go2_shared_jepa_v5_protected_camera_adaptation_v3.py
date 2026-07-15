"""Frozen contract for the single final protected Camera adaptation V3 attempt."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_AUTHOR = "/root/camera_v3_implement"
SCHEMA_PREFIX = "lewm_go2_shared_jepa_v5_protected_camera_adaptation_v3"
CONTRACT_RELATIVE_PATH = "lewm/benchmarks/go2_shared_jepa_v5_protected_camera_adaptation_v3.py"
RUNNER_RELATIVE_PATH = "scripts/run_go2_shared_jepa_v5_protected_camera_adaptation_v3.py"
TEST_RELATIVE_PATH = "lewm/tests/test_go2_shared_jepa_v5_protected_camera_adaptation_v3.py"

V2_CONTRACT_RELATIVE_PATH = "lewm/benchmarks/go2_shared_jepa_v5_protected_camera_adaptation_v2.py"
V2_RUNNER_RELATIVE_PATH = "scripts/run_go2_shared_jepa_v5_protected_camera_adaptation_v2.py"
V2_TEST_RELATIVE_PATH = "lewm/tests/test_go2_shared_jepa_v5_protected_camera_adaptation_v2.py"
V2_SOURCE_SHA256 = {
    V2_CONTRACT_RELATIVE_PATH: "5a36f7b83139baa231d616090cdb4ec4ce00db89d1c0f447b8b812403cb1da7f",
    V2_RUNNER_RELATIVE_PATH: "e593144dbb46fc7d4e02742c03a557d90dd6e1facab46efc0952033aace1acfc",
    V2_TEST_RELATIVE_PATH: "f7431728a05bb46d6cceb82c89a9f91bdbd6e710ea14c295e4c914c90fd9f033",
}


def _load_exact(path: str, name: str, digest: str) -> ModuleType:
    source = ROOT / path
    raw = source.read_bytes()
    if source.is_symlink() or not source.is_file() or hashlib.sha256(raw).hexdigest() != digest:
        raise PermissionError(f"frozen V2 source changed: {path}")
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen V2 source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_v2_contract = _load_exact(
    V2_CONTRACT_RELATIVE_PATH,
    "_lewm_protected_camera_adaptation_v2_contract_for_v3",
    V2_SOURCE_SHA256[V2_CONTRACT_RELATIVE_PATH],
)
_v1_contract = _v2_contract._v1_contract
_diagnostic = _v2_contract._diagnostic
_v1 = _v2_contract._v1

V1_CONTRACT_RELATIVE_PATH = _v2_contract.V1_CONTRACT_RELATIVE_PATH
V1_RUNNER_RELATIVE_PATH = _v2_contract.V1_RUNNER_RELATIVE_PATH
V1_TEST_RELATIVE_PATH = _v2_contract.V1_TEST_RELATIVE_PATH
V1_SOURCE_SHA256 = dict(_v2_contract.V1_SOURCE_SHA256)

V2_TERMINAL_AUDIT_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v2_terminal_audit_2026-07-15.json"
V2_TERMINAL_AUDIT_BINDING = {
    "path": V2_TERMINAL_AUDIT_RELATIVE_PATH,
    "file_sha256": "568941cedb1b9e127e9c12f625022f5d5937c49158510cfbf39fd5a9b8940bc8",
    "content_sha256": "9d4d9552d43e8782e46f0b48bbd61bd3e65972d23d6a6ed50025b682ca0f5285",
    "byte_count": 18_870,
    "schema": "lewm_go2_shared_jepa_v5_protected_camera_adaptation_v2_terminal_audit_v1",
    "verdict": "PASS_CONFIRMED_SCIENTIFIC_NUMERIC_PHYSICAL_GATE_FAILURE_NO_CHECKPOINT_QUALIFIED",
}
V2_TERMINAL_ROOT_RELATIVE_PATH = _v2_contract.OUTPUT_ROOT_RELATIVE_PATH
V2_TERMINAL_EXACT_PATHS = (
    "access.json",
    "checkpoint_metrics.json",
    "checkpoints/update_100.metrics.json",
    "checkpoints/update_100.pt",
    "checkpoints/update_1000.metrics.json",
    "checkpoints/update_1000.pt",
    "checkpoints/update_2000.metrics.json",
    "checkpoints/update_2000.pt",
    "checkpoints/update_400.metrics.json",
    "checkpoints/update_400.pt",
    "checkpoints/update_4000.metrics.json",
    "checkpoints/update_4000.pt",
    "failed.json",
    "reservation.json",
    "training_trace.jsonl",
)
V2_TERMINAL_EXACT_DIRECTORIES = (".", "checkpoints")

OUTPUT_ROOT_RELATIVE_PATH = ".generated/go2_shared_observable_camera_ray_jepa_v5/protected_camera_adaptation_v3"
REVIEW_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v3_independent_review_2026-07-15.json"
AUTHORIZATION_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v3_execution_authorization_2026-07-15.json"
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

SOURCE_PATHS = tuple(dict.fromkeys((
    CONTRACT_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    V2_TERMINAL_AUDIT_RELATIVE_PATH,
    *_v2_contract.SOURCE_PATHS,
)))

MAXIMUM_UPDATE = _v2_contract.MAXIMUM_UPDATE
CHECKPOINT_UPDATES = tuple(_v2_contract.CHECKPOINT_UPDATES)
SCHEDULE_PREFIX_INDICES_SHA256 = _v2_contract.SCHEDULE_PREFIX_INDICES_SHA256
CHECKPOINT_SCHEDULE_PREFIX_SHA256 = dict(_v2_contract.CHECKPOINT_SCHEDULE_PREFIX_SHA256)
ENCODER_LR_SCALE = 1.0
POST_CLIP_NORM_ASSERTION_TOLERANCE = _v2_contract.POST_CLIP_NORM_ASSERTION_TOLERANCE
TRAINABLE_PARAMETER_PREFIXES = tuple(_v2_contract.TRAINABLE_PARAMETER_PREFIXES)
FROZEN_STATE_PREFIXES = tuple(_v2_contract.FROZEN_STATE_PREFIXES)
EXPECTED_PARAMETER_COUNTS = dict(_v2_contract.EXPECTED_PARAMETER_COUNTS)
EXPECTED_PARAMETER_TENSOR_COUNTS = dict(_v2_contract.EXPECTED_PARAMETER_TENSOR_COUNTS)
OPTIMIZER_CONTRACT = copy.deepcopy(_v2_contract.OPTIMIZER_CONTRACT)
OPTIMIZER_CONTRACT["encoder_learning_rate_scale"] = ENCODER_LR_SCALE
DOWNSTREAM_DENIALS = dict(_v2_contract.DOWNSTREAM_DENIALS)
REVIEW_AUTHORITY = {
    **dict(_v2_contract.REVIEW_AUTHORITY),
    "predeclared_numeric_progress_control_authorized": False,
}
EXECUTION_AUTHORITY = {
    **dict(_v2_contract.EXECUTION_AUTHORITY),
    "mutation_scope": OUTPUT_ROOT_RELATIVE_PATH,
    "predeclared_numeric_progress_control_authorized": True,
}

METRIC_SIDECAR_DIRECTORY = "checkpoints"
TERMINAL_DIRECTORIES_INCLUDING_ROOT = (".", "checkpoints")

MARGIN_COUNT = 189
UPDATE1000_PASSED_MARGIN_STOP_AT_MOST = 79
UPDATE1000_SHORTFALL_STOP_ABOVE = 81.54285850209153
UPDATE2000_PASSED_MARGIN_FLOOR = 97
UPDATE2000_SHORTFALL_CEILING = 61.736501404834726
UPDATE2000_WORST_MARGIN_FLOOR = -8.111872744560243
UPDATE2000_LOSS_CEILING = 0.7496288327330893

CONTROL_ACTION_CONTINUE = "continue_to_next_fixed_checkpoint"
CONTROL_ACTION_QUALIFY = "qualify_earliest_all_nine_physical_pass"
CONTROL_ACTION_STOP_PROGRESS = "stop_predeclared_numeric_progress_cutoff"
CONTROL_ACTION_STOP_MAXIMUM = "stop_unqualified_at_maximum_update"


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def with_content_sha256(core: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(core), "content_sha256": canonical_json_sha256(core)}


def parse_canonical_json(raw: bytes, *, name: str) -> dict[str, Any]:
    return dict(_v2_contract.parse_canonical_json(raw, name=name))


def artifact_binding(path: str, raw: bytes, *, content_sha256: str) -> dict[str, Any]:
    return _v2_contract.artifact_binding(path, raw, content_sha256=content_sha256)


def validate_binding(value: object, *, path: str | None = None) -> dict[str, Any]:
    return _v2_contract.validate_binding(value, path=path)


def expected_raw_authority() -> dict[str, Any]:
    return _v2_contract.expected_raw_authority()


def expected_camera_authority() -> dict[str, Any]:
    return _v2_contract.expected_camera_authority()


def predecessor_contract() -> dict[str, Any]:
    return {
        "fresh_update0_initialization_not_v2_continuation": True,
        "v2_predecessor": _v2_contract.predecessor_contract(),
        "rejected_protected_camera_v2": {
            "terminal_audit": dict(V2_TERMINAL_AUDIT_BINDING),
            "root": V2_TERMINAL_ROOT_RELATIVE_PATH,
            "exact_paths": list(V2_TERMINAL_EXACT_PATHS),
            "exact_directories_including_root": list(V2_TERMINAL_EXACT_DIRECTORIES),
            "qualified_checkpoint_exists": False,
            "retry_or_extension_authorized": False,
        },
    }


def science_contract() -> dict[str, Any]:
    value = copy.deepcopy(_v2_contract.science_contract())
    value["optimizer"]["encoder_learning_rate_scale"] = ENCODER_LR_SCALE
    return value


def science_delta() -> dict[str, Any]:
    return {
        "base_science_contract_sha256": canonical_json_sha256(_v2_contract.science_contract()),
        "changed_path": "optimizer.encoder_learning_rate_scale",
        "before": _v2_contract.ENCODER_LR_SCALE,
        "after": ENCODER_LR_SCALE,
        "other_science_changes": [],
    }


def control_contract() -> dict[str, Any]:
    return {
        "margin_vector": {
            "order": "nine_declared_scopes_then_each_exact_21_element_physical_margin_vector",
            "count": MARGIN_COUNT,
            "passed_margin_count": "P_u=count(m>=0)",
            "total_shortfall": "S_u=sum(max(0,-m))",
            "worst_margin": "W_u=min(m)",
            "loss": "L_u=aggregate_complete_v4_loss",
        },
        "precedence": [
            "integrity_nonfinite_or_frozen_mutation_is_terminal",
            "earliest_all_nine_physical_pass_qualifies",
            "predeclared_update_specific_numeric_control",
        ],
        "update_1000_stop_only_if": {
            "passed_margin_count_at_most": UPDATE1000_PASSED_MARGIN_STOP_AT_MOST,
            "total_shortfall_strictly_above": UPDATE1000_SHORTFALL_STOP_ABOVE,
            "conjunction": True,
        },
        "update_2000_continue_only_if": {
            "passed_margin_count_at_least": UPDATE2000_PASSED_MARGIN_FLOOR,
            "total_shortfall_at_most": UPDATE2000_SHORTFALL_CEILING,
            "at_least_one_of_passed_count_or_shortfall_is_strict": True,
            "worst_margin_at_least": UPDATE2000_WORST_MARGIN_FLOOR,
            "aggregate_complete_v4_loss_at_most": UPDATE2000_LOSS_CEILING,
        },
        "updates_100_and_400": "continue_unless_earliest_all_nine_pass",
        "update_4000": "only_all_nine_qualifies_otherwise_terminal_numeric_no_pass",
        "retry_extension_threshold_relaxation_or_soft_promotion_authorized": False,
    }


def metric_sidecar_path(update: int) -> str:
    if type(update) is not int or update not in CHECKPOINT_UPDATES:
        raise ValueError("metric sidecar update is not a fixed checkpoint")
    return f"{METRIC_SIDECAR_DIRECTORY}/update_{update}.metrics.json"


def expected_metric_sidecar_paths(updates: Sequence[int]) -> tuple[str, ...]:
    return tuple(metric_sidecar_path(update) for update in validate_checkpoint_prefix(updates))


def reporting_contract() -> dict[str, Any]:
    return {
        "fixed_checkpoint_updates": list(CHECKPOINT_UPDATES),
        "sidecar_paths": list(expected_metric_sidecar_paths(CHECKPOINT_UPDATES)),
        "one_inline_physical_evaluation_per_published_sidecar": True,
        "publication_order": "snapshot_then_inline_nonmutating_evaluation_then_atomic_exclusive_canonical_sidecar_then_predeclared_control_branch",
        "publication_mechanism": "same_directory_fsynced_temporary_regular_file_then_atomic_exclusive_hard_link_then_directory_fsync",
        "sidecar_files_read_only_after_publication": True,
        "read_only_observers_must_not_rerun_evaluation": True,
        "numeric_continuation_rule": "exact_checkpoint_control_decision_bound_inside_each_immutable_sidecar",
        "numeric_progress_cutoff_at_update_400": False,
        "numeric_progress_cutoff_updates": [1_000, 2_000],
        "metric_controlled_stop_other_than_earliest_all_nine_pass": True,
        "integrity_nonfinite_or_frozen_mutation_remains_terminal": True,
        "final_metrics_collate_only_already_computed_sidecar_rows": True,
        "all_nine_gate_unchanged": True,
    }


def validate_v2_terminal_audit(raw: bytes) -> dict[str, Any]:
    binding = V2_TERMINAL_AUDIT_BINDING
    if len(raw) != binding["byte_count"] or hashlib.sha256(raw).hexdigest() != binding["file_sha256"]:
        raise PermissionError("protected Camera V2 terminal audit byte binding changed")
    value = parse_canonical_json(raw, name="protected Camera V2 terminal audit")
    decision = value.get("successor_decision", {})
    inventory = value.get("terminal_inventory", {})
    authority = value.get("authority", {})
    if (
        value.get("schema") != binding["schema"]
        or value.get("content_sha256") != binding["content_sha256"]
        or value.get("verdict") != binding["verdict"]
        or decision.get("decision") != "REJECT_ALL_PROTECTED_CAMERA_V2_CHECKPOINTS_AND_STOP_BEFORE_JEPA"
        or decision.get("qualified_camera_checkpoint_exists") is not False
        or decision.get("automatic_successor_authorized") is not False
        or decision.get("frozen_camera_jepa_training_may_start") is not False
        or decision.get("retry_or_extension_authorized") is not False
        or inventory.get("root") != V2_TERMINAL_ROOT_RELATIVE_PATH
        or inventory.get("exact_paths") != list(V2_TERMINAL_EXACT_PATHS)
        or inventory.get("exact_directories_including_root") != list(V2_TERMINAL_EXACT_DIRECTORIES)
        or inventory.get("exact_file_count") != len(V2_TERMINAL_EXACT_PATHS)
        or inventory.get("no_completed_or_result_artifact") is not True
        or inventory.get("no_g2_navigation_or_heldout_artifact") is not True
        or authority.get("retry_authorized") is not False
        or authority.get("training_extension_authorized") is not False
        or authority.get("jepa_training_authorized") is not False
        or authority.get("heldout_authorized") is not False
    ):
        raise PermissionError("protected Camera V2 no-pass conclusion changed")
    return value


def validate_update0_audit(raw: bytes) -> dict[str, Any]:
    return _v2_contract.validate_update0_audit(raw)


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    inherited = _v2_contract.current_source_bindings(root)
    if any(inherited.get(path) != digest for path, digest in V2_SOURCE_SHA256.items()):
        raise PermissionError("protected Camera V2 source binding changed")
    audit_raw = (root / V2_TERMINAL_AUDIT_RELATIVE_PATH).read_bytes()
    validate_v2_terminal_audit(audit_raw)
    result = dict(inherited)
    for path in (V2_TERMINAL_AUDIT_RELATIVE_PATH, CONTRACT_RELATIVE_PATH, RUNNER_RELATIVE_PATH, TEST_RELATIVE_PATH):
        source = root / path
        if source.is_symlink() or not source.is_file():
            raise PermissionError(f"review source is not one regular file: {path}")
        result[path] = hashlib.sha256(source.read_bytes()).hexdigest()
    if set(result) != set(SOURCE_PATHS) or result[V2_TERMINAL_AUDIT_RELATIVE_PATH] != V2_TERMINAL_AUDIT_BINDING["file_sha256"]:
        raise PermissionError("protected Camera V3 source closure changed")
    return result


def validate_review(value: object, *, expected_sources: Mapping[str, str]) -> dict[str, Any]:
    fields = {
        "schema", "status", "implementation_author", "reviewer", "reviewed_sources",
        "predecessor", "science_contract", "science_delta", "reporting_contract", "control_contract",
        "source_only", "findings", "authority", "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("independent review fields changed")
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
        or value["reporting_contract"] != reporting_contract()
        or value["control_contract"] != control_contract()
        or value["source_only"] is not True
        or value["findings"] != []
        or value["authority"] != REVIEW_AUTHORITY
        or not _v1.is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("independent review did not pass exact V3 sources")
    return dict(value)


def validate_authorization(value: object, *, review_binding: Mapping[str, Any], reviewer: str) -> dict[str, Any]:
    fields = {
        "schema", "status", "authorizer", "independent_review", "predecessor", "raw",
        "camera", "experiment", "science_delta", "reporting_contract", "control_contract", "authority", "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("execution authorization fields changed")
    core, declared = dict(value), value["content_sha256"]
    core.pop("content_sha256")
    authorizer = value["authorizer"]
    if (
        value["schema"] != AUTHORIZATION_SCHEMA
        or value["status"] != "authorized_one_exact_final_protected_camera_adaptation_v3_attempt"
        or type(authorizer) is not str
        or not authorizer.startswith("/root/")
        or authorizer in {IMPLEMENTATION_AUTHOR, reviewer}
        or value["independent_review"] != dict(review_binding)
        or value["predecessor"] != predecessor_contract()
        or value["raw"] != expected_raw_authority()
        or value["camera"] != expected_camera_authority()
        or value["experiment"] != science_contract()
        or value["science_delta"] != science_delta()
        or value["reporting_contract"] != reporting_contract()
        or value["control_contract"] != control_contract()
        or value["authority"] != EXECUTION_AUTHORITY
        or not _v1.is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("execution authorization changed")
    return dict(value)


def learning_rates(update: int) -> tuple[float, float]:
    if type(update) is not int or not 1 <= update <= MAXIMUM_UPDATE:
        raise ValueError("protected Camera V3 update must lie in [1,4000]")
    head = _v1.learning_rate(update)
    encoder = ENCODER_LR_SCALE * head
    if not math.isfinite(encoder) or encoder <= 0.0:
        raise ValueError("protected encoder learning rate is invalid")
    return head, encoder


def parameter_partition(name: str) -> str:
    return _v2_contract.parameter_partition(name)


def validate_checkpoint_prefix(updates: Sequence[int]) -> tuple[int, ...]:
    return _v2_contract.validate_checkpoint_prefix(updates)


def evaluate_physical_scopes(scopes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return _v2_contract.evaluate_physical_scopes(scopes)


def control_decision_from_progress(
    *,
    update: int,
    passed_margin_count: int,
    total_shortfall: float,
    worst_margin: float,
    aggregate_complete_v4_loss: float,
    all_nine_physical_pass: bool,
) -> dict[str, Any]:
    if type(update) is not int or update not in CHECKPOINT_UPDATES:
        raise ValueError("checkpoint control update is not fixed")
    if type(passed_margin_count) is not int or not 0 <= passed_margin_count <= MARGIN_COUNT:
        raise ValueError("passed physical margin count is invalid")
    if type(all_nine_physical_pass) is not bool:
        raise TypeError("all-nine physical decision is not Boolean")
    scalars = (total_shortfall, worst_margin, aggregate_complete_v4_loss)
    if any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in scalars):
        raise FloatingPointError("checkpoint progress scalar became nonfinite")
    shortfall, worst, loss = map(float, scalars)
    statistics = {
        "margin_count": MARGIN_COUNT,
        "passed_margin_count": passed_margin_count,
        "total_shortfall": shortfall,
        "worst_margin": worst,
        "aggregate_complete_v4_loss": loss,
        "all_nine_physical_pass": all_nine_physical_pass,
    }
    if all_nine_physical_pass:
        action = CONTROL_ACTION_QUALIFY
        reason = "earliest fixed checkpoint passed every physical margin in all nine scopes"
        terminal_stage = "earliest_all_nine_physical_pass"
        next_update = None
    elif update in (100, 400):
        action = CONTROL_ACTION_CONTINUE
        reason = "pre-control checkpoint continues because it did not yet qualify"
        terminal_stage = None
        next_update = CHECKPOINT_UPDATES[CHECKPOINT_UPDATES.index(update) + 1]
    elif update == 1_000:
        stop = passed_margin_count <= UPDATE1000_PASSED_MARGIN_STOP_AT_MOST and shortfall > UPDATE1000_SHORTFALL_STOP_ABOVE
        action = CONTROL_ACTION_STOP_PROGRESS if stop else CONTROL_ACTION_CONTINUE
        reason = "update 1000 met both clear-regression stop conditions" if stop else "update 1000 did not meet both clear-regression stop conditions"
        terminal_stage = "predeclared_numeric_progress_cutoff_at_update_1000" if stop else None
        next_update = None if stop else 2_000
    elif update == 2_000:
        weak_pareto = passed_margin_count >= UPDATE2000_PASSED_MARGIN_FLOOR and shortfall <= UPDATE2000_SHORTFALL_CEILING
        one_strict = passed_margin_count > UPDATE2000_PASSED_MARGIN_FLOOR or shortfall < UPDATE2000_SHORTFALL_CEILING
        guards = worst >= UPDATE2000_WORST_MARGIN_FLOOR and loss <= UPDATE2000_LOSS_CEILING
        keep_going = weak_pareto and one_strict and guards
        action = CONTROL_ACTION_CONTINUE if keep_going else CONTROL_ACTION_STOP_PROGRESS
        reason = "update 2000 met weak Pareto progress with one strict improvement and both guards" if keep_going else "update 2000 missed the predeclared continue-to-4000 condition"
        terminal_stage = None if keep_going else "predeclared_numeric_progress_cutoff_at_update_2000"
        next_update = 4_000 if keep_going else None
    else:
        action = CONTROL_ACTION_STOP_MAXIMUM
        reason = "update 4000 did not pass the unchanged all-nine physical gate"
        terminal_stage = "scientific_numeric_physical_gate_at_update_4000"
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
    if type(metric) is not dict:
        raise TypeError("checkpoint metric is not a mapping")
    update = metric.get("update")
    evaluation = metric.get("evaluation")
    scopes = metric.get("scopes")
    if type(evaluation) is not dict or type(scopes) is not dict or evaluate_physical_scopes(scopes) != evaluation:
        raise PermissionError("checkpoint physical evaluation changed")
    rows = evaluation.get("scope_evaluations")
    if type(rows) is not dict or tuple(rows) != _v1.SCOPES:
        raise PermissionError("checkpoint physical scope order changed")
    margins: list[float] = []
    for scope in _v1.SCOPES:
        row = rows[scope]
        values = row.get("physical_margins") if type(row) is dict else None
        if type(values) is not list or len(values) != 21:
            raise PermissionError("checkpoint physical margin vector changed")
        for value in values:
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise FloatingPointError("checkpoint physical margin became nonfinite")
            margins.append(float(value))
    if len(margins) != MARGIN_COUNT:
        raise PermissionError("checkpoint physical margin count changed")
    loss = metric.get("aggregate_complete_v4_loss")
    if type(loss) not in (int, float) or not math.isfinite(float(loss)):
        raise FloatingPointError("checkpoint aggregate complete V4 loss became nonfinite")
    all_nine = evaluation.get("all_nine_physical_pass")
    if type(all_nine) is not bool or all_nine != all(value >= 0.0 for value in margins):
        raise PermissionError("checkpoint all-nine decision changed")
    return {
        "update": update,
        "passed_margin_count": sum(value >= 0.0 for value in margins),
        "total_shortfall": sum(max(0.0, -value) for value in margins),
        "worst_margin": min(margins),
        "aggregate_complete_v4_loss": float(loss),
        "all_nine_physical_pass": all_nine,
    }


def checkpoint_control_decision(metric: Mapping[str, Any]) -> dict[str, Any]:
    return control_decision_from_progress(**checkpoint_progress(metric))


def validate_metric_sidecar(
    value: object,
    *,
    update: int | None = None,
    checkpoint: Mapping[str, Any] | None = None,
    metric: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fields = {
        "schema", "status", "update", "checkpoint", "metric", "inline_evaluation_count",
        "state_mutation_count", "publication", "continuation", "authority", "content_sha256",
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
        or value["status"] != "published_after_inline_nonmutating_physical_evaluation_before_control_branch"
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
        or (checkpoint is not None and value["checkpoint"] != dict(checkpoint))
        or (metric is not None and observed_metric != dict(metric))
        or not _v1.is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("checkpoint metric sidecar changed")
    return dict(value)


def __getattr__(name: str) -> Any:
    """Delegate unchanged V2/V1 lifecycle constants used by the exact rebound runner."""
    return getattr(_v2_contract, name)


__all__ = [name for name in globals() if name.isupper()] + [
    "artifact_binding", "canonical_json_bytes", "canonical_json_sha256", "checkpoint_control_decision",
    "checkpoint_progress", "control_contract", "control_decision_from_progress", "current_source_bindings",
    "evaluate_physical_scopes", "expected_camera_authority", "expected_metric_sidecar_paths",
    "expected_raw_authority", "learning_rates", "metric_sidecar_path", "parameter_partition",
    "parse_canonical_json", "predecessor_contract", "reporting_contract", "science_contract",
    "science_delta", "validate_authorization", "validate_binding", "validate_checkpoint_prefix",
    "validate_metric_sidecar", "validate_review", "validate_update0_audit", "validate_v2_terminal_audit",
    "with_content_sha256",
]
