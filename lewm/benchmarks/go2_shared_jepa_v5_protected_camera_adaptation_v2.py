"""Frozen contract for the one-knob protected Camera adaptation V2 attempt."""
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
IMPLEMENTATION_AUTHOR = "/root/camera_v2_implement"
SCHEMA_PREFIX = "lewm_go2_shared_jepa_v5_protected_camera_adaptation_v2"
CONTRACT_RELATIVE_PATH = "lewm/benchmarks/go2_shared_jepa_v5_protected_camera_adaptation_v2.py"
RUNNER_RELATIVE_PATH = "scripts/run_go2_shared_jepa_v5_protected_camera_adaptation_v2.py"
TEST_RELATIVE_PATH = "lewm/tests/test_go2_shared_jepa_v5_protected_camera_adaptation_v2.py"

V1_CONTRACT_RELATIVE_PATH = "lewm/benchmarks/go2_shared_jepa_v5_protected_camera_adaptation_v1.py"
V1_RUNNER_RELATIVE_PATH = "scripts/run_go2_shared_jepa_v5_protected_camera_adaptation_v1.py"
V1_TEST_RELATIVE_PATH = "lewm/tests/test_go2_shared_jepa_v5_protected_camera_adaptation_v1.py"
V1_SOURCE_SHA256 = {
    V1_CONTRACT_RELATIVE_PATH: "d58d3a3c6a3189c4099d75bd6335429b765516a9031d61d7939cb9da68e65f79",
    V1_RUNNER_RELATIVE_PATH: "df0d931813cb9307418bf5bfb710eccd7ae94031220c7db688e35853b544898c",
    V1_TEST_RELATIVE_PATH: "123c3db33c46c2c7d21a9f92f12bc91d7bf5c8231238710cd84a8b10c12d415a",
}


def _load_exact(path: str, name: str, digest: str) -> ModuleType:
    source = ROOT / path
    raw = source.read_bytes()
    if source.is_symlink() or not source.is_file() or hashlib.sha256(raw).hexdigest() != digest:
        raise PermissionError(f"frozen V1 source changed: {path}")
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen V1 source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_v1_contract = _load_exact(
    V1_CONTRACT_RELATIVE_PATH,
    "_lewm_protected_camera_adaptation_v1_contract_for_v2",
    V1_SOURCE_SHA256[V1_CONTRACT_RELATIVE_PATH],
)
_diagnostic = _v1_contract._diagnostic
_v1 = _v1_contract._v1

V1_TERMINAL_AUDIT_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v1_terminal_audit_2026-07-15.json"
V1_TERMINAL_AUDIT_BINDING = {
    "path": V1_TERMINAL_AUDIT_RELATIVE_PATH,
    "file_sha256": "c52bd5e58be3b76389d6f992675f6518ab5e062a8bbf84736123fe415476feb7",
    "content_sha256": "42108f767ce648a4b2e99f6303f922e5372f981922dd3a969a2d255795e03447",
    "byte_count": 78_571,
    "schema": "lewm_go2_shared_jepa_v5_protected_camera_adaptation_v1_terminal_audit_v1",
    "verdict": "PASS_CONFIRMED_SCIENTIFIC_NUMERIC_PHYSICAL_GATE_FAILURE_NO_CHECKPOINT_QUALIFIED",
}
V1_TERMINAL_ROOT_RELATIVE_PATH = _v1_contract.OUTPUT_ROOT_RELATIVE_PATH
V1_TERMINAL_EXACT_PATHS = (
    "access.json",
    "checkpoint_metrics.json",
    "checkpoints/update_100.pt",
    "checkpoints/update_1000.pt",
    "checkpoints/update_2000.pt",
    "checkpoints/update_400.pt",
    "checkpoints/update_4000.pt",
    "failed.json",
    "reservation.json",
    "training_trace.jsonl",
)
V1_TERMINAL_EXACT_DIRECTORIES = (".", "checkpoints")

DIAGNOSTIC_CONTRACT_RELATIVE_PATH = _v1_contract.DIAGNOSTIC_CONTRACT_RELATIVE_PATH
DIAGNOSTIC_RUNNER_RELATIVE_PATH = _v1_contract.DIAGNOSTIC_RUNNER_RELATIVE_PATH
DIAGNOSTIC_TEST_RELATIVE_PATH = _v1_contract.DIAGNOSTIC_TEST_RELATIVE_PATH
DIAGNOSTIC_SOURCE_SHA256 = dict(_v1_contract.DIAGNOSTIC_SOURCE_SHA256)
UPDATE0_AUDIT_RELATIVE_PATH = _v1_contract.UPDATE0_AUDIT_RELATIVE_PATH
UPDATE0_AUDIT_BINDING = dict(_v1_contract.UPDATE0_AUDIT_BINDING)
UPDATE0_ROOT_RELATIVE_PATH = _v1_contract.UPDATE0_ROOT_RELATIVE_PATH
UPDATE0_STATE_SHA256 = _v1_contract.UPDATE0_STATE_SHA256
UPDATE0_TERMINAL_ARTIFACTS = copy.deepcopy(_v1_contract.UPDATE0_TERMINAL_ARTIFACTS)

OUTPUT_ROOT_RELATIVE_PATH = ".generated/go2_shared_observable_camera_ray_jepa_v5/protected_camera_adaptation_v2"
REVIEW_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v2_independent_review_2026-07-15.json"
AUTHORIZATION_RELATIVE_PATH = "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v2_execution_authorization_2026-07-15.json"
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
    V1_TERMINAL_AUDIT_RELATIVE_PATH,
    *_v1_contract.SOURCE_PATHS,
)))

MAXIMUM_UPDATE = _v1_contract.MAXIMUM_UPDATE
CHECKPOINT_UPDATES = tuple(_v1_contract.CHECKPOINT_UPDATES)
SCHEDULE_PREFIX_INDICES_SHA256 = _v1_contract.SCHEDULE_PREFIX_INDICES_SHA256
CHECKPOINT_SCHEDULE_PREFIX_SHA256 = dict(_v1_contract.CHECKPOINT_SCHEDULE_PREFIX_SHA256)
ENCODER_LR_SCALE = 0.10
POST_CLIP_NORM_ASSERTION_TOLERANCE = _v1_contract.POST_CLIP_NORM_ASSERTION_TOLERANCE
TRAINABLE_PARAMETER_PREFIXES = tuple(_v1_contract.TRAINABLE_PARAMETER_PREFIXES)
FROZEN_STATE_PREFIXES = tuple(_v1_contract.FROZEN_STATE_PREFIXES)
EXPECTED_PARAMETER_COUNTS = dict(_v1_contract.EXPECTED_PARAMETER_COUNTS)
EXPECTED_PARAMETER_TENSOR_COUNTS = dict(_v1_contract.EXPECTED_PARAMETER_TENSOR_COUNTS)
OPTIMIZER_CONTRACT = copy.deepcopy(_v1_contract.OPTIMIZER_CONTRACT)
OPTIMIZER_CONTRACT["encoder_learning_rate_scale"] = ENCODER_LR_SCALE
DOWNSTREAM_DENIALS = dict(_v1_contract.DOWNSTREAM_DENIALS)
REVIEW_AUTHORITY = {
    **dict(_v1_contract.REVIEW_AUTHORITY),
    "checkpoint_metric_sidecar_publication_authorized": False,
    "read_only_checkpoint_metric_observation_authorized": False,
}
EXECUTION_AUTHORITY = {
    **dict(_v1_contract.EXECUTION_AUTHORITY),
    "mutation_scope": OUTPUT_ROOT_RELATIVE_PATH,
    "checkpoint_metric_sidecar_publication_authorized": True,
    "read_only_checkpoint_metric_observation_authorized": True,
}

METRIC_SIDECAR_DIRECTORY = "checkpoints"
TERMINAL_DIRECTORIES_INCLUDING_ROOT = (".", "checkpoints")


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def with_content_sha256(core: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(core), "content_sha256": canonical_json_sha256(core)}


def parse_canonical_json(raw: bytes, *, name: str) -> dict[str, Any]:
    return dict(_v1_contract.parse_canonical_json(raw, name=name))


def artifact_binding(path: str, raw: bytes, *, content_sha256: str) -> dict[str, Any]:
    return _v1_contract.artifact_binding(path, raw, content_sha256=content_sha256)


def validate_binding(value: object, *, path: str | None = None) -> dict[str, Any]:
    return _v1_contract.validate_binding(value, path=path)


def expected_raw_authority() -> dict[str, Any]:
    return _v1_contract.expected_raw_authority()


def expected_camera_authority() -> dict[str, Any]:
    return _v1_contract.expected_camera_authority()


def predecessor_contract() -> dict[str, Any]:
    return {
        "initialization": _v1_contract.predecessor_contract(),
        "rejected_protected_camera_v1": {
            "terminal_audit": dict(V1_TERMINAL_AUDIT_BINDING),
            "root": V1_TERMINAL_ROOT_RELATIVE_PATH,
            "exact_paths": list(V1_TERMINAL_EXACT_PATHS),
            "exact_directories_including_root": list(V1_TERMINAL_EXACT_DIRECTORIES),
            "qualified_checkpoint_exists": False,
            "retry_or_extension_authorized": False,
        },
    }


def science_contract() -> dict[str, Any]:
    value = copy.deepcopy(_v1_contract.science_contract())
    value["optimizer"]["encoder_learning_rate_scale"] = ENCODER_LR_SCALE
    return value


def science_delta() -> dict[str, Any]:
    return {
        "base_science_contract_sha256": canonical_json_sha256(_v1_contract.science_contract()),
        "changed_path": "optimizer.encoder_learning_rate_scale",
        "before": _v1_contract.ENCODER_LR_SCALE,
        "after": ENCODER_LR_SCALE,
        "other_science_changes": [],
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
        "publication_order": "snapshot_then_inline_nonmutating_evaluation_then_atomic_exclusive_canonical_sidecar_then_selection_branch",
        "publication_mechanism": "same_directory_fsynced_temporary_regular_file_then_atomic_exclusive_hard_link_then_directory_fsync",
        "sidecar_files_read_only_after_publication": True,
        "read_only_observers_must_not_rerun_evaluation": True,
        "numeric_continuation_rule": "continue_to_next_fixed_checkpoint_unless_current_checkpoint_is_earliest_all_nine_physical_pass",
        "numeric_progress_cutoff_at_update_400": False,
        "metric_controlled_stop_other_than_earliest_all_nine_pass": False,
        "integrity_nonfinite_or_frozen_mutation_remains_terminal": True,
        "final_metrics_collate_only_already_computed_sidecar_rows": True,
    }


def validate_v1_terminal_audit(raw: bytes) -> dict[str, Any]:
    binding = V1_TERMINAL_AUDIT_BINDING
    if len(raw) != binding["byte_count"] or hashlib.sha256(raw).hexdigest() != binding["file_sha256"]:
        raise PermissionError("protected Camera V1 terminal audit byte binding changed")
    value = parse_canonical_json(raw, name="protected Camera V1 terminal audit")
    decision = value.get("successor_decision", {})
    inventory = value.get("terminal_inventory", {})
    authority = value.get("authority", {})
    if (
        value.get("schema") != binding["schema"]
        or value.get("content_sha256") != binding["content_sha256"]
        or value.get("verdict") != binding["verdict"]
        or decision.get("decision") != "REJECT_ALL_PROTECTED_CAMERA_V1_CHECKPOINTS_AND_STOP_BEFORE_JEPA"
        or decision.get("qualified_camera_checkpoint_exists") is not False
        or decision.get("automatic_successor_authorized") is not False
        or decision.get("frozen_camera_jepa_training_may_start") is not False
        or decision.get("training_extension_or_retry_authorized") is not False
        or decision.get("recommendation_is_retry_or_execution_authority") is not False
        or inventory.get("root") != V1_TERMINAL_ROOT_RELATIVE_PATH
        or inventory.get("exact_paths") != list(V1_TERMINAL_EXACT_PATHS)
        or inventory.get("exact_directories_including_root") != list(V1_TERMINAL_EXACT_DIRECTORIES)
        or inventory.get("exact_file_count") != len(V1_TERMINAL_EXACT_PATHS)
        or inventory.get("no_result_or_completion") is not True
        or inventory.get("no_g2_navigation_or_heldout_artifact") is not True
        or authority.get("protected_camera_v1_retry_authorized") is not False
        or authority.get("protected_camera_v1_extension_authorized") is not False
        or authority.get("jepa_training_authorized") is not False
        or authority.get("heldout_authorized") is not False
    ):
        raise PermissionError("protected Camera V1 no-pass conclusion changed")
    return value


def validate_update0_audit(raw: bytes) -> dict[str, Any]:
    return _v1_contract.validate_update0_audit(raw)


def current_source_bindings(root: Path = ROOT) -> dict[str, str]:
    inherited = _v1_contract.current_source_bindings(root)
    if any(inherited.get(path) != digest for path, digest in V1_SOURCE_SHA256.items()):
        raise PermissionError("protected Camera V1 source binding changed")
    audit_raw = (root / V1_TERMINAL_AUDIT_RELATIVE_PATH).read_bytes()
    validate_v1_terminal_audit(audit_raw)
    result = dict(inherited)
    for path in (V1_TERMINAL_AUDIT_RELATIVE_PATH, CONTRACT_RELATIVE_PATH, RUNNER_RELATIVE_PATH, TEST_RELATIVE_PATH):
        source = root / path
        if source.is_symlink() or not source.is_file():
            raise PermissionError(f"review source is not one regular file: {path}")
        result[path] = hashlib.sha256(source.read_bytes()).hexdigest()
    if set(result) != set(SOURCE_PATHS) or result[V1_TERMINAL_AUDIT_RELATIVE_PATH] != V1_TERMINAL_AUDIT_BINDING["file_sha256"]:
        raise PermissionError("protected Camera V2 source closure changed")
    return result


def validate_review(value: object, *, expected_sources: Mapping[str, str]) -> dict[str, Any]:
    fields = {
        "schema", "status", "implementation_author", "reviewer", "reviewed_sources",
        "predecessor", "science_contract", "science_delta", "reporting_contract",
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
        or value["source_only"] is not True
        or value["findings"] != []
        or value["authority"] != REVIEW_AUTHORITY
        or not _v1.is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("independent review did not pass exact V2 sources")
    return dict(value)


def validate_authorization(value: object, *, review_binding: Mapping[str, Any], reviewer: str) -> dict[str, Any]:
    fields = {
        "schema", "status", "authorizer", "independent_review", "predecessor", "raw",
        "camera", "experiment", "science_delta", "reporting_contract", "authority", "content_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError("execution authorization fields changed")
    core, declared = dict(value), value["content_sha256"]
    core.pop("content_sha256")
    authorizer = value["authorizer"]
    if (
        value["schema"] != AUTHORIZATION_SCHEMA
        or value["status"] != "authorized_one_exact_protected_camera_adaptation_v2_attempt"
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
        or value["authority"] != EXECUTION_AUTHORITY
        or not _v1.is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError("execution authorization changed")
    return dict(value)


def learning_rates(update: int) -> tuple[float, float]:
    if type(update) is not int or not 1 <= update <= MAXIMUM_UPDATE:
        raise ValueError("protected Camera V2 update must lie in [1,4000]")
    head = _v1.learning_rate(update)
    encoder = ENCODER_LR_SCALE * head
    if not math.isfinite(encoder) or encoder <= 0.0:
        raise ValueError("protected encoder learning rate is invalid")
    return head, encoder


def parameter_partition(name: str) -> str:
    return _v1_contract.parameter_partition(name)


def validate_checkpoint_prefix(updates: Sequence[int]) -> tuple[int, ...]:
    return _v1_contract.validate_checkpoint_prefix(updates)


def evaluate_physical_scopes(scopes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return _v1_contract.evaluate_physical_scopes(scopes)


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
        or value["status"] != "published_after_inline_nonmutating_physical_evaluation"
        or type(value["checkpoint"]) is not dict
        or type(observed_metric) is not dict
        or observed_metric.get("update") != observed_update
        or observed_metric.get("role") != "checkpoint_selection"
        or observed_metric.get("state_mutation_count") != 0
        or value["inline_evaluation_count"] != 1
        or value["state_mutation_count"] != 0
        or value["publication"] != reporting_contract()["publication_order"]
        or value["continuation"] != reporting_contract()["numeric_continuation_rule"]
        or value["authority"] != {
            "read_only_observation_authorized": True,
            "observer_evaluation_rerun_authorized": False,
            "metric_controlled_stop_other_than_earliest_all_nine_pass": False,
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


__all__ = [name for name in globals() if name.isupper()] + [
    "artifact_binding", "canonical_json_bytes", "canonical_json_sha256", "current_source_bindings",
    "evaluate_physical_scopes", "expected_camera_authority", "expected_metric_sidecar_paths",
    "expected_raw_authority", "learning_rates", "metric_sidecar_path", "parameter_partition",
    "parse_canonical_json", "predecessor_contract", "reporting_contract", "science_contract",
    "science_delta", "validate_authorization", "validate_binding", "validate_checkpoint_prefix",
    "validate_metric_sidecar", "validate_review", "validate_update0_audit", "validate_v1_terminal_audit",
    "with_content_sha256",
]
