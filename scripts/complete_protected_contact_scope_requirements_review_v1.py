#!/usr/bin/env python3
"""Complete the frozen protected-contact requirements review.

This deterministic completion layer binds the frozen Stage-A requirements
decision to the already-published, development-only geometry mismatch audit.
It does not rerun geometry, change scope or labels, authorize Stage B, open a
model, or execute training, navigation, memory, routing, or beacon capture.

``generate`` writes exactly one Markdown report and two canonical JSON
receipts.  ``check`` reconstructs all three byte-for-byte and validates every
source-closure member.  The result and result-source-closure receipts exclude
themselves from the source list so that their content digests are acyclic.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from lewm.safety import protected_contact_scope_requirements_review_v1 as stage_a
from scripts import audit_contact_critical_geometry_mismatch_v1 as diagnostic_core


EXPERIMENT_ID = "PROTECTED_CONTACT_SCOPE_REQUIREMENTS_REVIEW_V1"
COMPLETION_SCHEMA = "protected_contact_scope_requirements_review_v1.result.v1"
SOURCE_CLOSURE_SCHEMA = (
    "protected_contact_scope_requirements_review_v1.result_source_closure.v1"
)
STAGE_A_FREEZE_COMMIT = "e9e8c41a327ddbe51c38fa04f05ae1d30266720b"
PREDECESSOR_RESULT_COMMIT = "99cfa17cddb2aaddde69b8bdb6c3ea4a8e5ca849"
PRIMARY_CLASSIFICATION = "PROTECTED_CONTACT_SCOPE_REQUIREMENTS_UNRESOLVED"
STAGE_B_STATUS = "STAGE_B_NOT_AUTHORIZED"
STAGE_B_EXECUTION = "NOT_RUN"
NEXT_DECISION = "REQUIREMENTS_ACQUISITION_REQUIRED"

DOCS = REPOSITORY / "docs"
STAGE_A_CONTRACT_PATH = DOCS / "lewm_protected_contact_scope_contract_v1.json"
STAGE_A_SOURCE_CLOSURE_PATH = (
    DOCS / "lewm_protected_contact_scope_source_closure_v1.json"
)
DIAGNOSTIC_JSON_PATH = DOCS / "lewm_contact_critical_geometry_mismatch_audit_v1.json"
DIAGNOSTIC_MARKDOWN_PATH = DOCS / "lewm_contact_critical_geometry_mismatch_audit_v1.md"
RESULT_SOURCE_CLOSURE_PATH = (
    DOCS / "lewm_protected_contact_scope_requirements_review_v1_result_source_closure.json"
)
RESULT_PATH = DOCS / "lewm_protected_contact_scope_requirements_review_v1_result.json"

STAGE_A_ARTIFACT_PATHS = (
    "docs/lewm_protected_contact_scope_requirements_review_v1.md",
    "docs/lewm_protected_contact_scope_traceability_matrix_v1.md",
    "docs/lewm_protected_contact_scope_decision_memo_v1.md",
    "docs/lewm_protected_contact_scope_assurance_fragment_v1.md",
    "docs/lewm_protected_contact_scope_contract_v1.json",
    "docs/lewm_protected_contact_scope_link_object_context_matrix_v1.json",
    "docs/lewm_protected_contact_scope_requirement_traceability_v1.json",
    "docs/lewm_protected_contact_scope_assumptions_unresolved_v1.json",
    "docs/lewm_protected_contact_scope_source_closure_v1.json",
)

RESULT_SOURCE_ADDITIONS = (
    (
        "scripts/audit_contact_critical_geometry_mismatch_v1.py",
        "READ_ONLY_DIAGNOSTIC_REDUCER",
    ),
    (
        "lewm/tests/test_audit_contact_critical_geometry_mismatch_v1.py",
        "DIAGNOSTIC_REDUCER_TEST",
    ),
    (
        "scripts/complete_protected_contact_scope_requirements_review_v1.py",
        "DETERMINISTIC_COMPLETION_GENERATOR",
    ),
    (
        "lewm/tests/test_complete_protected_contact_scope_requirements_review_v1.py",
        "COMPLETION_GENERATOR_TEST",
    ),
    (
        "docs/lewm_contact_critical_geometry_mismatch_audit_v1.json",
        "CANONICAL_DEVELOPMENT_ONLY_DIAGNOSTIC_RESULT",
    ),
    (
        "docs/lewm_contact_critical_geometry_mismatch_audit_v1.md",
        "HUMAN_READABLE_DIAGNOSTIC_REPORT",
    ),
)

EXCLUDED_SELF_REFERENTIAL_PATHS = (
    "docs/lewm_protected_contact_scope_requirements_review_v1_result_source_closure.json",
    "docs/lewm_protected_contact_scope_requirements_review_v1_result.json",
)

EXPECTED_DIAGNOSTIC_CAUSES = (
    "CONTACT_CRITICAL_PATCH_UNOBSERVED",
    "POINT_TO_PRIMITIVE_DISTANCE_MISMATCH",
    "GLOBAL_THRESHOLD_HETEROGENEITY",
    "SELF_OCCLUSION_AT_CONTACT_CRITICAL_REGION",
    "UNRESOLVED_GEOMETRIC_MISMATCH",
)


class CompletionError(RuntimeError):
    """Fail-closed completion validation error."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise CompletionError(f"value is not canonical JSON: {exc}") from exc


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def attach_content_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    if "content_digest" in value:
        raise CompletionError("content_digest may be attached exactly once")
    result = dict(value)
    result["content_digest"] = canonical_digest(result)
    return result


def validate_content_digest(value: Mapping[str, Any]) -> None:
    observed = value.get("content_digest")
    if not isinstance(observed, str) or len(observed) != 64:
        raise CompletionError("missing content_digest")
    core = {key: item for key, item in value.items() if key != "content_digest"}
    if canonical_digest(core) != observed:
        raise CompletionError("content_digest mismatch")


def canonical_file_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def sha256_file(path: Path, *, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompletionError(f"cannot load JSON receipt {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CompletionError(f"JSON receipt must be an object: {path}")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        os.chmod(path, 0o644)
    finally:
        temporary_path.unlink(missing_ok=True)


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPOSITORY.resolve()))


def _binding(path: Path, *, content_digest: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise CompletionError(f"bound artifact missing: {_relative(path)}")
    result: dict[str, Any] = {
        "path": _relative(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if content_digest is not None:
        result["content_digest"] = content_digest
    return result


def load_and_validate_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = _load_json(STAGE_A_CONTRACT_PATH)
    stage_a.validate_contract_receipt(contract)
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise CompletionError("Stage-A experiment identity drift")
    if contract.get("start_commit") != PREDECESSOR_RESULT_COMMIT:
        raise CompletionError("Stage-A predecessor result binding drift")
    if contract.get("classifications", {}).get("primary_exactly_one") != PRIMARY_CLASSIFICATION:
        raise CompletionError("Stage-A primary classification drift")
    if tuple(contract.get("classifications", {}).get("secondary_exactly", ())) != tuple(
        stage_a.ALLOWED_SECONDARY_CLASSIFICATIONS
    ):
        raise CompletionError("Stage-A secondary classification drift")
    gate = contract.get("stage_b_gate", {})
    if gate.get("status") != STAGE_B_STATUS or gate.get("authorized") is not False:
        raise CompletionError("Stage B is not fail-closed")
    scope_decision = contract.get("scope_decision", {})
    if (
        scope_decision.get("scope_narrowing_authorized") is not False
        or scope_decision.get("contact_labels_changed") is not False
        or scope_decision.get("protected_links_removed") != []
        or scope_decision.get("protected_shapes_removed") != []
    ):
        raise CompletionError("frozen protected scope or labels changed")

    stage_closure = _load_json(STAGE_A_SOURCE_CLOSURE_PATH)
    validate_content_digest(stage_closure)
    if stage_closure.get("experiment_id") != EXPERIMENT_ID:
        raise CompletionError("Stage-A source closure experiment drift")
    if stage_closure.get("file_count") != 40:
        raise CompletionError("Stage-A source closure must have exactly 40 rows")
    stage_rows = stage_closure.get("files")
    if not isinstance(stage_rows, list) or len(stage_rows) != 40:
        raise CompletionError("Stage-A source closure rows missing")
    for row in stage_rows:
        _validate_source_row(row)

    diagnostic = _load_json(DIAGNOSTIC_JSON_PATH)
    try:
        diagnostic_core.validate_audit_result(diagnostic)
    except diagnostic_core.AuditError as exc:
        raise CompletionError(f"diagnostic result invalid: {exc}") from exc
    if tuple(diagnostic.get("diagnostic_cause_ids", ())) != EXPECTED_DIAGNOSTIC_CAUSES:
        raise CompletionError("diagnostic cause vocabulary drift")
    if tuple(diagnostic.get("supported_diagnostic_cause_ids", ())) != EXPECTED_DIAGNOSTIC_CAUSES:
        raise CompletionError("all five diagnostic causes must have evidence")
    barrier = diagnostic.get("stage_a_barrier", {})
    if (
        barrier.get("freeze_commit") != STAGE_A_FREEZE_COMMIT
        or barrier.get("primary_classification") != PRIMARY_CLASSIFICATION
        or barrier.get("stage_b_authorized") is not False
        or barrier.get("scope_narrowing_authorized") is not False
        or barrier.get("protected_links") != 13
        or barrier.get("protected_collision_components") != 27
    ):
        raise CompletionError("diagnostic Stage-A barrier drift")
    if barrier.get("contract_file_sha256") != sha256_file(STAGE_A_CONTRACT_PATH):
        raise CompletionError("diagnostic contract SHA binding drift")
    if barrier.get("contract_content_digest") != contract.get("content_digest"):
        raise CompletionError("diagnostic contract content binding drift")
    if any(int(value) != 0 for value in diagnostic.get("prohibited_action_counters", {}).values()):
        raise CompletionError("diagnostic prohibited-action counter is nonzero")
    return contract, stage_closure, diagnostic


def _validate_source_row(row: Mapping[str, Any]) -> None:
    if set(row) != {"bytes", "path", "role", "sha256"}:
        raise CompletionError("source-closure row field drift")
    relative = str(row["path"])
    path = REPOSITORY / relative
    if not path.is_file():
        raise CompletionError(f"source-closure member missing: {relative}")
    if row.get("bytes") != path.stat().st_size:
        raise CompletionError(f"source-closure byte drift: {relative}")
    if row.get("sha256") != sha256_file(path):
        raise CompletionError(f"source-closure SHA drift: {relative}")


def extract_heldout_true_future(diagnostic: Mapping[str, Any]) -> dict[str, Any]:
    rows = diagnostic.get("exact_contact_event_diagnostic", {}).get(
        "condition_mode_by_role", {}
    )
    suffix = "|TRUE_FUTURE_OBSERVABILITY_CLOUD|heldout"
    selected = {
        key.removesuffix(suffix): copy.deepcopy(value)
        for key, value in sorted(rows.items())
        if key.endswith(suffix)
    }
    expected_conditions = set(
        diagnostic_core.SINGLE_CONDITION_IDS
        + diagnostic_core.MULTI_CONDITION_IDS
    )
    if set(selected) != expected_conditions or len(selected) != 11:
        raise CompletionError("heldout true-future condition coverage drift")
    for condition, row in selected.items():
        events = int(row.get("contact_events", -1))
        supported = int(row.get("patch_supported_events", -1))
        unsupported = int(row.get("patch_unobserved_events", -1))
        if events <= 0 or supported < 0 or unsupported < 0 or supported + unsupported != events:
            raise CompletionError(f"heldout support denominator drift: {condition}")
        expected_rate = supported / events
        if not math.isclose(
            float(row.get("patch_support_rate", -1.0)),
            expected_rate,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise CompletionError(f"heldout patch-support rate drift: {condition}")
        residual = row.get("observed_minus_exact_statistics_m", {})
        if residual.get("count") != supported:
            raise CompletionError(f"heldout residual/support denominator drift: {condition}")
    return selected


def extract_calibration_q95(diagnostic: Mapping[str, Any]) -> dict[str, Any]:
    source = diagnostic.get("calibration_only_threshold_shift_diagnostic", {})
    if (
        source.get("population") != "frozen internal-calibration role only"
        or source.get("selection_authority") is not False
        or source.get("heldout_used") is not False
        or source.get("interpolation_used") is not False
    ):
        raise CompletionError("calibration diagnostic authority drift")
    result = copy.deepcopy(source)
    for group_name in ("per_link", "per_collision_component"):
        group = result.get(group_name)
        if not isinstance(group, dict) or not group:
            raise CompletionError(f"calibration {group_name} missing")
        for key, row in group.items():
            if row.get("role") != "calibration" or row.get("threshold_reselected") is not False:
                raise CompletionError(f"calibration authority drift: {key}")
            positives = int(row.get("positives", -1))
            finite = int(row.get("finite_attributable_link_minimum_scores", -1))
            unsupported = int(row.get("unsupported_attributable_link_minimum_scores", -1))
            if positives < 0 or finite < 0 or unsupported < 0 or finite + unsupported != positives:
                raise CompletionError(f"calibration denominator drift: {key}")
            q95 = row.get("nearest_rank_q95_m")
            status = row.get("threshold_shift_status")
            if q95 == "POSITIVE_INFINITY":
                if status != "UNBOUNDED_BY_UNSUPPORTED_CALIBRATION_PATCHES":
                    raise CompletionError(f"calibration infinity status drift: {key}")
            elif not isinstance(q95, (int, float)) or not math.isfinite(float(q95)):
                raise CompletionError(f"calibration q95 invalid: {key}")
    return result


def build_result_source_closure(
    stage_closure: Mapping[str, Any],
) -> dict[str, Any]:
    frozen_rows = copy.deepcopy(stage_closure.get("files"))
    if not isinstance(frozen_rows, list) or len(frozen_rows) != 40:
        raise CompletionError("cannot extend noncanonical Stage-A source closure")
    rows = frozen_rows
    observed_paths = {str(row["path"]) for row in rows}
    for relative, role in RESULT_SOURCE_ADDITIONS:
        if relative in observed_paths:
            raise CompletionError(f"duplicate result source path: {relative}")
        observed_paths.add(relative)
        path = REPOSITORY / relative
        if not path.is_file():
            raise CompletionError(f"result source member missing: {relative}")
        rows.append(
            {
                "bytes": path.stat().st_size,
                "path": relative,
                "role": role,
                "sha256": sha256_file(path),
            }
        )
    if observed_paths.intersection(EXCLUDED_SELF_REFERENTIAL_PATHS):
        raise CompletionError("self-referential result artifact entered source closure")
    value = attach_content_digest(
        {
            "schema_version": SOURCE_CLOSURE_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "predecessor_result_commit": PREDECESSOR_RESULT_COMMIT,
            "stage_a_freeze_commit": STAGE_A_FREEZE_COMMIT,
            "stage_a_source_closure": {
                "path": _relative(STAGE_A_SOURCE_CLOSURE_PATH),
                "sha256": sha256_file(STAGE_A_SOURCE_CLOSURE_PATH),
                "bytes": STAGE_A_SOURCE_CLOSURE_PATH.stat().st_size,
                "content_digest": stage_closure["content_digest"],
                "file_count": 40,
            },
            "stage_a_rows_preserved_verbatim": True,
            "stage_a_file_count": 40,
            "completion_file_count": len(RESULT_SOURCE_ADDITIONS),
            "file_count": len(rows),
            "files": rows,
            "excluded_self_referential_paths": list(EXCLUDED_SELF_REFERENTIAL_PATHS),
            "stage_b_execution": STAGE_B_EXECUTION,
            "outcome_authority": "DEVELOPMENT_ONLY_DIAGNOSTIC_NOT_SCOPE_DECISION_AUTHORITY",
        }
    )
    validate_result_source_closure(value, stage_closure=stage_closure)
    return value


def validate_result_source_closure(
    value: Mapping[str, Any], *, stage_closure: Mapping[str, Any]
) -> None:
    validate_content_digest(value)
    if value.get("schema_version") != SOURCE_CLOSURE_SCHEMA:
        raise CompletionError("result source-closure schema drift")
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise CompletionError("result source-closure experiment drift")
    if (
        value.get("predecessor_result_commit") != PREDECESSOR_RESULT_COMMIT
        or value.get("stage_a_freeze_commit") != STAGE_A_FREEZE_COMMIT
    ):
        raise CompletionError("result source-closure commit binding drift")
    if value.get("stage_b_execution") != STAGE_B_EXECUTION:
        raise CompletionError("result source closure claims Stage-B execution")
    rows = value.get("files")
    if not isinstance(rows, list) or len(rows) != 46 or value.get("file_count") != 46:
        raise CompletionError("result source-closure cardinality drift")
    if value.get("stage_a_file_count") != 40 or value.get("completion_file_count") != 6:
        raise CompletionError("result source-closure partition drift")
    if rows[:40] != stage_closure.get("files"):
        raise CompletionError("frozen Stage-A source rows changed")
    expected_additions = [relative for relative, _role in RESULT_SOURCE_ADDITIONS]
    if [str(row.get("path")) for row in rows[40:]] != expected_additions:
        raise CompletionError("result source additions path/order drift")
    expected_roles = [role for _relative_path, role in RESULT_SOURCE_ADDITIONS]
    if [str(row.get("role")) for row in rows[40:]] != expected_roles:
        raise CompletionError("result source additions role drift")
    if value.get("excluded_self_referential_paths") != list(EXCLUDED_SELF_REFERENTIAL_PATHS):
        raise CompletionError("self-reference exclusion drift")
    if set(EXCLUDED_SELF_REFERENTIAL_PATHS).intersection(
        str(row.get("path")) for row in rows
    ):
        raise CompletionError("result or closure self entered source rows")
    for row in rows:
        _validate_source_row(row)


def _stage_a_artifact_bindings(
    contract: Mapping[str, Any], stage_closure: Mapping[str, Any]
) -> list[dict[str, Any]]:
    closure_rows = {str(row["path"]): row for row in stage_closure["files"]}
    output: list[dict[str, Any]] = []
    for relative in STAGE_A_ARTIFACT_PATHS:
        path = REPOSITORY / relative
        row = closure_rows.get(relative)
        # The frozen Stage-A closure cannot recursively list itself.  It is
        # nevertheless a frozen Stage-A artifact and is bound directly here.
        if row is None and path != STAGE_A_SOURCE_CLOSURE_PATH:
            raise CompletionError(f"Stage-A artifact absent from frozen closure: {relative}")
        binding = _binding(path)
        if row is not None and (
            binding["sha256"] != row["sha256"]
            or binding["bytes"] != row["bytes"]
        ):
            raise CompletionError(f"Stage-A artifact drift: {relative}")
        if path.suffix == ".json":
            receipt = _load_json(path)
            validate_content_digest(receipt)
            binding["content_digest"] = receipt["content_digest"]
        output.append(binding)
    if contract.get("content_digest") != next(
        binding["content_digest"]
        for binding in output
        if binding["path"] == _relative(STAGE_A_CONTRACT_PATH)
    ):
        raise CompletionError("Stage-A contract artifact content drift")
    return output


def _bound_evidence_bytes(diagnostic: Mapping[str, Any]) -> int:
    closure = diagnostic.get("input_closure", {})
    bindings = (
        closure.get("single_coverage_errors", {}),
        closure.get("single_per_link", {}),
        closure.get("multi_coverage_errors", {}),
        closure.get("multi_per_link", {}),
    )
    values = [int(binding.get("bytes", -1)) for binding in bindings]
    if any(value < 0 for value in values):
        raise CompletionError("bound evidence byte accounting missing")
    return sum(values)


def build_result(
    contract: Mapping[str, Any],
    stage_closure: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    result_source_closure: Mapping[str, Any],
) -> dict[str, Any]:
    heldout = extract_heldout_true_future(diagnostic)
    calibration = extract_calibration_q95(diagnostic)
    exact = diagnostic["exact_contact_event_diagnostic"]
    result = attach_content_digest(
        {
            "schema_version": COMPLETION_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "stage": "STAGE_A_COMPLETED_REQUIREMENTS_REVIEW_WITH_DEVELOPMENT_DIAGNOSTIC",
            "claim_bearing": False,
            "development_only": True,
            "commit_bindings": {
                "predecessor_result_commit": PREDECESSOR_RESULT_COMMIT,
                "stage_a_freeze_commit": STAGE_A_FREEZE_COMMIT,
                "future_result_commit_bound": False,
                "future_result_commit": None,
            },
            "artifact_bindings": {
                "stage_a_artifacts": _stage_a_artifact_bindings(
                    contract, stage_closure
                ),
                "diagnostic_json": _binding(
                    DIAGNOSTIC_JSON_PATH,
                    content_digest=str(diagnostic["content_digest"]),
                ),
                "diagnostic_markdown": _binding(DIAGNOSTIC_MARKDOWN_PATH),
                "result_source_closure": _binding(
                    RESULT_SOURCE_CLOSURE_PATH,
                    content_digest=str(result_source_closure["content_digest"]),
                ),
            },
            "stage_a_result": {
                "primary_classification": PRIMARY_CLASSIFICATION,
                "secondary_classifications": list(
                    stage_a.ALLOWED_SECONDARY_CLASSIFICATIONS
                ),
                "requirements_sufficiency_gate": copy.deepcopy(
                    contract["requirements_sufficiency_gate"]
                ),
                "scope_decision": copy.deepcopy(contract["scope_decision"]),
                "protected_links": 13,
                "protected_collision_components": 27,
                "stage_b": {
                    "status": STAGE_B_STATUS,
                    "authorized": False,
                    "execution": STAGE_B_EXECUTION,
                    "blockers": copy.deepcopy(contract["stage_b_gate"]["blockers"]),
                    "prohibitions": copy.deepcopy(
                        contract["stage_b_gate"]["prohibitions"]
                    ),
                },
            },
            "preserved_predecessor_authority": copy.deepcopy(
                contract["preserved_predecessor_authority"]
            ),
            "diagnostic_result": {
                "experiment_id": diagnostic["experiment_id"],
                "claim_bearing": False,
                "development_only": True,
                "cause_ids": list(EXPECTED_DIAGNOSTIC_CAUSES),
                "supported_cause_ids": copy.deepcopy(
                    diagnostic["supported_diagnostic_cause_ids"]
                ),
                "cause_evidence_counts": copy.deepcopy(
                    diagnostic["diagnostic_cause_evidence_counts"]
                ),
                "cause_evidence_detail": copy.deepcopy(
                    diagnostic["diagnostic_cause_evidence_detail"]
                ),
                "exact_contact_population": {
                    "population": exact["population"],
                    "states": exact["states"],
                    "frozen_contact_events": exact["frozen_contact_events"],
                    "resolved_contact_events": exact["resolved_contact_events"],
                    "unresolved_contact_events": exact["unresolved_contact_events"],
                    "by_link": copy.deepcopy(exact["exact_contact_events_by_link"]),
                    "by_collision_component": copy.deepcopy(
                        exact["exact_contact_events_by_collision_component"]
                    ),
                },
                "heldout_true_future_support_and_residuals": heldout,
                "calibration_only_q95": calibration,
                "heldout_mismatch_summary": copy.deepcopy(
                    diagnostic["heldout_mismatch_diagnostic"]
                ),
                "unresolved_errors": {
                    "unresolved_exact_contact_attribution_events": exact[
                        "unresolved_contact_events"
                    ],
                    "existing_coverage_error_counts": copy.deepcopy(
                        diagnostic["existing_coverage_error_counts"]
                    ),
                    "overlap_policy": diagnostic[
                        "diagnostic_cause_evidence_detail"
                    ]["unresolved_overlap_policy"],
                },
                "point_primitive_tolerance_m": diagnostic[
                    "point_primitive_tolerance_m"
                ],
                "limitations": copy.deepcopy(diagnostic["limitations"]),
                "interpretation_boundary": copy.deepcopy(
                    diagnostic["interpretation_boundary"]
                ),
            },
            "next_decision": {
                "classification": NEXT_DECISION,
                "required_action": (
                    "Acquire and approve the missing application, consequence, "
                    "platform-parity, stopping, recovery, task-performance, and "
                    "assurance requirements before any Stage-B scope change."
                ),
                "requirements_to_acquire": copy.deepcopy(
                    contract["stage_b_gate"]["blockers"]
                ),
                "stage_b_scientific_execution_authorized": False,
            },
            "runtime_and_storage": {
                "diagnostic_runtime_s": diagnostic["runtime_s"],
                "diagnostic_workers": diagnostic["workers"],
                "diagnostic_json_bytes": DIAGNOSTIC_JSON_PATH.stat().st_size,
                "diagnostic_markdown_bytes": DIAGNOSTIC_MARKDOWN_PATH.stat().st_size,
                "result_source_closure_bytes": RESULT_SOURCE_CLOSURE_PATH.stat().st_size,
                "source_closure_member_bytes": sum(
                    int(row["bytes"]) for row in result_source_closure["files"]
                ),
                "bound_evidence_ledger_bytes": _bound_evidence_bytes(diagnostic),
                "completion_runtime_s": "NOT_RECORDED_IN_DETERMINISTIC_RESULT",
                "result_file_bytes_and_sha256_embedded": False,
                "result_file_self_reference_exclusion": True,
            },
            "prohibited_action_counters": copy.deepcopy(
                diagnostic["prohibited_action_counters"]
            ),
            "custody": {
                "stage_a_artifacts_modified": False,
                "diagnostic_json_modified": False,
                "diagnostic_reducer_modified_by_completion": False,
                "protected_scope_or_contact_labels_changed": False,
                "stage_b_run": False,
                "new_geometry_or_raycasting": False,
                "fresh_panel_collected": False,
                "model_training_or_jepa_g2_open": False,
                "memory_navigation_routing_or_beacon_executed": False,
                "result_and_closure_self_reference_excluded": True,
                "diagnostic_development_attempts": {
                    "authoritative_receipt": _binding(
                        DIAGNOSTIC_JSON_PATH,
                        content_digest=str(diagnostic["content_digest"]),
                    ),
                    "attempts": [
                        {
                            "count": 2,
                            "status": "FAIL_CLOSED_NO_ARTIFACT",
                            "reasons": [
                                "wrong multi-origin materialization-index binding",
                                "scalar predecessor materialization-index axes",
                            ],
                        },
                        {
                            "count": 2,
                            "status": "PROVISIONAL_UNCOMMITTED_RECEIPT_DELETED_AND_REGENERATED",
                            "hardening": [
                                "conservative unresolved-cause accounting",
                                "support-only residual statistics",
                                "attributable-link 50-step calibration minima",
                            ],
                        },
                    ],
                    "attempts_used_existing_evidence_only": True,
                    "simulator_steps": 0,
                    "raycasts": 0,
                    "threshold_reselections": 0,
                    "model_steps": 0,
                    "scope_or_label_changes": 0,
                },
            },
        }
    )
    validate_result(
        result,
        contract=contract,
        diagnostic=diagnostic,
        result_source_closure=result_source_closure,
    )
    return result


def validate_result(
    result: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    result_source_closure: Mapping[str, Any],
) -> None:
    validate_content_digest(result)
    if result.get("schema_version") != COMPLETION_SCHEMA:
        raise CompletionError("completion result schema drift")
    if result.get("experiment_id") != EXPERIMENT_ID:
        raise CompletionError("completion result experiment drift")
    if result.get("claim_bearing") is not False or result.get("development_only") is not True:
        raise CompletionError("diagnostic completion claim boundary drift")
    commits = result.get("commit_bindings", {})
    if commits != {
        "predecessor_result_commit": PREDECESSOR_RESULT_COMMIT,
        "stage_a_freeze_commit": STAGE_A_FREEZE_COMMIT,
        "future_result_commit_bound": False,
        "future_result_commit": None,
    }:
        raise CompletionError("completion commit binding drift")
    stage_result = result.get("stage_a_result", {})
    if stage_result.get("primary_classification") != PRIMARY_CLASSIFICATION:
        raise CompletionError("completion primary classification drift")
    if tuple(stage_result.get("secondary_classifications", ())) != tuple(
        stage_a.ALLOWED_SECONDARY_CLASSIFICATIONS
    ):
        raise CompletionError("completion secondary classifications drift")
    gate = stage_result.get("requirements_sufficiency_gate", {})
    if (
        gate.get("criteria_passed") != 8
        or gate.get("criteria_total") != 9
        or gate.get("pass") is not False
    ):
        raise CompletionError("requirements sufficiency gate drift")
    stage_b = stage_result.get("stage_b", {})
    if (
        stage_b.get("status") != STAGE_B_STATUS
        or stage_b.get("authorized") is not False
        or stage_b.get("execution") != STAGE_B_EXECUTION
    ):
        raise CompletionError("Stage B execution or authority drift")
    scope_decision = stage_result.get("scope_decision", {})
    if scope_decision != contract.get("scope_decision"):
        raise CompletionError("completion scope decision drift")
    if result.get("preserved_predecessor_authority") != contract.get(
        "preserved_predecessor_authority"
    ):
        raise CompletionError("preserved predecessor authority drift")
    diag_result = result.get("diagnostic_result", {})
    if tuple(diag_result.get("cause_ids", ())) != EXPECTED_DIAGNOSTIC_CAUSES:
        raise CompletionError("completion diagnostic cause vocabulary drift")
    if tuple(diag_result.get("supported_cause_ids", ())) != EXPECTED_DIAGNOSTIC_CAUSES:
        raise CompletionError("completion did not preserve all supported causes")
    if diag_result.get("cause_evidence_counts") != diagnostic.get(
        "diagnostic_cause_evidence_counts"
    ):
        raise CompletionError("completion diagnostic counts drift")
    if diag_result.get("heldout_true_future_support_and_residuals") != extract_heldout_true_future(
        diagnostic
    ):
        raise CompletionError("completion heldout diagnostic drift")
    if diag_result.get("calibration_only_q95") != extract_calibration_q95(diagnostic):
        raise CompletionError("completion calibration q95 drift")
    if diag_result.get("unresolved_errors", {}).get(
        "unresolved_exact_contact_attribution_events"
    ) != diagnostic["exact_contact_event_diagnostic"]["unresolved_contact_events"]:
        raise CompletionError("completion unresolved-error count drift")
    if any(int(value) != 0 for value in result.get("prohibited_action_counters", {}).values()):
        raise CompletionError("completion prohibited-action counter nonzero")
    custody = result.get("custody", {})
    required_false = (
        "stage_a_artifacts_modified",
        "diagnostic_json_modified",
        "diagnostic_reducer_modified_by_completion",
        "protected_scope_or_contact_labels_changed",
        "stage_b_run",
        "new_geometry_or_raycasting",
        "fresh_panel_collected",
        "model_training_or_jepa_g2_open",
        "memory_navigation_routing_or_beacon_executed",
    )
    if any(custody.get(key) is not False for key in required_false):
        raise CompletionError("completion custody breach")
    if custody.get("result_and_closure_self_reference_excluded") is not True:
        raise CompletionError("completion self-reference exclusion missing")
    next_decision = result.get("next_decision", {})
    if (
        next_decision.get("classification") != NEXT_DECISION
        or next_decision.get("stage_b_scientific_execution_authorized") is not False
        or next_decision.get("requirements_to_acquire")
        != contract["stage_b_gate"]["blockers"]
    ):
        raise CompletionError("next requirements-acquisition decision drift")
    bindings = result.get("artifact_bindings", {})
    expected_diagnostic = _binding(
        DIAGNOSTIC_JSON_PATH, content_digest=str(diagnostic["content_digest"])
    )
    if bindings.get("diagnostic_json") != expected_diagnostic:
        raise CompletionError("completion diagnostic file binding drift")
    expected_closure = _binding(
        RESULT_SOURCE_CLOSURE_PATH,
        content_digest=str(result_source_closure["content_digest"]),
    )
    if bindings.get("result_source_closure") != expected_closure:
        raise CompletionError("completion source-closure binding drift")


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CompletionError("nonfinite Markdown value")
        return format(value, ".12g")
    return str(value)


def _escape_cell(value: Any) -> str:
    return _fmt(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(
    contract: Mapping[str, Any], diagnostic: Mapping[str, Any]
) -> str:
    heldout = extract_heldout_true_future(diagnostic)
    calibration = extract_calibration_q95(diagnostic)
    exact = diagnostic["exact_contact_event_diagnostic"]
    lines: list[str] = [
        "# Contact-critical geometry mismatch audit V1",
        "",
        "## Disposition",
        "",
        f"Primary Stage-A classification: `{PRIMARY_CLASSIFICATION}`.",
        "",
        "This is a development-only, post-freeze diagnostic. It does not revise "
        "the protected-contact scope, contact labels, or requirements decision. "
        f"Stage B is `{STAGE_B_EXECUTION}` and remains `{STAGE_B_STATUS}`.",
        "",
        "The diagnostic supports all five prospectively authorized descriptive causes. "
        "They explain evidence limitations; they are not authority to narrow a safety scope.",
        "",
        "## Immutable bindings",
        "",
        f"- Predecessor result commit: `{PREDECESSOR_RESULT_COMMIT}`",
        f"- Stage-A freeze commit: `{STAGE_A_FREEZE_COMMIT}`",
        f"- Stage-A contract SHA-256: `{sha256_file(STAGE_A_CONTRACT_PATH)}`",
        f"- Stage-A contract content digest: `{contract['content_digest']}`",
        f"- Diagnostic JSON SHA-256: `{sha256_file(DIAGNOSTIC_JSON_PATH)}`",
        f"- Diagnostic JSON content digest: `{diagnostic['content_digest']}`",
        f"- Diagnostic runtime: `{_fmt(diagnostic['runtime_s'])}` s with "
        f"`{diagnostic['workers']}` workers",
        "",
        "## Preserved scientific authority",
        "",
        contract["preserved_predecessor_authority"]["exact_geometry_finding"],
        "",
        contract["preserved_predecessor_authority"]["range_finding"],
        "",
        "Preserved classifications:",
        "",
    ]
    lines.extend(
        f"- `{classification}`"
        for classification in contract["preserved_predecessor_authority"][
            "classifications"
        ]
    )
    lines.extend(
        [
            "",
            "## Diagnostic-cause evidence",
            "",
            "| Cause | Evidence count | Mismatch-row count |",
            "|---|---:|---:|",
        ]
    )
    evidence = diagnostic["diagnostic_cause_evidence_counts"]
    mismatch = diagnostic["diagnostic_cause_evidence_detail"][
        "contact_classification_mismatch_rows"
    ]
    for cause in EXPECTED_DIAGNOSTIC_CAUSES:
        lines.append(f"| `{cause}` | {evidence[cause]} | {mismatch[cause]} |")
    lines.extend(
        [
            "",
            "The unresolved count additionally includes "
            f"{diagnostic['diagnostic_cause_evidence_detail']['unresolved_exact_contact_attribution_events']} "
            "frozen contact events without an authoritative contact step/link. Overlap policy: "
            + diagnostic["diagnostic_cause_evidence_detail"]["unresolved_overlap_policy"]
            + ".",
            "",
            "## Exact-contact population",
            "",
            f"Across {exact['states']} states there are {exact['frozen_contact_events']} frozen "
            f"contact events: {exact['resolved_contact_events']} have resolved exact step/link "
            f"attribution and {exact['unresolved_contact_events']} remain unresolved.",
            "",
            "### Contact events by protected link",
            "",
            "| Link | Events |",
            "|---|---:|",
        ]
    )
    for link, count in sorted(exact["exact_contact_events_by_link"].items()):
        lines.append(f"| `{link}` | {count} |")
    lines.extend(
        [
            "",
            "## Development-held-out true-future support and residuals",
            "",
            "The denominator in every row is the same 579 held-out exact contact events. "
            "Residuals are observed minus exact clearance and are computed only for "
            "patch-supported events; unsupported events are not silently entered as finite residuals.",
            "",
            "| Condition | Contacts | Supported | Support rate | Unobserved | Self-occluded unobserved | Event-time supported | Finite-scan inherited | Point/primitive mismatch | Risk-underestimate | Residual n | Residual P50 m | Residual P95 m | Residual P99 m | Absolute P95 m | Threshold misses | Nearest-rank Q95 incl. unsupported |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition, row in heldout.items():
        residual = row["observed_minus_exact_statistics_m"]
        values = (
            condition,
            row["contact_events"],
            row["patch_supported_events"],
            row["patch_support_rate"],
            row["patch_unobserved_events"],
            row["patch_unobserved_self_occluded_events"],
            row["event_time_supported_events"],
            row["finite_scan_inherited_events"],
            row["point_primitive_mismatch_events"],
            row["point_primitive_risk_underestimate_events"],
            residual["count"],
            residual["p50"],
            residual["p95"],
            residual["p99"],
            residual["absolute_p95"],
            row["threshold_patch_miss_events"],
            row["nearest_rank_q95_including_unsupported"],
        )
        lines.append("| " + " | ".join(_escape_cell(value) for value in values) + " |")
    lines.extend(
        [
            "",
            "## Calibration-only nearest-rank Q95 diagnostic",
            "",
            "These rows use only the frozen internal-calibration role. They do not "
            "reselect or interpolate a threshold, and held-out values have no selection authority. "
            "Unsupported attributable-link minima are treated as positive infinity.",
        ]
    )
    for group_name, title in (
        ("per_link", "Per protected link"),
        ("per_collision_component", "Per collision component"),
    ):
        lines.extend(
            [
                "",
                f"### {title}",
                "",
                "| Condition / mode / identity | Positives | Finite minima | Unsupported | Frozen global threshold m | Global threshold attributable-link recall | Nearest-rank Q95 m | Q95 minus global m | Status |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for key, row in sorted(calibration[group_name].items()):
            values = (
                key,
                row["positives"],
                row["finite_attributable_link_minimum_scores"],
                row["unsupported_attributable_link_minimum_scores"],
                row["frozen_global_threshold_m"],
                row["global_threshold_attributable_link_clearance_recall"],
                row["nearest_rank_q95_m"],
                row["q95_minus_frozen_global_threshold_m"],
                row["threshold_shift_status"],
            )
            lines.append("| " + " | ".join(_escape_cell(value) for value in values) + " |")
    errors = diagnostic["existing_coverage_error_counts"]
    lines.extend(
        [
            "",
            "## Existing unresolved and attributed errors",
            "",
            "| Evidence source | Existing error class | Rows |",
            "|---|---|---:|",
        ]
    )
    for source in ("single_origin", "multi_origin"):
        for error_class, count in sorted(errors[source].items()):
            lines.append(f"| `{source}` | `{error_class}` | {count} |")
    lines.extend(
        [
            "",
            f"The multi-origin ledger contains {errors['multi_origin_scope_counts']['CONTACT_CLASSIFICATION']} "
            "contact-classification error rows and "
            f"{errors['multi_origin_scope_counts']['DECISION_LEVEL']} decision-level rows. "
            "Decision failures are preserved exactly in the canonical diagnostic JSON.",
            "",
            "## Interpretation and next decision",
            "",
            "The observed heterogeneity, unobserved contact-critical patches, self-occlusion, "
            "and point-to-primitive residuals are descriptive evidence. They do not identify "
            "a valid link-specific threshold, prove physical severity, or authorize removing "
            "a link, shape, direction, swept volume, or label.",
            "",
            f"Next decision: `{NEXT_DECISION}`. Acquire and approve every Stage-B blocker "
            "before any scope revision or new scientific Stage-B execution:",
            "",
        ]
    )
    lines.extend(f"- {blocker}" for blocker in contract["stage_b_gate"]["blockers"])
    lines.extend(
        [
            "",
            "## Custody",
            "",
            "All prohibited-action counters are zero. No new geometry, raycast, simulator "
            "step, panel, model training, JEPA/G2 access, memory, navigation, routing, or "
            "beacon execution occurred. Stage A remains byte-frozen and Stage B was not run.",
            "",
            "Two earlier reducer invocations failed closed without writing an artifact "
            "(materialization-index binding and scalar-axis validation defects). Two "
            "provisional, uncommitted diagnostic receipts were deleted and regenerated "
            "after conservative unresolved-cause and support-only/calibration-minimum "
            "statistical hardening. All were read-only reductions of existing evidence; "
            "only the immutable binding above is authoritative.",
            "",
        ]
    )
    return "\n".join(lines)


def generate() -> dict[str, Any]:
    contract, stage_closure, diagnostic = load_and_validate_inputs()
    markdown = render_markdown(contract, diagnostic).encode("utf-8")
    _atomic_write(DIAGNOSTIC_MARKDOWN_PATH, markdown)
    closure = build_result_source_closure(stage_closure)
    _atomic_write(RESULT_SOURCE_CLOSURE_PATH, canonical_file_bytes(closure))
    result = build_result(contract, stage_closure, diagnostic, closure)
    _atomic_write(RESULT_PATH, canonical_file_bytes(result))
    check()
    return result


def check() -> dict[str, Any]:
    contract, stage_closure, diagnostic = load_and_validate_inputs()
    expected_markdown = render_markdown(contract, diagnostic).encode("utf-8")
    if not DIAGNOSTIC_MARKDOWN_PATH.is_file():
        raise CompletionError("diagnostic Markdown missing")
    if DIAGNOSTIC_MARKDOWN_PATH.read_bytes() != expected_markdown:
        raise CompletionError("diagnostic Markdown byte regeneration drift")

    if not RESULT_SOURCE_CLOSURE_PATH.is_file():
        raise CompletionError("result source closure missing")
    closure = _load_json(RESULT_SOURCE_CLOSURE_PATH)
    validate_result_source_closure(closure, stage_closure=stage_closure)
    expected_closure = build_result_source_closure(stage_closure)
    if RESULT_SOURCE_CLOSURE_PATH.read_bytes() != canonical_file_bytes(expected_closure):
        raise CompletionError("result source closure byte regeneration drift")

    if not RESULT_PATH.is_file():
        raise CompletionError("completion result missing")
    result = _load_json(RESULT_PATH)
    validate_result(
        result,
        contract=contract,
        diagnostic=diagnostic,
        result_source_closure=closure,
    )
    expected_result = build_result(contract, stage_closure, diagnostic, closure)
    if RESULT_PATH.read_bytes() != canonical_file_bytes(expected_result):
        raise CompletionError("completion result byte regeneration drift")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "check"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = generate() if arguments.command == "generate" else check()
    print(
        json.dumps(
            {
                "status": "PASS",
                "result": _relative(RESULT_PATH),
                "result_sha256": sha256_file(RESULT_PATH),
                "result_content_digest": result["content_digest"],
                "source_closure_sha256": sha256_file(RESULT_SOURCE_CLOSURE_PATH),
                "source_closure_content_digest": _load_json(
                    RESULT_SOURCE_CLOSURE_PATH
                )["content_digest"],
                "diagnostic_markdown_sha256": sha256_file(
                    DIAGNOSTIC_MARKDOWN_PATH
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CompletionError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
