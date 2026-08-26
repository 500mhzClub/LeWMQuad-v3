#!/usr/bin/env python3
"""Read-only audit of contact-critical range/primitive mismatches.

The audit consumes only already-materialised exact, single-origin, and
multi-origin evidence.  It never imports Genesis, opens G2/JEPA, casts a ray,
changes a contact label, changes the protected scope, or trains a model.

The executable is intentionally split into small deterministic reducers so
that focused tests can exercise the science without opening the canonical
evidence.  ``--execute`` is the only path that reads the full evidence set.
Writing is opt-in and restricted to the one canonical diagnostic JSON path;
no Stage-A artifact can be overwritten.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "CONTACT_CRITICAL_GEOMETRY_MISMATCH_AUDIT_V1"
SCHEMA_VERSION = "contact_critical_geometry_mismatch_audit_v1.result.v1"
STAGE_A_EXPERIMENT_ID = "PROTECTED_CONTACT_SCOPE_REQUIREMENTS_REVIEW_V1"
STAGE_A_FREEZE_COMMIT = "e9e8c41a327ddbe51c38fa04f05ae1d30266720b"
STAGE_A_CONTRACT_SHA256 = "f51f69f5c48bd2b4b256bb229c0156d8c801dbfdeb04eb9a52ab9249e212c5cb"
STAGE_A_PRIMARY = "PROTECTED_CONTACT_SCOPE_REQUIREMENTS_UNRESOLVED"
STAGE_A_STATUS = "STAGE_A_REQUIREMENTS_ONLY"
STAGE_B_STATUS = "STAGE_B_NOT_AUTHORIZED"

SINGLE_RESULT_SHA256 = "8844c0ee5a8bcdd28f505d64a670b5dee595933af05a00290f327cb8f3702019"
MULTI_RESULT_SHA256 = "082027dde3067a5afdc20f21f2fd859f630aac31d9c15252dc30333183cffeec"
EXACT_RESULT_SHA256 = "e3eb1a8a64f147eed3bb29d5a778b799f86b45e178c8400b92adf4c253e3f5b4"
EXACT_GEOMETRY_INDEX_SHA256 = (
    "67b473e6d0e0c422e4b2916b800399c98d5ec60383d398ff907fe2b8909a1d7f"
)
EXACT_RESULT_CONTENT_DIGEST = (
    "4b33b3fc38392728b097ea096607347a4879121459a76440e54471a42d63a4c8"
)
EXACT_GEOMETRY_INDEX_CONTENT_DIGEST = (
    "0ac0880f2b65914d2f98ce6e5f493678a0aa96abb1d35edc3854fc20aaf93545"
)
SINGLE_MATERIALIZATION_INDEX_SHA256 = (
    "4e35e813d6f1bc2593c6a889a24e3de952b2797ce48f2c28224cfe16e5af398d"
)
MULTI_INDEX_SHA256 = {
    "dual": "1cc9f49821b78a35204baaef30182a580db4ae724f08f80a1e21c4bbc9a8af99",
    "three": "757fa10922da85d2fb1674a42d9b788590db3c82f200e443f4ad2a05c22133ce",
    "diagnostic": "0d3d86696f626840af84364cf0e8d2cba7921498f754c38eb8127a08a2f0beb2",
}

CONTACT_CRITICAL_PATCH_UNOBSERVED = "CONTACT_CRITICAL_PATCH_UNOBSERVED"
POINT_TO_PRIMITIVE_DISTANCE_MISMATCH = "POINT_TO_PRIMITIVE_DISTANCE_MISMATCH"
GLOBAL_THRESHOLD_HETEROGENEITY = "GLOBAL_THRESHOLD_HETEROGENEITY"
SELF_OCCLUSION_AT_CONTACT_CRITICAL_REGION = (
    "SELF_OCCLUSION_AT_CONTACT_CRITICAL_REGION"
)
UNRESOLVED_GEOMETRIC_MISMATCH = "UNRESOLVED_GEOMETRIC_MISMATCH"

DIAGNOSTIC_CAUSE_IDS = (
    CONTACT_CRITICAL_PATCH_UNOBSERVED,
    POINT_TO_PRIMITIVE_DISTANCE_MISMATCH,
    GLOBAL_THRESHOLD_HETEROGENEITY,
    SELF_OCCLUSION_AT_CONTACT_CRITICAL_REGION,
    UNRESOLVED_GEOMETRIC_MISMATCH,
)

SINGLE_CONDITION_IDS = (
    "CURRENT_SPARSE_RANGE_BASELINE",
    "REALISTIC_PLATFORM_SCAN",
    "DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT",
    "DENSE_BODY_CENTRIC_SINGLE_ORIGIN",
)
DUAL_CONDITION_IDS = (
    "DUAL_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND",
    "DUAL_DENSE_L2_FOV_UPPER_BOUND",
    "DUAL_REALISTIC_L2_SCAN",
)
THREE_CONDITION_IDS = (
    "THREE_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND",
    "THREE_DENSE_L2_FOV_UPPER_BOUND",
    "THREE_REALISTIC_L2_SCAN",
)
DIAGNOSTIC_CONDITION_ID = "ALL_FOUR_DENSE_SPHERICAL_DIAGNOSTIC"
MULTI_CONDITION_IDS = DUAL_CONDITION_IDS + THREE_CONDITION_IDS + (
    DIAGNOSTIC_CONDITION_ID,
)
EVIDENCE_MODE_IDS = (
    "PLANNING_TIME_CAUSAL_CLOUD",
    "TRUE_FUTURE_OBSERVABILITY_CLOUD",
)

PROTECTED_LINK_NAMES = (
    "base",
    "FL_hip", "FR_hip", "RL_hip", "RR_hip",
    "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
    "FL_calf", "FR_calf", "RL_calf", "RR_calf",
)
LINK_REGION = {
    "base": "TRUNK",
    **{name: "HIP" for name in PROTECTED_LINK_NAMES if name.endswith("_hip")},
    **{name: "THIGH" for name in PROTECTED_LINK_NAMES if name.endswith("_thigh")},
    **{name: "CALF" for name in PROTECTED_LINK_NAMES if name.endswith("_calf")},
}
POINT_PRIMITIVE_TOLERANCE_M = 1.0e-6
CANONICAL_WORKERS = 32
DEFAULT_WORKERS = CANONICAL_WORKERS
CANONICAL_OUTPUT_RELATIVE_PATH = Path(
    "docs/lewm_contact_critical_geometry_mismatch_audit_v1.json"
)


class AuditError(RuntimeError):
    """Fail-closed audit validation error."""


@dataclass(frozen=True)
class AuditPaths:
    repo_root: Path
    stage_a_contract: Path
    exact_result: Path
    exact_geometry_index: Path
    single_result: Path
    multi_result: Path
    single_materialization_index: Path
    multi_dual_index: Path
    multi_three_index: Path
    multi_diagnostic_index: Path


def default_paths(repo_root: Path = REPO_ROOT) -> AuditPaths:
    recovery = Path(
        "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3"
    )
    single = recovery / "body_centric_range_coverage_qualification_v1"
    multi = recovery / "minimum_multi_origin_body_range_coverage_qualification_v1"
    return AuditPaths(
        repo_root=repo_root.resolve(),
        stage_a_contract=repo_root / "docs/lewm_protected_contact_scope_contract_v1.json",
        exact_result=repo_root
        / ".generated/explicit_per_link_geometric_micro_state_upper_bound_v1/result.json",
        exact_geometry_index=repo_root
        / ".generated/explicit_per_link_geometric_micro_state_upper_bound_v1/geometry_index.json",
        single_result=repo_root
        / "docs/lewm_go2_body_centric_range_coverage_qualification_v1_result_2026-08-25.json",
        multi_result=repo_root
        / "docs/lewm_go2_minimum_multi_origin_body_range_coverage_qualification_v1_result_2026-08-26.json",
        single_materialization_index=single / "materialization_index.json",
        multi_dual_index=multi / "materialization/dual_index.json",
        multi_three_index=multi / "materialization/three_index.json",
        multi_diagnostic_index=multi / "materialization/diagnostic_index.json",
    )


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
        raise AuditError(f"value is not canonical JSON: {exc}") from exc


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def attach_content_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    if "content_digest" in value:
        raise AuditError("content_digest may be attached exactly once")
    result = dict(value)
    result["content_digest"] = canonical_digest(result)
    return result


def validate_content_digest(value: Mapping[str, Any]) -> None:
    observed = value.get("content_digest")
    if not isinstance(observed, str) or len(observed) != 64:
        raise AuditError("missing content_digest")
    core = {key: item for key, item in value.items() if key != "content_digest"}
    if canonical_digest(core) != observed:
        raise AuditError("content_digest mismatch")


def sha256_file(path: Path, *, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot load JSON receipt {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"JSON receipt must be an object: {path}")
    return value


def validate_stage_a_contract(
    contract: Mapping[str, Any], *, expected_file_sha256: str | None = None,
    contract_path: Path | None = None,
) -> dict[str, Any]:
    """Validate the frozen unresolved requirements barrier before outcomes."""

    if expected_file_sha256 is not None:
        if contract_path is None or sha256_file(contract_path) != expected_file_sha256:
            raise AuditError("Stage-A contract file SHA-256 drift")
    validate_content_digest(contract)
    if contract.get("experiment_id") != STAGE_A_EXPERIMENT_ID:
        raise AuditError("Stage-A experiment identity drift")
    if contract.get("stage") != STAGE_A_STATUS:
        raise AuditError("requirements audit may run only after frozen Stage A")
    classifications = contract.get("classifications", {})
    if classifications.get("primary_exactly_one") != STAGE_A_PRIMARY:
        raise AuditError("Stage-A primary must remain requirements unresolved")
    gate = contract.get("stage_b_gate", {})
    if gate.get("status") != STAGE_B_STATUS or gate.get("authorized") is not False:
        raise AuditError("Stage B must remain unauthorized")
    decision = contract.get("scope_decision", {})
    if (
        decision.get("protected_links_removed") != []
        or decision.get("protected_shapes_removed") != []
        or decision.get("contact_labels_changed") is not False
        or decision.get("scope_narrowing_authorized") is not False
    ):
        raise AuditError("Stage-A protected scope or label changed")
    inventory = contract.get("component_inventory", {})
    if (
        inventory.get("protected_link_count") != 13
        or inventory.get("collision_component_count") != 27
        or tuple(inventory.get("protected_link_names", ())) != PROTECTED_LINK_NAMES
        or len(inventory.get("components", ())) != 27
    ):
        raise AuditError("Stage-A 13-link/27-component inventory drift")
    indices = [item.get("geom_index") for item in inventory["components"]]
    if indices != list(range(27)):
        raise AuditError("Stage-A collision component order drift")
    return {
        "experiment_id": STAGE_A_EXPERIMENT_ID,
        "freeze_commit": STAGE_A_FREEZE_COMMIT,
        "contract_content_digest": contract["content_digest"],
        "contract_file_sha256": expected_file_sha256,
        "primary_classification": STAGE_A_PRIMARY,
        "stage_b_status": STAGE_B_STATUS,
        "stage_b_authorized": False,
        "scope_narrowing_authorized": False,
        "contact_label_change_authorized": False,
        "protected_links": 13,
        "protected_collision_components": 27,
    }


def _thresholds(result: Mapping[str, Any]) -> dict[tuple[str, str], float]:
    output: dict[tuple[str, str], float] = {}
    for condition_id, modes in result.get("calibration_thresholds", {}).items():
        for mode_id, receipt in modes.items():
            try:
                value = float(receipt["selected"]["contact"]["threshold_m"])
            except (KeyError, TypeError, ValueError) as exc:
                raise AuditError(
                    f"missing frozen threshold: {condition_id}/{mode_id}"
                ) from exc
            if not math.isfinite(value):
                raise AuditError("frozen threshold must be finite")
            output[(str(condition_id), str(mode_id))] = value
    return output


def _validate_file_binding(binding: Mapping[str, Any], *, verify_hash: bool) -> Path:
    path = Path(str(binding.get("path", "")))
    if not path.is_file():
        raise AuditError(f"bound evidence file missing: {path}")
    if path.stat().st_size != int(binding.get("bytes", -1)):
        raise AuditError(f"bound evidence byte count drift: {path}")
    if int(binding.get("rows", -1)) <= 0:
        raise AuditError(f"bound evidence row count invalid: {path}")
    if verify_hash and sha256_file(path) != binding.get("sha256"):
        raise AuditError(f"bound evidence SHA-256 drift: {path}")
    return path


def _validate_index(
    path: Path,
    *, expected_sha256: str,
    expected_conditions: Sequence[str] | None,
    expected_records: int,
    verify_hash: bool,
) -> dict[str, Any]:
    if not path.is_file():
        raise AuditError(f"materialization index missing: {path}")
    if verify_hash and sha256_file(path) != expected_sha256:
        raise AuditError(f"materialization index SHA-256 drift: {path}")
    index = _load_json(path)
    if index.get("states") != 176 or index.get("transitions") != 29470:
        raise AuditError("materialization corpus cardinality drift")
    records = index.get("records")
    if not isinstance(records, list) or len(records) != expected_records:
        raise AuditError("materialization record cardinality drift")
    if expected_conditions is not None and tuple(index.get("conditions", ())) != tuple(
        expected_conditions
    ):
        raise AuditError("materialization condition identity/order drift")
    return index


def validate_input_closure(
    paths: AuditPaths, *, verify_hashes: bool,
    strict_frozen_files: bool = True,
) -> dict[str, Any]:
    stage_a = _load_json(paths.stage_a_contract)
    stage_binding = validate_stage_a_contract(
        stage_a,
        expected_file_sha256=(STAGE_A_CONTRACT_SHA256 if strict_frozen_files else None),
        contract_path=paths.stage_a_contract,
    )
    if strict_frozen_files and sha256_file(paths.exact_result) != EXACT_RESULT_SHA256:
        raise AuditError("exact upper-bound result SHA-256 drift")
    if (
        strict_frozen_files
        and sha256_file(paths.exact_geometry_index) != EXACT_GEOMETRY_INDEX_SHA256
    ):
        raise AuditError("exact geometry index SHA-256 drift")
    if strict_frozen_files and sha256_file(paths.single_result) != SINGLE_RESULT_SHA256:
        raise AuditError("single-origin tracked result SHA-256 drift")
    if strict_frozen_files and sha256_file(paths.multi_result) != MULTI_RESULT_SHA256:
        raise AuditError("multi-origin tracked result SHA-256 drift")
    exact_result = _load_json(paths.exact_result)
    exact_index = _load_json(paths.exact_geometry_index)
    if (
        exact_result.get("experiment")
        != "EXPLICIT_PER_LINK_GEOMETRIC_MICRO_STATE_UPPER_BOUND_V1"
        or exact_result.get("content_digest") != EXACT_RESULT_CONTENT_DIGEST
    ):
        raise AuditError("exact upper-bound result identity/content drift")
    if (
        exact_index.get("states") != 176
        or exact_index.get("transitions") != 29470
        or exact_index.get("content_digest") != EXACT_GEOMETRY_INDEX_CONTENT_DIGEST
        or len(exact_index.get("records", ())) != 176
    ):
        raise AuditError("exact geometry index identity/cardinality drift")
    single_result = _load_json(paths.single_result)
    multi_result = _load_json(paths.multi_result)
    if single_result.get("experiment_id") != "BODY_CENTRIC_RANGE_COVERAGE_QUALIFICATION_V1":
        raise AuditError("single-origin result identity drift")
    if multi_result.get("experiment_id") != "MINIMUM_MULTI_ORIGIN_BODY_RANGE_COVERAGE_QUALIFICATION_V1":
        raise AuditError("multi-origin result identity drift")
    single_rows = single_result.get("row_level_evidence_receipt", {})
    multi_rows = multi_result.get("row_level_evidence", {})
    single_per_link = _validate_file_binding(
        single_rows.get("per_link_evidence", {}), verify_hash=verify_hashes
    )
    multi_per_link = _validate_file_binding(
        multi_rows.get("per_link_evidence", {}), verify_hash=verify_hashes
    )
    single_errors = _validate_file_binding(
        single_result.get("coverage_error_receipt", {}), verify_hash=verify_hashes
    )
    multi_errors = _validate_file_binding(
        multi_result.get("coverage_errors", {}), verify_hash=verify_hashes
    )
    single_index_sha = str(single_rows.get("materialization_index_sha256"))
    if single_index_sha != SINGLE_MATERIALIZATION_INDEX_SHA256:
        raise AuditError("single-origin materialization index binding drift")
    single_index = _validate_index(
        paths.single_materialization_index,
        expected_sha256=single_index_sha,
        expected_conditions=None,
        expected_records=176,
        verify_hash=verify_hashes,
    )
    dual = _validate_index(
        paths.multi_dual_index,
        expected_sha256=MULTI_INDEX_SHA256["dual"],
        expected_conditions=DUAL_CONDITION_IDS,
        expected_records=528,
        verify_hash=verify_hashes,
    )
    three = _validate_index(
        paths.multi_three_index,
        expected_sha256=MULTI_INDEX_SHA256["three"],
        expected_conditions=THREE_CONDITION_IDS,
        expected_records=528,
        verify_hash=verify_hashes,
    )
    diagnostic = _validate_index(
        paths.multi_diagnostic_index,
        expected_sha256=MULTI_INDEX_SHA256["diagnostic"],
        expected_conditions=(DIAGNOSTIC_CONDITION_ID,),
        expected_records=176,
        verify_hash=verify_hashes,
    )
    single_thresholds = _thresholds(single_result)
    multi_thresholds = _thresholds(multi_result)
    expected_single_thresholds = {
        (condition, mode)
        for condition in SINGLE_CONDITION_IDS
        for mode in EVIDENCE_MODE_IDS
    }
    expected_multi_thresholds = {
        (condition, mode)
        for condition in DUAL_CONDITION_IDS + THREE_CONDITION_IDS
        for mode in EVIDENCE_MODE_IDS
    } | {(DIAGNOSTIC_CONDITION_ID, "TRUE_FUTURE_OBSERVABILITY_CLOUD")}
    if set(single_thresholds) != expected_single_thresholds:
        raise AuditError("single-origin threshold closure drift")
    if set(multi_thresholds) != expected_multi_thresholds:
        raise AuditError("multi-origin threshold closure drift")
    embedded_indices = multi_result.get("materialisation_indices", {})
    for phase, loaded in (("dual", dual), ("three", three), ("diagnostic", diagnostic)):
        embedded = embedded_indices.get(phase)
        if not isinstance(embedded, Mapping):
            raise AuditError(f"multi-origin embedded {phase} index missing")
        if (
            embedded.get("content_digest") != loaded.get("content_digest")
            or embedded.get("states") != loaded.get("states")
            or embedded.get("transitions") != loaded.get("transitions")
            or len(embedded.get("records", ())) != len(loaded.get("records", ()))
        ):
            raise AuditError(f"multi-origin embedded {phase} index drift")
    return {
        "stage_a": stage_binding,
        "exact_result": exact_result,
        "exact_index": exact_index,
        "single_result": single_result,
        "multi_result": multi_result,
        "single_index": single_index,
        "dual_index": dual,
        "three_index": three,
        "diagnostic_index": diagnostic,
        "single_thresholds": single_thresholds,
        "multi_thresholds": multi_thresholds,
        "paths": {
            "single_per_link": single_per_link,
            "multi_per_link": multi_per_link,
            "single_errors": single_errors,
            "multi_errors": multi_errors,
        },
    }


def _finite_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def classify_diagnostic_causes(
    *,
    oracle_contact: bool,
    predicted_contact: bool,
    patch_supported: bool,
    nominal_fov: bool,
    self_occluded: bool,
    exact_clearance_m: float | None,
    observed_clearance_m: float | None,
    threshold_m: float,
    tolerance_m: float = POINT_PRIMITIVE_TOLERANCE_M,
) -> dict[str, Any]:
    """Classify one already-persisted mismatch without changing its label.

    Causes are multi-label evidence flags.  ``primary_cause`` is a deterministic
    descriptive precedence, not a protected-scope decision.
    """

    if predicted_contact == oracle_contact:
        raise AuditError("diagnostic row must be a contact-classification mismatch")
    if not math.isfinite(threshold_m):
        raise AuditError("threshold must be finite")
    unobserved = not patch_supported
    self_cause = unobserved and self_occluded
    point_mismatch = bool(
        patch_supported
        and exact_clearance_m is not None
        and observed_clearance_m is not None
        and abs(observed_clearance_m - exact_clearance_m) > tolerance_m
    )
    threshold_candidate = bool(
        patch_supported
        and observed_clearance_m is not None
        and (
            (oracle_contact and not predicted_contact and observed_clearance_m > threshold_m)
            or (
                not oracle_contact
                and predicted_contact
                and observed_clearance_m <= threshold_m
            )
        )
    )
    flags = {
        CONTACT_CRITICAL_PATCH_UNOBSERVED: unobserved,
        POINT_TO_PRIMITIVE_DISTANCE_MISMATCH: point_mismatch,
        GLOBAL_THRESHOLD_HETEROGENEITY: threshold_candidate,
        SELF_OCCLUSION_AT_CONTACT_CRITICAL_REGION: self_cause,
        UNRESOLVED_GEOMETRIC_MISMATCH: False,
    }
    if self_cause:
        primary = SELF_OCCLUSION_AT_CONTACT_CRITICAL_REGION
    elif unobserved:
        primary = CONTACT_CRITICAL_PATCH_UNOBSERVED
    elif point_mismatch:
        primary = POINT_TO_PRIMITIVE_DISTANCE_MISMATCH
    elif threshold_candidate:
        primary = GLOBAL_THRESHOLD_HETEROGENEITY
    else:
        primary = UNRESOLVED_GEOMETRIC_MISMATCH
        flags[UNRESOLVED_GEOMETRIC_MISMATCH] = True
    return {
        "supported_cause_ids": [cause for cause in DIAGNOSTIC_CAUSE_IDS if flags[cause]],
        "primary_cause": primary,
        "flags": flags,
        "nominal_fov": nominal_fov,
        "scope_change_authorized": False,
    }


def _new_mismatch_bucket() -> dict[str, Any]:
    return {
        "rows": 0,
        "false_negative_rows": 0,
        "false_positive_rows": 0,
        "patch_unobserved_rows": 0,
        "outside_nominal_fov_rows": 0,
        "self_occluded_rows": 0,
        "finite_supported_clearance_pairs": 0,
        "point_primitive_mismatch_rows": 0,
        "observed_minus_exact_m": [],
        "primary_causes": Counter(),
        "supported_causes": Counter(),
        "existing_error_classes": Counter(),
        "threshold_mismatch_links": Counter(),
        "links": Counter(),
    }


def _finalize_numeric(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, np.float64)
    if not len(array):
        return {
            "count": 0, "minimum": None, "p50": None, "p90": None,
            "p95": None, "p99": None, "maximum": None, "absolute_p95": None,
        }
    return {
        "count": int(len(array)),
        "minimum": float(np.min(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(np.max(array)),
        "absolute_p95": float(np.quantile(np.abs(array), 0.95)),
    }


def reduce_coverage_error_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_id: str,
    thresholds: Mapping[tuple[str, str], float],
    multi_origin: bool,
) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = defaultdict(_new_mismatch_bucket)
    causes = Counter({cause: 0 for cause in DIAGNOSTIC_CAUSE_IDS})
    primary = Counter({cause: 0 for cause in DIAGNOSTIC_CAUSE_IDS})
    total = 0
    for row in rows:
        if row.get("error_scope") == "DECISION_LEVEL":
            continue
        condition = str(row.get("condition_id"))
        mode = str(row.get("evidence_mode"))
        threshold_key = (condition, mode)
        if threshold_key not in thresholds:
            raise AuditError(f"coverage-error row lacks frozen threshold: {threshold_key}")
        oracle = bool(row.get("oracle_contact"))
        predicted = bool(row.get("predicted_contact"))
        if oracle == predicted:
            raise AuditError("contact-classification error row is not a mismatch")
        if multi_origin:
            supported = bool(row.get("supporting_origin_ids", row.get("supporting_origins", ())))
            nominal = bool(row.get("nominal_fov"))
            self_occluded = bool(row.get("robot_self_occlusion"))
            link = str(row.get("robot_link"))
            observed = _finite_number(
                row.get("observed_clearance_m", row.get("sensor_derived_clearance_m"))
            )
        else:
            supported = bool(row.get("scan_point_availability"))
            nominal = bool(row.get("nominal_fov_inclusion"))
            self_occluded = bool(row.get("self_occlusion"))
            link = str(row.get("robot_link_or_body_region", "")).split("/", 1)[0]
            observed = _finite_number(row.get("sensor_derived_clearance_m"))
        exact = _finite_number(row.get("exact_clearance_m"))
        classification = classify_diagnostic_causes(
            oracle_contact=oracle,
            predicted_contact=predicted,
            patch_supported=supported,
            nominal_fov=nominal,
            self_occluded=self_occluded,
            exact_clearance_m=exact,
            observed_clearance_m=observed,
            threshold_m=float(thresholds[threshold_key]),
        )
        bucket = buckets[f"{source_id}|{condition}|{mode}"]
        bucket["rows"] += 1
        bucket["false_negative_rows"] += int(oracle and not predicted)
        bucket["false_positive_rows"] += int(not oracle and predicted)
        bucket["patch_unobserved_rows"] += int(not supported)
        bucket["outside_nominal_fov_rows"] += int(not nominal)
        bucket["self_occluded_rows"] += int(self_occluded)
        bucket["links"][link] += 1
        bucket["existing_error_classes"][str(row.get("error_class", "UNRESOLVED"))] += 1
        bucket["primary_causes"][classification["primary_cause"]] += 1
        primary[classification["primary_cause"]] += 1
        for cause in classification["supported_cause_ids"]:
            bucket["supported_causes"][cause] += 1
            causes[cause] += 1
        if classification["flags"][GLOBAL_THRESHOLD_HETEROGENEITY]:
            bucket["threshold_mismatch_links"][link] += 1
        if supported and exact is not None and observed is not None:
            delta = observed - exact
            bucket["finite_supported_clearance_pairs"] += 1
            bucket["observed_minus_exact_m"].append(delta)
            bucket["point_primitive_mismatch_rows"] += int(
                supported and abs(delta) > POINT_PRIMITIVE_TOLERANCE_M
            )
        total += 1
    finalized: dict[str, Any] = {}
    for key, bucket in sorted(buckets.items()):
        deltas = bucket.pop("observed_minus_exact_m")
        bucket["observed_minus_exact_statistics_m"] = _finalize_numeric(deltas)
        bucket["primary_causes"] = dict(sorted(bucket["primary_causes"].items()))
        bucket["supported_causes"] = dict(sorted(bucket["supported_causes"].items()))
        bucket["existing_error_classes"] = dict(
            sorted(bucket["existing_error_classes"].items())
        )
        bucket["links"] = dict(sorted(bucket["links"].items()))
        bucket["threshold_mismatch_rows_by_link"] = dict(
            sorted(bucket.pop("threshold_mismatch_links").items())
        )
        bucket["threshold_heterogeneity_links"] = sorted(
            bucket["threshold_mismatch_rows_by_link"]
        )
        finalized[key] = bucket
    return {
        "source_id": source_id,
        "population": "persisted heldout contact-classification mismatch rows only",
        "rows": total,
        "condition_mode": finalized,
        "supported_cause_counts": dict(sorted(causes.items())),
        "primary_cause_counts": dict(sorted(primary.items())),
    }


def _jsonl_rows(path: Path, *, compressed: bool) -> Iterator[dict[str, Any]]:
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.endswith("\n"):
                raise AuditError(f"JSONL row lacks LF at {path}:{line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise AuditError(f"JSONL row is not an object at {path}:{line_number}")
            yield value


def _new_link_bucket() -> dict[str, Any]:
    return {
        "rows": 0,
        "contact_rows": 0,
        "supported_full_sweep_rows": 0,
        "unsupported_any_rows": 0,
        "unsupported_fraction_sum": 0.0,
        "contact_unsupported_any_rows": 0,
        "contact_unsupported_fraction_sum": 0.0,
        "contact_outside_nominal_fov_rows": 0,
        "contact_self_occluded_rows": 0,
        "finite_exact_observed_contact_pairs": 0,
        "contact_point_primitive_mismatch_rows": 0,
        "contact_observed_minus_exact_m": [],
    }


def reduce_per_link_rows(
    rows: Iterable[Mapping[str, Any]], *, source_id: str, multi_origin: bool,
) -> dict[str, Any]:
    condition_mode: dict[str, dict[str, Any]] = defaultdict(_new_link_bucket)
    per_link: dict[str, dict[str, Any]] = defaultdict(_new_link_bucket)
    per_region: dict[str, dict[str, Any]] = defaultdict(_new_link_bucket)
    count = 0

    def update(bucket: dict[str, Any], *, contact: bool, support: bool,
               unsupported_fraction: float, outside_fov: bool, self_occluded: bool,
               exact: float | None, observed: float | None) -> None:
        bucket["rows"] += 1
        bucket["contact_rows"] += int(contact)
        bucket["supported_full_sweep_rows"] += int(support)
        bucket["unsupported_any_rows"] += int(unsupported_fraction > 0.0)
        bucket["unsupported_fraction_sum"] += unsupported_fraction
        if contact:
            bucket["contact_unsupported_any_rows"] += int(unsupported_fraction > 0.0)
            bucket["contact_unsupported_fraction_sum"] += unsupported_fraction
            bucket["contact_outside_nominal_fov_rows"] += int(outside_fov)
            bucket["contact_self_occluded_rows"] += int(self_occluded)
            if support and exact is not None and observed is not None:
                delta = observed - exact
                bucket["finite_exact_observed_contact_pairs"] += 1
                bucket["contact_observed_minus_exact_m"].append(delta)
                bucket["contact_point_primitive_mismatch_rows"] += int(
                    abs(delta) > POINT_PRIMITIVE_TOLERANCE_M
                )

    for row in rows:
        condition = str(row.get("condition_id"))
        mode = str(row.get("evidence_mode"))
        role = str(row.get("role", "UNKNOWN"))
        link = str(row.get("link_name", row.get("protected_link", "")))
        if link not in PROTECTED_LINK_NAMES:
            raise AuditError(f"unknown protected link in per-link ledger: {link}")
        region = str(row.get("collision_region", LINK_REGION[link])).upper()
        raw_contact = row.get("oracle_contact_link")
        contact = bool(raw_contact is True or raw_contact == link)
        support = bool(row.get("observation_support"))
        unsupported_fraction = float(row.get("unsupported_swept_volume_fraction"))
        if not 0.0 <= unsupported_fraction <= 1.0:
            raise AuditError("unsupported swept-volume fraction outside [0,1]")
        if multi_origin:
            nominal_by_origin = row.get("nominal_fov_inclusion_by_origin", {})
            outside_fov = not any(bool(value) for value in nominal_by_origin.values())
            self_occluded = float(row.get("robot_self_occlusion_fraction", 0.0)) > 0.0
            exact = _finite_number(row.get("exact_oracle_minimum_clearance_m"))
        else:
            outside_fov = float(row.get("nominal_fov_inclusion", 0.0)) == 0.0
            # The predecessor per-link ledger does not persist an unambiguous
            # self-occlusion fraction.  Do not infer it from visibility alone.
            self_occluded = False
            exact = None
        observed = _finite_number(row.get("minimum_observed_environment_clearance_m"))
        values = dict(
            contact=contact, support=support,
            unsupported_fraction=unsupported_fraction,
            outside_fov=outside_fov, self_occluded=self_occluded,
            exact=exact, observed=observed,
        )
        update(condition_mode[f"{source_id}|{condition}|{mode}|{role}"], **values)
        update(per_link[f"{source_id}|{condition}|{mode}|{role}|{link}"], **values)
        update(per_region[f"{source_id}|{condition}|{mode}|{role}|{region}"], **values)
        count += 1

    def finalize(groups: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, bucket in sorted(groups.items()):
            deltas = bucket.pop("contact_observed_minus_exact_m")
            rows_n = int(bucket["rows"])
            contacts = int(bucket["contact_rows"])
            bucket["unsupported_fraction_mean"] = (
                bucket.pop("unsupported_fraction_sum") / rows_n if rows_n else None
            )
            bucket["contact_unsupported_fraction_mean"] = (
                bucket.pop("contact_unsupported_fraction_sum") / contacts
                if contacts else None
            )
            bucket["contact_observed_minus_exact_statistics_m"] = _finalize_numeric(deltas)
            result[key] = bucket
        return result

    return {
        "source_id": source_id,
        "rows": count,
        "condition_mode_role": finalize(condition_mode),
        "per_link": finalize(per_link),
        "per_region": finalize(per_region),
    }


def _merge_raw_bucket(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, list):
            target.setdefault(key, []).extend(value)
        elif isinstance(value, (int, float)):
            target[key] = target.get(key, 0) + value
        else:
            raise AuditError(f"unsupported raw bucket field: {key}")


def _new_event_bucket() -> dict[str, Any]:
    return {
        "contact_events": 0,
        "patch_supported_events": 0,
        "event_time_supported_events": 0,
        "nominal_fov_events": 0,
        "direct_visibility_events": 0,
        "self_occluded_events": 0,
        "patch_unobserved_self_occluded_events": 0,
        "finite_scan_inherited_events": 0,
        "point_primitive_mismatch_events": 0,
        "point_primitive_risk_underestimate_events": 0,
        "threshold_patch_miss_events": 0,
        "clearance_threshold_detected_events": 0,
        "deltas": [],
        "supported_observed_clearance_m": [],
        "patch_clearance_with_unsupported_inf": [],
    }


def _event_update(
    bucket: dict[str, Any], *, support: bool, event_support: bool,
    nominal: bool, direct: bool, self_occluded: bool, inherited: bool,
    exact: float, observed: float, threshold: float,
) -> None:
    bucket["contact_events"] += 1
    bucket["patch_supported_events"] += int(support)
    bucket["event_time_supported_events"] += int(event_support)
    bucket["nominal_fov_events"] += int(nominal)
    bucket["direct_visibility_events"] += int(direct)
    bucket["self_occluded_events"] += int(self_occluded)
    bucket["patch_unobserved_self_occluded_events"] += int(
        (not support) and self_occluded
    )
    bucket["finite_scan_inherited_events"] += int(inherited)
    bucket["patch_clearance_with_unsupported_inf"].append(
        observed if support and math.isfinite(observed) else math.inf
    )
    bucket["clearance_threshold_detected_events"] += int(
        support and math.isfinite(observed) and observed <= threshold
    )
    if support and math.isfinite(exact) and math.isfinite(observed):
        delta = observed - exact
        bucket["deltas"].append(delta)
        mismatch = abs(delta) > POINT_PRIMITIVE_TOLERANCE_M
        bucket["point_primitive_mismatch_events"] += int(mismatch)
        bucket["point_primitive_risk_underestimate_events"] += int(
            mismatch and delta > 0.0
        )
        bucket["threshold_patch_miss_events"] += int(observed > threshold)
        bucket["supported_observed_clearance_m"].append(observed)


def reduce_state_contact_events(task: Mapping[str, Any]) -> dict[str, Any]:
    """Worker reducer for one state.  It opens only frozen NPZ evidence."""

    body_path = Path(str(task["body_path"]))
    expected_body_sha = task.get("body_sha256")
    if task.get("verify_shard_hashes") and sha256_file(body_path) != expected_body_sha:
        raise AuditError(f"single-origin shard SHA-256 drift: {body_path}")
    role = str(task["role"])
    family = str(task["family"])
    shape_names = tuple(task["shape_names"])
    thresholds = {
        tuple(key.split("|", 1)): float(value)
        for key, value in task["thresholds"].items()
    }
    condition_mode: dict[str, dict[str, Any]] = defaultdict(_new_event_bucket)
    per_link: dict[str, dict[str, Any]] = defaultdict(_new_event_bucket)
    per_region: dict[str, dict[str, Any]] = defaultdict(_new_event_bucket)
    per_shape: dict[str, dict[str, Any]] = defaultdict(_new_event_bucket)
    condition_mode_by_role: dict[str, dict[str, Any]] = defaultdict(_new_event_bucket)
    per_link_by_role: dict[str, dict[str, Any]] = defaultdict(_new_event_bucket)
    per_region_by_role: dict[str, dict[str, Any]] = defaultdict(_new_event_bucket)
    per_shape_by_role: dict[str, dict[str, Any]] = defaultdict(_new_event_bucket)
    calibration_per_link: dict[str, dict[str, Any]] = defaultdict(_new_event_bucket)
    calibration_per_shape: dict[str, dict[str, Any]] = defaultdict(_new_event_bucket)
    exact_links = Counter()
    exact_shapes = Counter()
    with np.load(body_path, allow_pickle=False) as body:
        frozen = np.asarray(body["frozen_contact"], bool)
        step = np.asarray(body["oracle_contact_step"], np.int16)
        link = np.asarray(body["oracle_contact_link"], np.int16)
        target_geom = np.asarray(body["target_geom_index"], np.int16)
        oracle = np.asarray(body["oracle_clearance_m"], np.float64)
        valid = frozen & (step >= 0) & (link >= 0)
        transitions = np.flatnonzero(valid)
        unresolved = int(frozen.sum() - len(transitions))
        for transition in transitions:
            event_step = int(step[transition])
            event_link = int(link[transition])
            geom = int(target_geom[transition, event_step, event_link])
            if not 0 <= event_link < len(PROTECTED_LINK_NAMES):
                raise AuditError("oracle contact link outside protected inventory")
            if not 0 <= geom < len(shape_names):
                raise AuditError("oracle contact primitive outside protected inventory")
            exact_links[PROTECTED_LINK_NAMES[event_link]] += 1
            exact_shapes[f"{geom:02d}:{shape_names[geom]}"] += 1

        def consume(
            condition_id: str, mode_id: str, arrays: Mapping[str, np.ndarray],
            mode_index: int,
        ) -> None:
            threshold = thresholds[(condition_id, mode_id)]
            for transition in transitions:
                event_step = int(step[transition])
                event_link = int(link[transition])
                geom = int(target_geom[transition, event_step, event_link])
                observed = float(arrays["clearance_m"][transition, mode_index, event_step, event_link])
                exact = float(oracle[transition, event_step, event_link])
                values = {
                    "support": bool(arrays["support"][transition, mode_index, event_step, event_link]),
                    "event_support": bool(arrays["event_time_support"][transition, mode_index, event_step, event_link]),
                    "nominal": bool(arrays["nominal_fov"][transition, mode_index, event_step, event_link]),
                    "direct": bool(arrays["direct_visibility"][transition, mode_index, event_step, event_link]),
                    "self_occluded": bool(arrays["self_occluded"][transition, mode_index, event_step, event_link]),
                    "inherited": bool(arrays["finite_scan_support_inherited"][transition, mode_index, event_step, event_link]),
                    "exact": exact,
                    "observed": observed,
                    "threshold": threshold,
                }
                link_name = PROTECTED_LINK_NAMES[event_link]
                region = LINK_REGION[link_name]
                shape_name = f"{geom:02d}:{shape_names[geom]}"
                for bucket in (
                    condition_mode[f"{condition_id}|{mode_id}"],
                    per_link[f"{condition_id}|{mode_id}|{link_name}"],
                    per_region[f"{condition_id}|{mode_id}|{region}"],
                    per_shape[f"{condition_id}|{mode_id}|{shape_name}"],
                    condition_mode_by_role[f"{condition_id}|{mode_id}|{role}"],
                    per_link_by_role[f"{condition_id}|{mode_id}|{role}|{link_name}"],
                    per_region_by_role[f"{condition_id}|{mode_id}|{role}|{region}"],
                    per_shape_by_role[f"{condition_id}|{mode_id}|{role}|{shape_name}"],
                ):
                    _event_update(bucket, **values)
                if role == "calibration":
                    link_minimum = float(
                        np.min(
                            arrays["clearance_m"][
                                transition, mode_index, :, event_link
                            ]
                        )
                    )
                    finite_link_minimum = math.isfinite(link_minimum)
                    calibration_values = {
                        **values,
                        "support": finite_link_minimum,
                        "event_support": finite_link_minimum,
                        "observed": link_minimum,
                    }
                    _event_update(
                        calibration_per_link[
                            f"{condition_id}|{mode_id}|{link_name}"
                        ],
                        **calibration_values,
                    )
                    _event_update(
                        calibration_per_shape[
                            f"{condition_id}|{mode_id}|{shape_name}"
                        ],
                        **calibration_values,
                    )

        single_arrays = {
            key: np.asarray(body[key])
            for key in (
                "clearance_m", "support", "event_time_support", "nominal_fov",
                "direct_visibility", "self_occluded", "finite_scan_support_inherited",
            )
        }
        for condition_index, condition_id in enumerate(SINGLE_CONDITION_IDS):
            sliced = {key: value[:, condition_index] for key, value in single_arrays.items()}
            for mode_index, mode_id in enumerate(EVIDENCE_MODE_IDS):
                consume(condition_id, mode_id, sliced, mode_index)

    for condition_id, receipt in task["multi_shards"].items():
        path = Path(str(receipt["path"]))
        if task.get("verify_shard_hashes") and sha256_file(path) != receipt.get("sha256"):
            raise AuditError(f"multi-origin shard SHA-256 drift: {path}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {
                key: np.asarray(archive[key])
                for key in (
                    "clearance_m", "support", "event_time_support", "nominal_fov",
                    "direct_visibility", "self_occluded", "finite_scan_support_inherited",
                )
            }
            modes = (
                (1, "TRUE_FUTURE_OBSERVABILITY_CLOUD"),
            ) if condition_id == DIAGNOSTIC_CONDITION_ID else tuple(
                enumerate(EVIDENCE_MODE_IDS)
            )
            for mode_index, mode_id in modes:
                consume(condition_id, mode_id, arrays, int(mode_index))
    return {
        "state_id": task["state_id"],
        "frozen_contact_events": int(frozen.sum()),
        "resolved_contact_events": int(len(transitions)),
        "unresolved_contact_events": unresolved,
        "exact_links": dict(exact_links),
        "exact_shapes": dict(exact_shapes),
        "condition_mode": dict(condition_mode),
        "per_link": dict(per_link),
        "per_region": dict(per_region),
        "per_shape": dict(per_shape),
        "condition_mode_by_role": dict(condition_mode_by_role),
        "per_link_by_role": dict(per_link_by_role),
        "per_region_by_role": dict(per_region_by_role),
        "per_shape_by_role": dict(per_shape_by_role),
        "calibration_per_link": dict(calibration_per_link),
        "calibration_per_shape": dict(calibration_per_shape),
    }


def _shape_names(stage_a_contract: Mapping[str, Any]) -> tuple[str, ...]:
    components = stage_a_contract["component_inventory"]["components"]
    return tuple(
        f"{row['component_id']}:{row['primitive']}" for row in components
    )


def build_state_tasks(closure: Mapping[str, Any], *, verify_shard_hashes: bool) -> list[dict[str, Any]]:
    single_records = {row["state_id"]: row for row in closure["single_index"]["records"]}
    multi_records: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for index_key in ("dual_index", "three_index", "diagnostic_index"):
        for row in closure[index_key]["records"]:
            state_id = str(row["state_id"])
            condition_id = str(row["condition_id"])
            if condition_id in multi_records[state_id]:
                raise AuditError("duplicate multi-origin state/condition shard")
            multi_records[state_id][condition_id] = {
                "path": row["shard_path"],
                "sha256": row["shard_sha256"],
            }
    if set(single_records) != set(multi_records) or len(single_records) != 176:
        raise AuditError("single/multi state identity closure drift")
    threshold_map = {
        f"{condition}|{mode}": value
        for mapping in (closure["single_thresholds"], closure["multi_thresholds"])
        for (condition, mode), value in mapping.items()
    }
    component_names = tuple(closure["stage_a_component_names"])
    tasks = []
    for state_id in sorted(single_records):
        row = single_records[state_id]
        if set(multi_records[state_id]) != set(MULTI_CONDITION_IDS):
            raise AuditError(f"multi-origin condition closure drift for {state_id}")
        tasks.append({
            "state_id": state_id,
            "role": row["role"],
            "family": row["family"],
            "body_path": row["shard_path"],
            "body_sha256": row["shard_sha256"],
            "multi_shards": multi_records[state_id],
            "shape_names": component_names,
            "thresholds": threshold_map,
            "verify_shard_hashes": verify_shard_hashes,
        })
    return tasks


def reduce_contact_event_tasks(
    tasks: Sequence[Mapping[str, Any]], *, workers: int,
) -> dict[str, Any]:
    if workers < 1:
        raise AuditError("workers must be positive")
    results: Iterable[dict[str, Any]]
    if workers == 1:
        results = map(reduce_state_contact_events, tasks)
    else:
        context = mp.get_context("fork")
        pool = context.Pool(processes=workers)
        results = pool.imap_unordered(reduce_state_contact_events, tasks, chunksize=1)
    totals = Counter()
    links = Counter()
    shapes = Counter()
    groups: dict[str, dict[str, dict[str, Any]]] = {
        "condition_mode": {}, "per_link": {}, "per_region": {}, "per_shape": {},
        "condition_mode_by_role": {}, "per_link_by_role": {},
        "per_region_by_role": {}, "per_shape_by_role": {},
        "calibration_per_link": {}, "calibration_per_shape": {},
    }
    try:
        for result in results:
            totals["states"] += 1
            for key in ("frozen_contact_events", "resolved_contact_events", "unresolved_contact_events"):
                totals[key] += int(result[key])
            links.update(result["exact_links"])
            shapes.update(result["exact_shapes"])
            for group_name in groups:
                target_group = groups[group_name]
                for key, source_bucket in result[group_name].items():
                    target = target_group.setdefault(key, _new_event_bucket())
                    _merge_raw_bucket(target, source_bucket)
    finally:
        if workers != 1:
            pool.close()
            pool.join()
    finalized_groups: dict[str, Any] = {}
    for group_name, values in groups.items():
        final: dict[str, Any] = {}
        for key, bucket in sorted(values.items()):
            deltas = bucket.pop("deltas")
            supported_observed = bucket.pop("supported_observed_clearance_m")
            patch_clearance = bucket.pop("patch_clearance_with_unsupported_inf")
            count = int(bucket["contact_events"])
            bucket["patch_unobserved_events"] = count - int(bucket["patch_supported_events"])
            bucket["patch_support_rate"] = (
                bucket["patch_supported_events"] / count if count else None
            )
            bucket["event_time_support_rate"] = (
                bucket["event_time_supported_events"] / count if count else None
            )
            bucket["observed_minus_exact_statistics_m"] = _finalize_numeric(deltas)
            bucket["supported_observed_clearance_statistics_m"] = _finalize_numeric(
                supported_observed
            )
            if patch_clearance:
                ordered = sorted(float(value) for value in patch_clearance)
                rank = max(0, math.ceil(0.95 * len(ordered)) - 1)
                q95 = ordered[rank]
                bucket["nearest_rank_q95_including_unsupported"] = (
                    "POSITIVE_INFINITY" if math.isinf(q95) else q95
                )
                bucket["nearest_rank_q95_rank_zero_based"] = rank
            else:
                bucket["nearest_rank_q95_including_unsupported"] = None
                bucket["nearest_rank_q95_rank_zero_based"] = None
            final[key] = bucket
        finalized_groups[group_name] = final
    return {
        "population": "all frozen transitions with resolved exact contact step/link",
        **dict(totals),
        "exact_contact_events_by_link": dict(sorted(links.items())),
        "exact_contact_events_by_collision_component": dict(sorted(shapes.items())),
        **finalized_groups,
    }


def _compact_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(binding["path"]),
        "sha256": str(binding["sha256"]),
        "bytes": int(binding["bytes"]),
        "rows": int(binding["rows"]),
    }


def _compact_index_binding(
    *, path: Path, sha256: str, index: Mapping[str, Any],
    condition_ids: Sequence[str], mode_ids: Sequence[str],
) -> dict[str, Any]:
    frozen_conditions = tuple(str(value) for value in condition_ids)
    frozen_modes = tuple(str(value) for value in mode_ids)
    if not frozen_conditions or not frozen_modes:
        raise AuditError("compact materialization axes must be nonempty")

    # The predecessor indices use two representations: the single-origin
    # index stores axis cardinalities while the multi-origin indices store
    # condition identities and omit the mode axis.  Never infer identities by
    # iterating these fields: an integer is a count, not an identifier list.
    for key, expected in (
        ("conditions", frozen_conditions),
        ("evidence_modes", frozen_modes),
    ):
        observed = index.get(key)
        if isinstance(observed, bool):
            raise AuditError(f"materialization index {key} has boolean cardinality")
        if isinstance(observed, int):
            if observed != len(expected):
                raise AuditError(f"materialization index {key} count drift")
        elif observed is not None:
            if not isinstance(observed, Sequence) or isinstance(
                observed, (str, bytes, bytearray)
            ):
                raise AuditError(f"materialization index {key} has invalid type")
            if tuple(str(value) for value in observed) != expected:
                raise AuditError(f"materialization index {key} identity/order drift")

    return {
        "path": str(path),
        "sha256": sha256,
        "content_digest": str(index["content_digest"]),
        "states": int(index["states"]),
        "transitions": int(index["transitions"]),
        "records": len(index["records"]),
        "condition_ids": list(frozen_conditions),
        "mode_ids": list(frozen_modes),
    }


def build_calibration_threshold_shift_diagnostic(
    contact_events: Mapping[str, Any],
    thresholds: Mapping[tuple[str, str], float],
) -> dict[str, Any]:
    """Describe calibration-role patch thresholds without selecting a new one."""

    def convert(groups: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, bucket in sorted(groups.items()):
            condition, mode, _identity = key.split("|", 2)
            frozen = float(thresholds[(condition, mode)])
            positives = int(bucket["contact_events"])
            support = int(bucket["patch_supported_events"])
            detected = int(bucket["clearance_threshold_detected_events"])
            q95 = bucket["nearest_rank_q95_including_unsupported"]
            output[key] = {
                "role": "calibration",
                "positives": positives,
                "finite_attributable_link_minimum_scores": support,
                "unsupported_attributable_link_minimum_scores": positives - support,
                "frozen_global_threshold_m": frozen,
                "global_threshold_attributable_link_clearance_recall": (
                    detected / positives if positives else None
                ),
                "nearest_rank_q95_definition":
                    "sorted(attributable_link_minimum_clearance_with_unsupported_as_positive_infinity)[ceil(0.95*n)-1]",
                "nearest_rank_q95_m": q95,
                "nearest_rank_q95_rank_zero_based":
                    bucket["nearest_rank_q95_rank_zero_based"],
                "q95_minus_frozen_global_threshold_m": (
                    float(q95) - frozen if isinstance(q95, (int, float)) else None
                ),
                "threshold_shift_status": (
                    "UNBOUNDED_BY_UNSUPPORTED_CALIBRATION_PATCHES"
                    if q95 == "POSITIVE_INFINITY" else "FINITE_DESCRIPTIVE_SHIFT"
                ),
                "threshold_reselected": False,
            }
        return output

    per_link = convert(contact_events["calibration_per_link"])
    per_shape = convert(contact_events["calibration_per_shape"])
    return {
        "population": "frozen internal-calibration role only",
        "selection_authority": False,
        "heldout_used": False,
        "interpolation_used": False,
        "per_link": per_link,
        "per_collision_component": per_shape,
    }


def summarize_diagnostic_cause_evidence(
    mismatch_diagnostics: Sequence[Mapping[str, Any]],
    *, unresolved_exact_contact_attribution_events: int,
) -> dict[str, Any]:
    """Combine explained mismatch rows with the conservative residual population."""

    if unresolved_exact_contact_attribution_events < 0:
        raise AuditError("unresolved exact-contact attribution count is negative")
    mismatch_counts = Counter({cause: 0 for cause in DIAGNOSTIC_CAUSE_IDS})
    for diagnostic in mismatch_diagnostics:
        observed = diagnostic.get("supported_cause_counts", {})
        if not set(observed).issubset(DIAGNOSTIC_CAUSE_IDS):
            raise AuditError("diagnostic cause evidence contains an unknown ID")
        mismatch_counts.update({cause: int(value) for cause, value in observed.items()})
    combined = Counter(mismatch_counts)
    combined[UNRESOLVED_GEOMETRIC_MISMATCH] += int(
        unresolved_exact_contact_attribution_events
    )
    return {
        "supported_diagnostic_cause_ids": [
            cause for cause in DIAGNOSTIC_CAUSE_IDS if combined[cause] > 0
        ],
        "diagnostic_cause_evidence_counts": {
            cause: int(combined[cause]) for cause in DIAGNOSTIC_CAUSE_IDS
        },
        "diagnostic_cause_evidence_detail": {
            "contact_classification_mismatch_rows": {
                cause: int(mismatch_counts[cause]) for cause in DIAGNOSTIC_CAUSE_IDS
            },
            "unresolved_exact_contact_attribution_events": int(
                unresolved_exact_contact_attribution_events
            ),
            "unresolved_overlap_policy": (
                "UNRESOLVED_GEOMETRIC_MISMATCH is supported only by frozen-contact "
                "events lacking an authoritative contact step/link; it does not overwrite "
                "rows quantitatively attributed to another diagnostic cause"
            ),
        },
    }


def build_audit_result(
    paths: AuditPaths = default_paths(), *, workers: int = DEFAULT_WORKERS,
    verify_hashes: bool = True, reduce_per_link_ledgers: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    if workers != CANONICAL_WORKERS:
        raise AuditError(
            f"canonical execution requires exactly {CANONICAL_WORKERS} workers"
        )
    closure = validate_input_closure(paths, verify_hashes=verify_hashes)
    stage_contract = _load_json(paths.stage_a_contract)
    closure["stage_a_component_names"] = _shape_names(stage_contract)
    tasks = build_state_tasks(closure, verify_shard_hashes=verify_hashes)
    contact_events = reduce_contact_event_tasks(tasks, workers=workers)
    if len(contact_events["condition_mode"]) != 21:
        raise AuditError("exact contact-event condition/mode cardinality must be 21")
    mismatch_single = reduce_coverage_error_rows(
        _jsonl_rows(closure["paths"]["single_errors"], compressed=False),
        source_id="SINGLE_ORIGIN",
        thresholds=closure["single_thresholds"],
        multi_origin=False,
    )
    mismatch_multi = reduce_coverage_error_rows(
        _jsonl_rows(closure["paths"]["multi_errors"], compressed=True),
        source_id="MULTI_ORIGIN",
        thresholds=closure["multi_thresholds"],
        multi_origin=True,
    )
    if mismatch_single["rows"] != int(
        closure["single_result"]["coverage_error_receipt"]["rows"]
    ):
        raise AuditError("single-origin coverage-error row cardinality drift")
    expected_multi_contact_errors = int(
        closure["multi_result"]["coverage_errors"]["scope_counts"][
            "CONTACT_CLASSIFICATION"
        ]
    )
    if mismatch_multi["rows"] != expected_multi_contact_errors:
        raise AuditError("multi-origin contact-classification error cardinality drift")
    per_link: dict[str, Any] | None = None
    if reduce_per_link_ledgers:
        per_link = {
            "single_origin": reduce_per_link_rows(
                _jsonl_rows(closure["paths"]["single_per_link"], compressed=False),
                source_id="SINGLE_ORIGIN", multi_origin=False,
            ),
            "multi_origin": reduce_per_link_rows(
                _jsonl_rows(closure["paths"]["multi_per_link"], compressed=True),
                source_id="MULTI_ORIGIN", multi_origin=True,
            ),
        }
        if per_link["single_origin"]["rows"] != int(
            closure["single_result"]["row_level_evidence_receipt"]
            ["per_link_evidence"]["rows"]
        ):
            raise AuditError("single-origin per-link ledger row cardinality drift")
        if per_link["multi_origin"]["rows"] != int(
            closure["multi_result"]["row_level_evidence"]
            ["per_link_evidence"]["rows"]
        ):
            raise AuditError("multi-origin per-link ledger row cardinality drift")
    single_result = closure["single_result"]
    multi_result = closure["multi_result"]
    thresholds = {**closure["single_thresholds"], **closure["multi_thresholds"]}
    calibration_shift = build_calibration_threshold_shift_diagnostic(
        contact_events, thresholds
    )
    unresolved_attribution_events = int(contact_events["unresolved_contact_events"])
    cause_evidence = summarize_diagnostic_cause_evidence(
        (mismatch_single, mismatch_multi),
        unresolved_exact_contact_attribution_events=unresolved_attribution_events,
    )
    components = [
        {
            "geom_index": int(row["geom_index"]),
            "link_index": int(row["link_index"]),
            "link_name": str(row["link_name"]),
            "component_id": str(row["component_id"]),
            "primitive": str(row["primitive"]),
        }
        for row in stage_contract["component_inventory"]["components"]
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "development_only": True,
        "claim_bearing": False,
        "diagnostic_cause_ids": list(DIAGNOSTIC_CAUSE_IDS),
        **cause_evidence,
        "point_primitive_tolerance_m": POINT_PRIMITIVE_TOLERANCE_M,
        "workers": workers,
        "runtime_s": None,
        "stage_a_barrier": closure["stage_a"],
        "input_closure": {
            "exact_upper_bound": {
                "result_path": str(paths.exact_result),
                "result_sha256": EXACT_RESULT_SHA256,
                "result_content_digest": EXACT_RESULT_CONTENT_DIGEST,
                "source_commit": str(closure["exact_result"]["source_commit"]),
                "geometry_index_path": str(paths.exact_geometry_index),
                "geometry_index_sha256": EXACT_GEOMETRY_INDEX_SHA256,
                "geometry_index_content_digest": (
                    EXACT_GEOMETRY_INDEX_CONTENT_DIGEST
                ),
                "states": int(closure["exact_index"]["states"]),
                "transitions": int(closure["exact_index"]["transitions"]),
            },
            "single_result_file_sha256": SINGLE_RESULT_SHA256,
            "multi_result_file_sha256": MULTI_RESULT_SHA256,
            "single_per_link": _compact_binding(
                single_result["row_level_evidence_receipt"]["per_link_evidence"]
            ),
            "single_coverage_errors": _compact_binding(
                single_result["coverage_error_receipt"]
            ),
            "multi_per_link": _compact_binding(
                multi_result["row_level_evidence"]["per_link_evidence"]
            ),
            "multi_coverage_errors": _compact_binding(
                multi_result["coverage_errors"]
            ),
            "materialization_indices": {
                "single": _compact_index_binding(
                    path=paths.single_materialization_index,
                    sha256=SINGLE_MATERIALIZATION_INDEX_SHA256,
                    index=closure["single_index"],
                    condition_ids=SINGLE_CONDITION_IDS,
                    mode_ids=EVIDENCE_MODE_IDS,
                ),
                "dual": _compact_index_binding(
                    path=paths.multi_dual_index,
                    sha256=MULTI_INDEX_SHA256["dual"],
                    index=closure["dual_index"],
                    condition_ids=tuple(closure["dual_index"]["conditions"]),
                    mode_ids=EVIDENCE_MODE_IDS,
                ),
                "three": _compact_index_binding(
                    path=paths.multi_three_index,
                    sha256=MULTI_INDEX_SHA256["three"],
                    index=closure["three_index"],
                    condition_ids=tuple(closure["three_index"]["conditions"]),
                    mode_ids=EVIDENCE_MODE_IDS,
                ),
                "diagnostic": _compact_index_binding(
                    path=paths.multi_diagnostic_index,
                    sha256=MULTI_INDEX_SHA256["diagnostic"],
                    index=closure["diagnostic_index"],
                    condition_ids=tuple(closure["diagnostic_index"]["conditions"]),
                    mode_ids=("TRUE_FUTURE_OBSERVABILITY_CLOUD",),
                ),
            },
            "single_source_freeze_commit": single_result["source_freeze_commit"],
            "single_source_lineage": single_result["source_lineage"],
            "single_predecessor_result_commit": single_result["predecessor_result_commit"],
            "single_contract_content_digest": single_result["contract_content_digest"],
            "multi_source_freeze_commit": multi_result["source_freeze_commit"],
            "multi_predecessor_result_commit": multi_result["predecessor_result_commit"],
            "multi_contract_content_digest": multi_result["contract_content_digest"],
            "multi_result_content_digest": multi_result["result_content_digest"],
            "corpus_action_ledger_bindings": multi_result["corpus_bindings"],
        },
        "collision_component_inventory": components,
        "exact_contact_event_diagnostic": contact_events,
        "calibration_only_threshold_shift_diagnostic": calibration_shift,
        "heldout_mismatch_diagnostic": {
            "single_origin": mismatch_single,
            "multi_origin": mismatch_multi,
        },
        "per_link_ledger_diagnostic": per_link,
        "existing_coverage_error_counts": {
            "single_origin": single_result["coverage_error_counts"],
            "multi_origin": multi_result["coverage_error_counts"],
            "multi_origin_scope_counts": multi_result["coverage_errors"]["scope_counts"],
            "multi_origin_decision_failure_counts":
                multi_result["coverage_errors"]["decision_failure_counts"],
        },
        "limitations": [
            "development-only post-freeze diagnostic; not claim-bearing",
            "actual current and successor trajectories are true-future upper-bound evidence",
            "range support is the predecessor's frozen 0.10 m same-object witness definition",
            "dense support copies exact primitive clearance when the exact critical patch is visible",
            "finite scans use nearest point-to-primitive clearance and can differ from exact patch clearance",
            "a global-threshold mismatch is descriptive and does not identify a valid per-link threshold",
            "simulated contact and analytical clearance conventions are not physical severity measures",
            "no diagnostic cause defines, narrows, or removes a protected-contact requirement",
        ],
        "prohibited_action_counters": {
            "raycasts": 0,
            "simulator_steps": 0,
            "contact_label_changes": 0,
            "protected_links_removed": 0,
            "protected_collision_components_removed": 0,
            "model_training_steps": 0,
            "fresh_panel_rows": 0,
            "jepa_or_g2_opens": 0,
            "memory_navigation_routing_or_beacon_executions": 0,
        },
        "interpretation_boundary": {
            "contact_labels_changed": False,
            "protected_links_or_shapes_removed": False,
            "scope_narrowing_authorized": False,
            "stage_b_authorized": False,
            "raycasting_rerun": False,
            "model_training": False,
            "fresh_panel": False,
            "jepa_g2_memory_navigation_or_routing": False,
            "causes_are_descriptive_not_requirements_authority": True,
        },
    }
    result["runtime_s"] = time.perf_counter() - started
    return attach_content_digest(result)


def validate_audit_result(result: Mapping[str, Any]) -> None:
    validate_content_digest(result)
    if result.get("schema_version") != SCHEMA_VERSION:
        raise AuditError("audit result schema drift")
    if tuple(result.get("diagnostic_cause_ids", ())) != DIAGNOSTIC_CAUSE_IDS:
        raise AuditError("diagnostic cause vocabulary drift")
    evidence_counts = result.get("diagnostic_cause_evidence_counts", {})
    if set(evidence_counts) != set(DIAGNOSTIC_CAUSE_IDS):
        raise AuditError("diagnostic cause evidence-count vocabulary drift")
    expected_supported = [
        cause for cause in DIAGNOSTIC_CAUSE_IDS
        if int(evidence_counts[cause]) > 0
    ]
    if result.get("supported_diagnostic_cause_ids") != expected_supported:
        raise AuditError("supported diagnostic causes do not follow actual evidence")
    runtime = _finite_number(result.get("runtime_s"))
    if runtime is None or runtime < 0.0:
        raise AuditError("audit runtime missing")
    if result.get("workers") != CANONICAL_WORKERS:
        raise AuditError("canonical audit did not use 32 workers")
    contact = result.get("exact_contact_event_diagnostic", {})
    if len(contact.get("condition_mode", {})) != 21:
        raise AuditError("audit must contain exactly 21 condition/mode summaries")
    roles = {
        key.split("|")[2]
        for key in contact.get("condition_mode_by_role", {})
    }
    if not {"training", "calibration", "heldout"}.issubset(roles):
        raise AuditError("contact-event diagnostic role separation incomplete")
    barrier = result.get("stage_a_barrier", {})
    if (
        barrier.get("primary_classification") != STAGE_A_PRIMARY
        or barrier.get("stage_b_authorized") is not False
        or barrier.get("scope_narrowing_authorized") is not False
    ):
        raise AuditError("audit result breached Stage-A barrier")
    boundary = result.get("interpretation_boundary", {})
    required_false = (
        "contact_labels_changed", "protected_links_or_shapes_removed",
        "scope_narrowing_authorized", "stage_b_authorized", "raycasting_rerun",
        "model_training", "fresh_panel", "jepa_g2_memory_navigation_or_routing",
    )
    if any(boundary.get(key) is not False for key in required_false):
        raise AuditError("audit result authorized a prohibited action")
    if boundary.get("causes_are_descriptive_not_requirements_authority") is not True:
        raise AuditError("diagnostic causes became requirements authority")
    counters = result.get("prohibited_action_counters", {})
    if not counters or any(int(value) != 0 for value in counters.values()):
        raise AuditError("prohibited-action counter is nonzero")


def write_canonical_result(
    result: Mapping[str, Any], output_path: Path, *, repo_root: Path = REPO_ROOT,
) -> None:
    validate_audit_result(result)
    output = output_path.resolve()
    repository = repo_root.resolve()
    expected = (repository / CANONICAL_OUTPUT_RELATIVE_PATH).resolve()
    if output != expected:
        raise AuditError(
            f"diagnostic output must be exactly {CANONICAL_OUTPUT_RELATIVE_PATH}"
        )
    protected_stage_a = {
        (repository / relative).resolve()
        for relative in (
            "docs/lewm_protected_contact_scope_contract_v1.json",
            "docs/lewm_protected_contact_scope_assumptions_unresolved_v1.json",
            "docs/lewm_protected_contact_scope_link_object_context_matrix_v1.json",
            "docs/lewm_protected_contact_scope_requirement_traceability_v1.json",
            "docs/lewm_protected_contact_scope_source_closure_v1.json",
        )
    }
    if output in protected_stage_a:
        raise AuditError("diagnostic output may not overwrite a Stage-A artifact")
    if output.exists():
        raise AuditError("diagnostic output path must not already exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise AuditError("diagnostic temporary path already exists")
    payload = canonical_json_bytes(result) + b"\n"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true", help="run the complete read-only reducer")
    mode.add_argument("--check", action="store_true", help="validate the persisted canonical receipt")
    parser.add_argument("--verify-hashes", action="store_true", help="hash every bound input")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    paths = default_paths()
    expected_output = (paths.repo_root / CANONICAL_OUTPUT_RELATIVE_PATH).resolve()
    if arguments.output.resolve() != expected_output:
        raise AuditError(f"--output must be {CANONICAL_OUTPUT_RELATIVE_PATH}")
    if arguments.check:
        persisted = _load_json(arguments.output)
        validate_audit_result(persisted)
        validate_input_closure(
            paths, verify_hashes=bool(arguments.verify_hashes)
        )
        print("PASS")
        return 0
    result = build_audit_result(
        paths,
        workers=int(arguments.workers),
        verify_hashes=bool(arguments.verify_hashes),
        reduce_per_link_ledgers=True,
    )
    validate_audit_result(result)
    write_canonical_result(result, arguments.output, repo_root=paths.repo_root)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(arguments.output.resolve()),
                "content_digest": result["content_digest"],
                "runtime_s": result["runtime_s"],
                "workers": result["workers"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
