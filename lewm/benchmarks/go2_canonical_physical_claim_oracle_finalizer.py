"""Independent finalization of the canonical 24x4 oracle claim regression.

The caller supplies already-loaded development manifests and candidate evidence.
This module performs no file IO and deliberately does not import the oracle,
eligibility audit, runner, controller, or trace-construction modules.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from lewm.benchmarks.go2_physical_claim_canonical import (
    canonical_content_sha256_valid as _stored_content_hash_valid,
    canonical_json_equal as _canonical_equal,
)
from lewm.benchmarks.go2_physical_claim_evaluator import (
    evaluate_physical_claim_trace,
)
from lewm_worlds.manifest import SceneManifest, manifest_sha256


CANDIDATE_SCHEMA = "lewm_go2_canonical_physical_claim_oracle_regression_v1"
FINALIZED_SCHEMA = (
    "lewm_go2_canonical_physical_claim_oracle_regression_finalized_v1"
)
ORACLE_REPORT_SCHEMA = "go2_oracle_coverage_positive_control_v1"
ELIGIBILITY_REPORT_SCHEMA = "lewm_go2_physical_scene_eligibility_v1"
ORACLE_ROUTE_SOURCE = "OnlineBeliefMap.shortest_path"
ORACLE_EVENT_ID_DOMAIN = "lewm-go2-oracle-claim-attempt-v1"
EXPECTED_SCENE_COUNT = 24
EXPECTED_TASKS_PER_SCENE = 4
EXPECTED_TASK_PAIR_COUNT = EXPECTED_SCENE_COUNT * EXPECTED_TASKS_PER_SCENE

ZERO_EVALUATOR_ACCESS_LEDGER = {
    "evaluator_output_reads_by_controller": 0,
    "evaluator_callbacks_into_controller": 0,
    "evaluator_derived_termination_signals": 0,
}
_HASH_READ_LEDGER = {
    "development_manifest_hash_reads": 1,
    "materialization_hash_reads": 1,
    "geometry_hash_reads": 1,
    "primitive_registry_hash_reads": 1,
    "directional_policy_hash_reads": 1,
    "prior_comparator_hash_reads": 0,
}
ZERO_FORBIDDEN_INPUT_ACCESS_LEDGER = {
    "preflight_hash_reads": _HASH_READ_LEDGER,
    "post_execution_hash_reads": _HASH_READ_LEDGER,
    "development_manifest_parse_calls": 1,
    "development_scene_manifest_parse_calls_by_parent": 24,
    "geometry_load_calls": 1,
    "primitive_registry_load_calls": 1,
    "directional_policy_load_calls": 1,
    "worker_runtime_input_file_opens": 0,
    "prior_comparator_payload_opens": 0,
    "heldout_payload_opens": 0,
    "sealed_payload_opens": 0,
    "g2_payload_opens": 0,
    "label_payload_opens": 0,
    "image_payload_opens": 0,
    "model_output_opens": 0,
}

_CANDIDATE_KEYS = frozenset(
    {
        "schema",
        "binding_sha256",
        "implementation_manifest_sha256",
        "source_map",
        "input_bindings",
        "command",
        "evaluator_access_ledger",
        "input_access_ledger",
        "oracle_report",
        "physical_eligibility_reports",
    }
)
_RAW_TRACE_KEYS = (
    "trace_id",
    "episode_id",
    "scene_id",
    "physical_manifest_sha256",
    "task_object_ids",
    "task_object_set_sha256",
    "controller_claim_attempts",
    "evaluator_feedback_to_controller",
)
_TRUE_FACTORS = {
    "identity_passes": True,
    "distance_passes": True,
    "line_of_sight_passes": True,
    "bearing_passes": True,
}


@dataclass(frozen=True)
class CanonicalPhysicalClaimOracleFinalization:
    """Fail-closed result; only a passing result has a publishable payload."""

    passed: bool
    errors: tuple[str, ...]
    finalized_payload: Mapping[str, Any] | None


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _plain_json(value: object) -> object:
    return json.loads(_canonical_bytes(value).decode("utf-8"))


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _is_exact_int(value: object, expected: int | None = None) -> bool:
    return type(value) is int and (expected is None or value == expected)


def _is_nonempty_string(value: object) -> bool:
    return type(value) is str and bool(value)


def _sorted_utf8(values: Sequence[str]) -> list[str]:
    return sorted(values, key=lambda value: value.encode("utf-8"))


def _freeze_or_error(
    value: object,
    *,
    label: str,
    errors: list[str],
) -> object | None:
    try:
        return _plain_json(value)
    except (OverflowError, TypeError, ValueError):
        errors.append(f"{label}_not_canonical_json")
        return None


def _manifest_task_binding(
    manifest: SceneManifest,
) -> tuple[list[str], str, str]:
    task_ids = _sorted_utf8([landmark.object_id for landmark in manifest.landmarks])
    manifest_hash = manifest_sha256(manifest)
    task_hash = _content_sha256(
        {
            "schema": "lewm_go2_claim_task_set_v1",
            "scene_id": manifest.scene_id,
            "physical_manifest_sha256": manifest_hash,
            "task_object_ids": task_ids,
        }
    )
    return task_ids, manifest_hash, task_hash


def _raw_trace_from_stored(stored: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "lewm_go2_claim_trace_v1",
        **{key: stored.get(key) for key in _RAW_TRACE_KEYS},
    }


def _recompute_trace(
    stored: object,
    *,
    manifest: SceneManifest,
    task_ids: Sequence[str],
    task_hash: str,
    label: str,
    errors: list[str],
) -> Mapping[str, Any] | None:
    if not isinstance(stored, Mapping):
        errors.append(f"{label}:canonical_trace_missing")
        return None
    try:
        recomputed = evaluate_physical_claim_trace(
            _raw_trace_from_stored(stored),
            manifest,
            task_ids,
            task_hash,
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"{label}:canonical_trace_invalid:{type(exc).__name__}")
        return None
    if not _canonical_equal(recomputed, stored):
        errors.append(f"{label}:canonical_trace_recomputation_mismatch")
    if not _stored_content_hash_valid(stored, hash_field="trace_content_sha256"):
        errors.append(f"{label}:trace_content_sha256_invalid")
    stored_events = stored.get("physical_claim_evaluations")
    if not isinstance(stored_events, list) or any(
        not _stored_content_hash_valid(event, hash_field="content_sha256")
        for event in stored_events
    ):
        errors.append(f"{label}:event_content_sha256_invalid")
    if not _stored_content_hash_valid(
        stored.get("physical_claim_summary"), hash_field="content_sha256"
    ):
        errors.append(f"{label}:summary_content_sha256_invalid")
    return recomputed


def _oracle_event_id(
    *,
    trace_id: str,
    episode_id: str,
    scene_id: str,
    task_object_id: str,
) -> str:
    return _content_sha256(
        {
            "domain": ORACLE_EVENT_ID_DOMAIN,
            "episode_id": episode_id,
            "scene_id": scene_id,
            "task_object_id": task_object_id,
            "trace_id": trace_id,
        }
    )


def _manifest_display_colors(
    manifest: SceneManifest,
    credited_ids: set[str],
) -> list[str]:
    colors: list[str] = []
    for landmark in manifest.landmarks:
        if landmark.object_id not in credited_ids:
            continue
        text = f"{landmark.material_id} {landmark.object_id}".lower()
        color = next(
            (item for item in ("green", "yellow", "blue", "red") if item in text),
            landmark.object_id,
        )
        colors.append(color)
    return sorted(colors)


def _trace_identity_checks(
    trace: Mapping[str, Any],
    *,
    scene_id: str,
    task_ids: Sequence[str],
    task_hash: str,
    manifest_hash: str,
    label: str,
    errors: list[str],
) -> None:
    if trace.get("scene_id") != scene_id:
        errors.append(f"{label}:trace_scene_identity_mismatch")
    if trace.get("task_object_ids") != list(task_ids):
        errors.append(f"{label}:trace_task_set_mismatch")
    if trace.get("task_object_set_sha256") != task_hash:
        errors.append(f"{label}:trace_task_commitment_mismatch")
    if trace.get("physical_manifest_sha256") != manifest_hash:
        errors.append(f"{label}:trace_manifest_commitment_mismatch")
    if trace.get("evaluator_feedback_to_controller") != []:
        errors.append(f"{label}:evaluator_feedback_nonempty")


def _accepted_event_checks(
    event: Mapping[str, Any],
    *,
    provenance: str,
    label: str,
    errors: list[str],
) -> None:
    if event.get("pose_provenance") != provenance:
        errors.append(f"{label}:pose_provenance_invalid")
    pose = event.get("robot_pose_world_xy_yaw")
    if not (
        isinstance(pose, list)
        and len(pose) == 3
        and all(type(value) in {int, float} and math.isfinite(value) for value in pose)
    ):
        errors.append(f"{label}:full_precision_pose_missing")
    if not (
        isinstance(event.get("pose_hex"), list)
        and len(event["pose_hex"]) == 3
        and all(type(value) is str for value in event["pose_hex"])
        and _is_sha256(event.get("pose_binary64_le_sha256"))
    ):
        errors.append(f"{label}:pose_commitment_missing")
    if not _canonical_equal(event.get("factors"), _TRUE_FACTORS):
        errors.append(f"{label}:physical_factors_not_all_true")
    if (
        event.get("decision") != "accepted"
        or event.get("accepted") is not True
        or event.get("physically_verified") is not True
        or event.get("credited") is not True
    ):
        errors.append(f"{label}:event_not_accepted_and_credited")
    if event.get("rejection_reasons") != []:
        errors.append(f"{label}:rejection_reasons_nonempty")
    if event.get("unverifiable_reasons") != []:
        errors.append(f"{label}:unverifiable_reasons_nonempty")
    if event.get("duplicate_physical_claim_not_credited") is not False:
        errors.append(f"{label}:duplicate_credit_present")


def _summary_checks(
    summary: Mapping[str, Any],
    *,
    task_ids: Sequence[str],
    label: str,
    errors: list[str],
) -> None:
    expected_count = len(task_ids)
    for field, expected in (
        ("attempted_count", expected_count),
        ("accepted_count", expected_count),
        ("rejected_count", 0),
        ("unverifiable_count", 0),
        ("credited_count", expected_count),
        ("duplicate_physical_claim_not_credited_count", 0),
    ):
        if not _is_exact_int(summary.get(field), expected):
            errors.append(f"{label}:summary_{field}_mismatch")
    if not _canonical_equal(summary.get("credited_object_ids"), list(task_ids)):
        errors.append(f"{label}:summary_credited_object_ids_mismatch")
    if summary.get("all_targets_claimed") is not True:
        errors.append(f"{label}:summary_not_all_targets_claimed")
    for field in ("unverifiable_reason_counts", "rejection_reason_counts"):
        values = summary.get(field)
        if not isinstance(values, Mapping) or any(
            not _is_exact_int(value, 0) for value in values.values()
        ):
            errors.append(f"{label}:summary_{field}_nonzero")
    if not _canonical_equal(
        summary.get("aggregate_reason_counts"),
        {"duplicate_physical_claim_not_credited": 0},
    ):
        errors.append(f"{label}:summary_duplicate_reason_nonzero")
    if summary.get("trace_unverifiable_reasons") != []:
        errors.append(f"{label}:summary_trace_unverifiable")


def _normalize_eligibility_reports(
    value: object,
    *,
    errors: list[str],
) -> dict[str, Mapping[str, Any]]:
    reports: dict[str, Mapping[str, Any]] = {}
    if isinstance(value, Mapping):
        items = list(value.items())
    elif isinstance(value, list):
        items = [
            (report.get("scene_id") if isinstance(report, Mapping) else None, report)
            for report in value
        ]
    else:
        errors.append("physical_eligibility_reports_not_collection")
        return reports
    for key, report in items:
        if not _is_nonempty_string(key) or not isinstance(report, Mapping):
            errors.append("physical_eligibility_report_invalid")
            continue
        if key in reports:
            errors.append(f"eligibility:{key}:duplicate_scene_report")
            continue
        if report.get("scene_id") != key:
            errors.append(f"eligibility:{key}:report_scene_identity_mismatch")
        reports[key] = report
    return reports


def _expected_input_access_ledger() -> dict[str, Any]:
    # Build from literals so mutation of an imported convenience constant cannot
    # weaken the finalizer's expectation.
    return {
        "preflight_hash_reads": {
            "development_manifest_hash_reads": 1,
            "materialization_hash_reads": 1,
            "geometry_hash_reads": 1,
            "primitive_registry_hash_reads": 1,
            "directional_policy_hash_reads": 1,
            "prior_comparator_hash_reads": 0,
        },
        "post_execution_hash_reads": {
            "development_manifest_hash_reads": 1,
            "materialization_hash_reads": 1,
            "geometry_hash_reads": 1,
            "primitive_registry_hash_reads": 1,
            "directional_policy_hash_reads": 1,
            "prior_comparator_hash_reads": 0,
        },
        "development_manifest_parse_calls": 1,
        "development_scene_manifest_parse_calls_by_parent": 24,
        "geometry_load_calls": 1,
        "primitive_registry_load_calls": 1,
        "directional_policy_load_calls": 1,
        "worker_runtime_input_file_opens": 0,
        "prior_comparator_payload_opens": 0,
        "heldout_payload_opens": 0,
        "sealed_payload_opens": 0,
        "g2_payload_opens": 0,
        "label_payload_opens": 0,
        "image_payload_opens": 0,
        "model_output_opens": 0,
    }


def _expected_evaluator_access_ledger() -> dict[str, int]:
    return {
        "evaluator_output_reads_by_controller": 0,
        "evaluator_callbacks_into_controller": 0,
        "evaluator_derived_termination_signals": 0,
    }


def finalize_canonical_physical_claim_oracle_regression(
    candidate: Mapping[str, Any],
    *,
    scene_manifests: Mapping[str, SceneManifest],
    expected_scene_ids: Sequence[str],
    expected_binding_sha256: str,
    expected_implementation_manifest_sha256: str,
    expected_source_map: Mapping[str, Any],
    expected_input_bindings: Mapping[str, Any],
    expected_command: object,
    expected_directional_policy_content_sha256: str,
) -> CanonicalPhysicalClaimOracleFinalization:
    """Independently verify and finalize the development-only 24x4 result."""

    errors: list[str] = []
    frozen_candidate = _freeze_or_error(
        candidate, label="candidate", errors=errors
    )
    frozen_source_map = _freeze_or_error(
        expected_source_map, label="expected_source_map", errors=errors
    )
    frozen_input_bindings = _freeze_or_error(
        expected_input_bindings, label="expected_input_bindings", errors=errors
    )
    frozen_command = _freeze_or_error(
        expected_command, label="expected_command", errors=errors
    )
    if not isinstance(frozen_candidate, dict):
        if frozen_candidate is not None:
            errors.append("candidate_not_object")
        return CanonicalPhysicalClaimOracleFinalization(False, tuple(errors), None)
    candidate = frozen_candidate

    if set(candidate) != _CANDIDATE_KEYS:
        errors.append("candidate_key_set_invalid")
    if candidate.get("schema") != CANDIDATE_SCHEMA:
        errors.append("candidate_schema_invalid")
    if not _is_sha256(expected_binding_sha256):
        errors.append("expected_binding_sha256_invalid")
    if candidate.get("binding_sha256") != expected_binding_sha256:
        errors.append("binding_sha256_mismatch")
    if not _is_sha256(expected_implementation_manifest_sha256):
        errors.append("expected_implementation_manifest_sha256_invalid")
    if (
        candidate.get("implementation_manifest_sha256")
        != expected_implementation_manifest_sha256
    ):
        errors.append("implementation_manifest_sha256_mismatch")
    if not isinstance(frozen_source_map, Mapping) or not frozen_source_map:
        errors.append("expected_source_map_invalid")
    if not _canonical_equal(candidate.get("source_map"), frozen_source_map):
        errors.append("source_map_binding_mismatch")
    if not isinstance(frozen_input_bindings, Mapping) or not frozen_input_bindings:
        errors.append("expected_input_bindings_invalid")
    if not _canonical_equal(candidate.get("input_bindings"), frozen_input_bindings):
        errors.append("input_bindings_mismatch")
    if isinstance(frozen_input_bindings, Mapping) and frozen_input_bindings.get(
        "directional_policy_content_sha256"
    ) != expected_directional_policy_content_sha256:
        errors.append("input_directional_policy_content_sha256_mismatch")
    if frozen_command in (None, "", [], {}):
        errors.append("expected_command_invalid")
    if not _canonical_equal(candidate.get("command"), frozen_command):
        errors.append("command_binding_mismatch")
    if not _is_sha256(expected_directional_policy_content_sha256):
        errors.append("expected_directional_policy_content_sha256_invalid")
    if not _canonical_equal(
        candidate.get("evaluator_access_ledger"),
        _expected_evaluator_access_ledger(),
    ):
        errors.append("evaluator_access_ledger_invalid")
    if not _canonical_equal(
        candidate.get("input_access_ledger"),
        _expected_input_access_ledger(),
    ):
        errors.append("input_access_ledger_invalid")

    manifests: dict[str, SceneManifest] = {}
    if not isinstance(scene_manifests, Mapping):
        errors.append("scene_manifests_not_mapping")
    else:
        for scene_id, manifest in scene_manifests.items():
            if not _is_nonempty_string(scene_id) or type(manifest) is not SceneManifest:
                errors.append("scene_manifest_entry_invalid")
                continue
            if manifest.scene_id != scene_id:
                errors.append(f"manifest:{scene_id}:scene_identity_mismatch")
            manifests[scene_id] = manifest
    frozen_scene_ids = _freeze_or_error(
        expected_scene_ids, label="expected_scene_ids", errors=errors
    )
    if not (
        isinstance(frozen_scene_ids, list)
        and len(frozen_scene_ids) == EXPECTED_SCENE_COUNT
        and all(_is_nonempty_string(scene_id) for scene_id in frozen_scene_ids)
        and len(set(frozen_scene_ids)) == EXPECTED_SCENE_COUNT
    ):
        errors.append("expected_scene_ids_not_24_unique_exact_ids")
        expected_scene_ids = []
    else:
        expected_scene_ids = frozen_scene_ids
    if len(manifests) != EXPECTED_SCENE_COUNT:
        errors.append("scene_manifest_count_not_24")
    if set(manifests) != set(expected_scene_ids):
        errors.append("scene_manifest_set_differs_from_expected_order")

    task_bindings: dict[str, tuple[list[str], str, str]] = {}
    expected_pairs: set[tuple[str, str]] = set()
    for scene_id in expected_scene_ids:
        manifest = manifests.get(scene_id)
        if manifest is None:
            errors.append(f"manifest:{scene_id}:missing_from_expected_panel")
            continue
        try:
            task_ids, manifest_hash, task_hash = _manifest_task_binding(manifest)
        except (OverflowError, TypeError, UnicodeEncodeError, ValueError):
            errors.append(f"manifest:{scene_id}:not_canonical")
            continue
        if (
            len(task_ids) != EXPECTED_TASKS_PER_SCENE
            or len(set(task_ids)) != EXPECTED_TASKS_PER_SCENE
            or any(not _is_nonempty_string(value) for value in task_ids)
        ):
            errors.append(f"manifest:{scene_id}:task_set_not_4_unique_objects")
        task_bindings[scene_id] = (task_ids, manifest_hash, task_hash)
        expected_pairs.update((scene_id, task_id) for task_id in task_ids)
    if len(expected_pairs) != EXPECTED_TASK_PAIR_COUNT:
        errors.append("manifest_task_pair_count_not_96")

    oracle_report = candidate.get("oracle_report")
    oracle_scenes: dict[str, Mapping[str, Any]] = {}
    if not isinstance(oracle_report, Mapping):
        errors.append("oracle_report_missing")
        oracle_report = {}
    else:
        if oracle_report.get("schema") != ORACLE_REPORT_SCHEMA:
            errors.append("oracle_report_schema_invalid")
        if oracle_report.get("development_only") is not True:
            errors.append("oracle_report_not_development_only")
        if not _canonical_equal(
            oracle_report.get("scene_execution"),
            {
                "kind": "spawn_process",
                "worker_count": 6,
                "threads_per_worker": 1,
                "merge_order": "development_manifest_index",
                "worker_runtime_input_file_access": False,
            },
        ):
            errors.append("oracle_report_scene_execution_contract_mismatch")
        raw_scenes = oracle_report.get("scenes")
        if not isinstance(raw_scenes, list):
            errors.append("oracle_report_scenes_not_list")
        else:
            for report in raw_scenes:
                scene_id = report.get("scene_id") if isinstance(report, Mapping) else None
                if not _is_nonempty_string(scene_id) or not isinstance(report, Mapping):
                    errors.append("oracle_scene_report_invalid")
                    continue
                if scene_id in oracle_scenes:
                    errors.append(f"oracle:{scene_id}:duplicate_scene_report")
                    continue
                oracle_scenes[scene_id] = report
        report_scene_ids = oracle_report.get("scene_ids")
        if (
            not isinstance(report_scene_ids, list)
            or report_scene_ids != expected_scene_ids
            or list(oracle_scenes) != expected_scene_ids
        ):
            errors.append("oracle_report_scene_ids_mismatch")
        geometry = oracle_report.get("geometry_contract")
        expected_geometry = (
            frozen_input_bindings.get("geometry_contract_sha256")
            if isinstance(frozen_input_bindings, Mapping)
            else None
        )
        if not _is_sha256(expected_geometry):
            errors.append("expected_geometry_contract_sha256_missing")
        if not isinstance(geometry, Mapping) or geometry.get("sha256") != expected_geometry:
            errors.append("oracle_report_geometry_binding_mismatch")
        expected_oracle_config = (
            frozen_input_bindings.get("oracle_config")
            if isinstance(frozen_input_bindings, Mapping)
            else None
        )
        if not _canonical_equal(oracle_report.get("config"), expected_oracle_config):
            errors.append("oracle_report_config_binding_mismatch")
    if set(oracle_scenes) != set(expected_scene_ids):
        errors.append("oracle_scene_set_mismatch")

    eligibility_reports = _normalize_eligibility_reports(
        candidate.get("physical_eligibility_reports"), errors=errors
    )
    if set(eligibility_reports) != set(expected_scene_ids):
        errors.append("eligibility_scene_set_mismatch")

    trace_ids: list[str] = []
    episode_ids: list[str] = []
    trace_tuples: list[tuple[object, object, object]] = []
    oracle_pairs: set[tuple[str, str]] = set()
    eligibility_pairs: set[tuple[str, str]] = set()
    oracle_raw_count = oracle_evaluation_count = 0
    oracle_accepted_count = oracle_credited_count = 0
    oracle_rejected_count = oracle_unverifiable_count = oracle_duplicate_count = 0
    eligibility_raw_count = eligibility_evaluation_count = 0
    eligibility_accepted_count = eligibility_credited_count = 0
    eligibility_rejected_count = eligibility_unverifiable_count = 0
    eligibility_duplicate_count = eligibility_reachable_anchor_count = 0
    eligibility_credited_anchor_count = 0
    shared_map_scene_count = all_oracle_complete_scene_count = 0
    eligible_scene_count = 0
    collision_count = stall_count = polygon_collision_segment_count = 0
    oracle_trace_hashes: dict[str, str] = {}
    eligibility_trace_hashes: dict[str, str] = {}

    for scene_id in expected_scene_ids:
        if scene_id not in task_bindings:
            continue
        task_ids, manifest_hash, task_hash = task_bindings[scene_id]
        manifest = manifests[scene_id]
        oracle_scene = oracle_scenes.get(scene_id)
        if oracle_scene is not None:
            label = f"oracle:{scene_id}"
            trace = _recompute_trace(
                oracle_scene.get("canonical_physical_claim_trace"),
                manifest=manifest,
                task_ids=task_ids,
                task_hash=task_hash,
                label=label,
                errors=errors,
            )
            if trace is not None:
                _trace_identity_checks(
                    trace,
                    scene_id=scene_id,
                    task_ids=task_ids,
                    task_hash=task_hash,
                    manifest_hash=manifest_hash,
                    label=label,
                    errors=errors,
                )
                trace_ids.append(str(trace.get("trace_id")))
                episode_ids.append(str(trace.get("episode_id")))
                trace_tuples.append(
                    (trace.get("trace_id"), trace.get("episode_id"), trace.get("scene_id"))
                )
                attempts = trace.get("controller_claim_attempts")
                evaluations = trace.get("physical_claim_evaluations")
                attempts = attempts if isinstance(attempts, list) else []
                evaluations = evaluations if isinstance(evaluations, list) else []
                oracle_raw_count += len(attempts)
                oracle_evaluation_count += len(evaluations)
                if len(attempts) != EXPECTED_TASKS_PER_SCENE:
                    errors.append(f"{label}:raw_attempt_count_not_4")
                if len(evaluations) != EXPECTED_TASKS_PER_SCENE:
                    errors.append(f"{label}:evaluation_count_not_4")
                scene_pairs: set[tuple[str, str]] = set()
                for index, (attempt, event) in enumerate(zip(attempts, evaluations)):
                    event_label = f"{label}:event:{index}"
                    if not isinstance(attempt, Mapping) or not isinstance(event, Mapping):
                        errors.append(f"{event_label}:entry_not_object")
                        continue
                    requested = attempt.get("requested_target")
                    claimed = attempt.get("claimed_target")
                    if (
                        not isinstance(requested, Mapping)
                        or set(requested) != {"namespace", "value"}
                        or requested.get("namespace") != "object_id"
                        or requested.get("value") not in task_ids
                        or not _canonical_equal(claimed, requested)
                    ):
                        errors.append(f"{event_label}:task_reference_invalid")
                        continue
                    task_object_id = requested["value"]
                    pair = (scene_id, task_object_id)
                    scene_pairs.add(pair)
                    oracle_pairs.add(pair)
                    expected_event_id = _oracle_event_id(
                        trace_id=str(trace.get("trace_id")),
                        episode_id=str(trace.get("episode_id")),
                        scene_id=scene_id,
                        task_object_id=task_object_id,
                    )
                    if attempt.get("event_id") != expected_event_id:
                        errors.append(f"{event_label}:oracle_event_id_domain_mismatch")
                    if not _canonical_equal(
                        event.get("event_id"), attempt.get("event_id")
                    ):
                        errors.append(f"{event_label}:attempt_evaluation_join_mismatch")
                    _accepted_event_checks(
                        event,
                        provenance="oracle_full_precision",
                        label=event_label,
                        errors=errors,
                    )
                if scene_pairs != {(scene_id, task_id) for task_id in task_ids}:
                    errors.append(f"{label}:task_pair_set_mismatch")
                summary = trace.get("physical_claim_summary")
                if isinstance(summary, Mapping):
                    _summary_checks(
                        summary, task_ids=task_ids, label=label, errors=errors
                    )
                    oracle_accepted_count += int(summary.get("accepted_count", 0))
                    oracle_credited_count += int(summary.get("credited_count", 0))
                    oracle_rejected_count += int(summary.get("rejected_count", 0))
                    oracle_unverifiable_count += int(summary.get("unverifiable_count", 0))
                    oracle_duplicate_count += int(
                        summary.get("duplicate_physical_claim_not_credited_count", 0)
                    )
                    if summary.get("all_targets_claimed") is True:
                        all_oracle_complete_scene_count += 1
                else:
                    errors.append(f"{label}:summary_missing")
                trace_hash = trace.get("trace_content_sha256")
                if _is_sha256(trace_hash):
                    oracle_trace_hashes[scene_id] = trace_hash

                credited_ids = set(task_ids)
                expected_ticks = {
                    str(event["claimed_target_object_id"]): int(event["tick"])
                    for event in evaluations
                    if isinstance(event, Mapping) and event.get("credited") is True
                }
                expected_poses = {
                    str(event["claimed_target_object_id"]): event["robot_pose_world_xy_yaw"]
                    for event in evaluations
                    if isinstance(event, Mapping) and event.get("credited") is True
                }
                top_level_expectations = {
                    "success": True,
                    "all_beacons_claimed": True,
                    "claimed_count": EXPECTED_TASKS_PER_SCENE,
                    "beacon_count": EXPECTED_TASKS_PER_SCENE,
                    "claimed_beacon_ids": task_ids,
                    "claimed_colors": _manifest_display_colors(manifest, credited_ids),
                    "claim_ticks": dict(sorted(expected_ticks.items())),
                    "claim_poses": dict(sorted(expected_poses.items())),
                    "failure_class": "success",
                    "geometry_failures": [],
                    "planner_failures": [],
                    "follower_failures": [],
                    "strict_directional_safe": True,
                    "directional_polygon_initial_pose_feasible": True,
                    "directional_polygon_collision_object_ids": [],
                }
                for field, expected in top_level_expectations.items():
                    if not _canonical_equal(oracle_scene.get(field), expected):
                        errors.append(f"{label}:top_level_{field}_mismatch")
                expected_geometry = (
                    frozen_input_bindings.get("geometry_contract_sha256")
                    if isinstance(frozen_input_bindings, Mapping)
                    else None
                )
                if oracle_scene.get("geometry_contract_sha256") != expected_geometry:
                    errors.append(f"{label}:geometry_binding_mismatch")

            route = oracle_scene.get("route_planner")
            if isinstance(route, Mapping) and route.get("source") == ORACLE_ROUTE_SOURCE:
                shared_map_scene_count += 1
            else:
                errors.append(f"{label}:route_source_not_online_belief_map")
            directional_policy = oracle_scene.get("directional_policy")
            if (
                not isinstance(directional_policy, Mapping)
                or directional_policy.get("content_sha256")
                != expected_directional_policy_content_sha256
            ):
                errors.append(f"{label}:directional_policy_content_mismatch")
            for field, counter_name in (
                ("collisions", "collision"),
                ("stalls", "stall"),
                ("directional_polygon_collision_segments", "polygon_collision"),
            ):
                value = oracle_scene.get(field)
                if not _is_exact_int(value, 0):
                    errors.append(f"{label}:{field}_nonzero")
                if _is_exact_int(value):
                    if counter_name == "collision":
                        collision_count += value
                    elif counter_name == "stall":
                        stall_count += value
                    else:
                        polygon_collision_segment_count += value

        eligibility = eligibility_reports.get(scene_id)
        if eligibility is not None:
            label = f"eligibility:{scene_id}"
            if eligibility.get("schema") != ELIGIBILITY_REPORT_SCHEMA:
                errors.append(f"{label}:schema_invalid")
            if eligibility.get("family") != manifest.family:
                errors.append(f"{label}:family_mismatch")
            if (
                eligibility.get("policy_content_sha256")
                != expected_directional_policy_content_sha256
            ):
                errors.append(f"{label}:policy_content_mismatch")
            if eligibility.get("policy_profile") != "observed_max_plus_margin":
                errors.append(f"{label}:policy_profile_invalid")
            expected_eligibility_config = (
                frozen_input_bindings.get("physical_eligibility_config")
                if isinstance(frozen_input_bindings, Mapping)
                else None
            )
            if not _canonical_equal(
                eligibility.get("config"), expected_eligibility_config
            ):
                errors.append(f"{label}:config_binding_mismatch")
            if eligibility.get("eligible") is not True or eligibility.get("failure_reason") != "":
                errors.append(f"{label}:not_cleanly_eligible")
            else:
                eligible_scene_count += 1
            for field in ("spawn_clear_at_actual_yaw", "spawn_snaps_to_lattice"):
                if eligibility.get(field) is not True:
                    errors.append(f"{label}:{field}_not_true")

            trace = _recompute_trace(
                eligibility.get("canonical_physical_claim_trace"),
                manifest=manifest,
                task_ids=task_ids,
                task_hash=task_hash,
                label=label,
                errors=errors,
            )
            attempts: list[Any] = []
            evaluations: list[Any] = []
            if trace is not None:
                _trace_identity_checks(
                    trace,
                    scene_id=scene_id,
                    task_ids=task_ids,
                    task_hash=task_hash,
                    manifest_hash=manifest_hash,
                    label=label,
                    errors=errors,
                )
                trace_ids.append(str(trace.get("trace_id")))
                episode_ids.append(str(trace.get("episode_id")))
                trace_tuples.append(
                    (trace.get("trace_id"), trace.get("episode_id"), trace.get("scene_id"))
                )
                attempts_value = trace.get("controller_claim_attempts")
                evaluations_value = trace.get("physical_claim_evaluations")
                attempts = attempts_value if isinstance(attempts_value, list) else []
                evaluations = evaluations_value if isinstance(evaluations_value, list) else []
                eligibility_raw_count += len(attempts)
                eligibility_evaluation_count += len(evaluations)
                if len(attempts) != EXPECTED_TASKS_PER_SCENE:
                    errors.append(f"{label}:raw_attempt_count_not_4")
                if len(evaluations) != EXPECTED_TASKS_PER_SCENE:
                    errors.append(f"{label}:evaluation_count_not_4")
                scene_pairs: set[tuple[str, str]] = set()
                for index, (attempt, event) in enumerate(zip(attempts, evaluations)):
                    event_label = f"{label}:event:{index}"
                    if not isinstance(attempt, Mapping) or not isinstance(event, Mapping):
                        errors.append(f"{event_label}:entry_not_object")
                        continue
                    requested = attempt.get("requested_target")
                    claimed = attempt.get("claimed_target")
                    if (
                        not isinstance(requested, Mapping)
                        or set(requested) != {"namespace", "value"}
                        or requested.get("namespace") != "object_id"
                        or requested.get("value") not in task_ids
                        or not _canonical_equal(claimed, requested)
                    ):
                        errors.append(f"{event_label}:task_reference_invalid")
                        continue
                    pair = (scene_id, requested["value"])
                    scene_pairs.add(pair)
                    eligibility_pairs.add(pair)
                    if not _canonical_equal(
                        event.get("event_id"), attempt.get("event_id")
                    ):
                        errors.append(f"{event_label}:attempt_evaluation_join_mismatch")
                    _accepted_event_checks(
                        event,
                        provenance="eligibility_candidate_full_precision",
                        label=event_label,
                        errors=errors,
                    )
                if scene_pairs != {(scene_id, task_id) for task_id in task_ids}:
                    errors.append(f"{label}:task_pair_set_mismatch")
                summary = trace.get("physical_claim_summary")
                if isinstance(summary, Mapping):
                    _summary_checks(
                        summary, task_ids=task_ids, label=label, errors=errors
                    )
                    eligibility_accepted_count += int(summary.get("accepted_count", 0))
                    eligibility_credited_count += int(summary.get("credited_count", 0))
                    eligibility_rejected_count += int(summary.get("rejected_count", 0))
                    eligibility_unverifiable_count += int(summary.get("unverifiable_count", 0))
                    eligibility_duplicate_count += int(
                        summary.get("duplicate_physical_claim_not_credited_count", 0)
                    )
                else:
                    errors.append(f"{label}:summary_missing")
                trace_hash = trace.get("trace_content_sha256")
                if _is_sha256(trace_hash):
                    eligibility_trace_hashes[scene_id] = trace_hash

            anchors = eligibility.get("claim_anchors")
            if not isinstance(anchors, list) or len(anchors) != EXPECTED_TASKS_PER_SCENE:
                errors.append(f"{label}:claim_anchor_count_not_4")
                anchors = []
            anchor_ids: set[str] = set()
            attempts_by_target: dict[str, Mapping[str, Any]] = {}
            for attempt in attempts:
                if not isinstance(attempt, Mapping):
                    continue
                requested = attempt.get("requested_target")
                if not isinstance(requested, Mapping):
                    continue
                requested_value = requested.get("value")
                if type(requested_value) is str:
                    attempts_by_target[requested_value] = attempt
            for index, anchor in enumerate(anchors):
                anchor_label = f"{label}:anchor:{index}"
                if not isinstance(anchor, Mapping):
                    errors.append(f"{anchor_label}:not_object")
                    continue
                object_id = anchor.get("object_id")
                if object_id not in task_ids or object_id in anchor_ids:
                    errors.append(f"{anchor_label}:object_identity_invalid")
                    continue
                anchor_ids.add(object_id)
                if anchor.get("reachable") is not True:
                    errors.append(f"{anchor_label}:not_reachable")
                else:
                    eligibility_reachable_anchor_count += 1
                if anchor.get("physical_claim_credited") is not True:
                    errors.append(f"{anchor_label}:not_physically_credited")
                else:
                    eligibility_credited_anchor_count += 1
                if anchor.get("physical_claim_decision") != "accepted":
                    errors.append(f"{anchor_label}:physical_decision_not_accepted")
                if anchor.get("physical_claim_unverifiable_reasons") != []:
                    errors.append(f"{anchor_label}:unverifiable_reasons_nonempty")
                if anchor.get("physical_claim_rejection_reasons") != []:
                    errors.append(f"{anchor_label}:rejection_reasons_nonempty")
                if anchor.get("anchor_has_line_of_sight") is not True:
                    errors.append(f"{anchor_label}:line_of_sight_not_true")
                if not (
                    _is_exact_int(anchor.get("reachable_claim_state_count"))
                    and anchor["reachable_claim_state_count"] > 0
                ):
                    errors.append(f"{anchor_label}:reachable_state_count_invalid")
                lattice_state = anchor.get("anchor_lattice_state")
                if not (
                    isinstance(lattice_state, list)
                    and len(lattice_state) == 3
                    and all(_is_exact_int(value) for value in lattice_state)
                ):
                    errors.append(f"{anchor_label}:lattice_state_invalid")
                target_xy = anchor.get("target_xy_m")
                landmark = next(
                    (item for item in manifest.landmarks if item.object_id == object_id),
                    None,
                )
                if landmark is None or not _canonical_equal(
                    target_xy,
                    [
                        float(landmark.center_xyz_m[0]),
                        float(landmark.center_xyz_m[1]),
                    ],
                ):
                    errors.append(f"{anchor_label}:target_position_mismatch")
                attempt = attempts_by_target.get(object_id)
                if (
                    not isinstance(attempt, Mapping)
                    or not _canonical_equal(
                        anchor.get("anchor_pose"),
                        attempt.get("robot_pose_world_xy_yaw"),
                    )
                ):
                    errors.append(f"{anchor_label}:anchor_pose_trace_mismatch")
            if anchor_ids != set(task_ids):
                errors.append(f"{label}:anchor_task_set_mismatch")

    if set(oracle_pairs) != expected_pairs:
        errors.append("oracle_task_pair_set_not_exact_96")
    if set(eligibility_pairs) != expected_pairs:
        errors.append("eligibility_task_pair_set_not_exact_96")
    if oracle_pairs != eligibility_pairs:
        errors.append("oracle_eligibility_task_pair_sets_differ")
    if len(trace_ids) != 2 * EXPECTED_SCENE_COUNT:
        errors.append("trace_count_not_48")
    if any(count != 1 for count in Counter(trace_ids).values()):
        errors.append("trace_ids_not_globally_unique")
    if any(count != 1 for count in Counter(episode_ids).values()):
        errors.append("episode_ids_not_globally_unique")
    if any(count != 1 for count in Counter(trace_tuples).values()):
        errors.append("trace_episode_scene_tuples_not_unique")

    expected_oracle_aggregate = {
        "scene_count": EXPECTED_SCENE_COUNT,
        "all_beacons_claimed_scenes": EXPECTED_SCENE_COUNT,
        "full_4_of_4_claim_scenes": EXPECTED_SCENE_COUNT,
        "positive_control_success_scenes": EXPECTED_SCENE_COUNT,
        "claimed_beacons": EXPECTED_TASK_PAIR_COUNT,
        "expected_beacons": EXPECTED_TASK_PAIR_COUNT,
        "collisions": 0,
        "stalls": 0,
        "directional_polygon_collision_segments": 0,
        "strict_directional_safe_scenes": EXPECTED_SCENE_COUNT,
        "shared_map_routed_scenes": EXPECTED_SCENE_COUNT,
        "all_claims_zero_collision_zero_stall_gate_passed": True,
        "development_24x4_strict_gate_passed": True,
        "failure_classes": {"success": EXPECTED_SCENE_COUNT},
    }
    stored_oracle_aggregate = (
        oracle_report.get("aggregate") if isinstance(oracle_report, Mapping) else None
    )
    if not isinstance(stored_oracle_aggregate, Mapping):
        errors.append("oracle_aggregate_missing")
    else:
        for field, expected in expected_oracle_aggregate.items():
            if not _canonical_equal(stored_oracle_aggregate.get(field), expected):
                errors.append(f"oracle_aggregate_{field}_mismatch")

    totals = {
        "scene_count": len(expected_scene_ids),
        "task_pair_count": len(expected_pairs),
        "oracle_raw_attempt_count": oracle_raw_count,
        "oracle_evaluation_count": oracle_evaluation_count,
        "oracle_accepted_count": oracle_accepted_count,
        "oracle_credited_count": oracle_credited_count,
        "oracle_rejected_count": oracle_rejected_count,
        "oracle_unverifiable_count": oracle_unverifiable_count,
        "oracle_duplicate_credit_count": oracle_duplicate_count,
        "eligibility_raw_attempt_count": eligibility_raw_count,
        "eligibility_evaluation_count": eligibility_evaluation_count,
        "eligibility_accepted_count": eligibility_accepted_count,
        "eligibility_credited_count": eligibility_credited_count,
        "eligibility_rejected_count": eligibility_rejected_count,
        "eligibility_unverifiable_count": eligibility_unverifiable_count,
        "eligibility_duplicate_credit_count": eligibility_duplicate_count,
        "eligibility_reachable_anchor_count": eligibility_reachable_anchor_count,
        "eligibility_credited_anchor_count": eligibility_credited_anchor_count,
        "collisions": collision_count,
        "stalls": stall_count,
        "directional_polygon_collision_segments": polygon_collision_segment_count,
    }
    expected_totals = {
        **totals,
        "scene_count": EXPECTED_SCENE_COUNT,
        "task_pair_count": EXPECTED_TASK_PAIR_COUNT,
        "oracle_raw_attempt_count": EXPECTED_TASK_PAIR_COUNT,
        "oracle_evaluation_count": EXPECTED_TASK_PAIR_COUNT,
        "oracle_accepted_count": EXPECTED_TASK_PAIR_COUNT,
        "oracle_credited_count": EXPECTED_TASK_PAIR_COUNT,
        "oracle_rejected_count": 0,
        "oracle_unverifiable_count": 0,
        "oracle_duplicate_credit_count": 0,
        "eligibility_raw_attempt_count": EXPECTED_TASK_PAIR_COUNT,
        "eligibility_evaluation_count": EXPECTED_TASK_PAIR_COUNT,
        "eligibility_accepted_count": EXPECTED_TASK_PAIR_COUNT,
        "eligibility_credited_count": EXPECTED_TASK_PAIR_COUNT,
        "eligibility_rejected_count": 0,
        "eligibility_unverifiable_count": 0,
        "eligibility_duplicate_credit_count": 0,
        "eligibility_reachable_anchor_count": EXPECTED_TASK_PAIR_COUNT,
        "eligibility_credited_anchor_count": EXPECTED_TASK_PAIR_COUNT,
        "collisions": 0,
        "stalls": 0,
        "directional_polygon_collision_segments": 0,
    }
    for field, expected in expected_totals.items():
        if totals.get(field) != expected:
            errors.append(f"recomputed_total_{field}_mismatch")

    aggregate = {
        "oracle_all_targets_claimed_scene_count": all_oracle_complete_scene_count,
        "eligible_scene_count": eligible_scene_count,
        "online_belief_map_routed_scene_count": shared_map_scene_count,
        "oracle_task_pairs_exact": oracle_pairs == expected_pairs,
        "eligibility_task_pairs_exact": eligibility_pairs == expected_pairs,
        "oracle_eligibility_task_pairs_equal": oracle_pairs == eligibility_pairs,
        "all_oracle_physical_claims_accepted_and_credited": bool(
            oracle_accepted_count == EXPECTED_TASK_PAIR_COUNT
            and oracle_credited_count == EXPECTED_TASK_PAIR_COUNT
            and oracle_rejected_count == 0
            and oracle_unverifiable_count == 0
            and oracle_duplicate_count == 0
        ),
        "all_eligibility_claims_accepted_and_credited": bool(
            eligibility_accepted_count == EXPECTED_TASK_PAIR_COUNT
            and eligibility_credited_count == EXPECTED_TASK_PAIR_COUNT
            and eligibility_rejected_count == 0
            and eligibility_unverifiable_count == 0
            and eligibility_duplicate_count == 0
        ),
        "zero_collision_stall_polygon_gate": bool(
            collision_count == 0
            and stall_count == 0
            and polygon_collision_segment_count == 0
        ),
    }
    expected_aggregate = {
        "oracle_all_targets_claimed_scene_count": EXPECTED_SCENE_COUNT,
        "eligible_scene_count": EXPECTED_SCENE_COUNT,
        "online_belief_map_routed_scene_count": EXPECTED_SCENE_COUNT,
        "oracle_task_pairs_exact": True,
        "eligibility_task_pairs_exact": True,
        "oracle_eligibility_task_pairs_equal": True,
        "all_oracle_physical_claims_accepted_and_credited": True,
        "all_eligibility_claims_accepted_and_credited": True,
        "zero_collision_stall_polygon_gate": True,
    }
    for field, expected in expected_aggregate.items():
        if aggregate.get(field) != expected:
            errors.append(f"recomputed_aggregate_{field}_mismatch")

    if errors:
        return CanonicalPhysicalClaimOracleFinalization(
            passed=False,
            errors=tuple(errors),
            finalized_payload=None,
        )

    finalized_core = {
        "schema": FINALIZED_SCHEMA,
        "binding_sha256": expected_binding_sha256,
        "implementation_manifest_sha256": expected_implementation_manifest_sha256,
        "source_map": frozen_source_map,
        "input_bindings": frozen_input_bindings,
        "command": frozen_command,
        "evaluator_access_ledger": _expected_evaluator_access_ledger(),
        "input_access_ledger": _expected_input_access_ledger(),
        "scene_ids": expected_scene_ids,
        "oracle_report": oracle_report,
        "physical_eligibility_reports": [
            eligibility_reports[scene_id] for scene_id in expected_scene_ids
        ],
        "oracle_trace_content_sha256_by_scene": dict(
            sorted(oracle_trace_hashes.items())
        ),
        "eligibility_trace_content_sha256_by_scene": dict(
            sorted(eligibility_trace_hashes.items())
        ),
        "totals": totals,
        "aggregate": aggregate,
        "finalization_passed": True,
    }
    finalized_payload = {
        **finalized_core,
        "content_sha256": _content_sha256(finalized_core),
    }
    return CanonicalPhysicalClaimOracleFinalization(
        passed=True,
        errors=(),
        finalized_payload=finalized_payload,
    )


__all__ = [
    "CANDIDATE_SCHEMA",
    "FINALIZED_SCHEMA",
    "CanonicalPhysicalClaimOracleFinalization",
    "finalize_canonical_physical_claim_oracle_regression",
]
