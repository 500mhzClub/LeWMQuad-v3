#!/usr/bin/env python3
"""Receipt-only analyzer for the 160-branch Go2 counterfactual calibration.

The analyzer never opens an RGB leaf, simulator input, or checkpoint.  It
starts from one caller-bound, checker-validated physics collection and derives
the physical-rank tolerances and terminal calibration decision.  The resulting
receipt is an input to the separate pilot join; it grants no runtime authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import check_go2_world_model_counterfactual_pilot_v1 as checker  # noqa: E402
from lewm.benchmarks import (  # noqa: E402
    go2_world_model_counterfactual_pilot_v1 as producer_contract,
)


CALIBRATION_RECEIPT_SCHEMA = (
    "lewm_go2_world_model_counterfactual_calibration_receipt_v1"
)
TEXTURED_V03_CALIBRATION_RECEIPT_SCHEMA = (
    "lewm_go2_world_model_counterfactual_calibration_receipt_v3"
)
CANDIDATE_BRANCH_SUPPORT_ANALYSIS_SCHEMA = (
    "lewm_go2_world_model_counterfactual_candidate_branch_support_analysis_v3"
)
PHYSICAL_RANK_CONTRACT_SCHEMA = (
    "lewm_go2_world_model_counterfactual_physical_rank_contract_v1"
)
TOLERANCE_DERIVATION_SCHEMA = (
    "lewm_go2_world_model_counterfactual_tolerance_derivation_v1"
)
TOLERANCE_DERIVATION_V2_SCHEMA = (
    "lewm_go2_world_model_counterfactual_tolerance_derivation_v2"
)
RESOURCE_MEASUREMENTS_SCHEMA = (
    "lewm_go2_world_model_counterfactual_calibration_resource_measurements_v1"
)
TEXTURED_V03_RESOURCE_MEASUREMENTS_SCHEMA = (
    "lewm_go2_world_model_counterfactual_calibration_resource_measurements_v2"
)
ANALYZER_SOURCE_NAME = "calibration_analyzer"
JOINER_SOURCE_NAME = "pilot_joiner"
CHECKER_SOURCE_NAME = "checker"
NUMERICAL_FLOOR_M = 1.0e-6
TEXTURED_V03_OUTCOME_EQUIVALENCE_TOLERANCE_M = 0.01
ACTION_COUNT = 9
CALIBRATION_STATE_COUNT = 16
CALIBRATION_SCENE_COUNT = 8
CALIBRATION_BRANCH_COUNT = 160
CALIBRATION_STATES_PER_FAMILY = 2
MIN_ELIGIBLE_QUERIES_PER_FAMILY = 9
MIN_ELIGIBLE_QUERIES_OVERALL = 72
CALIBRATED_MINIMUM_DISCRIMINATION_QUERY_COVERAGE = 0.5
BOUNDED_MINIMUM_DISCRIMINATION_QUERY_COVERAGE = 0.25
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
TEXTURED_V03_CALIBRATION_PURPOSE = "sizing_calibration_textured_v03_v3"
EQUIVALENCE_PARTITION_NAMES = (
    "executed_tape",
    "physical_trajectory",
    "endpoint_pose",
    "physical_outcome",
    "stored_rgb_file",
    "stored_rgb_pixels",
)
LEGACY_LOW_INFO_REASON_NAMES = (
    "low_rgb_texture",
    "near_wall_depth",
    "near_forward_geometry",
)
TEXTURED_V03_LOW_INFO_REASON_NAMES = (
    "camera_safety_unresolved",
    *LEGACY_LOW_INFO_REASON_NAMES,
)


class CalibrationAnalysisError(RuntimeError):
    """Raised before an invalid collection can mint a calibration receipt."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _binding(path: Path) -> dict[str, object]:
    selected = Path(path).resolve(strict=True)
    if selected.is_symlink() or not selected.is_file():
        raise CalibrationAnalysisError(f"bound input is not a regular file: {selected}")
    raw = selected.read_bytes()
    return {
        "path": str(selected),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }


def _validate_binding(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "file_sha256",
        "byte_count",
    }:
        raise CalibrationAnalysisError(f"{name} binding is malformed")
    path = value["path"]
    digest = value["file_sha256"]
    byte_count = value["byte_count"]
    if (
        not isinstance(path, str)
        or not Path(path).is_absolute()
        or str(Path(path)) != path
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or type(byte_count) is not int
        or byte_count <= 0
    ):
        raise CalibrationAnalysisError(f"{name} binding is malformed")
    return {
        "path": path,
        "file_sha256": digest,
        "byte_count": byte_count,
    }


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationAnalysisError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CalibrationAnalysisError(f"{name} must be finite")
    return result


def _calibration_variant(collection: Mapping[str, Any]) -> str:
    """Select the legacy or textured-v03 contract from checker-validated data."""

    purpose = collection.get("purpose")
    plan = collection.get("plan")
    plan_document = plan.get("document") if isinstance(plan, Mapping) else None
    plan_purpose = (
        plan_document.get("purpose") if isinstance(plan_document, Mapping) else None
    )
    render_contract = (
        plan_document.get("render_contract")
        if isinstance(plan_document, Mapping)
        else None
    )
    states = collection.get("states")
    state_documents = [
        state.get("document")
        for state in states
        if isinstance(states, Sequence)
        and not isinstance(states, (str, bytes))
        and isinstance(state, Mapping)
    ] if isinstance(states, Sequence) and not isinstance(states, (str, bytes)) else []
    has_v2_state_signal = any(
        isinstance(document, Mapping)
        and document.get("schema")
        == producer_contract.TEXTURED_V03_STATE_RECEIPT_SCHEMA
        for document in state_documents
    )
    textured_signal = (
        purpose == TEXTURED_V03_CALIBRATION_PURPOSE
        or plan_purpose == TEXTURED_V03_CALIBRATION_PURPOSE
        or render_contract == producer_contract.TEXTURED_V03_RENDER_CONTRACT
        or has_v2_state_signal
    )
    if not textured_signal:
        return "legacy_v1"
    if (
        purpose != TEXTURED_V03_CALIBRATION_PURPOSE
        or not isinstance(plan, Mapping)
        or not isinstance(plan_document, Mapping)
        or plan_purpose != TEXTURED_V03_CALIBRATION_PURPOSE
        or render_contract != producer_contract.TEXTURED_V03_RENDER_CONTRACT
        or not isinstance(states, Sequence)
        or isinstance(states, (str, bytes))
        or len(states) != CALIBRATION_STATE_COUNT
    ):
        raise CalibrationAnalysisError(
            "textured-v03 calibration identity is only partially present"
        )
    plan_states = plan.get("states")
    if (
        not isinstance(plan_states, Sequence)
        or isinstance(plan_states, (str, bytes))
        or len(plan_states) != CALIBRATION_STATE_COUNT
    ):
        raise CalibrationAnalysisError("textured-v03 validated plan states are absent")
    for index, (state, plan_state) in enumerate(zip(states, plan_states, strict=True)):
        if not isinstance(state, Mapping) or not isinstance(plan_state, Mapping):
            raise CalibrationAnalysisError(
                f"textured-v03 calibration state {index} is malformed"
            )
        state_document = state.get("document")
        audit = (
            state_document.get("candidate_response_audit")
            if isinstance(state_document, Mapping)
            else None
        )
        if (
            not isinstance(state_document, Mapping)
            or state_document.get("schema")
            != producer_contract.TEXTURED_V03_STATE_RECEIPT_SCHEMA
            or not isinstance(audit, Mapping)
            or audit.get("schema")
            != producer_contract.TEXTURED_V03_CANDIDATE_RESPONSE_AUDIT_SCHEMA
        ):
            raise CalibrationAnalysisError(
                "textured-v03 calibration requires V2 state and candidate-audit schemas"
            )
        checked_state = state.get("state")
        if (
            not isinstance(checked_state, Mapping)
            or checked_state.get("state_id") != plan_state.get("state_id")
            or checked_state.get("family") != plan_state.get("family")
            or checked_state.get("scene_id") != plan_state.get("scene_id")
        ):
            raise CalibrationAnalysisError(
                "textured-v03 state identity changed from the validated plan"
            )
        context = state.get("context")
        if (
            not isinstance(context, Mapping)
            or context.get("history_action_ids")
            != plan_state.get("history_action_ids")
        ):
            raise CalibrationAnalysisError(
                "textured-v03 history identity changed from the validated plan"
            )
    return "textured_v03_v3"


def _validate_equivalence_partition(
    value: object,
    *,
    name: str,
) -> int:
    if not isinstance(value, Mapping) or set(value) != {
        "unique_count",
        "collapsed",
        "groups",
    }:
        raise CalibrationAnalysisError(f"{name} partition is malformed")
    unique_count = value["unique_count"]
    groups = value["groups"]
    if (
        type(unique_count) is not int
        or not 1 <= unique_count <= ACTION_COUNT
        or type(value["collapsed"]) is not bool
        or value["collapsed"] is not (unique_count < ACTION_COUNT)
        or not isinstance(groups, list)
        or len(groups) != unique_count
    ):
        raise CalibrationAnalysisError(f"{name} partition summary changed")
    covered_actions: list[int] = []
    identities: set[str] = set()
    for group_index, group in enumerate(groups):
        if not isinstance(group, Mapping) or set(group) != {
            "identity_sha256",
            "action_ids",
        }:
            raise CalibrationAnalysisError(
                f"{name} partition group {group_index} is malformed"
            )
        identity = group["identity_sha256"]
        action_ids = group["action_ids"]
        if (
            not isinstance(identity, str)
            or _SHA256.fullmatch(identity) is None
            or identity in identities
            or not isinstance(action_ids, list)
            or not action_ids
            or any(type(action_id) is not int for action_id in action_ids)
            or action_ids != sorted(action_ids)
            or len(action_ids) != len(set(action_ids))
        ):
            raise CalibrationAnalysisError(
                f"{name} partition group {group_index} changed"
            )
        identities.add(identity)
        covered_actions.extend(action_ids)
    if sorted(covered_actions) != list(range(ACTION_COUNT)):
        raise CalibrationAnalysisError(f"{name} partition does not cover nine actions")
    return unique_count


def _equivalence_unique_counts(value: object, *, state_id: str) -> dict[str, int]:
    expected = {
        "schema",
        "candidate_action_ids",
        *EQUIVALENCE_PARTITION_NAMES,
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CalibrationAnalysisError(
            f"state {state_id} candidate response audit field set changed"
        )
    if (
        value["schema"]
        != producer_contract.TEXTURED_V03_CANDIDATE_RESPONSE_AUDIT_SCHEMA
        or value["candidate_action_ids"] != list(range(ACTION_COUNT))
    ):
        raise CalibrationAnalysisError(
            f"state {state_id} candidate response audit identity changed"
        )
    return {
        partition: _validate_equivalence_partition(
            value[partition], name=f"state {state_id} {partition}"
        )
        for partition in EQUIVALENCE_PARTITION_NAMES
    }


def _equivalence_identity_by_action(
    value: object,
    *,
    partition: str,
    state_id: str,
) -> list[str]:
    if not isinstance(value, Mapping) or partition not in value:
        raise CalibrationAnalysisError(
            f"state {state_id} {partition} partition is absent"
        )
    partition_value = value[partition]
    _validate_equivalence_partition(
        partition_value, name=f"state {state_id} {partition}"
    )
    assert isinstance(partition_value, Mapping)
    groups = partition_value["groups"]
    assert isinstance(groups, list)
    identities: list[str | None] = [None] * ACTION_COUNT
    for group in groups:
        assert isinstance(group, Mapping)
        identity = str(group["identity_sha256"])
        for action_id in group["action_ids"]:
            identities[int(action_id)] = identity
    if any(identity is None for identity in identities):
        raise CalibrationAnalysisError(
            f"state {state_id} {partition} identity mapping is incomplete"
        )
    return [str(identity) for identity in identities]


def _eligible_action_ids_from_signatures(
    signatures: Sequence[Mapping[str, Any]],
) -> list[int]:
    if len(signatures) != ACTION_COUNT:
        raise CalibrationAnalysisError(
            "joint discrimination signature action count changed"
        )
    eligible: list[int] = []
    for action_id, query in enumerate(signatures):
        if query.get("action_id") != action_id:
            raise CalibrationAnalysisError(
                "joint discrimination signatures are not action ordered"
            )
        physical_alternatives = [
            alternative
            for alternative in signatures
            if alternative.get("action_id") != action_id
            and alternative.get("physical_outcome_class_key")
            != query.get("physical_outcome_class_key")
        ]
        if physical_alternatives and all(
            alternative.get("executed_tape_class_sha256")
            != query.get("executed_tape_class_sha256")
            and alternative.get("stored_rgb_pixel_class_sha256")
            != query.get("stored_rgb_pixel_class_sha256")
            for alternative in physical_alternatives
        ):
            eligible.append(action_id)
    return eligible


def _count_histogram(values: Sequence[int]) -> dict[str, int]:
    return {
        str(unique_count): sum(value == unique_count for value in values)
        for unique_count in range(1, ACTION_COUNT + 1)
    }


def _support_stratum_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    if not rows:
        raise CalibrationAnalysisError("candidate support stratum cannot be empty")
    equivalence: dict[str, object] = {}
    for partition in EQUIVALENCE_PARTITION_NAMES:
        values = [int(row["equivalence_unique_counts"][partition]) for row in rows]
        equivalence[partition] = {
            "unique_count_histogram": _count_histogram(values),
            "minimum_unique_count": min(values),
            "maximum_unique_count": max(values),
            "collapsed_state_count": sum(value < ACTION_COUNT for value in values),
        }
    rank_values = [int(row["dense_physical_rank_class_count"]) for row in rows]
    identifiable_ids = [
        str(row["state_id"]) for row in rows if row["identifiable"] is True
    ]
    unidentifiable_ids = [
        str(row["state_id"]) for row in rows if row["identifiable"] is False
    ]
    return {
        "state_count": len(rows),
        "equivalence_unique_count_distributions": equivalence,
        "dense_physical_rank_class_count_distribution": {
            "class_count_histogram": _count_histogram(rank_values),
            "minimum_class_count": min(rank_values),
            "maximum_class_count": max(rank_values),
        },
        "identifiability": {
            "identifiable_state_count": len(identifiable_ids),
            "unidentifiable_state_count": len(unidentifiable_ids),
            "identifiable_fraction": len(identifiable_ids) / len(rows),
            "identifiable_state_ids": identifiable_ids,
            "unidentifiable_state_ids": unidentifiable_ids,
        },
    }


def _support_analysis_from_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    families = sorted({str(row["family"]) for row in rows})
    histories = sorted({tuple(row["history_action_ids"]) for row in rows})
    return {
        "overall": _support_stratum_summary(rows),
        "per_family": {
            family: _support_stratum_summary(
                [row for row in rows if row["family"] == family]
            )
            for family in families
        },
        "per_history": [
            {
                "history_action_ids": list(history),
                "summary": _support_stratum_summary(
                    [
                        row
                        for row in rows
                        if tuple(row["history_action_ids"]) == history
                    ]
                ),
            }
            for history in histories
        ],
    }


def _calibrated_discrimination_query_coverage_from_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    families = tuple(producer_contract.FAMILIES)
    if len(rows) != CALIBRATION_STATE_COUNT:
        raise CalibrationAnalysisError(
            "calibrated discrimination-query coverage state count changed"
        )
    observed_families = {str(row["family"]) for row in rows}
    if observed_families != set(families):
        raise CalibrationAnalysisError(
            "calibrated discrimination-query coverage family panel changed"
        )

    def summary(
        selected: Sequence[Mapping[str, Any]], *, required_count: int
    ) -> dict[str, object]:
        state_count = len(selected)
        eligible_query_count = sum(
            int(row["eligible_action_count"]) for row in selected
        )
        total_query_count = state_count * ACTION_COUNT
        discrimination_query_coverage = eligible_query_count / total_query_count
        physical_outcome_class_count = sum(
            int(row["dense_physical_rank_class_count"]) for row in selected
        )
        return {
            "state_count": state_count,
            "eligible_query_count": eligible_query_count,
            "total_query_count": total_query_count,
            "discrimination_query_coverage": discrimination_query_coverage,
            "physical_outcome_class_count": physical_outcome_class_count,
            "maximum_physical_outcome_class_count": total_query_count,
            "physical_outcome_class_coverage": (
                physical_outcome_class_count / total_query_count
            ),
            "required_minimum_eligible_query_count": required_count,
            "required_minimum_discrimination_query_coverage": (
                required_count / total_query_count
            ),
            "passed": (
                eligible_query_count >= required_count
                and discrimination_query_coverage
                >= CALIBRATED_MINIMUM_DISCRIMINATION_QUERY_COVERAGE
            ),
        }

    per_family: dict[str, dict[str, object]] = {}
    for family in families:
        selected = [row for row in rows if row["family"] == family]
        if len(selected) != CALIBRATION_STATES_PER_FAMILY:
            raise CalibrationAnalysisError(
                f"calibrated discrimination-query coverage {family} state count changed"
            )
        per_family[family] = summary(
            selected, required_count=MIN_ELIGIBLE_QUERIES_PER_FAMILY
        )
    overall = summary(
        rows, required_count=MIN_ELIGIBLE_QUERIES_OVERALL
    )
    all_families_passed = all(
        value["passed"] is True for value in per_family.values()
    )
    return {
        "definition": {
            "query_eligibility_requires": {
                "at_least_one_physically_nonequivalent_alternative": True,
                "every_physically_nonequivalent_alternative_has_different_"
                "executed_tape_class": True,
                "every_physically_nonequivalent_alternative_has_different_"
                "stored_rgb_pixel_class": True,
                "both_observable_differences_required_for_every_physical_"
                "alternative": True,
            },
            "physical_oracle_class_count_source": (
                "dense_physical_rank_class_count"
            ),
            "physical_outcome_equivalence_tolerance_m": (
                TEXTURED_V03_OUTCOME_EQUIVALENCE_TOLERANCE_M
            ),
            "aggregate_partition_nontriviality_is_diagnostic_only": True,
            "physical_outcome_class_coverage_rule": (
                "sum(dense_physical_rank_class_count)/(9*state_count)"
            ),
            "physical_outcome_class_coverage_is_diagnostic_only": True,
        },
        "requirements": {
            "states_per_family": CALIBRATION_STATES_PER_FAMILY,
            "queries_per_state": ACTION_COUNT,
            "minimum_eligible_queries_per_family": (
                MIN_ELIGIBLE_QUERIES_PER_FAMILY
            ),
            "minimum_eligible_queries_overall": (
                MIN_ELIGIBLE_QUERIES_OVERALL
            ),
            "calibrated_minimum_discrimination_query_coverage": (
                CALIBRATED_MINIMUM_DISCRIMINATION_QUERY_COVERAGE
            ),
            "bounded_applicability_minimum_discrimination_query_coverage": (
                BOUNDED_MINIMUM_DISCRIMINATION_QUERY_COVERAGE
            ),
            "calibrated_discrimination_query_coverage_strictly_exceeds_"
            "bounded_discrimination_query_coverage": True,
        },
        "overall": overall,
        "per_family": per_family,
        "all_families_passed": all_families_passed,
        "passed": all_families_passed and overall["passed"] is True,
    }


def _strict_json_loads(raw: bytes, *, name: str) -> dict[str, object]:
    def reject_constant(token: str) -> None:
        raise CalibrationAnalysisError(f"{name} contains non-finite token {token}")

    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CalibrationAnalysisError(f"{name} contains duplicate key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CalibrationAnalysisError(f"{name} is not strict JSON") from error
    if not isinstance(value, dict):
        raise CalibrationAnalysisError(f"{name} must be a JSON object")
    return value


def validate_calibration_receipt_v1(
    value: object,
    *,
    verify_external_bindings: bool,
) -> dict[str, object]:
    """Validate the complete analyzer receipt and optionally re-open its sources."""

    if (
        isinstance(value, Mapping)
        and value.get("schema") == TEXTURED_V03_CALIBRATION_RECEIPT_SCHEMA
    ):
        return _validate_textured_v03_calibration_receipt(
            value,
            verify_external_bindings=verify_external_bindings,
        )

    expected = {
        "schema",
        "status",
        "citable_as_scientific_evidence",
        "authorizes_retry_or_resume",
        "calibration_id",
        "role",
        "train_eval_scenes_accessed",
        "decision",
        "calibration_collection_receipt",
        "calibration_contract",
        "repeatability_analysis",
        "physics_validation",
        "visual_validation",
        "resource_measurements",
        "analyzer_binding",
        "checker_binding",
        "source_bindings",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CalibrationAnalysisError("calibration receipt field set changed")
    if (
        value["schema"] != CALIBRATION_RECEIPT_SCHEMA
        or value["status"] != "COMPLETE"
        or value["citable_as_scientific_evidence"] is not False
        or value["authorizes_retry_or_resume"] is not False
        or not isinstance(value["calibration_id"], str)
        or not value["calibration_id"]
        or value["role"] != "calibration"
        or value["train_eval_scenes_accessed"] is not False
        or value["decision"] not in {
            "FREEZE_PILOT_CONTRACT",
            "STOP_SOURCE_REDESIGN",
        }
    ):
        raise CalibrationAnalysisError("calibration receipt status/identity changed")
    external = _validate_binding(
        value["calibration_collection_receipt"],
        name="calibration collection",
    )
    analyzer = _validate_binding(value["analyzer_binding"], name="analyzer")
    checker_source = _validate_binding(value["checker_binding"], name="checker")
    contract = value["calibration_contract"]
    contract_fields = {
        "schema",
        "excluded_scene_ids",
        "progress_tolerance_m",
        "path_length_tolerance_m",
        "quantization_rule",
        "lexicographic_key",
        "proxy_fields_excluded",
        "tolerance_derivation",
    }
    if not isinstance(contract, Mapping) or set(contract) != contract_fields:
        raise CalibrationAnalysisError("calibration contract field set changed")
    excluded = contract["excluded_scene_ids"]
    if (
        contract["schema"] != PHYSICAL_RANK_CONTRACT_SCHEMA
        or not isinstance(excluded, list)
        or len(excluded) != CALIBRATION_SCENE_COUNT
        or excluded != sorted(excluded)
        or len(set(excluded)) != len(excluded)
        or any(not isinstance(item, str) or not item for item in excluded)
        or _finite(
            contract["progress_tolerance_m"], name="progress tolerance"
        ) <= 0.0
        or _finite(
            contract["path_length_tolerance_m"], name="path-length tolerance"
        ) <= 0.0
        or contract["quantization_rule"]
        != "sign(x)*floor(abs(x)/t+0.5)"
        or contract["lexicographic_key"] != [
            "physical_fell_ascending",
            "physical_tipped_ascending",
            "physical_target_progress_quantized_descending",
            "physical_path_length_quantized_ascending",
        ]
        or contract["proxy_fields_excluded"] is not True
    ):
        raise CalibrationAnalysisError("calibration contract semantics changed")
    derivation = contract["tolerance_derivation"]
    if (
        not isinstance(derivation, Mapping)
        or set(derivation) != {
            "schema",
            "method",
            "minimum_numerical_resolution_m",
            "repeat_controls",
            "repeated_action_ids",
            "all_requested_primitives_covered",
            "deterministic_repeat_gate_passed",
            "empirical_noise_scale_estimated",
        }
        or derivation["schema"] != TOLERANCE_DERIVATION_SCHEMA
        or derivation["method"]
        != "fixed_numerical_floor_after_exact_deterministic_repeat_gate"
        or derivation["minimum_numerical_resolution_m"] != NUMERICAL_FLOOR_M
        or derivation["repeat_controls"] != CALIBRATION_STATE_COUNT
        or derivation["repeated_action_ids"]
        != [index % ACTION_COUNT for index in range(CALIBRATION_STATE_COUNT)]
        or derivation["all_requested_primitives_covered"] is not True
        or derivation["deterministic_repeat_gate_passed"] is not True
        or derivation["empirical_noise_scale_estimated"] is not False
    ):
        raise CalibrationAnalysisError("calibration tolerance derivation changed")
    if (
        contract["progress_tolerance_m"] != NUMERICAL_FLOOR_M
        or contract["path_length_tolerance_m"] != NUMERICAL_FLOOR_M
    ):
        raise CalibrationAnalysisError("calibration numerical floor changed")
    repeatability = value["repeatability_analysis"]
    physics = value["physics_validation"]
    visual = value["visual_validation"]
    resources = value["resource_measurements"]
    if (
        not isinstance(repeatability, Mapping)
        or repeatability.get("repeat_controls") != CALIBRATION_STATE_COUNT
        or repeatability.get("repeated_action_ids")
        != [index % ACTION_COUNT for index in range(CALIBRATION_STATE_COUNT)]
        or repeatability.get("all_requested_primitives_covered") is not True
        or repeatability.get("interpretation")
        != "deterministic_replay_gate_not_empirical_noise_estimate"
        or repeatability.get("empirical_noise_scale_estimated") is not False
        or repeatability.get("executed_command_tapes_exact") is not True
        or repeatability.get("physical_trajectories_exact") is not True
        or repeatability.get("stored_rgb_exact") is not True
        or not isinstance(physics, Mapping)
        or physics.get("receipt_checker_passed") is not True
        or physics.get("common_prefix_exact") is not True
        or physics.get("nine_unique_executed_tapes_per_state") is not True
        or physics.get("physics_validated_for_branch_outcomes") is not True
        or not isinstance(visual, Mapping)
        or visual.get("camera_quality_receipts_passed") is not True
        or visual.get("endpoint_pose_replay_bound") is not True
        or visual.get("visual_domain_fidelity_claimed") is not False
        or visual.get("eligible_for_physical_branch_evaluation") is not True
        or visual.get("eligible_for_visual_domain_parity_claim") is not False
    ):
        raise CalibrationAnalysisError("calibration validation evidence changed")
    if not isinstance(resources, Mapping) or set(resources) != {
        "schema",
        "stored_rgb_png",
        "low_information_strata",
        "stage_wall_seconds",
        "outcome_counts",
        "gpu_peak_memory_measurement_scope",
    }:
        raise CalibrationAnalysisError("calibration resource measurements changed")
    stored_rgb = resources["stored_rgb_png"]
    low_information = resources["low_information_strata"]
    stages = resources["stage_wall_seconds"]
    outcomes = resources["outcome_counts"]
    if (
        resources["schema"] != RESOURCE_MEASUREMENTS_SCHEMA
        or resources["gpu_peak_memory_measurement_scope"]
        != "external_terminal_required_not_observed_by_analyzer"
        or not isinstance(stored_rgb, Mapping)
        or set(stored_rgb) != {
            "context_frames",
            "context_bytes",
            "target_frames",
            "target_bytes",
            "total_frames",
            "total_bytes",
            "raw_uncompressed_rgb_ceiling_bytes",
        }
        or stored_rgb["context_frames"] != 48
        or stored_rgb["target_frames"] != 160
        or stored_rgb["total_frames"] != 208
        or stored_rgb["total_frames"]
        != stored_rgb["context_frames"] + stored_rgb["target_frames"]
        or any(
            type(stored_rgb[name]) is not int or stored_rgb[name] < 0
            for name in ("context_bytes", "target_bytes", "total_bytes")
        )
        or stored_rgb["total_bytes"]
        != stored_rgb["context_bytes"] + stored_rgb["target_bytes"]
        or stored_rgb["raw_uncompressed_rgb_ceiling_bytes"] != 208 * 224 * 224 * 3
        or not isinstance(low_information, Mapping)
        or set(low_information) != {
            "total_frames",
            "context_frames",
            "target_frames",
            "reason_counts",
            "context_reason_counts",
            "target_reason_counts",
            "frame_receipt_tags_present",
            "hard_invalid_frames",
        }
        or any(
            type(low_information[name]) is not int or low_information[name] < 0
            for name in ("total_frames", "context_frames", "target_frames")
        )
        or low_information["total_frames"]
        != low_information["context_frames"] + low_information["target_frames"]
        or low_information["context_frames"] > 48
        or low_information["target_frames"] > 160
        or low_information["frame_receipt_tags_present"] is not True
        or low_information["hard_invalid_frames"] != 0
        or any(
            not isinstance(low_information[name], Mapping)
            or set(low_information[name])
            != {"low_rgb_texture", "near_wall_depth", "near_forward_geometry"}
            or any(type(count) is not int or count < 0 for count in low_information[name].values())
            for name in (
                "reason_counts",
                "context_reason_counts",
                "target_reason_counts",
            )
        )
        or any(
            low_information["reason_counts"][reason]
            != low_information["context_reason_counts"][reason]
            + low_information["target_reason_counts"][reason]
            or low_information["context_reason_counts"][reason]
            > low_information["context_frames"]
            or low_information["target_reason_counts"][reason]
            > low_information["target_frames"]
            for reason in (
                "low_rgb_texture",
                "near_wall_depth",
                "near_forward_geometry",
            )
        )
        or not isinstance(stages, Mapping)
        or set(stages) != {
            "collection_external_wall_seconds",
            "physics_scene_build_wall_seconds",
            "render_scene_build_wall_seconds",
            "common_prefix_step_wall_seconds",
            "branch_step_wall_seconds",
            "native_render_wall_seconds",
            "camera_quality_resize_wall_seconds",
            "png_encode_write_hash_wall_seconds",
            "post_lockstep_receipt_wall_seconds",
            "summed_scene_total_wall_seconds",
        }
        or any(_finite(number, name=f"stage {name}") < 0.0 for name, number in stages.items())
        or not isinstance(outcomes, Mapping)
        or set(outcomes) != {
            "complete_all_nine_action_groups",
            "executed_tape_distinct_groups",
            "prebranch_exact_groups",
            "clipped_candidate_branches",
            "fallen_candidate_branches",
            "tipped_candidate_branches",
            "camera_invalid_frames",
            "incomplete_states",
        }
        or outcomes["complete_all_nine_action_groups"] != 16
        or outcomes["executed_tape_distinct_groups"] != 16
        or outcomes["prebranch_exact_groups"] != 16
        or outcomes["camera_invalid_frames"] != 0
        or outcomes["incomplete_states"] != 0
        or any(type(number) is not int or number < 0 for number in outcomes.values())
    ):
        raise CalibrationAnalysisError("calibration resource evidence is invalid")
    minimum_classes = physics.get("minimum_physical_rank_classes_per_state")
    expected_decision = (
        "FREEZE_PILOT_CONTRACT"
        if type(minimum_classes) is int and minimum_classes >= 2
        else "STOP_SOURCE_REDESIGN"
    )
    if value["decision"] != expected_decision:
        raise CalibrationAnalysisError("calibration decision disagrees with evidence")
    sources = value["source_bindings"]
    expected_names = [CHECKER_SOURCE_NAME, ANALYZER_SOURCE_NAME, JOINER_SOURCE_NAME]
    if (
        not isinstance(sources, list)
        or [entry.get("name") if isinstance(entry, Mapping) else None for entry in sources]
        != expected_names
    ):
        raise CalibrationAnalysisError("calibration source closure changed")
    normalized_sources = []
    for entry in sources:
        if not isinstance(entry, Mapping) or set(entry) != {"name", "binding"}:
            raise CalibrationAnalysisError("calibration source binding changed")
        normalized_sources.append({
            "name": entry["name"],
            "binding": _validate_binding(
                entry["binding"], name=f"source {entry['name']}"
            ),
        })
    if normalized_sources[0]["binding"] != checker_source or normalized_sources[1][
        "binding"
    ] != analyzer:
        raise CalibrationAnalysisError("calibration source aliases changed")
    if verify_external_bindings:
        for name, binding in (
            ("calibration collection", external),
            *(
                (f"source {entry['name']}", entry["binding"])
                for entry in normalized_sources
            ),
        ):
            if _binding(Path(str(binding["path"]))) != binding:
                raise CalibrationAnalysisError(f"{name} changed after calibration")
    return dict(value)


def _validate_textured_v03_calibration_receipt(
    value: Mapping[str, object],
    *,
    verify_external_bindings: bool,
) -> dict[str, object]:
    expected = {
        "schema",
        "status",
        "citable_as_scientific_evidence",
        "authorizes_retry_or_resume",
        "calibration_id",
        "role",
        "train_eval_scenes_accessed",
        "decision",
        "calibration_collection_receipt",
        "visual_domain_parity_prerequisites",
        "calibration_contract",
        "repeatability_analysis",
        "technical_integrity",
        "physics_validation",
        "visual_validation",
        "candidate_branch_support_analysis",
        "resource_measurements",
        "analyzer_binding",
        "checker_binding",
        "source_bindings",
    }
    if set(value) != expected:
        raise CalibrationAnalysisError(
            "textured-v03 calibration receipt field set changed"
        )
    if (
        value["schema"] != TEXTURED_V03_CALIBRATION_RECEIPT_SCHEMA
        or value["status"] != "COMPLETE"
        or value["citable_as_scientific_evidence"] is not False
        or value["authorizes_retry_or_resume"] is not False
        or not isinstance(value["calibration_id"], str)
        or not value["calibration_id"]
        or value["role"] != "calibration"
        or value["train_eval_scenes_accessed"] is not False
        or value["decision"] not in {
            "FREEZE_PILOT_CONTRACT",
            "STOP_INSUFFICIENT_JOINT_COUNTERFACTUAL_DISCRIMINATION_SUPPORT",
        }
    ):
        raise CalibrationAnalysisError(
            "textured-v03 calibration status/identity changed"
        )
    external = _validate_binding(
        value["calibration_collection_receipt"],
        name="textured-v03 calibration collection",
    )
    prerequisites = value["visual_domain_parity_prerequisites"]
    if not isinstance(prerequisites, Mapping) or set(prerequisites) != {
        "result_binding",
        "terminal_binding",
        "review_binding",
    }:
        raise CalibrationAnalysisError(
            "textured-v03 parity prerequisite fields changed"
        )
    normalized_prerequisites = {
        name: _validate_binding(
            prerequisites[name], name=f"textured-v03 parity {name}"
        )
        for name in ("result_binding", "terminal_binding", "review_binding")
    }
    analyzer = _validate_binding(value["analyzer_binding"], name="analyzer")
    checker_source = _validate_binding(value["checker_binding"], name="checker")

    contract = value["calibration_contract"]
    if not isinstance(contract, Mapping) or set(contract) != {
        "schema",
        "excluded_scene_ids",
        "progress_tolerance_m",
        "path_length_tolerance_m",
        "quantization_rule",
        "lexicographic_key",
        "proxy_fields_excluded",
        "tolerance_derivation",
    }:
        raise CalibrationAnalysisError(
            "textured-v03 calibration contract field set changed"
        )
    excluded = contract["excluded_scene_ids"]
    if (
        contract["schema"] != PHYSICAL_RANK_CONTRACT_SCHEMA
        or not isinstance(excluded, list)
        or len(excluded) != CALIBRATION_SCENE_COUNT
        or excluded != sorted(excluded)
        or len(set(excluded)) != len(excluded)
        or any(not isinstance(item, str) or not item for item in excluded)
        or contract["progress_tolerance_m"]
        != TEXTURED_V03_OUTCOME_EQUIVALENCE_TOLERANCE_M
        or contract["path_length_tolerance_m"]
        != TEXTURED_V03_OUTCOME_EQUIVALENCE_TOLERANCE_M
        or contract["quantization_rule"]
        != "sign(x)*floor(abs(x)/t+0.5)"
        or contract["lexicographic_key"] != [
            "physical_fell_ascending",
            "physical_tipped_ascending",
            "physical_target_progress_quantized_descending",
            "physical_path_length_quantized_ascending",
        ]
        or contract["proxy_fields_excluded"] is not True
    ):
        raise CalibrationAnalysisError(
            "textured-v03 calibration contract semantics changed"
        )
    derivation = contract["tolerance_derivation"]
    if not isinstance(derivation, Mapping) or set(derivation) != {
        "schema",
        "method",
        "repeatability_numerical_floor_m",
        "outcome_equivalence_tolerance_m",
        "outcome_equivalence_applies_to",
        "outcome_equivalence_quantization_caveat",
        "exact_repeat_gate_separate_from_outcome_equivalence",
        "repeat_controls",
        "repeated_action_ids",
        "all_requested_primitives_covered",
        "deterministic_repeat_gate_passed",
        "empirical_noise_scale_estimated",
    }:
        raise CalibrationAnalysisError(
            "textured-v03 tolerance derivation field set changed"
        )
    repeated_action_ids = derivation["repeated_action_ids"]
    all_primitives_covered = (
        isinstance(repeated_action_ids, list)
        and set(repeated_action_ids) == set(range(ACTION_COUNT))
    )
    if (
        derivation["schema"] != TOLERANCE_DERIVATION_V2_SCHEMA
        or derivation["method"]
        != "fixed_preregistered_outcome_equivalence_after_exact_repeat_gate"
        or derivation["repeatability_numerical_floor_m"] != NUMERICAL_FLOOR_M
        or derivation["outcome_equivalence_tolerance_m"]
        != TEXTURED_V03_OUTCOME_EQUIVALENCE_TOLERANCE_M
        or derivation["outcome_equivalence_applies_to"]
        != ["physical_target_progress_m", "physical_path_length_m"]
        or derivation["outcome_equivalence_quantization_caveat"]
        != (
            "1cm_rounding_bins_have_boundary_artifacts_and_are_not_"
            "pairwise_distance_le_1cm_equivalence"
        )
        or derivation[
            "exact_repeat_gate_separate_from_outcome_equivalence"
        ]
        is not True
        or derivation["repeat_controls"] != CALIBRATION_STATE_COUNT
        or not isinstance(repeated_action_ids, list)
        or len(repeated_action_ids) != CALIBRATION_STATE_COUNT
        or any(
            type(action_id) is not int or not 0 <= action_id < ACTION_COUNT
            for action_id in repeated_action_ids
        )
        or derivation["all_requested_primitives_covered"]
        is not all_primitives_covered
        or derivation["deterministic_repeat_gate_passed"] is not True
        or derivation["empirical_noise_scale_estimated"] is not False
    ):
        raise CalibrationAnalysisError(
            "textured-v03 tolerance derivation changed"
        )

    repeatability = value["repeatability_analysis"]
    if not isinstance(repeatability, Mapping) or set(repeatability) != {
        "repeat_controls",
        "repeated_action_ids",
        "all_requested_primitives_covered",
        "interpretation",
        "empirical_noise_scale_estimated",
        "progress_max_abs_delta_m",
        "path_length_max_abs_delta_m",
        "endpoint_position_max_abs_delta_m",
        "endpoint_quaternion_max_abs_delta",
        "executed_command_tapes_exact",
        "physical_trajectories_exact",
        "stored_rgb_exact",
    }:
        raise CalibrationAnalysisError(
            "textured-v03 repeatability analysis field set changed"
        )
    if (
        repeatability["repeat_controls"] != CALIBRATION_STATE_COUNT
        or repeatability["repeated_action_ids"] != repeated_action_ids
        or repeatability["all_requested_primitives_covered"]
        is not all_primitives_covered
        or repeatability["interpretation"]
        != "deterministic_replay_gate_not_empirical_noise_estimate"
        or repeatability["empirical_noise_scale_estimated"] is not False
        or any(
            repeatability[name] != 0.0
            for name in (
                "progress_max_abs_delta_m",
                "path_length_max_abs_delta_m",
                "endpoint_position_max_abs_delta_m",
                "endpoint_quaternion_max_abs_delta",
            )
        )
        or repeatability["executed_command_tapes_exact"] is not True
        or repeatability["physical_trajectories_exact"] is not True
        or repeatability["stored_rgb_exact"] is not True
    ):
        raise CalibrationAnalysisError(
            "textured-v03 repeatability evidence changed"
        )

    analysis = value["candidate_branch_support_analysis"]
    if not isinstance(analysis, Mapping) or set(analysis) != {
        "schema",
        "criterion",
        "partition_names",
        "state_measurements",
        "calibrated_discrimination_query_coverage",
        "overall",
        "per_family",
        "per_history",
    }:
        raise CalibrationAnalysisError(
            "textured-v03 candidate support analysis field set changed"
        )
    criterion = {
        "aggregate_partition_nontrivial_diagnostic": {
            "executed_tape_min_unique_count": 2,
            "dense_physical_rank_min_class_count": 2,
            "stored_rgb_pixels_min_unique_count": 2,
            "all_three_required": True,
            "gating": False,
        },
        "joint_query_eligibility": {
            "physical_nonequivalent_alternative_required": True,
            "all_physical_nonequivalent_alternatives_must_have_different_"
            "executed_tape_class": True,
            "all_physical_nonequivalent_alternatives_must_have_different_"
            "stored_rgb_pixel_class": True,
            "physical_outcome_equivalence_tolerance_m": (
                TEXTURED_V03_OUTCOME_EQUIVALENCE_TOLERANCE_M
            ),
            "gating": True,
        },
    }
    if (
        analysis["schema"] != CANDIDATE_BRANCH_SUPPORT_ANALYSIS_SCHEMA
        or analysis["criterion"] != criterion
        or analysis["partition_names"] != list(EQUIVALENCE_PARTITION_NAMES)
    ):
        raise CalibrationAnalysisError(
            "textured-v03 candidate support criterion changed"
        )
    raw_rows = analysis["state_measurements"]
    if (
        not isinstance(raw_rows, list)
        or len(raw_rows) != CALIBRATION_STATE_COUNT
    ):
        raise CalibrationAnalysisError(
            "textured-v03 candidate support state grid changed"
        )
    rows: list[dict[str, object]] = []
    state_ids: set[str] = set()
    for index, row in enumerate(raw_rows):
        if not isinstance(row, Mapping) or set(row) != {
            "state_id",
            "family",
            "history_action_ids",
            "equivalence_unique_counts",
            "dense_physical_rank_class_count",
            "identifiable",
            "joint_contrast_signatures_by_action",
            "eligible_action_ids",
            "eligible_action_count",
        }:
            raise CalibrationAnalysisError(
                f"textured-v03 support state row {index} changed"
            )
        state_id = row["state_id"]
        family = row["family"]
        history = row["history_action_ids"]
        counts = row["equivalence_unique_counts"]
        rank_count = row["dense_physical_rank_class_count"]
        signatures = row["joint_contrast_signatures_by_action"]
        eligible_action_ids = row["eligible_action_ids"]
        eligible_action_count = row["eligible_action_count"]
        if (
            not isinstance(state_id, str)
            or not state_id
            or state_id in state_ids
            or not isinstance(family, str)
            or not family
            or not isinstance(history, list)
            or not history
            or any(
                type(action_id) is not int or not 0 <= action_id < ACTION_COUNT
                for action_id in history
            )
            or not isinstance(counts, Mapping)
            or set(counts) != set(EQUIVALENCE_PARTITION_NAMES)
            or any(
                type(counts[name]) is not int
                or not 1 <= counts[name] <= ACTION_COUNT
                for name in EQUIVALENCE_PARTITION_NAMES
            )
            or type(rank_count) is not int
            or not 1 <= rank_count <= ACTION_COUNT
            or type(row["identifiable"]) is not bool
            or not isinstance(signatures, list)
            or len(signatures) != ACTION_COUNT
            or not isinstance(eligible_action_ids, list)
            or eligible_action_ids != sorted(eligible_action_ids)
            or len(eligible_action_ids) != len(set(eligible_action_ids))
            or any(
                type(action_id) is not int
                or not 0 <= action_id < ACTION_COUNT
                for action_id in eligible_action_ids
            )
            or type(eligible_action_count) is not int
            or eligible_action_count != len(eligible_action_ids)
        ):
            raise CalibrationAnalysisError(
                f"textured-v03 support state row {index} is invalid"
            )
        expected_identifiable = (
            counts["executed_tape"] >= 2
            and rank_count >= 2
            and counts["stored_rgb_pixels"] >= 2
        )
        if row["identifiable"] is not expected_identifiable:
            raise CalibrationAnalysisError(
                f"textured-v03 support state row {index} misclassifies identifiability"
            )
        normalized_signatures: list[dict[str, object]] = []
        for action_id, signature in enumerate(signatures):
            if (
                not isinstance(signature, Mapping)
                or set(signature) != {
                    "action_id",
                    "executed_tape_class_sha256",
                    "physical_outcome_class_key",
                    "stored_rgb_pixel_class_sha256",
                }
                or signature.get("action_id") != action_id
                or _SHA256.fullmatch(
                    str(signature.get("executed_tape_class_sha256") or "")
                )
                is None
                or _SHA256.fullmatch(
                    str(signature.get("stored_rgb_pixel_class_sha256") or "")
                )
                is None
                or not isinstance(
                    signature.get("physical_outcome_class_key"), list
                )
                or len(signature["physical_outcome_class_key"]) != 4
                or any(
                    type(component) is not int
                    for component in signature["physical_outcome_class_key"]
                )
            ):
                raise CalibrationAnalysisError(
                    f"textured-v03 support state row {index} joint signature changed"
                )
            normalized_signatures.append(dict(signature))
        recomputed_eligible_action_ids = _eligible_action_ids_from_signatures(
            normalized_signatures
        )
        if (
            len(
                {
                    signature["executed_tape_class_sha256"]
                    for signature in normalized_signatures
                }
            )
            != counts["executed_tape"]
            or len(
                {
                    signature["stored_rgb_pixel_class_sha256"]
                    for signature in normalized_signatures
                }
            )
            != counts["stored_rgb_pixels"]
        ):
            raise CalibrationAnalysisError(
                f"textured-v03 support state row {index} signature partition counts changed"
            )
        if (
            eligible_action_ids != recomputed_eligible_action_ids
            or eligible_action_count != len(recomputed_eligible_action_ids)
            or len(
                {
                    tuple(signature["physical_outcome_class_key"])
                    for signature in normalized_signatures
                }
            )
            != rank_count
        ):
            raise CalibrationAnalysisError(
                f"textured-v03 support state row {index} joint eligibility changed"
            )
        state_ids.add(state_id)
        rows.append(dict(row))
    recomputed_analysis = _support_analysis_from_rows(rows)
    if any(
        analysis[name] != recomputed_analysis[name]
        for name in ("overall", "per_family", "per_history")
    ):
        raise CalibrationAnalysisError(
            "textured-v03 support distributions disagree with state measurements"
        )
    recomputed_coverage = _calibrated_discrimination_query_coverage_from_rows(rows)
    if analysis["calibrated_discrimination_query_coverage"] != recomputed_coverage:
        raise CalibrationAnalysisError(
            "textured-v03 calibrated discrimination-query coverage disagrees with states"
        )
    identifiable_state_count = int(
        recomputed_analysis["overall"]["identifiability"][
            "identifiable_state_count"
        ]
    )

    integrity = value["technical_integrity"]
    if (
        not isinstance(integrity, Mapping)
        or set(integrity) != {
            "receipt_checker_passed",
            "candidate_response_audit_v2_validated",
            "sentinel_command_endpoint_and_rgb_exact",
            "hard_invalid_frames",
        }
        or integrity["receipt_checker_passed"] is not True
        or integrity["candidate_response_audit_v2_validated"] is not True
        or integrity["sentinel_command_endpoint_and_rgb_exact"] is not True
        or type(integrity["hard_invalid_frames"]) is not int
        or integrity["hard_invalid_frames"] < 0
    ):
        raise CalibrationAnalysisError(
            "textured-v03 technical integrity evidence changed"
        )
    physics = value["physics_validation"]
    if not isinstance(physics, Mapping) or set(physics) != {
        "receipt_checker_passed",
        "common_prefix_exact",
        "candidate_equivalence_measured_not_rejected",
        "minimum_physical_rank_classes_per_state",
        "maximum_physical_rank_classes_per_state",
        "identifiable_state_count",
        "clipped_candidate_branches",
        "physics_validated_for_branch_outcomes",
    }:
        raise CalibrationAnalysisError(
            "textured-v03 physics validation field set changed"
        )
    rank_values = [
        int(row["dense_physical_rank_class_count"]) for row in rows
    ]
    if (
        physics["receipt_checker_passed"] is not True
        or physics["common_prefix_exact"] is not True
        or physics["candidate_equivalence_measured_not_rejected"] is not True
        or physics["minimum_physical_rank_classes_per_state"] != min(rank_values)
        or physics["maximum_physical_rank_classes_per_state"] != max(rank_values)
        or physics["identifiable_state_count"] != identifiable_state_count
        or type(physics["clipped_candidate_branches"]) is not int
        or physics["clipped_candidate_branches"] < 0
        or physics["physics_validated_for_branch_outcomes"] is not True
    ):
        raise CalibrationAnalysisError(
            "textured-v03 physics validation evidence changed"
        )
    visual = value["visual_validation"]
    if not isinstance(visual, Mapping) or visual != {
        "camera_quality_receipts_passed": integrity["hard_invalid_frames"] == 0,
        "endpoint_pose_replay_bound": True,
        "textured_v03_render_contract_validated": True,
        "visual_domain_fidelity_claimed": False,
        "eligible_for_physical_branch_evaluation": True,
        "eligible_for_visual_domain_parity_claim": False,
    }:
        raise CalibrationAnalysisError(
            "textured-v03 visual validation evidence changed"
        )

    resources = value["resource_measurements"]
    if not isinstance(resources, Mapping) or set(resources) != {
        "schema",
        "stored_rgb_png",
        "low_information_strata",
        "stage_wall_seconds",
        "outcome_counts",
        "gpu_peak_memory_measurement_scope",
    }:
        raise CalibrationAnalysisError(
            "textured-v03 resource measurement field set changed"
        )
    stored_rgb = resources["stored_rgb_png"]
    low_information = resources["low_information_strata"]
    stages = resources["stage_wall_seconds"]
    outcomes = resources["outcome_counts"]
    if (
        resources["schema"] != TEXTURED_V03_RESOURCE_MEASUREMENTS_SCHEMA
        or resources["gpu_peak_memory_measurement_scope"]
        != "external_terminal_required_not_observed_by_analyzer"
        or not isinstance(stored_rgb, Mapping)
        or set(stored_rgb) != {
            "context_frames",
            "context_bytes",
            "target_frames",
            "target_bytes",
            "total_frames",
            "total_bytes",
            "raw_uncompressed_rgb_ceiling_bytes",
        }
        or stored_rgb["context_frames"] != 48
        or stored_rgb["target_frames"] != 160
        or stored_rgb["total_frames"] != 208
        or stored_rgb["total_frames"]
        != stored_rgb["context_frames"] + stored_rgb["target_frames"]
        or any(
            type(stored_rgb[name]) is not int or stored_rgb[name] < 0
            for name in ("context_bytes", "target_bytes", "total_bytes")
        )
        or stored_rgb["total_bytes"]
        != stored_rgb["context_bytes"] + stored_rgb["target_bytes"]
        or stored_rgb["raw_uncompressed_rgb_ceiling_bytes"]
        != 208 * 224 * 224 * 3
    ):
        raise CalibrationAnalysisError(
            "textured-v03 stored RGB resource evidence changed"
        )
    if (
        not isinstance(low_information, Mapping)
        or set(low_information) != {
            "total_frames",
            "context_frames",
            "target_frames",
            "reason_counts",
            "context_reason_counts",
            "target_reason_counts",
            "frame_receipt_tags_present",
            "hard_invalid_frames",
        }
        or any(
            type(low_information[name]) is not int or low_information[name] < 0
            for name in ("total_frames", "context_frames", "target_frames")
        )
        or low_information["total_frames"]
        != low_information["context_frames"] + low_information["target_frames"]
        or low_information["context_frames"] > 48
        or low_information["target_frames"] > 160
        or low_information["frame_receipt_tags_present"] is not True
        or low_information["hard_invalid_frames"]
        != integrity["hard_invalid_frames"]
        or any(
            not isinstance(low_information[name], Mapping)
            or set(low_information[name])
            != set(TEXTURED_V03_LOW_INFO_REASON_NAMES)
            or any(
                type(count) is not int or count < 0
                for count in low_information[name].values()
            )
            for name in (
                "reason_counts",
                "context_reason_counts",
                "target_reason_counts",
            )
        )
        or any(
            low_information["reason_counts"][reason]
            != low_information["context_reason_counts"][reason]
            + low_information["target_reason_counts"][reason]
            or low_information["context_reason_counts"][reason]
            > low_information["context_frames"]
            or low_information["target_reason_counts"][reason]
            > low_information["target_frames"]
            for reason in TEXTURED_V03_LOW_INFO_REASON_NAMES
        )
    ):
        raise CalibrationAnalysisError(
            "textured-v03 low-information resource evidence changed"
        )
    expected_stage_fields = {
        "collection_external_wall_seconds",
        "physics_scene_build_wall_seconds",
        "render_scene_build_wall_seconds",
        "common_prefix_step_wall_seconds",
        "branch_step_wall_seconds",
        "native_render_wall_seconds",
        "camera_quality_resize_wall_seconds",
        "png_encode_write_hash_wall_seconds",
        "post_lockstep_receipt_wall_seconds",
        "summed_scene_total_wall_seconds",
    }
    expected_outcome_fields = {
        "complete_all_nine_action_groups",
        "candidate_response_audited_groups",
        "prebranch_exact_groups",
        "identifiable_groups",
        "clipped_candidate_branches",
        "fallen_candidate_branches",
        "tipped_candidate_branches",
        "camera_invalid_frames",
        "incomplete_states",
    }
    if (
        not isinstance(stages, Mapping)
        or set(stages) != expected_stage_fields
        or any(
            _finite(number, name=f"stage {name}") < 0.0
            for name, number in stages.items()
        )
        or not isinstance(outcomes, Mapping)
        or set(outcomes) != expected_outcome_fields
        or any(type(number) is not int or number < 0 for number in outcomes.values())
        or outcomes["complete_all_nine_action_groups"]
        != CALIBRATION_STATE_COUNT
        or outcomes["candidate_response_audited_groups"]
        != CALIBRATION_STATE_COUNT
        or outcomes["prebranch_exact_groups"] != CALIBRATION_STATE_COUNT
        or outcomes["identifiable_groups"] != identifiable_state_count
        or outcomes["clipped_candidate_branches"]
        != physics["clipped_candidate_branches"]
        or outcomes["camera_invalid_frames"]
        != integrity["hard_invalid_frames"]
        or outcomes["incomplete_states"] != 0
    ):
        raise CalibrationAnalysisError(
            "textured-v03 stage/outcome resource evidence changed"
        )
    expected_decision = (
        "FREEZE_PILOT_CONTRACT"
        if integrity["receipt_checker_passed"] is True
        and integrity["candidate_response_audit_v2_validated"] is True
        and integrity["sentinel_command_endpoint_and_rgb_exact"] is True
        and low_information["hard_invalid_frames"] == 0
        and recomputed_coverage["passed"] is True
        else "STOP_INSUFFICIENT_JOINT_COUNTERFACTUAL_DISCRIMINATION_SUPPORT"
    )
    if value["decision"] != expected_decision:
        raise CalibrationAnalysisError(
            "textured-v03 calibration decision disagrees with evidence"
        )

    sources = value["source_bindings"]
    expected_names = [CHECKER_SOURCE_NAME, ANALYZER_SOURCE_NAME, JOINER_SOURCE_NAME]
    if (
        not isinstance(sources, list)
        or [
            entry.get("name") if isinstance(entry, Mapping) else None
            for entry in sources
        ]
        != expected_names
    ):
        raise CalibrationAnalysisError(
            "textured-v03 calibration source closure changed"
        )
    normalized_sources = []
    for entry in sources:
        if not isinstance(entry, Mapping) or set(entry) != {"name", "binding"}:
            raise CalibrationAnalysisError(
                "textured-v03 calibration source binding changed"
            )
        normalized_sources.append({
            "name": entry["name"],
            "binding": _validate_binding(
                entry["binding"], name=f"source {entry['name']}"
            ),
        })
    if (
        normalized_sources[0]["binding"] != checker_source
        or normalized_sources[1]["binding"] != analyzer
    ):
        raise CalibrationAnalysisError(
            "textured-v03 calibration source aliases changed"
        )
    if verify_external_bindings:
        try:
            validated_prerequisites = (
                producer_contract.validate_textured_v03_parity_prerequisites(
                    **normalized_prerequisites
                )
            )
        except producer_contract.PilotContractError as exc:
            raise CalibrationAnalysisError(str(exc)) from exc
        if validated_prerequisites != normalized_prerequisites:
            raise CalibrationAnalysisError(
                "textured-v03 parity prerequisite bindings changed"
            )
        for name, binding in (
            ("textured-v03 calibration collection", external),
            *(
                (f"source {entry['name']}", entry["binding"])
                for entry in normalized_sources
            ),
        ):
            if _binding(Path(str(binding["path"]))) != binding:
                raise CalibrationAnalysisError(
                    f"{name} changed after textured-v03 calibration"
                )
    return dict(value)


def load_bound_calibration_receipt_v1(
    path: Path,
    *,
    expected_sha256: str,
    expected_byte_count: int,
) -> tuple[dict[str, object], dict[str, object], bytes]:
    if _SHA256.fullmatch(expected_sha256) is None or expected_byte_count <= 0:
        raise CalibrationAnalysisError("caller calibration binding is malformed")
    selected = Path(path)
    try:
        resolved = selected.resolve(strict=True)
    except OSError as exc:
        raise CalibrationAnalysisError(
            "cannot safely open caller calibration receipt"
        ) from exc
    if not selected.is_absolute() or selected != resolved:
        raise CalibrationAnalysisError(
            "caller calibration receipt path is not canonical and symlink-free"
        )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            selected,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CalibrationAnalysisError(
                "caller calibration receipt is not a regular file"
            )
        if before.st_size != expected_byte_count:
            raise CalibrationAnalysisError("caller calibration binding changed")
        chunks: list[bytes] = []
        total_read = 0
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            total_read += len(chunk)
            if total_read > expected_byte_count:
                raise CalibrationAnalysisError(
                    "caller calibration binding changed"
                )
        after = os.fstat(descriptor)
    except OSError as exc:
        raise CalibrationAnalysisError(
            "cannot safely open caller calibration receipt"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    raw = b"".join(chunks)
    stable_identity = lambda value: (  # noqa: E731
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if (
        stable_identity(before) != stable_identity(after)
        or len(raw) != expected_byte_count
        or hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise CalibrationAnalysisError("caller calibration binding changed")
    actual = {
        "path": str(resolved),
        "file_sha256": expected_sha256,
        "byte_count": expected_byte_count,
    }
    receipt = validate_calibration_receipt_v1(
        _strict_json_loads(raw, name="calibration receipt"),
        verify_external_bindings=True,
    )
    return receipt, actual, raw


def _quantize(value: float, tolerance: float) -> int:
    magnitude = math.floor(abs(value) / tolerance + 0.5)
    return magnitude if value >= 0.0 else -magnitude


def _physical_key(branch: Mapping[str, Any], tolerances: Mapping[str, float]) -> tuple[int, ...]:
    return (
        int(bool(branch["physical_fell"])),
        int(bool(branch["physical_tipped"])),
        -_quantize(
            _finite(branch["physical_target_progress_m"], name="target progress"),
            tolerances["progress_tolerance_m"],
        ),
        _quantize(
            _finite(branch["physical_path_length_m"], name="path length"),
            tolerances["path_length_tolerance_m"],
        ),
    )


def _derive_textured_v03_calibration_receipt(
    collection: Mapping[str, Any],
    *,
    collection_binding: Mapping[str, object],
    analyzer_binding: Mapping[str, object],
    checker_binding: Mapping[str, object],
    joiner_binding: Mapping[str, object],
) -> dict[str, object]:
    normalized_collection_binding = _validate_binding(
        collection_binding, name="textured-v03 calibration collection"
    )
    normalized_analyzer_binding = _validate_binding(
        analyzer_binding, name="calibration analyzer"
    )
    normalized_checker_binding = _validate_binding(
        checker_binding, name="receipt checker"
    )
    normalized_joiner_binding = _validate_binding(
        joiner_binding, name="pilot joiner"
    )
    if _calibration_variant(collection) != "textured_v03_v3":
        raise CalibrationAnalysisError(
            "textured-v03 derivation requires its exact validated contract"
        )
    counts = collection.get("counts")
    if not isinstance(counts, Mapping) or (
        counts.get("scenes") != CALIBRATION_SCENE_COUNT
        or counts.get("states") != CALIBRATION_STATE_COUNT
        or counts.get("total_branches") != CALIBRATION_BRANCH_COUNT
        or counts.get("candidate_branches")
        != CALIBRATION_STATE_COUNT * ACTION_COUNT
        or counts.get("sentinel_branches") != CALIBRATION_STATE_COUNT
        or counts.get("roles") != {"calibration": CALIBRATION_STATE_COUNT}
    ):
        raise CalibrationAnalysisError(
            "textured-v03 calibration collection size contract changed"
        )
    states = collection.get("states")
    plan = collection.get("plan")
    if not isinstance(plan, Mapping):
        raise CalibrationAnalysisError(
            "textured-v03 calibration plan wrapper changed"
        )
    plan_document = plan.get("document")
    if not isinstance(plan_document, Mapping):
        raise CalibrationAnalysisError(
            "textured-v03 calibration plan document changed"
        )
    parity_prerequisites = {
        "result_binding": _validate_binding(
            plan_document.get("visual_domain_parity_result_binding"),
            name="calibration plan parity result",
        ),
        "terminal_binding": _validate_binding(
            plan_document.get("visual_domain_parity_terminal_binding"),
            name="calibration plan parity terminal",
        ),
        "review_binding": _validate_binding(
            plan_document.get("visual_domain_parity_review_binding"),
            name="calibration plan parity review",
        ),
    }
    plan_states = plan.get("states")
    assert isinstance(states, Sequence) and not isinstance(states, (str, bytes))
    assert isinstance(plan_states, Sequence) and not isinstance(
        plan_states, (str, bytes)
    )

    progress_repeat_deltas: list[float] = []
    path_repeat_deltas: list[float] = []
    endpoint_position_repeat_deltas: list[float] = []
    endpoint_quaternion_repeat_deltas: list[float] = []
    calibration_scene_ids: set[str] = set()
    repeated_action_ids: list[int] = []
    clipped_candidate_count = 0
    support_rows: list[dict[str, object]] = []
    tolerances = {
        "progress_tolerance_m": TEXTURED_V03_OUTCOME_EQUIVALENCE_TOLERANCE_M,
        "path_length_tolerance_m": TEXTURED_V03_OUTCOME_EQUIVALENCE_TOLERANCE_M,
    }
    for state_index, (state, plan_state) in enumerate(
        zip(states, plan_states, strict=True)
    ):
        if not isinstance(state, Mapping) or not isinstance(plan_state, Mapping):
            raise CalibrationAnalysisError(
                f"textured-v03 calibration state {state_index} is malformed"
            )
        state_document = state["state"]
        receipt_document = state["document"]
        branches = state.get("branches")
        if (
            not isinstance(state_document, Mapping)
            or not isinstance(receipt_document, Mapping)
            or not isinstance(branches, Sequence)
            or isinstance(branches, (str, bytes))
            or len(branches) != ACTION_COUNT + 1
        ):
            raise CalibrationAnalysisError(
                "textured-v03 calibration branch grid changed"
            )
        state_id = str(state_document.get("state_id"))
        family = str(state_document.get("family"))
        history = plan_state.get("history_action_ids")
        if (
            not state_id
            or not family
            or not isinstance(history, list)
            or not history
        ):
            raise CalibrationAnalysisError(
                "textured-v03 calibration state identity is incomplete"
            )
        calibration_scene_ids.add(str(state_document.get("scene_id")))
        sentinel = branches[-1]
        if not isinstance(sentinel, Mapping):
            raise CalibrationAnalysisError(
                "textured-v03 calibration repeat branch is malformed"
            )
        duplicate_action_id = sentinel.get("action_id")
        if (
            type(duplicate_action_id) is not int
            or not 0 <= duplicate_action_id < ACTION_COUNT
        ):
            raise CalibrationAnalysisError(
                "textured-v03 calibration repeat action ID is invalid"
            )
        repeated_action_ids.append(duplicate_action_id)
        candidate = branches[duplicate_action_id]
        if not isinstance(candidate, Mapping):
            raise CalibrationAnalysisError(
                "textured-v03 repeated candidate is malformed"
            )
        progress_repeat_deltas.append(abs(
            _finite(candidate["physical_target_progress_m"], name="candidate progress")
            - _finite(sentinel["physical_target_progress_m"], name="sentinel progress")
        ))
        path_repeat_deltas.append(abs(
            _finite(candidate["physical_path_length_m"], name="candidate path length")
            - _finite(sentinel["physical_path_length_m"], name="sentinel path length")
        ))
        candidate_endpoint = candidate.get("endpoint_state")
        sentinel_endpoint = sentinel.get("endpoint_state")
        if not isinstance(candidate_endpoint, Mapping) or not isinstance(
            sentinel_endpoint, Mapping
        ):
            raise CalibrationAnalysisError(
                "textured-v03 calibration endpoint state is absent"
            )
        for field, output in (
            ("base_pos_world", endpoint_position_repeat_deltas),
            ("base_quat_wxyz", endpoint_quaternion_repeat_deltas),
        ):
            left = candidate_endpoint.get(field)
            right = sentinel_endpoint.get(field)
            if (
                not isinstance(left, Sequence)
                or isinstance(left, (str, bytes))
                or not isinstance(right, Sequence)
                or isinstance(right, (str, bytes))
                or len(left) != len(right)
            ):
                raise CalibrationAnalysisError(
                    f"textured-v03 repeat endpoint {field} changed shape"
                )
            output.append(max(
                abs(_finite(a, name=field) - _finite(b, name=field))
                for a, b in zip(left, right, strict=True)
            ))
        sentinel_audit = receipt_document.get("sentinel_audit")
        render_sentinel_audit = receipt_document.get("render_sentinel_audit")
        if (
            not isinstance(sentinel_audit, Mapping)
            or sentinel_audit.get("passed") is not True
            or not isinstance(render_sentinel_audit, Mapping)
            or render_sentinel_audit.get("passed") is not True
            or render_sentinel_audit.get("stored_rgb_equal") is not True
        ):
            raise CalibrationAnalysisError(
                "textured-v03 sentinel technical integrity changed"
            )
        candidates = branches[:ACTION_COUNT]
        physical_keys = [_physical_key(branch, tolerances) for branch in candidates]
        class_count = len(set(physical_keys))
        candidate_response_audit = receipt_document.get(
            "candidate_response_audit"
        )
        equivalence_counts = _equivalence_unique_counts(
            candidate_response_audit,
            state_id=state_id,
        )
        executed_tape_classes = _equivalence_identity_by_action(
            candidate_response_audit,
            partition="executed_tape",
            state_id=state_id,
        )
        stored_rgb_pixel_classes = _equivalence_identity_by_action(
            candidate_response_audit,
            partition="stored_rgb_pixels",
            state_id=state_id,
        )
        signatures = [
            {
                "action_id": action_id,
                "executed_tape_class_sha256": executed_tape_classes[action_id],
                "physical_outcome_class_key": list(physical_keys[action_id]),
                "stored_rgb_pixel_class_sha256": (
                    stored_rgb_pixel_classes[action_id]
                ),
            }
            for action_id in range(ACTION_COUNT)
        ]
        eligible_action_ids = _eligible_action_ids_from_signatures(signatures)
        identifiable = (
            equivalence_counts["executed_tape"] >= 2
            and class_count >= 2
            and equivalence_counts["stored_rgb_pixels"] >= 2
        )
        support_rows.append({
            "state_id": state_id,
            "family": family,
            "history_action_ids": list(history),
            "equivalence_unique_counts": equivalence_counts,
            "dense_physical_rank_class_count": class_count,
            "identifiable": identifiable,
            "joint_contrast_signatures_by_action": signatures,
            "eligible_action_ids": eligible_action_ids,
            "eligible_action_count": len(eligible_action_ids),
        })
        clipped_candidate_count += sum(
            int(branch.get("clipped") is True) for branch in candidates
        )

    if len(calibration_scene_ids) != CALIBRATION_SCENE_COUNT:
        raise CalibrationAnalysisError(
            "textured-v03 calibration scene identities repeat"
        )
    max_progress_repeat = max(progress_repeat_deltas)
    max_path_repeat = max(path_repeat_deltas)
    max_endpoint_position_repeat = max(endpoint_position_repeat_deltas)
    max_endpoint_quaternion_repeat = max(endpoint_quaternion_repeat_deltas)
    if any(
        value != 0.0
        for value in (
            max_progress_repeat,
            max_path_repeat,
            max_endpoint_position_repeat,
            max_endpoint_quaternion_repeat,
        )
    ):
        raise CalibrationAnalysisError(
            "textured-v03 repeat controls must be exactly deterministic"
        )
    all_primitives_covered = set(repeated_action_ids) == set(range(ACTION_COUNT))
    support_distributions = _support_analysis_from_rows(support_rows)
    calibrated_discrimination_query_coverage = (
        _calibrated_discrimination_query_coverage_from_rows(support_rows)
    )
    support_analysis = {
        "schema": CANDIDATE_BRANCH_SUPPORT_ANALYSIS_SCHEMA,
        "criterion": {
            "aggregate_partition_nontrivial_diagnostic": {
                "executed_tape_min_unique_count": 2,
                "dense_physical_rank_min_class_count": 2,
                "stored_rgb_pixels_min_unique_count": 2,
                "all_three_required": True,
                "gating": False,
            },
            "joint_query_eligibility": {
                "physical_nonequivalent_alternative_required": True,
                "all_physical_nonequivalent_alternatives_must_have_different_"
                "executed_tape_class": True,
                "all_physical_nonequivalent_alternatives_must_have_different_"
                "stored_rgb_pixel_class": True,
                "physical_outcome_equivalence_tolerance_m": (
                    TEXTURED_V03_OUTCOME_EQUIVALENCE_TOLERANCE_M
                ),
                "gating": True,
            },
        },
        "partition_names": list(EQUIVALENCE_PARTITION_NAMES),
        "state_measurements": support_rows,
        "calibrated_discrimination_query_coverage": (
            calibrated_discrimination_query_coverage
        ),
        **support_distributions,
    }
    identifiable_state_count = sum(
        row["identifiable"] is True for row in support_rows
    )

    frame_receipts = collection.get("frame_receipts")
    if not isinstance(frame_receipts, Mapping) or len(frame_receipts) != 208:
        raise CalibrationAnalysisError(
            "textured-v03 calibration frame receipt grid changed"
        )
    context_bytes = 0
    target_bytes = 0
    context_frames = 0
    target_frames = 0
    low_info_context_frames = 0
    low_info_target_frames = 0
    hard_invalid_frames = 0
    low_info_reason_counts = dict.fromkeys(TEXTURED_V03_LOW_INFO_REASON_NAMES, 0)
    low_info_context_reason_counts = dict.fromkeys(
        TEXTURED_V03_LOW_INFO_REASON_NAMES, 0
    )
    low_info_target_reason_counts = dict.fromkeys(
        TEXTURED_V03_LOW_INFO_REASON_NAMES, 0
    )
    for frame in frame_receipts.values():
        if not isinstance(frame, Mapping):
            raise CalibrationAnalysisError(
                "textured-v03 calibration frame receipt is malformed"
            )
        identity = frame.get("frame_identity")
        byte_count = frame.get("byte_count")
        low_info_reasons = frame.get("low_info_reasons")
        camera_valid = frame.get("camera_valid")
        pixel_sha256 = frame.get("pixel_sha256")
        if (
            not isinstance(identity, str)
            or type(byte_count) is not int
            or byte_count <= 0
            or type(frame.get("low_information")) is not bool
            or type(camera_valid) is not bool
            or not isinstance(pixel_sha256, str)
            or _SHA256.fullmatch(pixel_sha256) is None
            or not isinstance(low_info_reasons, list)
            or any(
                reason not in low_info_reason_counts
                for reason in low_info_reasons
            )
            or len(low_info_reasons) != len(set(low_info_reasons))
            or frame["low_information"] != bool(low_info_reasons)
        ):
            raise CalibrationAnalysisError(
                "textured-v03 calibration frame byte receipt changed"
            )
        hard_invalid_frames += int(camera_valid is False)
        if ":context:" in identity:
            context_frames += 1
            context_bytes += byte_count
            if low_info_reasons:
                low_info_context_frames += 1
            selected_reason_counts = low_info_context_reason_counts
        else:
            target_frames += 1
            target_bytes += byte_count
            if low_info_reasons:
                low_info_target_frames += 1
            selected_reason_counts = low_info_target_reason_counts
        for reason in low_info_reasons:
            low_info_reason_counts[reason] += 1
            selected_reason_counts[reason] += 1
    if context_frames != 48 or target_frames != 160:
        raise CalibrationAnalysisError(
            "textured-v03 calibration context/target frame split changed"
        )

    collection_document = collection.get("document")
    if not isinstance(collection_document, Mapping):
        raise CalibrationAnalysisError(
            "textured-v03 calibration collection document is absent"
        )
    scene_metrics = collection_document.get("scene_metrics")
    if not isinstance(scene_metrics, list) or len(scene_metrics) != 8:
        raise CalibrationAnalysisError(
            "textured-v03 calibration scene timing panel changed"
        )
    timing_fields = (
        "physics_build_wall_seconds",
        "render_scene_build_wall_seconds",
        "common_prefix_step_wall_seconds",
        "branch_step_wall_seconds",
        "native_render_wall_seconds",
        "camera_quality_resize_wall_seconds",
        "png_encode_write_hash_wall_seconds",
        "post_lockstep_receipt_wall_seconds",
        "scene_total_wall_seconds",
    )
    timing_sums = {name: 0.0 for name in timing_fields}
    for scene_metric in scene_metrics:
        if not isinstance(scene_metric, Mapping):
            raise CalibrationAnalysisError(
                "textured-v03 calibration scene timing row changed"
            )
        for name in timing_fields:
            timing_sums[name] += _finite(
                scene_metric.get(name), name=f"scene timing {name}"
            )
    collection_wall_seconds = _finite(
        collection_document.get("collection_wall_seconds"),
        name="collection wall seconds",
    )
    if collection_wall_seconds < 0.0:
        raise CalibrationAnalysisError("collection wall time cannot be negative")

    low_information_strata = {
        "total_frames": low_info_context_frames + low_info_target_frames,
        "context_frames": low_info_context_frames,
        "target_frames": low_info_target_frames,
        "reason_counts": low_info_reason_counts,
        "context_reason_counts": low_info_context_reason_counts,
        "target_reason_counts": low_info_target_reason_counts,
        "frame_receipt_tags_present": True,
        "hard_invalid_frames": hard_invalid_frames,
    }
    resource_measurements = {
        "schema": TEXTURED_V03_RESOURCE_MEASUREMENTS_SCHEMA,
        "stored_rgb_png": {
            "context_frames": context_frames,
            "context_bytes": context_bytes,
            "target_frames": target_frames,
            "target_bytes": target_bytes,
            "total_frames": context_frames + target_frames,
            "total_bytes": context_bytes + target_bytes,
            "raw_uncompressed_rgb_ceiling_bytes": 208 * 224 * 224 * 3,
        },
        "low_information_strata": low_information_strata,
        "stage_wall_seconds": {
            "collection_external_wall_seconds": collection_wall_seconds,
            "physics_scene_build_wall_seconds": timing_sums[
                "physics_build_wall_seconds"
            ],
            "render_scene_build_wall_seconds": timing_sums[
                "render_scene_build_wall_seconds"
            ],
            "common_prefix_step_wall_seconds": timing_sums[
                "common_prefix_step_wall_seconds"
            ],
            "branch_step_wall_seconds": timing_sums["branch_step_wall_seconds"],
            "native_render_wall_seconds": timing_sums[
                "native_render_wall_seconds"
            ],
            "camera_quality_resize_wall_seconds": timing_sums[
                "camera_quality_resize_wall_seconds"
            ],
            "png_encode_write_hash_wall_seconds": timing_sums[
                "png_encode_write_hash_wall_seconds"
            ],
            "post_lockstep_receipt_wall_seconds": timing_sums[
                "post_lockstep_receipt_wall_seconds"
            ],
            "summed_scene_total_wall_seconds": timing_sums[
                "scene_total_wall_seconds"
            ],
        },
        "outcome_counts": {
            "complete_all_nine_action_groups": CALIBRATION_STATE_COUNT,
            "candidate_response_audited_groups": CALIBRATION_STATE_COUNT,
            "prebranch_exact_groups": CALIBRATION_STATE_COUNT,
            "identifiable_groups": identifiable_state_count,
            "clipped_candidate_branches": clipped_candidate_count,
            "fallen_candidate_branches": sum(
                int(branch["physical_fell"] is True)
                for state in states
                for branch in state["branches"][:ACTION_COUNT]
            ),
            "tipped_candidate_branches": sum(
                int(branch["physical_tipped"] is True)
                for state in states
                for branch in state["branches"][:ACTION_COUNT]
            ),
            "camera_invalid_frames": hard_invalid_frames,
            "incomplete_states": 0,
        },
        "gpu_peak_memory_measurement_scope": (
            "external_terminal_required_not_observed_by_analyzer"
        ),
    }
    calibration_contract = {
        "schema": PHYSICAL_RANK_CONTRACT_SCHEMA,
        "excluded_scene_ids": sorted(calibration_scene_ids),
        **tolerances,
        "quantization_rule": "sign(x)*floor(abs(x)/t+0.5)",
        "lexicographic_key": [
            "physical_fell_ascending",
            "physical_tipped_ascending",
            "physical_target_progress_quantized_descending",
            "physical_path_length_quantized_ascending",
        ],
        "proxy_fields_excluded": True,
        "tolerance_derivation": {
            "schema": TOLERANCE_DERIVATION_V2_SCHEMA,
            "method": (
                "fixed_preregistered_outcome_equivalence_after_exact_repeat_gate"
            ),
            "repeatability_numerical_floor_m": NUMERICAL_FLOOR_M,
            "outcome_equivalence_tolerance_m": (
                TEXTURED_V03_OUTCOME_EQUIVALENCE_TOLERANCE_M
            ),
            "outcome_equivalence_applies_to": [
                "physical_target_progress_m",
                "physical_path_length_m",
            ],
            "outcome_equivalence_quantization_caveat": (
                "1cm_rounding_bins_have_boundary_artifacts_and_are_not_"
                "pairwise_distance_le_1cm_equivalence"
            ),
            "exact_repeat_gate_separate_from_outcome_equivalence": True,
            "repeat_controls": CALIBRATION_STATE_COUNT,
            "repeated_action_ids": repeated_action_ids,
            "all_requested_primitives_covered": all_primitives_covered,
            "deterministic_repeat_gate_passed": True,
            "empirical_noise_scale_estimated": False,
        },
    }
    technical_integrity = {
        "receipt_checker_passed": True,
        "candidate_response_audit_v2_validated": True,
        "sentinel_command_endpoint_and_rgb_exact": True,
        "hard_invalid_frames": hard_invalid_frames,
    }
    decision = (
        "FREEZE_PILOT_CONTRACT"
        if technical_integrity["receipt_checker_passed"] is True
        and technical_integrity["candidate_response_audit_v2_validated"] is True
        and technical_integrity["sentinel_command_endpoint_and_rgb_exact"] is True
        and hard_invalid_frames == 0
        and calibrated_discrimination_query_coverage["passed"] is True
        else "STOP_INSUFFICIENT_JOINT_COUNTERFACTUAL_DISCRIMINATION_SUPPORT"
    )
    rank_counts = [
        int(row["dense_physical_rank_class_count"]) for row in support_rows
    ]
    return {
        "schema": TEXTURED_V03_CALIBRATION_RECEIPT_SCHEMA,
        "status": "COMPLETE",
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "calibration_id": str(collection_document["attempt_id"]),
        "role": "calibration",
        "train_eval_scenes_accessed": False,
        "decision": decision,
        "calibration_collection_receipt": normalized_collection_binding,
        "visual_domain_parity_prerequisites": parity_prerequisites,
        "calibration_contract": calibration_contract,
        "repeatability_analysis": {
            "repeat_controls": CALIBRATION_STATE_COUNT,
            "repeated_action_ids": repeated_action_ids,
            "all_requested_primitives_covered": all_primitives_covered,
            "interpretation": (
                "deterministic_replay_gate_not_empirical_noise_estimate"
            ),
            "empirical_noise_scale_estimated": False,
            "progress_max_abs_delta_m": max_progress_repeat,
            "path_length_max_abs_delta_m": max_path_repeat,
            "endpoint_position_max_abs_delta_m": max_endpoint_position_repeat,
            "endpoint_quaternion_max_abs_delta": max_endpoint_quaternion_repeat,
            "executed_command_tapes_exact": True,
            "physical_trajectories_exact": True,
            "stored_rgb_exact": True,
        },
        "technical_integrity": technical_integrity,
        "physics_validation": {
            "receipt_checker_passed": True,
            "common_prefix_exact": True,
            "candidate_equivalence_measured_not_rejected": True,
            "minimum_physical_rank_classes_per_state": min(rank_counts),
            "maximum_physical_rank_classes_per_state": max(rank_counts),
            "identifiable_state_count": identifiable_state_count,
            "clipped_candidate_branches": clipped_candidate_count,
            "physics_validated_for_branch_outcomes": True,
        },
        "visual_validation": {
            "camera_quality_receipts_passed": hard_invalid_frames == 0,
            "endpoint_pose_replay_bound": True,
            "textured_v03_render_contract_validated": True,
            "visual_domain_fidelity_claimed": False,
            "eligible_for_physical_branch_evaluation": True,
            "eligible_for_visual_domain_parity_claim": False,
        },
        "candidate_branch_support_analysis": support_analysis,
        "resource_measurements": resource_measurements,
        "analyzer_binding": normalized_analyzer_binding,
        "checker_binding": normalized_checker_binding,
        "source_bindings": [
            {"name": CHECKER_SOURCE_NAME, "binding": normalized_checker_binding},
            {"name": ANALYZER_SOURCE_NAME, "binding": normalized_analyzer_binding},
            {"name": JOINER_SOURCE_NAME, "binding": normalized_joiner_binding},
        ],
    }


def derive_calibration_receipt_v1(
    collection: Mapping[str, Any],
    *,
    collection_binding: Mapping[str, object],
    analyzer_binding: Mapping[str, object],
    checker_binding: Mapping[str, object],
    joiner_binding: Mapping[str, object],
) -> dict[str, object]:
    """Derive a deterministic calibration receipt from validated receipts."""

    if _calibration_variant(collection) == "textured_v03_v3":
        return _derive_textured_v03_calibration_receipt(
            collection,
            collection_binding=collection_binding,
            analyzer_binding=analyzer_binding,
            checker_binding=checker_binding,
            joiner_binding=joiner_binding,
        )

    normalized_collection_binding = _validate_binding(
        collection_binding, name="calibration collection"
    )
    normalized_analyzer_binding = _validate_binding(
        analyzer_binding, name="calibration analyzer"
    )
    normalized_checker_binding = _validate_binding(
        checker_binding, name="receipt checker"
    )
    normalized_joiner_binding = _validate_binding(
        joiner_binding, name="pilot joiner"
    )
    if collection.get("purpose") != "sizing_calibration_only":
        raise CalibrationAnalysisError(
            "calibration analyzer requires a sizing_calibration_only collection"
        )
    counts = collection.get("counts")
    if not isinstance(counts, Mapping) or (
        counts.get("scenes") != CALIBRATION_SCENE_COUNT
        or counts.get("states") != CALIBRATION_STATE_COUNT
        or counts.get("total_branches") != CALIBRATION_BRANCH_COUNT
        or counts.get("candidate_branches") != CALIBRATION_STATE_COUNT * ACTION_COUNT
        or counts.get("sentinel_branches") != CALIBRATION_STATE_COUNT
        or counts.get("roles") != {"calibration": CALIBRATION_STATE_COUNT}
    ):
        raise CalibrationAnalysisError("calibration collection size contract changed")
    states = collection.get("states")
    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
        raise CalibrationAnalysisError("calibration states are absent")

    progress_repeat_deltas: list[float] = []
    path_repeat_deltas: list[float] = []
    endpoint_position_repeat_deltas: list[float] = []
    endpoint_quaternion_repeat_deltas: list[float] = []
    calibration_scene_ids: set[str] = set()
    repeated_action_ids: list[int] = []
    clipped_candidate_count = 0
    for state_index, state in enumerate(states):
        if not isinstance(state, Mapping):
            raise CalibrationAnalysisError(f"calibration state {state_index} is malformed")
        state_document = state.get("state")
        branches = state.get("branches")
        if (
            not isinstance(state_document, Mapping)
            or not isinstance(branches, Sequence)
            or isinstance(branches, (str, bytes))
            or len(branches) != ACTION_COUNT + 1
        ):
            raise CalibrationAnalysisError("calibration branch grid changed")
        calibration_scene_ids.add(str(state_document.get("scene_id")))
        sentinel = branches[-1]
        if not isinstance(sentinel, Mapping):
            raise CalibrationAnalysisError("calibration repeat branch is malformed")
        duplicate_action_id = sentinel.get("action_id")
        if type(duplicate_action_id) is not int or not 0 <= duplicate_action_id < ACTION_COUNT:
            raise CalibrationAnalysisError("calibration repeat action ID is invalid")
        repeated_action_ids.append(duplicate_action_id)
        candidate = branches[duplicate_action_id]
        if not isinstance(candidate, Mapping):
            raise CalibrationAnalysisError("calibration repeated candidate is malformed")
        progress_repeat_deltas.append(abs(
            _finite(candidate["physical_target_progress_m"], name="candidate progress")
            - _finite(sentinel["physical_target_progress_m"], name="sentinel progress")
        ))
        path_repeat_deltas.append(abs(
            _finite(candidate["physical_path_length_m"], name="candidate path length")
            - _finite(sentinel["physical_path_length_m"], name="sentinel path length")
        ))
        candidate_endpoint = candidate.get("endpoint_state")
        sentinel_endpoint = sentinel.get("endpoint_state")
        if not isinstance(candidate_endpoint, Mapping) or not isinstance(
            sentinel_endpoint, Mapping
        ):
            raise CalibrationAnalysisError("calibration endpoint state is absent")
        for field, output in (
            ("base_pos_world", endpoint_position_repeat_deltas),
            ("base_quat_wxyz", endpoint_quaternion_repeat_deltas),
        ):
            left = candidate_endpoint.get(field)
            right = sentinel_endpoint.get(field)
            if (
                not isinstance(left, Sequence)
                or isinstance(left, (str, bytes))
                or not isinstance(right, Sequence)
                or isinstance(right, (str, bytes))
                or len(left) != len(right)
            ):
                raise CalibrationAnalysisError(f"repeat endpoint {field} changed shape")
            output.append(max(
                abs(_finite(a, name=field) - _finite(b, name=field))
                for a, b in zip(left, right, strict=True)
            ))
        clipped_candidate_count += sum(
            int(branch.get("clipped") is True) for branch in branches[:ACTION_COUNT]
        )

    if len(calibration_scene_ids) != CALIBRATION_SCENE_COUNT:
        raise CalibrationAnalysisError("calibration scene identities repeat")
    if set(repeated_action_ids) != set(range(ACTION_COUNT)):
        raise CalibrationAnalysisError(
            "the 16 repeat controls do not cover all nine requested primitives"
        )
    max_progress_repeat = max(progress_repeat_deltas)
    max_path_repeat = max(path_repeat_deltas)
    max_endpoint_position_repeat = max(endpoint_position_repeat_deltas)
    max_endpoint_quaternion_repeat = max(endpoint_quaternion_repeat_deltas)
    if any(
        value != 0.0
        for value in (
            max_progress_repeat,
            max_path_repeat,
            max_endpoint_position_repeat,
            max_endpoint_quaternion_repeat,
        )
    ):
        raise CalibrationAnalysisError(
            "repeat controls are an exact deterministic gate and must have zero delta"
        )
    tolerances = {
        "progress_tolerance_m": NUMERICAL_FLOOR_M,
        "path_length_tolerance_m": NUMERICAL_FLOOR_M,
    }
    class_counts = [
        len({
            _physical_key(branch, tolerances)
            for branch in state["branches"][:ACTION_COUNT]
        })
        for state in states
    ]
    decision = (
        "FREEZE_PILOT_CONTRACT"
        if min(class_counts) >= 2
        else "STOP_SOURCE_REDESIGN"
    )
    frame_receipts = collection.get("frame_receipts")
    if not isinstance(frame_receipts, Mapping) or len(frame_receipts) != 208:
        raise CalibrationAnalysisError("calibration frame receipt grid changed")
    context_bytes = 0
    target_bytes = 0
    context_frames = 0
    target_frames = 0
    low_info_context_frames = 0
    low_info_target_frames = 0
    low_info_reason_counts = {
        "low_rgb_texture": 0,
        "near_wall_depth": 0,
        "near_forward_geometry": 0,
    }
    low_info_context_reason_counts = dict.fromkeys(low_info_reason_counts, 0)
    low_info_target_reason_counts = dict.fromkeys(low_info_reason_counts, 0)
    for frame in frame_receipts.values():
        if not isinstance(frame, Mapping):
            raise CalibrationAnalysisError("calibration frame receipt is malformed")
        identity = frame.get("frame_identity")
        byte_count = frame.get("byte_count")
        low_info_reasons = frame.get("low_info_reasons")
        if (
            not isinstance(identity, str)
            or type(byte_count) is not int
            or byte_count <= 0
            or type(frame.get("low_information")) is not bool
            or not isinstance(low_info_reasons, list)
            or any(reason not in low_info_reason_counts for reason in low_info_reasons)
            or len(low_info_reasons) != len(set(low_info_reasons))
            or frame["low_information"] != bool(low_info_reasons)
        ):
            raise CalibrationAnalysisError("calibration frame byte receipt changed")
        if ":context:" in identity:
            context_frames += 1
            context_bytes += byte_count
            if low_info_reasons:
                low_info_context_frames += 1
            selected_reason_counts = low_info_context_reason_counts
        else:
            target_frames += 1
            target_bytes += byte_count
            if low_info_reasons:
                low_info_target_frames += 1
            selected_reason_counts = low_info_target_reason_counts
        for reason in low_info_reasons:
            low_info_reason_counts[reason] += 1
            selected_reason_counts[reason] += 1
    if context_frames != 48 or target_frames != 160:
        raise CalibrationAnalysisError("calibration context/target frame split changed")

    collection_document = collection.get("document")
    if not isinstance(collection_document, Mapping):
        raise CalibrationAnalysisError("calibration collection document is absent")
    scene_metrics = collection_document.get("scene_metrics")
    if not isinstance(scene_metrics, list) or len(scene_metrics) != 8:
        raise CalibrationAnalysisError("calibration scene timing panel changed")
    timing_fields = (
        "physics_build_wall_seconds",
        "render_scene_build_wall_seconds",
        "common_prefix_step_wall_seconds",
        "branch_step_wall_seconds",
        "native_render_wall_seconds",
        "camera_quality_resize_wall_seconds",
        "png_encode_write_hash_wall_seconds",
        "post_lockstep_receipt_wall_seconds",
        "scene_total_wall_seconds",
    )
    timing_sums = {name: 0.0 for name in timing_fields}
    for scene_metric in scene_metrics:
        if not isinstance(scene_metric, Mapping):
            raise CalibrationAnalysisError("calibration scene timing row changed")
        for name in timing_fields:
            timing_sums[name] += _finite(
                scene_metric.get(name), name=f"scene timing {name}"
            )
    collection_wall_seconds = _finite(
        collection_document.get("collection_wall_seconds"),
        name="collection wall seconds",
    )
    if collection_wall_seconds < 0.0:
        raise CalibrationAnalysisError("collection wall time cannot be negative")
    resource_measurements = {
        "schema": RESOURCE_MEASUREMENTS_SCHEMA,
        "stored_rgb_png": {
            "context_frames": context_frames,
            "context_bytes": context_bytes,
            "target_frames": target_frames,
            "target_bytes": target_bytes,
            "total_frames": context_frames + target_frames,
            "total_bytes": context_bytes + target_bytes,
            "raw_uncompressed_rgb_ceiling_bytes": 208 * 224 * 224 * 3,
        },
        "low_information_strata": {
            "total_frames": low_info_context_frames + low_info_target_frames,
            "context_frames": low_info_context_frames,
            "target_frames": low_info_target_frames,
            "reason_counts": low_info_reason_counts,
            "context_reason_counts": low_info_context_reason_counts,
            "target_reason_counts": low_info_target_reason_counts,
            "frame_receipt_tags_present": True,
            "hard_invalid_frames": 0,
        },
        "stage_wall_seconds": {
            "collection_external_wall_seconds": collection_wall_seconds,
            "physics_scene_build_wall_seconds": timing_sums[
                "physics_build_wall_seconds"
            ],
            "render_scene_build_wall_seconds": timing_sums[
                "render_scene_build_wall_seconds"
            ],
            "common_prefix_step_wall_seconds": timing_sums[
                "common_prefix_step_wall_seconds"
            ],
            "branch_step_wall_seconds": timing_sums["branch_step_wall_seconds"],
            "native_render_wall_seconds": timing_sums[
                "native_render_wall_seconds"
            ],
            "camera_quality_resize_wall_seconds": timing_sums[
                "camera_quality_resize_wall_seconds"
            ],
            "png_encode_write_hash_wall_seconds": timing_sums[
                "png_encode_write_hash_wall_seconds"
            ],
            "post_lockstep_receipt_wall_seconds": timing_sums[
                "post_lockstep_receipt_wall_seconds"
            ],
            "summed_scene_total_wall_seconds": timing_sums[
                "scene_total_wall_seconds"
            ],
        },
        "outcome_counts": {
            "complete_all_nine_action_groups": CALIBRATION_STATE_COUNT,
            "executed_tape_distinct_groups": CALIBRATION_STATE_COUNT,
            "prebranch_exact_groups": CALIBRATION_STATE_COUNT,
            "clipped_candidate_branches": clipped_candidate_count,
            "fallen_candidate_branches": sum(
                int(branch["physical_fell"] is True)
                for state in states
                for branch in state["branches"][:ACTION_COUNT]
            ),
            "tipped_candidate_branches": sum(
                int(branch["physical_tipped"] is True)
                for state in states
                for branch in state["branches"][:ACTION_COUNT]
            ),
            "camera_invalid_frames": 0,
            "incomplete_states": 0,
        },
        "gpu_peak_memory_measurement_scope": (
            "external_terminal_required_not_observed_by_analyzer"
        ),
    }
    calibration_contract = {
        "schema": PHYSICAL_RANK_CONTRACT_SCHEMA,
        "excluded_scene_ids": sorted(calibration_scene_ids),
        **tolerances,
        "quantization_rule": "sign(x)*floor(abs(x)/t+0.5)",
        "lexicographic_key": [
            "physical_fell_ascending",
            "physical_tipped_ascending",
            "physical_target_progress_quantized_descending",
            "physical_path_length_quantized_ascending",
        ],
        "proxy_fields_excluded": True,
        "tolerance_derivation": {
            "schema": TOLERANCE_DERIVATION_SCHEMA,
            "method": "fixed_numerical_floor_after_exact_deterministic_repeat_gate",
            "minimum_numerical_resolution_m": NUMERICAL_FLOOR_M,
            "repeat_controls": CALIBRATION_STATE_COUNT,
            "repeated_action_ids": repeated_action_ids,
            "all_requested_primitives_covered": True,
            "deterministic_repeat_gate_passed": True,
            "empirical_noise_scale_estimated": False,
        },
    }
    return {
        "schema": CALIBRATION_RECEIPT_SCHEMA,
        "status": "COMPLETE",
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "calibration_id": str(collection["document"]["attempt_id"]),
        "role": "calibration",
        "train_eval_scenes_accessed": False,
        "decision": decision,
        "calibration_collection_receipt": normalized_collection_binding,
        "calibration_contract": calibration_contract,
        "repeatability_analysis": {
            "repeat_controls": CALIBRATION_STATE_COUNT,
            "repeated_action_ids": repeated_action_ids,
            "all_requested_primitives_covered": True,
            "interpretation": (
                "deterministic_replay_gate_not_empirical_noise_estimate"
            ),
            "empirical_noise_scale_estimated": False,
            "progress_max_abs_delta_m": max_progress_repeat,
            "path_length_max_abs_delta_m": max_path_repeat,
            "endpoint_position_max_abs_delta_m": max_endpoint_position_repeat,
            "endpoint_quaternion_max_abs_delta": max_endpoint_quaternion_repeat,
            "executed_command_tapes_exact": True,
            "physical_trajectories_exact": True,
            "stored_rgb_exact": True,
        },
        "physics_validation": {
            "receipt_checker_passed": True,
            "common_prefix_exact": True,
            "nine_unique_executed_tapes_per_state": True,
            "minimum_physical_rank_classes_per_state": min(class_counts),
            "maximum_physical_rank_classes_per_state": max(class_counts),
            "clipped_candidate_branches": clipped_candidate_count,
            "physics_validated_for_branch_outcomes": True,
        },
        "visual_validation": {
            "camera_quality_receipts_passed": True,
            "endpoint_pose_replay_bound": True,
            "visual_domain_fidelity_claimed": False,
            "eligible_for_physical_branch_evaluation": True,
            "eligible_for_visual_domain_parity_claim": False,
        },
        "resource_measurements": resource_measurements,
        "analyzer_binding": normalized_analyzer_binding,
        "checker_binding": normalized_checker_binding,
        "source_bindings": [
            {"name": CHECKER_SOURCE_NAME, "binding": normalized_checker_binding},
            {"name": ANALYZER_SOURCE_NAME, "binding": normalized_analyzer_binding},
            {"name": JOINER_SOURCE_NAME, "binding": normalized_joiner_binding},
        ],
    }


def _write_exclusive(path: Path, value: Mapping[str, object]) -> dict[str, object]:
    selected = Path(path)
    if selected.exists() or selected.is_symlink():
        raise FileExistsError(f"refusing to overwrite calibration receipt: {selected}")
    selected.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_json_bytes(value) + b"\n"
    descriptor = os.open(
        selected,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        written = 0
        while written < len(raw):
            written += os.write(descriptor, raw[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return _binding(selected)


def analyze_calibration(
    *,
    collection_path: Path,
    expected_collection_sha256: str,
    expected_collection_byte_count: int,
    output_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    selected_collection = Path(collection_path)
    try:
        resolved_collection = selected_collection.resolve(strict=True)
    except OSError as exc:
        raise CalibrationAnalysisError(
            "cannot resolve caller-bound calibration collection"
        ) from exc
    if (
        not selected_collection.is_absolute()
        or selected_collection != resolved_collection
    ):
        raise CalibrationAnalysisError(
            "calibration collection path is not canonical and symlink-free"
        )
    collection = checker.load_bound_collection_receipts(
        resolved_collection,
        expected_file_sha256=expected_collection_sha256,
        expected_byte_count=expected_collection_byte_count,
        verify_textured_pixels=True,
    )
    # The checker consumed exactly the caller-bound bytes above.  Preserve that
    # identity directly; reopening the pathname here could bind a replacement
    # file to the already-validated in-memory collection.
    collection_binding = {
        "path": str(resolved_collection),
        "file_sha256": expected_collection_sha256,
        "byte_count": expected_collection_byte_count,
    }
    receipt = derive_calibration_receipt_v1(
        collection,
        collection_binding=collection_binding,
        analyzer_binding=_binding(Path(__file__)),
        checker_binding=_binding(Path(checker.__file__)),
        joiner_binding=_binding(
            REPO_ROOT / "scripts/join_go2_world_model_counterfactual_pilot_v1.py"
        ),
    )
    output_binding = _write_exclusive(output_path, receipt)
    return receipt, output_binding


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--expected-collection-sha256", required=True)
    parser.add_argument("--expected-collection-byte-count", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt, binding = analyze_calibration(
        collection_path=args.collection,
        expected_collection_sha256=args.expected_collection_sha256,
        expected_collection_byte_count=args.expected_collection_byte_count,
        output_path=args.output,
    )
    print(json.dumps({
        "decision": receipt["decision"],
        "calibration_receipt": binding,
    }, sort_keys=True))
    return 0 if receipt["decision"] == "FREEZE_PILOT_CONTRACT" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BOUNDED_MINIMUM_DISCRIMINATION_QUERY_COVERAGE",
    "CALIBRATION_RECEIPT_SCHEMA",
    "CALIBRATED_MINIMUM_DISCRIMINATION_QUERY_COVERAGE",
    "CANDIDATE_BRANCH_SUPPORT_ANALYSIS_SCHEMA",
    "RESOURCE_MEASUREMENTS_SCHEMA",
    "MIN_ELIGIBLE_QUERIES_OVERALL",
    "MIN_ELIGIBLE_QUERIES_PER_FAMILY",
    "TEXTURED_V03_CALIBRATION_RECEIPT_SCHEMA",
    "TEXTURED_V03_OUTCOME_EQUIVALENCE_TOLERANCE_M",
    "TEXTURED_V03_RESOURCE_MEASUREMENTS_SCHEMA",
    "TOLERANCE_DERIVATION_SCHEMA",
    "TOLERANCE_DERIVATION_V2_SCHEMA",
    "CalibrationAnalysisError",
    "derive_calibration_receipt_v1",
    "load_bound_calibration_receipt_v1",
    "validate_calibration_receipt_v1",
]
