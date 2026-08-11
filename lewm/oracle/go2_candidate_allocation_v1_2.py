"""Deterministic pre-outcome candidate allocation for scorer-fit v1.2.

This module deliberately accepts only a six-field projection of the frozen
state-identity manifest.  In particular, it cannot consume branch validity,
oracle labels, utilities, renders, latents, or any other post-branch value.

The twelve allowed six-candidate blocks are rotations of one frozen offset
set.  The offset set has a rotational complement: block ``k + 6`` is exactly
the complement of block ``k``.  That property makes the required balances for
8-state calibration strata and 4-state fit family/stratum cells attainable.

The allocation is the lexicographically smallest feasible vector of rotation
indices after states are sorted by their bound identity digest.  It is found
by a sequence of exact mixed-integer feasibility problems, one state at a
time.  Consequently the result is independent of caller input order and does
not depend on which arbitrary feasible point the MILP solver returns.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


SCHEMA = "go2_candidate_allocation_v1_2_manifest"
STATUS = "FROZEN_PRE_OUTCOME_IDENTITY_ALLOCATION"
ALGORITHM_VERSION = "go2_candidate_allocation_v1_2_lexicographic_milp_v1"
AMENDMENT_SCHEMA = "go2_shared_utility_scorer_v1_2_allocation_amendment_v1"
AMENDMENT_VERSION = "allocation_reverse_coverage_resolution_v1"
PRE_IDENTITY_VALIDATION_SCHEMA = (
    "go2_candidate_allocation_v1_2_pre_identity_structural_validation"
)
POST_IDENTITY_VALIDATION_SCHEMA = (
    "go2_candidate_allocation_v1_2_post_identity_pre_outcome_validation"
)
GOAL_TYPE_VERSION = "snapshot_bound_landmark_material_id_v1"
GOAL_TYPE_DEFINITION = (
    "the material_id of the landmark bound at snapshot time, before any "
    "candidate branch is executed"
)
CANDIDATE_BANK_DIGEST = (
    "85471e44a0fe8f3c59fff258e9b23933e306f69b6d590c832e2b8da1f34a8cd9"
)
SCORER_FIT_ALLOCATION_DESIGN_DIGEST = (
    "a587b1de264dfb54176aa231e5183ae4b7b4229bbf65c02d62438f86af5e7116"
)
PREDECESSOR_ALLOCATOR_CONTRACT_DIGEST = (
    "bb2d9956947be64985f15970dc30f9f0e37cda8012f7c7f5da8808c5d601de5e"
)
AUTHORIZING_FAILURE_COMMIT = (
    "6a4d6a66c93d9461bdfb8bf4c2ccb5b882dcdb78"
)
FAILURE_RECEIPT_PATH = (
    "docs/lewm_go2_shared_utility_scorer_v1_2_"
    "preoutcome_allocation_failure_2026-08-11.json"
)
FAILURE_RECEIPT_DIGEST = (
    "550c52f9a3ff04f8a564f6f28e75e9d36fc8bc0f73da4795b95dedc3ad2e3cab"
)
FAILURE_RECEIPT_RAW_SHA256 = (
    "3e224158d43a4e75fc7a60436feaeb00cd538a5fabfae5a92983f7ede612df99"
)
AMENDMENT_ARTIFACT_PATH = (
    "docs/lewm_go2_shared_utility_scorer_v1_2_"
    "allocation_amendment_v1_2026-08-11.json"
)

CANDIDATE_COUNT = 12
CANDIDATES_PER_STATE = 6
ROTATION_OFFSETS = (0, 1, 3, 5, 8, 10)
STRATA = ("general", "safety_enriched", "completion_enriched")
SPLIT_ROLES = ("fit", "calibration")
FORWARD_CANDIDATES = frozenset((0, 1, 2))
TURNING_CANDIDATES = frozenset((3, 4, 5, 6, 7, 8, 9))
REVERSING_CANDIDATE = 10
HOLD_CANDIDATE = 11
FAMILIES = (
    "large_enclosed_maze",
    "local_composite_motifs",
    "loop_alias_stress",
    "medium_enclosed_maze",
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "visual_sensor_stress",
)

_INPUT_KEYS = frozenset(
    ("state_id", "state_identity_digest", "family", "stratum", "split_role",
     "goal_type")
)
_ASSIGNMENT_KEYS = _INPUT_KEYS | frozenset(("rotation_index", "candidate_indices"))
_HEX = frozenset("0123456789abcdef")


class CandidateAllocationError(ValueError):
    """The identity projection or allocation manifest violates the contract."""


class CandidateAllocationInfeasible(RuntimeError):
    """The frozen identities admit no allocation satisfying every hard margin."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def allocation_amendment_contract() -> dict[str, Any]:
    """Return the one prospective interpretation amendment.

    The amendment resolves the contradiction recorded before any scorer-fit
    state identity or branch outcome existed.  It does not change the bank,
    the twelve allowed subsets, the allocator, or any balance margin.
    """

    return {
        "schema": AMENDMENT_SCHEMA,
        "status": "AUTHORIZED_PROSPECTIVE_PRE_IDENTITY_AMENDMENT",
        "version": AMENDMENT_VERSION,
        "lineage": {
            "authorizing_failure_commit": AUTHORIZING_FAILURE_COMMIT,
            "failure_receipt_path": FAILURE_RECEIPT_PATH,
            "failure_receipt_digest": FAILURE_RECEIPT_DIGEST,
            "failure_receipt_raw_sha256": FAILURE_RECEIPT_RAW_SHA256,
            "original_scorer_fit_allocation_design_digest":
                SCORER_FIT_ALLOCATION_DESIGN_DIGEST,
            "predecessor_allocator_contract_digest":
                PREDECESSOR_ALLOCATOR_CONTRACT_DIGEST,
        },
        "preserved": {
            "candidate_bank_digest": CANDIDATE_BANK_DIGEST,
            "candidate_count": CANDIDATE_COUNT,
            "candidates_per_state": CANDIDATES_PER_STATE,
            "state_count": 120,
            "attempted_branch_count": 720,
            "rotation_offsets": list(ROTATION_OFFSETS),
            "allowed_subset_count": CANDIDATE_COUNT,
            "appearances_per_candidate": 60,
            "fit_states": 96,
            "calibration_states": 24,
            "strata": list(STRATA),
            "families": list(FAMILIES),
            "allocator_algorithm_version": ALGORITHM_VERSION,
        },
        "single_interpretation_change": {
            "superseded_infeasible_clause": (
                "every six-candidate subset contains reversing behaviour"
            ),
            "replacement_clause": (
                "the sole reversing candidate occurs in exactly 60 distinct "
                "state subsets, as required by exact-60 candidate balance"
            ),
            "reversing_candidate_index": REVERSING_CANDIDATE,
            "required_distinct_reverse_subset_count": 60,
            "per_subset_forward_requirement_preserved": True,
            "per_subset_turning_requirement_preserved": True,
            "per_subset_reverse_requirement": False,
        },
        "goal_type_boundary": {
            "unavailable_before_state_identity_selection": True,
            "pre_identity_status": "NOT_EVALUABLE_BEFORE_IDENTITIES",
            "mandatory_post_identity_pre_outcome_rule": (
                "for every snapshot-bound goal_type containing n states, each "
                "candidate count is floor(n/2) or ceil(n/2); for odd n exactly "
                "six candidates receive each count"
            ),
            "candidate_outcomes_consumed": False,
        },
        "scientific_changes": {
            "candidate_bank": False,
            "allowed_subsets": False,
            "candidate_margins": False,
            "fit_calibration_split": False,
            "state_strata": False,
            "goal_type_balance": False,
            "allocator_algorithm": False,
            "reverse_coverage_interpretation_only": True,
        },
    }


def allocation_amendment_digest() -> str:
    """Canonical digest of the prospective amendment overlay."""

    return _sha256(allocation_amendment_contract())


def validate_allocation_amendment_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the tracked amendment artifact against the code contract."""

    if not isinstance(artifact, Mapping):
        raise CandidateAllocationError("allocation amendment artifact must be a mapping")
    payload = dict(artifact)
    observed_digest = payload.pop("allocation_amendment_digest", None)
    if payload != allocation_amendment_contract():
        raise CandidateAllocationError(
            "tracked allocation amendment differs from the code contract"
        )
    if observed_digest != allocation_amendment_digest():
        raise CandidateAllocationError("tracked allocation amendment digest mismatch")


def candidate_block(rotation_index: int) -> tuple[int, ...]:
    """Return one of the twelve frozen rotated six-candidate blocks."""

    if isinstance(rotation_index, bool) or not isinstance(rotation_index, int):
        raise CandidateAllocationError("rotation_index must be an integer")
    if not 0 <= rotation_index < CANDIDATE_COUNT:
        raise CandidateAllocationError("rotation_index must be in [0, 11]")
    return tuple(sorted(
        (rotation_index + offset) % CANDIDATE_COUNT
        for offset in ROTATION_OFFSETS
    ))


ROTATION_BLOCKS = tuple(candidate_block(index) for index in range(CANDIDATE_COUNT))


def _pairwise_cooccurrence(
    blocks: Sequence[Sequence[int]],
) -> list[list[int]]:
    """Return a symmetric candidate co-occurrence matrix.

    Diagonal entries are candidate incidence counts.  Off-diagonal entries are
    counts of subsets containing both candidates.
    """

    matrix = [[0] * CANDIDATE_COUNT for _ in range(CANDIDATE_COUNT)]
    for raw_block in blocks:
        block = {int(candidate) for candidate in raw_block}
        if len(block) != CANDIDATES_PER_STATE or any(
                not 0 <= candidate < CANDIDATE_COUNT for candidate in block):
            raise CandidateAllocationError("invalid block in pairwise co-occurrence")
        for left in block:
            for right in block:
                matrix[left][right] += 1
    return matrix


def _subset_catalogue() -> list[dict[str, Any]]:
    return [
        {
            "rotation_index": rotation,
            "candidate_indices": list(block),
            "contains_forward": bool(FORWARD_CANDIDATES.intersection(block)),
            "contains_turning": bool(TURNING_CANDIDATES.intersection(block)),
            "contains_reverse": REVERSING_CANDIDATE in block,
            "contains_hold": HOLD_CANDIDATE in block,
        }
        for rotation, block in enumerate(ROTATION_BLOCKS)
    ]


def _pre_identity_slots() -> list[dict[str, Any]]:
    """Return the deterministic structural slots, without state identities."""

    slots: list[dict[str, Any]] = []
    slot_index = 0
    for family in FAMILIES:
        for stratum in STRATA:
            for ordinal in range(5):
                split_role = "calibration" if ordinal == 0 else "fit"
                slots.append({
                    "slot_index": slot_index,
                    "slot_id": f"{family}|{stratum}|{split_role}|{ordinal}",
                    "family": family,
                    "stratum": stratum,
                    "stratum_ordinal": ordinal,
                    "split_role": split_role,
                    "candidate_assignment_status": (
                        "DEFERRED_UNTIL_POST_IDENTITY_CANONICAL_ALLOCATION"
                    ),
                    "allowed_rotation_indices": list(range(CANDIDATE_COUNT)),
                })
                slot_index += 1
    return slots


def build_pre_identity_structural_validation() -> dict[str, Any]:
    """Build the outcome-free 120-slot allocation preflight artifact.

    Candidate rotations are deliberately *not* assigned here: the frozen
    canonical allocator needs the state-identity digest and snapshot-bound goal
    type.  This artifact validates everything knowable before those identities,
    including the allowed subset catalogue and its pairwise incidence table.
    Actual assignment co-occurrence and goal-type balance are mandatory in the
    later post-identity/pre-outcome validation embedded in the allocation
    manifest.
    """

    slots = _pre_identity_slots()
    subset_catalogue = _subset_catalogue()
    fixture_goal_type = "preidentity_placeholder_goal_type_all_120_slots"
    fixture_identities = [{
        "state_id": f"preidentity-structural-slot-{slot['slot_index']:03d}",
        "state_identity_digest": _sha256({
            "schema": "go2_candidate_allocation_v1_2_preidentity_fixture_state",
            "allocation_amendment_digest": allocation_amendment_digest(),
            "slot": slot,
        }),
        "family": slot["family"],
        "stratum": slot["stratum"],
        "split_role": slot["split_role"],
        "goal_type": fixture_goal_type,
    } for slot in slots]
    fixture_source_digest = _sha256({
        "schema": "go2_candidate_allocation_v1_2_preidentity_fixture_source",
        "actual_state_identities": False,
        "candidate_outcomes_consumed": False,
        "states": fixture_identities,
    })
    # This is an actual run of the unchanged canonical allocator over the 120
    # deterministic structural slots.  It is a feasibility witness only; the
    # actual state identities and snapshot-bound goal types are allocated and
    # revalidated separately before any branch outcome.
    fixture_allocation = build_allocation_manifest(
        fixture_identities,
        source_identity_manifest_digest=fixture_source_digest,
    )
    fixture_validation = fixture_allocation[
        "post_identity_pre_outcome_validation"
    ]
    fit_slots = [slot for slot in slots if slot["split_role"] == "fit"]
    calibration_slots = [
        slot for slot in slots if slot["split_role"] == "calibration"
    ]
    strata = [
        {
            "stratum": stratum,
            "state_slot_count": sum(slot["stratum"] == stratum for slot in slots),
            "candidate_slot_count":
                sum(slot["stratum"] == stratum for slot in slots)
                * CANDIDATES_PER_STATE,
            "fit_state_slot_count": sum(
                slot["stratum"] == stratum and slot["split_role"] == "fit"
                for slot in slots
            ),
            "calibration_state_slot_count": sum(
                slot["stratum"] == stratum
                and slot["split_role"] == "calibration"
                for slot in slots
            ),
        }
        for stratum in STRATA
    ]
    families = [
        {
            "family": family,
            "state_slot_count": sum(slot["family"] == family for slot in slots),
            "candidate_slot_count":
                sum(slot["family"] == family for slot in slots)
                * CANDIDATES_PER_STATE,
            "fit_state_slot_count": sum(
                slot["family"] == family and slot["split_role"] == "fit"
                for slot in slots
            ),
            "calibration_state_slot_count": sum(
                slot["family"] == family
                and slot["split_role"] == "calibration"
                for slot in slots
            ),
        }
        for family in FAMILIES
    ]
    reverse_catalogue_count = sum(
        bool(row["contains_reverse"]) for row in subset_catalogue
    )
    artifact: dict[str, Any] = {
        "schema": PRE_IDENTITY_VALIDATION_SCHEMA,
        "status": "PASS_PRE_IDENTITY_STRUCTURAL_VALIDATION",
        "allocation_contract_digest": allocation_contract_digest(),
        "allocation_amendment_digest": allocation_amendment_digest(),
        "candidate_bank_digest": CANDIDATE_BANK_DIGEST,
        "original_scorer_fit_allocation_design_digest":
            SCORER_FIT_ALLOCATION_DESIGN_DIGEST,
        "global": {
            "state_slot_count": len(slots),
            "candidates_per_state": CANDIDATES_PER_STATE,
            "candidate_slot_count": len(slots) * CANDIDATES_PER_STATE,
            "candidate_count": CANDIDATE_COUNT,
            "required_appearances_per_candidate": 60,
            "required_distinct_reverse_subset_count": 60,
        },
        "split_role": [
            {
                "split_role": "fit",
                "state_slot_count": len(fit_slots),
                "candidate_slot_count": len(fit_slots) * CANDIDATES_PER_STATE,
                "required_appearances_per_candidate": 48,
            },
            {
                "split_role": "calibration",
                "state_slot_count": len(calibration_slots),
                "candidate_slot_count":
                    len(calibration_slots) * CANDIDATES_PER_STATE,
                "required_appearances_per_candidate": 12,
            },
        ],
        "strata": strata,
        "families": families,
        "slots": slots,
        "allowed_subset_catalogue": {
            "assignment_status": (
                "CATALOGUE_VALIDATED_ASSIGNMENTS_DEFERRED_UNTIL_IDENTITIES"
            ),
            "subsets": subset_catalogue,
            "subset_type_count": len(subset_catalogue),
            "subset_types_with_reverse": reverse_catalogue_count,
            "subset_types_without_reverse":
                len(subset_catalogue) - reverse_catalogue_count,
            "pairwise_cooccurrence_across_allowed_subset_types": {
                "definition": (
                    "diagonal=candidate incidence across the 12 allowed subset "
                    "types; off-diagonal=number of allowed subset types containing "
                    "both candidates"
                ),
                "candidate_order": list(range(CANDIDATE_COUNT)),
                "matrix": _pairwise_cooccurrence(ROTATION_BLOCKS),
            },
        },
        "structural_allocator_fixture": {
            "status": "PASS_UNCHANGED_ALLOCATOR_120_SLOT_FIXTURE",
            "is_actual_state_identity_assignment": False,
            "candidate_outcomes_consumed": False,
            "purpose": (
                "pre-identity feasibility and balance validation only; it does "
                "not assign candidates to the later selected state identities"
            ),
            "placeholder_goal_type": fixture_goal_type,
            "placeholder_goal_type_rationale": (
                "one goal type across all 120 structural slots makes its exact-60 "
                "balance redundant with the global exact-60 margin; actual "
                "snapshot-bound goal types remain mandatory post-identity"
            ),
            "fixture_source_identity_manifest_digest": fixture_source_digest,
            "fixture_allocation_manifest_digest":
                fixture_allocation["allocation_manifest_digest"],
            "assignments": fixture_allocation["assignments"],
            "contingency_tables": fixture_allocation["contingency_tables"],
            "validation": fixture_validation,
        },
        "actual_assignment_validation": {
            "status": "DEFERRED_UNTIL_POST_IDENTITY_PRE_OUTCOME",
            "required": True,
            "will_validate": [
                "global candidate totals",
                "fit and calibration candidate totals",
                "stratum candidate totals",
                "family and family-stratum integer balance",
                "actual subset flags and rotation usage",
                "actual 120-assignment pairwise candidate co-occurrence",
                "exactly 60 distinct state subsets containing candidate 10",
            ],
        },
        "goal_type_validation": {
            "status": "NOT_EVALUABLE_BEFORE_STATE_IDENTITIES",
            "actual_goal_types_observed": False,
            "required_post_identity_pre_outcome": True,
            "rule": allocation_amendment_contract()["goal_type_boundary"][
                "mandatory_post_identity_pre_outcome_rule"
            ],
            "candidate_outcomes_consumed": False,
        },
        "checks": {
            "state_slot_count_is_120": len(slots) == 120,
            "candidate_slot_count_is_720":
                len(slots) * CANDIDATES_PER_STATE == 720,
            "fit_state_slot_count_is_96": len(fit_slots) == 96,
            "calibration_state_slot_count_is_24":
                len(calibration_slots) == 24,
            "eight_families_with_15_slots_each": all(
                row["state_slot_count"] == 15 for row in families
            ),
            "three_strata_with_40_slots_each": all(
                row["state_slot_count"] == 40 for row in strata
            ),
            "twelve_unique_allowed_subsets":
                len({tuple(block) for block in ROTATION_BLOCKS}) == 12,
            "every_allowed_subset_has_forward": all(
                row["contains_forward"] for row in subset_catalogue
            ),
            "every_allowed_subset_has_turning": all(
                row["contains_turning"] for row in subset_catalogue
            ),
            "six_allowed_subset_types_have_reverse":
                reverse_catalogue_count == 6,
            "per_subset_reverse_not_required_by_amendment": True,
            "unchanged_allocator_fixture_has_120_assignments":
                len(fixture_allocation["assignments"]) == 120,
            "unchanged_allocator_fixture_global_counts_are_exactly_60":
                fixture_validation["global"]["candidate_counts"]
                == [60] * CANDIDATE_COUNT,
            "unchanged_allocator_fixture_reverse_occurs_in_60_subsets":
                fixture_validation["reverse_coverage"][
                    "observed_distinct_state_subset_count"] == 60,
            "unchanged_allocator_fixture_goal_placeholder_passes":
                fixture_validation["goal_type_validation"]["status"]
                == "PASS_ACTUAL_POST_IDENTITY_PRE_OUTCOME_GOAL_TYPE_BALANCE",
            "goal_type_correctly_deferred": True,
        },
    }
    if not all(artifact["checks"].values()):
        raise CandidateAllocationError("pre-identity structural allocation check failed")
    artifact["pre_identity_validation_digest"] = _sha256(artifact)
    return artifact


def validate_pre_identity_structural_validation(
    artifact: Mapping[str, Any],
) -> None:
    """Validate the exact deterministic pre-identity artifact."""

    if not isinstance(artifact, Mapping):
        raise CandidateAllocationError("pre-identity validation must be a mapping")
    expected = build_pre_identity_structural_validation()
    if dict(artifact) != expected:
        raise CandidateAllocationError(
            "pre-identity structural validation differs from the frozen artifact"
        )


def algorithm_contract() -> dict[str, Any]:
    """Return the complete versioned, outcome-independent allocation contract."""

    return {
        "algorithm_version": ALGORITHM_VERSION,
        "candidate_bank_digest": CANDIDATE_BANK_DIGEST,
        "candidate_count": CANDIDATE_COUNT,
        "candidates_per_state": CANDIDATES_PER_STATE,
        "rotation_offsets": list(ROTATION_OFFSETS),
        "rotation_rule": "B_k={(k+offset) mod 12: offset in rotation_offsets}",
        "canonical_state_order": "(state_identity_digest, state_id)",
        "choice_rule": (
            "lexicographically smallest feasible rotation-index vector in "
            "canonical_state_order"
        ),
        "goal_type_version": GOAL_TYPE_VERSION,
        "goal_type_definition": GOAL_TYPE_DEFINITION,
        "hard_margins": {
            "fit_family_stratum_candidate": 2,
            "calibration_stratum_candidate": 4,
            "calibration_family_candidate": [1, 2],
            "goal_type_candidate": "floor(n_goal/2)..ceil(n_goal/2)",
        },
        "outcome_fields_consumed": [],
    }


def allocation_contract_digest() -> str:
    return _sha256(algorithm_contract())


def _normalise_identity_state(record: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(record, Mapping):
        raise CandidateAllocationError("every state must be a mapping")
    keys = frozenset(record)
    if keys != _INPUT_KEYS:
        missing = sorted(_INPUT_KEYS - keys)
        extra = sorted(keys - _INPUT_KEYS)
        raise CandidateAllocationError(
            f"state identity projection has wrong keys; missing={missing}, extra={extra}"
        )

    result: dict[str, str] = {}
    for key in _INPUT_KEYS:
        value = record[key]
        if not isinstance(value, str) or not value:
            raise CandidateAllocationError(f"{key} must be a non-empty string")
        result[key] = value
    if not _is_digest(result["state_identity_digest"]):
        raise CandidateAllocationError(
            "state_identity_digest must be a lowercase 64-character SHA-256"
        )
    if result["stratum"] not in STRATA:
        raise CandidateAllocationError(
            f"unknown stratum {result['stratum']!r}; expected one of {STRATA}"
        )
    if result["split_role"] not in SPLIT_ROLES:
        raise CandidateAllocationError(
            f"unknown split_role {result['split_role']!r}; expected one of {SPLIT_ROLES}"
        )
    return result


def _normalise_identity_states(
    states: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    if isinstance(states, (str, bytes)) or not isinstance(states, Sequence):
        raise CandidateAllocationError("states must be a sequence of identity mappings")
    normalised = [_normalise_identity_state(record) for record in states]
    normalised.sort(key=lambda row: (row["state_identity_digest"], row["state_id"]))
    if len(normalised) != 120:
        raise CandidateAllocationError(f"expected exactly 120 states, got {len(normalised)}")
    state_ids = [row["state_id"] for row in normalised]
    identity_digests = [row["state_identity_digest"] for row in normalised]
    if len(set(state_ids)) != len(state_ids):
        raise CandidateAllocationError("state_id values must be unique")
    if len(set(identity_digests)) != len(identity_digests):
        raise CandidateAllocationError("state_identity_digest values must be unique")

    families = sorted({row["family"] for row in normalised})
    if len(families) != 8:
        raise CandidateAllocationError(f"expected exactly 8 families, got {len(families)}")
    for family in families:
        for stratum in STRATA:
            cell = [
                row for row in normalised
                if row["family"] == family and row["stratum"] == stratum
            ]
            fit = [row for row in cell if row["split_role"] == "fit"]
            calibration = [
                row for row in cell if row["split_role"] == "calibration"
            ]
            if len(cell) != 5 or len(fit) != 4 or len(calibration) != 1:
                raise CandidateAllocationError(
                    f"{family}/{stratum} must contain 4 fit and 1 calibration "
                    f"state; got fit={len(fit)}, calibration={len(calibration)}"
                )
    return normalised


def pre_outcome_identity_digest(states: Sequence[Mapping[str, Any]]) -> str:
    """Digest the strict canonical identity projection used by the allocator."""

    normalised = _normalise_identity_states(states)
    return _sha256({
        "schema": "go2_candidate_allocation_v1_2_identity_projection",
        "goal_type_version": GOAL_TYPE_VERSION,
        "states": normalised,
    })


def _indices_matching(
    states: Sequence[Mapping[str, str]], **fields: str,
) -> list[int]:
    return [
        index for index, state in enumerate(states)
        if all(state[key] == value for key, value in fields.items())
    ]


def _constraint_system(states: Sequence[Mapping[str, str]]):
    """Build the sparse hard-margin system over state/rotation indicators."""

    try:
        from scipy.optimize import Bounds, LinearConstraint
        from scipy.sparse import coo_matrix
    except ImportError as exc:  # pragma: no cover - the project runtime binds SciPy.
        raise CandidateAllocationError(
            "scipy with scipy.optimize.milp is required for candidate allocation"
        ) from exc

    variable_count = len(states) * CANDIDATE_COUNT
    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    matrix_values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(coefficients: Sequence[int], minimum: int, maximum: int) -> None:
        row_index = len(lower)
        for variable in coefficients:
            matrix_rows.append(row_index)
            matrix_columns.append(variable)
            matrix_values.append(1.0)
        lower.append(float(minimum))
        upper.append(float(maximum))

    rotations_containing = tuple(
        tuple(k for k, block in enumerate(ROTATION_BLOCKS) if candidate in block)
        for candidate in range(CANDIDATE_COUNT)
    )

    def candidate_variables(state_indices: Sequence[int], candidate: int) -> list[int]:
        return [
            state_index * CANDIDATE_COUNT + rotation
            for state_index in state_indices
            for rotation in rotations_containing[candidate]
        ]

    # Exactly one rotation block per state.
    for state_index in range(len(states)):
        start = state_index * CANDIDATE_COUNT
        add(list(range(start, start + CANDIDATE_COUNT)), 1, 1)

    families = sorted({state["family"] for state in states})
    goal_types = sorted({state["goal_type"] for state in states})

    # Four fit states in every family/stratum cell contribute each candidate twice.
    for family in families:
        for stratum in STRATA:
            indices = _indices_matching(
                states, family=family, stratum=stratum, split_role="fit"
            )
            for candidate in range(CANDIDATE_COUNT):
                add(candidate_variables(indices, candidate), 2, 2)

    # Eight calibration states per stratum contribute each candidate four times.
    for stratum in STRATA:
        indices = _indices_matching(
            states, stratum=stratum, split_role="calibration"
        )
        for candidate in range(CANDIDATE_COUNT):
            add(candidate_variables(indices, candidate), 4, 4)

    # Three calibration states per family contain each candidate once or twice.
    for family in families:
        indices = _indices_matching(states, family=family, split_role="calibration")
        for candidate in range(CANDIDATE_COUNT):
            add(candidate_variables(indices, candidate), 1, 2)

    # Integer-optimal balance within every snapshot-time landmark material.
    for goal_type in goal_types:
        indices = _indices_matching(states, goal_type=goal_type)
        low = len(indices) // 2
        high = (len(indices) + 1) // 2
        for candidate in range(CANDIDATE_COUNT):
            add(candidate_variables(indices, candidate), low, high)

    matrix = coo_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(len(lower), variable_count), dtype=np.float64,
    ).tocsc()
    return (
        LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        Bounds(np.zeros(variable_count), np.ones(variable_count)),
    )


def _lexicographic_rotations(states: Sequence[Mapping[str, str]]) -> list[int]:
    """Find the canonical lexicographically smallest feasible rotation vector."""

    try:
        from scipy.optimize import Bounds, milp
    except ImportError as exc:  # pragma: no cover
        raise CandidateAllocationError(
            "scipy with scipy.optimize.milp is required for candidate allocation"
        ) from exc

    constraints, base_bounds = _constraint_system(states)
    variable_count = len(states) * CANDIDATE_COUNT
    integrality = np.ones(variable_count, dtype=np.uint8)
    lower = np.asarray(base_bounds.lb, dtype=np.float64).copy()
    upper = np.asarray(base_bounds.ub, dtype=np.float64).copy()
    selected: list[int] = []
    options = {"disp": False, "presolve": True, "mip_rel_gap": 0.0}

    for state_index in range(len(states)):
        objective = np.zeros(variable_count, dtype=np.float64)
        start = state_index * CANDIDATE_COUNT
        objective[start:start + CANDIDATE_COUNT] = np.arange(
            CANDIDATE_COUNT, dtype=np.float64
        )
        result = milp(
            c=objective,
            integrality=integrality,
            bounds=Bounds(lower, upper),
            constraints=constraints,
            options=options,
        )
        if result.status == 2:
            if state_index == 0:
                raise CandidateAllocationInfeasible(
                    "the frozen identity/goal-type contingency has no allocation "
                    "satisfying all exact candidate margins"
                )
            raise CandidateAllocationError(
                "internal lexicographic solver defect: a previously feasible "
                "prefix became infeasible"
            )
        if not result.success or result.x is None:
            raise CandidateAllocationError(
                f"candidate-allocation MILP did not complete: status={result.status}, "
                f"message={result.message!r}"
            )
        local = np.asarray(result.x[start:start + CANDIDATE_COUNT])
        rotation = int(np.argmax(local))
        if local[rotation] < 0.5:
            raise CandidateAllocationError(
                "candidate-allocation MILP returned a non-integral state choice"
            )
        if result.fun is None or abs(float(result.fun) - rotation) > 1e-6:
            raise CandidateAllocationError(
                "candidate-allocation MILP objective disagrees with its state choice"
            )
        selected.append(rotation)
        lower[start:start + CANDIDATE_COUNT] = 0.0
        upper[start:start + CANDIDATE_COUNT] = 0.0
        lower[start + rotation] = 1.0
        upper[start + rotation] = 1.0
    return selected


def _counts(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    result = [0] * CANDIDATE_COUNT
    for row in rows:
        for candidate in row["candidate_indices"]:
            result[int(candidate)] += 1
    return result


def _contingency_tables(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    families = sorted({str(row["family"]) for row in rows})
    goal_types = sorted({str(row["goal_type"]) for row in rows})

    def selected(**fields: str) -> list[Mapping[str, Any]]:
        return [
            row for row in rows
            if all(row[key] == value for key, value in fields.items())
        ]

    return {
        "global": _counts(rows),
        "split_role": [
            {"split_role": role, "counts": _counts(selected(split_role=role))}
            for role in SPLIT_ROLES
        ],
        "stratum": [
            {"stratum": stratum, "counts": _counts(selected(stratum=stratum))}
            for stratum in STRATA
        ],
        "family": [
            {"family": family, "counts": _counts(selected(family=family))}
            for family in families
        ],
        "family_stratum": [
            {
                "family": family, "stratum": stratum,
                "counts": _counts(selected(family=family, stratum=stratum)),
            }
            for family in families for stratum in STRATA
        ],
        "fit_family_stratum": [
            {
                "family": family, "stratum": stratum,
                "counts": _counts(selected(
                    family=family, stratum=stratum, split_role="fit"
                )),
            }
            for family in families for stratum in STRATA
        ],
        "calibration_stratum": [
            {
                "stratum": stratum,
                "counts": _counts(selected(stratum=stratum, split_role="calibration")),
            }
            for stratum in STRATA
        ],
        "calibration_family": [
            {
                "family": family,
                "counts": _counts(selected(family=family, split_role="calibration")),
            }
            for family in families
        ],
        "goal_type": [
            {
                "goal_type": goal_type,
                "state_count": len(selected(goal_type=goal_type)),
                "counts": _counts(selected(goal_type=goal_type)),
            }
            for goal_type in goal_types
        ],
    }


def _post_identity_pre_outcome_validation(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Materialise every allocation check once actual identities are known."""

    tables = _contingency_tables(rows)
    subset_catalogue = _subset_catalogue()
    subset_usage = []
    for catalogue_row in subset_catalogue:
        rotation = int(catalogue_row["rotation_index"])
        subset_usage.append({
            **catalogue_row,
            "assigned_state_subset_count": sum(
                int(row["rotation_index"]) == rotation for row in rows
            ),
        })
    reverse_state_ids = sorted(
        str(row["state_id"])
        for row in rows
        if REVERSING_CANDIDATE in row["candidate_indices"]
    )
    goal_rows: list[dict[str, Any]] = []
    for row in tables["goal_type"]:
        state_count = int(row["state_count"])
        low, high = state_count // 2, (state_count + 1) // 2
        counts = [int(value) for value in row["counts"]]
        allowed = all(value in (low, high) for value in counts)
        odd_exact = (
            state_count % 2 == 0
            or sorted(counts) == [low] * 6 + [high] * 6
        )
        goal_rows.append({
            "goal_type": row["goal_type"],
            "state_count": state_count,
            "candidate_counts": counts,
            "required_floor": low,
            "required_ceiling": high,
            "six_floor_six_ceiling_required": bool(state_count % 2),
            "status": "PASS" if allowed and odd_exact else "FAIL",
        })

    split = {row["split_role"]: row["counts"] for row in tables["split_role"]}
    validation: dict[str, Any] = {
        "schema": POST_IDENTITY_VALIDATION_SCHEMA,
        "status": "PASS_POST_IDENTITY_PRE_OUTCOME_ALLOCATION_VALIDATION",
        "allocation_contract_digest": allocation_contract_digest(),
        "allocation_amendment_digest": allocation_amendment_digest(),
        "candidate_bank_digest": CANDIDATE_BANK_DIGEST,
        "candidate_outcomes_consumed": False,
        "global": {
            "state_subset_count": len(rows),
            "candidate_slot_count": sum(
                len(row["candidate_indices"]) for row in rows
            ),
            "candidate_counts": tables["global"],
        },
        "split_role": tables["split_role"],
        "strata": tables["stratum"],
        "families": tables["family"],
        "family_strata": tables["family_stratum"],
        "fit_family_strata": tables["fit_family_stratum"],
        "calibration_strata": tables["calibration_stratum"],
        "calibration_families": tables["calibration_family"],
        "actual_subset_usage": subset_usage,
        "actual_pairwise_cooccurrence": {
            "definition": (
                "diagonal=candidate incidence across 120 assigned state subsets; "
                "off-diagonal=number of assigned state subsets containing both "
                "candidates"
            ),
            "candidate_order": list(range(CANDIDATE_COUNT)),
            "matrix": _pairwise_cooccurrence([
                row["candidate_indices"] for row in rows
            ]),
        },
        "reverse_coverage": {
            "reversing_candidate_index": REVERSING_CANDIDATE,
            "required_distinct_state_subset_count": 60,
            "observed_distinct_state_subset_count": len(set(reverse_state_ids)),
            "reverse_state_ids": reverse_state_ids,
            "reverse_state_ids_digest": _sha256(reverse_state_ids),
            "per_subset_reverse_required": False,
            "status": "PASS" if len(set(reverse_state_ids)) == 60 else "FAIL",
        },
        "goal_type_validation": {
            "status": (
                "PASS_ACTUAL_POST_IDENTITY_PRE_OUTCOME_GOAL_TYPE_BALANCE"
                if all(row["status"] == "PASS" for row in goal_rows)
                else "FAIL_ACTUAL_POST_IDENTITY_PRE_OUTCOME_GOAL_TYPE_BALANCE"
            ),
            "actual_goal_types_observed": True,
            "rule": allocation_amendment_contract()["goal_type_boundary"][
                "mandatory_post_identity_pre_outcome_rule"
            ],
            "rows": goal_rows,
            "candidate_outcomes_consumed": False,
        },
        "checks": {
            "state_subset_count_is_120": len(rows) == 120,
            "candidate_slot_count_is_720": sum(
                len(row["candidate_indices"]) for row in rows
            ) == 720,
            "global_candidate_counts_are_exactly_60":
                tables["global"] == [60] * CANDIDATE_COUNT,
            "fit_candidate_counts_are_exactly_48":
                split.get("fit") == [48] * CANDIDATE_COUNT,
            "calibration_candidate_counts_are_exactly_12":
                split.get("calibration") == [12] * CANDIDATE_COUNT,
            "stratum_candidate_counts_are_exactly_20": all(
                row["counts"] == [20] * CANDIDATE_COUNT
                for row in tables["stratum"]
            ),
            "family_candidate_counts_are_integer_balanced_7_8": all(
                sorted(row["counts"]) == [7] * 6 + [8] * 6
                for row in tables["family"]
            ),
            "every_assigned_subset_has_forward": all(
                FORWARD_CANDIDATES.intersection(row["candidate_indices"])
                for row in rows
            ),
            "every_assigned_subset_has_turning": all(
                TURNING_CANDIDATES.intersection(row["candidate_indices"])
                for row in rows
            ),
            "reverse_occurs_in_exactly_60_distinct_state_subsets":
                len(set(reverse_state_ids)) == 60,
            "goal_type_balance_passes_on_actual_identities": all(
                row["status"] == "PASS" for row in goal_rows
            ),
        },
    }
    if not all(validation["checks"].values()):
        failed = sorted(
            key for key, passed in validation["checks"].items() if not passed
        )
        raise CandidateAllocationError(
            f"post-identity/pre-outcome allocation validation failed: {failed}"
        )
    validation["post_identity_validation_digest"] = _sha256(validation)
    return validation


def allocation_manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Compute the canonical digest, excluding the digest field itself."""

    return _sha256({
        key: value for key, value in manifest.items()
        if key != "allocation_manifest_digest"
    })


def build_allocation_manifest(
    states: Sequence[Mapping[str, Any]], *, source_identity_manifest_digest: str,
) -> dict[str, Any]:
    """Allocate candidates using only frozen, snapshot-time identity metadata."""

    if not _is_digest(source_identity_manifest_digest):
        raise CandidateAllocationError(
            "source_identity_manifest_digest must be a lowercase SHA-256"
        )
    normalised = _normalise_identity_states(states)
    rotations = _lexicographic_rotations(normalised)
    assignments: list[dict[str, Any]] = []
    for state, rotation in zip(normalised, rotations, strict=True):
        assignments.append({
            **state,
            "rotation_index": rotation,
            "candidate_indices": list(candidate_block(rotation)),
        })
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "status": STATUS,
        "source_identity_manifest_digest": source_identity_manifest_digest,
        "pre_outcome_identity_digest": pre_outcome_identity_digest(normalised),
        "allocation_contract": algorithm_contract(),
        "allocation_contract_digest": allocation_contract_digest(),
        "allocation_amendment": allocation_amendment_contract(),
        "allocation_amendment_digest": allocation_amendment_digest(),
        "assignments": assignments,
        "contingency_tables": _contingency_tables(assignments),
        "post_identity_pre_outcome_validation":
            _post_identity_pre_outcome_validation(assignments),
    }
    manifest["allocation_manifest_digest"] = allocation_manifest_digest(manifest)
    validate_allocation_manifest(
        manifest,
        expected_source_identity_manifest_digest=source_identity_manifest_digest,
    )
    return manifest


def _validate_counts(manifest: Mapping[str, Any]) -> None:
    rows = manifest["assignments"]
    tables = manifest["contingency_tables"]
    expected_tables = _contingency_tables(rows)
    if tables != expected_tables:
        raise CandidateAllocationError("contingency_tables do not match assignments")

    if tables["global"] != [60] * CANDIDATE_COUNT:
        raise CandidateAllocationError("every candidate must appear exactly 60 times")
    split = {row["split_role"]: row["counts"] for row in tables["split_role"]}
    if split != {"fit": [48] * CANDIDATE_COUNT,
                 "calibration": [12] * CANDIDATE_COUNT}:
        raise CandidateAllocationError("fit/calibration candidate balance is invalid")
    if any(row["counts"] != [20] * CANDIDATE_COUNT for row in tables["stratum"]):
        raise CandidateAllocationError("each stratum must contain every candidate 20 times")
    for row in tables["family"]:
        if sorted(row["counts"]) != [7] * 6 + [8] * 6:
            raise CandidateAllocationError(
                f"family {row['family']!r} is not integer-balanced 7/8"
            )
    for row in tables["family_stratum"]:
        if sorted(row["counts"]) != [2] * 6 + [3] * 6:
            raise CandidateAllocationError(
                f"family/stratum {row['family']!r}/{row['stratum']!r} is not 2/3 balanced"
            )
    if any(row["counts"] != [2] * CANDIDATE_COUNT
           for row in tables["fit_family_stratum"]):
        raise CandidateAllocationError("fit family/stratum margin is not exactly 2")
    if any(row["counts"] != [4] * CANDIDATE_COUNT
           for row in tables["calibration_stratum"]):
        raise CandidateAllocationError("calibration stratum margin is not exactly 4")
    for row in tables["calibration_family"]:
        if sorted(row["counts"]) != [1] * 6 + [2] * 6:
            raise CandidateAllocationError(
                f"calibration family {row['family']!r} is not 1/2 balanced"
            )
    for row in tables["goal_type"]:
        count = int(row["state_count"])
        low, high = count // 2, (count + 1) // 2
        if any(value not in (low, high) for value in row["counts"]):
            raise CandidateAllocationError(
                f"goal type {row['goal_type']!r} is not integer-balanced"
            )
        if count % 2 and sorted(row["counts"]) != [low] * 6 + [high] * 6:
            raise CandidateAllocationError(
                f"odd-sized goal type {row['goal_type']!r} lacks six floor/six ceil counts"
            )


def validate_allocation_manifest(
    manifest: Mapping[str, Any], *,
    expected_source_identity_manifest_digest: str | None = None,
) -> None:
    """Strictly validate identity binding, blocks, margins, and canonical digest."""

    if not isinstance(manifest, Mapping):
        raise CandidateAllocationError("allocation manifest must be a mapping")
    expected_keys = {
        "schema", "status", "source_identity_manifest_digest",
        "pre_outcome_identity_digest", "allocation_contract",
        "allocation_contract_digest", "allocation_amendment",
        "allocation_amendment_digest", "assignments", "contingency_tables",
        "post_identity_pre_outcome_validation", "allocation_manifest_digest",
    }
    if set(manifest) != expected_keys:
        raise CandidateAllocationError("allocation manifest has unexpected or missing keys")
    if manifest["schema"] != SCHEMA or manifest["status"] != STATUS:
        raise CandidateAllocationError("allocation manifest schema/status mismatch")
    source_digest = manifest["source_identity_manifest_digest"]
    if not _is_digest(source_digest):
        raise CandidateAllocationError("invalid source identity manifest digest")
    if (expected_source_identity_manifest_digest is not None
            and source_digest != expected_source_identity_manifest_digest):
        raise CandidateAllocationError("source identity manifest digest mismatch")
    if manifest["allocation_contract"] != algorithm_contract():
        raise CandidateAllocationError("allocation contract differs from frozen v1.2")
    if manifest["allocation_contract_digest"] != allocation_contract_digest():
        raise CandidateAllocationError("allocation contract digest mismatch")
    if manifest["allocation_amendment"] != allocation_amendment_contract():
        raise CandidateAllocationError("allocation amendment differs from frozen v1")
    if manifest["allocation_amendment_digest"] != allocation_amendment_digest():
        raise CandidateAllocationError("allocation amendment digest mismatch")

    raw_assignments = manifest["assignments"]
    if not isinstance(raw_assignments, list) or len(raw_assignments) != 120:
        raise CandidateAllocationError("assignments must contain exactly 120 rows")
    identity_rows: list[dict[str, str]] = []
    previous_key: tuple[str, str] | None = None
    for row in raw_assignments:
        if not isinstance(row, Mapping) or frozenset(row) != _ASSIGNMENT_KEYS:
            raise CandidateAllocationError("assignment row has unexpected or missing keys")
        identity = _normalise_identity_state({key: row[key] for key in _INPUT_KEYS})
        key = (identity["state_identity_digest"], identity["state_id"])
        if previous_key is not None and key <= previous_key:
            raise CandidateAllocationError("assignments are not in canonical identity order")
        previous_key = key
        identity_rows.append(identity)
        rotation = row["rotation_index"]
        if isinstance(rotation, bool) or not isinstance(rotation, int):
            raise CandidateAllocationError("rotation_index must be an integer")
        candidates = row["candidate_indices"]
        if not isinstance(candidates, list) or candidates != list(candidate_block(rotation)):
            raise CandidateAllocationError("candidate_indices do not match frozen rotation")
        if not FORWARD_CANDIDATES.intersection(candidates):
            raise CandidateAllocationError("candidate block lacks a forward candidate")
        if not TURNING_CANDIDATES.intersection(candidates):
            raise CandidateAllocationError("candidate block lacks a turning candidate")

    normalised = _normalise_identity_states(identity_rows)
    expected_identity_digest = pre_outcome_identity_digest(normalised)
    if manifest["pre_outcome_identity_digest"] != expected_identity_digest:
        raise CandidateAllocationError("pre-outcome identity projection digest mismatch")
    _validate_counts(manifest)
    expected_validation = _post_identity_pre_outcome_validation(raw_assignments)
    if manifest["post_identity_pre_outcome_validation"] != expected_validation:
        raise CandidateAllocationError(
            "post-identity/pre-outcome validation does not match assignments"
        )

    # Balance alone is insufficient: many complementary allocations have the
    # same contingency tables.  Re-solving from the identity projection makes
    # the validator enforce the one canonical lexicographic assignment rather
    # than accepting any rehashed feasible substitute.
    expected_rotations = _lexicographic_rotations(normalised)
    actual_rotations = [int(row["rotation_index"]) for row in raw_assignments]
    if actual_rotations != expected_rotations:
        raise CandidateAllocationError(
            "rotation vector is feasible but is not the canonical lexicographic allocation"
        )

    reverse_state_ids = {
        str(row["state_id"]) for row in raw_assignments
        if REVERSING_CANDIDATE in row["candidate_indices"]
    }
    if len(reverse_state_ids) != 60:
        raise CandidateAllocationError(
            "the sole reversing candidate must occur in exactly 60 distinct "
            "state subsets"
        )
    expected_digest = allocation_manifest_digest(manifest)
    if manifest["allocation_manifest_digest"] != expected_digest:
        raise CandidateAllocationError("allocation manifest digest mismatch")
