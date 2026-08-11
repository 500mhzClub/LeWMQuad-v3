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
GOAL_TYPE_VERSION = "snapshot_bound_landmark_material_id_v1"
GOAL_TYPE_DEFINITION = (
    "the material_id of the landmark bound at snapshot time, before any "
    "candidate branch is executed"
)
CANDIDATE_BANK_DIGEST = (
    "85471e44a0fe8f3c59fff258e9b23933e306f69b6d590c832e2b8da1f34a8cd9"
)

CANDIDATE_COUNT = 12
CANDIDATES_PER_STATE = 6
ROTATION_OFFSETS = (0, 1, 3, 5, 8, 10)
STRATA = ("general", "safety_enriched", "completion_enriched")
SPLIT_ROLES = ("fit", "calibration")
FORWARD_CANDIDATES = frozenset((0, 1, 2))
TURNING_CANDIDATES = frozenset((3, 4, 5, 6, 7, 8, 9))
REVERSING_CANDIDATE = 10

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
        "assignments": assignments,
        "contingency_tables": _contingency_tables(assignments),
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
        "allocation_contract_digest", "assignments", "contingency_tables",
        "allocation_manifest_digest",
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

    reverse_count = sum(
        REVERSING_CANDIDATE in row["candidate_indices"] for row in raw_assignments
    )
    if reverse_count != 60:
        raise CandidateAllocationError("the sole reversing candidate must occur 60 times")
    expected_digest = allocation_manifest_digest(manifest)
    if manifest["allocation_manifest_digest"] != expected_digest:
        raise CandidateAllocationError("allocation manifest digest mismatch")
