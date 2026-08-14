"""Deterministic one-model formulation for the pre-outcome small completion join.

The module has no repository or generated-artifact readers.  It translates a
strict, caller-supplied identity/eligibility instance into one binary MILP,
solves that model when explicitly requested, and validates every returned bit
against the frozen integer constraints.  It has no branch, outcome, scorer,
predictor, rendering, or checkpoint API.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
import sys
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lewm.oracle import (
    go2_small_completion_global_execution_amendment_v1 as AUTHORITY,
)


GENERIC_PROBLEM_SCHEMA = "go2_small_completion_global_exact_v1_binary_problem"
MODEL_SCHEMA = "go2_small_completion_global_exact_v1_binary_model"
SOLUTION_SCHEMA = "go2_small_completion_global_exact_v1_solution"
INFEASIBILITY_SCHEMA = "go2_small_completion_global_exact_v1_infeasibility"
PRODUCTION_INSTANCE_SCHEMA = (
    "go2_small_completion_global_exact_v1_production_instance"
)
PRODUCTION_MODEL_SCHEMA = (
    "go2_small_completion_global_exact_v1_production_model_binding"
)
ALLOCATION_RESULT_SCHEMA = (
    "go2_small_completion_global_exact_v1_materialized_allocation"
)
ALLOCATION_RESULT_DIGEST_KEY = "materialized_allocation_digest"
ALLOCATION_CONTRACT_DISPOSITION_SCHEMA = (
    "go2_small_completion_global_exact_v1_legacy_allocation_contract_disposition"
)
ALLOCATION_CONTRACT_DISPOSITION_SELF_KEY = (
    "legacy_allocation_contract_disposition_digest"
)
EXECUTION_PLAN_SCHEMA = "go2_small_completion_global_exact_v1_execution_plan"
EXECUTION_PLAN_DIGEST_KEY = "execution_plan_digest"
EXECUTION_RESULT_SCHEMA = "go2_small_completion_global_exact_v1_execution_result"
EXECUTION_RESULT_DIGEST_KEY = "execution_result_digest"
EXECUTION_PASS_STATUS = "PASS_EXACT_GLOBAL_ALLOCATION_FOUND"
EXECUTION_INFEASIBLE_STATUS = "EXACT_GLOBAL_ALLOCATION_INFEASIBLE"
FIXTURE_SUITE_SCHEMA = "go2_small_completion_global_exact_v1_fixture_suite"
FIXTURE_SUITE_DIGEST_KEY = "fixture_suite_digest"
FROZEN_FIXTURE_SUITE_RESULT_DIGEST = (
    "89be75f62f2c36e6d758f94236a19f7d8a83dd5b09cc019404ed23665e91926b"
)
STRUCTURAL_SCENE_IDENTITY_SCHEMA = (
    "go2_small_completion_global_exact_v1_structural_scene_identity"
)
STRUCTURAL_SCENE_IDENTITY_DOMAIN = (
    "LEWM_GO2_SMALL_COMPLETION_GLOBAL_EXACT_STRUCTURAL_SCENE_IDENTITY_V1"
)
STATE_IDENTITY_LINEAGE_SCHEMA = "go2_branch_state_identity_v1_2_lineage"
MODEL_DIGEST_KEY = "model_digest"
SOLUTION_DIGEST_KEY = "solution_digest"
INFEASIBILITY_DIGEST_KEY = "infeasibility_digest"

OBJECTIVE_CONTRACT = copy.deepcopy(AUTHORITY.STABLE_HASH_OBJECTIVE_CONTRACT)
OBJECTIVE_CONTRACT_DIGEST = (
    AUTHORITY.STABLE_HASH_OBJECTIVE_CONTRACT_DIGEST
)
OBJECTIVE_DOMAIN = OBJECTIVE_CONTRACT["domain_separation_utf8"]
SOLVER_CONTRACT: dict[str, Any] = {
    "implementation": "scipy.optimize.milp/HiGHS",
    "integrality": "all variables binary",
    "disp": False,
    "presolve": True,
    "mip_rel_gap": 0.0,
    "time_limit": 7_200.0,
    "threads": 1,
    "random_seed": 0,
    "search_branching_policy": "HiGHS_default_no_user_branching_override",
}
SOLVER_CONTRACT_DIGEST = hashlib.sha256(
    json.dumps(
        SOLVER_CONTRACT, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")
).hexdigest()

# This is the exact trusted-control interpreter/runtime that is authorised to
# create the one global solve.  Persisted plan/result validation deliberately
# compares against this immutable record rather than importing the local
# SciPy/HiGHS extension: downstream corpus consumers run in a separate Torch
# environment and must be able to reopen the scientific certificate without
# acquiring solver authority.
FROZEN_SOLVER_RUNTIME_IDENTITY: dict[str, Any] = {
    "schema": "go2_small_completion_global_exact_v1_solver_runtime_identity",
    "python_implementation": "CPython",
    "python_version": "3.12.3",
    "python_cache_tag": "cpython-312",
    "numpy_version": "1.26.4",
    "scipy_version": "1.11.4",
    "scipy_distribution_version": "1.11.4",
    "highs_binding": "scipy.optimize._highs._highs_wrapper",
    "solver_version": (
        "SciPy-1.11.4-bundled-HiGHS-extension-sha256:"
        "de4128ae93dd54cd1338c3d32c9a5bde23bda75967c73a3925f40126df6c22a3"
    ),
    "highs_extension_filename": (
        "_highs_wrapper.cpython-312-x86_64-linux-gnu.so"
    ),
    "highs_extension_sha256": (
        "de4128ae93dd54cd1338c3d32c9a5bde23bda75967c73a3925f40126df6c22a3"
    ),
    "platform_system": "Linux",
    "platform_release": "7.0.11-76070011-generic",
    "platform_machine": "x86_64",
    "solver_contract_digest": SOLVER_CONTRACT_DIGEST,
    "solver_runtime_identity_digest": (
        "3bc9bd4ef87704c48e28cdbc2176c3a636beb3c745b5fcf96ef8f9147e2b4be9"
    ),
}

CANDIDATE_COUNT = 12
CANDIDATES_PER_STATE = 6
ROTATION_OFFSETS = (0, 1, 3, 5, 8, 10)
ROTATION_BLOCKS = tuple(tuple(sorted(
    (rotation + offset) % CANDIDATE_COUNT for offset in ROTATION_OFFSETS
)) for rotation in range(CANDIDATE_COUNT))
FAMILIES = (
    "large_enclosed_maze", "local_composite_motifs", "loop_alias_stress",
    "medium_enclosed_maze", "open_obstacle_field", "rough_local_dynamics",
    "small_enclosed_maze", "visual_sensor_stress",
)
STRATA = ("general", "safety_enriched", "completion_enriched")
SPLIT_ROLES = ("fit", "calibration")
SMALL_FAMILY = "small_enclosed_maze"
COMPLETION_STRATUM = "completion_enriched"
FORWARD_CANDIDATES = frozenset((0, 1, 2))
TURNING_CANDIDATES = frozenset((3, 4, 5, 6, 7, 8, 9))

_HEX = frozenset("0123456789abcdef")
_FIXED_KEYS = frozenset({
    "state_id", "state_identity_digest", "scene_id", "family", "stratum", "split_role",
    "goal_type", "completion_rotation_eligibility_owner_digest",
    "completion_rotation_eligibility", "completion_rotation_evidence",
})
_OPTIONAL_KEYS = frozenset({
    "scene_id", "structural_scene_identity_digest", "goal_type",
    "structural_scene_projection",
    "completion_rotation_eligibility_owner_digest",
    "completion_rotation_eligibility",
})
_DEFERRED_SCENE_FIELDS = frozenset({
    "state_id", "state_identity_digest", "state_index", "split_role",
    "candidate_indices", "candidate_rotation_index", "rotation_index",
    "branch_identities",
})
_RAW_OPTIONAL_CANDIDATE_KEYS = frozenset({
    "state_id", "family", "scene_id", "scene_dir", "scene_manifest_sha256",
    "scene_manifest_byte_count", "split", "drive_seed", "stratum",
    "split_role", "warmup_blocks", "source_step", "episode_id",
    "episode_cluster_id", "cell_id", "boundary", "goal", "goal_type",
    "body_clearance_m", "clearance_m",
    "completion_rotation_eligibility_vector", "snapshot_task_status",
    "previous_applied_command",
})
_RAW_OPTIONAL_STRUCTURAL_KEYS = (
    _RAW_OPTIONAL_CANDIDATE_KEYS - {"state_id", "split_role"}
)
_GOAL_KEYS = frozenset({
    "landmark_id", "landmark_cell", "material_id", "graph_edges",
    "start_geodesic_m", "bearing_body_rad", "range_m", "landmark_xy_m",
})
_BOUNDARY_KEYS = frozenset({"source_step", "boundary_digest"})
_IDENTITY_LINEAGE_KEYS = frozenset({
    "schema", "selection_digest", "scorer_contract_v1_2_digest", "pool",
    "pre_allocation_identity_static",
})
_PRE_ALLOCATION_COMMON_KEYS = (
    "selection_digest", "scorer_fit_allocation_design_digest",
    "candidate_allocator_contract_digest",
    "candidate_allocation_amendment_digest",
    "pre_identity_allocation_validation_digest",
    "invalid_scorer_identity_exclusion_digest",
    "state_selector_amendment_digest",
    "state_selector_feasibility_receipt_digest", "candidate_bank_digest",
    "clean_source_launch_receipt_digest", "source_repository_commit",
    "clean_source_binding_digest", "bound_implementations_digest",
    "scorer_contract_artifact_digest",
    "mixed_precontract_disposition_receipt_digest",
    "progress_contract_digest", "safety_contract_digest",
    "oracle_v1_2_digest", "scorer_contract_v1_2_digest", "boundary_digest",
    "render_contract_digest", "preprocess_contract_digest",
    "textured_v03_renderer_contract_digest", "preprocessing_digest",
    "target_encoder_digest", "target_encoder_checkpoint_sha256",
    "genesis_backend",
)
_SCORER_FIT_POOL_SPEC = {
    "states_per_family": 15,
    "candidates_per_state": 6,
    "strata": {
        "general": 5, "safety_enriched": 5, "completion_enriched": 5,
    },
    "calibration_per_stratum_per_family": 1,
}


class GlobalExactModelError(RuntimeError):
    """An instance, model, solver result, or receipt is not exact."""


class GlobalExactInfeasible(GlobalExactModelError):
    """Raised only by :func:`require_solution` for an exact infeasibility."""


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GlobalExactModelError("value is not canonical JSON") from exc


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _legacy_builder_identity_digest(value: Any) -> str:
    """Match the frozen corpus builder's pre-allocation digest convention.

    The candidate-allocation artifact itself uses compact canonical JSON, but
    its externally supplied ``source_identity_manifest_digest`` was frozen by
    the corpus builder with ``json.dumps(sort_keys=True)`` and default
    separators.  Keeping this convention distinct prevents a new execution
    backend from silently creating a different identity namespace.
    """

    try:
        encoded = json.dumps(
            value, sort_keys=True, ensure_ascii=True, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GlobalExactModelError(
            "legacy source identity is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


if canonical_digest(OBJECTIVE_CONTRACT) != OBJECTIVE_CONTRACT_DIGEST:
    raise RuntimeError("authority objective contract digest changed")


def validate_solver_runtime_identity_record(value: Mapping[str, Any]
                                            ) -> dict[str, Any]:
    """Validate the frozen solver identity without importing solver code."""

    if not isinstance(value, Mapping):
        raise GlobalExactModelError("solver runtime identity is not a mapping")
    payload = dict(value)
    if (set(payload) != set(FROZEN_SOLVER_RUNTIME_IDENTITY)
            or payload.get("solver_runtime_identity_digest")
            != canonical_digest(_without(
                payload, "solver_runtime_identity_digest"))
            or payload != FROZEN_SOLVER_RUNTIME_IDENTITY):
        raise GlobalExactModelError("solver runtime identity changed")
    return copy.deepcopy(FROZEN_SOLVER_RUNTIME_IDENTITY)


def _assert_frozen_contracts() -> None:
    if (canonical_digest(OBJECTIVE_CONTRACT) != OBJECTIVE_CONTRACT_DIGEST
            or canonical_digest(SOLVER_CONTRACT) != SOLVER_CONTRACT_DIGEST
            or FROZEN_SOLVER_RUNTIME_IDENTITY.get(
                "solver_runtime_identity_digest")
            != canonical_digest(_without(
                FROZEN_SOLVER_RUNTIME_IDENTITY,
                "solver_runtime_identity_digest"))):
        raise GlobalExactModelError("frozen model contract was mutated")


def solver_runtime_identity() -> dict[str, Any]:
    """Return the exact local SciPy/HiGHS runtime identity frozen pre-solve."""

    _assert_frozen_contracts()
    try:
        import numpy as np
        import scipy
        from scipy.optimize._highs import _highs_wrapper
    except ImportError as exc:  # pragma: no cover
        raise GlobalExactModelError("SciPy MILP runtime is unavailable") from exc
    extension_path = Path(str(_highs_wrapper.__file__))
    try:
        extension_digest = hashlib.sha256(extension_path.read_bytes()).hexdigest()
    except OSError as exc:  # pragma: no cover
        raise GlobalExactModelError("cannot bind the HiGHS extension binary") from exc
    identity = {
        "schema": "go2_small_completion_global_exact_v1_solver_runtime_identity",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_cache_tag": str(sys.implementation.cache_tag),
        "numpy_version": str(np.__version__),
        "scipy_version": str(scipy.__version__),
        "scipy_distribution_version": importlib.metadata.version("scipy"),
        "highs_binding": "scipy.optimize._highs._highs_wrapper",
        "solver_version": (
            f"SciPy-{scipy.__version__}-bundled-HiGHS-extension-"
            f"sha256:{extension_digest}"
        ),
        "highs_extension_filename": extension_path.name,
        "highs_extension_sha256": extension_digest,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "solver_contract_digest": SOLVER_CONTRACT_DIGEST,
    }
    live = _signed(identity, "solver_runtime_identity_digest")
    validate_solver_runtime_identity_record(live)
    return live


def _bound_solver_runtime_identity(
        value: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if value is None:
        return solver_runtime_identity()
    return validate_solver_runtime_identity_record(value)


def _is_digest(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in _HEX for character in value))


def _is_finite_number(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(float(value)))


def _signed(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(payload)
    if key in result:
        raise GlobalExactModelError("digest key already exists")
    result[key] = canonical_digest(result)
    return result


def _without(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    return {name: value for name, value in payload.items() if name != key}


def _bound(value: Any, *, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise GlobalExactModelError(f"{label} must be an integer or null")
    return value


def pair_objective_binding(pair_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Return the authority-exact pair preimage, digest and coefficient."""

    _assert_frozen_contracts()
    if not isinstance(pair_identity, Mapping):
        raise GlobalExactModelError("pair identity is not a mapping")
    pair = dict(pair_identity)
    kind = pair.get("kind")
    common = {"candidate_rotation_index", "candidate_indices"}
    if kind == "fixed_state":
        expected = common | {
            "kind", "state_identity_digest", "split_role",
        }
        identity = pair.get("state_identity_digest")
        role_key = "split_role"
    elif kind == "selectable_completion":
        expected = common | {
            "kind", "structural_scene_identity_digest", "assigned_split_role",
        }
        identity = pair.get("structural_scene_identity_digest")
        role_key = "assigned_split_role"
    else:
        raise GlobalExactModelError("pair identity kind changed")
    rotation = pair.get("candidate_rotation_index")
    if (set(pair) != expected or not _is_digest(identity)
            or pair.get(role_key) not in SPLIT_ROLES
            or isinstance(rotation, bool) or not isinstance(rotation, int)
            or not 0 <= rotation < CANDIDATE_COUNT
            or pair.get("candidate_indices") != list(ROTATION_BLOCKS[rotation])):
        raise GlobalExactModelError("pair identity fields changed")
    pair_json = _json_bytes(pair)
    preimage = OBJECTIVE_DOMAIN.encode("utf-8") + b"\x00" + pair_json
    digest = hashlib.sha256(preimage).hexdigest()
    coefficient = 1 + int(digest[:10], 16)
    return {
        "pair_identity": pair,
        "canonical_pair_identity_json": pair_json.decode("ascii"),
        "pair_digest": digest,
        "objective_coefficient": coefficient,
    }


def translate_binary_problem(
        problem: Mapping[str, Any], *,
        _solver_runtime_identity: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
    """Canonicalise a binary problem under the authority-exact pair objective."""

    runtime_identity = _bound_solver_runtime_identity(
        _solver_runtime_identity)

    if not isinstance(problem, Mapping) or set(problem) != {
            "schema", "variables", "constraints", "metadata"}:
        raise GlobalExactModelError("binary problem key surface changed")
    if problem.get("schema") != GENERIC_PROBLEM_SCHEMA:
        raise GlobalExactModelError("binary problem schema changed")
    raw_variables = problem.get("variables")
    if not isinstance(raw_variables, list) or not raw_variables:
        raise GlobalExactModelError("binary variable keys are malformed")
    bound_variables: list[dict[str, Any]] = []
    for raw in raw_variables:
        if not isinstance(raw, Mapping) or set(raw) != {"key", "pair_identity"}:
            raise GlobalExactModelError("binary variable surface changed")
        key = raw.get("key")
        if not isinstance(key, str) or not key:
            raise GlobalExactModelError("binary variable key is malformed")
        binding = pair_objective_binding(raw["pair_identity"])
        bound_variables.append({"key": key, **binding})
    keys = [row["key"] for row in bound_variables]
    if len(set(keys)) != len(keys):
        raise GlobalExactModelError("binary variable keys are not unique")
    bound_variables.sort(key=lambda row: (
        row["pair_digest"], row["canonical_pair_identity_json"]))
    # Distinct variables must be distinct scientific pair identities.  A full
    # SHA collision is ordered by canonical bytes, exactly as the authority says.
    pair_json_rows = [row["canonical_pair_identity_json"] for row in bound_variables]
    if len(set(pair_json_rows)) != len(pair_json_rows):
        raise GlobalExactModelError("duplicate pair identity variable")
    index = {row["key"]: ordinal for ordinal, row in enumerate(bound_variables)}
    variables = [{
        "index": index[row["key"]],
        "key": row["key"],
        "pair_identity": row["pair_identity"],
        "canonical_pair_identity_json": row["canonical_pair_identity_json"],
        "pair_digest": row["pair_digest"],
        "objective_coefficient": row["objective_coefficient"],
        "lower": 0,
        "upper": 1,
        "integrality": 1,
    } for row in bound_variables]

    raw_constraints = problem.get("constraints")
    if not isinstance(raw_constraints, list):
        raise GlobalExactModelError("binary constraints are not a list")
    names: set[str] = set()
    constraints: list[dict[str, Any]] = []
    for raw in raw_constraints:
        if not isinstance(raw, Mapping) or set(raw) != {
                "name", "terms", "lower", "upper"}:
            raise GlobalExactModelError("binary constraint surface changed")
        name = raw.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise GlobalExactModelError("binary constraint name is malformed")
        names.add(name)
        lower = _bound(raw.get("lower"), label=f"{name} lower")
        upper = _bound(raw.get("upper"), label=f"{name} upper")
        if lower is not None and upper is not None and lower > upper:
            raise GlobalExactModelError("binary constraint bounds are reversed")
        raw_terms = raw.get("terms")
        if not isinstance(raw_terms, list) or not raw_terms:
            raise GlobalExactModelError("binary constraint has no terms")
        combined: dict[int, int] = {}
        for term in raw_terms:
            if (not isinstance(term, list) or len(term) != 2
                    or term[0] not in index or isinstance(term[1], bool)
                    or not isinstance(term[1], int) or term[1] == 0):
                raise GlobalExactModelError("binary constraint term is malformed")
            variable_index = index[term[0]]
            combined[variable_index] = combined.get(variable_index, 0) + term[1]
        terms = [[variable, coefficient]
                 for variable, coefficient in sorted(combined.items())
                 if coefficient]
        if not terms:
            raise GlobalExactModelError("binary constraint cancels to zero")
        constraints.append({
            "name": name, "lower": lower, "upper": upper, "terms": terms,
        })
    constraints.sort(key=lambda row: row["name"])
    metadata = problem.get("metadata")
    if not isinstance(metadata, Mapping):
        raise GlobalExactModelError("binary problem metadata is malformed")
    payload = {
        "schema": MODEL_SCHEMA,
        "problem_digest": canonical_digest(dict(problem)),
        "canonical_problem": json.loads(_json_bytes(dict(problem))),
        "objective_contract": dict(OBJECTIVE_CONTRACT),
        "objective_contract_digest": OBJECTIVE_CONTRACT_DIGEST,
        "objective_rule": OBJECTIVE_CONTRACT["linear_objective"],
        "variable_order": OBJECTIVE_CONTRACT["variable_order"],
        "constraint_order": "lexicographic_constraint_name",
        "variables": variables,
        "constraints": constraints,
        "metadata": dict(metadata),
        "solver_contract": dict(SOLVER_CONTRACT),
        "solver_contract_digest": SOLVER_CONTRACT_DIGEST,
        "solver_runtime_identity": runtime_identity,
        "candidate_outcomes_consumed": False,
        "scientific_masks_are_input_bounds_only": True,
    }
    return _signed(payload, MODEL_DIGEST_KEY)


def validate_model(
        model: Mapping[str, Any], *,
        _solver_runtime_identity: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
    runtime_identity = _bound_solver_runtime_identity(
        _solver_runtime_identity)
    if not isinstance(model, Mapping):
        raise GlobalExactModelError("model is not a mapping")
    payload = dict(model)
    if (set(payload) != {
            "schema", "problem_digest", "canonical_problem", "objective_contract",
            "objective_contract_digest", "objective_rule", "variable_order",
            "constraint_order", "variables", "constraints", "metadata",
            "solver_contract", "solver_contract_digest",
            "solver_runtime_identity", "candidate_outcomes_consumed",
            "scientific_masks_are_input_bounds_only", MODEL_DIGEST_KEY}
            or payload.get("schema") != MODEL_SCHEMA
            or not _is_digest(payload.get("problem_digest"))
            or payload.get("objective_contract") != OBJECTIVE_CONTRACT
            or payload.get("objective_contract_digest")
            != OBJECTIVE_CONTRACT_DIGEST
            or payload.get("objective_rule")
            != OBJECTIVE_CONTRACT["linear_objective"]
            or payload.get("variable_order")
            != OBJECTIVE_CONTRACT["variable_order"]
            or payload.get("constraint_order")
            != "lexicographic_constraint_name"
            or payload.get("solver_contract") != SOLVER_CONTRACT
            or payload.get("solver_contract_digest") != SOLVER_CONTRACT_DIGEST
            or payload.get("solver_runtime_identity") != runtime_identity
            or payload.get("candidate_outcomes_consumed") is not False
            or payload.get("scientific_masks_are_input_bounds_only") is not True
            or payload.get(MODEL_DIGEST_KEY)
            != canonical_digest(_without(payload, MODEL_DIGEST_KEY))):
        raise GlobalExactModelError("model binding changed")
    variables = payload.get("variables")
    constraints = payload.get("constraints")
    if not isinstance(variables, list) or not variables:
        raise GlobalExactModelError("model order changed")
    variable_keys: set[str] = set()
    pair_json_rows: set[str] = set()
    observed_order: list[tuple[str, str]] = []
    expected_variable_keys = {
        "index", "key", "pair_identity", "canonical_pair_identity_json",
        "pair_digest", "objective_coefficient", "lower", "upper",
        "integrality",
    }
    for expected_index, row in enumerate(variables):
        if not isinstance(row, Mapping) or set(row) != expected_variable_keys:
            raise GlobalExactModelError("model variable surface changed")
        key = row.get("key")
        if (not isinstance(key, str) or not key or key in variable_keys
                or row.get("index") != expected_index
                or row.get("lower") != 0 or row.get("upper") != 1
                or row.get("integrality") != 1):
            raise GlobalExactModelError("model variable binding changed")
        variable_keys.add(key)
        binding = pair_objective_binding(row.get("pair_identity"))
        if any(row.get(name) != binding[name] for name in (
                "canonical_pair_identity_json", "pair_digest",
                "objective_coefficient")):
            raise GlobalExactModelError("model variable objective changed")
        pair_json = binding["canonical_pair_identity_json"]
        if pair_json in pair_json_rows:
            raise GlobalExactModelError("model pair identity is duplicated")
        pair_json_rows.add(pair_json)
        observed_order.append((binding["pair_digest"], pair_json))
    if observed_order != sorted(observed_order):
        raise GlobalExactModelError("model variable order changed")
    if not isinstance(constraints, list):
        raise GlobalExactModelError("model constraints changed")
    constraint_names: list[str] = []
    for row in constraints:
        if not isinstance(row, Mapping) or set(row) != {
                "name", "lower", "upper", "terms"}:
            raise GlobalExactModelError("model constraint surface changed")
        name = row.get("name")
        lower = row.get("lower")
        upper = row.get("upper")
        terms = row.get("terms")
        if (not isinstance(name, str) or not name
                or name in constraint_names
                or lower is not None and (isinstance(lower, bool)
                                          or not isinstance(lower, int))
                or upper is not None and (isinstance(upper, bool)
                                          or not isinstance(upper, int))
                or lower is not None and upper is not None and lower > upper
                or not isinstance(terms, list) or not terms):
            raise GlobalExactModelError("model constraint binding changed")
        constraint_names.append(name)
        previous_index = -1
        for term in terms:
            if (not isinstance(term, list) or len(term) != 2
                    or isinstance(term[0], bool) or not isinstance(term[0], int)
                    or term[0] <= previous_index or term[0] >= len(variables)
                    or isinstance(term[1], bool) or not isinstance(term[1], int)
                    or term[1] == 0):
                raise GlobalExactModelError("model constraint term changed")
            previous_index = term[0]
    if constraint_names != sorted(constraint_names):
        raise GlobalExactModelError("model constraint order changed")
    if not isinstance(payload.get("metadata"), Mapping):
        raise GlobalExactModelError("model metadata changed")
    canonical_problem = payload.get("canonical_problem")
    if (not isinstance(canonical_problem, Mapping)
            or canonical_digest(canonical_problem) != payload["problem_digest"]
            or translate_binary_problem(
                canonical_problem,
                _solver_runtime_identity=runtime_identity) != payload):
        raise GlobalExactModelError("model differs from its canonical problem")
    return payload


def _constraint_activity(row: Mapping[str, Any], bits: Sequence[int]) -> int:
    return sum(int(coefficient) * int(bits[int(index)])
               for index, coefficient in row["terms"])


def _receipt_runtime_identity(
        solver: str, *,
        scipy_runtime_identity: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
    if solver == SOLVER_CONTRACT["implementation"]:
        return _bound_solver_runtime_identity(scipy_runtime_identity)
    if solver == "exhaustive_binary_control_v1":
        return _signed({
            "schema": "go2_small_completion_global_exact_v1_control_runtime",
            "implementation": solver,
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
        }, "solver_runtime_identity_digest")
    raise GlobalExactModelError("unrecognised exact solver identity")


def _build_solution(
        model: Mapping[str, Any], bits: Sequence[int], *, solver: str,
        _solver_runtime_identity: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
    runtime_identity = _bound_solver_runtime_identity(
        _solver_runtime_identity)
    bound = validate_model(
        model, _solver_runtime_identity=runtime_identity)
    values = list(bits)
    if (len(values) != len(bound["variables"])
            or any(isinstance(value, bool) or value not in (0, 1)
                   for value in values)):
        raise GlobalExactModelError("solution is not an exact binary vector")
    for row in bound["constraints"]:
        activity = _constraint_activity(row, values)
        if ((row["lower"] is not None and activity < row["lower"])
                or (row["upper"] is not None and activity > row["upper"])):
            raise GlobalExactModelError(
                f"solution violates constraint {row['name']}")
    objective = sum(
        int(row["objective_coefficient"]) * values[row["index"]]
        for row in bound["variables"])
    selected = [row["key"] for row in bound["variables"]
                if values[row["index"]] == 1]
    return _signed({
        "schema": SOLUTION_SCHEMA,
        "status": "PASS_EXACT_GLOBAL_BINARY_MODEL",
        "model_digest": bound[MODEL_DIGEST_KEY],
        "solver": solver,
        "solver_contract": dict(SOLVER_CONTRACT),
        "solver_contract_digest": SOLVER_CONTRACT_DIGEST,
        "solver_runtime_identity": _receipt_runtime_identity(
            solver, scipy_runtime_identity=runtime_identity),
        "solver_status": 0,
        "binary_values": values,
        "selected_variable_keys": selected,
        "objective_value": objective,
        "deterministic_optimal_objective_value": objective,
        "optimal_objective_bound": objective,
        "mip_gap": 0.0,
        "constraint_count": len(bound["constraints"]),
        "candidate_outcomes_consumed": False,
    }, SOLUTION_DIGEST_KEY)


def validate_solution(
        model: Mapping[str, Any], solution: Mapping[str, Any], *,
        _solver_runtime_identity: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
    runtime_identity = _bound_solver_runtime_identity(
        _solver_runtime_identity)
    if not isinstance(solution, Mapping):
        raise GlobalExactModelError("solution receipt is not a mapping")
    receipt = dict(solution)
    expected_keys = {
        "schema", "status", "model_digest", "solver", "solver_contract",
        "solver_contract_digest", "solver_runtime_identity", "solver_status",
        "binary_values", "selected_variable_keys", "objective_value",
        "deterministic_optimal_objective_value", "optimal_objective_bound",
        "mip_gap",
        "constraint_count", "candidate_outcomes_consumed", SOLUTION_DIGEST_KEY,
    }
    if (set(receipt) != expected_keys
            or receipt.get("schema") != SOLUTION_SCHEMA
            or receipt.get("status") != "PASS_EXACT_GLOBAL_BINARY_MODEL"
            or not isinstance(receipt.get("solver"), str)
            or receipt.get(SOLUTION_DIGEST_KEY)
            != canonical_digest(_without(receipt, SOLUTION_DIGEST_KEY))):
        raise GlobalExactModelError("solution receipt binding changed")
    rebuilt = _build_solution(
        model, receipt.get("binary_values", []), solver=receipt["solver"],
        _solver_runtime_identity=runtime_identity)
    if rebuilt != receipt:
        raise GlobalExactModelError("solution receipt is not exact")
    return receipt


def _build_infeasibility(
        model: Mapping[str, Any], *, solver: str,
        _solver_runtime_identity: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
    runtime_identity = _bound_solver_runtime_identity(
        _solver_runtime_identity)
    bound = validate_model(
        model, _solver_runtime_identity=runtime_identity)
    return _signed({
        "schema": INFEASIBILITY_SCHEMA,
        "status": "EXACT_GLOBAL_BINARY_MODEL_INFEASIBLE",
        "model_digest": bound[MODEL_DIGEST_KEY],
        "solver": solver,
        "solver_status": 2,
        "solver_contract": dict(SOLVER_CONTRACT),
        "solver_contract_digest": SOLVER_CONTRACT_DIGEST,
        "solver_runtime_identity": _receipt_runtime_identity(
            solver, scipy_runtime_identity=runtime_identity),
        "candidate_outcomes_consumed": False,
    }, INFEASIBILITY_DIGEST_KEY)


def validate_infeasibility(
        model: Mapping[str, Any], receipt: Mapping[str, Any], *,
        _solver_runtime_identity: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
    runtime_identity = _bound_solver_runtime_identity(
        _solver_runtime_identity)
    if not isinstance(receipt, Mapping):
        raise GlobalExactModelError("infeasibility receipt is not a mapping")
    expected = _build_infeasibility(
        model, solver=str(receipt.get("solver", "")),
        _solver_runtime_identity=runtime_identity)
    if dict(receipt) != expected:
        raise GlobalExactModelError("infeasibility receipt binding changed")
    return expected


def solve_model(model: Mapping[str, Any]) -> dict[str, Any]:
    """Solve one already-translated model; status 2 is the sole non-exception."""

    bound = validate_model(model)
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
    except ImportError as exc:  # pragma: no cover
        raise GlobalExactModelError("SciPy MILP runtime is unavailable") from exc
    variables = bound["variables"]
    rows = bound["constraints"]
    matrix_row: list[int] = []
    matrix_col: list[int] = []
    matrix_value: list[float] = []
    for row_index, row in enumerate(rows):
        for variable, coefficient in row["terms"]:
            matrix_row.append(row_index)
            matrix_col.append(variable)
            matrix_value.append(float(coefficient))
    constraints: Any = None
    if rows:
        matrix = coo_matrix(
            (matrix_value, (matrix_row, matrix_col)),
            shape=(len(rows), len(variables)), dtype=np.float64).tocsc()
        constraints = LinearConstraint(
            matrix,
            np.asarray([-np.inf if row["lower"] is None else row["lower"]
                        for row in rows], dtype=np.float64),
            np.asarray([np.inf if row["upper"] is None else row["upper"]
                        for row in rows], dtype=np.float64),
        )
    result = milp(
        c=np.asarray([row["objective_coefficient"] for row in variables],
                     dtype=np.float64),
        integrality=np.ones(len(variables), dtype=np.uint8),
        bounds=Bounds(np.zeros(len(variables)), np.ones(len(variables))),
        constraints=constraints,
        options={key: value for key, value in SOLVER_CONTRACT.items()
                 if key not in {"implementation", "integrality",
                                "search_branching_policy"}},
    )
    if result.status == 2:
        return _build_infeasibility(bound, solver=SOLVER_CONTRACT["implementation"])
    if result.status != 0 or not result.success or result.x is None:
        raise GlobalExactModelError(
            f"global exact solver did not complete: status={result.status}; "
            f"message={result.message!r}")
    rounded = np.rint(np.asarray(result.x, dtype=np.float64))
    if np.max(np.abs(result.x - rounded), initial=0.0) > 1e-6:
        raise GlobalExactModelError("global exact solver returned nonbinary values")
    receipt = _build_solution(
        bound, [int(value) for value in rounded],
        solver=SOLVER_CONTRACT["implementation"])
    if (result.fun is None
            or abs(float(result.fun) - receipt["objective_value"]) > 0.25
            or getattr(result, "mip_gap", None) is None
            or float(result.mip_gap) != 0.0
            or getattr(result, "mip_dual_bound", None) is None
            or abs(float(result.mip_dual_bound)
                   - receipt["objective_value"]) > 0.25):
        raise GlobalExactModelError(
            "global exact solver did not prove the recorded optimum exactly")
    return receipt


def brute_force_model(model: Mapping[str, Any], *, maximum_variables: int = 24
                      ) -> dict[str, Any]:
    """Exhaustive reference control for deliberately small synthetic models."""

    bound = validate_model(model)
    count = len(bound["variables"])
    if count > maximum_variables:
        raise GlobalExactModelError("brute-force control variable limit exceeded")
    best: tuple[int, tuple[int, ...]] | None = None
    for bits in itertools.product((0, 1), repeat=count):
        if any((row["lower"] is not None
                and _constraint_activity(row, bits) < row["lower"])
               or (row["upper"] is not None
                   and _constraint_activity(row, bits) > row["upper"])
               for row in bound["constraints"]):
            continue
        objective = sum(row["objective_coefficient"] * bits[row["index"]]
                        for row in bound["variables"])
        candidate = (objective, bits)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return _build_infeasibility(bound, solver="exhaustive_binary_control_v1")
    return _build_solution(
        bound, best[1], solver="exhaustive_binary_control_v1")


def _eligibility(value: Any, *, owner: Any, identity: str) -> list[bool]:
    if owner != identity:
        raise GlobalExactModelError("rotation eligibility is not identity-owned")
    if (not isinstance(value, list) or len(value) != CANDIDATE_COUNT
            or any(type(flag) is not bool for flag in value)):
        raise GlobalExactModelError("rotation eligibility vector is malformed")
    return list(value)


def _validate_identity(value: Any, *, label: str) -> str:
    if not _is_digest(value):
        raise GlobalExactModelError(f"{label} is not a SHA-256 digest")
    return str(value)


def structural_scene_projection(
        raw_candidate: Mapping[str, Any], *,
        completion_rotation_eligibility: Sequence[bool],
        ) -> dict[str, Any]:
    """Project one complete pre-outcome candidate/evidence row.

    Only fields that do not exist until selection/allocation are excluded.  The
    eligibility vector is included in the structural envelope, so changing a
    scientific mask changes the identity used by every selectable pair.
    """

    if not isinstance(raw_candidate, Mapping):
        raise GlobalExactModelError("raw optional candidate is not a mapping")
    raw = json.loads(_json_bytes(dict(raw_candidate)))
    if set(raw) != _RAW_OPTIONAL_CANDIDATE_KEYS:
        raise GlobalExactModelError("raw optional candidate surface changed")
    if (raw.get("state_id") != "DEFERRED_SMALL_COMPLETION_JOINT_SEARCH"
            or raw.get("split_role")
            != "DEFERRED_SMALL_COMPLETION_JOINT_SEARCH"
            or not isinstance(raw.get("scene_id"), str) or not raw["scene_id"]
            or raw.get("family") != SMALL_FAMILY
            or raw.get("stratum") != COMPLETION_STRATUM
            or not isinstance(raw.get("goal_type"), str)
            or not raw["goal_type"]
            or not isinstance(raw.get("scene_dir"), str) or not raw["scene_dir"]
            or not isinstance(raw.get("split"), str) or not raw["split"]
            or not isinstance(raw.get("episode_cluster_id"), str)
            or not raw["episode_cluster_id"]
            or any(isinstance(raw.get(key), bool)
                   or not isinstance(raw.get(key), int)
                   for key in ("drive_seed", "warmup_blocks", "source_step",
                               "episode_id", "cell_id"))
            or raw["warmup_blocks"] <= 0 or raw["source_step"] < 0
            or raw["episode_id"] < 0 or raw["cell_id"] < 0
            or not _is_finite_number(raw.get("body_clearance_m"))
            or not _is_finite_number(raw.get("clearance_m"))
            or not _is_digest(raw.get("scene_manifest_sha256"))
            or isinstance(raw.get("scene_manifest_byte_count"), bool)
            or not isinstance(raw.get("scene_manifest_byte_count"), int)
            or raw["scene_manifest_byte_count"] <= 0
            or not isinstance(raw.get("boundary"), Mapping)
            or set(raw["boundary"]) != _BOUNDARY_KEYS
            or raw["boundary"].get("source_step") != raw.get("source_step")
            or not _is_digest(raw["boundary"].get("boundary_digest"))
            or not isinstance(raw.get("snapshot_task_status"), Mapping)
            or not isinstance(raw.get("goal"), Mapping)
            or set(raw["goal"]) != _GOAL_KEYS
            or raw["goal"].get("material_id") != raw["goal_type"]
            or not isinstance(raw["goal"].get("landmark_id"), str)
            or not raw["goal"]["landmark_id"]
            or any(isinstance(raw["goal"].get(key), bool)
                   or not isinstance(raw["goal"].get(key), int)
                   for key in ("landmark_cell", "graph_edges"))
            or any(not _is_finite_number(raw["goal"].get(key))
                   for key in ("start_geodesic_m", "bearing_body_rad", "range_m"))
            or not isinstance(raw["goal"].get("landmark_xy_m"), list)
            or len(raw["goal"]["landmark_xy_m"]) != 2
            or any(not _is_finite_number(value)
                   for value in raw["goal"]["landmark_xy_m"])
            or not isinstance(raw.get("previous_applied_command"), list)
            or len(raw["previous_applied_command"]) != 3
            or any(not _is_finite_number(value)
                   for value in raw["previous_applied_command"])):
        raise GlobalExactModelError("raw optional candidate identity changed")
    try:
        from lewm.oracle import go2_scorer_state_selector_amendment_v2 as selector
        rotations = raw["completion_rotation_eligibility_vector"]["rotations"]
        first = rotations[0]
        recomputed = selector.completion_rotation_eligibility_vector(
            graph_hops=int(first["graph_hops_diagnostic"]),
            reachable=bool(first["reachable"]),
            continuous_geodesic_m=float(first["continuous_geodesic_m"]),
            bearing_body_rad=float(first["bearing_body_rad"]),
            task_status=first["task_status"],
            previous_applied_command=raw["previous_applied_command"],
        )
        selector.validate_snapshot_task_status_binding(
            raw["snapshot_task_status"], first["task_status"],
            designated_goal_cell=int(raw["goal"]["landmark_cell"]))
    except (KeyError, TypeError, ValueError, IndexError,
            selector.StateSelectorAmendmentError) as exc:
        raise GlobalExactModelError(
            "optional completion rotation evidence is malformed") from exc
    if raw["completion_rotation_eligibility_vector"] != recomputed:
        raise GlobalExactModelError(
            "optional completion rotation evidence is not reproducible")
    derived_eligibility = [bool(row["eligible"])
                           for row in recomputed["rotations"]]
    eligibility = list(completion_rotation_eligibility)
    if (len(eligibility) != CANDIDATE_COUNT
            or any(type(flag) is not bool for flag in eligibility)
            or eligibility != derived_eligibility):
        raise GlobalExactModelError("rotation eligibility vector is malformed")
    projected_raw = {key: raw[key] for key in sorted(_RAW_OPTIONAL_STRUCTURAL_KEYS)}
    return {
        "schema": STRUCTURAL_SCENE_IDENTITY_SCHEMA,
        "raw_candidate": projected_raw,
        "completion_rotation_eligibility": eligibility,
        "excluded_deferred_assignment_fields": sorted(_DEFERRED_SCENE_FIELDS),
        "candidate_outcomes_consumed": False,
    }


def structural_scene_identity_digest(
        raw_candidate: Mapping[str, Any], *,
        completion_rotation_eligibility: Sequence[bool],
        ) -> str:
    projection = structural_scene_projection(
        raw_candidate,
        completion_rotation_eligibility=completion_rotation_eligibility)
    preimage = (STRUCTURAL_SCENE_IDENTITY_DOMAIN.encode("utf-8") + b"\x00"
                + _json_bytes(projection))
    return hashlib.sha256(preimage).hexdigest()


def _validate_identity_lineage(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _IDENTITY_LINEAGE_KEYS:
        raise GlobalExactModelError("state identity lineage surface changed")
    lineage = dict(value)
    static = lineage.get("pre_allocation_identity_static")
    digest_keys = set(_PRE_ALLOCATION_COMMON_KEYS) - {
        "source_repository_commit", "genesis_backend",
    }
    from lewm.oracle import go2_candidate_allocation_v1_2 as allocation
    if (lineage.get("schema") != STATE_IDENTITY_LINEAGE_SCHEMA
            or not _is_digest(lineage.get("selection_digest"))
            or not _is_digest(lineage.get("scorer_contract_v1_2_digest"))
            or lineage.get("pool") != "scorer_fit"
            or not isinstance(static, Mapping)
            or set(static) != {
                "schema", "pool", "spec", *_PRE_ALLOCATION_COMMON_KEYS}
            or static.get("schema")
            != "go2_branch_corpus_v1_2_pre_allocation_identity_manifest"
            or static.get("pool") != "scorer_fit"
            or static.get("spec") != _SCORER_FIT_POOL_SPEC
            or static.get("selection_digest") != lineage.get("selection_digest")
            or static.get("scorer_contract_v1_2_digest")
            != lineage.get("scorer_contract_v1_2_digest")
            or any(not _is_digest(static.get(key)) for key in digest_keys)
            or not isinstance(static.get("source_repository_commit"), str)
            or len(static["source_repository_commit"]) != 40
            or any(character not in _HEX
                   for character in static["source_repository_commit"])
            or static.get("genesis_backend") != "cpu"
            or static.get("candidate_allocator_contract_digest")
            != allocation.allocation_contract_digest()
            or static.get("candidate_allocation_amendment_digest")
            != allocation.allocation_amendment_digest()):
        raise GlobalExactModelError("state identity lineage changed")
    return json.loads(_json_bytes(lineage))


def _completion_evidence_eligibility(value: Any) -> list[bool]:
    if not isinstance(value, Mapping):
        raise GlobalExactModelError("completion rotation evidence is malformed")
    evidence = json.loads(_json_bytes(dict(value)))
    try:
        from lewm.oracle import go2_scorer_state_selector_amendment_v2 as selector
        rotations = evidence["rotations"]
        first = rotations[0]
        recomputed = selector.completion_rotation_eligibility_vector(
            graph_hops=int(first["graph_hops_diagnostic"]),
            reachable=bool(first["reachable"]),
            continuous_geodesic_m=float(first["continuous_geodesic_m"]),
            bearing_body_rad=float(first["bearing_body_rad"]),
            task_status=first["task_status"],
            previous_applied_command=first["previous_applied_command"],
        )
    except (KeyError, TypeError, ValueError, IndexError,
            selector.StateSelectorAmendmentError) as exc:
        raise GlobalExactModelError(
            "completion rotation evidence cannot be reconstructed") from exc
    if evidence != recomputed:
        raise GlobalExactModelError("completion rotation evidence changed")
    return [bool(row["eligible"]) for row in recomputed["rotations"]]


def build_production_instance(
        *, fixed_states: Sequence[Mapping[str, Any]],
        optional_candidates: Sequence[Mapping[str, Any]],
        state_identity_lineage: Mapping[str, Any],
        ) -> dict[str, Any]:
    """Build the strict production input without reading repository artifacts.

    Each optional input has exactly ``raw_candidate`` and
    ``completion_rotation_eligibility``.  The engine computes, rather than
    trusts, the structural identity and its ownership binding.
    """

    lineage = _validate_identity_lineage(state_identity_lineage)
    optional: list[dict[str, Any]] = []
    for item in optional_candidates:
        if (not isinstance(item, Mapping) or set(item) != {
                "raw_candidate", "completion_rotation_eligibility"}):
            raise GlobalExactModelError("optional candidate input surface changed")
        raw = item["raw_candidate"]
        eligibility = list(item["completion_rotation_eligibility"])
        projection = structural_scene_projection(
            raw, completion_rotation_eligibility=eligibility)
        identity = structural_scene_identity_digest(
            raw, completion_rotation_eligibility=eligibility)
        optional.append({
            "scene_id": projection["raw_candidate"]["scene_id"],
            "structural_scene_identity_digest": identity,
            "structural_scene_projection": projection,
            "goal_type": projection["raw_candidate"]["goal_type"],
            "completion_rotation_eligibility_owner_digest": identity,
            "completion_rotation_eligibility": eligibility,
        })
    optional.sort(key=lambda row: row["scene_id"])
    instance = {
        "schema": PRODUCTION_INSTANCE_SCHEMA,
        "rotation_blocks": [list(row) for row in ROTATION_BLOCKS],
        "fixed_states": [dict(row) for row in fixed_states],
        "optional_scenes": optional,
        "state_identity_lineage": lineage,
        "candidate_outcomes_consumed": False,
        "scientific_masks_are_frozen_search_inputs": True,
    }
    return validate_production_instance(instance)


def validate_production_instance(instance: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact 115-fixed/17-optional pre-outcome input surface."""

    if not isinstance(instance, Mapping) or set(instance) != {
            "schema", "rotation_blocks", "fixed_states", "optional_scenes",
            "state_identity_lineage",
            "candidate_outcomes_consumed",
            "scientific_masks_are_frozen_search_inputs"}:
        raise GlobalExactModelError("production instance key surface changed")
    payload = json.loads(_json_bytes(dict(instance)))
    if (payload.get("schema") != PRODUCTION_INSTANCE_SCHEMA
            or payload.get("rotation_blocks") != [list(row) for row in ROTATION_BLOCKS]
            or payload.get("candidate_outcomes_consumed") is not False
            or payload.get("scientific_masks_are_frozen_search_inputs") is not True):
        raise GlobalExactModelError("production instance contract changed")
    fixed = payload.get("fixed_states")
    optional = payload.get("optional_scenes")
    _validate_identity_lineage(payload.get("state_identity_lineage"))
    if not isinstance(fixed, list) or len(fixed) != 115:
        raise GlobalExactModelError("production instance must bind 115 fixed states")
    if not isinstance(optional, list) or len(optional) != 17:
        raise GlobalExactModelError("production instance must bind 17 optional scenes")
    identities: set[str] = set()
    state_ids: set[str] = set()
    fixed_scene_ids: set[str] = set()
    fixed_completion = 0
    for row in fixed:
        if not isinstance(row, Mapping) or set(row) != _FIXED_KEYS:
            raise GlobalExactModelError("fixed-state surface changed")
        identity = _validate_identity(row.get("state_identity_digest"), label="state identity")
        if (identity in identities
                or not isinstance(row.get("state_id"), str)
                or not row["state_id"] or row["state_id"] in state_ids
                or not isinstance(row.get("scene_id"), str)
                or not row["scene_id"] or row["scene_id"] in fixed_scene_ids
                or row.get("family") not in FAMILIES
                or row.get("stratum") not in STRATA
                or row.get("split_role") not in SPLIT_ROLES
                or not isinstance(row.get("goal_type"), str)
                or not row["goal_type"]):
            raise GlobalExactModelError("fixed-state identity fields changed")
        identities.add(identity)
        state_ids.add(row["state_id"])
        fixed_scene_ids.add(row["scene_id"])
        eligibility = _eligibility(
            row.get("completion_rotation_eligibility"),
            owner=row.get("completion_rotation_eligibility_owner_digest"),
            identity=identity)
        if row["stratum"] == COMPLETION_STRATUM:
            if eligibility != _completion_evidence_eligibility(
                    row.get("completion_rotation_evidence")):
                raise GlobalExactModelError(
                    "fixed completion mask differs from its full evidence")
            fixed_completion += 1
        elif (eligibility != [True] * CANDIDATE_COUNT
              or row.get("completion_rotation_evidence") is not None):
            raise GlobalExactModelError(
                "non-completion fixed state masks rotations or carries evidence")
    for family in FAMILIES:
        for stratum in STRATA:
            cell = [row for row in fixed
                    if row["family"] == family and row["stratum"] == stratum]
            expected = 0 if (family, stratum) == (
                SMALL_FAMILY, COMPLETION_STRATUM) else 5
            if len(cell) != expected:
                raise GlobalExactModelError("fixed family/stratum coverage changed")
            if cell and (sum(row["split_role"] == "fit" for row in cell) != 4
                         or sum(row["split_role"] == "calibration"
                                for row in cell) != 1):
                raise GlobalExactModelError("fixed fit/calibration coverage changed")
    if fixed_completion != 35:
        raise GlobalExactModelError("fixed completion-state count changed")
    scene_ids: list[str] = []
    for scene in optional:
        if not isinstance(scene, Mapping) or set(scene) != _OPTIONAL_KEYS:
            raise GlobalExactModelError("optional-scene surface changed")
        scene_id = scene.get("scene_id")
        structural_identity = _validate_identity(
            scene.get("structural_scene_identity_digest"),
            label="structural scene identity")
        projection = scene.get("structural_scene_projection")
        if not isinstance(projection, Mapping) or set(projection) != {
                "schema", "raw_candidate", "completion_rotation_eligibility",
                "excluded_deferred_assignment_fields",
                "candidate_outcomes_consumed"}:
            raise GlobalExactModelError("structural scene projection changed")
        raw_candidate = projection.get("raw_candidate")
        reconstructed_raw = ({
            **dict(raw_candidate),
            "state_id": "DEFERRED_SMALL_COMPLETION_JOINT_SEARCH",
            "split_role": "DEFERRED_SMALL_COMPLETION_JOINT_SEARCH",
        } if isinstance(raw_candidate, Mapping) else raw_candidate)
        eligibility = scene.get("completion_rotation_eligibility")
        expected_projection = structural_scene_projection(
            reconstructed_raw, completion_rotation_eligibility=eligibility)
        if (not isinstance(scene_id, str) or not scene_id
                or scene_id in fixed_scene_ids
                or not isinstance(scene.get("goal_type"), str)
                or not scene["goal_type"]
                or projection != expected_projection
                or scene_id != raw_candidate.get("scene_id")
                or scene["goal_type"] != raw_candidate.get("goal_type")
                or structural_identity != structural_scene_identity_digest(
                    reconstructed_raw,
                    completion_rotation_eligibility=eligibility)):
            raise GlobalExactModelError("optional-scene identity changed")
        scene_ids.append(scene_id)
        if structural_identity in identities:
            raise GlobalExactModelError("optional structural identity is not unique")
        identities.add(structural_identity)
        _eligibility(
            eligibility,
            owner=scene.get("completion_rotation_eligibility_owner_digest"),
            identity=structural_identity)
    if scene_ids != sorted(scene_ids) or len(set(scene_ids)) != 17:
        raise GlobalExactModelError("optional scenes are not unique lexical rows")
    return payload


class _ProblemBuilder:
    def __init__(self) -> None:
        self.variables: list[dict[str, Any]] = []
        self.constraints: list[dict[str, Any]] = []

    def variable(self, key: str, pair_identity: Mapping[str, Any]) -> str:
        if any(row["key"] == key for row in self.variables):
            raise GlobalExactModelError("duplicate production variable")
        self.variables.append({"key": key, "pair_identity": dict(pair_identity)})
        return key

    def add(self, name: str, terms: Sequence[tuple[str, int]],
            lower: int | None, upper: int | None) -> None:
        self.constraints.append({
            "name": name, "terms": [[key, coefficient] for key, coefficient in terms],
            "lower": lower, "upper": upper,
        })


def build_production_model(
        instance: Mapping[str, Any], *,
        _solver_runtime_identity: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
    """Compile the entire scene selection, allocation and masks into one MILP."""

    runtime_identity = _bound_solver_runtime_identity(
        _solver_runtime_identity)
    bound = validate_production_instance(instance)
    builder = _ProblemBuilder()
    variable_rows: list[dict[str, Any]] = []

    def add_variables(*, kind: str, identity_digest: str,
                      state_id: str | None, scene_id: str | None,
                      family: str, stratum: str, role: str, goal_type: str,
                      eligibility: Sequence[bool]) -> None:
        for rotation, block in enumerate(ROTATION_BLOCKS):
            if kind == "fixed":
                pair_identity = {
                    "kind": "fixed_state",
                    "state_identity_digest": identity_digest,
                    "split_role": role,
                    "candidate_rotation_index": rotation,
                    "candidate_indices": list(block),
                }
            else:
                pair_identity = {
                    "kind": "selectable_completion",
                    "structural_scene_identity_digest": identity_digest,
                    "assigned_split_role": role,
                    "candidate_rotation_index": rotation,
                    "candidate_indices": list(block),
                }
            objective = pair_objective_binding(pair_identity)
            key = objective["canonical_pair_identity_json"]
            builder.variable(key, pair_identity)
            variable_rows.append({
                "key": key, "kind": kind,
                "state_id": state_id,
                "state_identity_digest": (
                    identity_digest if kind == "fixed" else None),
                "structural_scene_identity_digest": (
                    identity_digest if kind == "optional" else None),
                "scene_id": scene_id,
                "family": family, "stratum": stratum, "split_role": role,
                "goal_type": goal_type, "rotation": rotation,
                "candidate_indices": list(block),
                "eligible": bool(eligibility[rotation]),
            })

    for state in bound["fixed_states"]:
        add_variables(
            kind="fixed", identity_digest=state["state_identity_digest"],
            state_id=state["state_id"], scene_id=None,
            family=state["family"], stratum=state["stratum"],
            role=state["split_role"], goal_type=state["goal_type"],
            eligibility=state["completion_rotation_eligibility"])
    for scene in bound["optional_scenes"]:
        for role in SPLIT_ROLES:
            add_variables(
                kind="optional",
                identity_digest=scene["structural_scene_identity_digest"],
                state_id=None, scene_id=scene["scene_id"],
                family=SMALL_FAMILY, stratum=COMPLETION_STRATUM, role=role,
                goal_type=scene["goal_type"],
                eligibility=scene["completion_rotation_eligibility"])

    def rows(**fields: Any) -> list[dict[str, Any]]:
        return [row for row in variable_rows
                if all(row[name] == value for name, value in fields.items())]

    def terms(selected: Sequence[Mapping[str, Any]], candidate: int | None = None,
              coefficient: int = 1) -> list[tuple[str, int]]:
        return [(row["key"], coefficient) for row in selected
                if candidate is None or candidate in row["candidate_indices"]]

    for state in bound["fixed_states"]:
        selected = rows(kind="fixed", state_identity_digest=state["state_identity_digest"])
        builder.add(f"fixed/one/{state['state_identity_digest']}", terms(selected), 1, 1)
    for scene in bound["optional_scenes"]:
        builder.add(f"optional/scene/{scene['scene_id']}/at-most-one",
                    terms(rows(kind="optional", scene_id=scene["scene_id"])), 0, 1)
    for role, expected in (("calibration", 1), ("fit", 4)):
        builder.add(f"optional/role/{role}",
                    terms(rows(kind="optional", split_role=role)), expected, expected)
    # The unique calibration scene is the lexically first selected scene.
    optional_scene_ids = [scene["scene_id"] for scene in bound["optional_scenes"]]
    for calibration_index, calibration_scene in enumerate(optional_scene_ids):
        cal_rows = rows(
            kind="optional", scene_id=calibration_scene,
            split_role="calibration")
        for lower_scene in optional_scene_ids[:calibration_index]:
            fit_rows = rows(
                kind="optional", scene_id=lower_scene, split_role="fit")
            builder.add(
                f"optional/calibration-first/{calibration_scene}/{lower_scene}",
                terms([*cal_rows, *fit_rows]), 0, 1)

    # Identity-owned completion masks are hard upper bounds in the same model.
    for row in variable_rows:
        if row["stratum"] == COMPLETION_STRATUM:
            builder.add(
                f"eligibility/{row['state_identity_digest'] or row['structural_scene_identity_digest']}/"
                f"{row['split_role']}/{row['rotation']}",
                [(row["key"], 1)], 0, 1 if row["eligible"] else 0)

    active_families = list(FAMILIES)
    for family in active_families:
        for stratum in STRATA:
            for candidate in range(CANDIDATE_COUNT):
                fit_rows = rows(family=family, stratum=stratum, split_role="fit")
                builder.add(
                    f"hard/fit-family-stratum/{family}/{stratum}/{candidate:02d}",
                    terms(fit_rows, candidate), 2, 2)
    for stratum in STRATA:
        for candidate in range(CANDIDATE_COUNT):
            cal_rows = rows(stratum=stratum, split_role="calibration")
            builder.add(
                f"hard/calibration-stratum/{stratum}/{candidate:02d}",
                terms(cal_rows, candidate), 4, 4)
    for family in active_families:
        for candidate in range(CANDIDATE_COUNT):
            cal_rows = rows(family=family, split_role="calibration")
            builder.add(
                f"hard/calibration-family/{family}/{candidate:02d}",
                terms(cal_rows, candidate), 1, 2)

    goal_types = sorted({row["goal_type"] for row in variable_rows})
    for goal_type in goal_types:
        goal_rows = rows(goal_type=goal_type)
        for candidate in range(CANDIDATE_COUNT):
            # For every active state, coefficient is 2*I(candidate in B_r)-1.
            dynamic = [(row["key"],
                        1 if candidate in row["candidate_indices"] else -1)
                       for row in goal_rows]
            builder.add(
                f"hard/goal-type/{goal_type}/{candidate:02d}/minus1<=2A-n<=1",
                dynamic, -1, 1)

    # Redundant, exact derived margins keep solution validation local and strict.
    for candidate in range(CANDIDATE_COUNT):
        builder.add(f"derived/global/{candidate:02d}",
                    terms(variable_rows, candidate), 60, 60)
        builder.add(f"derived/split/fit/{candidate:02d}",
                    terms(rows(split_role="fit"), candidate), 48, 48)
        builder.add(f"derived/split/calibration/{candidate:02d}",
                    terms(rows(split_role="calibration"), candidate), 12, 12)
        for stratum in STRATA:
            builder.add(f"derived/stratum/{stratum}/{candidate:02d}",
                        terms(rows(stratum=stratum), candidate), 20, 20)
        for family in FAMILIES:
            builder.add(f"derived/family/{family}/{candidate:02d}",
                        terms(rows(family=family), candidate), 7, 8)
            for stratum in STRATA:
                builder.add(
                    f"derived/family-stratum/{family}/{stratum}/{candidate:02d}",
                    terms(rows(family=family, stratum=stratum), candidate), 2, 3)

    problem = {
        "schema": GENERIC_PROBLEM_SCHEMA,
        "variables": builder.variables,
        "constraints": builder.constraints,
        "metadata": {
            "schema": PRODUCTION_MODEL_SCHEMA,
            "production_instance_digest": canonical_digest(bound),
            "fixed_state_count": 115,
            "optional_scene_count": 17,
            "selected_optional_state_count": 5,
            "active_state_count": 120,
            "active_completion_state_count": 40,
            "variable_metadata_digest": canonical_digest(
                sorted(variable_rows, key=lambda row: row["key"])),
            "rotation_blocks": [list(row) for row in ROTATION_BLOCKS],
            "calibration_is_lexically_first_selected_scene": True,
            "candidate_outcomes_consumed": False,
        },
    }
    model = translate_binary_problem(
        problem, _solver_runtime_identity=runtime_identity)
    # Variable metadata is returned separately by this deterministic public API.
    return {
        "model": model,
        "variable_metadata": sorted(variable_rows, key=lambda row: row["key"]),
        "production_binding_digest": canonical_digest({
            "model_digest": model[MODEL_DIGEST_KEY],
            "variable_metadata": sorted(variable_rows, key=lambda row: row["key"]),
        }),
    }


def validate_production_model(
        instance: Mapping[str, Any], bundle: Mapping[str, Any], *,
        _solver_runtime_identity: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
    expected = build_production_model(
        instance, _solver_runtime_identity=_solver_runtime_identity)
    if not isinstance(bundle, Mapping) or dict(bundle) != expected:
        raise GlobalExactModelError("production model bundle changed")
    return expected


def _build_execution_plan_with_runtime(
        instance: Mapping[str, Any],
        runtime_identity: Mapping[str, Any],
        ) -> dict[str, Any]:
    bound = validate_production_instance(instance)
    runtime = validate_solver_runtime_identity_record(runtime_identity)
    bundle = build_production_model(
        bound, _solver_runtime_identity=runtime)
    model = bundle["model"]
    return _signed({
        "schema": EXECUTION_PLAN_SCHEMA,
        "status": "FROZEN_BEFORE_ONE_GLOBAL_EXACT_SOLVE",
        "production_instance_digest": canonical_digest(bound),
        "production_binding_digest": bundle["production_binding_digest"],
        "model_digest": model[MODEL_DIGEST_KEY],
        "model_variable_count": len(model["variables"]),
        "model_constraint_count": len(model["constraints"]),
        "variable_order_digest": canonical_digest([{
            "index": row["index"], "key": row["key"],
            "pair_digest": row["pair_digest"],
        } for row in model["variables"]]),
        "constraint_order_digest": canonical_digest([{
            "name": row["name"], "terms": row["terms"],
            "lower": row["lower"], "upper": row["upper"],
        } for row in model["constraints"]]),
        "objective_contract": copy.deepcopy(OBJECTIVE_CONTRACT),
        "objective_contract_digest": OBJECTIVE_CONTRACT_DIGEST,
        "solver_contract": copy.deepcopy(SOLVER_CONTRACT),
        "solver_contract_digest": SOLVER_CONTRACT_DIGEST,
        "solver_runtime_identity": runtime,
        "single_global_model": True,
        "external_combination_enumeration": False,
        "performance_gate": None,
        "candidate_outcomes_consumed": False,
        "scientific_masks_are_frozen_model_inputs": True,
    }, EXECUTION_PLAN_DIGEST_KEY)


def build_execution_plan(instance: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze the complete deterministic model and live runtime before solve."""

    return _build_execution_plan_with_runtime(
        instance, solver_runtime_identity())


def validate_execution_plan(instance: Mapping[str, Any], plan: Mapping[str, Any]
                            ) -> dict[str, Any]:
    expected = build_execution_plan(instance)
    if not isinstance(plan, Mapping) or dict(plan) != expected:
        raise GlobalExactModelError("execution plan changed")
    return expected


def validate_execution_plan_solve_free(
        instance: Mapping[str, Any], plan: Mapping[str, Any],
        ) -> dict[str, Any]:
    """Reconstruct a persisted plan without importing the local solver.

    This grants no solve authority.  The stored runtime must equal the frozen
    trusted-control identity byte-for-structure; only :func:`build_execution_plan`
    and :func:`solve_once` compare that identity to the live local extension.
    """

    expected = _build_execution_plan_with_runtime(
        instance, FROZEN_SOLVER_RUNTIME_IDENTITY)
    if not isinstance(plan, Mapping) or dict(plan) != expected:
        raise GlobalExactModelError("execution plan changed")
    return expected


def _selected_completion_state(
        scene: Mapping[str, Any], *, ordinal: int,
        lineage: Mapping[str, Any],
        ) -> dict[str, Any]:
    projection = scene["structural_scene_projection"]
    state = dict(projection["raw_candidate"])
    state["state_id"] = (
        f"{lineage['pool']}-{SMALL_FAMILY}-{COMPLETION_STRATUM}-{ordinal:02d}"
    )
    state["split_role"] = "calibration" if ordinal == 0 else "fit"
    identity_payload = {
        "schema": "go2_branch_state_identity_v1_2",
        "selection_digest": lineage["selection_digest"],
        "scorer_contract_v1_2_digest": lineage["scorer_contract_v1_2_digest"],
        "state": dict(state),
    }
    # State identities predate this model and use the corpus builder's
    # default-separator JSON digest namespace, not the compact model-receipt
    # namespace.
    state["state_identity_digest"] = _legacy_builder_identity_digest(
        identity_payload)
    return state


def legacy_allocation_contract_disposition() -> dict[str, Any]:
    """Describe the legacy-shaped manifest's narrow compatibility role.

    The unchanged legacy manifest surface is required by pure structural
    validators and downstream candidate-count consumers.  Its historical
    lexicographic choice sentence is not the active selection certificate for
    this successor; the amendment-bound one-model result is.
    """

    return _signed({
        "schema": ALLOCATION_CONTRACT_DISPOSITION_SCHEMA,
        "status": "STRUCTURAL_COMPATIBILITY_PROJECTION_ONLY",
        "legacy_allocation_contract_digest":
            AUTHORITY.CANDIDATE_ALLOCATION_CONTRACT_DIGEST,
        "legacy_choice_rule": (
            "lexicographically smallest feasible rotation-index vector in "
            "canonical_state_order"
        ),
        "legacy_choice_rule_status":
            AUTHORITY.SUPERSEDED_CANONICAL_TIE_BREAK_STATUS,
        "active_choice_rule": copy.deepcopy(OBJECTIVE_CONTRACT),
        "active_choice_rule_digest": OBJECTIVE_CONTRACT_DIGEST,
        "hard_allocation_margins_preserved": True,
        "candidate_bank_and_rotation_blocks_preserved": True,
        "structural_validation_required": True,
        "global_exact_plan_and_result_certificate_required": True,
        "standalone_legacy_canonicality_claim_accepted": False,
        "candidate_outcomes_consumed": False,
    }, ALLOCATION_CONTRACT_DISPOSITION_SELF_KEY)


def materialize_allocation_manifest(
        instance: Mapping[str, Any], bundle: Mapping[str, Any],
        solution: Mapping[str, Any], *,
        _solver_runtime_identity: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
    """Materialize the global optimum as the legacy solve-free-valid manifest."""

    runtime_identity = _bound_solver_runtime_identity(
        _solver_runtime_identity)
    bound = validate_production_instance(instance)
    production = validate_production_model(
        bound, bundle, _solver_runtime_identity=runtime_identity)
    solved = validate_solution(
        production["model"], solution,
        _solver_runtime_identity=runtime_identity)
    selected_keys = set(solved["selected_variable_keys"])
    metadata = {row["key"]: row for row in production["variable_metadata"]}
    if not selected_keys.issubset(metadata):
        raise GlobalExactModelError("solution selected unknown production variables")
    selected = [metadata[key] for key in selected_keys]
    fixed_rows = [row for row in selected if row["kind"] == "fixed"]
    optional_rows = [row for row in selected if row["kind"] == "optional"]
    if (len(fixed_rows) != 115 or len(optional_rows) != 5
            or len({row["state_identity_digest"] for row in fixed_rows}) != 115
            or len({row["scene_id"] for row in optional_rows}) != 5):
        raise GlobalExactModelError("selected production assignment count changed")
    optional_rows.sort(key=lambda row: row["scene_id"])
    if ([row["split_role"] for row in optional_rows]
            != ["calibration", "fit", "fit", "fit", "fit"]):
        raise GlobalExactModelError(
            "selected calibration is not the lexical first scene")
    scenes = {row["scene_id"]: row for row in bound["optional_scenes"]}
    lineage = bound["state_identity_lineage"]
    selected_states = [
        _selected_completion_state(
            scenes[row["scene_id"]], ordinal=ordinal, lineage=lineage)
        for ordinal, row in enumerate(optional_rows)
    ]
    rotation_by_identity = {
        row["state_identity_digest"]: row["rotation"] for row in fixed_rows
    }
    for state, row in zip(selected_states, optional_rows, strict=True):
        rotation_by_identity[state["state_identity_digest"]] = row["rotation"]
    active_states = [*bound["fixed_states"], *selected_states]
    identity_rows = [{key: row[key] for key in (
        "state_id", "state_identity_digest", "family", "stratum",
        "split_role", "goal_type")}
        for row in active_states]
    from lewm.oracle import go2_candidate_allocation_v1_2 as allocation
    normalised = allocation._normalise_identity_states(identity_rows)
    source_state_order = sorted(active_states, key=lambda state: (
        str(state["family"]), STRATA.index(str(state["stratum"])),
        str(state["scene_id"])))
    source_projection = {
        **lineage["pre_allocation_identity_static"],
        "state_identities": [{key: state[key] for key in (
            "state_id", "state_identity_digest", "family", "stratum",
            "split_role", "goal_type")}
            for state in source_state_order],
    }
    assignments = [{
        **state,
        "rotation_index": int(rotation_by_identity[state["state_identity_digest"]]),
        "candidate_indices": list(allocation.candidate_block(
            int(rotation_by_identity[state["state_identity_digest"]]))),
    } for state in normalised]
    manifest: dict[str, Any] = {
        "schema": allocation.SCHEMA,
        "status": allocation.STATUS,
        "source_identity_manifest_digest":
            _legacy_builder_identity_digest(source_projection),
        "pre_outcome_identity_digest":
            allocation.pre_outcome_identity_digest(normalised),
        "allocation_contract": allocation.algorithm_contract(),
        "allocation_contract_digest": allocation.allocation_contract_digest(),
        "allocation_amendment": allocation.allocation_amendment_contract(),
        "allocation_amendment_digest": allocation.allocation_amendment_digest(),
        "assignments": assignments,
        "contingency_tables": allocation._contingency_tables(assignments),
        "post_identity_pre_outcome_validation":
            allocation._post_identity_pre_outcome_validation(assignments),
    }
    manifest["allocation_manifest_digest"] = \
        allocation.allocation_manifest_digest(manifest)
    from lewm.oracle import go2_scorer_state_selector_amendment_v2 as selector
    selector.validate_allocation_manifest_structure_solve_free(
        manifest,
        expected_source_identity_manifest_digest=
            manifest["source_identity_manifest_digest"])
    selected_scene_rows = [{
        "selected_scene_index": next(
            index for index, scene in enumerate(bound["optional_scenes"])
            if scene["scene_id"] == row["scene_id"]),
        "selected_scene_id": row["scene_id"],
        "structural_scene_identity_digest":
            scenes[row["scene_id"]]["structural_scene_identity_digest"],
        "selected_ordinal": ordinal,
        "assigned_split_role": row["split_role"],
        "state_id": selected_states[ordinal]["state_id"],
        "state_identity_digest": selected_states[ordinal][
            "state_identity_digest"],
        "candidate_rotation_index": row["rotation"],
        "candidate_indices": list(row["candidate_indices"]),
    } for ordinal, row in enumerate(optional_rows)]
    return _signed({
        "schema": ALLOCATION_RESULT_SCHEMA,
        "status": "PASS_MATERIALIZED_GLOBAL_EXACT_ALLOCATION",
        "production_instance_digest": canonical_digest(bound),
        "production_binding_digest": production["production_binding_digest"],
        "solution_digest": solved[SOLUTION_DIGEST_KEY],
        "deterministic_optimal_objective_value": solved["objective_value"],
        "selected_scene_indices": [
            row["selected_scene_index"] for row in selected_scene_rows],
        "selected_scene_ids": [
            row["selected_scene_id"] for row in selected_scene_rows],
        "selected_scene_rows": selected_scene_rows,
        "source_identity_manifest_projection": source_projection,
        "allocation_manifest": manifest,
        "legacy_allocation_contract_disposition":
            legacy_allocation_contract_disposition(),
        "candidate_outcomes_consumed": False,
    }, ALLOCATION_RESULT_DIGEST_KEY)


def validate_materialized_allocation(
        instance: Mapping[str, Any], bundle: Mapping[str, Any],
        solution: Mapping[str, Any], result: Mapping[str, Any], *,
        _solver_runtime_identity: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
    expected = materialize_allocation_manifest(
        instance, bundle, solution,
        _solver_runtime_identity=_solver_runtime_identity)
    if not isinstance(result, Mapping) or dict(result) != expected:
        raise GlobalExactModelError("materialized allocation changed")
    return expected


def _execution_result(
        plan: Mapping[str, Any], exact_result: Mapping[str, Any], *,
        materialized: Mapping[str, Any] | None,
        ) -> dict[str, Any]:
    status = (EXECUTION_PASS_STATUS if materialized is not None
              else EXECUTION_INFEASIBLE_STATUS)
    return _signed({
        "schema": EXECUTION_RESULT_SCHEMA,
        "status": status,
        "execution_plan_digest": plan[EXECUTION_PLAN_DIGEST_KEY],
        "production_instance_digest": plan["production_instance_digest"],
        "production_binding_digest": plan["production_binding_digest"],
        "model_digest": plan["model_digest"],
        "objective_contract_digest": OBJECTIVE_CONTRACT_DIGEST,
        "solver_contract_digest": SOLVER_CONTRACT_DIGEST,
        "solver_runtime_identity": plan["solver_runtime_identity"],
        "exact_model_result": dict(exact_result),
        "materialized_allocation": (
            None if materialized is None else dict(materialized)),
        "deterministic_optimal_objective_value": (
            None if materialized is None else
            exact_result["deterministic_optimal_objective_value"]),
        "selected_scene_indices": (
            [] if materialized is None else
            list(materialized["selected_scene_indices"])),
        "selected_scene_ids": (
            [] if materialized is None else
            list(materialized["selected_scene_ids"])),
        "performance_gate": None,
        "candidate_outcomes_consumed": False,
    }, EXECUTION_RESULT_DIGEST_KEY)


def solve_once(instance: Mapping[str, Any], plan: Mapping[str, Any]
               ) -> dict[str, Any]:
    """Run the frozen model once and return PASS or exact INFEASIBLE."""

    frozen_plan = validate_execution_plan(instance, plan)
    bundle = build_production_model(instance)
    exact = solve_model(bundle["model"])
    if exact["schema"] == INFEASIBILITY_SCHEMA:
        validate_infeasibility(bundle["model"], exact)
        return _execution_result(frozen_plan, exact, materialized=None)
    materialized = materialize_allocation_manifest(
        instance, bundle, require_solution(exact))
    return _execution_result(frozen_plan, exact, materialized=materialized)


def _validate_execution_result(
        instance: Mapping[str, Any], plan: Mapping[str, Any],
        result: Mapping[str, Any], *, solve_free: bool,
        ) -> dict[str, Any]:
    if type(solve_free) is not bool:
        raise GlobalExactModelError("solve-free validation flag changed")
    runtime_identity = (
        validate_solver_runtime_identity_record(FROZEN_SOLVER_RUNTIME_IDENTITY)
        if solve_free else solver_runtime_identity())
    frozen_plan = (
        validate_execution_plan_solve_free(instance, plan)
        if solve_free else validate_execution_plan(instance, plan))
    bundle = build_production_model(
        instance, _solver_runtime_identity=runtime_identity)
    if not isinstance(result, Mapping):
        raise GlobalExactModelError("execution result is not a mapping")
    payload = dict(result)
    exact = payload.get("exact_model_result")
    if (not isinstance(exact, Mapping)
            or exact.get("solver") != SOLVER_CONTRACT["implementation"]):
        raise GlobalExactModelError(
            "production execution result did not use the frozen solver")
    if payload.get("status") == EXECUTION_PASS_STATUS:
        solved = validate_solution(
            bundle["model"], exact,
            _solver_runtime_identity=runtime_identity)
        materialized = validate_materialized_allocation(
            instance, bundle, solved, payload.get("materialized_allocation"),
            _solver_runtime_identity=runtime_identity)
        expected = _execution_result(
            frozen_plan, solved, materialized=materialized)
    elif payload.get("status") == EXECUTION_INFEASIBLE_STATUS:
        infeasible = validate_infeasibility(
            bundle["model"], exact,
            _solver_runtime_identity=runtime_identity)
        expected = _execution_result(
            frozen_plan, infeasible, materialized=None)
    else:
        raise GlobalExactModelError("execution terminal status changed")
    if payload != expected:
        raise GlobalExactModelError("execution result binding changed")
    return expected


def validate_execution_result(
        instance: Mapping[str, Any], plan: Mapping[str, Any],
        result: Mapping[str, Any],
        ) -> dict[str, Any]:
    """Validate a result and require the exact live solver runtime."""

    return _validate_execution_result(
        instance, plan, result, solve_free=False)


def validate_execution_result_solve_free(
        instance: Mapping[str, Any], plan: Mapping[str, Any],
        result: Mapping[str, Any],
        ) -> dict[str, Any]:
    """Validate a persisted result without importing or invoking a solver."""

    return _validate_execution_result(
        instance, plan, result, solve_free=True)


def _fixture_specs() -> list[dict[str, Any]]:
    """Five closed semantic fixtures mandated by the amendment."""

    return [
        {
            "fixture_id": "KNOWN_FEASIBLE",
            "entities": [
                {"id": "a0", "required": True, "goal_type": "g",
                 "alternatives": [{"role": "fit", "rotation": 0,
                                   "eligible": True}]},
                {"id": "a1", "required": True, "goal_type": "g",
                 "alternatives": [{"role": "fit", "rotation": 6,
                                   "eligible": True}]},
            ],
            "role_counts": {"fit": 2},
            "rotation_counts": {"0": 1, "6": 1},
            "candidate_counts": {},
            "goal_balance_candidates": list(range(CANDIDATE_COUNT)),
            "calibration_lexical_first": False,
            "old_canonical_vector": None, "later_joint_vector": None,
        },
        {
            "fixture_id": "KNOWN_INFEASIBLE",
            "entities": [{"id": "b", "required": True, "goal_type": "g",
                          "alternatives": [{"role": "fit", "rotation": 0,
                                            "eligible": False}]}],
            "role_counts": {"fit": 1}, "rotation_counts": {"0": 1},
            "candidate_counts": {}, "goal_balance_candidates": [],
            "calibration_lexical_first": False,
            "old_canonical_vector": None, "later_joint_vector": None,
        },
        {
            "fixture_id": (
                "MULTIPLE_FEASIBLE_OLD_CANONICAL_MASK_FAIL_LATER_JOINT_VALID"),
            "entities": [
                {"id": "c0", "required": True, "goal_type": "g",
                 "alternatives": [
                     {"role": "fit", "rotation": 0, "eligible": False},
                     {"role": "fit", "rotation": 1, "eligible": True}]},
                {"id": "c1", "required": True, "goal_type": "g",
                 "alternatives": [
                     {"role": "fit", "rotation": 0, "eligible": True},
                     {"role": "fit", "rotation": 1, "eligible": False}]},
            ],
            "role_counts": {"fit": 2},
            "rotation_counts": {"0": 1, "1": 1},
            "candidate_counts": {}, "goal_balance_candidates": [],
            "calibration_lexical_first": False,
            "old_canonical_vector": [0, 1], "later_joint_vector": [1, 0],
        },
        {
            "fixture_id": "FIT_CALIBRATION_CONSTRAINTS",
            "entities": [{
                "id": f"d{index}", "required": False, "goal_type": "g",
                "alternatives": [
                    {"role": role, "rotation": 0, "eligible": True}
                    for role in SPLIT_ROLES],
            } for index in range(3)],
            "role_counts": {"fit": 1, "calibration": 1},
            "rotation_counts": {"0": 2}, "candidate_counts": {},
            "goal_balance_candidates": [],
            "calibration_lexical_first": True,
            "old_canonical_vector": None, "later_joint_vector": None,
        },
        {
            "fixture_id": "RESIDUAL_CANDIDATE_FREQUENCY_CONSTRAINTS",
            "entities": [{
                "id": f"e{index}", "required": True, "goal_type": "g",
                "alternatives": [
                    {"role": "fit", "rotation": rotation, "eligible": True}
                    for rotation in (0, 1)],
            } for index in range(3)],
            "role_counts": {"fit": 3},
            "rotation_counts": {},
            "candidate_counts": {
                str(candidate):
                    2 * int(candidate in ROTATION_BLOCKS[0])
                    + int(candidate in ROTATION_BLOCKS[1])
                for candidate in range(CANDIDATE_COUNT)},
            "goal_balance_candidates": [],
            "calibration_lexical_first": False,
            "old_canonical_vector": None, "later_joint_vector": None,
        },
    ]


def _fixture_alternatives(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    alternatives: list[dict[str, Any]] = []
    for entity in spec["entities"]:
        for alternative in entity["alternatives"]:
            rotation = int(alternative["rotation"])
            role = str(alternative["role"])
            identity = canonical_digest({"fixture": spec["fixture_id"],
                                         "entity": entity["id"]})
            pair = {
                "kind": "selectable_completion",
                "structural_scene_identity_digest": identity,
                "assigned_split_role": role,
                "candidate_rotation_index": rotation,
                "candidate_indices": list(ROTATION_BLOCKS[rotation]),
            }
            alternatives.append({
                "key": pair_objective_binding(pair)[
                    "canonical_pair_identity_json"],
                "pair_identity": pair, "entity_id": entity["id"],
                "required": entity["required"],
                "goal_type": entity["goal_type"], "role": role,
                "rotation": rotation, "eligible": alternative["eligible"],
            })
    return alternatives


def _fixture_assignment_passes(
        spec: Mapping[str, Any], selected: Sequence[Mapping[str, Any]], *,
        enforce_eligibility: bool,
        ) -> bool:
    by_entity = {entity["id"]: entity for entity in spec["entities"]}
    selected_by_entity = {entity_id: [row for row in selected
                                      if row["entity_id"] == entity_id]
                          for entity_id in by_entity}
    if any(len(selected_by_entity[entity_id]) != (
            1 if entity["required"] else min(1, len(selected_by_entity[entity_id])))
           for entity_id, entity in by_entity.items()):
        return False
    if any(len(rows) > 1 for rows in selected_by_entity.values()):
        return False
    if enforce_eligibility and any(not row["eligible"] for row in selected):
        return False
    for role, expected in spec["role_counts"].items():
        if sum(row["role"] == role for row in selected) != expected:
            return False
    for rotation, expected in spec["rotation_counts"].items():
        if sum(row["rotation"] == int(rotation) for row in selected) != expected:
            return False
    for candidate, expected in spec["candidate_counts"].items():
        if sum(int(candidate) in ROTATION_BLOCKS[row["rotation"]]
               for row in selected) != expected:
            return False
    for goal in sorted({row["goal_type"] for row in selected}):
        goal_rows = [row for row in selected if row["goal_type"] == goal]
        for candidate in spec["goal_balance_candidates"]:
            activity = sum(1 if candidate in ROTATION_BLOCKS[row["rotation"]]
                           else -1 for row in goal_rows)
            if not -1 <= activity <= 1:
                return False
    if spec["calibration_lexical_first"]:
        calibration = sorted(row["entity_id"] for row in selected
                             if row["role"] == "calibration")
        fit = sorted(row["entity_id"] for row in selected if row["role"] == "fit")
        if len(calibration) != 1 or not fit or calibration[0] >= fit[0]:
            return False
    return True


def _fixture_problem(spec: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    alternatives = _fixture_alternatives(spec)
    constraints: list[dict[str, Any]] = []

    def add(name: str, rows: Sequence[Mapping[str, Any]], lower: int, upper: int,
            coefficient: Any = None) -> None:
        terms = [[row["key"], (1 if coefficient is None else coefficient(row))]
                 for row in rows]
        constraints.append({"name": name, "terms": terms,
                            "lower": lower, "upper": upper})

    for entity in spec["entities"]:
        rows = [row for row in alternatives if row["entity_id"] == entity["id"]]
        add(f"entity/{entity['id']}", rows, 1 if entity["required"] else 0, 1)
    for row in alternatives:
        if not row["eligible"]:
            add(f"eligibility/{canonical_digest(row['key'])}", [row], 0, 0)
    for role, expected in spec["role_counts"].items():
        add(f"role/{role}", [row for row in alternatives if row["role"] == role],
            expected, expected)
    for rotation, expected in spec["rotation_counts"].items():
        add(f"rotation/{int(rotation):02d}", [row for row in alternatives
                                              if row["rotation"] == int(rotation)],
            expected, expected)
    for candidate, expected in spec["candidate_counts"].items():
        rows = [row for row in alternatives
                if int(candidate) in ROTATION_BLOCKS[row["rotation"]]]
        if rows:
            add(f"candidate/{int(candidate):02d}", rows, expected, expected)
        elif expected != 0:
            raise GlobalExactModelError("fixture has an impossible empty margin")
    for goal in sorted({row["goal_type"] for row in alternatives}):
        rows = [row for row in alternatives if row["goal_type"] == goal]
        for candidate in spec["goal_balance_candidates"]:
            add(f"goal/{goal}/{candidate:02d}", rows, -1, 1,
                lambda row, candidate=candidate:
                    1 if candidate in ROTATION_BLOCKS[row["rotation"]] else -1)
    if spec["calibration_lexical_first"]:
        entity_ids = sorted(entity["id"] for entity in spec["entities"])
        for later_index, later in enumerate(entity_ids):
            cal = [row for row in alternatives
                   if row["entity_id"] == later and row["role"] == "calibration"]
            for earlier in entity_ids[:later_index]:
                fit = [row for row in alternatives
                       if row["entity_id"] == earlier and row["role"] == "fit"]
                add(f"calibration-first/{later}/{earlier}", [*cal, *fit], 0, 1)
    problem = {
        "schema": GENERIC_PROBLEM_SCHEMA,
        "variables": [{"key": row["key"], "pair_identity": row["pair_identity"]}
                      for row in alternatives],
        "constraints": constraints,
        "metadata": {"fixture_id": spec["fixture_id"],
                     "semantic_spec_digest": canonical_digest(spec)},
    }
    return problem, alternatives


def _fixture_direct_control(
        spec: Mapping[str, Any], alternatives: Sequence[Mapping[str, Any]],
        model: Mapping[str, Any],
        ) -> dict[str, Any]:
    entities = list(spec["entities"])
    choices = []
    for entity in entities:
        rows = [row for row in alternatives if row["entity_id"] == entity["id"]]
        choices.append(rows if entity["required"] else [None, *rows])
    feasible: list[list[Mapping[str, Any]]] = []
    for picked in itertools.product(*choices):
        selected = [row for row in picked if row is not None]
        if _fixture_assignment_passes(spec, selected, enforce_eligibility=True):
            feasible.append(selected)
    if not feasible:
        return {"feasible": False, "optimal_objective_value": None,
                "valid_assignment_count": 0}
    coefficient = {row["key"]: row["objective_coefficient"]
                   for row in model["variables"]}
    objectives = [sum(coefficient[row["key"]] for row in selected)
                  for selected in feasible]
    return {"feasible": True, "optimal_objective_value": min(objectives),
            "valid_assignment_count": len(feasible)}


def build_fixture_suite_result() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in _fixture_specs():
        problem, alternatives = _fixture_problem(spec)
        model = translate_binary_problem(problem)
        control = _fixture_direct_control(spec, alternatives, model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            first = solve_model(model)
            second = solve_model(model)
        solver_feasible = first["schema"] == SOLUTION_SCHEMA
        if first != second or solver_feasible != control["feasible"]:
            raise GlobalExactModelError("fixture solver/control agreement failed")
        selected = ([row for row in alternatives
                     if row["key"] in first["selected_variable_keys"]]
                    if solver_feasible else [])
        if (solver_feasible
                and (first["objective_value"] != control["optimal_objective_value"]
                     or not _fixture_assignment_passes(
                         spec, selected, enforce_eligibility=True))):
            raise GlobalExactModelError("fixture optimum failed direct semantics")
        boundary: dict[str, Any] | None = None
        if spec["old_canonical_vector"] is not None:
            def vector_rows(vector: Sequence[int]) -> list[Mapping[str, Any]]:
                return [next(row for row in alternatives
                             if row["entity_id"] == entity["id"]
                             and row["rotation"] == rotation)
                        for entity, rotation in zip(
                            spec["entities"], vector, strict=True)]
            old = vector_rows(spec["old_canonical_vector"])
            later = vector_rows(spec["later_joint_vector"])
            selected_vector = [next(
                row["rotation"] for row in selected
                if row["entity_id"] == entity["id"])
                for entity in spec["entities"]]
            boundary = {
                "at_least_two_hard_margin_feasible_rotation_vectors":
                    _fixture_assignment_passes(
                        spec, old, enforce_eligibility=False)
                    and _fixture_assignment_passes(
                        spec, later, enforce_eligibility=False),
                "old_identity_ordered_canonical_vector_mask_passes":
                    _fixture_assignment_passes(
                        spec, old, enforce_eligibility=True),
                "later_hard_feasible_vector_mask_passes":
                    _fixture_assignment_passes(
                        spec, later, enforce_eligibility=True),
                "old_and_new_methods_agree_underlying_hard_margin_feasibility":
                    _fixture_assignment_passes(
                        spec, old, enforce_eligibility=False)
                    and _fixture_assignment_passes(
                        spec, selected, enforce_eligibility=False),
                "new_global_model_returns_mask_valid_solution":
                    solver_feasible and _fixture_assignment_passes(
                        spec, selected, enforce_eligibility=True),
                "new_solution_may_differ_from_old_canonical_vector":
                    selected_vector != spec["old_canonical_vector"]
                    and selected_vector == spec["later_joint_vector"],
                "every_scientific_constraint_still_validates":
                    _fixture_assignment_passes(
                        spec, selected, enforce_eligibility=True),
            }
            expected_boundary = dict(
                AUTHORITY.FIXTURE_VALIDATION_CONTRACT[
                    "mandatory_boundary_fixture"])
            expected_boundary.pop("fixture_id")
            if boundary != expected_boundary:
                raise GlobalExactModelError("mandatory boundary fixture failed")
        digest_key = (SOLUTION_DIGEST_KEY if solver_feasible
                      else INFEASIBILITY_DIGEST_KEY)
        rows.append({
            "fixture_id": spec["fixture_id"],
            "semantic_spec_digest": canonical_digest(spec),
            "model_digest": model[MODEL_DIGEST_KEY],
            "solver_feasible": solver_feasible,
            "control_feasible": control["feasible"],
            "control_valid_assignment_count": control["valid_assignment_count"],
            "deterministic_optimal_objective_value":
                control["optimal_objective_value"],
            "repeated_runs_identical_bytes": True,
            "exact_result_digest": first[digest_key],
            "all_returned_constraints_directly_validated": True,
            "boundary_predicates": boundary,
            "candidate_outcomes_consumed": False,
        })
    expected_ids = AUTHORITY.FIXTURE_VALIDATION_CONTRACT["required_fixture_ids"]
    if [row["fixture_id"] for row in rows] != expected_ids:
        raise GlobalExactModelError("mandatory fixture inventory changed")
    result = _signed({
        "schema": FIXTURE_SUITE_SCHEMA,
        "status": "PASS_MANDATORY_SYNTHETIC_FIXTURE_SUITE",
        "fixture_validation_contract": copy.deepcopy(
            AUTHORITY.FIXTURE_VALIDATION_CONTRACT),
        "objective_contract_digest": OBJECTIVE_CONTRACT_DIGEST,
        "solver_contract_digest": SOLVER_CONTRACT_DIGEST,
        "solver_runtime_identity": solver_runtime_identity(),
        "fixtures": rows,
        "candidate_outcomes_consumed": False,
    }, FIXTURE_SUITE_DIGEST_KEY)
    if result[FIXTURE_SUITE_DIGEST_KEY] != FROZEN_FIXTURE_SUITE_RESULT_DIGEST:
        raise GlobalExactModelError("frozen fixture suite result changed")
    return result


def validate_fixture_suite_result(result: Mapping[str, Any]) -> dict[str, Any]:
    expected = build_fixture_suite_result()
    if not isinstance(result, Mapping) or dict(result) != expected:
        raise GlobalExactModelError("fixture suite result changed")
    return expected


def require_solution(result: Mapping[str, Any]) -> dict[str, Any]:
    if result.get("schema") == INFEASIBILITY_SCHEMA:
        raise GlobalExactInfeasible("exact global small-completion model is infeasible")
    if result.get("schema") != SOLUTION_SCHEMA:
        raise GlobalExactModelError("global exact result schema changed")
    return dict(result)


__all__ = [
    "ALLOCATION_CONTRACT_DISPOSITION_SCHEMA",
    "ALLOCATION_CONTRACT_DISPOSITION_SELF_KEY",
    "ALLOCATION_RESULT_DIGEST_KEY", "ALLOCATION_RESULT_SCHEMA",
    "CANDIDATE_COUNT", "COMPLETION_STRATUM", "EXECUTION_INFEASIBLE_STATUS",
    "EXECUTION_PASS_STATUS", "EXECUTION_PLAN_DIGEST_KEY",
    "EXECUTION_PLAN_SCHEMA", "EXECUTION_RESULT_DIGEST_KEY",
    "EXECUTION_RESULT_SCHEMA", "FAMILIES", "FIXTURE_SUITE_DIGEST_KEY",
    "FROZEN_FIXTURE_SUITE_RESULT_DIGEST", "FROZEN_SOLVER_RUNTIME_IDENTITY",
    "FIXTURE_SUITE_SCHEMA", "GENERIC_PROBLEM_SCHEMA", "GlobalExactInfeasible",
    "GlobalExactModelError", "INFEASIBILITY_DIGEST_KEY",
    "INFEASIBILITY_SCHEMA", "MODEL_DIGEST_KEY", "MODEL_SCHEMA",
    "OBJECTIVE_CONTRACT", "OBJECTIVE_CONTRACT_DIGEST", "OBJECTIVE_DOMAIN",
    "PRODUCTION_INSTANCE_SCHEMA", "PRODUCTION_MODEL_SCHEMA", "ROTATION_BLOCKS",
    "SMALL_FAMILY", "SOLUTION_DIGEST_KEY", "SOLUTION_SCHEMA",
    "SOLVER_CONTRACT", "SOLVER_CONTRACT_DIGEST", "SPLIT_ROLES",
    "STATE_IDENTITY_LINEAGE_SCHEMA", "STRATA",
    "STRUCTURAL_SCENE_IDENTITY_DOMAIN", "STRUCTURAL_SCENE_IDENTITY_SCHEMA",
    "brute_force_model", "build_execution_plan", "build_fixture_suite_result",
    "build_production_instance", "build_production_model", "canonical_digest",
    "legacy_allocation_contract_disposition",
    "materialize_allocation_manifest", "pair_objective_binding",
    "require_solution", "solve_model", "solve_once", "solver_runtime_identity",
    "structural_scene_identity_digest", "structural_scene_projection",
    "translate_binary_problem", "validate_execution_plan",
    "validate_execution_plan_solve_free", "validate_execution_result",
    "validate_execution_result_solve_free", "validate_fixture_suite_result",
    "validate_infeasibility", "validate_materialized_allocation",
    "validate_model", "validate_production_instance",
    "validate_production_model", "validate_solution",
    "validate_solver_runtime_identity_record",
]
