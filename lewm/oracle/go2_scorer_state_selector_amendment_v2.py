"""Final prospective scorer-fit completion-enrichment amendment (V2).

This module changes one snapshot-time selector condition and nothing else.  A
completion-enriched state is distance-eligible when the continuous geodesic
gap to the unchanged 0.75 m selector parameter ``r_complete`` can be covered by
the maximum *nominal translational path length* in that state's exact frozen
six-candidate subset over the unchanged 20-tick horizon.

The arithmetic is pure and outcome-free.  It reconstructs every candidate's
twenty post-slew commands from the actual previous applied command, using the
already frozen limiter.  It never imports Genesis, executes a branch, or reads
a realised state transition.

Candidate masks are assigned only after all 120 state identities exist.  To
avoid pretending that the pre-identity structural fixture is an assignment,
the pre-identity API emits a twelve-rotation eligibility vector.  The active
identity manifest is gated on recomputing the predicate for the exact mask
assigned by the unchanged canonical allocator.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import operator
import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lewm.oracle import go2_candidate_allocation_v1_2 as ALLOCATION
from lewm.oracle import go2_scorer_state_selector_amendment_v1 as PREDECESSOR
from scripts import dev_action_slew_reconstruction_v1 as SLEW


ROOT = Path(__file__).resolve().parents[2]

AMENDMENT_SCHEMA = "go2_scorer_fit_state_selector_amendment_v2"
AMENDMENT_VERSION = "completion_horizon_reachability_v2"
AMENDMENT_ARTIFACT_PATH = (
    "docs/lewm_go2_shared_utility_scorer_v1_2_state_selector_"
    "completion_horizon_reachability_amendment_v2_2026-08-12.json"
)

AUTHORIZING_FAILURE_COMMIT = "568a9511aa7c54e72052768c27eb356cf10debbe"
FAILURE_REPORT_PATH = (
    "docs/lewm_go2_shared_utility_scorer_v1_2_preoutcome_"
    "selector_feasibility_failure_2026-08-12.json"
)
FAILURE_REPORT_DIGEST = (
    "81637cdf3889dc0856ea97aee9a644f182855ef49c4e466eee3f8aed4134a0b8"
)
FAILURE_REPORT_RAW_SHA256 = (
    "db2e025fe71164943bb214a602dfdeb249d629de2420aa758bb97717c8974b49"
)
FAILURE_REPORT_BYTE_COUNT = 28_618
FROZEN_FAILED_CENSUS_RECEIPT = {
    "path": (
        ".generated/go2_branch_corpus_v1_2/scorer_fit/"
        "state_selector_feasibility_receipt.json"
    ),
    "schema": "go2_scorer_fit_state_selector_feasibility_receipt_v1",
    "status": "FAIL_OUTCOME_FREE_SELECTOR_FEASIBILITY",
    "state_selector_feasibility_receipt_digest": (
        "2310c3d1b138b605fda483b39cbd4775479cbcc502a4e3707e7a8670457f54d7"
    ),
    "raw_sha256": (
        "28e852792b5de24b2d008c5bb3f95521da668927e555deb9eb3c508bb6b0e59f"
    ),
    "byte_count": 1_194_515,
    "scene_shards_reconstructed": 1_284,
    "scene_shards_expected": 1_284,
    "sole_failed_cell": {
        "family": "small_enclosed_maze",
        "stratum": "completion_enriched",
        "eligible_distinct_scenes": 0,
        "required_distinct_scenes": 5,
    },
}
FROZEN_FAILED_CENSUS_TASK_CENSUS = {
    "path": (
        ".generated/go2_branch_corpus_v1_2/scorer_fit/"
        "state_selector_feasibility_task_census.json"
    ),
    "raw_sha256": (
        "7ff35ec9feb864b1e9d6ef138a67874e6cce23e447e3de00d12773ca2ee56811"
    ),
    "byte_count": 1_300_740,
    "self_digest": (
        "0ee5fb6d073e6e8db33b0f63ce9b70b8346ba12f29f729f06c06de5982fbe109"
    ),
    "scene_task_count": 1_284,
}

PREDECESSOR_AMENDMENT_DIGEST = (
    "69e11a3efe665c4591fa29748b2f13ad08938b92acde763bda10608f93768628"
)
PREDECESSOR_AMENDMENT_ARTIFACT = {
    "path": PREDECESSOR.AMENDMENT_ARTIFACT_PATH,
    "raw_sha256": (
        "907f23421cc0c4e22746b6fecc580bf4509b2cc904ef7f212800d2597795d663"
    ),
    "byte_count": 10_804,
}
PREDECESSOR_SUCCESSOR_SELECTION_DIGEST = (
    "8cf65cc016c28ad34f1e50246561e72ee9d0f9c1c253fe8e32a4203a35b73ebe"
)
PREDECESSOR_SELECTION_DIGEST = PREDECESSOR.PREDECESSOR_SELECTION_DIGEST
PREDECESSOR_SCORER_CONTRACT_DIGEST = (
    PREDECESSOR.PREDECESSOR_SCORER_CONTRACT_DIGEST
)
PREDECESSOR_CONTRACT_ARTIFACT = PREDECESSOR.PREDECESSOR_CONTRACT_ARTIFACT

ALLOCATION_AMENDMENT_DIGEST = (
    "4dde3562cdd9e503d6e264a5d4982a189a9f43d338c3d6b87ee20de352bc3cbc"
)
CANDIDATE_BANK_DIGEST = (
    "85471e44a0fe8f3c59fff258e9b23933e306f69b6d590c832e2b8da1f34a8cd9"
)
ALLOCATION_SOURCE_BINDING = {
    "path": "lewm/oracle/go2_candidate_allocation_v1_2.py",
    "sha256": "9ebd494d979f73fe63863731740418e60ccd2d6d3f61d1f11171c879f598d0b7",
    "byte_count": 51_923,
}
SLEW_SOURCE_BINDING = {
    "path": "scripts/dev_action_slew_reconstruction_v1.py",
    "sha256": "17075cc10bdfc637a630da1b495f064156be9d481b6be631f50fd1e370b9203e",
    "byte_count": 13_469,
}
PLATFORM_MANIFEST_BINDING = {
    "path": "config/go2_platform_manifest.yaml",
    "sha256": "5ac4a08b17cfaa3552f3c3ccd45930b8a929ac5ca31eb1f9440923f037c78189",
    "byte_count": 4_613,
}

COMPLETION_RADIUS_M = 0.75
# Compatibility name for callers that used V1's start-radius constant.  The
# value is the unchanged selector parameter called ``r_complete`` in the V2
# formula.  It is not the oracle completion predicate and is not the
# production collector's range/FOV/LOS claim predicate.
COMPLETION_MAX_GEODESIC_M = COMPLETION_RADIUS_M
COMPLETION_MAX_ABS_BEARING_DEG = 75.0
COMPLETION_MAX_ABS_BEARING_RAD = math.radians(
    COMPLETION_MAX_ABS_BEARING_DEG
)
HORIZON_S = 2.0
HORIZON_BLOCKS = 4
TICKS_PER_BLOCK = 5
HORIZON_TICKS = HORIZON_BLOCKS * TICKS_PER_BLOCK
TICK_DT_S = 0.1
V_MAX_MPS = 0.30
MAX_TRANSLATION_M = HORIZON_S * V_MAX_MPS

SCORER_FIT_SELECTION_PRIORITY = PREDECESSOR.SCORER_FIT_SELECTION_PRIORITY
REQUIRED_FAMILIES = PREDECESSOR.REQUIRED_FAMILIES
REQUIRED_STRATA = PREDECESSOR.REQUIRED_STRATA
PRESERVED_STATE_SHARDS = PREDECESSOR.PRESERVED_STATE_SHARDS
ORACLE_V1_2_DIGEST = PREDECESSOR.ORACLE_V1_2_DIGEST
COMPLETION_SEMANTIC_SOURCE_BINDINGS = (
    PREDECESSOR.COMPLETION_SEMANTIC_SOURCE_BINDINGS
)

# New names prevent the accepted failed V1 receipts from ever being
# overwritten.  Generic manifest binding keys remain stable downstream.
STATE_SELECTOR_FEASIBILITY_SCHEMA = (
    "go2_scorer_fit_state_selector_reachability_feasibility_receipt_v2"
)
STATE_SELECTOR_FEASIBILITY_PASS_STATUS = (
    "PASS_OUTCOME_FREE_REACHABILITY_FEASIBILITY"
)
STATE_SELECTOR_FEASIBILITY_RECEIPT_NAME = (
    "state_selector_reachability_feasibility_receipt_v2.json"
)
STATE_SELECTOR_FEASIBILITY_RECEIPT_PATH = (
    ".generated/go2_branch_corpus_v1_2/scorer_fit/"
    + STATE_SELECTOR_FEASIBILITY_RECEIPT_NAME
)
PRESERVED_STATE_PRECONTRACT_REVALIDATION_SCHEMA = (
    "go2_scorer_fit_preserved_state_precontract_revalidation_reachability_v2"
)
PRESERVED_STATE_PRECONTRACT_REVALIDATION_RECEIPT_NAME = (
    "preserved_state_precontract_revalidation_reachability_v2.json"
)
PRESERVED_STATE_PRECONTRACT_REVALIDATION_RECEIPT_PATH = (
    ".generated/go2_branch_corpus_v1_2/scorer_fit/"
    + PRESERVED_STATE_PRECONTRACT_REVALIDATION_RECEIPT_NAME
)
PRESERVED_STATE_REVALIDATION_SCHEMA = (
    "go2_scorer_fit_preserved_state_revalidation_reachability_v2"
)
PRESERVED_STATE_REVALIDATION_RECEIPT_NAME = (
    "preserved_state_revalidation_reachability_v2.json"
)
PRESERVED_STATE_REVALIDATION_RECEIPT_PATH = (
    ".generated/go2_branch_corpus_v1_2/scorer_fit/"
    + PRESERVED_STATE_REVALIDATION_RECEIPT_NAME
)
ACTIVE_SELECTOR_BINDING_KEYS = (
    "state_selector_amendment_digest",
    "state_selector_feasibility_receipt_digest",
    "preserved_state_revalidation_receipt_digest",
)

# The phase-1 preserved-identity redrive is the last gate that can directly
# attest that no scorer scientific output exists.  These are deliberately
# exact, narrow generated-output paths; the audit must not recursively inspect
# unrelated custody roots.  Later encoder/trainer/transfer consumers validate
# the frozen attestation and its receipt lineage, but do not require these
# paths to remain empty after legitimate branch generation begins.
PHASE1_OUTCOME_SURFACE_ATTESTATION_SCHEMA = (
    "go2_scorer_fit_phase1_outcome_surface_absence_attestation_v1"
)
PHASE1_STATE_CHECK_SHARD_SCHEMA = (
    "go2_scorer_fit_preserved_state_precontract_check_shard_reachability_v2"
)
PHASE1_STATE_CHECK_SHARD_STATUS = (
    "COMPLETE_OUTCOME_FREE_PRESERVED_STATE_PRECONTRACT_CHECK"
)
PHASE1_STATE_CHECK_SHARD_ROOT = (
    ".generated/go2_branch_corpus_v1_2/scorer_fit/"
    "preserved_state_precontract_check_shards_reachability_v2"
)
PHASE1_FORBIDDEN_EXACT_FILE_PATHS = (
    ".generated/go2_branch_corpus_v1_2/scorer_fit/state_manifest.json",
    ".generated/go2_branch_corpus_v1_2/scorer_fit/"
    "candidate_allocation_manifest.json",
    ".generated/go2_branch_corpus_v1_2/scorer_fit/branch_rows.jsonl",
    ".generated/go2_branch_corpus_v1_2/scorer_fit/corpus_receipt.json",
    ".generated/go2_branch_corpus_v1_2/scorer_fit/smoke_branch_receipt.json",
    ".generated/go2_branch_corpus_v1_2/scorer_fit/smoke_encoding_receipt.json",
    ".generated/go2_branch_corpus_v1_2/scorer_fit/latents_index.json",
    ".generated/go2_branch_corpus_v1_2/scorer_fit/"
    "encoding_invocation_summary.json",
    ".generated/go2_branch_corpus_v1_2/scorer_fit/context.f16",
    ".generated/go2_branch_corpus_v1_2/scorer_fit/horizon.f16",
    ".generated/go2_utility_scorer_v1_2/qualification.json",
    ".generated/go2_utility_scorer_v1_2/scorer_package.pt",
    ".generated/go2_utility_scorer_v1_2/scorer_package_receipt.json",
    ".generated/go2_utility_scorer_v1_2/"
    "counterfactual_development_transfer_v1_2/development_transfer_spec.json",
    ".generated/go2_utility_scorer_v1_2/"
    "counterfactual_development_transfer_v1_2/result.json",
)
PHASE1_FORBIDDEN_DIRECTORY_ROOTS = (
    ".generated/go2_branch_corpus_v1_2/scorer_fit/row_records",
    ".generated/go2_branch_corpus_v1_2/scorer_fit/frames",
    ".generated/go2_branch_corpus_v1_2/scorer_fit/latents/context",
    ".generated/go2_branch_corpus_v1_2/scorer_fit/latents/horizon",
    ".generated/go2_branch_corpus_v1_2/scorer_fit/"
    "invalid_attempts/redrive_records",
    ".generated/go2_branch_corpus_v1_2/scorer_fit/invalid_attempts/latents",
    ".generated/go2_utility_scorer_v1_2/registered_initialisations",
    ".generated/go2_utility_scorer_v1_2/training",
    ".generated/go2_utility_scorer_v1_2/invalid_attempts",
    ".generated/go2_utility_scorer_v1_2/"
    "counterfactual_development_transfer_v1_2",
)
PHASE1_FORBIDDEN_GLOB_PATTERNS = (
    ".generated/go2_branch_corpus_v1_2/scorer_fit/branch_summary_*.json",
    ".generated/go2_utility_scorer_v1_2/no_latent_baseline_*.pt",
    ".generated/go2_utility_scorer_v1_2/"
    "no_latent_baseline_*.receipt.json",
    ".generated/go2_utility_scorer_v1_2/failed_scorer_*.pt",
)

# Exact frozen bank.  This literal is checked against the already frozen bank
# digest below; it is not a new bank or an alternate candidate representation.
CANDIDATE_PLANS = (
    ("straight_fast", ("forward_fast",) * 4),
    ("straight_medium", ("forward_medium",) * 4),
    ("straight_slow", ("forward_slow",) * 4),
    ("arc_left_sustained", ("arc_left",) * 4),
    ("arc_right_sustained", ("arc_right",) * 4),
    ("turn_left_sustained", ("yaw_left",) * 4),
    ("turn_right_sustained", ("yaw_right",) * 4),
    ("turn_left_then_go", (
        "yaw_left", "yaw_left", "forward_medium", "forward_medium",
    )),
    ("turn_right_then_go", (
        "yaw_right", "yaw_right", "forward_medium", "forward_medium",
    )),
    ("go_then_turn_left", (
        "forward_medium", "forward_medium", "yaw_left", "yaw_left",
    )),
    ("reverse_then_turn", (
        "backward", "backward", "yaw_left", "yaw_left",
    )),
    ("hold_all", ("hold",) * 4),
)

_TASK_STATUS_KEYS = (
    "task_completed", "goal_claimed", "terminated", "truncated",
)
_HEX = frozenset("0123456789abcdef")


class StateSelectorAmendmentError(RuntimeError):
    """The final selector amendment or exact reachability evidence changed."""


def _sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _normalise_previous_applied(
    previous_applied_command: Sequence[float],
) -> tuple[float, float, float]:
    if isinstance(previous_applied_command, (str, bytes)):
        raise StateSelectorAmendmentError(
            "previous_applied_command must contain three numeric channels"
        )
    try:
        previous = tuple(float(value) for value in previous_applied_command)
    except (TypeError, ValueError) as exc:
        raise StateSelectorAmendmentError(
            "previous_applied_command must contain three numeric channels"
        ) from exc
    if len(previous) != 3 or not all(math.isfinite(value) for value in previous):
        raise StateSelectorAmendmentError(
            "previous_applied_command must contain three finite channels"
        )
    if previous[1] != 0.0:
        raise StateSelectorAmendmentError(
            "the frozen bank and corpus forbid lateral command history"
        )
    if not (-0.3 <= previous[0] <= 0.3 and -0.5 <= previous[2] <= 0.5):
        raise StateSelectorAmendmentError(
            "previous_applied_command exceeds the frozen platform envelope"
        )
    return previous


def candidate_bank_contract_digest() -> str:
    """Reproduce the already frozen candidate-bank digest from pure literals."""

    return _sha256({
        "bank": [[name, list(plan)] for name, plan in CANDIDATE_PLANS],
        "primitives": {
            name: list(command) for name, command in SLEW.COMMANDS.items()
        },
        "blocks": HORIZON_BLOCKS,
        "ticks": TICKS_PER_BLOCK,
    })


def _validate_frozen_sources() -> None:
    if candidate_bank_contract_digest() != CANDIDATE_BANK_DIGEST:
        raise StateSelectorAmendmentError("candidate bank literal changed")
    if ALLOCATION.allocation_amendment_digest() != ALLOCATION_AMENDMENT_DIGEST:
        raise StateSelectorAmendmentError("candidate allocation amendment changed")
    if ALLOCATION.CANDIDATE_BANK_DIGEST != CANDIDATE_BANK_DIGEST:
        raise StateSelectorAmendmentError("allocator candidate bank changed")
    if (
        SLEW.TICKS != TICKS_PER_BLOCK
        or SLEW.TICK_DT_S != TICK_DT_S
        or SLEW.RATES != (0.25, 0.25, 0.35)
        or len(CANDIDATE_PLANS) != 12
    ):
        raise StateSelectorAmendmentError("slew or horizon arithmetic changed")


def candidate_post_slew_plan(
    candidate_index: int,
    previous_applied_command: Sequence[float],
) -> tuple[tuple[float, float, float], ...]:
    """Return the exact twenty post-slew commands for one frozen candidate."""

    _validate_frozen_sources()
    if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
        raise StateSelectorAmendmentError("candidate_index must be an integer")
    if not 0 <= candidate_index < len(CANDIDATE_PLANS):
        raise StateSelectorAmendmentError("candidate_index must be in [0, 11]")
    previous = _normalise_previous_applied(previous_applied_command)
    ticks: list[tuple[float, float, float]] = []
    for primitive in CANDIDATE_PLANS[candidate_index][1]:
        reconstructed, previous = SLEW.reconstruct_block(primitive, previous)
        ticks.extend(tuple(float(value) for value in tick) for tick in reconstructed)
    if len(ticks) != HORIZON_TICKS or any(len(tick) != 3 for tick in ticks):
        raise StateSelectorAmendmentError("candidate plan is not exactly 20 ticks")
    return tuple(ticks)


def candidate_translational_path_length_m(
    candidate_index: int,
    previous_applied_command: Sequence[float],
) -> float:
    """Nominal path length from exact post-slew translational commands."""

    plan = candidate_post_slew_plan(candidate_index, previous_applied_command)
    length = math.fsum(
        math.hypot(tick[0], tick[1]) * TICK_DT_S for tick in plan
    )
    if not math.isfinite(length) or length < 0.0:
        raise StateSelectorAmendmentError("candidate path length is invalid")
    return float(length)


def candidate_rotation_index(candidate_indices: Sequence[int]) -> int:
    """Return the unique frozen rotation for an exact six-candidate subset."""

    if isinstance(candidate_indices, (str, bytes)):
        raise StateSelectorAmendmentError("candidate subset is malformed")
    try:
        raw_indices = tuple(candidate_indices)
        if any(isinstance(value, bool) for value in raw_indices):
            raise TypeError("boolean candidate index")
        indices = tuple(operator.index(value) for value in raw_indices)
    except (TypeError, ValueError) as exc:
        raise StateSelectorAmendmentError("candidate subset is malformed") from exc
    if (
        len(indices) != ALLOCATION.CANDIDATES_PER_STATE
        or indices != tuple(sorted(set(indices)))
        or any(value < 0 or value >= len(CANDIDATE_PLANS) for value in indices)
    ):
        raise StateSelectorAmendmentError(
            "candidate subset must contain six sorted unique bank indices"
        )
    matches = [
        rotation for rotation, block in enumerate(ALLOCATION.ROTATION_BLOCKS)
        if tuple(block) == indices
    ]
    if len(matches) != 1:
        raise StateSelectorAmendmentError(
            "candidate subset is not one unchanged allocator rotation"
        )
    return matches[0]


def max_deterministic_translational_path_length_m(
    candidate_indices: Sequence[int],
    previous_applied_command: Sequence[float],
) -> dict[str, Any]:
    """Compute exact ``L_max`` for one frozen allocated candidate subset."""

    rotation = candidate_rotation_index(candidate_indices)
    indices = list(ALLOCATION.ROTATION_BLOCKS[rotation])
    previous = _normalise_previous_applied(previous_applied_command)
    lengths = [
        {
            "candidate_index": index,
            "candidate_name": CANDIDATE_PLANS[index][0],
            "translational_path_length_m":
                candidate_translational_path_length_m(index, previous),
        }
        for index in indices
    ]
    l_max = max(row["translational_path_length_m"] for row in lengths)
    maximisers = [
        row["candidate_index"] for row in lengths
        if row["translational_path_length_m"] == l_max
    ]
    return {
        "candidate_indices": indices,
        "candidate_rotation_index": rotation,
        "previous_applied_command": list(previous),
        "candidate_path_lengths_m": lengths,
        "l_max_m": float(l_max),
        "l_max_candidate_indices": maximisers,
        "horizon_blocks": HORIZON_BLOCKS,
        "ticks_per_block": TICKS_PER_BLOCK,
        "horizon_ticks": HORIZON_TICKS,
        "tick_dt_s": TICK_DT_S,
        "horizon_s": HORIZON_S,
        "slew_rates_per_tick": list(SLEW.RATES),
        "uses_actual_previous_applied_command": True,
        "branch_execution_used": False,
        "realised_outcome_used": False,
    }


def completion_distance_gap_m(continuous_geodesic_m: float) -> float:
    distance = float(continuous_geodesic_m)
    if not math.isfinite(distance) or distance < 0.0:
        raise StateSelectorAmendmentError(
            "continuous geodesic distance must be finite and nonnegative"
        )
    return max(distance - COMPLETION_RADIUS_M, 0.0)


def _task_status_projection(task_status: Mapping[str, Any]) -> dict[str, bool | None]:
    if not isinstance(task_status, Mapping):
        raise StateSelectorAmendmentError("task_status must be a mapping")
    # Read only the four frozen snapshot flags.  In particular, iteration over
    # a caller's mapping (which might expose outcomes) is deliberately avoided.
    projected: dict[str, bool | None] = {}
    for key in _TASK_STATUS_KEYS:
        try:
            value = task_status[key]
        except KeyError:
            value = None
        projected[key] = value if isinstance(value, bool) else None
    return projected


def completion_enriched_eligibility(
    *,
    graph_hops: int,
    reachable: bool,
    continuous_geodesic_m: float,
    bearing_body_rad: float,
    task_status: Mapping[str, Any],
    candidate_indices: Sequence[int],
    previous_applied_command: Sequence[float],
) -> dict[str, Any]:
    """Evaluate the final completion-enrichment predicate for an exact mask."""

    budget = max_deterministic_translational_path_length_m(
        candidate_indices, previous_applied_command
    )
    status = _task_status_projection(task_status)
    reasons: list[str] = []
    try:
        gap = completion_distance_gap_m(continuous_geodesic_m)
        distance = float(continuous_geodesic_m)
    except StateSelectorAmendmentError:
        gap = math.inf
        distance = float(continuous_geodesic_m)
        reasons.append("completion_unreachable")
    if not bool(reachable) or not math.isfinite(distance):
        if "completion_unreachable" not in reasons:
            reasons.append("completion_unreachable")
    elif gap > budget["l_max_m"]:
        reasons.append(
            "completion_geodesic_gap_gt_allocated_subset_l_max"
        )
    bearing = float(bearing_body_rad)
    if (
        not math.isfinite(bearing)
        or abs(bearing) > COMPLETION_MAX_ABS_BEARING_RAD
    ):
        reasons.append("completion_bearing_gt_75deg")
    for key in _TASK_STATUS_KEYS:
        value = status[key]
        if value is None:
            reasons.append(f"completion_snapshot_{key}_unavailable")
        elif value:
            reasons.append(f"completion_snapshot_{key}")
    return {
        "eligible": not reasons,
        "rejection_reasons": reasons,
        "reachable": bool(reachable),
        "continuous_geodesic_m": distance,
        "completion_radius_m": COMPLETION_RADIUS_M,
        "continuous_geodesic_gap_m": float(gap),
        "bearing_body_rad": bearing,
        "abs_bearing_rad": abs(bearing),
        "max_abs_bearing_rad": COMPLETION_MAX_ABS_BEARING_RAD,
        "graph_hops_diagnostic": int(graph_hops),
        "task_status": status,
        **budget,
    }


def completion_rotation_eligibility_vector(
    *,
    graph_hops: int,
    reachable: bool,
    continuous_geodesic_m: float,
    bearing_body_rad: float,
    task_status: Mapping[str, Any],
    previous_applied_command: Sequence[float],
) -> dict[str, Any]:
    """Pre-identity evidence for all twelve possible allocator rotations.

    This is not an allocation and cannot authorize an identity manifest.  It
    prevents the pre-identity structural fixture from being misused as if it
    were the later canonical assignment.
    """

    rows = [
        completion_enriched_eligibility(
            graph_hops=graph_hops,
            reachable=reachable,
            continuous_geodesic_m=continuous_geodesic_m,
            bearing_body_rad=bearing_body_rad,
            task_status=task_status,
            candidate_indices=ALLOCATION.ROTATION_BLOCKS[rotation],
            previous_applied_command=previous_applied_command,
        )
        for rotation in range(ALLOCATION.CANDIDATE_COUNT)
    ]
    eligible_rotations = [
        row["candidate_rotation_index"] for row in rows if row["eligible"]
    ]
    return {
        "allocation_status": "DEFERRED_UNTIL_ALL_120_IDENTITIES",
        "is_candidate_assignment": False,
        "pre_identity_fixture_used_as_assignment": False,
        "rotation_count": len(rows),
        "rotations": rows,
        "eligible_rotation_indices": eligible_rotations,
        "eligible_under_at_least_one_rotation": bool(eligible_rotations),
        "eligible_under_every_rotation": len(eligible_rotations) == len(rows),
        "candidate_outcomes_used": False,
    }


def validate_allocated_completion_evidence(
    evidence: Mapping[str, Any],
    *,
    candidate_indices: Sequence[int],
    previous_applied_command: Sequence[float],
) -> None:
    """Fail closed unless stored evidence exactly matches the assigned mask."""

    if not isinstance(evidence, Mapping):
        raise StateSelectorAmendmentError("completion evidence must be a mapping")
    expected = completion_enriched_eligibility(
        graph_hops=int(evidence.get("graph_hops_diagnostic")),
        reachable=bool(evidence.get("reachable")),
        continuous_geodesic_m=float(evidence.get("continuous_geodesic_m")),
        bearing_body_rad=float(evidence.get("bearing_body_rad")),
        task_status=evidence.get("task_status", {}),
        candidate_indices=candidate_indices,
        previous_applied_command=previous_applied_command,
    )
    if dict(evidence) != expected:
        raise StateSelectorAmendmentError(
            "completion evidence does not match exact allocated-subset arithmetic"
        )
    if expected["eligible"] is not True:
        raise StateSelectorAmendmentError(
            "completion identity fails its exact allocated-subset reachability gate"
        )


def state_selector_amendment_contract() -> dict[str, Any]:
    """Return the final prospective selector amendment."""

    return {
        "schema": AMENDMENT_SCHEMA,
        "status": "AUTHORIZED_FINAL_PROSPECTIVE_PRE_OUTCOME_SELECTOR_AMENDMENT",
        "version": AMENDMENT_VERSION,
        "lineage": {
            "authorizing_clean_failure_commit": AUTHORIZING_FAILURE_COMMIT,
            "complete_census_failure_report": {
                "path": FAILURE_REPORT_PATH,
                "failure_report_digest": FAILURE_REPORT_DIGEST,
                "raw_sha256": FAILURE_REPORT_RAW_SHA256,
                "byte_count": FAILURE_REPORT_BYTE_COUNT,
            },
            "frozen_failed_census_receipt": FROZEN_FAILED_CENSUS_RECEIPT,
            "frozen_failed_census_task_census":
                FROZEN_FAILED_CENSUS_TASK_CENSUS,
            "predecessor_selector_amendment_digest":
                PREDECESSOR_AMENDMENT_DIGEST,
            "predecessor_selector_amendment_artifact":
                PREDECESSOR_AMENDMENT_ARTIFACT,
            "predecessor_successor_selection_digest":
                PREDECESSOR_SUCCESSOR_SELECTION_DIGEST,
            "unchanged_candidate_allocation_amendment_digest":
                ALLOCATION_AMENDMENT_DIGEST,
            "unchanged_candidate_bank_digest": CANDIDATE_BANK_DIGEST,
        },
        "superseded_start_distance_rule": {
            "status": "SUPERSEDED_PRE_OUTCOME_START_RADIUS_NOT_HORIZON_REACHABILITY",
            "rule": "snapshot continuous geodesic distance d0 <= 0.75 m",
            "scientific_defect": (
                "a start-state radius is not the intended within-horizon "
                "physical reachability enrichment criterion"
            ),
            "outcomes_existed_when_superseded": False,
        },
        "single_replacement": {
            "definition": "max(d0 - r_complete, 0) <= L_max",
            "d0": "frozen continuous metric-geodesic distance at snapshot time",
            "r_complete_m": COMPLETION_RADIUS_M,
            "r_complete_semantic_boundary": (
                "the unchanged 0.75 m selector parameter in this prospective "
                "snapshot reachability formula; not an oracle completion "
                "predicate and not the collector claim predicate"
            ),
            "l_max": (
                "maximum nominal translational path length among the exact six "
                "candidates allocated to this state over twenty post-slew ticks"
            ),
            "l_max_calculation": {
                "previous_command": "actual snapshot-time previous applied (vx, vy, yaw)",
                "candidate_plans": "exact requested four-block plans in frozen bank",
                "slew_limiter": "scripts.dev_action_slew_reconstruction_v1.reconstruct_block",
                "slew_rates_per_tick": list(SLEW.RATES),
                "blocks": HORIZON_BLOCKS,
                "ticks_per_block": TICKS_PER_BLOCK,
                "horizon_ticks": HORIZON_TICKS,
                "tick_dt_s": TICK_DT_S,
                "candidate_path_length": (
                    "math.fsum(hypot(post_slew_vx, post_slew_vy) * 0.1 "
                    "for the exact 20 ticks)"
                ),
                "subset_aggregation": "maximum across the exact six unique candidates",
                "branch_execution": False,
                "realised_outcome": False,
            },
            "unchanged_requirements": {
                "finite_graph_reachability": True,
                "absolute_body_bearing_deg_max": COMPLETION_MAX_ABS_BEARING_DEG,
                "snapshot_goal_claimed": False,
                "snapshot_task_completed": False,
                "snapshot_terminated": False,
                "snapshot_truncated": False,
                "canonical_snapshot_boundary": True,
                "all_oracle_and_scorer_inputs_available": True,
                "graph_hops": "diagnostic only",
            },
        },
        "allocation_circularity_resolution": {
            "pre_identity_fixture_is_not_an_assignment": True,
            "pre_identity_evidence": (
                "retain the exact L_max and eligibility result for each of the "
                "twelve unchanged candidate rotations"
            ),
            "unchanged_allocator": True,
            "small_family_is_resolved_last": True,
            "deterministic_search": {
                "fixed_rows": (
                    "the seven already feasible family shards plus the frozen "
                    "small-enclosed general and safety choices"
                ),
                "candidate_pool_order": (
                    "continue the unchanged one-pass lexical scene traversal "
                    "strictly after the scene cursor consumed while filling "
                    "the five general and five safety-enriched small-family "
                    "slots; never revisit an earlier skipped scene; retain the "
                    "first eligible canonical snapshot per remaining scene, "
                    "and within that snapshot choose the designated goal by "
                    "(distance, landmark id, landmark cell, graph hops diagnostic)"
                ),
                "combination_order": (
                    "lexicographic five-distinct-scene combinations from that "
                    "ordered small-enclosed completion pool"
                ),
                "per_combination_operation": (
                    "construct the provisional exact 120 identity projection, "
                    "run the unchanged canonical allocator once, and recompute "
                    "all forty completion predicates against their assigned masks"
                ),
                "choice": "first combination for which all forty exact-mask predicates pass",
                "failure": (
                    "if the cursor-restricted pool has fewer than five scenes "
                    "or no combination passes, issue and terminally reuse one "
                    "non-overwriting self-bound pre-outcome failure receipt; "
                    "do not alter allocator, selector, quota, candidate bank, "
                    "or state strata"
                ),
                "candidate_outcomes_consumed": False,
            },
            "phase_2_gate": (
                "before the active manifest or any branch, every completion "
                "identity must bind its actual previous command, exact assigned "
                "mask, exact L_max, distance gap, and passing predicate"
            ),
        },
        "census_reuse": {
            "accepted_failed_receipt_preserved_byte_exact": True,
            "accepted_1284_scene_shards_overwritten": False,
            "general_and_safety_reused_unchanged": True,
            "seven_previously_passing_completion_cells": (
                "cached gap-zero evidence may prove provisional eligibility; "
                "selected identities still receive the exact-mask phase-2 check"
            ),
            "small_enclosed_first_action": "recompute from 182 existing shards",
            "small_enclosed_redrive_boundary": (
                "redrive only small-enclosed scenes missing actual previous "
                "command or another required pre-outcome field"
            ),
            "unrelated_family_redrive": False,
            "actual_allocated_mask_check_required_before_manifest": True,
            "actual_allocated_mask_check_status": (
                "MANDATORY_DEFERRED_TO_JOINT_SEARCH_AND_PHASE2"
            ),
        },
        "new_non_overwriting_receipts": {
            "feasibility": {
                "schema": STATE_SELECTOR_FEASIBILITY_SCHEMA,
                "status": STATE_SELECTOR_FEASIBILITY_PASS_STATUS,
                "path": STATE_SELECTOR_FEASIBILITY_RECEIPT_PATH,
            },
            "preserved_phase_1": {
                "schema": PRESERVED_STATE_PRECONTRACT_REVALIDATION_SCHEMA,
                "path": PRESERVED_STATE_PRECONTRACT_REVALIDATION_RECEIPT_PATH,
            },
            "preserved_phase_2": {
                "schema": PRESERVED_STATE_REVALIDATION_SCHEMA,
                "path": PRESERVED_STATE_REVALIDATION_RECEIPT_PATH,
            },
            "active_manifest_binding_keys": list(ACTIVE_SELECTOR_BINDING_KEYS),
        },
        "preserved": {
            "candidate_bank": True,
            "candidate_bank_digest": CANDIDATE_BANK_DIGEST,
            "candidate_allocation_amendment": True,
            "candidate_allocation_amendment_digest": ALLOCATION_AMENDMENT_DIGEST,
            "six_candidates_per_state": True,
            "candidate_appearances_exactly_60": True,
            "state_total_120": True,
            "fit_calibration_96_24": True,
            "family_split_and_stratum_allocation": True,
            "five_states_per_family_stratum": True,
            "oracle_v1_2_progress_safety_completion": True,
            "reachability_formula_r_complete_m": COMPLETION_RADIUS_M,
            "oracle_v1_2_completion_at_or_before_horizon": True,
            "horizon_s": HORIZON_S,
            "horizon_blocks": HORIZON_BLOCKS,
            "render_preprocess_encoder": True,
            "scorer_architecture_and_qualification": True,
            "graph_hops_diagnostic_only": True,
            "completion_semantic_source_bindings":
                COMPLETION_SEMANTIC_SOURCE_BINDINGS,
            "completion_semantic_separation": {
                "oracle_v1_2_label": (
                    "the snapshot-bound goal graph cell is reached at a branch "
                    "tick at or before H4"
                ),
                "snapshot_production_goal_claim": (
                    "the collector's actual range-envelope, camera-FOV and "
                    "line-of-sight gates populate its claimed-cell set"
                ),
                "r_complete_0_75m": (
                    "selector-only parameter in max(d0-r_complete,0)<=L_max"
                ),
                "not_interchangeable": True,
            },
        },
        "forbidden_selector_inputs": [
            "simulator branch execution", "collision outcomes",
            "realised progress", "realised completion",
            "rendered future frames", "model predictions",
        ],
        "freeze_policy": {
            "this_is_final_permitted_pre_outcome_selector_amendment": True,
            "once_branch_outcomes_exist_selector_revision_forbidden": True,
            "no_state_replacement_from_completion_prevalence": True,
        },
        "source_bindings": {
            "candidate_allocation": ALLOCATION_SOURCE_BINDING,
            "slew_reconstruction": SLEW_SOURCE_BINDING,
            "platform_command_envelope": PLATFORM_MANIFEST_BINDING,
        },
    }


def state_selector_amendment_digest() -> str:
    return _sha256(state_selector_amendment_contract())


def validate_state_selector_amendment_artifact(
    artifact: Mapping[str, Any],
) -> None:
    if not isinstance(artifact, Mapping):
        raise StateSelectorAmendmentError("selector amendment must be a mapping")
    payload = dict(artifact)
    observed = payload.pop("state_selector_amendment_digest", None)
    if payload != state_selector_amendment_contract():
        raise StateSelectorAmendmentError(
            "tracked selector amendment differs from code contract"
        )
    if observed != state_selector_amendment_digest():
        raise StateSelectorAmendmentError("selector amendment digest mismatch")


def validate_authority_artifacts(root: Path = ROOT) -> None:
    """Fail closed on tracked failure, predecessor, source, and V2 authority."""

    PREDECESSOR.validate_authority_artifacts(root)
    failure = root / FAILURE_REPORT_PATH
    if (
        not failure.is_file()
        or failure.stat().st_size != FAILURE_REPORT_BYTE_COUNT
        or _file_sha256(failure) != FAILURE_REPORT_RAW_SHA256
    ):
        raise StateSelectorAmendmentError("exhaustive census failure binding failed")
    failure_payload = json.loads(failure.read_text())
    observed = failure_payload.pop("failure_report_digest", None)
    if observed != FAILURE_REPORT_DIGEST or _sha256(failure_payload) != observed:
        raise StateSelectorAmendmentError("exhaustive census failure self binding failed")
    for label, binding in (
        ("predecessor amendment", PREDECESSOR_AMENDMENT_ARTIFACT),
        ("candidate allocation", ALLOCATION_SOURCE_BINDING),
        ("slew reconstruction", SLEW_SOURCE_BINDING),
        ("platform command envelope", PLATFORM_MANIFEST_BINDING),
    ):
        source = root / str(binding["path"])
        if (
            not source.is_file()
            or source.stat().st_size != int(binding["byte_count"])
            or _file_sha256(source) != binding[
                "raw_sha256" if label == "predecessor amendment" else "sha256"
            ]
        ):
            raise StateSelectorAmendmentError(f"{label} source binding failed")
    amendment = root / AMENDMENT_ARTIFACT_PATH
    if not amendment.is_file():
        raise StateSelectorAmendmentError("V2 selector amendment artifact is missing")
    validate_state_selector_amendment_artifact(json.loads(amendment.read_text()))


def validate_no_outcome_surface() -> None:
    """Assert the pure selector's public signature has no outcome parameter."""

    forbidden = {
        "branch", "outcome", "collision", "progress", "completion_label",
        "future_frame", "prediction", "latent",
    }
    parameters = set(inspect.signature(completion_enriched_eligibility).parameters)
    if parameters.intersection(forbidden):
        raise StateSelectorAmendmentError("selector exposes a forbidden outcome input")


def _load_frozen_failed_census_receipt(root: Path) -> dict[str, Any]:
    path = root / str(FROZEN_FAILED_CENSUS_RECEIPT["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(FROZEN_FAILED_CENSUS_RECEIPT["byte_count"])
        or _file_sha256(path) != FROZEN_FAILED_CENSUS_RECEIPT["raw_sha256"]
    ):
        raise StateSelectorAmendmentError(
            "frozen failed census receipt byte binding failed"
        )
    receipt = json.loads(path.read_text())
    observed = receipt.get("state_selector_feasibility_receipt_digest")
    if (
        observed != FROZEN_FAILED_CENSUS_RECEIPT[
            "state_selector_feasibility_receipt_digest"
        ]
        or _sha256({key: value for key, value in receipt.items()
                    if key != "state_selector_feasibility_receipt_digest"})
        != observed
        or receipt.get("schema") != FROZEN_FAILED_CENSUS_RECEIPT["schema"]
        or receipt.get("status") != FROZEN_FAILED_CENSUS_RECEIPT["status"]
    ):
        raise StateSelectorAmendmentError(
            "frozen failed census receipt semantic binding failed"
        )
    return receipt


def _load_frozen_failed_census_tasks(
    root: Path, predecessor_receipt: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    binding = FROZEN_FAILED_CENSUS_TASK_CENSUS
    path = root / str(binding["path"])
    if (
        not path.is_file() or path.stat().st_size != int(binding["byte_count"])
        or _file_sha256(path) != binding["raw_sha256"]
    ):
        raise StateSelectorAmendmentError(
            "frozen selector task census byte binding failed"
        )
    census = json.loads(path.read_text())
    observed = census.get("state_selector_feasibility_task_census_digest")
    if (
        observed != binding["self_digest"]
        or _sha256({key: value for key, value in census.items()
                    if key != "state_selector_feasibility_task_census_digest"})
        != observed
        or census.get("scene_task_count") != binding["scene_task_count"]
    ):
        raise StateSelectorAmendmentError(
            "frozen selector task census semantic binding failed"
        )
    small_families = [row for row in census.get("families", [])
                      if row.get("family") == "small_enclosed_maze"]
    if len(small_families) != 1:
        raise StateSelectorAmendmentError(
            "frozen selector task census small-family coverage changed"
        )
    tasks = small_families[0].get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 182:
        raise StateSelectorAmendmentError(
            "frozen selector task census small-family count changed"
        )
    by_scene = {str(task.get("scene_id")): dict(task) for task in tasks
                if isinstance(task, Mapping)}
    if len(by_scene) != 182:
        raise StateSelectorAmendmentError(
            "frozen selector task census repeats a small scene"
        )
    predecessor_lineage = predecessor_receipt.get("scene_shard_lineage")
    if (
        not isinstance(predecessor_lineage, list)
        or len(predecessor_lineage) != 1_284
        or predecessor_receipt.get("scene_shard_lineage_digest")
        != _sha256(predecessor_lineage)
    ):
        raise StateSelectorAmendmentError(
            "frozen predecessor scene lineage changed"
        )
    predecessor_by_scene = {
        str(row.get("scene_id")): str(row.get("scene_shard_digest"))
        for row in predecessor_lineage
        if isinstance(row, Mapping)
        and row.get("family") == "small_enclosed_maze"
    }
    if len(predecessor_by_scene) != 182 or set(predecessor_by_scene) != set(by_scene):
        raise StateSelectorAmendmentError(
            "frozen predecessor small-scene lineage changed"
        )
    return by_scene, predecessor_by_scene


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    finite = sorted(float(value) for value in values
                    if math.isfinite(float(value)))
    if not finite:
        return {"count": 0, "min": None, "q1": None, "median": None,
                "mean": None, "q3": None, "max": None}

    def quantile(fraction: float) -> float:
        position = (len(finite) - 1) * fraction
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return finite[lower]
        weight = position - lower
        return finite[lower] * (1.0 - weight) + finite[upper] * weight

    return {
        "count": len(finite), "min": finite[0], "q1": quantile(0.25),
        "median": quantile(0.5), "mean": math.fsum(finite) / len(finite),
        "q3": quantile(0.75), "max": finite[-1],
    }


def _same_numeric_structure(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _same_numeric_structure(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same_numeric_structure(a, b) for a, b in zip(left, right)
        )
    if (isinstance(left, (int, float)) and not isinstance(left, bool)
            and isinstance(right, (int, float)) and not isinstance(right, bool)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return left == right


def _validate_small_scene_evidence(row: Mapping[str, Any]) -> None:
    exact_keys = {
        "family", "scene_id", "stratum", "first_eligible_block",
        "source_step", "boundary", "cell_id", "episode_id",
        "episode_cluster_id", "goal_landmark_id", "goal_landmark_cell",
        "goal_material_id", "continuous_geodesic_m", "completion_radius_m",
        "continuous_geodesic_gap_m", "abs_bearing_rad", "bearing_body_rad",
        "range_m", "goal_landmark_xy_m", "graph_hops_diagnostic",
        "body_clearance_m", "clearance_m", "previous_applied_command",
        "allocation_rotation_evidence",
        "completion_rotation_eligibility_vector", "eligible_rotation_indices",
        "passes_any_allowed_allocation", "passes_every_allowed_allocation",
        "admitted_by_horizon_reachability_amendment", "snapshot_task_status",
        "eligible_designated_goal_count_at_first_eligible_snapshot",
    }
    if set(row) != exact_keys:
        raise StateSelectorAmendmentError(
            "small reachability scene evidence key surface changed"
        )
    try:
        vector = completion_rotation_eligibility_vector(
            graph_hops=int(row["graph_hops_diagnostic"]),
            reachable=math.isfinite(float(row["continuous_geodesic_m"])),
            continuous_geodesic_m=float(row["continuous_geodesic_m"]),
            bearing_body_rad=float(row["bearing_body_rad"]),
            task_status=row["snapshot_task_status"],
            previous_applied_command=row["previous_applied_command"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StateSelectorAmendmentError(
            "small reachability scene evidence is malformed"
        ) from exc
    rotations = vector["rotations"]
    expected = {
        "completion_radius_m": COMPLETION_RADIUS_M,
        "continuous_geodesic_gap_m": max(
            float(row["continuous_geodesic_m"]) - COMPLETION_RADIUS_M, 0.0),
        "abs_bearing_rad": abs(float(row["bearing_body_rad"])),
        "completion_rotation_eligibility_vector": vector,
        "allocation_rotation_evidence": rotations,
        "eligible_rotation_indices": vector["eligible_rotation_indices"],
        "passes_any_allowed_allocation":
            vector["eligible_under_at_least_one_rotation"],
        "passes_every_allowed_allocation":
            vector["eligible_under_every_rotation"],
        "admitted_by_horizon_reachability_amendment":
            float(row["continuous_geodesic_m"]) > COMPLETION_RADIUS_M,
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise StateSelectorAmendmentError(
            "small reachability evidence arithmetic changed"
        )
    goal_count = row.get(
        "eligible_designated_goal_count_at_first_eligible_snapshot")
    if not isinstance(goal_count, int) or isinstance(goal_count, bool) or goal_count < 1:
        raise StateSelectorAmendmentError(
            "small reachability evidence lacks designated-goal count provenance"
        )


def _load_live_small_reachability_shards(
    receipt: Mapping[str, Any], root: Path,
    predecessor_receipt: Mapping[str, Any],
) -> list[dict[str, Any]]:
    lineage = receipt.get("small_scene_shard_lineage")
    if (
        not isinstance(lineage, list) or len(lineage) != 182
        or receipt.get("small_scene_shard_lineage_digest") != _sha256(lineage)
    ):
        raise StateSelectorAmendmentError(
            "small reachability shard lineage is incomplete"
        )
    exact_lineage_keys = {
        "family", "scene_id", "scene_task_digest",
        "predecessor_scene_shard_digest", "reachability_scene_shard_digest",
    }
    seen_scenes: set[str] = set()
    seen_tasks: set[str] = set()
    seen_shards: set[str] = set()
    shards: list[dict[str, Any]] = []
    frozen_tasks, predecessor_by_scene = _load_frozen_failed_census_tasks(
        root, predecessor_receipt)
    shard_root = root / (
        ".generated/go2_branch_corpus_v1_2/scorer_fit/"
        "state_selector_reachability_feasibility_scene_shards_v2/"
        "small_enclosed_maze"
    )
    for binding in lineage:
        if not isinstance(binding, Mapping) or set(binding) != exact_lineage_keys:
            raise StateSelectorAmendmentError(
                "small reachability lineage row is malformed"
            )
        scene_id = str(binding["scene_id"])
        task_digest = str(binding["scene_task_digest"])
        shard_digest = str(binding["reachability_scene_shard_digest"])
        if (
            binding["family"] != "small_enclosed_maze"
            or not _is_digest(task_digest) or not _is_digest(shard_digest)
            or not _is_digest(binding["predecessor_scene_shard_digest"])
            or scene_id in seen_scenes or task_digest in seen_tasks
            or shard_digest in seen_shards
            or scene_id not in frozen_tasks
            or task_digest != frozen_tasks[scene_id].get("scene_task_digest")
            or binding["predecessor_scene_shard_digest"]
            != predecessor_by_scene.get(scene_id)
        ):
            raise StateSelectorAmendmentError(
                "small reachability lineage identities are not unique"
            )
        seen_scenes.add(scene_id)
        seen_tasks.add(task_digest)
        seen_shards.add(shard_digest)
        path = shard_root / f"{task_digest}.json"
        if not path.is_file():
            raise StateSelectorAmendmentError(
                f"small reachability shard is missing: {scene_id}"
            )
        shard = json.loads(path.read_text())
        observed = shard.get("state_selector_reachability_scene_shard_digest")
        if (
            observed != shard_digest
            or _sha256({key: value for key, value in shard.items()
                        if key != "state_selector_reachability_scene_shard_digest"})
            != observed
            or shard.get("task") != frozen_tasks[scene_id]
            or shard.get("frozen_predecessor_scene_shard_digest")
            != predecessor_by_scene[scene_id]
        ):
            raise StateSelectorAmendmentError(
                f"small reachability shard self binding failed: {scene_id}"
            )
        shards.append(shard)
    return shards


def _validate_feasibility_reconstruction(
    receipt: Mapping[str, Any], *, predecessor_receipt: Mapping[str, Any],
    small_scene_shards: Sequence[Mapping[str, Any]],
) -> None:
    predecessor_by_family = {
        str(row["family"]): row for row in predecessor_receipt.get("families", [])
    }
    if set(predecessor_by_family) != set(REQUIRED_FAMILIES):
        raise StateSelectorAmendmentError(
            "frozen predecessor family coverage changed"
        )
    family_by_name = {
        str(row["family"]): row for row in receipt.get("families", [])
    }
    unavailable = {
        "available": False,
        "reason": "V1_CENSUS_DID_NOT_RETAIN_GOAL_LEVEL_TOTALS",
    }
    for family in REQUIRED_FAMILIES:
        if family == "small_enclosed_maze":
            continue
        expected = copy.deepcopy(predecessor_by_family[family])
        expected["provenance"] = "REUSED_FROZEN_1284_SCENE_CENSUS"
        for stratum in REQUIRED_STRATA:
            expected["strata"][stratum]["eligible_designated_goal_count"] = \
                dict(unavailable)
        completion = expected["strata"]["completion_enriched"]
        cached_rows = completion["scene_evidence"]
        if any(float(row["continuous_geodesic_m"]) > COMPLETION_RADIUS_M
               for row in cached_rows):
            raise StateSelectorAmendmentError(
                f"cached completion evidence is not mask-independent: {family}"
            )
        completion["reachability_reclassification"] = {
            "status": "PASS_MASK_INDEPENDENT_ALREADY_WITHIN_COMPLETION_RADIUS",
            "candidate_subset_needed_for_verdict": False,
            "eligible_distinct_scenes": len(cached_rows),
            "redrive_performed": False,
            "graph_hops_zero_retained_scene_count": sum(
                int(row["graph_hops_diagnostic"]) == 0 for row in cached_rows),
            "graph_hops_positive_retained_scene_count": sum(
                int(row["graph_hops_diagnostic"]) > 0 for row in cached_rows),
        }
        if family_by_name.get(family) != expected:
            raise StateSelectorAmendmentError(
                f"cached predecessor family row changed: {family}"
            )

    if len(small_scene_shards) != 182:
        raise StateSelectorAmendmentError("small redrive must contain 182 shards")
    evidence: list[dict[str, Any]] = []
    rejections: dict[str, int] = {}
    lineage: list[dict[str, Any]] = []
    for shard in small_scene_shards:
        task = shard.get("task")
        result = shard.get("scene_result")
        shard_keys = {
            "schema", "status", "complete", "binding_receipt",
            "source_repository_commit", "clean_source_binding_digest",
            "bound_implementations_digest", "successor_selection_digest",
            "state_selector_amendment_digest",
            "candidate_allocation_amendment_digest",
            "frozen_predecessor_feasibility_receipt_digest",
            "frozen_predecessor_scene_shard_digest", "task", "scene_result",
            "runtime_s", "selected_state_identities_created",
            "candidate_outcomes_loaded", "branch_identities_created",
            "branches_attempted", "frames_rendered", "target_latents_encoded",
            "scorer_training_started",
            "state_selector_reachability_scene_shard_digest",
        }
        if (
            not isinstance(task, Mapping) or not isinstance(result, Mapping)
            or set(shard) != shard_keys
            or shard.get("schema")
            != "go2_scorer_fit_state_selector_reachability_scene_shard_v2"
            or shard.get("status")
            != "COMPLETE_OUTCOME_FREE_REACHABILITY_SCENE_EVIDENCE_NO_IDENTITY"
            or shard.get("complete") is not True
            or shard.get("binding_receipt") is not False
            or shard.get("source_repository_commit")
            != receipt.get("source_repository_commit")
            or shard.get("clean_source_binding_digest")
            != receipt.get("clean_source_binding_digest")
            or shard.get("bound_implementations_digest")
            != receipt.get("bound_implementations_digest")
            or shard.get("successor_selection_digest")
            != receipt.get("successor_selection_digest")
            or shard.get("state_selector_amendment_digest")
            != state_selector_amendment_digest()
            or shard.get("candidate_allocation_amendment_digest")
            != ALLOCATION_AMENDMENT_DIGEST
            or shard.get("frozen_predecessor_feasibility_receipt_digest")
            != FROZEN_FAILED_CENSUS_RECEIPT[
                "state_selector_feasibility_receipt_digest"
            ]
            or result.get("family") != "small_enclosed_maze"
            or result.get("scene_id") != task.get("scene_id")
            or set(result) != {
                "family", "scene_id", "completion_scene_evidence",
                "rejection_counts",
            }
            or not isinstance(result.get("completion_scene_evidence"), list)
            or len(result["completion_scene_evidence"]) > 1
            or not isinstance(result.get("rejection_counts"), Mapping)
            or any(shard.get(key) not in (False, 0) for key in (
                "selected_state_identities_created", "candidate_outcomes_loaded",
                "branch_identities_created", "branches_attempted",
                "frames_rendered", "target_latents_encoded",
                "scorer_training_started",
            ))
        ):
            raise StateSelectorAmendmentError(
                "small reachability shard binding is malformed"
            )
        for row in result["completion_scene_evidence"]:
            if not isinstance(row, Mapping) or row.get("scene_id") != task["scene_id"]:
                raise StateSelectorAmendmentError(
                    "small reachability evidence scene mismatch"
                )
            _validate_small_scene_evidence(row)
            evidence.append(dict(row))
        for reason, count in result["rejection_counts"].items():
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise StateSelectorAmendmentError(
                    "small reachability rejection count is malformed"
                )
            rejections[str(reason)] = rejections.get(str(reason), 0) + count
        lineage.append({
            "family": "small_enclosed_maze",
            "scene_id": task["scene_id"],
            "scene_task_digest": task["scene_task_digest"],
            "predecessor_scene_shard_digest":
                shard["frozen_predecessor_scene_shard_digest"],
            "reachability_scene_shard_digest":
                shard["state_selector_reachability_scene_shard_digest"],
        })
    if receipt.get("small_scene_shard_lineage") != lineage:
        raise StateSelectorAmendmentError(
            "small reachability receipt differs from live shard lineage"
        )
    if len({row["scene_id"] for row in evidence}) != len(evidence):
        raise StateSelectorAmendmentError(
            "small reachability evidence repeats an eligible scene"
        )
    evidence.sort(key=lambda row: (row["scene_id"], row["first_eligible_block"]))
    d0 = [float(row["continuous_geodesic_m"]) for row in evidence]
    gaps = [float(row["continuous_geodesic_gap_m"]) for row in evidence]
    signed_gaps = [value - COMPLETION_RADIUS_M for value in d0]
    bearings = [float(row["abs_bearing_rad"]) for row in evidence]
    signed_bearings = [float(row["bearing_body_rad"]) for row in evidence]
    graph_hops = [float(row["graph_hops_diagnostic"]) for row in evidence]
    first_blocks = [float(row["first_eligible_block"]) for row in evidence]
    all_lmax = [float(rotation["l_max_m"]) for row in evidence
                for rotation in row["allocation_rotation_evidence"]]
    min_lmax = [min(float(rotation["l_max_m"])
                    for rotation in row["allocation_rotation_evidence"])
                for row in evidence]
    max_lmax = [max(float(rotation["l_max_m"])
                    for rotation in row["allocation_rotation_evidence"])
                for row in evidence]
    predecessor_small = predecessor_by_family["small_enclosed_maze"]
    expected_small = {
        "family": "small_enclosed_maze",
        "allowed_scene_count": 182,
        "scanned_scene_count": 182,
        "all_allowed_scenes_scanned": True,
        "verdict": "PASS" if len(evidence) >= 5 else "FAIL",
        "provenance": "SCOPED_SMALL_FAMILY_REDRIVE_REQUIRED_MISSING_V1_FIELDS",
        "strata": {
            "general": {
                **predecessor_small["strata"]["general"],
                "eligible_designated_goal_count": dict(unavailable),
            },
            "safety_enriched": {
                **predecessor_small["strata"]["safety_enriched"],
                "eligible_designated_goal_count": dict(unavailable),
            },
            "completion_enriched": {
                "required_distinct_scenes": 5,
                "eligible_distinct_scenes": len(evidence),
                "verdict": "PASS" if len(evidence) >= 5 else "FAIL",
                "actual_allocated_subset_verification":
                    "MANDATORY_POST_IDENTITY_PRE_OUTCOME",
                "distributions": {
                    "continuous_geodesic_m_d0": _distribution(d0),
                    "continuous_geodesic_gap_m_d0_minus_0_75_clamped":
                        _distribution(gaps),
                    "d0_minus_0_75_m_signed": _distribution(signed_gaps),
                    "abs_bearing_rad": _distribution(bearings),
                    "bearing_body_rad_signed": _distribution(signed_bearings),
                    "graph_hops_diagnostic": _distribution(graph_hops),
                    "first_eligible_block": _distribution(first_blocks),
                    "l_max_m_all_allowed_rotation_state_pairs":
                        _distribution(all_lmax),
                    "minimum_l_max_m_per_state_across_allowed_rotations":
                        _distribution(min_lmax),
                    "maximum_l_max_m_per_state_across_allowed_rotations":
                        _distribution(max_lmax),
                },
                "admitted_specifically_by_horizon_reachability_amendment": sum(
                    bool(row["admitted_by_horizon_reachability_amendment"])
                    for row in evidence
                ),
                "allocation_robust_distinct_scenes": sum(
                    bool(row["passes_every_allowed_allocation"])
                    for row in evidence
                ),
                "retained_first_eligible_scene_row_count": len(evidence),
                "eligible_state_count_semantics": (
                    "one retained first-eligible snapshot row per eligible scene; "
                    "not an exhaustive count of snapshots or designated goals"
                ),
                "eligible_designated_goal_count_at_retained_first_eligible_snapshots":
                    sum(int(row[
                        "eligible_designated_goal_count_at_first_eligible_snapshot"
                    ]) for row in evidence),
                "graph_hops_zero_retained_scene_count": sum(
                    int(row["graph_hops_diagnostic"]) == 0 for row in evidence),
                "graph_hops_positive_retained_scene_count": sum(
                    int(row["graph_hops_diagnostic"]) > 0 for row in evidence),
                "scene_evidence": evidence,
            },
        },
        "rejection_counts": dict(sorted(rejections.items())),
    }
    if not _same_numeric_structure(
        family_by_name.get("small_enclosed_maze"), expected_small
    ):
        raise StateSelectorAmendmentError(
            "small reachability family summary differs from live reconstruction"
        )


def validate_state_selector_feasibility_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_source_commit: str | None = None,
    expected_successor_selection_digest: str | None = None,
    expected_clean_source_binding_digest: str | None = None,
    expected_bound_implementations_digest: str | None = None,
    root: Path = ROOT,
    predecessor_receipt: Mapping[str, Any] | None = None,
    small_scene_shards: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Validate the non-overwriting V2 cached-census/scoped-redrive gate."""

    if not isinstance(receipt, Mapping):
        raise StateSelectorAmendmentError("selector feasibility receipt must be a mapping")
    payload = dict(receipt)
    observed = payload.pop("state_selector_feasibility_receipt_digest", None)
    if not _is_digest(observed) or _sha256(payload) != observed:
        raise StateSelectorAmendmentError("selector feasibility receipt self digest failed")
    exact_keys = {
        "schema", "status", "complete", "binding_receipt",
        "source_repository_commit", "clean_source_binding_digest",
        "bound_implementations_digest", "successor_selection_digest",
        "state_selector_amendment_digest",
        "candidate_allocation_amendment_digest", "frozen_predecessor",
        "reuse_policy", "family_count", "families",
        "small_scene_shard_lineage", "small_scene_shard_lineage_digest",
        "required_distinct_scenes_per_stratum", "candidate_allocation_changed",
        "selector_completion_radius_m", "horizon_s", "horizon_ticks",
        "selected_state_identities_created", "candidate_outcomes_loaded",
        "branch_identities_created", "branches_attempted", "frames_rendered",
        "target_latents_encoded", "scorer_training_started",
    }
    if set(payload) != exact_keys:
        raise StateSelectorAmendmentError(
            "selector feasibility receipt key surface changed"
        )
    if (
        payload.get("schema") != STATE_SELECTOR_FEASIBILITY_SCHEMA
        or payload.get("status") != STATE_SELECTOR_FEASIBILITY_PASS_STATUS
        or payload.get("complete") is not True
        or payload.get("binding_receipt") is not True
    ):
        raise StateSelectorAmendmentError(
            "selector reachability feasibility receipt is not complete/pass"
        )
    if payload.get("state_selector_amendment_digest") != state_selector_amendment_digest():
        raise StateSelectorAmendmentError("selector feasibility amendment mismatch")
    if payload.get("candidate_allocation_amendment_digest") != ALLOCATION_AMENDMENT_DIGEST:
        raise StateSelectorAmendmentError("selector feasibility allocation changed")
    if (
        expected_source_commit is not None
        and payload.get("source_repository_commit") != expected_source_commit
    ):
        raise StateSelectorAmendmentError("selector feasibility source commit mismatch")
    if (
        not isinstance(payload.get("source_repository_commit"), str)
        or len(payload["source_repository_commit"]) != 40
        or not all(character in _HEX
                   for character in payload["source_repository_commit"])
        or not _is_digest(payload.get("clean_source_binding_digest"))
        or not _is_digest(payload.get("bound_implementations_digest"))
    ):
        raise StateSelectorAmendmentError(
            "selector feasibility clean-source binding is malformed"
        )
    if (expected_clean_source_binding_digest is not None
            and payload.get("clean_source_binding_digest")
            != expected_clean_source_binding_digest):
        raise StateSelectorAmendmentError(
            "selector feasibility clean-source digest mismatch"
        )
    if (expected_bound_implementations_digest is not None
            and payload.get("bound_implementations_digest")
            != expected_bound_implementations_digest):
        raise StateSelectorAmendmentError(
            "selector feasibility implementation binding mismatch"
        )
    # The V2 cached-census receipt can bind the prospective selection either as
    # a top-level field or in the clean-source contract.  If the caller asks
    # for it, omission is a hard failure.
    if expected_successor_selection_digest is not None and payload.get(
        "successor_selection_digest"
    ) != expected_successor_selection_digest:
        raise StateSelectorAmendmentError("selector feasibility selection mismatch")
    predecessor = payload.get("frozen_predecessor")
    if not isinstance(predecessor, Mapping):
        raise StateSelectorAmendmentError("selector feasibility lacks predecessor binding")
    expected_predecessor = {
        "failure_report_digest": FAILURE_REPORT_DIGEST,
        "failure_report_raw_sha256": FAILURE_REPORT_RAW_SHA256,
        "feasibility_receipt_digest": FROZEN_FAILED_CENSUS_RECEIPT[
            "state_selector_feasibility_receipt_digest"
        ],
        "feasibility_receipt_raw_sha256": FROZEN_FAILED_CENSUS_RECEIPT[
            "raw_sha256"
        ],
        "task_census_digest": (
            "0ee5fb6d073e6e8db33b0f63ce9b70b8346ba12f29f729f06c06de5982fbe109"
        ),
        "scene_shard_count": 1_284,
        "preserved_unchanged": True,
    }
    if dict(predecessor) != expected_predecessor:
        raise StateSelectorAmendmentError("selector feasibility predecessor changed")
    forbidden = (
        "selected_state_identities_created", "candidate_outcomes_loaded",
        "branch_identities_created", "branches_attempted", "frames_rendered",
        "target_latents_encoded", "scorer_training_started",
    )
    if any(payload.get(key) not in (False, 0) for key in forbidden):
        raise StateSelectorAmendmentError(
            "selector feasibility contains forbidden scientific operations"
        )
    if (
        payload.get("family_count") != len(REQUIRED_FAMILIES)
        or payload.get("required_distinct_scenes_per_stratum") != 5
        or payload.get("candidate_allocation_changed") is not False
        or payload.get("selector_completion_radius_m") != COMPLETION_RADIUS_M
        or payload.get("horizon_s") != HORIZON_S
        or payload.get("horizon_ticks") != HORIZON_TICKS
    ):
        raise StateSelectorAmendmentError("selector feasibility global contract mismatch")
    reuse = payload.get("reuse_policy")
    if (
        not isinstance(reuse, Mapping)
        or reuse.get("general_and_safety_criteria_unchanged") is not True
        or reuse.get("seven_family_cached_completion_rows_reclassified") is not True
        or reuse.get("unrelated_family_redrives") != 0
        or reuse.get("small_enclosed_maze_redrive_scene_count") != 182
        or reuse.get("actual_allocated_mask_check_required_before_manifest")
        is not True
        or reuse.get("actual_allocated_mask_check_status")
        != "MANDATORY_DEFERRED_TO_JOINT_SEARCH_AND_PHASE2"
    ):
        raise StateSelectorAmendmentError("selector feasibility reuse policy mismatch")
    rows = payload.get("families")
    if not isinstance(rows, list) or len(rows) != len(REQUIRED_FAMILIES):
        raise StateSelectorAmendmentError("selector feasibility family coverage mismatch")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise StateSelectorAmendmentError("selector feasibility family row malformed")
        family = str(row.get("family"))
        if family in seen or family not in REQUIRED_FAMILIES:
            raise StateSelectorAmendmentError("selector feasibility family identity mismatch")
        seen.add(family)
        if row.get("verdict") != "PASS":
            raise StateSelectorAmendmentError(f"selector feasibility family {family} failed")
        strata = row.get("strata")
        if not isinstance(strata, Mapping) or set(strata) != set(REQUIRED_STRATA):
            raise StateSelectorAmendmentError(
                f"selector feasibility {family} stratum coverage mismatch"
            )
        for stratum in REQUIRED_STRATA:
            evidence = strata[stratum]
            if (
                not isinstance(evidence, Mapping)
                or int(evidence.get("eligible_distinct_scenes", -1)) < 5
                or evidence.get("required_distinct_scenes") != 5
                or evidence.get("verdict") != "PASS"
            ):
                raise StateSelectorAmendmentError(
                    f"selector feasibility {family}/{stratum} failed"
                )
        if family == "small_enclosed_maze" and (
            row.get("scanned_scene_count") != 182
            or row.get("all_allowed_scenes_scanned") is not True
        ):
            raise StateSelectorAmendmentError(
                "small-enclosed reachability redrive is not exhaustive"
            )
    if seen != set(REQUIRED_FAMILIES):
        raise StateSelectorAmendmentError("selector feasibility omitted a family")
    if predecessor_receipt is None:
        predecessor_receipt = _load_frozen_failed_census_receipt(root)
    if small_scene_shards is None:
        small_scene_shards = _load_live_small_reachability_shards(
            receipt, root, predecessor_receipt)
    _validate_feasibility_reconstruction(
        receipt, predecessor_receipt=predecessor_receipt,
        small_scene_shards=small_scene_shards,
    )


def validate_phase1_outcome_surface_absence_attestation(
    attestation: Mapping[str, Any],
) -> None:
    """Validate the frozen issuance-time absence audit structurally.

    This intentionally does not reopen the paths.  Branches, latents, scorer
    packages, and transfer outputs are legitimate after phase 1.  Downstream
    consumers instead revalidate this immutable attestation through the
    phase-1 receipt digest that the clean launch and combined phase-2 receipt
    both bind.
    """

    if not isinstance(attestation, Mapping):
        raise StateSelectorAmendmentError(
            "phase-1 outcome-surface attestation must be a mapping"
        )
    payload = dict(attestation)
    observed = payload.pop("attestation_digest", None)
    if not _is_digest(observed) or _sha256(payload) != observed:
        raise StateSelectorAmendmentError(
            "phase-1 outcome-surface attestation self digest failed"
        )
    required = {
        "schema", "status", "issuance_gate",
        "live_reopen_after_legitimate_outcomes", "exact_file_checks",
        "directory_root_checks", "glob_checks", "forbidden_artifact_count",
        "all_forbidden_artifacts_absent", "candidate_outcomes_loaded",
        "branches_attempted", "frames_rendered", "target_latents_encoded",
        "scorer_training_started", "predictor_checkpoints_opened",
    }
    if set(payload) != required:
        raise StateSelectorAmendmentError(
            "phase-1 outcome-surface attestation fields changed"
        )
    if (
        payload.get("schema") != PHASE1_OUTCOME_SURFACE_ATTESTATION_SCHEMA
        or payload.get("status") != "PASS_PRE_OUTCOME_SURFACE_ABSENT"
        or payload.get("issuance_gate")
        != "BEFORE_PHASE1_REDRIVE_AND_BEFORE_ANY_SCIENTIFIC_OUTCOME"
        or payload.get("live_reopen_after_legitimate_outcomes") is not False
        or payload.get("forbidden_artifact_count") != 0
        or payload.get("all_forbidden_artifacts_absent") is not True
        or payload.get("candidate_outcomes_loaded") is not False
        or payload.get("branches_attempted") != 0
        or payload.get("frames_rendered") != 0
        or payload.get("target_latents_encoded") != 0
        or payload.get("scorer_training_started") is not False
        or payload.get("predictor_checkpoints_opened") != 0
    ):
        raise StateSelectorAmendmentError(
            "phase-1 outcome-surface absence verdict is not complete/pass"
        )

    file_checks = payload.get("exact_file_checks")
    if not isinstance(file_checks, list) or len(file_checks) != len(
        PHASE1_FORBIDDEN_EXACT_FILE_PATHS
    ):
        raise StateSelectorAmendmentError(
            "phase-1 exact-file absence coverage changed"
        )
    for row, expected_path in zip(
        file_checks, PHASE1_FORBIDDEN_EXACT_FILE_PATHS, strict=True
    ):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "exists", "kind", "artifact_absent"}
            or row.get("path") != expected_path
            or row.get("exists") is not False
            or row.get("kind") != "absent"
            or row.get("artifact_absent") is not True
        ):
            raise StateSelectorAmendmentError(
                f"phase-1 forbidden file was present or unbound: {expected_path}"
            )

    directory_checks = payload.get("directory_root_checks")
    if not isinstance(directory_checks, list) or len(directory_checks) != len(
        PHASE1_FORBIDDEN_DIRECTORY_ROOTS
    ):
        raise StateSelectorAmendmentError(
            "phase-1 directory absence coverage changed"
        )
    for row, expected_path in zip(
        directory_checks, PHASE1_FORBIDDEN_DIRECTORY_ROOTS, strict=True
    ):
        if (
            not isinstance(row, Mapping)
            or set(row) != {
                "path", "exists", "kind", "descendant_artifact_count",
                "descendant_artifacts", "artifact_absent",
            }
            or row.get("path") != expected_path
            or not isinstance(row.get("exists"), bool)
            or row.get("kind") not in {"absent", "directory"}
            or row.get("descendant_artifact_count") != 0
            or row.get("descendant_artifacts") != []
            or row.get("artifact_absent") is not True
            or (row.get("kind") == "absent" and row.get("exists") is not False)
            or (row.get("kind") == "directory" and row.get("exists") is not True)
        ):
            raise StateSelectorAmendmentError(
                f"phase-1 forbidden directory contained an artifact: {expected_path}"
            )

    glob_checks = payload.get("glob_checks")
    if not isinstance(glob_checks, list) or len(glob_checks) != len(
        PHASE1_FORBIDDEN_GLOB_PATTERNS
    ):
        raise StateSelectorAmendmentError(
            "phase-1 glob absence coverage changed"
        )
    for row, expected_pattern in zip(
        glob_checks, PHASE1_FORBIDDEN_GLOB_PATTERNS, strict=True
    ):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"pattern", "match_count", "matches", "artifact_absent"}
            or row.get("pattern") != expected_pattern
            or row.get("match_count") != 0
            or row.get("matches") != []
            or row.get("artifact_absent") is not True
        ):
            raise StateSelectorAmendmentError(
                f"phase-1 forbidden glob matched an artifact: {expected_pattern}"
            )


def validate_phase1_state_check_shard(
    shard: Mapping[str, Any],
    *,
    expected_state: Mapping[str, Any] | None = None,
    expected_predecessor_shard: Mapping[str, Any] | None = None,
    expected_source_commit: str | None = None,
    expected_successor_selection_digest: str | None = None,
    expected_feasibility_receipt_digest: str | None = None,
    expected_outcome_surface_attestation_digest: str | None = None,
) -> None:
    """Validate one durable outcome-free phase-1 redrive check.

    A worker writes this shard before releasing its native simulator context.
    The parent may therefore accept a nonzero teardown return code only after
    this exact self-bound record has been reopened successfully.
    """

    if not isinstance(shard, Mapping):
        raise StateSelectorAmendmentError(
            "phase-1 state-check shard must be a mapping"
        )
    payload = dict(shard)
    observed = payload.pop("state_check_shard_digest", None)
    if not _is_digest(observed) or _sha256(payload) != observed:
        raise StateSelectorAmendmentError(
            "phase-1 state-check shard self digest failed"
        )
    required = {
        "schema", "status", "complete", "binding_receipt",
        "source_repository_commit", "clean_source_binding_digest",
        "bound_implementations_digest", "successor_selection_digest",
        "state_selector_amendment_digest",
        "state_selector_feasibility_receipt_digest",
        "outcome_surface_absence_attestation_digest",
        "predecessor_state_shard", "family", "stratum", "scene_id",
        "state_id", "state_identity_digest", "source_state_digest",
        "candidate_outcomes_loaded", "branch_identities_created",
        "branches_attempted", "frames_rendered", "target_latents_encoded",
        "scorer_training_started", "predictor_checkpoints_opened",
        "state_check",
    }
    if set(payload) != required:
        raise StateSelectorAmendmentError(
            "phase-1 state-check shard fields changed"
        )
    if (
        payload.get("schema") != PHASE1_STATE_CHECK_SHARD_SCHEMA
        or payload.get("status") != PHASE1_STATE_CHECK_SHARD_STATUS
        or payload.get("complete") is not True
        or payload.get("binding_receipt") is not False
        or payload.get("state_selector_amendment_digest")
        != state_selector_amendment_digest()
        or not _is_digest(payload.get("clean_source_binding_digest"))
        or not _is_digest(payload.get("bound_implementations_digest"))
        or not _is_digest(payload.get("successor_selection_digest"))
        or not _is_digest(
            payload.get("state_selector_feasibility_receipt_digest")
        )
        or not _is_digest(
            payload.get("outcome_surface_absence_attestation_digest")
        )
        or not _is_digest(payload.get("state_identity_digest"))
        or not _is_digest(payload.get("source_state_digest"))
        or payload.get("candidate_outcomes_loaded") is not False
        or payload.get("branch_identities_created") is not False
        or payload.get("branches_attempted") != 0
        or payload.get("frames_rendered") != 0
        or payload.get("target_latents_encoded") != 0
        or payload.get("scorer_training_started") is not False
        or payload.get("predictor_checkpoints_opened") != 0
    ):
        raise StateSelectorAmendmentError(
            "phase-1 state-check shard is not complete/outcome-free"
        )
    for observed_value, expected_value, label in (
        (payload.get("source_repository_commit"), expected_source_commit,
         "source commit"),
        (payload.get("successor_selection_digest"),
         expected_successor_selection_digest, "successor selection"),
        (payload.get("state_selector_feasibility_receipt_digest"),
         expected_feasibility_receipt_digest, "selector feasibility"),
        (payload.get("outcome_surface_absence_attestation_digest"),
         expected_outcome_surface_attestation_digest,
         "outcome-surface attestation"),
    ):
        if expected_value is not None and observed_value != expected_value:
            raise StateSelectorAmendmentError(
                f"phase-1 state-check shard {label} changed"
            )

    predecessor = payload.get("predecessor_state_shard")
    if not isinstance(predecessor, Mapping):
        raise StateSelectorAmendmentError(
            "phase-1 state-check predecessor shard binding is absent"
        )
    if (expected_predecessor_shard is not None
            and dict(predecessor) != dict(expected_predecessor_shard)):
        raise StateSelectorAmendmentError(
            "phase-1 state-check predecessor shard binding changed"
        )
    check = payload.get("state_check")
    base_check_keys = {
        "state_id", "state_identity_digest", "exclusion_checks_pass",
        "exact_redrive_pass", "amended_classification_pass",
        "goal_binding_unchanged", "oracle_completion_target_unchanged",
        "snapshot_production_designated_goal_claim_unchanged",
        "production_task_completion_reset_unchanged",
        "completion_state_task_status_all_false", "failure_reason",
    }
    allowed_check_keys = set(base_check_keys)
    if payload.get("stratum") == "completion_enriched":
        allowed_check_keys.add("completion_rotation_eligibility")
    if (
        not isinstance(check, Mapping)
        or not base_check_keys.issubset(check)
        or not set(check).issubset(allowed_check_keys)
        or check.get("state_id") != payload.get("state_id")
        or check.get("state_identity_digest")
        != payload.get("state_identity_digest")
        or any(not isinstance(check.get(key), bool) for key in (
            "exclusion_checks_pass", "exact_redrive_pass",
            "amended_classification_pass", "goal_binding_unchanged",
            "oracle_completion_target_unchanged",
            "snapshot_production_designated_goal_claim_unchanged",
            "production_task_completion_reset_unchanged",
            "completion_state_task_status_all_false",
        ))
        or (check.get("failure_reason") is not None
            and (not isinstance(check.get("failure_reason"), str)
                 or not check.get("failure_reason")))
    ):
        raise StateSelectorAmendmentError(
            "phase-1 state-check scientific check is malformed"
        )
    if check.get("failure_reason") is None and not all(
        check.get(key) is True for key in (
            "exclusion_checks_pass", "exact_redrive_pass",
            "amended_classification_pass", "goal_binding_unchanged",
            "oracle_completion_target_unchanged",
            "snapshot_production_designated_goal_claim_unchanged",
            "production_task_completion_reset_unchanged",
            "completion_state_task_status_all_false",
        )
    ):
        raise StateSelectorAmendmentError(
            "phase-1 passing state-check has a failed check"
        )
    if (payload.get("stratum") == "completion_enriched"
            and check.get("failure_reason") is None
            and not isinstance(
                check.get("completion_rotation_eligibility"), Mapping)):
        raise StateSelectorAmendmentError(
            "phase-1 passing completion check lacks rotation evidence"
        )

    if expected_state is not None:
        if (
            payload.get("source_state_digest") != _sha256(expected_state)
            or any(payload.get(key) != expected_state.get(key) for key in (
                "family", "stratum", "scene_id", "state_id",
                "state_identity_digest",
            ))
        ):
            raise StateSelectorAmendmentError(
                "phase-1 state-check source identity changed"
            )


def _validate_phase1_state_check_transport(
    payload: Mapping[str, Any], *, root: Path,
) -> None:
    transport = payload.get("state_check_subprocess_transport")
    provenance = payload.get("state_check_shard_provenance")
    required_transport = {
        "schema", "state_check_shard_schema", "state_check_shard_root",
        "state_count", "one_state_per_subprocess",
        "atomic_shard_write_before_native_cleanup",
        "return_code_ignored_only_after_valid_shard",
        "resume_scope", "candidate_outcomes_loaded",
        "state_check_shard_provenance_digest",
    }
    if (
        not isinstance(transport, Mapping)
        or set(transport) != required_transport
        or transport.get("schema")
        != "go2_scorer_fit_phase1_subprocess_transport_v1"
        or transport.get("state_check_shard_schema")
        != PHASE1_STATE_CHECK_SHARD_SCHEMA
        or transport.get("state_check_shard_root")
        != PHASE1_STATE_CHECK_SHARD_ROOT
        or transport.get("state_count") != 45
        or transport.get("one_state_per_subprocess") is not True
        or transport.get("atomic_shard_write_before_native_cleanup") is not True
        or transport.get("return_code_ignored_only_after_valid_shard") is not True
        or transport.get("resume_scope")
        != "MISSING_OR_INVALID_STATE_CHECK_SHARDS_ONLY"
        or transport.get("candidate_outcomes_loaded") is not False
        or not isinstance(provenance, list)
        or len(provenance) != 45
        or transport.get("state_check_shard_provenance_digest")
        != _sha256(provenance)
    ):
        raise StateSelectorAmendmentError(
            "phase-1 subprocess state-check transport is incomplete"
        )

    predecessor_shards = load_preserved_state_shards(root)
    expected_rows: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for expected_shard in PRESERVED_STATE_SHARDS:
        family = str(expected_shard["family"])
        for state in predecessor_shards[family]["states"]:
            expected_rows.append((expected_shard, state))
    aggregate_checks: dict[str, Mapping[str, Any]] = {}
    for receipt_shard in payload.get("shards", []):
        for check in receipt_shard.get("state_checks", []):
            identity = str(check.get("state_identity_digest", ""))
            if identity in aggregate_checks:
                raise StateSelectorAmendmentError(
                    "phase-1 aggregate repeats a state check"
                )
            aggregate_checks[identity] = check
    if len(aggregate_checks) != 45:
        raise StateSelectorAmendmentError(
            "phase-1 aggregate does not contain 45 state checks"
        )

    provenance_keys = {
        "family", "state_id", "state_identity_digest", "path",
        "raw_sha256", "byte_count", "state_check_shard_digest",
    }
    shard_root = (root / PHASE1_STATE_CHECK_SHARD_ROOT).resolve()
    for row, (expected_shard, expected_state) in zip(
        provenance, expected_rows, strict=True
    ):
        identity = str(expected_state["state_identity_digest"])
        expected_relative = f"{PHASE1_STATE_CHECK_SHARD_ROOT}/{identity}.json"
        if (
            not isinstance(row, Mapping)
            or set(row) != provenance_keys
            or row.get("family") != expected_state.get("family")
            or row.get("state_id") != expected_state.get("state_id")
            or row.get("state_identity_digest") != identity
            or row.get("path") != expected_relative
            or not _is_digest(row.get("raw_sha256"))
            or not isinstance(row.get("byte_count"), int)
            or isinstance(row.get("byte_count"), bool)
            or row.get("byte_count") <= 0
            or not _is_digest(row.get("state_check_shard_digest"))
        ):
            raise StateSelectorAmendmentError(
                "phase-1 state-check provenance changed"
            )
        path = root / expected_relative
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise StateSelectorAmendmentError(
                "phase-1 state-check shard is missing"
            ) from exc
        if (
            path.is_symlink()
            or resolved != shard_root / f"{identity}.json"
            or not resolved.is_file()
            or resolved.stat().st_size != row["byte_count"]
            or _file_sha256(resolved) != row["raw_sha256"]
        ):
            raise StateSelectorAmendmentError(
                "phase-1 state-check shard bytes changed"
            )
        try:
            shard = json.loads(resolved.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise StateSelectorAmendmentError(
                "phase-1 state-check shard JSON is invalid"
            ) from exc
        validate_phase1_state_check_shard(
            shard,
            expected_state=expected_state,
            expected_predecessor_shard=expected_shard,
            expected_source_commit=payload.get("source_repository_commit"),
            expected_successor_selection_digest=payload.get(
                "successor_selection_digest"
            ),
            expected_feasibility_receipt_digest=payload.get(
                "state_selector_feasibility_receipt_digest"
            ),
            expected_outcome_surface_attestation_digest=payload.get(
                "outcome_surface_absence_attestation_digest"
            ),
        )
        if (
            shard.get("clean_source_binding_digest")
            != payload.get("clean_source_binding_digest")
            or shard.get("bound_implementations_digest")
            != payload.get("bound_implementations_digest")
            or shard.get("state_check_shard_digest")
            != row.get("state_check_shard_digest")
            or shard.get("state_check") != aggregate_checks.get(identity)
        ):
            raise StateSelectorAmendmentError(
                "phase-1 aggregate differs from its durable state-check shard"
            )


def _as_predecessor_phase_1_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the unchanged V1 identity/exclusion checks for reuse by V2."""

    projected = dict(receipt)
    projected.pop("preserved_state_precontract_revalidation_receipt_digest", None)
    projected.pop("outcome_surface_absence_attestation", None)
    projected.pop("outcome_surface_absence_attestation_digest", None)
    projected.pop("outcome_surface_absence_verified_at_phase1_issuance", None)
    projected.pop("clean_source_binding_digest", None)
    projected.pop("bound_implementations_digest", None)
    projected.pop("state_check_subprocess_transport", None)
    projected.pop("state_check_shard_provenance", None)
    projected["schema"] = PREDECESSOR.PRESERVED_STATE_PRECONTRACT_REVALIDATION_SCHEMA
    projected["state_selector_amendment_digest"] = (
        PREDECESSOR.state_selector_amendment_digest()
    )
    projected["preserved_state_precontract_revalidation_receipt_digest"] = _sha256(
        projected
    )
    return projected


def validate_preserved_state_precontract_revalidation_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_source_commit: str | None = None,
    expected_successor_selection_digest: str | None = None,
    expected_feasibility_receipt_digest: str | None = None,
    root: Path = ROOT,
) -> None:
    """Validate 45 preserved identities plus pre-allocation rotation evidence."""

    if not isinstance(receipt, Mapping):
        raise StateSelectorAmendmentError("phase-1 state receipt must be a mapping")
    payload = dict(receipt)
    observed = payload.pop(
        "preserved_state_precontract_revalidation_receipt_digest", None
    )
    if not _is_digest(observed) or _sha256(payload) != observed:
        raise StateSelectorAmendmentError("phase-1 state receipt self digest failed")
    if (
        payload.get("schema") != PRESERVED_STATE_PRECONTRACT_REVALIDATION_SCHEMA
        or payload.get("status") != "PASS_PRECONTRACT_IDENTITY_REVALIDATION"
        or payload.get("complete") is not True
        or payload.get("state_selector_amendment_digest")
        != state_selector_amendment_digest()
    ):
        raise StateSelectorAmendmentError("phase-1 state receipt is not V2 complete/pass")
    outcome_surface = payload.get("outcome_surface_absence_attestation")
    validate_phase1_outcome_surface_absence_attestation(outcome_surface)
    if (
        payload.get("outcome_surface_absence_verified_at_phase1_issuance")
        is not True
        or payload.get("outcome_surface_absence_attestation_digest")
        != outcome_surface.get("attestation_digest")
        or not _is_digest(payload.get("clean_source_binding_digest"))
        or not _is_digest(payload.get("bound_implementations_digest"))
    ):
        raise StateSelectorAmendmentError(
            "phase-1 receipt does not bind its issuance-time outcome absence audit"
        )
    _validate_phase1_state_check_transport(payload, root=root)
    # Reuse the predecessor's exact byte-bound identity, exclusion, redrive,
    # goal, and task-semantic validator.  Strip only the V2 rotation evidence
    # before projecting because the V1 validator intentionally rejects extras.
    predecessor_receipt = _as_predecessor_phase_1_receipt(receipt)
    predecessor_receipt["shards"] = [
        {
            **dict(shard),
            "state_checks": [
                {
                    key: value for key, value in check.items()
                    if key != "completion_rotation_eligibility"
                }
                for check in shard.get("state_checks", [])
            ],
        }
        for shard in predecessor_receipt.get("shards", [])
    ]
    predecessor_receipt[
        "preserved_state_precontract_revalidation_receipt_digest"
    ] = _sha256({
        key: value for key, value in predecessor_receipt.items()
        if key != "preserved_state_precontract_revalidation_receipt_digest"
    })
    PREDECESSOR.validate_preserved_state_precontract_revalidation_receipt(
        predecessor_receipt,
        expected_source_commit=expected_source_commit,
        expected_successor_selection_digest=expected_successor_selection_digest,
        expected_feasibility_receipt_digest=expected_feasibility_receipt_digest,
        root=root,
    )
    predecessor_shards = load_preserved_state_shards(root)
    expected_states = {
        str(state["state_identity_digest"]): state
        for shard in predecessor_shards.values() for state in shard["states"]
    }
    completion_count = 0
    for shard in payload.get("shards", []):
        for check in shard.get("state_checks", []):
            source = expected_states.get(str(check.get("state_identity_digest")))
            if source is None:
                raise StateSelectorAmendmentError("phase-1 evidence has unknown identity")
            if source.get("stratum") != "completion_enriched":
                continue
            completion_count += 1
            vector = check.get("completion_rotation_eligibility")
            if not isinstance(vector, Mapping):
                raise StateSelectorAmendmentError(
                    "preserved completion identity lacks rotation evidence"
                )
            expected_vector = completion_rotation_eligibility_vector(
                graph_hops=int(vector["rotations"][0]["graph_hops_diagnostic"]),
                reachable=bool(vector["rotations"][0]["reachable"]),
                continuous_geodesic_m=float(
                    vector["rotations"][0]["continuous_geodesic_m"]
                ),
                bearing_body_rad=float(vector["rotations"][0]["bearing_body_rad"]),
                task_status=vector["rotations"][0]["task_status"],
                previous_applied_command=vector["rotations"][0][
                    "previous_applied_command"
                ],
            )
            if dict(vector) != expected_vector:
                raise StateSelectorAmendmentError(
                    "preserved completion rotation evidence changed"
                )
            if vector.get("eligible_under_at_least_one_rotation") is not True:
                raise StateSelectorAmendmentError(
                    "preserved completion identity is infeasible for every rotation"
                )
    if completion_count != 15:
        raise StateSelectorAmendmentError(
            "phase-1 must cover exactly 15 preserved completion identities"
        )


def _normalise_completion_check(
    row: Mapping[str, Any],
    *,
    assignment: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "state_identity_digest", "state_id", "family", "stratum",
        "candidate_indices", "previous_applied_command", "completion_eligibility",
    }
    if not isinstance(row, Mapping) or set(row) != required:
        raise StateSelectorAmendmentError(
            "completion reachability row has unexpected or missing fields"
        )
    if (
        row["state_identity_digest"] != assignment["state_identity_digest"]
        or row["state_id"] != assignment["state_id"]
        or row["family"] != assignment["family"]
        or row["stratum"] != "completion_enriched"
        or list(row["candidate_indices"]) != assignment["candidate_indices"]
    ):
        raise StateSelectorAmendmentError(
            "completion reachability row differs from allocated identity"
        )
    validate_allocated_completion_evidence(
        row["completion_eligibility"],
        candidate_indices=row["candidate_indices"],
        previous_applied_command=row["previous_applied_command"],
    )
    return {
        "state_identity_digest": str(row["state_identity_digest"]),
        "state_id": str(row["state_id"]),
        "family": str(row["family"]),
        "stratum": "completion_enriched",
        "candidate_indices": [int(value) for value in row["candidate_indices"]],
        "candidate_rotation_index": candidate_rotation_index(
            row["candidate_indices"]
        ),
        "previous_applied_command": list(
            _normalise_previous_applied(row["previous_applied_command"])
        ),
        "completion_eligibility": dict(row["completion_eligibility"]),
        "exact_allocated_mask_reachability_pass": True,
        "candidate_outcomes_loaded": False,
    }


def build_preserved_state_revalidation_receipt(
    *,
    allocation_manifest: Mapping[str, Any],
    completion_states: Sequence[Mapping[str, Any]],
    source_repository_commit: str,
    successor_selection_digest: str,
    state_selector_feasibility_receipt_digest: str,
    preserved_state_precontract_revalidation_receipt_digest: str,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Build the one phase-2 gate for 45 preserved and all 40 completion states."""

    ALLOCATION.validate_allocation_manifest(allocation_manifest)
    base = PREDECESSOR.build_preserved_state_revalidation_receipt(
        allocation_manifest=allocation_manifest,
        source_repository_commit=source_repository_commit,
        successor_selection_digest=successor_selection_digest,
        state_selector_feasibility_receipt_digest=
            state_selector_feasibility_receipt_digest,
        preserved_state_precontract_revalidation_receipt_digest=
            preserved_state_precontract_revalidation_receipt_digest,
        root=root,
    )
    assignments = {
        str(row["state_identity_digest"]): row
        for row in allocation_manifest["assignments"]
    }
    checks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in completion_states:
        state_digest = str(raw.get("state_identity_digest", ""))
        assignment = assignments.get(state_digest)
        if assignment is None or state_digest in seen:
            raise StateSelectorAmendmentError(
                "completion reachability rows are duplicated or unallocated"
            )
        seen.add(state_digest)
        checks.append(_normalise_completion_check(raw, assignment=assignment))
    expected_completion = {
        str(row["state_identity_digest"])
        for row in allocation_manifest["assignments"]
        if row["stratum"] == "completion_enriched"
    }
    if len(checks) != 40 or seen != expected_completion:
        raise StateSelectorAmendmentError(
            "phase-2 requires all 40 allocated completion identities exactly once"
        )
    checks.sort(key=lambda row: row["state_identity_digest"])
    payload = {
        key: value for key, value in base.items()
        if key != "preserved_state_revalidation_receipt_digest"
    }
    payload["schema"] = PRESERVED_STATE_REVALIDATION_SCHEMA
    payload["state_selector_amendment_digest"] = state_selector_amendment_digest()
    payload.update({
        "completion_enriched_state_count": len(checks),
        "completion_exact_allocated_reachability_pass_count": len(checks),
        "completion_exact_allocated_reachability_checks": checks,
        "completion_exact_allocated_reachability_set_digest": _sha256(checks),
        "all_completion_identities_pass_exact_allocated_mask": True,
    })
    payload["preserved_state_revalidation_receipt_digest"] = _sha256(payload)
    return payload


def validate_preserved_state_revalidation_receipt(
    receipt: Mapping[str, Any],
    *,
    allocation_manifest: Mapping[str, Any],
    expected_source_commit: str | None = None,
    expected_successor_selection_digest: str | None = None,
    expected_feasibility_receipt_digest: str | None = None,
    expected_precontract_revalidation_receipt_digest: str | None = None,
    root: Path = ROOT,
) -> None:
    """Validate one generic phase-2 digest over masks and 40 reachability rows."""

    if not isinstance(receipt, Mapping):
        raise StateSelectorAmendmentError("phase-2 state receipt must be a mapping")
    payload = dict(receipt)
    observed = payload.pop("preserved_state_revalidation_receipt_digest", None)
    if not _is_digest(observed) or _sha256(payload) != observed:
        raise StateSelectorAmendmentError("phase-2 state receipt self digest failed")
    if (
        payload.get("schema") != PRESERVED_STATE_REVALIDATION_SCHEMA
        or payload.get("status")
        != "PASS_POST_ALLOCATION_PRE_OUTCOME_STATE_REVALIDATION"
        or payload.get("complete") is not True
        or payload.get("state_selector_amendment_digest")
        != state_selector_amendment_digest()
    ):
        raise StateSelectorAmendmentError("phase-2 state receipt is not V2 complete/pass")
    for value, expected, label in (
        (payload.get("source_repository_commit"), expected_source_commit, "source"),
        (payload.get("successor_selection_digest"),
         expected_successor_selection_digest, "selection"),
        (payload.get("state_selector_feasibility_receipt_digest"),
         expected_feasibility_receipt_digest, "feasibility"),
        (payload.get("preserved_state_precontract_revalidation_receipt_digest"),
         expected_precontract_revalidation_receipt_digest, "phase-1"),
    ):
        if expected is not None and value != expected:
            raise StateSelectorAmendmentError(f"phase-2 {label} binding mismatch")
    completion_rows = payload.get(
        "completion_exact_allocated_reachability_checks"
    )
    if not isinstance(completion_rows, list):
        raise StateSelectorAmendmentError("phase-2 completion evidence is missing")
    # Rebuilding is intentionally the validator: it reuses the predecessor's
    # 45 exact mask checks and recomputes every one of the forty V2 predicates.
    source_rows = [
        {
            "state_identity_digest": row["state_identity_digest"],
            "state_id": row["state_id"],
            "family": row["family"],
            "stratum": row["stratum"],
            "candidate_indices": row["candidate_indices"],
            "previous_applied_command": row["previous_applied_command"],
            "completion_eligibility": row["completion_eligibility"],
        }
        for row in completion_rows
    ]
    expected = build_preserved_state_revalidation_receipt(
        allocation_manifest=allocation_manifest,
        completion_states=source_rows,
        source_repository_commit=str(payload["source_repository_commit"]),
        successor_selection_digest=str(payload["successor_selection_digest"]),
        state_selector_feasibility_receipt_digest=str(
            payload["state_selector_feasibility_receipt_digest"]
        ),
        preserved_state_precontract_revalidation_receipt_digest=str(
            payload["preserved_state_precontract_revalidation_receipt_digest"]
        ),
        root=root,
    )
    if dict(receipt) != expected:
        raise StateSelectorAmendmentError(
            "phase-2 receipt differs from exact masks/reachability reconstruction"
        )


# Outcome-free predecessor helpers whose semantics are unchanged in V2.
preserved_state_identity_set_digest = PREDECESSOR.preserved_state_identity_set_digest
load_preserved_state_shards = PREDECESSOR.load_preserved_state_shards
candidate_mask_digest = PREDECESSOR.candidate_mask_digest


_validate_frozen_sources()
