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
import struct
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from lewm.oracle import go2_candidate_allocation_v1_2 as ALLOCATION
from lewm.oracle import go2_scorer_state_selector_amendment_v1 as PREDECESSOR
from scripts import dev_action_slew_reconstruction_v1 as SLEW


ROOT = Path(__file__).resolve().parents[2]
MANAGED_GENERATED_ROOT_RELATIVE = Path(
    ".generated/go2_branch_corpus_v1_2"
)

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
# The complete 45-state V2 phase-1 receipt above is a frozen failed terminal.
# It is never overwritten or reinterpreted as a pass.  The prospectively
# authorised 37-retained/8-replacement disposition uses a distinct active
# schema and path while retaining the generic post-allocation binding below.
PRESERVED_STATE_MIXED_PRECONTRACT_DISPOSITION_SCHEMA = (
    "go2_scorer_fit_preserved_state_mixed_precontract_"
    "disposition_reachability_v2"
)
PRESERVED_STATE_MIXED_PRECONTRACT_DISPOSITION_RECEIPT_NAME = (
    "preserved_state_mixed_precontract_disposition_reachability_v2.json"
)
PRESERVED_STATE_MIXED_PRECONTRACT_DISPOSITION_RECEIPT_PATH = (
    ".generated/go2_branch_corpus_v1_2/scorer_fit/"
    + PRESERVED_STATE_MIXED_PRECONTRACT_DISPOSITION_RECEIPT_NAME
)
MIXED_PRECONTRACT_DISPOSITION_STATUS = (
    "PASS_PREOUTCOME_37_RETAINED_8_REPLACEMENT_DISPOSITION"
)
FROZEN_REACHABILITY_FEASIBILITY_PASS = {
    "path": STATE_SELECTOR_FEASIBILITY_RECEIPT_PATH,
    "raw_sha256": (
        "ece1dfc380797b7f651cd5bf9102dfab16f7505e389e7464aec01ba7b38f2758"
    ),
    "byte_count": 2_459_308,
    "receipt_digest": (
        "0e2013f40f506da6485bb5e2fe5a3108595243aeb9141a6437f8cac023642482"
    ),
    "source_repository_commit": "7047c601bf4fa3eb693bb94db41195d8f3f09451",
    "clean_source_binding_digest": (
        "ce075902a329ad1faf5145787748fd39a0c7eed43622a87a5d2298b06882c1f4"
    ),
    "bound_implementations_digest": (
        "ba78956878597fa71e08eeea0350a2e8d9f3c668466d32ac535bfb0ffcdf7fa8"
    ),
    "successor_selection_digest": (
        "0099c2d4d749f8d5a05cb20e209e8b8f977ae22dad2973da52597deff18dfa6b"
    ),
}
FROZEN_PRESERVED_PRECONTRACT_FAILURE = {
    "path": PRESERVED_STATE_PRECONTRACT_REVALIDATION_RECEIPT_PATH,
    "raw_sha256": (
        "2e49ed6e47caa98ed30ce45bddccd07cd3fad4c1950e5c6ff3fe051d0307de25"
    ),
    "byte_count": 334_260,
    "receipt_digest": (
        "0316e7f9b8462670eabe76da5fefc003274b4d08355373d14e1100cd6165e8e3"
    ),
    "source_repository_commit": "7047c601bf4fa3eb693bb94db41195d8f3f09451",
    "clean_source_binding_digest": (
        "ce075902a329ad1faf5145787748fd39a0c7eed43622a87a5d2298b06882c1f4"
    ),
    "bound_implementations_digest": (
        "ba78956878597fa71e08eeea0350a2e8d9f3c668466d32ac535bfb0ffcdf7fa8"
    ),
    "successor_selection_digest": (
        "0099c2d4d749f8d5a05cb20e209e8b8f977ae22dad2973da52597deff18dfa6b"
    ),
    "state_identity_set_digest": (
        "3924fd62c885342f97d93242a6ab46b6cedb0d1c35a57db5e9c52c71c16b2c38"
    ),
    "outcome_surface_absence_attestation_digest": (
        "c807e288204185c3f7670c3efdeae5719b9a92af7f2757656b159632ca705bb7"
    ),
}
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
_FULL_SNAPSHOT_TASK_STATUS_KEYS = frozenset({
    *_TASK_STATUS_KEYS,
    "production_claim_evidence",
    "production_task_completion_reset_evidence",
    "termination_flags",
})
_PRODUCTION_CLAIM_EVIDENCE_KEYS = frozenset({
    "active_collector_visited_accessor_callable",
    "active_collector_claimed_cells",
    "designated_goal_cell",
})
_PRODUCTION_RESET_EVIDENCE_KEYS = frozenset({
    "minimum_block_guard_pass",
    "scene_graph_available",
    "active_collector_route_like",
    "active_collector_non_revisit",
    "scene_landmark_cells_nonempty",
    "all_scene_landmark_cells_claimed",
})
_PRODUCTION_TERMINATION_FLAG_KEYS = frozenset({
    "fall", "out_of_bounds", "tipped", "nan",
})
_HEX = frozenset("0123456789abcdef")


def _binary32(value: float) -> float:
    """Round-trip one frozen literal through the runtime command dtype."""

    return float(struct.unpack("!f", struct.pack("!f", float(value)))[0])


# The rollout safety limiter constructs its absolute command bounds as
# ``np.float32`` and stores ``runner._last_executed`` as float32.  Decimal 0.3
# therefore has these exact observed endpoint encodings.  They are not a wider
# physical envelope: only the exact runtime representations are admitted.
PLATFORM_NOMINAL_VX_MIN_MPS = -0.3
PLATFORM_NOMINAL_VX_MAX_MPS = 0.3
PLATFORM_NOMINAL_MAX_YAW_RATE_RADPS = 0.5
PLATFORM_EXECUTED_VX_MIN_BINARY32_MPS = _binary32(
    PLATFORM_NOMINAL_VX_MIN_MPS)
PLATFORM_EXECUTED_VX_MAX_BINARY32_MPS = _binary32(
    PLATFORM_NOMINAL_VX_MAX_MPS)
PLATFORM_EXECUTED_YAW_MIN_BINARY32_RADPS = _binary32(
    -PLATFORM_NOMINAL_MAX_YAW_RATE_RADPS)
PLATFORM_EXECUTED_YAW_MAX_BINARY32_RADPS = _binary32(
    PLATFORM_NOMINAL_MAX_YAW_RATE_RADPS)


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


def _has_inaccessible_custody_component(path: Path) -> bool:
    return any(
        part == ".."
        or part == "sealed_test.json"
        or part == "sealed"
        or part.startswith("sealed_")
        for part in path.parts
    )


def _assert_no_generated_path_symlink(path: Path) -> None:
    """Reject custody names and every symlink in one absolute path."""

    if _has_inaccessible_custody_component(path):
        raise StateSelectorAmendmentError(
            "generated artifact crosses an inaccessible custody component"
        )
    absolute = path if path.is_absolute() else Path.cwd() / path
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise StateSelectorAmendmentError(
                "symlinked generated artifact component is inaccessible"
            )


def _managed_generated_artifact_path(
    path: Path, *, root: Path = ROOT,
) -> Path:
    """Pin one guarded artifact below the sole managed generated-root alias.

    The repository may relocate exactly
    ``.generated/go2_branch_corpus_v1_2`` behind one symlink.  No symlink is
    permitted before or below that exact alias.  Returning the canonical
    target path (rather than the alias path) ensures a subsequent alias swap
    cannot redirect the caller's read.
    """

    repository_root = Path(root)
    if not repository_root.is_absolute():
        repository_root = Path.cwd() / repository_root
    managed_root = repository_root / MANAGED_GENERATED_ROOT_RELATIVE
    artifact = Path(path)
    if not artifact.is_absolute():
        artifact = Path.cwd() / artifact
    for label, value in (
        ("managed generated root", managed_root),
        ("managed generated artifact", artifact),
    ):
        if _has_inaccessible_custody_component(value):
            raise StateSelectorAmendmentError(
                f"{label} crosses an inaccessible custody component"
            )
    try:
        relative = artifact.relative_to(managed_root)
    except ValueError as exc:
        raise StateSelectorAmendmentError(
            "generated artifact escaped the managed output root"
        ) from exc
    if not relative.parts:
        raise StateSelectorAmendmentError(
            "generated artifact path names only its managed root"
        )

    # Exactly the managed root may be an alias; its complete prefix may not.
    _assert_no_generated_path_symlink(managed_root.parent)
    if managed_root.is_symlink():
        raw_target = managed_root.readlink()
        target = (
            raw_target
            if raw_target.is_absolute()
            else managed_root.parent / raw_target
        )
        if (
            target.name != managed_root.name
            or _has_inaccessible_custody_component(target)
        ):
            raise StateSelectorAmendmentError(
                "managed generated-root alias target identity is inaccessible"
            )
        _assert_no_generated_path_symlink(target)
        try:
            canonical_root = target.resolve(strict=True)
        except OSError as exc:
            raise StateSelectorAmendmentError(
                "managed generated-root alias target is missing"
            ) from exc
    else:
        if not managed_root.is_dir():
            raise StateSelectorAmendmentError(
                "managed generated-output root is missing"
            )
        try:
            canonical_root = managed_root.resolve(strict=True)
        except OSError as exc:
            raise StateSelectorAmendmentError(
                "managed generated-output root is missing"
            ) from exc
    if (
        not canonical_root.is_dir()
        or canonical_root.name != managed_root.name
        or _has_inaccessible_custody_component(canonical_root)
    ):
        raise StateSelectorAmendmentError(
            "managed generated-output root identity changed"
        )
    _assert_no_generated_path_symlink(canonical_root)

    canonical_artifact = canonical_root.joinpath(*relative.parts)
    # This tests every existing descendant without following it.  Missing
    # leaves remain the loader's ordinary, explicit missing-artifact failure.
    _assert_no_generated_path_symlink(canonical_artifact)
    return canonical_artifact


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
    vx_in_nominal_envelope = (
        PLATFORM_NOMINAL_VX_MIN_MPS
        <= previous[0]
        <= PLATFORM_NOMINAL_VX_MAX_MPS
    )
    vx_is_exact_runtime_endpoint = previous[0] in (
        PLATFORM_EXECUTED_VX_MIN_BINARY32_MPS,
        PLATFORM_EXECUTED_VX_MAX_BINARY32_MPS,
    )
    yaw_in_runtime_envelope = (
        PLATFORM_EXECUTED_YAW_MIN_BINARY32_RADPS
        <= previous[2]
        <= PLATFORM_EXECUTED_YAW_MAX_BINARY32_RADPS
    )
    if not ((vx_in_nominal_envelope or vx_is_exact_runtime_endpoint)
            and yaw_in_runtime_envelope):
        raise StateSelectorAmendmentError(
            "previous_applied_command exceeds the frozen platform envelope"
        )
    # Do not clamp or round: L_max starts from the actual applied command that
    # passed the canonical snapshot boundary.
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


def snapshot_task_status_projection(
    task_status: Mapping[str, Any],
) -> dict[str, bool]:
    """Return the exact four frozen selector flags from a full snapshot status.

    The full production status deliberately carries claim, reset, and
    termination evidence that does not belong in each allocation-rotation
    row.  This projection is the only valid equality boundary between those
    two representations.
    """

    projected = _task_status_projection(task_status)
    if any(type(projected[key]) is not bool for key in _TASK_STATUS_KEYS):
        raise StateSelectorAmendmentError(
            "snapshot task status lacks four exact boolean selector flags"
        )
    return {key: bool(projected[key]) for key in _TASK_STATUS_KEYS}


def validate_snapshot_task_status_binding(
    task_status: Mapping[str, Any],
    projected_status: Mapping[str, Any],
    *,
    designated_goal_cell: int | None = None,
) -> dict[str, bool]:
    """Validate full production evidence and its four-flag selector binding."""

    if not isinstance(task_status, Mapping) or set(task_status) != \
            _FULL_SNAPSHOT_TASK_STATUS_KEYS:
        raise StateSelectorAmendmentError(
            "full snapshot task status has an unexpected key surface"
        )
    if not isinstance(projected_status, Mapping) or set(projected_status) != \
            set(_TASK_STATUS_KEYS):
        raise StateSelectorAmendmentError(
            "projected snapshot task status has an unexpected key surface"
        )
    projected = snapshot_task_status_projection(task_status)
    if projected != dict(projected_status):
        raise StateSelectorAmendmentError(
            "full snapshot task status differs from its selector projection"
        )

    claim = task_status["production_claim_evidence"]
    if not isinstance(claim, Mapping) or set(claim) != \
            _PRODUCTION_CLAIM_EVIDENCE_KEYS:
        raise StateSelectorAmendmentError(
            "snapshot production claim evidence has an unexpected key surface"
        )
    accessor_callable = claim["active_collector_visited_accessor_callable"]
    claimed_cells = claim["active_collector_claimed_cells"]
    designated_goal = claim["designated_goal_cell"]
    if (
        type(accessor_callable) is not bool
        or not isinstance(claimed_cells, list)
        or any(not isinstance(cell, int) or isinstance(cell, bool)
               for cell in claimed_cells)
        or claimed_cells != sorted(set(claimed_cells))
        or not isinstance(designated_goal, int)
        or isinstance(designated_goal, bool)
        or (designated_goal_cell is not None
            and designated_goal != operator.index(designated_goal_cell))
        or task_status["goal_claimed"] is not (
            designated_goal in claimed_cells
        )
    ):
        raise StateSelectorAmendmentError(
            "snapshot production claim evidence is inconsistent"
        )

    reset = task_status["production_task_completion_reset_evidence"]
    if (
        not isinstance(reset, Mapping)
        or set(reset) != _PRODUCTION_RESET_EVIDENCE_KEYS
        or any(type(reset[key]) is not bool
               for key in _PRODUCTION_RESET_EVIDENCE_KEYS)
        or task_status["task_completed"] is not all(
            reset[key] for key in _PRODUCTION_RESET_EVIDENCE_KEYS
        )
    ):
        raise StateSelectorAmendmentError(
            "snapshot production completion-reset evidence is inconsistent"
        )

    termination_flags = task_status["termination_flags"]
    if (
        not isinstance(termination_flags, Mapping)
        or set(termination_flags) != _PRODUCTION_TERMINATION_FLAG_KEYS
        or any(type(termination_flags[key]) is not bool
               for key in _PRODUCTION_TERMINATION_FLAG_KEYS)
        or task_status["terminated"] is not any(termination_flags.values())
    ):
        raise StateSelectorAmendmentError(
            "snapshot termination evidence is inconsistent"
        )
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
    path = _managed_generated_artifact_path(
        root / str(FROZEN_FAILED_CENSUS_RECEIPT["path"]), root=root
    )
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
    path = _managed_generated_artifact_path(
        root / str(binding["path"]), root=root
    )
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
        path = _managed_generated_artifact_path(
            shard_root / f"{task_digest}.json", root=root
        )
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
        try:
            resolved = _managed_generated_artifact_path(
                root / expected_relative, root=root
            )
        except (OSError, StateSelectorAmendmentError) as exc:
            raise StateSelectorAmendmentError(
                "phase-1 state-check shard is missing"
            ) from exc
        if (
            not resolved.is_file()
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


def _load_exact_frozen_json(
    binding: Mapping[str, Any], *, root: Path, label: str,
) -> dict[str, Any]:
    """Open one pre-outcome terminal only through its exact byte binding."""

    path = _managed_generated_artifact_path(
        root / str(binding["path"]), root=root
    )
    if (
        not path.is_file()
        or path.stat().st_size != int(binding["byte_count"])
        or _file_sha256(path) != binding["raw_sha256"]
    ):
        raise StateSelectorAmendmentError(f"{label} raw binding changed")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise StateSelectorAmendmentError(f"{label} JSON is invalid") from exc
    return payload


def validate_frozen_reachability_feasibility_pass(
    *, root: Path = ROOT,
) -> dict[str, Any]:
    """Reopen the exact 7047 feasibility pass under its historical bindings."""

    binding = FROZEN_REACHABILITY_FEASIBILITY_PASS
    receipt = _load_exact_frozen_json(
        binding, root=root, label="frozen reachability feasibility pass"
    )
    if receipt.get("state_selector_feasibility_receipt_digest") != binding[
        "receipt_digest"
    ]:
        raise StateSelectorAmendmentError(
            "frozen reachability feasibility self binding changed"
        )
    validate_state_selector_feasibility_receipt(
        receipt,
        expected_source_commit=str(binding["source_repository_commit"]),
        expected_successor_selection_digest=str(
            binding["successor_selection_digest"]
        ),
        expected_clean_source_binding_digest=str(
            binding["clean_source_binding_digest"]
        ),
        expected_bound_implementations_digest=str(
            binding["bound_implementations_digest"]
        ),
        root=root,
    )
    return receipt


def validate_frozen_preserved_precontract_failure(
    *, root: Path = ROOT,
) -> dict[str, Any]:
    """Validate the exact 45-check terminal failure without promoting it."""

    binding = FROZEN_PRESERVED_PRECONTRACT_FAILURE
    receipt = _load_exact_frozen_json(
        binding, root=root, label="frozen preserved precontract failure"
    )
    payload = dict(receipt)
    observed = payload.pop(
        "preserved_state_precontract_revalidation_receipt_digest", None
    )
    if observed != binding["receipt_digest"] or _sha256(payload) != observed:
        raise StateSelectorAmendmentError(
            "frozen preserved precontract failure self binding changed"
        )
    if (
        payload.get("schema") != PRESERVED_STATE_PRECONTRACT_REVALIDATION_SCHEMA
        or payload.get("status") != "FAIL_PRECONTRACT_IDENTITY_REVALIDATION"
        or payload.get("complete") is not True
        or payload.get("source_repository_commit")
        != binding["source_repository_commit"]
        or payload.get("clean_source_binding_digest")
        != binding["clean_source_binding_digest"]
        or payload.get("bound_implementations_digest")
        != binding["bound_implementations_digest"]
        or payload.get("successor_selection_digest")
        != binding["successor_selection_digest"]
        or payload.get("state_selector_feasibility_receipt_digest")
        != FROZEN_REACHABILITY_FEASIBILITY_PASS["receipt_digest"]
        or payload.get("state_identity_set_digest")
        != binding["state_identity_set_digest"]
        or payload.get("outcome_surface_absence_attestation_digest")
        != binding["outcome_surface_absence_attestation_digest"]
        or payload.get("preserved_state_count") != 45
        or payload.get("failure_count") != 8
    ):
        raise StateSelectorAmendmentError(
            "frozen preserved precontract failure lineage changed"
        )
    validate_phase1_outcome_surface_absence_attestation(
        payload.get("outcome_surface_absence_attestation")
    )
    if any(
        payload.get(key) not in (False, 0)
        for key in (
            "candidate_allocation_loaded",
            "candidate_outcomes_loaded",
            "branch_identities_created",
            "branches_attempted",
            "frames_rendered",
            "target_latents_encoded",
            "scorer_training_started",
        )
    ):
        raise StateSelectorAmendmentError(
            "frozen preserved precontract failure contains scientific output"
        )
    _validate_phase1_state_check_transport(receipt, root=root)
    checks = [
        dict(check)
        for shard in payload.get("shards", [])
        for check in shard.get("state_checks", [])
    ]
    check_booleans = (
        "exact_redrive_pass", "exclusion_checks_pass",
        "amended_classification_pass", "goal_binding_unchanged",
        "oracle_completion_target_unchanged",
        "production_task_completion_reset_unchanged",
        "snapshot_production_designated_goal_claim_unchanged",
        "completion_state_task_status_all_false",
    )
    passing = [
        check for check in checks
        if all(check.get(key) is True for key in check_booleans)
        and check.get("failure_reason") is None
    ]
    accepted_failure_reason = (
        "RuntimeError:amended classification failed: no_completion_enriched_goal"
    )
    failed = [
        check for check in checks
        if check.get("exclusion_checks_pass") is True
        and all(check.get(key) is False for key in check_booleans
                if key != "exclusion_checks_pass")
        and check.get("failure_reason") == accepted_failure_reason
    ]
    if (
        len(checks) != 45
        or len({check.get("state_identity_digest") for check in checks}) != 45
        or len(passing) != 37
        or len(failed) != 8
        or payload.get("failures") != failed
        or len(passing) + len(failed) != len(checks)
    ):
        raise StateSelectorAmendmentError(
            "frozen preserved precontract failure set changed"
        )
    source_states = {
        str(state["state_identity_digest"]): state
        for shard in load_preserved_state_shards(root).values()
        for state in shard["states"]
    }
    if set(source_states) != {
        str(check["state_identity_digest"]) for check in checks
    }:
        raise StateSelectorAmendmentError(
            "frozen preserved precontract failure identity coverage changed"
        )
    if any(
        source_states[str(check["state_identity_digest"])]["stratum"]
        != "completion_enriched"
        for check in failed
    ):
        raise StateSelectorAmendmentError(
            "frozen precontract rejection is not completion-only"
        )
    return receipt


def _mixed_identity_row(
    state: Mapping[str, Any], *, failure_reason: str | None = None,
) -> dict[str, Any]:
    row = {
        "state_identity_digest": str(state["state_identity_digest"]),
        "state_id": str(state["state_id"]),
        "scene_id": str(state["scene_id"]),
        "family": str(state["family"]),
        "stratum": str(state["stratum"]),
        "split_role": str(state["split_role"]),
    }
    if failure_reason is not None:
        row["failure_reason"] = str(failure_reason)
    return row


def mixed_precontract_disposition_sets(
    *, root: Path = ROOT,
) -> dict[str, list[dict[str, Any]]]:
    """Derive the exact 37/8 disposition solely from the frozen failed checks."""

    failed_receipt = validate_frozen_preserved_precontract_failure(root=root)
    checks = {
        str(check["state_identity_digest"]): check
        for shard in failed_receipt["shards"]
        for check in shard["state_checks"]
    }
    source_states = {
        str(state["state_identity_digest"]): state
        for shard in load_preserved_state_shards(root).values()
        for state in shard["states"]
    }
    retained: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    slots: list[dict[str, Any]] = []
    for identity in sorted(source_states):
        state = source_states[identity]
        check = checks[identity]
        pass_fields = (
            "exact_redrive_pass", "exclusion_checks_pass",
            "amended_classification_pass", "goal_binding_unchanged",
            "oracle_completion_target_unchanged",
            "production_task_completion_reset_unchanged",
            "snapshot_production_designated_goal_claim_unchanged",
            "completion_state_task_status_all_false",
        )
        full_pass = (
            all(check.get(key) is True for key in pass_fields)
            and check.get("failure_reason") is None
        )
        accepted_failure = (
            check.get("exclusion_checks_pass") is True
            and all(check.get(key) is False for key in pass_fields
                    if key != "exclusion_checks_pass")
            and check.get("failure_reason") == (
                "RuntimeError:amended classification failed: "
                "no_completion_enriched_goal"
            )
        )
        if full_pass:
            retained.append(_mixed_identity_row(state))
            continue
        if not accepted_failure:
            raise StateSelectorAmendmentError(
                "mixed disposition encountered an unregistered phase-1 failure shape"
            )
        reason = str(check.get("failure_reason"))
        rejected.append(_mixed_identity_row(state, failure_reason=reason))
        slots.append({
            "state_id": str(state["state_id"]),
            "family": str(state["family"]),
            "stratum": str(state["stratum"]),
            "split_role": str(state["split_role"]),
            "predecessor_state_identity_digest": identity,
            "predecessor_scene_id": str(state["scene_id"]),
        })
    if len(retained) != 37 or len(rejected) != 8 or len(slots) != 8:
        raise StateSelectorAmendmentError(
            "mixed precontract disposition is not exactly 37 retained and 8 rejected"
        )
    retained.sort(key=lambda row: row["state_id"])
    rejected.sort(key=lambda row: row["state_id"])
    slots.sort(key=lambda row: row["state_id"])
    return {
        "retained_predecessor_identities": retained,
        "rejected_predecessor_identities": rejected,
        "replacement_slots": slots,
    }


def build_preserved_state_mixed_precontract_disposition_receipt(
    *,
    source_repository_commit: str,
    clean_source_binding_digest: str,
    bound_implementations_digest: str,
    successor_selection_digest: str,
    outcome_surface_absence_attestation: Mapping[str, Any],
    root: Path = ROOT,
) -> dict[str, Any]:
    """Build the active pre-outcome 37-retained/8-replacement authority."""

    for label, value in (
        ("clean-source", clean_source_binding_digest),
        ("implementation", bound_implementations_digest),
        ("selection", successor_selection_digest),
    ):
        if not _is_digest(value):
            raise StateSelectorAmendmentError(f"mixed disposition {label} digest invalid")
    if (
        not isinstance(source_repository_commit, str)
        or len(source_repository_commit) != 40
        or any(character not in _HEX for character in source_repository_commit)
    ):
        raise StateSelectorAmendmentError("mixed disposition source commit invalid")
    validate_frozen_reachability_feasibility_pass(root=root)
    validate_frozen_preserved_precontract_failure(root=root)
    validate_phase1_outcome_surface_absence_attestation(
        outcome_surface_absence_attestation
    )
    sets = mixed_precontract_disposition_sets(root=root)
    retained = sets["retained_predecessor_identities"]
    rejected = sets["rejected_predecessor_identities"]
    slots = sets["replacement_slots"]
    payload: dict[str, Any] = {
        "schema": PRESERVED_STATE_MIXED_PRECONTRACT_DISPOSITION_SCHEMA,
        "status": MIXED_PRECONTRACT_DISPOSITION_STATUS,
        "complete": True,
        "binding_receipt": True,
        "source_repository_commit": source_repository_commit,
        "clean_source_binding_digest": clean_source_binding_digest,
        "bound_implementations_digest": bound_implementations_digest,
        "successor_selection_digest": successor_selection_digest,
        "state_selector_amendment_digest": state_selector_amendment_digest(),
        "frozen_reachability_feasibility_pass": dict(
            FROZEN_REACHABILITY_FEASIBILITY_PASS
        ),
        "frozen_preserved_precontract_failure": dict(
            FROZEN_PRESERVED_PRECONTRACT_FAILURE
        ),
        "outcome_surface_absence_attestation": dict(
            outcome_surface_absence_attestation
        ),
        "outcome_surface_absence_attestation_digest":
            outcome_surface_absence_attestation["attestation_digest"],
        "original_predecessor_state_count": 45,
        "retained_predecessor_state_count": 37,
        "rejected_predecessor_state_count": 8,
        "replacement_slot_count": 8,
        "retained_predecessor_identities": retained,
        "retained_predecessor_identity_set_digest": _sha256(retained),
        "rejected_predecessor_identities": rejected,
        "rejected_predecessor_identity_set_digest": _sha256(rejected),
        "replacement_slots": slots,
        "replacement_slot_set_digest": _sha256(slots),
        "candidate_allocation_loaded": False,
        "candidate_outcomes_loaded": False,
        "branch_identities_created": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
        "predictor_checkpoints_opened": 0,
    }
    payload["mixed_precontract_disposition_receipt_digest"] = _sha256(payload)
    return payload


def validate_preserved_state_mixed_precontract_disposition_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_source_commit: str | None = None,
    expected_successor_selection_digest: str | None = None,
    expected_clean_source_binding_digest: str | None = None,
    expected_bound_implementations_digest: str | None = None,
    root: Path = ROOT,
) -> None:
    """Fail closed on the active mixed disposition and both frozen terminals."""

    if not isinstance(receipt, Mapping):
        raise StateSelectorAmendmentError("mixed precontract disposition must be a mapping")
    payload = dict(receipt)
    observed = payload.pop("mixed_precontract_disposition_receipt_digest", None)
    if not _is_digest(observed) or _sha256(payload) != observed:
        raise StateSelectorAmendmentError("mixed precontract disposition self digest failed")
    expected_keys = {
        "schema", "status", "complete", "binding_receipt",
        "source_repository_commit", "clean_source_binding_digest",
        "bound_implementations_digest", "successor_selection_digest",
        "state_selector_amendment_digest",
        "frozen_reachability_feasibility_pass",
        "frozen_preserved_precontract_failure",
        "outcome_surface_absence_attestation",
        "outcome_surface_absence_attestation_digest",
        "original_predecessor_state_count",
        "retained_predecessor_state_count",
        "rejected_predecessor_state_count", "replacement_slot_count",
        "retained_predecessor_identities",
        "retained_predecessor_identity_set_digest",
        "rejected_predecessor_identities",
        "rejected_predecessor_identity_set_digest", "replacement_slots",
        "replacement_slot_set_digest", "candidate_allocation_loaded",
        "candidate_outcomes_loaded", "branch_identities_created",
        "branches_attempted", "frames_rendered", "target_latents_encoded",
        "scorer_training_started", "predictor_checkpoints_opened",
    }
    if set(payload) != expected_keys:
        raise StateSelectorAmendmentError(
            "mixed precontract disposition has an unexpected key surface"
        )
    if (
        payload.get("schema") != PRESERVED_STATE_MIXED_PRECONTRACT_DISPOSITION_SCHEMA
        or payload.get("status") != MIXED_PRECONTRACT_DISPOSITION_STATUS
        or payload.get("complete") is not True
        or payload.get("binding_receipt") is not True
        or payload.get("state_selector_amendment_digest")
        != state_selector_amendment_digest()
        or payload.get("frozen_reachability_feasibility_pass")
        != FROZEN_REACHABILITY_FEASIBILITY_PASS
        or payload.get("frozen_preserved_precontract_failure")
        != FROZEN_PRESERVED_PRECONTRACT_FAILURE
    ):
        raise StateSelectorAmendmentError("mixed precontract disposition is not complete/pass")
    for value, expected, label in (
        (payload.get("source_repository_commit"), expected_source_commit, "source"),
        (payload.get("successor_selection_digest"),
         expected_successor_selection_digest, "selection"),
        (payload.get("clean_source_binding_digest"),
         expected_clean_source_binding_digest, "clean-source"),
        (payload.get("bound_implementations_digest"),
         expected_bound_implementations_digest, "implementation"),
    ):
        if expected is not None and value != expected:
            raise StateSelectorAmendmentError(
                f"mixed precontract disposition {label} binding mismatch"
            )
    validate_frozen_reachability_feasibility_pass(root=root)
    validate_frozen_preserved_precontract_failure(root=root)
    absence = payload.get("outcome_surface_absence_attestation")
    validate_phase1_outcome_surface_absence_attestation(absence)
    if payload.get("outcome_surface_absence_attestation_digest") != absence.get(
        "attestation_digest"
    ):
        raise StateSelectorAmendmentError("mixed disposition absence binding changed")
    expected_sets = mixed_precontract_disposition_sets(root=root)
    for key, count, digest_key in (
        ("retained_predecessor_identities", 37,
         "retained_predecessor_identity_set_digest"),
        ("rejected_predecessor_identities", 8,
         "rejected_predecessor_identity_set_digest"),
        ("replacement_slots", 8, "replacement_slot_set_digest"),
    ):
        rows = payload.get(key)
        if (
            rows != expected_sets[key]
            or not isinstance(rows, list)
            or len(rows) != count
            or payload.get(digest_key) != _sha256(rows)
        ):
            raise StateSelectorAmendmentError(
                f"mixed precontract disposition {key} changed"
            )
    if (
        payload.get("original_predecessor_state_count") != 45
        or payload.get("retained_predecessor_state_count") != 37
        or payload.get("rejected_predecessor_state_count") != 8
        or payload.get("replacement_slot_count") != 8
        or any(
            payload.get(key) not in (False, 0)
            for key in (
                "candidate_allocation_loaded", "candidate_outcomes_loaded",
                "branch_identities_created", "branches_attempted",
                "frames_rendered", "target_latents_encoded",
                "scorer_training_started", "predictor_checkpoints_opened",
            )
        )
    ):
        raise StateSelectorAmendmentError(
            "mixed precontract disposition counts or no-outcome gate changed"
        )


def load_and_validate_preserved_state_mixed_precontract_disposition_receipt(
    *,
    expected_source_commit: str | None = None,
    expected_successor_selection_digest: str | None = None,
    expected_clean_source_binding_digest: str | None = None,
    expected_bound_implementations_digest: str | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Guard, open, and validate the one active mixed disposition receipt."""

    path = _managed_generated_artifact_path(
        root / PRESERVED_STATE_MIXED_PRECONTRACT_DISPOSITION_RECEIPT_PATH,
        root=root,
    )
    if not path.is_file():
        raise StateSelectorAmendmentError(
            "active mixed precontract disposition receipt is missing"
        )
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise StateSelectorAmendmentError(
            "active mixed precontract disposition receipt JSON is invalid"
        ) from exc
    validate_preserved_state_mixed_precontract_disposition_receipt(
        receipt,
        expected_source_commit=expected_source_commit,
        expected_successor_selection_digest=
            expected_successor_selection_digest,
        expected_clean_source_binding_digest=
            expected_clean_source_binding_digest,
        expected_bound_implementations_digest=
            expected_bound_implementations_digest,
        root=root,
    )
    return receipt



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


def _identity_payload_projection(state: Mapping[str, Any]) -> dict[str, Any]:
    """Match the builder's state-identity projection without importing it."""

    return {
        key: value for key, value in state.items()
        if key not in {
            "state_identity_digest", "state_index", "candidate_indices",
            "candidate_rotation_index", "branch_identities",
        }
    }


def _snapshot_core_projection(state: Mapping[str, Any]) -> dict[str, Any]:
    """Project contract-independent fields that identify one resolved snapshot."""

    return {
        key: state.get(key)
        for key in (
            "scene_id", "episode_cluster_id", "episode_id", "source_step",
            "cell_id", "boundary", "warmup_blocks",
        )
    }


def _phase1_completion_vectors(
    *, root: Path,
) -> dict[str, dict[str, Any]]:
    """Recover the seven byte-bound vectors for retained completion states."""

    failed = validate_frozen_preserved_precontract_failure(root=root)
    vectors: dict[str, dict[str, Any]] = {}
    for shard in failed["shards"]:
        for check in shard["state_checks"]:
            vector = check.get("completion_rotation_eligibility")
            if vector is None:
                continue
            identity = str(check["state_identity_digest"])
            if identity in vectors or not isinstance(vector, Mapping):
                raise StateSelectorAmendmentError(
                    "frozen phase-1 completion-vector coverage changed"
                )
            vectors[identity] = dict(vector)
    retained_completion = {
        str(row["state_identity_digest"])
        for row in mixed_precontract_disposition_sets(root=root)[
            "retained_predecessor_identities"
        ]
        if row["stratum"] == "completion_enriched"
    }
    if len(retained_completion) != 7 or set(vectors) != retained_completion:
        raise StateSelectorAmendmentError(
            "frozen phase-1 does not bind exactly the seven retained completion vectors"
        )
    return vectors


def _completion_source_row_from_active_state(
    state: Mapping[str, Any],
    *,
    assignment: Mapping[str, Any],
    preserved_vectors: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive one assigned-mask row from its identity-owned 12-rotation vector."""

    identity = str(state["state_identity_digest"])
    vector = state.get("completion_rotation_eligibility_vector")
    if vector is None:
        vector = preserved_vectors.get(identity)
    if not isinstance(vector, Mapping):
        raise StateSelectorAmendmentError(
            "completion identity lacks identity-owned rotation evidence"
        )
    rotations = vector.get("rotations")
    if not isinstance(rotations, list) or len(rotations) != ALLOCATION.CANDIDATE_COUNT:
        raise StateSelectorAmendmentError(
            "completion identity rotation evidence is malformed"
        )
    first = rotations[0]
    if not isinstance(first, Mapping):
        raise StateSelectorAmendmentError(
            "completion identity rotation evidence is malformed"
        )
    try:
        recomputed_vector = completion_rotation_eligibility_vector(
            graph_hops=int(first["graph_hops_diagnostic"]),
            reachable=bool(first["reachable"]),
            continuous_geodesic_m=float(first["continuous_geodesic_m"]),
            bearing_body_rad=float(first["bearing_body_rad"]),
            task_status=first["task_status"],
            previous_applied_command=first["previous_applied_command"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StateSelectorAmendmentError(
            "completion identity rotation evidence cannot be reconstructed"
        ) from exc
    if dict(vector) != recomputed_vector:
        raise StateSelectorAmendmentError(
            "completion identity rotation evidence changed"
        )

    goal = state.get("goal")
    if not isinstance(goal, Mapping):
        raise StateSelectorAmendmentError(
            "completion identity lacks its snapshot goal binding"
        )
    try:
        goal_matches = (
            int(goal.get("graph_edges")) == int(first["graph_hops_diagnostic"])
            and float(goal.get("start_geodesic_m"))
            == float(first["continuous_geodesic_m"])
            and float(goal.get("bearing_body_rad"))
            == float(first["bearing_body_rad"])
        )
    except (TypeError, ValueError) as exc:
        raise StateSelectorAmendmentError(
            "completion identity goal binding is malformed"
        ) from exc
    if not goal_matches:
        raise StateSelectorAmendmentError(
            "completion rotation evidence differs from its active goal binding"
        )
    if "snapshot_task_status" in state:
        try:
            validate_snapshot_task_status_binding(
                state["snapshot_task_status"], first["task_status"],
                designated_goal_cell=operator.index(goal["landmark_cell"]),
            )
        except (KeyError, TypeError, ValueError,
                StateSelectorAmendmentError) as exc:
            raise StateSelectorAmendmentError(
                "completion rotation evidence differs from active task status"
            ) from exc
    if "previous_applied_command" in state and (
        _normalise_previous_applied(state["previous_applied_command"])
        != _normalise_previous_applied(first["previous_applied_command"])
    ):
        raise StateSelectorAmendmentError(
            "completion rotation evidence differs from active previous command"
        )

    candidate_indices = [int(value) for value in assignment["candidate_indices"]]
    rotation = candidate_rotation_index(candidate_indices)
    if (
        ("rotation_index" in assignment
         and int(assignment["rotation_index"]) != rotation)
        or ("candidate_rotation_index" in assignment
            and int(assignment["candidate_rotation_index"]) != rotation)
    ):
        raise StateSelectorAmendmentError(
            "completion allocation rotation index differs from its exact mask"
        )
    selected = rotations[rotation]
    if (
        not isinstance(selected, Mapping)
        or list(selected.get("candidate_indices", [])) != candidate_indices
    ):
        raise StateSelectorAmendmentError(
            "completion identity vector does not own its assigned mask"
        )
    return {
        "state_identity_digest": identity,
        "state_id": str(state["state_id"]),
        "family": str(state["family"]),
        "stratum": "completion_enriched",
        "candidate_indices": candidate_indices,
        "previous_applied_command": list(selected["previous_applied_command"]),
        "completion_eligibility": dict(selected),
    }


_ALLOCATION_MANIFEST_KEYS = frozenset({
    "schema", "status", "source_identity_manifest_digest",
    "pre_outcome_identity_digest", "allocation_contract",
    "allocation_contract_digest", "allocation_amendment",
    "allocation_amendment_digest", "assignments", "contingency_tables",
    "post_identity_pre_outcome_validation", "allocation_manifest_digest",
})
_ALLOCATION_IDENTITY_KEYS = frozenset({
    "state_id", "state_identity_digest", "family", "stratum", "split_role",
    "goal_type",
})
_ALLOCATION_ASSIGNMENT_KEYS = _ALLOCATION_IDENTITY_KEYS | frozenset({
    "rotation_index", "candidate_indices",
})


def validate_allocation_manifest_structure_solve_free(
    allocation_manifest: Mapping[str, Any],
    *,
    expected_source_identity_manifest_digest: str | None = None,
) -> None:
    """Validate every allocation property except the MILP canonicality proof.

    This is deliberately *not* a replacement for certification of the
    lexicographically minimal rotation vector.  It validates the complete
    manifest key surface, identity projection, exact rotation masks, all
    balance tables, the post-identity validation object, and both manifest
    digests using pure deterministic helpers.  It never calls the allocator's
    builder, public validator, constraint builder, or MILP solver.  A caller
    using this function for scientific acceptance must separately validate a
    frozen solve-free canonicality certificate.
    """

    try:
        if not isinstance(allocation_manifest, Mapping):
            raise StateSelectorAmendmentError(
                "certified allocation manifest must be a mapping"
            )
        if set(allocation_manifest) != _ALLOCATION_MANIFEST_KEYS:
            raise StateSelectorAmendmentError(
                "certified allocation manifest has an unexpected key surface"
            )
        if (
            allocation_manifest.get("schema") != ALLOCATION.SCHEMA
            or allocation_manifest.get("status") != ALLOCATION.STATUS
        ):
            raise StateSelectorAmendmentError(
                "certified allocation manifest schema/status changed"
            )
        source_digest = allocation_manifest.get(
            "source_identity_manifest_digest"
        )
        if not _is_digest(source_digest):
            raise StateSelectorAmendmentError(
                "certified allocation source identity digest is invalid"
            )
        if (
            expected_source_identity_manifest_digest is not None
            and source_digest != expected_source_identity_manifest_digest
        ):
            raise StateSelectorAmendmentError(
                "certified allocation source identity digest changed"
            )
        if (
            allocation_manifest.get("allocation_contract")
            != ALLOCATION.algorithm_contract()
            or allocation_manifest.get("allocation_contract_digest")
            != ALLOCATION.allocation_contract_digest()
            or allocation_manifest.get("allocation_amendment")
            != ALLOCATION.allocation_amendment_contract()
            or allocation_manifest.get("allocation_amendment_digest")
            != ALLOCATION.allocation_amendment_digest()
        ):
            raise StateSelectorAmendmentError(
                "certified allocation contract lineage changed"
            )

        raw_assignments = allocation_manifest.get("assignments")
        if not isinstance(raw_assignments, list) or len(raw_assignments) != 120:
            raise StateSelectorAmendmentError(
                "certified allocation must contain exactly 120 assignments"
            )
        identity_rows: list[dict[str, str]] = []
        previous_key: tuple[str, str] | None = None
        for raw in raw_assignments:
            if (
                not isinstance(raw, Mapping)
                or frozenset(raw) != _ALLOCATION_ASSIGNMENT_KEYS
            ):
                raise StateSelectorAmendmentError(
                    "certified allocation assignment key surface changed"
                )
            identity = ALLOCATION._normalise_identity_state({
                key: raw[key] for key in _ALLOCATION_IDENTITY_KEYS
            })
            key = (identity["state_identity_digest"], identity["state_id"])
            if previous_key is not None and key <= previous_key:
                raise StateSelectorAmendmentError(
                    "certified allocation assignments are not canonical-order"
                )
            previous_key = key
            identity_rows.append(identity)
            rotation = raw.get("rotation_index")
            if isinstance(rotation, bool) or not isinstance(rotation, int):
                raise StateSelectorAmendmentError(
                    "certified allocation rotation index is not an integer"
                )
            candidates = raw.get("candidate_indices")
            if (
                not isinstance(candidates, list)
                or candidates != list(ALLOCATION.candidate_block(rotation))
            ):
                raise StateSelectorAmendmentError(
                    "certified allocation mask differs from its exact rotation"
                )

        normalised = ALLOCATION._normalise_identity_states(identity_rows)
        if allocation_manifest.get("pre_outcome_identity_digest") != (
            ALLOCATION.pre_outcome_identity_digest(normalised)
        ):
            raise StateSelectorAmendmentError(
                "certified allocation identity projection digest changed"
            )
        expected_tables = ALLOCATION._contingency_tables(raw_assignments)
        if allocation_manifest.get("contingency_tables") != expected_tables:
            raise StateSelectorAmendmentError(
                "certified allocation contingency tables changed"
            )
        # This helper materialises every count, coverage, goal-type, and exact
        # post-identity check.  It is arithmetic over the supplied rows only.
        expected_post_identity = ALLOCATION._post_identity_pre_outcome_validation(
            raw_assignments
        )
        if allocation_manifest.get(
            "post_identity_pre_outcome_validation"
        ) != expected_post_identity:
            raise StateSelectorAmendmentError(
                "certified allocation post-identity validation changed"
            )
        if allocation_manifest.get("allocation_manifest_digest") != (
            ALLOCATION.allocation_manifest_digest(allocation_manifest)
        ):
            raise StateSelectorAmendmentError(
                "certified allocation manifest digest changed"
            )
    except StateSelectorAmendmentError:
        raise
    except (KeyError, TypeError, ValueError,
            ALLOCATION.CandidateAllocationError) as exc:
        raise StateSelectorAmendmentError(
            "certified allocation failed solve-free structural validation"
        ) from exc


def validate_solve_free_certified_allocation_manifest(
    allocation_manifest: Mapping[str, Any],
    *,
    certify_allocation_solve_free: Callable[
        [Mapping[str, Any]], Mapping[str, Any]
    ],
    expected_source_identity_manifest_digest: str | None = None,
) -> None:
    """Require structural validity plus an exact external certificate replay.

    ``certify_allocation_solve_free`` is the fail-closed trust boundary.  It
    must replay the frozen parallel-search certificates without solving and
    return the exact certified allocation manifest.  Returning ``True``, a
    digest, a partial projection, or a merely equivalent manifest is rejected.
    The callback receives a deep copy so it cannot mutate the accepted input.
    """

    if not callable(certify_allocation_solve_free):
        raise StateSelectorAmendmentError(
            "solve-free allocation certification callback is missing"
        )
    validate_allocation_manifest_structure_solve_free(
        allocation_manifest,
        expected_source_identity_manifest_digest=(
            expected_source_identity_manifest_digest
        ),
    )
    frozen_manifest = copy.deepcopy(dict(allocation_manifest))
    try:
        certified = certify_allocation_solve_free(
            copy.deepcopy(frozen_manifest)
        )
    except Exception as exc:
        raise StateSelectorAmendmentError(
            "solve-free allocation certificate replay failed"
        ) from exc
    if not isinstance(certified, Mapping):
        raise StateSelectorAmendmentError(
            "solve-free allocation certificate did not return a manifest"
        )
    if dict(allocation_manifest) != frozen_manifest:
        raise StateSelectorAmendmentError(
            "solve-free allocation certification mutated its input"
        )
    certified_manifest = copy.deepcopy(dict(certified))
    if certified_manifest != frozen_manifest:
        raise StateSelectorAmendmentError(
            "solve-free allocation certificate binds a different manifest"
        )
    # Validate the returned object independently instead of trusting equality
    # implemented by an exotic Mapping subtype.
    validate_allocation_manifest_structure_solve_free(
        certified_manifest,
        expected_source_identity_manifest_digest=(
            expected_source_identity_manifest_digest
        ),
    )


def _build_preserved_state_revalidation_receipt_after_allocation_validation(
    *,
    allocation_manifest: Mapping[str, Any],
    active_states: Sequence[Mapping[str, Any]],
    completion_states: Sequence[Mapping[str, Any]],
    source_repository_commit: str,
    successor_selection_digest: str,
    state_selector_feasibility_receipt_digest: str,
    mixed_precontract_disposition_receipt_digest: str,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Bind 37 retained masks, eight replacements, and all 40 completion rows."""

    for label, value in (
        ("selection", successor_selection_digest),
        ("feasibility", state_selector_feasibility_receipt_digest),
        ("mixed disposition", mixed_precontract_disposition_receipt_digest),
    ):
        if not _is_digest(value):
            raise StateSelectorAmendmentError(f"phase-2 {label} digest invalid")
    mixed_path = _managed_generated_artifact_path(
        root / PRESERVED_STATE_MIXED_PRECONTRACT_DISPOSITION_RECEIPT_PATH,
        root=root,
    )
    if not mixed_path.is_file():
        raise StateSelectorAmendmentError("active mixed disposition receipt is missing")
    try:
        mixed = json.loads(mixed_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise StateSelectorAmendmentError(
            "active mixed disposition receipt is invalid JSON"
        ) from exc
    validate_preserved_state_mixed_precontract_disposition_receipt(
        mixed,
        expected_source_commit=source_repository_commit,
        expected_successor_selection_digest=successor_selection_digest,
        root=root,
    )
    if mixed.get("mixed_precontract_disposition_receipt_digest") != (
        mixed_precontract_disposition_receipt_digest
    ):
        raise StateSelectorAmendmentError("phase-2 mixed disposition digest changed")
    if state_selector_feasibility_receipt_digest != (
        FROZEN_REACHABILITY_FEASIBILITY_PASS["receipt_digest"]
    ):
        raise StateSelectorAmendmentError("phase-2 feasibility lineage changed")

    assignments = {
        str(row["state_identity_digest"]): row
        for row in allocation_manifest["assignments"]
    }
    states_by_identity: dict[str, Mapping[str, Any]] = {}
    states_by_id: dict[str, Mapping[str, Any]] = {}
    active_scenes: set[str] = set()
    active_episode_clusters: set[str] = set()
    active_observation_boundaries: set[str] = set()
    for state in active_states:
        identity = str(state.get("state_identity_digest", ""))
        state_id = str(state.get("state_id", ""))
        scene_id = str(state.get("scene_id", ""))
        episode_cluster_id = str(state.get("episode_cluster_id", ""))
        boundary = state.get("boundary")
        observation_boundary = _sha256({
            "scene_id": scene_id,
            "episode_cluster_id": episode_cluster_id,
            "episode_id": state.get("episode_id"),
            "source_step": state.get("source_step"),
            "boundary": boundary,
        })
        if (
            not identity or not state_id or not scene_id or not episode_cluster_id
            or not isinstance(boundary, Mapping)
            or identity in states_by_identity or state_id in states_by_id
            or scene_id in active_scenes
            or episode_cluster_id in active_episode_clusters
            or observation_boundary in active_observation_boundaries
        ):
            raise StateSelectorAmendmentError(
                "phase-2 active scene/episode/state/observation identities repeat"
            )
        assignment = assignments.get(identity)
        if assignment is None or any(
            assignment.get(key) != state.get(key)
            for key in (
                "state_id", "state_identity_digest", "family", "stratum",
                "split_role", "goal_type",
            )
        ):
            raise StateSelectorAmendmentError(
                "phase-2 active identity projection differs from allocation"
            )
        if (
            "candidate_indices" in state
            and state.get("candidate_indices")
            != assignment.get("candidate_indices")
        ) or (
            "candidate_rotation_index" in state
            and state.get("candidate_rotation_index")
            != assignment.get("rotation_index")
        ) or (
            "rotation_index" in state
            and state.get("rotation_index") != assignment.get("rotation_index")
        ):
            raise StateSelectorAmendmentError(
                "phase-2 active exact candidate mask differs from allocation"
            )
        states_by_identity[identity] = state
        states_by_id[state_id] = state
        active_scenes.add(scene_id)
        active_episode_clusters.add(episode_cluster_id)
        active_observation_boundaries.add(observation_boundary)
    if (
        len(states_by_identity) != 120
        or set(states_by_identity) != set(assignments)
    ):
        raise StateSelectorAmendmentError(
            "phase-2 active identities differ from the 120-state allocation"
        )

    retained_masks: list[dict[str, Any]] = []
    retained_by_family: dict[str, list[dict[str, Any]]] = {}
    predecessor_states = {
        str(state["state_identity_digest"]): state
        for shard in load_preserved_state_shards(root).values()
        for state in shard["states"]
    }
    for retained in mixed["retained_predecessor_identities"]:
        identity = str(retained["state_identity_digest"])
        state = states_by_identity.get(identity)
        assignment = assignments.get(identity)
        predecessor_state = predecessor_states.get(identity)
        if (
            state is None
            or assignment is None
            or predecessor_state is None
            or _mixed_identity_row(state) != retained
            or _identity_payload_projection(state)
            != _identity_payload_projection(predecessor_state)
            or assignment.get("state_id") != retained["state_id"]
            or assignment.get("family") != retained["family"]
        ):
            raise StateSelectorAmendmentError(
                "phase-2 retained predecessor identity is absent or changed"
            )
        candidate_indices = [int(value) for value in assignment["candidate_indices"]]
        row = {
            "state_identity_digest": identity,
            "state_id": str(retained["state_id"]),
            "family": str(retained["family"]),
            "candidate_indices": candidate_indices,
            "candidate_mask_digest": candidate_mask_digest(
                identity, candidate_indices
            ),
        }
        retained_masks.append(row)
        retained_by_family.setdefault(str(retained["family"]), []).append(row)
    retained_masks.sort(key=lambda row: row["state_identity_digest"])
    if len(retained_masks) != 37:
        raise StateSelectorAmendmentError("phase-2 must retain exactly 37 predecessors")

    rejected = {
        str(row["state_identity_digest"])
        for row in mixed["rejected_predecessor_identities"]
    }
    if rejected.intersection(states_by_identity):
        raise StateSelectorAmendmentError(
            "phase-2 active identities retain a rejected predecessor"
        )
    rejected_payloads = [
        _identity_payload_projection(predecessor_states[identity])
        for identity in sorted(rejected)
    ]
    rejected_snapshot_cores = [
        _snapshot_core_projection(predecessor_states[identity])
        for identity in sorted(rejected)
    ]
    if any(
        _identity_payload_projection(state) in rejected_payloads
        for state in active_states
    ):
        raise StateSelectorAmendmentError(
            "phase-2 re-signed an exact rejected predecessor payload"
        )
    retained_scene_ids = {
        str(row["scene_id"])
        for row in mixed["retained_predecessor_identities"]
    }
    if len(retained_scene_ids) != 37:
        raise StateSelectorAmendmentError(
            "phase-2 retained predecessor scene set changed"
        )
    rejected_scene_ids = {
        str(row["scene_id"])
        for row in mixed["rejected_predecessor_identities"]
    }
    replacement_state_ids = {
        str(row["state_id"]) for row in mixed["replacement_slots"]
    }
    if any(
        str(state["scene_id"]) in rejected_scene_ids
        and str(state["state_id"]) not in replacement_state_ids
        for state in active_states
    ):
        raise StateSelectorAmendmentError(
            "phase-2 reused a rejected scene outside its replacement slots"
        )
    replacements: list[dict[str, Any]] = []
    for slot in mixed["replacement_slots"]:
        state = states_by_id.get(str(slot["state_id"]))
        if (
            state is None
            or state.get("family") != slot["family"]
            or state.get("stratum") != slot["stratum"]
            or state.get("split_role") != slot["split_role"]
            or state.get("state_identity_digest")
            == slot["predecessor_state_identity_digest"]
        ):
            raise StateSelectorAmendmentError(
                "phase-2 replacement does not fill its exact rejected slot"
            )
        if _snapshot_core_projection(state) in rejected_snapshot_cores:
            raise StateSelectorAmendmentError(
                "phase-2 replacement reuses an exact rejected predecessor snapshot"
            )
        identity = str(state["state_identity_digest"])
        assignment = assignments.get(identity)
        if assignment is None or assignment.get("state_id") != slot["state_id"]:
            raise StateSelectorAmendmentError(
                "phase-2 replacement is absent from candidate allocation"
            )
        replacements.append({
            "state_identity_digest": identity,
            "state_id": str(state["state_id"]),
            "scene_id": str(state["scene_id"]),
            "family": str(state["family"]),
            "stratum": str(state["stratum"]),
            "split_role": str(state["split_role"]),
            "replaces_predecessor_state_identity_digest": str(
                slot["predecessor_state_identity_digest"]
            ),
            "replaces_predecessor_scene_id": str(slot["predecessor_scene_id"]),
        })
    replacements.sort(key=lambda row: row["state_id"])
    if len(replacements) != 8 or len({row["scene_id"] for row in replacements}) != 8:
        raise StateSelectorAmendmentError("phase-2 must bind eight distinct replacements")

    preserved_vectors = _phase1_completion_vectors(root=root)
    expected_source_rows = [
        _completion_source_row_from_active_state(
            state,
            assignment=assignments[str(state["state_identity_digest"])],
            preserved_vectors=preserved_vectors,
        )
        for state in active_states
        if state.get("stratum") == "completion_enriched"
    ]
    expected_source_rows.sort(key=lambda row: row["state_identity_digest"])
    supplied_source_rows = [dict(row) for row in completion_states]
    supplied_source_rows.sort(key=lambda row: str(row.get("state_identity_digest", "")))
    if supplied_source_rows != expected_source_rows:
        raise StateSelectorAmendmentError(
            "phase-2 completion rows differ from active identity-owned evidence"
        )
    checks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in expected_source_rows:
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
    family_rows = [
        {
            "family": family,
            "retained_predecessor_state_count": len(rows),
            "exact_candidate_masks_verified": True,
            "candidate_outcomes_loaded": False,
            "states": sorted(rows, key=lambda row: row["state_identity_digest"]),
        }
        for family, rows in sorted(retained_by_family.items())
    ]
    payload: dict[str, Any] = {
        "schema": PRESERVED_STATE_REVALIDATION_SCHEMA,
        "status": "PASS_POST_ALLOCATION_PRE_OUTCOME_STATE_REVALIDATION",
        "complete": True,
        "source_repository_commit": source_repository_commit,
        "successor_selection_digest": successor_selection_digest,
        "state_selector_amendment_digest": state_selector_amendment_digest(),
        "state_selector_feasibility_receipt_digest":
            state_selector_feasibility_receipt_digest,
        "mixed_precontract_disposition_receipt_digest":
            mixed_precontract_disposition_receipt_digest,
        "candidate_allocation_manifest_digest":
            allocation_manifest["allocation_manifest_digest"],
        "source_identity_manifest_digest":
            allocation_manifest["source_identity_manifest_digest"],
        "candidate_allocator_contract_digest":
            ALLOCATION.allocation_contract_digest(),
        "candidate_allocation_amendment_digest":
            ALLOCATION.allocation_amendment_digest(),
        "candidate_allocation_post_identity_validation_digest":
            allocation_manifest["post_identity_pre_outcome_validation"][
                "post_identity_validation_digest"
            ],
        "original_predecessor_state_count": 45,
        "retained_predecessor_state_count": 37,
        "rejected_predecessor_state_count": 8,
        "replacement_state_count": 8,
        "retained_predecessor_candidate_assignment_count": 37 * 6,
        "retained_predecessor_candidate_masks": retained_masks,
        "retained_predecessor_candidate_mask_set_digest": _sha256(retained_masks),
        "retained_predecessor_family_rows": family_rows,
        "rejected_predecessor_identity_set_digest": mixed[
            "rejected_predecessor_identity_set_digest"
        ],
        "replacement_identities": replacements,
        "replacement_identity_set_digest": _sha256(replacements),
        "completion_enriched_state_count": len(checks),
        "completion_exact_allocated_reachability_pass_count": len(checks),
        "completion_exact_allocated_reachability_checks": checks,
        "completion_exact_allocated_reachability_set_digest": _sha256(checks),
        "all_completion_identities_pass_exact_allocated_mask": True,
        "candidate_outcomes_loaded": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
    }
    payload["preserved_state_revalidation_receipt_digest"] = _sha256(payload)
    return payload


def build_preserved_state_revalidation_receipt(
    *,
    allocation_manifest: Mapping[str, Any],
    active_states: Sequence[Mapping[str, Any]],
    completion_states: Sequence[Mapping[str, Any]],
    source_repository_commit: str,
    successor_selection_digest: str,
    state_selector_feasibility_receipt_digest: str,
    mixed_precontract_disposition_receipt_digest: str,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Legacy builder retaining the allocator's live canonicality re-solve."""

    ALLOCATION.validate_allocation_manifest(allocation_manifest)
    return _build_preserved_state_revalidation_receipt_after_allocation_validation(
        allocation_manifest=allocation_manifest,
        active_states=active_states,
        completion_states=completion_states,
        source_repository_commit=source_repository_commit,
        successor_selection_digest=successor_selection_digest,
        state_selector_feasibility_receipt_digest=(
            state_selector_feasibility_receipt_digest
        ),
        mixed_precontract_disposition_receipt_digest=(
            mixed_precontract_disposition_receipt_digest
        ),
        root=root,
    )


def build_preserved_state_revalidation_receipt_from_solve_free_certified_allocation(
    *,
    allocation_manifest: Mapping[str, Any],
    active_states: Sequence[Mapping[str, Any]],
    completion_states: Sequence[Mapping[str, Any]],
    source_repository_commit: str,
    successor_selection_digest: str,
    state_selector_feasibility_receipt_digest: str,
    mixed_precontract_disposition_receipt_digest: str,
    certify_allocation_solve_free: Callable[
        [Mapping[str, Any]], Mapping[str, Any]
    ],
    root: Path = ROOT,
) -> dict[str, Any]:
    """Build byte-identical phase-2 evidence from a frozen certificate replay.

    The callback must return the exact allocation it certified.  No
    certificate metadata is added to the receipt: the existing allocation
    manifest and post-identity digests already bind those exact bytes, keeping
    the scientific receipt schema and digest semantics unchanged.
    """

    validate_solve_free_certified_allocation_manifest(
        allocation_manifest,
        certify_allocation_solve_free=certify_allocation_solve_free,
    )
    return _build_preserved_state_revalidation_receipt_after_allocation_validation(
        allocation_manifest=allocation_manifest,
        active_states=active_states,
        completion_states=completion_states,
        source_repository_commit=source_repository_commit,
        successor_selection_digest=successor_selection_digest,
        state_selector_feasibility_receipt_digest=(
            state_selector_feasibility_receipt_digest
        ),
        mixed_precontract_disposition_receipt_digest=(
            mixed_precontract_disposition_receipt_digest
        ),
        root=root,
    )


def _validate_preserved_state_revalidation_receipt(
    receipt: Mapping[str, Any],
    *,
    allocation_manifest: Mapping[str, Any],
    active_states: Sequence[Mapping[str, Any]],
    expected_source_commit: str | None = None,
    expected_successor_selection_digest: str | None = None,
    expected_feasibility_receipt_digest: str | None = None,
    expected_mixed_precontract_disposition_receipt_digest: str | None = None,
    certify_allocation_solve_free: Callable[
        [Mapping[str, Any]], Mapping[str, Any]
    ] | None = None,
    root: Path = ROOT,
) -> None:
    """Validate one generic phase-2 digest over masks and 40 reachability rows."""

    if not isinstance(receipt, Mapping):
        raise StateSelectorAmendmentError("phase-2 state receipt must be a mapping")
    payload = dict(receipt)
    observed = payload.pop("preserved_state_revalidation_receipt_digest", None)
    if not _is_digest(observed) or _sha256(payload) != observed:
        raise StateSelectorAmendmentError("phase-2 state receipt self digest failed")
    expected_keys = {
        "schema", "status", "complete", "source_repository_commit",
        "successor_selection_digest", "state_selector_amendment_digest",
        "state_selector_feasibility_receipt_digest",
        "mixed_precontract_disposition_receipt_digest",
        "candidate_allocation_manifest_digest", "source_identity_manifest_digest",
        "candidate_allocator_contract_digest",
        "candidate_allocation_amendment_digest",
        "candidate_allocation_post_identity_validation_digest",
        "original_predecessor_state_count", "retained_predecessor_state_count",
        "rejected_predecessor_state_count", "replacement_state_count",
        "retained_predecessor_candidate_assignment_count",
        "retained_predecessor_candidate_masks",
        "retained_predecessor_candidate_mask_set_digest",
        "retained_predecessor_family_rows",
        "rejected_predecessor_identity_set_digest", "replacement_identities",
        "replacement_identity_set_digest", "completion_enriched_state_count",
        "completion_exact_allocated_reachability_pass_count",
        "completion_exact_allocated_reachability_checks",
        "completion_exact_allocated_reachability_set_digest",
        "all_completion_identities_pass_exact_allocated_mask",
        "candidate_outcomes_loaded", "branches_attempted", "frames_rendered",
        "target_latents_encoded", "scorer_training_started",
    }
    if set(payload) != expected_keys:
        raise StateSelectorAmendmentError(
            "phase-2 state receipt has an unexpected key surface"
        )
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
        (payload.get("mixed_precontract_disposition_receipt_digest"),
         expected_mixed_precontract_disposition_receipt_digest,
         "mixed disposition"),
    ):
        if expected is not None and value != expected:
            raise StateSelectorAmendmentError(f"phase-2 {label} binding mismatch")
    completion_rows = payload.get(
        "completion_exact_allocated_reachability_checks"
    )
    if not isinstance(completion_rows, list):
        raise StateSelectorAmendmentError("phase-2 completion evidence is missing")
    # Rebuilding reopens the mixed disposition, the live 120 identities, all
    # 37 retained masks, all eight replacements and all forty V2 predicates.
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
    build_arguments = {
        "allocation_manifest": allocation_manifest,
        "active_states": active_states,
        "completion_states": source_rows,
        "source_repository_commit": str(payload["source_repository_commit"]),
        "successor_selection_digest": str(
            payload["successor_selection_digest"]
        ),
        "state_selector_feasibility_receipt_digest": str(
            payload["state_selector_feasibility_receipt_digest"]
        ),
        "mixed_precontract_disposition_receipt_digest": str(
            payload["mixed_precontract_disposition_receipt_digest"]
        ),
        "root": root,
    }
    if certify_allocation_solve_free is None:
        expected = build_preserved_state_revalidation_receipt(
            **build_arguments
        )
    else:
        expected = (
            build_preserved_state_revalidation_receipt_from_solve_free_certified_allocation(
                **build_arguments,
                certify_allocation_solve_free=certify_allocation_solve_free,
            )
        )
    if dict(receipt) != expected:
        raise StateSelectorAmendmentError(
            "phase-2 receipt differs from exact masks/reachability reconstruction"
        )


def validate_preserved_state_revalidation_receipt(
    receipt: Mapping[str, Any],
    *,
    allocation_manifest: Mapping[str, Any],
    active_states: Sequence[Mapping[str, Any]],
    expected_source_commit: str | None = None,
    expected_successor_selection_digest: str | None = None,
    expected_feasibility_receipt_digest: str | None = None,
    expected_mixed_precontract_disposition_receipt_digest: str | None = None,
    root: Path = ROOT,
) -> None:
    """Legacy validator retaining the allocator's live canonicality re-solve."""

    _validate_preserved_state_revalidation_receipt(
        receipt,
        allocation_manifest=allocation_manifest,
        active_states=active_states,
        expected_source_commit=expected_source_commit,
        expected_successor_selection_digest=(
            expected_successor_selection_digest
        ),
        expected_feasibility_receipt_digest=(
            expected_feasibility_receipt_digest
        ),
        expected_mixed_precontract_disposition_receipt_digest=(
            expected_mixed_precontract_disposition_receipt_digest
        ),
        root=root,
    )


def validate_preserved_state_revalidation_receipt_from_solve_free_certified_allocation(
    receipt: Mapping[str, Any],
    *,
    allocation_manifest: Mapping[str, Any],
    active_states: Sequence[Mapping[str, Any]],
    certify_allocation_solve_free: Callable[
        [Mapping[str, Any]], Mapping[str, Any]
    ],
    expected_source_commit: str | None = None,
    expected_successor_selection_digest: str | None = None,
    expected_feasibility_receipt_digest: str | None = None,
    expected_mixed_precontract_disposition_receipt_digest: str | None = None,
    root: Path = ROOT,
) -> None:
    """Reconstruct and validate phase-2 evidence with no allocation solve."""

    _validate_preserved_state_revalidation_receipt(
        receipt,
        allocation_manifest=allocation_manifest,
        active_states=active_states,
        expected_source_commit=expected_source_commit,
        expected_successor_selection_digest=(
            expected_successor_selection_digest
        ),
        expected_feasibility_receipt_digest=(
            expected_feasibility_receipt_digest
        ),
        expected_mixed_precontract_disposition_receipt_digest=(
            expected_mixed_precontract_disposition_receipt_digest
        ),
        certify_allocation_solve_free=certify_allocation_solve_free,
        root=root,
    )


# Outcome-free predecessor helpers whose semantics are unchanged in V2.
preserved_state_identity_set_digest = PREDECESSOR.preserved_state_identity_set_digest
candidate_mask_digest = PREDECESSOR.candidate_mask_digest


def load_preserved_state_shards(
    root: Path = ROOT,
) -> dict[str, dict[str, Any]]:
    """Load the exact three predecessor shards through the V2 custody guard."""

    loaded: dict[str, dict[str, Any]] = {}
    for expected in PRESERVED_STATE_SHARDS:
        path = _managed_generated_artifact_path(
            root / str(expected["path"]), root=root
        )
        if (
            not path.is_file()
            or path.stat().st_size != int(expected["byte_count"])
            or _file_sha256(path) != expected["raw_sha256"]
        ):
            raise StateSelectorAmendmentError(
                f"preserved state shard {expected['family']} raw binding failed"
            )
        try:
            shard = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise StateSelectorAmendmentError(
                f"preserved state shard {expected['family']} JSON is invalid"
            ) from exc
        observed = shard.get("state_shard_digest")
        if (
            observed != expected["state_shard_digest"]
            or _sha256({
                key: value for key, value in shard.items()
                if key != "state_shard_digest"
            }) != observed
            or shard.get("complete") is not True
            or shard.get("family") != expected["family"]
            or len(shard.get("states", ())) != expected["state_count"]
        ):
            raise StateSelectorAmendmentError(
                f"preserved state shard {expected['family']} content binding failed"
            )
        loaded[str(expected["family"])] = shard
    return loaded


_validate_frozen_sources()
