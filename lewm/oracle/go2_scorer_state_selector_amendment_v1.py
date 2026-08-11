"""Prospective scorer-fit state-selector amendment V1.

This module records the single pre-outcome repair authorised after the frozen
selector was proven unable to instantiate the completion-enriched quota in two
required scene families.  It changes goal *eligibility for state selection*
only.  The oracle, candidate bank, completion predicate, numerical threshold,
state quotas, allocation policy, and every scorer criterion remain unchanged.

The module is deliberately free of Genesis imports.  Runtime code consumes the
contract and emits a separately self-verifying, outcome-free receipt when it
revalidates the 45 already selected states under the amended selector.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]

AMENDMENT_SCHEMA = "go2_scorer_fit_state_selector_amendment_v1"
AMENDMENT_VERSION = "v1"
AMENDMENT_ARTIFACT_PATH = (
    "docs/lewm_go2_shared_utility_scorer_v1_2_state_selector_"
    "amendment_v1_2026-08-11.json"
)

FAILURE_RECEIPT_PATH = (
    "docs/lewm_go2_shared_utility_scorer_v1_2_preoutcome_"
    "state_selection_failure_2026-08-11.json"
)
FAILURE_RECEIPT_DIGEST = (
    "47c2bcc7cfaf79b328cd5a1cf2823554f2553fc419e020559fde1351df2ca75f"
)
FAILURE_RECEIPT_RAW_SHA256 = (
    "26e372229af4bcde8242062b0b9f9d9c5ba85bdb073537812e9395a608c2455d"
)
FAILURE_RECEIPT_BYTE_COUNT = 8_586
AUTHORIZING_FAILURE_COMMIT = "19e83aaf2d99463954d2c5889528a5016ec72095"

ALLOCATION_AMENDMENT_DIGEST = (
    "4dde3562cdd9e503d6e264a5d4982a189a9f43d338c3d6b87ee20de352bc3cbc"
)
ALLOCATION_AMENDMENT_RAW_SHA256 = (
    "1790429d6c02deebc794aa255be3b8c93ac5278de9c8c94920ee13b877fb5f38"
)

PREDECESSOR_SELECTION_DIGEST = (
    "341de51facbb34b7361175bb713bbcef0fedb9cfc837a5adb6e2c888829a1df1"
)
PREDECESSOR_SCORER_CONTRACT_DIGEST = (
    "06263907d8f8df0fe735f95da26c10fab9dff4af6827562622aa66463b456c0b"
)
PREDECESSOR_CONTRACT_ARTIFACT = {
    "path": ".generated/go2_utility_scorer_v1_2/scorer_contract_v1_2.json",
    "scorer_contract_v1_2_digest": PREDECESSOR_SCORER_CONTRACT_DIGEST,
    "contract_artifact_digest": (
        "116a7e77a7888788048a9fddcb3b7a1eaf62ea655890503ea09e08ebc91b898d"
    ),
    "raw_sha256": (
        "3ea9d04c4bf19e21713bac9d724581beac5b0ba41c1c4eb2ee5c017c785d0de2"
    ),
    "byte_count": 26_691,
    "clean_source_launch_receipt_digest": (
        "7ab90a7fc6cdde04a0982701b008bc9d00b47ea8c0baecf47f775dcef6d64520"
    ),
    "outcomes_generated": False,
    "disposition": "SUPERSEDED_PRE_OUTCOME_GRAPH_DISCRETIZATION_INFEASIBLE",
}

COMPLETION_MAX_GEODESIC_M = 0.75
COMPLETION_MAX_ABS_BEARING_DEG = 75.0
HORIZON_S = 2.0
V_MAX_MPS = 0.30
MAX_TRANSLATION_M = HORIZON_S * V_MAX_MPS
ORACLE_V1_2_DIGEST = (
    "3ffbe1a87f7975c97e7ff42e50a6a00ca0f47d8840a434d0ff215c303bf6f0e4"
)
COMPLETION_SEMANTIC_SOURCE_BINDINGS = {
    "oracle_v1_2_completion_target": {
        "path": "lewm/oracle/go2_branch_oracle_v1_2.py",
        "sha256": "6d7a6b20bcfb5da112ff10e95a7d3573ebf07884e7b4e58315a733254d6f4fc2",
        "byte_count": 13_637,
        "symbols": ["COMPLETION_CONTRACT", "branch_components"],
    },
    "snapshot_production_designated_goal_claim": {
        "path": "lewm_genesis/lewm_genesis/collectors/route_teacher.py",
        "sha256": "2ccd4fba3ef45bd3f75358dc617f4d575b96d7c6e4d3a2538b76e31bc3d79968",
        "byte_count": 51_313,
        "symbols": [
            "RouteTeacherPolicy._evaluate_arrival_gates",
            "RouteTeacherPolicy._landmark_visible",
            "RouteTeacherPolicy.visited_landmark_cells",
        ],
    },
    "production_task_completion_and_reset": {
        "path": "lewm_genesis/lewm_genesis/rollout.py",
        "sha256": "06501bbbdd1e071a3a91e765d77bd19da5f2c311c35d75df4631c452beea034a",
        "byte_count": 86_066,
        "symbols": ["RolloutRunner._check_and_reset_completed_envs"],
    },
}
SCORER_FIT_SELECTION_PRIORITY = (
    "general", "safety_enriched", "completion_enriched",
)

PRESERVED_STATE_PRECONTRACT_REVALIDATION_SCHEMA = (
    "go2_scorer_fit_preserved_state_precontract_revalidation_receipt_v1"
)
PRESERVED_STATE_PRECONTRACT_REVALIDATION_RECEIPT_NAME = (
    "preserved_state_precontract_revalidation_receipt.json"
)
PRESERVED_STATE_PRECONTRACT_REVALIDATION_RECEIPT_PATH = (
    ".generated/go2_branch_corpus_v1_2/scorer_fit/"
    + PRESERVED_STATE_PRECONTRACT_REVALIDATION_RECEIPT_NAME
)
PRESERVED_STATE_REVALIDATION_SCHEMA = (
    "go2_scorer_fit_preserved_state_revalidation_receipt_v1"
)
PRESERVED_STATE_REVALIDATION_RECEIPT_NAME = (
    "preserved_state_revalidation_receipt.json"
)
PRESERVED_STATE_REVALIDATION_RECEIPT_PATH = (
    ".generated/go2_branch_corpus_v1_2/scorer_fit/"
    + PRESERVED_STATE_REVALIDATION_RECEIPT_NAME
)
STATE_SELECTOR_FEASIBILITY_SCHEMA = (
    "go2_scorer_fit_state_selector_feasibility_receipt_v1"
)
STATE_SELECTOR_FEASIBILITY_RECEIPT_NAME = (
    "state_selector_feasibility_receipt.json"
)
STATE_SELECTOR_FEASIBILITY_RECEIPT_PATH = (
    ".generated/go2_branch_corpus_v1_2/scorer_fit/"
    + STATE_SELECTOR_FEASIBILITY_RECEIPT_NAME
)
REQUIRED_FAMILIES = (
    "large_enclosed_maze",
    "local_composite_motifs",
    "loop_alias_stress",
    "medium_enclosed_maze",
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "visual_sensor_stress",
)
REQUIRED_STRATA = (
    "general", "safety_enriched", "completion_enriched",
)
PRESERVED_STATE_SHARDS = (
    {
        "family": "large_enclosed_maze",
        "path": (
            ".generated/go2_branch_corpus_v1_2/scorer_fit/"
            "state_shard_large_enclosed_maze.json"
        ),
        "state_shard_digest": (
            "3b1cabb8cf104f0edc19e27d9f11922655a5424b32a4ad41ec3d5466b4193914"
        ),
        "raw_sha256": (
            "289f63e5a607fb49c62a996043019ca8b6ef2def31ae39afc0a87ddbca71e866"
        ),
        "byte_count": 41_994,
        "state_count": 15,
    },
    {
        "family": "local_composite_motifs",
        "path": (
            ".generated/go2_branch_corpus_v1_2/scorer_fit/"
            "state_shard_local_composite_motifs.json"
        ),
        "state_shard_digest": (
            "e066820d0a85e53a8b7e30a4b4fe1386df3e315674ebd094b0beb64c74242321"
        ),
        "raw_sha256": (
            "eca93582615e946b0a3c4692dc9353cb826f6e3d4b5e6ccd2c706ee9f53ee0dd"
        ),
        "byte_count": 36_323,
        "state_count": 15,
    },
    {
        "family": "loop_alias_stress",
        "path": (
            ".generated/go2_branch_corpus_v1_2/scorer_fit/"
            "state_shard_loop_alias_stress.json"
        ),
        "state_shard_digest": (
            "9aee1729903e555273a792c03a5290f7547645995164042f9f021fec48de3985"
        ),
        "raw_sha256": (
            "9b7d7ee1f0029b3cd64eb3127687ba03e67eab4c75e4c4600a375afbf721acf6"
        ),
        "byte_count": 37_216,
        "state_count": 15,
    },
)


class StateSelectorAmendmentError(RuntimeError):
    """Raised when a selector-amendment binding is incomplete or altered."""


def _sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


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
        and all(character in "0123456789abcdef" for character in value)
    )


def state_selector_amendment_contract() -> dict[str, Any]:
    """Return the sole prospective repair to scorer-fit state selection."""

    return {
        "schema": AMENDMENT_SCHEMA,
        "status": "AUTHORIZED_PROSPECTIVE_PRE_OUTCOME_SELECTOR_AMENDMENT",
        "version": AMENDMENT_VERSION,
        "lineage": {
            "authorizing_failure_commit": AUTHORIZING_FAILURE_COMMIT,
            "failure_receipt_path": FAILURE_RECEIPT_PATH,
            "failure_receipt_digest": FAILURE_RECEIPT_DIGEST,
            "failure_receipt_raw_sha256": FAILURE_RECEIPT_RAW_SHA256,
            "failure_receipt_byte_count": FAILURE_RECEIPT_BYTE_COUNT,
            "predecessor_selection_digest": PREDECESSOR_SELECTION_DIGEST,
            "predecessor_scorer_contract_digest":
                PREDECESSOR_SCORER_CONTRACT_DIGEST,
            "predecessor_contract_artifact": PREDECESSOR_CONTRACT_ARTIFACT,
            "unchanged_candidate_allocation_amendment_digest":
                ALLOCATION_AMENDMENT_DIGEST,
            "unchanged_candidate_allocation_amendment_raw_sha256":
                ALLOCATION_AMENDMENT_RAW_SHA256,
        },
        "superseded_conjunction": {
            "status": "SUPERSEDED_PRE_OUTCOME_GRAPH_DISCRETIZATION_INFEASIBLE",
            "rule": (
                "one shared bound-landmark enumeration rejects graph_hops < 1 "
                "before applying completion geodesic <= 0.75m"
            ),
            "failure": (
                "in 2.4m graph lattices a pose outside the goal cell is at "
                "least 1.2m from its centre, while a pose inside has hops=0"
            ),
            "affected_families": [
                "open_obstacle_field", "rough_local_dynamics",
            ],
            "candidate_outcomes_observed": False,
        },
        "replacement": {
            "separate_goal_enumeration_by_stratum": True,
            "general_and_safety": (
                "unchanged nearest finite reachable landmark enumeration; "
                "general requires graph_hops >= 2 and safety remains a subset"
            ),
            "completion": (
                "enumerate finite snapshot-bound landmark fields including "
                "graph_hops == 0; graph_hops is diagnostic only for completion"
            ),
            "completion_goal_order": (
                "argmin over (continuous_metric_geodesic_m, landmark_id, "
                "landmark_cell, graph_hops_diagnostic)"
            ),
            "completion_requirements": {
                "reachable": True,
                "continuous_metric_geodesic_m_max":
                    COMPLETION_MAX_GEODESIC_M,
                "absolute_body_bearing_deg_max":
                    COMPLETION_MAX_ABS_BEARING_DEG,
                "snapshot_task_completed": False,
                "snapshot_goal_claimed": False,
                "snapshot_terminated": False,
                "snapshot_truncated": False,
            },
            "snapshot_task_status_sources": {
                "goal_claimed": (
                    "designated goal cell is in the active production "
                    "collector visited_landmark_cells(env_idx); that set is "
                    "populated only by the collector's actual range-envelope, "
                    "camera-FOV and line-of-sight arrival gates; false when "
                    "the accessor is unavailable"
                ),
                "task_completed": (
                    "exact RolloutRunner._check_and_reset_completed_envs reset "
                    "predicate: minimum-block guard, active route-like collector, "
                    "non-revisit policy, and all scene landmark cells present in "
                    "the collector's production claimed set"
                ),
                "terminated": (
                    "any run_go2_oracle_branch_pilot_v1._termination_flags value "
                    "is true, including nan"
                ),
                "truncated": (
                    "literal false asserted at the pre-branch canonical state "
                    "boundary; no candidate branch or invocation truncation is "
                    "in progress"
                ),
            },
            "state_selection_priority": list(SCORER_FIT_SELECTION_PRIORITY),
        },
        "preserved": {
            "completion_geodesic_threshold_m": COMPLETION_MAX_GEODESIC_M,
            "completion_absolute_bearing_threshold_deg":
                COMPLETION_MAX_ABS_BEARING_DEG,
            "horizon_s": HORIZON_S,
            "candidate_bank_max_translation_speed_mps": V_MAX_MPS,
            "candidate_bank_max_translation_over_horizon_m": MAX_TRANSLATION_M,
            "unchanged_completion_semantics": {
                "oracle_v1_2_completion_target": {
                    "definition": (
                        "the snapshot-bound landmark goal cell is reached at any "
                        "candidate branch tick at or before the four-block horizon"
                    ),
                    "complete_oracle_v1_2_digest": ORACLE_V1_2_DIGEST,
                    "source_binding": COMPLETION_SEMANTIC_SOURCE_BINDINGS[
                        "oracle_v1_2_completion_target"
                    ],
                },
                "snapshot_production_designated_goal_claim": {
                    "definition": (
                        "at snapshot time the designated goal cell is present in "
                        "the active production collector's actual visited-landmark "
                        "set, populated by range-envelope, camera-FOV and "
                        "line-of-sight arrival gates"
                    ),
                    "source_binding": COMPLETION_SEMANTIC_SOURCE_BINDINGS[
                        "snapshot_production_designated_goal_claim"
                    ],
                },
                "production_task_completion_and_reset": {
                    "definition": (
                        "after the minimum-block guard, an active route-like "
                        "non-revisit collector has claimed every scene landmark "
                        "cell, causing RolloutRunner to reset that environment"
                    ),
                    "source_binding": COMPLETION_SEMANTIC_SOURCE_BINDINGS[
                        "production_task_completion_and_reset"
                    ],
                },
                "not_interchangeable": True,
            },
            "oracle_v1_2": True,
            "candidate_bank": True,
            "candidate_allocation_amendment": True,
            "state_and_family_quotas": True,
            "scene_order_and_one_state_per_scene": True,
            "warmup_blocks_40_through_120_inclusive": True,
            "drive_seed_and_cpu_backend": True,
            "all_scene_and_identity_exclusions": True,
            "fit_calibration_rule": True,
            "general_and_safety_stratum_semantics": True,
            "candidate_outcomes_used_for_selection": False,
        },
        "preserved_state_revalidation": {
            "expected_state_count": 45,
            "expected_family_count": 3,
            "source_shards": [dict(row) for row in PRESERVED_STATE_SHARDS],
            "candidate_outcomes_may_be_loaded": False,
            "failed_or_changed_state_disposition": (
                "preserve the predecessor shard; do not mix it into the "
                "successor identity manifest"
            ),
            "phase_1_precontract": {
                "required_before_successor_contract_issue": True,
                "receipt_schema":
                    PRESERVED_STATE_PRECONTRACT_REVALIDATION_SCHEMA,
                "receipt_path":
                    PRESERVED_STATE_PRECONTRACT_REVALIDATION_RECEIPT_PATH,
                "requirements": (
                    "exact redrive, amended classification, exclusion and "
                    "unchanged identity checks only; no candidate allocation"
                ),
            },
            "phase_2_post_allocation": {
                "required_before_active_identity_manifest": True,
                "receipt_schema": PRESERVED_STATE_REVALIDATION_SCHEMA,
                "receipt_path": PRESERVED_STATE_REVALIDATION_RECEIPT_PATH,
                "requirements": (
                    "all 45 unchanged identities carried into the complete "
                    "120-state allocation with exact six-candidate masks and "
                    "post-identity pre-outcome validation"
                ),
            },
            "active_manifest_mixed_preoutcome_provenance": {
                "predecessor_shards_retained_byte_exact": 3,
                "successor_shards_selected_under_amendment": 5,
                "predecessor_state_identity_digests_recomputed": False,
                "legacy_state_shard_digests_mapping_retained": True,
                "required_state_shard_provenance_entries": 8,
                "manifest_field": "state_shard_provenance",
                "required_per_entry_fields": [
                    "family", "path", "state_shard_digest", "raw_sha256",
                    "byte_count", "selection_provenance",
                ],
                "selection_provenance_values": {
                    "predecessor": "PREDECESSOR_BYTE_EXACT_REVALIDATED",
                    "successor": "SUCCESSOR_SELECTOR_AMENDMENT_V1",
                },
            },
        },
        "state_selector_feasibility": {
            "required_before_successor_contract_issue": True,
            "receipt_schema": STATE_SELECTOR_FEASIBILITY_SCHEMA,
            "receipt_path": STATE_SELECTOR_FEASIBILITY_RECEIPT_PATH,
            "required_families": list(REQUIRED_FAMILIES),
            "required_strata": list(REQUIRED_STRATA),
            "required_distinct_eligible_scenes_per_family_stratum": 5,
            "all_eligible_scenes_scanned": True,
            "selected_state_identities_created": False,
            "candidate_outcomes_may_be_loaded": False,
            "family_or_stratum_scoped_diagnostics_are_not_binding": True,
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
            "tracked selector amendment differs from the code contract"
        )
    if observed != state_selector_amendment_digest():
        raise StateSelectorAmendmentError("selector amendment digest mismatch")


def validate_authority_artifacts(root: Path = ROOT) -> None:
    """Fail closed on the tracked failure/amendment authority chain."""

    failure = root / FAILURE_RECEIPT_PATH
    if (not failure.is_file()
            or failure.stat().st_size != FAILURE_RECEIPT_BYTE_COUNT
            or _file_sha256(failure) != FAILURE_RECEIPT_RAW_SHA256):
        raise StateSelectorAmendmentError(
            "state-selection failure receipt raw binding failed"
        )
    failure_payload = json.loads(failure.read_text())
    observed = failure_payload.pop("failure_report_digest", None)
    if observed != FAILURE_RECEIPT_DIGEST or _sha256(failure_payload) != observed:
        raise StateSelectorAmendmentError(
            "state-selection failure receipt self binding failed"
        )

    amendment = root / AMENDMENT_ARTIFACT_PATH
    if not amendment.is_file():
        raise StateSelectorAmendmentError("selector amendment artifact is missing")
    validate_state_selector_amendment_artifact(json.loads(amendment.read_text()))
    for label, binding in COMPLETION_SEMANTIC_SOURCE_BINDINGS.items():
        source = root / binding["path"]
        if (not source.is_file()
                or source.stat().st_size != binding["byte_count"]
                or _file_sha256(source) != binding["sha256"]):
            raise StateSelectorAmendmentError(
                f"{label} source binding failed"
            )


def preserved_state_identity_set_digest(states: Sequence[Mapping[str, Any]]) -> str:
    digests = sorted(str(row.get("state_identity_digest", "")) for row in states)
    if not digests or any(not _is_digest(value) for value in digests):
        raise StateSelectorAmendmentError(
            "preserved states require complete state identity digests"
        )
    return _sha256(digests)


def load_preserved_state_shards(
    root: Path = ROOT,
) -> dict[str, dict[str, Any]]:
    """Load only the three exactly bound, outcome-free predecessor shards."""

    loaded: dict[str, dict[str, Any]] = {}
    for expected in PRESERVED_STATE_SHARDS:
        path = root / expected["path"]
        if (not path.is_file()
                or path.stat().st_size != expected["byte_count"]
                or _file_sha256(path) != expected["raw_sha256"]):
            raise StateSelectorAmendmentError(
                f"preserved state shard {expected['family']} raw binding failed"
            )
        shard = json.loads(path.read_text())
        observed = shard.get("state_shard_digest")
        if (observed != expected["state_shard_digest"]
                or _sha256({key: value for key, value in shard.items()
                            if key != "state_shard_digest"}) != observed
                or shard.get("complete") is not True
                or shard.get("family") != expected["family"]
                or len(shard.get("states", ())) != expected["state_count"]):
            raise StateSelectorAmendmentError(
                f"preserved state shard {expected['family']} content binding failed"
            )
        loaded[str(expected["family"])] = shard
    return loaded


def validate_state_selector_feasibility_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_source_commit: str | None = None,
    expected_successor_selection_digest: str | None = None,
) -> None:
    """Validate the all-family, identity-free selector feasibility gate."""

    if not isinstance(receipt, Mapping):
        raise StateSelectorAmendmentError("selector feasibility receipt must be a mapping")
    payload = dict(receipt)
    observed_digest = payload.pop("state_selector_feasibility_receipt_digest", None)
    if not _is_digest(observed_digest) or _sha256(payload) != observed_digest:
        raise StateSelectorAmendmentError("selector feasibility receipt self digest failed")
    if (payload.get("schema") != STATE_SELECTOR_FEASIBILITY_SCHEMA
            or payload.get("status") != "PASS_OUTCOME_FREE_ALL_SCENE_FEASIBILITY"
            or payload.get("complete") is not True):
        raise StateSelectorAmendmentError("selector feasibility receipt is not complete/pass")
    if payload.get("state_selector_amendment_digest") != state_selector_amendment_digest():
        raise StateSelectorAmendmentError("selector feasibility amendment mismatch")
    if (expected_source_commit is not None
            and payload.get("source_repository_commit") != expected_source_commit):
        raise StateSelectorAmendmentError("selector feasibility source commit mismatch")
    if (expected_successor_selection_digest is not None
            and payload.get("successor_selection_digest")
            != expected_successor_selection_digest):
        raise StateSelectorAmendmentError("selector feasibility selection mismatch")
    forbidden = (
        "selected_state_identities_created", "candidate_outcomes_loaded",
        "branch_identities_created", "branches_attempted", "frames_rendered",
        "target_latents_encoded", "scorer_training_started",
    )
    if any(payload.get(key) not in (False, 0) for key in forbidden):
        raise StateSelectorAmendmentError(
            "selector feasibility receipt contains forbidden scientific operations"
        )
    if (payload.get("family_count") != len(REQUIRED_FAMILIES)
            or payload.get("strata") != list(REQUIRED_STRATA)
            or payload.get("required_distinct_scenes_per_stratum") != 5):
        raise StateSelectorAmendmentError("selector feasibility global quota mismatch")
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
        if (row.get("all_allowed_scenes_scanned") is not True
                or row.get("verdict") != "PASS"):
            raise StateSelectorAmendmentError(
                f"selector feasibility family {family} did not pass full scan"
            )
        strata = row.get("strata")
        if not isinstance(strata, Mapping) or set(strata) != set(REQUIRED_STRATA):
            raise StateSelectorAmendmentError(
                f"selector feasibility family {family} stratum coverage mismatch"
            )
        for stratum in REQUIRED_STRATA:
            evidence = strata[stratum]
            if (not isinstance(evidence, Mapping)
                    or evidence.get("required_distinct_scenes") != 5
                    or int(evidence.get("eligible_distinct_scenes", -1)) < 5
                    or evidence.get("verdict") != "PASS"):
                raise StateSelectorAmendmentError(
                    f"selector feasibility {family}/{stratum} failed"
                )
    if seen != set(REQUIRED_FAMILIES):
        raise StateSelectorAmendmentError("selector feasibility missing family")


def validate_preserved_state_precontract_revalidation_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_source_commit: str | None = None,
    expected_successor_selection_digest: str | None = None,
    expected_feasibility_receipt_digest: str | None = None,
    root: Path = ROOT,
) -> None:
    """Validate phase 1: exact identity-only checks before contract issue."""

    if not isinstance(receipt, Mapping):
        raise StateSelectorAmendmentError("state revalidation receipt must be a mapping")
    payload = dict(receipt)
    observed_digest = payload.pop(
        "preserved_state_precontract_revalidation_receipt_digest", None,
    )
    if not _is_digest(observed_digest) or _sha256(payload) != observed_digest:
        raise StateSelectorAmendmentError("state revalidation receipt self digest failed")
    if (payload.get("schema")
            != PRESERVED_STATE_PRECONTRACT_REVALIDATION_SCHEMA
            or payload.get("status")
            != "PASS_PRECONTRACT_IDENTITY_REVALIDATION"
            or payload.get("complete") is not True):
        raise StateSelectorAmendmentError("state revalidation receipt is not complete/pass")
    if payload.get("state_selector_amendment_digest") != state_selector_amendment_digest():
        raise StateSelectorAmendmentError("state revalidation selector binding mismatch")
    if payload.get("predecessor_selection_digest") != PREDECESSOR_SELECTION_DIGEST:
        raise StateSelectorAmendmentError("state revalidation predecessor selection mismatch")
    if (payload.get("predecessor_scorer_contract_digest")
            != PREDECESSOR_SCORER_CONTRACT_DIGEST):
        raise StateSelectorAmendmentError("state revalidation predecessor contract mismatch")
    if (expected_source_commit is not None
            and payload.get("source_repository_commit") != expected_source_commit):
        raise StateSelectorAmendmentError("state revalidation source commit mismatch")
    if (expected_successor_selection_digest is not None
            and payload.get("successor_selection_digest")
            != expected_successor_selection_digest):
        raise StateSelectorAmendmentError("state revalidation successor selection mismatch")
    if (expected_feasibility_receipt_digest is not None
            and payload.get("state_selector_feasibility_receipt_digest")
            != expected_feasibility_receipt_digest):
        raise StateSelectorAmendmentError("state revalidation feasibility binding mismatch")

    forbidden_true = (
        "candidate_outcomes_loaded", "candidate_allocation_loaded",
        "branch_identities_created",
        "branches_attempted", "frames_rendered", "target_latents_encoded",
        "scorer_training_started",
    )
    if any(payload.get(key) not in (False, 0) for key in forbidden_true):
        raise StateSelectorAmendmentError(
            "state revalidation receipt contains forbidden scientific operations"
        )
    if payload.get("preserved_state_count") != 45:
        raise StateSelectorAmendmentError("state revalidation must cover exactly 45 states")

    expected_by_family = {row["family"]: row for row in PRESERVED_STATE_SHARDS}
    predecessor_shards = load_preserved_state_shards(root)
    rows = payload.get("shards")
    if not isinstance(rows, list) or len(rows) != len(expected_by_family):
        raise StateSelectorAmendmentError("state revalidation shard coverage mismatch")
    seen: set[str] = set()
    total = 0
    state_digests: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise StateSelectorAmendmentError("state revalidation shard row is malformed")
        family = str(row.get("family"))
        if family in seen or family not in expected_by_family:
            raise StateSelectorAmendmentError("state revalidation family mismatch")
        seen.add(family)
        expected = expected_by_family[family]
        for key in ("path", "state_shard_digest", "raw_sha256", "byte_count"):
            if row.get(key) != expected[key]:
                raise StateSelectorAmendmentError(
                    f"state revalidation {family} predecessor {key} mismatch"
                )
        count = int(row.get("revalidated_state_count", -1))
        if (count != expected["state_count"]
                or row.get("unchanged_state_identity_count") != count
                or row.get("failed_state_count") != 0
                or row.get("exact_redrive_pass") is not True
                or row.get("amended_classification_pass") is not True
                or row.get("exclusion_checks_pass") is not True
                or row.get("goal_binding_unchanged") is not True
                or row.get("oracle_completion_target_unchanged") is not True
                or row.get(
                    "snapshot_production_designated_goal_claim_unchanged"
                ) is not True
                or row.get("production_task_completion_reset_unchanged") is not True
                or row.get("completion_state_task_status_all_false") is not True
                or row.get("candidate_outcomes_loaded") is not False):
            raise StateSelectorAmendmentError(
                f"state revalidation {family} did not pass exactly"
            )
        digest_values = row.get("state_identity_digests")
        if (not isinstance(digest_values, list) or len(digest_values) != count
                or any(not _is_digest(value) for value in digest_values)):
            raise StateSelectorAmendmentError(
                f"state revalidation {family} identity digest coverage failed"
            )
        predecessor_digests = sorted(
            str(state["state_identity_digest"])
            for state in predecessor_shards[family]["states"]
        )
        if sorted(digest_values) != predecessor_digests:
            raise StateSelectorAmendmentError(
                f"state revalidation {family} changed predecessor identities"
            )
        if row.get("state_identity_set_digest") != _sha256(sorted(digest_values)):
            raise StateSelectorAmendmentError(
                f"state revalidation {family} identity set digest failed"
            )
        checks = row.get("state_checks")
        if not isinstance(checks, list) or len(checks) != count:
            raise StateSelectorAmendmentError(
                f"state revalidation {family} evidence coverage failed"
            )
        expected_state_ids = {
            str(state["state_identity_digest"]): str(state["state_id"])
            for state in predecessor_shards[family]["states"]
        }
        checked_digests: set[str] = set()
        evidence_keys = (
            "exclusion_checks_pass", "exact_redrive_pass",
            "amended_classification_pass", "goal_binding_unchanged",
            "oracle_completion_target_unchanged",
            "snapshot_production_designated_goal_claim_unchanged",
            "production_task_completion_reset_unchanged",
            "completion_state_task_status_all_false",
        )
        for check in checks:
            if not isinstance(check, Mapping):
                raise StateSelectorAmendmentError(
                    f"state revalidation {family} evidence row malformed"
                )
            digest_value = str(check.get("state_identity_digest", ""))
            if (digest_value in checked_digests
                    or digest_value not in expected_state_ids
                    or check.get("state_id") != expected_state_ids[digest_value]
                    or any(check.get(key) is not True for key in evidence_keys)
                    or check.get("failure_reason") is not None):
                raise StateSelectorAmendmentError(
                    f"state revalidation {family} evidence failed"
                )
            checked_digests.add(digest_value)
        if checked_digests != set(predecessor_digests):
            raise StateSelectorAmendmentError(
                f"state revalidation {family} evidence omitted identity"
            )
        state_digests.extend(str(value) for value in digest_values)
        total += count
    if set(expected_by_family) != seen or total != 45 or len(set(state_digests)) != 45:
        raise StateSelectorAmendmentError("state revalidation global state coverage failed")
    if payload.get("state_identity_set_digest") != _sha256(sorted(state_digests)):
        raise StateSelectorAmendmentError("state revalidation global identity digest failed")
    if payload.get("failure_count") != 0 or payload.get("failures") != []:
        raise StateSelectorAmendmentError("state revalidation contains failures")


def candidate_mask_digest(
    state_identity_digest: str, candidate_indices: Sequence[int],
) -> str:
    """Canonical binding of one preserved state to its six allocated candidates."""

    if not _is_digest(state_identity_digest):
        raise StateSelectorAmendmentError("candidate mask state digest is invalid")
    indices = [int(value) for value in candidate_indices]
    if (len(indices) != 6 or indices != sorted(set(indices))
            or any(value < 0 or value >= 12 for value in indices)):
        raise StateSelectorAmendmentError("candidate mask must contain six unique indices")
    return _sha256({
        "state_identity_digest": state_identity_digest,
        "candidate_indices": indices,
    })


def build_preserved_state_revalidation_receipt(
    *,
    allocation_manifest: Mapping[str, Any],
    source_repository_commit: str,
    successor_selection_digest: str,
    state_selector_feasibility_receipt_digest: str,
    preserved_state_precontract_revalidation_receipt_digest: str,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Build phase 2 from the complete allocation without loading outcomes."""

    from lewm.oracle import go2_candidate_allocation_v1_2 as allocation

    if (not isinstance(source_repository_commit, str)
            or len(source_repository_commit) != 40
            or any(value not in "0123456789abcdef"
                   for value in source_repository_commit)):
        raise StateSelectorAmendmentError("final revalidation source commit invalid")
    for label, value in (
        ("successor selection", successor_selection_digest),
        ("feasibility receipt", state_selector_feasibility_receipt_digest),
        ("precontract revalidation receipt",
         preserved_state_precontract_revalidation_receipt_digest),
    ):
        if not _is_digest(value):
            raise StateSelectorAmendmentError(f"{label} digest invalid")

    allocation.validate_allocation_manifest(allocation_manifest)
    assignments = {
        str(row["state_identity_digest"]): row
        for row in allocation_manifest["assignments"]
    }
    predecessor_shards = load_preserved_state_shards(root)
    shard_rows: list[dict[str, Any]] = []
    all_masks: list[dict[str, Any]] = []
    for expected in PRESERVED_STATE_SHARDS:
        family = str(expected["family"])
        states: list[dict[str, Any]] = []
        for source_state in sorted(
            predecessor_shards[family]["states"],
            key=lambda row: str(row["state_identity_digest"]),
        ):
            state_digest = str(source_state["state_identity_digest"])
            assignment = assignments.get(state_digest)
            if (assignment is None or assignment.get("family") != family
                    or assignment.get("state_id") != source_state.get("state_id")):
                raise StateSelectorAmendmentError(
                    f"preserved state {state_digest} is absent from allocation"
                )
            candidate_indices = list(assignment["candidate_indices"])
            states.append({
                "state_identity_digest": state_digest,
                "state_id": source_state["state_id"],
                "candidate_indices": candidate_indices,
                "candidate_mask_digest": candidate_mask_digest(
                    state_digest, candidate_indices,
                ),
            })
            all_masks.append({
                "state_identity_digest": state_digest,
                "candidate_indices": candidate_indices,
            })
        shard_rows.append({
            **dict(expected),
            "revalidated_state_count": len(states),
            "exact_candidate_masks_verified": True,
            "candidate_outcomes_loaded": False,
            "states": states,
        })

    all_masks.sort(key=lambda row: row["state_identity_digest"])
    payload: dict[str, Any] = {
        "schema": PRESERVED_STATE_REVALIDATION_SCHEMA,
        "status": "PASS_POST_ALLOCATION_PRE_OUTCOME_STATE_REVALIDATION",
        "complete": True,
        "source_repository_commit": source_repository_commit,
        "successor_selection_digest": successor_selection_digest,
        "state_selector_amendment_digest": state_selector_amendment_digest(),
        "state_selector_feasibility_receipt_digest":
            state_selector_feasibility_receipt_digest,
        "preserved_state_precontract_revalidation_receipt_digest":
            preserved_state_precontract_revalidation_receipt_digest,
        "predecessor_selection_digest": PREDECESSOR_SELECTION_DIGEST,
        "predecessor_scorer_contract_digest":
            PREDECESSOR_SCORER_CONTRACT_DIGEST,
        "candidate_allocation_manifest_digest":
            allocation_manifest["allocation_manifest_digest"],
        "source_identity_manifest_digest":
            allocation_manifest["source_identity_manifest_digest"],
        "candidate_allocator_contract_digest":
            allocation.allocation_contract_digest(),
        "candidate_allocation_amendment_digest":
            allocation.allocation_amendment_digest(),
        "candidate_allocation_post_identity_validation_digest":
            allocation_manifest["post_identity_pre_outcome_validation"][
                "post_identity_validation_digest"
            ],
        "candidate_outcomes_loaded": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
        "preserved_state_count": len(all_masks),
        "preserved_candidate_assignment_count": sum(
            len(row["candidate_indices"]) for row in all_masks
        ),
        "preserved_candidate_mask_set_digest": _sha256(all_masks),
        "shards": shard_rows,
    }
    payload["preserved_state_revalidation_receipt_digest"] = _sha256(payload)
    validate_preserved_state_revalidation_receipt(
        payload,
        allocation_manifest=allocation_manifest,
        expected_source_commit=source_repository_commit,
        expected_successor_selection_digest=successor_selection_digest,
        expected_feasibility_receipt_digest=
            state_selector_feasibility_receipt_digest,
        expected_precontract_revalidation_receipt_digest=
            preserved_state_precontract_revalidation_receipt_digest,
        root=root,
    )
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
    """Validate phase 2: exact preserved-state masks after all-120 allocation."""

    from lewm.oracle import go2_candidate_allocation_v1_2 as allocation

    if not isinstance(receipt, Mapping):
        raise StateSelectorAmendmentError("final state revalidation must be a mapping")
    payload = dict(receipt)
    observed_digest = payload.pop("preserved_state_revalidation_receipt_digest", None)
    if not _is_digest(observed_digest) or _sha256(payload) != observed_digest:
        raise StateSelectorAmendmentError("final state revalidation self digest failed")
    if (payload.get("schema") != PRESERVED_STATE_REVALIDATION_SCHEMA
            or payload.get("status")
            != "PASS_POST_ALLOCATION_PRE_OUTCOME_STATE_REVALIDATION"
            or payload.get("complete") is not True):
        raise StateSelectorAmendmentError("final state revalidation is not complete/pass")
    if payload.get("state_selector_amendment_digest") != state_selector_amendment_digest():
        raise StateSelectorAmendmentError("final state revalidation amendment mismatch")
    if payload.get("predecessor_selection_digest") != PREDECESSOR_SELECTION_DIGEST:
        raise StateSelectorAmendmentError("final state revalidation predecessor mismatch")
    if (payload.get("predecessor_scorer_contract_digest")
            != PREDECESSOR_SCORER_CONTRACT_DIGEST):
        raise StateSelectorAmendmentError("final state revalidation contract mismatch")
    if (expected_source_commit is not None
            and payload.get("source_repository_commit") != expected_source_commit):
        raise StateSelectorAmendmentError("final state revalidation source mismatch")
    if (expected_successor_selection_digest is not None
            and payload.get("successor_selection_digest")
            != expected_successor_selection_digest):
        raise StateSelectorAmendmentError("final state revalidation selection mismatch")
    if (expected_feasibility_receipt_digest is not None
            and payload.get("state_selector_feasibility_receipt_digest")
            != expected_feasibility_receipt_digest):
        raise StateSelectorAmendmentError("final state revalidation feasibility mismatch")
    if (expected_precontract_revalidation_receipt_digest is not None
            and payload.get("preserved_state_precontract_revalidation_receipt_digest")
            != expected_precontract_revalidation_receipt_digest):
        raise StateSelectorAmendmentError("final state revalidation phase-1 mismatch")

    forbidden = (
        "candidate_outcomes_loaded", "branches_attempted", "frames_rendered",
        "target_latents_encoded", "scorer_training_started",
    )
    if any(payload.get(key) not in (False, 0) for key in forbidden):
        raise StateSelectorAmendmentError(
            "final state revalidation contains forbidden scientific operations"
        )
    allocation.validate_allocation_manifest(allocation_manifest)
    if (payload.get("candidate_allocation_manifest_digest")
            != allocation_manifest["allocation_manifest_digest"]
            or payload.get("source_identity_manifest_digest")
            != allocation_manifest["source_identity_manifest_digest"]
            or payload.get("candidate_allocator_contract_digest")
            != allocation.allocation_contract_digest()
            or payload.get("candidate_allocation_amendment_digest")
            != allocation.allocation_amendment_digest()
            or payload.get("candidate_allocation_post_identity_validation_digest")
            != allocation_manifest["post_identity_pre_outcome_validation"][
                "post_identity_validation_digest"
            ]):
        raise StateSelectorAmendmentError("final state revalidation allocation binding failed")

    assignments = {
        str(row["state_identity_digest"]): row
        for row in allocation_manifest["assignments"]
    }
    predecessor_shards = load_preserved_state_shards(root)
    expected_by_family = {row["family"]: row for row in PRESERVED_STATE_SHARDS}
    rows = payload.get("shards")
    if not isinstance(rows, list) or len(rows) != len(expected_by_family):
        raise StateSelectorAmendmentError("final state revalidation shard coverage failed")
    seen_families: set[str] = set()
    all_masks: list[dict[str, Any]] = []
    total_states = 0
    total_assignments = 0
    for row in rows:
        if not isinstance(row, Mapping):
            raise StateSelectorAmendmentError("final state revalidation row malformed")
        family = str(row.get("family"))
        if family in seen_families or family not in expected_by_family:
            raise StateSelectorAmendmentError("final state revalidation family mismatch")
        seen_families.add(family)
        expected = expected_by_family[family]
        for key in ("path", "state_shard_digest", "raw_sha256", "byte_count"):
            if row.get(key) != expected[key]:
                raise StateSelectorAmendmentError(
                    f"final state revalidation {family} predecessor mismatch"
                )
        states = row.get("states")
        if (not isinstance(states, list) or len(states) != expected["state_count"]
                or row.get("revalidated_state_count") != expected["state_count"]
                or row.get("exact_candidate_masks_verified") is not True
                or row.get("candidate_outcomes_loaded") is not False):
            raise StateSelectorAmendmentError(
                f"final state revalidation {family} state coverage failed"
            )
        expected_ids = {
            str(state["state_identity_digest"]): str(state["state_id"])
            for state in predecessor_shards[family]["states"]
        }
        seen_ids: set[str] = set()
        for state in states:
            if not isinstance(state, Mapping):
                raise StateSelectorAmendmentError("final candidate-mask row malformed")
            state_digest = str(state.get("state_identity_digest"))
            if (state_digest in seen_ids or state_digest not in expected_ids
                    or state.get("state_id") != expected_ids[state_digest]):
                raise StateSelectorAmendmentError(
                    f"final state revalidation {family} identity changed"
                )
            seen_ids.add(state_digest)
            assignment = assignments.get(state_digest)
            indices = state.get("candidate_indices")
            if (assignment is None or assignment.get("family") != family
                    or assignment.get("state_id") != state.get("state_id")
                    or indices != assignment.get("candidate_indices")
                    or state.get("candidate_mask_digest")
                    != candidate_mask_digest(state_digest, indices)):
                raise StateSelectorAmendmentError(
                    f"final state revalidation {family} candidate mask mismatch"
                )
            all_masks.append({
                "state_identity_digest": state_digest,
                "candidate_indices": list(indices),
            })
            total_assignments += len(indices)
        if seen_ids != set(expected_ids):
            raise StateSelectorAmendmentError(
                f"final state revalidation {family} omitted predecessor identity"
            )
        total_states += len(states)
    if (seen_families != set(expected_by_family)
            or total_states != 45
            or total_assignments != 270
            or payload.get("preserved_state_count") != 45
            or payload.get("preserved_candidate_assignment_count") != 270):
        raise StateSelectorAmendmentError("final state revalidation global count failed")
    all_masks.sort(key=lambda row: row["state_identity_digest"])
    if payload.get("preserved_candidate_mask_set_digest") != _sha256(all_masks):
        raise StateSelectorAmendmentError("final state revalidation mask set digest failed")
