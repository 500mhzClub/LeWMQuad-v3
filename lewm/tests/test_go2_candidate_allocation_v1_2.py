"""Focused tests for the pre-outcome scorer-fit candidate allocator."""
from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path

import pytest

from lewm.oracle.go2_candidate_allocation_v1_2 import (
    CANDIDATE_COUNT,
    FORWARD_CANDIDATES,
    REVERSING_CANDIDATE,
    ROTATION_BLOCKS,
    STRATA,
    TURNING_CANDIDATES,
    CandidateAllocationError,
    allocation_amendment_contract,
    allocation_amendment_digest,
    allocation_contract_digest,
    allocation_manifest_digest,
    build_pre_identity_structural_validation,
    build_allocation_manifest,
    candidate_block,
    validate_allocation_amendment_artifact,
    validate_allocation_manifest,
    validate_pre_identity_structural_validation,
)
from lewm.oracle.go2_candidate_allocation_v1_2 import (
    _contingency_tables,
    _post_identity_pre_outcome_validation,
)


SOURCE_DIGEST = "a" * 64
ROOT = Path(__file__).resolve().parents[2]


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _states():
    rows = []
    # Goal type is the snapshot-bound landmark material_id.  Making it a
    # function of stratum gives three nontrivial 40-state exact margins while
    # leaving the fit/calibration and family constraints independently active.
    material = {
        "general": "landmark_red",
        "safety_enriched": "landmark_blue",
        "completion_enriched": "landmark_green",
    }
    for family_index in range(8):
        family = f"family_{family_index}"
        for stratum in STRATA:
            for ordinal in range(5):
                state_id = f"{family}|{stratum}|{ordinal}"
                rows.append({
                    "state_id": state_id,
                    "state_identity_digest": _digest(state_id),
                    "family": family,
                    "stratum": stratum,
                    "split_role": "calibration" if ordinal == 0 else "fit",
                    "goal_type": material[stratum],
                })
    return rows


@pytest.fixture(scope="module")
def allocation():
    return build_allocation_manifest(
        _states(), source_identity_manifest_digest=SOURCE_DIGEST
    )


def test_rotation_family_is_full_period_complementary_and_action_diverse():
    assert len(set(ROTATION_BLOCKS)) == CANDIDATE_COUNT
    universe = set(range(CANDIDATE_COUNT))
    for rotation, block in enumerate(ROTATION_BLOCKS):
        assert set(candidate_block(rotation + 6 if rotation < 6 else rotation - 6)) \
            == universe - set(block)
        assert FORWARD_CANDIDATES.intersection(block)
        assert TURNING_CANDIDATES.intersection(block)
    # The prospective amendment deliberately does not assert reverse in every
    # subset: half of the unchanged subset catalogue contains the sole reverse.
    assert sum(REVERSING_CANDIDATE in block for block in ROTATION_BLOCKS) == 6


def test_amendment_is_narrow_and_preserves_predecessor_allocator_digest():
    assert allocation_contract_digest() == (
        "bb2d9956947be64985f15970dc30f9f0e37cda8012f7c7f5da8808c5d601de5e"
    )
    amendment = allocation_amendment_contract()
    assert amendment["lineage"]["authorizing_failure_commit"] == (
        "6a4d6a66c93d9461bdfb8bf4c2ccb5b882dcdb78"
    )
    assert amendment["scientific_changes"] == {
        "candidate_bank": False,
        "allowed_subsets": False,
        "candidate_margins": False,
        "fit_calibration_split": False,
        "state_strata": False,
        "goal_type_balance": False,
        "allocator_algorithm": False,
        "reverse_coverage_interpretation_only": True,
    }
    artifact = json.loads((ROOT / (
        "docs/lewm_go2_shared_utility_scorer_v1_2_"
        "allocation_amendment_v1_2026-08-11.json"
    )).read_text())
    validate_allocation_amendment_artifact(artifact)
    assert artifact["allocation_amendment_digest"] == allocation_amendment_digest()


def test_pre_identity_structural_table_is_complete_and_goal_type_is_deferred():
    artifact = build_pre_identity_structural_validation()
    validate_pre_identity_structural_validation(artifact)
    assert len(artifact["slots"]) == 120
    assert artifact["global"]["candidate_slot_count"] == 720
    assert artifact["split_role"] == [
        {
            "split_role": "fit", "state_slot_count": 96,
            "candidate_slot_count": 576,
            "required_appearances_per_candidate": 48,
        },
        {
            "split_role": "calibration", "state_slot_count": 24,
            "candidate_slot_count": 144,
            "required_appearances_per_candidate": 12,
        },
    ]
    assert [row["state_slot_count"] for row in artifact["strata"]] == [40] * 3
    assert [row["state_slot_count"] for row in artifact["families"]] == [15] * 8
    catalogue = artifact["allowed_subset_catalogue"]
    assert catalogue["subset_types_with_reverse"] == 6
    assert catalogue["subset_types_without_reverse"] == 6
    pairwise = catalogue[
        "pairwise_cooccurrence_across_allowed_subset_types"
    ]["matrix"]
    assert len(pairwise) == 12 and all(len(row) == 12 for row in pairwise)
    assert all(pairwise[i][j] == pairwise[j][i]
               for i in range(12) for j in range(12))
    assert artifact["goal_type_validation"]["status"] == (
        "NOT_EVALUABLE_BEFORE_STATE_IDENTITIES"
    )
    fixture = artifact["structural_allocator_fixture"]
    assert fixture["is_actual_state_identity_assignment"] is False
    assert len(fixture["assignments"]) == 120
    validation = fixture["validation"]
    assert validation["global"]["candidate_counts"] == [60] * 12
    assert {row["split_role"]: row["counts"]
            for row in validation["split_role"]} == {
                "fit": [48] * 12, "calibration": [12] * 12,
            }
    assert all(row["counts"] == [20] * 12 for row in validation["strata"])
    assert all(sorted(row["counts"]) == [7] * 6 + [8] * 6
               for row in validation["families"])
    assert all(row["contains_forward"] and row["contains_turning"]
               for row in validation["actual_subset_usage"])
    assert validation["reverse_coverage"][
        "observed_distinct_state_subset_count"] == 60
    actual_pairwise = validation["actual_pairwise_cooccurrence"]["matrix"]
    assert len(actual_pairwise) == 12
    assert all(actual_pairwise[i][j] == actual_pairwise[j][i]
               for i in range(12) for j in range(12))
    tampered = copy.deepcopy(artifact)
    tampered["slots"][0]["split_role"] = "fit"
    with pytest.raises(CandidateAllocationError, match="differs"):
        validate_pre_identity_structural_validation(tampered)


def test_allocator_meets_every_exact_and_integer_optimal_margin(allocation):
    validate_allocation_manifest(
        allocation, expected_source_identity_manifest_digest=SOURCE_DIGEST
    )
    tables = allocation["contingency_tables"]
    assert tables["global"] == [60] * 12
    assert {row["split_role"]: row["counts"] for row in tables["split_role"]} == {
        "fit": [48] * 12,
        "calibration": [12] * 12,
    }
    assert all(row["counts"] == [20] * 12 for row in tables["stratum"])
    assert all(sorted(row["counts"]) == [7] * 6 + [8] * 6
               for row in tables["family"])
    assert all(row["counts"] == [20] * 12 for row in tables["goal_type"])
    validation = allocation["post_identity_pre_outcome_validation"]
    assert validation["status"] == (
        "PASS_POST_IDENTITY_PRE_OUTCOME_ALLOCATION_VALIDATION"
    )
    assert validation["reverse_coverage"][
        "observed_distinct_state_subset_count"] == 60
    assert validation["goal_type_validation"]["status"] == (
        "PASS_ACTUAL_POST_IDENTITY_PRE_OUTCOME_GOAL_TYPE_BALANCE"
    )


def test_allocation_is_independent_of_input_order(allocation):
    shuffled = _states()
    random.Random(20260811).shuffle(shuffled)
    repeated = build_allocation_manifest(
        shuffled, source_identity_manifest_digest=SOURCE_DIGEST
    )
    assert repeated == allocation
    assert repeated["allocation_manifest_digest"] == allocation_manifest_digest(repeated)


def test_odd_goal_type_counts_receive_the_unique_integer_optimal_balance():
    states = _states()
    # The red group falls from 40 to 39 states and yellow has one state.  Exact
    # equality is arithmetically impossible for each odd group; six candidates
    # must receive the floor and six the ceiling.
    states[0]["goal_type"] = "landmark_yellow"
    manifest = build_allocation_manifest(
        states, source_identity_manifest_digest=SOURCE_DIGEST
    )
    goal = {
        row["goal_type"]: row for row in manifest["contingency_tables"]["goal_type"]
    }
    assert sorted(goal["landmark_red"]["counts"]) == [19] * 6 + [20] * 6
    assert sorted(goal["landmark_yellow"]["counts"]) == [0] * 6 + [1] * 6


def test_allocator_rejects_any_outcome_bearing_or_unregistered_input_field():
    states = _states()
    states[0]["utility"] = 0.75
    with pytest.raises(CandidateAllocationError, match="extra=.*utility"):
        build_allocation_manifest(states, source_identity_manifest_digest=SOURCE_DIGEST)


def test_validator_rejects_a_rehashed_assignment_tamper(allocation):
    tampered = copy.deepcopy(allocation)
    row = tampered["assignments"][0]
    row["candidate_indices"] = list(candidate_block((row["rotation_index"] + 1) % 12))
    tampered["allocation_manifest_digest"] = allocation_manifest_digest(tampered)
    with pytest.raises(CandidateAllocationError, match="candidate_indices"):
        validate_allocation_manifest(tampered)


def test_validator_rejects_a_balanced_rehashed_noncanonical_allocation(allocation):
    tampered = copy.deepcopy(allocation)
    for row in tampered["assignments"]:
        row["rotation_index"] = (row["rotation_index"] + 6) % 12
        row["candidate_indices"] = list(candidate_block(row["rotation_index"]))
    # Global complementation preserves every hard balance.  Rebuild the
    # evidence table and digest too, so only the canonical-choice check rejects
    # this otherwise internally consistent substitute.
    tampered["contingency_tables"] = _contingency_tables(tampered["assignments"])
    tampered["post_identity_pre_outcome_validation"] = (
        _post_identity_pre_outcome_validation(tampered["assignments"])
    )
    tampered["allocation_manifest_digest"] = allocation_manifest_digest(tampered)
    with pytest.raises(CandidateAllocationError, match="not the canonical"):
        validate_allocation_manifest(tampered)


def test_goal_type_must_be_an_explicit_nonempty_material_identifier():
    states = _states()
    states[0]["goal_type"] = ""
    with pytest.raises(CandidateAllocationError, match="goal_type"):
        build_allocation_manifest(states, source_identity_manifest_digest=SOURCE_DIGEST)
