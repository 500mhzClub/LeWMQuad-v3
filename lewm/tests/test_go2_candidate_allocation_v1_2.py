"""Focused tests for the pre-outcome scorer-fit candidate allocator."""
from __future__ import annotations

import copy
import hashlib
import random

import pytest

from lewm.oracle.go2_candidate_allocation_v1_2 import (
    CANDIDATE_COUNT,
    FORWARD_CANDIDATES,
    ROTATION_BLOCKS,
    STRATA,
    TURNING_CANDIDATES,
    CandidateAllocationError,
    allocation_manifest_digest,
    build_allocation_manifest,
    candidate_block,
    validate_allocation_manifest,
)
from lewm.oracle.go2_candidate_allocation_v1_2 import _contingency_tables


SOURCE_DIGEST = "a" * 64


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
    tampered["allocation_manifest_digest"] = allocation_manifest_digest(tampered)
    with pytest.raises(CandidateAllocationError, match="not the canonical"):
        validate_allocation_manifest(tampered)


def test_goal_type_must_be_an_explicit_nonempty_material_identifier():
    states = _states()
    states[0]["goal_type"] = ""
    with pytest.raises(CandidateAllocationError, match="goal_type"):
        build_allocation_manifest(states, source_identity_manifest_digest=SOURCE_DIGEST)
