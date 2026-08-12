"""Pure tests for the final scorer-fit horizon-reachability amendment."""
from __future__ import annotations

import inspect
import json
import math
from collections.abc import Mapping
from pathlib import Path

import pytest

from lewm.oracle import go2_candidate_allocation_v1_2 as ALLOCATION
from lewm.oracle import go2_scorer_contract_v1_2 as CONTRACT
from lewm.oracle import go2_scorer_state_selector_amendment_v2 as S


FROZEN_BLOCK = ALLOCATION.ROTATION_BLOCKS[0]


def _status(**overrides):
    row = {
        "task_completed": False,
        "goal_claimed": False,
        "terminated": False,
        "truncated": False,
    }
    row.update(overrides)
    return row


def _eligibility(
    *,
    distance=1.0,
    bearing=0.0,
    reachable=True,
    status=None,
    previous=(0.0, 0.0, 0.0),
    candidate_indices=FROZEN_BLOCK,
):
    return S.completion_enriched_eligibility(
        graph_hops=0,
        reachable=reachable,
        continuous_geodesic_m=distance,
        bearing_body_rad=bearing,
        task_status=_status() if status is None else status,
        candidate_indices=candidate_indices,
        previous_applied_command=previous,
    )


def test_outside_radius_is_eligible_when_gap_is_within_exact_l_max():
    budget = S.max_deterministic_translational_path_length_m(
        FROZEN_BLOCK, (0.0, 0.0, 0.0)
    )
    distance = S.COMPLETION_RADIUS_M + budget["l_max_m"]
    evidence = _eligibility(distance=distance)
    assert evidence["continuous_geodesic_m"] > S.COMPLETION_RADIUS_M
    assert evidence["continuous_geodesic_gap_m"] == budget["l_max_m"]
    assert evidence["eligible"] is True


def test_same_state_is_ineligible_when_gap_exceeds_exact_l_max():
    budget = S.max_deterministic_translational_path_length_m(
        FROZEN_BLOCK, (0.0, 0.0, 0.0)
    )
    distance = math.nextafter(
        S.COMPLETION_RADIUS_M + budget["l_max_m"], math.inf
    )
    evidence = _eligibility(distance=distance)
    assert evidence["eligible"] is False
    assert (
        "completion_geodesic_gap_gt_allocated_subset_l_max"
        in evidence["rejection_reasons"]
    )


def test_unclaimed_state_inside_radius_remains_eligible():
    evidence = _eligibility(distance=math.nextafter(0.75, -math.inf))
    assert evidence["continuous_geodesic_gap_m"] == 0.0
    assert evidence["eligible"] is True


@pytest.mark.parametrize(
    "flag", ("task_completed", "goal_claimed", "terminated", "truncated")
)
def test_completed_claimed_or_inactive_state_is_rejected_regardless_of_distance(flag):
    evidence = _eligibility(distance=0.1, status=_status(**{flag: True}))
    assert evidence["eligible"] is False
    assert f"completion_snapshot_{flag}" in evidence["rejection_reasons"]


def test_unreachable_goal_is_rejected():
    evidence = _eligibility(distance=0.1, reachable=False)
    assert evidence["eligible"] is False
    assert "completion_unreachable" in evidence["rejection_reasons"]


def test_bearing_requirement_remains_exactly_75_degrees():
    assert S.COMPLETION_MAX_ABS_BEARING_DEG == 75.0
    assert _eligibility(
        distance=0.1, bearing=math.radians(75.0)
    )["eligible"] is True
    rejected = _eligibility(
        distance=0.1,
        bearing=math.nextafter(math.radians(75.0), math.inf),
    )
    assert rejected["eligible"] is False
    assert "completion_bearing_gt_75deg" in rejected["rejection_reasons"]


def test_l_max_uses_exact_post_slew_plans_from_actual_previous_command():
    standing = S.candidate_post_slew_plan(0, (0.0, 0.0, 0.0))
    reversing = S.candidate_post_slew_plan(0, (-0.2, 0.0, 0.0))
    assert len(standing) == len(reversing) == 20
    # forward_fast is limited from 0.0 to +0.25 at the first tick, but from an
    # actual previous -0.20 command it first reaches only +0.05.
    assert standing[0] == (0.25, 0.0, 0.0)
    assert reversing[0] == pytest.approx((0.05, 0.0, 0.0))
    standing_budget = S.max_deterministic_translational_path_length_m(
        FROZEN_BLOCK, (0.0, 0.0, 0.0)
    )
    reversing_budget = S.max_deterministic_translational_path_length_m(
        FROZEN_BLOCK, (-0.2, 0.0, 0.0)
    )
    assert standing_budget["l_max_m"] == pytest.approx(0.595)
    assert reversing_budget["l_max_m"] == pytest.approx(0.575)
    assert standing_budget["previous_applied_command"] == [0.0, 0.0, 0.0]
    assert reversing_budget["previous_applied_command"] == [-0.2, 0.0, 0.0]


class _OutcomeTripwireStatus(Mapping):
    """Mapping that fails if the selector iterates toward an outcome field."""

    def __getitem__(self, key):
        if key == "branch_outcome":
            raise AssertionError("selector read a branch outcome")
        if key in _status():
            return False
        raise KeyError(key)

    def __iter__(self):
        raise AssertionError("selector iterated over outcome-bearing mapping")

    def __len__(self):
        raise AssertionError("selector inspected outcome-bearing mapping length")


def test_no_branch_outcome_is_read_or_exposed_by_eligibility():
    evidence = _eligibility(distance=0.1, status=_OutcomeTripwireStatus())
    assert evidence["eligible"] is True
    S.validate_no_outcome_surface()
    parameters = inspect.signature(S.completion_enriched_eligibility).parameters
    assert not {
        "branch", "outcome", "collision", "progress", "completion_label",
        "future_frame", "prediction", "latent",
    }.intersection(parameters)


def test_candidate_bank_and_allocation_are_unchanged():
    assert S.candidate_bank_contract_digest() == S.CANDIDATE_BANK_DIGEST
    assert S.CANDIDATE_BANK_DIGEST == ALLOCATION.CANDIDATE_BANK_DIGEST
    assert (
        ALLOCATION.allocation_amendment_digest()
        == S.ALLOCATION_AMENDMENT_DIGEST
        == "4dde3562cdd9e503d6e264a5d4982a189a9f43d338c3d6b87ee20de352bc3cbc"
    )
    assert len(ALLOCATION.ROTATION_BLOCKS) == 12
    for rotation, block in enumerate(ALLOCATION.ROTATION_BLOCKS):
        assert S.candidate_rotation_index(block) == rotation
        assert len(block) == 6


def test_actual_selector_parameter_remains_exactly_point_75_without_tolerance():
    assert S.COMPLETION_RADIUS_M == 0.75
    assert S.COMPLETION_MAX_GEODESIC_M == 0.75
    assert S.completion_distance_gap_m(0.75) == 0.0
    assert S.completion_distance_gap_m(math.nextafter(0.75, math.inf)) > 0.0
    contract = S.state_selector_amendment_contract()
    separation = contract["preserved"]["completion_semantic_separation"]
    assert separation["not_interchangeable"] is True
    assert "graph cell" in separation["oracle_v1_2_label"]
    assert "range-envelope" in separation["snapshot_production_goal_claim"]
    assert "selector-only" in separation["r_complete_0_75m"]


def test_preidentity_vector_covers_all_rotations_but_is_not_an_assignment():
    vector = S.completion_rotation_eligibility_vector(
        graph_hops=0,
        reachable=True,
        continuous_geodesic_m=1.2,
        bearing_body_rad=0.0,
        task_status=_status(),
        previous_applied_command=(0.0, 0.0, 0.0),
    )
    assert vector["rotation_count"] == 12
    assert vector["is_candidate_assignment"] is False
    assert vector["pre_identity_fixture_used_as_assignment"] is False
    assert {row["candidate_rotation_index"] for row in vector["rotations"]} == set(
        range(12)
    )
    # The exact L_max is not invariant, proving why an arbitrary fixture mask
    # cannot satisfy the state-specific contract.
    assert len({row["l_max_m"] for row in vector["rotations"]}) > 1


def test_allocated_evidence_validator_recomputes_mask_and_fails_closed():
    evidence = _eligibility(distance=1.0)
    S.validate_allocated_completion_evidence(
        evidence,
        candidate_indices=FROZEN_BLOCK,
        previous_applied_command=(0.0, 0.0, 0.0),
    )
    tampered = dict(evidence)
    tampered["l_max_m"] += 0.01
    with pytest.raises(
        S.StateSelectorAmendmentError,
        match="exact allocated-subset arithmetic",
    ):
        S.validate_allocated_completion_evidence(
            tampered,
            candidate_indices=FROZEN_BLOCK,
            previous_applied_command=(0.0, 0.0, 0.0),
        )


def test_contract_binds_final_amendment_and_deterministic_circularity_resolution():
    contract = S.state_selector_amendment_contract()
    assert contract["superseded_start_distance_rule"]["status"] == (
        "SUPERSEDED_PRE_OUTCOME_START_RADIUS_NOT_HORIZON_REACHABILITY"
    )
    assert contract["freeze_policy"][
        "this_is_final_permitted_pre_outcome_selector_amendment"
    ] is True
    search = contract["allocation_circularity_resolution"][
        "deterministic_search"
    ]
    assert "lexicographic" in search["combination_order"]
    assert "unchanged canonical allocator" in search["per_combination_operation"]
    assert search["candidate_outcomes_consumed"] is False
    assert contract["census_reuse"][
        "actual_allocated_mask_check_required_before_manifest"
    ] is True
    assert contract["census_reuse"]["actual_allocated_mask_check_status"] == (
        "MANDATORY_DEFERRED_TO_JOINT_SEARCH_AND_PHASE2"
    )
    assert contract["lineage"]["frozen_failed_census_receipt"][
        "state_selector_feasibility_receipt_digest"
    ] == "2310c3d1b138b605fda483b39cbd4775479cbcc502a4e3707e7a8670457f54d7"
    assert contract["preserved"][
        "oracle_v1_2_completion_at_or_before_horizon"
    ] is True
    assert "actual_completion_at_or_before_horizon" not in contract["preserved"]
    assert contract["source_bindings"]["platform_command_envelope"][
        "sha256"
    ] == "5ac4a08b17cfaa3552f3c3ccd45930b8a929ac5ca31eb1f9440923f037c78189"


@pytest.mark.parametrize("previous", [
    [0.300001, 0.0, 0.0], [-0.300001, 0.0, 0.0],
    [0.0, 0.0, 0.500001], [0.0, 0.0, -0.500001],
])
def test_previous_applied_command_must_fit_frozen_platform_envelope(previous):
    with pytest.raises(
        S.StateSelectorAmendmentError, match="frozen platform envelope"
    ):
        S.max_deterministic_translational_path_length_m(
            list(S.ALLOCATION.ROTATION_BLOCKS[0]), previous)


def test_new_receipt_paths_cannot_overwrite_accepted_v1_failures():
    assert S.STATE_SELECTOR_FEASIBILITY_RECEIPT_NAME != (
        "state_selector_feasibility_receipt.json"
    )
    assert "reachability" in S.STATE_SELECTOR_FEASIBILITY_RECEIPT_NAME
    assert "reachability" in S.PRESERVED_STATE_REVALIDATION_RECEIPT_NAME
    assert tuple(S.ACTIVE_SELECTOR_BINDING_KEYS) == (
        "state_selector_amendment_digest",
        "state_selector_feasibility_receipt_digest",
        "preserved_state_revalidation_receipt_digest",
    )


def test_tracked_amendment_artifact_and_authority_chain_are_exact():
    artifact = json.loads(Path(S.AMENDMENT_ARTIFACT_PATH).read_text())
    S.validate_state_selector_amendment_artifact(artifact)
    assert artifact["state_selector_amendment_digest"] == (
        S.state_selector_amendment_digest()
    )
    S.validate_authority_artifacts()


def test_successor_scorer_contract_binds_v2_and_immediate_predecessor():
    selection = CONTRACT.CORPUS_SELECTION_CONTRACT
    assert selection["predecessor_selection_digest"] == (
        S.PREDECESSOR_SUCCESSOR_SELECTION_DIGEST
    )
    assert selection["state_selector_amendment_digest"] == (
        S.state_selector_amendment_digest()
    )
    assert "first lexicographically feasible" in selection["scorer_fit"]
    assert selection["preserved_state_revalidation_receipt"][
        "expected_completion_enriched_state_count"
    ] == 40
    bindings = CONTRACT.source_bindings()
    assert bindings["state_selector_amendment_implementation"]["path"].endswith(
        "go2_scorer_state_selector_amendment_v2.py"
    )
    assert bindings["qualified_development_transfer_consumer"]["path"] == (
        "scripts/apply_go2_utility_scorer_to_counterfactual_development_v1_2.py"
    )
