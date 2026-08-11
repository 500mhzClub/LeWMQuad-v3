"""Outcome-free tests for the scorer-fit state-selector successor."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from scripts import build_go2_branch_corpus_v1_2 as B


def _status(**overrides):
    status = {
        "task_completed": False,
        "goal_claimed": False,
        "terminated": False,
        "truncated": False,
        "termination_flags": {
            "fall": False, "out_of_bounds": False, "tipped": False,
            "nan": False,
        },
    }
    status.update(overrides)
    return status


def _eligibility(*, hops=0, distance=0.5, bearing=0.0, status=None):
    return B.completion_enriched_eligibility(
        graph_hops=hops,
        reachable=True,
        continuous_geodesic_m=distance,
        bearing_body_rad=bearing,
        task_status=_status() if status is None else status,
    )


def _scene_row(family: str, scene_index: int, *, hops: int = 0):
    return {
        "family": family,
        "scene_id": f"{family}-{scene_index:03d}",
        "stratum": "completion_enriched",
        "first_eligible_block": 40 + scene_index,
        "continuous_geodesic_m": 0.5,
        "abs_bearing_rad": 0.25,
        "graph_hops_diagnostic": hops,
        "body_clearance_m": 0.2,
        # This value is a tripwire: the reducer is deliberately unable to
        # consume candidate or branch outcomes.
        "branch_outcome": object(),
    }


def test_hops_zero_not_completed_meeting_continuous_contract_is_eligible():
    result = _eligibility(hops=0, distance=0.5, bearing=0.0)
    assert result["eligible"] is True
    assert result["graph_hops_diagnostic"] == 0
    assert result["rejection_reasons"] == []


@pytest.mark.parametrize(
    ("flag", "reason"),
    (
        ("task_completed", "completion_snapshot_task_completed"),
        ("goal_claimed", "completion_snapshot_goal_claimed"),
        ("terminated", "completion_snapshot_terminated"),
        ("truncated", "completion_snapshot_truncated"),
    ),
)
def test_completed_claimed_terminated_or_truncated_snapshot_is_rejected(flag, reason):
    result = _eligibility(status=_status(**{flag: True}))
    assert result["eligible"] is False
    assert reason in result["rejection_reasons"]


def test_missing_snapshot_task_status_is_rejected_fail_closed():
    result = _eligibility(status={})
    assert result["eligible"] is False
    assert set(result["rejection_reasons"]) == {
        "completion_snapshot_task_completed_unavailable",
        "completion_snapshot_goal_claimed_unavailable",
        "completion_snapshot_terminated_unavailable",
        "completion_snapshot_truncated_unavailable",
    }


def test_snapshot_task_status_matches_production_route_completion(monkeypatch):
    class Policy:
        revisit_after_arrival = False

        @staticmethod
        def visited_landmark_cells(env_idx):
            assert env_idx == 0
            return frozenset({3, 5})

    runner = SimpleNamespace(
        _scheduler=SimpleNamespace(policy_for=lambda env_idx: Policy()),
        _blocks_in_episode=[40],
        _scene_graph=object(),
        _landmark_cell_to_id={3: "a", 5: "b"},
    )
    ctx = SimpleNamespace(runner=runner)
    monkeypatch.setattr(
        B.V1, "_termination_flags",
        lambda _ctx: {"fall": False, "nan": False})
    status = B._snapshot_task_status(ctx, 5)
    assert status["goal_claimed"] is True
    assert status["task_completed"] is True
    assert status["terminated"] is False
    assert status["truncated"] is False
    assert B._snapshot_claim_semantics_unchanged(status) is True
    assert B._production_task_reset_semantics_unchanged(status) is True

    changed_claim = dict(status)
    changed_claim["goal_claimed"] = False
    assert B._snapshot_claim_semantics_unchanged(changed_claim) is False
    changed_reset = dict(status)
    changed_reset["task_completed"] = False
    assert B._production_task_reset_semantics_unchanged(changed_reset) is False


@pytest.mark.parametrize(
    ("blocks", "revisit", "visited", "expected_completed"),
    (
        (0, False, {3, 5}, False),
        (40, True, {3, 5}, False),
        (40, False, {3}, False),
        (40, False, {3, 5}, True),
    ),
)
def test_task_reset_check_is_distinct_from_designated_goal_claim(
        monkeypatch, blocks, revisit, visited, expected_completed):
    class Policy:
        revisit_after_arrival = revisit

        @staticmethod
        def visited_landmark_cells(_env_idx):
            return frozenset(visited)

    runner = SimpleNamespace(
        _scheduler=SimpleNamespace(policy_for=lambda _env_idx: Policy()),
        _blocks_in_episode=[blocks],
        _scene_graph=object(),
        _landmark_cell_to_id={3: "a", 5: "b"},
    )
    ctx = SimpleNamespace(runner=runner)
    monkeypatch.setattr(B.V1, "_termination_flags", lambda _ctx: {"nan": False})
    status = B._snapshot_task_status(ctx, 3)
    assert status["goal_claimed"] is (3 in visited)
    assert status["task_completed"] is expected_completed
    assert B._snapshot_claim_semantics_unchanged(status) is True
    assert B._production_task_reset_semantics_unchanged(status) is True


def test_oracle_completion_target_binding_is_not_a_snapshot_task_flag():
    assert B._oracle_completion_target_unchanged() is True
    assert B.v12_oracle_digest() == B.STATE_SELECTOR.ORACLE_V1_2_DIGEST


@pytest.mark.parametrize(
    ("hops", "distance", "bearing", "reason"),
    (
        (0, 0.750001, 0.0, "completion_geodesic_gt_0_75m"),
        (4, 0.750001, 0.0, "completion_geodesic_gt_0_75m"),
        (0, 0.5, math.radians(75.0) + 1e-9, "completion_bearing_gt_75deg"),
        (4, 0.5, math.radians(75.0) + 1e-9, "completion_bearing_gt_75deg"),
    ),
)
def test_unchanged_continuous_conditions_reject_regardless_of_hops(
        hops, distance, bearing, reason):
    result = _eligibility(hops=hops, distance=distance, bearing=bearing)
    assert result["eligible"] is False
    assert reason in result["rejection_reasons"]


def test_valid_positive_hops_completion_remains_eligible():
    result = _eligibility(hops=3, distance=0.6, bearing=math.radians(30.0))
    assert result["eligible"] is True
    assert result["graph_hops_diagnostic"] == 3


def test_exact_0_75_threshold_is_preserved_without_tolerance():
    assert B.COMPLETION_ENRICHED_MAX_GEODESIC_M == 0.75
    assert _eligibility(distance=0.75)["eligible"] is True
    assert _eligibility(distance=math.nextafter(0.75, math.inf))["eligible"] is False


def test_eligibility_and_dry_run_reducer_have_no_branch_outcome_read(monkeypatch):
    monkeypatch.setattr(
        B, "_outcome_generation_started",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run touched outcome plumbing")))
    result = _eligibility()
    summary = B.build_selector_feasibility_summary(
        family="rough_local_dynamics", allowed_scene_count=1,
        requested_strata=("completion_enriched",),
        scene_evidence=[_scene_row("rough_local_dynamics", 0)],
        rejection_counts={})
    assert result["eligible"] is True
    assert summary["per_stratum"]["completion_enriched"][
        "eligible_distinct_scenes"] == 1


def test_dry_run_can_prove_five_completion_scenes_for_rough_and_open():
    for family in ("rough_local_dynamics", "open_obstacle_field"):
        summary = B.build_selector_feasibility_summary(
            family=family, allowed_scene_count=9,
            requested_strata=("completion_enriched",),
            scene_evidence=[_scene_row(family, index) for index in range(5)],
            rejection_counts={"completion_snapshot_goal_claimed": 2})
        completion = summary["per_stratum"]["completion_enriched"]
        assert completion["required_distinct_scenes"] == 5
        assert completion["eligible_distinct_scenes"] == 5
        assert completion["quota_pass"] is True
        assert summary["all_requested_quotas_pass"] is True


def test_general_and_safety_hop_requirements_remain_frozen():
    contract = B.STATE_SELECTOR.state_selector_amendment_contract()
    assert contract["replacement"]["general_and_safety"].endswith(
        "general requires graph_hops >= 2 and safety remains a subset")
    assert tuple(contract["replacement"]["state_selection_priority"]) == B.STRATA


def test_completed_failed_feasibility_gate_is_retained_without_rerun(tmp_path):
    families = []
    for family_index, family in enumerate(B.STATE_SELECTOR.REQUIRED_FAMILIES):
        strata = {}
        for stratum in B.STRATA:
            eligible = (4 if family_index == 0
                        and stratum == "completion_enriched" else 5)
            strata[stratum] = {
                "required_distinct_scenes": 5,
                "eligible_distinct_scenes": eligible,
                "verdict": "PASS" if eligible >= 5 else "FAIL",
            }
        families.append({
            "family": family,
            "all_allowed_scenes_scanned": True,
            "verdict": "FAIL" if family_index == 0 else "PASS",
            "strata": strata,
        })
    receipt = {
        "schema": B.SELECTOR_FEASIBILITY_SCHEMA,
        "status": "FAIL_OUTCOME_FREE_SELECTOR_FEASIBILITY",
        "complete": True,
        "binding_receipt": True,
        "source_repository_commit": "a" * 40,
        "successor_selection_digest": "b" * 64,
        "state_selector_amendment_digest":
            B.STATE_SELECTOR.state_selector_amendment_digest(),
        "family_count": 8,
        "strata": list(B.STRATA),
        "required_distinct_scenes_per_stratum": 5,
        "families": families,
        "selected_state_identities_created": False,
        "candidate_outcomes_loaded": False,
        "branch_identities_created": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
    }
    receipt["state_selector_feasibility_receipt_digest"] = \
        B.canonical_digest(receipt)
    path = tmp_path / B.SELECTOR_FEASIBILITY_RECEIPT_NAME
    B.atomic_json(path, receipt)
    raw = path.read_bytes()
    loaded = B._load_completed_selector_feasibility(
        path, source_commit="a" * 40,
        successor_selection_digest="b" * 64)
    assert loaded == receipt
    assert path.read_bytes() == raw


def test_preserved_state_identity_is_exact_and_cannot_be_rewrapped():
    state = next(iter(B._preserved_states_by_digest().values()))
    assert B._state_identity_matches_active_or_preserved(dict(state)) is True
    changed = dict(state)
    changed["warmup_blocks"] = int(changed["warmup_blocks"]) + 1
    assert B._state_identity_matches_active_or_preserved(changed) is False
