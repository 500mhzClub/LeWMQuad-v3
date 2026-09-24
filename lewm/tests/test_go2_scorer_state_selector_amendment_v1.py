"""Focused non-Genesis tests for state-selector amendment V1."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lewm.oracle import go2_scorer_state_selector_amendment_v1 as S
from lewm.oracle import go2_scorer_state_selector_amendment_v2 as S2
from lewm.oracle import go2_scorer_contract_v1_2 as C


def _feasibility(*, commit: str = "a" * 40,
                 selection: str = "b" * 64) -> dict:
    rows = []
    for family in S.REQUIRED_FAMILIES:
        rows.append({
            "family": family,
            "all_allowed_scenes_scanned": True,
            "verdict": "PASS",
            "strata": {
                stratum: {
                    "required_distinct_scenes": 5,
                    "eligible_distinct_scenes": 6,
                    "verdict": "PASS",
                }
                for stratum in S.REQUIRED_STRATA
            },
        })
    payload = {
        "schema": S.STATE_SELECTOR_FEASIBILITY_SCHEMA,
        "status": "PASS_OUTCOME_FREE_ALL_SCENE_FEASIBILITY",
        "complete": True,
        "source_repository_commit": commit,
        "successor_selection_digest": selection,
        "state_selector_amendment_digest": S.state_selector_amendment_digest(),
        "family_count": 8,
        "strata": list(S.REQUIRED_STRATA),
        "required_distinct_scenes_per_stratum": 5,
        "selected_state_identities_created": False,
        "candidate_outcomes_loaded": False,
        "branch_identities_created": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
        "families": rows,
    }
    payload["state_selector_feasibility_receipt_digest"] = S._sha256(payload)
    return payload


def _precontract_revalidation(feasibility: dict, *, commit: str = "a" * 40,
                              selection: str = "b" * 64) -> dict:
    rows = []
    all_identities = []
    shards = S.load_preserved_state_shards()
    for expected in S.PRESERVED_STATE_SHARDS:
        source_states = shards[expected["family"]]["states"]
        identities = sorted(state["state_identity_digest"] for state in source_states)
        all_identities.extend(identities)
        rows.append({
            **expected,
            "revalidated_state_count": 15,
            "unchanged_state_identity_count": 15,
            "failed_state_count": 0,
            "exact_redrive_pass": True,
            "amended_classification_pass": True,
            "exclusion_checks_pass": True,
            "goal_binding_unchanged": True,
            "oracle_completion_target_unchanged": True,
            "snapshot_production_designated_goal_claim_unchanged": True,
            "production_task_completion_reset_unchanged": True,
            "completion_state_task_status_all_false": True,
            "candidate_outcomes_loaded": False,
            "state_identity_digests": identities,
            "state_identity_set_digest": S._sha256(sorted(identities)),
            "state_checks": [{
                "state_id": state["state_id"],
                "state_identity_digest": state["state_identity_digest"],
                "exclusion_checks_pass": True,
                "exact_redrive_pass": True,
                "amended_classification_pass": True,
                "goal_binding_unchanged": True,
                "oracle_completion_target_unchanged": True,
                "snapshot_production_designated_goal_claim_unchanged": True,
                "production_task_completion_reset_unchanged": True,
                "completion_state_task_status_all_false": True,
                "failure_reason": None,
            } for state in source_states],
        })
    payload = {
        "schema": S.PRESERVED_STATE_PRECONTRACT_REVALIDATION_SCHEMA,
        "status": "PASS_PRECONTRACT_IDENTITY_REVALIDATION",
        "complete": True,
        "source_repository_commit": commit,
        "successor_selection_digest": selection,
        "state_selector_amendment_digest": S.state_selector_amendment_digest(),
        "state_selector_feasibility_receipt_digest":
            feasibility["state_selector_feasibility_receipt_digest"],
        "predecessor_selection_digest": S.PREDECESSOR_SELECTION_DIGEST,
        "predecessor_scorer_contract_digest":
            S.PREDECESSOR_SCORER_CONTRACT_DIGEST,
        "candidate_outcomes_loaded": False,
        "candidate_allocation_loaded": False,
        "branch_identities_created": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
        "preserved_state_count": 45,
        "state_identity_set_digest": S._sha256(sorted(all_identities)),
        "shards": rows,
        "failure_count": 0,
        "failures": [],
    }
    payload["preserved_state_precontract_revalidation_receipt_digest"] = \
        S._sha256(payload)
    return payload


def test_tracked_amendment_and_authority_chain_are_exact():
    artifact = json.loads((S.ROOT / S.AMENDMENT_ARTIFACT_PATH).read_text())
    S.validate_state_selector_amendment_artifact(artifact)
    S.validate_authority_artifacts()
    assert artifact["state_selector_amendment_digest"] == (
        "69e11a3efe665c4591fa29748b2f13ad08938b92acde763bda10608f93768628"
    )
    assert artifact["superseded_conjunction"]["status"] == (
        "SUPERSEDED_PRE_OUTCOME_GRAPH_DISCRETIZATION_INFEASIBLE"
    )
    assert artifact["preserved"]["completion_geodesic_threshold_m"] == 0.75
    semantics = artifact["preserved"]["unchanged_completion_semantics"]
    assert semantics["not_interchangeable"] is True
    assert set(semantics) == {
        "oracle_v1_2_completion_target",
        "snapshot_production_designated_goal_claim",
        "production_task_completion_and_reset",
        "not_interchangeable",
    }
    assert "candidate branch tick" in semantics[
        "oracle_v1_2_completion_target"
    ]["definition"]
    assert semantics["oracle_v1_2_completion_target"][
        "complete_oracle_v1_2_digest"
    ] == S.ORACLE_V1_2_DIGEST
    assert "range-envelope" in semantics[
        "snapshot_production_designated_goal_claim"
    ]["definition"]
    assert "reset" in semantics[
        "production_task_completion_and_reset"
    ]["definition"]
    assert {
        value["source_binding"]["path"]
        for key, value in semantics.items() if key != "not_interchangeable"
    } == {
        "lewm/oracle/go2_branch_oracle_v1_2.py",
        "lewm_genesis/lewm_genesis/collectors/route_teacher.py",
        "lewm_genesis/lewm_genesis/rollout.py",
    }
    assert "production_completion_predicate" not in artifact["preserved"]


def test_selector_amendment_is_narrow_and_preserves_priority():
    contract = S.state_selector_amendment_contract()
    assert contract["replacement"]["state_selection_priority"] == [
        "general", "safety_enriched", "completion_enriched",
    ]
    assert "graph_hops == 0" in contract["replacement"]["completion"]
    assert contract["replacement"]["completion_requirements"] == {
        "reachable": True,
        "continuous_metric_geodesic_m_max": 0.75,
        "absolute_body_bearing_deg_max": 75.0,
        "snapshot_task_completed": False,
        "snapshot_goal_claimed": False,
        "snapshot_terminated": False,
        "snapshot_truncated": False,
    }
    assert contract["preserved"]["candidate_bank_max_translation_over_horizon_m"] == 0.6
    assert contract["preserved"]["general_and_safety_stratum_semantics"] is True


def test_v1_successor_is_preserved_as_v2_predecessor_without_oracle_change():
    selection = C.CORPUS_SELECTION_CONTRACT
    assert selection["predecessor_selection_digest"] == \
        S2.PREDECESSOR_SUCCESSOR_SELECTION_DIGEST
    assert S2.PREDECESSOR_SUCCESSOR_SELECTION_DIGEST == (
        "8cf65cc016c28ad34f1e50246561e72ee9d0f9c1c253fe8e32a4203a35b73ebe"
    )
    assert S2.PREDECESSOR_AMENDMENT_DIGEST == S.state_selector_amendment_digest()
    assert selection["state_selector_amendment_digest"] == \
        S2.state_selector_amendment_digest()
    assert C._digest(selection) != S2.PREDECESSOR_SUCCESSOR_SELECTION_DIGEST
    assert C.contract()["oracle_v1_2_digest"] == (
        "3ffbe1a87f7975c97e7ff42e50a6a00ca0f47d8840a434d0ff215c303bf6f0e4"
    )
    assert selection["state_selection_priority"] == list(
        S2.SCORER_FIT_SELECTION_PRIORITY
    )
    assert selection["completion_semantic_separation"] == \
        S2.state_selector_amendment_contract()["preserved"][
            "completion_semantic_separation"
        ]
    assert "not the production collector claim" in C.SCORER["heads"][
        "completion"
    ]["target"]
    bound = C.source_bindings()
    assert bound["oracle_v1_2_completion_target_implementation"]["sha256"] == \
        S2.COMPLETION_SEMANTIC_SOURCE_BINDINGS[
            "oracle_v1_2_completion_target"
        ]["sha256"]
    assert bound["production_designated_goal_claim_implementation"]["sha256"] == \
        S2.COMPLETION_SEMANTIC_SOURCE_BINDINGS[
            "snapshot_production_designated_goal_claim"
        ]["sha256"]
    assert bound["production_task_completion_reset_implementation"]["sha256"] == \
        S2.COMPLETION_SEMANTIC_SOURCE_BINDINGS[
            "production_task_completion_and_reset"
        ]["sha256"]


def test_all_family_feasibility_receipt_is_fail_closed():
    receipt = _feasibility()
    S.validate_state_selector_feasibility_receipt(
        receipt,
        expected_source_commit="a" * 40,
        expected_successor_selection_digest="b" * 64,
    )

    tampered = copy.deepcopy(receipt)
    tampered["families"][0]["strata"]["completion_enriched"][
        "eligible_distinct_scenes"
    ] = 4
    tampered["state_selector_feasibility_receipt_digest"] = S._sha256({
        key: value for key, value in tampered.items()
        if key != "state_selector_feasibility_receipt_digest"
    })
    with pytest.raises(S.StateSelectorAmendmentError, match="completion_enriched failed"):
        S.validate_state_selector_feasibility_receipt(tampered)


def test_preserved_45_precontract_revalidation_requires_exact_shards_and_no_outcomes():
    feasibility = _feasibility()
    receipt = _precontract_revalidation(feasibility)
    S.validate_preserved_state_precontract_revalidation_receipt(
        receipt,
        expected_source_commit="a" * 40,
        expected_successor_selection_digest="b" * 64,
        expected_feasibility_receipt_digest=
            feasibility["state_selector_feasibility_receipt_digest"],
    )

    tampered = copy.deepcopy(receipt)
    tampered["candidate_outcomes_loaded"] = True
    tampered["preserved_state_precontract_revalidation_receipt_digest"] = S._sha256({
        key: value for key, value in tampered.items()
        if key != "preserved_state_precontract_revalidation_receipt_digest"
    })
    with pytest.raises(S.StateSelectorAmendmentError, match="forbidden scientific"):
        S.validate_preserved_state_precontract_revalidation_receipt(tampered)


def test_post_allocation_revalidation_binds_exact_preserved_candidate_masks(
    monkeypatch: pytest.MonkeyPatch,
):
    from lewm.oracle import go2_candidate_allocation_v1_2 as allocation

    shards = S.load_preserved_state_shards()
    assignments = []
    for expected in S.PRESERVED_STATE_SHARDS:
        for state in shards[expected["family"]]["states"]:
            assignments.append({
                "state_identity_digest": state["state_identity_digest"],
                "state_id": state["state_id"],
                "family": expected["family"],
                "candidate_indices": [0, 1, 3, 5, 8, 10],
            })
    manifest = {
        "allocation_manifest_digest": "1" * 64,
        "source_identity_manifest_digest": "2" * 64,
        "assignments": assignments,
        "post_identity_pre_outcome_validation": {
            "post_identity_validation_digest": "3" * 64,
        },
    }
    monkeypatch.setattr(allocation, "validate_allocation_manifest", lambda _: None)
    receipt = S.build_preserved_state_revalidation_receipt(
        allocation_manifest=manifest,
        source_repository_commit="a" * 40,
        successor_selection_digest="b" * 64,
        state_selector_feasibility_receipt_digest="c" * 64,
        preserved_state_precontract_revalidation_receipt_digest="d" * 64,
    )
    S.validate_preserved_state_revalidation_receipt(
        receipt,
        allocation_manifest=manifest,
        expected_source_commit="a" * 40,
        expected_successor_selection_digest="b" * 64,
        expected_feasibility_receipt_digest="c" * 64,
        expected_precontract_revalidation_receipt_digest="d" * 64,
    )

    tampered = copy.deepcopy(receipt)
    tampered["shards"][0]["states"][0]["candidate_indices"] = [0, 1, 2, 3, 4, 5]
    tampered["preserved_state_revalidation_receipt_digest"] = S._sha256({
        key: value for key, value in tampered.items()
        if key != "preserved_state_revalidation_receipt_digest"
    })
    with pytest.raises(S.StateSelectorAmendmentError, match="candidate mask mismatch"):
        S.validate_preserved_state_revalidation_receipt(
            tampered, allocation_manifest=manifest,
        )


def test_authority_raw_tamper_is_rejected(tmp_path: Path):
    for relative in (S.FAILURE_RECEIPT_PATH, S.AMENDMENT_ARTIFACT_PATH):
        source = S.ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    failure = tmp_path / S.FAILURE_RECEIPT_PATH
    raw = bytearray(failure.read_bytes())
    raw[-2] ^= 1
    failure.write_bytes(raw)
    with pytest.raises(S.StateSelectorAmendmentError, match="raw binding failed"):
        S.validate_authority_artifacts(tmp_path)


def test_completion_semantic_source_tamper_is_rejected(tmp_path: Path):
    paths = [S.FAILURE_RECEIPT_PATH, S.AMENDMENT_ARTIFACT_PATH]
    paths.extend(
        binding["path"]
        for binding in S.COMPLETION_SEMANTIC_SOURCE_BINDINGS.values()
    )
    for relative in paths:
        source = S.ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    claim_path = tmp_path / S.COMPLETION_SEMANTIC_SOURCE_BINDINGS[
        "snapshot_production_designated_goal_claim"
    ]["path"]
    raw = bytearray(claim_path.read_bytes())
    raw[-1] ^= 1
    claim_path.write_bytes(raw)
    with pytest.raises(
        S.StateSelectorAmendmentError,
        match="snapshot_production_designated_goal_claim source binding failed",
    ):
        S.validate_authority_artifacts(tmp_path)
