"""Focused tests for the permanent invalid scorer-identity exclusion."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lewm.oracle import go2_invalid_scorer_identity_exclusion_v1_2 as X
from lewm.oracle import go2_scorer_contract_v1_2 as C


def _selector_receipts(commit: str, selection: str):
    selector = C.STATE_SELECTOR
    families = [{
        "family": family,
        "all_allowed_scenes_scanned": True,
        "verdict": "PASS",
        "strata": {
            stratum: {
                "required_distinct_scenes": 5,
                "eligible_distinct_scenes": 5,
                "verdict": "PASS",
            }
            for stratum in selector.REQUIRED_STRATA
        },
    } for family in selector.REQUIRED_FAMILIES]
    feasibility = {
        "schema": selector.STATE_SELECTOR_FEASIBILITY_SCHEMA,
        "status": "PASS_OUTCOME_FREE_ALL_SCENE_FEASIBILITY",
        "complete": True,
        "source_repository_commit": commit,
        "successor_selection_digest": selection,
        "state_selector_amendment_digest":
            selector.state_selector_amendment_digest(),
        "family_count": 8,
        "strata": list(selector.REQUIRED_STRATA),
        "required_distinct_scenes_per_stratum": 5,
        "selected_state_identities_created": False,
        "candidate_outcomes_loaded": False,
        "branch_identities_created": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
        "families": families,
    }
    feasibility["state_selector_feasibility_receipt_digest"] = \
        selector._sha256(feasibility)

    shard_rows = []
    all_digests = []
    preserved = selector.load_preserved_state_shards()
    for expected in selector.PRESERVED_STATE_SHARDS:
        source_states = preserved[expected["family"]]["states"]
        digests = sorted(state["state_identity_digest"] for state in source_states)
        all_digests.extend(digests)
        shard_rows.append({
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
            "state_identity_digests": digests,
            "state_identity_set_digest": selector._sha256(sorted(digests)),
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
    revalidation = {
        "schema": selector.PRESERVED_STATE_PRECONTRACT_REVALIDATION_SCHEMA,
        "status": "PASS_PRECONTRACT_IDENTITY_REVALIDATION",
        "complete": True,
        "source_repository_commit": commit,
        "successor_selection_digest": selection,
        "state_selector_amendment_digest":
            selector.state_selector_amendment_digest(),
        "state_selector_feasibility_receipt_digest":
            feasibility["state_selector_feasibility_receipt_digest"],
        "predecessor_selection_digest": selector.PREDECESSOR_SELECTION_DIGEST,
        "predecessor_scorer_contract_digest":
            selector.PREDECESSOR_SCORER_CONTRACT_DIGEST,
        "candidate_outcomes_loaded": False,
        "candidate_allocation_loaded": False,
        "branch_identities_created": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
        "preserved_state_count": 45,
        "state_identity_set_digest": selector._sha256(sorted(all_digests)),
        "shards": shard_rows,
        "failure_count": 0,
        "failures": [],
    }
    revalidation["preserved_state_precontract_revalidation_receipt_digest"] = \
        selector._sha256(revalidation)
    return feasibility, revalidation


def test_exact_three_witnesses_recover_the_bound_identity_namespaces():
    index = X.load_invalid_identity_index()
    binding = index.binding()

    assert X.invalid_identity_exclusion_digest() == (
        "6d644c34b822fb5fb8e30906875047d1677aa730c2db584470cabdbe8bf6abc3"
    )
    assert binding["scene_count"] == 45
    assert binding["scene_ids_digest"] == (
        "5d5c4fef96e5132ad443c4fbd2778ad7d13fb9190328a498ca56490d53e041fe"
    )
    assert len(index.episode_cluster_ids) == 45
    assert len(index.physical_state_keys) == 45
    assert len(index.snapshot_observation_keys) == 45
    assert len(index.registered_branch_keys) == 270


def test_disjointness_is_checked_at_every_recoverable_identity_level():
    index = X.load_invalid_identity_index()
    scene, episode, source_step, candidate = sorted(
        index.registered_branch_keys)[0]
    cluster = f"{scene}/env0/ep{episode}"
    report = X.disjointness_report([{
        "scene_id": scene,
        "episode_cluster_id": cluster,
        "episode_id": episode,
        "source_step": source_step,
        "candidate_index": candidate,
    }], index=index)

    assert report["overlap_counts"] == {
        "scene": 1,
        "episode_cluster": 1,
        "physical_state": 1,
        "snapshot_observation": 1,
        "registered_branch": 1,
    }
    assert report["scene_cluster_state_observation_branch_disjoint"] is False
    with pytest.raises(RuntimeError, match="preserved invalid scorer identities"):
        X.assert_disjoint([{
            "scene_id": scene,
            "episode_cluster_id": cluster,
            "episode_id": episode,
            "source_step": source_step,
            "candidate_indices": [candidate],
        }], label="synthetic manifest", index=index)


def test_scene_rule_excludes_all_descendants_not_only_old_candidate_subset():
    index = X.load_invalid_identity_index()
    scene = sorted(index.scene_ids)[0]
    report = X.disjointness_report([{
        "scene_id": scene,
        "episode_cluster_id": f"{scene}/env0/ep999",
        "episode_id": 999,
        "source_step": 999,
        "candidate_index": 11,
    }], index=index)
    assert report["overlap_counts"]["scene"] == 1
    assert report["scene_cluster_state_observation_branch_disjoint"] is False


def test_witness_byte_tamper_is_rejected_before_identity_use(tmp_path: Path):
    for witness in X.INVALID_SCORER_IDENTITY_EXCLUSION["witnesses"]:
        source = X.ROOT / witness["path"]
        target = tmp_path / witness["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    first = X.INVALID_SCORER_IDENTITY_EXCLUSION["witnesses"][0]
    corrupted = tmp_path / first["path"]
    raw = bytearray(corrupted.read_bytes())
    raw[-2] ^= 1
    corrupted.write_bytes(raw)

    with pytest.raises(RuntimeError, match="witness byte binding failed"):
        X.load_invalid_identity_index(tmp_path)


def test_known_graph_infeasible_contract_is_archived_before_successor_write(
        tmp_path: Path):
    predecessor = C.ROOT / \
        ".generated/go2_utility_scorer_v1_2/scorer_contract_v1_2.json"
    active = tmp_path / "scorer_contract_v1_2.json"
    raw = predecessor.read_bytes()
    active.write_bytes(raw)
    replacement = {
        "schema": "synthetic-current-contract",
        "scorer_contract_v1_2_digest": "f" * 64,
    }
    replacement["contract_artifact_digest"] = C._digest(replacement)

    disposition = C._prepare_contract_output(active, replacement)
    archive = tmp_path / "superseded_pre_run" / (
        "scorer_contract_v1_2."
        f"{C.SUPERSEDED_GRAPH_INFEASIBLE_CONTRACT_ARTIFACT['scorer_contract_v1_2_digest']}.json"
    )
    assert disposition == "superseded_archived"
    assert not active.exists()
    assert archive.read_bytes() == raw


def test_known_0fc7_contract_preservation_remains_supported(tmp_path: Path):
    predecessor = C.ROOT / \
        ".generated/go2_utility_scorer_v1_2/superseded_pre_run/" \
        "scorer_contract_v1_2." \
        "0fc7a3db0ca86ae206050ee6da2894208fa11707e840b112a8a6810e18ac3e21.json"
    active = tmp_path / "scorer_contract_v1_2.json"
    raw = predecessor.read_bytes()
    active.write_bytes(raw)
    replacement = {"schema": "synthetic-successor"}
    replacement["contract_artifact_digest"] = C._digest(replacement)

    assert C._prepare_contract_output(active, replacement) == "superseded_archived"
    archive = tmp_path / "superseded_pre_run" / predecessor.name
    assert not active.exists()
    assert archive.read_bytes() == raw


def test_unknown_contract_artifact_is_never_overwritten(tmp_path: Path):
    active = tmp_path / "scorer_contract_v1_2.json"
    unknown = {"schema": "unknown", "scorer_contract_v1_2_digest": "e" * 64}
    unknown["contract_artifact_digest"] = C._digest(unknown)
    active.write_text(json.dumps(unknown))
    replacement = {"schema": "replacement"}
    replacement["contract_artifact_digest"] = C._digest(replacement)
    with pytest.raises(RuntimeError, match="unknown scorer-contract artifact"):
        C._prepare_contract_output(active, replacement)
    assert active.exists()


def test_clean_source_binding_accepts_only_exact_clean_injected_git_state():
    binding = C._validated_repository_state(
        head="a" * 40,
        status="",
        top_level=str(C.ROOT),
        bindings={"source": {"path": "x", "sha256": "b" * 64}},
    )
    assert binding["source_repository_commit"] == "a" * 40
    assert binding["source_repository_clean"] is True
    assert binding["nonignored_tracked_or_untracked_changes_permitted"] is False

    with pytest.raises(RuntimeError, match="not clean"):
        C._validated_repository_state(
            head="a" * 40,
            status="?? untracked_source.py",
            top_level=str(C.ROOT),
            bindings={},
        )
    with pytest.raises(RuntimeError, match="custody repository root"):
        C._validated_repository_state(
            head="a" * 40,
            status="",
            top_level=str(C.ROOT.parent),
            bindings={},
        )


def test_contract_issue_payload_binds_launch_amendment_and_invalid45_without_encoder():
    source = C._validated_repository_state(
        head="c" * 40,
        status="",
        top_level=str(C.ROOT),
        bindings={"synthetic": {"path": "source.py", "sha256": "d" * 64}},
    )
    selection = C._digest(C.CORPUS_SELECTION_CONTRACT)
    feasibility, revalidation = _selector_receipts("c" * 40, selection)
    payload = C._contract_artifact_payload(source, feasibility, revalidation)
    frozen = payload["contract"]
    assert payload["schema"] == "go2_utility_scorer_contract_v1_2_artifact"
    assert payload["source_repository_commit"] == "c" * 40
    assert payload["source_repository_clean"] is True
    assert payload["clean_source_binding"] == source
    assert payload["candidate_allocation_amendment_verified"] is True
    assert payload["state_selector_amendment_verified"] is True
    assert payload["state_selector_feasibility_verified"] is True
    assert payload["preserved_state_precontract_revalidation_verified"] is True
    assert payload["state_selector_feasibility_receipt_digest"] == \
        feasibility["state_selector_feasibility_receipt_digest"]
    assert payload["preserved_state_precontract_revalidation_receipt_digest"] == \
        revalidation["preserved_state_precontract_revalidation_receipt_digest"]
    assert payload["preserved_state_post_allocation_revalidation"] == {
        "status": "PENDING_POST_IDENTITY_PRE_OUTCOME",
        "required_before_active_identity_manifest": True,
        "schema": C.STATE_SELECTOR.PRESERVED_STATE_REVALIDATION_SCHEMA,
        "path": C.STATE_SELECTOR.PRESERVED_STATE_REVALIDATION_RECEIPT_PATH,
        "realized_receipt_digest_bound_at_contract_issue": False,
    }
    assert payload["invalid_scorer_identity_exclusion_verified"] is True
    assert frozen["candidate_allocation_amendment_digest"] == \
        C.ALLOC.allocation_amendment_digest()
    assert frozen["state_selector_amendment_digest"] == \
        C.STATE_SELECTOR.state_selector_amendment_digest()
    assert frozen["invalid_scorer_identity_exclusion_digest"] == \
        X.invalid_identity_exclusion_digest()
    assert C.render_contract_digest() == (
        "2faa22e3b10a2c4199bdabdbc0ed0e1ff9c7c4ac48bb489daeb0fd70d5b65c17"
    )
    assert frozen["render_contract"]["runtime_wrapper_contract_digest"] == (
        "df70a0c16ad421ae93a93c4d9dda0fd4d6f154f42d9710c7fc2f0242c3e8cb1b"
    )
    assert C.preprocess_contract_digest() == (
        "2688ca405ed7e8bb86e82f1d111b7b865466f4d497b973a04a52af846b5da6a9"
    )
    assert C.target_encoder_digest() == (
        "15ff78a0205ba138a740f12f6eb9bb3f78bce9c5ba8c2849f7e83489a6b2b6a5"
    )
    assert frozen["oracle_v1_2_digest"] == (
        "3ffbe1a87f7975c97e7ff42e50a6a00ca0f47d8840a434d0ff215c303bf6f0e4"
    )
    assert payload["contract_artifact_digest"] == C._digest({
        key: value for key, value in payload.items()
        if key != "contract_artifact_digest"
    })
