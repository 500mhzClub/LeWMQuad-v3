"""Pure tests for final selector cached-census reuse and scoped redrive."""
from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import build_go2_branch_corpus_v1_2 as B


def _source():
    return {
        "source_repository_commit": "a" * 40,
        "source_repository_clean": True,
        "bound_implementations_digest": "b" * 64,
    }


def _full_snapshot_status(goal_cell: int) -> dict:
    return {
        "task_completed": False,
        "goal_claimed": False,
        "terminated": False,
        "truncated": False,
        "production_claim_evidence": {
            "active_collector_visited_accessor_callable": True,
            "active_collector_claimed_cells": [],
            "designated_goal_cell": goal_cell,
        },
        "production_task_completion_reset_evidence": {
            "minimum_block_guard_pass": True,
            "scene_graph_available": True,
            "active_collector_route_like": True,
            "active_collector_non_revisit": True,
            "scene_landmark_cells_nonempty": True,
            "all_scene_landmark_cells_claimed": False,
        },
        "termination_flags": {
            "fall": False,
            "out_of_bounds": False,
            "tipped": False,
            "nan": False,
        },
    }


def _synthetic_completion_state_capture() -> tuple[dict, dict]:
    family = "medium_enclosed_maze"
    request = {
        "state_resolution_scene_request_digest": "9" * 64,
        "pool": "scorer_fit",
        "family": family,
        "scene": {
            "scene_id": "scene-completion",
            "scene_dir": "/synthetic/scene-completion",
            "scene_manifest_sha256": "8" * 64,
            "scene_manifest_byte_count": 123,
            "split": "synthetic",
            "drive_seed": 17,
        },
        "requested_strata_in_priority_order": ["completion_enriched"],
        "found_before_scene": {"completion_enriched": 0},
    }
    status = _full_snapshot_status(goal_cell=7)
    vector = B.STATE_SELECTOR.completion_rotation_eligibility_vector(
        graph_hops=1, reachable=True, continuous_geodesic_m=0.8,
        bearing_body_rad=0.0, task_status=status,
        previous_applied_command=[0.0, 0.0, 0.0])
    first = vector["rotations"][0]
    chosen = {
        "state_id": f"scorer_fit-{family}-completion_enriched-00",
        "family": family,
        "scene_id": request["scene"]["scene_id"],
        "scene_dir": request["scene"]["scene_dir"],
        "scene_manifest_sha256": request["scene"]["scene_manifest_sha256"],
        "scene_manifest_byte_count": request["scene"][
            "scene_manifest_byte_count"],
        "split": request["scene"]["split"],
        "drive_seed": request["scene"]["drive_seed"],
        "stratum": "completion_enriched",
        "split_role": "calibration",
        "warmup_blocks": B.WARMUP_BLOCKS_MIN,
        "source_step": 200,
        "episode_id": 1,
        "episode_cluster_id": "scene-completion/env0/ep1",
        "cell_id": 3,
        "boundary": {"source_step": 200},
        "goal": {
            "landmark_id": "goal-7",
            "landmark_cell": 7,
            "material_id": "landmark_red",
            "graph_edges": first["graph_hops_diagnostic"],
            "start_geodesic_m": first["continuous_geodesic_m"],
            "bearing_body_rad": first["bearing_body_rad"],
            "range_m": first["continuous_geodesic_m"],
            "landmark_xy_m": [0.0, 0.0],
        },
        "goal_type": "landmark_red",
        "body_clearance_m": 0.2,
        "clearance_m": 0.3,
        "completion_rotation_eligibility_vector": vector,
        "snapshot_task_status": status,
        "previous_applied_command": first["previous_applied_command"],
    }
    chosen["state_identity_digest"] = B._state_identity_digest(chosen)
    capture = B._build_state_resolution_scene_capture(
        request=request, chosen_state=chosen, rejection_reasons={},
        worker_failure=None, blocks_driven=B.WARMUP_BLOCKS_MIN,
        attempt_trace=[{
            "block_index": B.WARMUP_BLOCKS_MIN,
            "attempts": [{
                "stratum": "completion_enriched",
                "verdict": "SELECT",
                "reason_key": None,
            }],
        }])
    return request, capture


def test_ordinary_completion_capture_accepts_full_production_snapshot_status():
    request, capture = _synthetic_completion_state_capture()
    assert capture["chosen_state"]["snapshot_task_status"] != \
        capture["chosen_state"][
            "completion_rotation_eligibility_vector"]["rotations"][0][
                "task_status"]
    B._validate_state_resolution_scene_capture(
        capture, expected_request=request)


def test_historical_capture_identity_replays_under_bound_contract_lineage():
    request, capture = _synthetic_completion_state_capture()
    lineage = {
        "selection_digest":
            B.PERFORMANCE_INTERRUPTION.INTERRUPTED_SELECTION_DIGEST,
        "scorer_contract_v1_2_digest":
            B.PERFORMANCE_INTERRUPTION.INTERRUPTED_SCORER_CONTRACT_DIGEST,
    }
    changed = copy.deepcopy(capture)
    changed["chosen_state"]["state_identity_digest"] = \
        B._state_identity_digest_for_bindings(
            changed["chosen_state"], lineage)
    changed["state_resolution_scene_capture_digest"] = B.canonical_digest({
        key: value for key, value in changed.items()
        if key != "state_resolution_scene_capture_digest"
    })
    with pytest.raises(RuntimeError, match="chosen identity changed"):
        B._validate_state_resolution_scene_capture(
            changed, expected_request=request)
    B._validate_state_resolution_scene_capture(
        changed, expected_request=request,
        expected_state_identity_bindings=lineage)


@pytest.mark.parametrize("surface", ("selector_flag", "claim", "reset"))
def test_ordinary_completion_capture_rejects_status_evidence_tamper(surface):
    request, capture = _synthetic_completion_state_capture()
    changed = copy.deepcopy(capture)
    status = changed["chosen_state"]["snapshot_task_status"]
    if surface == "selector_flag":
        status["truncated"] = True
    elif surface == "claim":
        status["production_claim_evidence"][
            "active_collector_claimed_cells"] = [7]
    else:
        status["production_task_completion_reset_evidence"][
            "all_scene_landmark_cells_claimed"] = True
    changed["chosen_state"]["state_identity_digest"] = \
        B._state_identity_digest(changed["chosen_state"])
    changed["state_resolution_scene_capture_digest"] = B.canonical_digest({
        key: value for key, value in changed.items()
        if key != "state_resolution_scene_capture_digest"
    })
    with pytest.raises(RuntimeError, match="snapshot task status changed"):
        B._validate_state_resolution_scene_capture(
            changed, expected_request=request)


def test_frozen_generated_artifact_guard_allows_only_exact_root_alias(
        tmp_path):
    lexical_root = tmp_path / "repository/.generated/go2_branch_corpus_v1_2"
    target_root = tmp_path / "external/go2_branch_corpus_v1_2"
    artifact = target_root / "scorer_fit/frozen.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("frozen")
    lexical_root.parent.mkdir(parents=True)
    lexical_root.symlink_to(target_root, target_is_directory=True)

    checked = B._frozen_generated_artifact_path(
        lexical_root / "scorer_fit/frozen.json",
        generated_root=lexical_root)
    assert checked == artifact
    assert checked.read_text() == "frozen"


def test_frozen_generated_artifact_guard_rejects_sealed_alias_target_pre_read(
        tmp_path):
    lexical_root = tmp_path / "repository/.generated/go2_branch_corpus_v1_2"
    # Deliberately leave this synthetic custody target nonexistent.  The guard
    # must reject its name before any target traversal or artifact read.
    target_root = (
        tmp_path / "sealed_synthetic_target/go2_branch_corpus_v1_2"
    )
    lexical_root.parent.mkdir(parents=True)
    lexical_root.symlink_to(target_root, target_is_directory=True)
    with pytest.raises(RuntimeError, match="alias target identity"):
        B._frozen_generated_artifact_path(
            lexical_root / "scorer_fit/frozen.json",
            generated_root=lexical_root)


def test_frozen_generated_artifact_guard_rejects_descendant_symlink(
        tmp_path):
    lexical_root = tmp_path / "repository/.generated/go2_branch_corpus_v1_2"
    target_root = tmp_path / "external/go2_branch_corpus_v1_2"
    redirected = tmp_path / "external/redirected"
    target_root.mkdir(parents=True)
    redirected.mkdir(parents=True)
    (redirected / "frozen.json").write_text("redirected")
    (target_root / "scorer_fit").symlink_to(
        redirected, target_is_directory=True)
    lexical_root.parent.mkdir(parents=True)
    lexical_root.symlink_to(target_root, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlinked corpus paths"):
        B._frozen_generated_artifact_path(
            lexical_root / "scorer_fit/frozen.json",
            generated_root=lexical_root)


def test_frozen_generated_artifact_guard_rejects_leaf_symlink(tmp_path):
    lexical_root = tmp_path / "repository/.generated/go2_branch_corpus_v1_2"
    target_root = tmp_path / "external/go2_branch_corpus_v1_2"
    other = tmp_path / "external/other.json"
    (target_root / "scorer_fit").mkdir(parents=True)
    other.write_text("redirected")
    (target_root / "scorer_fit/frozen.json").symlink_to(other)
    lexical_root.parent.mkdir(parents=True)
    lexical_root.symlink_to(target_root, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlinked corpus paths"):
        B._frozen_generated_artifact_path(
            lexical_root / "scorer_fit/frozen.json",
            generated_root=lexical_root)


def test_frozen_generated_artifact_guard_canonical_return_survives_alias_swap(
        tmp_path):
    lexical_root = tmp_path / "repository/.generated/go2_branch_corpus_v1_2"
    first_root = tmp_path / "first/go2_branch_corpus_v1_2"
    second_root = tmp_path / "second/go2_branch_corpus_v1_2"
    first_artifact = first_root / "scorer_fit/frozen.json"
    second_artifact = second_root / "scorer_fit/frozen.json"
    first_artifact.parent.mkdir(parents=True)
    second_artifact.parent.mkdir(parents=True)
    first_artifact.write_text("first-byte-identity")
    second_artifact.write_text("second-byte-identity")
    lexical_root.parent.mkdir(parents=True)
    lexical_root.symlink_to(first_root, target_is_directory=True)

    checked = B._frozen_generated_artifact_path(
        lexical_root / "scorer_fit/frozen.json",
        generated_root=lexical_root)
    lexical_root.unlink()
    lexical_root.symlink_to(second_root, target_is_directory=True)
    assert checked == first_artifact
    assert checked.read_text() == "first-byte-identity"
    assert (lexical_root / "scorer_fit/frozen.json").read_text() == \
        "second-byte-identity"


def test_frozen_generated_artifact_guard_rejects_escape_and_wrong_target_name(
        tmp_path):
    lexical_root = tmp_path / "repository/.generated/go2_branch_corpus_v1_2"
    target_root = tmp_path / "external/go2_branch_corpus_v1_2"
    target_root.mkdir(parents=True)
    lexical_root.parent.mkdir(parents=True)
    lexical_root.symlink_to(target_root, target_is_directory=True)
    with pytest.raises(RuntimeError, match="escaped the managed output root"):
        B._frozen_generated_artifact_path(
            tmp_path / "outside/frozen.json", generated_root=lexical_root)

    lexical_root.unlink()
    wrong_name = tmp_path / "external/different_artifact_root"
    wrong_name.mkdir()
    lexical_root.symlink_to(wrong_name, target_is_directory=True)
    with pytest.raises(RuntimeError, match="alias target identity"):
        B._frozen_generated_artifact_path(
            lexical_root / "scorer_fit/frozen.json",
            generated_root=lexical_root)


def test_pin_generated_path_requires_exact_logical_artifact_under_alias(
        tmp_path, monkeypatch):
    lexical_root = tmp_path / "repository/.generated/go2_branch_corpus_v1_2"
    target_root = tmp_path / "external/go2_branch_corpus_v1_2"
    expected = target_root / "scorer_fit/active.json"
    expected.parent.mkdir(parents=True)
    expected.write_text("active")
    (target_root / "scorer_fit/other.json").write_text("other")
    lexical_root.parent.mkdir(parents=True)
    lexical_root.symlink_to(target_root, target_is_directory=True)
    monkeypatch.setattr(B, "OUT_ROOT", lexical_root)

    raw_expected = lexical_root / "scorer_fit/active.json"
    assert B._pin_generated_path(raw_expected, raw_expected) == expected
    with pytest.raises(RuntimeError, match="path identity changed"):
        B._pin_generated_path(
            lexical_root / "scorer_fit/other.json", raw_expected)


def test_frozen_lineage_and_scene_shard_load_through_exact_output_alias(
        tmp_path, monkeypatch):
    lexical_root = tmp_path / "repository/.generated/go2_branch_corpus_v1_2"
    target_root = tmp_path / "external/go2_branch_corpus_v1_2"
    out = target_root / "scorer_fit"
    out.mkdir(parents=True)
    lexical_root.parent.mkdir(parents=True)
    lexical_root.symlink_to(target_root, target_is_directory=True)
    lexical_out = lexical_root / "scorer_fit"

    failure = {"status": "FAIL_OUTCOME_FREE_SELECTOR_FEASIBILITY"}
    failure["failure_report_digest"] = B.canonical_digest(failure)
    failure_path = tmp_path / "frozen_failure.json"
    failure_path.write_text(json.dumps(failure, sort_keys=True))
    task = {
        "family": "synthetic_family", "scene_id": "synthetic_scene",
        "scene_task_digest": "1" * 64,
    }
    shard = {
        "task": task, "complete": True,
        **{key: False if key not in {
            "branches_attempted", "frames_rendered", "target_latents_encoded"
        } else 0 for key in B.SELECTOR_FEASIBILITY_FORBIDDEN_FIELDS},
    }
    shard["state_selector_feasibility_scene_shard_digest"] = \
        B.canonical_digest(shard)
    shard_path = (
        out / B.SELECTOR_FEASIBILITY_SCENE_SHARD_ROOT
        / task["family"] / f"{task['scene_task_digest']}.json"
    )
    shard_path.parent.mkdir(parents=True)
    shard_path.write_text(json.dumps(shard, sort_keys=True))
    lineage = [{
        "family": task["family"], "scene_id": task["scene_id"],
        "scene_task_digest": task["scene_task_digest"],
        "scene_shard_digest":
            shard["state_selector_feasibility_scene_shard_digest"],
    }]
    receipt = {
        "status": "FAIL_OUTCOME_FREE_SELECTOR_FEASIBILITY",
        "scene_task_count": 1, "scene_shard_count": 1,
        "scene_shard_lineage": lineage,
        "scene_shard_lineage_digest": B.canonical_digest(lineage),
    }
    receipt["state_selector_feasibility_receipt_digest"] = \
        B.canonical_digest(receipt)
    receipt_path = out / B.SELECTOR_FEASIBILITY_RECEIPT_NAME
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    census = {
        "scene_task_count": 1,
        "families": [{"family": task["family"], "tasks": [task]}],
    }
    census["state_selector_feasibility_task_census_digest"] = \
        B.canonical_digest(census)
    census_path = out / B.SELECTOR_FEASIBILITY_TASK_CENSUS_NAME
    census_path.write_text(json.dumps(census, sort_keys=True))

    monkeypatch.setattr(B, "OUT_ROOT", lexical_root)
    monkeypatch.setattr(B, "FROZEN_FEASIBILITY_FAILURE_REPORT_PATH",
                        failure_path)
    monkeypatch.setattr(B, "FROZEN_FEASIBILITY_FAILURE_REPORT_RAW_SHA256",
                        B.file_sha256(failure_path))
    monkeypatch.setattr(B, "FROZEN_FEASIBILITY_FAILURE_REPORT_DIGEST",
                        failure["failure_report_digest"])
    monkeypatch.setattr(B, "FROZEN_FEASIBILITY_RECEIPT_RAW_SHA256",
                        B.file_sha256(receipt_path))
    monkeypatch.setattr(B, "FROZEN_FEASIBILITY_RECEIPT_DIGEST",
                        receipt["state_selector_feasibility_receipt_digest"])
    monkeypatch.setattr(B, "FROZEN_FEASIBILITY_TASK_CENSUS_DIGEST",
                        census[
                            "state_selector_feasibility_task_census_digest"])
    monkeypatch.setattr(B, "FROZEN_FEASIBILITY_SCENE_SHARD_COUNT", 1)

    loaded_failure, loaded_receipt, loaded_census = \
        B._load_frozen_selector_feasibility_lineage(lexical_out)
    assert loaded_failure == failure
    assert loaded_receipt == receipt
    assert loaded_census == census
    loaded_shards = B._frozen_selector_scene_shards(
        out=lexical_out, receipt=loaded_receipt, census=loaded_census)
    assert loaded_shards[(task["family"], task["scene_id"])] == shard


def _resign_feasibility(receipt):
    payload = copy.deepcopy(receipt)
    payload.pop("state_selector_feasibility_receipt_digest", None)
    payload["state_selector_feasibility_receipt_digest"] = \
        B.canonical_digest(payload)
    return payload


def _resign_phase1_attestation(attestation):
    payload = copy.deepcopy(attestation)
    payload.pop("attestation_digest", None)
    payload["attestation_digest"] = B.canonical_digest(payload)
    return payload


def test_phase1_outcome_surface_absence_attestation_covers_known_outputs(
        tmp_path):
    attestation = B._phase1_outcome_surface_absence_attestation(root=tmp_path)
    assert attestation["status"] == "PASS_PRE_OUTCOME_SURFACE_ABSENT"
    assert attestation["all_forbidden_artifacts_absent"] is True
    assert attestation["forbidden_artifact_count"] == 0
    assert B._phase1_present_outcome_paths(attestation) == []
    B.STATE_SELECTOR.validate_phase1_outcome_surface_absence_attestation(
        attestation)

    exact_paths = {row["path"] for row in attestation["exact_file_checks"]}
    directory_paths = {
        row["path"] for row in attestation["directory_root_checks"]}
    assert (
        ".generated/go2_branch_corpus_v1_2/scorer_fit/branch_rows.jsonl"
        in exact_paths
    )
    assert (
        ".generated/go2_branch_corpus_v1_2/scorer_fit/latents/context"
        in directory_paths
    )
    assert (
        ".generated/go2_branch_corpus_v1_2/scorer_fit/latents/horizon"
        in directory_paths
    )
    assert (
        ".generated/go2_utility_scorer_v1_2/"
        "counterfactual_development_transfer_v1_2"
        in directory_paths
    )


def test_phase1_outcome_surface_audit_rejects_orphan_latent_shard(tmp_path):
    relative = (
        ".generated/go2_branch_corpus_v1_2/scorer_fit/"
        "latents/horizon/orphan.f16"
    )
    orphan = tmp_path / relative
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"orphan-latent")

    attestation = B._phase1_outcome_surface_absence_attestation(root=tmp_path)
    assert attestation["status"] == "FAIL_PRE_OUTCOME_SURFACE_PRESENT"
    assert attestation["all_forbidden_artifacts_absent"] is False
    assert relative in B._phase1_present_outcome_paths(attestation)
    horizon = next(
        row for row in attestation["directory_root_checks"]
        if row["path"].endswith("latents/horizon")
    )
    assert horizon["descendant_artifact_count"] == 1
    assert horizon["descendant_artifacts"] == [relative]
    with pytest.raises(
        B.STATE_SELECTOR.StateSelectorAmendmentError,
        match="absence verdict is not complete/pass",
    ):
        B.STATE_SELECTOR.validate_phase1_outcome_surface_absence_attestation(
            attestation)


def test_phase1_attestation_validator_rejects_resigned_tamper(tmp_path):
    attestation = B._phase1_outcome_surface_absence_attestation(root=tmp_path)
    horizon = next(
        row for row in attestation["directory_root_checks"]
        if row["path"].endswith("latents/horizon")
    )
    horizon["descendant_artifact_count"] = 1
    horizon["descendant_artifacts"] = [
        horizon["path"] + "/fabricated.f16"
    ]
    tampered = _resign_phase1_attestation(attestation)
    with pytest.raises(
        B.STATE_SELECTOR.StateSelectorAmendmentError,
        match="forbidden directory contained an artifact",
    ):
        B.STATE_SELECTOR.validate_phase1_outcome_surface_absence_attestation(
            tampered)


def test_frozen_phase1_attestation_is_not_live_reopened_after_outputs(tmp_path):
    attestation = B._phase1_outcome_surface_absence_attestation(root=tmp_path)
    later = (
        tmp_path
        / ".generated/go2_branch_corpus_v1_2/scorer_fit/"
        "latents/context/later-legitimate.f16"
    )
    later.parent.mkdir(parents=True)
    later.write_bytes(b"later-legitimate-latent")

    # The validator is deliberately structural.  A later encoder output does
    # not rewrite history by turning the phase-1 issuance fact into a failure.
    B.STATE_SELECTOR.validate_phase1_outcome_surface_absence_attestation(
        attestation)
    live_now = B._phase1_outcome_surface_absence_attestation(root=tmp_path)
    assert live_now["all_forbidden_artifacts_absent"] is False


def _synthetic_phase1_state():
    return {
        "family": "large_enclosed_maze",
        "stratum": "general",
        "scene_id": "synthetic-scene",
        "state_id": "synthetic-state",
        "state_identity_digest": "1" * 64,
    }


def _synthetic_phase1_check(state):
    return {
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
    }


def test_phase1_state_check_shard_is_atomic_self_bound_and_resumable(tmp_path):
    state = _synthetic_phase1_state()
    predecessor = {
        "family": state["family"], "path": "preserved.json",
        "state_shard_digest": "2" * 64, "raw_sha256": "3" * 64,
        "byte_count": 10, "state_count": 15,
    }
    source = _source()
    shard = B._build_phase1_state_check_shard(
        entry=state, expected_shard=predecessor,
        check=_synthetic_phase1_check(state), source=source,
        successor_digest="4" * 64, feasibility_digest="5" * 64,
        outcome_surface_attestation_digest="6" * 64)
    path = B._phase1_state_check_shard_path(
        state["state_identity_digest"], root=tmp_path)
    B.atomic_json(path, shard)
    loaded = B._load_valid_phase1_state_check_shard(
        path=path, entry=state, expected_shard=predecessor, source=source,
        successor_digest="4" * 64, feasibility_digest="5" * 64,
        outcome_surface_attestation_digest="6" * 64, root=tmp_path)
    assert loaded == shard
    B.STATE_SELECTOR.validate_phase1_state_check_shard(
        loaded, expected_state=state,
        expected_predecessor_shard=predecessor,
        expected_source_commit=source["source_repository_commit"],
        expected_successor_selection_digest="4" * 64,
        expected_feasibility_receipt_digest="5" * 64,
        expected_outcome_surface_attestation_digest="6" * 64)

    tampered = copy.deepcopy(shard)
    tampered["state_check"]["goal_binding_unchanged"] = False
    tampered["state_check_shard_digest"] = B.canonical_digest({
        key: value for key, value in tampered.items()
        if key != "state_check_shard_digest"
    })
    B.atomic_json(path, tampered)
    assert B._load_valid_phase1_state_check_shard(
        path=path, entry=state, expected_shard=predecessor, source=source,
        successor_digest="4" * 64, feasibility_digest="5" * 64,
        outcome_surface_attestation_digest="6" * 64,
        root=tmp_path) is None


def test_phase1_parent_contains_no_genesis_context_or_shared_load():
    source = inspect.getsource(
        B.stage_preserved_state_precontract_revalidation)
    assert "V1._load_shared" not in source
    assert "V1.build_context" not in source
    assert "validate_frozen_preserved_precontract_failure" in source
    assert "build_preserved_state_mixed_precontract_disposition_receipt" in source
    assert "_phase1_outcome_surface_absence_attestation" in source


def _synthetic_terminal_phase1_failure(tmp_path, monkeypatch):
    source = _source()
    successor_digest = "4" * 64
    feasibility_digest = "5" * 64
    absence = B._phase1_outcome_surface_absence_attestation(root=tmp_path)
    preserved = {}
    expected_states = []
    shard_payloads = []
    serial = 0
    for expected in B.STATE_SELECTOR.PRESERVED_STATE_SHARDS:
        family = expected["family"]
        states = []
        for ordinal in range(15):
            state = {
                "family": family,
                "stratum": "general",
                "scene_id": f"{family}-scene-{ordinal:02d}",
                "state_id": f"{family}-state-{ordinal:02d}",
                "state_identity_digest": f"{serial + 10000:064x}",
            }
            serial += 1
            states.append(state)
            expected_states.append((dict(expected), state))
        preserved[family] = {"states": states}
    monkeypatch.setattr(
        B.STATE_SELECTOR, "load_preserved_state_shards",
        lambda _root: preserved)
    for index, (expected, state) in enumerate(expected_states):
        check = _synthetic_phase1_check(state)
        if index == 7:
            check["exact_redrive_pass"] = False
            check["failure_reason"] = "RuntimeError:synthetic frozen failure"
        shard = B._build_phase1_state_check_shard(
            entry=state, expected_shard=expected, check=check,
            source=source, successor_digest=successor_digest,
            feasibility_digest=feasibility_digest,
            outcome_surface_attestation_digest=absence["attestation_digest"])
        B.atomic_json(
            B._phase1_state_check_shard_path(
                state["state_identity_digest"], root=tmp_path),
            shard)
        shard_payloads.append(shard)
    receipt = B._build_phase1_aggregate_receipt(
        shard_payloads=shard_payloads,
        expected_states=expected_states,
        source=source,
        successor_digest=successor_digest,
        feasibility_digest=feasibility_digest,
        outcome_surface_absence=absence,
        root=tmp_path)
    assert receipt["status"] == "FAIL_PRECONTRACT_IDENTITY_REVALIDATION"
    return {
        "source": source,
        "successor_digest": successor_digest,
        "feasibility_digest": feasibility_digest,
        "expected_states": expected_states,
        "receipt": receipt,
    }


def test_phase1_failed_terminal_reconstructs_exact_45_atomic_checks(
        tmp_path, monkeypatch):
    fixture = _synthetic_terminal_phase1_failure(tmp_path, monkeypatch)
    rebuilt = B._reconstruct_terminal_phase1_failure(
        receipt=fixture["receipt"],
        expected_states=fixture["expected_states"],
        source=fixture["source"],
        successor_digest=fixture["successor_digest"],
        feasibility_digest=fixture["feasibility_digest"],
        root=tmp_path)
    assert rebuilt == fixture["receipt"]
    assert rebuilt["failure_count"] == 1


def test_phase1_failed_terminal_rejects_resigned_aggregate_tamper(
        tmp_path, monkeypatch):
    fixture = _synthetic_terminal_phase1_failure(tmp_path, monkeypatch)
    changed = copy.deepcopy(fixture["receipt"])
    changed["failures"][0]["failure_reason"] = \
        "RuntimeError:fabricated replacement failure"
    changed["preserved_state_precontract_revalidation_receipt_digest"] = \
        B.canonical_digest({
            key: value for key, value in changed.items()
            if key
            != "preserved_state_precontract_revalidation_receipt_digest"
        })
    with pytest.raises(RuntimeError, match="differs from its atomic checks"):
        B._reconstruct_terminal_phase1_failure(
            receipt=changed,
            expected_states=fixture["expected_states"],
            source=fixture["source"],
            successor_digest=fixture["successor_digest"],
            feasibility_digest=fixture["feasibility_digest"],
            root=tmp_path)


def test_mixed_replacement_plan_preserves_vacant_ordinal_anchor_intervals(
        monkeypatch):
    family = "large_enclosed_maze"
    states = []
    retained_rows = {}
    rejected_rows = {}
    slots = {}
    for stratum in ("general", "safety_enriched"):
        for ordinal in range(5):
            identity = f"{len(states) + 1:064x}"
            state = {
                "state_id": f"scorer_fit-{family}-{stratum}-{ordinal:02d}",
                "state_identity_digest": identity,
                "scene_id": f"{family}-{stratum}-{ordinal:02d}",
                "family": family, "stratum": stratum,
                "split_role": "calibration" if ordinal == 0 else "fit",
            }
            states.append(state)
            retained_rows[identity] = dict(state)
    anchor_identity = "f" * 64
    anchor = {
        "state_id": f"scorer_fit-{family}-completion_enriched-01",
        "state_identity_digest": anchor_identity,
        "scene_id": f"{family}-5757b144c54b", "family": family,
        "stratum": "completion_enriched", "split_role": "fit",
    }
    states.append(anchor)
    retained_rows[anchor_identity] = dict(anchor)
    for ordinal in (0, 2, 3, 4):
        identity = f"{100 + ordinal:064x}"
        state_id = f"scorer_fit-{family}-completion_enriched-{ordinal:02d}"
        rejected = {
            "state_id": state_id, "state_identity_digest": identity,
            "scene_id": f"{family}-rejected-{ordinal:02d}",
            "family": family, "stratum": "completion_enriched",
            "split_role": "calibration" if ordinal == 0 else "fit",
        }
        states.append(rejected)
        rejected_rows[identity] = dict(rejected)
        slots[state_id] = {
            "state_id": state_id, "family": family,
            "stratum": "completion_enriched",
            "split_role": rejected["split_role"],
            "predecessor_state_identity_digest": identity,
            "predecessor_scene_id": rejected["scene_id"],
        }
    monkeypatch.setattr(
        B.STATE_SELECTOR, "load_preserved_state_shards",
        lambda _root: {family: {"states": states}})
    monkeypatch.setattr(
        B, "_mixed_disposition_sets",
        lambda: (retained_rows, rejected_rows, slots))

    plan = B._mixed_family_replacement_plan(family)
    assert [(row["vacant_ordinals"], row["lower_scene_id_exclusive"],
             row["upper_scene_id_exclusive"])
            for row in plan["interval_groups"]] == [
        ([0], None, f"{family}-5757b144c54b"),
        ([2, 3, 4], f"{family}-5757b144c54b", None),
    ]


def test_replacement_rejects_same_physical_snapshot_even_if_goal_changes(
        monkeypatch):
    identity = "a" * 64
    predecessor = {
        "scene_id": "scene-a", "episode_cluster_id": "scene-a/env0/ep1",
        "episode_id": 1, "source_step": 400, "warmup_blocks": 40,
        "cell_id": 7, "boundary": {"source_step": 400},
        "goal": {"landmark_id": "red"},
    }
    monkeypatch.setattr(
        B, "_preserved_states_by_digest", lambda: {identity: predecessor})
    slot = {"predecessor_state_identity_digest": identity}
    changed_goal = copy.deepcopy(predecessor)
    changed_goal["goal"] = {"landmark_id": "blue"}
    changed_goal["completion_rotation_eligibility_vector"] = {"changed": True}
    assert B._replacement_reuses_rejected_snapshot(changed_goal, slot) is True
    later = copy.deepcopy(changed_goal)
    later["source_step"] = 401
    later["boundary"] = {"source_step": 401}
    assert B._replacement_reuses_rejected_snapshot(later, slot) is False


def test_replacement_rejects_snapshot_of_any_superseded_identity(monkeypatch):
    first = "a" * 64
    second = "b" * 64
    base = {
        "scene_id": "scene-a", "episode_cluster_id": "scene-a/env0/ep1",
        "episode_id": 1, "source_step": 400, "warmup_blocks": 40,
        "cell_id": 7, "boundary": {"source_step": 400},
    }
    other = {**base, "scene_id": "scene-b",
             "episode_cluster_id": "scene-b/env0/ep2", "episode_id": 2}
    monkeypatch.setattr(
        B, "_preserved_states_by_digest", lambda: {first: base, second: other})
    candidate = {**other, "goal": {"landmark_id": "changed"}}
    assert B._replacement_reuses_any_rejected_snapshot(
        candidate, [first, second]) is True
    candidate["source_step"] = 401
    candidate["boundary"] = {"source_step": 401}
    assert B._replacement_reuses_any_rejected_snapshot(
        candidate, [first, second]) is False


def test_mixed_worker_classification_error_is_atomically_durable(
        tmp_path, monkeypatch):
    request = {
        "family": "large_enclosed_maze",
        "scene_ordinal": 0,
        "scene": {"scene_id": "scene-a", "drive_seed": 1},
        "replacement_slot": {"state_id": "slot-a", "family":
                             "large_enclosed_maze",
                             "stratum": "completion_enriched",
                             "split_role": "fit",
                             "predecessor_state_identity_digest": "a" * 64,
                             "predecessor_scene_id": "old-scene"},
        "rejected_identity_digests": ["a" * 64],
        "mixed_replacement_scene_request_digest": "b" * 64,
    }
    capture_path = tmp_path / "capture.json"
    args = SimpleNamespace(
        pool="scorer_fit", family="large_enclosed_maze", backend="cpu")

    class Context:
        def begin_episode(self):
            return None

        def drive_one_block(self):
            return None

    monkeypatch.setattr(
        B, "_validate_mixed_replacement_scene_request",
        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        B, "_mixed_replacement_capture_path",
        lambda *_args, **_kwargs: capture_path)
    monkeypatch.setattr(B, "_pin_generated_path", lambda path, _expected: path)
    monkeypatch.setattr(B, "_load_valid_mixed_replacement_scene_capture",
                        lambda **_kwargs: None)
    monkeypatch.setattr(B.V1, "_load_shared", lambda _backend: object())
    monkeypatch.setattr(B.V1, "build_context", lambda *_args, **_kwargs: Context())
    monkeypatch.setattr(B.V12, "link_topology", lambda _ctx: {})
    monkeypatch.setattr(
        B, "classify_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic classifier failure")))

    capture = B._execute_mixed_replacement_scene_worker(
        args=args, request=request, out=tmp_path,
        pool={"large_enclosed_maze": [tmp_path / "scene-a"]}, exclusion={})
    assert capture_path.is_file()
    assert capture["worker_failure"] == "RuntimeError:synthetic classifier failure"
    assert capture["attempt_trace"][-1] == {
        "block_index": B.WARMUP_BLOCKS_MIN,
        "verdict": "ERROR",
        "reason_key": "RuntimeError:synthetic classifier failure",
    }
    B._validate_mixed_replacement_scene_capture(
        json.loads(capture_path.read_text()), expected_request=request)


def _synthetic_mixed_active_shard(tmp_path, monkeypatch):
    """Build one tiny but fully replayable mixed-family transport fixture."""

    family = "large_enclosed_maze"
    root = tmp_path / "repo"
    out_root = root / ".generated/go2_branch_corpus_v1_2"
    out = out_root / "scorer_fit"
    out.mkdir(parents=True)
    scenes = []
    for scene_id in ("scene-000", "scene-001"):
        scene = tmp_path / "development" / family / scene_id
        scene.mkdir(parents=True)
        (scene / "manifest.json").write_text("{}\n")
        scenes.append(scene)
    monkeypatch.setattr(B, "ROOT", root)
    monkeypatch.setattr(B, "OUT_ROOT", out_root)
    monkeypatch.setattr(B.V1, "_drive_seed", lambda name: len(name))

    retained = []
    for stratum in ("general", "safety_enriched"):
        for ordinal in range(5):
            retained.append({
                "state_id": f"scorer_fit-{family}-{stratum}-{ordinal:02d}",
                "state_identity_digest": f"{100 + len(retained):064x}",
                "scene_id": f"retained-{stratum}-{ordinal:02d}",
                "family": family,
                "stratum": stratum,
                "split_role": "calibration" if ordinal == 0 else "fit",
            })
    for ordinal in range(1, 5):
        retained.append({
            "state_id": (
                f"scorer_fit-{family}-completion_enriched-{ordinal:02d}"),
            "state_identity_digest": f"{200 + ordinal:064x}",
            "scene_id": f"scene-10{ordinal}",
            "family": family,
            "stratum": "completion_enriched",
            "split_role": "fit",
        })
    rejected_identity = "f" * 64
    slot = {
        "state_id": f"scorer_fit-{family}-completion_enriched-00",
        "family": family,
        "stratum": "completion_enriched",
        "split_role": "calibration",
        "predecessor_state_identity_digest": rejected_identity,
        "predecessor_scene_id": "rejected-scene",
    }
    interval = {
        "lower_scene_id_exclusive": None,
        "upper_scene_id_exclusive": "scene-101",
        "vacant_ordinals": [0],
        "replacement_slots": [slot],
    }
    plan = {
        "family": family,
        "retained_states": retained,
        "retained_state_count": len(retained),
        "retained_scene_ids": sorted(row["scene_id"] for row in retained),
        "retained_anchor_rows": [],
        "rejected_identity_digests": [rejected_identity],
        "rejected_identity_rows": [{
            **slot, "state_identity_digest": rejected_identity,
            "scene_id": "rejected-scene",
        }],
        "replacement_slots": [slot],
        "interval_groups": [interval],
    }
    rejected_predecessor = {
        "scene_id": "rejected-scene",
        "episode_cluster_id": "rejected-scene/env0/ep1",
        "episode_id": 1,
        "source_step": 400,
        "warmup_blocks": 40,
        "cell_id": 9,
        "boundary": {"source_step": 400},
    }
    retained_map = {
        row["state_identity_digest"]: dict(row) for row in retained
    }
    rejected_map = {rejected_identity: {
        **slot,
        "state_identity_digest": rejected_identity,
        "scene_id": "rejected-scene",
    }}
    monkeypatch.setattr(B, "_mixed_family_replacement_plan", lambda _family: plan)
    monkeypatch.setattr(
        B, "_mixed_disposition_sets",
        lambda: (retained_map, rejected_map, {slot["state_id"]: slot}))
    monkeypatch.setattr(
        B, "_preserved_states_by_digest",
        lambda: {rejected_identity: rejected_predecessor})
    monkeypatch.setattr(
        B, "scene_pool", lambda _pool: ({family: scenes}, {"synthetic": True}))
    monkeypatch.setattr(B.INVALID_IDS, "assert_disjoint", lambda *_a, **_k: {})

    launch = {
        "source_repository_commit": "a" * 40,
        "clean_source_launch_receipt_digest": "b" * 64,
        "clean_source_binding_digest": "c" * 64,
        "bound_implementations_digest": "d" * 64,
        "scorer_contract_artifact_digest": "e" * 64,
        "state_selector_feasibility_receipt_digest": "6" * 64,
        "mixed_precontract_disposition_receipt_digest": "7" * 64,
    }
    monkeypatch.setattr(
        B, "_load_clean_source_launch_receipt", lambda: dict(launch))
    bindings = {
        "selection_digest": B.selection_digest(),
        "scorer_fit_allocation_design_digest": "8" * 64,
        "candidate_allocator_contract_digest":
            B.ALLOC.allocation_contract_digest(),
        "candidate_allocation_amendment_digest":
            B.ALLOC.allocation_amendment_digest(),
        "pre_identity_allocation_validation_digest": "9" * 64,
        "invalid_scorer_identity_exclusion_digest":
            B.INVALID_IDS.invalid_identity_exclusion_digest(),
        "state_selector_amendment_digest":
            B.STATE_SELECTOR.state_selector_amendment_digest(),
        "state_selector_feasibility_receipt_digest":
            launch["state_selector_feasibility_receipt_digest"],
        **{key: launch[key] for key in B.LAUNCH_BINDING_KEYS},
        "candidate_bank_digest": B.V1.bank_digest(),
        "progress_contract_digest": B.progress_digest(),
        "safety_contract_digest": B.safety_digest(),
        "oracle_v1_2_digest": B.v12_oracle_digest(),
        "scorer_contract_v1_2_digest": B.scorer_contract_digest(),
        "boundary_digest": B.V1.BOUNDARY_DIGEST,
        "render_contract_digest": B.render_contract_digest(),
        "textured_v03_renderer_contract_digest":
            B.textured_v03_renderer_contract_digest(),
        "preprocess_contract_digest": B.preprocess_contract_digest(),
        "preprocessing_digest":
            B.TARGET_ENCODER["preprocessing_identity_sha256"],
        "target_encoder_digest": B.target_encoder_digest(),
        "target_encoder_checkpoint_sha256":
            B.TARGET_ENCODER["checkpoint_sha256"],
        "genesis_backend": "cpu",
        "exclusion_binding": {"synthetic": True},
        "family_allow_list_digest": B.canonical_digest(
            [scene.name for scene in scenes]),
    }
    monkeypatch.setattr(
        B, "_state_shard_bindings", lambda *_a, **_k: dict(bindings))

    predecessor_path = out / "predecessor-large.json"
    predecessor_path.write_text("{}\n")
    predecessor_binding = {
        "family": family,
        "path": str(predecessor_path.relative_to(root)),
        "raw_sha256": B.file_sha256(predecessor_path),
        "byte_count": predecessor_path.stat().st_size,
    }
    monkeypatch.setattr(
        B.STATE_SELECTOR, "PRESERVED_STATE_SHARDS", [predecessor_binding])

    args = SimpleNamespace(pool="scorer_fit", family=family, backend="cpu")
    request_reject = B._build_mixed_replacement_scene_request(
        args=args, out=out, scene_dir=scenes[0], scene_ordinal=0,
        interval=interval, slot=slot, accepted_scene_ids_before=[],
        exclusion={"synthetic": True},
        family_allow_list=[scene.name for scene in scenes])
    reject_trace = [{
        "block_index": block,
        "verdict": "REJECT",
        "reason_key": "no_completion_enriched_goal",
    } for block in range(B.WARMUP_BLOCKS_MIN, B.WARMUP_BLOCKS_MAX + 1)]
    capture_reject = B._build_mixed_replacement_scene_capture(
        request=request_reject, chosen_state=None,
        rejection_reasons={"no_completion_enriched_goal": len(reject_trace)},
        worker_failure=None, blocks_driven=B.WARMUP_BLOCKS_MAX,
        attempt_trace=reject_trace)
    reject_capture_path = B._mixed_replacement_capture_path(
        out, family,
        request_reject["mixed_replacement_scene_request_digest"])
    B.atomic_json(reject_capture_path, capture_reject)

    request_select = B._build_mixed_replacement_scene_request(
        args=args, out=out, scene_dir=scenes[1], scene_ordinal=1,
        interval=interval, slot=slot, accepted_scene_ids_before=[],
        exclusion={"synthetic": True},
        family_allow_list=[scene.name for scene in scenes])
    status = _full_snapshot_status(goal_cell=7)
    vector = B.STATE_SELECTOR.completion_rotation_eligibility_vector(
        graph_hops=0, reachable=True, continuous_geodesic_m=0.8,
        bearing_body_rad=0.0, task_status=status,
        previous_applied_command=[0.0, 0.0, 0.0])
    first = vector["rotations"][0]
    chosen = {
        "state_id": slot["state_id"],
        "family": family,
        "scene_id": scenes[1].name,
        "scene_dir": str(scenes[1].resolve()),
        "scene_manifest_sha256": request_select["scene"][
            "scene_manifest_sha256"],
        "scene_manifest_byte_count": request_select["scene"][
            "scene_manifest_byte_count"],
        "split": request_select["scene"]["split"],
        "drive_seed": request_select["scene"]["drive_seed"],
        "stratum": "completion_enriched",
        "split_role": "calibration",
        "warmup_blocks": B.WARMUP_BLOCKS_MIN,
        "source_step": 500,
        "episode_id": 2,
        "episode_cluster_id": f"{scenes[1].name}/env0/ep2",
        "cell_id": 3,
        "boundary": {"source_step": 500},
        "goal": {
            "landmark_id": "goal-7",
            "landmark_cell": 7,
            "material_id": "landmark_red",
            "graph_edges": first["graph_hops_diagnostic"],
            "start_geodesic_m": first["continuous_geodesic_m"],
            "bearing_body_rad": first["bearing_body_rad"],
            "range_m": first["continuous_geodesic_m"],
            "landmark_xy_m": [0.0, 0.0],
        },
        "goal_type": "landmark_red",
        "body_clearance_m": 0.2,
        "clearance_m": 0.3,
        "completion_rotation_eligibility_vector": vector,
        "snapshot_task_status": status,
        "previous_applied_command": first["previous_applied_command"],
    }
    chosen["state_identity_digest"] = B._state_identity_digest(chosen)
    capture_select = B._build_mixed_replacement_scene_capture(
        request=request_select, chosen_state=chosen, rejection_reasons={},
        worker_failure=None, blocks_driven=B.WARMUP_BLOCKS_MIN,
        attempt_trace=[{
            "block_index": B.WARMUP_BLOCKS_MIN,
            "verdict": "SELECT",
            "reason_key": None,
        }])
    select_capture_path = B._mixed_replacement_capture_path(
        out, family,
        request_select["mixed_replacement_scene_request_digest"])
    B.atomic_json(select_capture_path, capture_select)
    provenance = [
        B._mixed_capture_provenance(
            out=out, request=request_reject, capture=capture_reject,
            interval_index=0),
        B._mixed_capture_provenance(
            out=out, request=request_select, capture=capture_select,
            interval_index=0),
    ]
    states = sorted(
        [dict(row) for row in retained] + [chosen],
        key=lambda row: (
            B.STRATA.index(str(row["stratum"])), str(row["state_id"])))
    interval_row = {
        "interval_index": 0,
        "lower_scene_id_exclusive": None,
        "upper_scene_id_exclusive": "scene-101",
        "vacant_ordinals": [0],
        "replacement_slot_state_ids": [slot["state_id"]],
        "candidate_scene_ids": [scene.name for scene in scenes],
        "scanned_scene_ids": [scene.name for scene in scenes],
        "selected_scene_ids": [scenes[1].name],
        "stopped_at_first_complete_prefix": True,
    }
    shard = {
        "schema": B.MIXED_ACTIVE_STATE_SHARD_SCHEMA,
        "status": B.STATUS,
        "complete": True,
        "pool": "scorer_fit",
        "family": family,
        "spec": B.POOLS["scorer_fit"],
        "selection": B.SELECTION,
        **bindings,
        "predecessor_state_shard_binding": predecessor_binding,
        "retained_predecessor_identity_digests": sorted(retained_map),
        "rejected_predecessor_identity_digests": [rejected_identity],
        "replacement_slot_fills": [{
            "state_id": chosen["state_id"],
            "state_identity_digest": chosen["state_identity_digest"],
            "scene_id": chosen["scene_id"],
            "split_role": chosen["split_role"],
        }],
        "states": states,
        "scene_rejection_reasons": {
            scenes[0].name: capture_reject["scene_rejection_reasons"],
            scenes[1].name: {},
        },
        "mixed_replacement_subprocess_transport": {
            "schema": B.MIXED_REPLACEMENT_TRANSPORT_SCHEMA,
            "one_scene_per_subprocess": True,
            "atomic_capture_write_before_native_cleanup": True,
            "return_code_ignored_only_after_valid_capture": True,
            "resume_scope": "MISSING_OR_INVALID_REPLACEMENT_SCENE_CAPTURES_ONLY",
            "interval_rows": [interval_row],
            "scene_capture_count": len(provenance),
            "scene_capture_provenance_digest": B.canonical_digest(provenance),
            "candidate_outcomes_loaded": False,
        },
        "mixed_replacement_scene_capture_provenance": provenance,
    }
    shard["state_shard_digest"] = B.canonical_digest(shard)
    shard_path = B._mixed_active_state_shard_path(out, family)
    B.atomic_json(shard_path, shard)
    return {
        "args": args,
        "out": out,
        "pool": {family: scenes},
        "exclusion": {"synthetic": True},
        "family": family,
        "plan": plan,
        "shard": shard,
        "shard_path": shard_path,
        "requests": [request_reject, request_select],
        "captures": [capture_reject, capture_select],
        "capture_paths": [reject_capture_path, select_capture_path],
    }


def test_mixed_active_shard_replays_full_prefix_and_rejects_post_quota_tamper(
        tmp_path, monkeypatch):
    fixture = _synthetic_mixed_active_shard(tmp_path, monkeypatch)
    B._validate_mixed_active_state_shard(
        fixture["shard"], fixture["shard_path"])

    tampered = copy.deepcopy(fixture["shard"])
    tampered["mixed_replacement_scene_capture_provenance"].append(
        copy.deepcopy(tampered["mixed_replacement_scene_capture_provenance"][-1]))
    transport = tampered["mixed_replacement_subprocess_transport"]
    transport["scene_capture_count"] += 1
    transport["scene_capture_provenance_digest"] = B.canonical_digest(
        tampered["mixed_replacement_scene_capture_provenance"])
    tampered["state_shard_digest"] = B.canonical_digest({
        key: value for key, value in tampered.items()
        if key != "state_shard_digest"
    })
    B.atomic_json(fixture["shard_path"], tampered)
    with pytest.raises(RuntimeError, match="post-quota"):
        B._validate_mixed_active_state_shard(tampered, fixture["shard_path"])


@pytest.mark.parametrize(
    "surface", ("selector_flag", "claim", "reset", "termination")
)
def test_mixed_selected_capture_binds_full_snapshot_status_projection(
        tmp_path, monkeypatch, surface):
    fixture = _synthetic_mixed_active_shard(tmp_path, monkeypatch)
    capture = copy.deepcopy(fixture["captures"][1])
    request = fixture["requests"][1]
    B._validate_mixed_replacement_scene_capture(
        capture, expected_request=request)

    status = capture["chosen_state"]["snapshot_task_status"]
    if surface == "selector_flag":
        status["task_completed"] = True
    elif surface == "claim":
        status["production_claim_evidence"][
            "active_collector_claimed_cells"] = [7]
    elif surface == "reset":
        status["production_task_completion_reset_evidence"][
            "all_scene_landmark_cells_claimed"] = True
    else:
        status["termination_flags"]["fall"] = True
    capture["chosen_state"]["state_identity_digest"] = \
        B._state_identity_digest(capture["chosen_state"])
    capture["mixed_replacement_scene_capture_digest"] = B.canonical_digest({
        key: value for key, value in capture.items()
        if key != "mixed_replacement_scene_capture_digest"
    })
    with pytest.raises(RuntimeError, match="snapshot task status changed"):
        B._validate_mixed_replacement_scene_capture(
            capture, expected_request=request)


def test_mixed_active_shard_requires_full_retained_payload_equality(
        tmp_path, monkeypatch):
    fixture = _synthetic_mixed_active_shard(tmp_path, monkeypatch)
    tampered = copy.deepcopy(fixture["shard"])
    retained = next(
        row for row in tampered["states"] if row["stratum"] == "general")
    retained["goal"] = {"fabricated": True}
    tampered["state_shard_digest"] = B.canonical_digest({
        key: value for key, value in tampered.items()
        if key != "state_shard_digest"
    })
    B.atomic_json(fixture["shard_path"], tampered)
    with pytest.raises(RuntimeError, match="live replay"):
        B._validate_mixed_active_state_shard(tampered, fixture["shard_path"])


def test_mixed_terminal_failure_is_idempotent_and_live_replayed(
        tmp_path, monkeypatch):
    fixture = _synthetic_mixed_active_shard(tmp_path, monkeypatch)
    request = fixture["requests"][1]
    trace = [{
        "block_index": block,
        "verdict": "REJECT",
        "reason_key": "no_completion_enriched_goal",
    } for block in range(B.WARMUP_BLOCKS_MIN, B.WARMUP_BLOCKS_MAX + 1)]
    rejected = B._build_mixed_replacement_scene_capture(
        request=request, chosen_state=None,
        rejection_reasons={"no_completion_enriched_goal": len(trace)},
        worker_failure=None, blocks_driven=B.WARMUP_BLOCKS_MAX,
        attempt_trace=trace)
    B.atomic_json(fixture["capture_paths"][1], rejected)
    provenance = [
        B._mixed_capture_provenance(
            out=fixture["out"], request=fixture["requests"][0],
            capture=fixture["captures"][0], interval_index=0),
        B._mixed_capture_provenance(
            out=fixture["out"], request=request, capture=rejected,
            interval_index=0),
    ]
    candidates = [scene.name for scene in fixture["pool"][fixture["family"]]]
    first = B._issue_mixed_replacement_failure(
        args=fixture["args"], out=fixture["out"], plan=fixture["plan"],
        interval_index=0, candidate_scene_ids=candidates,
        accepted_states=[], provenance=provenance)
    second = B._issue_mixed_replacement_failure(
        args=fixture["args"], out=fixture["out"], plan=fixture["plan"],
        interval_index=0, candidate_scene_ids=candidates,
        accepted_states=[], provenance=provenance)
    assert first == second
    assert B._load_mixed_replacement_failure(
        args=fixture["args"], out=fixture["out"], pool=fixture["pool"],
        exclusion=fixture["exclusion"]) == first

    changed = copy.deepcopy(first)
    changed["scanned_scene_count"] = 1
    changed["mixed_replacement_failure_receipt_digest"] = B.canonical_digest({
        key: value for key, value in changed.items()
        if key != "mixed_replacement_failure_receipt_digest"
    })
    B.atomic_json(
        fixture["out"] / B.MIXED_REPLACEMENT_FAILURE_NAME, changed)
    with pytest.raises(RuntimeError, match="live replay"):
        B._load_mixed_replacement_failure(
            args=fixture["args"], out=fixture["out"], pool=fixture["pool"],
            exclusion=fixture["exclusion"])


def test_state_shard_provenance_mixes_three_active_and_five_successors_under_alias(
        tmp_path, monkeypatch):
    root = tmp_path / "repo"
    lexical_root = root / ".generated/go2_branch_corpus_v1_2"
    target_root = tmp_path / "managed/go2_branch_corpus_v1_2"
    (target_root / "scorer_fit").mkdir(parents=True)
    lexical_root.parent.mkdir(parents=True)
    lexical_root.symlink_to(target_root, target_is_directory=True)
    monkeypatch.setattr(B, "ROOT", root)
    monkeypatch.setattr(B, "OUT_ROOT", lexical_root)
    families = [f"family-{index}" for index in range(8)]
    monkeypatch.setattr(B.STATE_SELECTOR, "REQUIRED_FAMILIES", families)
    monkeypatch.setattr(
        B.STATE_SELECTOR, "PRESERVED_STATE_SHARDS",
        [{"family": family} for family in families[:3]])
    paths = []
    shards = []
    for index, family in enumerate(families):
        raw = B._active_state_shard_path(
            lexical_root / "scorer_fit", family, pool="scorer_fit")
        canonical = B._pin_generated_path(raw, raw)
        payload = {
            "family": family,
            "state_shard_digest": f"{index + 1:064x}",
        }
        B.atomic_json(canonical, payload)
        paths.append(canonical)
        shards.append(payload)
    rows = B._build_state_shard_provenance(
        paths, shards, pool_name="scorer_fit")
    assert [row["selection_provenance"] for row in rows].count(
        "MIXED_37_RETAINED_8_REPLACED_SELECTOR_AMENDMENT_V2") == 3
    assert [row["selection_provenance"] for row in rows].count(
        "SUCCESSOR_SELECTOR_AMENDMENT_V2") == 5
    assert all(not Path(row["path"]).is_absolute() for row in rows)


def test_active_family_loader_returns_canonical_path_under_managed_alias(
        tmp_path, monkeypatch):
    family = "large_enclosed_maze"
    lexical_root = tmp_path / "repo/.generated/go2_branch_corpus_v1_2"
    target_root = tmp_path / "managed/go2_branch_corpus_v1_2"
    raw = lexical_root / "scorer_fit" / B.MIXED_ACTIVE_STATE_SHARD_NAME.format(
        family=family)
    canonical = target_root / "scorer_fit" / raw.name
    canonical.parent.mkdir(parents=True)
    lexical_root.parent.mkdir(parents=True)
    lexical_root.symlink_to(target_root, target_is_directory=True)
    payload = {"family": family, "state_shard_digest": "a" * 64}
    B.atomic_json(canonical, payload)
    monkeypatch.setattr(B, "OUT_ROOT", lexical_root)
    monkeypatch.setattr(
        B.STATE_SELECTOR, "PRESERVED_STATE_SHARDS", [{"family": family}])
    monkeypatch.setattr(
        B, "_validate_mixed_active_state_shard", lambda *_a, **_k: None)
    returned, loaded = B._load_active_family_state_shard(
        lexical_root / "scorer_fit", family, pool="scorer_fit")
    assert returned == canonical
    assert loaded == payload


def _synthetic_state_request(*, requested=("general", "safety_enriched")):
    return {
        "pool": "scorer_fit",
        "family": "medium_enclosed_maze",
        "requested_strata_in_priority_order": list(requested),
        "found_before_scene": {
            "general": 0, "safety_enriched": 0, "completion_enriched": 0,
        },
    }


def test_state_resolution_trace_replays_dynamic_priority_and_rejection_ledger():
    request = _synthetic_state_request()
    trace = [
        {"block_index": B.WARMUP_BLOCKS_MIN, "attempts": [
            {"stratum": "general", "verdict": "REJECT",
             "reason_key": "general_miss"},
            {"stratum": "safety_enriched", "verdict": "REJECT",
             "reason_key": "safety_miss"},
        ]},
        {"block_index": B.WARMUP_BLOCKS_MIN + 1, "attempts": [
            {"stratum": "general", "verdict": "REJECT",
             "reason_key": "general_miss"},
            {"stratum": "safety_enriched", "verdict": "SELECT",
             "reason_key": None},
        ]},
    ]
    rejections, selected = B._replay_state_resolution_attempt_trace(
        request=request, attempt_trace=trace,
        blocks_driven=B.WARMUP_BLOCKS_MIN + 1, worker_failure=None)
    assert rejections == {
        "general:general_miss|safety_enriched:safety_miss": 1}
    assert selected == "safety_enriched"


def test_state_resolution_trace_rejects_omitted_earlier_priority_stratum():
    request = _synthetic_state_request()
    trace = [{"block_index": B.WARMUP_BLOCKS_MIN, "attempts": [
        {"stratum": "safety_enriched", "verdict": "SELECT",
         "reason_key": None},
    ]}]
    with pytest.raises(RuntimeError, match="stratum trace is malformed"):
        B._replay_state_resolution_attempt_trace(
            request=request, attempt_trace=trace,
            blocks_driven=B.WARMUP_BLOCKS_MIN, worker_failure=None)


def test_state_resolution_parent_contains_no_genesis_context_or_shared_load():
    source = inspect.getsource(B.resolve_states)
    assert "V1._load_shared" not in source
    assert "V1.build_context" not in source
    assert "_get_or_run_state_resolution_scene_capture" in source
    worker = inspect.getsource(B._execute_state_resolution_scene_worker)
    assert worker.index("atomic_json(capture_path, capture)") \
        < worker.index("del ctx")


def test_state_resolution_nonzero_teardown_is_accepted_only_with_valid_capture(
        monkeypatch, tmp_path):
    monkeypatch.setattr(B, "OUT_ROOT", tmp_path)
    args = SimpleNamespace(
        pool="scorer_fit", family="medium_enclosed_maze", backend="cpu")
    request = {
        "state_resolution_scene_request_digest": "7" * 64,
        "scene": {"scene_id": "synthetic-scene"},
    }
    capture = {"worker_failure": None, "complete": True}
    calls = []

    def load(*, path, request):
        calls.append(path)
        return None if len(calls) == 1 else capture

    monkeypatch.setattr(B, "_load_valid_state_resolution_scene_capture", load)
    result = B._get_or_run_state_resolution_scene_capture(
        args=args, request=request, out=tmp_path,
        runner=lambda _args, *, request_digest: -11)
    assert result is capture


def test_state_resolution_nonzero_without_capture_fails_missing_only(
        monkeypatch, tmp_path):
    monkeypatch.setattr(B, "OUT_ROOT", tmp_path)
    args = SimpleNamespace(
        pool="scorer_fit", family="medium_enclosed_maze", backend="cpu")
    request = {
        "state_resolution_scene_request_digest": "8" * 64,
        "scene": {"scene_id": "synthetic-scene"},
    }
    monkeypatch.setattr(
        B, "_load_valid_state_resolution_scene_capture",
        lambda **_kwargs: None)
    with pytest.raises(RuntimeError, match="without a valid durable capture"):
        B._get_or_run_state_resolution_scene_capture(
            args=args, request=request, out=tmp_path,
            runner=lambda _args, *, request_digest: -11)


def test_state_resolution_request_rejects_resigned_live_scene_task_tamper(
        monkeypatch, tmp_path):
    monkeypatch.setattr(B, "OUT_ROOT", tmp_path)
    family = "medium_enclosed_maze"
    scene_dir = tmp_path / "split" / "family" / "scene-000"
    scene_dir.mkdir(parents=True)
    (scene_dir / "manifest.json").write_text("{}\n")
    args = SimpleNamespace(pool="scorer_fit", family=family, backend="cpu")
    bindings = {"synthetic_binding": "9" * 64}
    monkeypatch.setattr(B, "_state_shard_bindings", lambda *args, **kwargs: bindings)
    monkeypatch.setattr(B.V1, "_drive_seed", lambda _scene_id: 17)
    request = B._build_state_resolution_scene_request(
        args=args, out=tmp_path, scene_dir=scene_dir, scene_ordinal=0,
        found={"general": 0, "safety_enriched": 0,
               "completion_enriched": 0},
        need={"general": 5, "safety_enriched": 5,
              "completion_enriched": 5},
        exclusion={}, family_allow_list=[scene_dir.name])
    B._validate_state_resolution_scene_request(
        request, args=args, out=tmp_path, pool={family: [scene_dir]},
        exclusion={})

    tampered = copy.deepcopy(request)
    tampered["scene"]["scene_id"] = "fabricated-scene"
    tampered["state_resolution_scene_request_digest"] = B.canonical_digest({
        key: value for key, value in tampered.items()
        if key != "state_resolution_scene_request_digest"
    })
    with pytest.raises(RuntimeError, match="live frozen inputs"):
        B._validate_state_resolution_scene_request(
            tampered, args=args, out=tmp_path, pool={family: [scene_dir]},
            exclusion={})

    post_quota = copy.deepcopy(request)
    post_quota["found_before_scene"] = {
        "general": 5, "safety_enriched": 5, "completion_enriched": 5,
    }
    post_quota["requested_strata_in_priority_order"] = []
    post_quota["state_resolution_scene_request_digest"] = B.canonical_digest({
        key: value for key, value in post_quota.items()
        if key != "state_resolution_scene_request_digest"
    })
    with pytest.raises(RuntimeError, match="live frozen inputs"):
        B._validate_state_resolution_scene_request(
            post_quota, args=args, out=tmp_path,
            pool={family: [scene_dir]}, exclusion={})


def test_transport_rejects_self_bound_post_quota_failure_and_advanced_cursor(
        monkeypatch, tmp_path):
    family = "medium_enclosed_maze"
    out_root = tmp_path / "generated"
    out = out_root / "scorer_fit"
    out.mkdir(parents=True)
    scenes = []
    for name in ("scene-a", "scene-b"):
        scene = tmp_path / "development" / family / name
        scene.mkdir(parents=True)
        (scene / "manifest.json").write_text("{}\n")
        scenes.append(scene)
    args = SimpleNamespace(pool="scorer_fit", family=family, backend="cpu")
    bindings = {"synthetic_binding": "9" * 64}
    synthetic_pools = copy.deepcopy(B.POOLS)
    synthetic_pools["scorer_fit"] = {
        **synthetic_pools["scorer_fit"],
        "states_per_family": 1,
        "strata": {
            "general": 1, "safety_enriched": 0,
            "completion_enriched": 0,
        },
    }
    monkeypatch.setattr(B, "ROOT", tmp_path)
    monkeypatch.setattr(B, "OUT_ROOT", out_root)
    monkeypatch.setattr(B, "POOLS", synthetic_pools)
    monkeypatch.setattr(B, "scene_pool", lambda _pool: ({family: scenes}, {}))
    monkeypatch.setattr(B, "_state_shard_bindings",
                        lambda *args, **kwargs: bindings)
    monkeypatch.setattr(B.V1, "_drive_seed", lambda name: len(name))

    request = B._build_state_resolution_scene_request(
        args=args, out=out, scene_dir=scenes[0], scene_ordinal=0,
        found={"general": 0, "safety_enriched": 0,
               "completion_enriched": 0},
        need={"general": 1, "safety_enriched": 0,
              "completion_enriched": 0},
        exclusion={}, family_allow_list=[scene.name for scene in scenes])
    chosen = {
        "state_id": f"scorer_fit-{family}-general-00",
        "family": family,
        "scene_id": scenes[0].name,
        "scene_dir": str(scenes[0].resolve()),
        "scene_manifest_sha256": request["scene"]["scene_manifest_sha256"],
        "scene_manifest_byte_count":
            request["scene"]["scene_manifest_byte_count"],
        "split": request["scene"]["split"],
        "drive_seed": request["scene"]["drive_seed"],
        "stratum": "general",
        "split_role": "calibration",
        "warmup_blocks": B.WARMUP_BLOCKS_MIN,
        "source_step": 200,
        "episode_id": 1,
        "episode_cluster_id": f"{scenes[0].name}/env0/ep1",
        "cell_id": 2,
        "boundary": {"source_step": 200},
        "goal": {"material_id": "landmark_red"},
        "goal_type": "landmark_red",
        "body_clearance_m": 0.2,
        "clearance_m": 0.3,
    }
    chosen["state_identity_digest"] = B._state_identity_digest(chosen)
    capture = B._build_state_resolution_scene_capture(
        request=request, chosen_state=chosen, rejection_reasons={},
        worker_failure=None, blocks_driven=B.WARMUP_BLOCKS_MIN,
        attempt_trace=[{
            "block_index": B.WARMUP_BLOCKS_MIN,
            "attempts": [{"stratum": "general", "verdict": "SELECT",
                          "reason_key": None}],
        }])

    def persist_pair(req, cap):
        digest = req["state_resolution_scene_request_digest"]
        request_path = B._state_resolution_request_path(out, family, digest)
        capture_path = B._state_resolution_capture_path(out, family, digest)
        B.atomic_json(request_path, req)
        B.atomic_json(capture_path, cap)
        return {
            "scene_id": req["scene"]["scene_id"],
            "state_resolution_scene_request_digest": digest,
            "state_resolution_scene_capture_digest":
                cap["state_resolution_scene_capture_digest"],
            "request_path": str(request_path.relative_to(tmp_path)),
            "request_raw_sha256": B.file_sha256(request_path),
            "request_byte_count": request_path.stat().st_size,
            "capture_path": str(capture_path.relative_to(tmp_path)),
            "capture_raw_sha256": B.file_sha256(capture_path),
            "capture_byte_count": capture_path.stat().st_size,
        }

    provenance = [persist_pair(request, capture)]
    shard = {
        "family": family,
        **bindings,
        "states": [chosen],
        "scene_rejection_reasons": {scenes[0].name: {}},
        "state_resolution_subprocess_transport": {
            "schema": "go2_branch_corpus_v1_2_state_resolution_transport_v1",
            "one_scene_per_subprocess": True,
            "atomic_capture_write_before_native_cleanup": True,
            "return_code_ignored_only_after_valid_capture": True,
            "resume_scope": "MISSING_OR_INVALID_SCENE_CAPTURES_ONLY",
            "resolver_algorithm_digest":
                B.canonical_digest(B.STATE_RESOLUTION_REDUCER_CONTRACT),
            "resolver_cursor_scene_id": scenes[0].name,
            "scene_capture_count": 1,
            "scene_capture_provenance_digest": B.canonical_digest(provenance),
            "candidate_outcomes_loaded": False,
        },
        "state_resolution_scene_capture_provenance": provenance,
    }
    B._validate_state_resolution_transport(shard, expected_pool="scorer_fit")

    # Individually self-bound request/capture bytes advance the cursor after
    # the quota was already full.  The final reducer must reject this even
    # though neither artifact has a broken digest.
    post_request = B._build_state_resolution_scene_request(
        args=args, out=out, scene_dir=scenes[1], scene_ordinal=1,
        found={"general": 0, "safety_enriched": 0,
               "completion_enriched": 0},
        need={"general": 1, "safety_enriched": 0,
              "completion_enriched": 0},
        exclusion={}, family_allow_list=[scene.name for scene in scenes])
    post_capture = B._build_state_resolution_scene_capture(
        request=post_request, chosen_state=None, rejection_reasons={},
        worker_failure="RuntimeError:synthetic-post-quota-failure",
        blocks_driven=0, attempt_trace=[])
    tampered = copy.deepcopy(shard)
    tampered_provenance = provenance + [persist_pair(
        post_request, post_capture)]
    tampered["state_resolution_scene_capture_provenance"] = \
        tampered_provenance
    tampered["scene_rejection_reasons"][scenes[1].name] = {}
    transport = tampered["state_resolution_subprocess_transport"]
    transport["resolver_cursor_scene_id"] = scenes[1].name
    transport["scene_capture_count"] = 2
    transport["scene_capture_provenance_digest"] = \
        B.canonical_digest(tampered_provenance)
    with pytest.raises(
        RuntimeError, match="dynamic quota prefix changed|first full quota"
    ):
        B._validate_state_resolution_transport(
            tampered, expected_pool="scorer_fit")


def _old_stratum(count: int, *, completion: bool = False):
    rows = [{
        "scene_id": f"scene-{index:03d}",
        "first_eligible_block": 40 + index,
        "continuous_geodesic_m": 0.5 if completion else 3.0,
        "abs_bearing_rad": 0.2,
        "graph_hops_diagnostic": 0 if completion else 3,
        "body_clearance_m": 0.2,
    } for index in range(count)]
    return {
        "required_distinct_scenes": 5,
        "eligible_distinct_scenes": count,
        "verdict": "PASS" if count >= 5 else "FAIL",
        "distributions": {},
        "scene_evidence": rows,
    }


def _old_receipt():
    families = []
    for family in B.STATE_SELECTOR.REQUIRED_FAMILIES:
        completion_count = 0 if family == B.REACHABILITY_REDRIVE_FAMILY else 5
        families.append({
            "family": family,
            "allowed_scene_count": (182 if family == B.REACHABILITY_REDRIVE_FAMILY
                                    else 10),
            "scanned_scene_count": (182 if family == B.REACHABILITY_REDRIVE_FAMILY
                                    else 10),
            "all_allowed_scenes_scanned": True,
            "verdict": "FAIL" if completion_count == 0 else "PASS",
            "strata": {
                "general": _old_stratum(5),
                "safety_enriched": _old_stratum(5),
                "completion_enriched": _old_stratum(
                    completion_count, completion=True),
            },
            "rejection_counts": {},
        })
    return {"families": families}


def _task(index: int):
    payload = {
        "family": B.REACHABILITY_REDRIVE_FAMILY,
        "scene_id": f"small-{index:03d}",
        "scene_task_digest": f"{index + 1:064x}",
    }
    return payload


def _evidence(index: int, *, distance: float = 0.8):
    status = {
        "task_completed": False,
        "goal_claimed": False,
        "terminated": False,
        "truncated": False,
    }
    vector = B.STATE_SELECTOR.completion_rotation_eligibility_vector(
        graph_hops=1, reachable=True,
        continuous_geodesic_m=distance, bearing_body_rad=0.1,
        task_status=status, previous_applied_command=[0.0, 0.0, 0.0])
    rotations = vector["rotations"]
    passing = [row for row in rotations if row["eligible"]]
    assert passing
    return {
        "family": B.REACHABILITY_REDRIVE_FAMILY,
        "scene_id": f"small-{index:03d}",
        "stratum": "completion_enriched",
        "first_eligible_block": 40,
        "source_step": 200,
        "boundary": {"source_step": 200, "boundary": "synthetic"},
        "cell_id": index,
        "episode_id": 1,
        "episode_cluster_id": f"small-{index:03d}/env0/ep1",
        "goal_landmark_id": f"goal-{index:03d}",
        "goal_landmark_cell": index + 1,
        "goal_material_id": "landmark_red",
        "continuous_geodesic_m": distance,
        "completion_radius_m": B.STATE_SELECTOR.COMPLETION_RADIUS_M,
        "continuous_geodesic_gap_m": max(
            distance - B.STATE_SELECTOR.COMPLETION_RADIUS_M, 0.0),
        "abs_bearing_rad": 0.1,
        "bearing_body_rad": 0.1,
        "range_m": distance,
        "goal_landmark_xy_m": [1.0, 2.0],
        "graph_hops_diagnostic": 1,
        "body_clearance_m": 0.2,
        "clearance_m": 0.3,
        "previous_applied_command": [0.0, 0.0, 0.0],
        "allocation_rotation_evidence": rotations,
        "completion_rotation_eligibility_vector": vector,
        "eligible_rotation_indices": [
            row["candidate_rotation_index"] for row in passing],
        "passes_any_allowed_allocation": True,
        "passes_every_allowed_allocation": len(passing) == 12,
        "eligible_designated_goal_count_at_first_eligible_snapshot": 1,
        "admitted_by_horizon_reachability_amendment": distance > 0.75,
        "snapshot_task_status": status,
    }


def _shard(index: int, *, include=True):
    task = _task(index)
    return B._build_reachability_scene_shard(
        task=task,
        predecessor_shard_digest=f"{index + 1000:064x}",
        scene_result={
            "family": B.REACHABILITY_REDRIVE_FAMILY,
            "scene_id": task["scene_id"],
            "completion_scene_evidence": [_evidence(index)] if include else [],
            "rejection_counts": {} if include else {
                "completion_gap_exceeds_every_allowed_subset_l_max": 1},
        },
        source=_source(), runtime_s=1.0)


def test_cached_seven_family_rows_are_reclassified_without_mutation():
    predecessor = _old_receipt()
    frozen = copy.deepcopy(predecessor)
    receipt = B.build_reachability_feasibility_receipt(
        predecessor_receipt=predecessor,
        small_scene_shards=[_shard(index) for index in range(182)],
        source=_source())
    assert predecessor == frozen
    reused = [row for row in receipt["families"]
              if row["family"] != B.REACHABILITY_REDRIVE_FAMILY]
    assert len(reused) == 7
    assert all(row["provenance"] == "REUSED_FROZEN_1284_SCENE_CENSUS"
               for row in reused)
    assert receipt["reuse_policy"]["unrelated_family_redrives"] == 0


def test_reachability_receipt_is_deterministic_and_binds_lineage():
    shards = [_shard(index) for index in range(182)]
    first = B.build_reachability_feasibility_receipt(
        predecessor_receipt=_old_receipt(), small_scene_shards=shards,
        source=_source())
    second = B.build_reachability_feasibility_receipt(
        predecessor_receipt=_old_receipt(), small_scene_shards=list(shards),
        source=_source())
    assert first == second
    assert first["status"] == B.REACHABILITY_FEASIBILITY_PASS_STATUS
    assert first["frozen_predecessor"]["scene_shard_count"] == 1284
    assert first["small_scene_shard_lineage_digest"] == B.canonical_digest(
        first["small_scene_shard_lineage"])
    B._validate_reachability_feasibility_receipt(
        first, predecessor_receipt=_old_receipt(),
        small_scene_shards=shards, source=_source())
    B.STATE_SELECTOR.validate_state_selector_feasibility_receipt(
        first,
        expected_source_commit=_source()["source_repository_commit"],
        expected_successor_selection_digest=B.selection_digest(),
        expected_clean_source_binding_digest=B.canonical_digest(_source()),
        expected_bound_implementations_digest=
            _source()["bound_implementations_digest"],
        predecessor_receipt=_old_receipt(), small_scene_shards=shards)
    B.STATE_SELECTOR.validate_state_selector_feasibility_receipt(
        first,
        expected_source_commit="a" * 40,
        expected_successor_selection_digest=B.selection_digest(),
        expected_clean_source_binding_digest=B.canonical_digest(_source()),
        expected_bound_implementations_digest="b" * 64,
        predecessor_receipt=_old_receipt(), small_scene_shards=shards,
    )


@pytest.mark.parametrize("deleted_key", [
    "small_scene_shard_lineage", "clean_source_binding_digest",
    "bound_implementations_digest",
])
def test_central_feasibility_validator_rejects_resigned_missing_fields(
        deleted_key):
    shards = [_shard(index) for index in range(182)]
    receipt = B.build_reachability_feasibility_receipt(
        predecessor_receipt=_old_receipt(), small_scene_shards=shards,
        source=_source())
    del receipt[deleted_key]
    tampered = _resign_feasibility(receipt)
    with pytest.raises(
        B.STATE_SELECTOR.StateSelectorAmendmentError,
        match="key surface changed",
    ):
        B.STATE_SELECTOR.validate_state_selector_feasibility_receipt(
            tampered, predecessor_receipt=_old_receipt(),
            small_scene_shards=shards)


def test_central_feasibility_validator_rejects_resigned_shard_outcome_surface():
    shards = [_shard(index) for index in range(182)]
    receipt = B.build_reachability_feasibility_receipt(
        predecessor_receipt=_old_receipt(), small_scene_shards=shards,
        source=_source())
    changed = copy.deepcopy(shards)
    changed[0]["branches_attempted"] = 1
    changed[0]["candidate_outcome"] = {"completion": True}
    changed[0]["state_selector_reachability_scene_shard_digest"] = \
        B.canonical_digest({
            key: value for key, value in changed[0].items()
            if key != "state_selector_reachability_scene_shard_digest"})
    with pytest.raises(
        B.STATE_SELECTOR.StateSelectorAmendmentError,
        match="shard binding is malformed",
    ):
        B.STATE_SELECTOR.validate_state_selector_feasibility_receipt(
            receipt, predecessor_receipt=_old_receipt(),
            small_scene_shards=changed)


def test_central_feasibility_validator_rejects_resigned_extra_evidence_field():
    shards = [_shard(index) for index in range(182)]
    receipt = B.build_reachability_feasibility_receipt(
        predecessor_receipt=_old_receipt(), small_scene_shards=shards,
        source=_source())
    changed = copy.deepcopy(shards)
    changed[0]["scene_result"]["completion_scene_evidence"][0][
        "realised_completion"] = True
    changed[0]["state_selector_reachability_scene_shard_digest"] = \
        B.canonical_digest({
            key: value for key, value in changed[0].items()
            if key != "state_selector_reachability_scene_shard_digest"})
    with pytest.raises(
        B.STATE_SELECTOR.StateSelectorAmendmentError,
        match="evidence key surface changed",
    ):
        B.STATE_SELECTOR.validate_state_selector_feasibility_receipt(
            receipt, predecessor_receipt=_old_receipt(),
            small_scene_shards=changed)


def test_live_small_shards_bind_frozen_task_and_predecessor_lineage(
        tmp_path, monkeypatch):
    shards = [_shard(index) for index in range(182)]
    receipt = B.build_reachability_feasibility_receipt(
        predecessor_receipt=_old_receipt(), small_scene_shards=shards,
        source=_source())
    expected_tasks = {shard["task"]["scene_id"]: shard["task"]
                      for shard in shards}
    expected_predecessors = {
        shard["task"]["scene_id"]:
            shard["frozen_predecessor_scene_shard_digest"]
        for shard in shards}
    monkeypatch.setattr(
        B.STATE_SELECTOR, "_load_frozen_failed_census_tasks",
        lambda *_args, **_kwargs: (expected_tasks, expected_predecessors))
    root = tmp_path / (
        ".generated/go2_branch_corpus_v1_2/scorer_fit/"
        "state_selector_reachability_feasibility_scene_shards_v2/"
        "small_enclosed_maze")
    root.mkdir(parents=True)
    for shard in shards:
        (root / f"{shard['task']['scene_task_digest']}.json").write_text(
            json.dumps(shard))
    loaded = B.STATE_SELECTOR._load_live_small_reachability_shards(
        receipt, tmp_path, _old_receipt())
    assert len(loaded) == 182

    changed = copy.deepcopy(shards[0])
    changed["task"]["scene_id"] = "fabricated-scene"
    changed["state_selector_reachability_scene_shard_digest"] = \
        B.canonical_digest({
            key: value for key, value in changed.items()
            if key != "state_selector_reachability_scene_shard_digest"})
    (root / f"{shards[0]['task']['scene_task_digest']}.json").write_text(
        json.dumps(changed))
    with pytest.raises(
        B.STATE_SELECTOR.StateSelectorAmendmentError,
        match="self binding failed",
    ):
        B.STATE_SELECTOR._load_live_small_reachability_shards(
            receipt, tmp_path, _old_receipt())

    (root / f"{shards[0]['task']['scene_task_digest']}.json").write_text(
        json.dumps(shards[0]))
    changed_receipt = copy.deepcopy(receipt)
    changed_receipt["small_scene_shard_lineage"][0][
        "predecessor_scene_shard_digest"] = "f" * 64
    changed_receipt["small_scene_shard_lineage_digest"] = B.canonical_digest(
        changed_receipt["small_scene_shard_lineage"])
    with pytest.raises(
        B.STATE_SELECTOR.StateSelectorAmendmentError,
        match="identities are not unique",
    ):
        B.STATE_SELECTOR._load_live_small_reachability_shards(
            changed_receipt, tmp_path, _old_receipt())


def test_small_family_fails_closed_below_five_distinct_eligible_scenes():
    shards = [_shard(index, include=index < 4) for index in range(182)]
    receipt = B.build_reachability_feasibility_receipt(
        predecessor_receipt=_old_receipt(), small_scene_shards=shards,
        source=_source())
    small = next(row for row in receipt["families"]
                 if row["family"] == B.REACHABILITY_REDRIVE_FAMILY)
    assert receipt["status"] == (
        "FAIL_OUTCOME_FREE_REACHABILITY_SELECTOR_FEASIBILITY")
    assert small["verdict"] == "FAIL"
    assert small["strata"]["completion_enriched"][
        "eligible_distinct_scenes"] == 4


def test_rotation_evidence_uses_exact_unchanged_allocation_catalogue():
    rows = B._completion_rotation_evidence(
        graph_hops=0, distance=0.8, bearing=0.0,
        task_status={
            "task_completed": False, "goal_claimed": False,
            "terminated": False, "truncated": False},
        previous_applied_command=[0.2, 0.0, -0.1])
    assert len(rows) == 12
    assert [row["candidate_indices"] for row in rows] == [
        list(block) for block in B.ALLOC.ROTATION_BLOCKS]
    assert all(row["candidate_path_lengths_m"] for row in rows)
    assert all(row["rejection_reasons"] == [] for row in rows)


def test_raw_v2_rotation_vector_uses_candidate_rotation_index_end_to_end():
    evidence = _evidence(0)
    rotations = evidence["allocation_rotation_evidence"]
    assert [row["candidate_rotation_index"] for row in rotations] == \
        list(range(12))
    assert all("rotation_index" not in row for row in rotations)

    task = _task(0)
    payload = B._build_reachability_scene_shard(
        task=task, predecessor_shard_digest="c" * 64,
        scene_result={
            "family": B.REACHABILITY_REDRIVE_FAMILY,
            "scene_id": task["scene_id"],
            "completion_scene_evidence": [evidence],
            "rejection_counts": {},
        },
        source=_source(), runtime_s=1.0)
    B._validate_reachability_scene_shard(
        payload, expected_task=task, predecessor_shard_digest="c" * 64,
        source=_source())


def test_reachability_scene_shard_validation_rejects_changed_provenance():
    task = _task(0)
    payload = B._build_reachability_scene_shard(
        task=task, predecessor_shard_digest="c" * 64,
        scene_result={
            "family": B.REACHABILITY_REDRIVE_FAMILY,
            "scene_id": task["scene_id"],
            "completion_scene_evidence": [_evidence(0)],
            "rejection_counts": {},
        },
        source=_source(), runtime_s=1.0)
    B._validate_reachability_scene_shard(
        payload, expected_task=task, predecessor_shard_digest="c" * 64,
        source=_source())
    changed = copy.deepcopy(payload)
    changed["frozen_predecessor_scene_shard_digest"] = "d" * 64
    changed["state_selector_reachability_scene_shard_digest"] = \
        B.canonical_digest({
            key: value for key, value in changed.items()
            if key != "state_selector_reachability_scene_shard_digest"})
    with pytest.raises(RuntimeError, match="binding failed"):
        B._validate_reachability_scene_shard(
            changed, expected_task=task, predecessor_shard_digest="c" * 64,
            source=_source())


def test_reachability_scene_shard_from_pre_fix_source_is_not_resumed(tmp_path):
    task = _task(0)
    original_source = _source()
    payload = B._build_reachability_scene_shard(
        task=task, predecessor_shard_digest="c" * 64,
        scene_result={
            "family": B.REACHABILITY_REDRIVE_FAMILY,
            "scene_id": task["scene_id"],
            "completion_scene_evidence": [_evidence(0)],
            "rejection_counts": {},
        },
        source=original_source, runtime_s=1.0)
    path = tmp_path / "pre-fix-reachability-shard.json"
    B.atomic_json(path, payload)
    successor_source = {
        **original_source,
        "source_repository_commit": "d" * 40,
        "bound_implementations_digest": "e" * 64,
    }
    assert B._load_reachability_scene_shard(
        path, expected_task=task, predecessor_shard_digest="c" * 64,
        source=successor_source) is None
    # The parent stage treats this exact state as invalid, preserves the old
    # file under invalid_attempts, and regenerates only that missing source-
    # bound shard; it never accepts it as a zero-new resume.
    stage_source = inspect.getsource(B.stage_selector_reachability_feasibility)
    assert stage_source.index("if path.exists():") < stage_source.index(
        "_run_reachability_scene_subprocess(args, task)")
    assert "_preserve_invalid(path, out, \"reachability-scene-invalid\")" \
        in stage_source

    changed = copy.deepcopy(payload)
    changed["state_selector_amendment_digest"] = "f" * 64
    changed["state_selector_reachability_scene_shard_digest"] = \
        B.canonical_digest({
            key: value for key, value in changed.items()
            if key != "state_selector_reachability_scene_shard_digest"})
    with pytest.raises(RuntimeError, match="binding failed"):
        B._validate_reachability_scene_shard(
            changed, expected_task=task, predecessor_shard_digest="c" * 64,
            source=_source())

    changed = copy.deepcopy(payload)
    changed["successor_selection_digest"] = "e" * 64
    changed["state_selector_reachability_scene_shard_digest"] = \
        B.canonical_digest({
            key: value for key, value in changed.items()
            if key != "state_selector_reachability_scene_shard_digest"})
    with pytest.raises(RuntimeError, match="binding failed"):
        B._validate_reachability_scene_shard(
            changed, expected_task=task, predecessor_shard_digest="c" * 64,
            source=_source())


def _joint_state(index, *, stratum="general", family="fixed-family"):
    state = {
        "state_id": f"fixed-{index:03d}",
        "state_identity_digest": f"{index + 3000:064x}",
        "family": family,
        "stratum": stratum,
        "split_role": "fit",
        "goal_type": "landmark_red",
        "scene_id": f"fixed-scene-{index:03d}",
    }
    return state


def _joint_candidate(index):
    return {
        "state_id": "deferred",
        "state_identity_digest": f"{index + 5000:064x}",
        "family": B.REACHABILITY_REDRIVE_FAMILY,
        "stratum": "completion_enriched",
        "split_role": "deferred",
        "goal_type": "landmark_red",
        "scene_id": f"small-scene-{index:03d}",
        "warmup_blocks": 40,
        "goal": {
            "start_geodesic_m": 0.8 + index * 0.01,
            "landmark_id": f"goal-{index:03d}",
            "landmark_cell": index,
            "graph_edges": 1,
        },
    }


def _fake_allocation(states, serial):
    assignments = [{
        "state_id": row["state_id"],
        "state_identity_digest": row["state_identity_digest"],
        "family": row["family"],
        "stratum": row["stratum"],
        "split_role": row["split_role"],
        "goal_type": row["goal_type"],
        "rotation_index": 0,
        "candidate_indices": list(B.ALLOC.ROTATION_BLOCKS[0]),
    } for row in states]
    return {
        "assignments": assignments,
        "allocation_manifest_digest": f"{serial + 7000:064x}",
    }


def test_serial_joint_search_is_tombstoned_without_allocator_use(monkeypatch):
    fixed = [_joint_state(index) for index in range(115)]
    candidates = [_joint_candidate(index) for index in range(6)]
    # A large d0 must not change the predecessor's cross-scene lexical order.
    candidates[0]["goal"]["start_geodesic_m"] = 9.0
    allocator_calls = []
    monkeypatch.setattr(
        B.ALLOC, "build_allocation_manifest",
        lambda *_args, **_kwargs: allocator_calls.append(True))
    with pytest.raises(
            RuntimeError, match="superseded.*bounded parallel coordinator"):
        B.select_small_completion_combination(
            fixed_states=fixed, raw_candidates=candidates,
            preserved_vectors={})
    assert allocator_calls == []


def test_cursor_restriction_never_revisits_earlier_completion_only_scene():
    rows = [
        {"scene_id": "small-scene-001", "first_eligible_block": 40},
        {"scene_id": "small-scene-003", "first_eligible_block": 41},
        {"scene_id": "small-scene-005", "first_eligible_block": 42},
        {"scene_id": "small-scene-006", "first_eligible_block": 43},
    ]
    retained = B._cursor_restricted_completion_rows(
        rows, resolver_cursor_scene_id="small-scene-003",
        excluded_scene_ids={"small-scene-006"})
    assert [row["scene_id"] for row in retained] == ["small-scene-005"]


def test_serial_joint_search_tombstone_precedes_pool_arithmetic():
    fixed = [_joint_state(index) for index in range(115)]
    candidates = [_joint_candidate(index) for index in range(4)]
    with pytest.raises(
            RuntimeError, match="superseded.*bounded parallel coordinator"):
        B.select_small_completion_combination(
            fixed_states=fixed, raw_candidates=candidates,
            preserved_vectors={}, resolver_cursor_scene_id="earlier")


def test_terminal_joint_search_failure_is_nonoverwriting_and_pre_genesis(
        tmp_path, monkeypatch):
    monkeypatch.setattr(B, "OUT_ROOT", tmp_path)
    out = tmp_path / "scorer_fit"
    out.mkdir(parents=True)
    launch = {
        "source_repository_commit": "a" * 40,
        "clean_source_launch_receipt_digest": "b" * 64,
        "state_selector_feasibility_receipt_digest": "c" * 64,
    }
    monkeypatch.setattr(B, "_load_clean_source_launch_receipt",
                        lambda: dict(launch))
    error = B.SmallCompletionJointSearchInfeasible(
        "cursor-restricted small completion pool has fewer than five scenes",
        attempt_count=0, allocator_infeasible_count=0,
        candidate_scene_ids=["small-scene-004"])
    first = B._issue_small_completion_search_failure(
        out=out, error=error, resolver_cursor_scene_id="small-scene-003")
    second = B._issue_small_completion_search_failure(
        out=out, error=error, resolver_cursor_scene_id="small-scene-003")
    assert first == second
    fixed = [_joint_state(index) for index in range(115)]
    live_candidates = [_joint_candidate(4)]
    monkeypatch.setattr(
        B, "_live_small_completion_search_inputs",
        lambda **_kwargs: (
            "small-scene-003", fixed, live_candidates, {}))
    monkeypatch.setattr(
        B.V1, "_load_shared",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("terminal failure resumed Genesis")))
    args = SimpleNamespace(
        pool="scorer_fit", family=B.REACHABILITY_REDRIVE_FAMILY,
        backend="cpu")
    with pytest.raises(RuntimeError, match="serial small-family resolution"):
        B.resolve_states(args)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cursor_restricted_candidate_scene_count", True, "malformed"),
        ("combination_attempt_count", 1, "reason/arithmetic"),
        ("allocator_infeasible_combination_count", 1, "reason/arithmetic"),
    ],
)
def test_terminal_failure_rejects_resigned_malformed_arithmetic(
        tmp_path, monkeypatch, field, value, message):
    monkeypatch.setattr(B, "OUT_ROOT", tmp_path)
    out = tmp_path / "scorer_fit"
    out.mkdir(parents=True)
    launch = {
        "source_repository_commit": "a" * 40,
        "clean_source_launch_receipt_digest": "b" * 64,
        "state_selector_feasibility_receipt_digest": "c" * 64,
    }
    monkeypatch.setattr(B, "_load_clean_source_launch_receipt",
                        lambda: dict(launch))
    error = B.SmallCompletionJointSearchInfeasible(
        "cursor-restricted small completion pool has fewer than five scenes",
        attempt_count=0, allocator_infeasible_count=0,
        candidate_scene_ids=["small-scene-004"])
    payload = B._issue_small_completion_search_failure(
        out=out, error=error, resolver_cursor_scene_id="small-scene-003")
    payload[field] = value
    payload["small_completion_joint_search_failure_digest"] = \
        B.canonical_digest({
            key: item for key, item in payload.items()
            if key != "small_completion_joint_search_failure_digest"
        })
    (out / B.SMALL_COMPLETION_SEARCH_FAILURE_NAME).write_text(
        json.dumps(payload))
    with pytest.raises(RuntimeError, match=message):
        B._load_small_completion_search_failure(
            out, args=SimpleNamespace(
                pool="scorer_fit", family=B.REACHABILITY_REDRIVE_FAMILY,
                backend="cpu"))


def test_terminal_exhaustion_cannot_reenter_legacy_serial_selector(monkeypatch):
    fixed = [_joint_state(index) for index in range(115)]
    candidates = [_joint_candidate(index) for index in range(6)]
    calls = []
    monkeypatch.setattr(
        B.ALLOC, "build_allocation_manifest",
        lambda projection, **kwargs: (
            calls.append(1) or _fake_allocation(projection, len(calls))))
    with pytest.raises(
            RuntimeError, match="superseded.*bounded parallel coordinator"):
        B.select_small_completion_combination(
            fixed_states=fixed, raw_candidates=candidates,
            preserved_vectors={}, resolver_cursor_scene_id="earlier")
    assert calls == []


def test_joint_search_fails_after_exhausting_all_combinations(monkeypatch):
    fixed = [_joint_state(index) for index in range(115)]
    candidates = [_joint_candidate(index) for index in range(6)]
    calls = []
    monkeypatch.setattr(
        B.ALLOC, "build_allocation_manifest",
        lambda projection, **kwargs: (
            calls.append(1) or _fake_allocation(projection, len(calls))))
    with pytest.raises(
            RuntimeError, match="superseded.*bounded parallel coordinator"):
        B.select_small_completion_combination(
            fixed_states=fixed, raw_candidates=candidates,
            preserved_vectors={})
    assert calls == []


def test_joint_search_continues_after_allocator_infeasible_combination(monkeypatch):
    fixed = [_joint_state(index) for index in range(115)]
    candidates = [_joint_candidate(index) for index in range(6)]
    calls = []

    def allocate(projection, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise B.ALLOC.CandidateAllocationInfeasible(
                "synthetic infeasible margins")
        return _fake_allocation(projection, len(calls))

    monkeypatch.setattr(B.ALLOC, "build_allocation_manifest", allocate)
    with pytest.raises(
            RuntimeError, match="superseded.*bounded parallel coordinator"):
        B.select_small_completion_combination(
            fixed_states=fixed, raw_candidates=candidates,
            preserved_vectors={})
    assert calls == []


def test_exact_completion_gate_requires_all_forty(monkeypatch):
    vector = B.STATE_SELECTOR.completion_rotation_eligibility_vector(
        graph_hops=1, reachable=True, continuous_geodesic_m=0.8,
        bearing_body_rad=0.0,
        task_status={"task_completed": False, "goal_claimed": False,
                     "terminated": False, "truncated": False},
        previous_applied_command=[0.0, 0.0, 0.0])
    states = []
    for index in range(40):
        state = _joint_state(index, stratum="completion_enriched")
        state["completion_rotation_eligibility_vector"] = vector
        states.append(state)
    allocation = _fake_allocation(states, 1)
    assert B._all_completion_masks_pass(
        states=states, allocation=allocation, preserved_vectors={}) is True
    with pytest.raises(RuntimeError, match="expected 40"):
        B._all_completion_masks_pass(
            states=states[:-1],
            allocation=_fake_allocation(states[:-1], 2),
            preserved_vectors={})


def test_exact_completion_gate_distinguishes_ineligible_from_malformed():
    eligible = B.STATE_SELECTOR.completion_rotation_eligibility_vector(
        graph_hops=1, reachable=True, continuous_geodesic_m=0.8,
        bearing_body_rad=0.0,
        task_status={"task_completed": False, "goal_claimed": False,
                     "terminated": False, "truncated": False},
        previous_applied_command=[0.0, 0.0, 0.0])
    ineligible = B.STATE_SELECTOR.completion_rotation_eligibility_vector(
        graph_hops=1, reachable=True, continuous_geodesic_m=10.0,
        bearing_body_rad=0.0,
        task_status={"task_completed": False, "goal_claimed": False,
                     "terminated": False, "truncated": False},
        previous_applied_command=[0.0, 0.0, 0.0])
    states = []
    for index in range(40):
        state = _joint_state(index, stratum="completion_enriched")
        state["completion_rotation_eligibility_vector"] = (
            ineligible if index == 0 else eligible)
        states.append(state)
    allocation = _fake_allocation(states, 1)
    assert B._all_completion_masks_pass(
        states=states, allocation=allocation, preserved_vectors={}) is False

    changed = copy.deepcopy(states)
    changed[0]["completion_rotation_eligibility_vector"] = copy.deepcopy(eligible)
    changed[0]["completion_rotation_eligibility_vector"]["rotations"][0][
        "l_max_m"] += 1.0
    with pytest.raises(RuntimeError, match="failed reconstruction"):
        B._all_completion_masks_pass(
            states=changed, allocation=allocation, preserved_vectors={})


def test_phase2_projection_has_exact_forty_allocated_rows():
    vector = B.STATE_SELECTOR.completion_rotation_eligibility_vector(
        graph_hops=1, reachable=True, continuous_geodesic_m=0.8,
        bearing_body_rad=0.0,
        task_status={"task_completed": False, "goal_claimed": False,
                     "terminated": False, "truncated": False},
        previous_applied_command=[0.0, 0.0, 0.0])
    states = []
    for index in range(40):
        state = _joint_state(index, stratum="completion_enriched")
        state["completion_rotation_eligibility_vector"] = vector
        states.append(state)
    rows = B._completion_states_for_phase2(
        allocation=_fake_allocation(states, 1), states=states,
        preserved_vectors={})
    assert len(rows) == 40
    assert all(row["candidate_indices"] == list(B.ALLOC.ROTATION_BLOCKS[0])
               for row in rows)


def test_final_assignment_digest_detects_masks_different_from_search():
    states = [_joint_state(index) for index in range(120)]
    first = _fake_allocation(states, 1)
    changed = copy.deepcopy(first)
    changed["assignments"][0]["rotation_index"] = 1
    changed["assignments"][0]["candidate_indices"] = list(
        B.ALLOC.ROTATION_BLOCKS[1])
    assert B._allocation_assignment_set_digest(first) != \
        B._allocation_assignment_set_digest(changed)


def _joint_manifest_fixture():
    states = [_joint_state(index) for index in range(105)]
    for index in range(10):
        state = _joint_state(
            200 + index,
            stratum="general" if index < 5 else "safety_enriched",
            family=B.REACHABILITY_REDRIVE_FAMILY)
        state["scene_id"] = f"small-fixed-{index:03d}"
        states.append(state)
    completion = []
    for index in range(5):
        state = _joint_candidate(300 + index)
        state["state_id"] = f"small-completion-{index}"
        state["state_identity_digest"] = f"{9000 + index:064x}"
        state["scene_id"] = f"small-pool-{index:03d}"
        state["split_role"] = "calibration" if index == 0 else "fit"
        completion.append(state)
    states.extend(completion)
    allocation = _fake_allocation(states, 1)
    assignment_digest = B._allocation_assignment_set_digest(allocation)
    pool_scenes = [state["scene_id"] for state in completion]
    receipt = {
        "status": "PASS_FIRST_LEXICOGRAPHIC_EXACT_MASK_COMBINATION",
        "combination_attempt_count": 1,
        "allocator_infeasible_combination_count": 0,
        "candidate_pool_count": 5,
        "candidate_pool_scene_ids": pool_scenes,
        "candidate_pool_scene_ids_digest": B.canonical_digest(pool_scenes),
        "resolver_cursor_scene_id": "small-fixed-009",
        "selected_scene_ids": pool_scenes,
        "provisional_allocation_manifest_digest": "a" * 64,
        "provisional_candidate_assignment_set_digest": assignment_digest,
        "candidate_outcomes_consumed": False,
        "final_candidate_assignment_set_digest": assignment_digest,
        "final_masks_equal_searched_masks": True,
    }
    return {"states": states,
            "small_completion_joint_allocation_search": receipt}, allocation


def test_manifest_rejects_legacy_serial_joint_receipt_without_replay(
        monkeypatch):
    manifest, allocation = _joint_manifest_fixture()
    monkeypatch.setattr(
        B, "select_small_completion_combination", lambda **_kwargs:
        pytest.fail("legacy serial selector was replayed"))
    with pytest.raises(RuntimeError, match="self digest mismatch"):
        B._validate_small_completion_joint_search_receipt(
            manifest=manifest, allocation=allocation, replay_live=True)


def _redrive_pair():
    vector = B.STATE_SELECTOR.completion_rotation_eligibility_vector(
        graph_hops=1, reachable=True, continuous_geodesic_m=0.8,
        bearing_body_rad=0.0,
        task_status={"task_completed": False, "goal_claimed": False,
                     "terminated": False, "truncated": False},
        previous_applied_command=[0.1, 0.0, -0.1])
    status = {
        "task_completed": False, "goal_claimed": False,
        "terminated": False, "truncated": False,
        "production_designated_goal_claim_evidence": {"bound": True},
    }
    boundary = {"source_step": 200, "digest": "boundary"}
    goal = {"landmark_id": "goal", "landmark_cell": 7,
            "material_id": "landmark_red", "graph_edges": 1,
            "start_geodesic_m": 0.8, "bearing_body_rad": 0.0,
            "range_m": 0.8, "landmark_xy_m": [1.0, 2.0]}
    entry = {
        "state_identity_digest": "f" * 64,
        "stratum": "completion_enriched", "source_step": 200,
        "episode_id": 3, "cell_id": 4, "boundary": boundary,
        "goal": goal, "body_clearance_m": 0.2, "clearance_m": 0.3,
    }
    record = {
        "boundary": boundary, "cell_id": 4, "goal": goal,
        "body_clearance_m": 0.2, "clearance_m": 0.3,
        "completion_rotation_eligibility_vector": vector,
        "snapshot_task_status": status,
        "previous_applied_command": [0.1, 0.0, -0.1],
    }
    ctx = SimpleNamespace(runner=SimpleNamespace(
        episode_states=[SimpleNamespace(episode_id=3)]))
    return entry, record, ctx, vector, status


def test_successor_completion_redrive_uses_v2_bound_evidence(monkeypatch):
    entry, record, ctx, vector, status = _redrive_pair()
    entry.update({
        "completion_rotation_eligibility_vector": vector,
        "snapshot_task_status": status,
        "previous_applied_command": [0.1, 0.0, -0.1],
    })
    monkeypatch.setattr(B, "_preserved_states_by_digest", lambda: {})
    assert B._redrive_mismatch(entry, record, ctx) is None
    changed = copy.deepcopy(record)
    changed["previous_applied_command"][0] = 0.2
    assert "previous_applied_command" in B._redrive_mismatch(
        entry, changed, ctx)


def test_preserved_completion_redrive_requires_bound_v1_membership(monkeypatch):
    entry, record, ctx, _vector, _status = _redrive_pair()
    monkeypatch.setattr(
        B, "_preserved_states_by_digest",
        lambda: {entry["state_identity_digest"]: {"bound": True}})
    assert B._redrive_mismatch(entry, record, ctx) is None
    monkeypatch.setattr(B, "_preserved_states_by_digest", lambda: {})
    assert "preserved_completion_identity" in B._redrive_mismatch(
        entry, record, ctx)
