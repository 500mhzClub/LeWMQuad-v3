"""Source-only tests for the frozen scorer-fit oracle-v1.3 contract."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from lewm.oracle import go2_scorer_fit_oracle_v1_3_contract as C


ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC = ROOT / C.DIAGNOSTIC_PATH


def _diagnostic() -> dict:
    return json.loads(DIAGNOSTIC.read_text())


def test_exact_eighteen_replay_allowlist_matches_committed_diagnostic():
    diagnostic = _diagnostic()
    expected = [{
        "branch_identity_digest": row["branch_identity_digest"],
        "state_identity_digest": row["state_identity_digest"],
        "state_id": row["state_id"],
        "scene_id": row["scene_id"],
        "split_role": row["split_role"],
        "candidate_index": row["candidate_index"],
        "candidate": row["candidate"],
        "diagnostic_category": row["primary_category"],
    } for row in diagnostic["failure_inventory"]]
    actual = [asdict(row) for row in C.FAILED_BRANCH_IDENTITIES]
    assert actual == expected
    assert len(C.FAILED_BRANCH_ALLOWLIST) == 18
    assert sum(row.split_role == "fit" for row in C.FAILED_BRANCH_IDENTITIES) == 6
    assert sum(row.split_role == "calibration"
               for row in C.FAILED_BRANCH_IDENTITIES) == 12
    assert sum(row.diagnostic_category == "OFF_NAVIGABLE_GRAPH_OUTCOME"
               for row in C.FAILED_BRANCH_IDENTITIES) == 4


def test_historical_calibration_disposition_matches_all_twenty_four_identities():
    diagnostic = _diagnostic()
    expected = [{
        "state_identity_digest": row["state_identity_digest"],
        "state_id": row["state_id"],
        "scene_id": row["scene_id"],
        "family": row["family"],
        "stratum": row["stratum"],
    } for row in diagnostic["identity_verification"]["calibration_identities"]]
    actual = [asdict(row) for row in C.HISTORICAL_CALIBRATION_STATES]
    assert actual == expected
    assert len({row.scene_id for row in C.HISTORICAL_CALIBRATION_STATES}) == 24
    disposition = C.contract()["historical_calibration_disposition"]
    assert disposition["status"] == "DEVELOPMENT_ONLY"
    assert disposition["qualification_eligible"] is False
    assert disposition["discarded"] is False
    assert disposition["replaced"] is False


def test_diagnostic_and_corpus_lineage_are_exactly_bound():
    diagnostic = _diagnostic()
    assert hashlib.sha256(DIAGNOSTIC.read_bytes()).hexdigest() \
        == C.DIAGNOSTIC_FILE_SHA256
    assert diagnostic["audit_digest"] == C.DIAGNOSTIC_AUDIT_DIGEST
    assert diagnostic["corpus"]["corpus_digest"] == C.FROZEN_CORPUS_DIGEST
    assert diagnostic["artifact_bindings"]["state_manifest_digest"] \
        == C.FROZEN_STATE_MANIFEST_DIGEST
    assert diagnostic["artifact_bindings"][
        "full_bank_assignment_manifest_digest"
    ] == C.FROZEN_ASSIGNMENT_MANIFEST_DIGEST


def test_fresh_selector_is_exact_eight_by_three_and_outcome_blind():
    selector = C.FRESH_CALIBRATION_SELECTOR
    assert len(C.FAMILIES) == 8
    assert len(C.STRATA) == 3
    assert selector["count"] == len(C.FAMILIES) * len(C.STRATA) == 24
    assert selector["states_per_family_stratum"] == 1
    assert selector["scene_order"].startswith("ascending lexical")
    assert selector["exclusions"]["all_120_historical_scorer_fit_scenes"] is True
    assert selector["manifest_frozen_before_candidate_branch_generation"] is True
    assert selector["post_selection_replacement"] is False
    assert {"candidate outcome", "oracle label", "latent", "scorer output"} \
        <= set(selector["forbidden_selection_inputs"])


def test_selector_integrity_replacement_is_narrow_one_shot_and_versioned():
    replacement = C.SELECTOR_INTEGRITY_REPLACEMENT
    assert C.contract_digest() == \
        "93532f22a0cbc0e57ccdab3d5c01419cd824bc402d637738c5004eb621c23a89"
    assert replacement["predecessor_authority_digest"] == \
        "aa7c536059017d4eb7953f6ebb42bf0d83f7ca4c6303f4c52d3631637a0fe48f"
    assert replacement["predecessor_source_commit"] == \
        "cfe087c36b2129510f4c75af36c1a2631b59f6ac"
    assert replacement["selector_digest"] == C.fresh_calibration_selector_digest()
    assert replacement["one_shot_selector_attempt"] is True
    assert replacement["selector_attempt_count"] == 1
    assert replacement["same_outcome_blind_selector_required"] is True
    assert replacement["per_scene_process_isolation_required"] is True
    assert replacement["exclusive_per_task_launch_marker_required"] is True
    assert replacement["launch_marker_path_rule"] == \
        "selection_tasks/<task_digest>.launch.json"
    assert replacement[
        "worker_refuses_existing_terminal_result_or_launch_marker"] is True
    assert replacement["resume_only_from_valid_task_result"] is True
    assert replacement["unresolved_child_is_terminal"] is True
    assert replacement["valid_result_survives_nonzero_teardown"] is True
    assert replacement["scene_skip_or_replacement"] is False
    assert replacement["post_selection_replacement"] is False
    authority = replacement["authority"]
    assert authority["selector_recomputation"] is True
    assert authority["fresh_calibration_state_manifest_publication"] is True
    assert all(value is False for key, value in authority.items() if key not in {
        "selector_recomputation", "fresh_calibration_state_manifest_publication",
    })
    continuation = replacement["preserved_predecessor_authority_continuation"]
    assert continuation["replacement_grants_new_candidate_branch_authority"] \
        is False
    assert continuation["predecessor_candidate_branch_authority_preserved"] is True
    assert continuation["continuation_requires_valid_selector_terminal"] is True
    assert continuation["continuation_requires_exact_24_state_manifest"] is True
    assert continuation["fresh_branch_attempts_before_replacement"] == 0
    assert continuation["fresh_branch_budget"] == 288


def test_selector_integrity_incident_and_predecessor_outputs_are_exactly_bound():
    replacement = C.SELECTOR_INTEGRITY_REPLACEMENT
    incident = replacement["incident"]
    assert incident == {
        "stage": "select-calibration",
        "process_id": 1822672,
        "observed_at_local": "2026-08-15T12:45:50+01:00",
        "observed_at_utc": "2026-08-15T11:45:50Z",
        "evidence_source": "kernel journal",
        "signal": "SIGSEGV",
        "fault_address": "0x1088",
        "page_fault_error_code": 4,
        "shared_object": "libc.so.6",
        "shared_object_offset": "0xa00d4",
        "likely_cpu": 8,
        "process_gone": True,
        "phase": "before any fresh-calibration manifest or durable output",
    }
    outputs = replacement["failed_selector_outputs"]
    assert outputs["fresh_calibration_root_exists"] is False
    assert outputs["state_manifest_exists"] is False
    assert outputs["selection_task_count"] == 0
    assert outputs["selection_result_count"] == 0
    assert outputs["fresh_branch_attempt_count"] == 0
    assert outputs["fresh_branch_outcome_count"] == 0
    preserved = replacement["preserved_predecessor_artifacts"]
    assert preserved["equivalence_receipt"]["compared_branch_count"] == 1422
    assert preserved["equivalence_receipt"]["mismatch_count"] == 0
    assert preserved["replay_attempt_markers"]["count"] == 18
    assert preserved["replay_overlays"]["count"] == 18
    assert preserved["replay_overlay_manifest"]["overlay_count"] == 18
    assert preserved["replay_overlay_manifest"]["fit_overlay_count"] == 6
    assert preserved["replay_overlay_manifest"][
        "historical_calibration_overlay_count"] == 12
    assert preserved["mutation_authorised"] is False


def test_selector_integrity_paths_are_closed_under_fresh_calibration_root():
    root = str(C.FRESH_CALIBRATION_ROOT) + "/"
    paths = C.SELECTOR_INTEGRITY_REPLACEMENT["paths"]
    assert paths == {
        "replacement_authority":
            str(C.SELECTOR_INTEGRITY_REPLACEMENT_AUTHORITY_PATH),
        "selection_attempt": str(C.FRESH_SELECTION_ATTEMPT_PATH),
        "selection_tasks_root": str(C.FRESH_SELECTION_TASKS_ROOT),
        "selection_results_root": str(C.FRESH_SELECTION_RESULTS_ROOT),
        "selection_terminal": str(C.FRESH_SELECTION_TERMINAL_PATH),
        "state_manifest": str(C.FRESH_CALIBRATION_STATE_MANIFEST_PATH),
    }
    assert all(path.startswith(root) for path in paths.values())


def test_stage_counts_form_one_complete_training_view_without_old_calibration():
    counts = C.STAGE_COUNTS
    assert counts["legacy_valid_adoptions"] + counts[
        "exact_failed_branch_replays"] == 1440
    assert counts["legacy_valid_fit_adoptions"] + counts[
        "failed_fit_branch_replays"] == counts["complete_fit_rows"] == 1152
    assert counts["fresh_calibration_states"] == 24
    assert counts["fresh_calibration_branches"] == 288
    assert counts["complete_fit_rows"] + counts[
        "complete_fresh_calibration_rows"] == counts["training_view_rows"] == 1440
    assert counts["shared_scorer_training_runs"] == 1
    assert counts["qualification_evaluations"] == 1
    assert counts["final_benchmark_states_generated"] == 0


def test_scorer_budget_and_qualification_are_the_frozen_predecessor_values():
    training = C.SCORER_TRAINING_CONTRACT
    assert training["training"] == {
        "epochs": 60,
        "batch": 64,
        "lr": 0.0003,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "optimiser": "AdamW",
        "seed": 20260811,
    }
    assert training["final_epoch_only"] is True
    assert training["paired_no_latent_baseline_required"] is True
    assert training["shared_scorer_runs"] == 1
    assert C.QUALIFICATION_THRESHOLDS[
        "composite_within_state_pairwise_accuracy_min"] == 0.65
    assert C.QUALIFICATION_THRESHOLDS["no_latent_pairwise_margin_min"] == 0.05
    assert C.QUALIFICATION_THRESHOLDS["failure_is_terminal"] is True


def test_output_authority_is_closed_under_one_versioned_root():
    root = str(C.GENERATED_ROOT) + "/"
    assert all(path.startswith(root) for path in C.OUTPUT_PATHS.values())
    assert C.OUTPUT_PATHS["training_view"] == str(C.TRAINING_VIEW_PATH)
    assert C.OUTPUT_PATHS["horizon_latents_root"] == str(C.HORIZON_LATENTS_ROOT)
    assert "context" not in C.OUTPUT_PATHS
    contract = C.contract()
    assert contract["qualification_pass_authorises_predictor_open_in_this_workflow"] \
        is False
    assert contract["final_200_state_benchmark_authorised"] is False


def test_source_and_test_closures_are_explicit_unique_and_custody_safe():
    paths = C.SOURCE_CLOSURE_PATHS + C.TEST_CLOSURE_PATHS
    assert len(paths) == len(set(paths))
    assert all(not Path(path).is_absolute() for path in paths)
    assert all("sealed" not in Path(path).parts for path in paths)
    assert str(C.PREREGISTRATION_PATH) not in paths


def test_contract_and_preregistration_digests_are_deterministic_and_fail_closed():
    assert C.contract_digest() == C.contract_digest()
    bindings = C.source_bindings(ROOT, paths=(
        "lewm/oracle/go2_branch_oracle_v1_3.py",
        "lewm/oracle/go2_scorer_fit_oracle_v1_3_contract.py",
        "lewm/tests/test_go2_branch_oracle_v1_3.py",
        "lewm/tests/test_go2_scorer_fit_oracle_v1_3_contract.py",
    ))
    artifact = C.build_preregistration(source_bindings_value=bindings)
    assert C.validate_preregistration(
        artifact, root=ROOT, require_complete_source_closure=False
    ) == artifact
    changed = copy.deepcopy(artifact)
    changed["branch_execution_started"] = True
    with pytest.raises(RuntimeError):
        C.validate_preregistration(
            changed, root=ROOT, require_complete_source_closure=False
        )


def test_original_preregistration_remains_immutable_predecessor_evidence():
    path = ROOT / C.ORIGINAL_PREREGISTRATION_PATH
    artifact = json.loads(path.read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == \
        C.ORIGINAL_PREREGISTRATION_BINDING["raw_sha256"]
    assert path.stat().st_size == C.ORIGINAL_PREREGISTRATION_BINDING["byte_count"]
    assert C.validate_original_preregistration(artifact, root=ROOT) == artifact


def test_contract_freezes_endpoint_rule_and_never_authorises_missing_labels():
    contract = C.contract()
    oracle = contract["oracle_contract"]
    assert oracle["graph_boundary_final"]["progress"] == -1.0
    assert oracle["transient_graph_status"].endswith("non-latching")
    assert contract["scorer_training"]["missing_label_policy"] \
        == "stop before encoding/training"
    assert contract["prohibitions"]["train_with_any_missing_label"] is True


def test_replay_contract_uses_preserved_physical_witness_and_exact_runtime():
    contract = C.contract()
    replay = contract["replay_policy"]
    assert replay["source_snapshot_digest_preserved_as_lineage"] is True
    assert replay["source_snapshot_digest_equality_required"] is False
    assert "proprio/control/action context" in replay["prebranch_physical_witness"]
    assert contract["genesis_runtime"] == C.GENESIS_RUNTIME_CONTRACT
    assert C.GENESIS_RUNTIME_CONTRACT["backend"] == "cpu"
    assert C.GENESIS_RUNTIME_CONTRACT["genesis_version"] == "0.3.14"
    assert C.GENESIS_RUNTIME_CONTRACT["gstaichi_version"] == "4.6.0"
    assert C.GENESIS_RUNTIME_CONTRACT["numpy_version"] == "2.4.6"
    superseded = contract["superseded_preattempt_execution_authority"]
    assert superseded["candidate_branch_execution_started"] is False
    assert superseded["replay_attempt_markers"] == 0
    assert superseded["replay_overlays"] == 0


def test_committed_preregistration_matches_the_complete_live_source_closure():
    artifact = json.loads((ROOT / C.PREREGISTRATION_PATH).read_text())
    assert C.validate_preregistration(artifact, root=ROOT) == artifact
