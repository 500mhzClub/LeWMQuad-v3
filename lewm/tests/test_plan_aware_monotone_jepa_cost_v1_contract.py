from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from lewm.safety import plan_aware_monotone_jepa_cost_v1_contract as C


def _split() -> dict[str, object]:
    return {
        "fit": [f"fit-{index}" for index in range(32)],
        "calibration": [f"cal-{index}" for index in range(8)],
        "heldout": [f"held-{index}" for index in range(8)],
        "policy": "synthetic fixture",
    }


def _route_rows(split: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for role in C.ROLE_IDS:
        for state_offset, state_id in enumerate(split[role]):
            for candidate_index in C.CANDIDATE_INDICES:
                horizons = {}
                for horizon in ("1", "2", "3"):
                    horizons[horizon] = {
                        "completed": candidate_index == 0,
                        "heading_error_end_rad": float(candidate_index),
                        "heading_error_start_rad": float(candidate_index + 1),
                        "p_d": float(12 - candidate_index),
                        "p_theta_deg": float(candidate_index),
                        "p_theta_rad": float(candidate_index) / 10.0,
                        "route_heading_world_rad": 0.0,
                        "safe": candidate_index % 2 == 0,
                    }
                rows.append(
                    {
                        "branch_id": f"{state_id}/{candidate_index:02d}",
                        "candidate_index": candidate_index,
                        "family": C.FAMILY_IDS[state_offset % len(C.FAMILY_IDS)],
                        "horizons": horizons,
                        "split": role,
                        "state_id": state_id,
                    }
                )
    return rows


def _valid_result() -> dict[str, object]:
    result = {
        "schema": "plan_aware_monotone_jepa_cost_v1.result.v1",
        "experiment_id": C.EXPERIMENT_ID,
        "source_commit": C.SOURCE_COMMIT,
        "source_freeze_commit": "1" * 40,
        "contract_freeze_commit": "1" * 40,
        "result_commit": None,
        "result_commit_binding_policy": C.RESULT_COMMIT_BINDING_POLICY,
        "ancestry_validation": {
            "requirements_ancestor_to_source": True,
            "source_to_contract_freeze": True,
            "result_commit_pending": True,
        },
        "contract_sha256": C.CONTRACT_SHA256,
        "output_schema_sha256": C.OUTPUT_SCHEMA_SHA256,
        "panel_bindings": {},
        "checkpoint_bindings": {},
        "stage_execution": {},
        "metrics": {},
        "primary_classification": "PLAN_AWARE_JEPA_COST_NO_SIGNAL",
        "secondary_classifications": ["ENCODER_ROUTE_INFORMATION_INSUFFICIENT"],
        "next_experiment": "NON_GREEDY_LOCAL_SUBGOAL_JEPA_PLANNING_V1",
        "next_experiment_specification": (
            C.next_experiment_specification_for_primary(
                "PLAN_AWARE_JEPA_COST_NO_SIGNAL"
            )
        ),
        "requirements_workstream": "REQUIREMENTS_ACQUISITION_REQUIRED",
        "predecessor_fact_authority": list(C.PRESERVED_PREDECESSOR_FACTS),
        "predecessor_narrative_authority": list(C.PRESERVED_PREDECESSOR_NARRATIVE),
        "prior_smoke_failure_custody": [],
        "prohibition_counters": {},
        "runtime_and_storage": {},
        "nothing_running": True,
    }
    return C.attach_self_digest(result)


def test_canonical_json_is_stable_and_rejects_nonfinite() -> None:
    left = {"b": [2, 3], "a": 1.25}
    right = {"a": 1.25, "b": [2, 3]}
    assert C.canonical_json_bytes(left) == C.canonical_json_bytes(right)
    assert C.canonical_json_sha256(left) == C.canonical_json_sha256(right)
    with pytest.raises(C.ContractError, match="non-finite"):
        C.canonical_json_bytes({"bad": float("nan")})


def test_self_digest_fails_closed_on_tamper() -> None:
    value = C.attach_self_digest({"schema": "fixture", "count": 3})
    assert C.validate_self_digest(value) == value
    changed = copy.deepcopy(value)
    changed["count"] = 4
    with pytest.raises(C.ContractError, match="mismatch"):
        C.validate_self_digest(changed)


def test_contract_freezes_source_hashes_and_missing_p1_pr_custody() -> None:
    assert C.SOURCE_COMMIT == "1d799eb24d8171cb6d90bc0d0e375d9e1b0cc4f0"
    assert C.PANEL_BINDINGS["state_manifest"]["sha256"].startswith("da67309c")
    assert C.PANEL_BINDINGS["split"]["sha256"].startswith("ebef7db8")
    assert C.PANEL_BINDINGS["branch_ledger"]["sha256"].startswith("9b25b227")
    assert C.PANEL_BINDINGS["route_labels"]["sha256"].startswith("e8d33671")
    assert C.PANEL_BINDINGS["true_future_index"]["sha256"].startswith("df5e55b6")
    assert C.CHECKPOINT_BINDINGS["R1_RGB_ONE_STEP"]["sha256"].startswith("20b6e3fa")
    assert C.CHECKPOINT_BINDINGS["RR_RGB_ROLLOUT"]["sha256"].startswith("75e7a8f5")
    assert C.CHECKPOINT_BINDINGS["P1_PROPRIO_ONE_STEP"]["sha256"].startswith("41d1c5a4")
    assert C.CHECKPOINT_BINDINGS["PR_PROPRIO_ROLLOUT"]["sha256"].startswith("75ab2a5d")
    assert C.PREDECESSOR_TENSOR_PACKAGE["bound_48_state_sources"] == [
        "TRUE_FUTURE",
        "R1_RGB_ONE_STEP",
        "RR_RGB_ROLLOUT",
    ]
    assert C.PREDECESSOR_TENSOR_PACKAGE["unbound_48_state_sources"] == [
        "P1_PROPRIO_ONE_STEP",
        "PR_PROPRIO_ROLLOUT",
    ]
    assert C.PROPRIO_INPUT_BINDINGS["48_state_proprio_context"]["status"] == (
        "MISSING_PROSPECTIVE_BINDING"
    )
    assert C.LEGACY_COUNTERFACTUAL_TENSOR_BINDINGS["status"] == (
        "INELIGIBLE_IDENTITY_WITNESSES_ONLY"
    )
    assert C.LEGACY_COUNTERFACTUAL_TENSOR_BINDINGS["per_state_prediction_shape"] == [
        12,
        4,
        768,
        1024,
    ]


def test_feature_layouts_and_parameter_caps_are_exact() -> None:
    assert C.BASE_FEATURE_DIM == 138
    assert [width for _, width, _ in C.BASE_FEATURE_LAYOUT] == [6, 3, 45, 45, 3, 30, 6]
    assert C.QUERY_FEATURE_DIM == 71
    assert [width for _, width, _ in C.QUERY_FEATURE_LAYOUT] == [6, 3, 30, 2, 30]
    assert C.NO_LATENT_PARAMETER_COUNT == 26_113
    assert C.LATENT_PARAMETER_COUNT == 232_514
    assert C.TOKENS_PER_TIMEPOINT == 768
    assert C.TOKEN_GRID_SHAPE == (24, 32)
    assert C.TOKEN_GRID_SHAPE[0] * C.TOKEN_GRID_SHAPE[1] == C.TOKENS_PER_TIMEPOINT
    assert C.NO_LATENT_PARAMETER_COUNT < 250_000
    assert C.LATENT_PARAMETER_COUNT < 500_000
    assert C.TRAINING_POLICY["models"]["NO_LATENT"]["score"].startswith(
        "-kinematic_rank_cost"
    )
    assert C.TRAINING_POLICY["models"]["LATENT"]["readout_input"] == 394
    assert C.TRAINING_POLICY["models"]["LATENT"]["tokens_per_timepoint"] == 768
    assert C.TRAINING_POLICY["models"]["LATENT"]["token_grid_shape"] == [24, 32]
    assert C.TRAINING_POLICY["models"]["LATENT"]["token_grid_storage_order"] == (
        "row-major"
    )
    assert C.TRAINING_POLICY["models"]["LATENT"]["token_flat_index"] == (
        "y * 32 + x"
    )


def test_training_and_role_barriers_are_frozen() -> None:
    assert C.TRAINING_POLICY["optimizer"] == "AdamW"
    assert C.TRAINING_POLICY["learning_rate"] == 1e-3
    assert C.TRAINING_POLICY["weight_decay"] == 1e-4
    assert C.TRAINING_POLICY["epochs"] == 60
    assert C.TRAINING_POLICY["loss"] == {
        "pairwise_weight": 1.0,
        "listwise_weight": 0.5,
        "residual_l2_weight": 1e-3,
        "listwise_temperature": 1.0,
        "pairwise": (
            "logistic loss with y_ij = sign(conditioned margin-Borda utility_i "
            "- utility_j) only when abs(delta) > 1e-12; otherwise y_ij = 0"
        ),
        "pairwise_utility_tolerance": 1e-12,
        "listwise": "cross-entropy from score softmax to margin-Borda target softmax",
        "residual": "mean squared learned residual before the fixed kinematic anchor",
    }
    assert C.STAGE_POLICY["STAGE_B_PREDICTOR_SUBSTITUTION"]["condition"] == (
        "TRUE_FUTURE_GATE passes"
    )
    assert C.STAGE_POLICY["STAGE_C_ATTRIBUTION"]["condition"] == (
        "PROPRIOCEPTIVE_ROUTE_CONTRIBUTION is supported"
    )


def test_role_map_requires_exact_disjoint_cardinalities() -> None:
    split = _split()
    role_map = C.build_role_map(split)
    assert len(role_map) == 48
    assert sum(role == "fit" for role in role_map.values()) == 32
    broken = copy.deepcopy(split)
    broken["calibration"][0] = broken["fit"][0]
    with pytest.raises(C.ContractError, match="multiple roles"):
        C.build_role_map(broken)


def test_route_role_mapping_is_exact() -> None:
    assert C.route_role_from_state_class("TRANSLATIONAL_PROGRESS_AVAILABLE") == "translational"
    assert C.route_role_from_state_class("ALIGNMENT_PROGRESS_AVAILABLE") == "alignment"
    assert C.route_role_from_state_class("SAFE_HOLD_OR_ABSTAIN") == "hold_abstain"
    assert C.route_role_from_state_class("NO_SAFE_CANDIDATE") == "hold_abstain"
    with pytest.raises(C.ContractError):
        C.route_role_from_state_class("invented")


def test_tracked_route_role_authority_is_exact_load_only_input() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / C.TRACKED_ROUTE_ROLE_AUTHORITY_PATH
    authority = C.load_and_validate_route_role_authority(path)
    role_map = C.load_route_role_map(path)
    assert authority["record_count"] == 48
    assert len(role_map) == 48
    assert all(
        set(row)
        == {"state_id", "family", "role", "route_population_class", "route_role"}
        for row in authority["records"]
    )
    assert authority["authority"]["source"] == C.ROUTE_ROLE_AUTHORITY_BINDING
    assert authority["authority"]["source_fields_read"] == [
        "state_id",
        "family",
        "role",
        "route_population_class",
    ]
    assert authority["authority"]["class_recomputation"] is False
    assert authority["custody"]["route_label_candidate_outcomes_read"] is False
    assert authority["custody"]["route_class_recomputed"] is False
    assert authority["custody"]["source_metric_fields_read"] == []
    assert authority["custody"][
        "live_role_inference_from_candidate_outcomes_forbidden"
    ] is True
    assert C.CONTRACT["feature_contract"]["route_role_runtime_policy"] == (
        "load by state_id from the bound receipt only; live inference from "
        "candidate route/contact outcomes is forbidden"
    )
    assert C.CONTRACT["feature_contract"]["route_role_receipt"] == (
        C.ROUTE_ROLE_RECEIPT_BINDING
    )


def test_route_role_writer_corrects_missing_and_stale_receipt(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    destination = tmp_path / "authority.json"

    assert C.write_route_role_authority(repo_root, destination) == destination
    expected = (
        repo_root / C.TRACKED_ROUTE_ROLE_AUTHORITY_PATH
    ).read_bytes()
    assert destination.read_bytes() == expected
    assert destination.stat().st_size == C.ROUTE_ROLE_RECEIPT_BINDING["bytes"]
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == (
        C.ROUTE_ROLE_RECEIPT_BINDING["sha256"]
    )

    destination.write_bytes(b"stale\n")
    C.write_route_role_authority(repo_root, destination)
    assert destination.read_bytes() == expected
    assert C.load_and_validate_route_role_authority(destination)["record_count"] == 48


def test_route_role_predecessor_metadata_authority_binding_is_exact() -> None:
    binding = C.ROUTE_ROLE_AUTHORITY_BINDING
    path = Path(binding["path"])
    assert path.is_file()
    assert path.stat().st_size == binding["bytes"] == 7_281_659
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == binding["sha256"] == (
        "cefc11d4640a5a7ad6475d52b9a525e3d528775da3af206b82b1a1a77d9870d4"
    )
    assert binding["content_digest"] == (
        "c5da5782627446ba479bad4e04ff98468196422895d7d5ceed373fee8b46d864"
    )
    assert binding["authority_path"] == (
        "by_source_population.KINEMATIC_ROUTE_BASELINE.ALL_CANDIDATES.per_state"
    )


def test_route_only_extraction_enforces_roles_and_drops_safe() -> None:
    split = _split()
    rows = _route_rows(split)
    fit = C.extract_route_only_rows(rows, split, phase="TRAINING")
    assert len(fit) == 384
    assert {row["role"] for row in fit} == {"fit"}
    assert all("safe" not in json.dumps(row) for row in fit)
    with pytest.raises(C.ContractError, match="final checkpoints"):
        C.extract_route_only_rows(rows, split, phase="CALIBRATION")
    calibration = C.extract_route_only_rows(
        rows, split, phase="CALIBRATION", final_checkpoints_locked=True
    )
    assert len(calibration) == 96
    with pytest.raises(C.ContractError, match="evaluation contract"):
        C.extract_route_only_rows(rows, split, phase="EVALUATION")
    heldout = C.extract_route_only_rows(
        rows, split, phase="EVALUATION", evaluation_contract_frozen=True
    )
    assert len(heldout) == 96


def test_route_only_extraction_rejects_schema_and_split_drift() -> None:
    split = _split()
    rows = _route_rows(split)
    rows[0]["contact"] = False
    with pytest.raises(C.ContractError, match="top-level schema drift"):
        C.extract_route_only_rows(rows, split, phase="TRAINING")
    rows = _route_rows(split)
    rows[0]["split"] = "heldout"
    with pytest.raises(C.ContractError, match="disagrees"):
        C.extract_route_only_rows(rows, split, phase="TRAINING")


def test_true_future_gate_uses_inclusive_thresholds() -> None:
    metrics = {
        "pairwise_accuracy": 0.75,
        "spearman_rho": 0.60,
        "normalized_regret": 0.20,
        "best_route_top3": 0.75,
        "selected_progress_fraction_of_oracle": 0.80,
        "no_family_complete_collapse": True,
        "candidate_derangement_material": True,
    }
    assert C.true_future_gate_passes(metrics)
    metrics["pairwise_accuracy"] -= 1e-12
    assert not C.true_future_gate_passes(metrics)


def test_predicted_source_absolute_preservation_and_all_fail_are_exact() -> None:
    passing = {
        "pairwise_accuracy": 0.70,
        "normalized_regret": 0.25,
        "best_route_top3": 0.75,
        "selected_progress_fraction_of_oracle": 0.75,
        "selected_progress_fraction_of_true": 0.85,
        "no_family_complete_collapse": True,
    }
    assert C.predicted_source_absolute_preservation_passes(passing)
    failing = copy.deepcopy(passing)
    failing["pairwise_accuracy"] = 0.70 - 1e-12
    assert not C.predicted_source_absolute_preservation_passes(failing)
    all_fail = {source: copy.deepcopy(failing) for source in C.PREDICTED_SOURCE_IDS}
    assert C.all_predicted_substitutions_fail_materially(all_fail)
    one_passes = copy.deepcopy(all_fail)
    one_passes["P1_PROPRIO_ONE_STEP"] = passing
    assert not C.all_predicted_substitutions_fail_materially(one_passes)
    with pytest.raises(C.ContractError, match="source drift"):
        C.all_predicted_substitutions_fail_materially(
            {source: failing for source in C.PREDICTED_SOURCE_IDS[:-1]}
        )


def test_primary_fails_closed_when_non_rr_prediction_passes_absolute_screen() -> None:
    with pytest.raises(C.ContractError, match="unresolved"):
        C.derive_primary_classification(
            true_gate_pass=True,
            true_incremental_gate_pass=True,
            rr_gate_pass=False,
            all_predicted_substitutions_fail_materially=False,
        )


def test_primary_secondary_and_next_experiment_vocabularies_are_unique() -> None:
    assert C.classification_vocabularies_are_unique()
    assert len(C.PRIMARY_CLASSIFICATIONS) == len(set(C.PRIMARY_CLASSIFICATIONS))
    assert len(C.SECONDARY_CLASSIFICATIONS) == len(set(C.SECONDARY_CLASSIFICATIONS))
    assert len(C.NEXT_EXPERIMENT_IDS) == len(set(C.NEXT_EXPERIMENT_IDS))
    assert set(C.NEXT_EXPERIMENT_SPECIFICATIONS) == set(C.NEXT_EXPERIMENT_IDS)
    assert C.EVALUATOR_FIXTURE["checks"]["classification_vocabularies_unique"]


def test_contract_source_has_no_duplicate_literal_dictionary_keys(tmp_path: Path) -> None:
    source = Path(C.__file__)
    assert C.duplicate_literal_dict_keys(source) == []
    C.validate_no_duplicate_literal_dict_keys(source)
    duplicate = tmp_path / "duplicate.py"
    duplicate.write_text("VALUE = {'sha256': 'first', 'sha256': 'second'}\n")
    rows = C.duplicate_literal_dict_keys(duplicate)
    assert rows == [{"key_repr": "'sha256'", "first_line": 1, "duplicate_line": 1}]
    with pytest.raises(C.ContractError, match="duplicate literal dictionary keys"):
        C.validate_no_duplicate_literal_dict_keys(duplicate)


def test_derangement_materiality_allows_tiny_mixed_changes_but_blocks_material_reversal() -> None:
    assert C.derangement_is_material(
        pairwise_accuracy_drop=0.05,
        selected_progress_fraction_drop=-0.099,
        normalized_regret_increase=0.0,
    )
    assert not C.derangement_is_material(
        pairwise_accuracy_drop=1.0,
        selected_progress_fraction_drop=-0.10,
        normalized_regret_increase=1.0,
    )


def test_true_incremental_gate_must_pass_against_both_comparators() -> None:
    row = {
        "selected_progress_gain_m": 0.02,
        "oracle_progress_fraction_gain": 0.0,
        "normalized_regret_reduction": 0.03,
        "family_tie_or_improve_count": 3,
        "maximum_population_progress_loss_fraction": 0.02,
        "all_candidates_contact_selections_no_worse": True,
        "all_candidates_nonviable_selections_no_worse": True,
    }
    comparisons = {"KINEMATIC": copy.deepcopy(row), "NO_LATENT": copy.deepcopy(row)}
    assert C.true_incremental_gate_passes(comparisons)
    comparisons["NO_LATENT"]["selected_progress_gain_m"] = 0.0
    assert not C.true_incremental_gate_passes(comparisons)
    comparisons["NO_LATENT"]["oracle_progress_fraction_gain"] = 0.05
    assert C.true_incremental_gate_passes(comparisons)


def test_failed_true_gate_incremental_payload_is_exact_and_non_evaluated() -> None:
    expected = {
        "schema": "plan_aware_incremental_route_value_gate_v1",
        "thresholds": {
            "selected_progress_gain_m_minimum": 0.02,
            "oracle_normalized_progress_gain_minimum": 0.05,
            "normalized_regret_reduction_minimum": 0.03,
            "families_improved_or_tied_minimum": 3,
            "population_progress_loss_maximum": 0.02,
        },
        "comparisons": {},
        "pass": False,
        "classification": (
            "TRUE_FUTURE_JEPA_INCREMENTAL_ROUTE_VALUE_NOT_EVALUATED"
        ),
        "status": "NOT_EVALUATED_TRUE_GATE_FAILED",
        "evaluated": False,
        "reason": "TRUE_FUTURE_GATE_FAILED",
    }
    assert C.true_incremental_not_evaluated_payload() == expected
    assert C.TRUE_INCREMENTAL_NOT_EVALUATED_PAYLOAD == expected
    assert C.STAGE_POLICY["STAGE_A_TRUE_FUTURE"]["on_failure"][
        "true_incremental_route_value"
    ] == expected
    assert C.OUTPUT_SCHEMA["required_metric_fields"][
        "stage_a_failed_true_gate_incremental_payload"
    ] == expected


def test_training_smoke_retry_policy_and_custody_schema_are_exact() -> None:
    policy = C.EXECUTION_RETRY_POLICY
    assert C.CONTRACT_FREEZE_COMMIT_SUBJECT == (
        "Freeze plan-aware monotone JEPA route cost"
    )
    assert policy["default"] == "NO_RETRY"
    assert policy["eligible_failure_receipt_exact"] == {
        "phase": "TRAINING_SMOKE",
        "full_training_epochs_completed": 0,
        "calibration_rows_opened": 0,
        "heldout_rows_opened": 0,
        "final_checkpoint_published": False,
        "partial_artifacts_reusable": False,
        "nothing_running": True,
    }
    assert policy["same_source_freeze_retry"] is False
    assert policy["prior_artifact_reuse"] is False
    assert policy["later_failure"]["retry_allowed"] is False
    assert policy["corrected_freeze"][
        "mutable_python_correction_required_against_every_archive"
    ] is True
    assert policy["corrected_freeze"][
        "empty_or_closure_only_descendant_retry"
    ] is False
    assert C.validate_prior_smoke_failure_custody([]) == []
    binding = {
        "path": "/tmp/failure.json",
        "sha256": "a" * 64,
        "bytes": 123,
        "content_digest": "b" * 64,
    }
    record = {
        "archive_path": "/tmp/.plan-aware.failed-1",
        "failure_receipt": binding,
        "source_freeze_commit": "1" * 40,
        "source_closure": {
            **binding,
            "path": "/tmp/source_closure.json",
        },
        "inventory": {
            "files": 2,
            "bytes": 456,
            "manifest_sha256": "c" * 64,
        },
        "files_reused": 0,
    }
    assert C.validate_prior_smoke_failure_custody([record]) == [record]
    duplicate = [record, {**copy.deepcopy(record), "archive_path": "/tmp/second"}]
    with pytest.raises(C.ContractError, match="same-source-freeze"):
        C.validate_prior_smoke_failure_custody(duplicate)
    missing_snapshot = {**copy.deepcopy(record), "source_closure": None}
    with pytest.raises(C.ContractError, match="source_closure binding drift"):
        C.validate_prior_smoke_failure_custody([missing_snapshot])
    assert C.OUTPUT_SCHEMA["artifacts"]["preexecution_receipt"][
        "required_fields"
    ] == ["prior_smoke_failure_custody", "source_closure_snapshot"]
    snapshot = C.OUTPUT_SCHEMA["artifacts"][
        "pre_smoke_source_closure_snapshot"
    ]
    assert snapshot["path"] == "receipts/source_closure.json"
    assert snapshot["byte_exact_copy_of_tracked_source_closure"] is True
    assert C.EXECUTION_RETRY_POLICY["corrected_freeze"][
        "immutable_scientific_authority_paths"
    ] == list(C.CORRECTION_REFREEZE_IMMUTABLE_AUTHORITY_PATHS)
    assert "prior_smoke_failure_custody" in C.OUTPUT_SCHEMA[
        "result_required_fields"
    ]


def test_correction_refreeze_rejects_changed_scientific_authority(
    tmp_path: Path,
) -> None:
    for index, relative in enumerate(
        C.CORRECTION_REFREEZE_IMMUTABLE_AUTHORITY_PATHS
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"authority-{index}".encode("utf-8"))
    snapshot = C.build_source_closure(
        tmp_path,
        paths=C.CORRECTION_REFREEZE_IMMUTABLE_AUTHORITY_PATHS,
        require_complete=True,
    )
    snapshot_path = tmp_path / "archive/receipts/source_closure.json"
    snapshot_path.parent.mkdir(parents=True)
    snapshot_bytes = C.source_closure_receipt_bytes(snapshot)
    snapshot_path.write_bytes(snapshot_bytes)
    custody = [{
        "source_closure": {
            "path": str(snapshot_path),
            "sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
            "bytes": len(snapshot_bytes),
            "content_digest": snapshot["content_digest"],
        }
    }]
    C._validate_correction_immutable_authorities(tmp_path, custody)
    changed = tmp_path / C.CORRECTION_REFREEZE_IMMUTABLE_AUTHORITY_PATHS[0]
    changed.write_bytes(b"post-smoke scientific change")
    with pytest.raises(C.ContractError, match="scientific authority changed"):
        C._validate_correction_immutable_authorities(tmp_path, custody)


def test_rr_gate_and_proprio_gate() -> None:
    rr = {
        "pairwise_accuracy": 0.70,
        "normalized_regret": 0.25,
        "best_route_top3": 0.75,
        "selected_progress_fraction_of_oracle": 0.75,
        "selected_progress_fraction_of_true": 0.85,
        "pairwise_accuracy_delta_vs_r1": 0.01,
        "selected_progress_delta_vs_r1": 0.0,
        "normalized_regret_reduction_vs_r1": 0.01,
        "no_family_complete_collapse": True,
        "all_candidates_contact_selections_no_worse_than_r1": True,
        "all_candidates_nonviable_selections_no_worse_than_r1": True,
        "all_candidates_stuck_selections_no_worse_than_r1": True,
    }
    assert C.rr_gate_passes(rr)
    rr["pairwise_accuracy_delta_vs_r1"] = 0.0
    assert not C.rr_gate_passes(rr)
    proprio = {
        "pairwise_accuracy_gain": 0.05,
        "selected_progress_fraction_gain": 0.05,
        "normalized_regret_reduction": 0.0,
        "best_route_top3_gain": 0.0,
        "all_candidates_contact_selections_no_worse": True,
        "all_candidates_nonviable_selections_no_worse": True,
    }
    assert C.proprio_contribution_passes(proprio)
    proprio["all_candidates_contact_selections_no_worse"] = False
    assert not C.proprio_contribution_passes(proprio)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            dict(
                true_gate_pass=False,
                true_incremental_gate_pass=False,
                rr_gate_pass=False,
                all_predicted_substitutions_fail_materially=True,
            ),
            "PLAN_AWARE_JEPA_COST_NO_SIGNAL",
        ),
        (
            dict(
                true_gate_pass=True,
                true_incremental_gate_pass=False,
                rr_gate_pass=True,
                all_predicted_substitutions_fail_materially=False,
            ),
            "KINEMATIC_BASELINE_DOMINANT",
        ),
        (
            dict(
                true_gate_pass=True,
                true_incremental_gate_pass=True,
                rr_gate_pass=True,
                all_predicted_substitutions_fail_materially=False,
            ),
            "TWO_STEP_PLAN_AWARE_JEPA_COST_SIGNAL",
        ),
        (
            dict(
                true_gate_pass=True,
                true_incremental_gate_pass=True,
                rr_gate_pass=False,
                all_predicted_substitutions_fail_materially=True,
            ),
            "TRUE_FUTURE_COST_SIGNAL_PREDICTOR_ROUTE_NO_GO",
        ),
    ],
)
def test_primary_decision_tree(kwargs: dict[str, bool], expected: str) -> None:
    primary = C.derive_primary_classification(**kwargs)
    assert primary == expected
    assert C.next_experiment_for_primary(primary) in C.NEXT_EXPERIMENT_IDS
    specification = C.next_experiment_specification_for_primary(primary)
    assert specification["experiment_id"] == C.next_experiment_for_primary(primary)


def test_primary_decision_tree_fails_on_unresolved_predicted_case() -> None:
    with pytest.raises(C.ContractError, match="unresolved"):
        C.derive_primary_classification(
            true_gate_pass=True,
            true_incremental_gate_pass=True,
            rr_gate_pass=False,
            all_predicted_substitutions_fail_materially=False,
        )


def test_next_experiment_specifications_are_exact_and_result_bearing() -> None:
    mpc = C.NEXT_EXPERIMENT_SPECIFICATIONS[
        "ORACLE_ADMISSIBLE_CLOSED_LOOP_JEPA_MPC_V1"
    ]
    assert mpc["route_ranking"] == "fixed plan-aware JEPA route cost"
    assert mpc["execution"] == "execute a short prefix, reobserve, and replan"
    assert "fixed candidate bank" in mpc["candidate_generation"]
    assert mpc["separate_outcomes"] == [
        "contact",
        "successor_viability",
        "route_progress",
        "abstention",
    ]
    predictor = C.NEXT_EXPERIMENT_SPECIFICATIONS[
        "PLAN_AWARE_PREDICTOR_TRAINING_V1"
    ]
    assert predictor["predictor_seeds"] == 1
    assert "route-consistency" in predictor["objective"]
    assert predictor["safety_target_or_model_change"] is False
    assert predictor["protected_contact_scope_change"] is False
    non_greedy = C.NEXT_EXPERIMENT_SPECIFICATIONS[
        "NON_GREEDY_LOCAL_SUBGOAL_JEPA_PLANNING_V1"
    ]
    assert "temporary movement away" in non_greedy["task"]
    assert non_greedy["minimum_geometrically_plausible_route_alternatives"] == 2
    assert non_greedy["topological_memory_initially"] is False
    assert non_greedy["beacon_layer_initially"] is False
    assert "next_experiment_specification" in C.OUTPUT_SCHEMA[
        "result_required_fields"
    ]
    assert C.OUTPUT_SCHEMA["required_next_experiment_specifications"] == (
        C.NEXT_EXPERIMENT_SPECIFICATIONS
    )


def test_predecessor_candidate_evidence_is_bound_for_matched_rereduction() -> None:
    binding = C.PREDECESSOR_CANDIDATE_EVIDENCE_BINDING
    path = Path(binding["path"])
    assert path.is_file()
    assert path.stat().st_size == binding["bytes"] == 603_913
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == binding["sha256"] == (
        "9084b501f2d47a4d366c5739caf55bbe5f103f36e97023ea7db081103c4efe02"
    )
    assert binding["rows"] == 1_728
    assert binding["rows_per_source"] == 576
    assert binding["score_field"] == "cost_h3"
    assert binding["score_transform"] == "score = -cost_h3 (higher is better)"
    assert C.RAW_COST_REREDUCED_SOURCE_IDS == (
        "RAW_TRUE_FUTURE_GOAL_COSINE",
        "RAW_R1_GOAL_COSINE",
        "RAW_RR_GOAL_COSINE",
    )
    assert "after both final ranker checkpoints" in binding["open_barrier"]
    assert "never rerun cosine" in binding["reduction_policy"]
    assert "non-comparable" in binding["historical_aggregate_policy"]
    artifact = C.OUTPUT_SCHEMA["artifacts"][
        "stage_a_raw_cost_rereduced_rows"
    ]
    assert artifact["rows"] == 1_728
    assert artifact["source_ids"] == list(C.RAW_COST_REREDUCED_SOURCE_IDS)
    assert artifact["no_cosine_or_predictor_inference"] is True
    assert artifact["also_merged_into_stage_a_score_maps_and_summaries"] is True
    stage_a = C.OUTPUT_SCHEMA["artifacts"]["stage_a_rows"]
    assert stage_a["required_merged_raw_score_fields"] == list(
        C.RAW_COST_REREDUCED_SOURCE_IDS
    )
    assert stage_a["raw_cost_scores_merged_after_authorised_barrier_open"] is True


def test_stage_c_donor_mapping_is_deterministic_bijective_and_strict() -> None:
    states = ["state-a", "state-b", "state-c", "state-d"]
    first = C.build_stage_c_donor_mapping(
        states,
        split="fit",
        family="large_enclosed_maze",
        ablation="PR_PROPRIO_HISTORY_DERANGED",
    )
    second = C.build_stage_c_donor_mapping(
        list(reversed(states)),
        split="fit",
        family="large_enclosed_maze",
        ablation="PR_PROPRIO_HISTORY_DERANGED",
    )
    assert first == second
    assert C.validate_self_digest(first) == first
    recipients = {row["recipient_state_id"] for row in first["rows"]}
    donors = {row["donor_state_id"] for row in first["rows"]}
    assert recipients == donors == set(states)
    assert all(row["recipient_state_id"] != row["donor_state_id"] for row in first["rows"])
    assert first["ablation_condition_id"] == "PR_PROPRIO_HISTORY_DERANGED"
    assert first["ablation_feature"] == C.STAGE_C_ABLATION_INPUTS[
        "PR_PROPRIO_HISTORY_DERANGED"
    ]
    assert all(
        row["ablation_condition_id"] == "PR_PROPRIO_HISTORY_DERANGED"
        and row["ablation_feature_id"] == "proprio_history"
        for row in first["rows"]
    )
    assert first["required_persisted_execution_fields"] == [
        "recipient_state_id",
        "donor_state_id",
        "donor_input_sha256",
    ]
    with pytest.raises(C.ContractError, match="singleton"):
        C.build_stage_c_donor_mapping(
            ["only"],
            split="fit",
            family="large_enclosed_maze",
            ablation="PR_PROPRIO_HISTORY_DERANGED",
        )


def test_stage_c_ablation_ids_map_exactly_to_pre_predictor_inputs() -> None:
    assert C.STAGE_POLICY["STAGE_C_ATTRIBUTION"]["ablation_condition_ids"] == list(
        C.STAGE_C_ABLATION_INPUTS
    )
    assert C.STAGE_C_ABLATION_INPUTS == {
        "PR_VISUAL_CONTEXT_DERANGED": {
            "feature_id": "visual_context_sequence",
            "predictor_input_component": "visual",
        },
        "PR_PROPRIO_HISTORY_DERANGED": {
            "feature_id": "proprio_history",
            "predictor_input_component": "proprio",
        },
        "PR_CONTROL_HISTORY_DERANGED": {
            "feature_id": "previous_applied_control_history",
            "predictor_input_component": "control",
        },
    }
    with pytest.raises(C.ContractError, match="unknown Stage-C ablation"):
        C.build_stage_c_donor_mapping(
            ["state-a", "state-b"],
            split="fit",
            family="large_enclosed_maze",
            ablation="proprio_history",
        )


def test_contract_schema_fixture_and_preregistration_are_self_consistent() -> None:
    assert C.validate_contract(C.CONTRACT) == C.CONTRACT
    assert C.validate_output_schema(C.OUTPUT_SCHEMA) == C.OUTPUT_SCHEMA
    assert C.validate_evaluator_fixture(C.EVALUATOR_FIXTURE) == C.EVALUATOR_FIXTURE
    assert all(C.EVALUATOR_FIXTURE["checks"].values())
    markdown = C.build_preregistration_markdown()
    assert C.EXPERIMENT_ID in markdown
    assert C.CONTRACT_SHA256 in markdown
    assert "development-only" in markdown


def test_cpu_worker_benchmark_freezes_outcome_free_selection() -> None:
    benchmark = C.CPU_WORKER_BENCHMARK
    assert benchmark["fixture_digest"] == (
        "67c81175a6f610a56567491c6e0a36f5b42daed07ec31ef8338cdbc6b64eb305"
    )
    assert benchmark["selected_workers"] == 24
    assert benchmark["scientific_outcomes_read"] == 0
    assert benchmark["swap_free_before_bytes"] == benchmark["swap_free_after_bytes"]


def test_source_closure_builder_is_explicit_and_canonical(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "b.json").write_text("{}\n", encoding="utf-8")
    closure = C.build_source_closure(tmp_path, paths=("a.py", "b.json"))
    assert closure["complete"] is True
    assert closure["row_count"] == 2
    assert closure["declared_paths"] == ["a.py", "b.json"]
    assert C.validate_source_closure(closure) == closure
    with pytest.raises(C.ContractError, match="duplicate"):
        C.build_source_closure(tmp_path, paths=("a.py", "a.py"))
    with pytest.raises(C.ContractError, match="repository-relative"):
        C.build_source_closure(tmp_path, paths=("../outside",))


def test_source_closure_covers_all_stage_b_runtime_sources_and_tests() -> None:
    required = {
        "scripts/materialize_plan_aware_proprio_predictor_substitution_v1.py",
        "scripts/dev_proprio_predictor_v1.py",
        "scripts/build_dev_v03_proprio_action_manifest_v1.py",
        "scripts/dev_action_slew_reconstruction_v1.py",
        "scripts/run_dev_v03_temporal_action_jepa_v1.py",
        "scripts/materialize_deployment_valid_dense_proprioception_v1.py",
        "scripts/run_go2_oracle_branch_pilot_v1.py",
        "scripts/run_go2_oracle_branch_pilot_v1_2.py",
        "scripts/dev_checkpoint_v1.py",
        "scripts/dev_frozen_dense_representation_encoders_v1.py",
        "scripts/materialize_dense_route_intent_true_future_v1.py",
        "scripts/replay_safe_local_waypoint_route_intent_v2.py",
        "scripts/render_replay_v03.py",
        "lewm/oracle/go2_branch_oracle_v1_2.py",
        "lewm/oracle/go2_textured_v03_renderer.py",
        "lewm_genesis/lewm_genesis/rollout.py",
        "lewm_genesis/lewm_genesis/scene_builder.py",
        "lewm_genesis/lewm_genesis/scene_loader.py",
        "lewm_worlds/lewm_worlds/planning_grid.py",
        "lewm_worlds/lewm_worlds/labels/derived.py",
        "lewm/tests/test_plan_aware_monotone_jepa_cost_v1_contract.py",
        "lewm/tests/test_plan_aware_monotone_jepa_cost_v1.py",
        "lewm/tests/test_plan_aware_monotone_jepa_cost_metrics_v1.py",
        "lewm/tests/test_evaluate_plan_aware_monotone_jepa_cost_v1.py",
        "lewm/tests/test_materialize_plan_aware_proprio_predictor_substitution_v1.py",
    }
    assert required <= set(C.SOURCE_CLOSURE_DEFAULT_PATHS)


def test_immutable_writers_and_exact_loaders(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    schema_path = tmp_path / "schema.json"
    fixture_path = tmp_path / "fixture.json"
    prereg_path = tmp_path / "prereg.md"
    assert C.write_contract(contract_path) == contract_path
    assert C.write_output_schema(schema_path) == schema_path
    assert C.write_evaluator_fixture(fixture_path) == fixture_path
    assert C.write_preregistration(prereg_path) == prereg_path
    assert C.load_and_validate_contract(contract_path) == C.CONTRACT
    assert C.load_and_validate_output_schema(schema_path) == C.OUTPUT_SCHEMA
    assert C.load_and_validate_evaluator_fixture(fixture_path) == C.EVALUATOR_FIXTURE
    contract_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(C.ContractError, match="bytes drift"):
        C.load_and_validate_contract(contract_path)


def test_result_receipt_validator_binds_decision_and_ancestry() -> None:
    result = _valid_result()
    assert C.validate_result_receipt(result) == result
    changed = copy.deepcopy(result)
    changed["next_experiment"] = "PLAN_AWARE_PREDICTOR_TRAINING_V1"
    changed = C.attach_self_digest(changed)
    with pytest.raises(C.ContractError, match="next experiment"):
        C.validate_result_receipt(changed)
    changed = copy.deepcopy(result)
    changed["next_experiment_specification"]["topological_memory_initially"] = True
    changed = C.attach_self_digest(changed)
    with pytest.raises(C.ContractError, match="specification"):
        C.validate_result_receipt(changed)
    changed = copy.deepcopy(result)
    changed["ancestry_validation"]["result_commit_pending"] = False
    changed = C.attach_self_digest(changed)
    with pytest.raises(C.ContractError, match="ancestry"):
        C.validate_result_receipt(changed)


def test_output_paths_are_exact() -> None:
    assert str(C.TRACKED_PREREGISTRATION_PATH) == (
        "docs/lewm_plan_aware_monotone_jepa_cost_v1_preregistration_2026-08-27.md"
    )
    assert str(C.TRACKED_CONTRACT_PATH) == "docs/lewm_plan_aware_monotone_jepa_cost_v1_contract.json"
    assert str(C.TRACKED_OUTPUT_SCHEMA_PATH) == (
        "docs/lewm_plan_aware_monotone_jepa_cost_v1_output_schema.json"
    )
    assert str(C.TRACKED_FIXTURE_PATH) == (
        "docs/lewm_plan_aware_monotone_jepa_cost_v1_evaluator_fixture.json"
    )
    assert str(C.TRACKED_SOURCE_CLOSURE_PATH) == (
        "docs/lewm_plan_aware_monotone_jepa_cost_v1_source_closure.json"
    )
    assert str(C.TRACKED_ROUTE_ROLE_AUTHORITY_PATH) == (
        "docs/lewm_plan_aware_monotone_jepa_cost_v1_route_role_authority.json"
    )
    assert C.OUTPUT_ROOT.name == "plan_aware_monotone_jepa_cost_v1"


def test_predecessor_narrative_authority_is_exact_and_result_required() -> None:
    assert C.PRESERVED_PREDECESSOR_NARRATIVE[:5] == (
        "rollout training improves direct counterfactual future fidelity at H1–H4.",
        (
            "rollout training improves selected action-specific retrieval metrics, "
            "strongest at H3–H4."
        ),
        (
            "predicted latents retain partial occupancy information but remain "
            "substantially below true-target occupancy."
        ),
        "the registered proprioception interaction was broadly null.",
        "planning utility was not tested by the predictor qualification assay.",
    )
    assert C.PRESERVED_PREDECESSOR_NARRATIVE[-2] == (
        "Raw token-wise cosine distance between future V-JEPA latents and the "
        "virtual goal-view latent did not provide useful route ordering, even "
        "with true-future latents and oracle viability."
    )
    assert C.PRESERVED_PREDECESSOR_NARRATIVE[-1] == (
        "The virtual goal renderer contained the floor plane but not the maze "
        "walls or landmarks. That result was therefore not a valid visual "
        "wall-avoidance test."
    )
    assert C.OUTPUT_SCHEMA["required_predecessor_narrative_authority"] == list(
        C.PRESERVED_PREDECESSOR_NARRATIVE
    )
    assert "predecessor_narrative_authority" in C.OUTPUT_SCHEMA[
        "result_required_fields"
    ]
    assert C.OUTPUT_SCHEMA["required_predecessor_fact_authority"] == list(
        C.PRESERVED_PREDECESSOR_FACTS
    )
    assert "predecessor_fact_authority" in C.OUTPUT_SCHEMA["result_required_fields"]
    assert C.PRESERVED_PREDECESSOR_FACT_SCOPE[
        "successor_plan_aware_result_may_overwrite"
    ] is False
    assert "predecessor raw-cost result only" in C.PRESERVED_PREDECESSOR_FACT_SCOPE[
        "incremental_not_supported_scope"
    ]


def test_output_schema_matches_multi_population_rows_and_nested_checkpoint_seed_metadata() -> None:
    assert C.OUTPUT_SCHEMA["common_row_identity"] == [
        "state_id",
        "candidate_index",
        "family",
        "split",
        "latent_source",
        "population_membership",
    ]
    for artifact, condition in (
        ("no_latent_checkpoint", "KINEMATIC_PLUS_NO_LATENT_RESIDUAL"),
        ("latent_checkpoint", "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL"),
    ):
        checkpoint = C.OUTPUT_SCHEMA["artifacts"][artifact]
        assert checkpoint["condition"] == condition
        assert "seed_metadata" in checkpoint["required_root_fields"]
        assert "model_contract" in checkpoint["required_root_fields"]
        assert "fit_optimization" in checkpoint["required_root_fields"]
        assert "condition_id" not in checkpoint["required_root_fields"]
        assert checkpoint["required_seed_metadata_fields"] == list(
            C.CHECKPOINT_SEED_METADATA[condition]
        )
        assert checkpoint["seed_metadata_binding"] == C.CHECKPOINT_SEED_METADATA[
            condition
        ]
        assert checkpoint["model_contract_location"] == "root.model_contract"
        assert checkpoint["fit_optimization_location"] == "root.fit_optimization"
        assert checkpoint["required_fit_optimization_fields"] == list(
            C.FIT_OPTIMIZATION_RECEIPT_FIELDS
        )


def test_active_policies_are_frozen_exactly() -> None:
    assert C.ACTIVE_POLICIES == (
        "EVALUATION_FIRST_SINGLE_SEED",
        "ROW_LEVEL_EVIDENCE_PERSISTENCE",
        "DEVELOPMENT_MODE_END_TO_END_EXECUTION",
        "EXPLORATORY_SINGLE_SEED_FIRST",
    )
    assert C.CONTRACT["active_policies"] == list(C.ACTIVE_POLICIES)
    prereg = C.build_preregistration_markdown()
    assert all(f"`{policy}`" in prereg for policy in C.ACTIVE_POLICIES)


def test_oracle_viability_is_disclosed_conditioning_not_a_route_target() -> None:
    conditioning = C.TRAINING_POLICY["population_conditioning"]
    assert conditioning["fit_population"] == "ORACLE_VIABILITY_ADMISSIBLE"
    assert conditioning["mask_is_score_input"] is False
    assert conditioning["mask_is_route_target"] is False
    ledger = C.OUTPUT_SCHEMA["artifacts"]["route_only_target_ledger"]
    assert ledger["conditioning_policy"] == (
        "TRAIN_ONLY_ON_ORACLE_VIABILITY_ADMISSIBLE_CANDIDATE_SETS"
    )
    assert ledger["required_training_conditioning_fields"] == [
        "oracle_viability_admissible_conditioning",
        "fit_state_optimization_status",
        "used_for_fit",
        "conditioned_margin_borda_utility",
    ]
    assert ledger["all_fit_state_rows_persisted_including_skipped_states"] is True


def test_fit_state_optimizer_policy_retains_all_rows_and_skips_small_sets() -> None:
    conditioning = C.TRAINING_POLICY["population_conditioning"]
    assert conditioning["fit_state_identities_total"] == 32
    assert conditioning["retain_every_fit_state_identity_and_row"] is True
    assert conditioning["minimum_admissible_candidates_for_optimization"] == 2
    assert conditioning["fit_state_optimization_statuses"] == list(
        C.FIT_STATE_OPTIMIZATION_STATUSES
    )
    assert C.fit_state_optimization_status(0) == "SKIPPED_ZERO_ADMISSIBLE"
    assert C.fit_state_optimization_status(1) == "SKIPPED_SINGLETON_ADMISSIBLE"
    assert C.fit_state_optimization_status(2) == "CONTRIBUTING"
    assert C.fit_state_optimization_status(12) == "CONTRIBUTING"
    for invalid in (-1, 13, 1.0, True):
        with pytest.raises(C.ContractError, match="candidate count"):
            C.fit_state_optimization_status(invalid)  # type: ignore[arg-type]
    assert conditioning["zero_or_singleton_state_optimizer_step"] is False
    assert conditioning["zero_or_singleton_state_loss_contribution"] == {
        "pairwise": 0.0,
        "listwise": 0.0,
        "residual": 0.0,
    }
    assert "CONTRIBUTING fit states only" in conditioning[
        "epoch_loss_averaging_denominator"
    ]
    receipt = C.OUTPUT_SCHEMA["artifacts"]["training_receipt"]
    assert receipt["required_root_fields"] == ["fit_optimization"]
    assert receipt["fit_optimization_location"] == "root.fit_optimization"
    assert receipt["required_fit_optimization_fields"] == list(
        C.FIT_OPTIMIZATION_RECEIPT_FIELDS
    )
    assert receipt["fit_states_total_value"] == 32
    assert receipt["epoch_average_denominator_semantics"] == (
        "integer equal to fit_states_contributing"
    )
    epoch_ledger = C.OUTPUT_SCHEMA["artifacts"]["training_ledger"]
    assert epoch_ledger["epoch_average_denominator_semantics"] == (
        "integer equal to fit_states_contributing"
    )


def test_pairwise_targets_use_conditioned_margin_borda_utility_strictly() -> None:
    loss = C.TRAINING_POLICY["loss"]
    assert loss["pairwise_utility_tolerance"] == C.PAIRWISE_UTILITY_TOLERANCE == 1e-12
    assert "conditioned margin-Borda utility_i" in loss["pairwise"]
    assert C.margin_borda_pairwise_target(0.75, 0.50) == 1
    assert C.margin_borda_pairwise_target(0.50, 0.75) == -1
    assert C.margin_borda_pairwise_target(0.50, 0.50 + 1e-12) == 0
    assert C.TRAINING_POLICY["target"]["primitive_route_preference_use"].endswith(
        "not a separate pairwise-loss target"
    )
    preregistration = C.build_preregistration_markdown()
    assert "sign of conditioned margin-Borda utility" in preregistration


def test_evaluation_ordering_uses_same_borda_authority_with_tie_aware_credit() -> None:
    authority = C.METRIC_CONTRACT["ordering_authority"]
    assert authority["pairwise_target"] == (
        "sign(population-conditioned margin-Borda utility_i - utility_j) "
        "when abs(delta) > 1e-12; otherwise tied and excluded"
    )
    assert authority["oracle_best_set"] == (
        "every candidate within 1e-12 of the maximum population-conditioned "
        "margin-Borda utility"
    )
    assert "any oracle-best-set member" in authority["topk_mrr_mean_rank"]
    assert "minimum model rank" in authority["topk_mrr_mean_rank"]
    assert "realised distance progress" in authority["route_progress_correlation"]
    assert "maximum population progress" in authority["normalized_regret"]
    assert "separately from the Borda ordering target" in authority[
        "normalized_regret"
    ]
    assert C.TRUE_FUTURE_GATE["ordering_authority"].startswith(
        "population-conditioned margin-Borda"
    )
    assert C.RR_GATE["ordering_authority"].startswith(
        "population-conditioned margin-Borda"
    )
    assert "margin-Borda deltas" in C.DERANGEMENT_MATERIALITY[
        "ordering_authority"
    ]
    preregistration = C.build_preregistration_markdown()
    assert "Every candidate within 1e-12 of maximum Borda utility" in preregistration
    assert "Route-progress Spearman/Kendall" in preregistration


def test_derangement_reports_metre_and_top3_losses_without_changing_gate() -> None:
    assert C.DERANGEMENT_MATERIALITY["required_reported_damage_fields"] == [
        "pairwise_accuracy_loss",
        "selected_progress_m_loss",
        "selected_progress_ratio_loss",
        "normalized_regret_worsening",
        "best_route_top3_loss",
    ]
    assert C.DERANGEMENT_MATERIALITY["descriptive_only_damage_fields"] == [
        "selected_progress_m_loss",
        "best_route_top3_loss",
    ]
    assert C.DERANGEMENT_MATERIALITY["gate_trigger_fields"] == [
        "pairwise_accuracy_loss",
        "selected_progress_ratio_loss",
        "normalized_regret_worsening",
    ]
    reporting = C.METRIC_CONTRACT["future_derangement_reporting"]
    assert reporting["new_descriptive_fields_do_not_change_gate"] is True
    assert reporting["gate_trigger_fields"] == C.DERANGEMENT_MATERIALITY[
        "gate_trigger_fields"
    ]


def test_legitimate_unavailable_gate_metrics_fail_closed_without_type_error() -> None:
    true_metrics = {
        "pairwise_accuracy": 0.80,
        "spearman_rho": None,
        "normalized_regret": 0.10,
        "best_route_top3": 0.80,
        "selected_progress_fraction_of_oracle": 0.90,
        "no_family_complete_collapse": False,
        "candidate_derangement_material": False,
    }
    assert C.true_future_gate_passes(true_metrics) is False

    incremental_cell = {
        "selected_progress_gain_m": 0.03,
        "oracle_progress_fraction_gain": 0.06,
        "normalized_regret_reduction": None,
        "family_tie_or_improve_count": 4,
        "maximum_population_progress_loss_fraction": 0.0,
        "all_candidates_contact_selections_no_worse": True,
        "all_candidates_nonviable_selections_no_worse": True,
    }
    assert C.true_incremental_gate_passes(
        {"KINEMATIC": incremental_cell, "NO_LATENT": incremental_cell}
    ) is False

    rr = {
        "pairwise_accuracy": None,
        "normalized_regret": None,
        "best_route_top3": None,
        "selected_progress_fraction_of_oracle": 0.80,
        "selected_progress_fraction_of_true": 0.90,
        "pairwise_accuracy_delta_vs_r1": None,
        "selected_progress_delta_vs_r1": 0.01,
        "normalized_regret_reduction_vs_r1": None,
        "no_family_complete_collapse": False,
        "all_candidates_contact_selections_no_worse_than_r1": True,
        "all_candidates_nonviable_selections_no_worse_than_r1": True,
        "all_candidates_stuck_selections_no_worse_than_r1": True,
    }
    assert C.rr_gate_passes(rr) is False

    absolute = {
        "pairwise_accuracy": None,
        "normalized_regret": None,
        "best_route_top3": None,
        "selected_progress_fraction_of_oracle": 0.80,
        "selected_progress_fraction_of_true": 0.90,
        "no_family_complete_collapse": False,
    }
    assert C.predicted_source_absolute_preservation_passes(absolute) is False
    assert C.all_predicted_substitutions_fail_materially(
        {source: absolute for source in C.PREDICTED_SOURCE_IDS}
    ) is True

    contribution = {
        "pairwise_accuracy_gain": None,
        "selected_progress_fraction_gain": 0.06,
        "normalized_regret_reduction": None,
        "best_route_top3_gain": None,
        "all_candidates_contact_selections_no_worse": True,
        "all_candidates_nonviable_selections_no_worse": True,
    }
    assert C.proprio_contribution_passes(contribution) is False
    assert C.derangement_is_material(
        pairwise_accuracy_drop=None,
        selected_progress_fraction_drop=None,
        normalized_regret_increase=None,
    ) is False
    policy = C.METRIC_CONTRACT["unavailable_metric_policy"]
    assert policy["evidence_value"] is None
    assert policy["required_gate_criterion"] is False
    assert policy["gate_or_classification_exception"] is False
    assert policy["fail_closed"] is True


def test_selected_combined_route_utility_is_required_separately() -> None:
    selected = C.METRIC_CONTRACT["selected_combined_route_utility"]
    assert selected["per_state_field"] == "selected_combined_route_utility"
    assert selected["aggregate_fields"] == [
        "selected_combined_route_utility_sum",
        "selected_combined_route_utility_mean",
        "selected_combined_route_utility_count",
    ]
    assert "abstaining states are excluded" in selected["mean_denominator"]
    assert selected["sum_abstention_treatment"] == "no contribution"
    assert selected["abstentions_reported_separately"] is True
    assert selected["required_for_every_population_and_source"] is True
    required = C.OUTPUT_SCHEMA["required_metric_fields"]
    assert required["per_state"] == ["selected_combined_route_utility"]
    assert required["aggregate_per_population_source"] == [
        "selected_combined_route_utility_sum",
        "selected_combined_route_utility_mean",
        "selected_combined_route_utility_count",
    ]
    assert required["descriptive_per_source"] == [
        "descriptive_adverse_downranking"
    ]
    assert required["stage_a_matched_raw_cost_rereduction"]["source_ids"] == list(
        C.RAW_COST_REREDUCED_SOURCE_IDS
    )
    assert required["historical_raw_aggregate_context"][
        "comparable_to_successor_metrics"
    ] is False


def test_family_collapse_and_adverse_downranking_are_frozen_descriptively() -> None:
    collapse = C.METRIC_CONTRACT["family_collapse"]
    assert collapse["complete_score_tie_is_collapse"] is True
    assert collapse[
        "all_nonabstaining_score_spreads_at_or_below_tolerance_is_collapse"
    ] is True
    assert collapse[
        "tie_broken_top3_or_progress_cannot_negate_complete_score_tie"
    ] is True
    downranking = C.METRIC_CONTRACT["descriptive_adverse_downranking"]
    assert downranking["population"] == "ALL_CANDIDATES"
    assert downranking["outcomes"] == [
        "immediate_contact",
        "successor_nonviable",
        "stuck",
    ]
    assert downranking["descriptive_only"] is True
    assert downranking["score_or_route_target"] is False
    assert downranking["classification_gate"] is False


def test_p1_checkpoint_path_exists_and_matches_exact_custody() -> None:
    binding = C.CHECKPOINT_BINDINGS["P1_PROPRIO_ONE_STEP"]
    path = Path(binding["path"])
    assert str(path) == (
        "/home/andrewknowles/.cache/lewm_go2_temporal_v03/factorial_v1/"
        "seed_2026080901/seed_2026080901_proprio_one_step_epoch21.pt"
    )
    assert path.is_file()
    assert path.stat().st_size == binding["bytes"] == 206_691_255
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == binding["sha256"] == (
        "41d1c5a48d7adacf2e2b698318782de29c7b95342181bdf5fd5578d35346f1d1"
    )


def test_condition_keyed_seed_derivation_is_exact_and_shared_base_is_distinct() -> None:
    values = C.CONDITION_KEYED_SEEDS
    assert values["seed_family"] == C.RANKER_SEED == 2026082701
    expected = {
        "KINEMATIC_PLUS_NO_LATENT_RESIDUAL": (
            "fc2fc9da26214d120bdfa61071c3d8c66409d74131f71c067e5e808a1ee317ed",
            "cf4b986fac761239d9849c3e532788f135cb49e95a44107485f8f70c23e549d2",
            5713828157651817017,
        ),
        "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL": (
            "e7db447c76f03fc549a8e210f90ca537fffd5ddcc7a90fec839d4f1927436b86",
            "b9dada5d0d1f4628b4110e3a09997b47395b0d84bc8c03d056c10a52c87fb8ea",
            4168884498271782440,
        ),
    }
    for condition, (key_sha, keyed_sha, torch_seed) in expected.items():
        record = values["condition_keys"][condition]
        assert record == C.derive_keyed_seed(condition)
        assert record["key_utf8_sha256"] == key_sha
        assert record["keyed_seed_sha256"] == keyed_sha
        assert record["torch_seed"] == torch_seed
        metadata = C.CHECKPOINT_SEED_METADATA[condition]
        assert metadata["seed_family"] == C.RANKER_SEED
        assert metadata["condition_id"] == condition
        assert metadata["condition_key_sha256"] == key_sha
        assert metadata["condition_keyed_seed_sha256"] == keyed_sha
        assert metadata["condition_torch_seed"] == torch_seed
    shared = values["shared_base_subkey"]
    assert shared == C.derive_keyed_seed("SHARED_BASE_RESIDUAL")
    assert shared["key_utf8_sha256"] == (
        "6ce956d67065cf76665d42aeb6a39a10cabd1973dda2a20323147601fdf303ef"
    )
    assert shared["keyed_seed_sha256"] == (
        "b52ee5677368a328cb0f1dea45f236c991280568628e7f685f321876d4e02978"
    )
    assert shared["torch_seed"] == 3832252565419500328
    assert C.NO_LATENT_PARAMETER_COUNT == 26_113
    assert C.LATENT_PARAMETER_COUNT == 232_514
