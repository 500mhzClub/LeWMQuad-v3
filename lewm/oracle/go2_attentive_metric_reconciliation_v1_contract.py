"""Frozen evidence-only reconciliation contract for the attentive scorer.

This contract authorises one read-only inspection of the seven immutable
scientific-successor artifacts.  It authorises no tensor deserialisation,
model construction, forward pass, training, predictor access, or simulation.
The known closed evidence is incomplete, so the expected terminal is a
fail-closed technical unrecoverability receipt, not a scorer metric table.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
STATUS = "EXPLORATORY_ATTENTIVE_METRIC_RECONCILIATION_V1"
CONTRACT_SCHEMA = "go2_attentive_metric_reconciliation_v1_contract_v1"
CONTRACT_SELF_KEY = "attentive_metric_reconciliation_contract_digest"
SOURCE_SCHEMA = "go2_attentive_metric_reconciliation_v1_source_closure_v1"
SOURCE_SELF_KEY = "attentive_metric_reconciliation_source_closure_digest"
BASE_SOURCE_COMMIT = "8d48aa6b4cf7c0ea9d102f7f40d0946bd97b5499"

NEW_SOURCE_PATHS = (
    "lewm/oracle/go2_attentive_metric_reconciliation_v1_contract.py",
    "lewm/tests/test_go2_attentive_metric_reconciliation_v1_contract.py",
    "scripts/reconcile_go2_attentive_metric_evidence_v1.py",
    "lewm/tests/test_reconcile_go2_attentive_metric_evidence_v1.py",
)

FROZEN_SOURCE_FILES = {
    "lewm/oracle/go2_attentive_readout_layernorm_affine_scientific_successor_v1_contract.py":
        ("c9567202c99f208225db2b0dd2d3cdbecc88be03771b74d6e5fc17627d89de52", 198_786),
    "scripts/train_go2_utility_scorer_v1_3_attentive_readout_layernorm_affine_successor_v1.py":
        ("ab3d384adf07d953b5f5d9abf1950ff067fba56779794e4ac00973e9f1728eb6", 73_171),
    "scripts/train_go2_utility_scorer_v1_2.py":
        ("69f48ce630a42e5b440c31c19d9496da2dc54fb1435c096ba419c033cfb9abaa", 200_028),
    "scripts/train_go2_utility_scorer_v1_3_attentive_readout_v1.py":
        ("c7f2bd4945a0d39264ac369469a0102caa09d3dc3d5b8fa32021bda040fcb597", 62_036),
    "scripts/train_go2_utility_scorer_v1_3_attentive_readout_amendment_v1.py":
        ("4eb907fe53da00324a5e8f95181f991038faa4b0455c8db63040dbe4b0ac0a6f", 90_695),
}

GENERATED_PARENT = Path(".generated/go2_scorer_failure_attribution_v1")
REGISTERED_PARENT = Path(
    "/home/andrewknowles/.local/share/lewm_go2_planning_utility_v1_2/active/"
    "go2_scorer_failure_attribution_v1")
PREDECESSOR_RUNTIME_RELATIVE = GENERATED_PARENT / (
    "attentive_readout_layernorm_affine_scientific_successor_v1")
RUNTIME_RELATIVE = GENERATED_PARENT / "attentive_metric_reconciliation_v1"
CONTRACT_RELATIVE = RUNTIME_RELATIVE / "contract.json"

ORIGINAL_ARTIFACTS = {
    "contract.json": {
        "sha256": "b9dd0ae05072c383919d0468190d43632b42de77b57063b837bf606e7ff3d8d0",
        "byte_count": 17_001,
        "self_key": "layernorm_affine_scientific_successor_contract_digest",
        "self_digest": "e3488a0465d86356fd7e12903cfff7b323ac695b5a50a48ea91df6700dfb5b74",
    },
    "attempt.json": {
        "sha256": "7172465be54dca9a32a3bc2bc1fe0f87904d91036ab7e77c7b7d557304ca1617",
        "byte_count": 2_604,
        "self_key": "layernorm_affine_successor_attempt_digest",
        "self_digest": "782e504da75ce85e39bebbc4522c375d383ed1a5a5492910affb4969a0c5783e",
    },
    "initialisation.pt": {
        "sha256": "95805550c8224de0dd7ebe33246816e1724d67c705fa52fe634b4d164b993cca",
        "byte_count": 54_770_275,
        "content_access": "HASH_BYTES_ONLY_NEVER_DESERIALISE",
    },
    "final_checkpoint.pt": {
        "sha256": "f60f0efca0d09df8bdf596948c3cfff0a1bd8dd3913a6cf87d155573d294ce6e",
        "byte_count": 164_303_237,
        "content_access": "HASH_BYTES_ONLY_NEVER_DESERIALISE",
    },
    "evaluation_authorisation.json": {
        "sha256": "7abb718dccb37ab692bed0c3bbc36a32eabcd0cec1a942b90c5e9f8e43a2a859",
        "byte_count": 5_620,
        "self_key": "layernorm_affine_successor_evaluation_authorisation_digest",
        "self_digest": "668f02d4f5da4f511e9f11a683c65f70f1423e5bf2f29c2e9f64666efb1ad060",
    },
    "calibration_evidence.json": {
        "sha256": "6decbc875261aaad147bf15fec5181686408026e0e3e95d40ade5dfe856e1977",
        "byte_count": 220_223,
        "self_key": "calibration_evidence_digest",
        "self_digest": "bd63b21887694e074b14e9663de47bc8f9f32b84f00e11feed47c9f0a03869c0",
    },
    "technical_failure.json": {
        "sha256": "220862ac1a24c33f9b709e2436564a7d0b552655ebb88ab642f72a79d48bc6f2",
        "byte_count": 3_267,
        "self_key": "layernorm_affine_successor_technical_failure_digest",
        "self_digest": "c6f73db1302f0e53df8cf5d09631646b57fd6c548c7ebb389ecc1da488b729d6",
    },
}

FROZEN_FAILURE = {
    "stage": "evidence_metric_replay",
    "exception_type": "ScientificSuccessorError",
    "exception": "sole forward metrics and evidence replay differ",
    "completed_epochs": 60,
    "completed_optimizer_updates": 1_080,
    "calibration_evaluation_session_consumed": True,
    "calibration_evaluation_completed": True,
    "closed_evidence_rows": 288,
    "retry_resume_or_replacement_authorised": False,
}

SCIENTIFIC_ATTEMPT_LINEAGE = {
    "original_scientific_source_commit":
        "89dde156d56aaa32d94fae9c54c8eec26b15c8cd",
    "installed_scientific_source_commit": BASE_SOURCE_COMMIT,
    "scientific_source_closure_digest":
        "33a497e32a7109ccf9d3ccf243e9ed77940ded10f01d3c484f4a828ac0f2eab8",
    "scientific_contract_digest":
        "e3488a0465d86356fd7e12903cfff7b323ac695b5a50a48ea91df6700dfb5b74",
    "scientific_attempt_digest":
        "782e504da75ce85e39bebbc4522c375d383ed1a5a5492910affb4969a0c5783e",
    "registered_scorer_seed": 1_063_471_220,
    "registered_data_order_seed": 20_260_811,
    "attentive_architecture_digest":
        "0c5edc716e8bfba944d2ca89de918ca05ff571748df2b8f64f59eeea285df20d",
    "initial_state_digest":
        "02a30a879ec2cc775bd552dc4c0889a97818feadd9cd35c2c25a1a68882fa36f",
    "final_checkpoint_sha256":
        "f60f0efca0d09df8bdf596948c3cfff0a1bd8dd3913a6cf87d155573d294ce6e",
    "final_state_digest":
        "0b87368dc2c87377108e81fc609b1a6e089e8cbe094133e54559d6988088f0d2",
    "optimizer_state_digest":
        "d079a489669e8c9998fb699f93260589ff45ad50c375be392a35e69a24d039d5",
    "optimizer_state_digest_provenance": (
        "FROZEN_AUTHORITY_SUPPLIED_NOT_INDEPENDENTLY_REVERIFIED_BECAUSE_"
        "CHECKPOINT_DESERIALISATION_IS_FORBIDDEN"),
    "evaluation_authorisation_digest":
        "668f02d4f5da4f511e9f11a683c65f70f1423e5bf2f29c2e9f64666efb1ad060",
    "calibration_evidence_digest":
        "bd63b21887694e074b14e9663de47bc8f9f32b84f00e11feed47c9f0a03869c0",
    "technical_terminal_digest":
        "c6f73db1302f0e53df8cf5d09631646b57fd6c548c7ebb389ecc1da488b729d6",
    "technical_terminal_file_sha256":
        "220862ac1a24c33f9b709e2436564a7d0b552655ebb88ab642f72a79d48bc6f2",
}

FROZEN_EVIDENCE = {
    "schema": "go2_v1_3_final_layer_attentive_readout_amendment_v1_calibration_evidence_v1",
    "status": "EXPLORATORY_FINAL_LAYER_ATTENTIVE_READOUT",
    "rows": 288,
    "states": 24,
    "candidates_per_state": 12,
    "model_forward_batch_count": 72,
    "training_view_row_order_digest":
        "b676e0fbaa8729a3a8c9c647db67bdc28633ba278308a32fe1a0ac80ab97f990",
    "training_view_row_identity_set_digest":
        "b6c14d00b0c600a6195208be0d57d834765854878f3805a3e7aa7dafa970461d",
    "branch_identity_set_digest":
        "ec80511f59961faa4224842b9319d70b83a59f83ffaecb5e67db8e28aa06948e",
    "evaluation_authorisation_digest":
        "668f02d4f5da4f511e9f11a683c65f70f1423e5bf2f29c2e9f64666efb1ad060",
    "final_checkpoint_sha256":
        "f60f0efca0d09df8bdf596948c3cfff0a1bd8dd3913a6cf87d155573d294ce6e",
    "final_state_digest":
        "0b87368dc2c87377108e81fc609b1a6e089e8cbe094133e54559d6988088f0d2",
}

FROZEN_EVIDENCE_EXECUTION_BINDINGS = {
    "contract_digest":
        "e3488a0465d86356fd7e12903cfff7b323ac695b5a50a48ea91df6700dfb5b74",
    "fit_only_binding": {
        "calibration_latent_shards_opened": 0,
        "calibration_overlay_records_opened": 0,
        "calibration_row_records_opened": 0,
        "fit_only_ledger_digest":
            "b26351b06e9c7b5e56318c1bb9982dfd13f15f77d03f82c040612edaff627997",
        "fit_replay_overlays": 6,
        "fit_rows": 1_152,
        "fit_states": 96,
        "global_encoding_receipt_bytes_read": False,
        "global_latent_index_bytes_read": False,
        "global_training_view_opened": False,
        "latent_index_digest":
            "25bbd7731fc2e3026063544c64d31abff2c0ded43991504eab4d11938401b758",
        "training_view_digest":
            "9eefff24953fdfc1eb7718ff6067a9bc06f5f8bd321f62769521234d6393291c",
        "v2_adoptions": 1_146,
    },
    "implementation_digest":
        "26478275f19d2f9b1d9cc4e44a37b386cb62263cc88f9fc49f69ec7850451111",
    "predecessor_conditional_smoke_digest":
        "cde52904f02e07a7f1c70bf03ea0a3bef8ff2295c22bc54df46e42d255cdbaf1",
    "predecessor_local_cases_digest":
        "ce7493274546911478ae7c49177a285957db3bf18ba9a9d8b94b300764dce50b",
    "predecessor_terminal_digest":
        "f8429157e30e4cce8dd902b0c062704c77fa4eba65a9c5757be266664ecd2448",
}

FROZEN_NO_LATENT_BASELINE_BINDING = {
    "checkpoint_sha256":
        "cfd07d2ad739ef884f3d8ebc3faa01a0b807ef6f19049874eb7fc6ecc9c418ca",
    "state_digest":
        "33e7bcffbfab16371fb8e7e233490c33c442336edac823c19733214fa87d91d1",
    "receipt_digest":
        "454bc81c3077d62cac661a4ccac7212b3eb3860eda3177f9b8879f27632abc25",
    "metric_tree_digests": {
        "overall": "b880ea86950c8d1f1c6aba522cdaf219cbf745e1d63c102edc750f0a881a5ad5",
        "per_family": "c6c6bb9085e13de8d9c7bfbd2d75fd78252fed451ebf63429f1dc997b8966f1d",
        "per_stratum": "83bd9d67f911a9f778d0c4ca3f08adc673cd9ae6427244bad2f542c8eb1011ce",
    },
    "metric_tree_values_retained_in_the_seven_artifacts": False,
    "row_aligned_predictions_retained_in_the_seven_artifacts": False,
    "checkpoint_access_authorised": False,
}

SOURCE_RECONSTRUCTION = {
    "direct_source_file": "scripts/train_go2_utility_scorer_v1_3_attentive_readout_v1.py",
    "direct_function": "_evaluate_streaming",
    "replay_source_file": "scripts/train_go2_utility_scorer_v1_3_attentive_readout_amendment_v1.py",
    "replay_function": "metrics_from_evidence",
    "successor_call_and_equality_source_file": (
        "scripts/train_go2_utility_scorer_v1_3_attentive_readout_"
        "layernorm_affine_successor_v1.py"),
    "reducer_source_file": "scripts/train_go2_utility_scorer_v1_2.py",
    "reducer_function": "_evaluate_arrays",
    "direct_path": (
        "component targets are constructed as torch.float32, then copied to "
        "NumPy float64 for the online reducer; utility target remains the "
        "stored scalar"),
    "replay_path": (
        "persisted component target scalars are converted directly to NumPy "
        "float64; utility target remains the stored scalar"),
    "source_reconstructed_direct_semantics_tree_digests": {
        "overall": "a6e2026b924e02dd7a92b6e4034f82dfe1b1b4ab1d81ae184de4a0ab801e4beb",
        "per_family": "8178ec3afb2b81e23677de0247e2bd9e19030856cce37c290ffae320319ec677",
        "per_stratum": "f89f96cffc6cf187e6626a0575a9b56bb14a5b4fe6022101566ca77c6fef09b6",
    },
    "source_reconstructed_replay_semantics_tree_digests": {
        "overall": "a0f8f5fe777724605b834a7ca05a9a13327165d682d3df437ed5cd5f615e5465",
        "per_family": "bd405ce91538c69063b8ef853c52b50fbb94e8676562ba3b491a536c7ccd72d5",
        "per_stratum": "8185f60f5876de502356b24bca51f8c37e085519b78a3a6bb0267ef01f18da6f",
    },
    "different_leaf_counts": {
        "overall": 12, "per_family": 84, "per_stratum": 33,
    },
    "first_overall_metric_leaf": {
        "path": "progress.mae",
        "direct": 0.24382571417987087,
        "replay": 0.24382571380843376,
    },
    "first_divergent_row": {
        "row_index": 0,
        "state_id": "oracle_v1_3-calibration-large_enclosed_maze-completion_enriched",
        "family": "large_enclosed_maze",
        "stratum": "completion_enriched",
        "candidate_index": 0,
        "branch_identity_digest":
            "4924778c451dbbd0715abbbeb080ff44d1fd4b28366652f5fdecb9dd04fa5359",
        "training_view_row_digest":
            "a9361da1c3ea8c26beafc34a88f74bee66202304607d9157e92931a5c0529ce1",
        "metric_component": "progress_absolute_error",
        "stored_target": 0.6953460756919387,
        "direct_float32_projected_target": 0.6953460574150085,
        "prediction": 1.0584787130355835,
        "replay_absolute_error": 0.36313263734364476,
        "direct_absolute_error": 0.36313265562057495,
    },
    "first_tie_induced_state": {
        "state_id": "oracle_v1_3-calibration-local_composite_motifs-general",
        "family": "local_composite_motifs",
        "stratum": "general",
        "row_indices": [50, 53],
        "candidate_indices": [2, 5],
        "stored_safety_targets": [0.10000000000000002, 0.09999999999999999],
        "float32_projected_safety_targets": [
            0.10000000149011612, 0.10000000149011612],
        "direct_pairs_considered": 62,
        "replay_pairs_considered": 63,
    },
    "source_hashes_prove_dtype_path": True,
    "retained_online_metric_tree_or_accumulators": False,
    "cast_difference_is_a_source_derived_explanation_not_a_proven_sole_cause": True,
    "direct_online_tree_provenance": (
        "only the overall direct_metrics tree existed transiently in memory; "
        "no direct tree was persisted, and family/stratum direct trees here are "
        "source-reconstructed semantics"),
}

METRIC_PATH_SPECIFICATION = {
    "direct": {
        "inputs": "corpus rows plus action-goal and latent model tensors",
        "targets": "progress/safety/completion torch.float32 tensors",
        "model_outputs": "raw progress, safety-logit, completion-logit",
        "probability_transform": (
            "sigmoid exactly once for safety and completion; no progress "
            "denormalisation"),
        "utility": "frozen composite calculated in float32",
    },
    "replay": {
        "inputs": "persisted JSON identity, target, and prediction rows",
        "predictions": (
            "raw progress, post-sigmoid safety/completion probabilities, and "
            "stored composite utility; utility is not recomposed"),
    },
    "batch_reduction": (
        "none; 72 batches of four rows are concatenated to 288 rows in frozen "
        "order, with no partial final batch and no batch means"),
    "numeric_reduction": "NumPy float64 arrays and float64 accumulation",
    "grouping": {
        "overall": "corpus/row weighted, not a mean of family metrics",
        "state": "twelve candidates by state_id",
        "per_family": "the same reducer on each family subset",
        "per_stratum": "the same reducer on each stratum subset",
    },
    "component_targets": {
        "progress": "graded scalar",
        "safety": "graded for errors/ECE; target > 0 only for binary AUC",
        "completion": "binary",
    },
    "auc": (
        "average ranks with exact score ties; undefined for a single class"),
    "ece": "ten equal-width bins on [0,1], with the final upper edge inclusive",
    "ranking": {
        "stable_sort": "numpy mergesort",
        "selected_candidate": (
            "first numpy argmax in frozen state/candidate row order"),
        "composite_tie_tolerance": 0.02,
        "component_target_ties": "exact equality",
    },
    "baseline": (
        "frozen aggregate baseline trees can support aggregate comparison, but "
        "cannot substitute for row-aligned baseline predictions"),
}

EXPECTED_EVIDENCE_INVENTORY = {
    "unique_training_view_row_digests": 288,
    "unique_branch_identity_digests": 288,
    "unique_states": 24,
    "unique_state_candidate_pairs": 288,
    "states_with_exact_candidates_0_through_11": 24,
    "states_with_single_observed_family": 24,
    "states_with_single_observed_stratum": 24,
    "row_target_prediction_key_schemas_exact": True,
    "all_targets_finite": True,
    "all_predictions_finite": True,
    "all_safety_and_completion_predictions_are_probabilities": True,
    "all_completion_targets_binary": True,
    "rows_missing_action_blocks": 288,
    "rows_missing_goal_binding_input": 288,
    "rows_missing_no_latent_prediction": 288,
    "rows_missing_split_role": 288,
    "rows_missing_scene_id": 288,
    "rows_missing_state_identity_digest": 288,
    "rows_missing_safety_logit": 288,
    "rows_missing_completion_logit": 288,
    "rows_with_attentive_safety_probability": 288,
    "rows_with_attentive_completion_probability": 288,
    "attentive_prediction_scalars_exactly_float32_representable": 1_152,
    "observed_family_count": 8,
    "observed_states_per_family": 3,
    "observed_rows_per_family": 36,
    "frozen_calibration_manifest_or_state_family_mapping_retained": False,
    "direct_metric_table_retained": False,
    "direct_online_accumulators_retained": False,
    "direct_online_dtype_precision_declaration_retained": False,
    "progress_targets_changed_by_online_float32_projection": 264,
    "safety_targets_changed_by_online_float32_projection": 190,
    "maximum_progress_target_projection_delta": 2.9522020938976823e-08,
    "maximum_safety_target_projection_delta": 6.953875225645945e-09,
}

METRIC_SUITE = {
    "progress": ["mae", "rmse", "spearman"],
    "safety": ["mae", "rmse", "auc", "ece"],
    "completion": ["prevalence", "mae", "brier", "auc", "ece"],
    "composite": [
        "absolute_rank_regret", "normalised_rank_regret",
        "realised_selected_utility", "pairwise_ordering_accuracy",
        "ranking_spearman", "top1_recovery", "top3_recovery",
        "top_score_tie_rate", "all_pair_tie_rate", "candidate_score_spread",
    ],
    "grouping": "per_family",
}
TOLERANCES = {"absolute": 1e-10, "relative": 1e-9}
TIE_TOLERANCE = 0.02

ORIGINAL_GATES = {
    "progress_spearman_min": 0.50,
    "safety_auc_min": 0.75,
    "safety_ece_max": 0.10,
    "completion_auc_min": 0.75,
    "completion_ece_max": 0.10,
    "pairwise_ordering_accuracy_min": 0.65,
    "decimal_latent_over_baseline_pairwise_gain_min": "0.05",
    "completion_labels_nondegenerate_in_fit_and_calibration": True,
}

RECOVERABILITY_GATES = {
    "all_288_rows_and_24x12_identity_structure_exact": True,
    "true_component_and_utility_targets_finite": True,
    "attentive_probabilities_and_utility_predictions_finite": True,
    "complete_action_blocks_present": True,
    "complete_goal_binding_present": True,
    "row_aligned_no_latent_predictions_present": True,
    "complete_split_scene_state_and_family_manifest_provenance_present": True,
    "source_and_evidence_sufficient_to_reconstruct_direct_and_replay_paths": True,
    "consumer_a_and_b_discrete_outputs_exact": True,
    "consumer_a_and_b_float_outputs_within_frozen_tolerance": True,
    "direct_vs_replay_first_metric_divergence_exactly_localised": True,
    "baseline_identity_reorder_verified": True,
    "all_original_gate_inputs_reconstructable": None,
}

RECOVERABLE_CLASSIFICATION = "POST_EVALUATION_METRIC_CONSUMER_DEFECT_RECOVERABLE"
EXPECTED_PRIMARY_TERMINAL = "INVALID_TECHNICAL_UNRECOVERABLE_METRIC_EVIDENCE"
REPAIRED_RESULT_LABEL = "POST_EVALUATION_CONSUMER_REPAIR"
TERMINAL_KINDS = (
    EXPECTED_PRIMARY_TERMINAL,
    RECOVERABLE_CLASSIFICATION,
)
AUTHORITY = {
    "one_reconciliation_attempt": True,
    "evidence_only": True,
    "torch_import_or_torch_load": False,
    "tensor_deserialisation": False,
    "model_construction_or_forward": False,
    "training_or_optimizer": False,
    "predictor_or_utility_shard_access": False,
    "simulation_or_branch_replay": False,
    "modify_original_seven_artifacts": False,
    "publish_attentive_metrics_when_evidence_incomplete": False,
    "publish_repaired_result_only_if_all_recoverability_gates_pass": True,
    "retry_resume_or_replacement": False,
}

HEX64 = re.compile(r"[0-9a-f]{64}")


class MetricReconciliationContractError(RuntimeError):
    """The evidence-only source, lineage, or storage contract changed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MetricReconciliationContractError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path, block_size: int = 8 << 20) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(block_size), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(),
            f"{label} is absent or non-regular")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise MetricReconciliationContractError(
            f"cannot read {label}: {exc}") from exc
    require(isinstance(value, dict), f"{label} is not an object")
    return value


def validate_signed(value: Mapping[str, Any], key: str,
                    label: str) -> dict[str, Any]:
    result = dict(value)
    recorded = result.pop(key, None)
    require(isinstance(recorded, str) and HEX64.fullmatch(recorded) is not None
            and recorded == digest(result), f"{label} self digest changed")
    result[key] = recorded
    return result


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=root, text=True,
            stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MetricReconciliationContractError(
            f"cannot bind reconciliation source: {exc}") from exc


def source_closure(root: Path = ROOT) -> dict[str, Any]:
    require(_git(root, "status", "--porcelain=v1") == "",
            "reconciliation source must be clean and committed")
    head = _git(root, "rev-parse", "HEAD")
    require(subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_SOURCE_COMMIT, head],
        cwd=root, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL).returncode == 0,
        "reconciliation source does not descend from its base")
    changed = tuple(sorted(filter(None, _git(
        root, "diff", "--name-only", f"{BASE_SOURCE_COMMIT}..{head}"
    ).splitlines())))
    require(changed == tuple(sorted(NEW_SOURCE_PATHS)),
            "committed reconciliation diff is not exactly four additive paths")
    frozen = {}
    for relative, (sha256, byte_count) in FROZEN_SOURCE_FILES.items():
        target = root / relative
        require(target.is_file() and not target.is_symlink()
                and target.stat().st_size == byte_count
                and file_sha256(target) == sha256,
                f"frozen metric source changed: {relative}")
        frozen[relative] = {"sha256": sha256, "byte_count": byte_count}
    additive = {}
    for relative in NEW_SOURCE_PATHS:
        target = root / relative
        require(target.is_file() and not target.is_symlink(),
                f"additive reconciliation source is absent: {relative}")
        additive[relative] = {
            "sha256": file_sha256(target),
            "byte_count": target.stat().st_size,
        }
    payload = {
        "schema": SOURCE_SCHEMA,
        "source_repository_commit": head,
        "source_repository_clean": True,
        "base_source_commit": BASE_SOURCE_COMMIT,
        "exact_committed_additive_path_diff": list(changed),
        "frozen_source_files": frozen,
        "additive_files": additive,
    }
    return {**payload, SOURCE_SELF_KEY: digest(payload)}


def predecessor_root(root: Path = ROOT) -> Path:
    return root / PREDECESSOR_RUNTIME_RELATIVE


def runtime_root(root: Path = ROOT) -> Path:
    return root / RUNTIME_RELATIVE


def contract_path(root: Path = ROOT) -> Path:
    return root / CONTRACT_RELATIVE


def lineage_binding(root: Path = ROOT) -> dict[str, Any]:
    directory = predecessor_root(root)
    require(directory.is_dir() and not directory.is_symlink(),
            "scientific-successor artifact directory changed")
    children = list(directory.iterdir())
    require({child.name for child in children} == set(ORIGINAL_ARTIFACTS)
            and all(child.is_file() and not child.is_symlink()
                    and stat.S_IMODE(child.stat().st_mode) == 0o444
                    for child in children),
            "original seven-artifact inventory or immutability changed")
    observed = {}
    decoded: dict[str, dict[str, Any]] = {}
    for name, expected in ORIGINAL_ARTIFACTS.items():
        target = directory / name
        require(target.stat().st_size == expected["byte_count"]
                and file_sha256(target) == expected["sha256"],
                f"original artifact bytes changed: {name}")
        receipt = {
            "sha256": expected["sha256"],
            "byte_count": expected["byte_count"],
            "mode": "0444",
        }
        if "self_key" in expected:
            value = validate_signed(read_json(target, name),
                                    expected["self_key"], name)
            require(value[expected["self_key"]] == expected["self_digest"],
                    f"original artifact identity changed: {name}")
            receipt["self_digest"] = expected["self_digest"]
            decoded[name] = value
        else:
            receipt["content_access"] = expected["content_access"]
        observed[name] = receipt
    failure = read_json(directory / "technical_failure.json",
                        "technical failure")
    require(all(failure.get(key) == value
                for key, value in FROZEN_FAILURE.items()),
            "frozen scientific failure changed")
    contract = decoded["contract.json"]
    attempt = decoded["attempt.json"]
    evaluation = decoded["evaluation_authorisation.json"]
    evidence = decoded["calibration_evidence.json"]
    require(
        contract.get("source_closure", {}).get("source_repository_commit")
        == SCIENTIFIC_ATTEMPT_LINEAGE["installed_scientific_source_commit"]
        and contract.get("source_closure", {}).get(
            "layernorm_affine_scientific_successor_source_closure_digest")
        == SCIENTIFIC_ATTEMPT_LINEAGE["scientific_source_closure_digest"]
        and contract.get("architecture_digest")
        == SCIENTIFIC_ATTEMPT_LINEAGE["attentive_architecture_digest"]
        and contract.get("no_latent_baseline") == {
            "checkpoint_sha256": FROZEN_NO_LATENT_BASELINE_BINDING[
                "checkpoint_sha256"],
            "receipt_digest": FROZEN_NO_LATENT_BASELINE_BINDING[
                "receipt_digest"],
            "reevaluated": False, "retrained": False,
            "state_digest": FROZEN_NO_LATENT_BASELINE_BINDING["state_digest"],
        }
        and contract.get("frozen_metric_tree_digests", {}).get("no_latent")
        == FROZEN_NO_LATENT_BASELINE_BINDING["metric_tree_digests"]
        and attempt.get("registered_seed")
        == SCIENTIFIC_ATTEMPT_LINEAGE["registered_scorer_seed"]
        and attempt.get("data_order_seed")
        == SCIENTIFIC_ATTEMPT_LINEAGE["registered_data_order_seed"]
        and attempt.get("initial_state_digest")
        == SCIENTIFIC_ATTEMPT_LINEAGE["initial_state_digest"]
        and evaluation.get("final_checkpoint_sha256")
        == SCIENTIFIC_ATTEMPT_LINEAGE["final_checkpoint_sha256"]
        and evaluation.get("final_state_digest")
        == SCIENTIFIC_ATTEMPT_LINEAGE["final_state_digest"]
        and evaluation.get(
            "layernorm_affine_successor_evaluation_authorisation_digest")
        == SCIENTIFIC_ATTEMPT_LINEAGE["evaluation_authorisation_digest"]
        and evidence.get("calibration_evidence_digest")
        == SCIENTIFIC_ATTEMPT_LINEAGE["calibration_evidence_digest"]
        and failure.get("layernorm_affine_successor_technical_failure_digest")
        == SCIENTIFIC_ATTEMPT_LINEAGE["technical_terminal_digest"],
        "named scientific attempt lineage changed")
    return {
        "original_scientific_source_commit":
            SCIENTIFIC_ATTEMPT_LINEAGE["original_scientific_source_commit"],
        "installed_scientific_source_commit":
            SCIENTIFIC_ATTEMPT_LINEAGE["installed_scientific_source_commit"],
        "artifacts": observed,
        "artifact_set_digest": digest(observed),
        "failure": dict(FROZEN_FAILURE),
        "scientific_attempt_lineage": dict(SCIENTIFIC_ATTEMPT_LINEAGE),
    }


def storage_binding(root: Path = ROOT) -> dict[str, Any]:
    logical = root / GENERATED_PARENT
    require(logical.is_symlink() and logical.resolve() == REGISTERED_PARENT,
            "registered generated-parent symlink changed")
    target = runtime_root(root)
    require(not target.exists() and not target.is_symlink(),
            "one-shot reconciliation namespace already exists")
    return {
        "logical_parent": str(GENERATED_PARENT),
        "registered_parent": str(REGISTERED_PARENT),
        "resolved_parent": str(logical.resolve()),
        "runtime_relative": str(RUNTIME_RELATIVE),
        "runtime_namespace_absent_before_issue": True,
    }


def static_contract() -> dict[str, Any]:
    return {
        "status": STATUS,
        "repaired_result_label": REPAIRED_RESULT_LABEL,
        "expected_primary_terminal": EXPECTED_PRIMARY_TERMINAL,
        "terminal_kinds": list(TERMINAL_KINDS),
        "frozen_evidence": FROZEN_EVIDENCE,
        "frozen_evidence_execution_bindings": FROZEN_EVIDENCE_EXECUTION_BINDINGS,
        "frozen_no_latent_baseline_binding": FROZEN_NO_LATENT_BASELINE_BINDING,
        "source_reconstruction": SOURCE_RECONSTRUCTION,
        "metric_path_specification": METRIC_PATH_SPECIFICATION,
        "expected_evidence_inventory": EXPECTED_EVIDENCE_INVENTORY,
        "metric_suite": METRIC_SUITE,
        "consumer_float_tolerances": TOLERANCES,
        "tie_tolerance": TIE_TOLERANCE,
        "original_gates": ORIGINAL_GATES,
        "recoverability_gates": RECOVERABILITY_GATES,
        "authority": AUTHORITY,
    }


def build_contract(source: Mapping[str, Any], lineage: Mapping[str, Any],
                   storage: Mapping[str, Any]) -> dict[str, Any]:
    require(source.get("schema") == SOURCE_SCHEMA
            and source.get(SOURCE_SELF_KEY)
            == digest({key: value for key, value in source.items()
                       if key != SOURCE_SELF_KEY}),
            "reconciliation source closure changed")
    require(lineage.get("artifact_set_digest")
            == digest(lineage.get("artifacts"))
            and lineage.get("original_scientific_source_commit")
            == SCIENTIFIC_ATTEMPT_LINEAGE["original_scientific_source_commit"]
            and lineage.get("installed_scientific_source_commit")
            == SCIENTIFIC_ATTEMPT_LINEAGE["installed_scientific_source_commit"]
            and lineage.get("scientific_attempt_lineage")
            == SCIENTIFIC_ATTEMPT_LINEAGE
            and lineage.get("failure") == FROZEN_FAILURE,
            "reconciliation lineage binding changed")
    require(storage.get("runtime_namespace_absent_before_issue") is True
            and storage.get("runtime_relative") == str(RUNTIME_RELATIVE)
            and storage.get("resolved_parent") == str(REGISTERED_PARENT),
            "reconciliation storage binding changed")
    payload = {
        "schema": CONTRACT_SCHEMA,
        "source_closure": dict(source),
        "lineage": dict(lineage),
        "storage": dict(storage),
        **static_contract(),
    }
    return {**payload, CONTRACT_SELF_KEY: digest(payload)}


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    result = validate_signed(value, CONTRACT_SELF_KEY,
                             "metric reconciliation contract")
    require(result == build_contract(result["source_closure"],
                                     result["lineage"], result["storage"]),
            "metric reconciliation contract changed")
    return result


__all__ = [name for name in globals() if name.isupper()] + [
    "MetricReconciliationContractError", "build_contract", "canonical_bytes",
    "contract_path", "digest", "file_sha256", "lineage_binding",
    "predecessor_root", "read_json", "require", "runtime_root",
    "source_closure", "static_contract", "storage_binding",
    "validate_contract", "validate_signed",
]
