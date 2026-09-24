"""Focused source-only tests for the frozen scorer-attribution contract."""
from __future__ import annotations

import copy
import hashlib

import pytest

from lewm.oracle import go2_scorer_failure_attribution_v1_contract as C


def _source_closure() -> dict:
    files = {
        path: {
            "path": path,
            "sha256": hashlib.sha256(path.encode("utf-8")).hexdigest(),
            "byte_count": len(path.encode("utf-8")),
        }
        for path in C.SOURCE_CLOSURE_PATHS
    }
    payload = {
        "schema": C.SOURCE_CLOSURE_SCHEMA,
        "source_repository_commit": "1" * 40,
        "source_repository_clean": True,
        "git_status_porcelain_v1": "",
        "files": files,
    }
    payload[C.SOURCE_CLOSURE_SELF_KEY] = C.canonical_digest(payload)
    return payload


def test_status_paths_and_safety_receipt_schemas_are_frozen():
    assert C.STATUS == "EXPLORATORY_SCORER_FAILURE_ATTRIBUTION"
    assert str(C.GENERATED_ROOT) == \
        ".generated/go2_scorer_failure_attribution_v1"
    assert str(C.REGISTERED_GENERATED_TARGET_ROOT) == (
        "/home/andrewknowles/.local/share/lewm_go2_planning_utility_v1_2/"
        "active/go2_scorer_failure_attribution_v1"
    )
    assert C.PLAN_PATH.parent == C.SAFETY_OBSERVABILITY_ROOT
    assert C.TRACE_ROWS_PATH.parent == C.SAFETY_OBSERVABILITY_ROOT
    assert C.ATTEMPTS_ROOT.parent == C.SAFETY_OBSERVABILITY_ROOT
    for schema, self_key in (
        (C.PLAN_SCHEMA, C.PLAN_SELF_KEY),
        (C.TRACE_ROW_SCHEMA, C.TRACE_ROW_SELF_KEY),
        (C.ATTEMPT_SCHEMA, C.ATTEMPT_SELF_KEY),
        (C.TERMINAL_SCHEMA, C.TERMINAL_SELF_KEY),
        (C.AUDIT_SCHEMA, C.AUDIT_SELF_KEY),
    ):
        assert schema.startswith("go2_scorer_failure_attribution_v1_")
        assert self_key.endswith("_digest")


def test_frozen_scientific_lineage_and_identity_sets_are_exact():
    static = C._static_contract()
    lineage = static["frozen_lineage"]
    assert lineage["oracle_v1_3_digest"] == \
        "0592876e7768a627198f1154da64b4ed492237fe68196e011fcbfcfef7706e63"
    assert lineage["training_view_digest"] == \
        "9eefff24953fdfc1eb7718ff6067a9bc06f5f8bd321f62769521234d6393291c"
    assert lineage["latent_index_digest"] == \
        "25bbd7731fc2e3026063544c64d31abff2c0ded43991504eab4d11938401b758"
    identities = static["identity_sets"]
    assert identities["fit"] == {
        "state_count": 96,
        "identity_set_digest":
            "858ad55b14d0079ea11c49a1c79b2245c7adb71846493c449e7eb3cf1d16900a",
    }
    assert identities["fresh_calibration"]["state_count"] == 24
    historical = identities["historical_development_only"]
    assert historical["state_count"] == 24
    assert len(historical["state_identity_digests"]) == 24
    assert len(set(historical["state_identity_digests"])) == 24
    assert C.canonical_digest(sorted(
        historical["state_identity_digests"])) == \
        C.FROZEN_HISTORICAL_CALIBRATION_STATE_IDENTITY_SET_DIGEST
    assert historical["qualification_eligible"] is False
    assert historical["discarded"] is False


def test_existing_results_are_bound_without_reinterpretation():
    scorers = C._static_contract()["frozen_scorers"]
    assert scorers["vitl"]["safety_auc"] == 0.7043234199
    assert scorers["vitl"]["latent_over_baseline_pairwise_gain"] \
        == 0.0317880795
    assert scorers["vitl"]["terminal"] == \
        "valid scientific qualification failure"
    assert scorers["vitg_scale_ablation"] == {
        "source_head": "8d36aeea09d1dc069d53dfb48675da560ea0c343",
        "result_digest":
            "b8b98bb7f5ae607d023a20876107cead59c3bdfa0a858955ea0d760ea5973f0a",
        "safety_auc": 0.6332379770,
        "latent_over_baseline_pairwise_gain": 0.0019867550,
        "conclusion": "NO_SCALING_SIGNAL",
    }
    assert scorers["no_latent_baseline"]["retrain"] is False


def test_current_scorer_is_global_pooling_and_not_attentive_equivalent():
    architecture = C.CURRENT_SCORER_ARCHITECTURE
    assert architecture["classification"] == "GLOBAL_OR_FIXED_POOLING"
    assert architecture["raw_latent_shape"] == ["batch", 4, 768, 1024]
    assert architecture["model_latent_shape"] == ["batch", 4, 1024]
    assert architecture["spatial_token_order_preserved"] is False
    assert architecture["horizon_order_explicit"] is False
    assert architecture["horizon_permutation_invariant"] is True
    assert architecture["self_attention"] is False
    assert architecture["cross_attention"] is False
    counts = architecture["parameter_counts"]
    assert sum(value for key, value in counts.items() if key != "total") \
        == counts["total"] == 1_599_492
    assert architecture[
        "duplicate_equivalent_to_permitted_attentive_readout"] is False


def test_safety_observability_is_all_288_exact_replays():
    safety = C.SAFETY_OBSERVABILITY_CONTRACT
    assert (C.EXPECTED_STATES, C.EXPECTED_BRANCHES) == (24, 288)
    assert (C.ADOPTED_TRACES, C.REPLAY_TRACES) == (0, 288)
    assert safety["prior_trace_references_compared_as_lineage"] == 12
    assert safety["policy_ticks"] == list(range(20))
    assert safety["sample_ticks"] == [4, 9, 14, 19]
    assert "contact_type" in safety["required_tick_fields"]
    assert safety["target_latent_encoding"] is False
    assert safety["state_replacement"] is False
    assert safety["candidate_replacement"] is False


def test_safety_mass_attribution_sums_to_frozen_outer_max_target():
    base_dominates = C.safety_mass_attribution(
        contact=0.3, clearance=0.6, stuck=0.0, fall=0.2, safety=0.3)
    assert base_dominates == pytest.approx({
        "contact": 0.1, "clearance": 0.2, "stuck": 0.0, "fall": 0.0})
    assert sum(base_dominates.values()) == pytest.approx(0.3)

    fall_dominates = C.safety_mass_attribution(
        contact=0.3, clearance=0.6, stuck=0.0, fall=1.0, safety=1.0)
    assert fall_dominates["fall"] == pytest.approx(0.7)
    assert sum(fall_dominates.values()) == pytest.approx(1.0)

    with pytest.raises(C.ScorerFailureAttributionContractError,
                       match="do not reproduce"):
        C.safety_mass_attribution(
            contact=0.3, clearance=0.6, stuck=0.0, fall=0.2, safety=0.4)


def test_transformation_suite_is_closed_and_fit_statistics_only():
    suite = C.TRANSFORMATION_SUITE
    assert list(suite) == [
        "A_MATCHED", "B_WITHIN_STATE_CANDIDATE_DERANGEMENT",
        "C_HORIZON_REVERSAL", "D_FIXED_SPATIAL_TOKEN_PERMUTATION",
        "E_SPATIAL_MEAN_REPEATED", "F_FIT_SET_MEAN_TRAJECTORY",
        "G_SINGLE_HORIZON",
    ]
    assert suite["C_HORIZON_REVERSAL"]["source_horizons"] == [4, 3, 2, 1]
    statistic = suite["F_FIT_SET_MEAN_TRAJECTORY"]["statistic"]
    assert statistic["source_split"] == "fit"
    assert statistic["rows"] == 1_152
    assert statistic["calibration_statistics_used"] is False
    assert statistic["accumulation_dtype"] == "float64"
    assert statistic["materialised_dtype"] == "float32"
    assert suite["G_SINGLE_HORIZON"]["conditions"] == [
        "H1", "H2", "H3", "H4"]
    assert suite["G_SINGLE_HORIZON"]["absence_representation_used"] is False
    assert all(condition["action_goal_unchanged"] is True
               for condition in suite.values())


def test_hash_derived_candidate_derangement_is_deterministic_and_fixed_point_free():
    state = hashlib.sha256(b"state").hexdigest()
    candidates = [hashlib.sha256(f"candidate-{index}".encode()).hexdigest()
                  for index in range(12)]
    first = C.within_state_candidate_derangement(state, candidates)
    second = C.within_state_candidate_derangement(state, list(reversed(candidates)))
    assert first == second
    assert set(first) == set(candidates)
    assert set(first.values()) == set(candidates)
    assert all(source != target for source, target in first.items())
    with pytest.raises(C.ScorerFailureAttributionContractError,
                       match="twelve unique"):
        C.within_state_candidate_derangement(state, candidates[:-1])


def test_spatial_permutation_is_one_frozen_bijection():
    permutation = C.SPATIAL_TOKEN_PERMUTATION
    assert len(permutation) == 768
    assert sorted(permutation) == list(range(768))
    assert permutation != tuple(range(768))
    assert C.SPATIAL_TOKEN_PERMUTATION_DIGEST == \
        "4585b86cd8978197298b4d865bc7e29cbb9b8d99c9cab54bf8d5851e00cb340a"
    assert C.TRANSFORMATION_SUITE[
        "D_FIXED_SPATIAL_TOKEN_PERMUTATION"]["permutation_digest"] \
        == C.SPATIAL_TOKEN_PERMUTATION_DIGEST


def test_official_pooler_binding_and_rectangular_adaptation_are_exact():
    official = C.OFFICIAL_ATTENTIVE_POOLER_BINDING
    assert official["commit"] == \
        "204698b45b3712590f06245fbfba32d3be539812"
    assert official["binding_digest"] == \
        "f436439c72e725bfd7f3caab517f2b7c870cac1cf4060623fe0c1f6da63591e6"
    config = official["config"]
    assert config == {
        "embed_dim": 512, "depth": 4, "num_heads": 16,
        "mlp_ratio": 4.0, "norm_layer": "torch.nn.LayerNorm",
        "norm_eps": 1e-5, "activation": "GELU", "qkv_bias": True,
        "complete_block": True, "dropout": 0.0,
        "attention_dropout": 0.0, "drop_path": 0.0, "init_std": 0.02,
        "use_activation_checkpointing": True,
    }
    assert official["rectangular_sequence_compatible"] is True
    assert official["grid_assumption"] is None
    assert set(official["files"]) == {
        "src/models/attentive_pooler.py", "src/models/utils/modules.py",
        "src/utils/tensors.py",
        "configs/eval_2_1/vitl-384/in1k.yaml",
        "evals/image_classification_frozen/eval.py",
    }
    assert official["files"]["src/utils/tensors.py"] == {
        "sha256":
            "782b58bd2af456e184750e5318ab773105108383f61b280fe4c7a90f46add2c8",
        "byte_count": 1_832,
    }
    payload = C.official_attentive_pooler_binding_payload()
    assert set(payload) == {"repository", "commit", "files", "config"}
    assert C.canonical_digest(payload) == official["binding_digest"]
    assert C.OFFICIAL_ATTENTIVE_POOLER_BINDING_PAYLOAD == payload


def test_attentive_architecture_seed_and_fixed_horizon_buffer_are_exact():
    architecture = C.ATTENTIVE_READOUT_ARCHITECTURE
    assert architecture["input_shape"] == ["batch", 4, 768, 1024]
    assert architecture["token_projection"] == [1024, 512]
    assert architecture["flattened_shape"] == ["batch", 3072, 512]
    assert architecture["component_queries"] == [
        "progress", "safety", "completion"]
    assert architecture["official_pooler_parameter_count"] == 12_348_416
    assert architecture["trainable_parameter_count"] == 13_684_739
    horizon = architecture["horizon_embedding"]
    assert horizon["trainable"] is False
    assert len(C.horizon_embedding_float32_bytes()) == 4 * 512 * 4
    assert C.HORIZON_EMBEDDING_SHA256 == \
        "aea9cfadd234a5b4ed1ce151d7c65fa0f5733cc1df246f81848057d895de25aa"
    assert C.ATTENTIVE_SEED_KEY_DIGEST == \
        "29b8e09b3f63487485abbee3f5b71f1c71a84f9ec5a67fa2b7eb93e9acf5363b"
    assert C.ATTENTIVE_SEED == 1_063_471_220
    initialisation = architecture["initialisation"]
    assert initialisation["construction_seed"] == C.ATTENTIVE_SEED
    assert initialisation["copy_frozen_vitl_parameter_tensors"] is False
    assert initialisation["mixed_initialisation"] is False


def test_training_metrics_gate_replay_and_stops_are_frozen():
    training = C.TRAINING_CONTRACT
    assert training["epochs"] == 60
    assert training["effective_batch_size"] == 64
    assert training["microbatch_size"] == 4
    assert training["gradient_accumulation_steps"] == 16
    assert training["model_construction_seed"] == C.ATTENTIVE_SEED
    assert training["data_order_seed"] == C.DATA_ORDER_SEED == 20_260_811
    order = training["data_order"]
    assert order == C.DATA_ORDER_CONTRACT
    assert order["generator"] == "torch.Generator(device='cpu')"
    assert order["algorithm"] == "torch.randperm(1152, generator=generator)"
    assert order["base_training_view_row_digest_sequence_digest"] == \
        "c862d0814efb0cbac179eedf9835d869a4dd3588e66c2df668feb44e469e1296"
    assert order["permutation_plan_digest"] == \
        "8e0f2c195f57fa3b883bb8830a4067f95e7965716c851be31b369d5e997c255d"
    assert order["row_presentation_plan_digest"] == \
        "85b1b96ad3aab1442c71a90e6afdbb3e3dc87e8115cb0f9c127953531f7efefb"
    assert "before attempt creation" in order["recomputation_boundary"]
    assert training["optimizer_updates"] == 1_080
    assert training["attempts"] == 1
    assert training["final_epoch_only"] is True
    assert training["best_epoch_selection"] is False
    assert training["calibration_access_during_training"] is False
    assert set(C.DIAGNOSTIC_METRICS) >= {
        "progress_mae", "progress_rmse", "progress_spearman",
        "safety_mae", "safety_rmse", "safety_auc", "safety_ece",
        "completion_mae", "completion_brier", "completion_auc",
        "completion_ece", "normalized_rank_regret",
        "realised_selected_utility", "pairwise_ordering_accuracy",
        "top_1_recovery", "top_3_recovery", "per_family_values",
    }
    gates = C.ORIGINAL_GATE_REPLAY
    assert gates["safety_auc_min"] == 0.75
    assert gates["latent_over_baseline_pairwise_gain_min"] == 0.05
    assert gates["result_is_exploratory_not_qualification"] is True
    stops = C.STOPPING_RULES
    assert stops["attentive_training_attempts"] == 1
    assert stops["attentive_evaluations"] == 1
    assert stops["additional_probe_architectures"] is False
    assert stops["predictor_checkpoint_or_utility_path"] is False
    assert stops["final_200_state_corpus"] is False


def test_diagnostics_are_completed_before_attentive_model_construction():
    prerequisites = C.DIAGNOSTIC_PREREQUISITES
    assert prerequisites["execution_order"] == [
        "safety_observability", "latent_dependence", "attentive_readout"]
    safety = prerequisites["safety_observability"]
    assert safety["required_states"] == 24
    assert safety["required_branches"] == 288
    assert safety["required_complete_tick_rows"] == 5_760
    latent = prerequisites["latent_dependence"]
    assert latent["required_calibration_rows_per_variant"] == 288
    assert len(latent["required_variants"]) == 10
    boundary = prerequisites["validation_boundary"]
    assert "before attentive model construction" in boundary


def test_strong_signal_is_predeclared_to_require_family_consistency():
    rules = C.INTERPRETATION_RULES
    assert "no inconsistent per-family primary improvement" in \
        rules["strong_requires"]
    family = rules["per_family_consistency"]
    assert family["comparison"] == "attentive minus existing ViT-L"
    assert "strictly negative delta" in family["inconsistent_if"]
    assert family["post_hoc_tolerance"] is None


def test_source_closure_is_exact_clean_and_custody_safe():
    assert len(C.SOURCE_CLOSURE_PATHS) == 8
    assert len(set(C.SOURCE_CLOSURE_PATHS)) == 8
    assert all(not path.startswith("/") for path in C.SOURCE_CLOSURE_PATHS)
    assert all("sealed" not in path for path in C.SOURCE_CLOSURE_PATHS)
    assert all("predictor" not in path.lower() for path in C.SOURCE_CLOSURE_PATHS)
    closure = _source_closure()
    assert C.validate_source_closure(closure) == closure

    dirty = copy.deepcopy(closure)
    dirty["source_repository_clean"] = False
    dirty["git_status_porcelain_v1"] = " M runner.py"
    unsigned = {key: value for key, value in dirty.items()
                if key != C.SOURCE_CLOSURE_SELF_KEY}
    dirty[C.SOURCE_CLOSURE_SELF_KEY] = C.canonical_digest(unsigned)
    with pytest.raises(C.ScorerFailureAttributionContractError,
                       match="clean and committed"):
        C.validate_source_closure(dirty)

    incomplete = copy.deepcopy(closure)
    incomplete["files"].pop(C.SOURCE_CLOSURE_PATHS[-1])
    unsigned = {key: value for key, value in incomplete.items()
                if key != C.SOURCE_CLOSURE_SELF_KEY}
    incomplete[C.SOURCE_CLOSURE_SELF_KEY] = C.canonical_digest(unsigned)
    with pytest.raises(C.ScorerFailureAttributionContractError,
                       match="eight-path"):
        C.validate_source_closure(incomplete)


def test_contract_builder_and_validator_are_pure_deterministic_and_fail_closed():
    closure = _source_closure()
    first = C.build_contract(closure)
    second = C.contract(closure)
    assert first == second
    assert first[C.CONTRACT_SELF_KEY] == C.contract_digest(closure)
    assert C.validate_contract(first) == first
    assert first["status"] == C.STATUS
    assert first["scientific_claim_status"] == \
        "exploratory_only_already_examined_data"

    changed = copy.deepcopy(first)
    changed["original_gate_replay"]["safety_auc_min"] = 0.70
    with pytest.raises(C.ScorerFailureAttributionContractError,
                       match="self digest"):
        C.validate_contract(changed)


def test_contract_grants_no_predictor_or_follow_on_authority():
    contract = C.build_contract(_source_closure())
    stops = contract["stopping_rules"]
    assert stops["predictor_checkpoint_or_utility_path"] is False
    assert stops["predictor_retraining"] is False
    assert stops["vitg_or_vitG_training"] is False
    assert stops["new_qualification_data"] is False
    assert stops["oracle_change"] is False
