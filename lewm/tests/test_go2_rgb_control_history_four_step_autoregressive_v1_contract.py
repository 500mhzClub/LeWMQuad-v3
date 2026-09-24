"""Focused source-only tests for the bounded four-step rollout contract."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from lewm.oracle import (
    go2_rgb_control_history_four_step_autoregressive_v1_contract as C,
)


EXPECTED_SEEDS = (
    2_026_080_901, 2_026_080_902, 2_026_080_903, 2_026_080_904,
    2_026_080_905, 2_026_080_906, 2_026_080_907, 2_026_080_908,
)

EXPECTED_COMPARATORS = {
    2_026_080_901: {
        "rgb_one_step":
            "20b6e3fa2a2d3c3ec2c20ea37e524f9c2872fdcfd5226b114822efa26872261a",
        "rgb_two_step_rollout":
            "75e7a8f5eb5416100dd91fdd07c6aeae1c8fa2255ef189bfde2a5ce300f881b4",
    },
    2_026_080_902: {
        "rgb_one_step":
            "085702386da4b36bafe6ff432ca955a2b1a9a69de9a8023aa4fc3b099953f0ff",
        "rgb_two_step_rollout":
            "90bbf9a8117dbf528d9693415becd5c9e9605ecad02520f3e00513dfee691530",
    },
    2_026_080_903: {
        "rgb_one_step":
            "a7878c6159cceae8f69f84927bd1ee3a4c3d8dbf6d1e97003eb9ebdae1f91bc4",
        "rgb_two_step_rollout":
            "b769ef91f1ef17377f7c7f184c85ea0a9859ead2b87aa8351a89b7a05192aad1",
    },
    2_026_080_904: {
        "rgb_one_step":
            "1386b6303ac5b47fea7a67e831a375d164ba372ee4bf60fd87609ed35352d1ff",
        "rgb_two_step_rollout":
            "aad6711b6d15e6664038ace1fe0f376516256062c2235334b74bfb68135e419a",
    },
    2_026_080_905: {
        "rgb_one_step":
            "5d78f18e0d0052479cb81a43acbaa953bebeb6fc13dac58c506211c46416a1e9",
        "rgb_two_step_rollout":
            "c474a5b09c041aa263950b3b2b8bd2369d3644aec7019268610fea4b846b6386",
    },
    2_026_080_906: {
        "rgb_one_step":
            "846fbe05f78e9b513841cb08f71858e9fdb7dd4430181bca140d29f72574a200",
        "rgb_two_step_rollout":
            "fc480799cc637f5c3d4bd582da233e38b76d422b48833075e018d49df517aa1a",
    },
    2_026_080_907: {
        "rgb_one_step":
            "86d9f6108f40b8d2cf49e5264fc998412493258258b86391067c71193066afbc",
        "rgb_two_step_rollout":
            "4501841125eee43568e6031d4061d23b309c080f11b129538dadb6cfc8a05432",
    },
    2_026_080_908: {
        "rgb_one_step":
            "025aff4d9bc7380b4a51e4ac08282bbeafb2be189bf27d31d48bcf247f2b02f2",
        "rgb_two_step_rollout":
            "a39f5050c02ab7b002c6b1c76256dc2b5783046cf5b877cc6d5354880c45b89a",
    },
}


def _synthetic_source() -> dict:
    payload = {
        "schema": C.SOURCE_SCHEMA,
        "base_source_commit": C.BASE_SOURCE_COMMIT,
        "source_repository_commit": "1" * 40,
        "source_repository_clean": True,
        "exact_committed_additive_path_diff": list(C.NEW_SOURCE_PATHS),
        "frozen_source_files": {},
        "additive_files": {},
    }
    return {**payload, C.SOURCE_SELF_KEY: C.digest(payload)}


def _synthetic_storage() -> dict:
    return {
        "registered_runtime_parent": str(C.REGISTERED_RUNTIME_PARENT),
        "runtime_relative": str(C.RUNTIME_RELATIVE),
        "runtime_path": str(C.runtime_root()),
        "runtime_namespace_absent_before_issue": True,
        "workspace_generated_output": False,
    }


def test_source_closure_is_exactly_the_four_additive_paths() -> None:
    assert C.BASE_SOURCE_COMMIT == "b84f03a9e270f1d1eb3b5b0e12c2a03d711f00f9"
    assert C.NEW_SOURCE_PATHS == (
        "lewm/oracle/go2_rgb_control_history_four_step_autoregressive_v1_contract.py",
        "scripts/run_go2_rgb_control_history_four_step_autoregressive_v1.py",
        "lewm/tests/test_go2_rgb_control_history_four_step_autoregressive_v1_contract.py",
        "lewm/tests/test_run_go2_rgb_control_history_four_step_autoregressive_v1.py",
    )
    assert len(C.FROZEN_SOURCE_FILES) == 25
    assert all("utility" not in path for path in C.FROZEN_SOURCE_FILES)
    assert all("sealed" not in path for path in C.FROZEN_SOURCE_FILES)
    assert C.RUNTIME_RELATIVE == Path("four_step_rollout_v1")
    assert C.runtime_root() == C.REGISTERED_RUNTIME_PARENT / C.RUNTIME_RELATIVE


def test_all_eight_historical_comparator_checkpoints_are_frozen_exactly() -> None:
    assert C.FROZEN_SEEDS == EXPECTED_SEEDS
    assert C.COMPARATOR_CHECKPOINT_SHA256 == EXPECTED_COMPARATORS
    assert tuple(C.BASE_WEIGHT_SHA256) == EXPECTED_SEEDS
    assert tuple(C.BASE_STATE_DIGEST) == EXPECTED_SEEDS
    assert len(set(C.BASE_WEIGHT_SHA256.values())) == 8
    assert len(set(C.BASE_STATE_DIGEST.values())) == 8
    for seed in EXPECTED_SEEDS:
        assert C.HEX64.fullmatch(C.BASE_WEIGHT_SHA256[seed]) is not None
        assert C.HEX64.fullmatch(C.BASE_STATE_DIGEST[seed]) is not None
        assert set(C.COMPARATOR_CHECKPOINT_SHA256[seed]) == {
            "rgb_one_step", "rgb_two_step_rollout"}
        assert all(C.HEX64.fullmatch(value) is not None
                   for value in C.COMPARATOR_CHECKPOINT_SHA256[seed].values())
    assert C.FROZEN_FACTORIAL["checkpoint_epoch"] == 21
    assert C.FROZEN_FACTORIAL["historical_cells_retrained_or_reselected"] is False


def test_common_h4_manifest_counts_exclusions_and_digests_are_frozen() -> None:
    availability = C.TARGET_AVAILABILITY
    assert availability["horizon_counts"] == {
        "H1": {"train": 3_922, "selection": 475},
        "H2": {"train": 3_922, "selection": 475},
        "H3": {"train": 3_892, "selection": 471},
        "H4": {"train": 3_854, "selection": 466},
    }
    expected_family_counts = {
        "H1": {
            "train": (486, 374, 551, 465, 528, 530, 492, 496),
            "selection": (62, 62, 61, 60, 62, 60, 46, 62),
        },
        "H2": {
            "train": (486, 374, 551, 465, 528, 530, 492, 496),
            "selection": (62, 62, 61, 60, 62, 60, 46, 62),
        },
        "H3": {
            "train": (485, 370, 547, 462, 521, 526, 485, 496),
            "selection": (62, 62, 60, 60, 61, 59, 46, 61),
        },
        "H4": {
            "train": (483, 364, 543, 457, 514, 522, 479, 492),
            "selection": (61, 62, 59, 60, 60, 59, 45, 60),
        },
    }
    for horizon, splits in expected_family_counts.items():
        for split, expected in splits.items():
            observed = availability["horizon_family_counts"][horizon][split]
            assert tuple(observed) == C.FAMILIES
            assert tuple(observed.values()) == expected
            assert sum(observed.values()) == availability["horizon_counts"][
                horizon][split]
    assert availability["incremental_exclusions"] == {
        "H2_to_H3": {
            "train": {"reset_boundary": 8, "endpoint_or_end_of_rollout": 22},
            "selection": {"reset_boundary": 0,
                          "endpoint_or_end_of_rollout": 4},
        },
        "H3_to_H4": {
            "train": {"reset_boundary": 12, "endpoint_or_end_of_rollout": 26},
            "selection": {"reset_boundary": 1,
                          "endpoint_or_end_of_rollout": 4},
        },
    }
    assert availability["common_rows"] == 4_320
    assert availability["common_train_rows"] == 3_854
    assert availability["common_selection_rows"] == 466
    assert sum(availability["common_train_family_counts"].values()) == 3_854
    assert sum(availability["common_selection_family_counts"].values()) == 466
    assert tuple(availability["common_train_family_counts"]) == C.FAMILIES
    assert availability["stable_id_list_digest"] == (
        "6eed553a7a3a09ef90be5a55e64209991ec8ef405fbe5981eb7356d0872efe49")
    assert availability["factorial_position_list_digest"] == (
        "26e5f1abea18829d42793893a6237d018962255b3598ffa747d4e23c2fb1b07c")
    assert availability["exclusion_digest"] == (
        "a9e26628bf750800c35d7cef3d43f5ae7efcc18acf2c0e46e1a85daaa3b55b22")
    assert availability["new_simulator_data_required"] is False


def test_historical_controls_are_explicitly_not_sample_matched() -> None:
    availability = C.TARGET_AVAILABILITY
    assert C.FROZEN_FACTORIAL["historical_train_rows"] == 3_922
    assert availability["historical_control_train_row_difference"] == 68
    assert availability["historical_control_train_row_difference"] == (
        C.FROZEN_FACTORIAL["historical_train_rows"]
        - availability["common_train_rows"])
    assert availability["historical_control_train_row_difference_fraction"] \
        == pytest.approx(68 / 3_922)
    assert availability["sample_matched_controls"] is False
    assert availability["historical_controls_only"] is True


def test_target_encoding_is_only_the_missing_h3_h4_train_cache() -> None:
    target = C.TARGET_CACHE_CONTRACT
    assert target["reuse_existing_H1_H2_train_targets"] is True
    assert target["reuse_existing_H1_H4_selection_targets_for_common_rows"] is True
    assert target["encode_only_missing_train_H3_H4"] is True
    assert target["dense_cache_shape_each"] == [3_854, 768, 1_024]
    assert target["dense_cache_dtype"] == "float16"
    assert target["missing_dense_cache_bytes_total"] == (
        2 * target["missing_dense_cache_bytes_each"])
    assert target["unique_train_frames_requiring_encoder_execution"] == 5_398
    assert target["row_horizon_cache_misses"] == 5_690
    assert target["output_entries"] == 7_708
    assert target["no_new_simulator_corpus"] is True
    assert target["no_intermediate_encoder_layers"] is True


def test_model_objective_order_and_checkpoint_policy_are_exact() -> None:
    model = C.MODEL_AND_OBJECTIVE
    training = C.TRAINING
    assert model["use_proprio"] is False
    assert model["input_cell"] == "RGB_PLUS_CONTROL_HISTORY"
    assert model["autoregressive_horizons"] == [1, 2, 3, 4]
    assert model["aggregate_loss"] == "(L1 + L2 + L3 + L4) / 4"
    assert model["aggregate_loss_scale"] == 0.25
    assert model["own_preceding_prediction_after_H1"] is True
    assert model["teacher_forcing_after_H1"] is False
    assert model["detach_preceding_prediction"] is False
    assert model["horizon_specific_weights"] is False
    assert model["utility_occupancy_safety_or_proprio_loss"] is False
    assert training["seeds"] == list(EXPECTED_SEEDS)
    assert training["data_order"] == (
        "filtered historical 3922-row batch plan; see data_order_contract")
    order = C.DATA_ORDER_CONTRACT
    assert order["historical_plan_arguments"] == "batch_plan(seed, epoch, 3922, 4)"
    assert order["historical_train_rows"] == 3_922
    assert order["common_train_rows"] == 3_854
    assert order["excluded_historical_train_rows"] == 68
    assert order["direct_batch_plan_on_3854_forbidden"] is True
    assert order["additional_rng_draws"] == 0
    assert order["survivor_relative_order_identical_to_historical"] is True
    assert "8 x 24" in order["per_seed_epoch_order_digests"]
    assert "8 x 24" in order["per_seed_epoch_batch_digests"]
    assert training["epochs"] == 24
    assert training["checkpoint_epoch"] == 21
    assert training["checkpoint_selection"] is False
    assert training["run_count"] == 8
    assert training["finite_weak_runs_retained"] is True
    assert training["extension_or_best_epoch"] is False


def test_smoke_resource_statistics_and_occupancy_boundaries_are_frozen() -> None:
    assert C.SMOKE_GATES["H3_H4_backpropagate_through_autoregressive_chain"] is True
    assert C.SMOKE_GATES[
        "each_component_perturbation_changes_only_its_registered_Li"] is True
    assert C.SMOKE_GATES["combined_loss_derivative_per_component"] == 0.25
    assert C.SMOKE_GATES["adaln_zero_warmup_permitted_only_for_chain_test"] is True
    assert C.SMOKE_GATES["warmup_state_discarded"] is True
    assert C.SMOKE_GATES["exact_registered_base_reloaded_after_smoke"] is True
    assert C.RESOURCE_GATES["preflight_full_epochs"] == 1
    assert C.RESOURCE_GATES["peak_vram_strictly_below_bytes"] == 28 * 2**30
    assert C.RESOURCE_GATES["free_system_ram_strictly_above_bytes"] == 20 * 2**30
    assert C.RESOURCE_GATES["batch_size_change_after_launch"] is False
    assert C.STATISTICAL_CONTRACT["paired_effect_count"] == 8
    assert C.STATISTICAL_CONTRACT["normalized_error_effect"] == (
        "two_step error minus four_step error")
    assert C.STATISTICAL_CONTRACT["primary_weighting"] == "equal-family"
    assert C.OCCUPANCY["qualified_true_target_horizons"] == [2, 3, 4]
    assert C.OCCUPANCY["H1_unavailable_and_not_reinterpreted"] is True
    assert C.OCCUPANCY["refit"] is False


def test_authority_is_predictive_dynamics_only_and_fail_closed() -> None:
    authority = C.AUTHORITY
    assert authority["train_eight_four_step_RGB_control_history_models"] is True
    assert authority["reuse_historical_RGB_one_step_and_two_step_metrics"] is True
    for prohibited in (
        "retrain_or_reselect_historical_comparators",
        "use_proprioceptive_cells",
        "new_simulator_corpus_or_branch_generation",
        "utility_scorer_or_predictor_utility_shard_access",
        "selected_action_or_planning_endpoint",
        "sealed_or_held_out_access",
        "attentive_readout_metric_reconstruction_or_execution",
        "longer_H5_plus_hierarchical_or_architecture_variant",
        "navigation_or_final_corpus",
        "retry_extend_reseed_or_best_epoch_select",
    ):
        assert authority[prohibited] is False
    assert C.FROZEN_EVALUATION["historical_comparator_model_forward"] is False
    assert C.FROZEN_EVALUATION["new_four_step_model_forward_only"] is True
    assert C.INTERPRETATION["H4_direct_fidelity_endpoints"] == [
        "changed_token_correct_future_cosine", "normalized_error_reduction"]
    assert C.INTERPRETATION[
        "useful_requires_both_H4_direct_fidelity_paired_means_strictly_positive"
    ] is True
    assert "95% t interval lies wholly below zero" in C.INTERPRETATION[
        "material_H1_H2_regression_rule"]
    assert C.INTERPRETATION[
        "H4_improves_with_material_H1_H2_regression"] == "REPORT_HORIZON_TRADEOFF"
    assert C.INTERPRETATION["planning_improvement_claim"] is False
    assert C.INTERPRETATION["utility_or_selected_action_endpoint"] is False


def test_failure_receipts_preserve_one_shot_custody_and_stop_scope() -> None:
    assert C.FAILURE_STAGE_CLASSIFICATION["smoke"] == "INVALID_SMOKE"
    assert C.FAILURE_STAGE_CLASSIFICATION["preflight"] == (
        "INVALID_RESOURCE_PREFLIGHT")
    assert C.FAILURE_STAGE_CLASSIFICATION["train-seed"] == "INVALID_TRAINING"
    assert C.FAILURE_STAGE_CLASSIFICATION["evaluate"] == "INVALID_EVALUATION"
    required = set(C.FAILURE_RECEIPT_REQUIRED_FIELDS)
    assert {
        "classification", "failed_stage", "exception_type", "exception",
        "contract_digest", "source_commit", "completed_training_seeds",
        "completed_evaluation_seed_count", "artifacts_present",
        "retry_resume_or_replacement_authorised", "nothing_remains_running",
    } <= required
    assert C.TERMINAL_REQUIREMENTS == {
        "all_started_child_processes_joined": True,
        "training_or_evaluation_processes_remaining": 0,
        "target_encoder_processes_remaining": 0,
        "no_automatic_follow_on_experiment": True,
        "no_predictor_utility_scoring": True,
        "no_final_corpus_generation": True,
    }


def test_contract_round_trip_rejects_authority_or_metric_mutation() -> None:
    contract = C.build_contract(_synthetic_source(), _synthetic_storage())
    assert C.validate_contract(contract) == contract
    changed = copy.deepcopy(contract)
    changed["authority"]["utility_scorer_or_predictor_utility_shard_access"] = True
    body = {key: value for key, value in changed.items()
            if key != C.CONTRACT_SELF_KEY}
    changed[C.CONTRACT_SELF_KEY] = C.digest(body)
    with pytest.raises(C.FourStepContractError):
        C.validate_contract(changed)


def test_installed_source_rebinding_fails_closed_without_touching_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _synthetic_source()
    contract = C.build_contract(source, _synthetic_storage())
    monkeypatch.setattr(C, "source_closure", lambda root=C.ROOT: source)
    assert C.validate_installed_source(contract) == contract
    drifted = copy.deepcopy(source)
    drifted["source_repository_commit"] = "2" * 40
    body = {key: value for key, value in drifted.items()
            if key != C.SOURCE_SELF_KEY}
    drifted[C.SOURCE_SELF_KEY] = C.digest(body)
    monkeypatch.setattr(C, "source_closure", lambda root=C.ROOT: drifted)
    with pytest.raises(C.FourStepContractError, match="installed four-step source"):
        C.validate_installed_source(contract)
