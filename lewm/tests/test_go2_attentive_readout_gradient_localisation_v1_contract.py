"""Focused source-only tests for the gradient-localisation contract."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from lewm.oracle import go2_attentive_readout_gradient_localisation_v1_contract as C


ROOT = Path(__file__).resolve().parents[2]


def _closure() -> dict:
    payload = {
        "schema": C.SOURCE_CLOSURE_SCHEMA,
        "source_repository_commit": "a" * 40,
        "source_repository_clean": True,
        "base_source_commit": C.BASE_SOURCE_COMMIT,
        "exact_committed_additive_path_diff": list(C.NEW_SOURCE_PATHS),
        "frozen_dependency_files": {},
        "additive_files": {},
    }
    return {**payload, C.SOURCE_CLOSURE_SELF_KEY: C.digest(payload)}


def test_frozen_failure_lineage_replays_exact_immutable_bytes() -> None:
    lineage = C.failed_smoke_lineage(ROOT)
    assert lineage == {
        "prerequisite_amendment_digest": C.AMENDMENT_ARTIFACT_DIGEST,
        "prerequisite_amendment_sha256": C.AMENDMENT_ARTIFACT_SHA256,
        "prerequisite_amendment_byte_count": C.AMENDMENT_ARTIFACT_BYTE_COUNT,
        "amendment_source_closure_digest": C.AMENDMENT_SOURCE_CLOSURE_DIGEST,
        "production_smoke_failure_digest": C.SMOKE_FAILURE_DIGEST,
        "production_smoke_failure_sha256": C.SMOKE_FAILURE_SHA256,
        "production_smoke_failure_byte_count": C.SMOKE_FAILURE_BYTE_COUNT,
        "traceback_sha256": C.SMOKE_TRACEBACK_SHA256,
        "exception_binding_sha256": C.SMOKE_EXCEPTION_BINDING_SHA256,
        "completed_optimizer_updates": 0,
        "checkpoint_published": False,
        "scientific_attempt_started": False,
        "production_smoke_work_directory_present": True,
        "production_smoke_work_directory_empty": True,
        "preserved_artifacts_mutable": False,
    }


def test_build_validate_binds_source_lineage_and_static_contract() -> None:
    value = C.build_contract(_closure(), C.failed_smoke_lineage(ROOT))
    assert C.validate_contract(value) == value
    tampered = copy.deepcopy(value)
    tampered["backend_matrix"]["A"]["autocast"] = True
    unsigned = dict(tampered)
    unsigned.pop(C.CONTRACT_SELF_KEY)
    tampered[C.CONTRACT_SELF_KEY] = C.digest(unsigned)
    with pytest.raises(C.GradientLocalisationContractError,
                       match="contract changed"):
        C.validate_contract(tampered)


def test_matrix_freezes_production_truth_and_exact_ab_equivalence() -> None:
    assert C.BACKEND_MATRIX["A"]["parameter_dtype"] == "float32"
    assert C.BACKEND_MATRIX["A"]["autocast"] is False
    assert C.BACKEND_MATRIX["B"]["semantic_relation_to_A"].startswith(
        "identical")
    assert C.BACKEND_MATRIX["C"]["sdpa"].startswith("force math-only")
    assert "official non-SDPA" in C.BACKEND_MATRIX["D"]["attention"]
    assert C.FORWARD_EQUIVALENCE["a_b"].startswith("exact")
    assert C.FORWARD_EQUIVALENCE["a_c_d_absolute_tolerance"] == 1e-5
    assert C.FORWARD_EQUIVALENCE["a_c_d_relative_tolerance"] == 1.3e-6
    assert C.FORWARD_EQUIVALENCE["relative_denominator_floor"] == 1e-12


def test_exact_ten_pass_budget_has_frozen_empty_adamw_and_no_step_or_clip() -> None:
    assert C.EXECUTION_COUNTS == {
        "exact_reproduction": 1,
        "hook_inventory": 1,
        "loss_isolation": 4,
        "backend_matrix": 4,
        "fresh_model_constructions": 10,
        "forwards": 10,
        "backwards": 10,
        "optimizer_constructions": 10,
        "optimizer_steps": 0,
        "gradient_clips": 0,
        "fixture_validation_row_record_opens": 4,
        "fixture_validation_latent_shard_opens": 4,
        "unique_fit_row_record_files": 4,
        "unique_fit_latent_shard_files": 4,
        "pass_latent_shard_loads": 40,
        "batch_presentations": 10,
        "examples_presented": 40,
    }
    assert C.LOSS_ISOLATION_PASSES == (
        "progress_only", "safety_only", "completion_only",
        "frozen_summed_loss")


def test_hook_contract_covers_every_requested_semantic_module() -> None:
    by_role: dict[str, list[str]] = {}
    for row in C.HOOK_TARGETS:
        by_role.setdefault(row["role"], []).append(row["path"])
    assert by_role["token_projection"] == ["token_projection"]
    assert len(by_role["self_attention_block"]) == 3
    assert len(by_role["self_attention_kernel"]) == 3
    assert len(by_role["cross_attention_block"]) == 1
    assert len(by_role["cross_attention_kernel"]) == 1
    assert len(by_role["layer_norm"]) == 8
    assert len(by_role["mlp"]) == 4
    assert by_role["action_goal_encoder"] == ["context"]
    assert set(by_role[role][0] for role in (
        "progress_output_head", "safety_output_head",
        "completion_output_head")) == {"progress", "safety", "completion"}
    assert by_role["virtual_horizon_embedding_buffer"] == [
        "horizon_embeddings"]
    assert by_role["virtual_component_queries"] == ["pooler.query_tokens"]


def test_inventory_schemas_and_terminal_classes_are_exact() -> None:
    for field in (
        "fully_qualified_name", "module_path", "module_type", "shape",
        "parameter_dtype", "gradient_dtype", "gradient_is_none",
        "finite_count", "nan_count", "positive_infinity_count",
        "negative_infinity_count", "maximum_absolute_finite_value",
        "finite_only_l2_norm", "first_nonfinite_flat_index",
        "first_nonfinite_multi_index", "gradient_tensor_digest",
    ):
        assert field in C.PARAMETER_INVENTORY_FIELDS
    assert C.MECHANISM_CLASSIFICATIONS == (
        "BACKEND_NUMERICAL_DEFECT_CONTRACT_PRESERVING",
        "IMPLEMENTATION_DEFECT_CONTRACT_PRESERVING",
        "ARCHITECTURE_OR_OBJECTIVE_CHANGE_REQUIRED",
    )
    assert C.STOP_RULES["automatic_repair_or_training"] is False
    assert C.CLASSIFICATION_RULE["both_C_and_D_pass"].startswith("BACKEND")
    assert C.CLASSIFICATION_RULE[
        "nonreproduction_or_harness_failure_mechanism"] is None
    assert set(C.PRE_BACKWARD_FINITE_FIELDS) >= {
        "all_model_parameters_finite", "all_inputs_finite",
        "all_activations_finite", "all_targets_finite",
        "all_component_losses_finite", "total_loss_finite",
    }
    assert set(C.TERMINAL_OFFENDER_FIELDS) == {
        "first_reverse_module_with_finite_downstream_and_nonfinite_upstream",
        "first_nonfinite_parameter_gradient", "all_nonfinite_parameter_gradients",
    }
    assert C.static_contract()["authority"]["repair_authorised"] is False
    assert C.static_contract()["authority"]["training_authorised"] is False


def test_runtime_is_pinned_to_authorised_rocm_environment() -> None:
    assert C.EXECUTION_ENVIRONMENT["python"].endswith(
        "genesis_rocm_0_4_6_v1/bin/python")
    assert C.EXECUTION_ENVIRONMENT["torch_version"] == "2.12.0+rocm7.2"
    assert C.EXECUTION_ENVIRONMENT["torch_hip_version"] == "7.2.53211"
    assert C.EXECUTION_ENVIRONMENT["device"] == "cuda:0"
    assert C.EXECUTION_ENVIRONMENT["device_architecture"] == "gfx1201"
    assert C.EXECUTION_ENVIRONMENT["device_capability"] == [12, 0]


def test_additive_closure_has_exactly_four_paths_and_no_repair_source() -> None:
    assert len(C.NEW_SOURCE_PATHS) == 4
    assert all("repair" not in path and "train" not in Path(path).name
               for path in C.NEW_SOURCE_PATHS)
    assert all("predictor" not in path for path in C.NEW_SOURCE_PATHS)


def test_all_passes_share_exact_fixture_state_loss_and_no_step() -> None:
    assert C.ALL_PASS_INVARIANTS["fixture_digest"] == C.FROZEN_FIXTURE_DIGEST
    assert C.ALL_PASS_INVARIANTS["initial_state_digest"] == (
        C.FROZEN_INITIAL_STATE_DIGEST)
    assert C.ALL_PASS_INVARIANTS["model_mode"] == "train"
    assert C.ALL_PASS_INVARIANTS["activation_checkpointing"] is True
    assert C.ALL_PASS_INVARIANTS["optimiser_constructed"] is True
    assert C.ALL_PASS_INVARIANTS[
        "optimizer_state_before_and_after_backward"] == "empty and unchanged"
    assert C.ALL_PASS_INVARIANTS["optimizer_step"] is False
    assert C.LOSS_CONTRACT["effective_batch_denominator"] == 64
    assert C.LOSS_CONTRACT["progress"].startswith("raw mse_loss")
    assert C.LOSS_CONTRACT["safety"].startswith(
        "raw binary_cross_entropy_with_logits")
    assert "one final FP32 division" in C.LOSS_CONTRACT[
        "frozen_summed_loss"]
    assert C.OFFICIAL_CALCULATION[
        "expected_sdpa_invocations_per_forward_backward"]["total"] == 7
