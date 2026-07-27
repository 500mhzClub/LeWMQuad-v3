from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import pytest

from lewm.benchmarks import (
    go2_geometry_anchored_deformable_bev_lift_joint_jepa_v1 as contract,
)


def _common(presentations: int) -> dict[str, object]:
    return {
        **{field: True for field in contract.COMMON_GATE_BOOLEAN_FIELDS},
        "presentations": presentations,
    }


def _u0() -> dict[str, object]:
    return {
        **_common(0),
        "A": 1.0,
        "aggregate_raster_nll": 0.8,
        "aggregate_raster_balanced_accuracy": 0.34,
        "aggregate_unknown_recall": 0.8,
        "aggregate_free_recall": 0.1,
        "aggregate_occupied_recall": 0.1,
        "rough_raster_balanced_accuracy": 0.34,
        "rough_raster_occupied_recall": 0.1,
        "paired_rgb_margin": 0.1,
        "paired_rgb_scene_wins": 5,
        "online_optimizer_update_count": 0,
        "target_ema_update_count": 0,
        "predictor_forward_count": 0,
        "predictor_objective_count": 0,
        "predictor_backward_count": 0,
        "predictor_optimizer_update_count": 0,
        "joint_optimizer_update_count": 0,
        "shared_gradient_ratio_evaluation_count": 0,
        "online_target_representation_bitwise_equal": True,
        "predictor_parameter_group_present": True,
        "semantic_objective_formula_exact": True,
        "latent_prediction_objective_formula_exact": True,
        "action_objective_formula_exact": True,
        "same_action_contrast_formula_exact": True,
        "deformable_lift_synthetic_mechanism_exact": True,
        "paired_correct_wrong_rgb_latents_finite_nonidentical": True,
        "initial_target_hard_sync_count": 1,
    }


def _u100() -> dict[str, object]:
    return {
        **_common(1_600),
        "A": 0.8,
        "aggregate_raster_nll": 0.6,
        "aggregate_raster_balanced_accuracy": 0.60,
        "aggregate_unknown_recall": 0.8,
        "aggregate_free_recall": 0.70,
        "aggregate_occupied_recall": 0.30,
        "rough_raster_balanced_accuracy": 0.50,
        "rough_raster_occupied_recall": 0.20,
        "paired_rgb_margin": 0.20,
        "paired_rgb_scene_wins": 6,
        "online_optimizer_update_count": 100,
        "target_ema_update_count": 100,
        "predictor_forward_count": 0,
        "predictor_objective_count": 0,
        "predictor_backward_count": 0,
        "predictor_optimizer_update_count": 0,
        "joint_optimizer_update_count": 0,
        "shared_gradient_ratio_evaluation_count": 0,
    }


def _u400() -> dict[str, object]:
    return {
        **_common(6_400),
        "A": 0.70,
        "aggregate_raster_nll": 0.42,
        "aggregate_raster_balanced_accuracy": 0.74,
        "aggregate_unknown_recall": 0.75,
        "aggregate_free_recall": 0.80,
        "aggregate_occupied_recall": 0.60,
        "rough_raster_balanced_accuracy": 0.72,
        "rough_raster_occupied_recall": 0.60,
        "paired_rgb_margin": 0.30,
        "paired_rgb_scene_wins": 8,
        "B400": 1.0,
        "B400_content_sha256": "a" * 64,
        "B400_frozen_before_joint_phase": True,
        "target_effective_rank": 8.0,
        "target_channel_variance": 2.0,
        "target_spatial_diversity": 0.5,
        "target_collapse_baselines_frozen_before_joint_phase": True,
        "online_optimizer_update_count": 400,
        "target_ema_update_count": 400,
        "predictor_forward_count": 0,
        "predictor_objective_count": 0,
        "predictor_backward_count": 0,
        "predictor_optimizer_update_count": 0,
        "joint_optimizer_update_count": 0,
        "shared_gradient_ratio_evaluation_count": 0,
    }


def _u1000() -> dict[str, object]:
    return {
        **_common(16_000),
        "A": 0.70,
        "aggregate_raster_nll": 0.38,
        "aggregate_raster_balanced_accuracy": 0.80,
        "aggregate_unknown_recall": 0.80,
        "aggregate_free_recall": 0.85,
        "aggregate_occupied_recall": 0.70,
        "rough_raster_balanced_accuracy": 0.772,
        "rough_raster_occupied_recall": 0.65,
        "paired_rgb_margin": 0.01,
        "paired_rgb_scene_wins": 8,
        "latent_prediction_loss": 0.90,
        "action_nll": 0.90 * contract.LOG9,
        "action_macro_balanced_accuracy": 0.23,
        "executed_action_beats_hardest_wrong_family_count": 6,
        "mean_wrong_action_energy": 2.0,
        "mean_executed_action_energy": 1.0,
        "non_hold_mean_hold_or_zero_action_energy": 2.0,
        "non_hold_mean_executed_action_energy": 1.0,
        "same_action_correct_next_deranged_nll": 0.90 * contract.LOG2,
        "same_action_correct_next_strict_win_rate": 0.65,
        "same_action_correct_next_positive_family_count": 6,
        "target_effective_rank": 6.0,
        "target_channel_variance": 1.5,
        "target_spatial_diversity": 0.375,
        "shared_gradient_ratio_evaluation_count": 600,
        "shared_gradient_ratio_pass_count": 600,
        "shared_gradient_ratio_failure_count": 0,
        "minimum_semantic_to_dynamics_gradient_ratio": 1.0 / 32.0,
        "maximum_semantic_to_dynamics_gradient_ratio": 32.0,
        "minimum_dynamics_to_semantic_gradient_ratio": 1.0 / 32.0,
        "maximum_dynamics_to_semantic_gradient_ratio": 32.0,
        "representation_gradient_finite_nonzero_update_count": 1_000,
        "predictor_gradient_finite_nonzero_update_count": 600,
        "semantic_gradient_finite_nonzero_joint_update_count": 600,
        "dynamics_gradient_finite_nonzero_joint_update_count": 600,
        "online_optimizer_update_count": 1_000,
        "target_ema_update_count": 1_000,
        "predictor_optimizer_update_count": 600,
        "joint_optimizer_update_count": 600,
        "target_gradient_tensor_count": 0,
        "target_optimizer_membership_count": 0,
    }


def _prior() -> dict[int, dict[str, object]]:
    return {0: _u0(), 100: _u100(), 400: _u400()}


def _rebound_raw(value: dict[str, object]) -> bytes:
    core = copy.deepcopy(value)
    core.pop("content_sha256", None)
    rebound = contract.with_content_sha256(core)
    return contract.canonical_json_bytes(rebound) + b"\n"


def test_governing_documents_and_exact_frozen_inputs_are_bound() -> None:
    assert contract.validate_governing_documents() == {
        contract.PREREGISTRATION_RELATIVE_PATH: contract.PREREGISTRATION_FILE_SHA256,
        contract.V3_TERMINAL_AUDIT_RELATIVE_PATH: contract.V3_TERMINAL_AUDIT_FILE_SHA256,
    }
    assert contract.PREREGISTRATION_COMMIT == "561e946443dd9eec668255708982565739de033b"
    assert contract.V3_TERMINAL_AUDIT_COMMIT == "cfc5253304f94d56c422ad3c09880c78be0513bc"
    assert contract.RUNTIME_BINDINGS == {
        path: copy.deepcopy(contract._direct.RUNTIME_BINDINGS[path])
        for path in contract.RUNTIME_BINDINGS
    }
    assert contract.TRAIN_ROLE_COUNTS == {"pairs": 4262, "unique_endpoints": 7777, "scenes": 72}
    assert contract.SELECTION_ROLE_COUNTS == {"pairs": 495, "unique_endpoints": 924, "scenes": 8}
    assert contract.ACTION_VOCABULARY[contract.HOLD_ACTION_INDEX] == "hold"
    assert tuple(contract.SELECTION_FAMILY_BINDINGS) == contract.SCENE_FAMILIES
    assert contract.TARGET_MAPPING_BINDINGS["train"]["mapping_sha256"] == (
        "c9c914422927670ffce8e2a967bf264725b9ae3c55c353ee0a1a16e44044196b"
    )


def test_source_paths_are_recursive_frozen_base_plus_new_sources() -> None:
    assert set(contract.REUSED_SOURCE_PATHS) == set(contract._direct.SOURCE_PATHS)
    assert set(contract.ADDITIVE_SOURCE_PATHS).issubset(contract.SOURCE_PATHS)
    assert len(contract.SOURCE_PATHS) > len(contract.REUSED_SOURCE_PATHS)
    assert list(contract.SOURCE_PATHS) == sorted(set(contract.SOURCE_PATHS))
    assert all(contract.safe_relative_source_path(path) == path for path in contract.SOURCE_PATHS)
    assert not any("sealed" in Path(path).parts or "heldout" in Path(path).parts for path in contract.SOURCE_PATHS)


def test_model_objective_optimizer_schedule_and_caps_match_preregistration() -> None:
    assert contract.IMPLEMENTATION_AUTHORS == (
        "/root",
        "/root/semantic_v3_terminal_audit_fast",
        "/root/semantic_v3_terminal_audit",
        "/root/projective_jepa_prereg_draft",
    )
    assert contract.OPERATIONAL_FAILURE_RECEIPT_PATHS == (
        "metrics.json", "artifact.json", "access.json", "result.json",
        "failure.json", "completed.json",
    )
    model = contract.model_config()
    assert model["bev_lattice"] == {
        "shape": [64, 64],
        "latent_width": 64,
        "forward_range_m": [-0.95, 5.35],
        "left_range_m": [-3.15, 3.15],
    }
    assert model["geometry_anchored_deformable_lift"]["samples_per_cell"] == 4
    assert model["geometry_anchored_deformable_lift"]["global_attention_pooling_mixing_or_auxiliary_bypass"] is False
    assert contract.objective_contract()["joint_updates_401_1000"]["total"] == (
        "S+P_latent_prediction+R_action+C_same_action_contrast"
    )
    assert contract.optimizer_contract()["ordered_groups"] == [
        "encoder", "lift_semantic", "predictor"
    ]
    schedule = contract.build_schedule_identity()
    assert schedule["presentations"] == 16_000
    assert schedule["observation_updates"] == [0, 100, 400, 1_000]
    runtime_inputs = contract.runtime_authorization_template()
    assert runtime_inputs == contract._direct.runtime_authorization_template()
    assert set(runtime_inputs) == {"raw", "n320", "schedule", "access_counter_fields"}
    assert runtime_inputs["raw"]["roles"] == {
        "train": contract.TRAIN_ROLE_COUNTS,
        "checkpoint_selection": contract.SELECTION_ROLE_COUNTS,
    }
    assert contract.GPU_ACTIVE_MINUTES_MAX == 30
    assert contract.WARMUP_UPDATES == 400


def test_canonical_helpers_are_duplicate_safe_content_bound() -> None:
    value = contract.with_content_sha256({"b": 2, "a": 1})
    raw = contract.canonical_json_bytes(value) + b"\n"
    assert raw.startswith(b'{"a":1,"b":2,"content_sha256":')
    assert contract.parse_canonical_json(raw, name="synthetic") == value
    with pytest.raises(ValueError, match="duplicate"):
        contract.parse_canonical_json(
            b'{"a":1,"a":2,"content_sha256":"' + b"0" * 64 + b'"}\n',
            name="duplicate",
        )


def test_update_zero_gate_is_fully_conjunctive() -> None:
    passed = contract.evaluate_gate(0, _u0())
    assert passed["passed"] is True
    assert passed["scientific_gate_evidence"] is True
    failed_metrics = _u0()
    failed_metrics["predictor_forward_count"] = 1
    failed = contract.evaluate_gate(0, failed_metrics)
    assert failed["passed"] is False
    assert failed["conjuncts"]["predictor_forward_count_equals_0"] is False
    assert failed["control"] == contract.CONTROL_UPDATE_ZERO_FAIL


def test_update_100_gate_exact_strict_and_inclusive_comparators() -> None:
    metrics = _u100()
    assert contract.evaluate_gate(100, metrics, {0: _u0()})["passed"] is True
    for field, replacement, conjunct in (
        ("A", 1.0, "A_strictly_lower_than_update_0"),
        ("aggregate_raster_nll", 0.8, "raster_nll_strictly_lower_than_update_0"),
        ("rough_raster_balanced_accuracy", 0.34, "rough_balanced_accuracy_strictly_above_update_0"),
        ("paired_rgb_margin", 0.1, "paired_rgb_margin_strictly_above_update_0"),
    ):
        candidate = copy.deepcopy(metrics)
        candidate[field] = replacement
        receipt = contract.evaluate_gate(100, candidate, {0: _u0()})
        assert receipt["passed"] is False
        assert receipt["conjuncts"][conjunct] is False


def test_update_400_gate_freezes_positive_jepa_and_collapse_baselines() -> None:
    receipt = contract.evaluate_gate(400, _u400(), {100: _u100()})
    assert receipt["passed"] is True
    assert receipt["conjuncts"]["raster_nll_at_most_point42"] is True
    assert receipt["conjuncts"]["B400_finite_strictly_positive"] is True
    assert receipt["conjuncts"]["target_spatial_diversity_baseline_finite_strictly_positive"] is True
    candidate = _u400()
    candidate["paired_rgb_margin"] = _u100()["paired_rgb_margin"]
    failed = contract.evaluate_gate(400, candidate, {100: _u100()})
    assert failed["passed"] is False
    assert failed["conjuncts"]["paired_rgb_margin_strictly_above_update_100"] is False
    candidate = _u400()
    candidate["B400_frozen_before_joint_phase"] = False
    assert contract.evaluate_gate(400, candidate, {100: _u100()})["passed"] is False


def test_update_1000_joint_jepa_perception_collapse_and_gradient_gate_pass() -> None:
    receipt = contract.evaluate_gate(1_000, _u1000(), _prior())
    assert receipt["passed"] is True
    assert receipt["conjuncts"]["latent_prediction_loss_at_most_point90_B400"] is True
    assert receipt["conjuncts"]["target_effective_rank_retains_point75_update_400"] is True
    assert receipt["conjuncts"]["all_600_joint_updates_passed_shared_gradient_gate"] is True
    assert receipt["control"] == contract.CONTROL_PASS


@pytest.mark.parametrize(
    ("field", "replacement", "conjunct"),
    (
        ("action_nll", 0.95 * math.log(9.0), "action_nll_strictly_below_point95_log9"),
        ("action_macro_balanced_accuracy", 2.0 / 9.0, "action_macro_balanced_accuracy_strictly_above_two_ninths"),
        ("mean_wrong_action_energy", 1.0, "mean_wrong_action_energy_strictly_above_executed"),
        ("same_action_correct_next_deranged_nll", 0.95 * math.log(2.0), "same_action_correct_next_deranged_nll_strictly_below_point95_log2"),
        ("minimum_semantic_to_dynamics_gradient_ratio", (1.0 / 32.0) - 1e-9, "semantic_to_dynamics_gradient_ratio_bounds_exact"),
        ("shared_gradient_ratio_pass_count", 599, "all_600_joint_updates_passed_shared_gradient_gate"),
    ),
)
def test_update_1000_strict_jepa_and_gradient_failures(
    field: str, replacement: float | int, conjunct: str
) -> None:
    metrics = _u1000()
    metrics[field] = replacement
    receipt = contract.evaluate_gate(1_000, metrics, _prior())
    assert receipt["passed"] is False
    assert receipt["conjuncts"][conjunct] is False
    assert receipt["control"] == contract.CONTROL_UPDATE_1000_FAIL


def test_update_401_non_scientific_phase_switch_receipt() -> None:
    metrics = {
        "optimizer_identity_unchanged": True,
        "optimizer_parameter_group_membership_unchanged": True,
        "joint_objective_formula_exact": True,
        "online_representation_gradient_finite_nonzero": True,
        "predictor_gradient_finite_nonzero": True,
        "target_gradients_absent": True,
        "shared_gradient_contribution_gate_passed": True,
        "online_optimizer_update_count": 401,
        "target_ema_update_count": 401,
        "predictor_optimizer_update_count": 1,
        "joint_optimizer_update_count": 1,
    }
    receipt = contract.evaluate_update_401_phase_switch(metrics)
    assert receipt["passed"] is True
    assert receipt["scientific_gate_evidence"] is False
    metrics["optimizer_identity_unchanged"] = False
    assert contract.evaluate_update_401_phase_switch(metrics)["passed"] is False


def test_gate_rejects_missing_prior_nonfinite_and_prior_failure() -> None:
    with pytest.raises(ValueError, match="prior metrics"):
        contract.evaluate_gate(100, _u100())
    candidate = _u100()
    candidate["A"] = math.nan
    with pytest.raises(ValueError, match="finite"):
        contract.evaluate_gate(100, candidate, {0: _u0()})
    receipt = contract.evaluate_gate(
        100, _u100(), {0: _u0()}, prior_gates_passed=False
    )
    assert receipt["passed"] is False
    assert receipt["conjuncts"]["prior_gates_passed"] is False


def _synthetic_authority_root(tmp_path: Path) -> tuple[bytes, dict[str, object]]:
    for index, relative in enumerate(contract.SOURCE_PATHS):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"# synthetic source {index}\n".encode("ascii"))
    bindings = []
    for relative in contract.SOURCE_PATHS:
        raw = (tmp_path / relative).read_bytes()
        bindings.append({
            "path": relative,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
        })
    manifest = contract.with_content_sha256({
        "schema": contract.SOURCE_MANIFEST_SCHEMA,
        "status": "PASS_SOURCE_CLOSURE",
        "entrypoints": list(contract.SOURCE_MANIFEST_ENTRYPOINTS),
        "forced_dynamic_sources": list(contract.SOURCE_PATHS),
        "excluded_runtime_categories": list(contract.PROHIBITED_RUNTIME_CATEGORIES),
        "source_paths": list(contract.SOURCE_PATHS),
        "source_bindings": bindings,
        "source_bindings_sha256": contract.canonical_json_sha256(bindings),
        "source_count": len(bindings),
        "generated_input_open_count": 0,
        "checkpoint_or_tensor_open_count": 0,
        "sealed_or_heldout_open_count": 0,
        "whole_tree_export_authorized": False,
        "authority": contract.SOURCE_ONLY_AUTHORITY,
    })
    raw = contract.canonical_json_bytes(manifest) + b"\n"
    path = tmp_path / contract.SOURCE_MANIFEST_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    for relative in (
        contract.PREREGISTRATION_RELATIVE_PATH,
        contract.V3_TERMINAL_AUDIT_RELATIVE_PATH,
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((contract.ROOT / relative).read_bytes())
    return raw, manifest


def test_manifest_review_and_authorization_validators_are_fail_closed(
    tmp_path: Path,
) -> None:
    manifest_raw, manifest = _synthetic_authority_root(tmp_path)
    assert contract.validate_source_manifest(manifest_raw, tmp_path) == manifest
    manifest_binding = contract.artifact_binding(
        contract.SOURCE_MANIFEST_RELATIVE_PATH,
        manifest_raw,
        content_sha256=manifest["content_sha256"],
    )
    expected_sources = contract.current_source_bindings(tmp_path)
    source_freeze_commit = "0123456789abcdef0123456789abcdef01234567"
    review = contract.with_content_sha256({
        "schema": contract.REVIEW_SCHEMA,
        "status": "PASS_SOURCE_AND_SCIENCE",
        "implementation_authors": list(contract.IMPLEMENTATION_AUTHORS),
        "reviewer": "/root/independent_contract_reviewer",
        "source_freeze_commit": source_freeze_commit,
        "reviewed_sources": expected_sources,
        "source_manifest": manifest_binding,
        "preregistration": contract.preregistration_binding(),
        "v3_terminal_audit": contract.v3_terminal_audit_binding(),
        "science_contract": contract.science_contract(),
        "source_only_checks": {
            "generated_inputs_opened": [],
            "checkpoints_or_tensors_opened": [],
            "runtime_outputs_or_traces_opened": [],
            "sealed_or_heldout_opened": [],
        },
        "scientific_checks": contract.SCIENTIFIC_REVIEW_CHECKS,
        "findings": [],
        "authority": contract.REVIEW_AUTHORITY,
    })
    review_raw = contract.canonical_json_bytes(review) + b"\n"
    assert contract.validate_review(
        review_raw, manifest_binding, root=tmp_path
    ) == review
    review_path = tmp_path / contract.REVIEW_RELATIVE_PATH
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_bytes(review_raw)
    review_binding = contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=review["content_sha256"],
    )
    authorization = contract.with_content_sha256({
        "schema": contract.AUTHORIZATION_SCHEMA,
        "status": contract.AUTHORIZATION_STATUS,
        "authorizer": "/root/independent_authorizer",
        "source_freeze_commit": source_freeze_commit,
        "independent_source_review": review_binding,
        "preregistration": contract.preregistration_binding(),
        "v3_terminal_audit": contract.v3_terminal_audit_binding(),
        "runtime_inputs": contract.runtime_authorization_template(),
        "experiment": contract.science_contract(),
        "authority": contract.EXECUTION_AUTHORITY,
    })
    authorization_raw = contract.canonical_json_bytes(authorization) + b"\n"
    assert contract.validate_authorization(
        authorization_raw, review_binding, root=tmp_path
    ) == authorization

    assert contract.SCIENTIFIC_REVIEW_CHECKS[
        "source_freeze_commit_matches_reviewed_tree"
    ] is True
    for implementation_author in contract.IMPLEMENTATION_AUTHORS:
        changed_review = copy.deepcopy(review)
        changed_review["reviewer"] = implementation_author
        with pytest.raises(PermissionError, match="source review"):
            contract.validate_review(
                _rebound_raw(changed_review), manifest_binding, root=tmp_path
            )
    changed_review = copy.deepcopy(review)
    changed_review["implementation_authors"] = list(
        contract.IMPLEMENTATION_AUTHORS[:-1]
    )
    with pytest.raises(PermissionError, match="source review"):
        contract.validate_review(
            _rebound_raw(changed_review), manifest_binding, root=tmp_path
        )
    for malformed_commit in (
        "A" * 40,
        "0" * 39,
        "g" * 40,
    ):
        changed_review = copy.deepcopy(review)
        changed_review["source_freeze_commit"] = malformed_commit
        with pytest.raises(PermissionError, match="lowercase 40-hex"):
            contract.validate_review(
                _rebound_raw(changed_review), manifest_binding, root=tmp_path
            )

    changed_authorization = copy.deepcopy(authorization)
    changed_authorization["source_freeze_commit"] = "1" * 40
    with pytest.raises(PermissionError, match="execution authorization"):
        contract.validate_authorization(
            _rebound_raw(changed_authorization), review_binding, root=tmp_path
        )
    for forbidden_authorizer in (
        *contract.IMPLEMENTATION_AUTHORS,
        review["reviewer"],
    ):
        changed_authorization = copy.deepcopy(authorization)
        changed_authorization["authorizer"] = forbidden_authorizer
        with pytest.raises(PermissionError, match="execution authorization"):
            contract.validate_authorization(
                _rebound_raw(changed_authorization), review_binding, root=tmp_path
            )

    changed = bytearray(manifest_raw)
    changed[-2] = ord(" ")
    with pytest.raises((ValueError, PermissionError)):
        contract.validate_source_manifest(bytes(changed), tmp_path)
