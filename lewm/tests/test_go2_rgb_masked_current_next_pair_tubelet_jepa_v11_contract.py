from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT
    / "lewm/benchmarks/go2_rgb_masked_current_next_pair_tubelet_jepa_v11.py"
)
_MODULES_BEFORE = set(sys.modules)
_SPEC = importlib.util.spec_from_file_location(
    "_test_masked_pair_tubelet_v11_contract", CONTRACT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
contract = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = contract
_SPEC.loader.exec_module(contract)
_IMPORTED = set(sys.modules) - _MODULES_BEFORE


_ACTION_ROW_COUNTS = {
    action: (60 if action == "hold" else (55 if index < 3 else 54))
    for index, action in enumerate(contract.ACTION_VOCABULARY)
}


def _retrieval(
    *,
    action_nll: float,
    target_nll: float,
    same_action_target_nll: float,
    two_target_nll: float,
    correct_per_action: int | list[int],
    energy_ratio: float,
    strict_win_count: int,
    positive_family_count: int,
) -> dict[str, Any]:
    correct_counts = (
        [correct_per_action] * 9
        if type(correct_per_action) is int
        else list(correct_per_action)
    )
    per_action: dict[str, dict[str, int | float]] = {}
    recalls: list[float] = []
    total_correct = 0
    for action, correct in zip(contract.ACTION_VOCABULARY, correct_counts):
        count = _ACTION_ROW_COUNTS[action]
        recall = correct / count
        per_action[action] = {
            "row_count": count,
            "mean_nll": action_nll,
            "recall": recall,
        }
        recalls.append(recall)
        total_correct += correct

    families: dict[str, dict[str, Any]] = {}
    for index, family in enumerate(contract.SCENE_FAMILIES):
        binding = contract.SELECTION_FAMILY_BINDINGS[family]
        margin = 0.2 if index < positive_family_count else 0.0
        families[family] = {
            "scene_id": binding["scene_id"],
            "row_count": binding["row_count"],
            "same_action_row_count": binding["same_action_row_count"],
            "non_hold_row_count": binding["non_hold_row_count"],
            "deranged_minus_correct_energy": margin,
            "current_target_minus_correct_energy": margin,
            "cyclic_wrong_minus_executed_energy": margin,
            "hardest_wrong_minus_executed_energy": margin,
            "hold_minus_non_hold_executed_energy": margin,
            "permuted_minus_executed_energy": margin,
            "hold_action_rows_match_non_hold_rows": True,
        }

    return {
        "all_values_finite": True,
        "energy_values_within_closed_zero_four": True,
        "target_candidate_order_and_counts_exact": True,
        "same_action_target_mapping_exact": True,
        "selection_action_permutation_exact": True,
        "reference_values_immutable": True,
        "action_equal_logit_reference": math.log(9.0),
        "two_target_equal_logit_reference": math.log(2.0),
        "action_retrieval_nll": action_nll,
        "action_retrieval_top1_accuracy": total_correct / 495,
        "per_executed_action_action_retrieval": per_action,
        "action_retrieval_macro_balanced_accuracy": sum(recalls) / 9,
        "target_retrieval_nll": target_nll,
        "same_action_target_retrieval_nll": same_action_target_nll,
        "hold_target_retrieval_nll": target_nll,
        "non_hold_target_retrieval_nll": target_nll,
        "same_action_two_target_nll": two_target_nll,
        "target_retrieval_top1_count": 300,
        "target_retrieval_top1_accuracy": 300 / 495,
        "same_action_strict_win_count": strict_win_count,
        "same_action_strict_win_rate": strict_win_count / 494,
        "same_action_correct_energy": energy_ratio,
        "same_action_deranged_energy": 1.0,
        "same_action_correct_to_deranged_ratio": energy_ratio,
        "non_hold_correct_energy": energy_ratio,
        "non_hold_current_target_energy": 1.0,
        "non_hold_correct_to_current_ratio": energy_ratio,
        "executed_action_energy": energy_ratio,
        "cyclic_wrong_action_energy": 1.0,
        "hardest_wrong_action_energy": 1.0,
        "permuted_action_energy": 1.0,
        "non_hold_executed_action_energy": energy_ratio,
        "non_hold_hold_action_energy": 1.0,
        "executed_to_cyclic_ratio": energy_ratio,
        "executed_to_hardest_wrong_ratio": energy_ratio,
        "executed_to_permuted_ratio": energy_ratio,
        "non_hold_executed_to_hold_ratio": energy_ratio,
        "all_row_count": 495,
        "same_action_row_count": 494,
        "fallback_row_count": 1,
        "hold_row_count": 60,
        "non_hold_row_count": 435,
        "target_candidate_count": 1_425,
        "action_candidate_count": 9,
        "all_wrong_action_candidate_count": 3_960,
        "selection_target_mapping_sha256": (
            contract.TARGET_MAPPING_BINDINGS["checkpoint_selection"]
            ["mapping_sha256"]
        ),
        "selection_action_permutation_sha256": (
            contract.SELECTION_ACTION_PERMUTATION_BINDING["mapping_sha256"]
        ),
        "per_family": families,
        "deranged_positive_family_margin_count": positive_family_count,
        "current_target_positive_family_margin_count": positive_family_count,
        "cyclic_positive_family_margin_count": positive_family_count,
        "hold_positive_family_margin_count": positive_family_count,
        "permuted_positive_family_margin_count": positive_family_count,
    }


def _metrics(
    *,
    masked_loss: float,
    rank: float,
    action_nll_factor: float,
    target_nll: float,
    same_action_target_nll: float,
    two_target_nll: float,
    correct_per_action: int | list[int],
    energy_ratio: float,
    strict_win_count: int,
    positive_family_count: int,
) -> dict[str, Any]:
    return {
        "all_values_finite": True,
        "ema_target_gradient_free": True,
        "pair_count": 495,
        "scene_family_count": 8,
        "non_hold_pair_count": 435,
        "masked_future_jepa_loss": masked_loss,
        "normalized_projected_future_effective_rank": rank,
        "normalized_projected_future_cross_sample_variance": 2.0,
        "normalized_projected_future_off_diagonal_covariance": 0.1,
        "normalized_projected_future_spatial_diversity": 4.0,
        "detached_target_future_effective_rank": 48.0,
        "detached_target_future_cross_sample_variance": 2.0,
        "detached_target_future_off_diagonal_covariance": 0.1,
        "detached_target_future_spatial_diversity": 4.0,
        "true_pair_mse": 0.80,
        "shuffled_next_mse": 1.0,
        "shuffled_current_mse": 1.0,
        "mean_target_mse": 1.0,
        "factorized_retrieval": _retrieval(
            action_nll=action_nll_factor * math.log(9.0),
            target_nll=target_nll,
            same_action_target_nll=same_action_target_nll,
            two_target_nll=two_target_nll,
            correct_per_action=correct_per_action,
            energy_ratio=energy_ratio,
            strict_win_count=strict_win_count,
            positive_family_count=positive_family_count,
        ),
    }


def _update0() -> dict[str, Any]:
    metric = _metrics(
        masked_loss=1.0,
        rank=10.0,
        action_nll_factor=1.0,
        target_nll=1.0,
        same_action_target_nll=1.0,
        two_target_nll=math.log(2.0),
        correct_per_action=[55, 0, 0, 0, 0, 0, 0, 0, 0],
        energy_ratio=1.0,
        strict_win_count=0,
        positive_family_count=0,
    )
    metric["normalized_projected_future_cross_sample_variance"] = 4.0
    metric["normalized_projected_future_spatial_diversity"] = 8.0
    return {
        **metric,
        "all_action_predictions_bitwise_equal": True,
        "all_action_unordered_pair_count": 36,
        "all_action_prediction_row_count": 495,
    }


def _update0_metric() -> dict[str, Any]:
    update0 = _update0()
    return {
        field: deepcopy(update0[field])
        for field in contract.PHASE_A_METRIC_FIELDS
    }


def _integrity(update: int) -> dict[str, Any]:
    return {
        "rng_state_preserved": True,
        "state_mutation_count": 0,
        "future_leakage_prohibition_passed": True,
        "target_path_nonvacuity_passed": True,
        "online_target_autograd_separation_passed": True,
        "ema_inventory_exact": True,
        "ema_update_count": update,
        "expected_ema_update_count": update,
        "normalized_population_exact": True,
        "all_nine_candidates_exact": True,
        "observation_row_count": 495,
    }


def _update100() -> dict[str, Any]:
    return _metrics(
        masked_loss=0.80,
        rank=20.0,
        action_nll_factor=0.98,
        target_nll=0.90,
        same_action_target_nll=0.89,
        two_target_nll=0.65,
        correct_per_action=10,
        energy_ratio=0.98,
        strict_win_count=300,
        positive_family_count=6,
    )


def _update400() -> dict[str, Any]:
    return _metrics(
        masked_loss=0.70,
        rank=33.0,
        action_nll_factor=0.94,
        target_nll=0.80,
        same_action_target_nll=0.79,
        two_target_nll=0.60,
        correct_per_action=14,
        energy_ratio=0.98,
        strict_win_count=320,
        positive_family_count=6,
    )


def _terminal() -> dict[str, Any]:
    return _metrics(
        masked_loss=0.60,
        rank=48.0,
        action_nll_factor=0.89,
        target_nll=0.70,
        same_action_target_nll=0.69,
        two_target_nll=0.55,
        correct_per_action=19,
        energy_ratio=0.95,
        strict_win_count=340,
        positive_family_count=6,
    )


def _set_action_nll(metrics: dict[str, Any], value: float) -> None:
    retrieval = metrics["factorized_retrieval"]
    retrieval["action_retrieval_nll"] = value
    for row in retrieval["per_executed_action_action_retrieval"].values():
        row["mean_nll"] = value


def test_contract_import_is_stdlib_only_and_binds_frozen_preregistration() -> None:
    imported_roots = {name.partition(".")[0] for name in _IMPORTED}
    assert imported_roots.isdisjoint(
        {"torch", "numpy", "PIL", "cv2", "jax", "tensorflow"}
    )
    assert contract.PREREGISTRATION_COMMIT == (
        "46de4c1b6a89dad43550b62a6e9327dec0a7b9da"
    )
    raw = (ROOT / contract.PREREGISTRATION_RELATIVE_PATH).read_bytes()
    assert len(raw) == 27_808
    assert hashlib.sha256(raw).hexdigest() == (
        "bbc4fa556788ce8df90c417aaa074bb7daf2aea47e4465434a4f119e18530dee"
    )
    assert contract.OUTPUT_ROOT_RELATIVE_PATH.endswith(
        "rgb_masked_current_next_pair_tubelet_jepa_probe_v11"
    )


def test_current_source_bindings_accept_bound_pretty_json_documents() -> None:
    bindings = contract.current_source_bindings(ROOT)
    assert bindings[contract.PREREGISTRATION_RELATIVE_PATH] == (
        contract.PREREGISTRATION_FILE_SHA256
    )
    assert bindings[contract.PRIOR_TERMINAL_AUDIT_RELATIVE_PATH] == (
        contract.PRIOR_TERMINAL_AUDIT_FILE_SHA256
    )
    assert bindings[contract.SOURCE_MANIFEST_RELATIVE_PATH]


def test_model_schedule_mapping_and_source_contract_are_exact() -> None:
    assert contract.v11_model_config() == {
        "image_size": 112,
        "patch_size": 7,
        "feature_dim": 192,
        "encoder_depth": 6,
        "encoder_heads": 6,
        "encoder_mlp_ratio": 4,
        "encoder_dropout": 0.0,
        "future_token_count": 256,
        "action_count": 9,
        "target_ema_momentum": 0.996,
        "normalization_epsilon": 1e-8,
        "whitening_epsilon": 1e-4,
        "whitening_variance_weight": 0.50,
        "whitening_covariance_weight": 0.02,
    }
    assert contract.ACTION_VOCABULARY[6] == "hold"
    assert contract.TARGET_MAPPING_BINDINGS["train"]["mapping_sha256"] == (
        "c9c914422927670ffce8e2a967bf264725b9ae3c55c353ee0a1a16e44044196b"
    )
    assert contract.TARGET_MAPPING_BINDINGS["checkpoint_selection"][
        "mapping_sha256"
    ] == "95d42273a8319316ad68781cb2158146e7672eda529984c3aeddc0937d87a9c1"
    assert contract.SELECTION_ACTION_PERMUTATION_BINDING["mapping_sha256"] == (
        "2740be362829c172a06aebae0d077e69ede8af80cbf6f00569eb460dc559bb0f"
    )
    schedule = contract.build_schedule_identity("phase_a")
    assert schedule["updates"] == 1_000
    assert schedule["presentations"] == 16_000
    assert schedule["prefix_sha256"] == {
        "100": "9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51",
        "400": "6e7e5cc766c0a768b5771181cfaf2583598c1c22e5d4fc19e6ff1b245a5c8f92",
        "1000": "3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528",
    }
    assert contract.build_schedule_identity("phase_b") == {
        "phase": "phase_b",
        "authorized": False,
        "source": None,
        "seed": None,
        "updates": 0,
        "presentations": 0,
        "microbatch_size": 0,
        "microbatches_per_update": 0,
        "effective_batch_size": 0,
        "checkpoints": [],
        "prefix_sha256": {},
        "reuse_same_frozen_prefix_independently": False,
    }
    with pytest.raises(ValueError, match="phase_a or disabled phase_b"):
        contract.build_schedule_identity("other")
    assert contract.FROZEN_V10_RUNNER_RELATIVE_PATH in contract.SOURCE_PATHS
    assert contract.V11_MODEL_RELATIVE_PATH in contract.SOURCE_PATHS


def test_parameter_and_ema_ownership_are_complete_and_phase_b_is_denied() -> None:
    assert contract.PHASE_A_EXACT_FROZEN_PARAMETER_NAMES == (
        "encoder.cls_token",
    )
    assert len(contract.TARGET_EMA_PARAMETER_PAIRS) == 81
    assert contract.TARGET_EMA_PARAMETER_PAIRS[0] == (
        "encoder.cls_token",
        "target_encoder.cls_token",
    )
    assert contract.TARGET_EMA_PARAMETER_PAIRS[-3:] == (
        (
            "online_future_temporal_embedding",
            "target_future_temporal_embedding",
        ),
        (
            "online_future_projector.weight",
            "target_future_projector.weight",
        ),
        (
            "online_future_projector.bias",
            "target_future_projector.bias",
        ),
    )
    science = contract.science_contract()
    assert science["initialization"]["new_transformer_block_count"] == 0
    assert science["lifecycle"]["phase_b_authorized"] is False
    assert science["lifecycle"]["maximum_attempts"] == 1
    assert contract.EXECUTION_AUTHORITY["maximum_presentations"] == 16_000
    assert contract.EXECUTION_AUTHORITY["phase_b_authorized"] is False
    assert all(
        contract.DOWNSTREAM_DENIALS[field] is False
        for field in (
            "checkpoint_qualified",
            "g2_authorized",
            "navigation_authorized",
            "heldout_authorized",
            "sealed_authorized",
            "production_authorized",
            "promotion_authorized",
        )
    )


def test_update_zero_common_invariants_are_terminal_before_training() -> None:
    metric = _update0_metric()
    update0 = _update0()
    integrity = _integrity(0)
    gate = contract.evaluate_phase_a_update_zero(
        metric, update0, integrity
    )
    assert gate["passed"] is True
    assert gate["control"] == contract.CONTROL_CONTINUE

    broken = dict(integrity)
    broken["future_leakage_prohibition_passed"] = False
    failed = contract.evaluate_phase_a_update_zero(
        metric, update0, broken
    )
    assert failed["passed"] is False
    assert failed["control"] == contract.CONTROL_PHASE_A_UPDATE_ZERO_FAIL


def test_staged_update_100_400_and_terminal_pass_fixture() -> None:
    update0 = _update0()
    update100 = _update100()
    gate100 = contract.evaluate_phase_a_continuation(
        100, update100, update0, _integrity(100)
    )
    assert gate100["passed"] is True
    assert gate100["control"] == "CONTINUE_INFORMATIONAL"

    update400 = _update400()
    gate400 = contract.evaluate_phase_a_continuation(
        400, update400, update0, _integrity(400), update100
    )
    assert gate400["passed"] is True
    assert gate400["control"] == "CONTINUE_INFORMATIONAL"

    terminal = contract.evaluate_phase_a(
        _terminal(), update0, _integrity(1_000), update400
    )
    assert terminal["passed"] is True
    assert terminal["control"] == (
        "PASS_MASKED_PAIR_TUBELET_PROXY_SEPARATE_REQUALIFICATION_ONLY"
    )


def test_staged_comparators_fail_at_the_registered_strict_boundaries() -> None:
    update0 = _update0()
    update100 = _update100()
    update100["normalized_projected_future_effective_rank"] = (
        17.426651000976562
    )
    failed100 = contract.evaluate_phase_a_continuation(
        100, update100, update0, _integrity(100)
    )
    assert failed100["passed"] is False
    assert failed100["control"] == (
        "FAIL_PHASE_A_UPDATE_100_CONTINUATION_GATE_TERMINAL"
    )

    good100 = _update100()
    update400 = _update400()
    update400["factorized_retrieval"][
        "same_action_correct_energy"
    ] = 0.99
    update400["factorized_retrieval"][
        "same_action_correct_to_deranged_ratio"
    ] = 0.99
    failed400 = contract.evaluate_phase_a_continuation(
        400, update400, update0, _integrity(400), good100
    )
    assert failed400["passed"] is False
    assert failed400["control"] == (
        "FAIL_PHASE_A_UPDATE_400_CONTINUATION_GATE_TERMINAL"
    )

    terminal = _terminal()
    terminal["normalized_projected_future_effective_rank"] = 47.999
    failed_terminal = contract.evaluate_phase_a(
        terminal, update0, _integrity(1_000), _update400()
    )
    assert failed_terminal["passed"] is False
    assert failed_terminal["control"] == "FAIL_PHASE_A_TERMINAL_NO_RETRY"


def test_update_zero_action_nll_uses_frozen_v10r_integrity_tolerance() -> None:
    reference = math.log(9.0)
    tolerance = 8.0 * (2.0 ** -23)
    assert contract.UPDATE_ZERO_ACTION_NLL_ABSOLUTE_TOLERANCE == tolerance

    nearest_inside = math.nextafter(reference + tolerance, reference)
    inside = _update0()
    _set_action_nll(inside, nearest_inside)
    passed = contract.evaluate_phase_a_continuation(
        100, _update100(), inside, _integrity(100)
    )
    assert passed["passed"] is True

    nearest_outside = math.nextafter(reference + tolerance, math.inf)
    outside = _update0()
    _set_action_nll(outside, nearest_outside)
    failed = contract.evaluate_phase_a_continuation(
        100, _update100(), outside, _integrity(100)
    )
    assert failed["passed"] is False
    assert not failed["conjuncts"][
        "update_zero_action_symmetry_and_chance_exact"
    ]


def test_gate_rejects_integrity_or_hidden_field_changes() -> None:
    integrity = _integrity(100)
    integrity["future_leakage_prohibition_passed"] = False
    result = contract.evaluate_phase_a_continuation(
        100, _update100(), _update0(), integrity
    )
    assert result["passed"] is False
    assert not result["conjuncts"][
        "future_leakage_target_nonavuity_and_autograd_isolation"
    ]

    malformed = _update100()
    malformed["unregistered_metric"] = 1.0
    with pytest.raises(ValueError, match="metric fields changed"):
        contract.evaluate_phase_a_continuation(
            100, malformed, _update0(), _integrity(100)
        )


def test_authorization_template_is_one_phase_and_content_bound() -> None:
    runtime = contract.runtime_authorization_template()
    assert runtime["raw"]["phase_b_grant"] is None
    assert runtime["raw"]["role_policy"]["model_facing_roles"] == [
        "train",
        "checkpoint_selection",
    ]
    assert contract.validate_runtime_inputs(runtime) == runtime
    changed = deepcopy(runtime)
    changed["raw"]["phase_b_grant"] = {"authorized": True}
    with pytest.raises(PermissionError, match="runtime input authority"):
        contract.validate_runtime_inputs(changed)
