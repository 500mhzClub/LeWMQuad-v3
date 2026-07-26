from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load(
    "lewm/benchmarks/"
    "go2_rgb_masked_current_next_pair_tubelet_jepa_"
    "v13_learning_curve_continuation.py",
    "_test_pair_tubelet_v13_learning_curve_contract",
)
v12 = _load(
    "lewm/benchmarks/"
    "go2_rgb_masked_current_next_pair_tubelet_jepa_v12_gate_timing.py",
    "_test_pair_tubelet_v13_frozen_v12",
)
fixtures = _load(
    "lewm/tests/"
    "test_go2_rgb_masked_current_next_pair_tubelet_jepa_v11_contract.py",
    "_test_pair_tubelet_v13_v11_fixtures",
)


def _normalized_v12_evidence() -> dict[str, Any]:
    common = {
        name: True for name in contract.V13_UPDATE_100_STRUCTURAL_CONJUNCTS
    }
    # The two mature shuffled-ratio conditions failed in V12 and are
    # deliberately measured but nonterminal at V13 update 100.
    common.update({
        "true_at_most_point90_shuffled_next": False,
        "true_at_most_point95_shuffled_current": False,
    })
    update0_true = 0.005849023349583149
    update0_mean = 0.0036162424366921186
    return {
        "common": common,
        "values": {
            "normalized_projected_future_effective_rank": 99.0425033569336,
        },
        "update0_values": {
            "normalized_projected_future_effective_rank": 42.52288818359375,
            "true_pair_mse": update0_true,
            "mean_target_mse": update0_mean,
        },
        "retrieval": {
            "non_hold_correct_to_current_ratio": 0.9996770620346069,
            "action_retrieval_nll": 2.1962249279022217,
            "action_retrieval_macro_balanced_accuracy": 0.11819767036339542,
            "target_retrieval_nll": 1.0360678434371948,
            "same_action_target_retrieval_nll": 1.0360008478164673,
            "same_action_two_target_nll": 0.6736655235290527,
            "same_action_strict_win_rate": 0.5951417004048583,
            "same_action_correct_to_deranged_ratio": 0.9759560227394104,
        },
        "update0_retrieval": {
            "action_retrieval_nll": 2.1972239017486572,
            "action_retrieval_macro_balanced_accuracy": 0.1111111111111111,
            "target_retrieval_nll": 1.0483659505844116,
            "same_action_target_retrieval_nll": 1.0483002662658691,
            "same_action_two_target_nll": 0.6921065449714661,
            "same_action_strict_win_rate": 0.52834008097166,
            "same_action_correct_to_deranged_ratio": 0.9924365282058716,
        },
        "ratios": {
            "true_to_mean_target": 1.3812083279836236,
            "true_to_shuffled_next": 0.9720841018307778,
            "true_to_shuffled_current": 0.9729033465204676,
        },
        "per_family": {},
    }


def _evaluate_normalized(
    monkeypatch: pytest.MonkeyPatch,
    normalized: dict[str, Any],
) -> dict[str, Any]:
    monkeypatch.setattr(
        contract,
        "_normalize_v11_phase_a_inputs",
        lambda metrics, update0, integrity: normalized,
    )
    return contract.evaluate_phase_a_continuation(100, {}, {}, {})


def test_v13_frozen_bindings_identity_and_source_closure_are_exact() -> None:
    assert contract.IMPLEMENTATION_AUTHOR == "/root/plan_full_stack"
    assert contract.PREREGISTRATION_COMMIT == (
        "6b47a1dd1361c7c9fc059fd0b61ec6278bc38efc"
    )
    assert contract.PREREGISTRATION_FILE_SHA256 == (
        "6a4c81010444db766c831d738e1857b63753b1da72d4805ad0340ff9c085f7db"
    )
    assert contract.PREREGISTRATION_CONTENT_SHA256 == (
        "ece8762df23610bb820c92dd2c17a98f90847aa3810dee436e64fdc12643582b"
    )
    assert contract.PREREGISTRATION_BYTE_COUNT == 8_344

    assert contract.PRIOR_TERMINAL_AUDIT_COMMIT == (
        "19ddee22a04772538870004c3179f5f8ea19f7d3"
    )
    audit = (ROOT / contract.PRIOR_TERMINAL_AUDIT_RELATIVE_PATH).read_bytes()
    assert len(audit) == 40_794
    assert hashlib.sha256(audit).hexdigest() == (
        "cfa2f75f932e4afe95071838653e8a7ce6b1c575e070a665206985f95b525b99"
    )
    assert contract.PRIOR_TERMINAL_AUDIT_CONTENT_SHA256 == (
        "64ffef189a08493ab4944627a5578b56dfcbcfe961d29c6d0ab21c6a5c5fe356"
    )

    assert contract.OUTPUT_ROOT_RELATIVE_PATH == (
        ".generated/go2_shared_observable_camera_ray_jepa_v5/"
        "rgb_masked_current_next_pair_tubelet_jepa_probe_v13"
    )
    assert contract.V12_OUTPUT_ROOT_RELATIVE_PATH in (
        contract.PROHIBITED_RUNTIME_OUTPUT_ROOTS
    )
    assert set(contract.ADDITIVE_SOURCE_PATHS) == {
        contract.CONTRACT_RELATIVE_PATH,
        contract.RUNNER_RELATIVE_PATH,
        contract.LAUNCHER_RELATIVE_PATH,
        contract.CONTRACT_TEST_RELATIVE_PATH,
        contract.RUNNER_TEST_RELATIVE_PATH,
        contract.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
        contract.SOURCE_CLOSURE_TEST_RELATIVE_PATH,
    }
    assert set(v12.SOURCE_PATHS).issubset(contract.SOURCE_PATHS)


def test_science_identity_changes_only_v13_gate_and_lifecycle() -> None:
    frozen = contract.frozen_v12_science_contract()
    assert frozen == v12.science_contract()
    expected = deepcopy(frozen)
    expected["schema"] = f"{contract.SCHEMA_PREFIX}_science_contract_v1"
    expected["scientific_question"] = (
        "does_the_unchanged_masked_current_to_next_pair_tubelet_jepa_"
        "convert_verified_update100_directional_learning_into_the_complete_"
        "absolute_mechanism_gate_by_update400"
    )
    expected["gates"]["update_100"] = deepcopy(
        contract.V13_UPDATE_100_GATE_CONTRACT
    )
    expected["gates"]["update_400"].update({
        "complete_frozen_v12_evaluator": True,
        "restores_every_original_update100_absolute_conjunct": True,
    })
    expected["lifecycle"].update({
        "output_root": contract.OUTPUT_ROOT_RELATIVE_PATH,
        "initialize_fresh_from_original_bound_n320_encoder": True,
        "predecessor_checkpoint_trace_or_runtime_output_reuse_authorized":
            False,
        "update100_or_update400_failure_permanently_closes_mechanism": True,
        "v14_timing_successor_authorized": False,
    })
    assert contract.science_contract() == expected

    assert contract.phase_a_model_config() == v12.phase_a_model_config()
    assert contract.build_schedule_identity("phase_a") == (
        v12.build_schedule_identity("phase_a")
    )
    assert contract.PHASE_A_PASS_THRESHOLDS == v12.PHASE_A_PASS_THRESHOLDS
    assert contract.PHASE_A_UPDATE_100_THRESHOLDS == (
        v12.PHASE_A_UPDATE_100_THRESHOLDS
    )
    assert contract.PHASE_A_UPDATE_400_THRESHOLDS == (
        v12.PHASE_A_UPDATE_400_THRESHOLDS
    )
    assert contract.EXECUTION_AUTHORITY == v12.EXECUTION_AUTHORITY
    assert contract.SUCCESSOR_BOUNDARY["v14_timing_successor_authorized"] is False


def test_update_zero_is_exact_v12() -> None:
    metric = fixtures._update0_metric()
    update0 = fixtures._update0()
    integrity = fixtures._integrity(0)
    assert contract.evaluate_phase_a_update_zero(
        metric, update0, integrity
    ) == v12.evaluate_phase_a_update_zero(metric, update0, integrity)


def test_exact_v12_directional_evidence_passes_v13_update100(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _evaluate_normalized(monkeypatch, _normalized_v12_evidence())
    assert result["passed"] is True
    assert result["control"] == contract.CONTROL_CONTINUE
    expected = {
        *contract.V13_UPDATE_100_STRUCTURAL_CONJUNCTS,
        "non_hold_correct_to_fixed_current_strictly_below_one",
        *contract.V13_UPDATE_100_DIRECTIONAL_CONJUNCTS,
    }
    assert set(result["conjuncts"]) == expected
    assert result["ratios"]["true_to_shuffled_next"] > 0.90
    assert result["ratios"]["true_to_shuffled_current"] > 0.95
    assert result["thresholds"] == contract.V13_UPDATE_100_GATE_CONTRACT


@pytest.mark.parametrize(
    ("conjunct", "group", "field"),
    (
        (
            "normalized_projected_future_effective_rank_strictly_above_update0",
            "values",
            "normalized_projected_future_effective_rank",
        ),
        (
            "action_retrieval_nll_strictly_below_update0",
            "retrieval",
            "action_retrieval_nll",
        ),
        (
            "action_retrieval_macro_balanced_accuracy_strictly_above_update0",
            "retrieval",
            "action_retrieval_macro_balanced_accuracy",
        ),
        (
            "target_retrieval_nll_strictly_below_update0",
            "retrieval",
            "target_retrieval_nll",
        ),
        (
            "same_action_target_retrieval_nll_strictly_below_update0",
            "retrieval",
            "same_action_target_retrieval_nll",
        ),
        (
            "same_action_two_target_nll_strictly_below_update0",
            "retrieval",
            "same_action_two_target_nll",
        ),
        (
            "same_action_strict_win_rate_strictly_above_update0",
            "retrieval",
            "same_action_strict_win_rate",
        ),
        (
            "same_action_correct_to_deranged_ratio_strictly_below_update0",
            "retrieval",
            "same_action_correct_to_deranged_ratio",
        ),
        (
            "true_to_mean_target_ratio_strictly_below_update0",
            "ratios",
            "true_to_mean_target",
        ),
    ),
)
def test_each_directional_comparator_is_strict_without_epsilon(
    monkeypatch: pytest.MonkeyPatch,
    conjunct: str,
    group: str,
    field: str,
) -> None:
    normalized = _normalized_v12_evidence()
    if group == "values":
        baseline = normalized["update0_values"][field]
    elif group == "retrieval":
        baseline = normalized["update0_retrieval"][field]
    else:
        baseline = (
            normalized["update0_values"]["true_pair_mse"]
            / normalized["update0_values"]["mean_target_mse"]
        )
    normalized[group][field] = baseline
    result = _evaluate_normalized(monkeypatch, normalized)
    assert result["passed"] is False
    assert result["control"] == contract.CONTROL_PHASE_A_UPDATE_100_FAIL
    assert result["conjuncts"][conjunct] is False


def test_structural_and_single_absolute_update100_checks_remain_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structural = _normalized_v12_evidence()
    name = contract.V13_UPDATE_100_STRUCTURAL_CONJUNCTS[0]
    structural["common"][name] = False
    result = _evaluate_normalized(monkeypatch, structural)
    assert result["passed"] is False
    assert result["conjuncts"][name] is False

    absolute = _normalized_v12_evidence()
    absolute["retrieval"]["non_hold_correct_to_current_ratio"] = 1.0
    result = _evaluate_normalized(monkeypatch, absolute)
    assert result["passed"] is False
    assert result["conjuncts"][
        "non_hold_correct_to_fixed_current_strictly_below_one"
    ] is False


def test_update400_and_terminal_delegate_exactly_to_frozen_v12() -> None:
    update0 = fixtures._update0()
    update100 = fixtures._update100()
    update400 = fixtures._update400()
    integrity400 = fixtures._integrity(400)
    expected400 = v12.evaluate_phase_a_continuation(
        400, update400, update0, integrity400, update100
    )
    actual400 = contract.evaluate_phase_a_continuation(
        400, update400, update0, integrity400, update100
    )
    assert actual400 == expected400

    terminal = fixtures._terminal()
    expected_terminal = v12.evaluate_phase_a(
        terminal, update0, fixtures._integrity(1_000), update400
    )
    actual_terminal = contract.evaluate_phase_a(
        terminal, update0, fixtures._integrity(1_000), update400
    )
    assert actual_terminal == expected_terminal


def test_authorization_binds_v13_science_and_independent_roles() -> None:
    review_binding = {
        "path": contract.REVIEW_RELATIVE_PATH,
        "file_sha256": "a" * 64,
        "content_sha256": "b" * 64,
        "byte_count": 123,
    }
    reviewer = "/root/v13_independent_reviewer"
    core = {
        "schema": contract.AUTHORIZATION_SCHEMA,
        "status": contract.AUTHORIZATION_STATUS,
        "authorizer": "/root/v13_machine_authorizer",
        "independent_source_review": dict(review_binding),
        "preregistration": contract.preregistration_binding(),
        "runtime_inputs": contract.runtime_authorization_template(),
        "experiment": contract.science_contract(),
        "authority": dict(contract.EXECUTION_AUTHORITY),
    }
    authorization = contract.with_content_sha256(core)
    assert contract.validate_authorization(
        authorization,
        review_binding=review_binding,
        reviewer=reviewer,
    ) == authorization

    changed = deepcopy(core)
    changed["experiment"]["lifecycle"]["v14_timing_successor_authorized"] = True
    with pytest.raises(PermissionError, match="authorization changed"):
        contract.validate_authorization(
            contract.with_content_sha256(changed),
            review_binding=review_binding,
            reviewer=reviewer,
        )
