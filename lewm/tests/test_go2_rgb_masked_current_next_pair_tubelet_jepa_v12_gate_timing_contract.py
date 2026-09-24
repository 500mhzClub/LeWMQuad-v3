from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys

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
    "go2_rgb_masked_current_next_pair_tubelet_jepa_v12_gate_timing.py",
    "_test_masked_pair_tubelet_v12_gate_timing_contract",
)
v11 = _load(
    "lewm/benchmarks/go2_rgb_masked_current_next_pair_tubelet_jepa_v11.py",
    "_test_masked_pair_tubelet_v12_frozen_v11",
)
fixtures = _load(
    "lewm/tests/"
    "test_go2_rgb_masked_current_next_pair_tubelet_jepa_v11_contract.py",
    "_test_masked_pair_tubelet_v12_v11_fixtures",
)


def test_v12_identity_documents_and_complete_source_closure_are_exact() -> None:
    assert contract.IMPLEMENTATION_AUTHOR == "/root/plan_full_stack"
    assert contract.PREREGISTRATION_COMMIT == (
        "2c4a70fa360cbf26b300359835d105a9f6b213a1"
    )
    preregistration = (ROOT / contract.PREREGISTRATION_RELATIVE_PATH).read_bytes()
    assert len(preregistration) == 19_829
    assert hashlib.sha256(preregistration).hexdigest() == (
        "0f13ab590a8446a2be609d900d7e74dfe562ec39a714befa29ec328af17cc912"
    )
    assert contract.PREREGISTRATION_CONTENT_SHA256 == (
        "62f3d3219d57ae05fc1dcb4a929999acc5aab8ff2c0c8bf115bf3ed1473426fa"
    )

    assert contract.PRIOR_TERMINAL_AUDIT_COMMIT == (
        "4d3e967f1d30bc3843626a9b5aaecd79e6f1dca0"
    )
    audit = (ROOT / contract.PRIOR_TERMINAL_AUDIT_RELATIVE_PATH).read_bytes()
    assert len(audit) == 7_876
    assert hashlib.sha256(audit).hexdigest() == (
        "89ac1155e7108118133d6eb0648437e3a337f03e31c6c93e6ca63cc590f27044"
    )
    assert contract.PRIOR_TERMINAL_AUDIT_CONTENT_SHA256 == (
        "9641274f58e84b4a3c3603f7cf19714e006ec27d062d57a0f24f0bb38677aec9"
    )

    assert contract.OUTPUT_ROOT_RELATIVE_PATH == (
        ".generated/go2_shared_observable_camera_ray_jepa_v5/"
        "rgb_masked_current_next_pair_tubelet_jepa_probe_v12"
    )
    assert contract.V11_OUTPUT_ROOT_RELATIVE_PATH in (
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
    assert set(contract.ADDITIVE_SOURCE_PATHS).issubset(contract.SOURCE_PATHS)
    assert set(v11.SOURCE_PATHS).issubset(contract.SOURCE_PATHS)


def test_current_source_bindings_accept_exact_v12_governing_documents() -> None:
    bindings = contract.current_source_bindings(ROOT)
    assert bindings[contract.PREREGISTRATION_RELATIVE_PATH] == (
        contract.PREREGISTRATION_FILE_SHA256
    )
    assert bindings[contract.PRIOR_TERMINAL_AUDIT_RELATIVE_PATH] == (
        contract.PRIOR_TERMINAL_AUDIT_FILE_SHA256
    )
    assert contract.is_sha256(
        bindings[contract.SOURCE_MANIFEST_RELATIVE_PATH]
    )
    assert contract.V11_OUTPUT_ROOT_RELATIVE_PATH not in bindings


def test_science_contract_is_exactly_v11_plus_identity_and_gate_timing() -> None:
    frozen = contract.frozen_v11_science_contract()
    assert frozen == v11.science_contract()
    expected = deepcopy(frozen)
    expected["schema"] = f"{contract.SCHEMA_PREFIX}_science_contract_v1"
    expected["scientific_question"] = (
        "can_the_unchanged_v11_masked_current_to_next_pair_tubelet_jepa_"
        "learn_action_specific_future_structure_by_the_first_trained_"
        "observation_when_update_zero_is_integrity_only"
    )
    expected["gates"]["update_0"].update({
        "removed_learned_selectivity_terminal_conjuncts": [
            "true_at_most_point90_shuffled_next",
            "true_at_most_point95_shuffled_current",
        ],
        "learned_selectivity_ratios_still_measured_and_recorded": True,
        "first_learned_selectivity_enforcement_update": 100,
    })
    expected["lifecycle"]["output_root"] = contract.OUTPUT_ROOT_RELATIVE_PATH
    assert contract.science_contract() == expected

    assert contract.phase_a_model_config() == v11.phase_a_model_config()
    assert contract.build_schedule_identity("phase_a") == (
        v11.build_schedule_identity("phase_a")
    )
    assert contract.PHASE_A_PASS_THRESHOLDS == v11.PHASE_A_PASS_THRESHOLDS
    assert contract.PHASE_A_UPDATE_100_THRESHOLDS == (
        v11.PHASE_A_UPDATE_100_THRESHOLDS
    )
    assert contract.PHASE_A_UPDATE_400_THRESHOLDS == (
        v11.PHASE_A_UPDATE_400_THRESHOLDS
    )
    assert contract.EXECUTION_AUTHORITY == v11.EXECUTION_AUTHORITY
    assert contract.GATE_TIMING_DELTA[
        "update_zero_removed_terminal_conjuncts"
    ] == list(contract.MOVED_LEARNED_SELECTIVITY_CONJUNCTS)


def test_update_zero_removes_only_the_paired_learned_selectivity_gates() -> None:
    metric = fixtures._update0_metric()
    metric["true_pair_mse"] = 0.94
    metric["shuffled_next_mse"] = 1.0
    metric["shuffled_current_mse"] = 1.0
    update0 = fixtures._update0()
    integrity = fixtures._integrity(0)

    predecessor = v11.evaluate_phase_a_update_zero(metric, update0, integrity)
    successor = contract.evaluate_phase_a_update_zero(
        metric, update0, integrity
    )
    moved = set(contract.MOVED_LEARNED_SELECTIVITY_CONJUNCTS)
    assert predecessor["passed"] is False
    assert successor["passed"] is True
    assert set(successor["conjuncts"]) == set(predecessor["conjuncts"]) - moved
    assert moved.isdisjoint(successor["conjuncts"])
    assert successor["ratios"]["true_to_shuffled_next"] == 0.94
    assert successor["ratios"]["true_to_shuffled_current"] == 0.94

    broken = dict(integrity)
    broken["future_leakage_prohibition_passed"] = False
    failed = contract.evaluate_phase_a_update_zero(metric, update0, broken)
    assert failed["passed"] is False
    assert failed["control"] == contract.CONTROL_PHASE_A_UPDATE_ZERO_FAIL


def test_both_moved_ratios_are_enforced_unchanged_from_update_100() -> None:
    update0 = fixtures._update0()
    integrity = fixtures._integrity(100)

    passing = fixtures._update100()
    assert contract.evaluate_phase_a_continuation(
        100, passing, update0, integrity
    ) == v11.evaluate_phase_a_continuation(100, passing, update0, integrity)

    next_failure = fixtures._update100()
    next_failure["true_pair_mse"] = 0.91
    failed_next = contract.evaluate_phase_a_continuation(
        100, next_failure, update0, integrity
    )
    assert failed_next["passed"] is False
    assert failed_next["conjuncts"][
        "true_at_most_point90_shuffled_next"
    ] is False

    current_failure = fixtures._update100()
    current_failure["true_pair_mse"] = 0.80
    current_failure["shuffled_next_mse"] = 1.0
    current_failure["shuffled_current_mse"] = 0.84
    failed_current = contract.evaluate_phase_a_continuation(
        100, current_failure, update0, integrity
    )
    assert failed_current["passed"] is False
    assert failed_current["conjuncts"][
        "true_at_most_point95_shuffled_current"
    ] is False


def test_v12_authorization_accepts_only_exact_identity_and_separation() -> None:
    review_binding = {
        "path": contract.REVIEW_RELATIVE_PATH,
        "file_sha256": "a" * 64,
        "content_sha256": "b" * 64,
        "byte_count": 123,
    }
    reviewer = "/root/v12_independent_reviewer"
    core = {
        "schema": contract.AUTHORIZATION_SCHEMA,
        "status": contract.AUTHORIZATION_STATUS,
        "authorizer": "/root/v12_machine_authorizer",
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

    for forbidden_authorizer in (contract.IMPLEMENTATION_AUTHOR, reviewer):
        changed = dict(core)
        changed["authorizer"] = forbidden_authorizer
        with pytest.raises(PermissionError, match="authorization changed"):
            contract.validate_authorization(
                contract.with_content_sha256(changed),
                review_binding=review_binding,
                reviewer=reviewer,
            )

    changed = deepcopy(core)
    changed["experiment"]["gates"]["update_0"].pop(
        "first_learned_selectivity_enforcement_update"
    )
    with pytest.raises(PermissionError, match="authorization changed"):
        contract.validate_authorization(
            contract.with_content_sha256(changed),
            review_binding=review_binding,
            reviewer=reviewer,
        )
