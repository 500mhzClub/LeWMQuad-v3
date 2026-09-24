"""Source-only V12 gate-timing successor to the frozen V11 RGB JEPA.

V12 executes the tracked V11 contract into this module so inherited helpers
resolve the V12 operational identity without mutating V11.  The sole
scientific delta removes the paired learned true-versus-shuffled selectivity
checks from the update-zero terminal gate.  Both ratios remain measured and
recorded at update zero, and the unchanged V11 continuation gates enforce
both checks from update 100 onward.  Importing this module reads tracked
Python source only; it opens no generated input, runtime output, checkpoint,
tensor, trace, RGB, held-out, or sealed material.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


_V11_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/"
    "go2_rgb_masked_current_next_pair_tubelet_jepa_v11.py"
)
_V11_CONTRACT_PATH = (
    Path(__file__).resolve().parents[2] / _V11_CONTRACT_RELATIVE_PATH
)
_V11_SOURCE = _V11_CONTRACT_PATH.read_bytes()
exec(compile(_V11_SOURCE, str(_V11_CONTRACT_PATH), "exec"), globals())
del _V11_SOURCE

_FROZEN_V11_SOURCE_PATHS = tuple(SOURCE_PATHS)
_FROZEN_V11_SOURCE_REVIEW_ADDITIONAL_PATHS = tuple(
    SOURCE_REVIEW_ADDITIONAL_PATHS
)
_FROZEN_V11_SCIENCE_CONTRACT = deepcopy(science_contract())
_FROZEN_V11_SCIENTIFIC_REVIEW_CHECKS = dict(SCIENTIFIC_REVIEW_CHECKS)

V11_CONTRACT_RELATIVE_PATH = _V11_CONTRACT_RELATIVE_PATH
V11_RUNNER_RELATIVE_PATH = RUNNER_RELATIVE_PATH
V11_LAUNCHER_RELATIVE_PATH = LAUNCHER_RELATIVE_PATH
V11_OUTPUT_ROOT_RELATIVE_PATH = OUTPUT_ROOT_RELATIVE_PATH


# V12 operational identity and complete source closure.
IMPLEMENTATION_AUTHOR = "/root/plan_full_stack"
SCHEMA_PREFIX = (
    "lewm_go2_rgb_masked_current_next_pair_tubelet_jepa_v12_gate_timing"
)
CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/"
    "go2_rgb_masked_current_next_pair_tubelet_jepa_v12_gate_timing.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/"
    "run_go2_rgb_masked_current_next_pair_tubelet_jepa_v12_gate_timing.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/"
    "launch_go2_rgb_masked_current_next_pair_tubelet_jepa_v12_gate_timing.py"
)
CONTRACT_TEST_RELATIVE_PATH = (
    "lewm/tests/"
    "test_go2_rgb_masked_current_next_pair_tubelet_jepa_v12_gate_timing_"
    "contract.py"
)
RUNNER_TEST_RELATIVE_PATH = (
    "lewm/tests/"
    "test_go2_rgb_masked_current_next_pair_tubelet_jepa_v12_gate_timing_"
    "runner.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/"
    "check_go2_rgb_masked_current_next_pair_tubelet_jepa_v12_gate_timing_"
    "source_closure.py"
)
SOURCE_CLOSURE_TEST_RELATIVE_PATH = (
    "lewm/tests/"
    "test_go2_rgb_masked_current_next_pair_tubelet_jepa_v12_gate_timing_"
    "source_closure.py"
)
FROZEN_V11_CONTRACT_RELATIVE_PATH = V11_CONTRACT_RELATIVE_PATH
FROZEN_V11_RUNNER_RELATIVE_PATH = V11_RUNNER_RELATIVE_PATH
FROZEN_V11_LAUNCHER_RELATIVE_PATH = V11_LAUNCHER_RELATIVE_PATH
TEST_RELATIVE_PATH = CONTRACT_TEST_RELATIVE_PATH

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_masked_current_next_pair_tubelet_jepa_v12_gate_"
    "timing_preregistration_2026-07-26.json"
)
PREREGISTRATION_COMMIT = "2c4a70fa360cbf26b300359835d105a9f6b213a1"
PREREGISTRATION_FILE_SHA256 = (
    "0f13ab590a8446a2be609d900d7e74dfe562ec39a714befa29ec328af17cc912"
)
PREREGISTRATION_CONTENT_SHA256 = (
    "62f3d3219d57ae05fc1dcb4a929999acc5aab8ff2c0c8bf115bf3ed1473426fa"
)
PREREGISTRATION_BYTE_COUNT = 19_829

PRIOR_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_masked_current_next_pair_tubelet_jepa_v11_"
    "terminal_audit_2026-07-26.json"
)
PRIOR_TERMINAL_AUDIT_COMMIT = "4d3e967f1d30bc3843626a9b5aaecd79e6f1dca0"
PRIOR_TERMINAL_AUDIT_FILE_SHA256 = (
    "89ac1155e7108118133d6eb0648437e3a337f03e31c6c93e6ca63cc590f27044"
)
PRIOR_TERMINAL_AUDIT_CONTENT_SHA256 = (
    "9641274f58e84b4a3c3603f7cf19714e006ec27d062d57a0f24f0bb38677aec9"
)
PRIOR_TERMINAL_AUDIT_BYTE_COUNT = 7_876
V11_TERMINAL_AUDIT_RELATIVE_PATH = PRIOR_TERMINAL_AUDIT_RELATIVE_PATH

SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_masked_current_next_pair_tubelet_jepa_v12_gate_"
    "timing_source_manifest_2026-07-26.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_masked_current_next_pair_tubelet_jepa_v12_gate_"
    "timing_source_review_2026-07-26.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_masked_current_next_pair_tubelet_jepa_v12_gate_"
    "timing_execution_authorization_2026-07-26.json"
)

ADDITIVE_SOURCE_PATHS = tuple(sorted((
    CONTRACT_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    CONTRACT_TEST_RELATIVE_PATH,
    RUNNER_TEST_RELATIVE_PATH,
    SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    SOURCE_CLOSURE_TEST_RELATIVE_PATH,
)))
REUSED_SOURCE_PATHS = tuple(sorted(dict.fromkeys((
    V11_CONTRACT_RELATIVE_PATH,
    V11_RUNNER_RELATIVE_PATH,
    V11_LAUNCHER_RELATIVE_PATH,
    *_FROZEN_V11_SOURCE_PATHS,
))))
SOURCE_PATHS = tuple(sorted(dict.fromkeys((
    *ADDITIVE_SOURCE_PATHS,
    *REUSED_SOURCE_PATHS,
))))
SOURCE_MANIFEST_ENTRYPOINTS = (LAUNCHER_RELATIVE_PATH, RUNNER_RELATIVE_PATH)
SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES = SOURCE_PATHS
SOURCE_REVIEW_ADDITIONAL_PATHS = tuple(dict.fromkeys((
    SOURCE_MANIFEST_RELATIVE_PATH,
    PREREGISTRATION_RELATIVE_PATH,
    PRIOR_TERMINAL_AUDIT_RELATIVE_PATH,
    *_FROZEN_V11_SOURCE_REVIEW_ADDITIONAL_PATHS,
)))

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "rgb_masked_current_next_pair_tubelet_jepa_probe_v12"
)
PROHIBITED_RUNTIME_OUTPUT_ROOTS = (V11_OUTPUT_ROOT_RELATIVE_PATH,)


# V12 receipt, review, and authorization schemas.
SOURCE_MANIFEST_SCHEMA = f"{SCHEMA_PREFIX}_source_manifest_v1"
REVIEW_SCHEMA = f"{SCHEMA_PREFIX}_source_review_v1"
AUTHORIZATION_SCHEMA = f"{SCHEMA_PREFIX}_execution_authorization_v1"
RESERVATION_SCHEMA = f"{SCHEMA_PREFIX}_reservation_v1"
PHASE_A_METRICS_SCHEMA = f"{SCHEMA_PREFIX}_phase_a_metrics_v1"
PHASE_A_ARTIFACT_SCHEMA = f"{SCHEMA_PREFIX}_phase_a_artifact_v1"
ACCESS_SCHEMA = f"{SCHEMA_PREFIX}_access_v1"
RESULT_SCHEMA = f"{SCHEMA_PREFIX}_result_v1"
COMPLETION_SCHEMA = f"{SCHEMA_PREFIX}_completion_v1"
FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v1"

AUTHORIZATION_STATUS = "AUTHORIZED_ONE_EXACT_V12_GATE_TIMING_PROBE"


# The sole scientific delta.  These exact V11 conjuncts remain present in
# normalized evidence and in every inherited continuation/terminal gate.
MOVED_LEARNED_SELECTIVITY_CONJUNCTS = (
    "true_at_most_point90_shuffled_next",
    "true_at_most_point95_shuffled_current",
)
GATE_TIMING_DELTA = {
    "science_delta_count": 1,
    "delta": "gate_timing_only",
    "update_zero_removed_terminal_conjuncts": list(
        MOVED_LEARNED_SELECTIVITY_CONJUNCTS
    ),
    "ratios_measured_and_recorded_at_update_zero": True,
    "first_enforcement_update": 100,
    "first_enforcement_presentations": 1_600,
    "true_to_shuffled_next_maximum": 0.90,
    "true_to_shuffled_current_maximum": 0.95,
    "update_100_and_later_other_gates_unchanged": True,
    "model_data_loss_initialization_optimizer_seed_schedule_caps_unchanged":
        True,
}

SCIENTIFIC_REVIEW_CHECKS = {
    **_FROZEN_V11_SCIENTIFIC_REVIEW_CHECKS,
    "v11_closed_terminal_audit_bound_without_runtime_payload_access": True,
    "only_paired_learned_selectivity_gate_timing_changed": True,
    "both_ratios_recorded_at_update_zero_and_enforced_from_update100": True,
    "model_data_loss_initialization_optimizer_seed_schedule_caps_unchanged":
        True,
}


def frozen_v11_science_contract() -> dict[str, Any]:
    """Return the captured V11 science witness without V12 mutation."""

    return deepcopy(_FROZEN_V11_SCIENCE_CONTRACT)


def science_contract() -> dict[str, Any]:
    """Return V11 science with only V12 identity and gate timing changed."""

    value = frozen_v11_science_contract()
    value["schema"] = f"{SCHEMA_PREFIX}_science_contract_v1"
    value["scientific_question"] = (
        "can_the_unchanged_v11_masked_current_to_next_pair_tubelet_jepa_"
        "learn_action_specific_future_structure_by_the_first_trained_"
        "observation_when_update_zero_is_integrity_only"
    )
    value["gates"]["update_0"].update({
        "removed_learned_selectivity_terminal_conjuncts": list(
            MOVED_LEARNED_SELECTIVITY_CONJUNCTS
        ),
        "learned_selectivity_ratios_still_measured_and_recorded": True,
        "first_learned_selectivity_enforcement_update": 100,
    })
    value["lifecycle"]["output_root"] = OUTPUT_ROOT_RELATIVE_PATH
    return value


def evaluate_phase_a_update_zero(
    metrics: Mapping[str, Any],
    update0_metrics: Mapping[str, Any],
    observation_integrity: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply V11 normalization but gate only its integrity conjuncts."""

    normalized = _normalize_v11_phase_a_inputs(
        metrics, update0_metrics, observation_integrity
    )
    conjuncts = {
        name: passed
        for name, passed in normalized["common"].items()
        if name not in MOVED_LEARNED_SELECTIVITY_CONJUNCTS
    }
    passed = all(conjuncts.values())
    return {
        "update": 0,
        "passed": passed,
        "control": (
            CONTROL_CONTINUE
            if passed
            else CONTROL_PHASE_A_UPDATE_ZERO_FAIL
        ),
        "conjuncts": conjuncts,
        "ratios": dict(normalized["ratios"]),
        "thresholds": {"common_invariants_only": True},
        "per_family": dict(normalized["per_family"]),
        "factorized_retrieval": dict(normalized["retrieval"]),
    }


__all__ = sorted(set(__all__) | {
    name for name in globals()
    if name.isupper() or name in {
        "evaluate_phase_a_update_zero",
        "frozen_v11_science_contract",
        "science_contract",
    }
})
