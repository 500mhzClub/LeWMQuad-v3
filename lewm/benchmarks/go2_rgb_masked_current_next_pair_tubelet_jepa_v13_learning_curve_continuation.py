"""Source-only final V13 learning-curve successor to frozen V12.

V13 executes the tracked V12 contract into this module.  Update zero remains
exactly V12.  At update 100, mature absolute performance checks are replaced
by the preregistered strict directional-learning gate.  Update 400 delegates
to the complete frozen V12 evaluator, restoring every original update-100
and update-400 conjunct without changing a threshold or comparator.  The
terminal evaluator is likewise inherited unchanged.

Import reads tracked Python source only and opens no generated input, runtime
output, checkpoint, tensor, trace, RGB, held-out, or sealed material.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


_V12_CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/"
    "go2_rgb_masked_current_next_pair_tubelet_jepa_v12_gate_timing.py"
)
_V12_CONTRACT_PATH = (
    Path(__file__).resolve().parents[2] / _V12_CONTRACT_RELATIVE_PATH
)
_V12_SOURCE = _V12_CONTRACT_PATH.read_bytes()
exec(compile(_V12_SOURCE, str(_V12_CONTRACT_PATH), "exec"), globals())
del _V12_SOURCE

_FROZEN_V12_SOURCE_PATHS = tuple(SOURCE_PATHS)
_FROZEN_V12_SOURCE_REVIEW_ADDITIONAL_PATHS = tuple(
    SOURCE_REVIEW_ADDITIONAL_PATHS
)
_FROZEN_V12_SCIENCE_CONTRACT = deepcopy(science_contract())
_FROZEN_V12_SCIENTIFIC_REVIEW_CHECKS = dict(SCIENTIFIC_REVIEW_CHECKS)
_FROZEN_V12_PROHIBITED_RUNTIME_OUTPUT_ROOTS = tuple(
    PROHIBITED_RUNTIME_OUTPUT_ROOTS
)
_FROZEN_V12_EVALUATE_PHASE_A_CONTINUATION = evaluate_phase_a_continuation

V12_CONTRACT_RELATIVE_PATH = _V12_CONTRACT_RELATIVE_PATH
V12_RUNNER_RELATIVE_PATH = RUNNER_RELATIVE_PATH
V12_LAUNCHER_RELATIVE_PATH = LAUNCHER_RELATIVE_PATH
V12_OUTPUT_ROOT_RELATIVE_PATH = OUTPUT_ROOT_RELATIVE_PATH


# V13 operational identity and complete source closure.
IMPLEMENTATION_AUTHOR = "/root/plan_full_stack"
SCHEMA_PREFIX = (
    "lewm_go2_rgb_masked_current_next_pair_tubelet_jepa_"
    "v13_learning_curve_continuation"
)
CONTRACT_RELATIVE_PATH = (
    "lewm/benchmarks/"
    "go2_rgb_masked_current_next_pair_tubelet_jepa_"
    "v13_learning_curve_continuation.py"
)
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_rgb_masked_current_next_pair_tubelet_jepa_"
    "v13_learning_curve_continuation.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_rgb_masked_current_next_pair_tubelet_jepa_"
    "v13_learning_curve_continuation.py"
)
CONTRACT_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_rgb_masked_current_next_pair_tubelet_jepa_"
    "v13_learning_curve_continuation_contract.py"
)
RUNNER_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_rgb_masked_current_next_pair_tubelet_jepa_"
    "v13_learning_curve_continuation_runner.py"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_rgb_masked_current_next_pair_tubelet_jepa_"
    "v13_learning_curve_continuation_source_closure.py"
)
SOURCE_CLOSURE_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_rgb_masked_current_next_pair_tubelet_jepa_"
    "v13_learning_curve_continuation_source_closure.py"
)
FROZEN_V12_CONTRACT_RELATIVE_PATH = V12_CONTRACT_RELATIVE_PATH
FROZEN_V12_RUNNER_RELATIVE_PATH = V12_RUNNER_RELATIVE_PATH
FROZEN_V12_LAUNCHER_RELATIVE_PATH = V12_LAUNCHER_RELATIVE_PATH
TEST_RELATIVE_PATH = CONTRACT_TEST_RELATIVE_PATH

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_masked_current_next_pair_tubelet_jepa_v13_"
    "learning_curve_continuation_preregistration_2026-07-26.json"
)
PREREGISTRATION_COMMIT = "6b47a1dd1361c7c9fc059fd0b61ec6278bc38efc"
PREREGISTRATION_FILE_SHA256 = (
    "6a4c81010444db766c831d738e1857b63753b1da72d4805ad0340ff9c085f7db"
)
PREREGISTRATION_CONTENT_SHA256 = (
    "ece8762df23610bb820c92dd2c17a98f90847aa3810dee436e64fdc12643582b"
)
PREREGISTRATION_BYTE_COUNT = 8_344

PRIOR_TERMINAL_AUDIT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_masked_current_next_pair_tubelet_jepa_v12_gate_"
    "timing_terminal_audit_2026-07-26.json"
)
PRIOR_TERMINAL_AUDIT_COMMIT = "19ddee22a04772538870004c3179f5f8ea19f7d3"
PRIOR_TERMINAL_AUDIT_FILE_SHA256 = (
    "cfa2f75f932e4afe95071838653e8a7ce6b1c575e070a665206985f95b525b99"
)
PRIOR_TERMINAL_AUDIT_CONTENT_SHA256 = (
    "64ffef189a08493ab4944627a5578b56dfcbcfe961d29c6d0ab21c6a5c5fe356"
)
PRIOR_TERMINAL_AUDIT_BYTE_COUNT = 40_794
V12_TERMINAL_AUDIT_RELATIVE_PATH = PRIOR_TERMINAL_AUDIT_RELATIVE_PATH

SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_masked_current_next_pair_tubelet_jepa_v13_"
    "learning_curve_continuation_source_manifest_2026-07-26.json"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_masked_current_next_pair_tubelet_jepa_v13_"
    "learning_curve_continuation_source_review_2026-07-26.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_masked_current_next_pair_tubelet_jepa_v13_"
    "learning_curve_continuation_execution_authorization_2026-07-26.json"
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
    V12_CONTRACT_RELATIVE_PATH,
    V12_RUNNER_RELATIVE_PATH,
    V12_LAUNCHER_RELATIVE_PATH,
    *_FROZEN_V12_SOURCE_PATHS,
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
    *_FROZEN_V12_SOURCE_REVIEW_ADDITIONAL_PATHS,
)))

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "rgb_masked_current_next_pair_tubelet_jepa_probe_v13"
)
PROHIBITED_RUNTIME_OUTPUT_ROOTS = tuple(dict.fromkeys((
    *_FROZEN_V12_PROHIBITED_RUNTIME_OUTPUT_ROOTS,
    V12_OUTPUT_ROOT_RELATIVE_PATH,
)))


# V13 receipt, review, and authorization schemas.
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
AUTHORIZATION_STATUS = (
    "AUTHORIZED_ONE_EXACT_V13_LEARNING_CURVE_CONTINUATION"
)


V13_UPDATE_100_STRUCTURAL_CONJUNCTS = (
    "control_populations_and_one_scene_per_family_exact",
    "ema_inventory_and_update_count_exact",
    "factorized_retrieval_health_exact",
    "finite_and_ema_gradient_free",
    "future_leakage_target_nonavuity_and_autograd_isolation",
    "normalized_population_and_all_nine_candidates_exact",
    "normalized_projected_future_spatial_diversity_at_least_quarter_update0",
    "normalized_projected_future_variance_at_least_quarter_update0",
    "observation_rng_and_model_state_preserved",
    "retrieval_references_and_mappings_immutable",
    "update_zero_action_symmetry_and_chance_exact",
    "update_zero_factorized_retrieval_health_exact",
)
V13_UPDATE_100_DIRECTIONAL_CONJUNCTS = (
    "normalized_projected_future_effective_rank_strictly_above_update0",
    "action_retrieval_nll_strictly_below_update0",
    "action_retrieval_macro_balanced_accuracy_strictly_above_update0",
    "target_retrieval_nll_strictly_below_update0",
    "same_action_target_retrieval_nll_strictly_below_update0",
    "same_action_two_target_nll_strictly_below_update0",
    "same_action_strict_win_rate_strictly_above_update0",
    "same_action_correct_to_deranged_ratio_strictly_below_update0",
    "true_to_mean_target_ratio_strictly_below_update0",
)
V13_UPDATE_100_MEASURED_NOT_TERMINAL = (
    "masked_future_jepa_loss",
    "true_to_shuffled_next",
    "true_to_shuffled_current",
    "mature_action_target_and_family_margin_absolute_thresholds",
)
V13_UPDATE_100_GATE_CONTRACT = {
    "structural_conjuncts": list(V13_UPDATE_100_STRUCTURAL_CONJUNCTS),
    "absolute_conjuncts": [
        "non_hold_correct_to_fixed_current_strictly_below_one"
    ],
    "strict_directional_conjuncts_without_epsilon": list(
        V13_UPDATE_100_DIRECTIONAL_CONJUNCTS
    ),
    "measured_but_not_terminal": list(V13_UPDATE_100_MEASURED_NOT_TERMINAL),
}
SUCCESSOR_BOUNDARY = {
    "update_100_or_update_400_failure_permanently_closes_mechanism": True,
    "v14_timing_successor_authorized": False,
    "v11_or_v12_checkpoint_trace_or_runtime_output_reuse_authorized": False,
}
SCIENTIFIC_REVIEW_CHECKS = {
    **_FROZEN_V12_SCIENTIFIC_REVIEW_CHECKS,
    "update_zero_bit_exact_to_v12": True,
    "update100_strict_directional_gate_exact_without_epsilon": True,
    "update400_delegates_to_complete_frozen_v12_evaluator": True,
    "terminal_evaluator_and_all_numeric_thresholds_unchanged": True,
    "fresh_original_n320_initialization_and_no_predecessor_checkpoint_reuse":
        True,
    "hard_stop_and_no_v14_timing_successor": True,
}


def frozen_v12_science_contract() -> dict[str, Any]:
    """Return the captured V12 science witness without V13 mutation."""

    return deepcopy(_FROZEN_V12_SCIENCE_CONTRACT)


def science_contract() -> dict[str, Any]:
    """Return V12 science with only V13 gate/lifecycle identity changed."""

    value = frozen_v12_science_contract()
    value["schema"] = f"{SCHEMA_PREFIX}_science_contract_v1"
    value["scientific_question"] = (
        "does_the_unchanged_masked_current_to_next_pair_tubelet_jepa_"
        "convert_verified_update100_directional_learning_into_the_complete_"
        "absolute_mechanism_gate_by_update400"
    )
    value["gates"]["update_100"] = deepcopy(V13_UPDATE_100_GATE_CONTRACT)
    value["gates"]["update_400"] = {
        **value["gates"]["update_400"],
        "complete_frozen_v12_evaluator": True,
        "restores_every_original_update100_absolute_conjunct": True,
    }
    value["lifecycle"].update({
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "initialize_fresh_from_original_bound_n320_encoder": True,
        "predecessor_checkpoint_trace_or_runtime_output_reuse_authorized":
            False,
        "update100_or_update400_failure_permanently_closes_mechanism": True,
        "v14_timing_successor_authorized": False,
    })
    return value


def _v13_update_100_conjuncts(
    normalized: Mapping[str, Any],
) -> dict[str, bool]:
    common = normalized["common"]
    values = normalized["values"]
    update0_values = normalized["update0_values"]
    retrieval = normalized["retrieval"]
    update0_retrieval = normalized["update0_retrieval"]
    ratios = normalized["ratios"]
    update0_true_to_mean = _positive_denominator_ratio(
        update0_values["true_pair_mse"],
        update0_values["mean_target_mse"],
        name="V13 update-zero mean-target ratio",
    )
    threshold = PHASE_A_UPDATE_100_THRESHOLDS
    return {
        **{
            name: bool(common[name])
            for name in V13_UPDATE_100_STRUCTURAL_CONJUNCTS
        },
        "non_hold_correct_to_fixed_current_strictly_below_one": (
            retrieval["non_hold_correct_to_current_ratio"]
            < threshold[
                "non_hold_correct_current_ratio_strictly_less_than"
            ]
        ),
        "normalized_projected_future_effective_rank_strictly_above_update0": (
            values["normalized_projected_future_effective_rank"]
            > update0_values["normalized_projected_future_effective_rank"]
        ),
        "action_retrieval_nll_strictly_below_update0": (
            retrieval["action_retrieval_nll"]
            < update0_retrieval["action_retrieval_nll"]
        ),
        "action_retrieval_macro_balanced_accuracy_strictly_above_update0": (
            retrieval["action_retrieval_macro_balanced_accuracy"]
            > update0_retrieval[
                "action_retrieval_macro_balanced_accuracy"
            ]
        ),
        "target_retrieval_nll_strictly_below_update0": (
            retrieval["target_retrieval_nll"]
            < update0_retrieval["target_retrieval_nll"]
        ),
        "same_action_target_retrieval_nll_strictly_below_update0": (
            retrieval["same_action_target_retrieval_nll"]
            < update0_retrieval["same_action_target_retrieval_nll"]
        ),
        "same_action_two_target_nll_strictly_below_update0": (
            retrieval["same_action_two_target_nll"]
            < update0_retrieval["same_action_two_target_nll"]
        ),
        "same_action_strict_win_rate_strictly_above_update0": (
            retrieval["same_action_strict_win_rate"]
            > update0_retrieval["same_action_strict_win_rate"]
        ),
        "same_action_correct_to_deranged_ratio_strictly_below_update0": (
            retrieval["same_action_correct_to_deranged_ratio"]
            < update0_retrieval["same_action_correct_to_deranged_ratio"]
        ),
        "true_to_mean_target_ratio_strictly_below_update0": (
            ratios["true_to_mean_target"] < update0_true_to_mean
        ),
    }


def evaluate_phase_a_continuation(
    update: int,
    metrics: Mapping[str, Any],
    update0_metrics: Mapping[str, Any],
    observation_integrity: Mapping[str, Any],
    previous_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Use the V13 directional gate at 100 and frozen V12 at 400."""

    if update == 400:
        return _FROZEN_V12_EVALUATE_PHASE_A_CONTINUATION(
            update,
            metrics,
            update0_metrics,
            observation_integrity,
            previous_metrics,
        )
    if update != 100:
        raise ValueError("V13 continuation update must be 100 or 400")
    if previous_metrics is not None:
        raise ValueError("V13 update-100 has no previous checkpoint")

    normalized = _normalize_v11_phase_a_inputs(
        metrics, update0_metrics, observation_integrity
    )
    conjuncts = _v13_update_100_conjuncts(normalized)
    passed = all(conjuncts.values())
    return {
        "update": 100,
        "passed": passed,
        "control": (
            CONTROL_CONTINUE
            if passed
            else CONTROL_PHASE_A_UPDATE_100_FAIL
        ),
        "conjuncts": conjuncts,
        "ratios": dict(normalized["ratios"]),
        "thresholds": deepcopy(V13_UPDATE_100_GATE_CONTRACT),
        "per_family": dict(normalized["per_family"]),
        "factorized_retrieval": dict(normalized["retrieval"]),
    }


__all__ = sorted(set(__all__) | {
    name for name in globals()
    if name.isupper() or name in {
        "evaluate_phase_a_continuation",
        "frozen_v12_science_contract",
        "science_contract",
    }
})
