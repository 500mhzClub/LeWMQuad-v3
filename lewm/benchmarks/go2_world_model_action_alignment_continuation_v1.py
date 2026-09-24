"""Pure metrics for the fixed u700-to-u900 alignment continuation.

The continuation is a bounded progress gate.  Its primary statistic is
the absolute change in the treatment arm's hardest family/scene-macro action
margin.  Concurrent-baseline deltas are retained only as diagnostics, because
a worsening baseline can otherwise make a stalled treatment look better.

This module performs no filesystem, tensor, model, or device access.
"""
from __future__ import annotations

import math
import random
from typing import Any, Mapping, Sequence

import numpy as np

from lewm.benchmarks import go2_world_model_action_alignment_successor_v1 as successor
from lewm.benchmarks import go2_world_model_existing_pool_three_arm_v1 as three_arm
from lewm.benchmarks import go2_world_model_v3_action_localization_v1 as localization


SCHEMA = "lewm_go2_world_model_action_alignment_continuation_v1"
CHECKPOINT_UPDATES = (700, 900)
RANK_UPDATES = (700, 800, 900)
PAIRED_BOOTSTRAP_SEED = 20_260_812
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_LOWER_INDEX = 500
BOOTSTRAP_MEDIAN_INDEX = 5_000
BOOTSTRAP_UPPER_INDEX = 9_499
ABSOLUTE_GAIN_THRESHOLD = 0.001298360001376009
RECOVERY_GAIN_THRESHOLD_DIAGNOSTIC_ONLY = 0.0004319475801099867
BALANCED_ACCURACY_CHANCE = 1.0 / three_arm.ACTION_COUNT
RANK_RATIO_RETENTION_MINIMUM = 0.25
RANK_RETENTION_PASS_COUNT = 2
PRESERVED_POSITIVE_ACTION_IDS = (0, 4, 7)

# Frozen u700 values are descriptive trajectory anchors only.  They do not
# replace the registered retention floors and do not affect classification.
U700_BALANCED_ACCURACY_LOWER_ANCHOR = 0.34701964075333114
U700_RANK_RATIO_ANCHOR = 0.47287848726118314
U700_PERSISTENCE_LOWER_ANCHOR = -0.22601831547011703

BOOTSTRAP_ALGORITHM = (
    "python_random_mt19937_getrandbits52_open01_neg_log1p_"
    "shared_checkpoint_family_scene_weights_v1"
)


class AlignmentContinuationMetricError(ValueError):
    """Raised when continuation metric inputs violate the frozen contract."""


def _numeric_array(value: Any, *, name: str) -> np.ndarray:
    try:
        untyped = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise AlignmentContinuationMetricError(f"{name} must be numeric") from error
    if untyped.dtype.kind not in "fiu" or untyped.dtype.kind == "b":
        raise AlignmentContinuationMetricError(f"{name} must be numeric, not boolean or text")
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise AlignmentContinuationMetricError(f"{name} must be numeric") from error


def _candidate_matrix(value: Any, *, count: int, name: str) -> np.ndarray:
    result = _numeric_array(value, name=name)
    if (
        result.shape != (count, three_arm.ACTION_COUNT)
        or not np.isfinite(result).all()
        or bool((result < 0.0).any())
    ):
        raise AlignmentContinuationMetricError(
            f"{name} must be finite, nonnegative, and shaped (N,9)"
        )
    return result


def _positive_vector(value: Any, *, count: int, name: str) -> np.ndarray:
    result = _numeric_array(value, name=name)
    if (
        result.shape != (count,)
        or not np.isfinite(result).all()
        or bool((result <= 0.0).any())
    ):
        raise AlignmentContinuationMetricError(
            f"{name} must be a finite positive vector of length N"
        )
    return result


def _finite_scalar(value: Any, *, name: str, nonnegative: bool = False) -> float:
    if type(value) not in (int, float):
        raise AlignmentContinuationMetricError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite nonnegative" if nonnegative else "finite"
        raise AlignmentContinuationMetricError(f"{name} must be {qualifier}")
    return result


def _boolean_checks(value: Any, *, name: str) -> dict[str, bool]:
    if not isinstance(value, Mapping) or not value:
        raise AlignmentContinuationMetricError(f"{name} must be a nonempty mapping")
    result: dict[str, bool] = {}
    for key, item in value.items():
        if type(key) is not str or not key or type(item) is not bool:
            raise AlignmentContinuationMetricError(
                f"{name} keys must be nonempty strings and values exact booleans"
            )
        result[key] = item
    return result


def _validation_metadata(
    validation_rows: Sequence[Any],
) -> tuple[tuple[Any, ...], tuple[str, ...], tuple[str, ...], dict[str, str], np.ndarray]:
    try:
        rows = three_arm.normalize_h6_metadata_rows(validation_rows)
    except (TypeError, ValueError) as error:
        raise AlignmentContinuationMetricError("validation rows are invalid") from error
    count = len(rows)
    if count < 1 or tuple(row.index for row in rows) != tuple(range(count)):
        raise AlignmentContinuationMetricError("validation row order changed")
    if any(row.role != "val" for row in rows):
        raise AlignmentContinuationMetricError("continuation metrics require validation rows")
    scenes = tuple(row.scene_id for row in rows)
    families = tuple(row.family for row in rows)
    actions = np.asarray([row.candidate_action_id for row in rows], dtype=np.int64)
    scene_family: dict[str, str] = {}
    for scene, family in zip(scenes, families, strict=True):
        previous = scene_family.setdefault(scene, family)
        if previous != family:
            raise AlignmentContinuationMetricError("a scene crossed registered families")
    if set(families) != set(three_arm.REGISTERED_FAMILIES):
        raise AlignmentContinuationMetricError("a registered family is absent")
    if set(int(action) for action in actions) != set(range(three_arm.ACTION_COUNT)):
        raise AlignmentContinuationMetricError("a registered action is absent")
    return rows, scenes, families, scene_family, actions


def _positive_exponential_weight(rng: random.Random) -> float:
    bits = rng.getrandbits(52)
    uniform_open = (bits + 1) / (2**52 + 1)
    result = -math.log1p(-uniform_open)
    if not math.isfinite(result) or result <= 0.0:
        raise AssertionError("strictly positive bootstrap weight failed")
    return result


def paired_absolute_hardest_margin_gain(
    *,
    treatment_candidate_energy_u700: Any,
    treatment_candidate_energy_u900: Any,
    validation_rows: Sequence[Any],
) -> dict[str, Any]:
    """Return H900-H700 using one shared family/scene bootstrap draw."""

    rows, scenes, _families, scene_family, actions = _validation_metadata(
        validation_rows
    )
    count = len(rows)
    candidates = {
        "u700": _candidate_matrix(
            treatment_candidate_energy_u700,
            count=count,
            name="treatment candidate energy u700",
        ),
        "u900": _candidate_matrix(
            treatment_candidate_energy_u900,
            count=count,
            name="treatment candidate energy u900",
        ),
    }
    row_ids = np.arange(count)
    margins: dict[str, np.ndarray] = {}
    for checkpoint, candidate in candidates.items():
        factual = candidate[row_ids, actions]
        wrong = candidate.copy()
        wrong[row_ids, actions] = math.inf
        margins[checkpoint] = wrong.min(axis=1) - factual

    family_scenes = {
        family: tuple(
            scene
            for scene in sorted(scene_family)
            if scene_family[scene] == family
        )
        for family in three_arm.REGISTERED_FAMILIES
    }
    cells: dict[str, dict[tuple[str, int], float]] = {
        checkpoint: {} for checkpoint in ("u700", "u900")
    }
    for action in range(three_arm.ACTION_COUNT):
        for scene in sorted(scene_family):
            selected = [
                index
                for index, (row_scene, row_action) in enumerate(
                    zip(scenes, actions, strict=True)
                )
                if row_scene == scene and int(row_action) == action
            ]
            if selected:
                for checkpoint in cells:
                    cells[checkpoint][(scene, action)] = float(
                        margins[checkpoint][selected].mean()
                    )

    def action_points(checkpoint: str) -> list[float]:
        result: list[float] = []
        for action in range(three_arm.ACTION_COUNT):
            family_values = []
            for family in three_arm.REGISTERED_FAMILIES:
                selected = [
                    cells[checkpoint][(scene, action)]
                    for scene in family_scenes[family]
                    if (scene, action) in cells[checkpoint]
                ]
                if not selected:
                    raise AlignmentContinuationMetricError(
                        "every family/action cell requires support"
                    )
                family_values.append(math.fsum(selected) / len(selected))
            result.append(math.fsum(family_values) / len(family_values))
        return result

    points = {checkpoint: action_points(checkpoint) for checkpoint in cells}
    hardest = {checkpoint: min(points[checkpoint]) for checkpoint in cells}
    point_gain = hardest["u900"] - hardest["u700"]

    rng = random.Random(PAIRED_BOOTSTRAP_SEED)
    draws: list[float] = []
    for _replicate in range(BOOTSTRAP_REPLICATES):
        weights = {
            family: {
                scene: _positive_exponential_weight(rng)
                for scene in family_scenes[family]
            }
            for family in three_arm.REGISTERED_FAMILIES
        }
        replicate_hardest: dict[str, float] = {}
        for checkpoint in cells:
            per_action = []
            for action in range(three_arm.ACTION_COUNT):
                family_values = []
                for family in three_arm.REGISTERED_FAMILIES:
                    selected = [
                        (scene, weights[family][scene])
                        for scene in family_scenes[family]
                        if (scene, action) in cells[checkpoint]
                    ]
                    denominator = math.fsum(weight for _scene, weight in selected)
                    family_values.append(
                        math.fsum(
                            weight * cells[checkpoint][(scene, action)]
                            for scene, weight in selected
                        )
                        / denominator
                    )
                per_action.append(
                    math.fsum(family_values) / len(three_arm.REGISTERED_FAMILIES)
                )
            replicate_hardest[checkpoint] = min(per_action)
        draws.append(replicate_hardest["u900"] - replicate_hardest["u700"])
    ordered = sorted(draws)
    return {
        "definition": "hardest_action_margin_u900_minus_u700_within_alignment_arm",
        "decision_relevant": True,
        "bootstrap_algorithm": BOOTSTRAP_ALGORITHM,
        "bootstrap_interpretation": localization.BOOTSTRAP_INTERPRETATION,
        "bootstrap_seed": PAIRED_BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_lower_index": BOOTSTRAP_LOWER_INDEX,
        "bootstrap_median_index": BOOTSTRAP_MEDIAN_INDEX,
        "bootstrap_upper_index": BOOTSTRAP_UPPER_INDEX,
        "per_action_points": points,
        "hardest_action_margin": hardest,
        "point": point_gain,
        "one_sided_95_lower_quantile": ordered[BOOTSTRAP_LOWER_INDEX],
        "median_quantile": ordered[BOOTSTRAP_MEDIAN_INDEX],
        "one_sided_95_upper_quantile": ordered[BOOTSTRAP_UPPER_INDEX],
        "absolute_gain_threshold": ABSOLUTE_GAIN_THRESHOLD,
        "recovery_gain_threshold_diagnostic_only": (
            RECOVERY_GAIN_THRESHOLD_DIAGNOSTIC_ONLY
        ),
        "recovery_threshold_affects_decision": False,
    }


def classify_continuation_outcome(
    *,
    contract_passed: bool,
    retention_passed: bool,
    action_alignment_repaired: bool,
    persistence_lower_failure_count: int,
    aggregate_persistence_lower: float,
    absolute_gain_point: float,
    absolute_gain_lower: float,
    absolute_gain_upper: float,
) -> tuple[str, str, str]:
    """Apply the terminal continuation precedence to validated summaries."""

    for name, value in (
        ("contract passed", contract_passed),
        ("retention passed", retention_passed),
        ("action alignment repaired", action_alignment_repaired),
    ):
        if type(value) is not bool:
            raise AlignmentContinuationMetricError(f"{name} must be boolean")
    if (
        type(persistence_lower_failure_count) is not int
        or not 0 <= persistence_lower_failure_count <= three_arm.ACTION_COUNT
    ):
        raise AlignmentContinuationMetricError(
            "persistence failure count must be an integer from zero through nine"
        )
    persistence_lower = _finite_scalar(
        aggregate_persistence_lower, name="aggregate persistence lower"
    )
    gain_point = _finite_scalar(absolute_gain_point, name="absolute gain point")
    gain_lower = _finite_scalar(absolute_gain_lower, name="absolute gain lower")
    gain_upper = _finite_scalar(absolute_gain_upper, name="absolute gain upper")

    close = "NO_FURTHER_ALIGNMENT_TRAINING_OR_PLANNING_GATE"
    if not contract_passed:
        return "FAIL_CONTRACT_CLOSE_ALIGNMENT_BRANCH", "CLOSE", close
    if not retention_passed:
        return "FAIL_RETENTION_CLOSE_ALIGNMENT_BRANCH", "CLOSE", close
    if action_alignment_repaired:
        if persistence_lower_failure_count >= localization.PERSISTENCE_SYSTEMIC_FAILURE_COUNT:
            return (
                "PASS_ACTION_ALIGNMENT_PROXY_REPAIR_PERSISTENCE_SYSTEMIC",
                "PERSISTENCE_SYSTEMIC",
                "PERSISTENCE_RESIDUAL_VS_MATCHED_BASELINE",
            )
        if persistence_lower_failure_count > 0 or persistence_lower <= 0.0:
            return (
                "PASS_ACTION_ALIGNMENT_PROXY_REPAIR_PLANNING_WITH_PROXY_CAVEAT",
                "PLANNING_WITH_PROXY_CAVEAT",
                "PLANNING_USEFULNESS_GATE_WITH_PROXY_CAVEAT",
            )
        return (
            "PASS_EXPLORATORY_ACTION_ALIGNMENT_AND_PREDICTOR_USEFULNESS_PROXY",
            "CLEAN_PROXY_PASS",
            "PROCEED_TO_PLANNING_USEFULNESS_GATE",
        )

    if gain_point <= 0.0 or gain_upper <= 0.0:
        return "STALLED_OR_HARMFUL_CLOSE_ALIGNMENT_BRANCH", "CLOSE", close
    if gain_point >= ABSOLUTE_GAIN_THRESHOLD and gain_lower > 0.0:
        return (
            "MEANINGFUL_ABSOLUTE_PROGRESS_INCOMPLETE_CONTINUE_SAME_MECHANISM",
            "CONTINUE_SAME_MECHANISM",
            "PREREGISTER_NEXT_FIXED_SAME_MECHANISM_BLOCK",
        )
    if 0.0 < gain_point < ABSOLUTE_GAIN_THRESHOLD and gain_lower > 0.0:
        return "POSITIVE_BUT_INSUFFICIENT_RATE_CLOSE_ALIGNMENT_BRANCH", "CLOSE", close
    return "INCONCLUSIVE_ABSOLUTE_CHANGE_CLOSE_ALIGNMENT_BRANCH", "CLOSE", close


def decide_alignment_continuation(
    *,
    baseline_candidate_energy_u700: Any,
    baseline_candidate_energy_u900: Any,
    treatment_candidate_energy_u700: Any,
    treatment_factual_energy_u700: Any,
    treatment_persistence_energy_u700: Any,
    treatment_wrong_history_energy_u700: Any,
    treatment_candidate_energy_u900: Any,
    treatment_factual_energy_u900: Any,
    treatment_persistence_energy_u900: Any,
    treatment_wrong_history_energy_u900: Any,
    validation_rows: Sequence[Any],
    treatment_rank_ratio_by_update: Mapping[int, float],
    contract_checks: Mapping[str, bool],
    train_fit_checks: Mapping[str, bool],
) -> dict[str, Any]:
    """Compute and classify one fixed continuation block without side effects."""

    rows, _scenes, _families, _scene_family, _actions = _validation_metadata(
        validation_rows
    )
    count = len(rows)
    candidates = {
        "baseline_u700": _candidate_matrix(
            baseline_candidate_energy_u700, count=count, name="baseline candidate energy u700"
        ),
        "baseline_u900": _candidate_matrix(
            baseline_candidate_energy_u900, count=count, name="baseline candidate energy u900"
        ),
        "treatment_u700": _candidate_matrix(
            treatment_candidate_energy_u700, count=count, name="treatment candidate energy u700"
        ),
        "treatment_u900": _candidate_matrix(
            treatment_candidate_energy_u900, count=count, name="treatment candidate energy u900"
        ),
    }
    vectors = {
        "factual_u700": _positive_vector(
            treatment_factual_energy_u700, count=count, name="treatment factual energy u700"
        ),
        "persistence_u700": _positive_vector(
            treatment_persistence_energy_u700,
            count=count,
            name="treatment persistence energy u700",
        ),
        "wrong_history_u700": _positive_vector(
            treatment_wrong_history_energy_u700,
            count=count,
            name="treatment wrong-history energy u700",
        ),
        "factual_u900": _positive_vector(
            treatment_factual_energy_u900, count=count, name="treatment factual energy u900"
        ),
        "persistence_u900": _positive_vector(
            treatment_persistence_energy_u900,
            count=count,
            name="treatment persistence energy u900",
        ),
        "wrong_history_u900": _positive_vector(
            treatment_wrong_history_energy_u900,
            count=count,
            name="treatment wrong-history energy u900",
        ),
    }
    if not isinstance(treatment_rank_ratio_by_update, Mapping) or set(
        treatment_rank_ratio_by_update
    ) != set(RANK_UPDATES):
        raise AlignmentContinuationMetricError("rank tail must be exactly u700/u800/u900")
    ranks = {
        str(update): _finite_scalar(
            treatment_rank_ratio_by_update[update],
            name=f"rank ratio u{update}",
            nonnegative=True,
        )
        for update in RANK_UPDATES
    }
    contract = _boolean_checks(contract_checks, name="contract checks")
    train_fit = _boolean_checks(train_fit_checks, name="train-fit checks")

    try:
        localizations = {
            "u700": localization.localize_action_and_controls(
                candidate_energies=candidates["treatment_u700"],
                factual_energy=vectors["factual_u700"],
                persistence_energy=vectors["persistence_u700"],
                wrong_history_energy=vectors["wrong_history_u700"],
                validation_rows=rows,
            ),
            "u900": localization.localize_action_and_controls(
                candidate_energies=candidates["treatment_u900"],
                factual_energy=vectors["factual_u900"],
                persistence_energy=vectors["persistence_u900"],
                wrong_history_energy=vectors["wrong_history_u900"],
                validation_rows=rows,
            ),
        }
        absolute_gain = paired_absolute_hardest_margin_gain(
            treatment_candidate_energy_u700=candidates["treatment_u700"],
            treatment_candidate_energy_u900=candidates["treatment_u900"],
            validation_rows=rows,
        )
        relative_deltas = {
            "u700": successor.paired_minimum_action_margin_delta(
                baseline_candidate_energy=candidates["baseline_u700"],
                treatment_candidate_energy=candidates["treatment_u700"],
                validation_rows=rows,
            ),
            "u900": successor.paired_minimum_action_margin_delta(
                baseline_candidate_energy=candidates["baseline_u900"],
                treatment_candidate_energy=candidates["treatment_u900"],
                validation_rows=rows,
            ),
        }
    except (localization.ActionLocalizationError, successor.AlignmentSuccessorMetricError) as error:
        raise AlignmentContinuationMetricError(str(error)) from error

    u700 = localizations["u700"]
    u900 = localizations["u900"]
    margins700 = u700["action_margin_localization"]["per_action"]
    margins900 = u900["action_margin_localization"]["per_action"]
    rank_pass_count = sum(
        value >= RANK_RATIO_RETENTION_MINIMUM for value in ranks.values()
    )
    preserved_checks = {
        str(action): (
            margins700[action]["family_equal_scene_macro_point"] > 0.0
            and margins700[action]["one_sided_95_lower_quantile"] > 0.0
            and margins900[action]["family_equal_scene_macro_point"] > 0.0
            and margins900[action]["one_sided_95_lower_quantile"] > 0.0
        )
        for action in PRESERVED_POSITIVE_ACTION_IDS
    }
    u900_ba_lower = float(
        u900["action_identification"]["balanced_accuracy_bootstrap_lower_95"]
    )
    u700_ba_lower = float(
        u700["action_identification"]["balanced_accuracy_bootstrap_lower_95"]
    )
    u900_persistence_lower = float(
        u900["registered_control_reproduction"]["persistence"]["bootstrap_lower_95"]
    )
    u700_persistence_lower = float(
        u700["registered_control_reproduction"]["persistence"]["bootstrap_lower_95"]
    )
    u900_wrong_lower = float(
        u900["registered_control_reproduction"]["wrong_history"]["bootstrap_lower_95"]
    )
    u700_wrong_lower = float(
        u700["registered_control_reproduction"]["wrong_history"]["bootstrap_lower_95"]
    )
    contract_passed = all(contract.values())
    retention_checks = {
        "balanced_accuracy_lower_above_chance": u900_ba_lower > BALANCED_ACCURACY_CHANCE,
        "wrong_history_lower_positive": u900_wrong_lower > 0.0,
        "rank_ratio_at_least_0_25_at_two_of_three_tail_updates": (
            rank_pass_count >= RANK_RETENTION_PASS_COUNT
        ),
        "preserve_u700_positive_action_point_and_lower_ids_0_4_7": all(
            preserved_checks.values()
        ),
        "all_contract_checks": contract_passed,
        "all_train_fit_checks": all(train_fit.values()),
    }
    retention_passed = all(retention_checks.values())
    action_alignment_repaired = bool(
        u900["routing_decision"]["alignment_repaired"]
    )
    persistence_failure_ids = tuple(
        int(action)
        for action in u900["failure_topology"]["persistence_lower_failure_action_ids"]
    )
    status, branch, next_step = classify_continuation_outcome(
        contract_passed=contract_passed,
        retention_passed=retention_passed,
        action_alignment_repaired=action_alignment_repaired,
        persistence_lower_failure_count=len(persistence_failure_ids),
        aggregate_persistence_lower=u900_persistence_lower,
        absolute_gain_point=absolute_gain["point"],
        absolute_gain_lower=absolute_gain["one_sided_95_lower_quantile"],
        absolute_gain_upper=absolute_gain["one_sided_95_upper_quantile"],
    )

    strict_proxy_checks = {
        "all_action_margin_points_positive": all(
            row["family_equal_scene_macro_point"] > 0.0 for row in margins900
        ),
        "all_action_margin_lower_quantiles_positive": all(
            row["one_sided_95_lower_quantile"] > 0.0 for row in margins900
        ),
        "shared_minimum_margin_lower_positive": (
            u900["action_margin_localization"]
            ["shared_across_action_minimum_quantiles"]
            ["one_sided_95_lower_quantile"]
            > 0.0
        ),
        "balanced_accuracy_lower_above_chance": retention_checks[
            "balanced_accuracy_lower_above_chance"
        ],
        "persistence_aggregate_lower_positive": u900_persistence_lower > 0.0,
        "all_persistence_action_lower_quantiles_positive": not persistence_failure_ids,
        "wrong_history_lower_positive": retention_checks["wrong_history_lower_positive"],
        "rank_tail_pass": rank_pass_count >= RANK_RETENTION_PASS_COUNT,
        "all_contract_checks": contract_passed,
        "all_train_fit_checks": all(train_fit.values()),
    }
    return {
        "schema": SCHEMA,
        "status": status,
        "branch_class": branch,
        "selected_next_step": next_step,
        "authorizes_next_execution": False,
        "authorizes_further_alignment_training": False,
        "permits_separate_same_mechanism_preregistration": (
            status
            == "MEANINGFUL_ABSOLUTE_PROGRESS_INCOMPLETE_CONTINUE_SAME_MECHANISM"
        ),
        "citable_as_fresh_confirmation": False,
        "absolute_treatment_hardest_margin_gain": absolute_gain,
        "concurrent_baseline_relative_delta_diagnostic_only": {
            "decision_relevant": False,
            "warning": (
                "relative improvement can be caused by concurrent-baseline degradation"
            ),
            "by_update": relative_deltas,
            "point_change_u900_minus_u700": (
                relative_deltas["u900"]["point"] - relative_deltas["u700"]["point"]
            ),
        },
        "contract": {
            "passed": contract_passed,
            "checks": contract,
            "failed_checks": [name for name, value in contract.items() if not value],
        },
        "retention": {
            "passed": retention_passed,
            "checks": retention_checks,
            "failed_checks": [
                name for name, value in retention_checks.items() if not value
            ],
            "preserved_positive_action_ids": list(PRESERVED_POSITIVE_ACTION_IDS),
            "preservation_checks_by_action_id": preserved_checks,
            "rank_ratio_by_update": ranks,
            "rank_pass_update_count": rank_pass_count,
            "train_fit_checks": train_fit,
        },
        "strict_proxy_thresholds": {
            "passed": all(strict_proxy_checks.values()),
            "checks": strict_proxy_checks,
            "action_alignment_repaired": action_alignment_repaired,
            "persistence_lower_failure_action_ids": list(persistence_failure_ids),
        },
        "descriptive_trajectory_anchors_not_used_for_decision": {
            "decision_relevant": False,
            "u700_frozen_anchors": {
                "balanced_accuracy_lower": U700_BALANCED_ACCURACY_LOWER_ANCHOR,
                "rank_ratio": U700_RANK_RATIO_ANCHOR,
                "persistence_lower": U700_PERSISTENCE_LOWER_ANCHOR,
            },
            "observed": {
                "balanced_accuracy_lower_u700": u700_ba_lower,
                "balanced_accuracy_lower_u900": u900_ba_lower,
                "balanced_accuracy_lower_change": u900_ba_lower - u700_ba_lower,
                "rank_ratio_u700": ranks["700"],
                "rank_ratio_u900": ranks["900"],
                "rank_ratio_change": ranks["900"] - ranks["700"],
                "persistence_lower_u700": u700_persistence_lower,
                "persistence_lower_u900": u900_persistence_lower,
                "persistence_lower_change": u900_persistence_lower - u700_persistence_lower,
                "wrong_history_lower_u700": u700_wrong_lower,
                "wrong_history_lower_u900": u900_wrong_lower,
                "wrong_history_lower_change": u900_wrong_lower - u700_wrong_lower,
            },
        },
        "localizations": localizations,
        "conditional_continuation_caveat": (
            "the u700-to-u900 continuation was selected conditionally on the frozen "
            "u700 result; bootstrap quantiles are conditional descriptive Bayesian "
            "quantiles and are not a fresh confirmation role"
        ),
        "claim_boundary": [
            "bounded changed-objective exploratory development continuation",
            "meaningful absolute progress permits only a separately preregistered unchanged-mechanism block",
            "no outcome automatically authorizes further alignment training",
            "concurrent-baseline-relative changes are diagnostic only",
            "no fresh blind or shuffled confirmation role is present",
            "no planning, navigation, promotion, deployment, or production claim",
        ],
    }


__all__ = [
    "ABSOLUTE_GAIN_THRESHOLD",
    "AlignmentContinuationMetricError",
    "BALANCED_ACCURACY_CHANCE",
    "BOOTSTRAP_ALGORITHM",
    "BOOTSTRAP_LOWER_INDEX",
    "BOOTSTRAP_MEDIAN_INDEX",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_UPPER_INDEX",
    "CHECKPOINT_UPDATES",
    "PAIRED_BOOTSTRAP_SEED",
    "PRESERVED_POSITIVE_ACTION_IDS",
    "RANK_RETENTION_PASS_COUNT",
    "RANK_RATIO_RETENTION_MINIMUM",
    "RANK_UPDATES",
    "RECOVERY_GAIN_THRESHOLD_DIAGNOSTIC_ONLY",
    "SCHEMA",
    "U700_BALANCED_ACCURACY_LOWER_ANCHOR",
    "U700_PERSISTENCE_LOWER_ANCHOR",
    "U700_RANK_RATIO_ANCHOR",
    "classify_continuation_outcome",
    "decide_alignment_continuation",
    "paired_absolute_hardest_margin_gain",
]
