"""Metrics for the frozen two-arm action-alignment successor.

This module has no filesystem or model access.  It compares a concurrently
trained factual-loss baseline with the single preregistered global alignment
treatment.  Positive action margins mean that the factual requested action has
lower prediction energy than every wrong requested action.
"""
from __future__ import annotations

import math
import random
from typing import Any, Mapping, Sequence

import numpy as np

from lewm.benchmarks import go2_world_model_existing_pool_three_arm_v1 as three_arm
from lewm.benchmarks import go2_world_model_v3_action_localization_v1 as localization


SCHEMA = "lewm_go2_world_model_action_alignment_successor_v1"
ARM_NAMES = ("baseline", "alignment")
TAIL_UPDATES = (500, 600, 700)
PAIRED_BOOTSTRAP_SEED = 20_260_811
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_LOWER_INDEX = 500
BOOTSTRAP_MEDIAN_INDEX = 5_000
BOOTSTRAP_UPPER_INDEX = 9_499
V3_ALIGNMENT_POINT_DEFICIT = 0.009453551490358742
MEANINGFUL_POINT_THRESHOLD = 0.25 * V3_ALIGNMENT_POINT_DEFICIT
STALL_UPPER_THRESHOLD = 0.10 * V3_ALIGNMENT_POINT_DEFICIT
RANK_RATIO_MINIMUM = 0.25
RANK_PASS_COUNT = 2


class AlignmentSuccessorMetricError(ValueError):
    """Raised when a two-arm metric input violates the frozen contract."""


def _candidate_matrix(value: Any, *, count: int, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AlignmentSuccessorMetricError(f"{name} must be numeric") from error
    if (
        result.shape != (count, three_arm.ACTION_COUNT)
        or not np.isfinite(result).all()
        or bool((result < 0.0).any())
    ):
        raise AlignmentSuccessorMetricError(
            f"{name} must be finite, nonnegative, and shaped (N,9)"
        )
    return result


def _positive_exponential_weight(rng: random.Random) -> float:
    bits = rng.getrandbits(52)
    uniform_open = (bits + 1) / (2**52 + 1)
    weight = -math.log1p(-uniform_open)
    if not math.isfinite(weight) or weight <= 0.0:
        raise AssertionError("strictly positive bootstrap weight failed")
    return weight


def paired_minimum_action_margin_delta(
    *,
    baseline_candidate_energy: Any,
    treatment_candidate_energy: Any,
    validation_rows: Sequence[Any],
) -> dict[str, Any]:
    """Return the preregistered paired delta of minimum action margins."""

    rows = three_arm.normalize_h6_metadata_rows(validation_rows)
    count = len(rows)
    if count < 1 or tuple(row.index for row in rows) != tuple(range(count)):
        raise AlignmentSuccessorMetricError("validation row order changed")
    if any(row.role != "val" for row in rows):
        raise AlignmentSuccessorMetricError("paired delta requires validation rows")
    baseline = _candidate_matrix(
        baseline_candidate_energy, count=count, name="baseline candidate energy"
    )
    treatment = _candidate_matrix(
        treatment_candidate_energy, count=count, name="treatment candidate energy"
    )
    actions = np.asarray([row.candidate_action_id for row in rows], dtype=np.int64)
    scenes = tuple(row.scene_id for row in rows)
    families = tuple(row.family for row in rows)
    scene_family: dict[str, str] = {}
    for scene, family in zip(scenes, families, strict=True):
        previous = scene_family.setdefault(scene, family)
        if previous != family:
            raise AlignmentSuccessorMetricError("a scene crossed registered families")
    if set(families) != set(three_arm.REGISTERED_FAMILIES):
        raise AlignmentSuccessorMetricError("a registered family is absent")

    row_ids = np.arange(count)
    margins: dict[str, np.ndarray] = {}
    for name, candidate in (("baseline", baseline), ("alignment", treatment)):
        factual = candidate[row_ids, actions]
        wrong = candidate.copy()
        wrong[row_ids, actions] = math.inf
        margins[name] = wrong.min(axis=1) - factual

    family_scenes = {
        family: tuple(
            scene
            for scene in sorted(scene_family)
            if scene_family[scene] == family
        )
        for family in three_arm.REGISTERED_FAMILIES
    }
    cells: dict[str, dict[tuple[str, int], float]] = {
        name: {} for name in ARM_NAMES
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
                for name in ARM_NAMES:
                    cells[name][(scene, action)] = float(
                        margins[name][selected].mean()
                    )

    def action_points(name: str) -> list[float]:
        result: list[float] = []
        for action in range(three_arm.ACTION_COUNT):
            family_values = []
            for family in three_arm.REGISTERED_FAMILIES:
                selected = [
                    cells[name][(scene, action)]
                    for scene in family_scenes[family]
                    if (scene, action) in cells[name]
                ]
                if not selected:
                    raise AlignmentSuccessorMetricError(
                        "every family/action cell requires support"
                    )
                family_values.append(math.fsum(selected) / len(selected))
            result.append(math.fsum(family_values) / len(family_values))
        return result

    points = {name: action_points(name) for name in ARM_NAMES}
    arm_minimum = {name: min(points[name]) for name in ARM_NAMES}
    point_delta = arm_minimum["alignment"] - arm_minimum["baseline"]

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
        replicate_minima: dict[str, float] = {}
        for name in ARM_NAMES:
            per_action = []
            for action in range(three_arm.ACTION_COUNT):
                family_values = []
                for family in three_arm.REGISTERED_FAMILIES:
                    selected = [
                        (scene, weights[family][scene])
                        for scene in family_scenes[family]
                        if (scene, action) in cells[name]
                    ]
                    denominator = math.fsum(weight for _scene, weight in selected)
                    family_values.append(
                        math.fsum(
                            weight * cells[name][(scene, action)]
                            for scene, weight in selected
                        )
                        / denominator
                    )
                per_action.append(
                    math.fsum(family_values) / len(three_arm.REGISTERED_FAMILIES)
                )
            replicate_minima[name] = min(per_action)
        draws.append(replicate_minima["alignment"] - replicate_minima["baseline"])
    ordered = sorted(draws)
    return {
        "definition": "min_action_margin_alignment_minus_concurrent_baseline",
        "bootstrap_algorithm": (
            "python_random_mt19937_getrandbits52_open01_neg_log1p_"
            "shared_arm_family_scene_weights_v1"
        ),
        "bootstrap_interpretation": localization.BOOTSTRAP_INTERPRETATION,
        "bootstrap_seed": PAIRED_BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_lower_index": BOOTSTRAP_LOWER_INDEX,
        "bootstrap_median_index": BOOTSTRAP_MEDIAN_INDEX,
        "bootstrap_upper_index": BOOTSTRAP_UPPER_INDEX,
        "per_action_points": points,
        "minimum_action_margin": arm_minimum,
        "point": point_delta,
        "one_sided_95_lower_quantile": ordered[BOOTSTRAP_LOWER_INDEX],
        "median_quantile": ordered[BOOTSTRAP_MEDIAN_INDEX],
        "one_sided_95_upper_quantile": ordered[BOOTSTRAP_UPPER_INDEX],
        "meaningful_point_threshold": MEANINGFUL_POINT_THRESHOLD,
        "stall_upper_threshold": STALL_UPPER_THRESHOLD,
    }


def classify_alignment_outcome(
    *, alignment_repaired: bool, retention_passed: bool,
    delta_point: float, delta_lower: float, delta_upper: float,
    repaired_next_step: str,
) -> tuple[str, str, str]:
    """Apply only the frozen terminal precedence to already-computed values."""

    if type(alignment_repaired) is not bool or type(retention_passed) is not bool:
        raise AlignmentSuccessorMetricError("repair and retention flags must be boolean")
    values = (delta_point, delta_lower, delta_upper)
    if any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in values):
        raise AlignmentSuccessorMetricError("paired delta values must be finite")
    if type(repaired_next_step) is not str or not repaired_next_step:
        raise AlignmentSuccessorMetricError("repaired next step must be nonempty")
    if alignment_repaired and retention_passed:
        return (
            "PASS_EXPLORATORY_ACTION_ALIGNMENT_PROXY_REPAIR",
            "REPAIR",
            repaired_next_step,
        )
    if not retention_passed:
        return (
            "FAIL_RETENTION_CLOSE_ALIGNMENT_BRANCH",
            "CLOSE",
            "NO_FURTHER_ALIGNMENT_TWEAK",
        )
    if delta_point >= MEANINGFUL_POINT_THRESHOLD and delta_lower > 0.0:
        return (
            "MEANINGFUL_ALIGNMENT_IMPROVEMENT_INCOMPLETE",
            "MEANINGFUL",
            "ONE_FIXED_SAME_MECHANISM_CONTINUATION_MAY_BE_PREREGISTERED",
        )
    if delta_upper < STALL_UPPER_THRESHOLD:
        return (
            "STALLED_CLOSE_ALIGNMENT_BRANCH",
            "CLOSE",
            "NO_FURTHER_ALIGNMENT_TWEAK",
        )
    return (
        "INCONCLUSIVE_ALIGNMENT_COMPARISON",
        "INCONCLUSIVE",
        "ONE_IDENTICAL_REPLICATION_MAY_BE_PREREGISTERED",
    )


def decide_alignment_successor(
    *,
    baseline_candidate_energy: Any,
    baseline_factual_energy: Any,
    baseline_persistence_energy: Any,
    baseline_wrong_history_energy: Any,
    treatment_candidate_energy: Any,
    treatment_factual_energy: Any,
    treatment_persistence_energy: Any,
    treatment_wrong_history_energy: Any,
    validation_rows: Sequence[Any],
    treatment_rank_ratio_by_update: Mapping[int, float],
    contract_checks: Mapping[str, bool],
    train_fit_checks: Mapping[str, bool],
) -> dict[str, Any]:
    """Apply the frozen repair, meaningful-improvement, and stall precedence."""

    if set(treatment_rank_ratio_by_update) != set(TAIL_UPDATES):
        raise AlignmentSuccessorMetricError("rank tail must be u500/u600/u700")
    if any(type(key) is not str or type(value) is not bool for key, value in contract_checks.items()):
        raise AlignmentSuccessorMetricError("contract checks must be boolean")
    if any(type(key) is not str or type(value) is not bool for key, value in train_fit_checks.items()):
        raise AlignmentSuccessorMetricError("train-fit checks must be boolean")
    ranks = {str(update): float(treatment_rank_ratio_by_update[update]) for update in TAIL_UPDATES}
    if any(not math.isfinite(value) or value < 0.0 for value in ranks.values()):
        raise AlignmentSuccessorMetricError("rank ratios must be finite nonnegative")

    localizations = {
        "baseline": localization.localize_action_and_controls(
            candidate_energies=baseline_candidate_energy,
            factual_energy=baseline_factual_energy,
            persistence_energy=baseline_persistence_energy,
            wrong_history_energy=baseline_wrong_history_energy,
            validation_rows=validation_rows,
        ),
        "alignment": localization.localize_action_and_controls(
            candidate_energies=treatment_candidate_energy,
            factual_energy=treatment_factual_energy,
            persistence_energy=treatment_persistence_energy,
            wrong_history_energy=treatment_wrong_history_energy,
            validation_rows=validation_rows,
        ),
    }
    delta = paired_minimum_action_margin_delta(
        baseline_candidate_energy=baseline_candidate_energy,
        treatment_candidate_energy=treatment_candidate_energy,
        validation_rows=validation_rows,
    )
    treatment = localizations["alignment"]
    baseline = localizations["baseline"]
    treatment_margin = treatment["action_margin_localization"]
    baseline_margin = baseline["action_margin_localization"]
    alignment_repaired = bool(treatment["routing_decision"]["alignment_repaired"])
    rank_pass_count = sum(value >= RANK_RATIO_MINIMUM for value in ranks.values())
    newly_nonpositive = [
        treatment_row["action_id"]
        for baseline_row, treatment_row in zip(
            baseline_margin["per_action"], treatment_margin["per_action"], strict=True
        )
        if baseline_row["family_equal_scene_macro_point"] > 0.0
        and treatment_row["family_equal_scene_macro_point"] <= 0.0
    ]
    retention_checks = {
        "balanced_accuracy_lower_above_chance": (
            treatment["action_identification"]["balanced_accuracy_bootstrap_lower_95"]
            > 1.0 / three_arm.ACTION_COUNT
        ),
        "wrong_history_lower_positive": (
            treatment["registered_control_reproduction"]["wrong_history"]["bootstrap_lower_95"]
            > 0.0
        ),
        "rank_ratio_at_least_0_25_at_two_tail_updates": rank_pass_count >= RANK_PASS_COUNT,
        "no_newly_nonpositive_action_point": not newly_nonpositive,
        "all_contract_checks": all(contract_checks.values()),
        "all_train_fit_checks": all(train_fit_checks.values()),
    }
    retention_passed = all(retention_checks.values())

    status, branch, next_step = classify_alignment_outcome(
        alignment_repaired=alignment_repaired,
        retention_passed=retention_passed,
        delta_point=delta["point"],
        delta_lower=delta["one_sided_95_lower_quantile"],
        delta_upper=delta["one_sided_95_upper_quantile"],
        repaired_next_step=treatment["routing_decision"]["selected_next_step"],
    )

    latent_checks = {
        "balanced_accuracy_lower_above_chance": retention_checks[
            "balanced_accuracy_lower_above_chance"
        ],
        "all_action_margin_points_positive": all(
            row["family_equal_scene_macro_point"] > 0.0
            for row in treatment_margin["per_action"]
        ),
        "all_action_margin_lower_quantiles_positive": all(
            row["one_sided_95_lower_quantile"] > 0.0
            for row in treatment_margin["per_action"]
        ),
        "shared_minimum_margin_lower_positive": (
            treatment_margin["shared_across_action_minimum_quantiles"]
            ["one_sided_95_lower_quantile"]
            > 0.0
        ),
        "persistence_lower_positive": (
            treatment["registered_control_reproduction"]["persistence"]
            ["bootstrap_lower_95"]
            > 0.0
        ),
        "wrong_history_lower_positive": retention_checks[
            "wrong_history_lower_positive"
        ],
        "rank_tail_pass": rank_pass_count >= RANK_PASS_COUNT,
    }
    return {
        "schema": SCHEMA,
        "status": status,
        "branch_class": branch,
        "passed_exploratory_alignment_repair": (
            status == "PASS_EXPLORATORY_ACTION_ALIGNMENT_PROXY_REPAIR"
        ),
        "citable_as_original_factual_learnability_claim": False,
        "authorizes_next_execution": False,
        "selected_next_step": next_step,
        "paired_alignment_delta": delta,
        "retention": {
            "passed": retention_passed,
            "checks": retention_checks,
            "failed_checks": [name for name, value in retention_checks.items() if not value],
            "newly_nonpositive_action_ids": newly_nonpositive,
            "rank_ratio_by_update": ranks,
            "rank_pass_update_count": rank_pass_count,
        },
        "latent_proxy_thresholds": {
            "passed": all(latent_checks.values()),
            "checks": latent_checks,
            "original_blind_and_shuffled_confirmation_measured": False,
        },
        "localizations": localizations,
        "claim_boundary": [
            "changed-objective exploratory development comparison",
            "the optimized identification proxy cannot establish untaken-action causality",
            "no fresh blind or shuffled confirmation role is present",
            "no planning, navigation, promotion, deployment, or production claim",
        ],
    }


__all__ = [
    "ARM_NAMES",
    "AlignmentSuccessorMetricError",
    "BOOTSTRAP_LOWER_INDEX",
    "BOOTSTRAP_MEDIAN_INDEX",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_UPPER_INDEX",
    "MEANINGFUL_POINT_THRESHOLD",
    "PAIRED_BOOTSTRAP_SEED",
    "RANK_PASS_COUNT",
    "RANK_RATIO_MINIMUM",
    "SCHEMA",
    "STALL_UPPER_THRESHOLD",
    "TAIL_UPDATES",
    "V3_ALIGNMENT_POINT_DEFICIT",
    "classify_alignment_outcome",
    "decide_alignment_successor",
    "paired_minimum_action_margin_delta",
]
