"""Read-only action-level localization for the completed three-arm V3 run.

This module performs no filesystem access.  It consumes the metric vectors
already frozen inside the conditioned update-700 snapshot plus validation-row
metadata and expands the deliberately terse public receipt into per-action
diagnostics.  Positive log-energy advantage means the learned predictor has
lower energy than the named control; positive action margin means the factual
candidate has lower energy than every wrong candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Mapping, Sequence

import numpy as np

from lewm.benchmarks import go2_world_model_existing_pool_three_arm_v1 as three_arm


SCHEMA = "lewm_go2_world_model_v3_action_localization_v1"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_LOWER_INDEX = 500
BOOTSTRAP_MEDIAN_INDEX = 5_000
BOOTSTRAP_UPPER_INDEX = BOOTSTRAP_REPLICATES - 1 - BOOTSTRAP_LOWER_INDEX
MARGIN_BOOTSTRAP_SEED = three_arm.ACTION_IDENTIFICATION_BOOTSTRAP_SEED
PERSISTENCE_BOOTSTRAP_SEED = 20_260_807
WRONG_HISTORY_BOOTSTRAP_SEED = 20_260_808
BOOTSTRAP_ALGORITHM = (
    "python_random_mt19937_getrandbits52_open01_neg_log1p_"
    "shared_family_scene_weights_per_action_v1"
)
BOOTSTRAP_INTERPRETATION = (
    "deterministic_positive_weight_scene_cluster_bayesian_quantiles_"
    "not_frequentist_coverage"
)
TRAIN_EXPOSURE_COUNTS = (2_959, 1_197, 1_075, 545, 4_303, 447, 767, 2_893, 1_814)
TRAIN_ROW_COUNT = sum(TRAIN_EXPOSURE_COUNTS)
ALIGNMENT_EXPOSURE_WEIGHT_THRESHOLD = 2.0
PERSISTENCE_SYSTEMIC_FAILURE_COUNT = 5


class ActionLocalizationError(ValueError):
    """Raised when the frozen localization inputs violate their contract."""


@dataclass(frozen=True)
class _Metadata:
    actions: np.ndarray
    scenes: tuple[str, ...]
    families: tuple[str, ...]
    scene_family: Mapping[str, str]


def _finite_positive_vector(value: Any, *, name: str, count: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ActionLocalizationError(f"{name} must be numeric") from error
    if (
        result.shape != (count,)
        or not np.isfinite(result).all()
        or bool((result <= 0.0).any())
    ):
        raise ActionLocalizationError(
            f"{name} must be a finite positive vector of length {count}"
        )
    return result


def _candidate_matrix(value: Any, *, count: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ActionLocalizationError("candidate energies must be numeric") from error
    if (
        result.shape != (count, three_arm.ACTION_COUNT)
        or not np.isfinite(result).all()
        or bool((result < 0.0).any())
    ):
        raise ActionLocalizationError(
            "candidate energies must be finite, nonnegative, and shaped (N,9)"
        )
    return result


def _metadata(rows: Sequence[Any]) -> _Metadata:
    normalized = three_arm.normalize_h6_metadata_rows(rows)
    if any(row.role != "val" for row in normalized):
        raise ActionLocalizationError("localization metadata must be validation-only")
    if tuple(int(row.index) for row in normalized) != tuple(range(len(normalized))):
        raise ActionLocalizationError("validation row order or integer identity changed")
    actions = np.asarray(
        [row.candidate_action_id for row in normalized], dtype=np.int64
    )
    scenes = tuple(row.scene_id for row in normalized)
    families = tuple(row.family for row in normalized)
    scene_family: dict[str, str] = {}
    for scene, family in zip(scenes, families, strict=True):
        previous = scene_family.setdefault(scene, family)
        if previous != family:
            raise ActionLocalizationError("a validation scene crossed families")
    if set(families) != set(three_arm.REGISTERED_FAMILIES):
        raise ActionLocalizationError("validation metadata lost a registered family")
    if set(int(value) for value in actions) != set(range(three_arm.ACTION_COUNT)):
        raise ActionLocalizationError("validation metadata lost a requested action")
    return _Metadata(actions, scenes, families, scene_family)


def _positive_exponential_weight(rng: random.Random) -> float:
    bits = rng.getrandbits(52)
    uniform_open = (bits + 1) / (2**52 + 1)
    weight = -math.log1p(-uniform_open)
    if not math.isfinite(weight) or weight <= 0.0:
        raise AssertionError("strictly positive bootstrap weight construction failed")
    return weight


def _action_cluster_summary(
    values: np.ndarray,
    metadata: _Metadata,
    *,
    seed: int,
) -> dict[str, Any]:
    """Family-equal, scene-equal per-action points and frozen quantiles."""

    if values.shape != metadata.actions.shape or not np.isfinite(values).all():
        raise ActionLocalizationError("per-action values are invalid")
    family_scenes = {
        family: tuple(
            scene
            for scene in sorted(metadata.scene_family)
            if metadata.scene_family[scene] == family
        )
        for family in three_arm.REGISTERED_FAMILIES
    }
    cell: dict[tuple[str, int], float] = {}
    row_counts: dict[int, int] = {}
    for action in range(three_arm.ACTION_COUNT):
        row_counts[action] = int((metadata.actions == action).sum())
        for scene in sorted(metadata.scene_family):
            indices = [
                index
                for index, (row_scene, row_action) in enumerate(
                    zip(metadata.scenes, metadata.actions, strict=True)
                )
                if row_scene == scene and int(row_action) == action
            ]
            if indices:
                cell[(scene, action)] = float(values[indices].mean())

    by_family: dict[int, dict[str, float]] = {}
    supporting_scenes: dict[int, dict[str, int]] = {}
    points: dict[int, float] = {}
    for action in range(three_arm.ACTION_COUNT):
        by_family[action] = {}
        supporting_scenes[action] = {}
        for family in three_arm.REGISTERED_FAMILIES:
            selected = [
                cell[(scene, action)]
                for scene in family_scenes[family]
                if (scene, action) in cell
            ]
            if not selected:
                raise ActionLocalizationError(
                    "every family/action cell requires validation support"
                )
            by_family[action][family] = math.fsum(selected) / len(selected)
            supporting_scenes[action][family] = len(selected)
        points[action] = (
            math.fsum(by_family[action].values())
            / len(three_arm.REGISTERED_FAMILIES)
        )

    rng = random.Random(seed)
    draws = {action: [] for action in range(three_arm.ACTION_COUNT)}
    shared_minimum_draws: list[float] = []
    for _replicate in range(BOOTSTRAP_REPLICATES):
        weights = {
            family: {
                scene: _positive_exponential_weight(rng)
                for scene in family_scenes[family]
            }
            for family in three_arm.REGISTERED_FAMILIES
        }
        replicate_values = []
        for action in range(three_arm.ACTION_COUNT):
            family_values = []
            for family in three_arm.REGISTERED_FAMILIES:
                selected = [
                    (scene, weights[family][scene])
                    for scene in family_scenes[family]
                    if (scene, action) in cell
                ]
                denominator = math.fsum(weight for _scene, weight in selected)
                family_values.append(
                    math.fsum(
                        weight * cell[(scene, action)]
                        for scene, weight in selected
                    )
                    / denominator
                )
            value = math.fsum(family_values) / len(three_arm.REGISTERED_FAMILIES)
            draws[action].append(value)
            replicate_values.append(value)
        shared_minimum_draws.append(min(replicate_values))

    per_action = []
    for action, name in enumerate(three_arm.ACTION_VOCABULARY):
        ordered = sorted(draws[action])
        per_action.append(
            {
                "action_id": action,
                "action_name": name,
                "row_count": row_counts[action],
                "family_equal_scene_macro_point": points[action],
                "one_sided_95_lower_quantile": ordered[BOOTSTRAP_LOWER_INDEX],
                "median_quantile": ordered[BOOTSTRAP_MEDIAN_INDEX],
                "one_sided_95_upper_quantile": ordered[BOOTSTRAP_UPPER_INDEX],
                "positive_family_count": sum(
                    value > 0.0 for value in by_family[action].values()
                ),
                "minimum_supporting_scene_count": min(
                    supporting_scenes[action].values()
                ),
                "total_supporting_scene_count": sum(
                    supporting_scenes[action].values()
                ),
                "supporting_scene_count_by_family": supporting_scenes[action],
                "point_by_family": by_family[action],
            }
        )
    ordered_minimum = sorted(shared_minimum_draws)
    return {
        "bootstrap_seed": seed,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_lower_index": BOOTSTRAP_LOWER_INDEX,
        "bootstrap_median_index": BOOTSTRAP_MEDIAN_INDEX,
        "bootstrap_upper_index": BOOTSTRAP_UPPER_INDEX,
        "bootstrap_algorithm": BOOTSTRAP_ALGORITHM,
        "bootstrap_interpretation": BOOTSTRAP_INTERPRETATION,
        "shared_across_action_minimum_quantiles": {
            "one_sided_95_lower_quantile": ordered_minimum[
                BOOTSTRAP_LOWER_INDEX
            ],
            "median_quantile": ordered_minimum[BOOTSTRAP_MEDIAN_INDEX],
            "one_sided_95_upper_quantile": ordered_minimum[
                BOOTSTRAP_UPPER_INDEX
            ],
        },
        "per_action": per_action,
    }


def _routing_decision(
    margin_rows: Sequence[Mapping[str, Any]],
    persistence_rows: Sequence[Mapping[str, Any]],
    *,
    alignment_shared_minimum_lower: float,
    aggregate_persistence_lower: float,
) -> dict[str, Any]:
    """Apply the frozen post-localization successor routing without authority."""

    alignment_point_failures = [
        int(row["action_id"])
        for row in margin_rows
        if float(row["family_equal_scene_macro_point"]) <= 0.0
    ]
    alignment_lower_failures = [
        int(row["action_id"])
        for row in margin_rows
        if float(row["one_sided_95_lower_quantile"]) <= 0.0
    ]
    if not alignment_point_failures:
        if alignment_lower_failures or alignment_shared_minimum_lower <= 0.0:
            alignment_route = "UNCERTAINTY_LIMITED"
            alignment_successor = (
                "SIZE_LARGER_EXISTING_POOL_SCENE_DISJOINT_EVALUATION"
            )
        else:
            alignment_route = "ALIGNMENT_PASSED"
            alignment_successor = "NO_ALIGNMENT_SUCCESSOR"
    elif (
        len(alignment_point_failures) <= 2
        and all(
            TRAIN_ROW_COUNT
            / (three_arm.ACTION_COUNT * TRAIN_EXPOSURE_COUNTS[action])
            > ALIGNMENT_EXPOSURE_WEIGHT_THRESHOLD
            for action in alignment_point_failures
        )
    ):
        alignment_route = "TEST_ACTION_REWEIGHTING_HYPOTHESIS"
        alignment_successor = "ACTION_BALANCED_FACTUAL_LOSS_VS_MATCHED_BASELINE"
    else:
        alignment_route = "TEST_GLOBAL_ALIGNMENT_HYPOTHESIS"
        alignment_successor = "EXPLICIT_ACTION_ALIGNMENT_OBJECTIVE_VS_MATCHED_BASELINE"

    persistence_lower_failures = [
        int(row["action_id"])
        for row in persistence_rows
        if float(row["one_sided_95_lower_quantile"]) <= 0.0
    ]
    alignment_repaired = (
        not alignment_point_failures
        and not alignment_lower_failures
        and alignment_shared_minimum_lower > 0.0
    )
    if not alignment_repaired:
        persistence_route = "DEFERRED_UNTIL_ALIGNMENT_REPAIRED"
        persistence_successor = "DEFERRED"
    elif (
        len(persistence_lower_failures) >= PERSISTENCE_SYSTEMIC_FAILURE_COUNT
    ):
        persistence_route = "PERSISTENCE_SYSTEMIC"
        persistence_successor = "PERSISTENCE_RESIDUAL_VS_MATCHED_BASELINE"
    elif persistence_lower_failures or aggregate_persistence_lower <= 0.0:
        persistence_route = "PERSISTENCE_LOCALIZED_OR_AGGREGATE_UNREPAIRED"
        persistence_successor = "PLANNING_USEFULNESS_GATE_WITH_PROXY_CAVEAT"
    else:
        persistence_route = "PERSISTENCE_PASSED"
        persistence_successor = "PROCEED_TO_PLANNING_USEFULNESS_GATE"

    selected_next_step = (
        alignment_successor if not alignment_repaired else persistence_successor
    )

    return {
        "alignment_route": alignment_route,
        "selected_alignment_next_step": alignment_successor,
        "selected_persistence_next_step": persistence_successor,
        "selected_next_step": selected_next_step,
        "alignment_repaired": alignment_repaired,
        "alignment_point_failure_action_ids": alignment_point_failures,
        "alignment_lower_failure_action_ids": alignment_lower_failures,
        "alignment_shared_minimum_lower_quantile": (
            alignment_shared_minimum_lower
        ),
        "exposure_weight_threshold_strictly_above": (
            ALIGNMENT_EXPOSURE_WEIGHT_THRESHOLD
        ),
        "failing_action_inverse_uniform_train_weights": {
            str(action): TRAIN_ROW_COUNT
            / (three_arm.ACTION_COUNT * TRAIN_EXPOSURE_COUNTS[action])
            for action in alignment_point_failures
        },
        "train_count_evidence_kind": (
            "unique_frozen_train_row_candidate_action_counts_not_scheduled_"
            "presentation_counts"
        ),
        "provenance_route": "UNAVAILABLE_WITHIN_BOUND_DIAGNOSTIC",
        "complete_executed_command_join_available_to_this_diagnostic": False,
        "persistence_route": persistence_route,
        "persistence_systemic_failure_count_threshold": (
            PERSISTENCE_SYSTEMIC_FAILURE_COUNT
        ),
        "persistence_lower_failure_action_ids": persistence_lower_failures,
        "aggregate_persistence_lower_bound": aggregate_persistence_lower,
        "authorizes_successor_execution": False,
    }


def _descriptive_action_points(
    values: np.ndarray,
    metadata: _Metadata,
) -> tuple[float, ...]:
    """Return scene-then-family equal points for each factual action."""

    if values.shape != metadata.actions.shape or not np.isfinite(values).all():
        raise ActionLocalizationError("descriptive per-action values are invalid")
    result = []
    for action in range(three_arm.ACTION_COUNT):
        family_values = []
        for family in three_arm.REGISTERED_FAMILIES:
            scene_values = []
            for scene in sorted(metadata.scene_family):
                if metadata.scene_family[scene] != family:
                    continue
                indices = [
                    index
                    for index, (row_scene, row_action) in enumerate(
                        zip(metadata.scenes, metadata.actions, strict=True)
                    )
                    if row_scene == scene and int(row_action) == action
                ]
                if indices:
                    scene_values.append(float(values[indices].mean()))
            if not scene_values:
                raise ActionLocalizationError(
                    "every family/action cell requires validation support"
                )
            family_values.append(math.fsum(scene_values) / len(scene_values))
        result.append(math.fsum(family_values) / len(family_values))
    return tuple(result)


def _aggregate_control_receipt(
    comparison: three_arm.PairedLogEnergyComparison,
) -> dict[str, Any]:
    """Retain registered aggregate/family evidence without scene identifiers."""

    return {
        "control_name": comparison.control_name,
        "row_count": comparison.row_count,
        "scene_count": comparison.scene_count,
        "family_count": comparison.family_count,
        "bootstrap_replicates": comparison.bootstrap_replicates,
        "bootstrap_seed": comparison.bootstrap_seed,
        "bootstrap_lower_index": comparison.bootstrap_lower_index,
        "macro_log_advantage": comparison.macro_log_advantage,
        "bootstrap_lower_95": comparison.bootstrap_lower_95,
        "positive_family_count": comparison.positive_family_count,
        "log_advantage_by_family": dict(comparison.log_advantage_by_family),
    }


def localize_action_and_controls(
    *,
    candidate_energies: Any,
    factual_energy: Any,
    persistence_energy: Any,
    wrong_history_energy: Any,
    validation_rows: Sequence[Any],
) -> dict[str, Any]:
    """Expand the frozen V3 u700 vectors into action-level diagnostics."""

    metadata = _metadata(validation_rows)
    count = len(metadata.actions)
    candidate = _candidate_matrix(candidate_energies, count=count)
    factual = _finite_positive_vector(factual_energy, name="factual energy", count=count)
    persistence = _finite_positive_vector(
        persistence_energy, name="persistence energy", count=count
    )
    wrong_history = _finite_positive_vector(
        wrong_history_energy, name="wrong-history energy", count=count
    )
    gathered = candidate[np.arange(count), metadata.actions]
    maximum_factual_mismatch = float(np.max(np.abs(gathered - factual)))
    if maximum_factual_mismatch != 0.0:
        raise ActionLocalizationError(
            "snapshot factual energy is not the factual candidate-energy column"
        )

    wrong = candidate.copy()
    wrong[np.arange(count), metadata.actions] = math.inf
    action_margin = wrong.min(axis=1) - factual
    persistence_advantage = np.log(persistence) - np.log(factual)
    wrong_history_advantage = np.log(wrong_history) - np.log(factual)

    candidate_ids = np.arange(three_arm.ACTION_COUNT, dtype=np.int64)
    factual_rank = np.empty(count, dtype=np.int64)
    for index, action in enumerate(metadata.actions):
        factual_value = candidate[index, int(action)]
        factual_rank[index] = 1 + int((candidate[index] < factual_value).sum()) + int(
            np.logical_and(
                candidate[index] == factual_value,
                candidate_ids < int(action),
            ).sum()
        )
    candidate_spread = candidate.max(axis=1) - candidate.min(axis=1)
    factual_energy_points = _descriptive_action_points(factual, metadata)
    candidate_spread_points = _descriptive_action_points(candidate_spread, metadata)
    pairwise_points: list[list[float]] = []
    for competitor in range(three_arm.ACTION_COUNT):
        pairwise_points.append(
            list(_descriptive_action_points(candidate[:, competitor] - factual, metadata))
        )
    pairwise_matrix = [
        [pairwise_points[competitor][action] for competitor in range(three_arm.ACTION_COUNT)]
        for action in range(three_arm.ACTION_COUNT)
    ]

    identification = three_arm.summarize_nine_way_action_identification(
        candidate,
        metadata.actions.tolist(),
        metadata.scenes,
        metadata.families,
    )
    identification_payload = identification.to_dict()
    minimum = candidate.min(axis=1)
    unique_winner = np.equal(candidate, minimum[:, None]).sum(axis=1) == 1
    predicted = np.argmin(candidate, axis=1)
    identification_payload["unique_winner_correct_count"] = int(
        np.logical_and(unique_winner, predicted == metadata.actions).sum()
    )
    persistence_aggregate = three_arm.paired_log_energy_comparison(
        factual,
        persistence,
        metadata.scenes,
        metadata.families,
        control_name="persistence",
    )
    wrong_history_aggregate = three_arm.paired_log_energy_comparison(
        factual,
        wrong_history,
        metadata.scenes,
        metadata.families,
        control_name="wrong_history",
    )
    margin = _action_cluster_summary(
        action_margin, metadata, seed=MARGIN_BOOTSTRAP_SEED
    )
    persistence_payload = _action_cluster_summary(
        persistence_advantage, metadata, seed=PERSISTENCE_BOOTSTRAP_SEED
    )
    wrong_history_payload = _action_cluster_summary(
        wrong_history_advantage, metadata, seed=WRONG_HISTORY_BOOTSTRAP_SEED
    )

    summary_margin = tuple(identification.scene_family_margin_by_action)
    localized_margin = tuple(
        item["family_equal_scene_macro_point"] for item in margin["per_action"]
    )
    if any(
        not math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-15)
        for left, right in zip(summary_margin, localized_margin, strict=True)
    ):
        raise ActionLocalizationError("independent per-action margin points disagree")
    localized_shared_lower = margin["shared_across_action_minimum_quantiles"][
        "one_sided_95_lower_quantile"
    ]
    if not math.isclose(
        localized_shared_lower,
        identification.hardest_margin_bootstrap_lower_95,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ActionLocalizationError(
            "shared per-action bootstrap does not reproduce the registered "
            "hardest-margin lower quantile"
        )

    alignment_point_failures = [
        item["action_id"]
        for item in margin["per_action"]
        if item["family_equal_scene_macro_point"] <= 0.0
    ]
    alignment_lower_failures = [
        item["action_id"]
        for item in margin["per_action"]
        if item["one_sided_95_lower_quantile"] <= 0.0
    ]
    persistence_point_failures = [
        item["action_id"]
        for item in persistence_payload["per_action"]
        if item["family_equal_scene_macro_point"] <= 0.0
    ]
    persistence_lower_failures = [
        item["action_id"]
        for item in persistence_payload["per_action"]
        if item["one_sided_95_lower_quantile"] <= 0.0
    ]
    action_names = three_arm.ACTION_VOCABULARY
    action_diagnostics = []
    for action, action_name in enumerate(action_names):
        selected_ranks = factual_rank[metadata.actions == action]
        histogram = [int((selected_ranks == rank).sum()) for rank in range(1, 10)]
        competitors = [value for value in range(three_arm.ACTION_COUNT) if value != action]
        minimum_pairwise = min(
            competitors,
            key=lambda competitor: (pairwise_matrix[action][competitor], competitor),
        )
        exposure = TRAIN_EXPOSURE_COUNTS[action]
        action_diagnostics.append(
            {
                "action_id": action,
                "action_name": action_name,
                "train_exposure_count": exposure,
                "inverse_uniform_train_weight": TRAIN_ROW_COUNT
                / (three_arm.ACTION_COUNT * exposure),
                "validation_row_count": int((metadata.actions == action).sum()),
                "factual_rank_histogram_rank_1_through_9": histogram,
                "row_weighted_factual_mean_reciprocal_rank": float(
                    np.mean(1.0 / selected_ranks.astype(np.float64))
                ),
                "family_equal_scene_macro_factual_energy": factual_energy_points[action],
                "family_equal_scene_macro_candidate_spread": candidate_spread_points[action],
                "minimum_pairwise_macro_competitor_id": minimum_pairwise,
                "minimum_pairwise_macro_competitor_name": action_names[
                    minimum_pairwise
                ],
                "minimum_pairwise_macro_margin": pairwise_matrix[action][
                    minimum_pairwise
                ],
            }
        )
    routing = _routing_decision(
        margin["per_action"],
        persistence_payload["per_action"],
        alignment_shared_minimum_lower=localized_shared_lower,
        aggregate_persistence_lower=persistence_aggregate.bootstrap_lower_95,
    )
    return {
        "schema": SCHEMA,
        "status": "PASS_READ_ONLY_LOCALIZATION",
        "row_count": count,
        "scene_count": len(metadata.scene_family),
        "family_count": len(three_arm.REGISTERED_FAMILIES),
        "action_count": three_arm.ACTION_COUNT,
        "factual_candidate_energy_max_abs_error": maximum_factual_mismatch,
        "action_identification": identification_payload,
        "registered_control_reproduction": {
            "persistence": _aggregate_control_receipt(persistence_aggregate),
            "wrong_history": _aggregate_control_receipt(wrong_history_aggregate),
        },
        "action_diagnostics": action_diagnostics,
        "pairwise_family_equal_scene_macro_margin_matrix": {
            "row_definition": "factual_action",
            "column_definition": "candidate_action",
            "value_definition": "candidate_energy_minus_factual_energy",
            "action_vocabulary": list(action_names),
            "values": pairwise_matrix,
        },
        "action_margin_localization": margin,
        "persistence_localization": persistence_payload,
        "wrong_history_localization": wrong_history_payload,
        "routing_decision": routing,
        "failure_topology": {
            "registered_hardest_action_id": int(identification.hardest_action_id),
            "registered_hardest_action_name": action_names[
                identification.hardest_action_id
            ],
            "alignment_point_failure_action_ids": alignment_point_failures,
            "alignment_point_failure_action_names": [
                action_names[action] for action in alignment_point_failures
            ],
            "alignment_lower_failure_action_ids": alignment_lower_failures,
            "alignment_lower_failure_action_names": [
                action_names[action] for action in alignment_lower_failures
            ],
            "persistence_point_failure_action_ids": persistence_point_failures,
            "persistence_point_failure_action_names": [
                action_names[action] for action in persistence_point_failures
            ],
            "persistence_lower_failure_action_ids": persistence_lower_failures,
            "persistence_lower_failure_action_names": [
                action_names[action] for action in persistence_lower_failures
            ],
            "alignment_point_failure_scope": (
                "none"
                if not alignment_point_failures
                else "localized" if len(alignment_point_failures) <= 2 else "broad"
            ),
            "persistence_point_failure_scope": (
                "none"
                if not persistence_point_failures
                else "localized" if len(persistence_point_failures) <= 2 else "broad"
            ),
        },
    }


__all__ = [
    "ActionLocalizationError",
    "ALIGNMENT_EXPOSURE_WEIGHT_THRESHOLD",
    "BOOTSTRAP_ALGORITHM",
    "BOOTSTRAP_INTERPRETATION",
    "BOOTSTRAP_LOWER_INDEX",
    "BOOTSTRAP_MEDIAN_INDEX",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_UPPER_INDEX",
    "MARGIN_BOOTSTRAP_SEED",
    "PERSISTENCE_BOOTSTRAP_SEED",
    "PERSISTENCE_SYSTEMIC_FAILURE_COUNT",
    "SCHEMA",
    "TRAIN_EXPOSURE_COUNTS",
    "TRAIN_ROW_COUNT",
    "WRONG_HISTORY_BOOTSTRAP_SEED",
    "localize_action_and_controls",
]
