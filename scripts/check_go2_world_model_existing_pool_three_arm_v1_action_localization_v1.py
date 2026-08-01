#!/usr/bin/env python3
"""Receipt-only checker for the one-shot V3 action localization.

The checker never opens the snapshot, validation index, pack, RGB, checkpoint,
or any row-level payload.  It validates the immutable aggregate result, its
public anchors, internal arithmetic, source/authority chain, and custody
receipt, then emits one small checker receipt in the fresh localization root.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import (  # noqa: E402
    go2_world_model_existing_pool_three_arm_v1 as three_arm,
)
from lewm.benchmarks import (  # noqa: E402
    go2_world_model_v3_action_localization_v1 as localization_metrics,
)
from scripts import (  # noqa: E402
    extract_go2_world_model_existing_pool_three_arm_v1_action_localization_v1 as worker,
)


CHECK_SCHEMA = (
    "lewm_go2_world_model_existing_pool_three_arm_v1_action_localization_v1_"
    "receipt_check_v1"
)
FORBIDDEN_RESULT_KEYS = {
    "metric_vectors",
    "arm_state_dict",
    "optimizer_state_dict",
    "prediction_tokens",
    "target_tokens",
    "training_factual_energy",
    "validation_candidate_energy",
    "validation_factual_energy",
    "validation_persistence_energy",
    "validation_wrong_history_energy",
    "scene_id",
    "rgb",
}


class LocalizationCheckError(RuntimeError):
    """The aggregate localization receipt is invalid."""


def _all_keys(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if type(key) is str:
                result.add(key)
            result.update(_all_keys(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_all_keys(child))
    return result


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LocalizationCheckError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise LocalizationCheckError(f"{label} is nonfinite")
    return result


def _exact_public_anchors(result: Mapping[str, Any]) -> None:
    anchors = result.get("public_anchor_reproduction")
    if anchors != dict(sorted(worker.PUBLIC_ANCHORS.items())):
        raise LocalizationCheckError("public-anchor reproduction changed")
    localization = result["localization"]
    action = localization["action_identification"]
    controls = localization["registered_control_reproduction"]
    observed = {
        "balanced_accuracy": action["scene_family_balanced_accuracy"],
        "balanced_accuracy_one_sided_95_lower_bound": action[
            "balanced_accuracy_bootstrap_lower_95"
        ],
        "hardest_wrong_action_margin": action["hardest_action_margin"],
        "hardest_wrong_action_margin_one_sided_95_lower_bound": action[
            "hardest_margin_bootstrap_lower_95"
        ],
        "persistence_log_energy_advantage": controls["persistence"][
            "macro_log_advantage"
        ],
        "persistence_one_sided_95_lower_bound": controls["persistence"][
            "bootstrap_lower_95"
        ],
        "wrong_history_log_energy_advantage": controls["wrong_history"][
            "macro_log_advantage"
        ],
        "wrong_history_one_sided_95_lower_bound": controls["wrong_history"][
            "bootstrap_lower_95"
        ],
    }
    if set(observed) != set(worker.PUBLIC_ANCHORS) or any(
        not math.isclose(
            _finite(observed[name], label=name),
            expected,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        for name, expected in worker.PUBLIC_ANCHORS.items()
    ):
        raise LocalizationCheckError("aggregate public anchors disagree")


def _check_localization(
    localization: Any,
    *,
    expected_rows: int = 2_048,
    expected_scenes: int = 150,
) -> None:
    expected_localization_keys = {
        "schema",
        "status",
        "row_count",
        "scene_count",
        "family_count",
        "action_count",
        "factual_candidate_energy_max_abs_error",
        "action_identification",
        "registered_control_reproduction",
        "action_diagnostics",
        "pairwise_family_equal_scene_macro_margin_matrix",
        "action_margin_localization",
        "persistence_localization",
        "wrong_history_localization",
        "routing_decision",
        "failure_topology",
    }
    if (
        not isinstance(localization, dict)
        or set(localization) != expected_localization_keys
        or localization.get("schema") != localization_metrics.SCHEMA
    ):
        raise LocalizationCheckError("localization schema changed")
    if (
        localization.get("status") != "PASS_READ_ONLY_LOCALIZATION"
        or localization.get("row_count") != expected_rows
        or localization.get("scene_count") != expected_scenes
        or localization.get("family_count") != 8
        or localization.get("action_count") != 9
        or localization.get("factual_candidate_energy_max_abs_error") != 0.0
    ):
        raise LocalizationCheckError("localization population contract changed")

    action = localization["action_identification"]
    expected_action_keys = {
        "schema",
        "status",
        "row_count",
        "scene_count",
        "family_count",
        "action_count",
        "prediction_rule",
        "margin_definition",
        "favorable_direction",
        "bootstrap_replicates",
        "bootstrap_seed",
        "bootstrap_lower_index",
        "bootstrap_algorithm",
        "bootstrap_interpretation",
        "family_action_supporting_scene_counts",
        "minimum_family_action_supporting_scene_count",
        "confusion_matrix",
        "factual_action_counts",
        "predicted_action_counts",
        "row_weighted_accuracy",
        "row_weighted_per_action_recall",
        "row_weighted_balanced_accuracy",
        "scene_family_per_action_recall",
        "scene_family_balanced_accuracy",
        "balanced_accuracy_bootstrap_lower_95",
        "scene_family_margin_by_action",
        "hardest_action_id",
        "hardest_action_margin",
        "hardest_margin_bootstrap_lower_95",
        "exact_tie_row_count",
        "exact_tie_rate",
        "unique_winner_count",
        "unique_winner_correct_count",
        "unique_winner_accuracy",
    }
    if (
        not isinstance(action, dict)
        or set(action) != expected_action_keys
        or action["schema"] != three_arm.ACTION_IDENTIFICATION_SCHEMA
        or action["status"] != "PASS"
        or action["row_count"] != expected_rows
        or action["scene_count"] != expected_scenes
        or action["family_count"] != 8
        or action["action_count"] != 9
        or action["prediction_rule"] != "lowest_action_id_argmin_exact_ties"
        or action["margin_definition"]
        != "minimum_wrong_action_energy_minus_factual_action_energy"
        or action["favorable_direction"] != "positive"
        or action["bootstrap_replicates"] != three_arm.BOOTSTRAP_REPLICATES
        or action["bootstrap_seed"]
        != three_arm.ACTION_IDENTIFICATION_BOOTSTRAP_SEED
        or action["bootstrap_lower_index"]
        != localization_metrics.BOOTSTRAP_LOWER_INDEX
        or action["bootstrap_algorithm"]
        != three_arm.ACTION_IDENTIFICATION_BOOTSTRAP_ALGORITHM
        or action["bootstrap_interpretation"]
        != three_arm.ACTION_IDENTIFICATION_BOOTSTRAP_INTERPRETATION
    ):
        raise LocalizationCheckError("registered action summary contract changed")
    confusion = action["confusion_matrix"]
    if (
        not isinstance(confusion, list)
        or len(confusion) != 9
        or any(not isinstance(row, list) or len(row) != 9 for row in confusion)
        or any(type(value) is not int or value < 0 for row in confusion for value in row)
    ):
        raise LocalizationCheckError("confusion matrix is invalid")
    factual_counts = [sum(row) for row in confusion]
    predicted_counts = [sum(confusion[row][column] for row in range(9)) for column in range(9)]
    if (
        sum(factual_counts) != expected_rows
        or action["factual_action_counts"] != factual_counts
        or action["predicted_action_counts"] != predicted_counts
    ):
        raise LocalizationCheckError("confusion accounting changed")
    row_recalls = action["row_weighted_per_action_recall"]
    expected_row_recalls = [
        confusion[index][index] / factual_counts[index] for index in range(9)
    ]
    if (
        not isinstance(row_recalls, list)
        or len(row_recalls) != 9
        or any(
            not math.isclose(
                _finite(observed, label="row recall"),
                expected,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            for observed, expected in zip(row_recalls, expected_row_recalls, strict=True)
        )
        or not math.isclose(
            _finite(action["row_weighted_accuracy"], label="row accuracy"),
            sum(confusion[index][index] for index in range(9)) / expected_rows,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or not math.isclose(
            _finite(
                action["row_weighted_balanced_accuracy"],
                label="row balanced accuracy",
            ),
            math.fsum(expected_row_recalls) / 9,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    ):
        raise LocalizationCheckError("row-weighted action arithmetic changed")
    recalls = action["scene_family_per_action_recall"]
    margins = action["scene_family_margin_by_action"]
    if not isinstance(recalls, list) or len(recalls) != 9 or not isinstance(margins, list) or len(margins) != 9:
        raise LocalizationCheckError("per-action registered summaries changed")
    if not math.isclose(
        math.fsum(_finite(value, label="recall") for value in recalls) / 9,
        _finite(action["scene_family_balanced_accuracy"], label="balanced accuracy"),
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise LocalizationCheckError("balanced accuracy arithmetic changed")
    support = action["family_action_supporting_scene_counts"]
    if (
        not isinstance(support, dict)
        or set(support) != set(three_arm.REGISTERED_FAMILIES)
        or any(
            not isinstance(counts, list)
            or len(counts) != 9
            or any(type(value) is not int or value < 2 for value in counts)
            for counts in support.values()
        )
        or action["minimum_family_action_supporting_scene_count"]
        != min(min(counts) for counts in support.values())
    ):
        raise LocalizationCheckError("registered family/action support changed")
    hardest = min(range(9), key=lambda index: (_finite(margins[index], label="margin"), index))
    if (
        action["hardest_action_id"] != hardest
        or not math.isclose(
            _finite(action["hardest_action_margin"], label="hardest margin"),
            _finite(margins[hardest], label="hardest per-action margin"),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    ):
        raise LocalizationCheckError("hardest-action arithmetic changed")

    if (
        type(action["exact_tie_row_count"]) is not int
        or not 0 <= action["exact_tie_row_count"] <= expected_rows
        or not math.isclose(
            _finite(action["exact_tie_rate"], label="exact tie rate"),
            action["exact_tie_row_count"] / expected_rows,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or type(action["unique_winner_count"]) is not int
        or action["unique_winner_count"]
        != expected_rows - action["exact_tie_row_count"]
        or type(action["unique_winner_correct_count"]) is not int
        or not 0
        <= action["unique_winner_correct_count"]
        <= action["unique_winner_count"]
        or not math.isclose(
            _finite(action["unique_winner_accuracy"], label="unique winner accuracy"),
            (
                action["unique_winner_correct_count"] / action["unique_winner_count"]
                if action["unique_winner_count"]
                else 0.0
            ),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    ):
        raise LocalizationCheckError("tie accounting changed")

    controls = localization["registered_control_reproduction"]
    if not isinstance(controls, dict) or set(controls) != {"persistence", "wrong_history"}:
        raise LocalizationCheckError("control reproduction fields changed")
    expected_control_keys = {
        "control_name",
        "row_count",
        "scene_count",
        "family_count",
        "bootstrap_replicates",
        "bootstrap_seed",
        "bootstrap_lower_index",
        "macro_log_advantage",
        "bootstrap_lower_95",
        "positive_family_count",
        "log_advantage_by_family",
    }
    for control_name, control in controls.items():
        if (
            not isinstance(control, dict)
            or set(control) != expected_control_keys
            or control["control_name"] != control_name
            or control["row_count"] != expected_rows
            or control["scene_count"] != expected_scenes
            or control["family_count"] != 8
            or control["bootstrap_replicates"] != three_arm.BOOTSTRAP_REPLICATES
            or control["bootstrap_seed"]
            != three_arm.CONTROL_BOOTSTRAP_SEEDS[control_name]
            or control["bootstrap_lower_index"]
            != localization_metrics.BOOTSTRAP_LOWER_INDEX
            or not isinstance(control["log_advantage_by_family"], dict)
            or set(control["log_advantage_by_family"])
            != set(three_arm.REGISTERED_FAMILIES)
        ):
            raise LocalizationCheckError(f"{control_name} aggregate contract changed")
        family_values = [
            _finite(value, label=f"{control_name} family advantage")
            for value in control["log_advantage_by_family"].values()
        ]
        if (
            control["positive_family_count"]
            != sum(value > 0.0 for value in family_values)
            or not math.isclose(
                _finite(
                    control["macro_log_advantage"],
                    label=f"{control_name} macro advantage",
                ),
                math.fsum(family_values) / 8,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
        ):
            raise LocalizationCheckError(f"{control_name} aggregate arithmetic changed")
        _finite(control["bootstrap_lower_95"], label=f"{control_name} lower")

    diagnostics = localization["action_diagnostics"]
    if not isinstance(diagnostics, list) or len(diagnostics) != 9:
        raise LocalizationCheckError("action diagnostics are incomplete")
    pairwise = localization["pairwise_family_equal_scene_macro_margin_matrix"]
    if not isinstance(pairwise, dict) or set(pairwise) != {
        "row_definition",
        "column_definition",
        "value_definition",
        "action_vocabulary",
        "values",
    }:
        raise LocalizationCheckError("pairwise margin envelope changed")
    matrix = pairwise["values"]
    if (
        pairwise["row_definition"] != "factual_action"
        or pairwise["column_definition"] != "candidate_action"
        or pairwise["value_definition"] != "candidate_energy_minus_factual_energy"
        or
        pairwise["action_vocabulary"] != list(three_arm.ACTION_VOCABULARY)
        or not isinstance(matrix, list)
        or len(matrix) != 9
        or any(not isinstance(row, list) or len(row) != 9 for row in matrix)
    ):
        raise LocalizationCheckError("pairwise margin matrix changed")
    expected_diagnostic_keys = {
        "action_id",
        "action_name",
        "train_exposure_count",
        "inverse_uniform_train_weight",
        "validation_row_count",
        "factual_rank_histogram_rank_1_through_9",
        "row_weighted_factual_mean_reciprocal_rank",
        "family_equal_scene_macro_factual_energy",
        "family_equal_scene_macro_candidate_spread",
        "minimum_pairwise_macro_competitor_id",
        "minimum_pairwise_macro_competitor_name",
        "minimum_pairwise_macro_margin",
    }
    for action_id, diagnostic in enumerate(diagnostics):
        if not isinstance(diagnostic, dict):
            raise LocalizationCheckError(f"action diagnostic {action_id} changed")
        histogram = diagnostic.get("factual_rank_histogram_rank_1_through_9", [])
        expected_weight = localization_metrics.TRAIN_ROW_COUNT / (
            9 * localization_metrics.TRAIN_EXPOSURE_COUNTS[action_id]
        )
        if (
            set(diagnostic) != expected_diagnostic_keys
            or
            diagnostic["action_id"] != action_id
            or diagnostic["action_name"] != three_arm.ACTION_VOCABULARY[action_id]
            or diagnostic["train_exposure_count"]
            != localization_metrics.TRAIN_EXPOSURE_COUNTS[action_id]
            or diagnostic["validation_row_count"] != factual_counts[action_id]
            or not isinstance(histogram, list)
            or len(histogram) != 9
            or any(type(value) is not int or value < 0 for value in histogram)
            or sum(histogram) != factual_counts[action_id]
            or not math.isclose(
                _finite(
                    diagnostic["inverse_uniform_train_weight"],
                    label="inverse uniform train weight",
                ),
                expected_weight,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            or not math.isclose(
                _finite(
                    diagnostic["row_weighted_factual_mean_reciprocal_rank"],
                    label="factual MRR",
                ),
                math.fsum(
                    count / rank for rank, count in enumerate(histogram, start=1)
                )
                / factual_counts[action_id],
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            or _finite(
                diagnostic["family_equal_scene_macro_factual_energy"],
                label="factual energy",
            )
            <= 0.0
            or _finite(
                diagnostic["family_equal_scene_macro_candidate_spread"],
                label="candidate spread",
            )
            < 0.0
            or _finite(matrix[action_id][action_id], label="pairwise diagonal") != 0.0
        ):
            raise LocalizationCheckError(f"action diagnostic {action_id} changed")
        competitors = [value for value in range(9) if value != action_id]
        minimum = min(
            competitors,
            key=lambda competitor: (
                _finite(matrix[action_id][competitor], label="pairwise margin"),
                competitor,
            ),
        )
        if (
            diagnostic["minimum_pairwise_macro_competitor_id"] != minimum
            or diagnostic["minimum_pairwise_macro_competitor_name"]
            != three_arm.ACTION_VOCABULARY[minimum]
            or not math.isclose(
                _finite(
                    diagnostic["minimum_pairwise_macro_margin"],
                    label="minimum pairwise margin",
                ),
                _finite(matrix[action_id][minimum], label="matrix minimum"),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
        ):
            raise LocalizationCheckError("minimum competitor accounting changed")

    expected_summary_keys = {
        "bootstrap_seed",
        "bootstrap_replicates",
        "bootstrap_lower_index",
        "bootstrap_median_index",
        "bootstrap_upper_index",
        "bootstrap_algorithm",
        "bootstrap_interpretation",
        "shared_across_action_minimum_quantiles",
        "per_action",
    }
    expected_row_keys = {
        "action_id",
        "action_name",
        "row_count",
        "family_equal_scene_macro_point",
        "one_sided_95_lower_quantile",
        "median_quantile",
        "one_sided_95_upper_quantile",
        "positive_family_count",
        "minimum_supporting_scene_count",
        "total_supporting_scene_count",
        "supporting_scene_count_by_family",
        "point_by_family",
    }

    def check_cluster_summary(name: str, expected_seed: int) -> list[dict[str, Any]]:
        summary = localization[name]
        if (
            not isinstance(summary, dict)
            or set(summary) != expected_summary_keys
            or summary["bootstrap_seed"] != expected_seed
            or summary["bootstrap_replicates"]
            != localization_metrics.BOOTSTRAP_REPLICATES
            or summary["bootstrap_lower_index"]
            != localization_metrics.BOOTSTRAP_LOWER_INDEX
            or summary["bootstrap_median_index"]
            != localization_metrics.BOOTSTRAP_MEDIAN_INDEX
            or summary["bootstrap_upper_index"]
            != localization_metrics.BOOTSTRAP_UPPER_INDEX
            or summary["bootstrap_algorithm"] != localization_metrics.BOOTSTRAP_ALGORITHM
            or summary["bootstrap_interpretation"]
            != localization_metrics.BOOTSTRAP_INTERPRETATION
        ):
            raise LocalizationCheckError(f"{name} bootstrap contract changed")
        joint = summary["shared_across_action_minimum_quantiles"]
        if not isinstance(joint, dict) or set(joint) != {
            "one_sided_95_lower_quantile",
            "median_quantile",
            "one_sided_95_upper_quantile",
        }:
            raise LocalizationCheckError(f"{name} joint quantiles changed")
        joint_values = [
            _finite(joint[key], label=f"{name} joint quantile")
            for key in (
                "one_sided_95_lower_quantile",
                "median_quantile",
                "one_sided_95_upper_quantile",
            )
        ]
        if not joint_values[0] <= joint_values[1] <= joint_values[2]:
            raise LocalizationCheckError(f"{name} joint quantiles are unordered")
        rows = summary["per_action"]
        if not isinstance(rows, list) or len(rows) != 9:
            raise LocalizationCheckError(f"{name} action intervals are incomplete")
        for action_id, row in enumerate(rows):
            family_support = row.get("supporting_scene_count_by_family")
            family_points = row.get("point_by_family")
            if (
                not isinstance(row, dict)
                or set(row) != expected_row_keys
                or row["action_id"] != action_id
                or row["action_name"] != three_arm.ACTION_VOCABULARY[action_id]
                or row["row_count"] != factual_counts[action_id]
                or not isinstance(family_support, dict)
                or set(family_support) != set(three_arm.REGISTERED_FAMILIES)
                or family_support
                != {
                    family: support[family][action_id]
                    for family in three_arm.REGISTERED_FAMILIES
                }
                or row["minimum_supporting_scene_count"] != min(family_support.values())
                or row["total_supporting_scene_count"] != sum(family_support.values())
                or not isinstance(family_points, dict)
                or set(family_points) != set(three_arm.REGISTERED_FAMILIES)
            ):
                raise LocalizationCheckError(f"{name} action {action_id} contract changed")
            point_values = [
                _finite(value, label=f"{name} family point")
                for value in family_points.values()
            ]
            point = _finite(
                row["family_equal_scene_macro_point"], label=f"{name} point"
            )
            lower = _finite(
                row["one_sided_95_lower_quantile"], label=f"{name} lower"
            )
            median = _finite(row["median_quantile"], label=f"{name} median")
            upper = _finite(
                row["one_sided_95_upper_quantile"], label=f"{name} upper"
            )
            if (
                row["positive_family_count"] != sum(value > 0.0 for value in point_values)
                or not math.isclose(
                    point,
                    math.fsum(point_values) / 8,
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                )
                or not lower <= median <= upper
            ):
                raise LocalizationCheckError(f"{name} action arithmetic changed")
        return rows

    margin_rows = check_cluster_summary(
        "action_margin_localization", localization_metrics.MARGIN_BOOTSTRAP_SEED
    )
    persistence_rows = check_cluster_summary(
        "persistence_localization", localization_metrics.PERSISTENCE_BOOTSTRAP_SEED
    )
    wrong_history_rows = check_cluster_summary(
        "wrong_history_localization", localization_metrics.WRONG_HISTORY_BOOTSTRAP_SEED
    )
    for action_id, row in enumerate(margin_rows):
        if not math.isclose(
            _finite(row["family_equal_scene_macro_point"], label="localized margin"),
            _finite(margins[action_id], label="registered margin"),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise LocalizationCheckError("localized and registered margin points disagree")
    margin_joint_lower = localization["action_margin_localization"][
        "shared_across_action_minimum_quantiles"
    ]["one_sided_95_lower_quantile"]
    if not math.isclose(
        _finite(margin_joint_lower, label="joint margin lower"),
        _finite(action["hardest_margin_bootstrap_lower_95"], label="registered lower"),
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise LocalizationCheckError("joint margin lower does not reproduce V3")

    topology = localization["failure_topology"]
    point_failures = [
        row["action_id"]
        for row in margin_rows
        if _finite(row["family_equal_scene_macro_point"], label="margin point") <= 0.0
    ]
    lower_failures = [
        row["action_id"]
        for row in margin_rows
        if _finite(row["one_sided_95_lower_quantile"], label="margin lower") <= 0.0
    ]
    persistence_point_failures = [
        row["action_id"]
        for row in persistence_rows
        if _finite(row["family_equal_scene_macro_point"], label="persistence point") <= 0.0
    ]
    persistence_lower_failures = [
        row["action_id"]
        for row in persistence_rows
        if _finite(row["one_sided_95_lower_quantile"], label="persistence lower") <= 0.0
    ]
    action_names = list(three_arm.ACTION_VOCABULARY)
    expected_topology = {
        "registered_hardest_action_id": hardest,
        "registered_hardest_action_name": action_names[hardest],
        "alignment_point_failure_action_ids": point_failures,
        "alignment_point_failure_action_names": [action_names[value] for value in point_failures],
        "alignment_lower_failure_action_ids": lower_failures,
        "alignment_lower_failure_action_names": [action_names[value] for value in lower_failures],
        "persistence_point_failure_action_ids": persistence_point_failures,
        "persistence_point_failure_action_names": [
            action_names[value] for value in persistence_point_failures
        ],
        "persistence_lower_failure_action_ids": persistence_lower_failures,
        "persistence_lower_failure_action_names": [
            action_names[value] for value in persistence_lower_failures
        ],
        "alignment_point_failure_scope": (
            "none" if not point_failures else "localized" if len(point_failures) <= 2 else "broad"
        ),
        "persistence_point_failure_scope": (
            "none"
            if not persistence_point_failures
            else "localized" if len(persistence_point_failures) <= 2 else "broad"
        ),
    }
    if topology != expected_topology:
        raise LocalizationCheckError("failure topology changed")
    expected_routing = localization_metrics._routing_decision(
        margin_rows,
        persistence_rows,
        alignment_shared_minimum_lower=_finite(
            margin_joint_lower, label="routing joint margin lower"
        ),
        aggregate_persistence_lower=_finite(
            controls["persistence"]["bootstrap_lower_95"],
            label="routing aggregate persistence lower",
        ),
    )
    if localization["routing_decision"] != expected_routing:
        raise LocalizationCheckError("frozen successor routing changed")
    # Consume the wrong-history rows in an explicit validation assertion so a
    # future refactor cannot silently drop that namespace from receipt checks.
    if len(wrong_history_rows) != 9:
        raise LocalizationCheckError("wrong-history localization disappeared")


def check_result(
    result: Any,
    *,
    result_binding: Mapping[str, Any],
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "status",
        "citable_as_scientific_evidence",
        "authorizes_training_or_data_generation",
        "authorizes_retry_or_resume",
        "attempt",
        "source_commit",
        "review_commit",
        "execution_head",
        "authority_binding",
        "input_bindings",
        "snapshot_contract",
        "validation_index_audit",
        "public_anchor_reproduction",
        "localization",
        "access_accounting",
        "custody",
        "claim_boundary",
        "reservation_binding",
    }
    if not isinstance(result, dict) or set(result) != expected_keys:
        raise LocalizationCheckError("result top-level fields changed")
    if (
        result["schema"] != worker.RESULT_SCHEMA
        or result["status"] != "PASS_COMPLETE_READ_ONLY_LOCALIZATION"
        or result["citable_as_scientific_evidence"] is not False
        or result["authorizes_training_or_data_generation"] is not False
        or result["authorizes_retry_or_resume"] is not False
        or result["authority_binding"] != authority_binding
        or result["input_bindings"] != authority["input_bindings"]
        or result["source_commit"] != authority["source_commit"]
        or result["review_commit"] != authority["review_commit"]
        or result["execution_head"] != authority["execution_head"]
        or result["access_accounting"] != authority["access_contract"]
        or result["claim_boundary"] != authority["claim_boundary"]
    ):
        raise LocalizationCheckError("result authority or status chain changed")
    attempt = result["attempt"]
    if attempt != {
        "id": worker.ATTEMPT_ID,
        "root": str(worker.ATTEMPT_ROOT),
        "consumed": True,
        "retry": False,
        "resume": False,
        "refill": False,
        "overwrite": False,
    }:
        raise LocalizationCheckError("result attempt envelope changed")
    if result["custody"] != {
        "snapshot_bytes_opened": True,
        "validation_index_bytes_opened": True,
        "pack_payloads_opened": False,
        "rgb_paths_followed": False,
        "other_snapshots_or_checkpoints_opened": False,
        "train_index_opened": False,
        "model_forward_performed": False,
        "training_or_optimizer_step_performed": False,
        "model_or_optimizer_state_restored_or_emitted": False,
        "network_access_used": False,
        "write_beneath_v3_attempt_root": False,
    }:
        raise LocalizationCheckError("result custody receipt changed")
    reservation = worker._binding_shape(
        result["reservation_binding"], label="reservation"
    )
    reservation_raw = worker._read_absolute_regular_once(
        reservation, label="reservation"
    )
    reservation_document = worker.strict_json_bytes(reservation_raw)
    if (
        Path(reservation["path"]) != worker.ATTEMPT_ROOT / "reservation.json"
        or not isinstance(reservation_document, dict)
        or reservation_document
        != worker.expected_reservation(
            authority,
            authority_binding,
            supervisor_nonce=reservation_document.get("supervisor_nonce", ""),
        )
    ):
        raise LocalizationCheckError("reservation binding changed")
    snapshot_contract = result["snapshot_contract"]
    if (
        not isinstance(snapshot_contract, dict)
        or set(snapshot_contract)
        != {
            "schema",
            "status",
            "arm",
            "update",
            "metric_vector_keys",
            "model_or_optimizer_state_consumed_computationally",
            "model_or_optimizer_state_restored_or_emitted",
        }
        or snapshot_contract.get("schema") != worker.SNAPSHOT_SCHEMA
        or snapshot_contract.get("status") != "INERT_AUDIT_SNAPSHOT"
        or snapshot_contract.get("arm") != "conditioned"
        or snapshot_contract.get("update") != 700
        or snapshot_contract.get("model_or_optimizer_state_consumed_computationally")
        is not False
        or snapshot_contract.get("model_or_optimizer_state_restored_or_emitted")
        is not False
        or snapshot_contract.get("metric_vector_keys")
        != sorted(worker.EXPECTED_METRIC_VECTOR_KEYS)
    ):
        raise LocalizationCheckError("snapshot contract receipt changed")
    validation_audit = result["validation_index_audit"]
    if (
        not isinstance(validation_audit, dict)
        or set(validation_audit)
        != {
            "role",
            "path",
            "file_sha256",
            "byte_count",
            "row_count",
            "scene_count",
            "family_rows",
            "family_scenes",
            "minimum_future_action_position_scene_breadth",
            "ordered_row_identity_sha256",
            "rgb_open_count",
        }
        or validation_audit.get("role") != "val"
        or validation_audit.get("path") != worker.h6.VALIDATION_INDEX.as_posix()
        or validation_audit.get("file_sha256")
        != worker.VALIDATION_INDEX_BINDING["file_sha256"]
        or validation_audit.get("byte_count")
        != worker.VALIDATION_INDEX_BINDING["byte_count"]
        or validation_audit.get("row_count") != 2_048
        or validation_audit.get("scene_count") != 150
        or validation_audit.get("family_rows")
        != {family: 256 for family in worker.h6.LEXICOGRAPHIC_FAMILIES}
        or validation_audit.get("family_scenes")
        != {
            family: worker.h6.EXPECTED_SCENES["val"][family]
            for family in worker.h6.LEXICOGRAPHIC_FAMILIES
        }
        or type(
            validation_audit.get("minimum_future_action_position_scene_breadth")
        )
        is not int
        or validation_audit["minimum_future_action_position_scene_breadth"] < 1
        or type(validation_audit.get("ordered_row_identity_sha256")) is not str
        or len(validation_audit["ordered_row_identity_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in validation_audit["ordered_row_identity_sha256"]
        )
        or validation_audit.get("rgb_open_count") != 0
    ):
        raise LocalizationCheckError("validation-index audit changed")
    forbidden = sorted(_all_keys(result) & FORBIDDEN_RESULT_KEYS)
    if forbidden:
        raise LocalizationCheckError(f"result emitted forbidden raw fields: {forbidden}")
    _exact_public_anchors(result)
    _check_localization(result["localization"])
    return {
        "schema": CHECK_SCHEMA,
        "status": "PASS",
        "citable_as_scientific_evidence": False,
        "scientific_verdict_emitted": False,
        "manifest_binding": dict(result_binding),
        "reservation_binding": dict(reservation),
        "authority_binding": dict(authority_binding),
        "attempt_id": worker.ATTEMPT_ID,
        "result_status": result["status"],
        "aggregate_arithmetic_recomputed": True,
        "public_anchors_exact": True,
        "forbidden_raw_result_fields": forbidden,
        "snapshot_or_checkpoint_payloads_opened_by_checker": False,
        "validation_index_opened_by_checker": False,
        "pack_or_rgb_opened_by_checker": False,
        "network_access_used_by_checker": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-file-sha256", required=True)
    parser.add_argument("--expected-byte-count", type=int, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--expected-authority-byte-count", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    worker.validate_exact_child_environment()
    authority, authority_binding = worker.load_and_validate_authority(
        arguments.authority,
        expected_sha256=arguments.expected_authority_sha256,
        expected_byte_count=arguments.expected_authority_byte_count,
    )
    manifest_path = arguments.manifest.resolve(strict=True)
    if manifest_path != worker.ATTEMPT_ROOT / "localization.json":
        raise LocalizationCheckError("localization result path changed")
    manifest_binding = {
        "path": str(manifest_path),
        "file_sha256": arguments.expected_file_sha256,
        "byte_count": arguments.expected_byte_count,
    }
    raw = worker._read_absolute_regular_once(manifest_binding, label="localization result")
    worker.exact_root_inventory({"reservation.json", "localization.json"})
    result = worker.strict_json_bytes(raw)
    receipt = check_result(
        result,
        result_binding=manifest_binding,
        authority=authority,
        authority_binding=authority_binding,
    )
    output = worker.ATTEMPT_ROOT / "receipt_check.json"
    worker.write_immutable_json(output, receipt)
    worker.exact_root_inventory(
        {"reservation.json", "localization.json", "receipt_check.json"}
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
