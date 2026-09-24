"""Deterministic metrics for the post-action projective-support V1 probe.

This module is deliberately pure: it accepts already-produced probabilities and
labels and performs no model, checkpoint, dataset, or filesystem access.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Mapping, Sequence

import numpy as np


ACTION_VOCABULARY = (
    "arc_left",
    "arc_right",
    "backward",
    "forward_fast",
    "forward_medium",
    "forward_slow",
    "hold",
    "yaw_left",
    "yaw_right",
)
HOLD_ACTION_INDEX = 6
NON_HOLD_ACTION_INDICES = np.asarray((0, 1, 2, 3, 4, 5, 7, 8), dtype=np.int64)
ARM_NAMES = (
    "full",
    "coordinate_matched_persistence",
    "shuffled_action",
    "wrong_rgb",
    "action_prior",
)
REGISTERED_FAMILIES = (
    "large_enclosed_maze",
    "local_composite_motifs",
    "loop_alias_stress",
    "medium_enclosed_maze",
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "visual_sensor_stress",
)
CALIBRATION_PRECISION_MIN = 0.99
BOOTSTRAP_SCENE_COUNT = 8
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_728
BOOTSTRAP_LOWER_INDEX = 249


@dataclass(frozen=True)
class StationMetrics:
    row_count: int
    safe_count: int
    unsafe_count: int
    admitted_count: int
    admitted_safe_count: int
    rejected_unsafe_count: int
    safety_precision: float
    safe_recall: float
    unsafe_recall: float
    brier_score: float
    ece: float


@dataclass(frozen=True)
class ArmCalibration:
    arm_name: str
    eligible: bool
    threshold: float | None
    metrics: StationMetrics | None
    per_family: Mapping[str, StationMetrics]
    candidate_count: int
    nonempty_candidate_count: int
    eligible_candidate_count: int


@dataclass(frozen=True)
class CalibrationSuite:
    arms: Mapping[str, ArmCalibration]
    comparable: bool
    failed_arms: tuple[str, ...]


@dataclass(frozen=True)
class GroupMetrics:
    scene_count: int
    state_count: int
    primary_state_count: int
    station_row_count: int
    station_safe_count: int
    station_unsafe_count: int
    station_admitted_count: int
    station_safety_precision: float
    station_safe_recall: float
    station_unsafe_recall: float
    mean_primary_utility: float
    nonempty_selected_prefix_rate: float
    selected_station_admitted_count: int
    selected_station_precision: float
    selected_action_shares: tuple[float, ...]
    oracle_action_shares: tuple[float, ...]


@dataclass(frozen=True)
class ArmEvaluation:
    arm_name: str
    threshold: float
    scene_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    primary_mask: np.ndarray
    predicted_prefix_lengths: np.ndarray
    target_prefix_lengths: np.ndarray
    selected_action_indices: np.ndarray
    oracle_action_indices: np.ndarray
    selected_target_prefix: np.ndarray
    oracle_target_prefix: np.ndarray
    primary_utility: np.ndarray
    outside_subset_state_count: int
    outside_subset_selected_infeasible_count: int
    outside_subset_state_indices: tuple[int, ...]
    outside_subset_utility_values: tuple[float | None, ...]
    outside_subset_mean_utility: float | None
    scenes: Mapping[str, GroupMetrics]
    families: Mapping[str, GroupMetrics]
    overall: GroupMetrics


@dataclass(frozen=True)
class PairedBootstrap:
    scene_count: int
    replicate_count: int
    seed: int
    point_delta: float
    lower_95: float


@dataclass(frozen=True)
class ArmComparison:
    control_name: str
    bootstrap: PairedBootstrap
    family_deltas: Mapping[str, float]
    positive_family_count: int


@dataclass(frozen=True)
class IntegrityMetrics:
    exact_accounting: bool
    outputs_and_gradients_finite: bool
    target_gradients_zero: bool
    target_optimizer_membership_zero: bool
    online_gradients_nonzero_every_update: bool
    predictor_forward_count: int
    predictor_objective_count: int
    backward_count: int
    predictor_optimizer_update_count: int
    forbidden_input_count: int
    bypass_count: int
    forbidden_open_count: int
    current_latents_nonconstant: bool
    paired_latents_nonconstant: bool
    current_and_paired_latents_nonidentical: bool
    one_step_zero_support_witnessed: bool
    all_corridor_masks_nonempty: bool
    corridor_masks_inside_support: bool


@dataclass(frozen=True)
class SemanticRetentionMetrics:
    balanced_accuracy: float
    free_recall: float
    occupied_recall: float
    unknown_recall: float
    rough_family_occupied_recall: float


@dataclass(frozen=True)
class GateDecision:
    status: str
    passed: bool
    checks: Mapping[str, bool]
    failed_checks: tuple[str, ...]
    comparisons: Mapping[str, ArmComparison]
    failed_calibration_arms: tuple[str, ...] = ()


@dataclass(frozen=True)
class WrongRgbEndpointMapping:
    rows: tuple[tuple[str, str, str, str], ...]
    by_endpoint: Mapping[tuple[str, str, str], str]
    mapping_sha256: str


@dataclass(frozen=True)
class OracleMetricPipelinePreflight:
    passed: bool
    checks: Mapping[str, bool]
    failed_checks: tuple[str, ...]
    calibration: ArmCalibration
    selection: ArmEvaluation | None
    bootstrap: PairedBootstrap | None


def _confusion_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (3, 3) or raw.dtype == np.bool_ or not np.issubdtype(
        raw.dtype, np.number
    ):
        raise ValueError(f"{name} must be a numeric 3x3 count matrix")
    matrix = raw.astype(np.float64)
    if (
        not np.isfinite(matrix).all()
        or (matrix < 0.0).any()
        or not np.equal(matrix, np.floor(matrix)).all()
    ):
        raise ValueError(f"{name} must contain finite nonnegative integer counts")
    return matrix


def semantic_retention_from_confusions(
    overall_3x3: np.ndarray,
    rough_3x3: np.ndarray,
) -> SemanticRetentionMetrics:
    """Compute UNKNOWN/FREE/OCC recalls from true-row, predicted-column counts."""
    overall = _confusion_matrix(overall_3x3, name="overall_3x3")
    rough = _confusion_matrix(rough_3x3, name="rough_3x3")
    support = overall.sum(axis=1)
    if bool((support == 0).any()):
        raise ValueError("overall_3x3 must support UNKNOWN, FREE, and OCCUPIED")
    rough_occupied_support = float(rough[2].sum())
    if rough_occupied_support == 0.0:
        raise ValueError("rough_3x3 must support OCCUPIED")
    recalls = np.diag(overall) / support
    return SemanticRetentionMetrics(
        balanced_accuracy=float(recalls.mean()),
        free_recall=float(recalls[1]),
        occupied_recall=float(recalls[2]),
        unknown_recall=float(recalls[0]),
        rough_family_occupied_recall=float(rough[2, 2] / rough_occupied_support),
    )


def _station_safe_array(station_safe: np.ndarray, *, name: str) -> np.ndarray:
    raw = np.asarray(station_safe)
    if raw.ndim != 3 or raw.shape[1:] != (len(ACTION_VOCABULARY), 11):
        raise ValueError(f"{name} must have shape (N, 9, 11)")
    if raw.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one state")
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be binary") from error
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError(f"{name} must be binary")
    return numeric.astype(bool)


def action_prior_probabilities(station_safe: np.ndarray) -> np.ndarray:
    """Return the train-only per-action/station empirical safe frequency."""
    labels = _station_safe_array(station_safe, name="station_safe")
    safe_counts = labels.sum(axis=0)
    row_count = labels.shape[0]
    non_hold_counts = safe_counts[NON_HOLD_ACTION_INDICES]
    if bool(((non_hold_counts == 0) | (non_hold_counts == row_count)).any()):
        raise ValueError(
            "every non-HOLD train action/station requires safe and unsafe support"
        )
    return labels.mean(axis=0, dtype=np.float64)


def wrong_rgb_endpoint_mapping(
    role_scene_endpoints: Sequence[tuple[str, str, str]],
) -> WrongRgbEndpointMapping:
    """Build the frozen role/scene-local lexicographic cyclic derangement."""
    groups: dict[tuple[str, str], set[str]] = {}
    endpoint_scenes: dict[tuple[str, str], str] = {}
    for item in role_scene_endpoints:
        if len(item) != 3 or any(type(value) is not str or not value for value in item):
            raise ValueError("endpoint rows must be nonempty (role, scene, endpoint) strings")
        role, scene, endpoint = item
        location = (role, endpoint)
        if location in endpoint_scenes and endpoint_scenes[location] != scene:
            raise ValueError("an endpoint cannot belong to two scenes within a role")
        endpoint_scenes[location] = scene
        groups.setdefault((role, scene), set()).add(endpoint)
    if not groups:
        raise ValueError("endpoint population is empty")
    rows: list[tuple[str, str, str, str]] = []
    for (role, scene), endpoints_set in sorted(groups.items()):
        endpoints = sorted(endpoints_set)
        if len(endpoints) < 2:
            raise ValueError(f"role/scene {role!r}/{scene!r} has fewer than two endpoints")
        rows.extend(
            (role, scene, endpoint, endpoints[(index + 1) % len(endpoints)])
            for index, endpoint in enumerate(endpoints)
        )
    frozen_rows = tuple(rows)
    payload = json.dumps(
        [list(row) for row in frozen_rows],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return WrongRgbEndpointMapping(
        rows=frozen_rows,
        by_endpoint={row[:3]: row[3] for row in frozen_rows},
        mapping_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _scores_and_labels(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels)
    if score_array.shape != label_array.shape or score_array.size == 0:
        raise ValueError("scores and labels must have the same nonempty shape")
    if not np.isfinite(score_array).all() or not np.isfinite(label_array).all():
        raise ValueError("scores and labels must be finite")
    if (score_array < 0.0).any() or (score_array > 1.0).any():
        raise ValueError("scores must lie in [0, 1]")
    if not np.isin(label_array, (0, 1, False, True)).all():
        raise ValueError("labels must be binary")
    return score_array, label_array.astype(bool, copy=False)


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    """Return the exact finite unique-score candidate set, sorted ascending."""
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.size == 0 or not np.isfinite(score_array).all():
        raise ValueError("scores must be nonempty and finite")
    if (score_array < 0.0).any() or (score_array > 1.0).any():
        raise ValueError("scores must lie in [0, 1]")
    upper = np.nextafter(float(score_array.max()), math.inf)
    return np.unique(np.concatenate((score_array.ravel(), np.asarray((0.0, upper)))))


def expected_calibration_error(
    scores: np.ndarray, labels: np.ndarray, *, bins: int = 15
) -> float:
    """Binary ECE with left-closed bins and an inclusive final endpoint."""
    score_array, label_array = _scores_and_labels(scores, labels)
    if bins != 15:
        raise ValueError("this preregistered metric requires exactly 15 bins")
    total = float(score_array.size)
    error = 0.0
    for index in range(15):
        lower, upper = index / 15.0, (index + 1) / 15.0
        selected = (score_array >= lower) & (
            score_array <= upper if index == 14 else score_array < upper
        )
        count = int(selected.sum())
        if count:
            error += count / total * abs(
                float(score_array[selected].mean())
                - float(label_array[selected].mean())
            )
    return float(error)


def station_classification_metrics(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> StationMetrics:
    score_array, safe = _scores_and_labels(scores, labels)
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    admitted = score_array >= float(threshold)
    unsafe = ~safe
    admitted_count = int(admitted.sum())
    safe_count = int(safe.sum())
    unsafe_count = int(unsafe.sum())
    admitted_safe = int((admitted & safe).sum())
    rejected_unsafe = int((~admitted & unsafe).sum())
    return StationMetrics(
        row_count=int(score_array.size),
        safe_count=safe_count,
        unsafe_count=unsafe_count,
        admitted_count=admitted_count,
        admitted_safe_count=admitted_safe,
        rejected_unsafe_count=rejected_unsafe,
        safety_precision=_ratio(admitted_safe, admitted_count),
        safe_recall=_ratio(admitted_safe, safe_count),
        unsafe_recall=_ratio(rejected_unsafe, unsafe_count),
        brier_score=float(np.square(score_array - safe.astype(np.float64)).mean()),
        ece=expected_calibration_error(score_array, safe),
    )


def select_calibration_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    arm_name: str = "full",
    family_ids: Sequence[object] | None = None,
    minimum_safe_precision: float = CALIBRATION_PRECISION_MIN,
) -> ArmCalibration:
    """Select the frozen threshold, with no fallback when none is eligible."""
    score_array, safe = _scores_and_labels(scores, labels)
    if not 0.0 <= minimum_safe_precision <= 1.0:
        raise ValueError("minimum_safe_precision must lie in [0, 1]")
    candidates = threshold_candidates(score_array)
    nonempty: list[tuple[float, StationMetrics]] = []
    eligible: list[tuple[float, StationMetrics]] = []
    for threshold in candidates:
        metrics = station_classification_metrics(score_array, safe, float(threshold))
        if metrics.admitted_count == 0:
            continue
        item = (float(threshold), metrics)
        nonempty.append(item)
        if metrics.safety_precision >= minimum_safe_precision:
            eligible.append(item)
    if not eligible:
        return ArmCalibration(
            arm_name=arm_name,
            eligible=False,
            threshold=None,
            metrics=None,
            per_family={},
            candidate_count=int(candidates.size),
            nonempty_candidate_count=len(nonempty),
            eligible_candidate_count=0,
        )
    threshold, metrics = max(
        eligible,
        key=lambda item: (
            item[1].safe_recall,
            item[1].unsafe_recall,
            item[1].safety_precision,
            item[0],
        ),
    )
    per_family: dict[str, StationMetrics] = {}
    if family_ids is not None:
        families = np.asarray(tuple(str(value) for value in family_ids), dtype=object)
        if score_array.ndim == 0 or families.shape != (score_array.shape[0],):
            raise ValueError("family_ids must match the leading score dimension")
        for family in sorted(set(families.tolist())):
            selected = families == family
            per_family[family] = station_classification_metrics(
                score_array[selected], safe[selected], threshold
            )
    return ArmCalibration(
        arm_name=arm_name,
        eligible=True,
        threshold=threshold,
        metrics=metrics,
        per_family=per_family,
        candidate_count=int(candidates.size),
        nonempty_candidate_count=len(nonempty),
        eligible_candidate_count=len(eligible),
    )


def calibrate_arms(
    scores_by_arm: Mapping[str, np.ndarray],
    labels_by_arm: np.ndarray | Mapping[str, np.ndarray],
    *,
    family_ids: Sequence[object] | None = None,
) -> CalibrationSuite:
    """Calibrate all five arms and make any missing threshold terminal."""
    if set(scores_by_arm) != set(ARM_NAMES):
        raise ValueError(f"scores_by_arm must contain exactly {ARM_NAMES}")
    if isinstance(labels_by_arm, Mapping):
        if set(labels_by_arm) != set(ARM_NAMES):
            raise ValueError(f"labels_by_arm must contain exactly {ARM_NAMES}")
        reference = np.asarray(labels_by_arm[ARM_NAMES[0]])
        if any(
            not np.array_equal(np.asarray(labels_by_arm[name]), reference)
            for name in ARM_NAMES[1:]
        ):
            raise ValueError("all arms must use identical target labels")
        labels = reference
    else:
        labels = np.asarray(labels_by_arm)
    arms = {
        name: select_calibration_threshold(
            scores_by_arm[name],
            labels,
            arm_name=name,
            family_ids=family_ids,
        )
        for name in ARM_NAMES
    }
    failed = tuple(name for name in ARM_NAMES if not arms[name].eligible)
    return CalibrationSuite(arms=arms, comparable=not failed, failed_arms=failed)


def safe_prefix_lengths(station_safe: np.ndarray) -> np.ndarray:
    """Count consecutive accepted stations from station zero."""
    safe = np.asarray(station_safe, dtype=bool)
    if safe.ndim == 0 or safe.shape[-1] == 0:
        raise ValueError("station_safe must have a nonempty station dimension")
    return np.logical_and.accumulate(safe, axis=-1).sum(axis=-1, dtype=np.int64)


def select_non_hold_actions(prefix_lengths: np.ndarray) -> np.ndarray:
    """Select the longest non-HOLD prefix; numpy's first max freezes ties."""
    prefixes = np.asarray(prefix_lengths)
    if prefixes.ndim < 1 or prefixes.shape[-1] != len(ACTION_VOCABULARY):
        raise ValueError("prefix_lengths must end with the nine-action dimension")
    relative = np.argmax(prefixes[..., NON_HOLD_ACTION_INDICES], axis=-1)
    return NON_HOLD_ACTION_INDICES[relative]


def primary_subset_mask(
    station_safe: np.ndarray,
    immediate_feasible: np.ndarray,
    blind_bridge_feasible: np.ndarray,
) -> np.ndarray:
    """Frozen informative/all-non-HOLD-feasible subset, defined from labels only."""
    safe = np.asarray(station_safe, dtype=bool)
    immediate = np.asarray(immediate_feasible, dtype=bool)
    bridge = np.asarray(blind_bridge_feasible, dtype=bool)
    if safe.ndim != 3 or safe.shape[1:] != (len(ACTION_VOCABULARY), 11):
        raise ValueError("station_safe must have shape (N, 9, 11)")
    if immediate.shape != safe.shape[:2] or bridge.shape != safe.shape[:2]:
        raise ValueError("feasibility arrays must have shape (N, 9)")
    target_prefix = safe_prefix_lengths(safe)[:, NON_HOLD_ACTION_INDICES]
    return (
        immediate[:, NON_HOLD_ACTION_INDICES].all(axis=1)
        & bridge[:, NON_HOLD_ACTION_INDICES].all(axis=1)
        & (target_prefix.max(axis=1) > 0)
        & (target_prefix.max(axis=1) != target_prefix.min(axis=1))
    )


def _mean(values: Sequence[float]) -> float:
    return float(math.fsum(float(value) for value in values) / len(values)) if values else 0.0


def _one_scene_group(
    indices: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    primary: np.ndarray,
    selected: np.ndarray,
    oracle: np.ndarray,
    predicted_prefix: np.ndarray,
    utility: np.ndarray,
) -> GroupMetrics:
    station = station_classification_metrics(
        probabilities[indices], labels[indices], threshold
    )
    local_primary = primary[indices]
    primary_indices = indices[local_primary]
    primary_count = int(primary_indices.size)
    selected_scores = probabilities[
        primary_indices, selected[primary_indices], :
    ]
    selected_labels = labels[primary_indices, selected[primary_indices], :]
    if primary_count:
        selected_station = station_classification_metrics(
            selected_scores, selected_labels, threshold
        )
        selected_counts = np.bincount(
            selected[primary_indices], minlength=len(ACTION_VOCABULARY)
        )
        oracle_counts = np.bincount(
            oracle[primary_indices], minlength=len(ACTION_VOCABULARY)
        )
        selected_shares = tuple((selected_counts / primary_count).tolist())
        oracle_shares = tuple((oracle_counts / primary_count).tolist())
        mean_utility = float(utility[primary_indices].mean())
        nonempty = float((predicted_prefix[primary_indices, selected[primary_indices]] > 0).mean())
    else:
        selected_station = StationMetrics(0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
        selected_shares = (0.0,) * len(ACTION_VOCABULARY)
        oracle_shares = (0.0,) * len(ACTION_VOCABULARY)
        mean_utility = 0.0
        nonempty = 0.0
    return GroupMetrics(
        scene_count=1,
        state_count=int(indices.size),
        primary_state_count=primary_count,
        station_row_count=station.row_count,
        station_safe_count=station.safe_count,
        station_unsafe_count=station.unsafe_count,
        station_admitted_count=station.admitted_count,
        station_safety_precision=station.safety_precision,
        station_safe_recall=station.safe_recall,
        station_unsafe_recall=station.unsafe_recall,
        mean_primary_utility=mean_utility,
        nonempty_selected_prefix_rate=nonempty,
        selected_station_admitted_count=selected_station.admitted_count,
        selected_station_precision=selected_station.safety_precision,
        selected_action_shares=selected_shares,
        oracle_action_shares=oracle_shares,
    )


def _combine_groups(groups: Sequence[GroupMetrics]) -> GroupMetrics:
    if not groups:
        raise ValueError("cannot aggregate an empty group")
    return GroupMetrics(
        scene_count=sum(group.scene_count for group in groups),
        state_count=sum(group.state_count for group in groups),
        primary_state_count=sum(group.primary_state_count for group in groups),
        station_row_count=sum(group.station_row_count for group in groups),
        station_safe_count=sum(group.station_safe_count for group in groups),
        station_unsafe_count=sum(group.station_unsafe_count for group in groups),
        station_admitted_count=sum(group.station_admitted_count for group in groups),
        station_safety_precision=_mean([group.station_safety_precision for group in groups]),
        station_safe_recall=_mean([group.station_safe_recall for group in groups]),
        station_unsafe_recall=_mean([group.station_unsafe_recall for group in groups]),
        mean_primary_utility=_mean([group.mean_primary_utility for group in groups]),
        nonempty_selected_prefix_rate=_mean(
            [group.nonempty_selected_prefix_rate for group in groups]
        ),
        selected_station_admitted_count=sum(
            group.selected_station_admitted_count for group in groups
        ),
        selected_station_precision=_mean(
            [group.selected_station_precision for group in groups]
        ),
        selected_action_shares=tuple(
            _mean([group.selected_action_shares[index] for group in groups])
            for index in range(len(ACTION_VOCABULARY))
        ),
        oracle_action_shares=tuple(
            _mean([group.oracle_action_shares[index] for group in groups])
            for index in range(len(ACTION_VOCABULARY))
        ),
    )


def evaluate_arm(
    probabilities: np.ndarray,
    station_safe: np.ndarray,
    threshold: float,
    scene_ids: Sequence[object],
    family_ids: Sequence[object],
    immediate_feasible: np.ndarray,
    blind_bridge_feasible: np.ndarray,
    *,
    arm_name: str = "full",
) -> ArmEvaluation:
    """Evaluate action choice and target-derived utility on the frozen subset."""
    scores, labels = _scores_and_labels(probabilities, station_safe)
    if scores.ndim != 3 or scores.shape[1:] != (len(ACTION_VOCABULARY), 11):
        raise ValueError("probabilities and station_safe must have shape (N, 9, 11)")
    scenes = np.asarray(tuple(str(value) for value in scene_ids), dtype=object)
    families = np.asarray(tuple(str(value) for value in family_ids), dtype=object)
    if scenes.shape != (scores.shape[0],) or families.shape != (scores.shape[0],):
        raise ValueError("scene_ids and family_ids must match the state dimension")
    scene_family: dict[str, str] = {}
    for scene, family in zip(scenes.tolist(), families.tolist(), strict=True):
        if scene in scene_family and scene_family[scene] != family:
            raise ValueError("each scene must belong to exactly one family")
        scene_family[scene] = family
    predicted_prefix = safe_prefix_lengths(scores >= float(threshold))
    target_prefix = safe_prefix_lengths(labels)
    selected = select_non_hold_actions(predicted_prefix)
    oracle = select_non_hold_actions(target_prefix)
    rows = np.arange(scores.shape[0])
    selected_target = target_prefix[rows, selected]
    oracle_target = target_prefix[rows, oracle]
    immediate = np.asarray(immediate_feasible, dtype=bool)
    bridge = np.asarray(blind_bridge_feasible, dtype=bool)
    primary = primary_subset_mask(labels, immediate, bridge)
    utility = np.full(scores.shape[0], np.nan, dtype=np.float64)
    utility[primary] = selected_target[primary] / oracle_target[primary]
    outside_indices = np.flatnonzero(~primary)
    selected_feasible = (
        immediate[rows, selected] & bridge[rows, selected]
    )
    outside_utility_values = tuple(
        -1.0
        if not selected_feasible[index]
        else (
            float(selected_target[index] / oracle_target[index])
            if oracle_target[index] > 0
            else None
        )
        for index in outside_indices.tolist()
    )
    defined_outside_utility = tuple(
        value for value in outside_utility_values if value is not None
    )
    scene_metrics: dict[str, GroupMetrics] = {}
    for scene in sorted(scene_family):
        indices = np.flatnonzero(scenes == scene)
        scene_metrics[scene] = _one_scene_group(
            indices,
            scores,
            labels,
            float(threshold),
            primary,
            selected,
            oracle,
            predicted_prefix,
            utility,
        )
    family_metrics = {
        family: _combine_groups(
            [
                scene_metrics[scene]
                for scene in sorted(scene_family)
                if scene_family[scene] == family
            ]
        )
        for family in sorted(set(scene_family.values()))
    }
    return ArmEvaluation(
        arm_name=arm_name,
        threshold=float(threshold),
        scene_ids=tuple(scenes.tolist()),
        family_ids=tuple(families.tolist()),
        primary_mask=primary,
        predicted_prefix_lengths=predicted_prefix,
        target_prefix_lengths=target_prefix,
        selected_action_indices=selected,
        oracle_action_indices=oracle,
        selected_target_prefix=selected_target,
        oracle_target_prefix=oracle_target,
        primary_utility=utility,
        outside_subset_state_count=int(outside_indices.size),
        outside_subset_selected_infeasible_count=int(
            (~selected_feasible[outside_indices]).sum()
        ),
        outside_subset_state_indices=tuple(int(index) for index in outside_indices),
        outside_subset_utility_values=outside_utility_values,
        outside_subset_mean_utility=(
            _mean(defined_outside_utility) if defined_outside_utility else None
        ),
        scenes=scene_metrics,
        families=family_metrics,
        overall=_combine_groups(list(scene_metrics.values())),
    )


def paired_scene_bootstrap(
    full_scene_values: Mapping[str, float],
    control_scene_values: Mapping[str, float],
) -> PairedBootstrap:
    """Frozen eight-scene paired cluster bootstrap (sorted index 249)."""
    if set(full_scene_values) != set(control_scene_values):
        raise ValueError("full and control scene sets must match")
    scenes = sorted(full_scene_values)
    if len(scenes) != BOOTSTRAP_SCENE_COUNT:
        raise ValueError("paired bootstrap requires exactly eight scenes")
    deltas = [float(full_scene_values[key]) - float(control_scene_values[key]) for key in scenes]
    if not all(math.isfinite(value) for value in deltas):
        raise ValueError("scene values must be finite")
    rng = random.Random(BOOTSTRAP_SEED)
    replicates = sorted(
        math.fsum(deltas[rng.randrange(BOOTSTRAP_SCENE_COUNT)] for _ in scenes)
        / BOOTSTRAP_SCENE_COUNT
        for _ in range(BOOTSTRAP_REPLICATES)
    )
    return PairedBootstrap(
        scene_count=BOOTSTRAP_SCENE_COUNT,
        replicate_count=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
        point_delta=_mean(deltas),
        lower_95=float(replicates[BOOTSTRAP_LOWER_INDEX]),
    )


def compare_arms(full: ArmEvaluation, control: ArmEvaluation) -> ArmComparison:
    """Compare target utility with paired scenes and registered-family margins."""
    if full.arm_name != "full" or control.arm_name == "full":
        raise ValueError("comparison requires a full arm and a non-full control")
    if (
        full.scene_ids != control.scene_ids
        or full.family_ids != control.family_ids
        or not np.array_equal(full.primary_mask, control.primary_mask)
        or not np.array_equal(full.target_prefix_lengths, control.target_prefix_lengths)
        or not np.array_equal(full.oracle_target_prefix, control.oracle_target_prefix)
    ):
        raise ValueError("arms must share the exact target-defined evaluation population")
    if set(full.scenes) != set(control.scenes) or set(full.families) != set(control.families):
        raise ValueError("arm scene and family sets must match")
    bootstrap = paired_scene_bootstrap(
        {key: value.mean_primary_utility for key, value in full.scenes.items()},
        {key: value.mean_primary_utility for key, value in control.scenes.items()},
    )
    family_deltas = {
        key: full.families[key].mean_primary_utility
        - control.families[key].mean_primary_utility
        for key in sorted(full.families)
    }
    return ArmComparison(
        control_name=control.arm_name,
        bootstrap=bootstrap,
        family_deltas=family_deltas,
        positive_family_count=sum(value > 0.0 for value in family_deltas.values()),
    )


def oracle_metric_pipeline_preflight(
    calibration_station_safe: np.ndarray,
    selection_station_safe: np.ndarray,
    selection_scene_ids: Sequence[object],
    selection_family_ids: Sequence[object],
    immediate_feasible: np.ndarray,
    blind_bridge_feasible: np.ndarray,
    *,
    calibration_family_ids: Sequence[object] | None = None,
) -> OracleMetricPipelinePreflight:
    """Run the frozen metric path with exact labels as oracle probabilities."""
    calibration_labels = _station_safe_array(
        calibration_station_safe, name="calibration_station_safe"
    )
    selection_labels = _station_safe_array(
        selection_station_safe, name="selection_station_safe"
    )
    calibration = select_calibration_threshold(
        calibration_labels.astype(np.float64),
        calibration_labels,
        arm_name="oracle",
        family_ids=calibration_family_ids,
    )
    checks: dict[str, bool] = {
        "calibration_threshold_eligible": calibration.eligible,
        "calibration_threshold_exact_one": calibration.threshold == 1.0,
        "calibration_precision_exact_one": bool(
            calibration.metrics is not None
            and calibration.metrics.safety_precision == 1.0
        ),
        "calibration_safe_recall_exact_one": bool(
            calibration.metrics is not None and calibration.metrics.safe_recall == 1.0
        ),
        "calibration_unsafe_recall_exact_one": bool(
            calibration.metrics is not None and calibration.metrics.unsafe_recall == 1.0
        ),
    }
    if calibration_family_ids is not None:
        checks["calibration_has_exact_registered_families"] = (
            set(calibration.per_family) == set(REGISTERED_FAMILIES)
        )
    if calibration.threshold is None:
        failed = tuple(name for name, value in checks.items() if not value)
        return OracleMetricPipelinePreflight(
            passed=False,
            checks=checks,
            failed_checks=failed,
            calibration=calibration,
            selection=None,
            bootstrap=None,
        )
    selection = evaluate_arm(
        selection_labels.astype(np.float64),
        selection_labels,
        calibration.threshold,
        selection_scene_ids,
        selection_family_ids,
        immediate_feasible,
        blind_bridge_feasible,
        arm_name="oracle",
    )
    checks.update(
        {
            "selection_has_eight_scenes": len(selection.scenes) == BOOTSTRAP_SCENE_COUNT,
            "selection_has_exact_registered_families": (
                set(selection.families) == set(REGISTERED_FAMILIES)
            ),
            "selection_precision_exact_one": (
                selection.overall.station_safety_precision == 1.0
            ),
            "selection_safe_recall_exact_one": (
                selection.overall.station_safe_recall == 1.0
            ),
            "selection_unsafe_recall_exact_one": (
                selection.overall.station_unsafe_recall == 1.0
            ),
            "selection_utility_exact_one": (
                selection.overall.mean_primary_utility == 1.0
            ),
        }
    )
    for family in REGISTERED_FAMILIES:
        metrics = selection.families.get(family)
        checks[f"family:{family}:nonempty_admission"] = bool(
            metrics is not None and metrics.station_admitted_count > 0
        )
        checks[f"family:{family}:nonempty_prefix_exact_one"] = bool(
            metrics is not None and metrics.nonempty_selected_prefix_rate == 1.0
        )
    bootstrap: PairedBootstrap | None = None
    if len(selection.scenes) == BOOTSTRAP_SCENE_COUNT:
        oracle_scene_utility = {
            scene: metrics.mean_primary_utility
            for scene, metrics in selection.scenes.items()
        }
        bootstrap = paired_scene_bootstrap(
            oracle_scene_utility,
            {scene: 0.0 for scene in oracle_scene_utility},
        )
    checks["bootstrap_oracle_delta_exact_one"] = bool(
        bootstrap is not None
        and bootstrap.point_delta == 1.0
        and bootstrap.lower_95 == 1.0
    )
    failed = tuple(name for name, value in checks.items() if not value)
    return OracleMetricPipelinePreflight(
        passed=not failed,
        checks=checks,
        failed_checks=failed,
        calibration=calibration,
        selection=selection,
        bootstrap=bootstrap,
    )


def evaluate_conjunctive_gate(
    calibrations: CalibrationSuite,
    evaluations: Mapping[str, ArmEvaluation],
    integrity: IntegrityMetrics,
    semantic: SemanticRetentionMetrics,
) -> GateDecision:
    """Apply every preregistered gate; failed control calibration is terminal."""
    if not calibrations.comparable:
        return GateDecision(
            status="TERMINAL_NON_COMPARABLE_CONTROL_CALIBRATION",
            passed=False,
            checks={"all_arm_calibrations_eligible": False},
            failed_checks=("all_arm_calibrations_eligible",),
            comparisons={},
            failed_calibration_arms=calibrations.failed_arms,
        )
    if set(calibrations.arms) != set(ARM_NAMES):
        raise ValueError(f"calibrations must contain exactly {ARM_NAMES}")
    if set(evaluations) != set(ARM_NAMES):
        raise ValueError(f"evaluations must contain exactly {ARM_NAMES}")
    for name in ARM_NAMES:
        calibration = calibrations.arms[name]
        if calibration.threshold is None or evaluations[name].threshold != calibration.threshold:
            raise ValueError("evaluation thresholds must be the selected calibration thresholds")
    full = evaluations["full"]
    checks: dict[str, bool] = {
        "all_arm_calibrations_eligible": True,
        "integrity_exact_accounting": integrity.exact_accounting,
        "integrity_finite": integrity.outputs_and_gradients_finite,
        "integrity_target_gradients_zero": integrity.target_gradients_zero,
        "integrity_target_optimizer_membership_zero": integrity.target_optimizer_membership_zero,
        "integrity_online_gradients_nonzero": integrity.online_gradients_nonzero_every_update,
        "integrity_predictor_forward_count": integrity.predictor_forward_count == 4_000,
        "integrity_predictor_objective_count": integrity.predictor_objective_count == 4_000,
        "integrity_backward_count": integrity.backward_count == 4_000,
        "integrity_predictor_optimizer_updates": integrity.predictor_optimizer_update_count == 1_000,
        "integrity_forbidden_input_count": integrity.forbidden_input_count == 0,
        "integrity_bypass_count": integrity.bypass_count == 0,
        "integrity_forbidden_open_count": integrity.forbidden_open_count == 0,
        "integrity_current_latents_nonconstant": integrity.current_latents_nonconstant,
        "integrity_paired_latents_nonconstant": integrity.paired_latents_nonconstant,
        "integrity_latents_nonidentical": integrity.current_and_paired_latents_nonidentical,
        "integrity_one_step_zero_support": integrity.one_step_zero_support_witnessed,
        "integrity_corridor_masks_nonempty": integrity.all_corridor_masks_nonempty,
        "integrity_corridor_masks_inside_support": integrity.corridor_masks_inside_support,
        "semantic_balanced_accuracy": semantic.balanced_accuracy >= 0.80,
        "semantic_free_recall": semantic.free_recall >= 0.85,
        "semantic_occupied_recall": semantic.occupied_recall >= 0.70,
        "semantic_unknown_recall": semantic.unknown_recall >= 0.90,
        "semantic_rough_family_occupied_recall": semantic.rough_family_occupied_recall >= 0.65,
        "selection_has_eight_scenes": len(full.scenes) == 8,
        "selection_has_exact_registered_families": (
            set(full.families) == set(REGISTERED_FAMILIES)
        ),
        "corridor_station_precision": full.overall.station_safety_precision >= 0.99,
        "corridor_unsafe_recall": full.overall.station_unsafe_recall >= 0.95,
        "corridor_safe_recall": full.overall.station_safe_recall >= 0.90,
        "corridor_mean_primary_utility": full.overall.mean_primary_utility >= 0.90,
        "corridor_nonempty_selected_prefix": full.overall.nonempty_selected_prefix_rate >= 0.90,
        "corridor_selected_station_precision": full.overall.selected_station_precision >= 0.99,
        "corridor_action_share": max(full.overall.selected_action_shares)
        <= max(full.overall.oracle_action_shares) + 0.20,
    }
    for family, metrics in full.families.items():
        prefix = f"family:{family}"
        checks[f"{prefix}:station_denominators"] = (
            metrics.station_admitted_count > 0 and metrics.station_safe_count > 0
        )
        checks[f"{prefix}:admitted_count"] = metrics.station_admitted_count >= 32
        checks[f"{prefix}:safe_recall"] = metrics.station_safe_recall >= 0.70
        checks[f"{prefix}:unsafe_recall"] = metrics.station_unsafe_recall >= 0.70
        checks[f"{prefix}:nonempty_selected_prefix"] = (
            metrics.nonempty_selected_prefix_rate >= 0.80
        )
        checks[f"{prefix}:selected_station_precision"] = (
            metrics.selected_station_admitted_count > 0
            and metrics.selected_station_precision >= 0.95
        )
    comparisons = {
        name: compare_arms(full, evaluations[name]) for name in ARM_NAMES[1:]
    }
    for name, comparison in comparisons.items():
        checks[f"comparison:{name}:bootstrap_lower_positive"] = (
            comparison.bootstrap.lower_95 > 0.0
        )
        checks[f"comparison:{name}:six_positive_families"] = (
            comparison.positive_family_count >= 6
        )
    passed = all(checks.values())
    return GateDecision(
        status="PASS" if passed else "FAIL",
        passed=passed,
        checks=checks,
        failed_checks=tuple(name for name, value in checks.items() if not value),
        comparisons=comparisons,
    )
