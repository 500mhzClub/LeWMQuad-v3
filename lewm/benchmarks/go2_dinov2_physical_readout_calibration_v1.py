"""Pure evaluator for the preregistered DINOv2 physical-readout calibration.

This module performs no filesystem, RGB, Torch, encoder, or checkpoint access.
It consumes strict-loader group objects and already ordered frozen feature
caches.  Source/custody validation and the independent replay remain runner
responsibilities.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from lewm.benchmarks.go2_world_model_counterfactual_pilot_v1 import FAMILIES
from scripts.evaluate_go2_world_model_counterfactual_action_regret_v1 import (
    ActionSpecificRidgeReadoutsV1,
    RidgeReadoutV1,
    fit_ridge_readout_v1,
    predict_action_specific_scores_v1,
    selection_metrics_v1,
    task_conditioned_feature_v1,
)


SCHEMA = "lewm_go2_dinov2_physical_readout_calibration_v1"
ACTION_COUNT = 9
CONTEXT_FRAME_COUNT = 3
ROLE_STATE_COUNT = 128
ROLE_SCENE_COUNT = 16
ROLE_ARTIFACT_COUNT = ROLE_STATE_COUNT * (CONTEXT_FRAME_COUNT + ACTION_COUNT)
TOKEN_COUNT = 256
TOKEN_DIMENSION = 384
GRID_SIDE = 16
QUADRANT_SIDE = 8
DESCRIPTOR_DIMENSION = 3_072
RELATIONAL_DIMENSION = 3 * DESCRIPTOR_DIMENSION
RIDGE_LAMBDA = 1.0e-3
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2_026_080_302
HOLD_ACTION_ID = 6
SAFETY_STATUS = "NOT_TESTABLE_ZERO_EVENT_SUPPORT"
PASS_STATUS = "PASS_DINO_TRUE_FUTURE_PHYSICAL_READOUT_HEADROOM_ESTABLISHED"
STOP_STATUS = "STOP_DINO_TRUE_FUTURE_PHYSICAL_READOUT_HEADROOM_NOT_ESTABLISHED"
INFRASTRUCTURE_FAILURE_STATUS = "FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION"
SCIENTIFIC_GATE_NAMES = frozenset(
    {
        "2_privileged_physical_oracle",
        "3_true_future_beats_task_action_only",
        "4_true_future_beats_current_state",
        "5_true_future_beats_relational_persistence",
        "6_true_future_beats_random_expected",
    }
)


class DINOv2PhysicalReadoutCalibrationError(ValueError):
    """Raised when model-independent calibration inputs violate the contract."""


@dataclass(frozen=True)
class RoleStateIndexV1:
    role_state_index: int
    state_id: str
    role: str
    family: str
    scene_id: str
    group_index: int
    state_index_in_scene: int
    relative_target_xy_body_m: tuple[float, float]
    context_artifact_indices: tuple[int, int, int]
    target_artifact_indices: tuple[int, ...]
    dense_ranks: tuple[int, ...]
    target_progress_m: tuple[float, ...]
    physical_fell: tuple[bool, ...]
    physical_tipped: tuple[bool, ...]


@dataclass(frozen=True)
class RoleFeaturePlanV1:
    role: str
    artifact_ids: tuple[str, ...]
    artifact_index_by_id: Mapping[str, int]
    states: tuple[RoleStateIndexV1, ...]
    groups: tuple[Any, ...]
    identity_sha256: str


@dataclass(frozen=True)
class CalibrationFeaturePlansV1:
    train: RoleFeaturePlanV1
    eval: RoleFeaturePlanV1
    identity_sha256: str


@dataclass(frozen=True)
class CalibrationReadoutsV1:
    relational: ActionSpecificRidgeReadoutsV1
    current_state: ActionSpecificRidgeReadoutsV1
    task_action_only: ActionSpecificRidgeReadoutsV1
    identity_sha256: str


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise DINOv2PhysicalReadoutCalibrationError(f"{name} must be nonempty text")
    return value


def dinov2_quadrant_descriptor_v1(tokens: object) -> np.ndarray:
    """Return the exact float64 four-quadrant mean/population-std descriptor."""

    try:
        array = np.asarray(tokens)
    except (TypeError, ValueError) as exc:
        raise DINOv2PhysicalReadoutCalibrationError("DINO tokens must be numeric") from exc
    if array.shape != (TOKEN_COUNT, TOKEN_DIMENSION):
        raise DINOv2PhysicalReadoutCalibrationError(
            "DINO tokens must have exact shape (256,384)"
        )
    if array.dtype != np.float16:
        raise DINOv2PhysicalReadoutCalibrationError(
            "cached DINO tokens must use exact float16 storage"
        )
    promoted = array.astype(np.float64, copy=False)
    if not np.isfinite(promoted).all():
        raise DINOv2PhysicalReadoutCalibrationError("DINO tokens must be finite")
    grid = promoted.reshape(GRID_SIDE, GRID_SIDE, TOKEN_DIMENSION)
    quadrants = (
        grid[:QUADRANT_SIDE, :QUADRANT_SIDE],
        grid[:QUADRANT_SIDE, QUADRANT_SIDE:],
        grid[QUADRANT_SIDE:, :QUADRANT_SIDE],
        grid[QUADRANT_SIDE:, QUADRANT_SIDE:],
    )
    result = np.concatenate(
        [
            np.concatenate(
                [quadrant.mean(axis=(0, 1)), quadrant.std(axis=(0, 1), ddof=0)]
            )
            for quadrant in quadrants
        ]
    )
    if result.shape != (DESCRIPTOR_DIMENSION,) or not np.isfinite(result).all():
        raise DINOv2PhysicalReadoutCalibrationError("DINO descriptor is invalid")
    return np.ascontiguousarray(result, dtype=np.float64)


def relational_descriptor_v1(current: object, successor: object) -> np.ndarray:
    current_array = np.asarray(current, dtype=np.float64)
    successor_array = np.asarray(successor, dtype=np.float64)
    if (
        current_array.shape != (DESCRIPTOR_DIMENSION,)
        or successor_array.shape != (DESCRIPTOR_DIMENSION,)
        or not np.isfinite(current_array).all()
        or not np.isfinite(successor_array).all()
    ):
        raise DINOv2PhysicalReadoutCalibrationError(
            "current and successor descriptors must be finite 3072-vectors"
        )
    return np.concatenate(
        [current_array, successor_array, successor_array - current_array]
    )


def _labels(branch: object) -> tuple[float, bool, bool]:
    labels = getattr(branch, "labels", None)
    progress = float(getattr(labels, "target_progress_m", math.nan))
    fell = getattr(labels, "fell", None)
    tipped = getattr(labels, "tipped", None)
    if not math.isfinite(progress) or type(fell) is not bool or type(tipped) is not bool:
        raise DINOv2PhysicalReadoutCalibrationError("physical labels are malformed")
    return progress, fell, tipped


def build_role_feature_plan_v1(
    groups: Sequence[Any], *, role: str
) -> RoleFeaturePlanV1:
    """Index one strict-loader role using metadata only; no RGB leaf is opened."""

    if role not in {"train", "eval"}:
        raise DINOv2PhysicalReadoutCalibrationError("role must be train or eval")
    if len(groups) != ROLE_STATE_COUNT:
        raise DINOv2PhysicalReadoutCalibrationError(
            f"{role} must contain exactly {ROLE_STATE_COUNT} states"
        )
    try:
        ordered = tuple(
            sorted(groups, key=lambda group: (int(group.group_index), str(group.state_id)))
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise DINOv2PhysicalReadoutCalibrationError("group ordering is malformed") from exc

    artifact_ids: list[str] = []
    artifact_index_by_id: dict[str, int] = {}
    states: list[RoleStateIndexV1] = []
    seen_states: set[str] = set()
    seen_group_indices: set[int] = set()

    def append_artifact(value: object) -> int:
        artifact_id = _text(value, name="RGB artifact ID")
        if artifact_id in artifact_index_by_id:
            raise DINOv2PhysicalReadoutCalibrationError(
                f"{role} artifact is reused across state slots"
            )
        index = len(artifact_ids)
        artifact_ids.append(artifact_id)
        artifact_index_by_id[artifact_id] = index
        return index

    for role_state_index, group in enumerate(ordered):
        if getattr(group, "role", None) != role:
            raise DINOv2PhysicalReadoutCalibrationError("group crossed role boundary")
        state_id = _text(getattr(group, "state_id", None), name="state ID")
        family = _text(getattr(group, "family", None), name="family")
        scene_id = _text(getattr(group, "scene_id", None), name="scene ID")
        try:
            group_index = int(group.group_index)
            state_index_in_scene = int(group.state_index_in_scene)
        except (AttributeError, TypeError, ValueError) as exc:
            raise DINOv2PhysicalReadoutCalibrationError("state indices are malformed") from exc
        if (
            state_id in seen_states
            or group_index in seen_group_indices
            or group_index < 0
            or state_index_in_scene < 0
        ):
            raise DINOv2PhysicalReadoutCalibrationError("state identity repeats")
        seen_states.add(state_id)
        seen_group_indices.add(group_index)
        target = np.asarray(getattr(group, "relative_target_xy_body_m", ()), dtype=np.float64)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise DINOv2PhysicalReadoutCalibrationError("task target is malformed")
        contexts = tuple(getattr(group, "context_rgb_artifact_ids", ()))
        branches = tuple(getattr(group, "branches", ()))
        if len(contexts) != CONTEXT_FRAME_COUNT or len(branches) != ACTION_COUNT:
            raise DINOv2PhysicalReadoutCalibrationError("group geometry changed")
        try:
            branches = tuple(sorted(branches, key=lambda branch: int(branch.action_id)))
        except (AttributeError, TypeError, ValueError) as exc:
            raise DINOv2PhysicalReadoutCalibrationError("branch actions are malformed") from exc
        if tuple(getattr(branch, "action_id", None) for branch in branches) != tuple(
            range(ACTION_COUNT)
        ):
            raise DINOv2PhysicalReadoutCalibrationError(
                "branches must be ordered by the exact nine requested actions"
            )
        context_indices = tuple(append_artifact(value) for value in contexts)
        target_indices = tuple(
            append_artifact(getattr(branch, "target_rgb_artifact_id", None))
            for branch in branches
        )
        ranks = tuple(getattr(branch, "oracle_dense_rank", None) for branch in branches)
        if any(type(rank) is not int or rank < 0 for rank in ranks) or max(ranks) <= 0:
            raise DINOv2PhysicalReadoutCalibrationError("dense physical ranks are invalid")
        physical = tuple(_labels(branch) for branch in branches)
        states.append(
            RoleStateIndexV1(
                role_state_index=role_state_index,
                state_id=state_id,
                role=role,
                family=family,
                scene_id=scene_id,
                group_index=group_index,
                state_index_in_scene=state_index_in_scene,
                relative_target_xy_body_m=(float(target[0]), float(target[1])),
                context_artifact_indices=context_indices,  # type: ignore[arg-type]
                target_artifact_indices=target_indices,
                dense_ranks=ranks,  # type: ignore[arg-type]
                target_progress_m=tuple(item[0] for item in physical),
                physical_fell=tuple(item[1] for item in physical),
                physical_tipped=tuple(item[2] for item in physical),
            )
        )

    families = Counter(state.family for state in states)
    scenes = {(state.family, state.scene_id) for state in states}
    if set(families) != set(FAMILIES) or any(
        families[family] != ROLE_STATE_COUNT // len(FAMILIES) for family in FAMILIES
    ):
        raise DINOv2PhysicalReadoutCalibrationError("role family balance changed")
    scenes_by_family = Counter(family for family, _scene in scenes)
    if len(scenes) != ROLE_SCENE_COUNT or any(
        scenes_by_family[family] != 2 for family in FAMILIES
    ):
        raise DINOv2PhysicalReadoutCalibrationError("role scene balance changed")
    if len(artifact_ids) != ROLE_ARTIFACT_COUNT:
        raise DINOv2PhysicalReadoutCalibrationError("role artifact count changed")
    identity_document = {
        "role": role,
        "artifact_ids": artifact_ids,
        "states": [
            {
                "state_id": state.state_id,
                "family": state.family,
                "scene_id": state.scene_id,
                "group_index": state.group_index,
                "state_index_in_scene": state.state_index_in_scene,
                "target": list(state.relative_target_xy_body_m),
                "contexts": list(state.context_artifact_indices),
                "targets": list(state.target_artifact_indices),
                "dense_ranks": list(state.dense_ranks),
            }
            for state in states
        ],
    }
    return RoleFeaturePlanV1(
        role=role,
        artifact_ids=tuple(artifact_ids),
        artifact_index_by_id=MappingProxyType(artifact_index_by_id),
        states=tuple(states),
        groups=ordered,
        identity_sha256=hashlib.sha256(_canonical_bytes(identity_document)).hexdigest(),
    )


def build_calibration_feature_plans_v1(
    train_groups: Sequence[Any], eval_groups: Sequence[Any]
) -> CalibrationFeaturePlansV1:
    train = build_role_feature_plan_v1(train_groups, role="train")
    evaluation = build_role_feature_plan_v1(eval_groups, role="eval")
    if (
        set(train.artifact_ids) & set(evaluation.artifact_ids)
        or {state.state_id for state in train.states}
        & {state.state_id for state in evaluation.states}
        or {state.scene_id for state in train.states}
        & {state.scene_id for state in evaluation.states}
    ):
        raise DINOv2PhysicalReadoutCalibrationError(
            "train and eval identities must be disjoint"
        )
    digest = hashlib.sha256(
        bytes.fromhex(train.identity_sha256) + bytes.fromhex(evaluation.identity_sha256)
    ).hexdigest()
    return CalibrationFeaturePlansV1(train=train, eval=evaluation, identity_sha256=digest)


def _descriptor_cache(plan: RoleFeaturePlanV1, features: Sequence[Any]) -> tuple[np.ndarray, ...]:
    if len(features) != len(plan.artifact_ids):
        raise DINOv2PhysicalReadoutCalibrationError("feature cache length changed")
    return tuple(dinov2_quadrant_descriptor_v1(features[index]) for index in range(len(features)))


def _assemble_heads(heads: Sequence[RidgeReadoutV1]) -> ActionSpecificRidgeReadoutsV1:
    if len(heads) != ACTION_COUNT:
        raise DINOv2PhysicalReadoutCalibrationError("readout requires nine heads")
    digest = hashlib.sha256()
    for action_id, head in enumerate(heads):
        digest.update(action_id.to_bytes(2, "little"))
        digest.update(bytes.fromhex(head.identity_sha256))
    return ActionSpecificRidgeReadoutsV1(tuple(heads), digest.hexdigest())


def _fit_heads(
    plan: RoleFeaturePlanV1,
    descriptors: Sequence[np.ndarray],
    feature_kind: str,
) -> ActionSpecificRidgeReadoutsV1:
    heads: list[RidgeReadoutV1] = []
    for action_id in range(ACTION_COUNT):
        features: list[np.ndarray] = []
        targets: list[float] = []
        for state in plan.states:
            current = descriptors[state.context_artifact_indices[-1]]
            if feature_kind == "relational":
                latent: object | None = relational_descriptor_v1(
                    current, descriptors[state.target_artifact_indices[action_id]]
                )
            elif feature_kind == "current_state":
                latent = current
            elif feature_kind == "task_action_only":
                latent = None
            else:
                raise AssertionError("unknown readout feature kind")
            features.append(
                task_conditioned_feature_v1(
                    latent, relative_target_xy_body_m=state.relative_target_xy_body_m
                )
            )
            targets.append(state.dense_ranks[action_id] / max(state.dense_ranks))
        heads.append(
            fit_ridge_readout_v1(
                np.stack(features), np.asarray(targets), ridge_lambda=RIDGE_LAMBDA
            )
        )
    return _assemble_heads(heads)


def fit_calibration_readouts_v1(
    train_plan: RoleFeaturePlanV1, train_features: Sequence[Any]
) -> CalibrationReadoutsV1:
    if train_plan.role != "train":
        raise DINOv2PhysicalReadoutCalibrationError("readouts must fit train role")
    descriptors = _descriptor_cache(train_plan, train_features)
    relational = _fit_heads(train_plan, descriptors, "relational")
    current = _fit_heads(train_plan, descriptors, "current_state")
    task = _fit_heads(train_plan, descriptors, "task_action_only")
    digest = hashlib.sha256()
    for name, value in (
        ("relational", relational),
        ("current_state", current),
        ("task_action_only", task),
    ):
        digest.update(name.encode("ascii") + b"\0")
        digest.update(bytes.fromhex(value.identity_sha256))
    return CalibrationReadoutsV1(relational, current, task, digest.hexdigest())


def _augment_report(report: dict[str, object]) -> dict[str, object]:
    rows = report["group_results"]
    assert isinstance(rows, list)

    def summarize(selected: Sequence[Mapping[str, object]]) -> dict[str, object]:
        return {
            "states": len(selected),
            "normalized_rank_regret": float(np.mean([row["normalized_rank_regret"] for row in selected])),
            "oracle_equivalent_selection_rate": float(np.mean([row["oracle_match"] for row in selected])),
            "physical_target_progress_m": float(np.mean([row["physical_target_progress_m"] for row in selected])),
            "physical_path_length_m": float(np.mean([row["physical_path_length_m"] for row in selected])),
            "chosen_action_histogram": {
                str(action): sum(int(row["selected_action_id"]) == action for row in selected)
                for action in range(ACTION_COUNT)
            },
        }

    summary = report["summary"]
    assert isinstance(summary, dict)
    summary["oracle_equivalent_selection_rate"] = summary["oracle_match_rate"]
    summary["chosen_action_histogram"] = summarize(rows)["chosen_action_histogram"]
    report["per_family"] = {
        family: summarize([row for row in rows if row["family"] == family])
        for family in FAMILIES
    }
    report["per_scene"] = [
        {
            "scene_id": scene,
            "family": next(str(row["family"]) for row in rows if row["scene_id"] == scene),
            **summarize([row for row in rows if row["scene_id"] == scene]),
        }
        for scene in sorted({str(row["scene_id"]) for row in rows})
    ]
    return report


def _random_expected_report(plan: RoleFeaturePlanV1) -> dict[str, object]:
    rows = []
    for state in plan.states:
        ranks = np.asarray(state.dense_ranks, dtype=np.float64)
        rows.append(
            {
                "state_id": state.state_id,
                "scene_id": state.scene_id,
                "family": state.family,
                "selected_action_id": "NOT_APPLICABLE",
                "normalized_rank_regret": float(ranks.mean() / ranks.max()),
                "oracle_equivalent_selection_rate": float((ranks == ranks.min()).mean()),
                "physical_target_progress_m": "NOT_APPLICABLE",
                "physical_path_length_m": "NOT_APPLICABLE",
            }
        )

    def summarize(selected: Sequence[Mapping[str, object]]) -> dict[str, object]:
        return {
            "states": len(selected),
            "normalized_rank_regret": float(np.mean([row["normalized_rank_regret"] for row in selected])),
            "oracle_equivalent_selection_rate": float(np.mean([row["oracle_equivalent_selection_rate"] for row in selected])),
            "physical_target_progress_m": "NOT_APPLICABLE",
            "physical_path_length_m": "NOT_APPLICABLE",
            "chosen_action_histogram": "NOT_APPLICABLE",
        }

    return {
        "selection_policy": "uniform_random_expectation_no_realized_action",
        "summary": summarize(rows),
        "group_results": rows,
        "per_family": {
            family: summarize([row for row in rows if row["family"] == family])
            for family in FAMILIES
        },
        "per_scene": [
            {
                "scene_id": scene,
                "family": next(str(row["family"]) for row in rows if row["scene_id"] == scene),
                **summarize([row for row in rows if row["scene_id"] == scene]),
            }
            for scene in sorted({str(row["scene_id"]) for row in rows})
        ],
    }


def score_calibration_arms_v1(
    eval_plan: RoleFeaturePlanV1,
    eval_features: Sequence[Any],
    readouts: CalibrationReadoutsV1,
) -> dict[str, dict[str, object]]:
    if eval_plan.role != "eval":
        raise DINOv2PhysicalReadoutCalibrationError("scores must use eval role")
    descriptors = _descriptor_cache(eval_plan, eval_features)
    score_maps: dict[str, dict[str, list[float]]] = {
        name: {} for name in (
            "privileged_physical_oracle",
            "dinov2_true_future",
            "dinov2_current_state",
            "task_action_only",
            "relational_persistence",
            "hold_constant",
        )
    }
    for state in eval_plan.states:
        current = descriptors[state.context_artifact_indices[-1]]
        task = state.relative_target_xy_body_m
        true_features = [
            task_conditioned_feature_v1(
                relational_descriptor_v1(current, descriptors[state.target_artifact_indices[action]]),
                relative_target_xy_body_m=task,
            )
            for action in range(ACTION_COUNT)
        ]
        current_features = [
            task_conditioned_feature_v1(current, relative_target_xy_body_m=task)
            for _action in range(ACTION_COUNT)
        ]
        task_features = [
            task_conditioned_feature_v1(None, relative_target_xy_body_m=task)
            for _action in range(ACTION_COUNT)
        ]
        persistence = relational_descriptor_v1(current, current)
        persistence_features = [
            task_conditioned_feature_v1(persistence, relative_target_xy_body_m=task)
            for _action in range(ACTION_COUNT)
        ]
        score_maps["privileged_physical_oracle"][state.state_id] = [float(value) for value in state.dense_ranks]
        score_maps["dinov2_true_future"][state.state_id] = predict_action_specific_scores_v1(readouts.relational, true_features).tolist()
        score_maps["dinov2_current_state"][state.state_id] = predict_action_specific_scores_v1(readouts.current_state, current_features).tolist()
        score_maps["task_action_only"][state.state_id] = predict_action_specific_scores_v1(readouts.task_action_only, task_features).tolist()
        score_maps["relational_persistence"][state.state_id] = predict_action_specific_scores_v1(readouts.relational, persistence_features).tolist()
        score_maps["hold_constant"][state.state_id] = [0.0 if action == HOLD_ACTION_ID else 1.0 for action in range(ACTION_COUNT)]
    reports = {
        name: _augment_report(selection_metrics_v1(eval_plan.groups, scores))
        for name, scores in score_maps.items()
    }
    reports["random_expected"] = _random_expected_report(eval_plan)
    return reports


def paired_family_scene_cluster_comparison_v1(
    candidate_results: Sequence[Mapping[str, object]],
    baseline_results: Sequence[Mapping[str, object]],
    *,
    field: str = "normalized_rank_regret",
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    if resamples <= 0:
        raise DINOv2PhysicalReadoutCalibrationError("resamples must be positive")
    candidate = {str(row["state_id"]): row for row in candidate_results}
    baseline = {str(row["state_id"]): row for row in baseline_results}
    if not candidate or set(candidate) != set(baseline):
        raise DINOv2PhysicalReadoutCalibrationError("paired state identities changed")
    by_scene: dict[tuple[str, str], list[float]] = {}
    for state_id in sorted(candidate):
        left, right = candidate[state_id], baseline[state_id]
        if left["scene_id"] != right["scene_id"] or left["family"] != right["family"]:
            raise DINOv2PhysicalReadoutCalibrationError("paired scene identity changed")
        key = (str(left["family"]), str(left["scene_id"]))
        delta = float(left[field]) - float(right[field])
        if not math.isfinite(delta):
            raise DINOv2PhysicalReadoutCalibrationError("paired metric is nonfinite")
        by_scene.setdefault(key, []).append(delta)
    by_family: dict[str, list[float]] = {family: [] for family in FAMILIES}
    for (family, _scene), values in sorted(by_scene.items()):
        if family not in by_family:
            raise DINOv2PhysicalReadoutCalibrationError("unexpected family")
        by_family[family].append(float(np.mean(values)))
    if any(len(by_family[family]) != 2 for family in FAMILIES):
        raise DINOv2PhysicalReadoutCalibrationError("each family must have two scenes")
    rng = np.random.default_rng(seed)
    draws = []
    family_points: dict[str, float] = {}
    for family in FAMILIES:
        values = np.asarray(by_family[family], dtype=np.float64)
        family_points[family] = float(values.mean())
        indices = rng.integers(0, len(values), size=(resamples, len(values)))
        draws.append(values[indices].mean(axis=1))
    samples = np.stack(draws, axis=1).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "field": field,
        "direction": "candidate_minus_baseline_lower_is_better",
        "paired_states": len(candidate),
        "scene_clusters": len(by_scene),
        "family_strata": len(FAMILIES),
        "scenes_per_family": {family: len(by_family[family]) for family in FAMILIES},
        "resamples": resamples,
        "seed": seed,
        "mean_delta": float(np.mean(list(family_points.values()))),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "mean_delta_by_family": family_points,
    }


def _readout_report(readouts: CalibrationReadoutsV1) -> dict[str, object]:
    def one(value: ActionSpecificRidgeReadoutsV1) -> dict[str, object]:
        return {
            "identity_sha256": value.identity_sha256,
            "heads": [
                {
                    "action_id": action,
                    "identity_sha256": head.identity_sha256,
                    "training_rows": head.training_rows,
                    "feature_dimension": int(head.feature_mean.size),
                    "ridge_lambda": head.ridge_lambda,
                    "solver": head.solver,
                }
                for action, head in enumerate(value.heads)
            ],
        }
    return {
        "identity_sha256": readouts.identity_sha256,
        "relational": one(readouts.relational),
        "current_state": one(readouts.current_state),
        "task_action_only": one(readouts.task_action_only),
    }


def _safety_support(plans: CalibrationFeaturePlansV1) -> dict[str, object]:
    result: dict[str, object] = {
        "status": SAFETY_STATUS,
        "applicable": False,
        "passed": False,
        "claim": "NOT_APPLICABLE",
    }
    for role, plan in (("train", plans.train), ("eval", plans.eval)):
        falls = sum(sum(state.physical_fell) for state in plan.states)
        tips = sum(sum(state.physical_tipped) for state in plan.states)
        if falls or tips:
            raise DINOv2PhysicalReadoutCalibrationError(
                "zero-event safety-support contract changed"
            )
        result[role] = {
            "branches": len(plan.states) * ACTION_COUNT,
            "falls": falls,
            "tips": tips,
        }
    return result


def evaluate_calibration_v1(
    train_groups: Sequence[Any],
    eval_groups: Sequence[Any],
    train_features: Sequence[Any],
    eval_features: Sequence[Any],
) -> dict[str, object]:
    """Fit train-only readouts and return JSON-safe eval evidence and gates 2--6."""

    plans = build_calibration_feature_plans_v1(train_groups, eval_groups)
    readouts = fit_calibration_readouts_v1(plans.train, train_features)
    arms = score_calibration_arms_v1(plans.eval, eval_features, readouts)
    true_rows = arms["dinov2_true_future"]["group_results"]
    comparisons = {
        name: paired_family_scene_cluster_comparison_v1(
            true_rows, arms[baseline]["group_results"]
        )
        for name, baseline in (
            ("true_future_vs_task_action_only", "task_action_only"),
            ("true_future_vs_current_state", "dinov2_current_state"),
            ("true_future_vs_relational_persistence", "relational_persistence"),
        )
    }
    oracle_summary = arms["privileged_physical_oracle"]["summary"]
    true_summary = arms["dinov2_true_future"]["summary"]
    random_summary = arms["random_expected"]["summary"]
    gates = {
        "2_privileged_physical_oracle": {
            "passed": oracle_summary["normalized_rank_regret"] == 0.0
            and oracle_summary["oracle_equivalent_selection_rate"] == 1.0,
            "normalized_rank_regret": oracle_summary["normalized_rank_regret"],
            "oracle_equivalent_selection_rate": oracle_summary["oracle_equivalent_selection_rate"],
        },
        "3_true_future_beats_task_action_only": {
            "passed": comparisons["true_future_vs_task_action_only"]["upper_95"] < 0.0,
            "measurement": comparisons["true_future_vs_task_action_only"],
        },
        "4_true_future_beats_current_state": {
            "passed": comparisons["true_future_vs_current_state"]["upper_95"] < 0.0,
            "measurement": comparisons["true_future_vs_current_state"],
        },
        "5_true_future_beats_relational_persistence": {
            "passed": comparisons["true_future_vs_relational_persistence"]["upper_95"] < 0.0,
            "measurement": comparisons["true_future_vs_relational_persistence"],
        },
        "6_true_future_beats_random_expected": {
            "passed": true_summary["normalized_rank_regret"] < random_summary["normalized_rank_regret"],
            "true_future": true_summary["normalized_rank_regret"],
            "random_expected": random_summary["normalized_rank_regret"],
            "per_family_true_minus_random": {
                family: arms["dinov2_true_future"]["per_family"][family]["normalized_rank_regret"]
                - arms["random_expected"]["per_family"][family]["normalized_rank_regret"]
                for family in FAMILIES
            },
        },
    }
    result: dict[str, object] = {
        "schema": SCHEMA,
        "status": "COMPLETE_MODEL_INDEPENDENT_EVALUATION",
        "claim_scope": "DEVELOPMENT_ONLY_EVALUATOR_CALIBRATION",
        "feature_plan": {
            "identity_sha256": plans.identity_sha256,
            "train_identity_sha256": plans.train.identity_sha256,
            "eval_identity_sha256": plans.eval.identity_sha256,
            "states_per_role": ROLE_STATE_COUNT,
            "artifacts_per_role": ROLE_ARTIFACT_COUNT,
        },
        "descriptor_contract": {
            "token_shape": [TOKEN_COUNT, TOKEN_DIMENSION],
            "storage_dtype": "float16",
            "statistics_dtype": "float64",
            "quadrants": ["top_left", "top_right", "bottom_left", "bottom_right"],
            "standard_deviation_ddof": 0,
            "descriptor_dimension": DESCRIPTOR_DIMENSION,
            "relational_definition": ["current", "successor", "successor_minus_current"],
            "relational_dimension": RELATIONAL_DIMENSION,
        },
        "readouts": _readout_report(readouts),
        "arms": arms,
        "paired_scene_cluster_comparisons": comparisons,
        "safety": _safety_support(plans),
        "gates": gates,
        "scientific_gates_2_to_6_passed": all(bool(gate["passed"]) for gate in gates.values()),
    }
    result["replay_identity_sha256"] = calibration_replay_identity_v1(result)
    return result


def calibration_replay_identity_v1(evaluation: Mapping[str, object]) -> str:
    document = dict(evaluation)
    document.pop("replay_identity_sha256", None)
    return hashlib.sha256(_canonical_bytes(document)).hexdigest()


def calibration_verdict_v1(
    evaluation: Mapping[str, object],
    *,
    infrastructure_checks_passed: bool,
    deterministic_replay_passed: bool,
) -> dict[str, object]:
    if evaluation.get("schema") != SCHEMA:
        raise DINOv2PhysicalReadoutCalibrationError("evaluation schema changed")
    if type(infrastructure_checks_passed) is not bool or type(deterministic_replay_passed) is not bool:
        raise DINOv2PhysicalReadoutCalibrationError("external gate values must be bool")
    scientific_gates = evaluation.get("gates")
    if (
        not isinstance(scientific_gates, Mapping)
        or set(scientific_gates) != SCIENTIFIC_GATE_NAMES
        or any(
            not isinstance(gate, Mapping) or type(gate.get("passed")) is not bool
            for gate in scientific_gates.values()
        )
    ):
        raise DINOv2PhysicalReadoutCalibrationError(
            "registered scientific gates changed"
        )
    scientific_passed = all(bool(gate["passed"]) for gate in scientific_gates.values())
    if evaluation.get("scientific_gates_2_to_6_passed") is not scientific_passed:
        raise DINOv2PhysicalReadoutCalibrationError(
            "scientific gate aggregate changed"
        )
    gates = {
        "1_infrastructure_and_custody": {"passed": infrastructure_checks_passed},
        **dict(scientific_gates),
        "7_deterministic_replay": {"passed": deterministic_replay_passed},
    }
    passed = all(bool(gate["passed"]) for gate in gates.values())
    if not infrastructure_checks_passed or not deterministic_replay_passed:
        terminal_status = INFRASTRUCTURE_FAILURE_STATUS
    elif passed:
        terminal_status = PASS_STATUS
    else:
        terminal_status = STOP_STATUS
    return {"gates": gates, "passed": passed, "terminal_status": terminal_status}


__all__ = [
    "ACTION_COUNT",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CalibrationFeaturePlansV1",
    "CalibrationReadoutsV1",
    "DINOv2PhysicalReadoutCalibrationError",
    "RoleFeaturePlanV1",
    "RoleStateIndexV1",
    "SCHEMA",
    "SCIENTIFIC_GATE_NAMES",
    "build_calibration_feature_plans_v1",
    "build_role_feature_plan_v1",
    "calibration_replay_identity_v1",
    "calibration_verdict_v1",
    "dinov2_quadrant_descriptor_v1",
    "evaluate_calibration_v1",
    "fit_calibration_readouts_v1",
    "paired_family_scene_cluster_comparison_v1",
    "relational_descriptor_v1",
    "score_calibration_arms_v1",
]
