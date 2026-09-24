"""Scene-diversity replication of the frozen recurrent H1 experiment.

Only the data role changes here.  Model construction, optimization, projections,
scoring, bootstrap, and gate thresholds are delegated to the frozen task-coupled
recurrent benchmark.  In particular this module does not provide a route for
successor observations.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import math
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from lewm.benchmarks import go2_grounded_dense_dino_joint_jepa_v1 as grounded
from lewm.benchmarks import go2_matched_branch_physical_outcome_screen_v1 as physical
from lewm.benchmarks import go2_task_coupled_recurrent_dynamics_v1 as frozen
from lewm.benchmarks import go2_dinov2_physical_readout_calibration_v1 as calibration


SCHEMA = "lewm_go2_scene_diversity_recurrent_replication_v1"
PASS_STATUS = "PASS_SCENE_DIVERSITY_RECURRENT_REPLICATION_H1"
STOP_STATUS = "STOP_SCENE_DIVERSITY_RECURRENT_REPLICATION_H1"
TASK_CONTROL_ASSERTION_STATUS = "NOT_APPLICABLE_NEW_SCENE_ROLES"

STATE_COUNT = frozen.STATE_COUNT
ACTION_COUNT = frozen.ACTION_COUNT
ARM_ORDER = frozen.ARM_ORDER
VISUAL_ARM = frozen.VISUAL_ARM
NO_VISION_ARM = frozen.NO_VISION_ARM
MODEL_SEEDS = frozen.MODEL_SEEDS
SAMPLER_SEED = frozen.SAMPLER_SEED
UPDATES = frozen.UPDATES
BATCH_STATES = frozen.BATCH_STATES


class SceneDiversityRecurrentReplicationError(RuntimeError):
    """Raised when the fixed replication contract changes."""


def config_v1() -> dict[str, object]:
    """Return the data intervention plus the byte-for-byte scientific config."""

    return {
        "schema": SCHEMA,
        "frozen_recurrent_protocol": frozen.config_v1(),
        "data_intervention": {
            "scenes_per_role": 32,
            "scenes_per_family_per_role": 4,
            "states_per_scene": 4,
            "states_per_role": 128,
            "total_scenes": 64,
            "total_states": 256,
        },
    }


def validate_role_scene_geometry_v1(role: Any) -> dict[str, object]:
    """Retain frozen tensor checks and require the new scene intervention."""

    frozen._validate_role_v1(role)  # noqa: SLF001
    states = tuple(getattr(getattr(role, "plan", None), "states", ()))
    if len(states) != STATE_COUNT:
        raise SceneDiversityRecurrentReplicationError("role state count changed")
    scene_rows: dict[str, list[Any]] = {}
    family_scenes: dict[str, set[str]] = {}
    state_ids: set[str] = set()
    for state in states:
        state_id = str(getattr(state, "state_id", ""))
        scene_id = str(getattr(state, "scene_id", ""))
        family = str(getattr(state, "family", ""))
        if not state_id or not scene_id or not family or state_id in state_ids:
            raise SceneDiversityRecurrentReplicationError("role state identity changed")
        state_ids.add(state_id)
        scene_rows.setdefault(scene_id, []).append(state)
        family_scenes.setdefault(family, set()).add(scene_id)
    if (
        len(scene_rows) != 32
        or any(len(rows) != 4 for rows in scene_rows.values())
        or len(family_scenes) != 8
        or any(len(scenes) != 4 for scenes in family_scenes.values())
    ):
        raise SceneDiversityRecurrentReplicationError(
            "role must contain 4 states in each of 4 scenes per family"
        )
    return {
        "role": str(role.role),
        "state_count": len(states),
        "scene_count": len(scene_rows),
        "family_count": len(family_scenes),
        "states_per_scene": 4,
        "scenes_per_family": 4,
        "scene_ids": sorted(scene_rows),
    }


def build_role_feature_plan_v1(
    groups: Sequence[Any], *, role: str
) -> calibration.RoleFeaturePlanV1:
    """Build the frozen role index with only its scene-count check generalized."""

    if role not in {"train", "eval"} or len(groups) != STATE_COUNT:
        raise SceneDiversityRecurrentReplicationError("role state geometry changed")
    try:
        ordered = tuple(
            sorted(groups, key=lambda group: (int(group.group_index), str(group.state_id)))
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise SceneDiversityRecurrentReplicationError("group ordering is malformed") from error
    artifact_ids: list[str] = []
    artifact_index_by_id: dict[str, int] = {}
    states: list[calibration.RoleStateIndexV1] = []
    seen_states: set[str] = set()
    seen_group_indices: set[int] = set()

    def append_artifact(value: object) -> int:
        try:
            artifact_id = calibration._text(value, name="RGB artifact ID")  # noqa: SLF001
        except Exception as error:
            raise SceneDiversityRecurrentReplicationError(
                "RGB artifact ID changed"
            ) from error
        if artifact_id in artifact_index_by_id:
            raise SceneDiversityRecurrentReplicationError(
                "artifact is reused across state slots"
            )
        index = len(artifact_ids)
        artifact_ids.append(artifact_id)
        artifact_index_by_id[artifact_id] = index
        return index

    for role_state_index, group in enumerate(ordered):
        if getattr(group, "role", None) != role:
            raise SceneDiversityRecurrentReplicationError("group crossed role boundary")
        try:
            state_id = calibration._text(group.state_id, name="state ID")  # noqa: SLF001
            family = calibration._text(group.family, name="family")  # noqa: SLF001
            scene_id = calibration._text(group.scene_id, name="scene ID")  # noqa: SLF001
            group_index = int(group.group_index)
            state_index_in_scene = int(group.state_index_in_scene)
        except (AttributeError, TypeError, ValueError) as error:
            raise SceneDiversityRecurrentReplicationError(
                "state identity is malformed"
            ) from error
        if (
            state_id in seen_states
            or group_index in seen_group_indices
            or group_index < 0
            or state_index_in_scene < 0
        ):
            raise SceneDiversityRecurrentReplicationError("state identity repeats")
        seen_states.add(state_id)
        seen_group_indices.add(group_index)
        target = np.asarray(group.relative_target_xy_body_m, dtype=np.float64)
        contexts = tuple(group.context_rgb_artifact_ids)
        branches = tuple(group.branches)
        if (
            target.shape != (2,)
            or not np.isfinite(target).all()
            or len(contexts) != calibration.CONTEXT_FRAME_COUNT
            or len(branches) != ACTION_COUNT
        ):
            raise SceneDiversityRecurrentReplicationError("group geometry changed")
        try:
            branches = tuple(sorted(branches, key=lambda branch: int(branch.action_id)))
        except (AttributeError, TypeError, ValueError) as error:
            raise SceneDiversityRecurrentReplicationError(
                "branch actions are malformed"
            ) from error
        if tuple(branch.action_id for branch in branches) != tuple(range(ACTION_COUNT)):
            raise SceneDiversityRecurrentReplicationError(
                "branches must contain the exact nine actions"
            )
        context_indices = tuple(append_artifact(value) for value in contexts)
        target_indices = tuple(
            append_artifact(branch.target_rgb_artifact_id) for branch in branches
        )
        ranks = tuple(branch.oracle_dense_rank for branch in branches)
        if any(type(rank) is not int or rank < 0 for rank in ranks) or max(ranks) <= 0:
            raise SceneDiversityRecurrentReplicationError("dense ranks are invalid")
        try:
            labels = tuple(calibration._labels(branch) for branch in branches)  # noqa: SLF001
        except Exception as error:
            raise SceneDiversityRecurrentReplicationError(
                "physical labels are malformed"
            ) from error
        states.append(
            calibration.RoleStateIndexV1(
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
                dense_ranks=ranks,
                target_progress_m=tuple(item[0] for item in labels),
                physical_fell=tuple(item[1] for item in labels),
                physical_tipped=tuple(item[2] for item in labels),
            )
        )
    families = Counter(state.family for state in states)
    scenes = {(state.family, state.scene_id) for state in states}
    scenes_by_family = Counter(family for family, _scene in scenes)
    expected_families = set(calibration.FAMILIES)
    if (
        set(families) != expected_families
        or any(families[family] != 16 for family in calibration.FAMILIES)
        or len(scenes) != 32
        or any(scenes_by_family[family] != 4 for family in calibration.FAMILIES)
        or len(artifact_ids) != STATE_COUNT * (3 + ACTION_COUNT)
    ):
        raise SceneDiversityRecurrentReplicationError("role balance changed")
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
    return calibration.RoleFeaturePlanV1(
        role=role,
        artifact_ids=tuple(artifact_ids),
        artifact_index_by_id=MappingProxyType(artifact_index_by_id),
        states=tuple(states),
        groups=ordered,
        identity_sha256=hashlib.sha256(
            frozen.canonical_bytes_v1(identity_document)
        ).hexdigest(),
    )


def paired_family_scene_bootstrap_v1(
    candidate_results: Sequence[Mapping[str, object]],
    baseline_results: Sequence[Mapping[str, object]],
    *,
    field: str = "normalized_rank_regret",
) -> dict[str, object]:
    """Run the frozen family-equal whole-scene bootstrap over four scenes."""

    candidate = {str(row["state_id"]): row for row in candidate_results}
    baseline = {str(row["state_id"]): row for row in baseline_results}
    if not candidate or set(candidate) != set(baseline):
        raise SceneDiversityRecurrentReplicationError("paired state identities changed")
    by_scene: dict[tuple[str, str], list[float]] = {}
    for state_id in sorted(candidate):
        left, right = candidate[state_id], baseline[state_id]
        if left["scene_id"] != right["scene_id"] or left["family"] != right["family"]:
            raise SceneDiversityRecurrentReplicationError("paired scene identity changed")
        delta = float(left[field]) - float(right[field])
        if not math.isfinite(delta):
            raise SceneDiversityRecurrentReplicationError("paired metric is nonfinite")
        key = (str(left["family"]), str(left["scene_id"]))
        by_scene.setdefault(key, []).append(delta)
    by_family: dict[str, list[float]] = {
        family: [] for family in calibration.FAMILIES
    }
    for (family, _scene), values in sorted(by_scene.items()):
        if family not in by_family:
            raise SceneDiversityRecurrentReplicationError("unexpected family")
        by_family[family].append(float(np.mean(values)))
    if any(len(by_family[family]) != 4 for family in calibration.FAMILIES):
        raise SceneDiversityRecurrentReplicationError(
            "each family must have four scenes"
        )
    resamples = grounded.BOOTSTRAP_RESAMPLES
    seed = grounded.BOOTSTRAP_SEED
    rng = np.random.default_rng(seed)
    draws = []
    family_points: dict[str, float] = {}
    for family in calibration.FAMILIES:
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
        "family_strata": len(calibration.FAMILIES),
        "scenes_per_family": {
            family: len(by_family[family]) for family in calibration.FAMILIES
        },
        "resamples": resamples,
        "seed": seed,
        "mean_delta": float(np.mean(list(family_points.values()))),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "mean_delta_by_family": family_points,
    }


def fit_checkpoint_v1(
    train_role: Any, train_context_tokens: torch.Tensor, *, device: torch.device
) -> dict[str, object]:
    """Fit the exact historical recurrent checkpoint on the new train role."""

    if getattr(train_role, "role", None) != "train":
        raise SceneDiversityRecurrentReplicationError("training role changed")
    validate_role_scene_geometry_v1(train_role)
    checkpoint = frozen.fit_checkpoint_v1(
        train_role, train_context_tokens, device=device
    )
    if checkpoint.get("config") != frozen.config_v1():
        raise SceneDiversityRecurrentReplicationError(
            "frozen recurrent checkpoint config changed"
        )
    return checkpoint


def checkpoint_identity_v1(checkpoint: Mapping[str, object]) -> str:
    return frozen.checkpoint_identity_v1(checkpoint)


def validate_context_tokens_v1(tokens: torch.Tensor, *, role: str) -> torch.Tensor:
    return frozen.validate_context_tokens_v1(tokens, role=role)


def _prediction_diagnostics_v1(
    predictions: torch.Tensor, targets: torch.Tensor, scales: torch.Tensor
) -> dict[str, object]:
    errors = predictions.to(torch.float64) - targets.to(torch.float64)
    return {
        "per_output_rmse": torch.sqrt(torch.mean(errors.square(), dim=(0, 1))).tolist(),
        "joint_standardized_mse": float(
            torch.mean((errors / scales.to(torch.float64)).square())
        ),
    }


def evaluate_checkpoint_v1(
    checkpoint: Mapping[str, object],
    train_role: Any,
    eval_role: Any,
    eval_context_tokens: torch.Tensor,
    *,
    device: torch.device,
    integrity_passed: bool,
) -> dict[str, object]:
    """Evaluate the frozen three-arm protocol on fresh scene-disjoint roles."""

    if checkpoint.get("identity_sha256") != frozen.checkpoint_identity_v1(checkpoint):
        raise SceneDiversityRecurrentReplicationError("checkpoint identity changed")
    if checkpoint.get("config") != frozen.config_v1():
        raise SceneDiversityRecurrentReplicationError("checkpoint protocol changed")
    train_geometry = validate_role_scene_geometry_v1(train_role)
    eval_geometry = validate_role_scene_geometry_v1(eval_role)
    if train_role.role != "train" or eval_role.role != "eval":
        raise SceneDiversityRecurrentReplicationError("role labels changed")
    frozen.validate_context_tokens_v1(eval_context_tokens, role="eval")
    projected_eval_context = frozen.project_context_tokens_v1(
        eval_context_tokens, checkpoint["visual_projection"], role="eval"
    )

    reports: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for arm in ARM_ORDER:
        predictions: list[torch.Tensor] = []
        members: list[dict[str, object]] = []
        for member in checkpoint["arms"][arm]:
            outcomes = frozen.predict_member_v1(
                member["state_dict"],
                eval_role,
                projected_eval_context,
                checkpoint["input_statistics"],
                checkpoint["outcome_statistics"],
                arm=arm,
                device=device,
            )
            scores = grounded.physical_score_matrix_v1(eval_role.plan, outcomes)
            report = grounded.report_physical_scores_v1(eval_role.plan, scores)
            predictions.append(outcomes)
            members.append(
                {
                    "seed": member["seed"],
                    "outcome_identity_sha256": frozen._tensor_sha256(outcomes),  # noqa: SLF001
                    "score_identity_sha256": physical._array_identity_v1(scores),  # noqa: SLF001
                    "report": report,
                }
            )
        ensemble = torch.stack(predictions).mean(dim=0)
        scores = grounded.physical_score_matrix_v1(eval_role.plan, ensemble)
        reports[arm] = grounded.report_physical_scores_v1(eval_role.plan, scores)
        artifacts[arm] = {
            "ensemble_outcomes": ensemble.tolist(),
            "ensemble_outcome_identity_sha256": frozen._tensor_sha256(ensemble),  # noqa: SLF001
            "ensemble_scores": scores.tolist(),
            "ensemble_score_identity_sha256": physical._array_identity_v1(scores),  # noqa: SLF001
            "members": members,
        }
        diagnostics[arm] = _prediction_diagnostics_v1(
            ensemble,
            eval_role.targets,
            checkpoint["outcome_statistics"]["residual_scales"],
        )

    # This is the same live ridge control and regularization as V1.  Its old
    # exact-regret assertion referred to one historical role and is therefore
    # explicitly inapplicable on these new scenes.
    task = frozen._fit_task_control_v1(train_role.plan)  # noqa: SLF001
    task_scores = frozen._score_task_control_v1(eval_role.plan, task)  # noqa: SLF001
    reports["task_action_only"] = grounded.report_physical_scores_v1(
        eval_role.plan, task_scores
    )
    task_regret = float(
        reports["task_action_only"]["summary"]["normalized_rank_regret"]
    )
    oracle_scores = np.asarray(
        [state.dense_ranks for state in eval_role.plan.states], dtype=np.float64
    )
    reports["privileged_physical_oracle"] = grounded.report_physical_scores_v1(
        eval_role.plan, oracle_scores
    )
    reports["random_expected"] = physical.prior._random_expected_report(  # noqa: SLF001
        eval_role.plan
    )
    comparisons = {
        "visual_vs_task_action_only": paired_family_scene_bootstrap_v1(
            reports[VISUAL_ARM]["group_results"],
            reports["task_action_only"]["group_results"],
        ),
        "visual_vs_no_vision": paired_family_scene_bootstrap_v1(
            reports[VISUAL_ARM]["group_results"],
            reports[NO_VISION_ARM]["group_results"],
        ),
        "no_vision_vs_task_action_only": paired_family_scene_bootstrap_v1(
            reports[NO_VISION_ARM]["group_results"],
            reports["task_action_only"]["group_results"],
        ),
    }
    frozen_gate = grounded.fixed_gate_v1(
        joint_report=reports[VISUAL_ARM],
        task_report=reports["task_action_only"],
        matched_report=reports[NO_VISION_ARM],
        random_report=reports["random_expected"],
        oracle_report=reports["privileged_physical_oracle"],
        joint_vs_task=comparisons["visual_vs_task_action_only"],
        joint_vs_matched=comparisons["visual_vs_no_vision"],
        integrity_passed=integrity_passed,
    )
    passed = bool(frozen_gate["passed"])
    gate = {
        "schema": "lewm_go2_scene_diversity_recurrent_replication_fixed_gate_v1",
        "passed": passed,
        "status": PASS_STATUS if passed else STOP_STATUS,
        "gates": {
            "1_integrity_and_oracle": frozen_gate["gates"]["1_integrity_and_oracle"],
            "2_absolute_regret": frozen_gate["gates"]["2_absolute_regret"],
            "3_visual_beats_task_action_only": frozen_gate["gates"][
                "3_joint_beats_task_action_only"
            ],
            "4_visual_beats_no_vision": frozen_gate["gates"][
                "4_joint_beats_matched_physical_only"
            ],
            "5_visual_beats_random": frozen_gate["gates"]["5_joint_beats_random"],
        },
        "threshold_source": frozen_gate["schema"],
        "thresholds": frozen.config_v1()["frozen_h1_thresholds"],
    }
    return {
        "schema": SCHEMA,
        "status": gate["status"],
        "claim_scope": "DEVELOPMENT_ONLY_H1_SCENE_DIVERSITY_REPLICATION",
        "checkpoint_identity_sha256": checkpoint["identity_sha256"],
        "train_role_identity_sha256": str(train_role.identity_sha256),
        "eval_role_identity_sha256": str(eval_role.identity_sha256),
        "role_geometry": {"train": train_geometry, "eval": eval_geometry},
        "reports": reports,
        "comparisons": comparisons,
        "gate": gate,
        "prediction_artifacts": artifacts,
        "prediction_diagnostics": diagnostics,
        "task_control": {
            "live_identity_sha256": task.identity_sha256,
            "behavioral_eval_regret": task_regret,
            "historical_exact_regret_assertion": {
                "status": TASK_CONTROL_ASSERTION_STATUS,
                "historical_value": physical.EXPECTED_TASK_EVAL_REGRET,
                "reason": "the assertion binds the historical V1 role, not fresh scenes",
            },
            "same_task_relative_gate_thresholds_applied": True,
        },
        "successor_observations_opened": 0,
        "authorizes_blind_rollout_preregistration": passed,
        "authorizes_navigation_claim": False,
    }


__all__ = [
    "ACTION_COUNT",
    "ARM_ORDER",
    "BATCH_STATES",
    "MODEL_SEEDS",
    "NO_VISION_ARM",
    "PASS_STATUS",
    "SAMPLER_SEED",
    "SCHEMA",
    "STATE_COUNT",
    "STOP_STATUS",
    "SceneDiversityRecurrentReplicationError",
    "UPDATES",
    "VISUAL_ARM",
    "checkpoint_identity_v1",
    "build_role_feature_plan_v1",
    "config_v1",
    "evaluate_checkpoint_v1",
    "fit_checkpoint_v1",
    "paired_family_scene_bootstrap_v1",
    "validate_context_tokens_v1",
    "validate_role_scene_geometry_v1",
]
