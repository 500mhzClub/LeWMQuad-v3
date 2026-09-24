"""Pure losses and evaluation mechanics for grounded dense-DINO JEPA V1.

This module has no filesystem, RGB, encoder, checkpoint, or runtime path.  It
accepts already validated tensors and the existing matched-branch feature-plan
objects.  Dataset custody and model execution remain runner responsibilities.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from lewm.benchmarks import go2_dinov2_physical_readout_calibration_v1 as calibration
from lewm.benchmarks import go2_matched_branch_physical_outcome_screen_v1 as physical


ACTION_COUNT = 9
CONTEXT_FRAME_COUNT = 3
PHYSICAL_INPUT_WIDTH = 12
TARGET_WIDTH = 4
TRUNK_TOKEN_COUNT = 257
PATCH_TOKEN_COUNT = 256
TOKEN_DIMENSION = 384
MIN_SCALE = 1.0e-8
PATH_COST_WEIGHT = 0.01
PAIRWISE_RANK_SCALE = 0.05
INFONCE_TEMPERATURE = 0.10
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2_026_080_407

INPUT_STATS_SCHEMA = "lewm_go2_grounded_dense_dino_input_statistics_v1"
OUTCOME_STATS_SCHEMA = "lewm_go2_grounded_dense_dino_outcome_statistics_v1"
PASS_STATUS = "PASS_DEVELOPMENT_CLOSED_LOOP_EXPERIMENT_ELIGIBLE"
STOP_STATUS = "STOP_GROUNDED_DENSE_DINO_JOINT_JEPA_NOT_ELIGIBLE"


class GroundedDenseDINOJointJEPAError(ValueError):
    """Raised when a pure tensor or evaluation contract changes."""


@dataclass(frozen=True)
class DenseTrunkLayoutV1:
    """State-major trunk tensors selected solely by bound artifact identity."""

    context_trunk_tokens: torch.Tensor
    successor_trunk_tokens: torch.Tensor | None
    context_artifact_ids: tuple[tuple[str, str, str], ...]
    successor_artifact_ids: tuple[tuple[str, ...], ...] | None


def _finite_tensor(
    value: object,
    *,
    name: str,
    ndim: int | None = None,
    trailing_shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise GroundedDenseDINOJointJEPAError(f"{name} must be a floating tensor")
    if ndim is not None and value.ndim != ndim:
        raise GroundedDenseDINOJointJEPAError(f"{name} rank changed")
    if trailing_shape is not None and tuple(value.shape[-len(trailing_shape) :]) != trailing_shape:
        raise GroundedDenseDINOJointJEPAError(f"{name} trailing shape changed")
    if not bool(torch.isfinite(value).all()):
        raise GroundedDenseDINOJointJEPAError(f"{name} must be finite")
    return value


def _tensor_digest(value: torch.Tensor) -> str:
    array = value.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    canonical = np.ascontiguousarray(array.astype("<f4", copy=False))
    digest = hashlib.sha256()
    digest.update(str(canonical.shape).encode("ascii") + b"\0")
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _stats_identity(schema: str, tensors: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256(schema.encode("ascii") + b"\0")
    for tensor in tensors:
        digest.update(bytes.fromhex(_tensor_digest(tensor)))
    return digest.hexdigest()


def fit_input_statistics_v1(train_inputs: torch.Tensor) -> dict[str, object]:
    """Fit population mean/scale over state and candidate axes for 12D inputs."""

    values = _finite_tensor(
        train_inputs,
        name="training physical inputs",
        ndim=3,
        trailing_shape=(ACTION_COUNT, PHYSICAL_INPUT_WIDTH),
    ).detach().to(device="cpu", dtype=torch.float64)
    if values.shape[0] == 0:
        raise GroundedDenseDINOJointJEPAError("training physical inputs are empty")
    flat = values.reshape(-1, PHYSICAL_INPUT_WIDTH)
    mean = flat.mean(dim=0)
    scale = torch.sqrt(torch.mean((flat - mean).square(), dim=0))
    scale = torch.where(scale < MIN_SCALE, torch.ones_like(scale), scale)
    mean32, scale32 = mean.float(), scale.float()
    return {
        "schema": INPUT_STATS_SCHEMA,
        "mean": mean32,
        "scale": scale32,
        "identity_sha256": _stats_identity(INPUT_STATS_SCHEMA, (mean32, scale32)),
    }


def validate_input_statistics_v1(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "mean",
        "scale",
        "identity_sha256",
    }:
        raise GroundedDenseDINOJointJEPAError("input-statistics fields changed")
    mean = _finite_tensor(value["mean"], name="input mean", ndim=1)
    scale = _finite_tensor(value["scale"], name="input scale", ndim=1)
    if tuple(mean.shape) != (PHYSICAL_INPUT_WIDTH,) or tuple(scale.shape) != (
        PHYSICAL_INPUT_WIDTH,
    ):
        raise GroundedDenseDINOJointJEPAError("input-statistics shape changed")
    if mean.dtype != torch.float32 or scale.dtype != torch.float32:
        raise GroundedDenseDINOJointJEPAError("input statistics must be float32")
    if not bool((scale > 0.0).all()):
        raise GroundedDenseDINOJointJEPAError("input scales must be positive")
    expected = _stats_identity(INPUT_STATS_SCHEMA, (mean, scale))
    if value["schema"] != INPUT_STATS_SCHEMA or value["identity_sha256"] != expected:
        raise GroundedDenseDINOJointJEPAError("input-statistics identity changed")
    return value


def fit_outcome_statistics_v1(train_targets: torch.Tensor) -> dict[str, object]:
    """Fit nine action means and one shared four-output residual scale."""

    targets = _finite_tensor(
        train_targets,
        name="training physical targets",
        ndim=3,
        trailing_shape=(ACTION_COUNT, TARGET_WIDTH),
    ).detach().to(device="cpu", dtype=torch.float64)
    if targets.shape[0] == 0:
        raise GroundedDenseDINOJointJEPAError("training physical targets are empty")
    action_means = targets.mean(dim=0)
    residuals = targets - action_means.unsqueeze(0)
    residual_scales = torch.sqrt(torch.mean(residuals.square(), dim=(0, 1)))
    residual_scales = torch.where(
        residual_scales < MIN_SCALE,
        torch.ones_like(residual_scales),
        residual_scales,
    )
    means32, scales32 = action_means.float(), residual_scales.float()
    return {
        "schema": OUTCOME_STATS_SCHEMA,
        "action_means": means32,
        "residual_scales": scales32,
        "identity_sha256": _stats_identity(OUTCOME_STATS_SCHEMA, (means32, scales32)),
    }


def validate_outcome_statistics_v1(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "action_means",
        "residual_scales",
        "identity_sha256",
    }:
        raise GroundedDenseDINOJointJEPAError("outcome-statistics fields changed")
    means = _finite_tensor(value["action_means"], name="action means", ndim=2)
    scales = _finite_tensor(value["residual_scales"], name="residual scales", ndim=1)
    if tuple(means.shape) != (ACTION_COUNT, TARGET_WIDTH) or tuple(scales.shape) != (
        TARGET_WIDTH,
    ):
        raise GroundedDenseDINOJointJEPAError("outcome-statistics shape changed")
    if means.dtype != torch.float32 or scales.dtype != torch.float32:
        raise GroundedDenseDINOJointJEPAError("outcome statistics must be float32")
    if not bool((scales > 0.0).all()):
        raise GroundedDenseDINOJointJEPAError("residual scales must be positive")
    expected = _stats_identity(OUTCOME_STATS_SCHEMA, (means, scales))
    if value["schema"] != OUTCOME_STATS_SCHEMA or value["identity_sha256"] != expected:
        raise GroundedDenseDINOJointJEPAError("outcome-statistics identity changed")
    return value


def normalize_physical_inputs_v1(
    physical_inputs: torch.Tensor, input_stats: Mapping[str, object]
) -> torch.Tensor:
    values = _finite_tensor(
        physical_inputs,
        name="physical inputs",
        trailing_shape=(ACTION_COUNT, PHYSICAL_INPUT_WIDTH),
    )
    validated = validate_input_statistics_v1(input_stats)
    mean = validated["mean"].to(device=values.device, dtype=values.dtype)
    scale = validated["scale"].to(device=values.device, dtype=values.dtype)
    return (values - mean) / scale


def _broadcast_action_means(
    values: torch.Tensor,
    outcome_stats: Mapping[str, object],
    action_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    validated = validate_outcome_statistics_v1(outcome_stats)
    means = validated["action_means"].to(device=values.device, dtype=values.dtype)
    scales = validated["residual_scales"].to(device=values.device, dtype=values.dtype)
    if action_ids is None:
        if values.shape[-2] != ACTION_COUNT:
            raise GroundedDenseDINOJointJEPAError("canonical action axis changed")
        selected_means = means
    else:
        ids = torch.as_tensor(action_ids, device=values.device)
        if ids.is_floating_point() or tuple(ids.shape) != tuple(values.shape[:-1]):
            raise GroundedDenseDINOJointJEPAError("action IDs must align with outcomes")
        ids = ids.to(torch.long)
        if not bool(((ids >= 0) & (ids < ACTION_COUNT)).all()):
            raise GroundedDenseDINOJointJEPAError("action IDs are out of range")
        sorted_ids = torch.sort(ids, dim=-1).values
        canonical = torch.arange(ACTION_COUNT, device=ids.device).expand_as(sorted_ids)
        if not torch.equal(sorted_ids, canonical):
            raise GroundedDenseDINOJointJEPAError(
                "each candidate row must be an exact action permutation"
            )
        selected_means = means[ids]
    return selected_means, scales


def standardize_outcome_residuals_v1(
    targets: torch.Tensor,
    outcome_stats: Mapping[str, object],
    *,
    action_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    values = _finite_tensor(
        targets,
        name="physical targets",
        trailing_shape=(ACTION_COUNT, TARGET_WIDTH),
    )
    means, scales = _broadcast_action_means(values, outcome_stats, action_ids)
    return (values - means) / scales


def decode_standardized_outcomes_v1(
    standardized_residuals: torch.Tensor,
    outcome_stats: Mapping[str, object],
    *,
    action_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    residuals = _finite_tensor(
        standardized_residuals,
        name="standardized physical residuals",
        trailing_shape=(ACTION_COUNT, TARGET_WIDTH),
    )
    means, scales = _broadcast_action_means(residuals, outcome_stats, action_ids)
    return means + residuals * scales


def predicted_physical_cost_v1(
    decoded_outcomes: torch.Tensor, relative_goal_xy_body_m: torch.Tensor
) -> torch.Tensor:
    """Differentiable lower-is-better H1 cost used by the ranking loss."""

    outcomes = _finite_tensor(
        decoded_outcomes,
        name="decoded physical outcomes",
        trailing_shape=(ACTION_COUNT, TARGET_WIDTH),
    )
    goals = _finite_tensor(
        relative_goal_xy_body_m,
        name="relative body-frame goals",
        trailing_shape=(2,),
    )
    if tuple(goals.shape[:-1]) != tuple(outcomes.shape[:-2]):
        raise GroundedDenseDINOJointJEPAError("goal batch shape does not match outcomes")
    remaining = torch.linalg.vector_norm(
        goals.unsqueeze(-2) - outcomes[..., :2], dim=-1
    )
    return remaining + PATH_COST_WEIGHT * F.relu(outcomes[..., 3])


def strict_rank_pairwise_softplus_loss_v1(
    predicted_costs: torch.Tensor,
    true_dense_ranks: torch.Tensor,
    *,
    scale: float = PAIRWISE_RANK_SCALE,
) -> torch.Tensor:
    """Mean softplus over every strict true-rank pair; true ties are omitted."""

    costs = _finite_tensor(
        predicted_costs,
        name="predicted physical costs",
        trailing_shape=(ACTION_COUNT,),
    )
    ranks = torch.as_tensor(true_dense_ranks, device=costs.device)
    if tuple(ranks.shape) != tuple(costs.shape) or ranks.is_floating_point():
        raise GroundedDenseDINOJointJEPAError("dense ranks must be aligned integers")
    if not bool(((ranks >= 0) & torch.isfinite(ranks)).all()):
        raise GroundedDenseDINOJointJEPAError("dense ranks are invalid")
    if not math.isfinite(scale) or scale <= 0.0:
        raise GroundedDenseDINOJointJEPAError("pairwise scale must be positive")
    strict = ranks.unsqueeze(-1) < ranks.unsqueeze(-2)
    if not bool(strict.any()):
        raise GroundedDenseDINOJointJEPAError("dense ranks contain no strict pair")
    differences = costs.unsqueeze(-1) - costs.unsqueeze(-2)
    return F.softplus(differences[strict] / scale).mean()


def dense_patch_cosine_loss_v1(
    predicted_successors: torch.Tensor, target_successors: torch.Tensor
) -> torch.Tensor:
    predicted = _finite_tensor(
        predicted_successors,
        name="predicted successor patches",
        trailing_shape=(PATCH_TOKEN_COUNT, TOKEN_DIMENSION),
    )
    target = _finite_tensor(
        target_successors,
        name="EMA target successor patches",
        trailing_shape=(PATCH_TOKEN_COUNT, TOKEN_DIMENSION),
    ).detach()
    if tuple(predicted.shape) != tuple(target.shape):
        raise GroundedDenseDINOJointJEPAError("successor patch layouts differ")
    return (1.0 - F.cosine_similarity(predicted, target, dim=-1)).mean()


def within_state_action_logits_v1(
    predicted_successors: torch.Tensor,
    target_successors: torch.Tensor,
    *,
    temperature: float = INFONCE_TEMPERATURE,
) -> torch.Tensor:
    predicted = _finite_tensor(
        predicted_successors,
        name="predicted action successors",
        ndim=4,
        trailing_shape=(ACTION_COUNT, PATCH_TOKEN_COUNT, TOKEN_DIMENSION),
    )
    target = _finite_tensor(
        target_successors,
        name="target action successors",
        ndim=4,
        trailing_shape=(ACTION_COUNT, PATCH_TOKEN_COUNT, TOKEN_DIMENSION),
    ).detach()
    if tuple(predicted.shape) != tuple(target.shape):
        raise GroundedDenseDINOJointJEPAError("InfoNCE successor layouts differ")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise GroundedDenseDINOJointJEPAError("InfoNCE temperature must be positive")
    predicted = F.normalize(predicted, dim=-1)
    target = F.normalize(target, dim=-1)
    # Each predicted action is compared with all nine true successors using
    # same-position patch cosine averaged over the dense grid.
    similarities = torch.einsum("bapd,btpd->batp", predicted, target).mean(dim=-1)
    return similarities / temperature


def within_state_action_infonce_loss_v1(
    predicted_successors: torch.Tensor,
    target_successors: torch.Tensor,
    *,
    temperature: float = INFONCE_TEMPERATURE,
) -> torch.Tensor:
    logits = within_state_action_logits_v1(
        predicted_successors, target_successors, temperature=temperature
    )
    labels = torch.arange(ACTION_COUNT, device=logits.device).repeat(logits.shape[0])
    return F.cross_entropy(logits.reshape(-1, ACTION_COUNT), labels)


def true_successor_branch_retrieval_v1(
    predicted_successors: torch.Tensor, target_successors: torch.Tensor
) -> torch.Tensor:
    logits = within_state_action_logits_v1(predicted_successors, target_successors)
    labels = torch.arange(ACTION_COUNT, device=logits.device).view(1, -1)
    return (logits.argmax(dim=-1) == labels).to(torch.float32).mean()


def extract_dense_trunk_layout_v1(
    plan: Any,
    artifact_ids: Sequence[str],
    trunk_tokens: torch.Tensor,
    *,
    include_successors: bool,
) -> DenseTrunkLayoutV1:
    """Select state/action layouts by artifact IDs, never by cache row alignment.

    Context-only calls require a context-only artifact set.  This makes the
    physical-only and evaluation boundaries fail closed if successor tensors
    were accidentally included.
    """

    if type(include_successors) is not bool:
        raise GroundedDenseDINOJointJEPAError("include_successors must be boolean")
    tokens = _finite_tensor(
        trunk_tokens,
        name="DINO trunk tokens",
        ndim=3,
        trailing_shape=(TRUNK_TOKEN_COUNT, TOKEN_DIMENSION),
    )
    ids = tuple(artifact_ids)
    if len(ids) != tokens.shape[0] or any(not isinstance(item, str) or not item for item in ids):
        raise GroundedDenseDINOJointJEPAError("trunk artifact IDs are malformed")
    if len(set(ids)) != len(ids):
        raise GroundedDenseDINOJointJEPAError("trunk artifact IDs repeat")
    plan_ids = tuple(getattr(plan, "artifact_ids", ()))
    states = tuple(getattr(plan, "states", ()))
    if not plan_ids or not states or len(set(plan_ids)) != len(plan_ids):
        raise GroundedDenseDINOJointJEPAError("feature plan is malformed")

    context_rows: list[tuple[str, str, str]] = []
    successor_rows: list[tuple[str, ...]] = []
    for state in states:
        context_indices = tuple(getattr(state, "context_artifact_indices", ()))
        target_indices = tuple(getattr(state, "target_artifact_indices", ()))
        if len(context_indices) != CONTEXT_FRAME_COUNT or len(target_indices) != ACTION_COUNT:
            raise GroundedDenseDINOJointJEPAError("feature-plan state geometry changed")
        if any(type(index) is not int or not 0 <= index < len(plan_ids) for index in (*context_indices, *target_indices)):
            raise GroundedDenseDINOJointJEPAError("feature-plan artifact index changed")
        context_rows.append(tuple(plan_ids[index] for index in context_indices))  # type: ignore[arg-type]
        successor_rows.append(tuple(plan_ids[index] for index in target_indices))

    expected_contexts = tuple(item for row in context_rows for item in row)
    expected_successors = tuple(item for row in successor_rows for item in row)
    expected = expected_contexts + expected_successors if include_successors else expected_contexts
    if len(set(expected)) != len(expected) or set(ids) != set(expected) or len(ids) != len(expected):
        scope = "context-plus-successor" if include_successors else "context-only"
        raise GroundedDenseDINOJointJEPAError(f"trunk artifacts violate {scope} scope")
    by_id = {artifact_id: index for index, artifact_id in enumerate(ids)}
    contexts = torch.stack(
        [torch.stack([tokens[by_id[item]] for item in row]) for row in context_rows]
    )
    successors = None
    successor_ids = None
    if include_successors:
        successors = torch.stack(
            [torch.stack([tokens[by_id[item]] for item in row]) for row in successor_rows]
        )
        successor_ids = tuple(successor_rows)
    return DenseTrunkLayoutV1(
        context_trunk_tokens=contexts,
        successor_trunk_tokens=successors,
        context_artifact_ids=tuple(context_rows),
        successor_artifact_ids=successor_ids,
    )


def physical_score_matrix_v1(plan: Any, predicted_outcomes: torch.Tensor) -> np.ndarray:
    """Apply the frozen one-centimetre progress/path/action-ID ordering."""

    return physical.physical_score_matrix_v1(plan, predicted_outcomes)


def report_physical_scores_v1(plan: Any, scores: np.ndarray) -> dict[str, object]:
    """Report selected actions against the existing physical dense ranks."""

    return physical.report_arm_v1(plan, scores)


def paired_family_scene_bootstrap_v1(
    candidate_results: Sequence[Mapping[str, object]],
    baseline_results: Sequence[Mapping[str, object]],
    *,
    field: str = "normalized_rank_regret",
) -> dict[str, object]:
    """Frozen family-equal, paired whole-scene bootstrap."""

    return calibration.paired_family_scene_cluster_comparison_v1(
        candidate_results,
        baseline_results,
        field=field,
        resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
    )


def _summary_regret(report: Mapping[str, object], *, name: str) -> float:
    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        raise GroundedDenseDINOJointJEPAError(f"{name} summary is absent")
    value = float(summary.get("normalized_rank_regret", math.nan))
    if not math.isfinite(value):
        raise GroundedDenseDINOJointJEPAError(f"{name} regret is invalid")
    return value


def fixed_gate_v1(
    *,
    joint_report: Mapping[str, object],
    task_report: Mapping[str, object],
    matched_report: Mapping[str, object],
    random_report: Mapping[str, object],
    oracle_report: Mapping[str, object],
    joint_vs_task: Mapping[str, object],
    joint_vs_matched: Mapping[str, object],
    integrity_passed: bool,
) -> dict[str, object]:
    """Return the immutable five-part development-eligibility gate."""

    if type(integrity_passed) is not bool:
        raise GroundedDenseDINOJointJEPAError("integrity result must be boolean")
    joint = _summary_regret(joint_report, name="joint")
    task = _summary_regret(task_report, name="task/action")
    matched = _summary_regret(matched_report, name="matched physical-only")
    random = _summary_regret(random_report, name="random")
    oracle = _summary_regret(oracle_report, name="oracle")
    oracle_summary = oracle_report["summary"]
    oracle_rate = float(
        oracle_summary.get(
            "oracle_equivalent_selection_rate",
            oracle_summary.get("oracle_match_rate", math.nan),
        )
    )
    task_delta = joint - task
    matched_delta = joint - matched
    task_upper = float(joint_vs_task.get("upper_95", math.nan))
    matched_upper = float(joint_vs_matched.get("upper_95", math.nan))
    task_bootstrap_delta = float(joint_vs_task.get("mean_delta", math.nan))
    matched_bootstrap_delta = float(joint_vs_matched.get("mean_delta", math.nan))
    if not all(
        math.isfinite(value)
        for value in (
            oracle_rate,
            task_upper,
            matched_upper,
            task_bootstrap_delta,
            matched_bootstrap_delta,
        )
    ):
        raise GroundedDenseDINOJointJEPAError("gate measurements are nonfinite")
    if not math.isclose(task_delta, task_bootstrap_delta, rel_tol=0.0, abs_tol=1.0e-12):
        raise GroundedDenseDINOJointJEPAError("joint-task point effects disagree")
    if not math.isclose(
        matched_delta, matched_bootstrap_delta, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise GroundedDenseDINOJointJEPAError("joint-matched point effects disagree")

    gates: dict[str, dict[str, object]] = {
        "1_integrity_and_oracle": {
            "passed": integrity_passed and oracle == 0.0 and oracle_rate == 1.0,
            "integrity_passed": integrity_passed,
            "oracle_regret": oracle,
            "oracle_equivalent_selection_rate": oracle_rate,
        },
        "2_absolute_regret": {
            "passed": joint <= 0.13,
            "joint_regret": joint,
            "maximum": 0.13,
        },
        "3_joint_beats_task_action_only": {
            "passed": task_delta <= -0.02 and task_upper < 0.0,
            "joint_minus_task": task_delta,
            "required_maximum": -0.02,
            "upper_95": task_upper,
        },
        "4_joint_beats_matched_physical_only": {
            "passed": matched_delta <= -0.01 and matched_upper < 0.0,
            "joint_minus_matched": matched_delta,
            "required_maximum": -0.01,
            "upper_95": matched_upper,
        },
        "5_joint_beats_random": {
            "passed": joint < random,
            "joint_regret": joint,
            "random_expected_regret": random,
        },
    }
    passed = all(bool(item["passed"]) for item in gates.values())
    return {
        "schema": "lewm_go2_grounded_dense_dino_joint_jepa_v1_fixed_gate_v1",
        "passed": passed,
        "status": PASS_STATUS if passed else STOP_STATUS,
        "gates": gates,
    }


__all__ = [
    "ACTION_COUNT",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "DenseTrunkLayoutV1",
    "GroundedDenseDINOJointJEPAError",
    "decode_standardized_outcomes_v1",
    "dense_patch_cosine_loss_v1",
    "extract_dense_trunk_layout_v1",
    "fit_input_statistics_v1",
    "fit_outcome_statistics_v1",
    "fixed_gate_v1",
    "normalize_physical_inputs_v1",
    "paired_family_scene_bootstrap_v1",
    "physical_score_matrix_v1",
    "predicted_physical_cost_v1",
    "report_physical_scores_v1",
    "standardize_outcome_residuals_v1",
    "strict_rank_pairwise_softplus_loss_v1",
    "true_successor_branch_retrieval_v1",
    "validate_input_statistics_v1",
    "validate_outcome_statistics_v1",
    "within_state_action_infonce_loss_v1",
    "within_state_action_logits_v1",
]
