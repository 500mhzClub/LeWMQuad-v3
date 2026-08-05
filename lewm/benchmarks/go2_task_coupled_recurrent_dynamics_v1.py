"""Mechanics for the task-coupled recurrent physical-dynamics successor V1.

This module is deliberately filesystem-free.  It consumes role objects and
frozen full-DINO context tokens supplied by the runner, trains two exactly
matched recurrent arms, and applies the already frozen H1 scorer, bootstrap,
and progression thresholds.  Successor images or successor feature grids are
not accepted by any API in this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import time
from typing import Any

import numpy as np
import torch

from lewm.benchmarks import go2_grounded_dense_dino_joint_jepa_v1 as grounded
from lewm.benchmarks import go2_matched_branch_physical_outcome_screen_v1 as physical
from scripts.evaluate_go2_world_model_counterfactual_action_regret_v1 import (
    ActionSpecificRidgeReadoutsV1,
    fit_action_specific_ridge_readouts_v1,
    predict_action_specific_scores_v1,
    task_conditioned_feature_v1,
)
from lewm.models.go2_task_coupled_recurrent_dynamics_v1 import (
    CANDIDATE_WIDTH,
    CONTEXT_STEPS,
    MOTION_WIDTH,
    OUTPUT_WIDTH,
    PARAMETER_COUNT,
    TOKEN_COUNT,
    VISUAL_WIDTH,
    TaskCoupledRecurrentDynamicsV1,
    initialize_task_coupled_recurrent_dynamics_v1,
    recurrent_dynamics_state_identity_v1,
)


SCHEMA = "lewm_go2_task_coupled_recurrent_dynamics_v1"
CHECKPOINT_SCHEMA = "lewm_go2_task_coupled_recurrent_dynamics_checkpoint_v1"
VISUAL_PROJECTION_SCHEMA = "lewm_go2_task_coupled_recurrent_visual_projection_v1"
PASS_STATUS = "PASS_TASK_COUPLED_RECURRENT_DYNAMICS_H1"
STOP_STATUS = "STOP_TASK_COUPLED_RECURRENT_DYNAMICS_H1"

STATE_COUNT = 128
ACTION_COUNT = 9
VISUAL_ARM = "visual_recurrent_direct"
NO_VISION_ARM = "no_vision_recurrent_direct"
ARM_ORDER = (NO_VISION_ARM, VISUAL_ARM)
MODEL_SEEDS = (2_026_080_411, 2_026_080_412, 2_026_080_413)
SAMPLER_SEED = 2_026_080_414
UPDATES = 800
BATCH_STATES = 8
TRACE_UPDATES = (0, 400, 800)
LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 1.0e-4
RANK_WEIGHT = 0.25
GRADIENT_CLIP_NORM = 1.0
TASK_RIDGE_LAMBDA = 1.0e-3
RAW_TOKEN_COUNT = 256
RAW_VISUAL_WIDTH = 384


class TaskCoupledRecurrentDynamicsError(RuntimeError):
    """Raised when the frozen scientific geometry changes."""


def canonical_bytes_v1(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii") + b"\0")
    digest.update(canonical_bytes_v1(list(tensor.shape)) + b"\0")
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def config_v1() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "states_per_role": STATE_COUNT,
        "actions": ACTION_COUNT,
        "context_steps": CONTEXT_STEPS,
        "frozen_dino_context_grid": [RAW_TOKEN_COUNT, RAW_VISUAL_WIDTH],
        "model_visual_grid": [TOKEN_COUNT, VISUAL_WIDTH],
        "visual_projection": {
            "pool": "nonoverlapping_4x4_mean_16x16_to_4x4",
            "channel_pca": [RAW_VISUAL_WIDTH, VISUAL_WIDTH],
            "fit_role": "train_context_only",
            "standardize_projected_channels": True,
        },
        "motion_step_width": MOTION_WIDTH,
        "candidate_command_width": CANDIDATE_WIDTH,
        "output_width": OUTPUT_WIDTH,
        "trainable_parameters_per_member": PARAMETER_COUNT,
        "arms": list(ARM_ORDER),
        "model_seeds": list(MODEL_SEEDS),
        "shared_sampler_seed": SAMPLER_SEED,
        "updates": UPDATES,
        "batch_states": BATCH_STATES,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "betas": [0.9, 0.999],
            "epsilon": 1.0e-8,
        },
        "loss": {
            "standardized_physical_residual_mse_weight": 1.0,
            "all_strict_pair_rank_softplus_weight": RANK_WEIGHT,
            "rank_scale_m": 0.05,
        },
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "trace_updates": list(TRACE_UPDATES),
        "goal_available_to_model": False,
        "successor_observation_access": False,
        "frozen_h1_thresholds": {
            "maximum_regret": 0.13,
            "visual_minus_task_maximum": -0.02,
            "visual_minus_no_vision_maximum": -0.01,
            "paired_upper_95_must_be_below_zero": True,
            "must_beat_random": True,
        },
    }


def _validate_role_v1(role: Any) -> None:
    if (
        getattr(role, "role", None) not in {"train", "eval"}
        or tuple(getattr(role, "physical_inputs", torch.empty(0)).shape)
        != (STATE_COUNT, ACTION_COUNT, 12)
        or tuple(getattr(role, "targets", torch.empty(0)).shape)
        != (STATE_COUNT, ACTION_COUNT, OUTPUT_WIDTH)
        or tuple(getattr(role, "history_commands", torch.empty(0)).shape)
        != (STATE_COUNT, 2, CANDIDATE_WIDTH)
        or tuple(getattr(role, "candidate_commands", torch.empty(0)).shape)
        != (STATE_COUNT, ACTION_COUNT, CANDIDATE_WIDTH)
        or tuple(getattr(role, "relative_goals", torch.empty(0)).shape)
        != (STATE_COUNT, 2)
        or tuple(getattr(role, "dense_ranks", torch.empty(0)).shape)
        != (STATE_COUNT, ACTION_COUNT)
    ):
        raise TaskCoupledRecurrentDynamicsError("role tensor geometry changed")
    tensors = (
        role.physical_inputs,
        role.targets,
        role.history_commands,
        role.candidate_commands,
        role.relative_goals,
    )
    if any(value.dtype != torch.float32 for value in tensors) or not all(
        bool(torch.isfinite(value).all()) for value in tensors
    ):
        raise TaskCoupledRecurrentDynamicsError("role tensors must be finite float32")
    if role.dense_ranks.dtype != torch.long:
        raise TaskCoupledRecurrentDynamicsError("role ranks must use torch.long")


def validate_context_tokens_v1(tokens: torch.Tensor, *, role: str) -> torch.Tensor:
    if (
        role not in {"train", "eval"}
        or not isinstance(tokens, torch.Tensor)
        or tuple(tokens.shape)
        != (STATE_COUNT, CONTEXT_STEPS, RAW_TOKEN_COUNT, RAW_VISUAL_WIDTH)
        or tokens.device.type != "cpu"
        or tokens.dtype != torch.float32
        or not bool(torch.isfinite(tokens).all())
    ):
        raise TaskCoupledRecurrentDynamicsError(f"{role} context tokens changed")
    return tokens


def _pool_context_grid_v1(tokens: torch.Tensor, *, role: str) -> torch.Tensor:
    source = validate_context_tokens_v1(tokens, role=role).to(torch.float64)
    pooled = source.reshape(
        STATE_COUNT,
        CONTEXT_STEPS,
        4,
        4,
        4,
        4,
        RAW_VISUAL_WIDTH,
    ).mean(dim=(3, 5))
    result = pooled.reshape(
        STATE_COUNT, CONTEXT_STEPS, TOKEN_COUNT, RAW_VISUAL_WIDTH
    ).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise TaskCoupledRecurrentDynamicsError("pooled context became nonfinite")
    return result


def fit_visual_projection_v1(train_tokens: torch.Tensor) -> dict[str, object]:
    """Fit one deterministic train-only PCA across all spatial-temporal cells."""

    pooled = _pool_context_grid_v1(train_tokens, role="train")
    rows = pooled.reshape(-1, RAW_VISUAL_WIDTH).numpy()
    mean = rows.mean(axis=0, dtype=np.float64)
    centered = rows - mean
    _u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    components = np.ascontiguousarray(vh[:VISUAL_WIDTH].T, dtype=np.float64)
    for component in range(VISUAL_WIDTH):
        pivot = int(np.argmax(np.abs(components[:, component])))
        if components[pivot, component] < 0.0:
            components[:, component] *= -1.0
    projected = centered @ components
    scale = np.sqrt(np.mean(np.square(projected), axis=0, dtype=np.float64))
    scale = np.where(scale < 1.0e-8, 1.0, scale)
    result: dict[str, object] = {
        "schema": VISUAL_PROJECTION_SCHEMA,
        "training_rows": int(rows.shape[0]),
        "mean": torch.from_numpy(np.ascontiguousarray(mean)),
        "components": torch.from_numpy(components),
        "score_scale": torch.from_numpy(np.ascontiguousarray(scale)),
        "singular_values": torch.from_numpy(
            np.ascontiguousarray(singular_values[:VISUAL_WIDTH])
        ),
    }
    result["identity_sha256"] = _statistics_identity_v1(result)
    return result


def project_context_tokens_v1(
    tokens: torch.Tensor, projection: Mapping[str, object], *, role: str
) -> torch.Tensor:
    if (
        projection.get("schema") != VISUAL_PROJECTION_SCHEMA
        or projection.get("training_rows")
        != STATE_COUNT * CONTEXT_STEPS * TOKEN_COUNT
        or projection.get("identity_sha256")
        != _statistics_identity_v1(projection, omit_identity=True)
    ):
        raise TaskCoupledRecurrentDynamicsError("visual projection changed")
    pooled = _pool_context_grid_v1(tokens, role=role)
    mean = projection.get("mean")
    components = projection.get("components")
    scale = projection.get("score_scale")
    if (
        not isinstance(mean, torch.Tensor)
        or tuple(mean.shape) != (RAW_VISUAL_WIDTH,)
        or not isinstance(components, torch.Tensor)
        or tuple(components.shape) != (RAW_VISUAL_WIDTH, VISUAL_WIDTH)
        or not isinstance(scale, torch.Tensor)
        or tuple(scale.shape) != (VISUAL_WIDTH,)
    ):
        raise TaskCoupledRecurrentDynamicsError("visual projection tensors changed")
    result = (((pooled - mean) @ components) / scale).to(torch.float32)
    return validate_projected_context_v1(result, role=role)


def validate_projected_context_v1(
    tokens: torch.Tensor, *, role: str
) -> torch.Tensor:
    if (
        role not in {"train", "eval"}
        or not isinstance(tokens, torch.Tensor)
        or tuple(tokens.shape)
        != (STATE_COUNT, CONTEXT_STEPS, TOKEN_COUNT, VISUAL_WIDTH)
        or tokens.device.type != "cpu"
        or tokens.dtype != torch.float32
        or not bool(torch.isfinite(tokens).all())
    ):
        raise TaskCoupledRecurrentDynamicsError(
            f"{role} projected context tokens changed"
        )
    return tokens


def raw_temporal_inputs_v1(role: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ordered transition rows ``(N,3,18)`` and candidate tapes."""

    _validate_role_v1(role)
    physical_inputs = role.physical_inputs
    fixed = physical_inputs[:, 0, :10]
    if not torch.equal(
        physical_inputs[:, :, :10], fixed[:, None, :].expand(-1, ACTION_COUNT, -1)
    ):
        raise TaskCoupledRecurrentDynamicsError(
            "pre-candidate physical history differs across branches"
        )
    motion = torch.zeros((STATE_COUNT, CONTEXT_STEPS, MOTION_WIDTH), dtype=torch.float32)
    motion[:, 1] = torch.cat((fixed[:, :3], role.history_commands[:, 0]), dim=-1)
    motion[:, 2] = torch.cat((fixed[:, 3:6], role.history_commands[:, 1]), dim=-1)
    return motion, role.candidate_commands.contiguous()


def fit_input_statistics_v1(role: Any) -> dict[str, object]:
    motion, candidates = raw_temporal_inputs_v1(role)
    if role.role != "train":
        raise TaskCoupledRecurrentDynamicsError("input statistics must be train-only")
    transition_rows = motion[:, 1:].reshape(-1, MOTION_WIDTH).to(torch.float64)
    candidate_rows = candidates.reshape(-1, CANDIDATE_WIDTH).to(torch.float64)

    def moments(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = rows.mean(dim=0)
        scale = torch.sqrt(torch.mean((rows - mean).square(), dim=0))
        scale = torch.where(scale < 1.0e-8, torch.ones_like(scale), scale)
        return mean.to(torch.float32), scale.to(torch.float32)

    motion_mean, motion_scale = moments(transition_rows)
    candidate_mean, candidate_scale = moments(candidate_rows)
    result: dict[str, object] = {
        "train_identity_sha256": str(role.identity_sha256),
        "motion_mean": motion_mean,
        "motion_scale": motion_scale,
        "candidate_mean": candidate_mean,
        "candidate_scale": candidate_scale,
    }
    result["identity_sha256"] = hashlib.sha256(
        canonical_bytes_v1(
            {
                "train_identity_sha256": result["train_identity_sha256"],
                "motion_mean": _tensor_sha256(motion_mean),
                "motion_scale": _tensor_sha256(motion_scale),
                "candidate_mean": _tensor_sha256(candidate_mean),
                "candidate_scale": _tensor_sha256(candidate_scale),
            }
        )
    ).hexdigest()
    return result


def normalize_temporal_inputs_v1(
    role: Any, statistics: Mapping[str, object]
) -> tuple[torch.Tensor, torch.Tensor]:
    motion, candidates = raw_temporal_inputs_v1(role)
    required = ("motion_mean", "motion_scale", "candidate_mean", "candidate_scale")
    if any(not isinstance(statistics.get(name), torch.Tensor) for name in required):
        raise TaskCoupledRecurrentDynamicsError("input statistics changed")
    result_motion = torch.zeros_like(motion)
    result_motion[:, 1:] = (
        motion[:, 1:] - statistics["motion_mean"]
    ) / statistics["motion_scale"]
    result_candidates = (
        candidates - statistics["candidate_mean"]
    ) / statistics["candidate_scale"]
    if not bool(torch.isfinite(result_motion).all()) or not bool(
        torch.isfinite(result_candidates).all()
    ):
        raise TaskCoupledRecurrentDynamicsError("normalized input became nonfinite")
    return result_motion, result_candidates


def training_batches_v1(seed: int = SAMPLER_SEED) -> tuple[torch.Tensor, ...]:
    if seed != SAMPLER_SEED:
        raise TaskCoupledRecurrentDynamicsError("training seed changed")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result: list[torch.Tensor] = []
    while len(result) < UPDATES:
        permutation = torch.randperm(STATE_COUNT, generator=generator)
        result.extend(permutation.split(BATCH_STATES))
    rows = tuple(result[:UPDATES])
    if any(tuple(row.shape) != (BATCH_STATES,) for row in rows):
        raise TaskCoupledRecurrentDynamicsError("training batch geometry changed")
    return rows


def _clone_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


@torch.no_grad()
def predict_member_v1(
    state: Mapping[str, torch.Tensor],
    role: Any,
    context_tokens: torch.Tensor,
    input_statistics: Mapping[str, object],
    outcome_statistics: Mapping[str, object],
    *,
    arm: str,
    device: torch.device,
) -> torch.Tensor:
    _validate_role_v1(role)
    context = validate_projected_context_v1(context_tokens, role=role.role)
    motion, candidates = normalize_temporal_inputs_v1(role, input_statistics)
    model = TaskCoupledRecurrentDynamicsV1().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    rows: list[torch.Tensor] = []
    for start in range(0, STATE_COUNT, BATCH_STATES):
        stop = start + BATCH_STATES
        visual = context[start:stop].to(device)
        if arm == NO_VISION_ARM:
            visual = torch.zeros_like(visual)
        elif arm != VISUAL_ARM:
            raise TaskCoupledRecurrentDynamicsError("prediction arm changed")
        standardized = model(
            visual,
            motion[start:stop].to(device),
            candidates[start:stop].to(device),
        )
        rows.append(
            grounded.decode_standardized_outcomes_v1(
                standardized, outcome_statistics
            ).cpu()
        )
    result = torch.cat(rows)
    if tuple(result.shape) != (STATE_COUNT, ACTION_COUNT, OUTPUT_WIDTH):
        raise TaskCoupledRecurrentDynamicsError("member prediction geometry changed")
    return result


@torch.no_grad()
def _training_trace_row_v1(
    model: TaskCoupledRecurrentDynamicsV1,
    role: Any,
    context_tokens: torch.Tensor,
    motion: torch.Tensor,
    candidates: torch.Tensor,
    standardized_targets: torch.Tensor,
    outcome_statistics: Mapping[str, object],
    *,
    arm: str,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    predictions: list[torch.Tensor] = []
    mse_total = 0.0
    rank_total = 0.0
    for start in range(0, STATE_COUNT, BATCH_STATES):
        stop = start + BATCH_STATES
        visual = context_tokens[start:stop].to(device)
        if arm == NO_VISION_ARM:
            visual = torch.zeros_like(visual)
        standardized = model(
            visual,
            motion[start:stop].to(device),
            candidates[start:stop].to(device),
        )
        target = standardized_targets[start:stop].to(device)
        decoded = grounded.decode_standardized_outcomes_v1(
            standardized, outcome_statistics
        )
        costs = grounded.predicted_physical_cost_v1(
            decoded, role.relative_goals[start:stop].to(device)
        )
        mse_total += float(torch.mean((standardized - target).square())) * (
            stop - start
        )
        rank_total += float(
            grounded.strict_rank_pairwise_softplus_loss_v1(
                costs, role.dense_ranks[start:stop].to(device)
            )
        ) * (stop - start)
        predictions.append(decoded.cpu())
    outcomes = torch.cat(predictions)
    scores = grounded.physical_score_matrix_v1(role.plan, outcomes)
    report = grounded.report_physical_scores_v1(role.plan, scores)
    return {
        "physical_mse": mse_total / STATE_COUNT,
        "physical_rank_loss": rank_total / STATE_COUNT,
        "normalized_rank_regret": float(
            report["summary"]["normalized_rank_regret"]
        ),
    }


def train_member_v1(
    role: Any,
    context_tokens: torch.Tensor,
    input_statistics: Mapping[str, object],
    outcome_statistics: Mapping[str, object],
    *,
    arm: str,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    if arm not in ARM_ORDER:
        raise TaskCoupledRecurrentDynamicsError("training arm changed")
    _validate_role_v1(role)
    if role.role != "train":
        raise TaskCoupledRecurrentDynamicsError("member training must be train-only")
    context = validate_projected_context_v1(context_tokens, role="train")
    motion, candidates = normalize_temporal_inputs_v1(role, input_statistics)
    standardized_targets = grounded.standardize_outcome_residuals_v1(
        role.targets, outcome_statistics
    )
    model = initialize_task_coupled_recurrent_dynamics_v1(seed).to(device)
    initial_identity = recurrent_dynamics_state_identity_v1(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=WEIGHT_DECAY,
        foreach=False,
        fused=False,
    )
    trace = [
        {
            "update": 0,
            **_training_trace_row_v1(
                model,
                role,
                context,
                motion,
                candidates,
                standardized_targets,
                outcome_statistics,
                arm=arm,
                device=device,
            ),
        }
    ]
    started = time.perf_counter()
    last_objective: dict[str, float] = {}
    for update, indices in enumerate(training_batches_v1(), start=1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        visual = context[indices].to(device)
        if arm == NO_VISION_ARM:
            visual = torch.zeros_like(visual)
        prediction = model(
            visual,
            motion[indices].to(device),
            candidates[indices].to(device),
        )
        target = standardized_targets[indices].to(device)
        physical_mse = torch.mean((prediction - target).square())
        decoded = grounded.decode_standardized_outcomes_v1(
            prediction, outcome_statistics
        )
        costs = grounded.predicted_physical_cost_v1(
            decoded, role.relative_goals[indices].to(device)
        )
        rank_loss = grounded.strict_rank_pairwise_softplus_loss_v1(
            costs, role.dense_ranks[indices].to(device)
        )
        total = physical_mse + RANK_WEIGHT * rank_loss
        if not bool(torch.isfinite(total)):
            raise TaskCoupledRecurrentDynamicsError("training objective became nonfinite")
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRADIENT_CLIP_NORM
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise TaskCoupledRecurrentDynamicsError("training gradient became nonfinite")
        optimizer.step()
        last_objective = {
            "physical_mse": float(physical_mse.detach()),
            "physical_rank_loss": float(rank_loss.detach()),
            "total": float(total.detach()),
            "gradient_norm_before_clip": float(gradient_norm.detach()),
        }
        if update in TRACE_UPDATES[1:]:
            trace.append(
                {
                    "update": update,
                    **_training_trace_row_v1(
                        model,
                        role,
                        context,
                        motion,
                        candidates,
                        standardized_targets,
                        outcome_statistics,
                        arm=arm,
                        device=device,
                    ),
                    "last_minibatch_objective": last_objective,
                }
            )
    state = _clone_state(model.state_dict())
    return {
        "seed": seed,
        "initial_state_identity_sha256": initial_identity,
        "state_dict": state,
        "state_identity_sha256": recurrent_dynamics_state_identity_v1(state),
        "updates": UPDATES,
        "trace": trace,
        "training_seconds": time.perf_counter() - started,
    }


def _statistics_identity_v1(
    value: Mapping[str, object], *, omit_identity: bool = False
) -> str:
    payload = {}
    for name, item in sorted(value.items()):
        if omit_identity and name == "identity_sha256":
            continue
        payload[name] = _tensor_sha256(item) if isinstance(item, torch.Tensor) else item
    return hashlib.sha256(canonical_bytes_v1(payload)).hexdigest()


def checkpoint_identity_v1(checkpoint: Mapping[str, object]) -> str:
    arms = checkpoint.get("arms")
    if not isinstance(arms, Mapping):
        raise TaskCoupledRecurrentDynamicsError("checkpoint arms changed")
    payload = {
        "schema": checkpoint.get("schema"),
        "config": checkpoint.get("config"),
        "train_plan_identity_sha256": checkpoint.get("train_plan_identity_sha256"),
        "train_role_identity_sha256": checkpoint.get("train_role_identity_sha256"),
        "input_statistics": _statistics_identity_v1(checkpoint["input_statistics"]),
        "outcome_statistics": _statistics_identity_v1(
            checkpoint["outcome_statistics"]
        ),
        "visual_projection": _statistics_identity_v1(
            checkpoint["visual_projection"]
        ),
        "arms": {
            arm: [
                {
                    "seed": member["seed"],
                    "initial": member["initial_state_identity_sha256"],
                    "state": recurrent_dynamics_state_identity_v1(member["state_dict"]),
                    "updates": member["updates"],
                    "trace": member["trace"],
                }
                for member in arms[arm]
            ]
            for arm in ARM_ORDER
        },
    }
    return hashlib.sha256(canonical_bytes_v1(payload)).hexdigest()


def fit_checkpoint_v1(
    train_role: Any, train_context_tokens: torch.Tensor, *, device: torch.device
) -> dict[str, object]:
    _validate_role_v1(train_role)
    validate_context_tokens_v1(train_context_tokens, role="train")
    visual_projection = fit_visual_projection_v1(train_context_tokens)
    projected_context = project_context_tokens_v1(
        train_context_tokens, visual_projection, role="train"
    )
    input_statistics = fit_input_statistics_v1(train_role)
    outcome_statistics = grounded.fit_outcome_statistics_v1(train_role.targets)
    arms: dict[str, object] = {}
    initial_by_seed: dict[int, str] = {}
    for arm in ARM_ORDER:
        members = []
        for seed in MODEL_SEEDS:
            member = train_member_v1(
                train_role,
                projected_context,
                input_statistics,
                outcome_statistics,
                arm=arm,
                seed=seed,
                device=device,
            )
            initial = str(member["initial_state_identity_sha256"])
            if seed in initial_by_seed and initial_by_seed[seed] != initial:
                raise TaskCoupledRecurrentDynamicsError(
                    "matched arm initial states differ"
                )
            initial_by_seed[seed] = initial
            members.append(member)
        arms[arm] = members
    checkpoint: dict[str, object] = {
        "schema": CHECKPOINT_SCHEMA,
        "config": config_v1(),
        "train_plan_identity_sha256": str(train_role.plan.identity_sha256),
        "train_role_identity_sha256": str(train_role.identity_sha256),
        "input_statistics": input_statistics,
        "outcome_statistics": outcome_statistics,
        "visual_projection": visual_projection,
        "arms": arms,
    }
    checkpoint["identity_sha256"] = checkpoint_identity_v1(checkpoint)
    return checkpoint


def _fit_task_control_v1(train_plan: Any) -> ActionSpecificRidgeReadoutsV1:
    features = np.stack(
        [
            task_conditioned_feature_v1(
                None, relative_target_xy_body_m=state.relative_target_xy_body_m
            )
            for state in train_plan.states
        ]
    ).astype(np.float64, copy=False)
    targets = []
    for state in train_plan.states:
        ranks = np.asarray(state.dense_ranks, dtype=np.float64)
        targets.append(ranks / ranks.max())
    target_matrix = np.stack(targets)
    feature_sets = [features for _ in range(ACTION_COUNT)]
    target_sets = [target_matrix[:, action] for action in range(ACTION_COUNT)]
    first = fit_action_specific_ridge_readouts_v1(
        feature_sets, target_sets, ridge_lambda=TASK_RIDGE_LAMBDA
    )
    second = fit_action_specific_ridge_readouts_v1(
        feature_sets, target_sets, ridge_lambda=TASK_RIDGE_LAMBDA
    )
    if first.identity_sha256 != second.identity_sha256:
        raise TaskCoupledRecurrentDynamicsError("task control refit is not repeatable")
    return first


def _score_task_control_v1(plan: Any, readouts: ActionSpecificRidgeReadoutsV1) -> np.ndarray:
    rows = []
    for state in plan.states:
        feature = task_conditioned_feature_v1(
            None, relative_target_xy_body_m=state.relative_target_xy_body_m
        )
        rows.append(
            predict_action_specific_scores_v1(
                readouts, [feature for _ in range(ACTION_COUNT)]
            )
        )
    return np.stack(rows)


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
    if checkpoint.get("identity_sha256") != checkpoint_identity_v1(checkpoint):
        raise TaskCoupledRecurrentDynamicsError("checkpoint identity changed")
    _validate_role_v1(train_role)
    _validate_role_v1(eval_role)
    validate_context_tokens_v1(eval_context_tokens, role="eval")
    projected_eval_context = project_context_tokens_v1(
        eval_context_tokens, checkpoint["visual_projection"], role="eval"
    )
    reports: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for arm in ARM_ORDER:
        predictions = []
        members = []
        for member in checkpoint["arms"][arm]:
            outcomes = predict_member_v1(
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
                    "outcome_identity_sha256": _tensor_sha256(outcomes),
                    "score_identity_sha256": physical._array_identity_v1(scores),
                    "report": report,
                }
            )
        ensemble = torch.stack(predictions).mean(dim=0)
        scores = grounded.physical_score_matrix_v1(eval_role.plan, ensemble)
        reports[arm] = grounded.report_physical_scores_v1(eval_role.plan, scores)
        artifacts[arm] = {
            "ensemble_outcomes": ensemble.tolist(),
            "ensemble_outcome_identity_sha256": _tensor_sha256(ensemble),
            "ensemble_scores": scores.tolist(),
            "ensemble_score_identity_sha256": physical._array_identity_v1(scores),
            "members": members,
        }
        diagnostics[arm] = _prediction_diagnostics_v1(
            ensemble,
            eval_role.targets,
            checkpoint["outcome_statistics"]["residual_scales"],
        )

    task = _fit_task_control_v1(train_role.plan)
    task_scores = _score_task_control_v1(eval_role.plan, task)
    reports["task_action_only"] = grounded.report_physical_scores_v1(
        eval_role.plan, task_scores
    )
    task_regret = float(
        reports["task_action_only"]["summary"]["normalized_rank_regret"]
    )
    if task_regret != physical.EXPECTED_TASK_EVAL_REGRET:
        raise TaskCoupledRecurrentDynamicsError(
            "task/action behavioral control changed"
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
        "visual_vs_task_action_only": grounded.paired_family_scene_bootstrap_v1(
            reports[VISUAL_ARM]["group_results"],
            reports["task_action_only"]["group_results"],
        ),
        "visual_vs_no_vision": grounded.paired_family_scene_bootstrap_v1(
            reports[VISUAL_ARM]["group_results"],
            reports[NO_VISION_ARM]["group_results"],
        ),
        "no_vision_vs_task_action_only": grounded.paired_family_scene_bootstrap_v1(
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
    gate = {
        "schema": "lewm_go2_task_coupled_recurrent_dynamics_fixed_gate_v1",
        "passed": bool(frozen_gate["passed"]),
        "status": PASS_STATUS if frozen_gate["passed"] else STOP_STATUS,
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
    }
    return {
        "schema": SCHEMA,
        "status": gate["status"],
        "claim_scope": "DEVELOPMENT_ONLY_H1_MECHANISM_EVIDENCE",
        "checkpoint_identity_sha256": checkpoint["identity_sha256"],
        "train_role_identity_sha256": str(train_role.identity_sha256),
        "eval_role_identity_sha256": str(eval_role.identity_sha256),
        "reports": reports,
        "comparisons": comparisons,
        "gate": gate,
        "prediction_artifacts": artifacts,
        "prediction_diagnostics": diagnostics,
        "task_control": {
            "live_identity_sha256": task.identity_sha256,
            "behavioral_eval_regret": task_regret,
            "expected_behavioral_eval_regret": physical.EXPECTED_TASK_EVAL_REGRET,
        },
        "successor_observations_opened": 0,
        "authorizes_blind_rollout_preregistration": bool(gate["passed"]),
        "authorizes_navigation_claim": False,
    }


__all__ = [
    "ACTION_COUNT",
    "ARM_ORDER",
    "BATCH_STATES",
    "MODEL_SEEDS",
    "NO_VISION_ARM",
    "PASS_STATUS",
    "SCHEMA",
    "STATE_COUNT",
    "STOP_STATUS",
    "TaskCoupledRecurrentDynamicsError",
    "UPDATES",
    "VISUAL_ARM",
    "checkpoint_identity_v1",
    "config_v1",
    "evaluate_checkpoint_v1",
    "fit_checkpoint_v1",
    "fit_input_statistics_v1",
    "normalize_temporal_inputs_v1",
    "predict_member_v1",
    "raw_temporal_inputs_v1",
    "training_batches_v1",
    "validate_context_tokens_v1",
]
