"""Observability-ceiling assay V1: how low can rank regret go from observations?

This module implements the registered assay of
``docs/lewm_go2_observability_ceiling_assay_v1_preregistration_2026-08-05.md``.
It answers an achievability question, not a world-model question: with
prediction error removed by supplying *actual* successor observations, and with
readout capacity swept rather than fixed, what is the lowest scene-disjoint
normalized rank regret any readout of these observations attains?

Dense rank semantics are reused unchanged from
:mod:`lewm.benchmarks.go2_matched_branch_physical_outcome_screen_v1` so that
the assay cannot silently redefine its own target. Scoring uses the registered
complete-tie convention: the regret denominator is ``max(1, max_dense_rank)``
and every action in a complete tie is oracle-equivalent.

The module has no filesystem, RGB, or encoder access; those are runner
responsibilities.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from lewm.benchmarks import go2_matched_branch_physical_outcome_screen_v1 as physical
from lewm.datasets.go2_world_model_counterfactual_pilot_v1 import (
    CANONICAL_ACTION_COMMANDS,
)
from lewm.models.go2_observability_ceiling_readout_v1 import (
    CeilingReadoutV1,
    ceiling_state_identity_v1,
    initialize_ceiling_readout_v1,
    initialize_privileged_mlp_v1,
    parameter_count_v1,
)


SCHEMA = "lewm_go2_observability_ceiling_assay_v1"
CONFIG_SCHEMA = "lewm_go2_observability_ceiling_assay_config_v1"

ACTION_COUNT = 9
CONTEXT_FRAME_COUNT = 3
STATE_COUNT = 128
SCENE_COUNT = 32
TOKEN_COUNT = 256
FAMILY_COUNT = 8
SCENES_PER_FAMILY = 4
STATES_PER_SCENE = 4

# Registered seeds (preregistration section 5.5).
BOOTSTRAP_SEED = 2_026_080_502
SPLIT_SEED = 2_026_080_503
MODEL_SEEDS = (2_026_080_511, 2_026_080_512, 2_026_080_513)
BOOTSTRAP_RESAMPLES = 10_000

# Registered capacity ladder (preregistration section 5.3).
RUNGS = (
    {"name": "rung0", "pca_width": 8, "hidden_width": 4},
    {"name": "rung1", "pca_width": 32, "hidden_width": 32},
    {"name": "rung2", "pca_width": 128, "hidden_width": 128},
)

INNER_VALIDATION_SCENES_PER_FAMILY = 1

# Optimization contract.
EPOCHS = 256
BATCH_STATES = 16
LEARNING_RATE = 3.0e-3
WEIGHT_DECAY = 1.0e-4
GRADIENT_CLIP_NORM = 1.0
TASK_RIDGE_LAMBDA = 1.0e-3
PCA_EPSILON = 1.0e-12

# Registered arm names (preregistration section 5.1).
ORACLE_ARM = "physical_oracle"
PRIVILEGED_ARM = "privileged_physical_successor"
DINO_ARM = "dinov2_true_successor"
VJEPA_ARM = "vjepa2_1_true_successor"
CONTEXT_ARM = "context_only"
TASK_ARM = "task_action_only"
RANDOM_ARM = "random_expected"

ARM_ORDER = (
    ORACLE_ARM,
    PRIVILEGED_ARM,
    DINO_ARM,
    VJEPA_ARM,
    CONTEXT_ARM,
    TASK_ARM,
    RANDOM_ARM,
)

# Registered gates (preregistration sections 5.4 and 6).
CAPACITY_CONTROL_MAX_REGRET = 0.05
ABSOLUTE_GATE = 0.13

OUTCOME_I = "OUTCOME_I_GATE_ACHIEVABLE"
OUTCOME_II = "OUTCOME_II_GATE_TOO_TIGHT_VISUAL_INFORMATION_PRESENT"
OUTCOME_III = "OUTCOME_III_NO_VISUAL_HEADROOM"
OUTCOME_IV = "OUTCOME_IV_PANEL_DEGENERATE"
INCONCLUSIVE = "INCONCLUSIVE_NO_REGISTERED_OUTCOME"
CAPACITY_FAILURE = "FAIL_ASSAY_CAPACITY_CONTROL"

# The privileged physical successor feature (preregistration section 5.1).
PRIVILEGED_FEATURE_NAMES = ("dx_body", "dy_body", "dyaw", "path_length_m", "fell", "tipped")
PRIVILEGED_FEATURE_WIDTH = len(PRIVILEGED_FEATURE_NAMES)


class ObservabilityCeilingAssayError(RuntimeError):
    """Raised when the assay contract is violated."""


def canonical_bytes_v1(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def config_v1() -> dict[str, object]:
    """Return the frozen assay configuration."""

    return {
        "schema": CONFIG_SCHEMA,
        "action_count": ACTION_COUNT,
        "context_frame_count": CONTEXT_FRAME_COUNT,
        "state_count_per_role": STATE_COUNT,
        "scene_count_per_role": SCENE_COUNT,
        "token_count": TOKEN_COUNT,
        "rungs": [dict(rung) for rung in RUNGS],
        "rung_parameter_counts": [
            parameter_count_v1(rung["pca_width"], rung["hidden_width"])
            for rung in RUNGS
        ],
        "epochs": EPOCHS,
        "batch_states": BATCH_STATES,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "task_ridge_lambda": TASK_RIDGE_LAMBDA,
        "model_seeds": list(MODEL_SEEDS),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "split_seed": SPLIT_SEED,
        "arm_order": list(ARM_ORDER),
        "capacity_control_max_regret": CAPACITY_CONTROL_MAX_REGRET,
        "absolute_gate": ABSOLUTE_GATE,
        "rank_tolerance_m": physical.RANK_TOLERANCE_M,
        "regret_denominator": "max(1,max_dense_rank)",
        "privileged_feature_names": list(PRIVILEGED_FEATURE_NAMES),
    }


# --------------------------------------------------------------------------
# Role geometry
# --------------------------------------------------------------------------


def validate_role_geometry_v1(groups: Sequence[Any], *, role: str) -> dict[str, object]:
    """Require the registered 8 x 4 x 4 family/scene/state balance."""

    if role not in {"train", "eval"} or len(groups) != STATE_COUNT:
        raise ObservabilityCeilingAssayError(f"{role} state count changed")
    families = Counter(group.family for group in groups)
    scenes = {(group.family, group.scene_id) for group in groups}
    scenes_by_family = Counter(family for family, _scene in scenes)
    states_by_scene = Counter(group.scene_id for group in groups)
    if (
        len(families) != FAMILY_COUNT
        or any(count != STATE_COUNT // FAMILY_COUNT for count in families.values())
        or len(scenes) != SCENE_COUNT
        or any(count != SCENES_PER_FAMILY for count in scenes_by_family.values())
        or any(count != STATES_PER_SCENE for count in states_by_scene.values())
    ):
        raise ObservabilityCeilingAssayError(f"{role} role balance changed")
    return {
        "role": role,
        "states": len(groups),
        "families": sorted(families),
        "scenes": len(scenes),
        "states_per_scene": STATES_PER_SCENE,
    }


def require_role_disjointness_v1(
    train_groups: Sequence[Any], eval_groups: Sequence[Any]
) -> dict[str, object]:
    """Require that no scene identity appears in both roles."""

    train_scenes = {group.scene_id for group in train_groups}
    eval_scenes = {group.scene_id for group in eval_groups}
    overlap = sorted(train_scenes & eval_scenes)
    if overlap:
        raise ObservabilityCeilingAssayError("train and eval roles share scenes")
    return {
        "train_scenes": len(train_scenes),
        "eval_scenes": len(eval_scenes),
        "shared_scenes": 0,
    }


def inner_split_v1(groups: Sequence[Any]) -> dict[str, tuple[str, ...]]:
    """Split train scenes 24 fit / 8 inner-validation, one per family.

    The split is deterministic in the registered split seed and stratified so
    that exactly one scene per family is held out.
    """

    validate_role_geometry_v1(groups, role="train")
    by_family: dict[str, list[str]] = {}
    for group in groups:
        by_family.setdefault(str(group.family), [])
        if group.scene_id not in by_family[str(group.family)]:
            by_family[str(group.family)].append(str(group.scene_id))
    fit: list[str] = []
    validation: list[str] = []
    for family in sorted(by_family):
        scenes = sorted(by_family[family])
        if len(scenes) != SCENES_PER_FAMILY:
            raise ObservabilityCeilingAssayError("train family scene count changed")
        digest = hashlib.sha256()
        digest.update(str(SPLIT_SEED).encode("ascii") + b"\0")
        digest.update(family.encode("utf-8"))
        index = int.from_bytes(digest.digest()[:8], "little") % SCENES_PER_FAMILY
        for position, scene in enumerate(scenes):
            (validation if position == index else fit).append(scene)
    if (
        len(fit) != SCENE_COUNT - FAMILY_COUNT * INNER_VALIDATION_SCENES_PER_FAMILY
        or len(validation) != FAMILY_COUNT * INNER_VALIDATION_SCENES_PER_FAMILY
        or set(fit) & set(validation)
    ):
        raise ObservabilityCeilingAssayError("inner split geometry changed")
    return {"fit": tuple(sorted(fit)), "validation": tuple(sorted(validation))}


def state_indices_for_scenes_v1(
    groups: Sequence[Any], scenes: Sequence[str]
) -> tuple[int, ...]:
    wanted = set(scenes)
    return tuple(
        index for index, group in enumerate(groups) if str(group.scene_id) in wanted
    )


# --------------------------------------------------------------------------
# Targets, conditions, and privileged features
# --------------------------------------------------------------------------


def dense_rank_matrix_v1(groups: Sequence[Any]) -> np.ndarray:
    rows = []
    for group in groups:
        ranks = [int(branch.oracle_dense_rank) for branch in group.branches]
        if len(ranks) != ACTION_COUNT or any(rank < 0 for rank in ranks):
            raise ObservabilityCeilingAssayError("dense ranks changed")
        rows.append(ranks)
    result = np.asarray(rows, dtype=np.float64)
    if result.shape != (len(groups), ACTION_COUNT):
        raise ObservabilityCeilingAssayError("dense rank matrix shape changed")
    return result


def normalized_rank_targets_v1(groups: Sequence[Any]) -> np.ndarray:
    """Normalize dense ranks by ``max(1, max_dense_rank)``.

    Unlike the frozen predecessor this admits a complete-tie state, whose row
    is all zero, matching the registered complete-tie convention.
    """

    ranks = dense_rank_matrix_v1(groups)
    denominators = np.maximum(1.0, ranks.max(axis=1, keepdims=True))
    result = ranks / denominators
    if not np.isfinite(result).all():
        raise ObservabilityCeilingAssayError("normalized rank targets are invalid")
    return result


def conditions_v1(groups: Sequence[Any]) -> torch.Tensor:
    """Return the frozen ``(states, actions, 4)`` condition tensor."""

    conditions = torch.empty((len(groups), ACTION_COUNT, 4), dtype=torch.float32)
    for state_index, group in enumerate(groups):
        goal_x, goal_y = group.relative_target_xy_body_m
        for action in range(ACTION_COUNT):
            command = CANONICAL_ACTION_COMMANDS[action]
            conditions[state_index, action] = torch.tensor(
                (goal_x / 10.0, goal_y / 10.0, command[0] / 0.30, command[2] / 0.45),
                dtype=torch.float32,
            )
    if not bool(torch.isfinite(conditions).all()):
        raise ObservabilityCeilingAssayError("conditions are nonfinite")
    return conditions


def _yaw_from_quaternion_wxyz(quaternion: Sequence[float]) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def privileged_physical_features_v1(
    groups: Sequence[Any], receipt_by_id: Mapping[str, Mapping[str, Any]]
) -> torch.Tensor:
    """Build the capacity-control feature ``(states, actions, 6)``.

    The feature is the successor kinematic state relative to the predecessor:
    body-frame ``dx``, ``dy``, ``dyaw``, path length, ``fell``, ``tipped``.
    Target progress is recoverable from ``(dx, dy)`` together with the goal
    already present in the condition vector, so this arm can in principle
    reconstruct the dense rank exactly.  It is a control on the expressiveness
    of the readout family, not a scientific arm.
    """

    features = torch.zeros(
        (len(groups), ACTION_COUNT, PRIVILEGED_FEATURE_WIDTH), dtype=torch.float32
    )
    for state_index, group in enumerate(groups):
        receipt = receipt_by_id[group.state_id]
        context = receipt.get("context")
        branches = receipt.get("branches")
        if not isinstance(context, Mapping) or not isinstance(branches, list):
            raise ObservabilityCeilingAssayError("privileged receipt geometry changed")
        base = context.get("prebranch_base_pose_world")
        if not isinstance(base, Mapping):
            raise ObservabilityCeilingAssayError("prebranch pose missing")
        base_xyz = [float(value) for value in base["position_xyz_m"]]
        base_yaw = _yaw_from_quaternion_wxyz(base["quaternion_wxyz"])
        cos_yaw, sin_yaw = math.cos(-base_yaw), math.sin(-base_yaw)
        for action, branch in enumerate(branches):
            if int(branch.get("action_id", -1)) != action:
                raise ObservabilityCeilingAssayError("privileged action grid changed")
            endpoint = branch.get("endpoint_state")
            if not isinstance(endpoint, Mapping):
                raise ObservabilityCeilingAssayError("endpoint state missing")
            end_xyz = [float(value) for value in endpoint["base_pos_world"]]
            end_yaw = _yaw_from_quaternion_wxyz(endpoint["base_quat_wxyz"])
            world_dx = end_xyz[0] - base_xyz[0]
            world_dy = end_xyz[1] - base_xyz[1]
            body_dx = cos_yaw * world_dx - sin_yaw * world_dy
            body_dy = sin_yaw * world_dx + cos_yaw * world_dy
            dyaw = math.atan2(
                math.sin(end_yaw - base_yaw), math.cos(end_yaw - base_yaw)
            )
            features[state_index, action] = torch.tensor(
                (
                    body_dx,
                    body_dy,
                    dyaw,
                    float(branch["physical_path_length_m"]),
                    1.0 if bool(branch["physical_fell"]) else 0.0,
                    1.0 if bool(branch["physical_tipped"]) else 0.0,
                ),
                dtype=torch.float32,
            )
    if not bool(torch.isfinite(features).all()):
        raise ObservabilityCeilingAssayError("privileged features are nonfinite")
    return features


# --------------------------------------------------------------------------
# PCA, relational panels, task ridge
# --------------------------------------------------------------------------


def fit_pca_v1(
    tokens: torch.Tensor, *, width: int, state_indices: Sequence[int]
) -> dict[str, torch.Tensor]:
    """Fit a mean/component PCA on the selected states' tokens only."""

    if tokens.ndim != 4 or tokens.shape[1] < 1 or tokens.shape[2] != TOKEN_COUNT:
        raise ObservabilityCeilingAssayError("token cache shape changed")
    selected = tokens[list(state_indices)]
    flat = selected.reshape(-1, selected.shape[-1]).to(torch.float64)
    if flat.shape[0] < width:
        raise ObservabilityCeilingAssayError("PCA has fewer rows than components")
    mean = flat.mean(dim=0)
    centered = flat - mean
    # Economy SVD on the centered matrix; components are right singular vectors.
    _u, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    if int(vh.shape[0]) < width:
        raise ObservabilityCeilingAssayError("PCA rank is below the requested width")
    components = vh[:width]
    explained = (singular[:width] ** 2).sum() / (singular**2).sum().clamp_min(
        PCA_EPSILON
    )
    return {
        "mean": mean.to(torch.float32),
        "components": components.to(torch.float32),
        "explained_variance_ratio": float(explained),
        "width": int(width),
    }


def project_tokens_v1(
    tokens: torch.Tensor, pca: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    mean = pca["mean"]
    components = pca["components"]
    if tokens.shape[-1] != mean.shape[0]:
        raise ObservabilityCeilingAssayError("PCA token dimension changed")
    projected = (tokens - mean) @ components.T
    if not bool(torch.isfinite(projected).all()):
        raise ObservabilityCeilingAssayError("projected tokens are nonfinite")
    return projected.to(torch.float32)


def relational_panel_v1(
    current: torch.Tensor, successor: torch.Tensor
) -> torch.Tensor:
    """Return ``[z_c, z_s, z_s - z_c]`` for one batch of branches."""

    if current.shape != successor.shape:
        raise ObservabilityCeilingAssayError("relational panel shapes disagree")
    return torch.cat((current, successor, successor - current), dim=-1)


def broadcast_feature_panel_v1(
    features: torch.Tensor, *, relational_width: int
) -> torch.Tensor:
    """Embed a flat feature vector into a dense panel by zero-padded replication.

    This lets the privileged capacity control travel through the *same* readout
    family as the visual arms rather than a different one.
    """

    if features.ndim != 2 or features.shape[-1] > relational_width:
        raise ObservabilityCeilingAssayError("privileged feature panel shape changed")
    padded = torch.zeros(
        (features.shape[0], relational_width), dtype=torch.float32
    )
    padded[:, : features.shape[-1]] = features
    return padded.unsqueeze(1).expand(-1, TOKEN_COUNT, -1).contiguous()


def fit_task_ridge_v1(
    groups: Sequence[Any], state_indices: Sequence[int]
) -> dict[str, np.ndarray]:
    """Fit one ridge head per action on the goal-only task feature."""

    targets = normalized_rank_targets_v1(groups)
    features = np.stack(
        [
            np.asarray(
                (
                    1.0,
                    group.relative_target_xy_body_m[0],
                    group.relative_target_xy_body_m[1],
                    math.hypot(*group.relative_target_xy_body_m),
                ),
                dtype=np.float64,
            )
            for group in groups
        ]
    )
    selected = np.asarray(list(state_indices), dtype=np.int64)
    design = features[selected]
    gram = design.T @ design + TASK_RIDGE_LAMBDA * np.eye(design.shape[1])
    coefficients = np.stack(
        [
            np.linalg.solve(gram, design.T @ targets[selected, action])
            for action in range(ACTION_COUNT)
        ]
    )
    return {"features": features, "coefficients": coefficients}


def score_task_ridge_v1(ridge: Mapping[str, np.ndarray]) -> np.ndarray:
    scores = ridge["features"] @ ridge["coefficients"].T
    if not np.isfinite(scores).all():
        raise ObservabilityCeilingAssayError("task ridge scores are invalid")
    return scores


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def train_readout_v1(
    build_panel,
    conditions: torch.Tensor,
    residual_targets: torch.Tensor,
    state_indices: Sequence[int],
    *,
    seed: int,
    pca_width: int,
    hidden_width: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Fit one readout member by AdamW on the mean-squared residual."""

    model = initialize_ceiling_readout_v1(
        seed, pca_width=pca_width, hidden_width=hidden_width
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        amsgrad=False,
        foreach=False,
        fused=False,
    )
    indices = torch.as_tensor(list(state_indices), dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    step_count = 0
    last_loss = math.nan
    model.train()
    for _epoch in range(EPOCHS):
        order = indices[torch.randperm(len(indices), generator=generator)]
        for start in range(0, len(order), BATCH_STATES):
            selected = order[start : start + BATCH_STATES]
            if len(selected) < 1:
                continue
            panel = build_panel(selected).to(device)
            batch_conditions = conditions[selected].reshape(-1, 4).to(device)
            batch_targets = residual_targets[selected].reshape(-1).to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(panel, batch_conditions)
            loss = torch.mean((predictions - batch_targets) ** 2)
            if not bool(torch.isfinite(loss)):
                raise ObservabilityCeilingAssayError("training loss is nonfinite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRADIENT_CLIP_NORM, norm_type=2.0
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise ObservabilityCeilingAssayError("gradient norm is nonfinite")
            optimizer.step()
            step_count += 1
            last_loss = float(loss.detach())
    model.eval()
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    diagnostics = {
        "seed": seed,
        "pca_width": int(pca_width),
        "hidden_width": int(hidden_width),
        "parameter_count": parameter_count_v1(pca_width, hidden_width),
        "optimizer_steps": step_count,
        "final_batch_mse": last_loss,
        "state_identity_sha256": ceiling_state_identity_v1(
            state, state_shapes=model.state_shapes
        ),
    }
    return state, diagnostics


def closed_form_identifiability_scores_v1(
    groups: Sequence[Any], features: torch.Tensor
) -> np.ndarray:
    """Reconstruct the dense rank analytically from privileged physical state.

    This is the corrected identifiability control.  Unlike a learned control it
    has no parameters, no training set, and therefore no cross-scene
    generalization confound: it asks only whether the dense rank *is* a function
    of the privileged successor state, not whether that function can be learned
    from 128 states.

    Target progress is reconstructed as ``|g| - |g - d|`` from the body-frame
    displacement in ``features`` and the goal carried by the group.  The
    remaining rank-key components -- path length, fall and tip flags -- are read
    directly from ``features``.  The exact frozen rank rule of
    :mod:`lewm.benchmarks.go2_matched_branch_physical_outcome_screen_v1` is then
    applied unchanged.

    Returns lower-is-better scores, so ``argmin`` selects the reconstructed
    best branch.
    """

    if tuple(features.shape) != (
        len(groups),
        ACTION_COUNT,
        PRIVILEGED_FEATURE_WIDTH,
    ):
        raise ObservabilityCeilingAssayError("privileged feature shape changed")
    rows = []
    for state_index, group in enumerate(groups):
        goal_x, goal_y = group.relative_target_xy_body_m
        goal_distance = math.hypot(float(goal_x), float(goal_y))
        keys = []
        for action in range(ACTION_COUNT):
            dx = float(features[state_index, action, 0])
            dy = float(features[state_index, action, 1])
            path = float(features[state_index, action, 3])
            fell = bool(features[state_index, action, 4] >= 0.5)
            tipped = bool(features[state_index, action, 5] >= 0.5)
            progress = goal_distance - math.hypot(goal_x - dx, goal_y - dy)
            keys.append(
                (
                    int(fell),
                    int(tipped),
                    -physical._quantize(progress),  # noqa: SLF001
                    physical._quantize(path),  # noqa: SLF001
                )
            )
        mapping = {key: rank for rank, key in enumerate(sorted(set(keys)))}
        rows.append([mapping[key] for key in keys])
    result = np.asarray(rows, dtype=np.float64)
    if result.shape != (len(groups), ACTION_COUNT):
        raise ObservabilityCeilingAssayError("closed-form rank shape changed")
    return result


def train_privileged_mlp_v1(
    features: torch.Tensor,
    conditions: torch.Tensor,
    residual_targets: torch.Tensor,
    state_indices: Sequence[int],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Fit amendment-1 control 2a on the same schedule as the readout arms."""

    model = initialize_privileged_mlp_v1(
        seed, feature_width=features.shape[-1]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        foreach=False,
        fused=False,
    )
    indices = torch.as_tensor(list(state_indices), dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model.train()
    for _epoch in range(EPOCHS):
        order = indices[torch.randperm(len(indices), generator=generator)]
        for start in range(0, len(order), BATCH_STATES):
            selected = order[start : start + BATCH_STATES]
            if len(selected) < 1:
                continue
            batch_features = features[selected].reshape(-1, features.shape[-1]).to(device)
            batch_conditions = conditions[selected].reshape(-1, 4).to(device)
            batch_targets = residual_targets[selected].reshape(-1).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean(
                (model(batch_features, batch_conditions) - batch_targets) ** 2
            )
            if not bool(torch.isfinite(loss)):
                raise ObservabilityCeilingAssayError("privileged MLP loss is nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRADIENT_CLIP_NORM, norm_type=2.0
            )
            optimizer.step()
    model.eval()
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def predict_privileged_mlp_v1(
    states: Sequence[Mapping[str, torch.Tensor]],
    features: torch.Tensor,
    conditions: torch.Tensor,
    *,
    device: torch.device,
) -> np.ndarray:
    """Average control-2a member predictions over the registered seeds."""

    total_states = conditions.shape[0]
    accumulated = np.zeros((total_states, ACTION_COUNT), dtype=np.float64)
    for state in states:
        model = initialize_privileged_mlp_v1(
            MODEL_SEEDS[0], feature_width=features.shape[-1]
        ).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        with torch.no_grad():
            scores = model(
                features.reshape(-1, features.shape[-1]).to(device),
                conditions.reshape(-1, 4).to(device),
            )
        accumulated += (
            scores.reshape(total_states, ACTION_COUNT).detach().cpu().numpy()
        ).astype(np.float64)
    return accumulated / float(len(states))


def predict_scores_v1(
    states: Sequence[Mapping[str, torch.Tensor]],
    build_panel,
    conditions: torch.Tensor,
    *,
    pca_width: int,
    hidden_width: int,
    device: torch.device,
    batch_states: int = 16,
) -> np.ndarray:
    """Average member residual predictions over the registered seeds."""

    total_states = conditions.shape[0]
    accumulated = np.zeros((total_states, ACTION_COUNT), dtype=np.float64)
    for state in states:
        model = CeilingReadoutV1(
            pca_width=pca_width, hidden_width=hidden_width
        ).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        rows = []
        with torch.no_grad():
            for start in range(0, total_states, batch_states):
                selected = torch.arange(
                    start, min(start + batch_states, total_states), dtype=torch.long
                )
                panel = build_panel(selected).to(device)
                batch_conditions = conditions[selected].reshape(-1, 4).to(device)
                scores = model(panel, batch_conditions)
                rows.append(
                    scores.reshape(len(selected), ACTION_COUNT).detach().cpu().numpy()
                )
        accumulated += np.concatenate(rows, axis=0).astype(np.float64)
    return accumulated / float(len(states))


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def regret_rows_v1(
    groups: Sequence[Any], scores: np.ndarray | None, *, policy: str
) -> list[dict[str, object]]:
    """Score one arm under the registered complete-tie convention.

    ``policy`` is ``"argmin"`` for a scored arm, ``"oracle"`` for the
    privileged rank anchor, or ``"random"`` for uniform expectation.
    """

    ranks = dense_rank_matrix_v1(groups)
    rows: list[dict[str, object]] = []
    for index, group in enumerate(groups):
        state_ranks = ranks[index]
        denominator = max(1.0, float(state_ranks.max()))
        oracle_equivalent = state_ranks == state_ranks.min()
        if policy == "random":
            regret = float(state_ranks.mean() / denominator)
            selected: object = "NOT_APPLICABLE"
            equivalent = float(oracle_equivalent.mean())
        elif policy == "oracle":
            regret = 0.0
            selected = int(np.argmin(state_ranks))
            equivalent = 1.0
        elif policy == "argmin":
            if scores is None:
                raise ObservabilityCeilingAssayError("scored arm requires scores")
            action = int(np.argmin(scores[index]))
            regret = float(state_ranks[action] / denominator)
            selected = action
            equivalent = 1.0 if bool(oracle_equivalent[action]) else 0.0
        else:
            raise ObservabilityCeilingAssayError("selection policy changed")
        rows.append(
            {
                "state_id": str(group.state_id),
                "scene_id": str(group.scene_id),
                "family": str(group.family),
                "selected_action_id": selected,
                "normalized_rank_regret": regret,
                "oracle_equivalent_selection": equivalent,
            }
        )
    return rows


def summarize_rows_v1(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "states": len(rows),
        "normalized_rank_regret": float(
            np.mean([float(row["normalized_rank_regret"]) for row in rows])
        ),
        "oracle_equivalent_selection": float(
            np.mean([float(row["oracle_equivalent_selection"]) for row in rows])
        ),
    }


def arm_report_v1(
    groups: Sequence[Any], scores: np.ndarray | None, *, policy: str
) -> dict[str, object]:
    rows = regret_rows_v1(groups, scores, policy=policy)
    families = sorted({str(row["family"]) for row in rows})
    scenes = sorted({str(row["scene_id"]) for row in rows})
    return {
        "selection_policy": policy,
        "summary": summarize_rows_v1(rows),
        "state_results": rows,
        "per_family": {
            family: summarize_rows_v1(
                [row for row in rows if row["family"] == family]
            )
            for family in families
        },
        "per_scene": {
            scene: summarize_rows_v1(
                [row for row in rows if row["scene_id"] == scene]
            )
            for scene in scenes
        },
    }


def paired_family_scene_bootstrap_v1(
    candidate_rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Family-balanced whole-scene cluster bootstrap of the paired difference."""

    if len(candidate_rows) != len(baseline_rows):
        raise ObservabilityCeilingAssayError("paired row counts differ")
    by_scene: dict[str, dict[str, Any]] = {}
    for candidate, baseline in zip(candidate_rows, baseline_rows, strict=True):
        if (
            candidate["state_id"] != baseline["state_id"]
            or candidate["scene_id"] != baseline["scene_id"]
        ):
            raise ObservabilityCeilingAssayError("paired rows are misaligned")
        scene = str(candidate["scene_id"])
        entry = by_scene.setdefault(
            scene, {"family": str(candidate["family"]), "deltas": []}
        )
        entry["deltas"].append(
            float(candidate["normalized_rank_regret"])
            - float(baseline["normalized_rank_regret"])
        )
    families: dict[str, list[str]] = {}
    for scene, entry in by_scene.items():
        families.setdefault(entry["family"], []).append(scene)
    ordered_families = sorted(families)
    scene_means = {
        scene: float(np.mean(entry["deltas"])) for scene, entry in by_scene.items()
    }
    point = float(
        np.mean(
            [
                np.mean([scene_means[scene] for scene in sorted(families[family])])
                for family in ordered_families
            ]
        )
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    family_scenes = [sorted(families[family]) for family in ordered_families]
    for draw in range(BOOTSTRAP_RESAMPLES):
        family_values = []
        for scenes in family_scenes:
            picked = rng.integers(0, len(scenes), size=len(scenes))
            family_values.append(
                float(np.mean([scene_means[scenes[index]] for index in picked]))
            )
        draws[draw] = float(np.mean(family_values))
    lower, upper = (float(value) for value in np.quantile(draws, (0.025, 0.975)))
    return {
        "point_delta": point,
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_half_width": (upper - lower) / 2.0,
        "resamples": BOOTSTRAP_RESAMPLES,
        "scene_clusters": len(by_scene),
        "families": len(ordered_families),
    }


def scenes_to_resolve_effect_v1(half_width: float, effect: float = 0.02) -> float:
    """Scenes required to shrink the CI half-width below ``effect``.

    The bootstrap half-width scales approximately as ``1/sqrt(n)``, so the
    required cluster count is ``n * (h/effect)**2``.
    """

    if effect <= 0.0:
        raise ObservabilityCeilingAssayError("effect size must be positive")
    return float(SCENE_COUNT * (float(half_width) / float(effect)) ** 2)


# --------------------------------------------------------------------------
# Diagnostics and the registered decision rule
# --------------------------------------------------------------------------


def displacement_spread_bins_v1(groups: Sequence[Any]) -> dict[str, object]:
    """Bin states by within-state spread of ``physical_target_progress_m``."""

    spreads = []
    for group in groups:
        values = [
            float(branch.labels.target_progress_m) for branch in group.branches
        ]
        spreads.append(max(values) - min(values))
    array = np.asarray(spreads, dtype=np.float64)
    edges = np.quantile(array, (0.0, 0.25, 0.5, 0.75, 1.0))
    assignment = np.clip(np.searchsorted(edges[1:-1], array, side="right"), 0, 3)
    return {
        "spread_m": [float(value) for value in array],
        "quartile_edges_m": [float(value) for value in edges],
        "bin_index": [int(value) for value in assignment],
    }


def spread_conditioned_regret_v1(
    groups: Sequence[Any], rows: Sequence[Mapping[str, object]], bins: Mapping[str, Any]
) -> dict[str, object]:
    assignment = list(bins["bin_index"])
    result = {}
    for bin_index in range(4):
        selected = [
            row for row, value in zip(rows, assignment, strict=True) if value == bin_index
        ]
        result[f"quartile_{bin_index}"] = (
            summarize_rows_v1(selected) if selected else {"states": 0}
        )
    return result


def decide_v1(
    reports: Mapping[str, Mapping[str, Any]],
    comparisons: Mapping[str, Mapping[str, Any]],
    *,
    identifiability_regret: float,
    expressivity_regret: float,
) -> dict[str, object]:
    """Apply the registered decision rule in the registered order.

    Order is the amendment-1 validity controls, then Outcome I, IV, III, II.
    The first Outcome that holds is the terminal.  No threshold may be relaxed
    here.

    ``identifiability_regret`` is amendment-1 control 2a: the evaluation regret
    of an unconstrained MLP on the privileged physical successor feature, which
    establishes that the dense rank is a learnable function of that state.
    ``expressivity_regret`` is control 2b: the in-sample train regret of the
    primary dense visual arm at the top rung, which establishes that the readout
    family can express a ranking function from genuinely spatially-varying dense
    panels.
    """

    ceiling = float(reports[DINO_ARM]["summary"]["normalized_rank_regret"])
    task = float(reports[TASK_ARM]["summary"]["normalized_rank_regret"])

    validity = {
        "identifiability_regret": float(identifiability_regret),
        "expressivity_train_regret": float(expressivity_regret),
        "threshold": CAPACITY_CONTROL_MAX_REGRET,
        "identifiability_passed": float(identifiability_regret)
        <= CAPACITY_CONTROL_MAX_REGRET,
        "expressivity_passed": float(expressivity_regret)
        <= CAPACITY_CONTROL_MAX_REGRET,
    }
    if not (validity["identifiability_passed"] and validity["expressivity_passed"]):
        return {
            "terminal": CAPACITY_FAILURE,
            "assay_valid": False,
            "validity": validity,
            "reason": (
                "amendment-1 validity control failed: identifiability "
                f"{identifiability_regret} and expressivity {expressivity_regret} "
                f"against the registered {CAPACITY_CONTROL_MAX_REGRET}; no Outcome "
                "may be claimed"
            ),
            "ceiling_regret": ceiling,
        }

    dino_vs_task = comparisons["dinov2_true_successor_minus_task_action_only"]
    context_vs_dino = comparisons["context_only_minus_dinov2_true_successor"]

    if ceiling <= ABSOLUTE_GATE:
        terminal, reason = OUTCOME_I, (
            f"ceiling regret {ceiling} is at or below the registered "
            f"{ABSOLUTE_GATE} gate"
        )
    elif float(context_vs_dino["ci_lower"]) <= 0.0 <= float(context_vs_dino["ci_upper"]):
        terminal, reason = OUTCOME_IV, (
            "actual successors add nothing over context; the context-minus-"
            "ceiling interval includes zero"
        )
    elif ceiling >= task:
        terminal, reason = OUTCOME_III, (
            "the visual ceiling does not beat the non-visual task/action control"
        )
    elif (
        float(dino_vs_task["point_delta"]) < 0.0
        and float(dino_vs_task["ci_upper"]) < 0.0
    ):
        terminal, reason = OUTCOME_II, (
            f"ceiling regret {ceiling} exceeds {ABSOLUTE_GATE} but beats the "
            "task/action control with the whole interval below zero; the gate "
            "must be re-derived ceiling-relative"
        )
    else:
        terminal, reason = INCONCLUSIVE, (
            "no registered Outcome condition holds; no interpretation is claimed"
        )
    return {
        "terminal": terminal,
        "assay_valid": True,
        "validity": validity,
        "reason": reason,
        "ceiling_regret": ceiling,
        "task_regret": task,
        "privileged_bilinear_reference_regret": float(
            reports[PRIVILEGED_ARM]["summary"]["normalized_rank_regret"]
        ),
        "absolute_gate": ABSOLUTE_GATE,
    }


def result_identity_v1(result: Mapping[str, object]) -> str:
    payload = {key: value for key, value in result.items() if key != "identity_sha256"}
    return hashlib.sha256(canonical_bytes_v1(payload)).hexdigest()


__all__ = [
    "ABSOLUTE_GATE",
    "ARM_ORDER",
    "CAPACITY_CONTROL_MAX_REGRET",
    "CONTEXT_ARM",
    "DINO_ARM",
    "MODEL_SEEDS",
    "ORACLE_ARM",
    "PRIVILEGED_ARM",
    "RANDOM_ARM",
    "RUNGS",
    "SCHEMA",
    "TASK_ARM",
    "VJEPA_ARM",
    "ObservabilityCeilingAssayError",
    "arm_report_v1",
    "broadcast_feature_panel_v1",
    "canonical_bytes_v1",
    "closed_form_identifiability_scores_v1",
    "conditions_v1",
    "config_v1",
    "decide_v1",
    "dense_rank_matrix_v1",
    "displacement_spread_bins_v1",
    "fit_pca_v1",
    "fit_task_ridge_v1",
    "inner_split_v1",
    "normalized_rank_targets_v1",
    "paired_family_scene_bootstrap_v1",
    "predict_privileged_mlp_v1",
    "predict_scores_v1",
    "privileged_physical_features_v1",
    "project_tokens_v1",
    "regret_rows_v1",
    "relational_panel_v1",
    "require_role_disjointness_v1",
    "result_identity_v1",
    "scenes_to_resolve_effect_v1",
    "score_task_ridge_v1",
    "spread_conditioned_regret_v1",
    "state_indices_for_scenes_v1",
    "summarize_rows_v1",
    "train_privileged_mlp_v1",
    "train_readout_v1",
    "validate_role_geometry_v1",
]
