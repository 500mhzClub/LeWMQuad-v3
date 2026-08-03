"""Pure mechanics for the preregistered matched-branch physical screen.

This module deliberately has no filesystem, RGB-decoding, or encoder path.  It
consumes raw state-receipt documents that a runner has already rehashed and
validated, together with exact frozen DINO cache objects.  The only cached
tokens read here are the three current-context grids for each state.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from lewm.benchmarks import go2_dinov2_dense_shared_spatial_readout_calibration_v1 as dense
from lewm.benchmarks import go2_dinov2_physical_readout_calibration_v1 as prior
from lewm.datasets.go2_world_model_counterfactual_pilot_v1 import (
    CANONICAL_ACTION_COMMANDS,
)
from lewm.models.go2_matched_branch_physical_outcome_screen_v1 import (
    HIDDEN_WIDTH,
    INPUT_WIDTH,
    OUTPUT_WIDTH,
    PARAMETER_COUNT,
    PhysicalOutcomeMLPV1,
    initialize_physical_outcome_mlp_v1,
    physical_outcome_state_identity_v1,
)
from scripts.evaluate_go2_world_model_counterfactual_action_regret_v1 import (
    selection_metrics_v1,
)


SCHEMA = "lewm_go2_matched_branch_physical_outcome_screen_v1"
CONFIG_SCHEMA = "lewm_go2_matched_branch_physical_outcome_screen_config_v1"
CHECKPOINT_SCHEMA = "lewm_go2_matched_branch_physical_outcome_screen_checkpoint_v1"
PCA_SCHEMA = "lewm_go2_matched_branch_physical_outcome_screen_train_pca_v1"
OUTCOME_STATS_SCHEMA = "lewm_go2_matched_branch_physical_outcome_stats_v1"
INPUT_STATS_SCHEMA = "lewm_go2_matched_branch_physical_input_stats_v1"

PASS_VISUAL_STATUS = "PASS_VISUAL_PHYSICAL_DYNAMICS_HEADROOM"
PASS_ODOMETRY_STATUS = "PASS_ODOMETRY_ONLY_PHYSICAL_DYNAMICS_HEADROOM"
STOP_STATUS = "STOP_RETAINED_INPUT_PHYSICAL_DYNAMICS_HEADROOM_NOT_ESTABLISHED"
INFRASTRUCTURE_FAILURE_STATUS = "FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION"
TERMINAL_STATUSES = frozenset(
    (PASS_VISUAL_STATUS, PASS_ODOMETRY_STATUS, STOP_STATUS, INFRASTRUCTURE_FAILURE_STATUS)
)

ACTION_COUNT = 9
STATE_COUNT = 128
SCENE_COUNT = 16
CONTEXT_COUNT = 3
ARTIFACTS_PER_STATE = CONTEXT_COUNT + ACTION_COUNT
ARTIFACT_COUNT = STATE_COUNT * ARTIFACTS_PER_STATE
TOKEN_COUNT = 256
TOKEN_DIMENSION = 384
GRID_SIDE = 16
POOL_BLOCK_SIDE = 4
POOLED_SIDE = GRID_SIDE // POOL_BLOCK_SIDE
POOLED_FRAME_DIMENSION = POOLED_SIDE * POOLED_SIDE * TOKEN_DIMENSION
PCA_INPUT_DIMENSION = CONTEXT_COUNT * POOLED_FRAME_DIMENSION
PCA_DIMENSION = 16
PHYSICAL_INPUT_WIDTH = 12
TARGET_WIDTH = 4
MODEL_SEEDS = (2_026_080_311, 2_026_080_312, 2_026_080_313)
BATCH_STATES = 16
UPDATES_PER_MEMBER = 1_024
LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 1.0e-4
GRADIENT_CLIP_NORM = 1.0
MIN_SCALE = 1.0e-8
RANK_TOLERANCE_M = 0.01
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2_026_080_314
EXPECTED_TASK_IDENTITY = dense.EXPECTED_TASK_IDENTITY
EXPECTED_TASK_EVAL_REGRET = dense.EXPECTED_TASK_EVAL_REGRET
EXPECTED_TRAIN_PLAN_IDENTITY = dense.EXPECTED_TRAIN_PLAN_IDENTITY
EXPECTED_EVAL_PLAN_IDENTITY = dense.EXPECTED_EVAL_PLAN_IDENTITY
EXPECTED_COMBINED_PLAN_IDENTITY = dense.EXPECTED_COMBINED_PLAN_IDENTITY
EXPECTED_ACTION_COMMANDS = dense.EXPECTED_ACTION_COMMANDS

ODOMETRY_ARM = "odometry_and_command_history"
VISUAL_ARM = "odometry_and_command_history_plus_current_visual_context"
LEARNED_ARMS = (ODOMETRY_ARM, VISUAL_ARM)
TARGET_NAMES = ("endpoint_dx_body_m", "endpoint_dy_body_m", "endpoint_dyaw_rad", "path_length_m")
PHYSICAL_INPUT_NAMES = (
    "pose_0_to_1_dx_body_m",
    "pose_0_to_1_dy_body_m",
    "pose_0_to_1_dyaw_rad",
    "pose_1_to_2_dx_body_m",
    "pose_1_to_2_dy_body_m",
    "pose_1_to_2_dyaw_rad",
    "history_0_mean_executed_vx",
    "history_0_mean_executed_wz",
    "history_1_mean_executed_vx",
    "history_1_mean_executed_wz",
    "candidate_requested_vx",
    "candidate_requested_wz",
)

SCIENTIFIC_GATE_NAMES = frozenset(
    {
        "2_privileged_physical_oracle",
        "3_odometry_beats_task_action_only",
        "4_visual_beats_task_action_only",
        "5_visual_beats_odometry",
        "6a_odometry_beats_random_expected",
        "6b_visual_beats_random_expected",
    }
)


class PhysicalOutcomeScreenError(RuntimeError):
    """Raised when frozen data, mechanism, or deterministic contracts change."""


@dataclass(frozen=True)
class _PhysicalLabelsV1:
    target_progress_m: float
    path_length_m: float
    fell: bool
    tipped: bool
    planar_clearance_proxy_min_m: float | None
    grid_recoverability_proxy: float | bool | None


@dataclass(frozen=True)
class _PhysicalBranchV1:
    action_id: int
    target_rgb_artifact_id: str
    oracle_dense_rank: int
    labels: _PhysicalLabelsV1


@dataclass(frozen=True)
class _PhysicalGroupV1:
    role: str
    state_id: str
    family: str
    scene_id: str
    group_index: int
    state_index_in_scene: int
    relative_target_xy_body_m: tuple[float, float]
    context_rgb_artifact_ids: tuple[str, str, str]
    branches: tuple[_PhysicalBranchV1, ...]

    @property
    def group_id(self) -> str:
        return self.state_id


@dataclass(frozen=True)
class RolePhysicalDataV1:
    role: str
    plan: prior.RoleFeaturePlanV1
    physical_inputs: torch.Tensor
    targets: torch.Tensor
    pooled_context: torch.Tensor
    identity_sha256: str


@dataclass(frozen=True)
class PhysicalDatasetV1:
    train: RolePhysicalDataV1
    eval: RolePhysicalDataV1
    plans: prior.CalibrationFeaturePlansV1
    identity_sha256: str


@dataclass(frozen=True)
class TrainingDatasetV1:
    """Train-only payload used before the evaluation cache may be opened."""

    train: RolePhysicalDataV1
    identity_sha256: str


def canonical_bytes_v1(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _binding_v1(value: object) -> dict[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256", "byte_count"}
        or not isinstance(value.get("path"), str)
        or not value["path"]
        or not isinstance(value.get("sha256"), str)
        or len(str(value["sha256"])) != 64
        or any(character not in "0123456789abcdef" for character in str(value["sha256"]))
        or type(value.get("byte_count")) is not int
        or int(value["byte_count"]) <= 0
    ):
        raise PhysicalOutcomeScreenError("implementation source binding changed")
    return dict(value)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _tensor_identity_v1(value: torch.Tensor) -> str:
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise PhysicalOutcomeScreenError("identity input must be a CPU tensor")
    array = np.ascontiguousarray(value.detach().numpy())
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(canonical_bytes_v1(list(array.shape)) + b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _finite_tensor(
    value: object, *, shape: tuple[int, ...], dtype: torch.dtype, name: str
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != dtype
        or tuple(value.shape) != shape
        or not bool(torch.isfinite(value).all())
    ):
        raise PhysicalOutcomeScreenError(f"{name} tensor changed")
    return value


def _wrap_angle(value: float) -> float:
    wrapped = (float(value) + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if wrapped == -math.pi and value > 0.0 else wrapped


def _pose(value: object, *, name: str) -> tuple[float, float, float]:
    if not isinstance(value, Mapping) or set(value) != {
        "position_xyz_m",
        "quaternion_wxyz",
    }:
        raise PhysicalOutcomeScreenError(f"{name} pose changed")
    position = np.asarray(value["position_xyz_m"], dtype=np.float64)
    quaternion = np.asarray(value["quaternion_wxyz"], dtype=np.float64)
    if (
        position.shape != (3,)
        or quaternion.shape != (4,)
        or not np.isfinite(position).all()
        or not np.isfinite(quaternion).all()
    ):
        raise PhysicalOutcomeScreenError(f"{name} pose is nonfinite or malformed")
    norm = float(np.linalg.norm(quaternion))
    if norm == 0.0 or abs(norm - 1.0) > 1.0e-5:
        raise PhysicalOutcomeScreenError(f"{name} quaternion is not near-unit")
    qw, qx, qy, qz = (float(item / norm) for item in quaternion)
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    return float(position[0]), float(position[1]), yaw


def body_local_increment_v1(source_pose: object, target_pose: object) -> tuple[float, float, float]:
    """Return target minus source as source-body ``dx,dy,wrapped dyaw``."""

    sx, sy, syaw = _pose(source_pose, name="source")
    tx, ty, tyaw = _pose(target_pose, name="target")
    world_x, world_y = tx - sx, ty - sy
    cosine, sine = math.cos(syaw), math.sin(syaw)
    return (
        cosine * world_x + sine * world_y,
        -sine * world_x + cosine * world_y,
        _wrap_angle(tyaw - syaw),
    )


def _command_block(value: object, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (5, 3) or not np.isfinite(result).all():
        raise PhysicalOutcomeScreenError(f"{name} command block changed")
    return result


def _quantize(value: float, tolerance: float = RANK_TOLERANCE_M) -> int:
    scaled = abs(float(value)) / tolerance
    return (1 if value >= 0.0 else -1) * math.floor(scaled + 0.5)


def _dense_observed_ranks(branches: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    keys = []
    for action_id, branch in enumerate(branches):
        if branch.get("action_id") != action_id:
            raise PhysicalOutcomeScreenError("receipt candidate action grid changed")
        progress = float(branch.get("physical_target_progress_m", math.nan))
        path = float(branch.get("physical_path_length_m", math.nan))
        fell, tipped = branch.get("physical_fell"), branch.get("physical_tipped")
        if (
            not math.isfinite(progress)
            or not math.isfinite(path)
            or path < 0.0
            or type(fell) is not bool
            or type(tipped) is not bool
        ):
            raise PhysicalOutcomeScreenError("receipt physical outcome changed")
        keys.append((int(fell), int(tipped), -_quantize(progress), _quantize(path)))
    mapping = {key: rank for rank, key in enumerate(sorted(set(keys)))}
    return tuple(mapping[key] for key in keys)


def _raw_receipt(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PhysicalOutcomeScreenError("state receipt is not a mapping")
    if "document" in value and isinstance(value["document"], Mapping):
        value = value["document"]
    if not isinstance(value.get("state"), Mapping):
        raise PhysicalOutcomeScreenError("state receipt document changed")
    return value


def _groups_from_receipts(
    receipts: Sequence[Mapping[str, Any]], *, role: str
) -> tuple[tuple[_PhysicalGroupV1, ...], Mapping[str, Mapping[str, Any]]]:
    if role not in {"train", "eval"} or len(receipts) != STATE_COUNT:
        raise PhysicalOutcomeScreenError(f"{role} receipt count changed")
    normalized = tuple(_raw_receipt(item) for item in receipts)
    try:
        ordered = tuple(
            sorted(
                normalized,
                key=lambda receipt: (
                    int(receipt["state"]["group_index"]),
                    str(receipt["state"]["state_id"]),
                ),
            )
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PhysicalOutcomeScreenError("receipt state ordering changed") from exc
    groups: list[_PhysicalGroupV1] = []
    by_id: dict[str, Mapping[str, Any]] = {}
    for receipt in ordered:
        state = receipt["state"]
        context = receipt.get("context")
        branches = receipt.get("branches")
        if (
            state.get("role") != role
            or not isinstance(context, Mapping)
            or not isinstance(branches, list)
            or len(branches) != ACTION_COUNT
        ):
            raise PhysicalOutcomeScreenError(f"{role} state receipt geometry changed")
        state_id = str(state.get("state_id", ""))
        if not state_id or state_id in by_id:
            raise PhysicalOutcomeScreenError(f"{role} state identity changed")
        context_ids = context.get("rgb_artifact_ids")
        target = np.asarray(context.get("target_relative_body_xy_m"), dtype=np.float64)
        if (
            not isinstance(context_ids, list)
            or len(context_ids) != CONTEXT_COUNT
            or any(not isinstance(item, str) or not item for item in context_ids)
            or target.shape != (2,)
            or not np.isfinite(target).all()
        ):
            raise PhysicalOutcomeScreenError(f"{role} context identity changed")
        ranks = _dense_observed_ranks(branches)
        parsed_branches = []
        for action_id, branch in enumerate(branches):
            frame = branch.get("frame_receipt")
            artifact_id = frame.get("artifact_id") if isinstance(frame, Mapping) else None
            if not isinstance(artifact_id, str) or not artifact_id:
                raise PhysicalOutcomeScreenError("target artifact identity changed")
            parsed_branches.append(
                _PhysicalBranchV1(
                    action_id=action_id,
                    target_rgb_artifact_id=artifact_id,
                    oracle_dense_rank=ranks[action_id],
                    labels=_PhysicalLabelsV1(
                        target_progress_m=float(branch["physical_target_progress_m"]),
                        path_length_m=float(branch["physical_path_length_m"]),
                        fell=bool(branch["physical_fell"]),
                        tipped=bool(branch["physical_tipped"]),
                        # Bounded-branch state receipts retain direct physical
                        # outcomes but not these legacy nonphysical proxies.
                        planar_clearance_proxy_min_m=None,
                        grid_recoverability_proxy=None,
                    ),
                )
            )
        group = _PhysicalGroupV1(
            role=role,
            state_id=state_id,
            family=str(state.get("family", "")),
            scene_id=str(state.get("scene_id", "")),
            group_index=int(state.get("group_index")),
            state_index_in_scene=int(state.get("state_index_in_scene")),
            relative_target_xy_body_m=(float(target[0]), float(target[1])),
            context_rgb_artifact_ids=tuple(context_ids),  # type: ignore[arg-type]
            branches=tuple(parsed_branches),
        )
        groups.append(group)
        by_id[state_id] = receipt
    return tuple(groups), MappingProxyType(by_id)


def _cache_v1(value: object, plan: prior.RoleFeaturePlanV1) -> torch.Tensor:
    if not isinstance(value, Mapping) or set(value) != {"artifact_ids", "features"}:
        raise PhysicalOutcomeScreenError(f"{plan.role} cache object changed")
    artifact_ids = tuple(value["artifact_ids"])
    features = value["features"]
    if artifact_ids != plan.artifact_ids:
        raise PhysicalOutcomeScreenError(f"{plan.role} cache artifact order changed")
    if (
        not isinstance(features, torch.Tensor)
        or features.device.type != "cpu"
        or features.dtype != torch.float16
        or tuple(features.shape) != (ARTIFACT_COUNT, TOKEN_COUNT, TOKEN_DIMENSION)
    ):
        raise PhysicalOutcomeScreenError(f"{plan.role} cache tensor changed")
    return features


def pool_context_grids_v1(
    plan: prior.RoleFeaturePlanV1, cache_features: torch.Tensor
) -> torch.Tensor:
    """Return train/eval context descriptors without indexing target grids."""

    rows: list[np.ndarray] = []
    for state in plan.states:
        frames = []
        for index in state.context_artifact_indices:
            source = cache_features[index].detach().numpy().astype(np.float64, copy=False)
            if source.shape != (TOKEN_COUNT, TOKEN_DIMENSION) or not np.isfinite(source).all():
                raise PhysicalOutcomeScreenError("context DINO grid changed")
            norms = np.linalg.norm(source, axis=1)
            if np.max(np.abs(norms - 1.0)) > dense.TOKEN_NORM_TOLERANCE:
                raise PhysicalOutcomeScreenError("context DINO token normalization changed")
            grid = source.reshape(GRID_SIDE, GRID_SIDE, TOKEN_DIMENSION)
            pooled = grid.reshape(
                POOLED_SIDE,
                POOL_BLOCK_SIDE,
                POOLED_SIDE,
                POOL_BLOCK_SIDE,
                TOKEN_DIMENSION,
            ).mean(axis=(1, 3))
            frames.append(pooled.reshape(-1))
        rows.append(np.concatenate(frames))
    result = torch.from_numpy(np.stack(rows).astype(np.float64, copy=False))
    return _finite_tensor(
        result,
        shape=(STATE_COUNT, PCA_INPUT_DIMENSION),
        dtype=torch.float64,
        name="pooled context",
    )


def _role_arrays(
    plan: prior.RoleFeaturePlanV1,
    receipt_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = np.empty((STATE_COUNT, ACTION_COUNT, PHYSICAL_INPUT_WIDTH), dtype=np.float64)
    targets = np.empty((STATE_COUNT, ACTION_COUNT, TARGET_WIDTH), dtype=np.float64)
    for state_index, state in enumerate(plan.states):
        receipt = receipt_by_id.get(state.state_id)
        if receipt is None:
            raise PhysicalOutcomeScreenError("feature plan state lacks a receipt")
        context = receipt["context"]
        poses = context.get("context_base_pose_world_sequence")
        blocks = context.get("history_executed_blocks")
        branches = receipt["branches"]
        if (
            not isinstance(poses, list)
            or len(poses) != CONTEXT_COUNT
            or not isinstance(blocks, list)
            or len(blocks) != 2
        ):
            raise PhysicalOutcomeScreenError("pre-action receipt history changed")
        increment_01 = body_local_increment_v1(poses[0], poses[1])
        increment_12 = body_local_increment_v1(poses[1], poses[2])
        history = tuple(_command_block(block, name="historical executed") for block in blocks)
        fixed = (
            *increment_01,
            *increment_12,
            float(history[0][:, 0].mean()),
            float(history[0][:, 2].mean()),
            float(history[1][:, 0].mean()),
            float(history[1][:, 2].mean()),
        )
        for action_id, branch in enumerate(branches):
            requested = _command_block(branch.get("requested_block"), name="candidate requested")
            expected = np.tile(np.asarray(EXPECTED_ACTION_COMMANDS[action_id]), (5, 1))
            if not np.array_equal(requested, expected):
                raise PhysicalOutcomeScreenError("candidate requested action catalog changed")
            inputs[state_index, action_id] = (
                *fixed,
                float(requested[:, 0].mean()),
                float(requested[:, 2].mean()),
            )
            endpoint = branch.get("endpoint_state")
            if not isinstance(endpoint, Mapping):
                raise PhysicalOutcomeScreenError("branch endpoint changed")
            endpoint_pose = {
                "position_xyz_m": endpoint.get("base_pos_world"),
                "quaternion_wxyz": endpoint.get("base_quat_wxyz"),
            }
            dx, dy, dyaw = body_local_increment_v1(poses[-1], endpoint_pose)
            path_length = float(branch.get("physical_path_length_m", math.nan))
            if not math.isfinite(path_length) or path_length < 0.0:
                raise PhysicalOutcomeScreenError("branch path length changed")
            targets[state_index, action_id] = (dx, dy, dyaw, path_length)
    if not np.isfinite(inputs).all() or not np.isfinite(targets).all():
        raise PhysicalOutcomeScreenError("derived physical arrays are nonfinite")
    return (
        torch.from_numpy(inputs.astype(np.float32)),
        torch.from_numpy(targets.astype(np.float32)),
    )


def _role_data_identity_v1(
    role: str,
    plan: prior.RoleFeaturePlanV1,
    physical_inputs: torch.Tensor,
    targets: torch.Tensor,
    pooled_context: torch.Tensor,
) -> str:
    payload = {
        "role": role,
        "plan_identity_sha256": plan.identity_sha256,
        "physical_inputs_sha256": _tensor_identity_v1(physical_inputs),
        "targets_sha256": _tensor_identity_v1(targets),
        "pooled_context_sha256": _tensor_identity_v1(pooled_context),
    }
    return hashlib.sha256(canonical_bytes_v1(payload)).hexdigest()


def build_role_physical_data_v1(
    receipts: Sequence[Mapping[str, Any]],
    cache: Mapping[str, object],
    *,
    role: str,
) -> RolePhysicalDataV1:
    groups, receipt_by_id = _groups_from_receipts(receipts, role=role)
    plan = prior.build_role_feature_plan_v1(groups, role=role)
    expected = EXPECTED_TRAIN_PLAN_IDENTITY if role == "train" else EXPECTED_EVAL_PLAN_IDENTITY
    if plan.identity_sha256 != expected:
        raise PhysicalOutcomeScreenError(f"{role} feature-plan identity changed")
    features = _cache_v1(cache, plan)
    physical_inputs, targets = _role_arrays(plan, receipt_by_id)
    pooled_context = pool_context_grids_v1(plan, features)
    identity = _role_data_identity_v1(
        role, plan, physical_inputs, targets, pooled_context
    )
    return RolePhysicalDataV1(
        role=role,
        plan=plan,
        physical_inputs=physical_inputs,
        targets=targets,
        pooled_context=pooled_context,
        identity_sha256=identity,
    )


def build_physical_dataset_v1(
    train_receipts: Sequence[Mapping[str, Any]],
    eval_receipts: Sequence[Mapping[str, Any]] | None,
    train_cache: Mapping[str, object],
    eval_cache: Mapping[str, object] | None,
) -> PhysicalDatasetV1 | TrainingDatasetV1:
    train = build_role_physical_data_v1(train_receipts, train_cache, role="train")
    if eval_receipts is None and eval_cache is None:
        identity = hashlib.sha256(
            canonical_bytes_v1(
                {
                    "role": "train_only_before_durable_checkpoint",
                    "train_plan_identity_sha256": train.plan.identity_sha256,
                    "train_data_identity_sha256": train.identity_sha256,
                }
            )
        ).hexdigest()
        return TrainingDatasetV1(train=train, identity_sha256=identity)
    if eval_receipts is None or eval_cache is None:
        raise PhysicalOutcomeScreenError(
            "evaluation receipts and cache must be absent or present together"
        )
    evaluation = build_role_physical_data_v1(eval_receipts, eval_cache, role="eval")
    plans = prior.CalibrationFeaturePlansV1(
        train=train.plan,
        eval=evaluation.plan,
        identity_sha256=hashlib.sha256(
            bytes.fromhex(train.plan.identity_sha256)
            + bytes.fromhex(evaluation.plan.identity_sha256)
        ).hexdigest(),
    )
    if plans.identity_sha256 != EXPECTED_COMBINED_PLAN_IDENTITY:
        raise PhysicalOutcomeScreenError("combined feature-plan identity changed")
    if (
        {state.state_id for state in train.plan.states}
        & {state.state_id for state in evaluation.plan.states}
        or {state.scene_id for state in train.plan.states}
        & {state.scene_id for state in evaluation.plan.states}
        or set(train.plan.artifact_ids) & set(evaluation.plan.artifact_ids)
    ):
        raise PhysicalOutcomeScreenError("train/eval role identities overlap")
    identity = hashlib.sha256(
        canonical_bytes_v1(
            {
                "plan_identity_sha256": plans.identity_sha256,
                "train_identity_sha256": train.identity_sha256,
                "eval_identity_sha256": evaluation.identity_sha256,
            }
        )
    ).hexdigest()
    return PhysicalDatasetV1(train=train, eval=evaluation, plans=plans, identity_sha256=identity)


def config_v1() -> dict[str, object]:
    batches_per_permutation = STATE_COUNT // BATCH_STATES
    return {
        "schema": CONFIG_SCHEMA,
        "states_per_role": STATE_COUNT,
        "scenes_per_role": SCENE_COUNT,
        "actions": ACTION_COUNT,
        "plan_identities": {
            "train": EXPECTED_TRAIN_PLAN_IDENTITY,
            "eval": EXPECTED_EVAL_PLAN_IDENTITY,
            "combined": EXPECTED_COMBINED_PLAN_IDENTITY,
        },
        "physical_input_names": list(PHYSICAL_INPUT_NAMES),
        "physical_input_width": PHYSICAL_INPUT_WIDTH,
        "visual_slots": PCA_DIMENSION,
        "model_input_width": INPUT_WIDTH,
        "target_names": list(TARGET_NAMES),
        "target_width": TARGET_WIDTH,
        "visual_projection": {
            "context_frames": CONTEXT_COUNT,
            "source_grid": [GRID_SIDE, GRID_SIDE, TOKEN_DIMENSION],
            "pool": "nonoverlapping_4x4_mean_float64",
            "pooled_grid": [POOLED_SIDE, POOLED_SIDE, TOKEN_DIMENSION],
            "flatten_order": "time_row_column_channel",
            "input_dimension": PCA_INPUT_DIMENSION,
            "pca_dimension": PCA_DIMENSION,
            "fit_role": "train_only",
            "solver": "numpy.linalg.svd_thin_float64",
            "sign": "largest_absolute_loading_smallest_index_positive",
        },
        "model": {
            "architecture": [INPUT_WIDTH, HIDDEN_WIDTH, OUTPUT_WIDTH],
            "activation": "tanh",
            "parameters_per_member": PARAMETER_COUNT,
            "seeds": list(MODEL_SEEDS),
            "initialization": "dedicated_cpu_generator_xavier_uniform_zero_bias",
        },
        "training": {
            "updates_per_member": UPDATES_PER_MEMBER,
            "batch_states": BATCH_STATES,
            "complete_actions_per_state": ACTION_COUNT,
            "batches_per_seed_local_permutation": batches_per_permutation,
            "seed_local_permutations": UPDATES_PER_MEMBER // batches_per_permutation,
            "schedule": "independent_cpu_generator_manual_seed_member_seed_randperm_then_contiguous_batches",
            "same_schedule_and_initial_state_for_matched_b_c": True,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "betas": [0.9, 0.999],
            "epsilon": 1.0e-8,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "loss": "unweighted_mean_squared_standardized_residual_error",
            "dtype": "float32",
            "device": "cpu",
            "deterministic_algorithms": True,
            "torch_threads": 1,
        },
        "task_action_only": {
            "coefficients": 27,
            "identity_sha256": EXPECTED_TASK_IDENTITY,
            "required_eval_regret": EXPECTED_TASK_EVAL_REGRET,
        },
        "rank_tolerance_m": RANK_TOLERANCE_M,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def fit_train_pca_v1(
    train: RolePhysicalDataV1,
    *,
    implementation_source_binding: Mapping[str, object],
) -> dict[str, object]:
    if train.role != "train" or train.plan.identity_sha256 != EXPECTED_TRAIN_PLAN_IDENTITY:
        raise PhysicalOutcomeScreenError("PCA source role changed")
    binding = _binding_v1(implementation_source_binding)
    source = _finite_tensor(
        train.pooled_context,
        shape=(STATE_COUNT, PCA_INPUT_DIMENSION),
        dtype=torch.float64,
        name="train PCA source",
    ).numpy()
    mean = source.mean(axis=0, dtype=np.float64)
    centered = source - mean
    _u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if singular_values.shape[0] < PCA_DIMENSION or vh.shape[0] < PCA_DIMENSION:
        raise PhysicalOutcomeScreenError("train PCA rank is insufficient")
    components = np.ascontiguousarray(vh[:PCA_DIMENSION].T, dtype=np.float64)
    for component in range(PCA_DIMENSION):
        pivot = int(np.argmax(np.abs(components[:, component])))
        if components[pivot, component] < 0.0:
            components[:, component] *= -1.0
    result: dict[str, object] = {
        "schema": PCA_SCHEMA,
        "implementation_source_binding": binding,
        "train_plan_identity_sha256": train.plan.identity_sha256,
        "train_data_identity_sha256": train.identity_sha256,
        "training_rows": STATE_COUNT,
        "input_dimension": PCA_INPUT_DIMENSION,
        "dimension": PCA_DIMENSION,
        "mean": torch.from_numpy(np.ascontiguousarray(mean)),
        "components": torch.from_numpy(components),
        "singular_values": torch.from_numpy(
            np.ascontiguousarray(singular_values[:PCA_DIMENSION], dtype=np.float64)
        ),
    }
    result["identity_sha256"] = pca_identity_v1(result)
    return result


def pca_identity_v1(pca: Mapping[str, object]) -> str:
    required = {
        "schema",
        "implementation_source_binding",
        "train_plan_identity_sha256",
        "train_data_identity_sha256",
        "training_rows",
        "input_dimension",
        "dimension",
        "mean",
        "components",
        "singular_values",
    }
    if set(pca) not in (required, required | {"identity_sha256"}):
        raise PhysicalOutcomeScreenError("PCA inventory changed")
    if (
        pca.get("schema") != PCA_SCHEMA
        or pca.get("implementation_source_binding")
        != _binding_v1(pca.get("implementation_source_binding"))
        or pca.get("train_plan_identity_sha256") != EXPECTED_TRAIN_PLAN_IDENTITY
        or not _is_sha256(pca.get("train_data_identity_sha256"))
        or pca.get("training_rows") != STATE_COUNT
        or pca.get("input_dimension") != PCA_INPUT_DIMENSION
        or pca.get("dimension") != PCA_DIMENSION
    ):
        raise PhysicalOutcomeScreenError("PCA contract changed")
    mean = _finite_tensor(
        pca.get("mean"), shape=(PCA_INPUT_DIMENSION,), dtype=torch.float64, name="PCA mean"
    )
    components = _finite_tensor(
        pca.get("components"),
        shape=(PCA_INPUT_DIMENSION, PCA_DIMENSION),
        dtype=torch.float64,
        name="PCA components",
    )
    singular = _finite_tensor(
        pca.get("singular_values"),
        shape=(PCA_DIMENSION,),
        dtype=torch.float64,
        name="PCA singular values",
    )
    if (
        not bool((singular >= 0.0).all())
        or not bool((singular[:-1] >= singular[1:]).all())
        or not torch.allclose(
            components.T @ components,
            torch.eye(PCA_DIMENSION, dtype=torch.float64),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
    ):
        raise PhysicalOutcomeScreenError("PCA basis changed")
    for component in range(PCA_DIMENSION):
        pivot = int(torch.argmax(torch.abs(components[:, component])))
        if components[pivot, component] < 0.0:
            raise PhysicalOutcomeScreenError("PCA component sign changed")
    metadata = {
        key: pca[key]
        for key in sorted(required - {"mean", "components", "singular_values"})
    }
    metadata["mean_sha256"] = _tensor_identity_v1(mean)
    metadata["components_sha256"] = _tensor_identity_v1(components)
    metadata["singular_values_sha256"] = _tensor_identity_v1(singular)
    return hashlib.sha256(canonical_bytes_v1(metadata)).hexdigest()


def project_visual_context_v1(
    role: RolePhysicalDataV1, pca: Mapping[str, object]
) -> torch.Tensor:
    identity = pca_identity_v1(pca)
    if pca.get("identity_sha256") != identity:
        raise PhysicalOutcomeScreenError("PCA identity changed")
    source = _finite_tensor(
        role.pooled_context,
        shape=(STATE_COUNT, PCA_INPUT_DIMENSION),
        dtype=torch.float64,
        name=f"{role.role} pooled context",
    )
    projected = (source - pca["mean"]) @ pca["components"]
    result = projected.to(torch.float32)
    return _finite_tensor(
        result,
        shape=(STATE_COUNT, PCA_DIMENSION),
        dtype=torch.float32,
        name=f"{role.role} visual projection",
    )


def assemble_model_inputs_v1(
    role: RolePhysicalDataV1,
    visual: torch.Tensor,
    *,
    arm: str,
) -> torch.Tensor:
    physical = _finite_tensor(
        role.physical_inputs,
        shape=(STATE_COUNT, ACTION_COUNT, PHYSICAL_INPUT_WIDTH),
        dtype=torch.float32,
        name=f"{role.role} physical inputs",
    )
    visual = _finite_tensor(
        visual,
        shape=(STATE_COUNT, PCA_DIMENSION),
        dtype=torch.float32,
        name=f"{role.role} visual projection",
    )
    if arm == ODOMETRY_ARM:
        slots = torch.zeros(
            (STATE_COUNT, ACTION_COUNT, PCA_DIMENSION), dtype=torch.float32
        )
    elif arm == VISUAL_ARM:
        slots = visual[:, None, :].expand(-1, ACTION_COUNT, -1)
    else:
        raise PhysicalOutcomeScreenError("learned arm changed")
    result = torch.cat((physical, slots), dim=-1).contiguous()
    return _finite_tensor(
        result,
        shape=(STATE_COUNT, ACTION_COUNT, INPUT_WIDTH),
        dtype=torch.float32,
        name=f"{role.role} {arm} model inputs",
    )


def _stats_identity_v1(value: Mapping[str, object], tensor_names: Sequence[str]) -> str:
    metadata = {key: item for key, item in value.items() if key not in tensor_names and key != "identity_sha256"}
    for name in tensor_names:
        tensor = value.get(name)
        if not isinstance(tensor, torch.Tensor):
            raise PhysicalOutcomeScreenError("statistics tensor changed")
        metadata[f"{name}_sha256"] = _tensor_identity_v1(tensor)
    return hashlib.sha256(canonical_bytes_v1(metadata)).hexdigest()


def fit_outcome_statistics_v1(train: RolePhysicalDataV1) -> dict[str, object]:
    targets = _finite_tensor(
        train.targets,
        shape=(STATE_COUNT, ACTION_COUNT, TARGET_WIDTH),
        dtype=torch.float32,
        name="train targets",
    ).to(torch.float64)
    action_means = targets.mean(dim=0)
    residuals = targets - action_means[None, :, :]
    scales = torch.sqrt(torch.mean(residuals.square(), dim=(0, 1)))
    scales = torch.where(scales < MIN_SCALE, torch.ones_like(scales), scales)
    result: dict[str, object] = {
        "schema": OUTCOME_STATS_SCHEMA,
        "train_data_identity_sha256": train.identity_sha256,
        "action_means": action_means.to(torch.float32),
        "residual_scales": scales.to(torch.float32),
    }
    result["identity_sha256"] = _stats_identity_v1(
        result, ("action_means", "residual_scales")
    )
    return result


def fit_input_statistics_v1(
    inputs: torch.Tensor, *, arm: str, train_data_identity_sha256: str
) -> dict[str, object]:
    values = _finite_tensor(
        inputs,
        shape=(STATE_COUNT, ACTION_COUNT, INPUT_WIDTH),
        dtype=torch.float32,
        name=f"{arm} train inputs",
    ).to(torch.float64)
    flat = values.reshape(-1, INPUT_WIDTH)
    mean = flat.mean(dim=0)
    scale = torch.sqrt(torch.mean((flat - mean).square(), dim=0))
    scale = torch.where(scale < MIN_SCALE, torch.ones_like(scale), scale)
    result: dict[str, object] = {
        "schema": INPUT_STATS_SCHEMA,
        "arm": arm,
        "train_data_identity_sha256": train_data_identity_sha256,
        "mean": mean.to(torch.float32),
        "scale": scale.to(torch.float32),
    }
    result["identity_sha256"] = _stats_identity_v1(result, ("mean", "scale"))
    return result


def standardized_residual_targets_v1(
    role: RolePhysicalDataV1, outcome_stats: Mapping[str, object]
) -> torch.Tensor:
    means = _finite_tensor(
        outcome_stats.get("action_means"),
        shape=(ACTION_COUNT, TARGET_WIDTH),
        dtype=torch.float32,
        name="action means",
    )
    scales = _finite_tensor(
        outcome_stats.get("residual_scales"),
        shape=(TARGET_WIDTH,),
        dtype=torch.float32,
        name="residual scales",
    )
    if not bool((scales > 0.0).all()):
        raise PhysicalOutcomeScreenError("residual scales are not positive")
    result = (role.targets - means[None, :, :]) / scales
    return _finite_tensor(
        result,
        shape=(STATE_COUNT, ACTION_COUNT, TARGET_WIDTH),
        dtype=torch.float32,
        name=f"{role.role} standardized targets",
    )


def normalize_model_inputs_v1(
    inputs: torch.Tensor, input_stats: Mapping[str, object]
) -> torch.Tensor:
    mean = _finite_tensor(
        input_stats.get("mean"), shape=(INPUT_WIDTH,), dtype=torch.float32, name="input mean"
    )
    scale = _finite_tensor(
        input_stats.get("scale"), shape=(INPUT_WIDTH,), dtype=torch.float32, name="input scale"
    )
    if not bool((scale > 0.0).all()):
        raise PhysicalOutcomeScreenError("input scales are not positive")
    result = (inputs - mean) / scale
    return _finite_tensor(
        result,
        shape=(STATE_COUNT, ACTION_COUNT, INPUT_WIDTH),
        dtype=torch.float32,
        name="normalized model inputs",
    )


def training_orders_v1(seed: int) -> tuple[torch.Tensor, ...]:
    if type(seed) is not int or seed not in MODEL_SEEDS:
        raise PhysicalOutcomeScreenError("member seed changed")
    if STATE_COUNT % BATCH_STATES or UPDATES_PER_MEMBER % (STATE_COUNT // BATCH_STATES):
        raise PhysicalOutcomeScreenError("registered batch schedule is not integral")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation_count = UPDATES_PER_MEMBER // (STATE_COUNT // BATCH_STATES)
    return tuple(
        torch.randperm(STATE_COUNT, generator=generator)
        for _ in range(permutation_count)
    )


def _clone_state(value: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in value.items()}


def fit_member_v1(
    initial_state: Mapping[str, torch.Tensor],
    normalized_inputs: torch.Tensor,
    standardized_targets: torch.Tensor,
    orders: Sequence[torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    if not torch.are_deterministic_algorithms_enabled() or torch.get_num_threads() != 1:
        raise PhysicalOutcomeScreenError("deterministic CPU runtime changed")
    inputs = _finite_tensor(
        normalized_inputs,
        shape=(STATE_COUNT, ACTION_COUNT, INPUT_WIDTH),
        dtype=torch.float32,
        name="member inputs",
    )
    targets = _finite_tensor(
        standardized_targets,
        shape=(STATE_COUNT, ACTION_COUNT, TARGET_WIDTH),
        dtype=torch.float32,
        name="member targets",
    )
    expected_orders = UPDATES_PER_MEMBER // (STATE_COUNT // BATCH_STATES)
    if len(orders) != expected_orders:
        raise PhysicalOutcomeScreenError("member schedule length changed")
    model = PhysicalOutcomeMLPV1()
    model.load_state_dict(_clone_state(initial_state), strict=True)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        foreach=False,
        fused=False,
    )
    with torch.no_grad():
        initial_prediction = model(inputs.reshape(-1, INPUT_WIDTH)).reshape(
            STATE_COUNT, ACTION_COUNT, TARGET_WIDTH
        )
        initial_loss = float(torch.mean((initial_prediction - targets).square()).item())
    updates = 0
    for order in orders:
        if (
            not isinstance(order, torch.Tensor)
            or order.dtype != torch.int64
            or order.device.type != "cpu"
            or tuple(order.shape) != (STATE_COUNT,)
            or not torch.equal(torch.sort(order).values, torch.arange(STATE_COUNT))
        ):
            raise PhysicalOutcomeScreenError("member state permutation changed")
        for start in range(0, STATE_COUNT, BATCH_STATES):
            indices = order[start : start + BATCH_STATES]
            optimizer.zero_grad(set_to_none=True)
            batch = inputs[indices]
            prediction = model(batch.reshape(-1, INPUT_WIDTH)).reshape(
                len(indices), ACTION_COUNT, TARGET_WIDTH
            )
            loss = torch.mean((prediction - targets[indices]).square())
            if not bool(torch.isfinite(loss)):
                raise PhysicalOutcomeScreenError("member loss became nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            updates += 1
    if updates != UPDATES_PER_MEMBER:
        raise PhysicalOutcomeScreenError("member optimizer step count changed")
    model.eval()
    with torch.no_grad():
        final_prediction = model(inputs.reshape(-1, INPUT_WIDTH)).reshape(
            STATE_COUNT, ACTION_COUNT, TARGET_WIDTH
        )
        final_loss = float(torch.mean((final_prediction - targets).square()).item())
    state = _clone_state(model.state_dict())
    identity = physical_outcome_state_identity_v1(state)
    return state, {
        "optimizer_steps": updates,
        "initial_full_train_loss": initial_loss,
        "final_full_train_loss": final_loss,
        "state_identity_sha256": identity,
    }


def predict_member_v1(
    state: Mapping[str, torch.Tensor],
    inputs: torch.Tensor,
    input_stats: Mapping[str, object],
    outcome_stats: Mapping[str, object],
) -> torch.Tensor:
    model = PhysicalOutcomeMLPV1()
    model.load_state_dict(_clone_state(state), strict=True)
    model.eval()
    normalized = normalize_model_inputs_v1(inputs, input_stats)
    with torch.no_grad():
        standardized = model(normalized.reshape(-1, INPUT_WIDTH)).reshape(
            STATE_COUNT, ACTION_COUNT, TARGET_WIDTH
        )
    means = _finite_tensor(
        outcome_stats.get("action_means"),
        shape=(ACTION_COUNT, TARGET_WIDTH),
        dtype=torch.float32,
        name="action means",
    )
    scales = _finite_tensor(
        outcome_stats.get("residual_scales"),
        shape=(TARGET_WIDTH,),
        dtype=torch.float32,
        name="residual scales",
    )
    result = standardized * scales + means[None, :, :]
    return _finite_tensor(
        result,
        shape=(STATE_COUNT, ACTION_COUNT, TARGET_WIDTH),
        dtype=torch.float32,
        name="decoded member outcomes",
    )


def _outcome_stats_identity_v1(value: Mapping[str, object]) -> str:
    if (
        set(value)
        != {
            "schema",
            "train_data_identity_sha256",
            "action_means",
            "residual_scales",
            "identity_sha256",
        }
        or value.get("schema") != OUTCOME_STATS_SCHEMA
        or not _is_sha256(value.get("train_data_identity_sha256"))
    ):
        raise PhysicalOutcomeScreenError("outcome statistics contract changed")
    _finite_tensor(
        value.get("action_means"),
        shape=(ACTION_COUNT, TARGET_WIDTH),
        dtype=torch.float32,
        name="action means",
    )
    scales = _finite_tensor(
        value.get("residual_scales"),
        shape=(TARGET_WIDTH,),
        dtype=torch.float32,
        name="residual scales",
    )
    if not bool((scales > 0.0).all()):
        raise PhysicalOutcomeScreenError("outcome scales changed")
    identity = _stats_identity_v1(value, ("action_means", "residual_scales"))
    if value.get("identity_sha256") != identity:
        raise PhysicalOutcomeScreenError("outcome statistics identity changed")
    return identity


def _input_stats_identity_v1(value: Mapping[str, object], *, arm: str) -> str:
    if (
        set(value)
        != {
            "schema",
            "arm",
            "train_data_identity_sha256",
            "mean",
            "scale",
            "identity_sha256",
        }
        or value.get("schema") != INPUT_STATS_SCHEMA
        or not _is_sha256(value.get("train_data_identity_sha256"))
    ):
        raise PhysicalOutcomeScreenError("input statistics contract changed")
    mean = _finite_tensor(
        value.get("mean"),
        shape=(INPUT_WIDTH,),
        dtype=torch.float32,
        name="input mean",
    )
    scale = _finite_tensor(
        value.get("scale"),
        shape=(INPUT_WIDTH,),
        dtype=torch.float32,
        name="input scale",
    )
    if not bool((scale > 0.0).all()):
        raise PhysicalOutcomeScreenError("input scales changed")
    if arm == ODOMETRY_ARM and (
        torch.count_nonzero(mean[PHYSICAL_INPUT_WIDTH:]) != 0
        or not torch.equal(
            scale[PHYSICAL_INPUT_WIDTH:], torch.ones(PCA_DIMENSION)
        )
    ):
        raise PhysicalOutcomeScreenError("odometry zero-slot normalization changed")
    identity = _stats_identity_v1(value, ("mean", "scale"))
    if value.get("identity_sha256") != identity or value.get("arm") != arm:
        raise PhysicalOutcomeScreenError("input statistics identity changed")
    return identity


def _checkpoint_content_identity_v1(checkpoint: Mapping[str, object]) -> str:
    arms = checkpoint.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(LEARNED_ARMS):
        raise PhysicalOutcomeScreenError("checkpoint arm inventory changed")
    arm_payload: dict[str, object] = {}
    for arm in LEARNED_ARMS:
        value = arms[arm]
        if not isinstance(value, Mapping) or set(value) != {"input_stats", "members"}:
            raise PhysicalOutcomeScreenError("checkpoint learned arm changed")
        input_identity = _input_stats_identity_v1(value["input_stats"], arm=arm)
        members = value["members"]
        if not isinstance(members, list) or len(members) != len(MODEL_SEEDS):
            raise PhysicalOutcomeScreenError("checkpoint member inventory changed")
        rows = []
        for expected_seed, member in zip(MODEL_SEEDS, members, strict=True):
            if (
                not isinstance(member, Mapping)
                or set(member)
                != {
                    "seed",
                    "initial_state_identity_sha256",
                    "state_dict",
                    "state_identity_sha256",
                    "training",
                }
                or member.get("seed") != expected_seed
                or not isinstance(member.get("state_dict"), Mapping)
            ):
                raise PhysicalOutcomeScreenError("checkpoint member changed")
            initial = initialize_physical_outcome_mlp_v1(expected_seed).state_dict()
            if member.get("initial_state_identity_sha256") != physical_outcome_state_identity_v1(initial):
                raise PhysicalOutcomeScreenError("checkpoint initial state changed")
            state_identity = physical_outcome_state_identity_v1(member["state_dict"])
            training = member.get("training")
            if (
                member.get("state_identity_sha256") != state_identity
                or not isinstance(training, Mapping)
                or set(training)
                != {
                    "optimizer_steps",
                    "initial_full_train_loss",
                    "final_full_train_loss",
                    "state_identity_sha256",
                }
                or training.get("optimizer_steps") != UPDATES_PER_MEMBER
                or training.get("state_identity_sha256") != state_identity
                or not all(
                    math.isfinite(float(training[name]))
                    for name in ("initial_full_train_loss", "final_full_train_loss")
                )
            ):
                raise PhysicalOutcomeScreenError("checkpoint member training changed")
            rows.append(
                {
                    "seed": expected_seed,
                    "initial_state_identity_sha256": member["initial_state_identity_sha256"],
                    "state_identity_sha256": state_identity,
                    "training": dict(training),
                }
            )
        arm_payload[arm] = {"input_stats_identity_sha256": input_identity, "members": rows}
    metadata = {
        "schema": checkpoint.get("schema"),
        "config": checkpoint.get("config"),
        "implementation_source_binding": checkpoint.get("implementation_source_binding"),
        "train_plan_identity_sha256": checkpoint.get("train_plan_identity_sha256"),
        "train_data_identity_sha256": checkpoint.get("train_data_identity_sha256"),
        "pca_identity_sha256": pca_identity_v1(checkpoint["pca"]),
        "outcome_stats_identity_sha256": _outcome_stats_identity_v1(checkpoint["outcome_stats"]),
        "task_action_only_identity_sha256": checkpoint["task_action_only"].get("identity_sha256"),
        "arms": arm_payload,
    }
    return hashlib.sha256(canonical_bytes_v1(metadata)).hexdigest()


def fit_primary_checkpoint_v1(
    dataset: PhysicalDatasetV1 | TrainingDatasetV1,
    *,
    implementation_source_binding: Mapping[str, object],
) -> dict[str, object]:
    binding = _binding_v1(implementation_source_binding)
    if dataset.train.plan.identity_sha256 != EXPECTED_TRAIN_PLAN_IDENTITY:
        raise PhysicalOutcomeScreenError("training dataset plan changed")
    pca = fit_train_pca_v1(
        dataset.train, implementation_source_binding=binding
    )
    train_visual = project_visual_context_v1(dataset.train, pca)
    outcome_stats = fit_outcome_statistics_v1(dataset.train)
    standardized_targets = standardized_residual_targets_v1(dataset.train, outcome_stats)
    arms: dict[str, object] = {}
    initial_states = {
        seed: _clone_state(initialize_physical_outcome_mlp_v1(seed).state_dict())
        for seed in MODEL_SEEDS
    }
    for arm in LEARNED_ARMS:
        inputs = assemble_model_inputs_v1(dataset.train, train_visual, arm=arm)
        input_stats = fit_input_statistics_v1(
            inputs, arm=arm, train_data_identity_sha256=dataset.train.identity_sha256
        )
        normalized = normalize_model_inputs_v1(inputs, input_stats)
        members = []
        for seed in MODEL_SEEDS:
            initial = initial_states[seed]
            state, report = fit_member_v1(
                initial,
                normalized,
                standardized_targets,
                training_orders_v1(seed),
            )
            members.append(
                {
                    "seed": seed,
                    "initial_state_identity_sha256": physical_outcome_state_identity_v1(initial),
                    "state_dict": state,
                    "state_identity_sha256": physical_outcome_state_identity_v1(state),
                    "training": report,
                }
            )
        arms[arm] = {"input_stats": input_stats, "members": members}
    task = dense.fit_task_action_only_v1(dataset.train.plan)
    checkpoint: dict[str, object] = {
        "schema": CHECKPOINT_SCHEMA,
        "config": config_v1(),
        "implementation_source_binding": binding,
        "train_plan_identity_sha256": dataset.train.plan.identity_sha256,
        "train_data_identity_sha256": dataset.train.identity_sha256,
        "pca": pca,
        "outcome_stats": outcome_stats,
        "task_action_only": dense._task_payload(task),  # noqa: SLF001
        "arms": arms,
    }
    checkpoint["identity_sha256"] = _checkpoint_content_identity_v1(checkpoint)
    validate_checkpoint_v1(
        checkpoint, dataset=dataset, implementation_source_binding=binding
    )
    return checkpoint


def validate_checkpoint_v1(
    checkpoint: Mapping[str, object],
    *,
    implementation_source_binding: Mapping[str, object],
    dataset: PhysicalDatasetV1 | TrainingDatasetV1 | None = None,
) -> None:
    expected_keys = {
        "schema",
        "config",
        "implementation_source_binding",
        "train_plan_identity_sha256",
        "train_data_identity_sha256",
        "pca",
        "outcome_stats",
        "task_action_only",
        "arms",
        "identity_sha256",
    }
    if (
        set(checkpoint) != expected_keys
        or checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("config") != config_v1()
        or checkpoint.get("implementation_source_binding")
        != _binding_v1(implementation_source_binding)
        or checkpoint.get("train_plan_identity_sha256") != EXPECTED_TRAIN_PLAN_IDENTITY
        or not _is_sha256(checkpoint.get("train_data_identity_sha256"))
    ):
        raise PhysicalOutcomeScreenError("checkpoint contract changed")
    if dataset is not None and (
        checkpoint.get("train_plan_identity_sha256") != dataset.train.plan.identity_sha256
        or checkpoint.get("train_data_identity_sha256") != dataset.train.identity_sha256
    ):
        raise PhysicalOutcomeScreenError("checkpoint training-data binding changed")
    pca = checkpoint.get("pca")
    if not isinstance(pca, Mapping) or pca.get("identity_sha256") != pca_identity_v1(pca):
        raise PhysicalOutcomeScreenError("checkpoint PCA changed")
    if (
        pca.get("implementation_source_binding")
        != checkpoint.get("implementation_source_binding")
        or pca.get("train_data_identity_sha256")
        != checkpoint.get("train_data_identity_sha256")
    ):
        raise PhysicalOutcomeScreenError("checkpoint PCA binding changed")
    stats = checkpoint.get("outcome_stats")
    if not isinstance(stats, Mapping):
        raise PhysicalOutcomeScreenError("checkpoint outcome statistics changed")
    _outcome_stats_identity_v1(stats)
    if stats.get("train_data_identity_sha256") != checkpoint.get(
        "train_data_identity_sha256"
    ):
        raise PhysicalOutcomeScreenError("checkpoint outcome-statistics binding changed")
    task = checkpoint.get("task_action_only")
    if not isinstance(task, Mapping):
        raise PhysicalOutcomeScreenError("checkpoint task control changed")
    if dataset is None:
        dense._task_from_payload(task)  # noqa: SLF001
    else:
        dense._require_refitted_task_payload_v1(task, dataset.train.plan)  # noqa: SLF001
    identity = _checkpoint_content_identity_v1(checkpoint)
    if checkpoint.get("identity_sha256") != identity:
        raise PhysicalOutcomeScreenError("checkpoint content identity changed")


def physical_score_matrix_v1(
    plan: prior.RoleFeaturePlanV1, predicted_outcomes: torch.Tensor
) -> np.ndarray:
    predictions = _finite_tensor(
        predicted_outcomes,
        shape=(STATE_COUNT, ACTION_COUNT, TARGET_WIDTH),
        dtype=torch.float32,
        name="predicted physical outcomes",
    ).numpy()
    result = np.empty((STATE_COUNT, ACTION_COUNT), dtype=np.float64)
    for state_index, state in enumerate(plan.states):
        goal = np.asarray(state.relative_target_xy_body_m, dtype=np.float64)
        base_distance = float(np.linalg.norm(goal))
        keys = []
        for action_id in range(ACTION_COUNT):
            dx, dy, _dyaw, path = (float(item) for item in predictions[state_index, action_id])
            progress = base_distance - float(np.linalg.norm(goal - np.asarray((dx, dy))))
            keys.append(
                (-_quantize(progress), _quantize(max(0.0, path)), action_id)
            )
        for rank, (_key, action_id) in enumerate(
            sorted((key, action_id) for action_id, key in enumerate(keys))
        ):
            result[state_index, action_id] = float(rank)
    return result


def _score_map(
    plan: prior.RoleFeaturePlanV1, scores: np.ndarray
) -> dict[str, list[float]]:
    if scores.shape != (STATE_COUNT, ACTION_COUNT) or not np.isfinite(scores).all():
        raise PhysicalOutcomeScreenError("physical score matrix changed")
    return {
        state.state_id: [float(item) for item in scores[index]]
        for index, state in enumerate(plan.states)
    }


def report_arm_v1(
    plan: prior.RoleFeaturePlanV1, scores: np.ndarray
) -> dict[str, object]:
    return prior._augment_report(  # noqa: SLF001
        selection_metrics_v1(plan.groups, _score_map(plan, scores))
    )


def _array_identity_v1(value: np.ndarray) -> str:
    return _tensor_identity_v1(torch.from_numpy(np.ascontiguousarray(value)))


def _diagnostic_prediction_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    residual_scales: torch.Tensor,
) -> dict[str, object]:
    errors = predictions.to(torch.float64) - targets.to(torch.float64)
    rmse = torch.sqrt(torch.mean(errors.square(), dim=(0, 1)))
    standardized = errors / residual_scales.to(torch.float64)
    return {
        "per_output_rmse": {
            name: float(value) for name, value in zip(TARGET_NAMES, rmse.tolist(), strict=True)
        },
        "joint_standardized_mse": float(torch.mean(standardized.square()).item()),
    }


def score_checkpoint_arms_v1(
    checkpoint: Mapping[str, object], dataset: PhysicalDatasetV1
) -> tuple[dict[str, dict[str, object]], dict[str, object], dict[str, object]]:
    """Predict and score all arms; exposed so replay can aggregate independently."""

    eval_visual = project_visual_context_v1(dataset.eval, checkpoint["pca"])
    outcome_stats = checkpoint["outcome_stats"]
    prediction_artifacts: dict[str, object] = {}
    reports: dict[str, dict[str, object]] = {}
    diagnostics: dict[str, object] = {}
    for arm in LEARNED_ARMS:
        inputs = assemble_model_inputs_v1(dataset.eval, eval_visual, arm=arm)
        arm_checkpoint = checkpoint["arms"][arm]
        member_rows = []
        member_predictions = []
        for member in arm_checkpoint["members"]:
            predictions = predict_member_v1(
                member["state_dict"],
                inputs,
                arm_checkpoint["input_stats"],
                outcome_stats,
            )
            scores = physical_score_matrix_v1(dataset.eval.plan, predictions)
            report = report_arm_v1(dataset.eval.plan, scores)
            member_predictions.append(predictions)
            member_rows.append(
                {
                    "seed": member["seed"],
                    "outcomes": predictions.tolist(),
                    "outcome_identity_sha256": _tensor_identity_v1(predictions),
                    "physical_scores": scores.tolist(),
                    "score_identity_sha256": _array_identity_v1(scores),
                    "report": report,
                }
            )
        ensemble = torch.stack(member_predictions).mean(dim=0)
        ensemble_scores = physical_score_matrix_v1(dataset.eval.plan, ensemble)
        reports[arm] = report_arm_v1(dataset.eval.plan, ensemble_scores)
        prediction_artifacts[arm] = {
            "ensemble_outcomes": ensemble.tolist(),
            "ensemble_outcome_identity_sha256": _tensor_identity_v1(ensemble),
            "ensemble_physical_scores": ensemble_scores.tolist(),
            "ensemble_score_identity_sha256": _array_identity_v1(ensemble_scores),
            "members": member_rows,
        }
        diagnostics[arm] = _diagnostic_prediction_metrics(
            ensemble, dataset.eval.targets, outcome_stats["residual_scales"]
        )
    task = dense._require_refitted_task_payload_v1(  # noqa: SLF001
        checkpoint["task_action_only"], dataset.train.plan
    )
    task_scores = dense.score_task_action_only_v1(dataset.eval.plan, task)
    reports["task_action_only"] = report_arm_v1(dataset.eval.plan, task_scores)
    task_regret = reports["task_action_only"]["summary"]["normalized_rank_regret"]
    if task_regret != EXPECTED_TASK_EVAL_REGRET:
        raise PhysicalOutcomeScreenError("task/action-only evaluation regret changed")
    oracle_scores = np.asarray(
        [state.dense_ranks for state in dataset.eval.plan.states], dtype=np.float64
    )
    reports["privileged_physical_oracle"] = report_arm_v1(dataset.eval.plan, oracle_scores)
    reports["random_expected"] = prior._random_expected_report(dataset.eval.plan)  # noqa: SLF001
    zero = torch.zeros_like(dataset.eval.targets)
    means = outcome_stats["action_means"][None, :, :].expand_as(dataset.eval.targets)
    diagnostics["zero_motion"] = _diagnostic_prediction_metrics(
        zero, dataset.eval.targets, outcome_stats["residual_scales"]
    )
    diagnostics["train_only_action_means"] = _diagnostic_prediction_metrics(
        means, dataset.eval.targets, outcome_stats["residual_scales"]
    )
    prediction_artifacts["task_action_only"] = {
        "physical_scores": task_scores.tolist(),
        "score_identity_sha256": _array_identity_v1(task_scores),
    }
    prediction_artifacts["privileged_physical_oracle"] = {
        "physical_scores": oracle_scores.tolist(),
        "score_identity_sha256": _array_identity_v1(oracle_scores),
    }
    return reports, prediction_artifacts, diagnostics


def build_comparisons_v1(
    arms: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    return {
        "odometry_vs_task_action_only": prior.paired_family_scene_cluster_comparison_v1(
            arms[ODOMETRY_ARM]["group_results"],
            arms["task_action_only"]["group_results"],
            resamples=BOOTSTRAP_RESAMPLES,
            seed=BOOTSTRAP_SEED,
        ),
        "visual_vs_task_action_only": prior.paired_family_scene_cluster_comparison_v1(
            arms[VISUAL_ARM]["group_results"],
            arms["task_action_only"]["group_results"],
            resamples=BOOTSTRAP_RESAMPLES,
            seed=BOOTSTRAP_SEED,
        ),
        "visual_vs_odometry": prior.paired_family_scene_cluster_comparison_v1(
            arms[VISUAL_ARM]["group_results"],
            arms[ODOMETRY_ARM]["group_results"],
            resamples=BOOTSTRAP_RESAMPLES,
            seed=BOOTSTRAP_SEED,
        ),
    }


def scientific_gates_v1(
    arms: Mapping[str, Mapping[str, object]],
    prediction_artifacts: Mapping[str, object],
    comparisons: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    oracle = arms["privileged_physical_oracle"]["summary"]
    task_regret = float(arms["task_action_only"]["summary"]["normalized_rank_regret"])
    random_regret = float(arms["random_expected"]["summary"]["normalized_rank_regret"])
    b_members = prediction_artifacts[ODOMETRY_ARM]["members"]
    c_members = prediction_artifacts[VISUAL_ARM]["members"]
    b_seed_regrets = [float(row["report"]["summary"]["normalized_rank_regret"]) for row in b_members]
    c_seed_regrets = [float(row["report"]["summary"]["normalized_rank_regret"]) for row in c_members]
    matched_deltas = [right - left for left, right in zip(b_seed_regrets, c_seed_regrets, strict=True)]
    return {
        "2_privileged_physical_oracle": {
            "passed": oracle["normalized_rank_regret"] == 0.0
            and oracle["oracle_equivalent_selection_rate"] == 1.0,
            "normalized_rank_regret": oracle["normalized_rank_regret"],
            "oracle_equivalent_selection_rate": oracle["oracle_equivalent_selection_rate"],
        },
        "3_odometry_beats_task_action_only": {
            "passed": comparisons["odometry_vs_task_action_only"]["upper_95"] < 0.0
            and all(value < task_regret for value in b_seed_regrets),
            "measurement": comparisons["odometry_vs_task_action_only"],
            "task_regret": task_regret,
            "per_seed_odometry_regret": b_seed_regrets,
        },
        "4_visual_beats_task_action_only": {
            "passed": comparisons["visual_vs_task_action_only"]["upper_95"] < 0.0
            and all(value < task_regret for value in c_seed_regrets),
            "measurement": comparisons["visual_vs_task_action_only"],
            "task_regret": task_regret,
            "per_seed_visual_regret": c_seed_regrets,
        },
        "5_visual_beats_odometry": {
            "passed": comparisons["visual_vs_odometry"]["upper_95"] < 0.0
            and all(value < 0.0 for value in matched_deltas),
            "measurement": comparisons["visual_vs_odometry"],
            "matched_seed_visual_minus_odometry_regret": matched_deltas,
        },
        "6a_odometry_beats_random_expected": {
            "passed": float(arms[ODOMETRY_ARM]["summary"]["normalized_rank_regret"])
            < random_regret,
            "odometry": arms[ODOMETRY_ARM]["summary"]["normalized_rank_regret"],
            "random_expected": random_regret,
        },
        "6b_visual_beats_random_expected": {
            "passed": float(arms[VISUAL_ARM]["summary"]["normalized_rank_regret"])
            < random_regret,
            "visual": arms[VISUAL_ARM]["summary"]["normalized_rank_regret"],
            "random_expected": random_regret,
        },
    }


def _safety_report(dataset: PhysicalDatasetV1) -> dict[str, object]:
    result: dict[str, object] = {
        "status": prior.SAFETY_STATUS,
        "applicable": False,
        "passed": False,
        "claim": "NOT_APPLICABLE",
    }
    for role, plan in (("train", dataset.train.plan), ("eval", dataset.eval.plan)):
        falls = sum(sum(state.physical_fell) for state in plan.states)
        tips = sum(sum(state.physical_tipped) for state in plan.states)
        if falls or tips:
            raise PhysicalOutcomeScreenError("zero-event safety support changed")
        result[role] = {"branches": STATE_COUNT * ACTION_COUNT, "falls": falls, "tips": tips}
    return result


def access_accounting_v1() -> dict[str, object]:
    """Distinguish runner validation from scientific feature consumption."""

    return {
        "rgb_leaf_opens": 0,
        "encoder_executions": 0,
        "target_or_successor_token_grids_used_as_model_input": 0,
        "target_or_successor_token_grids_validation_only": (
            ACTION_COUNT * STATE_COUNT * 2
        ),
        "protected_material_accessed": False,
    }


def evaluate_primary_checkpoint_v1(
    checkpoint: Mapping[str, object],
    dataset: PhysicalDatasetV1,
    *,
    implementation_source_binding: Mapping[str, object],
) -> dict[str, object]:
    validate_checkpoint_v1(
        checkpoint,
        dataset=dataset,
        implementation_source_binding=implementation_source_binding,
    )
    arms, artifacts, diagnostics = score_checkpoint_arms_v1(checkpoint, dataset)
    comparisons = build_comparisons_v1(arms)
    gates = scientific_gates_v1(arms, artifacts, comparisons)
    result: dict[str, object] = {
        "schema": SCHEMA,
        "status": "COMPLETE_DEVELOPMENT_ONLY_PHYSICAL_OUTCOME_SCREEN",
        "claim_scope": "DEVELOPMENT_ONLY_MECHANISM_SCREEN_NOT_NAVIGATION_EVIDENCE",
        "config": config_v1(),
        "implementation_source_binding": _binding_v1(implementation_source_binding),
        "dataset_identity_sha256": dataset.identity_sha256,
        "train_data_identity_sha256": dataset.train.identity_sha256,
        "eval_data_identity_sha256": dataset.eval.identity_sha256,
        "feature_plan": {
            "identity_sha256": dataset.plans.identity_sha256,
            "train_identity_sha256": dataset.train.plan.identity_sha256,
            "eval_identity_sha256": dataset.eval.plan.identity_sha256,
        },
        "checkpoint_identity_sha256": checkpoint["identity_sha256"],
        "pca_identity_sha256": checkpoint["pca"]["identity_sha256"],
        "outcome_stats_identity_sha256": checkpoint["outcome_stats"]["identity_sha256"],
        "task_action_only_identity_sha256": checkpoint["task_action_only"]["identity_sha256"],
        "arms": arms,
        "prediction_artifacts": artifacts,
        "prediction_diagnostics": diagnostics,
        "paired_family_scene_cluster_comparisons": comparisons,
        "safety": _safety_report(dataset),
        "access_accounting": access_accounting_v1(),
        "gates": gates,
    }
    result["evaluation_identity_sha256"] = evaluation_identity_v1(result)
    return result


def evaluation_identity_v1(evaluation: Mapping[str, object]) -> str:
    document = dict(evaluation)
    document.pop("evaluation_identity_sha256", None)
    return hashlib.sha256(canonical_bytes_v1(document)).hexdigest()


def verdict_v1(
    evaluation: Mapping[str, object],
    *,
    infrastructure_checks_passed: bool,
    deterministic_replay_passed: bool,
) -> dict[str, object]:
    gates = evaluation.get("gates")
    if (
        evaluation.get("schema") != SCHEMA
        or evaluation.get("evaluation_identity_sha256") != evaluation_identity_v1(evaluation)
        or not isinstance(gates, Mapping)
        or set(gates) != SCIENTIFIC_GATE_NAMES
        or any(
            not isinstance(gate, Mapping) or type(gate.get("passed")) is not bool
            for gate in gates.values()
        )
        or type(infrastructure_checks_passed) is not bool
        or type(deterministic_replay_passed) is not bool
    ):
        raise PhysicalOutcomeScreenError("verdict inputs changed")
    external = {
        "1_infrastructure_and_custody": {"passed": infrastructure_checks_passed},
        **dict(gates),
        "7_deterministic_replay": {"passed": deterministic_replay_passed},
    }
    if not infrastructure_checks_passed or not deterministic_replay_passed:
        status = INFRASTRUCTURE_FAILURE_STATUS
    elif all(
        bool(gates[name]["passed"])
        for name in (
            "2_privileged_physical_oracle",
            "4_visual_beats_task_action_only",
            "5_visual_beats_odometry",
            "6b_visual_beats_random_expected",
        )
    ):
        status = PASS_VISUAL_STATUS
    elif all(
        bool(gates[name]["passed"])
        for name in (
            "2_privileged_physical_oracle",
            "3_odometry_beats_task_action_only",
            "6a_odometry_beats_random_expected",
        )
    ):
        status = PASS_ODOMETRY_STATUS
    else:
        status = STOP_STATUS
    return {
        "gates": external,
        "passed": status in {PASS_VISUAL_STATUS, PASS_ODOMETRY_STATUS},
        "terminal_status": status,
    }


__all__ = [
    "ACTION_COUNT",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CHECKPOINT_SCHEMA",
    "LEARNED_ARMS",
    "MODEL_SEEDS",
    "ODOMETRY_ARM",
    "PASS_ODOMETRY_STATUS",
    "PASS_VISUAL_STATUS",
    "PhysicalDatasetV1",
    "PhysicalOutcomeScreenError",
    "RolePhysicalDataV1",
    "SCHEMA",
    "SCIENTIFIC_GATE_NAMES",
    "STOP_STATUS",
    "TrainingDatasetV1",
    "VISUAL_ARM",
    "access_accounting_v1",
    "assemble_model_inputs_v1",
    "body_local_increment_v1",
    "build_comparisons_v1",
    "build_physical_dataset_v1",
    "build_role_physical_data_v1",
    "canonical_bytes_v1",
    "config_v1",
    "evaluate_primary_checkpoint_v1",
    "evaluation_identity_v1",
    "fit_input_statistics_v1",
    "fit_member_v1",
    "fit_outcome_statistics_v1",
    "fit_primary_checkpoint_v1",
    "fit_train_pca_v1",
    "normalize_model_inputs_v1",
    "pca_identity_v1",
    "physical_score_matrix_v1",
    "pool_context_grids_v1",
    "predict_member_v1",
    "project_visual_context_v1",
    "report_arm_v1",
    "scientific_gates_v1",
    "score_checkpoint_arms_v1",
    "standardized_residual_targets_v1",
    "training_orders_v1",
    "validate_checkpoint_v1",
    "verdict_v1",
]
