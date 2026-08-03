"""Train-only dense shared DINO readout and scene-disjoint evaluation.

The module has no filesystem or RGB access.  It consumes strict-loader group
objects and already rehashed token tensors.  Runtime custody, exclusive output,
and the distinct replay process are runner responsibilities.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from lewm.benchmarks import go2_dinov2_physical_readout_calibration_v1 as prior
from lewm.datasets.go2_world_model_counterfactual_pilot_v1 import (
    CANONICAL_ACTION_COMMANDS,
)
from lewm.models.go2_dinov2_dense_shared_spatial_readout_calibration_v1 import (
    PARAMETER_COUNT,
    DenseSharedSpatialReadoutV1,
    dense_shared_state_identity_v1,
    initialize_dense_shared_spatial_readout_v1,
)
from scripts.evaluate_go2_world_model_counterfactual_action_regret_v1 import (
    ActionSpecificRidgeReadoutsV1,
    RidgeReadoutV1,
    fit_ridge_readout_v1,
    predict_action_specific_scores_v1,
    selection_metrics_v1,
    task_conditioned_feature_v1,
)


SCHEMA = "lewm_go2_dinov2_dense_shared_spatial_readout_calibration_v1"
CONFIG_SCHEMA = "lewm_go2_dinov2_dense_shared_spatial_readout_config_v1"
CHECKPOINT_SCHEMA = (
    "lewm_go2_dinov2_dense_shared_spatial_readout_calibration_checkpoint_v1"
)
PCA_SCHEMA = "lewm_go2_dinov2_dense_shared_train_pca_v1"
TASK_PAYLOAD_SCHEMA = (
    "lewm_go2_dinov2_dense_shared_task_action_only_readout_payload_v1"
)
PASS_STATUS = "PASS_DENSE_SHARED_DINO_PHYSICAL_READOUT_HEADROOM_ESTABLISHED"
STOP_STATUS = "STOP_FROZEN_DINO_VISUAL_PLANNING_INTERFACE_NOT_ESTABLISHED"
INFRASTRUCTURE_FAILURE_STATUS = "FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION"
TERMINAL_STATUSES = frozenset((PASS_STATUS, STOP_STATUS, INFRASTRUCTURE_FAILURE_STATUS))

ACTION_COUNT = 9
STATE_COUNT = 128
TOKEN_COUNT = 256
TOKEN_DIMENSION = 384
PCA_DIMENSION = 8
RELATIONAL_DIMENSION = PCA_DIMENSION * 3
PCA_ROW_COUNT = STATE_COUNT * (1 + ACTION_COUNT) * TOKEN_COUNT
PCA_EPSILON = 1.0e-12
TOKEN_NORM_TOLERANCE = 2.0e-3
MODEL_SEEDS = (2_026_080_303, 2_026_080_304, 2_026_080_305)
EPOCHS = 256
BATCH_STATES = 16
STEPS_PER_EPOCH = STATE_COUNT // BATCH_STATES
OPTIMIZER_STEPS = EPOCHS * STEPS_PER_EPOCH
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-2
GRADIENT_CLIP_NORM = 1.0
TASK_RIDGE_LAMBDA = 1.0e-3
EXPECTED_TASK_IDENTITY = (
    "69895316b19bc179e35fdd76905aadbd50b6ad3e22e965b662ba59672c52886a"
)
EXPECTED_TASK_EVAL_REGRET = 0.17441406250000002
EXPECTED_TRAIN_PLAN_IDENTITY = (
    "f6f94cf589ec44324fdefe0939aa7076e25543d984464d5b264a0b2f0ff9535b"
)
EXPECTED_EVAL_PLAN_IDENTITY = (
    "5dbf9733fd245caff27ce5c5c2b3dc90a3fe9ca9e1bc894dc10a97d64dad9231"
)
EXPECTED_COMBINED_PLAN_IDENTITY = (
    "99e60638634eff6ac244cff023cd2ae8f7aa0c53326263ba7a36fa6847386375"
)
EXPECTED_ACTION_COMMANDS = (
    (0.20, 0.0, 0.45),
    (0.20, 0.0, -0.45),
    (-0.20, 0.0, 0.0),
    (0.30, 0.0, 0.0),
    (0.25, 0.0, 0.0),
    (0.20, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.45),
    (0.0, 0.0, -0.45),
)

SCIENTIFIC_GATE_NAMES = frozenset(
    {
        "2_privileged_physical_oracle",
        "3_true_future_beats_task_action_only",
        "4_true_future_beats_current_state",
        "5_true_future_beats_relational_persistence",
        "6_true_future_beats_random_expected",
    }
)


class DenseSharedCalibrationError(RuntimeError):
    """Raised for a mechanism, data, or deterministic-contract violation."""


def canonical_bytes_v1(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _implementation_binding_v1(value: object) -> dict[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256", "byte_count"}
        or not isinstance(value.get("path"), str)
        or not value["path"]
        or not isinstance(value.get("sha256"), str)
        or len(value["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in value["sha256"])
        or type(value.get("byte_count")) is not int
        or value["byte_count"] <= 0
    ):
        raise DenseSharedCalibrationError("implementation source binding changed")
    return dict(value)


def _require_exact_action_catalog_v1() -> None:
    if tuple(tuple(float(item) for item in command) for command in CANONICAL_ACTION_COMMANDS) != EXPECTED_ACTION_COMMANDS:
        raise DenseSharedCalibrationError("canonical requested action catalog changed")


def _require_plan_identities_v1(
    *, train: prior.RoleFeaturePlanV1, eval_plan: prior.RoleFeaturePlanV1 | None = None,
    combined_identity: str | None = None,
) -> None:
    if train.identity_sha256 != EXPECTED_TRAIN_PLAN_IDENTITY:
        raise DenseSharedCalibrationError("train feature-plan identity changed")
    if eval_plan is not None and eval_plan.identity_sha256 != EXPECTED_EVAL_PLAN_IDENTITY:
        raise DenseSharedCalibrationError("evaluation feature-plan identity changed")
    if combined_identity is not None and combined_identity != EXPECTED_COMBINED_PLAN_IDENTITY:
        raise DenseSharedCalibrationError("combined feature-plan identity changed")


def _require_rocm_determinism_v1(device: torch.device) -> None:
    if (
        device.type != "cuda"
        or not torch.cuda.is_available()
        or torch.version.hip is None
        or not torch.are_deterministic_algorithms_enabled()
        or torch.get_float32_matmul_precision() != "highest"
    ):
        raise DenseSharedCalibrationError("authorized ROCm deterministic runtime changed")


def config_v1() -> dict[str, object]:
    return {
        "schema": CONFIG_SCHEMA,
        "role_contract": {
            "states_per_role": STATE_COUNT,
            "actions": ACTION_COUNT,
            "tokens_per_grid": TOKEN_COUNT,
            "token_dimension": TOKEN_DIMENSION,
            "storage_dtype": "float16",
            "token_norm_tolerance": TOKEN_NORM_TOLERANCE,
            "train_plan_identity": EXPECTED_TRAIN_PLAN_IDENTITY,
            "eval_plan_identity": EXPECTED_EVAL_PLAN_IDENTITY,
            "combined_plan_identity": EXPECTED_COMBINED_PLAN_IDENTITY,
        },
        "pca": {
            "dimension": PCA_DIMENSION,
            "row_count": PCA_ROW_COUNT,
            "source_grids": STATE_COUNT * (1 + ACTION_COUNT),
            "source_order": "all_current_then_state_major_action_major_successors",
            "patch_order": "row_major_16x16",
            "statistics_dtype": "float64",
            "covariance": "population_divisor_327680",
            "eigensolver": "numpy.linalg.eigh",
            "ordering": "descending_eigenvalue_then_original_ascending_index",
            "sign": "largest_absolute_loading_smallest_channel_positive",
            "whitening_epsilon": PCA_EPSILON,
            "clipping": False,
        },
        "condition": {
            "fields": ["goal_x", "goal_y", "requested_vx", "requested_wz"],
            "divisors": [10.0, 10.0, 0.30, 0.45],
            "action_commands_vx_vy_wz": [list(value) for value in EXPECTED_ACTION_COMMANDS],
        },
        "scorer": {
            "relational_fields": ["current", "successor", "successor_minus_current"],
            "relational_dimension": RELATIONAL_DIMENSION,
            "patch_coordinates": "u_v_row_major_cell_centres_minus1_plus1",
            "hidden_width": 4,
            "value_width": 4,
            "attention": "softmax_over_256_patches",
            "q_only_shortcut": False,
        },
        "model_seeds": list(MODEL_SEEDS),
        "parameter_count_per_member": PARAMETER_COUNT,
        "true_ensemble_parameter_count": PARAMETER_COUNT * len(MODEL_SEEDS),
        "current_ensemble_parameter_count": PARAMETER_COUNT * len(MODEL_SEEDS),
        "checkpoint_dense_parameter_count": PARAMETER_COUNT * len(MODEL_SEEDS) * 2,
        "epochs": EPOCHS,
        "batch_states": BATCH_STATES,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "optimizer_steps": OPTIMIZER_STEPS,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "betas": [0.9, 0.999],
            "epsilon": 1.0e-8,
            "amsgrad": False,
            "maximize": False,
            "foreach": False,
            "fused": False,
        },
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "initialization": {
            "device": "cpu",
            "method": "xavier_uniform",
            "gain": 1.0,
            "draw_order": ["W_r", "W_p", "W_q", "W_v", "B", "w_alpha", "w_z"],
            "biases": "zero",
            "generator": "dedicated_cpu_generator_member_seed",
        },
        "task_action_only_base": {
            "feature_fields": ["goal_x", "goal_y", "constant_one"],
            "separate_action_heads": ACTION_COUNT,
            "fitted_coefficients": 27,
            "ridge_lambda": TASK_RIDGE_LAMBDA,
            "expected_identity_sha256": EXPECTED_TASK_IDENTITY,
            "expected_eval_regret": EXPECTED_TASK_EVAL_REGRET,
        },
        "residual_target": "normalized_dense_rank_minus_task_action_only_score",
        "complete_state_minibatches": True,
        "numeric_contract": {
            "scorer_dtype": "float32",
            "runtime": "ROCm",
            "deterministic_algorithms": True,
            "float32_matmul_precision": "highest",
        },
        "bootstrap_resamples": prior.BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": prior.BOOTSTRAP_SEED,
    }


def patch_coordinates_v1() -> torch.Tensor:
    values = [
        (2.0 * (col + 0.5) / 16.0 - 1.0, 2.0 * (row + 0.5) / 16.0 - 1.0)
        for row in range(16)
        for col in range(16)
    ]
    return torch.tensor(values, dtype=torch.float32)


def _validate_cache(features: object, *, label: str) -> torch.Tensor:
    if not isinstance(features, torch.Tensor):
        raise DenseSharedCalibrationError(f"{label} cache must be a tensor")
    if features.shape != (prior.ROLE_ARTIFACT_COUNT, TOKEN_COUNT, TOKEN_DIMENSION):
        raise DenseSharedCalibrationError(f"{label} cache shape changed")
    if features.dtype != torch.float16 or features.device.type != "cpu":
        raise DenseSharedCalibrationError(f"{label} cache must be CPU float16")
    if not bool(torch.isfinite(features).all()):
        raise DenseSharedCalibrationError(f"{label} cache contains nonfinite tokens")
    norms = torch.linalg.vector_norm(features.float(), dim=-1)
    maximum_error = float((norms - 1.0).abs().max())
    if not math.isfinite(maximum_error) or maximum_error > TOKEN_NORM_TOLERANCE:
        raise DenseSharedCalibrationError(
            f"{label} per-token normalization changed: {maximum_error}"
        )
    return features


def _pca_source_indices(plan: prior.RoleFeaturePlanV1) -> tuple[int, ...]:
    if plan.role != "train" or len(plan.states) != STATE_COUNT:
        raise DenseSharedCalibrationError("PCA requires the exact train role")
    current = tuple(state.context_artifact_indices[-1] for state in plan.states)
    successors = tuple(
        state.target_artifact_indices[action]
        for state in plan.states
        for action in range(ACTION_COUNT)
    )
    result = current + successors
    if len(result) != STATE_COUNT * (1 + ACTION_COUNT) or len(set(result)) != len(result):
        raise DenseSharedCalibrationError("PCA source order changed")
    return result


def _array_digest_update(digest: Any, array: np.ndarray) -> None:
    canonical = np.ascontiguousarray(array.astype("<f8", copy=False))
    digest.update(str(canonical.shape).encode("ascii") + b"\0")
    digest.update(canonical.tobytes())


def fit_train_pca_v1(
    train_plan: prior.RoleFeaturePlanV1,
    train_features: object,
    *,
    implementation_source_binding: Mapping[str, object],
) -> dict[str, object]:
    features = _validate_cache(train_features, label="train")
    indices = _pca_source_indices(train_plan)
    source = features[list(indices)].reshape(PCA_ROW_COUNT, TOKEN_DIMENSION).numpy()
    matrix = source.astype(np.float64)
    mean = matrix.mean(axis=0, dtype=np.float64)
    matrix -= mean
    covariance = (matrix.T @ matrix) / float(PCA_ROW_COUNT)
    if covariance.shape != (TOKEN_DIMENSION, TOKEN_DIMENSION) or not np.isfinite(
        covariance
    ).all():
        raise DenseSharedCalibrationError("PCA covariance is invalid")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    original_indices = np.arange(TOKEN_DIMENSION, dtype=np.int64)
    order = np.lexsort((original_indices, -eigenvalues))[:PCA_DIMENSION]
    selected_values = np.ascontiguousarray(eigenvalues[order], dtype=np.float64)
    components = np.ascontiguousarray(eigenvectors[:, order], dtype=np.float64)
    for column in range(PCA_DIMENSION):
        absolute = np.abs(components[:, column])
        pivot = int(np.flatnonzero(absolute == absolute.max())[0])
        if components[pivot, column] < 0.0:
            components[:, column] *= -1.0
    scales = np.sqrt(np.maximum(selected_values, PCA_EPSILON))
    if (
        selected_values.shape != (PCA_DIMENSION,)
        or components.shape != (TOKEN_DIMENSION, PCA_DIMENSION)
        or not np.isfinite(mean).all()
        or not np.isfinite(selected_values).all()
        or not np.isfinite(components).all()
        or not np.isfinite(scales).all()
    ):
        raise DenseSharedCalibrationError("PCA eigensystem is invalid")
    source_document = {
        "train_plan_identity": train_plan.identity_sha256,
        "artifact_indices": list(indices),
        "artifact_ids": [train_plan.artifact_ids[index] for index in indices],
        "patch_order": "row_major_16x16",
        "row_count": PCA_ROW_COUNT,
    }
    result: dict[str, object] = {
        "schema": PCA_SCHEMA,
        "source": source_document,
        "implementation_source_binding": _implementation_binding_v1(
            implementation_source_binding
        ),
        "epsilon": PCA_EPSILON,
        "mean": torch.from_numpy(mean.copy()),
        "eigenvalues": torch.from_numpy(selected_values.copy()),
        "components": torch.from_numpy(components.copy()),
        "scales": torch.from_numpy(scales.copy()),
    }
    result["identity_sha256"] = pca_identity_v1(result)
    _validate_pca(result, expected_train_plan=train_plan)
    return result


def _pca_arrays_v1(
    pca: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays: list[np.ndarray] = []
    for name, shape in (
        ("mean", (TOKEN_DIMENSION,)),
        ("eigenvalues", (PCA_DIMENSION,)),
        ("components", (TOKEN_DIMENSION, PCA_DIMENSION)),
        ("scales", (PCA_DIMENSION,)),
    ):
        value = pca.get(name)
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float64:
            raise DenseSharedCalibrationError(f"PCA {name} changed")
        array = value.detach().cpu().numpy()
        if array.shape != shape or not np.isfinite(array).all():
            raise DenseSharedCalibrationError(f"PCA {name} is invalid")
        arrays.append(array)
    return arrays[0], arrays[1], arrays[2], arrays[3]


def pca_identity_v1(pca: Mapping[str, object]) -> str:
    required = {
        "schema",
        "source",
        "implementation_source_binding",
        "epsilon",
        "mean",
        "eigenvalues",
        "components",
        "scales",
    }
    if set(pca) not in (required, required | {"identity_sha256"}):
        raise DenseSharedCalibrationError("PCA payload inventory changed")
    if pca.get("schema") != PCA_SCHEMA or pca.get("epsilon") != PCA_EPSILON:
        raise DenseSharedCalibrationError("PCA contract changed")
    source = pca.get("source")
    if (
        not isinstance(source, Mapping)
        or set(source)
        != {
            "train_plan_identity",
            "artifact_indices",
            "artifact_ids",
            "patch_order",
            "row_count",
        }
        or not isinstance(source.get("train_plan_identity"), str)
        or source.get("patch_order") != "row_major_16x16"
        or source.get("row_count") != PCA_ROW_COUNT
        or not isinstance(source.get("artifact_indices"), list)
        or not isinstance(source.get("artifact_ids"), list)
        or len(source["artifact_indices"]) != STATE_COUNT * (1 + ACTION_COUNT)
        or len(source["artifact_ids"]) != STATE_COUNT * (1 + ACTION_COUNT)
        or any(type(index) is not int for index in source["artifact_indices"])
        or any(not isinstance(item, str) or not item for item in source["artifact_ids"])
    ):
        raise DenseSharedCalibrationError("PCA source contract changed")
    implementation = _implementation_binding_v1(
        pca.get("implementation_source_binding")
    )
    mean, eigenvalues, components, scales = _pca_arrays_v1(pca)
    if np.any(eigenvalues[:-1] < eigenvalues[1:]):
        raise DenseSharedCalibrationError("PCA eigenvalue order changed")
    expected_scales = np.sqrt(np.maximum(eigenvalues, PCA_EPSILON))
    if not np.array_equal(scales, expected_scales):
        raise DenseSharedCalibrationError("PCA whitening scale changed")
    for column in range(PCA_DIMENSION):
        absolute = np.abs(components[:, column])
        pivot = int(np.flatnonzero(absolute == absolute.max())[0])
        if components[pivot, column] < 0.0:
            raise DenseSharedCalibrationError("PCA component sign changed")
    metadata = {
        "schema": PCA_SCHEMA,
        "source": dict(source),
        "implementation_source_binding": implementation,
        "epsilon": PCA_EPSILON,
    }
    digest = hashlib.sha256(canonical_bytes_v1(metadata))
    for array in (mean, eigenvalues, components, scales):
        _array_digest_update(digest, array)
    return digest.hexdigest()


def _validate_pca(
    pca: Mapping[str, object],
    *,
    expected_train_plan: prior.RoleFeaturePlanV1 | None = None,
    expected_implementation_source_binding: Mapping[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    identity = pca_identity_v1(pca)
    if pca.get("identity_sha256") != identity:
        raise DenseSharedCalibrationError("PCA identity changed")
    source = pca["source"]
    assert isinstance(source, Mapping)
    if expected_train_plan is not None:
        indices = _pca_source_indices(expected_train_plan)
        if (
            source.get("train_plan_identity") != expected_train_plan.identity_sha256
            or source.get("artifact_indices") != list(indices)
            or source.get("artifact_ids")
            != [expected_train_plan.artifact_ids[index] for index in indices]
        ):
            raise DenseSharedCalibrationError("PCA train source changed")
    if (
        expected_implementation_source_binding is not None
        and pca.get("implementation_source_binding")
        != _implementation_binding_v1(expected_implementation_source_binding)
    ):
        raise DenseSharedCalibrationError("PCA implementation source changed")
    mean, _eigenvalues, components, scales = _pca_arrays_v1(pca)
    return mean, components, scales


def project_cache_v1(features: object, pca: Mapping[str, object], *, label: str) -> torch.Tensor:
    cache = _validate_cache(features, label=label)
    mean, components, scales = _validate_pca(pca)
    output = np.empty(
        (prior.ROLE_ARTIFACT_COUNT, TOKEN_COUNT, PCA_DIMENSION), dtype=np.float32
    )
    block = 32
    for start in range(0, prior.ROLE_ARTIFACT_COUNT, block):
        stop = min(start + block, prior.ROLE_ARTIFACT_COUNT)
        selected = cache[start:stop].numpy().astype(np.float64)
        projected = ((selected - mean) @ components) / scales
        if not np.isfinite(projected).all():
            raise DenseSharedCalibrationError(f"{label} PCA projection is nonfinite")
        output[start:stop] = projected.astype(np.float32)
    return torch.from_numpy(output)


def _normalized_rank_targets(plan: prior.RoleFeaturePlanV1) -> np.ndarray:
    rows = []
    for state in plan.states:
        ranks = np.asarray(state.dense_ranks, dtype=np.float64)
        if ranks.shape != (ACTION_COUNT,) or ranks.max() <= 0:
            raise DenseSharedCalibrationError("dense ranks changed")
        rows.append(ranks / ranks.max())
    result = np.stack(rows)
    if result.shape != (STATE_COUNT, ACTION_COUNT) or not np.isfinite(result).all():
        raise DenseSharedCalibrationError("normalized ranks are invalid")
    return result


def _assemble_task_heads(heads: Sequence[RidgeReadoutV1]) -> ActionSpecificRidgeReadoutsV1:
    if len(heads) != ACTION_COUNT:
        raise DenseSharedCalibrationError("task control requires nine heads")
    digest = hashlib.sha256()
    for action, head in enumerate(heads):
        digest.update(action.to_bytes(2, "little"))
        digest.update(bytes.fromhex(head.identity_sha256))
    return ActionSpecificRidgeReadoutsV1(tuple(heads), digest.hexdigest())


def fit_task_action_only_v1(
    train_plan: prior.RoleFeaturePlanV1,
) -> ActionSpecificRidgeReadoutsV1:
    targets = _normalized_rank_targets(train_plan)
    heads = []
    for action in range(ACTION_COUNT):
        features = np.stack(
            [
                task_conditioned_feature_v1(
                    None, relative_target_xy_body_m=state.relative_target_xy_body_m
                )
                for state in train_plan.states
            ]
        )
        heads.append(
            fit_ridge_readout_v1(
                features, targets[:, action], ridge_lambda=TASK_RIDGE_LAMBDA
            )
        )
    result = _assemble_task_heads(heads)
    if result.identity_sha256 != EXPECTED_TASK_IDENTITY:
        raise DenseSharedCalibrationError("task/action-only control identity changed")
    return result


def score_task_action_only_v1(
    plan: prior.RoleFeaturePlanV1, readouts: ActionSpecificRidgeReadoutsV1
) -> np.ndarray:
    rows = []
    for state in plan.states:
        features = [
            task_conditioned_feature_v1(
                None, relative_target_xy_body_m=state.relative_target_xy_body_m
            )
            for _action in range(ACTION_COUNT)
        ]
        rows.append(predict_action_specific_scores_v1(readouts, features))
    result = np.stack(rows)
    if result.shape != (STATE_COUNT, ACTION_COUNT) or not np.isfinite(result).all():
        raise DenseSharedCalibrationError("task/action-only scores are invalid")
    return result


def _task_payload(readouts: ActionSpecificRidgeReadoutsV1) -> dict[str, object]:
    return {
        "schema": TASK_PAYLOAD_SCHEMA,
        "identity_sha256": readouts.identity_sha256,
        "heads": [
            {
                "feature_mean": torch.from_numpy(head.feature_mean.copy()),
                "feature_scale": torch.from_numpy(head.feature_scale.copy()),
                "coefficients": torch.from_numpy(head.coefficients.copy()),
                "ridge_lambda": head.ridge_lambda,
                "training_rows": head.training_rows,
                "solver": head.solver,
                "identity_sha256": head.identity_sha256,
            }
            for head in readouts.heads
        ],
    }


def _ridge_identity_v1(
    mean: np.ndarray,
    scale: np.ndarray,
    coefficients: np.ndarray,
    *,
    ridge_lambda: float,
    solver: str,
) -> str:
    digest = hashlib.sha256()
    for array in (mean, scale, coefficients):
        canonical = np.ascontiguousarray(array.astype("<f8", copy=False))
        digest.update(str(canonical.shape).encode("ascii") + b"\0")
        digest.update(canonical.tobytes())
    digest.update(np.asarray([ridge_lambda], dtype="<f8").tobytes())
    digest.update(solver.encode("ascii"))
    return digest.hexdigest()


def _task_from_payload(payload: Mapping[str, object]) -> ActionSpecificRidgeReadoutsV1:
    raw_heads = payload.get("heads")
    if (
        set(payload) != {"schema", "identity_sha256", "heads"}
        or payload.get("schema") != TASK_PAYLOAD_SCHEMA
        or payload.get("identity_sha256") != EXPECTED_TASK_IDENTITY
        or not isinstance(raw_heads, list)
        or len(raw_heads) != ACTION_COUNT
    ):
        raise DenseSharedCalibrationError("task readout payload changed")
    heads = []
    for value in raw_heads:
        if (
            not isinstance(value, Mapping)
            or set(value)
            != {
                "feature_mean",
                "feature_scale",
                "coefficients",
                "ridge_lambda",
                "training_rows",
                "solver",
                "identity_sha256",
            }
        ):
            raise DenseSharedCalibrationError("task head payload changed")
        arrays = []
        for name, shape in (
            ("feature_mean", (3,)),
            ("feature_scale", (3,)),
            ("coefficients", (4, 1)),
        ):
            tensor = value.get(name)
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.float64
                or tensor.device.type != "cpu"
                or tuple(tensor.shape) != shape
                or not bool(torch.isfinite(tensor).all())
            ):
                raise DenseSharedCalibrationError("task head tensor changed")
            arrays.append(tensor.detach().cpu().numpy())
        if (
            not np.all(arrays[1] > 0.0)
            or value.get("ridge_lambda") != TASK_RIDGE_LAMBDA
            or value.get("training_rows") != STATE_COUNT
            or value.get("solver") != "primal"
            or value.get("identity_sha256")
            != _ridge_identity_v1(
                arrays[0],
                arrays[1],
                arrays[2],
                ridge_lambda=TASK_RIDGE_LAMBDA,
                solver="primal",
            )
        ):
            raise DenseSharedCalibrationError("task head identity changed")
        heads.append(
            RidgeReadoutV1(
                feature_mean=arrays[0],
                feature_scale=arrays[1],
                coefficients=arrays[2],
                ridge_lambda=float(value["ridge_lambda"]),
                training_rows=int(value["training_rows"]),
                solver=str(value["solver"]),
                identity_sha256=str(value["identity_sha256"]),
            )
        )
    result = _assemble_task_heads(heads)
    if result.identity_sha256 != payload["identity_sha256"]:
        raise DenseSharedCalibrationError("task readout identity changed")
    return result


def _require_refitted_task_payload_v1(
    payload: Mapping[str, object], train_plan: prior.RoleFeaturePlanV1
) -> ActionSpecificRidgeReadoutsV1:
    loaded = _task_from_payload(payload)
    refitted = fit_task_action_only_v1(train_plan)
    for loaded_head, refitted_head in zip(loaded.heads, refitted.heads, strict=True):
        if (
            loaded_head.identity_sha256 != refitted_head.identity_sha256
            or loaded_head.ridge_lambda != refitted_head.ridge_lambda
            or loaded_head.training_rows != refitted_head.training_rows
            or loaded_head.solver != refitted_head.solver
            or not np.array_equal(loaded_head.feature_mean, refitted_head.feature_mean)
            or not np.array_equal(loaded_head.feature_scale, refitted_head.feature_scale)
            or not np.array_equal(loaded_head.coefficients, refitted_head.coefficients)
        ):
            raise DenseSharedCalibrationError("task readout no longer matches train refit")
    return refitted


def _dense_panels(
    plan: prior.RoleFeaturePlanV1, projected: torch.Tensor, *, successor_mode: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if projected.shape != (prior.ROLE_ARTIFACT_COUNT, TOKEN_COUNT, PCA_DIMENSION):
        raise DenseSharedCalibrationError("projected cache shape changed")
    relations = torch.empty(
        (STATE_COUNT, ACTION_COUNT, TOKEN_COUNT, RELATIONAL_DIMENSION),
        dtype=torch.float32,
    )
    conditions = torch.empty((STATE_COUNT, ACTION_COUNT, 4), dtype=torch.float32)
    for state_index, state in enumerate(plan.states):
        current = projected[state.context_artifact_indices[-1]]
        goal_x, goal_y = state.relative_target_xy_body_m
        for action in range(ACTION_COUNT):
            if successor_mode == "true_future":
                successor = projected[state.target_artifact_indices[action]]
            elif successor_mode == "current_state":
                successor = current
            else:
                raise DenseSharedCalibrationError("successor mode changed")
            relations[state_index, action] = torch.cat(
                (current, successor, successor - current), dim=-1
            )
            command = CANONICAL_ACTION_COMMANDS[action]
            conditions[state_index, action] = torch.tensor(
                (goal_x / 10.0, goal_y / 10.0, command[0] / 0.30, command[2] / 0.45),
                dtype=torch.float32,
            )
    if not bool(torch.isfinite(relations).all()) or not bool(
        torch.isfinite(conditions).all()
    ):
        raise DenseSharedCalibrationError("dense panels are nonfinite")
    return relations, conditions


def _clone_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _train_one_model(
    initial_state: Mapping[str, torch.Tensor],
    relations: torch.Tensor,
    conditions: torch.Tensor,
    residual_targets: torch.Tensor,
    orders: Sequence[torch.Tensor],
    *,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    model = DenseSharedSpatialReadoutV1().to(device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        amsgrad=False,
        maximize=False,
        foreach=False,
        fused=False,
    )
    step_count = 0
    last_loss = math.nan
    last_gradient_norm = math.nan
    model.train()
    for order in orders:
        if order.shape != (STATE_COUNT,):
            raise DenseSharedCalibrationError("epoch state order changed")
        for start in range(0, STATE_COUNT, BATCH_STATES):
            selected = order[start : start + BATCH_STATES]
            batch_relations = relations[selected].reshape(
                BATCH_STATES * ACTION_COUNT, TOKEN_COUNT, RELATIONAL_DIMENSION
            ).to(device)
            batch_conditions = conditions[selected].reshape(
                BATCH_STATES * ACTION_COUNT, 4
            ).to(device)
            batch_targets = residual_targets[selected].reshape(-1).to(device)
            optimizer.zero_grad(set_to_none=True)
            batch_output = model.forward_with_attention(
                batch_relations, batch_conditions
            )
            predictions = batch_output.score
            loss = torch.mean((predictions - batch_targets) ** 2)
            if not bool(torch.isfinite(loss)):
                raise DenseSharedCalibrationError("dense training loss is nonfinite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRADIENT_CLIP_NORM, norm_type=2.0
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise DenseSharedCalibrationError("dense gradient norm is nonfinite")
            optimizer.step()
            step_count += 1
            last_loss = float(loss.detach())
            last_gradient_norm = float(gradient_norm.detach())
    if step_count != OPTIMIZER_STEPS:
        raise DenseSharedCalibrationError("optimizer step count changed")
    model.eval()
    with torch.no_grad():
        full_output = model.forward_with_attention(
            relations.reshape(-1, TOKEN_COUNT, RELATIONAL_DIMENSION).to(device),
            conditions.reshape(-1, 4).to(device),
        )
        full_predictions = full_output.score
        attention = full_output.attention
        targets = residual_targets.reshape(-1).to(device)
        full_mse = float(torch.mean((full_predictions - targets) ** 2))
        entropy = float(
            (-(attention * torch.log(attention.clamp_min(1.0e-12))).sum(dim=-1))
            .div(math.log(TOKEN_COUNT))
            .mean()
        )
    state = _clone_state(model.state_dict())
    return state, {
        "optimizer_steps": step_count,
        "last_minibatch_residual_mse": last_loss,
        "last_gradient_norm_before_clip": last_gradient_norm,
        "full_train_residual_mse": full_mse,
        "mean_normalized_attention_entropy": entropy,
        "state_identity_sha256": dense_shared_state_identity_v1(state),
    }


_TRAINING_REPORT_FIELDS = frozenset(
    {
        "optimizer_steps",
        "last_minibatch_residual_mse",
        "last_gradient_norm_before_clip",
        "full_train_residual_mse",
        "mean_normalized_attention_entropy",
        "state_identity_sha256",
    }
)
_MEMBER_FIELDS = frozenset(
    {
        "seed",
        "initial_identity_sha256",
        "true_state",
        "true_identity_sha256",
        "true_training",
        "current_state",
        "current_identity_sha256",
        "current_training",
    }
)


def _validated_training_report_v1(
    value: object, *, expected_state_identity: str
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _TRAINING_REPORT_FIELDS:
        raise DenseSharedCalibrationError("training report inventory changed")
    if (
        value.get("optimizer_steps") != OPTIMIZER_STEPS
        or value.get("state_identity_sha256") != expected_state_identity
    ):
        raise DenseSharedCalibrationError("training report identity changed")
    numeric_names = (
        "last_minibatch_residual_mse",
        "last_gradient_norm_before_clip",
        "full_train_residual_mse",
        "mean_normalized_attention_entropy",
    )
    for name in numeric_names:
        number = value.get(name)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(float(number)):
            raise DenseSharedCalibrationError("training diagnostic is nonfinite")
    if (
        float(value["last_minibatch_residual_mse"]) < 0.0
        or float(value["last_gradient_norm_before_clip"]) < 0.0
        or float(value["full_train_residual_mse"]) < 0.0
        or not 0.0 <= float(value["mean_normalized_attention_entropy"]) <= 1.0
    ):
        raise DenseSharedCalibrationError("training diagnostic is out of range")
    return dict(value)


def _checkpoint_content_identity_v1(checkpoint: Mapping[str, object]) -> str:
    required = {
        "schema",
        "config",
        "train_plan_identity",
        "pca",
        "task_action_only",
        "members",
    }
    if set(checkpoint) not in (required, required | {"identity_sha256"}):
        raise DenseSharedCalibrationError("checkpoint inventory changed")
    if (
        checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("config") != config_v1()
        or checkpoint.get("train_plan_identity") != EXPECTED_TRAIN_PLAN_IDENTITY
    ):
        raise DenseSharedCalibrationError("checkpoint contract changed")
    pca = checkpoint.get("pca")
    task_payload = checkpoint.get("task_action_only")
    members = checkpoint.get("members")
    if (
        not isinstance(pca, Mapping)
        or not isinstance(task_payload, Mapping)
        or not isinstance(members, list)
        or len(members) != len(MODEL_SEEDS)
    ):
        raise DenseSharedCalibrationError("checkpoint content changed")
    pca_identity = pca_identity_v1(pca)
    if pca.get("identity_sha256") != pca_identity:
        raise DenseSharedCalibrationError("checkpoint PCA identity changed")
    task = _task_from_payload(task_payload)
    member_documents = []
    for expected_seed, member in zip(MODEL_SEEDS, members, strict=True):
        if (
            not isinstance(member, Mapping)
            or set(member) != _MEMBER_FIELDS
            or member.get("seed") != expected_seed
        ):
            raise DenseSharedCalibrationError("checkpoint member changed")
        expected_initial = dense_shared_state_identity_v1(
            initialize_dense_shared_spatial_readout_v1(expected_seed)
        )
        if member.get("initial_identity_sha256") != expected_initial:
            raise DenseSharedCalibrationError("checkpoint initial state changed")
        identities: dict[str, str] = {}
        reports: dict[str, dict[str, object]] = {}
        for prefix in ("true", "current"):
            state = member.get(f"{prefix}_state")
            if not isinstance(state, Mapping):
                raise DenseSharedCalibrationError("checkpoint model state changed")
            actual_identity = dense_shared_state_identity_v1(state)
            if member.get(f"{prefix}_identity_sha256") != actual_identity:
                raise DenseSharedCalibrationError("checkpoint model identity changed")
            identities[prefix] = actual_identity
            reports[prefix] = _validated_training_report_v1(
                member.get(f"{prefix}_training"),
                expected_state_identity=actual_identity,
            )
        member_documents.append(
            {
                "seed": expected_seed,
                "initial_identity_sha256": expected_initial,
                "true_identity_sha256": identities["true"],
                "true_training": reports["true"],
                "current_identity_sha256": identities["current"],
                "current_training": reports["current"],
            }
        )
    document = {
        "schema": CHECKPOINT_SCHEMA,
        "config": config_v1(),
        "train_plan_identity": checkpoint["train_plan_identity"],
        "pca_identity_sha256": pca_identity,
        "task_identity_sha256": task.identity_sha256,
        "members": member_documents,
    }
    return hashlib.sha256(canonical_bytes_v1(document)).hexdigest()


def validate_checkpoint_v1(
    checkpoint: Mapping[str, object],
    *,
    train_plan: prior.RoleFeaturePlanV1,
    implementation_source_binding: Mapping[str, object],
) -> None:
    _require_plan_identities_v1(train=train_plan)
    pca = checkpoint.get("pca")
    if not isinstance(pca, Mapping):
        raise DenseSharedCalibrationError("checkpoint PCA changed")
    _validate_pca(
        pca,
        expected_train_plan=train_plan,
        expected_implementation_source_binding=implementation_source_binding,
    )
    task_payload = checkpoint.get("task_action_only")
    if not isinstance(task_payload, Mapping):
        raise DenseSharedCalibrationError("checkpoint task payload changed")
    _require_refitted_task_payload_v1(task_payload, train_plan)
    identity = _checkpoint_content_identity_v1(checkpoint)
    if checkpoint.get("identity_sha256") != identity:
        raise DenseSharedCalibrationError("checkpoint identity changed")


def fit_primary_checkpoint_v1(
    train_groups: Sequence[Any],
    train_features: object,
    device: torch.device,
    *,
    implementation_source_binding: Mapping[str, object],
) -> dict[str, object]:
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    _require_rocm_determinism_v1(device)
    _require_exact_action_catalog_v1()
    train_plan = prior.build_role_feature_plan_v1(train_groups, role="train")
    _require_plan_identities_v1(train=train_plan)
    pca = fit_train_pca_v1(
        train_plan,
        train_features,
        implementation_source_binding=implementation_source_binding,
    )
    projected = project_cache_v1(train_features, pca, label="train")
    task = fit_task_action_only_v1(train_plan)
    task_scores = score_task_action_only_v1(train_plan, task)
    residual_targets = torch.from_numpy(
        (_normalized_rank_targets(train_plan) - task_scores).astype(np.float32)
    )
    true_relations, conditions = _dense_panels(
        train_plan, projected, successor_mode="true_future"
    )
    current_relations, current_conditions = _dense_panels(
        train_plan, projected, successor_mode="current_state"
    )
    if not torch.equal(conditions, current_conditions):
        raise DenseSharedCalibrationError("TRUE/CURRENT conditions changed")
    members = []
    for seed in MODEL_SEEDS:
        initialized = initialize_dense_shared_spatial_readout_v1(seed)
        initial_state = _clone_state(initialized.state_dict())
        generator = torch.Generator(device="cpu").manual_seed(seed)
        orders = tuple(torch.randperm(STATE_COUNT, generator=generator) for _ in range(EPOCHS))
        true_state, true_training = _train_one_model(
            initial_state,
            true_relations,
            conditions,
            residual_targets,
            orders,
            device=device,
        )
        current_state, current_training = _train_one_model(
            initial_state,
            current_relations,
            conditions,
            residual_targets,
            orders,
            device=device,
        )
        members.append(
            {
                "seed": seed,
                "initial_identity_sha256": dense_shared_state_identity_v1(initial_state),
                "true_state": true_state,
                "true_identity_sha256": dense_shared_state_identity_v1(true_state),
                "true_training": true_training,
                "current_state": current_state,
                "current_identity_sha256": dense_shared_state_identity_v1(current_state),
                "current_training": current_training,
            }
        )
    checkpoint: dict[str, object] = {
        "schema": CHECKPOINT_SCHEMA,
        "config": config_v1(),
        "train_plan_identity": train_plan.identity_sha256,
        "pca": pca,
        "task_action_only": _task_payload(task),
        "members": members,
    }
    checkpoint["identity_sha256"] = _checkpoint_content_identity_v1(checkpoint)
    validate_checkpoint_v1(
        checkpoint,
        train_plan=train_plan,
        implementation_source_binding=implementation_source_binding,
    )
    return checkpoint


def _model_from_state(state: object, *, device: torch.device) -> DenseSharedSpatialReadoutV1:
    if not isinstance(state, Mapping) or not state:
        raise DenseSharedCalibrationError("model state changed")
    model = DenseSharedSpatialReadoutV1().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    if dense_shared_state_identity_v1(model.state_dict()) != dense_shared_state_identity_v1(state):
        raise DenseSharedCalibrationError("loaded model state changed")
    return model


def _predict_members(
    checkpoint: Mapping[str, object],
    relations: torch.Tensor,
    conditions: torch.Tensor,
    *,
    state_key: str,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    members = checkpoint.get("members")
    if not isinstance(members, list) or len(members) != len(MODEL_SEEDS):
        raise DenseSharedCalibrationError("checkpoint members changed")
    scores = []
    diagnostics = []
    identity_key = {
        "true_state": "true_identity_sha256",
        "current_state": "current_identity_sha256",
    }.get(state_key)
    training_key = {
        "true_state": "true_training",
        "current_state": "current_training",
    }.get(state_key)
    if identity_key is None or training_key is None:
        raise DenseSharedCalibrationError("checkpoint state selector changed")
    for expected_seed, member in zip(MODEL_SEEDS, members, strict=True):
        if not isinstance(member, Mapping) or member.get("seed") != expected_seed:
            raise DenseSharedCalibrationError("checkpoint seed changed")
        selected_state = member.get(state_key)
        if not isinstance(selected_state, Mapping):
            raise DenseSharedCalibrationError("checkpoint member state changed")
        actual_identity = dense_shared_state_identity_v1(selected_state)
        training = member.get(training_key)
        if (
            member.get(identity_key) != actual_identity
            or not isinstance(training, Mapping)
            or training.get("state_identity_sha256") != actual_identity
        ):
            raise DenseSharedCalibrationError("checkpoint member identity changed")
        model = _model_from_state(selected_state, device=device)
        batches = []
        entropies = []
        with torch.no_grad():
            for start in range(0, STATE_COUNT, BATCH_STATES):
                stop = start + BATCH_STATES
                output = model.forward_with_attention(
                    relations[start:stop]
                    .reshape(-1, TOKEN_COUNT, RELATIONAL_DIMENSION)
                    .to(device),
                    conditions[start:stop].reshape(-1, 4).to(device),
                )
                prediction = output.score
                attention = output.attention
                batches.append(prediction.reshape(BATCH_STATES, ACTION_COUNT).cpu())
                entropies.append(
                    (-(attention * torch.log(attention.clamp_min(1.0e-12))).sum(dim=-1))
                    .div(math.log(TOKEN_COUNT))
                    .cpu()
                )
        selected_scores = torch.cat(batches).numpy().astype(np.float64)
        scores.append(selected_scores)
        diagnostics.append(
            {
                "seed": expected_seed,
                "state_identity_sha256": actual_identity,
                "score_shape": [STATE_COUNT, ACTION_COUNT],
                "mean_normalized_attention_entropy": float(torch.cat(entropies).mean()),
                "score_sha256": hashlib.sha256(
                    np.ascontiguousarray(selected_scores.astype("<f8")).tobytes()
                ).hexdigest(),
            }
        )
    stacked = np.stack(scores)
    ensemble = np.mean(stacked, axis=0)
    dispersion = np.std(stacked, axis=0, ddof=0)
    seed_argmins = np.argmin(stacked, axis=2)
    disagreements = np.any(seed_argmins != seed_argmins[:1], axis=0)
    if (
        stacked.shape != (len(MODEL_SEEDS), STATE_COUNT, ACTION_COUNT)
        or not np.isfinite(ensemble).all()
        or not np.isfinite(dispersion).all()
    ):
        raise DenseSharedCalibrationError("ensemble scores are invalid")
    return ensemble, {
        "members": diagnostics,
        "score_stack_shape": [len(MODEL_SEEDS), STATE_COUNT, ACTION_COUNT],
        "ensemble_score_sha256": hashlib.sha256(
            np.ascontiguousarray(ensemble.astype("<f8")).tobytes()
        ).hexdigest(),
        "seed_dispersion": {
            "definition": "population_std_across_three_seed_scores_per_state_action",
            "mean_cell_population_std": float(dispersion.mean()),
            "maximum_cell_population_std": float(dispersion.max()),
            "states_with_seed_argmin_disagreement": int(disagreements.sum()),
            "state_seed_argmin_disagreement_rate": float(disagreements.mean()),
        },
    }


def _score_map(plan: prior.RoleFeaturePlanV1, scores: np.ndarray) -> dict[str, list[float]]:
    if scores.shape != (STATE_COUNT, ACTION_COUNT) or not np.isfinite(scores).all():
        raise DenseSharedCalibrationError("arm scores are invalid")
    return {
        state.state_id: [float(value) for value in scores[index]]
        for index, state in enumerate(plan.states)
    }


def _report_arm(plan: prior.RoleFeaturePlanV1, scores: np.ndarray) -> dict[str, object]:
    return prior._augment_report(selection_metrics_v1(plan.groups, _score_map(plan, scores)))


def _task_readout_report(readouts: ActionSpecificRidgeReadoutsV1) -> dict[str, object]:
    return {
        "identity_sha256": readouts.identity_sha256,
        "heads": [
            {
                "action_id": action,
                "identity_sha256": head.identity_sha256,
                "training_rows": head.training_rows,
                "feature_dimension": int(head.feature_mean.size),
                "ridge_lambda": head.ridge_lambda,
                "solver": head.solver,
            }
            for action, head in enumerate(readouts.heads)
        ],
    }


def _scientific_gates_v1(
    arms: Mapping[str, Any], comparisons: Mapping[str, Any]
) -> dict[str, object]:
    oracle_summary = arms["privileged_physical_oracle"]["summary"]
    true_summary = arms["dense_shared_true_future"]["summary"]
    random_summary = arms["random_expected"]["summary"]
    return {
        "2_privileged_physical_oracle": {
            "passed": oracle_summary["normalized_rank_regret"] == 0.0
            and oracle_summary["oracle_equivalent_selection_rate"] == 1.0,
            "normalized_rank_regret": oracle_summary["normalized_rank_regret"],
            "oracle_equivalent_selection_rate": oracle_summary[
                "oracle_equivalent_selection_rate"
            ],
        },
        "3_true_future_beats_task_action_only": {
            "passed": comparisons["true_future_vs_task_action_only"]["upper_95"]
            < 0.0,
            "measurement": comparisons["true_future_vs_task_action_only"],
        },
        "4_true_future_beats_current_state": {
            "passed": comparisons["true_future_vs_current_state"]["upper_95"]
            < 0.0,
            "measurement": comparisons["true_future_vs_current_state"],
        },
        "5_true_future_beats_relational_persistence": {
            "passed": comparisons["true_future_vs_relational_persistence"][
                "upper_95"
            ]
            < 0.0,
            "measurement": comparisons["true_future_vs_relational_persistence"],
        },
        "6_true_future_beats_random_expected": {
            "passed": true_summary["normalized_rank_regret"]
            < random_summary["normalized_rank_regret"],
            "true_future": true_summary["normalized_rank_regret"],
            "random_expected": random_summary["normalized_rank_regret"],
            "per_family_true_minus_random": {
                family: arms["dense_shared_true_future"]["per_family"][family][
                    "normalized_rank_regret"
                ]
                - arms["random_expected"]["per_family"][family][
                    "normalized_rank_regret"
                ]
                for family in prior.FAMILIES
            },
        },
    }


def evaluate_primary_checkpoint_v1(
    checkpoint: Mapping[str, object],
    train_groups: Sequence[Any],
    eval_groups: Sequence[Any],
    train_features: object,
    eval_features: object,
    device: torch.device,
    *,
    implementation_source_binding: Mapping[str, object],
) -> dict[str, object]:
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    _require_rocm_determinism_v1(device)
    _require_exact_action_catalog_v1()
    plans = prior.build_calibration_feature_plans_v1(train_groups, eval_groups)
    _require_plan_identities_v1(
        train=plans.train,
        eval_plan=plans.eval,
        combined_identity=plans.identity_sha256,
    )
    validate_checkpoint_v1(
        checkpoint,
        train_plan=plans.train,
        implementation_source_binding=implementation_source_binding,
    )
    # Rehashing is a runner responsibility; tensor content is revalidated here.
    _validate_cache(train_features, label="train")
    projected_eval = project_cache_v1(eval_features, checkpoint["pca"], label="eval")
    task = _require_refitted_task_payload_v1(
        checkpoint["task_action_only"], plans.train
    )
    task_scores = score_task_action_only_v1(plans.eval, task)
    true_relations, conditions = _dense_panels(
        plans.eval, projected_eval, successor_mode="true_future"
    )
    current_relations, current_conditions = _dense_panels(
        plans.eval, projected_eval, successor_mode="current_state"
    )
    if not torch.equal(conditions, current_conditions):
        raise DenseSharedCalibrationError("evaluation conditions changed")
    true_residual, true_diagnostics = _predict_members(
        checkpoint,
        true_relations,
        conditions,
        state_key="true_state",
        device=device,
    )
    current_residual, current_diagnostics = _predict_members(
        checkpoint,
        current_relations,
        conditions,
        state_key="current_state",
        device=device,
    )
    persistence_residual, persistence_diagnostics = _predict_members(
        checkpoint,
        current_relations,
        conditions,
        state_key="true_state",
        device=device,
    )
    true_scores = task_scores + true_residual
    current_scores = task_scores + current_residual
    persistence_scores = task_scores + persistence_residual
    oracle_scores = np.stack(
        [np.asarray(state.dense_ranks, dtype=np.float64) for state in plans.eval.states]
    )
    hold_scores = np.ones((STATE_COUNT, ACTION_COUNT), dtype=np.float64)
    hold_scores[:, prior.HOLD_ACTION_ID] = 0.0
    arms = {
        "privileged_physical_oracle": _report_arm(plans.eval, oracle_scores),
        "dense_shared_true_future": _report_arm(plans.eval, true_scores),
        "dense_shared_current_state": _report_arm(plans.eval, current_scores),
        "task_action_only": _report_arm(plans.eval, task_scores),
        "dense_relational_persistence": _report_arm(plans.eval, persistence_scores),
        "hold_constant": _report_arm(plans.eval, hold_scores),
        "random_expected": prior._random_expected_report(plans.eval),
    }
    task_regret = arms["task_action_only"]["summary"]["normalized_rank_regret"]
    if task_regret != EXPECTED_TASK_EVAL_REGRET:
        raise DenseSharedCalibrationError("task/action-only evaluation changed")
    true_rows = arms["dense_shared_true_future"]["group_results"]
    comparisons = {
        name: prior.paired_family_scene_cluster_comparison_v1(
            true_rows, arms[baseline]["group_results"]
        )
        for name, baseline in (
            ("true_future_vs_task_action_only", "task_action_only"),
            ("true_future_vs_current_state", "dense_shared_current_state"),
            ("true_future_vs_relational_persistence", "dense_relational_persistence"),
        )
    }
    gates = _scientific_gates_v1(arms, comparisons)
    member_training = []
    for member in checkpoint["members"]:
        member_training.append(
            {
                "seed": member["seed"],
                "initial_identity_sha256": member["initial_identity_sha256"],
                "true_identity_sha256": member["true_identity_sha256"],
                "current_identity_sha256": member["current_identity_sha256"],
                "true_training": member["true_training"],
                "current_training": member["current_training"],
            }
        )
    result: dict[str, object] = {
        "schema": SCHEMA,
        "status": "COMPLETE_MODEL_INDEPENDENT_EVALUATION",
        "claim_scope": "REUSED_DEVELOPMENT_ROLE_DENSE_ORACLE_FUTURE_CALIBRATION",
        "config": config_v1(),
        "feature_plan": {
            "identity_sha256": plans.identity_sha256,
            "train_identity_sha256": plans.train.identity_sha256,
            "eval_identity_sha256": plans.eval.identity_sha256,
            "states_per_role": STATE_COUNT,
            "artifacts_per_role": prior.ROLE_ARTIFACT_COUNT,
        },
        "checkpoint_identity_sha256": checkpoint["identity_sha256"],
        "pca": {
            "identity_sha256": checkpoint["pca"]["identity_sha256"],
            "dimension": PCA_DIMENSION,
            "row_count": PCA_ROW_COUNT,
            "eigenvalues": [
                float(value) for value in checkpoint["pca"]["eigenvalues"].tolist()
            ],
        },
        "task_action_only_readout": _task_readout_report(task),
        "member_training": member_training,
        "member_diagnostics": {
            "true_future": true_diagnostics,
            "current_state": current_diagnostics,
            "relational_persistence": persistence_diagnostics,
        },
        "score_evidence": {
            name: {
                "shape": [STATE_COUNT, ACTION_COUNT],
                "sha256_float64_c_order": hashlib.sha256(
                    np.ascontiguousarray(scores.astype("<f8")).tobytes()
                ).hexdigest(),
            }
            for name, scores in (
                ("dense_shared_true_future", true_scores),
                ("dense_shared_current_state", current_scores),
                ("task_action_only", task_scores),
                ("dense_relational_persistence", persistence_scores),
                ("privileged_physical_oracle", oracle_scores),
                ("hold_constant", hold_scores),
            )
        },
        "arms": arms,
        "paired_scene_cluster_comparisons": comparisons,
        "safety": prior._safety_support(plans),
        "finiteness": {
            "pca": True,
            "training_diagnostics": True,
            "per_seed_scores": True,
            "ensemble_scores": True,
            "reported_metrics": True,
        },
        "gates": gates,
        "scientific_gates_2_to_6_passed": all(
            bool(gate["passed"]) for gate in gates.values()
        ),
    }
    result["replay_identity_sha256"] = evaluation_identity_v1(result)
    canonical_bytes_v1(result)
    return result


def evaluation_identity_v1(evaluation: Mapping[str, object]) -> str:
    document = dict(evaluation)
    document.pop("replay_identity_sha256", None)
    return hashlib.sha256(canonical_bytes_v1(document)).hexdigest()


def verdict_v1(
    evaluation: Mapping[str, object],
    *,
    infrastructure_checks_passed: bool,
    deterministic_replay_passed: bool,
) -> dict[str, object]:
    gates = evaluation.get("gates")
    required = {
        "schema",
        "status",
        "claim_scope",
        "config",
        "feature_plan",
        "checkpoint_identity_sha256",
        "pca",
        "task_action_only_readout",
        "member_training",
        "member_diagnostics",
        "score_evidence",
        "arms",
        "paired_scene_cluster_comparisons",
        "safety",
        "finiteness",
        "gates",
        "scientific_gates_2_to_6_passed",
        "replay_identity_sha256",
    }
    if (
        set(evaluation) != required
        or evaluation.get("schema") != SCHEMA
        or evaluation.get("status") != "COMPLETE_MODEL_INDEPENDENT_EVALUATION"
        or evaluation.get("claim_scope")
        != "REUSED_DEVELOPMENT_ROLE_DENSE_ORACLE_FUTURE_CALIBRATION"
        or evaluation.get("config") != config_v1()
        or evaluation.get("replay_identity_sha256")
        != evaluation_identity_v1(evaluation)
        or not isinstance(gates, Mapping)
        or set(gates) != SCIENTIFIC_GATE_NAMES
        or type(infrastructure_checks_passed) is not bool
        or type(deterministic_replay_passed) is not bool
        or any(
            not isinstance(gate, Mapping) or type(gate.get("passed")) is not bool
            for gate in gates.values()
        )
    ):
        raise DenseSharedCalibrationError("verdict inputs changed")
    arms = evaluation.get("arms")
    comparisons = evaluation.get("paired_scene_cluster_comparisons")
    if (
        not isinstance(arms, Mapping)
        or not isinstance(comparisons, Mapping)
        or dict(gates) != _scientific_gates_v1(arms, comparisons)
    ):
        raise DenseSharedCalibrationError("scientific gates no longer match measurements")
    all_scientific = all(gate["passed"] for gate in gates.values())
    if evaluation.get("scientific_gates_2_to_6_passed") is not all_scientific:
        raise DenseSharedCalibrationError("scientific gate aggregate changed")
    all_passed = infrastructure_checks_passed and deterministic_replay_passed and all_scientific
    status = (
        INFRASTRUCTURE_FAILURE_STATUS
        if not infrastructure_checks_passed or not deterministic_replay_passed
        else PASS_STATUS if all_passed else STOP_STATUS
    )
    return {
        "gates": {
            "1_infrastructure_and_custody": {"passed": infrastructure_checks_passed},
            **dict(gates),
            "7_deterministic_replay": {"passed": deterministic_replay_passed},
        },
        "passed": all_passed,
        "terminal_status": status,
    }


__all__ = (
    "CHECKPOINT_SCHEMA",
    "CONFIG_SCHEMA",
    "DenseSharedCalibrationError",
    "EXPECTED_COMBINED_PLAN_IDENTITY",
    "EXPECTED_EVAL_PLAN_IDENTITY",
    "EXPECTED_TASK_EVAL_REGRET",
    "EXPECTED_TASK_IDENTITY",
    "EXPECTED_TRAIN_PLAN_IDENTITY",
    "INFRASTRUCTURE_FAILURE_STATUS",
    "PASS_STATUS",
    "SCHEMA",
    "STOP_STATUS",
    "canonical_bytes_v1",
    "config_v1",
    "evaluate_primary_checkpoint_v1",
    "evaluation_identity_v1",
    "fit_primary_checkpoint_v1",
    "fit_task_action_only_v1",
    "fit_train_pca_v1",
    "patch_coordinates_v1",
    "pca_identity_v1",
    "project_cache_v1",
    "score_task_action_only_v1",
    "validate_checkpoint_v1",
    "verdict_v1",
)
