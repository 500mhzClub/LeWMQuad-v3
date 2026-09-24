"""Pure evaluator for the dense V-JEPA 2.1 physical-interface ceiling V1.

This module has no filesystem, RGB, encoder, or lifecycle access.  It consumes
strict-loader group objects, already validated CPU token caches, and the bound
published physical-predecessor evaluation.  The unchanged 245-parameter dense
shared readout is imported from the completed DINO calibration.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from lewm.benchmarks import (
    go2_dinov2_dense_shared_spatial_readout_calibration_v1 as dense,
)
from lewm.benchmarks import go2_dinov2_physical_readout_calibration_v1 as prior
from lewm.benchmarks import go2_matched_branch_physical_outcome_screen_v1 as physical
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
    selection_metrics_v1,
)


SCHEMA = "lewm_go2_dense_vjepa2_1_physical_interface_ceiling_v1"
CONFIG_SCHEMA = "lewm_go2_dense_vjepa2_1_physical_interface_ceiling_config_v1"
CHECKPOINT_SCHEMA = "lewm_go2_dense_vjepa2_1_physical_interface_ceiling_checkpoint_v1"
PCA_SCHEMA = "lewm_go2_dense_vjepa2_1_train_pca_v1"
ACTION_MEAN_SCHEMA = "lewm_go2_dense_vjepa2_1_train_action_mean_innovation_v1"

PASS_STATUS = (
    "QUALIFY_VJEPA_DENSE_INTERFACE_FOR_SEPARATE_BACKBONE_LEVEL_"
    "MATCHED_BRANCH_JEPA_PREREGISTRATION"
)
STOP_STATUS = "STOP_FROZEN_VJEPA_PHYSICAL_INTERFACE_NOT_ESTABLISHED"
INFRASTRUCTURE_FAILURE_STATUS = "FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION"

ACTION_COUNT = 9
STATE_COUNT = 128
SCENE_COUNT = 16
ROLE_ARTIFACT_COUNT = 1_536
TOKEN_COUNT = 256
TOKEN_DIMENSION = 768
PCA_DIMENSION = 8
RELATIONAL_DIMENSION = 3 * PCA_DIMENSION
PCA_ROW_COUNT = STATE_COUNT * (1 + ACTION_COUNT) * TOKEN_COUNT
PCA_EPSILON = 1.0e-12
TOKEN_NORM_TOLERANCE = 2.0e-3
MODEL_SEEDS = (2_026_080_303, 2_026_080_304, 2_026_080_305)
EPOCHS = 256
BATCH_STATES = 16
STEPS_PER_EPOCH = STATE_COUNT // BATCH_STATES
OPTIMIZER_STEPS = EPOCHS * STEPS_PER_EPOCH
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2_026_080_314
EXPECTED_TASK_IDENTITY = dense.EXPECTED_TASK_IDENTITY
EXPECTED_TASK_EVAL_REGRET = dense.EXPECTED_TASK_EVAL_REGRET
EXPECTED_TRAIN_PLAN_IDENTITY = dense.EXPECTED_TRAIN_PLAN_IDENTITY
EXPECTED_EVAL_PLAN_IDENTITY = dense.EXPECTED_EVAL_PLAN_IDENTITY
EXPECTED_COMBINED_PLAN_IDENTITY = dense.EXPECTED_COMBINED_PLAN_IDENTITY
EXPECTED_RETAINED_PHYSICAL_REGRET = 0.14896763392857143
EXPECTED_PHYSICAL_EVALUATION_IDENTITY = (
    "5e19a2547187f1101a4a19ee6ffd9d8892f38efc1f5c52842430d860430091cc"
)

TRUE_ARM = "dense_vjepa_true_future"
CURRENT_ARM = "dense_vjepa_current_state"
PERSISTENCE_ARM = "dense_vjepa_relational_persistence"
WRONG_SCENE_ARM = "dense_vjepa_same_action_wrong_scene"
ACTION_MEAN_ARM = "dense_vjepa_train_action_mean_innovation"
RETAINED_ARM = "retained_physical_predecessor"

COMPARISON_BASELINES = {
    "true_future_vs_task_action_only": "task_action_only",
    "true_future_vs_retained_physical_predecessor": RETAINED_ARM,
    "true_future_vs_current_state": CURRENT_ARM,
    "true_future_vs_relational_persistence": PERSISTENCE_ARM,
    "true_future_vs_same_action_wrong_scene": WRONG_SCENE_ARM,
    "true_future_vs_train_action_mean_innovation": ACTION_MEAN_ARM,
}
COMPARISON_NAMES = frozenset(COMPARISON_BASELINES)

SCIENTIFIC_GATE_NAMES = frozenset(
    {
        "2_privileged_physical_oracle",
        "3_true_future_beats_task_action_only",
        "4_true_future_beats_retained_physical_predecessor",
        "5_true_future_beats_current_state",
        "6_true_future_beats_relational_persistence",
        "7_true_future_beats_same_action_wrong_scene",
        "8_true_future_beats_train_action_mean_innovation",
        "9_true_future_beats_random_expected",
    }
)


class DenseVJEPAInterfaceError(RuntimeError):
    """Raised when a frozen scientific or deterministic contract changes."""


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
        raise DenseVJEPAInterfaceError("implementation source binding changed")
    return dict(value)


def _require_frozen_shared_contract_v1() -> None:
    commands = tuple(
        tuple(float(item) for item in command) for command in CANONICAL_ACTION_COMMANDS
    )
    if (
        commands != dense.EXPECTED_ACTION_COMMANDS
        or PARAMETER_COUNT != 245
        or dense.PCA_DIMENSION != PCA_DIMENSION
        or dense.RELATIONAL_DIMENSION != RELATIONAL_DIMENSION
        or dense.MODEL_SEEDS != MODEL_SEEDS
        or dense.EPOCHS != EPOCHS
        or dense.BATCH_STATES != BATCH_STATES
        or dense.OPTIMIZER_STEPS != OPTIMIZER_STEPS
    ):
        raise DenseVJEPAInterfaceError("shared readout or fitting contract changed")


def _require_plans_v1(
    train: prior.RoleFeaturePlanV1,
    eval_plan: prior.RoleFeaturePlanV1 | None = None,
    combined_identity: str | None = None,
) -> None:
    if train.identity_sha256 != EXPECTED_TRAIN_PLAN_IDENTITY:
        raise DenseVJEPAInterfaceError("train feature-plan identity changed")
    if eval_plan is not None and eval_plan.identity_sha256 != EXPECTED_EVAL_PLAN_IDENTITY:
        raise DenseVJEPAInterfaceError("evaluation feature-plan identity changed")
    if combined_identity is not None and combined_identity != EXPECTED_COMBINED_PLAN_IDENTITY:
        raise DenseVJEPAInterfaceError("combined feature-plan identity changed")


def _require_rocm_determinism_v1(device: torch.device) -> None:
    if (
        device.type != "cuda"
        or not torch.cuda.is_available()
        or torch.version.hip is None
        or not torch.are_deterministic_algorithms_enabled()
        or torch.get_float32_matmul_precision() != "highest"
    ):
        raise DenseVJEPAInterfaceError("authorized ROCm deterministic runtime changed")


def config_v1() -> dict[str, object]:
    _require_frozen_shared_contract_v1()
    return {
        "schema": CONFIG_SCHEMA,
        "role_contract": {
            "states_per_role": STATE_COUNT,
            "scenes_per_role": SCENE_COUNT,
            "actions": ACTION_COUNT,
            "artifacts_per_role": ROLE_ARTIFACT_COUNT,
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
            "covariance": "population",
            "eigensolver": "numpy.linalg.eigh",
            "ordering": "descending_eigenvalue_then_original_ascending_index",
            "sign": "largest_absolute_loading_smallest_channel_positive",
            "whitening_epsilon": PCA_EPSILON,
            "clipping": False,
        },
        "readout": {
            "model": "unchanged_DenseSharedSpatialReadoutV1",
            "parameter_count_per_member": PARAMETER_COUNT,
            "true_ensemble_parameters": PARAMETER_COUNT * len(MODEL_SEEDS),
            "current_ensemble_parameters": PARAMETER_COUNT * len(MODEL_SEEDS),
            "condition": ["goal_x_div_10", "goal_y_div_10", "vx_div_0.30", "wz_div_0.45"],
            "relational": ["current", "successor", "successor_minus_current"],
        },
        "training": {
            "seeds": list(MODEL_SEEDS),
            "epochs": EPOCHS,
            "batch_states": BATCH_STATES,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "optimizer_steps": OPTIMIZER_STEPS,
            "optimizer": "AdamW",
            "learning_rate": dense.LEARNING_RATE,
            "weight_decay": dense.WEIGHT_DECAY,
            "betas": [0.9, 0.999],
            "epsilon": 1.0e-8,
            "amsgrad": False,
            "foreach": False,
            "fused": False,
            "gradient_clip_norm": dense.GRADIENT_CLIP_NORM,
            "complete_state_minibatches": True,
            "deterministic_algorithms": True,
            "float32_matmul_precision": "highest",
            "device": "ROCm",
        },
        "task_action_only": {
            "identity_sha256": EXPECTED_TASK_IDENTITY,
            "required_eval_regret": EXPECTED_TASK_EVAL_REGRET,
            "ridge_lambda": dense.TASK_RIDGE_LAMBDA,
        },
        "retained_physical_predecessor": {
            "evaluation_identity_sha256": EXPECTED_PHYSICAL_EVALUATION_IDENTITY,
            "required_eval_regret": EXPECTED_RETAINED_PHYSICAL_REGRET,
            "checkpoint_loaded": False,
        },
        "wrong_scene": {
            "pairing": "other_lexicographic_scene_same_family_same_role_plan_ordinal",
            "states_per_scene": 8,
            "same_action": True,
        },
        "train_action_mean_innovation": {
            "source": "train_only_projected_successor_minus_current",
            "mean_axes": "128_states_per_action_patch",
            "accumulator_dtype": "float64",
            "scorer_dtype": "float32",
        },
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "unit": "whole_scene",
            "family_weighting": "equal_eight_families",
            "interval": "percentile_95",
        },
    }


def _validate_cache_v1(features: object, *, label: str) -> torch.Tensor:
    if not isinstance(features, torch.Tensor):
        raise DenseVJEPAInterfaceError(f"{label} cache must be a tensor")
    if tuple(features.shape) != (ROLE_ARTIFACT_COUNT, TOKEN_COUNT, TOKEN_DIMENSION):
        raise DenseVJEPAInterfaceError(f"{label} cache shape changed")
    if features.dtype != torch.float16 or features.device.type != "cpu":
        raise DenseVJEPAInterfaceError(f"{label} cache must be CPU float16")
    if not bool(torch.isfinite(features).all()):
        raise DenseVJEPAInterfaceError(f"{label} cache contains nonfinite tokens")
    norms = torch.linalg.vector_norm(features.float(), dim=-1)
    maximum_error = float((norms - 1.0).abs().max())
    if not math.isfinite(maximum_error) or maximum_error > TOKEN_NORM_TOLERANCE:
        raise DenseVJEPAInterfaceError(
            f"{label} per-token normalization changed: {maximum_error}"
        )
    return features


def _pca_source_indices_v1(plan: prior.RoleFeaturePlanV1) -> tuple[int, ...]:
    if plan.role != "train" or len(plan.states) != STATE_COUNT:
        raise DenseVJEPAInterfaceError("PCA requires the exact train role")
    current = tuple(state.context_artifact_indices[-1] for state in plan.states)
    successors = tuple(
        state.target_artifact_indices[action]
        for state in plan.states
        for action in range(ACTION_COUNT)
    )
    result = current + successors
    if (
        len(result) != STATE_COUNT * (1 + ACTION_COUNT)
        or len(set(result)) != len(result)
    ):
        raise DenseVJEPAInterfaceError("PCA source order changed")
    return result


def _array_digest_update_v1(digest: Any, array: np.ndarray) -> None:
    canonical = np.ascontiguousarray(array.astype("<f8", copy=False))
    digest.update(str(canonical.shape).encode("ascii") + b"\0")
    digest.update(canonical.tobytes())


def fit_train_pca_v1(
    train_plan: prior.RoleFeaturePlanV1,
    train_features: object,
    *,
    implementation_source_binding: Mapping[str, object],
) -> dict[str, object]:
    features = _validate_cache_v1(train_features, label="train")
    indices = _pca_source_indices_v1(train_plan)
    source = features[list(indices)].reshape(PCA_ROW_COUNT, TOKEN_DIMENSION).numpy()
    matrix = source.astype(np.float64)
    mean = matrix.mean(axis=0, dtype=np.float64)
    matrix -= mean
    covariance = (matrix.T @ matrix) / float(PCA_ROW_COUNT)
    if covariance.shape != (TOKEN_DIMENSION, TOKEN_DIMENSION) or not np.isfinite(covariance).all():
        raise DenseVJEPAInterfaceError("PCA covariance is invalid")
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
        or not all(np.isfinite(value).all() for value in (mean, selected_values, components, scales))
    ):
        raise DenseVJEPAInterfaceError("PCA eigensystem is invalid")
    result: dict[str, object] = {
        "schema": PCA_SCHEMA,
        "source": {
            "train_plan_identity": train_plan.identity_sha256,
            "artifact_indices": list(indices),
            "artifact_ids": [train_plan.artifact_ids[index] for index in indices],
            "patch_order": "row_major_16x16",
            "row_count": PCA_ROW_COUNT,
        },
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
    _validate_pca_v1(result, expected_train_plan=train_plan)
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
            raise DenseVJEPAInterfaceError(f"PCA {name} changed")
        array = value.detach().cpu().numpy()
        if array.shape != shape or not np.isfinite(array).all():
            raise DenseVJEPAInterfaceError(f"PCA {name} is invalid")
        arrays.append(array)
    return arrays[0], arrays[1], arrays[2], arrays[3]


def pca_identity_v1(pca: Mapping[str, object]) -> str:
    required = {
        "schema", "source", "implementation_source_binding", "epsilon",
        "mean", "eigenvalues", "components", "scales",
    }
    if set(pca) not in (required, required | {"identity_sha256"}):
        raise DenseVJEPAInterfaceError("PCA payload inventory changed")
    if pca.get("schema") != PCA_SCHEMA or pca.get("epsilon") != PCA_EPSILON:
        raise DenseVJEPAInterfaceError("PCA contract changed")
    source = pca.get("source")
    if (
        not isinstance(source, Mapping)
        or set(source) != {
            "train_plan_identity", "artifact_indices", "artifact_ids",
            "patch_order", "row_count",
        }
        or source.get("patch_order") != "row_major_16x16"
        or source.get("row_count") != PCA_ROW_COUNT
        or not isinstance(source.get("artifact_indices"), list)
        or not isinstance(source.get("artifact_ids"), list)
        or len(source["artifact_indices"]) != STATE_COUNT * (1 + ACTION_COUNT)
        or len(source["artifact_ids"]) != STATE_COUNT * (1 + ACTION_COUNT)
    ):
        raise DenseVJEPAInterfaceError("PCA source contract changed")
    implementation = _implementation_binding_v1(pca.get("implementation_source_binding"))
    mean, eigenvalues, components, scales = _pca_arrays_v1(pca)
    if np.any(eigenvalues[:-1] < eigenvalues[1:]):
        raise DenseVJEPAInterfaceError("PCA eigenvalue order changed")
    if not np.array_equal(scales, np.sqrt(np.maximum(eigenvalues, PCA_EPSILON))):
        raise DenseVJEPAInterfaceError("PCA whitening scales changed")
    for column in range(PCA_DIMENSION):
        absolute = np.abs(components[:, column])
        pivot = int(np.flatnonzero(absolute == absolute.max())[0])
        if components[pivot, column] < 0.0:
            raise DenseVJEPAInterfaceError("PCA component sign changed")
    digest = hashlib.sha256(
        canonical_bytes_v1(
            {
                "schema": PCA_SCHEMA,
                "source": dict(source),
                "implementation_source_binding": implementation,
                "epsilon": PCA_EPSILON,
            }
        )
    )
    for array in (mean, eigenvalues, components, scales):
        _array_digest_update_v1(digest, array)
    return digest.hexdigest()


def _validate_pca_v1(
    pca: Mapping[str, object],
    *,
    expected_train_plan: prior.RoleFeaturePlanV1 | None = None,
    expected_implementation_source_binding: Mapping[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    identity = pca_identity_v1(pca)
    if pca.get("identity_sha256") != identity:
        raise DenseVJEPAInterfaceError("PCA identity changed")
    source = pca["source"]
    assert isinstance(source, Mapping)
    if expected_train_plan is not None:
        indices = _pca_source_indices_v1(expected_train_plan)
        if (
            source.get("train_plan_identity") != expected_train_plan.identity_sha256
            or source.get("artifact_indices") != list(indices)
            or source.get("artifact_ids")
            != [expected_train_plan.artifact_ids[index] for index in indices]
        ):
            raise DenseVJEPAInterfaceError("PCA train source changed")
    if (
        expected_implementation_source_binding is not None
        and pca.get("implementation_source_binding")
        != _implementation_binding_v1(expected_implementation_source_binding)
    ):
        raise DenseVJEPAInterfaceError("PCA implementation source changed")
    mean, _eigenvalues, components, scales = _pca_arrays_v1(pca)
    return mean, components, scales


def project_cache_v1(features: object, pca: Mapping[str, object], *, label: str) -> torch.Tensor:
    cache = _validate_cache_v1(features, label=label)
    mean, components, scales = _validate_pca_v1(pca)
    output = np.empty(
        (ROLE_ARTIFACT_COUNT, TOKEN_COUNT, PCA_DIMENSION), dtype=np.float32
    )
    for start in range(0, ROLE_ARTIFACT_COUNT, 32):
        stop = min(start + 32, ROLE_ARTIFACT_COUNT)
        selected = cache[start:stop].numpy().astype(np.float64)
        projected = ((selected - mean) @ components) / scales
        if not np.isfinite(projected).all():
            raise DenseVJEPAInterfaceError(f"{label} PCA projection is nonfinite")
        output[start:stop] = projected.astype(np.float32)
    return torch.from_numpy(output)


def train_action_mean_innovation_v1(
    train_plan: prior.RoleFeaturePlanV1, projected_train: torch.Tensor
) -> dict[str, object]:
    if (
        not isinstance(projected_train, torch.Tensor)
        or tuple(projected_train.shape)
        != (ROLE_ARTIFACT_COUNT, TOKEN_COUNT, PCA_DIMENSION)
        or projected_train.dtype != torch.float32
        or projected_train.device.type != "cpu"
        or not bool(torch.isfinite(projected_train).all())
    ):
        raise DenseVJEPAInterfaceError("projected train cache contract changed")
    rows = np.empty(
        (ACTION_COUNT, STATE_COUNT, TOKEN_COUNT, PCA_DIMENSION), dtype=np.float64
    )
    values = projected_train.numpy()
    for state_index, state in enumerate(train_plan.states):
        current = values[state.context_artifact_indices[-1]].astype(np.float64)
        for action in range(ACTION_COUNT):
            successor = values[state.target_artifact_indices[action]].astype(np.float64)
            rows[action, state_index] = successor - current
    means = np.ascontiguousarray(rows.mean(axis=1, dtype=np.float64).astype(np.float32))
    if means.shape != (ACTION_COUNT, TOKEN_COUNT, PCA_DIMENSION) or not np.isfinite(means).all():
        raise DenseVJEPAInterfaceError("train action-mean innovation is invalid")
    tensor = torch.from_numpy(means)
    metadata = {
        "schema": ACTION_MEAN_SCHEMA,
        "train_plan_identity": train_plan.identity_sha256,
        "states_per_action_patch": STATE_COUNT,
        "source_order": "action_state_role_plan_patch_component",
        "accumulator_dtype": "float64",
        "storage_dtype": "float32",
    }
    digest = hashlib.sha256(canonical_bytes_v1(metadata))
    digest.update(means.astype("<f4", copy=False).tobytes(order="C"))
    return {**metadata, "values": tensor, "identity_sha256": digest.hexdigest()}


def validate_action_mean_innovation_v1(value: object) -> torch.Tensor:
    if not isinstance(value, Mapping) or set(value) != {
        "schema", "train_plan_identity", "states_per_action_patch", "source_order",
        "accumulator_dtype", "storage_dtype", "values", "identity_sha256",
    }:
        raise DenseVJEPAInterfaceError("train action-mean payload changed")
    tensor = value.get("values")
    if (
        value.get("schema") != ACTION_MEAN_SCHEMA
        or value.get("train_plan_identity") != EXPECTED_TRAIN_PLAN_IDENTITY
        or value.get("states_per_action_patch") != STATE_COUNT
        or value.get("source_order") != "action_state_role_plan_patch_component"
        or value.get("accumulator_dtype") != "float64"
        or value.get("storage_dtype") != "float32"
        or not isinstance(tensor, torch.Tensor)
        or tensor.dtype != torch.float32
        or tensor.device.type != "cpu"
        or tuple(tensor.shape) != (ACTION_COUNT, TOKEN_COUNT, PCA_DIMENSION)
        or not bool(torch.isfinite(tensor).all())
    ):
        raise DenseVJEPAInterfaceError("train action-mean payload is invalid")
    metadata = {key: value[key] for key in value if key not in {"values", "identity_sha256"}}
    digest = hashlib.sha256(canonical_bytes_v1(metadata))
    digest.update(tensor.numpy().astype("<f4", copy=False).tobytes(order="C"))
    if value.get("identity_sha256") != digest.hexdigest():
        raise DenseVJEPAInterfaceError("train action-mean identity changed")
    return tensor


def same_action_wrong_scene_donors_v1(
    eval_plan: prior.RoleFeaturePlanV1,
) -> tuple[tuple[int, ...], dict[str, object]]:
    if eval_plan.role != "eval" or len(eval_plan.states) != STATE_COUNT:
        raise DenseVJEPAInterfaceError("wrong-scene control requires exact eval role")
    by_family_scene: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, state in enumerate(eval_plan.states):
        by_family_scene[state.family][state.scene_id].append(index)
    donors = [-1] * STATE_COUNT
    pairs = []
    for family in sorted(by_family_scene):
        scenes = sorted(by_family_scene[family])
        if len(scenes) != 2:
            raise DenseVJEPAInterfaceError("wrong-scene family does not have two scenes")
        left = by_family_scene[family][scenes[0]]
        right = by_family_scene[family][scenes[1]]
        if len(left) != 8 or len(right) != 8:
            raise DenseVJEPAInterfaceError("wrong-scene scene does not have eight states")
        # Lists inherit the fixed evaluation role-plan order.  Ordinals are
        # positions in those lists, not mutable state_index_in_scene metadata.
        for ordinal, (left_index, right_index) in enumerate(zip(left, right, strict=True)):
            donors[left_index] = right_index
            donors[right_index] = left_index
            pairs.append(
                {
                    "family": family,
                    "ordinal": ordinal,
                    "left_state_id": eval_plan.states[left_index].state_id,
                    "right_state_id": eval_plan.states[right_index].state_id,
                }
            )
    if sorted(donors) != list(range(STATE_COUNT)) or any(
        donors[donors[index]] != index or donors[index] == index
        for index in range(STATE_COUNT)
    ):
        raise DenseVJEPAInterfaceError("wrong-scene donor mapping is not a unique swap")
    document = {
        "schema": "lewm_go2_dense_vjepa2_1_same_action_wrong_scene_mapping_v1",
        "eval_plan_identity": eval_plan.identity_sha256,
        "definition": "other_scene_same_family_same_role_plan_ordinal_same_action",
        "donor_state_indices": donors,
        "scene_pairs": pairs,
    }
    document["identity_sha256"] = hashlib.sha256(canonical_bytes_v1(document)).hexdigest()
    return tuple(donors), document


def relational_panels_v1(
    plan: prior.RoleFeaturePlanV1,
    projected: torch.Tensor,
    *,
    mode: str,
    wrong_scene_donors: Sequence[int] | None = None,
    train_action_means: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tuple(projected.shape) != (ROLE_ARTIFACT_COUNT, TOKEN_COUNT, PCA_DIMENSION):
        raise DenseVJEPAInterfaceError("projected cache shape changed")
    if mode == "wrong_scene" and (
        wrong_scene_donors is None or len(wrong_scene_donors) != STATE_COUNT
    ):
        raise DenseVJEPAInterfaceError("wrong-scene donor map changed")
    if mode == "train_action_mean" and (
        not isinstance(train_action_means, torch.Tensor)
        or tuple(train_action_means.shape) != (ACTION_COUNT, TOKEN_COUNT, PCA_DIMENSION)
        or train_action_means.dtype != torch.float32
    ):
        raise DenseVJEPAInterfaceError("train action means changed")
    relations = torch.empty(
        (STATE_COUNT, ACTION_COUNT, TOKEN_COUNT, RELATIONAL_DIMENSION),
        dtype=torch.float32,
    )
    conditions = torch.empty((STATE_COUNT, ACTION_COUNT, 4), dtype=torch.float32)
    for state_index, state in enumerate(plan.states):
        current = projected[state.context_artifact_indices[-1]]
        goal_x, goal_y = state.relative_target_xy_body_m
        for action in range(ACTION_COUNT):
            if mode == "true_future":
                successor = projected[state.target_artifact_indices[action]]
            elif mode == "current_state":
                successor = current
            elif mode == "wrong_scene":
                assert wrong_scene_donors is not None
                donor = plan.states[int(wrong_scene_donors[state_index])]
                if donor.family != state.family or donor.scene_id == state.scene_id:
                    raise DenseVJEPAInterfaceError("wrong-scene donor role changed")
                successor = projected[donor.target_artifact_indices[action]]
            elif mode == "train_action_mean":
                assert train_action_means is not None
                successor = current + train_action_means[action]
            else:
                raise DenseVJEPAInterfaceError("relational panel mode changed")
            relations[state_index, action] = torch.cat(
                (current, successor, successor - current), dim=-1
            )
            command = CANONICAL_ACTION_COMMANDS[action]
            conditions[state_index, action] = torch.tensor(
                (goal_x / 10.0, goal_y / 10.0, command[0] / 0.30, command[2] / 0.45),
                dtype=torch.float32,
            )
    if not bool(torch.isfinite(relations).all()) or not bool(torch.isfinite(conditions).all()):
        raise DenseVJEPAInterfaceError("relational panels are nonfinite")
    return relations, conditions


def _clone_state_v1(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _checkpoint_identity_v1(checkpoint: Mapping[str, object]) -> str:
    required = {
        "schema", "config", "train_plan_identity", "pca",
        "train_action_mean_innovation", "task_action_only", "members",
    }
    if set(checkpoint) not in (required, required | {"identity_sha256"}):
        raise DenseVJEPAInterfaceError("checkpoint inventory changed")
    members = checkpoint.get("members")
    if not isinstance(members, list):
        raise DenseVJEPAInterfaceError("checkpoint members changed")
    document = {
        "schema": checkpoint.get("schema"),
        "config": checkpoint.get("config"),
        "train_plan_identity": checkpoint.get("train_plan_identity"),
        "pca_identity_sha256": (
            checkpoint.get("pca", {}).get("identity_sha256")
            if isinstance(checkpoint.get("pca"), Mapping) else None
        ),
        "train_action_mean_identity_sha256": (
            checkpoint.get("train_action_mean_innovation", {}).get("identity_sha256")
            if isinstance(checkpoint.get("train_action_mean_innovation"), Mapping) else None
        ),
        "task_action_only_identity_sha256": (
            checkpoint.get("task_action_only", {}).get("identity_sha256")
            if isinstance(checkpoint.get("task_action_only"), Mapping) else None
        ),
        "members": [
            {
                "seed": member.get("seed"),
                "initial_identity_sha256": member.get("initial_identity_sha256"),
                "true_identity_sha256": member.get("true_identity_sha256"),
                "current_identity_sha256": member.get("current_identity_sha256"),
                "true_training": member.get("true_training"),
                "current_training": member.get("current_training"),
            }
            if isinstance(member, Mapping) else None
            for member in members
        ],
    }
    return hashlib.sha256(canonical_bytes_v1(document)).hexdigest()


def validate_checkpoint_v1(
    checkpoint: Mapping[str, object],
    *,
    train_plan: prior.RoleFeaturePlanV1,
    implementation_source_binding: Mapping[str, object],
) -> None:
    _require_plans_v1(train_plan)
    if (
        checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("config") != config_v1()
        or checkpoint.get("train_plan_identity") != EXPECTED_TRAIN_PLAN_IDENTITY
    ):
        raise DenseVJEPAInterfaceError("checkpoint contract changed")
    pca = checkpoint.get("pca")
    if not isinstance(pca, Mapping):
        raise DenseVJEPAInterfaceError("checkpoint PCA changed")
    _validate_pca_v1(
        pca,
        expected_train_plan=train_plan,
        expected_implementation_source_binding=implementation_source_binding,
    )
    validate_action_mean_innovation_v1(checkpoint.get("train_action_mean_innovation"))
    task = checkpoint.get("task_action_only")
    if not isinstance(task, Mapping):
        raise DenseVJEPAInterfaceError("checkpoint task readout changed")
    dense._require_refitted_task_payload_v1(task, train_plan)  # noqa: SLF001
    members = checkpoint.get("members")
    if not isinstance(members, list) or len(members) != len(MODEL_SEEDS):
        raise DenseVJEPAInterfaceError("checkpoint member count changed")
    for seed, member in zip(MODEL_SEEDS, members, strict=True):
        if not isinstance(member, Mapping) or member.get("seed") != seed:
            raise DenseVJEPAInterfaceError("checkpoint member seed changed")
        expected_initial = dense_shared_state_identity_v1(
            initialize_dense_shared_spatial_readout_v1(seed)
        )
        if member.get("initial_identity_sha256") != expected_initial:
            raise DenseVJEPAInterfaceError("checkpoint initial state changed")
        for prefix in ("true", "current"):
            state = member.get(f"{prefix}_state")
            if not isinstance(state, Mapping):
                raise DenseVJEPAInterfaceError("checkpoint model state changed")
            identity = dense_shared_state_identity_v1(state)
            if member.get(f"{prefix}_identity_sha256") != identity:
                raise DenseVJEPAInterfaceError("checkpoint model identity changed")
            dense._validated_training_report_v1(  # noqa: SLF001
                member.get(f"{prefix}_training"),
                expected_state_identity=identity,
            )
    if checkpoint.get("identity_sha256") != _checkpoint_identity_v1(checkpoint):
        raise DenseVJEPAInterfaceError("checkpoint identity changed")


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
    _require_frozen_shared_contract_v1()
    train_plan = prior.build_role_feature_plan_v1(train_groups, role="train")
    _require_plans_v1(train_plan)
    pca = fit_train_pca_v1(
        train_plan,
        train_features,
        implementation_source_binding=implementation_source_binding,
    )
    projected = project_cache_v1(train_features, pca, label="train")
    action_mean = train_action_mean_innovation_v1(train_plan, projected)
    task = dense.fit_task_action_only_v1(train_plan)
    task_scores = dense.score_task_action_only_v1(train_plan, task)
    residual_targets = torch.from_numpy(
        (dense._normalized_rank_targets(train_plan) - task_scores).astype(np.float32)  # noqa: SLF001
    )
    true_relations, conditions = relational_panels_v1(
        train_plan, projected, mode="true_future"
    )
    current_relations, current_conditions = relational_panels_v1(
        train_plan, projected, mode="current_state"
    )
    if not torch.equal(conditions, current_conditions):
        raise DenseVJEPAInterfaceError("true/current conditions changed")
    members = []
    for seed in MODEL_SEEDS:
        initialized = initialize_dense_shared_spatial_readout_v1(seed)
        initial_state = _clone_state_v1(initialized.state_dict())
        generator = torch.Generator(device="cpu").manual_seed(seed)
        orders = tuple(
            torch.randperm(STATE_COUNT, generator=generator) for _ in range(EPOCHS)
        )
        true_state, true_training = dense._train_one_model(  # noqa: SLF001
            initial_state, true_relations, conditions, residual_targets, orders,
            device=device,
        )
        current_state, current_training = dense._train_one_model(  # noqa: SLF001
            initial_state, current_relations, conditions, residual_targets, orders,
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
        "train_action_mean_innovation": action_mean,
        "task_action_only": dense._task_payload(task),  # noqa: SLF001
        "members": members,
    }
    checkpoint["identity_sha256"] = _checkpoint_identity_v1(checkpoint)
    validate_checkpoint_v1(
        checkpoint,
        train_plan=train_plan,
        implementation_source_binding=implementation_source_binding,
    )
    return checkpoint


def _predict_members_v1(
    checkpoint: Mapping[str, object],
    relations: torch.Tensor,
    conditions: torch.Tensor,
    *,
    state_key: str,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    identity_key = {"true_state": "true_identity_sha256", "current_state": "current_identity_sha256"}.get(state_key)
    training_key = {"true_state": "true_training", "current_state": "current_training"}.get(state_key)
    if identity_key is None or training_key is None:
        raise DenseVJEPAInterfaceError("model state selector changed")
    score_rows = []
    diagnostics = []
    for seed, member in zip(MODEL_SEEDS, checkpoint["members"], strict=True):
        if not isinstance(member, Mapping) or member.get("seed") != seed:
            raise DenseVJEPAInterfaceError("prediction member changed")
        state = member.get(state_key)
        if not isinstance(state, Mapping):
            raise DenseVJEPAInterfaceError("prediction state changed")
        identity = dense_shared_state_identity_v1(state)
        if member.get(identity_key) != identity:
            raise DenseVJEPAInterfaceError("prediction state identity changed")
        model = DenseSharedSpatialReadoutV1().to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        batches = []
        entropies = []
        with torch.no_grad():
            for start in range(0, STATE_COUNT, BATCH_STATES):
                stop = start + BATCH_STATES
                output = model.forward_with_attention(
                    relations[start:stop].reshape(-1, TOKEN_COUNT, RELATIONAL_DIMENSION).to(device),
                    conditions[start:stop].reshape(-1, 4).to(device),
                )
                batches.append(output.score.reshape(BATCH_STATES, ACTION_COUNT).cpu())
                entropies.append(
                    (-(output.attention * torch.log(output.attention.clamp_min(1.0e-12))).sum(dim=-1))
                    .div(math.log(TOKEN_COUNT)).cpu()
                )
        scores = torch.cat(batches).numpy().astype(np.float64)
        score_rows.append(scores)
        diagnostics.append(
            {
                "seed": seed,
                "state_identity_sha256": identity,
                "score_sha256": hashlib.sha256(
                    np.ascontiguousarray(scores.astype("<f8")).tobytes()
                ).hexdigest(),
                "mean_normalized_attention_entropy": float(torch.cat(entropies).mean()),
            }
        )
        del model
    stack = np.stack(score_rows)
    if stack.shape != (len(MODEL_SEEDS), STATE_COUNT, ACTION_COUNT) or not np.isfinite(stack).all():
        raise DenseVJEPAInterfaceError("member score stack is invalid")
    selected = np.argmin(stack, axis=2)
    return stack, {
        "members": diagnostics,
        "score_stack_shape": list(stack.shape),
        "score_stack_sha256": hashlib.sha256(
            np.ascontiguousarray(stack.astype("<f8")).tobytes()
        ).hexdigest(),
        "states_with_seed_argmin_disagreement": int(
            np.any(selected != selected[:1], axis=0).sum()
        ),
    }


def _score_map_v1(
    plan: prior.RoleFeaturePlanV1, scores: np.ndarray
) -> dict[str, list[float]]:
    if scores.shape != (STATE_COUNT, ACTION_COUNT) or not np.isfinite(scores).all():
        raise DenseVJEPAInterfaceError("arm scores are invalid")
    return {
        state.state_id: [float(item) for item in scores[index]]
        for index, state in enumerate(plan.states)
    }


def _report_arm_v1(
    plan: prior.RoleFeaturePlanV1, scores: np.ndarray
) -> dict[str, object]:
    return prior._augment_report(  # noqa: SLF001
        selection_metrics_v1(plan.groups, _score_map_v1(plan, scores))
    )


def _score_evidence_v1(scores: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(scores.shape),
        "sha256_float64_c_order": hashlib.sha256(
            np.ascontiguousarray(scores.astype("<f8")).tobytes()
        ).hexdigest(),
    }


def retained_physical_report_v1(
    evaluation: Mapping[str, object], eval_plan: prior.RoleFeaturePlanV1
) -> dict[str, object]:
    if (
        evaluation.get("schema") != physical.SCHEMA
        or evaluation.get("status") != "COMPLETE_DEVELOPMENT_ONLY_PHYSICAL_OUTCOME_SCREEN"
        or evaluation.get("evaluation_identity_sha256")
        != EXPECTED_PHYSICAL_EVALUATION_IDENTITY
        or physical.evaluation_identity_v1(evaluation)
        != EXPECTED_PHYSICAL_EVALUATION_IDENTITY
        or evaluation.get("feature_plan")
        != {
            "identity_sha256": EXPECTED_COMBINED_PLAN_IDENTITY,
            "train_identity_sha256": EXPECTED_TRAIN_PLAN_IDENTITY,
            "eval_identity_sha256": EXPECTED_EVAL_PLAN_IDENTITY,
        }
    ):
        raise DenseVJEPAInterfaceError("retained physical evaluation changed")
    arms = evaluation.get("arms")
    if not isinstance(arms, Mapping):
        raise DenseVJEPAInterfaceError("retained physical arms changed")
    report = arms.get(physical.ODOMETRY_ARM)
    if not isinstance(report, Mapping):
        raise DenseVJEPAInterfaceError("retained physical arm changed")
    rows = report.get("group_results")
    summary = report.get("summary")
    if (
        not isinstance(rows, list)
        or len(rows) != STATE_COUNT
        or not isinstance(summary, Mapping)
        or summary.get("normalized_rank_regret") != EXPECTED_RETAINED_PHYSICAL_REGRET
    ):
        raise DenseVJEPAInterfaceError("retained physical report changed")
    for state, row in zip(eval_plan.states, rows, strict=True):
        if (
            not isinstance(row, Mapping)
            or row.get("state_id") != state.state_id
            or row.get("family") != state.family
            or row.get("scene_id") != state.scene_id
        ):
            raise DenseVJEPAInterfaceError("retained physical row order changed")
    return dict(report)


def _arms_and_diagnostics_v1(
    checkpoint: Mapping[str, object],
    plans: prior.CalibrationFeaturePlansV1,
    projected_train: torch.Tensor,
    projected_eval: torch.Tensor,
    retained_physical_evaluation: Mapping[str, object],
    device: torch.device,
) -> tuple[dict[str, dict[str, object]], dict[str, object], dict[str, object]]:
    task = dense._require_refitted_task_payload_v1(  # noqa: SLF001
        checkpoint["task_action_only"], plans.train
    )
    task_scores = dense.score_task_action_only_v1(plans.eval, task)
    donors, donor_document = same_action_wrong_scene_donors_v1(plans.eval)
    action_mean_payload = train_action_mean_innovation_v1(plans.train, projected_train)
    stored_action_mean = validate_action_mean_innovation_v1(
        checkpoint["train_action_mean_innovation"]
    )
    if (
        action_mean_payload["identity_sha256"]
        != checkpoint["train_action_mean_innovation"]["identity_sha256"]
        or not torch.equal(action_mean_payload["values"], stored_action_mean)
    ):
        raise DenseVJEPAInterfaceError("train action means did not reproduce")
    panel_specs = {
        TRUE_ARM: ("true_future", "true_state"),
        CURRENT_ARM: ("current_state", "current_state"),
        PERSISTENCE_ARM: ("current_state", "true_state"),
        WRONG_SCENE_ARM: ("wrong_scene", "true_state"),
        ACTION_MEAN_ARM: ("train_action_mean", "true_state"),
    }
    arms: dict[str, dict[str, object]] = {}
    diagnostics: dict[str, object] = {}
    score_evidence: dict[str, object] = {}
    for arm, (mode, state_key) in panel_specs.items():
        relations, conditions = relational_panels_v1(
            plans.eval,
            projected_eval,
            mode=mode,
            wrong_scene_donors=donors if mode == "wrong_scene" else None,
            train_action_means=stored_action_mean if mode == "train_action_mean" else None,
        )
        residual_stack, prediction_diagnostics = _predict_members_v1(
            checkpoint, relations, conditions, state_key=state_key, device=device
        )
        score_stack = residual_stack + task_scores[None, :, :]
        ensemble = score_stack.mean(axis=0)
        member_reports = []
        for seed, scores in zip(MODEL_SEEDS, score_stack, strict=True):
            report = _report_arm_v1(plans.eval, scores)
            member_reports.append(
                {
                    "seed": seed,
                    "score_evidence": _score_evidence_v1(scores),
                    "normalized_rank_regret": report["summary"]["normalized_rank_regret"],
                    "oracle_equivalent_selection_rate": report["summary"]["oracle_equivalent_selection_rate"],
                }
            )
        arms[arm] = _report_arm_v1(plans.eval, ensemble)
        diagnostics[arm] = {
            **prediction_diagnostics,
            "per_seed": member_reports,
        }
        score_evidence[arm] = {
            "ensemble": _score_evidence_v1(ensemble),
            "members": [item["score_evidence"] for item in member_reports],
        }
    arms["task_action_only"] = _report_arm_v1(plans.eval, task_scores)
    if arms["task_action_only"]["summary"]["normalized_rank_regret"] != EXPECTED_TASK_EVAL_REGRET:
        raise DenseVJEPAInterfaceError("task/action-only evaluation changed")
    oracle_scores = np.asarray(
        [state.dense_ranks for state in plans.eval.states], dtype=np.float64
    )
    arms["privileged_physical_oracle"] = _report_arm_v1(plans.eval, oracle_scores)
    arms[RETAINED_ARM] = retained_physical_report_v1(
        retained_physical_evaluation, plans.eval
    )
    hold_scores = np.ones((STATE_COUNT, ACTION_COUNT), dtype=np.float64)
    hold_scores[:, prior.HOLD_ACTION_ID] = 0.0
    arms["hold_constant"] = _report_arm_v1(plans.eval, hold_scores)
    arms["random_expected"] = prior._random_expected_report(plans.eval)  # noqa: SLF001
    score_evidence.update(
        {
            "task_action_only": _score_evidence_v1(task_scores),
            "privileged_physical_oracle": _score_evidence_v1(oracle_scores),
            "hold_constant": _score_evidence_v1(hold_scores),
        }
    )
    diagnostics["wrong_scene_mapping"] = donor_document
    diagnostics["train_action_mean_innovation"] = {
        "identity_sha256": action_mean_payload["identity_sha256"],
        "shape": [ACTION_COUNT, TOKEN_COUNT, PCA_DIMENSION],
    }
    return arms, diagnostics, score_evidence


def build_comparisons_v1(
    arms: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    candidate_rows = arms[TRUE_ARM]["group_results"]
    result = {
        name: prior.paired_family_scene_cluster_comparison_v1(
            candidate_rows,
            arms[baseline]["group_results"],
            resamples=BOOTSTRAP_RESAMPLES,
            seed=BOOTSTRAP_SEED,
        )
        for name, baseline in COMPARISON_BASELINES.items()
    }
    if set(result) != COMPARISON_NAMES:
        raise DenseVJEPAInterfaceError("paired comparison inventory changed")
    return result


def scientific_gates_v1(
    arms: Mapping[str, Mapping[str, object]],
    comparisons: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    if set(comparisons) != COMPARISON_NAMES:
        raise DenseVJEPAInterfaceError("paired comparison inventory changed")
    oracle = arms["privileged_physical_oracle"]["summary"]
    candidate = arms[TRUE_ARM]["summary"]
    random_expected = arms["random_expected"]["summary"]
    mapping = (
        ("3_true_future_beats_task_action_only", "true_future_vs_task_action_only"),
        ("4_true_future_beats_retained_physical_predecessor", "true_future_vs_retained_physical_predecessor"),
        ("5_true_future_beats_current_state", "true_future_vs_current_state"),
        ("6_true_future_beats_relational_persistence", "true_future_vs_relational_persistence"),
        ("7_true_future_beats_same_action_wrong_scene", "true_future_vs_same_action_wrong_scene"),
        ("8_true_future_beats_train_action_mean_innovation", "true_future_vs_train_action_mean_innovation"),
    )
    gates = {
        "2_privileged_physical_oracle": {
            "passed": oracle["normalized_rank_regret"] == 0.0
            and oracle["oracle_equivalent_selection_rate"] == 1.0,
            "normalized_rank_regret": oracle["normalized_rank_regret"],
            "oracle_equivalent_selection_rate": oracle["oracle_equivalent_selection_rate"],
        },
        "9_true_future_beats_random_expected": {
            "passed": candidate["normalized_rank_regret"]
            < random_expected["normalized_rank_regret"],
            "true_future": candidate["normalized_rank_regret"],
            "random_expected": random_expected["normalized_rank_regret"],
        },
    }
    for gate_name, comparison_name in mapping:
        measurement = comparisons[comparison_name]
        gates[gate_name] = {
            "passed": measurement["upper_95"] < 0.0,
            "measurement": dict(measurement),
        }
    if set(gates) != SCIENTIFIC_GATE_NAMES:
        raise DenseVJEPAInterfaceError("scientific gate inventory changed")
    return gates


def evaluate_primary_checkpoint_v1(
    checkpoint: Mapping[str, object],
    train_groups: Sequence[Any],
    eval_groups: Sequence[Any],
    train_features: object,
    eval_features: object,
    retained_physical_evaluation: Mapping[str, object],
    historical_dino_comparators: Mapping[str, object],
    device: torch.device,
    *,
    implementation_source_binding: Mapping[str, object],
) -> dict[str, object]:
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    _require_rocm_determinism_v1(device)
    _require_frozen_shared_contract_v1()
    plans = prior.build_calibration_feature_plans_v1(train_groups, eval_groups)
    _require_plans_v1(plans.train, plans.eval, plans.identity_sha256)
    validate_checkpoint_v1(
        checkpoint,
        train_plan=plans.train,
        implementation_source_binding=implementation_source_binding,
    )
    projected_train = project_cache_v1(train_features, checkpoint["pca"], label="train")
    projected_eval = project_cache_v1(eval_features, checkpoint["pca"], label="eval")
    arms, diagnostics, score_evidence = _arms_and_diagnostics_v1(
        checkpoint,
        plans,
        projected_train,
        projected_eval,
        retained_physical_evaluation,
        device,
    )
    comparisons = build_comparisons_v1(arms)
    gates = scientific_gates_v1(arms, comparisons)
    if not isinstance(historical_dino_comparators, Mapping):
        raise DenseVJEPAInterfaceError("historical DINO comparators changed")
    result: dict[str, object] = {
        "schema": SCHEMA,
        "status": "COMPLETE_DEVELOPMENT_ONLY_PHYSICAL_INTERFACE_EVALUATION",
        "claim_scope": "PRIVILEGED_ACTUAL_FUTURE_VJEPA_REPRESENTATION_INTERFACE_CEILING",
        "config": config_v1(),
        "feature_plan": {
            "identity_sha256": plans.identity_sha256,
            "train_identity_sha256": plans.train.identity_sha256,
            "eval_identity_sha256": plans.eval.identity_sha256,
        },
        "checkpoint_identity_sha256": checkpoint["identity_sha256"],
        "pca": {
            "identity_sha256": checkpoint["pca"]["identity_sha256"],
            "dimension": PCA_DIMENSION,
            "row_count": PCA_ROW_COUNT,
            "eigenvalues": [float(item) for item in checkpoint["pca"]["eigenvalues"].tolist()],
        },
        "train_action_mean_innovation": diagnostics["train_action_mean_innovation"],
        "task_action_only_identity_sha256": checkpoint["task_action_only"]["identity_sha256"],
        "member_training": [
            {
                "seed": member["seed"],
                "initial_identity_sha256": member["initial_identity_sha256"],
                "true_identity_sha256": member["true_identity_sha256"],
                "current_identity_sha256": member["current_identity_sha256"],
                "true_training": member["true_training"],
                "current_training": member["current_training"],
            }
            for member in checkpoint["members"]
        ],
        "prediction_diagnostics": diagnostics,
        "score_evidence": score_evidence,
        "arms": arms,
        "paired_family_scene_cluster_comparisons": comparisons,
        "historical_dino_comparators_report_only": dict(historical_dino_comparators),
        "safety": prior._safety_support(plans),  # noqa: SLF001
        "finiteness": {
            "pca": True,
            "train_action_mean_innovation": True,
            "training": True,
            "member_scores": True,
            "ensemble_scores": True,
            "reported_metrics": True,
        },
        "gates": gates,
        "scientific_gates_2_to_9_passed": all(gate["passed"] for gate in gates.values()),
    }
    result["evaluation_identity_sha256"] = evaluation_identity_v1(result)
    canonical_bytes_v1(result)
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
        or evaluation.get("status")
        != "COMPLETE_DEVELOPMENT_ONLY_PHYSICAL_INTERFACE_EVALUATION"
        or evaluation.get("config") != config_v1()
        or evaluation.get("evaluation_identity_sha256")
        != evaluation_identity_v1(evaluation)
        or not isinstance(gates, Mapping)
        or set(gates) != SCIENTIFIC_GATE_NAMES
        or type(infrastructure_checks_passed) is not bool
        or type(deterministic_replay_passed) is not bool
    ):
        raise DenseVJEPAInterfaceError("verdict inputs changed")
    arms = evaluation.get("arms")
    comparisons = evaluation.get("paired_family_scene_cluster_comparisons")
    if (
        not isinstance(arms, Mapping)
        or not isinstance(comparisons, Mapping)
        or dict(gates) != scientific_gates_v1(arms, comparisons)
    ):
        raise DenseVJEPAInterfaceError("scientific gates changed")
    scientific_pass = all(gate["passed"] for gate in gates.values())
    if evaluation.get("scientific_gates_2_to_9_passed") is not scientific_pass:
        raise DenseVJEPAInterfaceError("scientific gate aggregate changed")
    all_passed = (
        infrastructure_checks_passed
        and deterministic_replay_passed
        and scientific_pass
    )
    status = (
        INFRASTRUCTURE_FAILURE_STATUS
        if not infrastructure_checks_passed or not deterministic_replay_passed
        else PASS_STATUS if all_passed else STOP_STATUS
    )
    return {
        "gates": {
            "1_infrastructure_and_custody": {"passed": infrastructure_checks_passed},
            **dict(gates),
            "10_exact_fresh_process_cache_only_replay": {
                "passed": deterministic_replay_passed
            },
        },
        "passed": all_passed,
        "terminal_status": status,
    }


__all__ = (
    "ACTION_MEAN_ARM",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CHECKPOINT_SCHEMA",
    "CONFIG_SCHEMA",
    "CURRENT_ARM",
    "DenseVJEPAInterfaceError",
    "INFRASTRUCTURE_FAILURE_STATUS",
    "PASS_STATUS",
    "PERSISTENCE_ARM",
    "SCHEMA",
    "SCIENTIFIC_GATE_NAMES",
    "STOP_STATUS",
    "TRUE_ARM",
    "WRONG_SCENE_ARM",
    "build_comparisons_v1",
    "canonical_bytes_v1",
    "config_v1",
    "evaluate_primary_checkpoint_v1",
    "evaluation_identity_v1",
    "fit_primary_checkpoint_v1",
    "fit_train_pca_v1",
    "pca_identity_v1",
    "project_cache_v1",
    "relational_panels_v1",
    "retained_physical_report_v1",
    "same_action_wrong_scene_donors_v1",
    "scientific_gates_v1",
    "train_action_mean_innovation_v1",
    "validate_action_mean_innovation_v1",
    "validate_checkpoint_v1",
    "verdict_v1",
)
