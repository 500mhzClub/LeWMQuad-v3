#!/usr/bin/env python3
"""Run one fresh RGB full-whitened predictive-state H4 JEPA probe."""
from __future__ import annotations

import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import (  # noqa: E402
    run_go2_rgb_whitened_delta_predictive_state_h4_jepa_v1 as base,
)


core = base.core
BASE_WRAPPER_SOURCE = ROOT / (
    "scripts/run_go2_rgb_whitened_delta_predictive_state_h4_jepa_v1.py"
)
BASE_WRAPPER_SOURCE_SHA256 = (
    "00ccb04b0541a597b4a9b73e55c4c9cf6daa40d1ea07728a058afe2473397746"
)
BASE_WRAPPER_SOURCE_BYTES = 38_611
MODEL_MODULE = "lewm.models.go2_rgb_full_whitened_predictive_state_h4_jepa_v1"
MODEL_SOURCE = ROOT / "lewm/models/go2_rgb_full_whitened_predictive_state_h4_jepa_v1.py"
MODEL_SOURCE_SHA256 = "c139176b5a67e259700c620546f08099bf4821ff9d2df256c0f784d81613d417"
MODEL_SOURCE_BYTES = 7_599
OUTPUT_ROOT = ROOT / (
    ".generated/go2_rgb_full_whitened_predictive_state_h4_jepa_v1/probe_v1"
)
SCHEMA = "lewm_go2_rgb_full_whitened_predictive_state_h4_jepa_v1"
PASS_DECISION = "PASS_MAIN_POOL_RGB_FULL_WHITENED_PREDICTIVE_STATE_H4_JEPA_V1"
STOP_DECISION = "STOP_MAIN_POOL_RGB_FULL_WHITENED_PREDICTIVE_STATE_H4_JEPA_V1"
_BASE_BUILD_MODEL = base._build_model
_BASE_EVALUATE = base._evaluate
_BASE_RUN = base._run
_BASE_STATE_ELIGIBLE = base._state_eligible
_BASE_STATE_GEOMETRY = base._state_geometry
_PAIR_CAPTURE_ACTIVE = False
_PAIR_CAPTURE_PENDING: tuple[Any, dict[str, list[float]]] | None = None


def _verify_source_closure() -> dict[str, dict[str, Any]]:
    wrapper_sha256 = os.environ.get("LEWM_FULL_WHITENED_H4_WRAPPER_SHA256", "")
    wrapper_bytes_text = os.environ.get("LEWM_FULL_WHITENED_H4_WRAPPER_BYTES", "")
    try:
        wrapper_bytes = int(wrapper_bytes_text)
    except ValueError as error:
        raise core.ContractError(
            "external full-whitened H4 wrapper byte binding is required"
        ) from error
    return {
        "full_whitened_h4_wrapper": base._source_binding(
            Path(__file__).resolve(), wrapper_sha256, wrapper_bytes
        ),
        "wdps_h4_wrapper_dependency": base._source_binding(
            BASE_WRAPPER_SOURCE,
            BASE_WRAPPER_SOURCE_SHA256,
            BASE_WRAPPER_SOURCE_BYTES,
        ),
        "shared_runner": base._source_binding(
            base.CORE_SOURCE,
            base.CORE_SOURCE_SHA256,
            base.CORE_SOURCE_BYTES,
        ),
        "full_whitened_h4_model": base._source_binding(
            MODEL_SOURCE,
            MODEL_SOURCE_SHA256,
            MODEL_SOURCE_BYTES,
        ),
        "wdps_h4_model_dependency": base._source_binding(
            base.MODEL_SOURCE,
            base.MODEL_SOURCE_SHA256,
            base.MODEL_SOURCE_BYTES,
        ),
        "dense_h4_model_dependency": base._source_binding(
            base.DENSE_MODEL_SOURCE,
            base.DENSE_MODEL_SOURCE_SHA256,
            base.DENSE_MODEL_SOURCE_BYTES,
        ),
        "inherited_v1_model": base._source_binding(
            base.BASE_MODEL_SOURCE,
            base.BASE_MODEL_SOURCE_SHA256,
            base.BASE_MODEL_SOURCE_BYTES,
        ),
        "encoder_dependency": base._source_binding(
            base.ENCODER_SOURCE,
            base.ENCODER_SOURCE_SHA256,
            base.ENCODER_SOURCE_BYTES,
        ),
    }


def _configure_core(source_bindings: Mapping[str, Mapping[str, Any]]) -> None:
    base.MODEL_MODULE = MODEL_MODULE
    base.MODEL_SOURCE = MODEL_SOURCE
    base.MODEL_SOURCE_SHA256 = MODEL_SOURCE_SHA256
    base.MODEL_SOURCE_BYTES = MODEL_SOURCE_BYTES
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.SCHEMA = SCHEMA
    base.PASS_DECISION = PASS_DECISION
    base.STOP_DECISION = STOP_DECISION
    base._configure_core(source_bindings)
    core.TARGET_DESCRIPTION = (
        "jointly_learned_eight_dimensional_full_whitened_future_change_state_"
        "with_fixed_N320_target_and_history_teacher_no_ema"
    )
    core.OBJECTIVE_DESCRIPTION = (
        "25*predicted_target_cross_covariance_identity+25*mean_predicted_and_"
        "target_within_covariance_identity+25*mean_predicted_and_target_zero_"
        "mean+1*online_history_to_fixed_N320_alignment; raw_training_mse_absent;"
        "hinge_variance_absent; all_controls_absent"
    )
    core.ADDITIONAL_SCIENCE = {
        "state": "four_horizons_times_eight_learned_future_change_dimensions",
        "target_state": (
            "shared_zero_preserving_spatial_attention_pool_of_fixed_N320_"
            "normalized_future_minus_e2_patch_deltas"
        ),
        "prediction": "dense_three_frame_history_plus_ordered_action_prefix",
        "joint_training": (
            "online_encoder+target_compressor+history+action+predictor_one_backward"
        ),
        "full_whitening": (
            "per_horizon_unbiased_within_and_cross_covariance_to_identity_"
            "all_64_entries"
        ),
        "removed_predecessor_terms": (
            "raw_prediction_mse+marginal_variance_hinge+weight_one_raw_"
            "offdiagonal_covariance"
        ),
        "persistence": "exact_zero_compact_change_state",
        "fixed_teacher_role": "fixed_target_and_history_teacher",
        "absent": [
            "distribution_atoms_or_learned_variance",
            "validation_fitted_whitening_or_inverse_covariance",
            "raw_prediction_mse_training_loss",
            "wrong_action_history_persistence_or_hold_training_control",
            "contrastive_negatives_or_codebook",
            "reconstruction_or_navigation_loss",
            "pose_depth_flow_bev_warp_transport_or_geometry_target",
        ],
        "predecessor_predictor_checkpoint_tensor_open_count": 0,
    }
    core.EXECUTION_SOURCE_BINDINGS = {
        name: dict(binding) for name, binding in source_bindings.items()
    }


def _build_model(runtime: Any, n320_encoder: Mapping[str, Any]) -> Any:
    module = runtime.model_module
    config_cls = getattr(module, "FullWhitenedPredictiveStateConfig", None)
    model_cls = getattr(module, "FullWhitenedPredictiveStateH4JEPA", None)
    if config_cls is None or model_cls is None:
        raise core.ContractError("full-whitened model module lacks its reviewed API")
    config = config_cls(
        image_size=core.IMAGE_SIZE,
        target_ema_momentum=core.EMA_MOMENTUM,
        variance_weight=0.0,
        action_vocabulary=core.PRIMITIVES,
    )
    model = model_cls(n320_encoder_state_dict=n320_encoder, config=config)
    for name in (
        "encode_history",
        "predict_from_belief",
        "encode_target_state",
        "hard_sync_target",
        "update_target",
    ):
        if not callable(getattr(model, name, None)):
            raise core.ContractError(f"full-whitened model is missing {name}()")
    if tuple(model.action_vocabulary) != core.PRIMITIVES:
        raise core.ContractError("full-whitened primitive vocabulary changed")
    return model


def _cross_covariance_geometry(
    predicted: Any,
    target: Any,
    runtime: Any,
) -> dict[str, list[float]]:
    torch = runtime.torch
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise core.ContractError("full-whitened paired state shape changed")
    predicted_value = predicted.to(dtype=torch.float64)
    target_value = target.to(dtype=torch.float64)
    predicted_centered = predicted_value - predicted_value.mean(dim=0, keepdim=True)
    target_centered = target_value - target_value.mean(dim=0, keepdim=True)
    cross_covariance = torch.einsum(
        "bhd,bhe->hde", predicted_centered, target_centered
    ) / max(1, int(predicted_value.shape[0]) - 1)
    dim = int(predicted_value.shape[-1])
    identity = torch.eye(
        dim, dtype=predicted_value.dtype, device=predicted_value.device
    )[None]
    difference = cross_covariance - identity
    off_diagonal = difference.clone()
    diagonal_index = torch.arange(dim, device=predicted_value.device)
    off_diagonal[:, diagonal_index, diagonal_index] = 0.0
    result = {
        "predicted_target_cross_covariance_identity_error": difference.square()
        .sum(dim=(-2, -1))
        .div(float(dim))
        .tolist(),
        "predicted_target_maximum_cross_diagonal_error": torch.diagonal(
            difference, dim1=-2, dim2=-1
        )
        .abs()
        .max(dim=-1)
        .values.tolist(),
        "predicted_target_maximum_offdiagonal_cross_covariance": off_diagonal.abs()
        .amax(dim=(-2, -1))
        .tolist(),
    }
    if not all(math.isfinite(item) for vector in result.values() for item in vector):
        raise core.ContractError("nonfinite full-whitened cross covariance geometry")
    return result


def _state_geometry(state: Any, runtime: Any) -> dict[str, list[float]]:
    global _PAIR_CAPTURE_PENDING
    geometry = _BASE_STATE_GEOMETRY(state, runtime)
    torch = runtime.torch
    value = state.to(dtype=torch.float64)
    centered = value - value.mean(dim=0, keepdim=True)
    covariance = torch.einsum("bhd,bhe->hde", centered, centered) / max(
        1, int(value.shape[0]) - 1
    )
    dim = int(value.shape[-1])
    identity = torch.eye(dim, dtype=value.dtype, device=value.device)[None]
    difference = covariance - identity
    eigenvalues = torch.linalg.eigvalsh(covariance)
    off_diagonal = difference.clone()
    diagonal_index = torch.arange(dim, device=value.device)
    off_diagonal[:, diagonal_index, diagonal_index] = 0.0
    geometry.update(
        {
            "covariance_identity_error": difference.square()
            .sum(dim=(-2, -1))
            .div(float(dim))
            .tolist(),
            "minimum_covariance_eigenvalue": eigenvalues.min(dim=-1).values.tolist(),
            "maximum_covariance_eigenvalue": eigenvalues.max(dim=-1).values.tolist(),
            "maximum_variance_error": torch.diagonal(difference, dim1=-2, dim2=-1)
            .abs()
            .max(dim=-1)
            .values.tolist(),
            "maximum_offdiagonal_covariance": off_diagonal.abs()
            .amax(dim=(-2, -1))
            .tolist(),
        }
    )
    if not all(
        math.isfinite(item)
        for vector in geometry.values()
        for item in vector
    ):
        raise core.ContractError("nonfinite full-whitened state geometry")
    if _PAIR_CAPTURE_ACTIVE:
        if _PAIR_CAPTURE_PENDING is None:
            _PAIR_CAPTURE_PENDING = (state, geometry)
        else:
            predicted, predicted_geometry = _PAIR_CAPTURE_PENDING
            cross_geometry = _cross_covariance_geometry(predicted, state, runtime)
            predicted_geometry.update(cross_geometry)
            geometry.update(cross_geometry)
            _PAIR_CAPTURE_PENDING = None
    return geometry


def _evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
    global _PAIR_CAPTURE_ACTIVE, _PAIR_CAPTURE_PENDING
    if _PAIR_CAPTURE_ACTIVE or _PAIR_CAPTURE_PENDING is not None:
        raise core.ContractError("nested full-whitened pair capture")
    _PAIR_CAPTURE_ACTIVE = True
    try:
        result = _BASE_EVALUATE(*args, **kwargs)
        if _PAIR_CAPTURE_PENDING is not None:
            raise core.ContractError("full-whitened evaluator captured only one branch")
        return result
    finally:
        _PAIR_CAPTURE_ACTIVE = False
        _PAIR_CAPTURE_PENDING = None


def _state_eligible(observation: Mapping[str, Any]) -> bool:
    if not _BASE_STATE_ELIGIBLE(observation):
        return False
    for role in ("predicted", "target"):
        geometry = observation["state_geometry"][role]
        if not all(value <= 0.50 for value in geometry["covariance_identity_error"]):
            return False
    cross = observation["state_geometry"]["predicted"][
        "predicted_target_cross_covariance_identity_error"
    ]
    return all(value <= 0.50 for value in cross)


def _run(
    *args: Any,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metrics, artifact, decision = _BASE_RUN(*args, **kwargs)
    training = metrics.get("training_losses")
    if not isinstance(training, dict):
        raise core.ContractError("full-whitened training receipt is absent")
    mapping = {
        "state_similarity": "predicted_target_cross_covariance_identity",
        "predicted_covariance": "predicted_within_covariance_identity",
        "target_covariance": "target_within_covariance_identity",
    }
    for bucket_name in ("mean_over_completed_updates", "last_completed_update"):
        bucket = training.get(bucket_name)
        if not isinstance(bucket, dict):
            raise core.ContractError("full-whitened training loss bucket is absent")
        for old_name, new_name in mapping.items():
            if old_name not in bucket or new_name in bucket:
                raise core.ContractError("full-whitened loss alias contract changed")
            bucket[new_name] = bucket.pop(old_name)
        for disabled_name in ("predicted_variance", "target_variance"):
            value = bucket.pop(disabled_name, None)
            if value is None or value != 0.0:
                raise core.ContractError("disabled hinge variance was not exact zero")
    training["disabled_terms"] = [
        "raw_state_prediction_mse",
        "predicted_marginal_variance_hinge",
        "target_marginal_variance_hinge",
        "weight_one_raw_offdiagonal_covariance",
    ]
    training["receipt_field_semantics"] = {
        "predicted_target_cross_covariance_identity": "X(p,q)",
        "predicted_within_covariance_identity": "W(p)",
        "target_within_covariance_identity": "W(q)",
        "predicted_mean": "M(p)",
        "target_mean": "M(q)",
    }
    return metrics, artifact, decision


def _install_runtime_adapters() -> None:
    if base._build_model not in (_BASE_BUILD_MODEL, _build_model):
        raise core.ContractError("WDPS model builder changed before whitening adapter")
    if base._state_geometry not in (_BASE_STATE_GEOMETRY, _state_geometry):
        raise core.ContractError("WDPS state geometry changed before whitening adapter")
    if base._evaluate not in (_BASE_EVALUATE, _evaluate):
        raise core.ContractError("WDPS evaluator changed before whitening adapter")
    if base._state_eligible not in (_BASE_STATE_ELIGIBLE, _state_eligible):
        raise core.ContractError("WDPS eligibility changed before whitening adapter")
    base._build_model = _build_model
    base._state_geometry = _state_geometry
    base._evaluate = _evaluate
    base._state_eligible = _state_eligible
    base._install_runtime_adapters()
    if core._run is not base._run:
        raise core.ContractError("WDPS run hook was not installed before receipt adapter")
    core._run = _run


def main(argv: Sequence[str] | None = None) -> int:
    if Path(core.__file__).resolve() != base.CORE_SOURCE:
        raise core.ContractError("shared runner imported from an unexpected path")
    if Path(base.__file__).resolve() != BASE_WRAPPER_SOURCE:
        raise core.ContractError("WDPS runner imported from an unexpected path")
    source_bindings = _verify_source_closure()
    base._install_bound_model_package_stubs()
    _configure_core(source_bindings)
    _install_runtime_adapters()
    return core.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
