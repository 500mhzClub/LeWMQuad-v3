"""Narrow pre-G2 boundary for the frozen RGB joint-JEPA V4 candidate.

The V4 semantic head predicts current-camera physical evidence.  It does not
predict body-inflated configuration-space occupancy, so routing is explicitly
deferred until the registered post-memory physical-to-configuration stage.
"""
from __future__ import annotations

import hashlib
import io
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from lewm.benchmarks.traversability_metrics import (
    TraversabilityThresholds,
    evaluate_traversability,
)
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v4_residual_local_semantic_decoder import (
    ACTION_VOCABULARY_V1,
    GeometryAnchoredSweptProgressSurvivalJointJepaV4,
)


CHECKPOINT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v4_residual_local_"
    "semantic_decoder_checkpoint_v1"
)
INFERENCE_INPUT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v4_inference_input_v1"
)
INFERENCE_OUTPUT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v4_inference_output_v1"
)
PHYSICAL_EVIDENCE_BATCH_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v4_physical_evidence_batch_v1"
)
PHYSICAL_EVIDENCE_METRICS_SCHEMA = (
    "lewm_go2_observable_physical_evidence_metrics_v1"
)
PHYSICAL_G2_GATE_SCHEMA = "lewm_go2_physical_evidence_g2_v1"
ROUTING_NOT_APPLICABLE_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v4_"
    "routing_not_applicable_v1"
)
PHYSICAL_OCCUPANCY_TARGET_SPACE = "observable_physical_occupancy"

CLASS_ORDER = ("unknown", "free", "occupied")
ACTION_ORDER = tuple(ACTION_VOCABULARY_V1)
SWEEP_MASK_STATE_KEY = "predictor.swept_progress_head.sweep_masks"

_CHECKPOINT_KEYS = frozenset(
    {
        "schema",
        "development_only",
        "resume_authorized",
        "qualified",
        "constructor_initialization_seed",
        "semantic_decoder_initialization_seed",
        "experiment_seed",
        "initialization_source",
        "predecessor_experiment_checkpoint_read",
        "auxiliary_objective",
        "initial_semantic_decoder",
        "accounting",
        "model_state_dict",
    }
)
_INFERENCE_INPUT_KEYS = frozenset({"schema", "rgb_f32_chw"})
_INFERENCE_OUTPUT_KEYS = frozenset(
    {
        "schema",
        "class_order",
        "action_order",
        "physical_logits_f32_chw",
        "physical_probabilities_f32_chw",
        "anchor_in_frustum_bool_hw",
        "all_action_survival_logits_f32",
    }
)

_TERMINAL_ACCOUNTING = {
    "updates": 1_000,
    "presentations": 16_000,
    "microbatch_graphs": 4_000,
    "backward_calls": 4_000,
    "optimizer_steps": 1_000,
    "ema_steps": 1_000,
    "predictor_forwards": 4_000,
    "predictor_objectives": 4_000,
}

_AUXILIARY_OBJECTIVE = {
    "name": "occupied_vs_rest_safety",
    "coefficient": 0.5,
    "logit_definition": (
        "occupied_semantic_logit_minus_logsumexp_free_and_unknown_semantic_logits"
    ),
    "row_balancing": (
        "per_raster_row_equal_average_of_present_occupied_and_rest_target_classes"
    ),
    "current_next_aggregation": "equal_average",
    "normalization": "binary_cross_entropy_with_logits_divided_by_log_2",
    "new_trainable_parameters": False,
}

_SEMANTIC_DECODER_ARCHITECTURE = {
    "schema": "lewm_residual_local_semantic_decoder_v4_architecture_v1",
    "merge": "base_logits_plus_residual_logits",
    "base": {
        "type": "Conv2d",
        "in_channels": 64,
        "out_channels": 3,
        "kernel_size": [1, 1],
        "bias": True,
        "identity": "exact_existing_v3_semantic_head",
    },
    "residual": {
        "local": {
            "type": "Conv2d",
            "in_channels": 64,
            "out_channels": 64,
            "kernel_size": [3, 3],
            "stride": [1, 1],
            "padding": [1, 1],
            "bias": True,
        },
        "activation": {"type": "GELU", "approximate": "none"},
        "output": {
            "type": "Conv2d",
            "in_channels": 64,
            "out_channels": 3,
            "kernel_size": [1, 1],
            "bias": True,
            "weight_initialization": "exact_zeros",
            "bias_initialization": "exact_zeros",
        },
    },
    "added_trainable_parameter_count": 37_123,
    "initialization_seed": 20_260_713,
    "visibility_mask": "inherited_bev_lift_anchor_in_frustum_post_logits",
    "normalization_layers": 0,
}

# Exact range of the frozen RGB -> float32 transform: channel / 255 followed
# by ImageNet mean/std normalization.  The input schema carries the resized,
# normalized tensor rather than image bytes, avoiding a second decoder route.
_RGB_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
_RGB_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
_RGB_CHANNEL_MIN = (np.float32(0.0) - _RGB_MEAN) / _RGB_STD
_RGB_CHANNEL_MAX = (np.float32(1.0) - _RGB_MEAN) / _RGB_STD


def _require_exact_keys(value: object, expected: frozenset[str], name: str) -> dict:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact dict")
    mapping = value
    if set(mapping) != expected:
        missing = sorted(expected - set(mapping))
        extra = sorted(set(mapping) - expected)
        raise ValueError(f"{name} keys changed; missing={missing}, extra={extra}")
    return mapping


def _require_exact_int(value: object, expected: int, name: str) -> None:
    if type(value) is not int or value != expected:
        raise ValueError(f"{name} must be exact integer {expected}")


def _validate_checkpoint_metadata(checkpoint: dict) -> None:
    if checkpoint["schema"] != CHECKPOINT_SCHEMA:
        raise ValueError("V4 checkpoint schema changed")
    if checkpoint["development_only"] is not True:
        raise ValueError("V4 checkpoint must remain development-only")
    if checkpoint["resume_authorized"] is not False:
        raise ValueError("V4 checkpoint unexpectedly authorizes resume")
    if checkpoint["qualified"] is not False:
        raise ValueError("raw V4 checkpoint unexpectedly claims qualification")
    _require_exact_int(
        checkpoint["constructor_initialization_seed"],
        20_260_712,
        "constructor_initialization_seed",
    )
    _require_exact_int(
        checkpoint["semantic_decoder_initialization_seed"],
        20_260_713,
        "semantic_decoder_initialization_seed",
    )
    _require_exact_int(checkpoint["experiment_seed"], 20_260_728, "experiment_seed")
    if checkpoint["initialization_source"] != "exact_n320_encoder_only":
        raise ValueError("V4 initialization source changed")
    if checkpoint["predecessor_experiment_checkpoint_read"] is not False:
        raise ValueError("V4 checkpoint claims predecessor checkpoint access")

    auxiliary = _require_exact_keys(
        checkpoint["auxiliary_objective"],
        frozenset(_AUXILIARY_OBJECTIVE),
        "auxiliary_objective",
    )
    if (
        auxiliary != _AUXILIARY_OBJECTIVE
        or type(auxiliary["coefficient"]) is not float
        or auxiliary["new_trainable_parameters"] is not False
    ):
        raise ValueError("V4 auxiliary-objective receipt changed")

    accounting = _require_exact_keys(
        checkpoint["accounting"],
        frozenset(_TERMINAL_ACCOUNTING),
        "accounting",
    )
    for name, expected in _TERMINAL_ACCOUNTING.items():
        _require_exact_int(accounting[name], expected, f"accounting.{name}")


def _validate_state_mapping(value: object) -> dict[str, torch.Tensor]:
    if type(value) is not dict or not value:
        raise TypeError("model_state_dict must be a nonempty exact dict")
    state: dict[str, torch.Tensor] = value
    for name, tensor in state.items():
        if type(name) is not str or not name:
            raise TypeError("model state names must be nonempty strings")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"model state {name!r} is not a tensor")
        if tensor.device.type != "cpu":
            raise TypeError(f"model state {name!r} is not CPU-resident")
        if tensor.layout is not torch.strided or not tensor.is_contiguous():
            raise TypeError(f"model state {name!r} is not dense contiguous strided")
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"model state {name!r} is nonfinite")
    return state


def _validate_initial_decoder_receipt(
    value: object,
    anchor_in_frustum: torch.Tensor,
) -> None:
    receipt = _require_exact_keys(
        value,
        frozenset(
            {
                "architecture",
                "initial_residual_output_exactly_zero",
                "semantic_parameter_count",
                "added_parameter_count",
                "all_semantic_parameters_in_lift_semantic_exactly_once",
                "visibility_mask",
            }
        ),
        "initial_semantic_decoder",
    )
    if receipt["architecture"] != _SEMANTIC_DECODER_ARCHITECTURE:
        raise ValueError("V4 initial semantic-decoder architecture receipt changed")
    if receipt["initial_residual_output_exactly_zero"] is not True:
        raise ValueError("V4 residual output was not receipted as zero-initialized")
    _require_exact_int(
        receipt["semantic_parameter_count"], 37_318, "semantic_parameter_count"
    )
    _require_exact_int(receipt["added_parameter_count"], 37_123, "added_parameter_count")
    if receipt["all_semantic_parameters_in_lift_semantic_exactly_once"] is not True:
        raise ValueError("V4 semantic parameter partition receipt changed")

    visibility = _require_exact_keys(
        receipt["visibility_mask"],
        frozenset({"shape", "dtype", "true_cell_count", "sha256", "application"}),
        "initial_semantic_decoder.visibility_mask",
    )
    if (
        visibility["shape"] != [64, 64]
        or visibility["dtype"] != "bool"
        or visibility["application"] != "inherited_post_logits"
    ):
        raise ValueError("V4 semantic visibility receipt changed")
    anchor = anchor_in_frustum.detach().cpu().contiguous()
    if tuple(anchor.shape) != (64, 64) or anchor.dtype != torch.bool:
        raise ValueError("V4 model anchor visibility mask changed")
    _require_exact_int(
        visibility["true_cell_count"],
        int(anchor.sum().item()),
        "visibility_mask.true_cell_count",
    )
    expected_sha256 = hashlib.sha256(anchor.numpy().tobytes(order="C")).hexdigest()
    if visibility["sha256"] != expected_sha256:
        raise ValueError("V4 semantic visibility SHA-256 changed")


def load_checkpoint(encoded: bytes) -> GeometryAnchoredSweptProgressSurvivalJointJepaV4:
    """Validate and reconstruct the exact terminal V4 model on CPU."""

    if type(encoded) is not bytes or not encoded:
        raise TypeError("encoded checkpoint must be nonempty exact bytes")
    checkpoint = torch.load(
        io.BytesIO(encoded),
        map_location="cpu",
        weights_only=True,
    )
    checkpoint = _require_exact_keys(checkpoint, _CHECKPOINT_KEYS, "checkpoint")
    _validate_checkpoint_metadata(checkpoint)
    state = _validate_state_mapping(checkpoint["model_state_dict"])

    encoder_prefix = "encoder."
    encoder_state = {
        name[len(encoder_prefix) :]: tensor.detach().clone()
        for name, tensor in state.items()
        if name.startswith(encoder_prefix)
    }
    if not encoder_state:
        raise ValueError("V4 checkpoint lacks its online encoder state")
    sweep_masks = state.get(SWEEP_MASK_STATE_KEY)
    if not isinstance(sweep_masks, torch.Tensor):
        raise ValueError("V4 checkpoint lacks exact swept-progress masks")

    model = GeometryAnchoredSweptProgressSurvivalJointJepaV4(
        encoder_state,
        sweep_masks.detach().clone(),
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    model.requires_grad_(False)

    loaded_state = model.state_dict()
    if set(loaded_state) != set(state) or any(
        not torch.equal(loaded_state[name], state[name]) for name in state
    ):
        raise RuntimeError("strict V4 reconstruction changed checkpoint tensors")
    if (
        int(model.target_hard_sync_count.item()) != 1
        or int(model.ema_update_count.item()) != 1_000
    ):
        raise ValueError("V4 model target-update counters disagree with accounting")
    _validate_initial_decoder_receipt(
        checkpoint["initial_semantic_decoder"],
        model.bev_lift.anchor_in_frustum,
    )
    return model


def _numeric_array(value: object, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} is not a rectangular numeric array") from exc
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise TypeError(f"{name} must contain only real JSON numbers")
    if raw.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError(f"{name} is nonfinite")
    return result


def _normalized_rgb(value: object) -> torch.Tensor:
    array = _numeric_array(value, name="rgb_f32_chw", shape=(3, 112, 112))
    tolerance = 2e-6
    for channel in range(3):
        if (
            (array[channel] < float(_RGB_CHANNEL_MIN[channel]) - tolerance).any()
            or (array[channel] > float(_RGB_CHANNEL_MAX[channel]) + tolerance).any()
        ):
            raise ValueError("rgb_f32_chw lies outside the frozen normalized RGB range")
    result = torch.from_numpy(np.asarray(array, dtype=np.float32).copy()).contiguous()
    if result.dtype != torch.float32:
        raise TypeError("normalized RGB conversion changed dtype")
    return result


def _require_frozen_cpu_model(
    model: object,
) -> GeometryAnchoredSweptProgressSurvivalJointJepaV4:
    if type(model) is not GeometryAnchoredSweptProgressSurvivalJointJepaV4:
        raise TypeError("model must be the exact V4 class")
    if model.training:
        raise ValueError("V4 inference model must be in evaluation mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("V4 inference model must be frozen")
    tensors = tuple(model.parameters()) + tuple(model.buffers())
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise TypeError("V4 adapter accepts only the CPU-loaded candidate")
    if tuple(model.action_vocabulary) != ACTION_ORDER:
        raise ValueError("V4 action vocabulary changed")
    return model


def infer_one(
    model: GeometryAnchoredSweptProgressSurvivalJointJepaV4,
    model_input: Mapping[str, Any],
) -> dict[str, Any]:
    """Infer one normalized RGB observation without accepting target fields."""

    frozen = _require_frozen_cpu_model(model)
    parsed = _require_exact_keys(model_input, _INFERENCE_INPUT_KEYS, "model_input")
    if parsed["schema"] != INFERENCE_INPUT_SCHEMA:
        raise ValueError("V4 inference-input schema changed")
    rgb = _normalized_rgb(parsed["rgb_f32_chw"])[None]

    with torch.inference_mode():
        sampling = frozen.encode_online_with_sampling(rgb)
        logits = frozen.semantic_logits_from_latent(sampling.latent)
        probabilities = torch.softmax(logits, dim=1)
        survival_logits = frozen.predict_all_actions_with_survival(
            sampling.latent
        ).survival_logits
    anchor = frozen.bev_lift.anchor_in_frustum

    expected = {
        "physical_logits_f32_chw": (logits, (1, 3, 64, 64), torch.float32),
        "physical_probabilities_f32_chw": (
            probabilities,
            (1, 3, 64, 64),
            torch.float32,
        ),
        "all_action_survival_logits_f32": (
            survival_logits,
            (1, 9, 16),
            torch.float32,
        ),
        "anchor_in_frustum_bool_hw": (anchor, (64, 64), torch.bool),
    }
    for name, (tensor, shape, dtype) in expected.items():
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise RuntimeError(f"{name} shape or dtype changed")
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"{name} is nonfinite")
    if not torch.allclose(
        probabilities.sum(dim=1),
        torch.ones_like(probabilities[:, 0]),
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError("V4 physical probabilities do not sum to one")

    return {
        "schema": INFERENCE_OUTPUT_SCHEMA,
        "class_order": list(CLASS_ORDER),
        "action_order": list(ACTION_ORDER),
        "physical_logits_f32_chw": logits[0].cpu().tolist(),
        "physical_probabilities_f32_chw": probabilities[0].cpu().tolist(),
        "anchor_in_frustum_bool_hw": anchor.cpu().tolist(),
        "all_action_survival_logits_f32": survival_logits[0].cpu().tolist(),
    }


def routing_not_applicable_receipt() -> dict[str, Any]:
    """Return the fixed reason V4 physical evidence cannot enter a route gate."""

    return {
        "schema": ROUTING_NOT_APPLICABLE_SCHEMA,
        "status": "NOT_APPLICABLE",
        "included_in_g2": False,
        "reason": (
            "v4_outputs_observable_physical_occupancy_not_body_inflated_"
            "configuration_space"
        ),
        "deferred_to": (
            "g3_post_memory_multi_view_fusion_then_registered_"
            "physical_to_configuration_projection"
        ),
    }


def _validated_output(output: object) -> tuple[np.ndarray, np.ndarray]:
    parsed = _require_exact_keys(output, _INFERENCE_OUTPUT_KEYS, "inference output")
    if parsed["schema"] != INFERENCE_OUTPUT_SCHEMA:
        raise ValueError("V4 inference-output schema changed")
    if parsed["class_order"] != list(CLASS_ORDER):
        raise ValueError("V4 physical class order changed")
    if parsed["action_order"] != list(ACTION_ORDER):
        raise ValueError("V4 action order changed")
    logits = _numeric_array(
        parsed["physical_logits_f32_chw"],
        name="physical_logits_f32_chw",
        shape=(3, 64, 64),
    )
    probabilities = _numeric_array(
        parsed["physical_probabilities_f32_chw"],
        name="physical_probabilities_f32_chw",
        shape=(3, 64, 64),
    )
    _numeric_array(
        parsed["all_action_survival_logits_f32"],
        name="all_action_survival_logits_f32",
        shape=(9, 16),
    )
    anchor = np.asarray(parsed["anchor_in_frustum_bool_hw"])
    if anchor.shape != (64, 64) or anchor.dtype.kind != "b":
        raise TypeError("anchor_in_frustum_bool_hw must have exact bool shape (64,64)")
    if (probabilities < 0.0).any() or (probabilities > 1.0).any():
        raise ValueError("V4 output probabilities lie outside [0,1]")
    if not np.allclose(probabilities.sum(axis=0), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("V4 output probabilities do not sum to one")
    shifted = logits - logits.max(axis=0, keepdims=True)
    expected_probabilities = np.exp(shifted)
    expected_probabilities /= expected_probabilities.sum(axis=0, keepdims=True)
    if not np.allclose(
        probabilities,
        expected_probabilities,
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError("V4 probabilities do not match the supplied logits")
    invisible = ~anchor
    if invisible.any():
        expected_invalid = np.asarray((0.0, -20.0, -20.0))[:, None]
        if not np.array_equal(
            logits[:, invisible],
            np.broadcast_to(expected_invalid, logits[:, invisible].shape),
        ):
            raise ValueError("V4 invisible-cell logits changed")
    return probabilities, anchor


def score_physical_evidence_batch(
    outputs: Sequence[Mapping[str, Any]],
    labels: object,
    distances_m: object,
    *,
    thresholds: TraversabilityThresholds,
    evaluation_mask: object | None = None,
) -> dict[str, Any]:
    """Score raw physical evidence without claiming calibration or routing."""

    if not outputs:
        raise ValueError("physical-evidence scoring requires at least one output")
    if not isinstance(thresholds, TraversabilityThresholds):
        raise TypeError("thresholds must be TraversabilityThresholds")

    validated = [_validated_output(output) for output in outputs]
    probabilities = np.stack([row[0] for row in validated], axis=0)
    first_anchor = validated[0][1]
    if any(not np.array_equal(anchor, first_anchor) for _, anchor in validated[1:]):
        raise ValueError("V4 inference outputs disagree on fixed anchor visibility")

    raw_labels = np.asarray(labels)
    if raw_labels.dtype.kind not in {"i", "u"} or raw_labels.shape != (
        len(outputs),
        64,
        64,
    ):
        raise TypeError("labels must have exact integer shape (N,64,64)")
    label_array = np.asarray(raw_labels, dtype=np.int64)
    distance_array = _numeric_array(
        distances_m,
        name="distances_m",
        shape=(len(outputs), 64, 64),
    )
    mask_array: np.ndarray | None
    if evaluation_mask is None:
        mask_array = None
    else:
        mask_array = np.asarray(evaluation_mask)
        if mask_array.dtype.kind != "b" or mask_array.shape != label_array.shape:
            raise TypeError("evaluation_mask must have exact bool shape (N,64,64)")

    traversability = evaluate_traversability(
        probabilities,
        label_array,
        distance_array,
        thresholds=thresholds,
        evaluation_mask=mask_array,
        obstacle_range_m=2.0,
        calibration_bins=15,
    )
    raw_metrics = traversability.to_dict()
    physical = {
        "schema": PHYSICAL_EVIDENCE_METRICS_SCHEMA,
        "admitted_observable_physical_free_precision": float(
            traversability.planner_admitted_free_precision
        ),
        "directly_observable_physical_obstacle_recall_within_2m": float(
            traversability.obstacle_detection_recall_within_range
        ),
        "useful_observable_physical_free_recall": float(
            traversability.useful_traversable_recall
        ),
        "observable_physical_obstacle_exclusion_recall_within_2m": float(
            traversability.obstacle_exclusion_recall_within_range
        ),
        "unknown_evidence_admission_rate": float(
            traversability.unknown_admission_rate
        ),
        "free_probability_brier": float(traversability.free_probability_brier),
        "free_probability_ece": float(traversability.free_probability_ece),
    }
    checks = {
        # This adapter deliberately emits and scores raw softmax probabilities.
        # A later frozen calibration stage must own any positive calibration
        # receipt; a caller cannot turn this check on with a boolean argument.
        "heldout_probability_calibration_applied": False,
        "admitted_observable_physical_free_precision_ge_0_99": (
            physical["admitted_observable_physical_free_precision"] >= 0.99
        ),
        "directly_observable_physical_obstacle_recall_within_2m_ge_0_95": (
            physical[
                "directly_observable_physical_obstacle_recall_within_2m"
            ]
            >= 0.95
        ),
        "useful_observable_physical_free_recall_ge_0_90": (
            physical["useful_observable_physical_free_recall"] >= 0.90
        ),
    }
    gate = {
        "schema": PHYSICAL_G2_GATE_SCHEMA,
        "target_space": PHYSICAL_OCCUPANCY_TARGET_SPACE,
        "routing_included": False,
        "passes": all(checks.values()),
        "checks": checks,
    }
    return {
        "schema": PHYSICAL_EVIDENCE_BATCH_SCHEMA,
        "occupancy_target_space": PHYSICAL_OCCUPANCY_TARGET_SPACE,
        "class_order": list(CLASS_ORDER),
        "sample_count": len(outputs),
        "calibration": {
            "applied": False,
            "reason": "raw_v4_softmax_requires_frozen_development_role_calibration",
        },
        "thresholds": {
            "free_probability_min": float(thresholds.free_probability_min),
            "occupied_probability_max": float(thresholds.occupied_probability_max),
            "unknown_probability_max": float(thresholds.unknown_probability_max),
            "occupied_detection_min": float(thresholds.occupied_detection_min),
        },
        "traversability": raw_metrics,
        "physical_evidence": physical,
        "routing": routing_not_applicable_receipt(),
        "g2": gate,
    }


__all__ = [
    "ACTION_ORDER",
    "CHECKPOINT_SCHEMA",
    "CLASS_ORDER",
    "INFERENCE_INPUT_SCHEMA",
    "INFERENCE_OUTPUT_SCHEMA",
    "PHYSICAL_EVIDENCE_BATCH_SCHEMA",
    "PHYSICAL_OCCUPANCY_TARGET_SPACE",
    "ROUTING_NOT_APPLICABLE_SCHEMA",
    "infer_one",
    "load_checkpoint",
    "routing_not_applicable_receipt",
    "score_physical_evidence_batch",
]
