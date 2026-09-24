"""Strict CPU checkpoint boundary for V9 physical-evidence calibration."""
from __future__ import annotations

import hashlib
import io
import math
from typing import Any

import torch
import torch.nn as nn

from lewm.benchmarks.go2_rgb_swept_progress_survival_joint_jepa_v4_g2_adapter import (
    _AUXILIARY_OBJECTIVE as _INHERITED_OCCUPIED_AUXILIARY,
    _SEMANTIC_DECODER_ARCHITECTURE as _INHERITED_DECODER_ARCHITECTURE,
    _validate_initial_decoder_receipt,
)
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift import (
    GeometryAnchoredSweptProgressSurvivalJointJepaV9,
)


PHYSICAL_CALIBRATION_PREREGISTRATION_COMMIT = (
    "2f561d26f0b6ca154b6f4eab00dba228f8bc8c9e"
)
PHYSICAL_CALIBRATION_SOURCE_CLOSURE_AMENDMENT_COMMIT = (
    "b2465b2148b999b216078d53fe9bd556e63703e0"
)
V9_PREREGISTRATION_COMMIT = "47043472466e7a258ad0f0be854c05393e233db8"
V9_PREIMPLEMENTATION_AMENDMENT_COMMIT = (
    "04db6b26d46875297e3aa515fdf1d688bee2b755"
)
CHECKPOINT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v9_"
    "content_adaptive_dense_local_token_lift_checkpoint_v1"
)
SWEEP_MASK_STATE_KEY = "predictor.swept_progress_head.sweep_masks"

_ATTENTION_PARAMETER_SUFFIXES = (
    "query_projection.weight",
    "query_projection.bias",
    "key_projection.weight",
    "value_projection.weight",
    "value_projection.bias",
    "output_projection.weight",
    "output_projection.bias",
)
_ATTENTION_PARAMETER_INVENTORY_SHA256 = hashlib.sha256(
    "\n".join(_ATTENTION_PARAMETER_SUFFIXES).encode("utf-8")
).hexdigest()
_CHECKPOINT_KEYS = frozenset(
    {
        "schema",
        "development_only",
        "resume_authorized",
        "qualified",
        "preregistration_commit",
        "preimplementation_amendment_commit",
        "constructor_initialization_seed",
        "semantic_decoder_initialization_seed",
        "dense_local_attention_initialization_seed",
        "experiment_seed",
        "initialization_source",
        "predecessor_experiment_checkpoint_read",
        "inherited_occupied_auxiliary",
        "initial_v9_model",
        "dense_local_attention_activity",
        "training_diagnostics",
        "accounting",
        "model_state_dict",
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

_DENSE_LOCAL_LIFT_ARCHITECTURE = {
    "schema": "lewm_v9_content_adaptive_dense_local_token_lift_architecture_v1",
    "input": {
        "source": "unchanged_projected_final_patch_tokens",
        "normalized_rgb_shape_chw": [3, 112, 112],
        "final_patch_token_shape": [256, 192],
        "projected_token_lattice_shape_chw": [64, 16, 16],
    },
    "geometry": {
        "anchor_grid": "exact_inherited_fixed_camera_ground_projection",
        "ground_z_m": -0.333,
        "anchor_visibility": "exact_inherited_boolean_anchor_in_frustum",
        "support_side": 5,
        "support_count": 25,
        "center_index": 12,
        "legacy_config_samples_per_cell_retained_but_unused": 4,
        "offset_order": (
            "row_major_y_then_x_for_integer_x_y_in_minus2_through_plus2"
        ),
        "normalized_token_cell_step_xy": [0.125, 0.125],
    },
    "sampling": {
        "operator": "torch.nn.functional.grid_sample",
        "mode": "bilinear",
        "padding_mode": "zeros",
        "align_corners": False,
        "invalid_coordinate_xy": [2.0, 2.0],
        "reported_support_valid_mask_shape": [64, 64, 25],
        "reported_support_grid_xy_shape": [64, 64, 25, 2],
    },
    "attention": {
        "query": {
            "type": "Linear",
            "in_features": 64,
            "out_features": 64,
            "bias": True,
        },
        "key": {
            "type": "Linear",
            "in_features": 64,
            "out_features": 64,
            "bias": False,
        },
        "value": {
            "type": "Linear",
            "in_features": 64,
            "out_features": 64,
            "bias": True,
        },
        "output": {
            "type": "Linear",
            "in_features": 64,
            "out_features": 64,
            "bias": True,
        },
        "head_count": 4,
        "head_width": 16,
        "logit_scale": "1/sqrt(16)",
        "parameter_tensor_count": 7,
        "added_parameter_count_per_lift": 16_576,
        "initialization": {
            "private_cpu_generator_seed": 20_260_729,
            "weight_order": ["query", "key", "value", "output"],
            "weights": "xavier_uniform_gain_1",
            "biases": "exact_zero",
            "caller_rng_state_restored": True,
        },
        "excluded": [
            "layer_normalization",
            "feed_forward_network",
            "positional_bias",
            "learned_temperature",
            "learned_gate",
            "dropout",
            "prototype_bank",
            "per_cell_query_parameters",
        ],
    },
    "aggregation": (
        "center_sample_residual_plus_output_projected_dense_local_attention"
    ),
    "all_invalid_cells": (
        "excluded_from_attention_softmax_with_exact_zero_reported_weights_"
        "then_exact_inherited_null_evidence_before_consumers"
    ),
    "preserved": [
        "token_projection",
        "null_evidence",
        "refinement_blocks",
        "semantic_decoder",
        "action_conditioned_jepa_predictor",
        "online_and_ema_target_routes",
    ],
    "removed": ["raw_offsets", "weight_logits"],
}

_REMOVED_STATE_NAMES = sorted(
    {
        "bev_lift.raw_offsets",
        "bev_lift.weight_logits",
        "target_bev_lift.raw_offsets",
        "target_bev_lift.weight_logits",
    }
)
_ADDED_STATE_NAMES = sorted(
    {f"bev_lift.{name}" for name in _ATTENTION_PARAMETER_SUFFIXES}
    | {f"target_bev_lift.{name}" for name in _ATTENTION_PARAMETER_SUFFIXES}
    | {
        "bev_lift.support_offsets_token_cells",
        "target_bev_lift.support_offsets_token_cells",
    }
)
_INITIAL_SAMPLING_RECEIPT = {
    "type": "ContentAdaptiveDenseLocalTokenLiftSamplingV9",
    "latent_shape": [1, 64, 64, 64],
    "anchor_in_frustum_shape": [1, 64, 64],
    "support_valid_mask_shape": [1, 64, 64, 25],
    "cell_valid_mask_shape": [1, 64, 64],
    "support_grid_xy_shape": [1, 64, 64, 25, 2],
    "support_offsets_token_cells_shape": [25, 2],
    "attention_weights_shape": [1, 64, 64, 4, 25],
    "support_offset_order_bit_exact": True,
    "safe_invalid_grid_xy": [2.0, 2.0],
    "invalid_support_attention_exact_zero": True,
    "valid_attention_sums_one_per_head": True,
    "all_invalid_attention_exact_zero": True,
    "all_invalid_latent_exact_inherited_null_evidence": True,
}


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


def _require_exact(value: object, expected: object, name: str) -> None:
    """Compare nested executor receipts without bool/int type coercion."""

    if type(value) is not type(expected):
        raise TypeError(f"{name} type changed")
    if type(expected) is dict:
        actual_dict = value
        expected_dict = expected
        if set(actual_dict) != set(expected_dict):
            raise ValueError(f"{name} keys changed")
        for key in expected_dict:
            _require_exact(actual_dict[key], expected_dict[key], f"{name}.{key}")
        return
    if type(expected) is list:
        actual_list = value
        expected_list = expected
        if len(actual_list) != len(expected_list):
            raise ValueError(f"{name} length changed")
        for index, expected_item in enumerate(expected_list):
            _require_exact(actual_list[index], expected_item, f"{name}[{index}]")
        return
    if value != expected:
        raise ValueError(f"{name} changed")


def _validate_metadata(checkpoint: dict) -> None:
    expected_scalars = {
        "schema": CHECKPOINT_SCHEMA,
        "development_only": True,
        "resume_authorized": False,
        "qualified": False,
        "preregistration_commit": V9_PREREGISTRATION_COMMIT,
        "preimplementation_amendment_commit": (
            V9_PREIMPLEMENTATION_AMENDMENT_COMMIT
        ),
        "constructor_initialization_seed": 20_260_712,
        "semantic_decoder_initialization_seed": 20_260_713,
        "dense_local_attention_initialization_seed": 20_260_729,
        "experiment_seed": 20_260_728,
        "initialization_source": (
            "exact_n320_encoder_and_clean_v4_with_preregistered_lift_replacement"
        ),
        "predecessor_experiment_checkpoint_read": False,
    }
    for name, expected in expected_scalars.items():
        _require_exact(checkpoint[name], expected, f"checkpoint.{name}")
    _require_exact(
        checkpoint["inherited_occupied_auxiliary"],
        _INHERITED_OCCUPIED_AUXILIARY,
        "checkpoint.inherited_occupied_auxiliary",
    )
    _require_exact(
        checkpoint["accounting"],
        _TERMINAL_ACCOUNTING,
        "checkpoint.accounting",
    )


def _validate_state_mapping(value: object) -> dict[str, torch.Tensor]:
    if type(value) is not dict or not value:
        raise TypeError("model_state_dict must be a nonempty exact dict")
    state: dict[str, torch.Tensor] = value
    for name, tensor in state.items():
        if type(name) is not str or not name:
            raise TypeError("model state names must be nonempty exact strings")
        if type(tensor) is not torch.Tensor:
            raise TypeError(f"model state {name!r} is not an exact tensor")
        if tensor.device.type != "cpu":
            raise TypeError(f"model state {name!r} is not CPU-resident")
        if tensor.layout is not torch.strided or not tensor.is_contiguous():
            raise TypeError(f"model state {name!r} is not dense contiguous strided")
        if tensor.requires_grad:
            raise TypeError(f"model state {name!r} unexpectedly requires gradients")
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"model state {name!r} is nonfinite")
    return state


def _validate_initial_migration(value: object) -> None:
    expected_keys = frozenset(
        {
            "source",
            "removed_state_names",
            "added_state_names",
            "all_inherited_state_tensors_bit_exact",
            "inherited_state_tensor_count",
            "inherited_state_name_inventory_sha256",
            "removed_parameter_count_per_online_or_target_lift",
            "added_attention_parameter_count_per_online_or_target_lift",
            "added_attention_parameter_tensor_count_per_online_or_target_lift",
            "online_target_attention_initial_copy_exact",
            "online_target_support_offsets_initial_copy_exact",
            "target_attention_initial_gradient_tensor_count",
            "attention_initialization_bit_exact",
            "attention_biases_exact_zero",
            "key_projection_bias",
            "sampling_receipt",
            "caller_cpu_rng_state_restored",
        }
    )
    migration = _require_exact_keys(value, expected_keys, "initial_v9_model.migration")
    expected = {
        "source": "fresh clean V4 construction with identical N320 state and masks",
        "removed_state_names": _REMOVED_STATE_NAMES,
        "added_state_names": _ADDED_STATE_NAMES,
        "all_inherited_state_tensors_bit_exact": True,
        "inherited_state_tensor_count": 220,
        "inherited_state_name_inventory_sha256": (
            "55439423b29b61060e9a89279f0f19ecd4cf81cafb64bf3c15769a565647602c"
        ),
        "removed_parameter_count_per_online_or_target_lift": 49_152,
        "added_attention_parameter_count_per_online_or_target_lift": 16_576,
        "added_attention_parameter_tensor_count_per_online_or_target_lift": 7,
        "online_target_attention_initial_copy_exact": True,
        "online_target_support_offsets_initial_copy_exact": True,
        "target_attention_initial_gradient_tensor_count": 0,
        "attention_initialization_bit_exact": True,
        "attention_biases_exact_zero": True,
        "key_projection_bias": False,
        "sampling_receipt": _INITIAL_SAMPLING_RECEIPT,
        "caller_cpu_rng_state_restored": True,
    }
    _require_exact(migration, expected, "initial_v9_model.migration")


def _validate_initial_receipt(
    value: object, model: GeometryAnchoredSweptProgressSurvivalJointJepaV9
) -> None:
    expected_keys = frozenset(
        {
            "architecture",
            "migration",
            "inherited_v4_decoder",
            "online_attention_parameter_count",
            "online_attention_parameter_tensor_count",
            "target_attention_parameter_count",
            "target_attention_parameter_tensor_count",
            "attention_parameter_suffix_inventory_sha256",
            "all_online_attention_parameters_in_lift_semantic_exactly_once",
            "all_target_attention_parameters_frozen_in_target_exactly_once",
            "target_initial_copy_exact",
            "initial_hard_sync_count",
            "initial_ema_update_count",
        }
    )
    receipt = _require_exact_keys(value, expected_keys, "initial_v9_model")
    _require_exact(
        receipt["architecture"],
        _DENSE_LOCAL_LIFT_ARCHITECTURE,
        "initial_v9_model.architecture",
    )
    _validate_initial_migration(receipt["migration"])
    expected_scalars = {
        "online_attention_parameter_count": 16_576,
        "online_attention_parameter_tensor_count": 7,
        "target_attention_parameter_count": 16_576,
        "target_attention_parameter_tensor_count": 7,
        "attention_parameter_suffix_inventory_sha256": (
            _ATTENTION_PARAMETER_INVENTORY_SHA256
        ),
        "all_online_attention_parameters_in_lift_semantic_exactly_once": True,
        "all_target_attention_parameters_frozen_in_target_exactly_once": True,
        "target_initial_copy_exact": True,
        "initial_hard_sync_count": 1,
        "initial_ema_update_count": 0,
    }
    for name, expected in expected_scalars.items():
        _require_exact(receipt[name], expected, f"initial_v9_model.{name}")
    decoder = _require_exact_keys(
        receipt["inherited_v4_decoder"],
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
        "initial_v9_model.inherited_v4_decoder",
    )
    _require_exact(
        decoder["architecture"],
        _INHERITED_DECODER_ARCHITECTURE,
        "initial_v9_model.inherited_v4_decoder.architecture",
    )
    _validate_initial_decoder_receipt(
        decoder,
        model.bev_lift.anchor_in_frustum,
    )


def _finite_positive_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive float")
    return value


def _validate_attention_activity(value: object) -> dict:
    expected_keys = frozenset(
        {
            "schema",
            "update_count",
            "online_parameter_count",
            "online_parameter_tensor_count",
            "parameter_suffix_inventory_sha256",
            "all_online_parameter_tensors_active_by_update_2",
            "first_active_update",
            "latest_first_active_update",
            "active_update_count",
            "minimum_active_parameter_tensor_count",
            "maximum_active_parameter_tensor_count",
            "minimum_gradient_l2",
            "maximum_gradient_l2",
            "target_gradient_tensor_count",
        }
    )
    activity = _require_exact_keys(
        value, expected_keys, "dense_local_attention_activity"
    )
    expected = {
        "schema": "lewm_v9_dense_local_attention_training_activity_v1",
        "update_count": 1_000,
        "online_parameter_count": 16_576,
        "online_parameter_tensor_count": 7,
        "parameter_suffix_inventory_sha256": (
            _ATTENTION_PARAMETER_INVENTORY_SHA256
        ),
        "all_online_parameter_tensors_active_by_update_2": True,
        "active_update_count": 1_000,
        "minimum_active_parameter_tensor_count": 7,
        "maximum_active_parameter_tensor_count": 7,
        "target_gradient_tensor_count": 0,
    }
    for name, expected_value in expected.items():
        _require_exact(
            activity[name], expected_value, f"dense_local_attention_activity.{name}"
        )
    first = _require_exact_keys(
        activity["first_active_update"],
        frozenset(_ATTENTION_PARAMETER_SUFFIXES),
        "dense_local_attention_activity.first_active_update",
    )
    for name, update in first.items():
        if type(update) is not int or update not in (1, 2):
            raise ValueError(f"attention tensor {name} was not active by update 2")
    if (
        type(activity["latest_first_active_update"]) is not int
        or activity["latest_first_active_update"] != max(first.values())
    ):
        raise ValueError("latest_first_active_update is inconsistent")
    minimum = _finite_positive_float(
        activity["minimum_gradient_l2"], "minimum_gradient_l2"
    )
    maximum = _finite_positive_float(
        activity["maximum_gradient_l2"], "maximum_gradient_l2"
    )
    if minimum > maximum:
        raise ValueError("attention gradient range is reversed")
    return activity


def _validate_training_diagnostics(value: object, activity: dict) -> None:
    diagnostics = _require_exact_keys(
        value,
        frozenset(
            {
                "ranking_active_microbatch_count",
                "ranking_eligible_pair_count",
                "survival_supervised_decision_count",
                "minimum_gradient_l2",
                "maximum_gradient_l2",
                "dense_local_attention",
            }
        ),
        "training_diagnostics",
    )
    _require_exact(
        diagnostics["dense_local_attention"],
        activity,
        "training_diagnostics.dense_local_attention",
    )
    _require_exact_int(
        diagnostics["ranking_active_microbatch_count"],
        4_000,
        "ranking_active_microbatch_count",
    )
    for name in (
        "ranking_eligible_pair_count",
        "survival_supervised_decision_count",
    ):
        value = diagnostics[name]
        if type(value) is not int or value <= 0:
            raise ValueError(f"training_diagnostics.{name} must be positive integer")
    minimum = _require_exact_keys(
        diagnostics["minimum_gradient_l2"],
        frozenset({"encoder", "lift_semantic", "predictor"}),
        "training_diagnostics.minimum_gradient_l2",
    )
    maximum = _require_exact_keys(
        diagnostics["maximum_gradient_l2"],
        frozenset({"encoder", "lift_semantic", "predictor"}),
        "training_diagnostics.maximum_gradient_l2",
    )
    for name in minimum:
        low = _finite_positive_float(minimum[name], f"minimum_gradient_l2.{name}")
        high = _finite_positive_float(maximum[name], f"maximum_gradient_l2.{name}")
        if low > high:
            raise ValueError(f"training gradient range is reversed for {name}")


def _tensor_bit_exact(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.layout is right.layout is torch.strided
        and left.is_contiguous()
        and right.is_contiguous()
        and torch.equal(
            left.reshape(-1).view(torch.uint8),
            right.reshape(-1).view(torch.uint8),
        )
    )


def _validate_loaded_mechanism(
    model: GeometryAnchoredSweptProgressSurvivalJointJepaV9,
) -> None:
    lift = model.bev_lift
    projections = (
        lift.query_projection,
        lift.key_projection,
        lift.value_projection,
        lift.output_projection,
    )
    if any(
        type(projection) is not nn.Linear
        or projection.in_features != 64
        or projection.out_features != 64
        for projection in projections
    ) or tuple(projection.bias is not None for projection in projections) != (
        True,
        False,
        True,
        True,
    ):
        raise RuntimeError("loaded V9 Q/K/V/O mechanism changed")
    attention = {
        name: parameter
        for name, parameter in lift.named_parameters()
        if name.startswith(
            (
                "query_projection.",
                "key_projection.",
                "value_projection.",
                "output_projection.",
            )
        )
    }
    if tuple(attention) != _ATTENTION_PARAMETER_SUFFIXES or sum(
        parameter.numel() for parameter in attention.values()
    ) != 16_576:
        raise RuntimeError("loaded V9 attention inventory changed")
    expected_offsets = torch.tensor(
        [(float(x), float(y)) for y in range(-2, 3) for x in range(-2, 3)],
        dtype=torch.float32,
    )
    if not torch.equal(lift.support_offsets_token_cells, expected_offsets):
        raise RuntimeError("loaded V9 support offset order changed")

    counters = (model.target_hard_sync_count.clone(), model.ema_update_count.clone())
    with torch.inference_mode():
        sampling = lift.forward_with_sampling(torch.zeros((1, 256, 192)))
    if (
        tuple(sampling.support_valid_mask.shape) != (1, 64, 64, 25)
        or tuple(sampling.attention_weights.shape) != (1, 64, 64, 4, 25)
        or not bool(torch.isfinite(sampling.latent).all())
        or not bool(torch.isfinite(sampling.attention_weights).all())
    ):
        raise RuntimeError("loaded V9 sampling receipt changed")
    invalid_support = ~sampling.support_valid_mask[..., None, :].expand_as(
        sampling.attention_weights
    )
    if torch.count_nonzero(
        sampling.attention_weights.masked_select(invalid_support)
    ).item() != 0:
        raise RuntimeError("loaded V9 invalid support received attention")
    valid_sums = sampling.attention_weights.sum(dim=-1).masked_select(
        sampling.cell_valid_mask[..., None].expand(-1, -1, -1, 4)
    )
    if not torch.allclose(valid_sums, torch.ones_like(valid_sums), rtol=0.0, atol=1e-6):
        raise RuntimeError("loaded V9 valid attention does not sum to one")
    invalid_cells = ~sampling.cell_valid_mask
    if torch.count_nonzero(
        sampling.attention_weights.masked_select(
            invalid_cells[..., None, None].expand_as(sampling.attention_weights)
        )
    ).item() != 0:
        raise RuntimeError("loaded V9 all-invalid attention is not zero")
    expected_null = lift.null_evidence[None, :, None].expand(
        -1, -1, int(invalid_cells.sum().item())
    )
    invalid_latent = sampling.latent.masked_select(
        invalid_cells[:, None].expand_as(sampling.latent)
    ).reshape(1, 64, -1)
    if not torch.equal(invalid_latent, expected_null):
        raise RuntimeError("loaded V9 invalid latent is not inherited null evidence")
    if not all(
        torch.equal(before, after)
        for before, after in zip(
            counters,
            (model.target_hard_sync_count, model.ema_update_count),
            strict=True,
        )
    ):
        raise RuntimeError("V9 mechanism validation mutated target counters")


def load_checkpoint(
    encoded: bytes,
) -> GeometryAnchoredSweptProgressSurvivalJointJepaV9:
    """Validate and reconstruct the exact terminal V9 model on CPU."""

    if type(encoded) is not bytes or not encoded:
        raise TypeError("encoded checkpoint must be nonempty exact bytes")
    checkpoint = torch.load(
        io.BytesIO(encoded),
        map_location="cpu",
        weights_only=True,
    )
    checkpoint = _require_exact_keys(checkpoint, _CHECKPOINT_KEYS, "checkpoint")
    _validate_metadata(checkpoint)
    activity = _validate_attention_activity(
        checkpoint["dense_local_attention_activity"]
    )
    _validate_training_diagnostics(checkpoint["training_diagnostics"], activity)
    state = _validate_state_mapping(checkpoint["model_state_dict"])

    encoder_prefix = "encoder."
    encoder_state = {
        name[len(encoder_prefix) :]: tensor.detach().clone()
        for name, tensor in state.items()
        if name.startswith(encoder_prefix)
    }
    if not encoder_state:
        raise ValueError("V9 checkpoint lacks its online encoder state")
    sweep_masks = state.get(SWEEP_MASK_STATE_KEY)
    if type(sweep_masks) is not torch.Tensor:
        raise ValueError("V9 checkpoint lacks exact swept-progress masks")

    model = GeometryAnchoredSweptProgressSurvivalJointJepaV9(
        encoder_state,
        sweep_masks.detach().clone(),
    )
    model.load_state_dict(state, strict=True)
    loaded_state = model.state_dict()
    if set(loaded_state) != set(state):
        raise RuntimeError("strict V9 reconstruction changed state inventory")
    for name, expected in state.items():
        if not _tensor_bit_exact(loaded_state[name], expected):
            raise RuntimeError(f"strict V9 reconstruction changed tensor {name!r}")
    if (
        model.target_hard_sync_count.dtype != torch.long
        or model.ema_update_count.dtype != torch.long
        or tuple(model.target_hard_sync_count.shape) != ()
        or tuple(model.ema_update_count.shape) != ()
        or int(model.target_hard_sync_count.item()) != 1
        or int(model.ema_update_count.item()) != 1_000
    ):
        raise ValueError("V9 target-update counters disagree with terminal accounting")

    _validate_initial_receipt(checkpoint["initial_v9_model"], model)
    model.eval().requires_grad_(False)
    _validate_loaded_mechanism(model)
    tensors = tuple(model.parameters()) + tuple(model.buffers())
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise TypeError("V9 calibration model must remain CPU-resident")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("V9 calibration model was not completely frozen")
    if model.training or any(module.training for module in model.modules()):
        raise RuntimeError("V9 calibration model was not placed in evaluation mode")
    return model


__all__ = [
    "CHECKPOINT_SCHEMA",
    "PHYSICAL_CALIBRATION_PREREGISTRATION_COMMIT",
    "PHYSICAL_CALIBRATION_SOURCE_CLOSURE_AMENDMENT_COMMIT",
    "V9_PREIMPLEMENTATION_AMENDMENT_COMMIT",
    "V9_PREREGISTRATION_COMMIT",
    "load_checkpoint",
]
