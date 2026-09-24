"""Strict CPU checkpoint boundary for V10 physical-evidence calibration."""
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
)
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v10_projective_cell_volume_token_lift import (
    CELL_VOLUME_ATTENTION_ADDED_PARAMETER_COUNT_V10,
    CELL_VOLUME_ATTENTION_HEADS_V10,
    CELL_VOLUME_ATTENTION_HEAD_WIDTH_V10,
    CELL_VOLUME_ATTENTION_INITIALIZATION_SEED_V10,
    CELL_VOLUME_ATTENTION_PARAMETER_TENSOR_COUNT_V10,
    CELL_VOLUME_HEIGHTS_M_V10,
    CELL_VOLUME_HORIZONTAL_OFFSETS_M_V10,
    CELL_VOLUME_VALID_CELL_COUNT_V10,
    CELL_VOLUME_VALID_MASK_SHA256_V10,
    CELL_VOLUME_WITHIN_TWO_METERS_CELL_COUNT_V10,
    CELL_VOLUME_WITHIN_TWO_METERS_VALID_CELL_COUNT_V10,
    GeometryAnchoredSweptProgressSurvivalJointJepaV10,
    ProjectiveCellVolumeTokenLiftSamplingV10,
)


PHYSICAL_CALIBRATION_PREREGISTRATION_COMMIT = (
    "6bc4dca93daf0e220bbaa4fc524470addb880e21"
)
V10_PREREGISTRATION_COMMIT = "b9eaae6560c42e588c86fb8bf949cc95bd9e29e9"
CHECKPOINT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v10_"
    "projective_cell_volume_token_lift_checkpoint_v1"
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
        "constructor_initialization_seed",
        "semantic_decoder_initialization_seed",
        "cell_volume_attention_initialization_seed",
        "experiment_seed",
        "initialization_source",
        "predecessor_experiment_checkpoint_read",
        "objective",
        "inherited_occupied_auxiliary",
        "initial_v10_model",
        "cell_volume_attention_activity",
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

_CELL_VOLUME_LIFT_ARCHITECTURE = {
    "schema": "lewm_v10_projective_cell_volume_token_lift_architecture_v1",
    "predecessor": "fresh_v9_parameters_from_clean_v4_and_n320_encoder",
    "only_change_from_v9": "registered_3d_support_geometry_and_masked_mean_base",
    "input": {
        "source": "unchanged_projected_final_patch_tokens",
        "normalized_rgb_shape_chw": [3, 112, 112],
        "final_patch_token_shape": [256, 192],
        "projected_token_lattice_shape_chw": [64, 16, 16],
    },
    "geometry": {
        "horizontal_offsets_xy_m": [
            [0.0, 0.0],
            [-0.05, -0.05],
            [-0.05, 0.05],
            [0.05, -0.05],
            [0.05, 0.05],
        ],
        "heights_m": [-0.333, -0.133, 0.067, 0.267, 0.467],
        "order": "horizontal_major_then_height_ascending",
        "support_count": 25,
        "camera_origin_xyz_m": [0.326, 0.0, 0.043],
        "camera_mount_rpy_degrees": [0.0, 0.0, 0.0],
        "horizontal_fov_degrees": 78.323,
        "vertical_fov_degrees": 62.8370386364,
        "inclusive_near_m": 0.05,
        "cell_validity": "or_over_25_closed_frustum_support_bits",
        "cell_valid_count": 2_062,
        "cell_valid_mask_row_major_uint8_sha256": (
            "4ebbafb6d4dd5fb13b96df978abfa7b81bc2f879b2ba6dec2fcda38dec54e60b"
        ),
        "near_field_lte_2m_cell_count": 1_016,
        "near_field_lte_2m_valid_cell_count": 222,
    },
    "sampling": {
        "operator": "torch.nn.functional.grid_sample",
        "mode": "bilinear",
        "padding_mode": "zeros",
        "align_corners": False,
        "invalid_coordinate_xy": [2.0, 2.0],
    },
    "aggregation": {
        "base": "arithmetic_mean_of_valid_samples_with_invalid_exact_zero",
        "residual": "unchanged_v9_four_head_qkvo_attention_over_25_supports",
        "head_count": 4,
        "head_width": 16,
        "attention_initialization_seed": 20_260_729,
        "attention_parameter_tensor_count": 7,
        "attention_parameter_count_per_lift": 16_576,
    },
    "invalid_cells": (
        "inherited_null_evidence_after_initial_lift_and_each_refinement_block; "
        "semantic_logits_exact_(0,-20,-20)"
    ),
    "new_loss_or_head": False,
}

_REMOVED_GEOMETRY_BUFFER_NAMES = sorted(
    {
        "bev_lift.support_offsets_token_cells",
        "target_bev_lift.support_offsets_token_cells",
    }
)
_ADDED_GEOMETRY_BUFFER_NAMES = sorted(
    {
        f"{prefix}.{name}"
        for prefix in ("bev_lift", "target_bev_lift")
        for name in (
            "support_offsets_xy_m",
            "support_heights_m",
            "support_xyz_m",
            "support_grid_xy",
            "support_valid_mask",
            "cell_valid_mask",
        )
    }
)
_INITIAL_SAMPLING_RECEIPT = {
    "schema": "lewm_v10_projective_cell_volume_token_lift_sampling_audit_v1",
    "type": "ProjectiveCellVolumeTokenLiftSamplingV10",
    "cell_valid_count": 2_062,
    "cell_valid_mask_row_major_uint8_sha256": (
        "4ebbafb6d4dd5fb13b96df978abfa7b81bc2f879b2ba6dec2fcda38dec54e60b"
    ),
    "near_field_lte_2m_cell_count": 1_016,
    "near_field_lte_2m_valid_cell_count": 222,
    "support_order_bit_exact": True,
    "safe_invalid_grid_xy": [2.0, 2.0],
    "invalid_support_attention_exact_zero": True,
    "valid_attention_sums_one_per_head": True,
    "all_invalid_latent_exact_inherited_null_evidence": True,
    "all_invalid_semantic_logits_exact_unknown": True,
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
        "preregistration_commit": V10_PREREGISTRATION_COMMIT,
        "constructor_initialization_seed": 20_260_712,
        "semantic_decoder_initialization_seed": 20_260_713,
        "cell_volume_attention_initialization_seed": 20_260_729,
        "experiment_seed": 20_260_728,
        "initialization_source": (
            "exact_n320_encoder_and_fresh_v9_v4_with_only_preregistered_"
            "geometry_replacement"
        ),
        "predecessor_experiment_checkpoint_read": False,
        "objective": "S+P+U+R+O",
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


def _validate_nested_v9_migration(value: object) -> None:
    suffixes = _ATTENTION_PARAMETER_SUFFIXES
    expected = {
        "source": "fresh clean V4 construction with identical N320 state and masks",
        "removed_state_names": sorted(
            {
                "bev_lift.raw_offsets",
                "bev_lift.weight_logits",
                "target_bev_lift.raw_offsets",
                "target_bev_lift.weight_logits",
            }
        ),
        "added_state_names": sorted(
            {f"bev_lift.{name}" for name in suffixes}
            | {f"target_bev_lift.{name}" for name in suffixes}
            | {
                "bev_lift.support_offsets_token_cells",
                "target_bev_lift.support_offsets_token_cells",
            }
        ),
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
        "sampling_receipt": {
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
        },
    }
    _require_exact(value, expected, "initial_v10_model.migration.fresh_v9_clean_v4_migration")


def _validate_initial_migration(
    value: object, model: GeometryAnchoredSweptProgressSurvivalJointJepaV10
) -> None:
    expected_keys = frozenset(
        {
            "schema",
            "source",
            "all_v9_parameter_names_and_values_bit_exact",
            "inherited_parameter_tensor_count",
            "removed_geometry_buffer_names",
            "added_geometry_buffer_names",
            "all_common_buffers_bit_exact",
            "online_target_attention_initial_copy_exact",
            "target_attention_initial_gradient_tensor_count",
            "attention_initialization_bit_exact",
            "sampling_receipt",
            "fresh_v9_clean_v4_migration",
            "caller_cpu_rng_state_restored",
        }
    )
    migration = _require_exact_keys(value, expected_keys, "initial_v10_model.migration")
    expected = {
        "schema": "lewm_v10_projective_cell_volume_token_lift_migration_v1",
        "source": "fresh V9 and clean V4 construction with identical N320 state",
        "all_v9_parameter_names_and_values_bit_exact": True,
        "inherited_parameter_tensor_count": len(tuple(model.named_parameters())),
        "removed_geometry_buffer_names": _REMOVED_GEOMETRY_BUFFER_NAMES,
        "added_geometry_buffer_names": _ADDED_GEOMETRY_BUFFER_NAMES,
        "all_common_buffers_bit_exact": True,
        "online_target_attention_initial_copy_exact": True,
        "target_attention_initial_gradient_tensor_count": 0,
        "attention_initialization_bit_exact": True,
        "sampling_receipt": _INITIAL_SAMPLING_RECEIPT,
        "caller_cpu_rng_state_restored": True,
    }
    for name, expected_value in expected.items():
        _require_exact(migration[name], expected_value, f"initial_v10_model.migration.{name}")
    _validate_nested_v9_migration(migration["fresh_v9_clean_v4_migration"])


def _validate_initial_receipt(
    value: object, model: GeometryAnchoredSweptProgressSurvivalJointJepaV10
) -> None:
    expected_keys = frozenset(
        {
            "schema",
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
            "attention_receipt_source",
        }
    )
    receipt = _require_exact_keys(value, expected_keys, "initial_v10_model")
    _require_exact(
        receipt["architecture"],
        _CELL_VOLUME_LIFT_ARCHITECTURE,
        "initial_v10_model.architecture",
    )
    _validate_initial_migration(receipt["migration"], model)
    expected_scalars = {
        "schema": "lewm_v10_projective_cell_volume_token_lift_initial_model_v1",
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
        "attention_receipt_source": (
            "unchanged_v9_qkvo_inventory_and_initialization_audit"
        ),
    }
    for name, expected in expected_scalars.items():
        _require_exact(receipt[name], expected, f"initial_v10_model.{name}")
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
                "decoder_parameters_changed_from_v4",
                "semantic_mask_route_changed_from_v9",
            }
        ),
        "initial_v10_model.inherited_v4_decoder",
    )
    _require_exact(
        decoder["architecture"],
        _INHERITED_DECODER_ARCHITECTURE,
        "initial_v10_model.inherited_v4_decoder.architecture",
    )
    expected_decoder = {
        "initial_residual_output_exactly_zero": True,
        "semantic_parameter_count": 37_318,
        "added_parameter_count": 37_123,
        "all_semantic_parameters_in_lift_semantic_exactly_once": True,
        "decoder_parameters_changed_from_v4": False,
        "semantic_mask_route_changed_from_v9": True,
    }
    for name, expected_value in expected_decoder.items():
        _require_exact(decoder[name], expected_value, f"initial_v10_model.inherited_v4_decoder.{name}")
    visibility = _require_exact_keys(
        decoder["visibility_mask"],
        frozenset(
            {"schema", "shape", "dtype", "true_cell_count", "sha256", "application", "invalid_logits"}
        ),
        "initial_v10_model.inherited_v4_decoder.visibility_mask",
    )
    expected_visibility = {
        "schema": "lewm_v10_cell_volume_semantic_validity_mask_v1",
        "shape": [64, 64],
        "dtype": "bool",
        "true_cell_count": CELL_VOLUME_VALID_CELL_COUNT_V10,
        "sha256": CELL_VOLUME_VALID_MASK_SHA256_V10,
        "application": "v10_post_decoder_cell_volume_validity",
        "invalid_logits": [0.0, -20.0, -20.0],
    }
    _require_exact(visibility, expected_visibility, "initial_v10_model.inherited_v4_decoder.visibility_mask")
    cell_valid = model.bev_lift.cell_valid_mask.detach().cpu().contiguous()
    if (
        int(cell_valid.sum().item()) != CELL_VOLUME_VALID_CELL_COUNT_V10
        or hashlib.sha256(cell_valid.to(torch.uint8).numpy().tobytes()).hexdigest()
        != CELL_VOLUME_VALID_MASK_SHA256_V10
    ):
        raise ValueError("V10 semantic cell-valid mask changed")


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
            "implementation",
        }
    )
    activity = _require_exact_keys(
        value, expected_keys, "cell_volume_attention_activity"
    )
    expected = {
        "schema": "lewm_v10_cell_volume_attention_training_activity_v1",
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
        "implementation": "unchanged_v9_attention_gradient_receipts",
    }
    for name, expected_value in expected.items():
        _require_exact(
            activity[name], expected_value, f"cell_volume_attention_activity.{name}"
        )
    first = _require_exact_keys(
        activity["first_active_update"],
        frozenset(_ATTENTION_PARAMETER_SUFFIXES),
        "cell_volume_attention_activity.first_active_update",
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
                "v10_contract",
            }
        ),
        "training_diagnostics",
    )
    _require_exact(
        diagnostics["dense_local_attention"],
        activity,
        "training_diagnostics.dense_local_attention",
    )
    _require_exact(
        diagnostics["v10_contract"],
        {
            "schema": "lewm_v10_unchanged_joint_training_contract_v1",
            "objective": "S+P+U+R+O",
            "occupied_auxiliary_coefficient": 0.5,
            "new_loss_or_head": False,
            "training_core": "unchanged_v9_wrapper_over_v3_v4",
        },
        "training_diagnostics.v10_contract",
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
    model: GeometryAnchoredSweptProgressSurvivalJointJepaV10,
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
        raise RuntimeError("loaded V10 Q/K/V/O mechanism changed")
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
    ) != CELL_VOLUME_ATTENTION_ADDED_PARAMETER_COUNT_V10:
        raise RuntimeError("loaded V10 attention inventory changed")
    expected_offsets = torch.tensor(CELL_VOLUME_HORIZONTAL_OFFSETS_M_V10)
    expected_heights = torch.tensor(CELL_VOLUME_HEIGHTS_M_V10)
    if not torch.equal(lift.support_offsets_xy_m, expected_offsets) or not torch.equal(
        lift.support_heights_m, expected_heights
    ):
        raise RuntimeError("loaded V10 support order changed")
    if (
        tuple(lift.support_xyz_m.shape) != (64, 64, 25, 3)
        or tuple(lift.support_grid_xy.shape) != (64, 64, 25, 2)
        or tuple(lift.support_valid_mask.shape) != (64, 64, 25)
        or tuple(lift.cell_valid_mask.shape) != (64, 64)
        or not torch.equal(lift.cell_valid_mask, lift.support_valid_mask.any(dim=-1))
        or int(lift.cell_valid_mask.sum().item()) != CELL_VOLUME_VALID_CELL_COUNT_V10
        or hashlib.sha256(
            lift.cell_valid_mask.to(torch.uint8).contiguous().numpy().tobytes()
        ).hexdigest()
        != CELL_VOLUME_VALID_MASK_SHA256_V10
    ):
        raise RuntimeError("loaded V10 static cell-volume geometry changed")
    invalid_static_grid = lift.support_grid_xy[~lift.support_valid_mask]
    if not torch.equal(invalid_static_grid, torch.full_like(invalid_static_grid, 2.0)):
        raise RuntimeError("loaded V10 invalid grid coordinate changed")
    near = lift.bev_ground_xyz_m[..., :2].square().sum(dim=-1) <= 4.0
    if (
        int(near.sum().item()) != CELL_VOLUME_WITHIN_TWO_METERS_CELL_COUNT_V10
        or int((near & lift.cell_valid_mask).sum().item())
        != CELL_VOLUME_WITHIN_TWO_METERS_VALID_CELL_COUNT_V10
    ):
        raise RuntimeError("loaded V10 near-field geometry changed")
    for name in (
        "support_offsets_xy_m",
        "support_heights_m",
        "support_xyz_m",
        "support_grid_xy",
        "support_valid_mask",
        "cell_valid_mask",
    ):
        if not torch.equal(getattr(lift, name), getattr(model.target_bev_lift, name)):
            raise RuntimeError(f"loaded V10 target geometry differs for {name}")

    counters = (model.target_hard_sync_count.clone(), model.ema_update_count.clone())
    with torch.inference_mode():
        sampling = lift.forward_with_sampling(torch.zeros((1, 256, 192)))
        raw_logits = model.semantic_head(sampling.latent)
        semantic_logits = model.semantic_logits_from_latent(sampling.latent)
    if (
        type(sampling) is not ProjectiveCellVolumeTokenLiftSamplingV10
        or tuple(sampling.latent.shape) != (1, 64, 64, 64)
        or tuple(sampling.anchor_in_frustum.shape) != (1, 64, 64)
        or tuple(sampling.support_valid_mask.shape) != (1, 64, 64, 25)
        or tuple(sampling.cell_valid_mask.shape) != (1, 64, 64)
        or tuple(sampling.support_grid_xy.shape) != (1, 64, 64, 25, 2)
        or tuple(sampling.support_xyz_m.shape) != (64, 64, 25, 3)
        or tuple(sampling.masked_mean.shape) != (1, 64, 64, 64)
        or tuple(sampling.attention_weights.shape) != (1, 64, 64, 4, 25)
        or not bool(torch.isfinite(sampling.latent).all())
        or not bool(torch.isfinite(sampling.masked_mean).all())
        or not bool(torch.isfinite(sampling.attention_weights).all())
    ):
        raise RuntimeError("loaded V10 sampling receipt changed")
    if (
        not torch.equal(sampling.cell_valid_mask[0], lift.cell_valid_mask)
        or not torch.equal(sampling.support_valid_mask[0], lift.support_valid_mask)
        or not torch.equal(sampling.support_grid_xy[0], lift.support_grid_xy)
        or not torch.equal(sampling.support_xyz_m, lift.support_xyz_m)
    ):
        raise RuntimeError("loaded V10 dynamic/static geometry receipt differs")
    invalid_support = ~sampling.support_valid_mask[..., None, :].expand_as(
        sampling.attention_weights
    )
    if torch.count_nonzero(
        sampling.attention_weights.masked_select(invalid_support)
    ).item() != 0:
        raise RuntimeError("loaded V10 invalid support received attention")
    valid_sums = sampling.attention_weights.sum(dim=-1).masked_select(
        sampling.cell_valid_mask[..., None].expand(-1, -1, -1, 4)
    )
    if not torch.allclose(valid_sums, torch.ones_like(valid_sums), rtol=0.0, atol=1e-6):
        raise RuntimeError("loaded V10 valid attention does not sum to one")
    invalid_cells = ~sampling.cell_valid_mask
    if torch.count_nonzero(
        sampling.attention_weights.masked_select(
            invalid_cells[..., None, None].expand_as(sampling.attention_weights)
        )
    ).item() != 0:
        raise RuntimeError("loaded V10 all-invalid attention is not zero")
    expected_null = lift.null_evidence[None, :, None].expand(
        -1, -1, int(invalid_cells.sum().item())
    )
    invalid_latent = sampling.latent.masked_select(
        invalid_cells[:, None].expand_as(sampling.latent)
    ).reshape(1, 64, -1)
    if not torch.equal(invalid_latent, expected_null):
        raise RuntimeError("loaded V10 invalid latent is not inherited null evidence")
    if torch.count_nonzero(
        sampling.masked_mean.masked_select(invalid_cells[..., None].expand_as(sampling.masked_mean))
    ).item() != 0:
        raise RuntimeError("loaded V10 invalid masked mean is not exact zero")

    semantic_valid = sampling.cell_valid_mask[:, None].expand_as(semantic_logits)
    if not torch.equal(
        semantic_logits.masked_select(semantic_valid),
        raw_logits.masked_select(semantic_valid),
    ):
        raise RuntimeError("loaded V10 cell-valid logits differ from raw decoder")
    newly_valid = sampling.cell_valid_mask & ~sampling.anchor_in_frustum
    if not bool(newly_valid.any()) or not torch.equal(
        semantic_logits.masked_select(newly_valid[:, None].expand_as(semantic_logits)),
        raw_logits.masked_select(newly_valid[:, None].expand_as(raw_logits)),
    ):
        raise RuntimeError("loaded V10 volume-valid ground-hidden logits were masked")
    invalid_logits = semantic_logits.masked_select(
        invalid_cells[:, None].expand_as(semantic_logits)
    ).reshape(1, 3, -1)
    exact_unknown = semantic_logits.new_tensor((0.0, -20.0, -20.0))[
        None, :, None
    ].expand_as(invalid_logits)
    if not torch.equal(invalid_logits, exact_unknown):
        raise RuntimeError("loaded V10 cell-invalid logits are not exact UNKNOWN")
    if not all(
        torch.equal(before, after)
        for before, after in zip(
            counters,
            (model.target_hard_sync_count, model.ema_update_count),
            strict=True,
        )
    ):
        raise RuntimeError("V10 mechanism validation mutated target counters")


def load_checkpoint(
    encoded: bytes,
) -> GeometryAnchoredSweptProgressSurvivalJointJepaV10:
    """Validate and reconstruct the exact terminal V10 model on CPU."""

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
        checkpoint["cell_volume_attention_activity"]
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
        raise ValueError("V10 checkpoint lacks its online encoder state")
    sweep_masks = state.get(SWEEP_MASK_STATE_KEY)
    if type(sweep_masks) is not torch.Tensor:
        raise ValueError("V10 checkpoint lacks exact swept-progress masks")

    model = GeometryAnchoredSweptProgressSurvivalJointJepaV10(
        encoder_state,
        sweep_masks.detach().clone(),
    )
    model.load_state_dict(state, strict=True)
    loaded_state = model.state_dict()
    if set(loaded_state) != set(state):
        raise RuntimeError("strict V10 reconstruction changed state inventory")
    for name, expected in state.items():
        if not _tensor_bit_exact(loaded_state[name], expected):
            raise RuntimeError(f"strict V10 reconstruction changed tensor {name!r}")
    if (
        model.target_hard_sync_count.dtype != torch.long
        or model.ema_update_count.dtype != torch.long
        or tuple(model.target_hard_sync_count.shape) != ()
        or tuple(model.ema_update_count.shape) != ()
        or int(model.target_hard_sync_count.item()) != 1
        or int(model.ema_update_count.item()) != 1_000
    ):
        raise ValueError("V10 target-update counters disagree with terminal accounting")

    _validate_initial_receipt(checkpoint["initial_v10_model"], model)
    model.eval().requires_grad_(False)
    _validate_loaded_mechanism(model)
    tensors = tuple(model.parameters()) + tuple(model.buffers())
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise TypeError("V10 calibration model must remain CPU-resident")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("V10 calibration model was not completely frozen")
    if model.training or any(module.training for module in model.modules()):
        raise RuntimeError("V10 calibration model was not placed in evaluation mode")
    return model


__all__ = [
    "CHECKPOINT_SCHEMA",
    "PHYSICAL_CALIBRATION_PREREGISTRATION_COMMIT",
    "V10_PREREGISTRATION_COMMIT",
    "load_checkpoint",
]
