"""Strict CPU checkpoint boundary for V12 physical-evidence calibration."""
from __future__ import annotations

import hashlib
import io
import math
from typing import Any, Sequence

import torch
import torch.nn as nn

from lewm.benchmarks.go2_rgb_swept_progress_survival_joint_jepa_v4_g2_adapter import (
    _AUXILIARY_OBJECTIVE as _INHERITED_OCCUPIED_AUXILIARY,
)
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v12_neutral_disjoint_ternary_competition import (
    CELL_VOLUME_HEIGHTS_M_V10,
    CELL_VOLUME_HORIZONTAL_OFFSETS_M_V10,
    CELL_VOLUME_VALID_CELL_COUNT_V10,
    CELL_VOLUME_VALID_MASK_SHA256_V10,
    CELL_VOLUME_WITHIN_TWO_METERS_CELL_COUNT_V10,
    CELL_VOLUME_WITHIN_TWO_METERS_VALID_CELL_COUNT_V10,
    ELEVATED_ONLY_VALID_CELL_COUNT_V11,
    ELEVATED_SUPPORT_INDICES_V11,
    ELEVATED_VALID_CELL_COUNT_V11,
    ELEVATED_VALID_MASK_SHA256_V11,
    ELEVATED_WITHIN_TWO_METERS_VALID_CELL_COUNT_V11,
    FLOOR_SUPPORT_INDICES_V11,
    FLOOR_VALID_CELL_COUNT_V11,
    FLOOR_VALID_MASK_SHA256_V11,
    FLOOR_WITHIN_TWO_METERS_VALID_CELL_COUNT_V11,
    HEIGHT_ROLE_ATTENTION_PARAMETER_COUNT_V11,
    HEIGHT_ROLE_ATTENTION_PARAMETER_SUFFIXES_V11,
    HEIGHT_ROLE_ATTENTION_PARAMETER_TENSOR_COUNT_V11,
    HEIGHT_ROLE_INITIALIZATION_SEED_V11,
    HEIGHT_ROLE_SEMANTIC_PARAMETER_COUNT_V11,
    HEIGHT_ROLE_SEMANTIC_PARAMETER_SUFFIXES_V11,
    HEIGHT_ROLE_SEMANTIC_PARAMETER_TENSOR_COUNT_V11,
    GeometryAnchoredSweptProgressSurvivalJointJepaV12,
    HeightRoleFactorizedEvidenceLiftSamplingV11,
    HeightRoleFactorizedEvidenceLiftV11,
    HeightRoleNeutralDisjointTernarySemanticDecoderV12,
    neutral_disjoint_ternary_log_probabilities_v12,
)


PHYSICAL_CALIBRATION_PREREGISTRATION_COMMIT = (
    "c63e98162a1b03a33225e6e0a04b67a357c7ed89"
)
V12_PREREGISTRATION_COMMIT = "ae1568e8f434d715d379eefc3eaf644369154f76"
CHECKPOINT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v12_"
    "neutral_disjoint_ternary_competition_checkpoint_v1"
)
SWEEP_MASK_STATE_KEY = "predictor.swept_progress_head.sweep_masks"

_BRANCH_ATTENTION_PARAMETER_SUFFIXES = tuple(
    HEIGHT_ROLE_ATTENTION_PARAMETER_SUFFIXES_V11
)
_SEMANTIC_AXIS_PARAMETER_SUFFIXES = tuple(
    HEIGHT_ROLE_SEMANTIC_PARAMETER_SUFFIXES_V11
)
_BRANCH_ATTENTION_PARAMETER_INVENTORY_SHA256 = hashlib.sha256(
    "\n".join(_BRANCH_ATTENTION_PARAMETER_SUFFIXES).encode("utf-8")
).hexdigest()
_SEMANTIC_AXIS_PARAMETER_INVENTORY_SHA256 = hashlib.sha256(
    "\n".join(_SEMANTIC_AXIS_PARAMETER_SUFFIXES).encode("utf-8")
).hexdigest()

_ONLINE_BRANCH_PARAMETER_NAMES = tuple(
    f"bev_lift.{suffix}" for suffix in _BRANCH_ATTENTION_PARAMETER_SUFFIXES
)
_TARGET_BRANCH_PARAMETER_NAMES = tuple(
    f"target_bev_lift.{suffix}" for suffix in _BRANCH_ATTENTION_PARAMETER_SUFFIXES
)
_SEMANTIC_AXIS_PARAMETER_NAMES = tuple(
    f"semantic_head.{suffix}" for suffix in _SEMANTIC_AXIS_PARAMETER_SUFFIXES
)
_V11_REPLACEMENT_PARAMETER_NAMES = frozenset(
    _ONLINE_BRANCH_PARAMETER_NAMES
    + _TARGET_BRANCH_PARAMETER_NAMES
    + _SEMANTIC_AXIS_PARAMETER_NAMES
)

_V10_ATTENTION_PARAMETER_SUFFIXES = (
    "query_projection.weight",
    "query_projection.bias",
    "key_projection.weight",
    "value_projection.weight",
    "value_projection.bias",
    "output_projection.weight",
    "output_projection.bias",
)
_V10_SEMANTIC_PARAMETER_SUFFIXES = (
    "base.weight",
    "base.bias",
    "local.weight",
    "local.bias",
    "residual_output.weight",
    "residual_output.bias",
)
_REMOVED_V10_PARAMETER_NAMES = sorted(
    {
        *(f"bev_lift.{suffix}" for suffix in _V10_ATTENTION_PARAMETER_SUFFIXES),
        *(
            f"target_bev_lift.{suffix}"
            for suffix in _V10_ATTENTION_PARAMETER_SUFFIXES
        ),
        *(f"semantic_head.{suffix}" for suffix in _V10_SEMANTIC_PARAMETER_SUFFIXES),
    }
)
_ROLE_BUFFER_SUFFIXES = frozenset(
    {
        "floor_support_role_mask",
        "elevated_support_role_mask",
        "floor_cell_valid_mask",
        "elevated_cell_valid_mask",
    }
)

_CHECKPOINT_KEYS = frozenset(
    {
        "schema",
        "development_only",
        "resume_authorized",
        "qualified",
        "preregistration_commit",
        "constructor_initialization_seed",
        "height_role_initialization_seed",
        "experiment_seed",
        "initialization_source",
        "predecessor_experiment_checkpoint_read",
        "objective",
        "inherited_occupied_auxiliary",
        "initial_v12_model",
        "height_role_branch_attention_activity",
        "factorized_semantic_axes_activity",
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

_NEUTRAL_DISJOINT_TERNARY_ARCHITECTURE = {
    "schema": "lewm_v12_neutral_disjoint_ternary_competition_architecture_v1",
    "predecessor": "fresh_v11_source_architecture_with_no_v11_runtime_reuse",
    "sole_change": "neutral_unknown_free_occupied_evidence_competition",
    "v11_parameter_or_buffer_change": False,
    "added_parameter_count": 0,
    "axis_inputs": {
        "free": {"latent_channels": [0, 32], "invalid_evidence": -20.0},
        "occupied": {"latent_channels": [32, 64], "invalid_evidence": -20.0},
    },
    "supported_cell_logits": {
        "unknown": "0",
        "free": "f",
        "occupied": "o",
        "normalization": "log_softmax",
    },
    "all_invalid_logits": [0.0, -20.0, -20.0],
    "objective": "S+P+U+R+O",
    "occupied_auxiliary_coefficient": 0.5,
    "new_loss_or_loss_weight": False,
    "predictor_consumes_shared_role_ordered_64_channel_state": True,
}

_V11_SAMPLING_RECEIPT = {
    "schema": "lewm_v11_height_role_factorized_evidence_sampling_audit_v1",
    "support_roles_disjoint_and_exhaustive": True,
    "floor_support_indices": list(FLOOR_SUPPORT_INDICES_V11),
    "elevated_support_indices": list(ELEVATED_SUPPORT_INDICES_V11),
    "floor_valid_cell_count": FLOOR_VALID_CELL_COUNT_V11,
    "floor_valid_mask_row_major_uint8_sha256": FLOOR_VALID_MASK_SHA256_V11,
    "elevated_valid_cell_count": ELEVATED_VALID_CELL_COUNT_V11,
    "elevated_valid_mask_row_major_uint8_sha256": ELEVATED_VALID_MASK_SHA256_V11,
    "role_valid_overlap_cell_count": FLOOR_VALID_CELL_COUNT_V11,
    "elevated_only_cell_count": ELEVATED_ONLY_VALID_CELL_COUNT_V11,
    "near_field_floor_valid_cell_count": FLOOR_WITHIN_TWO_METERS_VALID_CELL_COUNT_V11,
    "near_field_elevated_valid_cell_count": ELEVATED_WITHIN_TWO_METERS_VALID_CELL_COUNT_V11,
    "invalid_and_cross_role_attention_exact_zero": True,
    "valid_attention_sums_one_per_head": True,
    "shared_role_ordered_latent_shape": [1, 64, 64, 64],
    "finite_normalized_three_class_log_probabilities": True,
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
    """Compare nested receipts without bool/integer coercion."""

    if type(value) is not type(expected):
        raise TypeError(f"{name} type changed")
    if type(expected) is dict:
        actual_dict = value
        expected_dict = expected
        if set(actual_dict) != set(expected_dict):
            raise ValueError(f"{name} keys changed")
        for key, expected_value in expected_dict.items():
            _require_exact(actual_dict[key], expected_value, f"{name}.{key}")
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


def _names_sha256(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _mask_sha256(mask: torch.Tensor) -> str:
    return hashlib.sha256(
        mask.detach().to(device="cpu", dtype=torch.uint8).contiguous().numpy().tobytes()
    ).hexdigest()


def _finite_positive_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive float")
    return value


def _validate_metadata(checkpoint: dict) -> None:
    expected_scalars = {
        "schema": CHECKPOINT_SCHEMA,
        "development_only": True,
        "resume_authorized": False,
        "qualified": False,
        "preregistration_commit": V12_PREREGISTRATION_COMMIT,
        "constructor_initialization_seed": 20_260_712,
        "height_role_initialization_seed": HEIGHT_ROLE_INITIALIZATION_SEED_V11,
        "experiment_seed": 20_260_728,
        "initialization_source": (
            "accepted_n320_encoder_and_fresh_v11_source_state_with_only_"
            "zero_parameter_neutral_ternary_algebra"
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
    _require_exact(checkpoint["accounting"], _TERMINAL_ACCOUNTING, "checkpoint.accounting")


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


def _validate_activity(
    value: object,
    *,
    name: str,
    schema: str,
    suffixes: tuple[str, ...],
    parameter_count: int,
    target_parameter_tensor_count: int,
    minimum_active_parameter_tensor_count: int,
) -> dict:
    activity = _require_exact_keys(
        value,
        frozenset(
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
                "target_parameter_tensor_count",
                "target_gradient_tensor_count",
            }
        ),
        name,
    )
    expected = {
        "schema": schema,
        "update_count": 1_000,
        "online_parameter_count": parameter_count,
        "online_parameter_tensor_count": len(suffixes),
        "parameter_suffix_inventory_sha256": _names_sha256(suffixes),
        "all_online_parameter_tensors_active_by_update_2": True,
        "active_update_count": 1_000,
        "minimum_active_parameter_tensor_count": minimum_active_parameter_tensor_count,
        "maximum_active_parameter_tensor_count": len(suffixes),
        "target_parameter_tensor_count": target_parameter_tensor_count,
        "target_gradient_tensor_count": 0,
    }
    for field, expected_value in expected.items():
        _require_exact(activity[field], expected_value, f"{name}.{field}")
    first = _require_exact_keys(
        activity["first_active_update"], frozenset(suffixes), f"{name}.first_active_update"
    )
    for suffix, update in first.items():
        if type(update) is not int or update not in (1, 2):
            raise ValueError(f"{name} tensor {suffix} was not active by update 2")
    _require_exact(
        activity["latest_first_active_update"],
        max(first.values()),
        f"{name}.latest_first_active_update",
    )
    minimum = _finite_positive_float(
        activity["minimum_gradient_l2"], f"{name}.minimum_gradient_l2"
    )
    maximum = _finite_positive_float(
        activity["maximum_gradient_l2"], f"{name}.maximum_gradient_l2"
    )
    if minimum > maximum:
        raise ValueError(f"{name} gradient range is reversed")
    return activity


def _validate_training_diagnostics(
    value: object, branch_activity: dict, semantic_activity: dict
) -> None:
    diagnostics = _require_exact_keys(
        value,
        frozenset(
            {
                "ranking_active_microbatch_count",
                "ranking_eligible_pair_count",
                "survival_supervised_decision_count",
                "minimum_gradient_l2",
                "maximum_gradient_l2",
                "height_role_branch_attention",
                "factorized_semantic_axes",
                "v12_contract",
            }
        ),
        "training_diagnostics",
    )
    _require_exact(
        diagnostics["height_role_branch_attention"],
        branch_activity,
        "training_diagnostics.height_role_branch_attention",
    )
    _require_exact(
        diagnostics["factorized_semantic_axes"],
        semantic_activity,
        "training_diagnostics.factorized_semantic_axes",
    )
    contract = _require_exact_keys(
        diagnostics["v12_contract"],
        frozenset(
            {
                "schema",
                "training_helper",
                "objective",
                "occupied_auxiliary_coefficient",
                "new_loss_or_weight",
                "height_role_branch_attention",
                "factorized_semantic_axes",
            }
        ),
        "training_diagnostics.v12_contract",
    )
    expected_contract = {
        "schema": "lewm_v12_unchanged_joint_training_contract_v1",
        "training_helper": (
            "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v11_"
            "height_role_factorized_evidence_lift"
        ),
        "objective": "S+P+U+R+O",
        "occupied_auxiliary_coefficient": 0.5,
        "new_loss_or_weight": False,
    }
    for field, expected in expected_contract.items():
        _require_exact(contract[field], expected, f"training_diagnostics.v12_contract.{field}")
    _require_exact(
        contract["height_role_branch_attention"],
        branch_activity,
        "training_diagnostics.v12_contract.height_role_branch_attention",
    )
    _require_exact(
        contract["factorized_semantic_axes"],
        semantic_activity,
        "training_diagnostics.v12_contract.factorized_semantic_axes",
    )
    _require_exact_int(
        diagnostics["ranking_active_microbatch_count"],
        4_000,
        "training_diagnostics.ranking_active_microbatch_count",
    )
    for field in ("ranking_eligible_pair_count", "survival_supervised_decision_count"):
        observed = diagnostics[field]
        if type(observed) is not int or observed <= 0:
            raise ValueError(f"training_diagnostics.{field} must be a positive integer")
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
    for field in minimum:
        low = _finite_positive_float(minimum[field], f"minimum_gradient_l2.{field}")
        high = _finite_positive_float(maximum[field], f"maximum_gradient_l2.{field}")
        if low > high:
            raise ValueError(f"training gradient range is reversed for {field}")


def _validate_v11_migration(value: object, model: nn.Module) -> None:
    receipt = _require_exact_keys(
        value,
        frozenset(
            {
                "schema",
                "source",
                "predecessor_experiment_checkpoint_read",
                "all_common_v10_parameter_values_bit_exact",
                "all_common_v10_buffer_values_bit_exact",
                "inherited_state_name_inventory_sha256",
                "removed_v10_parameter_names",
                "added_role_buffer_names",
                "online_branch_attention_parameter_names",
                "target_branch_attention_parameter_names",
                "factorized_semantic_parameter_names",
                "online_branch_attention_parameter_count",
                "target_branch_attention_parameter_count",
                "factorized_semantic_parameter_count",
                "online_target_branch_attention_initial_copy_exact",
                "target_branch_attention_initial_gradient_tensor_count",
                "sampling_receipt",
            }
        ),
        "initial_v12_model.fresh_v11_state_identity.v11_source_migration_witness",
    )
    common_names = tuple(
        name for name, _ in model.named_parameters() if name not in _V11_REPLACEMENT_PARAMETER_NAMES
    )
    added_role_buffers = [
        name
        for name, _ in model.named_buffers()
        if name.rsplit(".", 1)[-1] in _ROLE_BUFFER_SUFFIXES
    ]
    expected = {
        "schema": "lewm_v11_height_role_factorized_evidence_lift_migration_v1",
        "source": "fresh V11 and fresh V10 from identical N320 encoder state",
        "predecessor_experiment_checkpoint_read": False,
        "all_common_v10_parameter_values_bit_exact": True,
        "all_common_v10_buffer_values_bit_exact": True,
        "inherited_state_name_inventory_sha256": _names_sha256(common_names),
        "removed_v10_parameter_names": _REMOVED_V10_PARAMETER_NAMES,
        "added_role_buffer_names": added_role_buffers,
        "online_branch_attention_parameter_names": list(_ONLINE_BRANCH_PARAMETER_NAMES),
        "target_branch_attention_parameter_names": list(_TARGET_BRANCH_PARAMETER_NAMES),
        "factorized_semantic_parameter_names": list(_SEMANTIC_AXIS_PARAMETER_NAMES),
        "online_branch_attention_parameter_count": HEIGHT_ROLE_ATTENTION_PARAMETER_COUNT_V11,
        "target_branch_attention_parameter_count": HEIGHT_ROLE_ATTENTION_PARAMETER_COUNT_V11,
        "factorized_semantic_parameter_count": HEIGHT_ROLE_SEMANTIC_PARAMETER_COUNT_V11,
        "online_target_branch_attention_initial_copy_exact": True,
        "target_branch_attention_initial_gradient_tensor_count": 0,
        "sampling_receipt": _V11_SAMPLING_RECEIPT,
    }
    _require_exact(receipt, expected, "initial_v12_model.fresh_v11_state_identity.v11_source_migration_witness")


def _validate_initial_receipt(
    value: object, model: GeometryAnchoredSweptProgressSurvivalJointJepaV12
) -> None:
    receipt = _require_exact_keys(
        value,
        frozenset(
            {
                "schema",
                "architecture",
                "fresh_v11_state_identity",
                "online_branch_attention_parameter_count",
                "online_branch_attention_parameter_tensor_count",
                "target_branch_attention_parameter_count",
                "target_branch_attention_parameter_tensor_count",
                "factorized_semantic_parameter_count",
                "factorized_semantic_parameter_tensor_count",
                "all_v11_parameters_partitioned_exactly_once",
                "optimizer_parameter_membership_changed_from_v11",
                "target_initial_gradient_tensor_count",
                "initial_hard_sync_count",
                "initial_ema_update_count",
            }
        ),
        "initial_v12_model",
    )
    expected_initial = {
        "schema": "lewm_v12_neutral_disjoint_ternary_initial_model_v1",
        "architecture": _NEUTRAL_DISJOINT_TERNARY_ARCHITECTURE,
        "online_branch_attention_parameter_count": HEIGHT_ROLE_ATTENTION_PARAMETER_COUNT_V11,
        "online_branch_attention_parameter_tensor_count": HEIGHT_ROLE_ATTENTION_PARAMETER_TENSOR_COUNT_V11,
        "target_branch_attention_parameter_count": HEIGHT_ROLE_ATTENTION_PARAMETER_COUNT_V11,
        "target_branch_attention_parameter_tensor_count": HEIGHT_ROLE_ATTENTION_PARAMETER_TENSOR_COUNT_V11,
        "factorized_semantic_parameter_count": HEIGHT_ROLE_SEMANTIC_PARAMETER_COUNT_V11,
        "factorized_semantic_parameter_tensor_count": HEIGHT_ROLE_SEMANTIC_PARAMETER_TENSOR_COUNT_V11,
        "all_v11_parameters_partitioned_exactly_once": True,
        "optimizer_parameter_membership_changed_from_v11": False,
        "target_initial_gradient_tensor_count": 0,
        "initial_hard_sync_count": 1,
        "initial_ema_update_count": 0,
    }
    for field, expected in expected_initial.items():
        _require_exact(receipt[field], expected, f"initial_v12_model.{field}")

    identity = _require_exact_keys(
        receipt["fresh_v11_state_identity"],
        frozenset(
            {
                "schema",
                "source",
                "predecessor_experiment_checkpoint_read",
                "v11_source_migration_witness",
                "parameter_name_inventory_sha256",
                "buffer_name_inventory_sha256",
                "state_name_inventory_sha256",
                "v12_parameter_tensor_count",
                "v11_parameter_tensor_count",
                "v12_parameter_count",
                "v11_parameter_count",
                "added_parameter_tensor_count",
                "added_parameter_count",
                "all_parameter_values_bit_exact",
                "all_buffer_values_bit_exact",
                "semantic_axis_modules_reused_without_aliasing",
                "neutral_algebra_exact",
                "supported_probabilities_finite_and_normalized",
                "branch_invalid_evidence_fixed_to_minus_20",
                "all_invalid_logits_exact",
                "shared_predictor_state_unchanged",
                "ema_target_state_unchanged_and_frozen",
            }
        ),
        "initial_v12_model.fresh_v11_state_identity",
    )
    parameters = tuple(model.named_parameters())
    buffers = tuple(model.named_buffers())
    expected_identity = {
        "schema": "lewm_v12_fresh_v11_zero_parameter_state_identity_v1",
        "source": "fresh V12 and fresh V11 from identical N320 encoder state",
        "predecessor_experiment_checkpoint_read": False,
        "parameter_name_inventory_sha256": _names_sha256(tuple(name for name, _ in parameters)),
        "buffer_name_inventory_sha256": _names_sha256(tuple(name for name, _ in buffers)),
        "state_name_inventory_sha256": _names_sha256(tuple(model.state_dict())),
        "v12_parameter_tensor_count": len(parameters),
        "v11_parameter_tensor_count": len(parameters),
        "v12_parameter_count": sum(parameter.numel() for _, parameter in parameters),
        "v11_parameter_count": sum(parameter.numel() for _, parameter in parameters),
        "added_parameter_tensor_count": 0,
        "added_parameter_count": 0,
        "all_parameter_values_bit_exact": True,
        "all_buffer_values_bit_exact": True,
        "semantic_axis_modules_reused_without_aliasing": True,
        "neutral_algebra_exact": True,
        "supported_probabilities_finite_and_normalized": True,
        "branch_invalid_evidence_fixed_to_minus_20": True,
        "all_invalid_logits_exact": True,
        "shared_predictor_state_unchanged": True,
        "ema_target_state_unchanged_and_frozen": True,
    }
    for field, expected in expected_identity.items():
        _require_exact(identity[field], expected, f"initial_v12_model.fresh_v11_state_identity.{field}")
    if len(parameters) != 233 or expected_identity["v12_parameter_count"] != 6_122_053:
        raise RuntimeError("loaded V12 parameter inventory changed")
    _validate_v11_migration(identity["v11_source_migration_witness"], model)


def _tensor_bit_exact(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.layout is right.layout is torch.strided
        and left.is_contiguous()
        and right.is_contiguous()
        and torch.equal(left.reshape(-1).view(torch.uint8), right.reshape(-1).view(torch.uint8))
    )


def _validate_projection(
    projection: object, *, in_features: int, out_features: int, bias: bool, name: str
) -> None:
    if (
        type(projection) is not nn.Linear
        or projection.in_features != in_features
        or projection.out_features != out_features
        or (projection.bias is not None) is not bias
    ):
        raise RuntimeError(f"loaded V12 projection changed: {name}")


def _validate_loaded_mechanism(
    model: GeometryAnchoredSweptProgressSurvivalJointJepaV12,
) -> None:
    lift = model.bev_lift
    if type(lift) is not HeightRoleFactorizedEvidenceLiftV11:
        raise RuntimeError("loaded V12 height-role lift type changed")
    if type(model.semantic_head) is not HeightRoleNeutralDisjointTernarySemanticDecoderV12:
        raise RuntimeError("loaded V12 neutral semantic wrapper type changed")
    for role in ("floor", "elevated"):
        for stem, in_features, out_features, bias in (
            ("query_projection", 64, 32, True),
            ("key_projection", 64, 32, False),
            ("value_projection", 64, 32, True),
            ("output_projection", 32, 32, True),
        ):
            name = f"{role}_{stem}"
            _validate_projection(
                getattr(lift, name),
                in_features=in_features,
                out_features=out_features,
                bias=bias,
                name=name,
            )
    attention = tuple(
        (name, parameter)
        for name, parameter in lift.named_parameters()
        if name in _BRANCH_ATTENTION_PARAMETER_SUFFIXES
    )
    if tuple(name for name, _ in attention) != _BRANCH_ATTENTION_PARAMETER_SUFFIXES or sum(
        parameter.numel() for _, parameter in attention
    ) != HEIGHT_ROLE_ATTENTION_PARAMETER_COUNT_V11:
        raise RuntimeError("loaded V12 role-attention inventory changed")
    semantic = tuple(model.semantic_head.named_parameters())
    if tuple(name for name, _ in semantic) != _SEMANTIC_AXIS_PARAMETER_SUFFIXES or sum(
        parameter.numel() for _, parameter in semantic
    ) != HEIGHT_ROLE_SEMANTIC_PARAMETER_COUNT_V11:
        raise RuntimeError("loaded V12 semantic-axis inventory changed")
    if set(map(id, model.semantic_head.free_axis.parameters())) & set(
        map(id, model.semantic_head.occupied_axis.parameters())
    ):
        raise RuntimeError("loaded V12 semantic axes alias")

    expected_floor = torch.zeros(25, dtype=torch.bool)
    expected_floor[list(FLOOR_SUPPORT_INDICES_V11)] = True
    expected_elevated = ~expected_floor
    if not torch.equal(lift.floor_support_role_mask, expected_floor) or not torch.equal(
        lift.elevated_support_role_mask, expected_elevated
    ):
        raise RuntimeError("loaded V12 support-role masks changed")
    if not torch.equal(lift.support_offsets_xy_m, torch.tensor(CELL_VOLUME_HORIZONTAL_OFFSETS_M_V10)) or not torch.equal(
        lift.support_heights_m, torch.tensor(CELL_VOLUME_HEIGHTS_M_V10)
    ):
        raise RuntimeError("loaded V12 support order changed")
    floor_valid = lift.floor_cell_valid_mask
    elevated_valid = lift.elevated_cell_valid_mask
    cell_valid = lift.cell_valid_mask
    if (
        int(cell_valid.sum()) != CELL_VOLUME_VALID_CELL_COUNT_V10
        or _mask_sha256(cell_valid) != CELL_VOLUME_VALID_MASK_SHA256_V10
        or int(floor_valid.sum()) != FLOOR_VALID_CELL_COUNT_V11
        or _mask_sha256(floor_valid) != FLOOR_VALID_MASK_SHA256_V11
        or int(elevated_valid.sum()) != ELEVATED_VALID_CELL_COUNT_V11
        or _mask_sha256(elevated_valid) != ELEVATED_VALID_MASK_SHA256_V11
        or not torch.equal(elevated_valid, cell_valid)
        or int((elevated_valid & ~floor_valid).sum()) != ELEVATED_ONLY_VALID_CELL_COUNT_V11
    ):
        raise RuntimeError("loaded V12 role-valid geometry changed")
    near = lift.bev_ground_xyz_m[..., :2].square().sum(dim=-1) <= 4.0
    if (
        int(near.sum()) != CELL_VOLUME_WITHIN_TWO_METERS_CELL_COUNT_V10
        or int((near & cell_valid).sum()) != CELL_VOLUME_WITHIN_TWO_METERS_VALID_CELL_COUNT_V10
        or int((near & floor_valid).sum()) != FLOOR_WITHIN_TWO_METERS_VALID_CELL_COUNT_V11
        or int((near & elevated_valid).sum()) != ELEVATED_WITHIN_TWO_METERS_VALID_CELL_COUNT_V11
    ):
        raise RuntimeError("loaded V12 near-field role geometry changed")
    for name in (
        "support_offsets_xy_m",
        "support_heights_m",
        "support_xyz_m",
        "support_grid_xy",
        "support_valid_mask",
        "cell_valid_mask",
        "floor_support_role_mask",
        "elevated_support_role_mask",
        "floor_cell_valid_mask",
        "elevated_cell_valid_mask",
    ):
        if not torch.equal(getattr(lift, name), getattr(model.target_bev_lift, name)):
            raise RuntimeError(f"loaded V12 target role geometry differs for {name}")

    counters = (model.target_hard_sync_count.clone(), model.ema_update_count.clone())
    with torch.inference_mode():
        sampling = lift.forward_with_sampling(torch.zeros((1, 256, 192)))
        free, occupied = model.semantic_head.evidence_logits(sampling.latent)
        expected_free = torch.where(
            sampling.floor_cell_valid_mask,
            free,
            torch.full_like(free, -20.0),
        )
        expected_occupied = torch.where(
            sampling.elevated_cell_valid_mask,
            occupied,
            torch.full_like(occupied, -20.0),
        )
        expected_logits = neutral_disjoint_ternary_log_probabilities_v12(
            expected_free, expected_occupied
        )
        invalid_logits = expected_logits.new_tensor((0.0, -20.0, -20.0))[
            None, :, None, None
        ]
        expected_logits = torch.where(
            sampling.cell_valid_mask[:, None], expected_logits, invalid_logits
        )
        observed_logits = model.semantic_logits_from_latent(sampling.latent)
    if (
        type(sampling) is not HeightRoleFactorizedEvidenceLiftSamplingV11
        or tuple(sampling.latent.shape) != (1, 64, 64, 64)
        or tuple(sampling.floor_attention_weights.shape) != (1, 64, 64, 2, 25)
        or tuple(sampling.elevated_attention_weights.shape) != (1, 64, 64, 2, 25)
        or not bool(torch.isfinite(sampling.latent).all())
        or not bool(torch.isfinite(sampling.floor_attention_weights).all())
        or not bool(torch.isfinite(sampling.elevated_attention_weights).all())
    ):
        raise RuntimeError("loaded V12 role sampling receipt changed")
    for weights, support_valid, valid in (
        (
            sampling.floor_attention_weights,
            sampling.floor_support_valid_mask,
            sampling.floor_cell_valid_mask,
        ),
        (
            sampling.elevated_attention_weights,
            sampling.elevated_support_valid_mask,
            sampling.elevated_cell_valid_mask,
        ),
    ):
        invalid_support = ~support_valid[..., None, :].expand_as(weights)
        if torch.count_nonzero(weights.masked_select(invalid_support)).item() != 0:
            raise RuntimeError("loaded V12 invalid or cross-role support received attention")
        valid_sums = weights.sum(dim=-1).masked_select(valid[..., None].expand(-1, -1, -1, 2))
        if not torch.allclose(valid_sums, torch.ones_like(valid_sums), rtol=0.0, atol=1e-6):
            raise RuntimeError("loaded V12 valid role attention does not sum to one")
    if not torch.equal(observed_logits, expected_logits):
        raise RuntimeError("loaded V12 neutral role-mask algebra changed")
    supported = observed_logits.permute(0, 2, 3, 1)[0][cell_valid]
    if not torch.allclose(
        torch.logsumexp(supported, dim=-1),
        torch.zeros_like(supported[:, 0]),
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError("loaded V12 supported probabilities are not normalized")
    observed_invalid = observed_logits.permute(0, 2, 3, 1)[0][~cell_valid]
    exact_invalid = observed_logits.new_tensor((0.0, -20.0, -20.0))[None].expand_as(
        observed_invalid
    )
    if not torch.equal(observed_invalid, exact_invalid):
        raise RuntimeError("loaded V12 all-invalid logits changed")
    probe_free = torch.tensor([[[-3.0, 4.0, 2.0, -2.0]]])
    probe_occupied = torch.tensor([[[-2.0, 1.0, 5.0, -3.0]]])
    observed_probe = neutral_disjoint_ternary_log_probabilities_v12(
        probe_free, probe_occupied
    )
    expected_probe = torch.log_softmax(
        torch.stack((torch.zeros_like(probe_free), probe_free, probe_occupied), dim=1),
        dim=1,
    )
    if not torch.equal(observed_probe, expected_probe) or not torch.equal(
        observed_probe.argmax(dim=1), torch.tensor([[[0, 1, 2, 0]]])
    ):
        raise RuntimeError("loaded V12 neutral ternary competition changed")
    if not all(
        torch.equal(before, after)
        for before, after in zip(
            counters,
            (model.target_hard_sync_count, model.ema_update_count),
            strict=True,
        )
    ):
        raise RuntimeError("V12 mechanism validation mutated target counters")


def load_checkpoint(
    encoded: bytes,
) -> GeometryAnchoredSweptProgressSurvivalJointJepaV12:
    """Validate and reconstruct the exact terminal V12 model on CPU."""

    if type(encoded) is not bytes or not encoded:
        raise TypeError("encoded checkpoint must be nonempty exact bytes")
    checkpoint = torch.load(
        io.BytesIO(encoded),
        map_location="cpu",
        weights_only=True,
    )
    checkpoint = _require_exact_keys(checkpoint, _CHECKPOINT_KEYS, "checkpoint")
    _validate_metadata(checkpoint)
    branch_activity = _validate_activity(
        checkpoint["height_role_branch_attention_activity"],
        name="height_role_branch_attention_activity",
        schema="lewm_v11_height_role_branch_attention_training_activity_v1",
        suffixes=_BRANCH_ATTENTION_PARAMETER_SUFFIXES,
        parameter_count=HEIGHT_ROLE_ATTENTION_PARAMETER_COUNT_V11,
        target_parameter_tensor_count=HEIGHT_ROLE_ATTENTION_PARAMETER_TENSOR_COUNT_V11,
        minimum_active_parameter_tensor_count=14,
    )
    semantic_activity = _validate_activity(
        checkpoint["factorized_semantic_axes_activity"],
        name="factorized_semantic_axes_activity",
        schema="lewm_v11_factorized_semantic_axes_training_activity_v1",
        suffixes=_SEMANTIC_AXIS_PARAMETER_SUFFIXES,
        parameter_count=HEIGHT_ROLE_SEMANTIC_PARAMETER_COUNT_V11,
        target_parameter_tensor_count=0,
        minimum_active_parameter_tensor_count=8,
    )
    _validate_training_diagnostics(
        checkpoint["training_diagnostics"], branch_activity, semantic_activity
    )
    state = _validate_state_mapping(checkpoint["model_state_dict"])

    encoder_prefix = "encoder."
    encoder_state = {
        name[len(encoder_prefix) :]: tensor.detach().clone()
        for name, tensor in state.items()
        if name.startswith(encoder_prefix)
    }
    if not encoder_state:
        raise ValueError("V12 checkpoint lacks its online encoder state")
    sweep_masks = state.get(SWEEP_MASK_STATE_KEY)
    if type(sweep_masks) is not torch.Tensor:
        raise ValueError("V12 checkpoint lacks exact swept-progress masks")

    model = GeometryAnchoredSweptProgressSurvivalJointJepaV12(
        encoder_state, sweep_masks.detach().clone()
    )
    model.load_state_dict(state, strict=True)
    loaded_state = model.state_dict()
    if set(loaded_state) != set(state):
        raise RuntimeError("strict V12 reconstruction changed state inventory")
    for name, expected in state.items():
        if not _tensor_bit_exact(loaded_state[name], expected):
            raise RuntimeError(f"strict V12 reconstruction changed tensor {name!r}")
    if (
        model.target_hard_sync_count.dtype != torch.long
        or model.ema_update_count.dtype != torch.long
        or tuple(model.target_hard_sync_count.shape) != ()
        or tuple(model.ema_update_count.shape) != ()
        or int(model.target_hard_sync_count.item()) != 1
        or int(model.ema_update_count.item()) != 1_000
    ):
        raise ValueError("V12 target-update counters disagree with terminal accounting")

    _validate_initial_receipt(checkpoint["initial_v12_model"], model)
    model.eval().requires_grad_(False)
    _validate_loaded_mechanism(model)
    tensors = tuple(model.parameters()) + tuple(model.buffers())
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise TypeError("V12 calibration model must remain CPU-resident")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("V12 calibration model was not completely frozen")
    if model.training or any(module.training for module in model.modules()):
        raise RuntimeError("V12 calibration model was not placed in evaluation mode")
    return model


__all__ = [
    "CHECKPOINT_SCHEMA",
    "PHYSICAL_CALIBRATION_PREREGISTRATION_COMMIT",
    "V12_PREREGISTRATION_COMMIT",
    "load_checkpoint",
]
