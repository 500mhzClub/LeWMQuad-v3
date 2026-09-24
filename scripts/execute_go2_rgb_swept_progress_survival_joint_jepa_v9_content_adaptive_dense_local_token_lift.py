#!/usr/bin/env python3
"""Execute the one-shot V9 content-adaptive dense-local-token lift probe."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import io
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_v4 = importlib.import_module(
    "scripts.execute_go2_rgb_swept_progress_survival_joint_jepa_v4_"
    "residual_local_semantic_decoder"
)
_v3 = _v4._v3
_v1 = _v4._v1

OUTPUT_RELATIVE_PATH = (
    ".generated/go2_rgb_swept_progress_survival_joint_jepa_v9_"
    "content_adaptive_dense_local_token_lift/attempt_v1"
)
CHECKPOINT_SCHEMA = "lewm_go2_rgb_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift_checkpoint_v1"
TRACE_SCHEMA = "lewm_go2_rgb_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift_trace_v1"
RESULT_SCHEMA = "lewm_go2_rgb_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift_result_v1"
FAILURE_SCHEMA = "lewm_go2_rgb_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift_failure_v1"
PREREGISTRATION_COMMIT = "47043472466e7a258ad0f0be854c05393e233db8"
PREIMPLEMENTATION_AMENDMENT_COMMIT = "04db6b26d46875297e3aa515fdf1d688bee2b755"

LABEL_ROOT_RELATIVE_PATH = _v4.LABEL_ROOT_RELATIVE_PATH
LABEL_MANIFEST_NAME = _v4.LABEL_MANIFEST_NAME
LABEL_MANIFEST_CONTENT_SHA256 = _v4.LABEL_MANIFEST_CONTENT_SHA256
LABEL_MANIFEST_FILE_SHA256 = _v4.LABEL_MANIFEST_FILE_SHA256
LABEL_MANIFEST_BYTE_COUNT = _v4.LABEL_MANIFEST_BYTE_COUNT
REQUIRED_GPU_NAME = _v4.REQUIRED_GPU_NAME
REQUIRED_GPU_MEMORY_BYTES = _v4.REQUIRED_GPU_MEMORY_BYTES
ACTION_ORDER = _v4.ACTION_ORDER
ROLE_FILES = _v4.ROLE_FILES
MICROBATCH_SIZE = _v4.MICROBATCH_SIZE
MICROBATCHES_PER_UPDATE = _v4.MICROBATCHES_PER_UPDATE
PRESENTATIONS_PER_UPDATE = _v4.PRESENTATIONS_PER_UPDATE
MAXIMUM_UPDATES = _v4.MAXIMUM_UPDATES
MAXIMUM_PRESENTATIONS = _v4.MAXIMUM_PRESENTATIONS
CONSTRUCTOR_INITIALIZATION_SEED = _v4.CONSTRUCTOR_INITIALIZATION_SEED
SEMANTIC_DECODER_INITIALIZATION_SEED = _v4.SEMANTIC_DECODER_INITIALIZATION_SEED
EXPERIMENT_SEED = _v4.EXPERIMENT_SEED
BOOTSTRAP_SEED = _v4.BOOTSTRAP_SEED
CONTROL_NAMES = _v4.CONTROL_NAMES
ALL_ARM_NAMES = _v4.ALL_ARM_NAMES
GATE_THRESHOLDS = _v4.GATE_THRESHOLDS
PROGRESS_SEGMENT_M = _v4.PROGRESS_SEGMENT_M
AUXILIARY_OBJECTIVE = dict(_v4.AUXILIARY_OBJECTIVE)

DENSE_LOCAL_ATTENTION_INITIALIZATION_SEED_V9 = 20_260_729
DENSE_LOCAL_SUPPORT_SIDE_V9 = 5
DENSE_LOCAL_SUPPORT_COUNT_V9 = 25
DENSE_LOCAL_SUPPORT_CENTER_INDEX_V9 = 12
DENSE_LOCAL_ATTENTION_HEADS_V9 = 4
DENSE_LOCAL_ATTENTION_HEAD_WIDTH_V9 = 16
DENSE_LOCAL_ATTENTION_PARAMETER_TENSOR_COUNT_V9 = 7
DENSE_LOCAL_ATTENTION_ADDED_PARAMETER_COUNT_V9 = 16_576
REMOVED_DEFORMABLE_PARAMETER_COUNT_PER_LIFT_V9 = 49_152
ATTENTION_PARAMETER_SUFFIXES_V9 = (
    "query_projection.weight",
    "query_projection.bias",
    "key_projection.weight",
    "value_projection.weight",
    "value_projection.bias",
    "output_projection.weight",
    "output_projection.bias",
)
DENSE_LOCAL_LIFT_ARCHITECTURE_V9 = {
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
        "support_side": DENSE_LOCAL_SUPPORT_SIDE_V9,
        "support_count": DENSE_LOCAL_SUPPORT_COUNT_V9,
        "center_index": DENSE_LOCAL_SUPPORT_CENTER_INDEX_V9,
        "legacy_config_samples_per_cell_retained_but_unused": 4,
        "offset_order": "row_major_y_then_x_for_integer_x_y_in_minus2_through_plus2",
        "normalized_token_cell_step_xy": [2.0 / 16.0, 2.0 / 16.0],
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
        "query": {"type": "Linear", "in_features": 64, "out_features": 64, "bias": True},
        "key": {"type": "Linear", "in_features": 64, "out_features": 64, "bias": False},
        "value": {"type": "Linear", "in_features": 64, "out_features": 64, "bias": True},
        "output": {"type": "Linear", "in_features": 64, "out_features": 64, "bias": True},
        "head_count": DENSE_LOCAL_ATTENTION_HEADS_V9,
        "head_width": DENSE_LOCAL_ATTENTION_HEAD_WIDTH_V9,
        "logit_scale": "1/sqrt(16)",
        "parameter_tensor_count": DENSE_LOCAL_ATTENTION_PARAMETER_TENSOR_COUNT_V9,
        "added_parameter_count_per_lift": DENSE_LOCAL_ATTENTION_ADDED_PARAMETER_COUNT_V9,
        "initialization": {
            "private_cpu_generator_seed": DENSE_LOCAL_ATTENTION_INITIALIZATION_SEED_V9,
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
    "aggregation": "center_sample_residual_plus_output_projected_dense_local_attention",
    "all_invalid_cells": "excluded_from_attention_softmax_with_exact_zero_reported_weights_then_exact_inherited_null_evidence_before_consumers",
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

scientific_metrics_v9 = _v4.scientific_metrics_v4
semantic_metrics_v9 = _v4.semantic_metrics_v4
paired_control_comparison_v9 = _v4.paired_control_comparison_v4
evaluate_gate_v9 = _v4.evaluate_gate_v4


def dense_local_lift_architecture_receipt_v9() -> dict[str, Any]:
    return copy.deepcopy(DENSE_LOCAL_LIFT_ARCHITECTURE_V9)


def _fresh_output_root_v9(repository_root: Path) -> Path:
    output = Path(repository_root) / OUTPUT_RELATIVE_PATH
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            "fresh content-adaptive-dense-local-token-lift attempt_v1 already exists"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    return output


def _validate_training_core_v9(training_v1: Any, training_v3: Any, training_v9: Any) -> None:
    _v4._validate_training_core_v4(training_v1, training_v3)
    for name in (
        "ACTION_ORDER", "MICROBATCH_SIZE", "MICROBATCHES_PER_UPDATE",
        "PRESENTATIONS_PER_UPDATE", "MAXIMUM_UPDATES", "MAXIMUM_PRESENTATIONS",
        "OCCUPIED_CLASS_INDEX", "OCCUPIED_SAFETY_AUX_COEFFICIENT",
        "OCCUPIED_SAFETY_AUX_NORMALIZATION",
    ):
        if getattr(training_v9, name, None) != getattr(training_v3, name):
            raise PermissionError(f"V9 training wrapper changed inherited {name}")
    if (
        getattr(training_v9, "DENSE_LOCAL_ATTENTION_ADDED_PARAMETER_COUNT_V9", None)
        != DENSE_LOCAL_ATTENTION_ADDED_PARAMETER_COUNT_V9
        or tuple(getattr(training_v9, "ATTENTION_PARAMETER_SUFFIXES_V9", ()))
        != ATTENTION_PARAMETER_SUFFIXES_V9
        or not callable(getattr(training_v9, "run_fixed_training_v9", None))
    ):
        raise PermissionError("V9 training wrapper contract changed")


def _validate_model_api_v9(model_api: Any) -> None:
    for name, expected in (
        (
            "DENSE_LOCAL_ATTENTION_INITIALIZATION_SEED_V9",
            DENSE_LOCAL_ATTENTION_INITIALIZATION_SEED_V9,
        ),
        ("DENSE_LOCAL_SUPPORT_SIDE_V9", DENSE_LOCAL_SUPPORT_SIDE_V9),
        ("DENSE_LOCAL_SUPPORT_COUNT_V9", DENSE_LOCAL_SUPPORT_COUNT_V9),
        (
            "DENSE_LOCAL_SUPPORT_CENTER_INDEX_V9",
            DENSE_LOCAL_SUPPORT_CENTER_INDEX_V9,
        ),
        ("DENSE_LOCAL_ATTENTION_HEADS_V9", DENSE_LOCAL_ATTENTION_HEADS_V9),
        (
            "DENSE_LOCAL_ATTENTION_HEAD_WIDTH_V9",
            DENSE_LOCAL_ATTENTION_HEAD_WIDTH_V9,
        ),
        (
            "DENSE_LOCAL_ATTENTION_PARAMETER_TENSOR_COUNT_V9",
            DENSE_LOCAL_ATTENTION_PARAMETER_TENSOR_COUNT_V9,
        ),
        (
            "DENSE_LOCAL_ATTENTION_ADDED_PARAMETER_COUNT_V9",
            DENSE_LOCAL_ATTENTION_ADDED_PARAMETER_COUNT_V9,
        ),
    ):
        if getattr(model_api, name, None) != expected:
            raise PermissionError(f"V9 model changed {name}")
    for name in (
        "ContentAdaptiveDenseLocalTokenLiftV9",
        "ContentAdaptiveDenseLocalTokenLiftSamplingV9",
        "GeometryAnchoredSweptProgressSurvivalJointJepaV9",
    ):
        if not callable(getattr(model_api, name, None)):
            raise PermissionError(f"V9 model API lacks {name}")
    if (
        getattr(model_api, "GeometryAnchoredDeformableBevLiftJointJepaV1", None)
        is not model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV9
    ):
        raise PermissionError("V9 historical runner model alias changed")


def _names_sha256_v9(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _attention_inventory_v9(lift: Any) -> tuple[tuple[str, Any], ...]:
    parameters = dict(lift.named_parameters())
    if any(name not in parameters for name in ATTENTION_PARAMETER_SUFFIXES_V9):
        raise RuntimeError("V9 attention parameter inventory is incomplete")
    inventory = tuple((name, parameters[name]) for name in ATTENTION_PARAMETER_SUFFIXES_V9)
    attention_names = tuple(
        name
        for name in parameters
        if name.startswith(
            ("query_projection.", "key_projection.", "value_projection.", "output_projection.")
        )
    )
    if attention_names != ATTENTION_PARAMETER_SUFFIXES_V9:
        raise RuntimeError("V9 attention parameter inventory changed")
    if len(inventory) != DENSE_LOCAL_ATTENTION_PARAMETER_TENSOR_COUNT_V9 or sum(
        parameter.numel() for _, parameter in inventory
    ) != DENSE_LOCAL_ATTENTION_ADDED_PARAMETER_COUNT_V9:
        raise RuntimeError("V9 attention parameter count changed")
    return inventory


def _validate_attention_initialization_v9(
    inventory: Sequence[tuple[str, Any]], *, torch: Any
) -> None:
    actual = dict(inventory)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(DENSE_LOCAL_ATTENTION_INITIALIZATION_SEED_V9)
    for projection in ("query_projection", "key_projection", "value_projection", "output_projection"):
        weight = actual[f"{projection}.weight"].detach().cpu()
        expected = torch.empty_like(weight)
        torch.nn.init.xavier_uniform_(expected, gain=1.0, generator=generator)
        if not torch.equal(weight, expected):
            raise RuntimeError(f"V9 {projection} weight initialization changed")
        bias = actual.get(f"{projection}.bias")
        if bias is not None and int(torch.count_nonzero(bias.detach()).item()) != 0:
            raise RuntimeError(f"V9 {projection} bias is not exactly zero")


def _migration_receipt_v9(
    model: Any,
    clean_v4: Any,
    *,
    torch: Any,
    model_api: Any,
) -> Mapping[str, Any]:
    """Prove that V9 is clean V4 with exactly one lift replacement."""

    if not isinstance(model.bev_lift, model_api.ContentAdaptiveDenseLocalTokenLiftV9) or not isinstance(
        model.target_bev_lift, model_api.ContentAdaptiveDenseLocalTokenLiftV9
    ):
        raise RuntimeError("V9 online/target lift types changed")
    if (
        model.bev_lift.config != clean_v4.bev_lift.config
        or model.target_bev_lift.config != clean_v4.target_bev_lift.config
        or model.bev_lift.config.samples_per_cell != 4
    ):
        raise RuntimeError("V9 inherited clean-V4 lift config changed")
    v9_state = model.state_dict()
    v4_state = clean_v4.state_dict()
    removed = {
        "bev_lift.raw_offsets",
        "bev_lift.weight_logits",
        "target_bev_lift.raw_offsets",
        "target_bev_lift.weight_logits",
    }
    online_added = {f"bev_lift.{name}" for name in ATTENTION_PARAMETER_SUFFIXES_V9}
    target_added = {
        f"target_bev_lift.{name}" for name in ATTENTION_PARAMETER_SUFFIXES_V9
    }
    support_buffers = {
        "bev_lift.support_offsets_token_cells",
        "target_bev_lift.support_offsets_token_cells",
    }
    added = online_added | target_added | support_buffers
    if set(v4_state) - set(v9_state) != removed:
        raise RuntimeError("V9 removed state inventory changed")
    if set(v9_state) - set(v4_state) != added:
        raise RuntimeError("V9 added state inventory changed")
    common = tuple(name for name in v4_state if name not in removed)
    changed = tuple(
        name for name in common if not torch.equal(v4_state[name], v9_state[name])
    )
    if changed:
        raise RuntimeError(f"V9 changed inherited V4 tensor {changed[0]}")

    removed_counts = {
        prefix: sum(
            v4_state[f"{prefix}.{suffix}"].numel()
            for suffix in ("raw_offsets", "weight_logits")
        )
        for prefix in ("bev_lift", "target_bev_lift")
    }
    if any(
        count != REMOVED_DEFORMABLE_PARAMETER_COUNT_PER_LIFT_V9
        for count in removed_counts.values()
    ):
        raise RuntimeError("V9 removed deformable parameter count changed")

    online = _attention_inventory_v9(model.bev_lift)
    target = _attention_inventory_v9(model.target_bev_lift)
    if any(not parameter.requires_grad for _, parameter in online) or any(
        parameter.requires_grad for _, parameter in target
    ):
        raise RuntimeError("V9 attention online/target trainability changed")
    if any(parameter.grad is not None for _, parameter in target):
        raise RuntimeError("V9 target attention has an initial gradient")
    if any(
        not torch.equal(left.detach(), right.detach())
        for (_, left), (_, right) in zip(online, target, strict=True)
    ):
        raise RuntimeError("V9 target attention is not an exact initial copy")
    projections = (
        model.bev_lift.query_projection,
        model.bev_lift.key_projection,
        model.bev_lift.value_projection,
        model.bev_lift.output_projection,
    )
    if any(
        not isinstance(projection, torch.nn.Linear)
        or projection.in_features != 64
        or projection.out_features != 64
        for projection in projections
    ) or tuple(projection.bias is not None for projection in projections) != (
        True,
        False,
        True,
        True,
    ):
        raise RuntimeError("V9 Q/K/V/O projection architecture changed")
    _validate_attention_initialization_v9(online, torch=torch)

    with torch.no_grad():
        sampling = model.bev_lift.forward_with_sampling(
            torch.zeros((1, 256, 192), dtype=torch.float32)
        )
    if not isinstance(sampling, model_api.ContentAdaptiveDenseLocalTokenLiftSamplingV9):
        raise RuntimeError("V9 dedicated sampling receipt type changed")
    expected_offsets = torch.tensor(
        [(float(x), float(y)) for y in range(-2, 3) for x in range(-2, 3)],
        dtype=torch.float32,
    )
    if not torch.equal(sampling.support_offsets_token_cells.cpu(), expected_offsets):
        raise RuntimeError("V9 support offset order changed")
    if not torch.equal(
        model.bev_lift.support_offsets_token_cells,
        model.target_bev_lift.support_offsets_token_cells,
    ):
        raise RuntimeError("V9 online/target support offsets differ")
    if not torch.equal(
        sampling.support_offsets_token_cells[DENSE_LOCAL_SUPPORT_CENTER_INDEX_V9],
        sampling.support_offsets_token_cells.new_zeros(2),
    ):
        raise RuntimeError("V9 center support index changed")
    anchor = model.bev_lift.anchor_grid_xy[None]
    proposed = anchor[..., None, :] + expected_offsets * (2.0 / 16.0)
    expected_valid = model.bev_lift.anchor_in_frustum[None, ..., None] & (
        (proposed[..., 0] >= -1.0)
        & (proposed[..., 0] <= 1.0)
        & (proposed[..., 1] >= -1.0)
        & (proposed[..., 1] <= 1.0)
    )
    expected_grid = torch.where(
        expected_valid[..., None], proposed, torch.full_like(proposed, 2.0)
    )
    if (
        tuple(sampling.latent.shape) != (1, 64, 64, 64)
        or tuple(sampling.anchor_in_frustum.shape) != (1, 64, 64)
        or tuple(sampling.support_valid_mask.shape) != (1, 64, 64, 25)
        or tuple(sampling.cell_valid_mask.shape) != (1, 64, 64)
        or tuple(sampling.support_grid_xy.shape) != (1, 64, 64, 25, 2)
        or tuple(sampling.attention_weights.shape) != (1, 64, 64, 4, 25)
    ):
        raise RuntimeError("V9 sampling receipt shape changed")
    if not torch.equal(sampling.support_valid_mask, expected_valid) or not torch.equal(
        sampling.support_grid_xy, expected_grid
    ):
        raise RuntimeError("V9 fixed dense-local sampling geometry changed")
    if not torch.equal(sampling.cell_valid_mask, expected_valid.any(dim=-1)) or not torch.equal(
        sampling.cell_valid_mask, sampling.anchor_in_frustum
    ):
        raise RuntimeError("V9 cell-valid or anchor visibility semantics changed")
    weights = sampling.attention_weights
    if not bool(torch.isfinite(weights).all()):
        raise RuntimeError("V9 attention weights are nonfinite")
    invalid_support = ~sampling.support_valid_mask[..., None, :].expand_as(weights)
    if int(torch.count_nonzero(weights.masked_select(invalid_support)).item()) != 0:
        raise RuntimeError("V9 invalid support received attention weight")
    valid_sums = weights.sum(dim=-1).masked_select(
        sampling.cell_valid_mask[..., None].expand(-1, -1, -1, DENSE_LOCAL_ATTENTION_HEADS_V9)
    )
    if not torch.allclose(valid_sums, torch.ones_like(valid_sums), rtol=0.0, atol=1e-6):
        raise RuntimeError("V9 valid attention weights do not sum to one")
    invalid_cells = ~sampling.cell_valid_mask
    if int(
        torch.count_nonzero(
            weights.masked_select(invalid_cells[..., None, None].expand_as(weights))
        ).item()
    ) != 0:
        raise RuntimeError("V9 all-invalid cell attention receipt is not exact zero")
    null = model.bev_lift.null_evidence[None, :, None].expand(
        -1, -1, int(invalid_cells.sum().item())
    )
    invalid_latent = sampling.latent.masked_select(
        invalid_cells[:, None].expand_as(sampling.latent)
    ).reshape(1, 64, -1)
    if not torch.equal(invalid_latent, null):
        raise RuntimeError("V9 all-invalid latent is not exact inherited null evidence")

    return {
        "source": "fresh clean V4 construction with identical N320 state and masks",
        "removed_state_names": sorted(removed),
        "added_state_names": sorted(added),
        "all_inherited_state_tensors_bit_exact": True,
        "inherited_state_tensor_count": len(common),
        "inherited_state_name_inventory_sha256": _names_sha256_v9(common),
        "removed_parameter_count_per_online_or_target_lift": REMOVED_DEFORMABLE_PARAMETER_COUNT_PER_LIFT_V9,
        "added_attention_parameter_count_per_online_or_target_lift": DENSE_LOCAL_ATTENTION_ADDED_PARAMETER_COUNT_V9,
        "added_attention_parameter_tensor_count_per_online_or_target_lift": DENSE_LOCAL_ATTENTION_PARAMETER_TENSOR_COUNT_V9,
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


def _initial_model_receipt_v9(
    model: Any,
    partition: Any,
    migration: Mapping[str, Any],
    *,
    torch: Any,
    model_api: Any,
    inherited_semantic_method: Any,
) -> Mapping[str, Any]:
    online = _attention_inventory_v9(model.bev_lift)
    target = _attention_inventory_v9(model.target_bev_lift)
    online_names = tuple(f"bev_lift.{name}" for name, _ in online)
    target_names = tuple(f"target_bev_lift.{name}" for name, _ in target)
    partition_online = tuple(
        name for name in partition.names["lift_semantic"] if name in online_names
    )
    partition_target = tuple(
        name for name in partition.names["target"] if name in target_names
    )
    if partition_online != online_names or partition_target != target_names:
        raise RuntimeError("V9 attention parameter partition changed")
    inventory = [
        name
        for group in ("encoder", "lift_semantic", "predictor", "target")
        for name in partition.names[group]
    ]
    if any(inventory.count(name) != 1 for name in online_names + target_names):
        raise RuntimeError("V9 attention parameter was not partitioned exactly once")
    if any(
        not torch.equal(left.detach(), right.detach())
        for (_, left), (_, right) in zip(online, target, strict=True)
    ):
        raise RuntimeError("V9 target attention is not an exact initial copy")
    inherited_decoder = _v4._initial_decoder_receipt_v4(
        model,
        partition,
        torch=torch,
        inherited_semantic_method=inherited_semantic_method,
    )
    if int(model.target_hard_sync_count.item()) != 1 or int(model.ema_update_count.item()) != 0:
        raise RuntimeError("V9 initial target synchronization counters changed")
    return {
        "architecture": dense_local_lift_architecture_receipt_v9(),
        "migration": dict(migration),
        "inherited_v4_decoder": inherited_decoder,
        "online_attention_parameter_count": sum(parameter.numel() for _, parameter in online),
        "online_attention_parameter_tensor_count": len(online),
        "target_attention_parameter_count": sum(parameter.numel() for _, parameter in target),
        "target_attention_parameter_tensor_count": len(target),
        "attention_parameter_suffix_inventory_sha256": _names_sha256_v9(ATTENTION_PARAMETER_SUFFIXES_V9),
        "all_online_attention_parameters_in_lift_semantic_exactly_once": True,
        "all_target_attention_parameters_frozen_in_target_exactly_once": True,
        "target_initial_copy_exact": True,
        "initial_hard_sync_count": 1,
        "initial_ema_update_count": 0,
    }


def _physical_calibration_stage_v9(full_arm_passed: bool) -> Mapping[str, Any]:
    return {
        "status": "STAGED_FOR_SEPARATELY_FROZEN_ONE_SHOT" if full_arm_passed else "CLOSED_FULL_ARM_GATE_FAILED",
        "physical_calibration_run_in_this_attempt": False,
        "requires_full_arm_pass": True,
        "protocol_changed_from_reviewed_v4_calibration": False,
        "threshold_tuple_count": 2_016,
        "physical_gate_passed": False,
    }


def execute_v9(*, repository_root: Path = ROOT) -> Mapping[str, Any]:
    repository_root = Path(repository_root).absolute()
    _v1._install_repository_import_roots_v1(repository_root)
    output = _fresh_output_root_v9(repository_root)
    initial_model: Mapping[str, Any] | None = None
    attention_activity: Mapping[str, Any] | None = None
    checkpoint_binding: Mapping[str, Any] | None = None
    trace_binding: Mapping[str, Any] | None = None
    try:
        labels_api = importlib.import_module("lewm.benchmarks.go2_swept_progress_survival_labels_v1")
        manifest, rows_by_role = _v1.load_label_bundle_v1(repository_root, labels_api=labels_api)
        context = _v1._prepare_runtime_v1(repository_root, manifest, labels_api)
        torch, np = context["torch"], context["np"]
        if labels_api.summarize_preflight_v1(rows_by_role, context["schedule"]) != manifest.get("preflight"):
            raise PermissionError("label preflight no longer matches its manifest")
        training_v1 = importlib.import_module("scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v1")
        training_v3 = importlib.import_module("scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v3_half_occupied_safety_aux")
        training_v9 = importlib.import_module("scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift")
        _validate_training_core_v9(training_v1, training_v3, training_v9)
        frozen = {role: training_v1.freeze_role_labels_v1(rows, role=role, np=np) for role, rows in rows_by_role.items()}
        informative = {
            role: np.asarray([group[0]["informative_state"] for group in labels.state_groups], dtype=np.bool_)
            for role, labels in frozen.items()
        }
        pairs = {role: context["inputs"].role_pairs(role) for role in ROLE_FILES}
        for role in ROLE_FILES:
            training_v1.validate_pairs_against_labels_v1(pairs[role], frozen[role])

        model_api = importlib.import_module("lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift")
        _validate_model_api_v9(model_api)
        v4_model_api = importlib.import_module(
            "lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v4_"
            "residual_local_semantic_decoder"
        )
        parent_model_api = importlib.import_module("lewm.models.geometry_anchored_deformable_bev_lift_joint_jepa_v1")
        survival_scoring = importlib.import_module("lewm.benchmarks.go2_swept_progress_survival_joint_jepa_v1")
        metrics_api = importlib.import_module("lewm.benchmarks.go2_post_action_projective_support_metrics_v1")
        torch.manual_seed(EXPERIMENT_SEED)
        torch.cuda.manual_seed_all(EXPERIMENT_SEED)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

        n320_state = {name: value.detach().cpu().float().contiguous().clone() for name, value in context["fit"].encoder.state_dict().items()}
        masks = survival_scoring.build_swept_progress_masks_v1()
        current_frame_persistence_masks = survival_scoring.build_current_frame_swept_progress_masks_v1()
        constructor_rng = torch.random.get_rng_state().clone()
        model = model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV9(
            n320_state, masks
        )
        if not torch.equal(torch.random.get_rng_state(), constructor_rng):
            raise RuntimeError("V9 constructor did not restore the caller CPU RNG")
        clean_v4 = v4_model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV4(
            n320_state, masks
        )
        if not torch.equal(torch.random.get_rng_state(), constructor_rng):
            raise RuntimeError("clean V4 audit constructor changed the caller CPU RNG")
        migration = _migration_receipt_v9(
            model, clean_v4, torch=torch, model_api=model_api
        )
        migration = {**migration, "caller_cpu_rng_state_restored": True}
        del clean_v4
        model = model.to(context["device"])
        model.train()
        partition = training_v1.partition_parameters_v1(model)
        initial_model = _initial_model_receipt_v9(
            model,
            partition,
            migration,
            torch=torch,
            model_api=model_api,
            inherited_semantic_method=(
                parent_model_api.GeometryAnchoredDeformableBevLiftJointJepaV1.
                semantic_logits_from_latent
            ),
        )
        optimizer = training_v1.build_frozen_optimizer_v1(partition)
        if not any(name.startswith("predictor.swept_progress_head.") for name in partition.names["predictor"]):
            raise RuntimeError("survival head escaped the predictor optimizer group")

        accounting_state, trace, training_diagnostics = training_v9.run_fixed_training_v9(
            model, optimizer, context["loader"], pairs["train"], frozen["train"], context["schedule"], context["device"]
        )
        accounting = dict(accounting_state.__dict__)
        attention_activity = training_diagnostics["dense_local_attention"]
        model.eval()
        model.requires_grad_(False)
        state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
        checkpoint_buffer = io.BytesIO()
        torch.save({
            "schema": CHECKPOINT_SCHEMA, "development_only": True,
            "resume_authorized": False, "qualified": False,
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "preimplementation_amendment_commit": PREIMPLEMENTATION_AMENDMENT_COMMIT,
            "constructor_initialization_seed": CONSTRUCTOR_INITIALIZATION_SEED,
            "semantic_decoder_initialization_seed": SEMANTIC_DECODER_INITIALIZATION_SEED,
            "dense_local_attention_initialization_seed": DENSE_LOCAL_ATTENTION_INITIALIZATION_SEED_V9,
            "experiment_seed": EXPERIMENT_SEED,
            "initialization_source": "exact_n320_encoder_and_clean_v4_with_preregistered_lift_replacement",
            "predecessor_experiment_checkpoint_read": False,
            "inherited_occupied_auxiliary": dict(AUXILIARY_OBJECTIVE),
            "initial_v9_model": initial_model,
            "dense_local_attention_activity": attention_activity,
            "training_diagnostics": training_diagnostics, "accounting": accounting,
            "model_state_dict": state,
        }, checkpoint_buffer)
        checkpoint_binding = _v1._atomic_write_v1(output / "checkpoint_update_1000.pt", checkpoint_buffer.getvalue())
        _, trace_binding = _v1._write_json_v1(output / "training_trace.json", {
            "schema": TRACE_SCHEMA, "status": "COMPLETE",
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "preimplementation_amendment_commit": PREIMPLEMENTATION_AMENDMENT_COMMIT,
            "initial_v9_model": initial_model,
            "dense_local_attention_activity": attention_activity,
            "training_diagnostics": training_diagnostics, "accounting": accounting,
            "rows": list(trace),
        })

        action_prior_m = frozen["train"].prefix_lengths.mean(axis=0, dtype=np.float64) * PROGRESS_SEGMENT_M
        scored = {
            role: _v1.score_role_v1(
                model, context["loader"], pairs[role], frozen[role], action_prior_m,
                context["device"], torch=torch, np=np, training_core=training_v1,
                current_frame_persistence_masks=current_frame_persistence_masks,
                metrics_api=metrics_api,
            ) for role in ("probability_calibration", "checkpoint_selection")
        }
        role_metrics = {
            role: {
                arm: scientific_metrics_v9(
                    scored[role]["scores_m"][arm], frozen[role].prefix_lengths,
                    informative[role], frozen[role].scene_ids, frozen[role].family_ids, np=np,
                ) for arm in ALL_ARM_NAMES
            } for role in scored
        }
        selection_semantic = semantic_metrics_v9(
            scored["checkpoint_selection"]["semantic_confusion"],
            scored["checkpoint_selection"]["rough_semantic_confusion"], np=np,
        )
        selection_scores = scored["checkpoint_selection"]["scores_m"]
        selection_labels = frozen["checkpoint_selection"]
        comparisons = {
            name: paired_control_comparison_v9(
                selection_scores["full"], selection_scores[name], selection_labels.prefix_lengths,
                informative["checkpoint_selection"], selection_labels.scene_ids,
                selection_labels.family_ids, np=np,
            ) for name in CONTROL_NAMES
        }
        gate = evaluate_gate_v9(role_metrics["checkpoint_selection"], selection_semantic, comparisons)
        if len(gate.get("checks", {})) != 24:
            raise RuntimeError("V9 inherited 24-check full-arm gate changed")
        full_arm_passed = bool(gate["passed"])
        checkpoint_access = (
            "STAGED_FOR_SEPARATE_PHYSICAL_CALIBRATION"
            if full_arm_passed
            else "CLOSED_FULL_ARM_GATE_FAILED"
        )
        calibration_stage = _physical_calibration_stage_v9(full_arm_passed)
        access_receipt = _v1._access_receipt_v1(context)
        mask_receipts = {
            "predicted_next_post_action_frame": _v1._mask_receipt_v1(masks),
            "coordinate_matched_current_frame_persistence": _v1._mask_receipt_v1(current_frame_persistence_masks),
        }
        result, _ = _v1._write_json_v1(output / "result.json", {
            "schema": RESULT_SCHEMA,
            "status": "PASS_FULL_ARM_STAGED_FOR_PHYSICAL_CALIBRATION" if full_arm_passed else "FAIL_DEVELOPMENT_FULL_ARM",
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "preimplementation_amendment_commit": PREIMPLEMENTATION_AMENDMENT_COMMIT,
            "full_arm_gate": gate, "gate": gate,
            "physical_evidence_calibration": calibration_stage,
            "caps": {"updates": MAXIMUM_UPDATES, "microbatch_graphs": 4_000, "presentations": MAXIMUM_PRESENTATIONS},
            "seeds": {
                "inherited_fresh_component_constructor": CONSTRUCTOR_INITIALIZATION_SEED,
                "semantic_decoder": SEMANTIC_DECODER_INITIALIZATION_SEED,
                "dense_local_attention_private_cpu_generator": DENSE_LOCAL_ATTENTION_INITIALIZATION_SEED_V9,
                "experiment_and_stochastic_execution": EXPERIMENT_SEED, "bootstrap": BOOTSTRAP_SEED,
            },
            "label_manifest": {
                "path": f"{LABEL_ROOT_RELATIVE_PATH}/{LABEL_MANIFEST_NAME}",
                "file_sha256": LABEL_MANIFEST_FILE_SHA256,
                "content_sha256": manifest["content_sha256"], "byte_count": LABEL_MANIFEST_BYTE_COUNT,
                "role_files": manifest["files"],
            },
            "n320": {
                "gate_content_sha256": context["n320_gate"]["content_sha256"],
                "checkpoint": context["n320_checkpoint"], "encoder_only_initialization": True,
                "predecessor_experiment_checkpoint_read": False,
            },
            "hardware": context["hardware"],
            "schedule_prefix_sha256": labels_api.v4.SCHEDULE_PREFIX_SHA256,
            "masks": mask_receipts,
            "scientific_change_from_v4": {
                "only_change": "content_adaptive_dense_local_token_lift",
                "initial_v9_model": initial_model,
                "architecture": dense_local_lift_architecture_receipt_v9(),
                "inherited_occupied_auxiliary": dict(AUXILIARY_OBJECTIVE),
                "model_changed": True,
                "data_changed": False,
                "dataset_identity_changed": False,
                "input_tensorization_changed": False,
                "inherited_nonreplacement_state_bit_exact": True,
                "removed_parameters_per_online_or_target_lift": REMOVED_DEFORMABLE_PARAMETER_COUNT_PER_LIFT_V9,
                "added_parameters_per_online_or_target_lift": DENSE_LOCAL_ATTENTION_ADDED_PARAMETER_COUNT_V9,
                "added_parameter_tensors_per_online_or_target_lift": DENSE_LOCAL_ATTENTION_PARAMETER_TENSOR_COUNT_V9,
                "optimizer_rules_changed": False,
                "optimizer_parameter_tensor_membership_changed": True,
                "losses_changed": False, "schedule_changed": False, "evaluation_changed": False,
            },
            "training": {
                "core": "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift",
                "accounting": accounting, "diagnostics": training_diagnostics,
                "dense_local_attention_activity": attention_activity,
                "joint_from_update_one": True,
                "separate_head_or_predictor_training": False,
                "checkpoint_access_status": checkpoint_access,
                "checkpoint": checkpoint_binding, "trace": trace_binding,
            },
            "action_prior_mean_progress_m": action_prior_m.tolist(), "roles": role_metrics,
            "selection_semantic": selection_semantic, "selection_control_comparisons": comparisons,
            "wrong_rgb_mapping_sha256": {role: scored[role]["wrong_rgb_mapping_sha256"] for role in scored},
            "determinism": {
                "algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()), "warn_only": True,
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            },
            "access": access_receipt,
            "authority": {
                "development_only": True, "g2_navigation_final_evaluation_opened": False,
                "heldout_or_sealed_opened": False, "physical_evidence_gate_passed": False,
                "checkpoint_qualified": False, "promotion_performed": False,
                "retry_or_resume_authorized": False,
                "checkpoint_access_authorized_for_physical_calibration": full_arm_passed,
            },
        })
        return result
    except Exception as error:
        if not (output / "result.json").exists() and not (output / "failure.json").exists():
            try:
                _v1._write_json_v1(output / "failure.json", {
                    "schema": FAILURE_SCHEMA, "status": "FAILED_NO_RETRY_OR_RESUME",
                    "error_type": type(error).__name__, "error_message": str(error),
                    "traceback": traceback.format_exc(), "preregistration_commit": PREREGISTRATION_COMMIT,
                    "preimplementation_amendment_commit": PREIMPLEMENTATION_AMENDMENT_COMMIT,
                    "dense_local_lift_architecture": dense_local_lift_architecture_receipt_v9(),
                    "initial_v9_model": initial_model,
                    "dense_local_attention_activity": attention_activity,
                    "checkpoint": checkpoint_binding,
                    "training_trace": trace_binding,
                    "predecessor_experiment_checkpoint_read": False,
                    "physical_calibration_run_in_this_attempt": False,
                    "authority": {
                        "development_only": True, "g2_navigation_final_evaluation_opened": False,
                        "heldout_or_sealed_opened": False, "checkpoint_qualified": False,
                        "retry_or_resume_authorized": False,
                    },
                })
            except Exception:
                pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = execute_v9(repository_root=args.repository_root)
    print(_v1._canonical_json_bytes({"status": result["status"], "result": f"{OUTPUT_RELATIVE_PATH}/result.json"}).decode("utf-8"))
    return 0 if result["full_arm_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
