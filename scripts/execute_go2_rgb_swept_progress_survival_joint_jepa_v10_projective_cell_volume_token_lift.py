#!/usr/bin/env python3
"""Execute the one-shot V10 projective cell-volume token-lift probe.

V10 changes only V9's fixed image-token support geometry and base aggregation.
The V9/V4 data, objective, optimizer, training cap, controls, 24 checks, and
attention-gradient accounting are imported unchanged.
"""
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

_v9 = importlib.import_module(
    "scripts.execute_go2_rgb_swept_progress_survival_joint_jepa_v9_"
    "content_adaptive_dense_local_token_lift"
)
_v4 = _v9._v4
_v1 = _v9._v1

OUTPUT_RELATIVE_PATH = (
    ".generated/go2_rgb_swept_progress_survival_joint_jepa_v10_"
    "projective_cell_volume_token_lift/attempt_v1"
)
CHECKPOINT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v10_"
    "projective_cell_volume_token_lift_checkpoint_v1"
)
TRACE_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v10_"
    "projective_cell_volume_token_lift_trace_v1"
)
RESULT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v10_"
    "projective_cell_volume_token_lift_result_v1"
)
FAILURE_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v10_"
    "projective_cell_volume_token_lift_failure_v1"
)
PREREGISTRATION_COMMIT = "b9eaae6560c42e588c86fb8bf949cc95bd9e29e9"

LABEL_ROOT_RELATIVE_PATH = _v9.LABEL_ROOT_RELATIVE_PATH
LABEL_MANIFEST_NAME = _v9.LABEL_MANIFEST_NAME
LABEL_MANIFEST_FILE_SHA256 = _v9.LABEL_MANIFEST_FILE_SHA256
LABEL_MANIFEST_BYTE_COUNT = _v9.LABEL_MANIFEST_BYTE_COUNT
ACTION_ORDER = _v9.ACTION_ORDER
ROLE_FILES = _v9.ROLE_FILES
MICROBATCH_SIZE = _v9.MICROBATCH_SIZE
MICROBATCHES_PER_UPDATE = _v9.MICROBATCHES_PER_UPDATE
PRESENTATIONS_PER_UPDATE = _v9.PRESENTATIONS_PER_UPDATE
MAXIMUM_UPDATES = _v9.MAXIMUM_UPDATES
MAXIMUM_PRESENTATIONS = _v9.MAXIMUM_PRESENTATIONS
CONSTRUCTOR_INITIALIZATION_SEED = _v9.CONSTRUCTOR_INITIALIZATION_SEED
SEMANTIC_DECODER_INITIALIZATION_SEED = _v9.SEMANTIC_DECODER_INITIALIZATION_SEED
EXPERIMENT_SEED = _v9.EXPERIMENT_SEED
BOOTSTRAP_SEED = _v9.BOOTSTRAP_SEED
CONTROL_NAMES = _v9.CONTROL_NAMES
ALL_ARM_NAMES = _v9.ALL_ARM_NAMES
GATE_THRESHOLDS = _v9.GATE_THRESHOLDS
PROGRESS_SEGMENT_M = _v9.PROGRESS_SEGMENT_M
AUXILIARY_OBJECTIVE = dict(_v9.AUXILIARY_OBJECTIVE)

CELL_VOLUME_ATTENTION_INITIALIZATION_SEED_V10 = 20_260_729
CELL_VOLUME_HORIZONTAL_SUPPORT_COUNT_V10 = 5
CELL_VOLUME_HEIGHT_COUNT_V10 = 5
CELL_VOLUME_SUPPORT_COUNT_V10 = 25
CELL_VOLUME_ATTENTION_HEADS_V10 = 4
CELL_VOLUME_ATTENTION_HEAD_WIDTH_V10 = 16
CELL_VOLUME_ATTENTION_PARAMETER_TENSOR_COUNT_V10 = 7
CELL_VOLUME_ATTENTION_ADDED_PARAMETER_COUNT_V10 = 16_576
CELL_VOLUME_VALID_CELL_COUNT_V10 = 2_062
NEAR_FIELD_CELL_COUNT_V10 = 1_016
NEAR_FIELD_VALID_CELL_COUNT_V10 = 222
CELL_VOLUME_VALID_MASK_SHA256_V10 = (
    "4ebbafb6d4dd5fb13b96df978abfa7b81bc2f879b2ba6dec2fcda38dec54e60b"
)
ATTENTION_PARAMETER_SUFFIXES_V10 = _v9.ATTENTION_PARAMETER_SUFFIXES_V9
HORIZONTAL_SUPPORT_OFFSETS_XY_M_V10 = (
    (0.0, 0.0),
    (-0.05, -0.05),
    (-0.05, 0.05),
    (0.05, -0.05),
    (0.05, 0.05),
)
SUPPORT_HEIGHTS_M_V10 = (-0.333, -0.133, 0.067, 0.267, 0.467)

CELL_VOLUME_LIFT_ARCHITECTURE_V10 = {
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
        "horizontal_offsets_xy_m": [list(value) for value in HORIZONTAL_SUPPORT_OFFSETS_XY_M_V10],
        "heights_m": list(SUPPORT_HEIGHTS_M_V10),
        "order": "horizontal_major_then_height_ascending",
        "support_count": CELL_VOLUME_SUPPORT_COUNT_V10,
        "camera_origin_xyz_m": [0.326, 0.0, 0.043],
        "camera_mount_rpy_degrees": [0.0, 0.0, 0.0],
        "horizontal_fov_degrees": 78.323,
        "vertical_fov_degrees": 62.8370386364,
        "inclusive_near_m": 0.05,
        "cell_validity": "or_over_25_closed_frustum_support_bits",
        "cell_valid_count": CELL_VOLUME_VALID_CELL_COUNT_V10,
        "cell_valid_mask_row_major_uint8_sha256": CELL_VOLUME_VALID_MASK_SHA256_V10,
        "near_field_lte_2m_cell_count": NEAR_FIELD_CELL_COUNT_V10,
        "near_field_lte_2m_valid_cell_count": NEAR_FIELD_VALID_CELL_COUNT_V10,
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
        "head_count": CELL_VOLUME_ATTENTION_HEADS_V10,
        "head_width": CELL_VOLUME_ATTENTION_HEAD_WIDTH_V10,
        "attention_initialization_seed": CELL_VOLUME_ATTENTION_INITIALIZATION_SEED_V10,
        "attention_parameter_tensor_count": CELL_VOLUME_ATTENTION_PARAMETER_TENSOR_COUNT_V10,
        "attention_parameter_count_per_lift": CELL_VOLUME_ATTENTION_ADDED_PARAMETER_COUNT_V10,
    },
    "invalid_cells": (
        "inherited_null_evidence_after_initial_lift_and_each_refinement_block; "
        "semantic_logits_exact_(0,-20,-20)"
    ),
    "new_loss_or_head": False,
}

scientific_metrics_v10 = _v9.scientific_metrics_v9
semantic_metrics_v10 = _v9.semantic_metrics_v9
paired_control_comparison_v10 = _v9.paired_control_comparison_v9
evaluate_gate_v10 = _v9.evaluate_gate_v9


def cell_volume_lift_architecture_receipt_v10() -> dict[str, Any]:
    return copy.deepcopy(CELL_VOLUME_LIFT_ARCHITECTURE_V10)


def _fresh_output_root_v10(repository_root: Path) -> Path:
    output = Path(repository_root) / OUTPUT_RELATIVE_PATH
    if output.exists() or output.is_symlink():
        raise FileExistsError("fresh projective-cell-volume attempt_v1 already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    return output


def _validate_training_core_v10(
    training_v1: Any, training_v3: Any, training_v9: Any
) -> None:
    """Require the unchanged S+P+U+R+O V9/V4 core and gradient wrapper."""

    _v9._validate_training_core_v9(training_v1, training_v3, training_v9)
    if (
        training_v9.run_fixed_training_v9.__module__
        != "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift"
    ):
        raise PermissionError("V10 no longer delegates to the reviewed V9 training core")
    if dict(_v4.AUXILIARY_OBJECTIVE) != AUXILIARY_OBJECTIVE:
        raise PermissionError("V10 changed the inherited occupied objective")


def _validate_model_api_v10(model_api: Any) -> None:
    expected = {
        "CELL_VOLUME_ATTENTION_INITIALIZATION_SEED_V10": CELL_VOLUME_ATTENTION_INITIALIZATION_SEED_V10,
        "CELL_VOLUME_HORIZONTAL_SUPPORT_COUNT_V10": CELL_VOLUME_HORIZONTAL_SUPPORT_COUNT_V10,
        "CELL_VOLUME_HEIGHT_COUNT_V10": CELL_VOLUME_HEIGHT_COUNT_V10,
        "CELL_VOLUME_SUPPORT_COUNT_V10": CELL_VOLUME_SUPPORT_COUNT_V10,
        "CELL_VOLUME_ATTENTION_HEADS_V10": CELL_VOLUME_ATTENTION_HEADS_V10,
        "CELL_VOLUME_ATTENTION_HEAD_WIDTH_V10": CELL_VOLUME_ATTENTION_HEAD_WIDTH_V10,
        "CELL_VOLUME_ATTENTION_PARAMETER_TENSOR_COUNT_V10": CELL_VOLUME_ATTENTION_PARAMETER_TENSOR_COUNT_V10,
        "CELL_VOLUME_ATTENTION_ADDED_PARAMETER_COUNT_V10": CELL_VOLUME_ATTENTION_ADDED_PARAMETER_COUNT_V10,
    }
    for name, value in expected.items():
        if getattr(model_api, name, None) != value:
            raise PermissionError(f"V10 model changed {name}")
    for name in (
        "ProjectiveCellVolumeTokenLiftV10",
        "ProjectiveCellVolumeTokenLiftSamplingV10",
        "GeometryAnchoredSweptProgressSurvivalJointJepaV10",
    ):
        if not callable(getattr(model_api, name, None)):
            raise PermissionError(f"V10 model API lacks {name}")
    if (
        getattr(model_api, "GeometryAnchoredDeformableBevLiftJointJepaV1", None)
        is not model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV10
    ):
        raise PermissionError("V10 historical runner model alias changed")


def _mask_sha256_v10(mask: Any, *, torch: Any) -> str:
    payload = mask.detach().to(device="cpu", dtype=torch.uint8).contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _migration_receipt_v10(
    model: Any,
    fresh_v9: Any,
    *,
    torch: Any,
    model_api: Any,
) -> Mapping[str, Any]:
    """Prove that V10 retains every V9 parameter and changes only geometry."""

    if not isinstance(model.bev_lift, model_api.ProjectiveCellVolumeTokenLiftV10) or not isinstance(
        model.target_bev_lift, model_api.ProjectiveCellVolumeTokenLiftV10
    ):
        raise RuntimeError("V10 online/target lift type changed")
    if model.bev_lift.config != fresh_v9.bev_lift.config:
        raise RuntimeError("V10 inherited lift config changed")

    v10_parameters = dict(model.named_parameters())
    v9_parameters = dict(fresh_v9.named_parameters())
    if tuple(v10_parameters) != tuple(v9_parameters):
        raise RuntimeError("V10 parameter inventory differs from V9")
    changed_parameters = tuple(
        name
        for name in v9_parameters
        if not torch.equal(v9_parameters[name].detach(), v10_parameters[name].detach())
    )
    if changed_parameters:
        raise RuntimeError(f"V10 changed inherited V9 parameter {changed_parameters[0]}")

    v10_buffers = dict(model.named_buffers())
    v9_buffers = dict(fresh_v9.named_buffers())
    removed = set(v9_buffers) - set(v10_buffers)
    added = set(v10_buffers) - set(v9_buffers)
    expected_removed = {
        "bev_lift.support_offsets_token_cells",
        "target_bev_lift.support_offsets_token_cells",
    }
    geometry_names = (
        "support_offsets_xy_m",
        "support_heights_m",
        "support_xyz_m",
        "support_grid_xy",
        "support_valid_mask",
        "cell_valid_mask",
    )
    expected_added = {
        f"{prefix}.{name}"
        for prefix in ("bev_lift", "target_bev_lift")
        for name in geometry_names
    }
    if removed != expected_removed or added != expected_added:
        raise RuntimeError("V10 geometry-buffer migration inventory changed")
    common_buffers = tuple(name for name in v9_buffers if name in v10_buffers)
    changed_buffers = tuple(
        name
        for name in common_buffers
        if not torch.equal(v9_buffers[name], v10_buffers[name])
    )
    if changed_buffers:
        raise RuntimeError(f"V10 changed inherited V9 buffer {changed_buffers[0]}")

    online = _v9._attention_inventory_v9(model.bev_lift)
    target = _v9._attention_inventory_v9(model.target_bev_lift)
    _v9._validate_attention_initialization_v9(online, torch=torch)
    if any(parameter.grad is not None for _, parameter in target):
        raise RuntimeError("V10 target attention has an initial gradient")
    if any(
        not torch.equal(left.detach(), right.detach())
        for (_, left), (_, right) in zip(online, target, strict=True)
    ):
        raise RuntimeError("V10 online/target attention initial copies differ")

    with torch.no_grad():
        sampling = model.bev_lift.forward_with_sampling(
            torch.zeros((1, 256, 192), dtype=torch.float32)
        )
    expected_fields = (
        "latent", "anchor_in_frustum", "support_valid_mask", "cell_valid_mask",
        "support_grid_xy", "support_xyz_m", "support_offsets_xy_m",
        "support_heights_m", "masked_mean", "attention_weights",
    )
    if not isinstance(sampling, model_api.ProjectiveCellVolumeTokenLiftSamplingV10) or sampling._fields != expected_fields:
        raise RuntimeError("V10 sampling receipt type or fields changed")
    expected_offsets = torch.tensor(HORIZONTAL_SUPPORT_OFFSETS_XY_M_V10, dtype=torch.float32)
    expected_heights = torch.tensor(SUPPORT_HEIGHTS_M_V10, dtype=torch.float32)
    if not torch.equal(sampling.support_offsets_xy_m.cpu(), expected_offsets) or not torch.equal(
        sampling.support_heights_m.cpu(), expected_heights
    ):
        raise RuntimeError("V10 support order changed")
    shapes = (
        tuple(sampling.latent.shape), tuple(sampling.anchor_in_frustum.shape),
        tuple(sampling.support_valid_mask.shape), tuple(sampling.cell_valid_mask.shape),
        tuple(sampling.support_grid_xy.shape), tuple(sampling.support_xyz_m.shape),
        tuple(sampling.masked_mean.shape), tuple(sampling.attention_weights.shape),
    )
    if shapes != (
        (1, 64, 64, 64), (1, 64, 64), (1, 64, 64, 25), (1, 64, 64),
        (1, 64, 64, 25, 2), (64, 64, 25, 3), (1, 64, 64, 64),
        (1, 64, 64, 4, 25),
    ):
        raise RuntimeError("V10 sampling receipt shape changed")
    support_valid = sampling.support_valid_mask
    cell_valid = sampling.cell_valid_mask
    if not torch.equal(cell_valid, support_valid.any(dim=-1)):
        raise RuntimeError("V10 cell-volume validity is not support OR")
    static_cell_valid = model.bev_lift.cell_valid_mask
    if not torch.equal(cell_valid[0], static_cell_valid):
        raise RuntimeError("V10 sampling/static cell-valid masks differ")
    if int(static_cell_valid.sum().item()) != CELL_VOLUME_VALID_CELL_COUNT_V10 or (
        _mask_sha256_v10(static_cell_valid, torch=torch)
        != CELL_VOLUME_VALID_MASK_SHA256_V10
    ):
        raise RuntimeError("V10 frozen cell-valid count or hash changed")
    near = model.bev_lift.bev_ground_xyz_m[..., :2].square().sum(dim=-1) <= 4.0
    if int(near.sum().item()) != NEAR_FIELD_CELL_COUNT_V10 or int(
        (near & static_cell_valid).sum().item()
    ) != NEAR_FIELD_VALID_CELL_COUNT_V10:
        raise RuntimeError("V10 frozen near-field support count changed")
    invalid_grid = sampling.support_grid_xy.masked_select(
        (~support_valid)[..., None].expand_as(sampling.support_grid_xy)
    )
    if invalid_grid.numel() and not torch.equal(invalid_grid, torch.full_like(invalid_grid, 2.0)):
        raise RuntimeError("V10 invalid sampling coordinate changed")
    weights = sampling.attention_weights
    if not bool(torch.isfinite(weights).all()) or int(
        torch.count_nonzero(weights.masked_select((~support_valid)[..., None, :].expand_as(weights))).item()
    ):
        raise RuntimeError("V10 invalid support received attention weight")
    valid_sums = weights.sum(dim=-1).masked_select(
        cell_valid[..., None].expand(-1, -1, -1, CELL_VOLUME_ATTENTION_HEADS_V10)
    )
    if not torch.allclose(valid_sums, torch.ones_like(valid_sums), rtol=0.0, atol=1e-6):
        raise RuntimeError("V10 valid attention weights do not sum to one")
    invalid_cells = ~cell_valid
    null = model.bev_lift.null_evidence[None, :, None].expand(
        -1, -1, int(invalid_cells.sum().item())
    )
    invalid_latent = sampling.latent.masked_select(
        invalid_cells[:, None].expand_as(sampling.latent)
    ).reshape(1, 64, -1)
    if not torch.equal(invalid_latent, null):
        raise RuntimeError("V10 invalid latent is not exact inherited null evidence")
    logits = model.semantic_logits_from_latent(sampling.latent)
    invalid_logits = logits.masked_select(
        invalid_cells[:, None].expand_as(logits)
    ).reshape(1, 3, -1)
    expected_unknown = logits.new_tensor((0.0, -20.0, -20.0))[None, :, None].expand_as(invalid_logits)
    if not torch.equal(invalid_logits, expected_unknown):
        raise RuntimeError("V10 invalid semantic logits are not exact UNKNOWN")

    return {
        "schema": "lewm_v10_projective_cell_volume_token_lift_migration_v1",
        "source": "fresh V9 and clean V4 construction with identical N320 state",
        "all_v9_parameter_names_and_values_bit_exact": True,
        "inherited_parameter_tensor_count": len(v10_parameters),
        "removed_geometry_buffer_names": sorted(removed),
        "added_geometry_buffer_names": sorted(added),
        "all_common_buffers_bit_exact": True,
        "attention_initialization_bit_exact": True,
        "online_target_attention_initial_copy_exact": True,
        "target_attention_initial_gradient_tensor_count": 0,
        "sampling_receipt": {
            "schema": "lewm_v10_projective_cell_volume_token_lift_sampling_audit_v1",
            "type": "ProjectiveCellVolumeTokenLiftSamplingV10",
            "cell_valid_count": int(static_cell_valid.sum().item()),
            "cell_valid_mask_row_major_uint8_sha256": _mask_sha256_v10(static_cell_valid, torch=torch),
            "near_field_lte_2m_cell_count": int(near.sum().item()),
            "near_field_lte_2m_valid_cell_count": int((near & static_cell_valid).sum().item()),
            "support_order_bit_exact": True,
            "safe_invalid_grid_xy": [2.0, 2.0],
            "invalid_support_attention_exact_zero": True,
            "valid_attention_sums_one_per_head": True,
            "all_invalid_latent_exact_inherited_null_evidence": True,
            "all_invalid_semantic_logits_exact_unknown": True,
        },
    }


def _initial_model_receipt_v10(
    model: Any,
    partition: Any,
    migration: Mapping[str, Any],
    *,
    torch: Any,
) -> Mapping[str, Any]:
    inherited = dict(
        _v9._initial_model_receipt_v9(
            model,
            partition,
            migration,
            torch=torch,
            model_api=None,
            # V10 intentionally overrides only the post-decoder validity mask.
            # Passing its own method lets the inherited V4 helper audit the
            # unchanged decoder modules and parameter partition.
            inherited_semantic_method=type(model).semantic_logits_from_latent,
        )
    )
    decoder = copy.deepcopy(dict(inherited["inherited_v4_decoder"]))
    visibility = model.bev_lift.cell_valid_mask.detach().cpu().contiguous()
    if (
        tuple(visibility.shape) != (64, 64)
        or visibility.dtype != torch.bool
        or int(visibility.sum().item()) != CELL_VOLUME_VALID_CELL_COUNT_V10
        or _mask_sha256_v10(visibility, torch=torch)
        != CELL_VOLUME_VALID_MASK_SHA256_V10
    ):
        raise RuntimeError("V10 semantic cell-volume validity changed")
    decoder["visibility_mask"] = {
        "schema": "lewm_v10_cell_volume_semantic_validity_mask_v1",
        "shape": [64, 64],
        "dtype": "bool",
        "true_cell_count": int(visibility.sum().item()),
        "sha256": _mask_sha256_v10(visibility, torch=torch),
        "application": "v10_post_decoder_cell_volume_validity",
        "invalid_logits": [0.0, -20.0, -20.0],
    }
    decoder["decoder_parameters_changed_from_v4"] = False
    decoder["semantic_mask_route_changed_from_v9"] = True
    inherited.update(
        {
            "schema": "lewm_v10_projective_cell_volume_token_lift_initial_model_v1",
            "architecture": cell_volume_lift_architecture_receipt_v10(),
            "migration": dict(migration),
            "inherited_v4_decoder": decoder,
            "attention_receipt_source": "unchanged_v9_qkvo_inventory_and_initialization_audit",
        }
    )
    return inherited


def _rebadge_attention_receipt_v10(receipt: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(receipt))
    schema = str(result.get("schema", ""))
    expected = "lewm_v9_dense_local_attention_post_backward_gradient_v1"
    if schema != expected:
        raise RuntimeError("V9 per-update attention receipt schema changed")
    result["schema"] = "lewm_v10_cell_volume_attention_post_backward_gradient_v1"
    result["implementation"] = "unchanged_v9_attention_gradient_receipt"
    return result


def _rebadge_attention_activity_v10(receipt: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(receipt))
    if result.get("schema") != "lewm_v9_dense_local_attention_training_activity_v1":
        raise RuntimeError("V9 attention activity schema changed")
    result["schema"] = "lewm_v10_cell_volume_attention_training_activity_v1"
    result["implementation"] = "unchanged_v9_attention_gradient_receipts"
    return result


def _run_fixed_training_v10(training_v9: Any, *args: Any) -> tuple[Any, tuple[dict[str, Any], ...], dict[str, Any]]:
    accounting, trace, diagnostics = training_v9.run_fixed_training_v9(*args)
    activity = _rebadge_attention_activity_v10(diagnostics["dense_local_attention"])
    rebadged_trace = tuple(
        {**row, "dense_local_attention": _rebadge_attention_receipt_v10(row["dense_local_attention"])}
        for row in trace
    )
    rebadged_diagnostics = {
        **diagnostics,
        "dense_local_attention": activity,
        "v10_contract": {
            "schema": "lewm_v10_unchanged_joint_training_contract_v1",
            "objective": "S+P+U+R+O",
            "occupied_auxiliary_coefficient": 0.5,
            "new_loss_or_head": False,
            "training_core": "unchanged_v9_wrapper_over_v3_v4",
        },
    }
    return accounting, rebadged_trace, rebadged_diagnostics


def _physical_calibration_stage_v10(full_arm_passed: bool) -> Mapping[str, Any]:
    result = dict(_v9._physical_calibration_stage_v9(full_arm_passed))
    result["schema"] = "lewm_v10_unchanged_physical_calibration_stage_v1"
    result["source"] = "unchanged_v9_v4_2016_tuple_protocol"
    return result


def execute_v10(*, repository_root: Path = ROOT) -> Mapping[str, Any]:
    repository_root = Path(repository_root).absolute()
    _v1._install_repository_import_roots_v1(repository_root)
    output = _fresh_output_root_v10(repository_root)
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
        _validate_training_core_v10(training_v1, training_v3, training_v9)
        frozen = {role: training_v1.freeze_role_labels_v1(rows, role=role, np=np) for role, rows in rows_by_role.items()}
        informative = {
            role: np.asarray([group[0]["informative_state"] for group in labels.state_groups], dtype=np.bool_)
            for role, labels in frozen.items()
        }
        pairs = {role: context["inputs"].role_pairs(role) for role in ROLE_FILES}
        for role in ROLE_FILES:
            training_v1.validate_pairs_against_labels_v1(pairs[role], frozen[role])

        model_api = importlib.import_module("lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v10_projective_cell_volume_token_lift")
        _validate_model_api_v10(model_api)
        v9_model_api = importlib.import_module("lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift")
        v4_model_api = importlib.import_module("lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v4_residual_local_semantic_decoder")
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
        model = model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV10(n320_state, masks)
        fresh_v9 = v9_model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV9(n320_state, masks)
        clean_v4 = v4_model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV4(n320_state, masks)
        if not torch.equal(torch.random.get_rng_state(), constructor_rng):
            raise RuntimeError("V10 audit constructors did not restore the caller CPU RNG")
        inherited_v9_migration = _v9._migration_receipt_v9(
            fresh_v9, clean_v4, torch=torch, model_api=v9_model_api
        )
        migration = dict(_migration_receipt_v10(model, fresh_v9, torch=torch, model_api=model_api))
        migration["fresh_v9_clean_v4_migration"] = inherited_v9_migration
        migration["caller_cpu_rng_state_restored"] = True
        del fresh_v9, clean_v4

        model = model.to(context["device"])
        model.train()
        partition = training_v1.partition_parameters_v1(model)
        initial_model = _initial_model_receipt_v10(
            model,
            partition,
            migration,
            torch=torch,
        )
        optimizer = training_v1.build_frozen_optimizer_v1(partition)
        accounting_state, trace, training_diagnostics = _run_fixed_training_v10(
            training_v9,
            model, optimizer, context["loader"], pairs["train"], frozen["train"],
            context["schedule"], context["device"],
        )
        accounting = dict(accounting_state.__dict__)
        attention_activity = training_diagnostics["dense_local_attention"]
        model.eval()
        model.requires_grad_(False)
        state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
        checkpoint_buffer = io.BytesIO()
        torch.save(
            {
                "schema": CHECKPOINT_SCHEMA, "development_only": True,
                "resume_authorized": False, "qualified": False,
                "preregistration_commit": PREREGISTRATION_COMMIT,
                "constructor_initialization_seed": CONSTRUCTOR_INITIALIZATION_SEED,
                "semantic_decoder_initialization_seed": SEMANTIC_DECODER_INITIALIZATION_SEED,
                "cell_volume_attention_initialization_seed": CELL_VOLUME_ATTENTION_INITIALIZATION_SEED_V10,
                "experiment_seed": EXPERIMENT_SEED,
                "initialization_source": "exact_n320_encoder_and_fresh_v9_v4_with_only_preregistered_geometry_replacement",
                "predecessor_experiment_checkpoint_read": False,
                "objective": "S+P+U+R+O", "inherited_occupied_auxiliary": dict(AUXILIARY_OBJECTIVE),
                "initial_v10_model": initial_model,
                "cell_volume_attention_activity": attention_activity,
                "training_diagnostics": training_diagnostics, "accounting": accounting,
                "model_state_dict": state,
            },
            checkpoint_buffer,
        )
        checkpoint_binding = _v1._atomic_write_v1(output / "checkpoint_update_1000.pt", checkpoint_buffer.getvalue())
        _, trace_binding = _v1._write_json_v1(
            output / "training_trace.json",
            {
                "schema": TRACE_SCHEMA, "status": "COMPLETE",
                "preregistration_commit": PREREGISTRATION_COMMIT,
                "initial_v10_model": initial_model,
                "cell_volume_attention_activity": attention_activity,
                "training_diagnostics": training_diagnostics, "accounting": accounting,
                "rows": list(trace),
            },
        )

        action_prior_m = frozen["train"].prefix_lengths.mean(axis=0, dtype=np.float64) * PROGRESS_SEGMENT_M
        scored = {
            role: _v1.score_role_v1(
                model, context["loader"], pairs[role], frozen[role], action_prior_m,
                context["device"], torch=torch, np=np, training_core=training_v1,
                current_frame_persistence_masks=current_frame_persistence_masks,
                metrics_api=metrics_api,
            )
            for role in ("probability_calibration", "checkpoint_selection")
        }
        role_metrics = {
            role: {
                arm: scientific_metrics_v10(
                    scored[role]["scores_m"][arm], frozen[role].prefix_lengths,
                    informative[role], frozen[role].scene_ids, frozen[role].family_ids, np=np,
                )
                for arm in ALL_ARM_NAMES
            }
            for role in scored
        }
        selection_semantic = semantic_metrics_v10(
            scored["checkpoint_selection"]["semantic_confusion"],
            scored["checkpoint_selection"]["rough_semantic_confusion"], np=np,
        )
        selection_scores = scored["checkpoint_selection"]["scores_m"]
        selection_labels = frozen["checkpoint_selection"]
        comparisons = {
            name: paired_control_comparison_v10(
                selection_scores["full"], selection_scores[name], selection_labels.prefix_lengths,
                informative["checkpoint_selection"], selection_labels.scene_ids,
                selection_labels.family_ids, np=np,
            )
            for name in CONTROL_NAMES
        }
        gate = evaluate_gate_v10(role_metrics["checkpoint_selection"], selection_semantic, comparisons)
        if len(gate.get("checks", {})) != 24:
            raise RuntimeError("V10 inherited 24-check full-arm gate changed")
        full_arm_passed = bool(gate["passed"])
        calibration_stage = _physical_calibration_stage_v10(full_arm_passed)
        access_receipt = _v1._access_receipt_v1(context)
        mask_receipts = {
            "predicted_next_post_action_frame": _v1._mask_receipt_v1(masks),
            "coordinate_matched_current_frame_persistence": _v1._mask_receipt_v1(current_frame_persistence_masks),
        }
        result, _ = _v1._write_json_v1(
            output / "result.json",
            {
                "schema": RESULT_SCHEMA,
                "status": "PASS_FULL_ARM_STAGED_FOR_PHYSICAL_CALIBRATION" if full_arm_passed else "FAIL_DEVELOPMENT_FULL_ARM",
                "preregistration_commit": PREREGISTRATION_COMMIT,
                "full_arm_gate": gate, "gate": gate,
                "physical_evidence_calibration": calibration_stage,
                "caps": {"updates": MAXIMUM_UPDATES, "microbatch_graphs": 4_000, "presentations": MAXIMUM_PRESENTATIONS},
                "seeds": {
                    "inherited_fresh_component_constructor": CONSTRUCTOR_INITIALIZATION_SEED,
                    "semantic_decoder": SEMANTIC_DECODER_INITIALIZATION_SEED,
                    "cell_volume_attention_private_cpu_generator": CELL_VOLUME_ATTENTION_INITIALIZATION_SEED_V10,
                    "experiment_and_stochastic_execution": EXPERIMENT_SEED,
                    "bootstrap": BOOTSTRAP_SEED,
                },
                "label_manifest": {
                    "path": f"{LABEL_ROOT_RELATIVE_PATH}/{LABEL_MANIFEST_NAME}",
                    "file_sha256": LABEL_MANIFEST_FILE_SHA256,
                    "content_sha256": manifest["content_sha256"],
                    "byte_count": LABEL_MANIFEST_BYTE_COUNT,
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
                "scientific_change_from_v9": {
                    "only_change": "projective_cell_volume_token_lift_geometry_and_masked_mean_base",
                    "initial_v10_model": initial_model,
                    "architecture": cell_volume_lift_architecture_receipt_v10(),
                    "objective": "S+P+U+R+O", "inherited_occupied_auxiliary": dict(AUXILIARY_OBJECTIVE),
                    "model_changed": True, "data_changed": False,
                    "dataset_identity_changed": False, "input_tensorization_changed": False,
                    "inherited_parameter_state_bit_exact": True,
                    "optimizer_rules_changed": False,
                    "optimizer_parameter_tensor_membership_changed": False,
                    "losses_changed": False, "new_loss_or_head": False,
                    "schedule_changed": False, "evaluation_changed": False,
                },
                "training": {
                    "core": "unchanged scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift",
                    "accounting": accounting, "diagnostics": training_diagnostics,
                    "cell_volume_attention_activity": attention_activity,
                    "joint_from_update_one": True, "separate_head_or_predictor_training": False,
                    "checkpoint_access_status": "STAGED_FOR_SEPARATE_PHYSICAL_CALIBRATION" if full_arm_passed else "CLOSED_FULL_ARM_GATE_FAILED",
                    "checkpoint": checkpoint_binding, "trace": trace_binding,
                },
                "action_prior_mean_progress_m": action_prior_m.tolist(), "roles": role_metrics,
                "selection_semantic": selection_semantic,
                "selection_control_comparisons": comparisons,
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
            },
        )
        return result
    except Exception as error:
        if not (output / "result.json").exists() and not (output / "failure.json").exists():
            try:
                _v1._write_json_v1(
                    output / "failure.json",
                    {
                        "schema": FAILURE_SCHEMA, "status": "FAILED_NO_RETRY_OR_RESUME",
                        "error_type": type(error).__name__, "error_message": str(error),
                        "traceback": traceback.format_exc(),
                        "preregistration_commit": PREREGISTRATION_COMMIT,
                        "cell_volume_lift_architecture": cell_volume_lift_architecture_receipt_v10(),
                        "initial_v10_model": initial_model,
                        "cell_volume_attention_activity": attention_activity,
                        "checkpoint": checkpoint_binding, "training_trace": trace_binding,
                        "predecessor_experiment_checkpoint_read": False,
                        "physical_calibration_run_in_this_attempt": False,
                        "authority": {
                            "development_only": True, "g2_navigation_final_evaluation_opened": False,
                            "heldout_or_sealed_opened": False, "checkpoint_qualified": False,
                            "retry_or_resume_authorized": False,
                        },
                    },
                )
            except Exception:
                pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = execute_v10(repository_root=args.repository_root)
    print(
        _v1._canonical_json_bytes(
            {"status": result["status"], "result": f"{OUTPUT_RELATIVE_PATH}/result.json"}
        ).decode("utf-8")
    )
    return 0 if result["full_arm_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
