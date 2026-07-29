#!/usr/bin/env python3
"""Execute the one-shot V11 height-role factorized-evidence probe.

V11 preserves V10's data, schedule, joint objective, predictor, survival head,
evaluation, and cap.  It replaces only V10's unordered support attention and
three-way semantic decoder with the preregistered floor/elevated evidence
routes.  The tensor update remains delegated to the reviewed V11 training
helper; this file owns only authority, integrity receipts, scoring, and
write-once terminalization.
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

_v10 = importlib.import_module(
    "scripts.execute_go2_rgb_swept_progress_survival_joint_jepa_v10_"
    "projective_cell_volume_token_lift"
)
_v9 = _v10._v9
_v4 = _v10._v4
_v1 = _v10._v1

OUTPUT_RELATIVE_PATH = (
    ".generated/go2_rgb_swept_progress_survival_joint_jepa_v11_"
    "height_role_factorized_evidence_lift/attempt_v1"
)
CHECKPOINT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v11_"
    "height_role_factorized_evidence_lift_checkpoint_v1"
)
TRACE_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v11_"
    "height_role_factorized_evidence_lift_trace_v1"
)
RESULT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v11_"
    "height_role_factorized_evidence_lift_result_v1"
)
FAILURE_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v11_"
    "height_role_factorized_evidence_lift_failure_v1"
)
PREREGISTRATION_COMMIT = "b8ca8bd267e233a11f29da82842dcf5429743c18"

LABEL_ROOT_RELATIVE_PATH = _v10.LABEL_ROOT_RELATIVE_PATH
LABEL_MANIFEST_NAME = _v10.LABEL_MANIFEST_NAME
LABEL_MANIFEST_FILE_SHA256 = _v10.LABEL_MANIFEST_FILE_SHA256
LABEL_MANIFEST_BYTE_COUNT = _v10.LABEL_MANIFEST_BYTE_COUNT
ACTION_ORDER = _v10.ACTION_ORDER
ROLE_FILES = _v10.ROLE_FILES
MICROBATCH_SIZE = _v10.MICROBATCH_SIZE
MICROBATCHES_PER_UPDATE = _v10.MICROBATCHES_PER_UPDATE
PRESENTATIONS_PER_UPDATE = _v10.PRESENTATIONS_PER_UPDATE
MAXIMUM_UPDATES = _v10.MAXIMUM_UPDATES
MAXIMUM_PRESENTATIONS = _v10.MAXIMUM_PRESENTATIONS
CONSTRUCTOR_INITIALIZATION_SEED = _v10.CONSTRUCTOR_INITIALIZATION_SEED
EXPERIMENT_SEED = _v10.EXPERIMENT_SEED
BOOTSTRAP_SEED = _v10.BOOTSTRAP_SEED
CONTROL_NAMES = _v10.CONTROL_NAMES
ALL_ARM_NAMES = _v10.ALL_ARM_NAMES
GATE_THRESHOLDS = _v10.GATE_THRESHOLDS
PROGRESS_SEGMENT_M = _v10.PROGRESS_SEGMENT_M
AUXILIARY_OBJECTIVE = dict(_v10.AUXILIARY_OBJECTIVE)

HEIGHT_ROLE_INITIALIZATION_SEED_V11 = 20_260_730
FLOOR_SUPPORT_INDICES_V11 = (0, 5, 10, 15, 20)
ELEVATED_SUPPORT_INDICES_V11 = tuple(
    index for index in range(25) if index not in FLOOR_SUPPORT_INDICES_V11
)
FLOOR_SUPPORT_COUNT_V11 = 5
ELEVATED_SUPPORT_COUNT_V11 = 20
HEIGHT_ROLE_ATTENTION_HEADS_V11 = 2
HEIGHT_ROLE_ATTENTION_HEAD_WIDTH_V11 = 16
HEIGHT_ROLE_ATTENTION_PARAMETER_TENSOR_COUNT_V11 = 14
HEIGHT_ROLE_ATTENTION_PARAMETER_COUNT_V11 = 14_528
HEIGHT_ROLE_SEMANTIC_PARAMETER_TENSOR_COUNT_V11 = 12
HEIGHT_ROLE_SEMANTIC_PARAMETER_COUNT_V11 = 18_628
CELL_VALID_COUNT_V11 = 2_062
FLOOR_VALID_CELL_COUNT_V11 = 2_024
ELEVATED_VALID_CELL_COUNT_V11 = 2_062
ROLE_VALID_OVERLAP_CELL_COUNT_V11 = 2_024
ELEVATED_ONLY_CELL_COUNT_V11 = 38
NEAR_FIELD_CELL_COUNT_V11 = 1_016
NEAR_FIELD_FLOOR_VALID_CELL_COUNT_V11 = 184
NEAR_FIELD_ELEVATED_VALID_CELL_COUNT_V11 = 222
CELL_VALID_MASK_SHA256_V11 = (
    "4ebbafb6d4dd5fb13b96df978abfa7b81bc2f879b2ba6dec2fcda38dec54e60b"
)
FLOOR_VALID_MASK_SHA256_V11 = (
    "8b6b4202d04cf08de9813a4fc12deff9ea35de8d8c7adc8eb40a117593694bbc"
)

scientific_metrics_v11 = _v10.scientific_metrics_v10
semantic_metrics_v11 = _v10.semantic_metrics_v10
paired_control_comparison_v11 = _v10.paired_control_comparison_v10
evaluate_gate_v11 = _v10.evaluate_gate_v10


HEIGHT_ROLE_FACTORIZED_ARCHITECTURE_V11 = {
    "schema": "lewm_v11_height_role_factorized_evidence_lift_architecture_v1",
    "predecessor": "fresh_v10_source_architecture_with_no_v10_runtime_reuse",
    "sole_mechanism": (
        "fixed_floor_and_elevated_support_routes_preserved_in_one_role_ordered_"
        "shared_jepa_state_with_occupied_priority_abstention"
    ),
    "geometry": {
        "changed_from_v10": False,
        "support_count": 25,
        "support_order": "v10_horizontal_major_then_height_ascending",
        "cell_valid_count": CELL_VALID_COUNT_V11,
        "cell_valid_mask_row_major_uint8_sha256": CELL_VALID_MASK_SHA256_V11,
        "near_field_lte_2m_cell_count": NEAR_FIELD_CELL_COUNT_V11,
        "near_field_lte_2m_valid_cell_count": NEAR_FIELD_ELEVATED_VALID_CELL_COUNT_V11,
    },
    "roles": {
        "floor_free": {
            "support_indices": list(FLOOR_SUPPORT_INDICES_V11),
            "support_count": FLOOR_SUPPORT_COUNT_V11,
            "height_m": -0.333,
            "valid_cell_count": FLOOR_VALID_CELL_COUNT_V11,
            "valid_mask_row_major_uint8_sha256": FLOOR_VALID_MASK_SHA256_V11,
            "near_field_valid_cell_count": NEAR_FIELD_FLOOR_VALID_CELL_COUNT_V11,
            "latent_channels": [0, 32],
        },
        "elevated_occupied": {
            "support_indices": list(ELEVATED_SUPPORT_INDICES_V11),
            "support_count": ELEVATED_SUPPORT_COUNT_V11,
            "heights_m": [-0.133, 0.067, 0.267, 0.467],
            "valid_cell_count": ELEVATED_VALID_CELL_COUNT_V11,
            "valid_mask_row_major_uint8_sha256": CELL_VALID_MASK_SHA256_V11,
            "near_field_valid_cell_count": NEAR_FIELD_ELEVATED_VALID_CELL_COUNT_V11,
            "latent_channels": [32, 64],
        },
        "disjoint_and_exhaustive": True,
        "valid_overlap_cell_count": ROLE_VALID_OVERLAP_CELL_COUNT_V11,
        "elevated_only_cell_count": ELEVATED_ONLY_CELL_COUNT_V11,
    },
    "aggregation": {
        "heads_per_role": HEIGHT_ROLE_ATTENTION_HEADS_V11,
        "head_width": HEIGHT_ROLE_ATTENTION_HEAD_WIDTH_V11,
        "qkv_dimensions": [64, 32],
        "output_dimensions": [32, 32],
        "parameter_count_per_online_or_target_lift": HEIGHT_ROLE_ATTENTION_PARAMETER_COUNT_V11,
        "parameter_tensor_count_per_online_or_target_lift": HEIGHT_ROLE_ATTENTION_PARAMETER_TENSOR_COUNT_V11,
        "initialization_seed": HEIGHT_ROLE_INITIALIZATION_SEED_V11,
        "cross_role_attention_weight": 0.0,
        "shared_state_channel_order": ["floor_free_0_31", "elevated_occupied_32_63"],
    },
    "semantic": {
        "parameter_count": HEIGHT_ROLE_SEMANTIC_PARAMETER_COUNT_V11,
        "parameter_tensor_count": HEIGHT_ROLE_SEMANTIC_PARAMETER_TENSOR_COUNT_V11,
        "axis_order": ["free", "occupied"],
        "free_axis_input_channels": [0, 32],
        "occupied_axis_input_channels": [32, 64],
        "log_probabilities": {
            "occupied": "logsigmoid(o)",
            "free": "logsigmoid(-o)+logsigmoid(f)",
            "unknown": "logsigmoid(-o)+logsigmoid(-f)",
        },
        "all_invalid_logits": [0.0, -20.0, -20.0],
    },
    "predictor_consumes_shared_role_ordered_64_channel_state": True,
    "new_loss_or_loss_weight": False,
}


def height_role_factorized_architecture_receipt_v11() -> dict[str, Any]:
    return copy.deepcopy(HEIGHT_ROLE_FACTORIZED_ARCHITECTURE_V11)


def _fresh_output_root_v11(repository_root: Path) -> Path:
    output = Path(repository_root) / OUTPUT_RELATIVE_PATH
    if output.exists() or output.is_symlink():
        raise FileExistsError("fresh height-role factorized attempt_v1 already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    return output


def _mask_sha256_v11(mask: Any, *, torch: Any) -> str:
    payload = (
        mask.detach()
        .to(device="cpu", dtype=torch.uint8)
        .contiguous()
        .numpy()
        .tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def _names_sha256_v11(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _validate_training_core_v11(
    training_v1: Any,
    training_v3: Any,
    training_v9: Any,
    training_v11: Any,
) -> None:
    _v10._validate_training_core_v10(training_v1, training_v3, training_v9)
    if training_v11.run_fixed_training_v11.__module__ != (
        "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v11_"
        "height_role_factorized_evidence_lift"
    ):
        raise PermissionError("V11 no longer delegates to its reviewed training helper")
    inherited = (
        training_v11.MICROBATCH_SIZE,
        training_v11.MICROBATCHES_PER_UPDATE,
        training_v11.PRESENTATIONS_PER_UPDATE,
        training_v11.MAXIMUM_UPDATES,
        training_v11.MAXIMUM_PRESENTATIONS,
    )
    if inherited != (4, 4, 16, 1_000, 16_000):
        raise PermissionError("V11 changed the frozen training cap")
    parameter_contract = (
        training_v11.HEIGHT_ROLE_BRANCH_ATTENTION_PARAMETER_COUNT_V11,
        training_v11.HEIGHT_ROLE_BRANCH_ATTENTION_PARAMETER_TENSOR_COUNT_V11,
        training_v11.FACTORIZED_SEMANTIC_AXIS_PARAMETER_COUNT_V11,
        training_v11.FACTORIZED_SEMANTIC_AXIS_PARAMETER_TENSOR_COUNT_V11,
    )
    if parameter_contract != (
        HEIGHT_ROLE_ATTENTION_PARAMETER_COUNT_V11,
        HEIGHT_ROLE_ATTENTION_PARAMETER_TENSOR_COUNT_V11,
        HEIGHT_ROLE_SEMANTIC_PARAMETER_COUNT_V11,
        HEIGHT_ROLE_SEMANTIC_PARAMETER_TENSOR_COUNT_V11,
    ):
        raise PermissionError("V11 training-helper parameter contract changed")
    if (
        training_v11.OCCUPIED_SAFETY_AUX_COEFFICIENT != 0.5
        or dict(_v4.AUXILIARY_OBJECTIVE) != AUXILIARY_OBJECTIVE
    ):
        raise PermissionError("V11 changed the inherited occupied objective")


def _validate_model_api_v11(model_api: Any) -> None:
    expected = {
        "HEIGHT_ROLE_INITIALIZATION_SEED_V11": HEIGHT_ROLE_INITIALIZATION_SEED_V11,
        "FLOOR_SUPPORT_INDICES_V11": FLOOR_SUPPORT_INDICES_V11,
        "ELEVATED_SUPPORT_INDICES_V11": ELEVATED_SUPPORT_INDICES_V11,
        "FLOOR_SUPPORT_COUNT_V11": FLOOR_SUPPORT_COUNT_V11,
        "ELEVATED_SUPPORT_COUNT_V11": ELEVATED_SUPPORT_COUNT_V11,
        "HEIGHT_ROLE_ATTENTION_HEADS_V11": HEIGHT_ROLE_ATTENTION_HEADS_V11,
        "HEIGHT_ROLE_ATTENTION_HEAD_WIDTH_V11": HEIGHT_ROLE_ATTENTION_HEAD_WIDTH_V11,
        "HEIGHT_ROLE_ATTENTION_PARAMETER_TENSOR_COUNT_V11": HEIGHT_ROLE_ATTENTION_PARAMETER_TENSOR_COUNT_V11,
        "HEIGHT_ROLE_ATTENTION_PARAMETER_COUNT_V11": HEIGHT_ROLE_ATTENTION_PARAMETER_COUNT_V11,
        "HEIGHT_ROLE_SEMANTIC_PARAMETER_TENSOR_COUNT_V11": HEIGHT_ROLE_SEMANTIC_PARAMETER_TENSOR_COUNT_V11,
        "HEIGHT_ROLE_SEMANTIC_PARAMETER_COUNT_V11": HEIGHT_ROLE_SEMANTIC_PARAMETER_COUNT_V11,
        "FLOOR_VALID_CELL_COUNT_V11": FLOOR_VALID_CELL_COUNT_V11,
        "ELEVATED_VALID_CELL_COUNT_V11": ELEVATED_VALID_CELL_COUNT_V11,
        "FLOOR_VALID_MASK_SHA256_V11": FLOOR_VALID_MASK_SHA256_V11,
        "ELEVATED_VALID_MASK_SHA256_V11": CELL_VALID_MASK_SHA256_V11,
        "FLOOR_WITHIN_TWO_METERS_VALID_CELL_COUNT_V11": NEAR_FIELD_FLOOR_VALID_CELL_COUNT_V11,
        "ELEVATED_WITHIN_TWO_METERS_VALID_CELL_COUNT_V11": NEAR_FIELD_ELEVATED_VALID_CELL_COUNT_V11,
        "ELEVATED_ONLY_VALID_CELL_COUNT_V11": ELEVATED_ONLY_CELL_COUNT_V11,
    }
    for name, value in expected.items():
        if getattr(model_api, name, None) != value:
            raise PermissionError(f"V11 model changed {name}")
    for name in (
        "HeightRoleFactorizedEvidenceLiftV11",
        "HeightRoleFactorizedEvidenceLiftSamplingV11",
        "HeightRoleOccupiedPrioritySemanticDecoderV11",
        "occupied_priority_log_probabilities_v11",
    ):
        if not callable(getattr(model_api, name, None)):
            raise PermissionError(f"V11 model API lacks {name}")
    model_class = getattr(
        model_api, "GeometryAnchoredSweptProgressSurvivalJointJepaV11", None
    )
    if not callable(model_class):
        raise PermissionError("V11 model API lacks its registered model class")
    if (
        getattr(model_api, "GeometryAnchoredDeformableBevLiftJointJepaV1", None)
        is not model_class
    ):
        raise PermissionError("V11 historical runner model alias changed")


def _role_parameter_names_v11(
    model: Any, fresh_v10: Any, training_v11: Any
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    v11 = dict(model.named_parameters())
    v10 = dict(fresh_v10.named_parameters())
    added = tuple(name for name in v11 if name not in v10)
    online_inventory, target_inventory, semantic_inventory = (
        training_v11.v11_parameter_inventories(model)
    )
    online_attention = tuple(
        f"bev_lift.{name}" for name, _ in online_inventory
    )
    target_attention = tuple(
        f"target_bev_lift.{name}" for name, _ in target_inventory
    )
    semantic = tuple(
        f"semantic_head.{name}" for name, _ in semantic_inventory
    )
    if set(added) != set(online_attention + target_attention + semantic):
        raise RuntimeError("V11 added an unregistered parameter family")
    return online_attention, target_attention, semantic


def _migration_receipt_v11(
    model: Any,
    fresh_v10: Any,
    *,
    torch: Any,
    model_api: Any,
    training_v11: Any,
) -> Mapping[str, Any]:
    """Prove exact V10 inheritance outside the two registered replacements."""

    if model.bev_lift.config != fresh_v10.bev_lift.config:
        raise RuntimeError("V11 inherited V10 lift config changed")
    if type(model).__module__ != model_api.__name__:
        raise RuntimeError("V11 model type is not owned by the reviewed module")
    if tuple(model_api.HEIGHT_ROLE_ATTENTION_PARAMETER_SUFFIXES_V11) != tuple(
        training_v11.BRANCH_ATTENTION_PARAMETER_SUFFIXES_V11
    ) or tuple(model_api.HEIGHT_ROLE_SEMANTIC_PARAMETER_SUFFIXES_V11) != tuple(
        training_v11.SEMANTIC_AXIS_PARAMETER_SUFFIXES_V11
    ):
        raise RuntimeError("V11 model/training replacement inventories disagree")

    v11_parameters = dict(model.named_parameters())
    v10_parameters = dict(fresh_v10.named_parameters())
    removed = set(v10_parameters) - set(v11_parameters)
    expected_removed = {
        *(f"bev_lift.{suffix}" for suffix in _v10.ATTENTION_PARAMETER_SUFFIXES_V10),
        *(f"target_bev_lift.{suffix}" for suffix in _v10.ATTENTION_PARAMETER_SUFFIXES_V10),
        *(f"semantic_head.{name}" for name, _ in fresh_v10.semantic_head.named_parameters()),
    }
    if removed != expected_removed:
        raise RuntimeError("V11 replaced parameter inventory changed")
    online_attention, target_attention, semantic = _role_parameter_names_v11(
        model, fresh_v10, training_v11
    )
    if (
        len(online_attention) != HEIGHT_ROLE_ATTENTION_PARAMETER_TENSOR_COUNT_V11
        or sum(v11_parameters[name].numel() for name in online_attention)
        != HEIGHT_ROLE_ATTENTION_PARAMETER_COUNT_V11
        or len(target_attention) != HEIGHT_ROLE_ATTENTION_PARAMETER_TENSOR_COUNT_V11
        or sum(v11_parameters[name].numel() for name in target_attention)
        != HEIGHT_ROLE_ATTENTION_PARAMETER_COUNT_V11
        or len(semantic) != HEIGHT_ROLE_SEMANTIC_PARAMETER_TENSOR_COUNT_V11
        or sum(v11_parameters[name].numel() for name in semantic)
        != HEIGHT_ROLE_SEMANTIC_PARAMETER_COUNT_V11
    ):
        raise RuntimeError("V11 replacement parameter count changed")
    common = tuple(name for name in v10_parameters if name in v11_parameters)
    changed = tuple(
        name
        for name in common
        if not torch.equal(
            v10_parameters[name].detach(), v11_parameters[name].detach()
        )
    )
    if changed:
        raise RuntimeError(f"V11 changed inherited V10 parameter {changed[0]}")

    paired_target = {
        name.removeprefix("target_bev_lift."): parameter
        for name, parameter in v11_parameters.items()
        if name in target_attention
    }
    for name in online_attention:
        suffix = name.removeprefix("bev_lift.")
        if suffix not in paired_target or not torch.equal(
            v11_parameters[name].detach(), paired_target[suffix].detach()
        ):
            raise RuntimeError("V11 online/target role attention initial copy differs")
    if any(v11_parameters[name].requires_grad is False for name in online_attention + semantic):
        raise RuntimeError("V11 online replacement parameter is frozen")
    if any(v11_parameters[name].requires_grad for name in target_attention):
        raise RuntimeError("V11 target role attention is trainable")
    if any(v11_parameters[name].grad is not None for name in target_attention):
        raise RuntimeError("V11 target role attention has an initial gradient")

    v11_buffers = dict(model.named_buffers())
    v10_buffers = dict(fresh_v10.named_buffers())
    removed_buffers = set(v10_buffers) - set(v11_buffers)
    if removed_buffers:
        raise RuntimeError("V11 removed an inherited V10 buffer")
    common_buffers = tuple(name for name in v10_buffers if name in v11_buffers)
    changed_buffers = tuple(
        name
        for name in common_buffers
        if not torch.equal(v10_buffers[name], v11_buffers[name])
    )
    if changed_buffers:
        raise RuntimeError(f"V11 changed inherited V10 buffer {changed_buffers[0]}")
    added_buffers = tuple(name for name in v11_buffers if name not in v10_buffers)
    if not added_buffers or any(
        not name.startswith(("bev_lift.", "target_bev_lift."))
        for name in added_buffers
    ):
        raise RuntimeError("V11 role-buffer inventory changed")

    lift = model.bev_lift
    support_valid = lift.support_valid_mask
    floor_role = lift.floor_support_role_mask
    elevated_role = lift.elevated_support_role_mask
    if tuple(torch.nonzero(floor_role, as_tuple=False).flatten().tolist()) != (
        FLOOR_SUPPORT_INDICES_V11
    ) or tuple(
        torch.nonzero(elevated_role, as_tuple=False).flatten().tolist()
    ) != ELEVATED_SUPPORT_INDICES_V11 or bool(
        (floor_role & elevated_role).any()
    ) or not bool(
        (floor_role | elevated_role).all()
    ):
        raise RuntimeError("V11 fixed support-role masks changed")
    floor_support_valid = support_valid & floor_role
    elevated_support_valid = support_valid & elevated_role
    floor_valid = floor_support_valid.any(dim=-1)
    elevated_valid = elevated_support_valid.any(dim=-1)
    cell_valid = support_valid.any(dim=-1)
    if not torch.equal(elevated_valid, cell_valid):
        raise RuntimeError("V11 elevated validity no longer equals V10 cell validity")
    if (
        int(cell_valid.sum().item()) != CELL_VALID_COUNT_V11
        or _mask_sha256_v11(cell_valid, torch=torch) != CELL_VALID_MASK_SHA256_V11
        or int(floor_valid.sum().item()) != FLOOR_VALID_CELL_COUNT_V11
        or _mask_sha256_v11(floor_valid, torch=torch) != FLOOR_VALID_MASK_SHA256_V11
        or int(elevated_valid.sum().item()) != ELEVATED_VALID_CELL_COUNT_V11
        or int((floor_valid & elevated_valid).sum().item())
        != ROLE_VALID_OVERLAP_CELL_COUNT_V11
        or int((elevated_valid & ~floor_valid).sum().item())
        != ELEVATED_ONLY_CELL_COUNT_V11
    ):
        raise RuntimeError("V11 frozen role-valid count or hash changed")
    near = lift.bev_ground_xyz_m[..., :2].square().sum(dim=-1) <= 4.0
    if (
        int(near.sum().item()) != NEAR_FIELD_CELL_COUNT_V11
        or int((near & floor_valid).sum().item())
        != NEAR_FIELD_FLOOR_VALID_CELL_COUNT_V11
        or int((near & elevated_valid).sum().item())
        != NEAR_FIELD_ELEVATED_VALID_CELL_COUNT_V11
    ):
        raise RuntimeError("V11 frozen near-field role support count changed")

    with torch.no_grad():
        sampling = lift.forward_with_sampling(
            torch.zeros((1, 256, 192), dtype=torch.float32)
        )
    floor_weights = sampling.floor_attention_weights
    elevated_weights = sampling.elevated_attention_weights
    if tuple(floor_weights.shape) != (1, 64, 64, 2, 25) or tuple(
        elevated_weights.shape
    ) != (1, 64, 64, 2, 25):
        raise RuntimeError("V11 role attention receipt shape changed")
    if not bool(torch.isfinite(floor_weights).all()) or not bool(
        torch.isfinite(elevated_weights).all()
    ):
        raise FloatingPointError("V11 role attention weight is nonfinite")
    floor_invalid_weight = (~floor_support_valid)[None, :, :, None, :].expand_as(
        floor_weights
    )
    elevated_invalid_weight = (
        (~elevated_support_valid)[None, :, :, None, :].expand_as(elevated_weights)
    )
    if int(torch.count_nonzero(floor_weights.masked_select(floor_invalid_weight))) or int(
        torch.count_nonzero(elevated_weights.masked_select(elevated_invalid_weight))
    ):
        raise RuntimeError("V11 invalid or cross-role support received attention")
    floor_sums = floor_weights.sum(dim=-1).masked_select(
        floor_valid[None, :, :, None].expand(-1, -1, -1, 2)
    )
    elevated_sums = elevated_weights.sum(dim=-1).masked_select(
        elevated_valid[None, :, :, None].expand(-1, -1, -1, 2)
    )
    if not torch.allclose(floor_sums, torch.ones_like(floor_sums), rtol=0.0, atol=1e-6) or not torch.allclose(
        elevated_sums, torch.ones_like(elevated_sums), rtol=0.0, atol=1e-6
    ):
        raise RuntimeError("V11 valid role attention weights do not sum to one")
    if tuple(sampling.latent.shape) != (1, 64, 64, 64):
        raise RuntimeError("V11 shared latent shape changed")
    logits = model.semantic_logits_from_latent(sampling.latent)
    if tuple(logits.shape) != (1, 3, 64, 64):
        raise RuntimeError("V11 semantic log-probability shape changed")
    learned = logits.permute(0, 2, 3, 1)[0][cell_valid]
    if not bool(torch.isfinite(learned).all()) or not torch.allclose(
        torch.logsumexp(learned, dim=-1),
        torch.zeros_like(learned[:, 0]),
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError("V11 semantic probabilities are not finite and normalized")
    invalid_logits = logits.permute(0, 2, 3, 1)[0][~cell_valid]
    expected_unknown = logits.new_tensor((0.0, -20.0, -20.0))[None].expand_as(
        invalid_logits
    )
    if not torch.equal(invalid_logits, expected_unknown):
        raise RuntimeError("V11 all-invalid semantic output is not exact UNKNOWN")

    return {
        "schema": "lewm_v11_height_role_factorized_evidence_lift_migration_v1",
        "source": "fresh V11 and fresh V10 from identical N320 encoder state",
        "predecessor_experiment_checkpoint_read": False,
        "all_common_v10_parameter_values_bit_exact": True,
        "all_common_v10_buffer_values_bit_exact": True,
        "inherited_state_name_inventory_sha256": _names_sha256_v11(common),
        "removed_v10_parameter_names": sorted(removed),
        "added_role_buffer_names": list(added_buffers),
        "online_branch_attention_parameter_names": list(online_attention),
        "target_branch_attention_parameter_names": list(target_attention),
        "factorized_semantic_parameter_names": list(semantic),
        "online_branch_attention_parameter_count": HEIGHT_ROLE_ATTENTION_PARAMETER_COUNT_V11,
        "target_branch_attention_parameter_count": HEIGHT_ROLE_ATTENTION_PARAMETER_COUNT_V11,
        "factorized_semantic_parameter_count": HEIGHT_ROLE_SEMANTIC_PARAMETER_COUNT_V11,
        "online_target_branch_attention_initial_copy_exact": True,
        "target_branch_attention_initial_gradient_tensor_count": 0,
        "sampling_receipt": {
            "schema": "lewm_v11_height_role_factorized_evidence_sampling_audit_v1",
            "support_roles_disjoint_and_exhaustive": True,
            "floor_support_indices": list(FLOOR_SUPPORT_INDICES_V11),
            "elevated_support_indices": list(ELEVATED_SUPPORT_INDICES_V11),
            "floor_valid_cell_count": int(floor_valid.sum().item()),
            "floor_valid_mask_row_major_uint8_sha256": _mask_sha256_v11(
                floor_valid, torch=torch
            ),
            "elevated_valid_cell_count": int(elevated_valid.sum().item()),
            "elevated_valid_mask_row_major_uint8_sha256": _mask_sha256_v11(
                elevated_valid, torch=torch
            ),
            "role_valid_overlap_cell_count": int(
                (floor_valid & elevated_valid).sum().item()
            ),
            "elevated_only_cell_count": int(
                (elevated_valid & ~floor_valid).sum().item()
            ),
            "near_field_floor_valid_cell_count": int(
                (near & floor_valid).sum().item()
            ),
            "near_field_elevated_valid_cell_count": int(
                (near & elevated_valid).sum().item()
            ),
            "invalid_and_cross_role_attention_exact_zero": True,
            "valid_attention_sums_one_per_head": True,
            "shared_role_ordered_latent_shape": [1, 64, 64, 64],
            "finite_normalized_three_class_log_probabilities": True,
            "all_invalid_semantic_logits_exact_unknown": True,
        },
    }


def _initial_model_receipt_v11(
    model: Any,
    partition: Any,
    migration: Mapping[str, Any],
    *,
    torch: Any,
) -> Mapping[str, Any]:
    online_attention = tuple(migration["online_branch_attention_parameter_names"])
    target_attention = tuple(migration["target_branch_attention_parameter_names"])
    semantic = tuple(migration["factorized_semantic_parameter_names"])
    if tuple(
        name for name in partition.names["lift_semantic"] if name in online_attention
    ) != online_attention or tuple(
        name for name in partition.names["target"] if name in target_attention
    ) != target_attention or tuple(
        name for name in partition.names["lift_semantic"] if name in semantic
    ) != semantic:
        raise RuntimeError("V11 replacement parameter partition changed")
    inventory = [
        name
        for group in ("encoder", "lift_semantic", "predictor", "target")
        for name in partition.names[group]
    ]
    if any(
        inventory.count(name) != 1
        for name in online_attention + target_attention + semantic
    ):
        raise RuntimeError("V11 replacement parameter was not partitioned exactly once")
    parameters = dict(model.named_parameters())
    if any(parameters[name].grad is not None for name in target_attention):
        raise RuntimeError("V11 target role parameter has an initial gradient")
    if int(model.target_hard_sync_count.item()) != 1 or int(
        model.ema_update_count.item()
    ) != 0:
        raise RuntimeError("V11 initial target synchronization counters changed")
    return {
        "schema": "lewm_v11_height_role_factorized_evidence_lift_initial_model_v1",
        "architecture": height_role_factorized_architecture_receipt_v11(),
        "migration": dict(migration),
        "online_branch_attention_parameter_count": sum(
            parameters[name].numel() for name in online_attention
        ),
        "online_branch_attention_parameter_tensor_count": len(online_attention),
        "target_branch_attention_parameter_count": sum(
            parameters[name].numel() for name in target_attention
        ),
        "target_branch_attention_parameter_tensor_count": len(target_attention),
        "factorized_semantic_parameter_count": sum(
            parameters[name].numel() for name in semantic
        ),
        "factorized_semantic_parameter_tensor_count": len(semantic),
        "online_branch_parameter_name_sha256": _names_sha256_v11(online_attention),
        "target_branch_parameter_name_sha256": _names_sha256_v11(target_attention),
        "semantic_axis_parameter_name_sha256": _names_sha256_v11(semantic),
        "all_online_replacement_parameters_in_lift_semantic_exactly_once": True,
        "all_target_branch_parameters_frozen_in_target_exactly_once": True,
        "predictor_consumes_shared_role_ordered_64_channel_state": True,
        "target_initial_copy_exact": True,
        "target_initial_gradient_tensor_count": 0,
        "initial_hard_sync_count": 1,
        "initial_ema_update_count": 0,
    }


def _validate_training_activity_v11(
    diagnostics: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    branch = diagnostics.get("height_role_branch_attention")
    semantic = diagnostics.get("factorized_semantic_axes")
    if not isinstance(branch, Mapping) or branch.get("schema") != (
        "lewm_v11_height_role_branch_attention_training_activity_v1"
    ):
        raise RuntimeError("V11 branch-attention activity receipt changed")
    if not isinstance(semantic, Mapping) or semantic.get("schema") != (
        "lewm_v11_factorized_semantic_axes_training_activity_v1"
    ):
        raise RuntimeError("V11 semantic-axis activity receipt changed")
    for name, receipt in (("branch attention", branch), ("semantic axes", semantic)):
        if receipt.get("all_online_parameter_tensors_active_by_update_2") is not True:
            raise RuntimeError(f"V11 {name} was not fully active by update two")
        if int(receipt.get("target_gradient_tensor_count", 0)) != 0:
            raise RuntimeError(f"V11 {name} recorded a target gradient")
    return branch, semantic


def _run_fixed_training_v11(
    training_v11: Any, *args: Any
) -> tuple[Any, tuple[dict[str, Any], ...], dict[str, Any]]:
    accounting, trace, diagnostics = training_v11.run_fixed_training_v11(*args)
    branch, semantic = _validate_training_activity_v11(diagnostics)
    result = {
        **diagnostics,
        "v11_contract": {
            "schema": "lewm_v11_unchanged_joint_training_contract_v1",
            "objective": "S+P+U+R+O",
            "occupied_auxiliary_coefficient": 0.5,
            "new_loss_or_weight": False,
            "height_role_branch_attention": dict(branch),
            "factorized_semantic_axes": dict(semantic),
            "training_core": "v11_wrapper_over_unchanged_v3_v4_joint_update",
        },
    }
    return accounting, tuple(trace), result


def _physical_calibration_stage_v11(full_arm_passed: bool) -> Mapping[str, Any]:
    result = dict(_v10._physical_calibration_stage_v10(full_arm_passed))
    result["schema"] = "lewm_v11_unchanged_physical_calibration_stage_v1"
    result["source"] = "numerically_unchanged_v10_v4_2016_tuple_protocol"
    result["v10_directional_baselines_are_interpretation_only"] = True
    return result


def execute_v11(*, repository_root: Path = ROOT) -> Mapping[str, Any]:
    repository_root = Path(repository_root).absolute()
    _v1._install_repository_import_roots_v1(repository_root)
    output = _fresh_output_root_v11(repository_root)
    initial_model: Mapping[str, Any] | None = None
    branch_activity: Mapping[str, Any] | None = None
    semantic_activity: Mapping[str, Any] | None = None
    checkpoint_binding: Mapping[str, Any] | None = None
    trace_binding: Mapping[str, Any] | None = None
    try:
        labels_api = importlib.import_module(
            "lewm.benchmarks.go2_swept_progress_survival_labels_v1"
        )
        manifest, rows_by_role = _v1.load_label_bundle_v1(
            repository_root, labels_api=labels_api
        )
        context = _v1._prepare_runtime_v1(repository_root, manifest, labels_api)
        torch, np = context["torch"], context["np"]
        if labels_api.summarize_preflight_v1(
            rows_by_role, context["schedule"]
        ) != manifest.get("preflight"):
            raise PermissionError("label preflight no longer matches its manifest")
        training_v1 = importlib.import_module(
            "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v1"
        )
        training_v3 = importlib.import_module(
            "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v3_"
            "half_occupied_safety_aux"
        )
        training_v9 = importlib.import_module(
            "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v9_"
            "content_adaptive_dense_local_token_lift"
        )
        training_v11 = importlib.import_module(
            "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v11_"
            "height_role_factorized_evidence_lift"
        )
        _validate_training_core_v11(
            training_v1, training_v3, training_v9, training_v11
        )
        frozen = {
            role: training_v1.freeze_role_labels_v1(rows, role=role, np=np)
            for role, rows in rows_by_role.items()
        }
        informative = {
            role: np.asarray(
                [group[0]["informative_state"] for group in labels.state_groups],
                dtype=np.bool_,
            )
            for role, labels in frozen.items()
        }
        pairs = {
            role: context["inputs"].role_pairs(role) for role in ROLE_FILES
        }
        for role in ROLE_FILES:
            training_v1.validate_pairs_against_labels_v1(
                pairs[role], frozen[role]
            )

        model_api = importlib.import_module(
            "lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_"
            "v11_height_role_factorized_evidence_lift"
        )
        _validate_model_api_v11(model_api)
        v10_model_api = importlib.import_module(
            "lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_"
            "v10_projective_cell_volume_token_lift"
        )
        survival_scoring = importlib.import_module(
            "lewm.benchmarks.go2_swept_progress_survival_joint_jepa_v1"
        )
        metrics_api = importlib.import_module(
            "lewm.benchmarks.go2_post_action_projective_support_metrics_v1"
        )
        torch.manual_seed(EXPERIMENT_SEED)
        torch.cuda.manual_seed_all(EXPERIMENT_SEED)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

        n320_state = {
            name: value.detach().cpu().float().contiguous().clone()
            for name, value in context["fit"].encoder.state_dict().items()
        }
        masks = survival_scoring.build_swept_progress_masks_v1()
        current_frame_persistence_masks = (
            survival_scoring.build_current_frame_swept_progress_masks_v1()
        )
        constructor_rng = torch.random.get_rng_state().clone()
        model = model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV11(
            n320_state, masks
        )
        fresh_v10 = (
            v10_model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV10(
                n320_state, masks
            )
        )
        if not torch.equal(torch.random.get_rng_state(), constructor_rng):
            raise RuntimeError("V11 audit constructors did not restore caller CPU RNG")
        migration = _migration_receipt_v11(
            model,
            fresh_v10,
            torch=torch,
            model_api=model_api,
            training_v11=training_v11,
        )
        del fresh_v10

        model = model.to(context["device"])
        model.train()
        partition = training_v1.partition_parameters_v1(model)
        initial_model = _initial_model_receipt_v11(
            model, partition, migration, torch=torch
        )
        optimizer = training_v1.build_frozen_optimizer_v1(partition)
        accounting_state, trace, training_diagnostics = _run_fixed_training_v11(
            training_v11,
            model,
            optimizer,
            context["loader"],
            pairs["train"],
            frozen["train"],
            context["schedule"],
            context["device"],
        )
        accounting = dict(accounting_state.__dict__)
        branch_activity = training_diagnostics["height_role_branch_attention"]
        semantic_activity = training_diagnostics["factorized_semantic_axes"]
        model.eval()
        model.requires_grad_(False)
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in model.state_dict().items()
        }
        checkpoint_buffer = io.BytesIO()
        torch.save(
            {
                "schema": CHECKPOINT_SCHEMA,
                "development_only": True,
                "resume_authorized": False,
                "qualified": False,
                "preregistration_commit": PREREGISTRATION_COMMIT,
                "constructor_initialization_seed": CONSTRUCTOR_INITIALIZATION_SEED,
                "height_role_initialization_seed": HEIGHT_ROLE_INITIALIZATION_SEED_V11,
                "experiment_seed": EXPERIMENT_SEED,
                "initialization_source": (
                    "exact_n320_encoder_and_fresh_v10_source_state_with_only_"
                    "registered_attention_and_semantic_replacements"
                ),
                "predecessor_experiment_checkpoint_read": False,
                "objective": "S+P+U+R+O",
                "inherited_occupied_auxiliary": dict(AUXILIARY_OBJECTIVE),
                "initial_v11_model": initial_model,
                "height_role_branch_attention_activity": branch_activity,
                "factorized_semantic_axes_activity": semantic_activity,
                "training_diagnostics": training_diagnostics,
                "accounting": accounting,
                "model_state_dict": state,
            },
            checkpoint_buffer,
        )
        checkpoint_binding = _v1._atomic_write_v1(
            output / "checkpoint_update_1000.pt", checkpoint_buffer.getvalue()
        )
        _, trace_binding = _v1._write_json_v1(
            output / "training_trace.json",
            {
                "schema": TRACE_SCHEMA,
                "status": "COMPLETE",
                "preregistration_commit": PREREGISTRATION_COMMIT,
                "initial_v11_model": initial_model,
                "height_role_branch_attention_activity": branch_activity,
                "factorized_semantic_axes_activity": semantic_activity,
                "training_diagnostics": training_diagnostics,
                "accounting": accounting,
                "rows": list(trace),
            },
        )

        action_prior_m = (
            frozen["train"].prefix_lengths.mean(axis=0, dtype=np.float64)
            * PROGRESS_SEGMENT_M
        )
        scored = {
            role: _v1.score_role_v1(
                model,
                context["loader"],
                pairs[role],
                frozen[role],
                action_prior_m,
                context["device"],
                torch=torch,
                np=np,
                training_core=training_v1,
                current_frame_persistence_masks=current_frame_persistence_masks,
                metrics_api=metrics_api,
            )
            for role in ("probability_calibration", "checkpoint_selection")
        }
        role_metrics = {
            role: {
                arm: scientific_metrics_v11(
                    scored[role]["scores_m"][arm],
                    frozen[role].prefix_lengths,
                    informative[role],
                    frozen[role].scene_ids,
                    frozen[role].family_ids,
                    np=np,
                )
                for arm in ALL_ARM_NAMES
            }
            for role in scored
        }
        selection_semantic = semantic_metrics_v11(
            scored["checkpoint_selection"]["semantic_confusion"],
            scored["checkpoint_selection"]["rough_semantic_confusion"],
            np=np,
        )
        selection_scores = scored["checkpoint_selection"]["scores_m"]
        selection_labels = frozen["checkpoint_selection"]
        comparisons = {
            name: paired_control_comparison_v11(
                selection_scores["full"],
                selection_scores[name],
                selection_labels.prefix_lengths,
                informative["checkpoint_selection"],
                selection_labels.scene_ids,
                selection_labels.family_ids,
                np=np,
            )
            for name in CONTROL_NAMES
        }
        gate = evaluate_gate_v11(
            role_metrics["checkpoint_selection"],
            selection_semantic,
            comparisons,
        )
        if len(gate.get("checks", {})) != 24:
            raise RuntimeError("V11 inherited 24-check full-arm gate changed")
        full_arm_passed = bool(gate["passed"])
        calibration_stage = _physical_calibration_stage_v11(full_arm_passed)
        access_receipt = _v1._access_receipt_v1(context)
        mask_receipts = {
            "predicted_next_post_action_frame": _v1._mask_receipt_v1(masks),
            "coordinate_matched_current_frame_persistence": _v1._mask_receipt_v1(
                current_frame_persistence_masks
            ),
        }
        result, _ = _v1._write_json_v1(
            output / "result.json",
            {
                "schema": RESULT_SCHEMA,
                "status": (
                    "PASS_FULL_ARM_STAGED_FOR_PHYSICAL_CALIBRATION"
                    if full_arm_passed
                    else "FAIL_DEVELOPMENT_FULL_ARM"
                ),
                "preregistration_commit": PREREGISTRATION_COMMIT,
                "full_arm_gate": gate,
                "gate": gate,
                "physical_evidence_calibration": calibration_stage,
                "caps": {
                    "updates": MAXIMUM_UPDATES,
                    "microbatch_graphs": 4_000,
                    "presentations": MAXIMUM_PRESENTATIONS,
                },
                "seeds": {
                    "inherited_fresh_component_constructor": CONSTRUCTOR_INITIALIZATION_SEED,
                    "height_role_private_cpu_generators": HEIGHT_ROLE_INITIALIZATION_SEED_V11,
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
                    "checkpoint": context["n320_checkpoint"],
                    "encoder_only_initialization": True,
                    "predecessor_experiment_checkpoint_read": False,
                },
                "hardware": context["hardware"],
                "schedule_prefix_sha256": labels_api.v4.SCHEDULE_PREFIX_SHA256,
                "masks": mask_receipts,
                "scientific_change_from_v10": {
                    "only_change": (
                        "height_role_factorized_evidence_lift_and_occupied_priority_"
                        "abstaining_semantic_adapter"
                    ),
                    "initial_v11_model": initial_model,
                    "architecture": height_role_factorized_architecture_receipt_v11(),
                    "objective": "S+P+U+R+O",
                    "inherited_occupied_auxiliary": dict(AUXILIARY_OBJECTIVE),
                    "model_changed": True,
                    "data_changed": False,
                    "dataset_identity_changed": False,
                    "input_tensorization_changed": False,
                    "inherited_v10_state_outside_replacements_bit_exact": True,
                    "optimizer_rules_changed": False,
                    "optimizer_parameter_tensor_membership_changed": True,
                    "losses_changed": False,
                    "new_loss_or_loss_weight": False,
                    "schedule_changed": False,
                    "evaluation_changed": False,
                },
                "training": {
                    "core": (
                        "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_"
                        "v11_height_role_factorized_evidence_lift"
                    ),
                    "accounting": accounting,
                    "diagnostics": training_diagnostics,
                    "height_role_branch_attention_activity": branch_activity,
                    "factorized_semantic_axes_activity": semantic_activity,
                    "joint_from_update_one": True,
                    "separate_head_or_predictor_training": False,
                    "checkpoint_access_status": (
                        "STAGED_FOR_SEPARATE_PHYSICAL_CALIBRATION"
                        if full_arm_passed
                        else "CLOSED_FULL_ARM_GATE_FAILED"
                    ),
                    "checkpoint": checkpoint_binding,
                    "trace": trace_binding,
                },
                "action_prior_mean_progress_m": action_prior_m.tolist(),
                "roles": role_metrics,
                "selection_semantic": selection_semantic,
                "selection_control_comparisons": comparisons,
                "wrong_rgb_mapping_sha256": {
                    role: scored[role]["wrong_rgb_mapping_sha256"] for role in scored
                },
                "determinism": {
                    "algorithms_enabled": bool(
                        torch.are_deterministic_algorithms_enabled()
                    ),
                    "warn_only": True,
                    "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                    "cudnn_deterministic": bool(
                        torch.backends.cudnn.deterministic
                    ),
                    "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                    "matmul_allow_tf32": bool(
                        torch.backends.cuda.matmul.allow_tf32
                    ),
                },
                "access": access_receipt,
                "authority": {
                    "development_only": True,
                    "g2_navigation_final_evaluation_opened": False,
                    "heldout_or_sealed_opened": False,
                    "physical_evidence_gate_passed": False,
                    "checkpoint_qualified": False,
                    "promotion_performed": False,
                    "retry_or_resume_authorized": False,
                    "checkpoint_access_authorized_for_physical_calibration": full_arm_passed,
                },
            },
        )
        return result
    except Exception as error:
        if not (output / "result.json").exists() and not (
            output / "failure.json"
        ).exists():
            try:
                _v1._write_json_v1(
                    output / "failure.json",
                    {
                        "schema": FAILURE_SCHEMA,
                        "status": "FAILED_NO_RETRY_OR_RESUME",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "traceback": traceback.format_exc(),
                        "preregistration_commit": PREREGISTRATION_COMMIT,
                        "height_role_factorized_architecture": (
                            height_role_factorized_architecture_receipt_v11()
                        ),
                        "initial_v11_model": initial_model,
                        "height_role_branch_attention_activity": branch_activity,
                        "factorized_semantic_axes_activity": semantic_activity,
                        "checkpoint": checkpoint_binding,
                        "training_trace": trace_binding,
                        "predecessor_experiment_checkpoint_read": False,
                        "physical_calibration_run_in_this_attempt": False,
                        "authority": {
                            "development_only": True,
                            "g2_navigation_final_evaluation_opened": False,
                            "heldout_or_sealed_opened": False,
                            "checkpoint_qualified": False,
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
    result = execute_v11(repository_root=args.repository_root)
    print(
        _v1._canonical_json_bytes(
            {
                "status": result["status"],
                "result": f"{OUTPUT_RELATIVE_PATH}/result.json",
            }
        ).decode("utf-8")
    )
    return 0 if result["full_arm_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
