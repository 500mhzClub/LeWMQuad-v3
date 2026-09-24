#!/usr/bin/env python3
"""Execute the one-shot V12 neutral disjoint ternary competition probe.

V12 preserves V11's complete height-role model, data, training, predictor,
controls, evaluation, and cap.  Its sole change is the zero-parameter neutral
UNKNOWN/FREE/OCCUPIED composition of the existing disjoint evidence axes.
This executor owns only fresh-attempt authority, source-state receipts,
scoring, and write-once terminalization.
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

_v11 = importlib.import_module(
    "scripts.execute_go2_rgb_swept_progress_survival_joint_jepa_v11_"
    "height_role_factorized_evidence_lift"
)
_v10 = _v11._v10
_v9 = _v11._v9
_v4 = _v11._v4
_v1 = _v11._v1

OUTPUT_RELATIVE_PATH = (
    ".generated/go2_rgb_swept_progress_survival_joint_jepa_v12_"
    "neutral_disjoint_ternary_competition/attempt_v1"
)
CHECKPOINT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v12_"
    "neutral_disjoint_ternary_competition_checkpoint_v1"
)
TRACE_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v12_"
    "neutral_disjoint_ternary_competition_trace_v1"
)
RESULT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v12_"
    "neutral_disjoint_ternary_competition_result_v1"
)
FAILURE_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v12_"
    "neutral_disjoint_ternary_competition_failure_v1"
)
PREREGISTRATION_COMMIT = "ae1568e8f434d715d379eefc3eaf644369154f76"

LABEL_ROOT_RELATIVE_PATH = _v11.LABEL_ROOT_RELATIVE_PATH
LABEL_MANIFEST_NAME = _v11.LABEL_MANIFEST_NAME
LABEL_MANIFEST_FILE_SHA256 = _v11.LABEL_MANIFEST_FILE_SHA256
LABEL_MANIFEST_BYTE_COUNT = _v11.LABEL_MANIFEST_BYTE_COUNT
ACTION_ORDER = _v11.ACTION_ORDER
ROLE_FILES = _v11.ROLE_FILES
MICROBATCH_SIZE = _v11.MICROBATCH_SIZE
MICROBATCHES_PER_UPDATE = _v11.MICROBATCHES_PER_UPDATE
PRESENTATIONS_PER_UPDATE = _v11.PRESENTATIONS_PER_UPDATE
MAXIMUM_UPDATES = _v11.MAXIMUM_UPDATES
MAXIMUM_PRESENTATIONS = _v11.MAXIMUM_PRESENTATIONS
CONSTRUCTOR_INITIALIZATION_SEED = _v11.CONSTRUCTOR_INITIALIZATION_SEED
EXPERIMENT_SEED = _v11.EXPERIMENT_SEED
BOOTSTRAP_SEED = _v11.BOOTSTRAP_SEED
CONTROL_NAMES = _v11.CONTROL_NAMES
ALL_ARM_NAMES = _v11.ALL_ARM_NAMES
GATE_THRESHOLDS = _v11.GATE_THRESHOLDS
PROGRESS_SEGMENT_M = _v11.PROGRESS_SEGMENT_M
AUXILIARY_OBJECTIVE = dict(_v11.AUXILIARY_OBJECTIVE)

HEIGHT_ROLE_INITIALIZATION_SEED_V11 = _v11.HEIGHT_ROLE_INITIALIZATION_SEED_V11
FLOOR_SUPPORT_INDICES_V11 = _v11.FLOOR_SUPPORT_INDICES_V11
ELEVATED_SUPPORT_INDICES_V11 = _v11.ELEVATED_SUPPORT_INDICES_V11
HEIGHT_ROLE_ATTENTION_PARAMETER_TENSOR_COUNT_V11 = (
    _v11.HEIGHT_ROLE_ATTENTION_PARAMETER_TENSOR_COUNT_V11
)
HEIGHT_ROLE_ATTENTION_PARAMETER_COUNT_V11 = (
    _v11.HEIGHT_ROLE_ATTENTION_PARAMETER_COUNT_V11
)
HEIGHT_ROLE_SEMANTIC_PARAMETER_TENSOR_COUNT_V11 = (
    _v11.HEIGHT_ROLE_SEMANTIC_PARAMETER_TENSOR_COUNT_V11
)
HEIGHT_ROLE_SEMANTIC_PARAMETER_COUNT_V11 = (
    _v11.HEIGHT_ROLE_SEMANTIC_PARAMETER_COUNT_V11
)

scientific_metrics_v12 = _v11.scientific_metrics_v11
semantic_metrics_v12 = _v11.semantic_metrics_v11
paired_control_comparison_v12 = _v11.paired_control_comparison_v11
evaluate_gate_v12 = _v11.evaluate_gate_v11


NEUTRAL_DISJOINT_TERNARY_ARCHITECTURE_V12 = {
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


def neutral_disjoint_ternary_architecture_receipt_v12() -> dict[str, Any]:
    return copy.deepcopy(NEUTRAL_DISJOINT_TERNARY_ARCHITECTURE_V12)


def _fresh_output_root_v12(repository_root: Path) -> Path:
    output = Path(repository_root) / OUTPUT_RELATIVE_PATH
    if output.exists() or output.is_symlink():
        raise FileExistsError("fresh neutral-disjoint attempt_v1 already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    return output


def _names_sha256_v12(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _validate_training_core_v12(
    training_v1: Any,
    training_v3: Any,
    training_v9: Any,
    training_v11: Any,
) -> None:
    _v11._validate_training_core_v11(
        training_v1,
        training_v3,
        training_v9,
        training_v11,
    )
    if (
        training_v11.OCCUPIED_SAFETY_AUX_COEFFICIENT != 0.5
        or dict(_v4.AUXILIARY_OBJECTIVE) != AUXILIARY_OBJECTIVE
        or (
            training_v11.MICROBATCH_SIZE,
            training_v11.MICROBATCHES_PER_UPDATE,
            training_v11.PRESENTATIONS_PER_UPDATE,
            training_v11.MAXIMUM_UPDATES,
            training_v11.MAXIMUM_PRESENTATIONS,
        )
        != (4, 4, 16, 1_000, 16_000)
    ):
        raise PermissionError("V12 inherited training identity changed")


def _validate_model_api_v12(model_api: Any) -> None:
    helper = getattr(
        model_api,
        "neutral_disjoint_ternary_log_probabilities_v12",
        None,
    )
    model_class = getattr(
        model_api,
        "GeometryAnchoredSweptProgressSurvivalJointJepaV12",
        None,
    )
    semantic_wrapper = getattr(
        model_api,
        "HeightRoleNeutralDisjointTernarySemanticDecoderV12",
        None,
    )
    if (
        not callable(helper)
        or not callable(model_class)
        or not callable(semantic_wrapper)
    ):
        raise PermissionError("V12 model API lacks its neutral algebra or model class")
    if (
        getattr(model_api, "GeometryAnchoredDeformableBevLiftJointJepaV1", None)
        is not model_class
    ):
        raise PermissionError("V12 historical runner model alias changed")


def _state_identity_receipt_v12(
    model: Any,
    fresh_v11: Any,
    fresh_v10: Any,
    *,
    torch: Any,
    model_api: Any,
    v11_model_api: Any,
    training_v11: Any,
) -> Mapping[str, Any]:
    """Prove zero new state and exact fresh-V11 scientific inheritance."""

    if type(model).__module__ != model_api.__name__:
        raise RuntimeError("V12 model type is not owned by the reviewed module")
    v11_migration = _v11._migration_receipt_v11(
        fresh_v11,
        fresh_v10,
        torch=torch,
        model_api=v11_model_api,
        training_v11=training_v11,
    )

    v12_parameters = dict(model.named_parameters())
    v11_parameters = dict(fresh_v11.named_parameters())
    if tuple(v12_parameters) != tuple(v11_parameters):
        raise RuntimeError("V12 parameter-name inventory differs from fresh V11")
    changed_parameters = tuple(
        name
        for name in v11_parameters
        if (
            tuple(v12_parameters[name].shape) != tuple(v11_parameters[name].shape)
            or v12_parameters[name].dtype != v11_parameters[name].dtype
            or v12_parameters[name].requires_grad
            != v11_parameters[name].requires_grad
            or not torch.equal(
                v12_parameters[name].detach(),
                v11_parameters[name].detach(),
            )
        )
    )
    if changed_parameters:
        raise RuntimeError(
            f"V12 changed fresh V11 parameter {changed_parameters[0]}"
        )

    v12_buffers = dict(model.named_buffers())
    v11_buffers = dict(fresh_v11.named_buffers())
    if tuple(v12_buffers) != tuple(v11_buffers):
        raise RuntimeError("V12 buffer-name inventory differs from fresh V11")
    changed_buffers = tuple(
        name
        for name in v11_buffers
        if (
            tuple(v12_buffers[name].shape) != tuple(v11_buffers[name].shape)
            or v12_buffers[name].dtype != v11_buffers[name].dtype
            or not torch.equal(v12_buffers[name], v11_buffers[name])
        )
    )
    if changed_buffers:
        raise RuntimeError(f"V12 changed fresh V11 buffer {changed_buffers[0]}")

    v12_online, v12_target, v12_semantic = (
        training_v11.v11_parameter_inventories(model)
    )
    v11_online, v11_target, v11_semantic = (
        training_v11.v11_parameter_inventories(fresh_v11)
    )
    for label, observed, witness in (
        ("online attention", v12_online, v11_online),
        ("target attention", v12_target, v11_target),
        ("semantic axes", v12_semantic, v11_semantic),
    ):
        if tuple(name for name, _ in observed) != tuple(
            name for name, _ in witness
        ) or any(
            not torch.equal(left.detach(), right.detach())
            for (_, left), (_, right) in zip(observed, witness, strict=True)
        ):
            raise RuntimeError(f"V12 {label} differs from fresh V11")

    free_parameter_ids = {
        id(parameter) for parameter in model.semantic_head.free_axis.parameters()
    }
    occupied_parameter_ids = {
        id(parameter)
        for parameter in model.semantic_head.occupied_axis.parameters()
    }
    semantic_parameter_ids = {
        id(parameter) for _, parameter in v12_semantic
    }
    if (
        free_parameter_ids & occupied_parameter_ids
        or free_parameter_ids | occupied_parameter_ids != semantic_parameter_ids
    ):
        raise RuntimeError("V12 disjoint semantic-axis module identity changed")

    free_evidence = torch.tensor(
        [[[-3.0, 4.0, 2.0, -2.0]]], dtype=torch.float32
    )
    occupied_evidence = torch.tensor(
        [[[-2.0, 1.0, 5.0, -3.0]]], dtype=torch.float32
    )
    neutral = model_api.neutral_disjoint_ternary_log_probabilities_v12(
        free_evidence,
        occupied_evidence,
    )
    expected_neutral = torch.log_softmax(
        torch.stack(
            (
                torch.zeros_like(free_evidence),
                free_evidence,
                occupied_evidence,
            ),
            dim=1,
        ),
        dim=1,
    )
    if not torch.equal(neutral, expected_neutral) or not torch.equal(
        neutral.argmax(dim=1),
        torch.tensor([[[0, 1, 2, 0]]], dtype=torch.int64),
    ):
        raise RuntimeError("V12 neutral ternary algebra changed")

    with torch.no_grad():
        sampling = model.bev_lift.forward_with_sampling(
            torch.zeros((1, 256, 192), dtype=torch.float32)
        )
        free, occupied = model.semantic_head.evidence_logits(sampling.latent)
        floor_valid = model.bev_lift.floor_cell_valid_mask[None]
        elevated_valid = model.bev_lift.elevated_cell_valid_mask[None]
        free = torch.where(floor_valid, free, torch.full_like(free, -20.0))
        occupied = torch.where(
            elevated_valid,
            occupied,
            torch.full_like(occupied, -20.0),
        )
        expected_logits = model_api.neutral_disjoint_ternary_log_probabilities_v12(
            free,
            occupied,
        )
        valid = model.bev_lift.cell_valid_mask[None, None]
        invalid_logits = expected_logits.new_tensor((0.0, -20.0, -20.0))[
            None, :, None, None
        ]
        expected_logits = torch.where(valid, expected_logits, invalid_logits)
        observed_logits = model.semantic_logits_from_latent(sampling.latent)
    if not torch.equal(observed_logits, expected_logits):
        raise RuntimeError("V12 semantic wrapper changed neutral algebra or masks")
    supported = observed_logits.permute(0, 2, 3, 1)[0][
        model.bev_lift.cell_valid_mask
    ]
    if not bool(torch.isfinite(supported).all()) or not torch.allclose(
        torch.logsumexp(supported, dim=-1),
        torch.zeros_like(supported[:, 0]),
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError("V12 supported-cell probabilities are not normalized")
    observed_invalid = observed_logits.permute(0, 2, 3, 1)[0][
        ~model.bev_lift.cell_valid_mask
    ]
    expected_invalid = observed_logits.new_tensor((0.0, -20.0, -20.0))[
        None
    ].expand_as(observed_invalid)
    if not torch.equal(observed_invalid, expected_invalid):
        raise RuntimeError("V12 all-invalid logits changed")

    state_names = tuple(model.state_dict())
    return {
        "schema": "lewm_v12_fresh_v11_zero_parameter_state_identity_v1",
        "source": "fresh V12 and fresh V11 from identical N320 encoder state",
        "predecessor_experiment_checkpoint_read": False,
        "v11_source_migration_witness": dict(v11_migration),
        "parameter_name_inventory_sha256": _names_sha256_v12(
            tuple(v12_parameters)
        ),
        "buffer_name_inventory_sha256": _names_sha256_v12(tuple(v12_buffers)),
        "state_name_inventory_sha256": _names_sha256_v12(state_names),
        "v12_parameter_tensor_count": len(v12_parameters),
        "v11_parameter_tensor_count": len(v11_parameters),
        "v12_parameter_count": sum(
            parameter.numel() for parameter in v12_parameters.values()
        ),
        "v11_parameter_count": sum(
            parameter.numel() for parameter in v11_parameters.values()
        ),
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


def _initial_model_receipt_v12(
    model: Any,
    partition: Any,
    state_identity: Mapping[str, Any],
    *,
    training_v11: Any,
) -> Mapping[str, Any]:
    online, target, semantic = training_v11.v11_parameter_inventories(model)
    online_names = tuple(f"bev_lift.{name}" for name, _ in online)
    target_names = tuple(f"target_bev_lift.{name}" for name, _ in target)
    semantic_names = tuple(f"semantic_head.{name}" for name, _ in semantic)
    if (
        tuple(
            name
            for name in partition.names["lift_semantic"]
            if name in online_names
        )
        != online_names
        or tuple(
            name for name in partition.names["target"] if name in target_names
        )
        != target_names
        or tuple(
            name
            for name in partition.names["lift_semantic"]
            if name in semantic_names
        )
        != semantic_names
    ):
        raise RuntimeError("V12 inherited replacement parameter partition changed")
    all_names = tuple(
        name
        for group in ("encoder", "lift_semantic", "predictor", "target")
        for name in partition.names[group]
    )
    if any(
        all_names.count(name) != 1
        for name in online_names + target_names + semantic_names
    ):
        raise RuntimeError("V12 inherited parameter was not partitioned exactly once")
    if int(model.target_hard_sync_count.item()) != 1 or int(
        model.ema_update_count.item()
    ) != 0:
        raise RuntimeError("V12 initial EMA synchronization counters changed")
    return {
        "schema": "lewm_v12_neutral_disjoint_ternary_initial_model_v1",
        "architecture": neutral_disjoint_ternary_architecture_receipt_v12(),
        "fresh_v11_state_identity": dict(state_identity),
        "online_branch_attention_parameter_count": sum(
            parameter.numel() for _, parameter in online
        ),
        "online_branch_attention_parameter_tensor_count": len(online),
        "target_branch_attention_parameter_count": sum(
            parameter.numel() for _, parameter in target
        ),
        "target_branch_attention_parameter_tensor_count": len(target),
        "factorized_semantic_parameter_count": sum(
            parameter.numel() for _, parameter in semantic
        ),
        "factorized_semantic_parameter_tensor_count": len(semantic),
        "all_v11_parameters_partitioned_exactly_once": True,
        "optimizer_parameter_membership_changed_from_v11": False,
        "target_initial_gradient_tensor_count": 0,
        "initial_hard_sync_count": 1,
        "initial_ema_update_count": 0,
    }


def _validate_training_activity_v12(
    diagnostics: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    return _v11._validate_training_activity_v11(diagnostics)


def _run_fixed_training_v12(
    training_v11: Any,
    *args: Any,
) -> tuple[Any, tuple[dict[str, Any], ...], dict[str, Any]]:
    accounting, trace, diagnostics = training_v11.run_fixed_training_v11(*args)
    branch, semantic = _validate_training_activity_v12(diagnostics)
    result = {
        **diagnostics,
        "v12_contract": {
            "schema": "lewm_v12_unchanged_joint_training_contract_v1",
            "training_helper": (
                "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v11_"
                "height_role_factorized_evidence_lift"
            ),
            "objective": "S+P+U+R+O",
            "occupied_auxiliary_coefficient": 0.5,
            "new_loss_or_weight": False,
            "height_role_branch_attention": dict(branch),
            "factorized_semantic_axes": dict(semantic),
        },
    }
    return accounting, tuple(trace), result


def _physical_calibration_stage_v12(full_arm_passed: bool) -> Mapping[str, Any]:
    result = dict(_v11._physical_calibration_stage_v11(full_arm_passed))
    result["schema"] = "lewm_v12_unchanged_physical_calibration_stage_v1"
    result["source"] = "numerically_unchanged_v10_v4_2016_tuple_protocol"
    result["physical_calibration_authorized_in_this_attempt"] = False
    return result


def execute_v12(*, repository_root: Path = ROOT) -> Mapping[str, Any]:
    repository_root = Path(repository_root).absolute()
    _v1._install_repository_import_roots_v1(repository_root)
    output = _fresh_output_root_v12(repository_root)
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
            repository_root,
            labels_api=labels_api,
        )
        context = _v1._prepare_runtime_v1(repository_root, manifest, labels_api)
        torch, np = context["torch"], context["np"]
        if labels_api.summarize_preflight_v1(
            rows_by_role,
            context["schedule"],
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
        _validate_training_core_v12(
            training_v1,
            training_v3,
            training_v9,
            training_v11,
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
                pairs[role],
                frozen[role],
            )

        model_api = importlib.import_module(
            "lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_"
            "v12_neutral_disjoint_ternary_competition"
        )
        _validate_model_api_v12(model_api)
        v11_model_api = importlib.import_module(
            "lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_"
            "v11_height_role_factorized_evidence_lift"
        )
        _v11._validate_model_api_v11(v11_model_api)
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
        model = model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV12(
            n320_state,
            masks,
        )
        fresh_v11 = (
            v11_model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV11(
                n320_state,
                masks,
            )
        )
        fresh_v10 = (
            v10_model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV10(
                n320_state,
                masks,
            )
        )
        if not torch.equal(torch.random.get_rng_state(), constructor_rng):
            raise RuntimeError("V12 audit constructors did not restore caller CPU RNG")
        state_identity = _state_identity_receipt_v12(
            model,
            fresh_v11,
            fresh_v10,
            torch=torch,
            model_api=model_api,
            v11_model_api=v11_model_api,
            training_v11=training_v11,
        )
        del fresh_v10, fresh_v11

        model = model.to(context["device"])
        model.train()
        partition = training_v1.partition_parameters_v1(model)
        initial_model = _initial_model_receipt_v12(
            model,
            partition,
            state_identity,
            training_v11=training_v11,
        )
        optimizer = training_v1.build_frozen_optimizer_v1(partition)
        accounting_state, trace, training_diagnostics = _run_fixed_training_v12(
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
                    "accepted_n320_encoder_and_fresh_v11_source_state_with_only_"
                    "zero_parameter_neutral_ternary_algebra"
                ),
                "predecessor_experiment_checkpoint_read": False,
                "objective": "S+P+U+R+O",
                "inherited_occupied_auxiliary": dict(AUXILIARY_OBJECTIVE),
                "initial_v12_model": initial_model,
                "height_role_branch_attention_activity": branch_activity,
                "factorized_semantic_axes_activity": semantic_activity,
                "training_diagnostics": training_diagnostics,
                "accounting": accounting,
                "model_state_dict": state,
            },
            checkpoint_buffer,
        )
        checkpoint_binding = _v1._atomic_write_v1(
            output / "checkpoint_update_1000.pt",
            checkpoint_buffer.getvalue(),
        )
        _, trace_binding = _v1._write_json_v1(
            output / "training_trace.json",
            {
                "schema": TRACE_SCHEMA,
                "status": "COMPLETE",
                "preregistration_commit": PREREGISTRATION_COMMIT,
                "initial_v12_model": initial_model,
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
                arm: scientific_metrics_v12(
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
        selection_semantic = semantic_metrics_v12(
            scored["checkpoint_selection"]["semantic_confusion"],
            scored["checkpoint_selection"]["rough_semantic_confusion"],
            np=np,
        )
        selection_scores = scored["checkpoint_selection"]["scores_m"]
        selection_labels = frozen["checkpoint_selection"]
        comparisons = {
            name: paired_control_comparison_v12(
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
        gate = evaluate_gate_v12(
            role_metrics["checkpoint_selection"],
            selection_semantic,
            comparisons,
        )
        if len(gate.get("checks", {})) != 24:
            raise RuntimeError("V12 inherited 24-check full-arm gate changed")
        full_arm_passed = bool(gate["passed"])
        calibration_stage = _physical_calibration_stage_v12(full_arm_passed)
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
                "scientific_change_from_v11": {
                    "only_change": "neutral_disjoint_ternary_semantic_algebra",
                    "initial_v12_model": initial_model,
                    "architecture": neutral_disjoint_ternary_architecture_receipt_v12(),
                    "objective": "S+P+U+R+O",
                    "inherited_occupied_auxiliary": dict(AUXILIARY_OBJECTIVE),
                    "model_code_changed": True,
                    "parameter_or_buffer_state_changed": False,
                    "added_parameter_count": 0,
                    "data_changed": False,
                    "dataset_identity_changed": False,
                    "input_tensorization_changed": False,
                    "optimizer_rules_changed": False,
                    "optimizer_parameter_tensor_membership_changed": False,
                    "loss_source_or_coefficient_changed": False,
                    "loss_gradient_surface_changed_by_registered_semantic_algebra": True,
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
                    role: scored[role]["wrong_rgb_mapping_sha256"]
                    for role in scored
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
                    "physical_calibration_run": False,
                    "physical_evidence_gate_passed": False,
                    "checkpoint_qualified": False,
                    "promotion_performed": False,
                    "retry_or_resume_authorized": False,
                    "checkpoint_access_authorized_for_physical_calibration": False,
                    "separate_physical_preregistration_required": full_arm_passed,
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
                        "neutral_disjoint_ternary_architecture": (
                            neutral_disjoint_ternary_architecture_receipt_v12()
                        ),
                        "initial_v12_model": initial_model,
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
    result = execute_v12(repository_root=args.repository_root)
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
