#!/usr/bin/env python3
"""Source-only V24 predictor-core-protected survival-output adapter.

V24 privately loads the frozen V23 training source and preserves its direct
survival-output objective exactly.  The only scientific change is the J24
gradient destination: perception plus the two swept-progress output tensors
receive J24, while the thirteen latent predictor-core tensors receive only
their inherited joint-JEPA gradient.  This module performs no experiment I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE_TRAINING_PATH = ROOT / (
    "scripts/run_go2_rgb_action_prior_residualized_wrong_scene_survival_output_"
    "joint_jepa_v23.py"
)
BASE_FROZEN_SOURCE_AND_REVIEW_COMMIT = (
    "44938145362e5accdf8e12b906bfbaa970d62f25"
)
BASE_TRAINING_FILE_SHA256 = (
    "b12bf477313f6c227d1a76c4abcf37209f14a221b6a3ce3f74f015b9a207d911"
)
BASE_TRAINING_BYTE_COUNT = 36_255
BASE_PUBLIC_MODULE_NAME = (
    "scripts.run_go2_rgb_action_prior_residualized_wrong_scene_survival_output_"
    "joint_jepa_v23"
)
PRIVATE_BASE_MODULE_NAME = f"{__name__}.__private_v23_training"
_PUBLIC_BASE_WAS_LOADED_BEFORE_ADAPTER = BASE_PUBLIC_MODULE_NAME in sys.modules

PREREGISTRATION_COMMIT_V24 = "475f1867149f5c5b764973bb5a371de83c29c3eb"
PREREGISTRATION_FILE_SHA256_V24 = (
    "ad0514668b20fd3bb58a2c70e71bb153428f3a9b121c1f8b64ca6e08965c6933"
)
PREREGISTRATION_BYTE_COUNT_V24 = 12_137
PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V24 = (
    "predictor_core_protected_survival_output"
)
PREDICTOR_CORE_PROTECTED_SURVIVAL_GRADIENT_NORM_CAP_V24 = 1.0
PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_TENSOR_COUNT_V24 = 96
PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_COUNT_V24 = 3_106_409
PROTECTED_PREDICTOR_CORE_PARAMETER_TENSOR_COUNT_V24 = 13
PROTECTED_PREDICTOR_CORE_PARAMETER_COUNT_V24 = 259_008
PROTECTED_PREDICTOR_CORE_PARAMETER_NAMES_V24 = (
    "predictor.action_embedding.weight",
    "predictor.input_projection.weight",
    "predictor.input_projection.bias",
    "predictor.residual_blocks.0.conv1.weight",
    "predictor.residual_blocks.0.conv1.bias",
    "predictor.residual_blocks.0.conv2.weight",
    "predictor.residual_blocks.0.conv2.bias",
    "predictor.residual_blocks.1.conv1.weight",
    "predictor.residual_blocks.1.conv1.bias",
    "predictor.residual_blocks.1.conv2.weight",
    "predictor.residual_blocks.1.conv2.bias",
    "predictor.residual_head.weight",
    "predictor.residual_head.bias",
)
SWEPT_PROGRESS_OUTPUT_PARAMETER_NAMES_V24 = (
    "predictor.swept_progress_head.output.weight",
    "predictor.swept_progress_head.output.bias",
)


def _load_private_base_training_v24() -> ModuleType:
    if BASE_TRAINING_PATH.is_symlink() or not BASE_TRAINING_PATH.is_file():
        raise FileNotFoundError("frozen V23 training source is absent or not regular")
    source = BASE_TRAINING_PATH.read_bytes()
    if (
        len(source) != BASE_TRAINING_BYTE_COUNT
        or hashlib.sha256(source).hexdigest() != BASE_TRAINING_FILE_SHA256
    ):
        raise RuntimeError("frozen V23 training source binding changed")
    if PRIVATE_BASE_MODULE_NAME in sys.modules:
        raise RuntimeError("private V23 training module name is already occupied")
    module = ModuleType(PRIVATE_BASE_MODULE_NAME)
    module.__file__ = str(BASE_TRAINING_PATH)
    module.__package__ = None
    module.__cached__ = None
    sys.modules[PRIVATE_BASE_MODULE_NAME] = module
    try:
        exec(
            compile(source, str(BASE_TRAINING_PATH), "exec", dont_inherit=True),
            module.__dict__,
        )
    finally:
        if sys.modules.get(PRIVATE_BASE_MODULE_NAME) is module:
            sys.modules.pop(PRIVATE_BASE_MODULE_NAME)
    return module


_v23 = _load_private_base_training_v24()
_base = _v23._base
_tensor_core = _v23._tensor_core
if (
    _v23.STATE_RESIDUAL_SURVIVAL_PARAMETER_TENSOR_COUNT_V23 != 109
    or _v23.STATE_RESIDUAL_SURVIVAL_PARAMETER_COUNT_V23 != 3_365_417
    or _v23.MAXIMUM_UPDATES != 1_000
    or _v23.MAXIMUM_PRESENTATIONS != 16_000
):
    raise RuntimeError("frozen V23 route or cap identity changed")

for _name in _v23.__all__:
    globals()[_name] = getattr(_v23, _name)

REQUIRED_BATCH_KEYS_V24 = tuple(_v23.REQUIRED_BATCH_KEYS_V23)


@dataclass(frozen=True)
class JointTrainingAccountingV24:
    updates: int = 0
    presentations: int = 0
    microbatch_graphs: int = 0
    backward_calls: int = 0
    camera_route_grad_calls: int = 0
    joint_route_grad_calls: int = 0
    predictor_core_protected_survival_grad_calls: int = 0
    camera_frame_objectives: int = 0
    optimizer_steps: int = 0
    ema_steps: int = 0
    predictor_forwards: int = 0
    predictor_objectives: int = 0
    predictor_core_protected_survival_objectives: int = 0


@dataclass(frozen=True)
class JointUpdateResultV24:
    accounting: JointTrainingAccountingV24
    mean_losses: Mapping[str, float]
    gradient_routes: Mapping[str, Any]
    gradient_l2: Mapping[str, float]
    ranking_active_microbatches: int
    ranking_eligible_pairs: int
    survival_supervised_decisions: int
    target_gradient_tensor_count: int
    optimizer_steps_this_update: int
    ema_steps_this_update: int
    predictor_core_protected_survival_diagnostics: Mapping[str, float | int]


@dataclass(frozen=True)
class PredictorCoreProtectedSurvivalParameterSubsetV24:
    parameters: tuple[Any, ...]
    names: tuple[str, ...]
    parameter_count: int
    protected_predictor_core_parameters: tuple[Any, ...]
    protected_predictor_core_names: tuple[str, ...]
    protected_predictor_core_parameter_count: int


@dataclass(frozen=True)
class PredictorCoreProtectedSurvivalObjectiveV24:
    loss: Any
    fit: Any
    rank: Any
    positive_energy: Any
    scene_negative_energy: Any
    prior_negative_energy: Any
    scene_eligible: Any
    prior_eligible: Any
    scene_rank_sum: Any
    prior_rank_sum: Any
    scene_advantage_sum: Any
    prior_advantage_sum: Any
    scene_eligible_count: int
    prior_eligible_count: int


def validate_accounting_v24(accounting: JointTrainingAccountingV24) -> None:
    if not isinstance(accounting, JointTrainingAccountingV24):
        raise TypeError("V24 accounting has the wrong receipt type")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in accounting.__dict__.values()
    ):
        raise ValueError("V24 accounting values must be nonnegative integers")
    updates = accounting.updates
    expected = JointTrainingAccountingV24(
        updates=updates,
        presentations=updates * PRESENTATIONS_PER_UPDATE,
        microbatch_graphs=updates * MICROBATCHES_PER_UPDATE,
        backward_calls=updates * 3 * MICROBATCHES_PER_UPDATE,
        camera_route_grad_calls=updates * MICROBATCHES_PER_UPDATE,
        joint_route_grad_calls=updates * MICROBATCHES_PER_UPDATE,
        predictor_core_protected_survival_grad_calls=(
            updates * MICROBATCHES_PER_UPDATE
        ),
        camera_frame_objectives=(
            updates * 2 * MICROBATCH_SIZE * MICROBATCHES_PER_UPDATE
        ),
        optimizer_steps=updates,
        ema_steps=updates,
        predictor_forwards=updates * MICROBATCHES_PER_UPDATE,
        predictor_objectives=updates * 2 * MICROBATCHES_PER_UPDATE,
        predictor_core_protected_survival_objectives=(
            updates * MICROBATCHES_PER_UPDATE
        ),
    )
    if accounting != expected:
        raise RuntimeError("V24 accounting is inconsistent")


def _advance_accounting_v24(
    accounting: JointTrainingAccountingV24,
) -> JointTrainingAccountingV24:
    return JointTrainingAccountingV24(
        updates=accounting.updates + 1,
        presentations=accounting.presentations + PRESENTATIONS_PER_UPDATE,
        microbatch_graphs=accounting.microbatch_graphs + MICROBATCHES_PER_UPDATE,
        backward_calls=accounting.backward_calls + 3 * MICROBATCHES_PER_UPDATE,
        camera_route_grad_calls=(
            accounting.camera_route_grad_calls + MICROBATCHES_PER_UPDATE
        ),
        joint_route_grad_calls=(
            accounting.joint_route_grad_calls + MICROBATCHES_PER_UPDATE
        ),
        predictor_core_protected_survival_grad_calls=(
            accounting.predictor_core_protected_survival_grad_calls
            + MICROBATCHES_PER_UPDATE
        ),
        camera_frame_objectives=(
            accounting.camera_frame_objectives
            + 2 * MICROBATCH_SIZE * MICROBATCHES_PER_UPDATE
        ),
        optimizer_steps=accounting.optimizer_steps + 1,
        ema_steps=accounting.ema_steps + 1,
        predictor_forwards=accounting.predictor_forwards + MICROBATCHES_PER_UPDATE,
        predictor_objectives=(
            accounting.predictor_objectives + 2 * MICROBATCHES_PER_UPDATE
        ),
        predictor_core_protected_survival_objectives=(
            accounting.predictor_core_protected_survival_objectives
            + MICROBATCHES_PER_UPDATE
        ),
    )


def _validate_update_capacity_v24(accounting: JointTrainingAccountingV24) -> None:
    validate_accounting_v24(accounting)
    if (
        accounting.updates >= MAXIMUM_UPDATES
        or accounting.presentations + PRESENTATIONS_PER_UPDATE > MAXIMUM_PRESENTATIONS
    ):
        raise PermissionError("V24 training cap leaves no complete update available")


def _validate_microbatches_v24(
    torch: Any, microbatches: Sequence[Mapping[str, Any]]
) -> None:
    if len(microbatches) != MICROBATCHES_PER_UPDATE:
        raise ValueError("V24 update must contain exactly four microbatches")
    if any(tuple(batch) != REQUIRED_BATCH_KEYS_V24 for batch in microbatches):
        raise ValueError("V24 microbatch schema changed from frozen V23")
    _v23._validate_microbatches_v23(torch, microbatches)


def predictor_core_protected_survival_parameter_subset_v24(
    partition: Any,
) -> PredictorCoreProtectedSurvivalParameterSubsetV24:
    """Select exact J24 recipients and bind the excluded predictor core."""

    encoder = tuple(zip(partition.names["encoder"], partition.encoder, strict=True))
    evidence = tuple(
        zip(partition.names["evidence_head"], partition.evidence_head, strict=True)
    )
    representation = tuple(
        (name, parameter)
        for name, parameter in zip(
            partition.names["representation"], partition.representation, strict=True
        )
        if name.startswith(
            ("bev_lift.point_projection.", "bev_lift.volume_block.")
        )
    )
    predictor = tuple(
        zip(partition.names["predictor"], partition.predictor, strict=True)
    )
    output = tuple(
        (name, parameter)
        for name, parameter in predictor
        if name in SWEPT_PROGRESS_OUTPUT_PARAMETER_NAMES_V24
    )
    protected = tuple(
        (name, parameter)
        for name, parameter in predictor
        if name not in SWEPT_PROGRESS_OUTPUT_PARAMETER_NAMES_V24
    )
    selected = (*encoder, *evidence, *representation, *output)
    names = tuple(name for name, _ in selected)
    parameters = tuple(parameter for _, parameter in selected)
    protected_names = tuple(name for name, _ in protected)
    protected_parameters = tuple(parameter for _, parameter in protected)
    parameter_count = sum(int(parameter.numel()) for parameter in parameters)
    protected_count = sum(
        int(parameter.numel()) for parameter in protected_parameters
    )
    if (
        len(parameters)
        != PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_TENSOR_COUNT_V24
        or parameter_count
        != PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_COUNT_V24
        or len(protected_parameters)
        != PROTECTED_PREDICTOR_CORE_PARAMETER_TENSOR_COUNT_V24
        or protected_count != PROTECTED_PREDICTOR_CORE_PARAMETER_COUNT_V24
        or protected_names != PROTECTED_PREDICTOR_CORE_PARAMETER_NAMES_V24
        or tuple(name for name, _ in output)
        != SWEPT_PROGRESS_OUTPUT_PARAMETER_NAMES_V24
        or any(name.startswith("semantic_head.") for name in names)
        or any(name.startswith(("target_encoder.", "target_bev_lift.")) for name in names)
        or any(name.startswith("predictor.") for name in names[:-2])
        or any(name in names for name in protected_names)
        or len({id(parameter) for parameter in (*parameters, *protected_parameters)})
        != len(parameters) + len(protected_parameters)
        or len(parameters) + len(protected_parameters) != 109
        or parameter_count + protected_count != 3_365_417
    ):
        raise RuntimeError("V24 predictor-core-protected parameter subset changed")
    return PredictorCoreProtectedSurvivalParameterSubsetV24(
        parameters=parameters,
        names=names,
        parameter_count=parameter_count,
        protected_predictor_core_parameters=protected_parameters,
        protected_predictor_core_names=protected_names,
        protected_predictor_core_parameter_count=protected_count,
    )


def predictor_core_protected_survival_objective_v24(
    torch: Any,
    survival_api: Any,
    survival_logits: Any,
    prefix_lengths: Any,
    negative_rows: Any,
    action_prior_m: Any,
) -> PredictorCoreProtectedSurvivalObjectiveV24:
    """Return V23's exact F, R, and total under V24 receipt identity."""

    inherited = _v23.state_residual_survival_objective_v23(
        torch,
        survival_api,
        survival_logits,
        prefix_lengths,
        negative_rows,
        action_prior_m,
    )
    return PredictorCoreProtectedSurvivalObjectiveV24(
        **{
            name: getattr(inherited, name)
            for name in PredictorCoreProtectedSurvivalObjectiveV24.__dataclass_fields__
        }
    )


def joint_training_update_v24(
    model: Any,
    optimizer: Any,
    microbatches: Sequence[Mapping[str, Any]],
    *,
    accounting: JointTrainingAccountingV24 | None = None,
) -> JointUpdateResultV24:
    """Run one inherited joint-JEPA update plus core-protected J24."""

    torch, semantic_api, survival_api, *_ = _tensor_core._runtime_apis()
    state = JointTrainingAccountingV24() if accounting is None else accounting
    _validate_update_capacity_v24(state)
    _validate_microbatches_v24(torch, microbatches)
    partition = _base.partition_parameters_v18(model)
    _base.validate_optimizer_v18(optimizer, partition)
    auxiliary_subset = predictor_core_protected_survival_parameter_subset_v24(
        partition
    )
    auxiliary_parameter_ids = {id(value) for value in auxiliary_subset.parameters}
    protected_parameter_ids = {
        id(value) for value in auxiliary_subset.protected_predictor_core_parameters
    }
    if auxiliary_parameter_ids & protected_parameter_ids:
        raise RuntimeError("V24 protected predictor core entered the J24 route")
    ema_before = int(model.ema_update_count.item())
    if ema_before != state.ema_steps:
        raise RuntimeError("model EMA count disagrees with V24 accounting")
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("V24 EMA target already has a gradient")

    optimizer.zero_grad(set_to_none=True)
    camera_shared = _tensor_core._zero_accumulators(partition.shared)
    joint_shared = _tensor_core._zero_accumulators(partition.shared)
    joint_representation = _tensor_core._zero_accumulators(partition.representation)
    joint_predictor = _tensor_core._zero_accumulators(partition.predictor)
    auxiliary_gradients = _tensor_core._zero_accumulators(
        auxiliary_subset.parameters
    )
    absent = {
        name: 0
        for name in (
            "camera_shared",
            "joint_shared",
            "representation",
            "predictor",
            PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V24,
        )
    }
    sums = {
        name: 0.0
        for name in (
            "S", "P", "U", "R", "O", "F", "J_rank", "J24", "N", "C", "L",
            "positive_energy_sum", "scene_negative_energy_sum",
            "prior_negative_energy_sum", "scene_advantage_sum",
            "prior_advantage_sum", "scene_rank_sum", "prior_rank_sum",
        )
    }
    active_ranking = eligible_pairs = supervised_decisions = 0
    scene_count = prior_count = positive_count = 0

    for batch in microbatches:
        current_encoding = model.encode_online_training(
            batch[CURRENT_RGB_KEY],
            camera_origin_body_m=batch[CURRENT_CAMERA_ORIGIN_KEY],
            camera_basis_body_fru=batch[CURRENT_CAMERA_BASIS_KEY],
            ground_plane_z_body_m=batch[CURRENT_GROUND_PLANE_Z_KEY],
        )
        next_encoding = model.encode_online_training(
            batch[NEXT_RGB_KEY],
            camera_origin_body_m=batch[NEXT_CAMERA_ORIGIN_KEY],
            camera_basis_body_fru=batch[NEXT_CAMERA_BASIS_KEY],
            ground_plane_z_body_m=batch[NEXT_GROUND_PLANE_Z_KEY],
        )
        current_latent = current_encoding.latent
        next_latent = next_encoding.latent
        current_logits = model.semantic_logits_from_latent(current_latent)
        next_logits = model.semantic_logits_from_latent(next_latent)
        semantic = semantic_api.semantic_loss_v1(
            current_logits,
            batch[CURRENT_LABELS_KEY],
            next_logits,
            batch[NEXT_LABELS_KEY],
        )
        occupied = _tensor_core._v3.occupied_safety_aux_loss_v3(
            current_logits,
            batch[CURRENT_LABELS_KEY],
            next_logits,
            batch[NEXT_LABELS_KEY],
        )
        prediction = model.predict_all_actions_with_survival(current_latent)
        predicted, survival_logits = _tensor_core._v3._v2._v1._prediction_parts(
            prediction
        )
        with torch.no_grad():
            ema_current = model.encode_target(batch[CURRENT_RGB_KEY])
            ema_next = model.encode_target(batch[NEXT_RGB_KEY])
        auxiliary = predictor_core_protected_survival_objective_v24(
            torch,
            survival_api,
            survival_logits,
            batch[PREFIX_LENGTHS_KEY],
            batch[SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21],
            batch[ACTION_PRIOR_M_KEY_V23],
        )
        persistence = semantic_api.microbatch_persistence_loss_v1(
            predicted,
            batch[EXECUTED_ACTION_KEY],
            ema_current,
            ema_next,
        )
        joint = survival_api.joint_survival_loss_v1(
            semantic_loss=semantic.loss,
            executed_action_ema_latent_loss=persistence.loss,
            survival_logits=survival_logits,
            immediate_feasible=batch[IMMEDIATE_FEASIBLE_KEY],
            prefix_lengths=batch[PREFIX_LENGTHS_KEY],
        )
        navigation = joint.loss + occupied.loss
        camera = _base.camera_evidence_pair_loss_v13(
            current_encoding.auxiliary_evidence,
            next_encoding.auxiliary_evidence,
            CameraEvidenceFrameSupervisionV13(
                batch[CURRENT_PIXEL_HIT_KEY],
                batch[CURRENT_PIXEL_DISTANCE_KEY],
                batch[CURRENT_GROUND_IN_FRUSTUM_KEY],
                batch[CURRENT_GROUND_CLEAR_KEY],
            ),
            CameraEvidenceFrameSupervisionV13(
                batch[NEXT_PIXEL_HIT_KEY],
                batch[NEXT_PIXEL_DISTANCE_KEY],
                batch[NEXT_GROUND_IN_FRUSTUM_KEY],
                batch[NEXT_GROUND_CLEAR_KEY],
            ),
        )
        for name, value in (
            ("current latent", current_latent),
            ("next latent", next_latent),
            ("current semantic logits", current_logits),
            ("next semantic logits", next_logits),
            ("predicted latent", predicted),
            ("survival logits", survival_logits),
            ("predictor-core-protected survival J24", auxiliary.loss),
            ("joint N", navigation),
            ("Camera C", camera.total),
        ):
            _tensor_core._finite_tensor(torch, value, name)
        if (
            not camera.total.requires_grad
            or not navigation.requires_grad
            or not auxiliary.loss.requires_grad
        ):
            raise RuntimeError("V24 C, N, and J24 must retain gradient graphs")

        c_gradients = torch.autograd.grad(
            camera.total / MICROBATCHES_PER_UPDATE,
            partition.shared,
            retain_graph=True,
            allow_unused=True,
        )
        n_parameters = partition.shared + partition.representation + partition.predictor
        n_gradients = torch.autograd.grad(
            navigation / MICROBATCHES_PER_UPDATE,
            n_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        j_gradients = torch.autograd.grad(
            auxiliary.loss / MICROBATCHES_PER_UPDATE,
            auxiliary_subset.parameters,
            allow_unused=True,
        )
        shared_end = len(partition.shared)
        representation_end = shared_end + len(partition.representation)
        absent["camera_shared"] += _tensor_core._accumulate_gradients(
            camera_shared, c_gradients
        )
        absent["joint_shared"] += _tensor_core._accumulate_gradients(
            joint_shared, n_gradients[:shared_end]
        )
        absent["representation"] += _tensor_core._accumulate_gradients(
            joint_representation, n_gradients[shared_end:representation_end]
        )
        absent["predictor"] += _tensor_core._accumulate_gradients(
            joint_predictor, n_gradients[representation_end:]
        )
        absent[PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V24] += (
            _tensor_core._accumulate_gradients(auxiliary_gradients, j_gradients)
        )

        values = {
            "S": joint.semantic,
            "P": joint.executed_action_ema_latent,
            "U": joint.survival,
            "R": joint.progress_ranking,
            "O": occupied.loss,
            "F": auxiliary.fit,
            "J_rank": auxiliary.rank,
            "J24": auxiliary.loss,
            "N": navigation,
            "C": camera.total,
            "L": navigation + camera.total + auxiliary.loss,
            "positive_energy_sum": auxiliary.positive_energy.sum(),
            "scene_negative_energy_sum": auxiliary.scene_negative_energy[
                auxiliary.scene_eligible
            ].sum(),
            "prior_negative_energy_sum": auxiliary.prior_negative_energy[
                auxiliary.prior_eligible
            ].sum(),
            "scene_advantage_sum": auxiliary.scene_advantage_sum,
            "prior_advantage_sum": auxiliary.prior_advantage_sum,
            "scene_rank_sum": auxiliary.scene_rank_sum,
            "prior_rank_sum": auxiliary.prior_rank_sum,
        }
        for name, value in values.items():
            sums[name] += _tensor_core._scalar(value)
        positive_count += MICROBATCH_SIZE * NON_HOLD_ACTION_COUNT_V23
        scene_count += auxiliary.scene_eligible_count
        prior_count += auxiliary.prior_eligible_count
        pairs = int(joint.ranking_terms.eligible_pair_count.item())
        active_ranking += int(pairs > 0)
        eligible_pairs += pairs
        supervised_decisions += int(
            joint.survival_terms.supervised_decision_count.item()
        )

    route_tensors = {
        "camera_shared": (partition.shared, camera_shared),
        "joint_shared": (partition.shared, joint_shared),
        "representation": (partition.representation, joint_representation),
        "predictor": (partition.predictor, joint_predictor),
        PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V24: (
            auxiliary_subset.parameters,
            auxiliary_gradients,
        ),
    }
    route_values = {
        name: _tensor_core._route_norm_and_scale_v13(torch, gradients)
        for name, (_, gradients) in route_tensors.items()
    }
    for name in (
        "camera_shared",
        "joint_shared",
        "predictor",
        PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V24,
    ):
        if not (_tensor_core._scalar(route_values[name][0]) > 0.0):
            raise RuntimeError(f"required V24 gradient route {name!r} is zero")
    if absent[PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V24] != 0:
        raise RuntimeError("V24 output auxiliary has an absent gradient")

    c_scale = route_values["camera_shared"][1]
    n_scale = route_values["joint_shared"][1]
    representation_scale = route_values["representation"][1]
    predictor_scale = route_values["predictor"][1]
    auxiliary_scale = route_values[
        PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V24
    ][1]
    auxiliary_by_id = {
        id(parameter): gradient
        for parameter, gradient in zip(
            auxiliary_subset.parameters, auxiliary_gradients, strict=True
        )
    }
    for parameter, c_gradient, n_gradient in zip(
        partition.shared, camera_shared, joint_shared, strict=True
    ):
        parameter.grad = (
            c_scale * c_gradient
            + n_scale * n_gradient
            + auxiliary_scale * auxiliary_by_id[id(parameter)]
        )
    for parameter, inherited_gradient in zip(
        partition.representation, joint_representation, strict=True
    ):
        gradient = representation_scale * inherited_gradient
        if id(parameter) in auxiliary_by_id:
            gradient = gradient + auxiliary_scale * auxiliary_by_id[id(parameter)]
        parameter.grad = gradient
    for name, parameter, inherited_gradient in zip(
        partition.names["predictor"],
        partition.predictor,
        joint_predictor,
        strict=True,
    ):
        gradient = predictor_scale * inherited_gradient
        if name in SWEPT_PROGRESS_OUTPUT_PARAMETER_NAMES_V24:
            if id(parameter) not in auxiliary_by_id:
                raise RuntimeError("V24 swept-progress output left the J24 route")
            gradient = gradient + auxiliary_scale * auxiliary_by_id[id(parameter)]
        elif id(parameter) in auxiliary_by_id:
            raise RuntimeError("V24 latent predictor core received J24")
        parameter.grad = gradient

    target_gradient_count = sum(
        parameter.grad is not None for parameter in partition.target
    )
    if target_gradient_count:
        raise RuntimeError("V24 EMA target received a gradient")
    optimizer.step()
    for parameter in partition.online:
        _tensor_core._finite_tensor(torch, parameter, "V24 online parameter")
    model.update_target_ema_after_optimizer_step()
    ema_after = int(model.ema_update_count.item())
    if ema_after != ema_before + 1:
        raise RuntimeError("V24 EMA did not update exactly once")
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("V24 EMA target received a gradient")

    advanced = _advance_accounting_v24(state)
    if advanced.ema_steps != ema_after:
        raise RuntimeError("post-update V24 EMA count disagrees with accounting")
    receipts = {
        name: _tensor_core._receipt_v13(
            route_values[name][0],
            route_values[name][1],
            parameters,
            absent[name],
        )
        for name, (parameters, _) in route_tensors.items()
    }
    mean = {
        name: sums[name] / MICROBATCHES_PER_UPDATE
        for name in ("S", "P", "U", "R", "O", "F", "J_rank", "J24", "N", "C", "L")
    }
    if positive_count != PRESENTATIONS_PER_UPDATE * NON_HOLD_ACTION_COUNT_V23:
        raise RuntimeError("V24 positive-energy accounting changed")
    if scene_count < MICROBATCHES_PER_UPDATE or prior_count < MICROBATCHES_PER_UPDATE:
        raise RuntimeError("V24 comparison accounting changed")
    return JointUpdateResultV24(
        accounting=advanced,
        mean_losses=mean,
        gradient_routes=receipts,
        gradient_l2={name: receipt.preclip_l2 for name, receipt in receipts.items()},
        ranking_active_microbatches=active_ranking,
        ranking_eligible_pairs=eligible_pairs,
        survival_supervised_decisions=supervised_decisions,
        target_gradient_tensor_count=target_gradient_count,
        optimizer_steps_this_update=1,
        ema_steps_this_update=1,
        predictor_core_protected_survival_diagnostics={
            "positive_energy_sum": sums["positive_energy_sum"],
            "positive_energy_count": positive_count,
            "positive_energy_mean": sums["positive_energy_sum"] / positive_count,
            "scene_negative_energy_sum": sums["scene_negative_energy_sum"],
            "scene_eligible_count": scene_count,
            "scene_negative_energy_mean": sums["scene_negative_energy_sum"] / scene_count,
            "scene_advantage_sum": sums["scene_advantage_sum"],
            "scene_advantage_mean": sums["scene_advantage_sum"] / scene_count,
            "scene_rank_sum": sums["scene_rank_sum"],
            "prior_negative_energy_sum": sums["prior_negative_energy_sum"],
            "prior_eligible_count": prior_count,
            "prior_negative_energy_mean": sums["prior_negative_energy_sum"] / prior_count,
            "prior_advantage_sum": sums["prior_advantage_sum"],
            "prior_advantage_mean": sums["prior_advantage_sum"] / prior_count,
            "prior_rank_sum": sums["prior_rank_sum"],
            "non_hold_action_count_per_row": NON_HOLD_ACTION_COUNT_V23,
        },
    )


def private_training_adapter_receipt_v24() -> dict[str, Any]:
    return {
        "schema": (
            "lewm_go2_rgb_predictor_core_protected_survival_output_joint_jepa_"
            "v24_training_adapter_v1"
        ),
        "base_training": str(BASE_TRAINING_PATH.relative_to(ROOT)),
        "base_frozen_source_and_review_commit": BASE_FROZEN_SOURCE_AND_REVIEW_COMMIT,
        "base_training_file_sha256": BASE_TRAINING_FILE_SHA256,
        "base_training_byte_count": BASE_TRAINING_BYTE_COUNT,
        "public_base_was_loaded_before_adapter": _PUBLIC_BASE_WAS_LOADED_BEFORE_ADAPTER,
        "public_base_loaded_by_adapter": False,
        "private_module_registered": PRIVATE_BASE_MODULE_NAME in sys.modules,
        "preregistration_commit": PREREGISTRATION_COMMIT_V24,
        "preregistration_file_sha256": PREREGISTRATION_FILE_SHA256_V24,
        "preregistration_byte_count": PREREGISTRATION_BYTE_COUNT_V24,
        "objective_bit_identical_to_v23": True,
        "predictor_core_protected_survival_gradient_norm_cap": (
            PREDICTOR_CORE_PROTECTED_SURVIVAL_GRADIENT_NORM_CAP_V24
        ),
        "j24_parameter_tensor_count": (
            PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_TENSOR_COUNT_V24
        ),
        "j24_parameter_count": (
            PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_COUNT_V24
        ),
        "protected_predictor_core_parameter_tensor_count": (
            PROTECTED_PREDICTOR_CORE_PARAMETER_TENSOR_COUNT_V24
        ),
        "protected_predictor_core_parameter_count": (
            PROTECTED_PREDICTOR_CORE_PARAMETER_COUNT_V24
        ),
        "inherited_joint_predictor_parameter_tensor_count": 15,
        "maximum_updates": MAXIMUM_UPDATES,
        "maximum_presentations": MAXIMUM_PRESENTATIONS,
    }


partition_parameters_v24 = _base.partition_parameters_v18
build_frozen_optimizer_v24 = _base.build_frozen_optimizer_v18
validate_optimizer_v24 = _base.validate_optimizer_v18
J24_PARAMETER_TENSOR_COUNT_V24 = (
    PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_TENSOR_COUNT_V24
)
J24_PARAMETER_COUNT_V24 = PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_COUNT_V24
PROTECTED_PREDICTOR_CORE_TENSOR_COUNT_V24 = (
    PROTECTED_PREDICTOR_CORE_PARAMETER_TENSOR_COUNT_V24
)

# Compatibility names consumed by the unchanged inherited lifecycle.
joint_training_update_v23 = joint_training_update_v24
validate_accounting_v23 = validate_accounting_v24
state_residual_survival_objective_v23 = predictor_core_protected_survival_objective_v24
state_residual_survival_parameter_subset_v23 = (
    predictor_core_protected_survival_parameter_subset_v24
)
joint_training_update_v21 = joint_training_update_v24
validate_accounting_v21 = validate_accounting_v24
partition_parameters_v21 = partition_parameters_v24
build_frozen_optimizer_v21 = build_frozen_optimizer_v24
validate_optimizer_v21 = validate_optimizer_v24
joint_training_update_v19 = joint_training_update_v24
validate_accounting_v19 = validate_accounting_v24
joint_training_update_v18 = joint_training_update_v24
validate_accounting_v18 = validate_accounting_v24
JointTrainingAccountingV13 = JointTrainingAccountingV24
JointTrainingAccountingV19 = JointTrainingAccountingV24
JointTrainingAccountingV21 = JointTrainingAccountingV24
JointTrainingAccountingV23 = JointTrainingAccountingV24
JointUpdateResultV13 = JointUpdateResultV24
JointUpdateResultV19 = JointUpdateResultV24
JointUpdateResultV21 = JointUpdateResultV24
JointUpdateResultV23 = JointUpdateResultV24
partition_parameters_v13 = partition_parameters_v24
build_frozen_optimizer_v13 = build_frozen_optimizer_v24
validate_optimizer_v13 = validate_optimizer_v24
joint_training_update_v13 = joint_training_update_v24
validate_accounting_v13 = validate_accounting_v24
_validate_microbatches_v21 = _v23._validate_microbatches_v21
_validate_microbatches_v23 = _v23._validate_microbatches_v23
_validate_microbatches_v13 = _validate_microbatches_v24


__all__ = tuple(
    dict.fromkeys(
        (
            *_v23.__all__,
            "JointTrainingAccountingV24",
            "JointUpdateResultV24",
            "J24_PARAMETER_COUNT_V24",
            "J24_PARAMETER_TENSOR_COUNT_V24",
            "PREDICTOR_CORE_PROTECTED_SURVIVAL_GRADIENT_NORM_CAP_V24",
            "PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_COUNT_V24",
            "PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_TENSOR_COUNT_V24",
            "PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V24",
            "PREREGISTRATION_BYTE_COUNT_V24",
            "PREREGISTRATION_COMMIT_V24",
            "PREREGISTRATION_FILE_SHA256_V24",
            "PROTECTED_PREDICTOR_CORE_PARAMETER_COUNT_V24",
            "PROTECTED_PREDICTOR_CORE_PARAMETER_NAMES_V24",
            "PROTECTED_PREDICTOR_CORE_PARAMETER_TENSOR_COUNT_V24",
            "PROTECTED_PREDICTOR_CORE_TENSOR_COUNT_V24",
            "PredictorCoreProtectedSurvivalObjectiveV24",
            "PredictorCoreProtectedSurvivalParameterSubsetV24",
            "REQUIRED_BATCH_KEYS_V24",
            "SWEPT_PROGRESS_OUTPUT_PARAMETER_NAMES_V24",
            "build_frozen_optimizer_v24",
            "joint_training_update_v24",
            "partition_parameters_v24",
            "predictor_core_protected_survival_objective_v24",
            "predictor_core_protected_survival_parameter_subset_v24",
            "private_training_adapter_receipt_v24",
            "validate_accounting_v24",
            "validate_optimizer_v24",
        )
    )
)
