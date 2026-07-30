#!/usr/bin/env python3
"""Source-only V23 direct survival-output contrast training adapter.

V23 privately loads the frozen V21 tensor core used by V22, removes V22's
latent proxy loss, and adds one output-level objective over the already
computed all-action survival logits.  The objective makes the current-scene
output beat both an in-batch wrong-scene output and the frozen train-action
mean prior.  The inherited joint JEPA remains active and jointly trains the
online RGB encoder, representation, and predictor.  This module performs no
experiment I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE_TRAINING_PATH = ROOT / (
    "scripts/run_go2_rgb_same_action_cross_scene_contrastive_innovation_"
    "joint_jepa_v21.py"
)
BASE_FROZEN_SOURCE_AND_REVIEW_COMMIT = (
    "7071a006dda3851280fbdf030e156862c4f19ab3"
)
BASE_TRAINING_FILE_SHA256 = (
    "6229e0bf863f29e0f407286a76ba0f2fa42089bef498e50491b55125e5020d7c"
)
BASE_TRAINING_BYTE_COUNT = 37_115
BASE_PUBLIC_MODULE_NAME = (
    "scripts.run_go2_rgb_same_action_cross_scene_contrastive_innovation_"
    "joint_jepa_v21"
)
PRIVATE_BASE_MODULE_NAME = f"{__name__}.__private_v21_training"
_PUBLIC_BASE_WAS_LOADED_BEFORE_ADAPTER = BASE_PUBLIC_MODULE_NAME in sys.modules

PREREGISTRATION_COMMIT_V23 = "a7cf9692dd93212a82cb598d3175ff1c3598941b"
PREREGISTRATION_FILE_SHA256_V23 = (
    "d5702759866138db1467778553ef8494d05f4593fcca14822050b1e0991180ae"
)
PREREGISTRATION_BYTE_COUNT_V23 = 14_294
ACTION_PRIOR_M_KEY_V23 = "train_action_prior_m"
STATE_RESIDUAL_SURVIVAL_ROUTE_NAME_V23 = "state_residual_survival_output"
STATE_RESIDUAL_SURVIVAL_GRADIENT_NORM_CAP_V23 = 1.0
STATE_RESIDUAL_SURVIVAL_PARAMETER_TENSOR_COUNT_V23 = 109
STATE_RESIDUAL_SURVIVAL_PARAMETER_COUNT_V23 = 3_365_417
ACTION_COUNT_V23 = 9
NON_HOLD_ACTION_INDICES_V23 = (0, 1, 2, 3, 4, 5, 7, 8)
NON_HOLD_ACTION_COUNT_V23 = len(NON_HOLD_ACTION_INDICES_V23)
PROGRESS_HORIZON_M_V23 = 1.5
PROGRESS_SEGMENT_M_V23 = 0.1


def _load_private_base_training_v23() -> ModuleType:
    if BASE_TRAINING_PATH.is_symlink() or not BASE_TRAINING_PATH.is_file():
        raise FileNotFoundError("frozen V21 training source is absent or not regular")
    source = BASE_TRAINING_PATH.read_bytes()
    if (
        len(source) != BASE_TRAINING_BYTE_COUNT
        or hashlib.sha256(source).hexdigest() != BASE_TRAINING_FILE_SHA256
    ):
        raise RuntimeError("frozen V21 training source binding changed")
    if PRIVATE_BASE_MODULE_NAME in sys.modules:
        raise RuntimeError("private V21 training module name is already occupied")
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


_v21 = _load_private_base_training_v23()
_base = _v21._base
_tensor_core = _v21._tensor_core
if (
    _v21.MICROBATCH_SIZE != 4
    or len(_v21.ACTION_ORDER) != ACTION_COUNT_V23
    or _v21.MICROBATCHES_PER_UPDATE != 4
    or _v21.PRESENTATIONS_PER_UPDATE != 16
    or _v21.MAXIMUM_UPDATES != 1_000
    or _v21.MAXIMUM_PRESENTATIONS != 16_000
):
    raise RuntimeError("frozen V21 action, batching, or cap identity changed")

for _name in _v21.__all__:
    globals()[_name] = getattr(_v21, _name)

REQUIRED_BATCH_KEYS_V23 = (
    *_v21.REQUIRED_BATCH_KEYS_V21,
    ACTION_PRIOR_M_KEY_V23,
)


@dataclass(frozen=True)
class JointTrainingAccountingV23:
    updates: int = 0
    presentations: int = 0
    microbatch_graphs: int = 0
    backward_calls: int = 0
    camera_route_grad_calls: int = 0
    joint_route_grad_calls: int = 0
    state_residual_survival_grad_calls: int = 0
    camera_frame_objectives: int = 0
    optimizer_steps: int = 0
    ema_steps: int = 0
    predictor_forwards: int = 0
    predictor_objectives: int = 0
    state_residual_survival_objectives: int = 0


@dataclass(frozen=True)
class JointUpdateResultV23:
    accounting: JointTrainingAccountingV23
    mean_losses: Mapping[str, float]
    gradient_routes: Mapping[str, Any]
    gradient_l2: Mapping[str, float]
    ranking_active_microbatches: int
    ranking_eligible_pairs: int
    survival_supervised_decisions: int
    target_gradient_tensor_count: int
    optimizer_steps_this_update: int
    ema_steps_this_update: int
    state_residual_survival_diagnostics: Mapping[str, float | int]


def validate_accounting_v23(accounting: JointTrainingAccountingV23) -> None:
    if not isinstance(accounting, JointTrainingAccountingV23):
        raise TypeError("V23 accounting has the wrong receipt type")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in accounting.__dict__.values()
    ):
        raise ValueError("V23 accounting values must be nonnegative integers")
    updates = accounting.updates
    expected = JointTrainingAccountingV23(
        updates=updates,
        presentations=updates * PRESENTATIONS_PER_UPDATE,
        microbatch_graphs=updates * MICROBATCHES_PER_UPDATE,
        backward_calls=updates * 3 * MICROBATCHES_PER_UPDATE,
        camera_route_grad_calls=updates * MICROBATCHES_PER_UPDATE,
        joint_route_grad_calls=updates * MICROBATCHES_PER_UPDATE,
        state_residual_survival_grad_calls=updates * MICROBATCHES_PER_UPDATE,
        camera_frame_objectives=updates * 2 * MICROBATCH_SIZE * MICROBATCHES_PER_UPDATE,
        optimizer_steps=updates,
        ema_steps=updates,
        predictor_forwards=updates * MICROBATCHES_PER_UPDATE,
        predictor_objectives=updates * 2 * MICROBATCHES_PER_UPDATE,
        state_residual_survival_objectives=updates * MICROBATCHES_PER_UPDATE,
    )
    if accounting != expected:
        raise RuntimeError("V23 accounting is inconsistent")


def _advance_accounting_v23(
    accounting: JointTrainingAccountingV23,
) -> JointTrainingAccountingV23:
    return JointTrainingAccountingV23(
        updates=accounting.updates + 1,
        presentations=accounting.presentations + PRESENTATIONS_PER_UPDATE,
        microbatch_graphs=accounting.microbatch_graphs + MICROBATCHES_PER_UPDATE,
        backward_calls=accounting.backward_calls + 3 * MICROBATCHES_PER_UPDATE,
        camera_route_grad_calls=accounting.camera_route_grad_calls + MICROBATCHES_PER_UPDATE,
        joint_route_grad_calls=accounting.joint_route_grad_calls + MICROBATCHES_PER_UPDATE,
        state_residual_survival_grad_calls=(
            accounting.state_residual_survival_grad_calls + MICROBATCHES_PER_UPDATE
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
        state_residual_survival_objectives=(
            accounting.state_residual_survival_objectives + MICROBATCHES_PER_UPDATE
        ),
    )


def _validate_update_capacity_v23(accounting: JointTrainingAccountingV23) -> None:
    validate_accounting_v23(accounting)
    if (
        accounting.updates >= MAXIMUM_UPDATES
        or accounting.presentations + PRESENTATIONS_PER_UPDATE > MAXIMUM_PRESENTATIONS
    ):
        raise PermissionError("V23 training cap leaves no complete update available")


def _validate_microbatches_v23(
    torch: Any, microbatches: Sequence[Mapping[str, Any]]
) -> None:
    if len(microbatches) != MICROBATCHES_PER_UPDATE:
        raise ValueError("V23 update must contain exactly four microbatches")
    if any(tuple(batch) != REQUIRED_BATCH_KEYS_V23 for batch in microbatches):
        raise ValueError("V23 microbatch schema changed from frozen V21 plus prior")
    inherited = tuple(
        {name: batch[name] for name in _v21.REQUIRED_BATCH_KEYS_V21}
        for batch in microbatches
    )
    _v21._validate_microbatches_v21(torch, inherited)
    for batch in microbatches:
        prior = batch[ACTION_PRIOR_M_KEY_V23]
        anchor = batch[CURRENT_RGB_KEY]
        if (
            not isinstance(prior, torch.Tensor)
            or tuple(prior.shape) != (ACTION_COUNT_V23,)
            or prior.dtype != torch.float32
            or prior.device != anchor.device
            or prior.requires_grad
            or not bool(torch.isfinite(prior).all().item())
            or not bool(((prior >= 0.0) & (prior <= PROGRESS_HORIZON_M_V23)).all().item())
        ):
            raise ValueError(
                "V23 train-action prior must be detached finite float32 (9,) on device"
            )



@dataclass(frozen=True)
class StateResidualSurvivalParameterSubsetV23:
    """Exact online path reached by the V23 output objective."""

    parameters: tuple[Any, ...]
    names: tuple[str, ...]
    parameter_count: int


@dataclass(frozen=True)
class StateResidualSurvivalObjectiveV23:
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


def state_residual_survival_parameter_subset_v23(
    partition: Any,
) -> StateResidualSurvivalParameterSubsetV23:
    """Select every online survival-output parameter and exclude semantics/EMA."""

    selected: list[tuple[str, Any]] = []
    selected.extend(zip(partition.names["encoder"], partition.encoder, strict=True))
    selected.extend(
        zip(partition.names["evidence_head"], partition.evidence_head, strict=True)
    )
    selected.extend(
        (name, parameter)
        for name, parameter in zip(
            partition.names["representation"], partition.representation, strict=True
        )
        if not name.startswith("semantic_head.")
    )
    selected.extend(
        zip(partition.names["predictor"], partition.predictor, strict=True)
    )
    names = tuple(name for name, _ in selected)
    parameters = tuple(parameter for _, parameter in selected)
    if (
        len(parameters) != STATE_RESIDUAL_SURVIVAL_PARAMETER_TENSOR_COUNT_V23
        or sum(int(parameter.numel()) for parameter in parameters)
        != STATE_RESIDUAL_SURVIVAL_PARAMETER_COUNT_V23
        or any(name.startswith("semantic_head.") for name in names)
        or any(name.startswith(("target_encoder.", "target_bev_lift.")) for name in names)
        or "predictor.swept_progress_head.output.weight" not in names
        or "predictor.swept_progress_head.output.bias" not in names
    ):
        raise RuntimeError("V23 survival-output parameter subset changed")
    return StateResidualSurvivalParameterSubsetV23(
        parameters=parameters,
        names=names,
        parameter_count=sum(int(parameter.numel()) for parameter in parameters),
    )


def state_residual_survival_objective_v23(
    torch: Any,
    survival_api: Any,
    survival_logits: Any,
    prefix_lengths: Any,
    negative_rows: Any,
    action_prior_m: Any,
) -> StateResidualSurvivalObjectiveV23:
    """Compute the preregistered direct wrong-scene/prior output contrast."""

    if (
        not isinstance(survival_logits, torch.Tensor)
        or tuple(survival_logits.shape) != (MICROBATCH_SIZE, ACTION_COUNT_V23, 16)
        or survival_logits.dtype != torch.float32
        or not bool(torch.isfinite(survival_logits).all().item())
    ):
        raise ValueError("V23 survival logits must be finite float32 (4,9,16)")
    if (
        not isinstance(prefix_lengths, torch.Tensor)
        or tuple(prefix_lengths.shape) != (MICROBATCH_SIZE, ACTION_COUNT_V23)
        or prefix_lengths.dtype == torch.bool
        or prefix_lengths.is_floating_point()
        or prefix_lengths.device != survival_logits.device
        or prefix_lengths.requires_grad
        or bool(((prefix_lengths < 0) | (prefix_lengths > 15)).any().item())
    ):
        raise ValueError("V23 prefix labels must be detached integer (4,9) in [0,15]")
    negatives = _v21._validate_negative_rows_v21(
        torch, negative_rows, survival_logits
    )
    if (
        not isinstance(action_prior_m, torch.Tensor)
        or tuple(action_prior_m.shape) != (ACTION_COUNT_V23,)
        or action_prior_m.dtype != torch.float32
        or action_prior_m.device != survival_logits.device
        or action_prior_m.requires_grad
        or not bool(torch.isfinite(action_prior_m).all().item())
        or not bool(
            ((action_prior_m >= 0.0) & (action_prior_m <= PROGRESS_HORIZON_M_V23))
            .all()
            .item()
        )
    ):
        raise ValueError("V23 action prior must be detached finite float32 (9,)")

    scores = survival_api.survival_scores_v1(survival_logits)
    expected_progress = scores.expected_progress_m
    if (
        tuple(expected_progress.shape) != (MICROBATCH_SIZE, ACTION_COUNT_V23)
        or expected_progress.dtype != torch.float32
        or expected_progress.device != survival_logits.device
        or not expected_progress.requires_grad
    ):
        raise RuntimeError("V23 expected-progress graph changed")
    action_indices = torch.tensor(
        NON_HOLD_ACTION_INDICES_V23,
        dtype=torch.int64,
        device=survival_logits.device,
    )
    target_m = prefix_lengths.detach().to(dtype=torch.float32) * PROGRESS_SEGMENT_M_V23
    positive_m = expected_progress.index_select(1, action_indices)
    target_nonhold = target_m.index_select(1, action_indices)
    scene_m = expected_progress.index_select(0, negatives).index_select(
        1, action_indices
    )
    scene_target = target_m.index_select(0, negatives).index_select(1, action_indices)
    prior_m = action_prior_m.detach().index_select(0, action_indices)[None, :].expand(
        MICROBATCH_SIZE, -1
    )
    divisor = PROGRESS_HORIZON_M_V23
    positive_energy = torch.nn.functional.smooth_l1_loss(
        positive_m / divisor,
        target_nonhold / divisor,
        beta=1.0,
        reduction="none",
    )
    scene_negative_energy = torch.nn.functional.smooth_l1_loss(
        scene_m / divisor,
        target_nonhold / divisor,
        beta=1.0,
        reduction="none",
    )
    prior_negative_energy = torch.nn.functional.smooth_l1_loss(
        prior_m / divisor,
        target_nonhold / divisor,
        beta=1.0,
        reduction="none",
    ).detach()
    scene_eligible = (target_nonhold != scene_target).detach()
    prior_eligible = (prior_negative_energy > 0.0).detach()
    scene_count = int(scene_eligible.sum().item())
    prior_count = int(prior_eligible.sum().item())
    if scene_count < 1 or prior_count < 1:
        raise ValueError("V23 microbatch must contain both scene and prior comparisons")
    scene_rank_terms = torch.nn.functional.softplus(
        positive_energy[scene_eligible] - scene_negative_energy[scene_eligible]
    ) / math.log(2.0)
    prior_rank_terms = torch.nn.functional.softplus(
        positive_energy[prior_eligible] - prior_negative_energy[prior_eligible]
    ) / math.log(2.0)
    rank = torch.cat((scene_rank_terms, prior_rank_terms)).mean()
    fit = positive_energy.mean()
    loss = fit + rank
    scene_advantage = (
        scene_negative_energy[scene_eligible] - positive_energy[scene_eligible]
    )
    prior_advantage = (
        prior_negative_energy[prior_eligible] - positive_energy[prior_eligible]
    )
    for name, value in (
        ("positive energy", positive_energy),
        ("scene energy", scene_negative_energy),
        ("prior energy", prior_negative_energy),
        ("fit", fit),
        ("rank", rank),
        ("loss", loss),
        ("scene advantage", scene_advantage),
        ("prior advantage", prior_advantage),
    ):
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"V23 state-residual survival {name} is nonfinite")
    return StateResidualSurvivalObjectiveV23(
        loss=loss,
        fit=fit,
        rank=rank,
        positive_energy=positive_energy,
        scene_negative_energy=scene_negative_energy,
        prior_negative_energy=prior_negative_energy,
        scene_eligible=scene_eligible,
        prior_eligible=prior_eligible,
        scene_rank_sum=scene_rank_terms.sum(),
        prior_rank_sum=prior_rank_terms.sum(),
        scene_advantage_sum=scene_advantage.sum(),
        prior_advantage_sum=prior_advantage.sum(),
        scene_eligible_count=scene_count,
        prior_eligible_count=prior_count,
    )



def joint_training_update_v23(
    model: Any,
    optimizer: Any,
    microbatches: Sequence[Mapping[str, Any]],
    *,
    accounting: JointTrainingAccountingV23 | None = None,
) -> JointUpdateResultV23:
    """Run one update with inherited joint JEPA plus direct output contrast."""

    torch, semantic_api, survival_api, *_ = _tensor_core._runtime_apis()
    state = JointTrainingAccountingV23() if accounting is None else accounting
    _validate_update_capacity_v23(state)
    _validate_microbatches_v23(torch, microbatches)
    partition = _base.partition_parameters_v18(model)
    _base.validate_optimizer_v18(optimizer, partition)
    auxiliary_subset = state_residual_survival_parameter_subset_v23(partition)
    ema_before = int(model.ema_update_count.item())
    if ema_before != state.ema_steps:
        raise RuntimeError("model EMA count disagrees with V23 accounting")
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("V23 EMA target already has a gradient")

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
            STATE_RESIDUAL_SURVIVAL_ROUTE_NAME_V23,
        )
    }
    sums = {
        name: 0.0
        for name in (
            "S", "P", "U", "R", "O", "F", "J_rank", "J23", "N", "C", "L",
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
        auxiliary = state_residual_survival_objective_v23(
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
            ("state-residual survival J23", auxiliary.loss),
            ("joint N", navigation),
            ("Camera C", camera.total),
        ):
            _tensor_core._finite_tensor(torch, value, name)
        if (
            not camera.total.requires_grad
            or not navigation.requires_grad
            or not auxiliary.loss.requires_grad
        ):
            raise RuntimeError("V23 C, N, and J23 must retain gradient graphs")

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
        absent[STATE_RESIDUAL_SURVIVAL_ROUTE_NAME_V23] += (
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
            "J23": auxiliary.loss,
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
        STATE_RESIDUAL_SURVIVAL_ROUTE_NAME_V23: (
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
        STATE_RESIDUAL_SURVIVAL_ROUTE_NAME_V23,
    ):
        if not (_tensor_core._scalar(route_values[name][0]) > 0.0):
            raise RuntimeError(f"required V23 gradient route {name!r} is zero")
    if absent[STATE_RESIDUAL_SURVIVAL_ROUTE_NAME_V23] != 0:
        raise RuntimeError("V23 output auxiliary has an absent gradient")

    c_scale = route_values["camera_shared"][1]
    n_scale = route_values["joint_shared"][1]
    representation_scale = route_values["representation"][1]
    predictor_scale = route_values["predictor"][1]
    auxiliary_scale = route_values[STATE_RESIDUAL_SURVIVAL_ROUTE_NAME_V23][1]
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
    for parameter, inherited_gradient in zip(
        partition.predictor, joint_predictor, strict=True
    ):
        parameter.grad = (
            predictor_scale * inherited_gradient
            + auxiliary_scale * auxiliary_by_id[id(parameter)]
        )

    target_gradient_count = sum(
        parameter.grad is not None for parameter in partition.target
    )
    if target_gradient_count:
        raise RuntimeError("V23 EMA target received a gradient")
    optimizer.step()
    for parameter in partition.online:
        _tensor_core._finite_tensor(torch, parameter, "V23 online parameter")
    model.update_target_ema_after_optimizer_step()
    ema_after = int(model.ema_update_count.item())
    if ema_after != ema_before + 1:
        raise RuntimeError("V23 EMA did not update exactly once")
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("V23 EMA target received a gradient")

    advanced = _advance_accounting_v23(state)
    if advanced.ema_steps != ema_after:
        raise RuntimeError("post-update V23 EMA count disagrees with accounting")
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
        for name in ("S", "P", "U", "R", "O", "F", "J_rank", "J23", "N", "C", "L")
    }
    if positive_count != PRESENTATIONS_PER_UPDATE * NON_HOLD_ACTION_COUNT_V23:
        raise RuntimeError("V23 positive-energy accounting changed")
    if scene_count < MICROBATCHES_PER_UPDATE or prior_count < MICROBATCHES_PER_UPDATE:
        raise RuntimeError("V23 comparison accounting changed")
    return JointUpdateResultV23(
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
        state_residual_survival_diagnostics={
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


def private_training_adapter_receipt_v23() -> dict[str, Any]:
    return {
        "schema": (
            "lewm_go2_rgb_action_prior_residualized_wrong_scene_survival_output_joint_jepa_"
            "v23_training_adapter_v1"
        ),
        "base_training": str(BASE_TRAINING_PATH.relative_to(ROOT)),
        "base_frozen_source_and_review_commit": BASE_FROZEN_SOURCE_AND_REVIEW_COMMIT,
        "base_training_file_sha256": BASE_TRAINING_FILE_SHA256,
        "base_training_byte_count": BASE_TRAINING_BYTE_COUNT,
        "public_base_was_loaded_before_adapter": _PUBLIC_BASE_WAS_LOADED_BEFORE_ADAPTER,
        "public_base_loaded_by_adapter": False,
        "private_module_registered": PRIVATE_BASE_MODULE_NAME in sys.modules,
        "preregistration_commit": PREREGISTRATION_COMMIT_V23,
        "preregistration_file_sha256": PREREGISTRATION_FILE_SHA256_V23,
        "preregistration_byte_count": PREREGISTRATION_BYTE_COUNT_V23,
        "scene_negative_row_key": SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21,
        "action_prior_batch_key": ACTION_PRIOR_M_KEY_V23,
        "state_residual_survival_gradient_norm_cap": (
            STATE_RESIDUAL_SURVIVAL_GRADIENT_NORM_CAP_V23
        ),
        "state_residual_survival_parameter_tensor_count": (
            STATE_RESIDUAL_SURVIVAL_PARAMETER_TENSOR_COUNT_V23
        ),
        "state_residual_survival_parameter_count": (
            STATE_RESIDUAL_SURVIVAL_PARAMETER_COUNT_V23
        ),
        "non_hold_action_indices": NON_HOLD_ACTION_INDICES_V23,
        "progress_horizon_m": PROGRESS_HORIZON_M_V23,
        "maximum_updates": MAXIMUM_UPDATES,
        "maximum_presentations": MAXIMUM_PRESENTATIONS,
    }


partition_parameters_v23 = _base.partition_parameters_v18
build_frozen_optimizer_v23 = _base.build_frozen_optimizer_v18
validate_optimizer_v23 = _base.validate_optimizer_v18
joint_training_update_v21 = joint_training_update_v23
validate_accounting_v21 = validate_accounting_v23
partition_parameters_v21 = partition_parameters_v23
build_frozen_optimizer_v21 = build_frozen_optimizer_v23
validate_optimizer_v21 = validate_optimizer_v23
joint_training_update_v19 = joint_training_update_v23
validate_accounting_v19 = validate_accounting_v23
joint_training_update_v18 = joint_training_update_v23
validate_accounting_v18 = validate_accounting_v23
JointTrainingAccountingV13 = JointTrainingAccountingV23
JointTrainingAccountingV19 = JointTrainingAccountingV23
JointTrainingAccountingV21 = JointTrainingAccountingV23
JointUpdateResultV13 = JointUpdateResultV23
JointUpdateResultV19 = JointUpdateResultV23
JointUpdateResultV21 = JointUpdateResultV23
partition_parameters_v13 = partition_parameters_v23
build_frozen_optimizer_v13 = build_frozen_optimizer_v23
validate_optimizer_v13 = validate_optimizer_v23
joint_training_update_v13 = joint_training_update_v23
validate_accounting_v13 = validate_accounting_v23
# The inherited V21 executor validates a V21-key projection before the V23
# training update sees the full batch.  Preserve its original validator for
# that compatibility boundary; the active update calls the V23 validator
# directly above.
_validate_microbatches_v21 = _v21._validate_microbatches_v21
_validate_microbatches_v13 = _validate_microbatches_v23


__all__ = tuple(
    dict.fromkeys(
        (
            *_v21.__all__,
            "ACTION_PRIOR_M_KEY_V23",
            "ACTION_COUNT_V23",
            "JointTrainingAccountingV23",
            "JointUpdateResultV23",
            "NON_HOLD_ACTION_COUNT_V23",
            "NON_HOLD_ACTION_INDICES_V23",
            "PREREGISTRATION_BYTE_COUNT_V23",
            "PREREGISTRATION_COMMIT_V23",
            "PREREGISTRATION_FILE_SHA256_V23",
            "PROGRESS_HORIZON_M_V23",
            "PROGRESS_SEGMENT_M_V23",
            "REQUIRED_BATCH_KEYS_V23",
            "STATE_RESIDUAL_SURVIVAL_GRADIENT_NORM_CAP_V23",
            "STATE_RESIDUAL_SURVIVAL_PARAMETER_COUNT_V23",
            "STATE_RESIDUAL_SURVIVAL_PARAMETER_TENSOR_COUNT_V23",
            "STATE_RESIDUAL_SURVIVAL_ROUTE_NAME_V23",
            "StateResidualSurvivalObjectiveV23",
            "StateResidualSurvivalParameterSubsetV23",
            "build_frozen_optimizer_v23",
            "joint_training_update_v23",
            "partition_parameters_v23",
            "private_training_adapter_receipt_v23",
            "state_residual_survival_objective_v23",
            "state_residual_survival_parameter_subset_v23",
            "validate_accounting_v23",
            "validate_optimizer_v23",
        )
    )
)
