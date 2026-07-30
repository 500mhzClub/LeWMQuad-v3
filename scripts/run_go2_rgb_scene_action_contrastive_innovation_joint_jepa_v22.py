#!/usr/bin/env python3
"""Source-only V22 scene-action contrastive-innovation training adapter.

V22 privately loads frozen V21 and changes only its predictor-only auxiliary:
the requested prediction must beat both a different-scene same-action negative
and the mean independently evaluated same-scene non-requested-action energy.
The model, inherited joint JEPA, batching, optimizer, route isolation, caps,
and EMA lifecycle are unchanged.  This module performs no experiment I/O.
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

PREREGISTRATION_COMMIT_V22 = "43053ae49c28082c616f45ed857eedb727380952"
PREREGISTRATION_FILE_SHA256_V22 = (
    "7ee36433d739663654de593cf018500cc5547e249173f08201ad4ac5c6b1959e"
)
PREREGISTRATION_BYTE_COUNT_V22 = 11_986
TWO_AXIS_INNOVATION_ROUTE_NAME_V22 = "two_axis_innovation_predictor"
TWO_AXIS_INNOVATION_GRADIENT_NORM_CAP_V22 = 1.0
TWO_AXIS_INNOVATION_PREDICTOR_PARAMETER_TENSOR_COUNT_V22 = 13
TWO_AXIS_INNOVATION_PREDICTOR_PARAMETER_COUNT_V22 = 259_008
NONREQUESTED_ACTION_COUNT_V22 = 8
ACTION_COUNT_V22 = 9
RANK_AXIS_WEIGHT_V22 = 0.5


def _load_private_base_training_v22() -> ModuleType:
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


_v21 = _load_private_base_training_v22()
_base = _v21._base
_tensor_core = _v21._tensor_core
if (
    _v21.MICROBATCH_SIZE != 4
    or len(_v21.ACTION_ORDER) != ACTION_COUNT_V22
    or _v21.MICROBATCHES_PER_UPDATE != 4
    or _v21.PRESENTATIONS_PER_UPDATE != 16
    or _v21.MAXIMUM_UPDATES != 1_000
    or _v21.MAXIMUM_PRESENTATIONS != 16_000
):
    raise RuntimeError("frozen V21 action, batching, or cap identity changed")

for _name in _v21.__all__:
    globals()[_name] = getattr(_v21, _name)

REQUIRED_BATCH_KEYS_V22 = tuple(_v21.REQUIRED_BATCH_KEYS_V21)


@dataclass(frozen=True)
class TwoAxisInnovationPredictorSubsetV22:
    parameters: tuple[Any, ...]
    names: tuple[str, ...]
    predictor_indices: tuple[int, ...]
    parameter_count: int


@dataclass(frozen=True)
class TwoAxisInnovationObjectiveV22:
    loss: Any
    fit: Any
    scene_rank: Any
    action_rank: Any
    positive_energy: Any
    scene_negative_energy: Any
    action_negative_energy: Any
    scene_advantage: Any
    action_advantage: Any
    nonrequested_actions: Any
    action_candidate_energy: Any
    high_flat_indices: Any
    low_flat_indices: Any
    scale: Any
    valid_cell_count: int


@dataclass(frozen=True)
class JointTrainingAccountingV22:
    updates: int = 0
    presentations: int = 0
    microbatch_graphs: int = 0
    backward_calls: int = 0
    camera_route_grad_calls: int = 0
    joint_route_grad_calls: int = 0
    two_axis_innovation_grad_calls: int = 0
    camera_frame_objectives: int = 0
    optimizer_steps: int = 0
    ema_steps: int = 0
    predictor_forwards: int = 0
    predictor_objectives: int = 0
    two_axis_innovation_objectives: int = 0


@dataclass(frozen=True)
class JointUpdateResultV22:
    accounting: JointTrainingAccountingV22
    mean_losses: Mapping[str, float]
    gradient_routes: Mapping[str, Any]
    gradient_l2: Mapping[str, float]
    ranking_active_microbatches: int
    ranking_eligible_pairs: int
    survival_supervised_decisions: int
    target_gradient_tensor_count: int
    optimizer_steps_this_update: int
    ema_steps_this_update: int
    two_axis_innovation_diagnostics: Mapping[str, float | int]


def two_axis_innovation_predictor_subset_v22(
    partition: Any,
) -> TwoAxisInnovationPredictorSubsetV22:
    inherited = _v21.scene_innovation_predictor_subset_v21(partition)
    if (
        len(inherited.parameters)
        != TWO_AXIS_INNOVATION_PREDICTOR_PARAMETER_TENSOR_COUNT_V22
        or inherited.parameter_count
        != TWO_AXIS_INNOVATION_PREDICTOR_PARAMETER_COUNT_V22
    ):
        raise RuntimeError("V22 two-axis predictor subset changed")
    return TwoAxisInnovationPredictorSubsetV22(
        parameters=tuple(inherited.parameters),
        names=tuple(inherited.names),
        predictor_indices=tuple(inherited.predictor_indices),
        parameter_count=inherited.parameter_count,
    )


def validate_accounting_v22(accounting: JointTrainingAccountingV22) -> None:
    if not isinstance(accounting, JointTrainingAccountingV22):
        raise TypeError("V22 accounting has the wrong receipt type")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in accounting.__dict__.values()
    ):
        raise ValueError("V22 accounting values must be nonnegative integers")
    updates = accounting.updates
    expected = JointTrainingAccountingV22(
        updates=updates,
        presentations=updates * PRESENTATIONS_PER_UPDATE,
        microbatch_graphs=updates * MICROBATCHES_PER_UPDATE,
        backward_calls=updates * 3 * MICROBATCHES_PER_UPDATE,
        camera_route_grad_calls=updates * MICROBATCHES_PER_UPDATE,
        joint_route_grad_calls=updates * MICROBATCHES_PER_UPDATE,
        two_axis_innovation_grad_calls=updates * MICROBATCHES_PER_UPDATE,
        camera_frame_objectives=updates * 2 * MICROBATCH_SIZE * MICROBATCHES_PER_UPDATE,
        optimizer_steps=updates,
        ema_steps=updates,
        predictor_forwards=updates * MICROBATCHES_PER_UPDATE,
        predictor_objectives=updates * 2 * MICROBATCHES_PER_UPDATE,
        two_axis_innovation_objectives=updates * MICROBATCHES_PER_UPDATE,
    )
    if accounting != expected:
        raise RuntimeError("V22 accounting is inconsistent")


def _advance_accounting_v22(
    accounting: JointTrainingAccountingV22,
) -> JointTrainingAccountingV22:
    return JointTrainingAccountingV22(
        updates=accounting.updates + 1,
        presentations=accounting.presentations + PRESENTATIONS_PER_UPDATE,
        microbatch_graphs=accounting.microbatch_graphs + MICROBATCHES_PER_UPDATE,
        backward_calls=accounting.backward_calls + 3 * MICROBATCHES_PER_UPDATE,
        camera_route_grad_calls=accounting.camera_route_grad_calls + MICROBATCHES_PER_UPDATE,
        joint_route_grad_calls=accounting.joint_route_grad_calls + MICROBATCHES_PER_UPDATE,
        two_axis_innovation_grad_calls=(
            accounting.two_axis_innovation_grad_calls + MICROBATCHES_PER_UPDATE
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
        two_axis_innovation_objectives=(
            accounting.two_axis_innovation_objectives + MICROBATCHES_PER_UPDATE
        ),
    )


def _validate_update_capacity_v22(accounting: JointTrainingAccountingV22) -> None:
    validate_accounting_v22(accounting)
    if (
        accounting.updates >= MAXIMUM_UPDATES
        or accounting.presentations + PRESENTATIONS_PER_UPDATE > MAXIMUM_PRESENTATIONS
    ):
        raise PermissionError("V22 training cap leaves no complete update available")


def _validate_microbatches_v22(
    torch: Any, microbatches: Sequence[Mapping[str, Any]]
) -> None:
    if len(microbatches) != MICROBATCHES_PER_UPDATE:
        raise ValueError("V22 update must contain exactly four microbatches")
    if any(tuple(batch) != REQUIRED_BATCH_KEYS_V22 for batch in microbatches):
        raise ValueError("V22 microbatch schema changed from frozen V21")
    _v21._validate_microbatches_v21(torch, microbatches)


def two_axis_innovation_objective_v22(
    torch: Any,
    predicted: Any,
    current_latent: Any,
    ema_current: Any,
    ema_next: Any,
    executed_actions: Any,
    negative_rows: Any,
    cell_valid_mask: Any,
) -> TwoAxisInnovationObjectiveV22:
    """Compute the exact V22 scene-plus-action conditional innovation loss."""

    if (
        not isinstance(predicted, torch.Tensor)
        or tuple(predicted.shape)
        != (MICROBATCH_SIZE, ACTION_COUNT_V22, 64, 64, 64)
        or predicted.dtype != torch.float32
        or not bool(torch.isfinite(predicted).all().item())
    ):
        raise ValueError("V22 all-action prediction must be finite float32 (4,9,64,64,64)")
    _v21._validate_latents_v21(
        torch,
        ("current latent", current_latent),
        ("EMA current latent", ema_current),
        ("EMA next latent", ema_next),
    )
    if any(value.device != predicted.device for value in (current_latent, ema_current, ema_next)):
        raise ValueError("V22 innovation tensors must share one device")
    if (
        not isinstance(executed_actions, torch.Tensor)
        or tuple(executed_actions.shape) != (MICROBATCH_SIZE,)
        or executed_actions.dtype == torch.bool
        or executed_actions.is_floating_point()
        or executed_actions.device != predicted.device
    ):
        raise ValueError("V22 executed actions must be integer B=4 on prediction device")
    actions = executed_actions.long().detach()
    if not bool(((actions >= 0) & (actions < ACTION_COUNT_V22)).all().item()):
        raise ValueError("V22 executed action escaped the frozen vocabulary")
    negatives = _v21._validate_negative_rows_v21(torch, negative_rows, predicted)
    if (
        not isinstance(cell_valid_mask, torch.Tensor)
        or tuple(cell_valid_mask.shape) != (64, 64)
        or cell_valid_mask.dtype != torch.bool
        or cell_valid_mask.device != predicted.device
        or cell_valid_mask.requires_grad
    ):
        raise ValueError("V22 cell-valid mask must be detached bool (64,64) on device")
    valid_flat = torch.nonzero(
        cell_valid_mask.detach().flatten(), as_tuple=False
    ).flatten()
    valid_count = int(valid_flat.numel())
    if valid_count < 2 * SALIENCE_CELL_COUNT_V21:
        raise ValueError("V22 cell-valid mask contains fewer than 256 cells")

    rows = torch.arange(MICROBATCH_SIZE, device=predicted.device)
    positive = predicted[rows, actions] - current_latent.detach()
    scene_negative = predicted[negatives, actions] - current_latent.detach()[negatives]
    target = (ema_next.detach() - ema_current.detach()).detach()
    positive_valid = positive.flatten(start_dim=2)[:, :, valid_flat]
    scene_negative_valid = scene_negative.flatten(start_dim=2)[:, :, valid_flat]
    target_valid = target.flatten(start_dim=2)[:, :, valid_flat]
    scale = torch.sqrt(target_valid.square().mean(dim=(1, 2))).clamp_min(
        INNOVATION_SCALE_FLOOR_V21
    ).detach()
    divisor = scale[:, None, None]
    normalized_target = target_valid / divisor

    def _cell_error(residual_valid: Any) -> Any:
        return torch.nn.functional.smooth_l1_loss(
            residual_valid / divisor,
            normalized_target,
            beta=1.0,
            reduction="none",
        ).mean(dim=1)

    positive_error = _cell_error(positive_valid)
    scene_negative_error = _cell_error(scene_negative_valid)
    salience = normalized_target.square().mean(dim=1).detach()
    order = torch.argsort(salience, dim=1, descending=False, stable=True)
    low_positions = order[:, :SALIENCE_CELL_COUNT_V21]
    high_positions = order[:, -SALIENCE_CELL_COUNT_V21:]
    low_flat = valid_flat[low_positions]
    high_flat = valid_flat[high_positions]
    if bool((low_flat[:, :, None] == high_flat[:, None, :]).any().item()):
        raise RuntimeError("V22 high/low salience sets overlap")

    def _row_energy(error: Any) -> Any:
        return 0.5 * (
            error.gather(1, high_positions).mean(dim=1)
            + error.gather(1, low_positions).mean(dim=1)
        )

    positive_energy = _row_energy(positive_error)
    scene_negative_energy = _row_energy(scene_negative_error)

    action_ids = torch.arange(ACTION_COUNT_V22, device=predicted.device)
    action_grid = action_ids[None, :].expand(MICROBATCH_SIZE, -1)
    nonrequested_actions = action_grid[action_grid != actions[:, None]].reshape(
        MICROBATCH_SIZE, NONREQUESTED_ACTION_COUNT_V22
    ).detach()
    if bool((nonrequested_actions == actions[:, None]).any().item()):
        raise RuntimeError("V22 requested action entered its negative set")
    action_negative = (
        predicted[rows[:, None], nonrequested_actions]
        - current_latent.detach()[:, None]
    )
    action_negative_valid = action_negative.flatten(start_dim=3)[
        :, :, :, valid_flat
    ]
    action_error = torch.nn.functional.smooth_l1_loss(
        action_negative_valid / scale[:, None, None, None],
        normalized_target[:, None].expand(
            -1, NONREQUESTED_ACTION_COUNT_V22, -1, -1
        ),
        beta=1.0,
        reduction="none",
    ).mean(dim=2)
    expanded_high = high_positions[:, None].expand(-1, NONREQUESTED_ACTION_COUNT_V22, -1)
    expanded_low = low_positions[:, None].expand(-1, NONREQUESTED_ACTION_COUNT_V22, -1)
    action_candidate_energy = 0.5 * (
        action_error.gather(2, expanded_high).mean(dim=2)
        + action_error.gather(2, expanded_low).mean(dim=2)
    )
    action_negative_energy = action_candidate_energy.mean(dim=1)

    fit = positive_energy.mean()
    scene_rank = torch.nn.functional.softplus(
        positive_energy - scene_negative_energy
    ).mean() / math.log(2.0)
    action_rank = torch.nn.functional.softplus(
        positive_energy - action_negative_energy
    ).mean() / math.log(2.0)
    loss = fit + RANK_AXIS_WEIGHT_V22 * (scene_rank + action_rank)
    for name, value in (
        ("positive energy", positive_energy),
        ("scene-negative energy", scene_negative_energy),
        ("action-negative energy", action_negative_energy),
        ("action candidate energy", action_candidate_energy),
        ("fit", fit),
        ("scene rank", scene_rank),
        ("action rank", action_rank),
        ("loss", loss),
    ):
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"V22 two-axis innovation {name} is nonfinite")
    return TwoAxisInnovationObjectiveV22(
        loss=loss,
        fit=fit,
        scene_rank=scene_rank,
        action_rank=action_rank,
        positive_energy=positive_energy,
        scene_negative_energy=scene_negative_energy,
        action_negative_energy=action_negative_energy,
        scene_advantage=scene_negative_energy - positive_energy,
        action_advantage=action_negative_energy - positive_energy,
        nonrequested_actions=nonrequested_actions,
        action_candidate_energy=action_candidate_energy,
        high_flat_indices=high_flat.detach(),
        low_flat_indices=low_flat.detach(),
        scale=scale,
        valid_cell_count=valid_count,
    )


def joint_training_update_v22(
    model: Any,
    optimizer: Any,
    microbatches: Sequence[Mapping[str, Any]],
    *,
    accounting: JointTrainingAccountingV22 | None = None,
) -> JointUpdateResultV22:
    """Run one frozen V21 update with only the V22 auxiliary replaced."""

    torch, semantic_api, survival_api, *_ = _tensor_core._runtime_apis()
    state = JointTrainingAccountingV22() if accounting is None else accounting
    _validate_update_capacity_v22(state)
    _validate_microbatches_v22(torch, microbatches)
    partition = _base.partition_parameters_v18(model)
    _base.validate_optimizer_v18(optimizer, partition)
    innovation_subset = two_axis_innovation_predictor_subset_v22(partition)
    ema_before = int(model.ema_update_count.item())
    if ema_before != state.ema_steps:
        raise RuntimeError("model EMA count disagrees with V22 accounting")
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("V22 EMA target already has a gradient")
    cell_valid_mask = model.bev_lift.cell_valid_mask

    optimizer.zero_grad(set_to_none=True)
    camera_shared = _tensor_core._zero_accumulators(partition.shared)
    joint_shared = _tensor_core._zero_accumulators(partition.shared)
    joint_representation = _tensor_core._zero_accumulators(partition.representation)
    joint_predictor = _tensor_core._zero_accumulators(partition.predictor)
    innovation_predictor = _tensor_core._zero_accumulators(innovation_subset.parameters)
    absent = {
        name: 0
        for name in (
            "camera_shared",
            "joint_shared",
            "representation",
            "predictor",
            TWO_AXIS_INNOVATION_ROUTE_NAME_V22,
        )
    }
    sums = {
        name: 0.0
        for name in (
            "S", "P", "U", "R", "O", "I_fit", "I_scene_rank",
            "I_action_rank", "I_two_axis", "N", "C", "L",
            "positive_energy", "scene_negative_energy", "action_negative_energy",
            "scene_advantage", "action_advantage",
        )
    }
    active_ranking = eligible_pairs = supervised_decisions = 0
    scene_advantage_count = action_advantage_count = 0
    observed_valid_count: int | None = None

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
        predicted, survival_logits = _tensor_core._v3._v2._v1._prediction_parts(prediction)
        with torch.no_grad():
            ema_current = model.encode_target(batch[CURRENT_RGB_KEY])
            ema_next = model.encode_target(batch[NEXT_RGB_KEY])
        innovation = two_axis_innovation_objective_v22(
            torch,
            predicted,
            current_latent,
            ema_current,
            ema_next,
            batch[EXECUTED_ACTION_KEY],
            batch[SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21],
            cell_valid_mask,
        )
        if observed_valid_count is None:
            observed_valid_count = innovation.valid_cell_count
        elif observed_valid_count != innovation.valid_cell_count:
            raise RuntimeError("V22 valid-cell count changed within one update")
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
            ("two-axis innovation I", innovation.loss),
            ("joint N", navigation),
            ("Camera C", camera.total),
        ):
            _tensor_core._finite_tensor(torch, value, name)
        if (
            not camera.total.requires_grad
            or not navigation.requires_grad
            or not innovation.loss.requires_grad
        ):
            raise RuntimeError("V22 C, N, and I_two_axis must retain gradient graphs")

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
        i_gradients = torch.autograd.grad(
            innovation.loss / MICROBATCHES_PER_UPDATE,
            innovation_subset.parameters,
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
        absent[TWO_AXIS_INNOVATION_ROUTE_NAME_V22] += _tensor_core._accumulate_gradients(
            innovation_predictor, i_gradients
        )

        values = {
            "S": joint.semantic,
            "P": joint.executed_action_ema_latent,
            "U": joint.survival,
            "R": joint.progress_ranking,
            "O": occupied.loss,
            "I_fit": innovation.fit,
            "I_scene_rank": innovation.scene_rank,
            "I_action_rank": innovation.action_rank,
            "I_two_axis": innovation.loss,
            "N": navigation,
            "C": camera.total,
            "L": navigation + camera.total + innovation.loss,
            "positive_energy": innovation.positive_energy.mean(),
            "scene_negative_energy": innovation.scene_negative_energy.mean(),
            "action_negative_energy": innovation.action_negative_energy.mean(),
            "scene_advantage": innovation.scene_advantage.sum(),
            "action_advantage": innovation.action_advantage.sum(),
        }
        for name, value in values.items():
            sums[name] += _tensor_core._scalar(value)
        scene_advantage_count += int(innovation.scene_advantage.numel())
        action_advantage_count += int(innovation.action_advantage.numel())
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
        TWO_AXIS_INNOVATION_ROUTE_NAME_V22: (
            innovation_subset.parameters,
            innovation_predictor,
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
        TWO_AXIS_INNOVATION_ROUTE_NAME_V22,
    ):
        if not (_tensor_core._scalar(route_values[name][0]) > 0.0):
            raise RuntimeError(f"required V22 gradient route {name!r} is zero")
    if absent[TWO_AXIS_INNOVATION_ROUTE_NAME_V22] != 0:
        raise RuntimeError("V22 two-axis innovation route has an absent gradient")

    matching_inherited = tuple(
        joint_predictor[index] for index in innovation_subset.predictor_indices
    )
    gradient_cosine = _v21._matching_gradient_cosine_v21(
        torch, matching_inherited, innovation_predictor
    )
    c_scale = route_values["camera_shared"][1]
    n_scale = route_values["joint_shared"][1]
    for parameter, c_gradient, n_gradient in zip(
        partition.shared, camera_shared, joint_shared, strict=True
    ):
        parameter.grad = c_scale * c_gradient + n_scale * n_gradient
    representation_scale = route_values["representation"][1]
    for parameter, gradient in zip(
        partition.representation, joint_representation, strict=True
    ):
        parameter.grad = representation_scale * gradient
    predictor_scale = route_values["predictor"][1]
    innovation_scale = route_values[TWO_AXIS_INNOVATION_ROUTE_NAME_V22][1]
    innovation_by_predictor_index = {
        index: gradient
        for index, gradient in zip(
            innovation_subset.predictor_indices, innovation_predictor, strict=True
        )
    }
    for index, (parameter, inherited_gradient) in enumerate(
        zip(partition.predictor, joint_predictor, strict=True)
    ):
        gradient = predictor_scale * inherited_gradient
        if index in innovation_by_predictor_index:
            gradient = gradient + innovation_scale * innovation_by_predictor_index[index]
        parameter.grad = gradient

    target_gradient_count = sum(
        parameter.grad is not None for parameter in partition.target
    )
    if target_gradient_count:
        raise RuntimeError("V22 EMA target received a gradient")
    optimizer.step()
    for parameter in partition.online:
        _tensor_core._finite_tensor(torch, parameter, "V22 online parameter")
    model.update_target_ema_after_optimizer_step()
    ema_after = int(model.ema_update_count.item())
    if ema_after != ema_before + 1:
        raise RuntimeError("V22 EMA did not update exactly once")
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("V22 EMA target received a gradient")

    advanced = _advance_accounting_v22(state)
    if advanced.ema_steps != ema_after:
        raise RuntimeError("post-update V22 EMA count disagrees with accounting")
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
        for name in (
            "S", "P", "U", "R", "O", "I_fit", "I_scene_rank",
            "I_action_rank", "I_two_axis", "N", "C", "L",
        )
    }
    if (
        scene_advantage_count != PRESENTATIONS_PER_UPDATE
        or action_advantage_count != PRESENTATIONS_PER_UPDATE
    ):
        raise RuntimeError("V22 two-axis advantage count changed")
    if observed_valid_count is None:
        raise RuntimeError("V22 update produced no valid-cell receipt")
    return JointUpdateResultV22(
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
        two_axis_innovation_diagnostics={
            "positive_energy_mean": sums["positive_energy"] / MICROBATCHES_PER_UPDATE,
            "scene_negative_energy_mean": (
                sums["scene_negative_energy"] / MICROBATCHES_PER_UPDATE
            ),
            "scene_advantage_sum": sums["scene_advantage"],
            "scene_advantage_count": scene_advantage_count,
            "scene_advantage_mean": sums["scene_advantage"] / scene_advantage_count,
            "action_negative_energy_mean": (
                sums["action_negative_energy"] / MICROBATCHES_PER_UPDATE
            ),
            "action_advantage_sum": sums["action_advantage"],
            "action_advantage_count": action_advantage_count,
            "action_advantage_mean": sums["action_advantage"] / action_advantage_count,
            "nonrequested_action_count_per_row": NONREQUESTED_ACTION_COUNT_V22,
            "action_candidate_energy_count": (
                PRESENTATIONS_PER_UPDATE * NONREQUESTED_ACTION_COUNT_V22
            ),
            "matching_predictor_gradient_cosine": gradient_cosine,
            "valid_cell_count": observed_valid_count,
            "high_salience_cell_count": SALIENCE_CELL_COUNT_V21,
            "low_salience_cell_count": SALIENCE_CELL_COUNT_V21,
        },
    )


def private_training_adapter_receipt_v22() -> dict[str, Any]:
    return {
        "schema": (
            "lewm_go2_rgb_scene_action_contrastive_innovation_joint_jepa_"
            "v22_training_adapter_v1"
        ),
        "base_training": str(BASE_TRAINING_PATH.relative_to(ROOT)),
        "base_frozen_source_and_review_commit": BASE_FROZEN_SOURCE_AND_REVIEW_COMMIT,
        "base_training_file_sha256": BASE_TRAINING_FILE_SHA256,
        "base_training_byte_count": BASE_TRAINING_BYTE_COUNT,
        "public_base_was_loaded_before_adapter": _PUBLIC_BASE_WAS_LOADED_BEFORE_ADAPTER,
        "public_base_loaded_by_adapter": False,
        "private_module_registered": PRIVATE_BASE_MODULE_NAME in sys.modules,
        "preregistration_commit": PREREGISTRATION_COMMIT_V22,
        "preregistration_file_sha256": PREREGISTRATION_FILE_SHA256_V22,
        "preregistration_byte_count": PREREGISTRATION_BYTE_COUNT_V22,
        "scene_negative_row_key": SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21,
        "two_axis_innovation_gradient_norm_cap": (
            TWO_AXIS_INNOVATION_GRADIENT_NORM_CAP_V22
        ),
        "two_axis_predictor_parameter_tensor_count": (
            TWO_AXIS_INNOVATION_PREDICTOR_PARAMETER_TENSOR_COUNT_V22
        ),
        "two_axis_predictor_parameter_count": (
            TWO_AXIS_INNOVATION_PREDICTOR_PARAMETER_COUNT_V22
        ),
        "nonrequested_action_count": NONREQUESTED_ACTION_COUNT_V22,
        "rank_axis_weight": RANK_AXIS_WEIGHT_V22,
        "maximum_updates": MAXIMUM_UPDATES,
        "maximum_presentations": MAXIMUM_PRESENTATIONS,
    }


partition_parameters_v22 = _base.partition_parameters_v18
build_frozen_optimizer_v22 = _base.build_frozen_optimizer_v18
validate_optimizer_v22 = _base.validate_optimizer_v18
joint_training_update_v21 = joint_training_update_v22
validate_accounting_v21 = validate_accounting_v22
partition_parameters_v21 = partition_parameters_v22
build_frozen_optimizer_v21 = build_frozen_optimizer_v22
validate_optimizer_v21 = validate_optimizer_v22
joint_training_update_v19 = joint_training_update_v22
validate_accounting_v19 = validate_accounting_v22
joint_training_update_v18 = joint_training_update_v22
validate_accounting_v18 = validate_accounting_v22
JointTrainingAccountingV13 = JointTrainingAccountingV22
JointTrainingAccountingV19 = JointTrainingAccountingV22
JointTrainingAccountingV21 = JointTrainingAccountingV22
JointUpdateResultV13 = JointUpdateResultV22
JointUpdateResultV19 = JointUpdateResultV22
JointUpdateResultV21 = JointUpdateResultV22
partition_parameters_v13 = partition_parameters_v22
build_frozen_optimizer_v13 = build_frozen_optimizer_v22
validate_optimizer_v13 = validate_optimizer_v22
joint_training_update_v13 = joint_training_update_v22
validate_accounting_v13 = validate_accounting_v22
_validate_microbatches_v21 = _validate_microbatches_v22
_validate_microbatches_v13 = _validate_microbatches_v22


__all__ = tuple(
    dict.fromkeys(
        (
            *_v21.__all__,
            "ACTION_COUNT_V22",
            "JointTrainingAccountingV22",
            "JointUpdateResultV22",
            "NONREQUESTED_ACTION_COUNT_V22",
            "PREREGISTRATION_BYTE_COUNT_V22",
            "PREREGISTRATION_COMMIT_V22",
            "PREREGISTRATION_FILE_SHA256_V22",
            "RANK_AXIS_WEIGHT_V22",
            "REQUIRED_BATCH_KEYS_V22",
            "TWO_AXIS_INNOVATION_GRADIENT_NORM_CAP_V22",
            "TWO_AXIS_INNOVATION_PREDICTOR_PARAMETER_COUNT_V22",
            "TWO_AXIS_INNOVATION_PREDICTOR_PARAMETER_TENSOR_COUNT_V22",
            "TWO_AXIS_INNOVATION_ROUTE_NAME_V22",
            "TwoAxisInnovationObjectiveV22",
            "TwoAxisInnovationPredictorSubsetV22",
            "build_frozen_optimizer_v22",
            "joint_training_update_v22",
            "partition_parameters_v22",
            "private_training_adapter_receipt_v22",
            "two_axis_innovation_objective_v22",
            "two_axis_innovation_predictor_subset_v22",
            "validate_accounting_v22",
            "validate_optimizer_v22",
        )
    )
)
