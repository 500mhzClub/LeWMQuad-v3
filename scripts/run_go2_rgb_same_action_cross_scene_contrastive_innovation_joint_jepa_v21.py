#!/usr/bin/env python3
"""Source-only V21 same-action cross-scene innovation adapter.

V21 privately loads the frozen V20/V19 training adapter.  It replaces the
terminal absolute-successor semantic route with one independently clipped
predictor-only contrastive latent-innovation route.  All inherited Camera,
joint-JEPA, optimizer, batching, clipping, and EMA paths remain unchanged.

This module performs no data discovery, checkpoint access, accelerator
selection, or experiment I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE_TRAINING_PATH = ROOT / (
    "scripts/run_go2_rgb_object_space_height_volume_executed_successor_"
    "semantic_grounding_joint_jepa_v19.py"
)
BASE_PUBLIC_MODULE_NAME = (
    "scripts.run_go2_rgb_object_space_height_volume_executed_successor_"
    "semantic_grounding_joint_jepa_v19"
)
PRIVATE_BASE_MODULE_NAME = f"{__name__}.__private_v20_training"
_PUBLIC_BASE_WAS_LOADED_BEFORE_ADAPTER = BASE_PUBLIC_MODULE_NAME in sys.modules

PREREGISTRATION_COMMIT_V21 = "c2bbce067175dd980c9ed2511dc14db5a222afe4"
PREREGISTRATION_FILE_SHA256_V21 = (
    "f4ff1453e5cb63677dad66253d568c9204bd5504b3b3871e2b0c341402b1850e"
)
PREREGISTRATION_BYTE_COUNT_V21 = 11_594
SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21 = "scene_innovation_negative_row"
SCENE_INNOVATION_ROUTE_NAME_V21 = "scene_innovation_predictor"
SCENE_INNOVATION_GRADIENT_NORM_CAP_V21 = 1.0
SCENE_INNOVATION_PREDICTOR_PARAMETER_TENSOR_COUNT_V21 = 13
SCENE_INNOVATION_PREDICTOR_PARAMETER_COUNT_V21 = 259_008
SCENE_INNOVATION_EXCLUDED_PREFIX_V21 = "predictor.swept_progress_head."
SCENE_INNOVATION_EXCLUDED_PARAMETER_TENSOR_COUNT_V21 = 2
SCENE_INNOVATION_EXCLUDED_PARAMETER_COUNT_V21 = 65
SALIENCE_CELL_COUNT_V21 = 128
INNOVATION_SCALE_FLOOR_V21 = 1.0e-3
ACTION_ONLY_EQUALITY_TOLERANCE_V21 = 1.0e-6
MATERIAL_SCENE_ADVANTAGE_V21 = 1.0e-4


def _load_private_base_training_v21() -> ModuleType:
    if BASE_TRAINING_PATH.is_symlink() or not BASE_TRAINING_PATH.is_file():
        raise FileNotFoundError("reviewed V20 training source is absent or not regular")
    source = BASE_TRAINING_PATH.read_bytes()
    if not source:
        raise RuntimeError("reviewed V20 training source is empty")
    if PRIVATE_BASE_MODULE_NAME in sys.modules:
        raise RuntimeError("private V20 training module name is already occupied")
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


_v20 = _load_private_base_training_v21()
_base = _v20._base
_tensor_core = _v20._tensor_core
if (
    _v20.MICROBATCH_SIZE != 4
    or _v20.MICROBATCHES_PER_UPDATE != 4
    or _v20.PRESENTATIONS_PER_UPDATE != 16
    or _v20.MAXIMUM_UPDATES != 1_000
    or _v20.MAXIMUM_PRESENTATIONS != 16_000
):
    raise RuntimeError("reviewed V20 training cap or batching changed")
if tuple(_v20.ACTION_ORDER).index("hold") != 6:
    raise RuntimeError("reviewed V20 HOLD action index changed")
HOLD_ACTION_INDEX_V21 = 6

for _name in _v20.__all__:
    globals()[_name] = getattr(_v20, _name)

INHERITED_REQUIRED_BATCH_KEYS_V21 = tuple(_v20.REQUIRED_BATCH_KEYS)
REQUIRED_BATCH_KEYS_V21 = (
    *INHERITED_REQUIRED_BATCH_KEYS_V21,
    SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21,
)


@dataclass(frozen=True)
class SceneInnovationPredictorSubsetV21:
    """The exact V20 latent-transition predictor subset."""

    parameters: tuple[Any, ...]
    names: tuple[str, ...]
    predictor_indices: tuple[int, ...]
    parameter_count: int


@dataclass(frozen=True)
class SceneInnovationObjectiveV21:
    """One four-row contrastive innovation objective and auditable tensors."""

    loss: Any
    fit: Any
    rank: Any
    positive_energy: Any
    negative_energy: Any
    advantage: Any
    high_flat_indices: Any
    low_flat_indices: Any
    scale: Any
    valid_cell_count: int


@dataclass(frozen=True)
class JointTrainingAccountingV21:
    """V20 accounting with the replacement scene-innovation route."""

    updates: int = 0
    presentations: int = 0
    microbatch_graphs: int = 0
    backward_calls: int = 0
    camera_route_grad_calls: int = 0
    joint_route_grad_calls: int = 0
    scene_innovation_grad_calls: int = 0
    camera_frame_objectives: int = 0
    optimizer_steps: int = 0
    ema_steps: int = 0
    predictor_forwards: int = 0
    predictor_objectives: int = 0
    scene_innovation_objectives: int = 0


@dataclass(frozen=True)
class JointUpdateResultV21:
    """One V21 update, including all inherited and innovation receipts."""

    accounting: JointTrainingAccountingV21
    mean_losses: Mapping[str, float]
    gradient_routes: Mapping[str, Any]
    gradient_l2: Mapping[str, float]
    ranking_active_microbatches: int
    ranking_eligible_pairs: int
    survival_supervised_decisions: int
    target_gradient_tensor_count: int
    optimizer_steps_this_update: int
    ema_steps_this_update: int
    scene_innovation_diagnostics: Mapping[str, float | int]


def scene_innovation_predictor_subset_v21(
    partition: Any,
) -> SceneInnovationPredictorSubsetV21:
    """Resolve and strictly validate the registered 13-tensor subset."""

    predictor = tuple(partition.predictor)
    names = tuple(partition.names["predictor"])
    if len(predictor) != len(names) or not predictor:
        raise RuntimeError("V21 predictor parameter/name inventory changed")
    if any(not name.startswith("predictor.") for name in names):
        raise RuntimeError("V21 predictor contains a parameter outside predictor.*")
    selected_indices = tuple(
        index
        for index, name in enumerate(names)
        if not name.startswith(SCENE_INNOVATION_EXCLUDED_PREFIX_V21)
    )
    excluded_indices = tuple(
        index
        for index, name in enumerate(names)
        if name.startswith(SCENE_INNOVATION_EXCLUDED_PREFIX_V21)
    )
    selected = tuple(predictor[index] for index in selected_indices)
    selected_names = tuple(names[index] for index in selected_indices)
    excluded = tuple(predictor[index] for index in excluded_indices)
    parameter_count = sum(int(parameter.numel()) for parameter in selected)
    excluded_parameter_count = sum(int(parameter.numel()) for parameter in excluded)
    if (
        len(selected) != SCENE_INNOVATION_PREDICTOR_PARAMETER_TENSOR_COUNT_V21
        or parameter_count != SCENE_INNOVATION_PREDICTOR_PARAMETER_COUNT_V21
        or len(excluded)
        != SCENE_INNOVATION_EXCLUDED_PARAMETER_TENSOR_COUNT_V21
        or excluded_parameter_count
        != SCENE_INNOVATION_EXCLUDED_PARAMETER_COUNT_V21
        or any(not parameter.requires_grad for parameter in selected)
        or len({id(parameter) for parameter in selected}) != len(selected)
    ):
        raise RuntimeError("V21 scene-innovation predictor subset changed")
    return SceneInnovationPredictorSubsetV21(
        parameters=selected,
        names=selected_names,
        predictor_indices=selected_indices,
        parameter_count=parameter_count,
    )


def validate_accounting_v21(accounting: JointTrainingAccountingV21) -> None:
    """Validate the exact 12-call, eight-predictor-objective lifecycle."""

    if not isinstance(accounting, JointTrainingAccountingV21):
        raise TypeError("V21 accounting has the wrong receipt type")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in accounting.__dict__.values()
    ):
        raise ValueError("V21 accounting values must be nonnegative integers")
    updates = accounting.updates
    expected = JointTrainingAccountingV21(
        updates=updates,
        presentations=updates * PRESENTATIONS_PER_UPDATE,
        microbatch_graphs=updates * MICROBATCHES_PER_UPDATE,
        backward_calls=updates * 3 * MICROBATCHES_PER_UPDATE,
        camera_route_grad_calls=updates * MICROBATCHES_PER_UPDATE,
        joint_route_grad_calls=updates * MICROBATCHES_PER_UPDATE,
        scene_innovation_grad_calls=updates * MICROBATCHES_PER_UPDATE,
        camera_frame_objectives=(
            updates
            * MICROBATCHES_PER_UPDATE
            * _tensor_core.CAMERA_FRAMES_PER_MICROBATCH
        ),
        optimizer_steps=updates,
        ema_steps=updates,
        predictor_forwards=updates * MICROBATCHES_PER_UPDATE,
        predictor_objectives=updates * 2 * MICROBATCHES_PER_UPDATE,
        scene_innovation_objectives=updates * MICROBATCHES_PER_UPDATE,
    )
    if accounting != expected:
        raise RuntimeError("V21 joint-training accounting is inconsistent")


def _advance_accounting_v21(
    accounting: JointTrainingAccountingV21,
) -> JointTrainingAccountingV21:
    result = JointTrainingAccountingV21(
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
        scene_innovation_grad_calls=(
            accounting.scene_innovation_grad_calls + MICROBATCHES_PER_UPDATE
        ),
        camera_frame_objectives=(
            accounting.camera_frame_objectives
            + MICROBATCHES_PER_UPDATE * _tensor_core.CAMERA_FRAMES_PER_MICROBATCH
        ),
        optimizer_steps=accounting.optimizer_steps + 1,
        ema_steps=accounting.ema_steps + 1,
        predictor_forwards=accounting.predictor_forwards + MICROBATCHES_PER_UPDATE,
        predictor_objectives=(
            accounting.predictor_objectives + 2 * MICROBATCHES_PER_UPDATE
        ),
        scene_innovation_objectives=(
            accounting.scene_innovation_objectives + MICROBATCHES_PER_UPDATE
        ),
    )
    validate_accounting_v21(result)
    return result


def _validate_update_capacity_v21(accounting: JointTrainingAccountingV21) -> None:
    validate_accounting_v21(accounting)
    if (
        accounting.updates >= MAXIMUM_UPDATES
        or accounting.presentations + PRESENTATIONS_PER_UPDATE
        > MAXIMUM_PRESENTATIONS
    ):
        raise PermissionError("V21 training cap leaves no complete update available")


def _validate_negative_rows_v21(torch: Any, value: Any, reference: Any) -> Any:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != (MICROBATCH_SIZE,)
        or value.dtype != torch.int64
        or value.device != reference.device
    ):
        raise ValueError(
            "V21 scene-innovation negative rows must be int64 B=4 on batch device"
        )
    rows = torch.arange(MICROBATCH_SIZE, dtype=torch.int64, device=value.device)
    if not bool(((value >= 0) & (value < MICROBATCH_SIZE)).all().item()):
        raise ValueError("V21 scene-innovation negative row escaped the microbatch")
    if bool((value == rows).any().item()):
        raise ValueError("V21 scene-innovation negative row retained a self match")
    return value.detach()


def _validate_microbatches_v21(
    torch: Any, microbatches: Sequence[Mapping[str, Any]]
) -> None:
    if len(microbatches) != MICROBATCHES_PER_UPDATE:
        raise ValueError("V21 update must contain exactly four microbatches")
    inherited: list[Mapping[str, Any]] = []
    for index, batch in enumerate(microbatches):
        if tuple(batch) != REQUIRED_BATCH_KEYS_V21:
            raise ValueError(f"V21 microbatch {index} key order or membership changed")
        projected = {name: batch[name] for name in INHERITED_REQUIRED_BATCH_KEYS_V21}
        inherited.append(projected)
        _validate_negative_rows_v21(
            torch,
            batch[SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21],
            batch[CURRENT_RGB_KEY],
        )
    _base._validate_microbatches_v13(torch, tuple(inherited))


def _validate_latents_v21(torch: Any, *values: Any) -> None:
    for name, value in values:
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != (MICROBATCH_SIZE, 64, 64, 64)
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all().item())
        ):
            raise ValueError(f"V21 {name} must be finite float32 (4,64,64,64)")


def scene_innovation_objective_v21(
    torch: Any,
    predicted: Any,
    current_latent: Any,
    ema_current: Any,
    ema_next: Any,
    executed_actions: Any,
    negative_rows: Any,
    cell_valid_mask: Any,
) -> SceneInnovationObjectiveV21:
    """Compute the exact valid-cell balanced V21 innovation objective."""

    if (
        not isinstance(predicted, torch.Tensor)
        or tuple(predicted.shape)
        != (MICROBATCH_SIZE, len(ACTION_ORDER), 64, 64, 64)
        or predicted.dtype != torch.float32
        or not bool(torch.isfinite(predicted).all().item())
    ):
        raise ValueError("V21 all-action prediction must be finite float32 (4,9,64,64,64)")
    _validate_latents_v21(
        torch,
        ("current latent", current_latent),
        ("EMA current latent", ema_current),
        ("EMA next latent", ema_next),
    )
    if any(value.device != predicted.device for value in (current_latent, ema_current, ema_next)):
        raise ValueError("V21 innovation tensors must share one device")
    if (
        not isinstance(executed_actions, torch.Tensor)
        or tuple(executed_actions.shape) != (MICROBATCH_SIZE,)
        or executed_actions.dtype == torch.bool
        or executed_actions.is_floating_point()
        or executed_actions.device != predicted.device
    ):
        raise ValueError("V21 executed actions must be integer B=4 on prediction device")
    actions = executed_actions.long()
    if not bool(((actions >= 0) & (actions < len(ACTION_ORDER))).all().item()):
        raise ValueError("V21 executed action escaped the frozen vocabulary")
    negatives = _validate_negative_rows_v21(torch, negative_rows, predicted)
    if (
        not isinstance(cell_valid_mask, torch.Tensor)
        or tuple(cell_valid_mask.shape) != (64, 64)
        or cell_valid_mask.dtype != torch.bool
        or cell_valid_mask.device != predicted.device
        or cell_valid_mask.requires_grad
    ):
        raise ValueError("V21 cell-valid mask must be detached bool (64,64) on device")
    valid_flat = torch.nonzero(
        cell_valid_mask.detach().flatten(), as_tuple=False
    ).flatten()
    valid_count = int(valid_flat.numel())
    if valid_count < 2 * SALIENCE_CELL_COUNT_V21:
        raise ValueError("V21 cell-valid mask contains fewer than 256 cells")

    rows = torch.arange(MICROBATCH_SIZE, device=predicted.device)
    positive = predicted[rows, actions] - current_latent.detach()
    negative = predicted[negatives, actions] - current_latent.detach()[negatives]
    target = (ema_next.detach() - ema_current.detach()).detach()
    positive_valid = positive.flatten(start_dim=2)[:, :, valid_flat]
    negative_valid = negative.flatten(start_dim=2)[:, :, valid_flat]
    target_valid = target.flatten(start_dim=2)[:, :, valid_flat]
    scale = torch.sqrt(target_valid.square().mean(dim=(1, 2))).clamp_min(
        INNOVATION_SCALE_FLOOR_V21
    ).detach()
    divisor = scale[:, None, None]
    normalized_target = target_valid / divisor
    positive_error = torch.nn.functional.smooth_l1_loss(
        positive_valid / divisor,
        normalized_target,
        beta=1.0,
        reduction="none",
    ).mean(dim=1)
    negative_error = torch.nn.functional.smooth_l1_loss(
        negative_valid / divisor,
        normalized_target,
        beta=1.0,
        reduction="none",
    ).mean(dim=1)
    salience = normalized_target.square().mean(dim=1).detach()
    order = torch.argsort(salience, dim=1, descending=False, stable=True)
    low_positions = order[:, :SALIENCE_CELL_COUNT_V21]
    high_positions = order[:, -SALIENCE_CELL_COUNT_V21:]
    low_flat = valid_flat[low_positions]
    high_flat = valid_flat[high_positions]
    if bool((low_flat[:, :, None] == high_flat[:, None, :]).any().item()):
        raise RuntimeError("V21 high/low salience sets overlap")
    positive_energy = 0.5 * (
        positive_error.gather(1, high_positions).mean(dim=1)
        + positive_error.gather(1, low_positions).mean(dim=1)
    )
    negative_energy = 0.5 * (
        negative_error.gather(1, high_positions).mean(dim=1)
        + negative_error.gather(1, low_positions).mean(dim=1)
    )
    fit = positive_energy.mean()
    rank = torch.nn.functional.softplus(positive_energy - negative_energy).mean()
    rank = rank / math.log(2.0)
    loss = fit + rank
    for name, value in (
        ("positive energy", positive_energy),
        ("negative energy", negative_energy),
        ("fit", fit),
        ("rank", rank),
        ("loss", loss),
    ):
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"V21 scene-innovation {name} is nonfinite")
    return SceneInnovationObjectiveV21(
        loss=loss,
        fit=fit,
        rank=rank,
        positive_energy=positive_energy,
        negative_energy=negative_energy,
        advantage=negative_energy - positive_energy,
        high_flat_indices=high_flat.detach(),
        low_flat_indices=low_flat.detach(),
        scale=scale,
        valid_cell_count=valid_count,
    )


def _matching_gradient_cosine_v21(
    torch: Any, inherited: Sequence[Any], innovation: Sequence[Any]
) -> float:
    if len(inherited) != len(innovation) or not inherited:
        raise RuntimeError("V21 cosine gradient inventories changed")
    dot = inherited[0].new_zeros((), dtype=torch.float32)
    inherited_square = dot.clone()
    innovation_square = dot.clone()
    for inherited_gradient, innovation_gradient in zip(
        inherited, innovation, strict=True
    ):
        left = inherited_gradient.float()
        right = innovation_gradient.float()
        dot = dot + (left * right).sum(dtype=torch.float32)
        inherited_square = inherited_square + (left * left).sum(dtype=torch.float32)
        innovation_square = innovation_square + (right * right).sum(dtype=torch.float32)
    inherited_norm = torch.sqrt(inherited_square)
    innovation_norm = torch.sqrt(innovation_square)
    if not bool(torch.isfinite(inherited_norm)) or not bool(torch.isfinite(innovation_norm)):
        raise FloatingPointError("V21 matching predictor gradient norm is nonfinite")
    if not (_tensor_core._scalar(inherited_norm) > 0.0) or not (
        _tensor_core._scalar(innovation_norm) > 0.0
    ):
        raise RuntimeError("V21 matching predictor gradient is zero")
    return _tensor_core._scalar(
        torch.clamp(dot / (inherited_norm * innovation_norm), -1.0, 1.0)
    )


def joint_training_update_v21(
    model: Any,
    optimizer: Any,
    microbatches: Sequence[Mapping[str, Any]],
    *,
    accounting: JointTrainingAccountingV21 | None = None,
) -> JointUpdateResultV21:
    """Run one frozen V20 update with the replacement V21 route."""

    torch, semantic_api, survival_api, *_ = _tensor_core._runtime_apis()
    state = JointTrainingAccountingV21() if accounting is None else accounting
    _validate_update_capacity_v21(state)
    _validate_microbatches_v21(torch, microbatches)
    partition = _base.partition_parameters_v18(model)
    _base.validate_optimizer_v18(optimizer, partition)
    innovation_subset = scene_innovation_predictor_subset_v21(partition)
    ema_before = int(model.ema_update_count.item())
    if ema_before != state.ema_steps:
        raise RuntimeError("model EMA count disagrees with V21 accounting")
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("V21 EMA target already has a gradient")
    cell_valid_mask = model.bev_lift.cell_valid_mask

    optimizer.zero_grad(set_to_none=True)
    camera_shared = _tensor_core._zero_accumulators(partition.shared)
    joint_shared = _tensor_core._zero_accumulators(partition.shared)
    joint_representation = _tensor_core._zero_accumulators(partition.representation)
    joint_predictor = _tensor_core._zero_accumulators(partition.predictor)
    innovation_predictor = _tensor_core._zero_accumulators(
        innovation_subset.parameters
    )
    absent = {
        name: 0
        for name in (
            "camera_shared",
            "joint_shared",
            "representation",
            "predictor",
            SCENE_INNOVATION_ROUTE_NAME_V21,
        )
    }
    sums = {
        name: 0.0
        for name in (
            "S",
            "P",
            "U",
            "R",
            "O",
            "I_fit",
            "I_rank",
            "I_scene",
            "N",
            "C",
            "L",
            "positive_energy",
            "negative_energy",
            "advantage",
        )
    }
    active_ranking = eligible_pairs = supervised_decisions = 0
    advantage_count = 0
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
        predicted, survival_logits = _tensor_core._v3._v2._v1._prediction_parts(
            prediction
        )
        with torch.no_grad():
            ema_current = model.encode_target(batch[CURRENT_RGB_KEY])
            ema_next = model.encode_target(batch[NEXT_RGB_KEY])
        innovation = scene_innovation_objective_v21(
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
            raise RuntimeError("V21 valid-cell count changed within one update")
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
            ("scene innovation I", innovation.loss),
            ("joint N", navigation),
            ("Camera C", camera.total),
        ):
            _tensor_core._finite_tensor(torch, value, name)
        if (
            not camera.total.requires_grad
            or not navigation.requires_grad
            or not innovation.loss.requires_grad
        ):
            raise RuntimeError("V21 C, N, and I_scene must retain gradient graphs")

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
        absent[SCENE_INNOVATION_ROUTE_NAME_V21] += _tensor_core._accumulate_gradients(
            innovation_predictor, i_gradients
        )

        values = {
            "S": joint.semantic,
            "P": joint.executed_action_ema_latent,
            "U": joint.survival,
            "R": joint.progress_ranking,
            "O": occupied.loss,
            "I_fit": innovation.fit,
            "I_rank": innovation.rank,
            "I_scene": innovation.loss,
            "N": navigation,
            "C": camera.total,
            "L": navigation + camera.total + innovation.loss,
            "positive_energy": innovation.positive_energy.mean(),
            "negative_energy": innovation.negative_energy.mean(),
            "advantage": innovation.advantage.sum(),
        }
        for name, value in values.items():
            sums[name] += _tensor_core._scalar(value)
        advantage_count += int(innovation.advantage.numel())
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
        SCENE_INNOVATION_ROUTE_NAME_V21: (
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
        SCENE_INNOVATION_ROUTE_NAME_V21,
    ):
        if not (_tensor_core._scalar(route_values[name][0]) > 0.0):
            raise RuntimeError(f"required V21 gradient route {name!r} is zero")
    if absent[SCENE_INNOVATION_ROUTE_NAME_V21] != 0:
        raise RuntimeError("V21 scene-innovation route has an absent gradient")

    matching_inherited = tuple(
        joint_predictor[index] for index in innovation_subset.predictor_indices
    )
    gradient_cosine = _matching_gradient_cosine_v21(
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
    innovation_scale = route_values[SCENE_INNOVATION_ROUTE_NAME_V21][1]
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
        raise RuntimeError("V21 EMA target received a gradient")
    optimizer.step()
    for parameter in partition.online:
        _tensor_core._finite_tensor(torch, parameter, "V21 online parameter")
    model.update_target_ema_after_optimizer_step()
    ema_after = int(model.ema_update_count.item())
    if ema_after != ema_before + 1:
        raise RuntimeError("V21 EMA did not update exactly once")
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("V21 EMA target received a gradient")

    advanced = _advance_accounting_v21(state)
    if advanced.ema_steps != ema_after:
        raise RuntimeError("post-update V21 EMA count disagrees with accounting")
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
            "S",
            "P",
            "U",
            "R",
            "O",
            "I_fit",
            "I_rank",
            "I_scene",
            "N",
            "C",
            "L",
        )
    }
    if advantage_count != PRESENTATIONS_PER_UPDATE:
        raise RuntimeError("V21 scene-innovation advantage count changed")
    if observed_valid_count is None:
        raise RuntimeError("V21 update produced no valid-cell receipt")
    return JointUpdateResultV21(
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
        scene_innovation_diagnostics={
            "positive_energy_mean": sums["positive_energy"] / MICROBATCHES_PER_UPDATE,
            "negative_energy_mean": sums["negative_energy"] / MICROBATCHES_PER_UPDATE,
            "advantage_sum": sums["advantage"],
            "advantage_count": advantage_count,
            "advantage_mean": sums["advantage"] / advantage_count,
            "matching_predictor_gradient_cosine": gradient_cosine,
            "valid_cell_count": observed_valid_count,
            "high_salience_cell_count": SALIENCE_CELL_COUNT_V21,
            "low_salience_cell_count": SALIENCE_CELL_COUNT_V21,
        },
    )


def private_training_adapter_receipt_v21() -> dict[str, Any]:
    return {
        "schema": (
            "lewm_go2_rgb_same_action_cross_scene_contrastive_innovation_"
            "joint_jepa_v21_training_adapter_v1"
        ),
        "base_training": str(BASE_TRAINING_PATH.relative_to(ROOT)),
        "public_base_was_loaded_before_adapter": _PUBLIC_BASE_WAS_LOADED_BEFORE_ADAPTER,
        "public_base_loaded_by_adapter": False,
        "private_module_registered": PRIVATE_BASE_MODULE_NAME in sys.modules,
        "preregistration_commit": PREREGISTRATION_COMMIT_V21,
        "preregistration_file_sha256": PREREGISTRATION_FILE_SHA256_V21,
        "preregistration_byte_count": PREREGISTRATION_BYTE_COUNT_V21,
        "negative_row_key": SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21,
        "scene_innovation_gradient_norm_cap": SCENE_INNOVATION_GRADIENT_NORM_CAP_V21,
        "scene_innovation_predictor_parameter_tensor_count": (
            SCENE_INNOVATION_PREDICTOR_PARAMETER_TENSOR_COUNT_V21
        ),
        "scene_innovation_predictor_parameter_count": (
            SCENE_INNOVATION_PREDICTOR_PARAMETER_COUNT_V21
        ),
        "salience_cell_count_per_role": SALIENCE_CELL_COUNT_V21,
        "innovation_scale_floor": INNOVATION_SCALE_FLOOR_V21,
        "maximum_updates": MAXIMUM_UPDATES,
        "maximum_presentations": MAXIMUM_PRESENTATIONS,
    }


partition_parameters_v21 = _base.partition_parameters_v18
build_frozen_optimizer_v21 = _base.build_frozen_optimizer_v18
validate_optimizer_v21 = _base.validate_optimizer_v18
partition_parameters_v19 = partition_parameters_v21
build_frozen_optimizer_v19 = build_frozen_optimizer_v21
validate_optimizer_v19 = validate_optimizer_v21
joint_training_update_v19 = joint_training_update_v21
validate_accounting_v19 = validate_accounting_v21
partition_parameters_v18 = partition_parameters_v21
build_frozen_optimizer_v18 = build_frozen_optimizer_v21
validate_optimizer_v18 = validate_optimizer_v21
joint_training_update_v18 = joint_training_update_v21
validate_accounting_v18 = validate_accounting_v21
JointTrainingAccountingV13 = JointTrainingAccountingV21
JointTrainingAccountingV19 = JointTrainingAccountingV21
JointUpdateResultV13 = JointUpdateResultV21
JointUpdateResultV19 = JointUpdateResultV21
partition_parameters_v13 = partition_parameters_v21
build_frozen_optimizer_v13 = build_frozen_optimizer_v21
validate_optimizer_v13 = validate_optimizer_v21
joint_training_update_v13 = joint_training_update_v21
validate_accounting_v13 = validate_accounting_v21
_validate_microbatches_v13 = _validate_microbatches_v21


__all__ = tuple(
    dict.fromkeys(
        (
            *_v20.__all__,
            "ACTION_ONLY_EQUALITY_TOLERANCE_V21",
            "INHERITED_REQUIRED_BATCH_KEYS_V21",
            "INNOVATION_SCALE_FLOOR_V21",
            "JointTrainingAccountingV21",
            "JointUpdateResultV21",
            "MATERIAL_SCENE_ADVANTAGE_V21",
            "REQUIRED_BATCH_KEYS_V21",
            "SALIENCE_CELL_COUNT_V21",
            "SCENE_INNOVATION_GRADIENT_NORM_CAP_V21",
            "SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21",
            "SCENE_INNOVATION_PREDICTOR_PARAMETER_COUNT_V21",
            "SCENE_INNOVATION_PREDICTOR_PARAMETER_TENSOR_COUNT_V21",
            "SCENE_INNOVATION_ROUTE_NAME_V21",
            "SceneInnovationObjectiveV21",
            "SceneInnovationPredictorSubsetV21",
            "build_frozen_optimizer_v21",
            "joint_training_update_v21",
            "partition_parameters_v21",
            "private_training_adapter_receipt_v21",
            "scene_innovation_objective_v21",
            "scene_innovation_predictor_subset_v21",
            "validate_accounting_v21",
            "validate_optimizer_v21",
        )
    )
)
