#!/usr/bin/env python3
"""Source-only V25 per-row persistence-contrastive temporal adapter.

V25 privately loads the frozen V24 training source, delegates J24 and its
parameter subset without modification, and mechanically replaces only the
inherited global persistence ratio with the preregistered row-local energy-gap
softplus.  This module performs no experiment I/O.
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
    "scripts/run_go2_rgb_predictor_core_protected_survival_output_joint_jepa_"
    "v24.py"
)
BASE_FROZEN_SOURCE_AND_REVIEW_COMMIT = (
    "2b6178a4d876dc17c45fb340a4ab03ee302649b0"
)
BASE_TRAINING_FILE_SHA256 = (
    "0a149aadfc8f4f0860c4bdfd9fe330e96ab95cbbe556e5b2c16ef1e390e819c6"
)
BASE_TRAINING_BYTE_COUNT = 34_726
BASE_PUBLIC_MODULE_NAME = (
    "scripts.run_go2_rgb_predictor_core_protected_survival_output_joint_jepa_"
    "v24"
)
PRIVATE_BASE_MODULE_NAME = f"{__name__}.__private_v24_training"
_PUBLIC_BASE_WAS_LOADED_BEFORE_ADAPTER = BASE_PUBLIC_MODULE_NAME in sys.modules

PREREGISTRATION_COMMIT_V25 = "f00e20df3b429f9242516ac38f67fea587e04b22"
PREREGISTRATION_FILE_SHA256_V25 = (
    "b9ce16b251415c50cb643daad919699c32965e23ddcd77d22bb3b69334f8b299"
)
PREREGISTRATION_BYTE_COUNT_V25 = 18_965
SOFTPLUS_BETA_V25 = 1.0
SOFTPLUS_THRESHOLD_V25 = 20.0
LOG2_NORMALIZER_V25 = math.log(2.0)
LEGACY_PERSISTENCE_BASELINE_MIN_V25 = 1.0e-6
PER_ROW_PERSISTENCE_CONTRASTIVE_MECHANISM_V25 = (
    "per_row_persistence_contrastive_temporal_v1"
)


def _load_private_base_training_v25() -> ModuleType:
    if BASE_TRAINING_PATH.is_symlink() or not BASE_TRAINING_PATH.is_file():
        raise FileNotFoundError("frozen V24 training source is absent or not regular")
    source = BASE_TRAINING_PATH.read_bytes()
    if (
        len(source) != BASE_TRAINING_BYTE_COUNT
        or hashlib.sha256(source).hexdigest() != BASE_TRAINING_FILE_SHA256
    ):
        raise RuntimeError("frozen V24 training source binding changed")
    if PRIVATE_BASE_MODULE_NAME in sys.modules:
        raise RuntimeError("private V24 training module name is already occupied")
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


_v24 = _load_private_base_training_v25()
_base = _v24._base
_tensor_core = _v24._tensor_core
if (
    _v24.PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_TENSOR_COUNT_V24 != 96
    or _v24.PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_COUNT_V24 != 3_106_409
    or _v24.PROTECTED_PREDICTOR_CORE_PARAMETER_TENSOR_COUNT_V24 != 13
    or _v24.PROTECTED_PREDICTOR_CORE_PARAMETER_COUNT_V24 != 259_008
    or _v24.MAXIMUM_UPDATES != 1_000
    or _v24.MAXIMUM_PRESENTATIONS != 16_000
):
    raise RuntimeError("frozen V24 route or cap identity changed")

for _name in _v24.__all__:
    globals()[_name] = getattr(_v24, _name)

REQUIRED_BATCH_KEYS_V25 = tuple(_v24.REQUIRED_BATCH_KEYS_V24)
PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25 = (
    _v24.PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V24
)


@dataclass(frozen=True)
class PerRowPersistenceContrastiveTemporalTermsV25:
    loss: Any
    prediction_energy_per_row: Any
    persistence_energy_per_row: Any
    gap_per_row: Any
    row_loss_per_row: Any
    legacy_global_ratio: Any


@dataclass(frozen=True)
class JointTrainingAccountingV25:
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
class JointUpdateResultV25:
    accounting: JointTrainingAccountingV25
    mean_losses: Mapping[str, float]
    gradient_routes: Mapping[str, Any]
    gradient_l2: Mapping[str, float]
    ranking_active_microbatches: int
    ranking_eligible_pairs: int
    survival_supervised_decisions: int
    target_gradient_tensor_count: int
    optimizer_steps_this_update: int
    ema_steps_this_update: int
    per_row_persistence_contrastive_diagnostics: Mapping[str, Any]
    predictor_core_protected_survival_diagnostics: Mapping[str, float | int]


def per_row_persistence_contrastive_temporal_loss_v25(
    torch: Any,
    semantic_api: Any,
    predicted_latents: Any,
    executed_action_indices: Any,
    ema_current_latent: Any,
    ema_next_latent: Any,
) -> PerRowPersistenceContrastiveTemporalTermsV25:
    """Return the exact preregistered row-local temporal term P25."""

    if (
        not isinstance(predicted_latents, torch.Tensor)
        or predicted_latents.ndim != 5
        or tuple(predicted_latents.shape[:3])
        != (MICROBATCH_SIZE, ACTION_COUNT_V23, 64)
        or predicted_latents.shape[-2] < 1
        or predicted_latents.shape[-1] < 1
    ):
        raise ValueError("V25 predicted latents must have shape (4,9,64,H,W)")
    expected_target = (
        MICROBATCH_SIZE,
        64,
        predicted_latents.shape[-2],
        predicted_latents.shape[-1],
    )
    if (
        not isinstance(ema_current_latent, torch.Tensor)
        or not isinstance(ema_next_latent, torch.Tensor)
        or tuple(ema_current_latent.shape) != expected_target
        or tuple(ema_next_latent.shape) != expected_target
    ):
        raise ValueError("V25 EMA latents must have shape (4,64,H,W)")
    if (
        predicted_latents.dtype != ema_current_latent.dtype
        or predicted_latents.dtype != ema_next_latent.dtype
    ):
        raise TypeError("V25 prediction and EMA latent dtypes differ")
    if (
        predicted_latents.device != ema_current_latent.device
        or predicted_latents.device != ema_next_latent.device
    ):
        raise TypeError("V25 prediction and EMA latents must share a device")
    if (
        not isinstance(executed_action_indices, torch.Tensor)
        or tuple(executed_action_indices.shape) != (MICROBATCH_SIZE,)
    ):
        raise ValueError("V25 executed action indices must have shape (4,)")
    if (
        executed_action_indices.is_floating_point()
        or executed_action_indices.dtype == torch.bool
    ):
        raise TypeError("V25 executed action indices must use an integer dtype")
    action_indices = executed_action_indices.to(
        device=predicted_latents.device, dtype=torch.long
    )
    if bool(((action_indices < 0) | (action_indices >= ACTION_COUNT_V23)).any()):
        raise ValueError("V25 executed action index escaped the frozen vocabulary")

    rows = torch.arange(MICROBATCH_SIZE, device=predicted_latents.device)
    executed = predicted_latents[rows, action_indices]
    target_current = ema_current_latent.detach()
    target_next = ema_next_latent.detach()
    prediction_energy = semantic_api.latent_energy_per_row(executed, target_next)
    persistence_energy = semantic_api.latent_energy_per_row(
        target_current, target_next
    ).detach()
    for name, value in (
        ("prediction energy", prediction_energy),
        ("persistence energy", persistence_energy),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != (MICROBATCH_SIZE,)
            or value.dtype != predicted_latents.dtype
            or value.device != predicted_latents.device
            or not bool(torch.isfinite(value).all())
        ):
            raise RuntimeError(f"V25 {name} contract changed")

    gap = prediction_energy - persistence_energy
    row_loss = torch.nn.functional.softplus(
        gap,
        beta=SOFTPLUS_BETA_V25,
        threshold=SOFTPLUS_THRESHOLD_V25,
    ) / LOG2_NORMALIZER_V25
    loss = row_loss.mean()
    legacy_global_ratio = (
        prediction_energy.detach().mean()
        / persistence_energy.detach().mean().clamp_min(
            LEGACY_PERSISTENCE_BASELINE_MIN_V25
        )
    ).detach()
    for name, value in (
        ("gap", gap),
        ("row loss", row_loss),
        ("loss", loss),
        ("detached legacy global ratio", legacy_global_ratio),
    ):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"V25 {name} is nonfinite")
    return PerRowPersistenceContrastiveTemporalTermsV25(
        loss=loss,
        prediction_energy_per_row=prediction_energy.detach(),
        persistence_energy_per_row=persistence_energy.detach(),
        gap_per_row=gap.detach(),
        row_loss_per_row=row_loss.detach(),
        legacy_global_ratio=legacy_global_ratio,
    )


def validate_accounting_v25(accounting: JointTrainingAccountingV25) -> None:
    if not isinstance(accounting, JointTrainingAccountingV25):
        raise TypeError("V25 accounting has the wrong receipt type")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in accounting.__dict__.values()
    ):
        raise ValueError("V25 accounting values must be nonnegative integers")
    updates = accounting.updates
    expected = JointTrainingAccountingV25(
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
        raise RuntimeError("V25 accounting is inconsistent")


def _advance_accounting_v25(
    accounting: JointTrainingAccountingV25,
) -> JointTrainingAccountingV25:
    return JointTrainingAccountingV25(
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


def _validate_update_capacity_v25(accounting: JointTrainingAccountingV25) -> None:
    validate_accounting_v25(accounting)
    if (
        accounting.updates >= MAXIMUM_UPDATES
        or accounting.presentations + PRESENTATIONS_PER_UPDATE
        > MAXIMUM_PRESENTATIONS
    ):
        raise PermissionError("V25 training cap leaves no complete update available")


def _validate_microbatches_v25(
    torch: Any, microbatches: Sequence[Mapping[str, Any]]
) -> None:
    if len(microbatches) != MICROBATCHES_PER_UPDATE:
        raise ValueError("V25 update must contain exactly four microbatches")
    if any(tuple(batch) != REQUIRED_BATCH_KEYS_V25 for batch in microbatches):
        raise ValueError("V25 microbatch schema changed from frozen V24")
    _v24._validate_microbatches_v24(torch, microbatches)


def _detached_row_values_v25(value: Any) -> tuple[float, ...]:
    flattened = value.detach().reshape(-1).cpu().tolist()
    return tuple(float(item) for item in flattened)


def _add_summary_v25(
    destination: dict[str, Any], prefix: str, values: Sequence[float]
) -> None:
    if not values or any(not math.isfinite(value) for value in values):
        raise RuntimeError(f"V25 {prefix} diagnostics are empty or nonfinite")
    destination[f"{prefix}_count"] = len(values)
    destination[f"{prefix}_sum"] = sum(values)
    destination[f"{prefix}_mean"] = sum(values) / len(values)
    destination[f"{prefix}_minimum"] = min(values)
    destination[f"{prefix}_maximum"] = max(values)


def joint_training_update_v25(
    model: Any,
    optimizer: Any,
    microbatches: Sequence[Mapping[str, Any]],
    *,
    accounting: JointTrainingAccountingV25 | None = None,
) -> JointUpdateResultV25:
    """Run one V24 joint update with only P replaced by P25."""

    torch, semantic_api, survival_api, *_ = _tensor_core._runtime_apis()
    state = JointTrainingAccountingV25() if accounting is None else accounting
    _validate_update_capacity_v25(state)
    _validate_microbatches_v25(torch, microbatches)
    partition = _v24.partition_parameters_v24(model)
    _v24.validate_optimizer_v24(optimizer, partition)
    auxiliary_subset = (
        _v24.predictor_core_protected_survival_parameter_subset_v24(partition)
    )
    auxiliary_parameter_ids = {id(value) for value in auxiliary_subset.parameters}
    protected_parameter_ids = {
        id(value) for value in auxiliary_subset.protected_predictor_core_parameters
    }
    if auxiliary_parameter_ids & protected_parameter_ids:
        raise RuntimeError("V25 protected predictor core entered the J24 route")
    ema_before = int(model.ema_update_count.item())
    if ema_before != state.ema_steps:
        raise RuntimeError("model EMA count disagrees with V25 accounting")
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("V25 EMA target already has a gradient")

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
            PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25,
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
            "F",
            "J_rank",
            "J24",
            "N",
            "C",
            "L",
            "positive_energy_sum",
            "scene_negative_energy_sum",
            "prior_negative_energy_sum",
            "scene_advantage_sum",
            "prior_advantage_sum",
            "scene_rank_sum",
            "prior_rank_sum",
        )
    }
    prediction_energy_rows: list[float] = []
    persistence_energy_rows: list[float] = []
    gap_rows: list[float] = []
    temporal_row_losses: list[float] = []
    legacy_global_ratios: list[float] = []
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
        auxiliary = _v24.predictor_core_protected_survival_objective_v24(
            torch,
            survival_api,
            survival_logits,
            batch[PREFIX_LENGTHS_KEY],
            batch[SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21],
            batch[ACTION_PRIOR_M_KEY_V23],
        )
        temporal = per_row_persistence_contrastive_temporal_loss_v25(
            torch,
            semantic_api,
            predicted,
            batch[EXECUTED_ACTION_KEY],
            ema_current,
            ema_next,
        )
        joint = survival_api.joint_survival_loss_v1(
            semantic_loss=semantic.loss,
            executed_action_ema_latent_loss=temporal.loss,
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
            ("per-row persistence-contrastive P25", temporal.loss),
            ("predictor-core-protected survival J24", auxiliary.loss),
            ("joint N25", navigation),
            ("Camera C", camera.total),
        ):
            _tensor_core._finite_tensor(torch, value, name)
        if (
            not camera.total.requires_grad
            or not navigation.requires_grad
            or not auxiliary.loss.requires_grad
        ):
            raise RuntimeError("V25 C, N25, and J24 must retain gradient graphs")

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
        absent[PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25] += (
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
        prediction_energy_rows.extend(
            _detached_row_values_v25(temporal.prediction_energy_per_row)
        )
        persistence_energy_rows.extend(
            _detached_row_values_v25(temporal.persistence_energy_per_row)
        )
        gap_rows.extend(_detached_row_values_v25(temporal.gap_per_row))
        temporal_row_losses.extend(
            _detached_row_values_v25(temporal.row_loss_per_row)
        )
        legacy_global_ratios.append(_tensor_core._scalar(temporal.legacy_global_ratio))
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
        PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25: (
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
        PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25,
    ):
        if not (_tensor_core._scalar(route_values[name][0]) > 0.0):
            raise RuntimeError(f"required V25 gradient route {name!r} is zero")
    if absent[PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25] != 0:
        raise RuntimeError("V25 J24 auxiliary has an absent gradient")

    c_scale = route_values["camera_shared"][1]
    n_scale = route_values["joint_shared"][1]
    representation_scale = route_values["representation"][1]
    predictor_scale = route_values["predictor"][1]
    auxiliary_scale = route_values[
        PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25
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
                raise RuntimeError("V25 swept-progress output left the J24 route")
            gradient = gradient + auxiliary_scale * auxiliary_by_id[id(parameter)]
        elif id(parameter) in auxiliary_by_id:
            raise RuntimeError("V25 latent predictor core received J24")
        parameter.grad = gradient

    target_gradient_count = sum(
        parameter.grad is not None for parameter in partition.target
    )
    if target_gradient_count:
        raise RuntimeError("V25 EMA target received a gradient")
    optimizer.step()
    for parameter in partition.online:
        _tensor_core._finite_tensor(torch, parameter, "V25 online parameter")
    model.update_target_ema_after_optimizer_step()
    ema_after = int(model.ema_update_count.item())
    if ema_after != ema_before + 1:
        raise RuntimeError("V25 EMA did not update exactly once")
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("V25 EMA target received a gradient")

    advanced = _advance_accounting_v25(state)
    if advanced.ema_steps != ema_after:
        raise RuntimeError("post-update V25 EMA count disagrees with accounting")
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
            "F",
            "J_rank",
            "J24",
            "N",
            "C",
            "L",
        )
    }
    if positive_count != PRESENTATIONS_PER_UPDATE * NON_HOLD_ACTION_COUNT_V23:
        raise RuntimeError("V25 positive-energy accounting changed")
    if scene_count < MICROBATCHES_PER_UPDATE or prior_count < MICROBATCHES_PER_UPDATE:
        raise RuntimeError("V25 comparison accounting changed")
    if not (
        len(prediction_energy_rows)
        == len(persistence_energy_rows)
        == len(gap_rows)
        == len(temporal_row_losses)
        == PRESENTATIONS_PER_UPDATE
        and len(legacy_global_ratios) == MICROBATCHES_PER_UPDATE
    ):
        raise RuntimeError("V25 temporal diagnostic accounting changed")

    temporal_diagnostics: dict[str, Any] = {
        "mechanism": PER_ROW_PERSISTENCE_CONTRASTIVE_MECHANISM_V25,
        "prediction_energy_per_row": tuple(prediction_energy_rows),
        "persistence_energy_per_row": tuple(persistence_energy_rows),
        "gap_per_row": tuple(gap_rows),
        "row_loss_per_row": tuple(temporal_row_losses),
        "legacy_global_ratio_per_microbatch": tuple(legacy_global_ratios),
        "negative_gap_count": sum(value < 0.0 for value in gap_rows),
        "negative_gap_fraction": (
            sum(value < 0.0 for value in gap_rows) / len(gap_rows)
        ),
        "log2_normalizer": LOG2_NORMALIZER_V25,
        "softplus_beta": SOFTPLUS_BETA_V25,
        "softplus_threshold": SOFTPLUS_THRESHOLD_V25,
        "denominator_used": False,
    }
    for prefix, values in (
        ("prediction_energy", prediction_energy_rows),
        ("persistence_energy", persistence_energy_rows),
        ("gap", gap_rows),
        ("row_loss", temporal_row_losses),
        ("legacy_global_ratio", legacy_global_ratios),
    ):
        _add_summary_v25(temporal_diagnostics, prefix, values)
    if not math.isclose(
        mean["P"],
        temporal_diagnostics["row_loss_mean"],
        rel_tol=1.0e-6,
        abs_tol=1.0e-7,
    ):
        raise RuntimeError("V25 P receipt differs from the per-row loss mean")

    return JointUpdateResultV25(
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
        per_row_persistence_contrastive_diagnostics=temporal_diagnostics,
        predictor_core_protected_survival_diagnostics={
            "positive_energy_sum": sums["positive_energy_sum"],
            "positive_energy_count": positive_count,
            "positive_energy_mean": sums["positive_energy_sum"] / positive_count,
            "scene_negative_energy_sum": sums["scene_negative_energy_sum"],
            "scene_eligible_count": scene_count,
            "scene_negative_energy_mean": (
                sums["scene_negative_energy_sum"] / scene_count
            ),
            "scene_advantage_sum": sums["scene_advantage_sum"],
            "scene_advantage_mean": sums["scene_advantage_sum"] / scene_count,
            "scene_rank_sum": sums["scene_rank_sum"],
            "prior_negative_energy_sum": sums["prior_negative_energy_sum"],
            "prior_eligible_count": prior_count,
            "prior_negative_energy_mean": (
                sums["prior_negative_energy_sum"] / prior_count
            ),
            "prior_advantage_sum": sums["prior_advantage_sum"],
            "prior_advantage_mean": sums["prior_advantage_sum"] / prior_count,
            "prior_rank_sum": sums["prior_rank_sum"],
            "non_hold_action_count_per_row": NON_HOLD_ACTION_COUNT_V23,
        },
    )


def private_training_adapter_receipt_v25() -> dict[str, Any]:
    return {
        "schema": (
            "lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
            "v25_training_adapter_v1"
        ),
        "base_training": str(BASE_TRAINING_PATH.relative_to(ROOT)),
        "base_frozen_source_and_review_commit": BASE_FROZEN_SOURCE_AND_REVIEW_COMMIT,
        "base_training_file_sha256": BASE_TRAINING_FILE_SHA256,
        "base_training_byte_count": BASE_TRAINING_BYTE_COUNT,
        "public_base_was_loaded_before_adapter": _PUBLIC_BASE_WAS_LOADED_BEFORE_ADAPTER,
        "public_base_loaded_by_adapter": False,
        "private_module_registered": PRIVATE_BASE_MODULE_NAME in sys.modules,
        "preregistration_commit": PREREGISTRATION_COMMIT_V25,
        "preregistration_file_sha256": PREREGISTRATION_FILE_SHA256_V25,
        "preregistration_byte_count": PREREGISTRATION_BYTE_COUNT_V25,
        "temporal_mechanism": PER_ROW_PERSISTENCE_CONTRASTIVE_MECHANISM_V25,
        "softplus_beta": SOFTPLUS_BETA_V25,
        "softplus_threshold": SOFTPLUS_THRESHOLD_V25,
        "log2_normalizer": LOG2_NORMALIZER_V25,
        "denominator_used": False,
        "legacy_global_ratio_diagnostic_only": True,
        "j24_delegated_bit_identical_to_v24": True,
        "j24_parameter_tensor_count": (
            _v24.PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_TENSOR_COUNT_V24
        ),
        "j24_parameter_count": (
            _v24.PREDICTOR_CORE_PROTECTED_SURVIVAL_PARAMETER_COUNT_V24
        ),
        "protected_predictor_core_parameter_tensor_count": (
            _v24.PROTECTED_PREDICTOR_CORE_PARAMETER_TENSOR_COUNT_V24
        ),
        "protected_predictor_core_parameter_count": (
            _v24.PROTECTED_PREDICTOR_CORE_PARAMETER_COUNT_V24
        ),
        "inherited_joint_predictor_parameter_tensor_count": 15,
        "maximum_updates": MAXIMUM_UPDATES,
        "maximum_presentations": MAXIMUM_PRESENTATIONS,
    }


predictor_core_protected_survival_objective_v25 = (
    _v24.predictor_core_protected_survival_objective_v24
)
predictor_core_protected_survival_parameter_subset_v25 = (
    _v24.predictor_core_protected_survival_parameter_subset_v24
)
partition_parameters_v25 = _v24.partition_parameters_v24
build_frozen_optimizer_v25 = _v24.build_frozen_optimizer_v24
validate_optimizer_v25 = _v24.validate_optimizer_v24

# Compatibility names consumed by the unchanged inherited lifecycle.
joint_training_update_v24 = joint_training_update_v25
validate_accounting_v24 = validate_accounting_v25
joint_training_update_v23 = joint_training_update_v25
validate_accounting_v23 = validate_accounting_v25
state_residual_survival_objective_v23 = (
    predictor_core_protected_survival_objective_v25
)
state_residual_survival_parameter_subset_v23 = (
    predictor_core_protected_survival_parameter_subset_v25
)
joint_training_update_v21 = joint_training_update_v25
validate_accounting_v21 = validate_accounting_v25
partition_parameters_v21 = partition_parameters_v25
build_frozen_optimizer_v21 = build_frozen_optimizer_v25
validate_optimizer_v21 = validate_optimizer_v25
joint_training_update_v19 = joint_training_update_v25
validate_accounting_v19 = validate_accounting_v25
joint_training_update_v18 = joint_training_update_v25
validate_accounting_v18 = validate_accounting_v25
JointTrainingAccountingV13 = JointTrainingAccountingV25
JointTrainingAccountingV19 = JointTrainingAccountingV25
JointTrainingAccountingV21 = JointTrainingAccountingV25
JointTrainingAccountingV23 = JointTrainingAccountingV25
JointTrainingAccountingV24 = JointTrainingAccountingV25
JointUpdateResultV13 = JointUpdateResultV25
JointUpdateResultV19 = JointUpdateResultV25
JointUpdateResultV21 = JointUpdateResultV25
JointUpdateResultV23 = JointUpdateResultV25
JointUpdateResultV24 = JointUpdateResultV25
partition_parameters_v13 = partition_parameters_v25
partition_parameters_v24 = partition_parameters_v25
build_frozen_optimizer_v13 = build_frozen_optimizer_v25
build_frozen_optimizer_v24 = build_frozen_optimizer_v25
validate_optimizer_v13 = validate_optimizer_v25
validate_optimizer_v24 = validate_optimizer_v25
joint_training_update_v13 = joint_training_update_v25
validate_accounting_v13 = validate_accounting_v25
_validate_microbatches_v13 = _validate_microbatches_v25
_validate_microbatches_v21 = _validate_microbatches_v25
_validate_microbatches_v23 = _validate_microbatches_v25
_validate_microbatches_v24 = _validate_microbatches_v25


__all__ = tuple(
    dict.fromkeys(
        (
            *_v24.__all__,
            "JointTrainingAccountingV25",
            "JointUpdateResultV25",
            "LEGACY_PERSISTENCE_BASELINE_MIN_V25",
            "LOG2_NORMALIZER_V25",
            "PER_ROW_PERSISTENCE_CONTRASTIVE_MECHANISM_V25",
            "PREDICTOR_CORE_PROTECTED_SURVIVAL_ROUTE_NAME_V25",
            "PREREGISTRATION_BYTE_COUNT_V25",
            "PREREGISTRATION_COMMIT_V25",
            "PREREGISTRATION_FILE_SHA256_V25",
            "PerRowPersistenceContrastiveTemporalTermsV25",
            "REQUIRED_BATCH_KEYS_V25",
            "SOFTPLUS_BETA_V25",
            "SOFTPLUS_THRESHOLD_V25",
            "build_frozen_optimizer_v25",
            "joint_training_update_v25",
            "partition_parameters_v25",
            "per_row_persistence_contrastive_temporal_loss_v25",
            "predictor_core_protected_survival_objective_v25",
            "predictor_core_protected_survival_parameter_subset_v25",
            "private_training_adapter_receipt_v25",
            "validate_accounting_v25",
            "validate_optimizer_v25",
        )
    )
)
