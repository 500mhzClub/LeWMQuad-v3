#!/usr/bin/env python3
"""Source-only V19 executed-successor semantic-grounding adapter.

V19 privately loads the frozen V18 training adapter and preserves its Camera,
joint-JEPA, optimizer, clipping, batching, and EMA paths.  The sole scientific
addition is one independently clipped gradient route from the factual
executed-action prediction's next-scene semantic NLL to the inherited latent
transition predictor (excluding the swept-progress head).

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
BASE_TRAINING_PATH = (
    ROOT / "scripts/run_go2_rgb_object_space_height_volume_joint_jepa_v18.py"
)
BASE_PUBLIC_MODULE_NAME = (
    "scripts.run_go2_rgb_object_space_height_volume_joint_jepa_v18"
)
PRIVATE_BASE_MODULE_NAME = f"{__name__}.__private_v18_training"
_PUBLIC_BASE_WAS_LOADED_BEFORE_ADAPTER = BASE_PUBLIC_MODULE_NAME in sys.modules

FACTUAL_SUCCESSOR_GRADIENT_NORM_CAP_V19 = 1.0
FACTUAL_SUCCESSOR_PREDICTOR_PARAMETER_TENSOR_COUNT_V19 = 13
FACTUAL_SUCCESSOR_PREDICTOR_PARAMETER_COUNT_V19 = 259_008
FACTUAL_SUCCESSOR_EXCLUDED_PREFIX_V19 = "predictor.swept_progress_head."
FACTUAL_SUCCESSOR_EXCLUDED_PARAMETER_TENSOR_COUNT_V19 = 2
FACTUAL_SUCCESSOR_EXCLUDED_PARAMETER_COUNT_V19 = 65
SEMANTIC_CLASS_NORMALIZER_V19 = math.log(3.0)


def _load_private_base_training_v19() -> ModuleType:
    if BASE_TRAINING_PATH.is_symlink() or not BASE_TRAINING_PATH.is_file():
        raise FileNotFoundError("reviewed V18 training source is absent or not regular")
    source = BASE_TRAINING_PATH.read_bytes()
    if not source:
        raise RuntimeError("reviewed V18 training source is empty")
    if PRIVATE_BASE_MODULE_NAME in sys.modules:
        raise RuntimeError("private V18 training module name is already occupied")
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


_base = _load_private_base_training_v19()
_tensor_core = _base._base
if (
    _base.MICROBATCH_SIZE != 4
    or _base.MICROBATCHES_PER_UPDATE != 4
    or _base.PRESENTATIONS_PER_UPDATE != 16
    or _base.MAXIMUM_UPDATES != 1_000
    or _base.MAXIMUM_PRESENTATIONS != 16_000
):
    raise RuntimeError("reviewed V18 training cap or batching changed")
if tuple(_base.ACTION_ORDER).index("hold") != 6:
    raise RuntimeError("reviewed V18 HOLD action index changed")
HOLD_ACTION_INDEX_V19 = 6

# Re-export the exact reviewed V18 surface before replacing only the training
# lifecycle hooks whose accounting and result receipts V19 extends.
for _name in _base.__all__:
    globals()[_name] = getattr(_base, _name)


@dataclass(frozen=True)
class FactualSuccessorPredictorSubsetV19:
    """The registered latent-transition subset of the V18 predictor."""

    parameters: tuple[Any, ...]
    names: tuple[str, ...]
    predictor_indices: tuple[int, ...]
    parameter_count: int


@dataclass(frozen=True)
class JointTrainingAccountingV19:
    """V18 accounting plus the independently requested factual route."""

    updates: int = 0
    presentations: int = 0
    microbatch_graphs: int = 0
    backward_calls: int = 0
    camera_route_grad_calls: int = 0
    joint_route_grad_calls: int = 0
    factual_successor_grad_calls: int = 0
    camera_frame_objectives: int = 0
    optimizer_steps: int = 0
    ema_steps: int = 0
    predictor_forwards: int = 0
    predictor_objectives: int = 0
    factual_successor_objectives: int = 0


@dataclass(frozen=True)
class JointUpdateResultV19:
    """One V19 update, including all inherited and factual-route receipts."""

    accounting: JointTrainingAccountingV19
    mean_losses: Mapping[str, float]
    gradient_routes: Mapping[str, Any]
    gradient_l2: Mapping[str, float]
    ranking_active_microbatches: int
    ranking_eligible_pairs: int
    survival_supervised_decisions: int
    target_gradient_tensor_count: int
    optimizer_steps_this_update: int
    ema_steps_this_update: int
    factual_successor_diagnostics: Mapping[str, float | int]


def factual_successor_predictor_subset_v19(
    partition: Any,
) -> FactualSuccessorPredictorSubsetV19:
    """Resolve and strictly validate the 13-tensor V19 transition subset."""

    predictor = tuple(partition.predictor)
    names = tuple(partition.names["predictor"])
    if len(predictor) != len(names) or not predictor:
        raise RuntimeError("V19 predictor parameter/name inventory changed")
    if any(not name.startswith("predictor.") for name in names):
        raise RuntimeError("V19 predictor contains a parameter outside predictor.*")

    selected_indices = tuple(
        index
        for index, name in enumerate(names)
        if not name.startswith(FACTUAL_SUCCESSOR_EXCLUDED_PREFIX_V19)
    )
    excluded_indices = tuple(
        index
        for index, name in enumerate(names)
        if name.startswith(FACTUAL_SUCCESSOR_EXCLUDED_PREFIX_V19)
    )
    selected = tuple(predictor[index] for index in selected_indices)
    selected_names = tuple(names[index] for index in selected_indices)
    excluded = tuple(predictor[index] for index in excluded_indices)
    parameter_count = sum(int(parameter.numel()) for parameter in selected)
    excluded_parameter_count = sum(int(parameter.numel()) for parameter in excluded)
    if (
        len(selected) != FACTUAL_SUCCESSOR_PREDICTOR_PARAMETER_TENSOR_COUNT_V19
        or parameter_count != FACTUAL_SUCCESSOR_PREDICTOR_PARAMETER_COUNT_V19
        or len(excluded)
        != FACTUAL_SUCCESSOR_EXCLUDED_PARAMETER_TENSOR_COUNT_V19
        or excluded_parameter_count
        != FACTUAL_SUCCESSOR_EXCLUDED_PARAMETER_COUNT_V19
        or any(not parameter.requires_grad for parameter in selected)
        or len({id(parameter) for parameter in selected}) != len(selected)
    ):
        raise RuntimeError("V19 factual-successor predictor subset changed")
    return FactualSuccessorPredictorSubsetV19(
        parameters=selected,
        names=selected_names,
        predictor_indices=selected_indices,
        parameter_count=parameter_count,
    )


def validate_accounting_v19(accounting: JointTrainingAccountingV19) -> None:
    """Validate the exact 12-call, eight-predictor-objective lifecycle."""

    if not isinstance(accounting, JointTrainingAccountingV19):
        raise TypeError("V19 accounting has the wrong receipt type")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in accounting.__dict__.values()
    ):
        raise ValueError("V19 accounting values must be nonnegative integers")
    updates = accounting.updates
    expected = JointTrainingAccountingV19(
        updates=updates,
        presentations=updates * PRESENTATIONS_PER_UPDATE,
        microbatch_graphs=updates * MICROBATCHES_PER_UPDATE,
        backward_calls=updates * 3 * MICROBATCHES_PER_UPDATE,
        camera_route_grad_calls=updates * MICROBATCHES_PER_UPDATE,
        joint_route_grad_calls=updates * MICROBATCHES_PER_UPDATE,
        factual_successor_grad_calls=updates * MICROBATCHES_PER_UPDATE,
        camera_frame_objectives=(
            updates
            * MICROBATCHES_PER_UPDATE
            * _tensor_core.CAMERA_FRAMES_PER_MICROBATCH
        ),
        optimizer_steps=updates,
        ema_steps=updates,
        predictor_forwards=updates * MICROBATCHES_PER_UPDATE,
        predictor_objectives=updates * 2 * MICROBATCHES_PER_UPDATE,
        factual_successor_objectives=updates * MICROBATCHES_PER_UPDATE,
    )
    if accounting != expected:
        raise RuntimeError("V19 joint-training accounting is inconsistent")


def _advance_accounting_v19(
    accounting: JointTrainingAccountingV19,
) -> JointTrainingAccountingV19:
    result = JointTrainingAccountingV19(
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
        factual_successor_grad_calls=(
            accounting.factual_successor_grad_calls + MICROBATCHES_PER_UPDATE
        ),
        camera_frame_objectives=(
            accounting.camera_frame_objectives
            + MICROBATCHES_PER_UPDATE
            * _tensor_core.CAMERA_FRAMES_PER_MICROBATCH
        ),
        optimizer_steps=accounting.optimizer_steps + 1,
        ema_steps=accounting.ema_steps + 1,
        predictor_forwards=accounting.predictor_forwards + MICROBATCHES_PER_UPDATE,
        predictor_objectives=(
            accounting.predictor_objectives + 2 * MICROBATCHES_PER_UPDATE
        ),
        factual_successor_objectives=(
            accounting.factual_successor_objectives + MICROBATCHES_PER_UPDATE
        ),
    )
    validate_accounting_v19(result)
    return result


def _validate_update_capacity_v19(accounting: JointTrainingAccountingV19) -> None:
    validate_accounting_v19(accounting)
    if (
        accounting.updates >= MAXIMUM_UPDATES
        or accounting.presentations + PRESENTATIONS_PER_UPDATE
        > MAXIMUM_PRESENTATIONS
    ):
        raise PermissionError("V19 training cap leaves no complete update available")


def _factual_latents_v19(
    torch: Any,
    predicted: Any,
    executed_actions: Any,
) -> Any:
    if (
        not isinstance(predicted, torch.Tensor)
        or predicted.ndim != 5
        or predicted.shape[0] != MICROBATCH_SIZE
        or predicted.shape[1] != len(ACTION_ORDER)
    ):
        raise ValueError("V19 all-action prediction shape changed")
    if (
        not isinstance(executed_actions, torch.Tensor)
        or tuple(executed_actions.shape) != (MICROBATCH_SIZE,)
        or executed_actions.is_floating_point()
        or executed_actions.dtype == torch.bool
        or executed_actions.device != predicted.device
    ):
        raise ValueError("V19 executed actions must be integer B=4 on prediction device")
    actions = executed_actions.long()
    if not bool(((actions >= 0) & (actions < len(ACTION_ORDER))).all().item()):
        raise ValueError("V19 executed action index escaped the frozen vocabulary")
    rows = torch.arange(MICROBATCH_SIZE, device=predicted.device)
    return predicted[rows, actions]


def _matching_gradient_cosine_v19(
    torch: Any,
    inherited: Sequence[Any],
    factual: Sequence[Any],
) -> float:
    if len(inherited) != len(factual) or not inherited:
        raise RuntimeError("V19 cosine gradient inventories changed")
    dot = inherited[0].new_zeros((), dtype=torch.float32)
    inherited_square = dot.clone()
    factual_square = dot.clone()
    for inherited_gradient, factual_gradient in zip(
        inherited, factual, strict=True
    ):
        left = inherited_gradient.float()
        right = factual_gradient.float()
        dot = dot + (left * right).sum(dtype=torch.float32)
        inherited_square = inherited_square + (left * left).sum(dtype=torch.float32)
        factual_square = factual_square + (right * right).sum(dtype=torch.float32)
    inherited_norm = torch.sqrt(inherited_square)
    factual_norm = torch.sqrt(factual_square)
    if not bool(torch.isfinite(inherited_norm)) or not bool(torch.isfinite(factual_norm)):
        raise FloatingPointError("V19 matching predictor gradient norm is nonfinite")
    if not (_tensor_core._scalar(inherited_norm) > 0.0) or not (
        _tensor_core._scalar(factual_norm) > 0.0
    ):
        raise RuntimeError("V19 matching predictor gradient is zero")
    cosine = torch.clamp(dot / (inherited_norm * factual_norm), -1.0, 1.0)
    return _tensor_core._scalar(cosine)


def joint_training_update_v19(
    model: Any,
    optimizer: Any,
    microbatches: Sequence[Mapping[str, Any]],
    *,
    accounting: JointTrainingAccountingV19 | None = None,
) -> JointUpdateResultV19:
    """Run one V18 update plus the isolated factual-successor route."""

    torch, semantic_api, survival_api, *_ = _tensor_core._runtime_apis()
    state = JointTrainingAccountingV19() if accounting is None else accounting
    _validate_update_capacity_v19(state)
    _base._validate_microbatches_v13(torch, microbatches)
    partition = _base.partition_parameters_v18(model)
    _base.validate_optimizer_v18(optimizer, partition)
    factual_subset = factual_successor_predictor_subset_v19(partition)
    ema_before = int(model.ema_update_count.item())
    if ema_before != state.ema_steps:
        raise RuntimeError("model EMA count disagrees with V19 accounting")
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("V19 EMA target already has a gradient")

    optimizer.zero_grad(set_to_none=True)
    camera_shared = _tensor_core._zero_accumulators(partition.shared)
    joint_shared = _tensor_core._zero_accumulators(partition.shared)
    joint_representation = _tensor_core._zero_accumulators(
        partition.representation
    )
    joint_predictor = _tensor_core._zero_accumulators(partition.predictor)
    factual_predictor = _tensor_core._zero_accumulators(
        factual_subset.parameters
    )
    absent = {
        name: 0
        for name in (
            "camera_shared",
            "joint_shared",
            "representation",
            "predictor",
            "factual_successor_predictor",
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
            "Q",
            "N",
            "C",
            "L",
            "Q_persistence",
            "changed_cell_fraction",
        )
    }
    active_ranking = eligible_pairs = supervised_decisions = 0
    non_hold_rows = 0

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
        factual_latent = _factual_latents_v19(
            torch, predicted, batch[EXECUTED_ACTION_KEY]
        )
        factual_logits = model.semantic_logits_from_latent(factual_latent)
        factual_rows = semantic_api.final_class_macro_nll_per_row(
            factual_logits, batch[NEXT_LABELS_KEY]
        )
        factual_successor = factual_rows.mean() / SEMANTIC_CLASS_NORMALIZER_V19
        with torch.no_grad():
            persistence_logits = model.semantic_logits_from_latent(
                current_latent.detach()
            )
            persistence_rows = semantic_api.final_class_macro_nll_per_row(
                persistence_logits, batch[NEXT_LABELS_KEY]
            )
            factual_persistence = (
                persistence_rows.mean() / SEMANTIC_CLASS_NORMALIZER_V19
            )
            changed_fraction = (
                batch[CURRENT_LABELS_KEY] != batch[NEXT_LABELS_KEY]
            ).to(dtype=torch.float32).mean()

        with torch.no_grad():
            ema_current = model.encode_target(batch[CURRENT_RGB_KEY])
            ema_next = model.encode_target(batch[NEXT_RGB_KEY])
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
            ("factual successor Q", factual_successor),
            ("factual persistence diagnostic", factual_persistence),
            ("changed-cell fraction", changed_fraction),
            ("joint N", navigation),
            ("Camera C", camera.total),
        ):
            _tensor_core._finite_tensor(torch, value, name)
        if (
            not camera.total.requires_grad
            or not navigation.requires_grad
            or not factual_successor.requires_grad
        ):
            raise RuntimeError("V19 C, N, and Q must retain gradient graphs")

        c_gradients = torch.autograd.grad(
            camera.total / MICROBATCHES_PER_UPDATE,
            partition.shared,
            retain_graph=True,
            allow_unused=True,
        )
        n_parameters = (
            partition.shared + partition.representation + partition.predictor
        )
        n_gradients = torch.autograd.grad(
            navigation / MICROBATCHES_PER_UPDATE,
            n_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        q_gradients = torch.autograd.grad(
            factual_successor / MICROBATCHES_PER_UPDATE,
            factual_subset.parameters,
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
        absent["factual_successor_predictor"] += (
            _tensor_core._accumulate_gradients(factual_predictor, q_gradients)
        )

        for name, value in (
            ("S", joint.semantic),
            ("P", joint.executed_action_ema_latent),
            ("U", joint.survival),
            ("R", joint.progress_ranking),
            ("O", occupied.loss),
            ("Q", factual_successor),
            ("N", navigation),
            ("C", camera.total),
            ("L", navigation + camera.total + factual_successor),
            ("Q_persistence", factual_persistence),
            ("changed_cell_fraction", changed_fraction),
        ):
            sums[name] += _tensor_core._scalar(value)
        actions = batch[EXECUTED_ACTION_KEY].long()
        non_hold_rows += int((actions != HOLD_ACTION_INDEX_V19).sum().item())
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
        "factual_successor_predictor": (
            factual_subset.parameters,
            factual_predictor,
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
        "factual_successor_predictor",
    ):
        if not (_tensor_core._scalar(route_values[name][0]) > 0.0):
            raise RuntimeError(f"required V19 gradient route {name!r} is zero")
    if absent["factual_successor_predictor"] != 0:
        raise RuntimeError("V19 factual-successor route has an absent gradient")

    matching_inherited = tuple(
        joint_predictor[index] for index in factual_subset.predictor_indices
    )
    gradient_cosine = _matching_gradient_cosine_v19(
        torch, matching_inherited, factual_predictor
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
    factual_scale = route_values["factual_successor_predictor"][1]
    factual_by_predictor_index = {
        index: gradient
        for index, gradient in zip(
            factual_subset.predictor_indices, factual_predictor, strict=True
        )
    }
    for index, (parameter, inherited_gradient) in enumerate(
        zip(partition.predictor, joint_predictor, strict=True)
    ):
        gradient = predictor_scale * inherited_gradient
        if index in factual_by_predictor_index:
            gradient = gradient + factual_scale * factual_by_predictor_index[index]
        parameter.grad = gradient

    target_gradient_count = sum(
        parameter.grad is not None for parameter in partition.target
    )
    if target_gradient_count:
        raise RuntimeError("V19 EMA target received a gradient")
    optimizer.step()
    for parameter in partition.online:
        _tensor_core._finite_tensor(torch, parameter, "V19 online parameter")
    model.update_target_ema_after_optimizer_step()
    ema_after = int(model.ema_update_count.item())
    if ema_after != ema_before + 1:
        raise RuntimeError("V19 EMA did not update exactly once")
    if any(parameter.grad is not None for parameter in partition.target):
        raise RuntimeError("V19 EMA target received a gradient")

    advanced = _advance_accounting_v19(state)
    if advanced.ema_steps != ema_after:
        raise RuntimeError("post-update V19 EMA count disagrees with accounting")
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
        for name in ("S", "P", "U", "R", "O", "Q", "N", "C", "L")
    }
    successor = mean["Q"]
    persistence_diagnostic = sums["Q_persistence"] / MICROBATCHES_PER_UPDATE
    return JointUpdateResultV19(
        accounting=advanced,
        mean_losses=mean,
        gradient_routes=receipts,
        gradient_l2={
            name: receipt.preclip_l2 for name, receipt in receipts.items()
        },
        ranking_active_microbatches=active_ranking,
        ranking_eligible_pairs=eligible_pairs,
        survival_supervised_decisions=supervised_decisions,
        target_gradient_tensor_count=target_gradient_count,
        optimizer_steps_this_update=1,
        ema_steps_this_update=1,
        factual_successor_diagnostics={
            "successor_semantic_nll_normalized": successor,
            "persistence_semantic_nll_normalized": persistence_diagnostic,
            "successor_minus_persistence_nll_normalized": (
                successor - persistence_diagnostic
            ),
            "changed_cell_fraction": (
                sums["changed_cell_fraction"] / MICROBATCHES_PER_UPDATE
            ),
            "non_hold_row_count": non_hold_rows,
            "matching_predictor_gradient_cosine": gradient_cosine,
        },
    )


def private_training_adapter_receipt_v19() -> dict[str, Any]:
    return {
        "schema": (
            "lewm_go2_rgb_object_space_height_volume_executed_successor_"
            "semantic_grounding_joint_jepa_v19_training_adapter_v1"
        ),
        "base_training": str(BASE_TRAINING_PATH.relative_to(ROOT)),
        "public_base_was_loaded_before_adapter": (
            _PUBLIC_BASE_WAS_LOADED_BEFORE_ADAPTER
        ),
        "public_base_loaded_by_adapter": False,
        "private_module_registered": PRIVATE_BASE_MODULE_NAME in sys.modules,
        "factual_successor_gradient_norm_cap": (
            FACTUAL_SUCCESSOR_GRADIENT_NORM_CAP_V19
        ),
        "factual_successor_predictor_parameter_tensor_count": (
            FACTUAL_SUCCESSOR_PREDICTOR_PARAMETER_TENSOR_COUNT_V19
        ),
        "factual_successor_predictor_parameter_count": (
            FACTUAL_SUCCESSOR_PREDICTOR_PARAMETER_COUNT_V19
        ),
        "excluded_predictor_prefix": FACTUAL_SUCCESSOR_EXCLUDED_PREFIX_V19,
        "maximum_updates": MAXIMUM_UPDATES,
        "maximum_presentations": MAXIMUM_PRESENTATIONS,
    }


# The inherited executor intentionally resolves these compatibility spellings.
partition_parameters_v19 = _base.partition_parameters_v18
build_frozen_optimizer_v19 = _base.build_frozen_optimizer_v18
validate_optimizer_v19 = _base.validate_optimizer_v18
partition_parameters_v18 = partition_parameters_v19
build_frozen_optimizer_v18 = build_frozen_optimizer_v19
validate_optimizer_v18 = validate_optimizer_v19
joint_training_update_v18 = joint_training_update_v19
validate_accounting_v18 = validate_accounting_v19
JointTrainingAccountingV13 = JointTrainingAccountingV19
JointUpdateResultV13 = JointUpdateResultV19
partition_parameters_v13 = partition_parameters_v19
build_frozen_optimizer_v13 = build_frozen_optimizer_v19
validate_optimizer_v13 = validate_optimizer_v19
joint_training_update_v13 = joint_training_update_v19
validate_accounting_v13 = validate_accounting_v19
_validate_microbatches_v13 = _base._validate_microbatches_v13


__all__ = tuple(
    dict.fromkeys(
        (
            *_base.__all__,
            "FACTUAL_SUCCESSOR_EXCLUDED_PARAMETER_COUNT_V19",
            "FACTUAL_SUCCESSOR_EXCLUDED_PARAMETER_TENSOR_COUNT_V19",
            "FACTUAL_SUCCESSOR_EXCLUDED_PREFIX_V19",
            "FACTUAL_SUCCESSOR_GRADIENT_NORM_CAP_V19",
            "FACTUAL_SUCCESSOR_PREDICTOR_PARAMETER_COUNT_V19",
            "FACTUAL_SUCCESSOR_PREDICTOR_PARAMETER_TENSOR_COUNT_V19",
            "FactualSuccessorPredictorSubsetV19",
            "HOLD_ACTION_INDEX_V19",
            "JointTrainingAccountingV19",
            "JointUpdateResultV19",
            "build_frozen_optimizer_v19",
            "factual_successor_predictor_subset_v19",
            "joint_training_update_v19",
            "partition_parameters_v19",
            "private_training_adapter_receipt_v19",
            "validate_accounting_v19",
            "validate_optimizer_v19",
        )
    )
)
