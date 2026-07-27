"""Geometry-anchored two-mode event-delta joint JEPA V1.

The RGB encoder, geometry-anchored BEV lift, semantic head, and EMA target
are the frozen V3 representation.  The predictor is the sole model change:
it emits one learned event-delta mean and one learned event-occurrence logit;
the other mode is the parameter-free, exact zero delta.
"""
from __future__ import annotations

import copy
from typing import Mapping, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.geometry_anchored_deformable_bev_lift_joint_jepa_v1 import (
    ACTION_VOCABULARY_V1,
    FREE_CLASS_V1,
    OCCUPIED_CLASS_V1,
    UNKNOWN_CLASS_V1,
    GeometryAnchoredBevSamplingV1,
    GeometryAnchoredDeformableBevLiftJointJepaV1 as _FrozenRepresentationJointJepaV1,
    GeometryAnchoredDeformableBevLiftJointJepaV1Config,
    GeometryAnchoredDeformableBevLiftV1,
    _construct_n320_encoder_without_rng_draw,
    _LocalResidualBlockV1,
    _validate_action_one_hot,
    _validate_n320_encoder_state,
    final_class_macro_nll_per_row,
    latent_energy_per_row,
)


ACTION_EMBEDDING_DIM_V1 = 16
LATENT_LAYER_NORM_EPSILON_V1 = 1e-5
EVENT_PREDICTOR_PARAMETER_COUNT_V1 = 231_505
EVENT_PREDICTOR_PARAMETER_TENSOR_COUNT_V1 = 15


class EventDeltaPrediction(NamedTuple):
    """The fixed-identity two-mode predictor's trainable outputs."""

    mu_event: torch.Tensor
    event_logit: torch.Tensor


class EventDeltaCellEnergies(NamedTuple):
    """Unmixed cell energy for ZERO_EVENT and LEARNED_EVENT."""

    zero_event: torch.Tensor
    learned_event: torch.Tensor


class ChangedStaticBalancedEnergy(NamedTuple):
    """Per-row soft changed/static reductions and their equal-weight mean."""

    changed: torch.Tensor
    static: torch.Tensor
    balanced: torch.Tensor


def _validate_latent_nchw(value: torch.Tensor, *, name: str) -> None:
    if value.ndim != 4 or tuple(value.shape[1:]) != (64, 64, 64):
        raise ValueError(f"{name} must have shape (B,64,64,64)")
    if value.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one row")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must use exact float32")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} is nonfinite")


def normalize_latent_per_cell_v1(latent: torch.Tensor) -> torch.Tensor:
    """Apply affine-free channel LayerNorm independently at every BEV cell."""

    _validate_latent_nchw(latent, name="latent")
    moved = latent.movedim(1, -1)
    normalized = F.layer_norm(
        moved,
        (64,),
        weight=None,
        bias=None,
        eps=LATENT_LAYER_NORM_EPSILON_V1,
    ).movedim(-1, 1)
    if normalized.dtype != torch.float32 or not bool(torch.isfinite(normalized).all()):
        raise FloatingPointError("normalized latent is nonfinite or not float32")
    return normalized


def _validate_prediction(
    target_delta: torch.Tensor,
    prediction: EventDeltaPrediction,
) -> bool:
    """Validate a one-action or exact-nine-action prediction.

    Returns ``True`` for an all-action prediction and ``False`` for one action.
    """

    _validate_latent_nchw(target_delta, name="target_delta")
    if not isinstance(prediction, EventDeltaPrediction):
        raise TypeError("prediction must be EventDeltaPrediction")
    mu_event, event_logit = prediction
    if mu_event.dtype != torch.float32 or event_logit.dtype != torch.float32:
        raise TypeError("event outputs must use exact float32")
    if mu_event.device != target_delta.device or event_logit.device != target_delta.device:
        raise TypeError("event outputs and target_delta must share a device")
    if not bool(torch.isfinite(mu_event).all()) or not bool(
        torch.isfinite(event_logit).all()
    ):
        raise FloatingPointError("event output is nonfinite")
    batch = target_delta.shape[0]
    if mu_event.ndim == 4:
        if tuple(mu_event.shape) != tuple(target_delta.shape):
            raise ValueError("one-action mu_event must match target_delta")
        if tuple(event_logit.shape) != (batch, 1, 64, 64):
            raise ValueError("one-action event_logit must have shape (B,1,64,64)")
        return False
    if mu_event.ndim == 5:
        if tuple(mu_event.shape) != (batch, 9, 64, 64, 64):
            raise ValueError("all-action mu_event must have shape (B,9,64,64,64)")
        if tuple(event_logit.shape) != (batch, 9, 1, 64, 64):
            raise ValueError(
                "all-action event_logit must have shape (B,9,1,64,64)"
            )
        return True
    raise ValueError("mu_event must be one-action or exact-nine-action")


def event_delta_cell_energies_v1(
    target_delta: torch.Tensor,
    prediction: EventDeltaPrediction,
) -> EventDeltaCellEnergies:
    """Return exact beta-one Smooth-L1 component energies per cell."""

    all_actions = _validate_prediction(target_delta, prediction)
    zero = F.smooth_l1_loss(
        target_delta,
        torch.zeros_like(target_delta),
        beta=1.0,
        reduction="none",
    ).mean(dim=1)
    if all_actions:
        target = target_delta[:, None].expand_as(prediction.mu_event)
        learned = F.smooth_l1_loss(
            target,
            prediction.mu_event,
            beta=1.0,
            reduction="none",
        ).mean(dim=2)
        zero = zero[:, None].expand(-1, 9, -1, -1)
    else:
        learned = F.smooth_l1_loss(
            target_delta,
            prediction.mu_event,
            beta=1.0,
            reduction="none",
        ).mean(dim=1)
    if (
        zero.shape != learned.shape
        or not bool(torch.isfinite(zero).all())
        or not bool(torch.isfinite(learned).all())
        or bool((zero < 0.0).any())
        or bool((learned < 0.0).any())
    ):
        raise FloatingPointError("event component cell energy is invalid")
    return EventDeltaCellEnergies(zero_event=zero, learned_event=learned)


def _validate_cell_energy_pair(
    zero_event: torch.Tensor,
    learned_event: torch.Tensor,
    event_logit: torch.Tensor,
) -> torch.Tensor:
    if zero_event.shape != learned_event.shape:
        raise ValueError("ZERO_EVENT and LEARNED_EVENT energies must have equal shape")
    if zero_event.ndim == 3:
        expected_energy = (zero_event.shape[0], 64, 64)
        expected_logit = (zero_event.shape[0], 1, 64, 64)
        if tuple(zero_event.shape) != expected_energy:
            raise ValueError("one-action cell energy must have shape (B,64,64)")
        if tuple(event_logit.shape) != expected_logit:
            raise ValueError("one-action event_logit singleton axis changed")
        squeezed = event_logit.squeeze(1)
    elif zero_event.ndim == 4:
        expected_energy = (zero_event.shape[0], 9, 64, 64)
        expected_logit = (zero_event.shape[0], 9, 1, 64, 64)
        if tuple(zero_event.shape) != expected_energy:
            raise ValueError("all-action cell energy must have shape (B,9,64,64)")
        if tuple(event_logit.shape) != expected_logit:
            raise ValueError("all-action event_logit singleton axis changed")
        squeezed = event_logit.squeeze(2)
    else:
        raise ValueError("cell energy must be one-action or exact-nine-action")
    if squeezed.shape != zero_event.shape:
        raise ValueError("event_logit squeeze did not exactly match cell energy")
    if (
        zero_event.dtype != torch.float32
        or learned_event.dtype != torch.float32
        or event_logit.dtype != torch.float32
    ):
        raise TypeError("event energies and logits must use exact float32")
    if (
        zero_event.device != learned_event.device
        or zero_event.device != event_logit.device
    ):
        raise TypeError("event energies and logits must share a device")
    if (
        not bool(torch.isfinite(zero_event).all())
        or not bool(torch.isfinite(learned_event).all())
        or not bool(torch.isfinite(event_logit).all())
        or bool((zero_event < 0.0).any())
        or bool((learned_event < 0.0).any())
    ):
        raise FloatingPointError("event energies or logits are invalid")
    return squeezed


def _temperature_like(reference: torch.Tensor, temperature: float | torch.Tensor) -> torch.Tensor:
    if isinstance(temperature, torch.Tensor):
        if temperature.ndim != 0:
            raise ValueError("temperature must be scalar")
        if temperature.dtype != torch.float32 or temperature.device != reference.device:
            raise TypeError("tensor temperature must match float32 energy device")
        if temperature.requires_grad:
            raise ValueError("temperature must be frozen")
        result = temperature
    else:
        if isinstance(temperature, bool):
            raise TypeError("temperature must be a positive scalar")
        result = reference.new_tensor(float(temperature))
    if not bool(torch.isfinite(result)) or not bool(result > 0.0):
        raise FloatingPointError("temperature must be finite and strictly positive")
    return result


def two_mode_event_energy_v1(
    zero_event: torch.Tensor,
    learned_event: torch.Tensor,
    event_logit: torch.Tensor,
    temperature: float | torch.Tensor,
) -> torch.Tensor:
    """Mix fixed ZERO_EVENT and LEARNED_EVENT energies stably in log space."""

    ell = _validate_cell_energy_pair(zero_event, learned_event, event_logit)
    frozen_temperature = _temperature_like(zero_event, temperature)
    log_p0 = F.logsigmoid(-ell)
    log_p1 = F.logsigmoid(ell)
    mixed = -frozen_temperature * torch.logaddexp(
        log_p0 - zero_event / frozen_temperature,
        log_p1 - learned_event / frozen_temperature,
    )
    if mixed.shape != zero_event.shape or not bool(torch.isfinite(mixed).all()):
        raise FloatingPointError("two-mode cell energy is invalid")
    return mixed


def event_posterior_responsibility_v1(
    zero_event: torch.Tensor,
    learned_event: torch.Tensor,
    event_logit: torch.Tensor,
    temperature: float | torch.Tensor,
) -> torch.Tensor:
    """Return the analytic target-posterior LEARNED_EVENT responsibility."""

    ell = _validate_cell_energy_pair(zero_event, learned_event, event_logit)
    frozen_temperature = _temperature_like(zero_event, temperature)
    log_p0 = F.logsigmoid(-ell)
    log_p1 = F.logsigmoid(ell)
    learned_log_joint = log_p1 - learned_event / frozen_temperature
    zero_log_joint = log_p0 - zero_event / frozen_temperature
    log_odds = learned_log_joint - zero_log_joint
    responsibility = torch.sigmoid(log_odds)
    if not bool(torch.isfinite(responsibility).all()):
        raise FloatingPointError("event responsibility is nonfinite")
    return responsibility


def event_prior_probability_v1(prediction: EventDeltaPrediction) -> torch.Tensor:
    """Return the learned LEARNED_EVENT prior after exact singleton removal."""

    mu_event = prediction.mu_event
    if mu_event.ndim == 4:
        dummy = mu_event.new_zeros((mu_event.shape[0], 64, 64))
    elif mu_event.ndim == 5:
        dummy = mu_event.new_zeros((mu_event.shape[0], 9, 64, 64))
    else:
        raise ValueError("mu_event must be one-action or exact-nine-action")
    ell = _validate_cell_energy_pair(dummy, dummy, prediction.event_logit)
    probability = torch.sigmoid(ell)
    if not bool(torch.isfinite(probability).all()):
        raise FloatingPointError("event prior probability is nonfinite")
    return probability


def matched_single_mean_delta_v1(
    prediction: EventDeltaPrediction,
) -> torch.Tensor:
    """Collapse the two modes to the prior-matched deterministic mean."""

    probability = event_prior_probability_v1(prediction)
    if prediction.mu_event.ndim == 4:
        return probability[:, None] * prediction.mu_event
    return probability[:, :, None] * prediction.mu_event


def matched_single_mean_cell_energy_v1(
    target_delta: torch.Tensor,
    prediction: EventDeltaPrediction,
) -> torch.Tensor:
    """Cell Smooth-L1 energy of the exact prior-matched single mean."""

    all_actions = _validate_prediction(target_delta, prediction)
    matched = matched_single_mean_delta_v1(prediction)
    if all_actions:
        target = target_delta[:, None].expand_as(matched)
        energy = F.smooth_l1_loss(
            target, matched, beta=1.0, reduction="none"
        ).mean(dim=2)
    else:
        energy = F.smooth_l1_loss(
            target_delta, matched, beta=1.0, reduction="none"
        ).mean(dim=1)
    if not bool(torch.isfinite(energy).all()) or bool((energy < 0.0).any()):
        raise FloatingPointError("matched-single cell energy is invalid")
    return energy


def change_weight_v1(
    persistence_cell_energy: torch.Tensor,
    temperature: float | torch.Tensor,
) -> torch.Tensor:
    """Return the frozen-temperature soft change weight ``e/(e+T400)``."""

    if (
        persistence_cell_energy.ndim != 3
        or tuple(persistence_cell_energy.shape[1:]) != (64, 64)
        or persistence_cell_energy.shape[0] < 1
    ):
        raise ValueError("persistence_cell_energy must have shape (B,64,64)")
    if persistence_cell_energy.dtype != torch.float32:
        raise TypeError("persistence_cell_energy must use exact float32")
    if not bool(torch.isfinite(persistence_cell_energy).all()) or bool(
        (persistence_cell_energy < 0.0).any()
    ):
        raise FloatingPointError("persistence cell energy is invalid")
    frozen_temperature = _temperature_like(persistence_cell_energy, temperature)
    weight = persistence_cell_energy / (
        persistence_cell_energy + frozen_temperature
    )
    if not bool(torch.isfinite(weight).all()) or bool(
        ((weight < 0.0) | (weight >= 1.0)).any()
    ):
        raise FloatingPointError("change weight is invalid")
    return weight


def changed_static_balanced_energy_per_row_v1(
    cell_energy: torch.Tensor,
    change_weight: torch.Tensor,
) -> ChangedStaticBalancedEnergy:
    """Reduce one- or all-action cell values with exact soft balancing."""

    if (
        change_weight.ndim != 3
        or tuple(change_weight.shape[1:]) != (64, 64)
        or change_weight.shape[0] < 1
    ):
        raise ValueError("change_weight must have shape (B,64,64)")
    if cell_energy.ndim == 3:
        expected = tuple(change_weight.shape)
        weight = change_weight
    elif cell_energy.ndim == 4:
        expected = (change_weight.shape[0], 9, 64, 64)
        weight = change_weight[:, None]
    else:
        raise ValueError("cell_energy must be one-action or exact-nine-action")
    if tuple(cell_energy.shape) != expected:
        raise ValueError("cell_energy and change_weight shapes are incompatible")
    if cell_energy.dtype != torch.float32 or change_weight.dtype != torch.float32:
        raise TypeError("cell_energy and change_weight must use exact float32")
    if cell_energy.device != change_weight.device:
        raise TypeError("cell_energy and change_weight must share a device")
    if not bool(torch.isfinite(cell_energy).all()) or not bool(
        torch.isfinite(change_weight).all()
    ):
        raise FloatingPointError("balanced-reduction input is nonfinite")
    if bool(((change_weight < 0.0) | (change_weight >= 1.0)).any()):
        raise ValueError("change_weight must lie in [0,1)")
    changed_denominator = change_weight.sum(dim=(-2, -1))
    static_denominator = (1.0 - change_weight).sum(dim=(-2, -1))
    if bool((changed_denominator <= 1e-6).any()) or bool(
        (static_denominator <= 1e-6).any()
    ):
        raise FloatingPointError("changed/static denominator is not above 1e-6")
    changed_numerator = (weight * cell_energy).sum(dim=(-2, -1))
    static_numerator = ((1.0 - weight) * cell_energy).sum(dim=(-2, -1))
    if cell_energy.ndim == 4:
        changed_denominator = changed_denominator[:, None]
        static_denominator = static_denominator[:, None]
    changed = changed_numerator / changed_denominator
    static = static_numerator / static_denominator
    balanced = 0.5 * changed + 0.5 * static
    if not all(
        bool(torch.isfinite(value).all()) for value in (changed, static, balanced)
    ):
        raise FloatingPointError("changed/static/balanced reduction is nonfinite")
    return ChangedStaticBalancedEnergy(changed, static, balanced)


class TwoModeEventDeltaPredictorV1(nn.Module):
    """Shared local trunk with exact ZERO_EVENT and one learned event mode."""

    def __init__(
        self, config: GeometryAnchoredDeformableBevLiftJointJepaV1Config
    ) -> None:
        super().__init__()
        self.config = config
        self.action_embedding = nn.Embedding(config.action_dim, ACTION_EMBEDDING_DIM_V1)
        nn.init.zeros_(self.action_embedding.weight)
        self.input_projection = nn.Conv2d(
            config.bev_dim + ACTION_EMBEDDING_DIM_V1,
            config.bev_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.residual_blocks = nn.ModuleList(
            [_LocalResidualBlockV1(config.bev_dim) for _ in range(2)]
        )
        self.event_mean_head = nn.Conv2d(
            config.bev_dim,
            config.bev_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        nn.init.normal_(self.event_mean_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.event_mean_head.bias)
        self.event_logit_head = nn.Conv2d(
            config.bev_dim,
            1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        nn.init.zeros_(self.event_logit_head.weight)
        nn.init.zeros_(self.event_logit_head.bias)

    def forward(
        self,
        normalized_current_latent: torch.Tensor,
        action_one_hot: torch.Tensor,
    ) -> EventDeltaPrediction:
        _validate_latent_nchw(
            normalized_current_latent, name="normalized_current_latent"
        )
        if normalized_current_latent.device != self.action_embedding.weight.device:
            raise TypeError("normalized_current_latent and predictor must share a device")
        action_indices = _validate_action_one_hot(
            action_one_hot,
            batch=normalized_current_latent.shape[0],
            reference=normalized_current_latent,
        )
        action = self.action_embedding(action_indices)
        action = action[:, :, None, None].expand(
            -1, -1, *self.config.bev_size
        )
        value = F.gelu(
            self.input_projection(
                torch.cat((normalized_current_latent, action), dim=1)
            )
        )
        for block in self.residual_blocks:
            value = block(value)
        prediction = EventDeltaPrediction(
            mu_event=self.event_mean_head(value),
            event_logit=self.event_logit_head(value),
        )
        if not bool(torch.isfinite(prediction.mu_event).all()) or not bool(
            torch.isfinite(prediction.event_logit).all()
        ):
            raise FloatingPointError("event-delta predictor output is nonfinite")
        return prediction


class GeometryAnchoredTwoModeEventDeltaJointJepaV1(
    _FrozenRepresentationJointJepaV1
):
    """Frozen geometry-grounded representation with the event-delta predictor."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor],
        config: GeometryAnchoredDeformableBevLiftJointJepaV1Config | None = None,
    ) -> None:
        # Construct in the exact predecessor order without first creating and
        # discarding its closed predictor.
        nn.Module.__init__(self)
        self.config = config or GeometryAnchoredDeformableBevLiftJointJepaV1Config()
        self.encoder = _construct_n320_encoder_without_rng_draw(self.config)
        _validate_n320_encoder_state(self.encoder, n320_encoder_state_dict)
        self.encoder.load_state_dict(n320_encoder_state_dict, strict=True)

        caller_rng = torch.random.get_rng_state().clone()
        try:
            torch.random.default_generator.manual_seed(self.config.initialization_seed)
            self.bev_lift = GeometryAnchoredDeformableBevLiftV1(self.config)
            self.semantic_head = nn.Conv2d(
                self.config.bev_dim,
                self.config.state_classes,
                kernel_size=1,
                bias=True,
            )
            self.predictor = TwoModeEventDeltaPredictorV1(self.config)
        finally:
            torch.random.set_rng_state(caller_rng)

        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_bev_lift = copy.deepcopy(self.bev_lift)
        self.register_buffer(
            "target_hard_sync_count", torch.zeros((), dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "ema_update_count", torch.zeros((), dtype=torch.long), persistent=True
        )
        self.hard_sync_target_from_online()

    def predict(
        self,
        normalized_current_latent: torch.Tensor,
        action_one_hot: torch.Tensor,
    ) -> EventDeltaPrediction:
        return self.predictor(normalized_current_latent, action_one_hot)

    def predict_event_delta(
        self,
        normalized_current_latent: torch.Tensor,
        action_one_hot: torch.Tensor,
    ) -> EventDeltaPrediction:
        return self.predict(normalized_current_latent, action_one_hot)

    def predict_event(
        self,
        normalized_current_latent: torch.Tensor,
        action_one_hot: torch.Tensor,
    ) -> EventDeltaPrediction:
        return self.predict(normalized_current_latent, action_one_hot)

    def predict_all_actions(
        self, normalized_current_latent: torch.Tensor
    ) -> EventDeltaPrediction:
        _validate_latent_nchw(
            normalized_current_latent, name="normalized_current_latent"
        )
        batch = normalized_current_latent.shape[0]
        repeated = normalized_current_latent[:, None].expand(
            -1, self.config.action_dim, -1, -1, -1
        ).reshape(batch * self.config.action_dim, 64, 64, 64)
        actions = torch.eye(
            self.config.action_dim,
            dtype=normalized_current_latent.dtype,
            device=normalized_current_latent.device,
        )[None].expand(batch, -1, -1).reshape(
            batch * self.config.action_dim, self.config.action_dim
        )
        prediction = self.predict(repeated, actions)
        return EventDeltaPrediction(
            mu_event=prediction.mu_event.reshape(batch, 9, 64, 64, 64),
            event_logit=prediction.event_logit.reshape(batch, 9, 1, 64, 64),
        )

    def predict_all_action_event_deltas(
        self, normalized_current_latent: torch.Tensor
    ) -> EventDeltaPrediction:
        return self.predict_all_actions(normalized_current_latent)


EventDeltaPredictionV1 = EventDeltaPrediction
EventDeltaCellEnergiesV1 = EventDeltaCellEnergies
ChangedStaticBalancedEnergyV1 = ChangedStaticBalancedEnergy
GeometryAnchoredTwoModeEventDeltaJointJepaV1Config = (
    GeometryAnchoredDeformableBevLiftJointJepaV1Config
)
# The frozen runner resolves this historical class name from the selected
# model module.  Here it intentionally denotes the registered replacement.
GeometryAnchoredDeformableBevLiftJointJepaV1 = (
    GeometryAnchoredTwoModeEventDeltaJointJepaV1
)

# Concise compatibility aliases for the source-bound runner.
normalize_latent_per_cell = normalize_latent_per_cell_v1
event_delta_cell_energies = event_delta_cell_energies_v1
two_mode_event_energy = two_mode_event_energy_v1
event_posterior_responsibility = event_posterior_responsibility_v1
event_prior_probability = event_prior_probability_v1
matched_single_mean_delta = matched_single_mean_delta_v1
matched_single_mean_cell_energy = matched_single_mean_cell_energy_v1
change_weight = change_weight_v1
changed_static_balanced_energy_per_row = (
    changed_static_balanced_energy_per_row_v1
)
final_class_macro_nll_per_row_v1 = final_class_macro_nll_per_row
latent_energy_per_row_v1 = latent_energy_per_row


__all__ = [
    "ACTION_EMBEDDING_DIM_V1",
    "ACTION_VOCABULARY_V1",
    "ChangedStaticBalancedEnergy",
    "ChangedStaticBalancedEnergyV1",
    "EVENT_PREDICTOR_PARAMETER_COUNT_V1",
    "EVENT_PREDICTOR_PARAMETER_TENSOR_COUNT_V1",
    "EventDeltaCellEnergies",
    "EventDeltaCellEnergiesV1",
    "EventDeltaPrediction",
    "EventDeltaPredictionV1",
    "FREE_CLASS_V1",
    "GeometryAnchoredBevSamplingV1",
    "GeometryAnchoredDeformableBevLiftJointJepaV1",
    "GeometryAnchoredTwoModeEventDeltaJointJepaV1",
    "GeometryAnchoredTwoModeEventDeltaJointJepaV1Config",
    "LATENT_LAYER_NORM_EPSILON_V1",
    "OCCUPIED_CLASS_V1",
    "TwoModeEventDeltaPredictorV1",
    "UNKNOWN_CLASS_V1",
    "change_weight",
    "change_weight_v1",
    "changed_static_balanced_energy_per_row",
    "changed_static_balanced_energy_per_row_v1",
    "event_delta_cell_energies",
    "event_delta_cell_energies_v1",
    "event_posterior_responsibility",
    "event_posterior_responsibility_v1",
    "event_prior_probability",
    "event_prior_probability_v1",
    "final_class_macro_nll_per_row",
    "final_class_macro_nll_per_row_v1",
    "latent_energy_per_row",
    "latent_energy_per_row_v1",
    "matched_single_mean_cell_energy",
    "matched_single_mean_cell_energy_v1",
    "matched_single_mean_delta",
    "matched_single_mean_delta_v1",
    "normalize_latent_per_cell",
    "normalize_latent_per_cell_v1",
    "two_mode_event_energy",
    "two_mode_event_energy_v1",
]
