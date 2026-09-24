"""Finite spatial-token memory for the V18 RGB joint-JEPA.

The model keeps three observed V18 token maps in an explicit newest-first
delay line, predicts four future maps recursively, and uses the inherited EMA
V18 encoder for stop-gradient targets.  The delay line itself is immutable
state: only the shared causal convolution and its ordered-action FiLM are
learned.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v18_object_space_height_volume import (
    PREDICTOR_PARAMETER_PREFIXES_V18,
    PROJECTION_INITIALIZATION_SEED_V13,
    REPRESENTATION_PARAMETER_PREFIXES_V18,
    SHARED_PARAMETER_PREFIXES_V18,
    GeometryAnchoredDeformableBevLiftJointJepaV1Config,
    V13TrainableParameterGroups,
)
from lewm.models.memory_role_factorized_joint_jepa_v1 import (
    NEW_PREFIXES_MEMORY_ROLE_FACTORIZED_V1,
    ParameterGroupV1,
)
from lewm.models.memory_role_spatial_contrastive_joint_jepa_v3 import (
    MemoryRoleSpatialContrastiveJointJepaV3,
)
from lewm.models.observable_camera_ray_evidence_v4 import (
    ObservableCameraRayEvidenceV4Model,
)


INITIALIZATION_SEED_V18_SPATIAL_TOKEN_DELAY_LINE_V1 = 20_260_733
MEMORY_PREDICTOR_PREFIX_V18_SPATIAL_TOKEN_DELAY_LINE_V1 = "memory_predictor."


@dataclass(frozen=True)
class V18SpatialTokenDelayLineCausalConvolutionJointJepaV1Config:
    """Immutable constants for the preregistered finite-memory mechanism."""

    source_shape: tuple[int, int, int] = (64, 64, 64)
    token_shape: tuple[int, int, int] = (64, 16, 16)
    pooling_kernel: int = 4
    history_slots: int = 4
    observed_history: int = 3
    rollout_horizon: int = 4
    action_dim: int = 9
    action_film_hidden_dim: int = 128
    mask_block_size: int = 4
    masked_current_loss_weight: float = 0.5
    normalization_epsilon: float = 1.0e-6
    initialization_seed: int = INITIALIZATION_SEED_V18_SPATIAL_TOKEN_DELAY_LINE_V1

    def __post_init__(self) -> None:
        expected = {
            "source_shape": (64, 64, 64),
            "token_shape": (64, 16, 16),
            "pooling_kernel": 4,
            "history_slots": 4,
            "observed_history": 3,
            "rollout_horizon": 4,
            "action_dim": 9,
            "action_film_hidden_dim": 128,
            "mask_block_size": 4,
            "masked_current_loss_weight": 0.5,
            "normalization_epsilon": 1.0e-6,
            "initialization_seed": (
                INITIALIZATION_SEED_V18_SPATIAL_TOKEN_DELAY_LINE_V1
            ),
        }
        changed = [
            name for name, value in expected.items() if getattr(self, name) != value
        ]
        if changed:
            raise ValueError(
                "V18 spatial-token delay-line constants cannot change: "
                + ", ".join(changed)
            )


class SpatialTokenDelayLineStateV1(NamedTuple):
    """Newest-first token, validity, and executed-action tapes."""

    tokens: torch.Tensor
    valid: torch.Tensor
    actions: torch.Tensor


class SpatialTokenDelayLineStepV1(NamedTuple):
    prediction: torch.Tensor
    state: SpatialTokenDelayLineStateV1


class V18SpatialTokenDelayLineOutputV1(NamedTuple):
    online_history_tokens: torch.Tensor
    target_future_tokens: torch.Tensor
    full_predictions: torch.Tensor
    masked_current_predictions: torch.Tensor
    newest_keep_mask: torch.Tensor
    initial_state: SpatialTokenDelayLineStateV1
    full_final_state: SpatialTokenDelayLineStateV1
    masked_final_state: SpatialTokenDelayLineStateV1
    full_loss: torch.Tensor
    masked_current_loss: torch.Tensor
    loss: torch.Tensor


class V18SpatialTokenDelayLineTrainableParameterGroupsV1(NamedTuple):
    inherited_v18: V13TrainableParameterGroups
    memory_predictor: ParameterGroupV1

    @property
    def online(self) -> ParameterGroupV1:
        return (
            self.inherited_v18.shared
            + self.inherited_v18.representation
            + self.inherited_v18.predictor
            + self.memory_predictor
        )


class SpatialTokenDelayLineCausalConvolutionPredictorV1(nn.Module):
    """One local causal reader shared across actions and rollout horizons."""

    def __init__(
        self,
        config: V18SpatialTokenDelayLineCausalConvolutionJointJepaV1Config,
    ) -> None:
        super().__init__()
        self.config = config
        token_channels = config.token_shape[0]
        convolution_channels = token_channels + 1

        self.age_embeddings = nn.Parameter(
            torch.zeros(config.history_slots, token_channels)
        )
        self.depthwise_causal = nn.Conv3d(
            convolution_channels,
            convolution_channels,
            kernel_size=(config.history_slots, 3, 3),
            padding=(0, 1, 1),
            groups=convolution_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            convolution_channels,
            token_channels,
            kernel_size=1,
            bias=True,
        )
        self.action_film = nn.Sequential(
            nn.Linear(
                config.history_slots * config.action_dim,
                config.action_film_hidden_dim,
            ),
            nn.GELU(approximate="none"),
            nn.Linear(config.action_film_hidden_dim, 2 * token_channels),
        )
        self._initialize_as_newest_persistence()

    def _initialize_as_newest_persistence(self) -> None:
        with torch.no_grad():
            self.age_embeddings.zero_()
            self.depthwise_causal.weight.zero_()
            self.depthwise_causal.weight[:, 0, 0, 1, 1] = 1.0
            self.pointwise.weight.zero_()
            self.pointwise.bias.zero_()
            token_channels = self.config.token_shape[0]
            identity_indices = torch.arange(token_channels)
            self.pointwise.weight[identity_indices, identity_indices, 0, 0] = 1.0
            final_film = self.action_film[2]
            final_film.weight.zero_()
            final_film.bias.zero_()

    def forward(
        self,
        token_planes: torch.Tensor,
        validity: torch.Tensor,
        action_tape: torch.Tensor,
    ) -> torch.Tensor:
        batch = token_planes.shape[0] if token_planes.ndim >= 1 else 0
        expected_tokens = (
            batch,
            self.config.history_slots,
            *self.config.token_shape,
        )
        if token_planes.ndim != 5 or tuple(token_planes.shape) != expected_tokens:
            raise ValueError(
                "token_planes must have shape "
                f"(B,{self.config.history_slots},64,16,16)"
            )
        if tuple(validity.shape) != (batch, self.config.history_slots):
            raise ValueError("validity must have shape (B,4)")
        if validity.dtype != torch.bool:
            raise TypeError("validity must use bool")
        if tuple(action_tape.shape) != (
            batch,
            self.config.history_slots,
            self.config.action_dim,
        ):
            raise ValueError("action_tape must have shape (B,4,9)")
        if (
            token_planes.device != self.depthwise_causal.weight.device
            or validity.device != token_planes.device
            or action_tape.device != token_planes.device
        ):
            raise TypeError("predictor inputs and module must share a device")

        age = self.age_embeddings.to(dtype=token_planes.dtype)
        augmented = token_planes + age[None, :, :, None, None]
        augmented = augmented * validity[:, :, None, None, None].to(
            dtype=token_planes.dtype
        )
        validity_planes = validity[:, :, None, None, None].expand(
            -1,
            -1,
            1,
            self.config.token_shape[1],
            self.config.token_shape[2],
        )
        causal_input = torch.cat(
            (augmented, validity_planes.to(dtype=token_planes.dtype)),
            dim=2,
        ).permute(0, 2, 1, 3, 4)

        collapsed = self.depthwise_causal(causal_input).squeeze(2)
        spatial = self.pointwise(collapsed)
        film = self.action_film(action_tape.flatten(start_dim=1))
        scale_delta, bias = film.chunk(2, dim=1)
        prediction = (
            spatial * (1.0 + scale_delta[:, :, None, None])
            + bias[:, :, None, None]
        )
        return F.normalize(
            prediction,
            dim=1,
            eps=self.config.normalization_epsilon,
        )


class V18SpatialTokenDelayLineCausalConvolutionJointJepaV1(
    MemoryRoleSpatialContrastiveJointJepaV3
):
    """V18 perception plus a finite causal spatial-token memory predictor."""

    def __init__(
        self,
        n320_fit_model: ObservableCameraRayEvidenceV4Model,
        sweep_masks: torch.Tensor,
        config: GeometryAnchoredDeformableBevLiftJointJepaV1Config | None = None,
        memory_config: (
            V18SpatialTokenDelayLineCausalConvolutionJointJepaV1Config | None
        ) = None,
    ) -> None:
        caller_rng = torch.random.get_rng_state().clone()
        try:
            super().__init__(n320_fit_model, sweep_masks, config)
            self.memory_config = (
                memory_config
                or V18SpatialTokenDelayLineCausalConvolutionJointJepaV1Config()
            )
            if (
                tuple((self.config.bev_dim, *self.config.bev_size))
                != self.memory_config.source_shape
                or self.config.action_dim != self.memory_config.action_dim
                or self.config.target_ema_momentum != 0.996
            ):
                raise RuntimeError("inherited V18 memory geometry or EMA changed")
            torch.random.default_generator.manual_seed(
                self.memory_config.initialization_seed
            )
            self.memory_predictor = (
                SpatialTokenDelayLineCausalConvolutionPredictorV1(
                    self.memory_config
                )
            )
        finally:
            torch.random.set_rng_state(caller_rng)

        self._freeze_role_diagnostics()
        self._freeze_target()
        self.trainable_parameter_groups_delay_line_v1()

    def _freeze_role_diagnostics(self) -> None:
        for name in ("role_factorizer", "place_predictor", "local_predictor"):
            if hasattr(self, name):
                module = getattr(self, name)
                module.requires_grad_(False)
                module.eval()

    def train(
        self, mode: bool = True
    ) -> V18SpatialTokenDelayLineCausalConvolutionJointJepaV1:
        super().train(mode)
        self._freeze_role_diagnostics()
        return self

    def trainable_parameter_groups_v18(self) -> V13TrainableParameterGroups:
        if not hasattr(self, "memory_predictor"):
            return super().trainable_parameter_groups_v18()

        excluded = (
            *NEW_PREFIXES_MEMORY_ROLE_FACTORIZED_V1,
            MEMORY_PREDICTOR_PREFIX_V18_SPATIAL_TOKEN_DELAY_LINE_V1,
        )
        named = tuple(
            (name, parameter)
            for name, parameter in self.named_parameters(remove_duplicate=False)
            if parameter.requires_grad and not name.startswith(excluded)
        )

        def select(prefixes: tuple[str, ...]) -> ParameterGroupV1:
            return tuple(item for item in named if item[0].startswith(prefixes))

        groups = V13TrainableParameterGroups(
            shared=select(SHARED_PARAMETER_PREFIXES_V18),
            representation=select(REPRESENTATION_PARAMETER_PREFIXES_V18),
            predictor=select(PREDICTOR_PARAMETER_PREFIXES_V18),
        )
        selected = tuple(item for group in groups for item in group)
        if (
            len({id(parameter) for _, parameter in selected}) != len(selected)
            or {name for name, _ in selected} != {name for name, _ in named}
        ):
            raise RuntimeError("inherited V18 parameter view is incomplete")
        return groups

    def trainable_parameter_groups_memory_role_factorized_v1(self):
        if not hasattr(self, "memory_predictor"):
            return super().trainable_parameter_groups_memory_role_factorized_v1()
        return self.trainable_parameter_groups_delay_line_v1()

    def trainable_parameter_groups_delay_line_v1(
        self,
    ) -> V18SpatialTokenDelayLineTrainableParameterGroupsV1:
        memory_predictor = tuple(
            (name, parameter)
            for name, parameter in self.named_parameters(remove_duplicate=False)
            if parameter.requires_grad
            and name.startswith(
                MEMORY_PREDICTOR_PREFIX_V18_SPATIAL_TOKEN_DELAY_LINE_V1
            )
        )
        groups = V18SpatialTokenDelayLineTrainableParameterGroupsV1(
            inherited_v18=self.trainable_parameter_groups_v18(),
            memory_predictor=memory_predictor,
        )
        selected = groups.online
        trainable = tuple(
            parameter for parameter in self.parameters() if parameter.requires_grad
        )
        if (
            not groups.memory_predictor
            or len({id(parameter) for _, parameter in selected}) != len(selected)
            or {id(parameter) for _, parameter in selected}
            != {id(parameter) for parameter in trainable}
        ):
            raise RuntimeError("delay-line online parameter view is incomplete")
        return groups

    def _validate_rgb_sequence(
        self,
        rgb_sequence: torch.Tensor,
        *,
        length: int,
        name: str,
    ) -> None:
        expected = (
            length,
            3,
            self.config.image_size,
            self.config.image_size,
        )
        if rgb_sequence.ndim != 5 or tuple(rgb_sequence.shape[1:]) != expected:
            raise ValueError(f"{name} must have shape (B,{length},3,112,112)")
        if rgb_sequence.shape[0] < 1:
            raise ValueError(f"{name} must contain at least one row")
        if rgb_sequence.dtype != torch.float32:
            raise TypeError(f"{name} must use exact float32")
        if rgb_sequence.device != self.memory_predictor.pointwise.weight.device:
            raise TypeError(f"{name} and model must share a device")
        if not bool(torch.isfinite(rgb_sequence).all()):
            raise FloatingPointError(f"{name} is nonfinite")

    def _pool_and_normalize_memory_tokens(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        if latent.ndim != 4 or tuple(latent.shape[1:]) != (
            self.memory_config.source_shape
        ):
            raise ValueError("V18 latent must have shape (B,64,64,64)")
        pooled = F.avg_pool2d(
            latent,
            kernel_size=self.memory_config.pooling_kernel,
            stride=self.memory_config.pooling_kernel,
        )
        if tuple(pooled.shape[1:]) != self.memory_config.token_shape:
            raise RuntimeError("fixed V18 token pooling shape changed")
        return F.normalize(
            pooled,
            dim=1,
            eps=self.memory_config.normalization_epsilon,
        )

    def encode_online_memory_sequence(
        self,
        rgb_sequence: torch.Tensor,
    ) -> torch.Tensor:
        length = rgb_sequence.shape[1] if rgb_sequence.ndim == 5 else -1
        if length < 1:
            raise ValueError("online_rgb_sequence must contain at least one endpoint")
        self._validate_rgb_sequence(
            rgb_sequence,
            length=length,
            name="online_rgb_sequence",
        )
        batch = rgb_sequence.shape[0]
        flattened = rgb_sequence.reshape(
            batch * length,
            3,
            self.config.image_size,
            self.config.image_size,
        )
        tokens = self._pool_and_normalize_memory_tokens(
            self.encode_online(flattened)
        )
        return tokens.reshape(batch, length, *self.memory_config.token_shape)

    @torch.no_grad()
    def encode_target_memory_sequence(
        self,
        rgb_sequence: torch.Tensor,
    ) -> torch.Tensor:
        length = rgb_sequence.shape[1] if rgb_sequence.ndim == 5 else -1
        if length < 1:
            raise ValueError("target_rgb_sequence must contain at least one endpoint")
        self._validate_rgb_sequence(
            rgb_sequence,
            length=length,
            name="target_rgb_sequence",
        )
        batch = rgb_sequence.shape[0]
        flattened = rgb_sequence.reshape(
            batch * length,
            3,
            self.config.image_size,
            self.config.image_size,
        )
        tokens = self._pool_and_normalize_memory_tokens(
            self.encode_target(flattened)
        )
        return tokens.reshape(
            batch,
            length,
            *self.memory_config.token_shape,
        ).detach()

    def _validate_action_sequence(
        self,
        action_sequence: torch.Tensor,
        *,
        batch: int,
        length: int,
        name: str,
    ) -> None:
        if tuple(action_sequence.shape) != (
            batch,
            length,
            self.memory_config.action_dim,
        ):
            raise ValueError(f"{name} must have shape (B,{length},9)")
        if action_sequence.dtype != torch.float32:
            raise TypeError(f"{name} must use exact float32")
        if (
            action_sequence.device
            != self.memory_predictor.pointwise.weight.device
        ):
            raise TypeError(f"{name} and model must share a device")
        if not bool(torch.isfinite(action_sequence).all()):
            raise FloatingPointError(f"{name} is nonfinite")
        if not bool(
            ((action_sequence == 0.0) | (action_sequence == 1.0)).all()
        ):
            raise ValueError(f"{name} must contain exact zeros and ones")
        if length:
            expected = torch.ones(
                (batch, length),
                dtype=action_sequence.dtype,
                device=action_sequence.device,
            )
            if not torch.equal(action_sequence.sum(dim=2), expected):
                raise ValueError(f"each {name} row must be one-hot")

    def _validate_state(self, state: SpatialTokenDelayLineStateV1) -> None:
        if not isinstance(state, SpatialTokenDelayLineStateV1):
            raise TypeError("state must be SpatialTokenDelayLineStateV1")
        if state.tokens.ndim != 5 or tuple(state.tokens.shape[1:]) != (
            self.memory_config.history_slots,
            *self.memory_config.token_shape,
        ):
            raise ValueError("state.tokens must have shape (B,4,64,16,16)")
        batch = state.tokens.shape[0]
        if batch < 1:
            raise ValueError("state must contain at least one row")
        if state.tokens.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            raise TypeError("state.tokens must use a supported floating dtype")
        if not bool(torch.isfinite(state.tokens).all()):
            raise FloatingPointError("state.tokens is nonfinite")
        if tuple(state.valid.shape) != (batch, self.memory_config.history_slots):
            raise ValueError("state.valid must have shape (B,4)")
        if state.valid.dtype != torch.bool:
            raise TypeError("state.valid must use bool")
        if tuple(state.actions.shape) != (
            batch,
            self.memory_config.history_slots,
            self.memory_config.action_dim,
        ):
            raise ValueError("state.actions must have shape (B,4,9)")
        if state.actions.dtype != torch.float32:
            raise TypeError("state.actions must use exact float32")
        reference_device = self.memory_predictor.pointwise.weight.device
        if (
            state.tokens.device != reference_device
            or state.valid.device != reference_device
            or state.actions.device != reference_device
        ):
            raise TypeError("state and model must share a device")
        if not bool(torch.isfinite(state.actions).all()):
            raise FloatingPointError("state.actions is nonfinite")
        if not bool(((state.actions == 0.0) | (state.actions == 1.0)).all()):
            raise ValueError("state.actions must contain exact zeros and ones")
        if bool((state.valid[:, 1:] & ~state.valid[:, :-1]).any()):
            raise ValueError("state.valid must be a contiguous newest-first prefix")
        invalid_tokens = state.tokens * (~state.valid)[:, :, None, None, None].to(
            dtype=state.tokens.dtype
        )
        if bool((invalid_tokens != 0).any()):
            raise ValueError("invalid state token slots must be zero")
        transition_valid = state.valid[:, :-1] & state.valid[:, 1:]
        action_counts = state.actions.sum(dim=2)
        if not torch.equal(
            action_counts[:, :-1],
            transition_valid.to(dtype=state.actions.dtype),
        ):
            raise ValueError("state action tape does not match valid transitions")
        oldest_action_count = action_counts[:, -1]
        if bool(
            (
                (oldest_action_count > state.valid[:, -1].to(state.actions.dtype))
                | (oldest_action_count < 0.0)
            ).any()
        ):
            raise ValueError("oldest state action must be empty or one-hot")

    def build_history_state(
        self,
        tokens: torch.Tensor,
        history_actions: torch.Tensor,
    ) -> SpatialTokenDelayLineStateV1:
        """Build newest-first state from chronological observed tokens/actions."""

        if tokens.ndim != 5 or tuple(tokens.shape[2:]) != (
            self.memory_config.token_shape
        ):
            raise ValueError("tokens must have shape (B,T,64,16,16)")
        batch, length = tokens.shape[:2]
        if batch < 1 or not 1 <= length <= self.memory_config.history_slots:
            raise ValueError("tokens must contain one to four chronological endpoints")
        if tokens.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("tokens must use a supported floating dtype")
        if tokens.device != self.memory_predictor.pointwise.weight.device:
            raise TypeError("tokens and model must share a device")
        if not bool(torch.isfinite(tokens).all()):
            raise FloatingPointError("tokens is nonfinite")
        self._validate_action_sequence(
            history_actions,
            batch=batch,
            length=length - 1,
            name="history_actions",
        )

        state_tokens = torch.zeros(
            (batch, self.memory_config.history_slots, *self.memory_config.token_shape),
            dtype=tokens.dtype,
            device=tokens.device,
        )
        state_valid = torch.zeros(
            (batch, self.memory_config.history_slots),
            dtype=torch.bool,
            device=tokens.device,
        )
        state_actions = torch.zeros(
            (
                batch,
                self.memory_config.history_slots,
                self.memory_config.action_dim,
            ),
            dtype=history_actions.dtype,
            device=history_actions.device,
        )
        chronological_indices = torch.arange(
            length - 1,
            -1,
            -1,
            device=tokens.device,
        )
        state_tokens[:, :length] = tokens.index_select(1, chronological_indices)
        state_valid[:, :length] = True
        if length > 1:
            action_indices = torch.arange(
                length - 2,
                -1,
                -1,
                device=history_actions.device,
            )
            state_actions[:, : length - 1] = history_actions.index_select(
                1,
                action_indices,
            )
        state = SpatialTokenDelayLineStateV1(
            state_tokens,
            state_valid,
            state_actions,
        )
        self._validate_state(state)
        return state

    def reset_history_state(
        self,
        state: SpatialTokenDelayLineStateV1,
        reset_mask: torch.Tensor | None = None,
    ) -> SpatialTokenDelayLineStateV1:
        """Keep the current token and clear older memory/actions on reset rows."""

        self._validate_state(state)
        batch = state.tokens.shape[0]
        if reset_mask is None:
            reset_mask = torch.ones(
                batch,
                dtype=torch.bool,
                device=state.tokens.device,
            )
        elif (
            tuple(reset_mask.shape) != (batch,)
            or reset_mask.dtype != torch.bool
            or reset_mask.device != state.tokens.device
        ):
            raise ValueError("reset_mask must be a same-device bool tensor of shape (B,)")

        reset_tokens = torch.zeros_like(state.tokens)
        reset_tokens[:, 0] = state.tokens[:, 0]
        reset_valid = torch.zeros_like(state.valid)
        reset_valid[:, 0] = state.valid[:, 0]
        reset_actions = torch.zeros_like(state.actions)
        row_mask = reset_mask[:, None, None, None, None]
        result = SpatialTokenDelayLineStateV1(
            torch.where(row_mask, reset_tokens, state.tokens),
            torch.where(reset_mask[:, None], reset_valid, state.valid),
            torch.where(reset_mask[:, None, None], reset_actions, state.actions),
        )
        self._validate_state(result)
        return result

    def predict_from_state(
        self,
        state: SpatialTokenDelayLineStateV1,
        action_one_hot: torch.Tensor,
    ) -> SpatialTokenDelayLineStepV1:
        """Predict one successor and insert it with its executed action."""

        self._validate_state(state)
        batch = state.tokens.shape[0]
        self._validate_action_sequence(
            action_one_hot[:, None, :]
            if action_one_hot.ndim == 2
            else action_one_hot,
            batch=batch,
            length=1,
            name="action_one_hot",
        )
        if action_one_hot.ndim != 2:
            raise ValueError("action_one_hot must have shape (B,9)")
        if not bool(state.valid[:, 0].all()):
            raise ValueError("every state row must contain a current token")

        action_tape = torch.cat(
            (action_one_hot[:, None, :], state.actions[:, :-1]),
            dim=1,
        )
        prediction = self.memory_predictor(
            state.tokens,
            state.valid,
            action_tape,
        )
        next_state = SpatialTokenDelayLineStateV1(
            torch.cat((prediction[:, None], state.tokens[:, :-1]), dim=1),
            torch.cat(
                (
                    torch.ones(
                        (batch, 1),
                        dtype=torch.bool,
                        device=state.valid.device,
                    ),
                    state.valid[:, :-1],
                ),
                dim=1,
            ),
            action_tape,
        )
        self._validate_state(next_state)
        return SpatialTokenDelayLineStepV1(prediction, next_state)

    def deterministic_newest_keep_mask(
        self,
        batch_size: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        side = self.memory_config.token_shape[1]
        block = self.memory_config.mask_block_size
        block_rows = torch.arange(side, device=device) // block
        keep = (
            block_rows[:, None] + block_rows[None, :]
        ).remainder(2).eq(1)
        return keep[None, None].expand(batch_size, 1, side, side)

    def _rollout_from_state(
        self,
        state: SpatialTokenDelayLineStateV1,
        future_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, SpatialTokenDelayLineStateV1]:
        predictions = []
        current_state = state
        for horizon in range(self.memory_config.rollout_horizon):
            step = self.predict_from_state(
                current_state,
                future_actions[:, horizon],
            )
            predictions.append(step.prediction)
            current_state = step.state
        return torch.stack(predictions, dim=1), current_state

    def forward_memory(
        self,
        history_rgb: torch.Tensor,
        action_sequence: torch.Tensor,
        future_rgb: torch.Tensor,
    ) -> V18SpatialTokenDelayLineOutputV1:
        """Run full and masked H4 JEPA branches from an observed H3 history."""

        self._validate_rgb_sequence(
            history_rgb,
            length=self.memory_config.observed_history,
            name="history_rgb",
        )
        self._validate_rgb_sequence(
            future_rgb,
            length=self.memory_config.rollout_horizon,
            name="future_rgb",
        )
        batch = history_rgb.shape[0]
        if future_rgb.shape[0] != batch:
            raise ValueError("history_rgb and future_rgb batch sizes differ")
        total_actions = (
            self.memory_config.observed_history
            - 1
            + self.memory_config.rollout_horizon
        )
        self._validate_action_sequence(
            action_sequence,
            batch=batch,
            length=total_actions,
            name="action_sequence",
        )

        online_history = self.encode_online_memory_sequence(history_rgb)
        initial_state = self.build_history_state(
            online_history,
            action_sequence[:, : self.memory_config.observed_history - 1],
        )
        future_actions = action_sequence[
            :,
            self.memory_config.observed_history - 1 :,
        ]
        full_predictions, full_final_state = self._rollout_from_state(
            initial_state,
            future_actions,
        )

        newest_keep_mask = self.deterministic_newest_keep_mask(
            batch,
            device=online_history.device,
        )
        masked_tokens = torch.cat(
            (
                initial_state.tokens[:, :1]
                * newest_keep_mask[:, None].to(dtype=initial_state.tokens.dtype),
                initial_state.tokens[:, 1:],
            ),
            dim=1,
        )
        masked_initial_state = SpatialTokenDelayLineStateV1(
            masked_tokens,
            initial_state.valid,
            initial_state.actions,
        )
        masked_predictions, masked_final_state = self._rollout_from_state(
            masked_initial_state,
            future_actions,
        )

        target_future = self.encode_target_memory_sequence(future_rgb).detach()
        full_loss = (
            1.0 - (full_predictions * target_future).sum(dim=2)
        ).mean()
        masked_current_loss = (
            1.0 - (masked_predictions * target_future).sum(dim=2)
        ).mean()
        loss = (
            full_loss
            + self.memory_config.masked_current_loss_weight * masked_current_loss
        )
        return V18SpatialTokenDelayLineOutputV1(
            online_history_tokens=online_history,
            target_future_tokens=target_future,
            full_predictions=full_predictions,
            masked_current_predictions=masked_predictions,
            newest_keep_mask=newest_keep_mask,
            initial_state=initial_state,
            full_final_state=full_final_state,
            masked_final_state=masked_final_state,
            full_loss=full_loss,
            masked_current_loss=masked_current_loss,
            loss=loss,
        )


__all__ = [
    "INITIALIZATION_SEED_V18_SPATIAL_TOKEN_DELAY_LINE_V1",
    "MEMORY_PREDICTOR_PREFIX_V18_SPATIAL_TOKEN_DELAY_LINE_V1",
    "PROJECTION_INITIALIZATION_SEED_V13",
    "SpatialTokenDelayLineCausalConvolutionPredictorV1",
    "SpatialTokenDelayLineStateV1",
    "SpatialTokenDelayLineStepV1",
    "V18SpatialTokenDelayLineCausalConvolutionJointJepaV1",
    "V18SpatialTokenDelayLineCausalConvolutionJointJepaV1Config",
    "V18SpatialTokenDelayLineOutputV1",
    "V18SpatialTokenDelayLineTrainableParameterGroupsV1",
]
