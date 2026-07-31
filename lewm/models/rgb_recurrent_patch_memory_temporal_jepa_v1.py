"""Action-conditioned recurrent patch-memory temporal JEPA V1.

This model warm-starts the complete trainable spatial path from the frozen
single-frame masked-spatial JEPA V1, then learns one shared recurrent update
over each of its 256 globally contextualized patch streams.  The sole output
is a prediction of registered tokens from the next RGB frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.rgb_single_frame_multiblock_masked_spatial_jepa_v1 import (
    MaskedSpatialTargetV1,
    SingleFrameMultiblockMaskedSpatialJepaV1,
    SingleFrameMultiblockMaskedSpatialJepaV1Config,
    _gather_spatial_tokens,
    normalized_half_squared_jepa_loss_v1,
)


_MIGRATED_PREFIXES_V1 = (
    "encoder.",
    "predictor_blocks.",
    "predictor_norm.",
    "predictor_output.",
)
_MIGRATED_EXACT_KEYS_V1 = (
    "predictor_position",
    "predictor_mask_token",
)


def temporal_v1_accepts_predecessor_key(name: str) -> bool:
    """Return whether a spatial-V1 state entry crosses the warm-start boundary."""

    return name in _MIGRATED_EXACT_KEYS_V1 or name.startswith(
        _MIGRATED_PREFIXES_V1
    )


@dataclass(frozen=True)
class RGBRecurrentPatchMemoryTemporalJepaV1Config(
    SingleFrameMultiblockMaskedSpatialJepaV1Config
):
    """Frozen architecture constants for the temporal V1 model."""

    context_length: int = 3
    action_count: int = 9
    time_embedding_count: int = 3
    temporal_hidden_dim: int = 192
    decoder_token_count: int = 320
    temporal_initialization_seed: int = 20260731

    def __post_init__(self) -> None:
        super().__post_init__()
        expected = {
            "context_length": 3,
            "action_count": 9,
            "time_embedding_count": 3,
            "temporal_hidden_dim": 192,
            "decoder_token_count": 320,
            "temporal_initialization_seed": 20260731,
        }
        changed = [
            name
            for name, value in expected.items()
            if getattr(self, name) != value
        ]
        if changed:
            raise ValueError(
                "recurrent patch-memory temporal JEPA V1 constants cannot "
                "change: " + ", ".join(changed)
            )


class TemporalPatchMemoryPredictionV1(NamedTuple):
    """Online prediction and the recurrent evidence used to produce it."""

    raw_predicted_target_tokens: torch.Tensor
    normalized_predicted_target_tokens: torch.Tensor
    target_indices: torch.Tensor
    encoded_history: torch.Tensor
    time_indices: torch.Tensor
    recurrent_step_states: torch.Tensor
    recurrent_memory: torch.Tensor
    predictor_input: torch.Tensor | None
    predictor_block_outputs: tuple[torch.Tensor, ...]


class TemporalPatchMemoryJepaOutputV1(NamedTuple):
    prediction: TemporalPatchMemoryPredictionV1
    target: MaskedSpatialTargetV1
    loss: torch.Tensor


class RGBRecurrentPatchMemoryTemporalJepaV1(
    SingleFrameMultiblockMaskedSpatialJepaV1
):
    """Jointly trainable RGB encoder, recurrent memory, and future predictor."""

    def __init__(
        self,
        predecessor_model_state_dict: Mapping[str, torch.Tensor],
        config: RGBRecurrentPatchMemoryTemporalJepaV1Config | None = None,
    ) -> None:
        if not isinstance(predecessor_model_state_dict, Mapping):
            raise TypeError("predecessor_model_state_dict must be a mapping")
        if any(not isinstance(name, str) for name in predecessor_model_state_dict):
            raise TypeError("predecessor state keys must be strings")

        temporal_config = config or RGBRecurrentPatchMemoryTemporalJepaV1Config()
        encoder_prefix = "encoder."
        encoder_state = {
            name[len(encoder_prefix) :]: value
            for name, value in predecessor_model_state_dict.items()
            if name.startswith(encoder_prefix)
        }
        super().__init__(encoder_state, temporal_config)

        predecessor_inventory = self.state_dict()
        predecessor_keys = set(predecessor_model_state_dict)
        expected_keys = set(predecessor_inventory)
        if predecessor_keys != expected_keys:
            missing = sorted(expected_keys - predecessor_keys)
            extra = sorted(predecessor_keys - expected_keys)
            raise ValueError(
                "spatial-V1 predecessor state inventory changed; "
                f"missing={missing}, extra={extra}"
            )

        accepted_expected = {
            name for name in expected_keys if temporal_v1_accepts_predecessor_key(name)
        }
        accepted_observed = {
            name
            for name in predecessor_keys
            if temporal_v1_accepts_predecessor_key(name)
        }
        if accepted_observed != accepted_expected:
            missing = sorted(accepted_expected - accepted_observed)
            extra = sorted(accepted_observed - accepted_expected)
            raise ValueError(
                "accepted spatial-V1 migration inventory changed; "
                f"missing={missing}, extra={extra}"
            )

        with torch.no_grad():
            for name in sorted(accepted_expected):
                value = predecessor_model_state_dict[name]
                expected = predecessor_inventory[name]
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"predecessor state {name!r} is not a tensor")
                if value.shape != expected.shape:
                    raise ValueError(
                        f"predecessor state {name!r} has shape "
                        f"{tuple(value.shape)}, expected {tuple(expected.shape)}"
                    )
                if value.dtype != expected.dtype:
                    raise TypeError(
                        f"predecessor state {name!r} has dtype {value.dtype}, "
                        f"expected {expected.dtype}"
                    )
                if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                    raise FloatingPointError(
                        f"predecessor state {name!r} contains a nonfinite value"
                    )
                expected.copy_(value)

        # Predecessor target lag and accounting never cross the migration
        # boundary.  The temporal target begins as an exact online copy.
        self.hard_sync_target_from_online()

        caller_rng = torch.random.get_rng_state()
        try:
            torch.random.manual_seed(temporal_config.temporal_initialization_seed)
            self.action_embedding = nn.Embedding(
                temporal_config.action_count,
                temporal_config.feature_dim,
            )
            self.time_embedding = nn.Embedding(
                temporal_config.time_embedding_count,
                temporal_config.feature_dim,
            )
            self.temporal_gru = nn.GRU(
                input_size=temporal_config.feature_dim,
                hidden_size=temporal_config.temporal_hidden_dim,
                num_layers=1,
                batch_first=True,
            )
        finally:
            torch.random.set_rng_state(caller_rng)

    def _validate_context_rgb(self, context_rgb: torch.Tensor) -> int:
        expected = (
            self.config.context_length,
            3,
            self.config.image_size,
            self.config.image_size,
        )
        if context_rgb.ndim != 5 or tuple(context_rgb.shape[1:]) != expected:
            raise ValueError(
                "context_rgb must have shape "
                f"(B,{expected[0]},{expected[1]},{expected[2]},{expected[3]})"
            )
        if context_rgb.shape[0] < 1:
            raise ValueError("context_rgb must contain at least one row")
        if context_rgb.dtype != torch.float32:
            raise TypeError("context_rgb must use exact float32")
        if not bool(torch.isfinite(context_rgb).all()):
            raise FloatingPointError("context_rgb contains a nonfinite value")
        if context_rgb.device != self.predictor_mask_token.device:
            raise TypeError("context_rgb and model must share a device")
        return context_rgb.shape[0]

    def _validate_actions(
        self,
        actions: torch.Tensor,
        *,
        batch: int,
        steps: int,
    ) -> None:
        expected = (batch, steps)
        if actions.shape != expected or actions.dtype != torch.long:
            raise TypeError(f"actions must be long with shape {expected}")
        if actions.device != self.predictor_mask_token.device:
            raise TypeError("actions and model must share a device")
        if bool((actions < 0).any()) or bool(
            (actions >= self.config.action_count).any()
        ):
            raise ValueError("actions must be in the closed range [0,8]")

    def _validate_encoded_history(self, encoded_history: torch.Tensor) -> tuple[int, int]:
        if encoded_history.ndim != 4:
            raise ValueError("encoded_history must have shape (B,S,256,192)")
        batch, steps, tokens, features = encoded_history.shape
        if (
            batch < 1
            or steps < 1
            or steps > self.config.context_length
            or tokens != self.config.spatial_token_count
            or features != self.config.feature_dim
        ):
            raise ValueError("encoded_history must have shape (B,S,256,192), 1<=S<=3")
        if encoded_history.dtype != torch.float32:
            raise TypeError("encoded_history must use exact float32")
        if encoded_history.device != self.predictor_mask_token.device:
            raise TypeError("encoded_history and model must share a device")
        if not bool(torch.isfinite(encoded_history).all()):
            raise FloatingPointError("encoded_history contains a nonfinite value")
        return batch, steps

    def _validated_time_indices(
        self,
        time_indices: torch.Tensor | None,
        *,
        batch: int,
        steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        if time_indices is None:
            result = torch.arange(steps, dtype=torch.long, device=device)
            result = result.unsqueeze(0).expand(batch, -1)
        else:
            result = time_indices
            if result.ndim == 1 and result.shape == (steps,):
                result = result.unsqueeze(0).expand(batch, -1)
            if result.shape != (batch, steps) or result.dtype != torch.long:
                raise TypeError(
                    f"time_indices must be long with shape {(batch, steps)}"
                )
            if result.device != device:
                raise TypeError("time_indices and model must share a device")
        if bool((result < 0).any()) or bool(
            (result >= self.config.time_embedding_count).any()
        ):
            raise ValueError("time_indices must be in the closed range [0,2]")
        return result

    def predict_from_encoded_history(
        self,
        encoded_history: torch.Tensor,
        actions: torch.Tensor,
        target_indices: torch.Tensor,
        *,
        time_indices: torch.Tensor | None = None,
        capture_intermediates: bool = False,
    ) -> TemporalPatchMemoryPredictionV1:
        """Recurrently combine encoded frames and decode next-frame queries."""

        batch, steps = self._validate_encoded_history(encoded_history)
        self._validate_actions(actions, batch=batch, steps=steps)
        self._validate_target_indices(target_indices, batch=batch)
        validated_times = self._validated_time_indices(
            time_indices,
            batch=batch,
            steps=steps,
            device=encoded_history.device,
        )

        conditioning = self.action_embedding(actions)
        conditioning = conditioning + self.time_embedding(validated_times)
        recurrent_input = encoded_history + conditioning.unsqueeze(2)
        patch_streams = recurrent_input.permute(0, 2, 1, 3).reshape(
            batch * self.config.spatial_token_count,
            steps,
            self.config.feature_dim,
        )
        initial_hidden = torch.zeros(
            1,
            batch * self.config.spatial_token_count,
            self.config.temporal_hidden_dim,
            dtype=patch_streams.dtype,
            device=patch_streams.device,
        )
        recurrent_streams, _ = self.temporal_gru(patch_streams, initial_hidden)
        recurrent_steps = recurrent_streams.reshape(
            batch,
            self.config.spatial_token_count,
            steps,
            self.config.temporal_hidden_dim,
        ).permute(0, 2, 1, 3)
        recurrent_memory = recurrent_steps[:, -1]

        memory_tokens = recurrent_memory + self.predictor_position.unsqueeze(0)
        query_positions = _gather_spatial_tokens(
            self.predictor_position.unsqueeze(0).expand(batch, -1, -1),
            target_indices,
        )
        future_queries = self.predictor_mask_token.expand(
            batch,
            self.config.target_token_count,
            -1,
        )
        future_queries = future_queries + query_positions
        predictor = torch.cat((memory_tokens, future_queries), dim=1)
        if predictor.shape[1] != self.config.decoder_token_count:
            raise RuntimeError("temporal decoder token count changed")
        predictor_input = predictor if capture_intermediates else None
        predictor_captures: list[torch.Tensor] = []
        for block in self.predictor_blocks:
            predictor = block(predictor)
            if capture_intermediates:
                predictor_captures.append(predictor)
        predicted_queries = self.predictor_norm(
            predictor[:, -self.config.target_token_count :]
        )
        raw_prediction = self.predictor_output(predicted_queries)
        normalized_prediction = F.normalize(
            raw_prediction,
            p=2.0,
            dim=-1,
            eps=self.config.normalization_epsilon,
        )
        return TemporalPatchMemoryPredictionV1(
            raw_predicted_target_tokens=raw_prediction,
            normalized_predicted_target_tokens=normalized_prediction,
            target_indices=target_indices,
            encoded_history=encoded_history,
            time_indices=validated_times,
            recurrent_step_states=recurrent_steps,
            recurrent_memory=recurrent_memory,
            predictor_input=predictor_input,
            predictor_block_outputs=tuple(predictor_captures),
        )

    def predict_future(
        self,
        context_rgb: torch.Tensor,
        actions: torch.Tensor,
        target_indices: torch.Tensor,
        *,
        capture_intermediates: bool = False,
    ) -> TemporalPatchMemoryPredictionV1:
        """Predict the registered next-frame tokens from three RGB frames."""

        batch = self._validate_context_rgb(context_rgb)
        self._validate_actions(
            actions,
            batch=batch,
            steps=self.config.context_length,
        )
        encoded = self.encode_online_full_frame(
            context_rgb.reshape(
                batch * self.config.context_length,
                3,
                self.config.image_size,
                self.config.image_size,
            )
        )
        encoded_history = encoded.reshape(
            batch,
            self.config.context_length,
            self.config.spatial_token_count,
            self.config.feature_dim,
        )
        return self.predict_from_encoded_history(
            encoded_history,
            actions,
            target_indices,
            capture_intermediates=capture_intermediates,
        )

    def predict_current_only(
        self,
        current_rgb: torch.Tensor,
        action: torch.Tensor,
        target_indices: torch.Tensor,
        *,
        capture_intermediates: bool = False,
    ) -> TemporalPatchMemoryPredictionV1:
        """Run the registered one-frame, time-index-two reset control."""

        batch = self._validate_rgb(current_rgb)
        if action.shape != (batch,) or action.dtype != torch.long:
            raise TypeError(f"action must be long with shape {(batch,)}")
        encoded = self.encode_online_full_frame(current_rgb).unsqueeze(1)
        return self.predict_from_encoded_history(
            encoded,
            action.unsqueeze(1),
            target_indices,
            time_indices=torch.full(
                (batch, 1),
                2,
                dtype=torch.long,
                device=current_rgb.device,
            ),
            capture_intermediates=capture_intermediates,
        )

    def forward(
        self,
        context_rgb: torch.Tensor,
        actions: torch.Tensor,
        future_rgb: torch.Tensor,
        target_indices: torch.Tensor,
        *,
        capture_intermediates: bool = False,
    ) -> TemporalPatchMemoryJepaOutputV1:
        prediction = self.predict_future(
            context_rgb,
            actions,
            target_indices,
            capture_intermediates=capture_intermediates,
        )
        target = self.encode_target(future_rgb, target_indices)
        loss = normalized_half_squared_jepa_loss_v1(
            prediction.raw_predicted_target_tokens,
            target.raw_target_tokens,
        )
        return TemporalPatchMemoryJepaOutputV1(
            prediction=prediction,
            target=target,
            loss=loss,
        )
