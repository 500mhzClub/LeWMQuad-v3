"""Dense action-conditioned temporal prediction over frozen DINOv2 tokens.

The predictor keeps the full three-frame spatial token history.  It does not
train or wrap the DINOv2 encoder: callers provide the encoder's normalized
patch tokens and this module predicts the next normalized patch-token grid.
"""
from __future__ import annotations

from numbers import Real

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.predictor import ActionEmbedder, ConditionalBlock


__all__ = ("DenseDINOv2TemporalPredictorV1",)


def _positive_int(name: str, value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class DenseDINOv2TemporalPredictorV1(nn.Module):
    """Predict one or more dense DINOv2 successor-token grids.

    The three context grids are jointly processed as one spatiotemporal token
    sequence.  Each frame receives a learned temporal embedding, every spatial
    position receives a learned spatial embedding, and the two executed actions
    plus the candidate action condition their aligned frame slots through
    AdaLN-zero transformer blocks.

    The output projection is zero-initialized.  Consequently, before learning,
    ``forward`` is exactly the normalized last-frame persistence predictor.
    """

    def __init__(
        self,
        feature_dim: int = 384,
        action_dim: int = 15,
        context_steps: int = 3,
        token_count: int = 256,
        n_layers: int = 6,
        n_heads: int = 6,
        dim_head: int = 64,
        mlp_dim: int = 1536,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_dim = _positive_int("feature_dim", feature_dim)
        self.action_dim = _positive_int("action_dim", action_dim)
        self.context_steps = _positive_int("context_steps", context_steps)
        self.token_count = _positive_int("token_count", token_count)
        n_layers = _positive_int("n_layers", n_layers)
        n_heads = _positive_int("n_heads", n_heads)
        dim_head = _positive_int("dim_head", dim_head)
        mlp_dim = _positive_int("mlp_dim", mlp_dim)
        if self.context_steps != 3:
            raise ValueError("context_steps must be 3 for the frozen H6 contract")
        if isinstance(dropout, bool) or not isinstance(dropout, Real):
            raise TypeError("dropout must be a real number")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.spatial_embedding = nn.Parameter(
            torch.empty(1, 1, self.token_count, self.feature_dim)
        )
        self.temporal_embedding = nn.Parameter(
            torch.empty(1, self.context_steps, 1, self.feature_dim)
        )
        self.action_embedder = ActionEmbedder(
            input_dim=self.action_dim,
            smoothed_dim=10,
            emb_dim=self.feature_dim,
        )
        self.blocks = nn.ModuleList(
            ConditionalBlock(
                self.feature_dim,
                n_heads,
                dim_head,
                mlp_dim,
                float(dropout),
            )
            for _ in range(n_layers)
        )
        self.output_norm = nn.LayerNorm(self.feature_dim)
        self.output_projection = nn.Linear(self.feature_dim, self.feature_dim)

        nn.init.trunc_normal_(self.spatial_embedding, std=0.02)
        nn.init.trunc_normal_(self.temporal_embedding, std=0.02)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    @property
    def history_steps(self) -> int:
        return self.context_steps - 1

    def _validate_context_and_history(
        self,
        context: torch.Tensor,
        history_actions: torch.Tensor,
    ) -> int:
        if not isinstance(context, torch.Tensor):
            raise TypeError("context must be a torch.Tensor")
        if not isinstance(history_actions, torch.Tensor):
            raise TypeError("history_actions must be a torch.Tensor")
        if context.ndim != 4:
            raise ValueError("context must have shape (B,3,256,D)")
        batch = int(context.shape[0])
        expected_context = (
            batch,
            self.context_steps,
            self.token_count,
            self.feature_dim,
        )
        if batch < 1 or tuple(context.shape) != expected_context:
            raise ValueError(
                "context must have shape "
                f"(B,{self.context_steps},{self.token_count},{self.feature_dim}) "
                "with B >= 1"
            )
        expected_history = (batch, self.history_steps, self.action_dim)
        if tuple(history_actions.shape) != expected_history:
            raise ValueError(
                "history_actions must have shape "
                f"(B,{self.history_steps},{self.action_dim})"
            )
        if context.dtype != torch.float32 or history_actions.dtype != torch.float32:
            raise TypeError("context and actions must use float32")
        if context.device != history_actions.device:
            raise TypeError("context and actions must share one device")
        if not bool(torch.isfinite(context).all()):
            raise FloatingPointError("context contains a nonfinite value")
        if not bool(torch.isfinite(history_actions).all()):
            raise FloatingPointError("history_actions contain a nonfinite value")
        return batch

    def _validate_candidate_action(
        self,
        candidate_action: torch.Tensor,
        *,
        batch: int,
        device: torch.device,
    ) -> None:
        if not isinstance(candidate_action, torch.Tensor):
            raise TypeError("candidate_action must be a torch.Tensor")
        if tuple(candidate_action.shape) != (batch, self.action_dim):
            raise ValueError(
                f"candidate_action must have shape (B,{self.action_dim})"
            )
        if candidate_action.dtype != torch.float32:
            raise TypeError("candidate_action must use float32")
        if candidate_action.device != device:
            raise TypeError("context and actions must share one device")
        if not bool(torch.isfinite(candidate_action).all()):
            raise FloatingPointError("candidate_action contains a nonfinite value")

    def forward(
        self,
        context: torch.Tensor,
        history_actions: torch.Tensor,
        candidate_action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the next unit-normalized token grid.

        Args:
            context: Three frozen DINOv2 grids, shaped ``[B,3,256,D]``.
            history_actions: Actions between the context frames, ``[B,2,15]``.
            candidate_action: Action leading to the predicted frame, ``[B,15]``.
        """

        batch = self._validate_context_and_history(context, history_actions)
        self._validate_candidate_action(
            candidate_action,
            batch=batch,
            device=context.device,
        )

        actions = torch.cat((history_actions, candidate_action.unsqueeze(1)), dim=1)
        frame_condition = self.action_embedder(actions)
        condition = (
            frame_condition.unsqueeze(2)
            .expand(-1, -1, self.token_count, -1)
            .reshape(batch, self.context_steps * self.token_count, self.feature_dim)
        )

        hidden = context + self.spatial_embedding + self.temporal_embedding
        hidden = hidden.reshape(
            batch,
            self.context_steps * self.token_count,
            self.feature_dim,
        )
        for block in self.blocks:
            hidden = block(hidden, condition, causal=False)
        final_frame = hidden.reshape(
            batch,
            self.context_steps,
            self.token_count,
            self.feature_dim,
        )[:, -1]
        residual = self.output_projection(self.output_norm(final_frame))
        prediction = F.normalize(context[:, -1] + residual, p=2.0, dim=-1)
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("dense temporal prediction became nonfinite")
        return prediction

    def rollout(
        self,
        context: torch.Tensor,
        history_actions: torch.Tensor,
        action_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """Autoregressively predict a candidate action sequence.

        Args:
            context: Initial context, ``[B,3,256,D]``.
            history_actions: Initial action history, ``[B,2,15]``.
            action_sequence: Future candidate actions, ``[B,H,15]``.

        Returns:
            Unit-normalized predicted grids shaped ``[B,H,256,D]``.
        """

        batch = self._validate_context_and_history(context, history_actions)
        if not isinstance(action_sequence, torch.Tensor):
            raise TypeError("action_sequence must be a torch.Tensor")
        if action_sequence.ndim != 3:
            raise ValueError("action_sequence must have shape (B,H,action_dim)")
        horizon = int(action_sequence.shape[1])
        if tuple(action_sequence.shape) != (batch, horizon, self.action_dim) or horizon < 1:
            raise ValueError(
                f"action_sequence must have shape (B,H,{self.action_dim}) with H >= 1"
            )
        if action_sequence.dtype != torch.float32:
            raise TypeError("action_sequence must use float32")
        if action_sequence.device != context.device:
            raise TypeError("context and actions must share one device")
        if not bool(torch.isfinite(action_sequence).all()):
            raise FloatingPointError("action_sequence contains a nonfinite value")

        rolling_context = context
        rolling_history = history_actions
        predictions: list[torch.Tensor] = []
        for step in range(horizon):
            action = action_sequence[:, step]
            prediction = self(rolling_context, rolling_history, action)
            predictions.append(prediction)
            rolling_context = torch.cat(
                (rolling_context[:, 1:], prediction.unsqueeze(1)), dim=1
            )
            rolling_history = torch.cat(
                (rolling_history[:, 1:], action.unsqueeze(1)), dim=1
            )
        return torch.stack(predictions, dim=1)
