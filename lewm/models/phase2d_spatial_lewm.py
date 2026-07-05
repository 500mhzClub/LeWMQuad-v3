"""Corrected normalized spatial JEPA objective registered for Phase 2D."""
from __future__ import annotations

import copy
import math
from typing import Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import Projector, VisionEncoder
from .sigreg import sigreg_stepwise
from .spatial_lewm import spatial_variance_floor_loss
from .spatial_predictor import SpatialTokenPredictor

PredictionInputMode = Literal["state_action", "state_only", "action_only"]
TargetGeometry = Literal["patch", "slot"]


def normalize_spatial_tokens(tokens: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize each spatial token independently."""

    if tokens.ndim < 2:
        raise ValueError("spatial tokens must have at least two dimensions")
    return F.normalize(tokens, p=2, dim=-1, eps=eps)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return a stable masked mean with zero result for an empty mask."""

    if values.shape != mask.shape:
        raise ValueError(
            f"values and mask must have equal shape, got {values.shape} and {mask.shape}"
        )
    weight = mask.to(dtype=values.dtype)
    return (values * weight).sum() / weight.sum().clamp(min=1.0)


def action_identifiability_losses(
    *,
    real_prediction: torch.Tensor,
    targets: torch.Tensor,
    previous_targets: torch.Tensor,
    wrong_predictions: torch.Tensor | None,
    wrong_mask: torch.Tensor | None,
    zero_prediction: torch.Tensor | None,
    non_hold_mask: torch.Tensor | None,
    transition_mask: torch.Tensor | None,
    margin_fraction: float = 0.10,
    margin_floor: float = 1e-4,
) -> dict[str, torch.Tensor]:
    """Compute registered exhaustive wrong-action and zero-action hinge losses."""

    if real_prediction.shape != targets.shape or targets.shape != previous_targets.shape:
        raise ValueError("real_prediction, targets, and previous_targets must align")
    if targets.ndim != 4:
        raise ValueError("normalized spatial tensors must have shape (B, H, N, D)")
    batch, horizon = targets.shape[:2]
    if transition_mask is None:
        valid = torch.ones((batch, horizon), dtype=torch.bool, device=targets.device)
    else:
        if tuple(transition_mask.shape) != (batch, horizon):
            raise ValueError(
                f"transition_mask must have shape {(batch, horizon)}, got "
                f"{tuple(transition_mask.shape)}"
            )
        valid = transition_mask.bool()

    real_mse = (real_prediction - targets).square().mean(dim=(2, 3))
    target_change_mse = (previous_targets - targets).square().mean(dim=(2, 3))
    margin = float(margin_fraction) * target_change_mse.clamp(min=float(margin_floor))

    zero = real_mse.new_zeros(())
    action_loss = zero
    mean_wrong_mse = torch.zeros_like(real_mse)
    wrong_mse = real_mse.new_zeros((batch, horizon, 0))
    wrong_pair_mask = torch.zeros(
        (batch, horizon, 0),
        dtype=torch.bool,
        device=targets.device,
    )
    eligible_wrong = torch.zeros_like(valid)
    if wrong_predictions is not None or wrong_mask is not None:
        if wrong_predictions is None or wrong_mask is None:
            raise ValueError("wrong_predictions and wrong_mask must be supplied together")
        expected_prefix = (batch, horizon)
        if wrong_predictions.ndim != 5 or tuple(wrong_predictions.shape[:2]) != expected_prefix:
            raise ValueError(
                "wrong_predictions must have shape (B, H, K, N, D), got "
                f"{tuple(wrong_predictions.shape)}"
            )
        if tuple(wrong_predictions.shape[3:]) != tuple(targets.shape[2:]):
            raise ValueError("wrong_predictions spatial shape must match targets")
        if tuple(wrong_mask.shape) != tuple(wrong_predictions.shape[:3]):
            raise ValueError("wrong_mask must have shape (B, H, K)")
        wrong_valid = wrong_mask.bool() & valid[:, :, None]
        wrong_pair_mask = wrong_valid
        wrong_mse = (wrong_predictions - targets[:, :, None]).square().mean(dim=(3, 4))
        action_hinge = F.relu(margin[:, :, None] + real_mse[:, :, None] - wrong_mse)
        action_loss = masked_mean(action_hinge, wrong_valid)
        wrong_count = wrong_valid.sum(dim=2)
        mean_wrong_mse = (wrong_mse * wrong_valid).sum(dim=2) / wrong_count.clamp(min=1)
        eligible_wrong = wrong_count > 0

    zero_loss = zero
    zero_mse = torch.zeros_like(real_mse)
    eligible_zero = torch.zeros_like(valid)
    if zero_prediction is not None or non_hold_mask is not None:
        if zero_prediction is None or non_hold_mask is None:
            raise ValueError("zero_prediction and non_hold_mask must be supplied together")
        if zero_prediction.shape != targets.shape:
            raise ValueError("zero_prediction must align with targets")
        if tuple(non_hold_mask.shape) != (batch, horizon):
            raise ValueError("non_hold_mask must have shape (B, H)")
        eligible_zero = non_hold_mask.bool() & valid
        zero_mse = (zero_prediction - targets).square().mean(dim=(2, 3))
        zero_hinge = F.relu(margin + real_mse - zero_mse)
        zero_loss = masked_mean(zero_hinge, eligible_zero)

    return {
        "action_identifiability_loss": action_loss,
        "zero_action_loss": zero_loss,
        "real_mse": real_mse,
        "wrong_mse": wrong_mse,
        "mean_wrong_mse": mean_wrong_mse,
        "zero_mse": zero_mse,
        "target_change_mse": target_change_mse,
        "margin": margin,
        "wrong_pair_mask": wrong_pair_mask,
        "eligible_wrong_mask": eligible_wrong,
        "eligible_zero_mask": eligible_zero,
        "transition_mask": valid,
    }


def action_utility_losses(
    *,
    utility_prediction: torch.Tensor,
    utility_targets: torch.Tensor,
    utility_mask: torch.Tensor,
    utility_group_ids: torch.Tensor,
    regression_weight: float = 0.1,
    ranking_loss: Literal["hard_ce", "soft_ce"] = "hard_ce",
    softmax_temperature: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Compute source-grouped utility ranking loss.

    Candidate utilities are only compared within the same source state. The
    cross-entropy target is the highest oracle utility in each valid source
    group by default; soft utility cross-entropy can be selected for bounded
    diagnostics that need the full within-source utility ordering. A regression
    term preserves score scale without dominating the ranking objective.
    """

    if utility_prediction.ndim != 1:
        raise ValueError("utility_prediction must have shape (B,)")
    batch = utility_prediction.shape[0]
    if tuple(utility_targets.shape) != (batch,):
        raise ValueError("utility_targets must have shape (B,)")
    if tuple(utility_mask.shape) != (batch,):
        raise ValueError("utility_mask must have shape (B,)")
    if tuple(utility_group_ids.shape) != (batch,):
        raise ValueError("utility_group_ids must have shape (B,)")
    if regression_weight < 0.0:
        raise ValueError("regression_weight must be non-negative")
    if ranking_loss not in ("hard_ce", "soft_ce"):
        raise ValueError(f"unsupported action utility ranking loss: {ranking_loss}")
    if softmax_temperature <= 0.0:
        raise ValueError("softmax_temperature must be positive")

    valid = utility_mask.bool()
    regression_loss = masked_mean(
        (utility_prediction - utility_targets).square(),
        valid,
    )
    ce_losses = []
    group_count = utility_prediction.new_zeros(())
    for group_id in torch.unique(utility_group_ids[valid]):
        group_mask = valid & (utility_group_ids == group_id)
        indices = torch.nonzero(group_mask, as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        group_targets = utility_targets[indices]
        if torch.isclose(group_targets.max(), group_targets.min()):
            continue
        group_prediction = utility_prediction[indices]
        if ranking_loss == "soft_ce":
            centered_targets = group_targets - group_targets.mean()
            target_distribution = F.softmax(
                centered_targets / float(softmax_temperature),
                dim=0,
            )
            ce_losses.append(
                -(target_distribution * F.log_softmax(group_prediction, dim=0)).sum()
            )
        else:
            best = torch.argmax(group_targets).reshape(1)
            ce_losses.append(F.cross_entropy(group_prediction.reshape(1, -1), best))
        group_count = group_count + 1.0
    if ce_losses:
        cross_entropy_loss = torch.stack(ce_losses).mean()
    else:
        cross_entropy_loss = utility_prediction.new_zeros(())
    total = cross_entropy_loss + float(regression_weight) * regression_loss
    return {
        "action_utility_loss": total,
        "action_utility_ce_loss": cross_entropy_loss,
        "action_utility_regression_loss": regression_loss,
        "action_utility_valid_count": valid.sum(),
        "action_utility_group_count": group_count,
    }


class LinearTokenProjector(nn.Module):
    """Independent token projector without batch-dependent normalization."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.linear = nn.Linear(latent_dim, latent_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim < 2:
            raise ValueError("tokens must have at least two dimensions")
        return self.linear(tokens)


class IdentityTokenGeometry(nn.Module):
    """Keep encoder patch-token geometry unchanged."""

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim < 3:
            raise ValueError("tokens must have shape (..., N, D)")
        return tokens


class LearnedSlotGeometry(nn.Module):
    """Pool image-aligned patch tokens into learned latent slots.

    This is a bounded Phase 2E target-geometry option. It does not claim object
    discovery; it only tests whether reducing fixed image-grid dominance makes
    action-conditioned consequences more measurable under the existing gates.
    """

    def __init__(self, *, latent_dim: int, num_slots: int):
        super().__init__()
        if num_slots < 1:
            raise ValueError("num_slots must be positive")
        self.latent_dim = int(latent_dim)
        self.num_slots = int(num_slots)
        self.slot_queries = nn.Parameter(torch.empty(num_slots, latent_dim))
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.out = nn.Linear(latent_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        nn.init.trunc_normal_(self.slot_queries, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim < 3:
            raise ValueError("tokens must have shape (..., N, D)")
        if tokens.shape[-1] != self.latent_dim:
            raise ValueError(
                f"expected token dim {self.latent_dim}, got {tokens.shape[-1]}"
            )
        prefix = tokens.shape[:-2]
        num_tokens = tokens.shape[-2]
        flat = tokens.reshape(-1, num_tokens, self.latent_dim)
        keys = self.key(flat)
        values = self.value(flat)
        logits = torch.einsum("sd,bnd->bsn", self.slot_queries, keys)
        logits = logits / math.sqrt(self.latent_dim)
        weights = torch.softmax(logits, dim=-1)
        slots = torch.einsum("bsn,bnd->bsd", weights, values)
        slots = self.norm(self.out(slots))
        return slots.reshape(*prefix, self.num_slots, self.latent_dim)


class Phase2DSpatialLeWorldModel(nn.Module):
    """Normalized spatial JEPA with separate target and prediction projection paths."""

    def __init__(
        self,
        *,
        latent_dim: int = 48,
        cmd_dim: int = 15,
        pred_layers: int = 2,
        pred_heads: int = 4,
        pred_dim_head: int = 12,
        pred_mlp_dim: int = 96,
        pred_dropout: float = 0.1,
        image_size: int = 224,
        patch_size: int = 14,
        encoder_depth: int = 2,
        encoder_heads: int = 3,
        encoder_mlp_ratio: int = 2,
        encoder_dropout: float = 0.0,
        appearance_sigreg_lambda: float = 0.09,
        spatial_variance_lambda: float = 1.0,
        action_identifiability_lambda: float = 0.0,
        zero_action_lambda: float = 0.0,
        action_margin_fraction: float = 0.10,
        action_margin_floor: float = 1e-4,
        detach_action_control_state: bool = False,
        consequence_dim: int = 0,
        consequence_loss_lambda: float = 0.0,
        action_utility_loss_lambda: float = 0.0,
        action_utility_regression_weight: float = 0.1,
        target_geometry: TargetGeometry = "patch",
        num_target_slots: int = 16,
        sigreg_projections: int = 64,
        sigreg_knots: int = 9,
        target_ema_momentum: float | None = None,
        prediction_input_mode: PredictionInputMode = "state_action",
    ):
        super().__init__()
        if target_ema_momentum is not None and not 0.0 <= target_ema_momentum < 1.0:
            raise ValueError("target_ema_momentum must lie in [0, 1)")
        if prediction_input_mode not in ("state_action", "state_only", "action_only"):
            raise ValueError(f"unsupported prediction_input_mode: {prediction_input_mode}")
        if target_geometry not in ("patch", "slot"):
            raise ValueError(f"unsupported target_geometry: {target_geometry}")
        if num_target_slots < 1:
            raise ValueError("num_target_slots must be positive")
        if consequence_dim < 0:
            raise ValueError("consequence_dim must be non-negative")
        if consequence_loss_lambda < 0.0:
            raise ValueError("consequence_loss_lambda must be non-negative")
        if action_utility_loss_lambda < 0.0:
            raise ValueError("action_utility_loss_lambda must be non-negative")
        if action_utility_regression_weight < 0.0:
            raise ValueError("action_utility_regression_weight must be non-negative")
        if prediction_input_mode != "state_action" and (
            action_identifiability_lambda > 0.0 or zero_action_lambda > 0.0
        ):
            raise ValueError(
                "diagnostic input controls cannot optimize action-identifiability losses"
            )
        self.latent_dim = int(latent_dim)
        self.appearance_sigreg_lambda = float(appearance_sigreg_lambda)
        self.spatial_variance_lambda = float(spatial_variance_lambda)
        self.action_identifiability_lambda = float(action_identifiability_lambda)
        self.zero_action_lambda = float(zero_action_lambda)
        self.action_margin_fraction = float(action_margin_fraction)
        self.action_margin_floor = float(action_margin_floor)
        self.detach_action_control_state = bool(detach_action_control_state)
        self.consequence_dim = int(consequence_dim)
        self.consequence_loss_lambda = float(consequence_loss_lambda)
        self.action_utility_loss_lambda = float(action_utility_loss_lambda)
        self.action_utility_regression_weight = float(action_utility_regression_weight)
        self.target_geometry = target_geometry
        self.num_target_slots = int(num_target_slots)
        self.sigreg_projections = int(sigreg_projections)
        self.sigreg_knots = int(sigreg_knots)
        self.target_ema_momentum = target_ema_momentum
        self.prediction_input_mode = prediction_input_mode
        self.spatial_target_std = 1.0 / math.sqrt(self.latent_dim)

        self.encoder = VisionEncoder(
            image_size=image_size,
            patch_size=patch_size,
            hidden_dim=latent_dim,
            depth=encoder_depth,
            n_heads=encoder_heads,
            mlp_ratio=encoder_mlp_ratio,
            dropout=encoder_dropout,
        )
        self.num_state_tokens = (
            self.encoder.num_patches
            if target_geometry == "patch"
            else self.num_target_slots
        )
        self.online_geometry = (
            IdentityTokenGeometry()
            if target_geometry == "patch"
            else LearnedSlotGeometry(
                latent_dim=latent_dim,
                num_slots=self.num_target_slots,
            )
        )
        self.appearance_projector = Projector(latent_dim, latent_dim)
        self.online_target_projector = LinearTokenProjector(latent_dim)
        self.prediction_projector = LinearTokenProjector(latent_dim)
        self.predictor = SpatialTokenPredictor(
            latent_dim=latent_dim,
            cmd_dim=cmd_dim,
            num_spatial_tokens=self.num_state_tokens,
            n_layers=pred_layers,
            n_heads=pred_heads,
            dim_head=pred_dim_head,
            mlp_dim=pred_mlp_dim,
            dropout=pred_dropout,
        )
        self.consequence_head = (
            nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, self.consequence_dim),
            )
            if self.consequence_dim > 0
            else None
        )
        self.action_utility_head = (
            nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, 1),
            )
            if self.action_utility_loss_lambda > 0.0
            else None
        )
        self.action_only_state = (
            nn.Parameter(
                torch.zeros(1, self.num_state_tokens, self.latent_dim)
            )
            if prediction_input_mode == "action_only"
            else None
        )
        if self.action_only_state is not None:
            nn.init.trunc_normal_(self.action_only_state, std=0.02)
        self.target_encoder = (
            copy.deepcopy(self.encoder) if target_ema_momentum is not None else None
        )
        self.target_geometry_module = (
            copy.deepcopy(self.online_geometry)
            if target_ema_momentum is not None
            else None
        )
        self.target_projector = (
            copy.deepcopy(self.online_target_projector)
            if target_ema_momentum is not None
            else None
        )
        if self.uses_ema_target:
            for module in (
                self.target_encoder,
                self.target_geometry_module,
                self.target_projector,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
                module.eval()

    @property
    def uses_ema_target(self) -> bool:
        return self.target_encoder is not None

    def train(self, mode: bool = True) -> Phase2DSpatialLeWorldModel:
        """Keep EMA target modules frozen in evaluation mode."""

        super().train(mode)
        if self.uses_ema_target:
            self.target_encoder.eval()
            self.target_geometry_module.eval()
            self.target_projector.eval()
        return self

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        """EMA-update target parameters without batch-normalization buffers."""

        if not self.uses_ema_target:
            return
        momentum = float(self.target_ema_momentum)
        for target_module, online_module in (
            (self.target_encoder, self.encoder),
            (self.target_geometry_module, self.online_geometry),
            (self.target_projector, self.online_target_projector),
        ):
            for target, online in zip(
                target_module.parameters(),
                online_module.parameters(),
                strict=True,
            ):
                target.mul_(momentum).add_(online, alpha=1.0 - momentum)

    def encode_seq(self, vision: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return online raw CLS and patch-token sequences."""

        if vision.ndim != 5:
            raise ValueError(
                "vision must have shape (B, T, C, H, W), got "
                f"{tuple(vision.shape)}"
            )
        batch, steps = vision.shape[:2]
        tokens = self.encoder.forward_tokens(
            vision.reshape(batch * steps, *vision.shape[2:])
        ).reshape(batch, steps, self.encoder.num_patches + 1, self.latent_dim)
        return tokens[:, :, 0], tokens[:, :, 1:]

    def state_spatial_seq(self, spatial_raw: torch.Tensor) -> torch.Tensor:
        """Return online model-state tokens for the selected target geometry."""

        return self.online_geometry(spatial_raw)

    def target_spatial_seq(
        self,
        vision: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pre-normalized and normalized online or EMA spatial targets."""

        if not self.uses_ema_target:
            _appearance, spatial_raw = self.encode_seq(vision)
            state_tokens = self.state_spatial_seq(spatial_raw)
            pre_normalized = self.online_target_projector(state_tokens)
            return pre_normalized, normalize_spatial_tokens(pre_normalized)
        with torch.no_grad():
            if vision.ndim != 5:
                raise ValueError(
                    "vision must have shape (B, T, C, H, W), got "
                    f"{tuple(vision.shape)}"
                )
            batch, steps = vision.shape[:2]
            tokens = self.target_encoder.forward_tokens(
                vision.reshape(batch * steps, *vision.shape[2:])
            ).reshape(batch, steps, self.encoder.num_patches + 1, self.latent_dim)
            state_tokens = self.target_geometry_module(tokens[:, :, 1:])
            pre_normalized = self.target_projector(state_tokens)
            return pre_normalized, normalize_spatial_tokens(pre_normalized)

    @torch.no_grad()
    def target_spatial_image(self, vision: torch.Tensor) -> torch.Tensor:
        """Return normalized target patch tokens for a batch of images."""

        if vision.ndim != 4:
            raise ValueError(
                "vision must have shape (B, C, H, W), got "
                f"{tuple(vision.shape)}"
            )
        return self.target_spatial_seq(vision[:, None])[1][:, 0]

    def project_predictions(self, predicted_raw: torch.Tensor) -> torch.Tensor:
        """Project and normalize predictor outputs."""

        return normalize_spatial_tokens(self.prediction_projector(predicted_raw))

    def rollout_spatial(
        self,
        start_spatial_raw: torch.Tensor,
        action_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """Return normalized free-running spatial-token predictions."""

        if self.prediction_input_mode == "state_only":
            action_sequence = torch.zeros_like(action_sequence)
        elif self.prediction_input_mode == "action_only":
            start_spatial_raw = self.action_only_state.expand(
                start_spatial_raw.shape[0],
                -1,
                -1,
            )
        return self.project_predictions(
            self.predictor.rollout(start_spatial_raw, action_sequence)
        )

    def _predict_actions(
        self,
        current: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        batch, horizon, num_tokens, dim = current.shape
        if actions.ndim != 3 or tuple(actions.shape[:2]) != (batch, horizon):
            raise ValueError("actions must have shape (B, H, cmd_dim)")
        if self.prediction_input_mode == "state_only":
            actions = torch.zeros_like(actions)
        elif self.prediction_input_mode == "action_only":
            current = self.action_only_state.expand(batch, -1, -1)[:, None].expand(
                -1,
                horizon,
                -1,
                -1,
            )
        raw = self.predictor.predict_step(
            current.reshape(-1, num_tokens, dim),
            actions.reshape(-1, actions.shape[-1]),
        ).reshape(batch, horizon, num_tokens, dim)
        return self.project_predictions(raw)

    def _predict_wrong_actions(
        self,
        current: torch.Tensor,
        wrong_actions: torch.Tensor,
    ) -> torch.Tensor:
        batch, horizon, num_tokens, dim = current.shape
        if wrong_actions.ndim != 4 or tuple(wrong_actions.shape[:2]) != (
            batch,
            horizon,
        ):
            raise ValueError("wrong_actions must have shape (B, H, K, cmd_dim)")
        if self.prediction_input_mode == "state_only":
            wrong_actions = torch.zeros_like(wrong_actions)
        elif self.prediction_input_mode == "action_only":
            current = self.action_only_state.expand(batch, -1, -1)[:, None].expand(
                -1,
                horizon,
                -1,
                -1,
            )
        negatives = wrong_actions.shape[2]
        expanded = current[:, :, None].expand(-1, -1, negatives, -1, -1)
        raw = self.predictor.predict_step(
            expanded.reshape(-1, num_tokens, dim),
            wrong_actions.reshape(-1, wrong_actions.shape[-1]),
        ).reshape(batch, horizon, negatives, num_tokens, dim)
        return self.project_predictions(raw)

    def predict_consequences(self, predicted_tokens: torch.Tensor) -> torch.Tensor:
        """Predict sequence-level consequence labels from latent future tokens."""

        if self.consequence_head is None:
            raise ValueError("consequence head is disabled")
        if predicted_tokens.ndim != 4:
            raise ValueError(
                "predicted_tokens must have shape (B, H, N, D), got "
                f"{tuple(predicted_tokens.shape)}"
            )
        pooled = predicted_tokens.mean(dim=(1, 2))
        return self.consequence_head(pooled)

    def predict_action_utilities(self, predicted_tokens: torch.Tensor) -> torch.Tensor:
        """Predict source-local action utility from latent future tokens."""

        if self.action_utility_head is None:
            raise ValueError("action utility head is disabled")
        if predicted_tokens.ndim != 4:
            raise ValueError(
                "predicted_tokens must have shape (B, H, N, D), got "
                f"{tuple(predicted_tokens.shape)}"
            )
        pooled = predicted_tokens.mean(dim=(1, 2))
        return self.action_utility_head(pooled).squeeze(-1)

    def forward(
        self,
        vision: torch.Tensor,
        cmd_seq: torch.Tensor,
        transition_mask: torch.Tensor | None = None,
        *,
        wrong_actions: torch.Tensor | None = None,
        wrong_mask: torch.Tensor | None = None,
        non_hold_mask: torch.Tensor | None = None,
        consequence_targets: torch.Tensor | None = None,
        consequence_mask: torch.Tensor | None = None,
        action_utility_targets: torch.Tensor | None = None,
        action_utility_mask: torch.Tensor | None = None,
        action_utility_group_ids: torch.Tensor | None = None,
        return_latents: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Compute the registered C0/C1/C2 normalized spatial JEPA objective."""

        if self.action_identifiability_lambda > 0.0 and (
            wrong_actions is None or wrong_mask is None
        ):
            raise ValueError(
                "wrong_actions and wrong_mask are required when "
                "action_identifiability_lambda is positive"
            )
        if self.zero_action_lambda > 0.0 and non_hold_mask is None:
            raise ValueError(
                "non_hold_mask is required when zero_action_lambda is positive"
            )
        if self.consequence_loss_lambda > 0.0 and (
            consequence_targets is None or consequence_mask is None
        ):
            raise ValueError(
                "consequence_targets and consequence_mask are required when "
                "consequence_loss_lambda is positive"
            )
        if self.action_utility_loss_lambda > 0.0 and (
            action_utility_targets is None
            or action_utility_mask is None
            or action_utility_group_ids is None
        ):
            raise ValueError(
                "action_utility_targets, action_utility_mask, and "
                "action_utility_group_ids are required when "
                "action_utility_loss_lambda is positive"
            )
        appearance_raw, spatial_raw = self.encode_seq(vision)
        batch, steps, num_tokens, dim = spatial_raw.shape
        horizon = steps - 1
        if horizon < 1:
            raise ValueError("Phase 2D spatial JEPA requires at least two frames")
        if cmd_seq.ndim != 3 or tuple(cmd_seq.shape[:2]) != (batch, horizon):
            raise ValueError("cmd_seq must have shape (B, T-1, cmd_dim)")
        if transition_mask is None:
            valid = torch.ones(
                (batch, horizon),
                dtype=torch.bool,
                device=vision.device,
            )
        else:
            if tuple(transition_mask.shape) != (batch, horizon):
                raise ValueError("transition_mask must have shape (B, T-1)")
            valid = transition_mask.bool()

        state_raw = self.state_spatial_seq(spatial_raw)
        current = state_raw[:, :-1]
        real_prediction = self._predict_actions(current, cmd_seq)
        target_pre_normalized, target_normalized_all = self.target_spatial_seq(vision)
        targets = target_normalized_all[:, 1:]
        previous_targets = target_normalized_all[:, :-1]
        prediction_mse = (real_prediction - targets).square().mean(dim=(2, 3))
        prediction_loss = masked_mean(prediction_mse, valid)

        control_current = current.detach() if self.detach_action_control_state else current
        wrong_predictions = (
            self._predict_wrong_actions(control_current, wrong_actions)
            if wrong_actions is not None
            else None
        )
        needs_zero = non_hold_mask is not None or self.zero_action_lambda > 0.0
        zero_prediction = (
            self._predict_actions(control_current, torch.zeros_like(cmd_seq))
            if needs_zero
            else None
        )
        controls = action_identifiability_losses(
            real_prediction=real_prediction,
            targets=targets,
            previous_targets=previous_targets,
            wrong_predictions=wrong_predictions,
            wrong_mask=wrong_mask,
            zero_prediction=zero_prediction,
            non_hold_mask=non_hold_mask,
            transition_mask=valid,
            margin_fraction=self.action_margin_fraction,
            margin_floor=self.action_margin_floor,
        )

        appearance_proj = self.appearance_projector(
            appearance_raw.reshape(batch * steps, dim)
        ).reshape(batch, steps, dim)
        appearance_sigreg_loss = sigreg_stepwise(
            appearance_proj,
            n_projections=self.sigreg_projections,
            n_knots=self.sigreg_knots,
        )
        online_target_pre_normalized = self.online_target_projector(state_raw)
        online_target_normalized = normalize_spatial_tokens(online_target_pre_normalized)
        spatial_variance_loss = spatial_variance_floor_loss(
            online_target_normalized,
            target_std=self.spatial_target_std,
        )
        zero = prediction_loss.new_zeros(())
        consequence_loss = zero
        consequence_mse = zero
        consequence_prediction = None
        valid_consequence_fields = torch.zeros((), device=vision.device)
        action_utility_loss = zero
        action_utility_ce_loss = zero
        action_utility_regression_loss = zero
        action_utility_prediction = None
        action_utility_valid_count = torch.zeros((), device=vision.device)
        action_utility_group_count = torch.zeros((), device=vision.device)
        if self.consequence_head is not None and consequence_targets is not None:
            if consequence_mask is None:
                raise ValueError("consequence_mask is required with consequence_targets")
            if tuple(consequence_targets.shape) != (batch, self.consequence_dim):
                raise ValueError(
                    "consequence_targets must have shape "
                    f"{(batch, self.consequence_dim)}, got "
                    f"{tuple(consequence_targets.shape)}"
                )
            if tuple(consequence_mask.shape) != tuple(consequence_targets.shape):
                raise ValueError("consequence_mask must align with consequence_targets")
            consequence_prediction = self.predict_consequences(real_prediction)
            consequence_errors = (
                consequence_prediction - consequence_targets
            ).square()
            valid_consequence = consequence_mask.bool()
            consequence_loss = masked_mean(consequence_errors, valid_consequence)
            consequence_mse = consequence_loss
            valid_consequence_fields = valid_consequence.sum()
        if self.action_utility_head is not None and action_utility_targets is not None:
            if action_utility_mask is None or action_utility_group_ids is None:
                raise ValueError(
                    "action_utility_mask and action_utility_group_ids are required "
                    "with action_utility_targets"
                )
            action_utility_prediction = self.predict_action_utilities(real_prediction)
            utility = action_utility_losses(
                utility_prediction=action_utility_prediction,
                utility_targets=action_utility_targets,
                utility_mask=action_utility_mask,
                utility_group_ids=action_utility_group_ids,
                regression_weight=self.action_utility_regression_weight,
            )
            action_utility_loss = utility["action_utility_loss"]
            action_utility_ce_loss = utility["action_utility_ce_loss"]
            action_utility_regression_loss = utility[
                "action_utility_regression_loss"
            ]
            action_utility_valid_count = utility["action_utility_valid_count"]
            action_utility_group_count = utility["action_utility_group_count"]
        total = (
            prediction_loss
            + self.action_identifiability_lambda
            * controls["action_identifiability_loss"]
            + self.zero_action_lambda * controls["zero_action_loss"]
            + self.consequence_loss_lambda * consequence_loss
            + self.action_utility_loss_lambda * action_utility_loss
            + self.appearance_sigreg_lambda * appearance_sigreg_loss
            + self.spatial_variance_lambda * spatial_variance_loss
        )
        output = {
            "loss": total,
            "prediction_loss": prediction_loss,
            "action_identifiability_loss": controls["action_identifiability_loss"],
            "zero_action_loss": controls["zero_action_loss"],
            "consequence_loss": consequence_loss,
            "consequence_mse": consequence_mse,
            "action_utility_loss": action_utility_loss,
            "action_utility_ce_loss": action_utility_ce_loss,
            "action_utility_regression_loss": action_utility_regression_loss,
            "appearance_sigreg_loss": appearance_sigreg_loss,
            "spatial_variance_loss": spatial_variance_loss,
            "real_prediction_mse": masked_mean(controls["real_mse"], valid),
            "hard_negative_mse": masked_mean(
                controls["wrong_mse"],
                controls["wrong_pair_mask"],
            ),
            "zero_action_mse": masked_mean(
                controls["zero_mse"],
                controls["eligible_zero_mask"],
            ),
            "mean_target_change_mse": masked_mean(
                controls["target_change_mse"],
                valid,
            ),
            "valid_transition_count": valid.sum(),
            "eligible_wrong_transition_count": controls["eligible_wrong_mask"].sum(),
            "eligible_wrong_pair_count": controls["wrong_pair_mask"].sum(),
            "eligible_zero_count": controls["eligible_zero_mask"].sum(),
            "valid_consequence_field_count": valid_consequence_fields,
            "action_utility_valid_count": action_utility_valid_count,
            "action_utility_group_count": action_utility_group_count,
        }
        if return_latents:
            output.update(
                {
                    "appearance_raw": appearance_raw,
                    "appearance_proj": appearance_proj,
                    "spatial_raw": spatial_raw,
                    "state_raw": state_raw,
                    "online_target_pre_normalized": online_target_pre_normalized,
                    "online_target_normalized": online_target_normalized,
                    "target_pre_normalized": target_pre_normalized,
                    "target_normalized_all": target_normalized_all,
                    "real_prediction": real_prediction,
                    "wrong_predictions": wrong_predictions,
                    "zero_prediction": zero_prediction,
                    "consequence_prediction": consequence_prediction,
                    "consequence_targets": consequence_targets,
                    "consequence_mask": consequence_mask,
                    "action_utility_prediction": action_utility_prediction,
                    "action_utility_targets": action_utility_targets,
                    "action_utility_mask": action_utility_mask,
                    "action_utility_group_ids": action_utility_group_ids,
                    **controls,
                }
            )
        return output
