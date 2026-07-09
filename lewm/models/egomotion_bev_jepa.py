"""Egomotion-aligned spatial JEPA with an auxiliary traversability head."""
from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import VisionEncoder


UNKNOWN_CLASS = 0
FREE_CLASS = 1
OCCUPIED_CLASS = 2


def bev_variance_floor_loss(
    features: torch.Tensor,
    *,
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Penalize input-independent BEV features at every channel/cell."""

    if features.ndim != 4:
        raise ValueError("features must have shape (B, D, H, W)")
    if features.shape[0] < 2:
        raise ValueError("variance loss requires at least two BEV samples")
    std = torch.sqrt(features.float().var(dim=0, unbiased=False) + float(eps))
    return torch.relu(float(target_std) - std).mean().to(features.dtype)


def _coordinate_axis(
    minimum: float,
    maximum: float,
    cells: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if cells < 2:
        return torch.tensor([(minimum + maximum) * 0.5], device=device, dtype=dtype)
    return torch.linspace(float(minimum), float(maximum), cells, device=device, dtype=dtype)


def _masked_spatial_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError("values and mask must have the same spatial shape")
    weight = mask.to(values.dtype)
    return (values * weight).sum() / weight.sum().clamp_min(1.0)


def _normalized_spatial_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return (
        F.normalize(prediction, dim=1) - F.normalize(target, dim=1)
    ).square().mean(dim=1)


def _validate_bool_mask(
    mask: torch.Tensor | None,
    expected_shape: tuple[int, ...],
    *,
    name: str,
) -> torch.Tensor | None:
    if mask is None:
        return None
    if tuple(mask.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    if mask.dtype is not torch.bool:
        raise ValueError(f"{name} must use boolean dtype")
    return mask


def _validate_class_weights(
    weights: torch.Tensor | None,
    *,
    classes: int,
    name: str,
    reference: torch.Tensor,
) -> torch.Tensor | None:
    if weights is None:
        return None
    if weights.shape != (classes,):
        raise ValueError(f"{name} must have shape ({classes},)")
    if not torch.is_floating_point(weights):
        raise ValueError(f"{name} must be floating point")
    if not bool(torch.isfinite(weights).all().item()):
        raise ValueError(f"{name} must be finite")
    if bool((weights < 0).any().item()) or not bool((weights.sum() > 0).item()):
        raise ValueError(f"{name} must be nonnegative with positive sum")
    return weights.to(device=reference.device, dtype=reference.dtype)


def _weighted_cross_entropy_mean(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Reduce CE by the sum of weights actually applied to valid cells."""

    loss = F.cross_entropy(logits, labels, reduction="none")
    applied_weights = torch.ones_like(loss)
    if class_weights is not None:
        applied_weights = class_weights[labels]
    applied_weights = applied_weights * mask.to(loss.dtype)
    denominator = applied_weights.sum()
    return (loss * applied_weights).sum() / denominator.clamp_min(
        torch.finfo(loss.dtype).tiny
    )


def warp_bev_current_to_next(
    current: torch.Tensor,
    delta_pose_current: torch.Tensor,
    *,
    forward_range_m: tuple[float, float],
    left_range_m: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp a current ego BEV into the next base frame using relative SE(2).

    ``delta_pose_current`` is ``(dx_forward, dy_left, dyaw)`` for the next base
    expressed in the current base frame. Rows are forward and columns are left.
    The returned mask marks next-frame cells whose source coordinates lie in the
    represented current grid.
    """

    if current.ndim != 4:
        raise ValueError("current must have shape (B, D, H, W)")
    if delta_pose_current.ndim != 2 or delta_pose_current.shape != (
        current.shape[0],
        3,
    ):
        raise ValueError("delta_pose_current must have shape (B, 3)")
    forward_min, forward_max = map(float, forward_range_m)
    left_min, left_max = map(float, left_range_m)
    if not forward_max > forward_min or not left_max > left_min:
        raise ValueError("BEV coordinate ranges must be increasing")

    batch, _channels, height, width = current.shape
    forward = _coordinate_axis(
        forward_min,
        forward_max,
        height,
        device=current.device,
        dtype=current.dtype,
    )
    left = _coordinate_axis(
        left_min,
        left_max,
        width,
        device=current.device,
        dtype=current.dtype,
    )
    next_forward, next_left = torch.meshgrid(forward, left, indexing="ij")
    next_forward = next_forward[None].expand(batch, -1, -1)
    next_left = next_left[None].expand(batch, -1, -1)

    dx = delta_pose_current[:, 0, None, None]
    dy = delta_pose_current[:, 1, None, None]
    yaw = delta_pose_current[:, 2, None, None]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    source_forward = dx + cos_yaw * next_forward - sin_yaw * next_left
    source_left = dy + sin_yaw * next_forward + cos_yaw * next_left

    grid_x = 2.0 * (source_left - left_min) / (left_max - left_min) - 1.0
    grid_y = 2.0 * (source_forward - forward_min) / (
        forward_max - forward_min
    ) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)
    overlap = (
        (grid_x >= -1.0)
        & (grid_x <= 1.0)
        & (grid_y >= -1.0)
        & (grid_y <= 1.0)
    )
    warped = F.grid_sample(
        current,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return warped, overlap[:, None]


class BevDecoder(nn.Module):
    """Globally lift image tokens into fixed-calibration metric BEV queries."""

    def __init__(
        self,
        *,
        token_dim: int,
        bev_dim: int,
        token_side: int,
        bev_size: tuple[int, int],
        forward_range_m: tuple[float, float],
        left_range_m: tuple[float, float],
        attention_heads: int,
    ) -> None:
        super().__init__()
        self.token_side = int(token_side)
        self.bev_size = (int(bev_size[0]), int(bev_size[1]))
        self.forward_range_m = tuple(map(float, forward_range_m))
        self.left_range_m = tuple(map(float, left_range_m))
        if not self.forward_range_m[1] > self.forward_range_m[0]:
            raise ValueError("forward_range_m must be increasing")
        if not self.left_range_m[1] > self.left_range_m[0]:
            raise ValueError("left_range_m must be increasing")
        if int(attention_heads) <= 0:
            raise ValueError("attention_heads must be positive")
        if int(bev_dim) % int(attention_heads) != 0:
            raise ValueError("bev_dim must be divisible by attention_heads")
        forward = torch.linspace(*self.forward_range_m, self.bev_size[0])
        left = torch.linspace(*self.left_range_m, self.bev_size[1])
        forward_grid, left_grid = torch.meshgrid(forward, left, indexing="ij")
        forward_grid = forward_grid / max(abs(value) for value in self.forward_range_m)
        left_grid = left_grid / max(abs(value) for value in self.left_range_m)
        coordinate_features = torch.stack(
            (
                forward_grid,
                left_grid,
                torch.sin(math.pi * forward_grid),
                torch.cos(math.pi * forward_grid),
                torch.sin(math.pi * left_grid),
                torch.cos(math.pi * left_grid),
            ),
            dim=-1,
        ).reshape(-1, 6)
        self.register_buffer("coordinate_features", coordinate_features)
        self.coordinate_query = nn.Sequential(
            nn.Linear(6, int(bev_dim)),
            nn.GELU(),
            nn.Linear(int(bev_dim), int(bev_dim)),
        )
        self.query_bias = nn.Parameter(
            torch.empty(self.bev_size[0] * self.bev_size[1], int(bev_dim))
        )
        nn.init.trunc_normal_(self.query_bias, std=0.02)
        self.token_project = nn.Linear(int(token_dim), int(bev_dim))
        self.cross_attention = nn.MultiheadAttention(
            int(bev_dim),
            int(attention_heads),
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(int(bev_dim))
        self.refine = nn.Sequential(
            nn.Conv2d(int(bev_dim), int(bev_dim), 3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(bev_dim), int(bev_dim), 3, padding=1),
            nn.GroupNorm(1, int(bev_dim)),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        expected_tokens = self.token_side * self.token_side
        if patch_tokens.ndim != 3 or patch_tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"patch_tokens must have shape (B, {expected_tokens}, D)"
            )
        tokens = self.token_project(patch_tokens)
        queries = self.coordinate_query(
            self.coordinate_features.to(dtype=patch_tokens.dtype)
        )
        queries = queries + self.query_bias.to(dtype=queries.dtype)
        queries = queries[None].expand(patch_tokens.shape[0], -1, -1)
        attended, _weights = self.cross_attention(
            queries,
            tokens,
            tokens,
            need_weights=False,
        )
        features = self.query_norm(queries + attended)
        features = features.transpose(1, 2).reshape(
            patch_tokens.shape[0],
            -1,
            self.bev_size[0],
            self.bev_size[1],
        )
        return self.refine(features)


class BevResidualPredictor(nn.Module):
    """Predict newly revealed/changed BEV latent content after geometric warp."""

    def __init__(self, *, bev_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.condition = nn.Sequential(
            nn.Linear(int(action_dim) + 3, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(bev_dim)),
        )
        self.net = nn.Sequential(
            nn.Conv2d(int(bev_dim) * 2, int(hidden_dim), 3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(hidden_dim), int(hidden_dim), 3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(hidden_dim), int(bev_dim), 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        warped_current: torch.Tensor,
        action: torch.Tensor,
        delta_pose_current: torch.Tensor,
    ) -> torch.Tensor:
        if action.ndim != 2 or action.shape[0] != warped_current.shape[0]:
            raise ValueError("action must have shape (B, action_dim)")
        condition = self.condition(torch.cat((action, delta_pose_current), dim=1))
        condition = condition[:, :, None, None].expand_as(warped_current)
        return warped_current + self.net(torch.cat((warped_current, condition), dim=1))


class EgomotionBevJepa(nn.Module):
    """Predict EMA BEV latents while learning traversability logits.

    The promoted predictive branch uses a commanded primitive and its frozen
    train-set nominal SE(2) delta. Realized future odometry is restricted to an
    auxiliary equivariance loss and must not enter that predictor.
    """

    def __init__(
        self,
        *,
        image_size: int = 128,
        patch_size: int = 16,
        encoder_dim: int = 192,
        encoder_depth: int = 6,
        encoder_heads: int = 6,
        encoder_mlp_ratio: int = 4,
        bev_dim: int = 64,
        bev_size: tuple[int, int] = (64, 64),
        forward_range_m: tuple[float, float] = (-0.95, 5.35),
        left_range_m: tuple[float, float] = (-3.15, 3.15),
        action_dim: int = 9,
        bev_attention_heads: int = 4,
        predictor_hidden_dim: int = 128,
        target_ema_momentum: float = 0.996,
        jepa_weight: float = 1.0,
        occupancy_weight: float = 1.0,
        equivariance_weight: float = 0.25,
        action_contrast_weight: float = 1.0,
        action_margin_fraction: float = 0.1,
        variance_weight: float = 0.1,
        variance_target_std: float = 0.5,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if not 0.0 <= float(target_ema_momentum) < 1.0:
            raise ValueError("target_ema_momentum must lie in [0, 1)")
        self.image_size = int(image_size)
        self.action_dim = int(action_dim)
        self.bev_size = (int(bev_size[0]), int(bev_size[1]))
        self.forward_range_m = tuple(map(float, forward_range_m))
        self.left_range_m = tuple(map(float, left_range_m))
        self.target_ema_momentum = float(target_ema_momentum)
        self.jepa_weight = float(jepa_weight)
        self.occupancy_weight = float(occupancy_weight)
        self.equivariance_weight = float(equivariance_weight)
        self.action_contrast_weight = float(action_contrast_weight)
        self.action_margin_fraction = float(action_margin_fraction)
        self.variance_weight = float(variance_weight)
        self.variance_target_std = float(variance_target_std)
        for name in (
            "jepa_weight",
            "occupancy_weight",
            "equivariance_weight",
            "action_contrast_weight",
            "action_margin_fraction",
            "variance_weight",
            "variance_target_std",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")

        self.encoder = VisionEncoder(
            image_size=int(image_size),
            patch_size=int(patch_size),
            hidden_dim=int(encoder_dim),
            depth=int(encoder_depth),
            n_heads=int(encoder_heads),
            mlp_ratio=int(encoder_mlp_ratio),
        )
        token_side = int(image_size) // int(patch_size)
        self.bev_decoder = BevDecoder(
            token_dim=int(encoder_dim),
            bev_dim=int(bev_dim),
            token_side=token_side,
            bev_size=self.bev_size,
            forward_range_m=self.forward_range_m,
            left_range_m=self.left_range_m,
            attention_heads=int(bev_attention_heads),
        )
        self.occupancy_head = nn.Conv2d(int(bev_dim), 3, kernel_size=1)
        self.predictor = BevResidualPredictor(
            bev_dim=int(bev_dim),
            action_dim=int(action_dim),
            hidden_dim=int(predictor_hidden_dim),
        )
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_bev_decoder = copy.deepcopy(self.bev_decoder)
        for module in (self.target_encoder, self.target_bev_decoder):
            module.requires_grad_(False)
            module.eval()

    def train(self, mode: bool = True) -> "EgomotionBevJepa":
        super().train(mode)
        self.target_encoder.eval()
        self.target_bev_decoder.eval()
        return self

    def _encode_online(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder.forward_tokens(image)[:, 1:]
        return self.bev_decoder(tokens)

    @torch.no_grad()
    def _encode_target(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.target_encoder.forward_tokens(image)[:, 1:]
        return self.target_bev_decoder(tokens)

    def occupancy_logits(self, image: torch.Tensor) -> torch.Tensor:
        """Return current-frame unknown/free/occupied logits."""

        return self.occupancy_head(self._encode_online(image))

    def predict_from_command(
        self,
        current_bev: torch.Tensor,
        action: torch.Tensor,
        commanded_delta_pose_current: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict from runtime-available command inputs only."""

        if action.ndim != 2 or action.shape != (
            current_bev.shape[0],
            self.action_dim,
        ):
            raise ValueError(f"action must have shape (B, {self.action_dim})")
        if commanded_delta_pose_current.shape != (current_bev.shape[0], 3):
            raise ValueError(
                "commanded_delta_pose_current must have shape (B, 3)"
            )
        warped, overlap = warp_bev_current_to_next(
            current_bev,
            commanded_delta_pose_current,
            forward_range_m=self.forward_range_m,
            left_range_m=self.left_range_m,
        )
        prediction = self.predictor(
            warped,
            action,
            commanded_delta_pose_current,
        )
        return prediction, warped, overlap

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        momentum = self.target_ema_momentum
        for target_module, online_module in (
            (self.target_encoder, self.encoder),
            (self.target_bev_decoder, self.bev_decoder),
        ):
            for target, online in zip(
                target_module.parameters(),
                online_module.parameters(),
                strict=True,
            ):
                target.mul_(momentum).add_(online, alpha=1.0 - momentum)
            for target, online in zip(
                target_module.buffers(), online_module.buffers(), strict=True
            ):
                target.copy_(online)

    @staticmethod
    def _occupancy_loss(
        logits: torch.Tensor,
        labels: torch.Tensor | None,
        mask: torch.Tensor | None,
        class_weights: torch.Tensor | None,
        *,
        unknown_known_weights: torch.Tensor | None = None,
        free_occupied_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.ndim != 4 or logits.shape[1] != 3:
            raise ValueError("occupancy logits must have shape (B, 3, H, W)")
        decomposed_weights_supplied = (
            unknown_known_weights is not None,
            free_occupied_weights is not None,
        )
        if any(decomposed_weights_supplied) and not all(
            decomposed_weights_supplied
        ):
            raise ValueError(
                "unknown/known and free/occupied class weights must be supplied together"
            )
        if class_weights is not None and all(decomposed_weights_supplied):
            raise ValueError(
                "three-class and decomposed occupancy weights are mutually exclusive"
            )
        if labels is None:
            if mask is not None:
                raise ValueError("occupancy mask requires occupancy labels")
            return logits.sum() * 0.0
        expected_shape = logits.shape[:1] + logits.shape[2:]
        if labels.shape != expected_shape:
            raise ValueError("occupancy labels must have shape (B, H, W)")
        if labels.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise ValueError("occupancy labels must use an integer dtype")
        if labels.numel() and (
            int(labels.min().item()) < UNKNOWN_CLASS
            or int(labels.max().item()) > OCCUPIED_CLASS
        ):
            raise ValueError("occupancy labels must be UNKNOWN/FREE/OCCUPIED")
        mask = _validate_bool_mask(
            mask,
            expected_shape,
            name="occupancy mask",
        )
        valid_mask = (
            mask
            if mask is not None
            else torch.ones(expected_shape, dtype=torch.bool, device=logits.device)
        )
        labels_long = labels.long()
        if all(decomposed_weights_supplied):
            unknown_known_weights = _validate_class_weights(
                unknown_known_weights,
                classes=2,
                name="unknown/known class weights",
                reference=logits,
            )
            free_occupied_weights = _validate_class_weights(
                free_occupied_weights,
                classes=2,
                name="free/occupied class weights",
                reference=logits,
            )
            known_logits = torch.logsumexp(logits[:, 1:], dim=1)
            unknown_known_logits = torch.stack(
                (logits[:, UNKNOWN_CLASS], known_logits),
                dim=1,
            )
            unknown_known_labels = (labels_long != UNKNOWN_CLASS).long()
            unknown_known_loss = _weighted_cross_entropy_mean(
                unknown_known_logits,
                unknown_known_labels,
                valid_mask,
                unknown_known_weights,
            )

            known_mask = valid_mask & (labels_long != UNKNOWN_CLASS)
            free_occupied_labels = labels_long - FREE_CLASS
            free_occupied_labels = free_occupied_labels.clamp_min(0)
            free_occupied_loss = _weighted_cross_entropy_mean(
                logits[:, 1:],
                free_occupied_labels,
                known_mask,
                free_occupied_weights,
            )
            return 0.5 * unknown_known_loss + 0.5 * free_occupied_loss

        class_weights = _validate_class_weights(
            class_weights,
            classes=3,
            name="occupancy class weights",
            reference=logits,
        )
        return _weighted_cross_entropy_mean(
            logits,
            labels_long,
            valid_mask,
            class_weights,
        )

    def forward(
        self,
        current_image: torch.Tensor,
        next_image: torch.Tensor,
        action: torch.Tensor,
        realized_delta_pose_current: torch.Tensor,
        *,
        commanded_delta_pose_current: torch.Tensor,
        current_occupancy: torch.Tensor | None = None,
        next_occupancy: torch.Tensor | None = None,
        current_occupancy_mask: torch.Tensor | None = None,
        next_occupancy_mask: torch.Tensor | None = None,
        next_prediction_mask: torch.Tensor | None = None,
        occupancy_class_weights: torch.Tensor | None = None,
        occupancy_unknown_known_weights: torch.Tensor | None = None,
        occupancy_free_occupied_weights: torch.Tensor | None = None,
        diagnostic_wrong_action: torch.Tensor | None = None,
        diagnostic_wrong_action_delta_pose_current: torch.Tensor | None = None,
        diagnostic_wrong_commanded_delta_pose_current: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if current_image.shape != next_image.shape:
            raise ValueError("current_image and next_image shapes must match")
        if action.ndim != 2 or action.shape != (
            current_image.shape[0],
            self.action_dim,
        ):
            raise ValueError(
                f"action must have shape (B, {self.action_dim})"
            )
        if realized_delta_pose_current.shape != (current_image.shape[0], 3):
            raise ValueError(
                "realized_delta_pose_current must have shape (B, 3)"
            )
        if commanded_delta_pose_current.shape != (current_image.shape[0], 3):
            raise ValueError(
                "commanded_delta_pose_current must have shape (B, 3)"
            )
        expected_grid_shape = (
            current_image.shape[0],
            self.bev_size[0],
            self.bev_size[1],
        )
        current_occupancy_mask = _validate_bool_mask(
            current_occupancy_mask,
            expected_grid_shape,
            name="current_occupancy_mask",
        )
        next_occupancy_mask = _validate_bool_mask(
            next_occupancy_mask,
            expected_grid_shape,
            name="next_occupancy_mask",
        )
        next_prediction_mask = _validate_bool_mask(
            next_prediction_mask,
            expected_grid_shape,
            name="next_prediction_mask",
        )
        if diagnostic_wrong_action is not None and diagnostic_wrong_action.shape != (
            current_image.shape[0], self.action_dim
        ):
            raise ValueError(
                f"diagnostic_wrong_action must have shape (B, {self.action_dim})"
            )
        if (diagnostic_wrong_action is None) != (
            diagnostic_wrong_action_delta_pose_current is None
        ):
            raise ValueError(
                "diagnostic wrong action and its commanded delta must be supplied together"
            )
        if (
            diagnostic_wrong_action_delta_pose_current is not None
            and diagnostic_wrong_action_delta_pose_current.shape
            != (current_image.shape[0], 3)
        ):
            raise ValueError(
                "diagnostic_wrong_action_delta_pose_current must have shape (B, 3)"
            )
        if (
            diagnostic_wrong_commanded_delta_pose_current is not None
            and diagnostic_wrong_commanded_delta_pose_current.shape
            != (current_image.shape[0], 3)
        ):
            raise ValueError(
                "diagnostic_wrong_commanded_delta_pose_current must have shape (B, 3)"
            )
        online_images = torch.cat((current_image, next_image), dim=0)
        online_bev = self._encode_online(online_images)
        current_bev, next_online_bev = online_bev.chunk(2, dim=0)
        target_next_bev = self._encode_target(next_image)
        predicted_next_bev, commanded_warped_current, commanded_overlap = (
            self.predict_from_command(
                current_bev,
                action,
                commanded_delta_pose_current,
            )
        )
        realized_warped_current, realized_overlap = warp_bev_current_to_next(
            current_bev,
            realized_delta_pose_current,
            forward_range_m=self.forward_range_m,
            left_range_m=self.left_range_m,
        )
        prediction_error = _normalized_spatial_error(
            predicted_next_bev,
            target_next_bev,
        )
        prediction_mask = commanded_overlap[:, 0]
        if next_prediction_mask is not None:
            prediction_mask = prediction_mask & next_prediction_mask
        jepa_loss = _masked_spatial_mean(prediction_error, prediction_mask)

        realized_prediction_mask = realized_overlap[:, 0]
        if next_prediction_mask is not None:
            realized_prediction_mask = realized_prediction_mask & next_prediction_mask
        equivariance_error = _normalized_spatial_error(
            realized_warped_current,
            target_next_bev,
        )
        equivariance_loss = _masked_spatial_mean(
            equivariance_error,
            realized_prediction_mask,
        )

        with torch.no_grad():
            persistence_error = _normalized_spatial_error(
                commanded_warped_current.detach(),
                target_next_bev,
            )
            persistence_loss = _masked_spatial_mean(
                persistence_error,
                prediction_mask,
            )
            persistence_ratio = jepa_loss.detach() / persistence_loss.clamp_min(1e-8)

        current_logits = self.occupancy_head(current_bev)
        next_logits = self.occupancy_head(next_online_bev)
        current_occ_loss = self._occupancy_loss(
            current_logits,
            current_occupancy,
            current_occupancy_mask,
            occupancy_class_weights,
            unknown_known_weights=occupancy_unknown_known_weights,
            free_occupied_weights=occupancy_free_occupied_weights,
        )
        next_occ_loss = self._occupancy_loss(
            next_logits,
            next_occupancy,
            next_occupancy_mask,
            occupancy_class_weights,
            unknown_known_weights=occupancy_unknown_known_weights,
            free_occupied_weights=occupancy_free_occupied_weights,
        )
        occupancy_terms = []
        if current_occupancy is not None:
            occupancy_terms.append(current_occ_loss)
        if next_occupancy is not None:
            occupancy_terms.append(next_occ_loss)
        occupancy_loss = (
            torch.stack(occupancy_terms).mean()
            if occupancy_terms
            else current_logits.sum() * 0.0
        )
        variance_loss = bev_variance_floor_loss(
            torch.cat((current_bev, next_online_bev), dim=0),
            target_std=self.variance_target_std,
        )
        action_contrast_loss = jepa_loss.new_zeros(())
        wrong_action_results: dict[str, torch.Tensor] = {}
        if diagnostic_wrong_action is not None:
            wrong_prediction, wrong_warped, wrong_overlap = self.predict_from_command(
                current_bev,
                diagnostic_wrong_action,
                diagnostic_wrong_action_delta_pose_current,
            )
            wrong_mask = prediction_mask & wrong_overlap[:, 0]
            real_matched_loss = _masked_spatial_mean(prediction_error, wrong_mask)
            persistence_matched_loss = _masked_spatial_mean(
                persistence_error,
                wrong_mask,
            )
            wrong_loss = _masked_spatial_mean(
                _normalized_spatial_error(wrong_prediction, target_next_bev),
                wrong_mask,
            )
            required_margin = (
                self.action_margin_fraction * persistence_matched_loss.detach()
            )
            wrong_action_contrast_loss = torch.relu(
                real_matched_loss + required_margin - wrong_loss
            )
            zero_action = torch.zeros_like(action)
            zero_delta = torch.zeros_like(commanded_delta_pose_current)
            zero_prediction, zero_warped, zero_overlap = self.predict_from_command(
                current_bev,
                zero_action,
                zero_delta,
            )
            zero_mask = prediction_mask & zero_overlap[:, 0]
            zero_real_loss = _masked_spatial_mean(prediction_error, zero_mask)
            zero_persistence_loss = _masked_spatial_mean(
                persistence_error,
                zero_mask,
            )
            zero_loss = _masked_spatial_mean(
                _normalized_spatial_error(zero_prediction, target_next_bev),
                zero_mask,
            )
            zero_action_contrast_loss = torch.relu(
                zero_real_loss
                + self.action_margin_fraction * zero_persistence_loss.detach()
                - zero_loss
            )
            action_contrast_loss = 0.5 * (
                wrong_action_contrast_loss + zero_action_contrast_loss
            )
            wrong_action_results = {
                "wrong_action_contrast_loss": wrong_action_contrast_loss.detach(),
                "wrong_action_loss": wrong_loss.detach(),
                "wrong_action_matched_real_loss": real_matched_loss.detach(),
                "wrong_action_advantage": (
                    wrong_loss.detach() - real_matched_loss.detach()
                ),
                "wrong_action_advantage_over_target_change": (
                    (wrong_loss.detach() - real_matched_loss.detach())
                    / persistence_matched_loss.clamp_min(1e-8)
                ),
                "wrong_action_prediction_sensitivity": _masked_spatial_mean(
                    _normalized_spatial_error(
                        wrong_prediction.detach(), predicted_next_bev.detach()
                    ),
                    wrong_mask,
                ),
                "wrong_action_valid_cells": wrong_mask.sum(),
                "wrong_action_matched_mask": wrong_mask[:, None],
                "wrong_action_commanded_warped_bev": wrong_warped.detach(),
                "wrong_action_predicted_next_bev": wrong_prediction.detach(),
                "zero_action_contrast_loss": zero_action_contrast_loss.detach(),
                "zero_action_matched_mask": zero_mask[:, None],
                "zero_action_commanded_warped_bev": zero_warped.detach(),
                "zero_action_predicted_next_bev": zero_prediction.detach(),
            }

        total = (
            self.jepa_weight * jepa_loss
            + self.occupancy_weight * occupancy_loss
            + self.equivariance_weight * equivariance_loss
            + self.action_contrast_weight * action_contrast_loss
            + self.variance_weight * variance_loss
        )
        result: dict[str, Any] = {
            "loss": total,
            "jepa_loss": jepa_loss,
            "equivariance_loss": equivariance_loss,
            "action_contrast_loss": action_contrast_loss,
            "warped_persistence_loss": persistence_loss,
            "prediction_to_persistence_ratio": persistence_ratio,
            "prediction_valid_cells": prediction_mask.sum(),
            "occupancy_loss": occupancy_loss,
            "variance_loss": variance_loss,
            "current_occupancy_logits": current_logits,
            "next_occupancy_logits": next_logits,
            "current_bev": current_bev,
            "commanded_warped_current_bev": commanded_warped_current,
            "warped_persistence_bev": commanded_warped_current,
            "realized_warped_current_bev": realized_warped_current,
            "predicted_next_bev": predicted_next_bev,
            "target_next_bev": target_next_bev,
            "prediction_overlap_mask": commanded_overlap,
            "prediction_valid_mask": prediction_mask[:, None],
            "realized_equivariance_overlap_mask": realized_overlap,
            "realized_equivariance_valid_mask": realized_prediction_mask[:, None],
        }
        result.update(wrong_action_results)
        if diagnostic_wrong_commanded_delta_pose_current is not None:
            with torch.no_grad():
                wrong_delta_prediction, _wrong_delta_warp, wrong_delta_overlap = (
                    self.predict_from_command(
                        current_bev.detach(),
                        action.detach(),
                        diagnostic_wrong_commanded_delta_pose_current,
                    )
                )
                wrong_delta_mask = prediction_mask & wrong_delta_overlap[:, 0]
                matched_real_loss = _masked_spatial_mean(
                    prediction_error.detach(),
                    wrong_delta_mask,
                )
                matched_persistence_loss = _masked_spatial_mean(
                    persistence_error,
                    wrong_delta_mask,
                )
                wrong_delta_loss = _masked_spatial_mean(
                    _normalized_spatial_error(
                        wrong_delta_prediction,
                        target_next_bev,
                    ),
                    wrong_delta_mask,
                )
                wrong_delta_advantage = wrong_delta_loss - matched_real_loss
                wrong_delta_sensitivity = _masked_spatial_mean(
                    _normalized_spatial_error(
                        wrong_delta_prediction,
                        predicted_next_bev.detach(),
                    ),
                    wrong_delta_mask,
                )
            result.update(
                {
                    "wrong_delta_loss": wrong_delta_loss,
                    "wrong_delta_matched_real_loss": matched_real_loss,
                    "wrong_delta_advantage": wrong_delta_advantage,
                    "wrong_delta_prediction_sensitivity": wrong_delta_sensitivity,
                    "wrong_delta_advantage_over_target_change": (
                        wrong_delta_advantage
                        / matched_persistence_loss.clamp_min(1e-8)
                    ),
                    "wrong_delta_valid_cells": wrong_delta_mask.sum(),
                    "wrong_delta_matched_mask": wrong_delta_mask[:, None],
                }
            )
        return result


__all__ = [
    "BevDecoder",
    "BevResidualPredictor",
    "EgomotionBevJepa",
    "FREE_CLASS",
    "OCCUPIED_CLASS",
    "UNKNOWN_CLASS",
    "bev_variance_floor_loss",
    "warp_bev_current_to_next",
]
