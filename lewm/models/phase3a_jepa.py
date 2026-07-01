"""Small foundational-JEPA model for the Phase 3A positive control."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .phase2d_spatial_lewm import (
    action_identifiability_losses,
    action_utility_losses,
    normalize_spatial_tokens,
)
from .spatial_lewm import spatial_variance_floor_loss
from .spatial_predictor import SpatialTokenPredictor

CONSEQUENCE_BINARY_INDICES = (0, 4, 5)
CONSEQUENCE_SCALAR_INDICES = (1, 2, 3, 6)
CONSEQUENCE_REACHED_GOAL_INDEX = 4


class Phase3APixelTokenEncoder(nn.Module):
    """Encode each RGB cell into a learned spatial token."""

    def __init__(self, *, view_size: int = 9, latent_dim: int = 32):
        super().__init__()
        if view_size < 3:
            raise ValueError("view_size must be at least 3")
        self.view_size = int(view_size)
        self.latent_dim = int(latent_dim)
        self.num_tokens = self.view_size * self.view_size
        self.input = nn.Linear(3, latent_dim)
        self.pos_embed = nn.Parameter(torch.empty(1, self.num_tokens, latent_dim))
        self.mlp = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, vision: torch.Tensor) -> torch.Tensor:
        """Return token grid with shape ``(B, N, D)``."""

        expected = (3, self.view_size, self.view_size)
        if vision.ndim != 4 or tuple(vision.shape[1:]) != expected:
            raise ValueError(
                f"vision must have shape (B, {expected}), got {tuple(vision.shape)}"
            )
        tokens = vision.permute(0, 2, 3, 1).reshape(vision.shape[0], self.num_tokens, 3)
        tokens = self.input(tokens) + self.pos_embed
        return tokens + self.mlp(tokens)


class Phase3AJepaModel(nn.Module):
    """Action-conditioned JEPA over learned pixel tokens."""

    def __init__(
        self,
        *,
        view_size: int = 9,
        spatial_memory_size: int | None = None,
        latent_dim: int = 32,
        action_dim: int = 4,
        pred_layers: int = 2,
        pred_heads: int = 4,
        pred_dim_head: int = 16,
        pred_mlp_dim: int = 128,
        pred_dropout: float = 0.0,
        target_ema_momentum: float | None = 0.99,
        action_margin_fraction: float = 0.10,
        action_margin_floor: float = 1e-4,
        prediction_loss_lambda: float = 1.0,
        action_identifiability_lambda: float = 1.0,
        zero_action_lambda: float = 1.0,
        free_running_action_contrast_lambda: float = 1.0,
        free_running_zero_contrast_lambda: float = 1.0,
        utility_loss_lambda: float = 0.1,
        utility_ranking_loss_lambda: float = 0.1,
        utility_ranking_regression_weight: float = 0.1,
        utility_ranking_loss_type: str = "hard_ce",
        utility_softmax_temperature: float = 0.25,
        utility_source: str = "candidate_score",
        candidate_score_loss_lambda: float = 1.0,
        candidate_score_regression_weight: float = 0.05,
        candidate_score_ranking_loss_type: str = "hard_ce",
        candidate_score_softmax_temperature: float = 0.25,
        detach_candidate_score_state: bool = True,
        candidate_score_gradient_mode: str | None = None,
        candidate_score_source_tokens: bool = False,
        candidate_score_action_summary: str = "statistics",
        candidate_claim_loss_lambda: float = 0.0,
        candidate_score_claim_logit_weight: float = 0.0,
        online_marker_memory_score_weight: float = 0.0,
        candidate_marker_memory_loss_lambda: float = 0.0,
        candidate_marker_memory_score_weight: float = 0.0,
        candidate_marker_memory_delta_loss_weight: float = 1.0,
        candidate_marker_memory_claim_loss_weight: float = 1.0,
        candidate_marker_memory_ranking_loss_lambda: float = 0.0,
        candidate_marker_memory_ranking_loss_type: str = "hard_ce",
        candidate_marker_memory_softmax_temperature: float = 0.25,
        candidate_marker_memory_score_mode: str = "claim_plus_distance",
        structured_marker_memory_loss_lambda: float = 0.0,
        structured_marker_memory_score_weight: float = 0.0,
        structured_marker_memory_ranking_loss_lambda: float = 0.0,
        structured_marker_memory_softmax_temperature: float = 0.25,
        categorical_marker_memory_loss_lambda: float = 0.0,
        categorical_marker_memory_score_weight: float = 0.0,
        categorical_marker_memory_ranking_loss_lambda: float = 0.0,
        categorical_marker_memory_softmax_temperature: float = 0.25,
        categorical_marker_memory_radius: int = 2,
        spatial_marker_memory_loss_lambda: float = 0.0,
        spatial_marker_memory_score_weight: float = 0.0,
        spatial_marker_memory_ranking_loss_lambda: float = 0.0,
        spatial_marker_memory_softmax_temperature: float = 0.25,
        spatial_marker_memory_score_temperature: float = 1.0,
        spatial_frontier_memory_loss_lambda: float = 0.0,
        spatial_frontier_observation_loss_lambda: float = 0.0,
        spatial_frontier_memory_score_loss_lambda: float = 0.0,
        spatial_frontier_memory_score_weight: float = 0.0,
        spatial_frontier_memory_ranking_loss_lambda: float = 0.0,
        spatial_frontier_memory_softmax_temperature: float = 0.25,
        spatial_frontier_memory_occupancy_loss_weight: float = 1.0,
        spatial_frontier_memory_marker_loss_weight: float = 1.0,
        spatial_frontier_memory_marker_cell_loss_weight: float = 1.0,
        spatial_frontier_memory_marker_mass_loss_weight: float = 1.0,
        spatial_frontier_memory_detector_init: str = "direct_rgb",
        spatial_frontier_memory_detector_arch: str = "linear",
        spatial_frontier_memory_gate_mode: str = "linear",
        spatial_frontier_marker_source: str = "frontier",
        spatial_frontier_collision_penalty: float = 2.0,
        spatial_frontier_novelty_reward: float = 0.35,
        spatial_frontier_marker_gate_threshold: float = 0.5,
        spatial_frontier_marker_gate_width: float = 0.25,
        spatial_frontier_marker_update_threshold: float = 0.0,
        spatial_frontier_marker_update_width: float = 1.0,
        detach_consequence_head_state: bool = True,
        consequence_loss_lambda: float = 0.2,
        consequence_dim: int = 7,
        rollout_delta_loss_lambda: float = 1.0,
        teacher_forced_delta_loss_lambda: float = 1.0,
        decision_token_count: int = 4,
        decision_rollout_mode: str = "recurrent",
        decision_recurrent_update: str = "absolute",
        decision_target_geometry: str = "normalized",
        decision_target_scale: float | None = None,
        decision_prediction_loss_lambda: float = 1.0,
        decision_delta_loss_lambda: float = 1.0,
        decision_teacher_forced_prediction_loss_lambda: float = 1.0,
        decision_teacher_forced_delta_loss_lambda: float = 1.0,
        decision_teacher_forced_action_contrast_lambda: float = 1.0,
        decision_teacher_forced_zero_contrast_lambda: float = 1.0,
        decision_action_contrast_lambda: float = 1.0,
        decision_zero_contrast_lambda: float = 1.0,
        use_memory_context: bool = False,
        memory_frame_summary: str = "summary",
        memory_marker_features: bool = False,
        spatial_variance_lambda: float = 0.1,
    ):
        super().__init__()
        if target_ema_momentum is not None and not 0.0 <= target_ema_momentum < 1.0:
            raise ValueError("target_ema_momentum must lie in [0, 1)")
        if consequence_dim != 7:
            raise ValueError(
                "consequence_dim must be 7 for the registered Phase 3A target"
            )
        if utility_source not in ("consequence", "head", "candidate_score"):
            raise ValueError(
                "utility_source must be 'consequence', 'head', or 'candidate_score'"
            )
        if utility_ranking_loss_type not in ("hard_ce", "soft_ce"):
            raise ValueError(
                "utility_ranking_loss_type must be 'hard_ce' or 'soft_ce'"
            )
        if candidate_score_ranking_loss_type not in ("hard_ce", "soft_ce"):
            raise ValueError(
                "candidate_score_ranking_loss_type must be 'hard_ce' or 'soft_ce'"
            )
        if candidate_marker_memory_ranking_loss_type not in ("hard_ce", "soft_ce"):
            raise ValueError(
                "candidate_marker_memory_ranking_loss_type must be 'hard_ce' "
                "or 'soft_ce'"
            )
        if candidate_marker_memory_score_mode not in ("claim_plus_distance", "distance"):
            raise ValueError(
                "candidate_marker_memory_score_mode must be "
                "'claim_plus_distance' or 'distance'"
            )
        if utility_softmax_temperature <= 0.0:
            raise ValueError("utility_softmax_temperature must be positive")
        if candidate_score_softmax_temperature <= 0.0:
            raise ValueError("candidate_score_softmax_temperature must be positive")
        if candidate_marker_memory_softmax_temperature <= 0.0:
            raise ValueError(
                "candidate_marker_memory_softmax_temperature must be positive"
            )
        if structured_marker_memory_loss_lambda < 0.0:
            raise ValueError(
                "structured_marker_memory_loss_lambda must be non-negative"
            )
        if structured_marker_memory_ranking_loss_lambda < 0.0:
            raise ValueError(
                "structured_marker_memory_ranking_loss_lambda must be non-negative"
            )
        if structured_marker_memory_softmax_temperature <= 0.0:
            raise ValueError(
                "structured_marker_memory_softmax_temperature must be positive"
            )
        if categorical_marker_memory_loss_lambda < 0.0:
            raise ValueError(
                "categorical_marker_memory_loss_lambda must be non-negative"
            )
        if categorical_marker_memory_ranking_loss_lambda < 0.0:
            raise ValueError(
                "categorical_marker_memory_ranking_loss_lambda must be non-negative"
            )
        if categorical_marker_memory_softmax_temperature <= 0.0:
            raise ValueError(
                "categorical_marker_memory_softmax_temperature must be positive"
            )
        if categorical_marker_memory_radius < 0:
            raise ValueError("categorical_marker_memory_radius must be non-negative")
        if spatial_marker_memory_loss_lambda < 0.0:
            raise ValueError("spatial_marker_memory_loss_lambda must be non-negative")
        if spatial_marker_memory_ranking_loss_lambda < 0.0:
            raise ValueError(
                "spatial_marker_memory_ranking_loss_lambda must be non-negative"
            )
        if spatial_marker_memory_softmax_temperature <= 0.0:
            raise ValueError(
                "spatial_marker_memory_softmax_temperature must be positive"
            )
        if spatial_marker_memory_score_temperature <= 0.0:
            raise ValueError("spatial_marker_memory_score_temperature must be positive")
        if spatial_frontier_memory_loss_lambda < 0.0:
            raise ValueError("spatial_frontier_memory_loss_lambda must be non-negative")
        if spatial_frontier_observation_loss_lambda < 0.0:
            raise ValueError(
                "spatial_frontier_observation_loss_lambda must be non-negative"
            )
        if spatial_frontier_memory_score_loss_lambda < 0.0:
            raise ValueError(
                "spatial_frontier_memory_score_loss_lambda must be non-negative"
            )
        if spatial_frontier_memory_ranking_loss_lambda < 0.0:
            raise ValueError(
                "spatial_frontier_memory_ranking_loss_lambda must be non-negative"
            )
        if spatial_frontier_memory_softmax_temperature <= 0.0:
            raise ValueError(
                "spatial_frontier_memory_softmax_temperature must be positive"
            )
        if spatial_frontier_memory_occupancy_loss_weight < 0.0:
            raise ValueError(
                "spatial_frontier_memory_occupancy_loss_weight must be non-negative"
            )
        if spatial_frontier_memory_marker_loss_weight < 0.0:
            raise ValueError(
                "spatial_frontier_memory_marker_loss_weight must be non-negative"
            )
        if spatial_frontier_memory_marker_cell_loss_weight < 0.0:
            raise ValueError(
                "spatial_frontier_memory_marker_cell_loss_weight must be non-negative"
            )
        if spatial_frontier_memory_marker_mass_loss_weight < 0.0:
            raise ValueError(
                "spatial_frontier_memory_marker_mass_loss_weight must be non-negative"
            )
        if spatial_frontier_memory_detector_init not in (
            "direct_rgb",
            "neutral",
            "random",
        ):
            raise ValueError(
                "spatial_frontier_memory_detector_init must be 'direct_rgb', "
                "'neutral', or 'random'"
            )
        if spatial_frontier_memory_detector_arch not in ("linear", "mlp"):
            raise ValueError(
                "spatial_frontier_memory_detector_arch must be 'linear' or 'mlp'"
            )
        if (
            spatial_frontier_memory_detector_arch == "mlp"
            and spatial_frontier_memory_detector_init == "direct_rgb"
        ):
            raise ValueError(
                "direct_rgb frontier detector init is only supported by the "
                "linear detector"
            )
        if spatial_frontier_memory_gate_mode not in ("linear", "threshold"):
            raise ValueError(
                "spatial_frontier_memory_gate_mode must be 'linear' or 'threshold'"
            )
        if spatial_frontier_marker_source not in ("frontier", "spatial_marker"):
            raise ValueError(
                "spatial_frontier_marker_source must be 'frontier' or "
                "'spatial_marker'"
            )
        if spatial_frontier_collision_penalty < 0.0:
            raise ValueError("spatial_frontier_collision_penalty must be non-negative")
        if spatial_frontier_novelty_reward < 0.0:
            raise ValueError("spatial_frontier_novelty_reward must be non-negative")
        if spatial_frontier_marker_gate_width <= 0.0:
            raise ValueError("spatial_frontier_marker_gate_width must be positive")
        if spatial_frontier_marker_update_threshold < 0.0:
            raise ValueError(
                "spatial_frontier_marker_update_threshold must be non-negative"
            )
        if spatial_frontier_marker_update_width <= 0.0:
            raise ValueError("spatial_frontier_marker_update_width must be positive")
        if decision_rollout_mode not in ("recurrent", "autoregressive"):
            raise ValueError(
                "decision_rollout_mode must be 'recurrent' or 'autoregressive'"
            )
        if decision_recurrent_update not in ("absolute", "residual"):
            raise ValueError(
                "decision_recurrent_update must be 'absolute' or 'residual'"
            )
        if decision_target_geometry not in ("normalized", "linear"):
            raise ValueError(
                "decision_target_geometry must be 'normalized' or 'linear'"
            )
        if candidate_score_gradient_mode is None:
            candidate_score_gradient_mode = (
                "detached" if detach_candidate_score_state else "full"
            )
        if candidate_score_gradient_mode not in ("detached", "start_only", "full"):
            raise ValueError(
                "candidate_score_gradient_mode must be 'detached', "
                "'start_only', or 'full'"
            )
        if candidate_score_action_summary not in ("statistics", "sequence"):
            raise ValueError(
                "candidate_score_action_summary must be 'statistics' or 'sequence'"
            )
        if memory_frame_summary not in ("summary", "spatial"):
            raise ValueError("memory_frame_summary must be 'summary' or 'spatial'")
        if candidate_claim_loss_lambda < 0.0:
            raise ValueError("candidate_claim_loss_lambda must be non-negative")
        if candidate_marker_memory_loss_lambda < 0.0:
            raise ValueError(
                "candidate_marker_memory_loss_lambda must be non-negative"
            )
        if candidate_marker_memory_delta_loss_weight < 0.0:
            raise ValueError(
                "candidate_marker_memory_delta_loss_weight must be non-negative"
            )
        if candidate_marker_memory_claim_loss_weight < 0.0:
            raise ValueError(
                "candidate_marker_memory_claim_loss_weight must be non-negative"
            )
        if candidate_marker_memory_ranking_loss_lambda < 0.0:
            raise ValueError(
                "candidate_marker_memory_ranking_loss_lambda must be non-negative"
            )
        if spatial_memory_size is None:
            spatial_memory_size = view_size
        if spatial_memory_size < view_size:
            raise ValueError("spatial_memory_size must be >= view_size")
        if spatial_memory_size % 2 != 1:
            raise ValueError("spatial_memory_size must be odd")
        self.view_size = int(view_size)
        self.spatial_memory_size = int(spatial_memory_size)
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.consequence_dim = int(consequence_dim)
        self.decision_token_count = int(decision_token_count)
        if self.decision_token_count < 1:
            raise ValueError("decision_token_count must be positive")
        self.decision_rollout_mode = decision_rollout_mode
        self.decision_recurrent_update = decision_recurrent_update
        self.decision_target_geometry = decision_target_geometry
        self.decision_target_scale = (
            float(latent_dim**0.5)
            if decision_target_scale is None
            else float(decision_target_scale)
        )
        if self.decision_target_scale <= 0.0:
            raise ValueError("decision_target_scale must be positive")
        self.action_margin_fraction = float(action_margin_fraction)
        self.action_margin_floor = float(action_margin_floor)
        self.prediction_loss_lambda = float(prediction_loss_lambda)
        self.action_identifiability_lambda = float(action_identifiability_lambda)
        self.zero_action_lambda = float(zero_action_lambda)
        self.free_running_action_contrast_lambda = float(
            free_running_action_contrast_lambda
        )
        self.free_running_zero_contrast_lambda = float(
            free_running_zero_contrast_lambda
        )
        self.utility_loss_lambda = float(utility_loss_lambda)
        self.utility_ranking_loss_lambda = float(utility_ranking_loss_lambda)
        self.utility_ranking_regression_weight = float(
            utility_ranking_regression_weight
        )
        self.utility_ranking_loss_type = utility_ranking_loss_type
        self.utility_softmax_temperature = float(utility_softmax_temperature)
        self.utility_source = utility_source
        self.candidate_score_loss_lambda = float(candidate_score_loss_lambda)
        self.candidate_score_regression_weight = float(
            candidate_score_regression_weight
        )
        self.candidate_score_ranking_loss_type = candidate_score_ranking_loss_type
        self.candidate_score_softmax_temperature = float(
            candidate_score_softmax_temperature
        )
        self.candidate_score_gradient_mode = candidate_score_gradient_mode
        self.candidate_score_source_tokens = bool(candidate_score_source_tokens)
        self.candidate_score_action_summary = candidate_score_action_summary
        self.candidate_claim_loss_lambda = float(candidate_claim_loss_lambda)
        self.candidate_score_claim_logit_weight = float(
            candidate_score_claim_logit_weight
        )
        self.online_marker_memory_score_weight = float(
            online_marker_memory_score_weight
        )
        self.candidate_marker_memory_loss_lambda = float(
            candidate_marker_memory_loss_lambda
        )
        self.candidate_marker_memory_score_weight = float(
            candidate_marker_memory_score_weight
        )
        self.candidate_marker_memory_delta_loss_weight = float(
            candidate_marker_memory_delta_loss_weight
        )
        self.candidate_marker_memory_claim_loss_weight = float(
            candidate_marker_memory_claim_loss_weight
        )
        self.candidate_marker_memory_ranking_loss_lambda = float(
            candidate_marker_memory_ranking_loss_lambda
        )
        self.candidate_marker_memory_ranking_loss_type = (
            candidate_marker_memory_ranking_loss_type
        )
        self.candidate_marker_memory_softmax_temperature = float(
            candidate_marker_memory_softmax_temperature
        )
        self.candidate_marker_memory_score_mode = candidate_marker_memory_score_mode
        self.structured_marker_memory_loss_lambda = float(
            structured_marker_memory_loss_lambda
        )
        self.structured_marker_memory_score_weight = float(
            structured_marker_memory_score_weight
        )
        self.structured_marker_memory_ranking_loss_lambda = float(
            structured_marker_memory_ranking_loss_lambda
        )
        self.structured_marker_memory_softmax_temperature = float(
            structured_marker_memory_softmax_temperature
        )
        self.categorical_marker_memory_loss_lambda = float(
            categorical_marker_memory_loss_lambda
        )
        self.categorical_marker_memory_score_weight = float(
            categorical_marker_memory_score_weight
        )
        self.categorical_marker_memory_ranking_loss_lambda = float(
            categorical_marker_memory_ranking_loss_lambda
        )
        self.categorical_marker_memory_softmax_temperature = float(
            categorical_marker_memory_softmax_temperature
        )
        self.categorical_marker_memory_radius = int(categorical_marker_memory_radius)
        self.categorical_marker_memory_cells = (
            2 * self.categorical_marker_memory_radius + 1
        ) ** 2
        self.spatial_marker_memory_loss_lambda = float(
            spatial_marker_memory_loss_lambda
        )
        self.spatial_marker_memory_score_weight = float(
            spatial_marker_memory_score_weight
        )
        self.spatial_marker_memory_ranking_loss_lambda = float(
            spatial_marker_memory_ranking_loss_lambda
        )
        self.spatial_marker_memory_softmax_temperature = float(
            spatial_marker_memory_softmax_temperature
        )
        self.spatial_marker_memory_score_temperature = float(
            spatial_marker_memory_score_temperature
        )
        self.spatial_frontier_memory_loss_lambda = float(
            spatial_frontier_memory_loss_lambda
        )
        self.spatial_frontier_observation_loss_lambda = float(
            spatial_frontier_observation_loss_lambda
        )
        self.spatial_frontier_memory_score_loss_lambda = float(
            spatial_frontier_memory_score_loss_lambda
        )
        self.spatial_frontier_memory_score_weight = float(
            spatial_frontier_memory_score_weight
        )
        self.spatial_frontier_memory_ranking_loss_lambda = float(
            spatial_frontier_memory_ranking_loss_lambda
        )
        self.spatial_frontier_memory_softmax_temperature = float(
            spatial_frontier_memory_softmax_temperature
        )
        self.spatial_frontier_memory_occupancy_loss_weight = float(
            spatial_frontier_memory_occupancy_loss_weight
        )
        self.spatial_frontier_memory_marker_loss_weight = float(
            spatial_frontier_memory_marker_loss_weight
        )
        self.spatial_frontier_memory_marker_cell_loss_weight = float(
            spatial_frontier_memory_marker_cell_loss_weight
        )
        self.spatial_frontier_memory_marker_mass_loss_weight = float(
            spatial_frontier_memory_marker_mass_loss_weight
        )
        self.spatial_frontier_memory_detector_init = (
            spatial_frontier_memory_detector_init
        )
        self.spatial_frontier_memory_detector_arch = (
            spatial_frontier_memory_detector_arch
        )
        self.spatial_frontier_memory_gate_mode = spatial_frontier_memory_gate_mode
        self.spatial_frontier_marker_source = spatial_frontier_marker_source
        self.spatial_frontier_collision_penalty = float(
            spatial_frontier_collision_penalty
        )
        self.spatial_frontier_novelty_reward = float(spatial_frontier_novelty_reward)
        self.spatial_frontier_marker_gate_threshold = float(
            spatial_frontier_marker_gate_threshold
        )
        self.spatial_frontier_marker_gate_width = float(
            spatial_frontier_marker_gate_width
        )
        self.spatial_frontier_marker_update_threshold = float(
            spatial_frontier_marker_update_threshold
        )
        self.spatial_frontier_marker_update_width = float(
            spatial_frontier_marker_update_width
        )
        self.detach_candidate_score_state = bool(detach_candidate_score_state)
        self.detach_consequence_head_state = bool(detach_consequence_head_state)
        self.consequence_loss_lambda = float(consequence_loss_lambda)
        self.rollout_delta_loss_lambda = float(rollout_delta_loss_lambda)
        self.teacher_forced_delta_loss_lambda = float(teacher_forced_delta_loss_lambda)
        self.decision_prediction_loss_lambda = float(decision_prediction_loss_lambda)
        self.decision_delta_loss_lambda = float(decision_delta_loss_lambda)
        self.decision_teacher_forced_prediction_loss_lambda = float(
            decision_teacher_forced_prediction_loss_lambda
        )
        self.decision_teacher_forced_delta_loss_lambda = float(
            decision_teacher_forced_delta_loss_lambda
        )
        self.decision_teacher_forced_action_contrast_lambda = float(
            decision_teacher_forced_action_contrast_lambda
        )
        self.decision_teacher_forced_zero_contrast_lambda = float(
            decision_teacher_forced_zero_contrast_lambda
        )
        self.decision_action_contrast_lambda = float(decision_action_contrast_lambda)
        self.decision_zero_contrast_lambda = float(decision_zero_contrast_lambda)
        self.use_memory_context = bool(use_memory_context)
        self.memory_frame_summary = memory_frame_summary
        self.memory_marker_features = bool(memory_marker_features)
        self.spatial_variance_lambda = float(spatial_variance_lambda)
        self.spatial_target_std = 1.0 / (self.latent_dim**0.5)
        self.target_ema_momentum = target_ema_momentum

        self.encoder = Phase3APixelTokenEncoder(
            view_size=view_size,
            latent_dim=latent_dim,
        )
        self.predictor = SpatialTokenPredictor(
            latent_dim=latent_dim,
            cmd_dim=action_dim,
            num_spatial_tokens=self.encoder.num_tokens,
            n_layers=pred_layers,
            n_heads=pred_heads,
            dim_head=pred_dim_head,
            mlp_dim=pred_mlp_dim,
            dropout=pred_dropout,
        )
        self.prediction_projector = nn.Linear(latent_dim, latent_dim)
        self.decision_seed_query = nn.Parameter(
            torch.empty(1, self.decision_token_count, latent_dim)
        )
        memory_input_dim = (
            self.encoder.num_tokens * latent_dim
            if memory_frame_summary == "spatial"
            else 3 * latent_dim
        ) + action_dim
        if self.memory_marker_features:
            memory_input_dim += 3
        self.memory_encoder = nn.GRU(
            input_size=memory_input_dim,
            hidden_size=latent_dim,
            batch_first=True,
        )
        self.decision_seed_from_image = nn.Sequential(
            nn.LayerNorm(4 * latent_dim),
            nn.Linear(4 * latent_dim, latent_dim),
        )
        self.decision_predictor = SpatialTokenPredictor(
            latent_dim=latent_dim,
            cmd_dim=action_dim,
            num_spatial_tokens=self.decision_token_count,
            n_layers=pred_layers,
            n_heads=pred_heads,
            dim_head=pred_dim_head,
            mlp_dim=pred_mlp_dim,
            dropout=pred_dropout,
        )
        self.decision_recurrent_initial = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
        )
        self.decision_recurrent = nn.GRU(
            input_size=action_dim,
            hidden_size=latent_dim,
            batch_first=True,
        )
        self.decision_recurrent_decoder = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, self.decision_token_count * latent_dim),
        )
        self.decision_projector = nn.Linear(latent_dim, latent_dim)
        self.decision_target_projector = nn.Linear(
            consequence_dim,
            self.decision_token_count * latent_dim,
            bias=False,
        )
        nn.init.trunc_normal_(self.decision_seed_query, std=0.02)
        nn.init.orthogonal_(self.decision_target_projector.weight)
        for parameter in self.decision_target_projector.parameters():
            parameter.requires_grad_(False)
        self.utility_head = nn.Sequential(
            nn.LayerNorm(2 * latent_dim),
            nn.Linear(2 * latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 1),
        )
        if candidate_score_action_summary == "sequence":
            self.candidate_action_encoder = nn.GRU(
                input_size=action_dim,
                hidden_size=latent_dim,
                batch_first=True,
            )
            action_feature_dim = latent_dim
        else:
            self.candidate_action_encoder = None
            action_feature_dim = 3 * action_dim
        self.candidate_action_feature_dim = action_feature_dim
        candidate_score_dim = 6 * latent_dim + 3 * consequence_dim + action_feature_dim
        self.candidate_score_head = nn.Sequential(
            nn.LayerNorm(candidate_score_dim),
            nn.Linear(candidate_score_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 1),
        )
        if (
            self.candidate_claim_loss_lambda > 0.0
            or self.candidate_score_claim_logit_weight != 0.0
        ):
            self.candidate_claim_head = nn.Sequential(
                nn.LayerNorm(candidate_score_dim),
                nn.Linear(candidate_score_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, 1),
            )
        else:
            self.candidate_claim_head = None
        if (
            self.candidate_marker_memory_loss_lambda > 0.0
            or self.candidate_marker_memory_score_weight != 0.0
            or self.candidate_marker_memory_ranking_loss_lambda > 0.0
        ):
            self.candidate_marker_memory_delta_head = nn.Sequential(
                nn.LayerNorm(candidate_score_dim),
                nn.Linear(candidate_score_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, 2),
            )
            self.candidate_marker_memory_claim_head = nn.Sequential(
                nn.LayerNorm(candidate_score_dim),
                nn.Linear(candidate_score_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, 1),
            )
        else:
            self.candidate_marker_memory_delta_head = None
            self.candidate_marker_memory_claim_head = None
        structured_marker_memory_dim = 4 * latent_dim
        if (
            self.structured_marker_memory_loss_lambda > 0.0
            or self.structured_marker_memory_score_weight != 0.0
            or self.structured_marker_memory_ranking_loss_lambda > 0.0
        ):
            self.structured_marker_memory_start_head = nn.Sequential(
                nn.LayerNorm(structured_marker_memory_dim),
                nn.Linear(structured_marker_memory_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, 2),
            )
        else:
            self.structured_marker_memory_start_head = None
        if (
            self.categorical_marker_memory_loss_lambda > 0.0
            or self.categorical_marker_memory_score_weight != 0.0
            or self.categorical_marker_memory_ranking_loss_lambda > 0.0
        ):
            self.categorical_marker_memory_logits_head = nn.Sequential(
                nn.LayerNorm(structured_marker_memory_dim),
                nn.Linear(structured_marker_memory_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, self.categorical_marker_memory_cells),
            )
        else:
            self.categorical_marker_memory_logits_head = None
        if (
            self.spatial_marker_memory_loss_lambda > 0.0
            or self.spatial_marker_memory_score_weight != 0.0
            or self.spatial_marker_memory_ranking_loss_lambda > 0.0
            or self.spatial_frontier_marker_source == "spatial_marker"
        ):
            self.spatial_marker_memory_detector = nn.Sequential(
                nn.Conv2d(3, latent_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(latent_dim, 1, kernel_size=1),
            )
        else:
            self.spatial_marker_memory_detector = None
        if (
            self.spatial_frontier_memory_loss_lambda > 0.0
            or self.spatial_frontier_observation_loss_lambda > 0.0
            or self.spatial_frontier_memory_score_loss_lambda > 0.0
            or self.spatial_frontier_memory_score_weight != 0.0
            or self.spatial_frontier_memory_ranking_loss_lambda > 0.0
        ):
            if self.spatial_frontier_memory_detector_arch == "linear":
                self.spatial_frontier_memory_detector = nn.Conv2d(
                    3,
                    4,
                    kernel_size=1,
                )
            else:
                self.spatial_frontier_memory_detector = nn.Sequential(
                    nn.Conv2d(3, latent_dim, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(latent_dim, 4, kernel_size=1),
                )
            self._initialize_spatial_frontier_memory_detector()
        else:
            self.spatial_frontier_memory_detector = None
        self.consequence_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, self.consequence_dim),
        )
        self.target_encoder = (
            copy.deepcopy(self.encoder) if target_ema_momentum is not None else None
        )
        if self.target_encoder is not None:
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
            self.target_encoder.eval()

    @property
    def uses_ema_target(self) -> bool:
        return self.target_encoder is not None

    def train(self, mode: bool = True) -> Phase3AJepaModel:
        super().train(mode)
        if self.target_encoder is not None:
            self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        if self.target_encoder is None:
            return
        momentum = float(self.target_ema_momentum)
        for target, online in zip(
            self.target_encoder.parameters(),
            self.encoder.parameters(),
            strict=True,
        ):
            target.mul_(momentum).add_(online, alpha=1.0 - momentum)

    def encode_seq(self, vision: torch.Tensor, *, target: bool = False) -> torch.Tensor:
        """Encode ``(B, T, C, H, W)`` into ``(B, T, N, D)`` tokens."""

        if vision.ndim != 5:
            raise ValueError(f"vision must have shape (B, T, C, H, W), got {vision.shape}")
        batch, steps = vision.shape[:2]
        encoder = self.target_encoder if target and self.target_encoder is not None else self.encoder
        tokens = encoder(vision.reshape(batch * steps, *vision.shape[2:]))
        return tokens.reshape(batch, steps, self.encoder.num_tokens, self.latent_dim)

    def project_predictions(self, tokens: torch.Tensor) -> torch.Tensor:
        return normalize_spatial_tokens(self.prediction_projector(tokens))

    def project_decision_predictions(self, tokens: torch.Tensor) -> torch.Tensor:
        projected = self.decision_projector(tokens)
        if self.decision_target_geometry == "normalized":
            return normalize_spatial_tokens(projected)
        return projected

    def source_token_summary(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        """Return pooled, beacon, and center features for a source token grid."""

        if spatial_tokens.ndim != 3:
            raise ValueError("spatial_tokens must have shape (B, N, D)")
        pooled = spatial_tokens.mean(dim=1)
        beacon = spatial_tokens[:, 0]
        center_index = (self.view_size // 2) * self.view_size + (self.view_size // 2)
        center = spatial_tokens[:, center_index]
        return torch.cat([pooled, beacon, center], dim=-1)

    def encode_memory_context(
        self,
        history_vision: torch.Tensor | None,
        history_actions: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Encode an online observation/action history into one belief vector."""

        if (
            not self.use_memory_context
            or history_vision is None
            or history_actions is None
            or history_vision.shape[1] == 0
        ):
            return torch.zeros(
                batch_size,
                self.latent_dim,
                device=device,
                dtype=dtype,
            )
        if history_vision.ndim != 5:
            raise ValueError(
                "history_vision must have shape (B, K, C, H, W), got "
                f"{tuple(history_vision.shape)}"
            )
        if history_actions.ndim != 3:
            raise ValueError(
                "history_actions must have shape (B, K, action_dim), got "
                f"{tuple(history_actions.shape)}"
            )
        if history_vision.shape[:2] != history_actions.shape[:2]:
            raise ValueError("history observations and actions must share B,K")
        if history_actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"expected history action dim {self.action_dim}, got "
                f"{history_actions.shape[-1]}"
            )
        history_tokens = self.encode_seq(history_vision, target=False)
        if self.memory_frame_summary == "spatial":
            frame_features = history_tokens.flatten(start_dim=2)
        else:
            pooled = history_tokens.mean(dim=2)
            beacon = history_tokens[:, :, 0]
            center_index = (
                (self.view_size // 2) * self.view_size + (self.view_size // 2)
            )
            center = history_tokens[:, :, center_index]
            frame_features = torch.cat([pooled, beacon, center], dim=-1)
        memory_inputs = torch.cat([frame_features, history_actions], dim=-1)
        if self.memory_marker_features:
            marker_features = self.marker_saliency_features(history_vision)
            memory_inputs = torch.cat([memory_inputs, marker_features], dim=-1)
        _, hidden = self.memory_encoder(memory_inputs)
        return hidden[-1]

    def marker_saliency_features(self, history_vision: torch.Tensor) -> torch.Tensor:
        """Return RGB-only marker presence and egocentric centroid features."""

        if history_vision.ndim != 5:
            raise ValueError(
                "history_vision must have shape (B, K, C, H, W), got "
                f"{tuple(history_vision.shape)}"
            )
        if history_vision.shape[2] != 3:
            raise ValueError("history_vision must contain RGB channels")
        red = history_vision[:, :, 0]
        green = history_vision[:, :, 1]
        blue = history_vision[:, :, 2]
        saliency = (green - torch.maximum(red, blue) - 0.05).relu()
        batch, steps, height, width = saliency.shape
        row_coords = (
            torch.arange(height, device=saliency.device, dtype=saliency.dtype)
        )
        col_coords = (
            torch.arange(width, device=saliency.device, dtype=saliency.dtype)
        )
        radius_y = float(height // 2)
        radius_x = float(width // 2)
        ahead = (radius_y - row_coords) / float(max(self.view_size, 1))
        lateral = (col_coords - radius_x) / float(max(self.view_size, 1))
        total = saliency.sum(dim=(-1, -2))
        denom = total.clamp_min(1e-6)
        ahead_feature = (
            saliency * ahead.view(1, 1, height, 1)
        ).sum(dim=(-1, -2)) / denom
        lateral_feature = (
            saliency * lateral.view(1, 1, 1, width)
        ).sum(dim=(-1, -2)) / denom
        presence = total.clamp(max=1.0)
        return torch.stack([presence, ahead_feature, lateral_feature], dim=-1)

    def marker_saliency_delta(
        self,
        vision: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return RGB-only marker detections and normalized egocentric deltas."""

        if vision.ndim != 4:
            raise ValueError(
                "vision must have shape (B, C, H, W), got "
                f"{tuple(vision.shape)}"
            )
        if vision.shape[1] != 3:
            raise ValueError("vision must contain RGB channels")
        red = vision[:, 0]
        green = vision[:, 1]
        blue = vision[:, 2]
        marker_mask = (green >= 0.7) & (red <= 0.35) & (blue <= 0.45)
        saliency = (green - torch.maximum(red, blue)).relu()
        saliency = saliency * marker_mask.to(dtype=saliency.dtype)
        batch, height, width = saliency.shape
        row_coords = torch.arange(height, device=vision.device, dtype=vision.dtype)
        col_coords = torch.arange(width, device=vision.device, dtype=vision.dtype)
        radius_y = float(height // 2)
        radius_x = float(width // 2)
        ahead = (radius_y - row_coords) / float(max(self.view_size, 1))
        lateral = (col_coords - radius_x) / float(max(self.view_size, 1))
        total = saliency.sum(dim=(-1, -2))
        detected = total > 1e-6
        denom = total.clamp_min(1e-6)
        ahead_delta = (
            saliency * ahead.view(1, height, 1)
        ).sum(dim=(-1, -2)) / denom
        lateral_delta = (
            saliency * lateral.view(1, 1, width)
        ).sum(dim=(-1, -2)) / denom
        delta = torch.stack([ahead_delta, lateral_delta], dim=-1)
        return detected, delta

    def online_marker_memory_start_delta(
        self,
        history_vision: torch.Tensor | None,
        history_actions: torch.Tensor | None,
        start_vision: torch.Tensor | None = None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Maintain RGB marker memory through observed egocentric actions."""

        delta = torch.zeros(batch_size, 2, device=device, dtype=dtype)
        seen = torch.zeros(batch_size, device=device, dtype=torch.bool)
        if (
            history_vision is not None
            and history_actions is not None
            and history_vision.shape[1] > 0
        ):
            if history_vision.ndim != 5:
                raise ValueError(
                    "history_vision must have shape (B, K, C, H, W), got "
                    f"{tuple(history_vision.shape)}"
                )
            if history_actions.ndim != 3:
                raise ValueError(
                    "history_actions must have shape (B, K, action_dim), got "
                    f"{tuple(history_actions.shape)}"
                )
            if history_vision.shape[:2] != history_actions.shape[:2]:
                raise ValueError("history observations and actions must share B,K")
            if history_vision.shape[0] != batch_size:
                raise ValueError("history batch size must match batch_size")
            for step in range(history_vision.shape[1]):
                detected, observed_delta = self.marker_saliency_delta(
                    history_vision[:, step].to(device=device, dtype=dtype)
                )
                delta = torch.where(detected[:, None], observed_delta, delta)
                seen = seen | detected
                rolled = self.rollout_marker_delta(
                    delta,
                    history_actions[:, step : step + 1].to(
                        device=device,
                        dtype=dtype,
                    ),
                )
                delta = torch.where(seen[:, None], rolled, delta)
        if start_vision is not None:
            if start_vision.ndim != 4:
                raise ValueError(
                    "start_vision must have shape (B, C, H, W), got "
                    f"{tuple(start_vision.shape)}"
                )
            detected, observed_delta = self.marker_saliency_delta(
                start_vision.to(device=device, dtype=dtype)
            )
            update = detected & ~seen
            delta = torch.where(update[:, None], observed_delta, delta)
            seen = seen | detected
        return seen, delta

    def online_marker_memory_score(
        self,
        history_vision: torch.Tensor | None,
        history_actions: torch.Tensor | None,
        actions: torch.Tensor,
        start_vision: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score candidates with non-privileged RGB marker memory and odometry."""

        if actions.ndim != 3:
            raise ValueError("actions must have shape (B, H, action_dim)")
        seen, start_delta = self.online_marker_memory_start_delta(
            history_vision,
            history_actions,
            start_vision,
            batch_size=actions.shape[0],
            device=actions.device,
            dtype=actions.dtype,
        )
        final_delta = self.rollout_marker_delta(start_delta, actions)
        distance_cells = torch.round(
            final_delta.abs().sum(dim=-1) * float(max(self.view_size, 1))
        )
        claimed = (distance_cells <= 0.0) & seen
        score = (100.0 * claimed.to(dtype=actions.dtype)) - distance_cells
        return torch.where(seen, score, actions.new_zeros(actions.shape[0]))

    def online_frontier_observation_maps(
        self,
        vision: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return observed and blocked egocentric maps from one RGB crop."""

        if vision.ndim != 4:
            raise ValueError(
                "vision must have shape (B, C, H, W), got "
                f"{tuple(vision.shape)}"
            )
        if vision.shape[1:] != (3, self.view_size, self.view_size):
            raise ValueError(
                "vision must have shape "
                f"(B, 3, {self.view_size}, {self.view_size}), got "
                f"{tuple(vision.shape)}"
            )
        observed = torch.ones(
            vision.shape[0],
            self.view_size,
            self.view_size,
            device=vision.device,
            dtype=vision.dtype,
        )
        blocked = (
            vision.amax(dim=1) < 0.25
        ).to(device=vision.device, dtype=vision.dtype)
        return observed, blocked

    def center_insert_view_map(self, view_map: torch.Tensor) -> torch.Tensor:
        """Insert a view-sized map into the center of the spatial memory map."""

        if view_map.ndim != 3:
            raise ValueError("view_map must have shape (B, H, W)")
        if view_map.shape[-2:] != (self.view_size, self.view_size):
            raise ValueError(
                "view_map must have shape "
                f"(B, {self.view_size}, {self.view_size}), got "
                f"{tuple(view_map.shape)}"
            )
        if self.spatial_memory_size == self.view_size:
            return view_map
        memory = view_map.new_zeros(
            view_map.shape[0],
            self.spatial_memory_size,
            self.spatial_memory_size,
        )
        start = self.spatial_memory_size // 2 - self.view_size // 2
        end = start + self.view_size
        memory[:, start:end, start:end] = view_map
        return memory

    def current_view_footprint_like(self, memory_map: torch.Tensor) -> torch.Tensor:
        """Return the currently visible crop footprint inside a memory map."""

        if memory_map.ndim != 3:
            raise ValueError("memory_map must have shape (B, H, W)")
        footprint = torch.zeros_like(memory_map)
        height, width = memory_map.shape[-2:]
        start_y = height // 2 - self.view_size // 2
        start_x = width // 2 - self.view_size // 2
        footprint[
            :,
            start_y : start_y + self.view_size,
            start_x : start_x + self.view_size,
        ] = 1.0
        return footprint

    def roll_online_frontier_maps(
        self,
        observed: torch.Tensor,
        blocked: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform observed/blocked maps through egocentric actions."""

        if observed.shape != blocked.shape:
            raise ValueError("observed and blocked maps must share shape")
        if observed.ndim != 3:
            raise ValueError("maps must have shape (B, H, W)")
        if observed.shape[-1] != observed.shape[-2]:
            raise ValueError("maps must be square")
        if actions.ndim != 3:
            raise ValueError("actions must have shape (B, T, action_dim)")
        if observed.shape[0] != actions.shape[0]:
            raise ValueError("maps and actions batch sizes differ")
        current_observed = observed
        current_blocked = blocked
        center = current_observed.shape[-1] // 2
        for step in range(actions.shape[1]):
            forward_observed = torch.zeros_like(current_observed)
            forward_observed[:, 1:, :] = current_observed[:, :-1, :]
            forward_blocked = torch.zeros_like(current_blocked)
            forward_blocked[:, 1:, :] = current_blocked[:, :-1, :]
            ahead_blocked = current_blocked[:, center - 1, center] > 0.5
            forward_observed = torch.where(
                ahead_blocked[:, None, None],
                current_observed,
                forward_observed,
            )
            forward_blocked = torch.where(
                ahead_blocked[:, None, None],
                current_blocked,
                forward_blocked,
            )
            observed_candidates = torch.stack(
                [
                    forward_observed,
                    torch.rot90(current_observed, k=1, dims=(-2, -1)),
                    torch.rot90(current_observed, k=-1, dims=(-2, -1)),
                    current_observed,
                ],
                dim=1,
            )
            blocked_candidates = torch.stack(
                [
                    forward_blocked,
                    torch.rot90(current_blocked, k=1, dims=(-2, -1)),
                    torch.rot90(current_blocked, k=-1, dims=(-2, -1)),
                    current_blocked,
                ],
                dim=1,
            )
            weights = actions[:, step, :4].to(dtype=current_observed.dtype)
            weights = weights[:, :, None, None]
            current_observed = (observed_candidates * weights).sum(dim=1)
            current_blocked = (blocked_candidates * weights).sum(dim=1)
        return current_observed, current_blocked

    def online_frontier_start_maps(
        self,
        history_vision: torch.Tensor | None,
        history_actions: torch.Tensor | None,
        start_vision: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Maintain non-privileged egocentric observed/blocked maps."""

        observed = torch.zeros(
            batch_size,
            self.spatial_memory_size,
            self.spatial_memory_size,
            device=device,
            dtype=dtype,
        )
        blocked = torch.zeros_like(observed)
        if (
            history_vision is not None
            and history_actions is not None
            and history_vision.shape[1] > 0
        ):
            if history_vision.ndim != 5:
                raise ValueError(
                    "history_vision must have shape (B, K, C, H, W), got "
                    f"{tuple(history_vision.shape)}"
                )
            if history_actions.ndim != 3:
                raise ValueError(
                    "history_actions must have shape (B, K, action_dim), got "
                    f"{tuple(history_actions.shape)}"
                )
            if history_vision.shape[:2] != history_actions.shape[:2]:
                raise ValueError("history observations and actions must share B,K")
            for step in range(history_vision.shape[1]):
                obs_seen, obs_blocked = self.online_frontier_observation_maps(
                    history_vision[:, step].to(device=device, dtype=dtype)
                )
                obs_seen = self.center_insert_view_map(obs_seen)
                obs_blocked = self.center_insert_view_map(obs_blocked)
                observed = torch.maximum(observed, obs_seen)
                blocked = torch.maximum(blocked, obs_blocked)
                observed, blocked = self.roll_online_frontier_maps(
                    observed,
                    blocked,
                    history_actions[:, step : step + 1].to(
                        device=device,
                        dtype=dtype,
                    ),
                )
        if start_vision is not None:
            obs_seen, obs_blocked = self.online_frontier_observation_maps(
                start_vision.to(device=device, dtype=dtype)
            )
            obs_seen = self.center_insert_view_map(obs_seen)
            obs_blocked = self.center_insert_view_map(obs_blocked)
            observed = torch.maximum(observed, obs_seen)
            blocked = torch.maximum(blocked, obs_blocked)
        return observed.clamp(0.0, 1.0), blocked.clamp(0.0, 1.0)

    def online_frontier_marker_score(
        self,
        history_vision: torch.Tensor | None,
        history_actions: torch.Tensor | None,
        actions: torch.Tensor,
        start_vision: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score novelty before marker sighting, then marker claiming."""

        if actions.ndim != 3:
            raise ValueError("actions must have shape (B, H, action_dim)")
        marker_seen, _ = self.online_marker_memory_start_delta(
            history_vision,
            history_actions,
            start_vision,
            batch_size=actions.shape[0],
            device=actions.device,
            dtype=actions.dtype,
        )
        marker_score = self.online_marker_memory_score(
            history_vision,
            history_actions,
            actions,
            start_vision=start_vision,
        )
        observed, blocked = self.online_frontier_start_maps(
            history_vision,
            history_actions,
            start_vision,
            batch_size=actions.shape[0],
            device=actions.device,
            dtype=actions.dtype,
        )
        score = actions.new_zeros(actions.shape[0])
        footprint = self.current_view_footprint_like(observed)
        center = observed.shape[-1] // 2
        for step in range(actions.shape[1]):
            ahead_blocked = blocked[:, center - 1, center].clamp(0.0, 1.0)
            score = (
                score
                - self.spatial_frontier_collision_penalty
                * actions[:, step, 0]
                * ahead_blocked
            )
            observed, blocked = self.roll_online_frontier_maps(
                observed,
                blocked,
                actions[:, step : step + 1],
            )
            novel = (footprint - observed).clamp(0.0, 1.0)
            score = (
                score
                + self.spatial_frontier_novelty_reward
                * novel.flatten(start_dim=1).sum(dim=-1)
            )
            observed = torch.maximum(observed, footprint)
        return torch.where(marker_seen, marker_score, score)

    def spatial_frontier_observation_supervision_loss(
        self,
        *,
        history_vision: torch.Tensor | None,
        vision: torch.Tensor,
        history_targets: torch.Tensor | None,
        vision_targets: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Train per-frame marker/occupancy evidence maps from label targets."""

        if self.spatial_frontier_memory_detector is None:
            zero = vision.new_zeros(())
            return zero, zero, zero, zero
        frame_batches = []
        target_batches = []
        expected_map_shape = (4, self.view_size, self.view_size)
        if history_targets is not None:
            if history_vision is None:
                raise ValueError(
                    "history_targets require matching history_vision frames"
                )
            if history_targets.shape[:2] != history_vision.shape[:2]:
                raise ValueError(
                    "history spatial frontier targets must share B,K with "
                    "history_vision"
                )
            if tuple(history_targets.shape[2:]) != expected_map_shape:
                raise ValueError(
                    "history spatial frontier targets must have shape "
                    f"(B, K, {expected_map_shape}), got "
                    f"{tuple(history_targets.shape)}"
                )
            if history_vision.shape[1] > 0:
                frame_batches.append(history_vision)
                target_batches.append(history_targets)
        if vision_targets is not None:
            if vision_targets.shape[:2] != vision.shape[:2]:
                raise ValueError(
                    "vision spatial frontier targets must share B,T with vision"
                )
            if tuple(vision_targets.shape[2:]) != expected_map_shape:
                raise ValueError(
                    "vision spatial frontier targets must have shape "
                    f"(B, T, {expected_map_shape}), got "
                    f"{tuple(vision_targets.shape)}"
                )
            frame_batches.append(vision)
            target_batches.append(vision_targets)
        if not frame_batches:
            zero = vision.new_zeros(())
            return zero, zero, zero, zero
        frames = torch.cat(frame_batches, dim=1).to(
            device=vision.device,
            dtype=vision.dtype,
        )
        targets = torch.cat(target_batches, dim=1).to(
            device=vision.device,
            dtype=vision.dtype,
        )
        batch, steps = frames.shape[:2]
        logits = self.spatial_frontier_memory_detector(
            frames.reshape(batch * steps, *frames.shape[2:])
        )
        flat_targets = targets.reshape(batch * steps, *targets.shape[2:])
        marker_target = flat_targets[:, 0]
        marker_pos_weight = logits.new_tensor(float(self.view_size * self.view_size))
        marker_loss = F.binary_cross_entropy_with_logits(
            logits[:, 0],
            marker_target,
            pos_weight=marker_pos_weight,
        )
        occupancy_loss = F.binary_cross_entropy_with_logits(
            logits[:, 1:],
            flat_targets[:, 1:],
        )
        return (
            marker_loss + occupancy_loss,
            marker_loss,
            occupancy_loss,
            logits.new_tensor(float(batch * steps)),
        )

    def spatial_frontier_observation_maps(
        self,
        vision: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return learned marker/occupancy evidence maps from one RGB crop."""

        if self.spatial_frontier_memory_detector is None:
            raise RuntimeError("spatial frontier memory detector is disabled")
        if vision.ndim != 4:
            raise ValueError(
                "vision must have shape (B, C, H, W), got "
                f"{tuple(vision.shape)}"
            )
        if vision.shape[1:] != (3, self.view_size, self.view_size):
            raise ValueError(
                "vision must have shape "
                f"(B, 3, {self.view_size}, {self.view_size}), got "
                f"{tuple(vision.shape)}"
            )
        logits = self.spatial_frontier_memory_detector(vision)
        marker_logits = logits[:, 0]
        flat_marker_logits = marker_logits.flatten(start_dim=1)
        marker_probs = flat_marker_logits.softmax(dim=-1).reshape_as(marker_logits)
        marker_presence = flat_marker_logits.max(dim=-1).values.sigmoid()
        observed = logits[:, 1].sigmoid()
        free = logits[:, 2].sigmoid()
        blocked = logits[:, 3].sigmoid()
        return marker_probs, marker_presence, observed, free, blocked

    def _initialize_spatial_frontier_memory_detector(self) -> None:
        """Initialize the trainable RGB-to-memory evidence detector."""

        if self.spatial_frontier_memory_detector is None:
            return
        if self.spatial_frontier_memory_detector_init == "random":
            return
        if isinstance(self.spatial_frontier_memory_detector, nn.Sequential):
            final = self.spatial_frontier_memory_detector[-1]
            if not isinstance(final, nn.Conv2d):
                raise RuntimeError("expected final frontier detector layer to be Conv2d")
            if self.spatial_frontier_memory_detector_init == "direct_rgb":
                raise RuntimeError("direct_rgb init is unsupported for MLP detector")
            with torch.no_grad():
                final.bias[0] = -4.0
                final.bias[1] = 4.0
            return
        with torch.no_grad():
            self.spatial_frontier_memory_detector.weight.zero_()
            self.spatial_frontier_memory_detector.bias.zero_()
            if self.spatial_frontier_memory_detector_init == "neutral":
                self.spatial_frontier_memory_detector.bias[0] = -4.0
                self.spatial_frontier_memory_detector.bias[1] = 4.0
                return
            weight = self.spatial_frontier_memory_detector.weight
            bias = self.spatial_frontier_memory_detector.bias
            weight[0, :, 0, 0] = weight.new_tensor([-4.0, 8.0, -4.0])
            bias[0] = -4.0
            bias[1] = 6.0
            weight[2, :, 0, 0] = weight.new_tensor([2.0, 2.0, 2.0])
            bias[2] = -2.0
            weight[3, :, 0, 0] = weight.new_tensor([-2.0, -2.0, -2.0])
            bias[3] = 2.0

    def roll_spatial_frontier_maps(
        self,
        marker_belief: torch.Tensor,
        observed: torch.Tensor,
        free: torch.Tensor,
        blocked: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform learned marker/frontier maps through egocentric actions."""

        marker_belief = self.roll_spatial_marker_belief(marker_belief, actions)
        observed, rolled_blocked = self.roll_online_frontier_maps(
            observed,
            blocked,
            actions,
        )
        free, _ = self.roll_online_frontier_maps(free, blocked, actions)
        return marker_belief, observed, free, rolled_blocked

    def spatial_frontier_marker_update_weight(
        self,
        marker_presence: torch.Tensor,
    ) -> torch.Tensor:
        """Return the marker-memory write strength for one observation."""

        return (
            (marker_presence - self.spatial_frontier_marker_update_threshold)
            / self.spatial_frontier_marker_update_width
        ).clamp(0.0, 1.0)

    def update_spatial_frontier_memory_from_observation(
        self,
        marker_belief: torch.Tensor,
        marker_mass: torch.Tensor,
        observed: torch.Tensor,
        free: torch.Tensor,
        blocked: torch.Tensor,
        vision: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fuse one RGB crop into learned egocentric marker/frontier memory."""

        if marker_belief.ndim != 3:
            raise ValueError("marker_belief must have shape (B, H, W)")
        (
            obs_marker,
            obs_marker_presence,
            obs_observed,
            obs_free,
            obs_blocked,
        ) = self.spatial_frontier_observation_maps(vision)
        obs_marker = self.center_insert_view_map(obs_marker)
        obs_observed = self.center_insert_view_map(obs_observed)
        obs_free = self.center_insert_view_map(obs_free)
        obs_blocked = self.center_insert_view_map(obs_blocked)
        update = self.spatial_frontier_marker_update_weight(obs_marker_presence)
        update_map = update[:, None, None]
        marker_belief = (1.0 - update_map) * marker_belief + update_map * obs_marker
        marker_mass = 1.0 - ((1.0 - marker_mass) * (1.0 - update))
        observed = torch.maximum(observed, obs_observed)
        free = torch.maximum(free, obs_free)
        blocked = torch.maximum(blocked, obs_blocked)
        return marker_belief, marker_mass, observed, free, blocked

    def step_spatial_frontier_memory(
        self,
        marker_belief: torch.Tensor,
        marker_mass: torch.Tensor,
        observed: torch.Tensor,
        free: torch.Tensor,
        blocked: torch.Tensor,
        actions: torch.Tensor,
        next_vision: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance persistent learned memory through actions and an optional crop."""

        if marker_belief.ndim == 2:
            marker_belief_map = marker_belief.reshape(
                marker_belief.shape[0],
                self.spatial_memory_size,
                self.spatial_memory_size,
            )
        elif marker_belief.ndim == 3:
            marker_belief_map = marker_belief
        else:
            raise ValueError("marker_belief must have shape (B, cells) or (B, H, W)")
        marker_belief_map, observed, free, blocked = self.roll_spatial_frontier_maps(
            marker_belief_map,
            observed,
            free,
            blocked,
            actions,
        )
        if next_vision is not None:
            (
                marker_belief_map,
                marker_mass,
                observed,
                free,
                blocked,
            ) = self.update_spatial_frontier_memory_from_observation(
                marker_belief_map,
                marker_mass,
                observed,
                free,
                blocked,
                next_vision,
            )
        marker_belief = marker_belief_map.flatten(start_dim=1)
        marker_belief = marker_belief / marker_belief.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)
        return (
            marker_belief,
            marker_mass.clamp(0.0, 1.0),
            observed.clamp(0.0, 1.0),
            free.clamp(0.0, 1.0),
            blocked.clamp(0.0, 1.0),
        )

    def spatial_frontier_memory_start_maps(
        self,
        history_vision: torch.Tensor | None,
        history_actions: torch.Tensor | None,
        start_vision: torch.Tensor | None = None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Maintain learned marker/explored/free/blocked maps through history."""

        cells = self.spatial_memory_size * self.spatial_memory_size
        marker_belief = torch.full(
            (batch_size, self.spatial_memory_size, self.spatial_memory_size),
            1.0 / float(cells),
            device=device,
            dtype=dtype,
        )
        marker_mass = torch.zeros(batch_size, device=device, dtype=dtype)
        observed = torch.zeros(
            batch_size,
            self.spatial_memory_size,
            self.spatial_memory_size,
            device=device,
            dtype=dtype,
        )
        free = torch.zeros_like(observed)
        blocked = torch.zeros_like(observed)
        if (
            history_vision is not None
            and history_actions is not None
            and history_vision.shape[1] > 0
        ):
            if history_vision.ndim != 5:
                raise ValueError(
                    "history_vision must have shape (B, K, C, H, W), got "
                    f"{tuple(history_vision.shape)}"
                )
            if history_actions.ndim != 3:
                raise ValueError(
                    "history_actions must have shape (B, K, action_dim), got "
                    f"{tuple(history_actions.shape)}"
                )
            if history_vision.shape[:2] != history_actions.shape[:2]:
                raise ValueError("history observations and actions must share B,K")
            if history_vision.shape[0] != batch_size:
                raise ValueError("history batch size must match batch_size")
            for step in range(history_vision.shape[1]):
                (
                    marker_belief,
                    marker_mass,
                    observed,
                    free,
                    blocked,
                ) = self.update_spatial_frontier_memory_from_observation(
                    marker_belief,
                    marker_mass,
                    observed,
                    free,
                    blocked,
                    history_vision[:, step].to(device=device, dtype=dtype),
                )
                marker_belief, observed, free, blocked = (
                    self.roll_spatial_frontier_maps(
                        marker_belief,
                        observed,
                        free,
                        blocked,
                        history_actions[:, step : step + 1].to(
                            device=device,
                            dtype=dtype,
                        ),
                    )
                )
        if start_vision is not None:
            (
                marker_belief,
                marker_mass,
                observed,
                free,
                blocked,
            ) = self.update_spatial_frontier_memory_from_observation(
                marker_belief,
                marker_mass,
                observed,
                free,
                blocked,
                start_vision.to(device=device, dtype=dtype),
            )
        marker_belief = marker_belief.flatten(start_dim=1)
        marker_belief = marker_belief / marker_belief.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)
        return (
            marker_belief,
            marker_mass.clamp(0.0, 1.0),
            observed.clamp(0.0, 1.0),
            free.clamp(0.0, 1.0),
            blocked.clamp(0.0, 1.0),
        )

    def spatial_frontier_memory_score(
        self,
        marker_belief: torch.Tensor,
        marker_mass: torch.Tensor,
        observed: torch.Tensor,
        blocked: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidates from learned marker/frontier memory maps."""

        marker_score, _ = self.spatial_marker_memory_score(
            marker_belief,
            marker_mass,
            actions,
        )
        frontier_observed = observed
        frontier_blocked = blocked
        frontier_score = actions.new_zeros(actions.shape[0])
        footprint = self.current_view_footprint_like(frontier_observed)
        center = frontier_observed.shape[-1] // 2
        for step in range(actions.shape[1]):
            ahead_blocked = frontier_blocked[:, center - 1, center].clamp(0.0, 1.0)
            frontier_score = (
                frontier_score
                - self.spatial_frontier_collision_penalty
                * actions[:, step, 0]
                * ahead_blocked
            )
            frontier_observed, frontier_blocked = self.roll_online_frontier_maps(
                frontier_observed,
                frontier_blocked,
                actions[:, step : step + 1],
            )
            novel = (footprint - frontier_observed).clamp(0.0, 1.0)
            frontier_score = (
                frontier_score
                + self.spatial_frontier_novelty_reward
                * novel.flatten(start_dim=1).sum(dim=-1)
            )
            frontier_observed = torch.maximum(frontier_observed, footprint)
        if self.spatial_frontier_memory_gate_mode == "linear":
            return marker_score + (1.0 - marker_mass) * frontier_score
        marker_gate = (
            (marker_mass - self.spatial_frontier_marker_gate_threshold)
            / self.spatial_frontier_marker_gate_width
        ).clamp(0.0, 1.0)
        raw_marker_score = marker_score / marker_mass.clamp_min(1e-6)
        return marker_gate * raw_marker_score + (1.0 - marker_gate) * frontier_score

    def spatial_marker_observation_probs(
        self,
        vision: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return learned marker observation probability and presence."""

        if self.spatial_marker_memory_detector is None:
            raise RuntimeError("spatial marker memory detector is disabled")
        if vision.ndim != 4:
            raise ValueError(
                "vision must have shape (B, C, H, W), got "
                f"{tuple(vision.shape)}"
            )
        if vision.shape[1:] != (3, self.view_size, self.view_size):
            raise ValueError(
                "vision must have shape "
                f"(B, 3, {self.view_size}, {self.view_size}), got "
                f"{tuple(vision.shape)}"
            )
        logits = self.spatial_marker_memory_detector(vision).squeeze(1)
        flat_logits = logits.flatten(start_dim=1)
        probs = flat_logits.softmax(dim=-1).reshape_as(logits)
        presence = flat_logits.max(dim=-1).values.sigmoid()
        return probs, presence

    def roll_spatial_marker_belief(
        self,
        belief: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Shift/rotate a marker belief map through egocentric actions."""

        if belief.ndim != 3:
            raise ValueError("belief must have shape (B, H, W)")
        if actions.ndim != 3:
            raise ValueError("actions must have shape (B, T, action_dim)")
        if belief.shape[0] != actions.shape[0]:
            raise ValueError("belief and actions batch sizes differ")
        if actions.shape[-1] < 4:
            raise ValueError("actions must include forward/left/right/hold logits")
        current = belief
        for step in range(actions.shape[1]):
            forward = torch.zeros_like(current)
            forward[:, 1:, :] = current[:, :-1, :]
            turn_left = torch.rot90(current, k=1, dims=(-2, -1))
            turn_right = torch.rot90(current, k=-1, dims=(-2, -1))
            candidates = torch.stack(
                [forward, turn_left, turn_right, current],
                dim=1,
            )
            weights = actions[:, step, :4].to(dtype=current.dtype)[:, :, None, None]
            current = (candidates * weights).sum(dim=1)
        return current

    def spatial_marker_memory_start_belief(
        self,
        history_vision: torch.Tensor | None,
        history_actions: torch.Tensor | None,
        start_vision: torch.Tensor | None = None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Maintain a learned marker belief map through observed actions."""

        cells = self.spatial_memory_size * self.spatial_memory_size
        belief = torch.full(
            (batch_size, self.spatial_memory_size, self.spatial_memory_size),
            1.0 / float(cells),
            device=device,
            dtype=dtype,
        )
        mass = torch.zeros(batch_size, device=device, dtype=dtype)
        if (
            history_vision is not None
            and history_actions is not None
            and history_vision.shape[1] > 0
        ):
            if history_vision.ndim != 5:
                raise ValueError(
                    "history_vision must have shape (B, K, C, H, W), got "
                    f"{tuple(history_vision.shape)}"
                )
            if history_actions.ndim != 3:
                raise ValueError(
                    "history_actions must have shape (B, K, action_dim), got "
                    f"{tuple(history_actions.shape)}"
                )
            if history_vision.shape[:2] != history_actions.shape[:2]:
                raise ValueError("history observations and actions must share B,K")
            if history_vision.shape[0] != batch_size:
                raise ValueError("history batch size must match batch_size")
            for step in range(history_vision.shape[1]):
                obs_probs, presence = self.spatial_marker_observation_probs(
                    history_vision[:, step].to(device=device, dtype=dtype)
                )
                obs_probs = self.center_insert_view_map(obs_probs)
                presence_map = presence[:, None, None]
                belief = (1.0 - presence_map) * belief + presence_map * obs_probs
                mass = 1.0 - ((1.0 - mass) * (1.0 - presence))
                belief = self.roll_spatial_marker_belief(
                    belief,
                    history_actions[:, step : step + 1].to(
                        device=device,
                        dtype=dtype,
                    ),
                )
        if start_vision is not None:
            obs_probs, presence = self.spatial_marker_observation_probs(
                start_vision.to(device=device, dtype=dtype)
            )
            obs_probs = self.center_insert_view_map(obs_probs)
            presence_map = presence[:, None, None]
            belief = (1.0 - presence_map) * belief + presence_map * obs_probs
            mass = 1.0 - ((1.0 - mass) * (1.0 - presence))
        belief = belief.flatten(start_dim=1)
        belief = belief / belief.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return belief, mass.clamp(0.0, 1.0)

    def spatial_marker_memory_support(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return normalized egocentric deltas for every memory-grid cell."""

        center = self.spatial_memory_size // 2
        cells = [
            (center - row, col - center)
            for row in range(self.spatial_memory_size)
            for col in range(self.spatial_memory_size)
        ]
        return torch.tensor(cells, device=device, dtype=dtype) / float(
            max(self.view_size, 1)
        )

    def spatial_marker_memory_score(
        self,
        start_belief: torch.Tensor,
        mass: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score candidates by learned belief mass that reaches the ego origin."""

        if start_belief.ndim != 2:
            raise ValueError("start_belief must have shape (B, cells)")
        if mass.shape != (start_belief.shape[0],):
            raise ValueError("mass must have shape (B,)")
        support = self.spatial_marker_memory_support(
            device=start_belief.device,
            dtype=start_belief.dtype,
        )
        score_belief = start_belief
        if self.spatial_marker_memory_score_temperature != 1.0:
            score_belief = torch.softmax(
                score_belief.clamp_min(1e-8).log()
                / self.spatial_marker_memory_score_temperature,
                dim=-1,
            )
        final_delta = self.rollout_marker_support(support, actions)
        distances = final_delta.abs().sum(dim=-1)
        claim_mask = distances <= (0.5 / float(max(self.view_size, 1)))
        claim_probability = (
            score_belief * claim_mask.to(dtype=score_belief.dtype)
        ).sum(dim=-1)
        expected_distance_cells = (
            score_belief * distances * float(max(self.view_size, 1))
        ).sum(dim=-1)
        score = (100.0 * claim_probability) - expected_distance_cells
        return mass * score, final_delta

    def spatial_marker_memory_target_indices(
        self,
        valid_mask: torch.Tensor,
        start_delta_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert normalized start deltas into memory-grid cell indices."""

        if valid_mask.ndim != 1:
            raise ValueError("valid_mask must have shape (B,)")
        if start_delta_targets.shape != (valid_mask.shape[0], 2):
            raise ValueError("start_delta_targets must have shape (B, 2)")
        center = self.spatial_memory_size // 2
        cell_delta = torch.round(
            start_delta_targets * float(max(self.view_size, 1))
        ).to(dtype=torch.long)
        ahead = cell_delta[:, 0]
        lateral = cell_delta[:, 1]
        row = center - ahead
        col = center + lateral
        in_bounds = (
            (row >= 0)
            & (row < self.spatial_memory_size)
            & (col >= 0)
            & (col < self.spatial_memory_size)
        )
        mask = valid_mask.to(dtype=torch.bool) & in_bounds
        target = (
            row.clamp(0, self.spatial_memory_size - 1) * self.spatial_memory_size
        ) + col.clamp(
            0,
            self.spatial_memory_size - 1,
        )
        return mask, target

    def seed_decision_tokens(
        self,
        spatial_tokens: torch.Tensor,
        memory_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Initialize learned decision tokens from the current image-token state."""

        if spatial_tokens.ndim != 3:
            raise ValueError("spatial_tokens must have shape (B, N, D)")
        if memory_context is None:
            memory_context = spatial_tokens.new_zeros(
                spatial_tokens.shape[0],
                self.latent_dim,
            )
        if memory_context.shape != (spatial_tokens.shape[0], self.latent_dim):
            raise ValueError(
                "memory_context must have shape "
                f"({spatial_tokens.shape[0]}, {self.latent_dim})"
            )
        source_summary = torch.cat(
            [self.source_token_summary(spatial_tokens), memory_context],
            dim=-1,
        )
        seeded = (
            self.decision_seed_query
            + self.decision_seed_from_image(source_summary)[:, None]
        )
        if self.decision_target_geometry == "normalized":
            return normalize_spatial_tokens(seeded)
        return seeded

    def encode_decision_targets(self, consequence_targets: torch.Tensor) -> torch.Tensor:
        """Map privileged consequence factors to fixed decision-token targets."""

        if consequence_targets.ndim != 3:
            raise ValueError("consequence_targets must have shape (B, H, C)")
        if consequence_targets.shape[-1] != self.consequence_dim:
            raise ValueError(
                f"expected {self.consequence_dim} consequence factors, got "
                f"{consequence_targets.shape[-1]}"
            )
        batch, horizon = consequence_targets.shape[:2]
        raw = self.decision_target_projector(consequence_targets).reshape(
            batch,
            horizon,
            self.decision_token_count,
            self.latent_dim,
        )
        if self.decision_target_geometry == "normalized":
            return normalize_spatial_tokens(raw).detach()
        return raw.mul(self.decision_target_scale).detach()

    def predict_utility(
        self,
        predicted_tokens: torch.Tensor,
        memory_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict scalar sequence utility from final imagined tokens."""

        if predicted_tokens.ndim != 4:
            raise ValueError("predicted_tokens must have shape (B, H, N, D)")
        final_pooled = predicted_tokens[:, -1].mean(dim=1)
        if memory_context is None:
            memory_context = final_pooled.new_zeros(final_pooled.shape)
        if memory_context.shape != final_pooled.shape:
            raise ValueError(
                "memory_context must have shape "
                f"{tuple(final_pooled.shape)}, got {tuple(memory_context.shape)}"
            )
        return self.utility_head(
            torch.cat([final_pooled, memory_context], dim=-1)
        ).squeeze(-1)

    def predict_consequences(self, predicted_tokens: torch.Tensor) -> torch.Tensor:
        """Predict per-step consequence factors from imagined tokens."""

        if predicted_tokens.ndim != 4:
            raise ValueError("predicted_tokens must have shape (B, H, N, D)")
        pooled = predicted_tokens.mean(dim=2)
        return self.consequence_head(pooled)

    def candidate_score_features(
        self,
        predicted_tokens: torch.Tensor,
        consequence_prediction: torch.Tensor,
        actions: torch.Tensor | None = None,
        start_tokens: torch.Tensor | None = None,
        source_tokens: torch.Tensor | None = None,
        memory_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the feature vector used to score a candidate sequence."""

        if predicted_tokens.ndim != 4:
            raise ValueError("predicted_tokens must have shape (B, H, N, D)")
        if consequence_prediction.ndim != 3:
            raise ValueError("consequence_prediction must have shape (B, H, C)")
        if predicted_tokens.shape[:2] != consequence_prediction.shape[:2]:
            raise ValueError("predicted tokens and consequences must share B,H")
        if consequence_prediction.shape[-1] != self.consequence_dim:
            raise ValueError(
                f"expected {self.consequence_dim} consequence factors, got "
                f"{consequence_prediction.shape[-1]}"
            )
        if actions is None:
            action_features = predicted_tokens.new_zeros(
                predicted_tokens.shape[0],
                self.candidate_action_feature_dim,
            )
        else:
            if actions.ndim != 3:
                raise ValueError("actions must have shape (B, H, action_dim)")
            if actions.shape[:2] != predicted_tokens.shape[:2]:
                raise ValueError("actions and predicted tokens must share B,H")
            if actions.shape[-1] != self.action_dim:
                raise ValueError(
                    f"expected action dim {self.action_dim}, got {actions.shape[-1]}"
                )
            if self.candidate_action_encoder is None:
                action_features = torch.cat(
                    [actions[:, 0], actions[:, -1], actions.mean(dim=1)],
                    dim=-1,
                )
            else:
                _, hidden = self.candidate_action_encoder(actions)
                action_features = hidden[-1]
        if start_tokens is None:
            start_features = predicted_tokens.new_zeros(
                predicted_tokens.shape[0],
                self.latent_dim,
            )
        else:
            if start_tokens.ndim != 3:
                raise ValueError("start_tokens must have shape (B, N, D)")
            if start_tokens.shape[0] != predicted_tokens.shape[0]:
                raise ValueError("start_tokens and predictions batch sizes differ")
            if start_tokens.shape[-1] != self.latent_dim:
                raise ValueError(
                    f"expected start token dim {self.latent_dim}, got "
                    f"{start_tokens.shape[-1]}"
                )
            start_features = start_tokens.mean(dim=1)
        if source_tokens is None:
            source_features = predicted_tokens.new_zeros(
                predicted_tokens.shape[0],
                3 * self.latent_dim,
            )
        else:
            if source_tokens.ndim != 3:
                raise ValueError("source_tokens must have shape (B, N, D)")
            if source_tokens.shape[0] != predicted_tokens.shape[0]:
                raise ValueError("source_tokens and predictions batch sizes differ")
            if source_tokens.shape[-1] != self.latent_dim:
                raise ValueError(
                    f"expected source token dim {self.latent_dim}, got "
                    f"{source_tokens.shape[-1]}"
                )
            center_index = min(
                (self.view_size // 2) * self.view_size + (self.view_size // 2),
                source_tokens.shape[1] - 1,
            )
            source_features = torch.cat(
                [
                    source_tokens.mean(dim=1),
                    source_tokens[:, 0],
                    source_tokens[:, center_index],
                ],
                dim=-1,
            )
        if memory_context is None:
            memory_features = predicted_tokens.new_zeros(
                predicted_tokens.shape[0],
                self.latent_dim,
            )
        else:
            if memory_context.shape != (predicted_tokens.shape[0], self.latent_dim):
                raise ValueError(
                    "memory_context must have shape "
                    f"({predicted_tokens.shape[0]}, {self.latent_dim})"
                )
            memory_features = memory_context
        final_tokens = predicted_tokens[:, -1].mean(dim=1)
        first_consequence = consequence_prediction[:, 0]
        final_consequence = consequence_prediction[:, -1]
        mean_consequence = consequence_prediction.mean(dim=1)
        features = torch.cat(
            [
                final_tokens,
                first_consequence,
                final_consequence,
                mean_consequence,
                action_features,
                start_features,
                source_features,
                memory_features,
            ],
            dim=-1,
        )
        return features

    def predict_candidate_score(
        self,
        predicted_tokens: torch.Tensor,
        consequence_prediction: torch.Tensor,
        actions: torch.Tensor | None = None,
        start_tokens: torch.Tensor | None = None,
        source_tokens: torch.Tensor | None = None,
        memory_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score a complete candidate sequence from imagined decision factors."""

        features = self.candidate_score_features(
            predicted_tokens,
            consequence_prediction,
            actions,
            start_tokens,
            source_tokens,
            memory_context,
        )
        score = self.candidate_score_head(features).squeeze(-1)
        if self.candidate_claim_head is None:
            claim_logit = score.new_zeros(score.shape)
        else:
            claim_logit = self.candidate_claim_head(features).squeeze(-1)
        marker_score = self.predict_candidate_marker_memory_score(features)
        return (
            score
            + self.candidate_score_claim_logit_weight * claim_logit
            + self.candidate_marker_memory_score_weight * marker_score
        )

    def predict_candidate_marker_memory_score(self, features: torch.Tensor) -> torch.Tensor:
        """Score a candidate by learned remembered-marker residual."""

        if self.candidate_marker_memory_delta_head is None:
            return features.new_zeros(features.shape[0])
        delta = self.candidate_marker_memory_delta_head(features)
        claim_logit = self.candidate_marker_memory_claim_head(features).squeeze(-1)
        distance_score = -delta.abs().sum(dim=-1)
        if self.candidate_marker_memory_score_mode == "distance":
            return distance_score
        return claim_logit + distance_score

    def structured_marker_memory_features(
        self,
        source_tokens: torch.Tensor,
        memory_context: torch.Tensor,
    ) -> torch.Tensor:
        """Build source-frame features for the structured marker-memory head."""

        return torch.cat(
            [self.source_token_summary(source_tokens), memory_context],
            dim=-1,
        )

    def rollout_marker_delta(
        self,
        start_delta: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Roll a normalized egocentric marker delta through candidate actions."""

        if start_delta.ndim != 2 or start_delta.shape[-1] != 2:
            raise ValueError("start_delta must have shape (B, 2)")
        if actions.ndim != 3:
            raise ValueError("actions must have shape (B, H, action_dim)")
        if actions.shape[0] != start_delta.shape[0]:
            raise ValueError("start_delta and actions batch sizes differ")
        if actions.shape[-1] < 4:
            raise ValueError("actions must include forward/left/right/hold logits")
        delta = start_delta
        forward_step = 1.0 / float(max(self.view_size, 1))
        for step in range(actions.shape[1]):
            ahead = delta[:, 0]
            lateral = delta[:, 1]
            candidates = torch.stack(
                [
                    torch.stack([ahead - forward_step, lateral], dim=-1),
                    torch.stack([lateral, -ahead], dim=-1),
                    torch.stack([-lateral, ahead], dim=-1),
                    delta,
                ],
                dim=1,
            )
            weights = actions[:, step, :4].to(dtype=delta.dtype).unsqueeze(-1)
            delta = (candidates * weights).sum(dim=1)
        return delta

    def categorical_marker_memory_support(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return normalized egocentric support cells as ``(S, 2)``."""

        radius = self.categorical_marker_memory_radius
        cells = [
            (ahead, lateral)
            for ahead in range(-radius, radius + 1)
            for lateral in range(-radius, radius + 1)
        ]
        return torch.tensor(cells, device=device, dtype=dtype) / float(
            max(self.view_size, 1)
        )

    def rollout_marker_support(
        self,
        support_delta: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Roll every support-cell marker delta through candidate actions."""

        if support_delta.ndim != 2 or support_delta.shape[-1] != 2:
            raise ValueError("support_delta must have shape (S, 2)")
        if actions.ndim != 3:
            raise ValueError("actions must have shape (B, H, action_dim)")
        delta = support_delta[None].expand(actions.shape[0], -1, -1)
        forward_step = 1.0 / float(max(self.view_size, 1))
        for step in range(actions.shape[1]):
            ahead = delta[..., 0]
            lateral = delta[..., 1]
            candidates = torch.stack(
                [
                    torch.stack([ahead - forward_step, lateral], dim=-1),
                    torch.stack([lateral, -ahead], dim=-1),
                    torch.stack([-lateral, ahead], dim=-1),
                    delta,
                ],
                dim=2,
            )
            weights = actions[:, step, :4].to(dtype=delta.dtype)[:, None, :, None]
            delta = (candidates * weights).sum(dim=2)
        return delta

    def categorical_marker_memory_score(
        self,
        logits: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score candidates by belief mass that reaches the ego origin."""

        support = self.categorical_marker_memory_support(
            device=logits.device,
            dtype=logits.dtype,
        )
        final_delta = self.rollout_marker_support(support, actions)
        probs = logits.softmax(dim=-1)
        distances = final_delta.abs().sum(dim=-1)
        claim_mask = distances <= (0.5 / float(max(self.view_size, 1)))
        claim_probability = (probs * claim_mask.to(dtype=probs.dtype)).sum(dim=-1)
        expected_distance = (probs * distances).sum(dim=-1)
        return (100.0 * claim_probability) - expected_distance, final_delta

    def rollout(self, start_tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.project_predictions(self.predictor.rollout(start_tokens, actions))

    def rollout_decision(
        self,
        start_tokens: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        if self.decision_rollout_mode == "recurrent":
            if start_tokens.ndim != 3:
                raise ValueError("start_tokens must have shape (B, N, D)")
            if actions.ndim != 3:
                raise ValueError("actions must have shape (B, H, action_dim)")
            if start_tokens.shape[0] != actions.shape[0]:
                raise ValueError("start_tokens and actions batch sizes differ")
            pooled = start_tokens.mean(dim=1)
            initial = self.decision_recurrent_initial(pooled).unsqueeze(0)
            recurrent, _ = self.decision_recurrent(actions, initial)
            decoded = self.decision_recurrent_decoder(
                recurrent.reshape(-1, self.latent_dim)
            )
            decoded = decoded.reshape(
                actions.shape[0],
                actions.shape[1],
                self.decision_token_count,
                self.latent_dim,
            )
            if self.decision_recurrent_update == "absolute":
                return self.project_decision_predictions(decoded)
            predictions = []
            current = start_tokens
            for step in range(actions.shape[1]):
                current = self.project_decision_predictions(
                    current + decoded[:, step]
                )
                predictions.append(current)
            return torch.stack(predictions, dim=1)
        return self.project_decision_predictions(
            self.decision_predictor.rollout(start_tokens, actions)
        )

    @staticmethod
    def rollout_deltas(
        rollout_tokens: torch.Tensor,
        *,
        start_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Return free-running step deltas for a rollout token sequence."""

        if rollout_tokens.ndim != 4:
            raise ValueError("rollout_tokens must have shape (B, H, N, D)")
        if start_tokens.shape != rollout_tokens[:, 0].shape:
            raise ValueError("start_tokens must align with rollout token steps")
        previous = torch.cat([start_tokens[:, None], rollout_tokens[:, :-1]], dim=1)
        return rollout_tokens - previous

    def predict_step_sequence(self, current_tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predict each step independently from teacher-forced current tokens."""

        batch, horizon, num_tokens, dim = current_tokens.shape
        raw = self.predictor.predict_step(
            current_tokens.reshape(batch * horizon, num_tokens, dim),
            actions.reshape(batch * horizon, actions.shape[-1]),
        )
        return self.project_predictions(raw.reshape(batch, horizon, num_tokens, dim))

    def predict_decision_step_sequence(
        self,
        current_tokens: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict each decision-token step from supplied current decision tokens."""

        batch, horizon, num_tokens, dim = current_tokens.shape
        raw = self.decision_predictor.predict_step(
            current_tokens.reshape(batch * horizon, num_tokens, dim),
            actions.reshape(batch * horizon, actions.shape[-1]),
        )
        return self.project_decision_predictions(
            raw.reshape(batch, horizon, num_tokens, dim)
        )

    def predict_wrong_steps(self, current_tokens: torch.Tensor, wrong_actions: torch.Tensor) -> torch.Tensor:
        """Predict teacher-forced futures under hard-negative actions."""

        batch, horizon, num_tokens, dim = current_tokens.shape
        negatives = wrong_actions.shape[2]
        expanded = current_tokens[:, :, None].expand(-1, -1, negatives, -1, -1)
        raw = self.predictor.predict_step(
            expanded.reshape(batch * horizon * negatives, num_tokens, dim),
            wrong_actions.reshape(batch * horizon * negatives, wrong_actions.shape[-1]),
        )
        return self.project_predictions(
            raw.reshape(batch, horizon, negatives, num_tokens, dim)
        )

    def predict_wrong_decision_steps(
        self,
        current_tokens: torch.Tensor,
        wrong_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict teacher-forced decision futures under hard-negative actions."""

        batch, horizon, num_tokens, dim = current_tokens.shape
        negatives = wrong_actions.shape[2]
        expanded = current_tokens[:, :, None].expand(-1, -1, negatives, -1, -1)
        raw = self.decision_predictor.predict_step(
            expanded.reshape(batch * horizon * negatives, num_tokens, dim),
            wrong_actions.reshape(batch * horizon * negatives, wrong_actions.shape[-1]),
        )
        return self.project_decision_predictions(
            raw.reshape(batch, horizon, negatives, num_tokens, dim)
        )

    def rollout_wrong_actions(
        self,
        start_tokens: torch.Tensor,
        wrong_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Free-run each hard-negative action sequence from the same start."""

        if wrong_actions.ndim != 4:
            raise ValueError("wrong_actions must have shape (B, H, K, action_dim)")
        batch, horizon, negatives, action_dim = wrong_actions.shape
        expanded_start = start_tokens[:, None].expand(-1, negatives, -1, -1)
        flat_rollout = self.rollout(
            expanded_start.reshape(
                batch * negatives,
                start_tokens.shape[1],
                start_tokens.shape[2],
            ),
            wrong_actions.permute(0, 2, 1, 3).reshape(
                batch * negatives,
                horizon,
                action_dim,
            ),
        )
        return flat_rollout.reshape(
            batch,
            negatives,
            horizon,
            start_tokens.shape[1],
            start_tokens.shape[2],
        ).permute(0, 2, 1, 3, 4)

    def rollout_wrong_decision_actions(
        self,
        start_tokens: torch.Tensor,
        wrong_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Free-run each hard-negative action sequence through decision tokens."""

        if wrong_actions.ndim != 4:
            raise ValueError("wrong_actions must have shape (B, H, K, action_dim)")
        batch, horizon, negatives, action_dim = wrong_actions.shape
        expanded_start = start_tokens[:, None].expand(-1, negatives, -1, -1)
        flat_rollout = self.rollout_decision(
            expanded_start.reshape(
                batch * negatives,
                start_tokens.shape[1],
                start_tokens.shape[2],
            ),
            wrong_actions.permute(0, 2, 1, 3).reshape(
                batch * negatives,
                horizon,
                action_dim,
            ),
        )
        return flat_rollout.reshape(
            batch,
            negatives,
            horizon,
            start_tokens.shape[1],
            start_tokens.shape[2],
        ).permute(0, 2, 1, 3, 4)

    def forward(
        self,
        *,
        vision: torch.Tensor,
        history_vision: torch.Tensor | None = None,
        history_actions: torch.Tensor | None = None,
        actions: torch.Tensor,
        utility_targets: torch.Tensor,
        consequence_targets: torch.Tensor | None = None,
        candidate_marker_memory_valid_mask: torch.Tensor | None = None,
        candidate_marker_memory_delta_targets: torch.Tensor | None = None,
        candidate_marker_memory_claim_targets: torch.Tensor | None = None,
        candidate_marker_memory_score_targets: torch.Tensor | None = None,
        structured_marker_memory_valid_mask: torch.Tensor | None = None,
        structured_marker_memory_start_delta_targets: torch.Tensor | None = None,
        categorical_marker_memory_valid_mask: torch.Tensor | None = None,
        categorical_marker_memory_cell_targets: torch.Tensor | None = None,
        spatial_frontier_history_observation_targets: torch.Tensor | None = None,
        spatial_frontier_vision_observation_targets: torch.Tensor | None = None,
        utility_group_ids: torch.Tensor | None = None,
        utility_mask: torch.Tensor | None = None,
        wrong_actions: torch.Tensor | None = None,
        wrong_mask: torch.Tensor | None = None,
        non_hold_mask: torch.Tensor | None = None,
        zero_actions: torch.Tensor | None = None,
        return_latents: bool = False,
    ) -> dict[str, torch.Tensor]:
        if actions.ndim != 3:
            raise ValueError("actions must have shape (B, H, action_dim)")
        horizon = actions.shape[1]
        if vision.shape[1] != horizon + 1:
            raise ValueError("vision must contain start plus one frame per action")

        online_tokens = self.encode_seq(vision, target=False)
        target_pre = self.encode_seq(vision, target=True)
        target_normalized = normalize_spatial_tokens(target_pre)
        targets = target_normalized[:, 1:]
        previous_targets = target_normalized[:, :-1]
        rollout = self.rollout(online_tokens[:, 0], actions)
        teacher_forced = self.predict_step_sequence(previous_targets, actions)
        memory_context = self.encode_memory_context(
            history_vision,
            history_actions,
            batch_size=vision.shape[0],
            device=vision.device,
            dtype=online_tokens.dtype,
        )
        decision_start = self.seed_decision_tokens(
            online_tokens[:, 0],
            memory_context=memory_context,
        )
        decision_rollout = self.rollout_decision(decision_start, actions)

        prediction_loss = (rollout - targets).square().mean()
        teacher_forced_prediction_loss = (teacher_forced - targets).square().mean()
        target_delta = targets - previous_targets
        rollout_delta = self.rollout_deltas(
            rollout,
            start_tokens=previous_targets[:, 0],
        )
        teacher_forced_delta = teacher_forced - previous_targets
        rollout_delta_loss = (rollout_delta - target_delta).square().mean()
        teacher_forced_delta_loss = (
            teacher_forced_delta - target_delta
        ).square().mean()
        if consequence_targets is not None:
            decision_targets = self.encode_decision_targets(consequence_targets)
            decision_start_targets = self.encode_decision_targets(
                consequence_targets.new_zeros(
                    consequence_targets.shape[0],
                    1,
                    consequence_targets.shape[-1],
                )
            )[:, 0]
            decision_previous_targets = torch.cat(
                [decision_start_targets[:, None], decision_targets[:, :-1]],
                dim=1,
            )
            decision_teacher_inputs = torch.cat(
                [decision_start[:, None], decision_targets[:, :-1]],
                dim=1,
            )
            decision_teacher_forced = self.predict_decision_step_sequence(
                decision_teacher_inputs,
                actions,
            )
            decision_prediction_loss = (decision_rollout - decision_targets).square().mean()
            decision_teacher_forced_prediction_loss = (
                decision_teacher_forced - decision_targets
            ).square().mean()
            decision_target_delta = decision_targets - decision_previous_targets
            decision_rollout_delta = self.rollout_deltas(
                decision_rollout,
                start_tokens=decision_previous_targets[:, 0],
            )
            decision_teacher_forced_delta = (
                decision_teacher_forced - decision_previous_targets
            )
            decision_delta_loss = (
                decision_rollout_delta - decision_target_delta
            ).square().mean()
            decision_teacher_forced_delta_loss = (
                decision_teacher_forced_delta - decision_target_delta
            ).square().mean()
        else:
            decision_targets = None
            decision_previous_targets = None
            decision_teacher_inputs = None
            decision_teacher_forced = None
            decision_prediction_loss = rollout.new_zeros(())
            decision_teacher_forced_prediction_loss = rollout.new_zeros(())
            decision_target_delta = None
            decision_rollout_delta = None
            decision_teacher_forced_delta = None
            decision_delta_loss = rollout.new_zeros(())
            decision_teacher_forced_delta_loss = rollout.new_zeros(())
        consequence_tokens = (
            decision_rollout.detach()
            if self.detach_consequence_head_state
            else decision_rollout
        )
        consequence_prediction = self.predict_consequences(consequence_tokens)
        if consequence_targets is not None:
            if consequence_targets.shape != consequence_prediction.shape:
                raise ValueError(
                    "consequence_targets must have shape "
                    f"{tuple(consequence_prediction.shape)}, got "
                    f"{tuple(consequence_targets.shape)}"
                )
            binary_indices = list(CONSEQUENCE_BINARY_INDICES)
            scalar_indices = list(CONSEQUENCE_SCALAR_INDICES)
            consequence_binary_loss = F.binary_cross_entropy_with_logits(
                consequence_prediction[..., binary_indices],
                consequence_targets[..., binary_indices],
            )
            consequence_scalar_loss = F.mse_loss(
                consequence_prediction[..., scalar_indices],
                consequence_targets[..., scalar_indices],
            )
            consequence_loss = consequence_binary_loss + consequence_scalar_loss
        else:
            consequence_loss = rollout.new_zeros(())
            consequence_binary_loss = rollout.new_zeros(())
            consequence_scalar_loss = rollout.new_zeros(())
        utility_head_prediction = self.predict_utility(
            decision_rollout,
            memory_context=memory_context,
        )
        consequence_utility_prediction = (
            consequence_prediction[..., CONSEQUENCE_SCALAR_INDICES[-1]].mean(dim=1)
            * float(horizon + 5)
        )
        if self.candidate_score_gradient_mode == "full":
            candidate_score_tokens = decision_rollout
            candidate_score_start = decision_start
            candidate_score_consequences = consequence_prediction
            candidate_score_source = online_tokens[:, 0]
            candidate_score_memory = memory_context
        elif self.candidate_score_gradient_mode == "start_only":
            candidate_score_tokens = decision_rollout.detach()
            candidate_score_start = decision_start
            candidate_score_consequences = consequence_prediction.detach()
            candidate_score_source = online_tokens[:, 0]
            candidate_score_memory = memory_context
        else:
            candidate_score_tokens = decision_rollout.detach()
            candidate_score_start = decision_start.detach()
            candidate_score_consequences = consequence_prediction.detach()
            candidate_score_source = online_tokens[:, 0].detach()
            candidate_score_memory = memory_context.detach()
        if not self.candidate_score_source_tokens:
            candidate_score_source = None
        candidate_features = self.candidate_score_features(
            candidate_score_tokens,
            candidate_score_consequences,
            actions,
            candidate_score_start,
            candidate_score_source,
            candidate_score_memory,
        )
        candidate_score_value_prediction = self.candidate_score_head(
            candidate_features
        ).squeeze(-1)
        if consequence_targets is None:
            candidate_claim_targets = candidate_score_value_prediction.new_zeros(
                candidate_score_value_prediction.shape
            )
        else:
            candidate_claim_targets = consequence_targets[
                ..., CONSEQUENCE_REACHED_GOAL_INDEX
            ].amax(dim=1)
        if self.candidate_claim_head is None:
            candidate_claim_logit = candidate_score_value_prediction.new_zeros(
                candidate_score_value_prediction.shape
            )
            candidate_claim_loss = candidate_score_value_prediction.new_zeros(())
        else:
            candidate_claim_logit = self.candidate_claim_head(
                candidate_features
            ).squeeze(-1)
            candidate_claim_loss = F.binary_cross_entropy_with_logits(
                candidate_claim_logit,
                candidate_claim_targets,
            )
        online_marker_memory_score_prediction = self.online_marker_memory_score(
            history_vision,
            history_actions,
            actions,
            start_vision=vision[:, 0],
        )
        online_frontier_marker_score_prediction = (
            self.online_frontier_marker_score(
                history_vision,
                history_actions,
                actions,
                start_vision=vision[:, 0],
            )
        )
        if self.candidate_marker_memory_delta_head is None:
            candidate_marker_memory_delta_prediction = (
                candidate_score_value_prediction.new_zeros(
                    candidate_score_value_prediction.shape[0],
                    2,
                )
            )
            candidate_marker_memory_claim_logit = (
                candidate_score_value_prediction.new_zeros(
                    candidate_score_value_prediction.shape
                )
            )
            candidate_marker_memory_score_prediction = (
                candidate_score_value_prediction.new_zeros(
                    candidate_score_value_prediction.shape
                )
            )
            candidate_marker_memory_delta_loss = (
                candidate_score_value_prediction.new_zeros(())
            )
            candidate_marker_memory_claim_loss = (
                candidate_score_value_prediction.new_zeros(())
            )
            candidate_marker_memory_valid_count = (
                candidate_score_value_prediction.new_zeros(())
            )
            candidate_marker_memory_claim_target_mean = (
                candidate_score_value_prediction.new_zeros(())
            )
        else:
            candidate_marker_memory_delta_prediction = (
                self.candidate_marker_memory_delta_head(candidate_features)
            )
            candidate_marker_memory_claim_logit = (
                self.candidate_marker_memory_claim_head(candidate_features)
                .squeeze(-1)
            )
            candidate_marker_memory_score_prediction = (
                -candidate_marker_memory_delta_prediction.abs().sum(dim=-1)
            )
            if self.candidate_marker_memory_score_mode == "claim_plus_distance":
                candidate_marker_memory_score_prediction = (
                    candidate_marker_memory_claim_logit
                    + candidate_marker_memory_score_prediction
                )
            if (
                candidate_marker_memory_valid_mask is None
                or candidate_marker_memory_delta_targets is None
                or candidate_marker_memory_claim_targets is None
            ):
                candidate_marker_memory_delta_loss = (
                    candidate_score_value_prediction.new_zeros(())
                )
                candidate_marker_memory_claim_loss = (
                    candidate_score_value_prediction.new_zeros(())
                )
                candidate_marker_memory_valid_count = (
                    candidate_score_value_prediction.new_zeros(())
                )
                candidate_marker_memory_claim_target_mean = (
                    candidate_score_value_prediction.new_zeros(())
                )
            else:
                if candidate_marker_memory_valid_mask.shape != (
                    candidate_score_value_prediction.shape[0],
                ):
                    raise ValueError(
                        "candidate_marker_memory_valid_mask must have shape "
                        f"({candidate_score_value_prediction.shape[0]},)"
                    )
                if candidate_marker_memory_delta_targets.shape != (
                    candidate_score_value_prediction.shape[0],
                    2,
                ):
                    raise ValueError(
                        "candidate_marker_memory_delta_targets must have shape "
                        f"({candidate_score_value_prediction.shape[0]}, 2)"
                    )
                if candidate_marker_memory_claim_targets.shape != (
                    candidate_score_value_prediction.shape[0],
                ):
                    raise ValueError(
                        "candidate_marker_memory_claim_targets must have shape "
                        f"({candidate_score_value_prediction.shape[0]},)"
                    )
                valid_mask = candidate_marker_memory_valid_mask.to(
                    device=candidate_score_value_prediction.device,
                    dtype=torch.bool,
                )
                delta_targets = candidate_marker_memory_delta_targets.to(
                    device=candidate_score_value_prediction.device,
                    dtype=candidate_marker_memory_delta_prediction.dtype,
                )
                claim_targets = candidate_marker_memory_claim_targets.to(
                    device=candidate_score_value_prediction.device,
                    dtype=candidate_marker_memory_claim_logit.dtype,
                )
                candidate_marker_memory_valid_count = valid_mask.sum().to(
                    candidate_score_value_prediction.dtype
                )
                if valid_mask.any():
                    candidate_marker_memory_delta_loss = F.mse_loss(
                        candidate_marker_memory_delta_prediction[valid_mask],
                        delta_targets[valid_mask],
                    )
                    candidate_marker_memory_claim_loss = (
                        F.binary_cross_entropy_with_logits(
                            candidate_marker_memory_claim_logit[valid_mask],
                            claim_targets[valid_mask],
                        )
                    )
                    candidate_marker_memory_claim_target_mean = (
                        claim_targets[valid_mask].mean()
                    )
                else:
                    candidate_marker_memory_delta_loss = (
                        candidate_score_value_prediction.new_zeros(())
                    )
                    candidate_marker_memory_claim_loss = (
                        candidate_score_value_prediction.new_zeros(())
                    )
                    candidate_marker_memory_claim_target_mean = (
                        candidate_score_value_prediction.new_zeros(())
                    )
        candidate_marker_memory_loss = (
            self.candidate_marker_memory_delta_loss_weight
            * candidate_marker_memory_delta_loss
            + self.candidate_marker_memory_claim_loss_weight
            * candidate_marker_memory_claim_loss
        )
        if self.structured_marker_memory_start_head is None:
            structured_marker_memory_start_delta_prediction = (
                candidate_score_value_prediction.new_zeros(
                    candidate_score_value_prediction.shape[0],
                    2,
                )
            )
            structured_marker_memory_delta_prediction = (
                structured_marker_memory_start_delta_prediction
            )
            structured_marker_memory_score_prediction = (
                candidate_score_value_prediction.new_zeros(
                    candidate_score_value_prediction.shape
                )
            )
            structured_marker_memory_loss = (
                candidate_score_value_prediction.new_zeros(())
            )
            structured_marker_memory_start_delta_loss = (
                candidate_score_value_prediction.new_zeros(())
            )
            structured_marker_memory_final_delta_loss = (
                candidate_score_value_prediction.new_zeros(())
            )
            structured_marker_memory_valid_count = (
                candidate_score_value_prediction.new_zeros(())
            )
        else:
            structured_source_tokens = (
                online_tokens[:, 0]
                if self.candidate_score_gradient_mode in ("full", "start_only")
                else online_tokens[:, 0].detach()
            )
            structured_memory_context = (
                memory_context
                if self.candidate_score_gradient_mode in ("full", "start_only")
                else memory_context.detach()
            )
            structured_features = self.structured_marker_memory_features(
                structured_source_tokens,
                structured_memory_context,
            )
            structured_marker_memory_start_delta_prediction = (
                self.structured_marker_memory_start_head(structured_features)
            )
            structured_marker_memory_delta_prediction = self.rollout_marker_delta(
                structured_marker_memory_start_delta_prediction,
                actions,
            )
            structured_marker_memory_score_prediction = (
                -structured_marker_memory_delta_prediction.abs().sum(dim=-1)
            )
            if (
                structured_marker_memory_valid_mask is None
                or structured_marker_memory_start_delta_targets is None
                or candidate_marker_memory_valid_mask is None
                or candidate_marker_memory_delta_targets is None
            ):
                structured_marker_memory_start_delta_loss = (
                    candidate_score_value_prediction.new_zeros(())
                )
                structured_marker_memory_final_delta_loss = (
                    candidate_score_value_prediction.new_zeros(())
                )
                structured_marker_memory_valid_count = (
                    candidate_score_value_prediction.new_zeros(())
                )
            else:
                if structured_marker_memory_valid_mask.shape != (
                    candidate_score_value_prediction.shape[0],
                ):
                    raise ValueError(
                        "structured_marker_memory_valid_mask must have shape "
                        f"({candidate_score_value_prediction.shape[0]},)"
                    )
                if structured_marker_memory_start_delta_targets.shape != (
                    candidate_score_value_prediction.shape[0],
                    2,
                ):
                    raise ValueError(
                        "structured_marker_memory_start_delta_targets must have "
                        f"shape ({candidate_score_value_prediction.shape[0]}, 2)"
                    )
                start_valid_mask = structured_marker_memory_valid_mask.to(
                    device=candidate_score_value_prediction.device,
                    dtype=torch.bool,
                )
                start_delta_targets = (
                    structured_marker_memory_start_delta_targets.to(
                        device=candidate_score_value_prediction.device,
                        dtype=structured_marker_memory_start_delta_prediction.dtype,
                    )
                )
                final_valid_mask = candidate_marker_memory_valid_mask.to(
                    device=candidate_score_value_prediction.device,
                    dtype=torch.bool,
                )
                final_delta_targets = candidate_marker_memory_delta_targets.to(
                    device=candidate_score_value_prediction.device,
                    dtype=structured_marker_memory_delta_prediction.dtype,
                )
                structured_marker_memory_valid_count = start_valid_mask.sum().to(
                    candidate_score_value_prediction.dtype
                )
                if start_valid_mask.any():
                    structured_marker_memory_start_delta_loss = F.mse_loss(
                        structured_marker_memory_start_delta_prediction[
                            start_valid_mask
                        ],
                        start_delta_targets[start_valid_mask],
                    )
                else:
                    structured_marker_memory_start_delta_loss = (
                        candidate_score_value_prediction.new_zeros(())
                    )
                if final_valid_mask.any():
                    structured_marker_memory_final_delta_loss = F.mse_loss(
                        structured_marker_memory_delta_prediction[final_valid_mask],
                        final_delta_targets[final_valid_mask],
                    )
                else:
                    structured_marker_memory_final_delta_loss = (
                        candidate_score_value_prediction.new_zeros(())
                    )
            structured_marker_memory_loss = (
                structured_marker_memory_start_delta_loss
                + structured_marker_memory_final_delta_loss
            )
        if self.categorical_marker_memory_logits_head is None:
            categorical_marker_memory_logits = (
                candidate_score_value_prediction.new_zeros(
                    candidate_score_value_prediction.shape[0],
                    self.categorical_marker_memory_cells,
                )
            )
            categorical_marker_memory_score_prediction = (
                candidate_score_value_prediction.new_zeros(
                    candidate_score_value_prediction.shape
                )
            )
            categorical_marker_memory_loss = (
                candidate_score_value_prediction.new_zeros(())
            )
            categorical_marker_memory_valid_count = (
                candidate_score_value_prediction.new_zeros(())
            )
            categorical_marker_memory_target_mean = (
                candidate_score_value_prediction.new_zeros(())
            )
        else:
            categorical_source_tokens = (
                online_tokens[:, 0]
                if self.candidate_score_gradient_mode in ("full", "start_only")
                else online_tokens[:, 0].detach()
            )
            categorical_memory_context = (
                memory_context
                if self.candidate_score_gradient_mode in ("full", "start_only")
                else memory_context.detach()
            )
            categorical_features = self.structured_marker_memory_features(
                categorical_source_tokens,
                categorical_memory_context,
            )
            categorical_marker_memory_logits = (
                self.categorical_marker_memory_logits_head(categorical_features)
            )
            categorical_marker_memory_score_prediction, _ = (
                self.categorical_marker_memory_score(
                    categorical_marker_memory_logits,
                    actions,
                )
            )
            if (
                categorical_marker_memory_valid_mask is None
                or categorical_marker_memory_cell_targets is None
            ):
                categorical_marker_memory_loss = (
                    candidate_score_value_prediction.new_zeros(())
                )
                categorical_marker_memory_valid_count = (
                    candidate_score_value_prediction.new_zeros(())
                )
                categorical_marker_memory_target_mean = (
                    candidate_score_value_prediction.new_zeros(())
                )
            else:
                if categorical_marker_memory_valid_mask.shape != (
                    candidate_score_value_prediction.shape[0],
                ):
                    raise ValueError(
                        "categorical_marker_memory_valid_mask must have shape "
                        f"({candidate_score_value_prediction.shape[0]},)"
                    )
                if categorical_marker_memory_cell_targets.shape != (
                    candidate_score_value_prediction.shape[0],
                ):
                    raise ValueError(
                        "categorical_marker_memory_cell_targets must have shape "
                        f"({candidate_score_value_prediction.shape[0]},)"
                    )
                categorical_valid_mask = categorical_marker_memory_valid_mask.to(
                    device=candidate_score_value_prediction.device,
                    dtype=torch.bool,
                )
                categorical_targets = categorical_marker_memory_cell_targets.to(
                    device=candidate_score_value_prediction.device,
                    dtype=torch.long,
                )
                categorical_marker_memory_valid_count = (
                    categorical_valid_mask.sum().to(
                        candidate_score_value_prediction.dtype
                    )
                )
                if categorical_valid_mask.any():
                    categorical_marker_memory_loss = F.cross_entropy(
                        categorical_marker_memory_logits[categorical_valid_mask],
                        categorical_targets[categorical_valid_mask],
                    )
                    categorical_marker_memory_target_mean = (
                        categorical_targets[categorical_valid_mask]
                        .to(candidate_score_value_prediction.dtype)
                        .mean()
                    )
                else:
                    categorical_marker_memory_loss = (
                        candidate_score_value_prediction.new_zeros(())
                    )
                    categorical_marker_memory_target_mean = (
                        candidate_score_value_prediction.new_zeros(())
                    )
        if self.spatial_marker_memory_detector is None:
            spatial_marker_memory_start_belief = (
                candidate_score_value_prediction.new_zeros(
                    candidate_score_value_prediction.shape[0],
                    self.spatial_memory_size * self.spatial_memory_size,
                )
            )
            spatial_marker_memory_mass = candidate_score_value_prediction.new_zeros(
                candidate_score_value_prediction.shape
            )
            spatial_marker_memory_score_prediction = (
                candidate_score_value_prediction.new_zeros(
                    candidate_score_value_prediction.shape
                )
            )
            spatial_marker_memory_loss = (
                candidate_score_value_prediction.new_zeros(())
            )
            spatial_marker_memory_cell_loss = (
                candidate_score_value_prediction.new_zeros(())
            )
            spatial_marker_memory_mass_loss = (
                candidate_score_value_prediction.new_zeros(())
            )
            spatial_marker_memory_valid_count = (
                candidate_score_value_prediction.new_zeros(())
            )
            spatial_marker_memory_target_mean = (
                candidate_score_value_prediction.new_zeros(())
            )
        else:
            spatial_marker_memory_start_belief, spatial_marker_memory_mass = (
                self.spatial_marker_memory_start_belief(
                    history_vision,
                    history_actions,
                    vision[:, 0],
                    batch_size=actions.shape[0],
                    device=actions.device,
                    dtype=candidate_score_value_prediction.dtype,
                )
            )
            spatial_marker_memory_score_prediction, _ = (
                self.spatial_marker_memory_score(
                    spatial_marker_memory_start_belief,
                    spatial_marker_memory_mass,
                    actions,
                )
            )
            if (
                structured_marker_memory_valid_mask is None
                or structured_marker_memory_start_delta_targets is None
            ):
                spatial_marker_memory_loss = (
                    candidate_score_value_prediction.new_zeros(())
                )
                spatial_marker_memory_cell_loss = (
                    candidate_score_value_prediction.new_zeros(())
                )
                spatial_marker_memory_mass_loss = (
                    candidate_score_value_prediction.new_zeros(())
                )
                spatial_marker_memory_valid_count = (
                    candidate_score_value_prediction.new_zeros(())
                )
                spatial_marker_memory_target_mean = (
                    candidate_score_value_prediction.new_zeros(())
                )
            else:
                if structured_marker_memory_valid_mask.shape != (
                    candidate_score_value_prediction.shape[0],
                ):
                    raise ValueError(
                        "structured_marker_memory_valid_mask must have shape "
                        f"({candidate_score_value_prediction.shape[0]},)"
                    )
                if structured_marker_memory_start_delta_targets.shape != (
                    candidate_score_value_prediction.shape[0],
                    2,
                ):
                    raise ValueError(
                        "structured_marker_memory_start_delta_targets must have "
                        f"shape ({candidate_score_value_prediction.shape[0]}, 2)"
                    )
                spatial_start_valid = structured_marker_memory_valid_mask.to(
                    device=candidate_score_value_prediction.device,
                    dtype=torch.bool,
                )
                spatial_start_delta_targets = (
                    structured_marker_memory_start_delta_targets.to(
                        device=candidate_score_value_prediction.device,
                        dtype=spatial_marker_memory_start_belief.dtype,
                    )
                )
                spatial_valid_mask, spatial_targets = (
                    self.spatial_marker_memory_target_indices(
                        spatial_start_valid,
                        spatial_start_delta_targets,
                    )
                )
                spatial_marker_memory_valid_count = (
                    spatial_valid_mask.sum().to(candidate_score_value_prediction.dtype)
                )
                spatial_marker_memory_mass_loss = F.binary_cross_entropy(
                    spatial_marker_memory_mass.clamp(1e-6, 1.0 - 1e-6),
                    spatial_start_valid.to(
                        dtype=spatial_marker_memory_mass.dtype,
                    ),
                )
                if spatial_valid_mask.any():
                    spatial_marker_memory_cell_loss = F.nll_loss(
                        spatial_marker_memory_start_belief[
                            spatial_valid_mask
                        ].clamp_min(1e-6).log(),
                        spatial_targets[spatial_valid_mask],
                    )
                    spatial_marker_memory_target_mean = (
                        spatial_targets[spatial_valid_mask]
                        .to(candidate_score_value_prediction.dtype)
                        .mean()
                    )
                else:
                    spatial_marker_memory_cell_loss = (
                        candidate_score_value_prediction.new_zeros(())
                    )
                    spatial_marker_memory_target_mean = (
                        candidate_score_value_prediction.new_zeros(())
                    )
                spatial_marker_memory_loss = (
                    spatial_marker_memory_cell_loss + spatial_marker_memory_mass_loss
                )
        if self.spatial_frontier_memory_detector is None:
            spatial_frontier_marker_belief = (
                candidate_score_value_prediction.new_zeros(
                    candidate_score_value_prediction.shape[0],
                    self.spatial_memory_size * self.spatial_memory_size,
                )
            )
            spatial_frontier_marker_mass = (
                candidate_score_value_prediction.new_zeros(
                    candidate_score_value_prediction.shape
                )
            )
            spatial_frontier_observed_map = (
                candidate_score_value_prediction.new_zeros(
                    candidate_score_value_prediction.shape[0],
                    self.spatial_memory_size,
                    self.spatial_memory_size,
                )
            )
            spatial_frontier_free_map = torch.zeros_like(
                spatial_frontier_observed_map
            )
            spatial_frontier_blocked_map = torch.zeros_like(
                spatial_frontier_observed_map
            )
            spatial_frontier_memory_score_prediction = (
                candidate_score_value_prediction.new_zeros(
                    candidate_score_value_prediction.shape
                )
            )
            spatial_frontier_memory_loss = (
                candidate_score_value_prediction.new_zeros(())
            )
            spatial_frontier_memory_score_loss = (
                candidate_score_value_prediction.new_zeros(())
            )
            spatial_frontier_memory_occupancy_loss = (
                candidate_score_value_prediction.new_zeros(())
            )
            spatial_frontier_memory_marker_loss = (
                candidate_score_value_prediction.new_zeros(())
            )
            spatial_frontier_memory_marker_cell_loss = (
                candidate_score_value_prediction.new_zeros(())
            )
            spatial_frontier_memory_marker_mass_loss = (
                candidate_score_value_prediction.new_zeros(())
            )
            spatial_frontier_observation_loss = (
                candidate_score_value_prediction.new_zeros(())
            )
            spatial_frontier_observation_marker_loss = (
                candidate_score_value_prediction.new_zeros(())
            )
            spatial_frontier_observation_occupancy_loss = (
                candidate_score_value_prediction.new_zeros(())
            )
            spatial_frontier_observation_frame_count = (
                candidate_score_value_prediction.new_zeros(())
            )
            spatial_frontier_memory_valid_count = (
                candidate_score_value_prediction.new_zeros(())
            )
        else:
            (
                spatial_frontier_observation_loss,
                spatial_frontier_observation_marker_loss,
                spatial_frontier_observation_occupancy_loss,
                spatial_frontier_observation_frame_count,
            ) = self.spatial_frontier_observation_supervision_loss(
                history_vision=history_vision,
                vision=vision,
                history_targets=spatial_frontier_history_observation_targets,
                vision_targets=spatial_frontier_vision_observation_targets,
            )
            (
                spatial_frontier_marker_belief,
                spatial_frontier_marker_mass,
                spatial_frontier_observed_map,
                spatial_frontier_free_map,
                spatial_frontier_blocked_map,
            ) = self.spatial_frontier_memory_start_maps(
                history_vision,
                history_actions,
                vision[:, 0],
                batch_size=actions.shape[0],
                device=actions.device,
                dtype=candidate_score_value_prediction.dtype,
            )
            if self.spatial_frontier_marker_source == "spatial_marker":
                spatial_frontier_score_marker_belief = (
                    spatial_marker_memory_start_belief
                )
                spatial_frontier_score_marker_mass = spatial_marker_memory_mass
            else:
                spatial_frontier_score_marker_belief = (
                    spatial_frontier_marker_belief
                )
                spatial_frontier_score_marker_mass = spatial_frontier_marker_mass
            spatial_frontier_memory_score_prediction = (
                self.spatial_frontier_memory_score(
                    spatial_frontier_score_marker_belief,
                    spatial_frontier_score_marker_mass,
                    spatial_frontier_observed_map,
                    spatial_frontier_blocked_map,
                    actions,
                )
            )
            spatial_frontier_memory_score_loss = F.mse_loss(
                spatial_frontier_memory_score_prediction,
                online_frontier_marker_score_prediction.detach(),
            )
            with torch.no_grad():
                target_observed_map, target_blocked_map = (
                    self.online_frontier_start_maps(
                        history_vision,
                        history_actions,
                        vision[:, 0],
                        batch_size=actions.shape[0],
                        device=actions.device,
                        dtype=candidate_score_value_prediction.dtype,
                    )
                )
                target_free_map = (
                    target_observed_map - target_blocked_map
                ).clamp(0.0, 1.0)
            spatial_frontier_memory_occupancy_loss = (
                F.binary_cross_entropy(
                    spatial_frontier_observed_map.clamp(1e-6, 1.0 - 1e-6),
                    target_observed_map,
                )
                + F.binary_cross_entropy(
                    spatial_frontier_free_map.clamp(1e-6, 1.0 - 1e-6),
                    target_free_map,
                )
                + F.binary_cross_entropy(
                    spatial_frontier_blocked_map.clamp(1e-6, 1.0 - 1e-6),
                    target_blocked_map,
                )
            )
            if (
                structured_marker_memory_valid_mask is None
                or structured_marker_memory_start_delta_targets is None
            ):
                spatial_frontier_memory_marker_cell_loss = (
                    candidate_score_value_prediction.new_zeros(())
                )
                spatial_frontier_memory_marker_mass_loss = (
                    candidate_score_value_prediction.new_zeros(())
                )
                spatial_frontier_memory_valid_count = (
                    candidate_score_value_prediction.new_zeros(())
                )
            else:
                if structured_marker_memory_valid_mask.shape != (
                    candidate_score_value_prediction.shape[0],
                ):
                    raise ValueError(
                        "structured_marker_memory_valid_mask must have shape "
                        f"({candidate_score_value_prediction.shape[0]},)"
                    )
                if structured_marker_memory_start_delta_targets.shape != (
                    candidate_score_value_prediction.shape[0],
                    2,
                ):
                    raise ValueError(
                        "structured_marker_memory_start_delta_targets must have "
                        f"shape ({candidate_score_value_prediction.shape[0]}, 2)"
                    )
                spatial_frontier_marker_start_valid = (
                    structured_marker_memory_valid_mask.to(
                        device=candidate_score_value_prediction.device,
                        dtype=torch.bool,
                    )
                )
                spatial_frontier_marker_start_delta_targets = (
                    structured_marker_memory_start_delta_targets.to(
                        device=candidate_score_value_prediction.device,
                        dtype=spatial_frontier_marker_belief.dtype,
                    )
                )
                (
                    spatial_frontier_marker_valid_mask,
                    spatial_frontier_marker_targets,
                ) = self.spatial_marker_memory_target_indices(
                    spatial_frontier_marker_start_valid,
                    spatial_frontier_marker_start_delta_targets,
                )
                spatial_frontier_memory_valid_count = (
                    spatial_frontier_marker_valid_mask.sum().to(
                        candidate_score_value_prediction.dtype
                    )
                )
                spatial_frontier_memory_marker_mass_loss = F.binary_cross_entropy(
                    spatial_frontier_marker_mass.clamp(1e-6, 1.0 - 1e-6),
                    spatial_frontier_marker_start_valid.to(
                        dtype=spatial_frontier_marker_mass.dtype,
                    ),
                )
                if spatial_frontier_marker_valid_mask.any():
                    spatial_frontier_memory_marker_cell_loss = F.nll_loss(
                        spatial_frontier_marker_belief[
                            spatial_frontier_marker_valid_mask
                        ]
                        .clamp_min(1e-6)
                        .log(),
                        spatial_frontier_marker_targets[
                            spatial_frontier_marker_valid_mask
                        ],
                    )
                else:
                    spatial_frontier_memory_marker_cell_loss = (
                        candidate_score_value_prediction.new_zeros(())
                    )
            spatial_frontier_memory_marker_loss = (
                self.spatial_frontier_memory_marker_cell_loss_weight
                * spatial_frontier_memory_marker_cell_loss
                + self.spatial_frontier_memory_marker_mass_loss_weight
                * spatial_frontier_memory_marker_mass_loss
            )
            spatial_frontier_memory_loss = (
                self.spatial_frontier_memory_occupancy_loss_weight
                * spatial_frontier_memory_occupancy_loss
                + self.spatial_frontier_memory_marker_loss_weight
                * spatial_frontier_memory_marker_loss
            )
        candidate_score_prediction = (
            candidate_score_value_prediction
            + self.candidate_score_claim_logit_weight * candidate_claim_logit
            + self.online_marker_memory_score_weight
            * online_marker_memory_score_prediction
            + self.candidate_marker_memory_score_weight
            * candidate_marker_memory_score_prediction
            + self.structured_marker_memory_score_weight
            * structured_marker_memory_score_prediction
            + self.categorical_marker_memory_score_weight
            * categorical_marker_memory_score_prediction
            + self.spatial_marker_memory_score_weight
            * spatial_marker_memory_score_prediction
            + self.spatial_frontier_memory_score_weight
            * spatial_frontier_memory_score_prediction
        )
        if self.utility_source == "consequence":
            utility_prediction = consequence_utility_prediction
        elif self.utility_source == "head":
            utility_prediction = utility_head_prediction
        else:
            utility_prediction = candidate_score_prediction
        utility_loss = F.mse_loss(utility_prediction, utility_targets)
        utility_head_loss = F.mse_loss(utility_head_prediction, utility_targets)
        candidate_score_loss = F.mse_loss(candidate_score_prediction, utility_targets)
        if utility_group_ids is not None:
            if utility_mask is None:
                utility_mask = torch.ones_like(utility_targets, dtype=torch.bool)
            utility_losses = action_utility_losses(
                utility_prediction=utility_prediction,
                utility_targets=utility_targets,
                utility_mask=utility_mask,
                utility_group_ids=utility_group_ids,
                regression_weight=self.utility_ranking_regression_weight,
                ranking_loss=self.utility_ranking_loss_type,
                softmax_temperature=self.utility_softmax_temperature,
            )
            candidate_score_losses = action_utility_losses(
                utility_prediction=candidate_score_prediction,
                utility_targets=utility_targets,
                utility_mask=utility_mask,
                utility_group_ids=utility_group_ids,
                regression_weight=self.candidate_score_regression_weight,
                ranking_loss=self.candidate_score_ranking_loss_type,
                softmax_temperature=self.candidate_score_softmax_temperature,
            )
            if candidate_marker_memory_score_targets is None:
                marker_score_targets = utility_targets.new_zeros(
                    utility_targets.shape
                )
            else:
                if candidate_marker_memory_score_targets.shape != utility_targets.shape:
                    raise ValueError(
                        "candidate_marker_memory_score_targets must have shape "
                        f"{tuple(utility_targets.shape)}"
                    )
                marker_score_targets = candidate_marker_memory_score_targets.to(
                    device=utility_targets.device,
                    dtype=utility_targets.dtype,
                )
            if candidate_marker_memory_valid_mask is None:
                marker_score_mask = torch.zeros_like(utility_mask, dtype=torch.bool)
            else:
                if candidate_marker_memory_valid_mask.shape != utility_targets.shape:
                    raise ValueError(
                        "candidate_marker_memory_valid_mask must have shape "
                        f"{tuple(utility_targets.shape)}"
                    )
                marker_score_mask = (
                    candidate_marker_memory_valid_mask.to(
                        device=utility_targets.device,
                        dtype=torch.bool,
                    )
                    & utility_mask
                )
            candidate_marker_memory_ranking_losses = action_utility_losses(
                utility_prediction=candidate_marker_memory_score_prediction,
                utility_targets=marker_score_targets,
                utility_mask=marker_score_mask,
                utility_group_ids=utility_group_ids,
                regression_weight=0.0,
                ranking_loss=self.candidate_marker_memory_ranking_loss_type,
                softmax_temperature=(
                    self.candidate_marker_memory_softmax_temperature
                ),
            )
            structured_marker_memory_ranking_losses = action_utility_losses(
                utility_prediction=structured_marker_memory_score_prediction,
                utility_targets=marker_score_targets,
                utility_mask=marker_score_mask,
                utility_group_ids=utility_group_ids,
                regression_weight=0.0,
                ranking_loss="hard_ce",
                softmax_temperature=(
                    self.structured_marker_memory_softmax_temperature
                ),
            )
            categorical_marker_memory_ranking_losses = action_utility_losses(
                utility_prediction=categorical_marker_memory_score_prediction,
                utility_targets=marker_score_targets,
                utility_mask=marker_score_mask,
                utility_group_ids=utility_group_ids,
                regression_weight=0.0,
                ranking_loss="hard_ce",
                softmax_temperature=(
                    self.categorical_marker_memory_softmax_temperature
                ),
            )
            spatial_marker_memory_ranking_losses = action_utility_losses(
                utility_prediction=spatial_marker_memory_score_prediction,
                utility_targets=marker_score_targets,
                utility_mask=marker_score_mask,
                utility_group_ids=utility_group_ids,
                regression_weight=0.0,
                ranking_loss="hard_ce",
                softmax_temperature=(
                    self.spatial_marker_memory_softmax_temperature
                ),
            )
            spatial_frontier_memory_ranking_losses = action_utility_losses(
                utility_prediction=spatial_frontier_memory_score_prediction,
                utility_targets=utility_targets,
                utility_mask=utility_mask,
                utility_group_ids=utility_group_ids,
                regression_weight=0.0,
                ranking_loss="hard_ce",
                softmax_temperature=(
                    self.spatial_frontier_memory_softmax_temperature
                ),
            )
        else:
            utility_losses = {
                "action_utility_loss": rollout.new_zeros(()),
                "action_utility_ce_loss": rollout.new_zeros(()),
                "action_utility_regression_loss": utility_loss,
                "action_utility_valid_count": rollout.new_zeros(()),
                "action_utility_group_count": rollout.new_zeros(()),
            }
            candidate_score_losses = {
                "action_utility_loss": rollout.new_zeros(()),
                "action_utility_ce_loss": rollout.new_zeros(()),
                "action_utility_regression_loss": candidate_score_loss,
                "action_utility_valid_count": rollout.new_zeros(()),
                "action_utility_group_count": rollout.new_zeros(()),
            }
            candidate_marker_memory_ranking_losses = {
                "action_utility_loss": rollout.new_zeros(()),
                "action_utility_ce_loss": rollout.new_zeros(()),
                "action_utility_regression_loss": rollout.new_zeros(()),
                "action_utility_valid_count": rollout.new_zeros(()),
                "action_utility_group_count": rollout.new_zeros(()),
            }
            structured_marker_memory_ranking_losses = {
                "action_utility_loss": rollout.new_zeros(()),
                "action_utility_ce_loss": rollout.new_zeros(()),
                "action_utility_regression_loss": rollout.new_zeros(()),
                "action_utility_valid_count": rollout.new_zeros(()),
                "action_utility_group_count": rollout.new_zeros(()),
            }
            categorical_marker_memory_ranking_losses = {
                "action_utility_loss": rollout.new_zeros(()),
                "action_utility_ce_loss": rollout.new_zeros(()),
                "action_utility_regression_loss": rollout.new_zeros(()),
                "action_utility_valid_count": rollout.new_zeros(()),
                "action_utility_group_count": rollout.new_zeros(()),
            }
            spatial_marker_memory_ranking_losses = {
                "action_utility_loss": rollout.new_zeros(()),
                "action_utility_ce_loss": rollout.new_zeros(()),
                "action_utility_regression_loss": rollout.new_zeros(()),
                "action_utility_valid_count": rollout.new_zeros(()),
                "action_utility_group_count": rollout.new_zeros(()),
            }
            spatial_frontier_memory_ranking_losses = {
                "action_utility_loss": rollout.new_zeros(()),
                "action_utility_ce_loss": rollout.new_zeros(()),
                "action_utility_regression_loss": rollout.new_zeros(()),
                "action_utility_valid_count": rollout.new_zeros(()),
                "action_utility_group_count": rollout.new_zeros(()),
            }

        if wrong_actions is not None:
            wrong_predictions = self.predict_wrong_steps(previous_targets, wrong_actions)
            wrong_rollouts = self.rollout_wrong_actions(online_tokens[:, 0], wrong_actions)
            decision_wrong_rollouts = self.rollout_wrong_decision_actions(
                decision_start,
                wrong_actions,
            )
            decision_wrong_teacher_forced = (
                self.predict_wrong_decision_steps(decision_teacher_inputs, wrong_actions)
                if decision_teacher_inputs is not None
                else None
            )
        else:
            wrong_predictions = None
            wrong_rollouts = None
            decision_wrong_rollouts = None
            decision_wrong_teacher_forced = None
        if zero_actions is None:
            zero_actions = torch.zeros_like(actions)
        zero_prediction = self.predict_step_sequence(previous_targets, zero_actions)
        zero_rollout = self.rollout(online_tokens[:, 0], zero_actions)
        decision_zero_rollout = self.rollout_decision(decision_start, zero_actions)
        decision_zero_teacher_forced = (
            self.predict_decision_step_sequence(decision_teacher_inputs, zero_actions)
            if decision_teacher_inputs is not None
            else None
        )
        action_losses = action_identifiability_losses(
            real_prediction=teacher_forced,
            targets=targets,
            previous_targets=previous_targets,
            wrong_predictions=wrong_predictions,
            wrong_mask=wrong_mask,
            zero_prediction=zero_prediction,
            non_hold_mask=non_hold_mask,
            transition_mask=None,
            margin_fraction=self.action_margin_fraction,
            margin_floor=self.action_margin_floor,
        )
        free_running_losses = action_identifiability_losses(
            real_prediction=rollout,
            targets=targets,
            previous_targets=previous_targets,
            wrong_predictions=wrong_rollouts,
            wrong_mask=wrong_mask,
            zero_prediction=zero_rollout,
            non_hold_mask=non_hold_mask,
            transition_mask=None,
            margin_fraction=self.action_margin_fraction,
            margin_floor=self.action_margin_floor,
        )
        if decision_targets is not None and decision_previous_targets is not None:
            decision_losses = action_identifiability_losses(
                real_prediction=decision_rollout,
                targets=decision_targets,
                previous_targets=decision_previous_targets,
                wrong_predictions=decision_wrong_rollouts,
                wrong_mask=wrong_mask,
                zero_prediction=decision_zero_rollout,
                non_hold_mask=non_hold_mask,
                transition_mask=None,
                margin_fraction=self.action_margin_fraction,
                margin_floor=self.action_margin_floor,
            )
            decision_teacher_forced_losses = action_identifiability_losses(
                real_prediction=decision_teacher_forced,
                targets=decision_targets,
                previous_targets=decision_previous_targets,
                wrong_predictions=decision_wrong_teacher_forced,
                wrong_mask=wrong_mask,
                zero_prediction=decision_zero_teacher_forced,
                non_hold_mask=non_hold_mask,
                transition_mask=None,
                margin_fraction=self.action_margin_fraction,
                margin_floor=self.action_margin_floor,
            )
        else:
            decision_losses = {
                "action_identifiability_loss": rollout.new_zeros(()),
                "zero_action_loss": rollout.new_zeros(()),
                "mean_wrong_mse": rollout.new_zeros(actions.shape[:2]),
                "zero_mse": rollout.new_zeros(actions.shape[:2]),
                "target_change_mse": rollout.new_zeros(actions.shape[:2]),
                "eligible_wrong_mask": torch.zeros(
                    actions.shape[:2],
                    dtype=torch.bool,
                    device=actions.device,
                ),
                "eligible_zero_mask": torch.zeros(
                    actions.shape[:2],
                    dtype=torch.bool,
                    device=actions.device,
                ),
            }
            decision_teacher_forced_losses = decision_losses
        variance_loss = spatial_variance_floor_loss(
            target_normalized,
            target_std=self.spatial_target_std,
        )
        loss = (
            self.prediction_loss_lambda * prediction_loss
            + teacher_forced_prediction_loss
            + self.action_identifiability_lambda
            * action_losses["action_identifiability_loss"]
            + self.zero_action_lambda * action_losses["zero_action_loss"]
            + self.free_running_action_contrast_lambda
            * free_running_losses["action_identifiability_loss"]
            + self.free_running_zero_contrast_lambda
            * free_running_losses["zero_action_loss"]
            + self.utility_loss_lambda * utility_loss
            + self.utility_ranking_loss_lambda
            * utility_losses["action_utility_loss"]
            + self.candidate_score_loss_lambda
            * candidate_score_losses["action_utility_loss"]
            + self.candidate_claim_loss_lambda * candidate_claim_loss
            + self.candidate_marker_memory_loss_lambda
            * candidate_marker_memory_loss
            + self.candidate_marker_memory_ranking_loss_lambda
            * candidate_marker_memory_ranking_losses["action_utility_loss"]
            + self.structured_marker_memory_loss_lambda
            * structured_marker_memory_loss
            + self.structured_marker_memory_ranking_loss_lambda
            * structured_marker_memory_ranking_losses["action_utility_loss"]
            + self.categorical_marker_memory_loss_lambda
            * categorical_marker_memory_loss
            + self.categorical_marker_memory_ranking_loss_lambda
            * categorical_marker_memory_ranking_losses["action_utility_loss"]
            + self.spatial_marker_memory_loss_lambda
            * spatial_marker_memory_loss
            + self.spatial_marker_memory_ranking_loss_lambda
            * spatial_marker_memory_ranking_losses["action_utility_loss"]
            + self.spatial_frontier_memory_loss_lambda
            * spatial_frontier_memory_loss
            + self.spatial_frontier_observation_loss_lambda
            * spatial_frontier_observation_loss
            + self.spatial_frontier_memory_score_loss_lambda
            * spatial_frontier_memory_score_loss
            + self.spatial_frontier_memory_ranking_loss_lambda
            * spatial_frontier_memory_ranking_losses["action_utility_loss"]
            + self.consequence_loss_lambda * consequence_loss
            + self.rollout_delta_loss_lambda * rollout_delta_loss
            + self.teacher_forced_delta_loss_lambda * teacher_forced_delta_loss
            + self.decision_prediction_loss_lambda * decision_prediction_loss
            + self.decision_delta_loss_lambda * decision_delta_loss
            + self.decision_teacher_forced_prediction_loss_lambda
            * decision_teacher_forced_prediction_loss
            + self.decision_teacher_forced_delta_loss_lambda
            * decision_teacher_forced_delta_loss
            + self.decision_action_contrast_lambda
            * decision_losses["action_identifiability_loss"]
            + self.decision_zero_contrast_lambda * decision_losses["zero_action_loss"]
            + self.decision_teacher_forced_action_contrast_lambda
            * decision_teacher_forced_losses["action_identifiability_loss"]
            + self.decision_teacher_forced_zero_contrast_lambda
            * decision_teacher_forced_losses["zero_action_loss"]
            + self.spatial_variance_lambda * variance_loss
        )
        result = {
            "loss": loss,
            "prediction_loss": prediction_loss,
            "teacher_forced_prediction_loss": teacher_forced_prediction_loss,
            "rollout_delta_loss": rollout_delta_loss,
            "teacher_forced_delta_loss": teacher_forced_delta_loss,
            "decision_prediction_loss": decision_prediction_loss,
            "decision_delta_loss": decision_delta_loss,
            "decision_teacher_forced_prediction_loss": (
                decision_teacher_forced_prediction_loss
            ),
            "decision_teacher_forced_delta_loss": decision_teacher_forced_delta_loss,
            "decision_action_contrast_loss": decision_losses[
                "action_identifiability_loss"
            ],
            "decision_zero_contrast_loss": decision_losses["zero_action_loss"],
            "decision_teacher_forced_action_contrast_loss": (
                decision_teacher_forced_losses["action_identifiability_loss"]
            ),
            "decision_teacher_forced_zero_contrast_loss": (
                decision_teacher_forced_losses["zero_action_loss"]
            ),
            "action_identifiability_loss": action_losses["action_identifiability_loss"],
            "zero_action_loss": action_losses["zero_action_loss"],
            "free_running_action_contrast_loss": free_running_losses[
                "action_identifiability_loss"
            ],
            "free_running_zero_contrast_loss": free_running_losses["zero_action_loss"],
            "utility_loss": utility_loss,
            "utility_head_loss": utility_head_loss,
            "candidate_score_loss": candidate_score_loss,
            "candidate_claim_loss": candidate_claim_loss,
            "candidate_marker_memory_loss": candidate_marker_memory_loss,
            "candidate_marker_memory_delta_loss": (
                candidate_marker_memory_delta_loss
            ),
            "candidate_marker_memory_claim_loss": (
                candidate_marker_memory_claim_loss
            ),
            "candidate_marker_memory_ranking_loss": (
                candidate_marker_memory_ranking_losses["action_utility_loss"]
            ),
            "candidate_marker_memory_ranking_ce_loss": (
                candidate_marker_memory_ranking_losses["action_utility_ce_loss"]
            ),
            "candidate_marker_memory_ranking_group_count": (
                candidate_marker_memory_ranking_losses["action_utility_group_count"]
            ),
            "structured_marker_memory_loss": structured_marker_memory_loss,
            "structured_marker_memory_start_delta_loss": (
                structured_marker_memory_start_delta_loss
            ),
            "structured_marker_memory_final_delta_loss": (
                structured_marker_memory_final_delta_loss
            ),
            "structured_marker_memory_ranking_loss": (
                structured_marker_memory_ranking_losses["action_utility_loss"]
            ),
            "structured_marker_memory_ranking_ce_loss": (
                structured_marker_memory_ranking_losses["action_utility_ce_loss"]
            ),
            "structured_marker_memory_ranking_group_count": (
                structured_marker_memory_ranking_losses["action_utility_group_count"]
            ),
            "categorical_marker_memory_loss": categorical_marker_memory_loss,
            "categorical_marker_memory_ranking_loss": (
                categorical_marker_memory_ranking_losses["action_utility_loss"]
            ),
            "categorical_marker_memory_ranking_ce_loss": (
                categorical_marker_memory_ranking_losses["action_utility_ce_loss"]
            ),
            "categorical_marker_memory_ranking_group_count": (
                categorical_marker_memory_ranking_losses["action_utility_group_count"]
            ),
            "spatial_marker_memory_loss": spatial_marker_memory_loss,
            "spatial_marker_memory_cell_loss": spatial_marker_memory_cell_loss,
            "spatial_marker_memory_mass_loss": spatial_marker_memory_mass_loss,
            "spatial_marker_memory_ranking_loss": (
                spatial_marker_memory_ranking_losses["action_utility_loss"]
            ),
            "spatial_marker_memory_ranking_ce_loss": (
                spatial_marker_memory_ranking_losses["action_utility_ce_loss"]
            ),
            "spatial_marker_memory_ranking_group_count": (
                spatial_marker_memory_ranking_losses["action_utility_group_count"]
            ),
            "spatial_frontier_memory_loss": spatial_frontier_memory_loss,
            "spatial_frontier_memory_score_loss": (
                spatial_frontier_memory_score_loss
            ),
            "spatial_frontier_memory_occupancy_loss": (
                spatial_frontier_memory_occupancy_loss
            ),
            "spatial_frontier_memory_marker_loss": (
                spatial_frontier_memory_marker_loss
            ),
            "spatial_frontier_memory_marker_cell_loss": (
                spatial_frontier_memory_marker_cell_loss
            ),
            "spatial_frontier_memory_marker_mass_loss": (
                spatial_frontier_memory_marker_mass_loss
            ),
            "spatial_frontier_observation_loss": spatial_frontier_observation_loss,
            "spatial_frontier_observation_marker_loss": (
                spatial_frontier_observation_marker_loss
            ),
            "spatial_frontier_observation_occupancy_loss": (
                spatial_frontier_observation_occupancy_loss
            ),
            "spatial_frontier_memory_ranking_loss": (
                spatial_frontier_memory_ranking_losses["action_utility_loss"]
            ),
            "spatial_frontier_memory_ranking_ce_loss": (
                spatial_frontier_memory_ranking_losses["action_utility_ce_loss"]
            ),
            "spatial_frontier_memory_ranking_group_count": (
                spatial_frontier_memory_ranking_losses["action_utility_group_count"]
            ),
            "candidate_score_ranking_loss": candidate_score_losses[
                "action_utility_loss"
            ],
            "candidate_score_ranking_ce_loss": candidate_score_losses[
                "action_utility_ce_loss"
            ],
            "candidate_score_ranking_regression_loss": candidate_score_losses[
                "action_utility_regression_loss"
            ],
            "candidate_score_ranking_group_count": candidate_score_losses[
                "action_utility_group_count"
            ],
            "utility_ranking_loss": utility_losses["action_utility_loss"],
            "utility_ranking_ce_loss": utility_losses["action_utility_ce_loss"],
            "utility_ranking_regression_loss": utility_losses[
                "action_utility_regression_loss"
            ],
            "utility_ranking_group_count": utility_losses[
                "action_utility_group_count"
            ],
            "consequence_loss": consequence_loss,
            "consequence_binary_loss": consequence_binary_loss,
            "consequence_scalar_loss": consequence_scalar_loss,
            "spatial_variance_loss": variance_loss,
            "real_prediction_mse": action_losses["real_mse"].mean(),
            "hard_negative_mse": action_losses["mean_wrong_mse"][
                action_losses["eligible_wrong_mask"]
            ].mean()
            if action_losses["eligible_wrong_mask"].any()
            else loss.new_zeros(()),
            "zero_action_mse": action_losses["zero_mse"][
                action_losses["eligible_zero_mask"]
            ].mean()
            if action_losses["eligible_zero_mask"].any()
            else loss.new_zeros(()),
            "free_running_hard_negative_mse": free_running_losses["mean_wrong_mse"][
                free_running_losses["eligible_wrong_mask"]
            ].mean()
            if free_running_losses["eligible_wrong_mask"].any()
            else loss.new_zeros(()),
            "free_running_zero_action_mse": free_running_losses["zero_mse"][
                free_running_losses["eligible_zero_mask"]
            ].mean()
            if free_running_losses["eligible_zero_mask"].any()
            else loss.new_zeros(()),
            "decision_hard_negative_mse": decision_losses["mean_wrong_mse"][
                decision_losses["eligible_wrong_mask"]
            ].mean()
            if decision_losses["eligible_wrong_mask"].any()
            else loss.new_zeros(()),
            "decision_zero_action_mse": decision_losses["zero_mse"][
                decision_losses["eligible_zero_mask"]
            ].mean()
            if decision_losses["eligible_zero_mask"].any()
            else loss.new_zeros(()),
            "decision_teacher_forced_hard_negative_mse": (
                decision_teacher_forced_losses["mean_wrong_mse"][
                    decision_teacher_forced_losses["eligible_wrong_mask"]
                ].mean()
                if decision_teacher_forced_losses["eligible_wrong_mask"].any()
                else loss.new_zeros(())
            ),
            "decision_teacher_forced_zero_action_mse": (
                decision_teacher_forced_losses["zero_mse"][
                    decision_teacher_forced_losses["eligible_zero_mask"]
                ].mean()
                if decision_teacher_forced_losses["eligible_zero_mask"].any()
                else loss.new_zeros(())
            ),
            "decision_mean_target_change_mse": decision_losses[
                "target_change_mse"
            ].mean(),
            "mean_target_change_mse": action_losses["target_change_mse"].mean(),
            "utility_prediction_mean": utility_prediction.mean(),
            "utility_head_prediction_mean": utility_head_prediction.mean(),
            "candidate_score_prediction_mean": candidate_score_prediction.mean(),
            "candidate_score_value_prediction_mean": (
                candidate_score_value_prediction.mean()
            ),
            "online_marker_memory_score_mean": (
                online_marker_memory_score_prediction.mean()
            ),
            "online_frontier_marker_score_mean": (
                online_frontier_marker_score_prediction.mean()
            ),
            "candidate_claim_logit_mean": candidate_claim_logit.mean(),
            "candidate_claim_target_mean": candidate_claim_targets.mean(),
            "candidate_marker_memory_score_mean": (
                candidate_marker_memory_score_prediction.mean()
            ),
            "candidate_marker_memory_claim_logit_mean": (
                candidate_marker_memory_claim_logit.mean()
            ),
            "candidate_marker_memory_delta_abs_mean": (
                candidate_marker_memory_delta_prediction.abs().mean()
            ),
            "candidate_marker_memory_valid_count": (
                candidate_marker_memory_valid_count
            ),
            "candidate_marker_memory_claim_target_mean": (
                candidate_marker_memory_claim_target_mean
            ),
            "structured_marker_memory_score_mean": (
                structured_marker_memory_score_prediction.mean()
            ),
            "structured_marker_memory_start_delta_abs_mean": (
                structured_marker_memory_start_delta_prediction.abs().mean()
            ),
            "structured_marker_memory_delta_abs_mean": (
                structured_marker_memory_delta_prediction.abs().mean()
            ),
            "structured_marker_memory_valid_count": (
                structured_marker_memory_valid_count
            ),
            "categorical_marker_memory_score_mean": (
                categorical_marker_memory_score_prediction.mean()
            ),
            "categorical_marker_memory_logit_abs_mean": (
                categorical_marker_memory_logits.abs().mean()
            ),
            "categorical_marker_memory_valid_count": (
                categorical_marker_memory_valid_count
            ),
            "categorical_marker_memory_target_mean": (
                categorical_marker_memory_target_mean
            ),
            "spatial_marker_memory_score_mean": (
                spatial_marker_memory_score_prediction.mean()
            ),
            "spatial_marker_memory_mass_mean": spatial_marker_memory_mass.mean(),
            "spatial_marker_memory_valid_count": spatial_marker_memory_valid_count,
            "spatial_marker_memory_target_mean": spatial_marker_memory_target_mean,
            "spatial_frontier_memory_score_mean": (
                spatial_frontier_memory_score_prediction.mean()
            ),
            "spatial_frontier_memory_marker_mass_mean": (
                spatial_frontier_marker_mass.mean()
            ),
            "spatial_frontier_memory_observed_mean": (
                spatial_frontier_observed_map.mean()
            ),
            "spatial_frontier_memory_free_mean": spatial_frontier_free_map.mean(),
            "spatial_frontier_memory_blocked_mean": (
                spatial_frontier_blocked_map.mean()
            ),
            "spatial_frontier_memory_valid_count": (
                spatial_frontier_memory_valid_count
            ),
            "spatial_frontier_observation_frame_count": (
                spatial_frontier_observation_frame_count
            ),
            "consequence_utility_prediction_mean": (
                consequence_utility_prediction.mean()
            ),
            "utility_target_mean": utility_targets.mean(),
        }
        if return_latents:
            result.update(
                {
                    "rollout": rollout,
                    "teacher_forced_prediction": teacher_forced,
                    "targets": targets,
                    "previous_targets": previous_targets,
                    "target_delta": target_delta,
                    "rollout_delta": rollout_delta,
                    "decision_start": decision_start,
                    "memory_context": memory_context,
                    "decision_rollout": decision_rollout,
                    "decision_teacher_forced_prediction": decision_teacher_forced,
                    "decision_targets": decision_targets,
                    "decision_previous_targets": decision_previous_targets,
                    "decision_teacher_inputs": decision_teacher_inputs,
                    "decision_target_delta": decision_target_delta,
                    "decision_rollout_delta": decision_rollout_delta,
                    "decision_teacher_forced_delta": decision_teacher_forced_delta,
                    "zero_prediction": zero_prediction,
                    "zero_rollout": zero_rollout,
                    "decision_zero_rollout": decision_zero_rollout,
                    "decision_zero_teacher_forced": decision_zero_teacher_forced,
                    "wrong_predictions": wrong_predictions,
                    "free_running_wrong_predictions": wrong_rollouts,
                    "decision_wrong_predictions": decision_wrong_rollouts,
                    "decision_wrong_teacher_forced": decision_wrong_teacher_forced,
                    "wrong_mask": wrong_mask,
                    "target_pre_normalized": target_pre,
                    "target_normalized_all": target_normalized,
                    "utility_prediction": utility_prediction,
                    "utility_head_prediction": utility_head_prediction,
                    "candidate_score_prediction": candidate_score_prediction,
                    "candidate_score_value_prediction": (
                        candidate_score_value_prediction
                    ),
                    "candidate_claim_logit": candidate_claim_logit,
                    "online_marker_memory_score_prediction": (
                        online_marker_memory_score_prediction
                    ),
                    "online_frontier_marker_score_prediction": (
                        online_frontier_marker_score_prediction
                    ),
                    "candidate_marker_memory_score_prediction": (
                        candidate_marker_memory_score_prediction
                    ),
                    "candidate_marker_memory_delta_prediction": (
                        candidate_marker_memory_delta_prediction
                    ),
                    "candidate_marker_memory_claim_logit": (
                        candidate_marker_memory_claim_logit
                    ),
                    "structured_marker_memory_score_prediction": (
                        structured_marker_memory_score_prediction
                    ),
                    "structured_marker_memory_start_delta_prediction": (
                        structured_marker_memory_start_delta_prediction
                    ),
                    "structured_marker_memory_delta_prediction": (
                        structured_marker_memory_delta_prediction
                    ),
                    "categorical_marker_memory_score_prediction": (
                        categorical_marker_memory_score_prediction
                    ),
                    "categorical_marker_memory_logits": (
                        categorical_marker_memory_logits
                    ),
                    "spatial_marker_memory_score_prediction": (
                        spatial_marker_memory_score_prediction
                    ),
                    "spatial_marker_memory_start_belief": (
                        spatial_marker_memory_start_belief
                    ),
                    "spatial_marker_memory_mass": spatial_marker_memory_mass,
                    "spatial_frontier_memory_score_prediction": (
                        spatial_frontier_memory_score_prediction
                    ),
                    "spatial_frontier_marker_belief": (
                        spatial_frontier_marker_belief
                    ),
                    "spatial_frontier_marker_mass": spatial_frontier_marker_mass,
                    "spatial_frontier_observed_map": spatial_frontier_observed_map,
                    "spatial_frontier_free_map": spatial_frontier_free_map,
                    "spatial_frontier_blocked_map": spatial_frontier_blocked_map,
                    "consequence_utility_prediction": (
                        consequence_utility_prediction
                    ),
                    "consequence_prediction": consequence_prediction,
                }
            )
        return result
