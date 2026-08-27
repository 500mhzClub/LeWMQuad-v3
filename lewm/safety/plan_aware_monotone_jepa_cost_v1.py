"""Small anchored route-only rankers for PLAN_AWARE_MONOTONE_JEPA_COST_V1.

This module contains architecture and loss primitives only.  It does not load a
panel, checkpoint, route outcome, contact label, or held-out value, and it does
not run training.  Scores use the frozen planning orientation: larger is better.

The two models differ only by the JEPA latent residual.  Both retain the exact
same deterministic kinematic anchor and byte-identically initialised 138-D base
residual.  The latent model additionally attends to CURRENT/H1/H2/H3 token grids
with one 71-D goal/action query.  No rendered GOAL token is an input.
"""

from __future__ import annotations

import hashlib
import math
import struct
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
MODEL_SEED = 2_026_082_701

KINEMATIC_PLUS_NO_LATENT_RESIDUAL = "KINEMATIC_PLUS_NO_LATENT_RESIDUAL"
KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL = "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL"

BASE_FEATURE_DIM = 138
QUERY_FEATURE_DIM = 71
TOKEN_DIM = 1024
TOKENS_PER_TIMEPOINT = 768
TOKEN_GRID_SHAPE = (24, 32)
TIMEPOINTS = ("CURRENT", "H1", "H2", "H3")
TIMEPOINT_COUNT = len(TIMEPOINTS)
LATENT_WIDTH = 64

PAIRWISE_LOSS_WEIGHT = 1.0
LISTWISE_LOSS_WEIGHT = 0.5
RESIDUAL_LOSS_WEIGHT = 1.0e-3
LISTWISE_TEMPERATURE = 1.0
PAIRWISE_UTILITY_TOLERANCE = 1.0e-12

NO_LATENT_PARAMETER_CAP_EXCLUSIVE = 250_000
JEPA_LATENT_PARAMETER_CAP_EXCLUSIVE = 500_000
NO_LATENT_PARAMETER_COUNT = 26_113
JEPA_LATENT_PARAMETER_COUNT = 232_514

SEED_KEY_NAMESPACE = "PLAN_AWARE_MONOTONE_JEPA_COST_V1/KEYED_SEED_V1"
SHARED_BASE_KEY_ID = "SHARED_BASE_RESIDUAL"


BASE_FEATURE_SLICES: Mapping[str, slice] = {
    "waypoint_body_dx_dy_distance_heading_sin_cos": slice(0, 6),
    # This is a task-local route-role encoding.  It must never encode the
    # fit/calibration/held-out custody role.
    "route_role_one_hot": slice(6, 9),
    "requested_action_plan_3x5x3": slice(9, 54),
    "applied_action_plan_3x5x3": slice(54, 99),
    "previous_command_vx_vy_yaw_rate": slice(99, 102),
    "observed_control_history_3x5x2": slice(102, 132),
    "deterministic_kinematic_outcome": slice(132, 138),
}

QUERY_FEATURE_SLICES: Mapping[str, slice] = {
    "waypoint_body_dx_dy_distance_heading_sin_cos": slice(0, 6),
    "route_role_one_hot": slice(6, 9),
    "applied_action_plan_active_vx_yaw_rate_3x5x2": slice(9, 39),
    "previous_command_active_vx_yaw_rate": slice(39, 41),
    "observed_control_history_3x5x2": slice(41, 71),
}


class PlanAwareCostContractError(ValueError):
    """Raised when a tensor violates the prospectively frozen interface."""


def _slice_width(value: slice) -> int:
    if value.start is None or value.stop is None or value.step not in (None, 1):
        raise RuntimeError("feature slices must be finite contiguous intervals")
    return int(value.stop) - int(value.start)


def _validate_feature_slice_contract() -> None:
    for mapping, expected in (
        (BASE_FEATURE_SLICES, BASE_FEATURE_DIM),
        (QUERY_FEATURE_SLICES, QUERY_FEATURE_DIM),
    ):
        cursor = 0
        for value in mapping.values():
            if value.start != cursor:
                raise RuntimeError("feature slices are not contiguous")
            cursor += _slice_width(value)
        if cursor != expected:
            raise RuntimeError(f"feature slices end at {cursor}, expected {expected}")


_validate_feature_slice_contract()


def _require_trailing_shape(
    value: torch.Tensor,
    trailing_shape: tuple[int, ...],
    *,
    name: str,
    leading_shape: torch.Size | None = None,
) -> torch.Size:
    if not isinstance(value, torch.Tensor):
        raise PlanAwareCostContractError(f"{name} must be a torch.Tensor")
    if value.ndim < len(trailing_shape) or tuple(
        value.shape[-len(trailing_shape) :]
    ) != trailing_shape:
        raise PlanAwareCostContractError(
            f"{name} must end in {trailing_shape}, got {tuple(value.shape)}"
        )
    leading = value.shape[: value.ndim - len(trailing_shape)]
    if leading_shape is not None and leading != leading_shape:
        raise PlanAwareCostContractError(
            f"{name} leading shape {tuple(leading)} != {tuple(leading_shape)}"
        )
    return leading


def reshape_tokens_to_spatial_grid(tokens: torch.Tensor) -> torch.Tensor:
    """View flat V-JEPA tokens as the frozen 24-by-32 row-major grid.

    The predecessor tensor interface persists each timepoint as ``[768,1024]``.
    This explicit, view-only transform freezes ``flat_index = y * 32 + x``;
    it neither permutes nor numerically changes token values.
    """

    leading = _require_trailing_shape(
        tokens,
        (TOKENS_PER_TIMEPOINT, TOKEN_DIM),
        name="tokens",
    )
    return tokens.reshape(*leading, *TOKEN_GRID_SHAPE, TOKEN_DIM)


def flatten_spatial_token_grid(token_grid: torch.Tensor) -> torch.Tensor:
    """Flatten a frozen 24-by-32 token grid in the same row-major order."""

    leading = _require_trailing_shape(
        token_grid,
        (*TOKEN_GRID_SHAPE, TOKEN_DIM),
        name="token_grid",
    )
    return token_grid.reshape(*leading, TOKENS_PER_TIMEPOINT, TOKEN_DIM)


def assemble_base_and_query_features(
    *,
    waypoint_features: torch.Tensor,
    route_role_one_hot: torch.Tensor,
    requested_action_plan: torch.Tensor,
    applied_action_plan: torch.Tensor,
    previous_command: torch.Tensor,
    observed_control_history: torch.Tensor,
    deterministic_kinematic_outcome: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble the exact 138-D base and 71-D goal/action-query inputs.

    All arguments have the same arbitrary leading dimensions.  Action tapes are
    three horizons by five commands by ``(vx, vy, yaw_rate)``.  The query uses
    only the active ``vx`` and ``yaw_rate`` channels, while the base preserves
    the complete requested and applied tapes.  ``route_role_one_hot`` is task
    metadata, never a dataset-split indicator.
    """

    leading = _require_trailing_shape(
        waypoint_features, (6,), name="waypoint_features"
    )
    _require_trailing_shape(
        route_role_one_hot,
        (3,),
        name="route_role_one_hot",
        leading_shape=leading,
    )
    _require_trailing_shape(
        requested_action_plan,
        (3, 5, 3),
        name="requested_action_plan",
        leading_shape=leading,
    )
    _require_trailing_shape(
        applied_action_plan,
        (3, 5, 3),
        name="applied_action_plan",
        leading_shape=leading,
    )
    _require_trailing_shape(
        previous_command,
        (3,),
        name="previous_command",
        leading_shape=leading,
    )
    _require_trailing_shape(
        observed_control_history,
        (3, 5, 2),
        name="observed_control_history",
        leading_shape=leading,
    )
    _require_trailing_shape(
        deterministic_kinematic_outcome,
        (6,),
        name="deterministic_kinematic_outcome",
        leading_shape=leading,
    )
    values = (
        waypoint_features,
        route_role_one_hot,
        requested_action_plan,
        applied_action_plan,
        previous_command,
        observed_control_history,
        deterministic_kinematic_outcome,
    )
    if any(not value.is_floating_point() for value in values):
        raise PlanAwareCostContractError("all feature components must be floating point")
    dtype, device = waypoint_features.dtype, waypoint_features.device
    if any(value.dtype != dtype or value.device != device for value in values):
        raise PlanAwareCostContractError(
            "all feature components must share one dtype and device"
        )

    def flattened(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(*leading, -1)

    base = torch.cat(
        [
            waypoint_features,
            route_role_one_hot,
            flattened(requested_action_plan),
            flattened(applied_action_plan),
            previous_command,
            flattened(observed_control_history),
            deterministic_kinematic_outcome,
        ],
        dim=-1,
    )
    applied_active = applied_action_plan[..., (0, 2)].reshape(*leading, -1)
    previous_active = previous_command[..., (0, 2)]
    query = torch.cat(
        [
            waypoint_features,
            route_role_one_hot,
            applied_active,
            previous_active,
            flattened(observed_control_history),
        ],
        dim=-1,
    )
    if base.shape[-1] != BASE_FEATURE_DIM or query.shape[-1] != QUERY_FEATURE_DIM:
        raise RuntimeError("assembled feature width disagrees with frozen contract")
    return base, query


def kinematic_rank_scores(rank_costs: torch.Tensor) -> torch.Tensor:
    """Convert the existing zero-best kinematic rank cost to a larger-best score."""

    if not isinstance(rank_costs, torch.Tensor) or not rank_costs.is_floating_point():
        raise PlanAwareCostContractError("kinematic rank costs must be floating point")
    if not torch.isfinite(rank_costs).all():
        raise PlanAwareCostContractError("kinematic rank costs must be finite")
    return -rank_costs


@contextmanager
def _isolated_cpu_seed(seed: int) -> Iterator[None]:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise PlanAwareCostContractError("model seed must be an integer in [0, 2**63)")
    # Construction consumes only CPU randomness.  fork_rng restores the caller's
    # RNG state and makes repeated direct construction deterministic.
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(seed)
        yield


def deterministic_seed_key(seed: int, key_id: str) -> str:
    """Derive one frozen SHA-256 key from the seed family and exact key ID."""

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise PlanAwareCostContractError("model seed must be an integer in [0, 2**63)")
    if not isinstance(key_id, str) or not key_id:
        raise PlanAwareCostContractError("seed key ID must be a nonempty string")
    preimage = (
        SEED_KEY_NAMESPACE.encode("utf-8")
        + b"\x00"
        + seed.to_bytes(8, "big", signed=False)
        + b"\x00"
        + key_id.encode("utf-8")
    )
    return hashlib.sha256(preimage).hexdigest()


def _seed_from_key(seed: int, key_id: str) -> int:
    # Torch accepts a signed-63-bit seed.  The full condition key remains the
    # persisted identity; this projection is frozen solely for RNG seeding.
    return int(deterministic_seed_key(seed, key_id)[:16], 16) & (2**63 - 1)


def condition_seed_keys(seed: int = MODEL_SEED) -> dict[str, str]:
    """Return the exact shared and condition-specific keyed-seed authorities."""

    return {
        SHARED_BASE_KEY_ID: deterministic_seed_key(seed, SHARED_BASE_KEY_ID),
        KINEMATIC_PLUS_NO_LATENT_RESIDUAL: deterministic_seed_key(
            seed, KINEMATIC_PLUS_NO_LATENT_RESIDUAL
        ),
        KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL: deterministic_seed_key(
            seed, KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL
        ),
    }


class _BaseResidual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(BASE_FEATURE_DIM, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        # Both learned scorers begin at the unchanged deterministic anchor.
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, base_features: torch.Tensor) -> torch.Tensor:
        return self.layers(base_features).squeeze(-1)


def _validate_base_and_anchor(
    base_features: torch.Tensor, kinematic_anchor: torch.Tensor
) -> torch.Size:
    leading = _require_trailing_shape(
        base_features, (BASE_FEATURE_DIM,), name="base_features"
    )
    if not isinstance(kinematic_anchor, torch.Tensor):
        raise PlanAwareCostContractError("kinematic_anchor must be a torch.Tensor")
    if kinematic_anchor.shape != leading:
        raise PlanAwareCostContractError(
            f"kinematic_anchor shape {tuple(kinematic_anchor.shape)} != {tuple(leading)}"
        )
    if not base_features.is_floating_point() or not kinematic_anchor.is_floating_point():
        raise PlanAwareCostContractError("model inputs must be floating point")
    if (
        base_features.dtype != kinematic_anchor.dtype
        or base_features.device != kinematic_anchor.device
    ):
        raise PlanAwareCostContractError(
            "base_features and kinematic_anchor must share dtype and device"
        )
    return leading


class KinematicPlusNoLatentResidual(nn.Module):
    """Unchanged kinematic score plus a learned 138-D non-latent residual."""

    model_id = KINEMATIC_PLUS_NO_LATENT_RESIDUAL

    def __init__(self, *, seed: int = MODEL_SEED) -> None:
        super().__init__()
        self.seed = seed
        self.condition_seed_key = deterministic_seed_key(
            seed, KINEMATIC_PLUS_NO_LATENT_RESIDUAL
        )
        self.shared_base_seed_key = deterministic_seed_key(seed, SHARED_BASE_KEY_ID)
        with _isolated_cpu_seed(_seed_from_key(seed, SHARED_BASE_KEY_ID)):
            self.base_residual = _BaseResidual()
        count = parameter_count(self)
        if count != NO_LATENT_PARAMETER_COUNT or count >= NO_LATENT_PARAMETER_CAP_EXCLUSIVE:
            raise RuntimeError(f"no-latent parameter contract violated: {count}")

    def residual(self, base_features: torch.Tensor) -> torch.Tensor:
        _require_trailing_shape(base_features, (BASE_FEATURE_DIM,), name="base_features")
        return self.base_residual(base_features)

    def forward(
        self, base_features: torch.Tensor, kinematic_anchor: torch.Tensor
    ) -> torch.Tensor:
        _validate_base_and_anchor(base_features, kinematic_anchor)
        return kinematic_anchor + self.base_residual(base_features)


@dataclass(frozen=True)
class LatentScoreComponents:
    score: torch.Tensor
    kinematic_plus_base_score: torch.Tensor
    base_residual: torch.Tensor
    latent_residual: torch.Tensor
    attention_weights: torch.Tensor


class KinematicPlusJepaLatentResidual(nn.Module):
    """Kinematic+base score plus a goal/action-query JEPA latent residual."""

    model_id = KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL

    def __init__(self, *, seed: int = MODEL_SEED) -> None:
        super().__init__()
        self.seed = seed
        self.condition_seed_key = deterministic_seed_key(
            seed, KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL
        )
        self.shared_base_seed_key = deterministic_seed_key(seed, SHARED_BASE_KEY_ID)
        # The common base has its own family key so its bytes match exactly.
        with _isolated_cpu_seed(_seed_from_key(seed, SHARED_BASE_KEY_ID)):
            self.base_residual = _BaseResidual()
        # Parameters unique to the latent condition use the exact condition-ID
        # key.  Their stream cannot perturb the matched shared-base stream.
        with _isolated_cpu_seed(
            _seed_from_key(seed, KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL)
        ):
            self.token_layer_norm = nn.LayerNorm(TOKEN_DIM)
            self.shared_token_projection = nn.Linear(TOKEN_DIM, LATENT_WIDTH)
            self.query_projection = nn.Linear(QUERY_FEATURE_DIM, LATENT_WIDTH)
            self.latent_residual_mlp = nn.Sequential(
                nn.Linear(TIMEPOINT_COUNT * LATENT_WIDTH + BASE_FEATURE_DIM, 256),
                nn.GELU(),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Linear(128, 1),
            )
            nn.init.zeros_(self.latent_residual_mlp[-1].weight)
            nn.init.zeros_(self.latent_residual_mlp[-1].bias)
        count = parameter_count(self)
        if count != JEPA_LATENT_PARAMETER_COUNT or count >= JEPA_LATENT_PARAMETER_CAP_EXCLUSIVE:
            raise RuntimeError(f"JEPA-latent parameter contract violated: {count}")

    def _attention_pool(
        self,
        query_features: torch.Tensor,
        trajectory_tokens: torch.Tensor,
        *,
        expected_leading: torch.Size,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _require_trailing_shape(
            query_features,
            (QUERY_FEATURE_DIM,),
            name="query_features",
            leading_shape=expected_leading,
        )
        _require_trailing_shape(
            trajectory_tokens,
            (TIMEPOINT_COUNT, TOKENS_PER_TIMEPOINT, TOKEN_DIM),
            name="trajectory_tokens",
            leading_shape=expected_leading,
        )
        if (
            query_features.dtype != trajectory_tokens.dtype
            or query_features.device != trajectory_tokens.device
        ):
            raise PlanAwareCostContractError(
                "query_features and trajectory_tokens must share dtype and device"
            )
        # Preserve the encoder's registered spatial semantics explicitly before
        # the shared projection/attention: [768,1024] -> [24,32,1024] ->
        # [768,1024], with flat_index = y * 32 + x throughout.
        token_grid = reshape_tokens_to_spatial_grid(trajectory_tokens)
        flattened_tokens = flatten_spatial_token_grid(token_grid)
        projected = self.shared_token_projection(
            self.token_layer_norm(flattened_tokens)
        )
        query = self.query_projection(query_features)
        logits = torch.einsum("...d,...tnd->...tn", query, projected) / math.sqrt(
            LATENT_WIDTH
        )
        attention = torch.softmax(logits, dim=-1)
        pooled = torch.einsum("...tn,...tnd->...td", attention, projected)
        return pooled, attention

    def _assemble_trajectory_tokens(
        self,
        base_features: torch.Tensor,
        current_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Broadcast candidate-invariant CURRENT and append candidate H1--H3."""

        leading = base_features.shape[:-1]
        _require_trailing_shape(
            future_tokens,
            (TIMEPOINT_COUNT - 1, TOKENS_PER_TIMEPOINT, TOKEN_DIM),
            name="future_tokens",
            leading_shape=leading,
        )
        candidate_specific_current_shape = torch.Size(
            (*leading, TOKENS_PER_TIMEPOINT, TOKEN_DIM)
        )
        if current_tokens.shape == torch.Size((TOKENS_PER_TIMEPOINT, TOKEN_DIM)):
            current = current_tokens.reshape(
                *((1,) * len(leading)), TOKENS_PER_TIMEPOINT, TOKEN_DIM
            ).expand(*leading, TOKENS_PER_TIMEPOINT, TOKEN_DIM)
        elif current_tokens.shape == candidate_specific_current_shape:
            current = current_tokens
        else:
            raise PlanAwareCostContractError(
                "current_tokens must be candidate-invariant [768,1024] or align "
                f"with candidates as {tuple(candidate_specific_current_shape)}; "
                f"got {tuple(current_tokens.shape)}"
            )
        if current.dtype != future_tokens.dtype or current.device != future_tokens.device:
            raise PlanAwareCostContractError(
                "current_tokens and future_tokens must share dtype and device"
            )
        return torch.cat([current.unsqueeze(-3), future_tokens], dim=-3)

    def kinematic_plus_base_score(
        self, base_features: torch.Tensor, kinematic_anchor: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate this model's own zero-latent comparison path exactly."""

        _validate_base_and_anchor(base_features, kinematic_anchor)
        return kinematic_anchor + self.base_residual(base_features)

    def score_components(
        self,
        base_features: torch.Tensor,
        query_features: torch.Tensor,
        kinematic_anchor: torch.Tensor,
        current_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
    ) -> LatentScoreComponents:
        leading = _validate_base_and_anchor(base_features, kinematic_anchor)
        trajectory_tokens = self._assemble_trajectory_tokens(
            base_features, current_tokens, future_tokens
        )
        pooled, attention = self._attention_pool(
            query_features, trajectory_tokens, expected_leading=leading
        )
        base_residual = self.base_residual(base_features)
        latent_and_nonlatent = torch.cat(
            [
                pooled.reshape(*leading, TIMEPOINT_COUNT * LATENT_WIDTH),
                base_features,
            ],
            dim=-1,
        )
        latent_residual = self.latent_residual_mlp(
            latent_and_nonlatent
        ).squeeze(-1)
        kinematic_plus_base = kinematic_anchor + base_residual
        return LatentScoreComponents(
            score=kinematic_plus_base + latent_residual,
            kinematic_plus_base_score=kinematic_plus_base,
            base_residual=base_residual,
            latent_residual=latent_residual,
            attention_weights=attention,
        )

    def forward(
        self,
        base_features: torch.Tensor,
        query_features: torch.Tensor,
        kinematic_anchor: torch.Tensor,
        current_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return self.score_components(
            base_features,
            query_features,
            kinematic_anchor,
            current_tokens,
            future_tokens,
        ).score


def parameter_count(model: nn.Module) -> int:
    """Return the exact trainable parameter count."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def matched_base_initialisation(
    no_latent: KinematicPlusNoLatentResidual,
    jepa_latent: KinematicPlusJepaLatentResidual,
) -> bool:
    """Whether the two models' base-residual state bytes are tensor-identical."""

    left = no_latent.base_residual.state_dict()
    right = jepa_latent.base_residual.state_dict()
    return left.keys() == right.keys() and all(
        torch.equal(left[name], right[name]) for name in left
    )


def _digest_blob(hasher: Any, value: bytes) -> None:
    hasher.update(struct.pack(">Q", len(value)))
    hasher.update(value)


def shared_base_digest(
    model: KinematicPlusNoLatentResidual | KinematicPlusJepaLatentResidual,
) -> str:
    """Canonical SHA-256 of one model's base-residual tensor state."""

    if not isinstance(
        model, (KinematicPlusNoLatentResidual, KinematicPlusJepaLatentResidual)
    ):
        raise PlanAwareCostContractError("model lacks the registered base residual")
    hasher = hashlib.sha256()
    _digest_blob(hasher, b"PLAN_AWARE_MONOTONE_JEPA_COST_V1/BASE_STATE_V1")
    for name, value in sorted(model.base_residual.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        _digest_blob(hasher, name.encode("utf-8"))
        _digest_blob(hasher, str(tensor.dtype).encode("ascii"))
        shape = b"".join(struct.pack(">Q", int(size)) for size in tensor.shape)
        _digest_blob(hasher, shape)
        _digest_blob(hasher, tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return hasher.hexdigest()


def assert_matched_initialisation(
    no_latent: KinematicPlusNoLatentResidual,
    jepa_latent: KinematicPlusJepaLatentResidual,
) -> None:
    """Fail closed unless seed and byte-canonical base state match exactly."""

    if no_latent.seed != jepa_latent.seed:
        raise RuntimeError("paired rankers use different seed identifiers")
    keys = condition_seed_keys(no_latent.seed)
    if (
        no_latent.shared_base_seed_key != keys[SHARED_BASE_KEY_ID]
        or jepa_latent.shared_base_seed_key != keys[SHARED_BASE_KEY_ID]
        or no_latent.condition_seed_key
        != keys[KINEMATIC_PLUS_NO_LATENT_RESIDUAL]
        or jepa_latent.condition_seed_key
        != keys[KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL]
    ):
        raise RuntimeError("paired ranker keyed-seed authority drift")
    if not matched_base_initialisation(no_latent, jepa_latent):
        raise RuntimeError("paired ranker base tensors are not identical")
    if shared_base_digest(no_latent) != shared_base_digest(jepa_latent):
        raise RuntimeError("paired ranker canonical base digests differ")


def model_contract(condition: str | None = None) -> dict[str, object]:
    """Return the machine-readable, value-free architecture contract."""

    allowed_conditions = {
        KINEMATIC_PLUS_NO_LATENT_RESIDUAL,
        KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL,
    }
    if condition is not None and condition not in allowed_conditions:
        raise PlanAwareCostContractError(f"unknown registered model {condition!r}")
    output: dict[str, object] = {
        "schema": "plan_aware_monotone_jepa_cost_v1.model_contract.v1",
        "status": STATUS,
        "seed": MODEL_SEED,
        "score_orientation": "larger_is_better",
        "kinematic_anchor": "negative_exact_kinematic_rank_costs",
        "goal_tokens_used": False,
        "base_features": {
            "dim": BASE_FEATURE_DIM,
            "slices": {
                name: [int(value.start), int(value.stop)]
                for name, value in BASE_FEATURE_SLICES.items()
            },
        },
        "query_features": {
            "dim": QUERY_FEATURE_DIM,
            "slices": {
                name: [int(value.start), int(value.stop)]
                for name, value in QUERY_FEATURE_SLICES.items()
            },
        },
        "models": {
            KINEMATIC_PLUS_NO_LATENT_RESIDUAL: {
                "base_mlp": [BASE_FEATURE_DIM, 128, 64, 1],
                "activations": ["GELU", "GELU"],
                "trainable_parameters": NO_LATENT_PARAMETER_COUNT,
                "parameter_cap_exclusive": NO_LATENT_PARAMETER_CAP_EXCLUSIVE,
            },
            KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL: {
                "base_mlp": [BASE_FEATURE_DIM, 128, 64, 1],
                "token_layer_norm": {
                    "dim": TOKEN_DIM,
                    "elementwise_affine": True,
                    "shared_across_timepoints": True,
                },
                "shared_token_projection": [TOKEN_DIM, LATENT_WIDTH],
                "query_projection": [QUERY_FEATURE_DIM, LATENT_WIDTH],
                "attention": {
                    "timepoints": list(TIMEPOINTS),
                    "tokens_per_timepoint": TOKENS_PER_TIMEPOINT,
                    "token_grid_shape": list(TOKEN_GRID_SHAPE),
                    "token_grid_storage_order": "row-major",
                    "token_flat_index": "y * 32 + x",
                    "token_grid_transform": (
                        "view [...,768,1024] as [...,24,32,1024], then flatten "
                        "the spatial grid in the same row-major order before "
                        "shared LayerNorm/projection/attention"
                    ),
                    "single_query_shared_across_timepoints": True,
                    "scale": f"1/sqrt({LATENT_WIDTH})",
                    "softmax_axis": "spatial_tokens",
                },
                "latent_residual_mlp": [
                    TIMEPOINT_COUNT * LATENT_WIDTH + BASE_FEATURE_DIM,
                    256,
                    128,
                    1,
                ],
                "latent_residual_activations": ["GELU", "GELU"],
                "trainable_parameters": JEPA_LATENT_PARAMETER_COUNT,
                "parameter_cap_exclusive": JEPA_LATENT_PARAMETER_CAP_EXCLUSIVE,
            },
        },
        "initialisation": {
            "seed_family": MODEL_SEED,
            "key_namespace": SEED_KEY_NAMESPACE,
            "condition_seed_keys": condition_seed_keys(MODEL_SEED),
            "paired_base_tensor_bytes_identical": True,
            "residual_output_layers_zero": True,
            "global_cpu_rng_state_restored": True,
        },
        "loss": {
            "pairwise": {
                "formula": "softplus(-y_ij*(S_i-S_j))",
                "target": (
                    "sign(conditioned margin-Borda utility_i - utility_j) when "
                    "absolute difference > 1e-12; otherwise zero"
                ),
                "utility_tolerance": PAIRWISE_UTILITY_TOLERANCE,
                "direct_route_preference_role": (
                    "constructs margin-Borda utility only; not a separate "
                    "pairwise target"
                ),
                "weight": PAIRWISE_LOSS_WEIGHT,
            },
            "listwise": {
                "target": "softmax(route_utility)",
                "temperature": LISTWISE_TEMPERATURE,
                "weight": LISTWISE_LOSS_WEIGHT,
            },
            "residual_l2": {
                "formula": "mean((score-kinematic_anchor)^2)",
                "weight": RESIDUAL_LOSS_WEIGHT,
            },
        },
    }
    if condition is not None:
        output["selected_model"] = condition
    return output


def build_matched_rankers(
    seed: int = MODEL_SEED,
) -> tuple[KinematicPlusNoLatentResidual, KinematicPlusJepaLatentResidual]:
    """Construct the exactly two prospectively registered paired rankers."""

    no_latent = KinematicPlusNoLatentResidual(seed=seed)
    jepa_latent = KinematicPlusJepaLatentResidual(seed=seed)
    assert_matched_initialisation(no_latent, jepa_latent)
    return no_latent, jepa_latent


@dataclass(frozen=True)
class RouteOnlyLoss:
    total: torch.Tensor
    pairwise: torch.Tensor
    listwise: torch.Tensor
    residual_l2: torch.Tensor
    ordered_pair_count: int
    candidate_count: int


def route_only_loss(
    *,
    scores: torch.Tensor,
    route_utilities: torch.Tensor,
    kinematic_anchor: torch.Tensor,
    pairwise_targets: torch.Tensor,
    admissible_mask: torch.Tensor | None = None,
) -> RouteOnlyLoss:
    """Pairwise/listwise local-route ordering plus a small anchor residual.

    ``route_utilities`` must be the prospectively frozen local-waypoint
    margin-Borda utility.  ``pairwise_targets`` is the sign of the conditioned
    margin-Borda utility difference only when its magnitude exceeds 1e-12 and
    must be antisymmetric with values in ``{-1, 0, +1}``; zero denotes a
    utility tie within that tolerance.  The frozen direct route comparator
    constructs Borda utility but is not a separate pairwise target.  This API
    intentionally has no contact, safety, material-hazard, stuck, or deployment
    label.  For every utility-ordered pair the loss is
    ``softplus(-y_ij * (S_i - S_j))``.  Listwise targets and
    predictions use temperature-1 softmax over the supplied candidate mask.
    """

    if not isinstance(scores, torch.Tensor) or scores.ndim != 2:
        raise PlanAwareCostContractError("scores must have shape [states, candidates]")
    if route_utilities.shape != scores.shape or kinematic_anchor.shape != scores.shape:
        raise PlanAwareCostContractError(
            "scores, route_utilities, and kinematic_anchor must have identical shape"
        )
    if (
        not scores.is_floating_point()
        or not route_utilities.is_floating_point()
        or not kinematic_anchor.is_floating_point()
    ):
        raise PlanAwareCostContractError("route-only loss inputs must be floating point")
    if route_utilities.dtype != scores.dtype or kinematic_anchor.dtype != scores.dtype:
        raise PlanAwareCostContractError("route-only loss inputs must share dtype")
    if route_utilities.device != scores.device or kinematic_anchor.device != scores.device:
        raise PlanAwareCostContractError("route-only loss inputs must share device")
    if (
        not torch.isfinite(scores).all()
        or not torch.isfinite(route_utilities).all()
        or not torch.isfinite(kinematic_anchor).all()
    ):
        raise PlanAwareCostContractError("route-only loss inputs must be finite")
    expected_pairwise_shape = (*scores.shape, scores.shape[-1])
    if not isinstance(pairwise_targets, torch.Tensor) or pairwise_targets.shape != expected_pairwise_shape:
        raise PlanAwareCostContractError(
            f"pairwise_targets shape must be {expected_pairwise_shape}"
        )
    if pairwise_targets.dtype == torch.bool or not (
        pairwise_targets.is_floating_point()
        or pairwise_targets.dtype
        in (torch.int8, torch.int16, torch.int32, torch.int64)
    ):
        raise PlanAwareCostContractError("pairwise_targets must be numeric, not Boolean")
    if pairwise_targets.device != scores.device or not torch.isfinite(pairwise_targets).all():
        raise PlanAwareCostContractError(
            "pairwise_targets must be finite and share the score device"
        )
    allowed_targets = (
        (pairwise_targets == -1)
        | (pairwise_targets == 0)
        | (pairwise_targets == 1)
    )
    if not torch.all(allowed_targets):
        raise PlanAwareCostContractError(
            "pairwise_targets values must belong to {-1,0,+1}"
        )
    if not torch.equal(pairwise_targets, -pairwise_targets.transpose(-1, -2)):
        raise PlanAwareCostContractError("pairwise_targets must be antisymmetric")
    if admissible_mask is None:
        mask = torch.ones_like(scores, dtype=torch.bool)
    else:
        if admissible_mask.shape != scores.shape or admissible_mask.dtype != torch.bool:
            raise PlanAwareCostContractError(
                "admissible_mask must be Boolean with the score shape"
            )
        mask = admissible_mask.to(device=scores.device)
    if not torch.all(mask.any(dim=-1)):
        raise PlanAwareCostContractError(
            "every state must contain at least one admissible candidate"
        )
    utility_difference = route_utilities.unsqueeze(-1) - route_utilities.unsqueeze(-2)
    expected_targets = torch.where(
        utility_difference.abs() > PAIRWISE_UTILITY_TOLERANCE,
        torch.sign(utility_difference),
        torch.zeros_like(utility_difference),
    )
    conditioned_pairs = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    if not torch.equal(
        pairwise_targets.to(dtype=route_utilities.dtype).masked_select(
            conditioned_pairs
        ),
        expected_targets.masked_select(conditioned_pairs),
    ):
        raise PlanAwareCostContractError(
            "pairwise_targets must equal sign of conditioned margin-Borda utility "
            "differences above the frozen tolerance"
        )

    candidate_count = scores.shape[-1]
    upper = torch.triu(
        torch.ones(candidate_count, candidate_count, dtype=torch.bool, device=scores.device),
        diagonal=1,
    )
    valid_pairs = mask.unsqueeze(-1) & mask.unsqueeze(-2) & upper
    ordered_pairs = valid_pairs & (pairwise_targets != 0)
    score_difference = scores.unsqueeze(-1) - scores.unsqueeze(-2)
    pair_targets = pairwise_targets.to(dtype=scores.dtype)
    pair_terms = F.softplus(-pair_targets * score_difference)
    ordered_pair_count = int(ordered_pairs.sum().item())
    if ordered_pair_count:
        pairwise = pair_terms.masked_select(ordered_pairs).mean()
    else:
        # Retain a valid zero-gradient graph when every route utility is tied.
        pairwise = scores.sum() * 0.0

    negative_infinity = torch.tensor(
        float("-inf"), dtype=scores.dtype, device=scores.device
    )
    target_logits = torch.where(mask, route_utilities / LISTWISE_TEMPERATURE, negative_infinity)
    score_logits = torch.where(mask, scores / LISTWISE_TEMPERATURE, negative_infinity)
    target_probability = torch.softmax(target_logits, dim=-1)
    log_probability = torch.log_softmax(score_logits, dim=-1)
    safe_log_probability = torch.where(mask, log_probability, torch.zeros_like(log_probability))
    listwise = -(target_probability * safe_log_probability).sum(dim=-1).mean()

    residual = scores - kinematic_anchor
    residual_l2 = residual.square().masked_select(mask).mean()
    total = (
        PAIRWISE_LOSS_WEIGHT * pairwise
        + LISTWISE_LOSS_WEIGHT * listwise
        + RESIDUAL_LOSS_WEIGHT * residual_l2
    )
    return RouteOnlyLoss(
        total=total,
        pairwise=pairwise,
        listwise=listwise,
        residual_l2=residual_l2,
        ordered_pair_count=ordered_pair_count,
        candidate_count=int(mask.sum().item()),
    )


def route_ordering_loss(
    scores: torch.Tensor,
    route_utilities: torch.Tensor,
    kinematic_anchor: torch.Tensor,
    pairwise_targets: torch.Tensor,
    *,
    admissible_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluator-facing names for the exact route-only loss components.

    The anchor is mandatory: using zero here would regularise absolute scores
    instead of the learned residual and would violate the registered objective.
    """

    result = route_only_loss(
        scores=scores,
        route_utilities=route_utilities,
        kinematic_anchor=kinematic_anchor,
        pairwise_targets=pairwise_targets,
        admissible_mask=admissible_mask,
    )
    return {
        "loss": result.total,
        "pair": result.pairwise,
        "list": result.listwise,
        "residual": result.residual_l2,
    }


__all__ = [
    "BASE_FEATURE_DIM",
    "BASE_FEATURE_SLICES",
    "JEPA_LATENT_PARAMETER_CAP_EXCLUSIVE",
    "JEPA_LATENT_PARAMETER_COUNT",
    "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL",
    "KINEMATIC_PLUS_NO_LATENT_RESIDUAL",
    "KinematicPlusJepaLatentResidual",
    "KinematicPlusNoLatentResidual",
    "LATENT_WIDTH",
    "LISTWISE_LOSS_WEIGHT",
    "LISTWISE_TEMPERATURE",
    "LatentScoreComponents",
    "MODEL_SEED",
    "NO_LATENT_PARAMETER_CAP_EXCLUSIVE",
    "NO_LATENT_PARAMETER_COUNT",
    "PAIRWISE_LOSS_WEIGHT",
    "PAIRWISE_UTILITY_TOLERANCE",
    "PlanAwareCostContractError",
    "QUERY_FEATURE_DIM",
    "QUERY_FEATURE_SLICES",
    "RESIDUAL_LOSS_WEIGHT",
    "RouteOnlyLoss",
    "SEED_KEY_NAMESPACE",
    "SHARED_BASE_KEY_ID",
    "STATUS",
    "TIMEPOINTS",
    "TOKEN_DIM",
    "TOKEN_GRID_SHAPE",
    "TOKENS_PER_TIMEPOINT",
    "assemble_base_and_query_features",
    "assert_matched_initialisation",
    "build_matched_rankers",
    "condition_seed_keys",
    "deterministic_seed_key",
    "flatten_spatial_token_grid",
    "kinematic_rank_scores",
    "matched_base_initialisation",
    "model_contract",
    "parameter_count",
    "route_only_loss",
    "route_ordering_loss",
    "reshape_tokens_to_spatial_grid",
    "shared_base_digest",
]
