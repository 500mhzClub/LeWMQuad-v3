"""Learned values over one immutable two-resolution G4 candidate set.

The module scores the exact supplied row order from detached Shared V5 caches.
It has no encoder, frame runner, image preprocessor, candidate generator,
candidate filter, planner, path loader, or fallback selector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re

import torch
import torch.nn as nn


FRONTIER_VALUE_HEAD_CONFIG_SCHEMA_V1 = "lewm_go2_two_resolution_frontier_value_head_config_v1"
FROZEN_CANDIDATE_FEATURE_BATCH_SCHEMA_V1 = "lewm_go2_frozen_candidate_feature_batch_v1"
PRODUCTION_TWO_RESOLUTION_FRONTIER_VALUE_HEAD_V1 = None
PRODUCTION_G4_HEAD_CONFIG_SHA256_V1 = None
PRODUCTION_G4_HEAD_CHECKPOINT_SHA256_V1 = None
PRODUCTION_G4_HEAD_CALIBRATION_SHA256_V1 = None
PRODUCTION_G4_CANDIDATE_CONFIGURATION_SHA256_V1 = None

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class TwoResolutionFrontierValueHeadV1Error(ValueError):
    """The exact candidate or detached-feature contract was violated."""


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise TwoResolutionFrontierValueHeadV1Error(f"{name} must be a lowercase SHA-256")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise TwoResolutionFrontierValueHeadV1Error(f"{name} must be a positive exact integer")
    return value


def _require_detached_finite_tensor(
    value: object,
    *,
    name: str,
    rank: int,
) -> torch.Tensor:
    if type(value) is not torch.Tensor:
        raise TwoResolutionFrontierValueHeadV1Error(f"{name} must be an exact torch.Tensor")
    if value.ndim != rank or not value.dtype.is_floating_point:
        raise TwoResolutionFrontierValueHeadV1Error(f"{name} rank/dtype is invalid")
    if value.requires_grad or value.grad_fn is not None:
        raise TwoResolutionFrontierValueHeadV1Error(f"{name} must be detached cached features")
    if value.numel() == 0 or not bool(torch.isfinite(value).all().item()):
        raise TwoResolutionFrontierValueHeadV1Error(f"{name} must be nonempty and finite")
    return value


def _tensor_content_sha256(value: torch.Tensor) -> str:
    if value.dtype not in {torch.float32, torch.float64}:
        raise TwoResolutionFrontierValueHeadV1Error(
            "candidate features must use float32 or float64 for canonical hashing"
        )
    canonical = value.detach().to(device="cpu").contiguous()
    header = json.dumps(
        {"dtype": str(canonical.dtype), "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenCandidateFeatureBatchV1:
    """Content-bound candidate rows in their already-canonical source order."""

    candidate_set_sha256: str
    candidate_row_sha256s: tuple[str, ...]
    features: torch.Tensor = field(repr=False, compare=False)
    schema: str = FROZEN_CANDIDATE_FEATURE_BATCH_SCHEMA_V1
    feature_tensor_sha256: str = field(init=False)
    content_sha256: str = field(init=False)
    _feature_version: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_sha256(self.candidate_set_sha256, name="candidate_set_sha256")
        if type(self.candidate_row_sha256s) is not tuple or not self.candidate_row_sha256s:
            raise TwoResolutionFrontierValueHeadV1Error(
                "candidate_row_sha256s must be a nonempty exact tuple"
            )
        for index, item in enumerate(self.candidate_row_sha256s):
            _require_sha256(item, name=f"candidate_row_sha256s[{index}]")
        if len(set(self.candidate_row_sha256s)) != len(self.candidate_row_sha256s):
            raise TwoResolutionFrontierValueHeadV1Error("candidate rows must be unique")
        features = _require_detached_finite_tensor(self.features, name="features", rank=3)
        if int(features.shape[1]) != len(self.candidate_row_sha256s):
            raise TwoResolutionFrontierValueHeadV1Error(
                "candidate feature rows do not match exact candidate set"
            )
        if type(self.schema) is not str or self.schema != FROZEN_CANDIDATE_FEATURE_BATCH_SCHEMA_V1:
            raise TwoResolutionFrontierValueHeadV1Error("candidate feature schema changed")
        tensor_sha = _tensor_content_sha256(features)
        object.__setattr__(self, "feature_tensor_sha256", tensor_sha)
        object.__setattr__(self, "_feature_version", features._version)
        object.__setattr__(
            self,
            "content_sha256",
            _canonical_sha256(
                {
                    "schema": self.schema,
                    "version": 1,
                    "candidate_set_sha256": self.candidate_set_sha256,
                    "candidate_row_sha256s": list(self.candidate_row_sha256s),
                    "feature_tensor_sha256": tensor_sha,
                }
            ),
        )

    def assert_unchanged(self) -> None:
        if self.features._version != self._feature_version:
            raise TwoResolutionFrontierValueHeadV1Error("candidate features were mutated")
        if _tensor_content_sha256(self.features) != self.feature_tensor_sha256:
            raise TwoResolutionFrontierValueHeadV1Error("candidate feature content changed")

    def binding_dict(self) -> dict[str, object]:
        self.assert_unchanged()
        return {
            "schema": self.schema,
            "version": 1,
            "candidate_set_sha256": self.candidate_set_sha256,
            "candidate_row_sha256s": list(self.candidate_row_sha256s),
            "feature_tensor_sha256": self.feature_tensor_sha256,
            "content_sha256": self.content_sha256,
        }

    def __copy__(self) -> object:
        raise TypeError("frozen candidate batch cannot be copied")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise TypeError("frozen candidate batch cannot be deep-copied")

    def __reduce__(self) -> object:
        raise TypeError("frozen candidate batch cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("frozen candidate batch cannot be serialized")


@dataclass(frozen=True)
class TwoResolutionFrontierValueHeadConfigV1:
    patch_feature_dim: int
    bev_feature_dim: int
    candidate_feature_dim: int
    hidden_dim: int = 128
    schema: str = FRONTIER_VALUE_HEAD_CONFIG_SCHEMA_V1

    def __post_init__(self) -> None:
        for name in (
            "patch_feature_dim",
            "bev_feature_dim",
            "candidate_feature_dim",
            "hidden_dim",
        ):
            _positive_int(getattr(self, name), name=name)
        if type(self.schema) is not str or self.schema != FRONTIER_VALUE_HEAD_CONFIG_SCHEMA_V1:
            raise TwoResolutionFrontierValueHeadV1Error("frontier value head schema changed")

    @property
    def content_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class FrontierValueScoresV1:
    """Finite scalar values preserving the exact input candidate object/order."""

    candidate_batch: FrozenCandidateFeatureBatchV1 = field(repr=False, compare=False)
    candidate_set_sha256: str
    candidate_row_sha256s: tuple[str, ...]
    scores: torch.Tensor

    def __post_init__(self) -> None:
        if type(self.candidate_batch) is not FrozenCandidateFeatureBatchV1:
            raise TwoResolutionFrontierValueHeadV1Error("candidate_batch type changed")
        self.candidate_batch.assert_unchanged()
        if self.candidate_set_sha256 != self.candidate_batch.candidate_set_sha256:
            raise TwoResolutionFrontierValueHeadV1Error("candidate set identity changed")
        if self.candidate_row_sha256s != self.candidate_batch.candidate_row_sha256s:
            raise TwoResolutionFrontierValueHeadV1Error("candidate row order changed")
        if type(self.scores) is not torch.Tensor or self.scores.ndim != 2:
            raise TwoResolutionFrontierValueHeadV1Error("scores must be exact [batch, rows] tensor")
        if tuple(self.scores.shape) != tuple(self.candidate_batch.features.shape[:2]):
            raise TwoResolutionFrontierValueHeadV1Error("score shape differs from candidate set")
        if not self.scores.dtype.is_floating_point or not bool(torch.isfinite(self.scores).all().item()):
            raise TwoResolutionFrontierValueHeadV1Error("scores must be finite floating values")

    def selected_row_indices(self) -> tuple[int, ...]:
        """Maximum score with torch.argmax's first-row tie break."""

        self.candidate_batch.assert_unchanged()
        return tuple(int(item) for item in torch.argmax(self.scores, dim=1).tolist())


class TwoResolutionFrontierValueHeadV1(nn.Module):
    """Scores but never generates, filters, sorts, or mutates G4 candidates."""

    owned_encoder_count = 0
    owned_rgb_preprocessor_count = 0
    owns_candidate_generator = False
    has_fallback_selector = False

    def __init__(self, config: TwoResolutionFrontierValueHeadConfigV1) -> None:
        super().__init__()
        if type(config) is not TwoResolutionFrontierValueHeadConfigV1:
            raise TypeError("config must be exact TwoResolutionFrontierValueHeadConfigV1")
        self.config = config
        context_dim = config.patch_feature_dim + config.bev_feature_dim
        self.context_projection = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, config.hidden_dim),
            nn.SiLU(),
        )
        self.candidate_projection = nn.Sequential(
            nn.LayerNorm(config.candidate_feature_dim),
            nn.Linear(config.candidate_feature_dim, config.hidden_dim),
            nn.SiLU(),
        )
        self.value_projection = nn.Sequential(
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    @property
    def architecture_config_sha256(self) -> str:
        return self.config.content_sha256

    def forward(
        self,
        patch_features: torch.Tensor,
        bev_features: torch.Tensor,
        candidate_batch: FrozenCandidateFeatureBatchV1,
    ) -> FrontierValueScoresV1:
        patch = _require_detached_finite_tensor(patch_features, name="patch_features", rank=3)
        bev = _require_detached_finite_tensor(bev_features, name="bev_features", rank=4)
        if patch.shape[-1] != self.config.patch_feature_dim:
            raise TwoResolutionFrontierValueHeadV1Error("patch feature dimension changed")
        if bev.shape[1] != self.config.bev_feature_dim:
            raise TwoResolutionFrontierValueHeadV1Error("BEV feature dimension changed")
        if type(candidate_batch) is not FrozenCandidateFeatureBatchV1:
            raise TypeError("candidate_batch must be exact FrozenCandidateFeatureBatchV1")
        candidate_batch.assert_unchanged()
        candidates = candidate_batch.features
        if candidates.shape[-1] != self.config.candidate_feature_dim:
            raise TwoResolutionFrontierValueHeadV1Error("candidate feature dimension changed")
        if patch.shape[0] != bev.shape[0] or patch.shape[0] != candidates.shape[0]:
            raise TwoResolutionFrontierValueHeadV1Error("feature batch sizes differ")
        if patch.device != bev.device or patch.device != candidates.device:
            raise TwoResolutionFrontierValueHeadV1Error("feature devices differ")
        patch_version = patch._version
        bev_version = bev._version
        candidate_version = candidates._version
        context = self.context_projection(
            torch.cat((patch.mean(dim=1), bev.mean(dim=(-2, -1))), dim=-1)
        )
        candidate_hidden = self.candidate_projection(candidates)
        expanded_context = context[:, None, :].expand(
            int(context.shape[0]), int(candidates.shape[1]), self.config.hidden_dim
        )
        scores = self.value_projection(
            torch.cat((expanded_context, candidate_hidden), dim=-1)
        ).squeeze(-1)
        if (
            patch._version != patch_version
            or bev._version != bev_version
            or candidates._version != candidate_version
        ):
            raise TwoResolutionFrontierValueHeadV1Error("an input feature tensor was mutated")
        candidate_batch.assert_unchanged()
        return FrontierValueScoresV1(
            candidate_batch=candidate_batch,
            candidate_set_sha256=candidate_batch.candidate_set_sha256,
            candidate_row_sha256s=candidate_batch.candidate_row_sha256s,
            scores=scores,
        )


def initialize_deterministic_mock_weights_v1(
    head: TwoResolutionFrontierValueHeadV1,
    *,
    seed: int,
) -> None:
    """Deterministic synthetic-test initialization; no trained identity is set."""

    if type(head) is not TwoResolutionFrontierValueHeadV1:
        raise TypeError("head must be exact TwoResolutionFrontierValueHeadV1")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be an exact nonnegative integer")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    with torch.no_grad():
        for parameter in head.parameters():
            values = torch.empty(parameter.shape, dtype=parameter.dtype, device="cpu")
            values.uniform_(-0.05, 0.05, generator=generator)
            parameter.copy_(values.to(device=parameter.device))


__all__ = [
    "FrozenCandidateFeatureBatchV1",
    "FrontierValueScoresV1",
    "TwoResolutionFrontierValueHeadConfigV1",
    "TwoResolutionFrontierValueHeadV1",
    "TwoResolutionFrontierValueHeadV1Error",
    "initialize_deterministic_mock_weights_v1",
]
