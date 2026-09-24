"""Source-only one-frame runtime and exact-object lease foundation.

Only an explicitly synthetic fake backend can be constructed in this source
version.  No production identity, artifact path, checkpoint loader, dataset,
scene, simulator, or evaluator surface exists here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

import torch
import torch.nn.functional as F

from lewm.benchmarks.go2_navigation_development_trace_v1 import (
    CANONICAL_COLORS,
    CallCounterPanelV1,
    canonical_json_sha256,
    decode_canonical_binary64_hex,
    require_identifier,
    require_nonnegative_int,
    require_sha256,
)
from lewm.models.shared_v5_target_observation_head_v1 import (
    FourColorTargetObservationOutputV1,
    SharedV5TargetObservationHeadV1,
)
from lewm.models.two_resolution_frontier_value_head_v1 import (
    FrozenCandidateFeatureBatchV1,
    FrontierValueScoresV1,
    TwoResolutionFrontierValueHeadV1,
)


PRODUCTION_QUALIFIED_SHARED_V5_NAVIGATION_RUNTIME_V1 = None
PRODUCTION_SHARED_V5_CHECKPOINT_FILE_SHA256_V1 = None
PRODUCTION_SHARED_V5_MODEL_STATE_SHA256_V1 = None
PRODUCTION_G2_REPORT_SHA256_V1 = None
PRODUCTION_G2_CANDIDATE_PUBLICATION_SHA256_V1 = None
PRODUCTION_PHYSICAL_CALIBRATION_SHA256_V1 = None
PRODUCTION_PHYSICAL_THRESHOLDS_SHA256_V1 = None
PRODUCTION_TARGET_HEAD_BINDING_SHA256_V1 = None
PRODUCTION_G4_HEAD_BINDING_SHA256_V1 = None

FRAME_TICK_BINDING_SCHEMA_V1 = "lewm_go2_qualified_frame_tick_binding_v1"
DETACHED_FEATURE_CACHE_SCHEMA_V1 = "lewm_go2_detached_shared_v5_feature_cache_v1"
QUALIFIED_FRAME_OUTCOME_SCHEMA_V1 = "lewm_go2_qualified_shared_v5_frame_outcome_v1"
TICK_ADMISSION_RECEIPT_SCHEMA_V1 = "lewm_go2_tick_admission_receipt_v1"


class QualifiedSharedV5NavigationRuntimeV1Error(ValueError):
    """Base error for the source-only qualified runtime."""


class QualifiedSharedV5NavigationRuntimeV1BindingError(
    QualifiedSharedV5NavigationRuntimeV1Error
):
    """An exact tick/session/reset/revision/object binding changed."""


class QualifiedSharedV5NavigationRuntimeV1ReplayError(
    QualifiedSharedV5NavigationRuntimeV1Error
):
    """A one-shot frame, head call, receipt, or lease was replayed."""


class QualifiedSharedV5NavigationRuntimeV1TerminalError(
    QualifiedSharedV5NavigationRuntimeV1Error
):
    """The synthetic runtime has already committed a terminal fault."""


def _tensor_sha256(value: torch.Tensor, *, name: str) -> str:
    if type(value) is not torch.Tensor or not value.dtype.is_floating_point:
        raise QualifiedSharedV5NavigationRuntimeV1BindingError(
            f"{name} must be an exact floating tensor"
        )
    if value.requires_grad or value.grad_fn is not None:
        raise QualifiedSharedV5NavigationRuntimeV1BindingError(
            f"{name} must be detached"
        )
    if value.numel() == 0 or not bool(torch.isfinite(value).all().item()):
        raise QualifiedSharedV5NavigationRuntimeV1BindingError(
            f"{name} must be nonempty and finite"
        )
    if value.dtype not in {torch.float32, torch.float64}:
        raise QualifiedSharedV5NavigationRuntimeV1BindingError(
            f"{name} must use canonical float32 or float64"
        )
    canonical = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(canonical.dtype), "shape": list(canonical.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def synthetic_frame_content_sha256_v1(value: torch.Tensor) -> str:
    """Content commitment for a synthetic test frame; no artifact is opened."""

    if type(value) is not torch.Tensor or value.ndim != 4 or value.shape[1] != 3:
        raise QualifiedSharedV5NavigationRuntimeV1BindingError(
            "synthetic frame must be exact [batch,3,height,width] tensor"
        )
    if value.requires_grad or value.grad_fn is not None:
        raise QualifiedSharedV5NavigationRuntimeV1BindingError(
            "synthetic frame must be detached"
        )
    if value.dtype not in {torch.uint8, torch.float32, torch.float64}:
        raise QualifiedSharedV5NavigationRuntimeV1BindingError(
            "synthetic frame dtype is not canonical"
        )
    if value.numel() == 0 or (
        value.dtype.is_floating_point and not bool(torch.isfinite(value).all().item())
    ):
        raise QualifiedSharedV5NavigationRuntimeV1BindingError(
            "synthetic frame must be nonempty and finite"
        )
    canonical = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(canonical.dtype), "shape": list(canonical.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _target_output_sha256(value: FourColorTargetObservationOutputV1) -> str:
    if type(value) is not FourColorTargetObservationOutputV1:
        raise QualifiedSharedV5NavigationRuntimeV1BindingError(
            "target output type changed"
        )
    return canonical_json_sha256(
        {
            "schema": "lewm_go2_four_color_target_output_tensor_binding_v1",
            "version": 1,
            "colors": list(value.colors),
            "presence_logit_sha256": _tensor_sha256(value.presence_logit.detach(), name="presence_logit"),
            "presence_probability_sha256": _tensor_sha256(value.presence_probability.detach(), name="presence_probability"),
            "bearing_mean_rad_sha256": _tensor_sha256(value.bearing_mean_rad.detach(), name="bearing_mean_rad"),
            "bearing_scale_rad_sha256": _tensor_sha256(value.bearing_scale_rad.detach(), name="bearing_scale_rad"),
            "range_mean_m_sha256": _tensor_sha256(value.range_mean_m.detach(), name="range_mean_m"),
            "range_scale_m_sha256": _tensor_sha256(value.range_scale_m.detach(), name="range_scale_m"),
            "uncertainty_sha256": _tensor_sha256(value.uncertainty.detach(), name="uncertainty"),
            "quality_sha256": _tensor_sha256(value.quality.detach(), name="quality"),
        }
    )


@dataclass(frozen=True)
class FrameTickBindingV1:
    tick_index: int
    reset_id: str
    session_id: str
    pre_physical_revision: int
    controller_input_sha256: str
    rgb_content_sha256: str
    timestamp_binary64_hex: str
    synchronization_id: str
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_nonnegative_int(self.tick_index, name="tick_index")
        require_identifier(self.reset_id, name="reset_id")
        require_identifier(self.session_id, name="session_id")
        if self.reset_id == self.session_id:
            raise QualifiedSharedV5NavigationRuntimeV1BindingError(
                "reset and session IDs must differ"
            )
        require_nonnegative_int(self.pre_physical_revision, name="pre_physical_revision")
        require_sha256(self.controller_input_sha256, name="controller_input_sha256")
        require_sha256(self.rgb_content_sha256, name="rgb_content_sha256")
        decode_canonical_binary64_hex(
            self.timestamp_binary64_hex, name="timestamp_binary64_hex"
        )
        require_identifier(self.synchronization_id, name="synchronization_id")
        object.__setattr__(self, "content_sha256", canonical_json_sha256(self._core_dict()))

    def _core_dict(self) -> dict[str, object]:
        return {
            "schema": FRAME_TICK_BINDING_SCHEMA_V1,
            "version": 1,
            "tick_index": self.tick_index,
            "reset_id": self.reset_id,
            "session_id": self.session_id,
            "pre_physical_revision": self.pre_physical_revision,
            "controller_input_sha256": self.controller_input_sha256,
            "rgb_content_sha256": self.rgb_content_sha256,
            "timestamp_binary64_hex": self.timestamp_binary64_hex,
            "synchronization_id": self.synchronization_id,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._core_dict(), "content_sha256": self.content_sha256}


@dataclass(frozen=True)
class DetachedSharedV5FeatureCacheV1:
    patch_features: torch.Tensor = field(repr=False, compare=False)
    bev_features: torch.Tensor = field(repr=False, compare=False)
    patch_features_sha256: str = field(init=False)
    bev_features_sha256: str = field(init=False)
    content_sha256: str = field(init=False)
    _patch_version: int = field(init=False, repr=False, compare=False)
    _bev_version: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.patch_features) is not torch.Tensor or self.patch_features.ndim != 3:
            raise QualifiedSharedV5NavigationRuntimeV1BindingError(
                "patch features must be exact [batch, patches, channels] tensor"
            )
        if type(self.bev_features) is not torch.Tensor or self.bev_features.ndim != 4:
            raise QualifiedSharedV5NavigationRuntimeV1BindingError(
                "BEV features must be exact [batch, channels, height, width] tensor"
            )
        if self.patch_features.shape[0] != self.bev_features.shape[0]:
            raise QualifiedSharedV5NavigationRuntimeV1BindingError(
                "patch and BEV batch sizes differ"
            )
        patch_sha = _tensor_sha256(self.patch_features, name="patch_features")
        bev_sha = _tensor_sha256(self.bev_features, name="bev_features")
        object.__setattr__(self, "patch_features_sha256", patch_sha)
        object.__setattr__(self, "bev_features_sha256", bev_sha)
        object.__setattr__(self, "_patch_version", self.patch_features._version)
        object.__setattr__(self, "_bev_version", self.bev_features._version)
        object.__setattr__(
            self,
            "content_sha256",
            canonical_json_sha256(
                {
                    "schema": DETACHED_FEATURE_CACHE_SCHEMA_V1,
                    "version": 1,
                    "patch_features_sha256": patch_sha,
                    "bev_features_sha256": bev_sha,
                }
            ),
        )

    def assert_unchanged(self) -> None:
        if (
            self.patch_features._version != self._patch_version
            or self.bev_features._version != self._bev_version
        ):
            raise QualifiedSharedV5NavigationRuntimeV1BindingError(
                "detached feature cache was mutated"
            )
        if (
            _tensor_sha256(self.patch_features, name="patch_features")
            != self.patch_features_sha256
            or _tensor_sha256(self.bev_features, name="bev_features")
            != self.bev_features_sha256
        ):
            raise QualifiedSharedV5NavigationRuntimeV1BindingError(
                "detached feature cache content changed"
            )

    def __copy__(self) -> object:
        raise TypeError("detached feature cache cannot be copied")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise TypeError("detached feature cache cannot be deep-copied")

    def __reduce__(self) -> object:
        raise TypeError("detached feature cache cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("detached feature cache cannot be serialized")


@dataclass(frozen=True)
class QualifiedSharedV5FrameOutcomeV1:
    binding: FrameTickBindingV1
    feature_cache: DetachedSharedV5FeatureCacheV1 = field(repr=False, compare=False)
    physical_head_output: torch.Tensor = field(repr=False, compare=False)
    shared_v5_checkpoint_file_sha256: None = None
    shared_v5_model_state_sha256: None = None
    g2_report_sha256: None = None
    g2_candidate_publication_sha256: None = None
    physical_calibration_sha256: None = None
    physical_thresholds_sha256: None = None
    physical_head_output_sha256: str = field(init=False)
    content_sha256: str = field(init=False)
    _physical_version: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.binding) is not FrameTickBindingV1:
            raise QualifiedSharedV5NavigationRuntimeV1BindingError("frame binding type changed")
        if type(self.feature_cache) is not DetachedSharedV5FeatureCacheV1:
            raise QualifiedSharedV5NavigationRuntimeV1BindingError("feature cache type changed")
        self.feature_cache.assert_unchanged()
        if any(
            value is not None
            for value in (
                self.shared_v5_checkpoint_file_sha256,
                self.shared_v5_model_state_sha256,
                self.g2_report_sha256,
                self.g2_candidate_publication_sha256,
                self.physical_calibration_sha256,
                self.physical_thresholds_sha256,
            )
        ):
            raise QualifiedSharedV5NavigationRuntimeV1BindingError(
                "source-only outcome cannot bind unresolved production artifacts"
            )
        physical_sha = _tensor_sha256(
            self.physical_head_output, name="physical_head_output"
        )
        object.__setattr__(self, "physical_head_output_sha256", physical_sha)
        object.__setattr__(self, "_physical_version", self.physical_head_output._version)
        object.__setattr__(
            self,
            "content_sha256",
            canonical_json_sha256(
                {
                    "schema": QUALIFIED_FRAME_OUTCOME_SCHEMA_V1,
                    "version": 1,
                    "binding_sha256": self.binding.content_sha256,
                    "feature_cache_sha256": self.feature_cache.content_sha256,
                    "physical_head_output_sha256": physical_sha,
                    "shared_v5_checkpoint_file_sha256": None,
                    "shared_v5_model_state_sha256": None,
                    "g2_report_sha256": None,
                    "g2_candidate_publication_sha256": None,
                    "physical_calibration_sha256": None,
                    "physical_thresholds_sha256": None,
                }
            ),
        )

    def assert_unchanged(self) -> None:
        self.feature_cache.assert_unchanged()
        if self.physical_head_output._version != self._physical_version:
            raise QualifiedSharedV5NavigationRuntimeV1BindingError(
                "physical head output was mutated"
            )
        if (
            _tensor_sha256(self.physical_head_output, name="physical_head_output")
            != self.physical_head_output_sha256
        ):
            raise QualifiedSharedV5NavigationRuntimeV1BindingError(
                "physical head output content changed"
            )

    def __copy__(self) -> object:
        raise TypeError("qualified frame outcome cannot be copied")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise TypeError("qualified frame outcome cannot be deep-copied")

    def __reduce__(self) -> object:
        raise TypeError("qualified frame outcome cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("qualified frame outcome cannot be serialized")


class FakeSharedV5FrameBackendV1:
    """Deterministic CPU tensor fixture proving the one-call accounting graph."""

    is_synthetic_test_fixture = True

    def __init__(
        self,
        *,
        patch_feature_dim: int,
        bev_feature_dim: int,
        _synthetic_test_fixture: bool = False,
    ) -> None:
        if _synthetic_test_fixture is not True:
            raise PermissionError("fake frame backend is available only to synthetic tests")
        if type(patch_feature_dim) is not int or patch_feature_dim <= 0:
            raise ValueError("patch_feature_dim must be a positive exact integer")
        if type(bev_feature_dim) is not int or bev_feature_dim <= 0:
            raise ValueError("bev_feature_dim must be a positive exact integer")
        self.patch_feature_dim = patch_feature_dim
        self.bev_feature_dim = bev_feature_dim
        self._counts = {
            "shared_v5_forward_frame_call_count": 0,
            "vision_encoder_forward_tokens_call_count": 0,
            "rgb_decode_call_count": 0,
            "rgb_preprocess_call_count": 0,
        }

    def count_snapshot(self) -> dict[str, int]:
        return dict(self._counts)

    def preprocess_synthetic_frame(self, raw_frame: torch.Tensor) -> torch.Tensor:
        if type(raw_frame) is not torch.Tensor or raw_frame.ndim != 4 or raw_frame.shape[1] != 3:
            raise QualifiedSharedV5NavigationRuntimeV1BindingError(
                "synthetic frame must be exact [batch,3,height,width] tensor"
            )
        if raw_frame.requires_grad or raw_frame.grad_fn is not None:
            raise QualifiedSharedV5NavigationRuntimeV1BindingError(
                "synthetic frame must be detached"
            )
        if raw_frame.numel() == 0 or not bool(torch.isfinite(raw_frame.float()).all().item()):
            raise QualifiedSharedV5NavigationRuntimeV1BindingError(
                "synthetic frame must be nonempty and finite"
            )
        self._counts["rgb_decode_call_count"] += 1
        self._counts["rgb_preprocess_call_count"] += 1
        frame = raw_frame.detach().to(dtype=torch.float32)
        if raw_frame.dtype == torch.uint8:
            frame = frame / 255.0
        return frame.contiguous()

    def _fake_forward_tokens(self, frame: torch.Tensor) -> torch.Tensor:
        self._counts["vision_encoder_forward_tokens_call_count"] += 1
        pooled = F.adaptive_avg_pool2d(frame, output_size=(2, 2))
        token_base = pooled.mean(dim=1).flatten(1)
        scales = torch.linspace(
            0.5,
            1.5,
            self.patch_feature_dim,
            dtype=frame.dtype,
            device=frame.device,
        )
        return token_base[:, :, None] * scales[None, None, :]

    def forward_synthetic_frame(
        self, frame: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._counts["shared_v5_forward_frame_call_count"] += 1
        tokens = self._fake_forward_tokens(frame)
        base = F.adaptive_avg_pool2d(frame.mean(dim=1, keepdim=True), output_size=(2, 2))
        channels = torch.linspace(
            0.25,
            1.25,
            self.bev_feature_dim,
            dtype=frame.dtype,
            device=frame.device,
        )
        bev = base * channels[None, :, None, None]
        physical = torch.cat((base, 1.0 - base, torch.full_like(base, 0.5)), dim=1)
        return tokens.detach(), bev.detach(), physical.detach()


@dataclass(frozen=True)
class PhysicalViewBindingPayloadV1:
    frame_outcome_sha256: str
    post_physical_revision: int
    post_physical_content_sha256: str
    post_configuration_revision: int
    configuration_snapshot_sha256: str
    configuration_component_sha256: str
    frontier_sha256: str
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.frame_outcome_sha256, name="frame_outcome_sha256")
        require_nonnegative_int(self.post_physical_revision, name="post_physical_revision")
        require_nonnegative_int(
            self.post_configuration_revision, name="post_configuration_revision"
        )
        for name in (
            "post_physical_content_sha256",
            "configuration_snapshot_sha256",
            "configuration_component_sha256",
            "frontier_sha256",
        ):
            require_sha256(getattr(self, name), name=name)
        object.__setattr__(
            self,
            "content_sha256",
            canonical_json_sha256(
                {
                    "schema": "lewm_go2_physical_view_binding_payload_v1",
                    "version": 1,
                    "frame_outcome_sha256": self.frame_outcome_sha256,
                    "post_physical_revision": self.post_physical_revision,
                    "post_physical_content_sha256": self.post_physical_content_sha256,
                    "post_configuration_revision": self.post_configuration_revision,
                    "configuration_snapshot_sha256": self.configuration_snapshot_sha256,
                    "configuration_component_sha256": self.configuration_component_sha256,
                    "frontier_sha256": self.frontier_sha256,
                }
            ),
        )


@dataclass(frozen=True)
class TargetColorObservationPayloadV1:
    color: str
    color_index: int
    four_color_output_sha256: str
    presence_logit: torch.Tensor = field(repr=False, compare=False)
    presence_probability: torch.Tensor = field(repr=False, compare=False)
    bearing_mean_rad: torch.Tensor = field(repr=False, compare=False)
    bearing_scale_rad: torch.Tensor = field(repr=False, compare=False)
    range_mean_m: torch.Tensor = field(repr=False, compare=False)
    range_scale_m: torch.Tensor = field(repr=False, compare=False)
    uncertainty: torch.Tensor = field(repr=False, compare=False)
    quality: torch.Tensor = field(repr=False, compare=False)
    content_sha256: str = field(init=False)
    _versions: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.color_index) is not int or not 0 <= self.color_index < 4:
            raise QualifiedSharedV5NavigationRuntimeV1BindingError("target color index is invalid")
        if type(self.color) is not str or self.color != CANONICAL_COLORS[self.color_index]:
            raise QualifiedSharedV5NavigationRuntimeV1BindingError("target color order changed")
        require_sha256(self.four_color_output_sha256, name="four_color_output_sha256")
        tensor_names = (
            "presence_logit",
            "presence_probability",
            "bearing_mean_rad",
            "bearing_scale_rad",
            "range_mean_m",
            "range_scale_m",
            "uncertainty",
            "quality",
        )
        tensor_hashes = {}
        versions = []
        shape = None
        for name in tensor_names:
            value = getattr(self, name)
            if type(value) is not torch.Tensor or value.ndim != 1:
                raise QualifiedSharedV5NavigationRuntimeV1BindingError(
                    f"{name} must be one exact selected-colour batch vector"
                )
            current_shape = tuple(value.shape)
            if shape is None:
                shape = current_shape
            elif current_shape != shape:
                raise QualifiedSharedV5NavigationRuntimeV1BindingError(
                    "selected-colour tensor shapes differ"
                )
            tensor_hashes[f"{name}_sha256"] = _tensor_sha256(value, name=name)
            versions.append(value._version)
        object.__setattr__(self, "_versions", tuple(versions))
        object.__setattr__(
            self,
            "content_sha256",
            canonical_json_sha256(
                {
                    "schema": "lewm_go2_target_color_observation_payload_v1",
                    "version": 1,
                    "color": self.color,
                    "color_index": self.color_index,
                    "four_color_output_sha256": self.four_color_output_sha256,
                    **tensor_hashes,
                }
            ),
        )

    def assert_unchanged(self) -> None:
        tensor_names = (
            "presence_logit",
            "presence_probability",
            "bearing_mean_rad",
            "bearing_scale_rad",
            "range_mean_m",
            "range_scale_m",
            "uncertainty",
            "quality",
        )
        if tuple(getattr(self, name)._version for name in tensor_names) != self._versions:
            raise QualifiedSharedV5NavigationRuntimeV1BindingError(
                "selected-colour target payload was mutated"
            )
        hashes = {
            f"{name}_sha256": _tensor_sha256(getattr(self, name), name=name)
            for name in tensor_names
        }
        expected = canonical_json_sha256(
            {
                "schema": "lewm_go2_target_color_observation_payload_v1",
                "version": 1,
                "color": self.color,
                "color_index": self.color_index,
                "four_color_output_sha256": self.four_color_output_sha256,
                **hashes,
            }
        )
        if expected != self.content_sha256:
            raise QualifiedSharedV5NavigationRuntimeV1BindingError(
                "selected-colour target payload content changed"
            )


class ExactObjectLeaseV1:
    """Non-copyable, non-serializable, exact-consumer single-use lease."""

    def __init__(
        self,
        *,
        receipt: "TickAdmissionReceiptV1",
        outcome: QualifiedSharedV5FrameOutcomeV1,
        kind: str,
        consumer: object,
        payload: object,
    ) -> None:
        if type(receipt) is not TickAdmissionReceiptV1:
            raise TypeError("receipt must be exact TickAdmissionReceiptV1")
        if type(outcome) is not QualifiedSharedV5FrameOutcomeV1:
            raise TypeError("outcome must be exact QualifiedSharedV5FrameOutcomeV1")
        if type(kind) is not str or kind not in {
            "physical_view",
            "target_red",
            "target_yellow",
            "target_blue",
            "target_green",
            "g4_cached_features",
        }:
            raise ValueError("lease kind is invalid")
        if consumer is None:
            raise ValueError("lease consumer must be an exact non-null object")
        self._receipt = receipt
        self._outcome = outcome
        self._kind = kind
        self._consumer = consumer
        self._payload = payload
        self._consumed = False

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def consumed(self) -> bool:
        return self._consumed

    def consume(
        self,
        *,
        receipt: "TickAdmissionReceiptV1",
        outcome: QualifiedSharedV5FrameOutcomeV1,
        consumer: object,
    ) -> object:
        if receipt is not self._receipt or outcome is not self._outcome or consumer is not self._consumer:
            self._receipt._runtime._seal_terminal()
            raise QualifiedSharedV5NavigationRuntimeV1BindingError(
                "lease receipt/outcome/consumer exact-object binding changed"
            )
        try:
            self._receipt._assert_live()
            self._outcome.assert_unchanged()
        except Exception:
            self._receipt._runtime._seal_terminal()
            raise
        if self._consumed:
            self._receipt._runtime._seal_terminal()
            raise QualifiedSharedV5NavigationRuntimeV1ReplayError("lease was replayed")
        self._consumed = True
        return self._payload

    def __copy__(self) -> object:
        raise TypeError("lease cannot be copied")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise TypeError("lease cannot be deep-copied")

    def __reduce__(self) -> object:
        raise TypeError("lease cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("lease cannot be serialized")


class TickAdmissionReceiptV1:
    """Exact admitted tick authority and sole lease mint."""

    def __init__(
        self,
        *,
        runtime: "QualifiedSharedV5NavigationRuntimeV1",
        outcome: QualifiedSharedV5FrameOutcomeV1,
        target_output: FourColorTargetObservationOutputV1,
        pre_physical_revision: int,
        post_physical_revision: int,
        physical_transaction_sha256: str,
        physical_retraction_sha256: str,
        post_physical_content_sha256: str,
        pre_configuration_revision: int,
        post_configuration_revision: int,
        configuration_snapshot_sha256: str,
        configuration_component_sha256: str,
        frontier_sha256: str,
    ) -> None:
        if type(runtime) is not QualifiedSharedV5NavigationRuntimeV1:
            raise TypeError("runtime must be exact QualifiedSharedV5NavigationRuntimeV1")
        if type(outcome) is not QualifiedSharedV5FrameOutcomeV1:
            raise TypeError("outcome must be exact QualifiedSharedV5FrameOutcomeV1")
        if type(target_output) is not FourColorTargetObservationOutputV1:
            raise TypeError("target_output must be exact FourColorTargetObservationOutputV1")
        self._runtime = runtime
        self._outcome = outcome
        self._target_output = target_output
        self._live = True
        self._issued_kinds: set[str] = set()
        self._target_output_sha256 = _target_output_sha256(target_output)
        self.pre_physical_revision = require_nonnegative_int(
            pre_physical_revision, name="pre_physical_revision"
        )
        self.post_physical_revision = require_nonnegative_int(
            post_physical_revision, name="post_physical_revision"
        )
        self.pre_configuration_revision = require_nonnegative_int(
            pre_configuration_revision, name="pre_configuration_revision"
        )
        self.post_configuration_revision = require_nonnegative_int(
            post_configuration_revision, name="post_configuration_revision"
        )
        if self.post_physical_revision != self.pre_physical_revision + 1:
            raise QualifiedSharedV5NavigationRuntimeV1BindingError(
                "physical revision must advance exactly once"
            )
        if self.post_configuration_revision != self.pre_configuration_revision + 1:
            raise QualifiedSharedV5NavigationRuntimeV1BindingError(
                "configuration revision must advance exactly once"
            )
        for name, value in (
            ("physical_transaction_sha256", physical_transaction_sha256),
            ("physical_retraction_sha256", physical_retraction_sha256),
            ("post_physical_content_sha256", post_physical_content_sha256),
            ("configuration_snapshot_sha256", configuration_snapshot_sha256),
            ("configuration_component_sha256", configuration_component_sha256),
            ("frontier_sha256", frontier_sha256),
        ):
            require_sha256(value, name=name)
            setattr(self, name, value)
        self._physical_view_payload = PhysicalViewBindingPayloadV1(
            frame_outcome_sha256=outcome.content_sha256,
            post_physical_revision=self.post_physical_revision,
            post_physical_content_sha256=post_physical_content_sha256,
            post_configuration_revision=self.post_configuration_revision,
            configuration_snapshot_sha256=configuration_snapshot_sha256,
            configuration_component_sha256=configuration_component_sha256,
            frontier_sha256=frontier_sha256,
        )
        target_tensor_names = (
            "presence_logit",
            "presence_probability",
            "bearing_mean_rad",
            "bearing_scale_rad",
            "range_mean_m",
            "range_scale_m",
            "uncertainty",
            "quality",
        )
        self._target_payloads = tuple(
            TargetColorObservationPayloadV1(
                color=color,
                color_index=index,
                four_color_output_sha256=self._target_output_sha256,
                **{
                    name: getattr(target_output, name)[:, index].detach()
                    for name in target_tensor_names
                },
            )
            for index, color in enumerate(CANONICAL_COLORS)
        )
        self.content_sha256 = canonical_json_sha256(
            {
                "schema": TICK_ADMISSION_RECEIPT_SCHEMA_V1,
                "version": 1,
                "frame_outcome_sha256": outcome.content_sha256,
                "target_output_sha256": self._target_output_sha256,
                "tick_index": outcome.binding.tick_index,
                "reset_id": outcome.binding.reset_id,
                "session_id": outcome.binding.session_id,
                "pre_physical_revision": self.pre_physical_revision,
                "post_physical_revision": self.post_physical_revision,
                "physical_transaction_sha256": physical_transaction_sha256,
                "physical_retraction_sha256": physical_retraction_sha256,
                "post_physical_content_sha256": post_physical_content_sha256,
                "pre_configuration_revision": self.pre_configuration_revision,
                "post_configuration_revision": self.post_configuration_revision,
                "configuration_snapshot_sha256": configuration_snapshot_sha256,
                "configuration_component_sha256": configuration_component_sha256,
                "frontier_sha256": frontier_sha256,
            }
        )

    @property
    def outcome(self) -> QualifiedSharedV5FrameOutcomeV1:
        return self._outcome

    def _assert_live(self) -> None:
        if not self._live or not self._runtime._is_exact_active_receipt(self):
            self._runtime._seal_terminal()
            raise QualifiedSharedV5NavigationRuntimeV1ReplayError(
                "tick receipt is expired, foreign, or inactive"
            )
        try:
            self._outcome.assert_unchanged()
            if _target_output_sha256(self._target_output) != self._target_output_sha256:
                raise QualifiedSharedV5NavigationRuntimeV1BindingError(
                    "four-colour target output content changed"
                )
            for payload in self._target_payloads:
                payload.assert_unchanged()
        except Exception:
            self._runtime._seal_terminal()
            raise

    def _mint(
        self,
        *,
        kind: str,
        consumer: object,
        payload: object,
    ) -> ExactObjectLeaseV1:
        self._assert_live()
        if kind in self._issued_kinds:
            self._runtime._seal_terminal()
            raise QualifiedSharedV5NavigationRuntimeV1ReplayError(
                f"{kind} lease was already issued"
            )
        self._issued_kinds.add(kind)
        return ExactObjectLeaseV1(
            receipt=self,
            outcome=self._outcome,
            kind=kind,
            consumer=consumer,
            payload=payload,
        )

    def mint_physical_view_lease(self, *, consumer: object) -> ExactObjectLeaseV1:
        return self._mint(
            kind="physical_view",
            consumer=consumer,
            payload=self._physical_view_payload,
        )

    def mint_target_evidence_leases(
        self,
        *,
        consumers: tuple[object, object, object, object],
    ) -> tuple[ExactObjectLeaseV1, ExactObjectLeaseV1, ExactObjectLeaseV1, ExactObjectLeaseV1]:
        if type(consumers) is not tuple or len(consumers) != 4 or any(item is None for item in consumers):
            self._runtime._seal_terminal()
            raise QualifiedSharedV5NavigationRuntimeV1BindingError(
                "target consumers must be an exact non-null four-tuple"
            )
        leases = []
        for index, color in enumerate(CANONICAL_COLORS):
            leases.append(
                self._mint(
                    kind=f"target_{color}",
                    consumer=consumers[index],
                    payload=self._target_payloads[index],
                )
            )
        return tuple(leases)  # type: ignore[return-value]

    def mint_g4_cached_feature_lease(self, *, consumer: object) -> ExactObjectLeaseV1:
        return self._mint(
            kind="g4_cached_features",
            consumer=consumer,
            payload=self._outcome.feature_cache,
        )

    def _expire(self) -> None:
        self._live = False

    def __copy__(self) -> object:
        raise TypeError("tick receipt cannot be copied")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise TypeError("tick receipt cannot be deep-copied")

    def __reduce__(self) -> object:
        raise TypeError("tick receipt cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("tick receipt cannot be serialized")


class QualifiedSharedV5NavigationRuntimeV1:
    """Synthetic-only state machine proving one inference and lease accounting."""

    def __init__(
        self,
        *,
        backend: FakeSharedV5FrameBackendV1,
        reset_id: str,
        session_id: str,
        initial_physical_revision: int = 0,
        initial_configuration_revision: int = 0,
        _synthetic_mock: bool = False,
    ) -> None:
        if _synthetic_mock is not True:
            raise PermissionError("source-only runtime permits synthetic mock construction only")
        if type(backend) is not FakeSharedV5FrameBackendV1:
            raise PermissionError("source-only runtime accepts only the exact fake frame backend")
        require_identifier(reset_id, name="reset_id")
        require_identifier(session_id, name="session_id")
        if reset_id == session_id:
            raise QualifiedSharedV5NavigationRuntimeV1BindingError(
                "reset and session IDs must differ"
            )
        self._backend = backend
        self._reset_id = reset_id
        self._session_id = session_id
        self._physical_revision = require_nonnegative_int(
            initial_physical_revision, name="initial_physical_revision"
        )
        self._configuration_revision = require_nonnegative_int(
            initial_configuration_revision, name="initial_configuration_revision"
        )
        self._next_tick_index = 0
        self._active_outcome: QualifiedSharedV5FrameOutcomeV1 | None = None
        self._active_target_output: FourColorTargetObservationOutputV1 | None = None
        self._active_receipt: TickAdmissionReceiptV1 | None = None
        self._target_called = False
        self._g4_called = False
        self._terminal_fault = False
        self._counts = CallCounterPanelV1.zero()
        self.shared_v5_checkpoint_file_sha256 = None
        self.shared_v5_model_state_sha256 = None
        self.g2_report_sha256 = None
        self.g2_candidate_publication_sha256 = None
        self.physical_calibration_sha256 = None
        self.physical_thresholds_sha256 = None

    @property
    def call_counters(self) -> CallCounterPanelV1:
        return self._counts

    def _require_operational(self) -> None:
        if self._terminal_fault:
            raise QualifiedSharedV5NavigationRuntimeV1TerminalError(
                "runtime is sealed after terminal fault"
            )

    def _seal_terminal(self) -> None:
        if self._active_receipt is not None:
            self._active_receipt._expire()
        self._active_receipt = None
        self._active_outcome = None
        self._active_target_output = None
        self._terminal_fault = True

    def _raise_terminal(self, error: Exception) -> None:
        self._seal_terminal()
        raise error

    def run_shared_frame_once(
        self,
        *,
        binding: FrameTickBindingV1,
        synthetic_frame: torch.Tensor,
    ) -> QualifiedSharedV5FrameOutcomeV1:
        self._require_operational()
        if self._active_outcome is not None:
            self._raise_terminal(QualifiedSharedV5NavigationRuntimeV1ReplayError(
                "a frame is already active; recomputation is forbidden"
            ))
        if type(binding) is not FrameTickBindingV1:
            self._raise_terminal(TypeError("binding must be exact FrameTickBindingV1"))
        if (
            binding.tick_index != self._next_tick_index
            or binding.reset_id != self._reset_id
            or binding.session_id != self._session_id
            or binding.pre_physical_revision != self._physical_revision
        ):
            self._raise_terminal(QualifiedSharedV5NavigationRuntimeV1BindingError(
                "frame tick/reset/session/pre-revision binding changed"
            ))
        if synthetic_frame_content_sha256_v1(synthetic_frame) != binding.rgb_content_sha256:
            self._raise_terminal(QualifiedSharedV5NavigationRuntimeV1BindingError(
                "synthetic frame content differs from tick commitment"
            ))
        before = self._backend.count_snapshot()
        try:
            frame = self._backend.preprocess_synthetic_frame(synthetic_frame)
            patch, bev, physical = self._backend.forward_synthetic_frame(frame)
        except Exception as exc:
            self._seal_terminal()
            raise QualifiedSharedV5NavigationRuntimeV1TerminalError(
                "frame backend failed; retry is forbidden"
            ) from exc
        after = self._backend.count_snapshot()
        required_delta = {
            "shared_v5_forward_frame_call_count": 1,
            "vision_encoder_forward_tokens_call_count": 1,
            "rgb_decode_call_count": 1,
            "rgb_preprocess_call_count": 1,
        }
        if any(after[name] - before[name] != delta for name, delta in required_delta.items()):
            self._raise_terminal(QualifiedSharedV5NavigationRuntimeV1TerminalError(
                "fake backend violated exact one-inference call graph"
            ))
        cache = DetachedSharedV5FeatureCacheV1(patch, bev)
        outcome = QualifiedSharedV5FrameOutcomeV1(
            binding=binding,
            feature_cache=cache,
            physical_head_output=physical,
        )
        delta_panel = CallCounterPanelV1(
            observation_tick_count=1,
            shared_frame_outcome_count=1,
            shared_v5_forward_frame_call_count=1,
            vision_encoder_forward_tokens_call_count=1,
            target_four_color_batch_count=0,
            g4_value_head_call_count=0,
            rgb_decode_call_count=1,
            rgb_preprocess_call_count=1,
            extra_rgb_decode_or_preprocess_count=0,
        )
        self._counts = self._counts.plus(delta_panel)
        self._active_outcome = outcome
        self._active_target_output = None
        self._target_called = False
        self._g4_called = False
        return outcome

    def run_target_four_color_batch_once(
        self,
        *,
        outcome: QualifiedSharedV5FrameOutcomeV1,
        head: SharedV5TargetObservationHeadV1,
    ) -> FourColorTargetObservationOutputV1:
        self._require_operational()
        if outcome is not self._active_outcome:
            self._raise_terminal(QualifiedSharedV5NavigationRuntimeV1BindingError(
                "target head received a reconstructed or foreign frame outcome"
            ))
        if self._target_called:
            self._raise_terminal(QualifiedSharedV5NavigationRuntimeV1ReplayError(
                "target four-colour batch was already called"
            ))
        if type(head) is not SharedV5TargetObservationHeadV1:
            self._raise_terminal(TypeError("head must be exact SharedV5TargetObservationHeadV1"))
        try:
            outcome.assert_unchanged()
            with torch.no_grad():
                result = head(
                    outcome.feature_cache.patch_features,
                    outcome.feature_cache.bev_features,
                )
        except Exception as exc:
            self._seal_terminal()
            raise QualifiedSharedV5NavigationRuntimeV1TerminalError(
                "target head failed; retry is forbidden"
            ) from exc
        if type(result) is not FourColorTargetObservationOutputV1 or result.colors != CANONICAL_COLORS:
            self._raise_terminal(QualifiedSharedV5NavigationRuntimeV1TerminalError(
                "target head did not emit one canonical four-colour batch"
            ))
        self._target_called = True
        self._active_target_output = result
        self._counts = self._counts.plus(
            CallCounterPanelV1(0, 0, 0, 0, 1, 0, 0, 0, 0)
        )
        return result

    def admit_tick(
        self,
        *,
        outcome: QualifiedSharedV5FrameOutcomeV1,
        post_physical_revision: int,
        physical_transaction_sha256: str,
        physical_retraction_sha256: str,
        post_physical_content_sha256: str,
        post_configuration_revision: int,
        configuration_snapshot_sha256: str,
        configuration_component_sha256: str,
        frontier_sha256: str,
    ) -> TickAdmissionReceiptV1:
        self._require_operational()
        if outcome is not self._active_outcome or self._active_receipt is not None:
            self._raise_terminal(QualifiedSharedV5NavigationRuntimeV1BindingError(
                "tick admission outcome is foreign, reconstructed, or already admitted"
            ))
        if not self._target_called or self._active_target_output is None:
            self._raise_terminal(QualifiedSharedV5NavigationRuntimeV1BindingError(
                "tick admission requires one four-colour target batch"
            ))
        try:
            outcome.assert_unchanged()
            receipt = TickAdmissionReceiptV1(
                runtime=self,
                outcome=outcome,
                target_output=self._active_target_output,
                pre_physical_revision=self._physical_revision,
                post_physical_revision=post_physical_revision,
                physical_transaction_sha256=physical_transaction_sha256,
                physical_retraction_sha256=physical_retraction_sha256,
                post_physical_content_sha256=post_physical_content_sha256,
                pre_configuration_revision=self._configuration_revision,
                post_configuration_revision=post_configuration_revision,
                configuration_snapshot_sha256=configuration_snapshot_sha256,
                configuration_component_sha256=configuration_component_sha256,
                frontier_sha256=frontier_sha256,
            )
        except Exception:
            self._seal_terminal()
            raise
        self._active_receipt = receipt
        return receipt

    def run_g4_value_head_once(
        self,
        *,
        receipt: TickAdmissionReceiptV1,
        feature_lease: ExactObjectLeaseV1,
        head: TwoResolutionFrontierValueHeadV1,
        candidate_batch: FrozenCandidateFeatureBatchV1,
    ) -> FrontierValueScoresV1:
        self._require_operational()
        if receipt is not self._active_receipt:
            self._raise_terminal(QualifiedSharedV5NavigationRuntimeV1BindingError("G4 receipt is not exact active object"))
        if self._g4_called:
            self._raise_terminal(QualifiedSharedV5NavigationRuntimeV1ReplayError("G4 head was already called"))
        if type(feature_lease) is not ExactObjectLeaseV1 or feature_lease.kind != "g4_cached_features":
            self._raise_terminal(QualifiedSharedV5NavigationRuntimeV1BindingError("G4 feature lease is invalid"))
        if type(head) is not TwoResolutionFrontierValueHeadV1:
            self._raise_terminal(TypeError("head must be exact TwoResolutionFrontierValueHeadV1"))
        try:
            payload = feature_lease.consume(
                receipt=receipt,
                outcome=receipt.outcome,
                consumer=head,
            )
        except Exception:
            self._seal_terminal()
            raise
        if payload is not receipt.outcome.feature_cache:
            self._raise_terminal(QualifiedSharedV5NavigationRuntimeV1BindingError("G4 cache object identity changed"))
        try:
            with torch.no_grad():
                result = head(
                    payload.patch_features,
                    payload.bev_features,
                    candidate_batch,
                )
        except Exception as exc:
            self._seal_terminal()
            raise QualifiedSharedV5NavigationRuntimeV1TerminalError(
                "G4 head failed; fallback or retry is forbidden"
            ) from exc
        self._g4_called = True
        self._counts = self._counts.plus(
            CallCounterPanelV1(0, 0, 0, 0, 0, 1, 0, 0, 0)
        )
        return result

    def _is_exact_active_receipt(self, receipt: TickAdmissionReceiptV1) -> bool:
        return receipt is self._active_receipt and self._active_outcome is receipt.outcome

    def commit_tick(self, *, receipt: TickAdmissionReceiptV1) -> CallCounterPanelV1:
        self._require_operational()
        if receipt is not self._active_receipt:
            self._raise_terminal(QualifiedSharedV5NavigationRuntimeV1BindingError(
                "commit receipt is not the exact active object"
            ))
        if self._active_outcome is None or not self._target_called:
            self._raise_terminal(QualifiedSharedV5NavigationRuntimeV1BindingError("active tick is incomplete"))
        try:
            self._active_outcome.assert_unchanged()
            self._counts.assert_one_encode_invariants()
        except Exception:
            self._seal_terminal()
            raise
        self._physical_revision = receipt.post_physical_revision
        self._configuration_revision = receipt.post_configuration_revision
        self._next_tick_index += 1
        receipt._expire()
        self._active_receipt = None
        self._active_outcome = None
        self._active_target_output = None
        self._target_called = False
        self._g4_called = False
        return self._counts

    def fault_tick(self, *, receipt: TickAdmissionReceiptV1 | None = None) -> None:
        if receipt is not None and receipt is not self._active_receipt:
            self._raise_terminal(QualifiedSharedV5NavigationRuntimeV1BindingError(
                "fault receipt is not the exact active object"
            ))
        self._seal_terminal()


__all__ = [
    "DetachedSharedV5FeatureCacheV1",
    "ExactObjectLeaseV1",
    "FakeSharedV5FrameBackendV1",
    "FrameTickBindingV1",
    "PhysicalViewBindingPayloadV1",
    "QualifiedSharedV5FrameOutcomeV1",
    "QualifiedSharedV5NavigationRuntimeV1",
    "QualifiedSharedV5NavigationRuntimeV1BindingError",
    "QualifiedSharedV5NavigationRuntimeV1Error",
    "QualifiedSharedV5NavigationRuntimeV1ReplayError",
    "QualifiedSharedV5NavigationRuntimeV1TerminalError",
    "TargetColorObservationPayloadV1",
    "TickAdmissionReceiptV1",
    "synthetic_frame_content_sha256_v1",
]
