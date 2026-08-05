"""Issuer-owned synthetic Foundation V2 runtime and registered leases.

This module is source/mock only.  It has no production backend, artifact path,
checkpoint loader, dataset, simulator, scene, evaluator, or dynamic import
surface.  Live authority is exact-object membership in issuer/receipt
registries; evidence hashes and reconstructed objects never confer authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import secrets

import torch
import torch.nn.functional as F

from lewm.benchmarks.go2_navigation_development_trace_v2 import (
    CANONICAL_COLORS_V2,
    OWNER_NAMES_V2,
    CallCounterPanelV2,
    OwnerStateBundleV2,
    OwnerStateV2,
    ResetReceiptV2,
    advance_owner_bundle_v2,
    canonical_binary64_hex_v2,
    canonical_json_sha256_v2,
    decode_canonical_binary64_hex_v2,
    initial_owner_content_sha256_v2,
    require_identifier_v2,
    require_nonnegative_int_v2,
    require_sha256_v2,
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


PRODUCTION_QUALIFIED_SHARED_V5_NAVIGATION_RUNTIME_V2 = None
PRODUCTION_FOUNDATION_AUTHORITY_ISSUER_V2 = None
PRODUCTION_SHARED_V5_CHECKPOINT_FILE_SHA256_V2 = None
PRODUCTION_SHARED_V5_MODEL_STATE_SHA256_V2 = None
PRODUCTION_G2_REPORT_SHA256_V2 = None
PRODUCTION_G2_CANDIDATE_PUBLICATION_SHA256_V2 = None
PRODUCTION_PHYSICAL_CALIBRATION_SHA256_V2 = None
PRODUCTION_PHYSICAL_THRESHOLDS_SHA256_V2 = None
PRODUCTION_TARGET_HEAD_BINDING_SHA256_V2 = None
PRODUCTION_G4_HEAD_BINDING_SHA256_V2 = None

_ISSUER_CONSTRUCTION_TOKEN = object()
_BACKEND_CONSTRUCTION_TOKEN = object()
_RESET_AUTHORITY_CONSTRUCTION_TOKEN = object()
_PRODUCER_CONSTRUCTION_TOKEN = object()
_FRAME_CONSTRUCTION_TOKEN = object()
_TARGET_CONSTRUCTION_TOKEN = object()
_PHYSICAL_RECEIPT_CONSTRUCTION_TOKEN = object()
_ADMISSION_CONSTRUCTION_TOKEN = object()
_LEASE_CONSTRUCTION_TOKEN = object()
_CANDIDATE_CONSTRUCTION_TOKEN = object()
_G4_RECEIPT_CONSTRUCTION_TOKEN = object()

_TARGET_TENSOR_NAMES = (
    "presence_logit",
    "presence_probability",
    "bearing_mean_rad",
    "bearing_scale_rad",
    "range_mean_m",
    "range_scale_m",
    "uncertainty",
    "quality",
)


class QualifiedSharedV5NavigationRuntimeV2Error(ValueError):
    """Base Foundation V2 runtime error."""


class QualifiedSharedV5NavigationRuntimeV2BindingError(
    QualifiedSharedV5NavigationRuntimeV2Error
):
    """An exact issuer/object/reset/session/tick/revision binding changed."""


class QualifiedSharedV5NavigationRuntimeV2ReplayError(
    QualifiedSharedV5NavigationRuntimeV2Error
):
    """A single-use authority was replayed or issued twice."""


class QualifiedSharedV5NavigationRuntimeV2TerminalError(
    QualifiedSharedV5NavigationRuntimeV2Error
):
    """The runtime is sealed after a terminal failure."""


def _noncopyable_reduce_error_v2(name: str) -> TypeError:
    return TypeError(f"{name} cannot be copied or serialized")


def _tensor_bytes_sha256_v2(value: torch.Tensor, *, name: str) -> str:
    if type(value) is not torch.Tensor:
        raise QualifiedSharedV5NavigationRuntimeV2BindingError(
            f"{name} must be exact torch.Tensor"
        )
    if value.dtype not in {torch.float32, torch.float64}:
        raise QualifiedSharedV5NavigationRuntimeV2BindingError(
            f"{name} must use float32 or float64"
        )
    if value.requires_grad or value.grad_fn is not None:
        raise QualifiedSharedV5NavigationRuntimeV2BindingError(f"{name} must be detached")
    if value.numel() == 0 or not bool(torch.isfinite(value).all().item()):
        raise QualifiedSharedV5NavigationRuntimeV2BindingError(
            f"{name} must be nonempty and finite"
        )
    canonical = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "dtype": str(canonical.dtype),
                "shape": list(canonical.shape),
                "stride": list(canonical.stride()),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def synthetic_frame_content_sha256_v2(value: torch.Tensor) -> str:
    if type(value) is not torch.Tensor or value.ndim != 4 or value.shape[1] != 3:
        raise QualifiedSharedV5NavigationRuntimeV2BindingError(
            "synthetic frame must be exact [batch,3,height,width] tensor"
        )
    if value.requires_grad or value.grad_fn is not None:
        raise QualifiedSharedV5NavigationRuntimeV2BindingError(
            "synthetic frame must be detached"
        )
    if value.dtype not in {torch.uint8, torch.float32, torch.float64}:
        raise QualifiedSharedV5NavigationRuntimeV2BindingError(
            "synthetic frame dtype is not canonical"
        )
    if value.numel() == 0 or (
        value.dtype.is_floating_point and not bool(torch.isfinite(value).all().item())
    ):
        raise QualifiedSharedV5NavigationRuntimeV2BindingError(
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


@dataclass(frozen=True)
class _TensorObservationV2:
    tensor: torch.Tensor = field(repr=False, compare=False)
    exact_object_id: int
    storage_data_ptr: int
    storage_nbytes: int
    storage_offset: int
    version: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str
    content_sha256: str

    @classmethod
    def capture(cls, tensor: torch.Tensor, *, name: str) -> "_TensorObservationV2":
        content = _tensor_bytes_sha256_v2(tensor, name=name)
        storage = tensor.untyped_storage()
        return cls(
            tensor=tensor,
            exact_object_id=id(tensor),
            storage_data_ptr=int(storage.data_ptr()),
            storage_nbytes=int(storage.nbytes()),
            storage_offset=int(tensor.storage_offset()),
            version=int(tensor._version),
            shape=tuple(int(item) for item in tensor.shape),
            stride=tuple(int(item) for item in tensor.stride()),
            dtype=str(tensor.dtype),
            device=str(tensor.device),
            content_sha256=content,
        )

    def assert_unchanged(self, current: object, *, name: str) -> None:
        if current is not self.tensor or type(current) is not torch.Tensor:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                f"{name} tensor identity changed"
            )
        storage = current.untyped_storage()
        observed = (
            id(current),
            int(storage.data_ptr()),
            int(storage.nbytes()),
            int(current.storage_offset()),
            int(current._version),
            tuple(int(item) for item in current.shape),
            tuple(int(item) for item in current.stride()),
            str(current.dtype),
            str(current.device),
            _tensor_bytes_sha256_v2(current, name=name),
        )
        expected = (
            self.exact_object_id,
            self.storage_data_ptr,
            self.storage_nbytes,
            self.storage_offset,
            self.version,
            self.shape,
            self.stride,
            self.dtype,
            self.device,
            self.content_sha256,
        )
        if observed != expected:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                f"{name} identity/version/storage/content changed"
            )


class FakeSharedV5FrameBackendV2:
    """Issuer-created deterministic CPU fixture; never a production backend."""

    __slots__ = (
        "_issuer",
        "_patch_feature_dim",
        "_bev_feature_dim",
        "_counts",
        "_consumed",
    )

    is_synthetic_test_fixture = True

    def __init__(
        self,
        token: object,
        *,
        issuer: "FoundationAuthorityIssuerV2",
        patch_feature_dim: int,
        bev_feature_dim: int,
    ) -> None:
        if token is not _BACKEND_CONSTRUCTION_TOKEN:
            raise PermissionError("V2 fake backend is issuer-created only")
        if type(patch_feature_dim) is not int or patch_feature_dim <= 0:
            raise ValueError("patch_feature_dim must be positive exact integer")
        if type(bev_feature_dim) is not int or bev_feature_dim <= 0:
            raise ValueError("bev_feature_dim must be positive exact integer")
        self._issuer = issuer
        self._patch_feature_dim = patch_feature_dim
        self._bev_feature_dim = bev_feature_dim
        self._counts = {
            "shared_v5_forward_frame_call_count": 0,
            "vision_encoder_forward_tokens_call_count": 0,
            "rgb_decode_call_count": 0,
            "rgb_preprocess_call_count": 0,
        }
        self._consumed = False

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("FakeSharedV5FrameBackendV2 cannot be subclassed")

    def _count_snapshot(self) -> dict[str, int]:
        return dict(self._counts)

    def _preprocess_once(self, raw: torch.Tensor) -> torch.Tensor:
        synthetic_frame_content_sha256_v2(raw)
        self._counts["rgb_decode_call_count"] += 1
        self._counts["rgb_preprocess_call_count"] += 1
        frame = raw.detach().to(dtype=torch.float32)
        if raw.dtype == torch.uint8:
            frame = frame / 255.0
        return frame.contiguous()

    def _forward_tokens_once(self, frame: torch.Tensor) -> torch.Tensor:
        self._counts["vision_encoder_forward_tokens_call_count"] += 1
        pooled = F.adaptive_avg_pool2d(frame, output_size=(2, 2)).mean(dim=1).flatten(1)
        scale = torch.linspace(
            0.5,
            1.5,
            self._patch_feature_dim,
            dtype=frame.dtype,
            device=frame.device,
        )
        return pooled[:, :, None] * scale[None, None, :]

    def _forward_frame_once(
        self, frame: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._counts["shared_v5_forward_frame_call_count"] += 1
        patches = self._forward_tokens_once(frame)
        base = F.adaptive_avg_pool2d(frame.mean(dim=1, keepdim=True), output_size=(2, 2))
        channels = torch.linspace(
            0.25,
            1.25,
            self._bev_feature_dim,
            dtype=frame.dtype,
            device=frame.device,
        )
        bev = base * channels[None, :, None, None]
        physical = torch.cat((base, 1.0 - base, torch.full_like(base, 0.5)), dim=1)
        return patches.detach(), bev.detach(), physical.detach()

    def __copy__(self) -> object:
        raise _noncopyable_reduce_error_v2("fake backend")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise _noncopyable_reduce_error_v2("fake backend")

    def __reduce__(self) -> object:
        raise _noncopyable_reduce_error_v2("fake backend")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise _noncopyable_reduce_error_v2("fake backend")


class _FrozenSharedFeatureCacheV2:
    __slots__ = (
        "_patch_features",
        "_bev_features",
        "_patch_observation",
        "_bev_observation",
        "_initial_content_sha256",
        "content_sha256",
    )

    def __init__(self, token: object, patch: torch.Tensor, bev: torch.Tensor) -> None:
        if token is not _FRAME_CONSTRUCTION_TOKEN:
            raise PermissionError("feature cache is runtime-created only")
        if type(patch) is not torch.Tensor or patch.ndim != 3:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError("patch feature shape changed")
        if type(bev) is not torch.Tensor or bev.ndim != 4 or patch.shape[0] != bev.shape[0]:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError("BEV feature shape changed")
        self._patch_features = patch.detach().clone().contiguous()
        self._bev_features = bev.detach().clone().contiguous()
        self._patch_observation = _TensorObservationV2.capture(
            self._patch_features, name="patch_features"
        )
        self._bev_observation = _TensorObservationV2.capture(
            self._bev_features, name="bev_features"
        )
        self.content_sha256 = canonical_json_sha256_v2(
            {
                "schema": "lewm_go2_frozen_shared_feature_cache_v2",
                "version": 2,
                "patch_features_sha256": self._patch_observation.content_sha256,
                "bev_features_sha256": self._bev_observation.content_sha256,
            }
        )
        self._initial_content_sha256 = self.content_sha256

    def _assert_unchanged(self) -> None:
        self._patch_observation.assert_unchanged(
            self._patch_features, name="patch_features"
        )
        self._bev_observation.assert_unchanged(self._bev_features, name="bev_features")
        expected = canonical_json_sha256_v2(
            {
                "schema": "lewm_go2_frozen_shared_feature_cache_v2",
                "version": 2,
                "patch_features_sha256": self._patch_observation.content_sha256,
                "bev_features_sha256": self._bev_observation.content_sha256,
            }
        )
        if self.content_sha256 != self._initial_content_sha256 or expected != self.content_sha256:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "feature cache commitment changed"
            )

    def __copy__(self) -> object:
        raise _noncopyable_reduce_error_v2("feature cache")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise _noncopyable_reduce_error_v2("feature cache")

    def __reduce__(self) -> object:
        raise _noncopyable_reduce_error_v2("feature cache")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise _noncopyable_reduce_error_v2("feature cache")


class QualifiedSharedV5FrameOutcomeV2:
    __slots__ = (
        "_runtime",
        "_tick_index",
        "_reset_id",
        "_session_id",
        "_pre_owner_states",
        "_controller_input_sha256",
        "_rgb_content_sha256",
        "_timestamp_binary64_hex",
        "_synchronization_id",
        "_feature_cache",
        "_physical_output",
        "_physical_observation",
        "content_sha256",
    )

    def __init__(
        self,
        token: object,
        *,
        runtime: "QualifiedSharedV5NavigationRuntimeV2",
        tick_index: int,
        reset_id: str,
        session_id: str,
        pre_owner_states: OwnerStateBundleV2,
        controller_input_sha256: str,
        rgb_content_sha256: str,
        timestamp_binary64_hex: str,
        synchronization_id: str,
        feature_cache: _FrozenSharedFeatureCacheV2,
        physical_output: torch.Tensor,
    ) -> None:
        if token is not _FRAME_CONSTRUCTION_TOKEN:
            raise PermissionError("frame outcome is runtime-created only")
        self._runtime = runtime
        self._tick_index = tick_index
        self._reset_id = reset_id
        self._session_id = session_id
        self._pre_owner_states = pre_owner_states
        self._controller_input_sha256 = require_sha256_v2(
            controller_input_sha256, name="controller_input_sha256"
        )
        self._rgb_content_sha256 = require_sha256_v2(
            rgb_content_sha256, name="rgb_content_sha256"
        )
        self._timestamp_binary64_hex = timestamp_binary64_hex
        self._synchronization_id = require_identifier_v2(
            synchronization_id, name="synchronization_id"
        )
        self._feature_cache = feature_cache
        self._physical_output = physical_output.detach().clone().contiguous()
        self._physical_observation = _TensorObservationV2.capture(
            self._physical_output, name="physical_output"
        )
        self.content_sha256 = canonical_json_sha256_v2(
            {
                "schema": "lewm_go2_qualified_shared_v5_frame_outcome_v2",
                "version": 2,
                "tick_index": tick_index,
                "reset_id": reset_id,
                "session_id": session_id,
                "pre_owner_states_sha256": pre_owner_states.content_sha256,
                "controller_input_sha256": self._controller_input_sha256,
                "rgb_content_sha256": self._rgb_content_sha256,
                "timestamp_binary64_hex": timestamp_binary64_hex,
                "synchronization_id": self._synchronization_id,
                "feature_cache_sha256": feature_cache.content_sha256,
                "physical_output_sha256": self._physical_observation.content_sha256,
                "shared_v5_checkpoint_file_sha256": None,
                "shared_v5_model_state_sha256": None,
                "g2_report_sha256": None,
                "g2_candidate_publication_sha256": None,
            }
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("QualifiedSharedV5FrameOutcomeV2 cannot be subclassed")

    @property
    def tick_index(self) -> int:
        return self._tick_index

    @property
    def reset_id(self) -> str:
        return self._reset_id

    @property
    def session_id(self) -> str:
        return self._session_id

    def _assert_unchanged(self) -> None:
        self._feature_cache._assert_unchanged()
        self._physical_observation.assert_unchanged(
            self._physical_output, name="physical_output"
        )

    def __copy__(self) -> object:
        raise _noncopyable_reduce_error_v2("frame outcome")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise _noncopyable_reduce_error_v2("frame outcome")

    def __reduce__(self) -> object:
        raise _noncopyable_reduce_error_v2("frame outcome")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise _noncopyable_reduce_error_v2("frame outcome")


class SyntheticPhysicalProjectionProducerV2:
    __slots__ = ("_authority", "producer_id", "_issued_receipts")

    def __init__(
        self,
        token: object,
        *,
        authority: "ResetAuthorityV2 | None",
        producer_id: str,
    ) -> None:
        if token is not _PRODUCER_CONSTRUCTION_TOKEN:
            raise PermissionError("physical producer is issuer-created only")
        self._authority = authority
        self.producer_id = require_identifier_v2(producer_id, name="producer_id")
        self._issued_receipts: list[PhysicalProjectionProducerReceiptV2] = []

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("SyntheticPhysicalProjectionProducerV2 cannot be subclassed")

    def _bind_authority_once(self, authority: "ResetAuthorityV2") -> None:
        if self._authority is not None:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "physical producer authority was already bound"
            )
        self._authority = authority

    def _mint(
        self,
        *,
        runtime: "QualifiedSharedV5NavigationRuntimeV2",
        outcome: QualifiedSharedV5FrameOutcomeV2,
        target_receipt: "TargetBatchReceiptV2",
    ) -> "PhysicalProjectionProducerReceiptV2":
        if self._authority is None or runtime._reset_authority is not self._authority:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "physical producer crossed reset authority"
            )
        target_receipt._assert_raw_and_snapshot_unchanged()
        pre = runtime._owner_states
        physical_content = canonical_json_sha256_v2(
            {
                "schema": "lewm_go2_synthetic_physical_transition_v2",
                "version": 2,
                "producer_id": self.producer_id,
                "frame_outcome_sha256": outcome.content_sha256,
                "target_batch_receipt_sha256": target_receipt.content_sha256,
                "pre_physical_state_sha256": pre.row("physical").content_sha256,
            }
        )
        configuration_content = canonical_json_sha256_v2(
            {
                "schema": "lewm_go2_synthetic_configuration_projection_v2",
                "version": 2,
                "producer_id": self.producer_id,
                "physical_content_sha256": physical_content,
                "pre_configuration_state_sha256": pre.row("configuration").content_sha256,
            }
        )
        post_physical = OwnerStateV2(
            owner_name="physical",
            owner_id=pre.row("physical").owner_id,
            revision=pre.row("physical").revision + 1,
            owner_content_sha256=physical_content,
            reset_id=pre.reset_id,
            session_id=pre.session_id,
        )
        post_configuration = OwnerStateV2(
            owner_name="configuration",
            owner_id=pre.row("configuration").owner_id,
            revision=pre.row("configuration").revision + 1,
            owner_content_sha256=configuration_content,
            reset_id=pre.reset_id,
            session_id=pre.session_id,
        )
        receipt = PhysicalProjectionProducerReceiptV2(
            _PHYSICAL_RECEIPT_CONSTRUCTION_TOKEN,
            producer=self,
            runtime=runtime,
            outcome=outcome,
            target_receipt=target_receipt,
            pre_owner_states=pre,
            post_physical_state=post_physical,
            post_configuration_state=post_configuration,
        )
        self._issued_receipts.append(receipt)
        return receipt

    def __copy__(self) -> object:
        raise _noncopyable_reduce_error_v2("physical producer")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise _noncopyable_reduce_error_v2("physical producer")

    def __reduce__(self) -> object:
        raise _noncopyable_reduce_error_v2("physical producer")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise _noncopyable_reduce_error_v2("physical producer")


class SyntheticCandidateProducerV2:
    __slots__ = ("_authority", "producer_id", "_issued_admissions")

    def __init__(
        self,
        token: object,
        *,
        authority: "ResetAuthorityV2 | None",
        producer_id: str,
    ) -> None:
        if token is not _PRODUCER_CONSTRUCTION_TOKEN:
            raise PermissionError("candidate producer is issuer-created only")
        self._authority = authority
        self.producer_id = require_identifier_v2(producer_id, name="producer_id")
        self._issued_admissions: list[CandidateSetAdmissionV2] = []

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("SyntheticCandidateProducerV2 cannot be subclassed")

    def _bind_authority_once(self, authority: "ResetAuthorityV2") -> None:
        if self._authority is not None:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "candidate producer authority was already bound"
            )
        self._authority = authority

    def _mint(
        self,
        *,
        runtime: "QualifiedSharedV5NavigationRuntimeV2",
        receipt: "TickAdmissionReceiptV2",
        physical_payload: "PhysicalViewPayloadV2",
        rows: tuple["SyntheticCandidateRowV2", ...],
        features: torch.Tensor,
    ) -> "CandidateSetAdmissionV2":
        if self._authority is None or runtime._reset_authority is not self._authority:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "candidate producer crossed reset authority"
            )
        candidate_identity = runtime._issuer._mint_live_identity()
        admission = CandidateSetAdmissionV2(
            _CANDIDATE_CONSTRUCTION_TOKEN,
            producer=self,
            runtime=runtime,
            tick_receipt=receipt,
            physical_payload=physical_payload,
            candidate_identity=candidate_identity,
            rows=rows,
            features=features,
        )
        self._issued_admissions.append(admission)
        return admission

    def __copy__(self) -> object:
        raise _noncopyable_reduce_error_v2("candidate producer")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise _noncopyable_reduce_error_v2("candidate producer")

    def __reduce__(self) -> object:
        raise _noncopyable_reduce_error_v2("candidate producer")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise _noncopyable_reduce_error_v2("candidate producer")


class ResetAuthorityV2:
    """Opaque live reset capability; its ResetReceiptV2 is evidence only."""

    __slots__ = (
        "_issuer",
        "_receipt",
        "_physical_producer",
        "_candidate_producer",
        "_used",
    )

    def __init__(
        self,
        token: object,
        *,
        issuer: "FoundationAuthorityIssuerV2",
        receipt: ResetReceiptV2,
        physical_producer: SyntheticPhysicalProjectionProducerV2,
        candidate_producer: SyntheticCandidateProducerV2,
    ) -> None:
        if token is not _RESET_AUTHORITY_CONSTRUCTION_TOKEN:
            raise PermissionError("ResetAuthorityV2 is issuer-created only")
        self._issuer = issuer
        self._receipt = receipt
        self._physical_producer = physical_producer
        self._candidate_producer = candidate_producer
        self._used = False

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("ResetAuthorityV2 cannot be subclassed")

    @property
    def evidence_receipt(self) -> ResetReceiptV2:
        return self._receipt

    def __copy__(self) -> object:
        raise _noncopyable_reduce_error_v2("reset authority")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise _noncopyable_reduce_error_v2("reset authority")

    def __reduce__(self) -> object:
        raise _noncopyable_reduce_error_v2("reset authority")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise _noncopyable_reduce_error_v2("reset authority")


class FoundationAuthorityIssuerV2:
    """Synthetic-only origin for fresh reset, producer, backend, and candidate IDs."""

    __slots__ = (
        "issuer_id",
        "_identity_sequence",
        "_authority_sequence",
        "_all_identities",
        "_live_authorities",
        "_spent_authorities",
        "_live_backends",
        "_spent_backends",
    )

    def __init__(self, token: object) -> None:
        if token is not _ISSUER_CONSTRUCTION_TOKEN:
            raise PermissionError("use FoundationAuthorityIssuerV2.create_for_source_tests")
        self.issuer_id = f"issuer-v2-{secrets.token_hex(16)}"
        self._identity_sequence = 0
        self._authority_sequence = 0
        self._all_identities: set[str] = {self.issuer_id}
        self._live_authorities: list[ResetAuthorityV2] = []
        self._spent_authorities: list[ResetAuthorityV2] = []
        self._live_backends: list[FakeSharedV5FrameBackendV2] = []
        self._spent_backends: list[FakeSharedV5FrameBackendV2] = []

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("FoundationAuthorityIssuerV2 cannot be subclassed")

    @classmethod
    def create_for_source_tests(
        cls, *, _source_test_capability: bool = False
    ) -> "FoundationAuthorityIssuerV2":
        if _source_test_capability is not True:
            raise PermissionError("Foundation V2 issuer requires explicit source-test capability")
        return cls(_ISSUER_CONSTRUCTION_TOKEN)

    def _mint_live_identity(self) -> str:
        while True:
            self._identity_sequence += 1
            identity = f"v2-{self._identity_sequence:016x}-{secrets.token_hex(16)}"
            if identity not in self._all_identities:
                self._all_identities.add(identity)
                return identity

    def mint_reset_authority(self) -> ResetAuthorityV2:
        self._authority_sequence += 1
        reset_id = self._mint_live_identity()
        session_id = self._mint_live_identity()
        capability_id = self._mint_live_identity()
        owner_ids = tuple(self._mint_live_identity() for _ in OWNER_NAMES_V2)
        physical_producer_id = self._mint_live_identity()
        candidate_producer_id = self._mint_live_identity()
        owner_rows = tuple(
            OwnerStateV2(
                owner_name=name,
                owner_id=owner_id,
                revision=0,
                owner_content_sha256=initial_owner_content_sha256_v2(
                    owner_name=name,
                    owner_id=owner_id,
                    reset_id=reset_id,
                    session_id=session_id,
                ),
                reset_id=reset_id,
                session_id=session_id,
            )
            for name, owner_id in zip(OWNER_NAMES_V2, owner_ids)
        )
        receipt = ResetReceiptV2(
            authority_sequence=self._authority_sequence,
            issuer_id=self.issuer_id,
            reset_id=reset_id,
            session_id=session_id,
            reset_capability_id=capability_id,
            physical_projection_producer_id=physical_producer_id,
            candidate_producer_id=candidate_producer_id,
            initial_owner_states=OwnerStateBundleV2(owner_rows),
        )
        physical = SyntheticPhysicalProjectionProducerV2(
            _PRODUCER_CONSTRUCTION_TOKEN,
            authority=None,
            producer_id=physical_producer_id,
        )
        candidate = SyntheticCandidateProducerV2(
            _PRODUCER_CONSTRUCTION_TOKEN,
            authority=None,
            producer_id=candidate_producer_id,
        )
        authority = ResetAuthorityV2(
            _RESET_AUTHORITY_CONSTRUCTION_TOKEN,
            issuer=self,
            receipt=receipt,
            physical_producer=physical,
            candidate_producer=candidate,
        )
        physical._bind_authority_once(authority)
        candidate._bind_authority_once(authority)
        self._live_authorities.append(authority)
        return authority

    def create_fake_frame_backend(
        self,
        *,
        patch_feature_dim: int,
        bev_feature_dim: int,
    ) -> FakeSharedV5FrameBackendV2:
        backend = FakeSharedV5FrameBackendV2(
            _BACKEND_CONSTRUCTION_TOKEN,
            issuer=self,
            patch_feature_dim=patch_feature_dim,
            bev_feature_dim=bev_feature_dim,
        )
        self._live_backends.append(backend)
        return backend

    def start_synthetic_runtime(
        self,
        *,
        reset_authority: ResetAuthorityV2,
        backend: FakeSharedV5FrameBackendV2,
    ) -> "QualifiedSharedV5NavigationRuntimeV2":
        if type(reset_authority) is not ResetAuthorityV2 or not any(
            reset_authority is item for item in self._live_authorities
        ):
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "reset authority is reconstructed, foreign, spent, or unregistered"
            )
        if type(backend) is not FakeSharedV5FrameBackendV2 or not any(
            backend is item for item in self._live_backends
        ):
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "fake backend is reconstructed, foreign, spent, or unregistered"
            )
        if reset_authority._issuer is not self or backend._issuer is not self:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "reset authority/backend crossed issuer"
            )
        if reset_authority._used or backend._consumed:
            raise QualifiedSharedV5NavigationRuntimeV2ReplayError(
                "reset authority or backend was already consumed"
            )
        reset_authority._used = True
        backend._consumed = True
        self._live_authorities.remove(reset_authority)
        self._spent_authorities.append(reset_authority)
        self._live_backends.remove(backend)
        self._spent_backends.append(backend)
        return QualifiedSharedV5NavigationRuntimeV2(
            _ISSUER_CONSTRUCTION_TOKEN,
            issuer=self,
            reset_authority=reset_authority,
            backend=backend,
        )

    def __copy__(self) -> object:
        raise _noncopyable_reduce_error_v2("authority issuer")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise _noncopyable_reduce_error_v2("authority issuer")

    def __reduce__(self) -> object:
        raise _noncopyable_reduce_error_v2("authority issuer")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise _noncopyable_reduce_error_v2("authority issuer")


def _validate_four_color_output_domain_v2(
    raw: FourColorTargetObservationOutputV1,
    *,
    maximum_range_m: float,
) -> tuple[int, str]:
    if type(raw) is not FourColorTargetObservationOutputV1:
        raise QualifiedSharedV5NavigationRuntimeV2BindingError(
            "target head output exact type changed"
        )
    if type(raw.colors) is not tuple or raw.colors != CANONICAL_COLORS_V2:
        raise QualifiedSharedV5NavigationRuntimeV2BindingError(
            "target output color order changed"
        )
    common_shape: tuple[int, int] | None = None
    common_device: str | None = None
    for name in _TARGET_TENSOR_NAMES:
        tensor = getattr(raw, name)
        if type(tensor) is not torch.Tensor or tensor.ndim != 2 or tensor.shape[1] != 4:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                f"{name} must be exact nonempty [batch,4] tensor"
            )
        _tensor_bytes_sha256_v2(tensor, name=name)
        shape = (int(tensor.shape[0]), int(tensor.shape[1]))
        if shape[0] <= 0:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "target output batch is empty"
            )
        if common_shape is None:
            common_shape = shape
            common_device = str(tensor.device)
        elif shape != common_shape or str(tensor.device) != common_device:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "target tensor shape/device differs"
            )
    if not torch.equal(raw.presence_probability, torch.sigmoid(raw.presence_logit)):
        raise QualifiedSharedV5NavigationRuntimeV2BindingError(
            "presence probability is not bit-consistent with logit"
        )
    if bool(
        ((raw.presence_probability < 0.0) | (raw.presence_probability > 1.0)).any().item()
    ):
        raise QualifiedSharedV5NavigationRuntimeV2BindingError(
            "presence probability left [0,1]"
        )
    if bool(((raw.quality < 0.0) | (raw.quality > 1.0)).any().item()):
        raise QualifiedSharedV5NavigationRuntimeV2BindingError("quality left [0,1]")
    if bool(
        ((raw.bearing_mean_rad < -math.pi) | (raw.bearing_mean_rad > math.pi)).any().item()
    ):
        raise QualifiedSharedV5NavigationRuntimeV2BindingError(
            "bearing mean left [-pi,pi]"
        )
    for name in ("bearing_scale_rad", "range_scale_m", "uncertainty"):
        if bool((getattr(raw, name) <= 0.0).any().item()):
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                f"{name} must be strictly positive"
            )
    if bool(
        ((raw.range_mean_m <= 0.0) | (raw.range_mean_m > maximum_range_m)).any().item()
    ):
        raise QualifiedSharedV5NavigationRuntimeV2BindingError(
            "range mean is outside (0, configured maximum]"
        )
    assert common_shape is not None and common_device is not None
    return common_shape[0], common_device


class FrozenFourColorTargetBatchV2:
    """Private independently-owned snapshot made inside the counted call."""

    __slots__ = (
        "_frame",
        "_head",
        "_head_config_sha256",
        "_counter_receipt_sha256",
        "_tensors",
        "_observations",
        "_batch_size",
        "_device",
        "_initial_binding",
        "content_sha256",
    )

    def __init__(
        self,
        token: object,
        *,
        frame: QualifiedSharedV5FrameOutcomeV2,
        head: SharedV5TargetObservationHeadV1,
        raw: FourColorTargetObservationOutputV1,
        batch_size: int,
        device: str,
        counter_receipt_sha256: str,
    ) -> None:
        if token is not _TARGET_CONSTRUCTION_TOKEN:
            raise PermissionError("target batch snapshot is runtime-created only")
        self._frame = frame
        self._head = head
        self._head_config_sha256 = head.architecture_config_sha256
        self._counter_receipt_sha256 = require_sha256_v2(
            counter_receipt_sha256, name="counter_receipt_sha256"
        )
        self._batch_size = batch_size
        self._device = device
        tensors: dict[str, torch.Tensor] = {}
        observations: dict[str, _TensorObservationV2] = {}
        for name in _TARGET_TENSOR_NAMES:
            raw_tensor = getattr(raw, name)
            clone = raw_tensor.detach().clone().contiguous()
            if int(clone.untyped_storage().data_ptr()) == int(
                raw_tensor.untyped_storage().data_ptr()
            ):
                raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                    "target snapshot aliases raw output storage"
                )
            tensors[name] = clone
            observations[name] = _TensorObservationV2.capture(
                clone, name=f"frozen_{name}"
            )
        self._tensors = tensors
        self._observations = observations
        self.content_sha256 = canonical_json_sha256_v2(
            {
                "schema": "lewm_go2_frozen_four_color_target_batch_v2",
                "version": 2,
                "frame_outcome_sha256": frame.content_sha256,
                "head_exact_object_id": id(head),
                "head_config_sha256": self._head_config_sha256,
                "counter_receipt_sha256": self._counter_receipt_sha256,
                "tick_index": frame.tick_index,
                "reset_id": frame.reset_id,
                "session_id": frame.session_id,
                "batch_size": batch_size,
                "device": device,
                "colors": list(CANONICAL_COLORS_V2),
                "tensor_sha256s": {
                    name: observations[name].content_sha256 for name in _TARGET_TENSOR_NAMES
                },
            }
        )
        self._initial_binding = (
            frame,
            head,
            self._head_config_sha256,
            self._counter_receipt_sha256,
            batch_size,
            device,
            self.content_sha256,
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("FrozenFourColorTargetBatchV2 cannot be subclassed")

    def _assert_unchanged(self) -> None:
        if (
            self._frame,
            self._head,
            self._head_config_sha256,
            self._counter_receipt_sha256,
            self._batch_size,
            self._device,
            self.content_sha256,
        ) != self._initial_binding:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "frozen target batch binding changed"
            )
        if (
            type(self._head) is not SharedV5TargetObservationHeadV1
            or self._head.architecture_config_sha256 != self._head_config_sha256
        ):
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "target head/config binding changed"
            )
        for name in _TARGET_TENSOR_NAMES:
            self._observations[name].assert_unchanged(
                self._tensors[name], name=f"frozen_{name}"
            )

    def __copy__(self) -> object:
        raise _noncopyable_reduce_error_v2("frozen target batch")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise _noncopyable_reduce_error_v2("frozen target batch")

    def __reduce__(self) -> object:
        raise _noncopyable_reduce_error_v2("frozen target batch")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise _noncopyable_reduce_error_v2("frozen target batch")


class TargetBatchReceiptV2:
    """Opaque diagnostic/live receipt; authoritative tensors stay private."""

    __slots__ = (
        "_runtime",
        "_frame",
        "_head",
        "_raw_output",
        "_raw_output_id",
        "_raw_observations",
        "_snapshot",
        "_head_config_sha256",
        "_counter_receipt_sha256",
        "_live",
        "_consumed_by_producer",
        "content_sha256",
    )

    def __init__(
        self,
        token: object,
        *,
        runtime: "QualifiedSharedV5NavigationRuntimeV2",
        frame: QualifiedSharedV5FrameOutcomeV2,
        head: SharedV5TargetObservationHeadV1,
        raw_output: FourColorTargetObservationOutputV1,
        raw_observations: dict[str, _TensorObservationV2],
        snapshot: FrozenFourColorTargetBatchV2,
        counter_receipt_sha256: str,
    ) -> None:
        if token is not _TARGET_CONSTRUCTION_TOKEN:
            raise PermissionError("target batch receipt is runtime-created only")
        self._runtime = runtime
        self._frame = frame
        self._head = head
        self._raw_output = raw_output
        self._raw_output_id = id(raw_output)
        self._raw_observations = raw_observations
        self._snapshot = snapshot
        self._head_config_sha256 = head.architecture_config_sha256
        self._counter_receipt_sha256 = require_sha256_v2(
            counter_receipt_sha256, name="counter_receipt_sha256"
        )
        self._live = True
        self._consumed_by_producer = False
        self.content_sha256 = canonical_json_sha256_v2(
            {
                "schema": "lewm_go2_target_batch_receipt_v2",
                "version": 2,
                "frame_outcome_sha256": frame.content_sha256,
                "head_exact_object_id": id(head),
                "head_config_sha256": head.architecture_config_sha256,
                "raw_output_exact_object_id": self._raw_output_id,
                "raw_tensor_exact_object_ids": {
                    name: raw_observations[name].exact_object_id for name in _TARGET_TENSOR_NAMES
                },
                "raw_tensor_sha256s": {
                    name: raw_observations[name].content_sha256 for name in _TARGET_TENSOR_NAMES
                },
                "frozen_batch_sha256": snapshot.content_sha256,
                "counter_receipt_sha256": self._counter_receipt_sha256,
                "tick_index": frame.tick_index,
                "reset_id": frame.reset_id,
                "session_id": frame.session_id,
            }
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("TargetBatchReceiptV2 cannot be subclassed")

    def _assert_raw_and_snapshot_unchanged(self) -> None:
        if not self._live:
            raise QualifiedSharedV5NavigationRuntimeV2ReplayError(
                "target batch receipt expired"
            )
        if id(self._raw_output) != self._raw_output_id or type(self._raw_output) is not FourColorTargetObservationOutputV1:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "raw target output identity/type changed"
            )
        if (
            type(self._head) is not SharedV5TargetObservationHeadV1
            or self._head.architecture_config_sha256 != self._head_config_sha256
            or self._snapshot._head is not self._head
        ):
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "target head/config binding changed"
            )
        for name in _TARGET_TENSOR_NAMES:
            self._raw_observations[name].assert_unchanged(
                getattr(self._raw_output, name), name=f"raw_{name}"
            )
        _validate_four_color_output_domain_v2(
            self._raw_output,
            maximum_range_m=float(self._head.config.maximum_range_m),
        )
        self._snapshot._assert_unchanged()

    def diagnostic_dict(self) -> dict[str, object]:
        return {
            "schema": "lewm_go2_target_batch_receipt_diagnostic_v2",
            "version": 2,
            "content_sha256": self.content_sha256,
            "frame_outcome_sha256": self._frame.content_sha256,
            "frozen_batch_sha256": self._snapshot.content_sha256,
            "counter_receipt_sha256": self._counter_receipt_sha256,
            "tick_index": self._frame.tick_index,
            "reset_id": self._frame.reset_id,
            "session_id": self._frame.session_id,
        }

    def __copy__(self) -> object:
        raise _noncopyable_reduce_error_v2("target batch receipt")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise _noncopyable_reduce_error_v2("target batch receipt")

    def __reduce__(self) -> object:
        raise _noncopyable_reduce_error_v2("target batch receipt")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise _noncopyable_reduce_error_v2("target batch receipt")


class PhysicalViewPayloadV2:
    """Immutable post-state binding; it contains no cache or caller cells."""

    __slots__ = (
        "producer_id",
        "reset_id",
        "session_id",
        "tick_index",
        "frame_outcome_sha256",
        "target_batch_receipt_sha256",
        "pre_owner_bundle_sha256",
        "post_physical_state",
        "post_configuration_state",
        "physical_transaction_sha256",
        "physical_retraction_sha256",
        "configuration_snapshot_sha256",
        "configuration_component_sha256",
        "frontier_sha256",
        "_initial_binding",
        "content_sha256",
    )

    def __init__(
        self,
        token: object,
        *,
        producer_id: str,
        reset_id: str,
        session_id: str,
        tick_index: int,
        frame_outcome_sha256: str,
        target_batch_receipt_sha256: str,
        pre_owner_bundle_sha256: str,
        post_physical_state: OwnerStateV2,
        post_configuration_state: OwnerStateV2,
    ) -> None:
        if token is not _PHYSICAL_RECEIPT_CONSTRUCTION_TOKEN:
            raise PermissionError("physical view payload is producer-created only")
        self.producer_id = require_identifier_v2(producer_id, name="producer_id")
        self.reset_id = require_identifier_v2(reset_id, name="reset_id")
        self.session_id = require_identifier_v2(session_id, name="session_id")
        self.tick_index = require_nonnegative_int_v2(tick_index, name="tick_index")
        self.frame_outcome_sha256 = require_sha256_v2(
            frame_outcome_sha256, name="frame_outcome_sha256"
        )
        self.target_batch_receipt_sha256 = require_sha256_v2(
            target_batch_receipt_sha256, name="target_batch_receipt_sha256"
        )
        self.pre_owner_bundle_sha256 = require_sha256_v2(
            pre_owner_bundle_sha256, name="pre_owner_bundle_sha256"
        )
        if type(post_physical_state) is not OwnerStateV2 or post_physical_state.owner_name != "physical":
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "physical payload post-physical state changed"
            )
        if type(post_configuration_state) is not OwnerStateV2 or post_configuration_state.owner_name != "configuration":
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "physical payload post-configuration state changed"
            )
        self.post_physical_state = post_physical_state
        self.post_configuration_state = post_configuration_state
        base = {
            "producer_id": self.producer_id,
            "reset_id": self.reset_id,
            "session_id": self.session_id,
            "tick_index": self.tick_index,
            "frame_outcome_sha256": self.frame_outcome_sha256,
            "target_batch_receipt_sha256": self.target_batch_receipt_sha256,
            "pre_owner_bundle_sha256": self.pre_owner_bundle_sha256,
            "post_physical_state_sha256": post_physical_state.content_sha256,
            "post_configuration_state_sha256": post_configuration_state.content_sha256,
        }
        self.physical_transaction_sha256 = canonical_json_sha256_v2(
            {"schema": "lewm_go2_synthetic_physical_transaction_v2", "version": 2, **base}
        )
        self.physical_retraction_sha256 = canonical_json_sha256_v2(
            {
                "schema": "lewm_go2_synthetic_physical_retraction_empty_v2",
                "version": 2,
                **base,
                "retractions": [],
            }
        )
        self.configuration_snapshot_sha256 = canonical_json_sha256_v2(
            {"schema": "lewm_go2_synthetic_configuration_snapshot_v2", "version": 2, **base}
        )
        self.configuration_component_sha256 = canonical_json_sha256_v2(
            {"schema": "lewm_go2_synthetic_configuration_component_v2", "version": 2, **base}
        )
        self.frontier_sha256 = canonical_json_sha256_v2(
            {"schema": "lewm_go2_synthetic_frontier_v2", "version": 2, **base}
        )
        self.content_sha256 = canonical_json_sha256_v2(
            {
                "schema": "lewm_go2_physical_view_payload_v2",
                "version": 2,
                **base,
                "physical_transaction_sha256": self.physical_transaction_sha256,
                "physical_retraction_sha256": self.physical_retraction_sha256,
                "configuration_snapshot_sha256": self.configuration_snapshot_sha256,
                "configuration_component_sha256": self.configuration_component_sha256,
                "frontier_sha256": self.frontier_sha256,
            }
        )
        self._initial_binding = (
            self.producer_id,
            self.reset_id,
            self.session_id,
            self.tick_index,
            self.frame_outcome_sha256,
            self.target_batch_receipt_sha256,
            self.pre_owner_bundle_sha256,
            self.post_physical_state,
            self.post_configuration_state,
            self.physical_transaction_sha256,
            self.physical_retraction_sha256,
            self.configuration_snapshot_sha256,
            self.configuration_component_sha256,
            self.frontier_sha256,
            self.content_sha256,
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("PhysicalViewPayloadV2 cannot be subclassed")

    def _assert_unchanged(self) -> None:
        current = (
            self.producer_id,
            self.reset_id,
            self.session_id,
            self.tick_index,
            self.frame_outcome_sha256,
            self.target_batch_receipt_sha256,
            self.pre_owner_bundle_sha256,
            self.post_physical_state,
            self.post_configuration_state,
            self.physical_transaction_sha256,
            self.physical_retraction_sha256,
            self.configuration_snapshot_sha256,
            self.configuration_component_sha256,
            self.frontier_sha256,
            self.content_sha256,
        )
        if current != self._initial_binding:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "physical payload binding changed"
            )
        require_sha256_v2(self.content_sha256, name="physical_view_payload.content_sha256")
        if (
            type(self.post_physical_state) is not OwnerStateV2
            or type(self.post_configuration_state) is not OwnerStateV2
            or self.post_physical_state.owner_name != "physical"
            or self.post_configuration_state.owner_name != "configuration"
        ):
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "physical payload owner state changed"
            )

    def __copy__(self) -> object:
        raise _noncopyable_reduce_error_v2("physical view payload")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise _noncopyable_reduce_error_v2("physical view payload")

    def __reduce__(self) -> object:
        raise _noncopyable_reduce_error_v2("physical view payload")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise _noncopyable_reduce_error_v2("physical view payload")


class PhysicalProjectionProducerReceiptV2:
    __slots__ = (
        "_producer",
        "_runtime",
        "_outcome",
        "_target_receipt",
        "_pre_owner_states",
        "_payload",
        "_consumed",
        "content_sha256",
    )

    def __init__(
        self,
        token: object,
        *,
        producer: SyntheticPhysicalProjectionProducerV2,
        runtime: "QualifiedSharedV5NavigationRuntimeV2",
        outcome: QualifiedSharedV5FrameOutcomeV2,
        target_receipt: TargetBatchReceiptV2,
        pre_owner_states: OwnerStateBundleV2,
        post_physical_state: OwnerStateV2,
        post_configuration_state: OwnerStateV2,
    ) -> None:
        if token is not _PHYSICAL_RECEIPT_CONSTRUCTION_TOKEN:
            raise PermissionError("physical producer receipt is producer-created only")
        self._producer = producer
        self._runtime = runtime
        self._outcome = outcome
        self._target_receipt = target_receipt
        self._pre_owner_states = pre_owner_states
        self._consumed = False
        self._payload = PhysicalViewPayloadV2(
            _PHYSICAL_RECEIPT_CONSTRUCTION_TOKEN,
            producer_id=producer.producer_id,
            reset_id=outcome.reset_id,
            session_id=outcome.session_id,
            tick_index=outcome.tick_index,
            frame_outcome_sha256=outcome.content_sha256,
            target_batch_receipt_sha256=target_receipt.content_sha256,
            pre_owner_bundle_sha256=pre_owner_states.content_sha256,
            post_physical_state=post_physical_state,
            post_configuration_state=post_configuration_state,
        )
        self.content_sha256 = canonical_json_sha256_v2(
            {
                "schema": "lewm_go2_physical_projection_producer_receipt_v2",
                "version": 2,
                "producer_id": producer.producer_id,
                "producer_exact_object_id": id(producer),
                "frame_outcome_sha256": outcome.content_sha256,
                "frame_outcome_exact_object_id": id(outcome),
                "target_batch_receipt_sha256": target_receipt.content_sha256,
                "target_batch_receipt_exact_object_id": id(target_receipt),
                "pre_owner_bundle_sha256": pre_owner_states.content_sha256,
                "physical_view_payload_sha256": self._payload.content_sha256,
                "reset_id": outcome.reset_id,
                "session_id": outcome.session_id,
                "tick_index": outcome.tick_index,
            }
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("PhysicalProjectionProducerReceiptV2 cannot be subclassed")

    def _assert_unchanged(self) -> None:
        if self._runtime._active_outcome is not self._outcome:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "physical producer receipt frame is not active exact object"
            )
        if self._runtime._owner_states.to_dict() != self._pre_owner_states.to_dict():
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "physical producer receipt pre-owner state changed"
            )
        self._outcome._assert_unchanged()
        self._target_receipt._assert_raw_and_snapshot_unchanged()
        self._payload._assert_unchanged()

    def __copy__(self) -> object:
        raise _noncopyable_reduce_error_v2("physical producer receipt")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise _noncopyable_reduce_error_v2("physical producer receipt")

    def __reduce__(self) -> object:
        raise _noncopyable_reduce_error_v2("physical producer receipt")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise _noncopyable_reduce_error_v2("physical producer receipt")


class FrozenTargetColorPayloadV2:
    __slots__ = (
        "color",
        "color_index",
        "frozen_batch_sha256",
        "_tensors",
        "_observations",
        "_initial_binding",
        "content_sha256",
    )

    def __init__(
        self,
        token: object,
        *,
        snapshot: FrozenFourColorTargetBatchV2,
        color: str,
        color_index: int,
    ) -> None:
        if token is not _ADMISSION_CONSTRUCTION_TOKEN:
            raise PermissionError("target color payload is admission-created only")
        if type(color_index) is not int or not 0 <= color_index < 4:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError("color index is invalid")
        if type(color) is not str or color != CANONICAL_COLORS_V2[color_index]:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError("color order changed")
        snapshot._assert_unchanged()
        self.color = color
        self.color_index = color_index
        self.frozen_batch_sha256 = snapshot.content_sha256
        tensors: dict[str, torch.Tensor] = {}
        observations: dict[str, _TensorObservationV2] = {}
        for name in _TARGET_TENSOR_NAMES:
            selected = snapshot._tensors[name][:, color_index].detach().clone().contiguous()
            tensors[name] = selected
            observations[name] = _TensorObservationV2.capture(
                selected, name=f"target_{color}_{name}"
            )
        self._tensors = tensors
        self._observations = observations
        self.content_sha256 = canonical_json_sha256_v2(
            {
                "schema": "lewm_go2_frozen_target_color_payload_v2",
                "version": 2,
                "color": color,
                "color_index": color_index,
                "frozen_batch_sha256": self.frozen_batch_sha256,
                "tensor_sha256s": {
                    name: observations[name].content_sha256 for name in _TARGET_TENSOR_NAMES
                },
            }
        )
        self._initial_binding = (
            self.color,
            self.color_index,
            self.frozen_batch_sha256,
            self.content_sha256,
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("FrozenTargetColorPayloadV2 cannot be subclassed")

    def tensor(self, name: str) -> torch.Tensor:
        if type(name) is not str or name not in _TARGET_TENSOR_NAMES:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "target payload tensor name is invalid"
            )
        return self._tensors[name]

    def _assert_unchanged(self) -> None:
        if (
            self.color,
            self.color_index,
            self.frozen_batch_sha256,
            self.content_sha256,
        ) != self._initial_binding:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "target color payload binding changed"
            )
        for name in _TARGET_TENSOR_NAMES:
            self._observations[name].assert_unchanged(
                self._tensors[name], name=f"target_{self.color}_{name}"
            )

    def __copy__(self) -> object:
        raise _noncopyable_reduce_error_v2("target color payload")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise _noncopyable_reduce_error_v2("target color payload")

    def __reduce__(self) -> object:
        raise _noncopyable_reduce_error_v2("target color payload")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise _noncopyable_reduce_error_v2("target color payload")


class ExactObjectLeaseV2:
    """Opaque registered handle. It deliberately stores no payload."""

    __slots__ = ("_receipt_marker", "_kind_marker", "_consumer_marker", "_epoch_marker")

    def __init__(
        self,
        token: object,
        *,
        receipt: "TickAdmissionReceiptV2",
        kind: str,
        consumer: object,
        expiry_epoch: int,
    ) -> None:
        if token is not _LEASE_CONSTRUCTION_TOKEN:
            raise PermissionError("ExactObjectLeaseV2 is receipt-created only")
        self._receipt_marker = receipt
        self._kind_marker = kind
        self._consumer_marker = consumer
        self._epoch_marker = expiry_epoch

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("ExactObjectLeaseV2 cannot be subclassed")

    def __copy__(self) -> object:
        raise _noncopyable_reduce_error_v2("lease")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise _noncopyable_reduce_error_v2("lease")

    def __reduce__(self) -> object:
        raise _noncopyable_reduce_error_v2("lease")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise _noncopyable_reduce_error_v2("lease")


@dataclass
class _LeaseRegistrationV2:
    lease: ExactObjectLeaseV2
    kind: str
    receipt: "TickAdmissionReceiptV2"
    runtime: "QualifiedSharedV5NavigationRuntimeV2"
    outcome: QualifiedSharedV5FrameOutcomeV2
    producer_receipt: PhysicalProjectionProducerReceiptV2
    consumer: object
    payload: object
    reset_id: str
    session_id: str
    tick_index: int
    expiry_epoch: int
    consumed: bool = False


class TickAdmissionReceiptV2:
    __slots__ = (
        "_runtime",
        "_outcome",
        "_target_receipt",
        "_producer_receipt",
        "_physical_payload",
        "_target_payloads",
        "_registry",
        "_live",
        "_expiry_epoch",
        "_candidate_admission",
        "content_sha256",
    )

    def __init__(
        self,
        token: object,
        *,
        runtime: "QualifiedSharedV5NavigationRuntimeV2",
        outcome: QualifiedSharedV5FrameOutcomeV2,
        target_receipt: TargetBatchReceiptV2,
        producer_receipt: PhysicalProjectionProducerReceiptV2,
        expiry_epoch: int,
    ) -> None:
        if token is not _ADMISSION_CONSTRUCTION_TOKEN:
            raise PermissionError("tick admission receipt is runtime-created only")
        self._runtime = runtime
        self._outcome = outcome
        self._target_receipt = target_receipt
        self._producer_receipt = producer_receipt
        self._physical_payload = producer_receipt._payload
        self._target_payloads = tuple(
            FrozenTargetColorPayloadV2(
                _ADMISSION_CONSTRUCTION_TOKEN,
                snapshot=target_receipt._snapshot,
                color=color,
                color_index=index,
            )
            for index, color in enumerate(CANONICAL_COLORS_V2)
        )
        self._registry: dict[str, _LeaseRegistrationV2] = {}
        self._live = True
        self._expiry_epoch = expiry_epoch
        self._candidate_admission: CandidateSetAdmissionV2 | None = None
        self.content_sha256 = canonical_json_sha256_v2(
            {
                "schema": "lewm_go2_tick_admission_receipt_v2",
                "version": 2,
                "frame_outcome_sha256": outcome.content_sha256,
                "frame_outcome_exact_object_id": id(outcome),
                "target_batch_receipt_sha256": target_receipt.content_sha256,
                "target_batch_receipt_exact_object_id": id(target_receipt),
                "physical_producer_receipt_sha256": producer_receipt.content_sha256,
                "physical_producer_receipt_exact_object_id": id(producer_receipt),
                "physical_view_payload_sha256": self._physical_payload.content_sha256,
                "target_payload_sha256s": [item.content_sha256 for item in self._target_payloads],
                "tick_index": outcome.tick_index,
                "reset_id": outcome.reset_id,
                "session_id": outcome.session_id,
                "expiry_epoch": expiry_epoch,
            }
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("TickAdmissionReceiptV2 cannot be subclassed")

    @property
    def tick_index(self) -> int:
        return self._outcome.tick_index

    @property
    def reset_id(self) -> str:
        return self._outcome.reset_id

    @property
    def session_id(self) -> str:
        return self._outcome.session_id

    def issued_kinds_diagnostic(self) -> tuple[str, ...]:
        return tuple(self._registry)

    def _fail_terminal(self, error: Exception) -> None:
        self._runtime._seal_terminal()
        raise error

    def _assert_live(self) -> None:
        if (
            not self._live
            or self._runtime._active_admission is not self
            or self._runtime._active_outcome is not self._outcome
            or self._runtime._expiry_epoch != self._expiry_epoch
        ):
            self._fail_terminal(
                QualifiedSharedV5NavigationRuntimeV2ReplayError(
                    "admission receipt is expired, foreign, or inactive"
                )
            )
        try:
            self._outcome._assert_unchanged()
            self._target_receipt._assert_raw_and_snapshot_unchanged()
            self._producer_receipt._assert_unchanged()
            self._physical_payload._assert_unchanged()
            for payload in self._target_payloads:
                payload._assert_unchanged()
        except Exception:
            self._runtime._seal_terminal()
            raise

    def issue_physical_view_lease(self, *, consumer: object) -> ExactObjectLeaseV2:
        self._assert_live()
        if consumer is None:
            self._fail_terminal(
                QualifiedSharedV5NavigationRuntimeV2BindingError(
                    "physical-view consumer must be exact non-null object"
                )
            )
        if "physical_view" in self._registry:
            self._fail_terminal(
                QualifiedSharedV5NavigationRuntimeV2ReplayError(
                    "physical-view lease was already issued"
                )
            )
        lease = ExactObjectLeaseV2(
            _LEASE_CONSTRUCTION_TOKEN,
            receipt=self,
            kind="physical_view",
            consumer=consumer,
            expiry_epoch=self._expiry_epoch,
        )
        self._registry["physical_view"] = _LeaseRegistrationV2(
            lease=lease,
            kind="physical_view",
            receipt=self,
            runtime=self._runtime,
            outcome=self._outcome,
            producer_receipt=self._producer_receipt,
            consumer=consumer,
            payload=self._physical_payload,
            reset_id=self.reset_id,
            session_id=self.session_id,
            tick_index=self.tick_index,
            expiry_epoch=self._expiry_epoch,
        )
        return lease

    def issue_target_evidence_leases_atomic(
        self,
        *,
        consumers: tuple[object, object, object, object],
    ) -> tuple[ExactObjectLeaseV2, ExactObjectLeaseV2, ExactObjectLeaseV2, ExactObjectLeaseV2]:
        self._assert_live()
        kinds = ("target_red", "target_yellow", "target_blue", "target_green")
        if type(consumers) is not tuple or len(consumers) != 4 or any(item is None for item in consumers):
            self._fail_terminal(
                QualifiedSharedV5NavigationRuntimeV2BindingError(
                    "target consumers must be exact non-null four-tuple"
                )
            )
        if any(kind in self._registry for kind in kinds):
            self._fail_terminal(
                QualifiedSharedV5NavigationRuntimeV2ReplayError(
                    "one or more target leases were already issued"
                )
            )
        leases = tuple(
            ExactObjectLeaseV2(
                _LEASE_CONSTRUCTION_TOKEN,
                receipt=self,
                kind=kind,
                consumer=consumers[index],
                expiry_epoch=self._expiry_epoch,
            )
            for index, kind in enumerate(kinds)
        )
        registrations = tuple(
            _LeaseRegistrationV2(
                lease=leases[index],
                kind=kind,
                receipt=self,
                runtime=self._runtime,
                outcome=self._outcome,
                producer_receipt=self._producer_receipt,
                consumer=consumers[index],
                payload=self._target_payloads[index],
                reset_id=self.reset_id,
                session_id=self.session_id,
                tick_index=self.tick_index,
                expiry_epoch=self._expiry_epoch,
            )
            for index, kind in enumerate(kinds)
        )
        self._registry.update(
            {kind: registration for kind, registration in zip(kinds, registrations)}
        )
        return leases  # type: ignore[return-value]

    def issue_g4_cached_feature_lease(self, *, consumer: object) -> ExactObjectLeaseV2:
        self._assert_live()
        if consumer is None:
            self._fail_terminal(
                QualifiedSharedV5NavigationRuntimeV2BindingError(
                    "G4 consumer must be exact non-null object"
                )
            )
        if "g4_cached_features" in self._registry:
            self._fail_terminal(
                QualifiedSharedV5NavigationRuntimeV2ReplayError(
                    "G4 cache lease was already issued"
                )
            )
        lease = ExactObjectLeaseV2(
            _LEASE_CONSTRUCTION_TOKEN,
            receipt=self,
            kind="g4_cached_features",
            consumer=consumer,
            expiry_epoch=self._expiry_epoch,
        )
        self._registry["g4_cached_features"] = _LeaseRegistrationV2(
            lease=lease,
            kind="g4_cached_features",
            receipt=self,
            runtime=self._runtime,
            outcome=self._outcome,
            producer_receipt=self._producer_receipt,
            consumer=consumer,
            payload=self._outcome._feature_cache,
            reset_id=self.reset_id,
            session_id=self.session_id,
            tick_index=self.tick_index,
            expiry_epoch=self._expiry_epoch,
        )
        return lease

    def __consume_registered(
        self,
        *,
        kind: str,
        lease: ExactObjectLeaseV2,
        consumer: object,
        expected_payload_type: type,
    ) -> object:
        self._assert_live()
        entry = self._registry.get(kind)
        if (
            entry is None
            or type(lease) is not ExactObjectLeaseV2
            or lease is not entry.lease
            or entry.kind != kind
            or entry.receipt is not self
            or entry.runtime is not self._runtime
            or entry.outcome is not self._outcome
            or entry.producer_receipt is not self._producer_receipt
            or entry.consumer is not consumer
            or entry.reset_id != self.reset_id
            or entry.session_id != self.session_id
            or entry.tick_index != self.tick_index
            or entry.expiry_epoch != self._expiry_epoch
            or type(entry.payload) is not expected_payload_type
        ):
            self._fail_terminal(
                QualifiedSharedV5NavigationRuntimeV2BindingError(
                    "lease is unregistered, forged, wrong-kind, foreign, or reconstructed"
                )
            )
        if (
            lease._receipt_marker is not self
            or lease._kind_marker != kind
            or lease._consumer_marker is not consumer
            or lease._epoch_marker != self._expiry_epoch
        ):
            self._fail_terminal(
                QualifiedSharedV5NavigationRuntimeV2BindingError(
                    "registered lease marker was grafted or changed"
                )
            )
        if entry.consumed:
            self._fail_terminal(
                QualifiedSharedV5NavigationRuntimeV2ReplayError("lease was replayed")
            )
        try:
            if type(entry.payload) is FrozenTargetColorPayloadV2:
                entry.payload._assert_unchanged()
            elif type(entry.payload) is PhysicalViewPayloadV2:
                entry.payload._assert_unchanged()
            elif type(entry.payload) is _FrozenSharedFeatureCacheV2:
                entry.payload._assert_unchanged()
            else:
                raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                    "registered payload type is not closed"
                )
            entry.consumed = True
            payload = entry.payload
            if type(payload) is FrozenTargetColorPayloadV2:
                payload._assert_unchanged()
            elif type(payload) is PhysicalViewPayloadV2:
                payload._assert_unchanged()
            else:
                payload._assert_unchanged()
            return payload
        except Exception:
            self._runtime._seal_terminal()
            raise

    def consume_physical_view_lease(
        self, *, lease: ExactObjectLeaseV2, consumer: object
    ) -> PhysicalViewPayloadV2:
        return self.__consume_registered(
            kind="physical_view",
            lease=lease,
            consumer=consumer,
            expected_payload_type=PhysicalViewPayloadV2,
        )  # type: ignore[return-value]

    def consume_target_red_lease(
        self, *, lease: ExactObjectLeaseV2, consumer: object
    ) -> FrozenTargetColorPayloadV2:
        return self.__consume_registered(
            kind="target_red", lease=lease, consumer=consumer,
            expected_payload_type=FrozenTargetColorPayloadV2,
        )  # type: ignore[return-value]

    def consume_target_yellow_lease(
        self, *, lease: ExactObjectLeaseV2, consumer: object
    ) -> FrozenTargetColorPayloadV2:
        return self.__consume_registered(
            kind="target_yellow", lease=lease, consumer=consumer,
            expected_payload_type=FrozenTargetColorPayloadV2,
        )  # type: ignore[return-value]

    def consume_target_blue_lease(
        self, *, lease: ExactObjectLeaseV2, consumer: object
    ) -> FrozenTargetColorPayloadV2:
        return self.__consume_registered(
            kind="target_blue", lease=lease, consumer=consumer,
            expected_payload_type=FrozenTargetColorPayloadV2,
        )  # type: ignore[return-value]

    def consume_target_green_lease(
        self, *, lease: ExactObjectLeaseV2, consumer: object
    ) -> FrozenTargetColorPayloadV2:
        return self.__consume_registered(
            kind="target_green", lease=lease, consumer=consumer,
            expected_payload_type=FrozenTargetColorPayloadV2,
        )  # type: ignore[return-value]

    def _consume_g4_cached_feature_lease(
        self, *, lease: ExactObjectLeaseV2, consumer: object
    ) -> _FrozenSharedFeatureCacheV2:
        return self.__consume_registered(
            kind="g4_cached_features", lease=lease, consumer=consumer,
            expected_payload_type=_FrozenSharedFeatureCacheV2,
        )  # type: ignore[return-value]

    def _expire(self) -> None:
        self._live = False
        self._target_receipt._live = False

    def __copy__(self) -> object:
        raise _noncopyable_reduce_error_v2("tick admission receipt")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise _noncopyable_reduce_error_v2("tick admission receipt")

    def __reduce__(self) -> object:
        raise _noncopyable_reduce_error_v2("tick admission receipt")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise _noncopyable_reduce_error_v2("tick admission receipt")


class SyntheticCandidateRowV2:
    """One caller-described row with no caller-selectable authority identity."""

    __slots__ = (
        "_selected_path_sha256",
        "_terminal_yaw_binary64_hex",
        "_baseline_value_binary64_hex",
        "_initial_binding",
        "content_sha256",
    )

    def __init__(
        self,
        *,
        selected_path_sha256: str,
        terminal_yaw_binary64_hex: str,
        baseline_value_binary64_hex: str,
    ) -> None:
        self._selected_path_sha256 = require_sha256_v2(
            selected_path_sha256, name="selected_path_sha256"
        )
        yaw = decode_canonical_binary64_hex_v2(
            terminal_yaw_binary64_hex, name="terminal_yaw_binary64_hex"
        )
        baseline = decode_canonical_binary64_hex_v2(
            baseline_value_binary64_hex, name="baseline_value_binary64_hex"
        )
        if canonical_binary64_hex_v2(yaw) != terminal_yaw_binary64_hex:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "terminal yaw encoding is not canonical"
            )
        if canonical_binary64_hex_v2(baseline) != baseline_value_binary64_hex:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "baseline value encoding is not canonical"
            )
        self._terminal_yaw_binary64_hex = terminal_yaw_binary64_hex
        self._baseline_value_binary64_hex = baseline_value_binary64_hex
        self.content_sha256 = self._derived_content_sha256()
        self._initial_binding = (
            self._selected_path_sha256,
            self._terminal_yaw_binary64_hex,
            self._baseline_value_binary64_hex,
            self.content_sha256,
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("SyntheticCandidateRowV2 cannot be subclassed")

    @property
    def selected_path_sha256(self) -> str:
        return self._selected_path_sha256

    @property
    def terminal_yaw_binary64_hex(self) -> str:
        return self._terminal_yaw_binary64_hex

    @property
    def baseline_value_binary64_hex(self) -> str:
        return self._baseline_value_binary64_hex

    def _derived_content_sha256(self) -> str:
        return canonical_json_sha256_v2(
            {
                "schema": "lewm_go2_synthetic_candidate_row_v2",
                "version": 2,
                "selected_path_sha256": self._selected_path_sha256,
                "terminal_yaw_binary64_hex": self._terminal_yaw_binary64_hex,
                "baseline_value_binary64_hex": self._baseline_value_binary64_hex,
            }
        )

    def _assert_unchanged(self) -> None:
        current = (
            self._selected_path_sha256,
            self._terminal_yaw_binary64_hex,
            self._baseline_value_binary64_hex,
            self.content_sha256,
        )
        if current != self._initial_binding or self._derived_content_sha256() != self.content_sha256:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "candidate row content changed"
            )


class CandidateSetAdmissionV2:
    """Exact producer-minted candidate set shared by baseline and learned G4."""

    __slots__ = (
        "_producer",
        "_runtime",
        "_tick_receipt",
        "_physical_payload",
        "_candidate_identity",
        "_rows",
        "_features",
        "_feature_observation",
        "_frozen_batch",
        "_initial_binding",
        "candidate_set_sha256",
        "candidate_row_sha256s",
        "baseline_scores_sha256",
        "baseline_selected_row_index",
        "content_sha256",
    )

    def __init__(
        self,
        token: object,
        *,
        producer: SyntheticCandidateProducerV2,
        runtime: "QualifiedSharedV5NavigationRuntimeV2",
        tick_receipt: TickAdmissionReceiptV2,
        physical_payload: PhysicalViewPayloadV2,
        candidate_identity: str,
        rows: tuple[SyntheticCandidateRowV2, ...],
        features: torch.Tensor,
    ) -> None:
        if token is not _CANDIDATE_CONSTRUCTION_TOKEN:
            raise PermissionError("candidate admission is producer-created only")
        if type(rows) is not tuple or not rows or any(
            type(row) is not SyntheticCandidateRowV2 for row in rows
        ):
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "candidate rows must be a nonempty exact tuple"
            )
        if type(features) is not torch.Tensor or features.ndim != 3:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "candidate features must be exact rank-three tensor"
            )
        if int(features.shape[0]) != 1 or int(features.shape[1]) != len(rows):
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "candidate feature batch/row shape changed"
            )
        if features.dtype not in {torch.float32, torch.float64}:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "candidate features must use float32 or float64"
            )
        _tensor_bytes_sha256_v2(features, name="candidate_features")
        self._producer = producer
        self._runtime = runtime
        self._tick_receipt = tick_receipt
        self._physical_payload = physical_payload
        self._candidate_identity = require_identifier_v2(
            candidate_identity, name="candidate_identity"
        )
        self._rows = tuple(
            SyntheticCandidateRowV2(
                selected_path_sha256=row.selected_path_sha256,
                terminal_yaw_binary64_hex=row.terminal_yaw_binary64_hex,
                baseline_value_binary64_hex=row.baseline_value_binary64_hex,
            )
            for row in rows
        )
        self._features = features.detach().clone().contiguous()
        self._feature_observation = _TensorObservationV2.capture(
            self._features, name="candidate_features"
        )
        self.candidate_row_sha256s = tuple(row.content_sha256 for row in self._rows)
        if len(set(self.candidate_row_sha256s)) != len(self.candidate_row_sha256s):
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "candidate rows must be content-unique"
            )
        self.candidate_set_sha256 = canonical_json_sha256_v2(
            {
                "schema": "lewm_go2_synthetic_candidate_set_identity_v2",
                "version": 2,
                "candidate_identity": self._candidate_identity,
                "candidate_producer_id": producer.producer_id,
                "tick_admission_sha256": tick_receipt.content_sha256,
                "physical_view_payload_sha256": physical_payload.content_sha256,
                "post_physical_revision": physical_payload.post_physical_state.revision,
                "post_configuration_revision": physical_payload.post_configuration_state.revision,
                "pre_view_revision": runtime._owner_states.row("view").revision,
                "pre_view_content_sha256": runtime._owner_states.row("view").content_sha256,
                "candidate_row_sha256s": list(self.candidate_row_sha256s),
                "feature_tensor_sha256": self._feature_observation.content_sha256,
            }
        )
        self._frozen_batch = FrozenCandidateFeatureBatchV1(
            candidate_set_sha256=self.candidate_set_sha256,
            candidate_row_sha256s=self.candidate_row_sha256s,
            features=self._features,
        )
        self.baseline_scores_sha256 = canonical_json_sha256_v2(
            {
                "schema": "lewm_go2_candidate_baseline_scores_v2",
                "version": 2,
                "candidate_set_sha256": self.candidate_set_sha256,
                "candidate_row_sha256s": list(self.candidate_row_sha256s),
                "scores_binary64_hex": [
                    row.baseline_value_binary64_hex for row in self._rows
                ],
            }
        )
        baseline_values = [
            decode_canonical_binary64_hex_v2(
                row.baseline_value_binary64_hex,
                name="candidate.baseline_value_binary64_hex",
            )
            for row in self._rows
        ]
        self.baseline_selected_row_index = max(
            range(len(baseline_values)), key=baseline_values.__getitem__
        )
        self.content_sha256 = canonical_json_sha256_v2(
            {
                "schema": "lewm_go2_candidate_set_admission_v2",
                "version": 2,
                "candidate_set_sha256": self.candidate_set_sha256,
                "candidate_identity": self._candidate_identity,
                "candidate_producer_id": producer.producer_id,
                "tick_admission_sha256": tick_receipt.content_sha256,
                "physical_view_payload_sha256": physical_payload.content_sha256,
                "candidate_row_sha256s": list(self.candidate_row_sha256s),
                "feature_batch_sha256": self._frozen_batch.content_sha256,
                "baseline_scores_sha256": self.baseline_scores_sha256,
                "baseline_selected_row_index": self.baseline_selected_row_index,
            }
        )
        self._initial_binding = (
            producer,
            runtime,
            tick_receipt,
            physical_payload,
            self._candidate_identity,
            self._rows,
            self._features,
            self.candidate_set_sha256,
            self.candidate_row_sha256s,
            self.baseline_scores_sha256,
            self.baseline_selected_row_index,
            self.content_sha256,
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("CandidateSetAdmissionV2 cannot be subclassed")

    @property
    def selected_baseline_path_sha256(self) -> str:
        return self._rows[self.baseline_selected_row_index].selected_path_sha256

    @property
    def selected_baseline_terminal_yaw_binary64_hex(self) -> str:
        return self._rows[
            self.baseline_selected_row_index
        ].terminal_yaw_binary64_hex

    def diagnostic_dict(self) -> dict[str, object]:
        return {
            "schema": "lewm_go2_candidate_set_admission_diagnostic_v2",
            "version": 2,
            "content_sha256": self.content_sha256,
            "candidate_set_sha256": self.candidate_set_sha256,
            "candidate_row_sha256s": list(self.candidate_row_sha256s),
            "baseline_scores_sha256": self.baseline_scores_sha256,
            "baseline_selected_row_index": self.baseline_selected_row_index,
        }

    def _assert_unchanged(self) -> None:
        current = (
            self._producer,
            self._runtime,
            self._tick_receipt,
            self._physical_payload,
            self._candidate_identity,
            self._rows,
            self._features,
            self.candidate_set_sha256,
            self.candidate_row_sha256s,
            self.baseline_scores_sha256,
            self.baseline_selected_row_index,
            self.content_sha256,
        )
        if current != self._initial_binding:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "candidate admission binding changed"
            )
        if (
            self._runtime._active_admission is not self._tick_receipt
            or self._tick_receipt._candidate_admission is not self
            or self._physical_payload is not self._tick_receipt._physical_payload
        ):
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "candidate admission is not exact active registered object"
            )
        for row in self._rows:
            row._assert_unchanged()
        if tuple(row.content_sha256 for row in self._rows) != self.candidate_row_sha256s:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "candidate row order/content changed"
            )
        self._physical_payload._assert_unchanged()
        self._feature_observation.assert_unchanged(
            self._features, name="candidate_features"
        )
        self._frozen_batch.assert_unchanged()
        if (
            self._frozen_batch.features is not self._features
            or self._frozen_batch.candidate_set_sha256 != self.candidate_set_sha256
            or self._frozen_batch.candidate_row_sha256s != self.candidate_row_sha256s
        ):
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "candidate head batch diverged from exact admitted set"
            )

    def __copy__(self) -> object:
        raise _noncopyable_reduce_error_v2("candidate admission")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise _noncopyable_reduce_error_v2("candidate admission")

    def __reduce__(self) -> object:
        raise _noncopyable_reduce_error_v2("candidate admission")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise _noncopyable_reduce_error_v2("candidate admission")


class CandidateBaselineReceiptV2:
    __slots__ = ("_runtime", "_candidate", "content_sha256")

    def __init__(
        self,
        token: object,
        *,
        runtime: "QualifiedSharedV5NavigationRuntimeV2",
        candidate: CandidateSetAdmissionV2,
    ) -> None:
        if token is not _CANDIDATE_CONSTRUCTION_TOKEN:
            raise PermissionError("baseline receipt is runtime-created only")
        self._runtime = runtime
        self._candidate = candidate
        self.content_sha256 = canonical_json_sha256_v2(
            {
                "schema": "lewm_go2_candidate_baseline_receipt_v2",
                "version": 2,
                "candidate_admission_sha256": candidate.content_sha256,
                "candidate_set_sha256": candidate.candidate_set_sha256,
                "candidate_exact_object_id": id(candidate),
                "baseline_scores_sha256": candidate.baseline_scores_sha256,
                "selected_row_index": candidate.baseline_selected_row_index,
                "selected_path_sha256": candidate.selected_baseline_path_sha256,
                "selected_terminal_yaw_binary64_hex": (
                    candidate.selected_baseline_terminal_yaw_binary64_hex
                ),
            }
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("CandidateBaselineReceiptV2 cannot be subclassed")

    def _assert_unchanged(self) -> None:
        self._candidate._assert_unchanged()
        if self._runtime._baseline_candidate is not self._candidate:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "baseline candidate object diverged"
            )

    def diagnostic_dict(self) -> dict[str, object]:
        self._assert_unchanged()
        return {
            "schema": "lewm_go2_candidate_baseline_receipt_diagnostic_v2",
            "version": 2,
            "content_sha256": self.content_sha256,
            "candidate_set_sha256": self._candidate.candidate_set_sha256,
            "baseline_scores_sha256": self._candidate.baseline_scores_sha256,
            "selected_row_index": self._candidate.baseline_selected_row_index,
            "selected_path_sha256": self._candidate.selected_baseline_path_sha256,
            "selected_terminal_yaw_binary64_hex": (
                self._candidate.selected_baseline_terminal_yaw_binary64_hex
            ),
        }

    def __copy__(self) -> object:
        raise _noncopyable_reduce_error_v2("baseline receipt")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise _noncopyable_reduce_error_v2("baseline receipt")

    def __reduce__(self) -> object:
        raise _noncopyable_reduce_error_v2("baseline receipt")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise _noncopyable_reduce_error_v2("baseline receipt")


class G4ScoringReceiptV2:
    """Opaque learned-score receipt over the exact registered candidate object."""

    __slots__ = (
        "_runtime",
        "_tick_receipt",
        "_candidate",
        "_baseline_receipt",
        "_head",
        "_head_config_sha256",
        "_raw_result",
        "_raw_scores_observation",
        "_scores_snapshot",
        "_scores_snapshot_observation",
        "learned_scores_sha256",
        "selected_row_index",
        "selected_path_sha256",
        "selected_terminal_yaw_binary64_hex",
        "content_sha256",
    )

    def __init__(
        self,
        token: object,
        *,
        runtime: "QualifiedSharedV5NavigationRuntimeV2",
        tick_receipt: TickAdmissionReceiptV2,
        candidate: CandidateSetAdmissionV2,
        baseline_receipt: CandidateBaselineReceiptV2,
        head: TwoResolutionFrontierValueHeadV1,
        raw_result: FrontierValueScoresV1,
    ) -> None:
        if token is not _G4_RECEIPT_CONSTRUCTION_TOKEN:
            raise PermissionError("G4 receipt is runtime-created only")
        if (
            type(raw_result) is not FrontierValueScoresV1
            or raw_result.candidate_batch is not candidate._frozen_batch
            or raw_result.candidate_set_sha256 != candidate.candidate_set_sha256
            or raw_result.candidate_row_sha256s != candidate.candidate_row_sha256s
        ):
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "G4 result diverged from exact candidate object/order"
            )
        if tuple(raw_result.scores.shape) != (1, len(candidate._rows)):
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "G4 score shape changed"
            )
        self._runtime = runtime
        self._tick_receipt = tick_receipt
        self._candidate = candidate
        self._baseline_receipt = baseline_receipt
        self._head = head
        self._head_config_sha256 = head.architecture_config_sha256
        self._raw_result = raw_result
        self._raw_scores_observation = _TensorObservationV2.capture(
            raw_result.scores, name="raw_g4_scores"
        )
        self._scores_snapshot = raw_result.scores.detach().clone().contiguous()
        if int(self._scores_snapshot.untyped_storage().data_ptr()) == int(
            raw_result.scores.untyped_storage().data_ptr()
        ):
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "G4 score snapshot aliases raw result"
            )
        self._scores_snapshot_observation = _TensorObservationV2.capture(
            self._scores_snapshot, name="frozen_g4_scores"
        )
        self.learned_scores_sha256 = canonical_json_sha256_v2(
            {
                "schema": "lewm_go2_learned_g4_scores_v2",
                "version": 2,
                "candidate_set_sha256": candidate.candidate_set_sha256,
                "candidate_row_sha256s": list(candidate.candidate_row_sha256s),
                "scores_tensor_sha256": self._scores_snapshot_observation.content_sha256,
            }
        )
        self.selected_row_index = int(torch.argmax(self._scores_snapshot, dim=1).item())
        selected = candidate._rows[self.selected_row_index]
        self.selected_path_sha256 = selected.selected_path_sha256
        self.selected_terminal_yaw_binary64_hex = selected.terminal_yaw_binary64_hex
        self.content_sha256 = canonical_json_sha256_v2(
            {
                "schema": "lewm_go2_g4_scoring_receipt_v2",
                "version": 2,
                "tick_admission_sha256": tick_receipt.content_sha256,
                "candidate_admission_sha256": candidate.content_sha256,
                "candidate_exact_object_id": id(candidate),
                "baseline_receipt_sha256": baseline_receipt.content_sha256,
                "head_exact_object_id": id(head),
                "head_config_sha256": self._head_config_sha256,
                "learned_scores_sha256": self.learned_scores_sha256,
                "selected_row_index": self.selected_row_index,
                "selected_path_sha256": self.selected_path_sha256,
                "selected_terminal_yaw_binary64_hex": self.selected_terminal_yaw_binary64_hex,
            }
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("G4ScoringReceiptV2 cannot be subclassed")

    def _assert_unchanged(self) -> None:
        self._candidate._assert_unchanged()
        self._baseline_receipt._assert_unchanged()
        if (
            self._runtime._active_admission is not self._tick_receipt
            or self._runtime._g4_receipt is not self
            or self._runtime._baseline_candidate is not self._candidate
            or type(self._head) is not TwoResolutionFrontierValueHeadV1
            or self._head.architecture_config_sha256 != self._head_config_sha256
            or type(self._raw_result) is not FrontierValueScoresV1
            or self._raw_result.candidate_batch is not self._candidate._frozen_batch
        ):
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "G4 exact object/config binding changed"
            )
        self._raw_scores_observation.assert_unchanged(
            self._raw_result.scores, name="raw_g4_scores"
        )
        self._scores_snapshot_observation.assert_unchanged(
            self._scores_snapshot, name="frozen_g4_scores"
        )

    def diagnostic_dict(self) -> dict[str, object]:
        self._assert_unchanged()
        return {
            "schema": "lewm_go2_g4_scoring_receipt_diagnostic_v2",
            "version": 2,
            "content_sha256": self.content_sha256,
            "candidate_set_sha256": self._candidate.candidate_set_sha256,
            "learned_scores_sha256": self.learned_scores_sha256,
            "selected_row_index": self.selected_row_index,
            "selected_path_sha256": self.selected_path_sha256,
            "selected_terminal_yaw_binary64_hex": self.selected_terminal_yaw_binary64_hex,
        }

    def __copy__(self) -> object:
        raise _noncopyable_reduce_error_v2("G4 scoring receipt")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise _noncopyable_reduce_error_v2("G4 scoring receipt")

    def __reduce__(self) -> object:
        raise _noncopyable_reduce_error_v2("G4 scoring receipt")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise _noncopyable_reduce_error_v2("G4 scoring receipt")


def _single_counter_delta_v2(**values: int) -> CallCounterPanelV2:
    unknown = set(values) - set(CallCounterPanelV2.names())
    if unknown:
        raise ValueError("unknown counter name")
    return CallCounterPanelV2(
        *(values.get(name, 0) for name in CallCounterPanelV2.names())
    )


class QualifiedSharedV5NavigationRuntimeV2:
    """Issuer-only synthetic state machine with exact registered authority."""

    __slots__ = (
        "_issuer",
        "_reset_authority",
        "_reset_receipt",
        "_backend",
        "_physical_producer",
        "_candidate_producer",
        "_owner_states",
        "_next_tick_index",
        "_expiry_epoch",
        "_counts",
        "_tick_counts",
        "_active_outcome",
        "_active_target_receipt",
        "_active_physical_receipt",
        "_active_admission",
        "_active_candidate",
        "_baseline_candidate",
        "_baseline_receipt",
        "_g4_receipt",
        "_retired_authority_objects",
        "_terminal_fault",
    )

    shared_v5_checkpoint_file_sha256 = None
    shared_v5_model_state_sha256 = None
    g2_report_sha256 = None
    g2_candidate_publication_sha256 = None
    physical_calibration_sha256 = None
    physical_thresholds_sha256 = None
    target_head_binding_sha256 = None
    g4_head_binding_sha256 = None

    def __init__(
        self,
        token: object,
        *,
        issuer: FoundationAuthorityIssuerV2,
        reset_authority: ResetAuthorityV2,
        backend: FakeSharedV5FrameBackendV2,
    ) -> None:
        if token is not _ISSUER_CONSTRUCTION_TOKEN:
            raise PermissionError("Foundation V2 runtime is issuer-created only")
        if (
            type(issuer) is not FoundationAuthorityIssuerV2
            or type(reset_authority) is not ResetAuthorityV2
            or type(backend) is not FakeSharedV5FrameBackendV2
            or reset_authority._issuer is not issuer
            or backend._issuer is not issuer
            or not reset_authority._used
            or not backend._consumed
            or not any(reset_authority is item for item in issuer._spent_authorities)
            or not any(backend is item for item in issuer._spent_backends)
        ):
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "runtime factory authority/backend binding changed"
            )
        receipt = reset_authority._receipt
        if type(receipt) is not ResetReceiptV2:
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "reset receipt exact type changed"
            )
        # ResetReceiptV2 construction already verifies all thirteen zero rows,
        # fresh empty contents, order, and pairwise identity distinctness.  Recheck
        # the exact producer and issuer binding at the live factory boundary.
        if (
            receipt.issuer_id != issuer.issuer_id
            or receipt.physical_projection_producer_id
            != reset_authority._physical_producer.producer_id
            or receipt.candidate_producer_id
            != reset_authority._candidate_producer.producer_id
            or any(row.revision != 0 for row in receipt.initial_owner_states.rows)
            or any(
                identity not in issuer._all_identities
                for identity in (
                    receipt.reset_id,
                    receipt.session_id,
                    receipt.reset_capability_id,
                    receipt.physical_projection_producer_id,
                    receipt.candidate_producer_id,
                    *(row.owner_id for row in receipt.initial_owner_states.rows),
                )
            )
        ):
            raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                "tick-zero reset/producer/owner binding changed"
            )
        self._issuer = issuer
        self._reset_authority = reset_authority
        self._reset_receipt = receipt
        self._backend = backend
        self._physical_producer = reset_authority._physical_producer
        self._candidate_producer = reset_authority._candidate_producer
        self._owner_states = receipt.initial_owner_states
        self._next_tick_index = 0
        self._expiry_epoch = 0
        self._counts = CallCounterPanelV2.zero()
        self._tick_counts = CallCounterPanelV2.zero()
        self._active_outcome: QualifiedSharedV5FrameOutcomeV2 | None = None
        self._active_target_receipt: TargetBatchReceiptV2 | None = None
        self._active_physical_receipt: PhysicalProjectionProducerReceiptV2 | None = None
        self._active_admission: TickAdmissionReceiptV2 | None = None
        self._active_candidate: CandidateSetAdmissionV2 | None = None
        self._baseline_candidate: CandidateSetAdmissionV2 | None = None
        self._baseline_receipt: CandidateBaselineReceiptV2 | None = None
        self._g4_receipt: G4ScoringReceiptV2 | None = None
        self._retired_authority_objects: list[object] = []
        self._terminal_fault = False

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise TypeError("QualifiedSharedV5NavigationRuntimeV2 cannot be subclassed")

    @property
    def reset_receipt(self) -> ResetReceiptV2:
        return self._reset_receipt

    @property
    def call_counters(self) -> CallCounterPanelV2:
        return self._counts

    @property
    def owner_states(self) -> OwnerStateBundleV2:
        return self._owner_states

    @property
    def terminal(self) -> bool:
        return self._terminal_fault

    def _require_operational(self) -> None:
        if self._terminal_fault:
            raise QualifiedSharedV5NavigationRuntimeV2TerminalError(
                "runtime is sealed after terminal fault"
            )

    def _add_counts(self, delta: CallCounterPanelV2) -> None:
        self._counts = self._counts.plus(delta)
        self._tick_counts = self._tick_counts.plus(delta)

    def _retire_active_objects(self) -> None:
        for item in (
            self._active_outcome,
            self._active_target_receipt,
            self._active_physical_receipt,
            self._active_admission,
            self._active_candidate,
            self._baseline_receipt,
            self._g4_receipt,
        ):
            if item is not None:
                self._retired_authority_objects.append(item)

    def _clear_active(self) -> None:
        self._active_outcome = None
        self._active_target_receipt = None
        self._active_physical_receipt = None
        self._active_admission = None
        self._active_candidate = None
        self._baseline_candidate = None
        self._baseline_receipt = None
        self._g4_receipt = None

    def _seal_terminal(self) -> None:
        if self._terminal_fault:
            return
        if self._active_admission is not None:
            self._active_admission._expire()
        elif self._active_target_receipt is not None:
            self._active_target_receipt._live = False
        # A started tick records terminal rollback by advancing only tick_chain.
        if self._active_outcome is not None:
            fault_content = canonical_json_sha256_v2(
                {
                    "schema": "lewm_go2_synthetic_foundation_terminal_rollback_v2",
                    "version": 2,
                    "tick_index": self._active_outcome.tick_index,
                    "reset_id": self._reset_receipt.reset_id,
                    "session_id": self._reset_receipt.session_id,
                    "pre_owner_bundle_sha256": self._owner_states.content_sha256,
                    "frame_outcome_sha256": self._active_outcome.content_sha256,
                    "call_counters": self._tick_counts.to_dict(),
                }
            )
            self._owner_states = advance_owner_bundle_v2(
                self._owner_states,
                advanced_owner_content_sha256={"tick_chain": fault_content},
            )
        self._expiry_epoch += 1
        self._retire_active_objects()
        self._clear_active()
        self._terminal_fault = True

    def _raise_terminal(self, error: Exception) -> None:
        self._seal_terminal()
        raise error

    def run_shared_frame_once(
        self,
        *,
        synthetic_frame: torch.Tensor,
        controller_input_sha256: str,
        timestamp_binary64_hex: str,
        synchronization_id: str,
    ) -> QualifiedSharedV5FrameOutcomeV2:
        self._require_operational()
        if self._active_outcome is not None:
            self._raise_terminal(
                QualifiedSharedV5NavigationRuntimeV2ReplayError(
                    "a frame is already active; recomputation is forbidden"
                )
            )
        try:
            controller_sha = require_sha256_v2(
                controller_input_sha256, name="controller_input_sha256"
            )
            synchronization = require_identifier_v2(
                synchronization_id, name="synchronization_id"
            )
            timestamp = decode_canonical_binary64_hex_v2(
                timestamp_binary64_hex, name="timestamp_binary64_hex"
            )
            if canonical_binary64_hex_v2(timestamp) != timestamp_binary64_hex:
                raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                    "timestamp binary64 is not canonical"
                )
            rgb_sha = synthetic_frame_content_sha256_v2(synthetic_frame)
            before = self._backend._count_snapshot()
            frame = self._backend._preprocess_once(synthetic_frame)
            patch, bev, physical = self._backend._forward_frame_once(frame)
            after = self._backend._count_snapshot()
            expected_delta = {
                "shared_v5_forward_frame_call_count": 1,
                "vision_encoder_forward_tokens_call_count": 1,
                "rgb_decode_call_count": 1,
                "rgb_preprocess_call_count": 1,
            }
            if any(
                after[name] - before[name] != expected
                for name, expected in expected_delta.items()
            ):
                raise QualifiedSharedV5NavigationRuntimeV2TerminalError(
                    "fake backend violated exact one-frame call graph"
                )
            cache = _FrozenSharedFeatureCacheV2(
                _FRAME_CONSTRUCTION_TOKEN, patch, bev
            )
            outcome = QualifiedSharedV5FrameOutcomeV2(
                _FRAME_CONSTRUCTION_TOKEN,
                runtime=self,
                tick_index=self._next_tick_index,
                reset_id=self._reset_receipt.reset_id,
                session_id=self._reset_receipt.session_id,
                pre_owner_states=self._owner_states,
                controller_input_sha256=controller_sha,
                rgb_content_sha256=rgb_sha,
                timestamp_binary64_hex=timestamp_binary64_hex,
                synchronization_id=synchronization,
                feature_cache=cache,
                physical_output=physical,
            )
        except Exception:
            self._seal_terminal()
            raise
        self._tick_counts = CallCounterPanelV2.zero()
        self._add_counts(
            _single_counter_delta_v2(
                observation_tick_count=1,
                shared_frame_outcome_count=1,
                shared_v5_forward_frame_call_count=1,
                vision_encoder_forward_tokens_call_count=1,
                rgb_decode_call_count=1,
                rgb_preprocess_call_count=1,
            )
        )
        self._active_outcome = outcome
        return outcome

    def run_target_four_color_batch_once(
        self,
        *,
        outcome: QualifiedSharedV5FrameOutcomeV2,
        head: SharedV5TargetObservationHeadV1,
    ) -> TargetBatchReceiptV2:
        self._require_operational()
        if outcome is not self._active_outcome or type(outcome) is not QualifiedSharedV5FrameOutcomeV2:
            self._raise_terminal(
                QualifiedSharedV5NavigationRuntimeV2BindingError(
                    "target outcome is reconstructed or foreign"
                )
            )
        if self._active_target_receipt is not None:
            self._raise_terminal(
                QualifiedSharedV5NavigationRuntimeV2ReplayError(
                    "target four-color batch was already called"
                )
            )
        if type(head) is not SharedV5TargetObservationHeadV1:
            self._raise_terminal(TypeError("head must be exact SharedV5TargetObservationHeadV1"))
        try:
            outcome._assert_unchanged()
            with torch.no_grad():
                raw = head(
                    outcome._feature_cache._patch_features,
                    outcome._feature_cache._bev_features,
                )
            self._add_counts(
                _single_counter_delta_v2(target_four_color_batch_count=1)
            )
            batch_size, device = _validate_four_color_output_domain_v2(
                raw, maximum_range_m=float(head.config.maximum_range_m)
            )
            raw_observations = {
                name: _TensorObservationV2.capture(
                    getattr(raw, name), name=f"raw_{name}"
                )
                for name in _TARGET_TENSOR_NAMES
            }
            counter_receipt_sha256 = canonical_json_sha256_v2(
                {
                    "schema": "lewm_go2_target_call_counter_receipt_v2",
                    "version": 2,
                    "tick_index": outcome.tick_index,
                    "frame_outcome_sha256": outcome.content_sha256,
                    "tick_call_counters": self._tick_counts.to_dict(),
                }
            )
            snapshot = FrozenFourColorTargetBatchV2(
                _TARGET_CONSTRUCTION_TOKEN,
                frame=outcome,
                head=head,
                raw=raw,
                batch_size=batch_size,
                device=device,
                counter_receipt_sha256=counter_receipt_sha256,
            )
            receipt = TargetBatchReceiptV2(
                _TARGET_CONSTRUCTION_TOKEN,
                runtime=self,
                frame=outcome,
                head=head,
                raw_output=raw,
                raw_observations=raw_observations,
                snapshot=snapshot,
                counter_receipt_sha256=counter_receipt_sha256,
            )
            receipt._assert_raw_and_snapshot_unchanged()
        except Exception:
            self._seal_terminal()
            raise
        self._active_target_receipt = receipt
        return receipt

    def mint_physical_projection_receipt_once(
        self,
        *,
        outcome: QualifiedSharedV5FrameOutcomeV2,
        target_batch_receipt: TargetBatchReceiptV2,
    ) -> PhysicalProjectionProducerReceiptV2:
        self._require_operational()
        if (
            outcome is not self._active_outcome
            or target_batch_receipt is not self._active_target_receipt
            or type(target_batch_receipt) is not TargetBatchReceiptV2
        ):
            self._raise_terminal(
                QualifiedSharedV5NavigationRuntimeV2BindingError(
                    "physical producer inputs are not exact active objects"
                )
            )
        if self._active_physical_receipt is not None or target_batch_receipt._consumed_by_producer:
            self._raise_terminal(
                QualifiedSharedV5NavigationRuntimeV2ReplayError(
                    "physical producer receipt was already minted"
                )
            )
        try:
            result = self._physical_producer._mint(
                runtime=self, outcome=outcome, target_receipt=target_batch_receipt
            )
            result._assert_unchanged()
        except Exception:
            self._seal_terminal()
            raise
        target_batch_receipt._consumed_by_producer = True
        self._active_physical_receipt = result
        return result

    def admit_tick(
        self,
        *,
        outcome: QualifiedSharedV5FrameOutcomeV2,
        target_batch_receipt: TargetBatchReceiptV2,
        producer_receipt: PhysicalProjectionProducerReceiptV2,
    ) -> TickAdmissionReceiptV2:
        self._require_operational()
        if (
            outcome is not self._active_outcome
            or target_batch_receipt is not self._active_target_receipt
            or producer_receipt is not self._active_physical_receipt
            or type(producer_receipt) is not PhysicalProjectionProducerReceiptV2
            or producer_receipt._producer is not self._physical_producer
            or producer_receipt._runtime is not self
            or producer_receipt._outcome is not outcome
            or producer_receipt._target_receipt is not target_batch_receipt
        ):
            self._raise_terminal(
                QualifiedSharedV5NavigationRuntimeV2BindingError(
                    "admission requires exact registered producer/frame/target objects"
                )
            )
        if self._active_admission is not None or producer_receipt._consumed:
            self._raise_terminal(
                QualifiedSharedV5NavigationRuntimeV2ReplayError(
                    "physical producer receipt was replayed"
                )
            )
        try:
            producer_receipt._assert_unchanged()
            receipt = TickAdmissionReceiptV2(
                _ADMISSION_CONSTRUCTION_TOKEN,
                runtime=self,
                outcome=outcome,
                target_receipt=target_batch_receipt,
                producer_receipt=producer_receipt,
                expiry_epoch=self._expiry_epoch,
            )
        except Exception:
            self._seal_terminal()
            raise
        producer_receipt._consumed = True
        self._active_admission = receipt
        return receipt

    def mint_synthetic_candidate_set_once(
        self,
        *,
        receipt: TickAdmissionReceiptV2,
        rows: tuple[SyntheticCandidateRowV2, ...],
        features: torch.Tensor,
    ) -> CandidateSetAdmissionV2:
        self._require_operational()
        if receipt is not self._active_admission or type(receipt) is not TickAdmissionReceiptV2:
            self._raise_terminal(
                QualifiedSharedV5NavigationRuntimeV2BindingError(
                    "candidate receipt is not exact active admission"
                )
            )
        if self._active_candidate is not None or receipt._candidate_admission is not None:
            self._raise_terminal(
                QualifiedSharedV5NavigationRuntimeV2ReplayError(
                    "candidate set was already minted"
                )
            )
        try:
            lease = receipt.issue_physical_view_lease(
                consumer=self._candidate_producer
            )
            physical_payload = receipt.consume_physical_view_lease(
                lease=lease, consumer=self._candidate_producer
            )
            if physical_payload is not receipt._physical_payload:
                raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                    "candidate producer received non-exact physical payload"
                )
            admission = self._candidate_producer._mint(
                runtime=self,
                receipt=receipt,
                physical_payload=physical_payload,
                rows=rows,
                features=features,
            )
            receipt._candidate_admission = admission
            self._active_candidate = admission
            admission._assert_unchanged()
        except Exception:
            self._seal_terminal()
            raise
        return admission

    def run_deterministic_baseline_once(
        self,
        *,
        receipt: TickAdmissionReceiptV2,
        candidate_admission: CandidateSetAdmissionV2,
    ) -> CandidateBaselineReceiptV2:
        self._require_operational()
        if (
            receipt is not self._active_admission
            or candidate_admission is not self._active_candidate
            or receipt._candidate_admission is not candidate_admission
            or type(candidate_admission) is not CandidateSetAdmissionV2
        ):
            self._raise_terminal(
                QualifiedSharedV5NavigationRuntimeV2BindingError(
                    "baseline candidate is reconstructed, copied, reordered, or foreign"
                )
            )
        if self._baseline_candidate is not None:
            self._raise_terminal(
                QualifiedSharedV5NavigationRuntimeV2ReplayError(
                    "deterministic baseline was already evaluated"
                )
            )
        try:
            candidate_admission._assert_unchanged()
            baseline = CandidateBaselineReceiptV2(
                _CANDIDATE_CONSTRUCTION_TOKEN,
                runtime=self,
                candidate=candidate_admission,
            )
            self._baseline_candidate = candidate_admission
            self._baseline_receipt = baseline
            baseline._assert_unchanged()
        except Exception:
            self._seal_terminal()
            raise
        return baseline

    def run_g4_value_head_once(
        self,
        *,
        receipt: TickAdmissionReceiptV2,
        candidate_admission: CandidateSetAdmissionV2,
        baseline_receipt: CandidateBaselineReceiptV2,
        head: TwoResolutionFrontierValueHeadV1,
    ) -> G4ScoringReceiptV2:
        self._require_operational()
        if (
            receipt is not self._active_admission
            or candidate_admission is not self._active_candidate
            or candidate_admission is not self._baseline_candidate
            or baseline_receipt is not self._baseline_receipt
            or type(candidate_admission) is not CandidateSetAdmissionV2
            or type(baseline_receipt) is not CandidateBaselineReceiptV2
        ):
            self._raise_terminal(
                QualifiedSharedV5NavigationRuntimeV2BindingError(
                    "baseline/head candidate exact object diverged"
                )
            )
        if self._g4_receipt is not None:
            self._raise_terminal(
                QualifiedSharedV5NavigationRuntimeV2ReplayError(
                    "G4 head was already called"
                )
            )
        if type(head) is not TwoResolutionFrontierValueHeadV1:
            self._raise_terminal(TypeError("head must be exact TwoResolutionFrontierValueHeadV1"))
        try:
            candidate_admission._assert_unchanged()
            baseline_receipt._assert_unchanged()
            lease = receipt.issue_g4_cached_feature_lease(consumer=head)
            cache = receipt._consume_g4_cached_feature_lease(
                lease=lease, consumer=head
            )
            if cache is not receipt._outcome._feature_cache:
                raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                    "G4 cache object identity changed"
                )
            with torch.no_grad():
                result = head(
                    cache._patch_features,
                    cache._bev_features,
                    candidate_admission._frozen_batch,
                )
            self._add_counts(_single_counter_delta_v2(g4_value_head_call_count=1))
            scoring = G4ScoringReceiptV2(
                _G4_RECEIPT_CONSTRUCTION_TOKEN,
                runtime=self,
                tick_receipt=receipt,
                candidate=candidate_admission,
                baseline_receipt=baseline_receipt,
                head=head,
                raw_result=result,
            )
            self._g4_receipt = scoring
            scoring._assert_unchanged()
        except Exception:
            self._seal_terminal()
            raise
        return scoring

    def _is_exact_active_receipt(self, receipt: TickAdmissionReceiptV2) -> bool:
        return (
            type(receipt) is TickAdmissionReceiptV2
            and receipt is self._active_admission
            and receipt._outcome is self._active_outcome
        )

    def commit_tick(self, *, receipt: TickAdmissionReceiptV2) -> OwnerStateBundleV2:
        self._require_operational()
        if receipt is not self._active_admission or type(receipt) is not TickAdmissionReceiptV2:
            self._raise_terminal(
                QualifiedSharedV5NavigationRuntimeV2BindingError(
                    "commit receipt is not exact active object"
                )
            )
        if (
            self._active_outcome is None
            or self._active_target_receipt is None
            or self._active_physical_receipt is None
        ):
            self._raise_terminal(
                QualifiedSharedV5NavigationRuntimeV2BindingError(
                    "active tick is incomplete"
                )
            )
        if (self._active_candidate is None) != (self._g4_receipt is None):
            self._raise_terminal(
                QualifiedSharedV5NavigationRuntimeV2BindingError(
                    "candidate/G4 completion differs"
                )
            )
        try:
            receipt._assert_live()
            self._tick_counts.assert_complete_observation(
                g4_calls=1 if self._g4_receipt is not None else 0
            )
            if self._g4_receipt is not None:
                self._g4_receipt._assert_unchanged()
            physical = receipt._physical_payload
            advanced = {
                "physical": physical.post_physical_state.owner_content_sha256,
                "configuration": physical.post_configuration_state.owner_content_sha256,
                "view": canonical_json_sha256_v2(
                    {
                        "schema": "lewm_go2_synthetic_view_commit_v2",
                        "version": 2,
                        "tick_admission_sha256": receipt.content_sha256,
                        "physical_view_payload_sha256": physical.content_sha256,
                        "candidate_admission_sha256": (
                            None
                            if self._active_candidate is None
                            else self._active_candidate.content_sha256
                        ),
                    }
                ),
                "integration": canonical_json_sha256_v2(
                    {
                        "schema": "lewm_go2_synthetic_foundation_integration_commit_v2",
                        "version": 2,
                        "tick_admission_sha256": receipt.content_sha256,
                        "target_batch_receipt_sha256": receipt._target_receipt.content_sha256,
                        "candidate_admission_sha256": (
                            None
                            if self._active_candidate is None
                            else self._active_candidate.content_sha256
                        ),
                        "g4_scoring_receipt_sha256": (
                            None if self._g4_receipt is None else self._g4_receipt.content_sha256
                        ),
                    }
                ),
            }
            for color, payload in zip(CANONICAL_COLORS_V2, receipt._target_payloads):
                payload._assert_unchanged()
                advanced[f"target_{color}"] = canonical_json_sha256_v2(
                    {
                        "schema": "lewm_go2_synthetic_target_owner_commit_v2",
                        "version": 2,
                        "color": color,
                        "target_payload_sha256": payload.content_sha256,
                        "tick_admission_sha256": receipt.content_sha256,
                    }
                )
            advanced["tick_chain"] = canonical_json_sha256_v2(
                {
                    "schema": "lewm_go2_synthetic_foundation_tick_chain_commit_v2",
                    "version": 2,
                    "tick_index": receipt.tick_index,
                    "pre_owner_bundle_sha256": self._owner_states.content_sha256,
                    "tick_admission_sha256": receipt.content_sha256,
                    "tick_call_counters": self._tick_counts.to_dict(),
                    "advanced_owner_content_sha256": advanced,
                }
            )
            post = advance_owner_bundle_v2(
                self._owner_states, advanced_owner_content_sha256=advanced
            )
            if (
                post.row("physical").to_dict()
                != physical.post_physical_state.to_dict()
                or post.row("configuration").to_dict()
                != physical.post_configuration_state.to_dict()
            ):
                raise QualifiedSharedV5NavigationRuntimeV2BindingError(
                    "producer-derived physical/configuration transition diverged"
                )
        except Exception:
            self._seal_terminal()
            raise
        self._owner_states = post
        self._next_tick_index += 1
        self._expiry_epoch += 1
        receipt._expire()
        self._retire_active_objects()
        self._clear_active()
        self._tick_counts = CallCounterPanelV2.zero()
        return post

    def fault_tick(self, *, receipt: TickAdmissionReceiptV2 | None = None) -> None:
        self._require_operational()
        if receipt is not None and receipt is not self._active_admission:
            self._raise_terminal(
                QualifiedSharedV5NavigationRuntimeV2BindingError(
                    "fault receipt is not exact active object"
                )
            )
        self._seal_terminal()

    def __copy__(self) -> object:
        raise _noncopyable_reduce_error_v2("navigation runtime")

    def __deepcopy__(self, memo: object) -> object:
        del memo
        raise _noncopyable_reduce_error_v2("navigation runtime")

    def __reduce__(self) -> object:
        raise _noncopyable_reduce_error_v2("navigation runtime")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise _noncopyable_reduce_error_v2("navigation runtime")


__all__ = [
    "CandidateBaselineReceiptV2",
    "CandidateSetAdmissionV2",
    "ExactObjectLeaseV2",
    "FakeSharedV5FrameBackendV2",
    "FoundationAuthorityIssuerV2",
    "FrozenTargetColorPayloadV2",
    "G4ScoringReceiptV2",
    "PhysicalProjectionProducerReceiptV2",
    "PhysicalViewPayloadV2",
    "QualifiedSharedV5FrameOutcomeV2",
    "QualifiedSharedV5NavigationRuntimeV2",
    "QualifiedSharedV5NavigationRuntimeV2BindingError",
    "QualifiedSharedV5NavigationRuntimeV2Error",
    "QualifiedSharedV5NavigationRuntimeV2ReplayError",
    "QualifiedSharedV5NavigationRuntimeV2TerminalError",
    "ResetAuthorityV2",
    "SyntheticCandidateRowV2",
    "TargetBatchReceiptV2",
    "TickAdmissionReceiptV2",
    "synthetic_frame_content_sha256_v2",
]
